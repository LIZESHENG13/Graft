"""Train prefix-conditioned queries using actual training-period arrivals."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .catalog import Catalog
from .evaluation import GraftRetriever, evaluate_queries
from .io import atomic_json, parameter_hash, project_path, seed_everything, sha256
from .scorer import PrefixConditioner, load_scorer
from .training import PathObjective, TrainingConfig


@dataclass(frozen=True)
class RefinementConfig:
    steps: int = 1000
    batch_size: int = 512
    negatives: int = 64
    sibling_budget: int = 32
    routed_budget: int = 16
    learning_rate: float = 0.0002
    weight_decay: float = 0.05
    anchor_weight: float = 10.0
    episode_weight: float = 2.0
    episode_path_weight: float = 1.0
    episode_batch_size: int = 256
    validation_size: int = 8192
    checkpoints: tuple[int, ...] = (0, 100, 250, 500, 1000)
    warmup_steps: int = 500
    scheduler_steps: int = 10000
    negative_strategy: str = "competition"
    base_path_weight: float = 1.0
    teacher_weight: float = 0.0
    episode_new_only: bool = False
    reachable_only: bool = False

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.negatives, self.episode_batch_size, self.validation_size) < 1:
            raise ValueError("Training and sampling budgets must be positive")
        if min(self.sibling_budget, self.routed_budget, self.warmup_steps) < 0:
            raise ValueError("Mining and warmup budgets must be nonnegative")
        if self.scheduler_steps <= self.warmup_steps or self.learning_rate <= 0:
            raise ValueError("Invalid scheduler or learning rate")
        if self.negative_strategy not in ("competition", "uniform"):
            raise ValueError("Unknown negative strategy")


class IdentityQuery(torch.nn.Module):
    def forward(self, query, prefix_vectors, level=2):
        return F.normalize(query, dim=-1)


def unique_visible_negatives(candidates, targets, birth, times, budget):
    """Keep first occurrence by priority; masked padding is never a real leaf."""
    valid = (candidates >= 0) & (candidates != targets[:, None])
    valid &= birth[candidates.clamp(min=0)] <= times[:, None]
    previous = torch.tril(
        torch.ones(candidates.shape[1], candidates.shape[1], device=candidates.device, dtype=torch.bool), diagonal=-1
    )
    duplicate = ((candidates[:, :, None] == candidates[:, None, :]) & previous[None]).any(-1)
    valid &= ~duplicate
    position = torch.arange(candidates.shape[1], device=candidates.device)[None].expand_as(candidates)
    priority = position.masked_fill(~valid, candidates.shape[1])
    chosen = priority.topk(min(budget, candidates.shape[1]), largest=False, sorted=True).indices
    selected = candidates.gather(1, chosen).masked_fill(~valid.gather(1, chosen), -1)
    if selected.shape[1] < budget:
        selected = F.pad(selected, (0, budget - selected.shape[1]), value=-1)
    return selected


class CompetitionObjective:
    """Match the deployed scorer while mining its frozen beam competitors."""

    def __init__(self, support, catalog, vectors, device, search, teacher=None):
        identity = IdentityQuery().to(device)
        self.retriever = GraftRetriever(support, catalog, vectors, identity, device)
        self.search = search
        self.device = torch.device(device)
        self.vectors = self.retriever.vectors
        self.catalog = self.retriever.catalog
        self.leaf_vectors = self.retriever.profiles[2][:-1]
        self.leaf_codes = torch.as_tensor(
            np.unique(catalog.codes[catalog.codes[:, 0] >= 0], axis=0), device=device, dtype=torch.long
        )
        self.sizes = [len(book) for book in catalog.books]
        self.pair_keys = torch.unique(self.leaf_codes[:, 0] * self.sizes[1] + self.leaf_codes[:, 1], sorted=True)
        self.pair_parent = self.pair_keys // self.sizes[1]
        self.pair_vectors = self.retriever.profiles[1][self.retriever.lookup[1][self.pair_keys]]
        self.root_vectors = self.retriever.profiles[0][self.retriever.lookup[0]]
        self.leaf_pair = torch.searchsorted(
            self.pair_keys, self.leaf_codes[:, 0] * self.sizes[1] + self.leaf_codes[:, 1]
        )
        self.pair_birth = self.catalog.second_layer_birth.flatten()[self.pair_keys]
        keys = (self.leaf_codes[:, 0] * self.sizes[1] + self.leaf_codes[:, 1]) * self.sizes[2] + self.leaf_codes[:, 2]
        self.leaf_birth = self.catalog._sparse_leaf_birth[keys]
        self.leaf_new = ~self.retriever.support[:-1]
        self.birth_order = torch.argsort(self.leaf_birth, stable=True)
        self.sorted_birth = self.leaf_birth[self.birth_order]
        self.generator = torch.Generator(device=device).manual_seed(2025)
        self.teacher = teacher

    def target_leaves(self, targets):
        codes = self.catalog.codes[targets]
        if not bool((codes >= 0).all()):
            raise ValueError("Target is not registered in this training catalog")
        keys = (codes[:, 0] * self.sizes[1] + codes[:, 1]) * self.sizes[2] + codes[:, 2]
        return self.retriever.lookup[2][keys]

    @torch.no_grad()
    def prefix_scores(self, query, times):
        root = query @ self.root_vectors.T / self.search.temperature
        root.masked_fill_(self.catalog.first_layer_birth[None] > times[:, None], -torch.inf)
        root_probability = root.log_softmax(-1)
        roots = root.topk(min(self.search.coarse_beam, root.shape[1]), dim=-1).indices
        root_retained = torch.zeros_like(root, dtype=torch.bool).scatter_(1, roots, True)
        pair = query @ self.pair_vectors.T / self.search.temperature
        pair.masked_fill_(self.pair_birth[None] > times[:, None], -torch.inf)
        parent = self.pair_parent[None].expand(len(query), -1)
        maximum = torch.full_like(root, -torch.inf)
        maximum.scatter_reduce_(1, parent, pair, reduce="amax", include_self=True)
        centered = torch.where(pair.isfinite(), pair - maximum.gather(1, parent), -torch.inf)
        sums = torch.zeros_like(maximum).scatter_add_(1, parent, centered.exp())
        normalizer = maximum + sums.clamp(min=1e-30).log()
        scores = root_probability.gather(1, parent) + pair - normalizer.gather(1, parent)
        scores = scores.masked_fill(~pair.isfinite(), -torch.inf)
        routed_scores = scores.masked_fill(~root_retained.gather(1, parent), -torch.inf)
        selected = routed_scores.topk(min(self.search.coarse_beam, scores.shape[1]), dim=-1).indices
        retained = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, selected, True)
        retained &= routed_scores.isfinite()
        return scores, retained

    @torch.no_grad()
    def mine(self, query, targets, times, config):
        target_leaf = self.target_leaves(targets)
        prefix_scores, retained = self.prefix_scores(query, times)
        reference = query @ self.leaf_vectors.T / self.search.temperature
        reference += self.search.coarse_weight * prefix_scores[:, self.leaf_pair]
        reference += self.leaf_new[None] * self.search.new_leaf_bias
        reference.masked_fill_(self.leaf_birth[None] > times[:, None], -torch.inf)
        reference.scatter_(1, target_leaf[:, None], -torch.inf)
        target_pair = self.leaf_pair[target_leaf]
        siblings = self.leaf_pair[None] == target_pair[:, None]
        sibling_scores = reference.masked_fill(~siblings, -torch.inf)
        _, sibling_ids = sibling_scores.topk(min(config.sibling_budget, len(self.leaf_codes)), dim=-1)
        sibling_ids = sibling_ids.masked_fill(~sibling_scores.gather(1, sibling_ids).isfinite(), -1)
        routed_scores = reference.masked_fill(~retained[:, self.leaf_pair] | siblings, -torch.inf)
        _, routed_ids = routed_scores.topk(min(config.routed_budget, len(self.leaf_codes)), dim=-1)
        routed_ids = routed_ids.masked_fill(~routed_scores.gather(1, routed_ids).isfinite(), -1)
        visible_counts = torch.searchsorted(self.sorted_birth, times.contiguous(), right=True)
        if not bool((visible_counts > 0).all()):
            raise ValueError("No available training leaves")
        random_positions = (
            torch.rand((len(query), 2 * config.negatives), device=self.device, generator=self.generator)
            * visible_counts[:, None]
        ).long()
        random_ids = self.birth_order[random_positions]
        combined = torch.cat([sibling_ids, routed_ids, random_ids], dim=-1)
        negatives = unique_visible_negatives(combined, target_leaf, self.leaf_birth, times, config.negatives)
        stats = {
            "sibling_negatives": ((self.leaf_pair[negatives.clamp(min=0)] == target_pair[:, None]) & (negatives >= 0))
            .sum(-1)
            .float()
            .mean(),
            "valid_negatives": (negatives >= 0).sum(-1).float().mean(),
        }
        reachable = retained.gather(1, target_pair[:, None]).squeeze(-1)
        stats["reachable_targets"] = reachable.float().mean()
        return target_leaf, negatives, prefix_scores, reachable, stats

    def __call__(self, model, query, targets, times, config):
        query = F.normalize(query.float(), dim=-1)
        positive, negatives, prefix_scores, reachable, stats = self.mine(query, targets, times, config)
        candidates = torch.cat([positive[:, None], negatives], dim=-1)
        safe = candidates.clamp(min=0)
        parent_vectors = self.pair_vectors[self.leaf_pair[safe]]
        repeated = query[:, None].expand_as(parent_vectors)
        adapted = model(repeated, parent_vectors)
        with torch.no_grad():
            reference_query = self.teacher(repeated, parent_vectors) if self.teacher is not None else repeated
        scores = (adapted * self.leaf_vectors[safe]).sum(-1) / self.search.temperature
        scores += self.search.coarse_weight * prefix_scores.gather(1, self.leaf_pair[safe])
        scores += self.leaf_new[safe] * self.search.new_leaf_bias
        scores = scores.masked_fill(candidates < 0, -torch.inf)
        if not bool(scores[:, 0].isfinite().all()):
            raise ValueError("Training target score is not finite")
        per_query_path = F.cross_entropy(
            scores, torch.zeros(len(query), device=self.device, dtype=torch.long), reduction="none"
        )
        supervised = self.leaf_new[positive] if config.episode_new_only else torch.ones_like(positive, dtype=torch.bool)
        if config.reachable_only:
            supervised &= reachable
        path = (per_query_path * supervised).sum() / supervised.sum().clamp(min=1)
        valid = candidates >= 0
        movement = (adapted - reference_query).square().sum(-1)
        anchor = (movement * valid).sum() / valid.sum()
        teacher_loss = scores[:, 0].sum() * 0
        if config.teacher_weight:
            with torch.no_grad():
                teacher = (reference_query * self.leaf_vectors[safe]).sum(-1) / self.search.temperature
                teacher += self.search.coarse_weight * prefix_scores.gather(1, self.leaf_pair[safe])
                teacher += self.leaf_new[safe] * self.search.new_leaf_bias
                teacher.masked_fill_(~valid, -torch.inf)
                log_teacher = teacher.log_softmax(-1)
                probability = log_teacher.exp()
            log_student = scores.log_softmax(-1)
            difference = torch.where(valid, log_teacher - log_student, torch.zeros_like(log_student))
            per_query_teacher = (probability * difference).sum(-1)
            preserve = ~supervised if config.episode_new_only else torch.ones_like(supervised)
            teacher_loss = (per_query_teacher * preserve).sum() / preserve.sum().clamp(min=1)
        stats |= {"path": path, "anchor": anchor, "teacher": teacher_loss}
        return (
            config.base_path_weight * path + config.anchor_weight * anchor + config.teacher_weight * teacher_loss
        ), stats


def train_refinement(dataset, output, device, config, episode_directory=None, bias=None, warm_start=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    seed_everything(dataset.seed)
    device = torch.device(device)
    torch.cuda.set_device(device) if device.type == "cuda" else None
    search = replace(
        dataset.search,
        new_leaf_bias=bias if bias is not None else dataset.search.new_leaf_bias,
        batch_size=dataset.search.batch_size,
    )
    base, deployed = Catalog.load(dataset.path("base_catalog")), Catalog.load(dataset.path("catalog"))
    vectors_np = np.load(dataset.path("cache") / "item_vectors.npy")
    if len(vectors_np) < len(base.codes):
        vectors_np = np.pad(vectors_np, ((0, len(base.codes) - len(vectors_np)), (0, 0)))
    vectors = torch.as_tensor(vectors_np, device=device).float()
    teacher, initial_state = None, None
    if warm_start:
        teacher, _ = load_scorer(warm_start, device)
        initial_state = teacher.state_dict()
        if not bool(torch.count_nonzero(teacher.net[-1].weight)) and not bool(
            torch.count_nonzero(teacher.net[-1].bias)
        ):
            teacher = None
    if config.negative_strategy == "competition":
        objective = CompetitionObjective(base, base, vectors, device, replace(search, new_leaf_bias=0), teacher)
        objective.generator.manual_seed(dataset.seed + 1)
    elif config.negative_strategy == "uniform":
        if episode_directory:
            raise ValueError("The uniform reference uses the original exemplar task, not arrival episodes")
        base._prefer_sparse_lookup = True
        objective = PathObjective(base, vectors, device)
    else:
        raise ValueError("Unknown negative strategy")
    with np.load(dataset.data_directory / "p2_train.npz") as source:
        targets_np, timestamps_np = source["item"], source["timestamp"]
    targets, times = torch.as_tensor(targets_np, device=device), torch.as_tensor(timestamps_np, device=device)
    with np.load(dataset.data_directory / "p2_valid.npz") as source:
        valid = {key: source[key] for key in ["item", "timestamp"]}
    with np.load(dataset.data_directory / "catalog.npz") as source:
        seen = source["training_seen"]
    train_query = np.load(dataset.path("cache") / "train_queries.npy", mmap_mode="r")
    valid_query = np.load(dataset.path("cache") / "valid_queries.npy", mmap_mode="r")
    query_generator = torch.Generator(device=device).manual_seed(dataset.seed + 100)
    episode_generator = np.random.default_rng(dataset.seed + 200)
    episodes = []
    if episode_directory:
        metadata = json.loads((episode_directory / "result.json").read_text())
        for info in metadata["episodes"]:
            for key in ("support_catalog", "catalog", "query_indices"):
                info[key] = str(project_path(info[key], episode_directory))
        assert metadata["dataset"] == dataset.dataset and metadata["test_access"] == "none"
        for info in metadata["episodes"]:
            support = Catalog.load(info["support_catalog"])
            catalog = Catalog.load(info["catalog"])
            assert sha256(info["support_catalog"]) == info["support_sha256"]
            assert sha256(info["catalog"]) == info["catalog_sha256"]
            assert sha256(info["query_indices"]) == info["queries_sha256"]
            ids = np.load(info["query_indices"])
            cold = support.codes[targets_np[ids], 0] < 0
            episode_objective = CompetitionObjective(support, catalog, vectors, device, search, teacher)
            episode_objective.generator.manual_seed(dataset.seed + 1)
            episodes.append((episode_objective, ids[cold], ids[~cold]))
    model_args = {"dimension": vectors.shape[-1], "hidden": 256}
    # Paired controls share initialization independently of catalog setup.
    torch.manual_seed(dataset.seed)
    torch.cuda.manual_seed_all(dataset.seed)
    model = PrefixConditioner(**model_args).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=device.type == "cuda"
    )

    def learning_rate_scale(step):
        if step < config.warmup_steps:
            return max(1, step) / config.warmup_steps
        return 0.5 * (
            1
            + math.cos(math.pi * min(1, (step - config.warmup_steps) / (config.scheduler_steps - config.warmup_steps)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    retriever = GraftRetriever(base, deployed, vectors, model, device)
    indices = np.sort(
        np.random.default_rng(2024).choice(
            len(valid["item"]), min(config.validation_size, len(valid["item"])), replace=False
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    source_snapshot = output / "source"
    source_snapshot.mkdir()
    for name in ["refinement.py", "training_episodes.py", "registration.py", "scorer.py", "training.py"]:
        import shutil

        shutil.copy2(Path(__file__).parent / name, source_snapshot / name)
    np.save(output / "diagnostic_query_indices.npy", indices)
    settings = dict(
        dataset=dataset.dataset,
        seed=dataset.seed,
        training=config.__dict__,
        search=search.__dict__,
        episodes=str(episode_directory) if episode_directory else None,
        test_access="none",
        episode_loss_combination="weighted mean of base and episode path plus anchor objectives",
        episode_sampling="equal new/support targets when both groups available",
        teacher_scope="previous scorer on same visible leaves"
        if teacher is not None
        else "identity-query score distribution",
        warm_start=str(warm_start) if warm_start else None,
        warm_start_sha256=sha256(warm_start) if warm_start else None,
        episode_label_scope="new leaves only" if config.episode_new_only else "all episode targets",
        path_supervision_scope="target prefix retained by frozen beam" if config.reachable_only else "all targets",
        initial_parameter_hash=parameter_hash(model),
        final_catalog_sha256=sha256(dataset.path("catalog")),
        base_catalog_sha256=sha256(dataset.path("base_catalog")),
        source_sha256={
            name: sha256(Path(__file__).parent / name)
            for name in ["refinement.py", "training_episodes.py", "registration.py", "scorer.py", "training.py"]
        },
    )
    atomic_json(output / "config.json", settings)
    started, history = time.monotonic(), []

    def save_checkpoint(step):
        path = output / f"step_{step:04d}.pt"
        torch.save(
            dict(
                model=model.state_dict(),
                model_args=model_args,
                step=step,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                experiment=settings,
                parameter_hash=parameter_hash(model),
            ),
            path,
        )

    def validate(step):
        model.eval()
        records, metrics, diagnostics = evaluate_queries(retriever, valid_query, valid, indices, seen, search)
        for group, mask in [("new", ~seen[records["item"]]), ("exposed", seen[records["item"]])]:
            from .metrics import stage_diagnostics

            diagnostics[group] = stage_diagnostics(records["stages"][mask], records["rank"][mask])
        history.append(dict(step=step, metrics=metrics, diagnostics=diagnostics))
        np.savez_compressed(output / f"diagnostic_step_{step:04d}.npz", **records)
        atomic_json(output / "diagnostic_history.json", history)
        print(json.dumps({"event": "diagnostic", **history[-1]}), flush=True)

    save_checkpoint(0)
    validate(0)
    for step in range(1, config.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = torch.randint(len(targets), (config.batch_size,), device=device, generator=query_generator)
        query = torch.as_tensor(np.array(train_query[ids.cpu().numpy()], copy=True), device=device)
        if config.negative_strategy == "uniform":
            uniform_config = TrainingConfig(
                negatives=config.negatives, graft_weight=2, anchor_weight=config.anchor_weight
            )
            loss, terms = objective(
                model, query, targets[ids], times[ids], uniform_config, search.coarse_weight, search.temperature
            )
        else:
            loss, terms = objective(model, query, targets[ids], times[ids], config)
        if episodes:
            episode_objective, new_ids, support_ids = episodes[(step - 1) % len(episodes)]
            if len(new_ids) and len(support_ids):
                half = config.episode_batch_size // 2
                chosen = np.r_[
                    episode_generator.choice(new_ids, half),
                    episode_generator.choice(support_ids, config.episode_batch_size - half),
                ]
            else:
                chosen = episode_generator.choice(np.r_[new_ids, support_ids], config.episode_batch_size)
            episode_ids = torch.as_tensor(chosen, device=device)
            episode_query = torch.as_tensor(np.array(train_query[chosen], copy=True), device=device)
            episode_config = replace(config, base_path_weight=config.episode_path_weight)
            extra, extra_terms = episode_objective(
                model, episode_query, targets[episode_ids], times[episode_ids], episode_config
            )
            loss = (loss + config.episode_weight * extra) / (1 + config.episode_weight)
            terms |= {"episode_" + key: value for key, value in extra_terms.items()}
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite refinement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            progress = dict(
                step=step,
                loss=float(loss.detach()),
                seconds=round(time.monotonic() - started, 2),
                **{key: float(value.detach()) for key, value in terms.items()},
            )
            atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if step in config.checkpoints or step == config.steps:
            save_checkpoint(step)
            validate(step)
    result = dict(
        status="completed",
        dataset=dataset.dataset,
        seed=dataset.seed,
        trained_steps=step,
        checkpoints=[x["step"] for x in history],
        test_access="none",
        seconds=round(time.monotonic() - started, 2),
        diagnostic_split="valid",
        checkpoint_policy="configured final step",
        final_checkpoint=f"step_{config.steps:04d}.pt",
    )
    atomic_json(output / "result.json", result)
    return result
