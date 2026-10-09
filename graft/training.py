"""The paper's sampled path loss, target-exemplar task, and state anchor."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .catalog import Catalog
from .config import DatasetConfig
from .descriptors import prototype_tree
from .evaluation import GraftRetriever, evaluate_queries
from .io import atomic_json, seed_everything
from .scorer import PrefixConditioner, conditional_query


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 10000
    batch_size: int = 512
    negatives: int = 64
    learning_rate: float = 0.0002
    weight_decay: float = 0.05
    graft_weight: float = 2.0
    anchor_weight: float = 10.0
    eval_interval: int = 500
    patience: int = 6
    validation_size: int = 2048
    hidden: int = 256


class PathObjective:
    def __init__(self, base, vectors, device):
        self.vectors, self.device = vectors, torch.device(device)
        self.catalog = base.torch(device, sparse=True)
        self.sizes = [len(book) for book in base.books]
        old = base.codes[:, 0] >= 0
        self.lookup, self.profiles = prototype_tree(base, vectors, old, device)
        self.leaf_profiles = self.profiles[2][:-1]
        self.leaf_codes = torch.as_tensor(np.unique(base.codes[old], axis=0), device=device, dtype=torch.long)
        self.pair_keys = torch.as_tensor(
            np.unique(base.codes[old, 0] * self.sizes[1] + base.codes[old, 1]), device=device
        )
        self.pair_vectors = self.profiles[1][self.lookup[1][self.pair_keys]]
        codes = self.catalog.codes[torch.as_tensor(np.flatnonzero(old), device=device)]
        keys = (codes[:, 0] * self.sizes[1] + codes[:, 1]) * self.sizes[2] + codes[:, 2]
        self.leaf_birth = torch.full((len(self.leaf_profiles),), torch.iinfo(torch.int64).max, device=device)
        self.leaf_birth.scatter_reduce_(
            0,
            self.lookup[2][keys],
            self.catalog.first[torch.as_tensor(np.flatnonzero(old), device=device)],
            reduce="amin",
            include_self=True,
        )

    def __call__(self, model, query, targets, times, config: TrainingConfig, coarse_weight=0.25, temperature=0.07):
        query = F.normalize(query.float(), dim=-1)
        codes = self.catalog.codes[targets]
        if not bool((codes >= 0).all()):
            raise ValueError("Training targets must be registered in the training catalog")
        keys = (codes[:, 0] * self.sizes[1] + codes[:, 1]) * self.sizes[2] + codes[:, 2]
        target_leaf = self.lookup[2][keys]
        negatives = torch.randint(len(self.leaf_profiles), (len(query), config.negatives), device=self.device)
        candidates = torch.cat([target_leaf[:, None], negatives], -1)
        candidate_codes = self.leaf_codes[candidates]
        repeated = query[:, None].expand(-1, candidates.shape[-1], -1)
        adapted = conditional_query(
            model,
            repeated.reshape(-1, self.vectors.shape[-1]),
            candidate_codes[..., :2].reshape(-1, 2),
            self.lookup,
            self.profiles,
            self.sizes,
        ).reshape_as(repeated)
        scores = (adapted * self.leaf_profiles[candidates]).sum(-1) / temperature
        root_logits = query @ self.profiles[0][self.lookup[0]].T / temperature
        root_logits.masked_fill_(~self.catalog.valid_children(0, codes[:, :0], times), -torch.inf)
        root_probability = root_logits.log_softmax(-1)
        pair_logits = query @ self.pair_vectors.T / temperature
        pair_birth = self.catalog.second_layer_birth[self.pair_keys // self.sizes[1], self.pair_keys % self.sizes[1]]
        pair_logits.masked_fill_(pair_birth[None] > times[:, None], -torch.inf)
        parents = (self.pair_keys // self.sizes[1])[None].expand(len(query), -1)
        maximum = torch.full((len(query), self.sizes[0]), -torch.inf, device=self.device)
        maximum.scatter_reduce_(1, parents, pair_logits, reduce="amax", include_self=True)
        centered = torch.where(pair_logits.isfinite(), pair_logits - maximum.gather(1, parents), -torch.inf)
        sums = torch.zeros_like(maximum).scatter_add_(1, parents, centered.exp())
        normalizer = maximum + sums.clamp(min=1e-30).log()
        candidate_pair = candidate_codes[..., 0] * self.sizes[1] + candidate_codes[..., 1]
        pair_position = torch.searchsorted(self.pair_keys, candidate_pair)
        coarse = (
            root_probability.gather(1, candidate_codes[..., 0])
            + pair_logits.gather(1, pair_position)
            - normalizer.gather(1, candidate_codes[..., 0])
        )
        scores = scores + coarse_weight * coarse
        legal = self.leaf_birth[candidates] <= times[:, None]
        legal[:, 1:] &= negatives != target_leaf[:, None]
        scores = scores.masked_fill(~legal, -torch.inf)
        path_loss = F.cross_entropy(scores, torch.zeros(len(query), device=self.device, dtype=torch.long))
        exemplar = (adapted[:, 0] * self.vectors[targets]).sum(-1) / temperature + coarse_weight * coarse[:, 0]
        graft_loss = torch.logaddexp(scores[:, 1:].logsumexp(-1), exemplar).sub(exemplar).mean()
        anchor = (adapted - repeated).square().sum(-1).mean()
        total = path_loss + config.graft_weight * graft_loss + config.anchor_weight * anchor
        return total, {"path": path_loss, "graft": graft_loss, "anchor": anchor}


def train_scorer(dataset: DatasetConfig, output: Path, device="cpu", config=TrainingConfig()):
    if config.steps < 2 or config.negatives < 1 or config.eval_interval < 1:
        raise ValueError("Training requires steps >= 2 and positive negative/evaluation budgets")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    seed_everything(dataset.seed)
    device = torch.device(device)
    selection_catalog = (
        dataset.path("scorer_selection_catalog") if dataset.scorer_selection_catalog else dataset.path("catalog")
    )
    base, expanded = Catalog.load(dataset.path("base_catalog")), Catalog.load(selection_catalog)
    vectors = np.load(dataset.path("cache") / "item_vectors.npy")
    if len(vectors) < len(base.codes):
        vectors = np.pad(vectors, ((0, len(base.codes) - len(vectors)), (0, 0)))
    vectors = torch.as_tensor(vectors, device=device).float()
    objective = PathObjective(base, vectors, device)
    with np.load(dataset.data_directory / "p2_train.npz") as source:
        targets = torch.as_tensor(source["item"], device=device)
        times = torch.as_tensor(source["timestamp"], device=device)
    with np.load(dataset.data_directory / "p2_valid.npz") as source:
        valid = {key: source[key] for key in ["item", "timestamp"]}
    with np.load(dataset.data_directory / "catalog.npz") as source:
        seen = source["training_seen"]
    train_query = np.load(dataset.path("cache") / "train_queries.npy", mmap_mode="r")
    valid_query = np.load(dataset.path("cache") / "valid_queries.npy", mmap_mode="r")
    model_args = {"dimension": vectors.shape[-1], "hidden": config.hidden}
    model = PrefixConditioner(**model_args).to(device)
    retriever = GraftRetriever(base, expanded, vectors, model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=device.type == "cuda"
    )
    warmup = max(1, config.steps // 20)

    def lr_scale(step):
        if step < warmup:
            return max(1, step) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / (config.steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    indices = np.sort(
        np.random.default_rng(2024).choice(
            len(valid["item"]), min(config.validation_size, len(valid["item"])), replace=False
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "config.json",
        {
            "dataset": dataset.__dict__ | {"search": dataset.search.__dict__},
            "training": config.__dict__,
            "test_access": "none",
        },
    )
    history, best, stale = [], -float("inf"), 0
    started = time.monotonic()

    def save(path, step):
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "model_args": model_args,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_score": best,
                "stale": stale,
                "config": config.__dict__,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
            },
            temporary,
        )
        temporary.replace(path)

    def validate(step):
        nonlocal best, stale
        model.eval()
        _, metrics, _ = evaluate_queries(retriever, valid_query, valid, indices, seen, dataset.search)
        score = metrics["overall"]["ndcg_50"]
        improved = score > best
        if improved:
            best, stale = score, 0
            save(output / "best.pt", step)
        else:
            stale += 1
        history.append({"step": step, "metrics": metrics, "improved": improved})
        atomic_json(output / "validation_history.json", history)

    validate(0)
    for step in range(1, config.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        selected = torch.randint(len(targets), (config.batch_size,), device=device)
        query = torch.as_tensor(np.array(train_query[selected.cpu().numpy()], copy=True), device=device)
        loss, terms = objective(
            model,
            query,
            targets[selected],
            times[selected],
            config,
            dataset.search.coarse_weight,
            dataset.search.temperature,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Path loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0:
            progress = {"step": step, "loss": float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()}}
            atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if step % config.eval_interval == 0 or step == config.steps:
            validate(step)
            save(output / "last.pt", step)
            if stale >= config.patience:
                break
    result = {
        "status": "completed",
        "dataset": dataset.dataset,
        "seed": dataset.seed,
        "test_access": "none",
        "trained_steps": step,
        "selected_step": torch.load(output / "best.pt", map_location="cpu", weights_only=False)["step"],
        "best_selection_ndcg_50": best,
        "seconds": round(time.monotonic() - started, 2),
    }
    atomic_json(output / "result.json", result)
    return result
