"""Explicit validation/test evaluation and nested path diagnostics."""

from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from .catalog import Catalog
from .config import SearchConfig
from .descriptors import prototype_tree
from .io import atomic_json, fresh_directory, parameter_hash, sha256
from .metrics import group_metrics, stage_diagnostics
from .retrieval import decode_paths, resolve_items
from .scorer import load_scorer


def training_leaf_support(base, expanded, lookup, profiles, device):
    sizes = [len(book) for book in expanded.books]
    codes = base.codes[base.codes[:, 0] >= 0].astype(np.int64)
    keys = (codes[:, 0] * sizes[1] + codes[:, 1]) * sizes[2] + codes[:, 2]
    support = torch.zeros(len(profiles[2]), dtype=torch.bool, device=device)
    support[lookup[2][torch.as_tensor(np.unique(keys), device=device)]] = True
    return support


class GraftRetriever:
    def __init__(self, base, catalog, vectors, scorer, device="cpu"):
        self.device, self.scorer = torch.device(device), scorer
        self.vectors = torch.as_tensor(vectors, device=self.device).float()
        old = base.codes[:, 0] >= 0
        catalog._prefer_sparse_lookup = True
        self.lookup, self.profiles = prototype_tree(catalog, self.vectors, old, self.device)
        self.catalog = catalog.torch(self.device, sparse=True)
        self.support = training_leaf_support(base, catalog, self.lookup, self.profiles, self.device)

    @torch.inference_mode()
    def recommend(self, query, times, config: SearchConfig):
        query = F.normalize(torch.as_tensor(query, device=self.device).float(), dim=-1)
        times = torch.as_tensor(times, device=self.device, dtype=torch.long)
        paths, scores, stages = decode_paths(
            query, times, self.catalog, self.lookup, self.profiles, self.scorer, self.support, config
        )
        items = resolve_items(query, times, paths, scores, self.catalog, self.vectors, config)
        return items, paths, scores, stages


@torch.inference_mode()
def evaluate_queries(retriever, queries, arrays, indices, seen, config):
    if not len(indices):
        raise ValueError("Evaluation requires at least one query")
    records = {key: [] for key in ("items", "rank", "sid_hit", "stages")}
    for start in range(0, len(indices), config.batch_size):
        chosen = indices[start : start + config.batch_size]
        items, _, _, stages = retriever.recommend(queries[chosen], arrays["timestamp"][chosen], config)
        targets = torch.as_tensor(arrays["item"][chosen], device=retriever.device)
        matches = items == targets[:, None]
        ranks = torch.where(matches.any(-1), matches.float().argmax(-1) + 1, 0)
        survival = torch.stack(
            [
                (
                    (stage["paths"] == retriever.catalog.codes[targets][:, None, : level + 1]).all(-1)
                    & stage["scores"].isfinite()
                ).any(-1)
                for level, stage in enumerate(stages)
            ],
            -1,
        )
        for key, value in {"items": items, "rank": ranks, "sid_hit": survival[:, 2], "stages": survival}.items():
            records[key].append(value.cpu().numpy())
    records = {key: np.concatenate(values) for key, values in records.items()}
    records.update(query_index=indices, item=arrays["item"][indices])
    metrics = group_metrics(records["rank"], records["sid_hit"], records["item"], seen)
    diagnostics = stage_diagnostics(records["stages"], records["rank"])
    for name, mask in (("new", ~seen[records["item"]]), ("exposed", seen[records["item"]])):
        diagnostics[name] = stage_diagnostics(records["stages"][mask], records["rank"][mask])
    return records, metrics, diagnostics


def load_retriever(
    config,
    device="cpu",
    catalog_path=None,
    scorer_path=None,
    split="valid",
    identity_query=False,
    parent_descriptor=False,
):
    base = Catalog.load(config.path("base_catalog"))
    catalog = Catalog.load(catalog_path or config.path("test_catalog" if split == "test" else "catalog"))
    vectors = np.load(config.path("cache") / "item_vectors.npy", allow_pickle=False)
    if len(vectors) != len(catalog.codes):
        raise ValueError("Cached vectors do not cover every catalog item")
    scorer, saved = load_scorer(scorer_path or config.path("scorer_checkpoint"), device)
    if identity_query:
        from .refinement import IdentityQuery

        scorer = IdentityQuery().to(device)
    retriever = GraftRetriever(base, catalog, vectors, scorer, device)
    if parent_descriptor:
        codes = np.unique(catalog.codes[catalog.codes[:, 0] >= 0], axis=0)
        sizes = catalog.base_sizes
        keys = (codes[:, 0] * sizes[1] + codes[:, 1]) * sizes[2] + codes[:, 2]
        positions = retriever.lookup[2][torch.as_tensor(keys, device=device)]
        pair_keys = codes[:, 0] * sizes[1] + codes[:, 1]
        parents = retriever.profiles[1][retriever.lookup[1][torch.as_tensor(pair_keys, device=device)]]
        is_new = ~retriever.support[positions]
        retriever.profiles[2][positions[is_new]] = parents[is_new]
    return retriever, saved


def run_evaluation(
    config,
    output,
    device="cpu",
    split="test",
    limit=None,
    bias=None,
    catalog_path=None,
    scorer_path=None,
    identity_query=False,
    parent_descriptor=False,
):
    if split not in ("valid", "test") or (limit is not None and limit < 1):
        raise ValueError("Use valid/test with a positive query limit")
    if config.search.top_k < 50:
        raise ValueError("Recall/NDCG@50 require top_k >= 50")
    if Path(output).exists() and any(Path(output).iterdir()):
        raise FileExistsError(output)
    started = time.monotonic()
    catalog_path = catalog_path or config.path("test_catalog" if split == "test" else "catalog")
    scorer_path = scorer_path or config.path("scorer_checkpoint")
    retriever, checkpoint = load_retriever(
        config, device, catalog_path, scorer_path, split, identity_query, parent_descriptor
    )
    before = parameter_hash(retriever.scorer)
    source = config.data_directory / f"p2_{split}.npz"
    with np.load(source, allow_pickle=False) as file:
        arrays = {key: file[key] for key in ("item", "timestamp")}
    with np.load(config.data_directory / "catalog.npz", allow_pickle=False) as file:
        seen = file["training_seen"]
    query_path = config.path("cache") / f"{split}_queries.npy"
    queries = np.load(query_path, mmap_mode="r", allow_pickle=False)
    if len(queries) != len(arrays["item"]):
        raise ValueError("Query cache and evaluation split have different row counts")
    indices = np.arange(min(limit, len(queries)) if limit is not None else len(queries))
    target_ids = arrays["item"][indices]
    if not bool((retriever.catalog.codes[target_ids] >= 0).all()):
        raise ValueError("An evaluation target has not been registered; register the selected split first")
    search = replace(config.search, new_leaf_bias=bias) if bias is not None else config.search
    records, metrics, diagnostics = evaluate_queries(retriever, queries, arrays, indices, seen, search)
    if parameter_hash(retriever.scorer) != before:
        raise AssertionError("Evaluation changed scorer parameters")
    items = records["items"]
    first = retriever.catalog.first.cpu().numpy()
    if not np.all((items == 0) | (first[items] <= arrays["timestamp"][indices, None])):
        raise AssertionError("Recommendation contains a future item")
    output = fresh_directory(output)
    np.savez_compressed(output / f"{split}_predictions.npz", **records)
    result = dict(
        status="completed",
        method="Graft",
        dataset=config.dataset,
        domain=config.domain,
        protocol="P2",
        seed=config.seed,
        split=split,
        full_split=len(indices) == len(queries),
        metric_units="fraction",
        selected_step=checkpoint.get("step"),
        metrics=metrics,
        diagnostics=diagnostics,
        search=search.__dict__,
        identity_query=identity_query,
        parent_descriptor=parent_descriptor,
        scorer_parameters_frozen=True,
        source_hashes=dict(
            queries=sha256(source), cache=sha256(query_path), scorer=sha256(scorer_path), catalog=sha256(catalog_path)
        ),
        seconds=round(time.monotonic() - started, 2),
    )
    atomic_json(output / "result.json", result)
    return result
