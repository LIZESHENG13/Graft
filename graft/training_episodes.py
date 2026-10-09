"""Training-period arrival catalogs using the deployment registration rule."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .catalog import Catalog, INF
from .io import atomic_json, sha256
from .registration import register_arrivals


def episode_support(base: Catalog, support_end: int) -> Catalog:
    """Exclude later items from support codes, descriptors, and memberships."""
    available = (base.codes[:, 0] >= 0) & (base.first <= support_end)
    codes = np.full_like(base.codes, -1)
    codes[available] = base.codes[available]
    first = base.first.copy()
    first[base.codes[:, 0] < 0] = INF
    return Catalog(
        [b.copy() for b in base.books], codes, first, base.code_birth.copy(), base.base_sizes, [], available.copy()
    )


def prepare_episodes(dataset, output: Path, device="cpu", fractions=(0.7, 0.85, 1.0)):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if not (
        len(fractions) >= 2
        and 0 < fractions[0] < 1
        and fractions[-1] == 1
        and all(a < b for a, b in zip(fractions[:-1], fractions[1:]))
    ):
        raise ValueError("Episode fractions must increase to one")
    started = time.monotonic()
    base = Catalog.load(dataset.path("base_catalog"))
    vectors = np.load(dataset.path("cache") / "item_vectors.npy")
    with np.load(dataset.data_directory / "p2_train.npz") as data:
        timestamps, targets = data["timestamp"], data["item"]
    with np.load(dataset.data_directory / "catalog.npz") as data:
        order = data["stable_order"]
    boundaries = np.quantile(timestamps, fractions, method="lower").astype(np.int64)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, (support_end, window_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        support = episode_support(base, int(support_end))
        catalog = register_arrivals(support, vectors, int(window_end), order, dataset.leaf_budget, device)
        query_ids = np.flatnonzero((timestamps > support_end) & (timestamps <= window_end))
        if not len(query_ids):
            raise ValueError("Episode contains no queries")
        registered = support.codes[:, 0] >= 0
        np.testing.assert_array_equal(catalog.codes[registered], support.codes[registered])
        assert np.all(catalog.codes[targets[query_ids], 0] >= 0)
        assert np.all(catalog.first[targets[query_ids]] <= timestamps[query_ids])
        before = np.unique(support.codes[registered, :2], axis=0)
        after = np.unique(catalog.codes[catalog.codes[:, 0] >= 0, :2], axis=0)
        np.testing.assert_array_equal(before, after)
        for old_book, new_book in zip(support.books, catalog.books):
            np.testing.assert_array_equal(old_book, new_book)
        stem = f"episode_{index}"
        support_file, catalog_file = output / f"{stem}_support.npz", output / f"{stem}_catalog.npz"
        queries_file = output / f"{stem}_queries.npy"
        support.save(support_file)
        catalog.save(catalog_file)
        np.save(queries_file, query_ids)
        summary = dict(
            episode=index,
            support_end=int(support_end),
            window_end=int(window_end),
            support_items=int(registered.sum()),
            arrivals=len(catalog.append_trace),
            new_leaves=sum(t["fresh_path_admitted"] for t in catalog.append_trace),
            queries=len(query_ids),
            new_target_queries=int((~registered[targets[query_ids]]).sum()),
            support_catalog=support_file.name,
            catalog=catalog_file.name,
            query_indices=queries_file.name,
            support_sha256=sha256(support_file),
            catalog_sha256=sha256(catalog_file),
            queries_sha256=sha256(queries_file),
            base_alphabets_unchanged=True,
            existing_prefixes_unchanged=True,
            support_addresses_unchanged=True,
        )
        summaries.append(summary)
        atomic_json(output / "progress.json", {"episodes": summaries})
        print(summary, flush=True)
    result = dict(
        status="completed",
        dataset=dataset.dataset,
        seed=dataset.seed,
        test_access="none",
        boundaries_axis="training-query timestamp quantiles",
        fractions=list(fractions),
        episodes=summaries,
        encoder_scope="frozen encoder learned on the training split",
        codebook_scope="fixed full-training codebooks; no validation/test content used in episodes",
        descriptors_scope="support-only means, then registration-time content exemplars",
        registration_scope="chronological deployment registration, fixed prefixes and alphabets",
        base_catalog_sha256=sha256(dataset.path("base_catalog")),
        seconds=round(time.monotonic() - started, 2),
    )
    atomic_json(output / "result.json", result)
    return result
