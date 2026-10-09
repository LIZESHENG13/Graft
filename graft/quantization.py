"""Fit base residual codebooks using only training-registered item vectors."""

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch

from .catalog import Catalog, INF
from .data import catalog_metadata
from .io import atomic_json, fresh_directory, sha256


def fit_codebooks(latents, ids, sizes, seed):
    residual = np.array(latents[ids], copy=True)
    if min(sizes) < 1 or max(sizes) > len(residual):
        raise ValueError("Every codebook size must fit the training-registered item count")
    books = []
    for level, size in enumerate(sizes):
        with threadpool_limits(limits=4):
            fitted = KMeans(n_clusters=size, n_init=10, random_state=seed + level).fit(residual)
        centers = fitted.cluster_centers_.astype(np.float32)
        books.append(centers)
        residual -= centers[fitted.labels_]
    return books


def assign(latents, books, batch_size=4096):
    chunks = []
    for start in range(0, len(latents), batch_size):
        residual = torch.as_tensor(np.array(latents[start : start + batch_size], copy=True)).float()
        digits = []
        for book in books:
            centers = torch.as_tensor(book)
            nearest = torch.cdist(residual, centers).square().argmin(-1)
            digits.append(nearest.numpy())
            residual = residual - centers[nearest]
        chunks.append(np.stack(digits, axis=-1))
    return np.concatenate(chunks).astype(np.int32)


def build_base_catalog(config, vectors_path, output):
    source = catalog_metadata(config)
    old_ids = source["train_items"]
    vectors = np.load(vectors_path, allow_pickle=False)
    sizes = tuple(config.codebook_sizes)
    books = fit_codebooks(vectors, old_ids, sizes, config.seed)
    codes = np.full((len(source["first_timestamp"]), 3), -1, np.int32)
    codes[old_ids] = assign(vectors[old_ids], books)
    base = Catalog(
        books,
        codes,
        source["first_timestamp"].copy(),
        np.full(sizes[2], -INF, np.int64),
        sizes,
        [],
        source["training_seen"].copy(),
    )
    output = fresh_directory(output)
    base.save(output / "train_catalog.npz")
    result = dict(
        status="completed",
        dataset=config.dataset,
        seed=config.seed,
        fit_split="train",
        training_registered_items=len(old_ids),
        codebook_sizes=list(sizes),
        input_vectors_sha256=sha256(vectors_path),
    )
    atomic_json(output / "result.json", result)
    return result
