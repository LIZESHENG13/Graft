"""Explicit chronological splits and one zero-padding row for item content."""

from __future__ import annotations

import json
import numpy as np

from .config import DatasetConfig, load_config

SPLITS = ("train", "valid", "test")


def right_pad(history):
    lengths = (history != 0).sum(-1)
    if np.any(lengths == 0):
        raise ValueError("Empty-history queries must be removed during preparation")
    width = history.shape[1]
    offsets = np.arange(width)[None] + (width - lengths[:, None])
    output = np.take_along_axis(history, np.minimum(offsets, width - 1), axis=1)
    output[np.arange(width)[None] >= lengths[:, None]] = 0
    return output.astype(np.int64), lengths.astype(np.int64)


def load_split(config, protocol="P2", split="train"):
    if not isinstance(config, DatasetConfig):
        config = load_config(config)
    if protocol != "P2" or split not in SPLITS:
        raise ValueError(f"Expected P2 and one of {SPLITS}")
    with np.load(config.data_directory / f"p2_{split}.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if not len(arrays["item"]):
        raise ValueError(f"The {split} split has no nonempty-history queries")
    arrays["right_history"], arrays["length"] = right_pad(arrays["history"])
    return arrays


def catalog_metadata(config):
    with np.load(config.data_directory / "catalog.npz", allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def load_domain(config, protocol="P2"):
    if not isinstance(config, DatasetConfig):
        config = load_config(config)
    metadata = catalog_metadata(config)
    first, seen = metadata["first_timestamp"], metadata["training_seen"].copy()
    seen[0] = False
    content = np.load(config.data_directory / "text_vectors.npy", mmap_mode="r", allow_pickle=False)
    if len(content) != len(first) or content.ndim != 2:
        raise ValueError("Text vectors and catalog item rows disagree")
    if content[0].any() or not np.isfinite(content).all():
        raise ValueError("Text vectors require finite values and one zero padding row")
    train = load_split(config, protocol, "train")
    valid = load_split(config, protocol, "valid")
    audit = json.loads((config.data_directory / "audit.json").read_text())
    train_ids = np.flatnonzero(first <= audit["timestamp_cutoffs_ms"]["train"])
    valid_ids = np.flatnonzero(first <= audit["timestamp_cutoffs_ms"]["valid"])
    train_limit = int(train_ids.max()) + 1
    valid_limit = int(valid_ids.max()) + 1
    if not np.array_equal(train_ids, np.arange(1, train_limit)):
        raise ValueError("Item IDs must follow first registration time")
    return train, valid, content, first, seen, train_limit, valid_limit
