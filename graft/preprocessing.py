"""Build strict-past histories from chronological interaction files."""

from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import atomic_json, fresh_directory, sha256


def _build(frame, item2id, user2id, vectors, output, domain, max_history=20):
    """Equal-time events are all queried before any of them enters history."""
    if max_history < 1:
        raise ValueError("max_history must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    mapped_users = frame.user_id.map(user2id)
    mapped_items = frame.item_id.map(item2id)
    if mapped_users.isna().any() or mapped_items.isna().any():
        raise ValueError("Interaction IDs are missing from the supplied mappings")
    users, items = mapped_users.to_numpy(np.int32), mapped_items.to_numpy(np.int32)
    timestamps, phases = frame.timestamp.to_numpy(np.int64), frame.phase.to_numpy(np.uint8)
    ni = max(item2id.values()) + 1
    if vectors.shape[0] != ni or vectors.ndim != 2:
        raise ValueError("Content rows must match item IDs, including padding ID 0")
    if vectors[0].any() or not np.isfinite(vectors).all():
        raise ValueError("Content must be finite with a zero padding row")
    if len(set(item2id.values())) != len(item2id) or not np.array_equal(
        np.sort(np.unique(list(item2id.values()))), np.arange(ni)
    ):
        raise ValueError("Item mappings must be unique and contiguous from zero")
    if (items <= 0).any():
        raise ValueError("Padding ID 0 cannot appear in interactions")
    cutoffs = {}
    for phase, name in enumerate(("train", "valid", "test")):
        time = timestamps[phases == phase]
        if not len(time):
            raise ValueError(f"Missing {name} interactions")
        cutoffs[name] = int(time.max())
        if phase and time.min() <= cutoffs[("train", "valid")[phase - 1]]:
            raise ValueError("Train, validation and test time ranges must not overlap")
    first = np.full(ni, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(first, items, timestamps)
    stable = np.zeros(ni, np.uint64)
    for raw, iid in item2id.items():
        stable[iid] = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")
    order = np.lexsort((stable[items], timestamps, users))
    histories = np.zeros((len(frame), max_history), np.int32)
    history_times = np.zeros((len(frame), max_history), np.int64)
    boundaries = np.flatnonzero(np.r_[True, users[order][1:] != users[order][:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = order[start:end]
        prior = deque(maxlen=max_history)
        position = 0
        while position < len(group):
            stop = position + 1
            stamp = timestamps[group[position]]
            while stop < len(group) and timestamps[group[stop]] == stamp:
                stop += 1
            tied = group[position:stop]
            if prior:
                ids, times = zip(*prior)
                histories[tied, -len(prior) :] = ids
                history_times[tied, -len(prior) :] = times
            for index in tied:
                prior.append((int(items[index]), int(stamp)))
            position = stop
    keep = np.any(histories != 0, axis=1)
    if not np.all((histories == 0) | (history_times < timestamps[:, None])):
        raise AssertionError("A history interaction is not strictly earlier than its query")
    train = (phases == 0) & keep
    exposed = np.union1d(items[train], histories[train].ravel())
    exposed = exposed[exposed != 0]
    seen = np.zeros(ni, bool)
    seen[exposed] = True
    registered = np.flatnonzero(first <= cutoffs["train"]).astype(np.int32)
    if not np.array_equal(registered, np.arange(1, int(registered.max()) + 1)):
        raise ValueError("Item IDs must be ordered by first registration time")
    fresh_directory(output)
    split_audit = {}
    for phase, name in enumerate(("train", "valid", "test")):
        chosen = np.flatnonzero((phases == phase) & keep)
        chosen = chosen[np.argsort(timestamps[chosen], kind="stable")]
        if not len(chosen):
            raise ValueError(f"No nonempty-history queries in {name}")
        np.savez_compressed(
            output / f"p2_{name}.npz",
            user=users[chosen],
            item=items[chosen],
            timestamp=timestamps[chosen],
            history=histories[chosen],
            source_row=chosen,
        )
        split_audit[name] = dict(
            raw_rows=int((phases == phase).sum()),
            p2_nonempty_rows=len(chosen),
            cold_target_rows=int((~seen[items[chosen]]).sum()),
        )
    np.savez_compressed(
        output / "catalog.npz",
        first_timestamp=first,
        training_seen=seen,
        stable_order=stable,
        train_items=registered,
        train_exposed_items=exposed,
    )
    target = np.lib.format.open_memmap(output / "text_vectors.npy", mode="w+", dtype=np.float32, shape=vectors.shape)
    for start in range(0, ni, 4096):
        target[start : start + 4096] = vectors[start : start + 4096]
    target.flush()
    del target
    atomic_json(output / "item_mapping.json", item2id)
    report = dict(
        status="passed",
        domain=domain,
        protocol="P2",
        items=ni - 1,
        text_dimension=int(vectors.shape[1]),
        timestamp_cutoffs_ms=cutoffs,
        split_audit=split_audit,
        strict_past_history_verified=True,
        max_history_len=max_history,
        training_visible_items=len(registered),
        training_interaction_exposed_items=len(exposed),
        equal_timestamp_rule="query all events before committing the tied group",
        registration_proxy="first observed interaction",
    )
    atomic_json(output / "audit.json", report)
    return report


def prepare_amazon(source, benchmarks, domain, output, max_history=20):
    """Import SpecGR-format mappings/features and the original timestamp CSVs."""
    source, benchmarks = Path(source), Path(benchmarks)
    mapping_path = source / f"{domain}.data_maps"
    mapping = json.loads(mapping_path.read_text())
    frames, files = [], {}
    for phase, name in enumerate(("train", "valid", "test")):
        path = benchmarks / f"{domain}.{name}.csv"
        frame = pd.read_csv(
            path, keep_default_na=False, dtype={"user_id": str, "parent_asin": str, "timestamp": np.int64}
        )
        frame = frame.rename(columns={"parent_asin": "item_id"})
        frame["phase"] = phase
        frames.append(frame[["user_id", "item_id", "timestamp", "phase"]])
        files[name] = {"filename": path.name, "sha256": sha256(path)}
    feature_path = source / f"{domain}.sent_emb"
    rows = max(mapping["item2id"].values()) + 1
    dimension, remainder = divmod(feature_path.stat().st_size, rows * 4)
    if remainder or dimension < 1:
        raise ValueError("Invalid float32 text feature file size")
    vectors = np.memmap(feature_path, dtype=np.float32, mode="r", shape=(rows, dimension))
    report = _build(
        pd.concat(frames, ignore_index=True),
        mapping["item2id"],
        mapping["user2id"],
        vectors,
        output,
        domain,
        max_history,
    )
    report["sources"] = files | {
        "features": {"filename": feature_path.name, "sha256": sha256(feature_path)},
        "mapping": {"filename": mapping_path.name, "sha256": sha256(mapping_path)},
    }
    atomic_json(Path(output) / "audit.json", report)
    return report


def prepare_events(events, features, train_cutoff, valid_cutoff, output, domain="Custom", max_history=20):
    """Custom CSV + NPZ features; assign item IDs in first-registration order."""
    if train_cutoff >= valid_cutoff:
        raise ValueError("train_cutoff must precede valid_cutoff")
    frame = pd.read_csv(events, keep_default_na=False, dtype={"user_id": str, "item_id": str, "timestamp": np.int64})
    frame["phase"] = np.where(frame.timestamp <= train_cutoff, 0, np.where(frame.timestamp <= valid_cutoff, 1, 2))
    first = frame.groupby("item_id").timestamp.min()
    raw_items = sorted(first.index, key=lambda iid: (int(first[iid]), hashlib.sha256(iid.encode()).digest()))
    item2id = {"[PAD]": 0} | {iid: i + 1 for i, iid in enumerate(raw_items)}
    if "[PAD]" in raw_items:
        raise ValueError("[PAD] is reserved for padding")
    user2id = {uid: i + 1 for i, uid in enumerate(sorted(frame.user_id.unique()))}
    with np.load(features, allow_pickle=False) as file:
        ids, values = file["ids"].astype(str), file["vectors"]
        if len(ids) != len(values) or len(set(ids)) != len(ids):
            raise ValueError("Feature IDs must be unique and aligned with vectors")
        source_ids = {iid: i for i, iid in enumerate(ids)}
        missing = set(raw_items) - set(source_ids)
        if missing:
            raise ValueError(f"Missing content for {len(missing)} items")
        vectors = np.zeros((len(raw_items) + 1, values.shape[1]), np.float32)
        vectors[1:] = values[[source_ids[iid] for iid in raw_items]]
    return _build(frame, item2id, user2id, vectors, output, domain, max_history)
