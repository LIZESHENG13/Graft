"""Overall and training-unseen-item metrics and nested path diagnostics."""

import numpy as np


def group_metrics(ranks, sid_hits, targets, seen):
    metrics = {}
    masks = {"overall": np.ones(len(targets), bool), "warm": seen[targets], "cold": ~seen[targets]}
    for name, selected in masks.items():
        count = int(selected.sum())
        if not count:
            metrics[name] = {"count": 0}
            continue
        rank = ranks[selected]
        result = {"count": count, "sid_recall_50": float(sid_hits[selected].mean())}
        for k in [10, 50]:
            valid = (rank > 0) & (rank <= k)
            result[f"recall_{k}"] = float(valid.mean())
            result[f"ndcg_{k}"] = float(np.where(valid, 1 / np.log2(np.maximum(1, rank) + 1), 0).mean())
        metrics[name] = result
    return metrics


def stage_diagnostics(stages, ranks):
    prefix, leaf, item = stages[:, 1].astype(bool), stages[:, 2].astype(bool), ranks > 0
    if np.any(leaf & ~prefix) or np.any(item & ~leaf):
        raise AssertionError("Item hits must be contained in leaf and prefix hits")
    p, s, i = int(prefix.sum()), int(leaf.sum()), int(item.sum())
    return {
        "count": len(ranks),
        "prefix_hits": p,
        "sid_hits": s,
        "item_hits": i,
        "prefix_coverage": p / len(ranks) if len(ranks) else None,
        "conditional_sid_hit": s / p if p else None,
        "conditional_item_hit": i / s if s else None,
    }
