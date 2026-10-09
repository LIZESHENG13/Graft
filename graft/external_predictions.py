"""Apply the same item-level protocol to predictions from external baselines."""

import numpy as np

from .data import catalog_metadata
from .io import atomic_json, fresh_directory, sha256
from .metrics import group_metrics


def evaluate_predictions(config, predictions, output, method, split="test"):
    query_file = config.data_directory / f"p2_{split}.npz"
    with np.load(query_file, allow_pickle=False) as file:
        targets, timestamps = file["item"], file["timestamp"]
    with np.load(predictions, allow_pickle=False) as file:
        items = file["items"]
        if "query_index" not in file:
            raise ValueError("Baseline predictions must include query_index")
        np.testing.assert_array_equal(file["query_index"], np.arange(len(targets)))
    if items.shape != (len(targets), 50) or not np.issubdtype(items.dtype, np.integer):
        raise ValueError("Expected one integer Top-50 row per query")
    metadata = catalog_metadata(config)
    if (items < 0).any() or (items >= len(metadata["first_timestamp"])).any():
        raise ValueError("Baseline outputs contain out-of-range item IDs")
    if not np.all((items == 0) | (metadata["first_timestamp"][items] <= timestamps[:, None])):
        raise ValueError("Baseline recommends an item unavailable at its query time")
    ordered = np.sort(items, axis=1)
    if np.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != 0)):
        raise ValueError("Baseline Top-50 outputs contain duplicate items")
    match = items == targets[:, None]
    ranks = np.where(match.any(1), match.argmax(1) + 1, 0)
    metrics = group_metrics(ranks, ranks > 0, targets, metadata["training_seen"])
    for group in metrics.values():
        group.pop("sid_recall_50", None)
    output = fresh_directory(output)
    np.savez_compressed(
        output / f"{split}_predictions.npz", items=items, item=targets, rank=ranks, query_index=np.arange(len(targets))
    )
    result = dict(
        status="completed",
        method=method,
        dataset=config.dataset,
        seed=config.seed,
        split=split,
        full_split=True,
        metric_units="fraction",
        metrics=metrics,
        source_hashes={"queries": sha256(query_file), "predictions": sha256(predictions)},
    )
    atomic_json(output / "result.json", result)
    return result
