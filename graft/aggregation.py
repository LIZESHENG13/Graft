"""Calculate seed means and SD from individual evaluation results."""

import json
from pathlib import Path
import numpy as np

from .io import atomic_json, sha256


def aggregate(paths, output, ddof=1):
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 (population) or 1 (sample)")
    records = [json.loads(Path(path).read_text()) for path in paths]
    if len(records) < 2:
        raise ValueError("At least two independent runs are required")
    seeds = [r["seed"] for r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed IDs must be distinct")
    for field in ("dataset", "method", "split", "full_split", "search", "identity_query", "parent_descriptor"):
        if any(r.get(field) != records[0].get(field) for r in records):
            raise ValueError(f"Runs disagree on {field}")
    if any(r["source_hashes"]["queries"] != records[0]["source_hashes"]["queries"] for r in records):
        raise ValueError("Runs use different query arrays")
    summary = {}
    for group in ("overall", "cold"):
        counts = [r["metrics"][group]["count"] for r in records]
        if len(set(counts)) != 1 or counts[0] == 0:
            raise ValueError(f"Runs have different or empty {group} query cohorts")
        summary[group] = {"count": counts[0]}
        for metric in ("recall_50", "ndcg_50"):
            values = np.array([r["metrics"][group][metric] for r in records], np.float64)
            summary[group][metric] = dict(
                mean=float(values.mean()),
                standard_deviation=float(values.std(ddof=ddof)),
                per_seed={str(seed): float(value) for seed, value in zip(seeds, values)},
            )
    result = dict(
        dataset=records[0]["dataset"],
        method=records[0]["method"],
        split=records[0]["split"],
        seeds=seeds,
        runs=len(records),
        dispersion="standard_deviation",
        ddof=ddof,
        metric_units="fraction",
        metrics=summary,
        sources=[dict(filename=Path(path).name, sha256=sha256(path)) for path in paths],
    )
    cells = []
    for group in ("overall", "cold"):
        for metric in ("recall_50", "ndcg_50"):
            value = summary[group][metric]
            cells.append(f"${100 * value['mean']:.3f}\\pm{100 * value['standard_deviation']:.3f}$")
    result["latex_row"] = records[0]["method"] + " & " + " & ".join(cells) + r" \\"
    if Path(output).exists():
        raise FileExistsError(output)
    atomic_json(output, result)
    return result
