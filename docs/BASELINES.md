# Comparing external baselines

All methods must share prepared item IDs, the query rows, strict-past user
histories, and the catalog available at each query timestamp.
The uniform evaluation adapter is described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#external-baselines).

Use the upstream implementation and configuration of each model. Relevant
source projects include:

| Model or family | Upstream source |
| --- | --- |
| SASRec and feature-based sequence models | [RecBole](https://github.com/RUCAIBox/RecBole) |
| UniSRec | [UniSRec](https://github.com/RUCAIBox/UniSRec) |
| TIGER, SpecGR-Aux, SpecGR++ | [SpecGR](https://github.com/Jamesding000/SpecGR) |

For other paper baselines, use the authors' released implementation and export
the same prediction NPZ format. Training repositories are kept separate from
Graft so their configurations and dependencies can be preserved.

An example export after converting predictions to Graft item IDs:

~~~python
import numpy as np

# predicted_item_ids: [N, 50], matching p2_test.npz query order.
np.savez_compressed(
    "baseline_test.npz",
    items=np.asarray(predicted_item_ids, dtype=np.int32),
    query_index=np.arange(len(predicted_item_ids)),
)
~~~

The adapter checks query indices, row count, item bounds, per-row duplicates,
and timestamp availability before scoring. Each baseline evaluation records
the query and prediction fingerprints.
