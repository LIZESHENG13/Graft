# Reproducing an experiment

## Staged execution

After data preparation, the pipeline performs these stages:

| Stage | Command | Output within `artifacts/DOMAIN/sSEED/` |
| --- | --- | --- |
| Content backbone | `train-encoder` | `encoder/best.pt`, `encoder/last.pt` |
| Frozen encoding | `cache` | `cache/item_vectors.npy`, split query arrays |
| Residual codebooks | `quantize` | `base/train_catalog.npz` |
| Chronological admission | `register` | `catalog_valid/catalog.npz` |
| Training arrivals | `episodes` | Support catalogs, arrival catalogs, query indices |
| Optional warmup | `warmup` | `warmup/last.pt` |
| Scorer refinement | `train` | `scorer/step_1000.pt` |
| Test admission | `register --split test` | `catalog_test/catalog.npz` |
| Evaluation | `evaluate --split test` | `evaluation_test/result.json`, predictions |

In this table `DOMAIN` means the short alias (`Games`, `Office`, `Phones`).
Prepared data is shared across seeds; learned artifacts are separated by seed.
`pipeline --until STAGE` stops after a stage, and `--resume` reuses completed
stages. An unfinished encoder run can be resumed with `train-encoder --resume`;
other unfinished stage outputs should be preserved separately and the stage
rerun into an empty directory.

The first-stage encoder uses validation Recall or NDCG to select `best.pt`.
Scorer refinement exports the configured final step; its intermediate
validation measurements are diagnostics. The declared new-leaf bias is used
directly: the pipeline does not search bias values automatically.
Test queries are encoded and evaluated after model training.

For example:

```bash
graft train-encoder --dataset Games --config configs/games.json --device cuda:0
graft cache --dataset Games --config configs/games.json --device cuda:0
graft quantize --dataset Games --config configs/games.json
graft register --dataset Games --config configs/games.json --device cuda:0
graft episodes --dataset Games --config configs/games.json --device cuda:0
graft warmup --dataset Games --config configs/games.json --device cuda:0 --steps 500
graft train --dataset Games --config configs/games.json --device cuda:0
graft cache --dataset Games --config configs/games.json --device cuda:0 --splits test
graft register --dataset Games --config configs/games.json --device cuda:0 --split test
graft evaluate --dataset Games --config configs/games.json --device cuda:0 \
  --split test --output artifacts/Games/s2024/evaluation_test
```

If training is run with a nondefault `--steps`, use the resulting
`step_NNNN.pt` explicitly with `evaluate --checkpoint`, or update
`scorer_checkpoint` in the configuration.

## Configuration

| Setting | Games | Office | Phones |
| --- | ---: | ---: | ---: |
| Codewords per layer | 144 | 256 | 288 |
| Additional leaf capacity | 2,048 | 8,192 | 8,192 |
| Initial scorer warmup steps | 500 | 0 | 0 |
| Scorer refinement steps | 1,000 | 1,000 | 1,000 |
| New-leaf bias | 3 | 2 | 1.5 |
| Prefix budget | 200 | 200 | 200 |
| Leaf budget per query | 50 | 50 | 50 |
| Temperature | 0.07 | 0.07 | 0.07 |
| Prefix weight | 0.25 | 0.25 | 0.25 |

Capacity is an upper bound. The registered number of new leaves depends on the
arrival stream and unused addresses within each prefix. Test registration
replays the complete arrival stream from the base catalog with the same
cumulative capacity; previously assigned validation addresses are checked for
stability.

The content backbone defaults to hidden width 300, two Transformer layers, two
attention heads, eight adaptor experts, and maximum history length 20.
Training settings are configurable through the JSON file and saved with each
run.

## Metrics and diagnostics

Each query has one next-item target. If the target appears at rank r <= 50,
Recall@50 is 1 and NDCG@50 is `1 / log2(r + 1)`; otherwise both are 0.
Average over the exact query cohort. JSON values are fractions; multiplying by
100 yields the percentage format used by the LaTeX aggregate.

`overall` covers all queries. `cold` denotes the training-unexposed New-item
cohort. The evaluator additionally saves path survival and training-exposed
metrics for component analysis. Recommendations are restricted to items
available at the query timestamp, and every final item belongs to a selected
leaf.

The saved predictions include `query_index`, target `item`, ranked `items`,
`rank`, and per-level `stages`. Stage diagnostics distinguish:

1. Retention of the target's two-level prefix.
2. Retention of the target leaf conditional on prefix retention.
3. Final item hit conditional on leaf retention.

## Seed aggregation

Record seed IDs explicitly and run each into its own artifact directory.
`aggregate` computes means and standard deviations from the per-run JSON
files. It checks the dataset, split, query-file hash, cohort counts, search
configuration, and component-control flags.

By default SD uses the sample convention (`ddof=1`). Use `--ddof 0` to
request population SD and report that convention consistently.
The result keeps each seed's metric values, so a diagnostic table can identify
the exact contributing run.

## Component controls

`ablate` runs a matched set at bias 1: full Graft, bias 0, identity conditional
query, parent descriptors for newly admitted leaves, parent descriptors with
bias 0, and reuse-only admission with bias 0. Every variant retains the same
query cohort and frozen content backbone.

For the training objective control, train a separate scorer with the same seed,
warm start, and episode data:

```bash
graft train --dataset Games --config configs/games.json --device cuda:0 \
  --no-graft --output artifacts/Games/s2024/scorer_no_graft

graft ablate --dataset Games --config configs/games.json --device cuda:0 \
  --output artifacts/Games/s2024/ablations \
  --no-graft-checkpoint artifacts/Games/s2024/scorer_no_graft/step_1000.pt
```

`--no-graft` removes arrival-episode path cross-entropy while retaining the
teacher and anchor terms and the same training schedule. Initial warmup is
unchanged in this control.

Registration budgets can be varied with `register --budget N --output ...`.
Evaluate the corresponding snapshot with `evaluate --catalog ...`.
Bias can be varied independently using `evaluate --bias B --output ...`.
Use distinct output directories for each setting.

## External baselines

Run baseline training in its original repository. Export an NPZ with integer
`items` of shape `[number_of_queries, 50]` and
`query_index = arange(number_of_queries)` in this repository's split order.
Use the item IDs from `item_mapping.json`. ID 0 pads unused ranks; all other
entries must be distinct within each row and available at that query time.

```bash
graft evaluate-predictions --dataset Games --config configs/games.json \
  --split test --method SpecGR++ \
  --predictions /path/to/specgr_predictions.npz \
  --output artifacts/Games/specgr_test
```

This applies the same query denominator, training-exposure grouping, and item
metrics. External model implementations are not bundled.

## Runtime measurement

`benchmark` measures path retrieval and item resolution from cached normalized
queries. It reports synchronized single-query mean/p95 latency, batched
throughput, and FLOPs for matrix operations supported by the PyTorch profiler.
Text/history encoding and preprocessing are outside this measurement. Compare
methods with the same measured scope, query sample, device, and candidate
budgets. FLOPs are profiler-supported matrix FLOPs, rather than an all-operator
total.

