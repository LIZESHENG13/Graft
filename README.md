# Graft

**New Items Need Grafting: Expanding Semantic Paths for Inductive Recommendation**

Graft admits new items into a semantic catalog after training. It appends complete
leaf paths under existing prefixes, describes each new leaf with item content,
and retrieves items through a shared prefix-conditioned scorer. Deployment keeps
the model parameters, base codebooks, existing prefixes, and registered item
addresses fixed.

This repository contains the content sequence backbone, residual quantization,
chronological registration, scorer training, hierarchical retrieval, evaluation,
ablations, and runtime measurement.

## Installation

Use Python 3.10 or 3.11. Install a PyTorch build appropriate for your CUDA runtime,
then install Graft from the repository root:

```bash
python -m pip install -e ".[encoder]"
python -m graft --help
```

The `encoder` extra supplies RecBole layers for content backbone training and
history encoding. For development:

```bash
python -m pip install -e ".[encoder,dev]"
python -m unittest discover -s tests -v
ruff check graft tests scripts
```

Both `graft` and `python -m graft` expose the same commands.

## A complete CPU example

```bash
graft demo --output /tmp/graft-demo
```

This creates a synthetic chronological dataset and runs content training,
query caching, quantization, leaf registration, training-period arrival episodes,
scorer refinement, and test evaluation. Results are written to:

```text
/tmp/graft-demo/artifacts/Demo/s2024/evaluation_test/result.json
```

## Games, Office, and Phones

The three JSON configurations use paths relative to this repository. Put prepared
inputs in `data/`, or import mappings, text embeddings, and the original
timestamp-split CSVs:

```bash
graft prepare --dataset Games --config configs/games.json \
  --source /path/to/Video_Games \
  --benchmarks /path/to/timestamp_w_his
```

Repeat with `Office` / `configs/office.json` and
`Phones` / `configs/phones.json`. See [data preparation](docs/DATA.md) for
the exact file formats, public data sources, and custom datasets.

Run a complete experiment:

```bash
graft pipeline --dataset Games --config configs/games.json \
  --seed 2024 --device cuda:0 --split test
```

Resume completed stages with `--resume`. Individual stages are also available:
`train-encoder`, `cache`, `quantize`, `register`, `episodes`, `warmup`, `train`,
and `evaluate`.

Configurations declare leaf capacity, scorer training, search budgets, and
new-leaf calibration. All three domains receive scorer refinement; Office and
Phones skip the optional initial warmup. Validation and test queries have
separate files and caches.

## Evaluation and multiple seeds

The evaluator reports Overall and New-item Recall@50 and NDCG@50. New items are
defined by absence from both training targets and training histories. JSON
metrics use fractions; the aggregation command formats both metrics as
percentages for LaTeX. Every evaluation saves its per-query predictions, query
cohort, seed, configuration, and input fingerprints.

```bash
python scripts/run_experiments.py --datasets Games Office Phones \
  --seeds 2024 2025 2026 --devices cuda:0 cuda:1 cuda:2

graft aggregate --results \
  artifacts/Games/s2024/evaluation_test/result.json \
  artifacts/Games/s2025/evaluation_test/result.json \
  artifacts/Games/s2026/evaluation_test/result.json \
  --output artifacts/Games/three_seeds.json
```

The default dispersion is sample standard deviation (`ddof=1`). The summary
retains each seed's scores and rejects mismatched query cohorts or search
settings.

## Further experiments

```bash
graft ablate --dataset Games --config configs/games.json \
  --device cuda:0 --split test --output artifacts/Games/s2024/ablations

graft benchmark --dataset Games --config configs/games.json \
  --device cuda:0 --split test --output artifacts/Games/s2024/runtime
```

External baseline predictions can be evaluated with the same protocol using
`evaluate-predictions`. Component controls, seed reporting, timing scope, and
baseline integration are described in [reproduction](docs/REPRODUCIBILITY.md).
The algorithm and module map are in [implementation](docs/ALGORITHM.md).

## Repository structure

```text
graft/
  backbone/             Content adaptor and sequence encoder
  preprocessing.py      Strict-past histories and training exposure
  train_encoder.py      Content backbone training
  encoder.py            Frozen item and history encoding
  quantization.py       Training-only residual codebooks
  catalog.py            Catalog snapshots and path memberships
  registration.py       Chronological leaf admission
  training_episodes.py  Training-period support/arrival episodes
  training.py           Initial path and target-content supervision
  refinement.py         Competition-aware scorer refinement
  scorer.py             Shared prefix-conditioned query network
  retrieval.py          Prefix/leaf search and item resolution
  evaluation.py         Item metrics and path diagnostics
  aggregation.py        Seed means and standard deviations
  workflows.py          Pipeline and component controls
configs/                Three domain configurations
scripts/                Experiment launcher and optional text encoding
docs/                   Data, protocol, and implementation documentation
tests/                  Catalog, chronology, training, and reporting tests
licenses/               Upstream license notices
```

## License and citation

Released under the [MIT license](LICENSE). UniSRec and RecBole attributions and
their original licenses are retained in [THIRD_PARTY.md](THIRD_PARTY.md).
Citation metadata is provided in [CITATION.cff](CITATION.cff).

