# Data preparation

## Public sources

The product domains are Video Games, Office Products, and Cell Phones and
Accessories from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
The official site provides reviews, metadata, and standard benchmark processing.
The [SpecGR repository](https://github.com/Jamesding000/SpecGR) provides its
processing entry point and compatible mappings/text embeddings.

The Amazon importer expects already filtered, globally 5-core timestamp splits.
It imports those splits; it does not recompute global 5-core filtering.

## Amazon import

For each domain, supply:

```text
source/
  Video_Games.data_maps
  Video_Games.sent_emb
timestamp_w_his/
  Video_Games.train.csv
  Video_Games.valid.csv
  Video_Games.test.csv
```

Replace `Video_Games` with `Office_Products` or
`Cell_Phones_and_Accessories` for the other domains.

`DOMAIN.data_maps` is JSON containing `item2id` and `user2id`. Item IDs start
with padding ID 0, are contiguous, and are ordered by first registration time.
`DOMAIN.sent_emb` is a headerless float32 matrix, with one row for every item ID
including an all-zero padding row. Its width is inferred from file size.
The default adaptor expects 768-dimensional features.

CSV columns required by the importer are:

| Column | Meaning |
| --- | --- |
| `user_id` | Original user identifier |
| `parent_asin` | Original product identifier |
| `timestamp` | Integer interaction timestamp |

Additional CSV columns are ignored. In particular, imported `history` strings
are not used: histories are rebuilt from interaction times.

```bash
graft prepare --dataset Office --config configs/office.json \
  --source /path/to/Office_Products \
  --benchmarks /path/to/timestamp_w_his
```

Input files are read without modification. Choose an empty output directory.
`--workspace /path/to/workspace` changes the root for output paths, and
`--output /path/to/data/Office_Products` can override the prepared-data location.
If output is overridden, set the configuration's `data` path accordingly.

## Query construction

For a query at time t, the history contains only interactions with timestamps
strictly less than t. All interactions in an equal-timestamp group are queried
before that group is committed to history. The history keeps the most recent
20 interactions by default. Queries with empty history are removed.
Train, validation, and test time ranges must not overlap.

Equal-time items are ordered deterministically by a SHA-256-derived key. The
same key orders equal-time registrations. Registration time is the first
observed interaction time; an item is available at a query if its registration
time is less than or equal to the query time.

Training exposure is the union of all non-padding training targets and training
history items after query construction. A New-item target is absent from this
union. This cohort differs from the subset of items registered after training:
a catalog item can exist at the training cutoff without appearing in a
training query. It also differs from the new-leaf cohort because new items can
reuse an existing leaf.

## Prepared format

```text
data/Video_Games/
  p2_train.npz
  p2_valid.npz
  p2_test.npz
  catalog.npz
  text_vectors.npy
  item_mapping.json
  audit.json
```

Each split NPZ contains `user`, `item`, `timestamp`, `history`, and `source_row`.
Histories are left-padded integer matrices. The loader converts them to
right-padded sequences for the history encoder while preserving event order.

The data-level `catalog.npz` contains first timestamps, the training-exposure
mask, deterministic registration keys, and training-registered IDs. It is
distinct from the learned path catalogs under `artifacts/`.

`audit.json` records split row counts, New-item query counts, timestamp cutoffs,
history rules, feature width, and Amazon source fingerprints. Use these counts
as the denominators for evaluation.

## Custom data

Supply an interaction CSV with columns `user_id,item_id,timestamp` and a feature
NPZ with:

- `ids`: one-dimensional Unicode/string item IDs; no pickled object arrays.
- `vectors`: finite float32 content features with one row per `ids` entry.

```bash
graft prepare-events --dataset Custom --config configs/custom.json \
  --events /path/to/events.csv --features /path/to/features.npz \
  --train-cutoff 1609459200000 --valid-cutoff 1640995200000
```

Set `dataset`, `domain`, `encoder.adaptor_layers`, and `codebook_sizes` in the
custom configuration. `workspace` is relative to the configuration file's
directory unless overridden on the command line. At least three chronological
phases with nonempty histories are required. The demo's generated `demo.json`
is a complete small example.

The optional text encoder accepts metadata JSONL:

```bash
python -m pip install -e ".[text]"
python scripts/encode_text.py --metadata /path/to/meta.jsonl \
  --output /path/to/features.npz --device cuda:0
```

By default it joins `title`, `features`, and `description`, identifies products
by `parent_asin`, and uses
[sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base).
Use `--id-field item_id --fields text` for custom text records.
The companion metadata file records the model revision, fields, and input hash.

