# Implementation guide

## Content sequence representation

`backbone/model.py` provides the content-only UniSRec history encoder.
The adaptor maps precomputed text features to raw item vectors.
Histories use these raw vectors; retrieval uses normalized item vectors and
normalized history-encoder output. The backbone is frozen before codebook
construction and scorer training.

## Base addresses

`quantization.py` fits three residual KMeans codebooks using only
training-registered item vectors. A complete address is `(c1,c2,c3)`, with
prefix `(c1,c2)`. Addresses can contain multiple items.

`catalog.py` stores the codebooks, item addresses, item registration times,
address birth times, training-exposure mask, and admission trace. Sparse
membership structures avoid allocating the whole three-dimensional address
space for retrieval.

`descriptors.py` constructs normalized training-item means for existing
first-level nodes, prefixes, and leaves. New leaves use the normalized content
of their first admitted item. Subsequent members do not update that descriptor.

## Admission

`registration.py` processes new items by first timestamp and deterministic tie
order. It routes content to an existing first-level node and one of that node's
existing second-level children. Under this prefix it selects the unused third
codeword that minimizes residual reconstruction error.

A new address is created while the global additional-leaf capacity and the
prefix's free addresses permit it. Otherwise the item joins the closest
existing leaf under that prefix. The new item cannot introduce a new prefix or
modify any codeword. Every assigned item address remains fixed.

Third-level codewords determine address allocation through reconstruction
error. Retrieval scores the complete leaf through its prefix and content
descriptor; it does not learn a separate classifier weight for the third
codeword.

## Shared scoring and hierarchical retrieval

`scorer.py` implements a shared residual MLP conditioned on the user query,
prefix descriptor, their elementwise product, and a leaf-level embedding.
Its residual output is added to the user query and normalized.

`retrieval.py` performs hierarchical prefix beam search, then scores leaves by
the weighted prefix log-score, conditional query/content matching, and the
additional-leaf bias. All leaves share the same MLP parameters.

Selected leaves supply the item candidates. The final item score adds the
within-leaf log-softmax of the original query's content similarity to the
leaf score. The normalizer covers the leaf's currently available members.
Birth-time masks are applied during search and item resolution, so a future
admission cannot change an earlier query's recommendation.

## Training

`training.py` provides initial path supervision and target-content substitution.
The auxiliary task replaces the positive leaf descriptor with the target item
vector while retaining its prefix and sampled competing leaves.

`training_episodes.py` builds support/arrival windows entirely from training
timestamps. Support items retain their addresses; later training items are
registered through the deployment admission rule.

`refinement.py` optimizes the shared scorer against same-prefix and routed
competitors. It combines episode path supervision, a frozen teacher score
distribution, and a query anchor. The configured main runs supervise reachable
new leaves, keeping the prefix selector and content backbone fixed.

Training and registration are separate operations. After scorer training,
deployment adds catalog entries and behavior history without updating model
parameters.

## Public entry points

Use `python -m graft --help` for the command list. The corresponding Python
functions are:

| Operation | Entry point |
| --- | --- |
| Amazon preparation | `preprocessing.prepare_amazon` |
| Custom preparation | `preprocessing.prepare_events` |
| Content training | `train_encoder.train_encoder` |
| Frozen encoding | `encoder.cache_features` |
| Base catalog | `quantization.build_base_catalog` |
| New-item admission | `registration.register_arrivals` |
| Prefix-conditioned scoring | `scorer.PrefixConditioner` |
| Recommendation | `evaluation.GraftRetriever.recommend` |
| Evaluation | `evaluation.run_evaluation` |
| Full experiment | `workflows.pipeline` |
| Component controls | `workflows.ablate` |

The tests cover frozen codebooks and old addresses, prefix support, future-item
visibility, strict-past histories, negative mining, gradient scope, frozen
teachers, uniform baseline evaluation, and seed summaries.

