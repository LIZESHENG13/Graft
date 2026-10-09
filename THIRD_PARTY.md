# Third-party attribution

## UniSRec and RecBole

The content-only backbone in `graft/backbone/` adapts the UniSRec content adaptor
and the SASRec sequence architecture. Module names are preserved for checkpoint
compatibility. The Transformer layers are imported from RecBole.

- [UniSRec source](https://github.com/RUCAIBox/UniSRec):
  copyright 2022 RUCAIBox, MIT. Original license:
  [licenses/UniSRec-MIT.txt](licenses/UniSRec-MIT.txt).
- [RecBole source](https://github.com/RUCAIBox/RecBole):
  copyright 2020 RUCAIBox, MIT. Original license:
  [licenses/RecBole-MIT.txt](licenses/RecBole-MIT.txt).

Please cite *Towards Universal Sequence Representation Learning for Recommender
Systems* (Hou et al., 2022) when using the content backbone, and the relevant
RecBole paper when using its components.

## Dataset interoperability and baselines

The Amazon importer reads the mapping and binary feature formats used by the
[official SpecGR repository](https://github.com/Jamesding000/SpecGR). The importer
and strict-past history reconstruction are implemented in this repository.
SpecGR model implementations and baseline training repositories are external
dependencies of comparative experiments and are not redistributed here.

Amazon Reviews 2023 data and pretrained text model weights retain their
respective upstream terms. Obtain them from the sources in [docs/DATA.md](docs/DATA.md).

## Runtime dependencies

PyTorch, NumPy, pandas, scikit-learn, threadpoolctl, and optional
sentence-transformers are installed as dependencies. Their licenses remain with
their upstream distributions.

