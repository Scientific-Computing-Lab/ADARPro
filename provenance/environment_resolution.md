# Environment resolution

The upstream `environment.yml` was resolved on 2026-08-21. The workstation's
Conda 23.1 classic solver entered an incompatibility search because the global
configuration enforces strict priority across `pytorch`, `bioconda`,
`conda-forge`, and `defaults`. Micromamba 2.9.0 resolved the unchanged YAML
with flexible channel priority.

The resulting checkpoint environment uses Python 3.10.20, PyTorch
2.7.0+cu126, PyTorch Geometric 2.6.1, NumPy 2.2.5, pandas 2.2.3, and
scikit-learn 1.6.1. The repository's documented H200 checkpoints used
PyTorch 2.7.0+cu128; the major/minor PyTorch and exact PyG versions match, and
checkpoint reproduction is verified numerically rather than assumed.

The upstream pip specification allowed packages from the user site to satisfy
transitive requirements during installation. ADARpro disables the user site
with `PYTHONNOUSERSITE=1` and explicitly installed the missing `psutil` and
`urllib3`; `pip check` then reported no broken requirements. ADARpro-only
packages (`pyarrow`, `pytest`, and `ruff`) and the editable local package were
installed after the canonical stack.

Hardware: NVIDIA GeForce RTX 3090 (24 GB), driver 535.309.01. PyTorch reports
CUDA runtime 12.6 and successfully enumerates the GPU.
