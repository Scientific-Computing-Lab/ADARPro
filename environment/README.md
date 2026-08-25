# Environments

`../environment.yml` is the authoritative AdarEdit checkpoint environment.
The local environment is created at `environment/conda/` and is excluded from
version control. ADARpro-specific table and test dependencies are installed
from `pyproject.toml` after the official environment resolves; exact installed
versions are captured under `provenance/`.
