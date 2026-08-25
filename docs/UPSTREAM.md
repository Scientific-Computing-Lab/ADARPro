# Relationship to AdarEdit

ADARPro begins from the complete AdarEdit reproducibility release at:

- repository: <https://github.com/Scientific-Computing-Lab/AdarEdit>
- revision: `5c8b57d70cdc2eb1c5e168431013275bcc9ed0c5`

The canonical human data, model definitions, strict training entry point,
split verifier, result matrices/predictions, and supplied checkpoints required
to reproduce the reference are retained here. The complete older AdarEdit
analysis tree, its manuscript, and non-human benchmark data are not duplicated;
they are obtainable exactly from the revision above. Their paths, source
hashes, and exclusion reason remain in the release inventory.

This repository does not rewrite AdarEdit history. The frozen ADARPro manifest
records both the upstream revision and the later development code revision.
The published supplied-checkpoint benchmark remains the authoritative final
reference, while grouped OOF predictions serve only as a leakage-free internal
development comparator.
