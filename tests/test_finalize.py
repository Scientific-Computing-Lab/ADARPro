import json

from adarpro.finalize import context_specification, frozen_input_inventory
from adarpro.quarantine import sha256_file


def test_context_specification_accepts_single_frozen_component(tmp_path):
    repository = tmp_path
    checkpoint = repository / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    summary = {
        "context": "Liver",
        "mode": "binary_pair_balanced",
        "checkpoint": "model.pth",
        "checkpoint_sha256": sha256_file(checkpoint),
        "threshold": 0.4,
        "canonical_test_access": False,
    }
    path = repository / "summary.json"
    path.write_text(json.dumps(summary))
    observed = context_specification(repository, path)
    assert observed["context"] == "Liver"
    assert observed["weights"] == [1.0]
    assert observed["components"][0]["mode"] == "binary_pair_balanced"


def test_frozen_input_inventory_hashes_and_sizes(monkeypatch, tmp_path):
    source = tmp_path / "input.txt"
    source.write_bytes(b"frozen")
    monkeypatch.setattr(
        "adarpro.finalize.FROZEN_INPUT_PATHS", {"example": "input.txt"}
    )
    observed = frozen_input_inventory(tmp_path)
    assert observed["example"] == {
        "path": "input.txt",
        "bytes": 6,
        "sha256": sha256_file(source),
    }
