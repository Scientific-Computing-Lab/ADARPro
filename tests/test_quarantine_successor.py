import json

import pytest

from adarpro.quarantine import (
    CONTEXTS,
    QuarantineViolation,
    consume_final_authorization,
    freeze_final,
    initialize,
    initialize_successor,
    load_policy,
    sha256_file,
    verify_audit_chain,
)


def _repository(tmp_path):
    for context in CONTEXTS:
        folder = tmp_path / "data" / "human" / context
        folder.mkdir(parents=True)
        (folder / "test.jsonl").write_text(f"{context} rows\n")
        (folder / "test.metadata.csv").write_text("pair_id,yes_no\n")
    (tmp_path / "provenance").mkdir()
    return tmp_path


def _consumed_predecessor(repository, monkeypatch):
    monkeypatch.setattr("adarpro.quarantine.subprocess.check_output", lambda *a, **k: "abc\n")
    policy = repository / "provenance" / "test_quarantine.json"
    initialize(repository, policy)
    manifest = repository / "provenance" / "manifest.json"
    manifest.write_text("{}")
    token = freeze_final(policy, manifest)
    consume_final_authorization(policy, token)
    return policy


def test_successor_continues_the_audit_chain_and_can_be_frozen_and_consumed(tmp_path, monkeypatch):
    repository = _repository(tmp_path)
    predecessor = _consumed_predecessor(repository, monkeypatch)
    before = sha256_file(predecessor)
    successor_path = repository / "provenance" / "test_quarantine_v2.json"
    successor = initialize_successor(repository, successor_path, predecessor)
    assert successor["state"] == "development_sealed"
    assert successor["evaluation_number"] == 2
    assert successor["predecessor"]["policy_sha256"] == before
    audit = repository / "provenance" / "test_access_audit.jsonl"
    assert verify_audit_chain(audit) == successor["audit_chain_head"]
    manifest = repository / "provenance" / "manifest_v2.json"
    manifest.write_text('{"v": 2}')
    token = freeze_final(successor_path, manifest)
    consumed = consume_final_authorization(successor_path, token)
    assert consumed["state"] == "final_consumed"
    assert sha256_file(predecessor) == before  # the predecessor is never written
    events = [json.loads(line)["event"] for line in audit.read_text().splitlines()]
    assert events[-4:] == ["final_test_authorization_consumed", "successor_policy_initialized",
                           "final_pipeline_frozen", "final_test_authorization_consumed"]
    assert verify_audit_chain(audit) == load_policy(successor_path)["audit_chain_head"]


def test_successor_needs_a_consumed_predecessor(tmp_path, monkeypatch):
    monkeypatch.setattr("adarpro.quarantine.subprocess.check_output", lambda *a, **k: "abc\n")
    repository = _repository(tmp_path)
    predecessor = repository / "provenance" / "test_quarantine.json"
    initialize(repository, predecessor)
    with pytest.raises(QuarantineViolation, match="consumed"):
        initialize_successor(repository, repository / "provenance" / "v2.json", predecessor)


def test_only_one_successor_per_predecessor(tmp_path, monkeypatch):
    repository = _repository(tmp_path)
    predecessor = _consumed_predecessor(repository, monkeypatch)
    initialize_successor(repository, repository / "provenance" / "v2.json", predecessor)
    with pytest.raises(QuarantineViolation, match="moved past the predecessor"):
        initialize_successor(repository, repository / "provenance" / "v2b.json", predecessor)


def test_successor_refuses_changed_test_artifacts(tmp_path, monkeypatch):
    repository = _repository(tmp_path)
    predecessor = _consumed_predecessor(repository, monkeypatch)
    (repository / "data" / "human" / "Liver" / "test.jsonl").write_text("changed\n")
    with pytest.raises(QuarantineViolation, match="artifacts changed"):
        initialize_successor(repository, repository / "provenance" / "v2.json", predecessor)


def test_successor_must_sit_beside_its_predecessor(tmp_path, monkeypatch):
    repository = _repository(tmp_path)
    predecessor = _consumed_predecessor(repository, monkeypatch)
    (repository / "elsewhere").mkdir()
    with pytest.raises(QuarantineViolation, match="beside its predecessor"):
        initialize_successor(repository, repository / "elsewhere" / "v2.json", predecessor)


def test_a_tampered_audit_record_is_detected(tmp_path, monkeypatch):
    repository = _repository(tmp_path)
    _consumed_predecessor(repository, monkeypatch)
    audit = repository / "provenance" / "test_access_audit.jsonl"
    lines = audit.read_text().splitlines()
    record = json.loads(lines[0])
    record["event"] = "edited"
    audit.write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
    with pytest.raises(QuarantineViolation, match="does not match its hash"):
        verify_audit_chain(audit)
