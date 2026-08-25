# Provenance

- `upstream.json`: repository, revision, checkout date, hardware/software, and
  hashes of canonical benchmark/checkpoint inputs.
- `test_quarantine.json`: state of the sealed canonical test partition.
- `test_access_audit.jsonl`: hash-chained authorized-access events.
- `unified_model_selection_policy.json`: unified architecture and staged rules
  frozen before its outcomes were known.
- `unified_development_decision.json`: hash-addressed development result and
  final ensemble-versus-unified decision.
- `final_pipeline_manifest.json`: immutable input/checkpoint/threshold manifest
  used for the one consumed canonical-test evaluation.
- External resources receive one JSON record containing URL/accession,
  retrieval date, version, genome build, checksum, license, and preprocessing.
