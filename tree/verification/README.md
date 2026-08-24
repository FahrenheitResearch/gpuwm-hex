# Verification assets

- `manifests/obs-referee-283.production.json` is the authoritative production
  scorecard manifest. It pins the repository base, known cases, pending controls,
  arm set, metrics, uncertainty, claim rules, and no-auto-promotion policy.
- `fixtures/build_synthetic_suite.py` creates a completely offline,
  byte-deterministic four-case test. It writes outside the repository unless
  explicitly pointed into it.
- `schemas/` documents the two receipt contracts for external producers/hooks.

The checked-in evidence under `tree/evidence/obs-referee-283/` is intentionally
`NOT_MEASURED`. It contains no real metric values. Replace it only with a
complete provenance-pinned run; never hand-edit values into it.
