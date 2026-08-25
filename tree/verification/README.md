# Verification assets

- `manifests/obs-referee-283.production.json` is the authoritative production
  scorecard manifest. It pins the repository base, the four selected cases, the
  arm set, metrics, uncertainty, claim rules, and the no-auto-promotion policy.
  The two control cases it once carried as `pending` were selected on
  2026-08-24 against a measured screen of the observation archive; the screen's
  rule and its numbers are in each case's `metadata.selection_basis` and in
  `docs/obs-referee.md`. The manifest carries no result: a manifest that did
  could not be corrected after a run without invalidating the
  `manifest_sha256` every run receipt pins.
- `fixtures/build_synthetic_suite.py` creates a completely offline,
  byte-deterministic four-case test. It writes outside the repository unless
  explicitly pointed into it.
- `schemas/` documents the two receipt contracts for external producers/hooks.

The checked-in evidence under `evidence/obs-referee-283/` was intentionally
`NOT_MEASURED` — no real metric values at all — until 2026-08-25, when it was
replaced wholesale by the output of a complete provenance-pinned run. Its
`RECEIPT.md` names the chain and the SHA-256 of every instrument in it. Replace
it only the same way; never hand-edit values into it.
