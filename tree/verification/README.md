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
- `engine-verdicts/` is the output of `tools/measure_engine_verdicts.py`: one
  JSON per measurement of every published `gpuwm` against this port's
  sixteen-file seam manifest, with the wheel files hashed, the tag commit,
  the build-road verdict and the paths that moved.  `hexcore.engine_pin`'s
  table is spliced from the newest one and `tests/test_engine_pin.py`
  re-renders it and demands equality, so the JSON is the table's source
  rather than a copy of it.  `premeasure-273-20260913.json` is the 2.7 line
  measured against the manifest 0.2.3 shipped (every 2.7 engine fails it);
  `repin-273-20260913.json` is every engine measured against the 2.7.3
  bytes (2.7.3 alone passes).  Beside them, `seam-ab-*.json` are the outputs
  of `tools/measure_engine_seam_ab.py`: the engine's column-batch seam driven
  on fixed columns under two engines and compared field by field, which is
  how a re-pin says which physics number moved and by how much.  Their
  ``.npz`` inputs are not carried; the JSON records the SHA-256 of the columns
  that produced them and of the engine sources on each side.

The checked-in evidence under `evidence/obs-referee-283/` was intentionally
`NOT_MEASURED` — no real metric values at all — until 2026-08-25, when it was
replaced wholesale by the output of a complete provenance-pinned run. Its
`RECEIPT.md` names the chain and the SHA-256 of every instrument in it. Replace
it only the same way; never hand-edit values into it.
