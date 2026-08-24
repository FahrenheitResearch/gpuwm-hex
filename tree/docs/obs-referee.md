# Observational referee

This subsystem answers a narrow question: **does a gpuwm-hex/ArWen arm produce
better observable weather than its reference arm?** It does not revive
native-MPAS parity as a product goal, and it does not use a historical model as
an oracle. MRMS and ASOS are the primary referees; historical native output and
an operational model may appear only as provenance-complete context arms.

The pinned integration base is:

```text
c14bc1045b0d4516f18004f13265d900ec12563b
```

## Scientific boundary

The implementation scores directly observable consequences:

- MRMS one-hour precipitation and reflectivity;
- ASOS temperature, dewpoint, wind speed, and surface pressure;
- continuous error/bias, threshold contingency statistics, fractions skill
  score, and deterministic connected-object displacement;
- paired case-block confidence intervals, resampling whole meteorological cases.

It deliberately does **not** claim that MRMS/ASOS can verify model theta above
level 45, cloud-water/rain-water partition, or the full GF causal chain. Those
claims remain `UNRESOLVED` until a secondary profile/process referee is supplied
(for example radiosondes or a provenance-pinned analysis profile).

## No second observation parser

Raw MRMS GRIB2 and raw METAR/ASOS parsing remain a rustwx boundary. The referee
accepts only two normalized, checksummed contracts:

- `gpuwm-hex.canonical-grid/v1` (`.npz`);
- `gpuwm-hex.canonical-stations/v1` (canonical JSON Lines).

Every bundle requires a
`gpuwm-hex.normalized-artifact-receipt/v1` receipt that records its producer,
producer version, artifact name, and SHA-256. A production manifest cannot
allow a producer whose name begins with `synthetic`.

A source can optionally declare `producer_command`. The command is an argv
array, never a shell string, and may use only these placeholders:

```text
{output} {receipt} {case_id} {arm_id}
```

This bridge lets the existing rustwx machinery materialize the canonical
artifact without coupling the referee to a guessed rustwx CLI. The producer
must create both the artifact and receipt; success without both is a refusal.

## Case and arm ownership

Dates and case identities live only in JSON manifests. The production manifest
contains the two existing divergence identities:

- `gfs-20260812-divergence`;
- `era5-20240521-divergence`.

The required independent and weak-convection controls are present as
`selection_status: pending` with null dates. This is intentional: code does not
invent weather cases. A pending case makes the scorecard `NOT_MEASURED`.

The production arm set is:

- `arwen-current`: pinned current behavior;
- `gf-subsidence-experiment`: optional isolated treatment;
- `native-history-context`: historical context, not a parity target;
- `operational-baseline`: optional external context.

## Optional GF treatment

gpuwm-hex does not own the upstream engine's GF tendency arithmetic, so this
patch does not guess a sibling file or function. Instead it enforces an external
receipt contract. An enabled experiment is scoreable only when its receipt
proves all of the following:

- treatment name and mode exactly match the manifest;
- scope is exactly `gf_subsidence_only`;
- the hook ran at least once and touched at least one column;
- pre/post tendency SHA-256 values differ;
- a full producer commit is recorded.

Disabled/default neutrality has two independent gates:

1. a disabled receipt must report zero calls, zero touched columns, and identical
   pre/post tendency hashes;
2. `compare-identity` hashes selected output trees and refuses any byte
   difference.

This establishes a safe integration seam without making the treatment
default-on. The scorecard always emits `DO_NOT_ENABLE`; explicit owner
acceptance is outside the automatic runner.

## Deterministic workflow

From the root of this distribution:

```bash
PYTHONPATH=src python tools/run_obs_referee.py validate \
  verification/manifests/obs-referee-283.production.json

PYTHONPATH=src python tools/run_obs_referee.py not-measured \
  verification/manifests/obs-referee-283.production.json \
  --output evidence/obs-referee-283 \
  --reason "Real MRMS/ASOS and model-arm artifacts have not been materialized."

# After canonical real artifacts and all four cases exist:
OBS_REFEREE_DATA_ROOT=/absolute/case-root \
PYTHONPATH=src python tools/run_obs_referee.py run \
  verification/manifests/obs-referee-283.production.json \
  --output /absolute/evidence/obs-referee-283-real
```

No result file contains wall-clock time, random UUIDs, temporary paths, or
platform-specific JSON ordering. The synthetic end-to-end test runs the same
suite twice and requires every output file to have the same SHA-256.

## Synthetic proof

The fixture script creates four tiny cases and all four arms, including an
enabled treatment receipt:

```bash
PYTHONPATH=src python verification/fixtures/build_synthetic_suite.py \
  --root /tmp/hex-obs-referee-synthetic --run
```

Its verdict is always `SYNTHETIC_ONLY`. It is a software test, not a skill
result and not evidence for changing model defaults.

## Refusal behavior

The runner refuses rather than silently degrading when:

- a required artifact or receipt is missing;
- a checksum, producer, schema, field shape, coordinate range, or monotonic time
  check fails;
- a production manifest admits synthetic producers;
- an enabled treatment receipt is absent or inconsistent;
- a raw/unknown adapter is requested;
- automatic default promotion is enabled;
- confidence claims have fewer paired cases than the manifest minimum.

An optional input can become an explicit per-metric `NOT_MEASURED` record.
Integrity errors are never downgraded to missing evidence.
