# Optional sibling GF-subsidence hook contract

This file is an integration contract, not a guessed patch against the hidden
`gpuwm` sibling.

## Required behavior

The sibling may expose one optional configuration value, disabled by default.
When disabled, the exact existing GF code path must execute with no arithmetic,
launch-order, allocation, serialization, or output change. The production
identity test compares complete selected outputs against an unmodified build.

When enabled, the treatment may alter **only the GF subsidence tendency**. It
must not change convection triggering, radiation, microphysics, PBL, surface
fluxes, or unrelated GF terms. The sibling writes one canonical JSON receipt per
case:

```json
{
  "schema": "gpuwm.gf-subsidence-treatment/v1",
  "treatment_name": "gf-subsidence-scale",
  "mode": "multiply_tendency",
  "value": 0.75,
  "enabled": true,
  "scope": "gf_subsidence_only",
  "call_count": 123,
  "columns_touched": 456789,
  "pre_tendency_sha256": "<64 lowercase hex>",
  "post_tendency_sha256": "<64 lowercase hex>",
  "producer_commit": "<40 lowercase hex>",
  "metadata": {}
}
```

The hashes cover a canonically ordered byte representation of the **unscaled**
and **post-treatment** GF subsidence tendency arrays over the run. They are an
audit signal, not a substitute for model-output checksums.

## Disabled receipt

A disabled receipt, when emitted, must use:

```text
enabled = false
call_count = 0
columns_touched = 0
pre_tendency_sha256 == post_tendency_sha256
```

The stronger requirement is still full output-tree byte identity:

```bash
PYTHONPATH=src python tools/run_obs_referee.py compare-identity \
  /path/to/unmodified-default /path/to/hook-present-but-disabled \
  --include '*.nc' --include '*.json'
```

## Acceptance order

1. unmodified vs hook-disabled byte identity;
2. enabled receipt validation;
3. process diagnostics showing only the GF subsidence term changed;
4. all four observational cases;
5. paired MRMS/ASOS scorecard and guardrails;
6. explicit owner review.

No result from this lane alone switches the default.
