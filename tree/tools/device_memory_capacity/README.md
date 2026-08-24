# Device-memory capacity protocol

These tools freeze the measurement protocol for the 12 GiB capacity
work. They do not change model arithmetic and they do not turn an extrapolation
into a hardware result.

## Receipts

Use one directory per isolated run. Keep the command, source commit, device
identity, process-memory trace, in-process phase snapshots, model receipt,
state hash, and timing output together. A run is not comparable when its source
commit, CuPy/CUDA build, device, mesh, physics configuration, allocator policy,
or launch order differs without being recorded.

The primary whole-process number is the command process tree's own
`nvidia-smi` rows. `cudaMemGetInfo` appears only as a labelled device-wide
cross-check because another process can move it.

## Instrument controls

Run these before believing a before/after ledger:

```bash
python tools/device_memory_capacity/runtime_ledger.py identity \
  --output evidence/capacity/device.json

python tools/device_memory_capacity/runtime_ledger.py selftest-allocation \
  --allocation-mib 512 \
  --output evidence/capacity/allocation-view-control.json

python tools/device_memory_capacity/kernel_reservation_probe.py matrix \
  --local-floats 2048 \
  --output-dir evidence/capacity/kernel-controls

python tools/device_memory_capacity/sibling_contamination_probe.py control \
  --allocation-mib 512 \
  --output evidence/capacity/sibling-control.json
```

The allocation/view control must show one allocation and no allocation for its
view. The kernel matrix must show zero local bytes for the negative control, a
non-zero `local_size_bytes` and process reservation for the positive control,
and no material second reservation after the context is already sized. The
sibling control must show device-wide free memory moving while the parent's
per-process row stays stable.

## Whole-process wrapper

Wrap each baseline and candidate command:

```bash
python tools/device_memory_capacity/process_memory_probe.py \
  --output evidence/capacity/candidate-x1-run1/process.json \
  --stdout evidence/capacity/candidate-x1-run1/stdout.log \
  --stderr evidence/capacity/candidate-x1-run1/stderr.log \
  --interval-ms 50 \
  -- \
  python tools/run_cuda_v841_full_physics_x4.py <runner arguments>
```

The sampled process peak is an anchor. Add calls to
`runtime_ledger.cuda_snapshot()` at admitted phase boundaries to capture CuPy
pool live/total/free blocks and an explicit array inventory. Synchronize the
intended stream immediately before each snapshot; do not insert synchronizes
inside kernels or change launch order merely for instrumentation.

For every resident or transient array, record:

- canonical owner and any aliases;
- dtype, shape, strides, bytes, allocation pointer, and data pointer;
- lifetime/phase and read/write access;
- allocation and release call site.

Use `ArrayContract` at every borrow boundary, `LeaseToken` to invalidate stale
workspace views after rebinding, `assert_lifetime_contracts()` for every arena
or alias, and `assert_parking_safe()` before moving a canonical owner to host.
Concurrent read-only aliases are accepted. A concurrent writable overlap, wrong
shape/dtype/layout, stale generation, or live shared allocation refuses by name.

## Static accounting

The copy-elision patch removes known allocation events. Compute their geometric
sizes without mislabelling them as a measured peak:

```bash
python tools/device_memory_capacity/copy_elision_accounting.py \
  --cells 163842 --edges 491520 --vertical-levels 55 \
  --output evidence/capacity/copy-elision-static-x4.json
```

The sum in that file must not be subtracted from `nvidia-smi` peak. Allocation
events repeat, overlap, and interact with pool retention.

## Fit and gates

After at least two mesh widths and dual runs at each required gate, create a
`gpuwm-hex.capacity-samples.v1` JSON file. Each label must be unique. A complete
entry has this shape (nullable measurements may be omitted until they exist):

```json
{
  "schema": "gpuwm-hex.capacity-samples.v1",
  "samples": [
    {
      "variant": "candidate",
      "label": "x4-candidate-a",
      "cells": 163842,
      "process_peak_bytes": 123456789,
      "success": true,
      "state_sha256": "<64 lowercase hex characters>",
      "seconds_per_step": 1.0,
      "forecast_seconds_per_wall_second": 120.0,
      "pool_live_peak_bytes": 1,
      "pool_total_peak_bytes": 1,
      "physical_device_total_bytes": 12884901888,
      "effective_device_limit_bytes": null,
      "whole_device_limit_enforced": false,
      "limit_includes_non_pool": false,
      "limit_includes_local_backing_store": false,
      "isolated_card": true,
      "source_commit": "<40 hex characters>",
      "command": ["python", "tools/run_cuda_v841_full_physics_x4.py"]
    }
  ]
}
```

For a physical <=12 GiB card, record `physical_device_total_bytes`. For an
enforced-limit experiment on a larger card, record the effective byte limit
and set all three limit booleans only after the separate controls prove them.
Then run:

```bash
python tools/device_memory_capacity/capacity_model.py \
  --input evidence/capacity/samples.json \
  --output evidence/capacity/report.json \
  --target-cells 163842 \
  --budgets-gib 12 16 24 32 \
  --headroom-mib 512
```

The report labels affine results `PROJECTION ONLY`. The 12 GiB gate can become
`PASS` only from at least two byte-identical target-cell runs on a physical
12 GiB device, or a validated whole-device limit that includes pool, non-pool,
and CUDA local-memory backing-store allocations. The qualifying process peak
must also leave the declared headroom (512 MiB by default).
`CUPY_GPU_MEMORY_LIMIT` is a pool limit and is not sufficient for that gate.
