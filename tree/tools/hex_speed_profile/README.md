# Forecast cost instrumentation

These tools call the real `gpuwm-hex forecast` CLI in process. They do not
edit model or pinned physics source files, bypass admission, change a
numerical argument, or supply a replacement numerical implementation.

## Run and export

Run from the source tree on the GPU host with the same admitted forecast
arguments in both commands. Use separate, absent output and scratch paths.
Input files, forecast settings, device and source revision must match.

```sh
python tools/hex_speed_profile/profile_forecast_step.py \
  --baseline --report baseline-profile -- <forecast arguments>

nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=forecast-trace \
  python tools/hex_speed_profile/profile_forecast_step.py \
  --report instrumented-profile --nsys-window 3:12 -- <forecast arguments>

nsys export --type=sqlite --output=forecast-trace.sqlite forecast-trace.nsys-rep
python tools/hex_speed_profile/analyze_nsys_profile.py forecast-trace.sqlite \
  --points <cells-times-levels> --out trace-analysis.json --csv ranked-kernels.csv
python tools/hex_speed_profile/summarize_forecast_profile.py \
  instrumented-profile/profile-steps.json --warmup 2 --out host-regions.json
python tools/hex_speed_profile/selftest.py
```

Use the engine's plain step log and loop timing for throughput. The
baseline wrapper retains stack and composite-result references, then
hashes them after the CLI returns. It adds no timing wrappers, copy census,
device events, trace ranges or per-step synchronization.

The instrumented run wraps existing phase calls, records host wall and
NVTX, and counts `cupy.asnumpy` calls and bytes. It synchronizes once at
trace start and stop. `--events` adds device-event pairs around every
region and is off by default. `--cprofile-window FIRST:LAST` profiles a
separate host window; it must not overlap the Nsight window. All such
instrumentation perturbs timings. A profile is attribution evidence,
and its wall time is not a throughput improvement measurement.

Use node granularity for CUDA graphs. Graph granularity omits the node
activities, so a kernel-only union would misreport graph execution as idle.
The analyzer refuses such traces before producing per-kernel or idle results.

## Accounting

`--points` is the three-dimensional point count. Without it, seconds per
step are reported and point-normalized fields are null. All selected,
complete `step N` NVTX ranges enter the denominator, including slow
radiation or output steps. `--first` and `--last` select explicit steps.
Slow steps are not automatically classified as radiation.

The analyzer reports GPU busy as the union of every recorded kernel, copy
and memset interval overlapping each host step, with the remainder as
device idle. Work launched in a prior or unselected step still contributes
to busy time where it executes. Kernel costs and counts use launch-step
attribution, so they can differ from that step's actual busy time. It
attributes operations to the innermost NVTX range on the launching thread
through the CUDA runtime or driver correlation. It reports every CUDA API
name and host duration, copy bytes and device durations, launch API host
span, and kernel costs by name. Device idle is an observed timeline gap.
It is not itself a measured launch-overhead quantity.

Kernel summed durations, inclusive phase host wall, CUDA API host time
and GPU busy are different views of overlapping work. Do not add them.
Launch API host span uses an interval union so nested runtime and driver
launch calls do not double count within a step. The host summary supplies
exclusive phase times; only a complete disjoint host partition plus its
unattributed remainder can be summed to the step wall. Occupancy and
latency classifications are estimates without hardware counters.
Without `--events`, the host summary's `device_span_s=0` denotes an absent
device-event measurement. Use Nsight for actual device timing.

For a measured disjoint removable cost `C` in wall time `T`, the ideal
speedup is `T / (T - C)`. This is an upper bound, with all other costs held
fixed. Do not apply that formula independently to overlapping views or
present removing a required numerical operation as an admissible change.

## Identity and refusals

Every attempted run writes `final-identity.json`, even when the CLI
returns nonzero or raises after it has built a device stack. A refusal
before stack construction records `boundary_available: false`. Hash
errors are recorded and make the receipt ineligible for a complete-run
identity claim. Complete-run eligibility requires a written driver receipt
with the current schema, `status: passed`, explicit no-refusal evidence,
confirmed source and authority stability, and matching positive requested,
executed, scheduled and retained composite counts. The final atmosphere and
next-step forcing timestamps must match the requested endpoint. An absent
or malformed receipt, an absent final forcing carrier, a truncated run
that returns zero, or a nonzero return cannot establish complete identity.

After the run returns, the wrapper calls the existing
`fingerprint_execution_boundary`: it hashes the atmosphere state and saved
diagnostics, plus all array and scalar fields exported by the physics
backend's existing restart payload. It also hashes the full final
`rthdynten` and `rqvdynten` arrays and their timestamp when the committed
composite result provides them. Hashing occurs outside the engine step
timers and profiler window.

Compare these complete fingerprints and every forecast history file's
full SHA-256 between the plain and instrumented runs. Check the existing
source pin and contract deck receipts independently. Fingerprints cover
the existing execution and restart contract; they do not claim to hash
every temporary allocation, CUDA address, kernel-cache byte, or output
file. No successful numerical execution or native-dycore parity is
established by a CPU tooling selftest or an admission refusal.
