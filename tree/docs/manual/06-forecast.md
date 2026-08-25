# 6. Running a forecast

## 6.1 The door

```sh
gpuwm-hex forecast \
  --mesh    x1.40962 \
  --grid    assets/x1.40962.grid.nc \
  --static  assets/x1.40962.static.nc \
  --init    work/x1.40962.init.nc \
  --init-source "GFS 2026-08-12 06Z" \
  --start-time 2026-08-12_06:00:00 \
  --hours 1.0 --history-every-minutes 30 \
  --out work/fc-01 \
  --gpuwm-checkout <gpuwm-checkout> \
  --case-label quickstart
```

The door resolves the mesh through the registry, cross-examines the
supplied files against that row's pinned bytes, admits the request against
device memory measured at that moment, runs the integration, and writes
history frames plus one run receipt into `--out`. On success its last line
is the render command for what it just produced.

**It needs the gpuwm-hex source checkout**, and says so by name if it does
not have one. The drivers live in `tree/tools/`, outside the wheel,
because they verify their own executing modules by SHA-256 before CUDA is
touched. Run the door from inside a checkout, or pass
`--repo <gpuwm-hex-checkout>/tree`.

**It needs a `gpuwm` source checkout too**, passed as `--gpuwm-checkout`.
The port pins the engine's physics seam by the SHA-256 of sixteen
individual gpuwm source files, one of which is a repository document that
no wheel places in site-packages — so an installed gpuwm satisfies pip and
does not satisfy the pin. The run verifies the checkout's git state and
the sixteen digests at launch and refuses any mismatch by name. No
published gpuwm version satisfies the manifest on its own (README, *The
engine pin*).

Beyond those two: a **registered mesh pair** (chapter 4), an **init** for
it (chapter 5), and **device memory for the mesh** — which the door now
decides for you rather than leaving you to discover mid-run (6.2).

Timing, for the shape of it: the 40,962-cell global mesh ran 1 h in 30
steps at dt=120 s, 97.8 s of integration and 146 s wall, on a 16 GiB
RTX 5070 Ti. Full physics throughout: WSM6 + Grell-Freitas + YSU +
YSU-GWDO + revised-MO + NoahMP + cloud fraction + legacy RRTMG. At the
other end, the 163,842-cell mesh has run 24 h in 720/720 steps at about
3.07 s/step on an RTX 5090 (README, *What is proven*).

Notable flags:

- `--init-source` — a provenance sentence recorded in the receipt. The
  init itself is the authority for the start time, and `--start-time` is
  an assertion against it.
- `--horiz-mixing` — default `2d_smagorinsky`, the native Registry
  default, ported and A/B-proven; `off` is a control lane, reported as the
  configuration native itself cannot integrate on convective cases.
- `--stop-on-refusal` — when the model refuses to publish a step, stop and
  write the receipt for the frames already committed instead of aborting
  with no receipt. No validation is relaxed.
- `--preflight` — 6.2.
- `--scratch` — the kernel cache. It defaults to a **sibling** of `--out`,
  never inside it, and an explicit `--scratch` inside `--out` is refused:
  a cache tree inside a directory you are about to hand to the renderer
  publishes temporaries as products.

Under the door, unchanged: `tools/run_cuda_v841_forecast.py` is the
arbitrary-case driver whose `execute_forecast` is the integration loop,
and `tools/mpas_mesh_binding.py` is the registry it binds through. Both
remain runnable directly, and
`tools/run_cuda_v841_forecast_mesh.py --selftest` still validates the mesh
binding in both directions — the correct bind must pass and a wrong name,
an unregistered name and a swapped file must each be refused by name.
Run for this manual: 5/5 PASS.

## 6.2 Will it fit? — the admission gate, and `--preflight`

**Before anything else, ask the card.** The door does, on every run:

```
ADMISSION mesh=x1.40962 cells=40,962 predicted=9,948.0 MiB headroom=512.0 MiB free=9,097.0 MiB of 10,239.5 MiB -> REFUSED
```

That is a real line from an RTX 3080 while this chapter was written. The
number it decides against is measured **at that moment**, from the driver,
because the memory your desktop is already holding is part of the answer —
never a budget carried from a previous run or a previous card. The
footprint it is compared to is the fitted row of chapter 4.6:
`6,296.5 MiB + 93,474 bytes per cell`.

**Why the door refuses rather than letting the driver try.** The driver
carries its own free-memory floor, scaled per mesh from the native row —
about 6.0 GiB on `x1.40962`. That is a floor, not a footprint. A card
between 6.0 GiB and 9,948 MiB passes it, loads the mesh, compiles the
kernels, and *then* dies inside a CuPy allocation part-way through the
integration, after burning the time it took to get there. That is the
concrete breakage this gate prevents.

The refusal names the shortfall and the fitted alternative:

```
device memory admission refused --mesh x1.40962: the fitted footprint for 40,962
cells is 9,948.0 MiB and the decision holds back 512.0 MiB, so it needs
10,460.0 MiB free; this device reports 9,097.0 MiB free of 10,239.5 MiB, short by
1,363.0 MiB. ... no registered mesh fits this card at this moment: the fixed term
alone is 6,296.5 MiB before a single cell is allocated ...
```

When a smaller registered mesh *does* fit the memory measured, the refusal
names it instead.

**On Windows, the number the door reads is not the number `nvidia-smi`
prints.** Measured in one process on the RTX 3080 above, at one moment: the
CUDA driver reported 9,097.0 MiB free where `nvidia-smi` reported
6,778 MiB — a 2,319 MiB disagreement. That is WDDM: the driver reports what
it believes it could *obtain*, including memory the OS could evict from
other clients, not what is unallocated right now. The door reads the CUDA
driver deliberately, because that is the allocator the run allocates
through — but it means an admission on Windows is optimistic relative to
`nvidia-smi`, and on a marginal card a run can be admitted that WDDM then
cannot deliver. If a run the door admitted dies in CuPy on Windows, this is
the first thing to check.

**The fixed term is a property of the card, and the row shipped was
measured on a 170-SM part.** Smaller parts previously measured carried
smaller fixed terms, so on a smaller card this prediction is an
over-estimate and the refusal it produces is conservative. The remedy for
that is to measure, not to widen the gate: run
`tools/device_memory_ledger/hex_ledger_probe.py` on the card and pass what
it reports as `--device-fixed-mib` and `--device-bytes-per-cell`. They are
one row and must be given together — half a row mixes this card's fixed
term with another card's slope, which is a footprint nothing ever
measured. `--headroom-mib` sets the margin the decision holds back
(default 512).

**The remedy is reachable only on an architecture the execution pin
admits.** The instrument runs the real driver, and the driver's first act
on the device is `require_cuda(min_compute=(12,0),
required_compute=(12,0))` — compute capability 12.0 exactly, because the
port's byte-pinned proofs are anchored to sm_120 compilation. On any other
architecture both the measurement and the run refuse by name, before a
single allocation. Measured on a 10 GiB RTX 3080 (sm_86, 2026-08-24,
`evidence/small-card-3080-20260824/`): with a measured row that admits,
the run stops at `cuda.compute_capability=8.6 is below required 12.0` —
so on a non-sm_120 card the binding constraint is the architecture pin,
not memory, and no `--device-fixed-mib` row changes that. One sharp edge
from the same session: `--preflight` is deliberately CUDA-free, so it
cannot see this gate and answers `preflight_passed` on a card the run then
refuses.

**`--preflight` answers the whole question.** It runs every check —
inputs, the mesh bind, the admission decision, and the driver's own
source-pin and host preparation — reports all of them, and exits 1 if any
would stop the run. It is the one mode where a missing input or an
unadmitted card is *reported* rather than raised, because a user asking
"will this run?" has often not built every input yet, and the card
question is the one no file fixes:

```
INPUT MISSING --init names a missing file: .../x1.40962.init.nc.  Build one with `gpuwm-hex init` ...
BIND mesh=x1.40962 rebound=True dt=120.0 s
ADMISSION mesh=x1.40962 cells=40,962 predicted=9,948.0 MiB headroom=512.0 MiB free=9,097.0 MiB of 10,239.5 MiB -> REFUSED
PREFLIGHT mesh=x1.40962 problems=2 status=preflight_refused
```

Preflight writes the same receipt a run does, with a `preflight_problems`
list. It touches no CUDA beyond the memory query.

## 6.3 What the door writes

`--out` receives, on a clean run:

- `cuda-history.<valid-time>.nc`, one per capture — the files the render
  door takes;
- `cuda-v841-forecast-receipt.json`, the driver's own receipt (6.4);
- `forecast-receipt.json`, the door's receipt: the resolved request, the
  admission decision with every number it was made from, the mesh-binding
  receipt, the driver's receipt embedded whole, the history files, and the
  exact `gpuwm-hex render` command for them.

The door prints that command as its last line, so the next step is a
paste rather than a lookup. Its form:

```
NEXT gpuwm-hex render --history <out>/cuda-history.<valid-time>.nc --mesh <grid> --out <out>/png --simulation-start <start>
```

## 6.4 What a run claims — read the receipt

Every run writes `cuda-v841-forecast-receipt.json`. Its `claim` names
exactly what happened (the mesh, the physics set, dt, the init). Its
`nonclaims` and `dropped_guarantees` are just as binding — among them:

- **no comparison against a native MPAS CPU run** is performed for these
  forecasts;
- **no checkpoint is written** and restart identity is not re-established
  per run (see 6.5);
- **forecast skill is not established** — these are engineering forecasts;
- the receipt carries `gf_native_parity_claim: false` beside
  `gf_declared_divergence`, the declaration of chapter 3's GF generation
  gap, and so does every history file written.

Determinism is a real guarantee: the same run twice produces byte-identical
output, which is also the memory-corruption screen on cards without ECC
(dual-run byte comparison). This manual's local-time-stepping check ran the
same case twice through two configurations that must coincide on a uniform
mesh and got byte-identical history, on hardware, in one session.

## 6.5 Restarting a run — what exists and what does not

**What is proven:** checkpoint/restart with bitwise-identical continuation
is a property of the port, established in the full-physics proof harness
(`tools/run_cuda_v841_full_physics_x4.py`): a host checkpoint is taken
mid-run in a fresh process, restored, and the restarted history file
matches the uninterrupted run's byte for byte. The checkpoint schema is
versioned (v3 — it additionally carries GF's per-step advective-forcing
pair, whose omission was found and closed by exactly this gate;
[`evidence/restart-step16-327/`], CHANGELOG). A pre-v3 checkpoint is
refused by name instead of resuming wrong.

**What does not exist in 0.1.0:** a user-facing resume flag on the
forecast driver. Engineering forecasts run uninterrupted and write no
checkpoint — that is a stated dropped guarantee in every receipt. If a
forecast stops (power, refusal, operator), you re-run it from its init;
determinism means the re-run reproduces the original trajectory exactly up
to where it stopped. Treat "restart" in this release as a proven property
of the engine and a capability of the proof harness, not a workflow
feature of the forecast lane.

## 6.6 Local time stepping — opt-in, with its measured cost

On a variable-resolution mesh most columns are far coarser than the finest
one, yet the acoustic sub-step is sized for the finest.
`--local-timestep` lets coarse columns take fewer, longer acoustic
sub-steps, chosen from the grid file's own `dcEdge`:

```sh
... tools/run_cuda_v841_forecast_mesh.py --repo <checkout>/tree --mesh x1.40962 \
  -- ... --local-timestep
```

`--local-timestep-rates` sets the ladder (default `1,3`) and
`--local-timestep-buffer-rings` the buffer width at class boundaries
(default 1).

**Off by default, deliberately.** Native MPAS-A v8.4.1 has no local time
stepping, so there is no byte-identical implementation and never can be:
turning it on takes a declared divergence from native; leaving it off gets
the pinned arithmetic unchanged. Two facts follow from the mesh, and both
were verified on hardware:

- On a **quasi-uniform** mesh every column lands in one class and the
  option is inert: the run above (x1.40962, `--local-timestep`) produced
  history **byte-identical** to the default run, re-verified for this
  manual on the proving node [`evidence/local-timestep/gate1-x1.40962-bit-identity.json`].
- On the published variable-resolution mesh **the option does not pay**:
  measured one model hour, x4.163842, dry lane, RTX 5070 Ti — 1.268 s/step
  default against 1.283 s/step with the option, 0.988x. The acoustic loop
  is 23.5 % of a model step and this mesh admits a 23.0 % acoustic saving,
  so the whole-step ceiling is 1.057x before bookkeeping, and the released
  `(1,3,6)` schedule caps the feature at 1.16x even in the
  every-column-coarse limit [`evidence/local-timestep/speed-qv-*.json`].

Conservation shape: class boundaries are refluxed (Berger–Colella on the
acoustic mass flux), so mass and passive vapour are conservative in the
flux and approximate at binary32 rounding — measured drifts 2.6e-11 (dry
mass) and 6.4e-10 (passive qv) over a model hour against a 2.0e-8 bound,
on the x4 mesh with 1,063 interface edges (README, *Limitations*). Runs
with the option on remain bit-reproducible run to run, so the dual-run
corruption screen still works.

The stronger per-class whole-step form was measured to a no-go on current
hardware (projected 1.254x on the one mesh big enough to profit, a
slowdown on small steep meshes) —
[`evidence/local-timestep/fullstep-projection.json`] and the README carry
the full measurement; `tools/probe_lts_fullstep_projection.py` is the
instrument to re-run if meshes an order of magnitude larger become real.

## 6.7 When a run stops early

Mid-run validation failures are refusals, not crashes (chapter 3.4). The
one seen in practice: a vertical-velocity divergence at levels 44–47 tripped
the 200 m/s validation bound and the case stopped rather than publish a bad
step. `--stop-on-refusal` converts that into a receipted stop with the
committed frames preserved. The generated-mesh step-0 refusal is a known
open defect (chapter 4.5). If a run refuses on *sources* — a pinned module
or seam file whose digest moved — nothing was integrated at all; restore
the pinned bytes rather than editing the pin.
