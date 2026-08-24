# 6. Running a forecast

## 6.1 What the forecast lane requires — plainly

There is no `gpuwm-hex forecast` console script in 0.1.0, and that is a
deliberate omission rather than an oversight. Running the model means
working from the **gpuwm-hex source checkout**, and it needs four things:

1. **The gpuwm-hex checkout** — the drivers live in `tree/tools/`, and the
   executing modules are SHA-256-pinned so the run proves its own sources
   before CUDA is touched.
2. **An installed `gpuwm`** (pip already brought it in) **plus a `gpuwm`
   source checkout at the pinned commit**, passed as `--arwen-checkout`.
   The port pins the engine's physics seam by the SHA-256 of sixteen
   individual gpuwm source files, one of which is a repository document
   that no wheel places in site-packages — so an installed gpuwm satisfies
   pip and does not satisfy the pin. The run verifies the checkout's git
   state and the sixteen digests at launch and refuses any mismatch by
   name. No published gpuwm version satisfies the manifest on its own
   (README, *The engine pin*).
3. **A registered mesh pair** (chapter 4) and **an init** for it
   (chapter 5).
4. **Device memory for the mesh** (chapter 4.6): `x1.40962` ran on a
   16 GiB card for this manual; `x4.163842` is a 32 GiB-card
   configuration.

## 6.2 The registered-mesh runner

`tools/run_cuda_v841_forecast_mesh.py` is the door for every registered
mesh: it binds the named mesh against the real files (shape, byte
digests, admitted timestep), refuses a wrong name, an unregistered mesh,
or a swapped file, and then hands everything after `--` to the forecast
driver. The quickstart's proven command:

```sh
cd <gpuwm-hex-checkout>/tree
PYTHONPATH=src python tools/run_cuda_v841_forecast_mesh.py \
  --repo <gpuwm-hex-checkout>/tree --mesh x1.40962 \
  --receipt-json work/fc-bind.json \
  -- \
  --grid   assets/x1.40962.grid.nc \
  --static assets/x1.40962.static.nc \
  --init   work/x1.40962.init.nc \
  --init-source "GFS 2026-08-12 06Z" \
  --start-time 2026-08-12_06:00:00 \
  --hours 1.0 --history-every-minutes 30 \
  --case-label quickstart \
  --arwen-checkout <gpuwm-checkout> \
  --cache-root work/cache --output work/out
```

That run: 30 steps at dt=120 s, full physics (WSM6 + Grell-Freitas + YSU +
YSU-GWDO + revised-MO + NoahMP + cloud fraction + legacy RRTMG), 97.8 s of
integration on an RTX 5070 Ti, three history files, exit 0. For scale at
the other end: the 163,842-cell mesh has run 24 h in 720/720 steps at about
3.07 s/step on an RTX 5090 (README, *What is proven*).

Notable flags:

- `--init-source` — a provenance sentence recorded in the receipt; the init
  itself is the authority for the start time, and `--start-time` is an
  assertion against it.
- `--horiz-mixing` — default `2d_smagorinsky`, the native Registry default,
  ported and A/B-proven; `off` is a control lane, reported as the
  configuration native itself cannot integrate on convective cases.
- `--stop-on-refusal` — when the port refuses to publish a step, stop and
  write the receipt for the frames already committed instead of aborting
  with no receipt. No validation is relaxed.
- `--preflight-only` — verify sources, authorities and host mappings
  without touching CUDA.

Instrument check: `--selftest` (before `--`) validates the mesh binding in
both directions — the correct bind must pass and a deliberately wrong
name, an unregistered name, and a swapped file must each be refused by
name. Run for this manual: 5/5 PASS.

## 6.3 What a run claims — read the receipt

Every run writes `cuda-v841-forecast-receipt.json`. Its `claim` names
exactly what happened (the mesh, the physics set, dt, the init). Its
`nonclaims` and `dropped_guarantees` are just as binding — among them:

- **no comparison against a native MPAS CPU run** is performed for these
  forecasts;
- **no checkpoint is written** and restart identity is not re-established
  per run (see 6.4);
- **forecast skill is not established** — these are engineering forecasts;
- the receipt carries `gf_native_parity_claim: false` beside
  `gf_declared_divergence`, the declaration of chapter 3's GF generation
  gap, and so does every history file written.

Determinism is a real guarantee: the same run twice produces byte-identical
output, which is also the memory-corruption screen on cards without ECC
(dual-run byte comparison). This manual's local-time-stepping check ran the
same case twice through two configurations that must coincide on a uniform
mesh and got byte-identical history, on hardware, in one session.

## 6.4 Restarting a run — what exists and what does not

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

## 6.5 Local time stepping — opt-in, with its measured cost

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

## 6.6 When a run stops early

Mid-run validation failures are refusals, not crashes (chapter 3.4). The
one seen in practice: a vertical-velocity divergence at levels 44–47 tripped
the 200 m/s validation bound and the case stopped rather than publish a bad
step. `--stop-on-refusal` converts that into a receipted stop with the
committed frames preserved. The generated-mesh step-0 refusal is a known
open defect (chapter 4.5). If a run refuses on *sources* — a pinned module
or seam file whose digest moved — nothing was integrated at all; restore
the pinned bytes rather than editing the pin.
