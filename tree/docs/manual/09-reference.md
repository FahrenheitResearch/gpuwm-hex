# 9. Reference

## 9.1 The doors

| command | what it does | needs |
| --- | --- | --- |
| `gpuwm-hex version` | report the installed distribution, version, package path | nothing |
| `gpuwm-hex doctor [--explain] [--json]` | report every estate this install can reach; exit 1 while a required one is missing | nothing |
| `gpuwm-hex mesh-check --grid G --static S` | validate a mesh pair; print dimensions and SHA-256 digests; a regional cull gains a `regional` receipt block; `--grid-only --grid G` validates a grid before its static exists | the pair (or the grid alone) |
| `gpuwm-hex oracle-gate --grid G --static S --fixtures DIR` | replay the source-extracted Fortran M1 fixtures against a mesh | a source checkout's oracle fixtures |
| `gpuwm-hex cull --parent-grid G --parent-static S --parent-init I --region R` | cut a limited-area grid, static and init out of a global case; `init` refuses a regional grid by name, so this is how a limited-area case gets an init at all | `rw_mpas_mesh`, the global triple, one region row |
| `gpuwm-hex init ...` | build initial conditions (chapter 5) | `rw_mpas_init`, met file, mesh pair, capsule |
| `gpuwm-hex forecast ...` | run the model on a registered mesh (chapter 6); `--preflight` answers "will it fit?" without integrating | a CUDA device with room for the mesh, a gpuwm-hex checkout, a `gpuwm` source checkout at the pinned commit, mesh pair, init |
| `gpuwm-hex swath {plan,metrics,explain}` | place fine grids from a coarse forecast's own fields, print the armed threat rows, explain why each candidate was taken or declined; `plan` prices every admitted swath through a real `rw_mpas_mesh --dry-run` unless `--no-size` | CPU only; a coarse forecast or its run receipt, `rw_mpas_mesh` to price |
| `gpuwm-hex cycle {plan,run}` | follow weather across cycles ([`docs/cycle-door.md`](../cycle-door.md)): `plan` says what each cycle would place and opens no device, `run` does cull → mid-window init → boundaries → forecast → render | `plan`: the parent case and `rw_mpas_mesh`. `run`: everything `forecast` needs |
| `gpuwm-hex render ...` | history → product PNGs (chapter 7) | `rw_mpas_convert`, `rw_wrfbatch` |
| `gpuwm mesh ...` | generate a grid + static pair (chapter 4; engine door) | `rw_mpas_mesh`, `rw_mpas_static`, WPS_GEOG |
| `gpuwm fetch-bridges` | stage the engine's published binary bundle into `~/.gpuwm/bridges` | network |
| `gpuwm fetch-geog --root DIR [--list]` | stage / inventory the WPS_GEOG archive | network, ~28 GiB unpacked |

Under the forecast door (same checkout, chapter 6):
`tools/run_cuda_v841_forecast_mesh.py` (registered-mesh runner, with
`--verify-only` and `--selftest`), `tools/run_cuda_v841_forecast.py` (the
arbitrary-case driver whose `execute_forecast` the door drives),
`tools/run_cuda_v841_full_physics_x4.py` (the sealed proof harness: native
comparison, checkpoint/restart proof),
`tools/device_memory_ledger/hex_ledger_probe.py` (the per-allocation
device-memory ledger a footprint row is fitted from).
Obs-referee: `tools/run_obs_referee.py` with manifests under
`verification/manifests/` ([`docs/obs-referee.md`](../obs-referee.md)).

**Device memory is not `fixed + slope × cells`.** The footprint model is
`core(card, configuration)` plus the Grell-Freitas workspace at
`min(cells, SMs × 4 × 64)` columns, plus the YSU workspace at
`min(cells, SMs × 16 × 32)`, plus `bytes_per_cell × cells`; the margin held
back is the card's own RRTMG shortwave workspace plus 11.2 MiB of
instrument convention, both named and measured. Ask it with
`hexcore.device_admission.model_for_card(card, configuration)` — the door
reads your card's multiprocessor count at the moment of the decision and
selects or derives its row, so the answer is your card's. Chapter 4.6 has
the shape, the knees and the inversion. `--device-fixed-mib` /
`--device-bytes-per-cell` survive as an escape hatch for a card whose own
ledger you have run; they are no longer the remedy of first resort, and
they are one row that must be given together.

`gpuwm-hex --help` and `<door> --help` are the authoritative flag lists;
chapter 5.3 tabulates the init switches, chapter 7.5 the render selection.

## 9.2 Engine resolution ladder

Every door resolves every engine the same way, best rung first; an explicit
flag or variable naming a missing file is a hard error, never a
fall-through.

| binary | order |
| --- | --- |
| `rw_mpas_init` | `--engine`, `$GPUWM_HEX_RW_MPAS_INIT`, `$RW_MPAS_INIT`, `$GPUWM_RW_MPAS_INIT`, gpuwm bridge directories, `PATH` |
| `rw_mpas_convert` | `--convert-exe`, `$GPUWM_HEX_RW_MPAS_CONVERT`, `$MPAS_PORT_RW_MPAS_CONVERT`, `$GPUWM_RW_MPAS_CONVERT`, gpuwm bridge directories, `PATH` |
| `rw_mpas_mesh` | `--mesh-exe` (`--engine` on `cull`), `$GPUWM_HEX_RW_MPAS_MESH`, `$RW_MPAS_MESH`, `$GPUWM_RW_MPAS_MESH`, gpuwm bridge directories, `PATH` |
| `rw_wrfbatch` | `--renderer-exe`, `$GPUWM_HEX_RW_WRFBATCH`, `$MPAS_PORT_RW_WRFBATCH`, `$GPUWM_RW_WRFBATCH`, gpuwm bridge directories, `PATH` |

The `GPUWM_HEX_*` spellings are preferred; the older spellings still work
and always will — a rename never invalidates an install line that already
works. "gpuwm bridge directories" means a gpuwm checkout's
`tools/rustwx/target/release`, `libexec/bridges` beside the installed
package, and `~/.gpuwm/bridges` (where `gpuwm fetch-bridges` stages).

Other variables: `GPUWM_HEX_NO_LOCAL_GPU=1` / `GPUWM_NO_LOCAL_GPU=1` ban
device contact; `GPUWM_WPS_GEOG` names the geog root for `gpuwm mesh`.

## 9.3 The mesh registry

`tools/mpas_mesh_binding.py` — one entry per runnable mesh: name, declared
`nCells`/`nEdges`, exact grid/static byte counts and SHA-256 digests, static
provenance, nominal spacing (compared FP32-bit-exactly against the static's
declaration), and the declared `dt_seconds`. Adding a mesh is adding a
row — data, not a code path. Chapter 4.1 tabulates the current entries and
their forecast status.

**A timestep passes two gates.** `src/hexcore/timestep_admission.py` is
the geometry gate: the versioned outer-step Courant policy, re-measured at
bind from the mesh's own complete `dcEdge` array and never from the nominal
spacing (4.1). `src/hexcore/dt_admission.py` is the evidence gate: an
earned-anchor registry keyed by CONFIGURATION — the timestep together with
the cumulus selection and the surface/PBL cadence — not by timestep alone,
and not per mesh digest, so a mesh finer than the one an anchor was earned
on inherits it. Seven rows stand across five timesteps: 120, 100, 75, 20
and 5 s with Grell-Freitas, and 20 and 5 s with convection off. Each names
its schedule receipt, its integration anchor (two byte-identical forecasts
on named hardware), and a measured `physics_health` verdict against a 120 s
control on the same card, mesh and init. Read that verdict: four of the
seven rows read `DIVERGES`, so an anchor certifies that a configuration
integrates finitely and deterministically at the cadences it names, and
nothing more. Only 120 s carries a native MPAS-A reference and only it can.
A configuration with no row is refused by name, with the roster and the
mint command. Rule of thumb for what a mesh wants: `dt ≈ 6 × dx(km)` — the
textbook 20 s at 3 km and 5 s at 750 m, which are exactly the timesteps
whose rows were earned on a 120 km mesh 35× and 140× below its own Courant
limit and diverge there.

## 9.4 Where receipts live

| receipt | written by | carries |
| --- | --- | --- |
| `<name>.cull.json` | cull door | the region row, the engine, one entry per cut file with its parent, its cell counts, its lineage and the engine's own `*.cull-receipt.json` |
| `<init>.provenance.json` | init door | SHA-256 of every input, engine binary, argv, engine receipt, output |
| `render-manifest.json` | render door | engine digests, weights/output digests, per-frame product results, exact invocations |
| `forecast-receipt.json` | forecast door | the resolved request, the admission decision with every number it was made from, the mesh-binding receipt, the driver's receipt whole, the history files, the render command for them |
| `cuda-v841-forecast-receipt.json` | forecast driver | claim, nonclaims, dropped guarantees, source/authority digests, per-step records, `gf_declared_divergence` |
| `swath-plan.json` / `threat-decision.json` | `swath plan` | the armed metric rows, every track and every drop with its reason, one mesh-spec and one cull-region row per admitted swath, priced or stamped `--no-size` |
| `cascade-receipt.json` | `cycle` | one block per cycle: what was admitted, what was declined or skipped and why, the churn (which slots moved and which stayed), and every slot's legs with their timings. `cycle-NN/` beside it holds that cycle's `swath-plan.json` and `swath-state.json` |
| `--receipt-json` bind receipt | registered-mesh runner | the mesh binding: names, digests, admitted dt, fingerprints |
| `demo.receipt.json` / static receipt | `gpuwm mesh` | generation parameters, gates applied, output digests |

Measurement campaigns are committed under `evidence/`; the ones this manual
cites most: `evidence/memory-shape-20260827/` (the device-footprint model of
record — the shape, its named margin, and seventeen measured peaks scored
against it; it supersedes the affine rows of
`evidence/gf-pin-move-measured-20260824/` and
`evidence/memory-row-refit-20260826/`), `evidence/nest-ratio-20260827/` (the
five concentric culls the limited-area core is an envelope over),
`evidence/dt-anchors-20260826/` and `evidence/convection-off-20260826/` (the
timestep anchors), `evidence/cycling-loop-20260827/` (two cycles of the
loop, end to end), `evidence/local-timestep/` (LTS gates, speed and
conservation), `evidence/restart-step16-327/` (restart bitwise identity),
`evidence/statics-330-unified-rebuild/` (unified static writer, generated
mesh status, native-free init status), `evidence/init-door/` and
`evidence/obs-referee-283/`.

## 9.5 The test battery

Three tiers, split by what a machine must own for each tier's result to mean anything
([`tools/battery/README.md`](../../tools/battery/README.md)):

```sh
PYTHONPATH=src python -m pytest tests -q -m "not gpu and not bigcard and not assets"   # tier 1: anywhere
PYTHONPATH=src python -m pytest tests -q -m assets    # tier 2: ~6.9 GiB byte-pinned authority files
PYTHONPATH=src python -m pytest tests -q -m bigcard   # tier 3: capacity preflight for the big-card gates
```

Tests that cannot run skip with the missing thing named. Anything touching
CuPy is auto-marked `gpu` by AST inspection, so it cannot leak into tier 1
by omission.

## 9.6 Names that matter

- Distribution and command: `gpuwm-hex`. Import namespace: `hexcore`,
  renamed from `mpas_port` at 0.2.0 and settled there (README, *The import
  namespace*). There is no alias shim.
- Engine range: `gpuwm>=2.6.1,<2.6.2` is what pip enforces — a bounded
  range, derived from a measured table in `hexcore.engine_pin` rather than
  typed. The floor is where it is for one reason and it is the strictest
  one: 2.6.1 is the only published engine whose bytes match the sixteen-file
  Arwen seam manifest, re-measured 2026-09-01 against every published 2.5.x
  and 2.6.x release. It clears, incidentally, the bundle rows that put the MPAS bridge
  binaries (`rw_mpas_init`, `rw_mpas_convert`, `rw_mpas_mesh`,
  `rw_mpas_static`, `rw_mpas_lbc`) within reach of `gpuwm fetch-bridges`;
  published 2.5.2 carries none of them and would strand every door that
  drives one. The CEILING is exclusive and sits at the first engine not
  measured usable, so an engine that has not been compared against the
  manifest is never resolved onto — including ones not published yet. The
  forecast lane's real wall is stricter still: the sixteen-file seam
  manifest, plus a *git* checkout at the pinned commit for receipt
  provenance (chapter 6.1). `gpuwm-hex doctor` compares the installed
  engine's bytes against the manifest and names the version it found; since
  2.5.8 all sixteen pinned paths resolve from the install, so the checkout
  is owed for the commit rather than for a file.
- Licence: Apache-2.0, with the MPAS-Atmosphere BSD-3-Clause notice
  travelling in `NOTICE`. This is not the version available from LANS and
  UCAR, and neither they nor their contributors endorse it.
