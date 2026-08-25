# 9. Reference

## 9.1 The doors

| command | what it does | needs |
| --- | --- | --- |
| `gpuwm-hex version` | report the installed distribution, version, package path | nothing |
| `gpuwm-hex doctor [--explain] [--json]` | report every estate this install can reach; exit 1 while a required one is missing | nothing |
| `gpuwm-hex mesh-check --grid G --static S` | validate a mesh pair; print dimensions and SHA-256 digests | the pair |
| `gpuwm-hex oracle-gate --grid G --static S --fixtures DIR` | replay the source-extracted Fortran M1 fixtures against a mesh | a source checkout's oracle fixtures |
| `gpuwm-hex init ...` | build initial conditions (chapter 5) | `rw_mpas_init`, met file, mesh pair, capsule |
| `gpuwm-hex forecast ...` | run the model on a registered mesh (chapter 6); `--preflight` answers "will it fit?" without integrating | a CUDA device with room for the mesh, a gpuwm-hex checkout, a `gpuwm` source checkout at the pinned commit, mesh pair, init |
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
`tools/device_memory_ledger/hex_ledger_probe.py` (measures a card's own
footprint row for `--device-fixed-mib` / `--device-bytes-per-cell`).
Obs-referee: `tools/run_obs_referee.py` with manifests under
`verification/manifests/` ([`docs/obs-referee.md`](../obs-referee.md)).

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
declaration), and the admitted `dt_seconds` under the versioned Courant
policy of `src/mpas_port/timestep_admission.py`. Adding a mesh is adding a
row — data, not a code path. Chapter 4.1 tabulates the current entries and
their forecast status.

## 9.4 Where receipts live

| receipt | written by | carries |
| --- | --- | --- |
| `<init>.provenance.json` | init door | SHA-256 of every input, engine binary, argv, engine receipt, output |
| `render-manifest.json` | render door | engine digests, weights/output digests, per-frame product results, exact invocations |
| `cuda-v841-forecast-receipt.json` | forecast driver | claim, nonclaims, dropped guarantees, source/authority digests, per-step records, `gf_declared_divergence` |
| `--receipt-json` bind receipt | registered-mesh runner | the mesh binding: names, digests, admitted dt, fingerprints |
| `demo.receipt.json` / static receipt | `gpuwm mesh` | generation parameters, gates applied, output digests |

Measurement campaigns are committed under `evidence/`; the ones this manual
cites most: `evidence/gf-pin-move-measured-20260824/` (device footprint
model and per-mesh peaks), `evidence/local-timestep/` (LTS gates, speed and
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

- Distribution and command: `gpuwm-hex`. Import namespace: `mpas_port` —
  declared inconsistency, scheduled against a re-proof, scripts should
  expect it to change (README, *The import namespace*).
- Engine floor: `gpuwm>=2.5.5` is what pip enforces, and it carries two
  reasons at once — the seam bytes the Arwen manifest pins (three of the
  sixteen still differ at the published 2.5.4 stamp; 2.5.5 is the first
  published version that matches), and the bundle rows that put the four
  MPAS bridge binaries (`rw_mpas_init`, `rw_mpas_convert`, `rw_mpas_mesh`,
  `rw_mpas_static`) within reach of `gpuwm fetch-bridges`. Published 2.5.2
  carries none of the four, which would strand both doors. The forecast lane's real wall is stricter still:
  the sixteen-file seam manifest and a source checkout at the pinned commit
  (chapter 6.1).
- Licence: Apache-2.0, with the MPAS-Atmosphere BSD-3-Clause notice
  travelling in `NOTICE`. This is not the version available from LANS and
  UCAR, and neither they nor their contributors endorse it.
