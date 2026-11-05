# TC climate diagnostics, experimental

This opt-in bundle adds six-hour resident TC detection and online association,
and daily GPI to the global ArWen forecast executor. It is not yet a validated
TC climatology or seasonal prediction system. The default is off.

The following plan command is executable without a GPU or downloads. It shows
five members, prescribed SST and the full May through November 2020 window.
The 120 km row is a planning demonstration only. The required 25 km global or
100-25 km VR observation proof has not run.

```sh
RECIPE=$(python -c 'from pathlib import Path; import hexcore.tc_climate as t; print(Path(t.__file__).with_name("recipes.json"))')
gpuwm-hex s2s --row global-120km --source gfs --cycle 2020-05-01T00 \
  --days 214 --members 5 --ocean-mode prescribed --history-hours 6 \
  --tc-climate "$RECIPE" --work-dir runs/tc-season-plan --dry-run
```

A prepared global forecast accepts the same recipe. Supply the registered mesh
row, its grid/static/init, and an existing native dated SST series. Run through
the site's GPU pool after measuring memory and setting a run-output cap.

```sh
GPUWM_HEX_SURFACE_SERIES="$SST_SERIES" gpuwm-hex forecast \
  --mesh "$MESH_ROW" --grid "$GRID" --static "$STATIC" --init "$INIT" \
  --hours 24 --history-every-minutes 360 --history-from-minutes 0 \
  --tc-climate "$RECIPE" --no-products --out runs/tc-day
```

The history files carry `tc_mslp` (Pa) at tracker boundaries, and
`tc_gpi`, `tc_potential_intensity` (m/s), and
`tc_pi_status` on daily boundaries. `missing_value=-9999` marks land or invalid
columns; status 0 means invalid/land, 1 valid, and 2 nonconverged. At completion,
`tc-climate.json` carries compact six-hour detections, qualified trajectories,
recipe parameters, candidate counts, host enqueue wall time and diagnostic stream-event times.
These event intervals are not the whole-run before/after step-time difference.
Each roster member owns an independent tracker and recipe. S2S recipe contents
are part of the plan signature. Admission includes an unmeasured diagnostic
memory allowance; it does not replace a measured peak for lease sizing.

Pressure minima are graph-local with deterministic ties. Closed contours use
spherical graph floods with a bounded hash set. The default recipe requires a
200 Pa pressure rise inside 5.5 degrees and a 6 m 300-500 hPa thickness drop
inside 6.5 degrees, allowing a warm maximum within 1 degree. Minima merge within
6 degrees. Winds are maxima within 2 degrees. Association is globally nearest
pair first, one-to-one, within 8 degrees and 24 hours. Qualification requires
54 hours and ten samples passing each of the wind, latitude and terrain tests.
This online association is a documented variant, not an assertion of identical
TempestExtremes tracks. Overflow invalidates the diagnostics instead of silently
discarding detections. Workspace capacities and thresholds are JSON table rows.
The shipped recipe reserves 4096 candidates and 8192 visited cells per flood.
Initialization uses the same 10 m wind fields as the engine's hour-zero history;
later samples use the resident surface diagnostics. A 24-hour demonstration
cannot qualify a 54-hour trajectory, even when it detects the storm at each sample.

GPI follows Eq. 1 of doi:10.1175/JCLI4282.1 using absolute 850 hPa vorticity,
600 hPa RH in percent, profile-derived PI and 850-200 hPa vector shear. Vorticity
uses a local least-squares spherical curl of earth-relative winds. No valid
curl is claimed poleward of 60 degrees. PI adapts the MIT tcpyPI reversible
ascent algorithm (doi:10.5194/gmd-14-2351-2021), with dissipative heating and the
Ck/Cd, surface-wind reduction and profile-top settings recorded in the recipe.
See the adjacent licence notice. Missing input profiles remain missing.

Diagnostic CUDA operations run only at existing history boundaries. No
atmospheric state is overwritten. Compact records are downloaded after
integration; no tracker scalars or trajectories are read by the host per step.
When disabled there is no sampler, device allocation or new history field.
The strict native-MPAS hl-3 reference gate has not been run for this bundle.

Current limitations: no tracker checkpoint restore, explicit native-MPAS suite
hook, active-class LTS hook or partitioned same-host hook. These selections
refuse rather than omit the requested diagnostic. Checkpoints may be written,
but a TC-enabled resumed forecast is refused. The idealized TC initializer,
one-day sanity run, full-resolution five-member season and model/IBTrACS track
density comparison remain outstanding.

A scaled real demonstration completed 24 hours from 2022-09-26 00Z on a
40962-cell global mesh with 55 levels, one member, 120 s steps and fast numerics.
It used initialized surface temperatures rather than an evolving SST series.
The detector retained one vortex through all five six-hour samples. The model
pressure stayed near 1001-1002 hPa while the observed center deepened from
991 to 965 hPa; the nearby-minimum displacement grew to about 111 km. This
checks the diagnostic integration and exposes the coarse model intensity miss.
It does not establish climatology, genesis skill or seasonal prediction skill.
The 4096-candidate capacity has not been validated on a high-resolution mesh.

The complete base, disabled and enabled forecasts each passed 720 steps.
Across five output frames, all 234506030 common array values matched bitwise
between base and disabled, and between disabled and enabled. This is a direct
fast-numerics same-case check, not the native hl-3 reference gate. Process walls
were 671.52 s for base, 824.20 s disabled and 644.74 s enabled in a shared GPU
pool with changing co-tenants; these differences do not establish a speedup.
Diagnostic stream intervals totaled 1048.82 ms across five samples, including
stream scheduling, with 0.2631 s of host enqueue time at history boundaries.

CPU algebra and front-door tests:

```sh
python -m pytest -q -m 'not gpu' tests/test_tc_climate.py
```

The GPU-marked test must run inside the approved site pool. CPU compilation of
scalar CUDA expressions and NVRTC-to-PTX compilation are not CUDA execution
proofs or meteorological validation.

Methods: doi:10.5194/gmd-10-1069-2017 and doi:10.5194/gmd-14-5023-2021
for graph detection, doi:10.1175/JCLI4282.1 for GPI,
doi:10.5194/gmd-14-2351-2021 and doi:10.1029/2001JD000776 for PI.

Observation-only chart from an existing NOAA IBTrACS v04r01 Atlantic CSV:

```sh
python demos/tc_climate_observations.py --csv "$IBTRACS_CSV" --year 2020 \
  --minimum-wind-kt 34 --out runs/observed
```

This small-table analysis selects main tracks with NHC TS/SS/HU status at
six-hour synoptic times. Its named-system threshold differs from the detector's
10 m/s persistence threshold; count definitions must be reconciled before a
model skill claim. It is not a weather-field map or a forecast comparison.
