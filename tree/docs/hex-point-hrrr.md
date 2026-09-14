# A fine hex core at any CONUS point, from HRRR

Four doors that were missing, in the order a case meets them, and the
measured chain that ran them end to end on 2026-09-13. Before this lane every
sub-kilometre row this program held sat at 35.0 N 97.0 W, because a person
typed its spec and its registry row; and no HRRR byte had ever reached a hex
init, because a Lambert product is refused by the intermediate writer and the
init engine alike (`docs/source-matrix.md`). Both of those were the reasons a
"hex nest at an arbitrary point from HRRR" was on the not-built list.

```
gpuwm-hex mesh-plan --point LAT,LON --fine-dx-m 937.5 --radius-km 100 --card 32gb \
    --generate --out-dir MESH --vertical-spec verification/vertical-specs/tc55-v1.json
gpuwm-hex intermediate --source hrrr --from-plan MESH/<stem>.point-generate.json \
    --grib-dir HRRR --cycle YYYY-MM-DDTHH --hours 0-3 --out-dir MET
gpuwm-hex init --met MET/MET:YYYY-MM-DD_HH --static MESH/<stem>-cull.static.nc \
    --capsule MESH/<stem>-cull.vertical.nc --reference MESH/<stem>-cull.vertical.nc ...
gpuwm-hex lbc --grid MESH/<stem>-cull.init.nc --met-dir MET --out-dir MESH/<stem>-cull.lbc ...
GPUWM_HEX_MESH_ROWS=MESH/mesh-rows.json gpuwm-hex forecast --mesh <cull row> --lbc-dir ...
```

Every door is on the complete surface: nothing here is a walk that hides a
knob. `--spec` still hands the generator an authored spec unchanged; `--point`
writes one for you and says what it wrote.

## `mesh-plan --point`: the ladder is data

The spec is a pure function of six numbers (`hexcore.mesh_point.ladder_spec`):
the point, the fine spacing, the core radius, the global background, the ramp
factor and the ring factor. It is the registered `v0.9.120.110533` recipe
restated -- nested caps on a 120 km background, each rung halving the spacing,
each ramp eighteen times its own spacing, each cap three ramps of the next
finer rung outside it -- and at 35.0 N 97.0 W, 937.5 m, 100 km it reproduces
the spec of record to the digit (tested). Moving the core moves the centre of
every cap and nothing else.

A fine spacing that is not a rung of the halving ladder is refused naming the
rungs on either side, because the generator snaps a request onto the nearest
rung ALWAYS FINER: a 900 m request would silently build 468.75 m and cost four
times the cells the card was priced for.

The plan prices two things and admits on the second, exactly as the swath
layer does:

- the GLOBAL parent, through the generator's own `--dry-run` with the gates a
  build applies (`hexcore.swath.sizing`);
- the limited-area CULL, an area integral over the ladder's own spacing
  profile at the generator's attained fine spacing (`basis: area_integral`,
  a bound the cull receipt replaces with the count), priced on the admission
  surface's limited-area row for the named card with that row's own margin
  (`hexcore.device_admission`). The verdict names the cells the card holds
  and the largest core radius that fits at this spacing and pad.

`--card` is a row in `hexcore.mesh_point.CARD_ALIASES` joined to a measured
row of the admission surface; a card nobody measured is refused rather than
given another card's number.

## `--generate`: build, admit, register, cull

With `--generate` the door builds the pair (`rw_mpas_mesh`, then
`rw_mpas_static` -- the static builder joins the engine ladder as
`hexcore.engines.STATIC`), measures the pair's own admissions before
registering anything (dual edges, cell coordination, Courant at the largest
anchored timestep the mesh's own `min(dcEdge)` admits), writes the RUNTIME ROW,
cuts the cull at the shipped 1.35x pad, admits and registers that too, and
with `--vertical-spec` mints the parent's native-free vertical artifact and
culls it.

**Why the vertical is minted on the parent and culled.** `gpuwm-hex init`
refuses a limited-area grid in native-free mode, correctly: the closed-sphere
vertical authority does not invent exterior state. A global init needs global
meteorology and HRRR covers only its own domain. Minting the vertical on the
GLOBAL parent (grid + static + spec, no meteorology) and culling the artifact
with the same region as the grid and static gives the cull a vertical that IS
the parent's, cell for cell; the culled artifact is then handed to
`gpuwm-hex init --capsule/--reference` with the regional intermediate.

The receipt is `<stem>.point-generate.json` (`gpuwm-hex.point-generate/v1`)
and its `next` list is the rest of the chain with every path filled in.

## Runtime rows: `mesh-rows.json` beside the mesh

`gpuwm-hex forecast` resolves `--mesh` against `tools/mpas_mesh_binding.py`
and refuses a name it does not hold. A mesh generated at an arbitrary point
cannot be a row in that file, and a checkout edit per point would be the
hand-written-row-per-swath defect the cascade retired. `hexcore.mesh_rows`
is the answer: the door writes a `gpuwm-hex.mesh-rows/v1` document beside the
mesh, and the registry module applies it -- before the cascade rows, so a
cascade may cull a generated parent -- when `$GPUWM_HEX_MESH_ROWS` names it,
returning the mapping unchanged on every ordinary run.

What a registry row buys is kept: the bytes are pinned by count and SHA-256
and re-hashed at bind; Courant, dual-edge, cell-coordination and regional
admissions are measurements of the files at bind; the timestep is refused
unless an anchor covers the configuration; the regional opener re-measures
the class key and requires a contract deck on THESE rings. What stands in for
a person reading the row is the door's own admission pass, recorded in the
row. Refused by name: a runtime row that shadows a shipped row; a cull row
whose parent is neither in the file nor registered; a row whose bytes moved;
a cull row with no `lbc_source`; a document of another schema.

## `intermediate`: HRRR onto the lat-lon record the engines read

The gap, measured in `docs/source-matrix.md`: the engine's intermediate writer
refuses to mint a projected grid as `iproj 0`, and `rw_mpas_init` /
`rw_mpas_lbc` invert projection code 0 only. This door resamples the
projected product onto a regular 0.025-degree lat-lon box over the cull (cut
radius plus the boundary halo plus a 30 km margin) and writes the WPS
version-5 records the engines already read. Everything with numbers in it is
the engine's own, driven, not re-implemented:

- decode: `hrrr_grib2_bridge`, resolved through `gpuwm.bridges`, over the
  `wrfnat` + soil pair `gpuwm fetch --source hrrr` writes, on its
  `--series-workers` route (`HOUR<TAB>WRFNAT<TAB>SOIL` rows), into the native
  window `gpuwm.ingest.hrrr` maps;
- regrid: the engine's projected-source plan (`_ProjectedCpuPlan`, WPS's
  overlapping-parabolic operator with FP64 donor selection, on the engine's
  CPU bridge), aimed at the lat-lon target; winds rotated to the earth basis
  on the source grid with the engine's own rotation first, because an
  `iproj 0` record is earth-relative by definition;
- soil: the engine's node-to-Noah-layer interpolation over its own HRRR
  depth nodes, moved by nearest neighbour;
- write: the inverse of this tree's own frozen reader
  (`hexcore.wps_intermediate`); every file is read back through that reader
  before the receipt is signed, and a slab carrying a non-finite value is
  refused.

Fifty hybrid levels are carried as level-indexed slabs with the 3-D
`PRESSURE` beside them (the init engine's model-level branch, the ERA5-ML
convention), plus the 200100.0 surface level, so `--nfglevels 51
--nfgsoillevels 4 --use-spechumd yes --extrap-airtemp constant`. Hydrometeors
are not carried: the init stream has no slot for them, and hour zero has no
ice (README, limited-area lane).

The source is a ROW (`hexcore.hrrr_intermediate.SOURCE_ROWS`): decoder key,
loader module, file names, field map, soil nodes, level count. Nothing below
the table branches on the string `hrrr` (tested); a second projected source
with an engine decoder and loader is a second row.

## `lbc`: one boundary file per intermediate valid time

`rw_mpas_lbc` on its wps-intermediate route over the hourly intermediates,
each admitted by the valid time in its own header, never its file name. A
series that does not cover `--start-time..--stop-time` at both ends, or that
is unevenly spaced, is refused by name: a boundary series with a missing end
freezes the boundary there without saying so.

The binary resolves through the engine ladder (`hexcore.engines.LBC`):
`--lbc-exe`, `$GPUWM_HEX_RW_MPAS_LBC`, `$RW_MPAS_LBC`, `$GPUWM_RW_MPAS_LBC`,
gpuwm's bridge directories (`libexec/bridges` beside the installed package and
`~/.gpuwm/bridges`, where `gpuwm fetch-bridges` stages the 2.7.3 bundle that
carries it), then `PATH`. `gpuwm-hex doctor` reports it beside the other
engines.

## The init carries its own clock

`rw_mpas_init` writes the init's global attributes by copying the
capsule's and does not stamp `config_start_time` itself. A native capsule
carries it from its namelist; the constructed route writes it into the
vertical artifact; the vertical `--generate` culls at mesh time is minted
before any start time exists and carries neither. The forecast driver
asserts `--start-time` against that attribute and refuses an init without
one, so the first point cull reached the card, spent seven minutes on the
contract deck, and stopped at the forecast opener. `gpuwm-hex init` now
stamps every run declaration the capsule copy left absent
(`hexcore.init_door.stamp_run_declarations`, the same nine `config_*`
attributes the constructed route writes, encoded by the one encoder both
share) and records `run_declarations.stamped` and
`run_declarations.already_present` in the init receipt. Attributes the
capsule carried are left byte for byte, so the native x4 route is untouched;
a carried clock that disagrees with `--start-time` is reported on stderr and
left for the forecast door's own refusal, because the cascade's delayed
start moves it on purpose.

## The same-ladder class band

The regional forecast opener measures a class key off the run (zone width,
column count, finest edge, timestep, kernel set) and refuses a configuration
no forecast mint covers. Until this lane the finest edge compared exact to the
millimetre, which was protection against a printed number, not real drift: two
culls of one parent carry `min(dcEdge)` in identical bits. A parent generated
at another point from the same ladder spec delivers the same rung and a finest
edge that differs by relaxation noise, so the two sub-kilometre classes
(`graded-869m-dt5-z7`, `graded-711m-dt5-z7`) now declare a five per cent band
(`SAME_LADDER_FINEST_EDGE_BAND`), which cannot reach from one to the other
(18 % apart) and which the Courant admission re-measures against the cull's
own `dcEdge` regardless. Exact matches are tried first. The contract deck on
the cull's own rings is still required and still run; nothing about the
geometry half is relaxed.

## Measured, 2026-09-13

See the section below; it is written from the receipts under
`C:\Users\drew\Downloads\evidence-gallery\soma-hex-point-hrrr-2026-09-13\`.

Point 41.95 N 94.79 W (central Iowa), HRRR cycle 2026-09-13 15Z, f00 to
f03, node-2 (RTX 5090, 32,607 MiB, sm_120), gpuwm 2.7.3 from PyPI with its
2.7.3 bridge bundle, gpuwm-hex 0.3.0 at this lane's tip. Every number below
is copied from a receipt in that folder; the chain scripts that produced them
are beside the receipts.

The mesh (`mesh-plan --point 41.95,-94.79 --fine-dx-m 937.5 --radius-km 100
--card 32gb --generate`, 276 s wall on node-2's CPU):

- Ladder 120 / 60 / 30 / 15 / 7.5 / 3.75 / 1.875 / 0.9375 km, transition
  factor 18, fine core radius 100 km, fine flat radius 83.1 km, cut radius
  135 km (pad scale 1.35), spacing at the cut 2.88 km, halo 20.2 km.
- Parent `p0.9375.120.112246.n41.95w94.79`: 112,246 cells, 336,732 edges;
  `rw_mpas_mesh` 90.0 s, `rw_mpas_static` 100.6 s; min `dcEdge` 841.324 m,
  min `dvEdge/dcEdge` 0.0402 (floor 0.02), coordination {5: 2000, 6: 108260,
  7: 1984, 8: 2}, dt 5 s against a 6.058 s Courant limit (1.21x).
- Cull `q0.9375.120.43884.n41.95w94.79`: 43,884 cells (43,421 predicted,
  1.1 % over), 131,956 edges, seven boundary rings, attained fine spacing
  0.93751 km; the same finest edge and dt; registered as the second runtime
  row in `mesh-rows.json` with `lbc_source` naming the boundary directory.
- Priced for the 32 GB card: 11,827.9 MiB predicted for the cull on the
  measured limited-area row (96,582 B per cell over a 663 MB core), margin
  1,756.8 MiB, 248,102 cells would fit, so the largest fine core this card
  admits at 937.5 m is 240 km in radius.

The state (`intermediate`, `init`, `lbc`, all CPU):

- `intermediate --source hrrr --hours 0-3 --workers 4`: `hrrr_grib2_bridge`
  decoded the four `wrfnat` + `soil` pairs in 17.6 s onto the cull's lat-lon
  box (rows 908 to 1043, columns 591 to 726 of the HRRR grid); each WPS file
  holds 318 records (51 levels of TT, UU, VV, SPECHUMD; 50 of GHT and
  PRESSURE; the surface and soil set) in 31.2 MB, regridded in 0.08 s.
- `init`: `rw_mpas_init` 1.44 s, 317 met records used, 55 levels; the door
  stamped all nine run declarations onto the init (`run_declarations.stamped`
  in its receipt; `already_present` empty), which is what the section above
  is about.
- `lbc`: four boundary files (15Z to 18Z, 87.3 MB each) in 2.08 s through
  `rw_mpas_lbc` resolved from the venv's own bridge directory.

The card (both passes under the shared mutex; 23.3 minutes of card time in
total):

- Contract deck on the cull's own rings (`run_cuda_regional_contract.py
  --class-id graded-869m-dt5-z7`): 8 decks, 8 passed, 22 of 22 regional
  kernels covered, every deck bitwise, every control with teeth, dual run
  identical; 422 s, peak 1,026 MiB.
- `forecast --hours 3 --history-every-minutes 60 --lbc-dir ...`: admitted on
  the measured limited-area row (11,874.3 MiB predicted, 31,642.6 MiB free,
  margin 1,756.8 MiB), convection off by resolution (841 m against the 3 km
  threshold), surface/PBL every step (the welded cadence, 720 calls an hour);
  2,160 steps of 5 s in 721.3 s of integration, **0.334 s per step**, 912.2 s
  driver execution, 943.6 s door wall including the pinned-source
  verification of 23 modules; four history frames; status passed. Peak
  memory on the card 9,485 MiB (nvidia-smi at 5 s cadence), utilisation
  75 to 97 %. Load average at launch 0.34 with nothing else on the node's
  CPUs; the mutex had been held 420 s by another lane before this pass.
- The first pass stopped at the forecast opener on the clockless init after
  the deck had run; the deck's receipt pins the boundary mask, the cell count
  and the kernel set, none of which the remade init changes, so the second
  pass reused it (`SKIP_DECK=1` in the chain script).

The picture (`render --products composite_reflectivity,2m_temperature_10m_winds`
on the 18Z frame): `rw_mpas_convert` framed the mesh window at 233 x 235
columns (924 m, mean nearest-neighbour 0.51 km) in 0.81 s and `rw_wrfbatch`
drew both panels; 5 s in all. `theta_e_2m_10m_winds` is in the renderer's
catalog but not realised by the wrfout import lane for this history, and the
door refuses it by name before drawing anything.


## The surface/PBL cadence, measured 2026-09-13

Same point mesh, same HRRR 15Z init and boundary files (one `intermediate`,
`init` and `lbc` pass, digests in the evidence folder), node-2's RTX 5090
with no other process on the card during any arm (a co-tenant job had
finished before the first arm; load average 1.0 to 1.3 throughout), one
session, one hour (720 steps of 5 s), history every 60 minutes. Five arms
back to back under one mutex hold: the untouched `lane/hex-point-hrrr` tree
in the shared venv at the welded default, then this tree at 120, 60 and
30 s, then this tree at the welded default. Card time 1,656 s (688 s for
the first two arms, 968 s for the last three; a fresh-scratch refusal split
the chain once). Receipts, step receipts, logs and the nvidia-smi trace are
under `C:\Users\drew\Downloads\evidence-gallery\hex-perf-cadence-2026-09-13\`.

**The default is unchanged, byte for byte.** This tree at `auto` wrote both
history frames byte-identical to the untouched tree's (15Z
`85cb8196...`, 16Z `4ed8001f...`, the same two digests the 2026-09-13 profile
lane recorded), at 0.3156 s per step after the first against the untouched
tree's 0.3148 (median composite step 0.2994 s in both).

**The hold is real, and the receipt proves it.** At 120 s the seam
reported the surface/PBL stack due on steps 1, 24, 48, ... -- 31 of 720,
689 held -- with the engine's own `sfclay`, `noah` and `ysu` counters all
reading 31 at the last step and radiation due 6 times, exactly the count
the declared cadence predicts (`physics.cadence` in the receipt:
`consistent: true`). 60 s: 61 due, 659 held. 30 s: 121 due, 599 held.

**What it buys: about five per cent per step, the same at every cadence.**

| arm | s/step after the first | median composite step | ratio (median) | due / held | peak card memory |
|---|---|---|---|---|---|
| every step, untouched tree | 0.3148 | 0.2994 | 1.000 | 720 / 0 | 8,914 MiB |
| every step, this tree | 0.3156 | 0.2994 | 1.000 | 720 / 0 | 8,914 MiB |
| 30 s (6 steps) | 0.3022 | 0.2845 | 0.950 | 121 / 599 | 8,914 MiB |
| 60 s (12 steps) | 0.3005 | 0.2842 | 0.949 | 61 / 659 | 8,914 MiB |
| 120 s (24 steps) | 0.3013 | 0.2844 | 0.950 | 31 / 689 | 9,421 MiB |

The saving is 14.9 to 15.2 ms per step and does not grow past 30 s: what a
held step skips is the surface layer, Noah-MP and YSU (the profile's 16 to
19 ms of Noah-MP host time is most of it), and what it keeps -- the
phase-one prep, the 447 MB rollback snapshot, GWDO, WSM6, the commit, the
host health gate and the dycore -- is 95 per cent of the step. The 120 s
arm's larger peak memory and its 356.5 s door wall (against 318 to 330 s
for the others) are its first step: 34.3 s of cold NVRTC compilation into a
fresh kernel cache, which the other arms of this tree then hit; the
per-step numbers above exclude the first step.

**What it costs: a different forecast, everywhere.** Differences at 16Z
against the every-step arm (same init, same card, same session; the
every-step arm's own dual-run noise floor is zero):

| field | 30 s max / RMS | 60 s max / RMS | 120 s max / RMS | 2.6.5 to 2.7.3 seam move (max) |
|---|---|---|---|---|
| theta, K | 1.245 / 0.0235 | 1.839 / 0.0340 | 2.949 / 0.0540 | 0.139 |
| qv, kg/kg | 1.04e-3 / 1.3e-5 | 1.20e-3 / 2.0e-5 | 1.80e-3 / 3.2e-5 | 3.7e-4 |
| u, m/s | 1.469 / 0.0262 | 2.487 / 0.0469 | 5.038 / 0.0859 | |
| v, m/s | 1.074 / 0.0238 | 1.820 / 0.0412 | 3.040 / 0.0737 | |
| w, m/s | 1.563 / 0.0204 | 3.001 / 0.0382 | 4.692 / 0.0705 | |
| T2, K | 0.456 / 0.0771 | 0.507 / 0.0929 | 0.657 / 0.1074 | |
| Q2, kg/kg | 1.66e-4 / 1.4e-5 | 1.85e-4 / 1.9e-5 | 2.97e-4 / 2.8e-5 | |
| U10, m/s | 0.384 / 0.0382 | 0.439 / 0.0558 | 0.530 / 0.0855 | |
| rain, mm | 0.045 / 0.0029 | 0.100 / 0.0056 | 0.220 / 0.0110 | |
| sensible heat flux, W m-2 | 58.5 / 7.3 | 62.4 / 10.1 | 95.4 / 13.0 | |
| PBL height, m | 420 / 19 | 480 / 26 | 417 / 32 | |

Every surface cell differs (43,884 of 43,884 for T2, Q2, U10 and the
fluxes) and 98 per cent of the theta and qv column points do; the maxima sit
in the convective cells (level 19 to 29, where the reference w reaches
23 m/s) and the surface fields move because the surface layer and Noah-MP
integrate with the cadence as their step. The yardstick the campaign named
is the 2.6.5 to 2.7.3 engine seam movement recorded in
`declared-divergences.md` -- a column-seam number, eight columns over forty
minutes, theta up to 0.139 K and qv up to 3.7e-4 kg/kg -- and the smallest
cadence exceeds it nine times over in theta and three times in qv at the
maximum; the card's own contribution, the other yardstick, is zero.

**The decision.** No cadence qualifies: none keeps its one-hour differences
below either yardstick, and the speed it buys is the same five per cent at
30 s as at 120 s. The knob ships selectable with the every-step default,
which is what this tree does. If a held cadence is ever graded against
observations (MRMS, ASOS; the referee `docs/obs-referee.md` names), 30 s is
the arm to grade: the same saving as 120 s with the smallest differences.
The time in a step is elsewhere -- the host health gate (a fifth of it),
the per-step rollback snapshot and the double-launched recovery kernels
the profile named -- and that is where the next lanes should look.

The reflectivity panels for the every-step arm and the 120 s arm
(`composite_reflectivity` and `2m_temperature_10m_winds` on the 16Z frame,
through `gpuwm-hex render`) are in the evidence folder beside this section's
receipts; they are model results, not evidence about the atmosphere.
