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

MEASURED-SECTION
