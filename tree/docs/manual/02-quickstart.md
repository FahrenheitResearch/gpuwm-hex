# 2. Quickstart

This chapter goes from nothing to rendered forecast products. Every command
in it was run, in this order, on a fresh install before it was printed here —
the door commands against the installed 0.1.0 wheel, the forecast against the
repository checkout, the GPU steps on an RTX 5070 Ti. Where output is shown,
it is that run's output.

The true shape of the journey: the wheel installs in one command; the
doors then need three estates the wheel cannot carry (a CUDA-matched CuPy,
the Rust engines, the mesh assets), and `gpuwm-hex doctor` walks you through
closing each one by name.

## 2.1 Install

```sh
pip install gpuwm-hex
```

That installs the Python package, its dependencies (numpy, netCDF4, scipy)
and the `gpuwm` engine distribution. Then add the CUDA lane. pip cannot
detect which CUDA major your driver speaks, so you choose:

```sh
nvidia-smi        # the CUDA version in the header is your driver's major
```

```sh
pip install "gpuwm-hex[gpu-cu12]"    # driver reports CUDA 12.x
pip install "gpuwm-hex[gpu-cu13]"    # driver reports CUDA 13.x
```

Exactly one of the two. A CuPy wheel built for the wrong CUDA major
installs cleanly, imports cleanly, and fails on the first real device call —
which is why this is a choice you make, not one pip guesses.

Confirm what landed:

```sh
gpuwm-hex version
```

```json
{
  "distribution": "gpuwm-hex",
  "package": ".../site-packages/mpas_port",
  "source_checkout": null,
  "version": "0.1.0"
}
```

Note `"package": .../mpas_port` — the import namespace is `mpas_port`, not
`gpuwm_hex`. That is a declared inconsistency (README, "The import
namespace"); everything you *type* is `gpuwm-hex`.

## 2.2 Doctor first

```sh
gpuwm-hex doctor
```

On a fresh install with the CUDA extra, the report looks like this:

```
INFO    distribution: gpuwm-hex 0.1.0
INFO    interpreter: Python 3.14.4 on Linux x86_64
OK      numpy (arrays; 48 modules import it at line one): imported, 2.5.2
OK      netCDF4 (reads and writes every mesh, static and history file): imported, 1.7.4
OK      scipy (the regridder's spatial index): imported, 1.18.1
PRESENT gpuwm (the physics seam): gpuwm 2.5.2 is installed
INFO    gpuwm source checkout (the forecast lane only): ...
OK      cupy (the CUDA lane): imported, 14.2.0
INFO    gpuwm fetch-bridges coverage: the gpuwm installed here bundles no rw_mpas_convert, rw_mpas_init, ...
MISSING rw_mpas_init (the initial-condition builder): not found on any rung. ...
          cargo build --release --locked --offline -p rw-mpas --bin rw_mpas_init
MISSING rw_mpas_convert (the history converter): not found on any rung. ...
          cargo build --release --locked --offline -p rw-mpas --bin rw_mpas_convert
OK      rw_wrfbatch (the product renderer): found via gpuwm's bridge directories: ~/.gpuwm/bridges/rw_wrfbatch
INFO    mesh grid + static pair: external assets. ...

2 required item(s) missing: rw_mpas_init, rw_mpas_convert.
Run `gpuwm-hex doctor --explain` for the full remedy for each.
```

Doctor exits 1 while any required item is missing, so it works as a gate in
a script. `gpuwm-hex doctor --explain` prints the whole pasteable remedy
block for each finding; `gpuwm-hex doctor --json` emits the same findings as
data. Run doctor after every step below until it says `Every check passed.`

## 2.3 Close the gaps doctor names

### The Rust engines

The doors are orchestration only — all field data is handled by Rust
binaries built from the `gpuwm` source tree. Two roads:

**The staged road.** `gpuwm` publishes prebuilt bundles:

```sh
gpuwm fetch-bridges
```

This stages verified binaries into `~/.gpuwm/bridges`, which gpuwm-hex reads
directly (the run that produced the transcript above staged 59 files, every
one verified against its packaged pin). What it can supply depends on the
gpuwm you have: the transcript above was taken against the published gpuwm
2.5.2 wheel, whose bundle carries `rw_wrfbatch` and **not** `rw_mpas_init`
or `rw_mpas_convert` — which is exactly the shortfall the doctor reports
there. The four MPAS bridge binaries enter the bundle at gpuwm 2.5.3, and
that is why this distribution's floor is `gpuwm>=2.5.3`: a conforming
install cannot land on the engine that transcript was taken from. Doctor
asks the gpuwm you actually have what its bundle declares, so it never sends
you to a staging command that cannot deliver the file.

**The build road.** From a `gpuwm` source checkout, at `tools/rustwx`:

```sh
cd <gpuwm-checkout>/tools/rustwx
cargo build --release --offline --locked -p rw-mpas       # rw_mpas_init, rw_mpas_convert, rw_mpas_mesh, rw_mpas_static
cargo build --release --offline --locked -p rw-wrfbatch   # rw_wrfbatch
```

(Both commands were run for this manual; they finished in 36 s and 34 s on a
warm target directory.) Then point the doors at what you built:

```sh
export GPUWM_HEX_RW_MPAS_INIT=<gpuwm-checkout>/tools/rustwx/target/release/rw_mpas_init
export GPUWM_HEX_RW_MPAS_CONVERT=<gpuwm-checkout>/tools/rustwx/target/release/rw_mpas_convert
export GPUWM_HEX_RW_WRFBATCH=<gpuwm-checkout>/tools/rustwx/target/release/rw_wrfbatch
```

A flag or environment variable naming a missing file is a hard error, never
a silent fall-through to the next rung of the search ladder (chapter 9 has
the full ladder). When everything is closed:

```sh
gpuwm-hex doctor
```

```
...
OK      rw_mpas_init (the initial-condition builder): found via $GPUWM_HEX_RW_MPAS_INIT: .../rw_mpas_init
OK      rw_mpas_convert (the history converter): found via $GPUWM_HEX_RW_MPAS_CONVERT: .../rw_mpas_convert
OK      rw_wrfbatch (the product renderer): found via gpuwm's bridge directories: ~/.gpuwm/bridges/rw_wrfbatch
...
Every check passed.
```

## 2.4 Obtain the assets, stated plainly

**There is no fetch path in gpuwm-hex for any of these.** No
`gpuwm-hex fetch` command exists. If a document ever implies otherwise, it
is wrong. You need four things:

1. **A mesh grid file** (`x1.40962.grid.nc`, `x4.163842.grid.nc`, …), from
   the MPAS-Atmosphere project's published mesh downloads, from the
   `MPAS-Tools` generator, or from the engine's own `gpuwm mesh` door
   (chapter 4). This quickstart uses the published 40,962-cell global mesh
   `x1.40962` — the one that fits a 12 GiB card.
2. **The matching static file** (`x1.40962.static.nc`), carrying terrain,
   land use, soil and vegetation for that exact mesh. The published meshes
   have published statics; a generated mesh gets its static from the same
   `gpuwm mesh` run. Grid and static must be the same mesh — the doors
   cross-check and refuse a mismatched pair by name.
3. **For the init door: a native-minted init-class file for this mesh, as a
   "capsule."** The init engine reads the vertical-grid arrays out of it
   rather than inventing them. This is the sharpest prerequisite in the
   product — one file, once per mesh, built by native
   `init_atmosphere_model` — and chapter 5 explains it fully, including the
   status of the in-progress path that removes it.
4. **Meteorological input**: a WPS intermediate file from `ungrib` (GFS,
   ERA5, whatever you drive with), valid at your start time.

Sanity-check the mesh pair before anything expensive touches it:

```sh
gpuwm-hex mesh-check --grid assets/x1.40962.grid.nc --static assets/x1.40962.static.nc
```

```json
{
  "connectivity_indexing": "zero-based with -1 padding",
  "dimensions": { "maxEdges": 10, "nCells": 40962, "nEdges": 122880, "nVertices": 81920 },
  "grid": ".../assets/x1.40962.grid.nc",
  "grid_sha256": "9a9e1909a755dac209462ceb0bfffd77ac1b37503169568b7f296707ee612bb9",
  "passed": true,
  "static": ".../assets/x1.40962.static.nc",
  "static_sha256": "cf1a47d4168327f06a8403555d6ed8b2fe1aff7f8b916bb7f6a754c34a10ac82"
}
```

## 2.5 First initial conditions

Build an init from your WPS intermediate file. Every physics switch is
required and has no default — each one changes the numbers in a file that
would open cleanly either way, so a default would be a silent wrong answer.
Each refusal prints the native namelist key it corresponds to (chapter 5
maps them all).

```sh
gpuwm-hex init \
  --met     <path-to-WPS-intermediate> \
  --grid    assets/x1.40962.grid.nc \
  --static  assets/x1.40962.static.nc \
  --capsule   assets/x1.40962-native-init.nc \
  --reference assets/x1.40962-native-init.nc \
  --out     work/x1.40962.init.nc \
  --start-time 2026-08-12_06:00:00 \
  --nfglevels 34 --nfgsoillevels 4 \
  --extrap-airtemp lapse-rate --use-spechumd no \
  --theta-adv-order 3 --coef-3rd-order 0.25 \
  --virtual-factor reproduce-fortran \
  --deep-soil-moisture reproduce-fortran \
  --landuse-table MODIFIED_IGBP_MODIS_NOAH \
  --frac-seaice yes --tsk-seaice-threshold 100.0 \
  --oned-underflow preserve
```

`--nfglevels` must cover the levels actually in your met file; declare too
few and the door refuses with the real count (the proving run's met file, a
GFS intermediate, carried 34). The run above wrote a 372 MB
`x1.40962.init.nc` plus `x1.40962.init.nc.provenance.json` — the SHA-256 of
every input, the engine binary, the argv, the engine's own receipt, and the
output — in 2.7 seconds, and exited 0.

## 2.6 First forecast

**The forecast lane needs the gpuwm-hex source checkout — stated plainly.**
There is no `gpuwm-hex forecast` console script in 0.1.0, deliberately: the
lane needs a `gpuwm` **source checkout at a pinned commit** on top of the
installed distribution (the seam pin includes a repository document no wheel
installs), and its authority meshes have no fetch path. Chapter 6 has the
whole story. What that means concretely:

- clone/obtain the gpuwm-hex repository and the `gpuwm` repository;
- check the `gpuwm` checkout out at the pinned commit (the run verifies
  the checkout's sixteen pinned files by SHA-256 at launch and refuses a
  mismatch naming the file and both digests);
- run the forecast driver from the gpuwm-hex checkout's `tree/` directory.

The command below ran a 1-hour, 30-step full-physics forecast on the
40,962-cell global mesh, on a 16 GiB RTX 5070 Ti, from the init built in
2.5, in 146 s wall:

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

```
[mesh-binding] mesh x1.40962: nCells=40962 nEdges=122880 nominal=120000.0 m; min(dcEdge)=97076.508 m; dt=120.000 s; limit=698.951 s
[mesh-binding] bound x1.40962: nCells=40962, nEdges=122880, dx=120000.0 m, dt=120.0 s; ...
{"history_frames": 3, "integration_seconds": 97.8, ..., "status": "passed", "steps": 30}
[mesh-run] mesh=x1.40962 rc=0 wall=145.968s
```

`work/out/` now holds three history files (analysis, +30 min, +60 min) and
`cuda-v841-forecast-receipt.json` — the receipt that states exactly what
this run claims and, at equal prominence, what it does not (chapter 6).

## 2.7 First rendered products

Back on the installed doors — `render` needs no checkout and no GPU:

```sh
gpuwm-hex render \
  --history work/out/cuda-history.2026-08-12_07.00.00.nc \
  --mesh    assets/x1.40962.grid.nc \
  --out     work/png \
  --simulation-start 2026-08-12_06:00:00 \
  --products all
```

```
CONVERT CONVERTED  .../frames/wrfout_d01_2026-08-12_07_00_00  0.053s  107377492 bytes  35 written  3 absent
FRAME 2026-08-12T07:00:00 rendered=41 skipped=0 failed=0
MANIFEST work/png/render-manifest.json
SCRATCH cleared work/png.render-scratch
DOOR rendered=41 skipped=0 failed=0 out=work/png
```

41 product PNGs, filed under `work/png/<domain>/<product>/<valid-day>/`
(for this mesh the focus window is `d01-22km`), with a
`render-manifest.json` carrying engine digests, per-frame results, and the
exact invocations. 3.6 seconds on the proving node. That is the whole loop:
install → doctor → assets → init → forecast → pictures.

## Where to next

- Something refused? Chapter 8 indexes the refusals you will actually meet.
- Want your own mesh, or to know what fits your card? Chapter 4.
- Want to understand what you just ran before trusting it? Chapter 3.
