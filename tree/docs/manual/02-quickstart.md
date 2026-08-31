# 2. Quickstart

This chapter goes from nothing to rendered forecast products. Every command
in it was run, in this order, on a fresh install before it was printed here —
the door commands against the installed 0.2.0 wheel, the forecast against the
repository checkout, the GPU steps on an RTX 5070 Ti. Where output is shown
it is that run's output, except where a block is marked as a *form* — the
shape of a line rather than a captured one. The card transcripts in 2.6 come
from a second machine, a 10 GiB RTX 3080; their `ADMISSION` lines are
re-derived on the model the door decides with today, from that card's own
driver reading [`evidence/memory-shape-20260827/ON-CARD-366.json`], because
the model the original run printed has since been retired (6.2).

The true shape of the journey: the wheel installs in one command; the
doors then need three estates the wheel cannot carry (a CUDA-matched CuPy,
the Rust engines, the mesh assets), and `gpuwm-hex doctor` walks you through
closing each one by name.

## 2.1 Install

```sh
pip install gpuwm-hex
```

That installs the Python package, its dependencies (numpy, netCDF4, scipy)
and the `gpuwm` engine distribution at a version this port has measured.

**No engine constraint of your own is needed, and that is new.** Until
2026-08-27 this page told you to type an engine version beside the package,
because the declared dependency was `gpuwm>=2.5.5` with no ceiling: a bare
install took the newest published engine, and the forecast lane pins sixteen
individual gpuwm source files by SHA-256. A bare install reached an engine the
pin refused, and the forecast door refused at launch with a digest mismatch
that named neither the version that works nor the fact that pip chose it.

The declaration now carries a bounded range — `gpuwm>=2.6.0,<2.6.1` — so pip
resolves the engine that works and refuses the one that does not. Three things
follow that are worth knowing rather than discovering:

- **2.6.0 is the only published engine that matches**, re-measured 2026-08-31
  against every published 2.5.x and 2.6.x release. So there is no fallback:
  the range is one version wide because only one version qualifies.
- **A newer gpuwm will not be resolved either.** The ceiling sits at the
  first engine this port has not measured against the manifest, so a future
  2.6.1 is excluded exactly as 2.5.8 is. That is deliberate: an unmeasured
  engine is what produced the defect.
- **If you already have a different gpuwm in the environment**, `gpuwm-hex
  doctor` compares the installed engine's bytes against the pin and names
  both the version it found and the version to install. It is a byte check,
  not a version check, and it runs by default. On a conforming install it
  reads `gpuwm 2.6.0: 16 of 16 pinned files are in this install and all 16
  match` (measured 2026-08-31 against the published 2.6.0 wheel).

Then add the CUDA lane:

```sh
pip install "gpuwm-hex[gpu]"     # or, spelled out, "gpuwm-hex[gpu-cu13]"
```

**There is one right answer here and it is CUDA 13.** Every GPU door in this
port goes through `require_cuda`, which refuses a CUDA runtime below `13000`
by name, so `cupy-cuda13x` is the only wheel the port can execute on. The
`[gpu-cu12]` extra still resolves — it is in 0.1.0's published metadata — and
a CuPy installed through it imports, allocates, runs cuBLAS and compiles an
NVRTC kernel, and is then refused at forecast launch with
`CudaRefusal: cuda.runtime_version=12090 < required 13000`. Do not use it.

Check the driver, because a CUDA-13 wheel needs a driver that serves CUDA 13:

```sh
nvidia-smi        # the CUDA version in the header is your driver's major
```

(On recent drivers that header field is spelled `CUDA UMD Version` — driver
610.74 reads `CUDA UMD Version: 13.3`. A driver below 13 cannot open this
port's CUDA lane at all, and no pip command changes that; `gpuwm-hex doctor`
reads the driver itself and says so rather than offering an install that
cannot work.)

Confirm what landed:

```sh
gpuwm-hex version
```

```json
{
  "distribution": "gpuwm-hex",
  "package": ".../site-packages/hexcore",
  "source_checkout": null,
  "version": "0.2.0"
}
```

Note `"package": .../hexcore` — the import namespace is `hexcore`, not
`gpuwm_hex`. It was renamed from `mpas_port` at 0.2.0 (README, "The import
namespace"); everything you *type* is `gpuwm-hex`.

## 2.2 Doctor first

```sh
gpuwm-hex doctor
```

On a fresh install with the CUDA extra, the report looks like this:

```
INFO    distribution: gpuwm-hex 0.2.0
INFO    interpreter: Python 3.14.4 on Linux x86_64
OK      numpy (arrays; 48 modules import it at line one): imported, 2.5.2
OK      netCDF4 (reads and writes every mesh, static and history file): imported, 1.7.4
OK      scipy (the regridder's spatial index): imported, 1.18.1
PRESENT gpuwm (the physics seam): gpuwm 2.5.2 is installed
INFO    gpuwm git checkout (the forecast lane only): ...
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
this distribution's declared range is `gpuwm>=2.6.0,<2.6.1` (the floor is
far past 2.5.3, for the seam bytes the port pins): a conforming install cannot
land on the engine that transcript was taken from. Doctor
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
   `x1.40962` — the one a 10 GiB card holds.
2. **The matching static file** (`x1.40962.static.nc`), carrying terrain,
   land use, soil and vegetation for that exact mesh. The published meshes
   have published statics; a generated mesh gets its static from the same
   `gpuwm mesh` run. Grid and static must be the same mesh — the doors
   cross-check and refuse a mismatched pair by name.
3. **For the init door: a vertical-grid declaration.** The normal path is
   a `--vertical-spec` JSON (`gpuwm-hex.vertical-spec/v1`), from which the
   door constructs the vertical grid itself — no native toolchain anywhere
   in the loop. The compatibility mode reads the vertical out of a
   native-minted init-class file (`--capsule`); chapter 5 explains both
   and when to prefer which.

   **The example specs are not in the wheel.** They live at
   `verification/vertical-specs/` in the *source distribution* and the
   repository, and `pip install` places a wheel, so the path 2.5 prints
   below does not exist on a fresh install — the door refuses with the
   absolute path it could not find, which is correct and is not a hint about
   where to get one. Take them from the sdist
   (`pip download --no-binary :all: gpuwm-hex`, then unpack) or from the
   gpuwm-hex repository, and put them where 2.5 expects.
4. **Meteorological input**: a WPS intermediate file from `ungrib` (GFS,
   ERA5, whatever you drive with), valid at your start time.

For 1 and 2, on this mesh, the public download and the registered row are the
same bytes — verified 2026-08-24 by fetching both archives fresh and hashing
what came out:

| file | from | bytes | SHA-256 |
| --- | --- | --- | --- |
| `x1.40962.grid.nc` | `x1.40962.tar.gz` | 56,039,332 | `9a9e1909a755dac209462ceb0bfffd77ac1b37503169568b7f296707ee612bb9` |
| `x1.40962.static.nc` | `x1.40962_static.tar.gz` | 94,766,584 | `cf1a47d4168327f06a8403555d6ed8b2fe1aff7f8b916bb7f6a754c34a10ac82` |

Both match `tools/mpas_mesh_binding.py` exactly, so the forecast door's byte
check passes on what the published downloads give you. That is a fact about
those archives, not a fetch path: gpuwm-hex still fetches nothing.

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
  --vertical-spec verification/vertical-specs/tc55-v1.json \
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

This is the native-free mint: `--vertical-spec` names a versioned JSON
vertical declaration and no native artifact appears anywhere in the
lineage (the receipt records `native_runtime_dependency: false`). The
compatibility mode passes `--capsule` + `--reference` naming a
native-minted init-class file instead; chapter 5.1 has both modes and
their receipts. `--nfglevels` must cover the levels actually in your met
file; declare too few and the door refuses with the real count (the
proving run's met file, a GFS intermediate, carried 34). The run above
wrote a 385 MB `x1.40962.init.nc` plus
`x1.40962.init.nc.provenance.json` — the SHA-256 of every input, the
engine binary, the argv, the engine's own receipt, and the output. The
engine step takes about 2 s; the first mint for a mesh spends minutes in
the door's geometry solve, and the keyed cache brings a re-mint down to
about a minute.

## 2.6 Ask the card first

The forecast lane is the one step whose answer depends on your hardware, so
ask before you spend anything on it. **Both checkouts below are required and
neither is optional**: `--repo` is your gpuwm-hex checkout (the drivers are
not in the wheel) and `--gpuwm-checkout` is a gpuwm checkout at **`v2.6.0`**,
the tag whose bytes the seam pin matches (2.1). Leave `--repo` off and the
door refuses before it reads anything else.

```sh
gpuwm-hex forecast --preflight \
  --mesh    x1.40962 \
  --grid    assets/x1.40962.grid.nc \
  --static  assets/x1.40962.static.nc \
  --init    work/x1.40962.init.nc \
  --init-source "GFS 2026-08-12 06Z" \
  --hours 1.0 --history-every-minutes 30 \
  --out work/fc-01 \
  --repo <gpuwm-hex-checkout>/tree \
  --gpuwm-checkout <gpuwm-checkout-at-v2.6.0>
```

`--gpuwm-checkout` must be a **git clone**, not an unpacked release tarball
and not the installed distribution: the driver asks it
`git rev-parse --show-toplevel`, because it writes that checkout's HEAD, tree
and dirty paths into every receipt so the seam source it executed can be named
by commit. A tarball or a `site-packages` directory is refused by name, saying
that. (Before 2026-08-28 the same case exited 128 and surfaced as a raw
`CalledProcessError` the door itself labelled a driver defect.)

On a card that holds the mesh, preflight binds it, admits it, runs the
driver's own source-pin and host checks, and exits 0. This transcript is
from a 10 GiB RTX 3080 with a desktop session on it, run 2026-08-25
[`evidence/l6-capacity-20260825/desktop3080/`]; the `ADMISSION` line is the
one the door prints for that card and that free reading today:

```
[mesh-binding] bound x1.40962: nCells=40962, nEdges=122880, dx=120000.0 m, dt=120.0 s
BIND mesh=x1.40962 rebound=True dt=120.0 s
ADMISSION mesh=x1.40962 cells=40,962 card=68 SM row=measured global predicted=5,682.0 MiB margin=884.0 MiB free=9,097.0 MiB of 10,239.5 MiB -> admitted
ARCHITECTURE sm=sm_86 -> admitted (per-architecture anchor of 2026-08-25 (evidence/sm86-tier-20260825/RECEIPT.md))
PREFLIGHT mesh=x1.40962 problems=0 status=preflight_passed
```

The card is read at the moment of the decision — `card=68 SM`, and the row
selected for it — so a 10 GiB desktop is priced as itself. On this free
reading it admits `x1.40962`, `v15.150.38857` and `u96.64002`; the refusals
earlier releases printed for those three were a 32 GiB card's arithmetic
quoted at a 10 GiB part, and 6.2 has the whole story.

On a request the card really does not hold, it says so with numbers. The
same card and the same free reading, asked for the 163,842-cell
`x4.163842`:

```
ADMISSION mesh=x4.163842 cells=163,842 card=68 SM row=measured global predicted=17,000.2 MiB margin=884.0 MiB free=9,097.0 MiB of 10,239.5 MiB -> REFUSED
PREFLIGHT REFUSED device memory admission refused --mesh x4.163842: the fitted
footprint for 163,842 cells is 17,000.2 MiB and the decision holds back 884.0
MiB, so it needs 17,884.2 MiB free; this device reports 9,097.0 MiB free of
10,239.5 MiB, short by 8,787.2 MiB. ... This card fits the registered mesh(es)
conus-x1.2971, r4.75.11020, ... u96.64002, v15.150.38857, x1.40962 at this
moment (68,440 cells fit); re-run --mesh with one of them, or free the memory
the shortfall names and re-run.
```

Preflight exits 1 on a refusal and touches no CUDA beyond the memory and
device-properties queries; a missing input is *reported* alongside the card
verdict rather than hiding it. Chapter 6.2 explains the model it decides
with.

Two more things the forecast lane needs, both stated here rather than
discovered mid-launch:

- **the gpuwm-hex source checkout** — the drivers live in `tree/tools/`,
  outside the wheel, and verify their own executing modules by SHA-256
  before CUDA is touched. Run the door from inside a checkout, or pass
  `--repo <gpuwm-hex-checkout>/tree`;
- **a `gpuwm` git checkout at the pinned commit** — `--gpuwm-checkout`. The
  run verifies the sixteen pinned files by SHA-256 at launch and refuses a
  mismatch naming the file and both digests; it also records the checkout's
  HEAD, tree and dirty paths into every receipt, which is why it is a git
  clone and not the installed distribution. The pin itself is satisfiable
  from an install at 2.5.8 — all sixteen pinned paths resolve from
  `site-packages` there, measured 2026-08-28 — so what the checkout supplies
  now is provenance, not a missing file.

## 2.7 First forecast

Same command without `--preflight`:

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
  --repo <gpuwm-hex-checkout>/tree \
  --gpuwm-checkout <gpuwm-checkout-at-v2.6.0> \
  --case-label quickstart
```

The same integration, from the init built in 2.5, took **97.8 s of
integration and 146 s wall on a 16 GiB RTX 5070 Ti**: 1 hour, 30 steps at
dt = 120 s, full physics, three history files, exit 0. That measurement was
taken through `tools/run_cuda_v841_forecast_mesh.py`, the driver this door
now drives; the door adds the admission decision in front of it and the
receipt and render command behind it, and changes nothing in between.

Re-run on 2026-08-27 on the 10 GiB RTX 3080 in a desktop, from the installed
wheel and a published `gpuwm` 2.5.6 clone: **165.4 s of integration and 244 s
wall**, same 30/30 steps, same three frames, `status=passed`.

When it finishes, `--out` holds three history files (analysis, +30 min,
+60 min), `cuda-v841-forecast-receipt.json` — the driver's receipt, stating
exactly what this run claims and, at equal prominence, what it does not —
and `forecast-receipt.json`, the door's own, carrying the admission
decision, the mesh binding, the driver's receipt embedded whole, and the
render command. The door's last two lines have the form:

```
DOOR mesh=<row> steps=<n> frames=<n> status=passed out=<out>
NEXT gpuwm-hex render --history <out>/cuda-history.<valid-time>.nc --mesh <grid> --out <out>/png --simulation-start <start>
```

## 2.8 First rendered products

Spell it out — and add `--window focus`, which the `NEXT` line does not
carry. The door's default window is `mesh`, and the `rw_mpas_convert` in
every published bundle answers `unknown render window 'mesh' (known: focus,
global)`, so the pasted `NEXT` line fails on a published engine (7.1).
`render` needs no checkout and no GPU:

```sh
gpuwm-hex render \
  --history work/fc-01/cuda-history.2026-08-12_07.00.00.nc \
  --mesh    assets/x1.40962.grid.nc \
  --out     work/png \
  --simulation-start 2026-08-12_06:00:00 \
  --window  focus \
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

Re-run 2026-08-27 on a desktop against the published 2.5.7 bridge bundle:
**49 rendered, 0 skipped, 0 failed, 7 s**, same `d01-22km` tree. The count
moved because the renderer's catalog grew, not because anything here changed.
End to end — empty machine to these pictures, including every command that
failed on the way — **16 min 46 s**, with the mesh pair and the met file
already on disk.

## Where to next

- Something refused? Chapter 8 indexes the refusals you will actually meet.
- Want your own mesh, or to know what fits your card? Chapter 4.
- Want to understand what you just ran before trusting it? Chapter 3.
