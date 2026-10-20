# The render door: `gpuwm-hex render`

MPAS history in, product PNGs out, entirely through the Rust path.

```
python -m hexcore.cli render \
    --history history.2025-03-15_00.00.00.nc \
    --mesh x4.163842.grid.nc \
    --out ./png \
    --simulation-start 2025-03-14_12:00:00 \
    --products all
```

## What it drives

Two Rust binaries do all field-data work; this command is orchestration only.

1. **`rw_mpas_convert`** (gpuwm `tools/rustwx/crates/rw-mpas`) reads the
   history frame(s) and the mesh, builds k-d nearest-neighbour resample
   weights for the named render window (`--window focus|global`), and writes
   one wrfout-shaped netCDF frame per history file plus a JSON receipt with
   the mesh, weights and per-frame output SHA-256 digests.
2. **`rw_wrfbatch`** (gpuwm `tools/rustwx/crates/rw-wrfbatch`) imports each
   frame into a per-frame store and draws the products. Availability is the
   renderer's own `--list-products` catalog, proven against the stored
   fields of the real import — never guessed from filenames.

There is no fallback plotter. If the Rust renderer is absent the door
refuses by name; it never draws a weather field in Python.

## Remap (`--remap`, default `auto`)

`auto` asks the converter for `--remap product` on a single run: linear
barycentric interpolation on the mesh's own Delaunay triangles for state,
first-order area-conservative remap for accumulated precipitation (each
frame records the window integral against the mesh's own), nearest cell for
categories and soil. A `--compose` render keeps the nearest-cell map (it
switches source point by point, so there is no single mesh to interpolate
across) and an explicit `--remap product` there is refused by name.
`--remap nearest` is the frozen nearest-cell map. The methods, their
measured accuracy and the per-variable table are in `products-door.md`.

## Executable resolution

| Binary | Order |
| --- | --- |
| `rw_mpas_convert` | `--convert-exe`, `$GPUWM_HEX_RW_MPAS_CONVERT`, `$MPAS_PORT_RW_MPAS_CONVERT`, `PATH` |
| `rw_wrfbatch` | `--renderer-exe`, `$GPUWM_HEX_RW_WRFBATCH`, `$MPAS_PORT_RW_WRFBATCH`, `$GPUWM_RW_WRFBATCH`, `PATH` |

The `GPUWM_HEX_*` pair is the preferred spelling: the distribution is
`gpuwm-hex` and the names a user types carry its name. The `MPAS_PORT_*` pair
is the import package's original name (`mpas_port` through 0.1.1, `hexcore`
from 0.2.0) and still works — an install line that already works is never
invalidated by a rename, and a variable that silently stops being read is the
worst possible way to learn about one. The 0.2.0 package rename deliberately
left this pair alone for that reason.

An environment variable naming a missing file is a hard error, not a fall
through — a ladder that silently skips a broken pin renders with the wrong
engine build.

## Delivered layout

PNGs are filed at render time into a fixed tree, never flat:

```
<--out>/<domain-token>/<product>/<valid-day>/<engine-filename>.png
```

The domain token comes from the engine's own filename (e.g. `d01-22km` for
the focus window), the valid day from the converter's receipt for that
frame. Facts that cannot be read file under `native_grid` / `unclassified`
/ `undated` — a picture is always somewhere nameable, never dropped.

A `render-manifest.json` is written beside the tree: engine binary digests,
mesh/weights/output SHA-256s, per-frame rendered/skipped/failed products,
placed paths and the exact invocations. A second run into the same tree
writes a timestamped manifest instead of clobbering the first.

## Scratch discipline

Converted frames, per-frame render stores and raw engine output live in a
scratch directory that is a **sibling** of `--out`
(`<out>.render-scratch` by default), never inside it. Scratch inside a
delivered png tree is a shipped defect (readers publish half-finished
temporaries); an explicit `--scratch` that resolves inside `--out` is
refused by name. Scratch is deleted after a clean run and the deletion is
reported; `--keep-scratch` retains it, and any failure retains it.

## One store for a series: `--series`

By default each frame is imported into its own renderer store. With
`--series` every frame of the render goes into ONE store and one renderer
start draws them all; each picture is then filed under the frame whose lead
its engine name carries, and each skip or failure under the frame file the
renderer names in it. That is the only arrangement in which the windowed
products (`qpf_1h`, `qpf_6h`, `qpf_12h`, `qpf_24h`, `10m_wind_run_max`,
`2m_temp_0_24h_max`, ...) can be drawn: a window is the difference or the
extreme of several stored frames, and a store per frame holds one, so every
windowed row listed as "windowed accumulations need more than one stored
whole-hour frame". Frames that do not carry distinct leads (or a single
frame) fall back to a store per frame, and the door says so. The product
door always asks for `--series`; a bare `render` keeps per-frame stores so
existing receipts keep their shape.

## Selection pass-through

* `--products all|direct|derived|heavy|windowed|slug,slug,...` — the
  renderer's own vocabulary, forwarded untouched (`var:<name>` generics
  included). `--heavy` gates the heavy family exactly as upstream.
* `--frames all|N` — frame selection within each store.
* `--window focus|global`, `--field-set full|surface`,
  `--simulation-start`, `--size WIDTHxHEIGHT` (default 1200x900).

## Refusals

Every refusal names the concrete breakage and the remedy:

* **Renderer or converter absent** — names the search ladder, states that
  no PNGs/frames exist without it, gives the exact `cargo build` line and
  the variable to set.
* **Missing `--mesh` / `--history` file** — names the file and what the
  converter cannot do without it.
* **Requested product not covered by the history** — after the real
  import, the catalog row for that product is not `renderable`: the
  refusal names the product, the catalog's own missing-fields detail and
  the WRF fields the conversion reported absent from the source history.
  The run stops before drawing anything, so a partial gallery cannot pass
  as the requested one. Group keywords (`all`, `derived`, ...) expand to
  whatever *is* renderable and never refuse.
* **Scratch inside the delivered tree** — see above.

A genuine engine `SKIPPED` (under `--products all`) is relayed per frame
and recorded in the manifest; only `FAILED` fails the run.

## Exit codes

`0` — rendered at least one product, no failures. `1` — render failures or
nothing rendered. `2` — a named refusal (any `MpasPortError`).

## Proof of record (2026-08-20)

Reference node, 24 h x4.163842 history (2025-03-14_12Z start): full catalog on
the +12 h frame, a restricted product list on the +0 h and +24 h frames,
binaries built from the gpuwm renderer workspace with `--offline --locked`.
Evidence gallery and manifest recorded with the 2026-08-20 render-door receipts.
