# 1. What gpuwm-hex is

## For someone who has never run an atmospheric model

An atmospheric model is a program that takes a snapshot of the atmosphere —
temperature, wind, moisture, pressure, everywhere at once — and advances it
forward in time using the equations of fluid motion. Feed it this morning's
atmosphere and it computes this evening's. The snapshot it starts from is
called the **initial conditions**; the ground it stands on — terrain height,
land use, soil type — is called the **static** data; and the arrangement of
points where the atmosphere is represented is called the **mesh**.

gpuwm-hex's mesh is not a flat rectangle over one region. It is a sphere tiled
with hexagonal cells — the whole planet at once — and the cells do not all
have to be the same size. You can make them 100 km wide over the oceans and
25 km wide over the region you care about, in one seamless mesh with no nest
boundaries. That is what "global variable-resolution" means.

The unusual part is where it runs: **one consumer graphics card**. The entire
model state lives in GPU memory and every step of the integration is CUDA.
There is no cluster, no MPI job, no supercomputer allocation. A 12 GiB gaming
card integrates the published 40,962-cell global mesh; a 32 GiB card
integrates the 163,842-cell mesh whose fine band is about 25 km; a 10 GiB
card runs a six-hour full-physics limited-area forecast cut out of a
global case.

## What it is, precisely

- **The dynamics** (the fluid-motion core) is a port of MPAS-Atmosphere
  v8.4.1, held **byte-identical** to the native model as its correctness
  anchor, and deterministic: two runs of the same case produce
  byte-for-byte identical output.
- **The physics** (radiation, clouds, rain, turbulence, land surface) is the
  ArWen engine's suite, reached through a pinned seam, so gpuwm-hex and the
  `gpuwm` distribution share one physics implementation rather than two that
  drift.
- **The tooling** is a set of doors: `doctor` reports what your install can
  reach, `mesh-check` validates a mesh, `cull` cuts a limited-area case out
  of a global one, `init` builds initial conditions, `forecast` runs the
  model, `swath` decides where a fine grid goes, `cycle` follows weather
  across cycles, `render` turns model output into product images. Each door
  refuses bad input by name rather than producing a plausible wrong answer.

## What it can do

Measured, not promised (receipts in the README and under `evidence/`):

- 24-hour global forecasts on one card, deterministic across dual runs.
- Full-physics limited-area forecasts cut out of a global case with the
  `cull` door and driven at their boundary: six hours, 1,080 of 1,080
  steps on an 11,020-cell cut, a 6,224 MiB footprint on a 10 GiB card
  [`evidence/regional-physics-20260826/`].
- Weather followed across forecast cycles: `swath` places fine grids from
  a coarse run's own fields, `cycle` cuts, forces and integrates them —
  two real cycles ran back to back, both fine forecasts completing
  1,080 of 1,080 steps [`evidence/cycling-loop-20260827/`].
- Initial conditions built from a WPS intermediate file (GFS, ERA5, or
  whatever you drive with) without running any native Fortran tool; a
  native-built "capsule" is an explicit compatibility mode, not a
  requirement (chapter 5).
- Product PNGs rendered from history files entirely through the Rust
  renderer, filed by domain, product, and valid day (chapter 7).
- New meshes at resolutions you choose, with their static fields, generated
  by the engine's `gpuwm mesh` door with no native toolchain (chapter 4).
- Checkpoint/restart with bitwise-identical continuation, proven in the
  full-physics proof harness (chapter 6).

## What it cannot do, stated up front

- **It is not bit-identical to native MPAS-Atmosphere and cannot be** —
  native is single-precision CPU; this is CUDA. Beyond the expected
  chaos-shaped divergence there are **three measured, one-signed
  differences** — the declared divergences of chapter 3. Read them before
  trusting a number.
- **No multi-GPU.** Two-node execution is built and proven bitwise
  partition-invariant, but it is not part of this release.
- **Forecasting needs two checkouts.** `gpuwm-hex forecast` is a shipped
  door — and `cycle run` drives it — but it refuses by name without a
  `gpuwm-hex` checkout for the drivers and a `gpuwm` **git** checkout at
  the pinned commit, the second for receipt provenance rather than for the
  seam bytes (chapter 6 states exactly why). The doors around it run from
  the wheel.
- **No data fetching.** gpuwm-hex ships no meshes, no static files, no
  meteorological data, and has no download command for them. Chapter 2 states
  plainly where each asset comes from.
- **Not long-range.** The measured upper-level warm drift (chapter 3)
  disqualifies this version for multi-day stratospheric work; it is bounded
  and fine at 24 h.
- **A run can stop early.** When a step fails validation — for example a
  vertical-velocity spike past 200 m/s — the model refuses to publish the
  step and stops rather than writing output it does not trust. A refusal is
  designed behavior, not a crash (chapter 3).

## What hardware it needs

| | |
| --- | --- |
| GPU | Any CUDA device for the doors. For forecasts, memory is set by mesh size *and* by the card, and the two do not combine into a line: the footprint is a per-card core, plus two physics column workspaces charged at `min(cells, tile)`, plus a per-cell term (chapter 4). Measured at the merged tip on an RTX 5090, 2026-08-26 [`evidence/memory-row-refit-20260826/node2/`]: the 40,962-cell global mesh peaked at **8,874 MiB** (fits a 12 GiB card); the 163,842-cell mesh peaked at **20,446 MiB** and admits at the measured floor — the prediction plus that card's own margin, about 21.7 GiB free at x4, computed by `hexcore.device_admission` — so it remains in practice a 32 GiB-card configuration. The core is a property of the card, and the door reads the card at the moment of the decision. |
| CPU/RAM | Modest. The doors are I/O-bound; the model state lives on the device. |
| Disk | Meshes, initial conditions, and history files are hundreds of MB to a few GB each. The optional byte-pinned test assets are about 6.9 GiB. |
| OS | Linux or Windows. Python 3.11 or newer. |
| CUDA | A driver whose CUDA major you know — the install chooses a matching CuPy wheel (chapter 2). |

Chapter 4 turns the footprint model into a sizing method for any card.
