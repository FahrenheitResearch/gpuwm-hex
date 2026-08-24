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
integrates the 163,842-cell mesh whose fine band is about 25 km.

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
  reach, `mesh-check` validates a mesh, `init` builds initial conditions,
  `render` turns model output into product images. Each door refuses bad
  input by name rather than producing a plausible wrong answer.

## What it can do

Measured, not promised (receipts in the README and under `evidence/`):

- 24-hour global forecasts on one card, deterministic across dual runs.
- Initial conditions built from a WPS intermediate file (GFS, ERA5, or
  whatever you drive with) without running any native Fortran tool, given a
  one-time native-built "capsule" for the mesh (chapter 5).
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
- **No regional (limited-area) mode.** 0.1 is global-only; use mesh
  refinement to put resolution where you want it.
- **No multi-GPU in 0.1.** Two-node execution is built and proven bitwise
  partition-invariant, but it is not part of this release.
- **No forecast front door yet.** Running the model itself requires the
  source checkout (chapter 6 states exactly why); the packaged doors cover
  doctor, mesh validation, initial conditions, and rendering.
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
| GPU | Any CUDA device for the doors. For forecasts, memory is set by mesh size — measured at the current engine pin on an RTX 5090: **6,296.5 MiB fixed + 93,474 bytes per cell** [`evidence/gf-pin-move-measured-20260824/`]. The 40,962-cell global mesh peaked at **9,948 MiB** (fits a 12 GiB card); the 163,842-cell mesh peaked at **20,902 MiB** and its run path admits at a 24 GiB free-memory floor, so it remains a 32 GiB-card configuration. The fixed term is a property of the card: smaller cards previously measured carried smaller fixed terms. |
| CPU/RAM | Modest. The doors are I/O-bound; the model state lives on the device. |
| Disk | Meshes, initial conditions, and history files are hundreds of MB to a few GB each. The optional byte-pinned test assets are about 6.9 GiB. |
| OS | Linux or Windows. Python 3.11 or newer. |
| CUDA | A driver whose CUDA major you know — the install chooses a matching CuPy wheel (chapter 2). |

Chapter 4 turns the footprint model into a sizing method for any card.
