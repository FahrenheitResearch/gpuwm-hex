# gpuwm-hex

A GPU-native global variable-resolution atmospheric model core, driven from the
ArWen engine.

The numerics are a port of MPAS-Atmosphere v8.4.1 to Python and CuPy, fully
device-resident: an unstructured global mesh with regional refinement, run on
one consumer CUDA card. The physics is not the port's own — every physics
column runs through the ArWen engine's column-batch seam, so gpuwm-hex and the
`gpuwm` distribution share one physics implementation rather than two that
drift.

*(MPAS-Atmosphere is a registered project of NCAR and LANL. This distribution
is an independent port and is not affiliated with or endorsed by them. The name
is used here only to say which model was ported.)*

**Version 0.2.0.** 0.1 was global variable-resolution on one consumer GPU,
deterministic, ArWen physics. 0.2 adds the **limited-area lane** — a
full-physics forecast on a culled regional mesh behind lateral boundary
conditions, which runs on a 10 GiB card — together with the machinery that
decides where to put a fine grid and keeps it there as weather moves:
variable-resolution mesh generation, storm-following placement, and a
coarse-then-corridor cycling loop. Multi-GPU is still not shipped; see
*Limitations*.

**New here? Start with the [User Manual](docs/manual/index.md)** — a
plain-language introduction, a quickstart in which every command was run
before it was printed, task guides for each door, and a troubleshooting
index built from the doors' own refusals. This README is the capability
contract; the manual is how you walk it.

---

## What is proven

Measured, on an RTX 5090:

- 24 h global forecasts at 720/720 steps, about 3.07 s/step, **deterministic** —
  two arms of the same run compared byte-for-byte and identical.
- 2-D Smagorinsky horizontal mixing, ported from the native Registry default and
  **on by default**, with a full A/B pass.
- Two-node multi-GPU, bitwise partition-invariant against the single-GPU
  reference. Built and proven, **not shipped** — there is no door on it.
- 702 product PNGs rendered through the Rust renderer.
- The init door reproduces 92/92 carried fields bit-identically against a native
  golden init, and an init it produced started the port in a 5-step smoke.
- The render door produced 31/31 renderable products from a native history.
- Restart: checkpoint and restore with bitwise-identical continuation — in the
  full-physics proof harness the restarted history file is byte-identical
  (same SHA-256) to the uninterrupted run's
  (`evidence/restart-step16-327/`).
- The engine-pin move onto the GF local-memory frame cut left the trajectory
  untouched: 132/132 history variables and every per-step atmosphere
  fingerprint byte-identical across 30 full-physics composite steps, against a
  1-ULP flip instrument the comparator flagged
  (`evidence/gf-pin-move-measured-20260824/`).
- Device footprint, measured at two meshes in one session at the merged tip
  (`evidence/memory-row-refit-20260826/node2/`): the published 40,962-cell
  global mesh peaks at **8,874 MiB** and the native x4.163842 at
  **20,446 MiB** on the 170 SM card. The model those points feed is **not a
  line in cell count** — it is a per-card core, plus the Grell-Freitas and
  YSU column workspaces charged at `min(cells, tile)`, plus a per-cell term,
  and it reproduces both peaks exactly (`evidence/memory-shape-20260827/`).
  Every admission gate — the forecast door, `--preflight`, and the driver's
  own floor — answers from that one surface (`hexcore.device_admission`),
  which reads the card's multiprocessor count at the moment of the decision.

Measured on an RTX 3080, a 10 GiB Ampere card:

- **A limited-area forecast runs the full physics stack** — WSM6,
  Grell-Freitas, YSU, YSU-GWDO, revised-MO, NoahMP, cloud fraction and RRTMG —
  behind lateral boundary conditions built from its own coarse parent. Six
  hours, 1,080/1,080 steps, 13 history frames on an 11,020-cell culled mesh,
  peak **6,224 MiB**, median 0.271 s/step, 343 rendered products. Against a
  global full-physics run over the same ground at t+6 h: theta 1.117 K RMS
  (r = 0.999973), precipitation r = 0.95, reflectivity r = 0.82, vertical
  velocity r = 0.621 (`evidence/regional-physics-20260826/`). The cull is
  11.0x fewer cells and 5.1x less memory than the global refined mesh that
  covers the same storm.
- **Four independently placed fine grids, over four different kinds of
  weather**, each built and run full physics in sequence on one card:
  1,080/1,080 steps every time (`evidence/four-swaths-20260827/`).
- **The cascade follows weather across cycles.** `gpuwm-hex cycle run`
  re-detected six hours on and continued all four slots — three reusing their
  mesh, one regenerating — over two cycles of one real case, 1,058 s total,
  peak 7,744 MiB. Starting a corridor from transplanted parent state instead
  of from the beginning saved **273.8 s, 43 %**, against a real baseline arm
  (`evidence/cycling-loop-20260827/`).
- **The domain-size question is measured, not assumed.** Five concentric culls
  of one parent against a no-boundary control: every field improves
  monotonically with cut width and the knee is at **1.35x** the fine core
  (`w` r 0.624 to 0.744, 2 m temperature r 0.578 to 0.852) for +27 % cells and
  +25 s of wall. That is the shipped default (`evidence/nest-ratio-20260827/`).

Also proven:

- **Earned timestep anchors at five timesteps** — 120, 100, 75, 20 and 5 s —
  seven rows in all, because an anchor certifies a *configuration* (the
  timestep together with the cumulus selection) rather than a timestep alone.
  Each was earned with two byte-identical forecasts and a same-card control.
  20 s and 5 s are the textbook values for 3 km and 750 m, so a fine mesh has
  a timestep it can defend (`hexcore.dt_admission`). **Only 120 s has a
  native reference, and only it ever can** — the rest are self-consistency
  anchors, and four of the seven rows record a divergence rather than a clean
  agreement. The lane was pinned to 120 s before this, which capped
  resolution at about 19 km.
- **Variable-resolution mesh generation**, spec-driven and deterministic, with
  a bind-time refusal for any mesh carrying a cell a Goldberg polyhedron
  cannot have. One graded mesh regenerates **bit-identical** to its registered
  bytes and completes a 6 h full-physics forecast.

## What is *not* claimed

It is not bit-identical to native MPAS and cannot be: native is single-precision
CPU, this is CUDA on sm_120. Over 24 h the two produce the same storms in the
same pattern with the same energy, with chaos-shaped divergence — and **three
bias-shaped differences that are declared divergences**. They are quantified
below, and [`docs/declared-divergences.md`](docs/declared-divergences.md)
carries the mechanism, the magnitude and the observational referee for each,
because a user deciding whether to trust a number needs them before the run,
not after.

**Physics parity against MPAS is not a goal of this project.** The port runs
ArWen's physics rather than MPAS's, so whole-model agreement with MPAS stopped
being reachable the moment that choice was made — which is why every remaining
difference below is physics-shaped. The **dynamical core stays pinned** to
native v8.4.1 and that pin is the correctness anchor. The physics is judged by
**obs-skill against MRMS and ASOS**, not by agreement with another model.

That changes the referee; it does not clear a finding. A bias that is wrong
against *observations* is still wrong. That comparison has now been made:
2026-08-25, four cases, Stage-IV precipitation, MRMS reflectivity and ASOS
surface, in receipt `evidence/obs-referee-283/` (see
receipt `evidence/EVIDENCE.md`). The
verdict for each divergence is in
[`docs/declared-divergences.md`](docs/declared-divergences.md).

---

## Requirements

| | |
| --- | --- |
| Python | 3.11 or newer |
| GPU | A CUDA device. Memory is set by mesh size *and* by the card, and not by a line through the two: the footprint is a per-card core, plus the Grell-Freitas and YSU column workspaces charged at `min(cells, tile)`, plus a per-cell term, all in `hexcore.device_admission`. Measured 2026-08-26 on an RTX 5090 at the merged tip: the published 40,962-cell global mesh (x1.40962, about 120 km) peaked at **8,874 MiB — inside a 12 GiB card's budget**; the 163,842-cell mesh (x4.163842, about 24 km) peaked at **20,446 MiB**, and every free-memory gate admits at the measured floor — the prediction plus that card's own margin, **about 21.7 GiB free at x4** — so x4 remains in practice a 32 GiB-card configuration. The core is a property of the card, not the mesh, and the door reads the card's multiprocessor count at the decision rather than assuming a 5090: the 16 GiB and 10 GiB parts each carry their own measured row, and a card nobody has measured gets a derived row that is labelled derived. A card with its own #264 ledger can still supply it with `--device-fixed-mib` / `--device-bytes-per-cell`. Receipts: `evidence/memory-row-refit-20260826/node2/`, `evidence/l6-capacity-20260825/` and `evidence/memory-shape-20260827/`. |
| CUDA | CuPy matching your driver's CUDA major — there is no way for pip to detect it, so you choose (below). |
| Engine | The `gpuwm` distribution, `>=2.5.8,<2.5.9` — a bounded range, and pip resolves it for you: the physics seam, and the bundle that carries the MPAS bridge binaries the doors drive. **Plus a `gpuwm` source checkout at `v2.5.8` for the forecast lane** — see *The engine pin*. |
| Assets | A mesh grid file, a matching static file, and (for the init door) a vertical-grid declaration — normally a `--vertical-spec` JSON; a native-minted init file as a capsule is the compatibility mode. **gpuwm-hex ships none of these and has no fetch path for them.** See *Assets you must supply*. |
| Rust binaries | `rw_mpas_init` for the init door; `rw_mpas_convert` and `rw_wrfbatch` for the render door. They are built from the `gpuwm` source tree — see *Building the Rust engines*. |

### Install

```sh
pip install gpuwm-hex
pip install "gpuwm-hex[gpu]"         # the CUDA lane; = "gpuwm-hex[gpu-cu13]"
```

**CUDA 13 is the only major this port runs on.** Every GPU door goes through
`require_cuda`, which refuses a CUDA runtime below `13000` by name, so
`cupy-cuda13x` is the wheel and `[gpu]` installs it. The `[gpu-cu12]` extra
still resolves for anyone who already types it, and what it installs imports
cleanly, runs cuBLAS, and is then refused at forecast launch with
`CudaRefusal: cuda.runtime_version=12090 < required 13000` — `gpuwm-hex
doctor` reports it as a gap before a card is opened. Check `nvidia-smi`: a
driver below CUDA 13 cannot open this lane at all, and no pip command
changes that.

```sh
gpuwm-hex version      # what is installed, and where
gpuwm-hex --help       # the doors
gpuwm-hex doctor       # what this install can actually reach
```

**Run `gpuwm-hex doctor` first.** A wheel for this project is deliberately
partial and says so: the Rust engines, the CUDA runtime and the mesh assets
cannot travel inside it. Doctor checks each of those estates for real and
prints, for every gap, the exact command that closes it on your platform.
`gpuwm-hex doctor --explain` prints the evidence and the whole pasteable
remedy block; `--json` emits the same findings as data. It exits 1 while any
required item is missing, so it works as a gate in a script.

### The import namespace

The distribution is `gpuwm-hex`, every command you type is `gpuwm-hex`, and the
**import namespace is `hexcore`** — `import hexcore`, not `import gpuwm_hex`.

It was `mpas_port` through 0.1.1 and the rename lands in 0.2.0. The old name
overclaimed the relationship. What this project keeps byte-identical is MPAS-A
v8.4.1's **dycore and mesh** — a specification, and it is pinned. It
deliberately does **not** match MPAS-A's physics, which is WRF physics run
through MPAS's own plumbing; ArWen's column-batch seam runs here instead. So
`mpas_port` put another project's name in every user's import line for a
relationship that only ever held over half the model. `hexcore` names the two
things that ARE pinned — the hexagonal Voronoi mesh and the dycore — and it
matches the distribution name.

**This is a breaking change for code that imports the package directly.** There
is no `mpas_port` alias shim, because a shim keeps the overclaiming name alive
in the import line, which is the thing the rename removes. Rewrite
`import mpas_port` as `import hexcore`; every module path underneath it is
unchanged.

*(The `hex` in the name is the hexagonal Voronoi mesh the model runs on.)*

### The engine pin

gpuwm-hex depends on `gpuwm>=2.5.8,<2.5.9` — a **bounded range**, and the
bound is the point. The port pins the engine's physics seam by the SHA-256 of
**sixteen individual gpuwm source files**, and the gap between a published
stamp and the pinned bytes has recurred at every cut. Re-measured 2026-08-28
against the real published bytes of every 2.5.x release, plus the `v2.5.5` tag
that has no release at all — JSON at
`evidence/repin-258-20260828/engine-verdicts.json`, instrument beside it:

| gpuwm | on PyPI | seam pin | `--offline` build road |
| --- | --- | --- | --- |
| 2.5.0 | yes | 10 of 16 moved | complete |
| 2.5.1 | yes | 10 of 16 moved | complete |
| 2.5.2 | yes | 9 of 16 moved | complete |
| 2.5.3 | yes | 6 of 16 moved | complete |
| 2.5.4 | yes | 6 of 16 moved | complete |
| 2.5.5 | **no — a git tag with no release** | 4 of 16 moved | **broken** |
| 2.5.6 | yes | 4 of 16 moved | complete |
| 2.5.7 | yes | 3 of 16 moved | complete |
| **2.5.8** | yes | **matches** | complete |

**2.5.8 is the only published engine whose bytes match the pin**, so it is the
floor, and with the exclusive ceiling one patch above it, it is the whole
range.

That is a narrower claim than this page carried before 2026-08-28, and the
reason is not that an engine regressed. The `moved` column is measured against
**this port's** seam manifest, so re-pinning that manifest moves every row at
once, including rows for engines cut long before it: 2.5.6 read `matches`
under the previous manifest and reads `4 of 16 moved` under this one. A row
from the old table is not comparable with a row from this one, and neither is
a row anyone patches by hand.

The consequence is worth stating rather than discovering: **this port has no
fallback engine.** If 2.5.8 were withdrawn, the answer would be to publish a
new engine and re-measure, not to widen the range.

Two facts about 2.5.5 that this page used to give as floor reasons are still
true and are now beside the point, because 2.5.5 fails the re-pinned manifest
on bytes before either one is reached: it is **a git tag with no PyPI
release** (so the `gpuwm>=2.5.5` this distribution once declared named a
version pip could not install), and its published `tools/rustwx/vendor/` is
missing a file of the vendored `cc` crate, so the source-build road below
stops there with `failed to open file .../cc/src/target/generated.rs`.

> **Why the ceiling exists, and why it excludes engines that do not exist
> yet.** Until 2026-08-27 the declaration was `gpuwm>=2.5.5` with no upper
> bound, so `pip install gpuwm-hex` resolved the *newest* published engine —
> which is by definition the one nobody has measured. Measured on a real
> install at that date (`evidence/userwalk-20260827/`): pip took 2.5.7, whose
> bytes the manifest of the day refused, and the forecast door then refused at
> launch with two digests and no version number while `gpuwm-hex doctor`
> reported the estate healthy and exited 0. A green install and a dead run.
>
> The ceiling is exclusive and sits at the first engine that is not measured
> usable, so a future 2.5.9 is excluded exactly as 2.5.7 is today. Admitting a
> new engine is a deliberate act: re-run
> `evidence/standalone-20260827/measure_engine_verdicts.py`, splice its JSON
> into the table in `hexcore.engine_pin` with
> `evidence/repin-258-20260828/render_engine_pin_table.py`, and the
> declaration follows it — `tests/test_packaging_declaration.py` fails if the
> two ever disagree.
>
> **A bare `pip install gpuwm-hex` is the whole install.** No engine
> constraint of your own is needed; if one is already in your environment,
> `gpuwm-hex doctor` compares the installed engine's bytes against the pin
> and names both the version it found and the version to install.

The range also keeps a **second and independent** refusal where pip can make
it. The MPAS bridge binaries — `rw_mpas_init`, `rw_mpas_convert`,
`rw_mpas_mesh`, `rw_mpas_static` — entered gpuwm's *bundle* at 2.5.3; their
rows landed the day after the 2.5.2 upload, so published 2.5.2 stages none of
them through `gpuwm fetch-bridges`. Both front doors drive those binaries, so
a user who resolved onto a 2.5.2 engine could open **neither door**. That is a
stranded install rather than a degraded one, and a dependency floor is the
only place pip can refuse it. Measured 2026-08-28 on the pinned engine:
`gpuwm fetch-bridges` from a published 2.5.8 install downloads
`gpuwm-bridges-v2.5.8-win-x86_64.zip` and stages **26 of 26 artifacts, each
verified against the packaged pin** — the four above, plus `rw_wrfbatch` and
the `rw_mpas_lbc` the limited-area lane needs.

*(The gpuwm source tree publishes too: `github.com/FahrenheitResearch/arwen`
carries tags `v2.5.0` through `v2.5.8` with the full tree, `docs/mpas-seam.md`
and `gpuwm/core/mpas_column_batch.py` included. The table above was taken by
cloning all nine of those tags and hashing the pinned files, so the build road
below is open to anyone, subject to the 2.5.5 vendor gap noted above.)*

**Installing `gpuwm` is necessary but not sufficient for the forecast lane.**
The `forecast` door refuses without `--gpuwm-checkout`: a gpuwm **git
checkout** at the pinned commit (`--arwen-checkout` on the driver beneath it),
in addition to the installed distribution. Clone `v2.5.8`. This is stated
plainly because the alternative — discovering it as a `FileNotFoundError` deep
in a launch — is the trap; the `forecast` door refuses by name instead, and so
does `gpuwm-hex doctor`.

**The reason for that requirement changed at 2.5.8, and which one is live
matters.** The old reason was the seam pin itself: one of the sixteen pinned
files is `docs/mpas-seam.md`, a repository document no wheel placed in
`site-packages`, so an install could not satisfy the pin at all. That is over.
Measured 2026-08-28 in a virtualenv holding only the published wheels,
`docs/mpas-seam.md` resolves under `site-packages` and this port's own seam
inspection over that install reports `checked=16, matched=16, moved=(),
absent=()` — and `gpuwm-hex doctor` prints `16 of 16 pinned files are in this
install and all 16 match` and exits 0. The door's own byte check accepts an
install root.

What still needs a checkout is **provenance**, not bytes. The run's proof
harness reads the checkout's git state and writes HEAD, tree and dirty paths
into every receipt and every history file, so the executed seam source can be
named by commit and a reader can see whether anything was uncommitted when it
ran. An installed distribution carries the bytes and no commit, and
`site-packages` is not a git working tree — measured the same day, the guard
refuses one, now by name rather than as a bare `CalledProcessError`. Retiring
it means giving the receipt an identity that does not spell a commit; that is
a named follow-up (`docs/release-checklist-0.2.md`) and a reader should not
assume it has happened. Every number in these two paragraphs, both doors'
answers and the 2.5.7 negative control are in
`evidence/checkout-reason-20260828/`.

**The init and render doors do not need any of that.** They import no `gpuwm`
at all; they drive Rust binaries.

---

## Assets you must supply

There is **no fetch path in gpuwm-hex for any of these**. No `gpuwm-hex fetch`
command exists. If a document ever implies otherwise, it is wrong.

**1. A mesh grid file** (`x1.2562.grid.nc`, `x4.163842.grid.nc`, …). Meshes
come from the MPAS-Atmosphere project's published mesh downloads, from the
`MPAS-Tools` mesh generator, or from the engine's own `gpuwm mesh` door, which
generates an icosahedral Goldberg grid **and its matching static** for a named
refinement region, sized against the measured footprint of a named card.
gpuwm-hex reads and validates meshes and registers a new pair as table work;
the generator lives in `gpuwm`.

**2. A matching static file** (`*.static.nc`) carrying terrain, land use, soil
category and vegetation interpolated onto that mesh. Static files are produced
by native `init_atmosphere_model`, or by the engine's `rw_mpas_static` (the
same writer `gpuwm mesh` drives, against a WPS geographical dataset).
gpuwm-hex does not build static files itself. The grid and static must be the
same mesh — the doors cross-check `nCells` and refuse a mismatched pair by
name.

**3. For the init door: a vertical grid, one of two ways.** The Rust init
engine does not invent the vertical grid.

- **Native-free (the normal path):** pass `--vertical-spec` with a
  `gpuwm-hex.vertical-spec/v1` JSON declaration; the door constructs the
  v8.4.1 vertical contract itself and writes a durable vertical artifact with
  its receipt. No native file is read. A new vertical configuration is a JSON
  file, not a code branch. Measured against a native golden on 2026-08-24: the
  mint is schema-complete (134/134 variables), the constructed vertical sits
  within 3.9 mm of native `zgrid` with the per-field cost quantified, and the
  minted init runs the dycore —
  receipt `evidence/native-free-proof-20260824/` (see
  receipt `evidence/EVIDENCE.md`); boundary in
  [`docs/native-free-init-admission.md`](docs/native-free-init-admission.md).
- **Native-capsule compatibility mode:** pass `--capsule`/`--reference` naming
  a native-minted init-class file; the door reads `zgrid, zz, fzm, fzp, dzu,
  rdzw, zb, zb3` and the smoothed terrain out of it and asserts the capsule's
  `zgrid` bit-identical against the reference before trusting a single level.

The two modes are mutually exclusive and there is no hidden native read.

**4. Meteorological input**: a WPS intermediate file from `ungrib` (GFS, ERA5,
whatever you drive with).

### Getting the Rust engines

Both doors are orchestration only; the field data is handled entirely by Rust
binaries built from the `gpuwm` source tree. Nothing compiled ships inside this
wheel, so the binaries are staged onto your machine one of two ways.

**The short way.** `gpuwm` publishes prebuilt bundles, and one command stages
the whole set — `rw_mpas_mesh`, `rw_mpas_static`, `rw_mpas_init`,
`rw_mpas_convert` and `rw_wrfbatch`:

```sh
gpuwm fetch-bridges
```

It writes into `~/.gpuwm/bridges`, which gpuwm-hex reads directly. No
environment variables to set. This works wherever a bundle is published for
your platform; `gpuwm-hex doctor` tells you whether it worked.

**The build.** On a platform with no published bundle, build from a checkout:

```sh
cd <gpuwm checkout>/tools/rustwx
cargo build --release --offline --locked -p rw-mpas       # rw_mpas_init, rw_mpas_convert
cargo build --release --offline --locked -p rw-wrfbatch   # rw_wrfbatch
```

Then point the doors at what you built:

```sh
export GPUWM_HEX_RW_MPAS_INIT=<gpuwm>/tools/rustwx/target/release/rw_mpas_init
export GPUWM_HEX_RW_MPAS_CONVERT=<gpuwm>/tools/rustwx/target/release/rw_mpas_convert
export GPUWM_HEX_RW_WRFBATCH=<gpuwm>/tools/rustwx/target/release/rw_wrfbatch
```

#### The resolution ladder

Every door resolves every engine the same way, best first:

1. the door's own flag (`--engine`, `--convert-exe`, `--renderer-exe`);
2. this distribution's variable, then any older spelling it has carried —
   `RW_MPAS_INIT`, `MPAS_PORT_RW_MPAS_CONVERT` and `MPAS_PORT_RW_WRFBATCH`
   still work and always will, because a rename must never invalidate an
   install line that already works;
3. `gpuwm`'s own variable (`GPUWM_RW_MPAS_INIT` and siblings) and its bridge
   directories, which is where `gpuwm fetch-bridges` stages;
4. `PATH`.

A flag or an environment variable naming a missing file is a **hard error**,
never a fall through to the next rung — a ladder that silently skips a broken
setting runs the wrong engine build and reports success. When nothing is
found, the refusal names every rung it searched and both commands above.

#### What `gpuwm fetch-bridges` can and cannot supply today

Measured against the **published** `gpuwm` 2.5.2 wheel, not a checkout: its
bundle carries `rw_wrfbatch` and **not** `rw_mpas_init` or `rw_mpas_convert`.
On that engine `gpuwm fetch-bridges` gives you the renderer and nothing that
opens either MPAS door. That measurement is why the dependency floor cleared
2.5.3 and never fell back: 2.5.3 is where the four MPAS bridge binaries
enter the bundle, and the current `gpuwm>=2.5.8,<2.5.9` range keeps that
guarantee, so pip cannot resolve you onto an engine that strands both doors. A
conforming install therefore never sees the 2.5.2 shortfall; it is recorded
here because it is the reason the floor first moved.

You are not asked to track that. `gpuwm-hex doctor` asks the gpuwm you
actually have which artifacts its bundle declares, and every refusal offers
the staging command only when it can really deliver the file — a remedy that
cannot work is worse than no remedy, because you run it, it succeeds, and the
door still refuses.

---

## Door 1: `gpuwm-hex init`

Build initial conditions without native Fortran `init_atmosphere_model`.

```sh
gpuwm-hex init \
  --met        WORK/MET:2025-03-14_12 \
  --static     assets/x4.163842.static.nc \
  --grid       assets/x4.163842.grid.nc \
  --vertical-spec verification/vertical-specs/tc55-v1.json \
  --out        run/init.nc \
  --start-time 2025-03-14_12:00:00 \
  --nfglevels 38 --nfgsoillevels 4 \
  --extrap-airtemp lapse-rate --use-spechumd no \
  --theta-adv-order 3 --coef-3rd-order 0.25 \
  --virtual-factor reproduce-fortran \
  --deep-soil-moisture reproduce-fortran \
  --landuse-table MODIFIED_IGBP_MODIS_NOAH \
  --frac-seaice yes --tsk-seaice-threshold 100.0 \
  --oned-underflow preserve
```

Writes `run/init.nc` and `run/init.nc.provenance.json` (SHA-256 of every input,
the engine binary, the argv, the engine's own receipt and the output), prints a
JSON summary, exits 0.

**Every physics switch is required and has no default.** That is deliberate:
each one changes the numbers in a file that opens cleanly and reads plausibly
either way, so a default would be a silent wrong answer. Each refusal prints the
native namelist key it corresponds to, so a captured `namelist.init_atmosphere`
transcribes without guessing. Full detail, including the sixteen named refusals:
[`docs/init-door.md`](docs/init-door.md).

What the door accepts is not a promise but a measured table:
[`docs/source-matrix.md`](docs/source-matrix.md) drives every source in the
RW-WPS registry with real bytes through intermediate, init and a
five-composite-step forecast, and records one verdict per source — a green
run with receipts, or the chain's refusal, verbatim.

`--vertical-spec` is the native-free path (see *Assets you must supply* and
[`docs/native-free-init-admission.md`](docs/native-free-init-admission.md));
`--capsule`/`--reference` with a native-minted init file is the compatibility
mode. The two are mutually exclusive.

## Door 2: `gpuwm-hex render`

History in, product PNGs out, entirely through the Rust path.

```sh
gpuwm-hex render \
  --history history.2025-03-15_00.00.00.nc \
  --mesh    assets/x4.163842.grid.nc \
  --out     ./png \
  --simulation-start 2025-03-14_12:00:00 \
  --products all
```

`rw_mpas_convert` resamples each history frame onto a render window and
`rw_wrfbatch` draws the products. PNGs are filed at render time into
`<out>/<domain>/<product>/<valid-day>/`, never flat, with a
`render-manifest.json` beside the tree carrying engine digests, per-frame
results and the exact invocations. Scratch lives in a sibling of `--out`, never
inside it, and is deleted after a clean run.

There is no fallback plotter. If the Rust renderer is absent the door refuses by
name; it never draws a weather field in Python. Full detail:
[`docs/render-door.md`](docs/render-door.md).

## Three smaller doors

`gpuwm-hex mesh-check --grid <grid> --static <static>` validates a mesh pair
before anything expensive touches it, and refuses a defective pair by name.
`gpuwm-hex oracle-gate` replays the source-extracted Fortran fixtures against
a mesh. `gpuwm-hex cull` cuts a limited-area grid, static and initial
condition out of a global case, which is what makes the regional lane cheap:
grid, static and init in about a second where a native regional init took
775 s. All three are listed by `gpuwm-hex --help`.

## The forecast lane

`gpuwm-hex forecast` is a door. It binds a registered mesh against its pinned
bytes, asks the card whether the run fits **before** the run starts, refuses
by name with numbers when it does not, and prints the render command when it
passes. `--preflight` gives the same answer without integrating anything.

Two things it still requires, and both are named refusals rather than silent
failures: a **gpuwm git checkout** at the pinned commit — the proof harness
records its HEAD, tree and dirty paths into every receipt so the executed
bytes can be named by commit, and an install has no commit — and a **gpuwm-hex
source checkout** for the drivers under `tools/`, which verify their own
executing modules by SHA-256 and therefore cannot live inside the wheel.
`gpuwm-hex doctor` reports the same two gaps. The gpuwm one is no longer about
a pinned file the wheel cannot reach: at 2.5.8 all sixteen resolve from an
install (*The engine pin*).

Its inputs are a mesh, a static file and an init. The registry makes the first
two table work; the init door makes the third. The card question is answered
from `hexcore.device_admission` — on the 170 SM part the registered
40,962-cell mesh peaks at 8,874 MiB and the 163,842-cell x4 at 20,446 MiB,
which with that card's own margin puts x4 in practice on a 32 GiB card.

### Limited area, behind lateral boundary conditions

`--lbc-dir` runs the same full physics stack on a **culled** regional mesh
driven from its parent. A cull of a placed fine grid is around eleven times
fewer cells than the global refined mesh covering the same storm, and it runs
on a 10 GiB card. The outermost seven rings are boundary data driven from the
parent every step rather than the model's own answer, and products drawn over
the whole domain include them.

### Placement and cycling

`gpuwm-hex swath` decides where a fine grid should go — detection on
sea-level-reduced pressure, a declarative threat grammar, commensurable
ranking across phenomena that carry different units, and hysteresis so a grid
does not chase noise. `gpuwm-hex cycle` runs the loop: a coarse parent, the
corridors placed inside it, and the next cycle's re-detection. See
[`docs/cycle-door.md`](docs/cycle-door.md).

### Local time stepping, opt-in

On a variable-resolution mesh most columns are far coarser than the finest one,
and the acoustic sub-step is sized for the finest. `--local-timestep` lets a
coarse column take fewer, longer acoustic sub-steps, chosen from the grid
file's own `dcEdge`.

```bash
python tools/run_cuda_v841_forecast.py \
    --grid x4.163842.grid.nc --static x4.163842.static.nc \
    --init <init>.nc --hours 6 \
    --cache-root <cache> --output <out> \
    --local-timestep
```

`--local-timestep-rates` sets the ladder (default `1,3`, two classes) and
`--local-timestep-buffer-rings` the width of the finer-rate buffer around a
class boundary (default 1).

**It is off by default and that is deliberate.** Native MPAS-A v8.4.1 has no
local time stepping: `Registry.xml:64-68` offers SRK3 only, and `dt_dynamics`,
`rk_timestep`, `rk_sub_timestep` and `number_sub_steps` are scalars at
`mpas_atm_time_integration.F:2053-2092`. There is therefore no byte-identical
implementation and there never can be. A user who turns it on takes a declared
divergence from native; a user who does not gets the pinned arithmetic
unchanged, byte for byte. This is a performance feature, not a correctness
remedy, so opt-in is the correct shape for it.

Two things follow from the mesh rather than the flag:

- On a **quasi-uniform** mesh every column lands in one class, so the option is
  inert and the run is bit-identical to a default run. Measured on x1.40962:
  all 40,962 cells in class 0, zero interface edges, identity permutation, and
  every history frame SHA-256-identical.
- The ladder is not free. A rate must divide every RK stage's acoustic
  sub-step count, and the released `(1, 3, 6)` schedule admits `1` and `3`
  only — `2` and `4` do not divide the RK2 stage's three sub-steps.

Class boundaries are refluxed, so mass and passive water vapour are conserved
to binary32 rounding rather than exactly; see *Local time stepping is
flux-conservative, not exactly conservative* under **Limitations**. A run with
the option on stays bit-reproducible run to run, so the dual-run byte
comparison that screens for memory corruption on cards without ECC still
works.

#### What it costs, measured

**On the published x4.163842 mesh the option does not pay, and the ceiling
says it cannot.** Dry lane, 163,842 columns, 55 levels, RTX 5070 Ti, one model
hour:

| | wall seconds per model step |
|---|---|
| default path | 1.268 |
| `--local-timestep` | 1.283 |

0.988x — about 1% slower. The reason is measurable rather than mysterious.
Re-timing the same arm at 12 acoustic sub-steps instead of 6 gives the cost of
one sub-step directly, and from it **the acoustic loop is 23.5% of a model
step**. This mesh admits a 23.0% acoustic saving, so the best a whole step
could do is **1.057x** before the option pays for its own bookkeeping, and the
bookkeeping is larger than that.

The ceiling is a property of the released schedule, not of this mesh. A rate
must divide every stage's sub-step count, so `(1, 3, 6)` admits only rate 3,
and a rate-3 column still runs 4 of the 10 sub-steps a fine column runs. Even
in the limit where **every** column is coarse the saving is 60% of the acoustic
loop, which at a 23.5% share is **1.16x for the whole step**. That number is
the honest cap on this feature as the port stands.

What would move it: a mesh with a much steeper resolution gradient (the
generated 15 km-in-136 km box mesh classes to a 49.7% acoustic saving, a 1.13x
ceiling), and an acoustic schedule whose sub-step counts admit a rate above 3.

#### The full-step form, measured to a no-go

The stronger form of this feature — every dycore kernel launched per rate
class, whole RK steps advancing at `rate * dt` — is not capped by the acoustic
share. Its arithmetic prize on real meshes is large: counting cell-steps under
a `(1,2,4)` ladder with one buffer ring, the published x4.163842 mesh admits
**1.47x** and the generated 15 km-in-136 km box mesh **2.73x**. Whether the
card can collect that prize is a question about launch cost: a kernel launched
over a class must cost proportionally less than a launch over the whole mesh,
and below the card's occupancy knee it does not.

`tools/probe_lts_fullstep_projection.py` measured it (RTX 5070 Ti, the port's
own pinned kernels and their landed index-list derivations, the real meshes'
own class index lists, `evidence/local-timestep/fullstep-projection.json`):

| mesh | cells | arithmetic prize | measured projection |
|---|---|---|---|
| x1.40962 uniform | 40,962 | 1.000x | 1.000x |
| x4.163842 published VR | 163,842 | 1.467x | **1.254x** |
| 15 km-in-136 km box | 38,857 | 2.725x | **0.979x** |

The uniform row is the instrument's known-answer: one class, and the
index-list launch prices within 0.2% of the pinned kernel. On the box mesh the
prize inverts into a projected slowdown because the whole mesh is already
below the occupancy knee — doubling its cells costs only 1.26–1.30x on the
cell kernels — so a launch over its 5,562-cell fine class costs nearly what
the whole mesh costs, and the fine class launches four times per macro step.
The steeper the refinement, the smaller the classes, the harder the floor
bites: `(1,2,4,8)` projects worse (0.904x), not better. Only the x4-size mesh
keeps every class above the knee, and even there the trio projects 1.254x
**before** interface bookkeeping; the shipped acoustic form measured its own
bookkeeping at about 7% of a step on that mesh, and the full-step form pays a
cost of the same character plus time interpolation at class boundaries.

So the verdict, measured rather than estimated: rebuilding the whole step
loop per class would buy roughly 1.1–1.2x on the one registered mesh large
enough to profit, and a slowdown on the small steep meshes the prize was
supposed to come from. Not worth a rewrite of every kernel's launch path.
What could reopen it: meshes an order of magnitude larger, where every class
sits above the occupancy knee, or overlapping different classes' independent
work in concurrent streams — neither is measured here, and the probe is the
instrument to re-run when either becomes real.

---

## Limitations

Read these before trusting a number.

### The GF convection scheme is a different generation from native

**Declared, measured, and not a defect in the seam.** The seam-level non-parity
was closed: the four auxiliary forcing lanes, shallow-on, and per-cell `dx` all
reach the scheme the way native feeds them. What remains is that the port's
Grell-Freitas body is **WRF v4.6.1's Freitas-2018 generation**, while MPAS
v8.4.1's `module_cu_gf.mpas.F` is the **2013 ensemble fork**. Verified by source
count on both sides:

- native has zero occurrences of `dicycle` and zero of `tau_ecmwf` — those
  closures do not exist in it at all;
- native carries Fritsch-Chappell `AA0/1200s` closure members the port does not;
- native runs `c0=.002`; the port runs a temperature-scaled `c0=.004`;
- native's shallow scheme is non-precipitating (`c0=0`); the port's shallow
  scheme folds `prets` into `pratec`.

Closing this means porting native's `cup_gf`/`cup_gf_sh` bodies. That is a
program, not a seam edit — and whether it should close is the referee's call.
Every run receipt carries `gf_native_parity_claim: false` next to
`gf_declared_divergence` naming exactly the above, and so does every history
file written. The field stopped being called a blocker when parity was
retired as a goal: this is a declared property of the product, judged by
obs-skill.

### Local time stepping is flux-conservative, not exactly conservative

**Only reachable with `--local-timestep`, which is off by default.** When two
neighbouring columns advance on different acoustic sub-steps, the mass each
carries across the edge between them is integrated at two different rates and
the two no longer cancel. Mass is then created or destroyed at every class
boundary, and that is what kills local time stepping in practice.

The remedy here is Berger-Colella refluxing on the acoustic mass flux: the
coarse column keeps predicting with its own sub-step, the interface edge
accumulates the fine-rate integral minus that prediction, and the residual is
handed to the coarse column at the end of the RK stage. No term is dropped and
none is double counted, so the statement is **conservative in the flux and
approximate at binary32 rounding** — the coarse side applies one rounded sum
where the fine side applied a sequence of rounded increments.

Measured, not asserted. Dry lane, water vapour a passive scalar, published
x4.163842 variable-resolution mesh (5.46 max/min spacing, 1,063 interface
edges), one model hour:

| arm | dry-mass drift | passive-qv drift |
|---|---|---|
| default | 3.09e-11 | 9.02e-10 |
| `--local-timestep` | 2.58e-11 | 6.36e-10 |

against a 2.0e-8 bound. A full-physics qv budget cannot decide this: water
vapour there has sources and sinks and drifts about 1.8e-4 over six minutes
from the microphysics alone.

### Three bias-shaped differences against native — declared divergences

Not blockers, and not evidence of a broken dycore. They are the measured price
of running ArWen's physics instead of MPAS's, stated so a user knows what they
are getting. The obs-skill comparison, which is the verification of record for
physics here, ran on 2026-08-25 and reaches two of the three; the register of
all three — the mechanism, the magnitude, the named observational referee and
now what that referee said — is
[`docs/declared-divergences.md`](docs/declared-divergences.md).

Measured over 24 h, two independent weather cases, two mixing regimes,
163,842-cell mesh, cell-aligned with no interpolation.

**When, exactly — because these three magnitudes carry no date of their own.**
They entered this repository already finished on 2026-08-20, and there is no
receipt directory, no card and no run commit for them anywhere in the tree.
What can be established is a ceiling: the commit that introduced them pinned
engine `629ddb6f0`, so they were measured at or before that pin. Three engine
pin moves have landed since — `0d04db712` (2026-08-24), `26daaab7e`
(2026-08-25) and `659962929` (2026-08-28). The **last** of the three is
measured to change nothing: a four-arm byte A/B on one card found the old and
new pins identical on the atmosphere half of the per-step fingerprint at all
31 steps and on 0 of 138 history variables
(`evidence/seam-258-ab-20260828/`) — one mesh, one case, one hour. The two
earlier moves have no such arm, and none of the three magnitudes has been
re-measured over 24 h since. Whether any of them moved is **NOT MEASURED**.
Read them as the numbers from the pre-2026-08-20 engine, not as today's.

1. **Upper-level warm drift.** Above level 45 the port warms relative to native
   at **+0.019 K/h, near-linear, one-signed**, reaching **+0.46 K at 24 h** —
   and it is identical in both weather cases and both mixing regimes. Case
   independence means it is a code path, not weather. The legacy-RRTMG radiation
   lane is the prime suspect. Fine at 24 h; extrapolated (not measured) a 7-day
   run carries about +3.2 K of stratospheric error, which disqualifies this
   version for long-range work.
2. **Convective-to-explicit precipitation repartition.** The port's GF produces
   about **a third less convective rain** (`rainc` -36 % / -34 % in the two
   cases); explicit microphysics makes up roughly half of it (`rainnc`
   +29 % / +25 %); **net domain-mean precipitation runs about 15 % dry against
   native MPAS-A**. This is the whole-model price of the declared GF
   generation gap above, quantified.

   **That number belongs to the retired referee, and the live one returned the
   opposite sign.** "15 % dry" is a global domain mean against native MPAS-A,
   and physics parity with native was retired as a goal on 2026-08-20. The
   verification of record is skill against observations; it ran on 2026-08-25
   against NCEP/EMC Stage-IV hourly QPE over a CONUS window and found the port
   **wet in all four cases** — paired case-block estimate **+0.0247 mm/h**,
   95 % interval **[+0.0041, +0.0606]**, 5,000 replicates — with the frequency
   bias at 1 mm/h above one in three of the four (1.59 / 1.35 / 1.38 / 0.77),
   so it rains over too much area rather than too little.

   **Read the obs result with its own limits: four cases is not a skill
   assessment.** Two of the four are the divergence cases themselves. One
   truncated at 23 h and carries the largest bias, +56.9 %. One is +41.5 % on
   almost no rain — 0.0156 against 0.0110 mm/h, an absolute difference of
   +0.0046 mm/h. The two clean, complete cases are +9.8 % and +2.4 %. Neither
   verdict cancels the other, because they are not the same domain or the same
   statistic: one is a global mean against a model, the other a CONUS window
   against gauge-and-radar analysis. What is settled is that being drier than
   native does not mean being drier than the atmosphere.
3. **Downstream condensate surplus.** With more rain made explicitly, the port
   carries **+50 % cloud water and +62 % rain water** in the domain mean by
   24 h, with much heavier point extrema (max-cell 24 h precipitation 502 vs
   308 mm). Probably a consequence of (2); it should be re-measured after any GF
   fix before anyone touches microphysics.

Everything else in the comparison is chaos-shaped — symmetric, growing with lead
time, driven by convective cells landing in different places, with envelope
statistics that match (peak updraft 11.49 vs 11.26 m/s; domain means within
0.05 %). That shape is expected between a single-precision CPU model and a CUDA
port and proves nothing wrong. The three above are one-signed and
case-independent, which chaos cannot be, and each names a lane to fix.

### What the limited-area lane does not do yet

> **The published-engine gap this section used to open with is closed, and
> this is what replaced it.** Through gpuwm 2.5.7 the lane could not be opened
> from published artefacts at all: no published `rw_mpas_mesh` carried
> `--cull-parent`, and `rw_mpas_lbc` was in no bundle and no published source
> (measured 2026-08-27, `evidence/userwalk-20260827/RECEIPT.md` — that receipt
> is a record of 2.5.7 and is not restated here as current fact). The engine
> then published 2.5.8, which this distribution requires.
>
> Measured 2026-08-28 against the real published artefacts: `gpuwm
> fetch-bridges` on a 2.5.8 install downloads
> `gpuwm-bridges-v2.5.8-win-x86_64.zip` and stages 26 of 26 artifacts against
> its packaged pins, `rw_mpas_lbc` among them; and `gpuwm-hex cull` drove the
> **staged published** `rw_mpas_mesh` through two real cuts of the
> 40,962-cell global parent — 338 and 606 cells, grid, static and initial
> condition each written, 0.9 s.
>
> **Still unmeasured from published artefacts:** a complete boundary set
> written by the published `rw_mpas_lbc`, and a `--lbc-dir` forecast behind
> it. That binary runs and refuses correctly by name on inputs that do not
> satisfy it, and no parent history stream carrying the edge-normal wind over
> a cullable region was to hand to drive it further. Every limited-area number
> in this file was taken with a source-built engine, and none of them has been
> reproduced from the published bundle.

The lane runs, and these are its edges rather than its promises.

- **One parent, windowed.** A cycle is one parent integration read at
  successive times, not a parent regenerated per cycle. Regenerating it is the
  operational remedy for a corridor that has moved far, and it is not built.
- **Two of four admitted slots per cycle** are skipped as background culls by
  a measured minimum-edge-length ratio, so a cycle can place fewer corridors
  than it detected.
- **Hour zero has no ice.** A corridor started from transplanted parent state
  begins with no cloud ice, snow or graupel, because the initial-condition
  stream carries no slot for them. Reflectivity does not correlate at hour
  zero; one hour on, the microphysics has re-formed the ice and r = 0.863.
  Temperature agrees to five decimals throughout.
- **No obs-skill score on a cycled case.** The limited-area verdicts above are
  against a global run over the same ground, not against observations.

### Other

- **Multi-GPU is built and proven but not shipped** (two-node, bitwise
  partition-invariant, 1.23x on 25 GbE). There is no door on it, and it has
  not been re-proven at the current engine pin.
- **A case has been seen to refuse mid-run** on a vertical-velocity divergence
  at levels 44-47. The refusal is the model declining to publish a step it does
  not trust, which is the designed behaviour, but it means a given case can stop
  early rather than produce a bad forecast.
## Licence and derivation

gpuwm-hex is **Apache-2.0**, the same licence gpuwm ships under.

It reimplements the dynamical core of MPAS-Atmosphere v8.4.1 (LANL/UCAR) for
CUDA devices, with pinned source-line citations. That upstream is BSD-3-Clause,
whose terms govern the MPAS-derived portions and travel with every copy of this
one: `NOTICE` reproduces the MPAS licence in full and carries the marking it
requires of derivative works. `LICENSE` and `NOTICE` both ship inside the wheel
and the sdist.

This is **not** the version available from LANS and UCAR, and neither they nor
their contributors endorse it. Results from gpuwm-hex are not results from
MPAS-Atmosphere; where the two are known to differ, the differences are the
measured ones stated above.

---

## Development

Tests run from this directory:

```sh
PYTHONPATH=src python -m pytest tests -q
```

Three tiers gate themselves and each names why it is skipping — see
[`tools/battery/README.md`](tools/battery/README.md) for what each covers and
how to run the gated ones:

| tier | selector | needs |
| --- | --- | --- |
| unit + packaging | `-m "not gpu and not bigcard and not assets"` | nothing but Python |
| assets | `-m assets` | about 6.9 GiB of byte-pinned mesh/static/init/native-history files |
| big card | `-m bigcard` | a CUDA device clearing the measured x4 floor — about 21.7 GiB free (the measured 20,446 MiB peak plus that card's own margin, computed by `hexcore.device_admission`; re-fitted at the merged tip 2026-08-26 and re-shaped 2026-08-27, `evidence/memory-row-refit-20260826/`) |

`GPUWM_HEX_NO_LOCAL_GPU=1` (or `GPUWM_NO_LOCAL_GPU=1`, honoured so a box
configured for the engine behaves the same) bans device contact outright.
