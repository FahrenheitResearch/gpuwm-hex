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

**Version 0.1.0.** The cut shape for this line is: global variable-resolution
on one consumer GPU, deterministic, ArWen physics. Multi-GPU is built and
proven but not part of 0.1. The regional (limited-area) lane does not exist —
see *Limitations*.

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
  reference (0.2 material, not shipped in 0.1).
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
- Device footprint, measured at two meshes and both pins in one session:
  **6,296.5 MiB fixed + 93,474 B/cell** at the current pin on this card; the
  published 40,962-cell global mesh peaks at **9,948 MiB**.

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
[`evidence/EVIDENCE.md`](evidence/EVIDENCE.md)). The
verdict for each divergence is in
[`docs/declared-divergences.md`](docs/declared-divergences.md).

---

## Requirements

| | |
| --- | --- |
| Python | 3.11 or newer |
| GPU | A CUDA device. Memory is set by mesh size and by the card. Measured 2026-08-24 on an RTX 5090 at the current engine pin: `footprint = 6,296.5 MiB + 93,474 B/cell`. The published 40,962-cell global mesh (x1.40962, about 120 km) peaked at **9,948 MiB — inside a 12 GiB card's budget**; the 163,842-cell mesh (x4.163842, about 24 km) peaked at **20,902 MiB**, and its run path still admits at a 24 GiB free-memory floor, so it remains a 32 GiB-card configuration until that floor is re-derived. The fixed term is a property of the card, not the mesh; both smaller parts previously measured carried smaller fixed terms. Receipts: `evidence/gf-pin-move-measured-20260824/`. |
| CUDA | CuPy matching your driver's CUDA major — there is no way for pip to detect it, so you choose (below). |
| Engine | The `gpuwm` distribution, 2.5.5 or newer: the physics seam, and the bundle that carries the four MPAS bridge binaries both doors drive. **Plus a `gpuwm` source checkout for the forecast lane** — see *The engine pin*. |
| Assets | A mesh grid file, a matching static file, and (for the init door) a vertical-grid declaration — normally a `--vertical-spec` JSON; a native-minted init file as a capsule is the compatibility mode. **gpuwm-hex ships none of these and has no fetch path for them.** See *Assets you must supply*. |
| Rust binaries | `rw_mpas_init` for the init door; `rw_mpas_convert` and `rw_wrfbatch` for the render door. They are built from the `gpuwm` source tree — see *Building the Rust engines*. |

### Install

```sh
pip install gpuwm-hex                # then one of:
pip install "gpuwm-hex[gpu-cu12]"    # driver reports CUDA 12.x
pip install "gpuwm-hex[gpu-cu13]"    # driver reports CUDA 13.x
```

Check which you need with `nvidia-smi` — the CUDA version in its header is the
driver's major. `pip install "gpuwm-hex[gpu]"` is an alias for `gpu-cu12`.

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

The distribution is `gpuwm-hex` and every command you type is `gpuwm-hex`. The
**import namespace is still `mpas_port`** — `import mpas_port`, not
`import gpuwm_hex`. That is a known inconsistency, stated rather than hidden.

It is not left alone out of laziness. Eleven of the seventeen modules that the
full-physics proof harness pins by SHA-256 carry the literal string
`mpas_port`, and in five of them it is a `module_key=` NVRTC compile-contract
identity rather than a comment — so editing them changes the kernel-cache
identity as well as breaking the pin, and the rename costs a re-proof on a
32 GiB card against the native authority. It is scheduled for a lane that can
pay that, not sneaked into a packaging change. Scripts that import the package
directly should expect the name to change in a later release.

*(The `hex` in the name is the hexagonal Voronoi mesh the model runs on.)*

### The engine pin

gpuwm-hex depends on `gpuwm>=2.5.5`. That floor is a coarse filter, not the
wall: the port pins the engine's physics seam by the SHA-256 of **sixteen
individual gpuwm source files**, and the gap between published stamps and the
pinned bytes recurred at every cut — thirteen of the sixteen differ at
gpuwm's published 2.5.0 stamp, six at 2.5.1, and three still differ at 2.5.4
(`docs/mpas-seam.md`, `gpuwm/core/mpas_column_batch.py`,
`gpuwm/io/restart.py`), because the seam-convergence work this tree pins
landed after the 2.5.4 cut. A lower floor would let pip resolve an install
that the port then refuses at launch; 2.5.5 is the first published version
whose bytes match the manifest.

The floor also keeps a **second and independent** refusal where pip can make
it. The four MPAS bridge binaries — `rw_mpas_init`, `rw_mpas_convert`,
`rw_mpas_mesh`, `rw_mpas_static` — entered gpuwm's *bundle* at 2.5.3; their
rows landed the day after the 2.5.2 upload, so published 2.5.2 stages none of
them through `gpuwm fetch-bridges`. Both front doors drive those binaries, and
the gpuwm source tree does not publish, so a user who resolved onto a 2.5.2
engine could open **neither door** and would have no route to build what was
missing. That is a stranded install rather than a degraded one, and a
dependency floor is the only place pip can refuse it. gpuwm 2.5.5 publishes
before this distribution does, which is the one hard ordering constraint the
release carries.

**Installing `gpuwm` is necessary but not sufficient for the forecast lane.**
One of the sixteen pinned files is `docs/mpas-seam.md`, a repository document
that no wheel places in `site-packages`. The forecast lane therefore needs a
**gpuwm source checkout** at the pinned commit, passed to `gpuwm-hex forecast`
as `--gpuwm-checkout` (`--arwen-checkout` on the driver beneath it), in
addition to the installed distribution. This is stated plainly because the
alternative — discovering it as a `FileNotFoundError` deep in a launch — is the
trap; the `forecast` door refuses by name instead, and so does `gpuwm-hex
doctor`.

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
  [`evidence/EVIDENCE.md`](evidence/EVIDENCE.md)); boundary in
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
enter the bundle, and the current `gpuwm>=2.5.5` floor keeps that guarantee,
so pip cannot resolve you onto an engine that strands both doors. A
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

## Two smaller doors

`gpuwm-hex mesh-check --grid <grid> --static <static>` validates a mesh pair
before anything expensive touches it, and refuses a defective pair by name.
`gpuwm-hex oracle-gate` replays the source-extracted Fortran fixtures against
a mesh. Both are listed by `gpuwm-hex --help`.

## The forecast lane

There is no console script for it in 0.1.0, and that is a deliberate omission
rather than an oversight. The driver exists and works
(`tools/run_cuda_v841_forecast.py` in the source checkout, an arbitrary-case
runner), but it is not a door a user can walk through: it needs a `gpuwm`
source checkout at a pinned commit, its mesh and static inputs are byte-pinned
authority files with no fetch path, and the proven x4.163842 mesh admits at a
24 GiB free-device-memory floor (measured peak 20,902 MiB; the registered
x1.40962 mesh peaked at 9,948 MiB). Putting a console script on that would be
a front door on a room with no floor. Running a forecast in 0.1.0 means
working from the source checkout, and the packaging verdict says so.

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
163,842-cell mesh, cell-aligned with no interpolation:

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
   +29 % / +25 %); **net domain-mean precipitation runs about 15 % dry**. This
   is the whole-model price of the declared GF generation gap above, quantified.
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

### Regional / limited-area is not available

0.1 is global variable-resolution only. Use mesh refinement to get resolution
where you want it. The regional lane is 0.3 material and the CUDA lane refuses
it today.

### Other

- **Multi-GPU is built and proven but not shipped in 0.1** (two-node, bitwise
  partition-invariant, 1.23x on 25 GbE). It is 0.2 material.
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
| big card | `-m bigcard` | a CUDA device with about 26.4 GiB free (the tier's own gate constant; the measured x4 peak at the current pin is 20,902 MiB, and the gate has not been re-derived since the pin move) |

`GPUWM_HEX_NO_LOCAL_GPU=1` (or `GPUWM_NO_LOCAL_GPU=1`, honoured so a box
configured for the engine behaves the same) bans device contact outright.
