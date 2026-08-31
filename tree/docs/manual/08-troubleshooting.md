# 8. Troubleshooting

Every failure surface in this product is a **named refusal**: it states the
concrete breakage and its remedy. This chapter indexes the ones you will
actually meet, by the text you will see. Each refusal below was produced on
a real machine while proving this manual, except where a receipt is cited
instead.

**First move, always:**

```sh
gpuwm-hex doctor --explain
```

Doctor exits 1 while any required estate is missing and prints the whole
pasteable remedy block per gap. If doctor passes and a door still refuses,
the refusal text itself is the index below.

---

## Install and environment

**`MISSING cupy (the CUDA lane)`** — you installed the base package without
a CUDA extra, or with the wrong one. `pip install "gpuwm-hex[gpu]"`, which
is `gpu-cu13`: every GPU door here refuses a CUDA runtime below `13000`, so
that is the only wheel this port runs on.

**`MISSING cupy (the CUDA lane): cupy imports but carries CUDA runtime
12090`** — a CuPy built for CUDA 12 is installed. It imports cleanly, probes
cleanly, runs cuBLAS, and is then refused by name at forecast launch with
`CudaRefusal: cuda.runtime_version=12090 < required 13000`. Doctor prints
the removal first because pip leaves both wheels installed and import order,
not intent, picks the winner:

```
  pip uninstall -y cupy-cuda12x
  pip install "gpuwm-hex[gpu-cu13]"
```

**`MISSING cupy (the CUDA lane): this box's driver serves CUDA 12`** — not a
pip problem. `cupy-cuda13x` needs a driver serving CUDA 13, and the CUDA-12
wheel is refused by the runtime floor, so no install command opens the CUDA
lane on this machine. Update the NVIDIA driver, or run the CUDA lane on a
machine that has one. Doctor deliberately offers no `pip install` line here.

**`MISSING rw_mpas_init ... not found on any rung`** (likewise
`rw_mpas_convert`, `rw_wrfbatch`) — the Rust engines are not staged. The
refusal lists every rung it searched and both roads: `gpuwm fetch-bridges`
(supplies what your installed gpuwm's bundle actually declares — from gpuwm
2.5.3 that includes the four MPAS bridge binaries; on the older 2.5.2 the
floor now excludes it was the renderer only) and the source build:

```
  cargo build --release --locked --offline -p rw-mpas --bin rw_mpas_init
```

then `export GPUWM_HEX_RW_MPAS_INIT=<built binary>`.

**`$GPUWM_HEX_RW_MPAS_CONVERT names ..., which is not a file.`** — an
explicit setting pointing at a missing binary is a hard error, never a
fall-through:

```
gpuwm-hex: $GPUWM_HEX_RW_MPAS_CONVERT names .../rw_mpas_convert, which is not a
file.  An explicit setting is never skipped in favour of a different binary --
that is how a box runs the wrong engine and reports success.  Point it at a
built rw_mpas_convert, or unset it to continue down the ladder.
```

Fix the path or unset the variable; do not expect the ladder to route
around it.

**`GPUWM_HEX_NO_LOCAL_GPU=1`** (or `GPUWM_NO_LOCAL_GPU=1`) bans device
contact outright — GPU-gated tests and lanes skip by name and
`CUDA_VISIBLE_DEVICES` is set to `-1`. If everything GPU-shaped is
skipping, check whether one of these is set.

---

## Mesh and assets

**`--grid was not given and there is no default outside a source
checkout.`** — installed doors have no default mesh, deliberately: gpuwm-hex
ships no meshes and a guessed default would name a file that does not
exist. Pass both `--grid` and `--static` explicitly.

**`static/grid dimensions disagree: {'nCells': (40962, 163842), ...}`** —
the two files are from different meshes. Get the matching static for the
grid (or vice versa); chapter 4.2's `mesh-check` digests identify which
registered bytes you hold.

**`--mesh 'NAME' is not a registered mesh.  Registered meshes are
conus-x1.2971, r4.75.11020, … x4.163842.`** — the forecast door only binds
registered meshes. The refusal prints the whole registry — 21 rows at this
release — so a deliberately wrong name is also how you list it (chapter
4.1). Register the pair (table work in `tools/mpas_mesh_binding.py`) or name
a registered one. The binding layer states the same thing as `unknown mesh
'NAME'; registered meshes are [...]` when it is called directly.

**`cell coordination refused before CUDA allocation. Cell N has 4 edges,
below the admitted floor 5`** — the mesh carries a cell a Goldberg
polyhedron cannot have, put there by a generator defect fixed on
2026-08-26. Do not reduce `dt`: the refusal shows the same failure at the
same model time across three timesteps. Regenerate with a current engine, or
run the replacement row the refusal names (chapter 4.5). Receipt-checked,
not reproduced here — it needs the refused mesh's own multi-GiB pair
[`evidence/meshgen-coordination-20260826/RECEIPT.md`, the control leg].

**`mesh 'NAME': grid byte count N != declared M`** — the file you passed is
not the registered bytes for that name (wrong file, wrong mesh name, or
grid/static swapped on the command line). The registry pins exact digests;
`--selftest` on the runner demonstrates all three directions.

**`REFUSED: the requested mesh is N %/cell at its steepest spacing
gradient`** (`gpuwm mesh`) — the refinement is steeper than the smoothness
bound. Widen the transition (fifth field on `--refine`, in km), coarsen the
refinement, or refine a smaller area. `--allow-rough-mesh` emits it anyway
and is reported as a workaround.

**`the mesh has no size` / `no resolution was given`** (`gpuwm mesh`) — a
mesh needs both a resolution spec (`--background-km`/`--refine`/`--spec`)
and a size (`--card`/`--cells`); each refusal names the missing half.

**Static build refuses naming five missing soilgrids datasets** — the
fetch-geog nested layout trap; bridge the soilgrids subdirectories to the
WPS_GEOG root (chapter 4.4).

---

## Init door

**`the intermediate file has 34 distinct first-guess levels but
--nfglevels declares 5; declare at least 34`** — the refusal names the met
file's real level count; declare at least that.

**A physics switch not given** — every switch is required (chapter 5.3
maps each to its native namelist key). Transcribe from a captured
`namelist.init_atmosphere` if you have one; the refusal names the key.

**`the M1 oracle replays source-extracted Fortran fixtures that live in
the port's own checkout`** (`oracle-gate`) — the fixtures do not ship in
the wheel; pass `--fixtures` pointing at a checkout's oracle directory.

**Engine refusal on a native-free mint (`--vertical-spec`)** — not
expected at this release: the native-free mint is proven end to end
(schema-complete, dycore rc 0 —
[`evidence/native-free-proof-20260824/RECEIPT.md`]). If the engine
refuses one naming missing init-stream variables, the door and engine
are from different releases; align the checkout and the installed wheel
(chapter 5.1).

**Forecast driver reports changed reconstruction-coefficient bytes** —
provenance mismatch, not corruption: the init was built against different
grid/static bytes than the run is pinning (chapter 5.5).

---

## Forecast door

**`device memory admission refused --mesh NAME: the fitted footprint for N
cells is ... short by ... MiB`** — the card cannot hold the mesh, decided
against memory measured at that moment (chapter 6.2). The door reads THIS
card's multiprocessor count and prices its row from that, so the number is
not a 5090's and "measure your own card" is no longer the first move. Two
real remedies, in order: free device memory (other CUDA processes, a
desktop compositor holding the card); or run a smaller registered mesh —
the refusal names the ones that fit what was measured, and how many cells
fit. Widening `--headroom-mib` past the shortfall is not a remedy: the
margin it removes is named and measured, this card's RRTMG shortwave
workspace (which moved a pool high-water by 1,707.2 MiB when it stopped
being servable from the free list) plus 11.2 MiB of instrument convention,
and the run then dies in CuPy at the first radiation call instead. If this
card's own footprint ledger HAS been run, `--device-fixed-mib` **and**
`--device-bytes-per-cell` replace the shipped core and per-cell term with
its own — an escape hatch, not the remedy of first resort.

**`--device-fixed-mib and --device-bytes-per-cell are one measured row and
must be given together`** — half a row mixes this card's fixed term with
another card's slope, which is a footprint nothing ever measured.

**`the forecast lane needs the gpuwm-hex SOURCE CHECKOUT and this is an
installed wheel`** — the drivers live in `tree/tools/`, which the wheel
does not carry. Run the door from inside a checkout or pass
`--repo <checkout>/tree`.

**`--gpuwm-checkout ... is not a directory`** — the forecast lane needs a
`gpuwm` git checkout, not the installed distribution (chapter 6.1).

**`... is not a git working tree, and the Arwen checkout has to be one`** —
you pointed `--gpuwm-checkout` at an unpacked tarball or at `site-packages`.
The bytes there may well be right: at 2.5.8 all sixteen pinned paths resolve
from an install. What is missing is the commit — the run writes the
checkout's HEAD, tree and dirty paths into every receipt so the executed seam
source can be named — so clone the tag instead of copying the files.

**`--gpuwm-checkout ... is gpuwm 2.5.8, and this port's physics seam is
pinned to gpuwm 2.6.0`** — the checkout is a real gpuwm tree at the wrong
version. The refusal names both versions, lists which of the sixteen pinned
files moved, and prints the two commands that close it: the bounded `pip
install` and the `git clone --depth 1 --branch v<version>` for the checkout
the forecast lane needs on top of the wheel. Nothing to work out; clone the
tag it names.

The same comparison runs at install time. `gpuwm-hex doctor` hashes the
pinned files that live inside `site-packages` — at 2.5.8 that is all sixteen,
and a conforming install reads `16 of 16 pinned files are in this install and
all 16 match` — and reports **`MISSING gpuwm seam bytes`** on a wrong engine,
with the same two versions and the same
remedy — so a wrong engine is a report you get before a run, not a digest
mismatch you meet after one. Before 2026-08-27 there was no such check: the
report said *Every check passed* and exited 0 while the forecast lane was
dead (`evidence/userwalk-20260827/`).

**`--out ... exists`** / **`--scratch ... exists`** — both must be fresh.
A second run into an existing output tree can be read as the first one's
continuation or silently mix frames from two trajectories; a stale kernel
cache can be loaded from another engine pin.

**`--scratch ... is inside the output tree`** — the kernel cache is not
output. Use the sibling default or point it outside `--out`.

**`--hours H is not a whole number of steps on mesh NAME, whose registered
timestep is T s`** / **`--history-every-minutes M does not divide the ... run`**
— the schedule is checked against the *registered row's* timestep, and the
rows declare five different ones (120, 100, 75, 60 and 20 s), so do not
assume 120. The refusal prints the row's timestep and a length that does
divide.

**`--mesh 'NAME' is not a registered mesh`** — see *Mesh and assets* above;
register the row or name a registered one.

**`the CUDA lane is not importable`** / **`cupy imported but reports no CUDA
device`** — no admission decision can be made, so none is guessed. Install
the matching CuPy extra, or check `nvidia-smi` and `CUDA_VISIBLE_DEVICES`.

**`GPUWM_HEX_NO_LOCAL_GPU is set`** from the forecast door — the box has
declared that no GPU work happens on it, and the door will not open a
device even to measure it.

## Forecast lane

**Launch refusal naming a seam file with expected and found digests
(`does not match the proven manifest ... re-prove before running`), or a
dirty pinned file** — the `--arwen-checkout` is not at the pinned commit,
or its working tree is dirty in a pinned file. Check out the pinned commit
and restore the bytes; do not edit the pin.

**`requested dt=... s` timestep admission refusal** — the declared timestep
fails the geometry gate against the file's real `min(dcEdge)`; the refusal
prints the measured minimum, the computed bound, and the remedy
(`Declare dt_seconds <= ... or use a mesh with a larger real minimum
dcEdge`). It is never silently reduced.

**`dual-edge geometry refused before CUDA allocation`, naming an edge and
two cells** — the mesh has a collapsed Voronoi edge and the TRiSK tangential
terms divide by it, so it cannot integrate at any timestep (chapter 4.5).
The refusal prints the edge, its `dvEdge` and `dcEdge`, the amplification,
how many edges are below the floor, and the remedy. Regenerate the mesh;
do not reduce `dt`, which is not the lever. This replaced the unattributed
step-0 `FloatingPointError` that generated meshes used to die on
[`evidence/genmesh-dual-edge-20260824/RECEIPT.md`].

**`FloatingPointError: v8.4.1 CUDA validation flag refused the outer step
before publish`** — the model declined to publish a step it does not trust
(vertical-velocity bound 200 m/s). With `--stop-on-refusal` the run stops
receipted with its committed frames. The flag itself names no array: to find
which one went non-finite, in which cell, at which launch, run
`tools/diagnose_genmesh_nonfinite.py` with the same arguments — it wraps
every kernel with a post-launch scan and reports the first array whose
non-finite population grows, and with `--trace` the magnitude climb that
preceded it.

**Out-of-device-memory partway into a run** — the mesh does not fit the
card (chapter 4.6). This is what the forecast door's admission gate exists
to prevent, so reaching it means the door was bypassed (the driver run
directly). The door and the driver cannot disagree: both answer from
`hexcore.device_admission`, and the door forwards its own resolved
requirement into the driver's argv as `--required-free-bytes`. If the door
admitted the run and CuPy ran out anyway, that is a card the model does not
yet cover — report the card, its multiprocessor count, the cell count, the
predicted peak and where in the run it died.

---

## Render door

**Renderer/converter absent, or requested product not covered by the
history** — chapter 7.2 and 7.5: the refusal names the missing engine (with
the build line) or the product's missing fields from the real import. Group
keywords (`all`, `derived`, …) never refuse; a named product can.

**`--scratch` inside `--out` refused** — scratch may not live in the
delivered tree; use the default sibling or point it elsewhere.

---

## When output exists but looks wrong

That is not a refusal — it is chapter 3.3. Check the declared divergences
before suspecting your run: higher explicit rain, a condensate surplus and an
upper-level warm drift are the declared, measured differences from native
physics, with their referees named. And chapter 6.4: reading the receipt's
nonclaims tells you what the run never promised.

**Precipitation needs its own paragraph, because the two referees disagree in
sign and the older one is the retired one.** Against native MPAS-A this port's
domain-mean precipitation runs about **15 % dry** — a global mean against
another model, and model parity was retired as a goal on 2026-08-20. Against
observations, which is the verification of record, it is the other way round:
scored on 2026-08-25 against NCEP/EMC Stage-IV hourly QPE over CONUS, the port
is **wet in all four cases** (+0.0247 mm/h paired, 95 % [+0.0041, +0.0606]),
and its frequency bias at 1 mm/h is above one in three of the four
(1.59 / 1.35 / 1.38 / 0.77) — it rains over **too much area**.

What that means for a run that looks wrong:

- **Too wet, or rain spread over too much area.** That is the direction the
  live referee measured on this build. It is a declared property, not a sign
  your run failed. Judge it against observations rather than against a native
  run before you change anything.
- **Drier than a native MPAS-A run you are comparing with.** That is
  divergence 2 and it is expected — the port's convective rain is about a
  third below native's, with explicit microphysics making up roughly half.
  It is not evidence that the run under-produced rain in absolute terms, and
  the obs comparison says it did not.
- **Either way, do not treat four cases as a skill assessment.** Two of the
  four are the divergence cases themselves, one truncated at 23 h and carries
  the largest bias (+56.9 %), and one is +41.5 % on almost no rain
  (0.0156 against 0.0110 mm/h). The two clean, complete cases are +9.8 % and
  +2.4 %. The numbers and their limits are in
  [`docs/declared-divergences.md`](../declared-divergences.md).
