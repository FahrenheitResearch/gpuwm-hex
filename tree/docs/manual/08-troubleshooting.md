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
a CUDA extra, or with the wrong one. Match the driver's CUDA major
(`nvidia-smi`, header): `pip install "gpuwm-hex[gpu-cu12]"` or
`"gpuwm-hex[gpu-cu13]"`, exactly one. A CuPy built for the wrong major
imports cleanly, probes cleanly, and fails on the first real device call —
if device calls die inside CuPy on a box that "looks fine," check this
first.

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

**`unknown mesh 'NAME'; registered meshes are ['u96.64002',
'v15.150.38857', 'x1.40962', 'x4.163842']`** — the forecast runner only binds registered
meshes. Register the pair (table work in `tools/mpas_mesh_binding.py`) or
name a registered one.

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
against memory measured at that moment (chapter 6.2). The refusal names
the shortfall and, when a smaller registered mesh fits what was measured,
that mesh by name. Three real remedies, in order: free device memory
(other CUDA processes, a desktop compositor holding the card); run a
smaller registered mesh; or, if this card's own footprint ledger has been
run, pass its row as `--device-fixed-mib` **and**
`--device-bytes-per-cell` — the shipped row was measured on a 170-SM part
and smaller parts measure smaller fixed terms. Widening `--headroom-mib`
past the shortfall is not a remedy; it removes the margin the decision
holds back and the run then dies in CuPy instead.

**`--device-fixed-mib and --device-bytes-per-cell are one measured row and
must be given together`** — half a row mixes this card's fixed term with
another card's slope, which is a footprint nothing ever measured.

**`the forecast lane needs the gpuwm-hex SOURCE CHECKOUT and this is an
installed wheel`** — the drivers live in `tree/tools/`, which the wheel
does not carry. Run the door from inside a checkout or pass
`--repo <checkout>/tree`.

**`--gpuwm-checkout ... is not a directory`** — the seam pin needs a
`gpuwm` source checkout, not the installed distribution (chapter 6.1).

**`--out ... exists`** / **`--scratch ... exists`** — both must be fresh.
A second run into an existing output tree can be read as the first one's
continuation or silently mix frames from two trajectories; a stale kernel
cache can be loaded from another engine pin.

**`--scratch ... is inside the output tree`** — the kernel cache is not
output. Use the sibling default or point it outside `--out`.

**`--hours H is not a whole number of steps on mesh NAME, whose registered
timestep is T s`** / **`--history-every-minutes M does not divide the ... run`**
— the schedule is checked against the *registered row's* timestep, which is
60 s on the generated 15 km row and 120 s on the published ones. The
refusal prints a length that does divide.

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
directly), or the fitted row and this card disagree. If the door admitted
the run and CuPy then ran out, report both numbers: the fitted row is a
measurement and the driver's floor is a proof constant, and only one of
them can be wrong.

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
before suspecting your run: a ~15 % dry domain-mean precipitation bias,
higher explicit rain and condensate, and an upper-level warm drift are the
declared, measured differences from native physics, with their referees
named. And chapter 6.4: reading the receipt's nonclaims tells you what the
run never promised.
