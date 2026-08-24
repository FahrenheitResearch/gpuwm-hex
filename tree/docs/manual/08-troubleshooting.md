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

**`unknown mesh 'NAME'; registered meshes are ['v15.150.38857',
'x1.40962', 'x4.163842']`** — the forecast runner only binds registered
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

**Engine refusal on a native-free mint (`--vertical-spec`)** — expected in
this release: the constructed vertical artifact does not yet carry the
init-stream variable slots and the engine refuses every native-free mint
[`evidence/statics-330-unified-rebuild/RECEIPT.md`]. Use capsule mode
(chapter 5.1).

**Forecast driver reports changed reconstruction-coefficient bytes** —
provenance mismatch, not corruption: the init was built against different
grid/static bytes than the run is pinning (chapter 5.5).

---

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

**`FloatingPointError: v8.4.1 CUDA validation flag refused the outer step
before publish` at step 0 on a generated mesh** — the known open defect of
chapter 4.5 [`evidence/statics-330-unified-rebuild/RECEIPT.md`]; not
fixable from the command line. On the published meshes this refusal
mid-run means the model declined to publish a step it does not trust
(vertical-velocity bound 200 m/s); with `--stop-on-refusal` the run stops
receipted with its committed frames.

**Out-of-device-memory partway into a run** — the mesh does not fit the
card (chapter 4.6). On a small card this dies inside a CuPy allocation
after burning the time to get there; `tests/test_device_capacity.py` is
the cheap version of that discovery, and `--preflight-only` verifies
everything before CUDA.

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
named. And chapter 6.3: reading the receipt's nonclaims tells you what the
run never promised.
