# 3. Concepts

Five ideas carry the whole product. This chapter states each one plainly.

## 3.1 Mesh, static, init

A run stands on three files, and they must agree with each other:

- **The mesh** (`*.grid.nc`) is the geometry: the centers, edges and
  vertices of the hexagonal cells tiling the sphere, and the connectivity
  between them. It says *where* the atmosphere is represented and how finely,
  and nothing about what the ground or the air is like.
- **The static** (`*.static.nc`) is the ground: terrain height, land use,
  soil category, vegetation fraction, gravity-wave-drag roughness — physical
  facts interpolated onto that exact mesh. A static built for one mesh is
  meaningless on another, which is why every door cross-checks the pair and
  refuses a mismatch by name.
- **The init** (initial conditions) is the atmosphere at the start moment:
  temperature, wind, moisture, pressure and soil state on that mesh at one
  valid time, interpolated from a meteorological source (GFS, ERA5, …) plus
  the vertical grid the model integrates on.

The mesh and static are reusable for years; the init is per-forecast.

## 3.2 The doors

A *door* is a command a user can walk through: it validates its inputs,
names any refusal, does its work through proven engines, and writes a
receipt. The 0.1.0 doors are `gpuwm-hex version`, `doctor`, `mesh-check`,
`oracle-gate`, `init`, and `render`. The engine side contributes
`gpuwm mesh` (mesh + static generation) and `gpuwm fetch-bridges` (engine
staging).

Two things are deliberately *not* doors in 0.1.0:

- **The forecast lane** — the model run itself — lives in the source
  checkout (`tools/run_cuda_v841_forecast.py` and the registered-mesh
  runner). A console script on a path that requires a pinned source
  checkout and multi-GiB unfetchable authority files would be a front door
  on a room with no floor. Chapter 6 walks the lane as it actually is.
- **Fetching.** Nothing in gpuwm-hex downloads meshes, statics, or
  meteorological data.

The doors' data paths are Rust (`rw_mpas_init`, `rw_mpas_convert`,
`rw_wrfbatch`); the Python layer orchestrates, validates and writes
provenance. There is no fallback plotter and no Python weather-field
rendering: if the renderer is absent, the render door refuses by name.

## 3.3 Why the dycore is pinned and the physics is refereed by observations

Two different correctness standards apply to the two halves of the model,
and the difference is the most important thing in this manual.

**The dynamical core is pinned.** The dynamics — the fluid-motion numerics —
is a port of MPAS-Atmosphere v8.4.1, and it is held byte-identical to the
native source's arithmetic as its correctness anchor, kernel by kernel, with
pinned source-line citations. Byte-identity is achievable there, it was
achieved, and every change to the executing modules is guarded by SHA-256
pins so it cannot drift silently. When this manual says "deterministic," it
is backed by dual runs compared byte for byte.

**The physics is not MPAS's, so agreement with MPAS is not its standard.**
gpuwm-hex runs the ArWen engine's physics suite through a pinned seam.
Whole-model agreement with native MPAS stopped being reachable the moment
that choice was made — so physics parity with MPAS is **retired as a goal**,
and the physics is judged by **skill against observations** (MRMS
precipitation and radar products, ASOS surface stations). Matching another
model's choices was never evidence of being right; matching the atmosphere
is.

**A declared divergence** is the stated-in-full form of a known difference: measured
magnitude, named mechanism, named observational referee. There are three
([`docs/declared-divergences.md`](../declared-divergences.md) is the
register; the README quantifies them):

1. **Upper-level warm drift** — above level 45 the port warms at +0.019 K/h
   relative to native, one-signed, +0.46 K at 24 h. Fine at 24 h;
   disqualifying for long-range work. Referee: a vertical-profile referee
   (radiosondes or a pinned analysis) — MRMS/ASOS cannot see it.
2. **Convective-to-explicit precipitation repartition** — the port's
   Grell-Freitas convection is WRF v4.6.1's 2018 generation, native's is
   the 2013 ensemble fork; about a third less convective rain, roughly half
   made up by explicit microphysics, net domain-mean precipitation about
   15 % dry. Referee: hourly precipitation — declared as MRMS, run against
   Stage-IV, because the shipped MRMS door decodes reflectivity only.
3. **Downstream condensate surplus** — +50 % cloud water, +62 % rain water
   in the domain mean by 24 h, heavier point extrema; a consequence of (2).
   Referee: MRMS reflectivity and the precipitation extreme tail.

None of the three is softened by declaring it. Two of them have now been
judged: the obs-referee ran for the first time on 2026-08-25, four cases
scored against Stage-IV precipitation, MRMS reflectivity and ASOS surface
reports ([`docs/obs-referee.md`](../obs-referee.md),
receipt `evidence/obs-referee-283/`; see
[`evidence/EVIDENCE.md`](../../evidence/EVIDENCE.md)). (2) is scored
against real rainfall; (3) is scored on its precipitation-extreme half, and its
reflectivity half cannot be scored at all against this build, because the
history stream carries no reflectivity field to compare. (1) stays outside what
these instruments can see. A bias that is wrong against observations is still
wrong — declaring MPAS no longer the referee changes the referee, it does not
clear the finding. Every run receipt and every history file carries
`gf_native_parity_claim: false` beside `gf_declared_divergence` so the
declaration travels with the data.

## 3.4 What a refusal is

A refusal is the model or a door **declining to publish a result it does not
trust**, before that result can be mistaken for a good one. Refusals are the
product's central safety behavior, and they follow two laws:

- **A refusal names the concrete breakage it prevents.** Not "invalid
  input" — the actual consequence: *"the intermediate file has 34 distinct
  first-guess levels but --nfglevels declares 5; declare at least 34."*
- **A refusal names its remedy.** Every doctor gap prints the command that
  closes it. Every missing-engine refusal prints the build line and the
  variable to set. A remedy is only offered when it can actually deliver —
  a remedy that succeeds while the door still refuses is worse than none.

This extends into the integration itself: if a step fails validation — for
example vertical velocity past 200 m/s at some level — the run refuses to
publish that step and stops, rather than writing a frame it does not trust.
A case has been seen to stop early this way (README, *Limitations*). The
forecast driver's `--stop-on-refusal` writes the receipt for the frames
already committed instead of aborting with no receipt; no validation is
relaxed. Chapter 8 is organized around refusals for exactly this reason:
the refusal text is the product's own index into its remedies.

## 3.5 Receipts

Every door writes down what it did and what it claims: the init door's
`*.provenance.json` (SHA-256 of every input, the engine, the argv, the
output), the render door's `render-manifest.json` (engine digests, per-frame
results, exact invocations), the forecast's `cuda-v841-forecast-receipt.json`
(the claim, and at equal prominence the non-claims and dropped guarantees).
Measurement campaigns commit their receipts under `evidence/`. When this
manual cites a number, the bracket names the receipt; if you cannot find a
receipt for a claim, treat the claim as wrong and say so.
