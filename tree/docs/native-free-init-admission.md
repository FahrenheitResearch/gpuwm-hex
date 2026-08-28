# Native-free initialization and mesh timestep admission

This lane removes the native-MPAS init file from the **normal runtime path** without removing it as an oracle or compatibility input.

## Native-free init

A normal invocation supplies:

```text
grid.nc + static.nc + WPS intermediate + vertical-spec.json
```

The door validates the JSON declaration, constructs the MPAS-A v8.4.1 vertical contract through `hexcore.vertical`, writes a durable NetCDF vertical artifact and receipt, and invokes `rw_mpas_init`. The artifact includes the required `ter`, `zgrid`, `zz`, `zxu`, `dss`, `dzu`, `rdzw`, `fzm`, `fzp`, `zb`, `zb3`, and surface coefficients — stored float32, the same single precision every native-lineage init carries, because the engine's emitter carries Float/Int/Char and refuses Double by name. It is mapped to both capsule/reference arguments because the current Rust ABI already checks those two inputs for vertical identity.

Because the engine lays out its output from the capsule schema verbatim (and refuses a computed value with no variable to land in), the artifact also declares the full met-state landing-site schema (`theta`, `rho`, `u`, `w`, the surface/soil fields, `initial_time`; `Time` unlimited, four soil levels) as float32 zeros the engine overwrites, and it fills the derived mesh geometry that static files carry as zero placeholders — `edgeNormalVectors`, `localVerticalUnitVectors`, `cellTangentPlane`, `coeffs_reconstruct` — using the port's frozen transcriptions of `mpas_vector_operations.F` / `mpas_vector_reconstruction.F` / `mpas_rbf_interpolation.F`, since on this path the door is the initialization that native MPAS would have performed. All three closures were measured against the sibling engine and forecast driver on real x1.40962 assets (2026-08-24); the artifact receipt names what was declared and what was completed.

```bash
gpuwm-hex init \
  --met MET:YYYY-MM-DD_HH \
  --grid MESH.grid.nc \
  --static MESH.static.nc \
  --vertical-spec verification/vertical-specs/tc55-v1.json \
  --out init.nc \
  ...all existing explicit physics/build switches...
```

A native file is opened only under explicit compatibility mode:

```bash
gpuwm-hex init ... --capsule NATIVE_INIT.nc --reference NATIVE_INIT.nc
```

The two modes are mutually exclusive. There is no hidden native read in validation.

## Vertical declarations

Schema: `gpuwm-hex.vertical-spec/v1`.

The declaration controls the native `tc`, `legacy`, or specified-interface branch; level count and model top; interface projection; terrain and coordinate-surface smoothing; minimum accepted layer fraction; hybrid or basic terrain-following coordinate; hybrid transition; damping; theta-advection order; and third-order coefficient.

A new vertical configuration is a JSON file, not a new branch in Python.

## Timestep admission

Timestep admission is TWO gates and a run passes both or it does not start: the geometry/Courant gate below, and `hexcore.dt_admission`, which asks the separate question of whether anyone has ever integrated this configuration at this timestep.

### The geometry gate

Every registered mesh declares `dt_seconds` and a versioned outer-step Courant policy. Admission reads the Earth-scaled static file’s complete `dcEdge` array, requires every value finite and positive, hashes it, records the minimum and low percentiles, and evaluates:

```text
dt <= safety_factor * min(dcEdge) / max_characteristic_speed
```

The bundled policy uses 125 m/s and a 0.90 safety factor for the outer RK horizontal transport/gravity-wave preflight. Split-explicit acoustic stability remains governed by the existing acoustic substeps; this gate is not represented as a complete nonlinear stability proof.

An unsafe declared value is refused before CUDA allocation with the actual `min(dcEdge)`, requested timestep, computed bound, and remedy. It is never silently reduced.

`x4.163842` remains an asserted no-op at 120 s. The generated `v15.150.38857` entry declares 60 s rather than inheriting x4’s constant, and must still pass its real-geometry admission.

### The anchor gate

Geometry admitting a timestep is not evidence that anything has been integrated at it. `hexcore.dt_admission` is an earned-anchor registry keyed by CONFIGURATION — the timestep together with the cumulus selection and the surface/PBL cadence, not by timestep alone and not per mesh digest — because how often a scheme is CALLED is part of what an anchor’s forecasts measured. WRF pins `cudt = 0` for Grell-Freitas, so the cumulus cadence IS the timestep, and switching convection off changes that from every step to never. The registry holds seven rows over five timesteps: 120, 100, 75, 20 and 5 s with Grell-Freitas, plus 20 and 5 s with convection off. An anchor is a property of the timestep and not of the mesh, so every mesh whose own Courant limit clears it inherits it — and `v15.150.38857`’s declared 60 s is covered by no row, which is a second refusal on top of its geometry.

A row is EARNED rather than declared. Each names a schedule receipt — the host-derivable half: cadence integrality, the RK schedule shape against the proven 120 s one, the WSM6 minor-loop count, and a proof that the run’s step endpoints are exact in binary64 — an integration anchor of real forecasts at that timestep on a named card, finite at every step, two runs byte-identical under masked digests, and a `physics_health` verdict measured against a 120 s control on the same card, mesh and init. Two rows read TRACKS and four read DIVERGES: determinism and physical plausibility are different questions, and a row recording only “two byte-identical forecasts” would have said the same thing about all six. Only 120 s carries a native reference and only it ever can — the one native MPAS-A v8.4.1 integration this program holds ran at 120 s — so every other row records `native_reference=None` rather than being quietly conflated with it.

A configuration holding no row is refused before any allocation, by name, and the refusal names a measured breakage rather than an asserted one: a mesh declaring 100 s bound clean, allocated 18,820 MiB, spent 285 s and died inside composite step 0 on `post-RK candidate time must equal the exact step endpoint: 120.0 != 100.0`. Minting a candidate is reachable on purpose — under a verbatim authorization string, stamped `CANDIDATE-UNANCHORED` in the receipt so the anchor verifier can never certify it — but registering the result moves the frozen lane off a proven timestep, which is a ruling and not an agent’s edit.

## Evidence boundary

The CPU/no-assets tests prove schema/refusal behavior, smoothing-loop corrections, vertical invariants, edge-metric orientation, real-edge authority handling, and no-auto-shrink admission. The card-and-assets half was measured on the proving node, 2026-08-24 (`evidence/native-free-proof-20260824/`): the x1.40962 native-free mint is schema-complete against its native golden (134/134 variables), the constructed vertical sits within 3.9 mm of native `zgrid` with the field-by-field cost quantified under the same-zgrid r3 tolerances, the mint is same-session byte-deterministic, the minted init ran the dycore, and compatibility-capsule output stayed byte-identical to the 2026-08-20 evidence. The generated-mesh row is CLOSED: `u96.64002` is ours end to end — generated grid, generated static, native-free init, no native binary at any stage — and ran 1 h and 6 h of full physics to rc 0, finite at every step (2026-08-24, `evidence/genmesh-dual-edge-20260824/`); a generated GRADED mesh has forecast since, `v20.80.151649` completing 6 h at 180/180 steps rc 0 in two runs with byte-identical history, from geometry that regenerates bit-identical to its registered bytes. The block that remains is narrower than “v15-class” and it is not an init question: `v15.150.38857` carries 61 dual edges under the 0.02 admission floor, worst `dvEdge/dcEdge` 1.685e-4, so the TRiSK tangential terms amplify the potential-vorticity gradient 5,935x on one edge and the first non-finite value is `exner` at an immediate neighbour of its cells. No timestep is a lever against that, it fires with native statics too, and the mesh is now refused at bind by name rather than dying on a flag that names nothing. Still **NOT MEASURED**: forecast skill, which is the obs referee's lane. The obs referee ran for the first time on 2026-08-25, but not on this path: all four of its cases were initialized by native `init_atmosphere`, so its verdicts say nothing about a native-free mint's skill and this row stays open.
