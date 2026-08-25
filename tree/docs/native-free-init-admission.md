# Native-free initialization and mesh timestep admission

This lane removes the native-MPAS init file from the **normal runtime path** without removing it as an oracle or compatibility input.

## Native-free init

A normal invocation supplies:

```text
grid.nc + static.nc + WPS intermediate + vertical-spec.json
```

The door validates the JSON declaration, constructs the MPAS-A v8.4.1 vertical contract through `mpas_port.vertical`, writes a durable NetCDF vertical artifact and receipt, and invokes `rw_mpas_init`. The artifact includes the required `ter`, `zgrid`, `zz`, `zxu`, `dss`, `dzu`, `rdzw`, `fzm`, `fzp`, `zb`, `zb3`, and surface coefficients — stored float32, the same single precision every native-lineage init carries, because the engine's emitter carries Float/Int/Char and refuses Double by name. It is mapped to both capsule/reference arguments because the current Rust ABI already checks those two inputs for vertical identity.

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

Every registered mesh declares `dt_seconds` and a versioned outer-step Courant policy. Admission reads the Earth-scaled static file’s complete `dcEdge` array, requires every value finite and positive, hashes it, records the minimum and low percentiles, and evaluates:

```text
dt <= safety_factor * min(dcEdge) / max_characteristic_speed
```

The bundled policy uses 125 m/s and a 0.90 safety factor for the outer RK horizontal transport/gravity-wave preflight. Split-explicit acoustic stability remains governed by the existing acoustic substeps; this gate is not represented as a complete nonlinear stability proof.

An unsafe declared value is refused before CUDA allocation with the actual `min(dcEdge)`, requested timestep, computed bound, and remedy. It is never silently reduced.

`x4.163842` remains an asserted no-op at 120 s. The generated `v15.150.38857` entry declares 60 s rather than inheriting x4’s constant, and must still pass its real-geometry admission.

## Evidence boundary

The CPU/no-assets tests prove schema/refusal behavior, smoothing-loop corrections, vertical invariants, edge-metric orientation, real-edge authority handling, and no-auto-shrink admission. The card-and-assets half was measured on the proving node, 2026-08-24 (`evidence/native-free-proof-20260824/`): the x1.40962 native-free mint is schema-complete against its native golden (134/134 variables), the constructed vertical sits within 3.9 mm of native `zgrid` with the field-by-field cost quantified under the same-zgrid r3 tolerances, the mint is same-session byte-deterministic, the minted init ran the dycore, and compatibility-capsule output stayed byte-identical to the 2026-08-20 evidence. Still **NOT MEASURED**: a generated-mesh (v15-class) forecast — blocked by the pre-existing v15 step-0 FloatingPointError, which fires with native statics too — and forecast skill, which is the obs referee's lane. The obs referee ran for the first time on 2026-08-25, but not on this path: all four of its cases were initialized by native `init_atmosphere`, so its verdicts say nothing about a native-free mint's skill and this row stays open.
