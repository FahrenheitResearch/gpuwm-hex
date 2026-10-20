# Stochastic physics on the global mesh

Both forecast physics routes accept `--stochastic JSON`. Single forecasts
default to SPPT and SKEB off. Global and hourly ensembles default to both on
when they have more than one output member and fewer than ten source members.
Ten source members default to off. The comparison required for qualification
must therefore explicitly turn both on for its stochastic arm.

For example, append this to a forecast command:

```sh
--stochastic '{"sppt":true,"skeb":true,"member":3,"seed":711}'
```

Global ensembles assign `member` from the source roster and accept
`--stochastic-seed 711`. Their JSON cannot replace member identity.
Resident forecast rosters accept a `stochastic` object in each record.
Hourly manifests accept manifest-wide and per-member `stochastic` objects,
`stochastic_seed`, and `source_members` when source cardinality differs from
the selected member count. Named rosters derive identity from the member
name, preserving seeds through reordering and card grouping. Explicit
booleans override the ensemble default. The control arm uses
`{"sppt":false,"skeb":false}`.

Overrides merge by option, preserving shared values a member does not name.
For hourly manifests the order is manifest `stochastic`, manifest forecast
flags, member `stochastic`, then member forecast flags. Within one scope,
the last `--stochastic` occurrence wins for the options it names. For resident
rosters a record's `stochastic` options override the shared forecast options,
including when a resident group contains only one member. A member can turn
SPPT off while retaining the shared seed, harmonic degree and time scale.
Named resident members keep their identity when a group is reduced to one
record, including a retry after admission splits a group. With explicit
stochastic settings, that member's derived seed stays the same through
reordering, selection and singleton grouping. A singleton roster still
defaults both schemes off when no explicit switches are supplied.

The pattern uses normalized real spherical harmonics, excluding degree zero,
on cell centers and vertices. CUDA builds the basis once per mesh and degree;
resident members sharing immutable mesh arrays share this resource. GPU
Philox4x32-10 generates independent coefficient innovations. cuBLAS evaluates
the harmonic sums. The spectrum is proportional to
`exp(-l*(l+1)*(length_m/radius)**2)`, normalized to unit point variance by the
addition theorem. `length_m` is the spectral length parameter, not a measured
geodesic e-fold distance. AR(1) memory is `exp(-dt/time_s)` with a stationary
first draw. The default degree is 48, length 500 km, time six hours, amplitude
0.5 and clip 1. These are initial settings awaiting spread-skill qualification.
Normalization cancels the common degree-one exponential before taking the
sum, retaining a finite dipole limit at long correlation lengths. The AR(1)
innovation uses `sqrt(-expm1(-2*dt/time_s))` directly so a rounded memory
coefficient of one does not erase a representable innovation. Repeatability
comparisons require the same build as well as the same member and seed.

Basis row offsets and harmonic work-item indices use 64-bit arithmetic.
A valid basis can exceed the signed 32-bit limit of float words. The Linux
host regression compiles the actual CUDA source with syntax shims and checks
one protected sparse row beyond that limit against the addition theorem for
every degree. This checks addressing and arithmetic without allocating a full
basis. It does not establish GPU execution or performance.

The ArWen door and driver memory floor add a component ledger for the
stochastic arrays. The exact float32 basis allocation is
`(3*nCells-4)*((degree+1)**2-1)*4` bytes for a closed trivalent spherical mesh.
At 40,962 cells and degree 48 this is 1,179,667,200 bytes. The ledger also
counts retained mesh topology/geometry, mode metadata, member tapers,
coefficients, weights, pattern outputs, the SPPT combination array and,
when SKEB is enabled, the named forcing arrays live at the tendency-write
kernels. Extra SPPT scales each carry their own spectral and pattern buffers.
Inactive scales are not charged. Admission and the direct driver's fallback
use the same sum, including at the one-byte free-memory boundary.

Ordinary, resident and one-rank same-host admission charge each member its
full ledger without a sharing credit. This is a logical array floor rather
than a measured peak. Construction scratch, temporary expression outputs,
allocator placement, reduction workspaces and opaque cuBLAS workspaces are
excluded. Full peak VRAM still needs GPU measurement on both physics routes.
The native route does not use the ArWen footprint fit.

SPPT multiplies held theta, qv and edge momentum tendencies.
Both routes keep the separately staged lagged microphysics heating carrier
unchanged.
The edge factor averages the two adjacent cell factors, applying a common
cell pattern to the projected horizontal wind. Density and other species
are untouched. The default clipped multiplier stays in [0,2]. The optional
logistic transform follows the pinned stochastic wrapper's
`2/(1+exp(taper*pattern))-1` perturbation. Height tapering suppresses the lowest
level and decreases from 18 to 25 km. Surface height and upper bounds are
configurable. Namelist surface limiting also supplies the pinned seven-level
surface weights.

SKEB estimates dissipation from the positive kinetic-energy sink of the
unperturbed physics momentum sources. A vertex streamfunction supplies a
dual-edge curl, with the taper applied before the curl. A separate thermal
pattern has its area mean removed. Default kinetic and thermal fractions are
0.1 and 0.01. Thermal conversion uses a 300 K reference and 1004 J/kg/K heat
capacity. Rate variance uses the integrated AR(1) covariance,
`2*fraction*dissipation/h*tanh(h/(2*time_s))`, with `h=max(dt,interval_s)`.
For held patterns this assumes the update interval is an integer multiple of
the model timestep. The previous white-noise normalization overstated the
long-time energy input by about 72 at dt=600 s and time_s=21600 s. It caused
the initial SKEB-only and combined global forecasts to fail within 27 hours.
The v2 checkpoint schema keeps states from that normalization separate. These
fractions set a nominal covariance budget; they are not an exact
net energy budget after cross terms with the existing wind. This mesh
adaptation has not established equivalence to WRF SKEBS or forecast skill.
It currently does not diagnose dycore numerical dissipation.

Two 128-thread CUDA kernels apply cell and edge forcing directly to the held
tendencies. They fuse clipping, tapering, SPPT multiplication and SKEB addition,
without allocating full level-by-edge curl/backscatter arrays or full cell
SPPT-factor/heat arrays. Dissipation is reduced before either source is changed.
Float32 source arrays may have padded columns and strided levels/cells; mesh
topology and taper resources are made contiguous once during construction.
cuBLAS writes each pattern into a reused member buffer. Returned pattern arrays
are borrowed storage and must be copied when retaining an earlier sample.
The independent GPU comparison test covers both transforms, individual and
combined schemes, missing qv, padding, reversed levels and untouched carriers.
Compilation succeeds offline; the padded/strided comparison also passed on
the RTX 5070 Ti. Complete-step performance remains unqualified.

Runtime receipts report logical retained member-array bytes for tapers,
coefficients, weights and pattern buffers. These counts exclude the shared
mesh resource, allocator pool, cuBLAS workspace and transient reductions.
The SKEB sink/reduction still creates level-by-edge temporary arrays.
Neither these counts nor fusion alone establishes peak VRAM or the five
percent step-cost requirement.

`sppt_scales` accepts up to five independent
`[amplitude,length_m,time_s,seed]` records. Legacy `do_sppt`, `do_skeb`,
`config_sppt_*`, `config_iseed_sppt*`, `config_spptint` and
`config_stochini` are consumed by the same resolver. Unknown active stochastic
controls raise an option-specific error. A namelist containing only switches,
taper or cadence controls uses the base pattern; scalar JSON seed, amplitude,
length and time overrides then control SPPT as well as the base SKEB pattern.
Indexed namelist scale controls declare independent SPPT scales whose explicit
seeds and settings take precedence over scalar controls. Replace `sppt_scales`
explicitly, or set it to `[]` to return to the base pattern.
`config_stochini` requires spectral
restart state. ArWen checkpoints and the native moving carrier include the
coefficients, clock and draw counters. A restart with changed stochastic
parameters is refused. The workflow restart flag itself may change.
The shared ArWen stack builder also constructs the adapter for the one-rank
same-host route. Rank checkpoints retain spectral state and bind seed/settings.
Partitioned and multi-class routes need halo-aware curls, global dissipation
reductions and shared spectral clocks; they report those specific missing
operations before execution. Open regional meshes are not supported yet.

Lineage: the engine's `gpuwm/ensemble/stochastic.py` and
`gpuwm/core/kernels/ensemble_stochastic.cu`, WRF's `module_stoch.F`, and
[the pinned stochastic wrapper](https://github.com/dtcenter/stochastic_physics/blob/1ed03f55444fdf7d49b7687fc4e477de0a276e99/stochastic_physics_m.F90).
The rectangular FFT cannot evaluate patterns directly on a Voronoi mesh;
the mesh transform and forcing adapters here are new. The Philox notice is
included in `licenses/LICENSE-Random123.txt`.

GPU statistics and restart tests are in `tests/test_stochastic.py`. Run them
only on an authorized card with `HEX_STOCHASTIC_GPU_TESTS=1`.
On 2026-10-17, ten GPU stochastic and verification tests passed on an
RTX 5070 Ti under the the reference GPU host OWNER mutex, using CuPy 14.2.0 and CUDA 13.2.
They cover GPU seed words, variance, temporal and spatial correlation,
restart, clipping, mesh curl, padded/strided forcing, extraction and scoring.
The curl check propagates the float32 carrier's output-rounding bound through
the incidence operator; reversing one edge's forcing fails that bound.

The published x1.40962 mesh, converted from its unit sphere to Earth metrics
on the GPU, also passed a two-day prescribed-tendency stress check with
55 levels, a 300 second update and default degree 48. Basis storage was
1,179,667,200 bytes. Construction took 63.15 ms wall time; warmed basis
kernel calls took about 57.23 ms. Pattern evaluation median was 3.99 ms,
and complete stochastic forcing median was 4.90 ms. Same seeds produced
identical words at 0, 300, 86400 and 172800 seconds; changing the seed changed
each output. Measured coefficient variance was 0.9948 and lag correlation
0.98613 versus the configured 0.98621. Basis, pattern and forcing computation
ran on the GPU, without CUDA graphs. This check uses prescribed sources,
not a weather forecast or either complete physics route. Its memory reading
is end-of-check occupancy, not peak VRAM. Receipts are in the lane delivery's
`receipts/gpu-mesh-measurements.json` and `receipts/gpu-stochastic-tests.json`.

Qualification still requires the paired ten-member, ten-day, 120 km forecasts, daily
area-weighted z500 and 2 m temperature spread and mean RMSE against the same
GFS analyses, inspected plots, complete-route repeatability and pattern cost
below five percent of complete step cost.
Receipts include nonblocking CUDA event samples for basis construction and
the first four pattern and forcing calls when those events have completed.
Sample zero includes cold setup. Compare warm samples with complete step
timings to establish the five percent limit. Enqueue wall time alone does
not establish GPU elapsed time.

The shared ArWen composite step and the MPAS outer-step loop also bracket
the first four compute steps with CUDA events on the active member stream.
`stochastic.step_cost` pairs each step with its pattern and complete forcing
calls. It excludes output and checkpoints. Event queries do not wait; pending
samples are reported as incomplete. Stream elapsed includes host dispatch
gaps and waits, so this is distinct from summed kernel execution time.
The first step and the first forcing call are cold samples. Failed steps,
steps without forcing, and samples crossing streams cannot pass the cost bar.
With at least three matched warm steps, `pattern_cost_passed` requires pattern
time strictly below five percent on every sample. Aggregate pattern and
forcing fractions are also reported. Until then the verdict is `null`.
This timing verdict always leaves full qualification incomplete. It does not
establish seed words, spread-skill, the no-single-core bar or landing gates.

The host restart unit compiles `tests/stochastic_pattern_shim.cpp` as a
test-only ABI for the actual CUDA-source AR(1). It exercises the production
Python pattern classes with tiny NumPy matrix products: snapshot ownership,
pickle roundtrip, resumed words, cadence without extra draws, stable member
identity, and rejection of changed configurations or invalid coefficient
state. It covers the base pattern and two extra SPPT scales. A disposable
mutation skipping a draw on restore fails resumed-word equality. These checks
do not establish CUDA words, cuBLAS behavior or model checkpoint execution.
Fractional cadence checks cover 0.1, 0.2, 0.7, 1.1 and 60 second intervals:
21 boundary updates, 41 held calls and resumed-word equality per case. The
clock quotient admits one floating-point ULP below an integral tick so a
decimal boundary such as `0.3/0.1` does not hold the previous pattern for an
extra interval. Times materially before a boundary still hold. This corrects
fractional-clock updates and does not change integer cadence or the seed law.

The ordinary unit suite also compiles `tests/stochastic_forcing.cpp` with
Linux g++ and host syntax shims for the actual CUDA kernels. It checks SPPT
bounds, the logistic transform, cell/qv and edge factors, reversed and padded
array layouts, and untouched padding. A tetrahedral mesh checks SKEB curl
divergence with nonuniform vertical taper. Thermal checks use unequal areas
and densities to check zero mean and prescribed variance with uniform taper;
zero dissipation produces no forcing. A disposable mutation applying taper
after the curl fails the divergence invariant. These are small host unit
checks, without a GPU context or a production CPU backend.

The ordinary unit suite also compiles `tests/stochastic_statistics.cpp` with
Linux g++ and executes the actual CUDA source through host syntax shims.
It checks three published Random123 Philox vectors, the upper draw-counter
word, sampled seed/member words, stationary point variance, independent
channels, temporal AR(1) correlation and the analytical Legendre spatial
covariance. Its small host matrix sum is an independent test oracle for the
spectral law. These checks use host math and do not validate GPU words,
cuBLAS execution, the GPU cache, restart plumbing or forecast performance.

The host unit `tests/stochastic_harmonic_covariance.cpp` independently checks
the addition theorem at degrees 1, 12, 48, 132 and 512. Ten points include
poles, near-poles, longitude seams and antipodes. Every degree band is checked
against ordinary Legendre polynomials of all 55 point-pair separations.
This tests spatial covariance as well as point variance. A disposable
longitude-phase mutation that preserves point variance fails the covariance
check. Host results do not establish GPU execution, words or timings.

`hexcore.stochastic_verification.forecast_fields_gpu` extracts z500 from
GPU `(cell, level)` pressure (Pa) and geopotential height (metres) arrays,
ordered from bottom to top, and passes through GPU `(cell,)` 2 m temperature
in Kelvin. It interpolates linearly in log pressure without extrapolation,
including exact 500 hPa endpoints. Invalid or nonmonotone columns return
NaN. The reader must supply geopotential height at mass levels; interface
`zgrid` and geometric height need conversion before this call. It uses GPU
array operations and has no CPU field fallback. Analytic extraction and
paired scoring fixtures passed on the GPU in the qualification above;
real forecast extraction and scoring remain unmeasured.

Paired verification uses `hexcore.stochastic_verification.score_pair_gpu`.
Pass two mappings from member identity to GPU `z500` (metres) and `t2m`
(Kelvin) cell arrays, the identically aligned GFS analysis arrays, and GPU
cell areas. The function aligns identities, computes sample variance with
`ddof=1`, area-weighted spread and ensemble-mean RMSE, and emits one small
daily JSON receipt. Both arms use the same finite cells. Area coverage and
forecast values missing on otherwise valid analysis cells are recorded.
Grading requires resolving those missing values or supplying a documented
shared terrain mask, so a failing forecast cannot improve its score by
silently losing cells. Decoding and remapping remain the Rust caller's job;
this API has no CPU weather-field path. GFS analysis acquisition and
remapping are not yet wired for this lane's run.

The daily file consumer is `tools/stochastic_verify_day.py`. It invokes
the engine's `rw_netcdf dump` once per input file with the requested
variables together, uploads its decoded binary transport through CuPy,
and calls extraction and paired scoring. Rust decodes variables and file
chunks concurrently. Actual worker counts and timings still need a run
with that reader and a granted GPU. It removes and inventories its own
temporary decoded payloads on success or failure. It leaves the caller's
input files in place. It reads no NetCDF through Python.

Its JSON request has schema `hex-stochastic-day-v1`, `lead_hours`,
`valid_time` with explicit UTC, and these entries (paths should be absolute):

```json
{
  "geometry": {
    "path": "/workspace/shared-prepared/init.nc",
    "height_variable": "zgrid",
    "height_kind": "mpas_constant_gravity_interfaces",
    "model_gravity_m_s2": 9.80616
  },
  "control": {
    "member-01": {"path": "/path/to/run/off/member-01/history.nc", "diagnostics": "/path/to/run/off/member-01/diag.nc"},
    "member-02": {"path": "/path/to/run/off/member-02/history.nc", "diagnostics": "/path/to/run/off/member-02/diag.nc"}
  },
  "stochastic": {
    "member-01": {"path": "/path/to/run/on/member-01/history.nc", "diagnostics": "/path/to/run/on/member-01/diag.nc"},
    "member-02": {"path": "/path/to/run/on/member-02/history.nc", "diagnostics": "/path/to/run/on/member-02/diag.nc"}
  },
  "analysis": {
    "source": "GFS",
    "path": "/path/to/run/aligned-gfs-analysis.nc",
    "analysis_id": "GFS analysis object and content hash",
    "remapping_provenance": "Rust/GPU remap receipt and source/mesh hashes",
    "terrain_mask_variable": "z500_support"
  }
}
```

This shows two members for a small check; the qualification request must
name all ten identical identities in both arms. Add the schema, lead and
valid time at the top level. Each history contains `pressure` in Pa,
`indexToCellID` and one embedded `xtime`; t2 in K may be in the same file
or the optional diagnostics file, which also needs the IDs and clock.
Dimensions are reordered on the GPU by name. Geometry supplies `areaCell`
in m2, cell IDs and the declared height variable. A supplied
`geopotential_mass` height uses `(nCells,nVertLevels)` in metres directly.
The explicit MPAS constant-gravity option averages interface `zgrid`
heights and computes geopotential height as `model_gravity*z/9.80665`.
The model gravity must come from the run configuration, not from a fit
to observations. The current preparation constant is
`cuda_physics_prep_v841.GRAVITY_F32`, 9.80616 m/s2.

With the default `native_netcdf` format, the analysis file contains `z500` in metres and `t2m` in K on the exact
same native cells, IDs and embedded valid time. It must already have been
remapped using the stated Rust/GPU provenance. An optional dimensionless
0/1 terrain mask defines shared z500 support before scoring, with 1
meaning included; it never masks t2m. Forecast missing values on that
support remain counted. IDs in every frame must match geometry in order.

Alternatively, replace `analysis` with a raw GFS analysis specification:

```json
{
  "source": "GFS",
  "format": "gfs_grib2",
  "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261017/00/atmos/gfs.t00z.pgrb2.0p25.f000"
}
```

The URL illustrates the existing GFS public-object convention.
Supply exactly one HTTPS `url` or local GRIB2 `path`. The URL path reads
the object's `.idx` and fetches just HGT at 500 mb, TMP at 2 m above ground
and PRES at the surface, in three concurrent bounded byte ranges. It
checks the analysis date, `anl` label, HTTP 206, exact Content-Range and
length. A separate 2026-10-17 00Z analysis smoke check fetched these three
fields successfully; each requested range returned HTTP 206.

The existing engine Rust `gpuwm_grib2_stack` ABI decodes these fields in
three concurrent calls through ctypes, which releases the GIL. There is
no new GRIB decoder. This generic exporter is serial within each field;
the wrapper overlaps the three independent fields. The receipt records
the native worker IDs and each call's specification and wall time. Actual
core occupancy must be measured on the granted box. `--grib-bridge` can
name its engine Rust shared library explicitly; otherwise the engine
resolver is used. Only instantaneous deterministic PDT 0, forecast step
0, the exact raw WMO selectors and matching reference/valid times pass.
Decoded payload sizes and Rust-recorded SHA-256 hashes are checked.

Geometry must additionally contain `latCell` and `lonCell` in radians.
GPU bilinear interpolation evaluates point values at these cells, with
periodic longitude and either axis scan direction. It accepts a regular
global lat/lon grid spanning both poles with no duplicate seam column;
alternating-row, column-major and projected grids are refused. Missing
positive-weight corners stay missing, and zero-weight missing neighbors
do not contaminate exact grid points. This remap is for point verification,
not a conservative flux transfer. GPU event time covers terrain support
and remapping, separately from Rust decode and upload.

GFS can extrapolate isobaric height below ground. A z500 source corner
is supported only when its surface pressure is finite and at least
50,000 Pa. Every positive-weight corner must pass, defining the same
analysis support for both arms. This does not mask t2m or forecast errors.
The consumer generates the analysis identity from the three content-bound
Rust manifests and records grid, field hashes, reader identity and remap
method. Every owned raw range and decoded payload is inventoried and
removed on exit; a caller-supplied GRIB file remains in place.

On the granted card, consume each day before plotting the ten receipts:

```sh
PYTHONPATH=src python tools/stochastic_verify_day.py day-01.request.json \
  --reader /path/to/rw_netcdf --work /path/to/run/decoded-day --out day-01.json
```

The decoder's provenance, exact commands or native ABI specifications, elapsed times and deleted
transport paths/sizes are included in the daily receipt. An error receipt
retains that audit and no scores. The real GFS acquisition and existing Rust
stack decoder passed a smoke check in 1.275 seconds, using three concurrent
field workers and an unmodified debug build of the engine library. The
receipt is `receipts/native-gfs-probe.json` in the lane delivery. Raw ranges
and decoded payloads were deleted after their hashes and manifests were
recorded. This transport check performed no weather-field arithmetic.
Separate decoded GPU fixtures now cover remapping and the daily consumer;
real GRIB-to-GPU analysis and forecast verification remain unmeasured.

Native ABI tests can run separately with an existing engine library:

```sh
HEX_STOCHASTIC_RUST_TESTS=1 \
GPUWM_CPU_PREPROCESS_BRIDGE=/path/to/libgpuwm_preprocess_cpu.so \
PYTHONPATH=src:/path/to/engine python -m pytest tests/test_stochastic_verification.py -q
```

These tests use generated GRIB2 records to check selectors, clocks, all four
supported scan directions, bitmap missing values and repeated decoded words.
They do not establish stochastic-pattern repeatability on the GPU.

The existing Rust `rw_netcdf` reader also passed real transport tests using
independent generated CDF-1, CDF-2, CDF-5 and compressed NetCDF-4 fixtures.
The tests check scale/offset unpacking, both missing markers, named axes,
units, padded MPAS timestamps, CF times and repeated payload words. The
transport context owns cleanup on success, reader failure and consumer
failure; numeric upload remains in `decode_gpu` through CuPy. Run these
checks with an existing reader and the test fixture writer installed:

```sh
HEX_STOCHASTIC_NETCDF_TESTS=1 \
HEX_STOCHASTIC_NETCDF_READER=/path/to/rw_netcdf \
PYTHONPATH=src python -m pytest tests/test_stochastic_verification.py -q
```

The four-container checks are small ABI tests, not real forecast-history
or GPU-transfer validation. The receipts retain reader/source hashes,
commands and deletion inventories; the fixtures and build products are deleted.

Once the ten real daily receipts exist, run:

```sh
PYTHONPATH=src python tools/stochastic_spread_skill.py \
  --cycle 2026-10-16T00:00:00Z --out verification/paired day-*.json
```

This checks ten identical member identities through days 1-10, verifies UTC
valid times, GFS provenance and units, and writes z500 and t2m analysis charts
plus `spread-skill.json`. Its stated policy requires increased spread and no
larger mean RMSE on every day for both fields. A numerical pass leaves
`qualification_complete=false`: visual inspection, complete-route GPU word
repeatability, step-cost measurements and landing gates still need separate receipts.
