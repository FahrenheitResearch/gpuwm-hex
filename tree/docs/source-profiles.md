# Source profiles

`src/hexcore/data/profiles/source-profiles.v1.json` pins the outside model
configurations hex is measured against. `hexcore.source_profiles` loads and
validates it. A row is a record, not a code path: nothing selects a kernel by
row id, and adding or re-pinning a configuration is an edit to the table plus
the stage proofs.

Each row carries:

- the model source, as full commits, with every physics submodule commit;
- the analysis code commit, when the configuration cycles;
- the dated mesh files it links, with byte counts and S3 ETags as listed;
- its vertical level file, with its SHA-256 and the hex vertical declaration
  built from it;
- its namelist values, per mesh where they differ;
- `hex_dycore_settings`: each dycore setting the namelist asks for, and,
  per hex lane, whether hex admits or refuses it today, naming every refusal
  layer by knob, reason, file and function;
- `dycore_audit`: every dycore routine of the outside model that differs from
  the release hex is a port of, classified at that configuration's settings
  as `inert`, `inert (VERIFY)`, `port` or `declare`, with source definition
  sites and links at the full candidate and upstream commits.

`tests/test_source_profiles.py` holds the table to the tree: on each lane a
refused setting must still meet the named refusal, raised by the named
function, so lifting a refusal fails the test until the row says so.

## The two rows today

**`mpas-a-v8.4.2`**: upstream MPAS-Atmosphere v8.4.2 (commit `f34984b2`), the
release the hex dycore is pinned to. Re-pinned from v8.4.1 (`91c5eac1`) on
2026-10-11: the two tags differ in twelve files, none of them dycore
arithmetic hex ports, and native builds of both wrote the same numeric arrays
from one namelist at 2 and at 4 acoustic sub-steps (the re-pin receipts, kept
with the release evidence). The one default that moved is
`config_number_of_sub_steps`, 2 to 4.

**`rrfs-v2`**: NOAA's RRFS v2 as the public `rrfs-mpas-jedi` workflow at
`85ea55c0` builds and runs it.

- Two model candidates, because which one the real-time runs execute is not
  public: the workflow's own pin `1d6e370e` (Registry version 8.3.1-noaa) and
  the fork's tip `c3f78a5a` (8.4.1-noaa). The fork tip moves the MYNN-EDMF and
  SMOKE pins and adds a `stochastic_physics` submodule; every other physics
  pin is shared.
- The analysis is RDASApp at `1bfecf33`, the workflow's gitlink.
- Meshes, as the workflow links them: grid, static and orographic files from
  the `20251017` folders for both `conus12km` and `conus3km`, and the L60
  invariant files from the `20260319` folders. The `conus12km` partition
  files come from `20250321`.
- Vertical: the 60-interface L60 file, 59 layers, top 25878.712 m, declared
  for hex as `verification/vertical-specs/rrfs-l60-v1.json`.

- Mesh files: the names the workflow's fix links resolve to were matched
  against the public bucket `fix_rrfs2/v2.1.4-3.4/main`. The links point at
  machine-local directories on NOAA's HPC systems, so treating that bucket
  folder as their mirror is an inference, recorded as one in the row.

## What the dycore audit found

By source reading, at this configuration's settings (a reading of source,
not a dycore L4 measurement):

- **Fork tip `c3f78a5a`**: its dycore routines do v8.4.1's arithmetic. Every
  difference is inert (SPPT off in the deterministic run, opt-in scalar
  mixing off, boundary `w` at the upstream default), an I/O contract (the DA
  cycling stream), or outside the dycore.
- **Workflow pin `1d6e370e`**: solves the same equations, but the vertically
  implicit acoustic solve and the Smagorinsky deformation use a different
  operation order from v8.4.1, so a v8.4.1 port can reach it only inside a
  rounding envelope unless those two routines are also ported in their
  8.3.1 form. Two inert calls wait on mesh values (`meshDensity`, and the
  reference winds `u_init`, `v_init`).
- **Both candidates** carry the DA cycling I/O contract: with
  `config_do_DAcycling` set, `atm_core_init` starts from the `da_state`
  stream and `atm_core_run` resets its output alarm. RRFS's warm cycles set
  it. Class `port`, stage H6.
- The scalar transport and the monotonic limiter are the same arithmetic in
  all three.

## How hex stands against the namelist's dycore settings

Every key in the public namelist's lines 1-26 (`&nhyd_model`, `&damping`,
`&limited_area`; a verbatim copy with the workflow's placeholders is
`verification/source-namelists/rrfs-v2-namelist.atmosphere.lines-1-26`, held
to its SHA-256) has a row, and each row names its state on two hex lanes:

- **namelist-dry**: hex's namelist door,
  `hexcore.config.dry_dycore_config_from_namelist(..., source_release="v8.4.1")`.
- **forecast-column**: the lane `gpuwm-hex forecast` builds. It reads no
  namelist; the door exposes only the timestep, the physics cadences and the
  convection, mixing, boundary, RK order, acoustic count and radiation cadence
  selections.

Read through the namelist door as written, the lines meet two refusals:
`config_start_time` and `config_run_duration` (calendar keys). With those
removed the door admits the rest,
4 substeps, the limiter, a 16 km damping start and `config_epssm = 0.1`
among it; the last is ignored with upstream's own warning, as native v8.4.1
does. The CPU and CUDA paths admit the monotonic limiter.

The forecast-column lane runs any order-2 or order-3 schedule; the keyed
dycore certificates (order 2 at 15 s/2, 20 s/4 and 60 s/2 acoustic substeps)
are evidence the run records, not a gate (retired 2026-10-10). Native upper
damping option subtypes carry the CAM level count and Rayleigh timescale
without adding fields to the released configuration objects. The CAM floor
and Rayleigh tendencies execute in Rust on CPU and in CUDA on the card.
The generic forecast options are `--dt`, `--time-integration-order`,
`--acoustic-substeps`, `--radiation-seconds` and `--monotonic/--no-monotonic`;
their defaults are MPAS-A v8.4.2's Registry defaults (order 2, four acoustic
substeps, radiation every 1800 s, the limiter on, and 6 s per km of the mesh's
nominal spacing for the timestep; `hexcore.mpas_defaults`). The native x4
reference's RK3 / six-substep / 600 s schedule is passed explicitly by the
hex self-reproducibility tests.

`--monotonic` selects the native scalar limiter through both forecast doors.
The native nonzero-tracer receipt is in
`verification/scalar-transport/nonzero-v841.json`. CPU and CUDA replays of
the real native mass-flux replays match their recorded operator arrays, repeated production steps match,
and the replay fits a source-derived binary32 accumulation budget against
native output. The regional residual and copy masks match blocks compiled
from the same native source. This is an operator proof, not a qualification
of coupled forecast endpoints or multi-card limiter execution.

`--cam-coefficient`, `--cam-damping-levels`, `--rayleigh-u` and
`--damping-start-m` select upper damping. Rust and CUDA match the compiled
v8.4.1 single-precision source-block arrays. The real-mesh driver executes
three CAM applications and nine Rayleigh applications per step and repeats
its result. See `verification/upper-damping/native-v841.json`. Active CPU
upper damping builds its Rust library in the selected CPU cache; it requires
`rustc`. These are operator and execution proofs, not native forecast
endpoint qualification.

The workflow-pinned 8.3.1-noaa CAM and horizontal-wind Rayleigh expressions
agree with upstream v8.4.1 after the viscosity field rename. The source
sites and full-file hashes are in
`verification/upper-damping/noaa-source-comparison.json`. Regional limiter
density scratch registers with the existing unit-density pool so the
post-launch cleanup preserves its sentinel between transport kernels.

The certificates hold native schedule bits, first-stage coefficient reuse,
scalar target-density bits and repeatable dry steps on a real mesh. Whole
CPU/CUDA endpoint differences are recorded. They do not qualify an endpoint
against native MPAS on a public regional mesh or the coupled physics.

The real imported L60 atmosphere executes one coupled 60 s step on the
157,859-cell conus12km mesh with all the above settings. It has 59 layers
and 60 interfaces. Both history frames pass the physical health gates,
and the step has finite fields, positive density and no negative moisture
in the reported prognostic block. The execution receipt is
`verification/vertical-specs/l60-execution-v841.json`. This uses the existing
unanchored measurement entry point and retains its numerical health gates.
It is an execution proof; a native regional endpoint and forecast skill
remain separate measurements.

Each refused row names the refusal that fires first on each lane and every
refusal behind it that a port must also lift. `tests/test_source_profiles.py`
reads the lines through the namelist door and applies each knob to the
forecast-column lane, and requires the refusals met to be exactly the rows,
by knob, reason and raising function; the layers behind the first are held
by source to their file and function.

## Native run receipts

Native history files carry run-specific `file_id` and `parent_id` attributes.
Their whole-file hashes change even when every variable value is unchanged.
Use `bash tools/digest_netcdf_variables.sh <history.nc> u w theta rho
surface_pressure` to record selected data values as canonical `ncdump` text
digests, with nine significant digits for float32 and seventeen for float64.
Compare the dimensions and variable types separately, and record the executable,
input, namelist and streams file hashes. These digests do not preserve NaN
payload bits and are not whole-file byte identity.
