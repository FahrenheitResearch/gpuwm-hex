# Test-only native MPAS physics

The boundary is `driver_X`, with Registry names and native Fortran dimensions,
owned cells plus halos. Pristine MPAS `X_from_MPAS`, scheme calls and `X_to_MPAS`
remain inside each driver. No scheme arithmetic is generated or translated.
`abi.json` records the exact source signatures and citations. Initialization,
allocation, conversion, updates and finalization are explicit native entries.
All schemes selected by upstream Makefiles, MMM, UGWP and Noah-MP are linked.
`--licence-stub noahmp` compiles the Apache-2.0 interface stub beside this file
(`licence_stub_noahmp.F90`, written by `make_licence_stubs.py`) in place of
Noah-MP v5.0 (UCAR terms: no sale); a called stub stops the run naming the
opt-in. `gpuwm-hex physics-resources build` passes it unless
`--licence-opt-in ucar-noahmp` names the terms; without the flag this builder
compiles exactly what it compiled before the stub existed.
MUSICA chemistry is excluded by the upstream physics object selection.

The library uses actual MPAS pool/field, distributed-memory, calendar, log and
stream-manager implementations. Each pool owns a copy of its fields. NumPy or
CuPy buffers are transferred at the boundary and never become native-owned
pointers. Time levels are one based and allocated separately. Fields must be
float32 (RKIND) or int32; table access preserves the compiler's actual kind.
The pinned RRTMG `parkind` defines `rb=kind(1.0)`, which is four bytes in this
single-precision recipe. An inactive commented real64 alternative is not used.
Native shared state persists and only one instance may live in a process.

Build from the offline pinned tree, using the MPI compiler recorded by f0.
On a shared host, run compilation through its build queue (`HEX_BUILD_QUEUE`). PnetCDF must be PIC:

```sh
# Prepend f0's recorded MPICH prefix, not the system OpenMPI wrapper.
export PATH=/path/to/mpich/bin:$PATH
bash tree/tools/native_physics_reference/backend/build_pnetcdf.sh \
  /path/to/local/pnetcdf.tar.gz /path/to/private/pnetcdf
python tree/tools/native_physics_reference/backend/build.py \
  --source "$MPAS_REF" --work /path/to/private/build \
  --pnetcdf /path/to/private/pnetcdf/install --probe
```

The builder copies sources without modification, preprocesses into a separate
directory, orders actual USE dependencies, and builds with make -j4. The stock
host maths match f0's first GNU strict artifact. Supply `--math-object` only when
f0's reference uses that same pinned PIC maths object. The receipt records its
hash. Build receipt success is not scientific acceptance.
Context creation applies GNU Fortran's big-endian unformatted-I/O runtime
setting, matching the startup generated for the strict native executable.
The builder checks the MPAS and external git pins and every compiled copied
source hash, and rebuilds when the compiler/dependency/flag recipe changes.

```python
from hexcore.mpasphys.native_backend import NativePhysics
with NativePhysics('/path/to/build/libmpasphys_native.so', test_only=True) as native:
    native.config('config_radt_cld_scheme', 'cld_fraction')
    native.put_field('mesh', 'areaCell', area_cell_float32)
    # Populate every native argument pool and initialize physics module state.
    native.driver_cloudiness(configs='configs', mesh='mesh',
        diag_physics='diag_physics', sfc_input='sfc_input', its=1, ite=nowned)
    native.get_field('diag_physics', 'cldfrac', out=device_cldfrac)
```

Use `put_variable`/`get_variable` to replay primitive shared module state,
including its Fortran lower bounds. The 477-variable manifest covers
`mpas_atmphys_vars`. `get_table` exposes 563 public RRTMG/manager values with
their actual word width and lower bounds. Noah-MP's derived state and private
scheme state remain native state; this interface is not a complete checkpoint.
Scalar character pool fields, such as `mminlu`, use `put_text_field`.

`get_noahmp_io(name)` copies the current component of native `mpas_noahmp`
without initializing or changing it. `abi['noahmp_io']` indexes every primitive
member of the pinned MPAS-preprocessed `NoahmpIO_type`; WRF-only members are
omitted. The result is `(array, lower_bounds)`, including scalar arrays, native
integer/logical words and complete fixed-width character bytes. An unallocated
component returns `None`. Native allocation and bounds are preserved; this
read-only service does not install a CUDA callback or create absent solver state.
Its source is `mpas_atmphys_vars.F:907` and
`physics_noahmp/drivers/mpas/NoahmpIOVarType.F90:18-938`. Historical libraries
without this optional ABI metadata remain usable for their existing entries.

For whole-model integration, initialize complete configs/dimensions/fields,
allocate shared storage, create the native clock and stream manager, then call
`physics_init` with `framework_object` handles and named pools. Preserve the
upstream driver/RK order. No user default or suite selection changes here.

The ABI includes 62 generated entries, including native table, Registry and
run initialization, the manager's physics_timetracker and surface updates.
`block` attaches the actual domain/block pools; `register_field` supplies native
Registry dimensions and separately owned time levels. `decomposition` collectively
constructs pristine MPAS send/receive/copy lists from owned and halo global IDs;
`exchange_halo` exchanges a registered array's selected time level. All ranks
must participate, including ranks with no halos. Exclude trailing dummy ID zero
from the global-ID vector; the field may retain its MPAS dummy storage.
`stream` creates an actual native input/output stream; `stream_field` attaches
registered fields and `transfer_stream` invokes native I/O and input halo exchange.
Provide all required mesh metadata and fields before file-reading initialization.
`advance_clock` advances the actual native clock. MPI stays initialized across contexts, and caller-owned MPI
is never finalized. Pool lookup follows the pinned non-MPAS_DEBUG framework's
native SILENT policy: missing optional pointers remain unassociated. Explicit
native fatal errors retain MPAS behavior.

The pristine `mpas_atmphys_todynamics.F` and `mpas_atm_dimensions.F` are linked.
Call `mpas_atm_set_dims` with the current native dimensions before `physics_get_tend`.
Its three tendency arrays transfer through typed rank-two C descriptors, preserving
native float32 words and MPAS dummy storage. `framework_object('block')` supplies
the actual block. The wrapper binds the native halo callback internally;
`exchange_group` registers the named field groups used by tendency assembly.
The ABI manifest retains the original procedure argument and its native binding.

`oracle_replay.py --library LIB --manifest JSON --receipt JSON` orchestrates f1
ORCL0001 captures. The manifest schema is `mpas-native-replay-v1`; ordered `calls`
contain `entry`, exact named `arguments`, `inputs` and `outputs`. Each capture mapping
states `capture`, `role` and `name`, plus `pool`/`level`/`nlevels` for pool fields.
Input roles are field, text_field, variable, config, dimension and argument;
output roles are field, variable, table and argument. Framework arguments use
`{"framework": "clock"}` (or domain/block/dminfo/stream_manager).
Rust validates and decodes file payloads and counts differing words; Python only
maps buffers to native calls. Captures remain read-only. Input completeness is
explicit in receipts and is never inferred from zero output differences.

`boundary_manifest.py` selects level-b pool storage from f1's association
metadata. Unassociated descriptors and arrays are listed explicitly in the
manifest. An associated field whose payload is missing fails capture validation:
silently skipping that payload would falsely report a complete replay. A requested
shared-state baseline must contain captured variables. This validates the capture
fixture and does not change native scientific behavior or exclude native words.
For an immutable published pool product, pass its selected `files` mapping as
`sealed_files` to `build_manifest`. Every consumed metadata and payload record
must match its published SHA256; the index also avoids repeated directory scans
of captures with hundreds of thousands of records.

Verification uses the real `.so`:

```sh
MPAS_NATIVE_LIBRARY=/path/to/build/libmpasphys_native.so \
  taskset -c 8-23 python -m pytest tests/test_mpas_native_backend.py -q
```

`--probe` also builds `probe_cloudiness.f90`, which separately constructs real
MPAS pools and calls the native driver without generated wrappers, and
`probe_tables.f90`, which calls native RRTMG initialization directly.
`probe_surface.f90` independently calls the pristine monthly surface update
with native StrKIND date storage. Generated string arguments retain that
storage and its blank padding: trimming a date before forwarding it to
monthly interpolation lets the native fixed-length callee read beyond the
temporary. The Python replay checks all twelve months, leap day, fractional
seconds, both albedo branches, ocean/land columns and untouched dummy storage.
Tests
compare every output word for three
selectors and all four condensate flag combinations, plus synthetic regrouping and halo
sentinels. With `MPAS_NATIVE_TABLES` set to f0's strict artifact directory, tests
exercise the schemes-disabled native initialization chain with LANDUSE.TBL and
compare six RRTMG tables directly. Tests also invoke native monthly surface
updates, stream file round trips, and two-rank MPI exchanges for real ranks 1-5
and integer ranks 1-3, preserving signed zero, NaN payloads and separate time levels.
RK tests exercise native longwave/shortwave assembly and PBL/convection wind
coupling with actual named halo groups. Build with `--tap-transport` pointing to
f1's existing `mpas_oracle_dump.F` to also capture/replay all three synthetic
cloudiness drivers through the exact f1 format, including inputs and outputs.
Set `MPAS_NATIVE_MESH` to a published MPAS mesh and `MPAS_NATIVE_GRAPH` to its
graph file prefix to run the CPU partition check. The native Fortran probe reads
mesh areas, densities, global IDs, adjacency and the `.part.2`/`.part.4` files,
constructs owned cells plus adjacent halos and dummy storage, and reverses owned
cell order. Each partition is replayed through Python's native loader; Rust
compares the owned output words against the unpartitioned native call by global
ID. Atmospheric columns in this check are synthetic. This proves the column
boundary's independence from mesh partition and ordering, not a forecast.
These checks do not replace f1 dump replay or
whole-forecast acceptance receipts.

`reproduce_written_forecasts.py` executes every deck in an explicitly supplied
sealed index through the given checkout's actual forecast door, under the GPU queue.
Each forecast fixes one process, authenticates the original native publication,
inputs and provider, and preserves the complete verify-mpas receipt and model
completion. The index selects suites, meshes, dates and output schedules.

The integrator's M0 ruling defers existing unwritten fields to their owners.
`check_written_forecast.py` checks that scope separately: every computed word
must be identical, every original sample must be inventoried, and a previously
computed field becoming absent fails. The sealed missing-field inventory is
discovered from authenticated strict native outputs through Rust. Newly
implemented fields must also match exactly. Original missing samples, official
failed grades and ratchet entries remain intact; the scoped receipt states
whether full-field identity was actually reached.

The production library also exports `native_manager_state` for f4's
`NativeCalendarState`. Pass `native.lib` to that provider. The compiled-artifact
checks cover leap and century boundaries, year rollover, fractional clock steps,
all eight manager flags and calendar argument binding through f4.

Set `MPAS_NATIVE_ORACLE` to an explicit published radiation capture directory
containing `manifest.json` and `PUBLICATION.json` to compare exposed RRTMG tables
against its pre-radiation snapshots. The test verifies each selected capture's
published hash, initializes both native lookup tables, and uses the Rust
comparator on every matched public table. This independent table check does not
establish complete level-c suite initialization or driver-output acceptance.

For initialized weather columns, provide a valid published initialization and
an explicit schemes-disabled namelist with `config_microp_scheme='mp_kessler'`.
The Kessler fixture preserves native qv/qc/qr; inactive ice species are zero.
Respect withdrawn publications in `ref/READY.md`. Generate native initial
diagnostics with the same strict reference executable, without advancing time:

```sh
taskset -c 8-23 python tree/tools/native_physics_reference/backend/prepare_columns.py \
  --init /path/to/published/init.nc --native /path/to/strict/native \
  --namelist /path/to/explicit/namelist.atmosphere \
  --work /path/to/fresh/private/columns --receipt /path/to/columns-receipt.json
```

Set `MPAS_NATIVE_INIT` to that initialization file and `MPAS_NATIVE_STATE` to
the resulting `columns.nc`, alongside the mesh/graph variables above. Three
additional tests decode the native columns in `case_columns.f90`, run pristine
`MPAS_to_physics`, then compare each full cloudiness driver through ctypes
and across 1/2/4 partitions. Source conversion arithmetic is unchanged. Receipts
seal initialization, diagnostic output and capture hashes, column counts and
actual level counts. Without those two environment variables these tests skip;
compilation alone does not establish real-column identity. Retire private raw
outputs only after receipts are sealed.

Framework ABI version 2 separates register_field's pool lookup name from its
registry_name (default: the lookup name). Pass components in native constituent
axis order and has_time from Registry Time metadata. allFields and stream/halo
lookups use Registry names; each time level retains its own native storage.
Older version-1 libraries are rejected before any incompatible framework call.

stream accepts input;output with independent input_interval/output_interval.
Native clock alarms are attached separately as <stream>:in and <stream>:out;
initial_only/final_only are one-time alarms and none attaches no alarm. alarm_ringing
queries the actual MPAS clock, including its native ringing envelope.

Reset a consumed clock alarm with reset_alarm, which directly calls native
mpas_reset_clock_alarm. MPAS keeps an alarm ringing until it is reset;
advance_clock alone does not consume it. Stream transfers retain their native
manager reset behavior.

`get_constant(name)` returns the compiled scalar with its original NumPy dtype.
`abi['constants']` records the declaration, source hash and accessor index;
Fortran evaluates every constant expression. Names are the native public names
(including `r_d`, `rcp`, `ep_1`, `ep_2` and framework `omega`); slot-specific
argument aliases belong in the suite table.

Character fields accept `level` and `nlevels`, just like numeric fields. Allocate
both `state.xtime` and `state.initial_time` levels when the Registry struct has
two levels. `initialize_state_time()` copies the initial timestamp, initializes
both valid-time levels and sets both CF Time attribute lists. Call it after the
native clock and state storage exist. `update_state_time(dt)` writes level 2 from
level 1 using native RKIND addition and clock-now plus dt, matching the native
post-dynamics point. Pass `clock_advanced=True` when the owner has already advanced
the native clock. Explicit source/target levels support the owner's state rotation;
the wrapper performs no clock advance or state rotation itself.

Driver-pool debugging can select an explicit f1 boundary prefix with
`boundary_manifest.py --library LIB --specification SPEC --manifest MANIFEST`,
then replay that manifest with `oracle_replay.py`. SPEC states the pool capture
root/prefix, exact native arguments, before/after allocation calls, process
count and input-completeness limits. Optional shared-state captures must match
f1's sealed manifest; these are primitive state, not a private-state checkpoint.
Native pool member kinds and character time levels come from captured metadata.
All scientific payload decoding and comparisons remain in Rust.

`replay(..., native=initialized_context)` preserves the caller's native context
and requires its library hash to match. This permits original initialization
and ordered predecessor drivers to prepare private/shared state. In particular,
sea ice uses the ocean-fraction arrays produced by surface layer; isolated sea
ice calls with a pre-cloudiness baseline omit those arrays. One-step replay
receipts are debugging and never substitute for full forecast acceptance.

`get_real_config(name, pool="configs")` reads the current native RKIND pointer
value after native initialization or core resolution. It does not return the
Python setter cache or calculate a replacement. The value retains float32 words.

Boundary replay also compares scalar character fields with Rust, byte for byte,
including all StrKIND padding and each captured time level.
`get_text_field_storage` copies native bytes without text normalization;
`get_text_field` remains the trimmed text interface for metadata consumers.

The frozen earlier replay fixture is at
`tests/native_backend_fixtures/replay_boundary_v20.py`. It initializes the real
native provider, restores original pre-call pool storage, allocates the caller's
scheme arrays, and then restores captured primitive globals at driver entry.
It compares post-call pool storage, primitive globals and exposed public tables
before native deallocation. It also checks all 563 exposed initialization
tables. Raw file decoding and comparisons use the Rust transport. The complete
fixture runs both native suites at one process; the separately sealed corpus
records the supported cross-host scope. Private, derived, external and alias
state remains outside this interface's checkpoint claim. Full forecast
acceptance still requires the real door and every requested native stream.

For a lander proof, invoke the delivery's `receipts/reproduce.sh HEX_CHECKOUT`.
It runs the given checkout, with no replacement tree, using the explicitly
sealed deck/index and immutable native provider. The f3 delivery index requires
one complete native-backend M0 deck, as directed by the integrator; the
eight-deck matrix and both-suite capture proofs are retained separately.
The checker returns success only after the full fixed-process forecast and
zero differing written words; original missing fields remain in the raw
verifier receipt under the integrator's documented M0 scope.
