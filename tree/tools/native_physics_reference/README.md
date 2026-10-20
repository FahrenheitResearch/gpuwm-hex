# Native MPAS-A full-physics reference

These tools compile and run the unmodified, pinned MPAS v8.4.2 physics. They do
not implement physics in Python. MPAS performs initialization, interpolation,
GRIB intermediate ingestion, and forecasting; the Rust comparator reads NetCDF
floating-point words without converting their bit patterns.

## Build and publish

On a shared host, run compilation through its build queue (`HEX_BUILD_QUEUE`), with the host's authorized CPU
affinity. Supply existing local source and dependencies; builds never download
sources. The source directory must contain the pinned MMM and UGWP externals.

```sh
bash build_reference.sh WORK SOURCE gnu-strict atmosphere MPI PNETCDF
bash build_reference.sh WORK SOURCE gnu-strict init_atmosphere MPI PNETCDF
bash build_reference.sh WORK SOURCE gnu-o3 atmosphere MPI PNETCDF
python publish_build.py BUILD SHARED_REF --core atmosphere --recipe gnu-strict
```

The builder copies the read-only source, changes only build flags and offline
external checkout configuration, and retains their diff. Single precision uses
the upstream GNU recipe; strict builds add `-ffp-contract=off`,
`-fno-tree-vectorize`, and `-fno-unsafe-math-optimizations`. The object audit covers
direct MPAS physics, MMM, UGWP, Noah-MP, and WRF objects. Missing external groups
or missing strict flags fail the audit. An optional final argument inserts a
pinned maths object before library archives and rejects remaining dynamic
references to the covered binary32 functions. Such a recipe must be separately
identified and proven against hex.

`build_tables` is produced by the atmosphere build. Run it in a fresh working
directory under the native CPU affinity, then hash and retain all four Thompson
DBL files. The published executable, data files, decks, build log, source diff,
toolchain information, and their hashes are indexed by `MANIFEST.json`. READY
publication is atomic and does not assert dycore acceptance.

## Inputs and decks

`download_inputs.py INDEX --directory DIRECTORY` copies explicitly authorized
public input URLs with curl. The index has a `files` list containing `url` and
`name`. Downloads resume partial files, retain HTTP headers, and record byte
counts and SHA256 hashes in `MANIFEST.json`; native ungrib interprets the GRIB
contents. There are no built-in dates, meshes, or data sources.

`cache_cases.py` accepts an explicit JSON `cases` list. Each row supplies `mesh`,
`grid`, `graph`, `start`, `stop`, `surface_stop`, `gribs`, `interval_seconds`,
`kind`, `levels`, and `fg_levels`. Dates use `YYYY-MM-DD_HH:MM:SS`; intervals are
10800 or 21600 seconds. Global rows execute native cases 7 and 8. Limited-area
rows execute native case 9 and require an already prepared regional input mesh;
this tool does not substitute a global mesh for a create_region cut.
An optional `stages` list selects `[7]` for immediate initialization publication
or `[8]` to add a surface sequence using the explicitly supplied `initial` file.
Separate work roots preserve the already published initialization products.
Inactive initialization output streams use a separate filename in case8:
native `xml_stream_parser.c:482-492` rejects an input/output filename collision
before package activation, even when output_interval is none. `execute_native`
separates the native stages from ungrib so an inspected setup failure can reuse
completed, independently hashed native intermediates without decoding again.

```sh
python cache_cases.py INPUT_INDEX --acceptance-index ACCEPTANCE_INDEX --work WORK --init INIT_BINARY \
  --ungrib UNGRIB_BINARY --vtable VTABLE --geog GEOG_DIRECTORY --mpi MPIRUN
python generate_decks.py --source SOURCE --out DECKS \
  --streams-template BUILD/streams.atmosphere --start START --duration DURATION \
  --output-interval INTERVAL --dt DT --mesh MESH --init INIT_FILENAME
```

The acceptance index contains a `checks` list, each with `receipt`, `hours`,
`cells`, `levels`, and `init_sha256`. Supply the completed one-step and 24-hour
identity_compare checks. Partial, empty, failed, or missing-sample comparisons
are refused before any case job starts.

The deck generator reads scheme choices and suite assignments from
`src/core_atmosphere/Registry.xml:2351-2416` and
`physics/mpas_atmphys_control.F:132-333`. Its INDEX records every option,
override, source hash, and forced companion. Surface layer companions accompany
PBL/LSM choices; radiation includes cloud fraction. All available output streams
are requested, including surface and UGWP diagnostics. Package-gated streams
remain subject to MPAS's native package activation; JEDI DA is not compiled by
this recipe. MPAS's moist-physics activation also controls whether a lone GWD or
logical NGW option executes; a generated deck alone is no execution proof.

## Forecast and identity

`run_forecast.py` requires explicit build, deck, initialized case directory,
Thompson table directory, MPI launcher, rank count, and pin-set hash. It refuses
an existing output directory, links unchanged native inputs, runs the actual
model, verifies its completion message, and hashes every produced NetCDF stream.
Use native graph partitions matching the requested rank count.

```sh
cargo build --release -j4 --manifest-path word_compare/Cargo.toml
python compare_runs.py REFERENCE OTHER --comparator RUST_BINARY --out RECEIPT
```

The Rust comparator checks every numeric and character variable, dimensions,
and raw words, including signed zero and NaN payloads. Empty outputs, missing
files/variables and unsupported field types fail. The receipt reports all-field,
floating-word and whole-file SHA256
equality separately: MPAS NetCDF metadata can contain generated file identifiers.
Whole-file identity must be witnessed, never inferred from floating-word identity.

`publish_cache.py SOURCE DESTINATION --include PATTERN --ready READY_FILE
--description DESCRIPTION` publishes selected case, deck, or forecast files
atomically. Repeat `--include` for each selection. Every selected symlink is
resolved and copied, so deleting scratch cannot invalidate the published cache.
`PUBLICATION.json` hashes every selected file; publication refuses an existing
destination. Keep graph partitions with initialized case products, and native
manifests/decks/logs with forecast streams.

The first dycore acceptance uses the existing native_dry_reference tools against
real native outputs and hex CUDA. On a shared host, GPU execution goes through its GPU queue (`HEX_GPU_QUEUE`).
No native-only run, comparator self-check, generated deck, or orchestration test
constitutes forecast acceptance. Chemistry excludes MUSICA; existing physics
defaults are not changed.

`acceptance_pipeline.py` continues the required one-step and 24-hour comparisons
through the existing CUDA driver. Every GPU/compiler launch uses its slot wrapper.
On a real word mismatch it immediately rebuilds and runs the supplied immutable
GNU13.3/glibc2.39 container image, retaining separate receipts; an interrupted
queue is retried without creating an identity claim.

`cache_matrix.py` and `run_matrix.py` take explicit case/job lists and publish each
completed product immediately. They wait for input files and the pool's 16 GiB
free-memory threshold. A resumed run verifies every published file hash before
reusing a result; an unfinished forecast directory needs inspection. The case
runner's `--join-existing` joins completed manifests from a previous invocation.
The forecast runner also supports `--join-existing`: it waits for an inspected
live run's manifest and can execute other independent rows while that run
continues. Parallel queues must have nonoverlapping output/publication paths.
An aggregate runner starts only after every cohort finishes and verifies the
complete original matrix; a retired controller's exit code is not completion.
Both runners remove their own copied NetCDF dumps only after verifying the
published copies; the case runner also removes hashed ungrib intermediates.
Native initialization rows can specify `ranks`; MPI runs require the matching
graph partition, which is hashed with the inputs. Published/resumed case
contracts must match, including rank count. Native cache rows can select case 7
or case 8 independently, so initialized files
can be published before the five-day surface sequence is available.
Case7 preflight requires the actual native 20-class MODIS geography. Aliasing
the WPS with-lakes dataset into that path introduces category21, which native
Noah rejects; the guard cites that observed driver failure. Case manifests
record geography index hashes.

`prepare_regions.py` waits for published native initializations and invokes the
installed engine Rust `rw_mpas_mesh --cull-parent` door with explicit region rows.
It hashes the tool, parent, region and resulting regional init/graph and keeps
the cull receipt. Native case9 then runs on the regional native fields. The
existing implementation follows MPAS-Limited-Area v2.2 conventions; this tool
does not assert a newly measured native `create_region` byte comparison.

`native_rank_identity.py` compares every regular NetCDF output file and every
numeric and character field through the Rust comparator, preserving raw words
for binary32/binary64 values and signed/unsigned integer storage.
Rank witnesses require completed manifests with matching decks, runtime,
executable and native inputs, allowing only the rank partition to differ. Each
compared output hash must match its completed manifest. A word mismatch returns1;
an invalid comparison contract returns2, so queue workers cannot mistake a
failed comparison launch for witnessed rank variation. `cross_host_identity.py`
first checks completed manifests for matching decks, runtimes and all input
hashes except the independently rebuilt executable. It compares both native
output hashes and raw floating words, retains native `ncdump -h` differences,
and writes the host eligibility marker only after the actual witness exists.
A zero-word difference with a changed generated NetCDF `file_id` produces the
DIFFERS marker; metadata is never removed or normalized to claim file identity.

`verify_products.py` checks native manifest hashes, the complete requested file
set, and stored `xtime`/Time record counts using the Rust `--inventory` door.
Surface streams without an `xtime` field retain an explicit filename-only clock
witness. Callers name inactive compiled packages explicitly. `verify_matrix_products.py`
witnesses listed published forecasts and surface/LBC caches as they appear,
retaining a separate receipt for every product. Missing scheduled files or a
stored clock hidden by a correct filename fail; these checks make no data identity claim.
Case7 witnesses also check the initialized clock, vertical/soil dimensions and
matching cell counts in the UGWP ancillary file. That static file has no Time
dimension or forecast clock in native MPAS and is recorded explicitly as such.
The Rust `--i32-range FIELD FILE` door scans integer categories in bounded
chunks. Case7 checks the actual `ivgtyp` maximum against native Noah's20-class
table, preserving the observed category21 failure instead of repairing fields.

`index_forecasts.py JOB_INDEX --details CLOCK_RECEIPTS --out CATALOG --receipt RECEIPT`
indexes the explicit run matrix by pin set, recipe, mesh, date, deck, ranks and
duration. A completed reference must match the published native manifest and a
passing clock witness for exactly those output hashes. `--watch` updates the
catalog atomically as runs publish. Pending executions retain their status;
native reference completion makes no claim of CUDA equivalence.

The proof host's foundation acceptance passed with the unmodified GNU15.2/glibc2.43
strict build: all12 dycore fields are equal at step1 and every6h through24h.
The canonical run receipts are listed in the foundation REPORT.md and READY.md;
this measured result required neither the optional maths object nor a container
fallback. Full-physics references retain separate forecast, clock and identity
receipts as they finish.

`verify_foundation.py EVIDENCE_INDEX --receipt RECEIPT` checks the explicit
foundation delivery scope: accepted dycore samples, complete object audits,
source-derived option/companion coverage and deck hashes, pinned build/table
artifacts, native output clock/hash receipts, requested forecast contracts and
nonempty identity witnesses for the declared runs. Missing queued products stay
pending. A witnessed rank or host difference is recorded as a limitation; it
never becomes an identity claim. The gate's complete status applies only to its
listed evidence scope; the foundation brief defines that scope.

Shared decoding and disk limits
------------------------------
`shared_ungrib.py` produces mesh-independent native WPS files with a locked,
atomic manifest. Supply an explicit `inputs` list of `{grib,time}` rows and
`--directory`; `--surface` derives a table containing only the native surface
reader's SST/SKINTEMP, SEAICE and LANDSEA fields. This is native decoding, with
no Python weather-field transformations. A completed row may use
`shared_ungrib` instead of requiring raw GRIB files: hashes and source provenance
are checked before linking the intermediates into its native case directory.
`COMPLETE.json` releases a shared sequence to a cache matrix. Initial analyses
remain full-field. Full LBC inputs should be fetched by their original URL/hash,
decoded and consumed one clock at a time, then removed after receipts, rather
than retaining a global full-field five-day intermediate sequence. Keep the raw
input manifest when retiring decoded GRIBs. Completed case publications are
checked before dependencies, so raw-input retirement cannot restart them.

Native I/O errors and storage
----------------------------
`native_completion.py` requires the correct final native banner and refuses
ERROR/CRITICAL/FATAL log records. This protects against witnessed MPI/SMIOL
write failures after disk exhaustion; a banner alone does not establish valid
output. It neither repairs nor disables native physics.

Delegating native references to a second host and archiving their outputs is
done by the maintainers' own host tooling, which is not distributed. Native
fields may be identical while random NetCDF file identifiers differ; a witness
must state that distinction. A failed ENOSPC output folder can be retired only
after preserving file hashes/logs and checking no process still owns it. An
archive receipt (`ARCHIVE.json`) beside an unchanged MANIFEST/PUBLICATION binds
a lossless archive whose every output was extracted and hashed again; a digest
receipt alone never replaces an unverified retained archive.

`rolling_lbc.py` consumes full native intermediate inputs one valid time at a
time, fetching their exact original URL/size/SHA256 pins. It executes unchanged
native case9 separately for each valid time and retains each native manifest.
These independent native clocks have Time=0 and the requested xtime; they are
not represented as a continuous multi-time case9 execution. SEGMENTS.json
records the per-clock provenance. No field is rewritten to imitate another
clock mode. Surface runs continue to share the compact completed sequence.
`--workers 2` runs two distinct dates concurrently, decoding one clock at a
time per date. Every decoded intermediate is retained for other meshes and
Rust replay; encoded GRIB bytes retire after successful native decoding. Completed publications are reused only after every
published hash and the native case/binary contract match. The shared decoder
records interrupted staging-file hashes under its exclusive directory lock,
then reruns native ungrib; it preserves unrecognized files and reports an error.
`restore_shared_ungrib.py` centrally restores previously retired intermediates.
It fetches the original frozen encoded source, runs the identical native decoder
and Vtable under the shared directory lock, and accepts only the original
intermediate SHA256. Ordinary decoding still rejects a missing sealed input;
restoration is explicit and records the encoded transport and restored hashes.

Bounded disposable forecasts
---------------------------
For explicitly disposable runs only, `run_forecast.py --digest-only --comparator
RUST_READER` hashes every native output and every raw field before retirement.
The Rust `--field-digests` door reads all native numeric/character words in C
order and hashes their little-endian bytes, preserving signed zero and NaN
payloads. It changes no model source or fields. An output must be older than
the native current timestep (mpas_atm_core.F:760,827,862), or the successful
native process must have ended. Stored clocks and record counts must match;
a changing file is refused. CLOCKS.json is committed before each deletion.

An indexed row with `digest_only: true`, `comparator` and explicit
`inactive_packages` uses this mode. Original bytes are absent by design and
are never advertised as a retained full reference. Catalog storage metadata
and the `native-digest-reference-complete` tier state that limitation; actual
field SHA256s remain available. Full G2 references use preserved original bytes
and lossless archives. A clock receipt or matching SHA does not claim a measured
word-by-word port comparison.
Supplemental `native-full-physics-v3` decks include every required stream-list
file locally and pin their hashes in each INDEX entry. The native variant table
adds OML-enabled suites, all three native hml0 source branches, three gamma
values and enabled/disabled relaxation. Nonpositive hml0 modes require the
native initial OML state fields named in the table; the negative branch is
present in source even though Registry describes hml0 as nonnegative. Original
inactive OML-only and GWDO-only driver gates remain available. Active proof
variants name their surface-layer/PBL activation companions explicitly.

`prepare_ugwp_coordinates.py` invokes p14's explicit Rust metadata tool/source
paths and publishes a separate coordinate companion with original/result/tool
hashes and its unchanged-science-word receipt. Existing case7 files remain
immutable. `run_forecast.py --static-coordinate-directory <companion>` validates
that receipt against the selected original case before using the seekable copy.

The companion table `deck_companions.v842.json` records both control requirements
and driver activation/allocation dependencies. Its hash is sealed in each INDEX.
NGW requires its UGWP parent selector; cloud options include radiation; Tiedtke
options include the native PBL/surface allocations they read. Separate inactive
gate variants preserve native behavior. `native_call_counts.py` instruments only
an owned child and counts selected pristine binary function entries. These counts
prove activation, never port parity; the native binary hash and fixed rank are
recorded. No native source or scheme default is changed.

`reproduce_acceptance.py CHECKOUT --config CONFIG --work WORK` freshly runs the
sealed native/hex acceptance case at the explicit MPI process count. It invokes
the trial checkout's real comparator through the GPU/build slots, requires every
field and requested sample, and retires disposable native output only after
zero differing words. Deliveries include `receipts/reproduce.sh` and the exact
input/deck/binary configuration needed by the lander.

`field_inventory.py` publishes a separate sealed inventory of actual native
output samples without editing old reference manifests. Rust supplies every
field's stored type and shape; Registry.xml supplies its declared units, state
structure and scalar membership. Unknown declarations remain visible. Optional
`--samples` accepts restored archived files only after checking their original
publication hashes. Atmospheric prognostics and raw water bindings are explicit;
`water_integral_complete: false` records that current streams lack a complete
column/soil/snow/canopy storage plus cumulative source/sink budget. This inventory
does not certify a whole-physics water integral or forecast identity.

`frozen_download.py` retrieves a single explicitly pinned public file using at
most four simultaneous HTTP byte ranges. It verifies each Content-Range and
length, assembles bytes with `cat`, then requires the original complete size and
SHA256 before atomic publication. Temporary storage is bounded to that file;
GRIB interpretation remains in native ungrib. Rolling LBC runs retain one
transport receipt per clock and reuse their completed native segment checkpoints.

Supplemental dry inputs use `word_compare`'s `copy_zero_fields` Rust binary and
`prepare_dry_inputs.py` with caller-named `--field` arguments. The copier retains
the original NetCDF bytes, sets only those floating variables to stored IEEE +0,
and reads them back exactly. A separate all-field Rust comparison requires every
unselected native field to stay word-identical before publication. The input
publication records original/result hashes, source publication, selected fields
and both tool hashes. This creates a new explicit input product; original native
initializations and thermodynamic fields remain unchanged.

`publish_runtime.py` copies the exact selected MPICH launcher, Hydra proxy and
shared libraries into a sealed `ref/native/<hash>` publication. RUNTIME.json pins
every source byte and the actual MPICH version. Set `LD_LIBRARY_PATH` to its
`lib/` and invoke its `bin/mpiexec`; compiled native executables stay unchanged.
Loader receipts and a complete 24-hour deck comparison are separate evidence:
copying identical runtime bytes alone does not assert scientific identity.

GF companion decks select a PBL and its surface layer because the pristine
convection driver reads `kpbl`, `hfx`, `qfx`, `rthblten` and `rqvblten`, whose
Registry allocations require these packages. The original suite-none plus GF
selection remains a recorded native first-call failure. This is an explicit
single-option deck dependency, without changes to suite defaults or source.

`publish_acceptance_inputs.py` deduplicates the explicitly supplied dry proof
inputs by SHA256 into a sealed reference cache. Its rewritten proof configuration
changes paths only, preserving every input hash. Keep the original executed
configuration with the actual proof receipts; publication is not a new forecast
or a substitute for running `receipts/reproduce.sh` on the trial checkout.

The relocation witness runs both native decks on the proof host at MPI1 and compares
all124 stored output files after authenticating their lossless archives. It
finds407,207,028 stored words identical; whole-file bytes differ. This does
not extend the separately measured cross-host or process-count scope. The
delivery reproducer uses sealed inputs and MPI libraries; its executed legacy
configuration and byte-identical path relocation receipt remain separate.

If a shared host's GPU queue is replaced and cancels a completed native run's pending GPU
comparison, `reproduce_acceptance.py --reuse-native DIR --executed-config FILE`
validates the original execution's binary, initial state, ranks, launcher bytes,
decks and sample contract. It reuses only complete original native output;
retired native samples are regenerated. Every GPU comparison runs anew against
the supplied checkout. A checkpoint pins output hashes before queueing, and
the actual GPU comparison must leave those hashes unchanged. Normal delivery
`receipts/reproduce.sh` always executes the complete native forecast itself.

Reference consumers must freeze their input publication before submitting a
forecast. A native publication may retain its original outputs in the lossless
archive named by ARCHIVE.json. Restore it into your own directory and verify
every original MANIFEST output SHA256 plus the PUBLICATION and archive hashes. If using live originals, copy and authenticate
them before queueing GPU execution; do not depend on shared disposable output
paths remaining present during a long run. Compare against that owned, frozen
copy. Clock/digest-only products state that contract explicitly and cannot
substitute for original bytes in an original-byte comparison.

The completed f0 foundation evidence gate checks61 explicit artifact, input,
deck, forecast and comparison contracts. Actual host dycore acceptance fixes
native16 MPI ranks and hex1 GPU worker, with12 fields compared after one step
and at6/12/18/24 hours. Native physics rank variation is measured separately:
updraft_helicity_max varies between1 and24 ranks. Use the process count in
each native publication for scheme comparisons. The additional coupled
native-cpu mesoscale run completed24 hours but did not pass bitwise; its
pressure startup difference and missing field samples remain integration
evidence, without a widened tolerance or a CUDA physics claim.
