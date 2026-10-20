# Resident point-cull members

`gpuwm-hex forecast --members roster.json` runs one to four member stacks in
one process on one visible card. It is an opt-in path pending a complete cold
and warm timing and full artifact identity gate. It has no qualified
speed claim or default-selection rule yet.

The roster is a JSON array, for example:

```json
[{"run_id":"member-0","init":"/inputs/zero/init.nc","lbc_dir":"/inputs/zero/lbc"},
 {"run_id":"member-1","init":"/inputs/one/init.nc","lbc_dir":"/inputs/one/lbc"}]
```

All ordinary forecast flags remain required and shared. Every member uses
the same pinned mesh, static files, backend and schedule. The roster may
override its run ID, initialization, boundary directory, initialization
source description, registered preset and analysis bundle. Presets resolve through the
existing metadata table and its scientific admission. Output and scratch directories gain one
child per run ID; `--out` and its scratch must be absent, and their parent must exist.
Partition-rank and preflight execution are separate doors.

A member record's `"analysis"` names that member's analysis bundle
(docs/analysis-bundle.md): the member starts from it exactly as a one-forecast
`--analysis` run does (direct insertion, or its background with the increment
over the bundle's window under `--iau on`), and its derived files go to
`<out>/<run_id>.analysis-work`. A roster-wide `--analysis` starts every member
that names no `init` or bundle of its own; a member that also names an `init`
is refused, because the two would both name its initial state.

```json
[{"run_id":"member-0","analysis":"/bundles/member-0"},
 {"run_id":"member-1","analysis":"/bundles/member-1"}]
```

The existing numerical function bodies and forecast step loop are reused.
Private proof namespaces isolate initialization pins, surface census and
history expectations. Each member has its own nonblocking stream, regional
scrub observer, atmosphere, physics workspaces and graph controller. The
existing physics resource scope isolates mutable radiation caches.

Content-identical host mesh, vertical, reference and terrain arrays share
read-only mappings. Available `/dev/shm` is checked before each buffer, and
POSIX allocation reserves its space before mapped pages are touched. Missing
capacity or an allocation race falls back to files in the supplied scratch
directory. `--member-buffers nvme` forces that route. NVMe is a deployment
property of that scratch directory, rather than a device inferred by this
module. The ledger names the selected storage and fallback reason; the group
receipt lists buffer hashes, byte counts, placement and timings. Only files
created by the pool are removed, with path and byte receipts.

When initialization bytes, full boundary bytes, paths, metadata and all host
configuration inputs match, the complete sealed preparation is reused in
process. Each member receives private host state and private derived proof
pins. An object without a safe host clone is prepared independently, with
the fallback recorded. Distinct input bytes or presets retain independent
preparation. The existing preparation-cache and scientific functions remain
unchanged.

The regional driver's explicit immutable residency list permits sharing the
matching device mesh, vertical, reference and terrain tables. All constructor
input bytes, dimensions and dtype arguments participate in their keys.
Numerical state and saved diagnostics remain private even when equal.
Compiled hex modules are shared under a lock. Kernel observation wrappers
remain private, since sharing a regional observer would scrub another
member's padded state with the wrong registration.

Admission is a table of mesh kind, cell/level bounds and card memory tier.
It charges every member's full solo memory floor, with no unmeasured sharing
discount. An excessive roster refuses before device construction, naming the
construction and first-radiation out-of-memory breakage it prevents. The
current point-cull rows cover at most 50,000 cells and 55 levels; resident
execution currently requires exactly 55 levels.

Every member emits `forecast.construct`, `forecast.first_step` and
`forecast.integrate` with its own run ID and physical card. Use
`gpuwm timeline` over the combined ledger to union integration intervals.
The benchmark tool retains raw history only on the box until the identity
comparison completes. Forecasts write no checkpoint files; carried-state
digests include the backend restart state, and this does not establish a
resumed checkpoint run.
