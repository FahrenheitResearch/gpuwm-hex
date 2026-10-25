# Full-storm evidence tools

These offline tools accompany `gpuwm-hex moving`, documented in
[`moving-nest.md`](../../docs/moving-nest.md). They do not add work to an engine
step and are not DA background producers. Run forecasts under the site's GPU
pool lease and use the same card, physics, compiler and initial member for the
moving and fixed arms. The case JSON owns storm-specific values.

```sh
gpuwm-hex moving stage --case CASE --casedir CASEDIR
gpuwm-hex moving prepare --case CASE --casedir CASEDIR --geog WPS_GEOG
gpuwm-hex moving run --case CASE --casedir CASEDIR --out MOVING --hours HOURS \
  --check-minutes 15 --rotate-template --forecast-arg=--no-products
gpuwm-hex moving fixed --case CASE --casedir CASEDIR --out FIXED --hours HOURS \
  --history-every-minutes 15 --forecast-arg=--no-products
```

`--rotate-template` requires concentric global rings. For a historical corridor
case, omit it and select `--whole-domain-remap` when qualifying the constrained
all-domain balance correction. The default corridor route uses zone carry and
is a different experiment. Neither option introduces lateral boundaries.

Wrap each producer with `run_guarded.py --root OWNED_ROOT --receipt EXIT.json --
COMMAND ...` inside the lease. Its asynchronous resource log records owned
process CPU time/RSS, sampled peak VRAM and disk use without CUDA synchronization.
The default disk stop is 28 GiB, leaving headroom below a 40 GB lane cap. Sampling
does not bound transients between observations. The source revision is supplied
through `GPUWM_HURRICANE_SOURCE_COMMIT`.

Launch `collect_moving.py --run RUN --out EVIDENCE --finished EXIT.json --window
WINDOW` outside the lease with devices hidden. Add `--fixed` for a fixed arm.
It reduces acknowledged histories, retains exact variable hashes and surface
fields, draws selected maps/sections with the ordinary Rust renderer, and only
then removes owned raw history. **Do not attach it to a DA background directory.**
The render lattice must stay fixed across members and arms. Failed collections
retain the raw history and return nonzero.

For long runs with a bounded member cache, pass
`--member-retirement-evidence EVIDENCE` to `moving run`. This opt-in removes only
inactive owned grids, initial states, centre lists and their overlap operators,
after all acknowledged histories have completed collector receipts and have
been retired. Active/source/prelaunch members and live preparation/operator
processes are protected. Static fields and build receipts remain for later
wind/rain rendering. A returned-to location is prepared again from staged inputs.
Rebuilding a retired location uses a new member identity, preserving the old
frame-to-static geometry binding. The controller writes `retired-members.jsonl`
before deletion. The member cache
must be inside this run; shared caches and symlinks are never retired. Do not use
this option when another workflow still needs old member grids or initial states.

After complete producer and collector exits:

```sh
python tools/hurricane/export_track.py --evidence EVIDENCE --case CASE --out TRACK
tc-verify export --input TRACK/fixes.json --storm STORM --model HEXM \
  --init YYYYMMDDHH --out moving.dat
python tools/hurricane/compare_off.py --off OFF_EVIDENCE --base BASE_EVIDENCE \
  --hours HOURS --cadence-minutes MINUTES --out off-bitwise.json
python tools/moving48/summarize_run.py MOVING --control FIXED --out jolt.json
python tools/hurricane/compare_cost.py --moving MOVING --fixed FIXED \
  --moving-exit MOVING_EXIT.json --fixed-exit FIXED_EXIT.json \
  --hours HOURS --cards CARDS --out cost.json
```

ATCF export rounds positions and intensities to ATCF precision. Missing fixes
stay missing; no temporal interpolation occurs. Native centering distances use
unrounded model fixes. Hex's reported maximum is instantaneous 10 m wind, whereas
the NHC/IBTrACS USA referee uses one-minute best-track wind. Preserve this
limitation in intensity comparisons. `compare_off.py` requires every scheduled
frame and compares every published history variable, including coordinates.
That is a whole-output identity check, not a claim about unrecorded device state.
The cost reader requires complete contiguous integration intervals and identical
GPU identities. It reports wall and integration costs separately; stage sums can
overlap in time and must not be added to the critical-path wall total.

`render_wind.py --run RUN --evidence EVIDENCE --window WINDOW --out WIDE_WIND`
redraws selected wind frames with a fixed 0-200 kt scale, checking retained fields
against their original hashes first. It refuses nonfinite or larger values.
The scale avoids the default map ladder's saturation in strong hurricanes.
Use `--keep-cf` to retain surface counters for the rainfall step.

`prepare_rain.py` takes `--start` and `--end` CF files from that tool, a Stage IV
daily `--observation` and its `--observation-url`, `--model`, explicit common
`--center-lon`, `--center-lat`, `--half-width-km`, and `--out`. It converts inches
to mm and uses the observation's `data_time` accumulation endpoint. The nominal
scalar time can be 12 hours earlier; both are recorded. Model endpoints must
exactly bracket the preceding 24 hours.

Each endpoint is conservatively remapped to the same 3 km equal-area grid before
subtraction. Missing observations and negative differences remain excluded;
negative differences mark the result incomplete. Rust overlap-volume receipts
apply to the common-grid remap. They do not bound the preceding native-mesh
product sampling error, which remains explicit in the receipt. `rain.json`
implements the shared `tc-verify` rain contract; `rain.cf.nc` retains the exact
common-support fields for `gpuwm.rustwx` weather maps. No score is computed by
this preparation tool. The scorer's FSS formulation cites DOI
10.1175/2007MWR2123.1.

`plot_evidence.py` takes a scored case JSON, cycle, duration and exported
centering CSV. It draws coordinate-track, intensity and centering charts only.
Use `--reference-label` to identify provisional sources explicitly; an IBTrACS
USA record marked `US-PROVISIONAL` is not a finalized best track. Agency flags
must be audited before scoring; original USA reports have `IFLAG[0] == 'O'`.
Inspect all output pictures across time before drawing a scientific conclusion.

`prepare_jma_reference.py --archive bst_all.zip --jma-id YYNN --storm WPnnYYYY
--out REFERENCE` converts the official JMA best-track archive to a separate
track-and-pressure ATCF reference. Its ten-minute winds remain in the source
audit but are absent from the scorer input. Preserve the source receipt and
attribute the derivative to the Japan Meteorological Agency under its published
PDL 1.0 / CC BY 4.0 terms. Do not merge this reference with USA wind records or
label an operational TC-vitals series as finalized best track.

Dependencies: the lane's normal hex/gpuwm environment, NumPy, netCDF4, pyproj,
matplotlib and psutil. Rain overlap uses the installed gpuwm `obs-score` bridge;
maps and sections use `gpuwm.rustwx`. Install extras in an owned environment.
When overriding an inherited `gpuwm` installation, install its exact matching
`gpuwm-data` companion too. A private environment using system site packages can
otherwise inherit different physics tables; the forecast correctly refuses that
mismatch. Record both versions and use the same pair for every comparison arm.
Stage IV source and terms: <https://water.noaa.gov/about/precipitation-data-access>
and <https://www.weather.gov/disclaimer>. Historical GFS archive attribution and
CC BY 4.0 terms are recorded in the `gfs-gdex` source-table row.

Moving carriers now use bounded parallel lossless protocol-5 buffer compression
(standard-library zlib, eight workers at most). `load_boundary` accepts both the
new codec and legacy pickle carriers after verifying their sealed SHA-256.
Every restart field is retained, including private halos and physics state;
no precision is reduced. Older readers cannot read the new codec. The storage
receipt records raw bytes, stored bytes, block size and worker count. This
changes moving-carrier storage only; ordinary fixed forecasts are unaffected.

With `--member-retirement-evidence`, cancelled speculative preparations that
have never supplied a forecast segment or frame are also retired after their
workers stop. Used members still require all closed collection receipts, and
their static geometry remains available for rendering.
