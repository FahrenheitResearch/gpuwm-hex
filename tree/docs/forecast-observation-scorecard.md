# Default forecast observation scorecard

`gpuwm-hex score RUN/fc` scores every retained history frame and writes
`RUN/fc/observation-scorecard/scorecard.json` and `scorecard.md`. Forecasts call
the same scorer on a CPU worker after their history writer has drained. The forecast receipt
records observation coverage separately from numerical forecast completion.

The native scorer must be built once against the engine's existing Rust
workspace. On a compute box:

```sh
python tools/build_obs_scorecard.py --engine ENGINE --out PRIVATE_BUILD --install
gpuwm-hex score RUN/fc
```

`--install` stages the binary under `hexcore/libexec`, already included by the
package-data rule. Set `GPUWM_HEX_OBS_SCORECARD` to an explicit native build
when running without staging it. No engine file is modified. No forecast is
started by scoring. Missing bridge binaries produce an explicit `NOT_MEASURED`
receipt. A score command returns 3 for incomplete coverage; forecast completion
retains its own return code.

Raw observations use `rw_obs::ncep_bufr` and `rw_obs::prepbufr`, the libraries
behind `rw_prepbufr`, plus `rw_asos` and `rw_mrms`.
Native netcrust and rustwx-io read forecast NetCDF and reference GRIB. Surface
wind is vector RMSE with speed bias. Station pressure comes from the native
ASOS altimeter inversion, never a comparison of station pressure with MSLP.
The station terrain difference must be at most 50 m before the hydrostatic
height adjustment is used.

Radiosonde QC, balloon drift coordinates, sensible-temperature conversion,
three-cell gnomonic weights and pressure interpolation are taken from
the observation referee's own script (obsref.py). Geometry is matched by spherical
position, so an ordered init can accompany an original-order history. All
sixteen mandatory levels have explicit rows. Height uses the original
referee's zgrid layer-midpoint operator. This remains an operator limitation.
The global GDAS dump needs a streaming native HTTP copy and the direct library
entry point because the old prepBUFR door has RAP-era transport/size bounds.
Reference levels absent from a public GRIB subset use log-pressure interpolation
between published bracketing pressure levels, with no extrapolation.

Precipitation intervals follow the common history/reference cadence. Six-hourly histories
are compared with six hourly MRMS QPE grids, not one hourly grid. The Rust
reader validates pack digests, units, masks and every interval endpoint. The
default is MRMS MultiSensor QPE Pass1; product selection is metadata.
The existing engine's paired FSS routine scores 1, 5 and 10 mm interval totals
at 30, 90 and 180 km full neighborhood widths, with complete common support.
Hourly histories use three-hour windows beside the default three-hourly IFS
archive. A configuration may set `precipitation_interval_hours` explicitly.
The fixed CONUS comparison lattice is 0.125 degrees, 125W to 66W and 22N to
50N. Widths use spacing at 36N and the receipt records actual rounded widths.
This latitude-dependent scale approximation and the point resampling do not
support a conservative precipitation-budget claim.

GFS and IFS forecasts use the forecast's initialization and lead. Secondary
analysis columns use step zero at the verifying time and can assimilate the
same reports. Missing reference fields stay visible and never erase hex's
available measurements. Every measured comparison uses common finite support
among the available arms. Station-block squared-error differences retain the
paired referee diagnostic. A single cycle does not establish independent-event
significance, even with many stations.

`--config metadata.json` can supply reference rows, public URL templates,
thresholds/scales, and decoded observation/reference paths by valid time.
Adding another model is a reference metadata row using the same native reader.
`--leads 12,24` selects an inspection window without changing its preceding
accumulation boundary. JSON carries every measured lead; the compact table
shows the latest measured lead per quantity, with counts.

Acquired inputs are private to the command and are deleted after the receipts
are extracted. The scorecard lists every deleted file and its bytes. Caller
histories, init files and decoded inputs are never deleted. Logs and native
receipts are retained. Failed sources are explicit missing rows, including
observations not yet posted for a future valid time. The default forecast hook
queues a CPU-only worker and returns immediately, so it does not retain a GPU
lease during observation retrieval. The worker scores leads as reports arrive;
the default wait is seven hours after valid time for the global GDAS dump and
is configurable with `observation_delay_hours`. A `.keep` marker names the
required history frames, and a hardlink holds the init without a second prepared
copy. The worker releases its own link and marker after writing the receipt.
On different filesystems the original init remains a stated input dependency.
`gpuwm-hex score` can also refresh the run manually.

Local parameter encodings and unit conversions are table rows in
`hexcore/data/observation-grib-parameters.json`. ECMWF's local total-precipitation
encoding is in metres and is converted to mm using its
[parameter definition](https://codes.ecmwf.int/grib/param-db/228).
Reference terrain uses the forecast cycle's step-zero geopotential. Public
mirror URLs are reference metadata; the default IFS route uses Google's mirror
and falls back to the AWS mirror.
