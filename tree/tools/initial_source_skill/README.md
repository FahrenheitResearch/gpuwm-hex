# Paired initial-source scoring instrument

This is a measurement tool. It adds no source-specific initialization path.
Stage this directory on the compute box, link `engine` to the read-only engine
checkout, and use that checkout's vendored Cargo dependencies with an owned
target directory. Build with `cargo build --release --offline -j 4`.

Run `initial-source-skill selfcheck` for the pressure bracket, longitude-origin
and wrap, and weighted RMSE checks. Run `initial-source-skill audit INIT.nc` to
record the actual initialization fills. Run `initial-source-skill CONFIG.json
scores.json` for the paired comparison. The config carries an ordered grid,
exactly two arms (each with an init and a history path per lead), leads with
valid times and field-only GRIB references, and plain IGRA sounding files.

The expected reference keys are z/t/u/v at 850, 500 and 250 hPa, plus surface
pressure `sp`. GRIB input must have exactly one field on a global regular
latitude/longitude grid. Units are metres, kelvin, m/s and pascals. The campaign
transport must bind each selected GRIB record to its publication index and
valid time; the scoring tool does not infer producer identity from its filename.

The temperature diagnostic is theta times (pressure / 100000) to the power
287.04/1004.5, matching the existing hex diagnostic. Height is the midpoint of
the init's zgrid. All variables interpolate linearly in log pressure between
bracketing model levels; there is no vertical extrapolation. A sampled pressure
must have exactly one decreasing crossing or one unique exact level. Ambiguous
crossings are omitted with the common mask. Pressure inversion counts and examples
are retained in frame health, including inversions outside the sampled levels.
Analyses interpolate
bilinearly onto the mesh cells. Scores use `areaCell`, with the same finite and
above-ground cells in both arms. Both reference surface pressures must put the
target level above ground. Regions are the globe, 20-90N, 20S-20N and 20-90S.

Radiosondes use nominal sounding times equal to each configured valid time,
header coordinates, and exact reported standard-pressure records. Missing and
quality-removed values (-9999 and -8888) are omitted. Blank/A/B processing flags
are accepted as NOAA documents. Temperature tenths of Celsius become kelvin;
wind speed tenths of m/s and direction from north become eastward/northward
components. Both arms use the same observations and nearest mesh cell. This
does not account for balloon drift, launch-time offsets or spatial interpolation
error. The report must give station/sample counts and the sampling design.

The output keeps per-observation values and input SHA-256 digests as receipts.
Scores describe the configured case. They do not establish general source skill
or a climatological anomaly correlation.
