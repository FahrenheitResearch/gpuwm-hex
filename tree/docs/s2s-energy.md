# Weekly S2S energy proxies

`gpuwm-hex s2s` includes energy products by default, at 6-hour history cadence.
`--hub-heights 87.5,153` selects arbitrary hub heights in metres above ground.
Heights must lie in (10,1000]. Out-of-column points remain missing. A daily
history request refuses because it cannot resolve the solar cycle.

The `data/s2s-energy.v1.json` rows define hub wind, surface downwelling shortwave
irradiance (GHI), turbine power and horizontal-array PV power. These are authored,
illustrative rows, not manufacturer performance data or calibrated plant forecasts.
`--energy-table PATH` replaces the rows. Adding a turbine or PV parameter set is
metadata work. Capacity is MW per row, with MWh integrated over each window.
There is no installed-capacity map, load model, irradiance-component decomposition,
plant observation verification or tilted-array transposition in this product.

The Rust reducer reads native NetCDF through the engine's netcrust reader and
writes CDF5 through rw-store. It interpolates wind **components** in geometric
height above each cell's own terrain, using the model's 10 m wind below the first
mass-level centre. It computes speed and table-driven conversion at each sample,
then reduces on the native grid with exact endpoint trapezoids. It does not
convert the mean wind to mean power. Turbine cut-out is explicit and density
correction is absent. PV uses a NOCT cell-temperature proxy, a linear temperature
coefficient, declared losses, inverter efficiency and clipping at AC capacity.
Six-hour samples are a sampled yield proxy, not an hourly generation integral.
Spherical nearest-cell remapping follows temporal reduction. All scheduled frames
and all roster members are mandatory. Missing points stay missing.

The product receipt records units, finite counts, ranges, cadence, geometry,
parallelism and wall times. Means, population spread, signed historical anomalies
and probability above the historical normal use a fixed ensemble denominator.
No historical normal means unavailable anomalies, never a zero anomaly or an
ensemble-relative historical claim. With one member, spread is zero and the
above-normal indicator is deterministic, rather than calibrated probability.

## Build

Use a complete read-only engine checkout as the first argument:

```sh
bash tools/build_s2s_energy.sh ENGINE OWNED_BUILD_DIR
bash tools/build_s2s_renderer.sh ENGINE OWNED_RENDERER_DIR
export GPUWM_HEX_RW_S2S_ENERGY=OWNED_BUILD_DIR/rw_s2s_energy
export GPUWM_HEX_RW_WRFBATCH=OWNED_RENDERER_DIR/rw_wrfbatch
export GPUWM_ENSEMBLE_RENDERER=OWNED_RENDERER_DIR/rw_ensbatch
```

The renderer build applies `0002-s2s-energy-scales.patch` only to its private
source copy and builds both renderers. The patch reads per-field declared scale
metadata from the aggregate. The route checks the renderer ABI for this feature.
All maps go through `gpuwm.rustwx.run_ensemble_product_renderer`.
An explicit positive `RAYON_NUM_THREADS` caps the route's CPU budget within the
process affinity. The route preserves a launcher's smaller thread allocation.

## Historical normal contract

`--energy-climatology PATH` names a NetCDF file on exactly the output grid,
including `XLAT(south_north,west_east)` and `XLONG` in degrees. Global attributes:

* `s2s_energy_table_sha256`: the SHA256 of the exact row file bytes.
* `s2s_energy_heights_m_agl`: JSON floating-point array of requested heights,
  for example `[87.5,153.0]`.
* `s2s_climatology_start_mmdd`: the forecast's start month/day, for example `03-31`.
* `climatology_year_start` and `climatology_year_end`: integer years, with at
  least two years and the last year preceding the forecast year.

For each requested window and expanded row, the file supplies a baseline plane
named `week_1_hub_wind_87p5m`, `week_1_wind_power_generic_87p5m`,
`week_1_wind_power_generic_87p5m_energy`, `week_1_solar_irradiance`,
`week_1_pv_power_horizontal` and `week_1_pv_power_horizontal_energy`, with declared
units `m s-1`, `MW`, `MWh`, `W m-2`, `MW` and `MWh`. The two-week windows use
`weeks_1_2`, `weeks_3_4` and, when covered, `weeks_5_6` prefixes. Normals must be
formed from the same conversion and sampling, including nonlinear conversion
before averaging; averaged weather fields cannot substitute for energy normals.
Grid, units, calendar, row identity and year leakage are checked before anomalies.
No historical energy normal or dated plant observations are bundled.

Mean maps use the conversion row's display limit. Anomaly maps use symmetric
limits spanning the finite anomaly range in each window, so small PV anomalies
remain visible. The native receipt records finite anomaly counts, extrema and
the display limit. An all-zero or unavailable anomaly retains the row limit.

The same Rust executable builds a reference from completed, calendar-matched
historical energy reductions:

```sh
rw_s2s_energy --build-climatology reference.json
```

The manifest has `target_start` (for example `2026-10-20_00:00:00`), `requests`
(paths to at least two historical `s2s-energy/request.json` files), and `output`
(the destination NetCDF path). The window aggregates beside those requests must
remain available. Conversion rows, heights, sampling, windows, coordinates,
units and aggregate provenance must match. Duplicate years, forecast-year
leakage and incomplete member rosters refuse. The reference averages the
already converted annual energy fields in parallel Rust and writes its year
list, reference type and timing receipt beside the result. A short hindcast
mean is labelled as such; it is not an observational climate normal.

## Dated forcing

Before generating forecast surface forcing, the S2S route expands every object
at the declared source cadence, including the endpoint. Each candidate object
must return HTTP 206, `Content-Range: bytes 0-15/SIZE` with positive full size, and
a 16-byte GRIB prefix. Mirror fallback and every error are retained in
`s2s-forcing-coverage.json`. A missing interior object refuses the full window.
This is object availability evidence; the existing Rust acquisition, decode and
surface-series writer still validate the consumed content. Reused native surface
series retain their object coverage receipt; older receipts acquire this check
before reuse. Supplied native series do not
claim that an external CFS archive was checked.

Prepared initial conditions are pinned into the run's `inits/` directory before
forecasting. On the same filesystem this uses a hard link, so producer cleanup
cannot remove a live input and no second payload copy is needed. A cross-device
source is copied. The run retains its source receipt and can resume after the
producer removes its file.
