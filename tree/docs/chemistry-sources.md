# Chemistry source preparation

`gpuwm-hex chem-sources` prepares inventory fluxes and composition for the
chemistry transport provider. Python chooses rows, fetches public files and
records hashes. `rw_hex_chem_sources` reads NetCDF, checks units and times,
maps grids, evaluates surface physics and writes the normalized NetCDF frame.
The binary is built and bundled by the wheel build. No model-specific Python
adapter is needed to add another compatible inventory or composition product.

```
gpuwm-hex chem-sources list
gpuwm-hex chem-sources fetch --source gbbepx-global-pm25 \
  --valid-time YYYY-MM-DDT00:00:00Z --output raw.nc
gpuwm-hex chem-sources prepare --source gbbepx-global-pm25 \
  --valid-time YYYY-MM-DDT00:00:00Z --input raw.nc --mesh mesh.nc \
  --output emissions.nc --threads 14 --map-output emissions-map.nc
```

The source table is `hexcore/data/chem-sources.v1.json`. `--table` selects a
user table with the same schema. `--species` selects an explicit subset of a
composition row without loading unrelated tracers. Unit strings are exact
and table-declared. The fetcher requires HTTPS and chooses the latest matching
creation revision for the requested time. Staged composition files avoid
embedding access credentials in source configuration.

RAVE rows cover North America and interpret documented no-fire -1 as zero.
Other missing emission values are refused. A declared QA mask has its rejected
mass in the receipt. RAVE uses the global coverage-start attribute because
published files may leave the CF time coordinate unwritten. The global GBBEPx
row supplies daily PM2.5 flux. Its CF midday timestamp is shifted by the
row-declared offset to the start of its daily coverage interval. Neither row
silently supplies another date when the requested file is absent.

Inventory mapping assigns each source cell's extensive mass to its nearest
hex cell, sums it, then divides by the physical hex area and source interval.
The receipt reports input, mapped, masked, outside and stored float32 mass.
An output rounding error over one part per million refuses preparation. MPAS
meshes with sphere_radius=1 are scaled by 6371229 squared; declared terrestrial
square-metre meshes retain their physical areas. A map output samples native
hex flux on a regular CF grid for `gpuwm.rustwx.run_cf_renderer`; it is a
presentation product and is not reimported into transport.

The GOCART surface rows require mesh-native fields with `nCells`, matching
`latCell`/`lonCell` in radians and an exact CF time. Dust requires 10 m wind, land flag, soil class, volumetric surface
moisture, air density and the two declared erodibility fields. The five-bin
parameters and wetness-dependent wind threshold follow the WRF public-domain
implementation. No erodibility climatology is invented. Sea salt follows the
Monahan arm of that implementation with four dry-radius bins. It uses explicit
open-water land flags and surface height; ice suppression must already be
represented by the supplied mask. Prepare one dated surface frame per source
interval needed by the forecast.

Composition preparation also needs the actual initialized model pressure:

```
gpuwm-hex chem-sources prepare --source merra2-aerosol \
  --valid-time YYYY-MM-DDTHH:00:00Z --input aerosol.nc4 \
  --humidity-input meteorology.nc4 --time-index INDEX \
  --mesh mesh.nc --pressure-file initialized.nc \
  --pressure-variables pressure_base pressure_p \
  --species smoke --output initial.nc --threads 14
```

The pressure file must carry matching latCell/lonCell and its native xtime
or an explicit valid_time_unix attribute. This prevents same-size pressure
from another grid or date being admitted. Coordinates must match at native
float32 storage precision. The pressure selectors are the sum
of model pressure components in Pa; use
the selectors actually present in the initialized file. When no Pa selectors
are supplied, composition rows declare a generic dry/moist equation of state:
physical dry rho, dry theta and dry-air qv yield
`p = p0 * (Rd*rho*theta*(1+(Rv/Rd)*qv)/p0) ** (cp/(cp-Rd))`.
Selectors, exact units and constants live in `target_pressure` metadata. The
vapor gas ratio uses the native float32 constant division. Reference rho/theta
are not added to the full physical fields. The receipt records the selected
operator, pressure range, initialized negative-qv count and minimum. Finite
initialized qv is preserved, including negative values awaiting the native
physics clamp; the EOS argument must remain finite and positive. Source-analysis
specific humidity still must be in [0, 1). This preprocessing does not claim
bitwise model diagnostics. The companion QV must
match the source coordinates, top-to-bottom 1..N level coordinate and exact
valid time. Moist-air source ratios are divided by 1-QV to produce dry-air
ratios. Horizontal mapping chooses the nearest source column. Vertical mapping
interpolates linearly in log pressure, with constant endpoint extension. This
preserves positive ratios but does not claim a conservative vertical layer
mass remap. Source grid spacing and the requested maximum distance must cover
every model cell. No absent composition is replaced by clean air.

The GEOS-FP row uses the same generic composition operator as an additional
public input route. It does not establish MERRA-2 or CAMS qualification.

The smoke composition is a carbonaceous aerosol proxy, including non-fire
carbon. It sums hydrophilic and hydrophobic black and organic carbon; it is
not source-attributed wildfire smoke. The MERRA-2 file specification describes
organic carbon here as particulate matter, so no extra organic-matter multiplier
is applied. Five source sea-salt bins are grouped into four target bins, retaining
the smallest source bin in the nearest first target bin. These interpretations
are explicit in the table and must remain in scientific receipts.

A normalized frame carries schema `gpuwm-hex.chem-source-frame/v1`, exact UTC
seconds, dry-air basis and the mesh SHA-256. Composition has `(nVertLevels,
nCells)` in kg kg-1 and emissions have `(nCells)` in kg m-2 s-1. Attach it using
the transport configuration's `inputs` block:

```json
{
  "valid_time": "YYYY-MM-DDTHH:00:00Z",
  "mesh_sha256": "SHA256_OF_PREPARED_GRID",
  "initial": {"path": "initial.nc"},
  "emissions": [
    {"path": "emissions.nc", "valid_time": "YYYY-MM-DDTHH:00:00Z", "interval_s": 86400}
  ],
  "zero_emission_species": [],
  "boundary": []
}
```

Each configured species needs initial composition and explicit source coverage,
or an explicit zero-emission declaration. Overlapping frames for one species
are refused. Steps crossing source intervals integrate their exact overlap.
Regional providers require bracketing composition and update prescribed cells;
the transport owner must also implement and admit its full relaxation-zone
forcing before a regional forecast can use this provider. Global forecasts
have no artificial lateral boundary. A source run is not evidence of transport
skill; the shared real plume forecast, mass ledger and rendered history are
separate acceptance evidence.

Source terms are attached to each row. WRF's public-domain notice is retained
in `data/licenses/WRF-GOCART-LICENSE.txt`. The binary's active Rust dependencies
select MIT and retain upstream notices in `data/licenses/chem-sources-dependencies`.
The NetCDF reader uses the system NetCDF-C ABI, seeded from the existing
Apache-2.0 reader in this tree. A table row marked `research_only` requires an
explicit matching `--research-license` value.

The `geos-fp-gocart-dust` inventory row imports five model-diagnosed GOCART
emission bins from the public GEOS-FP three-hour extended aerosol diagnostics.
It uses the same conservative inventory operator as NOAA fire emissions.
Its file timestamp is the midpoint of a three-hour mean: the download row's
`file_time_offset_s` maps an interval start to the file stamp, while the grid's
`valid_time_offset_s` maps the CF stamp back to the interval start. Calendar
directories follow the file stamp, including a date rollover. These are
modelled source estimates, with no independent dust skill claim.

For surface files with a numeric clock that loses seconds or remains zero,
`grid.time_text` and `grid.time_format` select the exact character timestamp
record. This is an explicit source clock authority, with the same exact instant
check and optional interval offset; it does not relax time matching.

`gocart-seasalt-native` binds the initialized native wind, terrain and land/ice
mask metadata directly. Its character `xtime` is authoritative; the initializer
currently writes numeric `Time=0` with a calendar origin. No field array is
rewritten to make the source fit.

Surface receipts include the integrated emission rate in kg/s using physical
mesh areas and stored fluxes. They do not infer interval mass from an
instantaneous wind field. Inventory receipts separately conserve interval mass.
