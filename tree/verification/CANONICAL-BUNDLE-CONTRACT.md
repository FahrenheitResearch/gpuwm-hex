# Canonical bundle contract

## Grid (`gpuwm-hex.canonical-grid/v1`)

A deterministic NPZ/ZIP with sorted `.npy` members and fixed ZIP timestamps:

- `schema_utf8`: one-dimensional `uint8` UTF-8 schema string;
- `metadata_utf8`: one-dimensional `uint8` canonical JSON object;
- `time_unix_s`: strictly increasing `int64`, shape `(time,)`;
- `latitude_deg`, `longitude_deg`: matching float arrays, shape `(y,x)` or
  `(cell,)`;
- `field__<canonical_name>`: float64, shape
  `(time, *latitude_deg.shape)`; NaN is missing, infinity is forbidden.

Canonical field names used by the production manifest:

```text
precip_1h_mm
reflectivity_dbz
temperature_k
dewpoint_k
wind_speed_ms
surface_pressure_pa
```

A rustwx producer is responsible for all raw format decode, quality control,
unit conversion, and accumulation-window semantics and records those choices in
receipt metadata. The referee does not guess them.

## Stations (`gpuwm-hex.canonical-stations/v1`)

Canonical JSON Lines. The first line is:

```json
{"metadata":{},"schema":"gpuwm-hex.canonical-stations/v1"}
```

Every later line has exactly:

```json
{
  "fields": {"temperature_k": 293.15},
  "latitude_deg": 35.2,
  "longitude_deg": -97.4,
  "station_id": "KOUN",
  "time_unix_s": 1716249600
}
```

Records are sorted by station ID, valid time, latitude, and longitude. Values are
finite numbers or null. Missing canonical fields remain missing; the referee
does not derive dewpoint from an undocumented humidity convention.

## Receipt

Each artifact has a sibling receipt, normally `<artifact>.receipt.json`, whose
`artifact_sha256` covers the exact bytes. Production manifests allow only named,
reviewed producers.
