# Init sources: one table, one path

`gpuwm-hex init-source` builds an initial condition from any row of
`src/hexcore/data/init-sources.v1.json`. Every row travels the same four
stages; no line of `src/hexcore/init_sources.py` names a source, and
`tests/test_init_sources.py::test_the_path_names_no_source` holds that.

```
acquire  ->  [remap]  ->  decode (met_intermediate)  ->  init door  ->  [order + case JSON]
```

```sh
python -m hexcore.cli init-source --list
python -m hexcore.cli init-source --source ifs-ens --member 1 --cycle 2026-10-03T12 \
  --out-dir OUT --grid GRID.nc --static STATIC.nc --vertical-spec SPEC.json \
  [--order-map ordering.nc --hex-order hex_order --case-template case.json] \
  [--staged DIR] [--writer met_intermediate] [--remapper gdt101_remap]
```

`--writer` and `--remapper` are optional. Without them the door finds both
binaries the way it finds every engine: its own variables
(`GPUWM_HEX_MET_INTERMEDIATE`, `GPUWM_HEX_GDT101_REMAP`), gpuwm's
(`GPUWM_MET_INTERMEDIATE`, `GPUWM_GDT101_REMAP`), the `libexec/bridges`
directory the gpuwm wheel installs them in, `~/.gpuwm/bridges`, then `PATH`.
`gpuwm-hex doctor` reports `met_intermediate`. The writer, the remapper and the
Vtable are all resolved before anything is downloaded.

The receipt `OUT/init-source-receipt.json` records the table digest, the row,
every product's address, byte count and SHA-256, the writer argv and digests,
and the init argv. The init door's own provenance receipt sits beside the init.

## What a row declares

| Key | Meaning |
|---|---|
| `products` | Files to acquire. `url` is a template or a list of mirror templates (`{yyyymmdd}`, `{hh}`, `{member}`, `{member_tag}`, set keys). `index` selects records of a large file by its JSON-lines index (`select` is a key/value map, or `"member"`). `transport: cds` is a Copernicus request. `codec: bz2` unwraps on arrival. `expand` repeats the product over one of the row's `sets`. |
| `members` | Control and perturbed member addressing: a filename tag, or index keys. |
| `sets` | Named lists of template variables (pressure levels, soil nodes, soil layers). |
| `remap` | For a source on an unstructured grid: mesh size, centre, the regular target, and which products carry latitude, longitude and land fraction. Each product then names its record selector and remap method. |
| `decode` | The Vtable (under the package's `data/vtables/`, so every wheel and sdist carries it) and the `met_intermediate` map-source label. |
| `first_guess` | `nfglevels`, `nfgsoillevels`, `use_spechumd`: facts of the source's ladder. |
| `vertical`, `surface`, `grid`, `limits`, `license` | What the source publishes, in words a reader can check. |
| `status: refused` + `refusal` | A source whose bytes cannot travel the path, with the breakage named. |

`run_switches` at the top of the table holds the init switches that belong to
the run and the mesh. They are the same for every row, so two sources differ
only in their meteorology.

## Adding a source

1. Inventory real bytes (`grib2_inventory`) and write the rows a Vtable needs.
   Reuse an existing Vtable when the centre's codes match.
2. Add the row. If the centre is new to `met_intermediate`, its closed
   map-source vocabulary needs a row there (the rules come from WPS `rrpr.F`).
3. Run `init-source`, then `initial-source-skill audit` on the init.

No Python is added. If a source needs a transform the path does not have, the
transform is added once as a declared stage (as `remap` was), never as a
branch on the source's name.

## Rows today

| Row | Products | Vtable / map source | First guess |
|---|---|---|---|
| `gfs` | one pgrb2 0.25-degree file | `Vtable.GFS` / `ncep-gfs` | 34 levels, 4 soil |
| `ifs` | one open-data oper file | `Vtable.ECMWF-OD.rw` / `ecmwf` | 15 levels, 4 soil |
| `aifs` | one open-data AIFS file | `Vtable.ECMWF-OD.rw` / `ecmwf` | 15 levels, 2 soil, specific humidity |
| `ifs-ens` | one member's records by index, plus the oper terrain record | `Vtable.ECMWF-OD.rw` / `ecmwf` | 15 levels, 4 soil |
| `gefs` | one member's pgrb2a + pgrb2b pair | `Vtable.GFS` / `ncep-gefs` | 32 levels, 4 soil |
| `era5` | CDS pressure-level + single-level GRIB1 | `Vtable.ERA5.rw` / `ecmwf` | 38 levels, 4 soil |
| `icon` | 120 bz2 records on the native icosahedral mesh, remapped | `Vtable.ICON.rw` / `dwd` | 19 levels, 8 soil |

The `icon` row needs the engine at or after the commit that gives
`gdt101_remap` a whole-sphere target and `met_intermediate` the `dwd` map
source and the soil-water-mass rule. Every other row runs on the unchanged
writer.
