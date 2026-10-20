# Country configuration

`gpuwm-hex country --row philippines --cycle 2025-11-03T00 --work-dir ./country`
runs prepare, forecast, products and score in order. `--stage all` is the default;
`--stage plan` prints the concrete plan without running it. The forecast stage runs
the installed `gpuwm` engine; `--gpuwm-checkout DIR` runs a source tree instead and
records its identity. The table connects a mesh specification, the GFS init row,
the vertical grid, a scorecard region and a standard product set. Adding a country
changes data only.

Use `--stage prepare`, `--stage forecast`, `--stage products` or `--stage score`
to run just that stage. Repeat `--skip STAGE` to leave stages out of the default
sequence, for example `--skip products --skip score`. Stages run in the same order
when others are omitted. Forecast, products and score need the registered mesh
from prepare. Static preparation defaults to `rw_mpas_static`'s geography lookup,
the same root `gpuwm fetch-geog` stages. Use `--geog WPS_GEOG` to override it.
If the archive is absent, run `gpuwm fetch-geog --datasets mesh`. You can also
supply existing `--grid`, `--static` and `--init` to register and reuse them instead.

Run forecasts through the host's GPU queue. For separate CPU and GPU scheduling,
run prepare first, forecast inside the queue, then products and score after
releasing the lease. Repeat the same cycle, work directory and input arguments.
Each invocation writes `country-plan.json`, `country-receipt.json`, `status.json`,
`events.jsonl` and `country.log`. The log records the same `stage-start` and
`stage-end` events as the global door; the receipt records each executed stage's
return code and wall time. Forecast receives the row's observation configuration
before it starts. Score joins that forecast's automatic scorer instead of scoring
the same run twice. Incomplete observation coverage stays explicit in the card.

`--observation-config` supplies local observation and reference files. The country
region remains the row's region. Missing observations remain explicit scorecard
rows. The country door does not claim observation-qualified skill or fetch a
private source. Its init-source door uses the public source row.

The Rust products door owns the standard meteorological products, point soundings
and meteograms. Its country product set declares views and points as data. Binary
overrides follow the existing products door. No country name selects a code path.
