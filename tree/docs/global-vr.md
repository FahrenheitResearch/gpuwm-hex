# Event-specific global variable resolution

`mesh-plan --global-mesh` sizes the whole sphere. `--point` makes the existing
halving ladder around any centre; `--spec` accepts the generator's authored
resolution spec, including boxes or polygons. Geography comes from
`rw_mpas_static`, and the generated pair is admitted and registered as a
byte-pinned runtime row. No shipped mesh or global-run registry edit is needed.

```sh
gpuwm-hex mesh-plan --global-mesh --point LAT,LON \
  --background-km 60 --fine-dx-m 15000 --radius-km 1800 \
  --card rtx-pro-6000 --hours 18 --out plan.json

gpuwm-hex mesh-plan --global-mesh --point LAT,LON \
  --background-km 60 --fine-dx-m 15000 --radius-km 1800 \
  --card rtx-pro-6000 --generate --run --source SOURCE --cycle YYYY-MM-DDTHH \
  --hours 18 --history-every-minutes 180 --geog WPS_GEOG --out-dir RUN \
  --region event=region:LAT,LON,5000,3500
```

`SOURCE` is an existing global init-source table row. New sources remain table
work. `--source-table`, `--member` and repeated `--staged` pass through to that
door. No meteorological decode, regrid, init or rendering is implemented here.
The default 55-level vertical declaration is the global door's bundled tc55-v1.
The run keeps normal forecast defaults, including fast numerics and physics.

The plan includes device admission's predicted peak, its explicit margin,
required free memory and model provenance. Completed whole-run observations in
`data/card-fit-observations.v1.json` also supply a default memory floor for the
same card and level count. The floor never lowers device admission: it retains
the observed footprint below its cell count and scales it by cells above. This
is a conservative single-anchor extrapolation, not a multi-size allocation fit.
The underlying model and observation remain separately visible in the plan. `96gb`, `96gib`, `rtx-pro-6000` and
`96gib-188sm` refer to the actual 188-SM part. The old generator's affine card
table is not used to price the forecast. A derived memory row remains labelled
derived. Runtime scales a measured rate on the same card by cells and steps,
with its reference row and assumptions retained. This is an extrapolation,
not a measured promise about a new mesh or card count.

`RUN/global-vr-receipt.json` is updated atomically at each stage. It includes
the prediction before generation, the prediction using emitted cell count
and admitted timestep, byte digests, the source receipt, exact stage argument
vectors, wall and CPU time, observed thread count and per-stage process VRAM.
`predicted_vs_measured` compares forecast measurements with the prediction.
Logs and 1-second telemetry CSVs are under `RUN/logs/`. Process VRAM includes
only the stage's process group, so other tenants are excluded. Sampled peaks
can miss subsecond allocations. Missing GPU telemetry stays null.

Failures retain a failed receipt and log path. Existing run directories are
refused to preserve evidence. Cancellation terminates the stage process group.
The normal forecast door rechecks the generated row and actual free memory
at launch. Completed forecasts call the existing Rust products door, including
global and regional views. Generating a mesh earns no forecast-skill claim.

The shipped observation includes a completed 216,189-cell, 55-level, 18-hour
60/30/15 km global run on the 188-SM PRO 6000. Its sampled process peak was
25,272 MiB; the original allocation model predicted 15,279.989 MiB. At that
shape the default floor plus the existing margin requires 28,475.2 MiB, so a
24,000 MiB budget now fails the fit check. Other sizes remain extrapolations.
The original receipt retains both the prediction error and the measured wall;
shared-card wall time is not an isolated speed benchmark.
