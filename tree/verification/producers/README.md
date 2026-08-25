# Canonical-bundle producers

The referee reads two normalized, checksummed contracts and nothing else, and
[`../CANONICAL-BUNDLE-CONTRACT.md`](../CANONICAL-BUNDLE-CONTRACT.md) says a
rustwx producer is responsible for every raw decode behind them. Until
2026-08-25 no such producer existed: the only thing that had ever written a
canonical bundle was `../fixtures/build_synthetic_suite.py`, and a production
manifest is forbidden to accept a producer whose name begins with `synthetic`.
The contract therefore named a producer nobody could run, which is why the
production scorecard had never been anything but `NOT_MEASURED`.

These three files are that producer. None of them parses an archive format.

| file | what it does |
| --- | --- |
| `select_control_cases.py` | picks the two control cases out of the observation archive by a stated rule, so the manifest's dates are screened rather than chosen |
| `observation_bundles.py` | drives `rw_stage4`, `rw_mrms` and `rw_asos`, and repacks their packs into `canonical-grid/v1` and `canonical-stations/v1` |
| `model_bundle.py` | drives `rw_mpas_convert` onto a render window, and packs the model arm |

## The boundary they keep

Every GRIB2 and METAR byte is decoded by a Rust front door resolved through
gpuwm's own ladder (`gpuwm.obs.frontdoor`, `mpas_port.render_door`), and the
unstructured-to-structured gather is `rw_mpas_convert`. What these files do is
read the packs those doors wrote — with gpuwm's own pack reader, which re-proves
the payload digest — and write the referee's containers.

They change no value. They derive exactly two quantities, both on the model
side, both declared in the bundle's own metadata: hourly precipitation as the
difference of the model's accumulated `RAINC`+`RAINNC`, because the observation
is an hourly accumulation and the model carries a run total; and wind speed as
`hypot(U10, V10)`, because the station door reports a speed and the model
carries the vector.

They also refuse to invent. Three canonical fields come out absent rather than
substituted, and the referee reports each absence with its cause: the model has
no `refl10cm` and no `q2` in its history stream, and the ASOS door reports MSLP
rather than station pressure.

## Running them

Freeze the station table once — the door reads it rather than taking
coordinates off each fetch, because coordinates can move between the fetch that
registers a case and the fetch that scores it:

```sh
rw_asos stations --networks "$NETWORKS" --bbox -126,21,-66,53 --out stations.json
```

`gpuwm.obs.surface_networks.networks_for_bbox(west, south, east, north)` names
the networks a box needs.

Then, per case:

```sh
PYTHONPATH=src python verification/producers/observation_bundles.py \
    --case CASE_ID --window-start 2025-07-14T12:00:00Z --hours 24 \
    --out-root "$OBS_REFEREE_DATA_ROOT" --stations stations.json --cache CACHE

PYTHONPATH=src python verification/producers/model_bundle.py \
    --case CASE_ID --history-dir RUN/out --mesh MESH.grid.nc \
    --simulation-start 2025-07-14_12:00:00 \
    --out-root "$OBS_REFEREE_DATA_ROOT" \
    --init-provenance "one sentence naming where the initial condition came from"
```

and once all cases exist:

```sh
OBS_REFEREE_DATA_ROOT=/abs/case-root PYTHONPATH=src \
python tools/run_obs_referee.py run \
    verification/manifests/obs-referee-283.production.json \
    --output /abs/evidence/obs-referee-283-real
```

The first real run of that command is recorded in
receipt `../../evidence/obs-referee-283/RECEIPT.md` (see
[`../../evidence/EVIDENCE.md`](../../evidence/EVIDENCE.md)).
