"""The forecast's start record, ``forecast-start.json``, written before the first step.

Which mesh and init a forecast runs on, its start, length and history cadence,
so a reader drawing each history hour as it lands (a site publishing a forecast
while it runs, ``gpuwm-hex add-nest`` joining a live parent) needs nothing but
the output directory, even when the door derived the init (an analysis bundle's
background, a remap onto a nest).

One writer for both physics routes: the ArWen-physics driver
(``drivers/run_cuda_v841_forecast``) and the MPAS-physics route
(``mpasphys/forecast_v842``).  THE BREAKAGE A SECOND WRITER WOULD BRING BACK:
until 2026-10-24 only the ArWen driver wrote it, so on the MPAS route
``add-nest`` had no live parent record and fell back to a receipt argv that a
bare ``gpuwm-hex forecast`` run does not have.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

START_RECORD_NAME = "forecast-start.json"
START_RECORD_SCHEMA = "gpuwm-hex.forecast-start/1"


def write_start_record(output_root: Path, *, paths: Mapping[str, Any], schedule: Mapping[str, Any]) -> Path:
    record = {"schema": START_RECORD_SCHEMA,
              "grid": str(Path(paths["grid"]).resolve()),
              "static": None if paths.get("static") is None else str(Path(paths["static"]).resolve()),
              "init": str(Path(paths["init"]).resolve()),
              "start_time": schedule.get("start_time"),
              # The run's length and history cadence are written at the START, so a
              # door that joins a live run (gpuwm-hex add-nest) reads them here
              # instead of from a receipt that is complete only after the run ends.
              "forecast_hours": schedule.get("forecast_hours"),
              "history_every_minutes": schedule.get("history_every_minutes"),
              "history_from_minutes": schedule.get("history_from_minutes")}
    target = Path(output_root) / START_RECORD_NAME
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(target)
    return target
