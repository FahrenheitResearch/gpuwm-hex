"""Real Rust scoring check against an analytical column and timestamp windows."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


def dump(root, fields):
    root.mkdir()
    variables = []
    for i, (name, (values, dimensions)) in enumerate(fields.items()):
        values = np.asarray(values, dtype="<f8")
        filename = f"{i:04d}.f64"
        values.tofile(root / filename)
        variables.append({"name": name, "filename": filename, "shape": list(values.shape),
                          "dimensions": dimensions, "dtype": "<f8"})
    (root / "metadata.json").write_text(json.dumps({"schema": "gpuwm-rw-netcdf-dump-v1", "variables": variables}))


def test_real_scorer_matches_linear_log_pressure_and_surface_values(tmp_path):
    binary = os.environ.get("GPUWM_HEX_NATIVE_SCORE")
    if not binary or not Path(binary).is_file():
        pytest.skip("real native-score executable is not built")
    geo, state = tmp_path / "geo", tmp_path / "state"
    dump(geo, {"latCell": ([0, 0.1], ["nCells"]), "lonCell": ([0, 0.1], ["nCells"])})
    pressure = np.array([100000, 85000, 50000], dtype=float)
    temperature = 270 + 2 * np.log(pressure / 50000)
    theta = temperature * (100000 / pressure) ** (287 / 1004.5)
    profile_dims = ["Time", "nCells", "nVertLevels"]
    fields = {"pressure": (np.tile(pressure, (1, 2, 1)), profile_dims),
              "theta": (np.tile(theta, (1, 2, 1)), profile_dims),
              "u_zonal": (np.full((1, 2, 3), 3), profile_dims),
              "v_meridional": (np.full((1, 2, 3), 4), profile_dims)}
    for name, value in (("t2", 301), ("u10", 3), ("v10", 4)):
        fields[name] = (np.full((1, 2), value), ["Time", "nCells"])
    dump(state, fields)
    stations, surface = tmp_path / "stations.json", tmp_path / "surface.json"
    stations.write_text(json.dumps({"stations": [{"station_id": "TEST", "latitude": 0, "longitude": 0}]}))
    surface.write_text(json.dumps({"reports": [{"station_id": "TEST", "observation_time": "2026-10-04T12:00:00",
        "values": {"temperature_2m": 300, "wind_speed_10m": 4}}]}))
    sonde, out = tmp_path / "sonde.csv", tmp_path / "scores.json"
    expected = 270 + 2 * np.log(70000 / 50000)
    sonde.write_text("station_id,latitude_deg,longitude_deg,level_pa,valid_time,variable,value\n"
        f"TEST,0,0,70000,2026-10-04T12:00:00Z,temperature_k,{expected}\n"
        "TEST,0,0,85000,2026-10-04T13:00:00Z,temperature_k,1\n"
        "TEST,0,0,110000,2026-10-04T12:00:00Z,temperature_k,1\n")
    subprocess.run([binary, str(geo), str(state), str(stations), str(surface), str(sonde),
                    "2026-10-04T12:00:00Z", str(out)], check=True)
    result = json.loads(out.read_text())
    assert result["metrics"]["surface.temperature_2m"]["bias_model_minus_observation"] == 1
    assert result["metrics"]["surface.wind_speed_10m"]["rmse"] == 1
    assert result["metrics"]["sonde.temperature_k.all"]["rmse"] < 1e-10
    assert result["metrics"]["sonde.temperature_k.all"]["n"] == 1
    assert result["skipped"]["time_outside_30_minutes"] == 1
    assert result["skipped"]["profile_not_bracketed_or_invalid"] == 1
