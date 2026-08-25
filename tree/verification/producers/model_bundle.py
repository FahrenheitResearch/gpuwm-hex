#!/usr/bin/env python3
"""Repack one case's forecast into the referee's canonical grid bundle.

The unstructured history is put on a structured grid by `rw_mpas_convert`, the
Rust converter the render door drives: a k-d nearest-cell gather onto a named
render window. No Python touches the field data path; this file reads the
converter's output and packs it.

Four derivations happen here and all are stated in the bundle metadata,
because a bundle that derives silently is a bundle nobody can check:

* `precip_1h_mm` is the hour-to-hour difference of the model's own accumulated
  `RAINC` + `RAINNC`. The observation is an hourly accumulation and the model
  carries a run total; differencing consecutive frames is the only way the two
  are the same quantity. This is why an N-frame run yields N-1 scored times.
* `wind_speed_ms` is hypot(`U10`, `V10`). The station door reports a speed and
  the model carries the vector.
* `reflectivity_dbz` is the column maximum of the model's own `REFL_10CM`
  (converted from history `refl10cm`, computed inside the due step's WSM6
  call). MRMS's product is composite -- column-maximum -- reflectivity, so the
  column maximum is the like-for-like model quantity, and a per-column maximum
  commutes with the converter's whole-column nearest-cell gather.
* `dewpoint_k` is formed from `Q2` and `PSFC` by the engine's own
  dewpoint-from-mixing-ratio (rustwx-calc/src/derived.rs:653-658, transcribed
  exactly): e = max(q*p/(0.622+q), 1e-10 hPa), Td = 243.5*ln(e/6.112) /
  (17.67 - ln(e/6.112)). q2 is clamped nonnegative at this boundary only --
  the history stream preserves the scheme's occasional small negatives.

```sh
python verification/producers/model_bundle.py \
    --case CASE_ID --history-dir /abs/run/out --mesh /abs/MESH.grid.nc \
    --simulation-start 2025-07-14_12:00:00 --out-root /abs/case-root \
    --init-provenance "GFS 2025-07-14 12Z analysis; ..."
```
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

from mpas_port.obs_referee.bundle import write_grid_bundle
from mpas_port.obs_referee.canonical import write_json
from mpas_port.render_door import resolve_convert_exe

#: Every WRF name this producer reads. A converted frame missing any of them is
#: refused by name rather than packed with a hole: a frame without Q2 has no
#: model dewpoint and a frame without REFL_10CM leaves all three MRMS
#: reflectivity metrics unscorable, which is exactly the silence the first
#: referee run reported.  A history stream predating the refl10cm/q2
#: publication must be re-run, not repacked around.
REQUIRED = ("XLAT", "XLONG", "RAINC", "RAINNC", "T2", "U10", "V10", "PSFC",
            "Q2", "REFL_10CM")


def composite_reflectivity_dbz(refl_levels: "np.ndarray") -> "np.ndarray":
    """Column-maximum reflectivity: MRMS's composite quantity, from the
    model's own per-level REFL_10CM (level axis first)."""

    return np.max(refl_levels, axis=0)


def dewpoint_k(q2_kgkg: "np.ndarray", psfc_pa: "np.ndarray") -> "np.ndarray":
    """The engine's dewpoint-from-mixing-ratio, transcribed exactly from
    rustwx-calc/src/derived.rs:653-658, in kelvin."""

    q = np.maximum(np.asarray(q2_kgkg, dtype=np.float64), 0.0)
    pressure_hpa = np.asarray(psfc_pa, dtype=np.float64) / 100.0
    vapor_pressure_hpa = np.maximum(
        q * pressure_hpa / (0.622 + q), 1.0e-10)
    ln_e = np.log(vapor_pressure_hpa / 6.112)
    return (243.5 * ln_e) / (17.67 - ln_e) + 273.15


def digest(path: pathlib.Path) -> str:
    running = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            running.update(chunk)
    return running.hexdigest()


def convert(exe: pathlib.Path, history: list[pathlib.Path],
            out_dir: pathlib.Path, mesh: pathlib.Path, window: str,
            simulation_start: str) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [str(exe), "--mesh", str(mesh), "--out-dir", str(out_dir),
            "--window", window, "--field-set", "surface",
            "--simulation-start", simulation_start, "--clobber",
            "--json", str(out_dir / "convert-receipt.json"), "--history"]
    argv += [str(path) for path in history]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"{exe.name} failed ({completed.returncode}):\n"
                         f"{completed.stderr.strip()[-3000:]}")
    # The converter writes WRF's own naming, which carries no extension.
    frames = sorted(out_dir.glob("wrfout_*"))
    if len(frames) != len(history):
        raise SystemExit(f"{exe.name} wrote {len(frames)} frames for "
                         f"{len(history)} history files")
    return frames


def read_frame(path: pathlib.Path) -> dict:
    with Dataset(path, "r") as dataset:
        missing = sorted(set(REQUIRED) - set(dataset.variables))
        if missing:
            raise SystemExit(
                f"{path.name} lacks converted fields {missing}; it carries "
                f"{sorted(dataset.variables)}")

        def surface(name: str) -> np.ndarray:
            return np.asarray(dataset.variables[name][0], dtype=np.float64)

        text = np.asarray(dataset.variables["Times"][0]).tobytes().decode("ascii")
        when = dt.datetime.strptime(text.strip(), "%Y-%m-%d_%H:%M:%S")
        return {
            "time": when.replace(tzinfo=dt.timezone.utc),
            "latitude": surface("XLAT"), "longitude": surface("XLONG"),
            "accumulated_precip": surface("RAINC") + surface("RAINNC"),
            "temperature_k": surface("T2"),
            "wind_speed_ms": np.hypot(surface("U10"), surface("V10")),
            "surface_pressure_pa": surface("PSFC"),
            "reflectivity_dbz": composite_reflectivity_dbz(surface("REFL_10CM")),
            "dewpoint_k": dewpoint_k(surface("Q2"), surface("PSFC")),
            "absent_wrf_fields": str(getattr(dataset, "MPAS_ABSENT_WRF_FIELDS", "")),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--history-dir", required=True,
                        help="the run's output directory of cuda-history frames")
    parser.add_argument("--mesh", required=True, help="the MESH.grid.nc the run used")
    parser.add_argument("--simulation-start", required=True,
                        help="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--out-root", required=True,
                        help="the manifest's OBS_REFEREE_DATA_ROOT")
    parser.add_argument("--init-provenance", required=True,
                        help="one sentence naming where the initial condition came from")
    parser.add_argument("--arm", default="arwen-current")
    parser.add_argument("--window", default="focus",
                        help="a render window name the converter knows")
    parser.add_argument("--convert-exe", default=None)
    parser.add_argument("--scratch", default=None,
                        help="where converted frames land (default: beside the history)")
    args = parser.parse_args(argv)

    history_dir = pathlib.Path(args.history_dir)
    history = sorted(history_dir.glob("cuda-history.*.nc"))
    if len(history) < 2:
        raise SystemExit(f"{history_dir} holds {len(history)} history frames; an "
                         "hourly difference needs at least two")
    exe = resolve_convert_exe(args.convert_exe)
    mesh = pathlib.Path(args.mesh)
    scratch = pathlib.Path(args.scratch) if args.scratch else history_dir.parent / "converted"
    frames = convert(exe, history, scratch, mesh, args.window, args.simulation_start)

    read = sorted((read_frame(path) for path in frames), key=lambda item: item["time"])
    times = [item["time"] for item in read]
    gaps = {int((later - earlier).total_seconds())
            for earlier, later in zip(times, times[1:])}
    if gaps != {3600}:
        raise SystemExit(f"history frames are not hourly: gaps {sorted(gaps)}")

    accumulated = np.stack([item["accumulated_precip"] for item in read], axis=0)
    hourly = np.diff(accumulated, axis=0)
    # A run total that goes backwards is not a rate; it is a broken frame set.
    if float(np.min(hourly)) < -1e-6:
        raise SystemExit("accumulated precipitation decreases between frames "
                         f"(minimum hourly {float(np.min(hourly))} mm)")
    hourly = np.clip(hourly, 0.0, None)

    fields = {
        "precip_1h_mm": hourly,
        "temperature_k": np.stack([item["temperature_k"] for item in read[1:]], axis=0),
        "wind_speed_ms": np.stack([item["wind_speed_ms"] for item in read[1:]], axis=0),
        "surface_pressure_pa": np.stack(
            [item["surface_pressure_pa"] for item in read[1:]], axis=0),
        "reflectivity_dbz": np.stack(
            [item["reflectivity_dbz"] for item in read[1:]], axis=0),
        "dewpoint_k": np.stack([item["dewpoint_k"] for item in read[1:]], axis=0),
    }
    case_dir = pathlib.Path(args.out_root) / args.case
    case_dir.mkdir(parents=True, exist_ok=True)
    artifact, receipt_path = write_grid_bundle(
        case_dir / f"model-{args.arm}.npz",
        time_unix_s=np.asarray([int(when.timestamp()) for when in times[1:]],
                               dtype=np.int64),
        latitude_deg=read[0]["latitude"], longitude_deg=read[0]["longitude"],
        fields=fields, producer="gpuwm-hex",
        producer_version=f"model_bundle.py via {exe.name}",
        metadata={
            "render_window": args.window,
            "init_provenance": args.init_provenance,
            "simulation_start": args.simulation_start,
            "derivations": {
                "precip_1h_mm": "hour-to-hour difference of RAINC+RAINNC",
                "wind_speed_ms": "hypot(U10, V10)",
                "reflectivity_dbz": (
                    "column maximum of the model's REFL_10CM (composite, "
                    "the MRMS quantity)"),
                "dewpoint_k": (
                    "engine dewpoint-from-mixing-ratio on Q2/PSFC "
                    "(rustwx-calc/src/derived.rs:653-658, exact transcription; "
                    "q2 clamped nonnegative at this boundary only)"),
            },
            "converter_absent_wrf_fields": read[0]["absent_wrf_fields"],
        })
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_artifacts"] = [{
        "door": exe.name, "door_sha256": digest(exe),
        "mesh": mesh.name, "mesh_sha256": digest(mesh),
        "history_frames": len(history),
        "history_sha256": [digest(path) for path in history],
        "converted_sha256": [digest(path) for path in frames],
    }]
    write_json(receipt_path, receipt)
    json.dump({"artifact": str(artifact), "sha256": digest(artifact),
               "times": len(times) - 1,
               "shape": list(fields["precip_1h_mm"].shape[1:]),
               "max_hourly_mm": float(np.max(hourly)),
               "mean_hourly_mm": float(np.mean(hourly))},
              sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
