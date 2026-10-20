"""Chunked endpoint verification. No weather-field plotting or GPU work."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np


def pressure_level(p, values, target):
    """Log-pressure interpolation, refusing terrain/top extrapolation."""
    p = np.asarray(p, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    good = (np.isfinite(p).all(axis=1) & (p > 0).all(axis=1)
            & (np.diff(p, axis=1) < 0).all(axis=1)
            & (p[:, 0] >= target) & (p[:, -1] <= target))
    upper = np.clip((p < target).argmax(axis=1), 1, p.shape[1] - 1)
    # A target exactly equal to the top level has no strictly smaller value.
    upper[p[:, -1] == target] = p.shape[1] - 1
    lower = upper - 1
    rows = np.arange(len(p))
    fraction = ((np.log(target) - np.log(p[rows, lower]))
                / (np.log(p[rows, upper]) - np.log(p[rows, lower])))
    result = values[rows, lower] + fraction * (values[rows, upper] - values[rows, lower])
    result[~good] = np.nan
    return result


def bilinear(field, lat_axis, lon_axis, lat, lon):
    """Regular global GFS grid, cyclic longitude and either latitude order."""
    data = np.asarray(field)
    if lat_axis[0] > lat_axis[-1]:
        lat_axis, data = lat_axis[::-1], data[::-1]
    if lon_axis[0] > lon_axis[-1]:
        lon_axis, data = lon_axis[::-1], data[:, ::-1]
    fy = (lat - lat_axis[0]) / (lat_axis[1] - lat_axis[0])
    fy = np.clip(fy, 0, len(lat_axis) - 1)
    y = np.minimum(np.floor(fy).astype(int), len(lat_axis) - 2)
    dy = fy - y
    fx = ((lon - lon_axis[0]) % 360) / (lon_axis[1] - lon_axis[0])
    x = np.floor(fx).astype(int) % len(lon_axis)
    dx = fx - np.floor(fx)
    xx = (x + 1) % len(lon_axis)
    return ((1 - dy) * ((1 - dx) * data[y, x] + dx * data[y, xx])
            + dy * ((1 - dx) * data[y + 1, x] + dx * data[y + 1, xx]))


def cells(variable, begin, end):
    """Read only this cell slab, dropping the singleton Time dimension."""
    index = [slice(begin, end) if d == "nCells" else
             0 if d.lower() == "time" else slice(None) for d in variable.dimensions]
    dimensions = [d for d in variable.dimensions if d.lower() != "time"]
    array = np.asarray(variable[tuple(index)], dtype=np.float64)
    return np.moveaxis(array, dimensions.index("nCells"), 0)


class Metrics:
    def __init__(self):
        self.n = 0
        self.weight = self.error = self.square = 0.0

    def add(self, error, weight):
        valid = np.isfinite(error) & np.isfinite(weight) & (weight > 0)
        self.n += int(valid.sum())
        self.weight += float(weight[valid].sum())
        self.error += float((error[valid] * weight[valid]).sum())
        self.square += float((error[valid] ** 2 * weight[valid]).sum())

    def result(self):
        return dict(count=self.n, weight=self.weight,
                    bias=self.error / self.weight if self.weight else None,
                    rmse=(self.square / self.weight) ** .5 if self.weight else None)


def main():
    from netCDF4 import Dataset
    from scipy.spatial import cKDTree
    from gpuwm.grib2_stack import read_stack
    p = argparse.ArgumentParser()
    for name in ("case", "history", "analysis", "asos", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--fine-max-m", type=float, default=3300)
    p.add_argument("--earth-radius-m", type=float, default=6371229)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    case = json.loads(a.case.read_text())
    valid = (datetime.strptime(case["start"], "%Y-%m-%d_%H:%M:%S")
             .replace(tzinfo=timezone.utc) + timedelta(hours=6))
    analyses = {field.key: field for field in read_stack(a.analysis)}
    for field in analyses.values():
        if field.valid_time != valid:
            raise ValueError(f"wrong verifying time: {field.valid_time}")
    metrics = {name: Metrics() for name in ("z500", "t850", "u850", "v850", "wind850_vector")}
    with Dataset(case["grid"]) as grid, Dataset(case["init"]) as init, Dataset(a.history) as state:
        for ds in (grid, init, state):
            ds.set_auto_mask(False)
        lat = np.asarray(grid["latCell"][:]).reshape(-1)
        lon = np.asarray(grid["lonCell"][:]).reshape(-1)
        area = np.asarray(grid["areaCell"][:]).reshape(-1)
        if np.max(area) < 1:
            area = area * a.earth_radius_m ** 2
        fine = np.sqrt(2 * area / np.sqrt(3)) <= a.fine_max_m
        for begin in range(0, len(lat), 65536):
            end = min(begin + 65536, len(lat))
            pressure = cells(state["pressure"], begin, end)
            theta = cells(state["theta"], begin, end)
            temperature = theta * (pressure / 100000) ** (287 / 1004.5)
            zi = cells(init["zgrid"], begin, end)
            z = (zi[:, :-1] + zi[:, 1:]) * .5
            # Convert geometric to geopotential height using the stated sphere.
            height = a.earth_radius_m * z / (a.earth_radius_m + z)
            model = {"z500": pressure_level(pressure, height, 50000),
                     "t850": pressure_level(pressure, temperature, 85000),
                     "u850": pressure_level(pressure, cells(state["u_zonal"], begin, end), 85000),
                     "v850": pressure_level(pressure, cells(state["v_meridional"], begin, end), 85000)}
            errors = {}
            for name, values in model.items():
                f = analyses[name]
                reference = bilinear(f.array[0], np.asarray(f.latitude), np.asarray(f.longitude),
                                     np.rad2deg(lat[begin:end]), np.rad2deg(lon[begin:end]))
                errors[name] = values - reference
                metrics[name].add(errors[name], area[begin:end])
            metrics["wind850_vector"].add(np.hypot(errors["u850"], errors["v850"]), area[begin:end])
        xyz = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
        tree = cKDTree(xyz)
        surface = json.loads(a.asos.read_text())
        stations = {s["station_id"]: s for s in surface["stations"]}
        pairs = []
        for report in surface["reports"]:
            s = stations[report["station_id"]]
            la, lo = np.deg2rad([s["latitude"], s["longitude"]])
            distance, cell = tree.query([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
            if not fine[cell]:
                continue
            values = report["values"]
            def one(name):
                return float(cells(state[name], int(cell), int(cell) + 1).reshape(-1)[0])
            model_t = one("t2")
            model_w = np.hypot(one("u10"), one("v10"))
            def observed(name):
                value = values.get(name)
                return np.nan if value is None else float(value)
            pairs.append(dict(station_id=report["station_id"], cell=int(cell),
                              distance_m=float(2 * np.arcsin(min(1, distance / 2)) * a.earth_radius_m),
                              observation_time=report["observation_time"],
                              model_t2_K=model_t, observed_t2_K=observed("temperature_2m"),
                              model_wind_m_s=model_w, observed_wind_m_s=observed("wind_speed_10m")))
    asos = {}
    for name, model, observation in (("t2_K", "model_t2_K", "observed_t2_K"),
                                     ("wind_speed_m_s", "model_wind_m_s", "observed_wind_m_s")):
        m = Metrics()
        errors = np.array([row[model] - row[observation] for row in pairs])
        m.add(errors, np.ones(len(errors)))
        asos[name] = m.result()
    if pairs:
        with (a.out / "asos-pairs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    doc = dict(schema="hex.six-hour-score.v1", case=str(a.case), history=str(a.history),
               analysis=str(a.analysis), valid_time=valid.isoformat(),
               global_area_weighted={k: v.result() for k, v in metrics.items()}, asos=asos,
               fine_max_m=a.fine_max_m, fine_cells=int(fine.sum()),
               method="Log-pressure interpolation, no vertical extrapolation; dry theta to temperature; spherical geopotential height; cyclic bilinear GFS; area weights. ASOS nearest global cell inside fine mask, no height or land adjustment; archive endpoint screening supplies time match.",
               vector_bias_note="wind850_vector bias is mean vector-error magnitude")
    (a.out / "scores.json").write_text(json.dumps(doc, indent=2, allow_nan=False))
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
