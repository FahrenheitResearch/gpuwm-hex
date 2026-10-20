from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from hexcore.hrrr_intermediate import SourceRow, LatLonTarget, SourceWindow, IntermediateRefusal, SURFACE_LEVEL, loader_envelope

def regrid_hour(
    row: SourceRow,
    pieces: Mapping[str, Any],
    snapshot: Any,
    target: LatLonTarget,
    *,
    backend: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every WPS record for one valid time, from the engine's own operator."""

    lat, lon = target.mesh()
    plan = pieces["plan"](snapshot, lat, lon, backend)
    source = snapshot.fields
    pieces["ranges"](source)
    sina, cosa = pieces["rotation"](snapshot)
    sina = np.asarray(sina, dtype=np.float32)
    cosa = np.asarray(cosa, dtype=np.float32)

    def earth_wind(u_name: str, v_name: str) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(source[u_name], dtype=np.float32)
        v = np.asarray(source[v_name], dtype=np.float32)
        u_earth = u * cosa - v * sina
        v_earth = v * cosa + u * sina
        return (
            np.asarray(plan.apply(u_earth, method="parabolic"), dtype=np.float32),
            np.asarray(plan.apply(v_earth, method="parabolic"), dtype=np.float32),
        )

    records: list[dict[str, Any]] = []

    def record(name: str, units: str, description: str, level: float, values: np.ndarray) -> None:
        records.append({
            "field": name, "units": units, "description": description,
            "level": float(level), "values": np.asarray(values, dtype=np.float32),
        })

    # 3-D: level-indexed slabs with the pressure beside them.
    three_d = {
        window_name: np.asarray(plan.apply(source[window_name], method="parabolic"), dtype=np.float32)
        for window_name in row.atmosphere_3d
    }
    u3, v3 = earth_wind(*row.wind_3d)
    levels = int(three_d[next(iter(row.atmosphere_3d))].shape[0])
    if levels != row.levels:
        raise IntermediateRefusal(
            f"the {row.name} window carries {levels} levels and the row declares "
            f"{row.levels}; the level table this door writes would be wrong"
        )
    for k in range(levels):
        tag = float(k + 1)
        for window_name, (wps_name, units, description) in row.atmosphere_3d.items():
            values = three_d[window_name][k]
            if wps_name == "SPECHUMD":
                values = np.maximum(values, np.float32(0.0))
            record(wps_name, units, description, tag, values)
        record("UU", "m s-1", "U", tag, u3[k])
        record("VV", "m s-1", "V", tag, v3[k])

    # Surface.
    for window_name, (wps_name, units, description, method) in row.surface.items():
        values = np.asarray(plan.apply(source[window_name], method=method), dtype=np.float32)
        if wps_name in ("SPECHUMD", "SNOW"):
            values = np.maximum(values, np.float32(0.0))
        if wps_name == "LANDSEA":
            values = np.where(values >= 0.5, np.float32(1.0), np.float32(0.0)).astype(np.float32)
        if wps_name == "SEAICE":
            values = np.clip(values, 0.0, 1.0).astype(np.float32)
        record(wps_name, units, description, SURFACE_LEVEL, values)
    u10, v10 = earth_wind(*row.wind_10m)
    record("UU", "m s-1", "U", SURFACE_LEVEL, u10)
    record("VV", "m s-1", "V", SURFACE_LEVEL, v10)

    # Soil: the engine's own node-to-layer interpolation, moved by nearest
    # neighbour so land values stay land values; the init masks by LANDSEA.
    soil_t = np.asarray(source[row.soil_temperature], dtype=np.float64)
    soil_m = np.asarray(source[row.soil_moisture], dtype=np.float64)
    layers_t = pieces["interp_nodes"](soil_t, pieces["soil_nodes_m"], pieces["noah_midpoints_m"])
    layers_m = pieces["interp_nodes"](soil_m, pieces["soil_nodes_m"], pieces["noah_midpoints_m"])
    for index, (t_name, m_name) in enumerate(row.soil_layer_names):
        t_values = plan.apply(np.asarray(layers_t[index], dtype=np.float32), method="nearest")
        m_values = plan.apply(np.asarray(layers_m[index], dtype=np.float32), method="nearest")
        record(t_name, "K", f"T {t_name[2:5]}-{t_name[5:8]} cm below ground layer (Upper)",
               SURFACE_LEVEL, np.asarray(t_values, dtype=np.float32))
        record(m_name, "m3 m-3", f"Soil moisture of {m_name[2:5]}-{m_name[5:8]} cm ground layer",
               SURFACE_LEVEL, np.clip(np.asarray(m_values, dtype=np.float32), 0.0, 1.0))
    # The door also preserves the source depth nodes for later soil remapping.
    # Keep these separate from the interpolated Noah layers above, including
    # their distinct level tag and their unmodified source moisture values.
    for index, depth in enumerate(pieces["soil_nodes_m"]):
        cm = round(float(depth) * 100.0)
        for values, prefix, units in ((soil_t, "ST", "K"), (soil_m, "SM", "m3 m-3")):
            record(f"{prefix}{cm:03}{cm:03}", units, "Source soil value at depth node",
                   200100.0, plan.apply(np.asarray(values[index], dtype=np.float32), method="nearest"))
    return records, {
        "operator": getattr(plan, "operator", "unknown"),
        "levels": levels,
        "records": len(records),
    }

def reference_window(target: LatLonTarget, row: SourceRow, *, extra_cells: int = 2) -> SourceWindow:
    """The decoder crop every target point's stencil needs, on the source grid."""

    import importlib

    loader = importlib.import_module(row.loader_module)
    grid = loader.hrrr_source_grid()
    lat, lon = target.mesh()
    x, y = grid.latlon_to_ij(lat, lon)
    zero_x = np.asarray(x, dtype=np.float64) - 1.0
    zero_y = np.asarray(y, dtype=np.float64) - 1.0
    if not (np.isfinite(zero_x).all() and np.isfinite(zero_y).all()):
        raise IntermediateRefusal(
            f"the target box maps to non-finite {row.name} source coordinates; "
            f"it lies outside the projection's domain"
        )
    halo = int(row.parabolic_halo_cells)
    window = SourceWindow(
        i_start=int(np.floor(zero_x.min())) - 1 - extra_cells,
        i_end=int(np.floor(zero_x.max())) + halo + extra_cells,
        j_start=int(np.floor(zero_y.min())) - 1 - extra_cells,
        j_end=int(np.floor(zero_y.max())) + halo + extra_cells,
    )
    limit_x = int(grid.e_we) - 1
    limit_y = int(grid.e_sn) - 1
    if window.i_start < 0 or window.j_start < 0 or window.i_end > limit_x - 1 or window.j_end > limit_y - 1:
        south, west, north, east = loader_envelope(loader)
        raise IntermediateRefusal(
            f"the target box ({target.south:.3f}..{target.north:.3f} N, "
            f"{target.west:.3f}..{target.east:.3f} E) plus its interpolation "
            f"halo leaves {row.name} coverage: it needs source cells "
            f"i={window.i_start}..{window.i_end}, j={window.j_start}..{window.j_end} "
            f"and the native mass grid is i=0..{limit_x - 1}, j=0..{limit_y - 1} "
            f"(envelope {south:.2f}..{north:.2f} N, {west:.2f}..{east:.2f} E).  "
            f"Move the point inland, shrink the core, or use a source that "
            f"covers it"
        )
    return window
