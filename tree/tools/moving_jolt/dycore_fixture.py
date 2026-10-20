from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
from scipy.sparse import diags
from hexcore.mesh import Mesh
from hexcore.rust_geometry import derived_geometry
from hexcore.vertical import build_vertical_grid, build_edge_vertical_metrics
from hexcore.terrain import build_terrain_coupling
from hexcore.driver import (
    DryReferenceState,
    TerrainMetrics,
    DrySavedDiagnostics,
    _pressure_exner,
    _pressure_perturbation,
    DryDycoreDriver,
)
from hexcore.config_v841 import V841DryDycoreConfig
from hexcore.dynamics_v841 import V841ReferenceWindProfiles
from hexcore.state import PrognosticState
from hexcore.transport import build_advection_coefficients, initialize_deriv_two
from hexcore.moving_remap import ConservativeMap
from hexcore.moving_vertical import HeightConservativeMap
from hexcore.moving_state import remap_atmosphere


def setup(path="experiments/grid.nc"):

    mesh = Mesh.from_netcdf(Path(path))
    scale = 6371229.0 / float(mesh.attrs["sphere_radius"])
    for entity in ("Cell", "Edge", "Vertex"):
        for axis in ("x", "y", "z"):
            mesh.arrays[axis + entity] *= scale
    for name in ("dcEdge", "dvEdge", "nominalMinDc"):
        mesh.arrays[name] *= scale
    for name in ("areaCell", "areaTriangle", "kiteAreasOnVertex"):
        mesh.arrays[name] *= scale * scale
    mesh.attrs["sphere_radius"] = 6371229.0
    mesh.arrays.update(derived_geometry(mesh))
    for name, a in list(mesh.arrays.items()):
        if a.dtype.kind == "f":
            mesh.arrays[name] = np.ascontiguousarray(a, dtype=np.float32)
    mesh.arrays["nominalMinDc"] = np.asarray(
        mesh.arrays["nominalMinDc"].reshape(-1)[0], dtype=np.float32
    )
    for name in ("fEdge", "fVertex"):
        mesh.arrays[name] = (
            2 * 7.29212e-5 * np.sin(mesh.arrays["lat" + name[1:]])
        ).astype(np.float32)
    for name in ("defc_a", "defc_b"):
        mesh.arrays[name] = np.zeros(mesh.arrays["edgesOnCell"].shape, np.float32)
    mesh.arrays["deriv_two"] = initialize_deriv_two(mesh)
    cfg = V841DryDycoreConfig(
        config_dt=30.0,
        config_split_dynamics_transport=True,
        config_dynamics_split_steps=3,
        config_number_of_sub_steps=6,
        config_smdiv=0.0,
        config_divergence_damping=False,
        config_terrain_following=True,
    )
    coeff = build_advection_coefficients(
        mesh, config_scalar_adv_order=3, n_vert_levels=8
    )
    winds = V841ReferenceWindProfiles(np.zeros(8, np.float32), np.zeros(8, np.float32))
    return mesh, cfg, coeff, winds


KW = dict(rgas=287.0, cp=1004.5, reference_pressure=1e5)


def fixture(mesh, terrain, amp=0):
    nc = len(mesh.arrays["areaCell"])
    ne = len(mesh.arrays["dcEdge"])
    nl = 8
    v = build_vertical_grid(
        mesh,
        terrain.astype(np.float32),
        n_vert_levels=nl,
        ztop=30000.0,
        terrain_smoothing_passes=0,
        smooth_surfaces=False,
        xnutr=0,
    )
    v = replace(
        v,
        **{
            name: np.ascontiguousarray(getattr(v, name))
            for name in v.__slots__
            if isinstance(getattr(v, name), np.ndarray)
        },
    )
    for name in ("dzu", "rdzu", "rdzwp", "rdzwm", "fzm", "fzp"):
        getattr(v, name)[0] = 0.0
    z = 0.5 * (v.zgrid[:-1].astype(float) + v.zgrid[1:])
    p = 1e5 * np.exp(-9.80616 * z / (287.0 * 288))
    r = (p / (287.0 * 288) / v.zz).astype(np.float32)
    h = (p / (287.0 * 288) * 288 / (p / 1e5) ** (287.0 / 1004.5) / v.zz).astype(
        np.float32
    )
    s = PrognosticState(
        r,
        h,
        np.zeros((nl, ne), np.float32),
        np.zeros((nl + 1, nc), np.float32),
        np.full((1, nl, nc), 0.001, np.float32),
        0.0,
    )
    p, ex, theta = _pressure_exner(s, v, **KW)
    ref = DryReferenceState(r.copy(), h.copy(), p.copy(), ex.copy())
    pattern = np.cos(mesh.arrays["latCell"]) * np.sin(2 * mesh.arrays["lonCell"])
    s.rho_theta[:] *= (1.0 + amp * pattern[None, :] * np.exp(-z / 6000.0)).astype(
        np.float32
    )
    metric = build_edge_vertical_metrics(mesh, v, theta_adv_order=3)
    t = build_terrain_coupling(
        n_edges_on_cell=mesh.arrays["nEdgesOnCell"],
        edges_on_cell=mesh.arrays["edgesOnCell"],
        cells_on_edge=mesh.arrays["cellsOnEdge"],
        zb=metric.zb,
        zb3=metric.zb3,
        config_coef_3rd_order=0.25,
    )
    terrain = TerrainMetrics(
        np.ascontiguousarray(t.zb_cell, dtype=np.float32),
        np.ascontiguousarray(t.zb3_cell, dtype=np.float32),
    )
    prep = NS(mesh=mesh, vertical=v, reference=ref, terrain_metrics=terrain)
    driver = DryDycoreDriver(
        mesh,
        v,
        ref,
        V841DryDycoreConfig(config_terrain_following=True),
        terrain_metrics=terrain,
        reference_wind_profiles=V841ReferenceWindProfiles(
            np.zeros(nl, np.float32), np.zeros(nl, np.float32)
        ),
    )
    saved = driver._rebuild_saved_diagnostics(s)
    return s, saved, prep


def payload_of(s, saved, prep):
    state = {
        name: getattr(s, name)
        for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
    }
    state.update({name: getattr(saved, name) for name in saved.__slots__})
    return dict(
        state=state,
        vertical=prep.vertical,
        mesh=prep.mesh,
        time_seconds=s.time_seconds,
        scalar_names=["passive"],
    )


def mapper(src, tgt):
    a = np.asarray(src.mesh.arrays["areaCell"], float)
    h = ConservativeMap(diags(a).tocsr(), a, a, {})
    return HeightConservativeMap(
        h,
        src.vertical.zgrid,
        tgt.vertical.zgrid,
        src.vertical.dzw,
        tgt.vertical.dzw,
        endpoint_alignment=False,
    )


if __name__ == "__main__":
    mesh, cfg, coeff, winds = setup()
    s, saved, p = fixture(mesh, np.zeros(len(mesh.arrays["areaCell"])), amp=1e-5)
    driver = DryDycoreDriver(
        mesh,
        p.vertical,
        p.reference,
        cfg,
        terrain_metrics=p.terrain_metrics,
        reference_wind_profiles=winds,
        advection_coefficients=coeff,
    )
    result = driver.step(s, saved_diagnostics=saved)
    print(
        result.state.time_seconds, abs(result.saved_diagnostics.vertical_velocity).max()
    )
