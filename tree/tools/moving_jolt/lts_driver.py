"""Small dry two-rate LTS adapter using the production class executor."""

from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
from hexcore.fullstep_lts_classes import build_class_plan
from hexcore.partition_local_mesh_v841 import slice_prepared_host
from hexcore.cuda_fullstep_lts_active import FullStepLtsActive
from hexcore.cuda_driver import CudaDryDycoreDriver
from hexcore.driver import DrySavedDiagnostics
from hexcore.state import PrognosticState


def build(s, saved, p, cfg, coeff, winds, shift):
    mesh = p.mesh
    a = mesh.arrays
    decl = dict(mesh.variable_dimensions or {})
    for name in ("defc_a", "defc_b"):
        decl[name] = ("nCells", "maxEdges")
    mesh.variable_dimensions = decl
    base_dt = 15.0
    phase = shift * (s.time_seconds // 600)
    rates = np.where(np.sin(a["lonCell"] + phase) > 0, 1, 2).astype(np.int32)
    plan = build_class_plan(
        rates=rates,
        cells_on_cell=a["cellsOnCell"],
        n_edges_on_cell=a["nEdgesOnCell"],
        cells_on_edge=a["cellsOnEdge"],
        cells_on_vertex=a["cellsOnVertex"],
        area_cell=a["areaCell"].astype(float),
        dc_edge_m=a["dcEdge"].astype(float),
        base_dt=base_dt,
        edges_on_edge=a["edgesOnEdge"],
        n_edges_on_edge=a["nEdgesOnEdge"],
    )
    complete = NS(
        mesh=mesh,
        state=s,
        saved_diagnostics=saved,
        vertical=p.vertical,
        reference=p.reference,
        terrain_metrics=p.terrain_metrics,
        advection_coefficients=coeff,
    )
    drivers = []
    areas = []
    for table in plan.classes:
        local = slice_prepared_host(complete, table.layout)
        config = replace(
            cfg, config_dt=base_dt * table.rate, config_time_integration_order=2
        )
        d = CudaDryDycoreDriver.from_host(
            local.mesh,
            local.state,
            local.vertical,
            local.reference,
            config,
            saved_diagnostics=local.saved_diagnostics,
            terrain_metrics=local.terrain_metrics,
            advection_coefficients=local.advection_coefficients,
            reference_wind_profiles=winds,
        )
        drivers.append(d)
        areas.append(a["areaCell"][table.layout.cell_l2g])
    active = FullStepLtsActive(
        plan,
        drivers,
        base_dt=base_dt,
        kernel_cache=drivers[0].cache,
        area_cell=areas,
        dzw=p.vertical.dzw,
    )
    active.fixture_rates = rates
    return active


def advance(d):
    import cupy as cp

    d.advance_macro(2)
    assert d.registers_empty()
    names = ("rho", "rho_theta", "rho_u", "rho_w") + DrySavedDiagnostics.__slots__
    fields = d.gather(names)
    first = d.classes[0].driver.atmosphere.state.scalars
    scalar = np.empty((first.shape[0], d.nlev, len(d.fixture_rates)), np.float32)
    for klass in d.classes:
        layout = klass.tables.layout
        n = klass.n_owned_cells
        scalar[:, :, layout.cell_l2g[:n]] = cp.asnumpy(
            klass.driver.atmosphere.state.scalars[:, :, :n]
        )
    s = PrognosticState(
        *(fields[k] for k in ("rho", "rho_theta", "rho_u", "rho_w")),
        scalar,
        d.time_seconds,
    )
    saved = DrySavedDiagnostics(**{k: fields[k] for k in DrySavedDiagnostics.__slots__})
    return s, saved
