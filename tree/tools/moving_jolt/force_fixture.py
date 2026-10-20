"""Resting and synthetic pressure/moist-loading native-force remap diagnostics.

No integration or physics call. One generated small mesh is held fixed to
isolate vertical-coordinate transfer. Zero momentum does not qualify general
momentum reconstruction. The driver AST supplies its vertical Euler block.
"""

import ast
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import diags
from scipy.optimize import brentq
import hexcore.driver as native
from hexcore.dynamics import pressure_gradient_euler_tendency
from hexcore.mesh import Mesh
from hexcore.moving_remap import ConservativeMap
from hexcore.moving_state import remap_atmosphere
from hexcore.moving_vertical import HeightConservativeMap
from hexcore.state import PrognosticState
from hexcore.vertical import build_vertical_grid
from hexcore.regional_v841 import compute_moist_coefficients

root = Path.cwd()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--domain-reconciliation", choices=("global", "local"), default="global"
)
options = parser.parse_args([])
prefix = "" if options.domain_reconciliation == "global" else "local-domain-"
started = time.monotonic()
grid = root / "experiments/grid.nc"
mesh = Mesh.from_netcdf(grid)
xyz = np.stack([mesh.arrays[key] for key in ("xCell", "yCell", "zCell")], axis=-1)
radius = float(np.linalg.norm(xyz, axis=1).mean())
scale = 6371229.0 / radius
if radius < 100:
    for entity in ("Cell", "Edge", "Vertex"):
        for axis in ("x", "y", "z"):
            mesh.arrays[axis + entity] *= scale
    for name in ("dcEdge", "dvEdge"):
        mesh.arrays[name] *= scale
    mesh.arrays["areaCell"] *= scale * scale
xyz = np.stack([mesh.arrays[key] for key in ("xCell", "yCell", "zCell")], axis=-1)
nc = len(xyz)
edges = mesh.arrays["cellsOnEdge"]
ne = len(edges)
direction = xyz[edges[:, 1]] - xyz[edges[:, 0]]
mesh.arrays["edgeNormalVectors"] = (
    direction / np.linalg.norm(direction, axis=1)[:, None]
)
mesh.arrays["coeffs_reconstruct"] = np.zeros((*mesh.arrays["edgesOnCell"].shape, 3))
area = np.asarray(mesh.arrays["areaCell"], dtype=float)
horizontal = ConservativeMap(
    diags(area).tocsr(),
    area,
    area,
    {"method": "identity horizontal transfer; fixed mesh"},
)
driver_path = Path(native.__file__)
driver_text = driver_path.read_text()
tree = ast.parse(driver_text)
blocks = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_advance_dynamics_subcycle":
        for index, statement in enumerate(node.body):
            if isinstance(statement, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "euler_rw"
                for t in statement.targets
            ):
                following = node.body[index + 1]
                if not isinstance(following, ast.For):
                    raise ValueError("native vertical Euler block shape changed")
                blocks = [statement, following]
if len(blocks) != 2:
    raise ValueError("native vertical Euler source was not uniquely located")
force_ast = ast.Module(body=blocks, type_ignores=[])
force_code = compile(force_ast, str(driver_path), "exec")
block_text = "\n".join(ast.get_source_segment(driver_text, node) for node in blocks)


def fixture(terrain):
    vertical = build_vertical_grid(
        mesh,
        np.asarray(terrain, dtype=np.float32),
        n_vert_levels=8,
        ztop=30000.0,
        terrain_smoothing_passes=0,
        smooth_surfaces=False,
        xnutr=0.0,
    )
    z = (vertical.zgrid[:-1].astype(float) + vertical.zgrid[1:]) * 0.5
    p = 100000 * np.exp(-9.80616 * z / (287 * 288))
    density = p / (287 * 288)
    theta = 288 / (p / 100000) ** (287 / 1004.5)
    rho = (density / vertical.zz).astype(np.float32)
    heat = (density * theta / vertical.zz).astype(np.float32)
    state = PrognosticState(
        rho,
        heat,
        np.zeros((8, ne), np.float32),
        np.zeros((9, nc), np.float32),
        np.zeros((1, 8, nc), np.float32),
        0.0,
    )
    pressure, exner, _ = native._pressure_exner(
        state, vertical, rgas=287.0, cp=1004.5, reference_pressure=100000.0
    )
    reference = native.DryReferenceState(
        rho.copy(), heat.copy(), pressure.copy(), exner.copy()
    )
    terrain_metrics = native.TerrainMetrics(
        np.zeros((9, nc, mesh.arrays["edgesOnCell"].shape[1]), np.float32),
        np.zeros((9, nc, mesh.arrays["edgesOnCell"].shape[1]), np.float32),
    )
    return state, SimpleNamespace(
        mesh=mesh,
        vertical=vertical,
        reference=reference,
        terrain_metrics=terrain_metrics,
    )


def force_arrays(state, prepared, *, moist=False):
    pp, _, _ = native._pressure_perturbation(
        state,
        prepared.reference,
        prepared.vertical,
        rgas=287.0,
        cp=1004.5,
        reference_pressure=100000.0,
    )
    qtot, cqw, cqu = (
        compute_moist_coefficients(
            state.scalars, moist_indices=(0,), cells_on_edge_remapped=edges, nlev=8
        )
        if moist
        else (None, None, np.ones((8, ne), np.float32))
    )
    perturbation = state.rho - prepared.reference.rho_base
    if moist:
        dpdz = -state.rho.dtype.type(9.80616) * (
            prepared.reference.rho_base * qtot
            + perturbation * (state.rho.dtype.type(1.0) + qtot)
        )
    else:
        dpdz = -state.rho.dtype.type(9.80616) * perturbation
    horizontal_force = pressure_gradient_euler_tendency(
        mesh, pp, dpdz, cqu, prepared.vertical.zz, prepared.vertical.zxu
    )
    namespace = dict(
        np=np,
        self=SimpleNamespace(nlev=8, vertical_grid=prepared.vertical),
        saved=state,
        pressure_perturbation_saved=pp,
        dpdz_saved=dpdz,
        moist_cqw=cqw,
    )
    exec(force_code, namespace)
    return namespace["euler_rw"], horizontal_force, pp


def forces(state, prepared, *, moist=False):
    vertical_force, horizontal_force, pp = force_arrays(state, prepared, moist=moist)
    return dict(
        vertical_euler_abs_max=float(abs(vertical_force).max()),
        horizontal_euler_abs_max=float(abs(horizontal_force).max()),
        pressure_perturbation_abs_max_pa=float(abs(pp).max()),
    )
