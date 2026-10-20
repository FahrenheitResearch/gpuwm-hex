import os, json
from pathlib import Path
import numpy as np
from force_fixture import *

source, old = fixture(np.zeros(nc))


# Complete the saved-state carrier with the native pressure split.
def payload_of(s, prep):
    pp, ex, theta = native._pressure_perturbation(
        s, prep.reference, prep.vertical, rgas=287.0, cp=1004.5, reference_pressure=1e5
    )
    return dict(
        state=dict(
            rho=s.rho,
            rho_theta=s.rho_theta,
            rho_u=s.rho_u,
            rho_w=s.rho_w,
            scalars=s.scalars,
            normal_velocity=np.zeros_like(s.rho_u),
            vertical_velocity=np.zeros_like(s.rho_w),
            pressure_perturbation=pp,
            density_perturbation=s.rho - prep.reference.rho_base,
            rho_theta_perturbation=s.rho_theta - prep.reference.rho_theta_base,
            theta_m=theta,
            exner=ex,
        ),
        vertical=prep.vertical,
        time_seconds=0.0,
        scalar_names=["qv"],
    )


pattern = np.sin(mesh.arrays["latCell"]) * np.cos(mesh.arrays["lonCell"])
pattern -= np.dot(pattern, area) / area.sum()
pattern /= abs(pattern).max()
angle = np.arccos(
    np.clip(
        xyz @ xyz[0] / np.linalg.norm(xyz, axis=1) / np.linalg.norm(xyz[0]), -1.0, 1.0
    )
)
localized = np.where(angle < 0.5, 200.0 * np.exp(-0.5 * (angle / 0.25) ** 2), 0.0)
rows = []
for name, terrain in [
    ("identity", np.zeros(nc)),
    ("uniform20", np.full(nc, 20.0)),
    ("zero_mean20", 20 * pattern),
    ("localized200", localized),
]:
    _, prep = fixture(terrain)
    for mode in ["global", "local"]:
        for balanced in ["0", "1"]:
            os.environ["GPUWM_HEX_MOVING_BALANCE"] = balanced
            w = HeightConservativeMap(
                horizontal,
                old.vertical.zgrid,
                prep.vertical.zgrid,
                old.vertical.dzw,
                prep.vertical.dzw,
                endpoint_alignment=False,
                domain_reconciliation=mode,
            )
            state, saved, budget = remap_atmosphere(
                payload_of(source, old),
                w,
                source_mesh=mesh,
                target_prepared=prep,
                config=None,
            )
            row = dict(
                case=name,
                reconciliation=mode,
                balanced=balanced == "1",
                forces=forces(state, prep),
                changed_far_cells=int(
                    np.any(
                        state.rho[:, terrain == 0] != source.rho[:, terrain == 0],
                        axis=0,
                    ).sum()
                ),
                mass_error=abs(
                    budget["mass_total_kg"]["absolute_error"]
                    / budget["mass_total_kg"]["before"]
                ),
                heat_error=abs(
                    budget["theta_mass_total_kg_kelvin"]["absolute_error"]
                    / budget["theta_mass_total_kg_kelvin"]["before"]
                ),
                balance=budget["balance"],
            )
            rows.append(row)
            print(row, flush=True)
Path("evidence/cpu-force.json").write_text(json.dumps(rows, indent=2) + "\n")
