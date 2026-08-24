#!/usr/bin/env python3
"""Execute the source-exact MPAS-A v8.4.1 compiled endpoint comparator.

This tool deliberately reports measurements without certifying them.  A
separate ruler may claim correctness only after every field has a declared
defect-sized budget and direct mutants prove that budget can catch the defect.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_source_capsule() -> dict[str, str]:
    source_root = (ROOT / "src" / "mpas_port").resolve(strict=True)
    result: dict[str, str] = {}
    for path in sorted(source_root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"invalid production source entry {path}")
        result[path.relative_to(source_root).as_posix()] = _bootstrap_sha256(path)
    if not result:
        raise RuntimeError("production source capsule is empty")
    return result


# This snapshot is deliberately taken before importing NumPy, NetCDF, or any
# mpas_port production module.  The in-gate snapshots must match it exactly.
BOOTSTRAP_SOURCE_CAPSULE_SHA256 = _bootstrap_source_capsule()
BOOTSTRAP_COMPARATOR_SHA256 = _bootstrap_sha256(Path(__file__).resolve())
BOOTSTRAP_PREEXISTING_MPAS_MODULES = tuple(
    sorted(
        name
        for name in sys.modules
        if name == "mpas_port" or name.startswith("mpas_port.")
    )
)

from netCDF4 import Dataset  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402


SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mpas_port.config_v841 import V841DryDycoreConfig  # noqa: E402
from mpas_port.driver import (  # noqa: E402
    DryDycoreDriver,
    SPLIT_FLUX_REDUCTION,
    V841_IMPLEMENTATION_EVIDENCE,
    V841_SOURCE,
    load_mpas_initial_state,
    load_mpas_vertical_grid,
)
from mpas_port.dynamics_v841 import (  # noqa: E402
    load_v841_reference_wind_profiles,
)
from mpas_port.errors import EvidenceError  # noqa: E402
from mpas_port.integration import (  # noqa: E402
    accumulate_split_flux,
    finish_split_flux,
)
from mpas_port.mesh import Mesh  # noqa: E402


DEFAULT_FIXTURE = (
    ROOT / "oracle" / "jw-x1.2562-v8.4.1-split3-endpoint-nonclaim"
)
FIXTURE_PINS = {
    "manifest.json": "500a33646c4a164564e5dc0d50f833eda3dc959a1abe91ae991fe59ebb6b23e6",
    "SHA256SUMS": "4252835d99c06df978726bcb0ad385d28756908ece8407b2347b86c932f7f093",
    "package_and_verify.py": "09b0962d00a9087d38c36ae722bb30928d8386e2514fce0891ab2b0bac595264",
    "reference-input.nc": "45c6879f794af984de791ca7da654a7da5d515dbdb6a131ea778f4edcf597970",
    "endpoint-t0.nc": "53daf8bb17f28724db41d88c1d007d1d2f845a9f5e340662ddbd01d68f7f60b3",
    "endpoint-t1.nc": "708869fe79041b301f584baa8effab57632ca75aababc8bf3b5201c832d31967",
}

MUTANT_TEST_FILES = (
    "tests/test_driver_split3.py",
    "tests/test_integration.py",
    "tests/test_dynamics.py",
    "tests/test_config_v841.py",
    "tests/test_acoustic_v841.py",
    "tests/test_offcentering_v841.py",
    "tests/test_damping_v841.py",
    "tests/test_dynamics_v841.py",
    "tests/test_driver_v841.py",
    "tests/test_diagnostics.py",
    "tests/test_transport.py",
    "tests/test_v841_compiled_endpoint_fixture.py",
)

EXPECTED_SAVED_FIELDS = (
    "theta_m",
    "exner",
    "density_perturbation",
    "rho_theta_perturbation",
    "pressure_perturbation",
    "normal_velocity",
    "vertical_velocity",
)
EXPECTED_RULER_FIELDS = frozenset(
    {
        "state.rho",
        "state.rho_theta",
        "state.rho_u",
        "state.rho_w",
        "state.qv",
        *(f"saved.{name}" for name in EXPECTED_SAVED_FIELDS),
        "scratch2.ru_save",
        "scratch2.rw_save",
        "scratch2.rho_p_save",
        "scratch2.rtheta_p_save",
        "flux.rho_zz_old_split",
        "flux.ruAvg_split",
        "flux.wwAvg_split",
        "flux.ruAvg",
        "flux.wwAvg",
    }
)
if len(EXPECTED_RULER_FIELDS) != 21:
    raise RuntimeError("v8.4.1 ruler inventory declaration must contain 21 fields")

# (basis key, receipt label, declared operator count, rationale, exact)
RULER_POLICY_SPECS: dict[str, tuple[str, str, int, str, bool]] = {
    "state.rho": (
        "state.rho", "state.rho", 9,
        "one float32 parent-scale rounding allowance per executed dynamics RK stage",
        False,
    ),
    "state.rho_theta": (
        "state.rho_theta", "state.rho_theta", 9,
        "one float32 parent-scale rounding allowance per executed dynamics RK stage",
        False,
    ),
    "state.rho_u": (
        "state.rho_u", "state.rho_u", 9,
        "nine executed dynamics RK stages; a tenth parent-momentum ULP is a defect",
        False,
    ),
    "state.rho_w": (
        "state.rho_u", "state.rho_u (rho_w parent momentum scale)", 9,
        "vertical momentum shares the nine-stage momentum schedule; its near-zero own scale is cancellation dominated",
        False,
    ),
    "state.qv": (
        "state.qv", "state.qv", 0,
        "the compiled fixture tracer is identically zero and must remain bitwise exact",
        True,
    ),
    "saved.theta_m": (
        "saved.theta_m", "saved.theta_m", 9,
        "recovered theta follows the nine-stage trajectory", False,
    ),
    "saved.exner": (
        "saved.exner", "saved.exner", 9,
        "recovered Exner follows the nine-stage thermodynamic trajectory", False,
    ),
    "saved.density_perturbation": (
        "state.rho", "state.rho (density perturbation parent)", 9,
        "rho_p is a cancellation-prone difference from the base density", False,
    ),
    "saved.rho_theta_perturbation": (
        "state.rho_theta", "state.rho_theta (rtheta_p parent)", 9,
        "rtheta_p is a cancellation-prone difference from the base state", False,
    ),
    "saved.pressure_perturbation": (
        "native.pressure_base", "pressure_base (pressure_p parent)", 9,
        "pressure_p subtracts the base pressure after the nine-stage thermodynamic chain",
        False,
    ),
    "saved.normal_velocity": (
        "saved.normal_velocity", "saved.normal_velocity", 9,
        "velocity recovery follows the nine-stage momentum trajectory", False,
    ),
    "saved.vertical_velocity": (
        "saved.normal_velocity", "saved.normal_velocity (w parent velocity scale)", 9,
        "recovered w divides cancellation-prone vertical momentum and uses the companion velocity scale",
        False,
    ),
    "scratch2.ru_save": (
        "scratch2.ru_save", "scratch2.ru_save", 6,
        "the subcycle-two checkpoint follows exactly six dynamics RK stages", False,
    ),
    "scratch2.rw_save": (
        "scratch2.ru_save", "scratch2.ru_save (rw parent momentum scale)", 6,
        "the vertical checkpoint shares the first six momentum stages", False,
    ),
    "scratch2.rho_p_save": (
        "state.rho", "state.rho (rho_p_save parent)", 6,
        "the density-perturbation checkpoint follows six stages", False,
    ),
    "scratch2.rtheta_p_save": (
        "state.rho_theta", "state.rho_theta (rtheta_p_save parent)", 6,
        "the rtheta-perturbation checkpoint follows six stages", False,
    ),
    "flux.rho_zz_old_split": (
        "flux.rho_zz_old_split", "flux.rho_zz_old_split", 0,
        "the outer t0 density is copied and must remain bitwise exact", True,
    ),
    "flux.ruAvg_split": (
        "flux.ruAvg_split", "flux.ruAvg_split", 11,
        "nine RK stages plus two ordered split-flux additions", False,
    ),
    "flux.wwAvg_split": (
        "flux.ruAvg_split", "flux.ruAvg_split (ww sum parent momentum scale)", 11,
        "vertical flux sum shares nine RK stages and two ordered additions", False,
    ),
    "flux.ruAvg": (
        "flux.ruAvg", "flux.ruAvg", 10,
        "nine RK stages plus the typed reciprocal finalization", False,
    ),
    "flux.wwAvg": (
        "flux.ruAvg", "flux.ruAvg (ww average parent momentum scale)", 10,
        "vertical flux average shares nine stages and typed reciprocal finalization", False,
    ),
}
if frozenset(RULER_POLICY_SPECS) != EXPECTED_RULER_FIELDS:
    raise RuntimeError("v8.4.1 ruler policy does not cover its exact inventory")

EXPECTED_CONFIG: dict[str, Any] = {
    "config_dt": 360.0,
    "config_time_integration_order": 3,
    "config_number_of_sub_steps": 6,
    "config_dynamics_split_steps": 3,
    "config_apply_lbcs": False,
    "config_split_dynamics_transport": True,
    "config_scalar_advection": True,
    "config_monotonic": True,
    "config_positive_definite": False,
    "config_scalar_adv_order": 3,
    "config_scalar_vadv_order": 3,
    "config_coef_3rd_order": 0.25,
    "config_apvm_upwinding": 0.5,
    "config_epssm": 0.0,
    "config_moist_physics": False,
    "config_physics_suite": "none",
    "config_iau_option": "off",
    "config_divergence_damping": False,
    "config_horiz_mixing": "off",
    "config_len_disp": 0.0,
    "config_visc4_2dsmag": 0.0,
    "config_smagorinsky_coef": 0.0,
    "config_del4u_div_factor": 10.0,
    "config_h_ScaleWithMesh": True,
    "config_mpas_cam_coef": 0.0,
    "config_h_theta_eddy_visc2": 0.0,
    "config_v_theta_eddy_visc2": 0.0,
    "config_h_mom_eddy_visc2": 0.0,
    "config_v_mom_eddy_visc2": 0.0,
    "config_h_theta_eddy_visc4": 0.0,
    "config_h_mom_eddy_visc4": 0.0,
    "config_smdiv": 0.0,
    "config_xnutr": 0.0,
    "config_zd": 22_000.0,
    "config_vertical_mixing": False,
    "config_rayleigh_damp_u": False,
    "config_curvature_terms": False,
    "config_terrain_following": None,
    "source_release": "v8.4.1",
    "config_epssm_minimum": 0.1,
    "config_epssm_maximum": 0.5,
    "config_epssm_transition_bottom_z": 30_000.0,
    "config_epssm_transition_top_z": 50_000.0,
    "config_gpu_aware_mpi": False,
    "config_les_model": "none",
    "config_les_surface": "none",
    "config_mix_scalars": False,
    "config_surface_heat_flux": 0.0,
    "config_surface_moisture_flux": 0.0,
    "config_surface_drag_coefficient": 0.0,
}


class TracingV841Driver(DryDycoreDriver):
    """Comparison-only capture of native dynamics-subcycle endpoints."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.subcycle_trace: list[Any] = []

    def _advance_dynamics_subcycle(self, *args: Any, **kwargs: Any) -> Any:
        result = super()._advance_dynamics_subcycle(*args, **kwargs)
        self.subcycle_trace.append(result)
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def hash_regular_tree(root: Path, *, suffix: str | None = None) -> dict[str, str]:
    """Hash a complete regular-file tree with an exact relative inventory."""

    base = root.resolve(strict=True)
    result: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise EvidenceError(f"evidence tree contains symlink {path}")
        if not path.is_file() or (suffix is not None and path.suffix != suffix):
            continue
        relative = path.relative_to(base).as_posix()
        result[relative] = sha256_file(path)
    if not result:
        raise EvidenceError(f"evidence tree is empty: {base}")
    return result


def capture_integrity(fixture: Path) -> dict[str, Any]:
    source_root = ROOT / "src" / "mpas_port"
    test_hashes = {
        relative: sha256_file(ROOT / relative) for relative in MUTANT_TEST_FILES
    }
    return {
        "source_capsule_sha256": hash_regular_tree(source_root, suffix=".py"),
        "fixture_tree_sha256": hash_regular_tree(fixture),
        "comparator_sha256": sha256_file(Path(__file__).resolve()),
        "mutant_test_sha256": test_hashes,
    }


def assert_integrity_unchanged(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    for name in (
        "source_capsule_sha256",
        "fixture_tree_sha256",
        "comparator_sha256",
        "mutant_test_sha256",
    ):
        if before[name] != after[name]:
            raise EvidenceError(f"{name} changed during the compiled endpoint gate")


def assert_preimport_bootstrap(integrity: dict[str, Any]) -> None:
    if BOOTSTRAP_PREEXISTING_MPAS_MODULES:
        raise EvidenceError(
            "compiled endpoint gate must start in a fresh process before mpas_port imports"
        )
    if integrity["source_capsule_sha256"] != BOOTSTRAP_SOURCE_CAPSULE_SHA256:
        raise EvidenceError(
            "production source changed between pre-import bootstrap and gate start"
        )
    if integrity["comparator_sha256"] != BOOTSTRAP_COMPARATOR_SHA256:
        raise EvidenceError(
            "comparator changed between pre-import bootstrap and gate start"
        )


def _run_checked(
    command: list[str],
    *,
    label: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    elapsed = perf_counter() - started
    receipt = {
        "label": label,
        "argv": command,
        "cwd": str(ROOT.resolve()),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_seconds": elapsed,
        "passed": completed.returncode == 0,
    }
    if completed.returncode != 0:
        raise EvidenceError(
            f"{label} failed with exit {completed.returncode}: "
            f"{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def run_packaged_verifier(fixture: Path) -> dict[str, Any]:
    resolved = fixture.resolve(strict=True)
    verify_fixture(resolved)
    return _run_checked(
        [
            sys.executable,
            str(resolved / "package_and_verify.py"),
            "--repo-root",
            str(ROOT.resolve()),
            "--output",
            str(resolved),
            "--verify-only",
        ],
        label="pinned v8.4.1 fixture package verifier",
        timeout_seconds=300,
    )


def run_required_mutant_suite() -> dict[str, Any]:
    return _run_checked(
        [sys.executable, "-m", "pytest", "-q", *MUTANT_TEST_FILES],
        label="v8.4.1 endpoint semantic mutant suite",
        timeout_seconds=900,
    )


def canonical_payload_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.kind in "iufc" and array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def verify_fixture(fixture: Path) -> dict[str, Any]:
    fixture = fixture.resolve(strict=True)
    for name, expected in FIXTURE_PINS.items():
        path = fixture / name
        if not path.is_file():
            raise EvidenceError(f"missing v8.4.1 authority fixture file {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise EvidenceError(
                f"v8.4.1 authority {name} SHA mismatch: {actual} != {expected}"
            )
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "mpas-port.v841-compiled-endpoint-fixture.v1":
        raise EvidenceError("v8.4.1 authority fixture schema is not pinned v1")
    authority = manifest.get("authority", {})
    if authority.get("release") != "v8.4.1":
        raise EvidenceError("compiled endpoint does not declare release v8.4.1")
    if authority.get("annotated_tag_object") != (
        "2a934b5008a7446df96d550bf2e21466feaec686"
    ):
        raise EvidenceError("compiled endpoint annotated tag object changed")
    if authority.get("peeled_commit") != (
        "91c5eac175eebeaf4206bacd5cb50c39dff3c152"
    ):
        raise EvidenceError("compiled endpoint peeled source commit changed")
    if authority.get("archive", {}).get("sha256") != (
        "772f565c2bd66999492085eff8ffa0b9aa9a2edd1e7f2c0e5d1a8bedc1160861"
    ):
        raise EvidenceError("compiled endpoint official archive SHA changed")
    if manifest.get("evidence", {}).get("authority_claim") is not False:
        raise EvidenceError("work-only endpoint fixture must not self-certify")
    mesh_contract = manifest.get("mesh", {}).get("relationship", {})
    if mesh_contract.get("authority_comparison_mesh_source") != "reference-input.nc":
        raise EvidenceError("authority comparator must use the executed input mesh")
    if not mesh_contract.get("required_executed_input_variables_present_and_hashed"):
        raise EvidenceError("executed mesh input inventory is incomplete")
    return manifest


def assert_mesh_payload(
    mesh: Mesh,
    manifest: dict[str, Any],
    name: str,
) -> None:
    declaration = manifest["mesh"]["relationship"]["required_variables"][name]
    expected = declaration["executed_input_payload_sha256"]
    actual = canonical_payload_sha256(np.asarray(mesh.arrays[name]))
    if actual != expected:
        raise EvidenceError(
            f"authority mesh field {name} payload mismatch: {actual} != {expected}"
        )


def assert_input_mesh_payload(
    path: Path,
    manifest: dict[str, Any],
    name: str,
) -> None:
    with Dataset(path, mode="r") as dataset:
        dataset.set_auto_mask(False)
        if name not in dataset.variables:
            raise EvidenceError(f"executed input mesh field {name} is missing")
        value = np.asarray(dataset.variables[name][...])
    declaration = manifest["mesh"]["relationship"]["required_variables"][name]
    expected = declaration["executed_input_payload_sha256"]
    actual = canonical_payload_sha256(value)
    if actual != expected:
        raise EvidenceError(
            f"executed input mesh field {name} payload mismatch: "
            f"{actual} != {expected}"
        )


def _native_field(path: Path, name: str) -> np.ndarray:
    with Dataset(path, mode="r") as dataset:
        dataset.set_auto_mask(False)
        value = np.asarray(dataset.variables[name][...])
        dimensions = tuple(dataset.variables[name].dimensions)
    if dimensions and dimensions[0] == "Time":
        value = value[0]
        dimensions = dimensions[1:]
    if len(dimensions) == 2 and dimensions[0] in ("nCells", "nEdges"):
        value = value.T
    return np.ascontiguousarray(value)


def arrays_bitwise_equal(actual: Any, expected: Any) -> bool:
    left = np.asarray(actual)
    right = np.asarray(expected)
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _config() -> V841DryDycoreConfig:
    return V841DryDycoreConfig(
        config_dt=360.0,
        config_time_integration_order=3,
        config_number_of_sub_steps=6,
        config_split_dynamics_transport=True,
        config_dynamics_split_steps=3,
        config_scalar_advection=True,
        config_monotonic=True,
        config_positive_definite=False,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        config_coef_3rd_order=0.25,
        config_apvm_upwinding=0.5,
        config_horiz_mixing="off",
        config_smdiv=0.0,
        config_xnutr=0.0,
        config_zd=22_000.0,
    )


def load_case(fixture: Path) -> dict[str, Any]:
    manifest = verify_fixture(fixture)
    fixture = fixture.resolve(strict=True)
    reference_input = fixture / "reference-input.nc"
    t0 = fixture / "endpoint-t0.nc"
    t1 = fixture / "endpoint-t1.nc"
    for name in manifest["mesh"]["relationship"]["required_variables"]:
        assert_input_mesh_payload(reference_input, manifest, name)
    mesh = Mesh.from_netcdf(reference_input)
    native = load_mpas_vertical_grid(
        reference_input,
        mesh,
        config_coef_3rd_order=0.25,
    )
    state, reference, saved = load_mpas_initial_state(
        t0,
        mesh,
        native.vertical_grid,
        scalar_names=("qv",),
        reference_path=t0,
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
    )
    target, _, target_saved = load_mpas_initial_state(
        t1,
        mesh,
        native.vertical_grid,
        scalar_names=("qv",),
        reference_path=t1,
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
    )
    profiles = load_v841_reference_wind_profiles(
        reference_input,
        n_vert_levels=native.vertical_grid.n_vert_levels,
    )
    return {
        "manifest": manifest,
        "mesh": mesh,
        "native": native,
        "state": state,
        "reference": reference,
        "saved": saved,
        "target": target,
        "target_saved": target_saved,
        "profiles": profiles,
        "t0": t0,
        "t1": t1,
    }


def assert_endpoint_mapping_is_bitwise(
    bundle: dict[str, Any],
    endpoint: str,
) -> None:
    if endpoint == "t0":
        path = bundle["t0"]
        state = bundle["state"]
        saved = bundle["saved"]
    elif endpoint == "t1":
        path = bundle["t1"]
        state = bundle["target"]
        saved = bundle["target_saved"]
    else:
        raise ValueError("endpoint must be 't0' or 't1'")
    expected_state = {
        "rho": _native_field(path, "rho_zz"),
        "rho_theta": _native_field(path, "rtheta_base")
        + _native_field(path, "rtheta_p"),
        "rho_u": _native_field(path, "ru"),
        "rho_w": _native_field(path, "rw"),
        "scalars": _native_field(path, "qv")[None, ...],
    }
    expected_saved = {
        "theta_m": _native_field(path, "theta_m"),
        "exner": _native_field(path, "exner"),
        "density_perturbation": _native_field(path, "rho_p"),
        "rho_theta_perturbation": _native_field(path, "rtheta_p"),
        "pressure_perturbation": _native_field(path, "pressure_p"),
        "normal_velocity": _native_field(path, "u"),
        "vertical_velocity": _native_field(path, "w"),
    }
    for name, expected in expected_state.items():
        if not arrays_bitwise_equal(getattr(state, name), expected):
            raise EvidenceError(f"{endpoint} state mapping changed bits for {name}")
    if tuple(saved.__slots__) != EXPECTED_SAVED_FIELDS:
        raise EvidenceError(
            f"{endpoint} saved-diagnostic inventory changed: {saved.__slots__}"
        )
    for name, expected in expected_saved.items():
        if not arrays_bitwise_equal(getattr(saved, name), expected):
            raise EvidenceError(
                f"{endpoint} diagnostics mapping changed bits for {name}"
            )


def assert_t0_mapping_is_bitwise(bundle: dict[str, Any]) -> None:
    assert_endpoint_mapping_is_bitwise(bundle, "t0")


def assert_t1_mapping_is_bitwise(bundle: dict[str, Any]) -> None:
    assert_endpoint_mapping_is_bitwise(bundle, "t1")


def _field_report(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise EvidenceError(
            f"endpoint shape/dtype mismatch: {actual.shape}/{actual.dtype} "
            f"!= {expected.shape}/{expected.dtype}"
        )
    if actual.dtype != np.dtype(np.float32):
        raise EvidenceError(
            f"v8.4.1 endpoint ruler requires float32 RKIND, got {actual.dtype}"
        )
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise EvidenceError("endpoint ruler received a non-finite field")
    gap = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    flat_index = int(np.argmax(gap))
    argmax = tuple(int(item) for item in np.unravel_index(flat_index, gap.shape))
    bitwise = actual.view(np.uint32) == expected.view(np.uint32)
    per_level = (
        np.max(gap.reshape((gap.shape[0], -1)), axis=1).tolist()
        if gap.ndim >= 2
        else None
    )
    return {
        "max_abs": float(gap.flat[flat_index]),
        "mean_abs": float(np.mean(gap)),
        "argmax": list(argmax),
        "actual_at_argmax": float(actual[argmax]),
        "expected_at_argmax": float(expected[argmax]),
        "different_bits": int(np.count_nonzero(~bitwise)),
        "per_level_max_abs": per_level,
    }


def declared_ulp_budget(
    basis: np.ndarray,
    *,
    basis_name: str,
    operator_count: int,
    rationale: str,
    exact: bool = False,
) -> dict[str, Any]:
    """Declare an authority-only absolute ruler from a reference field scale."""

    reference = np.asarray(basis)
    if reference.dtype != np.dtype(np.float32):
        raise EvidenceError(
            f"budget basis {basis_name} must be float32, got {reference.dtype}"
        )
    if not np.all(np.isfinite(reference)):
        raise EvidenceError(f"budget basis {basis_name} is non-finite")
    if type(operator_count) is not int or operator_count < 0:
        raise EvidenceError("operator_count must be a non-negative exact integer")
    maximum = np.max(np.abs(reference), initial=np.float32(0.0))
    spacing = float(np.spacing(np.float32(maximum)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise EvidenceError(f"budget basis {basis_name} has invalid spacing {spacing}")
    budget = 0.0 if exact else float(operator_count * spacing)
    next_ulp_defect = spacing if exact else float((operator_count + 1) * spacing)
    return {
        "basis_field": basis_name,
        "basis_reference_max_abs": float(maximum),
        "basis_float32_spacing": spacing,
        "declared_operator_count": operator_count,
        "absolute_budget": budget,
        "rejects_max_abs_above": budget,
        "next_basis_ulp_defect": next_ulp_defect,
        "rationale": rationale,
        "exact": exact,
    }


def evaluate_declared_budget(
    actual: np.ndarray,
    expected: np.ndarray,
    budget: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one field without deriving any ceiling from its measured gap."""

    measurement = _field_report(np.asarray(actual), np.asarray(expected))
    ceiling = float(budget["absolute_budget"])
    passed = measurement["different_bits"] == 0 if budget["exact"] else (
        measurement["max_abs"] <= ceiling
    )
    return {"passed": bool(passed), "budget": budget, "measurement": measurement}


def assert_ruler_inventory(
    policies: dict[str, Any],
    actual_fields: dict[str, np.ndarray],
    expected_fields: dict[str, np.ndarray],
) -> None:
    inventories = {
        "declared": frozenset(EXPECTED_RULER_FIELDS),
        "policy": frozenset(policies),
        "actual": frozenset(actual_fields),
        "expected": frozenset(expected_fields),
    }
    for name, inventory in inventories.items():
        if inventory != EXPECTED_RULER_FIELDS:
            missing = sorted(EXPECTED_RULER_FIELDS - inventory)
            extra = sorted(inventory - EXPECTED_RULER_FIELDS)
            raise EvidenceError(
                f"ruler {name} inventory mismatch: missing={missing}, extra={extra}"
            )
    if any(len(inventory) != 21 for inventory in inventories.values()):
        raise EvidenceError("ruler must evaluate exactly 21 named fields")


def resolve_ruler_policies(
    expected_fields: dict[str, np.ndarray],
    pressure_base: np.ndarray,
) -> dict[str, tuple[np.ndarray, str, int, str, bool]]:
    bases = {**expected_fields, "native.pressure_base": pressure_base}
    result: dict[str, tuple[np.ndarray, str, int, str, bool]] = {}
    for name, (basis_key, basis_name, count, rationale, exact) in (
        RULER_POLICY_SPECS.items()
    ):
        if basis_key not in bases:
            raise EvidenceError(f"ruler basis {basis_key!r} is absent")
        result[name] = (
            np.asarray(bases[basis_key]),
            basis_name,
            count,
            rationale,
            exact,
        )
    return result


def _assert_exact_value(name: str, actual: Any, expected: Any) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise EvidenceError(
            f"execution contract {name} changed: {actual!r} != {expected!r}"
        )


def assert_execution_contract(
    config: V841DryDycoreConfig,
    receipt: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    config.validate()
    actual_config = asdict(config)
    if set(actual_config) != set(EXPECTED_CONFIG):
        raise EvidenceError("v8.4.1 execution config inventory changed")
    for name, expected in EXPECTED_CONFIG.items():
        _assert_exact_value(f"config.{name}", actual_config[name], expected)

    authority_run = manifest.get("run", {})
    authority_contract = {
        "dt_seconds": 360.0,
        "dynamics_split_steps": 3,
        "horizontal_mixing": "off",
        "number_of_acoustic_substeps": 6,
        "split_dynamics_transport": True,
    }
    for name, expected in authority_contract.items():
        _assert_exact_value(
            f"manifest.run.{name}", authority_run.get(name), expected
        )
    authority_epssm_profile = {
        "minimum": 0.1,
        "maximum": 0.5,
        "transition_bottom_z": 30_000.0,
        "transition_top_z": 50_000.0,
    }
    _assert_exact_value(
        "manifest.run.config_epssm", authority_run.get("config_epssm"), 0.0
    )
    manifest_epssm_profile = authority_run.get("config_epssm_profile")
    if not isinstance(manifest_epssm_profile, dict) or set(
        manifest_epssm_profile
    ) != set(authority_epssm_profile):
        raise EvidenceError("execution contract manifest EPS profile changed")
    for name, expected in authority_epssm_profile.items():
        _assert_exact_value(
            f"manifest.run.config_epssm_profile.{name}",
            manifest_epssm_profile[name],
            expected,
        )
    epssm_port_bindings = {
        "minimum": config.config_epssm_minimum,
        "maximum": config.config_epssm_maximum,
        "transition_bottom_z": config.config_epssm_transition_bottom_z,
        "transition_top_z": config.config_epssm_transition_top_z,
    }
    _assert_exact_value(
        "port-authority.config_epssm",
        config.config_epssm,
        authority_run["config_epssm"],
    )
    for name, expected in authority_epssm_profile.items():
        _assert_exact_value(
            f"port-authority.config_epssm_profile.{name}",
            epssm_port_bindings[name],
            expected,
        )

    receipt_dict = receipt.as_dict()
    expected_receipt_members = {
        "evidence",
        "frozen_source",
        "source_release",
        "start_time_seconds",
        "end_time_seconds",
        "stage_acoustic_steps",
        "dynamics_split_steps",
        "dynamics_timestep_seconds",
        "dynamics_stage_timesteps",
        "scalar_transport_stage_timesteps",
        "split_flux_reduction",
        "before",
        "after",
        "mass_relative_drift",
        "energy_relative_drift",
    }
    if set(receipt_dict) != expected_receipt_members:
        raise EvidenceError("StepReceipt member inventory changed")
    receipt_contract = {
        "evidence": V841_IMPLEMENTATION_EVIDENCE,
        "frozen_source": V841_SOURCE,
        "source_release": "v8.4.1",
        "start_time_seconds": 0.0,
        "end_time_seconds": 360.0,
        "stage_acoustic_steps": (1, 3, 6),
        "dynamics_split_steps": 3,
        "dynamics_timestep_seconds": 120.0,
        "dynamics_stage_timesteps": (40.0, 60.0, 120.0),
        "scalar_transport_stage_timesteps": (120.0, 180.0, 360.0),
        "split_flux_reduction": SPLIT_FLUX_REDUCTION,
    }
    for name, expected in receipt_contract.items():
        _assert_exact_value(f"receipt.{name}", getattr(receipt, name), expected)
    for phase in (receipt.before, receipt.after):
        if set(phase.__slots__) != {
            "mass",
            "theta_mass",
            "energy_proxy",
            "min_density",
            "max_density",
            "max_abs_velocity",
            "all_finite",
        }:
            raise EvidenceError("StepReceipt StateMetrics inventory changed")
        if not phase.all_finite:
            raise EvidenceError("StepReceipt state metrics are not finite")
        for name in phase.__slots__:
            value = getattr(phase, name)
            if name != "all_finite" and not np.isfinite(value):
                raise EvidenceError(f"StepReceipt metric {name} is non-finite")
    for name in ("mass_relative_drift", "energy_relative_drift"):
        if not np.isfinite(getattr(receipt, name)):
            raise EvidenceError(f"StepReceipt {name} is non-finite")
    return {
        "passed": True,
        "config": actual_config,
        "authority_run": authority_contract,
        "authority_epssm": {
            "legacy_sentinel": 0.0,
            "profile": authority_epssm_profile,
            "port_bindings_equal": True,
        },
        "receipt": receipt_contract,
    }


def derive_and_assert_vacuities(
    bundle: dict[str, Any],
    result: Any,
    driver: TracingV841Driver,
    config: V841DryDycoreConfig,
) -> dict[str, bool]:
    def positive_zero(value: Any) -> bool:
        array = np.asarray(value)
        return bool(np.all(array == 0) and not np.any(np.signbit(array)))

    namelist = (bundle["t1"].parent / "namelist.atmosphere").read_text(
        encoding="utf-8"
    )
    qv_arrays = (
        np.asarray(bundle["state"].scalars),
        np.asarray(bundle["target"].scalars),
        np.asarray(result.state.scalars),
    )
    vacuities = {
        "qv_identically_zero": all(positive_zero(value) for value in qv_arrays),
        "u_init_identically_zero": positive_zero(
            bundle["profiles"].u_init
        ),
        "v_init_identically_zero": positive_zero(
            bundle["profiles"].v_init
        ),
        "dss_identically_zero": bool(
            positive_zero(bundle["native"].vertical_grid.dss)
            and positive_zero(driver.damping_coefficients)
        ),
        "physics_off": bool(
            config.config_physics_suite == "none"
            and not config.config_moist_physics
            and "config_physics_suite = 'none'" in namelist
        ),
        "horizontal_mixing_off": bool(
            config.config_horiz_mixing == "off"
            and bundle["manifest"].get("run", {}).get("horizontal_mixing") == "off"
            and "config_horiz_mixing = 'off'" in namelist
        ),
    }
    false_names = sorted(name for name, value in vacuities.items() if not value)
    if false_names:
        raise EvidenceError(
            f"compiled fixture vacuity declaration changed: {false_names}"
        )
    return vacuities


def assert_reference_profile_contract(
    bundle: dict[str, Any],
    driver: TracingV841Driver,
) -> dict[str, Any]:
    profiles = bundle["profiles"]
    state = bundle["state"]
    if driver.reference_wind_profiles is not profiles:
        raise EvidenceError("driver did not bind the executed v8.4.1 wind profiles")
    profiles.validate(driver.nlev, np.asarray(state.rho).dtype)
    return {
        "passed": True,
        "u_init_shape": list(np.asarray(profiles.u_init).shape),
        "v_init_shape": list(np.asarray(profiles.v_init).shape),
        "dtype": str(np.asarray(profiles.u_init).dtype),
        "u_init_payload_sha256": canonical_payload_sha256(profiles.u_init),
        "v_init_payload_sha256": canonical_payload_sha256(profiles.v_init),
    }


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def runtime_provenance(
    fixture: Path,
    integrity_before: dict[str, Any],
    integrity_after: dict[str, Any],
    fixture_verifier: dict[str, Any],
    mutant_suite: dict[str, Any],
) -> dict[str, Any]:
    return {
        "command_argv": [sys.executable, *sys.argv],
        "resolved_fixture": str(fixture.resolve()),
        "cwd": str(ROOT.resolve()),
        "port_git_head": _git_head(),
        "official_v841_peeled_commit": (
            "91c5eac175eebeaf4206bacd5cb50c39dff3c152"
        ),
        "integrity_start": integrity_before,
        "integrity_end": integrity_after,
        "integrity_start_end_equal": integrity_before == integrity_after,
        "preimport_bootstrap": {
            "source_capsule_sha256": BOOTSTRAP_SOURCE_CAPSULE_SHA256,
            "comparator_sha256": BOOTSTRAP_COMPARATOR_SHA256,
            "preexisting_mpas_modules": list(BOOTSTRAP_PREEXISTING_MPAS_MODULES),
            "matches_gate_start": bool(
                integrity_before["source_capsule_sha256"]
                == BOOTSTRAP_SOURCE_CAPSULE_SHA256
                and integrity_before["comparator_sha256"]
                == BOOTSTRAP_COMPARATOR_SHA256
            ),
        },
        "fixture_verifier": fixture_verifier,
        "mutant_suite": mutant_suite,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "netcdf4_python": netCDF4.__version__,
    }


def compare_endpoint(fixture: Path) -> dict[str, Any]:
    fixture = fixture.resolve(strict=True)
    integrity_before = capture_integrity(fixture)
    assert_preimport_bootstrap(integrity_before)
    fixture_manifest = verify_fixture(fixture)
    fixture_verifier = run_packaged_verifier(fixture)
    mutant_suite = run_required_mutant_suite()
    bundle = load_case(fixture)
    if bundle["manifest"] != fixture_manifest:
        raise EvidenceError("fixture manifest changed between verifier and load")
    assert_t0_mapping_is_bitwise(bundle)
    assert_t1_mapping_is_bitwise(bundle)
    config = _config()
    driver = TracingV841Driver(
        bundle["mesh"],
        bundle["native"].vertical_grid,
        bundle["reference"],
        config,
        terrain_metrics=bundle["native"].terrain_metrics,
        reference_wind_profiles=bundle["profiles"],
    )
    started = perf_counter()
    result = driver.step(bundle["state"], saved_diagnostics=bundle["saved"])
    elapsed = perf_counter() - started
    state_fields = {
        name: _field_report(actual, expected)
        for name, actual, expected in (
            ("rho", result.state.rho, bundle["target"].rho),
            ("rho_theta", result.state.rho_theta, bundle["target"].rho_theta),
            ("rho_u", result.state.rho_u, bundle["target"].rho_u),
            ("rho_w", result.state.rho_w, bundle["target"].rho_w),
            ("qv", result.state.scalars, bundle["target"].scalars),
        )
    }
    diagnostic_fields = {
        name: _field_report(
            np.asarray(getattr(result.saved_diagnostics, name)),
            np.asarray(getattr(bundle["target_saved"], name)),
        )
        for name in EXPECTED_SAVED_FIELDS
    }
    if len(driver.subcycle_trace) != 3:
        raise EvidenceError(
            f"expected three dynamics subcycles, captured {len(driver.subcycle_trace)}"
        )
    subcycle_two = driver.subcycle_trace[1]
    t1 = bundle["t1"]
    subcycle_two_fields = {
        "ru_save": _field_report(
            np.asarray(subcycle_two.state.rho_u), _native_field(t1, "ru_save")
        ),
        "rw_save": _field_report(
            np.asarray(subcycle_two.state.rho_w), _native_field(t1, "rw_save")
        ),
        "rho_p_save": _field_report(
            np.asarray(subcycle_two.saved_diagnostics.density_perturbation),
            _native_field(t1, "rho_p_save"),
        ),
        "rtheta_p_save": _field_report(
            np.asarray(subcycle_two.saved_diagnostics.rho_theta_perturbation),
            _native_field(t1, "rtheta_p_save"),
        ),
    }
    split_u_sum = None
    split_w_sum = None
    for subcycle in driver.subcycle_trace:
        split_u_sum = accumulate_split_flux(subcycle.mass_flux_u, split_u_sum)
        split_w_sum = accumulate_split_flux(subcycle.mass_flux_w, split_w_sum)
    if split_u_sum is None or split_w_sum is None:
        raise EvidenceError("split-flux trace did not accumulate")
    split_flux_fields = {
        "rho_zz_old_split": _field_report(
            np.asarray(bundle["state"].rho),
            _native_field(t1, "rho_zz_old_split"),
        ),
        "ruAvg_split": _field_report(
            split_u_sum, _native_field(t1, "ruAvg_split")
        ),
        "wwAvg_split": _field_report(
            split_w_sum, _native_field(t1, "wwAvg_split")
        ),
        "ruAvg": _field_report(
            finish_split_flux(split_u_sum, 3), _native_field(t1, "ruAvg")
        ),
        "wwAvg": _field_report(
            finish_split_flux(split_w_sum, 3), _native_field(t1, "wwAvg")
        ),
    }
    final_split_u = finish_split_flux(split_u_sum, 3)
    final_split_w = finish_split_flux(split_w_sum, 3)
    state_actual = {
        "rho": np.asarray(result.state.rho),
        "rho_theta": np.asarray(result.state.rho_theta),
        "rho_u": np.asarray(result.state.rho_u),
        "rho_w": np.asarray(result.state.rho_w),
        "qv": np.asarray(result.state.scalars),
    }
    state_expected = {
        "rho": np.asarray(bundle["target"].rho),
        "rho_theta": np.asarray(bundle["target"].rho_theta),
        "rho_u": np.asarray(bundle["target"].rho_u),
        "rho_w": np.asarray(bundle["target"].rho_w),
        "qv": np.asarray(bundle["target"].scalars),
    }
    saved_actual = {
        name: np.asarray(getattr(result.saved_diagnostics, name))
        for name in result.saved_diagnostics.__slots__
    }
    saved_expected = {
        name: np.asarray(getattr(bundle["target_saved"], name))
        for name in result.saved_diagnostics.__slots__
    }
    scratch_actual = {
        "ru_save": np.asarray(subcycle_two.state.rho_u),
        "rw_save": np.asarray(subcycle_two.state.rho_w),
        "rho_p_save": np.asarray(
            subcycle_two.saved_diagnostics.density_perturbation
        ),
        "rtheta_p_save": np.asarray(
            subcycle_two.saved_diagnostics.rho_theta_perturbation
        ),
    }
    scratch_expected = {
        name: _native_field(t1, name) for name in scratch_actual
    }
    flux_actual = {
        "rho_zz_old_split": np.asarray(bundle["state"].rho),
        "ruAvg_split": split_u_sum,
        "wwAvg_split": split_w_sum,
        "ruAvg": final_split_u,
        "wwAvg": final_split_w,
    }
    flux_expected = {
        name: _native_field(t1, name) for name in flux_actual
    }
    pressure_base = _native_field(t1, "pressure_base")
    actual_fields = {
        **{f"state.{name}": value for name, value in state_actual.items()},
        **{f"saved.{name}": value for name, value in saved_actual.items()},
        **{f"scratch2.{name}": value for name, value in scratch_actual.items()},
        **{f"flux.{name}": value for name, value in flux_actual.items()},
    }
    expected_fields = {
        **{f"state.{name}": value for name, value in state_expected.items()},
        **{f"saved.{name}": value for name, value in saved_expected.items()},
        **{f"scratch2.{name}": value for name, value in scratch_expected.items()},
        **{f"flux.{name}": value for name, value in flux_expected.items()},
    }
    policies = resolve_ruler_policies(expected_fields, pressure_base)
    assert_ruler_inventory(policies, actual_fields, expected_fields)
    ruler_fields: dict[str, Any] = {}
    for name, (basis, basis_name, count, rationale, exact) in policies.items():
        budget = declared_ulp_budget(
            basis,
            basis_name=basis_name,
            operator_count=count,
            rationale=rationale,
            exact=exact,
        )
        ruler_fields[name] = evaluate_declared_budget(
            actual_fields[name], expected_fields[name], budget
        )
    numeric_passed = all(item["passed"] for item in ruler_fields.values())
    execution_contract = assert_execution_contract(
        config, result.receipt, bundle["manifest"]
    )
    execution_contract["reference_wind_profiles"] = (
        assert_reference_profile_contract(bundle, driver)
    )
    vacuities = derive_and_assert_vacuities(bundle, result, driver, config)
    integrity_after = capture_integrity(fixture)
    assert_integrity_unchanged(integrity_before, integrity_after)
    ruler_passed = bool(
        numeric_passed
        and execution_contract["passed"]
        and fixture_verifier["passed"]
        and mutant_suite["passed"]
        and integrity_before == integrity_after
    )
    return {
        "schema": "mpas-port.v841-compiled-endpoint-comparison.v1",
        "certified": False,
        "certification_reason": (
            "implementation-only compiled dry endpoint ruler passed; broad v8.4.1 "
            "certification remains withheld because nonzero tracer, reference-Coriolis, "
            "dss, physics, and mixing branches are unexercised; native nonzero-tracer "
            "authority remains pending"
            if ruler_passed
            else "one or more declared authority-derived field budgets failed"
        ),
        "fixture_pins": FIXTURE_PINS,
        "t0_mapping_bitwise": True,
        "t1_mapping_bitwise": True,
        "elapsed_seconds": elapsed,
        "receipt": result.receipt.as_dict(),
        "state": state_fields,
        "saved_diagnostics": diagnostic_fields,
        "intermediate": {
            "after_dynamics_subcycle_2": subcycle_two_fields,
            "split_flux": split_flux_fields,
        },
        "ruler": {
            "passed": ruler_passed,
            "numeric_fields_passed": numeric_passed,
            "policy": "authority-reference float32 spacing times declared executed operator count",
            "measured_gap_used_to_set_budget": False,
            "field_inventory_count": len(ruler_fields),
            "field_inventory": sorted(ruler_fields),
            "fields": ruler_fields,
            "execution_contract": execution_contract,
            "required_mutant_suite_passed": mutant_suite["passed"],
        },
        "vacuities_and_nonclaims": {
            **vacuities,
            "nonzero_tracer_compiled_certified": False,
            "nonzero_reference_coriolis_compiled_certified": False,
            "nonzero_dss_compiled_certified": False,
            "physics_compiled_certified": False,
            "mixing_compiled_certified": False,
        },
        "runtime_provenance": runtime_provenance(
            fixture,
            integrity_before,
            integrity_after,
            fixture_verifier,
            mutant_suite,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_fixture(args.fixture)
        verifier = run_packaged_verifier(args.fixture)
        report: dict[str, Any] = {
            "verified": True,
            "schema": manifest["schema"],
            "fixture_pins": FIXTURE_PINS,
            "packaged_verifier": verifier,
        }
    else:
        report = compare_endpoint(args.fixture)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
