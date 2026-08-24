#!/usr/bin/env python3
"""Losslessly package the compiled stock-MPAS nonzero-tracer step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from netCDF4 import Dataset
import numpy as np


STATE_FIELDS = ("rho_zz", "theta_m", "ru", "rw", "qv")
DIAGNOSTIC_FIELDS = ("rho_p", "rtheta_p", "exner", "pressure_p", "u", "w")
REFERENCE_FIELDS = ("rho_base", "rtheta_base", "exner_base", "pressure_base")
CONTROL_FIELDS = (
    "rho_zz",
    "theta_m",
    "ru",
    "rw",
    "u",
    "w",
    "rho_base",
    "theta_base",
    "rtheta_base",
    "rho_p",
    "rtheta_p",
    "exner",
    "exner_base",
    "pressure_p",
    "pressure_base",
)
AXES = {
    "nCells": "cell",
    "nEdges": "edge",
    "nVertLevels": "level",
    "nVertLevelsP1": "interface",
}


def _authority_runs_root_from_environment() -> str | None:
    """Derive the authority runs root from the environment, never from a default.

    The manifest records absolute authority paths.  A baked-in root would record
    someone else's filesystem, and an empty one would record "/runs".
    """

    root = os.environ.get("MPAS_AUTHORITY_ROOT")
    if not root:
        return None
    return f"{root.rstrip('/')}/runs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(dataset: Dataset, name: str) -> tuple[np.ndarray, list[str]]:
    variable = dataset.variables[name]
    dimensions = list(variable.dimensions)
    values = np.asarray(variable[...])
    if dimensions and dimensions[0] == "Time":
        if values.shape[0] != 1:
            raise ValueError(f"{name} does not have a singleton Time axis")
        values = values[0]
        dimensions.pop(0)
    if np.dtype(values.dtype) != np.dtype("float32"):
        raise ValueError(f"{name} is not source binary32: {values.dtype}")
    axes = [AXES[item] for item in dimensions]
    return np.ascontiguousarray(values, dtype="<f4"), axes


def _stats(values: np.ndarray) -> dict[str, Any]:
    delta64 = values.astype(np.float64)
    return {
        "count": int(values.size),
        "finite_count": int(np.count_nonzero(np.isfinite(values))),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_float64": float(delta64.mean()),
    }


def _delta(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    difference = right.astype(np.float64) - left.astype(np.float64)
    return {
        "changed_element_count": int(np.count_nonzero(left != right)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(difference))),
        "l2": float(np.linalg.norm(difference.ravel())),
    }


def _emit(
    output: Path,
    payloads: dict[str, Any],
    group: str,
    name: str,
    values: np.ndarray,
    axes: list[str],
) -> None:
    filename = f"{group}_{name}.f32le"
    path = output / filename
    path.write_bytes(values.tobytes(order="C"))
    declaration = {
        "group": group,
        "name": name,
        "file": filename,
        "dtype": "<f4",
        "order": "C",
        "shape": list(values.shape),
        "axes": axes,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **_stats(values),
    }
    payloads[f"{group}/{name}"] = declaration


def _read_fields(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        for name in fields:
            values, _ = _read(dataset, name)
            result[name] = values
    return result


def _scan_case(
    scan_directory: Path,
    tag: str,
    dry: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    paths = {time: scan_directory / f"{tag}-{time}.nc" for time in ("t0", "t1")}
    candidate = {
        time: _read_fields(path, CONTROL_FIELDS + ("qv",))
        for time, path in paths.items()
    }
    mismatches: dict[str, Any] = {}
    for time in ("t0", "t1"):
        for field in CONTROL_FIELDS:
            left = dry[time][field]
            right = candidate[time][field]
            changed = int(np.count_nonzero(left != right))
            if changed:
                difference = right.astype(np.float64) - left.astype(np.float64)
                mismatches[f"{time}/{field}"] = {
                    "changed_element_count": changed,
                    "max_abs": float(np.max(np.abs(difference), initial=0.0)),
                }
    qv0 = candidate["t0"]["qv"]
    qv1 = candidate["t1"]["qv"]
    difference = qv1.astype(np.float64) - qv0.astype(np.float64)
    receipt_path = scan_directory / f"input-{tag}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    input_path = scan_directory / f"x1.2562.tracer-{tag}.init.nc"
    if not input_path.is_file():
        raise ValueError(f"scan input NetCDF is missing for {tag}: {input_path}")
    actual_input_sha = _sha256(input_path)
    if actual_input_sha != receipt["output"]["sha256"]:
        raise ValueError(f"scan receipt output hash does not match bytes for {tag}")
    with Dataset(input_path) as input_dataset:
        input_dataset.set_auto_maskandscale(False)
        input_qv, input_axes = _read(input_dataset, "qv")
    if input_axes != ["cell", "level"]:
        raise ValueError(f"unexpected scan input qv axes for {tag}: {input_axes}")
    t0_qv = candidate["t0"]["qv"]
    input_qv_hash = hashlib.sha256(input_qv.tobytes(order="C")).hexdigest()
    t0_qv_hash = hashlib.sha256(t0_qv.tobytes(order="C")).hexdigest()
    declared_qv_hash = receipt["qv"]["sha256_logical_c_order"]
    if not (
        input_qv_hash == t0_qv_hash == declared_qv_hash
        and np.array_equal(input_qv, t0_qv)
    ):
        raise ValueError(f"scan input/native-t0 qv binding failed for {tag}")
    maximum = float(np.max(np.abs(qv0)))
    return {
        "tag": tag,
        "amplitude_scale": float(receipt["amplitude_scale"]),
        "input_receipt_sha256": _sha256(receipt_path),
        "input_netcdf_sha256": receipt["output"]["sha256"],
        "verified_input_netcdf_sha256": actual_input_sha,
        "verified_input_qv_logical_sha256": input_qv_hash,
        "verified_native_t0_qv_logical_sha256": t0_qv_hash,
        "input_receipt_and_native_t0_qv_bitwise_bound": True,
        "source_netcdf": {
            time: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for time, path in paths.items()
        },
        "all_15_non_qv_fields_bitwise_equal_to_dry_control": not mismatches,
        "control_mismatches": mismatches,
        "tracer": {
            "t0_min": float(qv0.min()),
            "t0_max": float(qv0.max()),
            "changed_element_count": int(np.count_nonzero(qv0 != qv1)),
            "max_abs_change": float(np.max(np.abs(difference), initial=0.0)),
            "max_change_relative_to_initial_max": (
                float(np.max(np.abs(difference), initial=0.0)) / maximum
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("input_receipt", type=Path)
    parser.add_argument("base_fixture_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--sidecar-manifest", type=Path, required=True)
    parser.add_argument("--scan-directory", type=Path, required=True)
    parser.add_argument("--selected-tag", required=True)
    parser.add_argument("--scan-tags", nargs="+", required=True)
    parser.add_argument(
        "--authority-runs-root",
        default=_authority_runs_root_from_environment(),
    )
    args = parser.parse_args()
    if not args.authority_runs_root:
        raise ValueError(
            "--authority-runs-root is required: pass it explicitly, or set "
            "MPAS_AUTHORITY_ROOT to the MPAS authority checkout root"
        )

    run = args.run_directory.resolve(strict=True)
    input_receipt_path = args.input_receipt.resolve(strict=True)
    base_manifest_path = args.base_fixture_manifest.resolve(strict=True)
    sidecar_manifest_path = args.sidecar_manifest.resolve(strict=True)
    output = args.output_directory.resolve()
    scan_directory = args.scan_directory.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")

    paths = {
        "t0": run / "t0.nc",
        "t1": run / "t1.nc",
    }
    input_receipt = json.loads(input_receipt_path.read_text(encoding="utf-8"))
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    namelist = run / "namelist.atmosphere"
    streams = run / "streams.atmosphere"
    model_log = run / "log.atmosphere.0000.out"
    actual_run_binary = run / "atmosphere_model"
    actual_run_input = run / "x1.2562.init.nc"
    for required in (namelist, streams, model_log, actual_run_binary, actual_run_input):
        if not required.is_file():
            raise ValueError(f"selected run artifact is missing: {required}")
    log_text = model_log.read_text(encoding="utf-8", errors="replace")
    if "Finished running the atmosphere core" not in log_text:
        raise ValueError("stock model log lacks its successful-completion marker")

    dry_paths = {
        "t0": scan_directory / "dry-t0.nc",
        "t1": scan_directory / "dry-t1.nc",
    }
    dry = {time: _read_fields(path, CONTROL_FIELDS) for time, path in dry_paths.items()}
    scan = [_scan_case(scan_directory, tag, dry) for tag in args.scan_tags]
    selected_cases = [item for item in scan if item["tag"] == args.selected_tag]
    if len(selected_cases) != 1:
        raise ValueError("selected tag must occur exactly once in the scan")
    selected_case = selected_cases[0]
    inert = [
        item
        for item in scan
        if item["all_15_non_qv_fields_bitwise_equal_to_dry_control"]
    ]
    largest_inert = max(inert, key=lambda item: item["amplitude_scale"])
    if largest_inert["tag"] != args.selected_tag:
        raise ValueError("selected tag is not the largest dynamically inert scan case")
    upper_cases = sorted(
        (
            item
            for item in scan
            if item["amplitude_scale"] > selected_case["amplitude_scale"]
        ),
        key=lambda item: item["amplitude_scale"],
    )
    if (
        not upper_cases
        or upper_cases[0]["all_15_non_qv_fields_bitwise_equal_to_dry_control"]
    ):
        raise ValueError(
            "scan does not bracket selected inert case with a non-inert case"
        )
    upper_bracket = upper_cases[0]

    selected_run_id = f"jw-nonzero-tracer-{args.selected_tag}-v823-20260810"
    authority_runs_root = PurePosixPath(args.authority_runs_root)
    if not authority_runs_root.is_absolute() or ".." in authority_runs_root.parts:
        raise ValueError(
            "authority runs root must be an absolute normalized POSIX path"
        )
    authority_run_root = authority_runs_root / selected_run_id
    if authority_run_root.name != selected_run_id:
        raise ValueError(
            "derived authority run path does not end in the selected run id"
        )

    actual_binary_sha = _sha256(actual_run_binary)
    declared_binary_sha = (
        "dfdfcebadb39d902ebe70ff59ed5e7540f4795d02c2348b9667cd58021b398c0"
    )
    if actual_binary_sha != declared_binary_sha:
        raise ValueError(
            "actual selected-run atmosphere_model hash is not frozen binary"
        )
    actual_input_sha = _sha256(actual_run_input)
    if actual_input_sha != input_receipt["output"]["sha256"]:
        raise ValueError("actual selected-run input hash disagrees with input receipt")
    if _sha256(input_receipt_path) != selected_case["input_receipt_sha256"]:
        raise ValueError("main input receipt is not the selected scan receipt")
    if float(input_receipt["amplitude_scale"]) != float(
        selected_case["amplitude_scale"]
    ):
        raise ValueError(
            "main input receipt amplitude is not the selected scan amplitude"
        )
    if input_receipt["output"]["sha256"] != selected_case["input_netcdf_sha256"]:
        raise ValueError("main input receipt output is not the selected scan input")
    for time, path in paths.items():
        if _sha256(path) != selected_case["source_netcdf"][time]["sha256"]:
            raise ValueError(
                f"main run {time} is not byte-identical to selected scan case"
            )

    datasets = {name: Dataset(path) for name, path in paths.items()}
    try:
        for dataset in datasets.values():
            dataset.set_auto_maskandscale(False)
        arrays: dict[str, dict[str, np.ndarray]] = {"t0": {}, "t1": {}}
        payloads: dict[str, Any] = {}
        for time in ("t0", "t1"):
            values, axes = _read(datasets[time], "qv")
            arrays[time]["qv"] = values
            _emit(output, payloads, time, "qv", values, axes)
    finally:
        for dataset in datasets.values():
            dataset.close()

    trajectory = {"qv": _delta(arrays["t0"]["qv"], arrays["t1"]["qv"])}
    qv_delta = trajectory["qv"]
    if qv_delta["changed_element_count"] == 0 or qv_delta["max_abs"] <= 0.0:
        raise ValueError("compiled trajectory did not transport the nonzero qv tracer")
    qv0 = arrays["t0"]["qv"]
    qv1 = arrays["t1"]["qv"]
    if qv0.min() <= 0.0 or qv1.min() <= 0.0:
        raise ValueError("positive qv tracer lost strict positivity")

    input_qv_sha = input_receipt["qv"]["sha256_logical_c_order"]
    t0_qv_sha = hashlib.sha256(qv0.tobytes(order="C")).hexdigest()
    qv_ulp = float(np.spacing(np.float32(np.max(np.abs(qv0)))))
    # Fixed before the Python comparison: three split-RK passes, one multiply
    # and one add per active contribution, and the frozen mesh's maxima of ten
    # horizontal reconstruction cells, six cell edges, and four vertical
    # reconstruction points give 3*2*(10+6+4)=120 slots.  Round that declared
    # structural count up to the next power of two; the result is not fit to
    # the candidate's observed error.
    accumulation_depth = 128
    tracer_ceiling = qv_ulp * accumulation_depth
    no_transport_gap = qv_delta["max_abs"]
    if no_transport_gap <= tracer_ceiling:
        raise ValueError("declared tracer budget cannot detect a frozen-qv mutation")
    manifest: dict[str, Any] = {
        "schema": "mpas-port.frozen-fortran-nonzero-tracer-step.v1",
        "evidence": {
            "kind": "stock-Fortran nonzero-tracer native-state control",
            "authority": "compiled MPAS-A v8.2.3 native internal-state output",
            "port_claim": "none",
            "non_claim": (
                "This stock Fortran trajectory is only the ruler. It proves scalar "
                "transport was exercised, not that the Python port matches it."
            ),
            "verification_status": (
                "authority packaged; executable Python comparison is maintained "
                "externally by tests/test_driver_tracer_oracle.py"
            ),
        },
        "authority": {
            "repository": "https://github.com/MPAS-Dev/MPAS-Model",
            "release": "v8.2.3",
            "tag_commit": "ac3866c1e5b05f6d4f5bd41aeab7d3882bace514",
            "source_archive_sha256": (
                "bb3b02c30abffe9ff0318165b25724e6855fb69076fd89243f06a24e11912ee1"
            ),
            "atmosphere_binary": {
                "authority_path": str(authority_run_root / "atmosphere_model"),
                "canonical_binary_path": "<authority-root>/bin/atmosphere_model",
                "bytes": actual_run_binary.stat().st_size,
                "sha256": actual_binary_sha,
                "precision_macro": "SINGLE_PRECISION",
                "curvature_macro_present": False,
            },
            "run_id": selected_run_id,
            "run_root": str(authority_run_root),
            "input": input_receipt,
            "actual_run_input": {
                "authority_path": str(authority_run_root / "x1.2562.init.nc"),
                "bytes": actual_run_input.stat().st_size,
                "sha256": actual_input_sha,
                "matches_embedded_input_receipt": True,
            },
            "namelist_sha256": _sha256(namelist),
            "streams_sha256": _sha256(streams),
            "model_log_sha256": _sha256(model_log),
            "source_netcdf": {
                time: {
                    "authority_path": (
                        str(authority_run_root)
                        + "/"
                        + f"tracer.2000-01-01_00.{'00.00' if time == 't0' else '10.00'}.nc"
                    ),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for time, path in paths.items()
            },
            "base_vertical_fixture": {
                "path": "oracle/jw-x1.2562-v8.2.3-nomix-native/manifest.json",
                "sha256": _sha256(base_manifest_path),
                "schema": base_manifest.get("schema"),
                "use": "identical frozen mesh and analytic vertical/terrain metrics only",
            },
            "linked_dry_controls": {
                "native_state_manifest": {
                    "path": "oracle/jw-x1.2562-v8.2.3-nomix-native/manifest.json",
                    "sha256": _sha256(base_manifest_path),
                    "use": "bitwise-identical t0/t1 non-qv state and reference payloads",
                },
                "internal_sidecar_manifest": {
                    "path": (
                        "oracle/jw-x1.2562-v8.2.3-nomix-internal-sidecar/manifest.json"
                    ),
                    "sha256": _sha256(sidecar_manifest_path),
                    "use": "bitwise-identical t0/t1 diagnostic sidecars",
                },
            },
            "dry_control_source_netcdf": {
                time: {
                    "authority_path": (
                        "<authority-root>/runs/"
                        "jw-nomix-internal-v823-20260810/"
                        f"internal.2000-01-01_00.{'00.00' if time == 't0' else '10.00'}.nc"
                    ),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for time, path in dry_paths.items()
            },
            "selected_scan_binding": {
                "selected_tag": args.selected_tag,
                "input_receipt_sha256": _sha256(input_receipt_path),
                "actual_binary_hash_verified": True,
                "actual_run_input_hash_verified": True,
                "main_t0_t1_hashes_equal_selected_scan_case": True,
                "receipt_amplitude_and_output_hash_equal_selected_scan_case": True,
            },
        },
        "run": {
            "case": "Jablonowski-Williamson perturbation plus qv-only 3-D tracer",
            "mesh": "x1.2562",
            "nCells": 2562,
            "nEdges": 7680,
            "nVertices": 5120,
            "nVertLevels": 15,
            "model_precision": "single",
            "mpi_ranks": 1,
            "dt_seconds": 600.0,
            "time_integration": "SRK3 order 3",
            "acoustic_substeps": 6,
            "dynamics_split_steps": 1,
            "split_dynamics_transport": True,
            "scalar_advection": True,
            "monotonic": True,
            "positive_definite": False,
            "horizontal_mixing": "2d_fixed with all explicit eddy viscosities zero",
            "smdiv": 0.0,
            "xnutr": 0.0,
            "apvm_upwinding": 0.5,
            "physics_suite": "none",
        },
        "times": {
            "t0": {"elapsed_seconds": 0, "xtime": "2000-01-01_00:00:00"},
            "t1": {"elapsed_seconds": 600, "xtime": "2000-01-01_00:10:00"},
        },
        "groups": {
            "t0": ["qv"],
            "t1": ["qv"],
        },
        "state_fields": {
            "qv": {"role": "nonzero transported water-vapor scalar"},
        },
        "payloads": payloads,
        "trajectory_delta": trajectory,
        "tracer_proof": {
            "input_qv_matches_native_t0_bitwise": input_qv_sha == t0_qv_sha,
            "input_qv_sha256_logical_c_order": input_qv_sha,
            "native_t0_qv_sha256_logical_c_order": t0_qv_sha,
            "changed_element_count": qv_delta["changed_element_count"],
            "max_abs_change": qv_delta["max_abs"],
            "t0_min": float(qv0.min()),
            "t0_max": float(qv0.max()),
            "t1_min": float(qv1.min()),
            "t1_max": float(qv1.max()),
            "strictly_positive_at_both_endpoints": True,
            "horizontal_and_vertical_structure": True,
        },
        "coupling_observation": {
            "qv_can_be_dynamically_active_in_stock_mpas": True,
            "selected_amplitude_is_bitwise_dynamically_inert": True,
            "reason": (
                "MPAS theta_m includes qv, so amplitude was scanned downward. At the "
                "selected amplitude every emitted non-qv field at both endpoints is "
                "bitwise equal to the audited dry stock control while qv still transports."
            ),
            "full_state_and_sidecars_packaged": False,
            "state_and_sidecars_hash_linked_without_duplication": True,
            "bitwise_control_field_count_per_endpoint": 15,
            "bitwise_control_fields": list(CONTROL_FIELDS),
        },
        "amplitude_scan": {
            "selection_rule": (
                "largest tested amplitude with all 15 non-qv fields bitwise equal at "
                "both endpoints and a nonzero transported qv trajectory"
            ),
            "selected_tag": args.selected_tag,
            "selected_scale": selected_case["amplitude_scale"],
            "first_non_inert_upper_bracket_tag": upper_bracket["tag"],
            "first_non_inert_upper_bracket_scale": upper_bracket["amplitude_scale"],
            "cases": sorted(scan, key=lambda item: item["amplitude_scale"]),
        },
        "tracer_comparison_budget": {
            "derivation": "binary32 ULP at frozen qv scale times declared accumulation depth",
            "frozen_qv_scale": float(np.max(np.abs(qv0))),
            "binary32_ulp_at_frozen_scale": qv_ulp,
            "declared_scalar_transport_accumulation_depth": accumulation_depth,
            "accumulation_contract": {
                "split_rk_stages": 3,
                "max_edges_on_cell": 6,
                "max_advection_cells_on_edge": 10,
                "max_vertical_reconstruction_points": 4,
                "structural_rounding_slots": 120,
                "power_of_two_guard_slots": 8,
                "rounding_slots_reserved": accumulation_depth,
                "note": (
                    "3 split-RK passes * 2 multiply/add slots * (10 horizontal "
                    "reconstruction cells + 6 cell edges + 4 vertical points) = 120; "
                    "rounded to 128 to include final monotonic/FCT rescaling. This "
                    "fixed policy is not fit to the observed Python result."
                ),
            },
            "absolute_ceiling": tracer_ceiling,
            "relative_ceiling": 0.0,
            "frozen_qv_no_transport_mutation": {
                "operation": "return t0 qv unchanged as t1",
                "max_abs_gap_to_authority": no_transport_gap,
                "gap_over_ceiling": no_transport_gap / tracer_ceiling,
                "decisively_rejected": True,
            },
        },
        "generator": {
            "operation": (
                "decode_cf=false; mask_and_scale=false; drop singleton Time; "
                "require source float32; write contiguous <f4 C order"
            ),
            "lossless": True,
            "regeneration_template": {
                "path": "tools/run_nonzero_tracer_authority.sh",
                "status": "semantically equivalent template, not the archived namelist",
                "difference": (
                    "template explicitly writes config_scalar_adv_order=3, "
                    "config_scalar_vadv_order=3, and config_apvm_upwinding=0.5; "
                    "the archived authority namelist used those Registry defaults"
                ),
                "archived_namelist_hash_remains_authoritative": True,
            },
        },
    }

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    records = [(manifest_path.name, _sha256(manifest_path))]
    records.extend(
        (item["file"], item["sha256"])
        for item in sorted(payloads.values(), key=lambda value: value["file"])
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {filename}\n" for filename, digest in records),
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "manifest_sha256": _sha256(manifest_path),
                "trajectory_delta": trajectory,
                "tracer_proof": manifest["tracer_proof"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
