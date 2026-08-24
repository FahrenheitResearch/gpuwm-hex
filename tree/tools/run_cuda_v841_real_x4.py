#!/usr/bin/env python3
"""Run one fail-closed real-x4 MPAS-A v8.4.1 CUDA engineering step.

The closed F000 init is the execution input.  The native F001 and the three
rescue closures are byte-bound references only.  This tool runs exactly one
120-second, dry-dynamics CUDA step with one passive qv carrier.  It does not
run aggregate physics, write forecast fields, or make weather plots.

Production imports are deliberately lazy.  Destination scope, every closed
input byte, NetCDF metadata, the pinned negative-qv defect, and the complete
source snapshot are checked before CUDA is probed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST_FILE = ROOT / "tests" / "test_cuda_v841_real_x4.py"
RECEIPT_NAME = "cuda-v841-real-x4-one-step.json"
DIAGNOSTIC_CAPSULE_NAME = "cuda-v841-real-x4-lml-diagnostic.nc"
DIAGNOSTIC_RECEIPT_NAME = "cuda-v841-real-x4-lml-diagnostic.json"

SCHEMA = "mpas-port.cuda-v841-real-x4-engineering/v1"
DIAGNOSTIC_SCHEMA = "mpas-port.cuda-v841-real-x4-lml-diagnostic/v1"
DIAGNOSTIC_CAPSULE_SCHEMA = (
    "mpas-port.cuda-v841-real-x4-lml-diagnostic-capsule/v1"
)
PROFILE = "real-x4.163842-v8.4.1-one-step-dry-engineering"
TARGET = (
    "closed official x4.163842 F000, one 120-second resident CUDA v8.4.1 "
    "dry-dynamics step"
)
PREPARATION_METHOD = (
    "verify the closed grid/static/init/native-F001/rescue bytes; load the "
    "precision-preserving grid/static pair; recognize the exact pinned "
    "MPAS-Tools grid_rotate default-real-pi longitude signature and reconcile "
    "only longitude in memory from its literal branch formula; load native "
    "F000 vertical/state fields; attach receipt-bound inactive zero deformation "
    "arrays; transport one passive qv carrier; seal and compare complete "
    "host/device execution fingerprints"
)

MIN_FREE_DEVICE_BYTES = 16 * 1024**3
STEP_COUNT = 1
DT_SECONDS = 120.0
DIAGNOSTIC_NCELLS = 163_842
DIAGNOSTIC_FLOAT32_BYTES_EACH = DIAGNOSTIC_NCELLS * np.dtype(np.float32).itemsize
DIAGNOSTIC_D2H_BYTES = 3 * DIAGNOSTIC_FLOAT32_BYTES_EACH
DIAGNOSTIC_SCOPE_LABEL = (
    "MPAS-A v8.4.1 x4.163842 CUDA +120 s dry step "
    "(passive qv; no physics)"
)
EXPECTED_STAGE_ACOUSTIC_STEPS = (1, 3, 6)
EXPECTED_DYNAMICS_TIMESTEP_SECONDS = 40.0
EXPECTED_DYNAMICS_STAGE_TIMESTEPS = (40.0 / 3.0, 20.0, 40.0)
EXPECTED_SCALAR_STAGE_TIMESTEPS = (40.0, 60.0, 120.0)
EXPECTED_SPLIT_FLUX_REDUCTION = (
    "RKIND:first-copy,current-plus-accumulator,times-reciprocal"
)
EXPECTED_T0_DIAGNOSTICS_SOURCE = "uploaded-exact-sidecar"
EXPECTED_DRIVER_AUTHORITY_NONCLAIMS = (
    "native nonzero tracer",
    "native nonzero u_init/v_init",
    "native nonzero dss",
    "mixing",
    "physics",
)
EXPECTED_REACHED_KERNEL_COUNT = 46
EXPECTED_COMPILED_KERNEL_COUNT = 95
EXPECTED_TRANSLATION_UNITS = (
    "mpas_port.cuda_acoustic",
    "mpas_port.cuda_acoustic_v841",
    "mpas_port.cuda_backend.recovery",
    "mpas_port.cuda_driver",
    "mpas_port.cuda_dynamics_v841",
    "mpas_port.cuda_horizontal",
    "mpas_port.cuda_horizontal_v841",
    "mpas_port.cuda_transport_v841",
)

AUTHORITY_PINS: dict[str, dict[str, Any]] = {
    "grid": {
        "relative_path": (
            "work/v841-vr-static/run-static-v841-conus-official-full-a/"
            "x4.163842.grid.nc"
        ),
        "bytes": 224_139_172,
        "sha256": (
            "48e747157bb1f0b83b96505e268699dfb562b4c1428468cb91457fbb03b1be55"
        ),
    },
    "static": {
        "relative_path": (
            "work/v841-vr-static/run-static-v841-conus-official-full-a/"
            "x4.163842.static.nc"
        ),
        "bytes": 298_860_376,
        "sha256": (
            "f064ee8f8d40085db4bf77a3d5fc6081cd92368b7d3dd32d98110b8b64d177e8"
        ),
    },
    "init": {
        "relative_path": (
            "work/v841-vr-static/run-real-init-v841-conus-official-full-a/"
            "x4.163842.init.nc"
        ),
        "bytes": 1_489_665_020,
        "sha256": (
            "f6e6f41359554ad3b1103235ec4aef026409b0f085a28a7b0f7c38599b9ca2ba"
        ),
    },
    "native_f001": {
        "relative_path": (
            "work/v841-vr-native-forecast/"
            "run-native-v841-conus-official-full-noahmp-gwdo-f001-a/"
            "history.2026-08-10_13.00.00.nc"
        ),
        "bytes": 1_439_206_248,
        "sha256": (
            "d51865e60e37e6c1c548a475aba3cca5f65e92819a184707bb7ddf34b15b7c45"
        ),
    },
    "static_rescue": {
        "relative_path": (
            "work/v841-vr-static/"
            "rescue-static-v841-conus-official-full-a-posthoc-wrapper-a/"
            "rescue-closure.json"
        ),
        "bytes": 4_319,
        "sha256": (
            "d136078f002a9341bd8aad3d8699b8d07d3296b10291e65c30cc3786a7a5f5ed"
        ),
    },
    "init_rescue": {
        "relative_path": (
            "work/v841-vr-static/"
            "rescue-real-init-v841-conus-official-full-a-posthoc-seaice-a/"
            "rescue-closure.json"
        ),
        "bytes": 8_292,
        "sha256": (
            "f45106596f60ecfa848d61de6cbc17892b8f2300bfa37a38b3de5fd4945c9b9f"
        ),
    },
    "native_f001_rescue": {
        "relative_path": (
            "work/v841-vr-native-forecast/"
            "rescue-native-v841-conus-official-full-noahmp-gwdo-f001-a-"
            "posthoc-q2-a/rescue-closure.json"
        ),
        "bytes": 480_254,
        "sha256": (
            "add2be79b44b7e357eeaa14e6fb943ad871b734d463dd1d048393024f930fd0e"
        ),
    },
}

GRID_ROTATE_RECONCILIATION_PIN: dict[str, Any] = {
    "schema": "mpas-port.grid-rotate-longitude-reconciliation/v1",
    "mode": "grid_rotate_default_real_pi_reconciled",
    "grid_sha256": (
        "48e747157bb1f0b83b96505e268699dfb562b4c1428468cb91457fbb03b1be55"
    ),
    "producer_head": "4b5c11b4be471498da36a2637ad1cf49962b3d05",
    "producer_source_sha256": (
        "2be1c67cd2700ffd65b41f241c8858c0e24ca2b67bc7655465ef3807ab654d36"
    ),
    "producer_default_real_pi": 3.1415927410125732,
    "default_real_pi_delta_radians": 8.742278012618954e-08,
    "producer_match_max_ulp": 2.0,
    "entities": {
        "Cell": {
            "count": 163_842,
            "delta_class_counts_0_1_2": [17_474, 91_670, 54_698],
            "longitude_sha256_pre": (
                "4b87c471590174746c4e054def6929a94c5564aab3d9011b55950763061070b3"
            ),
            "longitude_sha256_post": (
                "220b5ef91ca0a9a63a707867c653835990074871d93966d2e50de01fe00e6ac4"
            ),
            "latitude_sha256": (
                "75d440707c48b108437b4522fbe41593603afddd4fe20004208272a0406d594a"
            ),
            "cartesian_sha256": {
                "xCell": (
                    "3eeb30d0812d9cfeaee41d6addcd0f1ddf73da694c25bdab28bf8ad49b11779b"
                ),
                "yCell": (
                    "64cd57df0df68a28246c69389fb5c0aba1950550eebcb2382b1db05028b251b0"
                ),
                "zCell": (
                    "eb71060917402b21c05ccd486eec4aefa98a90d222918553253f51f812e6f08c"
                ),
            },
        },
        "Edge": {
            "count": 491_520,
            "delta_class_counts_0_1_2": [52_432, 275_000, 164_088],
            "longitude_sha256_pre": (
                "b95de4d47a54e125ef17f7f8c92d58599718367bcfa28e20f850c6d709e39cf6"
            ),
            "longitude_sha256_post": (
                "3be12602e9a29b3553b334f43c1160f8a848aa890ec20ad410552ceec25f642e"
            ),
            "latitude_sha256": (
                "22aa62ec46dbe5b1544be310cfd417e7aade6a41471f599c5d7e42a74938a994"
            ),
            "cartesian_sha256": {
                "xEdge": (
                    "42f9bfc04fb2e1dc0cbd723267c76f816195b2baf318867ba5316dcae371ca94"
                ),
                "yEdge": (
                    "38b80db81481bf97968d4a536e2a788cfa5099367547806902bb7b076bc9d3d0"
                ),
                "zEdge": (
                    "00c624dfb74fcc704504f3ccad5edad5fb1d76d8328726c60cf020d13603d633"
                ),
            },
        },
        "Vertex": {
            "count": 327_680,
            "delta_class_counts_0_1_2": [34_953, 183_333, 109_394],
            "longitude_sha256_pre": (
                "ee524c5a8b1425208fa26972d96fe1855932b364300d014ef193ca50e3377658"
            ),
            "longitude_sha256_post": (
                "5a8120832ed17b6af486a952038c5d14c334976beca03fbb7fe378895d5befc5"
            ),
            "latitude_sha256": (
                "4d2ba0024970bb5713c3af6a0b220df53de99a9241526f3b5c32669f651d768f"
            ),
            "cartesian_sha256": {
                "xVertex": (
                    "03fcd8b5af9dada01c9e999945573e0cdb18b093c2985aaa58e7b2d38d472000"
                ),
                "yVertex": (
                    "a91bf75c96fec950fae4bf7fb22f666f41d2f27d199ea98f0c8867da5f93db39"
                ),
                "zVertex": (
                    "d4d2930347519d396f76e3e1452e757a6e3260534a58306a84593a1b9559abae"
                ),
            },
        },
    },
}

EXPECTED_DIMENSIONS: dict[str, dict[str, int]] = {
    "grid": {
        "nCells": 163_842,
        "nVertices": 327_680,
        "nEdges": 491_520,
        "maxEdges": 10,
        "maxEdges2": 20,
        "TWO": 2,
        "vertexDegree": 3,
        "codeLen": 1,
    },
    "static": {
        "StrLen": 64,
        "Time": 1,
        "nCells": 163_842,
        "nEdges": 491_520,
        "nVertices": 327_680,
        "TWO": 2,
        "maxEdges": 10,
        "maxEdges2": 20,
        "vertexDegree": 3,
        "R3": 3,
        "nMonths": 12,
        "nSoilComps": 8,
        "FIFTEEN": 15,
    },
    "init": {
        "nVertLevels": 55,
        "nCells": 163_842,
        "Time": 1,
        "StrLen": 64,
        "nEdges": 491_520,
        "nVertices": 327_680,
        "TWO": 2,
        "maxEdges": 10,
        "maxEdges2": 20,
        "vertexDegree": 3,
        "R3": 3,
        "nMonths": 12,
        "nSoilComps": 8,
        "FIFTEEN": 15,
        "nVertLevelsP1": 56,
        "nSoilLevels": 4,
    },
    "native_f001": {
        "nVertLevels": 55,
        "nCells": 163_842,
        "Time": 1,
        "nEdges": 491_520,
        "nVertices": 327_680,
        "TWO": 2,
        "maxEdges": 10,
        "maxEdges2": 20,
        "vertexDegree": 3,
        "nVertLevelsP1": 56,
        "StrLen": 64,
        "nOznLevels": 59,
        "nMonths": 12,
        "nSoilLevels": 4,
    },
}


def _variables(
    integer_names: Sequence[tuple[str, tuple[str, ...]]],
    float_names: Sequence[tuple[str, tuple[str, ...]]],
    *,
    float_dtype: str,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    result = {name: ("int32", dims) for name, dims in integer_names}
    result.update({name: (float_dtype, dims) for name, dims in float_names})
    return result


VARIABLE_CONTRACTS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "grid": _variables(
        (
            ("cellsOnEdge", ("nEdges", "TWO")),
            ("edgesOnCell", ("nCells", "maxEdges")),
            ("nEdgesOnCell", ("nCells",)),
            ("cellsOnCell", ("nCells", "maxEdges")),
            ("verticesOnEdge", ("nEdges", "TWO")),
            ("edgesOnEdge", ("nEdges", "maxEdges2")),
            ("nEdgesOnEdge", ("nEdges",)),
            ("verticesOnCell", ("nCells", "maxEdges")),
            ("edgesOnVertex", ("nVertices", "vertexDegree")),
            ("cellsOnVertex", ("nVertices", "vertexDegree")),
            ("indexToCellID", ("nCells",)),
        ),
        (
            ("weightsOnEdge", ("nEdges", "maxEdges2")),
            ("dcEdge", ("nEdges",)),
            ("dvEdge", ("nEdges",)),
            ("areaCell", ("nCells",)),
            ("areaTriangle", ("nVertices",)),
            ("kiteAreasOnVertex", ("nVertices", "vertexDegree")),
            ("latCell", ("nCells",)),
            ("lonCell", ("nCells",)),
            ("meshDensity", ("nCells",)),
            ("latEdge", ("nEdges",)),
            ("lonEdge", ("nEdges",)),
            ("angleEdge", ("nEdges",)),
        ),
        float_dtype="float64",
    ),
    "static": _variables(
        (),
        (
            ("deriv_two", ("nEdges", "TWO", "FIFTEEN")),
            ("fVertex", ("nVertices",)),
            ("fEdge", ("nEdges",)),
            ("ter", ("nCells",)),
        ),
        float_dtype="float32",
    ),
    "init": _variables(
        (),
        (
            ("zgrid", ("nCells", "nVertLevelsP1")),
            ("rdzw", ("nVertLevels",)),
            ("dzu", ("nVertLevels",)),
            ("rdzu", ("nVertLevels",)),
            ("fzm", ("nVertLevels",)),
            ("fzp", ("nVertLevels",)),
            ("zz", ("nCells", "nVertLevels")),
            ("zxu", ("nEdges", "nVertLevels")),
            ("dss", ("nCells", "nVertLevels")),
            ("zb", ("nEdges", "TWO", "nVertLevelsP1")),
            ("zb3", ("nEdges", "TWO", "nVertLevelsP1")),
            ("cf1", ()),
            ("cf2", ()),
            ("cf3", ()),
            ("rho", ("Time", "nCells", "nVertLevels")),
            ("theta", ("Time", "nCells", "nVertLevels")),
            ("qv", ("Time", "nCells", "nVertLevels")),
            ("u", ("Time", "nEdges", "nVertLevels")),
            ("w", ("Time", "nCells", "nVertLevelsP1")),
            ("rho_base", ("Time", "nCells", "nVertLevels")),
            ("theta_base", ("Time", "nCells", "nVertLevels")),
            ("u_init", ("nVertLevels",)),
            ("v_init", ("nVertLevels",)),
        ),
        float_dtype="float32",
    ),
    "native_f001": _variables(
        (("indexToCellID", ("nCells",)),),
        (
            ("surface_pressure", ("Time", "nCells")),
            ("pressure", ("Time", "nCells", "nVertLevels")),
            ("theta", ("Time", "nCells", "nVertLevels")),
            ("rho", ("Time", "nCells", "nVertLevels")),
            ("qv", ("Time", "nCells", "nVertLevels")),
            ("uReconstructZonal", ("Time", "nCells", "nVertLevels")),
            ("uReconstructMeridional", ("Time", "nCells", "nVertLevels")),
            ("zgrid", ("nCells", "nVertLevelsP1")),
            ("latCell", ("nCells",)),
            ("lonCell", ("nCells",)),
        ),
        float_dtype="float32",
    ),
}

NEGATIVE_QV_PIN = {
    "logical_shape": [55, 163_842],
    "dtype": "float32",
    "full_qv_sha256": (
        "c2f16355a7c67d51fd8f1e30c86f6ae43bddc35b75e668218525f4094024f383"
    ),
    "negative_count": 215,
    "negative_indices_sha256": (
        "d757607f3dec3ddae5a521ec6b4740770016253b0a1742af336a4b838e005f13"
    ),
    "negative_values_sha256": (
        "03de8cf95e8f207f5280935e276051257a994bd6680ca6b16e782f168b5221a8"
    ),
    "negative_min": -0.00012415398668963462,
    "negative_max": -1.279335248849378e-10,
    "negative_sum_float64": -0.0005964658577688742,
}

ZERO_DEFC_SHAPE = (163_842, 10)
ZERO_DEFC_BYTES = 6_553_680
ZERO_DEFC_ARRAY_SHA256 = (
    "f1c262d4848f9bf293ce3b084130a92e1b25d437451e07ca5294d43a52472257"
)

CLAIM = (
    "one work-only real-x4.163842 MPAS-A v8.4.1 CUDA engineering step from "
    "the pinned F000 bytes; dry dynamics only with qv as one passive carrier"
)
NONCLAIMS = (
    "not native or compiled-MPAS numerical authority",
    "not a full-physics, NoahMP, GWDO, microphysics, radiation, PBL, or Arwen run",
    "not a native-F001 reproduction or comparator claim",
    "not a forecast beyond one 120-second engineering step",
    "not a cleanup, positivity, moist-physics, or publication claim for qv",
    "no forecast history, regridding, Rust rendering, or weather plot is included "
    "in this production-step receipt; any optional post-receipt diagnostic is "
    "separate and non-authoritative",
)
NEGATIVE_QV_NONCLAIM = (
    "the exact 215 negative source qv values are retained as a pinned input "
    "defect; qv is passive and monotonic/positive-definite transport is off; "
    "no moist-physics or cleaned-moisture claim is made"
)


def sha256_file(path: str | Path) -> str:
    selected = Path(path)
    if not selected.is_file():
        raise FileNotFoundError(f"required authority is not a regular file: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def require_exact_grid_rotate_reconciliation(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the in-memory longitude bridge to the exact closed x4 witness."""

    try:
        producer = dict(evidence["producer_authority"])
        entities = {
            name: dict(evidence["entities"][name])
            for name in ("Cell", "Edge", "Vertex")
        }
        actual_pin = {
            "schema": evidence["schema"],
            "mode": evidence["mode"],
            "grid_sha256": evidence["input_grid_sha256"],
            "producer_head": producer["head"],
            "producer_source_sha256": producer["sha256"],
            "producer_default_real_pi": evidence["producer_default_real_pi"],
            "default_real_pi_delta_radians": evidence[
                "default_real_pi_delta_radians"
            ],
            "producer_match_max_ulp": evidence["producer_match_max_ulp"],
            "entities": {
                name: {
                    "count": entities[name]["count"],
                    "delta_class_counts_0_1_2": entities[name][
                        "delta_class_counts_0_1_2"
                    ],
                    "longitude_sha256_pre": entities[name][
                        "longitude_sha256_pre"
                    ],
                    "longitude_sha256_post": entities[name][
                        "longitude_sha256_post"
                    ],
                    "latitude_sha256": entities[name]["latitude_sha256_pre"],
                    "cartesian_sha256": entities[name]["cartesian_sha256_pre"],
                }
                for name in ("Cell", "Edge", "Vertex")
            },
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "x4 grid_rotate longitude reconciliation evidence is incomplete"
        ) from error
    if actual_pin != GRID_ROTATE_RECONCILIATION_PIN:
        raise RuntimeError(
            "x4 grid_rotate longitude reconciliation exact pin changed"
        )
    if (
        evidence.get("in_memory_only") is not True
        or evidence.get("input_grid_name") != "x4.163842.grid.nc"
        or evidence.get("true_binary64_pi") != float(np.pi)
        or evidence.get("latitude_match_max_ulp") != 256.0
        or evidence.get("corrected_atan2_atol_radians")
        != 16.0 * np.finfo(np.float64).eps
        or evidence.get("validation_tolerance_changed") is not False
        or evidence.get("strict_mesh_validation_after_reconciliation") != "passed"
        or evidence.get("cartesian_latitude_topology_metrics_changed") is not False
        or evidence.get("changed_longitude_fields")
        != ["lonCell", "lonEdge", "lonVertex"]
    ):
        raise RuntimeError(
            "x4 longitude reconciliation mutation scope or strict validation changed"
        )
    if (
        producer.get("repository") != "MPAS-Dev/MPAS-Tools"
        or producer.get("relative_path") != "mesh_tools/grid_rotate/grid_rotate.f90"
        or producer.get("source_lines") != "15,25,447-501"
        or producer.get("pi_expression") != "pii = 2.*asin(1.0)"
        or producer.get("storage_kind") != "RKIND=8"
        or producer.get("expression_kind")
        != "unsuffixed default real (binary32)"
    ):
        raise RuntimeError("x4 longitude reconciliation producer formula changed")

    for entity, pin in GRID_ROTATE_RECONCILIATION_PIN["entities"].items():
        details = entities[entity]
        expected_counts = {
            "0": pin["delta_class_counts_0_1_2"][0],
            "0.5": 0,
            "1": pin["delta_class_counts_0_1_2"][1],
            "1.5": 0,
            "2": pin["delta_class_counts_0_1_2"][2],
        }
        if details.get("pi_error_multiplicity_counts") != expected_counts:
            raise RuntimeError(
                f"x4 {entity} longitude correction branch counts changed"
            )
        if details.get("latitude_sha256_post") != pin["latitude_sha256"]:
            raise RuntimeError(f"x4 {entity} latitude changed during reconciliation")
        if details.get("cartesian_sha256_post") != pin["cartesian_sha256"]:
            raise RuntimeError(f"x4 {entity} Cartesian bytes changed during reconciliation")
        producer_ulp = details.get("producer_formula_max_ulp_error")
        latitude_ulp = details.get("producer_latitude_max_ulp_error")
        producer_gap = details.get("producer_formula_max_abs_gap_radians")
        post_angle_gap = details.get("post_max_atan2_angular_gap_radians")
        post_cartesian_gap = details.get("post_max_cartesian_component_gap")
        if (
            type(producer_ulp) not in (int, float)
            or not np.isfinite(producer_ulp)
            or float(producer_ulp) < 0.0
            or float(producer_ulp) > 2.0
            or type(latitude_ulp) not in (int, float)
            or not np.isfinite(latitude_ulp)
            or float(latitude_ulp) < 0.0
            or float(latitude_ulp) > 256.0
            or type(producer_gap) not in (int, float)
            or not np.isfinite(producer_gap)
            or float(producer_gap) > 4.0 * np.finfo(np.float64).eps
            or type(post_angle_gap) not in (int, float)
            or not np.isfinite(post_angle_gap)
            or float(post_angle_gap) > 16.0 * np.finfo(np.float64).eps
            or type(post_cartesian_gap) not in (int, float)
            or not np.isfinite(post_cartesian_gap)
            or float(post_cartesian_gap) > 5.0e-14
        ):
            raise RuntimeError(
                f"x4 {entity} longitude reconciliation numeric envelope changed"
            )
        canonicalization = details.get("canonicalization")
        if (
            not isinstance(canonicalization, Mapping)
            or canonicalization.get("method")
            != "(longitude + pi) modulo 2pi minus pi"
            or canonicalization.get("trig_equivalence_atol")
            != 8.0 * np.finfo(np.float64).eps
            or type(canonicalization.get("sin_max_abs_gap")) not in (int, float)
            or float(canonicalization["sin_max_abs_gap"])
            > 8.0 * np.finfo(np.float64).eps
            or type(canonicalization.get("cos_max_abs_gap")) not in (int, float)
            or float(canonicalization["cos_max_abs_gap"])
            > 8.0 * np.finfo(np.float64).eps
        ):
            raise RuntimeError(
                f"x4 {entity} longitude canonicalization contract changed"
            )

    return {
        "status": "passed",
        "exact_pin": actual_pin,
        "exact_pin_sha256": canonical_json_sha256(actual_pin),
        "full_evidence_sha256": canonical_json_sha256(dict(evidence)),
        "scope": (
            "in-memory longitude metadata reconciliation only; Cartesian, latitude, "
            "topology, metrics, input bytes, and strict mesh tolerance unchanged"
        ),
    }


def default_authority_paths() -> dict[str, Path]:
    return {
        role: ROOT / str(pin["relative_path"])
        for role, pin in AUTHORITY_PINS.items()
    }


def negative_qv_fingerprint(logical_qv: Any) -> dict[str, Any]:
    qv = np.ascontiguousarray(np.asarray(logical_qv))
    if qv.ndim != 2 or qv.dtype != np.dtype(np.float32):
        raise TypeError("logical qv must be a two-dimensional float32 array")
    indices = np.ascontiguousarray(np.argwhere(qv < 0.0), dtype=np.int64)
    values = np.ascontiguousarray(qv[qv < 0.0])
    return {
        "logical_shape": [int(value) for value in qv.shape],
        "dtype": str(qv.dtype),
        "full_qv_sha256": array_sha256(qv),
        "negative_count": int(values.size),
        "negative_indices_sha256": array_sha256(indices),
        "negative_values_sha256": array_sha256(values),
        "negative_min": float(values.min()) if values.size else None,
        "negative_max": float(values.max()) if values.size else None,
        "negative_sum_float64": float(values.astype(np.float64).sum()),
    }


def _file_record(role: str, path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    unresolved = path.expanduser().absolute()
    is_junction = getattr(unresolved, "is_junction", lambda: False)
    if unresolved.is_symlink() or is_junction():
        raise ValueError(f"{role} authority must not be a symlink or junction")
    selected = unresolved.resolve(strict=True)
    size = selected.stat().st_size
    digest = sha256_file(selected)
    if size != int(pin["bytes"]):
        raise RuntimeError(f"{role} byte count changed: {size} != {pin['bytes']}")
    if digest != pin["sha256"]:
        raise RuntimeError(
            f"{role} SHA-256 changed: {digest} != {pin['sha256']}"
        )
    return {
        "path": str(selected),
        "bytes": size,
        "sha256": digest,
        "literal_pin": dict(pin),
    }


def _inspect_netcdf(role: str, path: Path) -> dict[str, Any]:
    from netCDF4 import Dataset

    expected_dimensions = EXPECTED_DIMENSIONS[role]
    expected_variables = VARIABLE_CONTRACTS[role]
    with Dataset(path, mode="r") as dataset:
        dataset.set_auto_mask(False)
        dimensions = {
            name: len(dimension) for name, dimension in dataset.dimensions.items()
        }
        if dimensions != expected_dimensions:
            raise RuntimeError(
                f"{role} dimension contract changed: {dimensions} "
                f"!= {expected_dimensions}"
            )
        variables: dict[str, dict[str, Any]] = {}
        for name, (expected_dtype, expected_dims) in expected_variables.items():
            if name not in dataset.variables:
                raise RuntimeError(f"{role} is missing required variable {name}")
            variable = dataset.variables[name]
            actual_dtype = str(np.dtype(variable.dtype))
            actual_dims = tuple(variable.dimensions)
            if actual_dtype != expected_dtype or actual_dims != expected_dims:
                raise RuntimeError(
                    f"{role}.{name} metadata changed: "
                    f"{actual_dtype}/{actual_dims} != "
                    f"{expected_dtype}/{expected_dims}"
                )
            variables[name] = {
                "dtype": actual_dtype,
                "dimensions": list(actual_dims),
                "shape": [int(value) for value in variable.shape],
            }
        defc_presence = {
            name: name in dataset.variables for name in ("defc_a", "defc_b")
        }
        qv_fingerprint = None
        if role == "init":
            raw_qv = np.asarray(dataset.variables["qv"][0])
            logical_qv = np.ascontiguousarray(raw_qv.T)
            qv_fingerprint = negative_qv_fingerprint(logical_qv)
            if qv_fingerprint != NEGATIVE_QV_PIN:
                raise RuntimeError(
                    "init negative-qv fingerprint changed: "
                    f"{qv_fingerprint} != {NEGATIVE_QV_PIN}"
                )

    if role in {"grid", "static", "init"} and any(defc_presence.values()):
        raise RuntimeError(
            f"{role} unexpectedly materializes defc_a/defc_b: {defc_presence}"
        )
    return {
        "dimensions": dimensions,
        "variables": variables,
        "defc_presence": defc_presence,
        "negative_qv": qv_fingerprint,
    }


def verify_authorities(paths: Mapping[str, Path]) -> dict[str, Any]:
    if set(paths) != set(AUTHORITY_PINS):
        raise ValueError("authority path inventory changed")
    files = {
        role: _file_record(role, Path(paths[role]), AUTHORITY_PINS[role])
        for role in AUTHORITY_PINS
    }
    netcdf = {
        role: _inspect_netcdf(role, Path(paths[role]))
        for role in ("grid", "static", "init", "native_f001")
    }
    core = {
        "files": files,
        "netcdf": netcdf,
        "native_f001_role": (
            "byte-bound full-physics weather reference only; never loaded into "
            "the dry CUDA driver and never used as a numerical ruler"
        ),
        "negative_qv_nonclaim": NEGATIVE_QV_NONCLAIM,
    }
    return {**core, "sha256": canonical_json_sha256(core)}


def source_snapshot() -> dict[str, Any]:
    paths = sorted((ROOT / "src" / "mpas_port").rglob("*.py"))
    paths.extend((Path(__file__).resolve(), TEST_FILE.resolve(strict=True)))
    files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(paths))
    }
    return {
        "file_count": len(files),
        "files": files,
        "files_sha256": canonical_json_sha256(files),
    }


def _reject_symlink_ancestry(path: Path, label: str) -> None:
    selected = path.expanduser().absolute()
    for parent in (selected, *selected.parents):
        is_junction = getattr(parent, "is_junction", lambda: False)
        if parent.exists() and (parent.is_symlink() or is_junction()):
            raise ValueError(f"{label} ancestry contains a symlink: {parent}")


def _require_absent_directory(path: Path, label: str) -> None:
    if path.exists():
        raise FileExistsError(f"{label} must be absent: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"{label} parent does not exist: {path.parent}")


def validate_destination_paths(
    cache_root: Path,
    output_root: Path,
    *,
    protected_inputs: Sequence[Path],
) -> tuple[Path, Path]:
    """Require two fresh, nonoverlapping, authority-disjoint directories."""

    _reject_symlink_ancestry(cache_root, "cache root")
    _reject_symlink_ancestry(output_root, "output root")
    cache = cache_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    _require_absent_directory(cache, "cache root")
    _require_absent_directory(output, "output root")
    if cache == output or cache in output.parents or output in cache.parents:
        raise ValueError("cache root and output root must not overlap")

    protected = (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "tools",
        ROOT / ".git",
        *protected_inputs,
    )
    for label, destination in (("cache root", cache), ("output root", output)):
        for raw in protected:
            authority = raw.resolve()
            if destination == authority or authority in destination.parents:
                raise ValueError(f"{label} overlaps protected authority {authority}")
            if destination in authority.parents:
                raise ValueError(f"{label} contains protected authority {authority}")
    return cache, output


def validate_diagnostic_destination_path(
    diagnostic_root: Path,
    *,
    cache_root: Path,
    output_root: Path,
    protected_inputs: Sequence[Path],
) -> Path:
    """Require one fresh sibling root for the optional post-receipt lane."""

    _reject_symlink_ancestry(diagnostic_root, "diagnostic root")
    diagnostic = diagnostic_root.expanduser().resolve()
    _require_absent_directory(diagnostic, "diagnostic root")
    cache = cache_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    for other_name, other in (("cache root", cache), ("output root", output)):
        if diagnostic == other or diagnostic in other.parents or other in diagnostic.parents:
            raise ValueError(f"diagnostic root and {other_name} must not overlap")
    protected = (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "tools",
        ROOT / ".git",
        *protected_inputs,
    )
    for raw in protected:
        authority = raw.resolve()
        if diagnostic == authority or authority in diagnostic.parents:
            raise ValueError(
                f"diagnostic root overlaps protected authority {authority}"
            )
        if diagnostic in authority.parents:
            raise ValueError(
                f"diagnostic root contains protected authority {authority}"
            )
    return diagnostic


def require_exact_production_receipt_tree(
    output_root: Path, receipt: Path
) -> dict[str, Any]:
    """Keep the production root as exactly its one released receipt file."""

    root = output_root.resolve(strict=True)
    selected = receipt.resolve(strict=True)
    entries = tuple(root.iterdir())
    if entries != (selected,) or not selected.is_file():
        raise RuntimeError(
            "production output root must contain exactly the one-step receipt"
        )
    return {
        "root": str(root),
        "entry_count": 1,
        "only_entry": selected.name,
        "receipt_file_sha256": sha256_file(selected),
    }


def _write_exclusive_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _require_host_diagnostic_plane(name: str, value: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value))
    if (
        array.shape != (DIAGNOSTIC_NCELLS,)
        or array.dtype != np.dtype(np.float32)
        or array.nbytes != DIAGNOSTIC_FLOAT32_BYTES_EACH
        or not np.all(np.isfinite(array))
    ):
        raise RuntimeError(
            f"diagnostic {name} must be one finite x4 float32 cell plane"
        )
    return array


def prepare_lml_diagnostic_context(mesh: Any, state: Any, vertical: Any) -> dict[str, Any]:
    """Retain only host F000 inputs needed by the separate temperature lane."""

    arrays = getattr(mesh, "arrays", None)
    if not isinstance(arrays, Mapping):
        raise TypeError("diagnostic mesh must expose its closed arrays")
    cell_ids = np.ascontiguousarray(np.asarray(arrays["indexToCellID"]))
    latitude = np.ascontiguousarray(np.asarray(arrays["latCell"]))
    longitude = np.ascontiguousarray(np.asarray(arrays["lonCell"]))
    zz = np.ascontiguousarray(np.asarray(vertical.zz[0]))
    if (
        cell_ids.shape != (DIAGNOSTIC_NCELLS,)
        or cell_ids.dtype != np.dtype(np.int32)
        or np.unique(cell_ids).size != DIAGNOSTIC_NCELLS
    ):
        raise RuntimeError("diagnostic cell identity is not exact x4 int32")
    for name, coordinate in (("latitude", latitude), ("longitude", longitude)):
        if (
            coordinate.shape != (DIAGNOSTIC_NCELLS,)
            or coordinate.dtype != np.dtype(np.float64)
            or not np.all(np.isfinite(coordinate))
        ):
            raise RuntimeError(f"diagnostic {name} is not exact x4 binary64")
    if np.any(np.abs(latitude) > np.pi / 2.0 + 1.0e-12):
        raise RuntimeError("diagnostic latitude lies outside the sphere")
    zz = _require_host_diagnostic_plane("zz[0]", zz)
    if np.any(zz <= 0.0):
        raise RuntimeError("diagnostic zz[0] must remain positive")
    initial = {
        "rho": _require_host_diagnostic_plane("initial rho[0]", state.rho[0]).copy(),
        "rho_theta": _require_host_diagnostic_plane(
            "initial rho_theta[0]", state.rho_theta[0]
        ).copy(),
        "qv": _require_host_diagnostic_plane(
            "initial qv[0]", state.scalars[0, 0]
        ).copy(),
    }
    if np.any(initial["rho"] <= 0.0) or np.any(initial["rho_theta"] <= 0.0):
        raise RuntimeError("initial diagnostic rho/rho_theta must be positive")
    return {
        "cell_ids": cell_ids.copy(),
        "latitude_radians": latitude.copy(),
        "longitude_radians": longitude.copy(),
        "zz_lowest_model_level": zz.copy(),
        "initial": initial,
    }


def download_post_step_lml_diagnostic(cp: Any, state: Any) -> dict[str, Any]:
    """Copy exactly three post-step x4 float32 cell planes from the device."""

    selected = (
        ("rho", "rho[0,:]", state.rho[0]),
        ("rho_theta", "rho_theta[0,:]", state.rho_theta[0]),
        ("qv", "scalars[0,0,:]", state.scalars[0, 0]),
    )
    host: dict[str, np.ndarray] = {}
    fields: list[dict[str, Any]] = []
    for name, selector, device in selected:
        shape = tuple(int(extent) for extent in device.shape)
        dtype = np.dtype(device.dtype)
        nbytes = int(device.nbytes)
        c_contiguous = bool(device.flags.c_contiguous)
        if (
            shape != (DIAGNOSTIC_NCELLS,)
            or dtype != np.dtype(np.float32)
            or nbytes != DIAGNOSTIC_FLOAT32_BYTES_EACH
            or not c_contiguous
        ):
            raise RuntimeError(f"device diagnostic selector {selector} changed")
        copied = _require_host_diagnostic_plane(
            selector,
            cp.asnumpy(device, order="C", blocking=True),
        )
        host[name] = copied
        fields.append(
            {
                "name": name,
                "device_selector": selector,
                "shape": [DIAGNOSTIC_NCELLS],
                "dtype": "float32",
                "c_contiguous": True,
                "bytes": copied.nbytes,
                "array_sha256": array_sha256(copied),
            }
        )
    total = sum(int(field["bytes"]) for field in fields)
    if total != DIAGNOSTIC_D2H_BYTES:
        raise RuntimeError(
            f"post-step diagnostic D2H changed: {total} != {DIAGNOSTIC_D2H_BYTES}"
        )
    if np.any(host["rho"] <= 0.0) or np.any(host["rho_theta"] <= 0.0):
        raise RuntimeError("post-step diagnostic rho/rho_theta must be positive")
    return {
        "host": host,
        "receipt": {
            "timing": "after exclusive production-step receipt publication",
            "mechanism": "three explicit blocking cupy.asnumpy C-order copies",
            "fields": fields,
            "field_count": 3,
            "bytes_each": DIAGNOSTIC_FLOAT32_BYTES_EACH,
            "total_bytes": total,
            "separate_from_production_step_receipt_d2h": True,
        },
    }


def _write_lml_diagnostic_capsule(
    path: Path,
    *,
    context: Mapping[str, Any],
    endpoint: Mapping[str, np.ndarray],
    production_receipt_file_sha256: str,
    production_receipt_payload_sha256: str,
) -> None:
    from netCDF4 import Dataset

    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    records = {
        name: np.stack((context["initial"][name], endpoint[name]), axis=0)
        for name in ("rho", "rho_theta", "qv")
    }
    try:
        with Dataset(temporary, "w", format="NETCDF4_CLASSIC") as dataset:
            dataset.createDimension("Time", 2)
            dataset.createDimension("nCells", DIAGNOSTIC_NCELLS)
            dataset.setncatts(
                {
                    "Conventions": "CF-1.10",
                    "diagnostic_capsule_schema": DIAGNOSTIC_CAPSULE_SCHEMA,
                    "source_release": "MPAS-A v8.4.1",
                    "scope_label": DIAGNOSTIC_SCOPE_LABEL,
                    "authority_claim": 0,
                    "engineering_dry": 1,
                    "physics_executed": 0,
                    "qv_role": "passive carrier; no positivity cleanup or moist physics",
                    "production_step_receipt_file_sha256": production_receipt_file_sha256,
                    "production_step_receipt_payload_sha256": production_receipt_payload_sha256,
                    "post_step_d2h_bytes": DIAGNOSTIC_D2H_BYTES,
                }
            )
            time = dataset.createVariable("time", "f8", ("Time",))
            time.setncatts(
                {
                    "standard_name": "time",
                    "long_name": "model time",
                    "axis": "T",
                    "calendar": "gregorian",
                    "units": "seconds since 2026-08-10 12:00:00",
                }
            )
            time[:] = np.asarray((0.0, DT_SECONDS), dtype=np.float64)
            ids = dataset.createVariable("indexToCellID", "i4", ("nCells",))
            ids.long_name = "MPAS cell identifier"
            ids[:] = context["cell_ids"]
            for name, long_name in (
                ("latCell", "latitude of cell center"),
                ("lonCell", "longitude of cell center"),
            ):
                variable = dataset.createVariable(name, "f8", ("nCells",))
                variable.units = "radians"
                variable.long_name = long_name
                variable[:] = context[
                    "latitude_radians" if name == "latCell" else "longitude_radians"
                ]
            zz = dataset.createVariable(
                "zz_lowest_model_level", "f4", ("nCells",)
            )
            zz.units = "1"
            zz.long_name = "MPAS terrain-following inverse density metric at level 0"
            zz[:] = context["zz_lowest_model_level"]
            metadata = {
                "rho_zz_lowest_model_level": (
                    "rho",
                    "kg m-3",
                    "terrain-coupled dry-air density state at level 0",
                ),
                "rho_theta_lowest_model_level": (
                    "rho_theta",
                    "K kg m-3",
                    "terrain-coupled density times moist potential temperature at level 0",
                ),
                "qv_lowest_model_level": (
                    "qv",
                    "kg kg-1",
                    "passive water-vapor carrier at level 0",
                ),
            }
            for variable_name, (record_name, units, long_name) in metadata.items():
                variable = dataset.createVariable(
                    variable_name, "f4", ("Time", "nCells")
                )
                variable.units = units
                variable.long_name = long_name
                variable.coordinates = "latCell lonCell"
                variable[:] = records[record_name]
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_lml_diagnostic_capsule(
    path: Path,
    *,
    context: Mapping[str, Any],
    endpoint: Mapping[str, np.ndarray],
    production_receipt_file_sha256: str,
    production_receipt_payload_sha256: str,
) -> dict[str, Any]:
    from netCDF4 import Dataset

    expected_variables = {
        "time",
        "indexToCellID",
        "latCell",
        "lonCell",
        "zz_lowest_model_level",
        "rho_zz_lowest_model_level",
        "rho_theta_lowest_model_level",
        "qv_lowest_model_level",
    }
    with Dataset(path, "r") as dataset:
        dataset.set_auto_mask(False)
        if dataset.data_model != "NETCDF4_CLASSIC":
            raise RuntimeError("diagnostic capsule is not NETCDF4_CLASSIC")
        if set(dataset.dimensions) != {"Time", "nCells"}:
            raise RuntimeError("diagnostic capsule dimensions changed")
        if (
            len(dataset.dimensions["Time"]) != 2
            or dataset.dimensions["Time"].isunlimited()
            or len(dataset.dimensions["nCells"]) != DIAGNOSTIC_NCELLS
            or dataset.dimensions["nCells"].isunlimited()
        ):
            raise RuntimeError("diagnostic capsule fixed dimensions changed")
        if set(dataset.variables) != expected_variables:
            raise RuntimeError("diagnostic capsule variable inventory changed")
        required_attrs = {
            "diagnostic_capsule_schema": DIAGNOSTIC_CAPSULE_SCHEMA,
            "scope_label": DIAGNOSTIC_SCOPE_LABEL,
            "production_step_receipt_file_sha256": production_receipt_file_sha256,
            "production_step_receipt_payload_sha256": production_receipt_payload_sha256,
            "post_step_d2h_bytes": DIAGNOSTIC_D2H_BYTES,
        }
        for name, expected in required_attrs.items():
            if getattr(dataset, name, None) != expected:
                raise RuntimeError(f"diagnostic capsule attribute {name} changed")
        time = dataset.variables["time"]
        if (
            time.dimensions != ("Time",)
            or np.dtype(time.dtype) != np.dtype(np.float64)
            or getattr(time, "units", None) != "seconds since 2026-08-10 12:00:00"
            or not np.array_equal(
                np.asarray(time[:]), np.asarray((0.0, DT_SECONDS), dtype=np.float64)
            )
        ):
            raise RuntimeError("diagnostic capsule time axis changed")
        static_expectations = {
            "indexToCellID": context["cell_ids"],
            "latCell": context["latitude_radians"],
            "lonCell": context["longitude_radians"],
            "zz_lowest_model_level": context["zz_lowest_model_level"],
        }
        static_hashes: dict[str, str] = {}
        for name, expected in static_expectations.items():
            actual = np.ascontiguousarray(np.asarray(dataset.variables[name][:]))
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"diagnostic capsule changed {name}")
            static_hashes[name] = array_sha256(actual)
        record_variables = {
            "rho": "rho_zz_lowest_model_level",
            "rho_theta": "rho_theta_lowest_model_level",
            "qv": "qv_lowest_model_level",
        }
        record_hashes: dict[str, dict[str, str]] = {}
        for name, variable_name in record_variables.items():
            variable = dataset.variables[variable_name]
            if (
                variable.dimensions != ("Time", "nCells")
                or np.dtype(variable.dtype) != np.dtype(np.float32)
            ):
                raise RuntimeError(f"diagnostic capsule metadata changed for {name}")
            actual = np.ascontiguousarray(np.asarray(variable[:]))
            expected = np.stack((context["initial"][name], endpoint[name]), axis=0)
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"diagnostic capsule changed {name} records")
            record_hashes[name] = {
                "f000": array_sha256(actual[0]),
                "plus_120_seconds": array_sha256(actual[1]),
            }
    return {
        "data_model": "NETCDF4_CLASSIC",
        "dimensions": {"Time": 2, "nCells": DIAGNOSTIC_NCELLS},
        "variables": sorted(expected_variables),
        "valid_model_seconds": [0.0, DT_SECONDS],
        "static_array_sha256": static_hashes,
        "record_array_sha256": record_hashes,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def materialize_post_receipt_lml_diagnostic(
    *,
    context: Mapping[str, Any],
    diagnostic_root: Path,
    production_output_root: Path,
    production_receipt: Path,
    production_payload: Mapping[str, Any],
    paths: Mapping[str, Path],
    authorities_before: Mapping[str, Any],
    sources_before: Mapping[str, Any],
) -> Path:
    """Publish the explicitly separate post-receipt three-plane capsule."""

    tree_before = require_exact_production_receipt_tree(
        production_output_root, production_receipt
    )
    production_file_sha = sha256_file(production_receipt)
    production_payload_sha = str(production_payload["receipt_payload_sha256"])
    payload_without_digest = dict(production_payload)
    payload_without_digest.pop("receipt_payload_sha256")
    if canonical_json_sha256(payload_without_digest) != production_payload_sha:
        raise RuntimeError("production receipt payload SHA is false before diagnostic")
    if (
        production_payload.get("authority_claim") is not False
        or production_payload.get("engineering_dry") is not True
        or production_payload.get("physics_executed") is not False
        or production_payload.get("plotting_executed") is not False
        or production_payload["execution"]["step_receipt"]["d2h"]["bytes"] != 4
    ):
        raise RuntimeError("production receipt scope changed before diagnostic")

    transfer = download_post_step_lml_diagnostic(
        context["cp"], context["device_state"]
    )
    tree_after_transfer = require_exact_production_receipt_tree(
        production_output_root, production_receipt
    )
    if tree_after_transfer != tree_before:
        raise RuntimeError("production receipt tree changed during diagnostic D2H")

    capsule = diagnostic_root / DIAGNOSTIC_CAPSULE_NAME
    _write_lml_diagnostic_capsule(
        capsule,
        context=context,
        endpoint=transfer["host"],
        production_receipt_file_sha256=production_file_sha,
        production_receipt_payload_sha256=production_payload_sha,
    )
    capsule_validation = _validate_lml_diagnostic_capsule(
        capsule,
        context=context,
        endpoint=transfer["host"],
        production_receipt_file_sha256=production_file_sha,
        production_receipt_payload_sha256=production_payload_sha,
    )
    if source_snapshot() != sources_before:
        raise RuntimeError("source bytes changed during post-receipt diagnostic")
    if verify_authorities(paths) != authorities_before:
        raise RuntimeError("authority bytes changed during post-receipt diagnostic")
    if tuple(diagnostic_root.iterdir()) != (capsule,):
        raise RuntimeError("diagnostic root changed before receipt publication")

    payload: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "passed",
        "scope_label": DIAGNOSTIC_SCOPE_LABEL,
        "authority_claim": False,
        "engineering_dry": True,
        "full_forecast": False,
        "physics_executed": False,
        "plotting_executed": False,
        "production_step_receipt": {
            "path": str(production_receipt.resolve(strict=True)),
            "file_sha256": production_file_sha,
            "payload_sha256": production_payload_sha,
            "internal_d2h_bytes": 4,
            "tree_before_and_after_diagnostic": tree_after_transfer,
        },
        "post_step_diagnostic_transfer": transfer["receipt"],
        "capsule": {
            "path": str(capsule.resolve(strict=True)),
            **capsule_validation,
        },
        "source_and_authority_bytes_unchanged": True,
        "thermodynamic_materialization_deferred": (
            "the capsule contains only F000/+120s rho_zz[0], rho_theta[0], "
            "qv[0], host zz[0], and mesh identity/coordinates; temperature "
            "derivation, IDW4 visualization interpolation, and native Rust "
            "rendering occur in a separately receipted lane"
        ),
        "nonclaims": [
            "not native or compiled-MPAS numerical authority",
            "not full physics or a forecast beyond the one dry 120-second step",
            "qv is passive and its exact negative source defect is not cleaned",
            "not a conservative remap, forecast-skill result, or publication product",
            "this capsule receipt performs no interpolation or plotting",
        ],
    }
    payload["receipt_payload_sha256"] = canonical_json_sha256(payload)
    receipt = diagnostic_root / DIAGNOSTIC_RECEIPT_NAME
    _write_exclusive_json(receipt, payload)
    if set(diagnostic_root.iterdir()) != {capsule, receipt}:
        raise RuntimeError("diagnostic output root is not the exact two-file capsule")
    if source_snapshot() != sources_before or verify_authorities(paths) != authorities_before:
        raise RuntimeError("closed bytes changed during diagnostic receipt publication")
    if require_exact_production_receipt_tree(
        production_output_root, production_receipt
    ) != tree_before:
        raise RuntimeError("production receipt tree changed after diagnostic publication")
    return receipt


def engineering_config() -> Any:
    from mpas_port.config_v841 import V841DryDycoreConfig

    config = V841DryDycoreConfig(
        config_dt=DT_SECONDS,
        config_time_integration_order=3,
        config_number_of_sub_steps=6,
        config_dynamics_split_steps=3,
        config_apply_lbcs=False,
        config_split_dynamics_transport=True,
        config_scalar_advection=True,
        config_monotonic=False,
        config_positive_definite=False,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        config_coef_3rd_order=0.25,
        config_apvm_upwinding=0.5,
        config_epssm=0.0,
        config_epssm_minimum=0.1,
        config_epssm_maximum=0.5,
        config_epssm_transition_bottom_z=30_000.0,
        config_epssm_transition_top_z=50_000.0,
        config_moist_physics=False,
        config_physics_suite="none",
        config_iau_option="off",
        config_divergence_damping=False,
        config_horiz_mixing="off",
        config_len_disp=0.0,
        config_visc4_2dsmag=0.0,
        config_smagorinsky_coef=0.0,
        config_mpas_cam_coef=0.0,
        config_h_theta_eddy_visc2=0.0,
        config_v_theta_eddy_visc2=0.0,
        config_h_mom_eddy_visc2=0.0,
        config_v_mom_eddy_visc2=0.0,
        config_h_theta_eddy_visc4=0.0,
        config_h_mom_eddy_visc4=0.0,
        config_smdiv=0.0,
        config_xnutr=0.2,
        config_zd=22_000.0,
        config_vertical_mixing=False,
        config_rayleigh_damp_u=False,
        config_curvature_terms=False,
        config_terrain_following=True,
        config_gpu_aware_mpi=False,
        config_les_model="none",
        config_les_surface="none",
        config_mix_scalars=False,
        config_surface_heat_flux=0.0,
        config_surface_moisture_flux=0.0,
        config_surface_drag_coefficient=0.0,
    )
    config.validate()
    return config


def attach_inactive_zero_defc(mesh: Any, config: Any) -> dict[str, Any]:
    mixing = getattr(config, "config_horiz_mixing", None)
    if type(mixing) is not str or mixing != "off":
        raise ValueError(
            "zero defc_a/defc_b may only be generated for exact "
            "config_horiz_mixing='off'"
        )
    arrays = getattr(mesh, "arrays", None)
    if not isinstance(arrays, dict):
        raise TypeError("mesh must expose a mutable arrays dictionary")
    if "defc_a" in arrays or "defc_b" in arrays:
        raise ValueError("closed x4 mesh unexpectedly already contains defc_a/defc_b")
    first = np.zeros(ZERO_DEFC_SHAPE, dtype=np.float32, order="C")
    second = np.zeros(ZERO_DEFC_SHAPE, dtype=np.float32, order="C")
    for name, value in (("defc_a", first), ("defc_b", second)):
        if (
            value.shape != ZERO_DEFC_SHAPE
            or value.dtype != np.dtype(np.float32)
            or not value.flags.c_contiguous
            or value.nbytes != ZERO_DEFC_BYTES
            or array_sha256(value) != ZERO_DEFC_ARRAY_SHA256
        ):
            raise RuntimeError(f"generated inactive {name} changed its exact bytes")
        arrays[name] = value
    return {
        "policy": "generated only because config_horiz_mixing is exact 'off'",
        "active_mixing_claim": False,
        "shape": list(ZERO_DEFC_SHAPE),
        "dtype": "float32",
        "c_contiguous": True,
        "bytes_each": ZERO_DEFC_BYTES,
        "array_sha256_each": ZERO_DEFC_ARRAY_SHA256,
        "fields": ["defc_a", "defc_b"],
    }


def gpu_memory_admission(cp: Any) -> dict[str, Any]:
    free_bytes, total_bytes = (
        int(value) for value in cp.cuda.runtime.memGetInfo()
    )
    if free_bytes < MIN_FREE_DEVICE_BYTES:
        raise MemoryError(
            "CUDA upload refused: free memory "
            f"{free_bytes} < required {MIN_FREE_DEVICE_BYTES} bytes"
        )
    return {
        "measured_immediately_before_upload": True,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "required_free_bytes": MIN_FREE_DEVICE_BYTES,
        "admitted": True,
    }


def _finite_bounds(name: str, value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
        raise TypeError(f"{name} must be C-contiguous float32")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return {
        "dtype": "float32",
        "shape": [int(extent) for extent in array.shape],
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "max_abs": float(np.max(np.abs(array))),
        "array_sha256": array_sha256(array),
    }


def host_state_bounds(state: Any) -> dict[str, Any]:
    names = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
    bounds = {name: _finite_bounds(name, getattr(state, name)) for name in names}
    if bounds["rho"]["min"] <= 0.0 or bounds["rho_theta"]["min"] <= 0.0:
        raise FloatingPointError("rho and rho_theta must be strictly positive")
    if tuple(np.asarray(state.scalars).shape) != (1, 55, 163_842):
        raise ValueError("real-x4 execution requires exactly one qv scalar carrier")
    return {
        "fields": bounds,
        "rho_positive": True,
        "rho_theta_positive": True,
        "one_passive_qv_carrier": True,
    }


def prepared_qv_fingerprint(state: Any) -> dict[str, Any]:
    scalar = np.ascontiguousarray(np.asarray(state.scalars[0]))
    negative = negative_qv_fingerprint(scalar)
    if negative["negative_count"] != NEGATIVE_QV_PIN["negative_count"]:
        raise RuntimeError("prepared passive-qv negative count changed")
    if (
        negative["negative_indices_sha256"]
        != NEGATIVE_QV_PIN["negative_indices_sha256"]
    ):
        raise RuntimeError("prepared passive-qv negative locations changed")
    return {
        "prepared_carrier": negative,
        "source_qv": dict(NEGATIVE_QV_PIN),
        "location_preservation_exact": True,
        "nonclaim": NEGATIVE_QV_NONCLAIM,
    }


def require_exact_compile_relation(relation: Mapping[str, Any]) -> dict[str, Any]:
    translation_units = relation.get("translation_units")
    if not isinstance(translation_units, Mapping):
        raise RuntimeError("compile relation has no translation-unit mapping")
    if tuple(sorted(translation_units)) != EXPECTED_TRANSLATION_UNITS:
        raise RuntimeError("compile relation is not the exact eight-TU inventory")
    if relation.get("reached_kernel_count") != EXPECTED_REACHED_KERNEL_COUNT:
        raise RuntimeError("compile relation did not reach exactly 46 kernels")
    if relation.get("compiled_kernel_count") != EXPECTED_COMPILED_KERNEL_COUNT:
        raise RuntimeError("compile relation did not retain the 95-kernel surface")
    if relation.get("source_release") != "v8.4.1":
        raise RuntimeError("compile relation source release changed")
    if relation.get("authority_claim") is not False:
        raise RuntimeError("engineering compile relation acquired a false authority claim")
    return dict(relation)


def compile_exact_v841_manifest(cache: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from mpas_port.cuda_ftz import (
        V841_REACHED_TRANSLATION_UNITS,
        validate_v841_compile_manifest_relation,
        v841_reached_translation_units,
    )

    if tuple(V841_REACHED_TRANSLATION_UNITS) != EXPECTED_TRANSLATION_UNITS:
        raise RuntimeError("live v8.4.1 translation-unit inventory changed")
    inventory = v841_reached_translation_units()
    if tuple(sorted(inventory)) != EXPECTED_TRANSLATION_UNITS:
        raise RuntimeError("live v8.4.1 reached inventory changed")
    for module_key in EXPECTED_TRANSLATION_UNITS:
        source, names = inventory[module_key]
        cache.raw_kernels(names, source, module_key=module_key)
    manifest = cache.compile_manifest()
    relation = require_exact_compile_relation(
        validate_v841_compile_manifest_relation(manifest)
    )
    return manifest, relation


def require_step_receipt(
    receipt: Any,
    *,
    expected_configuration: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    expected_start_time: float,
    validate_manifest: Any,
) -> dict[str, Any]:
    if receipt.source_release != "v8.4.1":
        raise RuntimeError("step receipt source release changed")
    if int(receipt.d2h.bytes) != 4:
        raise RuntimeError(f"step receipt D2H changed: {receipt.d2h.bytes} != 4")
    if dict(receipt.configuration) != dict(expected_configuration):
        raise RuntimeError("step receipt configuration changed")
    configuration_sha = canonical_json_sha256(dict(expected_configuration))
    if receipt.configuration_sha256 != configuration_sha:
        raise RuntimeError("step receipt configuration SHA is false")
    if tuple(receipt.stage_acoustic_steps) != EXPECTED_STAGE_ACOUSTIC_STEPS:
        raise RuntimeError("step acoustic schedule changed")
    if int(receipt.dynamics_split_steps) != 3:
        raise RuntimeError("step dynamics split count changed")
    scalar_timesteps = receipt.scalar_transport_stage_timesteps
    if scalar_timesteps is None or tuple(scalar_timesteps) != (
        EXPECTED_SCALAR_STAGE_TIMESTEPS
    ):
        raise RuntimeError("step scalar transport schedule changed")
    if float(receipt.dynamics_timestep_seconds) != (
        EXPECTED_DYNAMICS_TIMESTEP_SECONDS
    ):
        raise RuntimeError("step dynamics timestep changed")
    if tuple(receipt.dynamics_stage_timesteps) != EXPECTED_DYNAMICS_STAGE_TIMESTEPS:
        raise RuntimeError("step dynamics RK stage schedule changed")
    if receipt.split_flux_reduction != EXPECTED_SPLIT_FLUX_REDUCTION:
        raise RuntimeError("step split-flux reduction contract changed")
    if receipt.t0_diagnostics_source != EXPECTED_T0_DIAGNOSTICS_SOURCE:
        raise RuntimeError("step t0 diagnostics source changed")
    if tuple(receipt.authority_nonclaims) != EXPECTED_DRIVER_AUTHORITY_NONCLAIMS:
        raise RuntimeError("step authority nonclaims changed")
    if receipt.authority_ruler is not None or receipt.authority_ruler_sha256 is not None:
        raise RuntimeError("engineering step acquired a false numerical authority ruler")
    if float(receipt.start_time_seconds) != float(expected_start_time):
        raise RuntimeError("step start time changed")
    if float(receipt.end_time_seconds) != float(expected_start_time + DT_SECONDS):
        raise RuntimeError("step end time changed")
    if dict(receipt.compile_manifest) != dict(expected_manifest):
        raise RuntimeError("step compile manifest changed after precompilation")
    manifest_sha = canonical_json_sha256(dict(expected_manifest))
    if receipt.compile_manifest_sha256 != manifest_sha:
        raise RuntimeError("step compile manifest SHA is false")
    relation = require_exact_compile_relation(validate_manifest(expected_manifest))
    if relation.get("compile_manifest_sha256") != manifest_sha:
        raise RuntimeError("compile relation manifest SHA is false")
    return {
        "d2h_exact_bytes": 4,
        "finite_state_and_saved_diagnostics": True,
        "positive_rho_and_rho_theta": True,
        "mechanism": (
            "resident validation kernels accumulate one int32 invalid flag; "
            "only that flag is copied to the host"
        ),
        "quantitative_post_step_extrema": (
            "not downloaded; exact four-byte D2H is preserved"
        ),
        "compile_relation": relation,
    }


def _run_one_step(
    *,
    paths: Mapping[str, Path],
    authorities_before: Mapping[str, Any],
    sources_before: Mapping[str, Any],
    cache_root: Path,
    retain_lml_diagnostic: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Prepare, upload, and run the sole admitted step."""

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from mpas_port.cuda_backend import KernelCache, require_cuda
    from mpas_port.cuda_driver import (
        CudaDryDycoreDriver,
        cuda_configuration_payload,
    )
    from mpas_port.cuda_dualrun import (
        PreparedCudaInputs,
        fingerprint_uploaded_execution,
    )
    from mpas_port.cuda_ftz import validate_v841_compile_manifest_relation
    from mpas_port.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from mpas_port.dynamics_v841 import load_v841_reference_wind_profiles
    from mpas_port.mesh import load_precision_preserving_mesh_pair

    if source_snapshot() != sources_before:
        raise RuntimeError("source bytes changed before host preparation")
    if verify_authorities(paths) != authorities_before:
        raise RuntimeError("authority bytes changed before host preparation")

    config = engineering_config()
    mesh, output_mesh, mesh_evidence = load_precision_preserving_mesh_pair(
        paths["grid"], paths["static"]
    )
    mesh_reconciliation_receipt = require_exact_grid_rotate_reconciliation(
        mesh_evidence["longitude_reconciliation"]
    )
    del output_mesh
    defc_receipt = attach_inactive_zero_defc(mesh, config)
    native = load_mpas_vertical_grid(
        paths["init"],
        mesh,
        config_coef_3rd_order=config.config_coef_3rd_order,
    )
    state, reference, saved = load_mpas_initial_state(
        paths["init"],
        mesh,
        native.vertical_grid,
        scalar_names=("qv",),
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
    )
    profiles = load_v841_reference_wind_profiles(
        paths["init"], n_vert_levels=55
    )
    state.validate(n_cells=163_842, n_edges=491_520, n_vert_levels=55)
    reference.validate((55, 163_842))
    saved.validate((55, 163_842), np.dtype(np.float32), 491_520)
    native.terrain_metrics.validate(nlev=55, ncells=163_842, max_edges=10)
    bounds = host_state_bounds(state)
    qv_receipt = prepared_qv_fingerprint(state)
    diagnostic_context = (
        prepare_lml_diagnostic_context(mesh, state, native.vertical_grid)
        if retain_lml_diagnostic
        else None
    )

    input_bytes = {
        role: dict(record)
        for role, record in authorities_before["files"].items()
    }
    prepared = PreparedCudaInputs.validated(
        config=config,
        profile=PROFILE,
        target=TARGET,
        preparation_method=PREPARATION_METHOD,
        mesh=mesh,
        state=state,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
        terrain_metrics=native.terrain_metrics,
        input_bytes=input_bytes,
        reference_wind_profiles=profiles,
    )
    prepared_fingerprint = prepared.expected_execution_fingerprint
    if not isinstance(prepared_fingerprint, Mapping):
        raise RuntimeError("prepared execution did not seal a complete fingerprint")

    capability = require_cuda(
        min_compute=(12, 0),
        required_compute=(12, 0),
        cache_dir=cache_root,
    )
    import cupy as cp

    cache = KernelCache(capability=capability, cache_dir=cache_root)
    compile_manifest, compile_relation = compile_exact_v841_manifest(cache)
    memory = gpu_memory_admission(cp)
    driver = CudaDryDycoreDriver.from_host(
        prepared.mesh,
        prepared.state,
        prepared.vertical,
        prepared.reference,
        config,
        saved_diagnostics=prepared.saved_diagnostics,
        terrain_metrics=prepared.terrain_metrics,
        advection_coefficients=prepared.advection_coefficients,
        kernel_cache=cache,
        reference_wind_profiles=prepared.reference_wind_profiles,
    )
    uploaded_fingerprint = fingerprint_uploaded_execution(driver)
    if uploaded_fingerprint != prepared_fingerprint:
        raise RuntimeError("complete uploaded execution differs from host preparation")

    expected_configuration = cuda_configuration_payload(config)
    start_time = float(driver.atmosphere.state.time_seconds)
    result = driver.step_device()
    result.atmosphere.validate()
    post_step = require_step_receipt(
        result.receipt,
        expected_configuration=expected_configuration,
        expected_manifest=compile_manifest,
        expected_start_time=start_time,
        validate_manifest=validate_v841_compile_manifest_relation,
    )
    if float(result.atmosphere.state.time_seconds) != start_time + DT_SECONDS:
        raise RuntimeError("resident result ended at the wrong model time")
    if cache.compile_manifest() != compile_manifest:
        raise RuntimeError("step changed the exact precompiled manifest")

    if diagnostic_context is not None:
        diagnostic_context["cp"] = cp
        diagnostic_context["device_state"] = result.atmosphere.state

    execution = {
        "configuration": asdict(config),
        "configuration_payload": expected_configuration,
        "capability": capability.as_dict(),
        "memory_admission": memory,
        "mesh_overlay": mesh_evidence,
        "grid_rotate_longitude_reconciliation": mesh_reconciliation_receipt,
        "generated_inactive_defc": defc_receipt,
        "host_state_bounds": bounds,
        "passive_qv": qv_receipt,
        "prepared_execution_fingerprint": dict(prepared_fingerprint),
        "uploaded_execution_fingerprint": uploaded_fingerprint,
        "prepared_uploaded_exact": True,
        "upload_fingerprint_audit_transfer": (
            "fingerprint_uploaded_execution explicitly downloads the complete "
            "pre-step upload boundary for equality; it is audit traffic outside "
            "the resident step receipt, whose D2H remains exactly four bytes"
        ),
        "compile_manifest": compile_manifest,
        "compile_relation": compile_relation,
        "step_receipt": asdict(result.receipt),
        "post_step_resident_validation": post_step,
        "steps_executed": STEP_COUNT,
    }
    return execution, diagnostic_context


def _paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        role: Path(getattr(args, role)).expanduser().absolute()
        for role in AUTHORITY_PINS
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = default_authority_paths()
    for role in AUTHORITY_PINS:
        parser.add_argument(
            "--" + role.replace("_", "-"),
            type=Path,
            default=defaults[role],
        )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        help=(
            "fresh sibling root for an optional post-receipt F000/+120s "
            "lowest-model-level diagnostic capsule"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify closed bytes/metadata/source scope without importing CUDA",
    )
    args = parser.parse_args(argv)
    if not args.preflight_only and (args.cache_root is None or args.output is None):
        parser.error("one-step mode requires --cache-root and --output")
    if args.preflight_only and args.diagnostic_output is not None:
        parser.error("--diagnostic-output is incompatible with --preflight-only")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _paths_from_args(args)

    destinations = None
    diagnostic_root = None
    if not args.preflight_only:
        assert args.cache_root is not None and args.output is not None
        destinations = validate_destination_paths(
            args.cache_root,
            args.output,
            protected_inputs=tuple(paths.values()),
        )
        if args.diagnostic_output is not None:
            diagnostic_root = validate_diagnostic_destination_path(
                args.diagnostic_output,
                cache_root=destinations[0],
                output_root=destinations[1],
                protected_inputs=tuple(paths.values()),
            )

    sources_before = source_snapshot()
    authorities_before = verify_authorities(paths)
    if args.preflight_only:
        sources_after = source_snapshot()
        authorities_after = verify_authorities(paths)
        if sources_after != sources_before or authorities_after != authorities_before:
            raise RuntimeError("preflight inputs or sources changed during verification")
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "mode": "preflight-only; CUDA not imported",
                    "authorities_sha256": authorities_before["sha256"],
                    "sources_sha256": sources_before["files_sha256"],
                    "negative_qv": NEGATIVE_QV_PIN,
                    "claim": CLAIM,
                    "nonclaims": list(NONCLAIMS),
                },
                sort_keys=True,
            )
        )
        return 0

    assert destinations is not None
    cache_root, output_root = destinations
    cache_root.mkdir()
    output_root.mkdir()
    if diagnostic_root is not None:
        diagnostic_root.mkdir()
    execution, diagnostic_context = _run_one_step(
        paths=paths,
        authorities_before=authorities_before,
        sources_before=sources_before,
        cache_root=cache_root,
        retain_lml_diagnostic=diagnostic_root is not None,
    )
    sources_after = source_snapshot()
    authorities_after = verify_authorities(paths)
    if sources_after != sources_before:
        raise RuntimeError("source bytes changed during the one-step execution")
    if authorities_after != authorities_before:
        raise RuntimeError("authority bytes changed during the one-step execution")

    payload = {
        "schema": SCHEMA,
        "claim": CLAIM,
        "authority_claim": False,
        "engineering_dry": True,
        "full_forecast": False,
        "physics_executed": False,
        "plotting_executed": False,
        "nonclaims": list(NONCLAIMS),
        "inputs_pre": authorities_before,
        "inputs_post": authorities_after,
        "sources_pre": sources_before,
        "sources_post": sources_after,
        "input_and_source_bytes_unchanged": True,
        "execution": execution,
    }
    payload["receipt_payload_sha256"] = canonical_json_sha256(payload)
    receipt = output_root / RECEIPT_NAME
    _write_exclusive_json(receipt, payload)

    if source_snapshot() != sources_before:
        raise RuntimeError("source bytes changed during exclusive receipt publication")
    if verify_authorities(paths) != authorities_before:
        raise RuntimeError("authority bytes changed during receipt publication")
    production_tree = require_exact_production_receipt_tree(output_root, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "receipt_file_sha256": sha256_file(receipt),
                "receipt_payload_sha256": payload["receipt_payload_sha256"],
                "production_tree": production_tree,
            },
            sort_keys=True,
        )
    )
    if diagnostic_root is not None:
        if diagnostic_context is None:
            raise RuntimeError("requested diagnostic context was not retained")
        diagnostic_receipt = materialize_post_receipt_lml_diagnostic(
            context=diagnostic_context,
            diagnostic_root=diagnostic_root,
            production_output_root=output_root,
            production_receipt=receipt,
            production_payload=payload,
            paths=paths,
            authorities_before=authorities_before,
            sources_before=sources_before,
        )
        print(
            json.dumps(
                {
                    "diagnostic_receipt": str(diagnostic_receipt),
                    "diagnostic_receipt_file_sha256": sha256_file(
                        diagnostic_receipt
                    ),
                    "diagnostic_receipt_payload_sha256": json.loads(
                        diagnostic_receipt.read_text(encoding="utf-8")
                    )["receipt_payload_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
