#!/usr/bin/env python3
"""Run and prove the pinned real-x4 MPAS-A v8.4.1 CUDA physics chain.

The production path is deliberately closed until every moving source below has
an exact frozen SHA-256.  Once released, one invocation performs:

* one uninterrupted 30 x 120 s full-physics CUDA forecast;
* F000/F030/F001 boundary snapshots and quantitative comparison with the
  sealed native CPU GF+YSU-GWDO authority;
* one checkpoint/restart continuation from F030 through F001; and
* an exact-byte comparison of uninterrupted and restored committed F001 state,
  saved diagnostics, Arwen persistence, surface/soil state, precipitation, and
  the six external YSU-GWDO diagnostics.

The finalized fa35 Arwen contract is the execution target.  Its documented GF
deviation is retained honestly: MPAS computes endpoint ``rthdynten/rqvdynten``,
but fa35 has no API lane for them and supplies zero GF advective forcing.  This
tool therefore never claims bitwise/source-native GF parity.  It does require
real weather evolution and records quantitative native comparisons.

CUDA and production imports are lazy.  Closed authority bytes, source pins,
constructor mappings, output scope, and the staged transaction API are checked
before a CUDA device is probed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST_FILE = ROOT / "tests" / "test_cuda_v841_full_physics_x4.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SCHEMA = "mpas-port.cuda-v841-real-x4-full-physics/v2"
SNAPSHOT_SCHEMA = "mpas-port.cuda-v841-real-x4-full-physics-snapshot/v2"
CHECKPOINT_SCHEMA = "mpas-port.cuda-v841-real-x4-full-physics-checkpoint/v3"
RESTART_WORKER_SCHEMA = "mpas-port.cuda-v841-real-x4-full-physics-restart-worker/v1"
PROFILE = "real-x4.163842-v8.4.1-full-physics-gf-ysu-gwdo"
SOURCE_RELEASE = "v8.4.1"
ARWEN_COMMIT = "0d04db71298d010a61fee3267c07277da3b8b64f"
ARWEN_CONTRACT_DOCUMENT_SHA256 = (
    "5c629e23be2af20c0b1660d262443c415256126b812493f6681590bf07aff92a"
)
ARWEN_CONTRACT_SURFACE_SHA256 = (
    "823af4a55018a71ad630144fae7b21a459095249cedb1180bc9f3e1a2fbfe511"
)
ARWEN_GLACIER_COMPOSED_TU_SHA256 = (
    "edafcac585d4786c0cdfddf07f8e767b64d0d40b6db0e4da3dc3b2fa8c21fb59"
)
ARWEN_XICE_THRESHOLD = np.float32(0.02)
ARWEN_GLACIER_CUDA_PROVENANCE = (
    "noahmp-glacier/cuda (gpuwm/core/kernels/noahmp_glacier.cu)"
)

N_CELLS = 163_842
N_EDGES = 491_520
N_LEVELS = 55
N_INTERFACES = 56
N_SOIL_LEVELS = 4
SCALAR_NAMES = ("qv", "qc", "qr", "qi", "qs", "qg")
SOURCE_SCALAR_NAMES = ("qv", "qc", "qr")
COLD_ZERO_SCALAR_NAMES = ("qi", "qs", "qg")
DT_SECONDS = 120.0
FULL_STEPS = 30
CHECKPOINT_STEP = 15
SNAPSHOT_STEPS = (0, CHECKPOINT_STEP, FULL_STEPS)
SNAPSHOT_LABELS = {0: "F000", CHECKPOINT_STEP: "F030", FULL_STEPS: "F001"}
START_TIME_TEXT = "2026-08-10_12:00:00"
NOMINAL_DX_M = np.float32(25_000.0)
EXPECTED_ARWEN_P_TOP_PA_F32 = np.float32(1_159.38818359375)
EXPECTED_TOP_PRESSURE_RANGE_PA = (592.24884, 1_233.00952, 1_342.08362)
MIN_FREE_DEVICE_BYTES = 24 * 1024**3
# The restart worker executes one device stack plus the 15-step continuation
# and one F001 capture (measured whole-process peak 23.7 GB for the larger
# uninterrupted arm), while the spawning proof process necessarily retains
# its CUDA context and module-resident tables.  Its admission floor is sized
# for that strictly smaller footprint; every numeric gate is unchanged.
RESTART_WORKER_MIN_FREE_DEVICE_BYTES = 22 * 1024**3

EXPECTED_SURFACE_CLASSIFICATION = MappingProxyType(
    {
        "xland_source": "native",
        "xland_land_columns": 61_528,
        "xland_water_columns": 102_314,
        "xice_threshold": float(ARWEN_XICE_THRESHOLD),
        "sea_ice_columns": 4_643,
        "open_water_columns": 102_314,
        "sflx_land_columns": 53_875,
        "glacier_columns": 3_010,
    }
)
EXPECTED_NOAHMP_CENSUS = MappingProxyType(
    {
        "land": 53_875,
        "water": 102_314,
        "sea_ice": 4_643,
        "glacier": 3_010,
        "glacier_path": ARWEN_GLACIER_CUDA_PROVENANCE,
    }
)
NATIVE_XLAND_SOURCE_SHA256 = (
    "e7d081437e55646d65fd635fee48fb9b8702a25c36a6950c806809a2187748bc"
)
NATIVE_XLAND_FLAT_SHA256 = (
    "2009116b7b34d03439d47980ad6d2977583c45189bd9cdbbad3c678576b36805"
)
GLACIER_INDEX_SHA256 = (
    "1ffe16d231cca7885ad72ae438479e82f59a253e36f73937f17bd45eb5a5f4fa"
)
SEA_ICE_INDEX_SHA256 = (
    "1d8b4c34c4c2797c6c578181fd2365ee195c8a5bfbe6b5e6e1c36c2d7740182d"
)
THRESHOLD_DELTA_INDEX_SHA256 = (
    "df9a33ed4b791778e6c99b9979fe7a1ce1a447333faf53e4f2f94545a2a0c2ea"
)


F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS = MappingProxyType(
    {
        "t2": {
            "source": "t2m",
            "sha256": "1fcebcd0117a1a069e2ddc67f14e2fe2a47b030d4bf884aed2c83c996690eb3f",
        },
        "u10": {
            "source": "u10",
            "sha256": "0fb96429b062ef7d1114f26b19443c2f3c3b82d50b6c07003304c357b9cdc9da",
        },
        "v10": {
            "source": "v10",
            "sha256": "131b6b1e56a761b8c68d5abf72d61b71302c170f38c4a488acc21465f9be0960",
        },
    }
)

RECEIPT_NAME = "cuda-v841-full-physics-x4-receipt.json"
BASELINE_DIAGNOSTIC_SCHEMA = (
    "mpas-port.cuda-v841-real-x4-full-physics-baseline-diagnostic/v1"
)
BASELINE_DIAGNOSTIC_RECEIPT_NAME = (
    "cuda-v841-full-physics-x4-baseline-diagnostic-receipt.json"
)
BASELINE_DIAGNOSTIC_STATUS = (
    "uninterrupted_baseline_passed_restart_not_evaluated_non_release"
)
BASELINE_DIAGNOSTIC_WARNING = (
    "ENGINEERING BASELINE; restart proof pending; NOT RELEASE"
)
SNAPSHOT_FILE_NAMES = {
    "F000": "cuda-history.2026-08-10_12.00.00.nc",
    "F030": "cuda-history.2026-08-10_12.30.00.nc",
    "F001": "cuda-history.2026-08-10_13.00.00.nc",
    "F001_RESTART": "cuda-history-restart.2026-08-10_13.00.00.nc",
}

CLAIM = (
    "one real initialized x4.163842 MPAS-A v8.4.1 CUDA forecast through "
    "one hour using WSM6 + GF + YSU + external YSU-GWDO + revised-MO + "
    "NoahMP + cloud fraction + legacy RRTMG, with an F030 restart arm"
)
NONCLAIMS = (
    "not a bitwise or source-native MPAS full-physics reproduction",
    "fa35 GF supplies zero rthften/rqvften because its finalized public seam "
    "does not accept MPAS endpoint rthdynten/rqvdynten",
    "fa35 phase one shares moist-hydrostatic pressure among its consumers",
    "fa35 legacy RRTMG uses nominal one-dimensional interface weights and "
    "performs a host round trip on radiation-due calls",
    "fa35 accepts one scalar p_top_pa, so the exact F000 per-column native "
    "pres2_p top interface is reduced to its areaCell-weighted mean; native "
    "MPAS phase one retains per-column plrad/top-interface pressure",
    "q2 is retained only as audit data; it is not required nonnegative and "
    "must not be rendered or published",
    "does not establish forecast skill",
)

# Every execution boundary is exact-byte pinned. A future None value remains
# a hard pre-CUDA refusal, never a wildcard, so partially released edits cannot
# silently enter this proof.
EXECUTION_SOURCE_PINS: dict[str, str | None] = {
    "src/mpas_port/cuda_physics_prep_v841.py": (
        "29fb9bb7c6f37f90e1f66fabd576810fa89db902ad7e4495eaf21a57610cbccf"
    ),
    "src/mpas_port/cuda_gwdo_v841.py": (
        "11e038bc2365964b6c8b8db36d3dd99ed200edc3f40e7795e208af3af08bd316"
    ),
    "src/mpas_port/cuda_physics_v841.py": (
        "ea6afd713883530e317936f93285b4d4ffe22c2fecf25d76f3f1b6af4041529f"
    ),
    # Re-frozen during the 12 GiB capacity work: the dynamics subcycle stopped
    # copying a scalar block nothing writes, stopped taking a private image of
    # the substep-start state for RK stage 1, and stopped asking recover_state
    # for six cell diagnostics it discards.  Every affected proof re-runs
    # against this digest.
    "src/mpas_port/cuda_driver.py": (
        "9daf917a89b3b9dd6f013be3d971c76d255bcfbbb9c1027b9de0c8823cb49e66"
    ),
    # THE BREAKAGE THIS PREVENTS: the same lane that re-froze cuda_driver.py
    # also changed ``recover_state`` -- it gained ``include_pressure`` and
    # ``to_host`` gained a refusal -- and the RK stage the driver runs executes
    # this module every substep.  Unpinned, the frozen-source proof would have
    # certified a driver digest while the recovery module underneath it moved
    # freely, so a later edit to the pressure-diagnostic path could change
    # forecast bytes with every proof still reporting frozen sources.
    "src/mpas_port/cuda_backend/recovery.py": (
        "40635e20e4de9f1cf49c2590dcc14f262fa03667dd4b547f0dd61fb47892dac3"
    ),
    "src/mpas_port/config_v841.py": (
        "2bc878868e41ffc71491479059d3bd9165ce980a38360ed683e2717f54a8111a"
    ),
    "src/mpas_port/cuda_arwen_physics_v841.py": (
        "20c4b22dcd36fa165d15642e45e3fac5cbe7b8de01dbffdfd0a84361c222d13b"
    ),
    # The v8.4.1 horizontal-mixing execution boundary (2-D Smagorinsky):
    # CPU authorities and the CUDA operator modules the RK1 saved-Euler
    # mixing runs through.
    "src/mpas_port/mixing.py": (
        "864f0686325108100afc10a8804ea4e2dd6de81e3269ee4cbc2747be82b09e2e"
    ),
    # Re-frozen for the public-release scrub: the module and V841MixingConfig
    # docstrings dropped private host/path/run-tag strings.  AST-identical
    # with docstrings stripped, so the executed numerics are unchanged; the
    # digest moves because the pin is exact-byte by design.
    "src/mpas_port/mixing_v841.py": (
        "f82e9f5c64547b6763db37ada8ba79a966e9ef8f310cf84fc71375f8380e3a73"
    ),
    "src/mpas_port/cuda_horizontal.py": (
        "97faf0869a0a5ea9ebbc4c67b3c2d6c68cefdfa10dece73cd204d818962efde4"
    ),
    "src/mpas_port/cuda_horizontal_v841.py": (
        "3fc0b860ebd67dfed453617c348810964ea1110e782fe85db10283afb406e2fe"
    ),
    # The two-rank (multi-GPU) execution boundary: the scalar-transport module
    # (which gained the guarded FCT halo-exchange hook and executes in every
    # lane) and the partition modules the 2-GPU runner executes.  Pinned here
    # so a partition-module drift refuses BEFORE CUDA on every lane that
    # verifies frozen sources, exactly as the mixing boundary was pinned.
    "src/mpas_port/cuda_transport_v841.py": (
        "55c66759d9c81f65ed71ce77570897c102fd64661da6ad6c37b438b27771ab23"
    ),
    "src/mpas_port/partition_assets_v841.py": (
        "dc5f2cb3f7bdadeca28854a15644273f7a94cdb710a36df18f4f91bdba70450e"
    ),
    "src/mpas_port/partition_local_mesh_v841.py": (
        "609955b3db527528f1e2ffd949483099a8d19dd0bd23f724d7a711fbba08e150"
    ),
    "src/mpas_port/partition_state_v841.py": (
        "a504e6f5c5abc2014d40e4a8e3e89885f97d6456ff428c585fbaf57910eccbe8"
    ),
    "src/mpas_port/partition_device_scheduler_v841.py": (
        "fcf2f94fc368e71b6b87ddb5c1d3b1b68a4cfc2bccfcf1e50e57f0f2d3432276"
    ),
    "src/mpas_port/partition_executor_v841.py": (
        "6343fe0f89f39d81b3ef0d61343330c2ad09f59e00295ceedc090c3e4a61879c"
    ),
    "src/mpas_port/partition_net_v841.py": (
        "30b0988b2d40bbda8d68be1ec236564ea87bb6690bfdcd08370eea12ca11753b"
    ),
}

KNOWN_CONTRACT_PINS = MappingProxyType(
    {
        "prep_contract_sha256": (
            "32bcdfdfeb7b0dff9608bbbd7076b7b7c100f1089c8a1687fdfa352f0402158b"
        ),
        "prep_kernel_sha256": (
            "8cf7ca0deb5ebe77baf00889a1c0de6a579a26ccb9c52a9d2ecb27bf544fc7c5"
        ),
        "gwdo_contract_sha256": (
            "514241e42f6154c6e00bf53a2cd050a12079d0315a7de4d8d35189c213f5ba83"
        ),
        "gwdo_kernel_sha256": (
            "b334506b289c6f002a2660e6c7796361e39372f6652b8e60268c1400900ab9ec"
        ),
        "coupling_contract_sha256": (
            "63d9edb9ea4a12b78ccdeec64c2424de2ddbc10ff3a8c58361aa943f19c517db"
        ),
        "coupling_kernel_sha256": (
            "70d2006d4687b67fe087fd4a5c9e69a76e4a39c648703913f4d79903249bdcab"
        ),
        "adapter_contract_sha256": (
            "6c3ca3bae5f92a7ffa3f9cf27db0f2329ab0506bbf5d76e362f010676a0c78e1"
        ),
        "arwen_contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
        "arwen_glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
    }
)

# The native authority was REGENERATED on 2026-08-20 on the reference node.
# The sealed ``...-20260810a`` bundle was lost -- it is on no reachable
# machine -- so the pins below name the 2026-08-20 rerun instead of the
# 2026-08-10 seal.
#
# The rerun is the same case, not a new one: MPAS-A v8.4.1, the surviving
# ``ifx`` binary sha256 6d33081c69b7d728530278cc2b0b2b262c9f6961dc3c0422b44aefa450d7f332,
# the same ``x4.163842.init.nc`` the ``init`` pin already names
# (sha256 f6e6f413...), 24 MPI ranks, ``config_dt`` 120 s, 30 steps,
# ``config_start_time`` 2026-08-10_12:00:00, history every 30 min,
# ``config_gwdo_scheme = bl_ysu_gwdo``, ``config_convection_scheme =
# cu_grell_freitas``; rc=0 in 173.890 s wall.
#
# It is also bit-exact against the surviving 2026-08-12 run of the identical
# setup: all three history files differ from that run in exactly nine bytes,
# and those nine bytes are the ``file_id`` global attribute MPAS fills with a
# random ten-character string per output file.
#
# The three history pins are therefore MASKED-CONTENT digests: the sha256 of
# the whole file with exactly the ``file_id`` attribute value bytes (located
# via the netCDF header, never a hardcoded offset) hashed as NULs, plus the
# exact byte count.  A bit-exact rerun of the deterministic 24-rank case
# satisfies the pin; a single flipped data byte refuses.  The whole-file
# digest is still computed and recorded in every receipt as provenance of the
# exact kept artifact, but it no longer gates execution -- pinning the random
# nonce is what made the lost 20260810a authority irreplaceable.  The kept
# copies are listed in ``mpas-authority-20260820-MANIFEST.md``.
AUTHORITY_PINS: dict[str, dict[str, Any]] = {
    "grid": {
        "relative_path": (
            "work/v841-vr-static/run-static-v841-conus-official-full-a/"
            "x4.163842.grid.nc"
        ),
        "bytes": 224_139_172,
        "sha256": "48e747157bb1f0b83b96505e268699dfb562b4c1428468cb91457fbb03b1be55",
    },
    "static": {
        "relative_path": (
            "work/v841-vr-static/run-static-v841-conus-official-full-a/"
            "x4.163842.static.nc"
        ),
        "bytes": 298_860_376,
        "sha256": "f064ee8f8d40085db4bf77a3d5fc6081cd92368b7d3dd32d98110b8b64d177e8",
    },
    "init": {
        "relative_path": (
            "work/v841-vr-static/run-real-init-v841-conus-official-full-a/"
            "x4.163842.init.nc"
        ),
        "bytes": 1_489_665_020,
        "sha256": "f6e6f41359554ad3b1103235ec4aef026409b0f085a28a7b0f7c38599b9ca2ba",
    },
    "native_f000": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/history.2026-08-10_12.00.00.nc"
        ),
        "bytes": 1_584_808_024,
        "masked_sha256": (
            "38575bfcbbe581c25ceffeec25d22061b6f22cea2308f639e8fcce093d58da17"
        ),
    },
    "native_f030": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/history.2026-08-10_12.30.00.nc"
        ),
        "bytes": 1_584_808_024,
        "masked_sha256": (
            "1cf267557cf394f0209fbce6a69350e386221d4e791c2ceefc72200d3a45da47"
        ),
    },
    "native_f001": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/history.2026-08-10_13.00.00.nc"
        ),
        "bytes": 1_584_808_024,
        "masked_sha256": (
            "2b867a3352d7580280c01120b5db7fb4e6979be528317770417dff04b9f58b4c"
        ),
    },
    "native_validation_receipt": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/native-gwdo-authority-receipt.json"
        ),
        "bytes": 24_457,
        "sha256": "2fdc8a2e0e5f0b5432773713c6d48e6d3342e59ac2abcebbbf9614fbc500a9bf",
    },
    "native_launch_receipt": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/native-gwdo-launch-receipt.json"
        ),
        "bytes": 1_811,
        "sha256": "ec36f8c9927e024faa40dd942cd0b6ba45cbcd8a90cf7ea4ec4e7576b30c87f8",
    },
    "native_closure": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/run-closure.status"
        ),
        "bytes": 361,
        "sha256": "0a1cd6a947daa5d943551d3c17808fe85ecd11f6d29984fee5f9482b9e3b2389",
    },
}

NEGATIVE_QV_PIN = MappingProxyType(
    {
        "logical_shape": [N_LEVELS, N_CELLS],
        "full_qv_sha256": (
            "c0180afa0fa99253f414472d8f767421c85b1324f67e161fdef9f6244b565099"
        ),
        "negative_count": 215,
        "negative_indices_sha256": (
            "b622daead41e76d988e5419d92bac9b28a2390e8cfdb619d38ff19c1188a10a4"
        ),
        "negative_values_sha256": (
            "192e06fa6a1302eb06f7148e0cb85483ca61a21caf0eed4bc95ccfcb3452eb4b"
        ),
    }
)

INIT_RECONSTRUCTION_COEFFICIENTS_PIN = MappingProxyType(
    {
        "dimensions": ("nCells", "maxEdges", "R3"),
        "shape": (N_CELLS, 10, 3),
        "dtype": "<f4",
        "static_placeholder_raw_sha256": (
            "b3b09d26538fe509096884906c93ff8b8d2c794300bafcab91449eee1c7bd31c"
        ),
        "init_carrier_raw_sha256": (
            "1d25d2439a6cdcc3cc4a3cabfb5b6720730bb548f4cd88208340cca9df883350"
        ),
        "active_slots": 983_040,
        "active_components": 2_949_120,
        "nonzero_components": 2_949_120,
    }
)


INIT_EDGE_NORMAL_VECTORS_PIN = MappingProxyType(
    {
        "dimensions": ("nEdges", "R3"),
        "shape": (N_EDGES, 3),
        "dtype": "<f4",
        "static_placeholder_raw_sha256": (
            "91e0c31eb2d6776a903dc5456c5e72e1e447c2d835cebbcde87581738cac735b"
        ),
        "init_carrier_raw_sha256": (
            "25d9ef5c70b38a2e7d2c9c60456d9835a1e0fc790b60a3c98d1dbb44482d41da"
        ),
        "nonzero_components": 1_474_550,
        "exact_zero_components": 10,
        "zero_rows": 0,
        "float64_norm_min": 0.999999867111912,
        "float64_norm_max": 1.000000135860109,
    }
)

LANDMASK_CONSTRUCTOR_CAST_PIN = MappingProxyType(
    {
        "source_dimensions": ("nCells",),
        "shape": (N_CELLS,),
        "source_dtype": "<i4",
        "source_array_sha256": (
            "6aee8da961b605b9aa21362840aec214beaa384d42041e232ff8f8a9f60b22b5"
        ),
        "source_unique_values": (0, 1),
        "target_dtype": "<f4",
        "target_array_sha256": (
            "31a7e78b7f0a6b10d719df443b0e2d629e60028c8ac628c5ad8556a87c5eae65"
        ),
        "target_uint32_values": (0, 1_065_353_216),
    }
)

PHYSICS_GEOMETRY_CARRIER_PIN = MappingProxyType(
    {
        "cellsOnEdge": {
            "dimensions": ("nEdges", "TWO"),
            "shape": (N_EDGES, 2),
            "dtype": "<i4",
            "raw_sha256": (
                "a53b1c9bf9e5c0e026c7253b9026f4711bc973acd376f70a6e996adfa584b2d3"
            ),
            "source_roles": ("grid", "static", "init"),
        },
        "east_north": {
            "field_names": ("east", "north"),
            "absent_source_roles": ("grid", "static", "init"),
            "fallback_source_fields": ("lonCell", "latCell"),
            "fallback_dtype": "<f8",
        },
        "edgeNormalVectors": {
            "grid_present": False,
            "static_present": True,
            "init_present": True,
        },
    }
)

NATIVE_FIELD_MAP = MappingProxyType(
    {
        "qv": ("qv", "level_cell"),
        "qc": ("qc", "level_cell"),
        "qr": ("qr", "level_cell"),
        "qi": ("qi", "level_cell"),
        "qs": ("qs", "level_cell"),
        "qg": ("qg", "level_cell"),
        "u_zonal": ("uReconstructZonal", "level_cell"),
        "v_meridional": ("uReconstructMeridional", "level_cell"),
        "normal_u": ("u", "level_edge"),
        "w": ("w", "interface_cell"),
        "rho": ("rho", "level_cell"),
        "theta": ("theta", "level_cell"),
        "pressure": ("pressure", "level_cell"),
        "surface_pressure": ("surface_pressure", "cell"),
        "tsk": ("skintemp", "cell"),
        "t2": ("t2m", "cell"),
        "hfx": ("hfx", "cell"),
        "qfx": ("qfx", "cell"),
        "lh": ("lh", "cell"),
        "u10": ("u10", "cell"),
        "v10": ("v10", "cell"),
        "smois": ("smois", "soil_cell"),
        "tslb": ("tslb", "soil_cell"),
        "rainc": ("rainc", "cell"),
        "rainnc": ("rainnc", "cell"),
        "dusfcg": ("dusfcg", "cell"),
        "dvsfcg": ("dvsfcg", "cell"),
        "dtaux3d": ("dtaux3d", "level_cell"),
        "dtauy3d": ("dtauy3d", "level_cell"),
        "rubldiff": ("rubldiff", "level_cell"),
        "rvbldiff": ("rvbldiff", "level_cell"),
    }
)

# Broad quantitative consistency limits, not parity tolerances.  They catch
# unit/layout/sign disasters while respecting the finalized seam's documented
# pressure, RRTMG, NoahMP-atmosphere, and GF forcing differences.
NATIVE_RMSE_LIMITS = MappingProxyType(
    {
        "qv": 1.0e-2,
        "qc": 5.0e-3,
        "qr": 5.0e-3,
        "qi": 5.0e-3,
        "qs": 5.0e-3,
        "qg": 5.0e-3,
        "u_zonal": 40.0,
        "v_meridional": 40.0,
        "normal_u": 40.0,
        "w": 20.0,
        "rho": 0.5,
        "theta": 30.0,
        "pressure": 5_000.0,
        "surface_pressure": 5_000.0,
        "tsk": 30.0,
        "t2": 30.0,
        "hfx": 1_500.0,
        "qfx": 2.0e-2,
        "lh": 1_500.0,
        "u10": 35.0,
        "v10": 35.0,
        "smois": 0.5,
        "tslb": 30.0,
        "rainc": 100.0,
        "rainnc": 100.0,
        "dusfcg": 150.0,
        "dvsfcg": 150.0,
        "dtaux3d": 0.5,
        "dtauy3d": 0.5,
        "rubldiff": 0.5,
        "rvbldiff": 0.5,
    }
)


class ReleaseNotFrozen(RuntimeError):
    """The executable path has one or more unresolved exact-byte pins."""


class CompositeTransactionError(RuntimeError):
    """A staged Arwen/MPAS composite step failed without being published."""


@dataclass(frozen=True, slots=True)
class CompositeStepResult:
    committed: Any
    backend_receipt: Mapping[str, Any]
    clamp_d2h: Any
    recovery: Any
    # GF's advective forcing formed by THIS step's dynamics; the next step's
    # begin_step consumes it, exactly as native MPAS consumes rthdynten.
    dynamics_tendencies: Any = None


@dataclass(frozen=True, slots=True)
class HostDriverCheckpoint:
    state: Any
    saved_diagnostics: Any
    backend_state: Mapping[str, Any]
    atmosphere_fingerprint: Mapping[str, Any]
    backend_fingerprint: Mapping[str, Any]
    model_time_seconds: float
    # GF's advective forcing (rthdynten/rqvdynten) formed by the checkpoint
    # step's own dynamics.  It is DRIVER-OWNED per-step carried state living
    # outside both the MPAS atmosphere and the Arwen backend restart payload
    # (execute_composite_step hands it forward; the NEXT begin_step consumes
    # it).  A checkpoint without it restores a run whose first resumed step
    # feeds GF zero forcing lanes while the unbroken run feeds the real
    # step-15 forcing -- the deterministic all-arms step-16 divergence #327
    # measured 5/5 on the reference node (checkpoint schema v3 closes it).
    gf_dynamics_tendencies: Mapping[str, Any]
    gf_forcing_fingerprint: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    selected = Path(path)
    if not selected.is_file():
        raise FileNotFoundError(f"required ordinary file is missing: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def arwen_source_manifest() -> Mapping[str, str]:
    """The sixteen-file sha-freeze of the gpuwm source the seam executes.

    Imported lazily from the pinned adapter module, so callers must verify
    ``EXECUTION_SOURCE_PINS`` before trusting the returned constants.
    """

    from mpas_port.cuda_arwen_physics_v841 import ARWEN_SOURCE_MANIFEST

    return ARWEN_SOURCE_MANIFEST


_NETCDF_FILE_ID_ATTRIBUTE = "file_id"
_NC_CHAR = 2
_NC_DIMENSION = 0x0A
_NC_ATTRIBUTE = 0x0C
_NETCDF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 8}


def netcdf_file_id_value_span(path: str | Path) -> tuple[int, int]:
    """Locate the byte span of the ``file_id`` global attribute VALUE.

    Walks the classic netCDF (CDF-1/2/5) header -- magic, record count,
    dimension list, then the global attribute list -- so the span follows
    the header wherever it sits.  A hardcoded offset would silently move the
    mask onto data bytes the moment the header changes shape.
    """

    selected = Path(path)
    with selected.open("rb") as stream:
        head = stream.read(1 << 22)

    if len(head) < 8 or head[:3] != b"CDF" or head[3] not in (1, 2, 5):
        raise RuntimeError(
            f"{selected} is not a classic CDF-1/2/5 netCDF file; the file_id "
            "mask is only defined against the classic header layout"
        )
    count_size = 8 if head[3] == 5 else 4

    def read_int4(offset: int) -> tuple[int, int]:
        if offset + 4 > len(head):
            raise RuntimeError(f"netCDF header of {selected} ends inside a field")
        return int.from_bytes(head[offset : offset + 4], "big"), offset + 4

    def read_count(offset: int) -> tuple[int, int]:
        if offset + count_size > len(head):
            raise RuntimeError(f"netCDF header of {selected} ends inside a count")
        return (
            int.from_bytes(head[offset : offset + count_size], "big"),
            offset + count_size,
        )

    def read_name(offset: int) -> tuple[bytes, int]:
        length, offset = read_count(offset)
        end = offset + length
        if end > len(head):
            raise RuntimeError(f"netCDF header of {selected} ends inside a name")
        return head[offset:end], end + ((-length) % 4)

    position = 4 + count_size  # magic/version, then numrecs (or STREAMING)

    tag, position = read_int4(position)
    count, position = read_count(position)
    if tag not in (0, _NC_DIMENSION) or (tag == 0 and count != 0):
        raise RuntimeError(f"netCDF dimension list of {selected} is malformed")
    for _ in range(count):
        _, position = read_name(position)
        _, position = read_count(position)  # dimension length

    tag, position = read_int4(position)
    count, position = read_count(position)
    if tag not in (0, _NC_ATTRIBUTE) or (tag == 0 and count != 0):
        raise RuntimeError(
            f"netCDF global attribute list of {selected} is malformed"
        )
    for _ in range(count):
        name, position = read_name(position)
        nc_type, position = read_int4(position)
        nelems, position = read_count(position)
        size = _NETCDF_TYPE_SIZES.get(nc_type)
        if size is None:
            raise RuntimeError(
                f"netCDF global attribute {name!r} of {selected} has "
                f"unsupported type {nc_type}"
            )
        if name == _NETCDF_FILE_ID_ATTRIBUTE.encode("ascii"):
            if nc_type != _NC_CHAR:
                raise RuntimeError(
                    f"file_id global attribute of {selected} is not NC_CHAR"
                )
            return position, nelems
        value_bytes = nelems * size
        position += value_bytes + ((-value_bytes) % 4)

    raise RuntimeError(
        "file_id global attribute is not present in the netCDF header of "
        f"{selected}"
    )


def netcdf_masked_digests(path: str | Path) -> dict[str, Any]:
    """Whole-file and file_id-masked SHA-256 of one netCDF file, in one pass.

    The masked digest hashes every byte of the file except the located
    ``file_id`` attribute value, which is hashed as NUL bytes -- exactly the
    masking the regenerated-authority digests were measured with.  MPAS
    stamps a fresh random ten-character ``file_id`` into every output file,
    so a whole-file digest can never survive a rerun while the masked digest
    identifies the content.
    """

    selected = Path(path)
    if not selected.is_file():
        raise FileNotFoundError(f"required ordinary file is missing: {selected}")
    offset, length = netcdf_file_id_value_span(selected)
    raw = hashlib.sha256()
    masked = hashlib.sha256()
    with selected.open("rb") as stream:
        remaining = offset
        while remaining:
            block = stream.read(min(16 * 1024 * 1024, remaining))
            if not block:
                raise RuntimeError(f"{selected} ends before the file_id attribute")
            raw.update(block)
            masked.update(block)
            remaining -= len(block)
        value = stream.read(length)
        if len(value) != length:
            raise RuntimeError(f"{selected} ends inside the file_id attribute value")
        raw.update(value)
        masked.update(b"\0" * length)
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            raw.update(block)
            masked.update(block)
    return {
        "sha256": raw.hexdigest(),
        "masked_sha256": masked.hexdigest(),
        "file_id": value.decode("ascii", "replace"),
        "file_id_offset": offset,
        "file_id_len": length,
    }


def _plain_absolute(path: str | Path, label: str) -> Path:
    selected = Path(path).expanduser().absolute()
    if selected.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {selected}")
    return selected


def default_authority_paths() -> dict[str, Path]:
    return {
        role: ROOT / str(pin["relative_path"])
        for role, pin in AUTHORITY_PINS.items()
    }


def _file_record(role: str, path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    selected = _plain_absolute(path, role)
    if not selected.is_file():
        raise FileNotFoundError(f"missing {role} authority: {selected}")
    size = selected.stat().st_size
    if size != int(pin["bytes"]):
        raise RuntimeError(f"{role} byte count changed: {size} != {pin['bytes']}")
    if "masked_sha256" in pin:
        digests = netcdf_masked_digests(selected)
        if digests["masked_sha256"] != pin["masked_sha256"]:
            raise RuntimeError(
                f"{role} masked content digest changed: "
                f"{digests['masked_sha256']} != {pin['masked_sha256']} "
                "(only the random file_id nonce may differ between reruns; "
                "the authority's data bytes moved)"
            )
        return {"path": str(selected), "bytes": size, **digests}
    digest = sha256_file(selected)
    if digest != pin["sha256"]:
        raise RuntimeError(f"{role} SHA-256 changed: {digest} != {pin['sha256']}")
    return {"path": str(selected), "bytes": size, "sha256": digest}


def verify_authorities(paths: Mapping[str, Path]) -> dict[str, Any]:
    if set(paths) != set(AUTHORITY_PINS):
        raise ValueError("authority path mapping does not cover the exact closed set")
    files = {
        role: _file_record(role, paths[role], AUTHORITY_PINS[role])
        for role in AUTHORITY_PINS
    }
    validation = json.loads(paths["native_validation_receipt"].read_text("utf-8"))
    if validation.get("status") != "passed":
        raise RuntimeError("native GF+YSU-GWDO validation receipt is not passed")
    if validation.get("execution") != {"cuda_launched": False, "device": "CPU"}:
        raise RuntimeError("native authority execution identity changed")
    closure = paths["native_closure"].read_text("ascii")
    for line in (
        "status=passed",
        "gravity_wave_drag=bl_ysu_gwdo",
        "direct_gwd_activity_gate=true",
        "timesteps=30",
        "dt_seconds=120",
    ):
        if line not in closure.splitlines():
            raise RuntimeError(f"native closure lost exact line {line!r}")
    return {
        "files": files,
        "native_validation_status": "passed",
        "sha256": canonical_json_sha256(files),
    }


def unresolved_source_pins() -> tuple[str, ...]:
    return tuple(sorted(path for path, digest in EXECUTION_SOURCE_PINS.items() if digest is None))


def require_frozen_execution_sources() -> dict[str, Any]:
    unresolved = unresolved_source_pins()
    if unresolved:
        raise ReleaseNotFrozen(
            "full-physics CUDA launch remains closed until exact SHA-256 pins "
            f"are released for {list(unresolved)}"
        )
    records: dict[str, Any] = {}
    for relative, expected in EXECUTION_SOURCE_PINS.items():
        assert expected is not None
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ReleaseNotFrozen(
                f"frozen execution source changed for {relative}: {actual} != {expected}"
            )
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return {"files": records, "sha256": canonical_json_sha256(records)}


def negative_qv_fingerprint(logical_qv: Any) -> dict[str, Any]:
    qv = np.ascontiguousarray(np.asarray(logical_qv, dtype=np.float32))
    negative = np.argwhere(qv < np.float32(0.0)).astype(np.int64, copy=False)
    values = np.ascontiguousarray(qv[qv < np.float32(0.0)])
    return {
        "logical_shape": list(qv.shape),
        "full_qv_sha256": hashlib.sha256(qv.tobytes(order="C")).hexdigest(),
        "negative_count": int(values.size),
        "negative_indices_sha256": hashlib.sha256(negative.tobytes(order="C")).hexdigest(),
        "negative_values_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def augment_exact_wsm6_scalars(state: Any) -> dict[str, Any]:
    """Append native-cold +0 FP32 qi/qs/qg without touching qv/qc/qr."""

    source = np.asarray(state.scalars)
    expected = (len(SOURCE_SCALAR_NAMES), N_LEVELS, N_CELLS)
    if source.dtype != np.dtype(np.float32) or source.shape != expected:
        raise TypeError(f"source qv/qc/qr must be FP32 {expected}, got {source.dtype} {source.shape}")
    if not np.all(np.isfinite(source)):
        raise FloatingPointError("source qv/qc/qr contains non-finite values")
    source_hashes = {
        name: array_sha256(source[index]) for index, name in enumerate(SOURCE_SCALAR_NAMES)
    }
    qv_fingerprint = negative_qv_fingerprint(source[0])
    for key, expected_value in NEGATIVE_QV_PIN.items():
        if qv_fingerprint[key] != expected_value:
            raise RuntimeError(f"stock qv fingerprint changed at {key}")
    cold = np.zeros(
        (len(COLD_ZERO_SCALAR_NAMES), N_LEVELS, N_CELLS),
        dtype=np.float32,
        order="C",
    )
    if np.any(cold.view(np.uint32) != np.uint32(0)):
        raise RuntimeError("cold WSM6 species are not exact +0 FP32")
    cold_hash = array_sha256(cold[0])
    augmented = np.concatenate((source, cold), axis=0)
    if augmented.shape != (len(SCALAR_NAMES), N_LEVELS, N_CELLS):
        raise RuntimeError("six-species WSM6 augmentation produced the wrong shape")
    state.scalars = np.ascontiguousarray(augmented, dtype=np.float32)
    return {
        "source_present": list(SOURCE_SCALAR_NAMES),
        "source_absent": list(COLD_ZERO_SCALAR_NAMES),
        "source_hashes": source_hashes,
        "cold_zero_plane_sha256": cold_hash,
        "cold_zero_planes_identical": True,
        "cold_zero_uint32": 0,
        "scalar_order": list(SCALAR_NAMES),
        "negative_qv": qv_fingerprint,
        "negative_qv_preserved_until_candidate_pre_wsm6_clamp": True,
    }


def validate_destination(cache_root: Path, output_root: Path, protected: Sequence[Path]) -> tuple[Path, Path]:
    cache = _plain_absolute(cache_root, "cache root")
    output = _plain_absolute(output_root, "output root")
    if cache == output or cache in output.parents or output in cache.parents:
        raise ValueError("cache and output roots must be disjoint")
    for label, path in (("cache root", cache), ("output root", output)):
        if path.exists():
            raise FileExistsError(f"{label} must be absent: {path}")
        if path == ROOT or ROOT in path.parents and path.parent == ROOT / "src":
            raise ValueError(f"{label} overlaps protected source scope")
        for authority in protected:
            absolute = authority.absolute()
            if path == absolute or path in absolute.parents or absolute in path.parents:
                raise ValueError(f"{label} overlaps protected authority {absolute}")
    return cache, output


def validate_baseline_diagnostic_destination(
    diagnostic_root: Path,
    *,
    cache_root: Path,
    output_root: Path,
    protected: Sequence[Path],
) -> Path:
    """Admit one fresh, isolated, explicitly non-release diagnostic root."""

    diagnostic = _plain_absolute(diagnostic_root, "baseline diagnostic output root")
    for label, other in (("cache root", cache_root), ("output root", output_root)):
        if diagnostic == other or diagnostic in other.parents or other in diagnostic.parents:
            raise ValueError(
                "baseline diagnostic output root must be disjoint from " + label
            )
    if diagnostic.exists():
        raise FileExistsError(
            f"baseline diagnostic output root must be absent: {diagnostic}"
        )
    if diagnostic == ROOT or ROOT in diagnostic.parents and diagnostic.parent == ROOT / "src":
        raise ValueError("baseline diagnostic output root overlaps protected source scope")
    for authority in protected:
        absolute = authority.absolute()
        if (
            diagnostic == absolute
            or diagnostic in absolute.parents
            or absolute in diagnostic.parents
        ):
            raise ValueError(
                "baseline diagnostic output root overlaps protected authority "
                f"{absolute}"
            )
    return diagnostic



def _mesh_value(mesh: Any, name: str) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    raise AttributeError(f"mesh has no {name!r}")


def attach_inactive_zero_deformation(mesh: Any) -> dict[str, Any]:
    counts = np.asarray(_mesh_value(mesh, "nEdgesOnCell"))
    n_cells = int(counts.size)
    edges = np.asarray(_mesh_value(mesh, "edgesOnCell"))
    max_edges = int(edges.shape[1] if edges.shape[0] == n_cells else edges.shape[0])
    shape = (n_cells, max_edges)
    arrays = getattr(mesh, "arrays", None)
    if not isinstance(arrays, dict):
        raise TypeError("precision-preserving mesh arrays must be mutable during host preparation")
    if "defc_a" in arrays or "defc_b" in arrays:
        raise ValueError("inactive deformation arrays must be generated exactly once")
    arrays["defc_a"] = np.zeros(shape, dtype=np.float32, order="C")
    arrays["defc_b"] = np.zeros(shape, dtype=np.float32, order="C")
    return {
        "shape": list(shape),
        "dtype": "float32",
        "active_mixing_claim": False,
        "defc_a_sha256": array_sha256(arrays["defc_a"]),
        "defc_b_sha256": array_sha256(arrays["defc_b"]),
    }


def _read_exact_variable(
    dataset: Any,
    name: str,
    *,
    dtype: Any,
    dimensions: tuple[str, ...],
) -> np.ndarray:
    if name not in dataset.variables:
        raise ValueError(f"official init is missing required variable {name!r}")
    variable = dataset.variables[name]
    if np.dtype(variable.dtype) != np.dtype(dtype):
        raise TypeError(f"{name} dtype {variable.dtype} != {np.dtype(dtype)}")
    if tuple(variable.dimensions) != dimensions:
        raise ValueError(f"{name} dimensions {variable.dimensions} != {dimensions}")
    variable.set_auto_maskandscale(False)
    value = np.ascontiguousarray(np.asarray(variable[...]))
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value




def load_f000_initialized_surface_diagnostics(
    init_path: Path,
) -> dict[str, Any]:
    """Load exact init-only diagnostics absent from the frozen seam constructor."""

    from netCDF4 import Dataset

    arrays: dict[str, np.ndarray] = {}
    with Dataset(init_path, "r") as dataset:
        for target, pin in F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items():
            source = str(pin["source"])
            value = _read_exact_variable(
                dataset,
                source,
                dtype=np.float32,
                dimensions=("Time", "nCells"),
            )
            if value.shape != (1, N_CELLS):
                raise ValueError(f"{source} shape changed: {value.shape}")
            field = np.ascontiguousarray(value[0], dtype=np.float32)
            digest = array_sha256(field)
            if digest != pin["sha256"]:
                raise RuntimeError(
                    f"initialized F000 {target} SHA-256 changed: {digest}"
                )
            arrays[target] = field
    return {
        "arrays": arrays,
        "receipt": {
            "source": "exact hash-pinned official x4.163842 init fields",
            "policy": (
                "F000 snapshot-only replacement of frozen Arwen pre-first-call "
                "exact +0 optional diagnostic placeholders"
            ),
            "fields": {
                target: {
                    "source_variable": pin["source"],
                    "dtype": arrays[target].dtype.str,
                    "shape": list(arrays[target].shape),
                    "sha256": array_sha256(arrays[target]),
                }
                for target, pin in F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items()
            },
        },
    }


def overlay_f000_initialized_surface_diagnostics(
    arrays: dict[str, np.ndarray],
    initialized: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace only exact +0 pre-first-call placeholders with sealed init bytes."""

    expected = set(F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS)
    if set(initialized) != expected:
        raise ValueError("F000 initialized surface diagnostic inventory changed")
    receipt: dict[str, Any] = {}
    for name, pin in F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items():
        if name not in arrays:
            raise ValueError(f"frozen seam lacks F000 optional diagnostic {name!r}")
        placeholder = np.ascontiguousarray(np.asarray(arrays[name]))
        source = np.ascontiguousarray(np.asarray(initialized[name]))
        if (
            placeholder.dtype != np.dtype(np.float32)
            or placeholder.shape != (N_CELLS,)
            or np.any(placeholder.view(np.uint32) != np.uint32(0))
        ):
            raise ValueError(
                f"frozen seam F000 {name} is not the exact +0 FP32 placeholder"
            )
        if (
            source.dtype != np.dtype(np.float32)
            or source.shape != (N_CELLS,)
            or not np.all(np.isfinite(source))
            or array_sha256(source) != pin["sha256"]
        ):
            raise ValueError(f"sealed initialized F000 {name} identity changed")
        arrays[name] = np.array(source, copy=True, order="C")
        receipt[name] = {
            "source_variable": pin["source"],
            "placeholder_sha256": array_sha256(placeholder),
            "initialized_sha256": array_sha256(source),
            "snapshot_sha256": array_sha256(arrays[name]),
            "source_correct": True,
        }
    return {
        "applied": True,
        "scope": "F000 diagnostic snapshot only; forecast/restart state unchanged",
        "fields": receipt,
    }
def overlay_exact_init_reconstruction_coefficients(
    mesh: Any,
    init_path: Path,
) -> dict[str, Any]:
    """Replace the intentional static zero placeholder with exact init bytes."""

    from netCDF4 import Dataset

    pin = INIT_RECONSTRUCTION_COEFFICIENTS_PIN
    expected_shape = tuple(pin["shape"])
    placeholder = np.asarray(_mesh_value(mesh, "coeffs_reconstruct"))
    if placeholder.dtype != np.dtype(np.float32) or placeholder.shape != expected_shape:
        raise TypeError(
            "static coeffs_reconstruct placeholder must be exact FP32 "
            f"{expected_shape}, got {placeholder.dtype} {placeholder.shape}"
        )
    placeholder = np.ascontiguousarray(placeholder)
    placeholder_raw_sha256 = hashlib.sha256(
        placeholder.tobytes(order="C")
    ).hexdigest()
    if placeholder_raw_sha256 != pin["static_placeholder_raw_sha256"]:
        raise RuntimeError(
            "static reconstruction placeholder bytes changed: "
            f"{placeholder_raw_sha256}"
        )
    if np.any(placeholder.view(np.uint32) != np.uint32(0)):
        raise RuntimeError(
            "static coeffs_reconstruct is no longer the exact all-+0 placeholder"
        )
    prior_source = str(mesh.variable_sources.get("coeffs_reconstruct", ""))
    if prior_source != "static":
        raise RuntimeError(
            "reconstruction placeholder did not come from the sealed static file"
        )

    with Dataset(init_path, "r") as dataset:
        coefficients = _read_exact_variable(
            dataset,
            "coeffs_reconstruct",
            dtype=np.float32,
            dimensions=tuple(pin["dimensions"]),
        )
        init_counts = _read_exact_variable(
            dataset,
            "nEdgesOnCell",
            dtype=np.int32,
            dimensions=("nCells",),
        )
        init_edges_raw = _read_exact_variable(
            dataset,
            "edgesOnCell",
            dtype=np.int32,
            dimensions=("nCells", "maxEdges"),
        )
        variable = dataset.variables["coeffs_reconstruct"]
        variable_attrs = {
            name: getattr(variable, name) for name in variable.ncattrs()
        }

    if coefficients.shape != expected_shape:
        raise ValueError(
            f"initialized coeffs_reconstruct shape changed: {coefficients.shape}"
        )
    coefficients_raw_sha256 = hashlib.sha256(
        coefficients.tobytes(order="C")
    ).hexdigest()
    if coefficients_raw_sha256 != pin["init_carrier_raw_sha256"]:
        raise RuntimeError(
            "initialized reconstruction coefficient bytes changed: "
            f"{coefficients_raw_sha256}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("initialized reconstruction coefficients are non-finite")

    mesh_counts = np.ascontiguousarray(
        np.asarray(_mesh_value(mesh, "nEdgesOnCell"), dtype=np.int32)
    )
    mesh_edges = np.ascontiguousarray(
        np.asarray(_mesh_value(mesh, "edgesOnCell"), dtype=np.int32)
    )
    if mesh_counts.shape != (N_CELLS,) or mesh_edges.shape != expected_shape[:2]:
        raise ValueError("prepared mesh reconstruction topology shape changed")
    if not np.array_equal(init_counts, mesh_counts):
        raise RuntimeError("init/static nEdgesOnCell topology differs")
    if np.any(init_edges_raw < 0):
        raise RuntimeError("initialized source edgesOnCell must use MPAS one-based/zero padding")
    init_edges = np.where(
        init_edges_raw > np.int32(0),
        init_edges_raw - np.int32(1),
        np.int32(-1),
    ).astype(np.int32, copy=False)
    init_edges = np.ascontiguousarray(init_edges)
    if not np.array_equal(init_edges, mesh_edges):
        raise RuntimeError("init/static edgesOnCell topology differs after canonicalization")

    slots = np.arange(expected_shape[1], dtype=np.int32)[None, :]
    active = slots < mesh_counts[:, None]
    active_slots = int(np.count_nonzero(active))
    active_components = active_slots * expected_shape[2]
    if active_slots != int(pin["active_slots"]):
        raise RuntimeError(f"active reconstruction slot count changed: {active_slots}")
    if active_components != int(pin["active_components"]):
        raise RuntimeError(
            f"active reconstruction component count changed: {active_components}"
        )
    active_values = np.ascontiguousarray(coefficients[active])
    nonzero_components = int(np.count_nonzero(active_values))
    if (
        active_values.size != active_components
        or nonzero_components != int(pin["nonzero_components"])
    ):
        raise RuntimeError(
            "every active initialized reconstruction 3-vector component "
            "must be nonzero"
        )
    padding_values = np.ascontiguousarray(coefficients[~active])
    if np.any(padding_values.view(np.uint32) != np.uint32(0)):
        raise RuntimeError(
            "initialized reconstruction coefficient padding must be bitwise +0"
        )

    coefficients = np.array(coefficients, copy=True, order="C")
    mesh.arrays["coeffs_reconstruct"] = coefficients
    mesh.variable_sources["coeffs_reconstruct"] = (
        "init_exact_in_memory_physics_reconstruction_overlay"
    )
    mesh.variable_dimensions["coeffs_reconstruct"] = tuple(pin["dimensions"])
    mesh.variable_attrs["coeffs_reconstruct"] = variable_attrs
    receipt = {
        "schema": "mpas-port.v841-init-reconstruction-overlay/v1",
        "field": "coeffs_reconstruct",
        "policy": "static intentional +0 placeholder -> sealed initialized carrier",
        "static_placeholder": {
            "source": prior_source,
            "shape": list(placeholder.shape),
            "dtype": placeholder.dtype.str,
            "raw_c_sha256": placeholder_raw_sha256,
            "nonzero_components": 0,
            "bitwise_positive_zero": True,
        },
        "init_carrier": {
            "path": str(Path(init_path).absolute()),
            "dimensions": list(pin["dimensions"]),
            "shape": list(coefficients.shape),
            "dtype": coefficients.dtype.str,
            "raw_c_sha256": coefficients_raw_sha256,
            "nonzero_components": nonzero_components,
        },
        "topology_identity": {
            "n_edges_on_cell_sha256": array_sha256(mesh_counts),
            "edges_on_cell_sha256": array_sha256(mesh_edges),
            "init_static_bit_identical": True,
            "active_slots": active_slots,
            "active_components": active_components,
        },
        "padding": {
            "components": int(padding_values.size),
            "bitwise_positive_zero": True,
        },
        "active_every_component_nonzero": True,
        "mesh_variable_source": mesh.variable_sources["coeffs_reconstruct"],
        "source_files_mutated": False,
        "overlay_scope": "dynamics mesh in memory only",
    }
    mesh.provenance["v841_init_reconstruction_overlay"] = receipt
    return receipt


def overlay_exact_init_edge_normal_vectors(
    mesh: Any,
    *,
    grid_path: Path,
    static_path: Path,
    init_path: Path,
) -> dict[str, Any]:
    """Bind exact initialized edge normals and receipt every physics carrier."""

    from netCDF4 import Dataset

    normal_pin = INIT_EDGE_NORMAL_VECTORS_PIN
    carrier_pin = PHYSICS_GEOMETRY_CARRIER_PIN
    expected_shape = tuple(normal_pin["shape"])
    placeholder = np.asarray(_mesh_value(mesh, "edgeNormalVectors"))
    if placeholder.dtype != np.dtype(normal_pin["dtype"]) or placeholder.shape != expected_shape:
        raise TypeError(
            "static edgeNormalVectors placeholder must be exact FP32 "
            f"{expected_shape}, got {placeholder.dtype} {placeholder.shape}"
        )
    placeholder = np.ascontiguousarray(placeholder)
    placeholder_raw_sha256 = hashlib.sha256(
        placeholder.tobytes(order="C")
    ).hexdigest()
    if placeholder_raw_sha256 != normal_pin["static_placeholder_raw_sha256"]:
        raise RuntimeError(
            "static edgeNormalVectors placeholder bytes changed: "
            f"{placeholder_raw_sha256}"
        )
    if np.any(placeholder.view(np.uint32) != np.uint32(0)):
        raise RuntimeError(
            "static edgeNormalVectors is no longer the exact all-+0 placeholder"
        )
    prior_source = str(mesh.variable_sources.get("edgeNormalVectors", ""))
    if prior_source != "static":
        raise RuntimeError("edge-normal placeholder did not come from the sealed static file")

    role_paths = {
        "grid": Path(grid_path),
        "static": Path(static_path),
        "init": Path(init_path),
    }
    raw_cells_by_role: dict[str, np.ndarray] = {}
    source_presence: dict[str, Any] = {}
    normals_by_role: dict[str, np.ndarray] = {}
    variable_attrs: dict[str, Any] = {}
    cells_pin = carrier_pin["cellsOnEdge"]
    normal_presence_pin = carrier_pin["edgeNormalVectors"]
    for role, path in role_paths.items():
        with Dataset(path, "r") as dataset:
            raw_cells = _read_exact_variable(
                dataset,
                "cellsOnEdge",
                dtype=np.int32,
                dimensions=tuple(cells_pin["dimensions"]),
            )
            if raw_cells.shape != tuple(cells_pin["shape"]):
                raise ValueError(f"{role} cellsOnEdge shape changed: {raw_cells.shape}")
            raw_cells_sha256 = hashlib.sha256(raw_cells.tobytes(order="C")).hexdigest()
            if raw_cells_sha256 != cells_pin["raw_sha256"]:
                raise RuntimeError(
                    f"{role} cellsOnEdge source bytes changed: {raw_cells_sha256}"
                )
            raw_cells_by_role[role] = raw_cells
            basis_presence = {
                name: name in dataset.variables
                for name in carrier_pin["east_north"]["field_names"]
            }
            if any(basis_presence.values()):
                raise RuntimeError(
                    f"{role} unexpectedly supplies east/north and bypasses the released fallback"
                )
            expected_normal_present = bool(normal_presence_pin[f"{role}_present"])
            actual_normal_present = "edgeNormalVectors" in dataset.variables
            if actual_normal_present != expected_normal_present:
                raise RuntimeError(
                    f"{role} edgeNormalVectors presence changed: {actual_normal_present}"
                )
            source_presence[role] = {
                "east": basis_presence["east"],
                "north": basis_presence["north"],
                "edgeNormalVectors": actual_normal_present,
            }
            if actual_normal_present:
                normals = _read_exact_variable(
                    dataset,
                    "edgeNormalVectors",
                    dtype=np.float32,
                    dimensions=tuple(normal_pin["dimensions"]),
                )
                if normals.shape != expected_shape:
                    raise ValueError(
                        f"{role} edgeNormalVectors shape changed: {normals.shape}"
                    )
                normals_by_role[role] = normals
                if role == "init":
                    variable = dataset.variables["edgeNormalVectors"]
                    variable_attrs = {
                        name: getattr(variable, name) for name in variable.ncattrs()
                    }

    for role in ("static", "init"):
        if not np.array_equal(raw_cells_by_role["grid"], raw_cells_by_role[role]):
            raise RuntimeError(f"grid/{role} raw cellsOnEdge topology differs")
    if not np.array_equal(normals_by_role["static"], placeholder):
        raise RuntimeError("mesh/static edgeNormalVectors placeholder differs")
    initialized = np.ascontiguousarray(normals_by_role["init"])
    initialized_raw_sha256 = hashlib.sha256(
        initialized.tobytes(order="C")
    ).hexdigest()
    if initialized_raw_sha256 != normal_pin["init_carrier_raw_sha256"]:
        raise RuntimeError(
            "initialized edgeNormalVectors bytes changed: "
            f"{initialized_raw_sha256}"
        )
    if not np.all(np.isfinite(initialized)):
        raise FloatingPointError("initialized edgeNormalVectors contains non-finite values")
    nonzero_components = int(np.count_nonzero(initialized))
    exact_zero_components = int(initialized.size - nonzero_components)
    zero_mask = initialized == np.float32(0.0)
    if (
        nonzero_components != int(normal_pin["nonzero_components"])
        or exact_zero_components != int(normal_pin["exact_zero_components"])
        or np.any(initialized.view(np.uint32)[zero_mask] != np.uint32(0))
    ):
        raise RuntimeError("initialized edge-normal component activity changed")
    zero_rows = int(np.count_nonzero(np.all(initialized == np.float32(0.0), axis=1)))
    if zero_rows != int(normal_pin["zero_rows"]):
        raise RuntimeError(f"initialized edge-normal zero-row count changed: {zero_rows}")
    initialized64 = initialized.astype(np.float64)
    norms = np.sqrt(np.sum(initialized64 * initialized64, axis=1, dtype=np.float64))
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))
    if (
        norm_min != float(normal_pin["float64_norm_min"])
        or norm_max != float(normal_pin["float64_norm_max"])
    ):
        raise RuntimeError(
            f"initialized edge-normal norm envelope changed: {(norm_min, norm_max)}"
        )

    raw_cells = raw_cells_by_role["init"]
    if np.any(raw_cells <= np.int32(0)):
        raise RuntimeError("cellsOnEdge must contain two valid one-based endpoints")
    canonical_cells = np.ascontiguousarray(
        raw_cells.astype(np.int64) - np.int64(1)
    )
    mesh_cells = np.ascontiguousarray(np.asarray(_mesh_value(mesh, "cellsOnEdge")))
    if mesh_cells.shape != tuple(cells_pin["shape"]) or not np.array_equal(
        canonical_cells, mesh_cells
    ):
        raise RuntimeError("source/prepared canonical cellsOnEdge topology differs")

    arrays = getattr(mesh, "arrays", None)
    if not isinstance(arrays, dict):
        raise TypeError("precision-preserving mesh arrays must be mutable during host preparation")
    if "east" in arrays or "north" in arrays:
        raise RuntimeError("prepared mesh unexpectedly supplies east/north source arrays")
    basis_receipt: dict[str, Any] = {}
    for name in carrier_pin["east_north"]["fallback_source_fields"]:
        value = np.ascontiguousarray(np.asarray(_mesh_value(mesh, name)))
        source = str(mesh.variable_sources.get(name, ""))
        if (
            value.dtype != np.dtype(carrier_pin["east_north"]["fallback_dtype"])
            or value.shape != (N_CELLS,)
            or not np.all(np.isfinite(value))
            or not source.startswith("grid_binary64")
        ):
            raise RuntimeError(f"{name} is not the sealed binary64 grid fallback carrier")
        basis_receipt[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "array_sha256": array_sha256(value),
            "mesh_variable_source": source,
        }

    initialized = np.array(initialized, copy=True, order="C")
    mesh.arrays["edgeNormalVectors"] = initialized
    mesh.variable_sources["edgeNormalVectors"] = (
        "init_exact_in_memory_physics_edge_normal_overlay"
    )
    mesh.variable_dimensions["edgeNormalVectors"] = tuple(normal_pin["dimensions"])
    mesh.variable_attrs["edgeNormalVectors"] = variable_attrs
    receipt = {
        "schema": "mpas-port.v841-init-edge-normal-overlay/v1",
        "field": "edgeNormalVectors",
        "policy": "static intentional +0 placeholder -> sealed initialized carrier",
        "static_placeholder": {
            "source": prior_source,
            "shape": list(placeholder.shape),
            "dtype": placeholder.dtype.str,
            "raw_c_sha256": placeholder_raw_sha256,
            "nonzero_components": 0,
            "bitwise_positive_zero": True,
        },
        "init_carrier": {
            "path": str(Path(init_path).absolute()),
            "dimensions": list(normal_pin["dimensions"]),
            "shape": list(initialized.shape),
            "dtype": initialized.dtype.str,
            "raw_c_sha256": initialized_raw_sha256,
            "nonzero_components": nonzero_components,
            "exact_positive_zero_components": exact_zero_components,
            "zero_rows": zero_rows,
            "float64_norm_min": norm_min,
            "float64_norm_max": norm_max,
        },
        "cuda_physics_geometry_carrier_audit": {
            "cellsOnEdge": {
                "dimensions": list(cells_pin["dimensions"]),
                "shape": list(cells_pin["shape"]),
                "source_dtype": cells_pin["dtype"],
                "raw_c_sha256": cells_pin["raw_sha256"],
                "grid_static_init_raw_bit_identical": True,
                "prepared_mesh_zero_based_canonical": True,
                "prepared_array_sha256": array_sha256(mesh_cells),
            },
            "east_north": {
                "source_presence": source_presence,
                "absent_in_grid_static_init_and_prepared_mesh": True,
                "released_fallback": "zonal_meridional_vectors(lonCell, latCell)",
                "fallback_basis": basis_receipt,
            },
            "edgeNormalVectors": {
                "source_presence": source_presence,
                "only_zero_placeholder_trap": True,
                "resolved_by_exact_init_overlay": True,
            },
        },
        "mesh_variable_source": mesh.variable_sources["edgeNormalVectors"],
        "source_files_mutated": False,
        "overlay_scope": "dynamics mesh in memory only",
    }
    mesh.provenance["v841_init_edge_normal_overlay"] = receipt
    return receipt

def derive_area_weighted_p_top_v841(
    *,
    pressure_base: Any,
    pressure_perturbation: Any,
    zgrid: Any,
    area_cell: Any,
) -> tuple[float, dict[str, Any]]:
    """Derive fa35's scalar top pressure from exact native F000 columns.

    The per-column top interface follows the released FP32 preparation
    operations. Only the unavoidable per-column-to-scalar reduction uses
    float64: an exact areaCell weighted mean, rounded once to FP32.
    """

    base = np.asarray(pressure_base)
    perturbation = np.asarray(pressure_perturbation)
    height = np.asarray(zgrid)
    area = np.asarray(area_cell)
    mass_shape = (N_LEVELS, N_CELLS)
    interface_shape = (N_INTERFACES, N_CELLS)
    for name, value, shape in (
        ("pressure_base", base, mass_shape),
        ("pressure_perturbation", perturbation, mass_shape),
        ("zgrid", height, interface_shape),
    ):
        if value.dtype != np.dtype(np.float32) or value.shape != shape:
            raise TypeError(f"{name} must be exact FP32 {shape}")
        if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite and C-contiguous")
    if area.dtype != np.dtype(np.float32) or area.shape != (N_CELLS,):
        raise TypeError("areaCell must be exact FP32 [cell]")
    if not area.flags.c_contiguous or not np.all(np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("areaCell must be finite, positive, and C-contiguous")

    pressure = np.add(base, perturbation, dtype=np.float32)
    half = np.float32(0.5)
    one = np.float32(1.0)
    z0 = height[-1]
    z1 = np.multiply(
        half,
        np.add(height[-1], height[-2], dtype=np.float32),
        dtype=np.float32,
    )
    z2 = np.multiply(
        half,
        np.add(height[-2], height[-3], dtype=np.float32),
        dtype=np.float32,
    )
    w1 = np.divide(
        np.subtract(z0, z2, dtype=np.float32),
        np.subtract(z1, z2, dtype=np.float32),
        dtype=np.float32,
    )
    w2 = np.subtract(one, w1, dtype=np.float32)
    logarithm = np.add(
        np.multiply(
            w1, np.log(pressure[-1], dtype=np.float32), dtype=np.float32
        ),
        np.multiply(
            w2, np.log(pressure[-2], dtype=np.float32), dtype=np.float32
        ),
        dtype=np.float32,
    )
    top = np.ascontiguousarray(np.exp(logarithm, dtype=np.float32))
    if not np.all(np.isfinite(top)) or np.any(top <= 0):
        raise FloatingPointError("derived native F000 top pressure is invalid")
    area64 = area.astype(np.float64, copy=False)
    weighted_mean64 = float(
        np.sum(top.astype(np.float64) * area64, dtype=np.float64)
        / np.sum(area64, dtype=np.float64)
    )
    scalar = np.float32(weighted_mean64)
    if scalar.view(np.uint32) != EXPECTED_ARWEN_P_TOP_PA_F32.view(np.uint32):
        raise RuntimeError(
            "exact F000 area-weighted p_top changed: "
            f"{float(scalar)} != {float(EXPECTED_ARWEN_P_TOP_PA_F32)}"
        )
    observed = (float(np.min(top)), float(np.median(top)), float(np.max(top)))
    for label, actual, expected in zip(
        ("minimum", "median", "maximum"),
        observed,
        EXPECTED_TOP_PRESSURE_RANGE_PA,
        strict=True,
    ):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5.0e-4):
            raise RuntimeError(f"exact F000 top pressure {label} changed: {actual}")
    receipt = {
        "source": "released FP32 prepare_mpas_to_phys pres2_p[top]",
        "per_column_shape": [N_CELLS],
        "per_column_sha256": array_sha256(top),
        "per_column_minimum_pa": observed[0],
        "per_column_median_pa": observed[1],
        "per_column_maximum_pa": observed[2],
        "area_cell_sha256": array_sha256(area),
        "area_weighted_mean_f64_pa": weighted_mean64,
        "area_weighted_mean_f32_pa": float(scalar),
        "area_weighted_mean_f32_uint32": int(scalar.view(np.uint32)),
        "reduction": "sum(float64(p_top_column)*float64(areaCell))/sum(float64(areaCell))",
        "limitation": "native per-column plrad reduced to one fa35 constructor scalar",
        "claimed_native_identity": False,
        "cavallo_buffer_layers": 3,
    }
    return float(scalar), receipt


def build_arwen_constructor_values(
    *,
    init_path: Path,
    mesh: Any,
    vertical: Any,
    reference: Any,
    saved_diagnostics: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Map exact official x4 init carriers to the finalized fa35 constructor."""

    from netCDF4 import Dataset

    with Dataset(init_path, "r") as dataset:
        missing_species = sorted(
            name for name in COLD_ZERO_SCALAR_NAMES if name in dataset.variables
        )
        if missing_species:
            raise RuntimeError(
                "official cold-start assumption changed; unexpected variables "
                f"{missing_species}"
            )
        landmask_i = _read_exact_variable(
            dataset, "landmask", dtype=np.int32, dimensions=("nCells",)
        )
        ivgtyp = _read_exact_variable(
            dataset, "ivgtyp", dtype=np.int32, dimensions=("nCells",)
        )
        isltyp = _read_exact_variable(
            dataset, "isltyp", dtype=np.int32, dimensions=("nCells",)
        )
        xland_source = _read_exact_variable(
            dataset,
            "xland",
            dtype=np.float32,
            dimensions=("Time", "nCells"),
        )
        if xland_source.shape != (1, N_CELLS):
            raise ValueError(f"xland shape changed: {xland_source.shape}")
        xland = np.ascontiguousarray(xland_source[0], dtype=np.float32)
        zgrid = _read_exact_variable(
            dataset,
            "zgrid",
            dtype=np.float32,
            dimensions=("nCells", "nVertLevelsP1"),
        )
        surface: dict[str, np.ndarray] = {}
        for source, target in (
            ("vegfra", "vegfra"),
            ("skintemp", "tsk"),
            ("tmn", "tmn"),
            ("xice", "xice"),
            ("snow", "snow"),
            ("snowh", "snow_depth"),
        ):
            value = _read_exact_variable(
                dataset,
                source,
                dtype=np.float32,
                dimensions=("Time", "nCells"),
            )
            if value.shape != (1, N_CELLS):
                raise ValueError(f"{source} shape changed: {value.shape}")
            surface[target] = np.ascontiguousarray(value[0], dtype=np.float32)
        soil: dict[str, np.ndarray] = {}
        for source, target in (("tslb", "soil_temperature"), ("smois", "soil_moisture")):
            value = _read_exact_variable(
                dataset,
                source,
                dtype=np.float32,
                dimensions=("Time", "nCells", "nSoilLevels"),
            )
            if value.shape != (1, N_CELLS, N_SOIL_LEVELS):
                raise ValueError(f"{source} shape changed: {value.shape}")
            soil[target] = np.ascontiguousarray(value[0].T, dtype=np.float32)
        nominal_min_dc = _read_exact_variable(
            dataset, "nominalMinDc", dtype=np.float32, dimensions=()
        )
        start_text = str(getattr(dataset, "config_start_time", ""))

    source_xland_sha256 = array_sha256(xland_source)
    flat_xland_sha256 = array_sha256(xland)
    xland_unique, xland_counts = np.unique(xland, return_counts=True)
    if (
        source_xland_sha256 != NATIVE_XLAND_SOURCE_SHA256
        or flat_xland_sha256 != NATIVE_XLAND_FLAT_SHA256
        or tuple(float(value) for value in xland_unique) != (1.0, 2.0)
        or tuple(int(value) for value in xland_counts) != (61_528, 102_314)
    ):
        raise RuntimeError("sealed native xland identity changed")
    xice = surface["xice"]
    sea_ice_mask = np.ascontiguousarray(xice >= ARWEN_XICE_THRESHOLD)
    open_water_mask = np.ascontiguousarray(
        (xland >= np.float32(1.5)) & ~sea_ice_mask
    )
    land_mask = np.ascontiguousarray(~(sea_ice_mask | open_water_mask))
    glacier_mask = np.ascontiguousarray(land_mask & (ivgtyp == np.int32(15)))
    sflx_land_mask = np.ascontiguousarray(land_mask & ~glacier_mask)
    surface_classification = {
        "xland_source": "native",
        "xland_land_columns": int(np.count_nonzero(xland < np.float32(1.5))),
        "xland_water_columns": int(np.count_nonzero(xland >= np.float32(1.5))),
        "xice_threshold": float(ARWEN_XICE_THRESHOLD),
        "sea_ice_columns": int(np.count_nonzero(sea_ice_mask)),
        "open_water_columns": int(np.count_nonzero(open_water_mask)),
        "sflx_land_columns": int(np.count_nonzero(sflx_land_mask)),
        "glacier_columns": int(np.count_nonzero(glacier_mask)),
    }
    if surface_classification != dict(EXPECTED_SURFACE_CLASSIFICATION):
        raise RuntimeError(
            f"real-x4 surface classification changed: {surface_classification}"
        )
    glacier_indices = np.ascontiguousarray(np.flatnonzero(glacier_mask))
    sea_ice_indices = np.ascontiguousarray(np.flatnonzero(sea_ice_mask))
    threshold_delta_indices = np.ascontiguousarray(
        np.flatnonzero(
            (xice >= ARWEN_XICE_THRESHOLD) & (xice < np.float32(0.5))
        )
    )
    if (
        glacier_indices.size != 3_010
        or int(glacier_indices[0]) != 30
        or array_sha256(glacier_indices) != GLACIER_INDEX_SHA256
        or sea_ice_indices.size != 4_643
        or array_sha256(sea_ice_indices) != SEA_ICE_INDEX_SHA256
        or threshold_delta_indices.size != 1_067
        or array_sha256(threshold_delta_indices) != THRESHOLD_DELTA_INDEX_SHA256
    ):
        raise RuntimeError("real-x4 glacier/sea-ice index identity changed")
    if not (
        np.all(ivgtyp[sea_ice_mask] == np.int32(15))
        and np.all(landmask_i[sea_ice_mask] == np.int32(0))
        and np.all(xland[sea_ice_mask] == np.float32(1.0))
        and np.all(ivgtyp[glacier_mask] == np.int32(15))
        and np.all(landmask_i[glacier_mask] == np.int32(1))
        and np.all(xland[glacier_mask] == np.float32(1.0))
        and np.all(xice[glacier_mask] == np.float32(0.0))
    ):
        raise RuntimeError("real-x4 native xland/ice category signature changed")
    surface_classification_receipt = {
        **surface_classification,
        "source_field": "init xland[0,:]",
        "source_shape": list(xland_source.shape),
        "source_array_sha256": source_xland_sha256,
        "constructor_shape": list(xland.shape),
        "constructor_array_sha256": flat_xland_sha256,
        "first_glacier_column": int(glacier_indices[0]),
        "glacier_index_sha256": array_sha256(glacier_indices),
        "sea_ice_index_sha256": array_sha256(sea_ice_indices),
        "threshold_0p02_to_0p5_delta_columns": int(threshold_delta_indices.size),
        "threshold_delta_index_sha256": array_sha256(threshold_delta_indices),
        "native_xland_consumed_verbatim": True,
    }

    if np.float32(nominal_min_dc).view(np.uint32) != NOMINAL_DX_M.view(np.uint32):
        raise RuntimeError("official x4 nominalMinDc is not exact FP32 25000 m")
    if start_text != START_TIME_TEXT:
        raise RuntimeError(f"official init start time changed: {start_text!r}")
    landmask_pin = LANDMASK_CONSTRUCTOR_CAST_PIN
    source_landmask_sha256 = array_sha256(landmask_i)
    source_unique_values = tuple(int(value) for value in np.unique(landmask_i))
    if (
        landmask_i.dtype != np.dtype(landmask_pin["source_dtype"])
        or landmask_i.shape != tuple(landmask_pin["shape"])
        or source_landmask_sha256 != landmask_pin["source_array_sha256"]
        or source_unique_values != tuple(landmask_pin["source_unique_values"])
    ):
        raise RuntimeError("sealed init landmask source identity changed")
    landmask = np.ascontiguousarray(landmask_i, dtype=np.float32)
    target_landmask_sha256 = array_sha256(landmask)
    target_uint32_values = tuple(
        int(value) for value in np.unique(landmask.view(np.uint32))
    )
    if (
        landmask.dtype != np.dtype(landmask_pin["target_dtype"])
        or target_landmask_sha256 != landmask_pin["target_array_sha256"]
        or target_uint32_values != tuple(landmask_pin["target_uint32_values"])
        or not np.array_equal(landmask.astype(np.int32), landmask_i)
    ):
        raise RuntimeError("landmask int32 -> FP32 constructor cast is not exact")
    landmask_receipt = {
        "source_field": "init landmask",
        "source_dimensions": list(landmask_pin["source_dimensions"]),
        "source_shape": list(landmask_i.shape),
        "source_dtype": landmask_i.dtype.str,
        "source_array_sha256": source_landmask_sha256,
        "source_unique_values": list(source_unique_values),
        "target_field": "SealedArwenConstructorV841.landmask",
        "target_shape": list(landmask.shape),
        "target_dtype": landmask.dtype.str,
        "target_array_sha256": target_landmask_sha256,
        "target_uint32_values": list(target_uint32_values),
        "value_preserving_exact_fp32_cast": True,
    }

    if zgrid.shape != (N_CELLS, N_INTERFACES):
        raise ValueError(f"zgrid shape changed: {zgrid.shape}")

    lat = np.asarray(_mesh_value(mesh, "latCell"), dtype=np.float64)
    lon = np.asarray(_mesh_value(mesh, "lonCell"), dtype=np.float64)
    if lat.shape != (N_CELLS,) or lon.shape != (N_CELLS,):
        raise ValueError("reconciled mesh latitude/longitude shape changed")
    latitude_deg = np.ascontiguousarray(lat * (180.0 / np.pi), dtype=np.float32)
    longitude_deg = np.ascontiguousarray(lon * (180.0 / np.pi), dtype=np.float32)
    terrain = np.ascontiguousarray(zgrid[:, 0], dtype=np.float32)
    nominal_z = np.ascontiguousarray(np.asarray(vertical.zw))
    if nominal_z.shape != (N_INTERFACES,) or nominal_z.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        raise TypeError("loaded vertical.zw is not the exact 56-interface host vector")
    if not np.all(np.isfinite(nominal_z)) or np.any(np.diff(nominal_z) <= 0.0):
        raise ValueError("loaded vertical.zw is not finite and strictly increasing")

    # GF's per-cell length scale, built by native's own construction and the
    # same one this port already feeds GWDO: len_disp / meshDensity**0.25,
    # with a non-positive config_len_disp resolved to the mesh nominalMinDc.
    from mpas_port.cuda_gwdo_v841 import native_cell_dx_m

    dx_column_m = native_cell_dx_m(
        _mesh_value(mesh, "meshDensity"), float(NOMINAL_DX_M)
    )
    if dx_column_m.shape != (N_CELLS,):
        raise ValueError(
            f"per-cell GF dx must have shape {(N_CELLS,)}, got {dx_column_m.shape}"
        )

    p_top_pa, p_top_receipt = derive_area_weighted_p_top_v841(
        pressure_base=reference.pressure_base,
        pressure_perturbation=saved_diagnostics.pressure_perturbation,
        zgrid=vertical.zgrid,
        area_cell=np.ascontiguousarray(_mesh_value(mesh, "areaCell")),
    )

    values: dict[str, Any] = {
        "n_levels": N_LEVELS,
        "n_columns": N_CELLS,
        "dt": DT_SECONDS,
        "radiation_seconds": 600.0,
        "surface_pbl_seconds": DT_SECONDS,
        "cumulus_seconds": DT_SECONDS,
        "cumulus_scheme": "gf",
        "start_time": datetime.strptime(start_text, "%Y-%m-%d_%H:%M:%S"),
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "terrain_height_m": terrain,
        "z_interface_nominal_m": nominal_z,
        "p_top_pa": p_top_pa,
        "dx_m": float(NOMINAL_DX_M),
        "dx_column_m": dx_column_m,
        # Native MPAS v8.4.1 hardwires GF's shallow scheme on
        # (mpas_atmphys_vars.F:340).
        "gf_ishallow": 1,
        "landmask": landmask,
        "xland": xland,
        "xice_threshold": float(ARWEN_XICE_THRESHOLD),
        "ivgtyp": np.ascontiguousarray(ivgtyp, dtype=np.int32),
        "isltyp": np.ascontiguousarray(isltyp, dtype=np.int32),
        **surface,
        **soil,
        "wsm6_hail_opt": 0,
    }
    arrays = {name: value for name, value in values.items() if isinstance(value, np.ndarray)}
    receipt = {
        "source": "exact official x4.163842 initialized fields",
        "mapping": {
            "latitude_deg": "reconciled mesh latCell radians -> FP32 degrees",
            "longitude_deg": "reconciled mesh lonCell radians -> FP32 degrees",
            "terrain_height_m": "init zgrid[:,0]",
            "landmask": "init int32 {0,1} -> exact FP32 sealed-constructor cast",
            "xland": "init native xland[0,:] consumed verbatim",
            "xice_threshold": "explicit MPAS config_frac_seaice threshold 0.02",
            "dx_column_m": (
                "native per-cell GF length scale len_disp/meshDensity**0.25, "
                "config_len_disp=0 resolved to nominalMinDc"
            ),
            "z_interface_nominal_m": "loaded native vertical.zw",
            "tsk": "init skintemp[0,:]",
            "snow_depth": "init snowh[0,:] (m)",
            "soil_temperature": "init tslb[0,:,:].T",
            "soil_moisture": "init smois[0,:,:].T",
        },
        "p_top_pa": p_top_pa,
        "landmask_exact_cast": landmask_receipt,
        "surface_classification": surface_classification_receipt,
        "p_top_derivation": p_top_receipt,
        "p_top_policy": "exact areaCell-weighted F000 native pres2_p top",
        "dx_m": float(NOMINAL_DX_M),
        "dx_column_policy": "native len_disp/meshDensity**0.25 per cell",
        "dx_column_min_m": float(dx_column_m.min()),
        "dx_column_max_m": float(dx_column_m.max()),
        "gf_ishallow": 1,
        "defaults_used": False,
        "arrays": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        },
    }
    static_for_gwdo = {
        "meshDensity": np.asarray(_mesh_value(mesh, "meshDensity")),
        "nominalMinDc": np.asarray(NOMINAL_DX_M),
    }
    for name in ("var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4"):
        # These exact initialized statics are not necessarily retained on the
        # precision-preserving grid/static overlay, so read them from init.
        with Dataset(init_path, "r") as dataset:
            static_for_gwdo[name] = _read_exact_variable(
                dataset, name, dtype=np.float32, dimensions=("nCells",)
            )
    return values, receipt, static_for_gwdo


def _phase_from_receipt(backend: Any) -> str:
    receipt = dict(backend.step_receipt())
    phase = receipt.get("phase")
    if not isinstance(phase, str):
        raise RuntimeError("backend step receipt does not expose a phase string")
    return phase


def require_staged_backend_api(backend: Any) -> dict[str, Any]:
    required = (
        "begin_step",
        "finish_step",
        "commit_step",
        "abort_step",
        "restart_state",
        "restore_restart_state",
        "step_receipt",
        "diagnostic_snapshot",
    )
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise ReleaseNotFrozen(f"frozen adapter lacks staged API methods {missing}")
    phase = _phase_from_receipt(backend)
    if phase not in ("boundary", "restored", "complete"):
        raise RuntimeError(f"backend is not at a committed boundary: {phase!r}")
    return {"required_methods": list(required), "initial_phase": phase}


_BACKEND_TRANSACTION_IDENTITY_KEYS = (
    "schema",
    "adapter_contract_sha256",
    "coupling_contract_sha256",
    "contract_document_sha256",
    "contract_surface_sha256",
    "glacier_composed_tu_sha256",
    "arwen_commit",
    "arwen_source_manifest",
    "dependencies",
    "constructor",
)


def _backend_transaction_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: receipt[name]
        for name in _BACKEND_TRANSACTION_IDENTITY_KEYS
        if name in receipt
    }


def _verify_backend_rollback_boundary(
    backend: Any,
    *,
    expected_phase: str,
    start_time: float,
    transaction_identity: Mapping[str, Any],
) -> None:
    """Prove that an explicit or automatic adapter rollback reached its boundary."""

    receipt = dict(backend.step_receipt())
    phase = receipt.get("phase")
    if phase != expected_phase:
        raise RuntimeError(
            f"adapter rollback receipt phase {phase!r} != {expected_phase!r}"
        )
    if _backend_transaction_identity(receipt) != dict(transaction_identity):
        raise RuntimeError("adapter identity changed across the failed transaction")
    if float(receipt.get("start_time_seconds", math.nan)) != start_time:
        raise RuntimeError("adapter rollback receipt lost the exact step start time")

    # restart_state is legal only at an internal committed boundary. Requiring
    # its exact public schema prevents an automatic_rollback receipt alone
    # from being mistaken for a successfully restored seam.
    restart = backend.restart_state()
    if not isinstance(restart, Mapping) or set(restart) != {
        "schema",
        "identity",
        "seam",
        "adapter",
    }:
        raise RuntimeError("adapter rollback did not expose a boundary restart state")
    if restart["schema"] != receipt.get("schema"):
        raise RuntimeError("adapter rollback restart schema changed")
    restart_identity = restart["identity"]
    seam = restart["seam"]
    if not isinstance(restart_identity, Mapping) or not isinstance(seam, Mapping):
        raise RuntimeError("adapter rollback restart identity/state is malformed")
    if set(seam) != {"identity", "arrays", "scalars"}:
        raise RuntimeError("adapter rollback seam state schema changed")
    if not isinstance(seam["identity"], Mapping) or not isinstance(
        seam["scalars"], Mapping
    ):
        raise RuntimeError("adapter rollback seam identity/scalars are malformed")
    if float(seam["scalars"].get("elapsed_seconds", math.nan)) != start_time:
        raise RuntimeError("adapter rollback did not restore the exact boundary time")

    dependencies = receipt.get("dependencies", {})
    constructor = receipt.get("constructor", {})
    expected_restart_identity = {
        "adapter_contract_sha256": receipt.get("adapter_contract_sha256"),
        "arwen_commit": receipt.get("arwen_commit"),
        "arwen_source_manifest": receipt.get("arwen_source_manifest"),
        "contract_surface_sha256": receipt.get("contract_surface_sha256"),
        "glacier_composed_tu_sha256": receipt.get("glacier_composed_tu_sha256"),
        "constructor_identity_sha256": (
            constructor.get("identity_sha256")
            if isinstance(constructor, Mapping)
            else None
        ),
        "prep_contract_sha256": (
            dependencies.get("prep_contract_sha256")
            if isinstance(dependencies, Mapping)
            else None
        ),
        "gwdo_contract_sha256": (
            dependencies.get("gwdo_contract_sha256")
            if isinstance(dependencies, Mapping)
            else None
        ),
    }
    for name, expected in expected_restart_identity.items():
        if expected is not None and restart_identity.get(name) != expected:
            raise RuntimeError(
                f"adapter rollback restart identity changed field {name!r}"
            )


def execute_composite_step(
    *,
    driver: Any,
    backend: Any,
    scalar_names: Sequence[str],
    physics_geometry: Any,
    kernel_cache: Any,
    previous_surface_updates: Mapping[str, Any] | None,
    dynamics_tendencies: Any = None,
    couple: Callable[..., Any] | None = None,
    clamp: Callable[..., Any] | None = None,
    recover: Callable[..., Any] | None = None,
) -> CompositeStepResult:
    """Execute exactly one staged two-owner full-physics transaction."""

    if tuple(scalar_names) != SCALAR_NAMES:
        raise ValueError(f"exact scalar order is {SCALAR_NAMES}")
    if couple is None or clamp is None or recover is None:
        from mpas_port.cuda_physics_v841 import (
            clamp_wsm6_scalars_in_place_v841,
            couple_raw_column_physics_v841,
            recover_post_rk_wsm6_state_v841,
        )

        couple = couple or couple_raw_column_physics_v841
        clamp = clamp or clamp_wsm6_scalars_in_place_v841
        recover = recover or recover_post_rk_wsm6_state_v841

    candidate = None
    backend_active = False
    driver_committed = False
    start_atmosphere = driver.atmosphere
    start_state = start_atmosphere.state
    start_time = float(start_state.time_seconds)
    transaction_identity = _backend_transaction_identity(
        dict(backend.step_receipt())
    )
    try:
        raw = backend.begin_step(
            atmosphere=driver.atmosphere,
            scalar_names=scalar_names,
            dt=DT_SECONDS,
            dynamics_tendencies=dynamics_tendencies,
        )
        backend_active = True
        edge_fields = driver.horizontal.recover_edge_fields(
            start_state.rho, start_state.rho_u
        )
        held = couple(
            raw,
            state=start_state,
            scalar_names=scalar_names,
            geometry=physics_geometry,
            rho_edge=edge_fields.rho_edge,
            kernel_cache=kernel_cache,
        )
        candidate = driver.step_device_with_physics(held)
        # Read GF's next-step advective forcing here, BEFORE publication:
        # nothing after the driver commit is allowed to raise, and a dry
        # candidate legitimately carries none.  A driver that stops
        # producing the carrier mid-run is caught by the adapter's own
        # mid-run refusal at the next begin_step, not silently ignored.
        next_dynamics_tendencies = getattr(candidate, "dynamics_tendencies", None)
        clamp_d2h = clamp(
            candidate.atmosphere.state.scalars,
            scalar_names=scalar_names,
            kernel_cache=kernel_cache,
        )
        update = backend.finish_step(
            atmosphere=candidate.atmosphere,
            scalar_names=scalar_names,
            dt=DT_SECONDS,
        )
        if _phase_from_receipt(backend) != "finished_unpublished":
            raise RuntimeError("adapter finish did not remain staged and unpublished")
        recovery = recover(
            candidate.atmosphere.state,
            update,
            scalar_names=scalar_names,
            kernel_cache=kernel_cache,
            phase2_dt_seconds=DT_SECONDS,
            previous_surface_updates=previous_surface_updates,
        )
        committed = driver.commit_post_wsm6_candidate(candidate, recovery)
        driver_committed = True
        backend.commit_step()
        backend_active = False
        if float(committed.atmosphere.state.time_seconds) != start_time + DT_SECONDS:
            raise RuntimeError("composite transaction committed the wrong endpoint time")
        receipt = dict(backend.step_receipt())
        if receipt.get("phase") not in ("complete", "boundary"):
            raise RuntimeError("adapter commit did not publish a clean boundary")
        return CompositeStepResult(
            committed=committed,
            backend_receipt=MappingProxyType(receipt),
            clamp_d2h=clamp_d2h,
            recovery=recovery,
            dynamics_tendencies=next_dynamics_tendencies,
        )
    except Exception as error:
        rollback_errors: list[str] = []
        if candidate is not None and not driver_committed:
            try:
                driver.abort_post_wsm6_candidate(candidate)
                if (
                    driver.atmosphere is not start_atmosphere
                    or driver.atmosphere.state is not start_state
                    or float(driver.atmosphere.state.time_seconds) != start_time
                ):
                    raise RuntimeError(
                        "MPAS driver did not restore the exact start-state identity"
                    )
            except Exception as rollback_error:  # pragma: no cover - catastrophic path
                rollback_errors.append(f"driver abort failed: {rollback_error}")
        if backend_active:
            try:
                phase = _phase_from_receipt(backend)
                if phase in ("begun", "finished_unpublished"):
                    backend.abort_step()
                    expected_phase = "rolled_back"
                elif phase == "automatic_rollback":
                    expected_phase = "automatic_rollback"
                else:
                    raise RuntimeError(
                        "adapter failure left an unrecognized transaction phase "
                        f"{phase!r}"
                    )
                _verify_backend_rollback_boundary(
                    backend,
                    expected_phase=expected_phase,
                    start_time=start_time,
                    transaction_identity=transaction_identity,
                )
            except Exception as rollback_error:  # pragma: no cover - catastrophic path
                rollback_errors.append(
                    f"adapter rollback verification failed: {rollback_error}"
                )
        if driver_committed:
            raise CompositeTransactionError(
                "adapter commit failed after MPAS publication; this is an "
                "unrecoverable violation of the frozen no-fail commit contract"
            ) from error
        if rollback_errors:
            raise CompositeTransactionError(
                "composite step failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from error
        raise CompositeTransactionError(
            f"composite step at {start_time} s was aborted without publication"
        ) from error


def _host_array(value: Any, cp: Any) -> np.ndarray:
    if isinstance(value, cp.ndarray):
        return np.ascontiguousarray(cp.asnumpy(value))
    return np.ascontiguousarray(np.asarray(value))


def require_arwen_v2_surface_execution(
    payload: Mapping[str, Any], *, executed: bool, label: str
) -> dict[str, Any]:
    classification = payload.get("surface_classification")
    if not isinstance(classification, Mapping):
        raise ValueError(f"{label} lacks surface_classification")
    if dict(classification) != dict(EXPECTED_SURFACE_CLASSIFICATION):
        raise ValueError(f"{label} surface classification changed")
    census = payload.get("last_noahmp_census")
    if not executed:
        if census is not None:
            raise ValueError(f"{label} unexpectedly claims NoahMP execution")
        return {"surface_classification": dict(classification), "last_noahmp_census": None}
    if not isinstance(census, Mapping) or dict(census) != dict(EXPECTED_NOAHMP_CENSUS):
        raise ValueError(f"{label} NoahMP census/provenance changed")
    return {
        "surface_classification": dict(classification),
        "last_noahmp_census": dict(census),
    }


def backend_diagnostic_mapping(snapshot: Any) -> dict[str, Any]:
    """Normalize the released detached diagnostic dataclass without copies."""

    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    names = ("surface", "soil", "precipitation", "gwdo", "metadata", "receipt")
    missing = [name for name in names if not hasattr(snapshot, name)]
    if missing:
        raise TypeError(f"adapter diagnostic snapshot lacks groups {missing}")
    mapped = {name: getattr(snapshot, name) for name in names}
    if any(not isinstance(mapped[name], Mapping) for name in names):
        raise TypeError("adapter diagnostic snapshot groups must be mappings")
    return mapped


def _flatten_backend_diagnostics(snapshot: Mapping[str, Any], cp: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(f"{prefix}/{key}" if prefix else str(key), item)
            return
        if isinstance(value, cp.ndarray) or isinstance(value, np.ndarray):
            name = prefix.rsplit("/", 1)[-1]
            if name in arrays:
                raise ValueError(f"duplicate backend diagnostic leaf {name!r}")
            arrays[name] = _host_array(value, cp)
            return
        metadata[prefix] = value

    walk("", snapshot)
    return arrays, metadata


def capture_snapshot(
    *,
    label: str,
    step: int,
    driver: Any,
    backend: Any,
    prep_geometry: Any,
    kernel_cache: Any,
    f000_surface_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Download one committed diagnostic boundary outside step receipts."""

    import cupy as cp
    from mpas_port.cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841

    if label != SNAPSHOT_LABELS.get(step):
        raise ValueError("snapshot label/step mismatch")
    if float(driver.atmosphere.state.time_seconds) != step * DT_SECONDS:
        raise ValueError("snapshot model time does not equal its exact step")
    phase = _phase_from_receipt(backend)
    if phase not in ("complete", "boundary", "restored"):
        raise RuntimeError("snapshot is legal only at a committed backend boundary")
    prepared = prepare_mpas_to_phys_cuda_v841(
        driver.atmosphere,
        scalar_names=SCALAR_NAMES,
        geometry=prep_geometry,
        kernel_cache=kernel_cache,
        post_rk_wsm6=False,
    )
    state = driver.atmosphere.state
    saved = driver.atmosphere.saved
    arrays = {
        name: _host_array(state.scalars[index], cp)
        for index, name in enumerate(SCALAR_NAMES)
    }
    arrays.update(
        {
            "u_zonal": _host_array(prepared.u_p, cp),
            "v_meridional": _host_array(prepared.v_p, cp),
            "normal_u": _host_array(saved.normal_velocity, cp),
            "rho": _host_array(prepared.rho_dry, cp),
            "theta": _host_array(prepared.th_p, cp),
            "pressure": _host_array(prepared.pres_p, cp),
            "surface_pressure": _host_array(prepared.psfc_p, cp),
            "w": _host_array(saved.vertical_velocity, cp),
        }
    )
    backend_snapshot = backend_diagnostic_mapping(backend.diagnostic_snapshot())
    executed = step > 0
    require_arwen_v2_surface_execution(
        dict(backend.step_receipt()), executed=executed, label=f"{label} step receipt"
    )
    require_arwen_v2_surface_execution(
        backend_snapshot["metadata"], executed=executed, label=f"{label} metadata"
    )
    surface_execution = require_arwen_v2_surface_execution(
        backend_snapshot["receipt"], executed=executed, label=f"{label} receipt"
    )
    diagnostics, diagnostic_metadata = _flatten_backend_diagnostics(
        backend_snapshot, cp
    )
    arrays.update(diagnostics)
    f000_overlay = None
    if step == 0:
        if (
            set(f000_surface_diagnostics) != {"arrays", "receipt"}
            or not isinstance(f000_surface_diagnostics["arrays"], Mapping)
            or not isinstance(f000_surface_diagnostics["receipt"], Mapping)
        ):
            raise ValueError("F000 initialized surface diagnostic carrier is malformed")
        f000_overlay = {
            "source": dict(f000_surface_diagnostics["receipt"]),
            "overlay": overlay_f000_initialized_surface_diagnostics(
                arrays, f000_surface_diagnostics["arrays"]
            ),
        }
    required_diagnostics = {
        "tsk",
        "smois",
        "tslb",
        "hfx",
        "qfx",
        "lh",
        "t2",
        "u10",
        "v10",
        "rainc",
        "rainnc",
        "snownc",
        "graupelnc",
        "dusfcg",
        "dvsfcg",
        "dtaux3d",
        "dtauy3d",
        "rubldiff",
        "rvbldiff",
    }
    missing = sorted(required_diagnostics - set(arrays))
    if missing:
        raise RuntimeError(f"adapter diagnostic snapshot lacks required fields {missing}")
    cp.cuda.get_current_stream().synchronize()
    receipt = {
        "schema": SNAPSHOT_SCHEMA,
        "label": label,
        "step": step,
        "time_seconds": step * DT_SECONDS,
        "arwen_v2_surface_execution": surface_execution,
        "audit_d2h_outside_step_receipt": True,
        "prep": prepared.receipt(),
        "backend": dict(backend.step_receipt()),
        "backend_diagnostic_metadata": diagnostic_metadata,
        "f000_initialized_surface_diagnostics": f000_overlay,
        "arrays": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": array_sha256(value),
                "minimum": float(value.min()) if value.size else None,
                "maximum": float(value.max()) if value.size else None,
                "nonzero": int(np.count_nonzero(value)),
                "negative": int(np.count_nonzero(value < 0)) if value.dtype.kind == "f" else 0,
            }
            for name, value in sorted(arrays.items())
        },
    }
    return {"arrays": arrays, "receipt": receipt}


def physical_snapshot_gate(snapshot: Mapping[str, Any], *, allow_initial_negative_qv: bool) -> dict[str, Any]:
    arrays = snapshot["arrays"]
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise FloatingPointError(f"snapshot {name} is non-finite")
    if np.min(arrays["rho"]) <= 0.0 or np.min(arrays["theta"]) <= 0.0:
        raise FloatingPointError("snapshot rho/theta must be strictly positive")
    if np.min(arrays["pressure"]) <= 0.0 or np.min(arrays["surface_pressure"]) <= 0.0:
        raise FloatingPointError("snapshot pressure must be strictly positive")
    moisture_negative = {
        name: int(np.count_nonzero(np.asarray(arrays[name]) < 0.0))
        for name in SCALAR_NAMES
    }
    expected_qv = NEGATIVE_QV_PIN["negative_count"] if allow_initial_negative_qv else 0
    if moisture_negative["qv"] != expected_qv:
        raise FloatingPointError(
            f"qv negative count {moisture_negative['qv']} != {expected_qv}"
        )
    if any(moisture_negative[name] for name in SCALAR_NAMES[1:]):
        raise FloatingPointError("a hydrometeor snapshot contains negative values")
    smois = np.asarray(arrays["smois"])
    if float(smois.min()) < 0.0 or float(smois.max()) > 1.0:
        raise FloatingPointError("soil moisture lies outside [0,1]")
    for name in ("rainc", "rainnc", "snownc", "graupelnc"):
        if np.min(arrays[name]) < 0.0:
            raise FloatingPointError(f"negative accumulated precipitation in {name}")
    # Native q2 has 44 negative values.  It may be retained by the adapter
    # audit snapshot, but it is deliberately excluded from this gate.
    return {
        "finite_all_fields": True,
        "positive_rho_theta_pressure": True,
        "moisture_negative_counts": moisture_negative,
        "soil_moisture_range": [float(smois.min()), float(smois.max())],
        "precipitation_nonnegative": True,
        "q2_policy": "audit-only; not gated or publishable",
    }


def _native_logical_array(variable: Any, layout: str) -> np.ndarray:
    variable.set_auto_maskandscale(False)
    raw = np.ascontiguousarray(np.asarray(variable[...]))
    if raw.dtype != np.dtype(np.float32):
        raise TypeError(f"native {variable.name} is not FP32")
    if not np.all(np.isfinite(raw)):
        raise FloatingPointError(f"native {variable.name} is non-finite")
    if layout == "cell":
        if raw.shape != (1, N_CELLS):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0])
    if layout == "level_cell":
        if raw.shape != (1, N_CELLS, N_LEVELS):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0].T)
    if layout == "level_edge":
        if raw.shape != (1, N_EDGES, N_LEVELS):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0].T)
    if layout == "interface_cell":
        if raw.shape != (1, N_CELLS, N_INTERFACES):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0].T)
    if layout == "soil_cell":
        if raw.shape != (1, N_CELLS, N_SOIL_LEVELS):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0].T)
    raise KeyError(layout)


def comparison_metrics(cuda: Any, native: Any) -> dict[str, Any]:
    left = np.asarray(cuda)
    right = np.asarray(native)
    if left.shape != right.shape:
        raise ValueError(f"comparison shape mismatch {left.shape} != {right.shape}")
    l64 = left.astype(np.float64, copy=False)
    r64 = right.astype(np.float64, copy=False)
    difference = l64 - r64
    rmse = math.sqrt(float(np.mean(difference * difference, dtype=np.float64)))
    mae = float(np.mean(np.abs(difference), dtype=np.float64))
    authority_rms = math.sqrt(float(np.mean(r64 * r64, dtype=np.float64)))
    return {
        "count": int(left.size),
        "rmse": rmse,
        "mae": mae,
        "bias": float(np.mean(difference, dtype=np.float64)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "native_rms": authority_rms,
        "normalized_rmse": rmse / max(authority_rms, np.finfo(np.float64).tiny),
        "cuda_sha256": array_sha256(left),
        "native_sha256": array_sha256(right),
    }


def compare_snapshot_to_native(snapshot: Mapping[str, Any], native_path: Path) -> dict[str, Any]:
    from netCDF4 import Dataset

    results: dict[str, Any] = {}
    with Dataset(native_path, "r") as dataset:
        for cuda_name, (native_name, layout) in NATIVE_FIELD_MAP.items():
            if cuda_name not in snapshot["arrays"]:
                raise RuntimeError(f"CUDA snapshot lacks comparable field {cuda_name!r}")
            if native_name not in dataset.variables:
                raise RuntimeError(f"native authority lacks field {native_name!r}")
            native = _native_logical_array(dataset.variables[native_name], layout)
            metrics = comparison_metrics(snapshot["arrays"][cuda_name], native)
            limit = float(NATIVE_RMSE_LIMITS[cuda_name])
            metrics["rmse_limit"] = limit
            metrics["broad_consistency_gate"] = metrics["rmse"] <= limit
            if not metrics["broad_consistency_gate"]:
                raise FloatingPointError(
                    f"{cuda_name} native consistency RMSE {metrics['rmse']} exceeds {limit}"
                )
            results[cuda_name] = metrics
    return {
        "comparison_kind": "broad quantitative consistency, not parity",
        "native_path": str(native_path),
        "fields": results,
        "all_broad_consistency_gates_passed": True,
        "gf_native_parity_claim": False,
        # The claim names what makes it false, measured, so a reader does
        # not have to guess whether "false" means "unmeasured" or "known".
        # Task #231 closed the SEAM non-parity (the four auxiliary forcing
        # lanes, shallow-on, per-cell dx).  What remains is not a seam gap:
        # the port's GF body is WRF v4.6.1's Freitas-2018 generation while
        # v8.4.1's module_cu_gf.mpas.F is the 2013 ensemble fork -- native
        # has no dicycle and no tau_ecmwf closures at all, and runs c0=.002
        # with a NON-precipitating shallow scheme against the port's c0=.004
        # and a precipitating one.  This is a DECLARED DIVERGENCE, not a
        # blocker: physics parity against native was retired as a goal on
        # 2026-08-20, the verification of record is obs-skill (MRMS/ASOS),
        # and docs/declared-divergences.md carries mechanism, magnitude and
        # referee.  Closing it would mean porting native's cup_gf/cup_gf_sh
        # bodies, not tuning this seam -- and whether it should close is the
        # referee's call.
        "gf_declared_divergence": (
            "GF scheme generation differs: port body is WRF v4.6.1 "
            "(dicycle + tau_ecmwf closures, c0=.004, precipitating shallow); "
            "native v8.4.1 module_cu_gf.mpas.F is the 2013 ensemble GF "
            "(Fritsch-Chappell AA0/1200s members, no dicycle, c0=.002, "
            "non-precipitating shallow). Declared divergence judged by "
            "obs-skill, not a parity blocker; see docs/declared-divergences.md"
        ),
    }


def fingerprint_nested_arrays(value: Any, *, prefix: str = "") -> dict[str, Any]:
    """Hash a detached restart/diagnostic payload without JSON-coercing arrays."""

    fields: dict[str, Any] = {}
    scalars: dict[str, Any] = {}

    def walk(path: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item):
                walk(f"{path}/{key}" if path else str(key), item[key])
        elif isinstance(item, np.ndarray):
            fields[path] = {
                "dtype": item.dtype.str,
                "shape": list(item.shape),
                "sha256": array_sha256(item),
            }
        elif isinstance(item, (str, int, float, bool)) or item is None:
            scalars[path] = item
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                walk(f"{path}/{index}", child)
        else:
            scalars[path] = repr(item)

    walk(prefix, value)
    core = {"arrays": fields, "scalars": scalars}
    return {**core, "sha256": canonical_json_sha256(core)}


def _fingerprint_leaf_projection(value: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten exact fingerprint leaves while omitting redundant group digests."""

    leaves: dict[str, Any] = {}

    def walk(path: str, item: Any) -> None:
        if isinstance(item, Mapping):
            has_nested_mapping = any(
                isinstance(child, Mapping) for child in item.values()
            )
            for key in sorted(item):
                if key == "sha256" and has_nested_mapping:
                    continue
                walk(f"{path}/{key}" if path else str(key), item[key])
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                walk(f"{path}/{index}" if path else str(index), child)
        else:
            leaves[path] = item

    walk(prefix, value)
    return leaves


def require_fingerprint_identity(
    label: str,
    uninterrupted: Mapping[str, Any],
    restored: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact identity and name every differing numerical leaf."""

    direct = _fingerprint_leaf_projection(uninterrupted)
    resumed = _fingerprint_leaf_projection(restored)
    missing = object()
    mismatches = sorted(
        path
        for path in set(direct) | set(resumed)
        if direct.get(path, missing) != resumed.get(path, missing)
    )
    if mismatches:
        raise RuntimeError(f"{label} differs at fingerprint paths {mismatches}")
    return {
        "bitwise_identical": True,
        "leaf_count": len(direct),
        "sha256": uninterrupted["sha256"],
    }


def fingerprint_execution_boundary(stack: Mapping[str, Any]) -> dict[str, Any]:
    """Hash all mutable MPAS and Arwen carriers at one committed boundary."""

    from mpas_port.cuda_dualrun import fingerprint_atmosphere

    return {
        "atmosphere": fingerprint_atmosphere(stack["driver"].atmosphere),
        "backend": fingerprint_nested_arrays(stack["backend"].restart_state()),
    }


def download_driver_checkpoint(
    driver: Any, backend: Any, *, dynamics_tendencies: Any
) -> HostDriverCheckpoint:
    if dynamics_tendencies is None:
        raise RuntimeError(
            "F030 checkpoint refused: the driver's GF advective-forcing "
            "carrier (rthdynten/rqvdynten) is absent.  Every full-physics "
            "step forms it, and a checkpoint written without it resumes by "
            "feeding GF zero forcing lanes at the first resumed step -- the "
            "deterministic step-16 restart divergence (#327)."
        )

    from mpas_port.cuda_dualrun import fingerprint_atmosphere
    from mpas_port.driver import DrySavedDiagnostics

    cp = driver.cp
    if _phase_from_receipt(backend) not in ("complete", "boundary"):
        raise RuntimeError("F030 checkpoint requires a committed backend boundary")
    cp.cuda.get_current_stream().synchronize()
    atmosphere_fingerprint = fingerprint_atmosphere(driver.atmosphere)
    state = driver.atmosphere.state.to_host()
    saved = driver.atmosphere.saved
    host_saved = DrySavedDiagnostics(
        theta_m=cp.asnumpy(saved.theta_m),
        exner=cp.asnumpy(saved.exner),
        density_perturbation=cp.asnumpy(saved.density_perturbation),
        rho_theta_perturbation=cp.asnumpy(saved.rho_theta_perturbation),
        pressure_perturbation=cp.asnumpy(saved.pressure_perturbation),
        normal_velocity=cp.asnumpy(saved.normal_velocity),
        vertical_velocity=cp.asnumpy(saved.vertical_velocity),
    )
    cp.cuda.get_current_stream().synchronize()
    if float(state.time_seconds) != CHECKPOINT_STEP * DT_SECONDS:
        raise RuntimeError("driver checkpoint is not at exact F030")
    host_atmosphere_fingerprint = fingerprint_atmosphere(
        SimpleNamespace(state=state, saved=host_saved)
    )
    require_fingerprint_identity(
        "F030 device-to-host MPAS checkpoint",
        atmosphere_fingerprint,
        host_atmosphere_fingerprint,
    )
    backend_state = backend.restart_state()
    backend_fingerprint = fingerprint_nested_arrays(backend_state)
    forcing_time = float(dynamics_tendencies.time_seconds)
    if forcing_time != float(state.time_seconds):
        raise RuntimeError(
            "GF advective forcing at the checkpoint must be the pair the "
            "checkpoint step's dynamics formed (stamped with its endpoint "
            f"time {float(state.time_seconds)} s); got t={forcing_time} s"
        )
    gf_host = {
        "rthdynten": np.ascontiguousarray(
            cp.asnumpy(dynamics_tendencies.rthdynten)
        ),
        "rqvdynten": np.ascontiguousarray(
            cp.asnumpy(dynamics_tendencies.rqvdynten)
        ),
        "time_seconds": forcing_time,
    }
    return HostDriverCheckpoint(
        state=state,
        saved_diagnostics=host_saved,
        backend_state=backend_state,
        atmosphere_fingerprint=atmosphere_fingerprint,
        backend_fingerprint=backend_fingerprint,
        model_time_seconds=float(state.time_seconds),
        gf_dynamics_tendencies=gf_host,
        gf_forcing_fingerprint=fingerprint_nested_arrays(gf_host),
    )


def _snapshot_hash_projection(snapshot: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: array_sha256(value)
        for name, value in sorted(snapshot["arrays"].items())
        if name != "q2"
    }


def require_bitwise_restart_identity(
    uninterrupted: Mapping[str, Any], restored: Mapping[str, Any]
) -> dict[str, Any]:
    direct = _snapshot_hash_projection(uninterrupted)
    resumed = _snapshot_hash_projection(restored)
    if direct != resumed:
        mismatches = sorted(
            name for name in set(direct) | set(resumed) if direct.get(name) != resumed.get(name)
        )
        raise RuntimeError(f"F030 restart continuation differs at F001 fields {mismatches}")
    return {
        "bitwise_identical": True,
        "field_count": len(direct),
        "fields": direct,
        "q2_excluded_from_publication_projection": True,
    }


def surface_evolution_gate(f000: Mapping[str, Any], f001: Mapping[str, Any]) -> dict[str, Any]:
    names = ("tsk", "smois", "tslb", "hfx", "qfx", "lh")
    evolved = {
        name: array_sha256(f000["arrays"][name]) != array_sha256(f001["arrays"][name])
        for name in names
    }
    if not any(evolved.values()):
        raise RuntimeError("surface and soil state did not evolve through F001")
    precipitation = sum(
        int(np.count_nonzero(f001["arrays"][name]))
        for name in ("rainc", "rainnc", "snownc", "graupelnc")
    )
    if precipitation == 0:
        raise RuntimeError("one-hour full-physics forecast produced no precipitation signal")
    gwd = {
        group: sum(int(np.count_nonzero(f001["arrays"][name])) for name in names_)
        for group, names_ in {
            "surface_stress": ("dusfcg", "dvsfcg"),
            "column_tendency": ("dtaux3d", "dtauy3d"),
            "pbl_increment": ("rubldiff", "rvbldiff"),
        }.items()
    }
    if any(count == 0 for count in gwd.values()):
        raise RuntimeError(f"external YSU-GWDO activity is absent: {gwd}")
    return {
        "surface_soil_evolved": evolved,
        "precipitation_nonzero_elements": precipitation,
        "gwd_nonzero_elements": gwd,
    }


def _write_exclusive_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def write_snapshot_netcdf(path: Path, snapshot: Mapping[str, Any], static: Mapping[str, Any]) -> dict[str, Any]:
    """Write one truthful native-grid CUDA history capsule for Rust rendering."""

    from netCDF4 import Dataset

    if path.exists():
        raise FileExistsError(path)
    arrays = snapshot["arrays"]
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("nCells", N_CELLS)
        dataset.createDimension("nVertLevels", N_LEVELS)
        dataset.createDimension("nEdges", N_EDGES)
        dataset.createDimension("nVertLevelsP1", N_INTERFACES)
        dataset.createDimension("nSoilLevels", N_SOIL_LEVELS)
        dataset.setncattr("schema", SNAPSHOT_SCHEMA)
        dataset.setncattr("source", "MPAS-A v8.4.1 full-physics CUDA port")
        dataset.setncattr("arwen_commit", ARWEN_COMMIT)
        dataset.setncattr("gf_native_parity_claim", "false")
        dataset.setncattr(
            "gf_declared_divergence",
            "scheme generation: port GF is WRF v4.6.1, native v8.4.1 is the "
            "2013 ensemble fork; declared divergence judged by obs-skill "
            "(see gf_declared_divergence in the run receipt and "
            "docs/declared-divergences.md)",
        )
        dataset.setncattr("gf_seam_parity_closed", "task-231 forcing-lanes+shallow+per-cell-dx")
        dataset.setncattr("q2_products_allowed", "false")
        for name, dtype in (("indexToCellID", "i4"), ("latCell", "f4"), ("lonCell", "f4"), ("ter", "f4")):
            variable = dataset.createVariable(name, dtype, ("nCells",))
            variable[:] = static[name]
        for name, dtype in (("indexToEdgeID", "i4"), ("latEdge", "f4"), ("lonEdge", "f4")):
            variable = dataset.createVariable(name, dtype, ("nEdges",))
            variable[:] = static[name]
        units = {
            **{name: "kg kg^{-1}" for name in SCALAR_NAMES},
            "u_zonal": "m s^{-1}",
            "v_meridional": "m s^{-1}",
            "normal_u": "m s^{-1}",
            "rho": "kg m^{-3}",
            "theta": "K",
            "pressure": "Pa",
            "surface_pressure": "Pa",
            "w": "m s^{-1}",
            "tsk": "K",
            "t2": "K",
            "hfx": "W m^{-2}",
            "qfx": "kg m^{-2} s^{-1}",
            "lh": "W m^{-2}",
            "u10": "m s^{-1}",
            "v10": "m s^{-1}",
            "smois": "m3 m^{-3}",
            "tslb": "K",
            "rainc": "mm",
            "rainnc": "mm",
            "snownc": "mm",
            "graupelnc": "mm",
            "dusfcg": "Pa",
            "dvsfcg": "Pa",
            "dtaux3d": "m s^{-2}",
            "dtauy3d": "m s^{-2}",
            "rubldiff": "m s^{-2}",
            "rvbldiff": "m s^{-2}",
        }
        for name, value in sorted(arrays.items()):
            if name == "q2":
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.shape == (N_LEVELS, N_CELLS):
                dims = ("Time", "nCells", "nVertLevels")
                payload = array.T[None]
            elif array.shape == (N_LEVELS, N_EDGES):
                dims = ("Time", "nEdges", "nVertLevels")
                payload = array.T[None]
            elif array.shape == (N_INTERFACES, N_CELLS):
                dims = ("Time", "nCells", "nVertLevelsP1")
                payload = array.T[None]
            elif array.shape == (N_SOIL_LEVELS, N_CELLS):
                dims = ("Time", "nCells", "nSoilLevels")
                payload = array.T[None]
            elif array.shape == (N_CELLS,):
                dims = ("Time", "nCells")
                payload = array[None]
            else:
                continue
            variable = dataset.createVariable(name, "f4", dims, zlib=True, complevel=1)
            variable.setncattr("units", units.get(name, ""))
            variable[:] = payload
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def gpu_memory_admission(cp: Any, *, minimum: int = MIN_FREE_DEVICE_BYTES) -> dict[str, Any]:
    free, total = map(int, cp.cuda.runtime.memGetInfo())
    if free < minimum:
        raise MemoryError(
            f"full-physics x4 requires at least {minimum} free device bytes; got {free}"
        )
    return {"free_bytes": free, "total_bytes": total, "minimum": minimum, "admitted": True}


def _prepare_host_execution(paths: Mapping[str, Path], authority_receipt: Mapping[str, Any]) -> dict[str, Any]:
    from mpas_port.config_v841 import V841MpasColumnPhysicsGwdoConfig
    from mpas_port.cuda_arwen_physics_v841 import SealedArwenConstructorV841
    from mpas_port.cuda_dualrun import PreparedCudaInputs
    from mpas_port.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from mpas_port.dynamics_v841 import load_v841_reference_wind_profiles
    from mpas_port.mesh import load_precision_preserving_mesh_pair

    config = V841MpasColumnPhysicsGwdoConfig()
    config.validate()
    mesh, output_mesh, mesh_evidence = load_precision_preserving_mesh_pair(
        paths["grid"], paths["static"]
    )
    del output_mesh
    reconstruction_overlay = overlay_exact_init_reconstruction_coefficients(
        mesh, paths["init"]
    )
    edge_normal_overlay = overlay_exact_init_edge_normal_vectors(
        mesh,
        grid_path=paths["grid"],
        static_path=paths["static"],
        init_path=paths["init"],
    )
    defc = attach_inactive_zero_deformation(mesh)
    native = load_mpas_vertical_grid(
        paths["init"], mesh, config_coef_3rd_order=config.config_coef_3rd_order
    )
    state, reference, saved = load_mpas_initial_state(
        paths["init"],
        mesh,
        native.vertical_grid,
        scalar_names=SOURCE_SCALAR_NAMES,
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
    )
    scalar_receipt = augment_exact_wsm6_scalars(state)
    state.validate(n_cells=N_CELLS, n_edges=N_EDGES, n_vert_levels=N_LEVELS)
    saved.validate((N_LEVELS, N_CELLS), np.dtype(np.float32), N_EDGES)
    profiles = load_v841_reference_wind_profiles(paths["init"], n_vert_levels=N_LEVELS)
    prepared = PreparedCudaInputs.validated(
        config=config,
        profile=PROFILE,
        target=CLAIM,
        preparation_method=(
            "precision-preserving grid/static overlay plus exact initialized "
            "reconstruction coefficients and edge-normal vectors; qv/qc/qr "
            "plus exact +0 qi/qs/qg"
        ),
        mesh=mesh,
        state=state,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
        terrain_metrics=native.terrain_metrics,
        input_bytes=dict(authority_receipt["files"]),
        reference_wind_profiles=profiles,
    )
    f000_surface_diagnostics = load_f000_initialized_surface_diagnostics(
        paths["init"]
    )
    constructor_values, constructor_receipt, gwdo_host = build_arwen_constructor_values(
        init_path=paths["init"],
        mesh=mesh,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
    )
    sealed_constructor_audit = SealedArwenConstructorV841.from_mapping(
        constructor_values
    )
    constructor_receipt["sealed_host_contract_audit"] = {
        "authority": "SealedArwenConstructorV841.from_mapping",
        "accepted": True,
        "all_required_keys_dtypes_shapes_validated": True,
        "array_fields": sorted(constructor_receipt["arrays"]),
        **dict(sealed_constructor_audit.receipt()),
    }
    return {
        "config": config,
        "prepared": prepared,
        "constructor_values": constructor_values,
        "constructor_receipt": constructor_receipt,
        "gwdo_host": gwdo_host,
        "mesh_evidence": mesh_evidence,
        "f000_surface_diagnostics": f000_surface_diagnostics,
        "reconstruction_coefficients": reconstruction_overlay,
        "edge_normal_vectors": edge_normal_overlay,
        "defc": defc,
        "scalar_receipt": scalar_receipt,
    }


def _construct_device_stack(
    *,
    host: Mapping[str, Any],
    cache: Any,
    arwen_checkout: Path,
    state: Any | None = None,
    saved_diagnostics: Any | None = None,
    backend_restart: Mapping[str, Any] | None = None,
    gf_dynamics_tendencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from mpas_port.cuda_arwen_physics_v841 import (
        PersistentTwoPhaseCudaPhysicsBackendV841,
        SealedArwenConstructorV841,
    )
    from mpas_port.cuda_driver import CudaDryDycoreDriver
    from mpas_port.cuda_gwdo_v841 import CudaYsuGwdoStaticV841
    from mpas_port.cuda_physics_prep_v841 import CudaMpasToPhysGeometryV841
    from mpas_port.cuda_physics_v841 import CudaPhysicsGeometryV841

    prepared = host["prepared"]
    selected_state = prepared.state if state is None else state
    selected_saved = prepared.saved_diagnostics if saved_diagnostics is None else saved_diagnostics
    driver = CudaDryDycoreDriver.from_host(
        prepared.mesh,
        selected_state,
        prepared.vertical,
        prepared.reference,
        host["config"],
        saved_diagnostics=selected_saved,
        terrain_metrics=prepared.terrain_metrics,
        advection_coefficients=prepared.advection_coefficients,
        kernel_cache=cache,
        reference_wind_profiles=prepared.reference_wind_profiles,
    )
    prep_geometry = CudaMpasToPhysGeometryV841.from_host(prepared.mesh)
    physics_geometry = CudaPhysicsGeometryV841.from_host(prepared.mesh)
    gwdo_static = CudaYsuGwdoStaticV841.from_host(host["gwdo_host"])
    constructor = SealedArwenConstructorV841.from_mapping(host["constructor_values"])
    backend = PersistentTwoPhaseCudaPhysicsBackendV841(
        constructor=constructor,
        prep_geometry=prep_geometry,
        kernel_cache=cache,
        gwdo_static=gwdo_static,
        gwdo_kernel_cache=cache,
        arwen_checkout=arwen_checkout,
    )
    api = require_staged_backend_api(backend)
    if backend_restart is not None:
        backend.restore_restart_state(backend_restart)
    diagnostic = backend_diagnostic_mapping(backend.diagnostic_snapshot())
    if float(driver.atmosphere.state.time_seconds) != float(diagnostic["metadata"]["time_seconds"]):
        raise RuntimeError("driver and public backend diagnostic clocks differ")
    stack: dict[str, Any] = {
        "driver": driver,
        "backend": backend,
        "prep_geometry": prep_geometry,
        "physics_geometry": physics_geometry,
        "gwdo_static": gwdo_static,
        "constructor": constructor,
        "api": api,
        "f000_surface_diagnostics": host["f000_surface_diagnostics"],
        "initial_diagnostic": diagnostic,
    }
    if gf_dynamics_tendencies is not None:
        # Restore side of #327: re-seed the driver-owned GF advective-forcing
        # carrier the checkpoint step's dynamics formed, so the FIRST resumed
        # step's begin_step consumes exactly the pair the unbroken run's next
        # step consumes.  Without this the resumed step 16 runs GF on zero
        # rthdynten/rqvdynten lanes and every downstream field diverges.
        from mpas_port.cuda_driver import CudaV841GfDynamicsTendencies

        cp = driver.cp
        carrier = CudaV841GfDynamicsTendencies(
            rthdynten=cp.asarray(
                np.ascontiguousarray(
                    gf_dynamics_tendencies["rthdynten"], dtype=np.float32
                )
            ),
            rqvdynten=cp.asarray(
                np.ascontiguousarray(
                    gf_dynamics_tendencies["rqvdynten"], dtype=np.float32
                )
            ),
            time_seconds=float(gf_dynamics_tendencies["time_seconds"]),
        )
        carrier.validate(cp=cp, n_vert_levels=N_LEVELS, n_cells=N_CELLS)
        if float(carrier.time_seconds) != float(
            driver.atmosphere.state.time_seconds
        ):
            raise RuntimeError(
                "restored GF advective forcing must be stamped with the "
                "restored model time "
                f"{float(driver.atmosphere.state.time_seconds)} s; got "
                f"t={float(carrier.time_seconds)} s"
            )
        stack["gf_dynamics_tendencies"] = carrier
    return stack


def _previous_surface_updates(stack: Mapping[str, Any]) -> Mapping[str, Any] | None:
    snapshot = backend_diagnostic_mapping(stack["backend"].diagnostic_snapshot())
    # Adapter diagnostic schema may group these under precipitation.  Walk by
    # leaf name without downloading the resident arrays.
    leaves: dict[str, Any] = {}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in ("rainnc", "snownc", "graupelnc"):
                    leaves[key] = item
                else:
                    walk(item)

    walk(snapshot)
    return leaves if set(leaves) == {"rainnc", "snownc", "graupelnc"} else None


def _run_steps(
    *,
    stack: Mapping[str, Any],
    start_step: int,
    end_step: int,
    capture_steps: set[int],
    boundary_observer: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[int, dict[str, Any]], Mapping[str, Any] | None, list[dict[str, Any]]]:
    snapshots: dict[int, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    previous = _previous_surface_updates(stack)
    if start_step in capture_steps:
        snapshots[start_step] = capture_snapshot(
            label=SNAPSHOT_LABELS[start_step],
            step=start_step,
            driver=stack["driver"],
            backend=stack["backend"],
            prep_geometry=stack["prep_geometry"],
            kernel_cache=stack["driver"].cache,
            f000_surface_diagnostics=stack["f000_surface_diagnostics"],
        )
    # GF advective forcing carried step to step.  None at the first step of
    # this leg is native's own start state (tend_physics is zero before the
    # first dynamics step forms it).
    gf_dynamics_tendencies = stack.get("gf_dynamics_tendencies")
    for step in range(start_step + 1, end_step + 1):
        result = execute_composite_step(
            driver=stack["driver"],
            backend=stack["backend"],
            scalar_names=SCALAR_NAMES,
            physics_geometry=stack["physics_geometry"],
            kernel_cache=stack["driver"].cache,
            previous_surface_updates=previous,
            dynamics_tendencies=gf_dynamics_tendencies,
        )
        gf_dynamics_tendencies = result.dynamics_tendencies
        stack["gf_dynamics_tendencies"] = gf_dynamics_tendencies
        previous = result.committed.surface_updates
        backend_receipt = dict(result.backend_receipt)
        surface_execution = require_arwen_v2_surface_execution(
            backend_receipt, executed=True, label=f"step {step} backend receipt"
        )
        receipts.append(
            {
                "step": step,
                "driver": asdict(result.committed.receipt),
                "backend": backend_receipt,
                "arwen_v2_surface_execution": surface_execution,
                "clamp_d2h": result.clamp_d2h.as_dict(),
                "recovery": result.recovery.receipt(),
            }
        )
        if boundary_observer is not None:
            boundary_observer(step, stack)
        if step in capture_steps:
            snapshots[step] = capture_snapshot(
                label=SNAPSHOT_LABELS[step],
                step=step,
                driver=stack["driver"],
                backend=stack["backend"],
                prep_geometry=stack["prep_geometry"],
                kernel_cache=stack["driver"].cache,
                f000_surface_diagnostics=stack["f000_surface_diagnostics"],
            )
    return snapshots, previous, receipts


def _static_output_fields(host: Mapping[str, Any]) -> dict[str, np.ndarray]:
    mesh = host["prepared"].mesh
    return {
        "indexToCellID": np.ascontiguousarray(_mesh_value(mesh, "indexToCellID"), dtype=np.int32),
        "indexToEdgeID": np.ascontiguousarray(_mesh_value(mesh, "indexToEdgeID"), dtype=np.int32),
        "latEdge": np.ascontiguousarray(_mesh_value(mesh, "latEdge"), dtype=np.float32),
        "lonEdge": np.ascontiguousarray(_mesh_value(mesh, "lonEdge"), dtype=np.float32),
        "latCell": np.ascontiguousarray(_mesh_value(mesh, "latCell"), dtype=np.float32),
        "lonCell": np.ascontiguousarray(_mesh_value(mesh, "lonCell"), dtype=np.float32),
        "ter": np.ascontiguousarray(host["constructor_values"]["terrain_height_m"], dtype=np.float32),
    }




def require_snapshot_receipt_surface_execution(
    snapshot_receipt: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Validate the exact capture receipt and its nested Arwen execution payload."""

    if label not in ("F000", "F030", "F001"):
        raise ValueError(f"unsupported baseline snapshot label {label!r}")
    step = {"F000": 0, "F030": CHECKPOINT_STEP, "F001": FULL_STEPS}[label]
    if (
        snapshot_receipt.get("schema") != SNAPSHOT_SCHEMA
        or snapshot_receipt.get("label") != label
        or snapshot_receipt.get("step") != step
        or snapshot_receipt.get("time_seconds") != step * DT_SECONDS
    ):
        raise ValueError(f"baseline diagnostic {label} capture receipt changed")
    nested = snapshot_receipt.get("arwen_v2_surface_execution")
    if not isinstance(nested, Mapping):
        raise ValueError(
            f"baseline diagnostic {label} lacks nested Arwen surface execution"
        )
    return require_arwen_v2_surface_execution(
        nested, executed=(step > 0), label=f"baseline diagnostic {label}"
    )


def _write_baseline_diagnostic_output(
    *,
    diagnostic_root: Path,
    host: Mapping[str, Any],
    baseline_snapshots: Mapping[int, Mapping[str, Any]],
    step_receipts: Sequence[Mapping[str, Any]],
    atmosphere_fingerprint: Mapping[str, Any],
    backend_fingerprint: Mapping[str, Any],
    physical_gates: Mapping[str, Any],
    weather_activity: Mapping[str, Any],
    native_comparisons: Mapping[str, Any],
    capability: Mapping[str, Any],
    memory_admission: Mapping[str, Any],
    arwen_pin: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish plot-capable baseline bytes without weakening release output."""

    classification = dict(host["constructor_receipt"]["surface_classification"])
    classification_core = {
        name: classification[name] for name in EXPECTED_SURFACE_CLASSIFICATION
    }
    if classification_core != dict(EXPECTED_SURFACE_CLASSIFICATION):
        raise RuntimeError("baseline diagnostic surface classification changed")
    snapshot_receipts = {
        SNAPSHOT_LABELS[step]: baseline_snapshots[step]["receipt"]
        for step in SNAPSHOT_STEPS
    }
    surface_execution = {
        label: require_snapshot_receipt_surface_execution(
            snapshot_receipts[label], label=label
        )
        for label in ("F000", "F030", "F001")
    }
    static = _static_output_fields(host)
    files = {
        label: write_snapshot_netcdf(
            diagnostic_root / SNAPSHOT_FILE_NAMES[label],
            baseline_snapshots[step],
            static,
        )
        for step, label in (
            (0, "F000"),
            (CHECKPOINT_STEP, "F030"),
            (FULL_STEPS, "F001"),
        )
    }
    proof = {
        "full_physics_cuda_executed": True,
        "uninterrupted_baseline_passed": True,
        "outer_steps": FULL_STEPS,
        "forecast_seconds": FULL_STEPS * DT_SECONDS,
        "restart_bitwise_identical": None,
        "checkpoint_restart": {
            "evaluated": False,
            "passed": False,
            "status": "not_evaluated",
        },
        "source_pins": source_receipt,
        "authority": authority_receipt,
        "capability": capability,
        "memory_admission": memory_admission,
        "arwen_pre_kernel_cache_pin": arwen_pin,
        "host_preparation": {
            "constructor": host["constructor_receipt"],
            "scalars": host["scalar_receipt"],
            "mesh_overlay": host["mesh_evidence"],
            "reconstruction_coefficients": host["reconstruction_coefficients"],
            "edge_normal_vectors": host["edge_normal_vectors"],
            "inactive_deformation": host["defc"],
        },
        "surface_classification": classification_core,
        "glacier_cuda_evidence": {
            "required_path": ARWEN_GLACIER_CUDA_PROVENANCE,
            "expected_census": dict(EXPECTED_NOAHMP_CENSUS),
            "snapshot_execution": surface_execution,
            "contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
            "glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
        },
        "uninterrupted": {
            "steps": FULL_STEPS,
            "step_receipts": list(step_receipts),
            "snapshot_receipts": snapshot_receipts,
            "f001_full_state_fingerprints": {
                "atmosphere": atmosphere_fingerprint,
                "backend": backend_fingerprint,
            },
        },
        "physical_gates": physical_gates,
        "weather_activity": weather_activity,
        "native_comparisons": native_comparisons,
        "snapshot_files": files,
    }
    payload = {
        "schema": BASELINE_DIAGNOSTIC_SCHEMA,
        "status": BASELINE_DIAGNOSTIC_STATUS,
        "warning": BASELINE_DIAGNOSTIC_WARNING,
        "release_eligible": False,
        "source_release": SOURCE_RELEASE,
        "arwen_commit": ARWEN_COMMIT,
        "arwen_contract_document_sha256": ARWEN_CONTRACT_DOCUMENT_SHA256,
        "arwen_contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
        "arwen_glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
        "claim": "uninterrupted 30-step engineering baseline only",
        "nonclaims": [
            *NONCLAIMS,
            "checkpoint/restart identity has not been evaluated or passed",
            "this diagnostic is not a release proof",
        ],
        "proof": proof,
        "weather_plot_policy": "native Rust/Arwen renderer only; q2 forbidden",
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    receipt = diagnostic_root / BASELINE_DIAGNOSTIC_RECEIPT_NAME
    _write_exclusive_json(receipt, payload)
    expected_inventory = {
        BASELINE_DIAGNOSTIC_RECEIPT_NAME,
        *(SNAPSHOT_FILE_NAMES[label] for label in ("F000", "F030", "F001")),
    }
    if {path.name for path in diagnostic_root.iterdir()} != expected_inventory:
        raise RuntimeError("baseline diagnostic output inventory changed")
    return {
        "path": str(receipt),
        "bytes": receipt.stat().st_size,
        "sha256": sha256_file(receipt),
        "payload_sha256": payload["payload_sha256"],
        "status": BASELINE_DIAGNOSTIC_STATUS,
        "release_eligible": False,
    }


def _execute_full_proof(
    *,
    host: Mapping[str, Any],
    cache_root: Path,
    output_root: Path,
    arwen_checkout: Path,
    source_receipt: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
    authority_paths: Mapping[str, Path],
    baseline_diagnostic_output: Path | None,
) -> dict[str, Any]:
    from mpas_port.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    arwen_pin = dict(pin_arwen_physics_v841(arwen_checkout))
    # This must precede KernelCache's gpuwm platform-binding construction.
    from mpas_port.cuda_backend import KernelCache, require_cuda


    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    import cupy as cp

    memory = gpu_memory_admission(cp)
    cache = KernelCache(capability=capability, cache_dir=cache_root)
    from mpas_port.cuda_dualrun import fingerprint_atmosphere
    stack = _construct_device_stack(
        host=host, cache=cache, arwen_checkout=arwen_checkout
    )
    baseline_snapshots, _, first_receipts = _run_steps(
        stack=stack,
        start_step=0,
        end_step=CHECKPOINT_STEP,
        capture_steps={0, CHECKPOINT_STEP},
    )
    checkpoint = download_driver_checkpoint(
        stack["driver"],
        stack["backend"],
        dynamics_tendencies=stack.get("gf_dynamics_tendencies"),
    )
    baseline_step16_fingerprints: dict[str, Any] = {}

    def record_baseline_step16(step: int, current_stack: Mapping[str, Any]) -> None:
        if step == CHECKPOINT_STEP + 1:
            baseline_step16_fingerprints.update(
                fingerprint_execution_boundary(current_stack)
            )

    continuation_snapshots, _, second_receipts = _run_steps(
        stack=stack,
        start_step=CHECKPOINT_STEP,
        end_step=FULL_STEPS,
        capture_steps={FULL_STEPS},
        boundary_observer=record_baseline_step16,
    )
    if set(baseline_step16_fingerprints) != {"atmosphere", "backend"}:
        raise RuntimeError("uninterrupted continuation did not capture exact step 16")
    baseline_snapshots.update(continuation_snapshots)
    baseline_f001_atmosphere = fingerprint_atmosphere(stack["driver"].atmosphere)
    baseline_f001_backend = fingerprint_nested_arrays(
        stack["backend"].restart_state()
    )

    gates = {
        SNAPSHOT_LABELS[step]: physical_snapshot_gate(
            baseline_snapshots[step], allow_initial_negative_qv=(step == 0)
        )
        for step in SNAPSHOT_STEPS
    }
    evolution = surface_evolution_gate(
        baseline_snapshots[0], baseline_snapshots[FULL_STEPS]
    )
    comparisons = {
        "F000": compare_snapshot_to_native(
            baseline_snapshots[0], authority_paths["native_f000"]
        ),
        "F030": compare_snapshot_to_native(
            baseline_snapshots[CHECKPOINT_STEP], authority_paths["native_f030"]
        ),
        "F001": compare_snapshot_to_native(
            baseline_snapshots[FULL_STEPS], authority_paths["native_f001"]
        ),
    }
    baseline_diagnostic = None
    if baseline_diagnostic_output is not None:
        baseline_diagnostic = _write_baseline_diagnostic_output(
            diagnostic_root=baseline_diagnostic_output,
            host=host,
            baseline_snapshots=baseline_snapshots,
            step_receipts=first_receipts + second_receipts,
            atmosphere_fingerprint=baseline_f001_atmosphere,
            backend_fingerprint=baseline_f001_backend,
            physical_gates=gates,
            weather_activity=evolution,
            native_comparisons=comparisons,
            capability=capability.as_dict(),
            memory_admission=memory,
            arwen_pin=arwen_pin,
            source_receipt=source_receipt,
            authority_receipt=authority_receipt,
        )

    # Release the uninterrupted device graph before the restart arm.  The
    # F030 checkpoint and F001 snapshots are detached host data and remain
    # valid.
    del stack
    # The driver/backend/seam graph carries reference cycles (state <->
    # driver <-> adapters), so its device arrays survive `del` until the
    # cycle collector runs; collect explicitly so the pools can return
    # their blocks to the driver and the fresh worker process below can
    # admit the full-physics footprint.
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # THE REAL RESTART PATH IS A FRESH PROCESS, so the restart arm runs in
    # a fresh worker process, exactly like a real resumed forecast, and
    # every bitwise identity gate below is unchanged.
    #
    # #327, closed by checkpoint schema v3: the GF advective-forcing pair
    # (rthdynten/rqvdynten) is driver-owned per-step carried state formed
    # by each step's dynamics and consumed by the NEXT begin_step.  It
    # lives outside both the MPAS atmosphere and the Arwen backend restart
    # payload, so the v2 checkpoint never carried it and every restored
    # arm re-entered step 16 on zero forcing lanes while the unbroken run
    # fed the real step-15 pair -- a deterministic all-fields divergence
    # at exactly step 16 (measured 5/5 on the reference node,
    # 2026-08-24).  The v3 checkpoint downloads the pair at F030 and the
    # restore re-seeds it, gated by its own fingerprint identity below.
    #
    # KNOWN LIMIT (measured 2026-08-12, escalated to the seam-contract
    # owner): the frozen Arwen phase-1 is not bitwise-pure with respect
    # to the CuPy device-pool layout at seam construction/run time.  With
    # a bitwise-verified F030 rehydration and identical code, perturbing
    # only the device pool before construction moved the very next
    # radiation-due phase-1: rthratenlw first diff at [g-point 0, level 0,
    # cell 64] 0x38511a95 vs 0x38511a4e (191,479 of 9,011,310 elements;
    # LW -> NoahMP -> YSU family only; SW, GF and raw dqr/dqs/dqg bitwise
    # identical; probe tools/probe_v841_lw_purity.py, arms none/none/host/
    # device).  Identical processes reproduce bit-for-bit; a step-16 FAIL
    # that survives the v3 forcing re-seed and its fingerprint gate is
    # that layout sensitivity, not a serialization defect.
    worker = _spawn_restart_worker(
        checkpoint=checkpoint,
        authority_paths=authority_paths,
        arwen_checkout=arwen_checkout,
        cache_root=cache_root,
        output_root=output_root,
    )
    restored_f030 = worker["restored_f030"]
    f030_rehydration = {
        "atmosphere": require_fingerprint_identity(
            "F030 restored MPAS atmosphere",
            checkpoint.atmosphere_fingerprint,
            restored_f030["atmosphere"],
        ),
        "backend": require_fingerprint_identity(
            "F030 restored Arwen backend",
            checkpoint.backend_fingerprint,
            restored_f030["backend"],
        ),
        "gf_forcing": require_fingerprint_identity(
            "F030 restored GF advective forcing",
            checkpoint.gf_forcing_fingerprint,
            worker["restored_gf_forcing"],
        ),
    }
    step16_identity = {
        "atmosphere": require_fingerprint_identity(
            "first resumed step 16 MPAS atmosphere",
            baseline_step16_fingerprints["atmosphere"],
            worker["step16"]["atmosphere"],
        ),
        "backend": require_fingerprint_identity(
            "first resumed step 16 Arwen backend",
            baseline_step16_fingerprints["backend"],
            worker["step16"]["backend"],
        ),
    }
    restart_f001 = worker["restart_f001"]
    restart_receipts = worker["continuation_step_receipts"]
    identity = require_bitwise_restart_identity(
        baseline_snapshots[FULL_STEPS], restart_f001
    )
    restored_f001_atmosphere = worker["f001_atmosphere_fingerprint"]
    restored_f001_backend = worker["f001_backend_fingerprint"]
    f001_full_state_identity = {
        "atmosphere": require_fingerprint_identity(
            "F030 restart continuation F001 MPAS atmosphere",
            baseline_f001_atmosphere,
            restored_f001_atmosphere,
        ),
        "backend": require_fingerprint_identity(
            "F030 restart continuation F001 Arwen backend",
            baseline_f001_backend,
            restored_f001_backend,
        ),
    }
    full_restart_identity = {
        "atmosphere": baseline_f001_atmosphere,
        "backend": baseline_f001_backend,
        "restart_arm_fresh_process": True,
        "f030_rehydration": f030_rehydration,
        "first_resumed_step16": step16_identity,
        "f001_full_state": f001_full_state_identity,
        "bitwise_identical": True,
    }

    gates["F001_RESTART"] = physical_snapshot_gate(
        restart_f001, allow_initial_negative_qv=False
    )
    static = _static_output_fields(host)
    files = {
        "F000": write_snapshot_netcdf(
            output_root / SNAPSHOT_FILE_NAMES["F000"], baseline_snapshots[0], static
        ),
        "F030": write_snapshot_netcdf(
            output_root / SNAPSHOT_FILE_NAMES["F030"],
            baseline_snapshots[CHECKPOINT_STEP],
            static,
        ),
        "F001": write_snapshot_netcdf(
            output_root / SNAPSHOT_FILE_NAMES["F001"],
            baseline_snapshots[FULL_STEPS],
            static,
        ),
        "F001_RESTART": write_snapshot_netcdf(
            output_root / SNAPSHOT_FILE_NAMES["F001_RESTART"], restart_f001, static
        ),
    }
    return {
        "capability": capability.as_dict(),
        "arwen_pre_kernel_cache_pin": arwen_pin,
        "memory_admission": memory,
        "source_pins": source_receipt,
        "authority": authority_receipt,
        "host_preparation": {
            "constructor": host["constructor_receipt"],
            "scalars": host["scalar_receipt"],
            "mesh_overlay": host["mesh_evidence"],
            "reconstruction_coefficients": host["reconstruction_coefficients"],
            "edge_normal_vectors": host["edge_normal_vectors"],
            "inactive_deformation": host["defc"],
        },
        "uninterrupted": {
            "steps": FULL_STEPS,
            "step_receipts": first_receipts + second_receipts,
            "snapshot_receipts": {
                SNAPSHOT_LABELS[step]: baseline_snapshots[step]["receipt"]
                for step in SNAPSHOT_STEPS
            },
        },
        "checkpoint_restart": {
            "schema": CHECKPOINT_SCHEMA,
            "f001_restart_snapshot_receipt": restart_f001["receipt"],
            "step": CHECKPOINT_STEP,
            "time_seconds": checkpoint.model_time_seconds,
            "checkpoint_atmosphere": dict(checkpoint.atmosphere_fingerprint),
            "checkpoint_backend": dict(checkpoint.backend_fingerprint),
            "restart_execution": {
                "fresh_process": True,
                "worker_schema": RESTART_WORKER_SCHEMA,
                "worker_files": worker["files"],
                "worker_capability": worker["capability"],
                "worker_memory_admission": worker["memory_admission"],
            },
            "continuation_step_receipts": restart_receipts,
            "f001_full_state_identity": full_restart_identity,
            "f001_identity": identity,
        },
        "physical_gates": gates,
        "weather_activity": evolution,
        "native_comparisons": comparisons,
        "snapshot_files": files,
        "baseline_diagnostic_output": baseline_diagnostic,
        "gf_deviation": {
            "mpas_dynamics_tendencies_computed": True,
            "fa35_public_api_accepts_them": False,
            "fa35_rthften_rqvften": "zero",
            "native_gf_parity_claim": False,
        },
        "full_physics_cuda_executed": True,
        "forecast_seconds": FULL_STEPS * DT_SECONDS,
        "outer_steps": FULL_STEPS,
        "restart_bitwise_identical": True,
    }


def _spawn_restart_worker(
    *,
    checkpoint: HostDriverCheckpoint,
    authority_paths: Mapping[str, Path],
    arwen_checkout: Path,
    cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run the F030 restart continuation in a genuinely fresh process."""

    worker_dir = output_root / "restart-worker"
    worker_dir.mkdir(parents=False, exist_ok=False)
    input_path = worker_dir / "restart-worker-input.pkl"
    output_path = worker_dir / "restart-worker-output.pkl"
    job = {
        "schema": RESTART_WORKER_SCHEMA,
        "arwen_commit": ARWEN_COMMIT,
        "checkpoint": checkpoint,
        "authority_paths": {
            role: str(path) for role, path in authority_paths.items()
        },
        "arwen_checkout": str(arwen_checkout),
        "cache_root": str(cache_root / "restart-worker-cache"),
    }
    with input_path.open("xb") as stream:
        pickle.dump(job, stream, protocol=4)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--restart-worker-input",
            str(input_path),
            "--restart-worker-output",
            str(output_path),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh-process restart worker failed with return code "
            f"{completed.returncode}"
        )
    with output_path.open("rb") as stream:
        results = pickle.load(stream)
    if not isinstance(results, Mapping) or results.get("schema") != RESTART_WORKER_SCHEMA:
        raise RuntimeError("restart worker returned an unrecognized payload schema")
    if results.get("arwen_commit") != ARWEN_COMMIT:
        raise RuntimeError("restart worker executed against a different Arwen commit")
    if results.get("fresh_process") is not True:
        raise RuntimeError("restart worker did not attest fresh-process execution")
    required = {
        "restored_f030",
        "restored_gf_forcing",
        "step16",
        "restart_f001",
        "continuation_step_receipts",
        "f001_atmosphere_fingerprint",
        "f001_backend_fingerprint",
        "capability",
        "memory_admission",
    }
    missing = sorted(required - set(results))
    if missing:
        raise RuntimeError(f"restart worker payload lacks fields {missing}")
    results = dict(results)
    results["files"] = {
        "input": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    return results


def _execute_restart_worker(input_path: Path, output_path: Path) -> int:
    """Worker entry: restore the F030 checkpoint and continue to F001."""

    with Path(input_path).open("rb") as stream:
        job = pickle.load(stream)
    if not isinstance(job, Mapping) or job.get("schema") != RESTART_WORKER_SCHEMA:
        raise RuntimeError("restart worker input schema mismatch")
    if job.get("arwen_commit") != ARWEN_COMMIT:
        raise RuntimeError("restart worker input Arwen commit mismatch")
    checkpoint = job["checkpoint"]
    if not isinstance(checkpoint, HostDriverCheckpoint):
        raise TypeError("restart worker requires a HostDriverCheckpoint payload")
    if float(checkpoint.model_time_seconds) != CHECKPOINT_STEP * DT_SECONDS:
        raise RuntimeError("restart worker checkpoint is not at exact F030")
    forcing = getattr(checkpoint, "gf_dynamics_tendencies", None)
    if forcing is None:
        raise RuntimeError(
            "checkpoint predates the GF advective-forcing capture "
            f"({CHECKPOINT_SCHEMA}): restoring it would feed GF zero "
            "rthdynten/rqvdynten lanes at the first resumed step -- the "
            "deterministic step-16 restart divergence (#327).  Re-run the "
            "baseline arm to mint a checkpoint that carries the pair."
        )
    paths = {role: Path(value) for role, value in dict(job["authority_paths"]).items()}
    arwen_checkout = _plain_absolute(Path(job["arwen_checkout"]), "Arwen checkout")
    # Source pins verify first: the checkout guard imports the manifest from a
    # pinned module, so that module's bytes are proven before its constants
    # are trusted.
    require_frozen_execution_sources()
    verify_arwen_checkout_git(arwen_checkout)
    authority_receipt = verify_authorities(paths)
    host = _prepare_host_execution(paths, authority_receipt)

    from mpas_port.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    pin_arwen_physics_v841(arwen_checkout)
    from mpas_port.cuda_backend import KernelCache, require_cuda

    cache_root = _plain_absolute(Path(job["cache_root"]), "restart worker cache root")
    cache_root.mkdir(parents=False, exist_ok=False)
    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    import cupy as cp

    memory = gpu_memory_admission(
        cp, minimum=RESTART_WORKER_MIN_FREE_DEVICE_BYTES
    )
    cache = KernelCache(capability=capability, cache_dir=cache_root)
    from mpas_port.cuda_dualrun import fingerprint_atmosphere

    stack = _construct_device_stack(
        host=host,
        cache=cache,
        arwen_checkout=arwen_checkout,
        state=checkpoint.state,
        saved_diagnostics=checkpoint.saved_diagnostics,
        backend_restart=checkpoint.backend_state,
        gf_dynamics_tendencies=forcing,
    )
    restored_f030 = fingerprint_execution_boundary(stack)
    require_fingerprint_identity(
        "F030 restored MPAS atmosphere (fresh restart process)",
        checkpoint.atmosphere_fingerprint,
        restored_f030["atmosphere"],
    )
    require_fingerprint_identity(
        "F030 restored Arwen backend (fresh restart process)",
        checkpoint.backend_fingerprint,
        restored_f030["backend"],
    )
    restored_carrier = stack["gf_dynamics_tendencies"]
    restored_gf_forcing = fingerprint_nested_arrays(
        {
            "rthdynten": np.ascontiguousarray(
                cp.asnumpy(restored_carrier.rthdynten)
            ),
            "rqvdynten": np.ascontiguousarray(
                cp.asnumpy(restored_carrier.rqvdynten)
            ),
            "time_seconds": float(restored_carrier.time_seconds),
        }
    )
    require_fingerprint_identity(
        "F030 restored GF advective forcing (fresh restart process)",
        checkpoint.gf_forcing_fingerprint,
        restored_gf_forcing,
    )
    step16: dict[str, Any] = {}

    def capture_step16(step: int, current_stack: Mapping[str, Any]) -> None:
        if step == CHECKPOINT_STEP + 1:
            step16.update(fingerprint_execution_boundary(current_stack))

    restart_snapshots, _, restart_receipts = _run_steps(
        stack=stack,
        start_step=CHECKPOINT_STEP,
        end_step=FULL_STEPS,
        capture_steps={FULL_STEPS},
        boundary_observer=capture_step16,
    )
    if set(step16) != {"atmosphere", "backend"}:
        raise RuntimeError("restart worker did not capture exact step 16")
    results = {
        "schema": RESTART_WORKER_SCHEMA,
        "arwen_commit": ARWEN_COMMIT,
        "fresh_process": True,
        "capability": capability.as_dict(),
        "memory_admission": memory,
        "restored_f030": restored_f030,
        "restored_gf_forcing": restored_gf_forcing,
        "step16": step16,
        "restart_f001": restart_snapshots[FULL_STEPS],
        "continuation_step_receipts": restart_receipts,
        "f001_atmosphere_fingerprint": fingerprint_atmosphere(
            stack["driver"].atmosphere
        ),
        "f001_backend_fingerprint": fingerprint_nested_arrays(
            stack["backend"].restart_state()
        ),
    }
    with Path(output_path).open("xb") as stream:
        pickle.dump(results, stream, protocol=4)
    print(json.dumps({"restart_worker": "complete", "schema": RESTART_WORKER_SCHEMA}))
    return 0


def verify_arwen_checkout_git(checkout: Path) -> dict[str, Any]:
    """Admit an Arwen checkout iff its executed seam source is the proven one.

    The gate is ``ARWEN_SOURCE_MANIFEST`` -- the sha-freeze of the sixteen
    gpuwm files the port executes.  The checkout's HEAD commit, tree, and
    dirty state are recorded into every receipt as provenance but do not gate
    execution: a commit that moves nothing the seam executes must not
    invalidate a proven pin.  A dirty manifest file still refuses, because
    the receipt could not then name the executed bytes by commit; dirt in
    unrelated files is recorded loudly and execution proceeds.
    """

    root = checkout.resolve(strict=True)

    def git(*arguments: str, strip: bool = True) -> str:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            raise RuntimeError(f"Arwen git command wrote stderr: {completed.stderr!r}")
        return completed.stdout.strip() if strip else completed.stdout

    top = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    # NUL-delimited and unstripped: a porcelain entry BEGINS with a status
    # column that may be a space, which str.strip would silently eat off the
    # first entry, shifting every parsed path by one byte.
    status = git(
        "status", "--porcelain=v1", "-z", "--untracked-files=all", strip=False
    )
    if top != root:
        raise RuntimeError("Arwen checkout is not the exact requested Git root")

    dirty_paths: list[str] = []
    expect_rename_origin = False
    for entry in status.split("\0"):
        if not entry:
            continue
        if expect_rename_origin:
            dirty_paths.append(entry)
            expect_rename_origin = False
            continue
        code, path = entry[:2], entry[3:]
        dirty_paths.append(path)
        if code and code[0] in "RC":
            expect_rename_origin = True
    dirty_paths = sorted(set(dirty_paths))

    manifest = arwen_source_manifest()
    files: dict[str, str] = {}
    for relative, expected in manifest.items():
        target = root / relative
        if not target.is_file():
            raise RuntimeError(
                f"{relative} is missing from the Arwen checkout {root} — the "
                "seam's executed source moved; re-prove before running"
            )
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(
                f"{relative} does not match the proven manifest (expected "
                f"{expected}, found {actual}) — the seam's executed source "
                "moved; re-prove before running"
            )
        files[relative] = actual

    dirty_manifest = sorted(set(dirty_paths) & set(manifest))
    if dirty_manifest:
        raise RuntimeError(
            f"{dirty_manifest[0]} is dirty in the Arwen checkout — its bytes "
            "cannot be provenanced to any commit; commit or restore it, then "
            "re-prove before running"
        )
    if dirty_paths:
        print(
            json.dumps(
                {
                    "arwen_checkout_dirty_non_manifest_paths": dirty_paths,
                    "note": (
                        "no manifest file moved; recorded as provenance and "
                        "execution proceeds"
                    ),
                },
                sort_keys=True,
            )
        )
    return {
        "root": str(root),
        "head": head,
        "tree": tree,
        "clean": not dirty_paths,
        "dirty_paths": dirty_paths,
        "provenance": (
            "head/tree/dirty-state are recorded, not gates; the gate is the "
            "sixteen-file executed-source manifest"
        ),
        "manifest": {
            "files": files,
            "sha256": canonical_json_sha256(files),
        },
    }


def _arwen_seam_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """The executed-seam identity that must not move during a proof.

    HEAD/tree/dirty-state are provenance: an unrelated file appearing in the
    checkout mid-run is recorded in the receipt, not turned into a refusal
    that aborts a passed proof.
    """

    return {"root": record["root"], "manifest": record["manifest"]}


def _paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        role: Path(getattr(args, role)).expanduser().absolute()
        for role in AUTHORITY_PINS
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for role, default in default_authority_paths().items():
        parser.add_argument("--" + role.replace("_", "-"), type=Path, default=default)
    parser.add_argument(
        "--arwen-checkout",
        type=Path,
        default=ROOT / "work" / "arwen19-mpas-column-corrected",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline-diagnostic-output",
        type=Path,
        help="fresh isolated F000/F030/F001 engineering-baseline output root",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify release/source/authority bytes and host mappings without CUDA",
    )
    parser.add_argument(
        "--restart-worker-input", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--restart-worker-output", type=Path, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.restart_worker_input is not None or args.restart_worker_output is not None:
        if args.restart_worker_input is None or args.restart_worker_output is None:
            parser.error(
                "restart worker requires both --restart-worker-input and "
                "--restart-worker-output"
            )
    elif not args.preflight_only and (args.cache_root is None or args.output is None):
        parser.error("execution requires --cache-root and --output")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.restart_worker_input is not None:
        assert args.restart_worker_output is not None
        return _execute_restart_worker(
            args.restart_worker_input, args.restart_worker_output
        )
    paths = _paths_from_args(args)
    arwen_checkout = _plain_absolute(args.arwen_checkout, "Arwen checkout")
    if not arwen_checkout.is_dir():
        raise FileNotFoundError(arwen_checkout)
    # None source pins refuse here, before authority hashing, host preparation,
    # destination creation, production imports, or CUDA probing.  They also
    # precede the checkout guard, which imports the seam manifest from a
    # pinned module: the module's bytes are proven before its constants are
    # trusted.
    source_before = require_frozen_execution_sources()
    arwen_git_before = verify_arwen_checkout_git(arwen_checkout)
    authority_before = verify_authorities(paths)
    host = _prepare_host_execution(paths, authority_before)

    if args.preflight_only:
        source_after = require_frozen_execution_sources()
        authority_after = verify_authorities(paths)
        arwen_git_after = verify_arwen_checkout_git(arwen_checkout)
        if (
            source_after != source_before
            or authority_after != authority_before
            or _arwen_seam_identity(arwen_git_after)
            != _arwen_seam_identity(arwen_git_before)
        ):
            raise RuntimeError(
                "source, authority, or Arwen seam bytes changed during preflight"
            )
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "mode": "preflight-only; CUDA not imported",
                    "sources": source_before,
                    "authorities_sha256": authority_before["sha256"],
                    "arwen_git": arwen_git_before,
                    "arwen_contract_document_sha256": ARWEN_CONTRACT_DOCUMENT_SHA256,
                    "arwen_contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
                    "arwen_glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
                    "constructor": host["constructor_receipt"],
                    "scalars": host["scalar_receipt"],
                    "reconstruction_coefficients": host["reconstruction_coefficients"],
                    "edge_normal_vectors": host["edge_normal_vectors"],
                    "claim": CLAIM,
                    "nonclaims": list(NONCLAIMS),
                },
                sort_keys=True,
            )
        )
        return 0

    assert args.cache_root is not None and args.output is not None
    cache_root, output_root = validate_destination(
        args.cache_root, args.output, tuple(paths.values())
    )
    baseline_diagnostic_output = None
    if args.baseline_diagnostic_output is not None:
        baseline_diagnostic_output = validate_baseline_diagnostic_destination(
            args.baseline_diagnostic_output,
            cache_root=cache_root,
            output_root=output_root,
            protected=tuple(paths.values()),
        )

    cache_root.mkdir(parents=False)
    output_root.mkdir(parents=False)
    if baseline_diagnostic_output is not None:
        baseline_diagnostic_output.mkdir(parents=False)
    started = time.perf_counter()
    proof = _execute_full_proof(
        host=host,
        cache_root=cache_root,
        output_root=output_root,
        arwen_checkout=arwen_checkout,
        source_receipt=source_before,
        authority_receipt=authority_before,
        authority_paths=paths,
        baseline_diagnostic_output=baseline_diagnostic_output,
    )
    source_after = require_frozen_execution_sources()
    authority_after = verify_authorities(paths)
    arwen_git_after = verify_arwen_checkout_git(arwen_checkout)
    if (
        source_after != source_before
        or authority_after != authority_before
        or _arwen_seam_identity(arwen_git_after)
        != _arwen_seam_identity(arwen_git_before)
    ):
        raise RuntimeError(
            "source, authority, or Arwen seam bytes changed during execution"
        )
    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "claim": CLAIM,
        "nonclaims": list(NONCLAIMS),
        "source_release": SOURCE_RELEASE,
        "arwen_commit": ARWEN_COMMIT,
        "arwen_contract_document_sha256": ARWEN_CONTRACT_DOCUMENT_SHA256,
        "arwen_contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
        "arwen_glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
        "arwen_git": {"before": arwen_git_before, "after": arwen_git_after},
        "arwen_checkout_unchanged": arwen_git_after == arwen_git_before,
        "execution_seconds": time.perf_counter() - started,
        "proof": proof,
        "sources_unchanged": True,
        "authorities_unchanged": True,
        "weather_plot_policy": "native Rust/Arwen renderer only; q2 forbidden",
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    receipt = output_root / RECEIPT_NAME
    _write_exclusive_json(receipt, payload)
    print(
        json.dumps(
            {
                "status": "passed",
                "receipt": str(receipt),
                "receipt_sha256": sha256_file(receipt),
                "payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
