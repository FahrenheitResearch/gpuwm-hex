#!/usr/bin/env python3
"""Build the immutable schema/hash record for the frozen MPAS step fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = {
    "rho": {
        "shape": [2562, 15],
        "location": "cell",
        "role": "prognostic state",
        "tolerance": {"atol": 1.0e-6, "rtol": 2.0e-5},
    },
    "theta": {
        "shape": [2562, 15],
        "location": "cell",
        "role": "prognostic state",
        "tolerance": {"atol": 5.0e-3, "rtol": 2.0e-5},
    },
    "u": {
        "shape": [7680, 15],
        "location": "edge",
        "role": "prognostic state",
        "tolerance": {"atol": 1.0e-3, "rtol": 2.0e-5},
    },
    "w": {
        "shape": [2562, 16],
        "location": "cell vertical interface",
        "role": "prognostic state",
        "tolerance": {"atol": 1.0e-4, "rtol": 2.0e-5},
    },
    "pressure": {
        "shape": [2562, 15],
        "location": "cell",
        "role": "diagnosed state",
        "tolerance": {"atol": 1.0, "rtol": 2.0e-5},
    },
    "qv": {
        "shape": [2562, 15],
        "location": "cell",
        "role": "advected scalar",
        "tolerance": {"atol": 5.0e-8, "rtol": 2.0e-5},
    },
    "divergence": {
        "shape": [2562, 15],
        "location": "cell",
        "role": "linked full-model diagnostic",
        "tolerance": {"atol": 5.0e-10, "rtol": 5.0e-5},
    },
    "vorticity": {
        "shape": [5120, 15],
        "location": "vertex",
        "role": "linked full-model diagnostic",
        "tolerance": {"atol": 5.0e-10, "rtol": 5.0e-5},
    },
    "ke": {
        "shape": [2562, 15],
        "location": "cell",
        "role": "linked full-model diagnostic",
        "tolerance": {"atol": 5.0e-4, "rtol": 2.0e-5},
    },
}

TIMES = {
    "t0": {
        "xtime": "2000-01-01_00:00:00",
        "elapsed_seconds": 0,
        "history": {
            "authority_path": (
                "<authority-root>/runs/"
                "jw-v823-20260810/step/history.2000-01-01_00.00.00.nc"
            ),
            "bytes": 14_150_368,
            "sha256": "71be0a94bededef47fd4e896543495c77be79b7e981ee73a852838aae54144d5",
        },
    },
    "t1": {
        "xtime": "2000-01-01_00:10:00",
        "elapsed_seconds": 600,
        "history": {
            "authority_path": (
                "<authority-root>/runs/"
                "jw-v823-20260810/step/history.2000-01-01_00.10.00.nc"
            ),
            "bytes": 14_150_368,
            "sha256": "35bd6cc44bec7874a09b8b34318aa1a7c735b28e660f6f76ef2be489267e0d18",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_stats(values: np.ndarray) -> dict[str, int | float | None]:
    finite = np.isfinite(values)
    selected = values[finite]
    return {
        "count": int(values.size),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(values).sum()),
        "positive_infinity_count": int(np.isposinf(values).sum()),
        "negative_infinity_count": int(np.isneginf(values).sum()),
        "min": float(selected.min()) if selected.size else None,
        "max": float(selected.max()) if selected.size else None,
        "mean_float64": float(selected.astype(np.float64).mean()) if selected.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_dir", type=Path)
    args = parser.parse_args()
    fixture_dir = args.fixture_dir.resolve()
    tool_dir = Path(__file__).resolve().parent

    expected = {
        f"{time_id}_{field}.f32le"
        for time_id in TIMES
        for field in FIELDS
    }
    present = {path.name for path in fixture_dir.glob("*.f32le")}
    if present != expected:
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        raise SystemExit(f"raw fixture set mismatch; missing={missing}, unexpected={unexpected}")

    files: dict[str, dict[str, object]] = {}
    values_by_key: dict[tuple[str, str], np.ndarray] = {}
    for time_id in TIMES:
        for field, definition in FIELDS.items():
            filename = f"{time_id}_{field}.f32le"
            path = fixture_dir / filename
            shape = tuple(definition["shape"])
            expected_bytes = int(np.prod(shape, dtype=np.int64)) * 4
            if path.stat().st_size != expected_bytes:
                raise SystemExit(
                    f"{filename}: expected {expected_bytes} bytes, got {path.stat().st_size}"
                )
            values = np.fromfile(path, dtype="<f4").reshape(shape)
            values_by_key[time_id, field] = values
            files[filename] = {
                "time": time_id,
                "field": field,
                "dtype": "<f4",
                "order": "C",
                "shape": list(shape),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                **finite_stats(values),
            }

    deltas: dict[str, dict[str, int | float]] = {}
    for field in FIELDS:
        initial = values_by_key["t0", field].astype(np.float64)
        final = values_by_key["t1", field].astype(np.float64)
        delta = final - initial
        deltas[field] = {
            "changed_element_count": int(np.count_nonzero(delta)),
            "max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta))),
            "l2": float(np.linalg.norm(delta.ravel())),
        }

    extractor_source = tool_dir / "extract_step_oracle.F90"
    extractor_binary = tool_dir / "build" / "extract_step_oracle"
    manifest = {
        "schema": "mpas-port.frozen-fortran-trajectory.v1",
        "evidence": {
            "kind": "full-model frozen-Fortran trajectory",
            "authority": "linked MPAS-A v8.2.3 atmosphere_model history output",
            "verification_status": "authority extracted; Python whole-step match not yet established",
            "non_claim": (
                "A passing stock Fortran run is not port evidence. These values are the element-wise "
                "target that the Python whole-step driver must match."
            ),
        },
        "authority": {
            "repository": "https://github.com/MPAS-Dev/MPAS-Model",
            "release": "v8.2.3",
            "tag_commit": "ac3866c1e5b05f6d4f5bd41aeab7d3882bace514",
            "source_archive": {
                "repo_path": "vendor/MPAS_source_v8.2.3_group/MPAS-Model-v8.2.3.tar.gz",
                "sha256": "bb3b02c30abffe9ff0318165b25724e6855fb69076fd89243f06a24e11912ee1",
            },
            "atmosphere_binary": {
                "authority_path": "<authority-root>/bin/atmosphere_model",
                "sha256": "dfdfcebadb39d902ebe70ff59ed5e7540f4795d02c2348b9667cd58021b398c0",
            },
            "initial_condition": {
                "authority_path": (
                    "<authority-root>/runs/"
                    "jw-v823-20260810/init/x1.2562.init.nc"
                ),
                "sha256": "45c6879f794af984de791ca7da654a7da5d515dbdb6a131ea778f4edcf597970",
            },
            "grid_sha256": "8a825312a713bbe959c33ed03c2b503e5ec626238de6b15a686cd0ad5b40c986",
            "namelist_sha256": "b4b882ab1fb1252b742c75ef6d04fb9c0dfefe19e78495f1490e341555565fd2",
            "streams_sha256": "1c638526858d6ce7b5008e18e74591a19d2d8c109da8a2e6cc13694b47366145",
            "model_log_sha256": "203a802fe6b27f4c3ae296623eeb24da0fe6a734ab8492f2717c702b99e4a67d",
        },
        "run": {
            "id": "jw-v823-20260810",
            "case": "Jablonowski-Williamson perturbation (config_init_case=2)",
            "mesh": "x1.2562",
            "nCells": 2562,
            "nEdges": 7680,
            "nVertices": 5120,
            "nVertLevels": 15,
            "dt_seconds": 600.0,
            "time_integration": "SRK3 order 3",
            "split_dynamics_transport": True,
            "acoustic_substeps": 6,
            "dynamics_split_steps": 1,
            "physics_suite": "none",
            "mpi_ranks": 1,
            "model_precision": "single",
        },
        "binary_layout": {
            "dtype": "IEEE-754 binary32 little-endian",
            "numpy_dtype": "<f4",
            "header": "none",
            "array_order": "C",
            "logical_axes": "horizontal point, vertical level",
        },
        "comparison_policy": {
            "gate": "every finite element must satisfy abs(candidate-reference) <= atol + rtol*abs(reference)",
            "shape_must_match": True,
            "missing_or_nonfinite_fails": True,
            "ulp": {
                "measurement": "candidate rounded to binary32 versus binary32 authority",
                "enforcement": "report-only",
                "reason": (
                    "near-zero divergence/vorticity cancellation makes ULP discontinuous; "
                    "the declared absolute/relative envelope is the authoritative gate"
                ),
            },
        },
        "times": TIMES,
        "fields": FIELDS,
        "files": files,
        "trajectory_delta": deltas,
        "generator": {
            "extractor_source_sha256": sha256(extractor_source),
            "extractor_binary_sha256": sha256(extractor_binary) if extractor_binary.exists() else None,
            "compiler": "nvfortran 26.1-0 via Open MPI mpifort",
            "netcdf_c": "4.9.2",
            "netcdf_fortran": "4.6.1",
        },
    }

    manifest_path = fixture_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    sums = [f"{sha256(manifest_path)}  manifest.json"]
    sums.extend(f"{record['sha256']}  {name}" for name, record in sorted(files.items()))
    (fixture_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"wrote {manifest_path}")
    print(f"manifest sha256 {sha256(manifest_path)}")
    print(f"fixture payload bytes {sum(int(item['bytes']) for item in files.values())}")


if __name__ == "__main__":
    main()
