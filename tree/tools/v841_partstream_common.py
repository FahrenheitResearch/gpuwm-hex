"""Shared pins, provenance and fingerprint IO for the brief-2 partition proofs.

This module never edits the frozen runner.  It imports the runner's helpers and
adds only what the partitioned/streamed proofs need:

* an explicit Arwen pin TABLE (the runner's own ``ARWEN_COMMIT`` constant stays
  untouched) with the same git verification laws;
* a source manifest of the port modules that actually execute, so a receipt can
  never be read as though it came from a different tree;
* real (non-symlink) authority paths, because the repo's default authority
  locations are symlinks into the asset store and the runner refuses those;
* boundary-fingerprint JSONL writing/reading used by both run tools and the
  comparator.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ARWEN_PIN_TABLE",
    "verify_arwen_checkout_pinned",
    "PORT_SOURCE_FILES",
    "port_source_manifest",
    "real_authority_paths",
    "BoundaryFingerprintWriter",
    "read_boundary_fingerprints",
    "sha256_file",
]

# These proofs do NOT run against the runner's own ARWEN_COMMIT.  The tree
# originally frozen for them, 925309da... (tree b3a171d5...), carries the
# proven phase-1 LW impurity: outputs depend on device pool contents.
# Partitioned execution changes allocation patterns, so every bitwise proof
# runs against the FIXED tree instead.  Selected by name only, and the two
# entries below are historical checkout records -- they do not follow the
# runner's pin and must not be edited when it moves.
ARWEN_PIN_TABLE: dict[str, dict[str, str]] = {
    "lwfix-20260812": {
        "commit": "7ce2f5de740afec274e589a5dae1015b994bad31",
        "tree": "196eb4fec235571f426ff905304a7c621030e5f2",
        "note": (
            "commit 7ce2f5de, forward of 925309da; six-arm "
            "purity probe green (15/15 pairs, 0/207 arrays differ)"
        ),
    },
    "frozen-925309da": {
        "commit": "925309da966c856a9622578502db4247d093bdec",
        "tree": "b3a171d57056b43a185f7d8994e72aba8b50e9d1",
        "note": "the originally frozen tree; carries the phase-1 LW impurity",
    },
}

# Every port module whose bytes can move a number in these proofs.  Recorded,
# not pinned to a constant: brief-1 legitimately extended two of them, so the
# receipt states what executed rather than refusing a moved hash.
#
# Every row must name a file that is in the tree, and `port_source_manifest`
# refuses if one does not.  The partitioned arm is
# `run_cuda_v841_forecast_2gpu.py` below; a twenty-fifth row named
# `run_cuda_v841_partitioned_x4.py` for the same arm under a spelling no
# commit ever carried, and it was recorded as a null hash rather than refused.
PORT_SOURCE_FILES = (
    "src/hexcore/cuda_driver.py",
    "src/hexcore/cuda_v841.py",
    "src/hexcore/cuda_acoustic_v841.py",
    "src/hexcore/cuda_transport_v841.py",
    "src/hexcore/cuda_dynamics_v841.py",
    "src/hexcore/cuda_horizontal.py",
    "src/hexcore/cuda_horizontal_v841.py",
    "src/hexcore/cuda_physics_v841.py",
    "src/hexcore/cuda_physics_prep_v841.py",
    "src/hexcore/cuda_gwdo_v841.py",
    "src/hexcore/cuda_arwen_physics_v841.py",
    "src/hexcore/config_v841.py",
    "src/hexcore/partition_assets_v841.py",
    "src/hexcore/partition_local_mesh_v841.py",
    "src/hexcore/partition_state_v841.py",
    "src/hexcore/partition_executor_v841.py",
    "src/hexcore/partition_net_v841.py",
    "src/hexcore/partition_device_scheduler_v841.py",
    "tools/run_cuda_v841_full_physics_x4.py",
    "tools/v841_partstream_common.py",
    "tools/run_cuda_v841_resident_baseline_lwfix.py",
    "tools/run_cuda_v841_forecast_2gpu.py",
    "tools/compare_2gpu_invariance_v841.py",
    "tools/build_2way_part_v841.py",
)

# The official METIS decomposition ships beside a *different* copy of the grid
# file: all eleven connectivity arrays are bitwise identical to the runner's
# authority copy and only latCell/lonCell differ, so the partition is valid
# against the authority grid and the cached layout assets are keyed to the
# authority copy's sha256.
AUTHORITATIVE_MESH_SHA256 = (
    "48e747157bb1f0b83b96505e268699dfb562b4c1428468cb91457fbb03b1be55"
)
OFFICIAL_PARTITION_SUBDIR = "meshes/official-vr-92to25"
GRAPH_INFO_NAME = "x4.163842.graph.info"


def official_partition_dir() -> Path:
    """The official METIS distribution under the asset store root.

    No path is baked in: a wrong default would be joined with the part-file
    name and the caller would refuse as a missing part file, hiding that the
    asset store was simply never named.
    """

    declared = os.environ.get("MPAS_ASSET_ROOT")
    if not declared:
        raise RuntimeError(
            "no official partition directory: pass one explicitly or set "
            "MPAS_ASSET_ROOT to the asset store root"
        )
    return Path(declared).expanduser() / OFFICIAL_PARTITION_SUBDIR


def official_part_file(parts: int, directory: Path | None = None) -> Path:
    root = official_partition_dir() if directory is None else Path(directory)
    path = root / f"{GRAPH_INFO_NAME}.part.{int(parts)}"
    if not path.is_file():
        raise FileNotFoundError(f"no official METIS part file for P={parts}: {path}")
    return path


def load_layouts(
    cache_npz: str | Path,
    *,
    parts: int,
    halo_rings: int,
    mesh: Any,
    partition_dir: Path | None = None,
    mesh_sha256: str = AUTHORITATIVE_MESH_SHA256,
) -> list[Any]:
    """Cache-first layout load keyed to the authority grid and official part file."""

    import numpy as np

    from hexcore.partition_assets_v841 import build_or_load_layouts

    root = official_partition_dir() if partition_dir is None else Path(partition_dir)
    return build_or_load_layouts(
        cache_npz,
        graph_info=root / GRAPH_INFO_NAME,
        part_file=official_part_file(parts, root),
        halo_rings=int(halo_rings),
        cells_on_vertex=np.asarray(mesh.arrays["cellsOnVertex"]),
        cells_on_edge=np.asarray(mesh.arrays["cellsOnEdge"]),
        mesh_sha256=mesh_sha256,
    )


_AUTHORITY_FILE_NAMES = {
    "grid": "x4.163842.grid.nc",
    "static": "x4.163842.static.nc",
    "init": "x4.163842.init.nc",
    "native_f000": "history.2026-08-10_12.00.00.nc",
    "native_f030": "history.2026-08-10_12.30.00.nc",
    "native_f001": "history.2026-08-10_13.00.00.nc",
    "native_validation_receipt": "native-gwdo-authority-receipt.json",
    "native_launch_receipt": "native-gwdo-launch-receipt.json",
    "native_closure": "run-closure.status",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def real_authority_paths(asset_root: str | Path) -> dict[str, Path]:
    """Authority roles mapped onto the real files, not the repo's symlinks.

    ``run_cuda_v841_full_physics_x4._plain_absolute`` refuses symbolic links,
    and every default authority location under ``work/`` is a symlink into the
    asset store.  The bytes are still verified against the runner's own
    ``AUTHORITY_PINS`` by ``verify_authorities``.
    """

    root = Path(asset_root).expanduser().resolve(strict=True)
    paths = {role: root / name for role, name in _AUTHORITY_FILE_NAMES.items()}
    missing = sorted(role for role, path in paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"asset store lacks authority roles {missing}")
    linked = sorted(role for role, path in paths.items() if path.is_symlink())
    if linked:
        raise ValueError(f"asset store authority roles are symlinks: {linked}")
    return paths


def verify_arwen_checkout_pinned(checkout: str | Path, pin_name: str) -> dict[str, Any]:
    """Same git laws as the runner's ``verify_arwen_checkout_git``, new pin."""

    if pin_name not in ARWEN_PIN_TABLE:
        raise ValueError(
            f"unknown Arwen pin {pin_name!r}; known {sorted(ARWEN_PIN_TABLE)}"
        )
    pin = ARWEN_PIN_TABLE[pin_name]
    root = Path(checkout).expanduser().resolve(strict=True)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            raise RuntimeError(f"Arwen git command wrote stderr: {completed.stderr!r}")
        return completed.stdout.strip()

    top = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if top != root:
        raise RuntimeError("Arwen checkout is not the exact requested Git root")
    if head != pin["commit"]:
        raise RuntimeError(f"Arwen checkout HEAD {head} != pin {pin['commit']}")
    if tree != pin["tree"]:
        raise RuntimeError(f"Arwen checkout tree {tree} != pin {pin['tree']}")
    if status:
        raise RuntimeError(f"Arwen checkout is not clean: {status!r}")
    return {
        "pin": pin_name,
        "root": str(root),
        "head": head,
        "tree": tree,
        "clean": True,
        "note": pin["note"],
    }


def port_source_manifest(repo_root: str | Path) -> dict[str, Any]:
    """Hash every declared port source, or refuse naming the ones that are gone.

    A declared row that is absent used to be recorded as ``null``.  The
    receipt then claimed to state the executing source set while one row stood
    for nothing, and two runs whose module had been deleted between them
    produced the same digest for that row -- so the receipt could not show
    what a reader would need to see.
    """

    root = Path(repo_root).expanduser().resolve(strict=True)
    absent = sorted(
        relative for relative in PORT_SOURCE_FILES if not (root / relative).is_file()
    )
    if absent:
        raise FileNotFoundError(
            f"port source manifest declares files that are not under {root}: "
            + ", ".join(absent)
        )
    files: dict[str, Any] = {}
    for relative in PORT_SOURCE_FILES:
        path = root / relative
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"files": files, "sha256": hashlib.sha256(payload).hexdigest()}


class BoundaryFingerprintWriter:
    """Append-only ``step -> {atmosphere, backend}`` JSONL at every boundary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self._stream = open(self.path, "w", encoding="utf-8")
        self._steps: list[int] = []

    def write(self, step: int, fingerprint: Mapping[str, Any]) -> None:
        if self._steps and step <= self._steps[-1]:
            raise ValueError(
                f"boundary fingerprints must ascend: {step} after {self._steps[-1]}"
            )
        self._steps.append(int(step))
        record = {"step": int(step), **dict(fingerprint)}
        self._stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._stream.flush()

    @property
    def steps(self) -> list[int]:
        return list(self._steps)

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "BoundaryFingerprintWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_boundary_fingerprints(path: str | Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = int(record.pop("step"))
            if step in records:
                raise ValueError(f"duplicate boundary fingerprint for step {step}")
            records[step] = record
    return records
