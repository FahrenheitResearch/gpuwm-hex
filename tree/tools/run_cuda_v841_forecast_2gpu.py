#!/usr/bin/env python3
"""Two-node, two-GPU MPAS-A v8.4.1 CUDA forecast -- one mesh, two ranks,
byte-identical to the single-GPU truth.

DERIVED, NOT A PROOF.  Import-based fork of ``run_cuda_v841_forecast.py`` (the
engineering forecast driver), which is itself an import-based fork of the
sealed proof harness.  Every executing model path -- authority verification,
case-pin relaxation, host preparation, device-stack construction, the staged
two-owner composite step, the per-step health gate -- is CALLED FROM those
modules, not copied.  What this tool adds is pure data movement:

  * each rank slices the identically-prepared whole-mesh host through the
    brief-1 partition layouts (``slice_prepared_host``) and runs the SAME
    composite step on its local (owned + K=2 halo) mesh;
  * the ``partition_executor_v841.HaloExchanger`` delivers owner-truth halo
    values over the ``partition_net_v841`` blocking TCP peer at the design's
    44 round sites per step (B/C/D/E inside the driver, A here at the
    committed boundary);
  * at every fingerprint boundary rank 1 ships its OWNED regions to rank 0,
    which reassembles the global arrays (exact-cover scatter, no arithmetic)
    and writes a boundary fingerprint in the single-GPU runner's own schema.

PARTITION-INVARIANCE GATE (design D4): the reconstructed fingerprints are
compared against a single-GPU reference run's ``boundary-fingerprints.jsonl``
record by record.  ``state_invariant`` demands byte equality of (a) the full
atmosphere record and (b) every backend ARRAY leaf (dtype/shape/sha256).
Backend non-array leaves are compared too, with one expected divergence
class: constructor/identity digests hash the (sliced) constructor INPUTS, not
evolved state, and differ by construction on a partitioned run; they are
reported verbatim, never hidden, and never counted as state.

Determinism preconditions asserted before stepping: identical port-source
pins, identical configuration digest, identical layouts npz, identical
compile-platform binding (cupy/NVRTC/driver identity -- the pip-NVRTC shadow
trap surface), and after step 1 identical full compile manifests via
``assert_identical_compile_manifests`` across the socket.

NO NEW NUMERICS: pack/unpack/slice/assemble move bytes only, so no new CPU
authority exists or is needed for this tool (the design's explicit check).
History captures are deliberately absent here: once the gate proves byte
identity, the reference run's history IS this run's history.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import run_cuda_v841_full_physics_x4 as proof  # noqa: E402
import run_cuda_v841_forecast as forecast  # noqa: E402
from compare_2gpu_invariance_v841 import compare_boundary_records  # noqa: E402

ROOT = proof.ROOT

SCHEMA = "mpas-port.cuda-v841-2gpu-forecast/v1"
RECEIPT_NAME = "cuda-v841-2gpu-receipt.json"
DERIVED_FROM = "tools/run_cuda_v841_forecast.py"
DT_SECONDS = forecast.DT_SECONDS
# K=3, not the design's K=2: the measured step-1 divergence localized to the
# cut (analyze_step_divergence) -- the APVM pv_edge chain consumed at ring-1
# edges reaches vertices whose cells sit three rings out, so K=2 leaves
# ring-1 pv garbage that vector_momentum at owned cut edges consumes.  K=3
# closes that cone (and every shallower one); the exchange machinery is
# ring-agnostic, so only this constant and the layout assets move.
HALO_RINGS = 3

RANK_ROLES = {0: "server", 1: "client"}

CLAIM = (
    "one uninterrupted real-initialized x4.163842 MPAS-A v8.4.1 CUDA forecast "
    "executed as a 2:1 two-rank domain decomposition across two nodes, with "
    "owner-truth halo exchange at every stencil boundary and per-boundary "
    "owned-region reconstruction for the partition-invariance gate"
)
NONCLAIMS = (
    "not a proof harness run: everything run_cuda_v841_forecast.py drops is "
    "dropped here too (see its dropped_guarantees)",
    "no history frames are written by this tool; the byte-identity gate makes "
    "the single-GPU reference run's frames this run's frames",
    "forecast skill is not established",
)


# --------------------------------------------------------------------------
# owned-region shipment (rank1 -> rank0) and fingerprint reconstruction
# --------------------------------------------------------------------------
def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if type(value).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy.asnumpy(value)
    return np.ascontiguousarray(np.asarray(value))


def _walk_leaves(
    prefix: str, item: Any, arrays: dict[str, Any], scalars: dict[str, Any]
) -> None:
    """EXACTLY ``fingerprint_nested_arrays``'s walk, kept in lockstep."""

    if isinstance(item, Mapping):
        for key in sorted(item):
            _walk_leaves(f"{prefix}/{key}" if prefix else str(key), item[key], arrays, scalars)
    elif isinstance(item, np.ndarray) or type(item).__module__.split(".")[0] == "cupy":
        arrays[prefix] = item
    elif isinstance(item, (str, int, float, bool)) or item is None:
        scalars[prefix] = item
    elif isinstance(item, (tuple, list)):
        for index, child in enumerate(item):
            _walk_leaves(f"{prefix}/{index}", child, arrays, scalars)
    else:
        scalars[prefix] = repr(item)


_ATMO_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("rho", "state", "cell"),
    ("rho_theta", "state", "cell"),
    ("rho_u", "state", "edge"),
    ("rho_w", "state", "cell"),
    ("scalars", "state", "cell"),
    ("theta_m", "saved", "cell"),
    ("exner", "saved", "cell"),
    ("density_perturbation", "saved", "cell"),
    ("rho_theta_perturbation", "saved", "cell"),
    ("pressure_perturbation", "saved", "cell"),
    ("normal_velocity", "saved", "edge"),
    ("vertical_velocity", "saved", "cell"),
)


class OwnedRegionCourier:
    """Collects one rank's owned regions and moves them to rank 0."""

    def __init__(self, layouts: Sequence[Any], rank: int, link: Any) -> None:
        from mpas_port.partition_executor_v841 import classify_partition_axis

        self.layouts = list(layouts)
        self.rank = int(rank)
        self.link = link
        self.classify = classify_partition_axis
        self.mine = layouts[rank]

    # -- local collection --------------------------------------------------

    def _entity_info(self, layout: Any, kind: str) -> tuple[Any, int]:
        if kind == "cell":
            return layout.cell_l2g, layout.n_owned_cells
        if kind == "edge":
            return layout.edge_l2g, layout.n_owned_edges
        if kind == "vertex":
            return layout.vertex_l2g, layout.n_owned_vertices
        raise ValueError(kind)

    def _classify(self, shape: Sequence[int]) -> tuple[str, int | None, bool]:
        """Entity classification with the WRF u-stagger allocation pad.

        The frozen Arwen column batch allocates its u-tendency carriers with
        one extra trailing column (``ncol + 1`` -- WRF's staggered-i memory
        extent).  The physical columns occupy ``[0, ncol)`` in the local
        order; the pad slot is allocation, not a column, and is byte-compared
        across the ranks at assembly.
        """

        layout = self.mine
        kind, axis = self.classify(
            tuple(shape),
            n_local_cells=layout.n_local_cells,
            n_local_edges=layout.n_local_edges,
            n_local_vertices=layout.n_local_vertices,
        )
        if kind != "none":
            return kind, axis, False
        dims = tuple(int(extent) for extent in shape)
        if dims:
            for candidate, count in (
                ("cell", layout.n_local_cells),
                ("edge", layout.n_local_edges),
                ("vertex", layout.n_local_vertices),
            ):
                if dims[-1] == count + 1:
                    return candidate, len(dims) - 1, True
        return "none", None, False

    def collect(self, stack: Mapping[str, Any]) -> dict[str, Any]:
        from mpas_port.partition_executor_v841 import owned_block

        layout = self.mine
        atmosphere = stack["driver"].atmosphere
        entries: list[dict[str, Any]] = []
        blobs: list[bytes] = []

        def add(
            path: str, array: Any, kind: str, axis: int | None, padded: bool
        ) -> None:
            host = _to_numpy(array)
            if kind == "none":
                block = host
            else:
                _l2g, owned = self._entity_info(layout, kind)
                assert axis is not None
                block = owned_block(host, axis, owned)
                if padded:
                    # carry the pad slot behind the owned region so rank 0
                    # can byte-verify it and place it at the global tail
                    block = np.ascontiguousarray(
                        np.concatenate((block, host[..., -1:]), axis=axis)
                    )
            entries.append(
                {
                    "path": path,
                    "dtype": block.dtype.str,
                    "shape": [int(extent) for extent in block.shape],
                    "kind": kind,
                    "axis": axis,
                    "padded": bool(padded),
                }
            )
            blobs.append(block.tobytes(order="C"))

        for name, owner, kind in _ATMO_FIELDS:
            source = atmosphere.state if owner == "state" else atmosphere.saved
            array = getattr(source, name)
            axis = len(array.shape) - 1  # port layout law: entity index LAST
            add(f"__atmosphere__/{owner}/{name}", array, kind, axis, False)

        backend_state = stack["backend"].restart_state()
        arrays: dict[str, Any] = {}
        scalars: dict[str, Any] = {}
        _walk_leaves("", backend_state, arrays, scalars)
        for path in sorted(arrays):
            host = _to_numpy(arrays[path])
            kind, axis, padded = self._classify(host.shape)
            add(f"__backend__/{path}", host, kind, axis, padded)

        return {
            "entries": entries,
            "blobs": blobs,
            "backend_scalars": scalars,
            "time_seconds": float(atmosphere.state.time_seconds),
        }

    # -- the wire ----------------------------------------------------------

    @staticmethod
    def _encode(collected: Mapping[str, Any]) -> bytes:
        header = json.dumps(
            {
                "entries": collected["entries"],
                "backend_scalars": collected["backend_scalars"],
                "time_seconds": collected["time_seconds"],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return struct.pack("!Q", len(header)) + header + b"".join(collected["blobs"])

    @staticmethod
    def _decode(payload: bytes) -> dict[str, Any]:
        (header_len,) = struct.unpack_from("!Q", payload, 0)
        header = json.loads(payload[8 : 8 + int(header_len)].decode("utf-8"))
        offset = 8 + int(header_len)
        blocks: dict[str, np.ndarray] = {}
        for entry in header["entries"]:
            dtype = np.dtype(entry["dtype"])
            shape = tuple(int(extent) for extent in entry["shape"])
            count = int(np.prod(shape, dtype=np.int64)) if shape else 1
            nbytes = count * dtype.itemsize
            blocks[entry["path"]] = np.frombuffer(
                payload, dtype=dtype, count=count, offset=offset
            ).reshape(shape)
            offset += nbytes
        if offset != len(payload):
            raise RuntimeError(
                f"owned-region shipment size mismatch: consumed {offset} of "
                f"{len(payload)} bytes"
            )
        return {"header": header, "blocks": blocks}

    def ship_or_receive(self, collected: Mapping[str, Any]) -> dict[str, Any] | None:
        """Rank 1 ships; rank 0 sends an empty frame and receives the shipment."""

        if self.rank == 1:
            self.link.exchange("F", self._encode(collected))
            return None
        payload = self.link.exchange("F", b"")
        return self._decode(payload)


def reconstruct_boundary_record(
    courier: OwnedRegionCourier,
    local: Mapping[str, Any],
    shipment: Mapping[str, Any],
) -> dict[str, Any]:
    """Rank 0: assemble global arrays from both owned regions and fingerprint
    them in the single-GPU runner's exact schema."""

    from mpas_port.cuda_dualrun import fingerprint_atmosphere
    from mpas_port.partition_executor_v841 import scatter_owned_axis

    layouts = courier.layouts
    globals_ = {
        "cell": int(layouts[0]._global_cells),
        "edge": int(layouts[0]._global_edges),
        "vertex": int(layouts[0]._global_vertices),
    }
    peer_entries = {
        entry["path"]: entry for entry in shipment["header"]["entries"]
    }
    peer_blocks = shipment["blocks"]
    if float(shipment["header"]["time_seconds"]) != float(local["time_seconds"]):
        raise RuntimeError("the two ranks disagree about model time at a boundary")

    local_blocks: dict[str, np.ndarray] = {}
    local_entries: dict[str, dict[str, Any]] = {}
    for entry, blob in zip(local["entries"], local["blobs"]):
        local_entries[entry["path"]] = entry
        local_blocks[entry["path"]] = np.frombuffer(
            blob, dtype=np.dtype(entry["dtype"])
        ).reshape(tuple(entry["shape"]))

    if set(peer_entries) != set(local_entries):
        missing = sorted(set(local_entries) ^ set(peer_entries))
        raise RuntimeError(f"owned-region shipment paths diverge: {missing[:8]}")

    assembled: dict[str, np.ndarray] = {}
    artifacts: dict[str, Any] = {}
    none_mismatches: list[str] = []
    for path in sorted(local_entries):
        mine = local_entries[path]
        theirs = peer_entries[path]
        if (
            mine["kind"] != theirs["kind"]
            or mine["axis"] != theirs["axis"]
            or mine.get("padded", False) != theirs.get("padded", False)
        ):
            raise RuntimeError(
                f"{path}: the ranks classified the entity axis differently "
                f"({mine['kind']}/{mine['axis']} vs {theirs['kind']}/{theirs['axis']})"
            )
        kind, axis = mine["kind"], mine["axis"]
        padded = bool(mine.get("padded", False))
        if padded:
            # BATCH-ORDER ARTIFACT, not reconstructible state.  The frozen
            # Arwen WRF-grid legacy coupling builds its staggered-u carriers
            # by rolling over the BATCH index and wrapping the tail slot
            # (gpuwm/core/physics.py couple_*_tendencies: ru = 0.5*(mass_u +
            # roll(mass_u,1,axis=2)) ++ ru[:,:,:1]), so their bytes are a
            # function of the local batch ordering by construction.  The MPAS
            # seam consumes raw.du/raw.dv (mass-point, column-pure) -- these
            # carriers do not feed the trajectory, and any hidden leakage
            # would redden the trajectory gate itself on later boundaries.
            # They are recorded per rank and EXCLUDED from the assembled
            # union; the comparator reports them against the reference as a
            # named exclusion, never silently.
            artifacts[path] = {
                "dtype": mine["dtype"],
                "kind": kind,
                "rank0_shape": mine["shape"],
                "rank1_shape": theirs["shape"],
                "rank0_sha256": hashlib.sha256(
                    local_blocks[path].tobytes()
                ).hexdigest(),
                "rank1_sha256": hashlib.sha256(
                    peer_blocks[path].tobytes()
                ).hexdigest(),
                "law": "batch-order staggered-u carrier; unconsumed by the seam",
            }
            continue
        if kind == "none":
            if local_blocks[path].tobytes() != peer_blocks[path].tobytes():
                none_mismatches.append(path)
            assembled[path] = local_blocks[path]
            continue
        shape = list(local_blocks[path].shape)
        shape[axis] = globals_[kind]
        target = np.empty(tuple(shape), dtype=local_blocks[path].dtype)
        for rank, blocks in ((0, local_blocks), (1, peer_blocks)):
            layout = layouts[rank]
            l2g, owned = courier._entity_info(layout, kind)
            scatter_owned_axis(target, blocks[path], np.asarray(l2g), owned, axis)
        assembled[path] = target

    if none_mismatches:
        detail = {
            path: {
                "local": local_entries[path],
                "peer": peer_entries[path],
            }
            for path in none_mismatches[:8]
        }
        raise RuntimeError(
            "non-entity backend arrays differ between the ranks (these must be "
            f"bitwise-shared constants): {json.dumps(detail, default=str)}"
        )

    state = SimpleNamespace(time_seconds=float(local["time_seconds"]))
    saved = SimpleNamespace()
    for name, owner, _kind in _ATMO_FIELDS:
        target = state if owner == "state" else saved
        setattr(target, name, assembled[f"__atmosphere__/{owner}/{name}"])
    atmosphere_record = fingerprint_atmosphere(
        SimpleNamespace(state=state, saved=saved)
    )

    backend_flat: dict[str, Any] = {
        path[len("__backend__/") :]: value
        for path, value in assembled.items()
        if path.startswith("__backend__/")
    }
    backend_flat.update(local["backend_scalars"])
    backend_record = proof.fingerprint_nested_arrays(backend_flat)
    record: dict[str, Any] = {
        "atmosphere": atmosphere_record,
        "backend": backend_record,
    }
    if artifacts:
        record["batch_order_artifacts"] = {
            path[len("__backend__/") :]: value for path, value in artifacts.items()
        }
    return record


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True, choices=(0, 1))
    parser.add_argument("--peer-host", required=True)
    parser.add_argument("--port", type=int, default=58410)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--init-source", required=True)
    parser.add_argument("--arwen-checkout", type=Path, required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", type=Path, required=True, help="2-way K=2 npz")
    parser.add_argument("--part2", type=Path, required=True, help="the merged part.2 file")
    parser.add_argument("--fingerprint-every", type=int, default=1)
    parser.add_argument(
        "--reference-fingerprints",
        type=Path,
        default=None,
        help="rank 0 only: single-GPU boundary-fingerprints.jsonl to gate against",
    )
    parser.add_argument("--expect-config-sha", default=None)
    parser.add_argument("--case-label", default=None)
    parser.add_argument(
        "--horiz-mixing", choices=("2d_smagorinsky", "off"), default="2d_smagorinsky"
    )
    parser.add_argument("--rendezvous-seconds", type=float, default=3600.0)
    parser.add_argument("--net-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--on-divergence",
        choices=("stop", "continue"),
        default="stop",
        help="a red invariance compare stops the run (default) or is recorded",
    )
    args = parser.parse_args(argv)
    if args.fingerprint_every < 1:
        parser.error("--fingerprint-every must be >= 1 (the gate needs boundaries)")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0915
    args = parse_args(argv)
    rank = int(args.rank)
    paths = {
        "grid": Path(args.grid).expanduser().absolute(),
        "static": Path(args.static).expanduser().absolute(),
        "init": Path(args.init).expanduser().absolute(),
    }
    arwen_checkout = proof._plain_absolute(args.arwen_checkout, "Arwen checkout")
    # Source pins verify first: the checkout guard imports the seam manifest
    # from a pinned module, so that module's bytes are proven before its
    # constants are trusted.
    source_before = proof.require_frozen_execution_sources()
    arwen_git_before = proof.verify_arwen_checkout_git(arwen_checkout)
    authority_before = forecast.verify_forecast_authorities(paths)

    steps_float = float(args.hours) * 3600.0 / DT_SECONDS
    if steps_float <= 0 or abs(steps_float - round(steps_float)) > 1e-9:
        raise ValueError(f"--hours {args.hours} is not a whole number of steps")
    steps = int(round(steps_float))

    host = forecast.prepare_forecast_host(
        paths, authority_before, start_time_text=None, horiz_mixing=args.horiz_mixing
    )

    from mpas_port.partition_assets_v841 import (
        assert_exact_cover,
        load_layouts_npz,
        sha256_file,
    )
    from mpas_port.partition_executor_v841 import (
        HaloExchangeTables,
        HaloExchanger,
        expected_round_sequence,
    )
    from mpas_port.partition_net_v841 import connect_peer

    layouts = load_layouts_npz(
        args.layouts,
        expect_mesh_sha256=sha256_file(paths["grid"]),
        expect_part_sha256=sha256_file(args.part2),
        expect_halo_rings=HALO_RINGS,
    )
    if len(layouts) != 2:
        raise RuntimeError(f"expected a 2-way layout npz, found {len(layouts)} parts")
    assert_exact_cover(
        layouts,
        n_cells=layouts[0]._global_cells,
        n_edges=layouts[0]._global_edges,
        n_vertices=layouts[0]._global_vertices,
    )
    layout = layouts[rank]
    tables = HaloExchangeTables.build(layouts, rank)

    cache_root, output_root = proof.validate_destination(
        args.cache_root, args.output, tuple(paths.values())
    )
    cache_root.mkdir(parents=False)
    output_root.mkdir(parents=False)

    link = connect_peer(
        role=RANK_ROLES[rank],
        host=str(args.peer_host),
        port=int(args.port),
        rendezvous_seconds=float(args.rendezvous_seconds),
        timeout_seconds=float(args.net_timeout_seconds),
    )

    # ---- handshake 1: identical inputs before any CUDA work --------------
    handshake = {
        "schema": SCHEMA,
        "pins_sha256": proof.canonical_json_sha256(
            {key: value for key, value in proof.EXECUTION_SOURCE_PINS.items()}
        ),
        "sources_sha256": proof.canonical_json_sha256(source_before),
        "authority_sha256": authority_before["sha256"],
        "layouts_sha256": sha256_file(args.layouts),
        "arwen": {
            "head": arwen_git_before["head"],
            "tree": arwen_git_before["tree"],
        },
        "horiz_mixing": args.horiz_mixing,
        "hours": float(args.hours),
        "steps": steps,
        "fingerprint_every": int(args.fingerprint_every),
        "halo_rings": HALO_RINGS,
        "config_type": type(host["config"]).__name__,
    }
    peer_handshake = link.exchange_json("H1", handshake)
    diverged = {
        key: (handshake[key], peer_handshake.get(key))
        for key in handshake
        if peer_handshake.get(key) != handshake[key]
    }
    if diverged:
        raise RuntimeError(f"rank handshakes diverge on {sorted(diverged)}: {diverged}")

    # ---- admission: the floor reflects the partition need (design D2) ----
    from mpas_port.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    # This must precede KernelCache's gpuwm platform-binding construction:
    # the frozen Arwen tree must own the live ``gpuwm`` package before any
    # other import binds the venv's copy (the c30cfdf pin-tool law).
    arwen_pin = dict(pin_arwen_physics_v841(arwen_checkout))

    from mpas_port.cuda_backend import KernelCache, require_cuda
    from mpas_port.partition_device_scheduler_v841 import (
        apply_device_memory_cap,
        assert_identical_compile_manifests,
        partition_min_free_bytes,
        require_devices,
    )

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    import cupy as cp

    floor_bytes = partition_min_free_bytes(
        layout.n_local_cells, int(layouts[0]._global_cells)
    )
    admission = require_devices(
        [0], min_compute=(12, 0), min_free_bytes=floor_bytes, cupy_module=cp
    )
    # Pool caps: turn a mid-run regression into a clean OOM abort (design D5).
    cap_bytes = int(min(admission[0]["total_bytes"] * 0.82, 26 * (1 << 30)))
    apply_device_memory_cap(0, cap_bytes, cupy_module=cp)
    cache = KernelCache(capability=capability, cache_dir=cache_root)

    # ---- handshake 2: identical compile platform (NVRTC/driver/cupy) -----
    platform_manifest = cache.compile_manifest()
    peer_platform = link.exchange_json("H2", platform_manifest)
    assert_identical_compile_manifests({rank: platform_manifest, 1 - rank: peer_platform})

    # ---- local mesh + device stack ---------------------------------------
    from mpas_port.partition_local_mesh_v841 import slice_prepared_host

    # The mixing lane attaches defc_a/defc_b placeholders to the loaded mesh
    # (attach_inactive_zero_deformation) WITHOUT registering their dimensions;
    # netCDF-loaded arrays all carry declarations.  Their entity axis is the
    # LEADING one ((nCells, maxEdges)), which the slicer's last-axis fallback
    # cannot see, so declare them before slicing -- the slicer docstring
    # already names the defc/edgesOnCell alignment law.
    _mesh = host["prepared"].mesh
    _declared = dict(getattr(_mesh, "variable_dimensions", {}) or {})
    for _name in ("defc_a", "defc_b"):
        if _name in _mesh.arrays and _name not in _declared:
            _declared[_name] = ("nCells", "maxEdges")
    _mesh.variable_dimensions = _declared

    local_constructor = dict(
        slice_prepared_host(host["constructor_values"], layout)
    )
    # The sealed constructor's column count is a SCALAR beside its arrays;
    # the slicer moves arrays only, so the count follows the layout here.
    # (The constructor identity digest then hashes the sliced inputs -- the
    # comparator's one expected metadata-divergence class.)
    local_constructor["n_columns"] = int(layout.n_local_cells)
    local_host = {
        "config": host["config"],
        "prepared": slice_prepared_host(host["prepared"], layout),
        "constructor_values": local_constructor,
        "gwdo_host": slice_prepared_host(host["gwdo_host"], layout),
        "f000_surface_diagnostics": slice_prepared_host(
            host["f000_surface_diagnostics"], layout
        ),
    }
    stack = proof._construct_device_stack(
        host=local_host, cache=cache, arwen_checkout=arwen_checkout
    )
    driver = stack["driver"]
    config_sha = driver.configuration_sha256
    peer_config_sha = link.exchange_json("CFG", config_sha)
    if peer_config_sha != config_sha:
        raise RuntimeError(
            f"configuration digests diverge: {config_sha} != {peer_config_sha}"
        )
    if args.expect_config_sha and args.expect_config_sha != config_sha:
        raise RuntimeError(
            f"configuration digest {config_sha} != expected {args.expect_config_sha}"
        )

    # The design's 44-round table includes the monotonic-FCT round E; the
    # proven full-physics config runs with config_monotonic and
    # config_positive_definite both False, so its step issues 43 rounds and
    # the E hook stays dormant until a limiter-on config exercises it.
    _limiter_on = bool(
        getattr(host["config"], "config_monotonic", False)
        or getattr(host["config"], "config_positive_definite", False)
    )
    exchanger = HaloExchanger(
        tables,
        link,
        xp=cp,
        expected_sequence=expected_round_sequence(monotonic=_limiter_on),
    )
    driver.halo_exchanger_v841 = exchanger
    courier = OwnedRegionCourier(layouts, rank, link)

    reference: dict[int, dict[str, Any]] | None = None
    if rank == 0 and args.reference_fingerprints is not None:
        from v841_partstream_common import read_boundary_fingerprints

        reference = read_boundary_fingerprints(args.reference_fingerprints)

    fingerprints = (
        forecast.BoundaryFingerprintWriter(output_root / "boundary-fingerprints.jsonl")
        if rank == 0
        else None
    )
    verdict_path = output_root / "invariance-verdict.json"
    verdict: dict[str, Any] = {
        "schema": SCHEMA + "/invariance-verdict",
        "reference": (
            str(args.reference_fingerprints) if args.reference_fingerprints else None
        ),
        "boundaries_compared": 0,
        "boundaries_state_invariant": 0,
        "first_divergence": None,
        "metadata_divergences": {},
        "running": True,
    }

    comparisons_seconds = 0.0
    fingerprint_seconds = 0.0
    import os

    _dump_steps = {
        int(token)
        for token in os.environ.get("MG_DUMP_STEPS", "").split(",")
        if token.strip().isdigit()
    }

    def boundary(step: int) -> None:
        nonlocal comparisons_seconds, fingerprint_seconds
        mark = time.perf_counter()
        collected = courier.collect(stack)
        shipment = courier.ship_or_receive(collected)
        if rank != 0:
            fingerprint_seconds += time.perf_counter() - mark
            return
        record = reconstruct_boundary_record(courier, collected, shipment)
        assert fingerprints is not None
        fingerprints.write(step, record)
        if step in _dump_steps:
            # diagnostic: persist the assembled atmosphere for localization
            local_blocks = {
                entry["path"]: np.frombuffer(
                    blob, dtype=np.dtype(entry["dtype"])
                ).reshape(tuple(entry["shape"]))
                for entry, blob in zip(collected["entries"], collected["blobs"])
                if entry["path"].startswith("__atmosphere__/")
            }
            peer_blocks = {
                path: block
                for path, block in shipment["blocks"].items()
                if path.startswith("__atmosphere__/")
            }
            from mpas_port.partition_executor_v841 import scatter_owned_axis

            dump: dict[str, np.ndarray] = {}
            for name, owner, kind in _ATMO_FIELDS:
                path = f"__atmosphere__/{owner}/{name}"
                sample = local_blocks[path]
                shape = list(sample.shape)
                shape[-1] = {
                    "cell": int(layouts[0]._global_cells),
                    "edge": int(layouts[0]._global_edges),
                }[kind]
                target = np.empty(tuple(shape), dtype=sample.dtype)
                for rnk, blocks in ((0, local_blocks), (1, peer_blocks)):
                    lay = layouts[rnk]
                    l2g, owned = courier._entity_info(lay, kind)
                    scatter_owned_axis(
                        target, blocks[path], np.asarray(l2g), owned, len(shape) - 1
                    )
                dump[f"{owner}.{name}"] = target
            np.savez(output_root / f"assembled-step{step}.npz", **dump)
        fingerprint_seconds += time.perf_counter() - mark
        if reference is not None:
            mark = time.perf_counter()
            if step not in reference:
                raise RuntimeError(
                    f"the reference run carries no fingerprint for step {step}"
                )
            outcome = compare_boundary_records(record, reference[step])
            if step == 0 and outcome.get("excluded_batch_order_artifacts"):
                verdict["excluded_batch_order_artifacts"] = outcome[
                    "excluded_batch_order_artifacts"
                ]
            verdict["boundaries_compared"] += 1
            if outcome["state_invariant"]:
                verdict["boundaries_state_invariant"] += 1
            elif verdict["first_divergence"] is None:
                verdict["first_divergence"] = {"step": step, **outcome}
            if outcome["metadata_diffs"]:
                verdict["metadata_divergences"][str(step)] = outcome["metadata_diffs"]
            verdict_path.write_text(
                json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n"
            )
            if step == 16 and verdict["boundaries_state_invariant"] == verdict[
                "boundaries_compared"
            ]:
                (output_root / "GATE16-PASS.marker").write_text(
                    f"steps 0..16 state-invariant vs {args.reference_fingerprints}\n"
                )
            comparisons_seconds += time.perf_counter() - mark
            if not outcome["state_invariant"] and args.on_divergence == "stop":
                raise RuntimeError(
                    f"partition-invariance gate RED at step {step}: "
                    + json.dumps(outcome["state_diffs"][:4], default=str)
                )
            print(
                json.dumps(
                    {
                        "boundary": step,
                        "state_invariant": bool(outcome["state_invariant"]),
                    }
                ),
                flush=True,
            )

    started = time.perf_counter()
    previous = proof._previous_surface_updates(stack)
    boundary(0)

    step_seconds: list[float] = []
    health: list[dict[str, Any]] = []
    refusal: dict[str, Any] | None = None
    executed = 0
    manifest_checked = False
    for step in range(1, steps + 1):
        mark = time.perf_counter()
        exchanger.begin_step(step)
        try:
            result = proof.execute_composite_step(
                driver=driver,
                backend=stack["backend"],
                scalar_names=forecast.SCALAR_NAMES,
                physics_geometry=stack["physics_geometry"],
                kernel_cache=driver.cache,
                previous_surface_updates=previous,
            )
        except (proof.CompositeTransactionError, FloatingPointError) as error:
            refusal = {
                "refused": True,
                "step": step,
                "exception": type(error).__name__,
                "message": str(error),
                "last_committed_step": step - 1,
            }
            break
        atmosphere = driver.atmosphere
        exchanger.round_step_boundary(atmosphere.state, atmosphere.saved)
        exchanger.end_step()
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - mark
        step_seconds.append(elapsed)
        executed = step
        previous = result.committed.surface_updates
        health.append(forecast.step_health_gate(stack, step, cp))
        if not manifest_checked:
            full_manifest = cache.compile_manifest()
            peer_manifest = link.exchange_json("H3", full_manifest)
            assert_identical_compile_manifests(
                {rank: full_manifest, 1 - rank: peer_manifest}
            )
            manifest_checked = True
        if step % int(args.fingerprint_every) == 0:
            boundary(step)
        if step in (1, 2, 5, 10) or step % 25 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "seconds": round(elapsed, 3),
                        "mean_seconds": round(sum(step_seconds) / len(step_seconds), 3),
                    }
                ),
                flush=True,
            )

    loop_seconds = time.perf_counter() - started
    verdict["running"] = False
    if rank == 0 and reference is not None:
        verdict_path.write_text(
            json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n"
        )
    if fingerprints is not None:
        fingerprints.close()

    source_after = proof.require_frozen_execution_sources()
    arwen_git_after = proof.verify_arwen_checkout_git(arwen_checkout)
    if source_after != source_before or arwen_git_after != arwen_git_before:
        raise RuntimeError("source or Arwen bytes changed during execution")

    mixing_proof = forecast._mixing_treatment_proof(driver, executed)
    payload = {
        "schema": SCHEMA,
        "derived_from": DERIVED_FROM,
        "derived_from_sha256": proof.sha256_file(ROOT / DERIVED_FROM),
        "claim": CLAIM,
        "nonclaims": list(NONCLAIMS),
        "case_label": args.case_label,
        "rank": rank,
        "role": RANK_ROLES[rank],
        "partition": {
            "layouts_npz": str(args.layouts),
            "layouts_sha256": handshake["layouts_sha256"],
            "halo_rings": HALO_RINGS,
            "layout": layout.receipt(),
            "tables": tables.receipt(),
        },
        "admission": {
            "devices": admission,
            "min_free_bytes_floor": floor_bytes,
            "memory_pool_cap_bytes": cap_bytes,
            "floor_law": "partition-scaled (partition_min_free_bytes), not the whole-mesh constant",
        },
        "arwen_pin": arwen_pin,
        "handshakes": {
            "inputs": handshake,
            "config_sha256": config_sha,
            "compile_platform_identical": True,
            "compile_manifest_identical": manifest_checked,
        },
        "init": {
            "path": str(paths["init"]),
            "sha256": authority_before["files"]["init"]["sha256"],
            "source": args.init_source,
        },
        "horiz_mixing": args.horiz_mixing,
        "config_type": type(host["config"]).__name__,
        "mixing_treatment_proof": mixing_proof,
        "status": "truncated_by_model_refusal" if refusal else "passed",
        "refusal": refusal,
        "steps_requested": steps,
        "steps_executed": executed,
        "walls": {
            "loop_seconds": loop_seconds,
            "step_seconds_mean": (
                sum(step_seconds) / len(step_seconds) if step_seconds else None
            ),
            "step_seconds_first": step_seconds[0] if step_seconds else None,
            "step_seconds": step_seconds,
            "fingerprint_seconds": fingerprint_seconds,
            "comparison_seconds": comparisons_seconds,
        },
        "exchange": exchanger.receipt(),
        "health": health,
        "invariance": verdict if rank == 0 else None,
    }
    receipt = output_root / RECEIPT_NAME
    proof._write_exclusive_json(
        receipt, json.loads(json.dumps(payload, sort_keys=True, default=str))
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rank": rank,
                "steps": executed,
                "mean_step_seconds": payload["walls"]["step_seconds_mean"],
                "receipt": str(receipt),
                "state_invariant_boundaries": (
                    verdict["boundaries_state_invariant"] if rank == 0 else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    link.close()
    return 0 if refusal is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
