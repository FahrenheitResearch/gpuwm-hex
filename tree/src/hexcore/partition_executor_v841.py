"""Two-rank halo-exchange executor for the v8.4.1 CUDA lane (multi-GPU v1).

This is the brief-2 "missing middle": the machinery that lets one partition's
device step consume owner-produced halo values from the far rank at exactly
the points where the step's stencils read them.  Design source:
``MPAS-MULTIGPU-DESIGN-2026-08-17.md`` (Probe 2/3, D2/D3/D4/D6).

Everything here is DATA MOVEMENT, not numerics: pack is a fancy-index gather
of owner-computed values, unpack is a contiguous tail assignment into the halo
region of the local arrays.  No arithmetic is performed on field values
anywhere in this module (the ``partition_state_v841`` "HOST MOVES BYTES ONLY"
law extended over the wire), so no new physics or dycore authority exists or
is needed for it.

Ownership and consumption laws
------------------------------
* The local ordering law (``partition_assets_v841``) puts owned entities
  first, halo entities last.  A rank's halo cells are exactly
  ``cell_l2g[n_owned_cells:]`` and its non-owned edges exactly
  ``edge_l2g[n_owned_edges:]``; at P=2 every one of them is OWNED by the far
  rank (exact-cover law), which is asserted when the tables are built.
* Pack order is the RECEIVER's halo order, so unpack is a single contiguous
  ``arr[..., n_owned:] = received`` -- the receiver's halo region becomes
  owner truth in one assignment, and owned values are never written by the
  wire.  Owner-computes is therefore structural: an owned value can only ever
  be produced by this rank's own kernels.
* Vertices carry no exchanged mutable state in the v8.4.1 step (vertex-space
  quantities -- vorticity, kite interpolants -- are derived locally from
  exchanged cell/edge state), so no vertex tables exist.

Round schedule (the design's 44 rounds/step, dt=120 s, splits 3 x RK3 x
acoustic (1,3,6)):

=====  =====================================  ==========================
round  where in the step                      fields (owner truth moved)
=====  =====================================  ==========================
A x1   step boundary, after commit / before   full truth: prognostic state
       the next step's phase-1 physics        (incl. scalars) + saved diag
B x9   end of every RK stage, after the       recovered state + saved diag
       recovered-state validation and BEFORE  (no scalars: unchanged in the
       ``solve_diagnostics``                  dynamics subcycle)
C x30  inside every acoustic sub-step, after  rho_pp, rtheta_pp (cells),
       ``advance_acoustic_step`` and BEFORE   ru_p (edges)
       the 3-D divergence damping
D x3   entry of every scalar-transport stage  the 6-species scalar block
E x1   monotonic FCT, after ``fct_scale`` and scale_in, scale_out
       BEFORE ``fct_limit_horizontal``
=====  =====================================  ==========================

Two placements are stated more precisely here than in the design's line
cites, with the reasoning on record:

* B runs BEFORE ``solve_diagnostics`` (not merely "at stage entry") because
  ``h_edge`` at an owned cut edge reads the ring-1 endpoint's rho; the
  diagnostics must therefore be solved from owner-truth halos.
* C runs BETWEEN the acoustic advance and the divergence damping because the
  damping at an owned cut edge reads ring-1 ``rtheta_pp`` of the CURRENT
  sub-step, which only the owner computes exactly (the ring-1 column's own
  divergence reads ring-2 edges whose tendencies are outside the K=2 cone).
  ``ru_p`` moves in the same frame: the peer then damps owner-truth values,
  which is bitwise the owner's own damped result.

Ring-validity bookkeeping (v1): the executor enforces the round schedule as a
hard sequence law -- every step must issue exactly the expected round string,
in order, or the run refuses.  Per-stencil ring-depth accounting beyond that
is carried by the design's Probe-3 analysis (K=2 cell halo + ring-0..2 edges
cover every single-round stencil) and is *proven empirically* by the
partition-invariance gate, which byte-compares the owned-region union against
the single-GPU run at every step boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .partition_assets_v841 import PartitionLayout

__all__ = [
    "EXECUTOR_VERSION",
    "ExecutorError",
    "HaloExchangeTables",
    "HaloExchanger",
    "expected_round_sequence",
]

EXECUTOR_VERSION = "partition-executor-v841/1"

_CELL = "cell"
_EDGE = "edge"


class ExecutorError(RuntimeError):
    """A structural refusal from the two-rank executor."""


def expected_round_sequence(
    *,
    dynamics_splits: int = 3,
    acoustic_steps: Sequence[int] = (1, 3, 6),
    scalar_stages: int = 3,
    monotonic: bool = True,
) -> tuple[str, ...]:
    """The exact per-step round string the exchanger must observe.

    A is LAST: the runner fires the step-boundary round immediately after the
    step's commit, which is the same wire placement as "before the next
    step's phase-1 physics" (the design's table) while also making the
    post-commit health gate and fingerprint reconstruction see owner-truth
    halos.  Step 1's phase-1 consumes the initial condition's halos, which
    are owner truth by construction (a slice of the global init).
    """

    sequence: list[str] = []
    for _ in range(int(dynamics_splits)):
        for count in acoustic_steps:
            sequence.extend(["C"] * int(count))
            sequence.append("B")
    for stage in range(1, int(scalar_stages) + 1):
        sequence.append("D")
        if monotonic and stage == int(scalar_stages):
            sequence.append("E")
    # the final publish round: the last stage's output scalars become owner
    # truth before phase-2 consumes the halo columns (see cuda_driver's site)
    sequence.append("D")
    sequence.append("A")
    return tuple(sequence)


@dataclass(frozen=True)
class HaloExchangeTables:
    """Send/recv index law for one rank of a two-rank decomposition."""

    rank: int
    n_owned_cells: int
    n_local_cells: int
    n_owned_edges: int
    n_local_edges: int
    cell_send: np.ndarray  # local indices of MY owned cells, in the PEER's halo order
    edge_send: np.ndarray  # local indices of MY owned edges, in the PEER's halo order

    @property
    def cell_recv_count(self) -> int:
        return int(self.n_local_cells - self.n_owned_cells)

    @property
    def edge_recv_count(self) -> int:
        return int(self.n_local_edges - self.n_owned_edges)

    @classmethod
    def build(cls, layouts: Sequence[PartitionLayout], rank: int) -> "HaloExchangeTables":
        if len(layouts) != 2:
            raise ExecutorError(
                f"the v1 executor is exactly two ranks; got {len(layouts)} layouts"
            )
        if rank not in (0, 1):
            raise ExecutorError(f"rank must be 0 or 1, not {rank}")
        mine = layouts[rank]
        peer = layouts[1 - rank]

        def send_table(
            my_l2g: np.ndarray,
            my_owned: int,
            n_global: int,
            peer_l2g: np.ndarray,
            peer_owned: int,
            what: str,
        ) -> np.ndarray:
            inverse = np.full(int(n_global), -1, dtype=np.int64)
            inverse[np.asarray(my_l2g, dtype=np.int64)] = np.arange(
                my_l2g.size, dtype=np.int64
            )
            wanted_global = np.asarray(peer_l2g, dtype=np.int64)[int(peer_owned):]
            table = inverse[wanted_global]
            if table.size and (table.min() < 0 or table.max() >= int(my_owned)):
                raise ExecutorError(
                    f"{what}: the peer's halo is not covered by this rank's owned "
                    "region -- the two layouts are not a P=2 exact cover"
                )
            return np.ascontiguousarray(table)

        cell_send = send_table(
            mine.cell_l2g,
            mine.n_owned_cells,
            mine._global_cells,
            peer.cell_l2g,
            peer.n_owned_cells,
            "cells",
        )
        edge_send = send_table(
            mine.edge_l2g,
            mine.n_owned_edges,
            mine._global_edges,
            peer.edge_l2g,
            peer.n_owned_edges,
            "edges",
        )
        return cls(
            rank=int(rank),
            n_owned_cells=int(mine.n_owned_cells),
            n_local_cells=int(mine.n_local_cells),
            n_owned_edges=int(mine.n_owned_edges),
            n_local_edges=int(mine.n_local_edges),
            cell_send=cell_send,
            edge_send=edge_send,
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "executor_version": EXECUTOR_VERSION,
            "rank": int(self.rank),
            "n_owned_cells": int(self.n_owned_cells),
            "n_local_cells": int(self.n_local_cells),
            "n_owned_edges": int(self.n_owned_edges),
            "n_local_edges": int(self.n_local_edges),
            "cells_sent_per_field": int(self.cell_send.size),
            "cells_received_per_field": self.cell_recv_count,
            "edges_sent_per_field": int(self.edge_send.size),
            "edges_received_per_field": self.edge_recv_count,
        }


class HaloExchanger:
    """Executes the round schedule against resident device (or host) arrays.

    ``xp`` is the array module the fields live in -- ``cupy`` in the real run,
    ``numpy`` in the CPU tests; the pack/unpack law is identical.  The wire is
    any object with the ``PeerLink.exchange(tag, bytes) -> bytes`` contract.
    """

    def __init__(
        self,
        tables: HaloExchangeTables,
        link: Any,
        *,
        xp: Any,
        expected_sequence: Sequence[str] | None = None,
    ) -> None:
        self.tables = tables
        self.link = link
        self.xp = xp
        self._expected = (
            expected_round_sequence()
            if expected_sequence is None
            else tuple(expected_sequence)
        )
        self._step: int | None = None
        self._observed: list[str] = []
        self._send_cell = xp.asarray(tables.cell_send)
        self._send_edge = xp.asarray(tables.edge_send)
        self.stats: dict[str, dict[str, float]] = {
            letter: {"rounds": 0, "bytes_sent": 0, "bytes_received": 0, "seconds": 0.0}
            for letter in ("A", "B", "C", "D", "E")
        }

    # -- step framing ------------------------------------------------------

    def begin_step(self, step: int) -> None:
        if self._step is not None:
            raise ExecutorError(
                f"begin_step({step}) while step {self._step} is still open"
            )
        self._step = int(step)
        self._observed = []

    def end_step(self) -> None:
        if self._step is None:
            raise ExecutorError("end_step without begin_step")
        observed = tuple(self._observed)
        if observed != self._expected:
            raise ExecutorError(
                f"step {self._step} issued round sequence {''.join(observed)!r} "
                f"but the schedule law requires {''.join(self._expected)!r}"
            )
        self._step = None
        self._observed = []

    # -- the exchange core -------------------------------------------------

    def _to_host(self, packed: Any) -> np.ndarray:
        if isinstance(packed, np.ndarray):
            return np.ascontiguousarray(packed)
        return self.xp.asnumpy(packed)

    def _round(self, letter: str, fields: Sequence[tuple[str, Any, str]]) -> None:
        if self._step is None:
            raise ExecutorError(f"round {letter} issued outside an open step")
        position = len(self._observed)
        if position >= len(self._expected) or self._expected[position] != letter:
            expected = (
                self._expected[position] if position < len(self._expected) else "<end>"
            )
            raise ExecutorError(
                f"step {self._step} round {position}: schedule law expects "
                f"{expected!r} but the driver issued {letter!r}"
            )
        started = time.perf_counter()
        tables = self.tables
        parts: list[bytes] = []
        recv_specs: list[tuple[str, Any, int, tuple[int, ...], int]] = []
        for name, array, kind in fields:
            if kind == _CELL:
                n_local, n_owned = tables.n_local_cells, tables.n_owned_cells
                send_index, recv_count = self._send_cell, tables.cell_recv_count
            elif kind == _EDGE:
                n_local, n_owned = tables.n_local_edges, tables.n_owned_edges
                send_index, recv_count = self._send_edge, tables.edge_recv_count
            else:
                raise ExecutorError(f"unknown entity kind {kind!r} for field {name!r}")
            if int(array.shape[-1]) != n_local:
                raise ExecutorError(
                    f"round {letter} field {name!r}: last axis {array.shape[-1]} "
                    f"!= local {kind} count {n_local}"
                )
            packed = array[..., send_index]
            parts.append(self._to_host(packed).tobytes())
            item = np.dtype(array.dtype).itemsize
            shape = tuple(int(extent) for extent in array.shape[:-1]) + (recv_count,)
            nbytes = int(np.prod(shape, dtype=np.int64)) * item
            recv_specs.append((name, array, n_owned, shape, nbytes))

        payload = b"".join(parts)
        received = self.link.exchange(f"{letter}", payload)
        expected_bytes = sum(spec[4] for spec in recv_specs)
        if len(received) != expected_bytes:
            raise ExecutorError(
                f"round {letter}: peer sent {len(received)} bytes, the layout "
                f"law requires {expected_bytes}"
            )
        offset = 0
        for name, array, n_owned, shape, nbytes in recv_specs:
            flat = np.frombuffer(received, dtype=array.dtype, count=-1, offset=offset)
            offset += nbytes
            count = int(np.prod(shape, dtype=np.int64))
            block = flat[:count].reshape(shape)
            array[..., n_owned:] = self.xp.asarray(block)

        stats = self.stats[letter]
        stats["rounds"] += 1
        stats["bytes_sent"] += len(payload)
        stats["bytes_received"] += len(received)
        stats["seconds"] += time.perf_counter() - started
        self._observed.append(letter)

    # -- the five rounds ---------------------------------------------------

    def _state_diag_fields(
        self, state: Any, saved: Any, *, include_scalars: bool
    ) -> list[tuple[str, Any, str]]:
        fields: list[tuple[str, Any, str]] = [
            ("rho", state.rho, _CELL),
            ("rho_theta", state.rho_theta, _CELL),
            ("rho_u", state.rho_u, _EDGE),
            ("rho_w", state.rho_w, _CELL),
        ]
        if include_scalars:
            fields.append(("scalars", state.scalars, _CELL))
        fields += [
            ("theta_m", saved.theta_m, _CELL),
            ("exner", saved.exner, _CELL),
            ("density_perturbation", saved.density_perturbation, _CELL),
            ("rho_theta_perturbation", saved.rho_theta_perturbation, _CELL),
            ("pressure_perturbation", saved.pressure_perturbation, _CELL),
            ("normal_velocity", saved.normal_velocity, _EDGE),
            ("vertical_velocity", saved.vertical_velocity, _CELL),
        ]
        return fields

    def round_step_boundary(self, state: Any, saved: Any) -> None:
        """A: full truth at the committed step boundary."""

        self._round("A", self._state_diag_fields(state, saved, include_scalars=True))

    def round_stage_entry(self, state: Any, saved: Any) -> None:
        """B: recovered state + diagnostics after every RK stage recovery."""

        self._round("B", self._state_diag_fields(state, saved, include_scalars=False))

    def round_acoustic(self, acoustic: Any) -> None:
        """C: acoustic perturbations, between the advance and the damping."""

        self._round(
            "C",
            [
                ("rho_pp", acoustic.rho_pp, _CELL),
                ("rtheta_pp", acoustic.rtheta_pp, _CELL),
                ("ru_p", acoustic.ru_p, _EDGE),
            ],
        )

    def round_transport(self, scalar_stage: Any) -> None:
        """D: the six-species scalar block at every transport stage entry."""

        self._round("D", [("scalars", scalar_stage, _CELL)])

    def round_fct_scale(self, scale_in: Any, scale_out: Any) -> None:
        """E: monotonic-limiter scale factors before the flux rescale."""

        self._round(
            "E",
            [("scale_in", scale_in, _CELL), ("scale_out", scale_out, _CELL)],
        )

    # -- receipts ----------------------------------------------------------

    def receipt(self) -> dict[str, Any]:
        return {
            "executor_version": EXECUTOR_VERSION,
            "tables": self.tables.receipt(),
            "expected_sequence": "".join(self._expected),
            "rounds": {
                letter: {
                    "rounds": int(stats["rounds"]),
                    "bytes_sent": int(stats["bytes_sent"]),
                    "bytes_received": int(stats["bytes_received"]),
                    "seconds": float(stats["seconds"]),
                }
                for letter, stats in self.stats.items()
            },
            "link": self.link.receipt() if hasattr(self.link, "receipt") else None,
        }


def owned_entity_counts(layouts: Sequence[PartitionLayout]) -> Mapping[str, int]:
    """Convenience: global entity extents from any layout pair."""

    first = layouts[0]
    return {
        "cells": int(first._global_cells),
        "edges": int(first._global_edges),
        "vertices": int(first._global_vertices),
    }


# ---------------------------------------------------------------------------
# owned-region shipment law (the invariance gate's reconstruction path)
# ---------------------------------------------------------------------------
#
# The gate byte-compares the OWNED-REGION UNION against the single-GPU run:
# each rank contributes exactly its owned entities, the exact-cover law makes
# the union the whole state, and assembly is pure index movement (no
# arithmetic).  The port's own arrays put the entity index LAST; frozen-Arwen
# seam arrays may put it FIRST, so both axes are recognised -- ambiguity is a
# refusal, never a guess.


def classify_partition_axis(
    shape: Sequence[int],
    *,
    n_local_cells: int,
    n_local_edges: int,
    n_local_vertices: int,
) -> tuple[str, int | None]:
    """Return ``(kind, axis)`` for one local array shape.

    ``kind`` is ``cell``/``edge``/``vertex`` with ``axis`` the entity axis
    (last axis preferred, then first), or ``("none", None)`` when no axis
    matches a local entity count.
    """

    counts = {
        "cell": int(n_local_cells),
        "edge": int(n_local_edges),
        "vertex": int(n_local_vertices),
    }
    dims = tuple(int(extent) for extent in shape)
    if not dims:
        return "none", None
    matches: list[tuple[str, int]] = []
    for kind, count in counts.items():
        if dims[-1] == count:
            matches.append((kind, len(dims) - 1))
    if not matches and len(dims) > 1:
        for kind, count in counts.items():
            if dims[0] == count:
                matches.append((kind, 0))
    if not matches:
        return "none", None
    if len(matches) > 1:
        raise ExecutorError(
            f"shape {dims} matches more than one entity class {matches}; "
            "the shipment law refuses to guess"
        )
    return matches[0]


def owned_block(array: np.ndarray, axis: int, n_owned: int) -> np.ndarray:
    """The owned-entity slice of one local array along its entity axis."""

    slicer = [slice(None)] * array.ndim
    slicer[axis] = slice(0, int(n_owned))
    return np.ascontiguousarray(array[tuple(slicer)])


def scatter_owned_axis(
    global_out: np.ndarray,
    block: np.ndarray,
    l2g: np.ndarray,
    n_owned: int,
    axis: int,
) -> None:
    """Place one rank's owned block into the global array (exact-value)."""

    if block.dtype != global_out.dtype:
        raise ExecutorError(
            f"shipment scatter never converts dtype ({block.dtype} into "
            f"{global_out.dtype})"
        )
    index = np.asarray(l2g[: int(n_owned)], dtype=np.int64)
    slicer: list[Any] = [slice(None)] * global_out.ndim
    slicer[axis] = index
    global_out[tuple(slicer)] = block
