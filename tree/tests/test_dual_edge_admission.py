"""Dual-edge admission, tested in BOTH directions on measured readings.

An admission gate that only ever passes is not an instrument.  Each test here
either proves a mesh that integrates is admitted, or proves a mesh that is
known to die inside step 0 is refused BY NAME with the numbers a reader can
act on.

The readings are measured, not invented (the proving node, RTX 5070 Ti, 2026-08-24,
read from the registered artifacts themselves):

    x1.40962        min dvEdge/dcEdge  0.394477   integrates
    x4.163842       min dvEdge/dcEdge  0.033650   integrates (frozen anchor)
    generated g96   min dvEdge/dcEdge  0.394671   integrates
    v15.150.38857   min dvEdge/dcEdge  1.6849e-04 dies at composite step 0
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mpas_port.dual_edge_admission import (
    DualEdgeAdmissionError,
    DualEdgePolicy,
    admit_dual_edges,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_binding() -> object:
    name = "_test_dual_edge_mesh_binding"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        name, TOOLS / "mpas_mesh_binding.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _uniform(n_edges: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    dc = np.full(n_edges, 120_000.0, dtype=np.float64)
    dv = dc * ratio
    return dv, dc


def test_policy_floor_sits_between_the_measured_anchors() -> None:
    """The floor admits the roughest published mesh and refuses the defect class."""

    floor = DualEdgePolicy().minimum_dv_over_dc
    assert 0.0 < floor < 0.033650, "the frozen x4.163842 anchor must be admitted"
    assert floor > 1.6849e-04, "the measured generated-mesh dislocation class must be refused"


@pytest.mark.parametrize(
    ("label", "ratio"),
    [
        ("x1.40962", 0.394477),
        ("x4.163842", 0.033650),
        ("generated g96", 0.394671),
    ],
)
def test_meshes_that_integrate_are_admitted(label: str, ratio: float) -> None:
    dv, dc = _uniform(4_096, ratio)
    admission = admit_dual_edges(dv, dc, mesh_name=label)
    assert admission.minimum_ratio == pytest.approx(ratio)
    assert admission.edges_below_floor == 0
    assert admission.amplification == pytest.approx(1.0 / ratio)


def test_the_measured_dislocation_ratio_is_refused_by_name() -> None:
    """One collapsed dual edge in 116,565 is enough, and the refusal says which."""

    dv, dc = _uniform(116_565, 0.59)
    dv[19_786] = 6.514
    dc[19_786] = 38_657.0
    cells = np.zeros((116_565, 2), dtype=np.int64)
    cells[19_786] = (6_477, 6_650)
    with pytest.raises(DualEdgeAdmissionError) as caught:
        admit_dual_edges(dv, dc, cells_on_edge=cells, mesh_name="v15.150.38857")
    message = str(caught.value)
    assert "v15.150.38857" in message
    assert "Edge 19786" in message
    assert "6477" in message and "6650" in message
    assert "one-based" in message
    # The breakage is named, not merely asserted.
    assert "pv_apvm_v841_f32" in message
    assert "cuda_horizontal_v841.py:215-217" in message
    assert "1 of 116565 edges" in message
    # The amplification is reported as the ratio it is: 38657 / 6.514.
    assert "5934" in message or "5935" in message
    # And the remedy names the seeding that removes the defect class.
    assert "Goldberg" in message


def test_a_row_cannot_waive_the_floor_with_a_looser_policy_default() -> None:
    """The gate is only as good as the floor the caller cannot move silently."""

    binding_mod = _load_binding()

    default = DualEdgePolicy().minimum_dv_over_dc
    for name, binding in binding_mod.MESH_BINDINGS.items():
        assert binding.dual_edge_policy().minimum_dv_over_dc == default, (
            f"row {name} carries a different dual-edge floor from the module default"
        )


def test_corrupt_geometry_is_refused_before_the_ratio_is_computed() -> None:
    dv, dc = _uniform(64, 0.5)
    dc[7] = 0.0
    with pytest.raises(DualEdgeAdmissionError, match="non-finite or non-positive"):
        admit_dual_edges(dv, dc)
    dv, dc = _uniform(64, 0.5)
    dv[9] = np.nan
    with pytest.raises(DualEdgeAdmissionError, match="non-finite or negative"):
        admit_dual_edges(dv, dc)


def test_shape_disagreement_is_refused() -> None:
    with pytest.raises(DualEdgeAdmissionError, match="disagree on nEdges"):
        admit_dual_edges(np.ones(63), np.ones(64))


def test_the_index_base_is_stated_in_the_refusal() -> None:
    """The same edge printed from two doors must not look like two edges."""

    dv, dc = _uniform(64, 0.5)
    dv[3] = 1.0
    cells = np.zeros((64, 2), dtype=np.int64)
    cells[3] = (10, 11)
    with pytest.raises(DualEdgeAdmissionError) as caught:
        admit_dual_edges(dv, dc, cells_on_edge=cells, cells_on_edge_base=0)
    assert "zero-based" in str(caught.value)
