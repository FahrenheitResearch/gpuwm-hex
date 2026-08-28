"""Cell-coordination admission, tested in BOTH directions on measured readings.

An admission gate that only ever passes is not an instrument.  Each test here
either proves a mesh that forecasts is admitted, or proves the mesh that is
known to die is refused BY NAME with the numbers a reader can act on.

The readings are measured, read from the registered grid files themselves on
the proving RTX 5090 (RTX 5090, 2026-08-26, ``evidence/graded-blowup-20260826/``):

    v20.80.151649  {5: 1029, 6: 149603, 7: 1017}          completed 6 h
    v16.66.195630  {4: 1, 5: 1028, 6: 193584, 7: 1016, 8: 1}
                                                          died at step 23/36

Both meshes come from the same generator, the same campaign and the same init
pipeline.  ``v20.80.151649`` ran at 94.9% of its own Courant limit with the
WORSE dual-edge amplification of the two (24.34x against 24.03x), so neither
the timestep margin nor the TRiSK amplification separates them; the
coordination histogram does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from hexcore.cell_coordination_admission import (
    CellCoordinationAdmissionError,
    CellCoordinationPolicy,
    admit_cell_coordination,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"

#: The mesh that completed a 6 h forecast, read from run20.grid.nc.
V20_HISTOGRAM = {5: 1029, 6: 149603, 7: 1017}
#: The mesh that died at step 23 of 36, read from fit.grid.nc.
V16_HISTOGRAM = {4: 1, 5: 1028, 6: 193584, 7: 1016, 8: 1}
#: The one 4-coordinated cell, and the one 8-coordinated cell beside it.
V16_QUAD_CELL = 195_615
V16_OCTA_CELL = 168_727


def _counts(histogram: dict[int, int]) -> np.ndarray:
    return np.concatenate(
        [np.full(count, edges, dtype=np.int64) for edges, count in sorted(histogram.items())]
    )


def _load_binding() -> object:
    name = "_test_cell_coordination_mesh_binding"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, TOOLS / "mpas_mesh_binding.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


# --------------------------------------------------------------- admitting
def test_the_mesh_that_completed_six_hours_is_admitted() -> None:
    admission = admit_cell_coordination(_counts(V20_HISTOGRAM), mesh_name="v20.80.151649")
    assert admission.cells_below_floor == 0
    assert admission.minimum_edges_on_cell == 5
    assert admission.histogram == V20_HISTOGRAM
    assert admission.count == sum(V20_HISTOGRAM.values())


def test_a_published_goldberg_histogram_is_admitted() -> None:
    """x1.40962's class: twelve pentagons and nothing else below six."""

    admission = admit_cell_coordination(_counts({5: 12, 6: 40_950}), mesh_name="x1.40962")
    assert admission.cells_below_floor == 0
    assert admission.coordination_defect == 12


def test_high_coordination_is_admitted_because_it_was_measured_harmless() -> None:
    """The floor is one-sided by measurement, not by symmetry.

    The 8-coordinated cell sits in the SAME mesh as the refused 4-coordinated
    one and is nowhere in the top forty cells by theta growth in any arm, so
    refusing it would be a gate with no breakage behind it.
    """

    admission = admit_cell_coordination(_counts({5: 1028, 6: 193_584, 7: 1016, 8: 1}))
    assert admission.cells_below_floor == 0
    assert admission.maximum_edges_on_cell == 8


# --------------------------------------------------------------- refusing
def test_the_mesh_that_died_is_refused() -> None:
    with pytest.raises(CellCoordinationAdmissionError) as caught:
        admit_cell_coordination(_counts(V16_HISTOGRAM), mesh_name="v16.66.195630")
    message = str(caught.value)
    assert "v16.66.195630" in message
    assert "4 edges" in message
    assert "1 of 195630 cells" in message


def test_the_refusal_names_the_breakage_and_the_remedy() -> None:
    """Gate law: a gate that does not name its breakage does not exist."""

    with pytest.raises(CellCoordinationAdmissionError) as caught:
        admit_cell_coordination(_counts(V16_HISTOGRAM), mesh_name="v16.66.195630")
    message = str(caught.value)
    assert "THE BREAKAGE THIS PREVENTS, MEASURED" in message
    assert "REMEDY: regenerate the mesh" in message
    # the measurement that killed each retired candidate rides in the refusal
    assert "step 23 of 36" in message
    assert "step 31 of 48" in message
    assert "same model time" in message.lower()
    assert "24.34x" in message and "24.03x" in message
    # and it refuses to sell a smaller timestep as the fix
    assert "A SMALLER TIMESTEP IS NOT THE REMEDY" in message


def test_the_refusal_records_that_20s_dies_too() -> None:
    """The 20 s arm finishes an hour and then dies at 2 h 45 m; both halves stay.

    Leg B/C finished 180 of 180 steps byte identically, which is easy to
    remember as "20 s works". Leg D ran the same thing to three hours and the
    same cell reached 281 m/s at step 495 of 540. Recording only the hour would
    turn a delay into a cure, so the refusal has to carry the death as well.
    """

    with pytest.raises(CellCoordinationAdmissionError) as caught:
        admit_cell_coordination(_counts(V16_HISTOGRAM), mesh_name="v16.66.195630")
    message = str(caught.value)
    assert "281 m/s" in message
    assert "step 495 of 540" in message
    assert "does not buy a forecast" in message


def test_the_remedy_names_the_fixed_generator_and_the_replacement_row() -> None:
    """'Regenerate' alone was a coin flip; it is not any more, and the refusal
    has to say which it is.

    Measured 2026-08-26: v16.66.195630 (cell 195615) and the registered 15 km
    row v15.60.224210 (cell 224206) both carried one, both at the end of the
    cell array; v20.80.151649 did not. Re-rolling with the SAME generator
    therefore had even odds of reproducing the defect, and the refusal said so.
    The generator-side fix landed the same day (gpuwm
    the meshgen coordination work): surgery now refuses its own emission below
    five, and every spec row regenerates clean. A refusal that still sent the
    reader to a coin flip -- or that failed to name the row that replaces this
    one -- would be pointing at a remedy that no longer exists.
    """

    with pytest.raises(CellCoordinationAdmissionError) as caught:
        admit_cell_coordination(_counts(V16_HISTOGRAM), mesh_name="v16.66.195630")
    message = str(caught.value)
    assert "coin flip" in message
    assert "two of the three graded meshes" in message.lower()
    # the durable fix, and the fact that it has been made
    assert "refuses its own emission below five" in message
    assert "the meshgen coordination work" in message
    assert "v16.66.195629" in message
    # and the gate does not pretend it is retired by that fix
    assert "not retired" in message


def test_a_triangle_cell_is_refused_too() -> None:
    with pytest.raises(CellCoordinationAdmissionError) as caught:
        admit_cell_coordination(np.array([3] + [6] * 99, dtype=np.int64))
    assert "3 edges" in str(caught.value)


def test_negative_coordination_is_corruption_not_roughness() -> None:
    with pytest.raises(CellCoordinationAdmissionError) as caught:
        admit_cell_coordination(np.array([6, -1, 6], dtype=np.int64))
    assert "corrupt" in str(caught.value)


def test_an_empty_or_two_dimensional_array_is_refused() -> None:
    for bad in (np.zeros((0,), dtype=np.int64), np.zeros((4, 2), dtype=np.int64)):
        with pytest.raises(CellCoordinationAdmissionError):
            admit_cell_coordination(bad)


# --------------------------------------------------------------- the policy
def test_the_policy_states_its_floor_and_validates() -> None:
    policy = CellCoordinationPolicy()
    assert policy.minimum_edges_on_cell == 5
    payload = policy.as_dict()
    assert payload["schema"] == "gpuwm-hex.cell-coordination-admission/v1"
    assert "surgery" in payload["description"]
    with pytest.raises(CellCoordinationAdmissionError):
        CellCoordinationPolicy(minimum_edges_on_cell=2).validate()


def test_the_histogram_is_recorded_even_when_it_admits() -> None:
    """A mesh that passes today is still characterised in its own receipt."""

    payload = admit_cell_coordination(_counts(V20_HISTOGRAM)).as_dict()
    assert payload["histogram"] == {"5": 1029, "6": 149603, "7": 1017}
    assert payload["cells_below_floor"] == 0
    assert payload["policy"]["minimum_edges_on_cell"] == 5


# --------------------------------------------------------------- the wiring
def test_every_registered_row_carries_the_coordination_policy() -> None:
    module = _load_binding()
    assert module.MESH_BINDINGS, "the registry is empty"
    for name, row in module.MESH_BINDINGS.items():
        policy = row.cell_coordination_policy()
        assert policy.minimum_edges_on_cell == 5, name


def test_bind_calls_the_gate_and_records_it() -> None:
    """The gate is wired into bind_mesh, not merely importable."""

    source = (TOOLS / "mpas_mesh_binding.py").read_text(encoding="utf-8")
    assert "admit_cell_coordination(" in source
    assert 'observed["cell_coordination_admission"]' in source
    assert "_grid_cell_coordination(" in source


def test_the_registry_row_records_the_explained_blowup() -> None:
    """The v16 row's 'NOT YET EXPLAINED' is retired by measurement, not deleted."""

    module = _load_binding()
    notes = module.MESH_BINDINGS["v16.66.195630"].notes
    assert "NOT YET EXPLAINED" not in notes
    assert "4-COORDINATED CELL" in notes.upper()
    assert "195615" in notes
    assert "cell_coordination_admission" in notes
    # the row must not read as though 20 s gave this mesh a forecast
    assert "281 m/s" in notes
    assert "NEVER COMPLETED ONE BEYOND AN HOUR" in notes.upper()
