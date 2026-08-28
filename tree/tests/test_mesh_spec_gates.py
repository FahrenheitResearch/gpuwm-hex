"""A spec the BUILD refuses must be refused by SIZING first.

THE DEFECT THESE PIN, MEASURED 2026-08-28 on the current engine.
``rw_mpas_mesh --dry-run`` sizes a resolution spec, prints a receipt and
exits 0 without applying the gates its own build applies.  Two designs were
reported lost to that, and both reproduce here:

* a 0.75/6/75 km design sized clean and the build refused it in 87 ms on the
  transition-band gate -- pure arithmetic on the same gradient the dry-run
  receipt had just printed;
* a 0.75/3/15/75 km design sized clean, cleared that gate, spent 1,251 s
  relaxing and deriving 217,621 cells, and was refused on the 200 m
  shortest-dual-edge floor at edge 562175 (36.8 m over an 886 m dcEdge) with
  no grid written.  The report that opened this lane read 711 s and 39.2 m
  for the same shape on another box.

The first is decidable from the spec and is now refused by sizing.  The
second is not decidable from a spec, and the fix for it is that sizing SAYS
so, with the quantities that decide it, instead of being silent.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore import mesh_spec_gates as gates  # noqa: E402
from hexcore.cli import build_parser, main  # noqa: E402
from hexcore.swath import sizing  # noqa: E402
from hexcore.swath.errors import SwathRefusal  # noqa: E402


#: A spec whose ramp is too steep for the surgery locality.  Reconstructed
#: from the reported 0.75/6/75 km design; the exact authored spec is not
#: available, so what is pinned here is the FAMILY and the refusal, not the
#: reporter's own number.
STEEP = {
    "name": "spec-gate-steep",
    "background_km": 75.0,
    "regions": [
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 400.0},
         "spacing_km": 6.0, "transition_cells": 40.0},
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 100.0},
         "spacing_km": 0.75, "transition_cells": 40.0},
    ],
}

#: The revised design: it CLEARS the transition-band gate, which is what
#: makes it the case the second gate has to speak about.
DEEP = {
    "name": "spec-gate-deep",
    "background_km": 75.0,
    "regions": [
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 800.0},
         "spacing_km": 15.0, "transition_cells": 108.0},
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 400.0},
         "spacing_km": 3.0, "transition_cells": 108.0},
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 120.0},
         "spacing_km": 0.75, "transition_cells": 108.0},
    ],
}

#: A spec no length floor can bind: 15 km finest is above every threshold in
#: :func:`hexcore.mesh_spec_gates.short_dual_edge_exposure`.
COARSE = {
    "name": "spec-gate-coarse",
    "background_km": 120.0,
    "regions": [
        {"shape": {"kind": "cap", "center_deg": [39.0, -98.0], "radius_km": 1200.0},
         "spacing_km": 15.0, "transition_cells": 81.0},
    ],
}


def _engine_or_skip() -> Path:
    try:
        return sizing.resolve_engine(None)
    except SwathRefusal as refusal:
        pytest.skip(f"rw_mpas_mesh is not staged on this box: {refusal}")


def _spec_file(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / f"{spec['name']}.json"
    path.write_text(
        json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    return path


def _engine(engine: Path, arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(engine), *arguments], capture_output=True, text=True, check=False
    )


# ---------------------------------------------------------------------------
# the defect, on the artifact
# ---------------------------------------------------------------------------
def test_the_generator_still_sizes_a_spec_its_own_build_refuses(tmp_path) -> None:
    """The reproduction, against the exe rather than a story about it.

    IF THIS GOES RED because ``--dry-run`` began refusing, the engine has
    grown the gate and the Python one beside it is a guard whose defect is
    fixed: retire it rather than patch this test.
    """

    engine = _engine_or_skip()
    path = _spec_file(tmp_path, STEEP)

    sized = _engine(engine, ["--spec", str(path), "--dry-run"])
    assert sized.returncode == 0, sized.stderr
    receipt = json.loads(sized.stdout)
    gradient = receipt["steepest_requested_gradient_percent_per_cell"]
    assert gradient > gates.MAX_GRADIENT_PER_CELL * 100.0

    built = _engine(
        engine, ["--spec", str(path), "--out", str(tmp_path / "g.nc"), "--clobber"]
    )
    assert built.returncode != 0
    assert "surgery locality" in built.stderr
    assert not (tmp_path / "g.nc").exists()


def test_a_spec_generation_refuses_is_refused_by_sizing_first(tmp_path) -> None:
    """The fix, stated as the thing that was missing.

    Sizing had the number in its hand -- the dry-run receipt prints it -- and
    returned it as a row instead of applying it as a gate.
    """

    _engine_or_skip()
    with pytest.raises(gates.MeshSpecRefusal) as refusal:
        sizing.dry_run(STEEP)
    assert "surgery locality" not in str(refusal.value)  # our words, not a relay
    assert "before anything is built" in str(refusal.value)


def test_the_refusal_carries_the_measurement_the_limit_and_a_numeric_remedy(
    tmp_path,
) -> None:
    _engine_or_skip()
    with pytest.raises(gates.MeshSpecRefusal) as refusal:
        sizing.dry_run(STEEP)
    text = str(refusal.value)
    assert "measured" in text and "% per cell" in text
    assert f"{gates.MAX_GRADIENT_PER_CELL * 100.0:.4f} % per cell" in text
    assert "mesh/hierarchy.rs" in text
    # The remedy is a spec edit with numbers in it, not "widen the transition".
    assert "transition_cells 40 ->" in text


def test_the_front_door_refuses_before_building_and_exits_two(
    tmp_path, capsys
) -> None:
    _engine_or_skip()
    path = _spec_file(tmp_path, STEEP)
    assert main(["mesh-plan", "--spec", str(path)]) == 2
    assert "refused before anything is built" in capsys.readouterr().err


def test_mesh_plan_is_a_subcommand_of_the_console_script() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    assert "mesh-plan" in commands


# ---------------------------------------------------------------------------
# the gate a spec cannot decide is NAMED, not skipped
# ---------------------------------------------------------------------------
def test_a_spec_that_clears_the_band_gate_still_reports_the_length_floor() -> None:
    _engine_or_skip()
    receipt = sizing.dry_run(DEEP)
    applied = receipt["gates_applied_by_hexcore"]
    assert applied["transition_band"]["band_cells"] >= gates.MIN_TRANSITION_BAND_CELLS
    edge = applied["short_dual_edge_floor"]
    assert applied["gates_this_sizing_cannot_apply"] == ["short_dual_edge_floor"]
    assert edge["decidable_before_the_build"] is False
    assert edge["limit_m"] == 200.0
    assert edge["verdict"] == "predicted_refusal"
    # A 0.75 km finest request: no measured graded mesh's dv/finest ratio
    # reaches 200 m from there, so the whole predicted band is under the
    # floor.  The reported build measured 39.2 m.
    low, high = edge["predicted_shortest_dual_edge_m"]
    assert low < high < 200.0
    assert edge["first_ladder_level_that_can_carry_a_refusal"] == 4
    assert "711" not in edge["earliest_this_can_be_known"]  # a level, not a duration
    assert "level 4 of 7" in edge["earliest_this_can_be_known"]


def test_a_coarse_spec_reports_that_the_length_floor_cannot_bind() -> None:
    exposure = gates.short_dual_edge_exposure(COARSE)
    assert exposure["verdict"] == "cannot_bind"
    assert exposure["finest_requested_spacing_km"] == 15.0
    assert exposure["first_ladder_level_that_can_carry_a_refusal"] is None
    assert min(exposure["predicted_shortest_dual_edge_m"]) > 200.0


def test_the_prediction_agrees_with_every_mesh_that_actually_built() -> None:
    """The instrument, tested in the direction that would embarrass it.

    A first draft predicted ``surgery flag floor x finest spacing`` and
    called every shipped 4 km swath a refusal while five real meshes had
    built.  A predictor is only worth printing if it reproduces the meshes
    that exist, so every row of the measured table is replayed as a spec.
    """

    for row in gates.MEASURED_GRADED_MESHES:
        label, finest, background, measured_m, ratio, emitted = row
        spec = {
            "name": label,
            "background_km": background,
            "regions": [
                {"shape": {"kind": "cap", "center_deg": [0.0, 0.0],
                           "radius_km": 400.0},
                 "spacing_km": finest, "transition_cells": 81.0},
            ],
        }
        exposure = gates.short_dual_edge_exposure(spec)
        low, high = exposure["predicted_shortest_dual_edge_m"]
        # The band's own endpoints are two of these rows, so the comparison
        # is loosened by a relative epsilon and by nothing else.
        assert low * (1 - 1e-9) <= measured_m <= high * (1 + 1e-9), (
            f"{label}: {measured_m} outside {low}-{high}"
        )
        assert ratio == pytest.approx(gates.SURGERY_FLAG_FLOOR_DV_OVER_DC, abs=0.003)
        if emitted:
            assert measured_m > gates.MIN_DV_EDGE_M, label
            assert exposure["verdict"] != "predicted_refusal", (
                f"{label} built at {measured_m:.1f} m and the predictor calls "
                "it a refusal"
            )
        else:
            assert measured_m < gates.MIN_DV_EDGE_M, label
            assert exposure["verdict"] == "predicted_refusal", label


def test_the_length_floor_is_never_a_refusal() -> None:
    """It is a prediction, and a prediction may not refuse.

    Refusing a spec on a prediction would refuse meshes the generator would
    have emitted -- the 7,500 m episode, in the other direction.
    """

    _engine_or_skip()
    receipt = sizing.dry_run(DEEP)
    assert receipt["predicted_cells"] > 0.0
    assert (
        receipt["gates_applied_by_hexcore"]["short_dual_edge_floor"]["verdict"]
        == "predicted_refusal"
    )


# ---------------------------------------------------------------------------
# the arithmetic, with no engine in the room
# ---------------------------------------------------------------------------
def test_the_band_arithmetic_crosses_six_cells_at_the_stated_ceiling() -> None:
    ceiling = gates.MAX_GRADIENT_PER_CELL
    assert gates.transition_band_cells(ceiling * 0.999) > 6.0
    assert gates.transition_band_cells(ceiling * 1.001) < 6.0
    # The published variable-resolution mesh, 1.53 % per cell, spends 45.6
    # cells crossing a doubling -- 7.6x the floor.
    assert gates.transition_band_cells(0.0153) == pytest.approx(45.649, abs=0.01)
    assert gates.transition_band_cells(0.0) == float("inf")


def test_the_ladder_is_the_generators_own() -> None:
    assert gates.ladder_km(DEEP) == [
        75.0, 37.5, 18.75, 9.375, 4.6875, 2.34375, 1.171875, 0.75
    ]
    assert gates.ladder_km({"background_km": 120.0, "regions": []}) == [120.0]


def test_widening_the_ramp_scales_whichever_spelling_a_region_uses() -> None:
    widened = gates.scaled_transitions(
        {
            "background_km": 75.0,
            "regions": [
                {"spacing_km": 3.0, "transition_cells": 40.0},
                {"spacing_km": 6.0, "transition_km": 200.0},
            ],
        },
        2.5,
    )
    assert widened["regions"][0]["transition_cells"] == 100.0
    assert widened["regions"][1]["transition_km"] == 500.0


def test_a_receipt_without_the_gradient_refuses_rather_than_skipping() -> None:
    with pytest.raises(gates.MeshSpecRefusal) as refusal:
        gates.gates_from_receipt(DEEP, {"predicted_cells": 1000.0})
    assert "cannot be applied here" in str(refusal.value)


# ---------------------------------------------------------------------------
# the constants are transcribed, so a drift must be findable
# ---------------------------------------------------------------------------
def _rust_mesh_source(engine: Path) -> Path | None:
    """``crates/rw-mpas/src/mesh`` beside a checkout-built engine, or None."""

    for parent in engine.resolve().parents:
        candidate = parent / "crates" / "rw-mpas" / "src" / "mesh"
        if candidate.is_dir():
            return candidate
    return None


def test_every_constant_still_reads_the_same_in_the_rust_that_enforces_it() -> None:
    """Skips on a box with only a staged binary, and says what went unchecked."""

    engine = _engine_or_skip()
    source = _rust_mesh_source(engine)
    if source is None:
        pytest.skip(
            f"{engine} is not a checkout build, so the rw-mpas mesh source is "
            "not beside it and the transcribed constants "
            "(SURGERY_LOCALITY_CELLS, MIN_DV_EDGE_M, MIN_DV_OVER_DC, "
            "LEVEL_SHIP_FLOOR_DV_OVER_DC, SURGERY_FLAG_FLOOR_DV_OVER_DC) "
            "go unchecked on this box"
        )
    hierarchy = (source / "hierarchy.rs").read_text(encoding="utf-8")
    validate = (source / "validate.rs").read_text(encoding="utf-8")
    surgery = (source / "surgery.rs").read_text(encoding="utf-8")
    assert (
        f"const SURGERY_LOCALITY_CELLS: f64 = {gates.SURGERY_LOCALITY_CELLS};"
        in hierarchy
    )
    assert "band_cells < 2.0 * SURGERY_LOCALITY_CELLS" in hierarchy
    assert f"min_dv_edge_m: {gates.MIN_DV_EDGE_M}," in validate
    assert f"min_dv_over_dc: {gates.MIN_DV_OVER_DC}," in validate
    assert f"flag_floor: {gates.SURGERY_FLAG_FLOOR_DV_OVER_DC}," in surgery
    assert f"ship_floor: {gates.LEVEL_SHIP_FLOOR_DV_OVER_DC}," in surgery
