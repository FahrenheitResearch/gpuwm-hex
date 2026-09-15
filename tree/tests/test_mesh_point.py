"""``gpuwm-hex mesh-plan --point``: the ladder, the pricing, the refusals.

Card-free and generator-free by construction: what needs ``rw_mpas_mesh`` is
faked with a tiny executable that prints the receipt shape the real one
prints, and what needs a card is priced on the admission surface's own rows.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore import mesh_point  # noqa: E402
from hexcore.cli import build_parser, main  # noqa: E402
from hexcore.mesh_point import PointPlanRefusal  # noqa: E402

REGISTERED_SPEC = ROOT.parent.parent.parent / "gpuwm" / "evidence" / "fine-mesh-20260829" / "meshes" / "v0.9.120.110533.spec.json"


# ---------------------------------------------------------------------------
# the ladder is the registered recipe, moved
# ---------------------------------------------------------------------------
def test_the_ladder_at_the_registered_point_is_the_registered_spec() -> None:
    spec = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    rows = [(r["shape"]["radius_km"], r["spacing_km"], r["transition_km"]) for r in spec["regions"]]
    assert spec["background_km"] == 120.0
    assert rows == [
        (3289.375, 60.0, 1080.0), (1669.375, 30.0, 540.0), (859.375, 15.0, 270.0),
        (454.375, 7.5, 135.0), (251.875, 3.75, 67.5), (150.625, 1.875, 33.75),
        (100.0, 0.9375, 16.875),
    ]
    assert all(r["shape"]["center_deg"] == [35.0, -97.0] for r in spec["regions"])


@pytest.mark.skipif(not REGISTERED_SPEC.is_file(), reason="the registered spec of record is not on this machine")
def test_the_ladder_reproduces_the_spec_of_record_to_the_digit() -> None:
    reference = json.loads(REGISTERED_SPEC.read_text(encoding="utf-8"))
    spec = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    for ours, theirs in zip(spec["regions"], reference["regions"]):
        assert ours["shape"]["radius_km"] == pytest.approx(theirs["shape"]["radius_km"])
        assert ours["spacing_km"] == theirs["spacing_km"]
        assert ours["transition_km"] == theirs["transition_km"]
    assert len(spec["regions"]) == len(reference["regions"])


def test_moving_the_point_moves_every_cap_and_nothing_else() -> None:
    here = mesh_point.ladder_spec((39.1, -94.58), fine_dx_m=937.5, radius_km=100.0)
    there = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    for a, b in zip(here["regions"], there["regions"]):
        assert a["shape"]["center_deg"] == [39.1, -94.58]
        assert a["shape"]["radius_km"] == b["shape"]["radius_km"]
        assert a["spacing_km"] == b["spacing_km"]


def test_a_spacing_off_the_ladder_is_refused_naming_both_rungs() -> None:
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.ladder_rungs(120.0, 900.0)
    text = str(refusal.value)
    assert "937.5 m" in text and "468.75 m" in text
    assert "ALWAYS FINER" in text


def test_the_ladder_rungs_halve_from_the_background() -> None:
    assert mesh_point.ladder_rungs(120.0, 937.5) == [60.0, 30.0, 15.0, 7.5, 3.75, 1.875, 0.9375]
    assert mesh_point.ladder_rungs(120.0, 1875.0) == [60.0, 30.0, 15.0, 7.5, 3.75, 1.875]
    with pytest.raises(PointPlanRefusal):
        mesh_point.ladder_rungs(120.0, 120_000.0)


def test_point_parsing_refuses_what_is_not_a_point() -> None:
    assert mesh_point.parse_point("39.1,-94.58") == (39.1, -94.58)
    assert mesh_point.parse_point("39.1, 265.42") == (39.1, pytest.approx(-94.58))
    for bad in ("39.1", "x,y", "95,0"):
        with pytest.raises(PointPlanRefusal):
            mesh_point.parse_point(bad)


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
def test_the_cull_prediction_bounds_the_registered_cull_from_above_within_three_percent() -> None:
    """r0.9.120.40520: a 125 km cap (pad 1.25) of the 100 km core -> 40,520 cells."""

    spec = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    cull = mesh_point.predicted_cull_cells(
        spec=spec, radius_km=100.0, pad_scale=1.25, attained_fine_km=0.94699,
    )
    assert cull["basis"] == "area_integral"
    assert 40_520 <= cull["predicted_cells"] <= 40_520 * 1.03
    assert cull["fine_flat_radius_km"] == pytest.approx(83.125)


def test_the_spacing_profile_is_flat_inside_the_ramp_and_ramps_to_the_next_rung() -> None:
    spec = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    profile = mesh_point.spacing_profile(spec)
    assert profile[0] == (100.0, 0.9375, 16.875)
    assert mesh_point.spacing_at(profile, 50.0, 120.0) == 0.9375
    assert mesh_point.spacing_at(profile, 100.0, 120.0) == pytest.approx(1.875)
    assert mesh_point.spacing_at(profile, 110.0, 120.0) == pytest.approx(1.875)
    assert mesh_point.spacing_at(profile, 150.625, 120.0) == pytest.approx(3.75)
    assert mesh_point.spacing_at(profile, 5000.0, 120.0) == 120.0


def test_the_card_table_prices_on_the_limited_area_row_with_the_models_own_margin() -> None:
    card = mesh_point.resolve_card("32gb")
    assert card["admission_card"] == "32gib-170sm" and card["generator_card"] == "rtx-5090"
    verdict = mesh_point.device_verdict(42_000, card)
    assert verdict["row"] == "limited-area/170sm" and verdict["row_measured"]
    assert verdict["fits"] and verdict["short_by_mib"] == 0.0
    assert verdict["required_free_mib"] == pytest.approx(
        verdict["predicted_mib"] + verdict["margin_mib"], abs=0.2)
    small = mesh_point.device_verdict(42_000, card, budget_mib=4_000.0)
    assert not small["fits"] and small["short_by_mib"] > 0.0
    assert small["budget_basis"] == "--vram-gib"


def test_an_unmeasured_card_is_refused_by_name() -> None:
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.resolve_card("h100")
    assert "not a card this plan can price" in str(refusal.value)
    assert mesh_point.resolve_card("10gb")["generator_card"] is None


def test_the_timestep_is_the_largest_anchored_one_under_the_courant_limit() -> None:
    chosen = mesh_point.choose_timestep(869.25, fine_dx_m=937.5)
    assert chosen["dt_seconds"] == 5.0 and chosen["cumulus_scheme"] is None
    assert chosen["courant_limit_seconds"] == pytest.approx(6.2586)
    coarse = mesh_point.choose_timestep(20_000.0, fine_dx_m=15_000.0)
    assert coarse["dt_seconds"] == 120.0 and coarse["cumulus_scheme"] == "gf"
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.choose_timestep(300.0, fine_dx_m=300.0)
    assert "no anchored timestep fits" in str(refusal.value)


def test_the_cull_region_is_a_cap_at_the_pad_and_a_pad_under_one_is_refused() -> None:
    region = mesh_point.cull_region((39.1, -94.58), 100.0, 1.35)
    assert region == {"kind": "cap", "center_deg": [39.1, -94.58], "radius_km": 135.0}
    with pytest.raises(PointPlanRefusal):
        mesh_point.cull_region((39.1, -94.58), 100.0, 0.9)


# ---------------------------------------------------------------------------
# the door, with a faked generator
# ---------------------------------------------------------------------------
FAKE_RECEIPT = {
    "predicted_cells": 110_437.7,
    "footprint_mib": 14_515.9,
    "card": "rtx-5090",
    "steepest_requested_gradient_percent_per_cell": 10.97,
    "ladder_snap": {"moved": False, "regions": []},
    "region_attainment": [{"attained_spacing_km": 0.9470}],
}


@pytest.fixture
def fake_mesh_exe(tmp_path: Path) -> Path:
    """An ``rw_mpas_mesh`` that prints one receipt for any ``--dry-run``."""

    script = tmp_path / "rw_mpas_mesh.py"
    script.write_text(
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "assert '--dry-run' in args, args\n"
        f"print({json.dumps(FAKE_RECEIPT)!r})\n",
        encoding="utf-8",
    )
    if sys.platform == "win32":
        launcher = tmp_path / "rw_mpas_mesh.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        launcher = tmp_path / "rw_mpas_mesh"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    return launcher


def test_the_plan_prices_the_parent_the_cull_and_the_card(fake_mesh_exe: Path, monkeypatch) -> None:
    from hexcore import mesh_spec_gates

    monkeypatch.setattr(
        mesh_spec_gates, "gates_from_receipt",
        lambda spec, receipt, measure=None: {
            "transition_band": {"steepest_gradient_percent_per_cell": 10.97, "band_cells": 6.5,
                                "band_cells_floor": 6, "gradient_percent_per_cell_ceiling": 12.25},
            "short_dual_edge_floor": {"verdict": "not decidable", "limit_m": 200, "gate": "storage"},
        },
    )
    parser = build_parser()
    arguments = parser.parse_args([
        "mesh-plan", "--point", "39.1,-94.58", "--fine-dx-m", "937.5", "--radius-km", "100",
        "--card", "32gb", "--mesh-exe", str(fake_mesh_exe),
    ])
    request = mesh_point.request_from_arguments(arguments)
    plan = mesh_point.plan_point(request)
    assert plan["schema"] == "gpuwm-hex.point-plan/v1"
    assert plan["parent"]["predicted_cells"] == pytest.approx(110_437.7)
    assert plan["cull"]["attained_fine_spacing_km"] == pytest.approx(0.947)
    assert 40_000 < plan["cull"]["predicted_cells"] < 46_000
    assert plan["device"]["fits"] and plan["device"]["row"] == "limited-area/170sm"
    assert plan["device"]["max_core_radius_km_on_this_card"] > 100.0
    assert plan["cull_region"]["radius_km"] == pytest.approx(135.0)
    report = "\n".join(mesh_point._report(plan))
    assert "FITS" in report and "limited-area/170sm" in report


def test_the_front_door_takes_point_or_spec_and_refuses_both_or_neither(capsys) -> None:
    assert main(["mesh-plan"]) == 2
    assert "neither --spec nor --point" in capsys.readouterr().err
    assert main(["mesh-plan", "--spec", "x.json", "--point", "1,2"]) == 2
    assert "both given" in capsys.readouterr().err


def test_generate_without_an_out_dir_refuses_before_pricing_anything(fake_mesh_exe: Path, monkeypatch) -> None:
    from hexcore import mesh_spec_gates

    monkeypatch.setattr(mesh_spec_gates, "gates_from_receipt", lambda *a, **k: {})
    parser = build_parser()
    arguments = parser.parse_args([
        "mesh-plan", "--point", "39.1,-94.58", "--generate", "--mesh-exe", str(fake_mesh_exe),
    ])
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.run_point(arguments)
    assert "--generate needs --out-dir" in str(refusal.value)


def test_a_cull_that_does_not_fit_the_card_is_refused_before_anything_is_built(
    fake_mesh_exe: Path, monkeypatch, tmp_path: Path
) -> None:
    from hexcore import mesh_spec_gates

    monkeypatch.setattr(mesh_spec_gates, "gates_from_receipt", lambda *a, **k: {})
    parser = build_parser()
    arguments = parser.parse_args([
        "mesh-plan", "--point", "39.1,-94.58", "--card", "32gb", "--vram-gib", "3",
        "--generate", "--out-dir", str(tmp_path / "out"), "--mesh-exe", str(fake_mesh_exe),
    ])
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.run_point(arguments)
    text = str(refusal.value)
    assert "refused before anything is built" in text and "reduce --radius-km" in text
    assert not (tmp_path / "out").exists()


def test_the_row_names_carry_the_spacing_the_background_the_count_and_the_point() -> None:
    assert mesh_point.parent_row_name((39.1, -94.58), 937.5, 120.0, 110_600) == "p0.9375.120.110600.n39.10w94.58"
    assert mesh_point.cull_row_name((39.1, -94.58), 937.5, 120.0, 42_000) == "q0.9375.120.42000.n39.10w94.58"
