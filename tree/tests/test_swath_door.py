"""The front door: what a ``pip install gpuwm-hex`` user actually reaches.

Engine-proven is not shipped.  These tests drive ``gpuwm-hex swath``
through ``hexcore.cli.main`` -- the same entry point the console script
binds -- rather than calling the placement functions directly, because a
capability with no door is not a feature.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import build_swath_fixture_history as fixture  # noqa: E402
from hexcore.cli import build_parser, main  # noqa: E402
from hexcore.swath import registry, sizing  # noqa: E402

CELLS = 40962
HOURS = [0.0, 3.0, 6.0, 9.0, 12.0]
SCENARIO = [
    {"kind": "low", "latitude_deg": 16.0, "longitude_deg": -52.0, "bearing_deg": 300.0,
     "speed_km_per_hour": 22.0, "radius_km": 420.0, "amplitude": 4200.0},
    {"kind": "low", "latitude_deg": 14.0, "longitude_deg": 132.0, "bearing_deg": 310.0,
     "speed_km_per_hour": 26.0, "radius_km": 380.0, "amplitude": 3400.0},
]


@pytest.fixture(scope="module")
def history(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("swath-door") / "coarse.nc"
    return fixture.build(target, cells=CELLS, hours=HOURS, scenario=SCENARIO)


def _engine_or_skip() -> Path:
    from hexcore import engine_pin as _engine_pin
    from hexcore.swath.errors import SwathRefusal

    try:
        engine = sizing.resolve_engine(None)
    except SwathRefusal as refusal:
        pytest.skip(f"rw_mpas_mesh is not staged on this box: {refusal}")
    if not _engine_measures_the_gradient_where_the_regions_are(str(engine)):
        pytest.fail(
            f"REFUSED: the staged {engine.name} predates the pinned engine's "
            f"own bridge bundle, so this box would judge nothing and read "
            f"green. This distribution pins {_engine_pin.gpuwm_requirement()} "
            f"and every {_engine_pin.wanted_version()} bundle probes the "
            f"gradient at the spec's own regions "
            f"(measured 2026-09-01: the published bundle's receipt reads "
            f"gradient_probe_coverage 'complete'), so a staged binary whose "
            f"--dry-run receipt carries no gradient_probe_coverage is a bridge "
            f"left over from an earlier release -- measured on the proving "
            f"desktop the same day as a v2.5.3-era bridge under a 2.6.1 "
            f"engine, with these ten gates skipping quietly past it. "
            f"THE BREAKAGE THIS REFUSAL PREVENTS: the transition-band gate "
            f"correctly declines to judge a spec whose steepest-gradient "
            f"number was measured on a 101 km global lattice, so every "
            f"mesh-spec gate and every swath sizing call stops being judged "
            f"while the battery still reports success. A bridge that is not "
            f"staged at all skips above, because a box with no bridge cannot "
            f"prove or disprove anything; a bridge that IS here and is older "
            f"than the pin is this refusal. Re-stage with `gpuwm "
            f"fetch-bridges`, which verifies every artifact against the "
            f"packaged pins, or point $GPUWM_HEX_RW_MPAS_MESH at the binary "
            f"from a {_engine_pin.wanted_version()} bundle.  staged: {engine}"
        )
    return engine


@functools.lru_cache(maxsize=None)
def _engine_measures_the_gradient_where_the_regions_are(engine: str) -> bool:
    """Does the staged engine probe the gradient at the spec's own regions?

    A STAGING PRECONDITION, not a softened gate. Before 2026-08-29 the
    generator sampled its steepest-gradient number on a Fibonacci lattice
    uniform over the whole sphere, whose 50,000 points sit 101 km apart, so it
    could not see a refinement transition narrower than that and reported the
    flat background it landed on. A receipt from such an engine carries no
    ``gradient_probe_coverage`` word, and
    :func:`hexcore.mesh_spec_gates.gates_from_receipt` refuses to judge a spec
    on a number that is not a measurement -- which is the correct behaviour and
    is what these tests would otherwise be asserting against. The tests that
    need a real reading skip until the engine is rebuilt from the gpuwm
    ``tools/rustwx`` workspace; they do not lower the bar to meet it.
    """

    with tempfile.TemporaryDirectory() as work:
        spec = Path(work) / "probe.mesh-spec.json"
        spec.write_text(
            json.dumps({"background_km": 240.0, "regions": []}),
            encoding="utf-8",
            newline="\n",
        )
        done = subprocess.run(
            [engine, "--spec", str(spec), "--out", str(Path(work) / "x.nc"),
             "--dry-run"],
            capture_output=True,
            text=True,
            check=False,
        )
    if done.returncode != 0:
        return False
    try:
        return "gradient_probe_coverage" in json.loads(done.stdout)
    except json.JSONDecodeError:
        return False


# ---------------------------------------------------------------------------
# the door exists and is reachable
# ---------------------------------------------------------------------------
def test_swath_is_a_subcommand_of_the_console_script() -> None:
    parser = build_parser()
    actions = [
        action for action in parser._subparsers._group_actions  # noqa: SLF001
    ]
    assert actions
    assert "swath" in actions[0].choices


def test_the_swath_door_has_three_legs() -> None:
    parser = build_parser()
    swath = parser._subparsers._group_actions[0].choices["swath"]  # noqa: SLF001
    legs = swath._subparsers._group_actions[0].choices  # noqa: SLF001
    assert set(legs) == {"plan", "metrics", "explain"}


def test_metrics_prints_the_armed_rows_and_the_manifest(capsys) -> None:
    assert main(["swath", "metrics"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == registry.METRICS_SCHEMA
    assert "surface_pressure" in document["publication_manifest"]
    assert {row["id"] for row in document["metrics"]} >= {
        "tropical_cyclone_centre", "deep_convection_area"
    }
    assert document["vocabularies"]["detector"] == list(registry.DETECTOR_KINDS)


def test_publication_manifest_prints_only_the_variables(capsys) -> None:
    assert main(["swath", "metrics", "--publication-manifest"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert isinstance(manifest, list)
    assert "refl10cm" in manifest


def test_plan_without_a_history_refuses_by_name(capsys) -> None:
    assert main(["swath", "plan"]) == 2
    assert "--history was not given" in capsys.readouterr().err


def test_explain_without_a_plan_refuses_by_name(capsys) -> None:
    assert main(["swath", "explain"]) == 2
    assert "--plan was not given" in capsys.readouterr().err


def test_a_missing_history_refuses_by_name(capsys, tmp_path) -> None:
    assert main(["swath", "plan", "--history", str(tmp_path / "absent.nc")]) == 2
    assert "does not exist" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# a plan on disk
# ---------------------------------------------------------------------------
def test_plan_writes_documents_and_one_spec_pair_per_swath(
    history, tmp_path, capsys
) -> None:
    _engine_or_skip()
    out = tmp_path / "cycle-000"
    assert main([
        "swath", "plan", "--history", str(history), "--out", str(out),
        "--card", "rtx-5070-ti",
    ]) == 0
    capsys.readouterr()
    plan = json.loads((out / "swath-plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == "gpuwm-hex.swath-plan.v1"
    assert (out / "threat-decision.json").exists()
    assert (out / "swath-state.json").exists()
    for row in plan["admitted"]:
        assert (out / "specs" / f"{row['slot_id']}.mesh-spec.json").exists()
        assert (out / "specs" / f"{row['slot_id']}.cull-region.json").exists()


def test_the_plan_quotes_attained_spacing_not_requested(history, tmp_path, capsys) -> None:
    """the ruling, enforced at the door.

    A swath spec asks for 4 km and does not get 4 km; the receipt must carry
    what it DOES get, measured through the generator, with the basis named.

    TWO THINGS MOVE THAT NUMBER NOW, IN OPPOSITE DIRECTIONS, and a receipt
    that quoted only the request could not tell them apart:

    * the ramp's own width means the resolution field does not reach the
      request at the swath's deepest point, which makes the attainment
      COARSER -- the original finding this test was written for;
    * the graded ladder refines by midpoint insertion, which halves a spacing
      exactly, so a refined core can only land on ``background / 2^k``.
      ``rw-mpas`` ``mesh::ladder_snap`` moves the request onto the nearest such
      rung, always FINER, and records the move.

    So the number the attainment is compared against is the DELIVERED spacing
    the generator will build, not the bare request, and the row now carries
    all three -- requested, delivered, attained -- plus the generator's own
    snap record. The ruling is unchanged: the receipt quotes what it gets.
    """

    _engine_or_skip()
    out = tmp_path / "cycle-000"
    assert main([
        "swath", "plan", "--history", str(history), "--out", str(out),
    ]) == 0
    capsys.readouterr()
    plan = json.loads((out / "swath-plan.json").read_text(encoding="utf-8"))
    for row in plan["admitted"]:
        size = row["sizing"]
        requested = row["mesh_spec"]["regions"][0]["spacing_km"]
        assert size["attained_basis"] == "inscribed_cap_probe"
        assert size["requested_spacing_km"] == pytest.approx(requested)
        delivered = size["delivered_spacing_km"]
        # Never coarser than asked for: the snap only ever goes finer, and
        # never by more than a factor of two.
        assert requested / 2.0 - 1e-9 <= delivered <= requested + 1e-9
        # The attainment is judged against what will be BUILT.  It is the
        # spacing the field reaches at the inscribed cap's deepest point and
        # is coarser everywhere nearer the boundary, so it is never finer than
        # the delivered request.
        assert size["attained_spacing_km"] >= delivered
        # And the move is on the record rather than inferred from the gap.
        if delivered != pytest.approx(requested):
            snap = size["ladder_snap"]
            assert snap is not None and snap["moved"] is True
            assert snap["regions"][0]["delivered_spacing_km"] == pytest.approx(delivered)
        assert size["parent_basis"] == "generator_dry_run"
        assert size["swath_basis"] == "area_integral_at_attained_spacing"
        assert "polygon" in size["polygon_attainment"]


def test_no_size_stamps_the_weaker_basis_rather_than_hiding_it(
    history, tmp_path, capsys
) -> None:
    out = tmp_path / "cycle-000"
    assert main([
        "swath", "plan", "--history", str(history), "--out", str(out), "--no-size",
    ]) == 0
    capsys.readouterr()
    plan = json.loads((out / "swath-plan.json").read_text(encoding="utf-8"))
    for row in plan["admitted"]:
        assert row["sizing"]["basis"] == "area_integral"
        assert "not_sized_because" in row["sizing"]


def test_a_second_cycle_continues_the_first_ones_slots(
    history, tmp_path, capsys
) -> None:
    first = tmp_path / "cycle-000"
    second = tmp_path / "cycle-001"
    assert main([
        "swath", "plan", "--history", str(history), "--out", str(first), "--no-size",
    ]) == 0
    assert main([
        "swath", "plan", "--history", str(history), "--out", str(second),
        "--state", str(first / "swath-state.json"), "--no-size",
    ]) == 0
    capsys.readouterr()
    plan = json.loads((second / "swath-plan.json").read_text(encoding="utf-8"))
    assert plan["cycle_index"] == 1
    assert plan["churn"]["evictions"] == 0
    assert plan["churn"]["continued"] == len(plan["admitted"])


def test_explain_names_every_term_and_every_refusal(history, tmp_path, capsys) -> None:
    out = tmp_path / "cycle-000"
    assert main([
        "swath", "plan", "--history", str(history), "--out", str(out), "--no-size",
    ]) == 0
    capsys.readouterr()
    assert main(["swath", "explain", "--plan", str(out / "swath-plan.json")]) == 0
    text = capsys.readouterr().out
    assert "ADMITTED" in text
    assert "DECLINED" in text
    assert "hysteresis:" in text
    assert "ATTAINED" in text
    for term in registry.load_policy().rank_terms:
        assert term.id in text


def test_the_mesh_engine_is_a_row_the_doctor_reports() -> None:
    """Adding the sizing engine must be a table row, not a second ladder."""

    from hexcore import engines

    assert engines.MESH in engines.ENGINES
    assert engines.MESH.name == "rw_mpas_mesh"
    assert engines.MESH.what_breaks
    assert "GPUWM_HEX_RW_MPAS_MESH" in engines.MESH.env_names
