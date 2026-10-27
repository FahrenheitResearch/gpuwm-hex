"""Regressions for a global VR request being priced and bound as global."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore import card_fit, device_admission, mesh_plan_door, mesh_point, mesh_rows
from hexcore.cli import build_parser, main
from hexcore.errors import MpasPortError


def test_pro6000_aliases_use_the_actual_part():
    for key in ("96gb", "96gib", "rtx-pro-6000", "96gib-188sm"):
        card = mesh_point.resolve_card(key)
        assert card["admission_card"] == "96gib-188sm"
        assert card["nameplate_mib"] == 97887
        assert card["generator_card"] is None  # the old affine generator table has no measured row


def test_prediction_uses_global_admission_and_keeps_the_model_provenance():
    fit = card_fit.predict(250000, card=mesh_point.resolve_card("96gb"), levels=55,
                           dt_seconds=90, hours=12, observations=[])
    model = device_admission.model_for_card(device_admission.KNOWN_CARDS["96gib-188sm"], "global")
    assert fit["required_free_mib"] == round(model.required_bytes(250000) / 1048576, 3)
    assert fit["predicted_peak_mib"] == round(model.predict_bytes(250000) / 1048576, 3)
    assert fit["model"]["configuration"] == "global"
    assert fit["model"]["provenance"]
    assert "EXTRAPOLATED" in fit["runtime"]["basis"]
    twice = card_fit.predict(250000, card=mesh_point.resolve_card("96gb"), levels=55,
                            dt_seconds=45, hours=12)
    assert twice["runtime"]["seconds"] == pytest.approx(2 * fit["runtime"]["seconds"], abs=.001)


def test_runtime_never_borrows_a_different_card_measurement():
    fit = card_fit.predict(100000, card=mesh_point.resolve_card("10gb"), levels=55,
                           dt_seconds=90, hours=12)
    assert fit["runtime"]["seconds"] is None


@pytest.mark.parametrize("key,value", [("cells", float("nan")), ("hours", 0), ("dt_seconds", -1),
                                       ("budget_mib", float("inf"))])
def test_invalid_cost_inputs_fail_before_compute(key, value):
    kwargs = dict(cells=100000, card=mesh_point.resolve_card("96gb"), levels=55,
                  dt_seconds=90, hours=12)
    kwargs[key] = value
    with pytest.raises(MpasPortError, match="finite"):
        card_fit.predict(**kwargs)


def test_process_memory_missing_is_not_reported_as_zero():
    fit = card_fit.predict(100000, card=mesh_point.resolve_card("96gb"), levels=55,
                           dt_seconds=90, hours=12)
    result = card_fit.comparison(fit, {"wall_seconds": 10, "peak_process_vram_mib": None})
    assert result["memory_error_percent"] is None
    assert result["measured_peak_process_vram_mib"] is None


def args(tmp_path, *extra):
    return build_parser().parse_args(["mesh-plan", "--global-mesh", "--point", "40,-98",
        "--background-km", "60", "--fine-dx-m", "15000", "--radius-km", "1800",
        "--card", "96gb", "--out-dir", str(tmp_path), *extra])


def test_global_plan_prices_parent_not_regional_cull(tmp_path, monkeypatch):
    from hexcore.swath import sizing
    monkeypatch.setattr(mesh_plan_door, "_load", lambda p: {})
    monkeypatch.setattr("hexcore.engines.resolve", lambda *a: Path("/engine"))
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 250000})
    request = args(tmp_path, "--out", str(tmp_path / "plan.json"))
    assert mesh_plan_door.run_global_mesh(request) == 0
    receipt = json.loads((tmp_path / "plan.json").read_text())
    assert receipt["prediction_before_generation"]["cells"] == 250000
    assert receipt["prediction_before_generation"]["model"]["configuration"] == "global"
    assert not (tmp_path / "mesh.grid.nc").exists()


def test_nonfitting_global_mesh_refused_before_generation(tmp_path, monkeypatch):
    from hexcore.swath import sizing
    monkeypatch.setattr("hexcore.engines.resolve", lambda *a: Path("/engine"))
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 2621442})
    monkeypatch.setattr(card_fit, "run_stage", lambda *a, **k: pytest.fail("must not start"))
    with pytest.raises(MpasPortError, match="whole global mesh requires"):
        mesh_plan_door.run_global_mesh(args(tmp_path, "--generate"))
    assert not (tmp_path / "global-vr-receipt.json").exists()


def test_run_without_global_mode_is_refused():
    with pytest.raises(MpasPortError, match="requires --global-mesh"):
        mesh_plan_door.run_mesh_plan(build_parser().parse_args(["mesh-plan", "--run"]))


def test_stage_records_failure_and_preserves_log(tmp_path, monkeypatch):
    monkeypatch.setattr(card_fit, "_gpu_memory", lambda pids: None)
    result = card_fit.run_stage([sys.executable, "-c", "print('specific failure'); raise SystemExit(7)"],
                                name="failure", directory=tmp_path)
    assert result["returncode"] == 7
    assert "specific failure" in Path(result["log"]).read_text()
    assert result["peak_process_vram_mib"] is None
    assert result["wall_seconds"] > 0
    assert Path(result["samples"]).is_file()


@pytest.mark.parametrize("route", ["mpas", "arwen"])
def test_whole_binding_keeps_generated_row_and_source_arguments(tmp_path, monkeypatch, route):
    from types import SimpleNamespace
    from hexcore import init_sources
    from hexcore.swath import sizing
    # MPAS physics is the hex default (0.4.6); the ArWen route is named.
    named = [] if route == "mpas" else ["--forecast-arg=--physics-suite=arwen_mpas_column"]
    request = args(tmp_path, "--generate", "--run", "--source", "gfs", "--cycle", "2023-03-31T12", *named)
    binary = tmp_path / "engine"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("hexcore.engines.resolve", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_writer", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_vtable", lambda *a: binary)
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 216000})
    monkeypatch.setattr(mesh_point, "resolve_geog", lambda *a: tmp_path)
    monkeypatch.setattr(mesh_point, "static_coordinate_errors", lambda *a: [])
    monkeypatch.setattr(mesh_point, "admit_pair", lambda *a, **k:
                        {"n_cells": 215000, "timestep_choice": {"dt_seconds": 90}, "regional": False})
    row = SimpleNamespace(name="generated-row", n_cells=215000, as_dict=lambda: {"name": "generated-row"})
    def describe(**kwargs):
        assert kwargs["core_radius_km"] == 1800
        return row
    monkeypatch.setattr(mesh_rows, "describe_resolution_spec", describe)
    monkeypatch.setattr(mesh_rows, "write_rows", lambda path, rows: path.write_text("runtime rows"))
    calls = []

    def run(argv, *, name, directory, env):
        calls.append((name, argv, dict(env)))
        if name == "mesh-template":
            Path(argv[argv.index("--out") + 1]).write_bytes(b"template grid")
        if name == "init-source":
            out = tmp_path / "init"
            out.mkdir()
            (out / "init-source-receipt.json").write_text(json.dumps({"source": "gfs", "cycle": "2023-03-31_12:00:00"}))
        return {"returncode": 0, "wall_seconds": 10, "peak_process_vram_mib": 16000}

    monkeypatch.setattr(card_fit, "run_stage", run)
    monkeypatch.setattr(card_fit, "start_stage", fake_background(calls))
    monkeypatch.setattr("hexcore.moving_mesh.rotate_grid", fake_rotation)
    assert mesh_plan_door.run_global_mesh(request) == 0
    assert [call[0] for call in calls] == ["init-source-decode", "mesh-template", "static", "init-source-decode.finish",
                                           "init-source", "forecast", "products"]
    forecast = calls[5]
    assert forecast[1][forecast[1].index("--mesh") + 1] == "generated-row"
    if route == "arwen":
        assert forecast[1][forecast[1].index("--dt") + 1] == "90"
        assert "--physics-suite=arwen_mpas_column" in forecast[1]
    else:  # the forecast door's own step for the suite, as on the global door
        assert "--dt" not in forecast[1] and "--physics-suite=convection_permitting" in forecast[1]
    assert str(tmp_path / "mesh-rows.json") in forecast[2]["GPUWM_HEX_MESH_ROWS"]
    receipt = json.loads((tmp_path / "global-vr-receipt.json").read_text())
    assert receipt["physics"]["suite"] == ("convection_permitting" if route == "mpas" else "arwen_mpas_column")
    assert receipt["status"] == "complete"
    assert receipt["prediction_on_generated_mesh"]["cells"] == 215000
    assert receipt["source_binding"]["source"] == "gfs"
    assert receipt["predicted_vs_measured"]["measured_peak_process_vram_mib"] == 16000


def fake_background(calls, returncode=0):
    """card_fit.start_stage stand-in: records the start and the join in CALLS."""
    def start(argv, *, name, directory, env):
        calls.append((name, [str(a) for a in argv], dict(env)))

        class Handle:
            started_utc = "2026-10-27T00:00:00+00:00"

            def __init__(self):
                self.argv = [str(a) for a in argv]
                self.cancelled = False

            def finish(self):
                calls.append((f"{name}.finish", self.argv, {}))
                return {"stage": name, "returncode": returncode, "wall_seconds": 13.4, "log": "decode.log",
                        "background": True}

            def cancel(self):
                calls.append((f"{name}.cancel", self.argv, {}))
        return Handle()
    return start


def fake_rotation(source, output, *, start, end, file_id=None):
    Path(output).write_bytes(Path(source).read_bytes() + repr((end, file_id)).encode())
    return {"sha256": "rotated", "wall_seconds": 0.5, "start": list(start), "end": list(end)}


def test_zoom_globe_rotates_the_cap_family_template_and_decodes_beside_the_mesh(tmp_path, monkeypatch):
    """U-H6 (production run_fde6921eaf18b793): a zoom globe generated a fresh mesh for its point (16.6 s) and
    fetched and decoded its source only after the static. The door now asks for the point's cap family about
    the template centre (a prep-store key with no point in it), rotates it to the point, and starts the source
    fetch and decode before the mesh; the init is built --from-decoded."""
    from types import SimpleNamespace
    from hexcore import init_sources
    from hexcore.moving_members import TEMPLATE_CENTER
    from hexcore.swath import sizing
    request = args(tmp_path, "--generate", "--run", "--source", "ifs", "--cycle", "2026-10-26T00")
    binary = tmp_path / "engine"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("hexcore.engines.resolve", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_writer", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_vtable", lambda *a: binary)
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 216000})
    monkeypatch.setattr(mesh_point, "resolve_geog", lambda *a: tmp_path)
    monkeypatch.setattr(mesh_point, "static_coordinate_errors", lambda *a: [])
    monkeypatch.setattr(mesh_point, "admit_pair", lambda *a, **k:
                        {"n_cells": 215000, "timestep_choice": {"dt_seconds": 90}, "regional": False})
    row = SimpleNamespace(name="generated-row", n_cells=215000, as_dict=lambda: {"name": "generated-row"})
    monkeypatch.setattr(mesh_rows, "describe_resolution_spec", lambda **k: row)
    monkeypatch.setattr(mesh_rows, "write_rows", lambda path, rows: path.write_text("runtime rows"))
    calls, rotations = [], []

    def run(argv, *, name, directory, env):
        calls.append((name, [str(a) for a in argv], dict(env)))
        if name == "mesh-template":
            Path(argv[argv.index("--out") + 1]).write_bytes(b"template grid")
        if name == "init-source":
            (tmp_path / "init").mkdir()
            (tmp_path / "init" / "init-source-receipt.json").write_text(json.dumps({"source": "ifs"}))
        return {"returncode": 0, "wall_seconds": 10, "peak_process_vram_mib": 9000}

    def rotate(source, output, *, start, end, file_id=None):
        rotations.append((start, end, file_id))
        return fake_rotation(source, output, start=start, end=end, file_id=file_id)

    monkeypatch.setattr(card_fit, "run_stage", run)
    monkeypatch.setattr(card_fit, "start_stage", fake_background(calls))
    monkeypatch.setattr("hexcore.moving_mesh.rotate_grid", rotate)
    assert mesh_plan_door.run_global_mesh(request) == 0
    names = [call[0] for call in calls]
    # The source fetch and decode start first and are joined only after the static.
    assert names[:3] == ["init-source-decode", "mesh-template", "static"]
    assert names.index("init-source-decode.finish") > names.index("static")
    decode = calls[0][1]
    assert decode[decode.index("--stop-after") + 1] == "decode" and "--grid" not in decode
    init = calls[names.index("init-source")][1]
    assert "--from-decoded" in init and init[init.index("--static") + 1] == str(tmp_path / "mesh.static.nc")
    assert "rw_mpas_mesh" not in names and "mesh" not in names  # no fresh generation for the point
    template = json.loads(Path(calls[1][1][calls[1][1].index("--spec") + 1]).read_text())
    assert template["name"] == mesh_plan_door.ROTATION_TEMPLATE_NAME
    assert {tuple(r["shape"]["center_deg"]) for r in template["regions"]} == {TEMPLATE_CENTER}
    spec = json.loads((tmp_path / "resolution-spec.json").read_text())
    assert [r["radius_km"] for r in (x["shape"] for x in template["regions"])] == \
        [r["radius_km"] for r in (x["shape"] for x in spec["regions"])]
    assert rotations[0][0] == TEMPLATE_CENTER and rotations[0][1] == (40.0, -98.0) and rotations[0][2]
    receipt = json.loads((tmp_path / "global-vr-receipt.json").read_text())
    mesh = next(stage for stage in receipt["stages"] if stage.get("name") == "mesh")
    assert mesh["source"] == "rotated-template" and mesh["end_deg"] == [40.0, -98.0]
    assert receipt["status"] == "complete"
    rotation = json.loads((tmp_path / "mesh-generator.json").read_text())
    assert rotation["schema"] == "gpuwm-hex.zoom-globe-rotation/v1"
    assert not (tmp_path / "template" / "rotation-template.grid.nc").exists()


def test_rotation_template_key_is_the_same_for_every_point():
    """Two zoom globes of one family at different points ask the prep store for the same template."""
    a = mesh_point.ladder_spec((40.0, -97.0), fine_dx_m=15000, radius_km=2000, background_km=120)
    b = mesh_point.ladder_spec((-33.9, 151.2), fine_dx_m=15000, radius_km=2000, background_km=120)
    assert a != b
    assert mesh_plan_door.rotation_template_spec(a) == mesh_plan_door.rotation_template_spec(b)
    assert mesh_plan_door.concentric_cap_center(a) == (40.0, -97.0)
    two = dict(a, regions=[*a["regions"], {"shape": {"kind": "cap", "center_deg": [10.0, 10.0], "radius_km": 500.0},
                                            "spacing_km": 30.0, "transition_km": 60.0}])
    assert mesh_plan_door.concentric_cap_center(two) is None          # several centres: generated
    assert mesh_plan_door.concentric_cap_center({"background_km": 60, "regions": []}) is None  # uniform: generated


def test_template_only_builds_the_family_template_for_a_bake(tmp_path, monkeypatch):
    """The bake stages a zoom family's template (no static, no run) so boxes rotate it instead of generating."""
    from hexcore.moving_members import TEMPLATE_CENTER
    from hexcore.swath import sizing
    binary = tmp_path / "engine"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("hexcore.engines.resolve", lambda *a: binary)
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 216000})
    monkeypatch.setattr(mesh_point, "resolve_geog", lambda *a: tmp_path)
    calls = []

    def run(argv, *, name, directory, env):
        calls.append((name, [str(a) for a in argv]))
        Path(argv[argv.index("--out") + 1]).write_bytes(b"template grid")
        return {"returncode": 0, "wall_seconds": 9}

    monkeypatch.setattr(card_fit, "run_stage", run)
    assert mesh_plan_door.run_global_mesh(args(tmp_path, "--generate", "--template-only")) == 0
    assert [c[0] for c in calls] == ["mesh-template"]
    receipt = json.loads((tmp_path / "global-vr-receipt.json").read_text())
    assert receipt["status"] == "template"
    assert {tuple(r["shape"]["center_deg"]) for r in receipt["template"]["spec"]["regions"]} == {TEMPLATE_CENTER}
    with pytest.raises(MpasPortError, match="no --run"):
        mesh_plan_door.run_global_mesh(args(tmp_path / "b", "--generate", "--template-only", "--run",
                                            "--source", "ifs", "--cycle", "2026-10-26T00"))


def test_a_failed_door_stops_the_decode_beside_it(tmp_path, monkeypatch):
    from hexcore import init_sources
    from hexcore.swath import sizing
    request = args(tmp_path, "--generate", "--run", "--source", "ifs", "--cycle", "2026-10-26T00")
    binary = tmp_path / "engine"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("hexcore.engines.resolve", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_writer", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_vtable", lambda *a: binary)
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 216000})
    monkeypatch.setattr(mesh_point, "resolve_geog", lambda *a: tmp_path)
    calls = []
    monkeypatch.setattr(card_fit, "run_stage", lambda argv, *, name, directory, env:
                        calls.append((name, argv, env)) or {"returncode": 3 if name == "mesh-template" else 0,
                                                            "log": "x"})
    monkeypatch.setattr(card_fit, "start_stage", fake_background(calls))
    with pytest.raises(MpasPortError, match="mesh-template exited 3"):
        mesh_plan_door.run_global_mesh(request)
    assert [c[0] for c in calls] == ["init-source-decode", "mesh-template", "init-source-decode.cancel"]


def test_background_stage_reports_its_own_wall_and_cpu(tmp_path):
    handle = card_fit.start_stage([sys.executable, "-c", "import time; sum(range(10**6)); time.sleep(0.3)"],
                                  name="beside", directory=tmp_path)
    import time
    time.sleep(1.0)  # the door is busy with the foreground stage
    row = handle.finish()
    assert row["returncode"] == 0 and row["background"]
    assert 0.3 <= row["wall_seconds"] < 0.95  # its own exit, not the join
    assert row["cpu_seconds"] > 0
    failing = card_fit.start_stage([sys.executable, "-c", "raise SystemExit(5)"], name="bad", directory=tmp_path)
    assert failing.finish()["returncode"] == 5
    slow = card_fit.start_stage([sys.executable, "-c", "import time; time.sleep(60)"], name="slow", directory=tmp_path)
    slow.cancel()
    assert slow.finish()["returncode"] != 0


def test_init_source_from_decoded_refuses_another_requests_decode(tmp_path):
    from hexcore import init_sources
    met = tmp_path / "MET:2026-10-26_00"
    met.write_bytes(b"met")
    base = {"schema": init_sources.RECEIPT_SCHEMA, "source": "ifs", "member": None,
            "source_cycle": "2026-10-26_00:00:00", "lead_hours": 0, "valid_time": "2026-10-26_00:00:00",
            "table": {"sha256": "t"}}
    prior = dict(base, decode={"met": str(met), "met_sha256": init_sources._sha256_file(met)},
                 stage_seconds={"acquire": 9.0, "decode": 4.4})
    (tmp_path / "init-source-receipt.json").write_text(json.dumps(prior))
    assert init_sources._decoded_receipt(tmp_path, dict(base))["decode"]["met"] == str(met)
    with pytest.raises(MpasPortError, match="lead_hours"):
        init_sources._decoded_receipt(tmp_path, dict(base, lead_hours=6))
    met.write_bytes(b"changed")
    with pytest.raises(MpasPortError, match="SHA-256"):
        init_sources._decoded_receipt(tmp_path, dict(base))
    with pytest.raises(MpasPortError, match="does not exist"):
        init_sources._decoded_receipt(tmp_path / "elsewhere", dict(base))


def test_observed_whole_run_floor_prevents_repeating_a_false_fit():
    anchor = {"status": "complete", "card": "96gib-188sm", "levels": 55,
              "cells": 216189, "peak_process_vram_mib": 25180, "receipt_sha256": "evidence"}
    fit = card_fit.predict(216189, card=mesh_point.resolve_card("96gb"), levels=55,
                           dt_seconds=90, hours=18, budget_mib=24000, observations=[anchor])
    margin = 1 + card_fit.WHOLE_RUN_MARGIN
    assert fit["base_predicted_peak_mib"] < 16000
    assert fit["predicted_peak_mib"] == pytest.approx(25180 * margin, abs=.001)
    assert fit["required_free_mib"] > 25180
    assert not fit["fits"]
    assert fit["whole_run_floor"]["observation"]["receipt_sha256"] == "evidence"
    larger = card_fit.predict(432378, card=mesh_point.resolve_card("96gb"), levels=55,
                              dt_seconds=90, hours=18, observations=[anchor])
    assert larger["predicted_peak_mib"] >= 50360 * margin - .01
    # Below its one run the floor falls with device_admission's own slope; it is
    # never held at the run's peak (128,942 cells priced at the 216k run's 25,272).
    smaller = card_fit.predict(108094, card=mesh_point.resolve_card("96gb"), levels=55,
                               dt_seconds=90, hours=18, observations=[anchor])
    assert smaller["predicted_peak_mib"] < 25180
    other = card_fit.predict(216189, card=mesh_point.resolve_card("32gb"), levels=55,
                             dt_seconds=90, hours=18, observations=[anchor])
    assert other["whole_run_floor"] is None
    unfinished = dict(anchor, status="running")
    assert card_fit.predict(216189, card=mesh_point.resolve_card("96gb"), levels=55,
                            dt_seconds=90, hours=18, observations=[unfinished])["whole_run_floor"] is None


def test_shipped_whole_run_observation_rejects_the_measured_false_fit():
    fit = card_fit.predict(216189, card=mesh_point.resolve_card("96gb"), levels=55,
                           dt_seconds=90, hours=18, budget_mib=21000)
    assert fit["base_predicted_peak_mib"] < 19816
    assert fit["predicted_peak_mib"] >= 19816
    assert fit["required_free_mib"] > fit["predicted_peak_mib"]
    assert not fit["fits"]
    assert fit["whole_run_floor"]["observation"]["cells"] == 216187
    assert len(fit["whole_run_floor"]["observation"]["receipt_sha256"]) == 64


#: Every whole run of today's build on disk (card, cells, sampled forecast peak MiB).
MEASURED_WHOLE_RUNS = [
    ("96gb", 52618, 7472.0),     # 120 -> 30 km, 24 h, RTX PRO 6000
    ("96gb", 216187, 19816.0),   # 60 -> 15 km, 15 h, RTX PRO 6000
    ("96gb", 581895, 44450.0),   # 80 -> 5 km, 1,800 km core, 3 h, RTX PRO 6000
    ("32gb", 128942, 13036.0),   # 80 -> 5 km, 400 km core, 3 h, RTX 5090
    ("96gb", 128942, 13036.0),   # the same run as the site prices it (96 GB card); was 25,272
]


@pytest.mark.parametrize("card,cells,peak", MEASURED_WHOLE_RUNS)
def test_shipped_curve_lands_10_to_15_percent_over_every_measured_run(card, cells, peak):
    fit = card_fit.predict(cells, card=mesh_point.resolve_card(card), levels=55, dt_seconds=30, hours=3)
    over = fit["predicted_peak_mib"] / peak - 1
    assert 0.10 - 1e-9 <= over <= 0.15, (card, cells, fit["predicted_peak_mib"], over)


def test_shipped_curve_rises_with_cells():
    card = mesh_point.resolve_card("96gb")
    peaks = [card_fit.predict(cells, card=card, levels=55, dt_seconds=30, hours=3)["predicted_peak_mib"]
             for cells in range(20000, 1400001, 20000)]
    assert peaks == sorted(peaks)
