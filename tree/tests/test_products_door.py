"""The product door over stand-in engines: views, layout, standard-set
expansion, the remap it asks for, and the points stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Event

import pytest

from hexcore import products_door, render_door
from hexcore.errors import MpasPortError


class TestRemapResolution:
    def test_auto_is_product_for_one_run_and_nearest_for_a_composite(self):
        assert render_door.resolve_remap("auto", composite=False) == "product"
        assert render_door.resolve_remap("auto", composite=True) == "nearest"
        assert render_door.resolve_remap("nearest", composite=False) == "nearest"

    def test_product_with_a_composite_is_refused_by_name(self):
        with pytest.raises(MpasPortError, match="no single mesh"):
            render_door.resolve_remap("product", composite=True)

    def test_unknown_remap_is_refused(self):
        with pytest.raises(MpasPortError):
            render_door.resolve_remap("bilinear", composite=False)

    @pytest.mark.parametrize("remap,expect", [("product", True), ("nearest", False)])
    def test_the_converter_is_asked_for_the_remap(self, monkeypatch, tmp_path, remap, expect):
        seen = {}

        class Result:
            returncode = 0
            stdout = "REMAP\tproduct\tabc\nBUDGET\tt\tRAINC+RAINNC\trel=1e-5\n"
            stderr = ""

        def fake_run(command, *, log_prefix, env=None):
            seen["command"] = [str(c) for c in command]
            Path(command[command.index("--json") + 1]).write_text("{}", encoding="utf-8")
            return Result()

        monkeypatch.setattr(render_door, "_run", fake_run)
        render_door.convert_frames(
            Path("rw_mpas_convert"), histories=[tmp_path / "h.nc"], mesh=tmp_path / "m.nc",
            scratch=tmp_path / "s", window="global-mesh", field_set="full",
            simulation_start=None, nc_format="cdf2", remap=remap)
        command = seen["command"]
        assert ("--remap" in command) is expect
        if expect:
            assert command[command.index("--remap") + 1] == "product"


class TestHelpers:
    def test_case_token_comes_from_the_data(self):
        token = products_door.case_token(None, Path("/m/x1.655362.ordered.grid.nc"), "2026-10-03_12:00:00")
        assert token == "x1.655362-2026100312"
        assert products_door.case_token("my run", Path("m.nc"), "2026-10-03_12:00:00") == "my-run"
        # a grid stored under a role name takes the forecast's mesh name, else its folder's
        nameless = Path("/meshes/global30/ordered.grid.nc")
        assert products_door.case_token(None, nameless, "2026-10-03_12:00:00") == "global30-2026100312"
        assert products_door.case_token(None, nameless, "2026-10-03_12:00:00",
                                        "x1.655362-ordered") == "x1.655362-ordered-2026100312"

    def test_window_centre_reads_both_window_kinds(self):
        lam = {"convert": {"window_spec": {"kind": "lambert", "centre_lat": 37.5, "centre_lon": -97.5,
                                           "dx_metres": 3000.0, "nx": 300, "ny": 200}}}
        assert products_door.window_centre(lam) == (37.5, -97.5, 300.0)
        ll = {"convert": {"window_spec": json.dumps({"kind": "latlon", "south": -10.0, "north": 10.0,
                                                    "west": 20.0, "east": 40.0, "spacing_degrees": 0.1})}}
        lat, lon, half = products_door.window_centre(ll)
        assert (lat, lon) == (0.0, 30.0) and half == pytest.approx(10 * 111.195)
        assert products_door.window_centre({}) is None

    def test_histories_are_discovered_in_time_order_or_refused(self, tmp_path):
        with pytest.raises(MpasPortError, match="no cuda-history"):
            products_door.discover_histories(tmp_path)
        for stamp in ("2026-10-03_18.00.00", "2026-10-03_12.00.00"):
            (tmp_path / f"cuda-history.{stamp}.nc").write_bytes(b"h")
        found = products_door.discover_histories(tmp_path)
        assert [p.name for p in found] == ["cuda-history.2026-10-03_12.00.00.nc",
                                           "cuda-history.2026-10-03_18.00.00.nc"]
        assert products_door.simulation_start(None, None, found) == "2026-10-03_12:00:00"

    def test_the_forecast_door_hands_its_own_files_to_the_product_door(self, tmp_path):
        from hexcore import forecast_door

        request = argparse.Namespace(out=tmp_path / "fc", grid=tmp_path / "g.nc",
                                     init=tmp_path / "i.nc", start_time="2026-10-03_12:00:00")
        command = forecast_door.products_command(request, [tmp_path / "fc" / "h.nc"])
        assert command[:2] == ["gpuwm-hex", "products"]
        assert command[command.index("--forecast") + 1] == str(tmp_path / "fc")
        assert command[command.index("--init") + 1] == str(tmp_path / "i.nc")
        assert command[command.index("--out") + 1] == str(tmp_path / "fc" / "products")
        parser = argparse.ArgumentParser()
        products_door.add_products_arguments(parser)
        parsed = parser.parse_args(command[2:])
        assert parsed.remap == "auto" and not parsed.no_points

    def test_the_forecast_door_offers_the_opt_out(self):
        from hexcore import forecast_door

        parser = argparse.ArgumentParser()
        forecast_door.add_forecast_arguments(parser)
        assert "--no-products" in parser.format_help()


def _stand_in_engines(monkeypatch, tmp_path, *, uniform: bool, seen: list):
    """Converter, catalog and renderer stand-ins through render_door._run."""

    basemap = tmp_path / "basemap"
    (basemap / "natural_earth_110m").mkdir(parents=True, exist_ok=True)
    (basemap / "natural_earth_110m" / "ne_110m_coastline.shp").write_bytes(b"s")
    monkeypatch.setenv("RUSTWX_BASEMAP_DIR", str(basemap))
    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda explicit=None: Path("rw_mpas_convert"))
    monkeypatch.setattr(render_door, "resolve_renderer_exe", lambda explicit=None: Path("rw_wrfbatch"))
    monkeypatch.setattr(render_door, "sha256_file", lambda path: "0" * 64)
    # the stand-in engines are current builds: they take the frame-parallel flags
    monkeypatch.setattr(render_door, "engine_accepts", lambda exe, flag: True)

    class Result:
        def __init__(self):
            self.returncode, self.stdout, self.stderr = 0, "", ""

    def fake_run(command, *, log_prefix, env=None):
        command = [str(c) for c in command]
        seen.append((log_prefix, command))
        result = Result()
        if log_prefix == "CONVERT":
            window = command[command.index("--window") + 1]
            if window == "mesh" and uniform:
                result.returncode = 1
                result.stderr = ("rw_mpas_convert: this mesh is uniform to within the refinement "
                                 "factor -- so there is no refined region to derive a window from")
                return result
            frames = Path(command[command.index("--out-dir") + 1])
            outs = []
            histories = command[command.index("--history") + 1:command.index("--mesh")]
            for k, _ in enumerate(histories):
                wrfout = frames / f"wrfout_d01_2026-10-03_{12 + 6 * k:02d}_00_00"
                wrfout.write_bytes(b"w")
                outs.append({"output": str(wrfout), "valid_time": f"2026-10-03T{12 + 6 * k:02d}:00:00",
                             "lead_seconds": 21600 * k, "absent_wrf_fields": [],
                             "precipitation_budget": {"relative_difference": 1e-5}})
            spec = ({"kind": "lambert", "centre_lat": 37.5, "centre_lon": -97.5, "dx_metres": 3000.0,
                     "nx": 300, "ny": 300} if window == "mesh" else {"kind": "latlon"})
            Path(command[command.index("--json") + 1]).write_text(json.dumps({
                "grid_label": "hex 48-3 km", "window_spec": spec,
                "product_remap": {"sha256": "r"}, "frames": outs}), encoding="utf-8")
        elif log_prefix == "CATALOG":
            result.stdout = ("PRODUCT\t2m_temperature_10m_winds\tdirect\trenderable\tready\t\n"
                             "PRODUCT\ttotal_qpf\tdirect\trenderable\tready\t\n"
                             "PRODUCT\tprecipitation_type\tdirect\tmissing-fields\tnot stored: categorical_rain_surface\t\n"
                             "PRODUCT\tqpf_6h\twindowed\trenderable\t6 h APCP differenced (F006 minus F000)\trenderable\n"
                             "PRODUCT\tqpf_24h\twindowed\tblocked\t24-h QPF requires forecast hour >= 24\twindowed-blocked\n")
        elif log_prefix == "RENDER":
            raw = Path(command[command.index("--out-dir") + 1])
            drawn = command[command.index("--products") + 1].split(",")
            lines = []
            # Every frame of the series arrives in one start, in one store.
            for wrfout in [c for c in command if Path(c).name.startswith("wrfout_d01_")]:
                lead = int(Path(wrfout).name.split("_")[3]) - 12
                for slug in drawn:
                    if slug == "qpf_6h" and lead < 6:
                        lines.append(f"SKIPPED {slug} {wrfout}: F{lead:03d}: 6-h QPF requires forecast hour >= 6")
                        continue
                    png = raw / f"rustwx_wrf_20261003_12z_f{lead:03d}_hex48-3km_{slug}.png"
                    png.write_bytes(b"png")
                    lines.append(f"RENDERED {slug} {png}")
            result.stdout = "\n".join(lines) + "\n"
        return result

    monkeypatch.setattr(render_door, "_run", fake_run)


def _points_stand_in(monkeypatch, seen: list):
    # Stand-in history bytes do not carry real NetCDF headers.
    monkeypatch.setattr(products_door, "histories_without_surface_pressure", lambda histories: [])
    class Done:
        returncode, stderr = 0, ""
        stdout = "POINT\tx\nFINISHED\tpoints=1\n"

    def fake_subprocess_run(command, **kwargs):
        command = [str(c) for c in command]
        seen.append(("POINTS", command))
        out = Path(command[command.index("--out-dir") + 1])
        names = [item.split("@")[0] for item in command[command.index("--points") + 1].split(";")]
        slug = names[0].lower().replace(" ", "-")
        (out / slug).mkdir(parents=True, exist_ok=True)
        (out / slug / f"sounding_{slug}_20261003_1800_f006.png").write_bytes(b"s")
        (out / slug / f"meteogram_{slug}_20261003_12.png").write_bytes(b"m")
        Path(command[command.index("--json") + 1]).write_text('{"soundings": []}', encoding="utf-8")
        return Done()

    monkeypatch.setattr(products_door.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(products_door, "resolve_points_exe", lambda explicit, convert: Path("rw_hex_points"))


def _arguments(tmp_path, histories, **over):
    parser = argparse.ArgumentParser()
    products_door.add_products_arguments(parser)
    init = tmp_path / "init.nc"
    init.write_bytes(b"i")
    argv = ["--history", *[str(h) for h in histories], "--mesh", str(tmp_path / "patch.grid.nc"),
            "--init", str(init), "--out", str(tmp_path / "out")]
    for key, value in over.items():
        argv += [f"--{key.replace('_', '-')}", value]
    return parser.parse_args(argv)


def _histories(tmp_path):
    (tmp_path / "patch.grid.nc").write_bytes(b"m")
    out = []
    for stamp in ("2026-10-03_12.00.00", "2026-10-03_18.00.00"):
        path = tmp_path / f"cuda-history.{stamp}.nc"
        path.write_bytes(b"h")
        out.append(path)
    return out


@pytest.mark.parametrize("context_before_global", [False, True])
def test_a_refined_mesh_gets_global_core_context_and_points_in_the_ruled_layout(
        monkeypatch, tmp_path, context_before_global):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    if context_before_global:
        # The core can finish and launch its context while global is delayed.
        run = render_door._run
        context_started = Event()

        def delayed_global(command, *, log_prefix, env=None):
            if log_prefix == "CONVERT":
                window = command[command.index("--window") + 1]
                if window == "global-mesh":
                    assert context_started.wait(timeout=10), "context conversion never started"
                elif window.startswith("region:"):
                    result = run(command, log_prefix=log_prefix, env=env)
                    context_started.set()
                    return result
            return run(command, log_prefix=log_prefix, env=env)

        monkeypatch.setattr(render_door, "_run", delayed_global)
    histories = _histories(tmp_path)
    args = _arguments(tmp_path, histories, case="patch-case")
    assert products_door.run_products(args) == 0

    case = tmp_path / "out" / "patch-case"
    # case -> domain -> product -> valid-day
    assert (case / "global" / "2m_temperature_10m_winds" / "2026-10-03"
            / "hex48-3km_global_2m_temperature_10m_winds_f006.png").is_file()
    assert (case / "refined-core" / "total_qpf" / "2026-10-03"
            / "hex48-3km_refined-core_total_qpf_f000.png").is_file()
    assert (case / "refined-context" / "total_qpf" / "2026-10-03").is_dir()
    assert (case / "points" / "sounding-oklahoma-city" / "2026-10-03"
            / "sounding_oklahoma-city_20261003_1800_f006.png").is_file()
    assert (case / "points" / "meteogram-oklahoma-city" / "2026-10-03").is_dir()
    assert not list(case.rglob("rustwx_*"))

    converts = [c for stage, c in seen if stage == "CONVERT"]
    assert all(c[c.index("--remap") + 1] == "product" for c in converts)
    # All three views run concurrently; converter arrival order is unspecified.
    windows = [c[c.index("--window") + 1] for c in converts]
    assert len(windows) == 3
    assert windows.count("global-mesh") == windows.count("mesh") == 1
    context = [c for c in converts if c[c.index("--window") + 1].startswith("region:")]
    assert len(context) == 1
    context_window = context[0][context[0].index("--window") + 1]
    assert context_window.startswith("region:37.500,-97.500,")
    if context_before_global:
        assert windows.index(context_window) < windows.index("global-mesh")
    renders = [c for stage, c in seen if stage == "RENDER"]
    assert all("precipitation_type" not in c[c.index("--products") + 1] for c in renders)
    # one renderer start per view, every frame in it, one shared store
    assert len(renders) == 3
    assert all(sum(Path(c).name.startswith("wrfout_d01_") for c in r) == 2 for r in renders)
    assert all(Path(r[r.index("--store-root") + 1]).name == "store-series" for r in renders)
    # the interval product is filed at the frame that closes its window ...
    assert (case / "refined-core" / "qpf_6h" / "2026-10-03"
            / "hex48-3km_refined-core_qpf_6h_f006.png").is_file()
    assert not list(case.rglob("*qpf_6h_f000.png"))
    render_manifest = json.loads((case / "refined-core" / "render-manifest.json").read_text())
    assert render_manifest["series_store"] is True
    first, second = render_manifest["frames"]
    # ... the frame that cannot close it carries the renderer's own skip ...
    assert [item["product"] for item in first["skipped"]] == ["qpf_6h"]
    assert second["skipped"] == [] and "qpf_6h" in second["rendered"]
    # ... and a window the run is too short for is named, never dropped
    index = (case / "index.html").read_text(encoding="utf-8")
    assert "refined-core/qpf_6h/2026-10-03/hex48-3km_refined-core_qpf_6h_f006.png" in index
    assert "points/sounding-oklahoma-city" in index

    manifest = json.loads((case / "products-manifest.json").read_text())
    core = next(v for v in manifest["views"] if v["view"] == "refined-core")
    assert "precipitation_type" in core["products_unavailable"]
    # a window the run is too short for is not requested, and named with the reason in the manifest
    assert "qpf_24h" not in core["products_unavailable"]
    assert "needs 24 h" in manifest["windows_not_requested"]["qpf_24h"]
    assert "categorical_rain_surface" in core["products_unavailable"]["precipitation_type"]
    # the door's one history-header read also runs through subprocess.run, which this stand-in records
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    named = points[points.index("--points") + 1]
    assert "Oklahoma City@" in named and "Tokyo" not in named  # only points inside the core
    assert manifest["failed_views"] == []


def test_a_uniform_mesh_draws_global_and_the_default_regions(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=True, seen=seen)
    _points_stand_in(monkeypatch, seen)
    histories = _histories(tmp_path)
    args = _arguments(tmp_path, histories)
    assert products_door.run_products(args) == 0
    manifest = json.loads(next((tmp_path / "out").rglob("products-manifest.json")).read_text())
    views = {v["view"]: v for v in manifest["views"]}
    assert set(views) == {"global", "refined-core", "north-america", "europe"}
    assert views["refined-core"]["note"].startswith("uniform mesh")
    # the door's one history-header read also runs through subprocess.run, which this stand-in records
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    assert "Tokyo@" in points[points.index("--points") + 1]


def test_a_missing_points_engine_is_refused_by_name(monkeypatch, tmp_path):
    from hexcore import engines

    monkeypatch.delenv(products_door.POINTS_ENV, raising=False)
    monkeypatch.delenv(engines.POINTS.gpuwm_env, raising=False)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: ())
    monkeypatch.setattr(engines.shutil, "which", lambda name: None)
    with pytest.raises(MpasPortError, match="rw_hex_points not found") as refusal:
        products_door.resolve_points_exe(None, tmp_path / "rw_mpas_convert")
    # the shared ladder's refusal, with the build line and the way round it
    assert "--bin rw_hex_points" in str(refusal.value)
    assert "--no-points" in str(refusal.value)
    # the build beside the converter in use is the one a run takes
    beside = tmp_path / "rw_hex_points"
    beside.write_bytes(b"x")
    assert products_door.resolve_points_exe(None, tmp_path / "rw_mpas_convert") == beside
    # doctor's engine table names it
    assert engines.POINTS in engines.ENGINES


def test_a_series_shares_a_store_only_when_every_frame_has_its_own_lead():
    assert render_door.series_leads([{"lead_seconds": 0}, {"lead_seconds": 10800}]) == [0, 10800]
    assert render_door.series_leads([{"lead_seconds": 0}]) is None            # nothing to window
    assert render_door.series_leads([{"lead_seconds": 0}, {"lead_seconds": 0}]) is None
    assert render_door.series_leads([{"lead_seconds": 0}, {}]) is None        # an older converter
    name = "rustwx_wrf_20261003_12z_f006_hex48-3km_qpf_6h.png"
    assert render_door.engine_lead_seconds(name) == 21600
    sub = "rustwx_wrf_20261003_12z_f000_hex48-3km_total_qpf_lead_000h30m00s.png"
    assert render_door.engine_lead_seconds(sub) == 1800
    assert render_door.engine_lead_seconds("something-else.png") is None


def test_without_distinct_leads_the_render_keeps_a_store_per_frame(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(render_door, "series_leads", lambda frames: None)
    args = _arguments(tmp_path, _histories(tmp_path), case="patch-case")
    args.no_points = True
    assert products_door.run_products(args) == 0
    renders = [c for stage, c in seen if stage == "RENDER"]
    assert len(renders) == 6  # three views, two frames, a store each
    assert {Path(r[r.index("--store-root") + 1]).name for r in renders} == {"store-000", "store-001"}


def test_the_default_product_stage_draws_maps_when_the_point_tool_is_absent(monkeypatch, tmp_path, capsys):
    # gpuwm 2.9.0 builds rw_hex_points standalone, so a pip install can lack
    # it. The forecast's automatic stage must still draw the maps (they never
    # need it) and name what it skipped; before, the whole stage refused.
    from hexcore import forecast_door, products_door, render_door

    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda explicit=None: tmp_path / "rw_mpas_convert")

    def absent(explicit, convert_exe):
        raise products_door.ProductDoorError("rw_hex_points not found\n  # build line")

    monkeypatch.setattr(products_door, "resolve_points_exe", absent)
    seen = {}

    def fake_run(args):
        seen["no_points"] = args.no_points
        return 0

    monkeypatch.setattr(products_door, "run_products", fake_run)
    monkeypatch.setattr(forecast_door, "release_device_memory", lambda: None)
    out = tmp_path / "fc"
    out.mkdir()
    request = argparse.Namespace(out=out, grid=tmp_path / "g.nc", init=tmp_path / "i.nc",
                                 start_time="2026-10-03_12:00:00")
    # the in-command stage (--no-products-detached); the detached default
    # builds the same command and hands it to its own process
    inline = argparse.Namespace(products_detached=False)
    assert forecast_door.run_default_products(request, [out / "h.nc"], inline) == 0
    assert seen["no_points"] is True
    printed = capsys.readouterr().out
    assert "soundings and meteograms skipped: rw_hex_points not found" in printed
    assert "the maps are drawn" in printed

    # With the point tool present the stage draws everything, as before.
    monkeypatch.setattr(products_door, "resolve_points_exe", lambda explicit, convert_exe: tmp_path / "rw_hex_points")
    assert forecast_door.run_default_products(request, [out / "h.nc"], inline) == 0
    assert seen["no_points"] is False


def test_the_forecast_door_can_hand_the_product_stage_to_its_own_process(monkeypatch, tmp_path):
    from hexcore import forecast_door

    parser = argparse.ArgumentParser()
    forecast_door.add_forecast_arguments(parser)
    assert "--products-detached" in parser.format_help()

    started = {}

    class Child:
        pid = 4242

    def fake_popen(command, **kwargs):
        started["command"], started["kwargs"] = [str(c) for c in command], kwargs
        return Child()

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out = tmp_path / "fc"
    out.mkdir()
    request = argparse.Namespace(out=out, grid=tmp_path / "g.nc", init=tmp_path / "i.nc",
                                 start_time="2026-10-03_12:00:00")
    arguments = argparse.Namespace(products_detached=True)
    assert forecast_door.run_default_products(request, [out / "h.nc"], arguments) == 0
    command = started["command"]
    assert command[1:4] == ["-m", "hexcore.cli", "products"]
    assert command[command.index("--forecast") + 1] == str(out)
    # no card for the child, and it outlives the forecast process
    assert started["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert started["kwargs"]["start_new_session"] is True
    assert (out / "products.log").is_file()


def test_the_product_stage_is_detached_by_default_so_the_cards_are_released(monkeypatch, tmp_path):
    """A bare forecast must not hold its card grant while CPU-only products draw
    (rented box 2026-10-13: four cards held ~66 GB each for 30 min after the forecast)."""
    from hexcore import forecast_door

    parser = argparse.ArgumentParser()
    forecast_door.add_forecast_arguments(parser)
    assert parser.parse_args([]).products_detached is True
    assert parser.parse_args(["--no-products-detached"]).products_detached is False

    started = {}

    class Child:
        pid = 4343

    def fake_popen(command, **kwargs):
        started["command"] = [str(c) for c in command]
        return Child()

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out = tmp_path / "fc"
    out.mkdir()
    request = argparse.Namespace(out=out, grid=tmp_path / "g.nc", init=tmp_path / "i.nc",
                                 start_time="2026-10-03_12:00:00")
    # a caller that never set the flag gets the default, the detached stage
    assert forecast_door.run_default_products(request, [out / "h.nc"], argparse.Namespace()) == 0
    assert started["command"][1:4] == ["-m", "hexcore.cli", "products"]


def test_releasing_device_memory_is_a_no_op_without_a_device_stack(monkeypatch):
    import sys

    from hexcore import forecast_door

    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy -> ImportError
    assert forecast_door.release_device_memory() is None


def test_what_the_point_pictures_need_is_asked_for_before_any_view_is_drawn(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)

    def absent(explicit, convert):
        raise products_door.ProductDoorError("rw_hex_points is not installed")

    monkeypatch.setattr(products_door, "resolve_points_exe", absent)
    args = _arguments(tmp_path, _histories(tmp_path), case="patch-case")
    with pytest.raises(MpasPortError, match="rw_hex_points is not installed"):
        products_door.run_products(args)
    assert seen == []  # no converter, catalog or renderer start was spent


def test_frames_in_flight_are_sized_from_the_host(tmp_path):
    histories = []
    for k in range(9):
        path = tmp_path / f"h{k}.nc"
        path.write_bytes(b"x" * 1000)
        histories.append(path)
    gib = 1 << 30
    # memory: half of 192 kB available / (8 x 1 kB x 3 views) = 4 frames
    assert products_door.auto_frame_jobs(histories, 3, available=192_000, cpus=256) == 4
    # cores: 12 cores / (2 x 3 views) = 2
    assert products_door.auto_frame_jobs(histories, 3, available=gib, cpus=12) == 2
    # never more than the frames there are, nor than the ceiling
    assert products_door.auto_frame_jobs(histories[:2], 1, available=gib, cpus=256) == 2
    assert products_door.auto_frame_jobs(histories, 1, available=gib, cpus=256) == products_door.MAX_FRAME_JOBS
    # a host that cannot hold two frames, or cannot be read, runs one at a time
    assert products_door.auto_frame_jobs(histories, 3, available=10_000, cpus=256) == 1
    assert products_door.auto_frame_jobs([], 3, available=gib, cpus=256) == 1


def test_frame_jobs_reach_both_engines_and_only_when_they_take_them(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    args = _arguments(tmp_path, _histories(tmp_path), case="patch-case", frame_jobs="2")
    args.no_points = True
    assert products_door.run_products(args) == 0
    converts = [c for stage, c in seen if stage == "CONVERT"]
    assert converts and all(c[c.index("--jobs") + 1] == "2" for c in converts)
    imports = [c for stage, c in seen if stage in ("CATALOG", "RENDER")]
    assert imports and all(c[c.index("--import-jobs") + 1] == "2" for c in imports)
    manifest = json.loads((tmp_path / "out" / "patch-case" / "products-manifest.json").read_text())
    assert manifest["frame_jobs"] == 2

    # an engine built before the flags is never handed them
    seen.clear()
    monkeypatch.setattr(render_door, "engine_accepts", lambda exe, flag: False)
    args = _arguments(tmp_path, _histories(tmp_path), case="older-engines", frame_jobs="2")
    args.no_points = True
    assert products_door.run_products(args) == 0
    assert not any("--jobs" in c or "--import-jobs" in c for _stage, c in seen)

    with pytest.raises(MpasPortError, match="--frame-jobs"):
        render_door.checked_frame_jobs("many")


def test_after_a_forecast_the_run_folder_is_the_whole_command(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    run = tmp_path / "fc"
    run.mkdir()
    grid = tmp_path / "meshes" / "patch.grid.nc"
    grid.parent.mkdir()
    grid.write_bytes(b"m")
    init = tmp_path / "meshes" / "patch.init.nc"
    init.write_bytes(b"i")
    for stamp in ("2026-10-03_18.00.00", "2026-10-04_00.00.00"):
        (run / f"cuda-history.{stamp}.nc").write_bytes(b"h")
    # the forecast door's own receipt: grid, init and the run's start (six hours
    # before the first frame kept here, which a guess from the frames would miss)
    (run / "forecast-receipt.json").write_text(json.dumps({
        "mesh": {"grid": str(grid)}, "init": {"path": str(init)},
        "schedule": {"start_time": "2026-10-03_12:00:00"}}), encoding="utf-8")
    parser = argparse.ArgumentParser()
    products_door.add_products_arguments(parser)
    assert products_door.run_products(parser.parse_args([str(run)])) == 0

    case = run / "products" / "patch-2026100312"
    manifest = json.loads((case / "products-manifest.json").read_text())
    assert manifest["mesh"] == str(grid) and manifest["init"] == str(init)
    assert manifest["simulation_start"] == "2026-10-03_12:00:00"
    converts = [c for stage, c in seen if stage == "CONVERT"]
    assert all(c[c.index("--mesh") + 1] == str(grid) and c[c.index("--init") + 1] == str(init)
               for c in converts)
    assert not list(run.glob("products/*/rustwx_*"))

    # a receipt that names a grid no longer on disk is refused as the receipt's
    grid.unlink()
    with pytest.raises(MpasPortError, match=r"forecast-receipt.json \(mesh.grid\) names a missing file"):
        products_door.run_products(parser.parse_args([str(run)]))
    # and with no receipt and no flag the door says what it lacks
    (run / "forecast-receipt.json").unlink()
    with pytest.raises(MpasPortError, match="no mesh: --mesh was not given"):
        products_door.run_products(parser.parse_args([str(run)]))


def test_completed_chemistry_run_adds_species_to_normal_products(monkeypatch, tmp_path):
    """No extra product flags are required after a completed aerosol forecast."""
    histories = _histories(tmp_path)
    args = _arguments(tmp_path, histories, case="aerosols")
    args.forecast_dir = str(tmp_path)
    args.no_points = args.no_mesh_view = True
    species = [dict(name=name, dry_deposition_m_s=0., settling_m_s=0.,
                    mass_extinction_m2_kg=1.) for name in ("dust_bin_7", "salt_fine")]
    (tmp_path / "chemistry-receipt.json").write_text(json.dumps({
        "schema": "gpuwm-hex.chemistry/v1", "species": species}))
    seen = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=True, seen=seen)
    renderer = render_door._run
    # Real Rust raw-plane import uses wrf_ store names, never NetCDF names.
    stored = {
        "var:wrf_dust_bin_7_column_mass", "var:wrf_dust_bin_7_surface_concentration",
        "var:wrf_dust_bin_7_aod550", "var:wrf_dust_bin_7_dry_deposition",
        "var:wrf_dust_bin_7_settled_mass", "var:wrf_salt_fine_column_mass",
        "var:wrf_salt_fine_surface_concentration", "var:wrf_salt_fine_aod550",
        "var:wrf_salt_fine_dry_deposition", "var:wrf_salt_fine_settled_mass",
        "var:wrf_aod550",
    }
    def with_chemistry_catalog(command, *, log_prefix, env=None):
        result = renderer(command, log_prefix=log_prefix, env=env)
        if log_prefix == "CATALOG":
            result.stdout += "".join(f"PRODUCT\t{slug}\tdirect\trenderable\tready\t\n"
                for slug in sorted(stored))
        return result
    monkeypatch.setattr(render_door, "_run", with_chemistry_catalog)
    assert products_door.run_products(args) == 0
    commands = [command for label, command in seen if label == "RENDER"]
    assert commands
    drawn = commands[0][commands[0].index("--products")+1].split(",")
    assert stored <= set(drawn)
    assert "var:dust_bin_7_column_mass" not in drawn
    manifest = json.loads((tmp_path / "out/aerosols/products-manifest.json").read_text())
    assert "var:wrf_dust_bin_7_dry_deposition" in manifest["chemistry_products"]


def test_limited_area_products_only_convert_the_domain(monkeypatch, tmp_path):
    seen = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(products_door, "forecast_receipt", lambda path: {
        "admission": {"configuration": "limited-area"}})
    args = _arguments(tmp_path, _histories(tmp_path))
    args.no_points = True
    assert products_door.run_products(args) == 0
    converts = [command for stage, command in seen if stage == "CONVERT"]
    assert [command[command.index("--window") + 1] for command in converts] == ["mesh"]
    manifest = json.loads(next((tmp_path / "out").rglob("products-manifest.json")).read_text())
    assert [view["view"] for view in manifest["views"]] == ["refined-core"]
    assert manifest["maps_rendered"] > 0 and manifest["failed_views"] == []


def test_window_products_follow_the_stored_frames(tmp_path):
    """Rolling 2.9.4 run: a 3-hourly global history was asked for hourly windows it cannot close."""
    def frames(hours, step):
        return [tmp_path / f"cuda-history.2024-10-{8 + (12 + h) // 24:02d}_{(12 + h) % 24:02d}.00.00.nc"
                for h in range(0, hours + 1, step)]
    start = "2024-10-08_12:00:00"
    hourly, withheld = products_door.windowed_for_history(frames(24, 1), start)
    assert hourly == products_door.WINDOWED_PRODUCTS and withheld == {}
    three, withheld = products_door.windowed_for_history(frames(24, 3), start)
    assert three == ("qpf_6h", "qpf_24h")
    assert set(withheld) == {"qpf_1h", "10m_wind_run_max", "2m_temp_0_24h_max", "2m_temp_0_24h_min"}
    assert "every 1 h" in withheld["qpf_1h"] and "F001" in withheld["qpf_1h"]
    short, withheld = products_door.windowed_for_history(frames(6, 1), start)
    assert short == ("qpf_1h", "qpf_6h", "10m_wind_run_max")
    assert "needs 24 h" in withheld["qpf_24h"]
