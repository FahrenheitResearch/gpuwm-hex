"""Focused tests for the render door's pure logic and named refusals.

The engine legs (rw_mpas_convert, rw_wrfbatch) are proven against the real
binaries on the render node; what lives here is everything the door decides
WITHOUT them: executable resolution refusals, the scratch-placement gate,
the ruled layout, engine-filename splitting and the product-coverage
refusal grammar.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hexcore.render_door import (  # noqa: E402
    RenderDoorError,
    default_scratch,
    place_path,
    refuse_scratch_inside_out,
    refuse_uncovered_products,
    resolve_convert_exe,
    resolve_renderer_exe,
    split_engine_name,
    valid_day,
)


class TestValidDay:
    def test_iso_instant(self):
        assert valid_day("2025-03-15T00:00:00") == "2025-03-15"

    def test_wrf_stamp(self):
        assert valid_day("2025-03-15_00:00:00") == "2025-03-15"

    def test_unreadable_is_undated_not_guessed(self):
        assert valid_day("hour twelve") == "undated"
        assert valid_day(None) == "undated"
        assert valid_day("2025-02-31_00:00:00") == "undated"


class TestSplitEngineName:
    def test_engine_name_splits_on_known_slug(self):
        domain, product = split_engine_name(
            "rustwx_arwen_20250314_12z_f012_d01-22km_composite_reflectivity.png",
            "composite_reflectivity")
        assert domain == "d01-22km"
        assert product == "composite_reflectivity"

    def test_underscore_domain_survives(self):
        domain, product = split_engine_name(
            "rustwx_arwen_20250314_12z_f000_native_grid_2m_temperature.png",
            "2m_temperature")
        assert domain == "native_grid"
        assert product == "2m_temperature"

    def test_unparseable_name_files_under_native_grid(self):
        domain, product = split_engine_name("whatever.png", "slp")
        assert domain == "native_grid"
        assert product == "slp"

    def test_generic_var_slug_colon_and_hash_suffix(self):
        # The event says var:wrf_psfc; the filename spells it var_wrf_psfc
        # plus a content-hash suffix, and a ':' directory cannot exist on
        # Windows -- the product dir must use the safe spelling.
        domain, product = split_engine_name(
            "rustwx_wrf_20250314_12z_f012_d01-22km_var_wrf_psfc_647248f9d12a5e11.png",
            "var:wrf_psfc")
        assert domain == "d01-22km"
        assert product == "var_wrf_psfc"


class TestLayout:
    def test_ruled_layout_domain_product_day(self, tmp_path):
        path = place_path(tmp_path, "d01-22km", "slp", "2025-03-15", "x.png")
        assert path == tmp_path / "d01-22km" / "slp" / "2025-03-15" / "x.png"

    def test_missing_facts_stay_nameable(self, tmp_path):
        path = place_path(tmp_path, "", "", "", "x.png")
        assert path == (tmp_path / "native_grid" / "unclassified"
                        / "undated" / "x.png")


class TestScratchPlacement:
    def test_default_scratch_is_a_sibling(self, tmp_path):
        out = tmp_path / "png"
        scratch = default_scratch(out)
        assert scratch.parent == out.parent
        assert scratch != out
        assert out not in scratch.parents

    def test_scratch_inside_out_is_refused_by_name(self, tmp_path):
        out = tmp_path / "png"
        with pytest.raises(RenderDoorError, match="inside the delivered"):
            refuse_scratch_inside_out(out / "scratch", out)
        with pytest.raises(RenderDoorError, match="inside the delivered"):
            refuse_scratch_inside_out(out, out)

    def test_sibling_scratch_is_accepted(self, tmp_path):
        refuse_scratch_inside_out(tmp_path / "png.render-scratch",
                                  tmp_path / "png")


class TestExecutableResolution:
    """The refusals, on a machine that has none of the binaries.

    Emptying ``PATH`` and the environment variables is no longer enough to
    simulate that machine: the ladder also reads gpuwm's bridge directories,
    which is the whole point of the change that added them -- a developer box
    that has ever run ``gpuwm fetch-bridges`` HAS these binaries, and these
    tests used to pass there only because the door could not see them.  So
    the gpuwm rung is emptied explicitly, and there is a test below proving
    it is a rung at all.
    """

    @pytest.fixture(autouse=True)
    def _a_machine_with_no_engines(self, monkeypatch):
        from hexcore import engines

        for name in (
            "GPUWM_HEX_RW_WRFBATCH",
            "MPAS_PORT_RW_WRFBATCH",
            "GPUWM_RW_WRFBATCH",
            "GPUWM_HEX_RW_MPAS_CONVERT",
            "MPAS_PORT_RW_MPAS_CONVERT",
            "GPUWM_RW_MPAS_CONVERT",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: ())

    def test_missing_renderer_names_breakage_and_remedy(self):
        with pytest.raises(RenderDoorError) as caught:
            resolve_renderer_exe(None)
        message = str(caught.value)
        assert "rw_wrfbatch not found" in message
        assert "cargo build" in message
        assert "MPAS_PORT_RW_WRFBATCH" in message
        # The staged-bundle remedy comes FIRST, because it is one command
        # and needs no toolchain.  A refusal that only offers a cargo build
        # sends a user who has already installed gpuwm the long way round.
        assert "gpuwm fetch-bridges" in message

    def test_missing_converter_names_breakage_and_remedy(self):
        with pytest.raises(RenderDoorError) as caught:
            resolve_convert_exe(None)
        message = str(caught.value)
        assert "rw_mpas_convert not found" in message
        assert "cargo build" in message
        assert "gpuwm fetch-bridges" in message

    def test_env_pointing_at_missing_file_is_a_hard_error(self, monkeypatch):
        monkeypatch.setenv("MPAS_PORT_RW_WRFBATCH", "/no/such/rw_wrfbatch")
        with pytest.raises(RenderDoorError, match="which is not a file"):
            resolve_renderer_exe(None)

    def test_the_gpuwm_bridge_directory_is_a_real_rung(self, monkeypatch, tmp_path):
        """A staged binary resolves with NO environment variable set.

        This is the defect the shared ladder closed, stated as a test: a
        user who ran the one command that installs the engine still met a
        refusal, because no door looked where that command writes.
        """

        from hexcore import engines

        staged = tmp_path / engines.executable_name("rw_wrfbatch")
        staged.write_bytes(b"")
        staged.chmod(0o755)
        monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (staged,))

        assert resolve_renderer_exe(None) == staged.resolve()


class TestProductCoverage:
    ROWS = {
        "slp": ("derived", "renderable", "sea-level pressure"),
        "helicity_0_3km": ("derived", "missing-fields",
                           "needs stored U, V on model levels"),
    }

    def test_group_keywords_never_refuse(self):
        refuse_uncovered_products("all", self.ROWS, Path("h.nc"), [])
        refuse_uncovered_products("derived", self.ROWS, Path("h.nc"), [])

    def test_renderable_slug_passes(self):
        refuse_uncovered_products("slp", self.ROWS, Path("h.nc"), [])

    def test_uncovered_product_names_product_and_fields(self):
        with pytest.raises(RenderDoorError) as caught:
            refuse_uncovered_products(
                "slp,helicity_0_3km", self.ROWS, Path("h.nc"), ["W", "REFL"])
        message = str(caught.value)
        assert "helicity_0_3km" in message
        assert "needs stored U, V" in message
        assert "W, REFL" in message

    def test_unknown_slug_lists_renderable_catalog(self):
        with pytest.raises(RenderDoorError) as caught:
            refuse_uncovered_products("no_such", self.ROWS, Path("h.nc"), [])
        message = str(caught.value)
        assert "no_such" in message
        assert "slp" in message


class TestCatalogListing:
    """The door reads the renderer's --list-products rows.

    gpuwm 2.8.0's rw_wrfbatch prints six tab-separated fields per row (a
    machine reason code after the detail). A parser that demanded exactly
    five dropped every row and refused every render, so both widths are
    read here, and a listing with no rows at all still refuses.
    """

    @staticmethod
    def _listing(monkeypatch, tmp_path, stdout):
        from hexcore import render_door

        def fake_run(command, *, log_prefix):
            class Result:
                returncode = 0
                stderr = ""

            Result.stdout = stdout
            return Result()

        monkeypatch.setattr(render_door, "_run", fake_run)
        return render_door.list_catalog(
            Path("rw_wrfbatch"), Path("wrfout_d01"), scratch=tmp_path,
            heavy=False)

    def test_six_field_rows_of_the_current_renderer_are_read(
            self, monkeypatch, tmp_path):
        stdout = (
            "THEME default\n"
            "PRODUCT\tslp\tderived\trenderable\tsea-level pressure\t\n"
            "PRODUCT\t2m_temp_0_24h_max\twindowed\texcluded\twindowed "
            "accumulations need more than one stored whole-hour frame\t"
            "windowed-needs-whole-hour-frames\n"
            "CATALOG total=2 excluded=1 missing-fields=0 renderable=1\n"
        )
        rows, summary, _ = self._listing(monkeypatch, tmp_path, stdout)
        assert rows["slp"] == ("derived", "renderable", "sea-level pressure")
        assert rows["2m_temp_0_24h_max"][:2] == ("windowed", "excluded")
        assert summary.endswith("renderable=1")

    def test_five_field_rows_are_still_read(self, monkeypatch, tmp_path):
        rows, _, _ = self._listing(
            monkeypatch, tmp_path,
            "PRODUCT\tslp\tderived\trenderable\tsea-level pressure\n")
        assert rows == {"slp": ("derived", "renderable",
                                "sea-level pressure")}

    def test_a_listing_with_no_rows_refuses(self, monkeypatch, tmp_path):
        with pytest.raises(RenderDoorError, match="no catalog rows"):
            self._listing(monkeypatch, tmp_path, "THEME default\n")


class TestComposeArguments:
    """The compose door's refusals and the command it builds.

    A composite is several model runs going into one frame.  Everything
    here is about the ways that can be asked for incoherently, because a
    composite built from a half-stated source list would render cleanly
    and show the wrong weather.
    """

    @staticmethod
    def _parser():
        from hexcore.cli import build_parser

        return build_parser()

    def test_compose_makes_history_and_mesh_optional(self):
        parsed = self._parser().parse_args(
            ["render", "--compose", "sources.json", "--out", "png",
             "--window", "composite"])
        assert parsed.compose == "sources.json"
        assert parsed.history is None and parsed.mesh is None

    def test_a_single_run_render_still_requires_both(self):
        parsed = self._parser().parse_args(
            ["render", "--history", "h.nc", "--mesh", "m.nc", "--out", "png"])
        assert parsed.compose is None
        assert parsed.window == "mesh"

    def test_composite_window_is_offered(self):
        parsed = self._parser().parse_args(
            ["render", "--compose", "s.json", "--out", "png",
             "--window", "composite"])
        assert parsed.window == "composite"

    def test_compose_command_carries_no_second_source_list(self, tmp_path):
        """--compose owns the meshes; --history/--mesh must not also appear.

        Two source lists in one command can disagree about which run is
        the base, and the composite would then draw the wrong run
        everywhere the overlays do not cover.
        """

        from hexcore import render_door

        seen: dict[str, list[str]] = {}

        def fake_run(command, *, log_prefix):
            seen["command"] = [str(part) for part in command]
            report = tmp_path / "scratch" / "convert-report.json"
            report.write_text('{"frames": []}', encoding="utf-8")

            class Result:
                returncode = 0
                stdout = "COMPOSITE\t0\tcoarse\tpoints=4\tshare=1.0000\n"
                stderr = ""

            return Result()

        original = render_door._run
        render_door._run = fake_run
        try:
            report, command = render_door.convert_frames(
                Path("rw_mpas_convert"), histories=None, mesh=None,
                scratch=tmp_path / "scratch", window="composite",
                field_set="full", simulation_start=None, nc_format="cdf2",
                compose=tmp_path / "sources.json")
        finally:
            render_door._run = original

        assert "--compose" in command
        assert "--history" not in command
        assert "--mesh" not in command
        assert report == {"frames": []}

    def test_base_only_rides_with_compose_and_not_alone(self, tmp_path):
        from hexcore import render_door

        def fake_run(command, *, log_prefix):
            (tmp_path / "scratch").mkdir(parents=True, exist_ok=True)
            (tmp_path / "scratch" / "convert-report.json").write_text(
                '{"frames": []}', encoding="utf-8")

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        original = render_door._run
        render_door._run = fake_run
        try:
            _, command = render_door.convert_frames(
                Path("rw_mpas_convert"), histories=None, mesh=None,
                scratch=tmp_path / "scratch", window="composite",
                field_set="full", simulation_start=None, nc_format="cdf2",
                compose=tmp_path / "sources.json", compose_base_only=True)
        finally:
            render_door._run = original
        assert "--compose-base-only" in command

    def test_incoherent_commands_are_refused_before_a_binary_is_needed(
            self, tmp_path):
        """Named refusals, reachable without an engine installed.

        Each names the concrete breakage: two source lists that can
        disagree, a subtraction with nothing to subtract from, and a
        window sized to regions that were never given.
        """

        import argparse

        from hexcore.render_door import run_render

        cases = [
            (argparse.Namespace(
                compose="s.json", compose_base_only=False, history=["h.nc"],
                mesh=None, out=str(tmp_path / "png"), scratch=None,
                window="composite"),
             "two source lists"),
            (argparse.Namespace(
                compose=None, compose_base_only=True, history=["h.nc"],
                mesh="m.nc", out=str(tmp_path / "png"), scratch=None,
                window="mesh"),
             "no composite without --compose"),
            (argparse.Namespace(
                compose=None, compose_base_only=False, history=None,
                mesh=None, out=str(tmp_path / "png"), scratch=None,
                window="composite"),
             "no fine region to frame"),
            (argparse.Namespace(
                compose=None, compose_base_only=False, history=None,
                mesh=None, out=str(tmp_path / "png"), scratch=None,
                window="mesh"),
             "required unless --compose"),
        ]
        for namespace, expected in cases:
            with pytest.raises(RenderDoorError) as caught:
                run_render(namespace)
            assert expected in str(caught.value), (
                f"refusal for {namespace.window}/"
                f"compose={namespace.compose} did not name the breakage: "
                f"{caught.value}")


class TestNativeResolutionAndNames:
    """The live hex gallery's two defects, as the door now answers them.

    MEASURED 2026-10-10: a 15 km mesh drawn on 20 km and 28 km grids,
    headlined ``(d01 20 km)``, filed as ``rustwx_wrf_..._d01-20km_<p>.png``.
    """

    def test_global_means_the_mesh_spacing_and_the_old_box_keeps_its_name(self):
        from hexcore.render_door import converter_window

        assert converter_window("global") == "global-mesh"
        assert converter_window("global-0.25") == "global"
        assert converter_window("mesh") == "mesh"
        spec = "region:40,-105,9000,7000,30,60"
        assert converter_window(spec) == spec
        with pytest.raises(RenderDoorError, match="not a render window"):
            converter_window("atlantic")

    def test_window_argument_takes_names_and_specs_only(self):
        import argparse

        from hexcore.render_door import window_argument

        assert window_argument("global") == "global"
        assert window_argument("region:85,0,8400,8400") == "region:85,0,8400,8400"
        with pytest.raises(argparse.ArgumentTypeError, match="region:"):
            window_argument("lambert-ish")

    def test_names_carry_model_mesh_view_product_and_lead(self):
        from hexcore.render_door import (
            default_view, grid_token, lead_token, product_file_name)

        assert grid_token("hex 15 km", "d01-20km") == "hex15km"
        assert grid_token("hex 60-3 km", "d01-3km") == "hex60-3km"
        assert grid_token(None, "d01-20km") == "d01-20km"
        assert lead_token(21600, "x.png") == "f006"
        assert lead_token(1800, "x.png") == "f000h30m"
        # Sub-minute histories must not overwrite f000 or each other.
        assert lead_token(30, "x.png") == "f000h00m30s"
        assert lead_token(90, "x.png") == "f000h01m30s"
        assert lead_token(None, "rustwx_wrf_20261010_0z_f012_d01_t2.png") == "f012"
        assert default_view("global") == "global"
        assert default_view("global-0.25") == "global"
        assert default_view("region:40,-105,9000,7000") == "region"
        assert product_file_name("hex15km", "global", "500mb_height_winds",
                                 "f006") == "hex15km_global_500mb_height_winds_f006.png"

    def test_a_view_that_would_split_a_path_is_refused(self):
        from hexcore.render_door import checked_view

        assert checked_view("north-america") == "north-america"
        for bad in ("North America", "a/b", "", "-x"):
            with pytest.raises(RenderDoorError, match="--view"):
                checked_view(bad)

    def test_variables_keyword_is_a_group_not_a_missing_product(self):
        refuse_uncovered_products("all,variables", {}, Path("h.nc"), [])

    @pytest.mark.parametrize("source_label", [None, "Hex LES | ArWen physics"])
    def test_a_render_files_native_names_and_carries_georef(
            self, monkeypatch, tmp_path, source_label):
        """The door end to end over stand-in engines: what it asks the
        converter for, and where and under what name each picture lands."""

        import argparse
        import json

        from hexcore import render_door

        history = tmp_path / "cuda-history.2026-10-10_06.00.00.nc"
        mesh = tmp_path / "mesh.nc"
        history.write_bytes(b"h")
        mesh.write_bytes(b"m")
        monkeypatch.setattr(render_door, "resolve_convert_exe",
                            lambda explicit=None: Path("rw_mpas_convert"))
        monkeypatch.setattr(render_door, "resolve_renderer_exe",
                            lambda explicit=None: Path("rw_wrfbatch"))
        monkeypatch.setattr(render_door, "engine_accepts", lambda exe, flag: True)
        monkeypatch.setattr(render_door, "sha256_file", lambda path: "0" * 64)
        basemap = tmp_path / "basemap"
        (basemap / "natural_earth_110m").mkdir(parents=True)
        (basemap / "natural_earth_110m" / "ne_110m_coastline.shp").write_bytes(b"s")
        monkeypatch.setenv("RUSTWX_BASEMAP_DIR", str(basemap))
        seen: dict[str, list[str]] = {}
        environments: dict[str, dict | None] = {}

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        def fake_run(command, *, log_prefix, env=None):
            command = [str(part) for part in command]
            seen[log_prefix] = command
            environments[log_prefix] = env
            result = Result()
            if log_prefix == "CONVERT":
                frames = Path(command[command.index("--out-dir") + 1])
                wrfout = frames / "wrfout_d01_2026-10-10_06_00_00"
                wrfout.write_bytes(b"w")
                Path(command[command.index("--json") + 1]).write_text(json.dumps({
                    "grid_label": "hex 15 km",
                    "frames": [{"output": str(wrfout),
                                "valid_time": "2026-10-10T06:00:00",
                                "lead_seconds": 21600,
                                "absent_wrf_fields": []}]}), encoding="utf-8")
            elif log_prefix == "CATALOG":
                result.stdout = ("PRODUCT\t500mb_height_winds\tdirect\t"
                                 "renderable\tready\t\n")
            elif log_prefix == "RENDER":
                raw = Path(command[command.index("--out-dir") + 1])
                png = raw / ("rustwx_wrf_20261010_0z_f006_hex15km_"
                             "500mb_height_winds.png")
                png.write_bytes(b"png")
                (raw / "render-georef.json").write_text(json.dumps({
                    "schema": "rustwx.render-georef/v1",
                    "panels": {png.name: {"projection": "latlon"}},
                    "without_georeference": []}), encoding="utf-8")
                result.stdout = f"RENDERED 500mb_height_winds {png}\n"
            return result

        monkeypatch.setattr(render_door, "_run", fake_run)
        out = tmp_path / "png"
        args = argparse.Namespace(
            history=[str(history)], mesh=str(mesh), init=None, compose=None,
            compose_base_only=False, out=str(out), scratch=None,
            products="500mb_height_winds", frames="all", window="global",
            field_set="full", simulation_start="2026-10-10_00:00:00",
            size="1600x1000", heavy=False, nc_format="cdf2",
            keep_scratch=False, convert_exe=None, renderer_exe=None,
            view=None, model_label="hex", source_label=source_label)
        assert render_door.run_render(args) == 0

        expected_source = source_label if source_label is not None else "Hex forecast"
        render = seen["RENDER"]
        assert render[render.index("--source-label") + 1] == expected_source

        convert = seen["CONVERT"]
        assert convert[convert.index("--window") + 1] == "global-mesh"
        assert convert[convert.index("--model-label") + 1] == "hex"
        filed = (out / "global" / "500mb_height_winds" / "2026-10-10"
                 / "hex15km_global_500mb_height_winds_f006.png")
        assert filed.read_bytes() == b"png"
        assert not list(out.rglob("rustwx_*"))
        georef = json.loads((out / "global" / "render-georef.json").read_text())
        assert list(georef["panels"]) == [
            "500mb_height_winds/2026-10-10/"
            "hex15km_global_500mb_height_winds_f006.png"]
        manifest = json.loads(
            (out / "global" / "render-manifest.json").read_text())
        assert manifest["frames"][0]["placed"] == [
            "global/500mb_height_winds/2026-10-10/"
            "hex15km_global_500mb_height_winds_f006.png"]
        assert manifest["view"] == "global"
        assert manifest["grid_label"] == "hex 15 km"
        assert manifest["source_label"] == expected_source
        # Both renderer calls are handed the basemap root by name, and the
        # receipt says which one was drawn from.
        for stage in ("CATALOG", "RENDER"):
            assert environments[stage]["RUSTWX_BASEMAP_DIR"] == str(basemap)
        assert manifest["engine"]["basemap_dir"] == str(basemap)

    def test_a_render_with_no_basemap_geometry_is_refused_before_converting(
            self, monkeypatch, tmp_path):
        """A relocated renderer drew 59 maps with no coastline, border or
        state line and reported every one RENDERED (2026-10-11).  With no
        Natural Earth set anywhere on the ladder the door refuses by name,
        and it does so before the converter is launched."""

        import argparse

        from hexcore import engines, render_door

        history = tmp_path / "cuda-history.2026-10-10_06.00.00.nc"
        mesh = tmp_path / "mesh.nc"
        history.write_bytes(b"h")
        mesh.write_bytes(b"m")
        renderer = tmp_path / "bin" / "rw_wrfbatch"
        renderer.parent.mkdir()
        renderer.write_bytes(b"x")
        monkeypatch.setattr(render_door, "resolve_convert_exe",
                            lambda explicit=None: Path("rw_mpas_convert"))
        monkeypatch.setattr(render_door, "resolve_renderer_exe",
                            lambda explicit=None: renderer)
        monkeypatch.setattr(render_door, "engine_accepts", lambda exe, flag: True)
        monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
        monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
        monkeypatch.setattr(engines, "USER_BRIDGE_DIR", tmp_path / "bridges")
        monkeypatch.setattr(render_door, "is_basemap_root", lambda path: False)
        launched: list[str] = []
        monkeypatch.setattr(
            render_door, "_run",
            lambda command, **kwargs: launched.append(kwargs["log_prefix"]))
        args = argparse.Namespace(
            history=[str(history)], mesh=str(mesh), init=None, compose=None,
            compose_base_only=False, out=str(tmp_path / "png"), scratch=None,
            products="500mb_height_winds", frames="all", window="global",
            field_set="full", simulation_start=None, size="1600x1000",
            heavy=False, nc_format="cdf2", keep_scratch=False,
            convert_exe=None, renderer_exe=None, view=None, model_label="hex")
        with pytest.raises(RenderDoorError, match="no basemap geometry"):
            render_door.run_render(args)
        assert launched == []

    def test_the_basemap_root_beside_a_built_renderer_is_found(
            self, monkeypatch, tmp_path):
        from hexcore import engines, render_door

        monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
        monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
        monkeypatch.setattr(engines, "USER_BRIDGE_DIR", tmp_path / "bridges")
        renderer = tmp_path / "rustwx" / "target" / "release" / "rw_wrfbatch"
        renderer.parent.mkdir(parents=True)
        renderer.write_bytes(b"x")
        root = tmp_path / "rustwx" / "assets" / "basemap"
        (root / "natural_earth_10m").mkdir(parents=True)
        (root / "natural_earth_10m" / "ne_10m_coastline.shp").write_bytes(b"s")
        assert render_door.resolve_basemap_dir(renderer) == root
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        with pytest.raises(RenderDoorError, match="--basemap-dir"):
            render_door.resolve_basemap_dir(renderer, str(elsewhere))


def test_region_latlon_is_a_window_the_export_door_passes_to_the_converter():
    """The ArWen parent route exports on region-latlon:, the regular lat-lon box the engine's mapped
    NetCDF route reads; the door must pass it through, not refuse it as an unknown window."""
    from hexcore import render_door

    spec = "region-latlon:36,-91,2400,1800"
    assert render_door.window_argument(spec) == spec
    assert render_door.converter_window(spec) == spec
    assert render_door.default_view(spec) == "region"


def test_the_arwen_backend_publishes_sea_ice_and_snow_when_the_land_surface_holds_them():
    from hexcore import cuda_arwen_physics_v841 as backend

    assert {"xice", "snow", "snowh"} <= set(backend._OPTIONAL_ARWEN_EXPORT_FIELDS)


def test_split_native_streams_are_pinned_and_require_one_history(tmp_path, monkeypatch):
    from hexcore import render_door as rd
    from types import SimpleNamespace
    import hashlib
    init = tmp_path / 'init.nc'
    diag = tmp_path / 'diagnostics.nc'
    init.write_bytes(b'init-coordinates')
    diag.write_bytes(b'valid-time-diagnostics')
    def fake_run(command, **kwargs):
        (tmp_path / 'scratch' / 'convert-report.json').write_text('{"frames": []}')
        return SimpleNamespace(stdout='', stderr='', returncode=0)
    monkeypatch.setattr(rd, '_run', fake_run)
    kwargs = dict(histories=[tmp_path/'history.nc'], mesh=tmp_path/'grid.nc',
                  scratch=tmp_path/'scratch', window='mesh', field_set='full',
                  simulation_start=None, nc_format='cdf2', init=init, diagnostics=diag)
    _, cmd = rd.convert_frames(Path('convert'), **kwargs)
    for flag, path in (('--init-sha256', init), ('--diagnostics-sha256', diag)):
        assert cmd[cmd.index(flag)+1] == hashlib.sha256(path.read_bytes()).hexdigest()
    for bad in ({'init': None}, {'compose': tmp_path/'compose.json'},
                {'histories': []}, {'histories': [Path('a'), Path('b')]}):
        with pytest.raises(rd.RenderDoorError, match='requires one history'):
            rd.convert_frames(Path('convert'), **{**kwargs, **bad})


def test_an_open_ocean_window_gets_a_labelled_graticule():
    """The AL10 storm-following box (GFS 2026-10-22 18Z) sat over open ocean
    and its maps carried no geography at all; every hex map now carries
    latitude/longitude lines and labels inside the frame."""

    import numpy as np
    from hexcore.render_door import graticule_overlays

    lat, lon = np.meshgrid(np.linspace(38.0, 42.5, 160), np.linspace(-69.0, -63.0, 160), indexing="ij")
    overlays = graticule_overlays(lat, lon)
    texts = [label["text"] for label in overlays["labels"]]
    assert overlays["step_deg"] == 1.0
    assert "40\u00b0N" in texts and "66\u00b0W" in texts
    for label in overlays["labels"]:
        assert 38.0 <= label["lat"] <= 42.5 and -69.0 <= label["lon"] <= -63.0
    # Across the antimeridian the lines stay on one side of the seam.
    lat, lon = np.meshgrid(np.linspace(40.0, 45.0, 80), np.linspace(178.0, 184.0, 80), indexing="ij")
    overlays = graticule_overlays(lat, ((lon + 180.0) % 360.0) - 180.0)
    assert {"180\u00b0", "179\u00b0E", "179\u00b0W"} <= {label["text"] for label in overlays["labels"]}
    # The whole globe is drawn every 30 degrees.
    lat, lon = np.meshgrid(np.linspace(-89.5, 89.5, 90), np.linspace(-179.5, 179.5, 180), indexing="ij")
    assert graticule_overlays(lat, lon)["step_deg"] == 30.0
