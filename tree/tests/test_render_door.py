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
