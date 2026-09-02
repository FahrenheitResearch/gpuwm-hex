"""The paired-run evidence gallery: what it finds, what it draws, what it says.

Everything here runs on a CPU-only box and starts no forecast.  A synthetic
pair directory is written by hand -- forty cells on a small latitude/longitude
patch, three levels, three valid times -- and the tool is run over it exactly
as a user runs it.

WHY THE BUDGET NUMBERS ARE ASSERTED OUTRIGHT.  The budget line is the one
panel a reader will quote as a number rather than read as a picture, so it is
an instrument and it is tested as one.  The fixture puts non-zero values in
exactly two cells whose areas differ by a factor of two, and every expected
total below is a different number under each of the three mistakes worth
worrying about: the weights dropped, the weights applied per value instead of
per cell, and the levels summed the wrong way.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy
import pytest
from netCDF4 import Dataset

#: The gallery draws with matplotlib, which is the engine's dependency
#: (gpuwm installs it) and not this package's.  The CI battery installs
#: no engine, so on that box the gallery's tests skip by name rather than
#: fail at the first figure; wherever the pair door runs, gpuwm is present.
pytest.importorskip(
    "matplotlib",
    reason="the pair gallery draws with matplotlib, the engine's dependency; "
           "absent on the engine-less CI battery")

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore.pair_door import MANIFEST_NAME, PAIR_SCHEMA  # noqa: E402

CELLS = 40
LEVELS = 3
ROWS, COLUMNS = 5, 8

#: The three valid times, in the frames' own ``xtime`` spelling and in the
#: filename-safe stamp the PNG names carry.
VALID_TIMES = (
    "2026-01-01_00:00:00",
    "2026-01-01_01:00:00",
    "2026-01-01_02:00:00",
)
TOKENS = ("2026-01-01T000000", "2026-01-01T010000", "2026-01-01T020000")

#: The two cells the treatment leg acts in.  Their grid areas are 4.0 and 8.0
#: (``areaCell[c] = 1 + c``), which is the factor of two the weighting test
#: turns on.
ACTIVE = (3, 7)

#: What the treatment leg carries and the control leg does not.
EXTRA_SCALAR = "x_number"
EXTRA_ACCUMULATOR = "x_deposit"

#: Hand-computed totals, stated rather than recomputed in the assertion.
#:
#: area-weighted, record t (zero-based):
#:   x_number   = 4.0 * 3 levels * (t + 1) + 8.0 * 3 levels * 2 * (t + 1)
#:              = 12 * (t + 1) + 48 * (t + 1) = 60 * (t + 1)
#:   x_deposit  = 4.0 * 10 * (t + 1) + 8.0 * 5 * (t + 1) = 80 * (t + 1)
WEIGHTED_NUMBER = (60.0, 120.0, 180.0)
WEIGHTED_DEPOSIT = (80.0, 160.0, 240.0)
#: unweighted: 3 levels * (t + 1) + 3 levels * 2 * (t + 1) = 9 * (t + 1),
#: and 10 * (t + 1) + 5 * (t + 1) = 15 * (t + 1).
PLAIN_NUMBER = (9.0, 18.0, 27.0)
PLAIN_DEPOSIT = (15.0, 30.0, 45.0)


def _load_gallery() -> object:
    name = "_test_plot_pair_gallery"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        name, TOOLS / "plot_pair_gallery.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:  # pragma: no cover - an import error fails louder
        sys.modules.pop(name, None)
        raise
    return module


gallery = _load_gallery()


# ---------------------------------------------------------------------------
# the synthetic pair
# ---------------------------------------------------------------------------
def _positions() -> tuple[numpy.ndarray, numpy.ndarray]:
    """Cell centres in RADIANS, the way a history frame carries them."""

    index = numpy.arange(CELLS)
    latitude = 40.0 + 0.1 * (index // COLUMNS)
    longitude = -100.0 + 0.1 * (index % COLUMNS)
    return numpy.radians(latitude), numpy.radians(longitude)


def _write_char(variable, values) -> None:
    array = numpy.full((len(values), 64), b" ", dtype="S1")
    for row, value in enumerate(values):
        encoded = value.encode("ascii")
        array[row, : len(encoded)] = numpy.frombuffer(encoded, dtype="S1")
    variable[...] = array


def _write_frame(path: Path, *, treatment: bool, step: int) -> Path:
    """One history frame in the shape the port's writer emits.

    Written with netCDF4 directly: what matters here is the dimension names
    the gallery keys on -- ``Time``, ``nCells``, ``nVertLevels`` -- the
    radian coordinates on every frame, and the ABSENCE of ``areaCell``, which
    is why an area-weighted budget needs ``--grid`` at all.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    latitude, longitude = _positions()
    with Dataset(str(path), "w", format="NETCDF4") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("StrLen", 64)
        dataset.createDimension("nCells", CELLS)
        dataset.createDimension("nVertLevels", LEVELS)

        _write_char(
            dataset.createVariable("xtime", "S1", ("Time", "StrLen")),
            [VALID_TIMES[step]],
        )
        dataset.createVariable("latCell", "f8", ("nCells",))[:] = latitude
        dataset.createVariable("lonCell", "f8", ("nCells",))[:] = longitude

        qc = numpy.full((CELLS, LEVELS), 0.001)
        theta = numpy.full((CELLS, LEVELS), 300.0)
        if treatment:
            for cell in ACTIVE:
                qc[cell, :] += 0.0005
            theta[ACTIVE[0], :] += 0.5
        for name, values, units in (
            ("qc", qc, "kg kg^{-1}"),
            # Identical on both legs on purpose: the difference panel for a
            # field the treatment never touched must still be drawn, and it
            # is what exercises the all-zero branch of the diverging scale.
            ("qi", numpy.full((CELLS, LEVELS), 0.0002), "kg kg^{-1}"),
            ("theta", theta, "K"),
        ):
            variable = dataset.createVariable(
                name, "f8", ("Time", "nCells", "nVertLevels")
            )
            variable.units = units
            variable[0, :, :] = values

        if treatment:
            number = numpy.zeros((CELLS, LEVELS))
            number[ACTIVE[0], :] = 1.0 * (step + 1)
            number[ACTIVE[1], :] = 2.0 * (step + 1)
            variable = dataset.createVariable(
                EXTRA_SCALAR, "f8", ("Time", "nCells", "nVertLevels")
            )
            variable.units = "m^{-3}"
            variable[0, :, :] = number

            deposit = numpy.zeros(CELLS)
            deposit[ACTIVE[0]] = 10.0 * (step + 1)
            deposit[ACTIVE[1]] = 5.0 * (step + 1)
            variable = dataset.createVariable(
                EXTRA_ACCUMULATOR, "f8", ("Time", "nCells")
            )
            variable.units = "kg m^{-2}"
            variable[0, :] = deposit
    return path


def _write_grid(path: Path) -> Path:
    """The mesh file: areas, the boundary zone and the edge-length range."""

    with Dataset(str(path), "w", format="NETCDF4") as dataset:
        dataset.createDimension("nCells", CELLS)
        dataset.createDimension("nEdges", 3 * CELLS)
        dataset.createVariable("areaCell", "f8", ("nCells",))[:] = (
            1.0 + numpy.arange(CELLS, dtype=float)
        )
        index = numpy.arange(CELLS)
        row, column = index // COLUMNS, index % COLUMNS
        mask = (
            (row == 0) | (row == ROWS - 1) | (column == 0) | (column == COLUMNS - 1)
        ).astype("i4")
        dataset.createVariable("bdyMaskCell", "i4", ("nCells",))[:] = mask
        dataset.createVariable("dcEdge", "f8", ("nEdges",))[:] = numpy.linspace(
            9000.0, 11000.0, 3 * CELLS
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture
def pair(tmp_path: Path) -> Path:
    """A finished pair directory: two legs, three valid times, one manifest."""

    pair_out = tmp_path / "pair"
    table = tmp_path / "sources.csv"
    table.write_text(
        "latitude,longitude,height_m,rate\n40.2,-99.6,1500.0,1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    legs: dict[str, dict[str, object]] = {}
    for leg in ("control", "treatment"):
        out = pair_out / leg / "out"
        frames = [
            _write_frame(
                out / f"cuda-history.{VALID_TIMES[step].replace(':', '.')}.nc",
                treatment=leg == "treatment",
                step=step,
            )
            for step in range(len(VALID_TIMES))
        ]
        legs[leg] = {
            "name": leg,
            "out": str(out),
            "history": [
                {"path": str(frame), "sha256": _sha256(frame)} for frame in frames
            ],
        }
    manifest = {
        "schema": PAIR_SCHEMA,
        "tool": "gpuwm-hex pair",
        "pair_out": str(pair_out),
        "source_table": {"path": str(table), "sha256": _sha256(table)},
        "legs": legs,
        "leg_order": ["control", "treatment"],
    }
    (pair_out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return pair_out


def _pngs(out: Path) -> set[str]:
    return {path.name for path in out.glob("*.png")}


# ---------------------------------------------------------------------------
# what the gallery finds
# ---------------------------------------------------------------------------
def test_the_declared_extras_are_the_fields_only_the_treatment_leg_carries(
    pair: Path, tmp_path: Path
) -> None:
    """Exactly the two: nothing shared, nothing from the mesh, nothing else."""

    result = gallery.build_gallery(pair_out=pair, out=tmp_path / "g", frames=5)

    assert result["declared_extras"]["three_dimensional"] == [EXTRA_SCALAR]
    assert result["declared_extras"]["two_dimensional"] == [EXTRA_ACCUMULATOR]
    assert result["declared_extras"]["source"] == "discovered"
    # The shared weather fields are NOT extras, and the mesh coordinates every
    # frame carries are not fields at all.
    for name in ("qc", "qi", "theta", "latCell", "lonCell", "xtime"):
        assert name not in result["declared_extras"]["three_dimensional"]
        assert name not in result["declared_extras"]["two_dimensional"]


def test_the_shared_weather_fields_are_differenced_and_qc_is_among_them(
    pair: Path, tmp_path: Path
) -> None:
    out = tmp_path / "g"
    result = gallery.build_gallery(pair_out=pair, out=out, frames=5)

    assert result["difference_fields"] == ["qc", "qi", "theta"]
    for token in TOKENS:
        assert (out / f"{token}_diff_qc.png").is_file()


def test_extras_given_on_the_command_line_replace_the_discovered_list(
    pair: Path, tmp_path: Path
) -> None:
    out = tmp_path / "g"
    result = gallery.build_gallery(
        pair_out=pair, out=out, frames=1, extras=[EXTRA_ACCUMULATOR]
    )

    assert result["declared_extras"]["three_dimensional"] == []
    assert result["declared_extras"]["two_dimensional"] == [EXTRA_ACCUMULATOR]
    assert result["declared_extras"]["source"] == "--extras"
    assert f"{TOKENS[0]}_colmax_{EXTRA_SCALAR}.png" not in _pngs(out)


def _give_the_control_leg_the_banks(pair_out: Path, *, scalar_peak: float) -> None:
    """Add the two extras to every CONTROL frame: zero, or one cell at a peak.

    The shape a seeded row run with no point-source table produces: the
    banks exist on both legs and only the treatment leg's are non-zero.
    """

    for path in sorted((pair_out / "control" / "out").glob("*.nc")):
        with Dataset(str(path), "a") as dataset:
            scalar = dataset.createVariable(
                EXTRA_SCALAR, "f8", ("Time", "nCells", "nVertLevels")
            )
            scalar.units = "m^{-3}"
            values = numpy.zeros((CELLS, LEVELS))
            values[ACTIVE[1], 0] = scalar_peak
            scalar[0, :, :] = values
            accumulator = dataset.createVariable(
                EXTRA_ACCUMULATOR, "f8", ("Time", "nCells")
            )
            accumulator.units = "kg m^{-2}"
            accumulator[0, :] = numpy.zeros(CELLS)


def test_an_extra_the_control_leg_carries_at_zero_is_captioned_as_measured(
    pair: Path, tmp_path: Path
) -> None:
    """The caption reads the control frame instead of asserting absence."""

    _give_the_control_leg_the_banks(pair, scalar_peak=0.0)
    out = tmp_path / "g"
    result = gallery.build_gallery(
        pair_out=pair, out=out, frames=1,
        extras=[EXTRA_SCALAR, EXTRA_ACCUMULATOR],
    )

    scalar = result["captions"][f"{TOKENS[0]}_colmax_{EXTRA_SCALAR}.png"]
    accumulator = result["captions"][f"{TOKENS[0]}_accum_{EXTRA_ACCUMULATOR}.png"]
    for caption in (scalar, accumulator):
        assert "carries this field too" in caption
        assert "zero in every cell" in caption
        assert "does not carry this field" not in caption
    # The discovered list is now empty, because nothing is treatment-only.
    discovered = gallery.build_gallery(pair_out=pair, out=tmp_path / "d", frames=1)
    assert discovered["declared_extras"]["three_dimensional"] == []
    assert discovered["declared_extras"]["two_dimensional"] == []


def test_an_extra_the_control_leg_carries_non_zero_states_its_peak(
    pair: Path, tmp_path: Path
) -> None:
    _give_the_control_leg_the_banks(pair, scalar_peak=0.25)
    out = tmp_path / "g"
    result = gallery.build_gallery(
        pair_out=pair, out=out, frames=1, extras=[EXTRA_SCALAR]
    )

    caption = result["captions"][f"{TOKENS[0]}_colmax_{EXTRA_SCALAR}.png"]
    assert "carries this field too" in caption
    assert "largest magnitude" in caption and "0.25" in caption
    assert "zero in every cell" not in caption


def test_an_extra_the_treatment_leg_does_not_carry_is_refused_by_name(
    pair: Path, tmp_path: Path
) -> None:
    with pytest.raises(gallery.GalleryRefusal) as refusal:
        gallery.build_gallery(
            pair_out=pair, out=tmp_path / "g", extras=["x_absent"]
        )

    assert "x_absent" in str(refusal.value)
    assert "missing the panel it was asked for" in str(refusal.value)


# ---------------------------------------------------------------------------
# what the gallery draws
# ---------------------------------------------------------------------------
def test_the_gallery_writes_exactly_the_expected_panel_set(
    pair: Path, tmp_path: Path
) -> None:
    """Every PNG named outright: three valid times, two extras, three shared."""

    out = tmp_path / "g"
    grid = _write_grid(tmp_path / "mesh.nc")
    result = gallery.build_gallery(pair_out=pair, out=out, frames=5, grid=grid)

    expected = {f"{token}_colmax_{EXTRA_SCALAR}.png" for token in TOKENS}
    expected |= {f"{token}_accum_{EXTRA_ACCUMULATOR}.png" for token in TOKENS}
    expected |= {
        f"{token}_diff_{name}.png"
        for token in TOKENS
        for name in ("qc", "qi", "theta")
    }
    expected |= {
        f"{TOKENS[0]}_budget_{EXTRA_SCALAR}.png",
        f"{TOKENS[0]}_budget_{EXTRA_ACCUMULATOR}.png",
        f"{TOKENS[0]}_overview_domain.png",
    }

    assert _pngs(out) == expected
    assert len(expected) == 18
    assert set(result["pngs"]) == expected
    for name in expected:
        assert (out / name).stat().st_size > 0


def test_two_frames_selects_the_first_and_the_last(
    pair: Path, tmp_path: Path
) -> None:
    out = tmp_path / "g"
    result = gallery.build_gallery(pair_out=pair, out=out, frames=2)

    assert result["valid_times_matched"] == list(VALID_TIMES)
    assert result["valid_times_drawn"] == [VALID_TIMES[0], VALID_TIMES[-1]]
    drawn = _pngs(out)
    assert f"{TOKENS[0]}_colmax_{EXTRA_SCALAR}.png" in drawn
    assert f"{TOKENS[2]}_colmax_{EXTRA_SCALAR}.png" in drawn
    assert f"{TOKENS[1]}_colmax_{EXTRA_SCALAR}.png" not in drawn
    assert len(drawn) == 13


def test_the_overview_reports_the_mesh_and_the_point_source_table(
    pair: Path, tmp_path: Path
) -> None:
    out = tmp_path / "g"
    grid = _write_grid(tmp_path / "mesh.nc")
    result = gallery.build_gallery(pair_out=pair, out=out, frames=1, grid=grid)

    assert result["cells"] == CELLS
    caption = result["captions"][f"{TOKENS[0]}_overview_domain.png"]
    assert "40 cells" in caption
    assert "bdyMaskCell" in caption
    # The manifest's own record of the table, carried onto the page.
    assert "sources.csv" in json.dumps(result)


# ---------------------------------------------------------------------------
# the budget, which is a number a reader will quote
# ---------------------------------------------------------------------------
def test_the_budget_is_area_weighted_when_a_grid_is_given(
    pair: Path, tmp_path: Path
) -> None:
    grid = _write_grid(tmp_path / "mesh.nc")
    result = gallery.build_gallery(
        pair_out=pair, out=tmp_path / "g", frames=5, grid=grid
    )

    def series(name: str) -> list[float]:
        return [row["value"] for row in result["budgets"][name]["series"]]

    assert series(EXTRA_SCALAR) == pytest.approx(list(WEIGHTED_NUMBER))
    assert series(EXTRA_ACCUMULATOR) == pytest.approx(list(WEIGHTED_DEPOSIT))
    assert "areaCell" in result["weighting"]
    assert "areaCell" in result["budgets"][EXTRA_SCALAR]["weighting"]


def test_the_budget_is_a_plain_sum_and_says_so_without_a_grid(
    pair: Path, tmp_path: Path
) -> None:
    result = gallery.build_gallery(pair_out=pair, out=tmp_path / "g", frames=5)

    def series(name: str) -> list[float]:
        return [row["value"] for row in result["budgets"][name]["series"]]

    assert series(EXTRA_SCALAR) == pytest.approx(list(PLAIN_NUMBER))
    assert series(EXTRA_ACCUMULATOR) == pytest.approx(list(PLAIN_DEPOSIT))
    # Never a plain sum presented as an area integral.
    assert result["weighting"].startswith("UNWEIGHTED")


# ---------------------------------------------------------------------------
# what the gallery says
# ---------------------------------------------------------------------------
def test_every_png_carries_a_caption_and_the_index_lists_it(
    pair: Path, tmp_path: Path
) -> None:
    out = tmp_path / "g"
    gallery.build_gallery(pair_out=pair, out=out, frames=5)

    captions = json.loads((out / "captions.json").read_text(encoding="utf-8"))
    assert captions["schema"] == gallery.GALLERY_SCHEMA
    assert set(captions["captions"]) == _pngs(out)

    index = (out / "index.md").read_text(encoding="utf-8")
    for name, caption in captions["captions"].items():
        assert name in index
        assert caption in index
        assert len(caption.split()) >= 12


def test_every_difference_and_extra_caption_says_it_is_a_model_result(
    pair: Path, tmp_path: Path
) -> None:
    """The clause travels with the picture, because the picture travels."""

    out = tmp_path / "g"
    result = gallery.build_gallery(pair_out=pair, out=out, frames=5)

    clause = gallery.MODEL_RESULT_CLAUSE
    for name, caption in result["captions"].items():
        kind = name.split("_")[1]
        if kind == "overview":
            continue
        assert clause in caption, name
        assert "valid" in caption or "valid time" in caption


def test_the_captions_state_the_units_and_the_leg(
    pair: Path, tmp_path: Path
) -> None:
    out = tmp_path / "g"
    result = gallery.build_gallery(pair_out=pair, out=out, frames=1)

    assert "m^{-3}" in result["captions"][f"{TOKENS[0]}_colmax_{EXTRA_SCALAR}.png"]
    assert "kg m^{-2}" in result["captions"][
        f"{TOKENS[0]}_accum_{EXTRA_ACCUMULATOR}.png"
    ]
    difference = result["captions"][f"{TOKENS[0]}_diff_theta.png"]
    assert "K" in difference
    assert "control" in difference and "treatment" in difference


# ---------------------------------------------------------------------------
# the command line, and the refusals
# ---------------------------------------------------------------------------
def test_the_help_sends_weather_field_renders_to_the_renderer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_:
        gallery.main(["--help"])

    assert exit_.value.code == 0
    printed = capsys.readouterr().out
    assert "gpuwm-hex render" in printed
    assert "not this tool's job" in printed
    assert "<valid-time>_<kind>_<field>.png" in printed


def test_the_command_line_draws_the_gallery(
    pair: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "g"
    code = gallery.main(
        ["--pair-out", str(pair), "--out", str(out), "--frames", "2"]
    )

    assert code == 0
    assert "GALLERY" in capsys.readouterr().out
    assert (out / "index.md").is_file()


def test_a_directory_with_no_manifest_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A named refusal, not an empty gallery."""

    empty = tmp_path / "not-a-pair"
    empty.mkdir()

    with pytest.raises(gallery.GalleryRefusal) as refusal:
        gallery.build_gallery(pair_out=empty, out=tmp_path / "g")

    message = str(refusal.value)
    assert MANIFEST_NAME in message
    assert "which tree was the control leg" in message
    assert str(empty) in message

    assert gallery.main(["--pair-out", str(empty), "--out", str(tmp_path / "g2")]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_a_second_run_over_a_shorter_selection_leaves_no_stale_panel(
    pair: Path, tmp_path: Path
) -> None:
    """Re-running is allowed; mixing two runs' panels in one folder is not."""

    out = tmp_path / "g"
    gallery.build_gallery(pair_out=pair, out=out, frames=5)
    assert len(_pngs(out)) == 18

    result = gallery.build_gallery(pair_out=pair, out=out, frames=1)
    assert _pngs(out) == set(result["pngs"])
    assert f"{TOKENS[2]}_diff_qc.png" not in _pngs(out)

    (out / "stranger.png").write_bytes(b"not ours")
    with pytest.raises(gallery.GalleryRefusal) as refusal:
        gallery.build_gallery(pair_out=pair, out=out, frames=1)
    assert "stranger.png" in str(refusal.value)
