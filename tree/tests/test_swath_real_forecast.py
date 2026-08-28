"""What the placement layer does when it meets THIS PROJECT'S forecast.

The swath layer was proven against a fixture whose fields are analytic and
whose file is one self-describing history with every frame on one ``Time``
axis.  A forecast from ``gpuwm-hex forecast`` is not that file.  It is one
file per frame, each with ``Time`` of 1, carrying ``latCell`` and
``lonCell`` but no ``areaCell``, no ``cellsOnCell``, no ``nEdgesOnCell``
and no ``xtime`` -- and the shipped reader refused it by name.

Two defects are pinned here, both found by pointing the door at real
output and neither reachable from the fixture:

1. THE REACH DEFECT.  A capability that cannot read its own project's
   forecast has no front door.  The fix is not to add variables to the
   history stream -- those frames' digests are pinned by the execution
   anchors -- but to read the run receipt written beside them, which
   already names the sequence, the times and the grid.

2. THE FIELD DEFECT.  ``low_pressure_centre`` searched RAW surface
   pressure for a minimum.  On a real 96 km global forecast that returns
   the Tibetan Plateau, not a cyclone: elevation beats every weather
   feature on Earth by about 45 kPa.  No threshold VALUE repairs that,
   because the ordering itself is wrong.  The field had to become a
   sea-level reduction.

The shapes below are a test double of the real artifact, and the real
artifact is what they were built from: see
``evidence/swath-first-real-20260826/`` for the same door driven against a
24 h u96.64002 global run on the proving RTX 5070 Ti.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import build_swath_fixture_history as fixture  # noqa: E402
from hexcore.cli import main  # noqa: E402
from hexcore.swath import registry  # noqa: E402
from hexcore.swath.errors import SwathRefusal  # noqa: E402
from hexcore.swath.history import HistoryReader  # noqa: E402

CELLS = 10242
HOURS = [0.0, 3.0, 6.0, 9.0, 12.0]
SCENARIO = [
    {"kind": "low", "latitude_deg": 16.0, "longitude_deg": -52.0, "bearing_deg": 300.0,
     "speed_km_per_hour": 22.0, "radius_km": 420.0, "amplitude": 4200.0},
]

#: A plateau under a quiet part of the world, at the height and width the
#: real thing has.  Tibet is about 5,000 m over some 2,000 km.
PLATEAU = [
    {"latitude_deg": 33.0, "longitude_deg": 85.0, "radius_km": 1100.0,
     "height_m": 5200.0},
]

#: The variables this project's forecast door actually writes on ``nCells``
#: -- verified against
#: ``<work-dir>/graded-mesh-g12/runs/run2/out/cuda-history.*.nc`` on
#: the proving RTX 5090, whose full variable list carries latCell and lonCell but none
#: of areaCell / cellsOnCell / nEdgesOnCell / xtime.
DOOR_WRITES_NO_CONNECTIVITY = ("areaCell", "cellsOnCell", "nEdgesOnCell", "xtime")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_like_the_forecast_door(source: Path, root: Path) -> Path:
    """Re-shape one fixture history into what ``gpuwm-hex forecast`` writes.

    One file per frame with ``Time`` of 1 and NO connectivity, a separate
    grid file that carries the mesh, and the run receipt that ties them
    together.  This is a double of the real layout, not an invention of
    one: every structural choice here was read off a real
    ``cuda-history.*.nc`` and its ``cuda-v841-forecast-receipt.json``.
    """

    from netCDF4 import Dataset

    root.mkdir(parents=True, exist_ok=True)
    out = root / "out"
    out.mkdir(exist_ok=True)
    frames: dict[str, dict[str, object]] = {}
    labels: dict[str, str] = {}

    with Dataset(str(source), "r") as whole:
        count = int(whole.dimensions["Time"].size)
        cells = int(whole.dimensions["nCells"].size)
        levels = int(whole.dimensions["nVertLevels"].size)

        grid_path = root / "mesh.grid.nc"
        with Dataset(str(grid_path), "w", format="NETCDF4_CLASSIC") as grid:
            grid.createDimension("nCells", cells)
            grid.createDimension("maxEdges", whole.dimensions["maxEdges"].size)
            for name in ("latCell", "lonCell", "areaCell", "nEdgesOnCell"):
                variable = grid.createVariable(name, "f8", ("nCells",))
                variable[:] = np.asarray(whole.variables[name][:])
            variable = grid.createVariable(
                "cellsOnCell", "i4", ("nCells", "maxEdges")
            )
            variable[:] = np.asarray(whole.variables["cellsOnCell"][:])

        stamps = [
            b"".join(
                bytes(entry) if isinstance(entry, (bytes, bytearray))
                else str(entry).encode()
                for entry in np.atleast_1d(row).ravel()
            ).decode("ascii", "ignore").strip()
            for row in np.asarray(whole.variables["xtime"][:])
        ]
        for index in range(count):
            label = stamps[index].replace(":", ".")
            frame_path = out / f"cuda-history.{label}.nc"
            with Dataset(str(frame_path), "w", format="NETCDF4_CLASSIC") as frame:
                frame.createDimension("Time", 1)
                frame.createDimension("nCells", cells)
                frame.createDimension("nVertLevels", levels)
                for name in ("latCell", "lonCell"):
                    variable = frame.createVariable(name, "f8", ("nCells",))
                    variable[:] = np.asarray(whole.variables[name][:])
                # DRIVEN BY THE PUBLICATION MANIFEST, not by a list kept in
                # step by hand.  The manifest is what a coarse run is told
                # to publish, so a double built from anything else stops
                # being a double the moment a threat row is added.
                extra_axes: dict[str, int] = {}
                for name in registry.load_metrics().publication_manifest():
                    source_variable = whole.variables[name]
                    values = np.asarray(source_variable[index])
                    if values.ndim == 1:
                        variable = frame.createVariable(
                            name, "f8", ("Time", "nCells")
                        )
                        variable[:] = values[None]
                        continue
                    # The shipped history writer already puts the level axis
                    # SECOND on disk -- (Time, nCells, nVertLevels) -- which
                    # is what this project's forecast door writes too, so
                    # the double copies the axis names through rather than
                    # inventing an orientation.
                    axes = tuple(source_variable.dimensions[1:])
                    for axis_name, size in zip(axes, values.shape):
                        if axis_name not in frame.dimensions:
                            frame.createDimension(axis_name, int(size))
                            extra_axes[axis_name] = int(size)
                    variable = frame.createVariable(name, "f8", ("Time", *axes))
                    variable[:] = values[None]
            step = str(index * 30)
            frames[step] = {
                "path": str(frame_path),
                "bytes": frame_path.stat().st_size,
                "sha256": _sha256(frame_path),
            }
            labels[step] = label

    receipt = root / "cuda-v841-forecast-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "case_label": "swath-reach-double",
                "forecast": {
                    "snapshot_files": frames,
                    "history_labels": labels,
                    "authority": {
                        "files": {
                            "grid": {
                                "path": str(grid_path),
                                "bytes": grid_path.stat().st_size,
                                "sha256": _sha256(grid_path),
                            }
                        }
                    },
                },
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return receipt


@pytest.fixture(scope="module")
def whole_history(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("swath-real") / "coarse.nc"
    return fixture.build(target, cells=CELLS, hours=HOURS, scenario=SCENARIO)


@pytest.fixture(scope="module")
def run_receipt(whole_history: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _split_like_the_forecast_door(
        whole_history, tmp_path_factory.mktemp("swath-real-run")
    )


# ---------------------------------------------------------------------------
# 1. the reach defect
# ---------------------------------------------------------------------------
def test_a_forecast_frame_really_does_lack_what_the_detector_needs(
    run_receipt: Path,
) -> None:
    """The double is faithful: it withholds exactly what the real one does."""

    from netCDF4 import Dataset

    receipt = json.loads(run_receipt.read_text(encoding="utf-8"))
    first = sorted(receipt["forecast"]["snapshot_files"], key=int)[0]
    frame = Path(receipt["forecast"]["snapshot_files"][first]["path"])
    with Dataset(str(frame), "r") as dataset:
        present = set(dataset.variables)
        assert {"latCell", "lonCell", "surface_pressure"} <= present
        for absent in DOOR_WRITES_NO_CONNECTIVITY:
            assert absent not in present
        assert int(dataset.dimensions["Time"].size) == 1


def test_one_forecast_frame_is_refused_and_the_refusal_names_the_fix(
    run_receipt: Path,
) -> None:
    """A single frame cannot be a plan, and saying so is not enough.

    A refusal that only says 'no areaCell' leaves the operator holding a
    forecast and no way in.  It must name the door.
    """

    receipt = json.loads(run_receipt.read_text(encoding="utf-8"))
    first = sorted(receipt["forecast"]["snapshot_files"], key=int)[0]
    frame = receipt["forecast"]["snapshot_files"][first]["path"]
    with pytest.raises(SwathRefusal) as refusal:
        HistoryReader(frame)
    text = str(refusal.value)
    assert "areaCell" in text
    assert "receipt" in text and "--grid" in text


def test_the_run_receipt_is_a_readable_history(run_receipt: Path) -> None:
    """The artifact the forecast door writes IS the one the swath door reads."""

    with HistoryReader(run_receipt) as reader:
        assert reader.kind == "forecast_run_receipt"
        assert reader.cell_count == CELLS
        frames = reader.frames()
        assert len(frames) == len(HOURS)
        # The times are real seconds off the labels, not frame indices.
        assert [frame.time_seconds for frame in frames] == [
            hour * 3600.0 for hour in HOURS
        ]
        # And the mesh came from the grid the run was bound to.
        neighbours = reader.neighbours()
        assert neighbours.shape[0] == CELLS
        assert reader.areas_km2.shape == (CELLS,)


def test_a_grid_can_be_named_directly_for_a_frameless_history(
    whole_history: Path, run_receipt: Path
) -> None:
    """``--grid`` reaches a history whose connectivity lives elsewhere."""

    root = run_receipt.parent
    with HistoryReader(whole_history, grid=root / "mesh.grid.nc") as reader:
        assert reader.cell_count == CELLS
        assert len(reader.frames()) == len(HOURS)


def test_a_frame_that_is_not_the_one_the_receipt_recorded_is_refused(
    run_receipt: Path, tmp_path: Path
) -> None:
    """The SHA-256 gate: a plan must not name a forecast it was not built from.

    Without this, copying a run directory and re-running one frame would
    produce a plan whose receipt cites the ORIGINAL forecast's digest
    while its numbers came from the replacement -- a provenance claim that
    is silently false.
    """

    receipt = json.loads(run_receipt.read_text(encoding="utf-8"))
    moved = tmp_path / "moved-receipt.json"
    files = receipt["forecast"]["snapshot_files"]
    victim = sorted(files, key=int)[1]
    tampered = tmp_path / "tampered.nc"
    tampered.write_bytes(Path(files[victim]["path"]).read_bytes() + b"\0")
    files[victim]["path"] = str(tampered)
    moved.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(SwathRefusal) as refusal:
        HistoryReader(moved)
    assert "SHA-256" in str(refusal.value)


def test_a_document_that_is_not_a_run_receipt_refuses_by_key_path(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "not-a-receipt.json"
    wrong.write_text(json.dumps({"schema": "something.else"}), encoding="utf-8")
    with pytest.raises(SwathRefusal) as refusal:
        HistoryReader(wrong)
    assert "forecast.snapshot_files" in str(refusal.value)


def test_the_front_door_plans_from_a_run_receipt(
    run_receipt: Path, tmp_path: Path
) -> None:
    """Through ``hexcore.cli.main``, the way an installed user reaches it."""

    out = tmp_path / "plan"
    code = main(
        [
            "swath", "plan",
            "--history", str(run_receipt),
            "--out", str(out),
            "--no-size",
        ]
    )
    assert code == 0
    document = json.loads((out / "swath-plan.json").read_text(encoding="utf-8"))
    assert document["history"]["kind"] == "forecast_run_receipt"
    assert document["history"]["frames_verified_against_receipt"] is True
    assert len(document["history"]["frame_files"]) == len(HOURS)


# ---------------------------------------------------------------------------
# 2. the field defect: a minimum in raw surface pressure is a mountain
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def history_with_terrain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("swath-terrain") / "coarse.nc"
    return fixture.build(
        target, cells=CELLS, hours=HOURS, scenario=SCENARIO, terrain=PLATEAU
    )


def _row(metrics: object, field_id: str) -> object:
    return metrics.field_rows[field_id]


def test_raw_surface_pressure_ranks_a_plateau_below_every_cyclone(
    history_with_terrain: Path,
) -> None:
    """The measured defect, in miniature and in both arms.

    Arm one is the field the layer shipped with: ``direct`` on
    ``surface_pressure``.  Its global minimum sits on the plateau, tens of
    kPa below the storm, so an ``extremum_ball`` minimum search hands back
    a mountain every cycle and the storm never places.

    Arm two is the shipped field now: the same search on the sea-level
    reduction puts the minimum back on the storm.
    """

    metrics = registry.load_metrics(None)
    reduced = _row(metrics, "surface_low")
    assert reduced.derivation_kind == "sea_level_reduction"

    raw = registry.FieldRow(
        id="raw_surface_pressure",
        source_variables=("surface_pressure",),
        derivation_kind="direct",
        units="Pa",
    )

    with HistoryReader(history_with_terrain) as reader:
        latitude = reader.latitudes_deg
        longitude = reader.longitudes_deg
        height = reader.derive(
            registry.FieldRow(
                id="terrain", source_variables=("ter",),
                derivation_kind="direct", units="m",
            ),
            0,
        )
        raw_values = reader.derive(raw, 0)
        reduced_values = reader.derive(reduced, 0)

    raw_minimum = int(np.argmin(raw_values))
    reduced_minimum = int(np.argmin(reduced_values))

    # Arm one lands on the plateau.
    assert height[raw_minimum] > 4000.0
    assert abs(latitude[raw_minimum] - PLATEAU[0]["latitude_deg"]) < 12.0

    # Arm two lands on the storm, at sea level.
    assert height[reduced_minimum] < 100.0
    assert abs(latitude[reduced_minimum] - SCENARIO[0]["latitude_deg"]) < 8.0
    assert abs(longitude[reduced_minimum] - SCENARIO[0]["longitude_deg"]) < 8.0

    # And the size of the error is the point: the mountain outranks the
    # storm by far more than any threshold could arbitrate.
    assert raw_values[raw_minimum] < raw_values[reduced_minimum] - 20_000.0


def test_the_reduction_is_the_identity_on_a_sea_level_world(
    whole_history: Path,
) -> None:
    """Nothing moves where there is no terrain, so no calibrated number moved."""

    metrics = registry.load_metrics(None)
    reduced = _row(metrics, "surface_low")
    raw = registry.FieldRow(
        id="raw", source_variables=("surface_pressure",),
        derivation_kind="direct", units="Pa",
    )
    with HistoryReader(whole_history) as reader:
        assert np.allclose(reader.derive(raw, 0), reader.derive(reduced, 0))


def test_a_temperature_that_is_not_kelvin_is_refused_by_name(
    history_with_terrain: Path,
) -> None:
    """A celsius temperature would move every centre and look ordinary."""

    celsius = registry.FieldRow(
        id="bad", source_variables=("surface_pressure", "ter", "u10"),
        derivation_kind="sea_level_reduction", units="Pa",
    )
    with HistoryReader(history_with_terrain) as reader:
        with pytest.raises(SwathRefusal) as refusal:
            reader.derive(celsius, 0)
    assert "kelvin" in str(refusal.value)


def test_the_publication_manifest_grew_with_the_field(capsys) -> None:
    """A row that needs new variables must say so at the door, not at read time."""

    assert main(["swath", "metrics", "--publication-manifest"]) == 0
    printed = json.loads(capsys.readouterr().out)
    # The sea-level reduction's three, which is the defect this file pins,
    # and the leaves the COMPOSED rows reach through their inputs -- q2 for
    # the humidity chain, smois for the fuel condition, u_zonal and
    # v_meridional for the shear, qv for the vapour transport, and the
    # three accumulators the rate rows difference.
    assert {"surface_pressure", "ter", "t2", "u10", "v10", "refl10cm"} <= set(printed)
    assert {"q2", "smois", "u_zonal", "v_meridional", "qv", "w"} <= set(printed)
    assert {"rainc", "rainnc", "snownc"} <= set(printed)
    assert printed == sorted(printed)


# ---------------------------------------------------------------------------
# 3. the units defect: a grid file's areaCell is on the UNIT sphere
# ---------------------------------------------------------------------------
def _write_mesh(path: Path, source: Path, *, sphere_radius: float | None) -> None:
    """The same mesh, written on the sphere ``sphere_radius`` names."""

    from netCDF4 import Dataset

    with Dataset(str(source), "r") as whole, Dataset(
        str(path), "w", format="NETCDF4_CLASSIC"
    ) as out:
        cells = int(whole.dimensions["nCells"].size)
        out.createDimension("nCells", cells)
        out.createDimension("maxEdges", whole.dimensions["maxEdges"].size)
        scale = 1.0
        if sphere_radius is not None:
            out.setncattr("sphere_radius", sphere_radius)
            if sphere_radius < 1000.0:
                # A unit-sphere file stores the non-dimensional area.
                scale = 1.0 / (6371229.0 ** 2)
        for name in ("latCell", "lonCell", "nEdgesOnCell"):
            variable = out.createVariable(name, "f8", ("nCells",))
            variable[:] = np.asarray(whole.variables[name][:])
        variable = out.createVariable("areaCell", "f8", ("nCells",))
        variable[:] = np.asarray(whole.variables["areaCell"][:]) * scale
        variable = out.createVariable("cellsOnCell", "i4", ("nCells", "maxEdges"))
        variable[:] = np.asarray(whole.variables["cellsOnCell"][:])


def test_a_unit_sphere_mesh_is_refused_instead_of_silently_zeroing_areas(
    whole_history: Path, tmp_path: Path
) -> None:
    """The defect that made an armed detector permanently silent.

    Measured on a real 151,649-cell global forecast: taking areas off the
    GRID file gave every cell about 8.3e-11 km^2, so the convection row's
    20,000 km^2 floor rejected all 266 connected 35 dBZ regions at every
    frame -- and reported nothing wrong.  A wrong answer with no error is
    worse than a refusal, so this is now a refusal.
    """

    unit = tmp_path / "unit-sphere.grid.nc"
    _write_mesh(unit, whole_history, sphere_radius=1.0)
    with pytest.raises(SwathRefusal) as refusal:
        HistoryReader(whole_history, grid=unit)
    text = str(refusal.value)
    assert "sphere_radius" in text
    assert "unit sphere" in text
    assert "minimum_area_km2" in text


def test_a_real_sphere_mesh_gives_areas_a_cell_actually_has(
    whole_history: Path, tmp_path: Path
) -> None:
    real = tmp_path / "real-sphere.static.nc"
    _write_mesh(real, whole_history, sphere_radius=6371229.0)
    with HistoryReader(whole_history, grid=real) as reader:
        areas = reader.areas_km2
    # 10,242 cells over the globe is about 49,800 km^2 each.
    assert 20_000.0 < float(areas.mean()) < 120_000.0
    assert float(areas.sum()) == pytest.approx(4.0 * np.pi * 6371.229 ** 2, rel=0.02)


def test_a_receipt_prefers_the_static_over_the_grid(run_receipt: Path) -> None:
    """Both carry the connectivity; only one carries a real sphere."""

    from hexcore.swath import history as history_module

    assert history_module._RECEIPT_STATIC_FILE[-1] == "static"
    receipt = json.loads(run_receipt.read_text(encoding="utf-8"))
    files = receipt["forecast"]["authority"]["files"]
    grid = Path(files["grid"]["path"])
    static = grid.parent / "mesh.static.nc"
    _write_mesh(static, grid, sphere_radius=6371229.0)
    files["static"] = {
        "path": str(static),
        "bytes": static.stat().st_size,
        "sha256": _sha256(static),
    }
    both = grid.parent / "receipt-with-static.json"
    both.write_text(json.dumps(receipt), encoding="utf-8")
    with HistoryReader(both) as reader:
        assert Path(reader.provenance()["mesh_source"]).name == "mesh.static.nc"


def test_a_sea_level_row_without_three_sources_refuses_by_name() -> None:
    from hexcore.swath.errors import SwathDocumentError

    with pytest.raises(SwathDocumentError) as refusal:
        registry.FieldRow.from_mapping(
            {
                "id": "short",
                "source_variables": ["surface_pressure", "ter"],
                "derivation": {"kind": "sea_level_reduction"},
            }
        )
    assert "exactly 3 source variables" in str(refusal.value)
    assert "pressure, height, temperature" in str(refusal.value)
    assert "pressure, height, temperature" in str(refusal.value)
