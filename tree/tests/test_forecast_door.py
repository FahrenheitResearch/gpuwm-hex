"""The forecast door: argument grammar, refusals, admission, receipt shape.

Everything here runs on a CPU-only box.  The door's device contact is
confined to three functions -- :func:`measure_device_memory`,
:func:`read_device_compute` and :func:`read_card_profile` -- so that the
DECISION they feed can be tested at every interesting free-memory value
without owning a card at that value, which is the only way to test the
refusal that fires on a card too small for the request.  All three are
substituted for every test in this file by the autouse fixtures below.

THE BREAKAGE THE THIRD ONE PREVENTS, measured on both CI matrix legs and
reproduced on the Windows desktop (``evidence/xmachine-20260827/`` §4):
``read_card_profile`` is a SECOND point of device contact that arrived after
this file was written, and it refuses outright when ``GPUWM_HEX_NO_LOCAL_GPU``
is set.  ``ci.yml``'s tier-1 step sets exactly that variable, so this
distribution's own CI environment made this distribution's own door refuse,
and ``test_the_door_leaves_destination_creation_to_the_driver`` -- a test
about directory creation, with no interest in any card -- went red on the
release commit.  The refusal itself is correct and stays: a box that has
declared no GPU work cannot run a forecast.  What was wrong is a test file
claiming its device contact was confined to one function when it was not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hexcore import forecast_door as door
from hexcore.cli import build_parser


X1_CELLS = 40_962
X4_CELLS = 163_842
V15_CELLS = 38_857
MIB = 1024**2
GIB = 1024**3


@pytest.fixture(autouse=True)
def _proven_architecture(monkeypatch):
    """Pin the architecture seam to the proven floor for every door test.

    The architecture half of admission has its own tests below; everything
    else in this file is about the door's other decisions and must not
    depend on which card (if any) the test box carries.
    """

    monkeypatch.setattr(door, "read_device_compute", lambda: (12, 0))


@pytest.fixture(autouse=True)
def _reference_card(monkeypatch):
    """Pin the card SHAPE seam for every door test.

    The footprint row is priced from the card's multiprocessor count, and
    ``read_card_profile`` reads that from the driver -- a device query, and
    a refusal when the box has banned GPU work.  Every test in this file
    decides something else, so the shape is supplied here rather than asked
    of whatever card (or ban) the runner happens to have.  The tests that
    are ABOUT the card's shape substitute their own row against the same
    seam, in tests/test_device_admission.py.
    """

    from hexcore.device_admission import REFERENCE_CARD

    monkeypatch.setattr(door, "read_card_profile", lambda: REFERENCE_CARD)


@pytest.fixture(autouse=True)
def _satisfied_seam_pin(monkeypatch):
    """Pin the gpuwm SEAM-PIN seam for every door test.

    ``--gpuwm-checkout`` in this file is an empty directory under
    ``tmp_path``: these tests are about the argument grammar, the schedule,
    the receipt and the driver vector, and none of them can carry a real
    gpuwm checkout at the pinned commit.  The door's byte check over that
    checkout is therefore supplied here, exactly as the card shape and the
    architecture are.

    The check ITSELF -- that it names the version found, the version wanted
    and the remedy, and that it stays silent on a checkout whose bytes match
    -- is exercised against real bytes in tests/test_engine_pin.py, where
    this substitution is not in force.
    """

    monkeypatch.setattr(door, "seam_pin_problem", lambda checkout: None)


def _registry() -> dict[str, door.MeshRow]:
    """A stand-in for the checkout registry, with the real three rows."""

    return {
        "x4.163842": door.MeshRow("x4.163842", X4_CELLS, 120.0),
        "x1.40962": door.MeshRow("x1.40962", X1_CELLS, 120.0),
        "v15.150.38857": door.MeshRow("v15.150.38857", V15_CELLS, 60.0),
    }


def _register_timestep_anchor(monkeypatch, dt_seconds: float) -> None:
    """Register one timestep anchor for the duration of a test.

    The same gesture a ruling would make in
    ``hexcore.dt_admission.ADMITTED_TIMESTEPS`` -- one row naming its
    evidence -- so a test can exercise behaviour past the gate without the
    gate being weakened for anyone else.
    """

    from hexcore import dt_admission

    anchor = dt_admission.DtAnchor(
        dt_seconds=float(dt_seconds),
        radiation_seconds=600.0,
        surface_pbl_seconds=float(dt_seconds),
        cumulus_seconds=float(dt_seconds),
        cumulus_scheme="gf",
        meshes=(),
        card="test-registered anchor",
        admitted_on="2026-08-26",
        schedule_receipt="evidence/dt-admission-20260826/",
        integration_anchor="evidence/dt-admission-20260826/",
        native_reference=None,
        basis="test-registered anchor",
        physics_health="TRACKS test-registered anchor",
    )
    monkeypatch.setattr(
        dt_admission,
        "ADMITTED_TIMESTEPS",
        {
            **dt_admission.ADMITTED_TIMESTEPS,
            dt_admission.dt_key(dt_seconds): anchor,
        },
    )


def _namespace(tmp_path: Path, **overrides) -> argparse.Namespace:
    """A complete, valid argument vector, so each test perturbs exactly one thing."""

    grid = tmp_path / "x1.40962.grid.nc"
    static = tmp_path / "x1.40962.static.nc"
    init = tmp_path / "init.nc"
    checkout = tmp_path / "gpuwm"
    for path in (grid, static, init):
        path.write_bytes(b"not really netcdf")
    checkout.mkdir()
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    arguments = parser.parse_args(
        [
            "--mesh", "x1.40962",
            "--grid", str(grid),
            "--static", str(static),
            "--init", str(init),
            "--init-source", "GFS 2026-08-24 00Z",
            "--hours", "1.0",
            "--history-every-minutes", "30",
            "--out", str(tmp_path / "out"),
            "--gpuwm-checkout", str(checkout),
            "--repo", str(door.PROJECT_ROOT or tmp_path),
        ]
    )
    for key, value in overrides.items():
        setattr(arguments, key, value)
    return arguments


# ---------------------------------------------------------------------------
# the door exists, on the console script, with the doors' grammar
# ---------------------------------------------------------------------------
def test_forecast_is_a_subcommand_of_the_console_script() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "forecast",
            "--mesh", "x1.40962",
            "--grid", "g.nc",
            "--static", "s.nc",
            "--init", "i.nc",
            "--init-source", "GFS",
            "--hours", "1",
            "--history-every-minutes", "30",
            "--out", "out",
        ]
    )
    assert arguments.command == "forecast"
    assert arguments.handler is not None


def test_defaults_are_the_pinned_configuration() -> None:
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    arguments = parser.parse_args([])
    # Fixed means default: the proven lane is what a bare run gets.
    assert arguments.horiz_mixing == "2d_smagorinsky"
    assert arguments.local_timestep is False
    # None means "the model's own margin", which is the default now: the
    # margin is priced from the card, so a flat default would be one card's
    # number handed to every other card -- the shape of ledger #366.
    assert arguments.headroom_mib is None
    assert arguments.device_fixed_mib is None
    assert arguments.device_bytes_per_cell is None
    assert arguments.scratch is None


def test_help_names_every_flag_the_manual_documents() -> None:
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    text = parser.format_help()
    for flag in (
        "--mesh", "--grid", "--static", "--init", "--init-source",
        "--hours", "--history-every-minutes", "--out", "--scratch",
        "--gpuwm-checkout", "--repo", "--case-label", "--horiz-mixing",
        "--local-timestep", "--headroom-mib", "--device-fixed-mib",
        "--device-bytes-per-cell", "--stop-on-refusal", "--preflight",
        "--receipt",
    ):
        assert flag in text, flag


# ---------------------------------------------------------------------------
# refusals: each names the breakage and the remedy
# ---------------------------------------------------------------------------
def _refusal(arguments: argparse.Namespace, **kwargs) -> str:
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.resolve_request(arguments, registry=_registry(), **kwargs)
    return str(caught.value)


def _admission_message(verdict: door.AdmissionVerdict) -> str:
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.require_admitted(verdict)
    return str(caught.value)


@pytest.mark.parametrize(
    "flag,attribute",
    [
        ("--mesh", "mesh"),
        ("--grid", "grid"),
        ("--static", "static"),
        ("--init", "init"),
        ("--init-source", "init_source"),
        ("--hours", "hours"),
        ("--history-every-minutes", "history_every_minutes"),
        ("--out", "out"),
    ],
)
def test_a_missing_required_flag_is_refused_by_name(
    tmp_path: Path, flag: str, attribute: str
) -> None:
    message = _refusal(_namespace(tmp_path, **{attribute: None}))
    assert flag in message
    assert len(message) > 80, "a refusal states the breakage, not just the flag"


def test_an_unregistered_mesh_names_the_registered_ones(tmp_path: Path) -> None:
    message = _refusal(_namespace(tmp_path, mesh="x9.999999"))
    assert "x9.999999" in message
    for name in _registry():
        assert name in message


def test_a_missing_mesh_file_is_refused_before_anything_expensive(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent.grid.nc"
    message = _refusal(_namespace(tmp_path, grid=missing))
    assert str(missing) in message
    assert "--grid" in message


def test_a_missing_gpuwm_checkout_names_the_pin_it_satisfies(
    tmp_path: Path,
) -> None:
    """The refusal must name the breakage that is LIVE, not the one that closed.

    This gate's own table was the stale side and moved 2026-08-28.  It used to
    demand the words "source checkout" and "sixteen", which the refusal
    supplied by saying one of the sixteen pinned paths reaches no wheel.  At
    engine 2.5.8 that is false -- all sixteen resolve from site-packages,
    measured against the published wheels -- so a refusal giving that reason
    names a breakage that no longer exists, which the refusal law forbids.
    What survives is git provenance, so that is what is demanded here.
    """

    message = _refusal(_namespace(tmp_path, gpuwm_checkout=tmp_path / "nope"))
    assert "--gpuwm-checkout" in message
    assert "git" in message.lower(), (
        "the requirement is a git working tree; naming only 'a checkout' "
        "leaves a user free to unpack a tarball and meet exit status 128"
    )
    assert "sixteen" in message, "the refusal names what the checkout satisfies"
    assert "named by commit" in message, (
        "the concrete breakage: an install carries the pinned bytes and no "
        "commit, so its receipt could provenance nothing"
    )
    assert "no wheel" not in message, "the retired reason must not come back"


def test_an_existing_output_directory_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "already"
    out.mkdir()
    message = _refusal(_namespace(tmp_path, out=out))
    assert str(out) in message
    assert "exists" in message


def test_an_output_whose_parent_is_absent_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "no" / "such" / "parent" / "out"
    message = _refusal(_namespace(tmp_path, out=out))
    assert str(out.parent) in message


def test_scratch_inside_the_output_tree_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    message = _refusal(_namespace(tmp_path, out=out, scratch=out / "inside"))
    assert "--scratch" in message
    assert str(out) in message


def test_the_default_scratch_is_a_sibling_of_the_output(tmp_path: Path) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    assert request.scratch.parent == request.out.parent
    assert request.scratch != request.out
    assert request.out not in request.scratch.parents


def test_a_forecast_length_that_is_not_whole_steps_is_refused(
    tmp_path: Path,
) -> None:
    message = _refusal(_namespace(tmp_path, hours=0.51))
    assert "--hours" in message
    assert "120" in message, "the refusal names the mesh's admitted timestep"


def test_a_history_cadence_that_does_not_divide_the_run_is_refused(
    tmp_path: Path,
) -> None:
    message = _refusal(_namespace(tmp_path, hours=1.0, history_every_minutes=45))
    assert "--history-every-minutes" in message


def test_the_generated_mesh_row_validates_the_schedule_at_its_own_timestep(
    tmp_path: Path, monkeypatch
) -> None:
    # v15.150.38857 declares 60 s, not x4's 120 s; a schedule check that used a
    # single module constant would admit a half-step run on this row.
    #
    # 60 s holds no timestep anchor, so the door refuses this row outright
    # (see the dt-admission test below).  The property under test here is the
    # SCHEDULE arithmetic, so this test registers a 60 s anchor the way a
    # ruling would -- one row -- and then checks the schedule.  That the whole
    # difference between "refused" and "runs" is a table row is itself the
    # point of the registry.
    _register_timestep_anchor(monkeypatch, 60.0)
    arguments = _namespace(
        tmp_path, mesh="v15.150.38857", hours=0.05, history_every_minutes=1
    )
    request = door.resolve_request(arguments, registry=_registry())
    assert request.dt_seconds == 60.0
    assert request.steps == 3


def test_command_line_problems_are_refused_before_filesystem_ones(
    tmp_path: Path,
) -> None:
    """A property of the request outranks a property of the disk.

    The order this pins replaced one where --init's existence was checked
    first: a user with a mistyped --hours and a not-yet-built init was told
    about the init, built it, and only then learned the schedule was wrong.
    """

    message = _refusal(
        _namespace(tmp_path, hours=0.51, init=tmp_path / "absent.init.nc")
    )
    assert "--hours" in message
    assert "--init" not in message


def test_a_half_model_row_outranks_a_missing_file_too(tmp_path: Path) -> None:
    message = _refusal(
        _namespace(
            tmp_path, device_fixed_mib=4000.0, init=tmp_path / "absent.init.nc"
        )
    )
    assert "--device-bytes-per-cell" in message


def test_a_local_timestep_ladder_that_does_not_start_at_one_is_refused(
    tmp_path: Path,
) -> None:
    message = _refusal(
        _namespace(tmp_path, local_timestep=True, local_timestep_rates="2,3")
    )
    assert "--local-timestep-rates" in message


def test_half_a_measured_footprint_row_is_refused(tmp_path: Path) -> None:
    message = _refusal(_namespace(tmp_path, device_fixed_mib=4000.0))
    assert "--device-bytes-per-cell" in message
    assert "--device-fixed-mib" in message


# ---------------------------------------------------------------------------
# admission: the measured model, decided against memory measured NOW
# ---------------------------------------------------------------------------
def test_the_shipped_model_is_the_measured_row() -> None:
    # The merged-tip row of record: the deeper pin against the raw evidence
    # ledgers lives in test_device_admission.py.  The row is SHAPED as of
    # 2026-08-27 -- a card core plus tiled physics workspaces plus a per-cell
    # term -- so what must reproduce the measured peaks is predict_bytes, not
    # any one coefficient.
    assert not hasattr(door.FOOTPRINT_MODEL, "fixed_bytes"), (
        "an affine fixed term must not survive as an attribute: reading one "
        "off the shaped row is how a caller silently prices a mesh without "
        "the workspace terms"
    )
    # x1.40962 at this row is the 8,874 MiB measured peak, within rounding.
    predicted = door.FOOTPRINT_MODEL.predict_bytes(X1_CELLS) / MIB
    assert predicted == pytest.approx(8874.0, abs=1.0)
    # x4.163842 is the row's other fitted point: the 20,446 MiB peak.
    assert door.FOOTPRINT_MODEL.predict_bytes(X4_CELLS) / MIB == pytest.approx(
        20446.0, abs=1.0
    )


def test_a_twelve_gib_card_admits_the_published_global_mesh() -> None:
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=int(11.2 * GIB),
        total_bytes=12 * GIB,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert verdict.admitted is True
    assert verdict.shortfall_bytes == 0


def test_a_ten_gib_card_in_use_refuses_and_names_the_shortfall() -> None:
    free = 7374 * MIB
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=free,
        total_bytes=10240 * MIB,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert verdict.admitted is False
    assert verdict.shortfall_bytes > 0
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.require_admitted(verdict)
    message = str(caught.value)
    assert "x1.40962" in message
    # The published row (5,016.5 MiB + 98,748 B/cell) reproduces the
    # measured 8,874 MiB x1 peak to publication rounding: 8,874.0 MiB.
    assert "8,874" in message or "8874" in message
    assert "MiB" in message
    # The remedy is not "try again": it is a fitted alternative, by name.
    assert "fits" in message or "fitted" in message
    assert "hex_ledger_probe" in message, (
        "the remedy for a card whose fixed term is smaller is to MEASURE it"
    )


def test_the_refusal_names_a_smaller_registered_mesh_when_one_fits() -> None:
    # A budget that holds the 38,857-cell row but not the 40,962-cell one.
    fitted = door.FOOTPRINT_MODEL.predict_bytes(V15_CELLS)
    free = int(fitted + door.DEFAULT_HEADROOM_BYTES + 8 * MIB)
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=free,
        total_bytes=12 * GIB,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert verdict.admitted is False
    assert "v15.150.38857" in verdict.alternatives
    assert "x4.163842" not in verdict.alternatives
    assert "v15.150.38857" in _admission_message(verdict)


def test_when_no_registered_mesh_fits_the_refusal_says_so_with_the_number() -> None:
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=2 * GIB,
        total_bytes=4 * GIB,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert verdict.alternatives == ()
    message = _admission_message(verdict)
    assert "no registered mesh" in message
    # The refusal must name THIS card's core, not "the fixed term": a card
    # is refused by its own arithmetic now, and a sentence that quotes a
    # shared constant is how a 32 GiB card's number ends up explaining a
    # 10 GiB card's refusal (ledger #366).
    assert "this card's own core" in message


def test_a_supplied_measured_row_replaces_the_shipped_one() -> None:
    model = door.ShapedFootprintModel(
        core_bytes=3000.0 * MIB,
        bytes_per_cell=93_474,
        card=door.CardProfile("a 68 SM part", 68),
        configuration="global",
        provenance="measured on this card",
    )
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=9500 * MIB,
        total_bytes=10240 * MIB,
        headroom_bytes=None,
        registry=_registry(),
        model=model,
    )
    assert verdict.admitted is True
    assert verdict.model_provenance == "measured on this card"


def test_the_headroom_is_part_of_the_decision() -> None:
    exact = int(round(door.FOOTPRINT_MODEL.predict_bytes(X1_CELLS)))
    tight = door.admission_verdict(
        mesh="x1.40962", cells=X1_CELLS, free_bytes=exact,
        total_bytes=12 * GIB, headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert tight.admitted is False
    zero = door.admission_verdict(
        mesh="x1.40962", cells=X1_CELLS, free_bytes=exact,
        total_bytes=12 * GIB, headroom_bytes=0, registry=_registry(),
    )
    assert zero.admitted is True


def test_admission_measures_the_card_at_decision_time(monkeypatch) -> None:
    """Never a frozen budget: the free-memory read happens per decision."""

    calls: list[int] = []

    def _measure() -> tuple[int, int]:
        calls.append(1)
        return 11 * GIB, 12 * GIB

    monkeypatch.setattr(door, "measure_device_memory", _measure)
    first = door.admit_device(
        mesh="x1.40962", cells=X1_CELLS,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES, registry=_registry(),
    )
    second = door.admit_device(
        mesh="x1.40962", cells=X1_CELLS,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES, registry=_registry(),
    )
    assert len(calls) == 2
    assert first.admitted and second.admitted


def test_a_card_that_cannot_be_measured_refuses_rather_than_guesses(
    monkeypatch,
) -> None:
    def _measure() -> tuple[int, int]:
        raise door.ForecastDoorRefusal(
            "cupy is not importable, so free device memory cannot be measured"
        )

    monkeypatch.setattr(door, "measure_device_memory", _measure)
    with pytest.raises(door.ForecastDoorRefusal):
        door.admit_device(
            mesh="x1.40962", cells=X1_CELLS,
            headroom_bytes=door.DEFAULT_HEADROOM_BYTES, registry=_registry(),
        )


# ---------------------------------------------------------------------------
# the run receipt, and the hand-off to the render door
# ---------------------------------------------------------------------------
def test_the_receipt_carries_what_ran_and_serialises(tmp_path: Path) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    verdict = door.admission_verdict(
        mesh="x1.40962", cells=X1_CELLS, free_bytes=11 * GIB,
        total_bytes=12 * GIB, headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    receipt = door.build_receipt(
        request=request,
        admission=verdict,
        bind_receipt={"mesh": "x1.40962", "rebound": True},
        driver_receipt={"status": "passed"},
        history=[tmp_path / "out" / "cuda-history.2026-08-24_01.00.00.nc"],
        driver_argv=["--hours", "1.0"],
        seconds=12.5,
        status="passed",
    )
    assert receipt["schema"] == door.RECEIPT_SCHEMA
    assert receipt["status"] == "passed"
    assert receipt["admission"]["admitted"] is True
    assert receipt["mesh"]["name"] == "x1.40962"
    assert receipt["render_command"][0] == "gpuwm-hex"
    assert receipt["render_command"][1] == "render"
    assert receipt["history"]
    json.dumps(receipt, sort_keys=True, allow_nan=False)


def test_the_render_command_names_the_files_the_render_door_takes(
    tmp_path: Path,
) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    history = [tmp_path / "out" / "cuda-history.2026-08-24_01.00.00.nc"]
    command = door.render_command(request, history)
    assert "--history" in command
    assert str(history[0]) in command
    assert "--mesh" in command
    assert str(request.grid) in command
    assert "--out" in command


def test_the_receipt_records_a_refusal_without_claiming_a_forecast(
    tmp_path: Path,
) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    verdict = door.admission_verdict(
        mesh="x1.40962", cells=X1_CELLS, free_bytes=2 * GIB,
        total_bytes=4 * GIB, headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    receipt = door.build_receipt(
        request=request, admission=verdict, bind_receipt=None,
        driver_receipt=None, history=[], driver_argv=[], seconds=0.4,
        status="refused_by_admission",
    )
    assert receipt["status"] == "refused_by_admission"
    assert receipt["admission"]["admitted"] is False
    assert receipt["history"] == []
    assert receipt["render_command"] is None


# ---------------------------------------------------------------------------
# what the door hands the driver
# ---------------------------------------------------------------------------
def test_the_driver_vector_carries_the_run_and_maps_the_checkout_flag(
    tmp_path: Path,
) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    argv = door.build_driver_argv(request)
    assert "--cache-root" in argv
    assert argv[argv.index("--cache-root") + 1] == str(request.scratch)
    assert argv[argv.index("--output") + 1] == str(request.out)
    # The door's spelling is the brand's; the driver's is the engine's.
    assert argv[argv.index("--arwen-checkout") + 1] == str(request.gpuwm_checkout)
    assert "--preflight-only" not in argv
    assert "--local-timestep" not in argv


def test_the_opt_in_lane_reaches_the_driver_only_when_asked(tmp_path: Path) -> None:
    arguments = _namespace(
        tmp_path, local_timestep=True, local_timestep_rates="1,3,6"
    )
    request = door.resolve_request(arguments, registry=_registry())
    argv = door.build_driver_argv(request)
    assert "--local-timestep" in argv
    assert argv[argv.index("--local-timestep-rates") + 1] == "1,3,6"


def test_the_door_leaves_destination_creation_to_the_driver(
    tmp_path: Path, monkeypatch
) -> None:
    """The driver's validate_destination requires --out and --scratch ABSENT
    and creates both itself.  A door that pre-creates them kills every
    admitted run with FileExistsError immediately after admission -- measured
    on the RTX 3080, 2026-08-24, where the door's success leg had never run.
    This test drives run_forecast with a driver stub enforcing the real
    driver's absence contract; it fails if the door ever creates either root
    before the hand-off.
    """

    seen: dict[str, bool] = {}

    class _Driver:
        @staticmethod
        def main(argv: list[str]) -> int:
            out = Path(argv[argv.index("--output") + 1])
            cache = Path(argv[argv.index("--cache-root") + 1])
            # the real driver's contract, verbatim in spirit:
            for label, path in (("output root", out), ("cache root", cache)):
                if path.exists():
                    raise FileExistsError(f"{label} must be absent: {path}")
            out.mkdir(parents=False)
            cache.mkdir(parents=False)
            seen["created_by_driver"] = True
            return 0

    monkeypatch.setattr(door, "load_registry", lambda repo: _registry())
    monkeypatch.setattr(
        door, "_load_drivers", lambda repo: (object(), _Driver())
    )
    monkeypatch.setattr(
        door, "_bind", lambda binding, driver, request: {"rebound": True}
    )
    monkeypatch.setattr(
        door, "measure_device_memory", lambda: (11 * GIB, 12 * GIB)
    )
    arguments = _namespace(tmp_path)
    rc = door.run_forecast(arguments)
    assert seen.get("created_by_driver") is True
    # rc 1 is the door's own "driver exited 0 but wrote no history frame"
    # verdict from the stub; the point of this test is that no
    # ForecastDoorRefusal wrapping FileExistsError was raised above.
    assert rc == 1


def test_preflight_asks_the_driver_for_a_preflight_and_no_destination(
    tmp_path: Path,
) -> None:
    request = door.resolve_request(
        _namespace(tmp_path, preflight=True), registry=_registry()
    )
    argv = door.build_driver_argv(request)
    assert "--preflight-only" in argv
    assert "--cache-root" not in argv
    assert "--output" not in argv


def test_preflight_collects_missing_inputs_instead_of_stopping_at_the_first(
    tmp_path: Path,
) -> None:
    # A user asking "will this run on my card?" has often not built the init
    # yet.  If a missing init stopped the answer, the card question -- the
    # one no file fixes -- could not be asked until every file existed.
    arguments = _namespace(
        tmp_path,
        preflight=True,
        init=tmp_path / "absent.init.nc",
        gpuwm_checkout=None,
    )
    request = door.resolve_request(arguments, registry=_registry())
    assert request.inputs_present is False
    joined = " | ".join(request.input_problems)
    assert "--init" in joined
    assert "--gpuwm-checkout" in joined
    # and the request is still complete enough to decide the card question
    assert request.cells == X1_CELLS


def test_a_real_run_refuses_the_same_missing_input(tmp_path: Path) -> None:
    message = _refusal(_namespace(tmp_path, init=tmp_path / "absent.init.nc"))
    assert "--init" in message


def test_preflight_does_not_demand_a_fresh_output_directory(tmp_path: Path) -> None:
    # Preflight writes nothing into --out, so refusing an existing one would
    # make the answer "will this run?" unaskable a second time.
    out = tmp_path / "out"
    out.mkdir()
    request = door.resolve_request(
        _namespace(tmp_path, out=out, preflight=True), registry=_registry()
    )
    assert request.preflight is True
    assert request.out == out


# ---------------------------------------------------------------------------
# the registry the door resolves through is the checkout's own
# ---------------------------------------------------------------------------
def test_the_registry_comes_from_the_mesh_binding_module() -> None:
    repo = door.PROJECT_ROOT
    if repo is None:  # pragma: no cover - only outside a checkout
        pytest.skip("not a source checkout; the registry lives in tools/")
    registry = door.load_registry(repo)
    # The oracle is the binding module itself, not a restated list: a frozen
    # set here would refuse every mesh the registry legitimately gains.
    binding = door._load_module(
        "mpas_mesh_binding_oracle", Path(repo) / "tools" / "mpas_mesh_binding.py"
    )
    assert set(registry) == set(binding.MESH_BINDINGS)
    for name, row in binding.MESH_BINDINGS.items():
        assert registry[name].cells == int(row.n_cells)
        assert registry[name].dt_seconds == float(row.dt_seconds)
    assert registry["x1.40962"].cells == X1_CELLS


def test_a_repo_without_tools_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.resolve_repo(tmp_path)
    message = str(caught.value)
    assert str(tmp_path) in message
    assert "checkout" in message


def test_a_wheel_install_honors_the_run_from_inside_gesture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal and quickstart 2.6 both promise two roads: pass
    ``--repo``, OR run the door from inside a checkout.  From an installed
    wheel ``PROJECT_ROOT`` is ``None`` (the package's own ancestry is
    site-packages), so the second road must resolve through the working
    directory or the sentence is a promise the door does not keep.  The
    0.1.1 fresh-install walk met exactly that: ``cd <checkout>/tree`` and
    the door still refused, with the init door happily working beside it.
    """

    repo = door.PROJECT_ROOT
    if repo is None:  # pragma: no cover - only outside a checkout
        pytest.skip("not a source checkout; nothing to stand inside")
    monkeypatch.setattr(door, "PROJECT_ROOT", None)  # the wheel shape

    monkeypatch.chdir(repo)  # inside tree/ itself
    assert door.resolve_repo(None) == repo

    monkeypatch.chdir(repo / "tools")  # deeper inside the checkout
    assert door.resolve_repo(None) == repo

    monkeypatch.chdir(repo.parent)  # the repo root above tree/
    assert door.resolve_repo(None) == repo


# ---------------------------------------------------------------------------
# the architecture half of admission
# ---------------------------------------------------------------------------


def test_architecture_at_the_proven_floor_is_admitted(monkeypatch):
    monkeypatch.setattr(door, "read_device_compute", lambda: (12, 0))
    verdict = door.admit_architecture()
    assert verdict["admitted"] is True
    assert verdict["sm"] == "sm_120"
    assert "proven contract floor" in verdict["basis"]


def test_unanchored_below_floor_architecture_is_refused_by_name(monkeypatch):
    from hexcore.cuda_backend import arch_admission

    monkeypatch.setattr(arch_admission, "ADMITTED_BELOW_FLOOR", {})
    monkeypatch.setattr(door, "read_device_compute", lambda: (8, 6))
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.admit_architecture()
    message = str(caught.value)
    assert "sm_86" in message
    assert "no per-architecture anchor" in message


def test_anchored_architecture_is_admitted_with_its_receipt_named(monkeypatch):
    from hexcore.cuda_backend import arch_admission

    anchor = arch_admission.ArchAnchor(
        compute=(8, 6),
        card="RTX 3080 (test double)",
        admitted_on="2026-08-25",
        contract_receipt="evidence/sm86-tier-20260825/RECEIPT.md",
        authority_anchor="evidence/sm86-tier-20260825/authority",
        basis="test-registered anchor",
    )
    monkeypatch.setattr(
        arch_admission, "ADMITTED_BELOW_FLOOR", {(8, 6): anchor}
    )
    monkeypatch.setattr(door, "read_device_compute", lambda: (8, 6))
    verdict = door.admit_architecture()
    assert verdict["admitted"] is True
    assert verdict["sm"] == "sm_86"
    assert "sm86-tier-20260825" in verdict["basis"]


def test_preflight_reports_the_architecture_refusal(monkeypatch, tmp_path):
    """Rung 4 of the small-card walk, closed: preflight and the run may no
    longer disagree about the architecture.  An unanchored card's preflight
    now carries the same named refusal the run would raise."""

    from hexcore.cuda_backend import arch_admission

    monkeypatch.setattr(arch_admission, "ADMITTED_BELOW_FLOOR", {})
    monkeypatch.setattr(door, "read_device_compute", lambda: (7, 5))
    monkeypatch.setattr(
        door, "measure_device_memory", lambda: (11 * GIB, 12 * GIB)
    )
    arguments = _namespace(
        tmp_path, preflight=True, receipt=tmp_path / "receipt.json"
    )
    registry = _registry()
    request = door.resolve_request(arguments, registry=registry)
    rc = door._run_preflight(request, registry, started=0.0)
    assert rc == 1
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    problems = receipt["preflight_problems"]
    assert any("sm_75" in problem for problem in problems)
    assert any(
        "no per-architecture anchor" in problem for problem in problems
    )


# ---------------------------------------------------------------------------
# the physics backend row and the point-source table
# ---------------------------------------------------------------------------
def test_help_names_the_backend_row_and_the_source_table() -> None:
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    text = parser.format_help()
    for flag in ("--physics-backend", "--source-table"):
        assert flag in text, flag


def test_an_unregistered_physics_backend_refuses_naming_the_rows(tmp_path):
    message = _refusal(_namespace(tmp_path, physics_backend="nope"))
    assert "--physics-backend nope" in message
    assert "wsm6_column" in message


def test_a_source_table_on_the_frozen_row_refuses_as_never_read(tmp_path):
    table = tmp_path / "sources.txt"
    table.write_text("# epoch lat lon alt on rate src\n")
    message = _refusal(_namespace(tmp_path, source_table=table))
    assert "carries no point source" in message
    assert "never read" in message


def _provider_row_name() -> str:
    """A PROVIDER row's name, read off the registry rather than spelled here.

    This file ships in the public tree and the provider does not, so the
    provider's own vocabulary must not be a byte of this file: the row is
    found through the module that owns it (its P3 row, the one these tests
    always exercised).  Skips by name where no provider is installed,
    exactly as the tests that use it did already.
    """

    pytest.importorskip("hexcore.mod")
    from hexcore import physics_backend_admission as admission

    names = sorted(
        name
        for name, row in admission.backend_rows().items()
        if row.adapter_module.startswith("hexcore.mod")
    )
    assert names, "the provider is importable but registered no row"
    p3 = [name for name in names if name.endswith("_p3")]
    return (p3 or names)[0]


def test_a_provider_row_is_pinned_by_its_own_adapter_at_the_door(tmp_path):
    """The frozen manifest can never verify a provider's engine, so the
    door asks the row's adapter; the refusal is the second manifest's
    own text, at the door, before a card is touched."""
    provider = _provider_row_name()
    message = _refusal(_namespace(tmp_path, physics_backend=provider))
    assert "pinned sibling source is missing" in message
    assert "re-pin" in message or "REMEDY" in message


def test_preflight_carries_the_row_and_the_table_into_the_driver_argv(tmp_path):
    provider = _provider_row_name()
    table = tmp_path / "sources.txt"
    table.write_text("# epoch lat lon alt on rate src\n")
    arguments = _namespace(
        tmp_path, physics_backend=provider, source_table=table, preflight=True
    )
    request = door.resolve_request(arguments, registry=_registry())
    assert request.physics_backend == provider
    assert request.source_table == table.absolute()
    # The pin problem is COLLECTED in preflight, and it is the row's own.
    assert any("pinned sibling source is missing" in p for p in request.input_problems)
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--physics-backend") + 1] == provider
    assert argv[argv.index("--source-table") + 1] == str(table.absolute())


def test_a_default_run_carries_no_backend_token(tmp_path):
    """Byte-stable: the frozen row's argv and receipt shape do not move."""
    request = door.resolve_request(
        _namespace(tmp_path, preflight=True), registry=_registry()
    )
    assert request.physics_backend == "wsm6_column"
    assert request.source_table is None
    argv = door.build_driver_argv(request)
    assert "--physics-backend" not in argv
    assert "--source-table" not in argv
