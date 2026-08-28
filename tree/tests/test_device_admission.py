"""The one admission surface: its shape, its evidence, and door/driver agreement.

Three families of pin live here.

**The shape is the evidence.**  ``FOOTPRINT_MODEL`` is no longer a line in
cell count.  It is a card-scaled core, plus the Grell-Freitas and YSU column
workspaces charged at ``min(cells, tile(card))``, plus a per-cell term.  Every
part of that claim is checked against the ARTIFACT: the raw #264 ledger JSONs
in ``evidence/memory-row-refit-20260826/node2/`` carry both the per-process
peaks the row is fitted from AND the measured size of each workspace block, so
the tiled arithmetic is re-derived here rather than asserted.  A constant that
drifts from the evidence it cites -- which is exactly how the door shipped a
superseded row for a day -- fails by name.

**Agreement to the cell.**  The forecast door's ``--preflight`` verdict and
the driver's own floor must be ONE number: a byte of divergence is a card
that passes one gate, spends the mesh bind and the kernel compile, and dies
on the other.  The tests assert the shared sum end to end: the re-derived
``NATIVE_DEVICE_FLOOR``, the per-mesh floor the binding installs, the door
verdict, and the ``--required-free-bytes`` value the door forwards into the
driver's argv.

**The retired arms stay computable and never become gates.**  Three rows have
been retired now -- the asserted 24 GiB proxy, the 2026-08-25 converged row,
and the 2026-08-26 affine row of record -- and each is re-fitted here from its
own raw ledgers so a before/after table is arithmetic on named rows rather
than two hand-typed columns.  The sweep tests then refuse any shipped surface
that quotes a retired coefficient, or the retired flat headroom, as the
requirement a run is admitted on.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from hexcore import device_admission as surface
from hexcore import forecast_door as door

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EVIDENCE = ROOT / "evidence" / "memory-row-refit-20260826" / "node2"
REFIT_RECORD = ROOT / "evidence" / "memory-row-refit-20260826" / "REFIT.json"
PRIOR_EVIDENCE = ROOT / "evidence" / "pin-move-335-20260825" / "node2"
REDERIVE_EVIDENCE = ROOT / "evidence" / "device-floor-rederive-20260826" / "node2"
SWATHS_EVIDENCE = ROOT / "evidence" / "four-swaths-20260827"

MIB = 1024**2
RETIRED_LINEAR_FLOOR = 24 * 1024**3

#: The two published meshes the row of record is fitted from.  The cell counts
#: are read out of the ledgers themselves below; these names only locate the
#: files.
X1_TAG = "x1"
X4_TAG = "x4"


def _load_tool(name: str, filename: str) -> object:
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def binding_mod() -> object:
    return _load_tool("mesh_binding_admission", "mpas_mesh_binding.py")


# ---------------------------------------------------------------------------
# the raw evidence, read rather than restated
# ---------------------------------------------------------------------------
def _ledger(tag: str, evidence: Path | None = None) -> dict:
    root = EVIDENCE if evidence is None else evidence
    path = root / f"gateB-{tag}" / f"ledger-{tag}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_cells(tag: str, evidence: Path | None = None) -> int:
    return int(_ledger(tag, evidence)["n_cells"])


def _ledger_peak_mib(tag: str, evidence: Path | None = None) -> float:
    """The of-record peak: this process's nvidia-smi row, from the raw ledger."""

    ledger = _ledger(tag, evidence)
    peaks = [phase.get("nvsmi_process_mib") or 0.0 for phase in ledger["phases"]]
    assert peaks and max(peaks) > 0.0, (
        f"the {tag} ledger carries no per-process nvidia-smi rows; the "
        "of-record peak convention cannot be applied and the row cannot be "
        "re-derived"
    )
    return float(max(peaks))


def _ledger_block_bytes(tag: str, path_tail: str, evidence: Path | None = None) -> int:
    """The largest single allocation the #264 ledger recorded at one source file.

    The ledger keys sites by absolute path on the measuring box, so the match
    is on the trailing package-relative segment.  Taking the largest
    ``max_single_bytes`` picks the workspace block out of the small allocations
    the same file also makes.
    """

    ledger = _ledger(tag, evidence)
    sites = [site for site in ledger["sites"] if path_tail in str(site["site"])]
    assert sites, (
        f"the {tag} ledger records no allocation site under {path_tail!r}; the "
        "tiled-workspace arithmetic cannot be checked against measurement"
    )
    return int(max(int(site["max_single_bytes"]) for site in sites))


def _affine_two_point_fit(
    x1_cells: int, x1_peak_mib: float, x4_cells: int, x4_peak_mib: float
) -> tuple[float, float]:
    """``(fixed_mib, bytes_per_cell)`` -- the RETIRED shape, fitted straight."""

    slope = (x4_peak_mib - x1_peak_mib) * MIB / (x4_cells - x1_cells)
    return x1_peak_mib - slope * x1_cells / MIB, slope


def _refit_record() -> dict:
    return json.loads(REFIT_RECORD.read_text(encoding="utf-8"))


def _desktop_3080_record() -> dict:
    """The 10 GiB desktop's row in the re-fit record -- ledger #366's card."""

    cards = _refit_record()["cards"]
    matches = [row for name, row in cards.items() if "3080" in name]
    assert len(matches) == 1, (
        "the re-fit record no longer carries exactly one RTX 3080 row; ledger "
        "#366's before/after cannot be computed from the evidence"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# the shipped row IS the evidence it cites
# ---------------------------------------------------------------------------
def test_the_shipped_row_reproduces_the_raw_merged_tip_peaks(receipts) -> None:
    """THE BREAKAGE: a coefficient that drifts from the ledger it names.

    The row claims to reproduce the two measured peaks in the #264 session it
    cites.  Read both peaks out of the raw ledgers and require the model to
    land on them, so the claim is checkable rather than asserted.
    """

    for tag in (X1_TAG, X4_TAG):
        cells = _ledger_cells(tag)
        predicted_mib = surface.FOOTPRINT_MODEL.predict_bytes(cells) / MIB
        assert predicted_mib == pytest.approx(_ledger_peak_mib(tag), abs=0.2), (
            f"the shipped row no longer reproduces the measured {tag} peak; "
            "either the coefficients moved without the evidence, or the "
            "evidence moved without the coefficients"
        )


def test_the_shipped_rows_terms_are_the_fit_of_the_ledgers_with_the_tiles_removed(
    receipts,
) -> None:
    """The fit is checkable, not asserted, and the tiles come off FIRST.

    Subtract each mesh's tiled-workspace bytes from its measured peak and fit
    a line through what is left; that line is the core and the per-cell slope.
    THE BREAKAGE this prevents: fitting the peaks straight charges the
    card-sized workspace growth between a mesh below both tile knees and a
    mesh above both as if it were per-cell, which is precisely the defect that
    retired the affine row.
    """

    model = surface.FOOTPRINT_MODEL
    x1_cells, x4_cells = _ledger_cells(X1_TAG), _ledger_cells(X4_TAG)
    x1_residue = _ledger_peak_mib(X1_TAG) * MIB - model.tiled_bytes(x1_cells)
    x4_residue = _ledger_peak_mib(X4_TAG) * MIB - model.tiled_bytes(x4_cells)

    slope = (x4_residue - x1_residue) / (x4_cells - x1_cells)
    core = x1_residue - slope * x1_cells

    assert model.bytes_per_cell == pytest.approx(slope, abs=2.0)
    assert model.core_bytes / MIB == pytest.approx(core / MIB, abs=0.2)


def test_the_shipped_row_names_its_evidence_and_pin() -> None:
    """A row with no pin on it outlives the tree it was measured on."""

    provenance = surface.FOOTPRINT_MODEL.provenance
    assert "memory-row-refit-20260826" in provenance
    assert "26daaab7e" in provenance, "the row must name the engine pin it was measured at"
    assert "2009db7" in provenance, "the row must name the hex tree it was measured at"


# ---------------------------------------------------------------------------
# the tiled workspaces: the term the affine row did not have
# ---------------------------------------------------------------------------
def test_the_tiled_workspace_arithmetic_is_the_ledgers_own_block_sizes(
    receipts,
) -> None:
    """THE BREAKAGE: a workspace term invented to make a fit work.

    The tiled terms are the whole difference between this model and the
    retired line, so they must be the blocks the instrument actually recorded,
    not a residual named after a kernel.  The #264 ledger stores each site's
    largest single allocation at both meshes; the module's formulas are card
    constants only and must land on those numbers.
    """

    pairs = (
        (surface.GRELL_FREITAS_WORKSPACE, "core/gf.py:"),
        (surface.YSU_WORKSPACE, "core/ysu.py:"),
    )
    for workspace, path_tail in pairs:
        for tag in (X1_TAG, X4_TAG):
            cells = _ledger_cells(tag)
            measured = _ledger_block_bytes(tag, path_tail)
            computed = workspace.bytes_for(cells, surface.REFERENCE_CARD)
            assert computed / MIB == pytest.approx(measured / MIB, abs=0.05), (
                f"{workspace.name} at {tag}: the module computes "
                f"{computed} B against the ledger's measured {measured} B"
            )


def test_the_tiled_terms_do_not_scale_with_the_mesh(receipts) -> None:
    """The point of the whole shape, stated as a number.

    Every other site in the #264 ledger scales 4.00x with a 4x cell count.
    These two do not, because both meshes straddle the tile knees, and that is
    why a line through the two peaks is the wrong function.  THE BREAKAGE: a
    future edit that quietly makes these terms proportional to cells restores
    the retired shape while keeping the new names.
    """

    x1_cells, x4_cells = _ledger_cells(X1_TAG), _ledger_cells(X4_TAG)
    assert x4_cells == 4 * x1_cells - 6, (
        "the two published meshes are no longer the 4x pair these ratios were "
        "read against"
    )

    gf_ratio = _ledger_block_bytes(X4_TAG, "core/gf.py:") / _ledger_block_bytes(
        X1_TAG, "core/gf.py:"
    )
    ysu_ratio = _ledger_block_bytes(X4_TAG, "core/ysu.py:") / _ledger_block_bytes(
        X1_TAG, "core/ysu.py:"
    )
    assert gf_ratio == pytest.approx(1.06, abs=0.01)
    assert ysu_ratio == pytest.approx(2.12, abs=0.01)
    assert gf_ratio < 4.0 and ysu_ratio < 4.0

    # and the module reproduces both ratios from card constants alone
    card = surface.REFERENCE_CARD
    for workspace, measured_ratio in (
        (surface.GRELL_FREITAS_WORKSPACE, gf_ratio),
        (surface.YSU_WORKSPACE, ysu_ratio),
    ):
        computed_ratio = workspace.bytes_for(x4_cells, card) / workspace.bytes_for(
            x1_cells, card
        )
        assert computed_ratio == pytest.approx(measured_ratio, abs=0.01)


def test_the_shortwave_block_is_mesh_independent_and_is_the_block_the_ledger_names(
    receipts,
) -> None:
    """THE BREAKAGE: pricing the margin at a block that moves with the mesh.

    The margin's first component is the RRTMG shortwave workspace, and it is
    the margin precisely because it does NOT scale with the mesh -- so its
    arena placement can move the pool high-water with no input changing.  The
    ledger records the same block, to the byte, at both meshes; if that ever
    stops being true the margin is priced at the wrong thing.
    """

    x1_block = _ledger_block_bytes(X1_TAG, "core/rrtmg_sw.py:")
    x4_block = _ledger_block_bytes(X4_TAG, "core/rrtmg_sw.py:")
    assert x1_block == x4_block, (
        "the shortwave workspace now differs between the two meshes, so it is "
        "no longer the mesh-independent block the margin is priced at"
    )
    assert surface.shortwave_workspace_bytes(surface.REFERENCE_CARD) == x1_block


def test_the_radiation_chunk_narrows_on_a_smaller_card() -> None:
    """The card-scaled step in the model, and the reason a small card gets a
    smaller margin.  THE BREAKAGE: charging every card the 170 SM part's
    1,745.6 MiB radiation block refuses small cards that never allocate it."""

    assert surface.radiation_chunk_columns(surface.REFERENCE_CARD) == 2048
    for sms in (68, 70):
        card = surface.CardProfile(f"{sms} SM part", sms)
        assert surface.radiation_chunk_columns(card) == 1024, (
            f"a {sms} SM card no longer halves its radiation chunk, so its "
            "margin would be priced at a block it does not allocate"
        )
        assert surface.shortwave_workspace_bytes(card) < surface.shortwave_workspace_bytes(
            surface.REFERENCE_CARD
        )


# ---------------------------------------------------------------------------
# the margin: two named, measured components, not a flat constant
# ---------------------------------------------------------------------------
def test_the_margin_names_what_it_absorbs() -> None:
    """the gate law: a gate must name the concrete breakage it prevents.

    The retired 512 MiB headroom named nothing and failed by 96 MiB on a real
    run.  Every component of the replacement must say what it absorbs, what
    breaks without it, and what measured it -- and the total must be the sum
    of exactly those components, so nothing can be padded in unnamed.
    """

    model = surface.FOOTPRINT_MODEL
    terms = model.margin_terms()
    assert set(terms) == {"arena_placement", "instrument_convention"}
    for name, term in terms.items():
        for field in ("absorbs", "breakage", "measured"):
            assert isinstance(term[field], str) and term[field].strip(), (
                f"the {name} margin component does not say what it {field}; "
                "an unnamed margin is padding"
            )
        assert int(term["bytes"]) > 0, name

    assert terms["arena_placement"]["bytes"] == surface.shortwave_workspace_bytes(
        model.card, model.levels
    )
    assert terms["instrument_convention"]["bytes"] == surface.CONVENTION_MARGIN_BYTES
    assert model.margin_bytes() == sum(int(t["bytes"]) for t in terms.values())
    assert model.margin_bytes() != surface.RETIRED_FLAT_HEADROOM_BYTES, (
        "the margin is back to the retired flat constant, which named no "
        "breakage and was measured to be insufficient"
    )


def test_the_door_reexports_the_surface_unchanged() -> None:
    """One surface, one set of objects.  A door that re-exports a COPY is two
    admission models again, which is the breakage this module exists to end."""

    assert door.FOOTPRINT_MODEL is surface.FOOTPRINT_MODEL
    assert door.ShapedFootprintModel is surface.ShapedFootprintModel
    assert door.FootprintModel is surface.FootprintModel
    assert door.CardProfile is surface.CardProfile
    assert door.REFERENCE_CARD is surface.REFERENCE_CARD
    assert door.model_for_card is surface.model_for_card
    assert door.card_profile_from_attributes is surface.card_profile_from_attributes
    # The retired flat constant is still re-exported under its old name so an
    # existing import resolves -- but it is NOT what a gate holds back.
    assert door.DEFAULT_HEADROOM_BYTES == surface.RETIRED_FLAT_HEADROOM_BYTES
    assert door.FOOTPRINT_MODEL.margin_bytes() != door.DEFAULT_HEADROOM_BYTES


# ---------------------------------------------------------------------------
# ledger #366: a card gets its own row, and a card nobody measured gets
# arithmetic that says it is derived
# ---------------------------------------------------------------------------
def test_a_small_card_gets_its_own_row_not_the_reference_cards() -> None:
    """THE BREAKAGE, as it happened (#366): the door returned the 170 SM row
    whenever the user did not type two numbers, so a 10 GiB desktop was priced
    with a 32 GiB card's fixed term."""

    small = surface.model_for_card(surface.KNOWN_CARDS["10gib-68sm"])
    reference = surface.FOOTPRINT_MODEL
    assert small.card is not reference.card
    assert small.core_bytes < 0.5 * reference.core_bytes, (
        "the small card's core is no longer distinguishable from the "
        "reference card's, which is the #366 defect returning"
    )


def test_the_small_card_admits_the_meshes_its_own_measurement_ran(receipts) -> None:
    """Ledger #366 as a before/after, computed from both surfaces.

    The re-fit record names the two registered meshes the retired row refused
    on the desktop RTX 3080 and the free memory that card actually offered.
    The new model must admit both at that free figure; the retired affine row
    must refuse both, which is what the ledger recorded.
    """

    desktop = _desktop_3080_record()
    free_bytes = int(desktop["free_mib"] * MIB)
    flipped = desktop["registered_meshes_that_flip"]
    assert flipped, "the re-fit record no longer names the meshes #366 was about"

    model = surface.model_for_card(surface.KNOWN_CARDS["10gib-68sm"])
    for mesh, row in flipped.items():
        cells = int(row["cells"])
        assert model.required_bytes(cells) < free_bytes, (
            f"{mesh}: the new model still refuses a mesh this card was "
            "measured running with room to spare"
        )
        assert surface.retired_affine_row_floor_bytes(cells) > free_bytes, (
            f"{mesh}: the retired affine row no longer refuses this mesh, so "
            "the before arm of #366 has stopped being the before arm"
        )


def test_an_unmeasured_card_still_gets_arithmetic_and_says_it_is_derived() -> None:
    """A card this project has never run on must still get a row, and the row
    must not be quoted as hardware truth.  THE BREAKAGE both ways: no row at
    all means no gate, and an unlabelled row means a derivation is read as a
    measurement."""

    card = surface.CardProfile("made up", 46)
    model = surface.model_for_card(card)

    assert model.measured is False
    assert model.derived_from == surface.row_key("global", surface.REFERENCE_CARD)
    assert "DERIVED, NOT MEASURED" in model.provenance
    assert 0.0 < model.core_bytes < surface.FOOTPRINT_MODEL.core_bytes, (
        "a 46 SM card's derived core is not between zero and the 170 SM "
        "reference's, so the derivation is not scaling with the card"
    )


# ---------------------------------------------------------------------------
# the re-derived floor
# ---------------------------------------------------------------------------
def test_the_native_floor_is_the_measured_requirement_not_an_assertion(
    receipts,
) -> None:
    floor = surface.native_device_floor_bytes()
    assert floor == surface.required_free_bytes(surface.NATIVE_MESH_CELLS)
    # Above the measured peak (else the floor admits a run that dies inside
    # a CuPy allocation), below the retired 24 GiB assertion (else nothing
    # was re-derived).
    assert floor > _ledger_peak_mib(X4_TAG) * MIB
    assert floor < RETIRED_LINEAR_FLOOR
    # And the gap to the peak is exactly the model's own margin, by
    # construction: nothing is added to the sum that the margin does not name.
    predicted = int(round(surface.FOOTPRINT_MODEL.predict_bytes(surface.NATIVE_MESH_CELLS)))
    assert floor - predicted == surface.FOOTPRINT_MODEL.margin_bytes()


def test_the_retired_linear_floor_admitted_below_the_measured_x1_peak(receipts) -> None:
    """The breakage the re-derivation fixes, kept as a checked fact.

    The linear floor scaled 24 GiB through the origin, so at x1.40962 it
    admitted at ~6,144 MiB free while the same session measured an 8,874 MiB
    peak: an admission that dies mid-run.  The shaped floor sits above the
    measured peak at EVERY registered cell count.
    """

    x1_cells = _ledger_cells(X1_TAG)
    retired_x1 = surface.retired_linear_floor_bytes(x1_cells)
    measured_x1 = _ledger_peak_mib(X1_TAG) * MIB
    assert retired_x1 < measured_x1  # the defect
    assert surface.required_free_bytes(x1_cells) > measured_x1  # the fix


# ---------------------------------------------------------------------------
# one surface: binding floor == door verdict == driver argv, to the cell
# ---------------------------------------------------------------------------
def test_the_binding_floors_are_the_shared_sum(binding_mod) -> None:
    assert binding_mod.NATIVE_DEVICE_FLOOR == surface.native_device_floor_bytes()
    assert binding_mod.NATIVE_RESTART_FLOOR == binding_mod.NATIVE_DEVICE_FLOOR
    assert surface.NATIVE_MESH_CELLS == binding_mod.MESH_BINDINGS[
        binding_mod.NATIVE_MESH_NAME
    ].n_cells


def test_door_and_driver_agree_to_the_cell_on_every_registered_mesh(
    binding_mod,
) -> None:
    registry = {
        name: door.MeshRow(name, int(row.n_cells), float(row.dt_seconds))
        for name, row in binding_mod.MESH_BINDINGS.items()
    }
    for name, row in registry.items():
        driver_floor = surface.required_free_bytes(row.cells)
        verdict = door.admission_verdict(
            mesh=name,
            cells=row.cells,
            free_bytes=driver_floor,  # exactly at the line
            total_bytes=32 * 1024**3,
            # None is the default: the model's own margin, the same one
            # required_free_bytes used to build driver_floor above.
            headroom_bytes=None,
            registry=registry,
        )
        assert verdict.required_bytes == driver_floor, (
            f"{name}: the door requires {verdict.required_bytes} while the "
            f"driver floor is {driver_floor}; a card between them passes one "
            "gate and dies on the other"
        )
        assert verdict.admitted is True
        assert verdict.headroom_bytes == surface.FOOTPRINT_MODEL.margin_bytes()
        # one byte below the line, both gates refuse
        below = door.admission_verdict(
            mesh=name,
            cells=row.cells,
            free_bytes=driver_floor - 1,
            total_bytes=32 * 1024**3,
            headroom_bytes=None,
            registry=registry,
        )
        assert below.admitted is False


def test_agreement_is_to_the_cell_at_any_free_value() -> None:
    """max_cells is exact against the shared sum: the largest admitted cell
    count admits, and one more cell refuses.

    The model is piecewise linear now -- the tiled terms stop growing at their
    knees -- so this cannot be checked by dividing by a slope.  The free
    values below place the answer below both knees, between them, and above
    both, so a solver that assumed one slope fails here.
    """

    model = surface.FOOTPRINT_MODEL
    knees = [ws.knee_cells(model.card) for ws in surface.TILED_WORKSPACES]
    fitted_counts = []
    for free in (8 * 1024**3, 10_066_521_290, 12 * 1024**3, 24 * 1024**3):
        fitted = model.max_cells(free)
        fitted_counts.append(fitted)
        assert model.required_bytes(fitted) <= free
        assert model.required_bytes(fitted + 1) > free
        assert surface.required_free_bytes(fitted) == model.required_bytes(fitted)
    for knee in knees:
        assert min(fitted_counts) < knee < max(fitted_counts), (
            f"no sampled free value straddles the {knee}-cell knee, so the "
            "piecewise solve is not being exercised"
        )


def test_the_door_forwards_its_own_requirement_into_the_driver_argv(
    receipts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver must be handed the number the door decided on.

    THE BREAKAGE: the door admits on this card's row and the driver refuses on
    the default one, after the mesh bind and the kernel compile have been
    spent.  The card is read once, at the door, and the resolved row is what
    both the verdict and the argv are computed from -- so the card is
    substituted here rather than required.
    """

    monkeypatch.setattr(door, "read_card_profile", lambda: surface.REFERENCE_CARD)
    # ...and the seam pin, for the same reason and by the same gesture: the
    # checkout below is an empty directory, this test decides a MEMORY
    # question, and the door's byte comparison against a real gpuwm checkout
    # is exercised in tests/test_engine_pin.py against real bytes.
    monkeypatch.setattr(door, "seam_pin_problem", lambda checkout: None)

    grid = tmp_path / "g.nc"
    static = tmp_path / "s.nc"
    init = tmp_path / "i.nc"
    for path in (grid, static, init):
        path.write_bytes(b"x")
    checkout = tmp_path / "engine"
    checkout.mkdir()
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    x1_cells = _ledger_cells(X1_TAG)
    registry = {"x1.40962": door.MeshRow("x1.40962", x1_cells, 120.0)}
    base = [
        "--mesh", "x1.40962",
        "--grid", str(grid), "--static", str(static), "--init", str(init),
        "--init-source", "GFS", "--hours", "1.0",
        "--history-every-minutes", "30",
        "--out", str(tmp_path / "out"),
        "--gpuwm-checkout", str(checkout),
        "--repo", str(door.PROJECT_ROOT or tmp_path),
    ]
    # Default: the card is read, its row is selected, and the forwarded number
    # is the shared sum on that row.
    request = door._with_resolved_model(
        door.resolve_request(parser.parse_args(base), registry=registry)
    )
    assert isinstance(request.model, surface.ShapedFootprintModel)
    assert request.model.card is surface.REFERENCE_CARD
    argv = door.build_driver_argv(request)
    forwarded = int(argv[argv.index("--required-free-bytes") + 1])
    assert forwarded == surface.required_free_bytes(x1_cells)

    # A card's own supplied row: the forwarded number moves WITH the door's
    # verdict, so the driver cannot refuse what the door admitted.
    own_row = parser.parse_args(
        base + ["--device-fixed-mib", "2237.7",
                "--device-bytes-per-cell", "103696"]
    )
    request = door._with_resolved_model(
        door.resolve_request(own_row, registry=registry)
    )
    assert isinstance(request.model, surface.ShapedFootprintModel)
    argv = door.build_driver_argv(request)
    forwarded = int(argv[argv.index("--required-free-bytes") + 1])
    assert forwarded == surface.required_free_bytes(
        x1_cells, request.model, request.headroom_bytes
    )
    assert forwarded < surface.required_free_bytes(x1_cells)
    # The supplied row is a CORE, and the tiled workspaces are still charged
    # on top of it from the card -- they are arithmetic on the card and are
    # not part of anything a user measured as a fixed term.
    assert request.model.tiled_bytes(x1_cells) == surface.FOOTPRINT_MODEL.tiled_bytes(
        x1_cells
    )


def test_the_driver_refuses_a_non_positive_requirement() -> None:
    forecast = _load_tool("forecast_driver_admission", "run_cuda_v841_forecast.py")
    with pytest.raises(SystemExit):
        forecast.parse_args(
            [
                "--init", "i.nc", "--init-source", "GFS", "--hours", "1",
                "--history-every-minutes", "30", "--preflight-only",
                "--required-free-bytes", "0",
            ]
        )


# ---------------------------------------------------------------------------
# the retired rows: computable, dated, re-fittable from their own ledgers
# ---------------------------------------------------------------------------
def test_the_retired_row_is_the_fit_of_its_own_raw_ledgers(receipts) -> None:
    """The before arm is measured too, and stays re-derivable from ITS ledgers.

    The 2026-08-25 row was not a bad measurement -- it was a good one quoted
    at a tree it was not taken on.  Keeping its ledgers and re-fitting them
    here is what makes that statement checkable instead of asserted.
    """

    fixed_mib, slope = _affine_two_point_fit(
        _ledger_cells(X1_TAG, PRIOR_EVIDENCE),
        _ledger_peak_mib(X1_TAG, PRIOR_EVIDENCE),
        _ledger_cells(X4_TAG, PRIOR_EVIDENCE),
        _ledger_peak_mib(X4_TAG, PRIOR_EVIDENCE),
    )
    retired = surface.RETIRED_ROW_20260825
    assert retired.bytes_per_cell == pytest.approx(slope, abs=2.0)
    assert retired.fixed_bytes / MIB == pytest.approx(fixed_mib, abs=0.2)

    # The specific number the merged-tip re-fit bought back at the smallest
    # mesh: the 08-25 row left x1.40962 only 27.9 MiB above its own measured
    # peak on the tree it was actually quoted at.
    x1_cells = _ledger_cells(X1_TAG)
    x1_peak = _ledger_peak_mib(X1_TAG) * MIB
    assert (
        surface.retired_converged_row_floor_bytes(x1_cells) - x1_peak
    ) / MIB == pytest.approx(27.9, abs=0.2)


def test_the_retired_affine_row_is_the_fit_of_the_merged_tip_ledgers(receipts) -> None:
    """The row THIS lane retired was measured impeccably and had the wrong
    shape.  Both halves must stay checkable: it re-fits from the same two raw
    ledgers the shipped core is derived from, and its provenance says when it
    was retired and that SHAPE, not provenance, is why.

    THE BREAKAGE a dated reason prevents: a reader who finds a retired row
    with no reason on it re-measures instead of re-shaping, and lands back on
    a line.
    """

    fixed_mib, slope = _affine_two_point_fit(
        _ledger_cells(X1_TAG),
        _ledger_peak_mib(X1_TAG),
        _ledger_cells(X4_TAG),
        _ledger_peak_mib(X4_TAG),
    )
    retired = surface.RETIRED_AFFINE_ROW_20260826
    assert retired.bytes_per_cell == pytest.approx(slope, abs=2.0)
    assert retired.fixed_bytes / MIB == pytest.approx(fixed_mib, abs=0.2)

    provenance = retired.provenance
    assert "RETIRED" in provenance
    assert "2026-08-27" in provenance, "a retired row must carry its retirement date"
    assert "SHAPE" in provenance or "shape" in provenance, (
        "the retired affine row must say it was retired for its shape, not "
        "for its provenance"
    )


def test_the_retired_row_names_the_tree_it_was_measured_at_and_says_it_retired() -> None:
    provenance = surface.RETIRED_ROW_20260825.provenance
    assert "RETIRED" in provenance
    assert "7fe514b" in provenance
    assert "pin-move-335-20260825" in provenance


def test_the_two_retired_rows_moved_both_terms_in_opposite_ways(receipts) -> None:
    """THE BREAKAGE, stated as the record must state it.

    A reader who is told only "the tip costs more memory" will move the fixed
    term and stop.  The measurement said something else: x1.40962 rose 484.0
    MiB while x4.163842 FELL 96.0 MiB, so between the 08-25 row and the 08-26
    affine row the fixed term rose and the slope fell.  A record that hides
    that shape licenses the next lane to patch one term.
    """

    older = surface.RETIRED_ROW_20260825
    newer = surface.RETIRED_AFFINE_ROW_20260826
    assert newer.fixed_bytes > older.fixed_bytes
    assert newer.bytes_per_cell < older.bytes_per_cell

    # and the two deltas are the raw ledgers' own, not a restatement
    assert _ledger_peak_mib(X1_TAG) - _ledger_peak_mib(
        X1_TAG, PRIOR_EVIDENCE
    ) == pytest.approx(484.0, abs=0.5)
    assert _ledger_peak_mib(X4_TAG) - _ledger_peak_mib(
        X4_TAG, PRIOR_EVIDENCE
    ) == pytest.approx(-96.0, abs=0.5)


# ---------------------------------------------------------------------------
# the 24 GiB proxy: the payoff, and the citations that must not outlive the
# constant they came from
# ---------------------------------------------------------------------------
LARGEST_REGISTERED_MESH = "v15.60.224210"


def test_the_retired_proxy_refuses_the_largest_mesh_that_the_measured_row_admits(
    receipts,
    binding_mod,
) -> None:
    """The payoff, both arms in one check.

    RED on the retired floor, GREEN on the measured row.  The retired proxy
    demanded more free memory than the RTX 5090 physically HAS for the
    224,210-cell graded mesh, so no amount of freeing admitted it; the shaped
    model asks for less than the card actually offered.  Reverting the shipped
    model to the linear proxy fails this test by name.
    """

    receipt = json.loads(
        (REDERIVE_EVIDENCE / "R1-forecast" / "cuda-v841-forecast-receipt.json")
        .read_text(encoding="utf-8")
    )
    admission = receipt["forecast"]["memory_admission"]
    node2_total = int(admission["total_bytes"])
    node2_free = int(admission["free_bytes"])

    cells = binding_mod.MESH_BINDINGS[LARGEST_REGISTERED_MESH].n_cells
    retired = surface.retired_linear_floor_bytes(cells)
    measured = surface.required_free_bytes(cells)

    # the defect: the proxy asks for more than the card contains
    assert retired > node2_total
    # the fix: the shaped row fits inside what the card actually offered
    assert measured < node2_free
    # and it still sits above the predicted peak by exactly the model's margin
    predicted = int(round(surface.FOOTPRINT_MODEL.predict_bytes(cells)))
    assert measured - predicted == surface.FOOTPRINT_MODEL.margin_bytes()


def test_the_retired_arm_reproduces_the_numbers_the_tree_actually_quoted() -> None:
    """The before arm is computed, never hand-typed, and it is the real one.

    Both figures are lifted from artifacts, not from arithmetic: the graded
    lane's own forecast receipt carries ``memory_admission.minimum =
    23,852,034,111`` at 151,649 cells, and its refusal for the canonical mesh
    read ``requires at least 35264753266 free device bytes``.
    """

    assert surface.retired_linear_floor_bytes(151_649) == 23_852_034_111
    assert surface.retired_linear_floor_bytes(224_210) == 35_264_753_266
    assert surface.retired_linear_floor_bytes(surface.NATIVE_MESH_CELLS) == (
        surface.RETIRED_LINEAR_FLOOR_BYTES
    )


# ---------------------------------------------------------------------------
# the guard sweep: a fix retires the guards that cite the defect it fixed
# ---------------------------------------------------------------------------
#: Files that govern a run -- code, the mesh registry's own notes, and the
#: shipped documentation.  Dated ledgers (``CHANGELOG.md``, ``STATE.md``) and
#: ``evidence/`` are deliberately OUT: they are historical records whose
#: entries carry their own date, and rewriting them would destroy the record
#: this lane exists to explain.
def _governing_files() -> list[Path]:
    paths: list[Path] = [ROOT / "README.md"]
    for pattern in ("src/**/*.py", "tools/**/*.py", "docs/**/*.md"):
        paths.extend(sorted(ROOT.glob(pattern)))
    return [path for path in paths if path.is_file()]


#: The ONE file allowed to carry the retired arms: it computes them and
#: records the history of every constant it replaced.
RETIRED_ARM_HOME = "src/hexcore/device_admission.py"


def test_no_governing_surface_quotes_the_retired_proxy_as_the_requirement(
    binding_mod,
) -> None:
    """RED at this lane's base.

    THE BREAKAGE: the graded-mesh lane measured its capacity boundaries
    against the retired proxy on a base that predated the re-derivation, and
    its conclusions merged beside the re-derived floor -- so three registry
    notes told a reader that meshes the shipped floor admits are "refused on
    every card this program owns".  A note is read as the answer; a stale one
    stops meshes from being run just as effectively as a stale gate.
    """

    banned: dict[str, str] = {}
    for name, row in binding_mod.MESH_BINDINGS.items():
        demand = surface.retired_linear_floor_bytes(int(row.n_cells))
        banned[f"{demand:,}"] = f"the retired proxy's demand at {name}"
        banned[str(demand)] = f"the retired proxy's demand at {name}"
    banned["cells-scaled"] = "the retired floor's shape, described as live"
    banned["CELLS-SCALED"] = "the retired floor's shape, described as live"

    offences: list[str] = []
    for path in _governing_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative == RETIRED_ARM_HOME:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token, why in banned.items():
            if token in text:
                offences.append(f"{relative}: {token!r} -- {why}")
    assert not offences, (
        "these surfaces still quote the retired linear floor as the governing "
        "requirement; compute the before arm from "
        "device_admission.retired_linear_floor_bytes instead:\n  "
        + "\n  ".join(sorted(offences))
    )


def test_the_retired_arm_is_never_wired_into_an_admission_decision() -> None:
    """The before arms must stay instruments, not become second gates.

    The project spent a lane collapsing two admission surfaces into one; a
    helper that computes a retired floor is a third one waiting to happen,
    so nothing under ``src/`` or ``tools/`` may call any of them.
    """

    callers: list[str] = []
    for path in _governing_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative == RETIRED_ARM_HOME or not relative.endswith(".py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for arm in (
            "retired_linear_floor_bytes(",
            "retired_converged_row_floor_bytes(",
            "retired_affine_row_floor_bytes(",
            "RETIRED_ROW_20260825",
            "RETIRED_AFFINE_ROW_20260826",
        ):
            if arm in text:
                callers.append(f"{relative}: {arm}")
    assert not callers, (
        "the retired floors are computable for evidence and tests only; "
        f"these shipped surfaces reach them: {callers}"
    )


def test_the_retired_tier_table_is_never_read_in_an_admission_path() -> None:
    """Ledger #366 in one line: the tier table was correct data with no caller.

    THE BREAKAGE if it gains one now: the tier table is keyed by a card's
    NAMEPLATE MEMORY, and the footprint's card-scaled terms are priced from
    the multiprocessor count.  A row selected by memory size is another card's
    row.  ``model_for_card`` is the surface that selects; nothing may reach
    around it to the retired table.
    """

    callers: list[str] = []
    for path in _governing_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative == RETIRED_ARM_HOME or not relative.endswith(".py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for arm in ("tier_model(", "CARD_TIER_ROWS", "RETIRED_CARD_TIER_ROWS_20260826"):
            if arm in text:
                callers.append(f"{relative}: {arm}")
    assert not callers, (
        "the per-card tier table is retired and keyed by nameplate memory; "
        "select a row with device_admission.model_for_card, which reads the "
        f"card's multiprocessor count: {callers}"
    )


#: Files whose whole job is to carry the HISTORY of these rows, and which
#: therefore may name a superseded coefficient without claiming it governs.
#: ``device-memory-ledger.md`` is the dated row-by-row ledger; the
#: copy-elision tool offers every historical row by name on purpose and has
#: its own test pinning its "of record" arm to ``FLOOR_DERIVATION``.
SUPERSEDED_ROW_HISTORY_HOMES = frozenset(
    {
        RETIRED_ARM_HOME,
        "docs/device-memory-ledger.md",
        "tools/device_memory_capacity/copy_elision_accounting.py",
    }
)


def test_no_governing_surface_quotes_a_superseded_row_as_the_footprint() -> None:
    """RED at this lane's base, twice over, and the reason this lane exists.

    THE BREAKAGE: the 2026-08-25 row was measured at hex ``7fe514b`` and then
    restated verbatim in the README, the manual, ``pyproject.toml`` and the
    driver comments.  Those restatements are copies with no test on them, so
    when the tip moved the footprint they kept telling operators that
    x1.40962 peaks at 8,390 MiB on a tree where it peaks at 8,874 -- and a
    reader sizing a 12 GiB card from the README would have sized it 484 MiB
    short.  A number is governing wherever an operator can read it.

    The 2026-08-26 affine row is now in the same position, and so is the flat
    512 MiB headroom: a doc that hands an operator ``fixed + slope x cells``
    and a flat headroom is handing them the shape that overran a real run by
    96 MiB.  Both retired rows and the flat headroom are computable from
    ``device_admission``; nothing else may hand-type them.
    """

    banned: dict[str, str] = {}
    for retired, label in (
        (surface.RETIRED_ROW_20260825, "the 2026-08-25 row"),
        (surface.RETIRED_AFFINE_ROW_20260826, "the retired 2026-08-26 affine row"),
    ):
        fixed_mib = retired.fixed_bytes / surface.MIB
        slope = int(retired.bytes_per_cell)
        banned[f"{fixed_mib:,.1f}"] = f"{label}'s fixed term"
        banned[f"{fixed_mib:.1f}"] = f"{label}'s fixed term"
        banned[f"{slope:,}"] = f"{label}'s per-cell slope"
        banned[str(slope)] = f"{label}'s per-cell slope"
    for peak, mesh in ((8_390, "x1.40962"), (20_542, "x4.163842")):
        banned[f"{peak:,} MiB"] = f"the superseded {mesh} peak"

    flat_mib = surface.RETIRED_FLAT_HEADROOM_BYTES // surface.MIB
    for token in (
        f"{flat_mib} MiB headroom",
        f"plus {flat_mib} MiB",
        f"{flat_mib} headroom",
    ):
        banned[token] = "the retired flat headroom, quoted as the live margin"

    offences: list[str] = []
    for path in _governing_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in SUPERSEDED_ROW_HISTORY_HOMES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token, why in banned.items():
            if token in text:
                offences.append(f"{relative}: {token!r} -- {why}")
    assert not offences, (
        "these surfaces still quote a retired row or the retired flat "
        "headroom as the footprint; restate the shaped model, or compute the "
        "before arm from device_admission:\n  " + "\n  ".join(sorted(offences))
    )


# ---------------------------------------------------------------------------
# the retired per-card tier table: kept computable, labelled retired
# ---------------------------------------------------------------------------
def test_the_retired_of_record_tier_row_is_the_retired_affine_row() -> None:
    """The tier table's 170 SM row is the affine row this lane retired, and it
    must stay arithmetic on that row rather than a second hand-typed copy of
    it."""

    row = surface.RETIRED_CARD_TIER_ROWS_20260826["32gib-170sm"]
    assert row["measured"] is True
    model = surface.tier_model("32gib-170sm")
    assert isinstance(model, surface.FootprintModel)
    assert model.fixed_bytes == surface.RETIRED_AFFINE_ROW_20260826.fixed_bytes
    assert model.bytes_per_cell == surface.RETIRED_AFFINE_ROW_20260826.bytes_per_cell


def test_the_twelve_gib_tier_row_is_labeled_derived_not_measured() -> None:
    """No 12 GiB card exists in the fleet.  A fabricated 'measured' row would
    be quoted as hardware truth; the label is what prevents that."""

    row = surface.RETIRED_CARD_TIER_ROWS_20260826["12gib"]
    assert row["measured"] is False
    assert "DERIVED, NOT MEASURED" in row["basis"]
    assert "12 GiB card exists" in row["card"]
    # The label survives into the model provenance every consumer prints.
    assert "DERIVED" in surface.tier_model("12gib").provenance


def test_every_retired_tier_row_states_its_basis_and_says_it_is_retired() -> None:
    """THE BREAKAGE: a retired table read as a live one.

    The tier rows are still computable so the before arm of ledger #366 has a
    home, but every model they build must announce its retirement in the
    provenance every consumer prints, and every row must still say what it was
    read from.
    """

    assert surface.CARD_TIER_ROWS is surface.RETIRED_CARD_TIER_ROWS_20260826
    for key, row in surface.RETIRED_CARD_TIER_ROWS_20260826.items():
        assert isinstance(row["measured"], bool), key
        assert isinstance(row["basis"], str) and row["basis"].strip(), key
        model = surface.tier_model(key)
        assert isinstance(model, surface.FootprintModel), key
        assert model.provenance.startswith("RETIRED"), (
            f"{key}: the tier model does not announce that it is retired, so a "
            "consumer printing its provenance quotes it as live"
        )


def test_the_ten_gib_tier_row_names_the_double_counted_desktop_baseline() -> None:
    """The second defect in the retired 10 GiB row, kept named.

    Its fixed term came from a WDDM device-view peak that includes the
    desktop's own baseline, and the door compares against FREE memory, which
    already excludes it -- so the row charged the desktop twice.  A retired row
    whose second defect goes unrecorded gets copied forward by the next lane
    that needs a small-card number.
    """

    basis = surface.RETIRED_CARD_TIER_ROWS_20260826["10gib-68sm"]["basis"]
    assert "baseline" in basis
    assert "twice" in basis
    # And the shaped row for the same card does NOT carry the baseline.
    shaped = surface.model_for_card(surface.KNOWN_CARDS["10gib-68sm"])
    assert shaped.core_bytes < surface.tier_model("10gib-68sm").fixed_bytes


# ---------------------------------------------------------------------------
# the derivation record: the ruling, the shape, and what stays open
# ---------------------------------------------------------------------------
def test_the_floor_derivation_record_carries_the_ruling_and_the_shape() -> None:
    """THE BREAKAGE: a machine-readable record that still describes a line.

    The record is what a receipt, a chart or a later lane reads instead of the
    module.  If it names a fixed term and a slope, the next reader re-fits a
    line, which is the thing that was just refuted.
    """

    record = surface.FLOOR_DERIVATION
    assert record["derived"] == "2026-08-27"
    assert "shaped" in record["schema"]

    ruling = record["ruling"]
    assert "wrong shape" in ruling
    assert "512 MiB headroom is not sufficient" in ruling

    model = record["model"]
    assert "core(card, configuration)" in model
    assert "min(cells, tile(card))" in model
    assert "bytes_per_cell * cells" in model
    assert "arena_placement_margin(card)" in model

    what_moved = record["what_moved"]
    for named in ("Grell-Freitas", "YSU", "RRTMG"):
        assert named in what_moved, (
            f"what_moved does not name {named}, so a reader cannot tell which "
            "term came out of the fixed and slope terms"
        )
    assert "knee" in what_moved

    assert "retired_affine_row_floor_bytes" in record["retired_rows_computable_at"]
    assert "retired_converged_row_floor_bytes" in record["retired_rows_computable_at"]
    assert "retired_linear_floor_bytes" in record["retired_rows_computable_at"]


def test_the_floor_derivation_reproduces_the_ledger_from_the_modules_own_functions(
    receipts,
) -> None:
    """The record's numbers are recomputed, not read.

    THE BREAKAGE: the record is the surface a chart quotes, so a figure in it
    that drifts from the module becomes a published number no test covers --
    the same failure mode as the README restatements this lane swept.
    """

    record = surface.FLOOR_DERIVATION["reproduces_the_ledger"]
    card = surface.REFERENCE_CARD
    x1_cells, x4_cells = _ledger_cells(X1_TAG), _ledger_cells(X4_TAG)

    pairs = (
        ("grell_freitas_x1_mib", surface.GRELL_FREITAS_WORKSPACE, x1_cells),
        ("grell_freitas_x4_mib", surface.GRELL_FREITAS_WORKSPACE, x4_cells),
        ("ysu_x1_mib", surface.YSU_WORKSPACE, x1_cells),
        ("ysu_x4_mib", surface.YSU_WORKSPACE, x4_cells),
    )
    for key, workspace, cells in pairs:
        assert record[key] == pytest.approx(
            workspace.bytes_for(cells, card) / MIB, abs=0.01
        ), f"{key} in the derivation record no longer matches the module"

    assert record["shortwave_spcvmc_bytes_170sm"] == surface.shortwave_workspace_bytes(
        card
    )
    # and the same figure is what the raw ledger recorded
    assert record["shortwave_spcvmc_bytes_170sm"] == _ledger_block_bytes(
        X1_TAG, "core/rrtmg_sw.py:"
    )

    margin_replaces = surface.FLOOR_DERIVATION["margin_replaces"]
    assert f"{surface.RETIRED_FLAT_HEADROOM_BYTES // MIB} MiB flat" in margin_replaces
    assert "96 MiB" in margin_replaces, (
        "the record must name the measured failure the flat headroom had, or "
        "the replacement reads as a preference"
    )
    assert "RETIRED_FLAT_HEADROOM_BYTES" in margin_replaces


def test_the_floor_derivation_states_what_is_open() -> None:
    """Stated limits on page one, not in a follow-up nobody reads.

    THE BREAKAGE: an envelope quoted as a fit, a short-baseline slope quoted
    as a property of the card, or a level-count dependence discovered by a
    later lane at run time.
    """

    open_items = surface.FLOOR_DERIVATION["open_and_stated"]
    assert set(open_items) == {
        "limited_area_core_is_an_envelope",
        "four_swath_spread_not_separated",
        "70sm_slope",
        "nz_dependence",
    }
    for key, text in open_items.items():
        assert isinstance(text, str) and text.strip(), key


# ---------------------------------------------------------------------------
# the points the model was NOT fitted to: the #365 answer, measured
# ---------------------------------------------------------------------------
def _r1_record() -> tuple[int, float]:
    """``(cells, peak_mib)`` for the 30-step graded run, from raw artifacts.

    The peak is the sampler's own recorded maximum and is cross-checked
    against the raw sample rows: a hand-copied peak is how a number outlives
    the run that produced it.
    """

    run = REDERIVE_EVIDENCE / "R1-forecast"
    peak = float((run / "VRAM_PEAK_MIB").read_text(encoding="utf-8").strip())
    samples = [
        float(line)
        for line in (run / "vram.samples").read_text(encoding="utf-8").split()
        if line.strip()
    ]
    assert samples, "the sampler wrote no rows; the peak cannot be cross-checked"
    assert max(samples) == pytest.approx(peak, abs=0.5)

    mesh_receipt = json.loads((run / "mesh-receipt.json").read_text(encoding="utf-8"))
    return int(mesh_receipt["observed"]["nCells"]), peak


def test_the_new_margin_covers_the_graded_point_the_retired_row_overran(
    receipts,
) -> None:
    """The #365 answer, both arms, from the artifacts.

    The graded mesh ``v20.80.151649`` was measured twice: once for 30 door
    steps (19,838 MiB) and once at the six-step #264 probe protocol
    (19,255.25 MiB).  The retired affine row plus its flat 512 MiB headroom
    did NOT cover the longer run -- that is the recorded overrun.  The shaped
    model's margin must cover both, because the concrete breakage is a card
    admitted at exactly the requirement that then dies inside a CuPy
    allocation minutes into the run.
    """

    cells, long_run_peak_mib = _r1_record()
    probe = json.loads(
        (SWATHS_EVIDENCE / "ledger-365.json").read_text(encoding="utf-8")
    )
    assert int(probe["n_cells"]) == cells, (
        "the six-step probe is no longer the same mesh as the 30-step run, so "
        "the two peaks are not two measurements of one point"
    )
    probe_peak_mib = float(probe["measured_peak_mib"])

    required = surface.required_free_bytes(cells)
    assert required > long_run_peak_mib * MIB, (
        "the shaped model no longer covers the 30-step graded peak; a card "
        "offered exactly this requirement would be admitted and would then "
        "die inside a CuPy allocation"
    )
    assert required > probe_peak_mib * MIB

    # the before arm: the retired row plus the flat headroom did not cover it
    assert surface.retired_affine_row_floor_bytes(cells) < long_run_peak_mib * MIB, (
        "the retired affine row no longer under-covers the graded run, so the "
        "recorded overrun has stopped being reproducible from the module"
    )

    # The run itself was admitted on the row of record AT THE TIME, which is
    # the 2026-08-25 arm.  Asserting it against today's row would silently
    # rewrite what the artifact says; asserting it against the retired arm
    # proves the arm reproduces a real receipt to the byte.
    receipt = json.loads(
        (REDERIVE_EVIDENCE / "R1-forecast" / "cuda-v841-forecast-receipt.json")
        .read_text(encoding="utf-8")
    )
    admission = receipt["forecast"]["memory_admission"]
    assert admission["admitted"] is True
    assert admission["minimum"] == surface.retired_converged_row_floor_bytes(cells)
    assert admission["minimum"] != required


def test_the_recorded_overrun_on_the_placed_mesh_is_covered_by_the_new_model(
    receipts,
) -> None:
    """The 96 MiB overrun, as it happened, and the fix, from the run record.

    Swath run ``s01`` was admitted on the retired affine row and then peaked
    ABOVE its own recorded ``required_free`` -- the flat headroom was spent
    and overrun on a run that had already passed the gate.  That is the
    concrete breakage the named margin exists to prevent, so the shaped
    model's requirement at the same cell count must sit above the same
    measured peak.
    """

    run = json.loads(
        (SWATHS_EVIDENCE / "runs" / "s01.run.json").read_text(encoding="utf-8")
    )
    cells = int(run["admission_cells"])
    peak_bytes = float(run["vram_peak_mib"]) * MIB
    recorded_required = int(run["admission_required_free_bytes"])

    assert recorded_required < peak_bytes, (
        "the recorded run no longer shows the overrun; if the artifact "
        "changed, re-record it rather than deleting the check"
    )
    assert surface.required_free_bytes(cells) > peak_bytes, (
        "the shaped model does not cover the peak that overran the retired "
        "gate, so the margin has not fixed what it was built to fix"
    )


def test_the_shaped_row_is_exact_on_the_two_meshes_it_was_fitted_from(receipts) -> None:
    """The margin is spent on what it names, not on fit error.

    On its own fit points the gap between the requirement and the measured
    peak must be the margin itself, to rounding.  THE BREAKAGE: a row that
    quietly absorbs fit error into the margin has less margin left for the
    arena placement it was sized for, on exactly the meshes it was fitted on.
    """

    for tag in (X1_TAG, X4_TAG):
        cells = _ledger_cells(tag)
        peak_bytes = _ledger_peak_mib(tag) * MIB
        gap = surface.required_free_bytes(cells) - peak_bytes
        assert gap == pytest.approx(
            surface.FOOTPRINT_MODEL.margin_bytes(), abs=0.2 * MIB
        )
