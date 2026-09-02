"""The species row is the door's one source of truth about scalars.

These gates hold the invariants every consumer in the v841 chain will read
instead of restating: vapour at index 0, the mass-loading species as a
contiguous prefix, the class tags that keep number and volume state out of
the moisture loading, and the declared-extras hook a later lane appends
through.  The rows themselves are checked against the PINNED ENGINE's own
species table, so hex and the engine cannot drift apart silently.
"""

from __future__ import annotations

import pytest

from hexcore import species_row as sr
from hexcore.errors import ConfigurationRefusal


# ---------------------------------------------------------------------------
# The declared rows.
# ---------------------------------------------------------------------------


def test_wsm6_row_is_the_six_in_order():
    row = sr.WSM6_SPECIES_ROW
    assert row.names() == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert row.mp_physics == 6
    assert row.engine_scheme == "wsm6"


def test_every_wsm6_species_loads_the_air():
    """Why the weld was invisible: on WSM6 the loading set IS the row."""

    row = sr.WSM6_SPECIES_ROW
    assert row.mass_loading_names() == row.names()
    assert row.n_mass_loading == row.n_species() == 6


def test_p3_row_is_wrfs_eight_in_engine_order():
    row = sr.P3_SPECIES_ROW
    assert row.names() == (
        "qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib")
    assert row.mp_physics == 50
    assert row.engine_scheme == "p3"
    # P3's single ice category writes neither snow nor graupel.
    assert "qs" not in row.names()
    assert "qg" not in row.names()


def test_p3_loading_set_excludes_number_volume_and_the_rime_component():
    """The defect this row exists to prevent, stated as a gate.

    ni and nr are concentrations, qib is a volume, and qir is the rime mass
    held INSIDE qi.  All four ride transport; none of them may reach qtot.
    """

    row = sr.P3_SPECIES_ROW
    assert row.mass_loading_names() == ("qv", "qc", "qr", "qi")
    assert row.n_mass_loading == 4
    assert row.n_species() == 8
    for name in ("ni", "nr", "qir", "qib"):
        assert not row.declaration(name).is_mass_loading
        assert row.declaration(name).transported
    assert row.declaration("qir").species_class == sr.MASS_COMPONENT
    assert row.declaration("qib").species_class == sr.VOLUME
    assert row.names_of_class(sr.NUMBER) == ("ni", "nr")


def test_thompson_aerosol_row_is_wrfs_eleven_in_engine_order():
    """The mp=28 refusal retired the way its own text said it would.

    From 2026-09-01 this row was declared and refused because the pinned
    engine carried no ``thompson_aero`` row; gpuwm 2.6.3 published one, the
    two-rows gate below failed, and the row took the engine's scheme name
    and the engine's number-block order.  What stays gated: six loading
    masses first, five number species after, none of them in qtot.
    """

    row = sr.THOMPSON_AEROSOL_SPECIES_ROW
    assert row.mp_physics == 28
    assert row.engine_scheme == "thompson_aero"
    assert row.require_engine_scheme() == "thompson_aero"
    assert row.names() == (
        "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa")
    assert row.n_species() == 11
    assert row.n_mass_loading == 6
    assert row.mass_loading_names() == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert row.names_of_class(sr.NUMBER) == ("ni", "nr", "nc", "nwfa", "nifa")
    for name in ("ni", "nr", "nc", "nwfa", "nifa"):
        assert not row.declaration(name).is_mass_loading
        assert row.declaration(name).transported
    # mp=28 has a graupel category: three buckets, WSM6's three radii.
    assert [item.name for item in row.surface_accumulators] == [
        "rainnc", "snownc", "graupelnc"]
    assert row.radius_names == ("effc", "effi", "effs")


def test_the_rows_with_no_engine_counterparty_still_refuse_by_name():
    """mp=1 and mp=8 stay declared-and-refused, and the refusal names the
    three engine rows that DO exist so the remedy is a real one."""

    for row in (sr.KESSLER_SPECIES_ROW, sr.THOMPSON_SPECIES_ROW):
        assert row.engine_scheme is None
        with pytest.raises(ConfigurationRefusal) as excinfo:
            row.require_engine_scheme()
        message = str(excinfo.value)
        assert "_SPECIES_BY_SCHEME" in message
        assert "engine-side row" in message
        assert "'thompson_aero'" in message
        assert "'mp_thompson_aerosols'" in message


def test_the_loading_prefix_is_contiguous_in_every_declared_row():
    """The invariant that lets the kernels take one int32 loop bound."""

    for name, row in sr.registered_species_rows().items():
        loading = [item.is_mass_loading for item in row.species]
        assert loading == sorted(loading, reverse=True), name
        assert row.mass_loading_names() == row.names()[: row.n_mass_loading]


def test_every_declared_species_is_transported_and_writes_under_a_name():
    for name, row in sr.registered_species_rows().items():
        history = row.history_names()
        assert set(history) == set(row.names()), name
        assert len(set(history.values())) == len(history), name
        for item in row.species:
            assert item.transported, (name, item.name)


# ---------------------------------------------------------------------------
# The row agrees with the PINNED ENGINE, or the door is building on sand.
# ---------------------------------------------------------------------------


def _pinned_engine_column_batch():
    """The engine the hex pin NAMES, or skip saying why.

    THE INSTRUMENT TRAP THIS AVOIDS, hit on 2026-09-01.  These gates grade
    hex's rows against the engine's species table, and the engine they used
    to read was whatever ``import gpuwm`` resolved.  In a tree beside a
    gpuwm development checkout that is NOT the pinned engine -- it can be a
    lane branch that has already gained a scheme, and its own
    ``__version__`` need not even match what pip would install -- so the
    gate grades a counterparty the shipped door will never call, and either
    fails for a scheme the pin does not carry or passes on bytes nobody
    ships.  A gate that cannot say which engine it read is not a
    measurement.
    """

    gpuwm = pytest.importorskip("gpuwm")
    mcb = pytest.importorskip("gpuwm.core.mpas_column_batch")
    version = getattr(gpuwm, "__version__", "unknown")
    pinned = _pinned_engine_range()
    if version not in pinned:
        pytest.skip(
            f"the importable gpuwm is {version} at {gpuwm.__file__}, which "
            f"is not the engine this tree pins ({sorted(pinned)}).  These "
            "gates grade hex's species rows against the PINNED engine's "
            "table; reading another one grades a counterparty the shipped "
            "door never calls"
        )
    return mcb


def _pinned_engine_range() -> frozenset[str]:
    """The exact engine versions this tree's packaging admits."""

    import re
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[1]
            / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"gpuwm>=([0-9.]+),<([0-9.]+)"', text)
    assert match, "the packaging no longer pins gpuwm with a >=,< range"
    # The range is always one release wide in this tree, by the 258 runbook.
    return frozenset({match.group(1)})


def test_rows_match_the_pinned_engine_species_table():
    """Cross-lane law: grade against the other side's real writer.

    The door hands species to ``gpuwm.core.mpas_column_batch``; if hex's row
    and the engine's ``_SPECIES_BY_SCHEME`` disagree on order or membership,
    the door feeds the seam the wrong array with no error anywhere.
    """

    table = getattr(_pinned_engine_column_batch(), "_SPECIES_BY_SCHEME")
    engine_selectors = getattr(
        _pinned_engine_column_batch(), "_MP_PHYSICS_BY_SCHEME")
    for row in sr.registered_species_rows().values():
        if row.engine_scheme is None:
            # The row is declared and refused; the engine must indeed not
            # carry it, or the refusal is stale and must be retired.
            assert row.scheme_name not in table
            assert row.mp_physics not in engine_selectors.values(), (
                f"the engine now carries a row for mp_physics={row.mp_physics}; "
                f"retire {row.scheme_name}'s engine_absent_reason and give "
                "it an engine_scheme"
            )
            continue
        assert row.names() == tuple(table[row.engine_scheme]), row.scheme_name
        assert engine_selectors[row.engine_scheme] == row.mp_physics


def test_the_engine_still_carries_only_three_rows():
    """The named deferrals' own retirement condition, as a gate.

    When this fails, the engine has gained a scheme and one of the door's
    remaining refused rows (mp=1 Kessler, mp=8 Thompson) is ready to be
    retired.  It has fired once already: the 2.6.3 engine took this table
    from two rows to three and the mp=28 refusal retired on that failure.
    """

    table = getattr(_pinned_engine_column_batch(), "_SPECIES_BY_SCHEME")
    assert sorted(table) == ["p3", "thompson_aero", "wsm6"], (
        "the PINNED engine has gained a microphysics row.  Re-check every "
        "declared row's species order against it, then retire that row's "
        "engine_absent_reason and give it an engine_scheme")


# ---------------------------------------------------------------------------
# Construction-time refusals.
# ---------------------------------------------------------------------------


def _decl(name, species_class, history="X"):
    return sr.SpeciesDeclaration(
        name=name,
        species_class=species_class,
        transported=True,
        history_name=history,
        description="test species",
    )


def test_a_row_that_does_not_start_with_vapour_is_refused():
    with pytest.raises(ValueError, match="index 0"):
        sr.MicrophysicsSpeciesRow(
            scheme_name="mp_test",
            mp_physics=999,
            engine_scheme="test",
            species=(_decl("qc", sr.MASS_LOADING),
                     _decl("qv", sr.MASS_LOADING)),
            surface_accumulators=(),
            radius_names=(),
            restart_extra_names=(),
        )


def test_a_row_with_a_loading_species_after_a_number_is_refused():
    """The silent-omission defect the prefix invariant prevents."""

    with pytest.raises(ValueError, match="contiguous LEADING run"):
        sr.MicrophysicsSpeciesRow(
            scheme_name="mp_test",
            mp_physics=999,
            engine_scheme="test",
            species=(
                _decl("qv", sr.MASS_LOADING),
                _decl("ni", sr.NUMBER),
                _decl("qc", sr.MASS_LOADING),
            ),
            surface_accumulators=(),
            radius_names=(),
            restart_extra_names=(),
        )


def test_an_unknown_species_class_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        _decl("qx", "mass")


def test_a_row_with_no_engine_and_no_reason_is_refused():
    with pytest.raises(ValueError, match="no reason"):
        sr.MicrophysicsSpeciesRow(
            scheme_name="mp_test",
            mp_physics=999,
            engine_scheme=None,
            species=(_decl("qv", sr.MASS_LOADING),),
            surface_accumulators=(),
            radius_names=(),
            restart_extra_names=(),
        )


def test_a_repeated_species_name_is_refused():
    with pytest.raises(ValueError, match="repeats a species name"):
        sr.MicrophysicsSpeciesRow(
            scheme_name="mp_test",
            mp_physics=999,
            engine_scheme="test",
            species=(_decl("qv", sr.MASS_LOADING),
                     _decl("qv", sr.MASS_LOADING)),
            surface_accumulators=(),
            radius_names=(),
            restart_extra_names=(),
        )


# ---------------------------------------------------------------------------
# Declared extras: the hook a later lane adds scalars through.
# ---------------------------------------------------------------------------


def test_an_extra_scalar_appends_without_moving_the_schemes_indices():
    base = sr.P3_SPECIES_ROW
    extra = _decl("ntest", sr.NUMBER, history="QNTEST")
    extended = base.with_extras(species=(extra,))
    assert extended.names() == base.names() + ("ntest",)
    for name in base.names():
        assert extended.index_of(name) == base.index_of(name)
    assert extended.n_species() == base.n_species() + 1
    # The whole point: the moisture loading is untouched.
    assert extended.n_mass_loading == base.n_mass_loading
    assert extended.mass_loading_names() == base.mass_loading_names()


def test_an_extra_is_marked_as_declared_and_findable():
    extra = _decl("ntest", sr.NUMBER, history="QNTEST")
    extended = sr.WSM6_SPECIES_ROW.with_extras(species=(extra,))
    assert [item.name for item in extended.declared_extras()] == ["ntest"]
    assert extended.declaration("ntest").declared_extra
    # A scheme's own species are never marked as extras.
    assert not extended.declaration("qv").declared_extra


def test_a_mass_loading_extra_is_refused_by_name():
    """A loading extra would be transported and silently left out of qtot."""

    with pytest.raises(ConfigurationRefusal) as excinfo:
        sr.WSM6_SPECIES_ROW.with_extras(
            species=(_decl("qextra", sr.MASS_LOADING),))
    message = str(excinfo.value)
    assert "outside that prefix" in message
    assert "silently omitted from qtot" in message


def test_an_extra_surface_accumulator_may_be_fp64():
    """A bucket integrating a rate over a forecast needs the width."""

    accumulator = sr.SurfaceAccumulatorDeclaration(
        name="testmass",
        dtype="float64",
        history_name="TESTMASS",
        description="test accumulated mass",
    )
    extended = sr.WSM6_SPECIES_ROW.with_extras(accumulators=(accumulator,))
    extras = extended.declared_extra_accumulators()
    assert [item.name for item in extras] == ["testmass"]
    assert extras[0].dtype == "float64"
    assert extras[0].declared_extra
    # The scheme's own accumulators are unchanged and still first.
    assert extended.surface_accumulators[: len(
        sr.WSM6_SPECIES_ROW.surface_accumulators)] == \
        sr.WSM6_SPECIES_ROW.surface_accumulators


def test_extras_do_not_mutate_the_registered_row():
    before = sr.WSM6_SPECIES_ROW.names()
    sr.WSM6_SPECIES_ROW.with_extras(
        species=(_decl("ntest", sr.NUMBER, history="QNTEST"),))
    assert sr.WSM6_SPECIES_ROW.names() == before
    assert sr.registered_species_rows()["mp_wsm6"].names() == before


def test_an_accumulator_of_an_unknown_width_is_refused():
    with pytest.raises(ValueError, match="only 'float32' and 'float64'"):
        sr.SurfaceAccumulatorDeclaration(
            name="testmass",
            dtype="float16",
            history_name="TESTMASS",
            description="test",
        )


# ---------------------------------------------------------------------------
# Lookup.
# ---------------------------------------------------------------------------


def test_the_row_table_is_exhaustive_over_the_seams_routed_domain():
    """Every scheme the seam can route to has a row, and vice versa.

    The seam refuses a selector with no requirement row because silence
    there declares qv alone -- a state substitution.  The door has the same
    hazard one layer down: a routed scheme with no species row would reach
    the door and find nothing to size its scalar block from.  Deriving both
    from the same table is only safe while the table covers the domain.
    """

    from hexcore import physics_seam as ps

    routed = {int(value) for value in
              ps._ROUTES["config_microp_scheme"].values()}
    routed.discard(0)  # 'off' transports nothing and needs no row
    declared = {row.mp_physics
                for row in sr.registered_species_rows().values()}
    assert declared == routed


def test_every_row_matches_the_seams_own_requirement_ladder():
    """The two answers to "which scalars" must be one answer.

    ``physics_seam._required_scalar_names`` is the seam's hand-written
    ladder and this table was derived independently from the engine's
    species table and WRF's mp driver arms.  They agree today; this gate is
    what keeps them agreeing.
    """

    from hexcore import physics_seam as ps

    for name, row in sr.registered_species_rows().items():
        selection = ps.resolve_mpas_physics(config_microp_scheme=name)
        assert ps._required_scalar_names(selection) ==             row.required_scalar_names(), name


def test_a_block_with_extras_resolves_to_its_LONGEST_matching_row():
    """First-match would resolve an aerosol block to the warm-rain row.

    mp_kessler declares (qv, qc, qr), which is a prefix of every other row.
    A first-match resolver hands an eleven-species block plus a lane's
    appended scalars back as the three-species row, and the caller then
    sums three species into qtot and prepares three of eleven -- with no
    error anywhere.  Measured against the in-tree provider block, which is
    exactly that shape.
    """

    base = sr.THOMPSON_AEROSOL_SPECIES_ROW
    block = base.names() + ("extra_a", "extra_b")
    assert sr.species_row_for_names(block) is base
    assert sr.species_row_for_names(base.names()) is base
    # The short row still resolves for its own block and for its own extras.
    kessler = sr.KESSLER_SPECIES_ROW
    assert sr.species_row_for_names(kessler.names()) is kessler
    assert sr.species_row_for_names(
        kessler.names() + ("nextra",)) is kessler


def test_lookup_by_scheme_name_and_selector_agree():
    for row in sr.registered_species_rows().values():
        assert sr.species_row_for_scheme(row.scheme_name) is row
        assert sr.species_row_for_mp_physics(row.mp_physics) is row


def test_an_undeclared_scheme_is_refused_with_the_declared_set_named():
    with pytest.raises(ConfigurationRefusal) as excinfo:
        sr.species_row_for_scheme("mp_morrison")
    assert "mp_wsm6" in str(excinfo.value)


def test_an_undeclared_selector_is_refused():
    with pytest.raises(ConfigurationRefusal):
        sr.species_row_for_mp_physics(10)
