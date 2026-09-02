"""A row carrying declared extras can be registered under its exact block.

The prefix arm of ``species_row_for_names`` answers "how many leading
species load the air" for a block it has never seen, and it answers that
correctly.  It cannot answer "what are the appended scalars' history names
and which accumulators ride with them", because it returns the BASE row and
the extras' declarations are not on it.  Registration is how a provider
makes its extended row the exact answer, and these tests measure both
arms: registered blocks resolve to the extended row, unregistered blocks
still resolve by prefix.
"""

from __future__ import annotations

import pytest

from hexcore import species_row as sr
from hexcore.errors import ConfigurationRefusal


def _number(name: str) -> sr.SpeciesDeclaration:
    return sr.SpeciesDeclaration(
        name=name,
        species_class=sr.NUMBER,
        transported=True,
        history_name=name.upper(),
        description=f"declared extra {name}",
    )


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(sr, "_EXTENDED_ROWS", {})
    return sr


def test_an_unregistered_extended_block_resolves_by_prefix(clean_registry):
    block = sr.P3_SPECIES_ROW.names() + ("x_one", "x_two")
    row = sr.species_row_for_names(block)
    assert row is sr.P3_SPECIES_ROW
    assert row.declared_extras() == ()


def test_a_registered_extended_block_resolves_to_the_extended_row(clean_registry):
    extended = sr.P3_SPECIES_ROW.with_extras(
        species=(_number("x_one"), _number("x_two")),
        accumulators=(
            sr.SurfaceAccumulatorDeclaration(
                "x_sfc", "float64", "X_SFC", "a whole-forecast integral"
            ),
        ),
    )
    assert sr.register_extended_row(extended) is extended
    row = sr.species_row_for_names(extended.names())
    assert row is extended
    assert [item.name for item in row.declared_extras()] == ["x_one", "x_two"]
    assert row.history_names()["x_one"] == "X_ONE"
    assert [item.name for item in row.declared_extra_accumulators()] == ["x_sfc"]
    assert row.declared_extra_accumulators()[0].dtype == "float64"
    assert row.n_mass_loading == sr.P3_SPECIES_ROW.n_mass_loading
    # The base rows themselves are untouched by the registration.
    assert sr.species_row_for_names(sr.P3_SPECIES_ROW.names()) is sr.P3_SPECIES_ROW
    assert dict(sr.registered_extended_rows()) == {extended.names(): extended}


def test_the_same_row_registers_twice_and_a_different_one_is_refused(clean_registry):
    first = sr.WSM6_SPECIES_ROW.with_extras(species=(_number("x_one"),))
    again = sr.WSM6_SPECIES_ROW.with_extras(species=(_number("x_one"),))
    assert first == again
    sr.register_extended_row(first)
    assert sr.register_extended_row(again) is first
    drifted = sr.WSM6_SPECIES_ROW.with_extras(
        species=(
            sr.SpeciesDeclaration(
                name="x_one",
                species_class=sr.VOLUME,
                transported=True,
                history_name="X_ONE",
                description="the same name, a different class",
            ),
        )
    )
    with pytest.raises(ConfigurationRefusal, match="not replaceable"):
        sr.register_extended_row(drifted)


def test_a_row_with_no_extras_is_not_an_extended_row(clean_registry):
    with pytest.raises(ValueError, match="declares no extras"):
        sr.register_extended_row(sr.WSM6_SPECIES_ROW)


def test_an_extended_row_off_an_unknown_scheme_refuses(clean_registry):
    from dataclasses import replace

    stranger = replace(
        sr.WSM6_SPECIES_ROW.with_extras(species=(_number("x_one"),)),
        scheme_name="mp_nobody",
    )
    with pytest.raises(ConfigurationRefusal, match="extends no registered"):
        sr.register_extended_row(stranger)


def test_the_extended_row_must_lead_with_its_base(clean_registry):
    from dataclasses import replace

    extended = sr.WSM6_SPECIES_ROW.with_extras(species=(_number("x_one"),))
    with pytest.raises(ValueError):
        # Reversing puts x_one first; the row's own qv-at-index-0 invariant
        # refuses at construction, and if it ever stopped, the base-prefix
        # check at registration refuses -- either is the right answer.
        sr.register_extended_row(
            replace(extended, species=extended.species[::-1])
        )
    # A row that leads with the WRONG base: P3's eight under WSM6's name.
    with pytest.raises(ValueError, match="does not lead with its base"):
        sr.register_extended_row(
            replace(
                sr.P3_SPECIES_ROW.with_extras(species=(_number("x_one"),)),
                scheme_name="mp_wsm6",
                mp_physics=6,
                engine_scheme="wsm6",
            )
        )


def test_a_scheme_alias_selects_the_extended_row_from_the_config(clean_registry, monkeypatch):
    monkeypatch.setattr(sr, "_EXTENDED_SCHEMES", {})
    extended = sr.P3_SPECIES_ROW.with_extras(species=(_number("x_one"),))
    sr.register_extended_row(extended, scheme_alias="mp_p3+lane_x")
    assert sr.species_row_for_scheme("mp_p3+lane_x") is extended
    assert sr.species_row_for_scheme("MP_P3+LANE_X") is extended
    assert extended.require_engine_scheme() == "p3"
    # The scheme rows themselves are untouched, and the alias is listed.
    assert sr.species_row_for_scheme("mp_p3") is sr.P3_SPECIES_ROW
    assert dict(sr.registered_scheme_aliases()) == {"mp_p3+lane_x": extended}
    # Registering the identical row again under the same alias is a no-op.
    assert sr.register_extended_row(extended, scheme_alias="mp_p3+lane_x") is extended


def test_an_alias_may_not_shadow_a_scheme_row(clean_registry, monkeypatch):
    monkeypatch.setattr(sr, "_EXTENDED_SCHEMES", {})
    extended = sr.WSM6_SPECIES_ROW.with_extras(species=(_number("x_one"),))
    with pytest.raises(ConfigurationRefusal, match="scheme row's own name"):
        sr.register_extended_row(extended, scheme_alias="mp_p3")


def test_an_alias_is_not_replaceable(clean_registry, monkeypatch):
    monkeypatch.setattr(sr, "_EXTENDED_SCHEMES", {})
    first = sr.WSM6_SPECIES_ROW.with_extras(species=(_number("x_one"),))
    other = sr.WSM6_SPECIES_ROW.with_extras(species=(_number("x_two"),))
    sr.register_extended_row(first, scheme_alias="mp_wsm6+lane_x")
    with pytest.raises(ConfigurationRefusal, match="not replaceable"):
        sr.register_extended_row(other, scheme_alias="mp_wsm6+lane_x")
