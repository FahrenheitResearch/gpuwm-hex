"""Further physics-backend rows arrive through an entry-point group, never by name.

THE BREAKAGE THIS REPLACES: the table used to import one named optional
module to find extra rows, which put the name of a component that is not part
of this distribution into its source.  Rows now come from any installed
distribution that declares an entry point in ``gpuwm_hex.rows``; this file
checks the three shapes a provider can take and the one invariant that
matters, that no provider can move what the DEFAULT configuration runs.
No card, no assets.
"""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType

import pytest

from hexcore import physics_backend_admission as admission
from hexcore.errors import ConfigurationRefusal


class _Point:
    """A stand-in for an ``importlib.metadata.EntryPoint``."""

    def __init__(self, name: str, value: str, loader):
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


def _row(name: str) -> admission.PhysicsBackendRow:
    return replace(
        admission.FROZEN_WSM6_COLUMN_ROW,
        name=name,
        adapter_module="provider_under_test.adapter",
        anchor_configuration_class=f"{name}-class",
        anchored=False,
        unanchored_remedy="mint the class",
    )


@pytest.fixture
def fresh_table(monkeypatch):
    """The table as a fresh process sees it, restored afterwards."""

    monkeypatch.setattr(
        admission,
        "_ROWS",
        {name: admission.backend_rows()[name] for name in admission.IN_TREE_ROWS},
    )
    monkeypatch.setattr(admission, "_providers_loaded", False)
    monkeypatch.setattr(admission, "_provider_failures", [])

    def install(points):
        import importlib.metadata as metadata

        monkeypatch.setattr(
            metadata, "entry_points",
            lambda group=None: [p for p in points] if group == admission.ROW_ENTRY_POINT_GROUP else [],
        )

    return install


def test_the_table_names_no_provider_module():
    source = open(admission.__file__, encoding="utf-8").read()
    assert "_ROW_PROVIDERS" not in source
    assert "hexcore.mod" not in source
    assert admission.ROW_ENTRY_POINT_GROUP == "gpuwm_hex.rows"


def test_no_provider_resolves_the_frozen_row_only(fresh_table):
    fresh_table([])
    assert admission.registered_backend_names() == tuple(sorted(admission.IN_TREE_ROWS))
    assert admission.resolve_backend() is admission.FROZEN_WSM6_COLUMN_ROW
    assert admission.row_provider_failures() == ()


def test_a_module_provider_registers_on_import(fresh_table, monkeypatch):
    module = ModuleType("provider_under_test_rows")

    def load():
        admission.register_backend_row(_row("provided_by_import"))
        return module

    fresh_table([_Point("rows", "provider_under_test_rows", load)])
    default_before = admission.resolve_backend()
    assert "provided_by_import" in admission.registered_backend_names()
    assert admission.resolve_backend("provided_by_import").name == "provided_by_import"
    assert admission.resolve_backend() is default_before is admission.FROZEN_WSM6_COLUMN_ROW


def test_a_callable_provider_returns_its_rows(fresh_table):
    fresh_table([_Point("rows", "provider_under_test:rows", lambda: (lambda: [_row("provided_by_call")]))])
    assert admission.resolve_backend("provided_by_call").name == "provided_by_call"


def test_a_provider_cannot_rebind_the_default_row(fresh_table):
    hostile = replace(admission.FROZEN_WSM6_COLUMN_ROW, adapter_module="somewhere.else")
    fresh_table([_Point("rows", "provider_under_test:rows", lambda: (lambda: [hostile]))])
    assert admission.resolve_backend() is admission.FROZEN_WSM6_COLUMN_ROW
    failures = admission.row_provider_failures()
    assert failures and "ConfigurationRefusal" in failures[0][1]


def test_a_broken_provider_is_named_when_its_row_is_asked_for(fresh_table):
    def load():
        raise ImportError("No module named 'provider_under_test'")

    fresh_table([_Point("rows", "provider_under_test", load)])
    with pytest.raises(ConfigurationRefusal) as caught:
        admission.resolve_backend("a_row_the_provider_would_have_carried")
    text = str(caught.value)
    assert "failed to load" in text
    assert "rows = provider_under_test" in text
    assert "No module named" in text
