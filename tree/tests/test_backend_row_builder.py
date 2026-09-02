"""A backend row constructs its own column-physics backend, one call site.

The frozen row constructs the production class with the production keyword
set and refuses a cell area or seam options by name (the frozen batch
would accept them and never read them).  A provider row hands everything
to its adapter module's ``build_column_backend``.  Measured with recorder
classes and modules; no card, no seam.
"""

from __future__ import annotations

import sys
import types

import pytest

from hexcore import physics_backend_admission as registry
from hexcore.errors import ConfigurationRefusal


class _Recorder:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        type(self).calls.append(dict(kwargs))


@pytest.fixture
def frozen_recorder(monkeypatch):
    from hexcore import cuda_arwen_physics_v841 as frozen

    _Recorder.calls = []
    monkeypatch.setattr(
        frozen, "PersistentTwoPhaseCudaPhysicsBackendV841", _Recorder
    )
    return _Recorder


def test_the_frozen_row_constructs_the_production_class_with_the_base_set(
    frozen_recorder,
):
    row = registry.resolve_backend()
    built = row.build_column_backend(
        constructor="c", prep_geometry="g", kernel_cache="k",
        gwdo_static="s", gwdo_kernel_cache="k2", checkout="/co",
    )
    assert isinstance(built, frozen_recorder)
    assert frozen_recorder.calls == [{
        "constructor": "c", "prep_geometry": "g", "kernel_cache": "k",
        "gwdo_static": "s", "gwdo_kernel_cache": "k2",
        "arwen_checkout": "/co",
    }]


@pytest.mark.parametrize("extra", [
    {"cell_area_m2": [1.0, 2.0]},
    {"seam_options": {"aircraft_track": "x"}},
])
def test_the_frozen_row_refuses_an_extra_it_would_never_read(
    frozen_recorder, extra,
):
    row = registry.resolve_backend()
    with pytest.raises(ConfigurationRefusal, match="never read"):
        row.build_column_backend(
            constructor="c", prep_geometry="g", kernel_cache="k", **extra,
        )
    assert frozen_recorder.calls == []
    # An EMPTY option mapping is no option at all.
    row.build_column_backend(
        constructor="c", prep_geometry="g", kernel_cache="k",
        seam_options={},
    )
    assert len(frozen_recorder.calls) == 1


def test_a_provider_row_hands_everything_to_its_builder(monkeypatch):
    seen: list[tuple] = []
    module = types.ModuleType("hexcore_test_provider_module")

    def build_column_backend(row, **kwargs):
        seen.append((row.name, kwargs))
        return "built"

    module.build_column_backend = build_column_backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    row = registry.PhysicsBackendRow(
        name="test_provider_row",
        adapter_module=module.__name__,
        batch_module="nowhere",
        batch_entrypoint="run",
        driven_scalar_names=("qv",),
        appended_scalar_names=("x_one",),
        anchor_configuration_class="test",
        anchored=True,
        unanchored_remedy="",
    )
    built = row.build_column_backend(
        constructor="c", prep_geometry="g", kernel_cache="k",
        checkout="/co", cell_area_m2=[1.0], seam_options={"a": 1},
    )
    assert built == "built"
    assert seen == [(
        "test_provider_row",
        {
            "constructor": "c", "prep_geometry": "g", "kernel_cache": "k",
            "gwdo_static": None, "gwdo_kernel_cache": None,
            "checkout": "/co", "cell_area_m2": [1.0],
            "seam_options": {"a": 1},
        },
    )]


def test_an_adapter_with_nothing_to_construct_refuses_by_name(monkeypatch):
    module = types.ModuleType("hexcore_test_empty_module")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    row = registry.PhysicsBackendRow(
        name="test_empty_row",
        adapter_module=module.__name__,
        batch_module="nowhere",
        batch_entrypoint="run",
        driven_scalar_names=("qv",),
        appended_scalar_names=(),
        anchor_configuration_class="test",
        anchored=True,
        unanchored_remedy="",
    )
    with pytest.raises(ConfigurationRefusal, match="nothing to construct"):
        row.build_column_backend(
            constructor="c", prep_geometry="g", kernel_cache="k",
        )
