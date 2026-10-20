"""The engine's land-kernel modules launch on the card that calls them (hexcore.engine_per_card).

CPU test with a two-card stand-in: a CUDA module belongs to the card it was loaded on, so a module
compiled on card 0 and called on card 1 must launch card 1's own copy of the same image.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from hexcore import engine_per_card


class _Card:
    current = 0


class _FakeModule:
    def __init__(self):
        self.card = None
        self.image = None

    def load(self, image):
        self.card, self.image = _Card.current, image

    def get_function(self, name):
        module = self

        def launch(*args):
            if _Card.current != module.card:
                raise RuntimeError("CUDA_ERROR_INVALID_HANDLE")
            return (module.card, name, module.image)
        launch.ptr = 0
        return launch


def _fake_cp():
    device = lambda *a: SimpleNamespace(id=_Card.current)  # noqa: E731
    return SimpleNamespace(cuda=SimpleNamespace(Device=device, function=SimpleNamespace(Module=_FakeModule)))


def _fake_ieee():
    ieee = ModuleType("fake_ieee_module")

    class _LoadedModule:
        def __init__(self, module, image):
            self.module, self.cubin = module, image

        def get_function(self, name):
            return _Function(self.module.get_function(name))

    class _Function:
        def __init__(self, function):
            self.function = function

        def __call__(self, *args):
            return self.function(*args)

    def compile_module(image):
        loaded = _FakeModule()
        loaded.load(image)
        return _LoadedModule(loaded, image)

    ieee._LoadedModule, ieee._Function, ieee.compile_module = _LoadedModule, _Function, compile_module
    return ieee


@pytest.fixture
def bound(monkeypatch):
    monkeypatch.setattr(engine_per_card, "_INSTALLED", False)
    ieee = _fake_ieee()
    _Card.current = 0
    assert engine_per_card.bind_engine_modules_per_card(_fake_cp(), ieee_module=ieee)
    return ieee


def test_without_the_binding_the_second_card_launches_the_first_cards_kernel():
    ieee = _fake_ieee()
    _Card.current = 0
    kernel = ieee.compile_module(b"glacier").get_function("noahmp_glacier_column")
    _Card.current = 1
    with pytest.raises(RuntimeError, match="INVALID_HANDLE"):
        kernel()


def test_each_card_launches_its_own_copy_of_the_same_image(bound):
    _Card.current = 0
    module = bound.compile_module(b"glacier")
    kernel = module.get_function("noahmp_glacier_column")  # held once, as the engine's caches hold it
    assert kernel() == (0, "noahmp_glacier_column", b"glacier")
    _Card.current = 1
    assert kernel() == (1, "noahmp_glacier_column", b"glacier")
    assert module.get_function("noahmp_glacier_column")() == (1, "noahmp_glacier_column", b"glacier")
    _Card.current = 0
    assert kernel() == (0, "noahmp_glacier_column", b"glacier")
    assert set(module._hex_copies) == {1}  # card 0 keeps the module it was compiled on


def test_a_changed_engine_shape_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(engine_per_card, "_INSTALLED", False)
    ieee = ModuleType("fake_ieee_module")
    with pytest.raises(RuntimeError, match="CUDA_ERROR_INVALID_HANDLE"):
        engine_per_card.bind_engine_modules_per_card(_fake_cp(), ieee_module=ieee)
