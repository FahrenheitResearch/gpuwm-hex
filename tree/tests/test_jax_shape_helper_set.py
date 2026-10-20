"""The measured shape seam retains strict source and imported-path binding."""
import hashlib
import json
from pathlib import Path
import pytest
from hexcore import cuda_arwen_helper_scope_v2 as scope

MODE = "jax-hex-shape-a7c38f9c-source"
DEPENDENCY = "gpuwm/core/kernel_shape.py"


def test_shape_row_binds_six_helpers_and_transformer():
    assert set(scope.ADMITTED_SOURCE_SETS[MODE]) == set(scope.PUBLISHED_SOURCES)
    assert set(scope.EXECUTION_DEPENDENCIES[MODE]) == {DEPENDENCY}
    assert MODE in scope.PUBLISHED_MODES
    assert "unqualified" in scope.RUNTIME_DESCRIPTIONS[MODE]
    pins_path = Path(__file__).with_name("jax_hex_shape") / "engine-source-pins.json"
    pins = json.loads(pins_path.read_text())
    # The public sdist contains tests/, while the home replay kit stays outside tree/.
    checkout_pins = Path(__file__).resolve().parents[2] / "tools/jax_hex_shape/engine-source-pins.json"
    if checkout_pins.is_file():
        assert pins_path.read_bytes() == checkout_pins.read_bytes()
    for name in ("gpuwm/core/rrtmg_lw.py", "gpuwm/core/rrtmg_sw.py"):
        assert scope.ADMITTED_SOURCE_SETS[MODE][name] == pins["files"][name]["candidate"]
    assert scope.EXECUTION_DEPENDENCIES[MODE][DEPENDENCY] == pins["files"][DEPENDENCY]["candidate"]


def test_unknown_dependency_refuses_before_import(tmp_path, monkeypatch):
    # Use synthetic complete radiation bytes to reach the dependency gate.
    expected = {}
    for name in scope.PUBLISHED_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(scope.ADMITTED_SOURCE_SETS, MODE, expected)
    dependency = tmp_path / DEPENDENCY
    dependency.write_bytes(b"unmeasured transformer")
    with pytest.raises(RuntimeError, match="unknown or mixed gpuwm execution dependency"):
        scope.bind_arwen_helpers_v2(tmp_path)


def test_dependency_escape_refuses(tmp_path, monkeypatch):
    expected = {}
    for name in scope.PUBLISHED_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(scope.ADMITTED_SOURCE_SETS, MODE, expected)
    # Resolve outside the engine root, while keeping the fixture under tmp_path.
    root = tmp_path / "engine"
    root.mkdir()
    for name in scope.PUBLISHED_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    (root / DEPENDENCY).symlink_to(tmp_path / "gpuwm/core/rrtmg_lw.py")
    with pytest.raises(RuntimeError, match="escapes its engine tree"):
        scope.bind_arwen_helpers_v2(root)
