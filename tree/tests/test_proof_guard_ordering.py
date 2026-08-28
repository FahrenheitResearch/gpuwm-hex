"""#232: every tool verifies the source pins BEFORE the checkout guard.

``verify_arwen_checkout_git`` gates on ``ARWEN_SOURCE_MANIFEST``, which
``arwen_source_manifest()`` imports lazily from
``hexcore.cuda_arwen_physics_v841`` -- a module that is itself pinned in
``EXECUTION_SOURCE_PINS``.  Calling the guard first therefore trusts that
module's constants before its bytes have been proven: an edited adapter
module can hand the guard a manifest that admits the edited seam, and the
receipt then records a checkout as "unchanged" on the strength of a hash
table the run never verified.

The #228 redesign applied the pins-then-guard ordering to the F001 proof's
``main`` and restart worker only.  These tests pin the ordering as a
property of every tool in the tree, so a new tool cannot reintroduce the
inversion, and prove it at runtime on a real secondary tool.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

PINS = "require_frozen_execution_sources"
GUARD = "verify_arwen_checkout_git"


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _scopes(tree: ast.Module):
    """Yield (label, body-node) for the module scope and every function."""

    yield "<module>", tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _own_calls(scope: ast.AST):
    """Walk ``scope`` without descending into nested function/class bodies.

    Each scope is then exactly its own statements, so a tool is reported
    once at the scope that actually owns the call.
    """

    for child in ast.iter_child_nodes(scope):
        if isinstance(child, _NESTED) and child is not scope:
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _own_calls(child)


def _first_call_lines(scope: ast.AST) -> tuple[int | None, int | None]:
    """Lowest line of a PINS call and of a GUARD call owned by ``scope``."""

    pins: list[int] = []
    guard: list[int] = []
    for node in _own_calls(scope):
        name = _called_name(node)
        if name == PINS:
            pins.append(node.lineno)
        elif name == GUARD:
            guard.append(node.lineno)
    return (min(pins) if pins else None, min(guard) if guard else None)


def _tool_sources() -> list[Path]:
    return sorted(path for path in TOOLS.rglob("*.py") if "__pycache__" not in path.parts)


def _ordering_offenders() -> list[str]:
    offenders: list[str] = []
    for path in _tool_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for label, scope in _scopes(tree):
            pins_line, guard_line = _first_call_lines(scope)
            if guard_line is None:
                continue
            if pins_line is None:
                offenders.append(
                    f"{path.relative_to(ROOT).as_posix()}::{label} calls {GUARD} "
                    f"at line {guard_line} and never calls {PINS}"
                )
            elif pins_line > guard_line:
                offenders.append(
                    f"{path.relative_to(ROOT).as_posix()}::{label} calls {GUARD} "
                    f"at line {guard_line} before {PINS} at line {pins_line}"
                )
    return offenders


def test_every_tool_verifies_source_pins_before_the_checkout_guard() -> None:
    offenders = _ordering_offenders()
    assert offenders == [], "pins must be verified before the checkout guard:\n" + "\n".join(
        offenders
    )


def test_the_ordering_scan_finds_a_deliberately_inverted_scope(tmp_path: Path) -> None:
    """Validate the instrument: an inverted scope must be caught."""

    inverted = ast.parse(
        "def run(checkout):\n"
        "    runner.verify_arwen_checkout_git(checkout)\n"
        "    runner.require_frozen_execution_sources()\n"
    )
    scope = next(scope for label, scope in _scopes(inverted) if label == "run")
    pins_line, guard_line = _first_call_lines(scope)
    assert (pins_line, guard_line) == (3, 2)
    assert pins_line > guard_line

    correct = ast.parse(
        "def run(checkout):\n"
        "    runner.require_frozen_execution_sources()\n"
        "    runner.verify_arwen_checkout_git(checkout)\n"
    )
    scope = next(scope for label, scope in _scopes(correct) if label == "run")
    pins_line, guard_line = _first_call_lines(scope)
    assert pins_line < guard_line


def test_the_scan_actually_reaches_the_secondary_tools() -> None:
    """A scan that silently covered nothing would pass forever."""

    covered: set[str] = set()
    for path in _tool_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for _, scope in _scopes(tree):
            _, guard_line = _first_call_lines(scope)
            if guard_line is not None:
                covered.add(path.name)
    expected = {
        "run_cuda_v841_full_physics_x4.py",
        "run_cuda_v841_forecast.py",
        "run_cuda_v841_forecast_2gpu.py",
        "diagnose_v841_restart_step16_x4.py",
        "probe_v841_lw_purity.py",
        "probe_v841_lw_purity_lwfix.py",
        "probe_v841_lw_purity_lwfix2.py",
        "probe_v841_lw_localize.py",
        "probe_v841_lw_capture_engine_inputs.py",
        "repro_v841_lw_engine_poison.py",
    }
    assert expected <= covered


def _load_tool(name: str):
    module_name = f"_test_ordering_{name}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.mark.parametrize(
    "tool",
    ["probe_v841_lw_purity", "probe_v841_lw_localize", "probe_v841_lw_capture_engine_inputs"],
)
def test_secondary_tool_calls_pins_first_at_runtime(
    tool: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the shipped tool's own ``main`` and record the real call order."""

    module = _load_tool(tool)
    runner = module.runner
    order: list[str] = []
    monkeypatch.setattr(
        runner, PINS, lambda: order.append(PINS) or {"files": {}, "sha256": "stub"}
    )
    monkeypatch.setattr(
        runner, GUARD, lambda checkout: order.append(GUARD) or {"root": str(checkout)}
    )

    argv = [
        "--checkpoint", str(tmp_path / "absent.pkl"),
        "--cache-root", str(tmp_path / "cache"),
        "--dump-root", str(tmp_path / "dump"),
        "--arwen-checkout", str(tmp_path),
        "--assets-root", str(tmp_path / "assets"),
    ]
    if tool == "probe_v841_lw_capture_engine_inputs":
        argv += ["--out", str(tmp_path / "out.json")]

    # The run dies at the FIRST missing authority file, which is downstream of
    # both guards -- so whatever landed in ``order`` is the shipped ordering.
    with pytest.raises(BaseException):
        module.main(argv)

    assert order[:2] == [PINS, GUARD], f"{tool} call order was {order}"
