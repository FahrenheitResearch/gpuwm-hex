"""The per-kernel capture-and-replay instrument, held to what its first run
on a card got wrong (2026-09-14, the point cull, 49 kernels).

Two instrument defects reported five kernels as differing whose arithmetic
was byte-identical, and each has a CPU test here so it cannot come back:

* the owner count of a kernel without an ``OWNER_COUNT_ARG`` rule is the
  trailing dimension of its LAST device argument, and one target's last
  device argument was a (1,) flag, so its baseline ran at one owner.  Every
  target's CUDA signature is parsed and the argument the rule resolves to
  must be a scalar ``int`` count, or, with no rule, the last pointer
  parameter must be a field and never an ``int`` array;
* the replay applied the garbage discipline's scrub with the step's unit
  pool released, writing 0.0 where the live launch wrote 1.0.  The recorder
  now asks the discipline which arguments are bound at the launch and the
  replay re-binds them before every launch; the helpers that do that, and
  the classifier that splits a difference into garbage-column and live
  values, are exercised on host arrays with a stand-in discipline.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

ab = importlib.import_module("kernel_element_ab")


# -- the CUDA signature of every target ---------------------------------------

_DIRECT = re.compile(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)', re.S)
_INSTANCE = re.compile(r'(DECLARE_\w+)\((\w+),\s*(\w+)\)')


def _macro_bodies(source: str) -> dict[str, str]:
    """``#define DECLARE_X(NAME, T)`` -> its body with continuations joined."""

    bodies: dict[str, str] = {}
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        head = re.match(r'#define\s+(DECLARE_\w+)\(NAME,\s*T\)', lines[i])
        if head is None:
            i += 1
            continue
        chunks = []
        while i < len(lines) and lines[i].rstrip().endswith("\\"):
            chunks.append(lines[i].rstrip()[:-1])
            i += 1
        if i < len(lines):
            chunks.append(lines[i])
        bodies[head.group(1)] = "\n".join(chunks)
        i += 1
    return bodies


def _signature(source: str, name: str) -> list[str]:
    """The parameter list of kernel ``name`` in ``source``, one string each."""

    for found, params in _DIRECT.findall(source):
        if found == name:
            return [p.strip() for p in params.replace("\n", " ").split(",") if p.strip()]
    bodies = _macro_bodies(source)
    for macro, instance, _ in _INSTANCE.findall(source):
        if instance != name or macro not in bodies:
            continue
        for found, params in _DIRECT.findall(bodies[macro]):
            if found == "NAME":
                return [p.strip() for p in params.replace("\n", " ").split(",") if p.strip()]
    raise AssertionError(f"no CUDA signature for {name}")


def _is_pointer(param: str) -> bool:
    return "*" in param


def _is_scalar_int(param: str) -> bool:
    return re.fullmatch(r'(const\s+)?int\s+\w+', param) is not None


def _targets() -> list[tuple[str, str]]:
    return [
        (module_key, name)
        for module_key, (_, names) in ab.TARGETS.items()
        for name in names
    ]


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    table = {}
    for module_key, (module_name, attribute) in ab.SOURCE_ATTRIBUTES.items():
        table[module_key] = getattr(ab.import_unit(module_name), attribute)
    return table


@pytest.mark.parametrize("module_key,name", _targets())
def test_the_owner_count_of_every_target_is_a_count_and_never_a_flag(sources, module_key, name):
    params = _signature(sources[module_key], name)
    rule = ab.OWNER_COUNT_ARG.get(name)
    if rule is not None:
        indices = rule if isinstance(rule, tuple) else (rule,)
        for index in indices:
            assert _is_scalar_int(params[index]), (
                f"{name}: OWNER_COUNT_ARG names argument {index}, which is "
                f"'{params[index]}' and not a scalar int count"
            )
        return
    pointers = [p for p in params if _is_pointer(p)]
    assert pointers, f"{name}: no pointer parameter to take an owner count from"
    last = pointers[-1]
    assert not re.match(r'(const\s+)?int\s*\*', last), (
        f"{name}: its last device argument is '{last}', an int array (a flag "
        "or an index list), so the instrument would launch its baseline at "
        "that array's length; name the owner count in OWNER_COUNT_ARG"
    )


def test_the_acoustic_ru_rule_names_the_edge_count(sources):
    params = _signature(sources["hexcore.cuda_regional_v841"], "acoustic_ru_regional_v841")
    assert params[ab.OWNER_COUNT_ARG["acoustic_ru_regional_v841"]] == "const int nedges"
    assert params[-1] == "int *invalid"


# -- the garbage-column classifier and the unit-pool replay ---------------------


def test_identical_bytes_report_nothing():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert ab._compare_host(0, a, a.copy(), {4: 3}) is None


def test_a_difference_is_split_into_garbage_column_and_live_values():
    n_solve = 5
    a = np.zeros((3, n_solve + 1), dtype=np.float32)
    b = a.copy()
    b[:, n_solve] = 1.0          # the garbage column, every level
    b[1, 2] = 0.25               # one live value
    report = ab._compare_host(7, a, b, {n_solve + 1: n_solve})
    assert report["argument"] == 7
    assert report["differing_values"] == 4
    assert report["garbage_column"] == n_solve
    assert report["differing_values_in_garbage_column"] == 3
    assert report["differing_values_outside_garbage_column"] == 1
    assert report["max_abs_difference"] == 1.0
    assert ab._outside_garbage_columns([report]) == 1


def test_an_unpadded_array_has_no_garbage_column_split():
    a = np.zeros((2, 7), dtype=np.float32)
    b = a.copy()
    b[0, 0] = -1.0
    report = ab._compare_host(0, a, b, {6: 5})
    assert "garbage_column" not in report
    assert ab._outside_garbage_columns([report]) == 1


def test_negative_zero_and_nan_payloads_count_as_different_bits():
    # +0.0 and a NaN with a zero payload, against -0.0 and a NaN whose
    # payload differs by one bit: equal under ==, different as bytes.
    a = np.frombuffer(b"\x00\x00\x00\x00" b"\x00\x00\xc0\x7f", dtype=np.float32).reshape(1, 2)
    b = np.frombuffer(b"\x00\x00\x00\x80" b"\x01\x00\xc0\x7f", dtype=np.float32).reshape(1, 2)
    report = ab._compare_host(0, a, b, None)
    assert report["differing_values"] == 2


class _Discipline:
    """A stand-in with the discipline's registry surface."""

    def __init__(self, n_cells: int, n_edges: int, n_vertices: int) -> None:
        self.n_cells_solve = n_cells
        self.n_edges_solve = n_edges
        self.n_vertices_solve = n_vertices
        self.bound: set[int] = set()
        self.calls: list[str] = []

    def bound_to_unit_pool(self, array) -> bool:
        return id(array) in self.bound

    def bind_unit_pool(self, array, *, permanent: bool = False):
        self.bound.add(id(array))
        self.calls.append(f"bind:{id(array)}")
        return array

    def release_unit_pool(self) -> None:
        self.bound.clear()
        self.calls.append("release")


def test_the_recorder_asks_which_arguments_are_bound_at_the_launch():
    rho = np.ones((2, 6), dtype=np.float32)
    other = np.zeros((2, 6), dtype=np.float32)
    d = _Discipline(5, 9, 13)
    d.bind_unit_pool(rho)
    args = (np.int32(2), rho, other, np.float32(0.5))
    is_device = lambda v: isinstance(v, np.ndarray)  # noqa: E731
    assert ab._unit_pool_positions(args, [d], is_device) == [1]
    d.release_unit_pool()
    assert ab._unit_pool_positions(args, [d], is_device) == []


def test_the_replay_rebinds_before_a_launch_and_releases_after():
    rho = np.ones((2, 6), dtype=np.float32)
    live = [np.int32(2), rho, np.zeros((2, 6), dtype=np.float32)]
    d = _Discipline(5, 9, 13)
    ab._rebind([d], [1], live)
    assert d.calls == ["release", f"bind:{id(rho)}"]
    assert d.bound_to_unit_pool(rho)
    ab._release([d])
    assert not d.bound_to_unit_pool(rho)


def test_the_extent_table_is_the_disciplines_own():
    d = _Discipline(43884, 131956, 88073)
    assert ab._extents([d]) == {43885: 43884, 131957: 131956, 88074: 88073}


def test_disciplines_are_found_through_the_armed_hook():
    class Cache:
        post_launch = None

    d = _Discipline(1, 2, 3)

    class Armed:
        post_launch = d.bind_unit_pool

    assert ab._disciplines([Cache()]) == []
    assert ab._disciplines([Armed(), Armed()]) == [d]
