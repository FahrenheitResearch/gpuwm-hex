"""The full-step projection probe's composition arithmetic, pinned by hand cases.

The probe (``tools/probe_lts_fullstep_projection.py``) composes measured
per-class launch costs into a projected whole-step speedup.  The GPU timing
half needs a card; the composition half is pure arithmetic, and a wrong
composition would turn a measured launch-cost table into a wrong go/no-go
verdict while every gate stayed green.  These are the hand cases that pin it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from probe_lts_fullstep_projection import (  # noqa: E402
    arithmetic_cell_step_speedup,
    compose,
    macro_step,
    self_test,
)


def test_probe_self_test_passes() -> None:
    self_test()


def test_macro_step_is_the_lcm() -> None:
    assert macro_step((1,)) == 1
    assert macro_step((1, 2, 4)) == 4
    assert macro_step((1, 2, 3)) == 6


def test_compose_two_classes_hand_case() -> None:
    # Macro step 2: the rate-1 class launches twice, the rate-2 class once,
    # the global arm launches the whole mesh twice.  Class launches at 1.0 ms
    # against a whole-mesh launch at 2.0 ms: (2 * 2.0) / (2 * 1.0 + 1 * 1.0).
    result = compose((1, 2), {1: 1.0, 2: 1.0}, 2.0)
    assert result["macro"] == 2.0
    assert result["global_cost"] == pytest.approx(4.0)
    assert result["lts_cost"] == pytest.approx(3.0)
    assert result["projected_speedup"] == pytest.approx(4.0 / 3.0)


def test_compose_single_class_is_the_global_arm() -> None:
    result = compose((1,), {1: 3.0}, 3.0)
    assert result["projected_speedup"] == pytest.approx(1.0)


def test_compose_launch_floor_erases_the_prize() -> None:
    # A class whose launch costs the same as the whole mesh saves nothing:
    # macro 2, fine class as dear as the full launch -> LTS arm 3 units
    # against the global arm's 2, a projected SLOWDOWN.  This is the box-mesh
    # regime the probe measured.
    result = compose((1, 2), {1: 1.0, 2: 1.0}, 1.0)
    assert result["projected_speedup"] == pytest.approx(2.0 / 3.0)


def test_arithmetic_bound_hand_cases() -> None:
    assert arithmetic_cell_step_speedup((1, 4), {1: 2, 4: 2}) == pytest.approx(1.6)
    assert arithmetic_cell_step_speedup((1,), {1: 7}) == pytest.approx(1.0)
    # Half the cells at rate 2: 8 cell-steps become 6.
    assert arithmetic_cell_step_speedup((1, 2), {1: 2, 2: 2}) == pytest.approx(8.0 / 6.0)
