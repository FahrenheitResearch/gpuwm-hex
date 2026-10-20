"""The public dry-step return must become the next committed atmosphere."""

import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from hexcore.partition_samehost_v841 import SameHostRanks

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from check_samehost_dycore_v841 import _ensure_deformation, step_and_publish  # noqa: E402


class ReturningDriver:
    """Match step_device's API: return new storage without changing the owner."""

    def __init__(self):
        self.atmosphere = SimpleNamespace(state=SimpleNamespace(time_seconds=0.))
        self.results = []

    def step_device(self):
        result = SimpleNamespace(
            atmosphere=SimpleNamespace(state=SimpleNamespace(
                time_seconds=self.atmosphere.state.time_seconds + 20.,
            )),
            receipt=SimpleNamespace(index=len(self.results) + 1),
        )
        self.results.append(result)
        return result


def test_reference_steps_chain_returned_storage_and_advance_clock():
    driver = ReturningDriver()
    initial = driver.atmosphere
    first = step_and_publish(driver)
    assert driver.atmosphere is first.atmosphere
    assert initial.state.time_seconds == 0.
    second = step_and_publish(driver)
    assert driver.atmosphere is second.atmosphere
    assert second.atmosphere.state.time_seconds == 40.
    assert first.atmosphere.state.time_seconds == 20.
    assert second.receipt.index == 2


def test_concurrent_ranks_publish_before_boundary_and_clock_check():
    drivers = [ReturningDriver(), ReturningDriver()]
    published = [[], []]
    aborted = []
    transport = SimpleNamespace(abort=aborted.append)

    def advance(rank):
        result = step_and_publish(drivers[rank])
        # Model the A boundary's consumer of the driver's committed state.
        published[rank].append(drivers[rank].atmosphere.state.time_seconds)
        return result.receipt.index

    with SameHostRanks([0, 0], transport) as ranks:
        clock = lambda rank: drivers[rank].atmosphere.state.time_seconds
        assert ranks.step(advance, clock, dt=20.) == [1, 1]
        assert ranks.time_seconds == 20.
        assert ranks.step(advance, clock, dt=20.) == [2, 2]
        assert ranks.time_seconds == 40.
    assert published == [[20., 40.], [20., 40.]]
    assert not aborted


def test_initialized_deformation_pair_is_retained_without_redeclaration():
    arrays = {name: np.arange(6, dtype=np.float32).reshape(3, 2)
              for name in ("defc_a", "defc_b")}
    originals = dict(arrays)
    _ensure_deformation(SimpleNamespace(arrays=arrays))
    assert arrays == originals
    assert all(arrays[name] is value for name, value in originals.items())


def test_partial_initialized_deformation_pair_is_refused_without_mutation():
    arrays = {"defc_a": np.zeros((3, 2), np.float32)}
    original = arrays["defc_a"]
    with pytest.raises(ValueError, match="both defc_a and defc_b"):
        _ensure_deformation(SimpleNamespace(arrays=arrays))
    assert set(arrays) == {"defc_a"}
    assert arrays["defc_a"] is original
