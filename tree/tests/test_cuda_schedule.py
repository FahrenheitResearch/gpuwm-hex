"""The host thread waits for the card with blocking sync, not CUDA's spin (cuda_backend/schedule.py).

Breakage it prevents: a GPU-bound forecast holding one host core at 100 % for its whole step loop
(the hex main thread spin-wait, SINGLE-CORE-FIXES-2026-10-26).
"""

import ctypes

import pytest

from hexcore.cuda_backend import schedule


class FakeDriver:
    def __init__(self, *, ctx=0x1234, flags=0x08, has_set=True, primary_rc=0):
        self.ctx, self.flags, self.primary_rc = ctx, flags, primary_rc
        self.primary = []
        if has_set:
            self.cuCtxSetFlags = self._set

    def cuCtxGetCurrent(self, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = self.ctx or None
        return 0

    def cuCtxGetFlags(self, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint))[0] = self.flags
        return 0

    def _set(self, flags):
        self.flags = flags.value
        return 0

    def cuDeviceGet(self, out, ordinal):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = ordinal
        return 0

    def cuDevicePrimaryCtxSetFlags(self, dev, flags):
        self.primary.append(flags.value)
        return self.primary_rc


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(schedule, "_done", {})
    monkeypatch.setattr(schedule, "_warned", False)


def test_default_is_blocking_and_keeps_other_context_flags():
    driver = FakeDriver(flags=0x08)  # LMEM_RESIZE_TO_MAX, auto scheduling
    assert schedule.apply(0, environ={}, driver=driver) == "blocking (context)"
    assert driver.flags == 0x08 | 0x04


def test_once_per_context():
    driver = FakeDriver(flags=0x01)
    schedule.apply(0, environ={}, driver=driver)
    driver.flags = 0x01  # a second call must not touch the context again
    assert schedule.apply(0, environ={}, driver=driver) == "blocking (context)"
    assert driver.flags == 0x01


def test_old_driver_falls_back_to_the_primary_context():
    driver = FakeDriver(ctx=0, has_set=False)
    assert schedule.apply(2, environ={}, driver=driver) == "blocking (primary context)"
    assert driver.primary == [0x04]


def test_neither_route_keeps_the_default_and_says_so(capsys):
    driver = FakeDriver(has_set=False, primary_rc=708)
    assert schedule.apply(0, environ={}, driver=driver) == "unchanged"
    assert "GPUWM_HEX_CUDA_SCHEDULE" in capsys.readouterr().err


def test_knob():
    driver = FakeDriver(flags=0x04)
    assert schedule.apply(0, environ={"GPUWM_HEX_CUDA_SCHEDULE": "auto"}, driver=driver) == "auto"
    assert driver.flags == 0x04
    assert schedule.apply(0, environ={"GPUWM_HEX_CUDA_SCHEDULE": "spin"}, driver=driver) == "spin (context)"
    assert driver.flags == 0x01
    with pytest.raises(ValueError, match="GPUWM_HEX_CUDA_SCHEDULE"):
        schedule.requested({"GPUWM_HEX_CUDA_SCHEDULE": "fast"})
