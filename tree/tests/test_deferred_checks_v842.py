"""U-MPASPHYS-1: slot refusals keep their message and exception type, immediate or deferred."""
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - CPU tier
    pytest.skip("no CUDA device", allow_module_level=True)

from hexcore.mpasphys.deferred_checks_v842 import DeferredChecks, collecting, defer


def test_immediate_outside_a_forecast():
    defer(cp.zeros(4, dtype=cp.int32), "never")
    with pytest.raises(ValueError, match="line 7"):
        defer(cp.array([0, 7, 3], dtype=cp.int32), "line {value}", error=ValueError)


def test_deferred_inside_a_forecast_raises_at_flush_or_finish():
    checks = DeferredChecks(cp)
    with collecting(checks):
        defer(cp.zeros(3, dtype=cp.int32), "clean")
        checks.flush(1)
        flag = cp.array([0, 5, 2], dtype=cp.int32)
        defer(flag, lambda v: f"first {v}", reduce="first", error=RuntimeError)
        flag[...] = 0          # later writes do not change the captured word
        checks.flush(2)
        with pytest.raises(RuntimeError, match=r"first 5 \(deferred device check, step 2\)"):
            checks.finish()
    assert checks.flushed == 2 and checks.words == 2
