"""SG2 on a limited-area mesh: the producer must not hand its caller's planes to the garbage scrub.

THE BREAKAGE, MEASURED 2026-10-14.  The 1 km regional cull q1.128.38027.n41.86w87.70 (dt 6 s,
``--convection off``, the 10-13 transferred derecho analysis, hourly boundaries) stopped in its
FIRST step on integrate/hex-mpas with "post-RK WSM6 numeric validation refused the candidate
state", in correct and in exact mode alike.  The refused candidate had theta = 0 at all 55
levels of column 38027, the padded garbage column, and nothing else wrong.  The mechanism:

* a limited-area run arms ``RegionalGarbageDiscipline.scrub`` as the shared KernelCache's
  post-launch observer; after every launch of a module not in ``SELF_MANAGED_GARBAGE_MODULES``
  it writes the pool value 0.0 into the garbage column of every padded float32 argument;
* the SG2 producer ``lagged_theta_m_rate`` (``hexcore.lagged_heating_v841``, not in that set)
  launched on the caller's planes, and its ``theta_after`` IS the post-RK update's theta
  (``prepared.th_p``), whose garbage column the physics preparation fills from its donor column;
* so the scrub zeroed th_p there between the producer and ``recover_wsm6_v841_f32``, which
  refuses theta <= 0 at every column.

Exempting the producer's module from the scrub let the same run go 150 steps (15 min).  The fix
keeps the module under the scrub, because the scrub's 0.0 is native's value for the OUTPUT's
garbage column (``rt_diabatic_tend`` is written only for i = its,ite), and launches on private
copies of the inputs when the cache carries an observer.

This test uses the project's own discipline and post-launch wrapper with numpy standing in for
the card, so it runs on every host.  It fails on the pre-fix producer (theta_after's garbage
column comes back 0.0) and passes with the fix.  The same contract on a card is
``test_lagged_heating_sg2_kernels.py::test_producer_on_a_regional_cache_leaves_the_callers_planes_alone``.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import _layout  # noqa: F401  (puts this tree's package on the path)

from hexcore.cuda_backend.runtime import _PostLaunchKernel
from hexcore.cuda_regional_forecast_v841 import (
    SELF_MANAGED_GARBAGE_MODULES,
    RegionalGarbageDiscipline,
)
from hexcore.lagged_heating_v841 import (
    LAGGED_HEATING_SOURCE,
    lagged_theta_m_rate,
    lagged_theta_m_rate_reference,
)

F32 = np.float32
RVORD = F32(F32(461.6) / F32(287.0))  # hexcore.cuda_physics_v841.RV_OVER_RD_F32
NLEV, NCELLS_SOLVE = 5, 23            # one garbage column at index NCELLS_SOLVE


class _DeviceArray(np.ndarray):
    """A numpy plane with the two device attributes the scrub and the producer read."""

    @property
    def data(self):  # noqa: D401 - the cupy MemoryPointer shape: .ptr
        return types.SimpleNamespace(ptr=int(self.__array_interface__["data"][0]))

    def get(self):
        return np.asarray(self)


def _device(value):
    return np.asarray(value).view(_DeviceArray)


class _NumpyCupy(types.ModuleType):
    """Stands in for cupy: same memory semantics (ascontiguousarray aliases, array(copy) copies)."""

    ndarray = _DeviceArray
    float32 = np.float32
    int32 = np.int32

    def __init__(self) -> None:
        super().__init__("cupy")

    @staticmethod
    def ascontiguousarray(value):
        return _device(np.ascontiguousarray(value))

    @staticmethod
    def array(value, dtype=None, copy=True, order="K"):
        return _device(np.array(value, dtype=dtype, copy=copy, order=order))

    @staticmethod
    def empty(shape, dtype=None):
        return _device(np.empty(shape, dtype=dtype))

    @staticmethod
    def zeros(shape, dtype=None):
        return _device(np.zeros(shape, dtype=dtype))

    @staticmethod
    def equal(a, b):
        return np.equal(a, b)


def _host_producer(grid, block, args):
    """lagged_theta_m_rate_v842_f32 on the host: its binary32 twin, written into ``rate``."""

    count, rvord, dt, before, theta, qv, rate, invalid = args
    value = lagged_theta_m_rate_reference(before, theta, qv, float(dt), rvord)
    rate[...] = value
    if not np.all(np.isfinite(value)):
        invalid[0] = 1


class _ObservedCache:
    """The KernelCache surface the producer uses, armed the way the regional runtime arms it."""

    def __init__(self, observer) -> None:
        self.post_launch = observer
        self.resolved: list[str] = []

    def raw_kernel(self, name, source, *, module_key):
        assert source is LAGGED_HEATING_SOURCE
        self.resolved.append(module_key)
        return _PostLaunchKernel(_host_producer, self, name, module_key)


@pytest.fixture
def numpy_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _NumpyCupy())


def _planes():
    rng = np.random.default_rng(905)
    shape = (NLEV, NCELLS_SOLVE + 1)
    before = rng.uniform(280.0, 330.0, shape).astype(F32)
    theta = rng.uniform(270.0, 320.0, shape).astype(F32)
    qv = rng.uniform(0.0, 0.02, shape).astype(F32)
    before[:, -1] = F32(0.0)        # saved.theta_m's garbage column: the pool value
    theta[:, -1] = theta[:, -2]     # th_p's garbage column: lent from the donor column
    qv[:, -1] = F32(0.0)            # the candidate scalars' garbage column: the pool value
    return before, theta, qv


def _discipline():
    return RegionalGarbageDiscipline(
        n_cells_solve=NCELLS_SOLVE,
        n_edges_solve=3 * NCELLS_SOLVE,
        n_vertices_solve=2 * NCELLS_SOLVE,
    )


def test_producer_runs_under_the_regional_scrub(numpy_cupy):
    """The producer is NOT a self-managed module: its output wants the scrub's pool value."""

    assert "hexcore.lagged_heating_v841" not in SELF_MANAGED_GARBAGE_MODULES
    discipline = _discipline()
    cache = _ObservedCache(discipline.scrub)
    before, theta, qv = (_device(p.copy()) for p in _planes())
    lagged_theta_m_rate(theta_m_before=before, theta_after=theta, qv_after=qv,
                        dt=6.0, rvord=RVORD, kernel_cache=cache)
    assert cache.resolved == ["hexcore.lagged_heating_v841"]
    assert discipline.scrubs > 0


def test_regional_scrub_leaves_the_callers_planes_alone(numpy_cupy):
    """The step-1 refusal: theta_after's garbage column must survive the producer's launch."""

    before_h, theta_h, qv_h = _planes()
    discipline = _discipline()
    cache = _ObservedCache(discipline.scrub)
    before, theta, qv = _device(before_h.copy()), _device(theta_h.copy()), _device(qv_h.copy())
    rate = lagged_theta_m_rate(theta_m_before=before, theta_after=theta, qv_after=qv,
                               dt=6.0, rvord=RVORD, kernel_cache=cache)
    # what recover_wsm6_v841_f32 reads next: the donor's theta, not the pool's 0 K
    assert np.all(np.asarray(theta)[:, -1] > F32(0.0))
    assert np.asarray(theta).tobytes() == theta_h.tobytes()
    assert np.asarray(before).tobytes() == before_h.tobytes()
    assert np.asarray(qv).tobytes() == qv_h.tobytes()
    # the real cells carry native's rate; the garbage column carries native's allocation value
    host = lagged_theta_m_rate_reference(before_h, theta_h, qv_h, 6.0, RVORD)
    assert np.asarray(rate)[:, :-1].tobytes() == host[:, :-1].tobytes()
    assert np.all(np.asarray(rate)[:, -1] == F32(0.0))


def test_global_cache_launches_on_the_callers_planes(numpy_cupy):
    """No observer (every global run): no private copies, the launch takes the caller's planes."""

    seen = {}

    def kernel(grid, block, args):
        seen["theta"] = args[4]
        _host_producer(grid, block, args)

    class _GlobalCache:
        post_launch = None

        def raw_kernel(self, name, source, *, module_key):
            return kernel

    before, theta, qv = (_device(p.copy()) for p in _planes())
    rate = lagged_theta_m_rate(theta_m_before=before, theta_after=theta, qv_after=qv,
                               dt=6.0, rvord=RVORD, kernel_cache=_GlobalCache())
    assert np.shares_memory(seen["theta"], theta)
    host = lagged_theta_m_rate_reference(np.asarray(before), np.asarray(theta), np.asarray(qv),
                                         6.0, RVORD)
    assert np.asarray(rate).tobytes() == host.tobytes()
