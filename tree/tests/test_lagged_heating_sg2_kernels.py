"""SG2 on a card: the four lagged-heating kernels against binary32 host twins.

* the producer ``lagged_theta_m_rate_v842_f32`` (mpas_atmphys_interface.F:909-919);
* the consumer ``add_lagged_diabatic_theta_v842_f32`` (time_integration.F:6937):
  ``tend_rt + rho*carry``;
* the recovery inside ``recover_cells_f32`` (F:4436-4437): at stage 3 only,
  ``rtheta_p - (dt*rho)*carry`` -- a flipped sign, a removal at stage 1 or 2,
  or a removal with the carry off fails here;
* the specified-zone zero ``zero_speczone_lagged_diabatic_theta_v842_f32``
  (F:7949), through the driver's own launcher: exactly the specified-zone
  columns, every level, nothing else.

Every comparison is bitwise.  The kernels are compiled by the project's
KernelCache with its own options (no FP contraction), as the forecast does.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

F32 = np.float32
RVORD = F32(F32(461.6) / F32(287.0))


@pytest.fixture(scope="module")
def cache():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as error:  # pragma: no cover - no driver
        pytest.skip(f"no CUDA device: {error}")
    from hexcore.cuda_backend.runtime import KernelCache

    return KernelCache()


def _rng_plane(rng, shape, low, high):
    return rng.uniform(low, high, shape).astype(F32)


def test_producer_kernel_equals_its_host_twin(cache):
    from hexcore.lagged_heating_v841 import lagged_theta_m_rate, lagged_theta_m_rate_reference

    rng = np.random.default_rng(909)
    shape = (7, 129)
    before = _rng_plane(rng, shape, 280.0, 330.0)
    theta = _rng_plane(rng, shape, 270.0, 320.0)
    qv = _rng_plane(rng, shape, -1e-5, 0.02)
    qv[0, :4] = F32(-0.0)
    for dt in (18.75, 37.5, 150.0):
        device = lagged_theta_m_rate(
            theta_m_before=cp.asarray(before), theta_after=cp.asarray(theta),
            qv_after=cp.asarray(qv), dt=dt, rvord=RVORD, kernel_cache=cache)
        host = lagged_theta_m_rate_reference(before, theta, qv, dt, RVORD)
        assert cp.asnumpy(device).tobytes() == host.tobytes()


def test_producer_on_a_regional_cache_leaves_the_callers_planes_alone():
    """The 1 km regional cull's step-1 stop (2026-10-14), on a card.

    A limited-area run's KernelCache carries RegionalGarbageDiscipline.scrub
    as its post-launch observer, exactly as CudaRegionalRuntimeV841 arms it.
    Before the fix the producer launched on the caller's planes, the scrub
    zeroed the garbage column of theta_after (the post-RK update's own theta,
    which the physics preparation fills from its donor column) and
    recover_wsm6_v841_f32 refused theta = 0 there.  The real cells must equal
    the host twin, every input must come back byte for byte, and the rate's
    garbage column must hold native's allocation value 0.0.
    """

    from hexcore.cuda_backend.runtime import KernelCache
    from hexcore.cuda_regional_forecast_v841 import RegionalGarbageDiscipline
    from hexcore.lagged_heating_v841 import lagged_theta_m_rate, lagged_theta_m_rate_reference

    rng = np.random.default_rng(905)
    nlev, ncells_solve = 6, 97
    shape = (nlev, ncells_solve + 1)
    before = _rng_plane(rng, shape, 280.0, 330.0)
    theta = _rng_plane(rng, shape, 270.0, 320.0)
    qv = _rng_plane(rng, shape, 0.0, 0.02)
    before[:, -1] = F32(0.0)            # theta_m's garbage column: the pool value
    theta[:, -1] = theta[:, -2]         # th_p's garbage column: lent from the donor
    qv[:, -1] = F32(0.0)
    regional_cache = KernelCache()
    discipline = RegionalGarbageDiscipline(
        n_cells_solve=ncells_solve, n_edges_solve=3 * ncells_solve, n_vertices_solve=2 * ncells_solve)
    regional_cache.post_launch = discipline.scrub
    planes = {"before": cp.asarray(before), "theta": cp.asarray(theta), "qv": cp.asarray(qv)}
    rate = lagged_theta_m_rate(
        theta_m_before=planes["before"], theta_after=planes["theta"], qv_after=planes["qv"],
        dt=6.0, rvord=RVORD, kernel_cache=regional_cache)
    for name, host in (("before", before), ("theta", theta), ("qv", qv)):
        assert cp.asnumpy(planes[name]).tobytes() == host.tobytes(), name
    host_rate = lagged_theta_m_rate_reference(before, theta, qv, 6.0, RVORD)
    device_rate = cp.asnumpy(rate)
    assert device_rate[:, :-1].tobytes() == host_rate[:, :-1].tobytes()
    assert np.all(device_rate[:, -1] == F32(0.0))
    assert discipline.scrubs > 0


def _driver_stub(cache, nlev, ncells, regional=None):
    from hexcore.cuda_driver import CudaDryDycoreDriver

    driver = object.__new__(CudaDryDycoreDriver)
    driver.cp = cp
    driver.cache = cache
    driver.nlev = nlev
    driver.ncells = ncells
    driver._physics_driver_kernels = {}
    driver.regional_v841 = regional
    return driver


def test_consumer_adds_rho_times_carry(cache):
    rng = np.random.default_rng(6937)
    nlev, ncells = 9, 257
    rho = _rng_plane(rng, (nlev, ncells), 0.1, 1.3)
    carry = _rng_plane(rng, (nlev, ncells), -0.2, 0.2)
    tend = _rng_plane(rng, (nlev, ncells), -5.0, 5.0)
    driver = _driver_stub(cache, nlev, ncells)
    device_tend = cp.asarray(tend)
    driver._launch_physics_driver(
        "add_lagged_diabatic_theta_v842_f32", ncells,
        (np.int32(nlev), np.int32(ncells), cp.asarray(rho), cp.asarray(carry), device_tend))
    expected = (tend + (rho * carry).astype(F32)).astype(F32)
    assert cp.asnumpy(device_tend).tobytes() == expected.tobytes()


def _kernel_parameters(source, name):
    """The kernel's parameter names in order, read from its own source text, so this test builds
    the launch the source declares (the hex-mpas line carries one more scalar, native_gnu_powf)."""

    head = source.split(f'void {name}(', 1)[1].split(')', 1)[0]
    return [part.strip().replace('*', ' ').split()[-1] for part in head.split(',')]


def _recover(cache, *, stage, has_diabatic, carry, dt, inputs, nlev, ncells):
    from hexcore.cuda_driver import _CUDA_SOURCE

    kernel = cache.raw_kernel("recover_cells_f32", _CUDA_SOURCE, module_key="hexcore.cuda_driver")
    outputs = {name: cp.empty((nlev, ncells), cp.float32)
               for name in ("rho", "rtheta", "rho_p", "rtheta_p", "theta", "exner", "pressure_p")}
    scalars = {"nlev": np.int32(nlev), "ncells": np.int32(ncells), "stage": np.int32(stage),
               "native_gnu_powf": np.int32(0), "rgas": np.float32(287.0), "cp": np.float32(1004.5),
               "p0": np.float32(100_000.0), "has_diabatic": np.int32(has_diabatic), "dt": np.float32(dt),
               "diabatic_theta": cp.asarray(carry)}
    arrays = {name: cp.asarray(value) for name, value in inputs.items()}
    args = []
    for name in _kernel_parameters(_CUDA_SOURCE, "recover_cells_f32"):
        for table in (scalars, arrays, outputs):
            if name in table:
                args.append(table[name])
                break
        else:
            raise AssertionError(f"recover_cells_f32 parameter {name!r} has no test value")
    count = nlev * ncells
    kernel(((count + 127) // 128,), (128,), tuple(args))
    return {name: cp.asnumpy(value) for name, value in outputs.items()}


@pytest.fixture(scope="module")
def recovery_inputs():
    rng = np.random.default_rng(4436)
    nlev, ncells = 5, 211
    shape = (nlev, ncells)
    inputs = dict(
        rho_p_saved=_rng_plane(rng, shape, -0.01, 0.01),
        rtheta_p_saved=_rng_plane(rng, shape, -3.0, 3.0),
        rho_pp=_rng_plane(rng, shape, -1e-3, 1e-3),
        rtheta_pp=_rng_plane(rng, shape, -0.5, 0.5),
        rho_base=_rng_plane(rng, shape, 0.2, 1.2),
        rtheta_base=_rng_plane(rng, shape, 90.0, 400.0),
        exner_base=_rng_plane(rng, shape, 0.7, 1.0),
        zz=_rng_plane(rng, shape, 0.98, 1.02),
        exner_saved=_rng_plane(rng, shape, 0.7, 1.0),
        pressure_p_saved=_rng_plane(rng, shape, -50.0, 50.0),
    )
    carry = _rng_plane(rng, shape, 0.01, 0.1)    # positive: net heating
    return inputs, carry, nlev, ncells


@pytest.mark.parametrize("dt", [18.75, 150.0])
def test_final_recovery_removes_dt_rho_carry(cache, recovery_inputs, dt):
    inputs, carry, nlev, ncells = recovery_inputs
    density_p = (inputs["rho_p_saved"] + inputs["rho_pp"]).astype(F32)
    full_rho = (density_p + inputs["rho_base"]).astype(F32)
    theta_p = (inputs["rtheta_p_saved"] + inputs["rtheta_pp"]).astype(F32)
    removed = ((F32(dt) * full_rho).astype(F32) * carry).astype(F32)
    expected = (theta_p - removed).astype(F32)

    with_carry = _recover(cache, stage=3, has_diabatic=1, carry=carry, dt=dt,
                          inputs=inputs, nlev=nlev, ncells=ncells)
    without = _recover(cache, stage=3, has_diabatic=0, carry=carry, dt=dt,
                       inputs=inputs, nlev=nlev, ncells=ncells)
    assert with_carry["rtheta_p"].tobytes() == expected.tobytes()
    assert without["rtheta_p"].tobytes() == theta_p.tobytes()
    assert with_carry["rho_p"].tobytes() == density_p.tobytes()
    assert with_carry["rho"].tobytes() == full_rho.tobytes()
    # sign: heating carried through the step is taken back out at the end
    assert np.all(with_carry["rtheta_p"] < without["rtheta_p"])
    # stage 3 forms exner from the reduced rtheta
    assert np.any(with_carry["exner"] != without["exner"])


@pytest.mark.parametrize("stage", [1, 2])
def test_intermediate_recoveries_remove_nothing(cache, recovery_inputs, stage):
    inputs, carry, nlev, ncells = recovery_inputs
    with_carry = _recover(cache, stage=stage, has_diabatic=1, carry=carry, dt=150.0,
                          inputs=inputs, nlev=nlev, ncells=ncells)
    without = _recover(cache, stage=stage, has_diabatic=0, carry=carry, dt=150.0,
                       inputs=inputs, nlev=nlev, ncells=ncells)
    for name in ("rho", "rtheta", "rho_p", "rtheta_p", "exner"):
        assert with_carry[name].tobytes() == without[name].tobytes(), name


def test_specified_zone_zero_touches_only_specified_columns(cache):
    rng = np.random.default_rng(7949)
    nlev, ncells = 6, 97
    spec = np.array([0, 5, 6, 40, 96], dtype=np.int32)
    regional = SimpleNamespace(masks=SimpleNamespace(n_spec_cells=len(spec), spec_cells=cp.asarray(spec)))
    driver = _driver_stub(cache, nlev, ncells, regional)
    carry = _rng_plane(rng, (nlev, ncells), -0.3, 0.3)
    device = cp.asarray(carry)
    driver._zero_speczone_lagged_heating(device)
    expected = carry.copy()
    expected[:, spec] = F32(0.0)
    assert cp.asnumpy(device).tobytes() == expected.tobytes()


def test_specified_zone_zero_is_a_no_op_on_a_global_run(cache):
    carry = np.full((3, 11), F32(0.5))
    device = cp.asarray(carry)
    _driver_stub(cache, 3, 11, None)._zero_speczone_lagged_heating(device)
    assert cp.asnumpy(device).tobytes() == carry.tobytes()


def test_specified_zone_zero_refuses_a_plane_of_another_shape(cache):
    regional = SimpleNamespace(masks=SimpleNamespace(n_spec_cells=1, spec_cells=cp.asarray(np.array([0], np.int32))))
    driver = _driver_stub(cache, 3, 11, regional)
    with pytest.raises(ValueError, match="carry must be float32"):
        driver._zero_speczone_lagged_heating(cp.zeros((3, 10), cp.float32))
