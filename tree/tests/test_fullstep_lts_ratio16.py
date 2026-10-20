"""Matched interface flux and the rate-32 ladder of active full-step LTS, on the even-N clock.

CPU tests of the interval law for N = 2, 4 and 8 acoustic sub-steps, plus one
GPU test (HEX_LTS_GPU_TESTS=1) of the record/keep/impose and heat-impose
kernels' arithmetic at N = 4.  The clock counts ticks of base_dt / (3 N): a
class of effective rate r steps 3 N r ticks and every acoustic sub-step of it
spans exactly one plane (r ticks).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from hexcore.fullstep_lts_classes import (
    DYADIC,
    child_flux_intervals,
    grade_rates,
    macro_schedule,
    plan_macros,
    planes_per_step,
    trajectory_plane,
)

EVEN_N = (2, 4, 8)


@pytest.mark.parametrize("substeps", EVEN_N)
@pytest.mark.parametrize("parent", [2, 4, 8, 16, 32])
def test_a_dyadic_childs_splits_cover_each_plane_of_its_parents_step_once(parent, substeps):
    planes = planes_per_step(substeps)
    child = parent // 2
    start = planes * parent * 3                # any parent step boundary
    seen = []
    for split in range(6):                    # two child steps, three splits each
        end = start + substeps * child * (split + 1)
        span = list(child_flux_intervals(start, parent, end, substeps * child, planes))
        assert len(span) == substeps // 2
        seen.extend(span)
    assert seen == list(range(planes))


@pytest.mark.parametrize("substeps", EVEN_N)
def test_a_child_sharing_its_parents_step_spans_one_split_of_planes(substeps):
    # an event tail: both effective rates equal, three splits cover the 3 N planes
    planes = planes_per_step(substeps)
    start, length = 16 * planes, 4
    spans = [list(child_flux_intervals(start, length, start + substeps * length * (s + 1),
                                       substeps * length, planes)) for s in range(3)]
    assert spans == [list(range(s * substeps, (s + 1) * substeps)) for s in range(3)]


def test_the_two_substep_clock_is_the_original_sixths():
    # N = 2 (planes = 6): one plane per dyadic child split, two in an event tail
    assert [list(child_flux_intervals(48, 8, 48 + 8 * (s + 1), 8, 6)) for s in range(6)] == [[s] for s in range(6)]
    assert list(child_flux_intervals(48, 4, 48 + 8, 8, 6)) == [0, 1]


def test_a_split_off_the_parents_planes_is_refused():
    with pytest.raises(ValueError):
        child_flux_intervals(0, 8, 7, 8, 6)          # ends between planes
    with pytest.raises(ValueError):
        child_flux_intervals(0, 8, 8, 12, 6)         # a split that does not tile the planes
    with pytest.raises(ValueError):
        child_flux_intervals(16, 8, 16, 8, 6)        # ends at the parent's step start
    with pytest.raises(ValueError):
        child_flux_intervals(0, 8, 24, 24, 6)        # three planes per split: longer than a split of N = 2
    with pytest.raises(ValueError):
        child_flux_intervals(0, 4, 4 * 13, 4, 12)    # past the parent's step end (12 planes at N = 4)


def test_the_ladder_reaches_rate_32_and_stops_there():
    assert DYADIC[-1] == 32
    for consumer in (32,):
        for provider in (1, 2, 4, 8, 16):
            ends = range(2 * provider, 6 * consumer + 1, 2 * provider)
            seen = sorted(p for p in (trajectory_plane(0, consumer, t, 6) for t in ends) if p is not None)
            assert seen == [1, 2, 3, 4, 5, 6]
    assert plan_macros(1200, 32) == (32,) * 37 + (16,)
    assert sum(plan_macros(1201, 32)) == 1201
    events = macro_schedule((32, 16, 8, 4, 2, 1), 32)
    assert sum(1 for e in events if e[0] == "step" and e[1] == 5) == 32
    coe = np.array([[0, 1], [1, 2], [2, 3]])
    assert list(grade_rates(np.array([32, 32, 1, 1]), coe)) == [4, 2, 1, 1]
    with pytest.raises(ValueError):
        grade_rates(np.array([64, 32, 16, 8]), coe)


def test_rate_32_is_an_active_rate_and_the_shadow_still_refuses_it():
    from hexcore.config_fullstep_lts import FullStepLtsConfig, validate_fullstep_lts_request
    from hexcore.errors import ConfigurationRefusal

    assert validate_fullstep_lts_request(FullStepLtsConfig(True, fullstep_lts_mode="active",
                                                           fullstep_lts_max_rate=32))
    with pytest.raises(ConfigurationRefusal):
        validate_fullstep_lts_request(FullStepLtsConfig(True, fullstep_lts_mode="shadow",
                                                        fullstep_lts_max_rate=32))
    with pytest.raises(ConfigurationRefusal):
        validate_fullstep_lts_request(FullStepLtsConfig(True, fullstep_lts_mode="active",
                                                        fullstep_lts_max_rate=64))


def _launch(kernel, total, args):
    kernel(((total + 255) // 256,), (256,), args)


@pytest.mark.skipif(os.environ.get("HEX_LTS_GPU_TESTS") != "1", reason="GPU test; set HEX_LTS_GPU_TESTS=1")
@pytest.mark.parametrize("substeps", [2, 4])
def test_the_imposed_interface_flux_reproduces_the_childs_integral(tmp_path, substeps):
    cp = pytest.importorskip("cupy")
    from hexcore.cuda_backend import require_cuda
    from hexcore.cuda_backend.runtime import KernelCache
    from hexcore.cuda_fullstep_lts_active import _SOURCE

    cache = KernelCache(capability=require_cuda(min_compute=(8, 0)), cache_dir=str(tmp_path))

    def kernel(name):
        return cache.raw_kernel(name, _SOURCE, module_key="hexcore.cuda_fullstep_lts_active")

    record_k, keep_k, impose_k, theta_k = (kernel(n) for n in (
        "lts_flux_record", "lts_flux_keep", "lts_flux_impose", "lts_theta_impose"))
    N = substeps
    planes = planes_per_step(N)
    rng = np.random.default_rng(7)
    nlev, nedges, ncells, nslots, nmail = 3, 10, 8, 6, 5
    fine = [1, 4, 7]
    fine_edge = cp.asarray(np.array(fine, dtype=np.int32))
    fine_slot = cp.asarray(np.array([5, 0, 2], dtype=np.int32))
    fine_cells = cp.asarray(np.array([0, 1, 2, 3, 4, 5], dtype=np.int32))
    dv_edge = cp.asarray(rng.uniform(1.0e4, 2.0e4, nedges).astype(np.float32))
    record = cp.zeros((planes, nlev, nslots), dtype=cp.float64)
    record_theta = cp.zeros_like(record)
    child_dt = 30.0
    means, heats = [], []
    for s in range(6):                       # six child splits (two child steps), N/2 planes each
        flux_u = cp.asarray(rng.normal(30.0, 5.0, (nlev, nedges)).astype(np.float32))
        ru_saved = cp.asarray(rng.normal(29.0, 5.0, (nlev, nedges)).astype(np.float32))
        theta_flux = cp.asarray(rng.normal(9000.0, 50.0, (nlev, nedges)).astype(np.float32))
        theta_saved = cp.asarray(rng.normal(300.0, 2.0, (nlev, ncells)).astype(np.float32))
        means.append(cp.asnumpy(flux_u)[:, fine].astype(np.float64))
        f, r, t, ts = (cp.asnumpy(a).astype(np.float64) for a in (flux_u, ru_saved, theta_flux, theta_saved))
        edge_theta = 0.5 * (ts[:, [0, 2, 4]] + ts[:, [1, 3, 5]])
        heats.append(child_dt * (t[:, fine] + cp.asnumpy(dv_edge)[fine] * (f[:, fine] - r[:, fine]) * edge_theta))
        _launch(record_k, nlev * 3, (
            np.int32(nlev), np.int32(3), np.int32(nedges), np.int32(ncells), np.int32(nslots),
            np.int32(s * (N // 2)), np.int32(N // 2), np.float64(child_dt), fine_edge, fine_cells, fine_slot,
            dv_edge, flux_u, ru_saved, theta_flux, theta_saved, record, record_theta))
    # the parent sees the same edges as halo edges 2, 6, 9 with mailbox slots 0, 3, 4
    coarse = [2, 6, 9]
    coarse_edge = cp.asarray(np.array(coarse, dtype=np.int32))
    mail = cp.asarray(np.array([0, 3, 4], dtype=np.int32))
    G = cp.asarray(rng.normal(28.0, 5.0, (planes + 1, 2, nlev, nmail)).astype(np.float32))
    ru_p = cp.asarray(rng.normal(0.0, 1.0, (nlev, nedges)).astype(np.float32))
    ru_avg = cp.zeros_like(ru_p)
    kept = cp.zeros((nlev, 3), dtype=cp.float32)
    parent_dt = 2 * child_dt
    for split in range(3):
        # third RK stage: N sub-steps spanning planes N*split .. N*split+N-1
        for small_step in range(1, N + 1):
            _launch(keep_k, nlev * 3, (np.int32(nlev), np.int32(3), np.int32(nedges), coarse_edge, ru_avg, kept))
            _launch(impose_k, nlev * 3, (
                np.int32(nlev), np.int32(3), np.int32(nedges), np.int32(nslots), np.int32(nmail),
                np.int32(N * split + small_step - 1), np.int32(N * split), np.int32(small_step == 1),
                coarse_edge, fine_slot, mail, record, G, kept, ru_p, ru_avg))
        save = cp.asnumpy(G)[N * split, 0][:, [0, 3, 4]].astype(np.float64)
        recovered = save + cp.asnumpy(ru_avg)[:, coarse].astype(np.float64) / N
        child = 0.5 * (means[2 * split] + means[2 * split + 1])
        # what the parent's density update and register now carry is the child's mean, to float32 rounding
        np.testing.assert_allclose(recovered, child, rtol=4e-7, atol=0)
        untouched = np.setdiff1d(np.arange(nedges), coarse)
        assert not cp.asnumpy(ru_avg)[:, untouched].any()
        # heat: the large-step flux becomes the child's integral over the split, less the acoustic part
        theta_saved = cp.asarray(rng.normal(300.0, 2.0, (nlev, ncells)).astype(np.float32))
        flux = cp.zeros((nlev, nedges), dtype=cp.float32)
        _launch(theta_k, nlev * 3, (
            np.int32(nlev), np.int32(3), np.int32(nedges), np.int32(ncells), np.int32(nslots), np.int32(nmail),
            np.int32(N * split), np.int32(N), np.int32(N * split), np.float64(parent_dt),
            coarse_edge, fine_cells, fine_slot, mail, record, record_theta, G, dv_edge, theta_saved, flux))
        ts = cp.asnumpy(theta_saved).astype(np.float64)
        edge_theta = 0.5 * (ts[:, [0, 2, 4]] + ts[:, [1, 3, 5]])
        perturbation = child - save
        expected = ((heats[2 * split] + heats[2 * split + 1]) / parent_dt
                    - cp.asnumpy(dv_edge)[coarse] * perturbation * edge_theta)
        np.testing.assert_allclose(cp.asnumpy(flux)[:, coarse].astype(np.float64), expected, rtol=2e-6, atol=1e-3)


def test_lts_class_drivers_keep_the_rk_theta_correction_in_the_edge_flux():
    # dycore-match (8322a3f) moved the v8.4.1 RK theta correction out of the
    # edge flux into the cell loop and passes rk_step 1 to the edge kernel.
    # The LTS capture recognises the third-stage edge flux by rk_step 3 and
    # books it as the whole heat through an edge, so every LTS step stopped
    # with "the third-stage theta edge flux was not observed" until LTS class
    # drivers kept the correction in the edge flux (G3, 2026-10-12).
    import ast
    import inspect

    from hexcore import cuda_driver, cuda_fullstep_lts_active

    active = inspect.getsource(cuda_fullstep_lts_active.FullStepLtsActive.__init__)
    assert "klass.driver.lts_theta_correction_in_edge_flux = True" in active
    tendency = inspect.getsource(cuda_driver.CudaDryDycoreDriver._theta_tendency)
    assert '"lts_theta_correction_in_edge_flux"' in tendency
    assert "np.int32(1 if separate_correction else rk_step)" in tendency
    assert "if rk_step > 1 and separate_correction else None" in tendency
    ast.parse(inspect.getsource(cuda_driver))


def test_every_kernel_the_lts_interfaces_tap_is_one_the_v841_path_launches():
    # A tap on a kernel name the driver no longer launches never fires, and the
    # interface then refuses mid-step (identity-extend 4688aaa renamed the v841
    # vertical low-order kernel; G3, 2026-10-12).  Every tapped name must be
    # launched by name in the module it is tapped from.
    import inspect
    import re

    from hexcore import (cuda_acoustic_v841, cuda_dynamics_v841, cuda_fullstep_lts_active,
                         cuda_fullstep_lts_multi, cuda_fullstep_lts_scalars, cuda_transport_v841)

    modules = {"dynamics": cuda_dynamics_v841, "transport": cuda_transport_v841,
               "acoustic_module": cuda_acoustic_v841}
    names = set(cuda_fullstep_lts_scalars._LOW_ORDER) | set(cuda_fullstep_lts_scalars._FINISH)
    for source in (inspect.getsource(cuda_fullstep_lts_scalars), inspect.getsource(cuda_fullstep_lts_multi),
                   inspect.getsource(cuda_fullstep_lts_active)):
        for module, name in re.findall(r'tap_kernel\((\w+), [^,]+, "(\w+)"', source):
            launched = inspect.getsource(modules[module])
            assert f'"{name}"' in launched.replace(f'void {name}(', ''), (module, name)
            names.add(name)
    transport = inspect.getsource(cuda_transport_v841)
    for name in set(cuda_fullstep_lts_scalars._LOW_ORDER) | set(cuda_fullstep_lts_scalars._FINISH):
        assert f'"{name}"' in transport, name
    assert "fct_vertical_low_order_v841" in names


@pytest.mark.parametrize("dimensions", [(), (("nlev", 55), ("nedges", 389), ("ncells", 129))])
def test_acoustic_lts_tap_wraps_the_resolved_launch_for_its_cache_and_shape(monkeypatch, dimensions):
    from hexcore import cuda_acoustic_v841 as acoustic
    from hexcore.cuda_fullstep_lts_scalars import tap_kernel

    events = []
    arguments = tuple(object() for _ in range(19))

    class Cache:
        def raw_kernel(self, name, source, *, module_key):
            def native(grid, block, args, **keywords):
                assert args is arguments
                events.append(("kernel", self))
                return "launched"
            return native

    cache, other = Cache(), Cache()
    monkeypatch.setattr(acoustic, "_KERNELS", {})
    def before(args):
        assert args is arguments
        events.append(("before", cache))
    def after(args):
        assert args is arguments
        events.append(("after", cache))
    tap_kernel(acoustic, cache, "acoustic_ru_v841", before=before, after=after,
               dimensions=dimensions)
    launch = acoustic._kernel("acoustic_ru_v841", cache, dimensions)
    assert launch((2,), (256,), arguments) == "launched"
    assert events == [("before", cache), ("kernel", cache), ("after", cache)]
    events.clear()
    acoustic._kernel("acoustic_ru_v841", other, dimensions)((2,), (256,), arguments)
    assert events == [("kernel", other)]
    with pytest.raises(RuntimeError, match="already tapped"):
        tap_kernel(acoustic, cache, "acoustic_ru_v841", dimensions=dimensions)
