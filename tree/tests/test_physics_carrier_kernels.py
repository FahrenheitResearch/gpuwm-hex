"""The coupling and recovery kernels, exercised outside the x4 driver.

THE GAP THIS CLOSES, found the hard way.  ``couple_cells_v841_f32``,
``recover_wsm6_v841_f32`` and ``validate_wsm6_surface_v841_f32`` had no
test at all: only ``tools/run_cuda_v841_full_physics_x4.py`` ever launched
them, so an argument-order mistake between the CUDA signature and the
Python launch site was invisible until a full-physics run on a big card.
The species-row generalization introduced exactly that mistake -- the
kernel read ``(nlev, ncells, rvord, nspecies, ...)`` while the launch site
passed ``(nlev, ncells, nspecies, rvord, ...)`` -- and it reached the tree
because nothing here could fail.  These gates launch each kernel directly
and check it against a host reference formed in the same operation order.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from hexcore import species_row as sr  # noqa: E402
from hexcore.cuda_physics_v841 import _CUDA_SOURCE, RV_OVER_RD_F32  # noqa: E402

NLEV, NCELLS = 16, 256
_OPTIONS = ("--std=c++17", "--fmad=false")


@pytest.fixture(scope="module")
def module():
    return cp.RawModule(code=_CUDA_SOURCE, options=_OPTIONS, backend="nvrtc")


def _f32(values):
    return cp.asarray(np.ascontiguousarray(values, dtype=np.float32))


def _table(arrays):
    return cp.asarray(
        np.asarray([int(a.data.ptr) for a in arrays], dtype=np.uint64))


def _launch(module, name, count, args):
    threads = 128
    module.get_function(name)(
        ((count + threads - 1) // threads,), (threads,), args)


@pytest.mark.parametrize("scheme", sorted(sr.registered_species_rows()))
def test_coupling_writes_one_slot_per_species_of_the_row(module, scheme):
    """mass * rate, per species, for a row of any length."""

    row = sr.registered_species_rows()[scheme]
    n = row.n_species()
    rng = np.random.default_rng(3)
    rho = _f32(rng.uniform(0.4, 1.2, (NLEV, NCELLS)))
    rho_theta = _f32(rng.uniform(120.0, 400.0, (NLEV, NCELLS)))
    qv = _f32(rng.uniform(0.0, 0.02, (NLEV, NCELLS)))
    dtheta = _f32(rng.uniform(-1.0e-3, 1.0e-3, (NLEV, NCELLS)))
    rates = [_f32(rng.uniform(-1.0e-6, 1.0e-6, (NLEV, NCELLS)))
             for _ in range(n)]
    out_rho = cp.zeros((NLEV, NCELLS), dtype=cp.float32)
    out_rtheta = cp.zeros_like(out_rho)
    out_q = cp.zeros((n, NLEV, NCELLS), dtype=cp.float32)
    flag = cp.zeros(1, dtype=cp.int32)
    _launch(module, "couple_cells_v841_f32", NCELLS,
            (np.int32(NLEV), np.int32(NCELLS), np.int32(n), RV_OVER_RD_F32,
             rho, rho_theta, qv, dtheta, _table(rates),
             out_rho, out_rtheta, out_q, flag))
    cp.cuda.Stream.null.synchronize()
    assert int(flag.get()[0]) == 0

    host_rho = cp.asnumpy(rho)
    result = cp.asnumpy(out_q)
    for index in range(n):
        expected = (host_rho * cp.asnumpy(rates[index])).astype(np.float32)
        assert result[index].tobytes() == expected.tobytes(), index
    # The uncoupled density tendency is identically zero, every row.
    assert not cp.asnumpy(out_rho).any()


@pytest.mark.parametrize("scheme", sorted(sr.registered_species_rows()))
def test_recovery_writes_every_species_and_reads_vapour_for_theta(
        module, scheme):
    row = sr.registered_species_rows()[scheme]
    n, nrad = row.n_species(), len(row.radius_names)
    count = NLEV * NCELLS
    rng = np.random.default_rng(5)
    rho = _f32(rng.uniform(0.4, 1.2, (NLEV, NCELLS)))
    theta = _f32(rng.uniform(250.0, 350.0, (NLEV, NCELLS)))
    water = [_f32(rng.uniform(0.0, 2.0e-3, (NLEV, NCELLS)))
             for _ in range(n)]
    water[0] = _f32(rng.uniform(0.0, 0.02, (NLEV, NCELLS)))
    radii = [_f32(rng.uniform(1.0e-6, 5.0e-5, (NLEV, NCELLS)))
             for _ in range(nrad)]
    out_rt = cp.zeros((NLEV, NCELLS), dtype=cp.float32)
    out_sc = cp.zeros((n, NLEV, NCELLS), dtype=cp.float32)
    flag = cp.zeros(1, dtype=cp.int32)
    args = [np.int32(count), np.int32(n), np.int32(nrad), RV_OVER_RD_F32,
            rho, theta, _table(water)]
    args.append(_table(radii) if nrad else _table(water))
    args += [out_rt, out_sc, flag]
    _launch(module, "recover_wsm6_v841_f32", count, tuple(args))
    cp.cuda.Stream.null.synchronize()
    assert int(flag.get()[0]) == 0

    result = cp.asnumpy(out_sc)
    for index in range(n):
        assert result[index].tobytes() == cp.asnumpy(
            water[index]).tobytes(), index


def test_the_surface_validator_refuses_a_decreasing_accumulator(module):
    """Every row's bucket count, and the refusal still has teeth."""

    for row in sr.registered_species_rows().values():
        buckets = len(row.surface_accumulators)
        acc = [_f32(np.abs(np.full(NCELLS, 10.0))) for _ in range(buckets)]
        inc = [_f32(np.abs(np.full(NCELLS, 1.0))) for _ in range(buckets)]
        good = [_f32(np.zeros(NCELLS)) for _ in range(buckets)]
        bad = [_f32(np.full(NCELLS, 1.0e9)) for _ in range(buckets)]
        sr_field = _f32(np.full(NCELLS, 0.5))

        def run(previous):
            flag = cp.zeros(1, dtype=cp.int32)
            _launch(module, "validate_wsm6_surface_v841_f32", NCELLS,
                    (np.int32(NCELLS), np.int32(buckets), np.float32(1.0),
                     _table(acc), _table(inc), _table(previous),
                     sr_field, flag))
            cp.cuda.Stream.null.synchronize()
            return int(flag.get()[0])

        assert run(good) == 0, row.scheme_name
        assert run(bad) == 1, row.scheme_name
