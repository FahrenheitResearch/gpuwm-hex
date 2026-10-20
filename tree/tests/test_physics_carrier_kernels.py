"""The coupling and recovery kernels, exercised outside the x4 driver.

THE GAP THIS CLOSES, found the hard way.  ``couple_cells_v841_f32``,
``recover_wsm6_v841_f32`` and ``validate_wsm6_surface_v841_f32`` had no
test at all: only ``src/hexcore/drivers/run_cuda_v841_full_physics_x4.py`` ever launched
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


@pytest.mark.parametrize("n", [1, 3, 5, 6, 8, 10, 23])
def test_the_commit_publishes_the_whole_block_and_nothing_past_it(module, n):
    """``commit_wsm6_state_v841_f32`` copies exactly the row's block.

    THE BREAKAGE THIS PREVENTS, measured 2026-10-11: the kernel looped
    ``s < 6`` whatever the row's length.  A three-species block (Kessler,
    microphysics off) was read and written three planes past both buffers
    and the card stopped with cudaErrorIllegalAddress at the first commit
    of a 60 km and a 30 km global forecast; a block longer than six left
    every plane after the sixth unpublished.  The live buffer here is
    allocated six planes longer than the block and filled with a sentinel,
    so an over-long copy is a failed assertion and not a card fault.
    """

    count = NLEV * NCELLS
    pad = 6
    sentinel = np.float32(-777.0)
    rng = np.random.default_rng(11)
    candidate_rt = _f32(rng.uniform(120.0, 400.0, (NLEV, NCELLS)))
    candidate = _f32(rng.uniform(0.0, 2.0e-3, (n + pad, NLEV, NCELLS)))
    live_rt = cp.zeros((NLEV, NCELLS), dtype=cp.float32)
    live = cp.full((n + pad, NLEV, NCELLS), sentinel, dtype=cp.float32)
    _launch(module, "commit_wsm6_state_v841_f32", count,
            (np.int32(count), np.int32(n), candidate_rt, candidate,
             live_rt, live))
    cp.cuda.Stream.null.synchronize()

    result = cp.asnumpy(live)
    assert result[:n].tobytes() == cp.asnumpy(candidate[:n]).tobytes()
    assert (result[n:] == sentinel).all()
    assert cp.asnumpy(live_rt).tobytes() == cp.asnumpy(candidate_rt).tobytes()


_OFF_ROW = sr.engine_species_row(
    scheme_name="test-carrier-off", mp_physics=0, species=("qv", "qc", "qr"),
    mass_species=("qv", "qc", "qr"), surface_accumulators=())


@pytest.mark.parametrize("off", [False, True], ids=["kessler", "microphysics-off"])
def test_the_recovery_commits_a_three_species_block(module, monkeypatch, off):
    """The whole recovery on Kessler's block of three, through the real
    launch site: three planes published, nothing written past them.

    The live scalars are the first three planes of a nine-plane buffer, so
    the launch site passing a wrong length shows as a touched sentinel."""

    from types import SimpleNamespace

    from hexcore import cuda_physics_v841 as physics
    from hexcore import cuda_solve_region_v841 as solve_region

    monkeypatch.setattr(
        physics, "_launch",
        lambda cache, name, count, args: _launch(module, name, count, args))
    monkeypatch.setattr(
        solve_region, "read_validation_word",
        lambda invalid, cache: int(invalid.get()[0]))

    # Microphysics off carries Kessler's three species and no grid-scale
    # bucket.  The names alone select Kessler's row (and its RAINNC), so the
    # off update names its own row; before it could, the recovery asked the
    # update for a RAINNC it never carried.
    row = _OFF_ROW if off else sr.KESSLER_SPECIES_ROW
    names = row.names()
    assert names == ("qv", "qc", "qr")
    n, nedges = len(names), 3 * NCELLS
    sentinel = np.float32(-777.0)
    rng = np.random.default_rng(13)
    backing = cp.full((n + 6, NLEV, NCELLS), sentinel, dtype=cp.float32)
    state = SimpleNamespace(
        rho=_f32(rng.uniform(0.4, 1.2, (NLEV, NCELLS))),
        rho_u=cp.zeros((NLEV, nedges), dtype=cp.float32),
        rho_theta=cp.zeros((NLEV, NCELLS), dtype=cp.float32),
        scalars=backing[:n],
        time_seconds=360.0,
        dtype=np.float32,
        validate=lambda **_: None,
    )
    species = {name: _f32(rng.uniform(0.0, 2.0e-3, (NLEV, NCELLS)))
               for name in names}
    # Microphysics off publishes no SR (no scheme forms one).
    surface = {} if off else {"sr": _f32(np.zeros(NCELLS))}
    for item in row.surface_accumulators:
        surface[item.name] = _f32(np.full(NCELLS, 2.0))
        surface[f"{item.name}v"] = _f32(np.full(NCELLS, 1.0))
    update = physics.CudaPostRkWsm6UpdateV841(
        theta=_f32(rng.uniform(250.0, 350.0, (NLEV, NCELLS))),
        species=species, surface=surface,
        radii={name: _f32(np.full((NLEV, NCELLS), 1.0e-5))
               for name in row.radius_names},
        time_seconds=360.0, row=row if off else None)
    physics.recover_post_rk_wsm6_state_v841(
        state, update, scalar_names=names, kernel_cache=None,
        phase2_dt_seconds=360.0)
    cp.cuda.Stream.null.synchronize()

    result = cp.asnumpy(backing)
    for index, name in enumerate(names):
        assert result[index].tobytes() == cp.asnumpy(species[name]).tobytes(), name
    assert (result[n:] == sentinel).all()
    assert cp.asnumpy(state.rho_theta).all()


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
                    (np.int32(NCELLS), np.int32(buckets), np.int32(1),
                     np.float32(1.0),
                     _table(acc), _table(inc), _table(previous),
                     cp.asarray([v.signed_fallout for v in row.surface_accumulators], dtype=cp.int32),
                     sr_field, flag))
            cp.cuda.Stream.null.synchronize()
            return int(flag.get()[0])

        assert run(good) == 0, row.scheme_name
        assert run(bad) == (0 if all(v.signed_fallout for v in row.surface_accumulators) else 1), row.scheme_name


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("broken", [None, "acc", "inc", "old", "sr"])
def test_surface_fallout_policy_preserves_words_and_refuses_nonfinite(module, signed, broken):
    # Includes the real Morrison refusal's magnitude and a positive bucket
    # decreasing from a previous positive word. Neither is clipped or resummed.
    acc = _f32([-1.3981764490221711e-14, 0.5])
    inc = _f32([-1.3981764490221711e-14, -0.5])
    old = _f32([0.0, 1.0])
    sr_field = _f32([0.0, 0.5])
    arrays = dict(acc=acc, inc=inc, old=old, sr=sr_field)
    if broken:
        arrays[broken][0] = cp.float32(cp.nan)
    before = {key: value.get().tobytes() for key, value in arrays.items()}
    flag = cp.zeros(1, dtype=cp.int32)
    _launch(module, "validate_wsm6_surface_v841_f32", 2,
            (np.int32(2), np.int32(1), np.int32(1), np.float32(1.0),
             _table([acc]), _table([inc]), _table([old]),
             cp.asarray([signed], dtype=cp.int32), sr_field, flag))
    assert int(flag.get()[0]) == (0 if signed and broken is None else 1)
    assert {key: value.get().tobytes() for key, value in arrays.items()} == before


def test_signed_fallout_does_not_relax_a_neighbouring_strict_bucket(module):
    acc = [_f32([-1e-14]), _f32([-1e-14])]
    inc = [_f32([-1e-14]), _f32([-1e-14])]
    old = [_f32([0.0]), _f32([0.0])]
    flag = cp.zeros(1, dtype=cp.int32)
    _launch(module, "validate_wsm6_surface_v841_f32", 1,
            (np.int32(1), np.int32(2), np.int32(1), np.float32(1.0),
             _table(acc), _table(inc), _table(old),
             cp.asarray([1, 0], dtype=cp.int32), _f32([0.0]), flag))
    assert int(flag.get()[0]) == 1
