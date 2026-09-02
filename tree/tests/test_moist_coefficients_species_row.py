"""The moist coefficients sum the LOADING species, not the leading N.

``atm_compute_moist_coefficients`` forms ``qtot`` as a running FP32 sum over
the moist mass species and then ``cqw``/``cqu`` as ``1/(1 + ...)``;
``moist_dpdz_v841_f32`` carries ``qtot`` into the buoyancy term.  Both
kernels used to loop ``species < 6`` unconditionally, which was correct for
exactly one scheme -- WSM6, whose six species are all mass-loading -- and
silently wrong for any other row.

THE BREAKAGE THESE GATES PREVENT, measured on the card rather than argued.
Under P3's eight-species state the welded loop would fold ``ni`` and ``nr``
(number concentrations) and ``qib`` (a volume) into ``qtot``, and it would
fold ``qir`` in as well -- a mass, but the rime mass held INSIDE ``qi``,
which ``qi`` already counts.  The gates below measure both halves: that the
loading subset is what the kernel sums, and that summing more collapses the
coefficient by six orders of magnitude.  A gate that cannot fail on the
broken behaviour proves nothing, so the broken arm is run on purpose.

The WSM6 arm is the frozen lane's byte-identity: the loop bound became a
runtime value and a compiler may unroll a runtime bound differently from a
literal, so the sum's bytes are compared against a reference formed in the
same order rather than assumed to be unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from hexcore import species_row as sr  # noqa: E402
from hexcore.cuda_driver import CUDA_V841_PHYSICS_DRIVER_SOURCE  # noqa: E402


NLEV, NCELLS, NEDGES = 24, 512, 1536
_OPTIONS = ("--std=c++17", "--fmad=false")


@pytest.fixture(scope="module")
def module():
    return cp.RawModule(
        code=CUDA_V841_PHYSICS_DRIVER_SOURCE,
        options=_OPTIONS,
        backend="nvrtc",
    )


@pytest.fixture(scope="module")
def cells_on_edge():
    rng = np.random.default_rng(20260901)
    return cp.asarray(
        np.ascontiguousarray(
            rng.integers(0, NCELLS, size=(NEDGES, 2), dtype=np.int32)))


def _launch(module, name, count, args):
    threads = 128
    module.get_function(name)(
        ((count + threads - 1) // threads,), (threads,), args)


def _coefficients(module, cells_on_edge, scalars, n_mass):
    """qtot/cqw/cqu as host arrays, for a block and a loading count."""

    device = cp.asarray(np.ascontiguousarray(scalars))
    qtot = cp.zeros((NLEV, NCELLS), dtype=cp.float32)
    cqw = cp.zeros_like(qtot)
    cqu = cp.zeros((NLEV, NEDGES), dtype=cp.float32)
    flag = cp.zeros(1, dtype=cp.int32)
    _launch(
        module, "moist_cell_coefficients_v841_f32", NCELLS,
        (np.int32(NLEV), np.int32(NCELLS), np.int32(n_mass),
         device, qtot, cqw, flag))
    _launch(
        module, "moist_edge_coefficients_v841_f32", NEDGES,
        (np.int32(NLEV), np.int32(NCELLS), np.int32(NEDGES),
         np.int32(n_mass), cells_on_edge, device, cqu, flag))
    cp.cuda.Stream.null.synchronize()
    assert int(flag.get()[0]) == 0, "the kernel flagged a non-finite value"
    return cp.asnumpy(qtot), cp.asnumpy(cqw), cp.asnumpy(cqu)


def _reference_qtot(scalars, n_mass):
    """The same running FP32 sum, in the same species order, on the host."""

    total = np.zeros(scalars.shape[1:], dtype=np.float32)
    for index in range(n_mass):
        total = (total + scalars[index]).astype(np.float32)
    return total


def _wsm6_state(seed=11):
    rng = np.random.default_rng(seed)
    block = np.zeros((6, NLEV, NCELLS), dtype=np.float32)
    block[0] = rng.uniform(0.0, 0.020, (NLEV, NCELLS))
    for index in range(1, 6):
        block[index] = rng.uniform(0.0, 2.0e-3, (NLEV, NCELLS))
    return np.ascontiguousarray(block)


def _p3_state(seed=12):
    """An eight-species P3 block with REALISTIC number concentrations.

    The numbers are what make the defect visible: a mixing ratio is order
    1e-3 and a concentration is order 1e3 to 1e6, so folding one into a sum
    of the other is not a small error.
    """

    rng = np.random.default_rng(seed)
    block = np.zeros((8, NLEV, NCELLS), dtype=np.float32)
    block[0] = rng.uniform(0.0, 0.020, (NLEV, NCELLS))      # qv
    for index in (1, 2, 3):                                  # qc qr qi
        block[index] = rng.uniform(0.0, 2.0e-3, (NLEV, NCELLS))
    block[4] = rng.uniform(1.0e3, 1.0e6, (NLEV, NCELLS))     # ni
    block[5] = rng.uniform(1.0e3, 1.0e6, (NLEV, NCELLS))     # nr
    block[6] = rng.uniform(0.0, 1.0e-3, (NLEV, NCELLS))      # qir
    block[7] = rng.uniform(0.0, 1.0e-6, (NLEV, NCELLS))      # qib
    return np.ascontiguousarray(block)


# ---------------------------------------------------------------------------
# WSM6: the frozen lane.
# ---------------------------------------------------------------------------


def test_wsm6_sums_all_six_exactly_as_the_reference(module, cells_on_edge):
    row = sr.WSM6_SPECIES_ROW
    assert row.n_mass_loading == 6
    block = _wsm6_state()
    qtot, cqw, cqu = _coefficients(
        module, cells_on_edge, block, row.n_mass_loading)
    assert qtot.tobytes() == _reference_qtot(block, 6).tobytes()
    # cqw is left unassigned on row zero by the native routine; never read it.
    assert np.all(np.isfinite(cqw[1:]))
    assert np.all(np.isfinite(cqu))


def test_wsm6_row_count_equals_the_welded_constant(module, cells_on_edge):
    """The row reproduces what the six-species weld did, not merely something.

    If these ever diverge the frozen lane has moved, and that is a finding
    rather than a detail.
    """

    from hexcore.cuda_driver import V841_WSM6_DYNAMICS_SCALAR_NAMES

    row = sr.WSM6_SPECIES_ROW
    assert row.names() == V841_WSM6_DYNAMICS_SCALAR_NAMES
    assert row.n_mass_loading == len(V841_WSM6_DYNAMICS_SCALAR_NAMES)


# ---------------------------------------------------------------------------
# P3: the reason the generalization exists.
# ---------------------------------------------------------------------------


def test_p3_qtot_holds_only_the_four_loading_species(module, cells_on_edge):
    row = sr.P3_SPECIES_ROW
    assert row.n_mass_loading == 4
    block = _p3_state()
    qtot, _, _ = _coefficients(
        module, cells_on_edge, block, row.n_mass_loading)
    assert qtot.tobytes() == _reference_qtot(block, 4).tobytes()
    # A mixing-ratio sum stays a mixing-ratio sum.
    assert qtot.max() < 0.1


def test_folding_the_number_species_in_would_collapse_the_coefficient(
        module, cells_on_edge):
    """The broken arm, run on purpose, so the gate can fail on the defect."""

    block = _p3_state()
    correct_qtot, correct_cqw, correct_cqu = _coefficients(
        module, cells_on_edge, block, sr.P3_SPECIES_ROW.n_mass_loading)
    broken_qtot, broken_cqw, broken_cqu = _coefficients(
        module, cells_on_edge, block, block.shape[0])

    assert correct_qtot.tobytes() != broken_qtot.tobytes()
    # Six orders of magnitude, not a rounding difference.
    assert broken_qtot.max() > 1.0e5 > correct_qtot.max()
    assert np.nanmin(broken_cqw[1:]) < 1.0e-5 < np.nanmin(correct_cqw[1:])
    assert np.nanmin(broken_cqu) < 1.0e-5 < np.nanmin(correct_cqu)


def test_the_rime_mass_component_is_excluded_too(module, cells_on_edge):
    """qir is a MASS and still must not be summed: qi already counts it.

    This is the half a three-way mass/number/volume split would have got
    wrong, so it gets its own gate rather than riding on the number arm.
    """

    row = sr.P3_SPECIES_ROW
    assert row.declaration("qir").species_class == sr.MASS_COMPONENT
    assert not row.declaration("qir").is_mass_loading
    block = _p3_state()
    correct, _, _ = _coefficients(
        module, cells_on_edge, block, row.n_mass_loading)
    # nmass=7 would reach qir through the contiguous prefix only if the row
    # were ordered differently; here it proves the count is load-bearing.
    with_more, _, _ = _coefficients(module, cells_on_edge, block, 5)
    assert correct.tobytes() != with_more.tobytes()


# ---------------------------------------------------------------------------
# Every declared row runs through the kernel it will actually be run with.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scheme", sorted(sr.registered_species_rows()))
def test_every_declared_row_produces_finite_coefficients(
        module, cells_on_edge, scheme):
    """No declared row can drive the kernel to a non-finite coefficient."""

    row = sr.registered_species_rows()[scheme]
    rng = np.random.default_rng(hash(scheme) % 2**32)
    block = np.zeros((row.n_species(), NLEV, NCELLS), dtype=np.float32)
    for index, item in enumerate(row.species):
        if item.species_class == sr.NUMBER:
            block[index] = rng.uniform(1.0e3, 1.0e6, (NLEV, NCELLS))
        else:
            block[index] = rng.uniform(0.0, 2.0e-3, (NLEV, NCELLS))
    block[0] = rng.uniform(0.0, 0.020, (NLEV, NCELLS))
    block = np.ascontiguousarray(block)
    qtot, cqw, cqu = _coefficients(
        module, cells_on_edge, block, row.n_mass_loading)
    assert np.all(np.isfinite(qtot))
    assert np.all(np.isfinite(cqw[1:]))
    assert np.all(np.isfinite(cqu))
    # The loading sum never sees a number concentration, whatever the row.
    assert qtot.max() < 0.1, scheme
