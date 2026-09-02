"""The preparation kernel serves the row's species, not six welded pointers.

``prep_mass_v841_f32`` took six ``float *q*_p`` parameters and built a
``float *qout[6]`` table inside the kernel, so a scheme with a different
inventory could not be prepared at all -- the door's hard stop.  It now
takes a DEVICE table of output pointers and ``nspecies``, and
``validate_prep_v841_f32`` checks the species from the same table instead
of a ``mass[21]`` list that grew with the scheme.

The arithmetic did not move, and these gates measure that rather than
asserting it: slot 0 is water vapour by the row's own invariant, and the
moist density, the dry-theta conversion and the temperature are formed from
slot 0 alone -- no other species enters any of them, on any row.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from hexcore import species_row as sr  # noqa: E402
from hexcore.cuda_physics_prep_v841 import (  # noqa: E402
    CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
    RV_OVER_RD_F32,
    _CUDA_SOURCE,
    _contract_sha256,
    _mass_fields,
)

NLEV, NCELLS = 18, 384
_OPTIONS = ("--std=c++17", "--fmad=false")
_OUT9 = ("zz_p", "rho_dry", "rho_p", "th_p", "t_p",
         "pi_p", "pres_p", "zmid_p", "dz_p")


@pytest.fixture(scope="module")
def module():
    return cp.RawModule(code=_CUDA_SOURCE, options=_OPTIONS, backend="nvrtc")


def _inputs(seed=7):
    rng = np.random.default_rng(seed)

    def f32(values):
        return cp.asarray(np.ascontiguousarray(values, dtype=np.float32))

    return {
        "rho_zz": f32(rng.uniform(0.3, 1.2, (NLEV, NCELLS))),
        "theta_m": f32(rng.uniform(280.0, 360.0, (NLEV, NCELLS))),
        "exner": f32(rng.uniform(0.4, 1.0, (NLEV, NCELLS))),
        "pbase": f32(rng.uniform(2.0e4, 1.0e5, (NLEV, NCELLS))),
        "ppert": f32(rng.uniform(-3.0e2, 3.0e2, (NLEV, NCELLS))),
        "zz": f32(rng.uniform(0.9, 1.1, (NLEV, NCELLS))),
        "zgrid": f32(np.cumsum(
            rng.uniform(80.0, 400.0, (NLEV + 1, NCELLS)), axis=0)),
    }


def _species_block(row, seed=8):
    """A state for this row: vapour, mixing ratios, real concentrations."""
    rng = np.random.default_rng(seed)
    block = np.zeros((row.n_species(), NLEV, NCELLS), dtype=np.float32)
    for index, item in enumerate(row.species):
        if item.species_class == sr.NUMBER:
            block[index] = rng.uniform(1.0e3, 1.0e6, (NLEV, NCELLS))
        else:
            # negatives on purpose: the kernel's clamp is bitwise max(+0,q)
            block[index] = rng.uniform(-1.0e-9, 2.0e-3, (NLEV, NCELLS))
    block[0] = rng.uniform(0.0, 0.02, (NLEV, NCELLS))
    return np.ascontiguousarray(block)


def _prepare(module, row, block, inputs):
    n = row.n_species()
    species = [cp.zeros((NLEV, NCELLS), dtype=cp.float32) for _ in range(n)]
    table = cp.asarray(
        np.asarray([int(x.data.ptr) for x in species], dtype=np.uint64))
    out = {k: cp.zeros((NLEV, NCELLS), dtype=cp.float32) for k in _OUT9}
    flag = cp.zeros(1, dtype=cp.int32)
    args = (np.int32(NLEV), np.int32(NCELLS), np.int32(n), RV_OVER_RD_F32,
            inputs["rho_zz"], inputs["theta_m"], cp.asarray(block),
            inputs["exner"], inputs["pbase"], inputs["ppert"],
            inputs["zgrid"], inputs["zz"], table,
            *[out[k] for k in _OUT9], flag)
    threads = 128
    module.get_function("prep_mass_v841_f32")(
        ((NCELLS + threads - 1) // threads,), (threads,), args)
    cp.cuda.Stream.null.synchronize()
    assert int(flag.get()[0]) == 0
    return ([cp.asnumpy(x) for x in species],
            {k: cp.asnumpy(v) for k, v in out.items()})


@pytest.mark.parametrize("scheme", sorted(sr.registered_species_rows()))
def test_every_declared_row_prepares_all_of_its_species(module, scheme):
    """The weld's hard stop, gone: any declared inventory is preparable."""

    row = sr.registered_species_rows()[scheme]
    block = _species_block(row)
    species, _ = _prepare(module, row, block, _inputs())
    assert len(species) == row.n_species()
    for index, item in enumerate(row.species):
        # bitwise max(+0, q): the clamp, applied to every species alike
        assert np.array_equal(species[index], np.maximum(block[index], 0.0))
        assert not np.signbit(species[index]).any(), item.name


@pytest.mark.parametrize("scheme", sorted(sr.registered_species_rows()))
def test_the_moist_derivations_read_slot_zero_alone(module, scheme):
    """rho_p, th_p and t_p come from vapour only, whatever else the row has.

    This is the invariant that lets an arbitrary species row through the
    preparation kernel without any new arithmetic: no species but qv enters
    the moist density or the dry-theta conversion.
    """

    row = sr.registered_species_rows()[scheme]
    inputs = _inputs()
    block = _species_block(row)
    _, out = _prepare(module, row, block, inputs)

    qv = np.maximum(block[0], 0.0)
    rho_dry = (cp.asnumpy(inputs["zz"]) * cp.asnumpy(inputs["rho_zz"])
               ).astype(np.float32)
    assert out["rho_dry"].tobytes() == rho_dry.tobytes()
    assert out["rho_p"].tobytes() == (
        rho_dry * (np.float32(1.0) + qv)).astype(np.float32).tobytes()
    theta = (cp.asnumpy(inputs["theta_m"])
             / (np.float32(1.0) + RV_OVER_RD_F32 * qv)).astype(np.float32)
    assert out["th_p"].tobytes() == theta.tobytes()

    # Perturbing a NON-vapour species must not move any derived field.
    if row.n_species() > 1:
        moved = block.copy()
        moved[1:] = moved[1:] * np.float32(3.0) + np.float32(1.0e-4)
        _, out2 = _prepare(module, row, moved, inputs)
        for key in ("rho_dry", "rho_p", "th_p", "t_p", "pi_p", "pres_p"):
            assert out[key].tobytes() == out2[key].tobytes(), (scheme, key)


def test_wsm6_contract_digest_did_not_move():
    """The generator reproduces the frozen document byte for byte."""

    assert CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256 == (
        "a9436205305c4127dc38399288ad47839dc797d0c786e1118e9e800cc7223c1d")
    assert _contract_sha256(sr.WSM6_SPECIES_ROW.names()) == (
        CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256)


def test_a_different_row_gets_a_different_contract_digest():
    """A contract that covers the row must move when the row does."""

    digests = {
        name: _contract_sha256(row.names())
        for name, row in sr.registered_species_rows().items()
    }
    assert len(set(digests.values())) == len(digests), digests


def test_mass_field_names_lead_with_the_rows_species():
    for row in sr.registered_species_rows().values():
        fields = _mass_fields(row.names())
        assert fields[: row.n_species()] == tuple(
            f"{name}_p" for name in row.names())
        # The 15 non-species outputs never change with the scheme.
        assert len(fields) == row.n_species() + 15


# ---------------------------------------------------------------------------
# The carriers themselves, not just the kernel.
# ---------------------------------------------------------------------------


def test_the_species_attribute_shim_is_a_function_not_a_descriptor():
    """The defect this catches cost an x4 card run, and no test saw it.

    The six literal ``q*_p`` fields became one mapping plus a
    ``__getattr__`` shim.  The edit that inserted the shim landed it
    between a bare ``@property`` and the function that decorator was meant
    to wrap, so ``__getattr__`` BECAME a property object: every attribute
    miss then raised ``TypeError: __getattr__() missing 1 required
    positional argument`` instead of resolving a species, and the whole
    battery stayed green because nothing here ever CONSTRUCTED one of these
    carriers -- the kernel gates launch CUDA directly.
    """

    from hexcore import cuda_physics_prep_v841 as prep

    for cls in (prep.CudaMpasToPhysColumnsV841,
                prep.CpuMpasToPhysColumnsV841,
                prep.CudaWsm6InputViewV841,
                prep.CpuWsm6InputViewV841):
        shim = cls.__dict__.get("__getattr__")
        assert shim is not None, cls.__name__
        assert type(shim).__name__ == "function", (cls.__name__, type(shim))

    # And the decorators that shim was inserted beside still bind.
    assert isinstance(
        prep.CudaMpasToPhysColumnsV841.__dict__["scalar_scratch"], property)


def test_a_column_carrier_resolves_every_species_by_its_old_spelling():
    """``prepared.qv_p`` keeps working, for any row, through the shim."""

    from hexcore import cuda_physics_prep_v841 as prep

    row = sr.P3_SPECIES_ROW
    plane = np.zeros((2, 3), dtype=np.float32)
    species = {name: plane.copy() for name in row.names()}
    carrier = object.__new__(prep.CpuMpasToPhysColumnsV841)
    object.__setattr__(carrier, "species_p", species)
    for index, name in enumerate(row.names()):
        object.__setattr__(carrier, "species_p", species)
        assert getattr(carrier, f"{name}_p") is species[name], name
    with pytest.raises(AttributeError):
        getattr(carrier, "qzzz_p")
