"""A hex region cut from a kept hex parent hands on number concentrations with the mass (hexcore.handover_numbers).

THE BREAKAGE (init-from-any-run proof, 2026-10-24): parent -> child routes handed hydrometeor mass with number 0.  On
the hex-region route the start and boundary remaps moved the six water masses only, and parent_state copied a number
only when the parent's init already held one (it never does).  Mass with no number is sized at the scheme's limits.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from hexcore import handover_numbers as hn


def native(nlev=4, ncell=5, *, qr=None, nr=None):
    """Native fields as analysis_remap.native_fields returns them: a warm, moist, 1 kg m-3 column."""
    shape = (nlev, ncell)
    rho = np.full(shape, 1.0, np.float32)
    theta = np.full(shape, 300.0, np.float32)
    qv = np.full(shape, 0.01, np.float32)
    fields = dict(rho=rho, theta=theta, theta_m=theta * (1 + 461.6 / 287 * qv), qv=qv,
                  qc=np.zeros(shape, np.float32), qi=np.zeros(shape, np.float32),
                  qs=np.zeros(shape, np.float32), qg=np.zeros(shape, np.float32),
                  qr=np.zeros(shape, np.float32) if qr is None else qr)
    if nr is not None:
        fields["nr"] = nr
    return fields


def test_a_missing_rain_number_is_diagnosed_from_the_rain_mass_never_zero():
    qr = np.zeros((4, 5), np.float32)
    qr[1, 2] = 4.5e-4          # A's 0.45 g/kg of rain, carried in with no number
    fields = native(qr=qr)
    receipt = hn.complete_numbers(fields, ("ni", "nr", "nc", "nwfa", "nifa"), scheme=28, xland=None)
    nr = fields["nr"]
    assert nr[1, 2] > 0 and np.count_nonzero(nr) == 1
    assert receipt["diagnosed"] == {"nr": 1} and receipt["cold_zero"] == ["nwfa", "nifa"]
    # make_RainNumber at the Marshall-Palmer intercept (warm): N = N0/lambda, lambda = (pi rho_w N0 / (rho q))^1/4.
    lam = (np.pi * 1000.0 * 8.0e6 / (1.01 * 4.5e-4)) ** 0.25
    assert nr[1, 2] == pytest.approx(8.0e6 / lam / 1.01, rel=2e-2)
    # Every number the region now carries is defined everywhere the planes are.
    assert all(fields[name].shape == qr.shape for name in hn.NUMBER_SPECIES)


def test_a_published_number_is_carried_and_only_its_holes_under_mass_are_filled():
    qr = np.full((4, 5), 1e-4, np.float32)
    nr = np.full((4, 5), 1234.0, np.float32)
    nr[0, 0] = 0.0             # a carried plane with a hole under mass
    fields = native(qr=qr, nr=nr.copy())
    receipt = hn.complete_numbers(fields, ("nr",), scheme=8, xland=None)
    assert receipt["carried"] == ["nr"] and receipt["diagnosed"] == {"nr": 1}
    assert fields["nr"][0, 0] > 0
    assert np.array_equal(fields["nr"].ravel()[1:], nr.ravel()[1:])


def test_an_exact_run_or_a_row_without_thompson_relations_diagnoses_nothing():
    qr = np.full((4, 5), 1e-4, np.float32)
    fields = native(qr=qr)
    receipt = hn.complete_numbers(fields, ("nr",), scheme=None, xland=None)
    assert receipt["cold_zero"] == ["nr"] and not fields["nr"].any()
    assert hn.diagnose_scheme(["--physics-backend=thompson_aero_column"]) == 28
    assert hn.diagnose_scheme(["--physics-backend=thompson_aero_column", "--physics-mode=exact"]) is None
    assert hn.diagnose_scheme(None) is None   # the frozen WSM6 row carries no number


def test_the_region_carries_its_rows_numbers_and_any_the_parent_publishes(tmp_path):
    frame = tmp_path / "cuda-history.nc"
    with Dataset(frame, "w") as d:
        d.createDimension("Time", None)
        d.createDimension("nCells", 5)
        d.createDimension("nVertLevels", 4)
        d.createVariable("nr", "f4", ("Time", "nCells", "nVertLevels"))[0] = np.ones((5, 4))
    assert hn.frame_number_species(frame) == ("nr",)
    # A WSM6 region under a parent that publishes nr still carries it (real data, never dropped)...
    assert hn.region_number_species(None, frame) == ("nr",)
    # ...and a Thompson region carries its whole number row.
    assert hn.region_number_species(["--physics-backend=thompson_aero_column"], frame) == hn.NUMBER_SPECIES


def test_a_boundary_frame_publishes_the_numbers_beside_the_masses(tmp_path):
    from hexcore import nest_boundary
    from hexcore import analysis_remap as transfer
    from datetime import datetime

    nlev, ncell, nedge = 4, 5, 7
    species = transfer.SPECIES + ("ni", "nr", "nc")
    qr = np.zeros((nlev, ncell), np.float32)
    qr[1, 2] = 4.5e-4
    fields = native(nlev, ncell, qr=qr)
    hn.complete_numbers(fields, ("ni", "nr", "nc"), scheme=28, xland=None)
    scalars = np.stack([fields[name] for name in species]).astype(np.float32)
    state = SimpleNamespace(rho=np.ones((nlev, ncell), np.float32), rho_u=np.zeros((nlev, nedge), np.float32),
                            rho_w=np.zeros((nlev + 1, ncell), np.float32), scalars=scalars)
    saved = SimpleNamespace(theta_m=fields["theta_m"], normal_velocity=np.zeros((nlev, nedge), np.float32),
                            vertical_velocity=np.zeros((nlev + 1, ncell), np.float32))
    vertical = SimpleNamespace(zz=np.ones((nlev, ncell), np.float32))
    out = tmp_path / "lbc"
    out.mkdir()
    nest_boundary.publish_frame(out, datetime(2024, 5, 6, 19), state, saved, vertical, species,
                                dict(parent="p.nc", parent_sha256="0" * 64))
    (path,) = out.glob("*.nc")
    with Dataset(path) as d:
        assert {"lbc_ni", "lbc_nr", "lbc_nc", "lbc_qr"} <= set(d.variables)
        lbc_nr = np.asarray(d["lbc_nr"][0]).T
        lbc_qr = np.asarray(d["lbc_qr"][0]).T
    # No boundary point brings rain mass in without a rain number.
    assert not ((lbc_qr > 0) & (lbc_nr <= 0)).any() and lbc_nr[1, 2] > 0


def _loaded(fields, species):
    scalars = np.stack([fields[name] for name in species]).astype(np.float32)
    return SimpleNamespace(state=SimpleNamespace(scalars=scalars), saved_diagnostics=None, vertical=None)


def test_an_exact_region_under_a_numberless_parent_publishes_no_number(tmp_path, monkeypatch):
    """Release review 2.9.7 MEDIUM 21: with nothing carried or diagnosed (exact mode, or no Thompson relations, under
    the default WSM6 parent that publishes no number), ni/nr/nc were written as +0 boundary planes and driven at the
    rings as number 0 under the driven mass.  They are now left out, so the region keeps the six-mass boundary."""
    from datetime import datetime
    from hexcore import analysis_remap as transfer
    from hexcore import nest_boundary

    nlev, ncell, nedge = 4, 5, 7
    qr = np.full((nlev, ncell), 1e-4, np.float32)
    fields = native(nlev, ncell, qr=qr)
    monkeypatch.setattr(transfer, "native_fields", lambda state, saved, vertical, species: fields)
    loaded = _loaded(fields, transfer.SPECIES)
    scheme = hn.diagnose_scheme(["--physics-backend=thompson_aero_column", "--physics-mode=exact"])
    assert scheme is None
    order, receipt = hn.complete_state_numbers(loaded, transfer.SPECIES, ("ni", "nr", "nc", "nwfa", "nifa"),
                                               scheme=scheme, xland=None, prefix=True)
    assert order == transfer.SPECIES and receipt["published"] == []
    assert loaded.state.scalars.shape[0] == len(transfer.SPECIES)
    monkeypatch.undo()
    state = SimpleNamespace(rho=np.ones((nlev, ncell), np.float32), rho_u=np.zeros((nlev, nedge), np.float32),
                            rho_w=np.zeros((nlev + 1, ncell), np.float32), scalars=loaded.state.scalars)
    saved = SimpleNamespace(theta_m=fields["theta_m"], normal_velocity=np.zeros((nlev, nedge), np.float32),
                            vertical_velocity=np.zeros((nlev + 1, ncell), np.float32))
    out = tmp_path / "lbc"
    out.mkdir()
    nest_boundary.publish_frame(out, datetime(2024, 5, 6, 19), state, saved,
                                SimpleNamespace(zz=np.ones((nlev, ncell), np.float32)), order,
                                dict(parent="p.nc", parent_sha256="0" * 64))
    (path,) = out.glob("*.nc")
    with Dataset(path) as d:
        assert not {"lbc_ni", "lbc_nr", "lbc_nc", "lbc_nwfa", "lbc_nifa"} & set(d.variables)
        assert "lbc_qr" in d.variables


def test_a_boundary_series_publishes_only_the_contiguous_prefix_it_has(monkeypatch):
    """A carried aerosol number after a cold hydrometeor number is not published on the ring (the door drives the
    contiguous prefix of the row's block); diagnosed numbers are."""
    from hexcore import analysis_remap as transfer

    fields = native(qr=np.full((4, 5), 1e-4, np.float32))
    fields["nwfa"] = np.full((4, 5), 1.0e9, np.float32)
    monkeypatch.setattr(transfer, "native_fields", lambda state, saved, vertical, species: fields)
    species = transfer.SPECIES + ("nwfa",)
    order, receipt = hn.complete_state_numbers(_loaded(fields, species), species, ("ni", "nr", "nc", "nwfa"),
                                               scheme=None, xland=None, prefix=True)
    assert order == transfer.SPECIES and receipt["carried"] == ["nwfa"]
    # Diagnosed (fast mode, Thompson row): ni/nr/nc ride; nifa (cold) is cut.
    fields = native(qr=np.full((4, 5), 1e-4, np.float32))
    order, receipt = hn.complete_state_numbers(_loaded(fields, transfer.SPECIES), transfer.SPECIES,
                                               ("ni", "nr", "nc", "nifa"), scheme=28, xland=None, prefix=True)
    assert order == transfer.SPECIES + ("ni", "nr", "nc")
    # The analysis (no prefix rule) keeps a carried number past a cold one.
    fields = native(qr=np.full((4, 5), 1e-4, np.float32))
    fields["nwfa"] = np.full((4, 5), 1.0e9, np.float32)
    order, _ = hn.complete_state_numbers(_loaded(fields, species), species, ("ni", "nr", "nc", "nwfa"),
                                         scheme=None, xland=None)
    assert order == transfer.SPECIES + ("nwfa",)
