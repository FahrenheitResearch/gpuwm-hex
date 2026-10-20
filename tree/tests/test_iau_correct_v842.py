"""config_physics_mode='correct' IAU (IN1, IN2, IN3): fails on v8.4.2, passes in correct mode,
and exact mode stays word-identical to the pristine module.

Native references (CPU, built on the same host with the pinned strict flags):
  HEXCORE_IAU_NATIVE          pristine v8.4.2 mpas_atm_iau.F, ABI v2 (tests/native_iau_v842/build.sh)
  HEXCORE_IAU_NATIVE_CORRECT  patched mpas_atm_iau.F with config_physics_mode, ABI v3 (build_v3.sh)
Finding ids are those of the 2026-10-13 correct-mode triage (cluster mpas-iau-lbc-init).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from hexcore.iau_v842 import (
    IAUConfig, atm_iau_coef, atm_iau_coef_window, atm_add_tend_anal_incr,
    iau_stream_request, iau_stream_request_correct, iau_window_position, iau_new_increments,
    bind_iau_increment_arrays,
)

from test_iau_v842 import DECKS, _inputs, _load_native  # noqa: E402  (the existing synthetic decks)

OPTION = {"off": 0, "on": 1, "ON": 2, "other": 3}
NAMES = ("zz", "rho_zz", "theta_m", "scalars", "rho_edge", "u_amb", "rho_amb", "theta_amb", "scalars_amb",
         "tend_ru", "tend_rtheta", "tend_rho", "tend_scalars")


def _load_v3(path):
    library = ctypes.CDLL(path)
    function = library.run_iau_v3
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS")
    function.argtypes = ([ctypes.c_int] * 11 + [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [f32] * 10
                         + [i32, ctypes.POINTER(ctypes.c_float)] + [f32] * 4 + [ctypes.POINTER(ctypes.c_float)])
    function.restype = None
    return function


@pytest.fixture(scope="module")
def natives():
    pristine = os.environ.get("HEXCORE_IAU_NATIVE")
    correct = os.environ.get("HEXCORE_IAU_NATIVE_CORRECT")
    if not pristine or not correct:
        pytest.skip("HEXCORE_IAU_NATIVE and HEXCORE_IAU_NATIVE_CORRECT must name the native libraries")
    return _load_native(pristine), _load_v3(correct)


def _mesh_extras(seed, nl, nc, ne, exterior=()):
    """State wind and one-based Fortran cellsOnEdge (2, ne); listed edges get the exterior cell nc+1."""
    rng = np.random.default_rng(1000 + seed)
    u = rng.uniform(-30, 30, (nl, ne)).astype(np.float32)
    cells = np.empty((2, ne), np.int32, order="F")
    for e in range(ne):
        c1, c2 = rng.choice(nc, 2, replace=False) + 1
        cells[:, e] = (c1, c2)
    for e in exterior:
        cells[1, e] = nc + 1
    return u, cells


def _run_v3(native, host, u, cells, deck, mode, elapsed):
    seed, option, window, dt, itime, ncs, nes, ms, me, qv = deck
    fields = {name: np.asfortranarray(host[name].copy()) for name in NAMES}
    nl, nc = host["zz"].shape
    ne = host["rho_edge"].shape[1]
    ns = host["scalars"].shape[0]
    position = ctypes.c_float(elapsed)
    weight = ctypes.c_float()
    args = [fields[n] for n in NAMES[:9]] + [np.asfortranarray(u.copy()), cells, ctypes.byref(position)] \
        + [fields[n] for n in NAMES[9:]] + [ctypes.byref(weight)]
    native(nl, nc, ne, ncs, nes, ns, host["scalars_amb"].shape[0], ms, me, qv, itime, dt, window,
           OPTION[option], {"exact": 0, "correct": 1}[mode], *args)
    return fields, weight.value, position.value


def _run_v2(native, host, deck):
    seed, option, window, dt, itime, ncs, nes, ms, me, qv = deck
    fields = {name: np.asfortranarray(host[name].copy()) for name in NAMES}
    nl, nc = host["zz"].shape
    ne = host["rho_edge"].shape[1]
    ns = host["scalars"].shape[0]
    weight = ctypes.c_float()
    native(nl, nc, ne, ncs, nes, ns, host["scalars_amb"].shape[0], ms, me, qv, itime, dt, window,
           OPTION[option], *fields.values(), ctypes.byref(weight))
    return fields, weight.value


def _deck_host(deck):
    seed = deck[0]
    host = _inputs(seed)
    if seed == 18:
        host["scalars_amb"] = host["scalars_amb"][:6].copy()
    if seed == 13:
        for name in ("u_amb", "rho_amb", "theta_amb", "scalars_amb", "tend_scalars", "tend_rho", "tend_ru",
                     "tend_rtheta"):
            host[name] = np.full_like(host[name], np.float32(1e-38))
        host["u_amb"][0, 0] = np.float32(-1e-40)
    if seed == 14:
        for name in ("u_amb", "rho_amb", "theta_amb", "scalars_amb"):
            host[name].fill(0)
    if seed == 15:
        host["scalars"][deck[9] - 1].fill(0)
    return host


# --------------------------------------------------------------------------------------------------
# exact mode: the patched module is the v8.4.2 module, word for word
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("deck", DECKS, ids=[f"exact-deck-{d[0]}" for d in DECKS])
def test_exact_mode_of_patched_module_is_word_identical_to_pristine(natives, deck):
    pristine, patched = natives
    host = _deck_host(deck)
    nl, nc = host["zz"].shape
    ne = host["rho_edge"].shape[1]
    u, cells = _mesh_extras(deck[0], nl, nc, ne)
    before, w_before = _run_v2(pristine, host, deck)
    after, w_after, position = _run_v3(patched, host, u, cells, deck, "exact", -1.0)
    assert np.float32(w_before).view(np.uint32) == np.float32(w_after).view(np.uint32)
    for name in NAMES:
        np.testing.assert_array_equal(after[name].view(np.uint32), before[name].view(np.uint32), err_msg=name)
    assert position == -1.0  # exact mode never touches the window position


# --------------------------------------------------------------------------------------------------
# IN1: null increments must be a no-op
# --------------------------------------------------------------------------------------------------

def _null_case():
    host = _inputs(7001)
    for name in ("u_amb", "rho_amb", "theta_amb", "scalars_amb"):
        host[name].fill(0)
    # physics left a vapour tendency in tend_scalars (as YSU / Tiedtke do before IAU is applied)
    host["tend_scalars"][0] = np.float32(1e-6)
    return host, (7001, "on", 900.0, 10.0, 1, 7, 13, 1, 6, 1)


def test_IN1_v842_heats_with_all_zero_increments(natives):
    pristine, _ = natives
    host, deck = _null_case()
    out, _ = _run_v2(pristine, host, deck)
    changed = np.count_nonzero(out["tend_rtheta"].view(np.uint32) != host["tend_rtheta"].view(np.uint32))
    assert changed == host["tend_rtheta"].size  # every cell: the physics vapour tendency is counted again


def test_IN1_correct_mode_zero_increments_change_nothing(natives):
    _, patched = natives
    host, deck = _null_case()
    nl, nc = host["zz"].shape
    ne = host["rho_edge"].shape[1]
    u, cells = _mesh_extras(7001, nl, nc, ne)
    out, weight, position = _run_v3(patched, host, u, cells, deck, "correct", 0.0)
    assert weight > 0 and position == 10.0
    for name in ("tend_ru", "tend_rtheta", "tend_rho", "tend_scalars"):
        np.testing.assert_array_equal(out[name].view(np.uint32), host[name].view(np.uint32), err_msg=name)


# --------------------------------------------------------------------------------------------------
# IN2: a density-only increment must not change wind or dry potential temperature
# --------------------------------------------------------------------------------------------------

def _density_case():
    host = _inputs(7002)
    for name in ("u_amb", "theta_amb", "scalars_amb"):
        host[name].fill(0)
    for name in ("tend_ru", "tend_rtheta", "tend_rho", "tend_scalars"):
        host[name].fill(0)
    return host, (7002, "on", 900.0, 10.0, 1, 7, 13, 1, 6, 1)


def _density_expected(host, u, cells, wgt):
    """float64 chain rule for a pure density increment: rho_zz*u and rho_zz*theta_m scale with rho_zz."""
    zz = host["zz"].astype(np.float64)
    drho = host["rho_amb"].astype(np.float64) / zz
    c1, c2 = cells[0] - 1, cells[1] - 1
    ru = wgt * u.astype(np.float64) * 0.5 * (drho[:, c1] + drho[:, c2])
    rtheta = wgt * host["theta_m"].astype(np.float64) * drho
    return ru, rtheta


def test_IN2_v842_density_increment_leaves_momentum_unscaled_and_overheats(natives):
    pristine, _ = natives
    host, deck = _density_case()
    nl, nc = host["zz"].shape
    ne = host["rho_edge"].shape[1]
    u, cells = _mesh_extras(7002, nl, nc, ne)
    out, wgt = _run_v2(pristine, host, deck)
    ru, rtheta = _density_expected(host, u, cells, wgt)
    assert np.all(out["tend_ru"] == 0)            # u*d(rho_edge) missing: the wind changes
    rel = np.abs(out["tend_rtheta"] - rtheta) / np.abs(rtheta)
    assert rel.max() > 1e-3                       # qv*d(rho) counted twice: about rvord*qv too warm


def test_IN2_correct_mode_matches_the_chain_rule(natives):
    _, patched = natives
    host, deck = _density_case()
    nl, nc = host["zz"].shape
    ne = host["rho_edge"].shape[1]
    u, cells = _mesh_extras(7002, nl, nc, ne)
    out, wgt, _ = _run_v3(patched, host, u, cells, deck, "correct", 0.0)
    ru, rtheta = _density_expected(host, u, cells, wgt)
    np.testing.assert_allclose(out["tend_ru"], ru, rtol=2e-6, atol=1e-30)
    np.testing.assert_allclose(out["tend_rtheta"], rtheta, rtol=2e-6)


# --------------------------------------------------------------------------------------------------
# IN3: whole dose for any window and dt; a restart continues the window
# --------------------------------------------------------------------------------------------------

WINDOWS = (1, 9, 10, 15, 601, 899, 900, 901)
STEPS = (1, 7, 10, 30)


def _weights_v842(window, dt, steps):
    config = IAUConfig("on", float(window))
    return [float(atm_iau_coef(config, k, dt)) for k in range(1, steps + 1)]


def _weights_correct(window, dt, steps, elapsed=0.0):
    config = IAUConfig("on", float(window), "correct")
    out, position = [], np.float32(elapsed)
    for _ in range(steps):
        out.append(float(atm_iau_coef_window(config, position, dt)))
        position = np.float32(position + np.float32(dt))
    return out


def test_IN3_v842_dose_is_wrong_for_windows_that_are_not_a_multiple_of_dt():
    wrong = [(w, s) for w in WINDOWS for s in STEPS
             if abs(sum(_weights_v842(w, s, int(2 * w / s) + 3)) * s - 1) > 1e-5]
    assert (601, 10) in wrong and (15, 10) in wrong and len(wrong) >= 20


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("dt", STEPS)
def test_IN3_correct_mode_applies_exactly_one_increment(window, dt):
    dose = sum(_weights_correct(window, dt, int(2 * window / dt) + 3)) * dt
    assert abs(dose - 1) <= 2e-5


def test_IN3_correct_mode_weight_is_v842_weight_inside_a_divisible_window():
    a = _weights_v842(900, 10, 95)
    b = _weights_correct(900, 10, 95)
    assert [np.float32(x).view(np.uint32) for x in a] == [np.float32(x).view(np.uint32) for x in b]


@pytest.mark.parametrize("split", (1, 45, 89))
def test_IN3_restart_continues_the_window(split):
    continuous = _weights_correct(900, 10, 120)
    # restart: the first run applies `split` steps; the restart file carries their position
    first = _weights_correct(900, 10, split)
    start = iau_window_position(IAUConfig("on", 900.0, "correct"), restart=True,
                                stored=float(np.float32(split * 10)))
    assert start.rule == "restart continues the window carried by the restart file"
    second = _weights_correct(900, 10, 120 - split, elapsed=float(start.position))
    assert first + second == continuous
    # v8.4.2 restarts itimestep at 1 and applies the window again: more than one increment in total
    v842 = _weights_v842(900, 10, split) + _weights_v842(900, 10, 120 - split)
    assert sum(v842) * 10 > 1.01
    assert abs(sum(continuous) * 10 - 1) <= 2e-5


def test_IN3_python_weight_is_the_native_weight(natives):
    _, patched = natives
    host = _inputs(7003)
    nl, nc = host["zz"].shape
    ne = host["rho_edge"].shape[1]
    u, cells = _mesh_extras(7003, nl, nc, ne)
    for window, dt, elapsed in ((901, 30, 870), (15, 10, 10), (600, 7, 0), (900, 10, 905), (1, 1, 0.5)):
        deck = (7003, "on", float(window), float(dt), 1, 7, 13, 1, 6, 1)
        _, weight, position = _run_v3(patched, host, u, cells, deck, "correct", float(elapsed))
        mirror = atm_iau_coef_window(IAUConfig("on", float(window), "correct"), elapsed, dt)
        assert np.float32(weight).view(np.uint32) == mirror.view(np.uint32)
        assert position == np.float32(np.float32(elapsed) + np.float32(dt))


def test_IN3_restart_without_window_position_begins_a_new_window():
    # A restart file written in exact mode, by JEDI or with IAU off carries no window: none is in
    # progress, so the restart begins a new window at its start time, as v8.4.2 does.
    config = IAUConfig("on", 900.0, "correct")
    assert iau_window_position(config, restart=False).position == 0
    for stored in (None, -1.0, float("nan")):
        start = iau_window_position(config, restart=True, stored=stored)
        assert start.position == 0 and start.rule == "the restart file carries no IAU window position"
        assert start.dropped_seconds == 0


def _cycle_doses(window, dt, steps, start):
    """Dose applied by one run of `steps` steps that starts at `start` (IAUWindowStart)."""
    return sum(_weights_correct(window, dt, steps, elapsed=float(start.position))) * dt


def test_IN3_da_cycling_restart_applies_the_new_analysis():
    # Refuted 2026-10-13: a restart with config_do_DAcycling = true from the previous cycle's forecast
    # carries that cycle's finished window (3600 s); the new analysis must still be applied in full.
    config = IAUConfig("on", 3600.0, "correct")
    cycle1 = iau_window_position(config, restart=False)
    assert abs(_cycle_doses(3600, 1200, 3, cycle1) - 1) <= 2e-5
    cycle2 = iau_window_position(config, restart=True, stored=3600.0, da_cycling=True)
    assert cycle2.rule == "restart with config_do_DAcycling = true (a new DA cycle)"
    assert cycle2.position == 0 and cycle2.dropped_seconds == 0
    assert abs(_cycle_doses(3600, 1200, 3, cycle2) - 1) <= 2e-5
    # the same restart file read as a continuation (no new cycle, no new increments) applies nothing
    assert _cycle_doses(3600, 1200, 3, iau_window_position(config, restart=True, stored=3600.0)) == 0
    # a JEDI-written restart file carries no position: a DA-cycling restart from it runs, it is not refused
    assert iau_window_position(config, restart=True, stored=None, da_cycling=True).position == 0


def test_IN3_new_cycle_cutting_a_window_short_reports_the_dropped_part():
    config = IAUConfig("on", 3600.0, "correct")
    start = iau_window_position(config, restart=True, stored=1200.0, da_cycling=True)
    assert start.position == 0 and start.dropped_seconds == 2400.0
    start = iau_window_position(config, restart=True, stored=1200.0, new_increments=True)
    assert start.position == 0 and start.dropped_seconds == 2400.0
    assert start.rule == "increments named for the restart time"


class _Stream:
    options = {"type": "input"}

    def fields(self, packages, physics, streams):
        return ("xtime", "u_amb", "rho_amb", "theta_amb", "qv_amb")

    def filename(self, when, initial):
        return when.strftime("AmB.%Y-%m-%d_%H.%M.%S.nc")


def test_IN3_correct_mode_reads_the_increments_of_the_window_start():
    config = IAUConfig("on", 3600.0, "correct")
    streams = {"iau": _Stream()}
    start = datetime(2023, 3, 31, 12, 20)
    request = iau_stream_request_correct(config, streams, start_time=start, initial_time=start,
                                         window_elapsed=np.array(1200.0, np.float32))
    assert request.filename == "AmB.2023-03-31_12.00.00.nc"
    assert iau_stream_request_correct(config, streams, start_time=start, initial_time=start,
                                      window_elapsed=np.array(3600.0, np.float32)) is None
    with pytest.raises(ValueError, match="window start"):
        iau_stream_request(config, streams, start_time=start, initial_time=start)


class _FixedNameStream(_Stream):
    def filename(self, when, initial):
        return "AmB.nc"


def test_IN3_increments_named_for_the_restart_time_begin_a_new_window():
    # Rule (4): an increment file named for the restart time that is not the carried window's file.
    config = IAUConfig("on", 3600.0, "correct")
    streams = {"iau": _Stream()}
    restart = datetime(2023, 3, 31, 13, 0)
    present = {"AmB.2023-03-31_12.00.00.nc", "AmB.2023-03-31_13.00.00.nc"}
    new = iau_new_increments(config, streams, start_time=restart, initial_time=restart, stored=3600.0,
                             exists=present.__contains__)
    assert new
    start = iau_window_position(config, restart=True, stored=3600.0, new_increments=new)
    assert start.position == 0
    request = iau_stream_request_correct(config, streams, start_time=restart, initial_time=restart,
                                         window_elapsed=start.position)
    assert request.filename == "AmB.2023-03-31_13.00.00.nc"
    # no file for the restart time: the carried window continues (and is complete here)
    assert not iau_new_increments(config, streams, start_time=restart, initial_time=restart, stored=3600.0,
                                  exists={"AmB.2023-03-31_12.00.00.nc"}.__contains__)
    # a fixed file name is the carried window's own file: it marks no new window
    assert not iau_new_increments(config, {"iau": _FixedNameStream()}, start_time=restart,
                                  initial_time=restart, stored=1200.0, exists=lambda name: True)
    # mid-window continuation: no new file, the carried window keeps its position and its own increments
    mid = datetime(2023, 3, 31, 12, 20)
    assert not iau_new_increments(config, streams, start_time=mid, initial_time=mid, stored=1200.0,
                                  exists={"AmB.2023-03-31_12.00.00.nc"}.__contains__)
    start = iau_window_position(config, restart=True, stored=1200.0)
    assert start.position == np.float32(1200.0) and start.dropped_seconds == 0
    assert iau_stream_request_correct(config, streams, start_time=mid, initial_time=mid,
                                      window_elapsed=start.position).filename == "AmB.2023-03-31_12.00.00.nc"


def test_mode_key_is_refused_unless_exact_or_correct():
    assert IAUConfig.from_namelist("&nhyd_model config_physics_mode='correct' /").config_physics_mode == "correct"
    assert IAUConfig.from_namelist("&nhyd_model config_dt=60. /").config_physics_mode == "exact"
    with pytest.raises(ValueError, match="config_physics_mode"):
        IAUConfig.from_namelist("&nhyd_model config_physics_mode='corect' /")


# --------------------------------------------------------------------------------------------------
# CUDA correct mode against the patched native module, every output word
# --------------------------------------------------------------------------------------------------

CORRECT_DECKS = [d for d in DECKS if d[1] == "on" and d[2] > 0] + [
    (31, "on", 901.0, 30.0, 1, 7, 13, 2, 7, 4),   # partial final step reached through the position below
    (32, "on", 900.0, 10.0, 1, 7, 13, 1, 6, 1),   # exterior (limited-area) edges
]
POSITIONS = {31: 870.0}
EXTERIOR = {32: (0, 5, 12)}


@pytest.fixture(scope="module")
def replay():
    native = os.environ.get("HEXCORE_IAU_NATIVE_CORRECT")
    if not native:
        pytest.skip("HEXCORE_IAU_NATIVE_CORRECT must name the patched native IAU library")
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    cache = KernelCache()
    records = []
    yield cp, _load_v3(native), cache, records
    destination = os.environ.get("HEXCORE_IAU_CORRECT_RECEIPT")
    if destination:
        Path(destination).write_text(json.dumps({
            "tier": "standalone whole-module CUDA/native replay, config_physics_mode='correct'; not forecast proof",
            "native_library_sha256": hashlib.sha256(Path(native).read_bytes()).hexdigest(),
            "port_source_sha256": hashlib.sha256(
                (Path(__file__).parents[1] / "src/hexcore/iau_v842.py").read_bytes()).hexdigest(),
            "process_count": 1, "records": records, "compile_manifest": cache.compile_manifest(),
        }, indent=2) + "\n")


@pytest.mark.gpu
@pytest.mark.parametrize("deck", CORRECT_DECKS, ids=[f"correct-deck-{d[0]}" for d in CORRECT_DECKS])
def test_cuda_correct_mode_words_match_patched_native(replay, deck):
    cp, native, cache, records = replay
    seed, option, window, dt, itime, ncs, nes, ms, me, qv = deck
    host = _deck_host(deck) if seed < 31 else _inputs(seed)
    nl, nc = host["zz"].shape
    ne = host["rho_edge"].shape[1]
    u, cells = _mesh_extras(seed, nl, nc, ne, EXTERIOR.get(seed, ()))
    start = POSITIONS.get(seed, float(max(itime - 1, 0)) * dt)
    expected, weight, position = _run_v3(native, host, u, cells, deck, "correct", start)

    device = {name: cp.asarray(host[name]) for name in NAMES}
    zero_based = np.ascontiguousarray((cells - 1).T.astype(np.int32))
    zero_based[zero_based >= nc] = nc  # the exterior cell, outside [0, nc)
    pools = {
        "mesh": {"nVertLevels": nl, "nCells": nc, "nEdges": ne, "nCellsSolve": ncs, "nEdgesSolve": nes,
                 "zz": device["zz"], "cellsOnEdge": cp.asarray(zero_based)},
        "state": {"theta_m": device["theta_m"], "rho_zz": device["rho_zz"], "scalars": device["scalars"],
                  "u": cp.asarray(u), "moist_start": ms, "moist_end": me, "index_qv": qv},
        "diag": {"rho_edge": device["rho_edge"], "iau_window_elapsed": np.array(start, np.float32)},
        "tend": {"scalars_tend": device["tend_scalars"]},
        "tend_iau": bind_iau_increment_arrays(device),
    }
    config = IAUConfig(option, window, "correct")
    atm_add_tend_anal_incr(config, pools, itime, dt, device["tend_ru"], device["tend_rtheta"],
                           device["tend_rho"], cache=cache)
    differing = {}
    for name in NAMES:
        got = cp.asnumpy(device[name])
        differing[name] = int(np.count_nonzero(got.view(np.uint32) != expected[name].view(np.uint32)))
        np.testing.assert_array_equal(got.view(np.uint32), expected[name].view(np.uint32), err_msg=name)
    assert pools["diag"]["iau_window_elapsed"] == np.float32(position)
    records.append({"deck": list(deck), "start_position": start, "native_weight": weight,
                    "differing_words": differing})
