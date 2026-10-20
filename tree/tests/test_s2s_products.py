"""Numerical product checks with an explicit CPU oracle, no runtime CPU fallback."""
from datetime import datetime
import numpy as np
import pytest
from hexcore.s2s_door import S2SError
from hexcore.s2s_products import (ensemble_statistics, integrate_window,
                                 interp_log_pressure, validate_history, windows)


def test_full_windows_only():
    assert ("weeks-3-4", 14, 28) in windows(30)
    assert not any(name == "weeks-5-6" for name, _, _ in windows(30))
    assert ("weeks-5-6", 28, 42) in windows(45)
    assert len([w for w in windows(90) if w[0].startswith("week-")]) == 12


def test_trapezoids_and_cumulative_precipitation():
    samples = [{"temperature2": np.array([t * 2.]), "z500": np.array([5000. + t]),
                "mslp": np.array([1000.]), "precipitation_rate": np.array([10. + t * 3.])}
               for t in range(8)]
    result = integrate_window(np, samples, 24, 7)
    np.testing.assert_allclose(result["temperature2"], 7)
    np.testing.assert_allclose(result["precipitation_rate"], 3)
    np.testing.assert_allclose(result["z500"], 5003.5)
    with pytest.raises(S2SError, match="exact endpoints"):
        integrate_window(np, samples[:-1], 24, 7)
    samples[-1]["precipitation_rate"] = np.array([0.])
    with pytest.raises(S2SError, match="decreased"):
        integrate_window(np, samples, 24, 7)


def test_fixed_roster_probability_and_baseline():
    members = np.array([[1., np.nan, 10.], [3., 5., 20.], [5., 7., 30.]])
    stats = ensemble_statistics(np, members, np.array([2., 4., 20.]))
    assert stats["mean"][0] == 3.
    assert stats["anomaly"][0] == 1.
    assert stats["probability"][0] == 2 / 3
    assert stats["probability"][2] == 1 / 3  # strict greater-than
    assert np.isnan(stats["probability"][1])
    np.testing.assert_allclose(stats["spread"][0], np.sqrt(8 / 3))
    own = ensemble_statistics(np, members)
    np.testing.assert_allclose(own["anomaly"][[0, 2]], 0)
    missing_baseline = ensemble_statistics(np, members, np.array([np.nan, 4., 20.]))
    assert np.isfinite(missing_baseline["mean"][0])
    assert np.isnan(missing_baseline["probability"][0])
    assert np.isnan(missing_baseline["anomaly_spread"][0])
    assert missing_baseline["anomaly_finite_count"][0] == 0


def test_log_pressure_interpolation_and_below_ground_mask():
    p = np.array([[100000., 70000., 40000.], [45000., 30000., 10000.]])
    z = 10000. - 1000 * np.log(p)
    actual = interp_log_pressure(np, p, z)
    np.testing.assert_allclose(actual[0], 10000. - 1000 * np.log(50000.))
    assert np.isnan(actual[1])


def test_missing_history_never_becomes_a_complete_week(tmp_path):
    member = {"run_id": "mem001"}
    with pytest.raises(S2SError, match="31 scheduled histories"):
        validate_history(member, [{"out": str(tmp_path), "members": ["mem001"]}], datetime(2026, 10, 1), 30, 24)


def test_verification_combines_restart_history_on_common_lag_clock(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from hexcore import global_ensemble
    from hexcore.s2s_products import verify_first_ten_days
    from hexcore import s2s_analysis
    monkeypatch.setattr(s2s_analysis, "verify_analyses", lambda *args: {"metric_rows": 0})
    attempts = []
    for name, days, state in (("old", (1, 6), "interrupted"), ("new", (6, 11), "complete")):
        out = tmp_path / name
        root = out / "mem001"
        root.mkdir(parents=True)
        for day in days:
            (root / f"cuda-history.2026-10-{day:02d}_00.00.00.nc").write_bytes(b"history")
        attempts.append({"out": str(out), "state": state, "members": ["mem001"]})
    calls = []
    monkeypatch.setattr(global_ensemble, "_cli", lambda argv, log, env: calls.append(argv) or 3)
    value = {"paths": {"work": str(tmp_path), "grid": "grid.nc"}, "score_config": {"mesh": "mesh"},
             "start_time": "2026-10-01_00:00:00", "members": [{"run_id": "mem001", "init": "older-init.nc"}]}
    result = verify_first_ten_days(value, {"attempts": attempts}, SimpleNamespace(), {})
    row = result["members"][0]
    assert row["frames"] == 3
    assert (tmp_path / "members/mem001/verification-history/cuda-history.2026-10-06_00.00.00.nc").resolve().parent.parent.name == "new"
    config = json.loads((tmp_path / "members/mem001/verification.config.json").read_text())
    assert config["start"] == value["start_time"]
    assert config["init"] == "older-init.nc"
    assert calls[0][-1] == "24,48,72,96,120,144,168,192,216,240"


def test_era5_calendar_crosses_leap_day_and_year_boundary():
    from hexcore.s2s_climatology import calendar_samples
    leap = calendar_samples(datetime(2024, 2, 28, 18), 0, 1)
    assert leap == [(18, 59), (0, 60), (6, 60), (12, 60), (18, 60)]
    year = calendar_samples(datetime(2023, 12, 31, 18), 0, 1)
    assert year == [(18, 365), (0, 1), (6, 1), (12, 1), (18, 1)]


@pytest.mark.parametrize("dangling", [False, True])
def test_verification_recovers_interrupted_history_link(tmp_path, monkeypatch, dangling):
    from types import SimpleNamespace
    from hexcore import global_ensemble, s2s_analysis
    from hexcore.s2s_products import verify_first_ten_days

    name = "cuda-history.2026-10-02_00.00.00.nc"
    old = tmp_path / "old" / "mem001" / name
    new = tmp_path / "new" / "mem001" / name
    for path, marker in ((old, b"original"), (new, b"resumed")):
        path.parent.mkdir(parents=True)
        path.write_bytes(marker)
    output = tmp_path / "members" / "mem001" / "verification-history"
    output.mkdir(parents=True)
    target = output / name
    target.symlink_to(old)
    partial = target.with_suffix(".link-part")
    partial.symlink_to(tmp_path / "missing" if dangling else new)
    calls = []
    monkeypatch.setattr(global_ensemble, "_cli", lambda argv, log, env: calls.append(argv) or 3)
    monkeypatch.setattr(s2s_analysis, "verify_analyses", lambda *args: {"metric_rows": 0})
    value = {"paths": {"work": str(tmp_path), "grid": "grid.nc"}, "score_config": {},
             "start_time": "2026-10-01_00:00:00",
             "members": [{"run_id": "mem001", "init": "init.nc"}]}
    attempts = [{"out": str(tmp_path / folder), "state": state, "members": ["mem001"]}
                for folder, state in (("old", "interrupted"), ("new", "complete"))]
    result = verify_first_ten_days(value, {"attempts": attempts}, SimpleNamespace(), {})
    assert result["members"][0]["frames"] == 1 and len(calls) == 1
    assert target.is_symlink() and target.resolve() == new
    assert not partial.is_symlink() and not partial.exists()
    assert old.read_bytes() == b"original" and new.read_bytes() == b"resumed"


def test_era5_window_uses_end_accumulations_and_trapezoidal_states():
    from hexcore.s2s_climatology import reduce_window
    # Independent integral of f(t)=2t+10 on [0,7] is mean 17.
    times = np.arange(29) / 4
    samples = [np.array([2*t+10]) for t in times]
    np.testing.assert_allclose(reduce_window(np, samples, precipitation=False, days=7), 17)
    # Initial six-hour rain belongs to the preceding interval and must be excluded.
    rain = [np.array([999.])] + [np.array([.75])] * 28
    np.testing.assert_allclose(reduce_window(np, rain, precipitation=True, days=7), 3)
