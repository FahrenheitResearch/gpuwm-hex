from hexcore import forecast_door
from hexcore.drivers import run_cuda_v841_forecast as driver


def test_hourly_frames_count_fractional_steps_without_truncation(monkeypatch):
    assert forecast_door._schedule(6, 60, 18.75, "test-mesh") == (1152, 7)
    monkeypatch.setattr(driver, "DT_SECONDS", 18.75)
    result = driver.build_schedule(hours=6, history_every_minutes=60,
                                   start_text="2026-10-03_12:00:00")
    assert result["capture_steps"] == [0, 192, 384, 576, 768, 960, 1152]


def test_lagged_daily_frames_keep_original_clock(monkeypatch):
    monkeypatch.setattr(driver, "DT_SECONDS", 600.)
    result = driver.build_schedule(hours=738, history_every_minutes=1440,
                                   history_from_minutes=1080, start_text="2026-10-15_06:00:00")
    assert len(result["capture_steps"]) == 31
    assert result["valid_times"][result["capture_steps"][0]] == "2026-10-16_00:00:00"
    assert result["valid_times"][result["capture_steps"][-1]] == "2026-11-15_00:00:00"
