"""`--skip score` on a hex door must not start the automatic observation scorer.

Breakage: `gpuwm-hex global --skip score` still queued the forecast's scorer, whose watcher downloaded the IGRA2
period of record (~50 GB over four 1 h runs on one box). The door now sets GPUWM_HEX_OBSERVATION_SCORING=off and
the forecast hook queues nothing.
"""
import json

from hexcore import observation_scorecard


def test_skip_score_queues_no_watcher(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise AssertionError("the watcher was started although scoring is off")

    monkeypatch.setattr(observation_scorecard, "queue_retry", refuse)
    monkeypatch.setenv(observation_scorecard.SCORING_ENV, "off")
    result = observation_scorecard.forecast_hook(tmp_path, init=tmp_path / "i.nc", grid=tmp_path / "g.nc",
                                                 start="2025-05-19_12:00:00")
    assert result["status"] == "SKIPPED"
    card = json.loads((tmp_path / "observation-scorecard" / "scorecard.json").read_text(encoding="utf-8"))
    assert card["status"] == "SKIPPED" and "--skip score" in card["reason"]
    assert not (tmp_path / "observation-scorecard" / "automatic-retry.json").exists()


def test_the_doors_turn_scoring_off_only_for_skip_score():
    from pathlib import Path

    src = Path(observation_scorecard.__file__).parent
    for door in ("global_door.py", "country_door.py"):
        text = (src / door).read_text(encoding="utf-8")
        assert 'if "score" in set(args.skip or []):' in text, door
        assert 'env["GPUWM_HEX_OBSERVATION_SCORING"] = "off"' in text, door
