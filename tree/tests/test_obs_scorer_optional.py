"""A build without the observation scorer installs, and says so by name.

THE BREAKAGE: setup.py demanded hex_obs_scorecard and built it through the
installed gpuwm's tools/rustwx, which an installed gpuwm wheel does not have,
so `pip install gpuwm-hex` from the sdist failed on every host the manylinux
wheel does not cover (golden 0.4.0 verifier, G3 sdist-build.log).  The build
now leaves the scorer out with a printed reason; these tests hold the runtime
half: the score path refuses by name and doctor reports a non-blocking gap.
"""

from pathlib import Path

import pytest

from hexcore import doctor, observation_scorecard


def _no_scorer(monkeypatch, tmp_path):
    monkeypatch.delenv("GPUWM_HEX_OBS_SCORECARD", raising=False)
    fake_module = tmp_path / "observation_scorecard.py"
    monkeypatch.setattr(observation_scorecard, "__file__", str(fake_module))


def test_a_missing_scorer_is_refused_by_name(monkeypatch, tmp_path):
    _no_scorer(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="hex_obs_scorecard is not in this install"):
        observation_scorecard.binary("hex_obs_scorecard")


def test_doctor_reports_a_missing_scorer_without_blocking_the_forecast(monkeypatch, tmp_path):
    _no_scorer(monkeypatch, tmp_path)
    finding = doctor._check_observation_scorer()
    assert finding.status == doctor.MISSING
    assert finding.required is False
    assert "NOT_MEASURED" in finding.detail
    assert finding not in doctor.blocking_gaps([finding])


def test_doctor_finds_a_named_scorer(monkeypatch, tmp_path):
    scorer = tmp_path / "hex_obs_scorecard"
    scorer.write_bytes(b"")
    monkeypatch.setenv("GPUWM_HEX_OBS_SCORECARD", str(scorer))
    finding = doctor._check_observation_scorer()
    assert finding.status == doctor.VERIFIED
    assert Path(finding.evidence["path"]) == scorer.resolve()


def test_setup_builds_the_scorer_only_from_a_source_checkout():
    text = (Path(__file__).resolve().parents[1] / "setup.py").read_text(encoding="utf-8")
    assert "def engine_source()" in text
    assert "hex_obs_scorecard NOT BUILT" in text
    assert "if scorer.is_file():" in text
