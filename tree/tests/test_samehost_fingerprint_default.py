"""The explicit same-host door fingerprints every step only for the identity gate."""
import argparse
from types import SimpleNamespace

import pytest

from hexcore import forecast_door
from hexcore.drivers import samehost_forecast_v841 as samehost


def _parsed(*argv):
    parser = argparse.ArgumentParser()
    forecast_door.add_forecast_arguments(parser)
    return parser.parse_args(list(argv))


def test_plain_samehost_forecast_records_initial_and_final_boundaries_only():
    arguments = _parsed("--samehost-devices", "0,1")
    assert arguments.samehost_fingerprint_every is None
    assert samehost.resolved_fingerprint_every(arguments) == 0


def test_identity_reference_keeps_every_step(tmp_path):
    arguments = _parsed("--samehost-devices", "0,1", "--samehost-reference", str(tmp_path / "ref.jsonl"))
    assert samehost.resolved_fingerprint_every(arguments) == 1


@pytest.mark.parametrize("value", [0, 1, 576])
def test_explicit_interval_is_kept_with_or_without_reference(tmp_path, value):
    assert samehost.resolved_fingerprint_every(_parsed("--samehost-fingerprint-every", str(value))) == value
    with_reference = _parsed("--samehost-fingerprint-every", str(value), "--samehost-reference", str(tmp_path / "r"))
    assert samehost.resolved_fingerprint_every(with_reference) == value


def test_negative_interval_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        samehost.resolved_fingerprint_every(SimpleNamespace(samehost_fingerprint_every=-1, samehost_reference=None))
