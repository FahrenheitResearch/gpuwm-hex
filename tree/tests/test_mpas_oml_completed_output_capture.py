"""Storage regressions for completed OML proof files; no forecast claims."""
import errno
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'oml_capture', Path(__file__).with_name('oml_preserving_comparator.py'))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def test_completed_capture_survives_verifier_unlink_without_duplicate_blocks(tmp_path):
    source = tmp_path / 'completed-output.bin'
    retained = tmp_path / 'retained-output.bin'
    source.write_bytes(b'opaque completed artifact bytes\0\x01\xff')
    expected = capture.sha(source)
    assert capture.retain_completed_file(source, retained) == 'hardlink'
    assert source.samefile(retained)
    source.unlink()
    assert capture.sha(retained) == expected
    assert retained.stat().st_nlink == 1


def test_actual_cross_device_error_copies_identical_bytes(tmp_path, monkeypatch):
    source, retained = tmp_path / 'source.bin', tmp_path / 'retained.bin'
    source.write_bytes(b'opaque artifact on another device')

    def cross_device(*args):
        raise OSError(errno.EXDEV, 'Cross-device link')

    monkeypatch.setattr(capture.os, 'link', cross_device)
    assert capture.retain_completed_file(source, retained) == 'cross-device-copy'
    assert not source.samefile(retained)
    assert capture.sha(source) == capture.sha(retained)


def test_existing_proof_is_never_overwritten(tmp_path):
    source, retained = tmp_path / 'source.bin', tmp_path / 'retained.bin'
    source.write_bytes(b'new bytes')
    retained.write_bytes(b'prior proof')
    with pytest.raises(ValueError, match='existing candidate capture'):
        capture.retain_completed_file(source, retained)
    assert retained.read_bytes() == b'prior proof'


def test_non_cross_device_failure_is_preserved(tmp_path, monkeypatch):
    source, retained = tmp_path / 'source.bin', tmp_path / 'retained.bin'
    source.write_bytes(b'opaque artifact')

    def denied(*args):
        raise PermissionError(errno.EACCES, 'Permission denied')

    monkeypatch.setattr(capture.os, 'link', denied)
    with pytest.raises(PermissionError):
        capture.retain_completed_file(source, retained)
    assert not retained.exists()
