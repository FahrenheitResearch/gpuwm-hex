"""``doctor`` and the doors agree about a bridge that is present and not
executable.

THE BREAKAGE THIS PREVENTS, measured on a pip install of 0.3.1: the engine
wheel's bridge binaries arrived mode 664, ``gpuwm-hex doctor`` reported
every one of them found, and the door refused the same file as not
executable.  ``engines.locate`` (what doctor reports through) now answers
the way ``engines.resolve`` (what the doors run through) does.
"""

from __future__ import annotations

import os

import pytest

from hexcore import doctor, engines

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="executable bits exist only on POSIX"
)


def _stage(monkeypatch, tmp_path, mode: int):
    spec = engines.INIT
    staged = tmp_path / engines.executable_name(spec.name)
    staged.write_bytes(b"")
    staged.chmod(mode)
    for name in (*spec.env_names, spec.gpuwm_env):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda _spec: (staged,))
    return spec, staged


def test_a_bridge_without_the_executable_bit_is_a_gap_with_the_chmod(
    monkeypatch, tmp_path
):
    spec, staged = _stage(monkeypatch, tmp_path, 0o644)

    path, source = engines.locate(spec)
    assert path is None
    assert "not executable" in source
    assert engines.chmod_remedy(source) == f"chmod +x {staged}"

    with pytest.raises(engines.EngineRefusal, match="not executable"):
        engines.resolve(spec)

    findings = [
        finding
        for finding in doctor.check_engines()
        if finding.subject.startswith(spec.name + " ")
    ]
    assert len(findings) == 1
    assert findings[0].status == doctor.MISSING
    assert findings[0].remedy == f"chmod +x {staged}"
    assert findings[0] in doctor.blocking_gaps(findings)


def test_an_executable_bridge_is_found_by_both(monkeypatch, tmp_path):
    spec, staged = _stage(monkeypatch, tmp_path, 0o755)

    path, source = engines.locate(spec)
    assert path == staged.resolve()
    assert engines.chmod_remedy(source) is None
    assert engines.resolve(spec) == staged.resolve()


def test_an_environment_variable_naming_an_unexecutable_file_is_a_gap(
    monkeypatch, tmp_path
):
    spec, staged = _stage(monkeypatch, tmp_path, 0o644)
    monkeypatch.setenv(spec.env_names[0], str(staged))

    path, source = engines.locate(spec)
    assert path is None
    assert source.startswith(f"${spec.env_names[0]} names {staged}")
    assert engines.chmod_remedy(source) == f"chmod +x {staged}"
