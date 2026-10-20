"""tools/release_hygiene.py refuses an artifact that carries private material.

The breakage it prevents (verify-hex-r2, 2026-10-13): the 0.4.0 candidate's
wheel and sdist carried the owner's name, home-directory paths, internal
cloud host names and the live address of a rented box.  The fixtures below
are built in the test, so this file carries no private literal itself.
"""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "release_hygiene.py"


@pytest.fixture(scope="module")
def hygiene():
    spec = importlib.util.spec_from_file_location("release_hygiene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return path


def _sdist(path: Path, files: dict[str, str]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def _rules(hygiene, owner: str):
    return list(hygiene.RULES) + hygiene.identity_rules([owner])


def test_a_clean_artifact_passes(hygiene, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: ["Zebulon"])
    wheel = _wheel(tmp_path / "x-1.0-py3-none-any.whl", {
        "pkg/a.py": "# owner ruling 2026-10-10; loopback 127.0.0.1, private 10.1.2.3\n",
        "pkg/b.txt": "version 2.9.0 and 8.4.2; " + "/wo" + "rk/box/run and /home/runner/work\n",
    })
    assert hygiene.main([str(wheel)]) == 0


# Each sample is assembled at run time: this file ships in the sdist, and the
# gate scans it there, so no sample may appear here as one literal.
_SAMPLES = [
    ("see " + "C:" + "\\Users\\someone\\x", "home-path"),
    ("cd " + "/ho" + "me/someone/x", "home-path"),
    ("cat " + "Down" + "loads/LANE/REPORT.md", "downloads-path"),
    ("host " + "ip-" + "172-31-47-178" + " ran it", "cloud-host-name"),
    ("built on " + "weather-" + "no" + "de-2", "lab-host-name"),
    ("clone in " + "co" + "dex-" + "1007", "workstation-folder"),
    ("ssh root@" + ".".join(["3", "14", "159", "26"]), "public-ipv4"),
    ("ruled by Zebulon on 2026-10-10", "owner-name"),
]


@pytest.mark.parametrize("text,rule", _SAMPLES)
def test_each_class_of_private_material_is_refused_by_name(hygiene, tmp_path, monkeypatch, capsys, text, rule):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: ["zebulon"])
    sdist = _sdist(tmp_path / "x-1.0.tar.gz", {"x-1.0/src/a.py": text + "\n"})
    assert hygiene.main([str(sdist)]) == 1
    printed = capsys.readouterr().out
    assert f":: x-1.0/src/a.py :: {rule} ::" in printed


def test_the_lower_case_login_is_not_an_english_verb_hit(hygiene):
    # a lower-case login that is also an English verb must not trip the gate;
    # the capitalised name does
    rules = _rules(hygiene, "d" + "rew")
    assert hygiene.scan([("a.txt", b"the door drew 92 maps")], rules) == []
    named = ("ruled by " + "D" + "rew").encode()
    assert [h[1] for h in hygiene.scan([("a.txt", named)], rules)] == ["owner-name"]


def test_an_allowed_public_address_is_not_a_hit(hygiene, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: [])
    address = ".".join(["8"] * 4)
    wheel = _wheel(tmp_path / "x-1.0-py3-none-any.whl", {"pkg/a.py": f"server {address}\n"})
    assert hygiene.main([str(wheel)]) == 1
    assert hygiene.main([str(wheel), "--allow-address", address]) == 0


def test_an_unreadable_artifact_is_its_own_exit_code(hygiene, tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, "identity_tokens", lambda extra, where: [])
    bogus = tmp_path / "x-1.0-py3-none-any.whl"
    bogus.write_bytes(b"not a zip")
    assert hygiene.main([str(bogus)]) == 2
