"""A door receipt names the machine it ran on by a salted digest, never by name.

THE BREAKAGE THIS PREVENTS: receipts recorded ``platform.node()``, so a
receipt a user shared carried their machine's network name.  No card, no
assets.
"""

from __future__ import annotations

import platform

from hexcore import host_identity


def test_the_record_carries_no_machine_name(tmp_path, monkeypatch):
    monkeypatch.setattr(host_identity, "SALT_PATH", tmp_path / "salt")
    record = host_identity.host_record()
    assert platform.node() not in str(record)
    assert record["scope"] == "user"
    assert len(record["sha256"]) == 16


def test_one_user_gets_one_digest_across_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(host_identity, "SALT_PATH", tmp_path / "salt")
    first = host_identity.host_record()
    assert host_identity.host_record() == first
    # a different user's salt names the same machine differently
    monkeypatch.setattr(host_identity, "SALT_PATH", tmp_path / "other-salt")
    assert host_identity.host_record()["sha256"] != first["sha256"]


def test_an_unwritable_salt_falls_back_to_a_per_run_salt(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    monkeypatch.setattr(host_identity, "SALT_PATH", blocker / "salt")
    monkeypatch.setattr(host_identity, "_PROCESS_SALT", None)
    record = host_identity.host_record()
    assert record["scope"] == "run"
    assert platform.node() not in str(record)
    assert host_identity.host_record() == record


def test_no_door_writes_the_machine_name():
    import inspect

    from hexcore import forecast_door, init_door, pair_door

    for module in (forecast_door, init_door, pair_door):
        source = inspect.getsource(module)
        assert "platform.node()" not in source, module.__name__
        assert "host_record()" in source, module.__name__
