"""Cycle-speed helpers: file-version digests, delivered map names, vertical cache dir."""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import pytest

from hexcore import file_digest
from hexcore.init_door import vertical_cache_dir
from hexcore.render_door import default_label, product_file_name


@pytest.fixture()
def digest_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_HEX_DIGEST_CACHE", str(tmp_path / "digests"))
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)
    return tmp_path / "digests"


def _age(path, seconds=60):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_digest_is_the_real_sha256_and_is_reused_for_the_same_version(tmp_path, digest_cache, monkeypatch):
    data = tmp_path / "grid.nc"
    data.write_bytes(b"x" * 4096)
    _age(data)
    want = hashlib.sha256(b"x" * 4096).hexdigest()
    assert file_digest.sha256_file(data) == want
    calls = []
    monkeypatch.setattr(file_digest, "_hash", lambda path: calls.append(path) or "never")
    assert file_digest.sha256_file(data) == want  # same version: no read
    assert calls == []


def test_any_rewrite_is_hashed_again(tmp_path, digest_cache):
    data = tmp_path / "init.nc"
    data.write_bytes(b"a" * 4096)
    _age(data)
    first = file_digest.sha256_file(data)
    data.write_bytes(b"b" * 4096)  # same size, new content
    assert file_digest.sha256_file(data) == hashlib.sha256(b"b" * 4096).hexdigest() != first
    # Two writes inside one timestamp tick share size, mtime and ctime; a
    # fresh version is never remembered, so the second write is still seen.
    data.write_bytes(b"c" * 4096)
    assert file_digest.sha256_file(data) == hashlib.sha256(b"c" * 4096).hexdigest()
    stamp = data.stat()
    os.utime(data, ns=(stamp.st_atime_ns, stamp.st_mtime_ns - 60 * 10**9))  # ctime moves, mtime goes back
    file_digest.sha256_file(data)
    data.write_bytes(b"d" * 4096)
    assert file_digest.sha256_file(data) == hashlib.sha256(b"d" * 4096).hexdigest()


def test_digest_cache_off_never_reuses(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_HEX_DIGEST_CACHE", "off")
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)
    data = tmp_path / "static.nc"
    data.write_bytes(b"s" * 100)
    file_digest.sha256_file(data)
    calls = []
    monkeypatch.setattr(file_digest, "_hash", lambda path: calls.append(path) or "h")
    assert file_digest.sha256_file(data) == "h" and len(calls) == 1


def test_delivered_names_carry_model_mesh_cycle_lead_and_product():
    engine = ("rustwx_wrf_20261010_12z_f003_d01-15.092km_500mb_height_winds_valid_20261010_150000z_"
              "lead_003h00m00s.png")
    assert product_file_name("hex_x1.2621442", engine, "500mb_height_winds") == \
        "hex_x1.2621442_20261010_12z_f003_500mb_height_winds.png"
    half = ("rustwx_wrf_20261010_0z_f000_d01-15.092km_2m_temperature_valid_20261010_003000z_"
            "lead_000h30m00s.png")
    assert product_file_name("hex_x1.2621442", half, "2m_temperature") == \
        "hex_x1.2621442_20261010_00z_f000h30m_2m_temperature.png"
    assert product_file_name("hex", "odd.png", "x") == "odd.png"


def test_default_label_drops_role_tokens():
    assert default_label(Path("/m/x1.2621442.static.nc")) == "hex_x1.2621442"
    assert default_label(Path("/meshes/global15/ordered.static.nc")) == "hex_global15"
    assert default_label(None) == "hex"


def test_vertical_cache_dir_default_off_and_explicit(tmp_path, monkeypatch):
    args = argparse.Namespace(vertical_artifact=None, vertical_cache=None)
    monkeypatch.delenv("GPUWM_HEX_VERTICAL_CACHE", raising=False)
    monkeypatch.setenv("GPUWM_HEX_CACHE_ROOT", str(tmp_path))
    assert vertical_cache_dir(args) == tmp_path / "vertical"
    monkeypatch.setenv("GPUWM_HEX_VERTICAL_CACHE", "off")
    assert vertical_cache_dir(args) is None
    args.vertical_cache = str(tmp_path / "v")
    assert vertical_cache_dir(args) == tmp_path / "v"
    args.vertical_artifact = tmp_path / "explicit.nc"
    assert vertical_cache_dir(args) is None
