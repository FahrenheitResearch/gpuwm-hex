"""Use the delivered rw-mpas engine for vertical consumer regression tests.

The pool's installed geometry bridge predates hex-vertical-v1.  These tests
exercise this lane's bridge, so their executable travels with the tests rather
than depending on an integrator-owned installation being updated first.
There is no mocked arithmetic or production engine-resolution change.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re

import pytest

PROTOCOL = b"rw_mpas_geometry --protocol hex-vertical-v1"


def _glibc_floor(binary: bytes) -> tuple[int, int] | None:
    found = [(int(a), int(b)) for a, b in re.findall(rb"GLIBC_(\d+)\.(\d+)", binary)]
    return max(found) if found else None


def _host_glibc() -> tuple[int, int] | None:
    name, version = platform.libc_ver()
    if name != "glibc" or not version:
        return None
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def _installed_vertical_bridge() -> Path | None:
    """The installed engine's rw_mpas_geometry, when it speaks hex-vertical-v1."""
    from hexcore.engines import GEOMETRY, resolve

    saved = {name: os.environ.pop(name, None) for name in GEOMETRY.env_names}
    try:
        path = Path(resolve(GEOMETRY))
    except Exception:
        return None
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
    return path if path.is_file() and PROTOCOL in path.read_bytes() else None


def _hex_vertical_missing() -> str | None:
    """None when this host can run the hex-vertical-v1 tests; otherwise why it cannot.

    THE BREAKAGE THIS PREVENTS (release 0.4.0 lane tail, B8, 2026-10-14): a public checkout has
    no delivered test engine, so the vertical tests ran on the installed engine's
    rw_mpas_geometry, and with an engine that predates hex-vertical-v1 25 of them failed with
    the door's "lacks hex-vertical-v1" refusal instead of skipping by name.  Only the tests that
    call the bridge carry the mark; the host-only ones in the same files still run.
    """
    root = Path(__file__).with_name("vertical_rust_fixture")
    if (root / "manifest.json").is_file():
        return None  # the delivered engine serves, or delivered_vertical_engine names why not
    if _installed_vertical_bridge() is not None:
        return None
    return ("needs an rw_mpas_geometry that speaks hex-vertical-v1 (the i1-init-real Rust side, "
            "gpuwm 47c68434e); the installed engine's bridge does not, and this checkout carries "
            "no delivered test engine")


_HEX_VERTICAL_MISSING = _hex_vertical_missing()
requires_hex_vertical = pytest.mark.skipif(_HEX_VERTICAL_MISSING is not None,
                                           reason=_HEX_VERTICAL_MISSING or "")


@pytest.fixture(scope="session")
def delivered_vertical_engine(tmp_path_factory):
    root = Path(__file__).with_name("vertical_rust_fixture")
    if not (root / "manifest.json").is_file():
        # A public checkout does not carry the delivered binary (it embeds its build host's
        # folders); the installed engine's rw_mpas_geometry serves hex-vertical-v1 there.
        return None
    manifest = json.loads((root / "manifest.json").read_text())
    compressed = (root / "rw_mpas_geometry.gz").read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == manifest["archive_sha256"]
    binary = gzip.decompress(compressed)
    assert hashlib.sha256(binary).hexdigest() == manifest["binary_sha256"]
    assert ("GPUWM_BRIDGE_SOURCE_REV=" + manifest["gpuwm_commit"]).encode() in binary
    for protocol in ("hex-geometry-v1", "hex-vertical-v1"):
        assert ("rw_mpas_geometry --protocol " + protocol).encode() in binary
    # THE BREAKAGE (lane H7, public CPU tier in a clean container, 2026-10-14):
    # the delivered binary was linked on a glibc 2.43 host, so on the public
    # runner (glibc 2.39) and every older host all 25 vertical tests failed
    # with "version GLIBC_2.43 not found".  Where it cannot load, the tests run
    # on the installed engine's bridge when that speaks hex-vertical-v1 (gpuwm
    # with the i1 Rust side), and otherwise skip naming both.
    needed, host = _glibc_floor(binary), _host_glibc()
    if needed is not None and (host is None or host < needed):
        installed = _installed_vertical_bridge()
        if installed is not None:
            return installed
        pytest.skip(f"the delivered vertical test engine needs glibc {needed[0]}.{needed[1]} "
                    f"(this host: {'%d.%d' % host if host else 'not glibc'}) and the installed "
                    "gpuwm's rw_mpas_geometry does not speak hex-vertical-v1; install a gpuwm "
                    "that carries it (the i1-init-real Rust side, gpuwm 47c68434e)")
    engine = tmp_path_factory.mktemp("vertical-rust-engine") / "rw_mpas_geometry"
    engine.write_bytes(binary)
    engine.chmod(0o700)
    return engine


@pytest.fixture(autouse=True)
def vertical_rust_engine(delivered_vertical_engine, monkeypatch):
    """Scope the real engine override to modules that import this fixture."""
    if delivered_vertical_engine is not None:
        monkeypatch.setenv("GPUWM_HEX_RW_MPAS_GEOMETRY", str(delivered_vertical_engine))
