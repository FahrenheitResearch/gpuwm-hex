"""The vertical test executable is real, authenticated, and keeps ABI refusal."""
import pytest

from _vertical_rust_fixture import delivered_vertical_engine, vertical_rust_engine
from hexcore.engines import EngineRefusal
from hexcore.rust_geometry import geometry_engine
from hexcore.vertical import _run_rust_vertical


def test_vertical_tests_resolve_the_delivered_engine(delivered_vertical_engine):
    if delivered_vertical_engine is None:
        pytest.skip("this checkout carries no delivered vertical engine (it embeds its build host's "
                    "folders); the vertical tests use the installed engine's rw_mpas_geometry")
    assert geometry_engine()[0] == delivered_vertical_engine


def test_stale_installed_geometry_still_refuses(tmp_path, monkeypatch):
    # A valid old geometry ABI is insufficient for the newly required vertical
    # ABI.  Shipping a test fixture must not weaken the production refusal.
    stale = tmp_path / "rw_mpas_geometry"
    stale.write_bytes(b"rw_mpas_geometry --protocol hex-geometry-v1")
    stale.chmod(0o700)
    monkeypatch.setenv("GPUWM_HEX_RW_MPAS_GEOMETRY", str(stale))
    with pytest.raises(EngineRefusal, match="lacks hex-vertical-v1"):
        _run_rust_vertical({})
