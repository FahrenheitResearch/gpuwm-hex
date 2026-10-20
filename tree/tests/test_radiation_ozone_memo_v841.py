"""The ozone latitude-table memo returns the engine's own bits and stays out of the native route."""

from types import SimpleNamespace

import numpy as np
import pytest

from hexcore import radiation_ozone_memo_v841 as memo_module
from hexcore.radiation_ozone_memo_v841 import OzoneLatitudeTableMemo, install_ozone_memo


class _FakeOzone:
    """The engine module's shape: a pure latitude table and a time blend."""

    LEVSIZ = 3

    def __init__(self, native=False):
        self.native = native
        self.table_calls = 0
        self.native_calls = 0
        self.default_climo = SimpleNamespace(scale=np.float32(2.0))

    def _native_ozone_entries(self):
        return (1, 2, 3) if self.native else None

    def load_ozone_climatology(self):
        return self.default_climo

    def interp_ozone_to_latitudes(self, xlat, climo=None):
        self.table_calls += 1
        y = np.asarray(xlat, dtype=np.float32)
        months = np.arange(12, dtype=np.float32)
        levels = np.arange(self.LEVSIZ, dtype=np.float32)
        return (y[:, None, None] * climo.scale + levels[None, :, None] + months[None, None, :]).astype(np.float32)

    def ozn_time_int(self, julday, julian, ozmixm):
        month = int(julian) % 11
        return ozmixm[..., month] * np.float32(0.25) + ozmixm[..., month + 1] * np.float32(0.75)

    def ozn_latitude_time_int(self, julday, julian, xlat, climo=None, *, out=None):
        if self.native:
            self.native_calls += 1
        if climo is None:
            climo = self.load_ozone_climatology()
        y = np.asarray(xlat, dtype=np.float32)
        return self.ozn_time_int(None, julian, self.interp_ozone_to_latitudes(y, climo))


def test_memo_matches_the_python_route_bit_for_bit_and_builds_once():
    reference, wrapped = _FakeOzone(), _FakeOzone()
    memo = OzoneLatitudeTableMemo(wrapped, 2**30)
    lat = np.linspace(-90, 90, 257, dtype=np.float32)
    climo = wrapped.default_climo
    for julian in (0.0, 3.5, 276.52, 364.99):
        expected = reference.ozn_latitude_time_int(1, julian, lat, reference.default_climo)
        got = memo.ozn_latitude_time_int(1, julian, lat.reshape(-1), climo)
        assert got.dtype == np.float32 and got.tobytes() == expected.tobytes()
    assert wrapped.table_calls == 1
    assert memo.stats == {"native_calls": 0, "table_builds": 1, "table_hits": 3, "over_cap_calls": 0}


def test_changed_latitudes_or_climatology_rebuild_the_table():
    wrapped = _FakeOzone()
    memo = OzoneLatitudeTableMemo(wrapped, 2**30)
    lat = np.linspace(-90, 90, 33, dtype=np.float32)
    memo.ozn_latitude_time_int(1, 10.0, lat, wrapped.default_climo)
    moved = lat.copy()
    moved[5] += np.float32(0.5)
    expected = _FakeOzone().ozn_latitude_time_int(1, 10.0, moved, wrapped.default_climo)
    assert memo.ozn_latitude_time_int(1, 10.0, moved, wrapped.default_climo).tobytes() == expected.tobytes()
    other = SimpleNamespace(scale=np.float32(3.0))
    expected = _FakeOzone().ozn_latitude_time_int(1, 10.0, moved, other)
    assert memo.ozn_latitude_time_int(1, 10.0, moved, other).tobytes() == expected.tobytes()
    assert memo.stats["table_builds"] == 3 and wrapped.table_calls == 3


def test_native_route_is_forwarded_untouched():
    wrapped = _FakeOzone(native=True)
    memo = OzoneLatitudeTableMemo(wrapped, 2**30)
    lat = np.linspace(-90, 90, 17, dtype=np.float32)
    memo.ozn_latitude_time_int(1, 10.0, lat, wrapped.default_climo)
    memo.ozn_latitude_time_int(1, 11.0, lat, wrapped.default_climo)
    assert wrapped.native_calls == 2 and memo.stats["native_calls"] == 2 and memo.stats["table_builds"] == 0


def test_table_over_the_cap_runs_the_engine_route_every_call():
    wrapped = _FakeOzone()
    memo = OzoneLatitudeTableMemo(wrapped, 16)
    lat = np.linspace(-90, 90, 17, dtype=np.float32)
    first = memo.ozn_latitude_time_int(1, 10.0, lat, wrapped.default_climo)
    second = memo.ozn_latitude_time_int(1, 10.0, lat, wrapped.default_climo)
    assert first.tobytes() == second.tobytes()
    assert wrapped.table_calls == 2 and memo.stats["over_cap_calls"] == 2 and memo._table is None


def test_other_attributes_forward_to_the_engine_module():
    wrapped = _FakeOzone()
    memo = OzoneLatitudeTableMemo(wrapped, 2**30)
    assert memo.LEVSIZ == 3 and memo.load_ozone_climatology() is wrapped.default_climo
    with pytest.raises(AttributeError):
        OzoneLatitudeTableMemo.__new__(OzoneLatitudeTableMemo).anything


class RRTMGLegacyRadiation:
    def __init__(self, ozone):
        self._ozone = ozone


def _seam(adapter):
    return SimpleNamespace(_driver=SimpleNamespace(radiation_callable=adapter))


def test_install_wraps_once_and_honours_off(monkeypatch):
    monkeypatch.delenv("GPUWM_HEX_OZONE_TABLE", raising=False)
    monkeypatch.delenv("GPUWM_HEX_OZONE_TABLE_MAX_GB", raising=False)
    adapter = RRTMGLegacyRadiation(_FakeOzone())
    receipt = install_ozone_memo(_seam(adapter))
    assert receipt["installed"] and isinstance(adapter._ozone, OzoneLatitudeTableMemo)
    first = adapter._ozone
    assert install_ozone_memo(_seam(adapter))["installed"] and adapter._ozone is first
    monkeypatch.setenv("GPUWM_HEX_OZONE_TABLE", "off")
    plain = RRTMGLegacyRadiation(_FakeOzone())
    assert install_ozone_memo(_seam(plain))["installed"] is False and isinstance(plain._ozone, _FakeOzone)
    monkeypatch.setenv("GPUWM_HEX_OZONE_TABLE", "maybe")
    with pytest.raises(ValueError, match="ambiguous radiation receipt"):
        install_ozone_memo(_seam(plain))


def test_install_leaves_adapters_without_climatological_ozone(monkeypatch):
    monkeypatch.delenv("GPUWM_HEX_OZONE_TABLE", raising=False)
    assert install_ozone_memo(_seam(RRTMGLegacyRadiation(None)))["installed"] is False
    other = type("RRTMGPRadiation", (), {"_ozone": _FakeOzone()})()
    assert install_ozone_memo(_seam(other))["installed"] is False
    assert isinstance(other._ozone, _FakeOzone)


def test_real_engine_python_route_is_reproduced_bit_for_bit():
    wrf_ozone = pytest.importorskip("gpuwm.ingest.wrf_ozone")
    if wrf_ozone._native_ozone_entries() is not None:
        pytest.skip("the native route is present; the memo forwards to it")
    climo = wrf_ozone.load_ozone_climatology()
    lat = np.random.default_rng(7).uniform(-90, 90, 4097).astype(np.float32)
    lat[:3] = (-90.0, 90.0, 0.0)
    memo = OzoneLatitudeTableMemo(wrf_ozone, memo_module._max_bytes())
    for julian in (0.0, 15.2, 276.5208, 364.9):
        expected = wrf_ozone.ozn_latitude_time_int(1, np.float32(julian), lat, climo)
        got = memo.ozn_latitude_time_int(1, np.float32(julian), lat, climo)
        assert got.tobytes() == expected.tobytes()
    assert memo.stats["table_builds"] == 1 and memo.stats["table_hits"] == 3
