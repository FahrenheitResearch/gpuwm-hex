"""Keep the radiation ozone table when the engine's native route is absent.

Breakage this removes (measured, RTX PRO 6000 host, 2026-10-12, global 15 km on 8 RTX PRO
6000 cards, integrate/hex-next 53559c5 + engine d547dc8c3): every radiation
call of every rank ran ``gpuwm.ingest.wrf_ozone.ozn_latitude_time_int`` on
its Python route, because the engine checkout had no built CPU bridge
(``libgpuwm_preprocess_cpu.so``; box cleanups sweep ``target/``, and an
override that names a missing file falls back without a word).  That route
re-interpolates the whole 12-month climatology to every column's latitude on
every call -- 1.7 s alone at 327,680 columns, 8-12 s with eight rank threads
doing it at once -- so a radiation step took 9.1 s against 0.82 s for an
ordinary step: 17 of the 53 s per forecast hour.

The latitude table depends only on the columns' latitudes and the
climatology, neither of which changes during a run.  This module memoizes
that one pure engine call per adapter and leaves the arithmetic to the
engine: the Python route is ``ozn_time_int(None, julian,
interp_ozone_to_latitudes(xlat, climo))`` and the memo evaluates exactly that
expression with the same two engine functions, calling the second with the
table the first returned earlier for identical inputs.  When the native
route is available nothing here runs: the call is forwarded untouched.

Cost: one float32 table of columns x 59 x 12 on the host per rank (0.86 GiB
at 327,680 columns).  Above ``GPUWM_HEX_OZONE_TABLE_MAX_GB`` (default 4) the
call is forwarded untouched.  ``GPUWM_HEX_OZONE_TABLE=off`` disables the memo.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

_REQUIRED = ("_native_ozone_entries", "interp_ozone_to_latitudes", "ozn_time_int",
             "ozn_latitude_time_int", "load_ozone_climatology", "LEVSIZ")


def _max_bytes() -> int:
    text = os.environ.get("GPUWM_HEX_OZONE_TABLE_MAX_GB", "4")
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if value <= 0:
        raise ValueError("GPUWM_HEX_OZONE_TABLE_MAX_GB must be a positive number; "
                         "prevents an unbounded host table")
    return int(value * 2**30)


class OzoneLatitudeTableMemo:
    """``gpuwm.ingest.wrf_ozone`` with its pure latitude table kept between calls."""

    def __init__(self, module: Any, max_bytes: int) -> None:
        self._module = module
        self._max_bytes = int(max_bytes)
        self._latitudes: np.ndarray | None = None
        self._climatology: Any = None
        self._table: np.ndarray | None = None
        self.stats = {"native_calls": 0, "table_builds": 0, "table_hits": 0, "over_cap_calls": 0}

    def __getattr__(self, name: str) -> Any:
        module = self.__dict__.get("_module")
        if module is None:  # a half-built copy must not recurse into itself
            raise AttributeError(name)
        return getattr(module, name)

    def ozn_latitude_time_int(self, julday, julian, xlat, climo=None, *, out=None):
        module = self._module
        if module._native_ozone_entries() is not None:
            # The engine's own fast route (and its own fallbacks inside it).
            self.stats["native_calls"] += 1
            return module.ozn_latitude_time_int(julday, julian, xlat, climo, out=out)
        # The engine's Python route, statement for statement.
        del julday
        if climo is None:
            climo = module.load_ozone_climatology()
        y = np.asarray(xlat, dtype=np.float32)
        if y.size * int(module.LEVSIZ) * 12 * 4 > self._max_bytes:
            self.stats["over_cap_calls"] += 1
            return module.ozn_time_int(None, julian, module.interp_ozone_to_latitudes(y, climo))
        if (self._table is None or self._climatology is not climo
                or self._latitudes is None or self._latitudes.shape != y.shape
                or not np.array_equal(self._latitudes, y)):
            self._table = None
            table = module.interp_ozone_to_latitudes(y, climo)
            self._latitudes = np.array(y, copy=True)
            self._climatology = climo
            self._table = table
            self.stats["table_builds"] += 1
        else:
            self.stats["table_hits"] += 1
        return module.ozn_time_int(None, julian, self._table)

    def receipt(self) -> dict[str, Any]:
        return {"installed": True, **self.stats,
                "table_bytes": 0 if self._table is None else int(self._table.nbytes),
                "max_bytes": self._max_bytes}


def install_ozone_memo(seam: Any) -> dict[str, Any]:
    """Wrap the seam's legacy RRTMG adapter ozone module; return a receipt."""
    setting = os.environ.get("GPUWM_HEX_OZONE_TABLE", "auto").strip().lower()
    if setting not in ("auto", "off"):
        raise ValueError("GPUWM_HEX_OZONE_TABLE must be auto or off; prevents an ambiguous radiation receipt")
    driver = getattr(seam, "_driver", None)
    adapter = getattr(driver, "radiation_callable", None)
    if setting == "off":
        return {"installed": False, "reason": "GPUWM_HEX_OZONE_TABLE=off"}
    if adapter is None or type(adapter).__name__ != "RRTMGLegacyRadiation":
        return {"installed": False, "reason": "no legacy RRTMG adapter on this seam"}
    module = getattr(adapter, "_ozone", None)
    if module is None:
        return {"installed": False, "reason": "the adapter does not interpolate climatological ozone"}
    if isinstance(module, OzoneLatitudeTableMemo):
        return module.receipt()
    missing = [name for name in _REQUIRED if not hasattr(module, name)]
    if missing:
        return {"installed": False, "reason": "the engine ozone module lacks " + ", ".join(missing)}
    memo = OzoneLatitudeTableMemo(module, _max_bytes())
    adapter._ozone = memo
    return memo.receipt()
