"""Device-resident rollback exports of the byte-pinned column seam.

The public exporter owns the array census and scalar schema. A shallow view
changes only its copy destination; the live seam and the CuPy module remain
untouched. The public restart and diagnostic exports still use host arrays.
"""

from __future__ import annotations

from copy import copy
import os
import weakref
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any

import numpy as np


class _DeviceExportCopies:
    def __init__(self, cp: Any) -> None:
        self._cp = cp

    def asnumpy(self, value: Any) -> Any:
        if isinstance(value, self._cp.ndarray):
            return self._cp.array(value, copy=True, order="C")
        # The manifest also carries host-owned radiation arrays. These
        # already live on the host and still need a detached rollback copy.
        return np.array(value, copy=True, order="C")


class _ReusingDeviceExportCopies(_DeviceExportCopies):
    """The device rollback export into buffers kept from the previous step.

    Breakage it prevents: the export allocated about 200 fresh device arrays
    (4.1 GB per 15 km rank) and ran a CuPy copy for each on every step; nsys
    on the 8-card RTX PRO 6000 host (global 15 km, 8 ranks, MPAS-A defaults, 2026-10-10) shows 51-61
    ms of host time per step in seam_export_state with the cards idle.  The
    pinned exporter asks for its manifest in the same order every step, so
    the n-th request reuses the n-th buffer when shape, dtype and device
    match, and the bytes move with one D2D memcpyAsync on the caller stream
    (ordered before every phase-one kernel, as the CuPy copy was).  The
    previous snapshot is already released at commit, so no live payload
    aliases a buffer being refilled.  Anything else takes the original path.
    """

    def __init__(self, cp: Any, buffers: list) -> None:
        super().__init__(cp)
        self._buffers = buffers
        self._ordinal = 0

    def asnumpy(self, value: Any) -> Any:
        cp = self._cp
        index = self._ordinal
        self._ordinal += 1
        if not isinstance(value, cp.ndarray) or not value.flags.c_contiguous or not value.size:
            return super().asnumpy(value)
        held = self._buffers[index] if index < len(self._buffers) else None
        if (held is None or held.shape != value.shape or held.dtype != value.dtype
                or int(held.device.id) != int(value.device.id)):
            held = cp.empty(value.shape, dtype=value.dtype)
            while len(self._buffers) <= index:
                self._buffers.append(None)
            self._buffers[index] = held
        cp.cuda.runtime.memcpyAsync(
            held.data.ptr, value.data.ptr, int(value.nbytes),
            cp.cuda.runtime.memcpyDeviceToDevice,
            cp.cuda.get_current_stream().ptr)
        return held


_REUSED_EXPORT_BUFFERS = weakref.WeakKeyDictionary()


class DeviceBoundarySnapshotV841:
    """Retain the pinned export payload on the device until a rollback.

    Use only after the adapter's source-manifest and seam-class guards.
    The pinned exporter reads ``_cp`` solely for ``asnumpy`` copies; all
    manifest keys, lazy carriers, identity and scalar metadata come from
    that exporter unchanged. Copy launches precede phase-one writes on the
    same stream, so rollback values are detached without a host synchronize.
    """

    def __init__(self, seam: Any) -> None:
        self._seam = seam
        self._cp = seam._cp

    def export_state(self) -> dict[str, Any]:
        view = copy(self._seam)
        reuse = os.environ.get("GPUWM_HEX_SNAPSHOT_REUSE", "1")
        if reuse not in ("0", "1"):
            raise ValueError("GPUWM_HEX_SNAPSHOT_REUSE must be 0 or 1")
        if reuse == "1":
            try:
                buffers = _REUSED_EXPORT_BUFFERS.setdefault(self._seam, [])
            except TypeError:
                buffers = None
            view._cp = (_DeviceExportCopies(self._cp) if buffers is None
                        else _ReusingDeviceExportCopies(self._cp, buffers))
        else:
            view._cp = _DeviceExportCopies(self._cp)
        try:
            return view.export_state()
        except MemoryError:
            # If the device snapshot cannot fit, use the host exporter.
            # Leave the exception scope before
            # exporting on the host so partial device copies are released.
            pass
        return self._seam.export_state()

    def to_host(self, payload: Any) -> dict[str, Any]:
        """Materialize the public restore payload only on an abort."""
        # The seam is replaced after a rollback; drop its reused buffers with it.
        _REUSED_EXPORT_BUFFERS.pop(self._seam, None)
        return {
            **payload,
            "arrays": {
                name: (
                    self._cp.asnumpy(value)
                    if isinstance(value, self._cp.ndarray)
                    else value
                )
                for name, value in payload["arrays"].items()
            },
        }


class _ReusableCopies:
    """Two slots own their allocations; subsequent captures only copy bytes."""

    def __init__(self, cp: Any) -> None:
        self.cp = cp
        self.buffers: dict[Any, Any] = {}
        self.ordinal = 0

    def array(self, key: Any, value: Any) -> Any:
        result = self.buffers.get(key)
        if result is None or result.shape != value.shape or result.dtype != value.dtype:
            result = (self.cp.array(value, copy=True, order="C")
                      if isinstance(value, self.cp.ndarray)
                      else np.array(value, copy=True, order="C"))
            self.buffers[key] = result
        else:
            result[...] = value
        return result

    def asnumpy(self, value: Any) -> Any:
        key = ("seam", self.ordinal)
        self.ordinal += 1
        return self.array(key, value)

    def clone(self, value: Any, key: Any) -> Any:
        if isinstance(value, (self.cp.ndarray, np.ndarray)):
            return self.array(key, value)
        if isinstance(value, Mapping):
            return {name: self.clone(item, (key, name)) for name, item in value.items()}
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{f.name: self.clone(getattr(value, f.name), (key, f.name))
                                    for f in fields(value) if f.init})
        if isinstance(value, tuple):
            return tuple(self.clone(item, (key, i)) for i, item in enumerate(value))
        if isinstance(value, list):
            return [self.clone(item, (key, i)) for i, item in enumerate(value)]
        return value


class TwoDeepForecastSnapshotV841:
    """Retain the two latest committed boundaries for one-step-late health.

    A health refusal keeps that committed endpoint, as the serial loop did.
    Restoring its slot removes the speculative successor, including physics,
    MPAS diagnostics, surface carry, GF carry and adapter bookkeeping.
    Static mesh and coefficients are read-only and are never duplicated.
    """

    _ADAPTER_FIELDS = ("_gwdo_calls", "_gf_forcing_seen", "_last_gwdo_result",
                       "_committed_refl10cm", "_last_receipt")

    def __init__(self, stack: Mapping[str, Any], cp: Any, *, observers: Mapping[str, Any] | None = None) -> None:
        self.stack, self.cp = stack, cp
        self.observers = {name: value for name, value in (observers or {}).items() if value is not None}
        self.slots = [{"copies": _ReusableCopies(cp)} for _ in range(2)]
        self.captures = self.rollbacks = 0

    @staticmethod
    def _metadata(value: Any) -> Any:
        # Opaque CUDA graph/allocator handles retain their identity. Their
        # dictionaries of host counters must detach from the next step.
        if isinstance(value, Mapping):
            return {k: TwoDeepForecastSnapshotV841._metadata(v) for k, v in value.items()}
        if isinstance(value, list):
            return [TwoDeepForecastSnapshotV841._metadata(v) for v in value]
        if isinstance(value, tuple):
            return tuple(TwoDeepForecastSnapshotV841._metadata(v) for v in value)
        if isinstance(value, set):
            return set(value)
        return value

    def capture(self, step: int, surface: Any, dynamics: Any) -> None:
        backend, driver = self.stack["backend"], self.stack["driver"]
        if backend._phase != "boundary":
            raise RuntimeError("late health snapshot requires a committed physics boundary")
        slot = self.slots[step % 2]
        copies = slot["copies"]
        copies.ordinal = 0
        try:
            view = copy(backend._seam)
            view._cp = copies
            seam = view.export_state()
            atmosphere = replace(driver.atmosphere,
                state=copies.clone(driver.atmosphere.state, "state"),
                saved=copies.clone(driver.atmosphere.saved, "saved"))
            adapter = {name: copies.clone(getattr(backend, name), name)
                       for name in self._ADAPTER_FIELDS}
            carried = (copies.clone(surface, "surface"), copies.clone(dynamics, "gf"))
            counters = {name: getattr(driver, name) for name in ("mixing_calls_v841",)
                        if hasattr(driver, name)}
            regional = getattr(driver, "regional_v841", None)
            region = None
            if regional is not None:
                driving = regional.driving
                region = {
                    "clock": regional.clock, "started": regional._started,
                    "offsets": copies.clone(regional._offsets, "regional_offsets"),
                    "acoustic_invalid": copies.clone(regional.acoustic_invalid, "regional_flag"),
                    "driving": {name: copies.clone(getattr(driving, name), ("lbc", name))
                                for name in ("_state", "_tend", "_state_scalars", "_tend_scalars")},
                    # LbcPool replaces its admitted files and tendency arrays;
                    # it never mutates the retained host arrays in place.
                    "pool": dict(driving._pool.__dict__),
                    "discipline": {name: getattr(regional.discipline, name)
                                   for name in ("scrubs", "fused_launches", "dirty_values", "last_kernel")},
                    "unit_pool": dict(regional.discipline._unit_pool),
                }
            observers = {name: self._metadata(value.__dict__) for name, value in self.observers.items()}
        except MemoryError as error:
            raise RuntimeError(
                "two-deep late-health rollback does not fit on this card; "
                "one-deep would overwrite the refused endpoint with its speculative successor"
            ) from error
        slot.update(step=step, seam=seam, atmosphere=atmosphere,
                    adapter=adapter, carried=carried, counters=counters,
                    regional=region, observers=observers)
        self.captures += 1

    def restore(self, step: int) -> tuple[Any, Any]:
        slot = self.slots[step % 2]
        if slot.get("step") != step:
            raise RuntimeError(f"late health rollback lost committed boundary {step}")
        backend, driver = self.stack["backend"], self.stack["driver"]
        backend._restore_boundary(slot["seam"])
        for name, value in slot["adapter"].items():
            setattr(backend, name, value)
        driver.atmosphere = slot["atmosphere"]
        driver._pending_v841_physics_candidate = None
        driver._step_gate_envelope = None
        for name, value in slot["counters"].items():
            setattr(driver, name, value)
        region = slot["regional"]
        if region is not None:
            regional = driver.regional_v841
            regional.clock, regional._started = region["clock"], region["started"]
            regional._offsets, regional.acoustic_invalid = region["offsets"], region["acoustic_invalid"]
            for name, value in region["driving"].items():
                setattr(regional.driving, name, value)
            regional.driving._pool.__dict__.clear()
            regional.driving._pool.__dict__.update(region["pool"])
            for name, value in region["discipline"].items():
                setattr(regional.discipline, name, value)
            regional.discipline._unit_pool = dict(region["unit_pool"])
            # Register the restored rho by address without changing a byte.
            # The old registration points at the pre-copy allocation.
            rho = driver.atmosphere.state.rho
            pointer = getattr(rho.data, "ptr", None)
            if pointer is not None:
                regional.discipline._unit_pool[int(pointer)] = rho
        for name, value in slot["observers"].items():
            observer = self.observers[name]
            observer.__dict__.clear()
            observer.__dict__.update(value)
            if name == "physics_park":
                observer._seam = backend._seam
        self.rollbacks += 1
        return slot["carried"]

    def receipt(self) -> dict[str, Any]:
        device = host = 0
        for slot in self.slots:
            for value in slot["copies"].buffers.values():
                if isinstance(value, self.cp.ndarray):
                    device += int(value.nbytes)
                else:
                    host += int(value.nbytes)
        return {"depth": 2, "device_bytes": device, "host_bytes": host,
                "memory_scope": "requested bytes in copied snapshot arrays; shared opaque handles and retained host LBC files require the real-run peak measurement",
                "captures": self.captures, "rollbacks": self.rollbacks,
                "slot_device_bytes": [sum(int(v.nbytes) for v in s["copies"].buffers.values()
                                          if isinstance(v, self.cp.ndarray)) for s in self.slots]}
