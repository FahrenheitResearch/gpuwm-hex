# SPDX-License-Identifier: Apache-2.0
"""Park the frozen-Arwen physics tier on pinned host memory between steps.

WHAT THIS IS FOR
----------------
The measured device ledger says the physics column state and its scratch hold
roughly 23 kB per cell resident for the whole run, and that every byte of it is
still live at the run's global device peak -- a peak that is reached inside the
dynamics half of the step, where none of it is read.  A card therefore has to
be big enough to hold the physics tier and the dynamics peak *at the same time*
even though the two never overlap in use.

This module moves that residency out of the way.  At a committed step boundary
it copies the seam's device allocations into pinned host memory, drops the
device references so the blocks go back to the allocator, and copies them back
before the next physics call.  The bytes that come back are the bytes that went
out -- the whole allocation is copied raw, padding included.

WHAT IT DOES NOT CLAIM
----------------------
It does not claim bit-identity of the run.  The restored allocations live at
different device addresses, so every allocation the process makes afterwards
lands somewhere else.  The frozen tree's phase-1 longwave path is known to
depend on device pool contents (the lwfix-20260812 pin), which is exactly the
kind of dependence a changed allocation order can move.  Bit-identity here is a
measurement, never an argument: run the two arms and compare the payload
digest.  This module is therefore constructed explicitly and is never on by
default.

THE SHARING RULE, AND THE BREAKAGE IT PREVENTS
----------------------------------------------
An allocation that is reachable from BOTH the physics seam and the dycore
driver is never parked.  Parking rebinds the seam's reference to a new device
allocation; the dycore would keep writing to the old one, and the two would
silently diverge from the first step after the park.  :meth:`park` refuses by
name if such an allocation is found and no exclusion set was supplied.
"""

from __future__ import annotations

from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any, Callable

import numpy as np


__all__ = [
    "CudaPhysicsTierParkV841",
    "PhysicsTierParkRefusal",
    "walk_device_allocations",
]


_ATOMIC = (str, bytes, bytearray, int, float, complex, bool, type(None))
_UNWALKABLE = (type, ModuleType, FunctionType, MethodType, BuiltinFunctionType)


class PhysicsTierParkRefusal(RuntimeError):
    """Raised when parking would change something other than where bytes live."""


def _root_base(array: Any) -> Any:
    """The allocation an array borrows from, following nested views."""

    base = array
    while getattr(base, "base", None) is not None:
        base = base.base
    return base


def walk_device_allocations(
    cp: Any, root: Any, *, max_depth: int = 16
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Find every CuPy array reachable from ``root``.

    Returns ``(allocations, holders)``.  ``allocations`` maps the device
    pointer of each owning allocation to the owning array.  ``holders`` is one
    record per *reference*, so an array reached through two attributes is
    rebound through both -- missing one would leave a live reference to a freed
    allocation.

    THE WALK IS ITERATIVE ON PURPOSE.  A recursive walk written with closures
    builds a reference cycle (the inner function's cell refers to itself), and
    that cycle keeps every array the walk touched alive until the cyclic
    collector happens to run.  That is fatal here: the whole point is that
    dropping the caller's references frees device memory NOW, and a first
    version of this module released exactly 0 bytes for that reason.
    """

    visited: set[int] = set()
    allocations: dict[int, Any] = {}
    holders: list[dict[str, Any]] = []
    # One token per distinct ndarray OBJECT.  The frozen tree asserts object
    # identity between some of its own fields (io/restart.py refuses when
    # ``sfclay_result.znt is not fields['znt']``), so two references to one
    # array must come back as one array, not two equal ones.  The array is
    # held in its own holder record, never in a side list, so clearing the
    # records is enough to let the allocation go.
    tokens: dict[int, int] = {}

    work: list[tuple[Any, int, Callable[[Any], None] | None]] = [(root, 0, None)]
    while work:
        obj, depth, rebind = work.pop()
        if isinstance(obj, _ATOMIC) or isinstance(obj, _UNWALKABLE):
            continue
        if isinstance(obj, cp.ndarray):
            # Arrays are never de-duplicated: every reference must be rebound,
            # and a reference that CANNOT be rebound is recorded too so its
            # allocation is refused rather than freed under a live holder.
            base = _root_base(obj)
            pointer = int(base.data.ptr)
            allocations.setdefault(pointer, base)
            token = tokens.get(id(obj))
            if token is None:
                token = len(tokens)
                tokens[id(obj)] = token
            holders.append(
                {
                    "rebind": rebind,
                    "token": token,
                    "array": obj,
                    "allocation": pointer,
                    "offset": int(obj.data.ptr) - pointer,
                    "shape": tuple(obj.shape),
                    "strides": tuple(obj.strides),
                    "dtype": obj.dtype,
                    "is_view": obj is not base,
                }
            )
            continue
        if depth > max_depth:
            continue
        identity = id(obj)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                work.append((obj[key], depth + 1, _dict_rebinder(obj, key)))
            continue
        if isinstance(obj, list):
            for index in range(len(obj)):
                work.append((obj[index], depth + 1, _list_rebinder(obj, index)))
            continue
        if isinstance(obj, np.ndarray):
            continue
        if isinstance(obj, (tuple, set, frozenset)):
            # Immutable containers cannot be rebound.  Walk them so a shared
            # allocation is still SEEN (the sharing rule needs to know), but
            # hand down no rebinder: an array found only here cannot be parked.
            for item in list(obj):
                work.append((item, depth + 1, None))
            continue
        slots: list[str] = []
        for klass in type(obj).__mro__:
            slots.extend(getattr(klass, "__slots__", ()) or ())
        state = getattr(obj, "__dict__", None)
        if state is not None:
            for key in list(state.keys()):
                work.append((state[key], depth + 1, _attr_rebinder(obj, key)))
        for key in slots:
            if hasattr(obj, key):
                work.append((getattr(obj, key), depth + 1, _attr_rebinder(obj, key)))

    return allocations, holders


def _dict_rebinder(mapping: dict, key: Any) -> Callable[[Any], None]:
    def rebind(value: Any) -> None:
        mapping[key] = value

    return rebind


def _list_rebinder(sequence: list, index: int) -> Callable[[Any], None]:
    def rebind(value: Any) -> None:
        sequence[index] = value

    return rebind


def _attr_rebinder(obj: Any, key: str) -> Callable[[Any], None] | None:
    """A setter for ``obj.key``, or None when the attribute cannot be set.

    Frozen dataclasses and read-only properties are common in this graph.  The
    test is the operation itself -- assigning the value already there -- so a
    custom ``__setattr__`` is judged by what it does, not by what it looks like.
    """

    def rebind(value: Any) -> None:
        setattr(obj, key, value)

    try:
        rebind(getattr(obj, key))
    except Exception:
        return None
    return rebind


class CudaPhysicsTierParkV841:
    """Move a physics seam's device allocations to pinned host memory and back.

    ``seam_root`` is the object graph to park -- the frozen Arwen seam.
    ``shared_roots`` are graphs whose allocations must never be rebound; the
    dycore driver belongs here.
    """

    def __init__(
        self,
        cp: Any,
        seam_root: Any,
        *,
        shared_roots: tuple[Any, ...] = (),
        max_depth: int = 16,
        diagnose: bool = False,
    ) -> None:
        self._cp = cp
        self._seam_root = seam_root
        self._shared_roots = tuple(shared_roots)
        self._max_depth = int(max_depth)
        self._diagnose = bool(diagnose)
        self._parked: dict[int, Any] = {}
        self._holders: list[dict[str, Any]] = []
        self._is_parked = False
        self._receipt: dict[str, Any] = {
            "parks": 0,
            "unparks": 0,
            "parked_bytes": 0,
            "allocation_count": 0,
            "holder_count": 0,
            "shared_skipped_bytes": 0,
            "shared_skipped_count": 0,
            "unrebindable_bytes": 0,
            "d2h_seconds": 0.0,
            "h2d_seconds": 0.0,
        }

    # -- receipt --------------------------------------------------------
    @property
    def is_parked(self) -> bool:
        return self._is_parked

    def receipt(self) -> dict[str, Any]:
        return dict(self._receipt)

    # -- diagnosis --------------------------------------------------------
    def _name_holdouts(self, parkable: dict[int, Any]) -> list[dict[str, Any]]:
        """Who still owns the biggest allocations after the references dropped.

        A park that releases nothing is indistinguishable from a park that
        works unless the remaining owners can be named.  This walks the
        interpreter's referrer graph for the three largest allocations and
        reports what is holding them.
        """

        import gc

        result: list[dict[str, Any]] = []
        largest = sorted(parkable.values(), key=lambda a: -int(a.nbytes))[:3]
        for array in largest:
            referrers = []
            for referrer in gc.get_referrers(array):
                if referrer is parkable or referrer is largest:
                    continue
                label = type(referrer).__name__
                if isinstance(referrer, dict):
                    keys = [k for k, v in referrer.items() if v is array]
                    label = f"dict keys={keys[:3]}"
                elif isinstance(referrer, list):
                    label = f"list len={len(referrer)}"
                referrers.append(label)
            result.append(
                {
                    "nbytes": int(array.nbytes),
                    "shape": list(array.shape),
                    "referrers": referrers[:8],
                    "referrer_count": len(referrers),
                }
            )
        return result

    # -- the two operations ---------------------------------------------
    def park(self, *, shared_roots: tuple[Any, ...] = ()) -> int:
        """Copy the tier to pinned host memory and free the device blocks.

        ``shared_roots`` names the object graphs whose allocations are still in
        flight at this call site, on top of the ones fixed at construction.
        """

        if self._is_parked:
            raise PhysicsTierParkRefusal(
                "the physics tier is already parked; a second park would copy "
                "host placeholders back over the parked bytes"
            )
        cp = self._cp
        import time

        shared: set[int] = set()
        for root in self._shared_roots + tuple(shared_roots):
            allocations, _ = walk_device_allocations(
                cp, root, max_depth=self._max_depth
            )
            shared.update(allocations)

        allocations, holders = walk_device_allocations(
            cp, self._seam_root, max_depth=self._max_depth
        )

        pinned_down: set[int] = set()
        reachable: set[int] = set()
        for holder in holders:
            reachable.add(holder["allocation"])
            if holder["rebind"] is None:
                pinned_down.add(holder["allocation"])

        parkable: dict[int, Any] = {}
        shared_bytes = 0
        shared_count = 0
        unrebindable_bytes = 0
        for pointer, base in allocations.items():
            if pointer in shared:
                shared_bytes += int(base.nbytes)
                shared_count += 1
                continue
            if pointer not in reachable or pointer in pinned_down:
                # At least one holder is a frozen dataclass field, a read-only
                # property or an immutable container slot.  Freeing the
                # allocation would leave that holder pointing at released
                # device memory, so the whole allocation stays where it is.
                unrebindable_bytes += int(base.nbytes)
                continue
            parkable[pointer] = base

        started = time.perf_counter()
        stored: dict[int, Any] = {}
        total = 0
        for pointer, base in list(parkable.items()):
            nbytes = int(base.nbytes)
            flat = cp.ndarray((nbytes,), dtype=cp.uint8, memptr=base.data)
            pinned = cp.cuda.alloc_pinned_memory(nbytes)
            host = np.frombuffer(pinned, dtype=np.uint8, count=nbytes)
            flat.get(out=host)
            stored[pointer] = {"pinned": pinned, "host": host, "nbytes": nbytes}
            total += nbytes
        cp.cuda.runtime.deviceSynchronize()
        d2h_seconds = time.perf_counter() - started

        pool = cp.get_default_memory_pool()
        used_before = int(pool.used_bytes())
        parkable_bytes = sum(int(a.nbytes) for a in parkable.values())
        kept = [h for h in holders if h["allocation"] in parkable]
        for holder in kept:
            holder["rebind"](None)
        # Everything this method still holds an array through has to go before
        # the pool is asked what it released, or the answer is always zero and
        # the instrument lies about the whole exercise.  The walk's records
        # carry the array itself, so clearing them is part of the release.
        for holder in holders:
            holder["array"] = None
        holdouts = self._name_holdouts(parkable) if self._diagnose else []
        del allocations, holders, parkable
        used_after = int(pool.used_bytes())

        self._parked = stored
        self._holders = kept
        self._is_parked = True
        self._receipt.update(
            {
                "parks": self._receipt["parks"] + 1,
                "parked_bytes": total,
                "allocation_count": len(stored),
                "holder_count": len(kept),
                "shared_skipped_bytes": shared_bytes,
                "shared_skipped_count": shared_count,
                "unrebindable_bytes": unrebindable_bytes,
                "d2h_seconds": self._receipt["d2h_seconds"] + d2h_seconds,
                "last_d2h_seconds": d2h_seconds,
                # The honest test of whether the park FREED anything: the
                # pool's outstanding bytes across the reference drop.  A
                # released tier shows a fall of parked_bytes here.  Anything
                # less means some holder outside this graph still owns them.
                "last_pool_used_before_release": used_before,
                "last_pool_used_after_release": used_after,
                "last_pool_released_bytes": used_before - used_after,
                "last_pool_total_bytes": int(pool.total_bytes()),
                "last_release_shortfall_bytes": (
                    parkable_bytes - (used_before - used_after)
                ),
                "last_holdouts": holdouts,
            }
        )
        return total

    def unpark(self) -> int:
        """Copy the tier back to the device and rebind every reference."""

        if not self._is_parked:
            raise PhysicsTierParkRefusal(
                "the physics tier is not parked; unparking would rebind "
                "references to allocations that were never freed"
            )
        cp = self._cp
        import time

        started = time.perf_counter()
        restored: dict[int, Any] = {}
        total = 0
        for pointer, record in self._parked.items():
            nbytes = record["nbytes"]
            device = cp.empty((nbytes,), dtype=cp.uint8)
            device.set(record["host"])
            restored[pointer] = device
            total += nbytes
        cp.cuda.runtime.deviceSynchronize()
        h2d_seconds = time.perf_counter() - started

        # One array object per token: two holders that shared an array object
        # before the park share one again.  The frozen tree asserts exactly
        # this (``sfclay_result.znt is fields['znt']``) and refuses otherwise.
        rebuilt: dict[int, Any] = {}
        for holder in self._holders:
            array = rebuilt.get(holder["token"])
            if array is None:
                device = restored[holder["allocation"]]
                array = cp.ndarray(
                    holder["shape"],
                    dtype=holder["dtype"],
                    memptr=device.data + holder["offset"],
                    strides=holder["strides"],
                )
                rebuilt[holder["token"]] = array
            holder["rebind"](array)

        # The flat uint8 carriers stay alive through the rebound views'
        # memptr, so dropping them here does not free anything.
        self._parked = {}
        self._holders = []
        self._is_parked = False
        self._receipt.update(
            {
                "unparks": self._receipt["unparks"] + 1,
                "h2d_seconds": self._receipt["h2d_seconds"] + h2d_seconds,
                "last_h2d_seconds": h2d_seconds,
                "last_unparked_bytes": total,
                "last_pool_used_after_restore": int(
                    cp.get_default_memory_pool().used_bytes()
                ),
                "last_pool_total_after_restore": int(
                    cp.get_default_memory_pool().total_bytes()
                ),
            }
        )
        return total
