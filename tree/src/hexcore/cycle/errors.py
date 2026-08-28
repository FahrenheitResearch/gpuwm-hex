"""Refusals the cycling cascade raises, each naming what would break."""

from __future__ import annotations

from ..errors import MpasPortError


class CycleRefusal(MpasPortError):
    """The cascade will not do this, and the message says what would break."""


class DelayedStartRefusal(CycleRefusal):
    """A mid-window initial condition cannot be composed from this parent."""


__all__ = ["CycleRefusal", "DelayedStartRefusal"]
