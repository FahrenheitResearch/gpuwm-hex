"""Coarse-then-corridor cycling: following weather from one cycle to the next.

The placement layer decides WHERE a fine grid goes and the forecast door runs
one.  This package is the loop between them -- the thing that was missing, and
the reason hysteresis, slot continuity and the move-or-stay decision existed
with nobody to hand them to.
"""

from __future__ import annotations

from .chain import CascadeConfig, run_cascade, run_slot
from .delayed_start import compose_mid_window_init
from .errors import CycleRefusal, DelayedStartRefusal

__all__ = [
    "CascadeConfig",
    "CycleRefusal",
    "DelayedStartRefusal",
    "compose_mid_window_init",
    "run_cascade",
    "run_slot",
]
