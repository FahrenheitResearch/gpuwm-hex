"""MPAS-A v8.4.2 physics driver seam; scheme implementations register slots."""

from .driver_v842 import (
    Backend, DriverConfig, DriverHooks, NativeCalendarState, NativeDriverHooks, NativePoolArguments, PhysicsDriver, PhysicsManager,
    SlotCall, SlotRegistry, StepFlags, resolve_namelist, route_namelist,
)

__all__ = [
    "Backend", "DriverConfig", "DriverHooks", "NativeCalendarState", "NativeDriverHooks", "NativePoolArguments", "PhysicsDriver", "PhysicsManager",
    "SlotCall", "SlotRegistry", "StepFlags", "resolve_namelist", "route_namelist",
]
