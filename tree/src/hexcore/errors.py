"""Failure types that keep unsupported behavior explicit."""

from __future__ import annotations


class MpasPortError(RuntimeError):
    """Base class for failures attributable to the port contract."""


class MeshValidationError(MpasPortError):
    """A mesh violates a named MPAS topology or metric invariant."""


class EvidenceError(MpasPortError):
    """Evidence is absent, malformed, or weaker than a requested claim."""


class ConfigurationRefusal(MpasPortError):
    """A requested configuration knob has no honest implementation yet."""

    def __init__(self, knob: str, value: object, reason: str, declaration: str) -> None:
        self.knob = knob
        self.value = value
        self.reason = reason
        self.declaration = declaration
        super().__init__(
            f"{knob}={value!r} is refused: {reason}. "
            f"To run, declare {declaration}."
        )

