"""Typed fail-closed errors for the observational referee."""

from __future__ import annotations


class RefereeError(RuntimeError):
    """Base class for deterministic, user-actionable referee failures."""


class SchemaError(RefereeError, ValueError):
    """A manifest or receipt did not satisfy its declared schema."""


class DataError(RefereeError, ValueError):
    """An input artifact was present but scientifically or structurally invalid."""


class MeasurementUnavailable(RefereeError):
    """A requested measurement cannot be made from the supplied evidence."""


class IntegrityError(RefereeError):
    """A content digest or byte-identity invariant failed."""


class ProducerError(RefereeError):
    """An external rustwx/model producer failed or violated its contract."""
