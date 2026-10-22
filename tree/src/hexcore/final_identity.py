"""Canonical C4 digest of the complete committed hex forecast carriers."""
from collections.abc import Mapping
import numpy as np
from .cuda_dualrun import STATE_FIELDS, SIDECAR_FIELDS


def carried_digest(atmosphere, backend_state, dynamics_tendencies, surface_updates, *, gf_required=True):
    from gpuwm.state_digest import _canonical_digest_document

    arrays, scalars = {}, {}

    def walk(name, value):
        if isinstance(value, Mapping):
            for key in sorted(value):
                walk(f"{name}/{key}", value[key])
        elif hasattr(value, "dtype") and hasattr(value, "shape"):
            arrays[name] = value
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(f"{name}/{index}", item)
        elif isinstance(value, np.generic):
            scalars[name] = value.item()
        elif value is None or isinstance(value, (str, int, float, bool)):
            scalars[name] = value
        else:
            raise TypeError(f"unregistered carried identity value {name}: {type(value).__name__}")

    walk("mpas/state", {key: getattr(atmosphere.state, key) for key in STATE_FIELDS})
    walk("mpas/saved", {key: getattr(atmosphere.saved, key) for key in SIDECAR_FIELDS})
    walk("mpas/time_seconds", float(atmosphere.state.time_seconds))
    walk("backend", backend_state)
    walk("surface_updates", surface_updates)
    if dynamics_tendencies is None:
        if gf_required:
            raise ValueError("final identity requires the committed GF forcing carrier")
        # The initial boundary and convection-off runs have no GF carrier.
        # Bind that absence explicitly rather than inventing zero tendencies.
        walk("gf", None)
    else:
        walk("gf", {key: getattr(dynamics_tendencies, key)
                    for key in ("rthdynten", "rqvdynten", "time_seconds")})
    return _canonical_digest_document(arrays, scalars, "hex-carried")


"""Bind retained hex carriers with the installed engine's canonical serializer."""

from collections.abc import Mapping
import dataclasses


def final_carriers(stack, dynamics_tendencies, surface_accumulators, surface_updates, *, to_host=False):
    """The committed carriers ``final_digest`` binds, as ``(manifest, scalars)``.

    ``to_host`` copies every device carrier to the host first (``.get()``, the same native-endian bytes the
    serializer's own ``.get()`` reads), so the hash can run after the card has been released
    (hexcore.card_handoff).  The digest of the copies is the digest of the device carriers.
    """
    from hexcore.cuda_dualrun import STATE_FIELDS, SIDECAR_FIELDS

    manifest, scalars = {}, {}

    def collect(path, value):
        if hasattr(value, "dtype") and hasattr(value, "shape"):
            manifest[path] = value.get() if to_host and hasattr(value, "get") and not isinstance(value, np.ndarray)                 else value
        elif isinstance(value, Mapping):
            for key, child in sorted(value.items()):
                collect(f"{path}/{key}", child)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                collect(f"{path}/{field.name}", getattr(value, field.name))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                collect(f"{path}/{index}", child)
        elif value is None or isinstance(value, (str, bool, int, float)):
            scalars[path] = value
        else:
            raise TypeError(f"unbound final-state carrier at {path}: {type(value).__name__}")

    atmosphere = stack["driver"].atmosphere
    for name in STATE_FIELDS:
        collect(f"atmosphere/state/{name}", getattr(atmosphere.state, name))
    for name in SIDECAR_FIELDS:
        collect(f"atmosphere/saved/{name}", getattr(atmosphere.saved, name))
    collect("atmosphere/time_seconds", float(atmosphere.state.time_seconds))
    collect("physics", stack["backend"].restart_state())
    collect("surface_accumulators", surface_accumulators)
    collect("surface_updates", surface_updates)
    if dynamics_tendencies is not None:
        for name in ("rthdynten", "rqvdynten", "time_seconds"):
            collect(f"dynamics_forcing/{name}", getattr(dynamics_tendencies, name))
    return manifest, scalars


def final_digest_of(carriers):
    # The production serializer owns dtype, byte-order and descriptor framing.
    # This adapter only selects carriers; it never re-encodes field values.
    from gpuwm.state_digest import _canonical_digest_document

    manifest, scalars = carriers
    return _canonical_digest_document(manifest, scalars, "hex-committed-boundary")


def final_digest(stack, dynamics_tendencies, surface_accumulators, surface_updates):
    return final_digest_of(final_carriers(stack, dynamics_tendencies, surface_accumulators, surface_updates))
