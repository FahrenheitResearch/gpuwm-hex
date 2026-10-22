"""Limit diagnostic host copies while retaining the pinned seam exporter.

Call only at a committed boundary, after the backend's binding guard. Restart
and rollback exports must continue to use the complete manifest.
"""

from collections.abc import Mapping
from copy import copy
import os


def _complete_host_export(seam):
    """Keep the complete public export, but drain its D2H copies together.

    The existing exporter still checks its phase and captures identity/scalars.
    Its shallow view queues every array on the producer stream into owned
    pinned storage. One completion fence replaces the blocking copy per field;
    returned host arrays are ready, detached, and have the original bytes.
    """
    cp = getattr(seam, "_cp", None)
    if cp is None or not hasattr(cp, "cuda"):
        return seam.export_state()
    from types import SimpleNamespace
    from .snapshot_transfer import SnapshotTransfer
    with SnapshotTransfer(cp) as transfer:
        view = copy(seam)
        view._cp = SimpleNamespace(asnumpy=lambda value: transfer.download(value, cp))
        exported = view.export_state()
        if transfer.sources:
            transfer.seal()
            transfer.wait()
        return exported


def diagnostic_export(seam, field_names, *, selective=False, resident=False):
    """Return the public payload and the original array inventory.

    The opt-in view changes only the manifest supplied to the existing public
    exporter. Its phase guard, scalar metadata and host-copy implementation
    still run. The live seam and its device arrays are never modified. Missing
    required fields remain missing so the caller's existing refusal fires.
    """
    mode = os.environ.get("GPUWM_HEX_SELECTIVE_DIAGNOSTICS", "0")
    if mode not in ("0", "1"):
        raise ValueError("GPUWM_HEX_SELECTIVE_DIAGNOSTICS must be 0 or 1")
    if mode == "0" and not selective and not resident:
        exported = _complete_host_export(seam)
        inventory = (
            tuple(exported["arrays"])
            if isinstance(exported, Mapping)
            and isinstance(exported.get("arrays"), Mapping)
            else ()
        )
        return exported, inventory
    manifest = seam._restart_manifest()
    # Engine-owned held diagnostics (including TOA OLR) are persisted under
    # diag/, while surface state remains under fields/. Both are public
    # committed-boundary exports and must reach the history capture.
    wanted = {f"{group}/{name}" for name in field_names for group in ("fields", "diag")}
    selected = {key: value for key, value in manifest.items() if key in wanted}
    view = copy(seam)
    view._restart_manifest = lambda: selected
    if resident:
        # The public exporter still enforces its boundary and captures scalars.
        # Only this private view retains device arrays for stream-ordered history
        # capture. Restarts and ordinary diagnostic callers keep host exports.
        from types import SimpleNamespace
        view._cp = SimpleNamespace(asnumpy=lambda value: value)
    return view.export_state(), tuple(manifest)
