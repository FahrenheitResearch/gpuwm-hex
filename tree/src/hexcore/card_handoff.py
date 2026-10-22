"""Hand one card from a finishing member group to the next without the card going idle.

THE BREAKAGE THIS REMOVES, MEASURED (RTX 5090, ``gpuwm-hex global --row global-60km --members 1-2``,
2026-10-20): the ensemble door ran each member group as its own forecast process strictly one after
another, so every hand-off left the card at 0 % for 60-80 s (and about 165 s for a cold first member)
while ONE Python thread imported, read the member's init, interned shared tables, built the device
stack, and, at the other end, hashed the final state and drained the history writer.  That was 25-30 %
of every member's wall.

THE CONTRACT.  The ensemble door starts group N+1's process while group N still integrates.  Both name
the same per-card gate file (``GPUWM_HEX_CARD_GATE``).  A group does all of its host work first (imports,
mesh bind, init reads, host preparation), then takes the gate (an exclusive ``flock``) before it builds
anything on the device.  The group holding the card releases the gate as soon as its device work is done
and its memory is returned, before its host-only tail (history drain, final identity hash, receipts), and
the kernel drops the lock anyway if the process dies.  After taking the gate the group writes its ticket
(``GPUWM_HEX_CARD_TICKET``) so the door knows to start the group after it: at most one group waits behind
the one on the card.

Nothing here touches numerics: each member still runs alone on its card's memory, in the same process
shape, with the same admission sentence; only WHEN host work happens moves.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

GATE_ENV = "GPUWM_HEX_CARD_GATE"
TICKET_ENV = "GPUWM_HEX_CARD_TICKET"
READMIT_ENV = "GPUWM_HEX_CARD_READMIT_SECONDS"


class CardGate:
    """One process's hold on a card shared, one group at a time, by an ensemble's groups."""

    def __init__(self, path: Path, ticket: Path | None = None):
        self.path = Path(path)
        self.ticket = None if ticket is None else Path(ticket)
        self.fd: int | None = None
        self.receipt: dict[str, Any] = {"gate": str(self.path), "ticket": None if ticket is None else str(ticket)}

    @classmethod
    def from_environ(cls, environ=os.environ) -> "CardGate | None":
        path = environ.get(GATE_ENV)
        if not path:
            return None
        ticket = environ.get(TICKET_ENV)
        return cls(Path(path), None if not ticket else Path(ticket))

    def acquire(self) -> float:
        """Block until the group before this one has released the card; returns the seconds waited."""
        import fcntl

        began = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        waited = time.monotonic() - began
        self.receipt.update(acquired_unix=time.time(), waited_seconds=round(waited, 3), pid=os.getpid())
        if self.ticket is not None:
            temporary = self.ticket.with_name(self.ticket.name + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(self.receipt) + "\n", encoding="utf-8")
            os.replace(temporary, self.ticket)
        return waited

    def release(self, **attrs: Any) -> None:
        if self.fd is None:
            return
        import fcntl

        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None
        self.receipt.update(released_unix=time.time(), **attrs)


def readmit_seconds(environ=os.environ) -> float:
    """How long a group that holds the gate waits for the previous group's memory to come back."""
    return float(environ.get(READMIT_ENV, "600"))


def release_device_memory(cp) -> dict[str, int]:
    """Return every cached block this process holds, so the next group can build; bytes freed, measured."""
    import gc

    gc.collect()
    free_before, total = map(int, cp.cuda.runtime.memGetInfo())
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    free_after, _ = map(int, cp.cuda.runtime.memGetInfo())
    return {"free_before_bytes": free_before, "free_after_bytes": free_after,
            "released_bytes": free_after - free_before, "total_bytes": total}
