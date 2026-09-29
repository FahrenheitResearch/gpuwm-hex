"""The machine a receipt was written on, named without naming it.

THE BREAKAGE THIS PREVENTS: every door receipt recorded ``platform.node()``,
the machine's network name, so a receipt a user shares -- attached to a bug
report, published beside a figure -- carried the name of their machine.  A
receipt still has to say whether two runs came from the same machine, which
is what the field was for, so it records a salted digest of the name instead.

The salt is random and per user: it lives in ``~/.gpuwm/hex-host-salt``,
written once, so one user's receipts agree with each other across runs and
cannot be matched to a machine name by hashing a dictionary of likely names.
Where that file cannot be read or written, the salt is random for this
process and the record says so (``"scope": "run"``), which still names no
machine.
"""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import platform
import secrets
from typing import Any

__all__ = ["SALT_PATH", "host_record"]

#: Where the per-user salt lives.
SALT_PATH = Path("~/.gpuwm/hex-host-salt")

_PROCESS_SALT: str | None = None


def _salt() -> tuple[str, str]:
    """(salt, scope): the per-user salt, or a per-process one."""

    global _PROCESS_SALT
    path = SALT_PATH.expanduser()
    try:
        text = path.read_text(encoding="ascii").strip()
        if len(text) >= 32:
            return text, "user"
    except OSError:
        pass
    salt = secrets.token_hex(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="ascii") as handle:
            handle.write(salt + "\n")
        os.chmod(path, 0o600)
        return salt, "user"
    except FileExistsError:
        try:
            text = path.read_text(encoding="ascii").strip()
            if len(text) >= 32:
                return text, "user"
        except OSError:
            pass
    except OSError:
        pass
    if _PROCESS_SALT is None:
        _PROCESS_SALT = salt
    return _PROCESS_SALT, "run"


def host_record() -> dict[str, Any]:
    """The receipt's ``host`` value: a salted digest, never the name."""

    salt, scope = _salt()
    digest = sha256((salt + "\0" + platform.node()).encode("utf-8")).hexdigest()
    return {"sha256": digest[:16], "scope": scope}
