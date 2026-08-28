"""Where a declared source path actually lives in THIS interpreter.

Ledger #379.  Two gates in this program digest the port's own modules by
name -- the regional kernel set in
:mod:`hexcore.cuda_backend.regional_admission`, and the 23-module frozen
execution set in ``tools/run_cuda_v841_full_physics_x4.py``.  Both spell
their members the way the repository does, ``src/hexcore/<module>.py``,
and both used to resolve that spelling by walking up from a file and
assuming a checkout underneath.

**The concrete breakage that assumption caused**, measured on two machines
before this module existed (``evidence/xmachine-20260827`` section 6b,
``evidence/wheel-reach-20260827``): from an installed wheel the walk lands
on ``<venv>/Lib`` or ``<venv>/lib/python3.13``, no ``src/`` exists beneath
it, and the regional digest refuses every limited-area run with *"the
regional kernel set cannot be digested"*.  0.2.0's headline capability was
therefore unreachable from a wheel on every machine, with no flag that
could open it -- ``--repo`` was never consulted by either gate and no
caller passed a root.

**The rule this module installs, and it is one rule rather than two.**  A
declared name under ``src/hexcore/`` is resolved against the ``hexcore``
package THIS INTERPRETER IMPORTED, wherever that is.  In a source checkout
that is ``<tree>/src/hexcore`` and the resolution is byte-for-byte what
it always was, so no digest moves and no anchor lapses.  From a wheel it is
``site-packages/hexcore`` -- the copy that will actually execute.

That second half is the part worth saying plainly: resolving against the
EXECUTING package is not a workaround for the wheel case, it is the
correct question in every case.  A gate that hashes a checkout's copy of a
module while the interpreter runs a different copy is reporting on bytes
that will not be launched.  The hybrid shape a wheel user is currently
told to assemble -- ``pip install gpuwm-hex`` for the package plus
``--repo <checkout>/tree`` for the drivers ``tools/`` holds -- is exactly
that shape, and both gates were silently on the wrong side of it.

The declared NAME follows the repository, and it is never rewritten for the
sake of a digest.  Both digests frame name-then-payload, so a resolution
change with the same bytes under it produces an identical digest, while a
NAME change moves every digest keyed to it even when no arithmetic moved.

0.2.0 moved every name once, and deliberately: the package was renamed from
``mpas_port`` to ``hexcore``, so the prefix below went with it.  That is
what a name-framed digest costs -- the whole table lapses and has to be
re-derived on the card that earned it -- and the alternative, keeping a
retired directory name in the declaration so the arithmetic would not
notice, is a gate reporting on a path nothing executes.  The names say what
the repository says.
"""

from __future__ import annotations

from pathlib import Path

#: Every declared source name this program digests begins here, in the
#: repository's own spelling.  The names are framed into two SHA-256 digests
#: whose minted values are pinned constants, so moving this prefix lapses
#: every anchor keyed to them without a single byte of arithmetic having
#: changed.  It moved exactly once, at the 0.2.0 package rename, and the
#: lapse it caused was paid rather than dodged.
DECLARED_PREFIX = "src/hexcore/"

#: The ``hexcore`` package directory this interpreter imported, resolved
#: from this module's own location rather than from a tree layout.  In a
#: checkout this is ``<tree>/src/hexcore``; from a wheel it is
#: ``site-packages/hexcore``.  Either way it is the directory holding the
#: modules that will be executed.
PACKAGE_ROOT: Path = Path(__file__).resolve().parent


class DeclaredSourceError(LookupError):
    """A declared source name cannot be mapped to a file, and why."""


def resolve(name: str, root: Path | str | None = None) -> Path:
    """The file ``name`` denotes, for the package this interpreter runs.

    ``root`` is an explicit checkout root and is honoured verbatim when
    given: a caller digesting SOMEBODY ELSE'S tree -- an archived checkout,
    a second worktree in an A/B -- means that tree's layout, not this
    process's imports.  Nothing in the shipped product passes it; it exists
    so the both-directions tests can point the same function at a known-good
    and a known-bad tree and watch the verdict change.

    With no ``root``, a name under :data:`DECLARED_PREFIX` resolves against
    :data:`PACKAGE_ROOT`.  A name outside the prefix has no package-relative
    meaning at all, so it is refused by name rather than silently resolved
    against a directory it does not belong to.
    """

    if root is not None:
        return Path(root) / name
    if not name.startswith(DECLARED_PREFIX):
        raise DeclaredSourceError(
            f"{name!r} is not a declared source of the hexcore package: "
            f"every name a digest gate carries begins {DECLARED_PREFIX!r}, "
            f"and a name outside it has no meaning relative to the installed "
            f"package.  Pass an explicit root to digest a tree instead"
        )
    relative = name[len(DECLARED_PREFIX):]
    return PACKAGE_ROOT.joinpath(*relative.split("/"))


def describe_root(root: Path | str | None = None) -> str:
    """What a refusal should say it looked under."""

    if root is not None:
        return f"the tree at {Path(root)}"
    return f"the installed hexcore package at {PACKAGE_ROOT}"


__all__ = [
    "DECLARED_PREFIX",
    "PACKAGE_ROOT",
    "DeclaredSourceError",
    "describe_root",
    "resolve",
]
