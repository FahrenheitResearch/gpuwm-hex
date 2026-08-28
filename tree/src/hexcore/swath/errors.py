"""Refusals this layer makes, each naming the breakage it prevents.

Every class here is an :class:`hexcore.errors.MpasPortError`, so the
console script's one ``except`` in :func:`hexcore.cli.main` turns any of
them into exit status 2 with the message on stderr, exactly as every other
door in this distribution behaves.

There are three, and the split is not decoration -- it is what lets a
caller tell "your document is wrong" from "the world did not offer a
placement" from "the placement does not fit the machine", which are three
different things for the operator to do about.
"""

from __future__ import annotations

from ..errors import MpasPortError


class SwathError(MpasPortError):
    """Anything this layer refuses."""


class SwathDocumentError(SwathError):
    """A metrics or policy document this layer will not load.

    Unknown keys, unknown vocabulary members, and rows whose numbers
    cannot describe a placement.  Raised at LOAD, before any history file
    is opened, so a typo in a table costs a message rather than a cycle.
    """


class SwathRefusal(SwathError):
    """A placement this layer will not emit.

    The geometry is degenerate, the ranking is not a total order, or the
    plan would ask the mesh lane for something it cannot build.  Raised
    with the numbers that made it true.
    """


class SwathCapacityRefusal(SwathError):
    """The placement is well formed and does not fit the declared machine."""


__all__ = [
    "SwathError",
    "SwathDocumentError",
    "SwathRefusal",
    "SwathCapacityRefusal",
]
