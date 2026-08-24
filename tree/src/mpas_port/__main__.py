"""``python -m mpas_port`` -- the same front doors as the ``gpuwm-hex`` script.

The console script is the supported spelling.  This module exists so that a
user who has the package importable but not on PATH (a checkout with
``PYTHONPATH=src``, a ``pip install --target``, a frozen CI step) reaches the
identical parser instead of being told to go and find an entry point.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
