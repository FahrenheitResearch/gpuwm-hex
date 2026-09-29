#!/usr/bin/env python3
"""Moved into the package: ``hexcore.drivers.run_cuda_v841_full_physics_x4``.

The forecast drivers ship inside the wheel since 0.3.2, so ``gpuwm-hex
forecast`` runs from an install.  This path stays so a script that runs this
file, or imports ``run_cuda_v841_full_physics_x4`` with ``tools/`` on ``sys.path``, keeps
working: both reach the one packaged module, never a second copy of it.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hexcore.drivers import run_cuda_v841_full_physics_x4 as _module  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_module.main())
# An import by name gets the packaged module itself; a loader that executes
# this file by path gets its public names, which is enough to call it.
globals().update(
    {key: value for key, value in vars(_module).items() if not key.startswith("__")}
)
sys.modules[__name__] = _module
