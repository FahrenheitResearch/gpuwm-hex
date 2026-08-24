#!/usr/bin/env python3
"""Checkout-local wrapper; no pyproject/primary CLI mutation is required."""

from __future__ import annotations

from pathlib import Path
import sys

TREE = Path(__file__).resolve().parents[1]
SRC = TREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mpas_port.obs_referee.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
