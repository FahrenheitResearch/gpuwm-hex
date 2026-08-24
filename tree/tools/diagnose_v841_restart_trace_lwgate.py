#!/usr/bin/env python3
"""Matched-schedule restart trace against the fixed work checkout,
executed from a separate pristine repo copy named by MPAS_PROOF_REPO.
Identical protocol to tools/diagnose_v841_restart_step16_x4.py;
only the frozen-checkout git/manifest pins are skipped, because the
work copy carries the LW purity fix."""
import os
import sys
from pathlib import Path

# No baked-in repo path: this wrapper runs a DIFFERENT checkout than the one
# it is stored in, and an unset root would put a nonexistent tools/ on
# sys.path so the import below would fail as a missing module rather than as
# a missing setting.
_PROOF_REPO = os.environ.get("MPAS_PROOF_REPO")
if not _PROOF_REPO:
    raise SystemExit(
        "no proof repo: set MPAS_PROOF_REPO to the pristine repo checkout "
        "whose tools/ and src/ this wrapper imports"
    )
ROOT = Path(_PROOF_REPO).expanduser().resolve()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_cuda_v841_full_physics_x4 as runner  # noqa: E402
import mpas_port.cuda_arwen_physics_v841 as pin_mod  # noqa: E402

runner.verify_arwen_checkout_git = lambda checkout: {
    "root": str(checkout), "pins": "SKIPPED (modified work checkout)"}
pin_mod._verify_checkout_root = lambda root: None
print("[wrapper] PINS SKIPPED (modified work checkout)", flush=True)

import diagnose_v841_restart_step16_x4 as diag  # noqa: E402

rc = diag.main()
import gpuwm  # noqa: E402
import mpas_port.cuda_driver as cd  # noqa: E402

print(f"[wrapper] gpuwm module: {gpuwm.__file__}", flush=True)
print(f"[wrapper] cuda_driver module: {cd.__file__}", flush=True)
sys.exit(rc)
