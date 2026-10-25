"""The full-step LTS shadow's pin of cuda_driver.py matches the shipped driver.

Breakage it prevents: a change to cuda_driver.py (any branch, any merge) leaves the shadow's audited digest
behind, and every --fullstep-local-timestep run on a variable mesh then refuses at start with "native driver
source ... changed" on both physics routes. It happened twice (0.4.1 to 0.4.4, and again on the 0.4.5 fold). A
red here means: re-audit the two advance_scalar_transport_cuda_v841 calls (source-old rho at position 4, split
fluxes at 6 and 7, stage duration at 8), then re-pin.
"""

import hashlib
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "hexcore"


def test_the_shadow_pin_is_the_shipped_driver():
    text = (SRC / "cuda_fullstep_lts_shadow.py").read_text(encoding="utf-8")
    pinned = text.split("NATIVE_DRIVER_SHA256 = '", 1)[1].split("'", 1)[0]
    assert hashlib.sha256((SRC / "cuda_driver.py").read_bytes()).hexdigest() == pinned
