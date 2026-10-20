"""The multi-card LTS executor carries the matched interface flux the one-card executor has.

Breakage this prevents: without it the coarse/fine interface is the reflux-driven instability
f3e284e/d5b1024 cured on one card.  On the 3 km Kalmaegi start member (2026-10-13) two cards
reached max|w| 47 m/s by 450 s and the validation flag refused the step on every block layout,
exactly as one card with interface_flux="own" did, while one card with the matched flux ran clean.
"""

import inspect
from pathlib import Path

from hexcore import cuda_fullstep_lts_active, cuda_fullstep_lts_multi

SOURCE = Path(cuda_fullstep_lts_multi.__file__).read_text(encoding="utf-8")


def _body(name: str) -> str:
    start = SOURCE.index(f"    def {name}(")
    end = SOURCE.index("\n    def ", start + 10)
    return SOURCE[start:end]


def test_the_executor_defaults_to_the_matched_flux_like_one_card():
    multi = inspect.signature(cuda_fullstep_lts_multi.FullStepLtsMulti.__init__).parameters["interface_flux"]
    one = inspect.signature(cuda_fullstep_lts_active.FullStepLtsActive.__init__).parameters["interface_flux"]
    assert multi.default == one.default == "matched"


def test_the_register_pull_carries_the_flux_and_heat_records():
    body = _body("_pull_registers")
    assert "src.flux_record" in body and "src.theta_record" in body


def test_both_sides_of_the_interface_are_wired():
    capture = _body("_attach_capture")
    assert "self._record_split_flux(" in capture       # fine side, at every split's third-stage recovery
    assert "self._impose_theta(" in capture            # coarse side, third-stage heat flux
    assert '"acoustic_ru_v841"' in _body("_attach_matched_flux")   # coarse side, acoustic momentum
    step = _body("_step_block")
    assert "imposed the child's interface flux" in step and "imposed the child's heat flux" in step


def test_the_one_card_kernels_are_the_ones_launched():
    for name in ("lts_flux_record", "lts_flux_keep", "lts_flux_impose", "lts_theta_impose"):
        assert name in cuda_fullstep_lts_active._KERNEL_NAMES
        assert f'"{name}"' in SOURCE
