"""Active full-step LTS imposes the matched interface flux on a limited-area driver too.

Breakage this prevents: a limited-area driver launches ``acoustic_ru_regional_v841`` through its
own ``CudaRegionalKernels.launch``, never ``acoustic_ru_v841`` through the global ``_kernel``
table, so the coarse-side tap never fired and the first class step on a culled region mesh with
two rate classes stopped with "class 0 imposed the child's interface flux 0 times in one step"
(hex 0.4.4, 2026-10-24).
"""

from types import SimpleNamespace

import pytest

from hexcore import cuda_fullstep_lts_active, cuda_fullstep_lts_multi
from hexcore.cuda_fullstep_lts_scalars import tap_acoustic_momentum, tap_regional_launch


class _Kernels:
    def __init__(self):
        self.calls = []

    def launch(self, name, count, args):
        self.calls.append(name)


def _regional_args():
    # acoustic_ru_regional_v841: the global kernel's list with spec_zone_mask_edge before zz
    # and the invalid flag last
    return (55, 101, 60, 2, 1.5, 9.8, 287.0, 1004.0, "coe", "inv_dc", "spec", "zz", "exner",
            "cqu", "zxu", "tend_ru", "rho_pp", "rtheta_pp", "ru_p", "ru_avg", "invalid")


def test_a_regional_driver_is_tapped_on_the_kernel_it_launches():
    kernels = _Kernels()
    driver = SimpleNamespace(regional_v841=SimpleNamespace(kernels=kernels), cache=None)
    seen = []
    tap_acoustic_momentum(driver, before=lambda a: seen.append(("before", a[1], a[3], a[-2], a[-1])),
                          after=lambda a: seen.append(("after", a[-2], a[-1])), dimensions=())
    kernels.launch("acoustic_rs_ts_regional_v841", 10, ())
    kernels.launch("acoustic_ru_regional_v841", 55 * 101, _regional_args())
    assert kernels.calls == ["acoustic_rs_ts_regional_v841", "acoustic_ru_regional_v841"]
    # the hooks read nedges, small_step, ru_p and ru_avg at the global kernel's positions
    assert seen == [("before", 101, 2, "ru_p", "ru_avg"), ("after", "ru_p", "ru_avg")]


def test_a_regional_kernel_is_tapped_once():
    kernels = _Kernels()
    tap_regional_launch(kernels, "acoustic_ru_regional_v841", before=lambda a: None)
    with pytest.raises(RuntimeError, match="already tapped"):
        tap_regional_launch(kernels, "acoustic_ru_regional_v841", before=lambda a: None)


@pytest.mark.parametrize("module", [cuda_fullstep_lts_active, cuda_fullstep_lts_multi])
def test_both_executors_tap_through_the_route_aware_helper(module):
    source = open(module.__file__, encoding="utf-8").read()
    start = source.index("    def _attach_matched_flux(")
    body = source[start:source.index("\n    def ", start + 10)]
    assert "tap_acoustic_momentum(" in body and "tap_kernel(" not in body
