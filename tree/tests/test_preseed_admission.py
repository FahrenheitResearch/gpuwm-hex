"""The corrected metadata row does not broaden any geometry or physics gate."""
from dataclasses import replace
import pytest

from hexcore.cuda_backend.regional_admission import ADMITTED_CLASSES


def test_corrected_class_is_exact_and_does_not_widen_the_previous_class():
    corrected = ADMITTED_CLASSES["graded-824m-dt5-z7"]
    previous = ADMITTED_CLASSES["graded-869m-dt5-z7"]
    assert corrected.finest_edge_band_relative == 0
    assert previous.finest_edge_band_relative == 0.05
    assert corrected.key.matches(corrected.key)
    assert not previous.key.matches(corrected.key, band=previous.finest_edge_band_relative)


@pytest.mark.parametrize("field,value", [("finest_edge_mm",824452), ("boundary_zone_width",6),
    ("n_vert_levels",54), ("dt_ms",5001), ("kernel_set_sha256","different")])
def test_corrected_class_keeps_every_key_gate(field,value):
    corrected = ADMITTED_CLASSES["graded-824m-dt5-z7"]
    assert not corrected.key.matches(replace(corrected.key, **{field:value}))
