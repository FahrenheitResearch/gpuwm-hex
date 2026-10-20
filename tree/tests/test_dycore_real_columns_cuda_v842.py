"""GPU-tier entry for every existing real-column non-LES replay case.

The original real-column fixture overrides the synthetic fixture, exactly as
in the existing real-column test module. No numerical test body is duplicated.
"""
from test_dycore_real_columns_v842 import native_d1
from test_dycore_options_v842 import (
    device_d1,
    test_complete_nonles_module_bits,
    test_native_cam_and_rayleigh_bits,
)
