"""GPU-tier entry for the existing complete synthetic non-LES replay.

Keep the original test implementation and its source-pinned receipts intact.
The separate entry lets the card battery collect every device case without
making the CPU configuration test file a member of two tiers.
"""
from test_dycore_options_v842 import (
    native_d1,
    device_d1,
    test_complete_nonles_module_bits,
    test_negative_gradient_zero_euler_seed_bits,
    test_native_cam_and_rayleigh_bits,
)
