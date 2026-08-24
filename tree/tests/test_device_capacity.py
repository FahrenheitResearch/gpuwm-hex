"""The preflight for the big-card tier.

The x4.163842 full-physics proof is a forty-minute run that holds about
26.4 GiB resident.  On a smaller card it does not refuse: it dies part-way
through, inside a CuPy allocation, several frames below anything the reader
recognises, after burning whatever time it took to get there.  This test is
the cheap version of that discovery -- it asks the question first, and on a
card that cannot answer it skips with the number rather than pretending the
tier passed.
"""

from __future__ import annotations

import pytest

from conftest import X4_FULL_PHYSICS_BYTES, free_device_bytes


@pytest.mark.gpu
@pytest.mark.bigcard
def test_the_card_can_host_the_x4_full_physics_footprint() -> None:
    free = free_device_bytes()
    assert free is not None, (
        "the big-card gate ran with no reachable device; the tier gate in "
        "conftest.py should have skipped this test by name before it started"
    )
    assert free >= X4_FULL_PHYSICS_BYTES, (
        f"device 0 has {free / 1024**3:.1f} GiB free; the x4.163842 "
        f"full-physics tier holds {X4_FULL_PHYSICS_BYTES / 1024**3:.1f} GiB "
        "resident and would fail mid-run"
    )
