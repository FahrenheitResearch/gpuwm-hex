"""#373: a float32 digest belongs to a C library, and now it says so.

These tests hold the instrument that identifies a libm by its own answer
rather than by a version string, and they validate it in BOTH directions --
it must agree with itself on one box, and it must be able to tell two
different answers apart.  An identifier that cannot separate is not an
identifier, and this program has been given confident wrong answers by five
instruments in one night for exactly that reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexcore.libm_identity import (
    KNOWN_SCALAR_POW_LIBMS,
    SCALAR_POW_PROBE_SIZE,
    correctly_rounded_square_fingerprint,
    describe_float_environment,
    refuse_unless_recorded_libm_matches,
    scalar_pow_fingerprint,
    scalar_pow_probe_inputs,
)
from hexcore.vertical import edge_dc_squared_over_twelve


def test_the_probe_inputs_are_the_same_bytes_on_every_numpy() -> None:
    """The input must not be the variable.

    A fingerprint built on ``numpy.random`` would move when numpy moves, and
    a lane would then read a numpy upgrade as a libm change.  These values
    come out of integer arithmetic and a cast, so two boxes with different
    numpy versions still probe identical bytes -- which is what makes a
    difference in the OUTPUT attributable to the library.
    """

    x = scalar_pow_probe_inputs(4096)
    assert x.dtype == np.float32
    assert x.size == 4096
    # Recomputed from scratch here rather than imported, so this is a second
    # opinion on the construction and not a restatement of it.
    i = np.arange(4096, dtype=np.uint32)
    golden = np.uint32(2654435761)
    bits = (i * golden) ^ ((i * golden) >> np.uint32(13))
    frac = (bits & np.uint32(0x00FFFFFF)).astype(np.float64) / float(1 << 24)
    expected = (1.0e3 + frac * 9.99e5).astype(np.float32)
    assert x.tobytes() == expected.tobytes()
    # The magnitudes a real mesh's edge lengths carry, so the probe exercises
    # the same corner of powf's domain the artifact does.
    assert 1.0e3 <= float(x.min()) and float(x.max()) < 1.0e6


def test_the_fingerprint_is_stable_on_this_box() -> None:
    """Direction one: the instrument agrees with itself."""

    assert scalar_pow_fingerprint(4096) == scalar_pow_fingerprint(4096)
    assert correctly_rounded_square_fingerprint(4096) == (
        correctly_rounded_square_fingerprint(4096)
    )


def test_the_fingerprint_separates_two_different_answers() -> None:
    """Direction two: the instrument is not blind.

    A libm that returned a one-ULP-different answer for a single value out of
    the probe must produce a different fingerprint, or the identifier cannot
    do the job it exists for.  Simulated by moving one bit of the OUTPUT,
    which is exactly the size of the real disagreement: every one of the 215
    differing edges on the real parent moved by one ULP and no more.
    """

    x = scalar_pow_probe_inputs(4096)
    twos = np.full(x.shape, 2.0, dtype=x.dtype)
    baseline = np.power(x, twos)
    nudged = baseline.copy()
    nudged.view(np.uint32)[17] += np.uint32(1)

    import hashlib

    digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()[:16]
    assert digest(baseline) == scalar_pow_fingerprint(4096)
    assert digest(nudged) != digest(baseline), (
        "the fingerprint cannot see a one-ULP move, so it cannot tell two "
        "libms apart and every conclusion drawn from it would be worthless"
    )


def test_this_box_is_a_libm_we_have_measured() -> None:
    """The estate is enumerated, and an unknown box is a stated gap.

    Not an assertion that every possible platform is listed -- it is an
    assertion that THIS one is, so a receipt written here names a real
    library instead of a null.
    """

    fingerprint = scalar_pow_fingerprint()
    assert fingerprint in KNOWN_SCALAR_POW_LIBMS, (
        f"this box's powf fingerprint {fingerprint} is not in "
        f"KNOWN_SCALAR_POW_LIBMS, so any float32 artifact minted here would "
        f"carry a digest belonging to a library nobody has written down. Add "
        f"a row -- additive, never editing an existing one."
    )
    row = KNOWN_SCALAR_POW_LIBMS[fingerprint]
    assert row["libm"] and row["measured_on"] and row["measured_date"]


def test_the_receipt_block_names_the_library_and_says_why() -> None:
    block = describe_float_environment()
    assert block["identified_libm"], "a receipt would carry a null library name"
    assert block["scalar_pow_fingerprint"] == scalar_pow_fingerprint()
    assert "powf" in block["why"]


def test_two_libms_disagreeing_is_refused_by_name_not_by_mismatch() -> None:
    """The refusal must say what breaks, not merely that two strings differ.

    Gate law: a gate that does not name its breakage does not exist.  The
    breakage here is a lane spending a day bisecting a defect that is not
    there, which is what happened on 2026-08-27 before this existed.
    """

    other = next(k for k in KNOWN_SCALAR_POW_LIBMS if k != scalar_pow_fingerprint())
    with pytest.raises(ValueError) as excinfo:
        refuse_unless_recorded_libm_matches(other, what="the vertical artifact")
    message = str(excinfo.value)
    assert "the vertical artifact" in message
    assert "215 of 338,022" in message
    assert "not evidence of a defect" in message
    assert "Do NOT widen a tolerance" in message


def test_an_artifact_with_no_fingerprint_is_refused_rather_than_assumed() -> None:
    with pytest.raises(ValueError) as excinfo:
        refuse_unless_recorded_libm_matches(None, what="an older artifact")
    message = str(excinfo.value)
    assert "carries no libm fingerprint" in message
    assert "cannot be told apart from a real defect" in message


def test_the_same_libm_is_admitted() -> None:
    """The refusal is not a blanket one: the matching box passes."""

    refuse_unless_recorded_libm_matches(
        scalar_pow_fingerprint(), what="an artifact minted here"
    )


def test_the_correctly_rounded_control_is_the_portable_one() -> None:
    """The measured fact the whole diagnosis rests on, held as a test.

    ``np.square`` was measured byte-identical on glibc 2.43 and the MSVC CRT
    while ``np.power`` was not.  So the portable form exists and the split is
    the library, not the inputs.  This asserts the property that makes it
    portable -- an exact multiply -- rather than a digest that would have to
    be re-recorded per platform, which would defeat the point.
    """

    x = scalar_pow_probe_inputs(1 << 14)
    square = np.square(x)
    by_hand = (x.astype(np.float64) * x.astype(np.float64)).astype(np.float32)
    assert square.tobytes() == by_hand.tobytes(), (
        "np.square is no longer a single correctly-rounded multiply, so the "
        "portable alternative named in #373 is not portable any more"
    )


def test_the_artifact_path_still_uses_the_scalar_power_on_purpose() -> None:
    """`edge_dc_squared_over_twelve` must keep reproducing the record.

    Until the coordinator rules on re-registration, this function's job is to
    reproduce the artifact of record exactly -- which means the libm answer,
    not the correctly-rounded one.  If someone "fixes" it to ``np.square``,
    every registered vertical digest stops reproducing on the box that made
    it, and this test is where they find that out.
    """

    x = scalar_pow_probe_inputs(1 << 15)
    twos = np.full(x.shape, 2.0, dtype=x.dtype)
    expected = (np.power(x, twos) / 12.0).astype(np.float32)
    got = edge_dc_squared_over_twelve(x, np.dtype(np.float32))
    assert got.tobytes() == expected.tobytes(), (
        "edge_dc_squared_over_twelve no longer reproduces numpy's scalar "
        "power. Every registered vertical artifact digest was minted through "
        "that path; changing it re-mints all of them and is a coordinator "
        "decision, not a cleanup."
    )


def test_the_probe_is_wide_enough_to_separate_the_measured_libms() -> None:
    """The width is justified by measurement, not chosen for comfort.

    The two libms measured disagree for roughly 0.065 % of float32 arguments.
    A probe must be wide enough that agreeing by luck is not possible; at the
    shipped width the expected number of differing values is in the hundreds.
    """

    assert SCALAR_POW_PROBE_SIZE >= 1 << 16
    expected_differences = SCALAR_POW_PROBE_SIZE * 0.00065
    assert expected_differences > 50.0
