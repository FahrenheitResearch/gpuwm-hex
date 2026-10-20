"""A pre-launched move segment defers its device admission until the carrier is sealed.

Breakage prevented: at 470k cells the member being replaced holds ~54 GB of a 96 GB card, so the
next member's process, started early to overlap its host preparation, was refused by admission on
every move and each move fell back to a cold launch.  The decision is taken again once the carrier
exists; it still refuses if the card never frees.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hexcore import moving_control


def _carrier(tmp_path):
    path = tmp_path / "boundary.pkl"
    path.write_bytes(b"x")
    path.with_suffix(".pkl.json").write_text("{}")
    return path


def test_deferred_admission_retries_until_the_card_frees(tmp_path):
    answers = iter([False, False, True])
    calls = []
    moving_control.defer_admission(lambda: SimpleNamespace(admitted=next(answers)),
                                   lambda v: calls.append(v.admitted))
    moving_control.wait_for_carrier(_carrier(tmp_path), 5)
    assert calls == [True]  # required once, on the admitted verdict
    assert moving_control._DEFERRED_ADMISSION == []


def test_deferred_admission_still_refuses_when_the_card_never_frees():
    def require(verdict):
        if not verdict.admitted:
            raise RuntimeError("device memory admission refused")

    with pytest.raises(RuntimeError, match="admission refused"):
        moving_control._readmit_until_free(lambda: SimpleNamespace(admitted=False), require,
                                           timeout_seconds=0.05, poll_seconds=0.01)


def test_no_deferred_admission_leaves_the_wait_unchanged(tmp_path):
    moving_control._DEFERRED_ADMISSION.clear()
    assert moving_control.wait_for_carrier(_carrier(tmp_path), 5) >= 0.0
