from types import SimpleNamespace
import numpy as np
import pytest
from hexcore.final_identity import carried_digest
from hexcore.cuda_dualrun import STATE_FIELDS, SIDECAR_FIELDS


def fixture():
    atmo = SimpleNamespace(
        state=SimpleNamespace(**{key: np.arange(4, dtype=np.float32) for key in STATE_FIELDS}, time_seconds=5.0),
        saved=SimpleNamespace(**{key: np.ones(4, dtype=np.float32) for key in SIDECAR_FIELDS}))
    backend = {"words": np.ones(4, dtype=np.float32), "counter": 1}
    gf = SimpleNamespace(rthdynten=np.ones(4, dtype=np.float32), rqvdynten=np.ones(4, dtype=np.float32), time_seconds=5.0)
    return atmo, backend, gf, {"tsk": np.ones(4, dtype=np.float32)}


def test_carried_digest_binds_all_owners_and_inventory():
    args = fixture()
    before = carried_digest(*args)
    assert before["schema"] == "gpuwm-canonical-state-v2"
    for owner in (args[0].state.rho, args[0].saved.__dict__[SIDECAR_FIELDS[0]], args[1]["words"], args[2].rqvdynten, args[3]["tsk"]):
        old = owner[0].copy()
        owner[0] += 1
        after = carried_digest(*args)
        assert after["sha256"] != before["sha256"]
        assert after["inventory_sha256"] == before["inventory_sha256"]
        owner[0] = old
    args[1]["counter"] += 1
    assert carried_digest(*args)["scalar_sha256"] != before["scalar_sha256"]


def test_carried_digest_refuses_unknown_or_missing_carriers():
    args = list(fixture())
    args[2] = None
    with pytest.raises(ValueError, match="GF forcing"):
        carried_digest(*args)
    args = list(fixture())
    args[1]["unknown"] = object()
    with pytest.raises(TypeError, match="unregistered"):
        carried_digest(*args)


def test_carried_digest_binds_declared_absent_gf():
    args = list(fixture())
    present = carried_digest(*args)
    args[2] = None
    absent = carried_digest(*args, gf_required=False)
    assert absent["sha256"] != present["sha256"]
    assert absent["inventory_sha256"] != present["inventory_sha256"]
    assert absent == carried_digest(*args, gf_required=False)
    args[1]["counter"] += 1
    assert carried_digest(*args, gf_required=False)["sha256"] != absent["sha256"]


"""Mutations in carried state must invalidate the complete boundary digest."""
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
import pytest

from hexcore.cuda_dualrun import STATE_FIELDS, SIDECAR_FIELDS
from hexcore.final_identity import final_digest


def test_every_carrier_and_scalar_changes_the_digest():
    pytest.importorskip("gpuwm.state_digest")
    state = SimpleNamespace(**{name: np.array([1.0], dtype="f4") for name in STATE_FIELDS}, time_seconds=5.0)
    saved = SimpleNamespace(**{name: np.array([-0.0], dtype="f4") for name in SIDECAR_FIELDS})
    physics = {"soil": np.array([0.5], dtype="f4"), "counts": {"surface": 1}}
    stack = {"driver": SimpleNamespace(atmosphere=SimpleNamespace(state=state, saved=saved)),
             "backend": SimpleNamespace(restart_state=lambda: physics)}
    forcing = SimpleNamespace(rthdynten=np.array([2.0], dtype="f4"),
                              rqvdynten=np.array([3.0], dtype="f4"), time_seconds=5.0)
    surface = {"rain": np.array([0.0], dtype="f4")}
    @dataclass
    class Updates:
        precipitation: object
    updates = Updates(np.array([4.0], dtype="f4"))
    reference = final_digest(stack, forcing, surface, updates)
    arrays = [getattr(state, name) for name in STATE_FIELDS] + [getattr(saved, name) for name in SIDECAR_FIELDS]
    arrays += [physics["soil"], forcing.rthdynten, forcing.rqvdynten, surface["rain"], updates.precipitation]
    for array in arrays:
        word = array.view("u4")
        previous = word[0].copy()
        word[0] ^= np.uint32(1)
        changed = final_digest(stack, forcing, surface, updates)
        assert changed["sha256"] != reference["sha256"]
        assert changed["inventory_sha256"] == reference["inventory_sha256"]
        word[0] = previous
    physics["counts"]["surface"] += 1
    changed = final_digest(stack, forcing, surface, updates)
    assert changed["scalar_sha256"] != reference["scalar_sha256"]
    assert changed["sha256"] != reference["sha256"]
    physics["counts"]["surface"] -= 1
    forcing.time_seconds += 1
    assert final_digest(stack, forcing, surface, updates)["scalar_sha256"] != reference["scalar_sha256"]
    forcing.time_seconds -= 1
    surface["unknown"] = object()
    with pytest.raises(TypeError, match="unbound final-state carrier"):
        final_digest(stack, forcing, surface, updates)
