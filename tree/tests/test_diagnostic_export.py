"""Diagnostic captures must not copy unused restart volumes to the host."""

from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.diagnostic_export import diagnostic_export


class Seam:
    def __init__(self):
        self.pending = False
        self.copies = []
        self.arrays = {
            # Preserve signed zero, a negative value and a NaN payload.
            "fields/q2": np.array([0x80000000, 0xb6000000, 0x7fc00042],
                                  dtype=np.uint32).view(np.float32)[None, None, :],
            "fields/tslb": np.arange(24, dtype=np.float32).reshape(4, 1, 6)[:, :, ::2],
            "scratch/unused_volume": np.zeros((55, 3), dtype=np.float32),
        }
        self._cp = SimpleNamespace(asnumpy=self.host_copy)

    def host_copy(self, value):
        self.copies.append(value)
        return np.array(value, copy=True)

    def _restart_manifest(self):
        return dict(self.arrays)

    def export_state(self):
        if self.pending:
            raise RuntimeError("export requires step boundary")
        return {"identity": {"row": "test"},
                "arrays": {k: self._cp.asnumpy(v) for k, v in self._restart_manifest().items()},
                "scalars": {"elapsed_seconds": 300, "call_counts": {"physics": 16}}}


def test_selected_bytes_shapes_metadata_and_live_seam_survive(monkeypatch):
    seam = Seam()
    before = seam.export_state()
    seam.copies.clear()
    monkeypatch.setenv("GPUWM_HEX_SELECTIVE_DIAGNOSTICS", "1")
    after, inventory = diagnostic_export(seam, ("q2", "tslb", "optional_absent"))
    assert inventory == tuple(before["arrays"])
    assert after["identity"] == before["identity"]
    assert after["scalars"] == before["scalars"]
    assert set(after["arrays"]) == {"fields/q2", "fields/tslb"}
    assert len(seam.copies) == 2
    assert all(value is not seam.arrays["scratch/unused_volume"] for value in seam.copies)
    for key, value in after["arrays"].items():
        expected = before["arrays"][key]
        assert value.shape == expected.shape and value.dtype == expected.dtype
        assert value.tobytes() == expected.tobytes()
        assert not np.shares_memory(value, seam.arrays[key])
    after["arrays"]["fields/q2"][...] = 1
    assert seam.arrays["fields/q2"].tobytes() == before["arrays"]["fields/q2"].tobytes()
    assert "_restart_manifest" not in seam.__dict__
    # Public restarts still cover every persisted field after diagnostics.
    assert set(seam.export_state()["arrays"]) == set(before["arrays"])


@pytest.mark.parametrize("mode", ["0", "1"])
def test_boundary_refusal_remains_armed(monkeypatch, mode):
    seam = Seam()
    seam.pending = True
    monkeypatch.setenv("GPUWM_HEX_SELECTIVE_DIAGNOSTICS", mode)
    with pytest.raises(RuntimeError, match="step boundary"):
        diagnostic_export(seam, ("q2",))
    assert seam.copies == []
    assert "_restart_manifest" not in seam.__dict__


def test_default_complete_and_bad_switch_refuses(monkeypatch):
    seam = Seam()
    monkeypatch.delenv("GPUWM_HEX_SELECTIVE_DIAGNOSTICS", raising=False)
    payload, inventory = diagnostic_export(seam, ("q2",))
    assert tuple(payload["arrays"]) == inventory == tuple(seam.arrays)
    monkeypatch.setenv("GPUWM_HEX_SELECTIVE_DIAGNOSTICS", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        diagnostic_export(seam, ("q2",))


def test_resident_history_keeps_boundary_guard_and_public_restart(monkeypatch):
    seam = Seam()
    monkeypatch.delenv("GPUWM_HEX_SELECTIVE_DIAGNOSTICS", raising=False)
    payload, inventory = diagnostic_export(seam, ("q2",), resident=True)
    assert payload["arrays"]["fields/q2"] is seam.arrays["fields/q2"]
    assert inventory == tuple(seam.arrays)
    assert not seam.copies
    assert "_cp" not in seam.__dict__ or seam._cp.asnumpy == seam.host_copy
    assert set(seam.export_state()["arrays"]) == set(seam.arrays)
    seam.pending = True
    with pytest.raises(RuntimeError, match="step boundary"):
        diagnostic_export(seam, ("q2",), resident=True)


def test_engine_held_olr_reaches_resident_history(monkeypatch):
    seam = Seam()
    seam.arrays['diag/olr'] = np.full((1, 3), 240., dtype=np.float32)
    monkeypatch.delenv('GPUWM_HEX_SELECTIVE_DIAGNOSTICS', raising=False)
    payload, inventory = diagnostic_export(seam, ('q2', 'olr'), resident=True)
    assert set(payload['arrays']) == {'fields/q2', 'diag/olr'}
    assert payload['arrays']['diag/olr'] is seam.arrays['diag/olr']
    assert inventory == tuple(seam.arrays)
    assert not seam.copies


@pytest.mark.parametrize("failure", [None, "before", "after"])
def test_complete_export_queues_all_fields_and_drains_on_refusal(monkeypatch, failure):
    from hexcore import snapshot_transfer
    seam = Seam()
    expected = seam.export_state()
    seam.copies.clear()
    seam._cp.cuda = object()  # CPU stand-in opts into the stream transfer path.
    events = []
    transfers = []

    class Transfer:
        def __init__(self, cp):
            self.sources = []
            transfers.append(self)
        def __enter__(self): return self
        def __exit__(self, kind, value, tb):
            if kind is not None:
                self.wait()
        def download(self, value, cp):
            target = np.empty(value.shape, dtype=value.dtype)
            self.sources.append((value, target))
            events.append("queue")
            return target
        def seal(self): events.append("seal")
        def wait(self):
            events.append("wait")
            for source, target in self.sources:
                np.copyto(target, source)
            self.sources.clear()

    monkeypatch.setattr(snapshot_transfer, "SnapshotTransfer", Transfer)
    monkeypatch.setenv("GPUWM_HEX_SELECTIVE_DIAGNOSTICS", "0")
    original = Seam.export_state
    def export(self):
        if failure == "before": raise RuntimeError("before copies")
        result = original(self)
        if failure == "after": raise RuntimeError("after copies")
        return result
    monkeypatch.setattr(Seam, "export_state", export)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            diagnostic_export(seam, ("q2",))
    else:
        actual, inventory = diagnostic_export(seam, ("q2",))
        assert inventory == tuple(expected["arrays"])
        assert actual["identity"] == expected["identity"]
        assert actual["scalars"] == expected["scalars"]
        for name in inventory:
            assert actual["arrays"][name].tobytes() == expected["arrays"][name].tobytes()
            assert not np.shares_memory(actual["arrays"][name], seam.arrays[name])
        assert events == ["queue"] * len(inventory) + ["seal", "wait"]
    assert events[-1] == "wait"
    assert transfers[0].sources == []
    assert seam.copies == []
    assert seam._cp.asnumpy == seam.host_copy
