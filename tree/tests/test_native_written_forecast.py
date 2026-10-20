"""Reject incomplete and nonzero M0 proof receipts; metadata tests only."""
import site_support
import copy
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).parents[1] / 'tools/native_physics_reference/backend/check_written_forecast.py'
_spec = importlib.util.spec_from_file_location('native_written_forecast', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.fixture
def evidence():
    rows = [dict(stream='output', field=name, step=step, time=str(step),
                 compared_words=4, differing_words=0, status='compared')
            for name in ('theta', 'qv') for step in (0, 72)]
    rows.append(dict(stream='output', field='deferred', step=72, time='72',
                     compared_words=4, differing_words=4, status='field missing in hex'))
    deck = dict(process_count=1, pin_set='pin', mesh='mesh', date='date',
                suite='suite', duration_steps=72, expected_samples={'output': [0, 72]},
                native={'sha256': 'publication'},
                hex_support_inputs={'native-library': {'sha256': 'library'}})
    inventory = {**{k: deck[k] for k in ('process_count', 'pin_set', 'mesh', 'date',
                                       'suite', 'duration_steps')},
                 'native_manifest_sha256': 'manifest',
                 'native_publication_sha256': 'publication',
                 'samples': [{**{k: r[k] for k in ('stream', 'field', 'step', 'time')},
                              'words': r['compared_words'],
                              'missing_allowed': r['status'] != 'compared'} for r in rows]}
    receipt = dict(model_execution=True, evidence_kind='FORECAST', failed_model_exit=None,
                   process_count=1, pin_set='pin', mesh='mesh', date='date',
                   provenance={'deck_sha256': 'deck', 'native_manifest_sha256': 'manifest'},
                   comparison={'fields': rows}, passed=False, tier=None,
                   failures=['output/deferred step 72: field missing in hex'])
    completion = dict(completed=True, steps_requested=72, steps_completed=72,
                      native_library_sha256='library', physics={'suite': 'suite'})
    return receipt, completion, deck, inventory


def test_scoped_bitwise_preserves_official_missing_grade(evidence):
    before = copy.deepcopy(evidence)
    result = _module.check(*evidence, deck_sha256='deck')
    assert result['passed'] and result['computed_words'] == 16
    assert result['missing_words'] == 4 and not result['full_field_bitwise']
    assert result['official_passed'] is False and result['official_tier'] is None
    assert evidence == before


@pytest.mark.parametrize('defect', ['one_word', 'new_missing', 'duplicate', 'truncated',
                                    'partial_run', 'rank', 'library', 'manifest',
                                    'word_count', 'refusal', 'unrelated_failure', 'deck', 'boolean_rank'])
def test_rejects_invalid_whole_forecast_evidence(evidence, defect):
    r, c, d, _ = evidence
    row = r['comparison']['fields'][0]
    if defect == 'one_word': row['differing_words'] = 1
    elif defect == 'new_missing': row['status'] = 'field missing in hex'
    elif defect == 'duplicate': r['comparison']['fields'].append(dict(row))
    elif defect == 'truncated': r['comparison']['fields'].pop()
    elif defect == 'partial_run': c['steps_completed'] = 1
    elif defect == 'rank': r['process_count'] = 4
    elif defect == 'library': c['native_library_sha256'] = 'other'
    elif defect == 'manifest': r['provenance']['native_manifest_sha256'] = 'other'
    elif defect == 'word_count': row['compared_words'] = 3
    elif defect == 'refusal': r['evidence_kind'] = 'REFUSAL'
    elif defect == 'unrelated_failure': r['failures'].append('input digest differs')
    elif defect == 'deck': r['provenance']['deck_sha256'] = 'other'
    elif defect == 'boolean_rank': r['process_count'] = True
    with pytest.raises(ValueError):
        _module.check(*evidence, deck_sha256='deck')


def test_newly_implemented_deferred_field_is_still_compared(evidence):
    r, _, _, _ = evidence
    row = r['comparison']['fields'][-1]
    row.update(status='compared', differing_words=0)
    r['failures'] = []
    result = _module.check(*evidence, deck_sha256='deck')
    assert result['computed_words'] == 20 and result['missing_words'] == 0
    assert result['full_field_bitwise']


# Complete MPAS diagnostics lifecycle and actual stream-writer contracts.
import itertools
from hexcore.mpasdiag import DiagnosticsUtils, Direction, StreamView, DiagnosticsManager


def test_field_queries_follow_current_output_alarm_and_active_fields():
    utils = DiagnosticsUtils()
    current = []
    utils.init(lambda: iter(current))
    for direction, active, alarm, field in itertools.product(Direction, (False, True), (False, True), (False, True)):
        current[:] = [StreamView("s", direction, active, alarm, {"mslp": field})]
        expected = active and alarm and field and direction in (Direction.OUTPUT, Direction.INPUT_OUTPUT)
        assert utils.field_will_be_written("mslp") == expected
        assert not utils.field_will_be_written("missing")
    current[:] = [StreamView("a", Direction.OUTPUT, True, False, {"mslp": True}),
                  StreamView("b", Direction.INPUT_OUTPUT, True, True, {"mslp": True})]
    assert utils.field_will_be_written("mslp")
    assert utils.field_will_be_written("mslp ")
    current.clear()
    assert not utils.field_will_be_written("mslp")


def test_precompiled_cuda_artifact_rejects_stale_source_flags_and_bytes(tmp_path):
    import hashlib
    import json
    import pytest
    from hexcore.mpasdiag._artifact import OPTIONS, authenticate, artifact_architecture
    path = tmp_path / "diagnostics.ptx"
    path.write_bytes(b"compiled artifact")
    source = "assembled device source"
    manifest = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    ptx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    options=[*OPTIONS, "--gpu-architecture=compute_120"])
    receipt = path.with_suffix(".ptx.json")
    receipt.write_text(json.dumps(manifest))
    assert authenticate(path, source, "compute_120") == path
    assert artifact_architecture(path, "120") == "compute_120"
    with pytest.raises(ValueError, match="source_sha256 mismatch"):
        authenticate(path, source + " changed", "compute_120")
    with pytest.raises(ValueError, match="options mismatch"):
        authenticate(path, source, "compute_100")
    path.write_bytes(b"corrupted artifact")
    with pytest.raises(ValueError, match="ptx_sha256 mismatch"):
        authenticate(path, source, "compute_120")
    manifest.update(format="cubin", cubin_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    options=[*OPTIONS, "--gpu-architecture=sm_89"])
    receipt.write_text(json.dumps(manifest))
    assert authenticate(path, source, "sm_89") == path
    assert artifact_architecture(path, "89") == "sm_89"
    with pytest.raises(ValueError, match="options mismatch"):
        authenticate(path, source, "sm_120")
    manifest['format'] = 'unknown'
    receipt.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format mismatch"):
        artifact_architecture(path, "89")


def test_inclusion_count_direction_interval_and_inactive_streams():
    utils = DiagnosticsUtils()
    current = []
    utils.init(lambda: iter(current))
    for sd, direction, active, field, ii, oi, inactive in itertools.product(
            Direction, Direction, (False, True), (False, True), ("none", " 1 ", " none "), ("none", "1"), (False, True)):
        current[:] = [StreamView("s", sd, active, False, {"mslp": field}, ii, oi)]
        eligible = inactive or (active and (ii.rstrip() != "none" if direction == Direction.INPUT else
                    oi != "none" if direction == Direction.OUTPUT else ii.rstrip() != "none" or oi != "none"))
        matches = direction == Direction.INPUT_OUTPUT or sd == Direction.INPUT_OUTPUT or sd == direction
        assert utils.stream_inclusion_count("mslp", direction, inactive) == int(eligible and matches and field)
    current *= 2
    assert utils.stream_inclusion_count("mslp", include_inactive=True) == 2
    assert utils.stream_inclusion_count("missing", include_inactive=True) == 0


class RecordingDiagnostic:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls
    def setup(self, *args):
        self.calls.append((self.name, "setup", args))
    def update(self):
        self.calls.append((self.name, "update"))
    def compute(self):
        self.calls.append((self.name, "compute"))
    def reset(self):
        self.calls.append((self.name, "reset"))
    def cleanup(self):
        self.calls.append((self.name, "cleanup"))


def test_manager_exact_native_lifecycle_and_setup_arguments():
    calls = []
    names = ("isobaric", "cloud", "convective", "pv", "soundings")
    manager = DiagnosticsManager(**{name: RecordingDiagnostic(name, calls) for name in names})
    manager.template = RecordingDiagnostic("template", calls)
    pools, configs, clock, dm = {}, object(), object(), object()
    manager.setup(lambda: (), configs, pools, clock, dm)
    assert [c[:2] for c in calls] == [(name, "setup") for name in ("template", *names)]
    assert calls[0][2] == (configs, pools, clock)
    assert all(call[2] == (pools, clock, manager.utils) for call in calls[1:5])
    assert calls[-1][2] == (configs, pools, clock, dm)
    calls.clear()
    # Native initial output resets, updates, computes, writes, resets.
    manager.reset(); manager.update(); manager.compute(); manager.reset(); manager.cleanup()
    assert calls == [("template", "reset"), ("convective", "reset"),
                     ("template", "update"), ("convective", "update"),
                     *((name, "compute") for name in ("template", *names)),
                     ("template", "reset"), ("convective", "reset"),
                     ("template", "cleanup"), ("soundings", "cleanup")]


def test_v842_citation_windows_and_no_uncited_kernel():
    """Pin citations against the supplied source, independently of v8.2.3 gate."""
    import hashlib
    import os
    import re
    from collections import Counter
    from pathlib import Path
    import pytest
    from hexcore.mpasdiag._citation_pins import PINS
    root = Path(__file__).parents[1] / "src/hexcore/mpasdiag"
    counts = Counter()
    pattern = re.compile(r'(?P<file>[A-Za-z0-9_]+\.F):(?P<ranges>\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)')
    for path in root.glob("*.py"):
        # Lines citing the RRFS fork (RRFSx/MPAS-Model@<commit>) name fork files, not v8.4.2;
        # test_mpas_convective_rrfs.py::test_fork_citations seals those windows.
        text = "\n".join(line for line in path.read_text().splitlines() if "RRFSx/MPAS-Model@" not in line)
        for match in pattern.finditer(text):
            for rg in match["ranges"].split(","):
                nums = list(map(int, rg.split("-")))
                counts[(match["file"], nums[0], nums[-1])] += 1
    assert counts == Counter({(n, first, last): count for n, _, first, last, _, count in PINS})
    ref = site_support.mpas_source(preferred=('MPAS_REF',))
    if not ref.is_dir():
        pytest.skip("pinned MPAS-A v8.4.2 source is not installed")
    for _, path, first, last, digest, _ in PINS:
        lines = (ref / path).read_text().splitlines()
        assert hashlib.sha256("\n".join(lines[first - 1:last]).encode()).hexdigest() == digest


def test_native_output_phase_initial_restart_and_completed_step():
    from hexcore.mpasdiag import OutputDiagnosticsCycle
    calls = []
    class Streams:
        alarm = True
        def reset_output_alarms(self, stream_id=None):
            calls.append(("alarm-reset", stream_id))
        def ringing_output_alarms(self):
            return self.alarm
        def write(self):
            calls.append("write")
    streams = Streams()
    manager = RecordingDiagnostic("manager", calls)
    cycle = OutputDiagnosticsCycle(manager, streams, lambda: calls.append("output-fields/level-1"))
    cycle.initial_output(restart=True)
    assert calls == [("alarm-reset", "restart"), ("alarm-reset", "output"),
                     ("alarm-reset", "diagnostics"), "output-fields/level-1",
                     ("manager", "reset"), ("manager", "update"), ("manager", "compute"),
                     "write", ("manager", "reset"), ("alarm-reset", None)]
    calls.clear()
    cycle.after_clock_advance()
    assert calls == ["output-fields/level-1", ("manager", "update"), ("manager", "compute"),
                     "write", ("manager", "reset"), ("alarm-reset", None)]
    calls.clear(); streams.alarm = False
    cycle.after_clock_advance()
    assert calls == [("manager", "update"), ("manager", "compute"), "write",
                     ("manager", "reset"), ("alarm-reset", None)]


def test_no_selected_isobaric_field_skips_state_and_halo_access():
    from hexcore.mpasdiag.isobaric import IsobaricDiagnostics
    calls = []
    utils = DiagnosticsUtils()
    streams = []
    utils.init(lambda: iter(streams))
    diagnostic = IsobaricDiagnostics(exchange_pressure_halo=lambda field: calls.append(field))
    diagnostic.setup({"mesh": {}, "state": {}, "diag": {}}, None, utils)
    for fields, alarm in (({}, True), ({"mslp": True}, False), ({"unrelated": True}, True), ({"mslp": False}, True)):
        streams[:] = [StreamView("diagnostics", Direction.OUTPUT, True, alarm, fields)]
        diagnostic.compute()
    assert calls == []


def test_field_inventory_covers_every_native_isobaric_stream_query():
    import os
    import re
    from pathlib import Path
    import pytest
    from hexcore.mpasdiag.isobaric import FIELDS
    root = site_support.mpas_source(preferred=('MPAS_REF',))
    if not root.is_dir():
        pytest.skip("pinned MPAS-A v8.4.2 source is not installed")
    source = (root / "src/core_atmosphere/diagnostics/mpas_isobaric_diagnostics.F").read_text()
    names = re.findall(r"MPAS_field_will_be_written\('([^']+)'\)", source)
    assert len(names) == len(FIELDS) == 68
    assert set(names) == set(FIELDS)


def test_live_alarm_adapter_preserves_field_activity_and_write_reset_order():
    from types import SimpleNamespace
    from hexcore.mpasdiag.streams import LiveStreamAdapter
    def stream(name, direction):
        return SimpleNamespace(name=name, options={"type": direction},
                               interval=lambda direction: "initial_only" if direction == "input" else "3:00:00")
    class Alarms:
        streams = {"input": stream("input", "input"),
                   "output": stream("output", "output"),
                   "inactive": stream("inactive", "input;output")}
        ringing = {"input", "output", "inactive"}
        def due(self, name, when, direction):
            assert direction == "output"
            return name in self.ringing
        def reset(self, name, when, direction):
            events.append(("reset", name, when, direction))
            self.ringing.discard(name)
    now = [1]
    events = []
    field_active = {"mslp": True}
    alarms = Alarms()
    adapter = LiveStreamAdapter(alarms, lambda: now[0],
        stream_active=lambda s: s.name != "inactive",
        field_activity=lambda s: dict(field_active),
        write_stream=lambda s, when: events.append(("write", s.name, when)))
    utils = DiagnosticsUtils(); utils.init(adapter)
    assert utils.field_will_be_written("mslp")
    assert utils.stream_inclusion_count("mslp", include_inactive=True) == 3
    assert utils.stream_inclusion_count("mslp") == 2
    assert adapter.ringing_output_alarms() == ("output",)
    adapter.write()
    assert events == [("write", "output", 1)]
    assert utils.field_will_be_written("mslp")  # Writing does not reset alarms.
    adapter.reset_output_alarms()
    assert events[1:] == [("reset", "output", 1, "output"),
                          ("reset", "inactive", 1, "output")]
    assert not utils.field_will_be_written("mslp")
    assert "input" in alarms.ringing
    now[0] = 2; alarms.ringing.add("output")
    field_active["mslp"] = False
    assert not utils.field_will_be_written("mslp")
    field_active["mslp"] = True
    assert utils.field_will_be_written("mslp")
    adapter.reset_output_alarms("output")
    adapter.reset_output_alarms("missing")
    assert events[-1] == ("reset", "output", 2, "output")
