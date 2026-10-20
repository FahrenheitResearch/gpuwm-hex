"""The measurement scope restores ordinary door defaults on every exit."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _instrument():
    path = Path(__file__).resolve().parents[1] / "src/hexcore/drivers/forecast_receipt.py"
    spec = importlib.util.spec_from_file_location("_samehost_forecast_check_instrument", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [False, True])
def test_explicit_mint_calls_unchanged_assembler_and_restores_defaults(monkeypatch, failure):
    """Since 2026-10-10 the scope opens nothing: no candidate mint is entered
    and the regional opener is never swapped, because neither gate exists."""

    from hexcore import dt_admission, cuda_regional_forecast_v841 as regional
    instrument = _instrument()
    original_registry = dt_admission.ADMITTED_TIMESTEPS
    original_open = regional.open_regional_forecast_v841
    called = []

    monkeypatch.setattr(dt_admission, "candidate_mint",
                        lambda *a, **k: called.append((a, k)))
    try:
        with instrument.measurement_scope(enabled=True, dt=60., card="test-card") as metadata:
            assert metadata["measurement_mint"] is True
            assert metadata["dt_seconds"] == 60.
            assert regional.open_regional_forecast_v841 is original_open
            if failure:
                raise RuntimeError("forced exit")
    except RuntimeError:
        assert failure
    assert called == []
    assert regional.open_regional_forecast_v841 is original_open
    assert dt_admission.ADMITTED_TIMESTEPS is original_registry


def test_no_mint_leaves_default_functions_and_registry_untouched(monkeypatch):
    from hexcore import dt_admission, cuda_regional_forecast_v841 as regional
    instrument = _instrument()
    registry, opener = dt_admission.ADMITTED_TIMESTEPS, regional.open_regional_forecast_v841
    monkeypatch.setattr(dt_admission, "candidate_mint", lambda *a, **k: pytest.fail("implicit mint"))
    with instrument.measurement_scope(enabled=False, dt=60., card=None):
        assert dt_admission.ADMITTED_TIMESTEPS is registry
        assert regional.open_regional_forecast_v841 is opener


def test_boundary_recording_keeps_every_original_driver_argument_and_restores_builder():
    instrument = _instrument()
    original = lambda request: ["--normal-option", "normal-value"]
    door = SimpleNamespace(build_driver_argv=original)
    with instrument.boundary_receipts(door, 1):
        assert door.build_driver_argv(None) == ["--normal-option", "normal-value", "--fingerprint-every", "1"]
    assert door.build_driver_argv is original


def test_source_snapshot_records_real_bytes_and_detects_a_changed_source(tmp_path):
    instrument = _instrument()
    path = tmp_path / "source.py"
    path.write_bytes(b"before\n")
    before = instrument.file_record(path)
    path.write_bytes(b"after!\n")
    assert before["bytes"] == instrument.file_record(path)["bytes"]
    assert before["sha256"] != instrument.file_record(path)["sha256"]


@pytest.mark.parametrize("failure", [False, True])
def test_explicit_profile_captures_real_worker_operations_and_restores_method(tmp_path, failure):
    from hexcore.partition_samehost_v841 import SameHostRanks
    import json
    import pstats
    import threading
    instrument = _instrument()
    previous = SameHostRanks._call
    aborted = []
    concurrent = threading.Barrier(2, timeout=2)

    def measured_worker_operation(rank):
        concurrent.wait()
        if failure and rank == 1:
            raise RuntimeError("profiled operation failed")
        return rank + 1

    directory = tmp_path / "profile"
    try:
        with instrument.profile_scope(directory) as result:
            with SameHostRanks([0, 0], SimpleNamespace(abort=aborted.append)) as ranks:
                assert ranks.call(measured_worker_operation) == [1, 2]
    except RuntimeError:
        assert failure
    assert SameHostRanks._call is previous
    assert result["worker_call_restored"] and result["workers_drained"]
    assert result["worker_calls"] == 2
    assert result["worker_original_code_sha256"] == result["worker_restored_code_sha256"]
    worker_stats = pstats.Stats(str(directory / "ranks.pstats"))
    assert any(name == "measured_worker_operation" for (_file, _line, name) in worker_stats.stats)
    assert sum(value[1] for (_file, _line, name), value in worker_stats.stats.items()
               if name == "measured_worker_operation") == 2
    for rank in (0, 1):
        one_rank = pstats.Stats(str(directory / f"rank-{rank}.pstats"))
        assert sum(value[1] for (_file, _line, name), value in one_rank.stats.items()
                   if name == "measured_worker_operation") == 1
    assert (directory / "main.pstats").is_file()
    assert (directory / "ranks.txt").is_file()
    summary = json.loads((directory / "profile-summary.json").read_text())
    assert summary["rank_profile_status"] == "measured"


def test_default_profile_scope_does_not_wrap_worker_or_create_artifacts():
    from hexcore.partition_samehost_v841 import SameHostRanks
    instrument = _instrument()
    previous = SameHostRanks._call
    with instrument.profile_scope(None) as result:
        assert result["enabled"] is False
        assert SameHostRanks._call is previous
    assert SameHostRanks._call is previous


def test_control_profile_without_workers_still_records_cli_and_empty_rank_summary(tmp_path):
    import json
    instrument = _instrument()
    directory = tmp_path / "control-profile"
    with instrument.profile_scope(directory):
        assert len([1, 2, 3]) == 3
    summary = json.loads((directory / "profile-summary.json").read_text())
    assert summary["worker_calls"] == 0
    assert summary["rank_profile_status"] == "no same-host worker calls reached"
    assert (directory / "ranks.pstats").is_file()
    assert summary["ranks"]["summary"]["total_calls"] == 0


def test_control_source_without_samehost_module_profiles_main_only(tmp_path, monkeypatch):
    import builtins
    import json
    import pstats
    instrument = _instrument()
    original_import = builtins.__import__

    def control_import(name, *args, **kwargs):
        if name == "hexcore.partition_samehost_v841":
            raise ModuleNotFoundError(
                f"No module named '{name}'", name=name
            )
        return original_import(name, *args, **kwargs)

    def control_cli_operation():
        return len([1, 2, 3])

    monkeypatch.setattr(builtins, "__import__", control_import)
    directory = tmp_path / "bare-control-profile"
    with instrument.profile_scope(directory) as result:
        assert control_cli_operation() == 3
    assert result["worker_calls"] == 0
    assert result["worker_original_code_sha256"] is None
    assert result["worker_call_restored"] and result["workers_drained"]
    stats = pstats.Stats(str(directory / "main.pstats"))
    assert any(name == "control_cli_operation" for (_file, _line, name) in stats.stats)
    summary = json.loads((directory / "profile-summary.json").read_text())
    assert summary["ranks"]["summary"]["total_calls"] == 0


def test_profile_scope_does_not_hide_a_missing_worker_dependency(tmp_path, monkeypatch):
    import builtins
    instrument = _instrument()
    original_import = builtins.__import__

    def broken_dependency(name, *args, **kwargs):
        if name == "hexcore.partition_samehost_v841":
            raise ModuleNotFoundError("No module named 'cupy'", name="cupy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_dependency)
    with pytest.raises(ModuleNotFoundError, match="cupy"):
        with instrument.profile_scope(tmp_path / "broken-dependency-profile"):
            pytest.fail("missing dependency silently changed the profile scope")


def test_existing_profiler_refusal_does_not_leave_worker_method_wrapped(tmp_path):
    import cProfile
    from hexcore.partition_samehost_v841 import SameHostRanks
    instrument = _instrument()
    previous = SameHostRanks._call
    existing = cProfile.Profile()
    existing.enable()
    try:
        with pytest.raises(ValueError, match="Another profiling tool is already active"):
            with instrument.profile_scope(tmp_path / "nested-profile"):
                pytest.fail("second main profiler unexpectedly activated")
    finally:
        existing.disable()
    assert SameHostRanks._call is previous


def test_final_forcing_observer_keeps_only_latest_carrier_and_defers_host_copies():
    import hashlib
    import weakref
    import numpy as np
    instrument = _instrument()
    transactions, carriers, copies = [], [], []

    class Transaction:
        pass

    class Carrier:
        pass

    def composite():
        carrier = Carrier()
        carrier.rthdynten = np.array([[0x80000000, 0x7FC00123]], dtype=np.uint32).view(np.float32)
        carrier.rqvdynten = np.array([[0x00000001, 0x3F800000]], dtype=np.uint32).view(np.float32)
        carrier.time_seconds = float(120 * (len(carriers) + 1))
        result = Transaction()
        result.dynamics_tendencies = carrier
        result.unrelated_transaction_owner = object()
        transactions.append(weakref.ref(result))
        carriers.append(weakref.ref(carrier))
        return result

    def host(array, cp):
        copies.append(array)
        return array

    proof = SimpleNamespace(execute_composite_step=composite, _host_array=host,
                            array_sha256=lambda value: hashlib.sha256(value.tobytes()).hexdigest())
    with instrument.final_forcing_observer(enabled=True, proof=proof) as observation:
        first = proof.execute_composite_step()
        assert first.dynamics_tendencies is observation.latest
        del first
        assert transactions[0]() is None
        assert carriers[0]() is observation.latest
        second = proof.execute_composite_step()
        expected = second.dynamics_tendencies
        del second
        assert transactions[1]() is None
        assert carriers[0]() is None
        assert copies == []
        observation.capture_hashes(proof, cp=object())
        assert len(copies) == 2
        for name in ("rthdynten", "rqvdynten"):
            array = getattr(expected, name)
            assert observation.metadata["hashes"][name] == {
                "dtype": "<f4", "shape": [1, 2],
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        assert observation.metadata["model_time_seconds"] == 240.
    assert proof.execute_composite_step is composite
    assert observation.latest is None
    assert observation.metadata["successful_steps_observed"] == 2
    assert observation.metadata["composite_step_restored"]


@pytest.mark.parametrize("where", ["step", "hash"])
def test_final_forcing_observer_restores_and_releases_on_failure(where):
    import numpy as np
    instrument = _instrument()
    carrier = SimpleNamespace(rthdynten=np.ones((1, 2), np.float32),
                              rqvdynten=np.ones((1, 2), np.float32), time_seconds=120.)

    def composite():
        if where == "step":
            raise RuntimeError("step refused")
        return SimpleNamespace(dynamics_tendencies=carrier)

    def host(*args):
        raise RuntimeError("hash transfer failed")

    proof = SimpleNamespace(execute_composite_step=composite, _host_array=host)
    with pytest.raises(RuntimeError):
        with instrument.final_forcing_observer(enabled=True, proof=proof) as observation:
            proof.execute_composite_step()
            observation.capture_hashes(proof, cp=object())
    assert proof.execute_composite_step is composite
    assert observation.latest is None
    assert observation.metadata["hashes"] is None
    assert observation.metadata["composite_step_restored"]


def test_rank_final_forcing_observer_disabled_does_not_wrap_or_import_proof(monkeypatch):
    import builtins
    instrument = _instrument()
    original_import = builtins.__import__

    def no_proof(name, *args, **kwargs):
        if name == "hexcore.drivers.run_cuda_v841_forecast":
            pytest.fail("disabled rank observer imported the single-card proof")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_proof)
    with instrument.final_forcing_observer(enabled=False) as observation:
        assert not observation.enabled
        assert observation.latest is None
    assert observation.metadata["hashes"] is None
    assert observation.metadata["composite_step_restored"]


@pytest.mark.parametrize("automatic_ranks", [1, 2])
def test_real_door_receipt_gets_final_forcing_hashes_with_bare_control_parser(tmp_path, monkeypatch, automatic_ranks):
    import json
    import numpy as np
    from hexcore import cli, forecast_door as door
    from hexcore.drivers.run_cuda_v841_forecast import proof
    instrument = _instrument()
    output = tmp_path / "forecast"
    output.mkdir()
    # e6's parser has no optional same-host arguments.
    parsed = SimpleNamespace(out=output, gpuwm_checkout=None, convection="auto")
    monkeypatch.setattr(cli, "build_parser", lambda: SimpleNamespace(parse_args=lambda argv: parsed))
    monkeypatch.setattr(instrument, "source_snapshot", lambda: {"unchanged": True})
    monkeypatch.setattr(instrument, "runtime_snapshot", lambda checkout: {"unchanged": True,
        "interpreter": {"Py_GIL_DISABLED": 1, "gil_enabled": False}})
    bits = np.array([[0x80000000, 0x7FC00123]], dtype=np.uint32).view(np.float32)
    carrier = SimpleNamespace(rthdynten=bits, rqvdynten=bits.copy(), time_seconds=120.)
    composite = lambda: SimpleNamespace(dynamics_tendencies=carrier)
    monkeypatch.setattr(proof, "execute_composite_step", composite)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(ndarray=np.ndarray,
                                                            asnumpy=lambda value: value.copy()))

    def forecast(argv):
        # --dt is forwarded to the forecast when its arguments carry none.
        assert argv == ["forecast", "--dt", "120.0"]
        proof.execute_composite_step()
        result = {"steps_executed": 1, "steps_requested": 1}
        if automatic_ranks > 1:
            result.update(devices=[0, 1], final_gf_tendency_hashes={"owned_union": "rank driver receipt"})
        (output / door.DRIVER_RECEIPT_NAME).write_text(json.dumps({"status": "passed", "forecast": result}))
        return 0

    monkeypatch.setattr(cli, "main", forecast)
    arguments = SimpleNamespace(devices=None, fingerprint_every=1,
                                report=tmp_path / "measurement.json", tip="test-tip",
                                measurement_mint=False, dt=120., card=None, profile_dir=None)
    assert instrument.run(arguments, []) == 0
    receipt = json.loads(arguments.report.read_text())
    assert receipt["driver_completed"] and receipt["ordinary_door_restored"]
    assert receipt["final_forcing_observation"]["successful_steps_observed"] == 1
    expected = {
        name: {"dtype": "<f4", "shape": [1, 2],
               "sha256": proof.array_sha256(bits)}
        for name in ("rthdynten", "rqvdynten")
    }
    assert receipt["final_gf_tendency_hashes"] == (expected if automatic_ranks == 1 else {"owned_union": "rank driver receipt"})
    assert receipt["parallel_runtime_qualification"]["required_for_samehost"] == (automatic_ranks > 1)
    assert proof.execute_composite_step is composite


def test_runtime_snapshot_records_actual_interpreter_build_and_live_gil_state():
    import sysconfig
    instrument = _instrument()
    record = instrument.runtime_snapshot(None)["interpreter"]
    assert record["version"] == sys.version
    assert record["Py_GIL_DISABLED"] == sysconfig.get_config_var("Py_GIL_DISABLED")
    assert record["gil_enabled"] == (sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else None)


def test_live_gil_change_is_recorded_without_claiming_source_bytes_changed():
    instrument = _instrument()
    before = {"source": {"sha": "same"}, "inputs": {"sha": "same"},
              "runtime": {"distribution_versions": {"gpuwm": "2.8.0"},
                          "interpreter": {"version": "same", "Py_GIL_DISABLED": 1,
                                          "gil_enabled": False}}}
    after = {**before, "runtime": {**before["runtime"], "interpreter": {
        **before["runtime"]["interpreter"], "gil_enabled": True}}}
    assert before != after
    assert instrument._protected_snapshot(before) == instrument._protected_snapshot(after)
    assert instrument._parallel_runtime_qualification(after["runtime"], required=True)["qualified"] is False
    assert instrument._parallel_runtime_qualification(after["runtime"], required=False)["qualified"] is True


@pytest.mark.parametrize("compiled,gil,qualified", [(0, True, False), (1, True, False), (1, False, True)])
def test_nrank_completion_requires_free_threaded_build_and_disabled_gil_after_cli(
    tmp_path, monkeypatch, compiled, gil, qualified,
):
    import json
    from hexcore import cli, forecast_door as door
    instrument = _instrument()
    output = tmp_path / "forecast"
    output.mkdir()
    parsed = SimpleNamespace(out=output, gpuwm_checkout=None, convection="auto", samehost_devices="0,1")
    monkeypatch.setattr(cli, "build_parser", lambda: SimpleNamespace(parse_args=lambda argv: parsed))
    monkeypatch.setattr(instrument, "source_snapshot", lambda: {"unchanged": True})
    monkeypatch.setattr(instrument, "runtime_snapshot", lambda checkout: {
        "interpreter": {"version": "test", "Py_GIL_DISABLED": compiled, "gil_enabled": gil}})
    final = {name: {"dtype": "<f4", "shape": [1, 2], "sha256": "a" * 64}
             for name in ("rthdynten", "rqvdynten")}

    def forecast(argv):
        (output / door.DRIVER_RECEIPT_NAME).write_text(json.dumps({
            "status": "passed", "forecast": {"steps_executed": 1, "steps_requested": 1,
                                               "final_gf_tendency_hashes": final}}))
        return 0

    monkeypatch.setattr(cli, "main", forecast)
    arguments = SimpleNamespace(devices=None, fingerprint_every=1,
                                report=tmp_path / "measurement.json", tip="test-tip",
                                measurement_mint=False, dt=120., card=None, profile_dir=None)
    assert instrument.run(arguments, []) == (0 if qualified else 2)
    receipt = json.loads(arguments.report.read_text())
    assert receipt["driver_completed"] is True
    assert receipt["run_identity_eligible"] is qualified
    assert receipt["parallel_runtime_qualification"]["qualified"] is qualified
    assert receipt["final_gf_tendency_hashes"] == final
    assert receipt["ordinary_door_restored"] and receipt["protected_bytes_unchanged"]


def _completed_measurement():
    return {"schema": "mpas-port.cuda-v841-real-door-measurement/v1", "cli_exit_code": 0,
            "driver_completed": True, "run_identity_eligible": True,
            "protected_bytes_unchanged": True, "ordinary_door_restored": True, "exception": None}


def test_serial_policy_loader_imports_no_hex_readers(monkeypatch):
    import builtins
    instrument = _instrument()
    previous = builtins.__import__

    def no_hex_readers(name, *args, **kwargs):
        if name == "hexcore" or name.startswith("hexcore."):
            pytest.fail("serial policy loading imported a hex reader before installing the guard")
        return previous(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_hex_readers)
    assert callable(instrument._load_serial_io_policy())


@pytest.mark.parametrize("compiled,environment,option,expected", [
    (0, "0", None, False), (1, None, None, False),
    (1, "0", None, True), (1, None, 0, True), (1, "1", None, False),
])
def test_serial_policy_requires_explicit_no_gil_on_free_threaded_build(
    monkeypatch, compiled, environment, option, expected,
):
    instrument = _instrument()
    monkeypatch.setattr(instrument.sysconfig, "get_config_var", lambda key: compiled)
    if environment is None:
        monkeypatch.delenv("PYTHON_GIL", raising=False)
    else:
        monkeypatch.setenv("PYTHON_GIL", environment)
    monkeypatch.setattr(instrument.sys, "_xoptions", {} if option is None else {"gil": option})
    assert instrument._explicit_free_threaded_runtime() is expected


def test_serial_policy_wraps_native_artifact_and_records_restoration_before_primary_receipt(
    tmp_path, monkeypatch,
):
    import hashlib
    import json
    import netCDF4
    import numpy as np
    instrument = _instrument()
    policy = instrument._load_serial_io_policy()
    primary = tmp_path / "measurement.json"
    plain, guarded = tmp_path / "plain.nc", tmp_path / "guarded.nc"
    previous = netCDF4.Dataset
    events = []

    def write(path):
        with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
            dataset.createDimension("columns", 3)
            variable = dataset.createVariable("field", "f4", ("columns",))
            variable[:] = np.array([1., -0., 3.], np.float32)

    write(plain)

    @contextmanager
    def observed_policy():
        with policy() as guard:
            events.append("guard entered")
            yield guard
            assert not primary.exists(), "receipt was written before guard restoration"
        events.append("guard restored")

    def model(*args):
        assert events == ["guard entered"]
        assert netCDF4.Dataset is not previous
        write(guarded)
        events.append("model finished")
        return primary, _completed_measurement()

    monkeypatch.setattr(instrument, "_explicit_free_threaded_runtime", lambda: True)
    monkeypatch.setattr(instrument, "_load_serial_io_policy", lambda: observed_policy)
    monkeypatch.setattr(instrument, "_run", model)
    assert instrument.run(SimpleNamespace(), []) == 0
    assert events == ["guard entered", "model finished", "guard restored"]
    assert netCDF4.Dataset is previous
    assert hashlib.sha256(plain.read_bytes()).digest() == hashlib.sha256(guarded.read_bytes()).digest()
    receipt = json.loads(primary.read_text())
    assert receipt["serial_extension_io"]["enabled"]
    assert receipt["serial_extension_io"]["module_entrypoints_restored"]
    assert receipt["serial_extension_io"]["worker_refusals"] == 0
    assert receipt["serial_extension_io"]["coordinator_calls"]["netCDF4.Dataset"] == 1
    assert receipt["run_identity_eligible"] and receipt["ordinary_door_restored"]


def test_worker_extension_refusal_is_retained_and_disqualifies_completed_model(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import json
    import netCDF4
    instrument = _instrument()
    primary = tmp_path / "measurement.json"
    forbidden = tmp_path / "worker.nc"

    def model(*args):
        with ThreadPoolExecutor(max_workers=1) as workers:
            with pytest.raises(RuntimeError, match="forecast coordinator"):
                workers.submit(netCDF4.Dataset, forbidden, "w").result()
        assert not forbidden.exists()
        return primary, _completed_measurement()

    monkeypatch.setattr(instrument, "_explicit_free_threaded_runtime", lambda: True)
    monkeypatch.setattr(instrument, "_run", model)
    assert instrument.run(SimpleNamespace(), []) == 2
    receipt = json.loads(primary.read_text())
    assert receipt["driver_completed"] and receipt["ordinary_door_restored"]
    assert receipt["serial_extension_io"]["worker_refusals"] == 1
    assert receipt["serial_extension_io"]["module_entrypoints_restored"]
    assert receipt["serial_extension_io"]["qualified"] is False
    assert receipt["run_identity_eligible"] is False


@pytest.mark.parametrize("model_failed", [False, True])
def test_guard_cleanup_failure_writes_final_failure_receipt_without_posthoc_mutation(
    tmp_path, monkeypatch, model_failed,
):
    import json
    instrument = _instrument()
    primary = tmp_path / "measurement.json"
    guard = SimpleNamespace(receipt=lambda: {"module_entrypoints_restored": False, "worker_refusals": 0})

    @contextmanager
    def broken_policy():
        yield guard
        assert not primary.exists()
        raise RuntimeError("extension cleanup failed")

    def model(*args):
        receipt = _completed_measurement()
        if model_failed:
            receipt.update(driver_completed=False, run_identity_eligible=False,
                           exception={"type": "ModelRefusal", "message": "original model refusal"})
        return primary, receipt

    monkeypatch.setattr(instrument, "_explicit_free_threaded_runtime", lambda: True)
    monkeypatch.setattr(instrument, "_load_serial_io_policy", lambda: broken_policy)
    monkeypatch.setattr(instrument, "_run", model)
    assert instrument.run(SimpleNamespace(), []) == 2
    receipt = json.loads(primary.read_text())
    assert not receipt["ordinary_door_restored"] and not receipt["run_identity_eligible"]
    assert receipt["instrument_cleanup_exception"]["message"] == "extension cleanup failed"
    assert receipt["exception"]["type"] == ("ModelRefusal" if model_failed else "RuntimeError")


def test_regular_interpreter_does_not_load_extension_policy(tmp_path, monkeypatch):
    import json
    instrument = _instrument()
    primary = tmp_path / "measurement.json"
    monkeypatch.setattr(instrument, "_explicit_free_threaded_runtime", lambda: False)
    monkeypatch.setattr(instrument, "_load_serial_io_policy", lambda: pytest.fail("regular control imported policy"))
    monkeypatch.setattr(instrument, "_run", lambda *args: (primary, _completed_measurement()))
    assert instrument.run(SimpleNamespace(), []) == 0
    receipt = json.loads(primary.read_text())
    assert receipt["serial_extension_io"]["enabled"] is False
    assert receipt["serial_extension_io"]["qualified"] and receipt["run_identity_eligible"]
