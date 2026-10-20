"""Read-only chronology witness for the copied IAU integration diagnostic.

Records actual native alarm calls and driver/density ordering. No observed value
is supplied to scientific execution. The separate native boundary probe remains
the authority for bitwise comparisons, not this chronology.
"""
import os

if os.environ.get("HEXCORE_IAU_SOURCE_SEEK_PROBE_OUT"):
    import atexit
    import hashlib
    import json
    from pathlib import Path
    import runpy

    if os.environ.get("HEXCORE_IAU_BOUNDARY_PROBE"):
        runpy.run_path(str(Path(__file__).parents[1] / "forecast_probe" / "sitecustomize.py"))

    from hexcore.cuda_horizontal import CudaHorizontal
    from hexcore.mpasphys.driver_v842 import PhysicsDriver
    from hexcore.mpasphys.native_backend import NativePhysics

    events = []
    active_step = None
    density_since_iau = 0
    original_reset = NativePhysics.reset_alarm
    original_add = PhysicsDriver.add_iau_tendencies
    original_density = CudaHorizontal.density_tendency
    original_finish = PhysicsDriver.finish_step

    def reset(self, name):
        if name != "iau:in":
            return original_reset(self, name)
        before = bool(self.alarm_ringing(name))
        result = original_reset(self, name)
        events.append(dict(operation="native_alarm_reset", name=name,
                           before=before, after=bool(self.alarm_ringing(name))))
        return result

    def add(self, **kwargs):
        global active_step, density_since_iau
        step, time, _, _ = self._pending
        result = original_add(self, **kwargs)
        active_step, density_since_iau = step, 0
        events.append(dict(operation="iau_add", step=step, time_seconds=time))
        return result

    def density(self, *args, **kwargs):
        global density_since_iau
        events.append(dict(operation="density_tendency", step=active_step,
                           density_call=density_since_iau))
        density_since_iau += 1
        return original_density(self, *args, **kwargs)

    def finish(self, **kwargs):
        global active_step
        step = self._pending[0]
        result = original_finish(self, **kwargs)
        events.append(dict(operation="finish_step", step=step,
                           density_calls=density_since_iau))
        active_step = None
        return result

    NativePhysics.reset_alarm = reset
    PhysicsDriver.add_iau_tendencies = add
    CudaHorizontal.density_tendency = density
    PhysicsDriver.finish_step = finish

    def write_witness():
        pool = Path(os.environ["HEXCORE_IAU_SOURCE_SEEK_POOL"])
        records = {}
        for name in ("request.json", "manifest.json"):
            path = pool / name
            if path.is_file():
                data = json.loads(path.read_text())
                records[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    source_records=data.get("source_records"),
                                    source_seek_witness=data.get("source_seek_witness"))
        result = dict(schema="iau-source-seek-chronology-diagnostic-v1",
                      acceptance=False, events=events, transport=records,
                      observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        Path(os.environ["HEXCORE_IAU_SOURCE_SEEK_PROBE_OUT"]).write_text(
            json.dumps(result, indent=2) + "\n")

    atexit.register(write_witness)
