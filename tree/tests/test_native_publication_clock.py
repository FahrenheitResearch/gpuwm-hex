"""Real MPAS alarms at fractional timesteps; no GPU or forecast-skill claim."""
import os
from types import SimpleNamespace

import pytest

from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.runtime_v842 import native_publication_required
from test_mpas_native_backend import prepare_stream_block


@pytest.mark.parametrize('steps', [64, 65])
def test_fractional_native_publication_preserves_stream_alarm_schedule(tmp_path, steps):
    library = os.environ.get('MPAS_NATIVE_LIBRARY')
    if not library:
        pytest.skip('requires the actual strict MPAS shared library')
    streams = [SimpleNamespace(name=name, directions=('output',),
                               output_interval='00:00:30')
               for name in ('restart', 'output', 'diagnostics', 'surface')]
    schedules = []
    boundaries = []
    for lazy in (False, True):
        written = []
        published = []
        with NativePhysics(library, test_only=True) as native:
            prepare_stream_block(native, [1, 2, 3, 4], 4, clock_options=dict(
                start='2000-01-01_00:00:00', stop='2000-01-01_00:02:00', dt=0.9375))
            for stream in streams:
                native.stream(stream.name, filename=str(tmp_path / (stream.name+'.nc')),
                              direction='output', output_interval=stream.output_interval)
            # A fresh forecast does not write a restart at initialization.
            native.reset_alarm('restart:out')
            for step in range(steps+1):
                if step:
                    native.advance_clock()
                before = [native.alarm_ringing(s.name+':out') for s in streams]
                required = native_publication_required(native, streams, step=step,
                                                        steps=steps, moving=None, lazy=lazy)
                # Policy queries must not consume the native alarm.
                assert before == [native.alarm_ringing(s.name+':out') for s in streams]
                if required:
                    published.append(step)
                    for stream, ringing in zip(streams, before):
                        if ringing:
                            written.append((stream.name, step))
                            native.reset_alarm(stream.name+':out')
        schedules.append(written)
        boundaries.append(published)
    assert schedules[0] == schedules[1]
    assert schedules[1] == [(s.name, step) for step in (0, 32, 64)
                            for s in streams if step or s.name != 'restart']
    assert boundaries[0] == list(range(steps+1))
    assert boundaries[1] == ([0, 32, 64] if steps == 64 else [0, 32, 64, 65])
