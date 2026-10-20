"""One-off warm-stage attribution. No extra GPU synchronization; not throughput.

Usage: python profile_native_warm_stages.py --profile-receipt FILE forecast ...
Run under the normal GPU ownership protocol. The first model step is excluded.
All original operations execute unchanged; nested intervals are not additive.
"""
import collections
import contextlib
import cProfile
import datetime
import functools
import hashlib
import inspect
import json
from pathlib import Path
import pstats
import sys
import time

if len(sys.argv) < 4 or sys.argv[1] != '--profile-receipt':
    raise SystemExit(__doc__)
destination = Path(sys.argv[2]).resolve()
del sys.argv[1:3]
from hexcore.mpasphys import runtime_v842 as runtime
from hexcore.mpasphys import native_backend
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
from hexcore.cli import main

context = {'step': 0, 'stage': None}
profiles = collections.defaultdict(cProfile.Profile)
operations = collections.defaultdict(lambda: dict(calls=0, logical_bytes=0, wall_seconds=0., thread_cpu_seconds=0.))
stages = []
original_stage = runtime.timed_stage

@contextlib.contextmanager
def measured_stage(rows, name, **kwargs):
    if name == 'outer-physics':
        context['step'] += 1
    active = context['step'] >= 2 and name in ('outer-physics', 'dycore-and-post-transport', 'publish-and-streams')
    context['stage'] = name if active else None
    with original_stage(rows, name, **kwargs) as timing:
        if not active:
            yield timing
        else:
            start = datetime.datetime.now(datetime.timezone.utc).isoformat()
            wall, cpu = time.monotonic(), time.thread_time()
            profiler = profiles[name]
            profiler.enable()
            try:
                yield timing
            finally:
                profiler.disable()
                stages.append(dict(step=context['step'], stage=name, started_utc=start,
                    ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    wall_seconds=time.monotonic()-wall, thread_cpu_seconds=time.thread_time()-cpu,
                    parallelism='One granted GPU and main host thread; native calls serialized; diagnostic overhead included'))
    context['stage'] = None

runtime.timed_stage = measured_stage

def observe(cls, method):
    original = getattr(cls, method)
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if context['stage'] is None:
            return original(self, *args, **kwargs)
        wall, cpu = time.monotonic(), time.thread_time()
        result = original(self, *args, **kwargs)
        key = (context['stage'], method)
        row = operations[key]
        row['calls'] += 1
        row['wall_seconds'] += time.monotonic()-wall
        row['thread_cpu_seconds'] += time.thread_time()-cpu
        value = result[0] if method == 'get_variable' and result is not None else result
        if method == 'put_field': value = args[2]
        if method == 'put_variable': value = args[1]
        row['logical_bytes'] += int(getattr(value, 'nbytes', 0))
        return result
    setattr(cls, method, wrapped)

# Class is resolved explicitly below; no native ABI or metadata calls are parallelized.
for method in ('get_field', 'put_field', 'get_variable', 'put_variable'):
    observe(native_backend.NativePhysics, method)
observe(NativeDevicePoolViews, 'commit')
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
tick = time.monotonic()
try:
    raise SystemExit(main(sys.argv[1:]))
finally:
    parsed = {}
    for phase, profiler in profiles.items():
        profiler.disable()
        rows = []
        for (file, line, function), (_, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
            rows.append(dict(file=file, line=line, function=function, calls=calls,
                             self_seconds=own, cumulative_seconds=cumulative))
        parsed[phase] = sorted(rows, key=lambda r: -r['self_seconds'])
    destination.write_text(json.dumps(dict(started_utc=started,
        ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        whole_diagnostic_wall_seconds=time.monotonic()-tick,
        profile_only=True, added_gpu_synchronizations=0, cuda_graphs=False,
        parallelism='One granted GPU plus serialized native host calls; CPU cProfile on main thread',
        excluded_steps=[1], stage_observations=stages, profiles=parsed,
        native_operations=[dict(stage=k[0], operation=k[1], **v) for k,v in operations.items()],
        sources={str(Path(inspect.getsourcefile(x)).name):hashlib.sha256(Path(inspect.getsourcefile(x)).read_bytes()).hexdigest()
                 for x in (runtime, native_backend, NativeDevicePoolViews)},
        scope='Nested wall/CPU intervals overlap; blocking gets include prior GPU work and CPU spin-wait. Logical bytes are native array payloads, not PCIe counters. First-step setup excluded; profiling overhead remains.'), indent=2)+'\n')
