#!/usr/bin/env python3
"""Allocation/stage peak census over the public forecast command.

Usage: python peak_stages.py RECEIPT.json -- forecast ...
Run under a GPU lease. Observation only: no allocator flushes, altered chunk
widths, cadence changes or admission overrides. This is not a speed benchmark.
"""
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
from pathlib import Path
import sys
import threading
import time


def main():
    receipt = Path(sys.argv[1])
    assert sys.argv[2] == '--'
    from hexcore import forecast_door
    from hexcore.cli import main as cli
    import cupy as cp

    state = {'step': 0}
    current = threading.local()
    peaks = {}
    allocations = {}
    start = time.time()
    pool = cp.get_default_memory_pool()
    initial_free, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(0)
    max_threads = cp.cuda.runtime.deviceGetAttribute(
        cp.cuda.runtime.cudaDevAttrMaxThreadsPerMultiProcessor, 0)

    def checkpoint(rc=None):
        result = dict(schema='hex-stage-peaks.v1', argv=sys.argv[3:], rc=rc,
            started_unix=start, wall_seconds=time.time() - start,
            baseline_free_bytes=initial_free, device_total_bytes=total,
            device_name=props['name'].decode(), sms=props['multiProcessorCount'],
            max_threads_per_sm=max_threads,
            final_step=state['step'], peaks=peaks, allocations=allocations,
            instrument='CuPy allocation hooks and stage boundaries; no flush; timing instrumented')
        pending = receipt.with_suffix('.pending')
        pending.write_text(json.dumps(result, indent=2) + '\n')
        pending.replace(receipt)

    def sample():
        free, _ = cp.cuda.runtime.memGetInfo()
        used, held = int(pool.used_bytes()), int(pool.total_bytes())
        row = dict(step=state['step'], device_used_bytes=total - free,
                   incremental_device_bytes=initial_free - free,
                   pool_used_bytes=used, pool_total_bytes=held,
                   pool_cached_bytes=held - used,
                   outside_pool_bytes=total - free - held)
        stage = getattr(current, 'stage', 'cuda_context')
        peak = peaks.setdefault(stage, {'samples': 0})
        peak['samples'] += 1
        for metric in ('device_used_bytes', 'pool_used_bytes', 'pool_total_bytes',
                       'pool_cached_bytes', 'outside_pool_bytes'):
            if metric not in peak or row[metric] > peak[metric][metric]:
                peak[metric] = row.copy()

    @contextmanager
    def stage(label):
        previous = getattr(current, 'stage', 'cuda_context')
        current.stage = label
        # CuPy hooks are thread local. Same-host ranks construct and step on
        # worker threads, so the CLI thread's hook cannot see their temporaries.
        active = getattr(current, 'hook_active', False)
        current.hook_active = True
        try:
            with nullcontext() if active else PeakHook():
                sample()
                try:
                    yield
                finally:
                    sample()
                    if label in ('construct', 'step', 'history.write'):
                        checkpoint()
        finally:
            current.stage = previous
            current.hook_active = active

    def wrap(obj, name, label, after=None):
        original = getattr(obj, name)
        if getattr(original, '_peak_stages_wrapped', False):
            return

        @wraps(original)
        def observed(*a, **kw):
            if label == 'step':
                state['step'] += 1
            with stage(label):
                result = original(*a, **kw)
                if after is not None:
                    after(result)
                return result
        observed._peak_stages_wrapped = True
        setattr(obj, name, observed)

    class PeakHook(cp.cuda.MemoryHook):
        name = 'hex-stage-peaks'

        def alloc_postprocess(self, mem_size, mem_ptr, **_):
            # Real cudaMalloc, before a temporary can disappear between calls.
            label = getattr(current, 'stage', 'cuda_context')
            row = allocations.setdefault(label, {})
            if mem_ptr:
                row['new_blocks'] = row.get('new_blocks', 0) + 1
                row['new_bytes'] = row.get('new_bytes', 0) + int(mem_size)
                row['largest_new_block_bytes'] = max(row.get('largest_new_block_bytes', 0), int(mem_size))
            sample()

        def malloc_postprocess(self, size, **_):
            # Pool reuse can raise live bytes without another cudaMalloc.
            label = getattr(current, 'stage', 'cuda_context')
            row = allocations.setdefault(label, {})
            row['largest_pool_request_bytes'] = max(row.get('largest_pool_request_bytes', 0), int(size))
            peak = peaks.get(label, {})
            if (int(pool.used_bytes()) > peak.get('pool_used_bytes', {}).get('pool_used_bytes', -1)
                    or int(pool.total_bytes()) > peak.get('pool_total_bytes', {}).get('pool_total_bytes', -1)):
                sample()

    def attach_physics(stack):
        driver = stack['backend']._seam._driver
        for name in dir(type(driver)):
            if name.startswith('_run_') and callable(getattr(type(driver), name)):
                wrap(type(driver), name, 'physics.' + name)

    load = forecast_door._load_drivers

    def load_observed(*a, **kw):
        binding, driver = load(*a, **kw)
        proof = driver.proof
        wrap(proof, '_construct_device_stack', 'construct', attach_physics)
        wrap(proof, 'execute_composite_step', 'step')
        wrap(proof, 'capture_snapshot', 'history.capture')
        wrap(proof, 'write_snapshot_netcdf', 'history.write')
        from hexcore.drivers import samehost_forecast_v841
        wrap(samehost_forecast_v841, '_capture_rank', 'history.capture')
        from hexcore.cuda_horizontal_v841 import CudaHorizontalV841
        wrap(CudaHorizontalV841, 'solve_diagnostics', 'dynamics.solve_diagnostics')
        return binding, driver

    forecast_door._load_drivers = load_observed
    rc = 1
    try:
        with PeakHook():
            current.hook_active = True
            rc = cli(sys.argv[3:])
        return rc
    finally:
        checkpoint(rc)


if __name__ == '__main__':
    raise SystemExit(main())
