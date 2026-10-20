"""Bounded, nonblocking GPU timing of stochastic forcing inside model steps.

Only event metadata crosses the CPU. No field arithmetic, stream drain or
CUDA graph is used. Cold and unmatched samples cannot pass the cost bar.
"""
from contextlib import contextmanager, nullcontext
from functools import wraps
import math
import time


def summarize_step_cost(samples, *, minimum_warm_steps=3):
    """Grade pattern cost against the complete step on the same GPU stream."""
    if type(minimum_warm_steps) is not int or minimum_warm_steps < 3:
        raise ValueError('stochastic cost grading requires at least three warm steps')
    warm = []
    for sample in samples:
        if (not sample.get('complete') or sample.get('failed') or sample.get('cold')
                or not sample.get('same_stream') or not sample.get('forcing_calls')):
            continue
        step, pattern, forcing = (sample[key] for key in
                                  ('step_ms', 'pattern_ms', 'forcing_total_ms'))
        if (any(not math.isfinite(x) or x < 0 for x in (step, pattern, forcing))
                or step <= 0 or pattern > forcing+1e-3 or forcing > step+1e-3):
            raise ValueError('stochastic GPU timings are not nested finite durations')
        warm.append(sample)
    enough = len(warm) >= minimum_warm_steps
    return dict(warm_steps=len(warm), minimum_warm_steps=minimum_warm_steps,
        pattern_fraction=(sum(s['pattern_ms'] for s in warm)/sum(s['step_ms'] for s in warm)
                          if warm else None),
        forcing_fraction=(sum(s['forcing_total_ms'] for s in warm)/sum(s['step_ms'] for s in warm)
                          if warm else None),
        pattern_cost_passed=(all(s['pattern_ms']/s['step_ms'] < .05 for s in warm)
                             if enough else None),
        policy='pattern GPU elapsed <5% of complete-step stream elapsed on every matched warm sample; at least three samples',
        qualification_complete=False)


class StepTimings:
    """Keep the first four steps; report completed events without waiting."""
    def __init__(self, cp):
        self.cp = cp
        self.entries = []
        self.active = None

    @contextmanager
    def measure(self, route):
        if self.active is not None:
            raise RuntimeError('nested stochastic step timing')
        if len(self.entries) >= 4:
            yield
            return
        cp = self.cp
        start, end = cp.cuda.Event(), cp.cuda.Event()
        entry = dict(route=route, start=start, end=end, cold=not self.entries, forcing=[],
                     same_stream=True, stream=cp.cuda.get_current_stream().ptr,
                     failed=False)
        self.active = entry
        wall = time.perf_counter()
        try:
            start.record()
            yield
            entry['same_stream'] &= entry['stream'] == cp.cuda.get_current_stream().ptr
            end.record()
        except BaseException:
            entry['failed'] = True
            raise
        finally:
            entry['enqueue_wall_seconds'] = time.perf_counter()-wall
            self.entries.append(entry)
            self.active = None

    def forcing(self, events, *, cold):
        if self.active is not None:
            self.active['cold'] |= cold
            self.active['same_stream'] &= (
                self.active['stream'] == self.cp.cuda.get_current_stream().ptr)
            self.active['forcing'].append(events)

    def receipt(self):
        cp = self.cp
        samples = []
        for index, entry in enumerate(self.entries):
            sample = dict(sample=index, route=entry['route'], failed=entry['failed'],
                          cold=entry['cold'], same_stream=entry['same_stream'],
                          forcing_calls=len(entry['forcing']),
                          enqueue_wall_seconds=entry['enqueue_wall_seconds'], complete=False)
            if (not entry['failed'] and entry['end'].done
                    and all(events[2].done for events in entry['forcing'])):
                sample.update(complete=True,
                    step_ms=cp.cuda.get_elapsed_time(entry['start'],entry['end']),
                    pattern_ms=sum(cp.cuda.get_elapsed_time(a,b) for a,b,c in entry['forcing']),
                    forcing_total_ms=sum(cp.cuda.get_elapsed_time(a,c) for a,b,c in entry['forcing']))
            samples.append(sample)
        return dict(samples=samples, **summarize_step_cost(samples),
            scope='model compute step, excludes output and checkpoints; same-stream elapsed includes CPU dispatch gaps and waits',
            parallelism='GPU CUDA events; no synchronization added')


def measure_stochastic_step(owner, route):
    stochastic = getattr(owner, 'stochastic', None)
    return (stochastic.step_timing.measure(route) if stochastic is not None
            else nullcontext())


def timed_arwen_step(function):
    """Shared composite entry covers ordinary, resident and one-rank doors."""
    @wraps(function)
    def measured(*args, **kwargs):
        with measure_stochastic_step(kwargs['driver'], 'arwen-composite'):
            return function(*args, **kwargs)
    return measured
