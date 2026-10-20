"""CPU admission and exact macro clocks. This module does not advance weather."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from .config_fullstep_lts import FullStepLtsConfig
from .fullstep_lts_classes import planes_per_step
from .mpas_defaults import DEFAULT_ACOUSTIC_SUBSTEPS


def geometric_rates(dc_edge, cells_on_edge, n_cells, max_rate=16):
    dc, coe = np.asarray(dc_edge, dtype=np.float64), np.asarray(cells_on_edge)
    if coe.shape != (dc.size, 2) or coe.dtype.kind not in 'iu' or np.any((coe < 0) | (coe >= n_cells)) or not np.all(np.isfinite(dc) & (dc > 0)):
        raise ValueError('invalid closed-mesh edge geometry')
    spacing = np.full(n_cells, np.inf)
    np.minimum.at(spacing, coe[:, 0], dc)
    np.minimum.at(spacing, coe[:, 1], dc)
    if not np.all(np.isfinite(spacing)):
        raise ValueError('a cell has no incident edge')
    rates = np.ones(n_cells, dtype=np.int32)
    for rate in (2, 4, 8, 16, 32):
        if rate <= max_rate:
            rates[spacing >= spacing.min()*rate] = rate
    return rates


def buffer_rates(rates, cells_on_edge):
    result = np.array(rates, dtype=np.int32, copy=True)
    coe = np.asarray(cells_on_edge)
    for _ in range(2):
        previous = result.copy()
        np.minimum.at(result, coe[:, 0], previous[coe[:, 1]])
        np.minimum.at(result, coe[:, 1], previous[coe[:, 0]])
    return result


def classify_fullstep_rates(geometry, budgets, previous=None, *, dt, policy=FullStepLtsConfig(),
                            acoustic_substeps=DEFAULT_ACOUSTIC_SUBSTEPS):
    """Complete rates, with immediate demotion and two-macro promotion memory.

    Budgets contain per-column s^-1 rates for transport, advection, acoustic,
    mixing, and physics (inverse maximum physics duration). All are required.
    Previous is (unbuffered rates, prior rate samples, headroom streak). Caller retains
    the returned history; buffering is recomputed from raw ceilings each time.

    The acoustic budget is held against one acoustic sub-step, ``dt / (3 N)``
    with three dynamics splits and N = ``acoustic_substeps`` (default: the
    MPAS-A v8.4.2 Registry default, 4; the same even-N rule as the active
    class clock, ``fullstep_lts_classes.planes_per_step``).
    """
    planes = planes_per_step(acoustic_substeps)
    ceiling = geometric_rates(**geometry, max_rate=policy.fullstep_lts_max_rate)
    allowed = ceiling.copy()
    limits = dict(transport=policy.fullstep_lts_target_transport,
                  advection=policy.fullstep_lts_target_advection,
                  acoustic=planes*policy.fullstep_lts_target_acoustic,
                  mixing=policy.fullstep_lts_target_mixing, physics=1.)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('base duration must be finite and positive')
    samples = {}
    margin = np.ones(allowed.size, dtype=bool)
    for name, target in limits.items():
        sample = np.asarray(budgets[name], dtype=np.float64)
        if sample.shape != allowed.shape or not np.all(np.isfinite(sample) & (sample >= 0)):
            raise ValueError(f'invalid per-column {name} rate')
        samples[name] = sample.copy()
        bound = sample if previous is None else np.maximum(sample, previous[1][name])
        if np.any(dt*bound > target):
            cell = int(np.flatnonzero(dt*bound > target)[0])
            raise ValueError(f'cell={cell} {name}={bound[cell]:.17g} required_dt={target/bound[cell]:.17g}: the base local step exceeds the donor/acoustic/mixing bound; promoting the limiter is not a valid timestep admission')
        while np.any(dt*allowed*bound > target):
            bad = dt*allowed*bound > target
            allowed[bad] //= 2
        if previous is not None:
            margin &= dt*np.minimum(ceiling, previous[0]*2)*bound <= .8*target
    streak = np.zeros_like(allowed) if previous is None else np.where(margin, previous[2]+1, 0)
    if previous is not None:
        allowed = np.minimum(allowed, np.where(streak >= 2, previous[0]*2, previous[0]))
    executed = buffer_rates(allowed, geometry['cells_on_edge'])
    return executed, (allowed.copy(), samples, streak)


def clock_ticks(seconds, *, base=False):
    value = Fraction(str(seconds))*48
    if value.denominator != 1 or (base and value.numerator % 6):
        raise ValueError('base durations require eighth seconds and internal clocks forty-eighth seconds')
    return value.numerator


@dataclass(frozen=True)
class MacroPlan:
    start_tick: int
    end_tick: int
    base_ticks: int
    effective_rates: tuple[int, ...]
    intervals: tuple[tuple[int, int, int], ...]


def build_macro_plan(clock, rates, events, *, base_dt):
    base = clock_ticks(base_dt, base=True)
    start = clock_ticks(clock)
    if base <= 0 or start < 0 or start % base:
        raise ValueError('macro start must be on the positive base clock')
    event_ticks = [clock_ticks(event) for event in events]
    if any(t % base for t in event_ticks):
        raise ValueError('rounding this event would shift radiation/output relative to one or more classes')
    next_event = min((t for t in event_ticks if t > start), default=None)
    if next_event is None:
        raise ValueError('a future synchronization event is required')
    rate_values = tuple(sorted(set(int(r) for r in rates)))
    if not rate_values or any(r not in (1, 2, 4, 8, 16) for r in rate_values):
        raise ValueError('invalid dyadic rate map')
    macro_rate = max(r for r in (1, 2, 4, 8, 16) if r <= max(rate_values) and start+r*base <= next_event)
    effective = tuple(sorted(set(min(r, macro_rate) for r in rate_values)))
    trace = []
    def visit(rate, a, b):
        if rate in effective:
            trace.append((rate, a, b))
        if rate > 1:
            mid = (a+b)//2
            visit(rate//2, a, mid)
            visit(rate//2, mid, b)
    end = start+macro_rate*base
    visit(macro_rate, start, end)
    return MacroPlan(start, end, base, effective, tuple(trace))


def transport_ceiling(geometric, transport_rate, dt, target=.90):
    """Optimistic ceiling only. Missing other budgets cannot increase it."""
    geometric = np.asarray(geometric)
    rate = np.asarray(transport_rate, dtype=np.float64)
    if rate.shape != geometric.shape or not np.all(np.isfinite(rate) & (rate >= 0)):
        raise ValueError('invalid transport rates')
    result = np.ones(geometric.size, dtype=np.int32)
    for r in (1, 2, 4, 8, 16):
        result[(geometric >= r) & (dt*r*rate <= target)] = r
    infeasible = dt*rate > target
    return result, infeasible
