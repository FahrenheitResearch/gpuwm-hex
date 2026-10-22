"""GPU stochastic physics on spherical Voronoi meshes.

WRF v4.6.1 module_stoch.F lineage: spectral AR(1), clipped SPPT and
streamfunction SKEBS. The transform here uses real spherical harmonics,
not the WRF rectangular FFT. No native bitwise or skill claim is implied.
Philox follows the engine's ensemble_stochastic.cu, under the Random123
BSD notice in licenses/LICENSE-Random123.txt. All field arithmetic is CUDA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import threading
import time
import weakref

import numpy as np

RESOURCE_LOCK = threading.Lock()
MESH_RESOURCES = weakref.WeakValueDictionary()
SCHEMA = 'gpuwm-hex.stochastic/v2'


def backscatter_rate_scale(dt, time_s, interval_s=0.0):
    """Rate variance per unit energy for a correlated, held AR(1) pattern.

    With update period h and phi=exp(-h/tau), the integrated autocovariance
    is h/2 * (1+phi)/(1-phi). Thus sigma**2 = 2*D/h*tanh(h/(2*tau)).
    Using the white-noise 2*D/dt with a six-hour pattern at dt=600 s
    overstates its long-time energy input by about 72. Spatial clipping and
    normalization make this a nominal budget, not an exact net KE budget.
    """
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(time_s) or time_s <= 0:
        raise ValueError('backscatter dt and time_s must be positive and finite')
    if not math.isfinite(interval_s) or interval_s < 0:
        raise ValueError('backscatter interval_s must be nonnegative and finite')
    period = max(dt, interval_s)
    return 2 * math.tanh(period / (2*time_s)) / period


@dataclass(frozen=True)
class StochasticConfig:
    sppt: bool = False
    skeb: bool = False
    member: int = 0
    seed: int = 53
    degree: int = 48
    length_m: float = 500000.0
    time_s: float = 21600.0
    amplitude: float = 0.5
    clip: float = 1.0
    skeb_fraction: float = 0.1
    thermal_fraction: float = 0.01
    surface_m: float = 500.0
    top_start_m: float = 18000.0
    top_end_m: float = 25000.0
    # Each optional scale is (amplitude, length metres, time seconds, seed).
    sppt_scales: tuple = ()
    logit: bool = False
    surface_limit: bool = True
    interval_s: float = 0.0
    restart_required: bool = False
    surface_profile: tuple = ()

    def __post_init__(self):
        for name in ('sppt', 'skeb', 'logit', 'surface_limit', 'restart_required'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be logical')
        for name in ('member', 'seed'):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**64:
                raise ValueError(f'{name} must be an unsigned 64-bit integer')
        if type(self.degree) is not int or not 1 <= self.degree <= 512:
            raise ValueError('degree must be 1..512 for the normalized harmonic recurrence')
        for name in ('length_m', 'time_s', 'amplitude', 'surface_m', 'top_end_m'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('skeb_fraction', 'thermal_fraction', 'top_start_m'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative and finite')
        if not math.isfinite(self.clip) or not 0 < self.clip <= 1:
            raise ValueError('clip must be in (0,1] so SPPT cannot reverse a tendency')
        if self.top_end_m <= self.top_start_m:
            raise ValueError('top_end_m must exceed top_start_m')
        if max(self.skeb_fraction, self.thermal_fraction) > 1:
            raise ValueError('backscatter fractions must not exceed the dissipated energy')
        if not math.isfinite(self.interval_s) or self.interval_s < 0:
            raise ValueError('interval_s must be nonnegative and finite')
        scales = tuple(tuple(scale) for scale in self.sppt_scales)
        if len(scales) > 5:
            raise ValueError('SPPT supports at most five independent spectral scales')
        for scale in scales:
            if len(scale) != 4:
                raise ValueError('SPPT scale needs amplitude, length_m, time_s, seed')
            amp, length, tau, seed = scale
            if not all(math.isfinite(v) and v > 0 for v in (amp, length, tau)):
                raise ValueError('SPPT scale amplitude, length and time must be positive and finite')
            if type(seed) is not int or not 0 <= seed < 2**64:
                raise ValueError('SPPT scale seed must be an unsigned 64-bit integer')
        object.__setattr__(self, 'sppt_scales', scales)
        profile = tuple(self.surface_profile)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in profile):
            raise ValueError('surface_profile weights must be finite in [0,1]')
        object.__setattr__(self, 'surface_profile', profile)

    @property
    def enabled(self):
        return self.sppt or self.skeb

    @property
    def member_seed(self):
        payload = f'{SCHEMA}:{self.member}:{self.seed}'.encode('ascii')
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')

    @classmethod
    def parse(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        options = json.loads(value) if isinstance(value, str) else value
        if not isinstance(options, Mapping):
            raise ValueError('stochastic configuration must be a JSON object')
        try:
            return cls(**options)
        except TypeError as error:
            raise ValueError(f'invalid stochastic configuration: {error}') from error

    def json(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))

    def state_json(self):
        controls = asdict(self)
        controls.pop('restart_required')
        return json.dumps(controls, sort_keys=True, separators=(',', ':'))


def member_options(member, *, seed=53, source_members=1, ensemble_size=1, **options):
    """Default ON for ensembles with fewer than ten independent source members.

    Count source members before co-resident grouping or perturbation fill.
    Explicit per-member booleans override the default in either direction.
    """
    if type(source_members) is not int or source_members < 1:
        raise ValueError('source_members must be positive')
    if type(ensemble_size) is not int or ensemble_size < 1:
        raise ValueError('ensemble_size must be positive')
    on = ensemble_size > 1 and source_members < 10
    return asdict(StochasticConfig(**{**dict(sppt=on, skeb=on, member=member, seed=seed), **options}))


def member_number(identity):
    """Stable identity for named rosters, independent of selection and grouping."""
    if type(identity) is int:
        return identity
    return int.from_bytes(hashlib.sha256(str(identity).encode('utf-8')).digest()[:8], 'little')


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    source = Path(__file__).with_name('stochastic.cu').read_text()
    return cp.RawModule(code=source, options=('--std=c++17',))


def _kernel(name):
    import cupy as cp
    return _module(cp.cuda.Device().id).get_function(name)


def _field_strides(field, levels, columns, label):
    """Validate raw-kernel fields using metadata only; return strides in words."""
    if (field.dtype != np.float32 or field.ndim != 2
            or field.shape[0] != levels or field.shape[1] < columns
            or any(stride % 4 for stride in field.strides)):
        raise ValueError(f'stochastic {label} requires float32[level,{columns} or more]')
    return tuple(np.int64(stride//4) for stride in field.strides)


class MeshHarmonics:
    """Immutable mesh resource, built on GPU and shared across member streams."""
    def __init__(self, mesh, degree):
        import cupy as cp
        started = time.perf_counter()
        self.degree = degree
        self.cells = mesh.n_cells
        self.edges = mesh.n_edges
        self.radius = float(mesh.attrs.get('sphere_radius', 6371220.0))
        a = mesh.arrays
        self.build_started = cp.cuda.Event()
        self.build_started.record()
        # Validate topology on the GPU once, never on one CPU or in a step.
        topology = {name: cp.ascontiguousarray(cp.asarray(a[name], dtype=cp.int32))
                    for name in ('cellsOnEdge', 'verticesOnEdge', 'cellsOnVertex')}
        for name, indices in topology.items():
            if bool(cp.any(indices < 0)):
                raise ValueError(f'stochastic global mesh has a missing neighbour in {name}; '
                    'partition halo and regional boundary stochastic forcing is not implemented')
        lat = cp.concatenate((cp.asarray(a['latCell'], dtype=cp.float64),
                              cp.asarray(a['latVertex'], dtype=cp.float64)))
        lon = cp.concatenate((cp.asarray(a['lonCell'], dtype=cp.float64),
                              cp.asarray(a['lonVertex'], dtype=cp.float64)))
        self.modes = (degree + 1)**2 - 1
        self.basis = cp.empty((lat.size, self.modes), dtype=cp.float32)
        count = int(lat.size) * (degree + 1)
        _kernel('harmonics')(((count+127)//128,), (128,),
            (lat, lon, self.basis, np.int32(lat.size), np.int32(degree)))
        # All mode metadata and spectral normalization remain on the GPU.
        index = cp.arange(1, self.modes+1, dtype=cp.float64)
        self.ell = cp.floor(cp.sqrt(index))
        self.area = cp.asarray(a['areaCell'], dtype=cp.float32)
        self.area = self.area / cp.sum(self.area)
        self.coe = topology['cellsOnEdge']
        self.voe = topology['verticesOnEdge']
        self.cov = topology['cellsOnVertex']
        self.dv = cp.asarray(a['dvEdge'], dtype=cp.float32)
        # Orient the dual tangent against r cross the primal edge normal.
        def xyz(kind):
            return cp.stack([cp.asarray(a[c+kind], dtype=cp.float64) for c in ('x','y','z')], axis=1)
        cells, vertices = xyz('Cell'), xyz('Vertex')
        normal = cells[self.coe[:,1]] - cells[self.coe[:,0]]
        radial = cells[self.coe[:,1]] + cells[self.coe[:,0]]
        tangent = vertices[self.voe[:,1]] - vertices[self.voe[:,0]]
        self.orientation = cp.sign(cp.sum(cp.cross(radial, normal)*tangent, axis=1)).astype(cp.float32)
        self.edge_weight = cp.asarray(a['dcEdge'], dtype=cp.float32) * self.dv
        self.edge_weight /= cp.sum(self.edge_weight)
        self.ready = cp.cuda.Event()
        self.ready.record()
        self.receipt = dict(parallelism='GPU', block_threads=128,
                            basis_bytes=int(self.basis.nbytes), degree=degree,
                            construction_enqueue_seconds=time.perf_counter()-started,
                            timing_note='enqueue wall, not GPU elapsed time')

    @classmethod
    def cached(cls, mesh, degree):
        import cupy as cp
        key = (cp.cuda.Device().id, degree)
        with RESOURCE_LOCK:
            resources = getattr(mesh, '_stochastic_harmonics', None)
            if resources is None:
                resources = mesh._stochastic_harmonics = {}
            if key not in resources:
                # Resident member preparation interns identical immutable host
                # arrays. Share their basis without hashing the mesh on a CPU.
                owners = tuple(mesh.arrays[name] for name in ('latCell', 'lonCell',
                    'latVertex', 'lonVertex', 'areaCell', 'cellsOnEdge', 'verticesOnEdge',
                    'cellsOnVertex', 'dvEdge', 'dcEdge', 'xCell', 'yCell', 'zCell',
                    'xVertex', 'yVertex', 'zVertex'))
                shared_key = (*key, float(mesh.attrs.get('sphere_radius',6371220.0)),
                              *(id(array) for array in owners))
                resource = MESH_RESOURCES.get(shared_key)
                if resource is None:
                    resource = cls(mesh, degree)
                    resource.host_owners = owners
                    MESH_RESOURCES[shared_key] = resource
                resources[key] = resource
            resource = resources[key]
        cp.cuda.get_current_stream().wait_event(resource.ready)
        return resource


class SphericalPattern:
    """Stationary AR(1) coefficients, independent Philox stream per process.

    A snapshot contains only coefficients and counters, never the basis.
    Repeated calls at the same clock return the same pattern without drawing.
    """
    def __init__(self, resource, config):
        import cupy as cp
        self.resource, self.config = resource, config
        self.coefficients = cp.zeros((resource.modes, 3), dtype=cp.float32)
        ell = resource.ell
        scale = config.length_m/resource.radius
        # Cancel the common l=1 exponent before normalizing. Otherwise a
        # long correlation length underflows every mode and yields 0/0.
        # Multiply by scale twice on the GPU, keeping the dipole exponent
        # zero even when scale**2 would overflow on the host.
        variance = cp.exp(-((ell*(ell+1)-2)*scale)*scale)
        # Addition theorem: sum_m Y_lm(x)^2 = (2l+1)/(4 pi).
        self.weights = cp.sqrt(variance*(4*math.pi)/cp.sum(variance)).astype(cp.float32)
        self.last_time = None
        self.draw = 0
        self.pattern = cp.empty((resource.basis.shape[0], 3), dtype=cp.float32)

    def advance(self, seconds):
        import cupy as cp
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('pattern clock must be finite and nonnegative')
        if self.last_time == seconds:
            return self.pattern
        if self.last_time is not None and seconds < self.last_time:
            raise ValueError('pattern clock moved backwards; restore its checkpoint')
        if self.last_time is None:
            phi, innovation = 0.0, 1.0
        else:
            elapsed = (seconds-self.last_time)/self.config.time_s
            phi = math.exp(-elapsed)
            # Use elapsed directly: exp(-elapsed) can round to one while
            # the innovation is still representable by a float32 word.
            innovation = math.sqrt(-math.expm1(-2*elapsed))
        n = self.resource.modes*3
        _kernel('ar1')(((n+127)//128,), (128,),
            (self.coefficients, self.weights, np.int32(n), np.float32(phi),
             np.float32(innovation),
             np.uint64(self.config.member_seed), np.uint64(self.draw)))
        cp.matmul(self.resource.basis, self.coefficients, out=self.pattern)
        self.last_time, self.draw = seconds, self.draw+1
        return self.pattern

    def snapshot(self, *, device=False):
        import cupy as cp
        return dict(schema=SCHEMA, config=self.config.state_json(), time=self.last_time,
                    draw=self.draw, coefficients=(self.coefficients.copy() if device
                                                 else cp.asnumpy(self.coefficients)))

    def restore(self, value):
        import cupy as cp
        if value['schema'] != SCHEMA or value['config'] != self.config.state_json():
            raise ValueError('stochastic checkpoint configuration differs')
        clock, draw = value['time'], value['draw']
        if (type(draw) is not int or not 0 <= draw < 2**64
                or (clock is None and draw != 0)
                or (clock is not None and (not math.isfinite(clock) or clock < 0 or draw == 0))):
            raise ValueError('invalid stochastic checkpoint clock or draw counter')
        a = value['coefficients']
        if a.shape != self.coefficients.shape or a.dtype != np.float32:
            raise ValueError('stochastic checkpoint coefficient layout or values differ')
        a = cp.asarray(a)
        if not bool(cp.isfinite(a).all()):
            raise ValueError('stochastic checkpoint coefficient layout or values differ')
        self.coefficients[...] = a
        self.last_time, self.draw = clock, draw
        cp.matmul(self.resource.basis, self.coefficients, out=self.pattern)


class ForcingPatterns:
    """Independent SPPT scales and two backscatter streams on one mesh basis."""
    def __init__(self, resource, config):
        self.config = config
        self.base = SphericalPattern(resource, config)
        self.scales = [(amp, SphericalPattern(resource, replace(config,
            length_m=length, time_s=tau, seed=seed, sppt_scales=())))
            for amp, length, tau, seed in config.sppt_scales if config.sppt]

    def advance(self, seconds):
        import cupy as cp
        interval = self.config.interval_s
        if interval:
            # Decimal boundary clocks such as .3/.1 can divide one ULP
            # below the integral tick. Admit that rounding error before
            # flooring, so the update is not held for another whole interval.
            quotient = seconds/interval
            tick = math.floor(quotient)
            if quotient != tick and math.nextafter(quotient, math.inf) == tick+1:
                tick += 1
            seconds = tick*interval
        base = self.base.advance(seconds)
        if self.scales:
            sppt = cp.zeros(base.shape[0], dtype=cp.float32)
            for amplitude, process in self.scales:
                sppt += amplitude*process.advance(seconds)[:, 0]
        else:
            sppt = self.config.amplitude*base[:, 0]
        return sppt, base[:, 1], base[:, 2]

    def snapshot(self, *, device=False):
        return dict(schema=SCHEMA, config=self.config.state_json(), base=self.base.snapshot(device=device),
                    scales=[process.snapshot(device=device) for _, process in self.scales])

    def restore(self, value):
        if value['schema'] != SCHEMA or value['config'] != self.config.state_json():
            raise ValueError('stochastic checkpoint configuration differs')
        if len(value['scales']) != len(self.scales):
            raise ValueError('stochastic checkpoint scale count differs')
        self.base.restore(value['base'])
        for (_, process), saved in zip(self.scales, value['scales']):
            process.restore(saved)


class StochasticPhysics:
    """Apply SPPT and dissipation-budget SKEB to held dry-mass sources.

    SKEB uses the resolved kinetic-energy sink of the unperturbed physics
    momentum tendency. Its globally normalized dual-edge curl is rotational.
    This is a mesh adaptation, not WRF's prescribed constant-dissipation mode.
    """
    def __init__(self, mesh, vertical, config):
        import cupy as cp
        self.config = StochasticConfig.parse(config)
        self.resource = MeshHarmonics.cached(mesh, self.config.degree)
        self.process = ForcingPatterns(self.resource, self.config)
        z = cp.asarray(vertical.zgrid, dtype=cp.float32)
        n = mesh.n_cells
        if z.shape != (vertical.n_vert_levels+1, n):
            raise ValueError('stochastic taper requires zgrid[interface,cell]')
        height = (z[1:]+z[:-1])*0.5 - z[0]
        c = self.config
        lower = cp.clip(height/c.surface_m,0,1) if c.surface_limit else 1
        self.taper = cp.ascontiguousarray(cp.minimum(lower,
            cp.clip((c.top_end_m-height)/(c.top_end_m-c.top_start_m),0,1)))
        if c.surface_limit:
            self.taper[0,:] = 0
            if c.surface_profile:
                count = min(len(c.surface_profile), int(height.shape[0]))
                self.taper[:count,:] = cp.minimum(self.taper[:count,:],
                    cp.asarray(c.surface_profile[:count],dtype=cp.float32)[:,None])
        self.vertex_taper = cp.ascontiguousarray(cp.mean(self.taper[:,self.resource.cov],axis=2))
        self.calls = 0
        self.timing_events = []
        from .stochastic_timing import StepTimings
        self.step_timing = StepTimings(cp)

    def apply(self, held, *, state, rho_edge, normal_velocity, time_seconds, dt, names):
        import cupy as cp
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('stochastic dt must be positive')
        r, c = self.resource, self.config
        sample = None
        step_timing = getattr(self, 'step_timing', None)
        if len(self.timing_events) < 4 or (step_timing is not None and step_timing.active is not None):
            sample = tuple(cp.cuda.Event() for _ in range(3))
            sample[0].record()
        sppt, streamfunction, temperature = self.process.advance(float(time_seconds))
        if sample is not None:
            sample[1].record()
        nc, ne = r.cells, r.edges
        energy_scale = backscatter_rate_scale(dt, c.time_s, c.interval_s)
        # Regional physics has a padded garbage column; only real cells/edges participate.
        theta = held.rho_theta[:, :nc]
        momentum = held.rho_u[:, :ne]
        density = state.rho[:, :nc]
        edge_density = rho_edge[:, :ne]
        wind = normal_velocity[:, :ne]
        levels = self.taper.shape[0]
        has_qv = c.sppt and 'qv' in names
        qv = held.scalars[names.index('qv'),:,:nc] if has_qv else theta
        ts = _field_strides(theta, levels, nc, 'theta tendency')
        qs = _field_strides(qv, levels, nc, 'qv tendency')
        rs = _field_strides(density, levels, nc, 'cell density')
        ms = _field_strides(momentum, levels, ne, 'momentum tendency')
        es = _field_strides(edge_density, levels, ne, 'edge density')
        # Unused pointers keep one kernel ABI for SPPT-only and SKEB-only calls.
        psi = thermal = dissipation = norm = sppt
        if c.skeb:
            sink = cp.maximum(-wind*momentum/edge_density, 0)
            dissipation = cp.sum(sink*r.edge_weight[None,:], axis=1, keepdims=True)
            psi = cp.clip(streamfunction[nc:], -c.clip, c.clip)
            curl = r.orientation*(psi[r.voe[:,1]]-psi[r.voe[:,0]])/r.dv
            norm = cp.sqrt(cp.sum(curl*curl*r.edge_weight))
            thermal = temperature[:nc].copy()
            thermal -= cp.sum(thermal*r.area)
            thermal /= cp.maximum(cp.sqrt(cp.sum(thermal**2*r.area)),cp.float32(1e-30))
        # Fuse taper, clipping, SPPT and SKEB writes. No level-by-edge curl,
        # backscatter, gathered SPPT factors or level-by-cell heat allocation.
        _kernel('force_cells')(((levels*nc+127)//128,), (128,),
            (theta, qv, density, self.taper, sppt, thermal, dissipation,
             *ts, *qs, *rs, np.int32(nc), np.int32(levels),
             np.int32(c.sppt), np.int32(c.skeb), np.int32(has_qv),
             np.int32(c.logit), np.float32(c.clip),
             np.float32(energy_scale*c.thermal_fraction*300/1004)))
        _kernel('force_edges')(((levels*ne+127)//128,), (128,),
            (momentum, edge_density, self.taper, self.vertex_taper, sppt, psi,
             dissipation, norm, r.coe, r.voe, r.orientation, r.dv,
             *ms, *es, np.int32(nc), np.int32(self.vertex_taper.shape[1]),
             np.int32(ne), np.int32(levels), np.int32(c.sppt), np.int32(c.skeb),
             np.int32(c.logit), np.float32(c.clip), np.float32(energy_scale*c.skeb_fraction)))
        if sample is not None:
            sample[2].record()
            self.timing_events.append(sample)
            if step_timing is not None:
                step_timing.forcing(sample, cold=self.calls == 0)
        self.calls += 1
        return held

    def receipt(self):
        import cupy as cp
        from .device_admission import stochastic_array_ledger
        basis = dict(self.resource.receipt)
        if self.resource.ready.done:
            basis['gpu_elapsed_ms'] = cp.cuda.get_elapsed_time(
                self.resource.build_started, self.resource.ready)
        timing = [dict(sample=i, pattern_ms=cp.cuda.get_elapsed_time(start, pattern),
                       forcing_total_ms=cp.cuda.get_elapsed_time(start, end))
                  for i, (start, pattern, end) in enumerate(self.timing_events) if end.done]
        processes = [self.process.base, *(process for _, process in self.process.scales)]
        buffers = dict(taper=int(self.taper.nbytes), vertex_taper=int(self.vertex_taper.nbytes),
                       coefficients=sum(int(p.coefficients.nbytes) for p in processes),
                       spectral_weights=sum(int(p.weights.nbytes) for p in processes),
                       pattern_outputs=sum(int(p.pattern.nbytes) for p in processes))
        return dict(schema=SCHEMA, config=asdict(self.config), member_seed=self.config.member_seed,
                    member_seed_hex=f'{self.config.member_seed:016x}',
                    basis=basis, calls=self.calls, gpu_timing_samples=timing,
                    step_cost=self.step_timing.receipt(),
                    admission_arrays=stochastic_array_ledger(
                        self.resource.cells, int(self.taper.shape[0]), self.config.degree,
                        extra_scales=len(self.process.scales), skeb=self.config.skeb),
                    admission_note='logical array floor; construction scratch, intermediate expressions, allocator placement and opaque library workspaces excluded; not measured peak VRAM',
                    member_buffer_bytes=buffers,
                    member_buffer_note='logical retained arrays only; excludes mesh resource, CuPy pool, cuBLAS workspace and transient reductions; not peak VRAM',
                    timing_note='CUDA stream elapsed including dispatch gaps; cold calls excluded from matched warm step-cost grading',
                    evaluation='GPU cuBLAS matrix product into reused output; GPU Philox and fused cell/edge forcing (128 threads per block)',
                    dissipation='positive kinetic-energy sink of unperturbed physics momentum sources',
                    backscatter_normalization='integrated AR(1) covariance: 2/h*tanh(h/(2*time_s)), h=max(dt,interval_s)',
                    scientific_status='spread-skill qualification required')


class LtsStochasticPhysics:
    """One GPU spectral process and global budget for all classes of a member.

    Fine-first LTS visits class clocks out of order. Patterns are therefore
    evaluated once at a synchronized macro start and held over that macro.
    Unperturbed owned-edge sinks are integrated over each class's own dt,
    converted to area-weighted owned-cell dissipation at the macro endpoint,
    and used by every class in the next macro. The first macro has zero SKEB
    budget. This explicit one-macro lag avoids an order-dependent reduction.
    """
    schema = 'gpuwm-hex.lts-stochastic/v1'

    def __init__(self, mesh, vertical, config, layouts):
        import cupy as cp
        # Harmonic caches must not attach GPU basis arrays to the host mesh
        # serialized by moving checkpoints. Keep its read-only array owners,
        # but put runtime-only cache attributes on a private shallow view.
        import copy
        runtime_mesh = copy.copy(mesh)
        runtime_mesh.__dict__.pop('_stochastic_harmonics', None)
        self.global_physics = StochasticPhysics(runtime_mesh, vertical, config)
        self.config = self.global_physics.config
        self.resource = self.global_physics.resource
        self.process = self.global_physics.process
        self.levels = int(vertical.n_vert_levels)
        self.layouts = []
        for layout in layouts:
            cells = cp.asarray(layout.cell_l2g, dtype=cp.int32)
            edges = cp.asarray(layout.edge_l2g, dtype=cp.int32)
            self.layouts.append(dict(cells=cells, edges=edges,
                owned_edges=int(layout.n_owned_edges)))
        self.eoc = cp.ascontiguousarray(cp.asarray(mesh.arrays['edgesOnCell'], dtype=cp.int32))
        self.nec = cp.ascontiguousarray(cp.asarray(mesh.arrays['nEdgesOnCell'], dtype=cp.int32))
        self.sink_integral = cp.zeros((self.levels, mesh.n_edges), dtype=cp.float32)
        self.cell_sink = cp.empty((self.levels, mesh.n_cells), dtype=cp.float32)
        self.dissipation = cp.zeros((self.levels, 1), dtype=cp.float32)
        self.boundary_seconds = None
        self.macro = None
        self.calls = [0] * len(layouts)
        self.macro_calls = 0

    def begin_macro(self, seconds, duration):
        import cupy as cp
        if (self.macro is not None or not math.isfinite(seconds) or seconds < 0
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError('invalid or overlapping stochastic macro interval')
        if self.boundary_seconds is not None and seconds != self.boundary_seconds:
            raise ValueError('stochastic macro does not start at its carried boundary')
        self.macro = (float(seconds), float(duration))
        self.class_elapsed = [0.0] * len(self.layouts)
        r, c, g = self.resource, self.config, self.global_physics
        sppt, streamfunction, temperature = self.process.advance(float(seconds))
        self.sppt = cp.ascontiguousarray(sppt)
        self.psi = cp.ascontiguousarray(cp.clip(streamfunction[r.cells:], -c.clip, c.clip))
        curl = r.orientation * (self.psi[r.voe[:, 1]] - self.psi[r.voe[:, 0]]) / r.dv
        self.norm = cp.sqrt(cp.sum(curl * curl * r.edge_weight))
        thermal = temperature[:r.cells].copy()
        thermal -= cp.sum(thermal * r.area)
        thermal /= cp.maximum(cp.sqrt(cp.sum(thermal * thermal * r.area)), cp.float32(1e-30))
        self.local_fields = [(
            cp.ascontiguousarray(g.taper[:, row['cells']]),
            cp.ascontiguousarray(self.sppt[row['cells']]),
            cp.ascontiguousarray(thermal[row['cells']])) for row in self.layouts]
        self.sink_integral.fill(0)

    def apply(self, index, held, *, state, rho_edge, normal_velocity, time_seconds, dt, names):
        import cupy as cp
        if self.macro is None:
            raise ValueError('stochastic class step is outside a synchronized macro')
        start, duration = self.macro
        if (not math.isfinite(dt) or dt <= 0 or time_seconds < start
                or time_seconds + dt > start + duration + 1e-8
                or abs(time_seconds - (start + self.class_elapsed[index])) > 1e-8):
            raise ValueError('stochastic class step straddles its macro interval')
        row, r, c = self.layouts[index], self.resource, self.config
        nc, ne = int(row['cells'].size), int(row['edges'].size)
        theta, momentum = held.rho_theta[:, :nc], held.rho_u[:, :ne]
        density, edge_density = state.rho[:, :nc], rho_edge[:, :ne]
        # Only edge owners contribute. Halo copies receive identical global
        # patterns but never duplicate the member's dissipation reduction.
        owned = row['owned_edges']
        if c.skeb:
            sink = cp.maximum(-normal_velocity[:, :owned] * momentum[:, :owned]
                              / edge_density[:, :owned], 0)
            self.sink_integral[:, row['edges'][:owned]] += sink * cp.float32(dt)
        taper, sppt, thermal = self.local_fields[index]
        has_qv = c.sppt and 'qv' in names
        qv = held.scalars[names.index('qv'), :, :nc] if has_qv else theta
        ts = _field_strides(theta, self.levels, nc, 'theta tendency')
        qs = _field_strides(qv, self.levels, nc, 'qv tendency')
        rs = _field_strides(density, self.levels, nc, 'cell density')
        ms = _field_strides(momentum, self.levels, ne, 'momentum tendency')
        es = _field_strides(edge_density, self.levels, ne, 'edge density')
        scale = backscatter_rate_scale(duration, c.time_s, c.interval_s)
        _kernel('force_cells')(((self.levels * nc + 127) // 128,), (128,),
            (theta, qv, density, taper, sppt, thermal, self.dissipation,
             *ts, *qs, *rs, np.int32(nc), np.int32(self.levels),
             np.int32(c.sppt), np.int32(c.skeb), np.int32(has_qv),
             np.int32(c.logit), np.float32(c.clip),
             np.float32(scale * c.thermal_fraction * 300 / 1004)))
        # Every local edge, including halos, reads the ORIGINAL global dual
        # vertices and cell tapers. No truncated class topology enters curl.
        _kernel('force_lts_edges')(((self.levels * ne + 127) // 128,), (128,),
            (momentum, edge_density, self.global_physics.taper,
             self.global_physics.vertex_taper, self.sppt, self.psi,
             self.dissipation, self.norm, r.coe, r.voe, r.orientation, r.dv, row['edges'],
             *ms, *es, np.int32(r.cells), np.int32(self.global_physics.vertex_taper.shape[1]),
             np.int32(ne), np.int32(self.levels), np.int32(c.sppt), np.int32(c.skeb),
             np.int32(c.logit), np.float32(c.clip), np.float32(scale * c.skeb_fraction)))
        self.calls[index] += 1
        self.class_elapsed[index] += dt
        return held

    def end_macro(self):
        import cupy as cp
        if self.macro is None:
            raise ValueError('no stochastic macro to finish')
        start, duration = self.macro
        if any(abs(value - duration) > 1e-8 for value in self.class_elapsed):
            raise ValueError('stochastic macro is missing an owned class interval')
        if self.config.skeb:
            r = self.resource
            _kernel('lts_sink_cells')(((self.levels * r.cells + 127) // 128,), (128,),
                (self.sink_integral, self.cell_sink, self.eoc, self.nec, r.edge_weight,
                 np.int32(r.cells), np.int32(r.edges), np.int32(self.eoc.shape[1]),
                 np.int32(self.levels), np.float32(duration)))
            self.dissipation[...] = cp.sum(self.cell_sink * r.area[None, :], axis=1, keepdims=True)
        self.boundary_seconds = start + duration
        self.macro = None
        self.local_fields = []
        self.macro_calls += 1

    def snapshot(self, *, device=True):
        import cupy as cp
        if self.macro is not None:
            raise ValueError('stochastic checkpoint requires a completed macro boundary')
        return dict(schema=self.schema, config=self.config.state_json(),
            process=self.process.snapshot(device=device), boundary_seconds=self.boundary_seconds,
            dissipation=(self.dissipation.copy() if device else cp.asnumpy(self.dissipation)),
            macro_calls=self.macro_calls)

    def restore(self, value):
        import cupy as cp
        if value['schema'] != self.schema or value['config'] != self.config.state_json():
            raise ValueError('LTS stochastic checkpoint configuration differs')
        if self.macro is not None:
            raise ValueError('cannot restore a stochastic process inside an active macro')
        budget = cp.asarray(value['dissipation'])
        boundary = value['boundary_seconds']
        macro_calls = value['macro_calls']
        if (type(macro_calls) is not int or macro_calls < 0
                or (boundary is None) != (macro_calls == 0)):
            raise ValueError('invalid LTS stochastic checkpoint macro counter')
        if (budget.shape != self.dissipation.shape or budget.dtype != cp.float32
                or not bool(cp.all(cp.isfinite(budget) & (budget >= 0)))
                or (boundary is not None and (not math.isfinite(boundary) or boundary < 0))):
            raise ValueError('invalid LTS stochastic checkpoint budget or boundary')
        self.process.restore(value['process'])
        if boundary is not None and not 0 <= self.process.base.last_time < boundary:
            raise ValueError('stochastic process clock is outside its carried macro boundary')
        self.dissipation[...] = budget
        self.boundary_seconds = boundary
        self.macro_calls = macro_calls

    def receipt(self):
        return dict(schema=self.schema, config=asdict(self.config), device='GPU',
            spectral_processes=1, pattern_clock='one global draw at each macro start',
            draws=self.process.base.draw, macro_calls=self.macro_calls, calls_by_class=self.calls,
            budget='area-weighted global owned-cell mean of time-integrated owned-edge sinks',
            budget_lag_macros=1, startup_budget='zero before the first accepted macro',
            curl='global dual-vertex field, gathered by global edge ID including halos',
            checkpoint='mesh-independent spectral coefficients, counters and per-level dissipation',
            basis=self.resource.receipt, scientific_status='spread-skill qualification required')


def resolve_options(value=None, namelist=None):
    """One validated configuration for the public door and either physics route."""
    options = {}
    if namelist is not None:
        from .config import parse_fortran_namelist
        groups = namelist if isinstance(namelist, Mapping) else parse_fortran_namelist(namelist)
        for group in groups.values():
            for key in ('do_sppt','do_skeb','config_stochini'):
                if key in group:
                    if type(group[key]) is not bool:
                        raise ValueError(f'{key} must be logical')
                    if key == 'config_stochini':
                        options['restart_required'] = group[key]
                    else:
                        options[{'do_sppt':'sppt','do_skeb':'skeb'}[key]] = group[key]
        params = {key: val for group in groups.values() for key,val in group.items()
                  if key.startswith('config_sppt') or key.startswith('config_iseed_sppt')}
        if options.get('sppt') or options.get('skeb'):
            mapping = {'config_sppt_hgt_top1':'top_start_m','config_sppt_hgt_top2':'top_end_m',
                       'config_sppt_logit':'logit', 'config_sppt_sfclimit':'surface_limit',
                       'config_spptint':'interval_s'}
            defaults = StochasticConfig()
            scales = []
            indexed = {f'config_sppt_{suffix}_{i}' for suffix in ('lscale', 'tau') for i in range(1, 6)}
            indexed.update(f'config_sppt_{i}' for i in range(1, 6))
            indexed.update(f'config_iseed_sppt{i}' for i in range(1, 6))
            explicit_scales = bool(indexed.intersection(params))
            consumed = set(mapping)
            for index in range(1, 6):
                keys = (f'config_sppt_{index}', f'config_sppt_lscale_{index}',
                        f'config_sppt_tau_{index}', f'config_iseed_sppt{index}')
                consumed.update(keys)
                # Switches and taper/cadence controls do not declare independent
                # scales. Keep the base pattern so later scalar JSON controls apply.
                if not explicit_scales or (index != 1 and not any(key in params for key in keys)):
                    continue
                amplitude = params.get(keys[0], defaults.amplitude if index == 1 else 0.0)
                if not math.isfinite(amplitude) or amplitude < 0:
                    raise ValueError(f'{keys[0]} must be nonnegative and finite')
                if amplitude:
                    seed = params.get(keys[3], defaults.seed+index-1)
                    # MPAS templates quote the decimal int64 seed. Preserve its
                    # exact integer value without a float conversion.
                    if isinstance(seed, str) and seed.isascii() and seed.isdecimal():
                        seed = int(seed)
                    scales.append((amplitude, params.get(keys[1], defaults.length_m),
                        params.get(keys[2], defaults.time_s), seed))
            if scales:
                options['sppt_scales'] = tuple(scales)
            elif any(key.startswith('config_sppt_') and key[-1:].isdigit() for key in params):
                options['sppt'] = False
            for key,val in params.items():
                if key in mapping:
                    options[mapping[key]] = val
                    if key == 'config_sppt_sfclimit' and val is True:
                        options['surface_profile'] = (0.0,0.5,1.0,1.0,1.0,1.0,1.0)
                elif key in consumed:
                    pass
                else:
                    raise ValueError(f'{key}={val!r} is not implemented by the spherical forcing; '
                                     'use --stochastic JSON for its length, time, amplitude and taper controls')
    if value is not None:
        options.update(json.loads(value) if isinstance(value,str) else value)
    return StochasticConfig.parse(options)
