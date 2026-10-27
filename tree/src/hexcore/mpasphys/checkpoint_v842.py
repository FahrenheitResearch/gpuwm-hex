"""Public checkpoint flags over the existing sealed MPAS physics carrier."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .moving_v842 import MovingRoute, TIME_FORMAT, load_carrier


def _seconds(hours):
    values = [Fraction(value.strip()) * 3600 for value in str(hours).split(',')]
    result = tuple(sorted({Fraction(round(value)) for value in values}))
    if (not result or any(value <= 0 for value in result)
            or any(abs(value-round(value)) > Fraction(1, 10**9) for value in values)):
        raise ValueError('--checkpoint-hours needs positive integral-second boundaries')
    return result


def _identity(inputs):
    mesh, grid, _, _, reference, _, names = inputs
    arrays = [(f'mesh/{name}', value) for name, value in sorted(mesh.arrays.items())
              if isinstance(value, np.ndarray)]
    arrays += [(f'vertical/{name}', getattr(grid.vertical_grid, name))
               for name in ('zgrid', 'dzw')]
    arrays += [(f'reference/{name}', getattr(reference, name))
               for name in ('rho_base', 'rho_theta_base', 'pressure_base', 'exner_base')]

    def digest(item):
        name, value = item
        value = np.ascontiguousarray(value)
        return name, dict(shape=list(value.shape), dtype=value.dtype.str,
                          sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())

    # Bind large static arrays before CUDA, without a serial hash queue.
    with ThreadPoolExecutor(max_workers=min(8, len(arrays))) as pool:
        result = dict(pool.map(digest, arrays))
    return dict(arrays=result, scalar_names=list(names))


@dataclass
class CheckpointRoute(MovingRoute):
    checkpoint_seconds: tuple = ()
    cli_restart: bool = False
    da_analysis: bool = False
    checkpoint_dt: str | None = None
    checkpoint_identity: dict | None = None
    checkpoint_physics: dict | None = None
    restart_metadata: dict | None = None
    checkpoints: list = field(default_factory=list)
    checkpoint_preparation: dict | None = None
    frame_from_cli: bool = False
    restart_dt_change: dict | None = None
    restart_corrections_change: dict | None = None

    def carried_rows(self, layout):
        selected = super().carried_rows(layout)
        if getattr(self, '_restoring_moving', False):
            return selected
        seen = {(row['pool'], row['name']) for row in selected}
        # A same-mesh restart must keep evolving SST/ice and surface roles.
        # Moving remaps deliberately take those fields from the new member.
        return selected + [row for row in layout.fields
            if row['pool'] == 'sfc_input' and row['kind'] != 'character'
            and row['dimensions'] and row['dimensions'][-1] == 'nCells'
            and not any(size == 0 for size in row['shape'])
            and (row['pool'], row['name']) not in seen]

    def restore_physics(self, *args, **kwargs):
        # Moving carriers predate the public checkpoint flags and keep the
        # target member's geography. A CLI same-mesh checkpoint carries it.
        self._restoring_moving = not self.cli_restart
        try:
            return super().restore_physics(*args, **kwargs)
        finally:
            self._restoring_moving = False

    def mark_restart(self, layout):
        super().mark_restart(layout)
        if self.da_analysis:
            layout.configs['config_do_DAcycling'] = True

    def resolve_schedule(self, duration, dt, streams):
        """CLI --hours is the original end, not another duration after resume."""
        dt, duration = Fraction(str(dt)), Fraction(str(duration))
        if dt > 0 and abs(duration/dt-round(duration/dt)) <= Fraction(1, 10**9):
            duration = round(duration/dt) * dt
        if not self.frame_from_cli:
            from .driver_v842 import interval_seconds
            history = next((stream for stream in streams if stream.name in ('output', 'history')
                            and 'output' in stream.directions), None)
            if history is None or history.output_interval == 'none':
                raise ValueError('native checkpoints need a recurring history stream')
            self.frame_seconds = float(interval_seconds(history.output_interval))
        offset = Fraction(str(self.offset_seconds))
        if dt <= 0 or offset % dt:
            raise ValueError('native checkpoint must lie on an outer-step boundary')
        # A resume may change the timestep (e.g. to run past an instability). The carrier holds
        # model state at one instant: prognostic and diagnostic fields, physics rates per second
        # and the physics alarms' phase in model seconds, none of it a per-step quantity, so it is
        # read the same under any step that lands on the checkpoint's instant. Native MPAS-A
        # restarts under a changed config_dt the same way. Only the instant must be a whole number
        # of new steps from the start, which the offset check above enforces.
        if self.restart_metadata is not None and Fraction(self.restart_metadata['dt']) != dt:
            self.restart_dt_change = dict(checkpoint_dt=str(Fraction(self.restart_metadata['dt'])),
                                          resumed_dt=str(dt))
            print(f"gpuwm-hex: resuming checkpoint written at dt {Fraction(self.restart_metadata['dt'])} s "
                  f"under dt {dt} s", flush=True)
        self.checkpoint_dt = str(dt)
        if self.cli_restart:
            duration -= offset
        if duration <= 0:
            raise ValueError('--restart-from must precede the original forecast end')
        end = offset + duration
        for boundary in self.checkpoint_seconds:
            if boundary > end or boundary % dt or boundary % Fraction(str(self.frame_seconds)):
                raise ValueError('--checkpoint-hours must name outer-step history frames before the forecast end')
        return duration

    def bind_inputs(self, inputs, config):
        began = time.perf_counter()
        started = datetime.now(timezone.utc).isoformat()
        from .. import numerics
        physics = dict(config=config.receipt(), numerics=numerics.mode())
        physics = json.loads(json.dumps(physics))
        if self.restart_metadata is not None and physics != self.restart_metadata.get('physics'):
            # The applied correct-mode findings may differ (a release that adds a correction, e.g. RP7):
            # a correction changes scheme arithmetic from the resume on, never the carried state's
            # fields or layout. Suite, schemes, options, backends, mode and numerics must still match.
            def _scheme_identity(value):
                value = json.loads(json.dumps(value))
                config = value.get('config') if isinstance(value, dict) else None
                if isinstance(config, dict):
                    for key in ('corrections', 'correct_skip', 'correct_mode'):
                        config.pop(key, None)
                return value
            carried = self.restart_metadata.get('physics')
            if _scheme_identity(physics) != _scheme_identity(carried):
                raise ValueError('native checkpoint physics configuration or numerics differ')
            before = (carried or {}).get('config', {}).get('corrections', [])
            after = physics['config'].get('corrections', [])
            self.restart_corrections_change = dict(checkpoint=list(before), resumed=list(after))
            print(f"gpuwm-hex: resuming checkpoint written with corrections {before} under {after}", flush=True)
        identity = _identity(inputs)
        if self.restart_metadata is not None and identity != self.restart_metadata['identity']:
            raise ValueError('native checkpoint mesh, vertical grid, base state or species differ')
        self.checkpoint_identity = identity
        self.checkpoint_physics = physics
        self.checkpoint_preparation = dict(seconds=time.perf_counter()-began,
            started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
            parallelism='up to 8 host hash workers; before CUDA')

    def checkpoint_due(self, step, dt):
        seconds = Fraction(str(self.offset_seconds)) + step * Fraction(str(dt))
        return step > 0 and seconds in self.checkpoint_seconds

    def after_step(self, step, **values):
        stop = super().after_step(step, **values)
        if self.checkpoint_due(step, values['dt']):
            seconds = int(Fraction(str(self.offset_seconds)) + step * Fraction(str(values['dt'])))
            label = f'f{seconds // 3600:03d}' if seconds % 3600 == 0 else f't{seconds:08d}'
            target = Path(values['out']) / f'checkpoint.{label}.pkl'
            export = {key: value for key, value in values.items() if key not in ('steps', 'columns', 'out')}
            self.checkpoints.append(self.export(target, step=step, **export))
        return stop

    def export(self, path, **values):
        if self.checkpoint_identity is None or self.checkpoint_dt is None:
            raise ValueError('native checkpoint identity/schedule must bind before export')
        began = time.perf_counter()
        started = datetime.now(timezone.utc).isoformat()
        receipt = super().export(path, **values)
        receipt['checkpoint'] = dict(schema='mpas-native-checkpoint/v1',
            dt=self.checkpoint_dt, identity=self.checkpoint_identity,
            physics=self.checkpoint_physics)
        receipt['checkpoint_export'] = dict(started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.perf_counter()-began,
            parallelism='host carrier serialization and SHA256; native state already published')
        # The payload is the existing carrier, written once. Metadata stays in
        # its small sealed receipt; never decode/rewrite a weather payload.
        Path(path).with_suffix(Path(path).suffix+'.json').write_text(
            json.dumps(receipt, indent=2, default=str)+'\n')
        return receipt

    def receipt(self):
        result = {**super().receipt(), 'native_checkpoints': self.checkpoints,
                  'checkpoint_preparation': self.checkpoint_preparation}
        if self.restart_dt_change is not None:
            result['restart_dt_change'] = self.restart_dt_change
        if self.restart_corrections_change is not None:
            result['restart_corrections_change'] = self.restart_corrections_change
        return result


def resolve_route(arguments, text):
    """Resolve moving environment plus CLI restart before native pool decoding."""
    from ..fullstep_lts_admission import moving_region_requested
    from .forecast_v842 import set_namelist_options
    moving = MovingRoute.from_environ(arguments) if moving_region_requested() else None
    restart = getattr(arguments, 'restart_from', None)
    hours = getattr(arguments, 'checkpoint_hours', None)
    analysis = bool(getattr(arguments, 'da_analysis', None))
    if analysis and not (restart or moving is not None and moving.carrier is not None):
        raise ValueError('native --da-analysis requires --restart-from or a moving carrier')
    if restart is None and hours is None and not analysis:
        if moving is not None:
            text = moving.resume_namelist(text)
        return moving, text
    route = CheckpointRoute(**({item.name: getattr(moving, item.name)
                               for item in fields(MovingRoute)} if moving is not None else {}))
    if hours is not None:
        route.checkpoint_seconds = _seconds(hours)
    minutes = getattr(arguments, 'history_every_minutes', None)
    if minutes is not None:
        route.frame_seconds = float(minutes) * 60
    route.frame_from_cli = minutes is not None or moving is not None
    if route.frame_seconds <= 0:
        raise ValueError('native checkpoint history interval must be positive')
    if restart is not None:
        path = Path(restart).expanduser().resolve()
        if route.carrier_path and Path(route.carrier_path).resolve() != path:
            raise ValueError('CLI and moving environment name different native restart carriers')
        route.carrier_path, route.carrier = str(path), load_carrier(path)
        route.origin_time = route.carrier['origin_time']
        route.offset_seconds = float(route.carrier['time_seconds'])
        valid = datetime.strptime(route.origin_time, TIME_FORMAT) + timedelta(seconds=route.offset_seconds)
        if (route.offset_seconds <= 0 or route.offset_seconds % 1 or
                valid.strftime(TIME_FORMAT) != route.carrier['valid_time']):
            raise ValueError('native checkpoint elapsed time and valid clock disagree')
        route.restart_metadata = json.loads(path.with_suffix(path.suffix+'.json').read_text()).get('checkpoint')
        if not route.restart_metadata or route.restart_metadata.get('schema') != 'mpas-native-checkpoint/v1':
            raise ValueError('--restart-from needs a checkpoint written by the native checkpoint flags')
        route.cli_restart = True
        if route.operator:
            raise ValueError('native --restart-from uses the same mesh; moving remaps use the moving carrier input')
    route.da_analysis = analysis
    text = route.resume_namelist(text)
    # Native init pools supply static fields. The carrier supplies restart
    # fields after decoding, avoiding init/restart Registry shape collisions.
    if route.carrier is not None:
        text = set_namelist_options(text, {'restart': {'config_do_restart': False}})
    return route, text
