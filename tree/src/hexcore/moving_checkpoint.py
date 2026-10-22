"""Experimental full-step LTS boundary export for moving-mesh experiments.

Only synchronized boundaries with empty transfer registers can be exported.
This is an explicit multi-backend carrier, separate from the single-backend
cycling interface. Loading an arbitrary pickle is deliberately not supported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import numpy as np


def export_boundary(path, *, active, stacks, previous, forcing, host, physics_scope):
    if not active.registers_empty():
        raise RuntimeError("moving checkpoint needs empty LTS transfer registers")
    started = time.perf_counter()
    state = active.gather(("rho", "rho_theta", "rho_u", "rho_w",
                           "theta_m", "exner", "density_perturbation",
                           "rho_theta_perturbation", "pressure_perturbation",
                           "normal_velocity", "vertical_velocity"))
    state["scalars"] = active.scalars.gather() if active.scalars is not None else None
    times = [float(stack["driver"].atmosphere.state.time_seconds) for stack in stacks]
    if len(set(times)) != 1:
        raise RuntimeError(f"moving checkpoint clocks differ: {times}")
    import cupy as cp
    def host_arrays(item):
        if isinstance(item, Mapping):
            return {key: host_arrays(value) for key, value in item.items()}
        if is_dataclass(item):
            return {field.name: host_arrays(getattr(item, field.name)) for field in fields(item)}
        if isinstance(item, cp.ndarray):
            return cp.asnumpy(item)
        return item
    backends = []
    local_atmospheres = []
    for index, stack in enumerate(stacks):
        with physics_scope.rank(index, int(cp.cuda.Device().id)):
            backends.append(stack["backend"].restart_state())
            atmosphere=stack["driver"].atmosphere
            local_atmospheres.append(dict(
                state={name:cp.asnumpy(getattr(atmosphere.state,name)) for name in
                       ('rho','rho_theta','rho_u','rho_w','scalars')},
                saved={name:cp.asnumpy(getattr(atmosphere.saved,name)) for name in
                       ('theta_m','exner','density_perturbation','rho_theta_perturbation',
                        'pressure_perturbation','normal_velocity','vertical_velocity')}))
    payload = dict(schema="gpuwm-hex.moving-boundary/v1", time_seconds=times[0],
                   state=state, backends=backends,
                   stochastic=(None if getattr(active, "stochastic", None) is None else
                               _host_arrays(active.stochastic.snapshot(device=True), cp)),
                   local_atmospheres=local_atmospheres,
                   mailbox={name:cp.asnumpy(getattr(active,name)) for name in ('G_cell','G_face','G_edge')},
                   scalar_mailbox=(None if active.scalars is None else
                       {name:cp.asnumpy(getattr(active.scalars,name)) for name in ('G','hold')}),
                   previous=[host_arrays(item) for item in previous],
                   forcing=[host_arrays(item) for item in forcing],
                   layouts=[tables.layout for tables in active.plan.classes],
                   rates=np.asarray(active.plan.rates)[active.plan.cell_class],
                   vertical=host["prepared"].vertical,
                   mesh=host["prepared"].mesh,
                   scalar_names=host["scalar_names"])
    return _seal(path, payload, backends, state, started)


#: The per-card buffers of the multi-card executor (hexcore.cuda_fullstep_lts_multi._Lane) that carry
#: published halo planes across a macro boundary; the registers are empty there and are not carried.
LANE_MAILBOXES = ("G_cell", "G_face", "G_edge", "G_s", "hold")


def _host_arrays(item, cp):
    if isinstance(item, list):
        return [_host_arrays(value, cp) for value in item]
    if isinstance(item, tuple):
        return tuple(_host_arrays(value, cp) for value in item)
    if isinstance(item, Mapping):
        return {key: _host_arrays(value, cp) for key, value in item.items()}
    if is_dataclass(item):
        return {field.name: _host_arrays(getattr(item, field.name), cp) for field in fields(item)}
    if isinstance(item, cp.ndarray):
        with cp.cuda.Device(item.device.id):
            return cp.asnumpy(item)
    return item


def export_boundary_blocks(path, *, active, stacks, previous, forcing, host, physics_scope):
    """The carrier of a region split across several cards (``FullStepLtsMulti``).

    The same schema and fields as :func:`export_boundary`, one entry per BLOCK where the
    one-card carrier has one per class: every block's backend restart state, local
    atmosphere and step carriers, read on the block's own card, and ``layouts`` are the
    blocks' layouts.  The blocks' owner columns cover the mesh once, so the remapped resume
    (``hexcore.moving_physics``), which gathers owner columns through ``layouts``, reads
    this carrier exactly as a one-card one and slices it into any target partition (one
    card or several).  ``lane_mailboxes`` (per card) replace the one-card ``mailbox`` for
    the same-partition identity resume.
    """
    if not active.registers_empty():
        raise RuntimeError("moving checkpoint needs empty LTS transfer registers")
    import cupy as cp

    started = time.perf_counter()
    state = active.gather(("rho", "rho_theta", "rho_u", "rho_w",
                           "theta_m", "exner", "density_perturbation",
                           "rho_theta_perturbation", "pressure_perturbation",
                           "normal_velocity", "vertical_velocity"))
    state["scalars"] = active.gather_scalars() if active.ntr else None
    blocks = list(active.blocks)
    if [block.gid for block in blocks] != list(range(len(stacks))):
        raise RuntimeError("moving checkpoint blocks are not in plan order")
    times = []
    for block in blocks:
        with cp.cuda.Device(block.device):
            times.append(float(stacks[block.gid]["driver"].atmosphere.state.time_seconds))
    if len(set(times)) != 1:
        raise RuntimeError(f"moving checkpoint clocks differ: {times}")
    backends, local_atmospheres, previous_host, forcing_host = [], [], [], []
    for block in blocks:
        stack = stacks[block.gid]
        with cp.cuda.Device(block.device), physics_scope.rank(block.gid, block.device):
            cp.cuda.get_current_stream().synchronize()
            backends.append(stack["backend"].restart_state())
            atmosphere = stack["driver"].atmosphere
            local_atmospheres.append(dict(
                state={name: cp.asnumpy(getattr(atmosphere.state, name)) for name in
                       ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars')},
                saved={name: cp.asnumpy(getattr(atmosphere.saved, name)) for name in
                       ('theta_m', 'exner', 'density_perturbation', 'rho_theta_perturbation',
                        'pressure_perturbation', 'normal_velocity', 'vertical_velocity')}))
            previous_host.append(_host_arrays(previous[block.gid], cp))
            forcing_host.append(_host_arrays(forcing[block.gid], cp))
    lane_mailboxes = {}
    for device, lane in active.lanes.items():
        with cp.cuda.Device(device):
            lane_mailboxes[int(device)] = {name: cp.asnumpy(getattr(lane, name)) for name in LANE_MAILBOXES
                                           if getattr(lane, name, None) is not None}
    plan = active.plan
    payload = dict(schema="gpuwm-hex.moving-boundary/v1", time_seconds=times[0],
                   state=state, backends=backends,
                   stochastic=(None if getattr(active, "stochastic", None) is None else
                               _host_arrays(active.stochastic.snapshot(device=True), cp)),
                   local_atmospheres=local_atmospheres,
                   lane_mailboxes=lane_mailboxes,
                   previous=previous_host, forcing=forcing_host,
                   layouts=[block.tables.layout for block in blocks],
                   partition=dict(kind="blocks", devices=[int(block.device) for block in blocks],
                                  block_class=[int(block.index) for block in blocks]),
                   rates=np.asarray(plan.rates)[plan.cell_class],
                   vertical=host["prepared"].vertical,
                   mesh=host["prepared"].mesh,
                   scalar_names=host["scalar_names"])
    return _seal(path, payload, backends, state, started)


def _seal(path, payload, backends, state, started):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    inventory = [{"identity": value["identity"],
                  "seam_identity": value["seam"]["identity"],
                  "scalars": value["seam"]["scalars"],
                  "arrays": {key: {"shape": list(array.shape), "dtype": str(array.dtype)}
                             for key, array in value["seam"]["arrays"].items()}}
                 for value in backends]
    receipt = dict(schema=payload["schema"], path=str(path), bytes=path.stat().st_size,
                   sha256=digest.hexdigest(), time_seconds=payload["time_seconds"],
                   registers_empty=True, backends=inventory,
                   partition=payload.get("partition", dict(kind="classes")),
                   state={key: None if value is None else list(value.shape)
                          for key, value in state.items()},
                   wall_seconds=time.perf_counter() - started)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    return receipt


def restore_lane_mailboxes(active, payload):
    """The same-partition identity resume of a multi-card carrier: every card's halo planes as sealed."""
    import cupy as cp

    stored = payload.get("lane_mailboxes")
    if stored is None:
        raise ValueError("a same-partition multi-card resume needs a multi-card carrier (lane_mailboxes); "
                         "this carrier was sealed on one card")
    if sorted(int(device) for device in stored) != sorted(int(device) for device in active.lanes):
        raise ValueError(f"restart cards differ: carrier {sorted(stored)}, run {sorted(active.lanes)}")
    for device, lane in active.lanes.items():
        with cp.cuda.Device(device):
            for name, array in stored[int(device)].items():
                target = getattr(lane, name)
                if target is None or target.shape != array.shape:
                    raise ValueError(f"restart mailbox differs on card {device}: {name}")
                target[...] = cp.asarray(array)
            cp.cuda.get_current_stream().synchronize()


def load_boundary(path):
    """Read an owned experiment carrier only after its receipt hash matches.

    The carrier is a Python pickle and must never come from an untrusted
    download. The run harness supplies only this lane's own boundary files.
    """
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != receipt['sha256'] or path.stat().st_size != receipt['bytes']:
        raise ValueError('moving carrier does not match its sealed receipt')
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    if payload['schema'] != 'gpuwm-hex.moving-boundary/v1':
        raise ValueError('unknown moving carrier schema')
    return payload


def require_same_layouts(current, stored):
    if len(current) != len(stored):
        raise ValueError('same-mesh restart class count differs')
    for fresh, old in zip(current, stored):
        for name in ('cell_l2g', 'edge_l2g', 'n_owned_cells', 'n_owned_edges'):
            if not np.array_equal(getattr(fresh, name), getattr(old, name)):
                raise ValueError(f'same-mesh restart layout differs: {name}')


def restore_local_atmosphere(driver, stored):
    """Keep private halo and scratch state for the exact identity control."""
    import cupy as cp
    for name,values in stored.items():
        current=getattr(driver.atmosphere,name)
        for key,array in values.items():
            target=getattr(current,key)
            if target.shape!=array.shape or target.dtype!=array.dtype:
                raise ValueError(f'local moving carrier shape/dtype differs: {name}/{key}')
            target[...]=cp.asarray(array)


def restore_mailboxes(active,payload):
    import cupy as cp
    for name,array in payload.get('mailbox',{}).items():
        target=getattr(active,name)
        if target.shape!=array.shape:
            raise ValueError(f'restart mailbox differs: {name}')
        target[...]=cp.asarray(array)
    if payload.get('scalar_mailbox') is not None:
        if active.scalars is None:
            raise ValueError('restart scalar mailbox has no live transport')
        for name,array in payload['scalar_mailbox'].items():
            target=getattr(active.scalars,name)
            if target.shape!=array.shape:
                raise ValueError(f'restart scalar mailbox differs: {name}')
            target[...]=cp.asarray(array)


def carry_physics_top_pressure(host, payload, *, sealer=None):
    """Preserve the original global physics boundary before constructing a new mesh."""
    sources = payload['backends']
    if not sources:
        raise ValueError('moving boundary has no physics classes')
    values = [float(item['seam']['identity']['p_top_pa']) for item in sources]
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError('moving physics top pressure must be finite and positive')
    if any(value != values[0] for value in values):
        raise ValueError('moving physics classes disagree on top pressure')
    constructor = dict(host['constructor_values'])
    derived = float(constructor['p_top_pa'])
    constructor['p_top_pa'] = values[0]
    if sealer is None:
        from hexcore.physics_backend_admission import resolve_backend, DEFAULT_BACKEND
        from hexcore.cuda_arwen_physics_v841 import SealedArwenConstructorV841
        row = resolve_backend(str(host.get('physics_backend', DEFAULT_BACKEND)))
        sealer = getattr(row.load_adapter(), 'seal_constructor',
                         SealedArwenConstructorV841.from_mapping)
    audit = sealer(constructor)
    receipt = dict(host['constructor_receipt'])
    carried = dict(source_p_top_pa=values[0], target_init_derived_p_top_pa=derived,
                   difference_pa=values[0]-derived, source_classes=len(values),
                   policy='carry original global physics boundary before constructor sealing')
    receipt.update(p_top_pa=values[0], p_top_policy=carried['policy'], moving_top_pressure=carried)
    receipt['sealed_host_contract_audit'] = dict(
        receipt['sealed_host_contract_audit'], **dict(audit.receipt()))
    return dict(host, constructor_values=constructor, constructor_receipt=receipt)


def resume_same_mesh_host(host, payload):
    from hexcore.state import PrognosticState
    from hexcore.driver import DrySavedDiagnostics
    prepared = host['prepared']
    if tuple(payload['scalar_names']) != tuple(host['scalar_names']):
        raise ValueError('moving carrier species differ from the running physics')
    # A same-mesh identity control must never admit a merely equal-size mesh.
    before = payload['mesh'].arrays
    after = prepared.mesh.arrays
    for name in ('areaCell', 'cellsOnEdge', 'edgesOnCell', 'xCell', 'yCell', 'zCell'):
        if not np.array_equal(before[name], after[name]):
            raise ValueError(f'same-mesh restart geometry differs: {name}')
    values = payload['state']
    state = PrognosticState(*(values[name] for name in ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars')),
                            time_seconds=float(payload['time_seconds']))
    saved = DrySavedDiagnostics(**{name: values[name] for name in DrySavedDiagnostics.__slots__})
    state.validate(n_cells=state.rho.shape[1], n_edges=state.rho_u.shape[1], n_vert_levels=state.rho.shape[0])
    saved.validate(state.rho.shape, state.rho.dtype, state.rho_u.shape[1])
    selected = replace(prepared, state=state, saved_diagnostics=saved)
    for name in ('advection_coefficients', 'expected_execution_fingerprint'):
        object.__setattr__(selected, name, getattr(prepared, name))
    return dict(host, prepared=selected)


def identify_source_geometry(mesh, source, target_radius):
    """Bind generator steradians to the exact prepared physical FP32 metrics."""
    stored=mesh.arrays
    coordinates=np.stack([stored[key] for key in ('xCell','yCell','zCell')],axis=-1)
    coordinates=coordinates/np.linalg.norm(coordinates,axis=-1,keepdims=True)
    radius=float(mesh.attrs['sphere_radius'])
    if radius!=float(target_radius):
        raise ValueError('moving forecast earth radii differ')
    scale=(radius/source.radius_m)**2
    if not np.array_equal(stored['areaCell'],(source.area*scale).astype(stored['areaCell'].dtype)) or not np.allclose(coordinates,source.centers,rtol=0,atol=4e-16):
        raise ValueError('source grid does not identify the stored moving boundary')
    return radius,scale


def resume_remapped_host(host, payload, *, source_grid, target_grid, operator_path, receipt_path):
    from .moving_remap import SphericalCells, ConservativeMap
    from .moving_state import remap_atmosphere
    began=time.perf_counter()
    print('[moving-remap] reading and identifying geometry',flush=True)
    source, target = SphericalCells.read(source_grid), SphericalCells.read(target_grid)
    radius,source_scale=identify_source_geometry(payload['mesh'],source,host['prepared'].mesh.attrs['sphere_radius'])
    target_scale=(radius/target.radius_m)**2
    weights=ConservativeMap.load(operator_path,source=source,target=target)
    if source_scale!=target_scale:
        raise ValueError('source and target operator spheres use different area units')
    # Operator files remain bound to the exact generator geometry. Runtime
    # integral receipts use the prepared dynamics sphere, not unit steradians.
    weights=ConservativeMap(weights.overlap*source_scale,weights.source_area*source_scale,
                            weights.target_area*target_scale,
                            dict(weights.receipt,effective_area_radius_m=radius,
                                 generator_area_scale=source_scale))
    geometry_seconds=time.perf_counter()-began
    print('[moving-remap] reconciling prepared physical areas',flush=True)
    stage=time.perf_counter()
    weights=weights.with_physical_areas(payload['mesh'].arrays['areaCell'],
                                        host['prepared'].mesh.arrays['areaCell'])
    area_seconds=time.perf_counter()-stage
    import os
    height_mode=os.environ.get('GPUWM_HEX_MOVING_HEIGHT_REMAP')
    if height_mode in ('1','absolute'):
        from .moving_vertical import HeightConservativeMap
        print('[moving-remap] constructing experimental physical-height integrals', flush=True)
        stage=time.perf_counter()
        weights=HeightConservativeMap(weights,payload['vertical'].zgrid,
            host['prepared'].vertical.zgrid,payload['vertical'].dzw,host['prepared'].vertical.dzw,
            endpoint_alignment=height_mode!='absolute')
        height_seconds=time.perf_counter()-stage
    else:
        height_seconds=0.
    print('[moving-remap] remapping atmospheric mass products and momentum',flush=True)
    stage=time.perf_counter()
    state,saved,receipt=remap_atmosphere(payload,weights,source_mesh=payload['mesh'],
                                       target_prepared=host['prepared'],config=host['config'])
    selected=replace(host['prepared'],state=state,saved_diagnostics=saved)
    for name in ('advection_coefficients','expected_execution_fingerprint'):
        object.__setattr__(selected,name,getattr(host['prepared'],name))
    # Retain the list itself: physics-rate carries append stages during stack construction.
    receipt['height_execution_timings'] = getattr(weights, 'execution_timings', [])
    receipt.update(operator=weights.receipt,time_seconds=payload['time_seconds'],
        geometry_read_identify_seconds=geometry_seconds,physical_area_reconciliation_seconds=area_seconds,
        physical_height_stencil_seconds=height_seconds,
        atmosphere_remap_seconds=time.perf_counter()-stage,total_atmospheric_transfer_seconds=time.perf_counter()-began)
    print('[moving-remap] atmospheric transfer complete',flush=True)
    Path(receipt_path).write_text(json.dumps(receipt,indent=2)+'\n')
    return dict(host,prepared=selected),weights,receipt


def restore_step_carriers(payload, stacks, devices=None):
    """Upload the carried surface updates and GF feedback, each onto its stack's card.

    ``devices`` (one per stack) is given by the multi-card executor, whose blocks live
    on different cards; without it every stack is on the current card (one card).
    """
    import cupy as cp
    from contextlib import nullcontext
    from hexcore.cuda_driver import CudaV841GfDynamicsTendencies
    if devices is not None and len(devices) != len(stacks):
        raise ValueError('one card per stack')
    if len(payload['previous']) != len(stacks) or len(payload['forcing']) != len(stacks):
        raise ValueError('moving step carriers do not match the partition: '
                         f"{len(payload['previous'])} carried, {len(stacks)} stacks")
    previous, forcing = [], []
    for index, (old, item, stack) in enumerate(zip(payload['previous'], payload['forcing'], stacks)):
        with (nullcontext() if devices is None else cp.cuda.Device(int(devices[index]))):
            previous.append(None if old is None else {key: cp.asarray(value, order='C') for key, value in old.items()})
            if item is None:
                forcing.append(None)
            else:
                value = CudaV841GfDynamicsTendencies(
                    rthdynten=cp.asarray(item['rthdynten'], order='C'),
                    rqvdynten=cp.asarray(item['rqvdynten'], order='C'),
                    time_seconds=float(item['time_seconds']))
                value.validate(cp=cp, n_vert_levels=stack['driver'].atmosphere.state.rho.shape[0],
                               n_cells=stack['driver'].horizontal.ncells)
                forcing.append(value)
            if devices is not None:
                cp.cuda.get_current_stream().synchronize()
    return previous, forcing
