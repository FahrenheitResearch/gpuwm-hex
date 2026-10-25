"""Force a native regional hex nest from the analysed parent forecast trajectory.

``python -m hexcore.nest_boundary --source-grid ... --source-static ... --analysis ...
--target-grid ... --target-static ... --target-init ... --operator ... --output DIR
--history H1 [--history H2 ...]``

Each endpoint uses the same conservative state transfer and mass-continuity
projection as nest initialization (:mod:`hexcore.analysis_remap`). All six water
species, the region's number species (carried or diagnosed, :mod:`hexcore.handover_numbers`) and physical w are published to the native LBC consumer with checked
atomic readiness markers.  The series' ``interval_seconds`` is the parent's
history interval (:func:`hexcore.lbc_stream.series_document` reads it), so a
15-minute parent history forces the nest every 15 minutes.  ``--lbc-dir DIR``
then starts the nest (``gpuwm-hex forecast --analysis ... --remap ROW``).
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset

from hexcore.lbc import LBC_REQUIRED_VARIABLES, LBC_OPTIONAL_SCALAR_VARIABLES
from hexcore.lbc_stream import FORMAT, READY, SERIES, atomic_json, file_name, sha256
from hexcore.moving_remap import SphericalCells
from hexcore import analysis_remap as transfer


def valid_time(path):
    """Use the record clock; older public history capsules encode it in the name."""
    with Dataset(path) as dataset:
        if 'xtime' in dataset.variables:
            value = np.asarray(dataset['xtime'][0]).tobytes().decode('ascii').strip('\x00 ')
            return datetime.strptime(value, FORMAT)
    match = re.fullmatch(r'(?:cuda-)?history\.(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})\.nc', Path(path).name)
    if not match:
        raise ValueError(f'parent history lacks a record clock or a recognized timestamped name: {path}')
    return datetime.strptime(match.group(1), '%Y-%m-%d_%H.%M.%S')


def load_history(path, reference_path, geometry, species):
    """Rehydrate public history on the same initialized parent grid."""
    from hexcore.driver import load_mpas_initial_state
    with Dataset(path) as dataset, Dataset(reference_path) as reference:
        for name in ('indexToCellID', 'indexToEdgeID'):
            if name not in dataset.variables or not np.array_equal(dataset[name][:], reference[name][:]):
                raise ValueError(f'parent history {name} differs from the analysis geometry: {path}')
        velocity = 'normal_u' if 'normal_u' in dataset.variables else 'u'
    state, reference, saved = load_mpas_initial_state(path, geometry.mesh, geometry.vertical,
        scalar_names=species, reference_path=reference_path, terrain_metrics=geometry.terrain_metrics,
        allow_regional_sentinels=bool(geometry.mesh.is_regional), cold_start=True,
        return_saved_diagnostics=True, normal_velocity_name=velocity)
    return SimpleNamespace(mesh=geometry.mesh, vertical=geometry.vertical, terrain_metrics=geometry.terrain_metrics,
                            state=state, reference=reference, saved_diagnostics=saved)


def publish_frame(output, moment, state, saved, vertical, species, provenance):
    stamp = moment.strftime(FORMAT)
    destination = Path(output) / file_name(stamp)
    temporary = destination.with_suffix('.nc.partial')
    marker = destination.with_name(destination.name + '.ready.json')
    if any(path.exists() for path in (destination, temporary, marker)):
        raise FileExistsError(destination)
    values = transfer.native_fields(state, saved, vertical, species)
    # Every required variable, and the optional scalars this transfer carries: the lbc stream's
    # optional packages (number concentrations, aerosols, tke) are present only when allocated.
    schema = dict(LBC_REQUIRED_VARIABLES, **{name: dims for name, dims in LBC_OPTIONAL_SCALAR_VARIABLES.items()
                                              if name[4:] in values})
    with Dataset(temporary, 'w', format='NETCDF3_64BIT_DATA') as dataset:
        for name, size in [('Time', None), ('StrLen', 64), ('nCells', state.rho.shape[1]),
                           ('nEdges', state.rho_u.shape[1]), ('nVertLevels', state.rho.shape[0]),
                           ('nVertLevelsP1', state.rho.shape[0]+1)]:
            dataset.createDimension(name, size)
        dataset.createVariable('xtime', 'S1', ('Time', 'StrLen'))[0] = np.frombuffer(stamp.ljust(64).encode(), dtype='S1')
        for name, dims in schema.items():
            dataset.createVariable(name, 'f4', dims)[0] = values[name[4:]].T
        dataset.setncattr('source', 'conservative native hex parent trajectory')
        dataset.setncattr('parent_state_sha256', provenance['parent_sha256'])
    temporary.replace(destination)
    document = dict(schema=READY, valid_time=stamp, path=destination.name,
                    bytes=destination.stat().st_size, sha256=sha256(destination), **provenance)
    atomic_json(marker, document)
    return document


def wait_parent(record, timeout):
    """Admit a streamed parent only after its source size and hash agree."""
    path = Path(record['path'])
    if not isinstance(record['bytes'], int) or record['bytes'] <= 0:
        raise ValueError('parent manifest requires a positive byte count')
    if not re.fullmatch('[0-9a-f]{64}', record['sha256']):
        raise ValueError('parent manifest requires a SHA-256 digest')
    started = time.monotonic()
    while True:
        if path.is_file():
            before = path.stat()
            if before.st_size > record['bytes']:
                raise ValueError(f'parent exceeds declared complete size: {path}')
            if before.st_size == record['bytes']:
                actual = sha256(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f'parent changed during admission: {path}')
                if actual != record['sha256']:
                    raise ValueError(f'parent SHA-256 differs from source manifest: {path}')
                return
        if time.monotonic()-started >= timeout:
            raise TimeoutError(f'parent endpoint is incomplete: {path}')
        time.sleep(1.)


# THE BREAKAGE THIS PREVENTS (hex fast start, 2026-10-23): the boundary series of a 937.5 m,
# 96,912-cell cull from the 773,478-cell 80->5 km VR parent took ~62 s per frame on one core
# (25 frames: ~26 min on a 16-core box), because every frame cold-loaded and remapped the WHOLE
# parent, of which the target can reach 2 %.  The parent is now cut to a cap around the target
# (grid, static and analysis by rw_mpas_mesh --cull-parent; HDF5 history frames by position, all
# cores) and the frames run concurrently.
SUBSET_MARGIN_KM = 120.0
from hexcore.netcdf_lock import NETCDF as _NETCDF


EARTH_RADIUS_KM = 6371.229


def target_cap(grid, margin_km=SUBSET_MARGIN_KM):
    """The cap (centre, radius) that holds every target cell plus the margin."""
    with Dataset(grid) as d:
        lat = np.asarray(d['latCell'][:], dtype=np.float64)
        lon = np.asarray(d['lonCell'][:], dtype=np.float64)
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    centre = xyz.mean(axis=0)
    centre /= np.linalg.norm(centre)
    radius = float(np.arccos(np.clip(xyz @ centre, -1, 1)).max()) * EARTH_RADIUS_KM
    clat, clon = np.degrees(np.arcsin(centre[2])), np.degrees(np.arctan2(centre[1], centre[0]))
    return dict(kind='cap', center_deg=[round(float(clat), 6), round(float(clon), 6)],
                radius_km=round(radius + float(margin_km), 3))


def _position_index(culled, parent):
    """Parent index of every culled cell, edge and vertex (the cull reindexes ids 1..n)."""
    out = {}
    with Dataset(culled) as g, Dataset(parent) as q:
        for dim, loc in (('nCells', 'Cell'), ('nEdges', 'Edge'), ('nVertices', 'Vertex')):
            def keys(d):
                xyz = np.stack([np.asarray(d[c + loc][:], dtype=np.float64) for c in 'xyz'], axis=1)
                return np.ascontiguousarray(xyz).view(np.dtype((np.void, 24))).ravel()
            pk, ck = keys(q), keys(g)
            order = np.argsort(pk)
            index = order[np.minimum(np.searchsorted(pk[order], ck), len(pk) - 1)]
            if not np.array_equal(pk[index], ck):
                raise ValueError(f'parent subset: {dim} of {culled} has entries with no exact parent position')
            out[dim] = index
    return out


def _cut_history(job):
    source, destination, index = job
    temporary = Path(str(destination) + '.partial')
    with Dataset(source) as s, Dataset(temporary, 'w', format='NETCDF3_64BIT_DATA') as d:
        for name, dim in s.dimensions.items():
            d.createDimension(name, len(index[name]) if name in index else (None if dim.isunlimited() else len(dim)))
        d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
        for name, var in s.variables.items():
            dtype = np.int32 if var.dtype == np.int64 else var.dtype
            v = d.createVariable(name, dtype if dtype != str else 'S1', var.dimensions)
            v.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'})
            data = var[:]
            for axis, dimension in enumerate(var.dimensions):
                if dimension in index:
                    data = np.take(data, index[dimension], axis=axis)
            if name in ('indexToCellID', 'indexToEdgeID', 'indexToVertexID'):
                data = np.arange(1, data.shape[-1] + 1, dtype=np.int32)  # the cull's contiguous ids
            v[:] = data
    temporary.replace(destination)
    return destination


def parent_subset(args, histories, workdir, jobs):
    """Cut the parent to the target's cap; return (grid, static, analysis, histories, receipt)."""
    import subprocess
    from concurrent.futures import ProcessPoolExecutor
    from hexcore.engines import MESH, resolve
    started = time.monotonic()
    workdir.mkdir(parents=True, exist_ok=True)
    region = target_cap(args.target_grid, args.parent_subset_margin_km)
    (workdir / 'region.json').write_text(json.dumps(region))
    engine = resolve(MESH, None)
    grid = workdir / 'parent.grid.nc'
    cut = {'grid': grid, 'static': workdir / 'parent.static.nc', 'analysis': workdir / 'parent.analysis.nc'}
    for role, source in (('grid', args.source_grid), ('static', args.source_static), ('analysis', args.analysis)):
        argv = [str(engine), '--cull-parent', str(source), '--region', str(workdir / 'region.json'),
                '--out', str(cut[role]), '--clobber']
        if role != 'grid':
            argv += ['--cull-provenance-grid', str(grid)]
        subprocess.run(argv, check=True, stdout=subprocess.DEVNULL)
    index = _position_index(grid, args.source_grid)
    folder = workdir / 'history'
    folder.mkdir(exist_ok=True)
    work = [(Path(h), folder / Path(h).name, index) for h in histories]
    with ProcessPoolExecutor(max_workers=max(1, min(len(work), jobs))) as pool:
        cut_histories = list(pool.map(_cut_history, work))
    receipt = dict(region=region, cells=int(len(index['nCells'])), frames=len(cut_histories),
                   seconds=round(time.monotonic() - started, 2))
    return cut['grid'], cut['static'], cut['analysis'], cut_histories, receipt


def produce(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    start = valid_time(args.analysis)
    manifest = getattr(args, 'parent_manifest', None)
    expected = {}
    subset_receipt = None
    if manifest is not None:
        if args.history:
            raise ValueError('choose --parent-manifest or --history')
        document = json.loads(manifest.read_text())
        if document.get('schema') != 'hex.parent-history.v1':
            raise ValueError('unknown parent history manifest')
        entries = document['files']
        expected = {Path(row['path']): row for row in entries}
        if len(expected) != len(entries):
            raise ValueError('parent history manifest repeats a file')
        timeline = [(start, args.analysis)] + [(datetime.strptime(row['valid_time'], FORMAT), Path(row['path'])) for row in entries]
    else:
        if getattr(args, 'parent_subset', 'off') == 'auto':
            # A history frame keeps its clock in xtime, so the cut frames are timed like the originals.
            grid, static, analysis, histories, subset_receipt = parent_subset(
                args, args.history, output.with_name(output.name + '.parent-subset'), args.jobs)
            args.source_grid, args.source_static, args.analysis, args.history = grid, static, analysis, histories
            print('PARENT SUBSET', json.dumps(subset_receipt), flush=True)
        timeline = [(start, args.analysis)] + [(valid_time(path), path) for path in args.history]
    timeline.sort(key=lambda item: item[0])
    if len(timeline) < 2 or timeline[0][0] != start:
        raise ValueError('boundary trajectory must start with the analysis and include a later parent forecast')
    interval = (timeline[1][0] - start).total_seconds()
    if interval <= 0 or interval != int(interval) or any((b[0]-a[0]).total_seconds() != interval for a,b in zip(timeline,timeline[1:])):
        raise ValueError('parent trajectory has duplicate or missing boundary endpoints')
    # The region's number species (hexcore.handover_numbers): carried from every frame that publishes them, else
    # diagnosed from the frame's mass (number_scheme), so each boundary file carries lbc_ni/lbc_nr/lbc_nc with the
    # masses instead of handing mass in with no number.  A number neither carried nor diagnosed is not published
    # (a +0 plane would drive number 0 under the driven mass), and the published ones stay the contiguous prefix the
    # door drives.  None given, or none carried or diagnosed: six masses.
    from hexcore import handover_numbers
    numbers = tuple(getattr(args, 'numbers', ()) or ())
    number_scheme = getattr(args, 'number_scheme', None)
    xland = handover_numbers.read_xland(args.analysis) if numbers else None

    def load_numbers(path, loaded_species, loader, wanted):
        published = handover_numbers.frame_number_species(path) if wanted else ()
        species = loaded_species + tuple(name for name in wanted if name in published)
        loaded = loader(species)
        if not wanted:
            return loaded, species, None
        species, receipt = handover_numbers.complete_state_numbers(
            loaded, species, wanted, scheme=number_scheme, xland=xland, prefix=True)
        return loaded, species, receipt

    source, species, source_numbers = load_numbers(
        args.analysis, transfer.SPECIES,
        lambda names: transfer.load(args.source_grid, args.source_static, args.analysis, names), numbers)
    # The analysis frame fixes the series' numbers, so every boundary file has one species order.
    numbers = tuple(source_numbers['published']) if source_numbers is not None else ()
    target = transfer.load(args.target_grid, args.target_static, args.target_init, transfer.SPECIES)
    if not target.mesh.is_regional:
        raise ValueError('nested boundary generation requires a regional target mesh')
    sg, tg = SphericalCells.read(args.source_grid), SphericalCells.read(args.target_grid)
    output.mkdir(parents=True)
    atomic_json(output/'lbc-series.json', dict(schema=SERIES, interval_seconds=int(interval),
        valid_times=[moment.strftime(FORMAT) for moment,_ in timeline],
        target_grid_sha256=sha256(args.target_grid), target_init_sha256=sha256(args.target_init)))
    records = []

    def frame(item):
        moment, path = item
        if path in expected:
            wait_parent(expected[path], args.wait_parent_seconds)
            if valid_time(path) != moment:
                raise ValueError(f'parent clock differs from source manifest: {path}')
        if path == args.analysis:
            parent, names, number_receipt = source, species, source_numbers
        else:
            with _NETCDF:
                parent, names, number_receipt = load_numbers(
                    path, transfer.SPECIES, lambda loaded: load_history(path, args.analysis, source, loaded),
                    numbers)
        state, saved, receipt = transfer.remap(parent, target, sg, tg, args.operator, names)
        provenance = dict(parent=str(path), parent_sha256=sha256(path), remap=receipt,
                          parent_representation='native analysis' if path==args.analysis else 'public history rehydrated by native cold loader')
        if number_receipt is not None:
            provenance['numbers'] = number_receipt
        with _NETCDF:
            document = publish_frame(output, moment, state, saved, target.vertical, names, provenance)
        print('READY',output/document['path'],flush=True)
        return document

    try:
        # The first frame builds (or reads) the remap operator; the rest share it and run concurrently,
        # each published with its own ready marker, which is all the consumer orders by.  A streamed
        # parent (--parent-manifest) stays in order, frame by frame, as its endpoints arrive.
        records.append(frame(timeline[0]))
        rest = timeline[1:]
        jobs = int(getattr(args, 'jobs', 1) or 1)
        if jobs > 1 and not expected and rest:
            # Threads with every netCDF4/HDF5 call under _NETCDF: the netCDF C library is not
            # thread-safe (unlocked threads segfaulted, exit 139) and forked workers deadlocked on
            # locks held at fork; the remap compute, which dominates, runs concurrently.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(jobs, len(rest)), thread_name_prefix='lbc-frame') as pool:
                records.extend(pool.map(frame, rest))
        else:
            records.extend(frame(item) for item in rest)
        atomic_json(output/'lbc-door-receipt.json', dict(schema='hex.parent-lbc.v1', files=records,
                    interval_seconds=int(interval), species=list(species),
                    parent_subset=subset_receipt))
    except BaseException as error:
        atomic_json(output/'lbc-failed.json', dict(error=str(error)))
        raise
    return records


def main():
    parser = argparse.ArgumentParser(prog="python -m hexcore.nest_boundary", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('source-grid', 'source-static', 'analysis', 'target-grid', 'target-static', 'target-init', 'operator', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--history', action='append', default=[], type=Path, help='later timestamped parent history; repeat in any order')
    parser.add_argument('--parent-manifest', type=Path, help='source hashes, sizes and clocks for endpoints arriving during the forecast')
    parser.add_argument('--wait-parent-seconds', type=float, default=3600.)
    parser.add_argument('--parent-subset', choices=('auto', 'off'), default='auto',
                        help='cut the parent to a cap around the target before remapping (--history mode)')
    parser.add_argument('--parent-subset-margin-km', type=float, default=SUBSET_MARGIN_KM)
    parser.add_argument('--jobs', type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                        help='boundary frames remapped at once')
    args = parser.parse_args()
    produce(args)


if __name__ == '__main__':
    main()
