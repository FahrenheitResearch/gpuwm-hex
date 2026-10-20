"""Force a native regional hex nest from the analysed parent forecast trajectory.

``python -m hexcore.nest_boundary --source-grid ... --source-static ... --analysis ...
--target-grid ... --target-static ... --target-init ... --operator ... --output DIR
--history H1 [--history H2 ...]``

Each endpoint uses the same conservative state transfer and mass-continuity
projection as nest initialization (:mod:`hexcore.analysis_remap`). All six water
species and physical w are published to the native LBC consumer with checked
atomic readiness markers.  The series' ``interval_seconds`` is the parent's
history interval (:func:`hexcore.lbc_stream.series_document` reads it), so a
15-minute parent history forces the nest every 15 minutes.  ``--lbc-dir DIR``
then starts the nest (``gpuwm-hex forecast --analysis ... --remap ROW``).
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
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


def produce(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    start = valid_time(args.analysis)
    manifest = getattr(args, 'parent_manifest', None)
    expected = {}
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
        timeline = [(start, args.analysis)] + [(valid_time(path), path) for path in args.history]
    timeline.sort(key=lambda item: item[0])
    if len(timeline) < 2 or timeline[0][0] != start:
        raise ValueError('boundary trajectory must start with the analysis and include a later parent forecast')
    interval = (timeline[1][0] - start).total_seconds()
    if interval <= 0 or interval != int(interval) or any((b[0]-a[0]).total_seconds() != interval for a,b in zip(timeline,timeline[1:])):
        raise ValueError('parent trajectory has duplicate or missing boundary endpoints')
    source = transfer.load(args.source_grid, args.source_static, args.analysis, transfer.SPECIES)
    target = transfer.load(args.target_grid, args.target_static, args.target_init, transfer.SPECIES)
    if not target.mesh.is_regional:
        raise ValueError('nested boundary generation requires a regional target mesh')
    sg, tg = SphericalCells.read(args.source_grid), SphericalCells.read(args.target_grid)
    output.mkdir(parents=True)
    atomic_json(output/'lbc-series.json', dict(schema=SERIES, interval_seconds=int(interval),
        valid_times=[moment.strftime(FORMAT) for moment,_ in timeline],
        target_grid_sha256=sha256(args.target_grid), target_init_sha256=sha256(args.target_init)))
    records = []
    try:
        for moment, path in timeline:
            if path in expected:
                wait_parent(expected[path], args.wait_parent_seconds)
                if valid_time(path) != moment:
                    raise ValueError(f'parent clock differs from source manifest: {path}')
            parent = source if path == args.analysis else load_history(path, args.analysis, source, transfer.SPECIES)
            state, saved, receipt = transfer.remap(parent, target, sg, tg, args.operator, transfer.SPECIES)
            provenance = dict(parent=str(path), parent_sha256=sha256(path), remap=receipt,
                              parent_representation='native analysis' if path==args.analysis else 'public history rehydrated by native cold loader')
            document = publish_frame(output, moment, state, saved, target.vertical, transfer.SPECIES, provenance)
            records.append(document)
            print('READY',output/document['path'],flush=True)
        atomic_json(output/'lbc-door-receipt.json', dict(schema='hex.parent-lbc.v1', files=records,
                    interval_seconds=int(interval), species=list(transfer.SPECIES)))
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
    args = parser.parse_args()
    produce(args)


if __name__ == '__main__':
    main()
