"""Observe resident carriers during an unchanged real forecast.

MPAS source: src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3088-3185;
src/core_atmosphere/physics/mpas_atmphys_interface.F:909-933. Rust compares original f1 words;
Python only copies seam buffers. This is localization, not forecast acceptance.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--capture-root', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--checkout-commit', required=True)
    p.add_argument('model_command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.model_command
    if command[:1] == ['--']:
        command = command[1:]
    if command[:2] != ['gpuwm-hex', 'forecast']:
        p.error('an actual gpuwm-hex forecast command is required')
    from hexcore.mpasphys.runtime_v842 import ResidentNativeCoupling
    from hexcore import cli
    rust = ct.CDLL(str(a.artifact.resolve()))
    compare = rust.compare_abi_record_v2
    compare.argtypes = [ct.c_char_p, ct.c_void_p, ct.c_size_t, ct.c_void_p,
                        ct.c_size_t, ct.c_char_p, ct.c_void_p, ct.c_size_t,
                        ct.c_size_t, ct.c_void_p, ct.c_size_t]
    compare.restype = ct.c_ssize_t
    rows = []
    prefix = 'boundary__driver_microphysics.s0000000001.e0000000001.r000000.post.'
    original = ResidentNativeCoupling.post_transport

    def observe(coupling, state, saved, phase):
        # Native interface carrier ownership: mpas_atmphys_interface.F:909-933.
        native, layout = coupling.session.native, coupling.session.layout
        bindings = [('rho', state.rho, 'state', 'rho_zz', 2),
                    ('scalars', state.scalars, 'state', 'scalars', 2),
                    ('rho_u', state.rho_u, 'diag', 'ru', 1),
                    ('rho_w', state.rho_w, 'diag', 'rw', 1),
                    ('normal_velocity', saved.normal_velocity, 'state', 'u', 2),
                    ('vertical_velocity', saved.vertical_velocity, 'state', 'w', 2),
                    ('theta_m', saved.theta_m, 'state', 'theta_m', 2),
                    ('rho_p', saved.density_perturbation, 'diag', 'rho_p', 1),
                    ('rtheta_p', saved.rho_theta_perturbation, 'diag', 'rtheta_p', 1),
                    ('exner', saved.exner, 'diag', 'exner', 1),
                    ('pressure_p', saved.pressure_perturbation, 'diag', 'pressure_p', 1)]
        for label, device, pool, name, level in bindings:
            try:
                dtype, shape, levels = native._fields[pool, name]
                path = a.capture_root / (prefix + pool + '.' + name +
                                        (f'.t{level}' if levels > 1 else '') + '.bin')
                if not path.is_file():
                    raise ValueError('original native capture is absent')
                field = next(r for r in layout.fields if r['pool'] == pool and r['pool_name'] == name)
                axis = next(i for i, d in enumerate(field['dimensions']) if d in {'nCells', 'nEdges', 'nVertices'})
                entity = field['dimensions'][axis]
                idname = {'nCells':'indexToCellID', 'nEdges':'indexToEdgeID', 'nVertices':'indexToVertexID'}[entity]
                idfile = a.capture_root / (prefix + 'mesh.' + idname + '.bin')
                if not idfile.is_file():
                    idfile = a.capture_root / ('boundary__driver_cloudiness.s0000000001.e0000000001.r000000.post.mesh.' + idname + '.bin')
                ids = native.get_field('mesh', idname)
                native_buffer = native.get_field(pool, name, level=level)
                device_buffer = native_buffer.copy(order='F')
                host = device.get()
                if shape[:-1] != host.shape[:-1] or shape[-1] != host.shape[-1] + 1:
                    raise ValueError('resident and native carrier extent differs')
                # Native dummy is copied separately and labelled, never accepted
                # as evidence of a resident dummy which does not exist.
                device_buffer[..., :-1] = host
                dimensions = (ct.c_int32 * len(shape))(*shape)
                result = {'carrier': label, 'pool': pool, 'field': name, 'level': level,
                          'phase': phase, 'shape': shape, 'reference': str(path),
                          'resident_dummy': 'absent; comparison uses native dummy copy'}
                for origin, buffer in [('resident', device_buffer), ('pool', native_buffer)]:
                    output = ct.create_string_buffer(8192)
                    length = compare(str(path).encode(), buffer.ctypes.data, buffer.nbytes,
                                     dimensions, len(shape), str(idfile).encode(), ids.ctypes.data,
                                     layout.dimensions[entity], axis, output, len(output))
                    if not 0 < length <= len(output):
                        raise ValueError(f'Rust comparison returned {length}')
                    result[origin] = json.loads(output.raw[:length])
                rows.append(result)
            except (ValueError, KeyError, StopIteration) as error:
                rows.append({'carrier': label, 'phase': phase, 'error': str(error)})
        a.receipt.write_text(json.dumps({'schema':'native-resident-carrier-observation-v1',
            'process_count':1, 'step':1, 'forecast_proof':False,
            'checkout_commit':a.checkout_commit,
            'library_sha256':native.build['library_sha256'],
            'comparator_sha256':hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
            'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'capture_manifest_sha256':hashlib.sha256((a.capture_root/'manifest.json').read_bytes()).hexdigest(),
            'fields':rows}, indent=2) + '\n')
        print('RESIDENT_CARRIER_OBSERVATION', phase, len(rows), flush=True)

    def post_transport(coupling, **kwargs):
        # MPAS time-step boundary: mpas_atm_time_integration.F:2770-2816.
        first = coupling.session.driver._pending[0] == 1
        if first:
            observe(coupling, kwargs['state'], kwargs['saved'], 'before_post_transport')
        result = original(coupling, **kwargs)
        if first:
            observe(coupling, kwargs['state'], kwargs['saved'], 'after_post_transport')
        return result

    ResidentNativeCoupling.post_transport = post_transport
    try:
        return cli.main(command[1:])
    finally:
        ResidentNativeCoupling.post_transport = original


if __name__ == '__main__':
    raise SystemExit(main())
