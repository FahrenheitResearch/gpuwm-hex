"""Optional test-only native boundary comparison in a full artifact forecast.

Rust reads native ORCL data; NumPy only asserts/counts bitwise test differences.
No test value is copied into the forecast or changes its scientific execution.
"""
import os
if os.environ.get('HEXCORE_IAU_BOUNDARY_PROBE'):
    import ctypes
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import hexcore.iau_v842 as iau

    manifest = Path(os.environ['HEXCORE_IAU_BOUNDARY_PROBE'])
    metadata = json.loads(manifest.read_text())
    reader = ctypes.CDLL(os.environ['HEXCORE_IAU_ORACLE_READER'])
    original = iau.atm_add_tend_anal_incr
    from hexcore.mpasphys.runtime_v842 import ResidentNativeCoupling
    stage_original = ResidentNativeCoupling.stage
    resident_ids = {}

    def stage_with_ids(self, **kwargs):
        if not resident_ids:
            for entity in ('Cell','Edge'):
                resident_ids[entity] = self.session.native.get_field(
                    'mesh','indexTo'+entity+'ID')[:-1].copy()
        return stage_original(self, **kwargs)

    ResidentNativeCoupling.stage = stage_with_ids

    def compare(name, device, phase):
        filename = metadata['fields' if phase == 'pre' else 'native_post_fields'][name]
        path = manifest.parent/filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['record_sha256'][filename]:
            raise ValueError('native boundary capture hash differs')
        read = reader.iau_oracle_read if phase == 'pre' else reader.iau_oracle_read_post
        read.argtypes = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
        read.restype = ctypes.c_ssize_t
        h = (ctypes.c_int32*21)()
        count = read(os.fsencode(path), h, None, None, 0)
        if count < 0:
            raise ValueError('Rust rejected native boundary record')
        shape = tuple(h[3:3+h[2]])
        f = np.empty(shape, dtype=np.float32, order='F')
        c = np.empty(shape, dtype=np.float32, order='C')
        if read(os.fsencode(path), h, f.ctypes.data, c.ctypes.data, count) != count:
            raise ValueError('native boundary record changed')
        actual = device.get()
        if actual.shape != c[..., :-1].shape:
            raise ValueError(f'{name}: resident/native shape differs')
        # f1's edge captures currently have no entity IDs. They cannot establish
        # numerical identity; preserve this gap without stopping the forecast.
        if h[16] == 0 and 'native_edge_ids' not in metadata:
            return dict(words=int(actual.size), differing_words=None,
                        status='uncompared-native-global-ids-absent',
                        native_entity_axis=int(h[17]))
        # The Rust decoder aligns the native local pool using authentic IDs.
        expected = np.empty(actual.shape, dtype=np.float32, order='C')
        ids = resident_ids['Edge' if name in ('rho_edge','u_amb','tend_ru') else 'Cell']
        aligned = reader.iau_oracle_read_ids
        aligned.argtypes = (ctypes.c_char_p, ctypes.c_int32, ctypes.c_void_p,
                            ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
        aligned.restype = ctypes.c_ssize_t
        args = (os.fsencode(path), int(phase == 'post'), ids.ctypes.data,
                ids.size, expected.ctypes.data, expected.size)
        if h[16] == 0:
            id_name = metadata['native_edge_ids']
            id_path = manifest.parent/id_name
            if hashlib.sha256(id_path.read_bytes()).hexdigest() != metadata['record_sha256'][id_name]:
                raise ValueError('native edge ID record hash differs')
            aligned = reader.iau_oracle_read_external_ids
            aligned.argtypes = (ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p,
                                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
            aligned.restype = ctypes.c_ssize_t
            args = (args[0], args[1], os.fsencode(id_path), *args[2:])
        if aligned(*args) != expected.size:
            raise ValueError(f'{name}: Rust rejected native/resident ID alignment')
        bad = actual.view(np.uint32) != expected.view(np.uint32)
        indices = np.argwhere(bad)
        first = tuple(int(x) for x in indices[0]) if indices.size else None
        return dict(words=int(actual.size), differing_words=int(np.count_nonzero(bad)),
                    first_index=first,
                    native_word=int(expected.view(np.uint32)[first]) if first else None,
                    hex_word=int(actual.view(np.uint32)[first]) if first else None)

    def probed(configs, structs, itimestep, dt, tend_ru, tend_rtheta, tend_rho, **kwargs):
        if itimestep != metadata['itimestep']:
            return original(configs, structs, itimestep, dt, tend_ru, tend_rtheta, tend_rho, **kwargs)
        fields = dict(zz=structs['mesh']['zz'], rho_zz=structs['state']['rho_zz'],
                      theta_m=structs['state']['theta_m'], scalars=structs['state']['scalars'],
                      rho_edge=structs['diag']['rho_edge'], u_amb=structs['tend_iau']['u'],
                      rho_amb=structs['tend_iau']['rho'], theta_amb=structs['tend_iau']['theta'],
                      scalars_amb=structs['tend_iau']['scalars'], tend_ru=tend_ru,
                      tend_rtheta=tend_rtheta, tend_rho=tend_rho,
                      tend_scalars=structs['tend']['scalars_tend'])
        result = dict(acceptance=False, controls=dict(step=itimestep, dt=dt,
                       window=configs.config_IAU_window_length_s,
                       **{k:structs['state'][k] for k in ('moist_start','moist_end','index_qv')}),
                       pre={name:compare(name,value,'pre') for name,value in fields.items()})
        original(configs, structs, itimestep, dt, tend_ru, tend_rtheta, tend_rho, **kwargs)
        result['post'] = {name:compare(name,fields[name],'post')
                         for name in ('tend_ru','tend_rtheta','tend_rho','tend_scalars')}
        Path(os.environ['HEXCORE_IAU_BOUNDARY_PROBE_OUT']).write_text(json.dumps(result, indent=2)+'\n')

    iau.atm_add_tend_anal_incr = probed
