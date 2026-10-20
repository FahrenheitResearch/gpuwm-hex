"""Check the Thompson binder against actual f3 storage and f4 pool transfers.

This CPU ABI witness performs no forecast or CUDA arithmetic. The supplied
foundation sources and compiled library are hashed in its receipt. Constants
must come from the actual compiled export; absent exports remain a refusal.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from hexcore.mpasphys import Backend, SlotCall, StepFlags
from hexcore.mpasphys import thompson_v842 as port


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def replay(reference, native_source, provider_source, layout_source, library, receipt):
    native_module = module('thompson_native_abi_witness', native_source)
    device_module = module('hexcore.mpasphys._thompson_device_witness', provider_source)
    layout_module = module('hexcore.mpasphys._thompson_layout_witness', layout_source)
    registry = ET.fromstring(layout_module.registry_text(reference / 'src/core_atmosphere/Registry.xml'))
    nz, solve, cells = 3, 4, 6
    dimensions = dict(nVertLevels=nz, nVertLevelsP1=nz + 1, nCells=cells, nCellsSolve=solve)
    fields = []
    for struct in registry.findall('./var_struct'):
        for field in struct:
            if field.tag not in ('var', 'var_array'):
                continue
            row = dict(field.attrib)
            row.update(pool=struct.get('name'), pool_name=field.get('name_in_code', field.get('name')),
                       dimensions=[d for d in field.get('dimensions', '').split() if d != 'Time'])
            if field.tag == 'var_array':
                row['dimensions'].insert(0, 'num_' + field.get('name'))
                row['components'] = [c.get('name_in_code', c.get('name')) for c in field.findall('var')
                                     if not c.get('packages') or 'mp_thompson_in' in c.get('packages').split(';')]
            fields.append(row)
    scalar_row = next(r for r in fields if r['pool'] == 'state' and r['pool_name'] == 'scalars')
    components = scalar_row['components']
    dimensions['num_scalars'] = len(components)
    indices = {('state', 'index_' + n): components.index(n) + 1 for n in port.MODEL_ARGUMENTS[:8]}
    wanted = {'state': ('scalars', *port.MODEL_ARGUMENTS[8:]), 'mesh': port.MESH_ARGUMENTS,
              'diag': (*[n for n in port.DIAG_ARGUMENTS if n != 'rt_diabatic_tend'], 'surface_pressure', 'relhum'),
              'tend': ('rt_diabatic_tend', 'tend_sfc_pressure'),
              'diag_physics': ('rainprod', 'evapprod', 're_cloud', 're_ice', 're_snow', 'nt_c', 'mu_c',
                               *port.PRECIP_ARGUMENTS, 'i_rainnc', 'refl10cm', 'refl10cm_max',
                               'refl10cm_1km', 'refl10cm_1km_max'),
              'tend_physics': (*port.TENDENCY_ARGUMENTS, 'mp_top_level')}
    rows = [next(r for r in fields if r['pool'] == pool and r['pool_name'] == name)
            for pool, names in wanted.items() for name in names]
    layout = SimpleNamespace(pools=set(wanted), fields=rows, dimensions=dimensions,
                             indices=indices, configs={})
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (Path(port.__file__), Path(__file__), native_source, provider_source,
                   layout_source, library, library.with_name('abi.json'))}
    result = dict(schema='mpas-thompson-native-binding-witness/v1', acceptance=False,
                  scope='CPU native ABI storage and pool ownership only', provenance=provenance)
    try:
        with native_module.NativePhysics(library, test_only=True) as native:
            native.config('config_microp_scheme', 'mp_thompson')
            saved = {}
            for row in rows:
                shape = tuple(dimensions[d] for d in row['dimensions'])
                dtype = np.int32 if row['type'] == 'integer' else np.float32
                a = np.arange(np.prod(shape) if shape else 1, dtype=dtype).reshape(shape)
                if row['pool_name'] == 'mp_top_level':
                    a[...] = nz
                levels = 2 if row['pool'] == 'state' else 1
                for level in range(1, levels + 1):
                    value = a.copy()
                    if level == 2:
                        value[...] += 1000
                    native.put_field(row['pool'], row['pool_name'], value, level=level, nlevels=levels)
                    saved[row['pool'], row['pool_name'], level] = value
            native_names = {'pii': 'pi_p', 'p': 'pres_p'}
            diagnostic = ('rainprod', 'evapprod', 're_cloud', 're_ice', 're_snow')
            for name in (*port.FIELD_ARGUMENTS, 'rho', 'z'):
                if name not in diagnostic:
                    levels = nz + 1 if name == 'w' else nz
                    native.put_variable(native_names.get(name, name + '_p'),
                                        np.full((cells, levels, 1), .125, np.float32))
            for name in port.SURFACE_ARGUMENTS[:7]:
                native.put_variable(name + '_p', np.full((cells, 1), .25, np.float32))
            for name, value in dict(dt_microp=np.float32(5), n_microp=np.int32(2), l_acrain=np.int32(1),
                                    has_reqc=np.int32(1), has_reqi=np.int32(0), has_reqs=np.int32(1)).items():
                native.put_variable(name, np.asarray(value))
            provider = device_module.NativeDevicePoolViews(SimpleNamespace(native=native, layout=layout), np)
            # Binding-only resources: no table values or mathematical code are executed.
            provider.thompson_v842_resources = dict(tables={'_file_sha256': dict(port.TABLE_FILE_SHA256)},
                                                    device_math_source='binding-only; no CUDA execution')
            call = SlotCall('microphysics', port.SCHEME, Backend.HEX_CUDA, 1, 0., 10.,
                            StepFlags(diagnostics=True), None, {'config_bucket_rainnc': 500.})
            provider.begin(call, {})
            p = port.bind_native_pools(provider, call)['thompson_v842']
            assert '_native_workspace' not in p
            for name in port.MODEL_ARGUMENTS[:8]:
                expected = saved['state', 'scalars', 2][indices['state', 'index_' + name] - 1, :, :solve]
                np.testing.assert_array_equal(p['model'][name].view(np.uint32), expected.view(np.uint32))
            for name in diagnostic:
                assert p['fields'][name] is provider['diag_physics'][name]
            assert p['surface']['ntc'] is provider['diag_physics']['nt_c']
            assert p['surface']['muc'] is provider['diag_physics']['mu_c']
            assert p['diagnostics']['rt_diabatic_tend'] is provider['tend']['rt_diabatic_tend']
            assert p['diagnostics']['tend_sfc_pressure'] is provider['tend']['tend_sfc_pressure']
            assert p['fields']['w'].shape == (nz, solve)
            assert p['n_microp'] == 2 and type(p['dt_microp']) is np.float32
            assert (p['has_reqc'], p['has_reqi'], p['has_reqs']) == (True, False, True)
            for name in port.CONVERSION_CONSTANTS:
                assert p['constants'][name].tobytes() == native.get_constant(name).tobytes()
            p['model']['qv'][0, 0] = np.float32(-0.)
            p['fields']['qv'][1, 2] = np.float32(23)
            provider.commit()
            np.testing.assert_array_equal(native.get_field('state', 'scalars', level=1).view(np.uint32),
                                          saved['state', 'scalars', 1].view(np.uint32))
            state = native.get_field('state', 'scalars', level=2)
            np.testing.assert_array_equal(state[..., solve:].view(np.uint32),
                                          saved['state', 'scalars', 2][..., solve:].view(np.uint32))
            assert state[indices['state', 'index_qv'] - 1, 0, 0].view(np.uint32) == 0x80000000
            assert native.get_variable('qv_p')[0][2, 1, 0] == np.float32(23)
            # Actual native allocation starts from the same unallocated state as
            # a real CUDA forecast. Bounds include the interface's extra k level.
            native.call('deallocate_microphysics', configs='configs')
            for name, value in dict(ims=1, ime=cells, kms=1, kme=nz + 1, jms=1, jme=1).items():
                native.put_variable(name, np.asarray(value, dtype=np.int32))
            descriptor = dict(module='thompson_v842', bind='bind_native_pools')
            for cycle in range(3):
                assert native.get_variable('qv_p') is None
                p = provider.begin(call, descriptor)['thompson_v842']
                assert p['_native_workspace'] is provider
                prepared_names = ('qv_p', 'qc_p', 'qr_p', 'qi_p', 'qs_p', 'qg_p',
                                  'ni_p', 'nr_p', 'rho_p', 'th_p', 'pi_p', 'pres_p',
                                  'z_p', 'dz_p', 'w_p', 'rainnc_p', 'rainncv_p', 'sr_p',
                                  'snownc_p', 'snowncv_p', 'graupelnc_p', 'graupelncv_p',
                                  'recloud_p', 'reice_p', 'resnow_p', 'rainprod_p',
                                  'evapprod_p', 'ntc_p', 'muc_p')
                assert all(native.get_variable(name) is not None for name in prepared_names)
                p['model']['qv'][0, 0] = np.float32(31 + cycle)
                p['fields']['qv'][...] = np.float32(.01)
                port._finish_native_workspace(p, commit=True)
                assert '_native_workspace' not in p and not provider.buffers
                assert all(native.get_variable(name) is None for name in prepared_names)
                provider.commit()  # Real outer adapter's subsequent commit.
                state = native.get_field('state', 'scalars', level=2)
                assert state[indices['state', 'index_qv'] - 1, 0, 0] == np.float32(31 + cycle)
                np.testing.assert_array_equal(state[..., solve:].view(np.uint32),
                                              saved['state', 'scalars', 2][..., solve:].view(np.uint32))
                np.testing.assert_array_equal(native.get_field('state', 'scalars', level=1).view(np.uint32),
                                              saved['state', 'scalars', 1].view(np.uint32))
            # A rejected resource certificate releases only its owned workspace.
            provider.thompson_v842_resources = None
            try:
                provider.begin(call, descriptor)
            except RuntimeError as exc:
                assert 'certified Thompson' in str(exc)
            else:
                raise AssertionError('missing resource certificate was accepted')
            assert all(native.get_variable(name) is None for name in prepared_names)
            result.update(binding_passed=True, scalar_components=components, halo_words_unchanged=True,
                          previous_time_level_unchanged=True, prepared_words_roundtrip=True,
                          native_workspace_cycles=3, native_workspace_arrays=len(prepared_names),
                          allocation_commit_release_passed=True, failed_binding_releases_workspace=True,
                          existing_workspace_retained=True)
    except Exception as exc:
        result.update(binding_passed=False, failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('reference', 'native_source', 'provider_source', 'layout_source', 'library', 'receipt'):
        parser.add_argument(name, type=Path)
    replay(**vars(parser.parse_args()))
