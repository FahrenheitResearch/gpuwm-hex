"""Exercise the complete binder against compiled f3 storage and f4 transfers.

This is a CPU ABI witness, not a forecast or numerical CUDA proof. Foundation
source paths and the actual native library are supplied and hashed explicitly.
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
from hexcore.mpasphys import ntiedtke_v842 as port


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def replay(reference, native_source, provider_source, layout_source, library, receipt):
    native_module = module('ntiedtke_native_abi_witness', native_source)
    device_module = module('hexcore.mpasphys._ntiedtke_device_witness', provider_source)
    layout_module = module('hexcore.mpasphys._ntiedtke_layout_witness', layout_source)
    registry = ET.fromstring(layout_module.registry_text(reference / 'src/core_atmosphere/Registry.xml'))
    nz, solve, cells = 3, 4, 6
    dimensions = dict(nVertLevels=nz, nVertLevelsP1=nz+1, nCells=cells, nCellsSolve=solve)
    wanted = {'mesh': ('meshDensity',), 'sfc_input': ('xland',),
              'diag_physics': ('hfx', 'qfx', 'raincv', 'cuprec'),
              'tend_physics': (*port.LEVEL_OUTPUTS, 'rthdynten', 'rthblten',
                              'rthratenlw', 'rthratensw', 'rqvdynten', 'rqvblten')}
    rows = []
    for struct in registry.findall('./var_struct'):
        for field in struct:
            name = field.get('name_in_code', field.get('name'))
            if name in wanted.get(struct.get('name'), ()):
                row = dict(field.attrib)
                row.update(pool=struct.get('name'), pool_name=name,
                           dimensions=[d for d in field.get('dimensions', '').split() if d != 'Time'])
                rows.append(row)
    assert len(rows) == sum(map(len, wanted.values()))
    layout = SimpleNamespace(pools=set(wanted), fields=rows, dimensions=dimensions, indices={},
                             configs=dict(config_len_disp=np.float32(120000), config_do_restart=True))
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (Path(port.__file__), Path(__file__), native_source, provider_source,
                   layout_source, library, library.with_name('abi.json'))}
    result = dict(schema='mpas-ntiedtke-native-binding-witness/v1', acceptance=False,
                  scope='actual CPU native ABI storage and pool transfers only', provenance=provenance)
    try:
        with native_module.NativePhysics(library, test_only=True) as native:
            saved = {}
            for row in rows:
                shape = tuple(dimensions[d] for d in row['dimensions'])
                value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                native.put_field(row['pool'], row['pool_name'], value)
                saved[row['pool'], row['pool_name']] = value.copy()
            profiles = ('t_p', 'qv_p', 'qc_p', 'qi_p', 'u_p', 'v_p', 'pres_hyd_p',
                        'dz_p', 'rho_p', 'pi_p', 'pres2_hyd_p', 'w_p')
            prepared = {}
            for name in profiles:
                value = np.arange(cells*(nz+1), dtype=np.float32).reshape(cells, nz+1, 1)
                native.put_variable(name, value)
                prepared[name] = value.copy()
            native.put_variable('cu_act_flag', np.zeros((cells, 1), dtype=np.int32))
            for name, value in dict(dt_dyn=np.float32(5), n_cu=np.int32(2),
                                    f_qc=np.int32(1), f_qi=np.int32(0)).items():
                native.put_variable(name, np.asarray(value))
            provider = device_module.NativeDevicePoolViews(SimpleNamespace(native=native, layout=layout), np)
            call = SlotCall('convection', port.SCHEME, Backend.HEX_CUDA, 19, 90., 10., StepFlags(), None, {})
            provider.begin(call, {})
            bound = port.bind_native_pools(provider, call)['ntiedtke_v842']
            binding = bound['binding']
            assert binding.dt_dyn.tobytes() == np.float32(5).tobytes()
            assert binding.stepcu == 2 and binding.restart
            assert (binding.f_qc, binding.f_qi) == (True, False)
            assert binding.fields['t3d'].shape == (nz, solve)
            assert binding.fields['p8w'].shape == (nz+1, solve)
            assert binding.fields['w'].shape == (nz+1, solve)
            np.testing.assert_array_equal(binding.fields['t3d'], prepared['t_p'][:solve, :nz, 0].T)
            for name in port.LEVEL_OUTPUTS:
                assert binding.target[name] is provider['tend_physics'][name]
            aliases = dict(rd='r_d', rv='r_v', grav='gravity')
            for name in port.Constants.__dataclass_fields__:
                assert getattr(bound['constants'], name).tobytes() == native.get_constant(aliases.get(name, name)).tobytes()
            binding.fields['t3d'][0, 0] = np.float32(-0.)
            binding.target['rqicuten'][1, 2] = np.float32(23)
            binding.target['cu_act_flag'][0] = np.int32(1)
            provider.commit()
            for row in rows:
                value = native.get_field(row['pool'], row['pool_name'])
                np.testing.assert_array_equal(value[..., solve:].view(np.uint32),
                                              saved[row['pool'], row['pool_name']][..., solve:].view(np.uint32))
            temperature = native.get_variable('t_p')[0]
            assert temperature[0, 0, 0].view(np.uint32) == 0x80000000
            np.testing.assert_array_equal(temperature[:, nz, :], prepared['t_p'][:, nz, :])
            np.testing.assert_array_equal(temperature[solve:], prepared['t_p'][solve:])
            assert native.get_field('tend_physics', 'rqicuten')[1, 2] == np.float32(23)
            assert native.get_variable('cu_act_flag')[0][0, 0] == 1
            result.update(binding_passed=True, halo_words_unchanged=True, allocated_top_unchanged=True,
                          compiled_constants_exact=True, separate_dynamical_dt_and_cadence=True)
    except Exception as exc:
        result.update(binding_passed=False, failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('reference', 'native_source', 'provider_source', 'layout_source', 'library', 'receipt'):
        parser.add_argument(name, type=Path)
    replay(**vars(parser.parse_args()))
