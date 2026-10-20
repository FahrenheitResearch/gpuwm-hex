"""Replay the entire proposed SW binder/slot against strict native columns.

MPAS: mpas_atmphys_driver_radiation_sw.F:355-794,870-948,1009-1050.
Rust owns decoding and byte order. Native POST words are comparison-only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Provider(dict):
    def __init__(self, values, cp, engine, nl, calendar):
        super().__init__(values)
        self.values = self
        self.cp = cp
        self.rrtmg_sw_v842_resources = engine
        self.session = SimpleNamespace(layout=SimpleNamespace(
            dimensions=dict(nVertLevels=nl, nCellsSolve=1)))
        self.native = self
        self.calendar = calendar

    def get_table(self, name):
        return np.asarray(self.calendar[name.split('.')[-1]]), {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('module', 'views', 'native-fixture', 'native-receipt',
                 'table', 'table-decoder', 'work', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from hexcore.mpasphys import radshared_v842
    native = json.loads(args.native_receipt.read_text())
    assert native['process_count'] == 1 and native['native_outputs_byte_identical']
    assert sha(args.native_fixture) == native['artifacts_sha256']['coupled-driver.swd']
    manifest = json.loads((args.views / 'manifest.json').read_text())
    assert manifest['native_fixture_sha256'] == sha(args.native_fixture)
    records = {r['name']: r for r in manifest['records']}
    assert len(records) == len(manifest['records'])
    cases = sorted({name.split('/')[0] for name in records})
    assert len(cases) == 181, 'requires the entire authentic native driver fixture'
    spec = importlib.util.spec_from_file_location(
        'hexcore.mpasphys.enabled_sw_radius_candidate', args.module.resolve())
    port = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = port
    spec.loader.exec_module(port)
    args.work.mkdir(exist_ok=False)
    engine = port.MPASShortwave.from_mpas_data(args.table,
        decoder=args.table_decoder, decoded_directory=args.work / 'decoded-tables')
    rows = []
    for case in cases:
        def view(key):
            meta = records[case + '/' + key]
            assert sha(args.views / meta['file']) == meta['sha256']
            assert meta['dtype'] == 0
            return np.frombuffer((args.views / meta['file']).read_bytes(),
                dtype='<f4').reshape(meta['shape'], order='F')
        def integer(key):
            meta = records[case + '/in/' + key]
            assert meta['dtype'] == 1 and meta['words'] == 1
            assert sha(args.views / meta['file']) == meta['sha256']
            return meta['integer_metadata'][0]
        def scalar(key):
            value = view('in/' + key)
            assert value.size == 1
            return value[()]
        nl = records[case + '/in/p3d']['shape'][0] - 1
        def field(key, levels=nl):
            value = cp.full((levels, 2), -987.25, dtype=cp.float32)
            # Native tagged scalars (including height) have rank zero; the
            # driver pool presents the same original word as a level carrier.
            value[:, 0] = cp.asarray(view('in/' + key).reshape(-1, order='F')[:levels])
            return value
        def surface(key):
            value = cp.full(2, -987.25, dtype=cp.float32)
            value[:1] = cp.asarray(view('in/' + key).reshape(1))
            return value
        aliases = dict(pres_hyd_p='p3d', t_p='t3d', dz_p='dz8w', pi_p='pi3d',
            qvrad_p='qv3d', qcrad_p='qc3d', qirad_p='qi3d', qsrad_p='qs3d',
            nwfa_p='nwfa', nifa_p='nifa', qv_p='aerosol_qv')
        prepared = {name: field(key) for name, key in aliases.items()}
        prepared.update(pres2_hyd_p=field('p8w', nl + 1), t2_p=field('t8w', nl + 1))
        diag = {name: cp.full(2, -987.25, dtype=cp.float32) for name in port.SCALAR_OUTPUTS}
        diag.update(cldfrac=field('cldfra3d'), sfc_albedo=surface('albedo'),
            o3clim=field('o3clim', records[case + '/in/o3clim']['shape'][0]),
            taod5502d=cp.full(2, -987.25, dtype=cp.float32),
            taod5503d=cp.full((nl, 2), -987.25, dtype=cp.float32))
        diag.update({name: field('raw_' + name) for name in ('re_cloud', 're_ice', 're_snow')})
        schemes = {1: 'mp_thompson', 2: 'mp_wsm6', 3: 'mp_thompson_aerosols'}
        overlap = integer('cldovrlp')
        options = dict(config_microp_scheme=schemes[integer('microp_case')],
            config_microp_re=bool(integer('config_microp_re')),
            config_radt_cld_overlap=('none', 'random', 'maximum_random', 'maximum',
                'exponential', 'exponential_random')[overlap],
            config_radt_cld_dcorrlen='latitude_varying' if integer('idcor') else 'constant',
            config_o3climatology=integer('o3input') == 2)
        state = {name: integer(name) for name in ('has_reqc', 'has_reqi', 'has_reqs')}
        state['dt_radtsw'] = scalar('dt_radtsw')
        calendar = dict(curr_julday=scalar('curr_julday'), gmt=scalar('gmt'), julday=integer('julday'))
        values = dict(prepared=prepared, diag_physics=diag,
            scalar_state=state, constants=dict(degrad=scalar('degrad'),
                dpd=np.asarray(0x3f7c7e3f, dtype='<u4').view('<f4')[()]),
            mesh=dict(latCell=surface('lat_cell'), lonCell=surface('lon_cell'), zgrid=field('ht', 1)),
            sfc_input={name: surface(key) for name, key in
                (('skintemp', 'tsk'), ('snow', 'snow'), ('xice', 'xice'), ('xland', 'xland'))},
            tend_physics=dict(rthratensw=cp.full((nl, 2), -987.25, dtype=cp.float32)),
            atm_input=dict(pin=cp.asarray(view('in/pin'))))
        provider = Provider(values, cp, engine, nl, calendar)
        call = SimpleNamespace(pools=provider, options=options, time_seconds=scalar('xtime_s'))
        port.bind_native_pools(provider, call)
        actual = port.shortwave_slot(call)
        arrays = {}
        # Read native scientific outputs only after the full binder/slot.
        for key in sorted(k for k in records if k.startswith(case + '/out/')):
            name = key.split('/')[-1]
            expected = cp.asarray(view('out/' + name)).ravel(order='F').view(cp.uint32)
            got = cp.ascontiguousarray(actual[name]).ravel().view(cp.uint32)
            assert expected.size == got.size, (case, name, expected.size, got.size)
            arrays[name] = dict(words=int(got.size), differing_words=int(cp.count_nonzero(expected != got).get()))
        targets = dict(diag, **provider['tend_physics'])
        for name in (*port.SCALAR_OUTPUTS, 'rthratensw',
                     *(['taod5502d', 'taod5503d'] if options['config_microp_scheme'] == 'mp_thompson_aerosols' else [])):
            target = targets[name]
            expected = cp.asarray(view('out/' + name)).ravel(order='F').view(cp.uint32)
            got = cp.ascontiguousarray(target[..., :1]).ravel().view(cp.uint32)
            arrays['write/' + name] = dict(words=int(got.size), differing_words=int(cp.count_nonzero(expected != got).get()))
            halo = target[..., 1:]
            arrays['halo/' + name] = dict(words=int(halo.size),
                differing_words=int(cp.count_nonzero(halo != np.float32(-987.25)).get()))
        rows.append(dict(case=case, options=options, arrays=arrays))
    result = dict(schema='p21-enabled-radius-complete-sw-consumer/v1', process_count=1,
        rows=rows, compared_words=sum(a['words'] for r in rows for a in r['arrays'].values()),
        differing_words=sum(a['differing_words'] for r in rows for a in r['arrays'].values()),
        enabled_cases=sum(r['options']['config_microp_re'] for r in rows),
        native_output_never_physics_input=True, production_applied=False, forecast_proven=False,
        artifacts={str(p.resolve()): sha(p) for p in (args.module, args.native_fixture,
            args.native_receipt, args.views / 'manifest.json', args.table, args.table_decoder,
            Path(radshared_v842.__file__), Path(__file__))})
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(columns=len(rows), enabled_columns=result['enabled_cases'],
        compared_words=result['compared_words'], differing_words=result['differing_words'])), flush=True)
    raise SystemExit(0 if result['differing_words'] == 0 else 1)


if __name__ == '__main__':
    main()
