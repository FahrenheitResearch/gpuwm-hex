"""Replay the complete proposed LW slot with enabled p21 radius plumbing.

MPAS: mpas_atmphys_driver_radiation_lw.F:297-719,805-884.
MPAS: module_ra_rrtmg_lw.F:11574-12535.
Rust decodes pristine native fixtures; no native output becomes a physics input.
This complements real disabled-radius pool captures, not a forecast proof.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('module', 'views', 'native-fixture', 'native-receipt', 'table', 'table-decoder', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from hexcore.mpasphys import radshared_v842
    native_receipt = json.loads(args.native_receipt.read_text())
    assert native_receipt['fixtures']['native-columns.bin'] == sha(args.native_fixture)
    assert native_receipt['native_process_count'] == 1
    assert native_receipt['total_differing_words'] == 0
    manifest = json.loads((args.views / 'manifest.json').read_text())
    assert manifest['schema'] == 'mpas-native-tagged-columns/v1'
    records = {r['name']: r for r in manifest['records']}
    cases = sorted({name.split('/')[0] for name in records})
    assert len(cases) == 300 and len(records) == len(manifest['records'])
    spec = importlib.util.spec_from_file_location('hexcore.mpasphys.enabled_lw_radius_candidate', args.module.resolve())
    port = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = port
    spec.loader.exec_module(port)
    tables = port.load_tables_device(args.table, decoder=args.table_decoder)
    rows = []
    for number, case in enumerate(cases):
        def view(name):
            meta = records[case + '/' + name]
            assert meta['dtype'] == 0
            assert sha(args.views / meta['file']) == meta['sha256']
            data = (args.views / meta['file']).read_bytes()
            assert len(data) == meta['words'] * 4
            return np.frombuffer(data, dtype='<f4').reshape(meta['shape'], order='F')
        def device(name):
            return cp.asarray(view(name))
        flags = records[case + '/flags']['integer_metadata']
        model, interfaces, surface = device('model'), device('interfaces'), device('surface')
        nl = model.shape[1]
        assert model.shape[0] == 12 and interfaces.shape == (2, nl + 1)
        prepared = {name: model[index].reshape(nl, 1) for name, index in
            (('pres_hyd', 0), ('t', 1), ('dz', 2), ('pi', 11))}
        prepared.update(pres2_hyd=interfaces[0].reshape(nl + 1, 1),
            t2=interfaces[1].reshape(nl + 1, 1), xlat=surface[5:6])
        diagnostic = dict(cldfrac=model[7].reshape(nl, 1), sfc_emiss=surface[1:2],
            o3clim=device('ozone').reshape(-1, 1), o3vmr=cp.empty((nl, 1), dtype=cp.float32))
        for index, name in enumerate(('re_cloud', 're_ice', 're_snow'), 8):
            diagnostic[name] = model[index].reshape(nl, 1)
        for name in ('rre_cloud', 'rre_ice', 'rre_snow'):
            diagnostic[name] = cp.zeros((nl, 1), dtype=cp.float32)
        for name in port.FLUX_FIELDS:
            diagnostic['olrtoa' if name == 'olr' else name] = cp.zeros(1, dtype=cp.float32)
        schemes = ('mp_thompson', 'mp_thompson_aerosols', 'mp_wsm6')
        options = dict(config_microp_re=True, config_microp_scheme=schemes[number % len(schemes)],
            config_radt_cld_overlap=('none', 'random', 'maximum_random', 'maximum', 'exponential', 'exponential_random')[flags[1]],
            config_radt_cld_dcorrlen='latitude_varying' if flags[2] else 'constant',
            config_o3climatology=flags[7] == 2, config_radt_lw_scheme='rrtmg_lw', config_radt_sw_scheme='rrtmg_sw')
        pools = dict(nCellsSolve=1, prepared=prepared, diag_physics=diagnostic,
            radshared={name: model[index].reshape(1, nl) for name, index in
                (('qvrad', 3), ('qcrad', 4), ('qirad', 5), ('qsrad', 6))},
            sfc_input={name: surface[index:index + 1] for name, index in
                (('skintemp', 0), ('xland', 2), ('xice', 3), ('snow', 4))},
            tend_physics={'rthratenlw': cp.zeros((nl, 1), dtype=cp.float32)},
            atm_input={'pin': device('pin')}, rrtmg_lw_tables=tables,
            icloud=flags[0], julday=flags[3], radius_flags=tuple(flags[4:7]))
        actual = port.radiation_lw_slot(SimpleNamespace(pools=pools, options=options))
        comparisons = {}
        def compare(name, value, expected):
            expected_device = cp.asarray(expected).ravel(order='F').view(cp.uint32)
            words = cp.ascontiguousarray(value).ravel().view(cp.uint32)
            assert words.size == expected_device.size
            differences = int(cp.count_nonzero(words != expected_device).get())
            comparisons[name] = dict(words=int(words.size), differing_words=differences)
        # Native outputs are first read only after the entire slot completes.
        flux = view('flux')
        for index, name in enumerate(port.FLUX_FIELDS):
            compare(name, diagnostic['olrtoa' if name == 'olr' else name], flux[index:index + 1])
        compare('rthratenlw', pools['tend_physics']['rthratenlw'], view('heating'))
        for name in ('rre_cloud', 'rre_ice', 'rre_snow'):
            compare(name, diagnostic[name], view(name))
        for name in ('uflx', 'uflxc', 'dflx', 'dflxc'):
            compare(name, actual['profiles'][0][name], view(name))
        rows.append(dict(case=case, options=options, radius_flags=flags[4:7], arrays=comparisons))
    result = dict(schema='p21-enabled-radius-complete-lw-consumer/v1', process_count=1,
        scope='Complete proposed LW slot, 300 pristine native columns, all written flux/heating/radius outputs and all four native flux profiles; standalone supporting evidence',
        rows=rows, compared_words=sum(a['words'] for r in rows for a in r['arrays'].values()),
        differing_words=sum(a['differing_words'] for r in rows for a in r['arrays'].values()),
        enabled_radius_schemes=sorted({r['options']['config_microp_scheme'] for r in rows}),
        native_output_never_physics_input=True, production_applied=False, forecast_proven=False,
        ozone_diagnostic_expected_absent=True,
        artifacts={str(p.resolve()): sha(p) for p in (args.module, args.native_fixture,
            args.native_receipt, args.table, args.table_decoder, args.views / 'manifest.json',
            Path(radshared_v842.__file__), Path(__file__))})
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(columns=len(rows), compared_words=result['compared_words'], differing_words=result['differing_words'])), flush=True)
    raise SystemExit(0 if result['differing_words'] == 0 else 1)


if __name__ == '__main__':
    main()
