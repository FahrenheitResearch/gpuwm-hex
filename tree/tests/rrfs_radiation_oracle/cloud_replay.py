"""Replay all recorded pristine whole-cloudiness calls through the public slot.

MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:
141-370,508-577 @ 1d6e370e. Rust owns record I/O and word comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess

from cloud_records import Records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path)
    args = parser.parse_args()
    from importlib.util import find_spec
    if args.dependencies and find_spec('hexcore.mpasphys.radshared_v842') is None:
        # Authenticated unlanded parent stand-ins remain diagnostic authority.
        import importlib.util
        import hexcore.mpasphys
        for name in ('radshared_v842',):
            module_path = args.dependencies / (name + '.py')
            if digest(module_path) != 'f6d6ee2b823aa102ccd6392a3250c37fd9dbd4a7860a0f5ba2355b0c83c4b902':
                raise ValueError('Authenticated parent stand-in changed')
            spec = importlib.util.spec_from_file_location('hexcore.mpasphys.' + name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            setattr(hexcore.mpasphys, name, module)
    import cupy as cp
    from hexcore.mpasphys.rrfs import radiation_rrfs as radiation
    from hexcore.mpasphys.driver_v842 import SlotCall, Backend, StepFlags
    checkout = Path(__file__).resolve().parents[3]
    if not Path(radiation.__file__).resolve().is_relative_to(checkout / 'tree/src'):
        raise ValueError('Production radiation was not imported from the supplied checkout')
    manifest_path = args.bank / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    passive_path = args.bank / 'passivity.json'
    if digest(passive_path) != manifest['observer_passivity_sha256']:
        raise ValueError('Native observer passivity receipt changed')
    passive = json.loads(passive_path.read_text())
    if not passive['all_identical'] or passive['errors'] or not passive['files']:
        raise ValueError('Pristine native observer passivity was not measured')
    if manifest['model_commit'] != '1d6e370e7b43a332074809f170532a216c2212fb' or manifest['process_count'] != 1:
        raise ValueError('Native fork or process count differs')
    observed = {p.name for p in (args.bank / 'dump').iterdir()}
    if observed != set(manifest['files']):
        raise ValueError('Native capture inventory differs')
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(args.bank / 'dump' / name) != sha:
            raise ValueError('Native captured member changed: ' + name)
    if args.output.resolve().is_relative_to(args.bank.resolve()):
        raise ValueError('Replay output must be outside the immutable capture bank')
    args.output.mkdir(parents=True, exist_ok=True)
    records = Records(args.library)
    prefixes = sorted(p.name.removesuffix('.scope.txt') for p in (args.bank / 'dump').glob('*.scope.txt'))
    if not prefixes:
        raise ValueError('No complete native cloudiness call was captured')
    comparisons = []
    cases = []
    for prefix in prefixes:
        root = args.bank / 'dump'
        paths = {p.name.split('.pre.', 1)[1].removesuffix('.bin'): p for p in root.glob(prefix + '.pre.*.bin')}
        values = {name: records.load(path)[0] for name, path in paths.items()}
        required = {'configs.config_len_disp', 'configs.config_radt_cld_scheme',
                    'configs.config_convection_scheme', 'mesh.meshDensity.t1', 'sfc_input.xland.t1',
                    'diag.cldfrac.t1'}
        missing = required - values.keys()
        if missing:
            raise ValueError('Full native driver inputs absent: ' + repr(sorted(missing)))
        its, ite = (int(values['argument.' + n]) for n in ('its', 'ite'))
        kts, kte, jts, jte = (int(values['global.' + n]) for n in ('kts', 'kte', 'jts', 'jte'))
        ims, kms, jms = (int(values['global.' + n]) for n in ('ims', 'kms', 'jms'))
        if (its, ims, jts, jte, jms, kts, kms) != (1, 1, 1, 1, 1, 1, 1):
            raise ValueError('This stated global MPI1 case requires native origin 1 and one tile')
        nz, nc = kte, ite
        device = {n: cp.ascontiguousarray(cp.asarray(a)) for n, a in values.items() if not isinstance(a, str)}
        profiles = {n: cp.ascontiguousarray(device['global.' + n + '_p'][:, :nz, 0].T)
                    for n in ('qv', 'qc', 'qi', 'qs', 't', 'pres', 'pres_hyd', 'rho')}
        diag = {n.removeprefix('diag.').removesuffix('.t1'): a for n, a in device.items() if n.startswith('diag.')}
        scheme = values['configs.config_radt_cld_scheme'].strip()
        convection = values['configs.config_convection_scheme'].strip()
        carrier_names = ['qvrad', 'qcrad', 'qirad', 'qsrad', 'cldfrac', 'dx', 'xland']
        if scheme == 'cld_fraction_mynn':
            carrier_names += ['qcbl', 'qibl']
            for n in ('cldfrac_bl', 'qc_bl', 'qi_bl'):
                if n not in diag:
                    raise ValueError('Missing native MYNN driver input: ' + n)
        if convection == 'cu_grell_freitas_li':
            carrier_names += ['qccu', 'qicu']
        carriers = {n: cp.ascontiguousarray(device['global.' + n + '_p'][:, :nz, 0].T)
                    if device['global.' + n + '_p'].ndim == 3 else
                    cp.ascontiguousarray(device['global.' + n + '_p'][:, 0]) for n in carrier_names}
        options = dict(config_len_disp=float(values['configs.config_len_disp']),
                       config_convection_scheme=convection)
        pools = dict(prepared=profiles, diag_physics=diag, radiation_carriers=carriers,
                     mesh={'meshDensity': device['mesh.meshDensity.t1']},
                     sfc_input={'xland': device['sfc_input.xland.t1']}, nCellsSolve=nc,
                     itimestep=int(values['argument.itimestep']), f_qc=bool(values['global.f_qc']),
                     f_qi=bool(values['global.f_qi']), radshared={})
        call = SlotCall('cloudiness', scheme, Backend.HEX_CUDA, pools['itimestep'],
                        0.0, 0.0, StepFlags(), pools, options)
        radiation.cloudiness_slot(call)
        for name, carrier in carriers.items():
            target = device['global.' + name + '_p']
            if target.ndim == 3:
                target[:, :nz, 0] = carrier.T
            else:
                target[:, 0] = carrier
        for name, path in paths.items():
            target = args.output / (prefix + '.post.' + name + '.bin')
            if target.exists():
                target.unlink()
            replacement = cp.asnumpy(device[name]) if name.startswith('global.') and name.endswith('_p') or name.startswith('diag.') else None
            records.post(path, target, replacement)
            expected = root / target.name
            entry = records.compare(expected, target)
            entry.setdefault('comparison_complete', not entry.get('schema_mismatch') and
                             isinstance(entry.get('different_words'), int))
            entry.update(case=prefix, field=name)
            comparisons.append(entry)
        post_names = {p.name for p in root.glob(prefix + '.post.*.bin')}
        if post_names != {prefix + '.post.' + name + '.bin' for name in paths}:
            raise ValueError('Every native output record must have a corresponding PRE carrier')
        cases.append(dict(case=prefix, scheme=scheme, convection=convection, nCellsSolve=nc,
                          nVertLevels=nz, itimestep=pools['itimestep'], complete_driver_inputs=True))
    complete = all(row['comparison_complete'] for row in comparisons)
    equal = complete and all(row['bitwise_equal'] for row in comparisons)
    output_names = {'diag.cldfrac.t1'} | {'global.' + n + '_p' for n in
        ('qvrad', 'qcrad', 'qirad', 'qsrad', 'cldfrac', 'dx', 'xland', 'qcbl', 'qibl', 'qccu', 'qicu')}
    result = dict(schema='r9.pristine-real-cloudiness-replay/v1', bitwise_equal=equal,
                  checkout_tip=subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip(),
                  replay_source_sha256=digest(Path(__file__)),
                  comparison_complete=complete, cases=cases, arrays=len(comparisons),
                  words=sum(row['words'] for row in comparisons),
                  scientific_output_words=sum(row['words'] for row in comparisons if row['field'] in output_names),
                  preserved_input_words=sum(row['words'] for row in comparisons if row['field'] not in output_names),
                  different_words=sum(row['different_words'] for row in comparisons) if complete else None,
                  native_manifest_sha256=digest(manifest_path), observer_passivity=True,
                  production_source=str(Path(radiation.__file__).resolve()),
                  production_sha256=digest(Path(radiation.__file__)),
                  parent_source=str(Path(sys.modules['hexcore.mpasphys.radshared_v842'].__file__).resolve()),
                  parent_sha256=digest(Path(sys.modules['hexcore.mpasphys.radshared_v842'].__file__)),
                  scope='all records of stated real whole cloud-driver calls; no SW/LW or forecast claim',
                  comparisons=comparisons)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('bitwise_equal', 'arrays', 'words', 'different_words')}))
    raise SystemExit(0 if equal else 1)


if __name__ == '__main__':
    main()
