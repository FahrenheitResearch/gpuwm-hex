"""Preserve real f1 radiation-support input words without changing parent seals.

MPAS mpas_atmphys_driver_radiation_lw.F:414-506,688-712,845-863.
Only copies authenticated bytes. Rust decodes records during replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def authenticate_manifest(path):
    manifest = json.loads(Path(path).read_text())
    content = {k: v for k, v in manifest.items() if k != 'dump_root_sha256'}
    actual = hashlib.sha256(json.dumps(content, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if actual != manifest['dump_root_sha256'] or manifest['process_count'] != 1:
        raise ValueError('native tap seal or fixed process count differs')
    return manifest


FIELDS = ('mesh.latCell', 'atm_input.ozmixm', 'atm_input.pin',
          'diag_physics.o3clim', 'state.mpas_atmphys_vars.pres_hyd_p',
          'block.dimensions.nCells', 'block.dimensions.nCellsSolve')
FIELDS += tuple('diag_physics.' + prefix + phase for prefix in ('re_', 'rre_')
                for phase in ('cloud', 'ice', 'snow'))
FIELDS += tuple('block.configs.config_' + name for name in
    ('radt_cld_overlap', 'radt_cld_dcorrlen', 'o3climatology', 'microp_re',
     'microp_scheme', 'radt_lw_scheme', 'radt_sw_scheme'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    rows = []
    for source in args.manifest:
        manifest = authenticate_manifest(source)
        root = manifest['dump_root_sha256']
        for prefix, capture in manifest['captures'].items():
            if capture['site'] != 'boundary__driver_radiation_lw':
                continue
            destination = args.out / root[:16] / prefix
            destination.mkdir(parents=True)
            members = [prefix + '.pre.' + name + '.bin' for name in FIELDS]
            files = {}
            for member in members:
                original = source.parent / member
                expected = manifest['files'][member]
                if original.stat().st_size != expected['bytes'] or sha(original) != expected['sha256']:
                    raise ValueError('native tap member changed: ' + member)
                shutil.copyfile(original, destination / member)
                files[member] = expected
            rows.append(dict(directory=str(destination), prefix=prefix, files=files,
                original_manifest=str(source.resolve()), original_manifest_sha256=sha(source),
                original_dump_root_sha256=root, native_process_count=1,
                native_input_provenance=manifest.get('input_provenance'),
                expected_scope='strict pristine module on native-tap inputs; not radiation scheme outputs'))
    if not rows:
        raise ValueError('no real native radiation support inputs')
    (args.out / 'input-index.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
