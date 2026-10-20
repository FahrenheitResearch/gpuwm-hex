"""Copy the authenticated full native cloud-driver scientific boundary.

MPAS mpas_atmphys_driver_cloudiness.F:92-234. Copies immutable record bytes;
Rust decodes them and the production slot processes the device fields. This
selection proves all eight written arrays and unchanged scientific inputs,
including storage outside the native solve extent, not private-state closure.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    state = 'state.mpas_atmphys_vars.'
    unchanged = [state + n + '_p' for n in
                 ('qv', 'qc', 'qi', 'qs', 't', 'pres', 'pres_hyd', 'rho')]
    unchanged += ['mesh.meshDensity', 'sfc_input.xland']
    outputs = ['diag_physics.cldfrac'] + [state + n + '_p' for n in
               ('cldfrac', 'qvrad', 'qcrad', 'qirad', 'qsrad', 'dx', 'xland')]
    inputs = unchanged + outputs + [state + n for n in
             ('f_qc', 'f_qi', 'its', 'ite', 'kts', 'kte', 'jts', 'jte')]
    inputs += ['block.dimensions.nCellsSolve', 'block.configs.config_len_disp',
               'block.configs.config_radt_cld_scheme']
    rows = []
    for source_manifest in args.manifest:
        manifest = json.loads(source_manifest.read_text())
        original_root = manifest['dump_root_sha256']
        content = {k: v for k, v in manifest.items() if k != 'dump_root_sha256'}
        assert hashlib.sha256(json.dumps(content, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest() == original_root
        assert manifest['process_count'] == 1
        for prefix, call in manifest['captures'].items():
            if call['site'] != 'boundary__driver_cloudiness':
                continue
            destination = args.out / original_root[:16] / prefix
            destination.mkdir(parents=True)
            members = [prefix + '.pre.' + n + '.bin' for n in inputs]
            members += [prefix + '.post.' + n + '.bin' for n in outputs + unchanged]
            for member in members:
                source = source_manifest.parent / member
                expected = manifest['files'][member]
                assert not source.is_symlink()
                assert source.stat().st_size == expected['bytes']
                assert sha(source) == expected['sha256']
                shutil.copyfile(source, destination / member)
                assert sha(destination / member) == expected['sha256']
            rows.append(dict(directory=str(destination), prefix=prefix,
                inputs=inputs, outputs=outputs, unchanged_inputs=unchanged,
                files={n: manifest['files'][n] for n in members},
                original_manifest=str(source_manifest.resolve()),
                original_manifest_sha256=sha(source_manifest),
                original_dump_root_sha256=original_root,
                capture_build_provenance=manifest.get('build_provenance'),
                capture_input_provenance=manifest.get('input_provenance'),
                native_process_count=1, hidden_state_complete=False,
                scope='all cloud-driver written scientific arrays and preserved inputs',
                original_manifest_unchanged=True, forecast_proven=False))
    assert rows
    (args.out / 'input-index.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(dict(captures=len(rows), output_arrays=len(outputs),
                         preserved_input_arrays=len(unchanged))))


if __name__ == '__main__':
    main()
