"""Compare every stored field of the native run prefix with its sealed referee.

Weather storage is read/compared by Rust. This is a native repeatability
diagnostic, never native/Hex acceptance and never an exclusion witness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=('3d_smagorinsky', 'prognostic_1.5_order'),
                        default=('3d_smagorinsky', 'prognostic_1.5_order'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        path = args.reference/('d2-les-'+model+'-native.json')
        manifest = json.loads(path.read_text())
        root = Path(manifest['output_root'])
        samples = []
        hashes = []
        for row in manifest['outputs']:
            original = root/row['path']
            candidate = args.candidate/model/row['path']
            assert digest(original) == row['sha256']
            assert candidate.is_file()
            sample = dict(strict=str(original), hex=str(candidate.resolve()),
                          stream=row['stream'], steps=row['steps'], times=row['times'])
            if 'clock_witness' in row:
                sample['clock_witness'] = row['clock_witness']
            samples.append(sample)
            hashes.append(dict(file=row['path'], reference_sha256=row['sha256'],
                               candidate_sha256=digest(candidate)))
        request = args.out/(model+'-request.json')
        request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1', samples=samples),
                                      indent=2)+'\n')
        receipt = args.out/(model+'-comparison.json')
        subprocess.run([str(args.comparator.resolve()), '--request', str(request.resolve()),
                        '--out', str(receipt.resolve())], check=True)
        provenance = dict(schema='d2-les-native-prefix-witness-v1', process_count=1,
                          native_hex_acceptance=False, excluded_words=0,
                          reference_manifest_sha256=digest(path),
                          comparator_sha256=digest(args.comparator), files=hashes)
        (args.out/(model+'-provenance.json')).write_text(json.dumps(provenance, indent=2)+'\n')
        comparison = json.loads(receipt.read_text())
        rows = comparison['fields']
        assert rows and all(row['status'] == 'compared' for row in rows)
        print(model+': '+str(sum(row['compared_words'] for row in rows))+' stored words; '+
              str(sum(row['differing_words'] for row in rows))+' differing', flush=True)
        if any(row['differing_words'] for row in rows):
            raise SystemExit('native prefix is not bitwise')


if __name__ == '__main__':
    main()
