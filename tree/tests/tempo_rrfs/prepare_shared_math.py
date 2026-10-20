"""Select existing pinned shared mathematics; implement no private functions."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    root = args.engine / 'gpuwm/core/kernels'
    files = [root / name for name in
             ('glibc_flt32.cuh', 'glibc_flt64.cuh', 'thompson_aerosol_libm.cuh')]
    source = '\n'.join(path.read_text() for path in files)
    for name in ('expf', 'exp', 'logf', 'log', 'log10f', 'log10', 'powf', 'pow'):
        source += f'\n#define mpas_{name} thompson_aa_{name}\n'
    for name in ('atan', 'atanf', 'atan2'):
        source += f'\n#define mpas_{name} {name}\n'
    source += '\n#define mpas_hypot thompson_aa_hypot\n'
    path = root / 'urban_bem.cuh'
    text = path.read_text()
    start = text.index('// log10f: CORE-MATH ')
    end = text.index('\n__device__ void bem_bem(', start)
    window = text[start:end]
    # The unchanged shared window independently audited by p3; notices travel
    # with it. This hash authenticates reuse, rather than rederiving log10f.
    expected = 'a604c98e82b95751b2651537e60332dd1121c9df55b1ca34cd6314a0a8b88b48'
    if digest(window.encode()) != expected:
        raise ValueError('pinned shared glibc 2.43 log10f source window changed')
    source += '\n' + window + '\n#undef mpas_log10f\n#define mpas_log10f ubm_log10f\n'
    args.output.write_text(source)
    args.receipt.write_text(json.dumps({
        'forecast_proven': False,
        'private_transcendental_added': False,
        'source_files': [{'path': str(p), 'sha256': digest(p.read_bytes())}
                         for p in files + [path]],
        'unchanged_log10f_window_sha256': expected,
        'log10f_entrypoint': 'ubm_log10f',
        'hypot_entrypoint': 'thompson_aa_hypot',
        'atan2_entrypoint': 'CUDA atan2; universal native identity unproven',
        'output': str(args.output), 'output_sha256': digest(source.encode()),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
