"""Select existing shared engine maths for complete aerosol development replay.

The glibc 2.43 log10f text, constants and license are copied unchanged from
the pinned engine's urban_bem.cuh. No scheme-private transcendental is added.
This source selection is diagnostic until the complete native proof holds.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(engine, output):
    root = engine / 'gpuwm/core/kernels'
    paths = [root / name for name in ('glibc_flt32.cuh', 'glibc_flt64.cuh',
                                     'thompson_aerosol_libm.cuh', 'urban_bem.cuh')]
    text = paths[-1].read_text()
    start = text.index('// log10f: CORE-MATH ')
    end = text.index('\n__device__ void bem_bem(', start)
    selected = text[start:end]
    source = '\n'.join(path.read_text() for path in paths[:-1])
    for name in ('expf', 'exp', 'logf', 'log', 'log10f', 'log10', 'powf', 'pow'):
        source += f'\n#define mpas_{name} thompson_aa_{name}\n'
    for name in ('atan', 'atanf', 'atan2', 'hypot'):
        source += f'\n#define mpas_{name} {name}\n'
    source += '\n' + selected + '\n#undef mpas_log10f\n#define mpas_log10f ubm_log10f\n'
    output.write_text(source)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    receipt = dict(acceptance=False, forecast_proven=False,
        scope='unchanged shared glibc 2.43 log10f selected for whole-module diagnostics',
        sources={str(path): sha(path.read_bytes()) for path in paths},
        selected_log10f_sha256=sha(selected.encode()),
        selected_line_start=text[:start].count('\n') + 1,
        selected_line_end=text[:end].count('\n') + 1,
        output=dict(path=str(output), sha256=sha(output.read_bytes())),
        remaining_limits='other shared entrypoints retain their original provenance; '
                         'no universal identity or accepted forecast is inferred')
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.engine.resolve(), args.output.resolve())
