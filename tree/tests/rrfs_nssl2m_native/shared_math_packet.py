#!/usr/bin/env python3
"""Assemble existing shared helpers for a whole-module dependency diagnostic.

This deliberately incomplete packet is NEVER the default production packet.
Missing native binary64 functions remain unresolved external device symbols;
no alternate arithmetic, builtin or narrower overload substitutes for them.
The full module compiler must expose the resulting dependency failure.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare import load_target


def assemble(header, expected, out, *, public_binary64=False):
    shared = header.read_bytes()
    assert hashlib.sha256(shared).hexdigest() == expected
    out.mkdir(parents=True, exist_ok=True)
    aliases = []
    for name in ('exp', 'log', 'pow'):
        for kind in ('float', 'double'):
            parameters = f'{kind} a, {kind} b' if name == 'pow' else f'{kind} a'
            arguments = 'a,b' if name == 'pow' else 'a'
            aliases.append(f'__device__ __noinline__ {kind} nssl_{name}({parameters}) '
                           f'{{ return mynn_{name}({arguments}); }}')
    for name in ('log10', 'atan', 'tanh'):
        aliases.append(f'__device__ __noinline__ float nssl_{name}(float a) {{ return mynn_{name}(a); }}')
    # Fork module_mp_nssl_2mom.F:3514,15872,16798 at 1d6e370e.
    # Only declarations: never pretend that an absent implementation exists.
    if public_binary64:
        for name in ('sin', 'tanh'):
            aliases.append(f'__device__ __noinline__ double nssl_{name}(double a) '
                           f'{{ return mynn_{name}(a); }}')
    else:
        aliases.extend(('extern __device__ double nssl_sin(double);',
                        'extern __device__ double nssl_tanh(double);'))
    data = shared + ('\n'+'\n'.join(aliases)+'\n').encode()
    assembled = out/('shared-math-public-PROPOSED.cuh' if public_binary64
                    else 'shared-math-incomplete.cuh')
    assembled.write_bytes(data)
    target = load_target()
    packet = dict(source_pin=target.SOURCE_PIN, source_sha256=target.SOURCE_SHA256,
        math=dict(path=str(assembled.resolve()), sha256=hashlib.sha256(data).hexdigest()),
        nssl_mp_params={}, production_acceptance=False,
        certification=('UNAPPLIED_SHARED_PUBLIC_API_PROPOSAL_DIAGNOSTIC_ONLY'
                       if public_binary64 else 'INCOMPLETE_DEPENDENCY_DIAGNOSTIC_ONLY'),
        unresolved=[] if public_binary64 else ['double nssl_sin(double)', 'double nssl_tanh(double)'])
    (out/'packet.json').write_text(json.dumps(packet, indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(
        existing_shared_header=str(header.resolve()), shared_header_sha256=expected,
        packet=packet, assembler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_transcendental_implementations=0, builtin_substitutions=0,
        public_binary64=public_binary64, installed=False,
        claim='Shared helper alias assembly; numerical identity requires a separate replay'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--public-binary64', action='store_true',
                        help='Explicit diagnostic of the unapplied owner header public double sin/tanh')
    args = parser.parse_args()
    assemble(args.header, args.sha256, args.out, public_binary64=args.public_binary64)
