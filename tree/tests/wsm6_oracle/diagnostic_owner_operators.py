"""Compose unaccepted WSM6 diagnostics from unchanged existing operations.

This is source/ABI orchestration. It never supplies a production factory or
implements a mathematical operation. The caller supplies frozen owner source,
its expected SHA256, and the earlier explicitly unaccepted diagnostic header.
All operation bodies are copied byte for byte; only ABI forwarding is added.
"""
from pathlib import Path
import argparse
import hashlib
import json


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def assemble(owner_source, expected_sha256, base_header, output):
    source = owner_source.read_text()
    if digest(source) != expected_sha256:
        raise ValueError('committed owner source identity differs')
    base = json.loads(base_header.with_suffix('.json').read_text())
    text = base_header.read_text()
    if base['acceptance'] or digest(text) != base['header_sha256']:
        raise ValueError('base diagnostic identity or acceptance differs')
    blocks = []
    lines = source.splitlines(True)
    for prefix in ('__device__ bool tc_nan(float',
                   '__device__ float tc_quiet(float',
                   '__device__ float tc_pow(float'):
        matches = [line for line in lines if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f'owner operation is not unique: {prefix}')
        blocks.append(matches[0])
    start = source.index('struct cd { double re, im; };')
    blocks.append(source[start:source.index('\n', start)+1])
    for name in ('cd_make', 'cd_div'):
        start = source.index('__device__ __forceinline__ cd '+name+'(')
        blocks.append(source[start:source.index('\n}', start)+3])
    old_pow = '__device__ float mp_pow(float x,float y) {return thompson_aa_powf(x,y);}'
    old_div = '__device__ cx mp_complex_div(cx x,cx y) {return x/y;}'
    if text.count(old_pow) != 1 or text.count(old_div) != 1:
        raise ValueError('base ABI does not match the measured diagnostic')
    forwards = ('\n__device__ float mpas_powf(float x,float y) '
                '{return thompson_aa_powf(x,y);}\n')
    text = text.replace(old_pow, forwards+'\n'.join(blocks)+
                        '\n\n__device__ float mp_pow(float x,float y) {return tc_pow(x,y);}')
    text = text.replace(old_div,
        '__device__ cx mp_complex_div(cx x,cx y) '
        '{cd r=cd_div(cd_make(x.real(),x.imag()),cd_make(y.real(),y.imag()));return cx(r.re,r.im);}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    certificate = dict(scope=__doc__, acceptance=False,
        header_sha256=digest(text), base_header_sha256=digest(base_header.read_text()),
        owner_source=dict(path=str(owner_source.resolve()), sha256=expected_sha256),
        unchanged_owner_bodies=[dict(text=b, sha256=digest(b)) for b in blocks],
        uncertified_operations=['sin(double)', 'atan(double)',
            'CuPy complex log/sqrt/abs', 'CuPy complex4 division',
            'full provider glibc2.43 identity'])
    output.with_suffix('.json').write_text(json.dumps(certificate, indent=2)+'\n')
    return certificate


def reuse_shared_division(shared_source, expected_sha256, base_header, output):
    """Measure the existing shared TEMPO division, without changing its body."""
    source = shared_source.read_text()
    if digest(source) != expected_sha256:
        raise ValueError('shared source identity differs')
    certificate = json.loads(base_header.with_suffix('.json').read_text())
    text = base_header.read_text()
    if certificate['acceptance'] or digest(text) != certificate['header_sha256']:
        raise ValueError('base diagnostic identity or acceptance differs')
    start = source.index('struct tcplx {')
    stop = source.index('\nTEMPO_HD double tc_abs', start)
    # Preserve the complete existing struct and scalar complex operators.
    # The macro supplies their device attribute, not a different operation.
    block = source[start:stop]
    prefix = '__device__ cx mp_complex_div(cx x,cx y) {'
    start = text.index(prefix)
    stop = text.index('\n', start)
    if text.count(prefix) != 1:
        raise ValueError('diagnostic double-complex ABI is not unique')
    forwards = ('\n#define TEMPO_HD __device__ __forceinline__\n'+block+
        '\n#undef TEMPO_HD\n'+
        '__device__ cx mp_complex_div(cx x,cx y) '
        '{tcplx r=tc_div(tc_make(x.real(),x.imag()),tc_make(y.real(),y.imag()));'
        'return cx(r.re,r.im);}')
    text = text[:start]+forwards+text[stop:]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    result = dict(scope='Existing shared TEMPO complex division through WSM6 ABI; diagnostic only',
        acceptance=False, header_sha256=digest(text),
        base_header_sha256=digest(base_header.read_text()),
        base_diagnostic_manifest=certificate,
        shared_source=dict(path=str(shared_source.resolve()), sha256=expected_sha256),
        unchanged_shared_body=dict(text=block, sha256=digest(block)),
        uncertified_operations=certificate['uncertified_operations'])
    output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def reuse_shared_complex(shared_source, expected_sha256, base_header, output):
    """Qualify the pinned engine's existing complex operations as a whole pack."""
    source = shared_source.read_text()
    if digest(source) != expected_sha256:
        raise ValueError('shared complex source identity differs')
    certificate = json.loads(base_header.with_suffix('.json').read_text())
    text = base_header.read_text()
    if certificate['acceptance'] or digest(text) != certificate['header_sha256']:
        raise ValueError('base diagnostic identity or acceptance differs')
    start = source.index('struct thompson_aa_complex { double re, im; };')
    stop = source.index('// m_complex_maxwellgarnett ', start)
    block = source[start:stop]
    forwards = {
        '__device__ cx mp_complex_div(cx x,cx y)':
            '{auto r=thompson_aa_cdiv(thompson_aa_cmake(x.real(),x.imag()),'
            'thompson_aa_cmake(y.real(),y.imag()));return cx(r.re,r.im);}',
        '__device__ cx mp_log(cx x)':
            '{auto r=thompson_aa_clog(thompson_aa_cmake(x.real(),x.imag()));return cx(r.re,r.im);}',
        '__device__ cx mp_sqrt(cx x)':
            '{auto r=thompson_aa_csqrt(thompson_aa_cmake(x.real(),x.imag()));return cx(r.re,r.im);}',
        '__device__ double mp_abs(cx x)':
            '{return thompson_aa_cabs(thompson_aa_cmake(x.real(),x.imag()));}',
    }
    for prefix, body in forwards.items():
        if text.count(prefix) != 1:
            raise ValueError('shared complex ABI is not unique: '+prefix)
        start = text.index(prefix)
        stop = text.index('\n', start)
        text = text[:start]+prefix+' '+body+text[stop:]
    # All referenced scalar dependencies already exist in the base shared pack.
    if text.count('__device__ float mp_exp(float x)') != 1:
        raise ValueError('shared complex insertion point is not unique')
    text = text.replace(
        '__device__ float mp_exp(float x)', block+'\n__device__ float mp_exp(float x)', 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    result = dict(scope='Pinned shared complex operation pack through WSM6 ABI; diagnostic only',
        acceptance=False, header_sha256=digest(text),
        base_header_sha256=digest(base_header.read_text()),
        base_diagnostic_manifest=certificate,
        shared_source=dict(path=str(shared_source.resolve()), sha256=expected_sha256),
        unchanged_shared_body=dict(text=block, sha256=digest(block)),
        uncertified_operations=['full provider glibc2.43 identity',
            'shared complex operations outside their documented finite moderate domain',
            'sin(double)', 'cos(double) versus glibc', 'atan(double)',
            'CuPy complex4 sqrt/division'])
    output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner_source', type=Path)
    parser.add_argument('owner_sha256')
    parser.add_argument('base_header', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--shared-division', action='store_true',
        help='reuse TEMPO division from the supplied pinned shared source instead')
    parser.add_argument('--shared-complex', action='store_true',
        help='reuse the pinned engine complex pack from the supplied source instead')
    args = parser.parse_args()
    if args.shared_division and args.shared_complex:
        parser.error('select one shared operation pack')
    method = (reuse_shared_complex if args.shared_complex else
              reuse_shared_division if args.shared_division else assemble)
    method(args.owner_source, args.owner_sha256, args.base_header, args.output)
