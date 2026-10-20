"""Preserve GNU's bitwise floating unary sign in the entire CAM candidate.

MPAS module_ra_cam.F:7308 negates the complete quotient. NVRTC's floating
negation canonicalizes a NaN instead of flipping its sign; explicit integer
XOR preserves the native sign and payload. This changes no finite arithmetic.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HELPER = r'''
// MPAS module_ra_cam.F:7308: GNU unary sign preserves NaN sign/payload.
template<class T> CAM_HD T cam_neg(T a) {
#if defined(__CUDA_ARCH__)
  if constexpr (CamSame<T,double>::value) {
    unsigned long long bits=__double_as_longlong(a);
    asm("xor.b64 %0,%1,0x8000000000000000;" : "=l"(bits) : "l"(bits));
    return __longlong_as_double(bits);
  }
  if constexpr (CamSame<T,float>::value) {
    unsigned int bits=__float_as_uint(a);
    asm("xor.b32 %0,%1,0x80000000;" : "=r"(bits) : "r"(bits));
    return __uint_as_float(bits);
  }
#endif
  return -a;
}
'''


def rewrite(code):
    result = []
    count = 0
    at = 0
    while at < len(code):
        begin = code.find('(-', at)
        if begin < 0:
            result.append(code[at:])
            break
        result.append(code[at:begin])
        depth, end = 1, begin+2
        while depth:
            if end >= len(code):
                raise ValueError('Unbalanced parenthesized CAM unary expression')
            depth += (code[end] == '(') - (code[end] == ')')
            end += 1
        inner, nested = rewrite(code[begin+2:end-1])
        result.append('cam_neg('+inner+')')
        count += 1+nested
        at = end
    return ''.join(result), count


def transform(code):
    parts, count = [], 0
    for line in code.splitlines(keepends=True):
        # GNU folds finite parameter signs at compile time.
        if 'static constexpr' in line:
            parts.append(line)
        else:
            changed, n = rewrite(line)
            parts.append(changed)
            count += n
    code = ''.join(parts)
    marker = 'template<class A,class B> CAM_HD constexpr auto cam_add'
    if code.count(marker) != 1 or not count:
        raise ValueError('Complete CAM unary transformation anchor missing')
    code = code.replace(marker, HELPER+'\n'+marker)
    return code, count


def prepare(source, output, receipt):
    spec = importlib.util.spec_from_file_location('cam_unary_base', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    code, count = transform(module.CUDA_SOURCE)
    text = source.read_text()
    marker = 'CUDA_SOURCE = r'+chr(34)*3
    start = text.index(marker)+len(marker)
    end = text.index('\n'+chr(34)*3, start)
    text = text[:start]+code+text[end:]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    spec = importlib.util.spec_from_file_location('cam_unary_emitted', output)
    emitted = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = emitted
    spec.loader.exec_module(emitted)
    receipt.write_text(json.dumps(dict(
        scope='whole CAM GNU unary-sign candidate; full replay qualification required',
        base_module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        cuda_source_sha256=hashlib.sha256(emitted.CUDA_SOURCE.encode()).hexdigest(),
        expressions=count, promoted=False), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.receipt)
