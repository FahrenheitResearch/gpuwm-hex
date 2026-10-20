"""Assemble existing shared device maths and MIT CORE-MATH for whole MYNN replay.

This is an explicit test resource, never an elementary-math fallback or a
production default. Preserve upstream arithmetic and coefficients. The shared
resource owner must publish the authenticated header before forecast use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def existing_fma_math(path):
    """Borrow only existing glibc math blocks, no WRF physics routines.

    The loaded native glibc IFUNC contracts these operations. This extraction
    preserves every function/table byte, adding only a namespace and real alias.
    """
    source = path.read_text()
    names = ('mynn_add', 'mynn_sub', 'mynn_mul', 'mynn_div', 'mynn_gt',
             'mynn_d2f_rn', 'mynn_glibc_logf', 'mynn_glibc_powf_log2',
             'mynn_glibc_powf_exp2', 'mynn_glibc_expf',
             'mynn_glibc_powf_checkint', 'mynn_glibc_zeroinfnan',
             'mynn_glibc_powf')
    macros = ('MYNN_ADD', 'MYNN_SUB', 'MYNN_MUL', 'MYNN_DIV',
              'MYNN_DADD', 'MYNN_DSUB', 'MYNN_DMUL', 'MYNN_DFMA')
    tables = ('MYNN_LOGF_TAB', 'MYNN_LOGF_MISC', 'MYNN_POWF_LOG2_TAB',
              'MYNN_POWF_LOG2_POLY', 'MYNN_EXP2F_TAB', 'MYNN_EXP2F_POLY',
              'MYNN_POWF_MISC', 'MYNN_LIBM_F32', 'MYNN_EXPF_MISC')
    ranges, blocks = [], []
    def retain(start, end, name):
        text = source[start:end]
        ranges.append(dict(name=name, first_line=source.count('\n',0,start)+1,
                           last_line=source.count('\n',0,end)+1,
                           sha256=digest(text.encode())))
        blocks.append(text)
    for name in macros:
        m = re.search(r'^#define '+name+r'\([^\n]+', source, re.M)
        if m is None: raise ValueError('Missing existing math macro '+name)
        retain(m.start(),m.end(),name)
    for name in tables:
        m = re.search(r'__constant__[^\n]+\b'+name+r'\[\d+\]\s*=\s*\{.*?\};',source,re.S)
        if m is None: raise ValueError('Missing existing math table '+name)
        retain(m.start(),m.end(),name)
    for name in names:
        m = re.search(r'__device__[^\n]+\b'+name+r'\([^\n]*\)\s*\{',source)
        if m is None: raise ValueError('Missing existing math function '+name)
        start, pos, depth = m.start(), m.end(), 1
        while depth:
            depth += (source[pos]=='{')-(source[pos]=='}')
            pos += 1
        retain(start,pos,name)
    text = ('// Existing engine glibc IFUNC math only; extraction ranges are authenticated.\n'
            'namespace p12_existing_fma { using real = float;\n' + '\n'.join(blocks) +
            '\n}\n' + '\n'.join('#undef '+name for name in macros)+'\n')
    return text, ranges


def cuda_core_math(source, name):
    # NVRTC supplies the integer types through the preceding shared header.
    # Host errno and floating exception flags are outside device array outputs.
    text = re.sub(r'^#include[^\n]*\n', '', source, flags=re.M)
    text = re.sub(r'^#pragma[^\n]*\n', '', text, flags=re.M)
    text = re.sub(r'^\s*feraiseexcept\([^;]*;\s*', '\n', text, flags=re.M)
    text = text.replace('static __attribute__((noinline)) float as_special',
                        '__device__ __noinline__ float as_special')
    text = text.replace('float cr_' + name + '(', '__device__ __noinline__ float cr_' + name + '(')
    # The pinned NVRTC C++17 parser lacks C designated union initializers.
    # Assign the same named member without changing its expression or bits.
    text = re.sub(r'(b(?:32u32|64u64)_u)\s+(\w+)\s*=\s*\{\.(f|u)\s*=\s*([^;]*)\};',
                  r'\1 \2; \2.\3 = \4;', text)
    replacements = {
        '__builtin_expect': 'p12_expect', '__builtin_fabsf': 'fabsf',
        '__builtin_copysignf': 'copysignf', '__builtin_copysign': 'copysign',
        '__builtin_floor': 'floor', '__builtin_clz': '__clz',
        '__builtin_fmaf': '__fmaf_rn',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return 'namespace p12_core_' + name + ' {\n' + text + '\n}\n'


def assemble(args):
    originals = {}
    pieces = ['// Explicit whole-MYNN replay resource; upstream source hashes in adjacent receipt.\n']
    for name in ('glibc_flt64.cuh', 'glibc_flt32.cuh', 'glibc_trig_flt32.cuh'):
        path = args.shared / name
        raw = path.read_bytes()
        originals[str(path)] = digest(raw)
        pieces.append(raw.decode())
    pieces.append('#define p12_expect(value, expected) (value)\n')
    transformed = {}
    for name in ('log10f', 'tanhf', 'erff'):
        path = args.core_math / (name + '.c')
        raw = path.read_bytes()
        originals[str(path)] = digest(raw)
        text = cuda_core_math(raw.decode(), name)
        transformed[name] = digest(text.encode())
        pieces.append(text)
    fma_ranges = []
    if args.fma_math is not None:
        originals[str(args.fma_math)] = digest(args.fma_math.read_bytes())
        text, fma_ranges = existing_fma_math(args.fma_math)
        pieces.append(text)
    pieces.append('''
#undef p12_expect
__device__ float mynn_pow(float a,float b) { return gfk_pow(a,b); }
__device__ double mynn_pow(double a,double b) { return glibc_pow(a,b); }
__device__ double mynn_pow(double a,float b) { return glibc_pow(a,(double)b); }
__device__ double mynn_pow(float a,double b) { return glibc_pow((double)a,b); }
__device__ float mynn_exp(float a) { return gfk_exp(a); }
__device__ double mynn_exp(double a) { return glibc_exp(a); }
__device__ float mynn_log(float a) { return gfk_log(a); }
__device__ double mynn_log(double a) { return glibc_log(a); }
__device__ float mynn_log10(float a) { return p12_core_log10f::cr_log10f(a); }
__device__ float mynn_atan(float a) { return glibc_atanf(a); }
__device__ float mynn_tanh(float a) { return p12_core_tanhf::cr_tanhf(a); }
__device__ float mynn_erf(float a) { return p12_core_erff::cr_erff(a); }
// No narrowing fallback for a binary64 call not covered by this resource.
__device__ double mynn_log10(double) = delete;
__device__ double mynn_atan(double) = delete;
__device__ double mynn_tanh(double) = delete;
__device__ double mynn_erf(double) = delete;
''')
    header = '\n'.join(pieces)
    if args.fma_math is not None:
        header = header.replace('return gfk_pow(a,b);', 'return p12_existing_fma::mynn_glibc_powf(a,b);')
        header = header.replace('return gfk_exp(a);', 'return p12_existing_fma::mynn_glibc_expf(a);')
        header = header.replace('return gfk_log(a);', 'return p12_existing_fma::mynn_glibc_logf(a);')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header)
    args.receipt.write_text(json.dumps(dict(
        scope='Existing shared math source assembly for full CUDA replay; not certified arithmetic or forecast proof',
        source_sha256=originals, transformed_core_math_sha256=transformed,
        existing_fma_math_source_ranges=fma_ranges,
        output_sha256=digest(header.encode()), coefficients_and_arithmetic_changed=False,
        transformations=['CUDA function qualifiers', 'isolated namespaces', 'NVRTC built-in spelling',
                         'designated union initialization becomes assignment to the same member',
                         'omit host includes/pragmas and host exception flags'],
        output=str(args.output), production_default=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('shared', 'core-math', 'output', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fma-math',type=Path,help='Existing engine glibc FMA math blocks, never its WRF scheme')
    assemble(parser.parse_args())
