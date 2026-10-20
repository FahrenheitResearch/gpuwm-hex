"""Bind every uncovered edge to the original native source and driver scope.

MPAS physics_mmm/bl_mynn.F90:405-419,1029-1057 fixes chemistry off
outside WRF_CHEM. MPAS physics_mmm/mynn_shared.F90:72-103 defines
waterice optional; every pinned MMM caller omits it. These observations
classify coverage only: no output word or required input bank is excluded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(args):
    union = json.loads(args.coverage.read_text())
    mmm = args.native / 'src/core_atmosphere/physics/physics_mmm'
    wrf = args.native / 'src/core_atmosphere/physics/physics_wrf'
    paths = {name: (mmm if name.endswith('.F90') else wrf) / name
             for name in union['files']}
    for name, path in paths.items():
        if sha(path) != union['files'][name]['native_source_sha256']:
            raise ValueError('Coverage refers to different science: '+name)
    run = paths['bl_mynn.F90'].read_text().splitlines()
    assert run[404].strip() == '#if(WRF_CHEM == 1)'
    assert run[413].strip() == '#else'
    assert run[414].strip() == 'logical,parameter:: mix_chem = .false.'
    assert run[418].strip() == '#endif'
    subr = paths['bl_mynn_subroutines.F90'].read_text().splitlines()
    calls = []
    for name, path in paths.items():
        for line, text in enumerate(path.read_text().splitlines(), 1):
            if text.lstrip().startswith('!'):
                continue
            match = re.search(r'\bqsat_blend\s*\((.*)', text, re.I)
            if match and not re.search(r'\bfunction\b', text, re.I):
                # All current call sites fit on one source line. Count top-level
                # arguments, preserving nested array references and expressions.
                depth, count = 0, 1
                for char in match[1]:
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        if depth == 0:
                            break
                        depth -= 1
                    elif char == ',' and depth == 0:
                        count += 1
                if count != 4:
                    raise ValueError('Optional waterice is now supplied: '+name)
                calls.append(dict(file=name, line=line, arguments=count,
                                  source=text.strip()))
    assert len(calls) == 6
    shared = paths['mynn_shared.F90'].read_text().splitlines()
    assert shared[80].strip() == "wrt = 'b'"
    scale_machine = None
    if args.scale_aware_witness:
        scale_machine = json.loads(args.scale_aware_witness.read_text())
        for item in scale_machine['files']:
            if sha(Path(item['path'])) != item['sha256']:
                raise ValueError('Original scale arithmetic witness changed: '+item['path'])
        assert scale_machine['science_sha256'] == sha(paths['bl_mynn_subroutines.F90'])
        assert scale_machine['native_binary32_square_products_sums_division']
        assert scale_machine['shcu_positive_coefficient_order_hex'] == ['3e147ae1', '3e3020c5']
        assert scale_machine['shcu_positive_denominator_addend_hex'] == '3e2e147b'
        assert subr[6297].strip() == 'Psig_bl= ((dxdh**2) + 0.106*(dxdh**0.667))/((dxdh**2) +0.066*(dxdh**0.667) + 0.071)'
        assert subr[6327].strip() == 'Psig_shcu= ((dxdh**2) + 0.145*(dxdh**0.667))/((dxdh**2) +0.172*(dxdh**0.667) + 0.170)'
    machine = None
    if args.machine_witness:
        machine = json.loads(args.machine_witness.read_text())
        for item in machine['files']:
            if sha(Path(item['path'])) != item['sha256']:
                raise ValueError('Original machine witness changed: '+item['path'])
        assert machine['science_sha256'] == sha(paths['bl_mynn_subroutines.F90'])
        assert subr[1687].strip() == 'elsq = el (k)**2'
        assert subr[1688].strip() == 'q3sq = qkw(k)**2'
        limit = 'IF ( q3sq/dlsq .LT. -gh(k) ) q3sq = -dlsq*gh(k)'
        assert subr[1736].strip() == subr[1881].strip() == limit
        # No source write can invalidate the first limit before its duplicate.
        for stmt in subr[1737:1881]:
            if stmt.lstrip().startswith('!'):
                continue
            assert not re.match(r'\s*(q3sq|elsq|gh\s*\(k\))\s*=', stmt, re.I)
        assert subr[2899].strip() == 'q1k=max(Q1(k),-2.5)'
        assert subr[2901].strip() == 'q1k=max(Q1(k),-2.0)'
        assert machine['binary32_square_then_binary64']
        assert machine['all_four_q1_clamps_return_lower_constant_on_unordered']
    rows = []
    for name, coverage in union['files'].items():
        for original in coverage['untaken_branches']:
            row = dict(original, file=name,
                       classification='OPEN: not witnessed; no unreachability claim')
            routine, line = row['routine'], row['line']
            if name == 'bl_mynn_subroutines.F90' and (
                    routine in ('mynn_mix_chem', 'tridiag3') or
                    (routine == 'dmp_mf' and line > 4800 and ('mix_chem' in row['source'] or
                     'chem' in row['source'].lower() or line in
                     (4852, 4864, 4882, 5111, 5112, 5197, 5321,
                      5401, 5402, 5403, 5456, 5491, 5492, 5493,
                      5498, 5499, 5500)))):
                row['classification'] = 'UNREACHABLE in pinned MPAS driver: WRF_CHEM chemistry'
                row['reason'] = ('bl_mynn.F90:405-419 fixes mix_chem=.false.; '
                                 '1029-1057 guards the only mynn_mix_chem calls; '
                                 'bl_mynn_subroutines.F90:4305 is the only active '
                                 'tridiag3 call, inside that routine. Chemistry '
                                 'routines remain ported; this is no runtime refusal.')
                if routine == 'dmp_mf':
                    row['source_context'] = subr[max(0, line-5):line+1]
            elif name == 'mynn_shared.F90' and line == 80 and row['branch'] == 0:
                row['classification'] = 'UNREACHABLE in pinned MMM call graph: optional waterice supplied'
                row['reason'] = 'All six original callers supply four arguments; waterice is always absent.'
            elif name == 'mynn_shared.F90' and line == 88 and row['branch'] == 2:
                row['classification'] = 'UNREACHABLE in pinned MMM call graph: wrt equals water'
                row['reason'] = "mynn_shared.F90:80-84 sets wrt='b' for every actual caller."
            elif machine and name == 'bl_mynn_subroutines.F90' and (
                    routine, line, row['branch']) == ('mym_turbulence', 1882, 0):
                row['classification'] = 'UNREACHABLE in pinned strict arithmetic: duplicate q3 limit'
                row['reason'] = (
                    'Lines1688-1689 square binary32 el/qkw before widening; '
                    '1641-1642 store elsq/q3sq/dlsq in binary64. Lines1736-1737 '
                    'already apply this exact limit, with no intervening write '
                    'to elsq/q3sq/gh. If the first comparison is false, the '
                    'second is unchanged. If it is true with finite nonzero '
                    'elsq and finite gh, their binary64 product is exact '
                    '(at most48significand bits; exponents inside binary64), '
                    'and division recovers -gh exactly, so strict LT is false. '
                    'Zero/NaN elsq cannot trigger the first limit; infinite '
                    'elsq or gh produces an infinite or unordered second '
                    'comparison, also false. Original MULSS/CVTSS2SD/mixed '
                    'precision instruction witness is retained separately.')
            elif machine and name == 'bl_mynn_subroutines.F90' and (
                    routine, line, row['branch']) == ('mym_condensation', 2920, 1):
                row['classification'] = 'UNREACHABLE in pinned strict arithmetic: lower q1 clamp'
                row['reason'] = (
                    'Lines2899-2904 clamp q1k to at least -2.5 (water) or '
                    '-2.0 (land). All four original MAXSS paths return their '
                    'constant source operand for unordered Q1 as well. '
                    'Therefore q1k is ordered and >=-2.5; after rejecting '
                    'q1k>=1 and q1k>=-1.7, the final >=-2.5/<-1.7 test '
                    'must hold. The line2923 else is unreachable, including '
                    'NaN/Inf inputs. No runtime branch or output is removed.')
            elif scale_machine and name == 'bl_mynn_subroutines.F90' and (
                    routine, line, row['branch']) in (
                        ('scale_aware', 6339, 0), ('scale_aware', 6341, 0),
                        ('scale_aware', 6342, 0)):
                row['classification'] = 'UNREACHABLE in pinned strict arithmetic: scale factor clamp'
                row['reason'] = scale_machine['proof']
            rows.append(row)
    result = dict(scope=__doc__, process_count=union['process_count'],
                  coverage_sha256=sha(args.coverage),
                  science_sources={name: dict(path=str(path), sha256=sha(path))
                                   for name, path in paths.items()},
                  qsat_call_sites=calls, uncovered_edges=rows,
                  strict_machine_witness=machine,
                  scale_aware_machine_witness=scale_machine,
                  full_branch_coverage=False, excluded_words=[],
                  original_banks_retained=True)
    result['counts'] = {kind: sum(row['classification'].startswith(kind) for row in rows)
                        for kind in ('OPEN', 'UNREACHABLE')}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--machine-witness', type=Path)
    parser.add_argument('--scale-aware-witness', type=Path)
    audit(parser.parse_args())
