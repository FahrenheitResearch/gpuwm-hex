"""Keep uncovered branches explicit, with source-bound entry invariants.

This does not turn an uncovered branch into a covered one, and excludes no
output words. Direct helper interfaces may reach modes fixed by the entry.
"""
import argparse
import hashlib
import json
from pathlib import Path


def classify(coverage, source, receipt):
    data = json.loads(coverage.read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != data['authority_sha256']:
        raise ValueError('coverage classification source differs from gcov authority')
    invariants = {
        ('cu_grell_freitas', 428): ('accepted-cloud invariant', 'HTOP is initialized to KTS at :168. A precipitating accepted deep cloud has KTOP>=KBCON+2 at :1249 and KBCON>=KTS; the false comparison cannot occur for that cloud.', 'module_cu_gf.mpas.F:167-168,419-429,1240-1249,3039-3044'),
        ('cu_grell_freitas', 429): ('accepted-cloud invariant', 'HBOT is initialized to KTE at :167. An accepted deep cloud has KBCON<=KTOP-2 at :1249; KTOP is initialized to zero at :239 and assigned k-1 with k<=KTF at :1242-1244. Its accepted cloud base is therefore below KTE.', 'module_cu_gf.mpas.F:167-168,239,419-429,1240-1249'),
        ('cup_gf', 877): ('entry invariant', 'IERR is set to zero at :846; no intervening error write before :877.', 'module_cu_gf.mpas.F:838-883'),
        ('cup_gf', 910): ('entry invariant', 'IERR is reset to zero at :846; cup_env and cup_env_clev take IERR as input and do not change it.', 'module_cu_gf.mpas.F:838-846,889-910,2265-2502'),
        ('cup_gf', 946): ('entry invariant', 'AAEQ=0 suppresses the only error write after :910, and cup_maximi takes IERR as input.', 'module_cu_gf.mpas.F:271,910-950,3176-3228'),
        ('cup_gf', 1034): ('monotone-pressure invariant', 'cup_minimi initializes KSTABI=KBCON and can only increase it; k<KBCON in this loop. Strictly decreasing pressure makes p(k)>p(KSTABI).', 'module_cu_gf.mpas.F:988-992,1027-1034,3259-3286'),
        ('cup_gf', 1272): ('positive-height invariant', 'ZKTOP is capped below the accepted cloud-top height; increasing finite positive heights encounter a level above it before exhausting the column.', 'module_cu_gf.mpas.F:1240-1249,1267-1279'),
        ('cup_gf', 1776): ('entry moisture invariant', 'XQ at KTF is overwritten with QO, floored to 1e-8 by the entry before the call.', 'module_cu_gf.mpas.F:282-288,1771-1776'),
        ('cup_gf', 911): ('entry-fixed argument', 'AAEQ is initialized to zero at :271 and passed unchanged by cu_grell_freitas.', 'module_cu_gf.mpas.F:265-277,389-400,905-915'),
        ('cup_gf', 972): ('entry-fixed argument', 'use_excess=0 at :76; the excess=2 branch is a direct-helper alternate mode.', 'module_cu_gf.mpas.F:72-80,968-982'),
        ('cup_gf', 976): ('entry-fixed argument', 'The sum is inside use_excess=2; the entry fixes use_excess=0.', 'module_cu_gf.mpas.F:72-80,968-982'),
        ('cup_gf', 977): ('entry-fixed argument', 'The entry fixes use_excess=0, satisfying use_excess<=1.', 'module_cu_gf.mpas.F:72-80,968-982'),
        ('cup_gf', 1569): ('entry-fixed argument', 'maxens2=1 at :80; the maxens2=3 mode is outside the entry configuration.', 'module_cu_gf.mpas.F:78-83,1563-1576'),
        ('cup_gf', 1918): ('entry-fixed argument', 'maxens=1 at :79; the nonpositive ensemble-count path is outside the entry configuration.', 'module_cu_gf.mpas.F:78-83,1913-1929'),
        ('cup_gf_sh', 4440): ('entry-fixed argument', 'AAEQ is initialized to zero at :271 and passed unchanged by cu_grell_freitas.', 'module_cu_gf.mpas.F:265-277,404-413,4437-4445'),
        ('cup_gf_sh', 4439): ('entry invariant', 'The entry initializes shallow IERRS=0; cup_env and cup_env_clev take IERR as input and do not change it.', 'module_cu_gf.mpas.F:243-244,404-413,2265-2502,4411-4439'),
        ('cup_gf_sh', 4466): ('entry invariant', 'AAEQ=0 suppresses the only earlier shallow error write; cup_maximi takes IERR as input.', 'module_cu_gf.mpas.F:271,4440-4442,4455-4473,3176-3228'),
        ('cup_gf_sh', 4487): ('entry-fixed argument', 'use_excess_sh=0 at :77; the excess=2 branch is a direct-helper alternate mode.', 'module_cu_gf.mpas.F:72-80,4484-4499'),
        ('cup_gf_sh', 4490): ('entry-fixed argument', 'The sum is inside use_excess=2; the entry fixes use_excess_sh=0.', 'module_cu_gf.mpas.F:72-80,4484-4499'),
        ('cup_gf_sh', 4491): ('entry-fixed argument', 'The sum is inside use_excess=2; the entry fixes use_excess_sh=0.', 'module_cu_gf.mpas.F:72-80,4484-4499'),
        ('cup_gf_sh', 4492): ('entry-fixed argument', 'The sum is inside use_excess=2; the entry fixes use_excess_sh=0.', 'module_cu_gf.mpas.F:72-80,4484-4499'),
        ('cup_gf_sh', 4494): ('entry-fixed argument', 'The entry fixes use_excess_sh=0, satisfying use_excess<=1.', 'module_cu_gf.mpas.F:72-80,4484-4499'),
        ('cup_gf_sh', 4625): ('initialized-cloud-top invariant', 'The entry initializes KTOPS=0. DBYO is zeroed through KTF before being computed only through KTF-1, so its first negative element can occur no higher than KTF-1. The search assigns KTOP=k-1, hence KTOP<=KTF-2.', 'module_cu_gf.mpas.F:241-247,4556-4563,4604-4628'),
        ('cup_gf_sh', 4953): ('valid-column invariant', 'cup_up_aa0 clips AA1 to nonnegative at :3347; :4675 rejects AA1=0 before this forcing block.', 'module_cu_gf.mpas.F:3344-3348,4670-4680,4948-4957'),
        ('cup_gf_sh', 5013): ('finite-closure invariant', 'For finite positive XMBMAX the first candidate is zero or max(0,...) capped by XMBMAX. GNU maxval therefore finds a non-NaN first element; its all-NaN initial search outcomes cannot occur.', 'module_cu_gf.mpas.F:4944-4972,5013'),
        ('cup_gf_sh', 5022): ('valid-closure invariant', 'The nine closure candidates use max(0,...) and their nonnegative average is capped by positive xmbmax.', 'module_cu_gf.mpas.F:4935-5026'),
        ('cup_kbcon', 3092): ('valid-column invariant', 'To reach the cap retry, KBCON-K22>=2. The search already enforces KBCON<=KBMAX+2. Incrementing K22 once gives new KBCON<=old KBCON (iloop=5 resets it even lower), so the later overflow test cannot be true.', 'module_cu_gf.mpas.F:3042-3055,3067-3081,3091-3101'),
        ('cup_kbcon', 3093): ('valid-column invariant', 'This error branch is inside the unreachable later overflow test at :3092; the earlier search overflow error remains covered.', 'module_cu_gf.mpas.F:3042-3055,3067-3081,3091-3101'),
    }
    rows = []
    for branch in data['uncovered_branches']:
        kind, reason, citation = invariants.get((branch['routine'], branch['line']),
            ('unreached', 'No entry invariant is claimed; retained as an uncovered native branch.',
             f"module_cu_gf.mpas.F:{branch['line']}"))
        rows.append(dict(branch, classification=kind, reason=reason, citation=citation))
    receipt.write_text(json.dumps(dict(authority_sha256=data['authority_sha256'],
        coverage_sha256=hashlib.sha256(coverage.read_bytes()).hexdigest(),
        branches=data['branches'], branches_taken=data['branches_taken'],
        uncovered_branches=rows, excluded_words=0,
        scope='Source classification only; coverage counts unchanged; no forecast proof'), indent=2)+'\n')
    print(json.dumps({'branches_taken':data['branches_taken'], 'uncovered':len(rows),
                      'unreached_without_invariant':sum(r['classification']=='unreached' for r in rows)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    classify(a.coverage, a.source, a.receipt)
