"""Summarize current strict-native gcov counters with explicit input receipts.

This reads coverage metadata, never weather fields. Prior edge explanations
are retained only for the same native line and branch number; missing outcomes
remain named and are never counted as covered.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket

RANGES = {'mp_gt_driver': (1001, 1387), 'mp_thompson': (1400, 3642),
    'eff_aero': (4176, 4212), 'activ_ncloud': (4391, 4466), 'iceDeMott': (4662, 4729),
    'iceKoop': (4736, 4761), 'delta_p': (4766, 4795), 'calc_effectRad': (4809, 4913)}

FIXED_AEROSOL_EDGES = {(1630,1),(1668,1),(1679,0),(1694,1),
    (2324,1),(2372,1),(2386,1),(2679,1),(2871,0),(3051,1),
    (3059,5),(3108,1),(3124,0)}

CONSERVATION_CATEGORIES = {2597:'cloud water',2611:'cloud ice',
    2623:'rain',2636:'snow',2648:'graupel'}

DRIVER_OPTIONAL_EDGES = {(1136,1),(1139,1),(1196,1),(1200,1)}
DRIVER_CLASSIC_EDGES = {(1168,0),(1176,0),(1176,1),(1211,1)}
DRIVER_NEGATIVE_LINES = {1242,1244,1254,1256,1266,1268,1278,1280,
    1290,1292,1302,1304,1314,1316}

FINITE_EDGE_REASONS = {
    (2198, 0): ('unreachable unordered Stokes edge for finite native intermediates',
        'The strict GNU optimized CFG reaches this test only after stoke_g>=0.4 '
        'has failed. Every finite such value satisfies stoke_g<0.4; the remaining '
        'edge requires an unordered value. GNU duplicates the later high-range '
        'path, sending stoke_g>=0.4 and stoke_g>10 directly to Ef_gw=0.77. '
        'aero-collector-native-optimized.txt retains the exact native CFG. '
        'All three finite ranges are exercised in aero-upper-gcov.txt; the '
        'elided Ef_gw=0 assignment is its PHI zero input, not an untested range.'),
    (2200, 1): ('unreachable unordered Stokes edge for finite native intermediates',
        'After excluding [0.4,10] and values below 0.4, every finite stoke_g '
        'exceeds 10. The strict GNU optimized CFG sends that high range '
        'directly to its Ef_gw=0.77 PHI input; the residual path preserves '
        'the earlier efficiency only for an unordered value. See '
        'aero-collector-native-optimized.txt and aero-upper-gcov.txt.'),
    (2417, 2): ('unreachable second OR true outcome when idx_i < ntb_i',
        'MPAS :1690-1709 bounds the initial ice diameter to 300 micrometers. '
        'The 9999e3 number cap in the lower-diameter correction cannot violate '
        'the 1 mm threshold while ri < r_i(ntb_i)=1e-3: with bm_i=3, mu_i=0, '
        'rho_i=890 and cig(2)=6 it bounds xDi below 133 micrometers. '
        'ri and ni are unchanged before :2397. When idx_i=ntb_i, the first OR '
        'operand succeeds and the second operand is not evaluated.'),
    (2450, 1): ('unreachable nonnegative graupel sublimation with finite positive intermediates',
        'MPAS :2446 requires ssati < -eps. At :2170-2175, alphsc >= 1e-9, '
        'xsat <= 0 and the sublimation polynomial has only positive terms; '
        'gamsc > 0 for ice temperatures and positive density/diffusivity/conductivity. '
        'Every other :2447-2449 factor is positive. The product is strictly negative '
        'in the valid native thermodynamic domain, so :2453 cannot execute. '
        'This does not classify nonfinite or underflow-invalid native states.'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(gcov, prior=None, evidence=()):
    previous = json.loads(prior.read_text()) if prior else {'routines': {}}
    lines = {}
    branches = {}
    location = 0
    for row in gcov.read_text().splitlines():
        match = re.match(r'\s*([^:]+):\s*(\d+):(.*)', row)
        if match:
            count, number, source = match.groups()
            location = int(number)
            count = count.strip().rstrip('*')
            if location and count != '-':
                lines[location] = (int(count) if count.isdigit() else 0, source)
            continue
        match = re.match(r'branch\s+(\d+)\s+(?:taken\s+(\d+)|never executed)', row)
        if match:
            index, count = match.groups()
            branches[(location, int(index))] = int(count or 0)
    routines = {}
    for name, (lo, hi) in RANGES.items():
        old = previous['routines'].get(name, {})
        local_lines = {n: value for n, value in lines.items() if lo <= n <= hi}
        local_branches = {key: value for key, value in branches.items() if lo <= key[0] <= hi}
        explanations = {(e['line'], e['branch']): e
                        for e in old.get('uncovered_outcomes', [])}
        missing = []
        for (n, b), count in local_branches.items():
            if count:
                continue
            entry = dict(explanations.get((n, b), {}))
            entry.update(line=n, branch=b, source=lines.get(n, (0, ''))[1].strip())
            entry.setdefault('classification', 'unresolved optimized edge; retained for targeted work')
            if name == 'mp_gt_driver':
                if (n,b) in DRIVER_OPTIONAL_EDGES:
                    entry['classification']='optional array absent outside the MPAS aerosol call'
                    entry['reason']=('The native MPAS aerosol call supplies snowncv, '
                        'snownc, graupelncv and graupelnc. Their absent branches '
                        'belong to other callers of this shared Fortran driver; '
                        'the complete MPAS argument list is replayed here.')
                elif (n,b) in DRIVER_CLASSIC_EDGES:
                    entry['classification']='fixed aerosol option'
                    entry['reason']=('The native aerosol initializer sets '
                        'is_aerosol_aware=true; the classic aerosol profile and '
                        'its loop cannot execute in this scheme.')
                elif n in DRIVER_NEGATIVE_LINES:
                    entry['classification']='negative condensate/number warning excluded by native cleanup'
                    entry['reason']=('Native :1662-1770 zeroes absent condensates '
                        'and their number carriers before the no_micro return. '
                        'Active columns pass :3578-3639, which zeroes nonpositive '
                        'condensates and bounds their numbers for positive finite '
                        'density. Hence these warnings and their logging-array '
                        'constructor loops cannot execute for valid finite states. '
                        'Vapor is different: :1821 can leave incoming negative '
                        'qv untouched; its :1321-1333 cleanup is reached by the '
                        'negative-vapor corpus and is not classified here.')
                elif n==1387:
                    entry['classification']='compiler optional descriptor absent outside the MPAS call'
                    entry['reason']=('The strict GNU optimized entry CFG tests '
                        'eleven optional dummy pointers before computing their '
                        'array strides; gcov attributes these tests to END '
                        'SUBROUTINE. All eleven arrays are supplied by the MPAS '
                        'aerosol call. aero-driver-native-optional-cfg.txt retains '
                        'the exact pointer tests and profile counters. This is '
                        'an absent-argument edge, not a physical cleanup branch.')
            if name == 'mp_thompson' and (n, b) in FINITE_EDGE_REASONS:
                entry['classification'], entry['reason'] = FINITE_EDGE_REASONS[(n, b)]
            if name == 'mp_thompson' and (n,b) in FIXED_AEROSOL_EDGES:
                entry['classification']='fixed aerosol option'
                entry['reason']=('The aerosol initializer sets is_aerosol_aware=true; '
                    'dustyIce and homogIce are native true parameters at :88-89. '
                    'The false aerosol-mode operand cannot occur in this scheme. '
                    'Other operands at this source line retain their actual counters.')
            if name == 'mp_thompson' and n in CONSERVATION_CATEGORIES and b==3:
                entry['classification']='redundant absent-category conservation test'
                entry['reason']=(f'When the native {CONSERVATION_CATEGORIES[n]} presence '
                    'flag is false, :1662-1770 gives its mass the positive R1 floor. '
                    'Sink rates are initialized to zero at :1532-1618 and their '
                    'category-dependent calculations at :1933-2548 require that '
                    'presence flag (or its positive diameter/mass threshold). '
                    'Thus their sum is zero and cannot be below -mass*odts. '
                    'After sump<rate_max succeeds, the presence flag is true. '
                    'This explanation requires finite valid native intermediates.')
            missing.append(entry)
        routines[name] = dict(source_range=[lo, hi], lines=len(local_lines),
            lines_hit=sum(count > 0 for count, _ in local_lines.values()),
            branches=len(local_branches),
            outcomes_taken=sum(count > 0 for count in local_branches.values()),
            uncovered_lines=[n for n, (count, _) in local_lines.items() if not count],
            uncovered_outcomes=missing)
    receipt = dict(acceptance=False, gcov_host=socket.gethostname(),
        input='Current-build cumulative counters; exact input receipts listed below. '
              'Nonfinite native stress remains explicit.',
        gcov_sha256=digest(gcov), prior_explanations_sha256=digest(prior) if prior else None,
        evidence=[dict(path=str(p), sha256=digest(p)) for p in evidence],
        module=dict(lines=len(lines), lines_hit=sum(count > 0 for count, _ in lines.values()),
                    branches=len(branches), outcomes_taken=sum(count > 0 for count in branches.values())),
        routines=routines)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gcov', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--evidence', type=Path, nargs='+', default=[])
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.gcov, args.prior, args.evidence), indent=2) + '\n')
