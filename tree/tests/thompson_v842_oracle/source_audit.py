"""Produce auditable v8.4.2 source windows and native gcov summaries.

This lane inventory uses the supplied read-only v8.4.2 checkout and records
source-window hashes independently of the integrator-owned checker.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re


def branch_reason(name,line):
    """Explain untaken gcov edges without claiming arbitrary input coverage."""
    routine=name.split('_MOD_')[-1]
    if routine=='thompson_init':
        return 'host-preparation',('Strict native initialization reads the pinned table files and builds auxiliary arrays. '
            'Repeated-allocation, alternative aerosol initialization, table-read/write failure and compiler cleanup edges '
            'are not forced by these column calls; table-generation identity has its own receipt.')
    if routine in ('qr_acr_qg','qr_acr_qs','freezeh2o','qi_aut_qs'):
        return 'table-generation-outside-column-scope','Column replay loads the same four f0 native-built files; these generation routines are not called by a runtime mp_thompson column.'
    if routine in ('table_efrw','table_efsw'):
        return 'fixed-native-auxiliary-grid','The pristine initializer visits its fixed size grid; the remaining compound-condition edges are not produced by that grid. No modified auxiliary grid is claimed.'
    if routine=='mp_gt_driver':
        if line in (1136,1139,1196,1200):
            return 'MPAS-always-present','MPAS passes both snow and graupel accumulation arguments at driver_microphysics.F:391-406; absent WRF-style arguments are outside this MPAS boundary.'
        if line in (1168,1169,1211,1216):
            return 'classic-option-invariant','is_aerosol_aware is false for mp_thompson; these edges belong to mp_thompson_aerosols.'
        if line in (1242,1244,1254,1256,1266,1268,1278,1280,1290,1292,1302,1304,1314,1316):
            return 'warning-path-not-reached','The numerical routine zeros incoming nonpositive condensate/number fields or enforces their lower bound before this warning check. No negative post-state or logging failure is observed.'
        return 'logging-or-compiler-cleanup','The remaining edge is a warning call, logger failure or generated array cleanup; numerical negative-vapor interior and boundary updates are reached by cleanup_replay.'
    if routine=='mp_thompson':
        if line in (1630,1631,1668,1679,1694,2324,2372,2386,2679,2871,3051,3059,3087,3089,3091,3108,3124):
            return 'classic-option-invariant','is_aerosol_aware is false for mp_thompson; nested aerosol evaporation and nucleation paths are outside this option.'
        if line in (2041,2043,2060,2062,2074,2076,2089,2091,2102,2104,2118,2120,2133,2135,2146,2148,2333,2335):
            return 'bounded-decimal-search','For positive values in this certified corpus, NINT(log10(x))-1..+1 contains a decade satisfying 1 <= x/10**n < 10; the search exits before loop exhaustion. Extreme overflow/nonfinite inputs are not claimed.'
        if line==2195:
            return 'size-bound-invariant',('F:2191 requires rg >= 1e-5; F:1926 caps N0_exp at 3e6. With mu_g=0, bm_g=3, '
                'rho_g=500, cgg(1)=cgg(3)=6, F:1929-1931 imply xDg >= 4/(3e6*(pi*500/6)*6/1e-5)**0.25, '
                'about 8.6e-4 m, greater than D0g=2.5e-4 m. The false edge cannot occur under these bounds.')
        if line in (2198,2200):
            return 'finite-Stokes-partition','The tested finite Stokes values cover <0.4, [0.4,10] and >10. Once the preceding interval tests fail, the remaining redundant comparison is forced; NaN is not a valid witness.'
        if line==2331:
            return 'classic-prescribed-IN','Classic MPAS assigns xni=1000 at F:2327 before this comparison with Nt_IN(1); the lower-table edge is not reached.'
        if line==2417:
            return 'ice-size-bound','The earlier ice diameter clamp F:1699-1709 limits xDi to 300 micrometres. Before tendencies are applied, the second OR term xDi > 5*D0s (1000 micrometres) cannot be true; idx_i maximum is covered.'
        if line==2450:
            return 'sublimation-sign','The enclosing F:2446 requires ssati < -eps and positive graupel mass. All other finite bounded prefactors are positive, so prg_gde is negative; its deposition else branch is not reached.'
        if line in (2597,2611,2623,2636,2648):
            return 'mass-conservation-invariant','The preceding sink terms require that species to be present. With L_q false their sinks are zero, so sump cannot be below the negative rate_max; the second AND operand cannot be false after the first is true in this corpus.'
        return 'not-yet-reached','No independent reason is established for this edge. Further real captures or targeted columns must resolve it; complete branch coverage is not claimed.'
    if routine=='activ_ncloud':
        return 'clamped-table-search','F:4407-4412 and 4422-4425 clamp inputs inside their finite table endpoints; a matching interval is found before loop exhaustion.'
    if routine=='calc_effectrad':
        if line==4838:return 'classic-option-invariant','Classic mp_thompson always replaces nc with Nt_c.'
        if line==4850:return 'prescribed-count-invariant','has_qc requires Nt_c > R2 at some level; the same prescribed Nt_c is used at every level, so the second OR edge nc <= R2 is then impossible.'
    if routine=='rayleigh_soak_wetgraupel':
        return 'initialized-radar-domain','The initialized particle diameter grid exceeds 1e-12 m and its fixed mixing strings are accepted. Tiny-particle and invalid-mixing error branches are not reached by the Thompson radar call.'
    if routine in ('get_m_mix_nested','get_m_mix','m_complex_maxwellgarnett'):
        return 'initialized-radar-domain','Thompson radar initialization uses its fixed air/water/icewater strings and spheroidal inclusions. Alternative host/matrix/inclusion strings and invalid volume closure are not exercised by this scheme boundary.'
    return 'not-yet-reached','No independent reason is established for this edge; complete coverage is not claimed.'


def audit(reference,production,oracle,receipts,supplemental=()):
    root=reference/'src/core_atmosphere'
    by_name={p.name:p for p in root.rglob('*') if p.is_file() and p.suffix in ('.F','.F90')}
    text=production.read_text()
    windows=Counter()
    for m in re.finditer(r'MPAS\s+(?:[\w]+/)*([\w]+\.F(?:90)?):(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)',text):
        for window in m[2].split(','):
            numbers=window.split('-');lo=int(numbers[0]);hi=int(numbers[-1])
            windows[m[1],lo,hi]+=1
    rows=[]
    for (name,lo,hi),count in sorted(windows.items()):
        source=by_name[name];lines=source.read_bytes().splitlines(keepends=True)
        assert 1<=lo<=hi<=len(lines),(name,lo,hi)
        rows.append(dict(source=str(source.relative_to(reference)),start=lo,end=hi,occurrences=count,
                         sha256=hashlib.sha256(b''.join(lines[lo-1:hi])).hexdigest()))
    (receipts/'source-citation-pins.json').write_text(json.dumps(dict(version='MPAS-A-v8.4.2',
        production_sha256=hashlib.sha256(production.read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
    combined={};coverage_artifacts=[]
    for directory in (oracle,*supplemental):
        for artifact in sorted(directory.glob('*.gcov.json.gz')):
            coverage_artifacts.append(dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
            data=json.loads(gzip.decompress(artifact.read_bytes()))
            for file in data['files']:
                name=Path(file['file']).name
                if name not in ('module_mp_thompson.F','mp_radar.F90'):continue
                if (directory/name).read_bytes()!=by_name[name].read_bytes():
                    raise ValueError(f'{directory/name}: coverage requires pristine source bytes')
                if name not in combined:
                    combined[name]=file;continue
                existing=combined[name]
                functions={f['name']:f for f in existing['functions']}
                for f in file['functions']:
                    g=functions[f['name']]
                    if (f['start_line'],f['end_line'])!=(g['start_line'],g['end_line']):
                        raise ValueError('coverage function ranges differ')
                    g['execution_count']+=f['execution_count']
                lines={(l.get('function_name'),l['line_number']):l for l in existing['lines']}
                for line in file['lines']:
                    other=lines[(line.get('function_name'),line['line_number'])]
                    if len(line['branches'])!=len(other['branches']):raise ValueError('coverage branch layouts differ')
                    other['count']+=line['count']
                    for a,b in zip(line['branches'],other['branches']):
                        if {k:v for k,v in a.items() if k!='count'}!={k:v for k,v in b.items() if k!='count'}:
                            raise ValueError('coverage branch metadata differs')
                        b['count']+=a['count']
    files=[];uncovered=[]
    for file in combined.values():
            name=Path(file['file']).name
            if name not in ('module_mp_thompson.F','mp_radar.F90'):continue
            lines=file['lines'];functions=[];source=by_name[name].read_text().splitlines()
            for function in file['functions']:
                relevant=[row for row in lines if row.get('function_name')==function['name']]
                branches=[branch for row in relevant for branch in row['branches']]
                untaken=sorted({row['line_number'] for row in relevant if any(b['count']==0 for b in row['branches'])})
                functions.append(dict(name=function['name'],start_line=function['start_line'],
                    end_line=function['end_line'],execution_count=function['execution_count'],
                    branch_edges=len(branches),edges_taken=sum(b['count']>0 for b in branches),untaken_lines=untaken))
                if untaken:
                    uncovered.append(dict(source=name,routine=function['name'],uncovered=[
                        dict(line=n,statement=source[n-1].strip()) for n in untaken]))
            files.append(dict(source=name,functions=functions))
    (receipts/'coverage-branches.json').write_text(json.dumps(dict(
        scope='Pristine standalone initialization and synthetic/helper calls, plus supplied supplemental real-input replays. Withdrawn-input diagnostics are not accepted evidence; no full-matrix or forecast coverage claim.',coverage_artifacts=coverage_artifacts,files=files),indent=2)+'\n')
    (receipts/'coverage-unreached.json').write_text(json.dumps(uncovered,indent=2)+'\n')
    ledger=[]
    for row in uncovered:
        for item in row['uncovered']:
            category,reason=branch_reason(row['routine'],item['line'])
            ledger.append(dict(source=row['source'],routine=row['routine'],**item,category=category,reason=reason))
    (receipts/'coverage-edge-reasons.json').write_text(json.dumps(dict(
        scope='Pristine standalone initialization, synthetic/helper calls and supplemental real-input replays. Reasons apply to the mp_thompson boundary and certified corpus; no forecast coverage claim.',
        every_untaken_line_named=True,complete_branch_coverage=False,rows=ledger),indent=2)+'\n')
    print(f'{len(rows)} pinned v8.4.2 citation windows')
    for file in files:
        for function in file['functions']:
            if function['branch_edges']:
                print(function['name'],f"{function['edges_taken']}/{function['branch_edges']}")


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('production',type=Path)
    p.add_argument('oracle',type=Path);p.add_argument('receipts',type=Path)
    p.add_argument('--supplement',action='append',type=Path,default=[])
    a=p.parse_args();audit(a.reference,a.production,a.oracle,a.receipts,a.supplement)
