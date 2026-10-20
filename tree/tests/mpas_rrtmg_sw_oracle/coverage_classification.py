"""Classify native SW untaken branches; unresolved outcomes remain explicit.

Counts are native gcov, never inferred as covered. Table bounds below are
non-weather coefficient analysis of pristine native table-reader output.
"""
import argparse
import hashlib
import json
from pathlib import Path
from mpas_rrtmg_sw_oracle.replay import read_swd


def classify(coverage, tables, output):
    coverage,tables,output = map(Path,(coverage,tables,output))
    native = json.loads(coverage.read_text())
    coefficient = read_swd(tables)
    bounds = {name:dict(minimum=float(value.min()),maximum=float(value.max()))
              for name,value in coefficient.items() if name.startswith('cld/')}
    fixed = {
        1742: ('fatal pressure order', 'KISS requires bottom-to-top pmid; native STOP SW:1742-1744'),
        1753: ('fixed control', 'SW:1719 normalizes every nonzero irng to1, making third RNG fallthrough impossible'),
        1775: ('fixed control', 'SW:1719 normalizes irng to0/1'),
        1800: ('fixed control', 'SW:1719 normalizes irng to0/1'),
        1834: ('fixed control', 'SW:1719 normalizes irng to0/1'),
        1874: ('fixed control', 'SW:1719 normalizes irng to0/1'),
        1927: ('fixed control', 'SW:1719 normalizes irng to0/1'),
        2241: ('fixed coefficient', 'wvn/wavenum2 native minimum2600 exceeds2500 for all14 bands'),
        4232: ('fixed coefficient', 'kg24/layreffr=1; setcoef clamps jp>=1, so jp<layreffr is false'),
        4782: ('fixed coefficient', 'wvn/ngc ranges2..12; mg=16, so ngc<mg is always true'),
        4797: ('fixed coefficient', 'else clause at4782 is unreachable under pinned ngc'),
        8472: ('fixed control', 'rrtmg_sw sets iout=0 at9296; optional per-band-output clause unreachable from module entry'),
        8627: ('fixed control', 'rrtmg_sw sets icpr=1 at9373 before spcvmc calls'),
        8639: ('fixed control', 'icpr=1 at9373; the unscaled cloud clause icpr=0 is inactive'),
        8640: ('fixed control', 'icpr=1 at9373; the unscaled cloud clause is inactive'),
        9333: ('MPAS driver range / unchecked lower selector', 'MPAS driver:929-940 passes only aer_opt0/3; whole-module lower aer_opt outside0..3 leaves iaer unset atSW:9331-9335. Unsupported-selector diagnostics retain every compared word and claim continuation only, never stable garbage or an exclusion'),
        9434: ('MPAS driver range / unchecked lower selector', 'MPAS driver:929-940 passes only aer_opt0/3; whole-module lower aer_opt outside0..3 leaves iaer unset atSW:9331-9335. Unsupported-selector diagnostics retain every compared word and claim continuation only, never stable garbage or an exclusion'),
        9436: ('MPAS driver range / unchecked lower selector', 'MPAS driver:929-940 passes only aer_opt0/3; whole-module lower aer_opt outside0..3 leaves iaer unset atSW:9331-9335. Unsupported-selector diagnostics retain every compared word and claim continuation only, never stable garbage or an exclusion'),
        9437: ('MPAS driver range / unchecked lower selector', 'MPAS driver:929-940 passes only aer_opt0/3; whole-module lower aer_opt outside0..3 leaves iaer unset atSW:9331-9335. Unsupported-selector diagnostics retain every compared word and claim continuation only, never stable garbage or an exclusion'),
        9438: ('MPAS driver range / unchecked lower selector', 'MPAS driver:929-940 passes only aer_opt0/3; whole-module lower aer_opt outside0..3 leaves iaer unset atSW:9331-9335. Unsupported-selector diagnostics retain every compared word and claim continuation only, never stable garbage or an exclusion'),
        9473: ('MPAS driver range / unchecked lower selector', 'MPAS driver:929-940 passes only aer_opt0/3; whole-module lower aer_opt outside0..3 leaves iaer unset atSW:9331-9335. Unsupported-selector diagnostics retain every compared word and claim continuation only, never stable garbage or an exclusion'),
        9950: ('MPAS driver range / unchecked lower selector', 'MPAS driver:929-940 passes only aer_opt0/3; whole-module lower aer_opt outside0..3 leaves iaer unset atSW:9331-9335. Unsupported-selector diagnostics retain every compared word and claim continuation only, never stable garbage or an exclusion'),
        2282: ('native undefined coefficient path', 'False requires nonzero ice/snow with iceflag<1 after flags1/2 excluded; SW:2217-2321 leaves ice coefficients unset; port-only refusal retired under NATIVE-DEFECT-POLICY, and no stable-value identity or exclusion is claimed'),
        2398: ('native undefined coefficient path', 'False requires nonzero liquid with liqflag!=1; SW:2392-2421 leaves liquid coefficients unset; zero-liquid flags are native-tested separately'),
        8769: ('fixed band partition', 'spcvmc loop ibm=1..14; previous if handles10..13, leaving only1..9/14'),
        9419: ('native option range', 'rrtmg_sw/inatm supports icld0..5; negative icld would leave local cloud storage unset'),
        10235: ('MPAS driver optional argument', 'mpas_atmphys_driver_radiation_sw.F:919-948 omits o33d; prescribed/climatological ozone options are tested'),
        10241: ('MPAS driver optional argument', 'MPAS SW driver:924 passes qcrad_p as qc3d'),
        10242: ('MPAS driver optional argument', 'MPAS SW driver:919-948 omits qr3d'),
        10243: ('MPAS driver optional argument', 'MPAS SW driver:924 passes qirad_p as qi3d'),
        10244: ('MPAS driver optional argument', 'MPAS SW driver:925 passes qsrad_p as qs3d'),
        10245: ('MPAS driver optional argument', 'MPAS SW driver:924-925 always supplies qc3d,qi3d,qs3d,cldfra3d'),
        10396: ('fixed wrapper control', 'SW:10185-10186 initializes inflgsw2 and iceflgsw3; effective-radius flags select inflgsw3..5'),
        10425: ('MPAS land category', 'Branch5 false requires xland=1.5 after ocean excluded; MPAS SW driver supplies land1/ocean2'),
        10502: ('fixed wrapper control', 'SW:10185-10186 sets iceflgsw3; has_reqi/has_reqs select4/5; third fallthrough unreachable'),
        1940: ('native overlap selector default', 'gcno block66 has six switch arcs: five native overlap cases1..5 and default block79 jumping directly to block247; gcov branch1 is untaken default, not the level-loop exit (branches6/7 both taken); mcica_subcol_sw:1502 returns icld0 and the MPAS driver selects1..5'),
        10468: ('MPAS effective-radius flags', 'MPAS manager:747-759 sets all three flags1 only when config_microp_re is enabled, microp_scheme is Thompson/Thompson_aerosols/WSM6, and both LW/SW use RRTMG; otherwise all flags0; no-reqi plus reqs1 is not this driver selection'),
        10562: ('MPAS driver optional argument', 'MPAS SW driver:937-938 passes all three aerosol arrays together; missing arguments are not this driver interface'),
        10609: ('MPAS driver optional argument', 'MPAS SW driver:938-940 always supplies scalar TOA/surface outputs'),
        10622: ('MPAS driver optional argument', 'MPAS SW driver:941 supplies all three direct/diffuse outputs together'),
        10628: ('MPAS driver optional argument', 'MPAS SW driver:919-948 omits profile outputs; standalone column oracle supplies them as additional diagnostics'),
        10645: ('MPAS driver optional argument', 'MPAS SW driver:938-940 always supplies scalar outputs, including night calls'),
    }
    rows = []
    for file in native['sources']:
        for branch in file['untaken']:
            result = dict(file=file['file'],**branch,classification='under review',evidence='No covered/unreachable claim yet')
            if file['file']=='module_ra_rrtmg_sw.F':
                line,text=branch['line'],branch['source']
                if line in fixed:
                    result['classification'],result['evidence']=fixed[line]
                elif line in (1889,1941,1951):
                    result.update(classification='compiler SPREAD allocation control',evidence='gcov branch1 directly precedes generated SPREAD call; both numerical WHERE outcomes are taken in this gcov entry')
                elif text.lower().startswith(('subroutine ','end subroutine ')) or text.split('!')[0].rstrip().endswith((',', '&', ')')) and not text.lower().startswith(('if','elseif','else if','where','do')):
                    result.update(classification='compiler array/descriptor control',evidence='gcov branch attached to assumed-shape interface or continued argument declarations; numerical branch coverage is not inferred')
                elif 'stop ' in text.lower() or 'fatal_error' in text.lower():
                    result.update(classification='native fatal optical bound',evidence='Native STOP path; pinned coefficient bounds are listed, returning columns do not count this as covered')
                elif line in (2302,2306,2351,2355):
                    result.update(classification='fixed coefficient bound',evidence='cld/fdlice3 lies strictly inside0..1; valid ice/snow radius5..140 interpolates between adjacent knots')
                elif line in (2363,2367,2371,2375,2379):
                    result.update(classification='fixed snow coefficient bound',evidence='Native snow radius is checked5..140 before linear interpolation; extice3>0,ssaice3 within0..1,asyice3 within0..1; table bounds are listed in this receipt')
            elif file['file']=='module_mp_thompson_aerosols.F':
                if branch['line']==143:
                    result.update(classification='previous condition excludes outcome',evidence='gt_aod:140 already handles RH<60; only RH>=60 reaches this elseif for finite clamped RH')
                elif branch['line']==154:
                    result.update(classification='fixed RH/index bound',evidence='gt_aod:143 restricts60<=RH<80; nint(.1*RH-4) is2..4, so idx+1<=5<rind8')
                elif branch['line']==170:
                    result.update(classification='fixed RH/index bound',evidence='gt_aod:136 clamps RH<=98; idx8 has rh_d<0 against rh_arr(8)=99.8 and bypasses this else; remaining idx<=7 gives idx+1<=8')
            elif file['file']=='module_ra_rrtmg_sw_aerosols.F':
                line,text=branch['line'],branch['source']
                if line in (558,587,703,818):
                    result.update(classification='fixed RH/index bound',evidence='calc_relative_humidity:916 clamps RH0..99; spectral RH table ends99.8, so the while search stops before ii>N_RH')
                elif 'FATAL_ERROR' in text:
                    result.update(classification='native fatal argument evaluation',evidence='Pristine physics_error_fatal does not return;18 native fatal witnesses exercise all source option/missing-array/value rejection paths; compiler argument branches are not inferred covered')
                elif 'WRITE_MESSAGE' in text or 'minval(' in text or 'maxval(' in text:
                    result.update(classification='compiler diagnostic argument control',evidence='gcov branch is attached to formatting/reduction of diagnostic arguments; all returning option combinations are replayed; no numerical coverage inferred')
                elif text.lower().startswith('end subroutine'):
                    result.update(classification='compiler array cleanup control',evidence='Branch attached to automatic-array cleanup at routine return, not a source conditional')
            elif file['file']=='mpas_atmphys_rrtmg_swinit.F':
                line=branch['line']
                if 'my_proc_id == IO_NODE' in branch['source']:
                    result.update(classification='native distributed I/O control',evidence='Standalone oracle is one IO rank; non-IO branch only broadcasts coefficient words, production Rust decodes once before upload')
                elif line==78:
                    result.update(classification='framework Fortran unit exhaustion',evidence='The standalone framework stub returns unit31; exhausting MPAS unit allocation is a framework fatal path, not a numerical coefficient branch')
                elif line in (95,96):
                    result.update(classification='compiler log argument control',evidence='Native open-failure witness reaches this error report; compiler array-argument descriptors are not inferred covered')
            elif file['file']=='module_ra_rrtmg_lw.F' and branch['line']==1922:
                result.update(classification='compiler duplicated RNG body',evidence='Native gcov same line branches18/19 are both taken in getRandomReal; branches0..17 belong to unused compiler-expanded copies at this source line; MT SW columns compare every generated output word')
            rows.append(result)
    result=dict(authority='Native gcov classification; no additional coverage inferred',
                coverage_sha256=hashlib.sha256(coverage.read_bytes()).hexdigest(),
                table_fixture_sha256=hashlib.sha256(tables.read_bytes()).hexdigest(),
                coefficient_bounds=bounds,untaken=rows,whole_model_proof=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('coverage');parser.add_argument('tables');parser.add_argument('output')
    classify(**vars(parser.parse_args()))
