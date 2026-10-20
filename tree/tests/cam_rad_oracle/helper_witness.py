"""Reach legacy CAM helpers using pristine native code and the whole port.

These supplemental calls include contexts the real driver never selects.
They are branch/debug evidence, not forecast or device acceptance.
"""
from pathlib import Path
import argparse
import json
import translate as tr
import witness as w
from driver_witness import TYPES


COUNTS = {'sortarray': 24, 'findvalue': 24, 'lin_interpol': 16,
          'lin_interpol2': 16, 'exp_interpol': 16, 'getfactors': 34,
          'validfactors': 104, 'cldclw': 4, 'gestbl': 4}


def prepare(directory, receipt):
    core, _, module, routines = tr.generate([
        directory / 'module_ra_cam_support.F', directory / 'module_ra_cam.F'])
    (directory / 'cam_core.cuh').write_text(core)
    schemas = {}
    f = ['module helper_capture', 'use module_ra_cam_support', 'implicit none',
         'integer :: before,after', 'contains']
    for name in COUNTS:
        routine = routines[name]
        fields = [routine.variables[a] for a in routine.args]
        if routine.result:
            fields.append(tr.Variable('helper_result', routine.variables[routine.result].typ, []))
        if name == 'gestbl':
            fields.extend(module[a] for a in ('estbl', 'tmin', 'tmax', 'epsqs', 'tmelt'))
        schemas[name] = fields
        args = routine.args + (['helper_result'] if routine.result else [])
        f += ['subroutine capture_' + name + '(' + ','.join(args) + ')']
        for v in fields:
            if v.name not in args:
                continue
            f += [TYPES[v.typ] + ' :: ' + v.name +
                  ('(' + ','.join(':' for _ in v.dims) + ')' if v.dims else '')]
        for v in fields:
            f += w.write_fortran(v, 'before')
        call = name + '(' + ','.join(routine.args) + ')'
        f += [('helper_result=' if routine.result else 'call ') + call]
        for v in fields:
            f += w.write_fortran(v, 'after')
        f += ['end subroutine']
    f += ['end module', 'program helpers', 'use helper_capture', 'implicit none',
          'integer :: i,j,k,n,ix,answer,indices(24)',
          'real(kind=8) :: a(24),x(5),fx(5),y,fact1,fact2,hl(6),zi(6,5),clwp(6,4),tpw(6)',
          'real(kind=8) :: choices(10)', 'real :: xf(5),ff(5),yf,rf',
          'logical :: ok,cyclic,icephase',
          "open(newunit=before,file='helpers-before.bin',access='stream',form='unformatted',status='replace')",
          "open(newunit=after,file='helpers-native-after.bin',access='stream',form='unformatted',status='replace')",
          'n=24', 'do i=1,24', 'do k=1,n',
          'a(k)=real(mod(k*13+i*7,31),8)/7._8', 'indices(k)=k', 'enddo',
          'call capture_sortarray(n,a,indices)', 'enddo',
          'do i=1,24', 'ix=i', 'answer=0', 'do k=1,n',
          'a(k)=real(mod(k*13+i*7,31),8)/7._8', 'indices(k)=k', 'enddo',
          'call capture_findvalue(ix,n,a,indices,answer)', 'enddo',
          'x=[-1._8,0._8,1._8,2._8,5._8]', 'fx=[1._8,2._8,5._8,10._8,20._8]']
    for name in ('lin_interpol', 'lin_interpol2', 'exp_interpol'):
        f += ['do i=1,16', 'y=-3._8+real(i-1,8)*0.8_8']
        if name == 'lin_interpol2':
            f += ['xf=real(x)', 'ff=real(fx)', 'yf=real(y)', 'rf=0.',
                  'call capture_lin_interpol2(xf,ff,yf,rf)']
        else:
            f += ['fact1=0._8', 'call capture_' + name + '(x,fx,y,fact1)']
        f += ['enddo']
    f += ['do i=1,24', 'cyclic=i<=12', 'ix=mod(i,12)+1',
          'fact1=0._8', 'fact2=0._8',
          'call capture_getfactors(cyclic,ix,350._8,15._8,real(i-1,8)*20._8-20._8,fact1,fact2)',
          'enddo',
          'do i=1,8', 'cyclic=i<=4', 'y=real(i-1,8)',
          'if(i==1.or.i==5) y=-1._8', 'if(i==2.or.i==6) y=0._8',
          'if(i==4.or.i==8) y=367._8', 'fact1=0._8;fact2=0._8',
          'call capture_getfactors(cyclic,2,350._8,15._8,y,fact1,fact2)', 'enddo',
          'cyclic=.false.;fact1=0._8;fact2=0._8',
          'call capture_getfactors(cyclic,2,0._8,1._8,1.e20_8,fact1,fact2)',
          'call capture_getfactors(cyclic,2,0._8,1._8,1.000000975_8,fact1,fact2)',
          'choices=[-1._8,-0.0000015_8,-0.0000005_8,0._8,0.3_8,0.7_8,1._8,1.0000005_8,1.0000015_8,2._8]',
          'do i=1,10', 'do j=1,10', 'ok=.false.',
          'call capture_validfactors(choices(i),choices(j),ok)', 'enddo', 'enddo',
          'ok=.false.;call capture_validfactors(-0.00000075_8,1.00000125_8,ok)',
          'ok=.false.;call capture_validfactors(1.0000005_8,-0.00000125_8,ok)',
          'ok=.false.;call capture_validfactors(1.00000125_8,-0.00000075_8,ok)',
          'ok=.false.;call capture_validfactors(-0.00000125_8,1.0000005_8,ok)',
          'do i=1,4', 'do k=1,5', 'zi(:,k)=real(5-k,8)*1000._8', 'enddo',
          'tpw=real(i,8)*2._8', 'clwp=0._8', 'hl=0._8',
          'call capture_cldclw(1,6,6,4,5,zi,clwp,tpw,hl)', 'enddo',
          'estbl=0._8', 'tmin=0._8', 'tmax=0._8', 'epsqs=0._8', 'tmelt=0._8',
          'do i=1,4', 'icephase=i/=1', 'y=20._8',
          'if(i==2) y=0._8', 'if(i==4) y=60._8',
          'call capture_gestbl(173.16_8,375.16_8,y,icephase,0.622_8,2.5e6_8,3.34e5_8,461.5_8,1004.5_8,273.16_8)',
          'enddo',
          'close(before)', 'close(after)', 'end program']
    (directory / 'helper_oracle.F90').write_text('\n'.join(f) + '\n')
    cpp = ['#include <type_traits>', '#include "array_abi.cuh"', '#include "cam_core.cuh"', w.IO,
           'int main(int argc,char**argv) {if(argc!=4) return 2;',
           'std::vector<char> memory(64*1024*1024);CamArena arena(memory.data(),memory.size());',
           'CamColumn cam(arena);Input state(argv[1]);']
    globals_ = [v for v in module.values() if v.typ != 'char' and (not v.const or v.dims)]
    for v in globals_:
        cpp += ['cam.' + v.name + '=state.' + w.reader(v) + ';']
    cpp += ['Input inputs(argv[2]);FILE*out=fopen(argv[3],"wb");if(!out) return 2;']
    for name, count in COUNTS.items():
        r = routines[name]
        cpp += ['for(int record=0;record<' + str(count) + ';++record) {', 'inputs.storage.clear();']
        for v in schemas[name]:
            target = ('auto ' if v.name in r.args or v.name == 'helper_result' else 'cam.') + v.name
            cpp += [target + '=inputs.' + w.reader(v) + ';']
        cpp += [('helper_result=' if r.result else '') + 'cam.' + name + '(' + ','.join(r.args) + ');']
        for v in schemas[name]:
            writer = 'output_array' if v.dims else 'output_logical' if v.typ == 'bool' else 'output_scalar'
            target = v.name if v.name in r.args or v.name == 'helper_result' else 'cam.' + v.name
            cpp += [writer + '(out,' + target + ');']
        cpp += ['if(cam.error) return 4;', '}']
    cpp += ['if(!inputs.eof()) return 3;fclose(out);return 0;}']
    (directory / 'helper_witness.cpp').write_text('\n'.join(cpp) + '\n')
    (directory / 'stop_oracle.F90').write_text('''! MPAS module_ra_cam_support.F:2634-2667
program native_stop
 use module_ra_cam_support, only: get_rf_scales
 implicit none
 real(kind=8) :: scales(12)
 scales=7._8
 call get_rf_scales(scales)
 error stop 'native STOP unexpectedly returned'
end program
''')
    stop_cpp = cpp[:cpp.index('Input inputs(argv[2]);FILE*out=fopen(argv[3],"wb");if(!out) return 2;')]
    stop_cpp += ['auto scales=arena.array<double>(1,12);for(int i=1;i<=12;++i) scales(i)=7.;',
                 'cam.get_rf_scales(scales);if(cam.error!=2651) return 5;',
                 'for(int i=1;i<=12;++i) if(scales(i)!=7.) return 6;',
                 'printf("CAM_NATIVE_STOP_LINE=%d\\n",cam.error);return 0;}']
    (directory / 'stop_witness.cpp').write_text('\n'.join(stop_cpp) + '\n')
    (directory / 'background_stop_oracle.F90').write_text('''! MPAS module_ra_cam_support.F:1792-1871
program native_background_stop
 use module_ra_cam_support, only: background,mxaerl
 implicit none
 real(kind=8) :: pressure(2,3),mmr(2,2)
 pressure=1.e5_8;mmr=7._8;mxaerl=0
 call background(1,2,pressure,2,2,3,mmr)
 error stop 'native fatal unexpectedly returned'
end program
''')
    background_cpp=cpp[:cpp.index('Input inputs(argv[2]);FILE*out=fopen(argv[3],"wb");if(!out) return 2;')]
    background_cpp += ['cam.mxaerl=0;auto pressure=arena.array<double>(1,2,1,3);',
        'auto mmr=arena.array<double>(1,2,1,2);',
        'for(int k=1;k<=3;++k) for(int i=1;i<=2;++i) pressure(i,k)=1.e5;',
        'for(int k=1;k<=2;++k) for(int i=1;i<=2;++i) mmr(i,k)=7.;',
        # The transcribed logical statement starts at the conditional on1846;
        # native backtraces name its continued call on1847.
        'cam.background(1,2,pressure,2,2,3,mmr);if(cam.error!=1846) return 5;',
        'for(int k=1;k<=2;++k) for(int i=1;i<=2;++i) if(mmr(i,k)!=7.) return 6;',
        'printf("CAM_NATIVE_FATAL_LINE=%d\\n",cam.error);return 0;}']
    (directory/'background_stop_witness.cpp').write_text('\n'.join(background_cpp)+'\n')
    objects = [p.name for p in directory.glob('*.o') if p.name not in
               ('run_columns.o', 'run_columns_capture.o', 'cam_capture.o')]
    make = (directory / 'Makefile').read_text().split('\nhelper_oracle:', 1)[0]
    make += '\nhelper_oracle: helper_oracle.F90 ' + ' '.join(objects) + '\n\t$(FC) $(FFLAGS) $^ -o $@\n'
    make += 'helper_witness: helper_witness.cpp cam_core.cuh array_abi.cuh\n\t$(CXX) -std=c++17 -O3 -g -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $< -o $@\n'
    make += 'stop_oracle: stop_oracle.F90 ' + ' '.join(objects) + '\n\t$(FC) $(FFLAGS) $^ -o $@\n'
    make += 'stop_witness: stop_witness.cpp cam_core.cuh array_abi.cuh\n\t$(CXX) -std=c++17 -O3 -g -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $< -o $@\n'
    make += 'background_stop_oracle: background_stop_oracle.F90 ' + ' '.join(objects) + '\n\t$(FC) $(FFLAGS) $^ -o $@\n'
    make += 'background_stop_witness: background_stop_witness.cpp cam_core.cuh array_abi.cuh\n\t$(CXX) -std=c++17 -O3 -g -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $< -o $@\n'
    (directory / 'Makefile').write_text(make)
    receipt.write_text(json.dumps({'scope': 'supplemental whole-port helper calls; no forecast proof',
                                  'schema': {n: [v.__dict__ for v in f] for n, f in schemas.items()},
                                  'records': COUNTS}, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    prepare(a.directory, a.receipt)
