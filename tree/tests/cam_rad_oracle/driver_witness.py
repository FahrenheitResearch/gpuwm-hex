"""Strict MPAS numeric driver-slice oracle; pool retrieval is the only adapter.

Numeric statements are copied verbatim from the selected CAM branches. Every
array is prefilled, including padding, so all conversion outputs are defined.
"""
from pathlib import Path
import argparse
import json
import hashlib
import driver_slice as ds
import witness as w
import translate as tr

TYPES={'float':'real','double':'real(kind=8)','int':'integer','bool':'logical'}


def prepare(reference,directory,receipt):
    code,schemas,inventory=ds.generate(reference)
    (directory/'cam_driver.cuh').write_text(code)
    fn=['module cam_driver_oracle','implicit none','contains']
    for name,fields in schemas.items():
        if name=='driver_clock':
            r=tr.Routine(name,[v.name for v in fields],None,'mpas_atmphys_driver_radiation_lw.F',888)
            r.end=900
            r.body=[(889,'xtime_m = xtime_s/60.'),(900,'radt = dt_radt/60.')]
            locals_=set();symbols={v.name:v for v in fields}
        else:
            r,locals_,symbols=ds.LAST_ROUTINES[name]
        fn+=['! MPAS '+r.file+':'+str(r.start)+'-'+str(r.end),
             'subroutine '+name+'('+','.join(v.name for v in fields)+')']
        for v in fields:
            fn+=[TYPES[v.typ]+' :: '+v.name+('('+','.join(':' for _ in v.dims)+')' if v.dims else '')]
        for s in sorted(locals_):fn+=[TYPES[symbols[s].typ]+' :: '+s]
        for n,line in r.body:fn+=['! MPAS '+r.file+':'+str(n),line.replace('_rkind','')]
        fn+=['end subroutine']
    fn+=['end module']
    fn+=['program driver_columns','use cam_driver_oracle','implicit none',
         'integer :: before,after',
         "open(newunit=before,file='driver-before.bin',access='stream',form='unformatted',status='replace')",
         "open(newunit=after,file='driver-native-after.bin',access='stream',form='unformatted',status='replace')"]
    for name in schemas:fn+=['call run_'+name+'()']
    fn+=['close(before)','close(after)','contains']
    integer_values={'its':2,'ite':4,'jts':1,'jte':2,'kts':1,'kte':4,
                    'cam_abs_dim1':5,'cam_abs_dim2':4,'num_aerosols':12,
                    'num_months':4,'num_oznlevels':5,'num_aerlevels':6}
    for name,fields in schemas.items():
        fn+=['subroutine run_'+name+'()','integer :: variant,i,k']
        for v in fields:
            fn+=[TYPES[v.typ]+' :: '+v.name+('('+','.join('16' for _ in v.dims)+')' if v.dims else '')]
        count=512 if name in ('radconst','driver_clock') else 3
        fn+=['do variant=1,'+str(count)]
        for index,v in enumerate(fields):
            if v.dims:
                fn+=[v.name+' = '+str(index+1)+'.125 + variant*.01']
            elif v.typ=='int':fn+=[v.name+' = '+str(integer_values.get(v.name,1))]
            elif v.typ=='float':
                value={'degrad':'3.1415926/180.','dpd':'360./365.',
                       'julian':'(variant-1)*.875-1.25','solcon_0':'1370.',
                       'xtime_s':'real(variant-1)*113.5',
                       'dt_radt':'real(variant)*0.125'}.get(v.name,str(index+1)+'.125')
                fn+=[v.name+' = '+value]
            else:raise ValueError(v)
        for v in fields:fn+=w.write_fortran(v,'before')
        fn+=['call '+name+'('+','.join(v.name for v in fields)+')']
        for v in fields:fn+=w.write_fortran(v,'after')
        fn+=['enddo','end subroutine']
    fn+=['end program']
    (directory/'driver_oracle.F90').write_text('\n'.join(fn)+'\n')
    cpp=['#include <type_traits>','#include "array_abi.cuh"','#include "cam_driver.cuh"',w.IO,
         'int main(int argc,char**argv) {if(argc!=3) return 2;Input inputs(argv[1]);FILE*out=fopen(argv[2],"wb");if(!out) return 2;']
    for name,fields in schemas.items():
        cpp+=['for(int record=0;record<'+str(512 if name in ('radconst','driver_clock') else 3)+';++record) {','inputs.storage.clear();']
        for v in fields:cpp+=['auto '+v.name+'=inputs.'+w.reader(v)+';']
        cpp+=[name+'('+','.join(v.name for v in fields)+');']
        for v in fields:
            writer='output_array' if v.dims else 'output_logical' if v.typ=='bool' else 'output_scalar'
            cpp+=[writer+'(out,'+v.name+');']
        cpp+=['}']
    cpp+=['if(!inputs.eof()) return 3;fclose(out);return 0;}']
    (directory/'driver_witness.cpp').write_text('\n'.join(cpp)+'\n')
    make=(directory/'Makefile').read_text().split('\ndriver_oracle:',1)[0]
    make+='\ndriver_oracle: driver_oracle.F90\n\t$(FC) $(FFLAGS) $< -o $@\n'
    make+='driver_witness: driver_witness.cpp cam_driver.cuh array_abi.cuh\n\t$(CXX) -std=c++17 -O3 -g -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $< -o $@\n'
    (directory/'Makefile').write_text(make)
    receipt.write_text(json.dumps({'routines':inventory,'schema':{n:[v.__dict__ for v in f] for n,f in schemas.items()},
                                 'records':{n:512 if n in ('radconst','driver_clock') else 3 for n in schemas},
                                 'source_sha256':{p:hashlib.sha256((reference/'src/core_atmosphere/physics'/p).read_bytes()).hexdigest() for p in {r['source'] for r in inventory}},
                                 'adapter':'pool retrieval replaced by explicit arguments; selected CAM branches copied unchanged'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();prepare(a.reference,a.directory,a.receipt)
