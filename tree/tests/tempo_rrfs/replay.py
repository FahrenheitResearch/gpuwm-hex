"""Full-module synthetic replay; standalone evidence is not forecast acceptance."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from hexcore.mpasphys.rrfs.tempo_v303 import TABLE_LAYOUT, STATE_FIELDS
ROOT=Path(__file__).resolve().parents[4]
def pointer(a):return C.c_void_p(a.ctypes.data)
def h(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('flags');parser.add_argument('--no-blob',action='store_true');parser.add_argument('--all-records',action='store_true');parser.add_argument('--presence',type=int);parser.add_argument('--math-audit',action='store_true');parser.add_argument('--levels',type=int,default=24);parser.add_argument('--oracle-prefix',default='oracle-');parser.add_argument('--initialize-aerosols-and-hail',action='store_true')
    parser.add_argument('--native-library',type=Path,default=ROOT/'scratch/tempo-build/libtempo_oracle.so')
    parser.add_argument('--stress-corpus',action='store_true',help='include 64 deterministic threshold and mixed-phase columns')
    args=parser.parse_args();args.native_library=args.native_library.resolve();flags=np.asarray(json.loads(args.flags),dtype=np.int32)
    if flags.shape!=(4,) or not np.isin(flags,[0,1]).all():raise ValueError('four native Boolean flags required')
    native=C.CDLL(str(args.native_library))
    rrfs_build=None
    if args.native_library.name=='libtempo_rrfs_oracle.so':
        dependency=ROOT/'scratch/r2-native/libmpasphys_native_rrfs.so'
        rrfs_build=json.loads(dependency.with_name('build-receipt.json').read_text())
        if rrfs_build['source_pin']!='1d6e370e7b43a332074809f170532a216c2212fb' or rrfs_build['library_sha256']!=h(dependency):
            raise ValueError('published RRFS native library source/hash changed')
        abi=dependency.with_name('abi.json')
        if h(abi)!=rrfs_build['abi_sha256']:raise ValueError('published RRFS ABI hash changed')
    native.oracle_init.argtypes=[C.c_void_p]
    native.oracle_step.argtypes=[C.c_int,C.c_float,C.c_void_p,C.c_void_p,C.c_void_p]
    native.oracle_blob.argtypes=[C.c_void_p]
    host=C.CDLL(str(ROOT/('scratch/tempo-build/libtempo_audit.so' if args.math_audit else 'scratch/tempo-build/libtempo_host.so')))
    host.cpp_step.argtypes=[C.c_int,C.c_float]+[C.c_void_p]*5+[C.c_int,C.c_int]
    transport=C.CDLL(str(ROOT/'n1out/libtempo_transport.so'))
    transport.tempo_write.argtypes=[C.c_char_p,C.c_void_p,C.c_size_t]
    tag=''.join(map(str,flags));out=ROOT/'n1out'/(args.oracle_prefix+tag);out.mkdir(exist_ok=True)
    os.chdir(ROOT/'scratch/tables')
    native.oracle_init(pointer(flags))
    presence=args.presence if args.presence is not None else 7+(24 if flags[1] else 0)
    native.oracle_presence.argtypes=[C.c_int];native.oracle_presence(presence)
    blob=np.empty(sum(math.prod(d) for _,d in TABLE_LAYOUT),np.float64)
    native.oracle_blob(pointer(blob))
    if not args.no_blob:assert transport.tempo_write(os.fsencode(out/'blob.bin'),pointer(blob),blob.nbytes)==0
    rows=[];first=True
    case_count=76 if args.stress_corpus else 12
    for case in range(case_count):
        nz=args.levels;z=np.arange(nz,dtype=np.float32)
        a=np.zeros((17,nz),np.float32);a[0]=np.linspace(295-case*4,220,nz,dtype=np.float32)
        a[1]=np.linspace(99000,20000,nz,dtype=np.float32);a[2]=np.float32((case%4)-1)
        a[3]=np.float32(180+case*30);a[4]=np.float32(0.005+0.001*(case%3))*np.exp(-z/6).astype(np.float32)
        if case>0:
            a[5:10]=np.float32(0.0002*(case%4+1))
            if case in (2,3,4,5,6):a[case+3]=0
            if case==10:a[5:10]=np.float32(0.01)
        a[10]=1.e5;a[11]=1.e6;a[12]=3.e8;a[13]=3.e8;a[14]=1.e6;a[15]=1.e4;a[16]=a[9]*2
        if args.initialize_aerosols_and_hail and case==0:
            a[5:10]=np.float32(0.001);a[13:17]=0
        bl=np.zeros((2,nz),np.float32);bl[0]=1.e-5;bl[1]=0.4
        if case>=12:
            # Whole-scheme synthetic columns spanning phase, activation,
            # sedimentation, collection and number-concentration thresholds.
            # They are test inputs, never forecast defaults or case selection.
            j=case-12
            phase=j%8;density=(j//8)%4;mass=j//32
            warm=(195.,220.,248.,263.,272.9,273.15,283.,307.)[phase]
            a[0]=np.linspace(warm,warm-8,nz,dtype=np.float32)
            a[1]=np.linspace(100000.,15000.,nz,dtype=np.float32)
            a[2]=(-12.,0.,2.,20.)[density]
            a[3]=(40.,180.,700.,1800.)[density]
            a[4]=np.float32((.00001,.001,.008,.03)[density])*np.exp(-z/nz).astype(np.float32)
            a[5:10]=np.float32((1.e-8,.008)[mass])
            a[5+(j%5)]*=np.float32(1.e-4)
            a[10:13]=np.float32((1.,1.e3,1.e6,1.e9)[density])
            a[13]=np.float32((0.,1.e4,3.e8,1.e10)[density])
            a[14]=np.float32((0.,1.,1.e5,1.e8)[density])
            a[15]=np.float32((1.,1.e2,1.e4,1.e7)[density])
            a[16]=a[9]*np.float32((1000./900.,1000./500.,1000./200.,1000./100.)[density])
            bl[0]=np.float32((0.,1.e-8,1.e-4,.01)[density])
            bl[1]=np.float32((0.,.1,.5,1.)[density])
        # compare from same inputs on every step, rather than letting drift conceal a defect
        for step in range(12):
            before=a.copy();b=a.copy();d=np.zeros((15,nz),np.float32);hd=d.copy()
            native.oracle_step(nz,C.c_float(10),pointer(a),pointer(bl),pointer(d))
            host.cpp_step(nz,C.c_float(10),pointer(b),pointer(bl),pointer(hd),pointer(blob),pointer(flags),int(first),presence)
            first=False
            row={'case':case,'step':step,'state':{},'diagnostics':{}}
            for kind,x,y,names in [('state',a,b,STATE_FIELDS),('diagnostics',d,hd,[str(i) for i in range(15)])]:
                for i,name in enumerate(names):
                    mask=x[i].view(np.uint32)!=y[i].view(np.uint32)
                    if mask.any():
                        k=int(np.flatnonzero(mask)[0]);row[kind][name]={'words':int(mask.sum()),'first_level':k,'native_word':int(x[i].view(np.uint32)[k]),'host_word':int(y[i].view(np.uint32)[k])}
            if step==0 or args.all_records:
                for name,value in [('input',before),('output',a),('diag',d),('pbl',bl)]:
                    assert transport.tempo_write(os.fsencode(out/(f'{case}-{name}.bin' if step==0 else f'{case}-{step}-{name}.bin')),pointer(value),value.nbytes)==0
            rows.append(row)
    if args.math_audit:
        host.cpp_math_tape.restype=C.POINTER(C.c_double);host.cpp_math_count.restype=C.c_ulonglong
        count=host.cpp_math_count()
        assert transport.tempo_write(os.fsencode(out/'math-tape.bin'),host.cpp_math_tape(),count*8)==0
        (out/'math-tape.json').write_text(json.dumps({'entries':count//4,'flags':flags.tolist(),'forecast_proven':False})+'\n')
    import socket
    receipt={'schema':'rrfs-tempo-standalone-debug-v1','forecast_proven':False,'host':socket.gethostname(),'flags':flags.tolist(),'presence':presence,
             'fork':'1d6e370e7b43a332074809f170532a216c2212fb','tempo':'9adf9ef2827104640125f0c322e1218fce74e94c',
             'native_library':str(args.native_library),'native_library_sha256':h(args.native_library),
             'cpp_library_sha256':h(ROOT/('scratch/tempo-build/libtempo_audit.so' if args.math_audit else 'scratch/tempo-build/libtempo_host.so')),
             'table_sha256':{p.name:h(p) for p in (ROOT/'scratch/tables').iterdir() if p.is_file()},
             'blob_sha256':h(out/'blob.bin') if not args.no_blob else None,'records_per_case':12 if args.all_records else 1,'module_sha256':h(Path(__file__).parents[2]/'src/hexcore/mpasphys/rrfs/tempo_v303.py'),'cases':rows,'all_outputs_bitwise_equal':all(not r['state'] and not r['diagnostics'] for r in rows)}
    receipt['initialized_blob_sha256']=hashlib.sha256(memoryview(blob).cast('B')).hexdigest();receipt['levels']=args.levels
    receipt['column_cases']=case_count;receipt['stress_corpus']=args.stress_corpus
    if rrfs_build:
        receipt['published_rrfs_native']={key:rrfs_build[key] for key in ('source_pin','pin_set_sha256','library_sha256','abi_sha256','flags','external_pins')}
        receipt['native_scheme_objects']='r2 full fork library; bridge contains only oracle.o'
    (out/'host-replay.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'flags':flags.tolist(),'passing_steps':sum(not r['state'] and not r['diagnostics'] for r in rows),'steps':len(rows),'first_mismatch':next((r for r in rows if r['state'] or r['diagnostics']),None)}))
if __name__=='__main__':main()
