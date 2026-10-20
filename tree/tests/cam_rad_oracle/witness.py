"""Generate full-module host witness and pristine boundary capture adapters.

Binary parsing here lives in compiled test code. MPAS numeric source is never
modified. The capture adapter serializes module state and every camrad argument
before/after the actual native call; host translation receives the same inputs.
"""
from pathlib import Path
import argparse
import shutil
import json
import hashlib
import translate as tr

HERE=Path(__file__).resolve().parent

IO=r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <cstdint>
struct Input {
 FILE* file;
 std::vector<std::unique_ptr<char[]>> storage;
 Input(const char* name):file(fopen(name,"rb")) {if(!file) std::abort();}
 ~Input() {fclose(file);}
 template<class T> T scalar() {
  unsigned char bytes[sizeof(T)],swapped[sizeof(T)];
  if(fread(bytes,1,sizeof(T),file)!=sizeof(T)) std::abort();
  for(size_t i=0;i<sizeof(T);++i) swapped[i]=bytes[sizeof(T)-1-i];
  T value;memcpy(&value,swapped,sizeof(T));return value;
 }
 bool logical() {return scalar<int>()!=0;}
 template<class T> CamArray<T> array() {
  CamArray<T> a;a.rank=scalar<int>();a.count=1;
  for(int d=0;d<a.rank;++d) {
   a.low[d]=scalar<int>();a.extent[d]=scalar<int>()-a.low[d]+1;
   a.stride[d]=a.count;a.count*=a.extent[d];
  }
  auto p=std::make_unique<char[]>(a.count*sizeof(T));
  a.data=a.count?reinterpret_cast<T*>(p.get()):nullptr;storage.push_back(std::move(p));
  for(int i=0;i<a.count;++i) {
   if constexpr(std::is_same_v<T,bool>) a.data[i]=logical();
   else a.data[i]=scalar<T>();
  }
  return a;
 }
 bool eof() {int c=fgetc(file);if(c==EOF) return true;ungetc(c,file);return false;}
};
template<class T> void output_scalar(FILE* file,T value) {
 unsigned char bytes[sizeof(T)],swapped[sizeof(T)];memcpy(bytes,&value,sizeof(T));
 for(size_t i=0;i<sizeof(T);++i) swapped[i]=bytes[sizeof(T)-1-i];
 fwrite(swapped,1,sizeof(T),file);
}
void output_logical(FILE* file,bool value) {output_scalar(file,int(value));}
template<class T> void output_array(FILE* file,CamArray<T> a) {
 output_scalar(file,a.rank);
 for(int d=0;d<a.rank;++d) {output_scalar(file,a.low[d]);output_scalar(file,a.low[d]+a.extent[d]-1);}
 for(int i=0;i<a.count;++i) {
  if constexpr(std::is_same_v<T,bool>) output_logical(file,a.linear(i));
  else output_scalar(file,a.linear(i));
 }
}
'''


def reader(v):
    if v.dims: return 'array<'+v.typ+'>()'
    return 'logical()' if v.typ=='bool' else 'scalar<'+v.typ+'>()'


def write_fortran(v,unit):
    out=[]
    if v.dims:
        out+=['write('+unit+') '+str(len(v.dims))]
        for d in range(1,len(v.dims)+1):
            out+=['write('+unit+') lbound('+v.name+','+str(d)+'),ubound('+v.name+','+str(d)+')']
    out+=['write('+unit+') '+v.name]
    return out


def write_optional_fortran(v,unit):
    if not v.optional:return write_fortran(v,unit)
    return ['if(present('+v.name+')) then']+write_fortran(v,unit)+['else',
            'write('+unit+') '+str(len(v.dims))]+['write('+unit+') 1,0']*len(v.dims)+['endif']


def prepare(directory: Path,receipts: Path,levels: int=26):
    if levels<22:raise ValueError('synthetic cloud profile requires at least 22 levels')
    paths=[directory/'module_ra_cam_support.F',directory/'module_ra_cam.F']
    core,inventory,mod,routines=tr.generate(paths)
    (directory/'cam_core.cuh').write_text(core)
    shutil.copyfile(HERE/'array_abi.cuh',directory/'array_abi.cuh')
    fields=[v for v in mod.values() if v.typ!='char' and (not v.const or v.dims)]
    rad=routines['camrad']
    fn=['module cam_capture','use module_ra_cam','implicit none','contains',
        'subroutine capture_state(after)','logical,optional::after','integer :: u',
        "character(len=32)::filename","filename='cam-state.bin'",
        "if(present(after)) then", "if(after) filename='cam-state-after.bin'",'endif',
        "open(newunit=u,file=trim(filename),form='unformatted',access='stream',status='replace')"]
    for v in fields: fn+=write_fortran(v,'u')
    fn+=['close(u)','end subroutine',
         'subroutine capture_camrad(before,'+','.join(rad.args)+')',
         'logical,intent(in)::before','integer::u']
    for a in sorted(rad.args,key=lambda a:bool(rad.variables[a].dims)):
        v=rad.variables[a]
        typ={'int':'integer','float':'real','double':'real(r8)','bool':'logical'}[v.typ]
        fn+=[typ+(',optional' if v.optional else '')+' :: '+a+('('+','.join(v.dims)+')' if v.dims else '')]
    fn+=['if(before)then',"open(newunit=u,file='cam-before.bin',form='unformatted',access='stream',status='unknown',position='append')",
         'else',"open(newunit=u,file='cam-after.bin',form='unformatted',access='stream',status='unknown',position='append')",'endif',
         'write(u) indirect']
    for a in rad.args: fn+=write_optional_fortran(rad.variables[a],'u')
    fn+=['close(u)','end subroutine','end module']
    (directory/'cam_capture.F90').write_text('\n'.join(fn)+'\n')
    source=(HERE/'run_columns.F90').read_text()
    source=source.replace('nv=26','nv='+str(levels),1)
    source=source.replace('program cam_columns','program cam_columns_capture')
    source=source.replace('  use mpas_kind_types','  use cam_capture\n  use mpas_kind_types',1)
    source=source.replace('  ! Last year','  call capture_state()\n  ! Last year',1)
    source=source.replace('  close(u)\n  write','  close(u)\n  call capture_state(.true.)\n  write',1)
    start=source.index('      call camrad(')
    end=source.index('\n      if (.not.diagnostic)',start)
    call=source[start:end]
    source=source[:start]+call.replace('call camrad(','call capture_camrad(.true.,')+'\n'+call+'\n'+call.replace('call camrad(','call capture_camrad(.false.,')+source[end:]
    (directory/'run_columns_capture.F90').write_text(source)
    host=['#include <type_traits>','#include <string>','#include "array_abi.cuh"','#include "cam_core.cuh"',IO,
          'int main(int argc,char** argv) {',
          'if(argc!=4 && argc!=5) return 2;',
          'std::vector<char> workspace(64*1024*1024);CamArena arena(workspace.data(),workspace.size());',
          'CamColumn cam(arena);Input state(argv[1]);']
    for v in fields: host+=['cam.'+v.name+'=state.'+reader(v)+';']
    host+=['Input inputs(argv[2]);FILE* out=fopen(argv[3],"wb");if(!out) return 2;',
           'int records=0;while(!inputs.eof()) {','inputs.storage.clear();cam.indirect=inputs.logical();']
    for a in rad.args: host+=['auto '+a+'=inputs.'+reader(rad.variables[a])+';']
    call='cam.camrad('+','.join(rad.args)+');'
    host+=['if(argc==5) {',
           'int chunk_first=its,chunk_last=ite;cam.native_chunk_its=its;',
           'bool native_doabs=doabsems || (abstot_3d(its,kts,kts,jts)==0.f && dolw);',
           'for(int column=chunk_first;column<=chunk_last;++column) {',
           'its=column;ite=column;doabsems=native_doabs;',call,
           'if(cam.error) break;}',
           'its=chunk_first;ite=chunk_last;cam.native_chunk_its=0;',
           '} else {',call,'}',
           'if(cam.error) {fprintf(stderr,"CAM_ERROR=%d\\n",cam.error);return 3;}',
           'output_logical(out,cam.indirect);']
    for a in rad.args:
        v=rad.variables[a]
        writer='output_array' if v.dims else 'output_logical' if v.typ=='bool' else 'output_scalar'
        host+=[writer+'(out,'+a+');']
    host+=['++records;}fclose(out);',
           'out=fopen((std::string(argv[3])+".state").c_str(),"wb");if(!out) return 2;']
    for v in fields:
        writer='output_array' if v.dims else 'output_logical' if v.typ=='bool' else 'output_scalar'
        host+=[writer+'(out,cam.'+v.name+');']
    host+=['fclose(out);printf("CAM_HOST_RECORDS=%d\\nCAM_HOST_WORKSPACE_PEAK=%zu\\n",records,arena.peak);return 0;}']
    (directory/'host_witness.cpp').write_text('\n'.join(host)+'\n')
    make=(directory/'Makefile').read_text().split('\ncam_columns_capture:',1)[0]
    native_link = next(line for line in make.splitlines()
                       if line.startswith('cam_columns: '))
    objects = [name for name in native_link.split(': ', 1)[1].split()
               if name != 'run_columns.o']
    make+='\ncam_columns_capture: '+ ' '.join(objects)+' cam_capture.o run_columns_capture.o\n\t$(FC) $(FFLAGS) -o $@ $^\n'
    make+='cam_capture.o: cam_capture.F90 mpas_atmphys_camrad_init.o\n\t$(FC) $(FFLAGS) -c $< -o $@\n'
    make+='run_columns_capture.o: run_columns_capture.F90 cam_capture.o\n\t$(FC) $(FFLAGS) -c $< -o $@\n'
    make+='host_witness: host_witness.cpp array_abi.cuh cam_core.cuh\n\t$(CXX) -std=c++17 -O3 -g -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $< -o $@\n'
    (directory/'Makefile').write_text(make)
    inventory['host_witness_scope']='complete CAM core/support; native state loaded from pristine initializer'
    inventory['schema']={'globals':[v.__dict__ for v in fields], 'arguments':[rad.variables[a].__dict__ for a in rad.args]}
    inventory['generator_sha256']=hashlib.sha256((HERE/'translate.py').read_bytes()).hexdigest()
    inventory['synthetic_levels']=levels
    (receipts/'translation-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--receipts',type=Path,required=True)
    p.add_argument('--levels',type=int,default=26)
    a=p.parse_args();prepare(a.directory,a.receipts,a.levels)
