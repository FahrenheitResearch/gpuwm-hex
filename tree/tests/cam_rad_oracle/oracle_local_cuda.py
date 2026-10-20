"""Restore every CAM local SAVE/DATA object at the f1 replay boundary.

The whole numeric transcription is unchanged apart from using the supplied
native storage for these objects. No native output is used as an input.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import translate as tr


def prepare(source, module_path, output, receipt):
    oracle = Path(__file__).resolve().parents[2] / 'tools/native_physics_reference/oracle'
    sys.path.insert(0, str(oracle))
    from fortran_source import routines as native_routines
    from local_state import local_variables
    files = [source/'module_ra_cam_support.F', source/'module_ra_cam.F']
    _, translated = tr.parse(files)
    locals_ = []
    for path in files:
        for native in native_routines(path, source):
            variables, issues = local_variables(native)
            if issues:
                raise ValueError('CAM native local inventory incomplete: '+repr(issues))
            for field in variables:
                if native.name not in translated:
                    raise ValueError('Local state outside complete CAM transcription')
                variable = translated[native.name].variables[field['name']]
                locals_.append(dict(name='local.'+native.module+'.'+native.name+'.'+field['name'],
                                    routine=native.name, variable=field['name'], typ=variable.typ,
                                    rank=len(variable.dims), source=path.name,
                                    first_line=native.start+1,last_line=native.end))
    spec=importlib.util.spec_from_file_location('cam_oracle_source',module_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    code=module.CUDA_SOURCE
    assert code.count('CamArena &arena; int error=0;')==1
    members='''CamArena &arena; int error=0;
const unsigned long long* oracle_locals=nullptr;
template<class T> __device__ CamArray<T> oracle_array(int offset) {
 const auto* p=oracle_locals+offset;CamArray<T> a;
 a.data=reinterpret_cast<T*>(p[0]);a.rank=int(p[1]);a.count=int(p[2]);
 for(int d=0;d<5;++d) {a.low[d]=int(p[3+d]);a.extent[d]=int(p[8+d]);a.stride[d]=int(p[13+d]);}
 return a;
}
'''
    code=code.replace('CamArena &arena; int error=0;',members)
    for index, field in enumerate(locals_):
        routine, name, typ=field['routine'],field['variable'],field['typ']
        start=code.index('CAM_ROUTINE ',code.index('// MPAS '+field['source']+':'+str(field['first_line'])+'-'))
        end=code.find('\n// MPAS ',start+1)
        # Find the next routine marker, not the statement citations in this one.
        following=re.search(r'\n// MPAS \w+\.F:\d+-\d+\nCAM_ROUTINE ',code[start:])
        end=start+following.start() if following else code.index('\n};',start)
        section=code[start:end]
        offset=index*18
        citation='// MPAS '+field['source']+':'+str(field['first_line'])+'-'+str(field['last_line'])+': native local SAVE/DATA storage.\n'
        if field['rank']:
            pattern=r'(auto '+re.escape(name)+r'=arena\.array<'+typ+r'>\([^;]+;\n\{ const '+typ+r' _init\[\]=\{[^\n]+\n)'
            replacement=r'\1'+citation+'if(oracle_locals && oracle_locals['+str(offset)+']) '+name+'=oracle_array<'+typ+'>('+str(offset)+');\n'
            section,count=re.subn(pattern,lambda m:m[1]+citation+'if(oracle_locals && oracle_locals['+str(offset)+']) '+name+'=oracle_array<'+typ+'>('+str(offset)+');\n',section)
        else:
            pattern=r'\b'+typ+' '+re.escape(name)+r'=([^;]+);'
            section,count=re.subn(pattern,lambda m:citation+typ+' _oracle_default_'+name+'='+m[1]+';\n'+typ+'& '+name+'=(oracle_locals && oracle_locals['+str(offset)+']) ? *reinterpret_cast<'+typ+'*>(oracle_locals['+str(offset)+']) : _oracle_default_'+name+';',section)
        if count!=1:
            raise ValueError('Local declaration not uniquely augmented: '+field['name'])
        code=code[:start]+section+code[end:]
    old='int* error,int first_column) {'
    assert code.count(old)==1
    code=code.replace(old,'int* error,int first_column,const unsigned long long* locals) {')
    old='CamArena arena(workspace+blockIdx.x*slice,slice);CamColumn cam(arena);'
    assert code.count(old)==1
    code=code.replace(old,old+'cam.oracle_locals=locals;')
    text=module_path.read_text()
    marker='CUDA_SOURCE = r'+chr(34)*3
    start=text.index(marker)+len(marker)
    end=text.index('\n'+chr(34)*3,start)
    text=text[:start]+code+text[end:]
    old='workspace=None, utility_state=None):\n    """Complete module_ra_cam.F:179-823'
    assert text.count(old)==1
    text=text.replace(old,
                      'workspace=None, local_state=None, utility_state=None):\n    """Complete module_ra_cam.F:179-823')
    old="        kernel=module.get_function('cam_boundary')"
    new="""        local_meta=cp.zeros(18*len(ORACLE_LOCAL_FIELDS),dtype=cp.uint64)
        if local_state:
            for index,(name,kind,rank) in enumerate(ORACLE_LOCAL_FIELDS):
                if name in local_state:
                    descriptor=_pack(((name,kind,1),),{name:local_state[name]},device)
                    local_meta[index*18:(index+1)*18]=descriptor
        kernel=module.get_function('cam_boundary')"""
    assert text.count(old)==1;text=text.replace(old,new)
    old='error,np.int32(start)))'
    assert text.count(old)==1;text=text.replace(old,'error,np.int32(start),local_meta))')
    old='DOABSEMS_OFFSET,global_meta,(arguments,state,utility_state))'
    assert text.count(old)==1;text=text.replace(old,'DOABSEMS_OFFSET,global_meta,(arguments,state,local_meta,local_state,utility_state))')
    fields=tuple((r['name'],r['typ'],r['rank']) for r in locals_)
    text+='\nORACLE_LOCAL_FIELDS = '+repr(fields)+'\n'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text)
    # Hash the emitted Python literal, including its closing-line newline:
    # that is the exact source passed to NVRTC by the launcher.
    emitted_spec=importlib.util.spec_from_file_location('cam_oracle_emitted',output)
    emitted=importlib.util.module_from_spec(emitted_spec)
    sys.modules[emitted_spec.name]=emitted
    emitted_spec.loader.exec_module(emitted)
    receipt.write_text(json.dumps(dict(scope='whole CAM native local SAVE/DATA storage adapter; execution unqualified',
        base_module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        cuda_source_sha256=hashlib.sha256(emitted.CUDA_SOURCE.encode()).hexdigest(),locals=locals_,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','module','output','receipt'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.module,a.output,a.receipt)
