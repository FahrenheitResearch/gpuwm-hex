"""Device replay of every CAM driver slice and complete native initialization."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
from compact_codec import codec
from compact_compare import slices as compare


def replay(directory,inventory,receipt_directory,table_library=None):
    import cupy as cp
    from hexcore.mpasphys import cam_rad_v842 as cam
    transport=codec()
    read_value,write_value=transport.read,transport.write
    print('CAM_STAGE_CUDA_REPLAY_START',flush=True)
    def upload(v,value):
        if v['dims']:
            a,lo,ex,st=value
            return cam.DeviceArray(cp.asarray(a),lo,ex,st)
        return value
    def schema(name):
        return [{'name':n,'typ':t,'dims':[':']*r} for n,t,r in cam.STAGE_FIELDS[name]]
    def metadata_scalars(fields,result,values):
        offset=0;host=result.metadata.get()
        for v in fields:
            if not v['dims']:
                word=host[offset:offset+1]
                if v['typ']=='float':value=word.astype(np.uint32).view(np.float32)[0]
                elif v['typ']=='double':value=word.view(np.float64)[0]
                elif v['typ']=='bool':value=bool(word[0])
                else:value=word.view(np.int64)[0]
                values[v['name']]=value
            offset+=18 if v['dims'] else 1
    def check(result):
        code=int(result.error.get()[0])
        if code:raise RuntimeError(f'CAM stage fatal source line {code}')
    inv=json.loads(inventory.read_text())
    groups=[]
    with (directory/'driver-before.bin').open('rb') as source,(directory/'driver-cuda-after.bin').open('wb') as out:
        for name,fields in inv['schema'].items():
            groups.append((name,fields,inv['records'][name]))
            for record in range(inv['records'][name]):
                values={v['name']:upload(v,read_value(source,v)) for v in fields}
                result=cam.launch_stage(name,values,device=0);check(result)
                metadata_scalars(fields,result,values)
                for v in fields:write_value(out,v,values[v['name']])
            print('CAM_DRIVER_CUDA',name,inv['records'][name],flush=True)
        assert not source.read(1)
    differences=compare(directory/'driver-native-after.bin',directory/'driver-cuda-after.bin',groups,
                        receipt_directory/'driver-cuda.json','strict native MPAS driver slice against hex-cuda')
    global_schema=[{'name':n,'typ':t,'dims':[':']*r} for n,t,r in cam.STATE_FIELDS]
    globals_={}
    with (directory/'init-input-state.bin').open('rb') as source:
        for v in global_schema:globals_[v['name']]=upload(v,read_value(source,v))
        assert not source.read(1)
    # Preserve only the decoded raw table records before the defaults kernel
    # zeroes persistent arrays. Every derived optical/ES table is regenerated.
    raw_names=('ah2onw','eh2onw','ah2ow','cn_ah2ow','cn_eh2ow','ln_ah2ow','ln_eh2ow',
               'kcb','wcb','gcb','kdst','wdst','gdst','kbg','wbg','gbg','kvolc','wvolc','gvolc')
    raw={n:globals_[n].storage.copy() for n in raw_names}
    decoded=None
    if table_library:
        with cam.RustTables(table_library,directory,ozone_levels=59,months=12) as table_owner:
            decoded=table_owner.upload(0)
        raw={n:decoded[n].storage for n in raw_names}
    state=cam.allocate_state(0);globals_=state.values
    stages={}
    with (directory/'init-arguments.bin').open('rb') as source:
        for name in ('camradinit','aer_optics_initialize','aerosol_init','oznini'):
            fields=schema(name)
            stages[name]={v['name']:upload(v,read_value(source,v)) for v in fields}
        assert not source.read(1)
    if decoded:
        # Exercise the real provider-facing initializer, including owned raw
        # Rust buffers, defaults, scalar chaining and the native max reduction.
        results=cam.initialize(state,decoded,constants=stages['camradinit'],
                               aerosols=stages['aerosol_init'],ozone=stages['oznini'],
                               reduce_max=lambda value:value)
        for result in results:check(result)
        print('CAM_INIT_CUDA public initialize API',flush=True)
    else:
        check(cam.launch_stage('initialize_defaults',{},state=state,device=0))
        for n in raw_names:globals_[n].storage[:]=raw[n]
        for name,values in stages.items():
            result=cam.launch_stage(name,values,state=state);check(result)
            if name=='aerosol_init':
                maximum=cam.stage_scalar(result,name,'max_mxaerl')
                offset=cam.STATE_SCALARS['mxaerl'][0]
                state.metadata[offset:offset+1]=maximum.storage[maximum.offset:maximum.offset+1]
            print('CAM_INIT_CUDA',name,flush=True)
    metadata_scalars(global_schema,cam.BoundaryResult(state.metadata,None,None,-1,None,None),globals_)
    with (directory/'init-cuda-state.bin').open('wb') as out:
        for v in global_schema:write_value(out,v,globals_[v['name']])
    differences+=compare(directory/'cam-state.bin',directory/'init-cuda-state.bin',[('CAM global state',global_schema,1)],
                        receipt_directory/'init-cuda-state.json','complete MPAS initializer against hex-cuda')
    pool_schema=[{'name':n,'typ':'float','dims':[':']*r} for n,r in
                 [('hybi',2),('ps',1),('aerp',3),('aern',3),('pin',1),('ozone',3)]]
    pools=[stages['aerosol_init'][n] for n in ('m_hybi','m_psp','aerosolcp','aerosolcn')]+[
           stages['oznini'][n] for n in ('pin','ozmixm')]
    with (directory/'init-cuda-pools.bin').open('wb') as out:
        for v,value in zip(pool_schema,pools,strict=True):write_value(out,v,value)
    differences+=compare(directory/'init-native-pools.bin',directory/'init-cuda-pools.bin',[('MPAS initialized pools',pool_schema,1)],
                        receipt_directory/'init-cuda-pools.json','complete MPAS initialized pools against hex-cuda')
    for filename in ('driver-cuda.json','init-cuda-state.json','init-cuda-pools.json'):
        path=receipt_directory/filename;r=json.loads(path.read_text())
        r.update(backend='hex-cuda',compile_options=cam.COMPILE_OPTIONS,
                 source_sha256=hashlib.sha256(cam.CUDA_SOURCE.encode()).hexdigest(),synthetic_only=True,
                 raw_table_decoder='rw-mpas Rust C ABI' if table_library else 'independent C++ test fixture',
                 initializer_entry='public initialize API' if table_library else 'individual numeric stages')
        path.write_text(json.dumps(r,indent=2)+'\n')
    return bool(differences)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','inventory','receipt_directory'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--table-library',type=Path)
    a=p.parse_args();raise SystemExit(replay(a.directory,a.inventory,a.receipt_directory,a.table_library))
