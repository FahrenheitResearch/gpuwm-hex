"""Same-input whole-forecast native/CUDA slot diagnostic; never acceptance."""
import hashlib,json,os,runpy,subprocess,sys
from pathlib import Path
root=Path.cwd();checkout=Path(os.environ.get('P16_FORECAST_CHECKOUT',str(root/'hex')))
sys.path.insert(0,str(checkout/'tree/src'))
from hexcore.mpasphys import noah_v842 as noah
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
label=os.environ.get('P16_DUAL_LABEL','noah-dual-driver-diagnostic')
directory=root/'scratch'/label;directory.mkdir(parents=True,exist_ok=True)
arrays=directory/'last-call';arrays.mkdir(exist_ok=True)
receipt=root/'receipts'/(label+'.json')
original_begin=NativeDevicePoolViews.begin;original_slot=noah.run_slot
provider=None;calls=[]
def begin(self,call,descriptor):
    global provider
    result=original_begin(self,call,descriptor)
    if call.slot=='land_surface':provider=self
    return result
NativeDevicePoolViews.begin=begin
def slot(call):
    assert provider is not None and call.slot=='land_surface'
    shapes={}
    for name,value in call.pools['fields'].items():
        host=value.get();(arrays/(name+'.input.bin')).write_bytes(host.tobytes(order='C'))
        shapes[name]=dict(shape=list(host.shape),dtype=str(host.dtype))
    # Preserve the actual bound scalar words and tables rather than inferring
    # a replay interval or category from the deck after a differing call.
    arguments={name:(value.item() if hasattr(value,'item') else value)
               for name,value in call.pools.items() if name not in ('fields','tables','stream')}
    arguments.update(itimestep=call.itimestep,options=dict(call.options))
    table_bytes=call.pools['tables'].get().tobytes(order='C')
    (arrays/'tables.input.bin').write_bytes(table_bytes)
    (arrays/'arguments.json').write_text(json.dumps(arguments,indent=2)+'\n')
    status=original_slot(call)
    # Native pools still contain the exact pre-CUDA words until provider.commit.
    # Native SAVE variables advance as in an all-native run. This diagnostic
    # side effect is explicitly outside the production forecast proof.
    provider.session.hooks.invoke('driver_lsm',call)
    names=[];bindings={}
    for key,(host,packed,selection) in provider.buffers.items():
        name='__'.join(map(str,key));names.append(name)
        if key[0]=='prepared':
            value=provider.native.get_variable(key[1])[0]
            native=value[:packed.shape[1],:,0].T if value.ndim==3 else value[:packed.shape[0],0]
        else:
            native=provider.native.get_field(key[0],key[1],level=key[2])[selection]
        candidate=packed.get()
        assert native.shape==candidate.shape and native.dtype==candidate.dtype
        (arrays/(name+'.native.bin')).write_bytes(native.tobytes(order='C'))
        (arrays/(name+'.cuda.bin')).write_bytes(candidate.tobytes(order='C'))
        bindings[name]=dict(shape=list(candidate.shape),dtype=str(candidate.dtype))
    (arrays/'fields.txt').write_text('\n'.join(names)+'\n')
    process=subprocess.run([str(root/'scratch/noah-dual-driver-diagnostic/compare_noah_call'),str(arrays)],capture_output=True,text=True,check=True)
    result=json.loads(process.stdout);result.update(itimestep=call.itimestep,time_seconds=call.time_seconds)
    calls.append(result)
    packet=dict(schema='p16-noah-whole-forecast-dual-driver-diagnostic-v1',acceptance=False,
        scope='same input actual CUDA slot and strict native driver; native module SAVE side effects retained; diagnostic only',
        process_count=1,production_sha256=hashlib.sha256((checkout/'tree/src/hexcore/mpasphys/noah_v842.py').read_bytes()).hexdigest(),
        calls=calls,first_difference=result if result['differing_words'] else None,
        inputs=shapes,bindings=bindings,
        arguments=arguments,tables_sha256=hashlib.sha256(table_bytes).hexdigest(),
        orchestration_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compiler_disabled=True)
    if result['differing_words']:
        packet['array_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in arrays.iterdir() if p.is_file()}
    receipt.write_text(json.dumps(packet,indent=2)+'\n')
    print('NOAH-DUAL',call.itimestep,call.time_seconds,'words',result['words'],'diff',result['differing_words'],flush=True)
    if result['differing_words']:raise RuntimeError('Noah same-input first difference preserved in '+str(receipt))
    return status
noah.run_slot=slot
runpy.run_path(str(root/'scratch/run-corrected-noah-forecast.py'),run_name='__main__')
