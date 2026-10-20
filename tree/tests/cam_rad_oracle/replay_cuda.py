"""Replay complete pristine CAM captures on CUDA, through the lane launcher."""
from pathlib import Path
import argparse
import json
import struct
import hashlib
import numpy as np


def validate_checkpoint(saved,identity,directory):
    if saved['identity']!=identity:raise ValueError('CAM replay checkpoint input/artifact changed')
    required={'cam-cuda-checkpoint-state.bin','cam-cuda-checkpoint-workspace.bin','cam-cuda-after.bin'}
    if set(saved['files'])!=required:raise ValueError('CAM replay checkpoint is incomplete')
    for name,digest in saved['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('CAM replay checkpoint bytes changed: '+name)
    if saved['records']<=0 or saved['input_offset']!=(directory/'cam-cuda-after.bin').stat().st_size:
        raise ValueError('CAM replay checkpoint capture offset changed')


def recover_incomplete_tail(saved,identity,directory):
    """Retire only an interrupted suffix after authenticating saved raw bytes."""
    if saved['identity']!=identity:raise ValueError('CAM replay checkpoint input/artifact changed')
    path=directory/'cam-cuda-after.bin'
    offset=saved['input_offset']
    size=path.stat().st_size
    if size==offset:
        validate_checkpoint(saved,identity,directory)
        return
    if size<offset:raise ValueError('CAM replay checkpoint output was truncated')
    for name,digest in saved['files'].items():
        if name==path.name:
            with path.open('rb') as stream: actual=hashlib.sha256(stream.read(offset)).hexdigest()
        else:actual=hashlib.sha256((directory/name).read_bytes()).hexdigest()
        if actual!=digest:raise ValueError('CAM replay checkpoint bytes changed: '+name)
    with path.open('r+b') as stream:stream.truncate(offset)
    validate_checkpoint(saved,identity,directory)
    receipt=directory/'cam-cuda-interrupted-tail.json'
    history=json.loads(receipt.read_text()) if receipt.exists() else []
    history.append(dict(scope='discarded uncommitted interrupted replay tail; no output proof',
                        authenticated_records=saved['records'],authenticated_bytes=offset,
                        discarded_bytes=size-offset,identity=identity))
    receipt.write_text(json.dumps(history,indent=2)+'\n')


def read_value(stream,field):
    count=1;low=[];extent=[];stride=[]
    if field['dims']:
        rank=struct.unpack('>i',stream.read(4))[0]
        assert rank==len(field['dims'])
        for _ in range(rank):
            lo,hi=struct.unpack('>ii',stream.read(8));low.append(lo)
            extent.append(hi-lo+1);stride.append(count);count*=hi-lo+1
        if count==0:
            if not field.get('optional'):raise ValueError('Absent nonoptional CAM descriptor: '+field['name'])
            return None
    dtype={'float':'>f4','double':'>f8','int':'>i4','bool':'>i4'}[field['typ']]
    a=np.frombuffer(stream.read(count*np.dtype(dtype).itemsize),dtype=dtype).astype(dtype[1:])
    if field['typ']=='bool':a=a.astype(np.bool_)
    return (a,tuple(low),tuple(extent),tuple(stride)) if field['dims'] else a[0].item()


def write_value(stream,field,value):
    if value is None:
        if not field.get('optional'):raise ValueError('Absent nonoptional CAM descriptor: '+field['name'])
        stream.write(struct.pack('>i',len(field['dims'])))
        for _ in field['dims']:stream.write(struct.pack('>ii',1,0))
        return
    # Device modules only for a present value: the absent-descriptor path is
    # checked on the CPU tier, which has no CuPy.
    from hexcore.mpasphys.cam_rad_v842 import DeviceArray
    import cupy as cp
    if isinstance(value,DeviceArray):
        stream.write(struct.pack('>i',len(value.extent)))
        for lo,n in zip(value.lower,value.extent):stream.write(struct.pack('>ii',lo,lo+n-1))
        a=cp.asnumpy(value.storage)
    else:a=np.asarray([value])
    dtype={'float':'>f4','double':'>f8','int':'>i4','bool':'>i4'}[field['typ']]
    stream.write(a.astype(dtype).tobytes())


def replay(directory,inventory,receipt,*,record_limit=None,checkpoint=None,native_chunk=False,recover_tail=False,rust_transport=False,native_entry=None):
    import cupy as cp
    from hexcore.mpasphys.cam_rad_v842 import DeviceArray,launch_boundary,COMPILE_OPTIONS,CUDA_SOURCE,make_device_state,STATE_SCALARS
    schema=json.loads(inventory.read_text())['schema']
    if native_entry and not rust_transport:
        raise ValueError('Observed native PRE requires Rust-transport replay')
    if rust_transport:
        from compact_codec import codec
        from compact_compare import boundary as compare, slices as compare_state
        transport=codec()
        read_field=transport.read;write_field=transport.write
    else:
        # Immutable historical qualification runs can finish their old path.
        from compare_boundary import compare
        from compare_slices import compare as compare_state
        read_field=read_value;write_field=write_value
    print('CAM_CUDA_REPLAY_START',flush=True)
    state={}
    def upload(field,value):
        if value is None:return None
        if field['dims']:
            a,lo,ex,st=value
            return DeviceArray(cp.asarray(a),lo,ex,st)
        return value
    identity={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
              [('input',directory/'cam-before.bin'),('inventory',inventory),
               ('initial_state',directory/'cam-state.bin')]}
    identity['cuda_source']=hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()
    if rust_transport:
        identity['transport_library']=hashlib.sha256(transport.path.read_bytes()).hexdigest()
    if native_chunk:identity['kernel_mode']='native-chunk'
    artifact=__import__('os').environ.get('HEX_CAM_CUDA_ARTIFACT')
    if checkpoint and not artifact:raise ValueError('Checkpoint replay requires a fixed CUDA artifact')
    identity['artifact']=hashlib.sha256(Path(artifact).read_bytes()).hexdigest() if artifact else None
    entry_streams={}
    if native_entry:
        for name in ('tauxci','tauxcl'):
            path=Path(native_entry)/('entry-'+name+'.bin')
            identity['entry_'+name]=hashlib.sha256(path.read_bytes()).hexdigest()
            entry_streams[name]=path.open('rb')
    saved=None
    if checkpoint and checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if recover_tail:recover_incomplete_tail(saved,identity,directory)
        validate_checkpoint(saved,identity,directory)
    state_path=directory/('cam-cuda-checkpoint-state.bin' if saved else 'cam-state.bin')
    with state_path.open('rb') as stream:
        for field in schema['globals']:state[field['name']]=upload(field,read_field(stream,field))
        assert not stream.read(1)
    persistent=make_device_state(state,0)
    count=saved['records'] if saved else 0
    for name,entry in entry_streams.items():
        for _ in range(count):
            read_field(entry,dict(name=name,typ='double',dims=['column','level']))
        if saved and entry.tell()!=saved['native_entry_offsets'][name]:
            raise ValueError('Observed native PRE checkpoint offset changed')
    workspace=cp.asarray(np.fromfile(directory/'cam-cuda-checkpoint-workspace.bin',dtype=np.uint8)) if saved else None
    def save_state(path):
        scalar_words=persistent.metadata.get()
        with path.open('wb') as out:
            for field in schema['globals']:
                name=field['name'];value=state[name]
                if not field['dims']:
                    offset,kind=STATE_SCALARS[name];word=scalar_words[offset:offset+1]
                    if kind=='double':value=word.view(np.float64)[0]
                    elif kind=='float':value=word.astype(np.uint32).view(np.float32)[0]
                    elif kind=='bool':value=bool(word[0])
                    else:value=word.view(np.int64)[0]
                write_field(out,field,value)
    processed=0
    with (directory/'cam-before.bin').open('rb') as stream,(directory/'cam-cuda-after.bin').open('ab' if saved else 'wb') as out:
        if saved:stream.seek(saved['input_offset'])
        while stream.tell()<(directory/'cam-before.bin').stat().st_size:
            lead=dict(name='indirect',typ='bool',dims=[])
            if rust_transport:
                indirect=read_field(stream,lead)
                state['indirect']=bool(indirect)
            else:
                raw=stream.read(4)
                state['indirect']=bool(struct.unpack('>i',raw)[0])
            persistent.metadata[STATE_SCALARS['indirect'][0]]=state['indirect']
            arguments={field['name']:upload(field,read_field(stream,field)) for field in schema['arguments']}
            free,total=cp.cuda.runtime.memGetInfo()
            if count==0:
                print('CAM_CUDA_COMPILE_AND_FIRST_CALL',flush=True)
                print('CAM_CUDA_FIRST_CALL_MEMORY',free,total,flush=True)
            try:
                kwargs={'native_chunk':True} if native_chunk else {}
                if native_entry:
                    kwargs['local_state']={
                        'allocation.module_ra_cam.camrad.'+name:upload(
                            dict(name=name,typ='double',dims=['column','level']),
                            read_field(entry,dict(name=name,typ='double',dims=['column','level'])))
                        for name,entry in entry_streams.items()}
                result=launch_boundary(arguments,persistent,workspace=workspace,**kwargs)
                workspace=result.workspace
                error=int(result.error.get()[0])
            except Exception as exc:
                receipt.with_name(receipt.stem+'-failure.json').write_text(json.dumps({
                    'scope':'CAM CUDA launch failure; no completed boundary identity proof',
                    'completed_records':count,'exception':type(exc).__name__,'message':str(exc),
                    'free_bytes_before_call':free,'total_bytes':total,
                    'cuda_source_sha256':hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
                    'compile_options':COMPILE_OPTIONS},indent=2)+'\n')
                raise
            if error:raise RuntimeError(f'MPAS CAM fatal source line {error}')
            arguments['doabsems']=bool(result.metadata[result.doabsems_offset].get())
            if rust_transport:write_field(out,lead,indirect)
            else:out.write(raw)
            for field in schema['arguments']:write_field(out,field,arguments[field['name']])
            count+=1
            processed+=1
            print('CAM_CUDA_RECORD',count,flush=True)
            if record_limit and processed>=record_limit and stream.tell()< (directory/'cam-before.bin').stat().st_size:
                if not checkpoint:raise ValueError('Bounded CAM replay requires a checkpoint')
                out.flush()
                save_state(directory/'cam-cuda-checkpoint-state.bin')
                cp.asnumpy(workspace).tofile(directory/'cam-cuda-checkpoint-workspace.bin')
                names=('cam-cuda-checkpoint-state.bin','cam-cuda-checkpoint-workspace.bin','cam-cuda-after.bin')
                saved=dict(identity=identity,records=count,input_offset=stream.tell(),
                           files={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names})
                if entry_streams:
                    saved['native_entry_offsets']={name:entry.tell() for name,entry in entry_streams.items()}
                checkpoint.write_text(json.dumps(saved,indent=2)+'\n')
                # The prefix contains complete captures, including all scratch words.
                prefix=directory/'cam-cuda-reference-prefix.bin'
                with (directory/'cam-after.bin').open('rb') as native:
                    prefix.write_bytes(native.read(out.tell()))
                compare(prefix,directory/'cam-cuda-after.bin',inventory,receipt)
                prefix.unlink()
                partial=json.loads(receipt.read_text())
                partial.update(complete=False,checkpoint=str(checkpoint),backend='hex-cuda',
                               cuda_source_sha256=identity['cuda_source'],artifact_sha256=identity['artifact'])
                if native_entry:
                    partial.update(observed_native_automatic_pre_test_only=True,
                                   original_unseeded_failure_superseded=False,
                                   observed_entry_sha256={name:identity['entry_'+name] for name in entry_streams})
                receipt.write_text(json.dumps(partial,indent=2)+'\n')
                for entry in entry_streams.values():entry.close()
                return 2
    for entry in entry_streams.values():
        assert not entry.read(1),'Native automatic PRE has excess records'
        entry.close()
    differences=compare(directory/'cam-after.bin',directory/'cam-cuda-after.bin',inventory,receipt)
    result=json.loads(receipt.read_text())
    result.update(backend='hex-cuda',compile_options=COMPILE_OPTIONS,
                  cuda_source_sha256=hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
                  shared_math='engine glibc_flt32/glibc_trig_flt32/glibc_flt64; TEMPO log10; CORE-MATH double cosine',
                  kernel_mode='native-chunk' if native_chunk else 'partitioned',
                  native_state_capture_test_only=True,complete=True)
    if native_entry:
        result.update(observed_native_automatic_pre_test_only=True,
                      observed_entry_sha256={name:identity['entry_'+name] for name in entry_streams},
                      whole_model_acceptance=False,original_unseeded_failure_superseded=False)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    if (directory/'cam-state-after.bin').exists():
        save_state(directory/'cam-cuda-state-after.bin')
        differences+=compare_state(directory/'cam-state-after.bin',directory/'cam-cuda-state-after.bin',
                                   [('CAM post-call module state',schema['globals'],1)],
                                   receipt.with_name(receipt.stem+'-state.json'),
                                   'complete pristine CAM module state after native/hex-cuda calls')
    if checkpoint:
        checkpoint.unlink(missing_ok=True)
        for name in ('cam-cuda-checkpoint-state.bin','cam-cuda-checkpoint-workspace.bin'):
            (directory/name).unlink(missing_ok=True)
    return bool(differences)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','inventory','receipt'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();raise SystemExit(replay(a.directory,a.inventory,a.receipt))
