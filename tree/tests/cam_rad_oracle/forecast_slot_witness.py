"""Compare each real CAM driver callback with native on identical pool words.

This diagnostic executes an extra pristine native driver call, restores its
explicit inputs and module words, then executes the production CUDA callback.
Private native SAVE is not a full checkpoint. Results therefore diagnose the
slot transaction and do not constitute an unmodified whole-forecast proof.
Rust encodes and compares every captured word. No automatic PRE is seeded.
"""
import hashlib
import json
import os
from pathlib import Path

import forecast_diagnostic


def main():
    import numpy as np
    from compact_codec import codec
    from hexcore.mpasphys import cam_rad_v842 as cam

    root=Path(os.environ['HEX_CAM_SLOT_WITNESS'])
    root.mkdir(parents=True,exist_ok=False)
    original=cam.run_slot
    calls=[]

    def snapshot(owner):
        values={}
        for key in owner.copies:
            if key[0]=='prepared':
                host,lower=owner.native.get_variable(key[1])
            else:
                host=owner.native.get_field(key[0],key[1],level=key[2])
                lower=(1,)*host.ndim
            values[key]=(host,lower)
        for name in ('declin','solcon','doabsems'):
            values[('prepared',name)]=owner.native.get_variable(name)
        return values

    def restore(owner,values):
        for key,(host,lower) in values.items():
            if key[0]=='prepared':owner.native.put_variable(key[1],host,lower=lower)
            else:owner.native.put_field(key[0],key[1],host,level=key[2],
                                       nlevels=owner.native._fields[key[0],key[1]][2])

    def capture(path,values,state):
        fields=[]
        with path.open('wb') as stream:
            for name,host,lower in [(str(key),host,lower) for key,(host,lower) in values.items()]+[
                    ('module.'+name,host,(1,)) for name,host in state.items()]:
                kind={'f':'float' if host.dtype.itemsize==4 else 'double','i':'int'}[host.dtype.kind]
                field=dict(name=name,typ=kind,dims=[':']*host.ndim)
                fields.append(field)
                codec().encode(stream,{'float':1,'double':2,'int':3}[kind],host.ndim,
                    list(lower)+[1]*(5-host.ndim),list(host.shape)+[1]*(5-host.ndim),
                    host.ravel(order='F'))
        return fields

    def witnessed(call):
        binding=call.pools;owner=binding.owner;native=owner.native
        index=len(calls);directory=root/str(index);directory.mkdir()
        before=snapshot(owner);state=owner.words.read();message=owner.message_words.copy()
        fields=capture(directory/'before.bin',before,state)
        arguments=dict(owner.provider.session.arguments('driver_'+call.slot,call))
        native.call('driver_'+call.slot,**arguments)
        expected=snapshot(owner);expected_state=owner.words.read()
        assert capture(directory/'native.bin',expected,expected_state)==fields
        restore(owner,before);owner.words.write(state);owner.message_words[:]=message
        result=original(call)
        actual=snapshot(owner);actual_state=owner.words.read()
        assert capture(directory/'cuda.bin',actual,actual_state)==fields
        comparison=codec().compare(directory/'native.bin',directory/'cuda.bin',[
            dict(name=field['name'],fields=[field],records=1) for field in fields])
        metadata=dict(scope='real prepared CAM driver transaction; extra native call',
            call=index,itimestep=call.itimestep,time_seconds=call.time_seconds,
            slot=call.slot,cam_abs=bool(call.flags.cam_abs),process_count=1,
            whole_model_acceptance=False,automatic_PRE_seed=False,excluded_words=0,
            source_sha256=hashlib.sha256(Path(cam.__file__).read_bytes()).hexdigest(),
            artifact_sha256=hashlib.sha256(Path(os.environ['HEX_CAM_CUDA_ARTIFACT']).read_bytes()).hexdigest(),
            comparison=comparison,fields=fields,
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in directory.iterdir() if p.is_file()})
        (directory/'comparison.json').write_text(json.dumps(metadata,indent=2)+'\n')
        calls.append(metadata)
        (root/'summary.json').write_text(json.dumps(dict(complete=False,calls=calls,
            whole_model_acceptance=False),indent=2)+'\n')
        print('CAM_SLOT_WITNESS',index,call.itimestep,comparison['differing_words'],flush=True)
        if index:
            for name in ('before.bin','native.bin','cuda.bin'):(directory/name).unlink()
        return result

    cam.run_slot=witnessed
    try:
        result=forecast_diagnostic.main()
        (root/'summary.json').write_text(json.dumps(dict(complete=True,calls=calls,
            whole_model_acceptance=False,returncode=result),indent=2)+'\n')
        return result
    finally:cam.run_slot=original


if __name__=='__main__':
    raise SystemExit(main())
