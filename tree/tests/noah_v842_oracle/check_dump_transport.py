"""Verify f1 typed transport against existing native ABI columns and halos."""
from pathlib import Path
import hashlib
import json
import argparse
import os
import subprocess
import sys
import tempfile
import numpy as np

base=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--state',action='store_true');a=p.parse_args()
z=np.load('n1out/driver-synthetic.npz')
with tempfile.TemporaryDirectory(prefix='dump-transport-',dir='n1out') as directory:
    root=Path(directory);abi=root/'abi';capture=root/'capture';packed=root/'packed';out=root/'output'
    Path(str(abi)+'.f32').write_bytes(z['input'].tobytes())
    Path(str(abi)+'.i32').write_bytes(z['input_int'].tobytes())
    command=[str(base/'dump_transport')]
    fixture=[*command,'fixture',str(abi),str(capture)]
    if a.state:
        table=root/'native.table';table.write_bytes(z['table_bytes'].tobytes());fixture.append(str(table))
    subprocess.run(fixture,check=True)
    subprocess.run([*command,'pack',str(capture),str(packed),str(base/'table_decode'),str(base)],check=True)
    floats=Path(str(packed)+'.f32').read_bytes();ints=Path(str(packed)+'.i32').read_bytes()
    assert floats==z['input'][:16].tobytes()
    assert ints==z['input_int'][:16].tobytes()
    assert Path(str(packed)+'.table').read_bytes()==z['table_bytes'].tobytes()
    names=[p.name[len('capture.pre.'):-4] for p in root.glob('capture.pre.*.bin')]
    listing=root/'names';listing.write_text('\n'.join(names)+'\n')
    subprocess.run([*command,'unpack',str(capture),str(out),str(packed),str(listing)],check=True)
    # Unchanged fields, all halo bytes and metadata survive a pack/unpack cycle.
    for name in names:
        before=Path(str(capture)+f'.pre.{name}.bin').read_bytes()
        after=Path(str(out)+f'.post.{name}.bin').read_bytes()
        assert before[:348]==after[:348] and before[604:]==after[604:]
        assert after[348:604]==b'post'+b' '*252
    result={'columns':16,'transport_records':len(names),'packed_float_bytes':len(floats),'packed_integer_bytes':len(ints),'differing_bytes':0,'inactive_halo_bytes_preserved':True,'table_bytes_equal_native':True,'source':'existing strict native driver synthetic columns','adapter_sha256':hashlib.sha256((base/'dump_transport').read_bytes()).hexdigest(),'scope':'f1 typed transport; no real f1 capture or forecast proof'}
    result['module_table_state_records']=sum(n.startswith('state.module_sf_noahlsm.') for n in names)
    if a.gpu:
        native=root/'native';candidate=root/'candidate'
        Path(str(native)+'.f32').write_bytes(z['native'][:16].tobytes())
        Path(str(native)+'.i32').write_bytes(z['native_int'][:16].tobytes())
        subprocess.run([*command,'unpack',str(capture),str(capture),str(native),str(listing)],check=True)
        # Metadata identifies the call; expected numeric payloads are exclusively
        # read below for comparison, after the executable returns its outputs.
        site='lsm__driver_lsm__L705'
        (root/'manifest.json').write_text(json.dumps({'captures':{'capture':{'site':site}},'sites':[{'site':site,'scheme':'lsm','callee':{'module':'module_sf_noahdrv'}}]}))
        executable = ([str(base/'replay_f1_cuda.sh')] if os.environ.get('P16_NOAH_CUBIN_MANIFEST')
                      else [sys.executable,str(base/'replay_f1_cuda.py')])
        subprocess.run([*executable,str(capture),str(candidate)],check=True)
        for name in names:
            want=Path(str(capture)+f'.post.{name}.bin').read_bytes()
            got=Path(str(candidate)+f'.post.{name}.bin').read_bytes()
            assert got==want, name
        result['cuda_differing_bytes']=0
        result['scope']='f1 typed synthetic driver replay against strict native; no real capture or forecast proof'
    filename=('receipts/dump-state-cuda.json' if a.gpu else 'n1out/dump-state.json') if a.state else ('receipts/dump-transport-cuda.json' if a.gpu else 'n1out/dump-transport.json')
    Path(filename).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
