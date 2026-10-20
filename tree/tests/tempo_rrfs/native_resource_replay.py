"""Compare the production public-table snapshot to pristine native packing.

The published r2 library owns every scientific module. The bridge owns only
the test ABI. Rust serializes the actual native public arrays in native order.
No forecast or native table-generator certification is inferred.
"""
import argparse,ctypes as C,hashlib,importlib.util,json,math,os
from pathlib import Path
import numpy as np
from hexcore.mpasphys.rrfs import tempo_v303 as tempo
ROOT=Path(__file__).resolve().parents[4]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--flags',required=True)
    parser.add_argument('--math',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();flags=json.loads(args.flags)
    args.math=args.math.resolve();args.receipt=args.receipt.resolve()
    module_spec=importlib.util.spec_from_file_location('r2_published_loader',ROOT/'scratch/r2-native/loader.py')
    loader=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(loader)
    native=loader.NativePhysics(ROOT/'scratch/r2-native',test_only=True,pin_set='rrfs')
    bridge=C.CDLL(str(ROOT/'scratch/tempo-build/libtempo_rrfs_oracle.so'))
    bridge.oracle_init.argtypes=bridge.oracle_blob.argtypes=[C.c_void_p]
    os.chdir(ROOT/'scratch/tables')
    flag_array=np.asarray(flags,np.int32);bridge.oracle_init(flag_array.ctypes.data)
    blob_path=args.receipt.with_suffix('.blob.bin')
    packet=tempo.pack_initialized_resources(native,ROOT/'scratch/tempo-build/libtempo_resource_transport.so',blob_path,
        table_files=sorted((ROOT/'scratch/tables').iterdir()),device_math=args.math,flags=flags,first_call_main=True)
    expected=np.empty(sum(math.prod(shape) for _,shape in tempo.TABLE_LAYOUT),np.float64)
    bridge.oracle_blob(expected.ctypes.data)
    expected_hash=hashlib.sha256(memoryview(expected).cast('B')).hexdigest()
    receipt={'schema':'rrfs-tempo-native-public-resource-replay-v1','forecast_proven':False,
        'process_count':1,'flags':flags,'public_table_count':len(tempo.TABLE_LAYOUT),
        'blob_retired_after_comparison':True,
        'native_packing_sha256':expected_hash,'production_rust_packing_sha256':packet['blob']['sha256'],
        'all_blob_words_equal':packet['blob']['sha256']==expected_hash,'resources':packet}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    blob_path.unlink();native.close()
    print(json.dumps(receipt))
    if not receipt['all_blob_words_equal']:raise RuntimeError('production public table/Rust packing differs from native')

if __name__=='__main__':main()
