"""Exercise the production CUDA resource loader and compare all packed words.

MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:23-857;
MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:42-182.
Python orchestrates device transfer; Rust writes the copied native words.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import cupy as cp
from hexcore.mpasphys.rrfs.tempo_v303 import load_resources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    packet=json.loads(args.resources.read_text())['mp_tempo']
    resource=load_resources(packet,cp)
    copied=resource['tables'].get()
    transport=C.CDLL(packet['rust_transport']['path'])
    transport.tempo_write.argtypes=[C.c_char_p,C.c_void_p,C.c_size_t]
    transport.tempo_write.restype=C.c_int
    temp=args.receipt.with_suffix('.roundtrip.bin')
    if temp.exists():raise FileExistsError(temp)
    try:
        if transport.tempo_write(str(temp).encode(),copied.ctypes.data,copied.nbytes):
            raise OSError('Rust device roundtrip write failed')
        status=subprocess.run(['cmp','--silent',packet['blob']['path'],str(temp)]).returncode
        with temp.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        receipt={'schema':'r5.forecast-resources-cuda-loader/v1','flags':list(resource['flags']),
                 'first_call_main':resource['first_call_main'],'compared_binary64_words':copied.size,
                 'differing_words':0 if status==0 else None,'complete_bytes_equal':status==0,
                 'cuda_device':cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name'].decode(),
                 'resources_sha256':hashlib.sha256(args.resources.read_bytes()).hexdigest(),
                 'blob_sha256':packet['blob']['sha256'],'device_roundtrip_sha256':digest,
                 'math_sha256':packet['device_math']['sha256'],'forecast_proven':False,
                 'production_loader_used':True,'roundtrip_retired':True}
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
        if status:raise ValueError('production CUDA loader roundtrip differs')
        print(json.dumps(receipt,indent=2))
    finally:
        temp.unlink(missing_ok=True)


if __name__=='__main__':main()
