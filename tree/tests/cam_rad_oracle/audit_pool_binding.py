"""Audit the CAM binding against an actual authenticated strict native ABI.

No synthetic provider or forecast claim: inspect every declared transfer,
read/write actual GNU module words and require immutable PARAMETER storage.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from hexcore.mpasphys import cam_rad_v842 as cam
from hexcore.mpasphys.native_backend import NativePhysics


def audit(library, receipt):
    with NativePhysics(library, test_only=True) as native:
        words = cam.NativeCAMWords(native)
        # Read and return all actual exported GNU CHARACTER bytes, MPAS utilities.F:18.
        message=(ct.c_ubyte*512).in_dll(native.lib,'__mpas_atmphys_utilities_MOD_mpas_err_message')
        message_before=bytes(message)
        message[:]=message_before
        message_after=bytes(message)
        before = words.read()
        words.write(before)
        after = words.read()
        differences = sum(a.tobytes() != after[name].tobytes()
                          for name, a in before.items())
        constants = set(words.views) - set(cam.NATIVE_STATE_MUTABLE)
        rejected = []
        for name in sorted(constants):
            changed = dict(before)
            value = before[name].copy()
            value.view('uint8')[0] ^= 1
            changed[name] = value
            try:
                words.write(changed)
            except ValueError as exc:
                if 'PARAMETER' not in str(exc):
                    raise
                rejected.append(name)
            else:
                raise AssertionError('read-only native PARAMETER was writable: ' + name)
        available = (set(cam.NATIVE_POOL_BINDINGS) | set(native.abi['variables']) |
                     {'its','ite','xtime_s','degrad','dpd','solcon_0'})
        managers = {'curr_julday','gmt','julday','year'}
        for name in managers:
            native.get_table('mpas_atmphys_manager.' + name)
        available |= managers
        missing = {}
        for stage in ('radiation_lw_from_mpas','radiation_lw_to_mpas',
                      'radiation_sw_from_mpas','radiation_sw_to_mpas','radconst'):
            required = {('curr_julday' if stage == 'radconst' and f == 'julian' else f)
                        for f,_,_ in cam.STAGE_FIELDS[stage]}
            if required - available:
                missing[stage] = sorted(required - available)
        dynamic = {'dolw','dosw','doabsems','n_cldadv','declin','solcon','xtime','radt'}
        required = {v for k,v in cam.NATIVE_ARGUMENT_BINDINGS.items() if k not in dynamic}
        if required - available:
            missing['camrad'] = sorted(required - available)
        hashes = {n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in before.items()}
        result = dict(scope='actual strict-native module word transport and driver ABI audit; not initialized arithmetic or forecast proof',
                      process_count=1, library_sha256=native.build['library_sha256'],
                      source_sha256=cam.NATIVE_SOURCE_PINS,state_fields=len(words.views),
                      state_words=sum(a.size for a in before.values()),
                      changed_fields=differences,framework_message_bytes=512,
                      framework_message_sha256=hashlib.sha256(message_before).hexdigest(),
                      framework_message_changed_bytes=sum(a!=b for a,b in zip(message_before,message_after)),
                      immutable_parameters_rejected=rejected,
                      missing=missing,hashes=hashes,passed=not missing and differences==0 and message_before==message_after,
                      whole_model_acceptance=False)
        receipt.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='hashes'}))
        return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(audit(args.library,args.receipt))
