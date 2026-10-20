"""Exercise the cloud binder against actual f3 storage and f4 pool ownership.

This is a one-process CPU ABI diagnostic, not GPU or forecast acceptance.
No mocked native API or replaced native storage is used.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from hexcore.mpasphys import Backend, SlotCall, StepFlags
from hexcore.mpasphys import radshared_v842 as rad


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','native_source','provider_source','library','receipt'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=parser.parse_args()
    abi=module('radshared_native_binding',a.native_source)
    device=module('hexcore.mpasphys._radshared_device_binding',a.provider_source)
    registry=a.reference/'src/core_atmosphere/Registry.xml'
    tree=ET.parse(registry).getroot()
    wanted={'mesh':('meshDensity',),'sfc_input':('xland',),'diag_physics':('cldfrac',)}
    fields=[]
    for pool,names in wanted.items():
        struct=tree.find(f"./var_struct[@name='{pool}']")
        for name in names:
            row=next(v for v in struct if v.get('name_in_code',v.get('name'))==name)
            fields.append(dict(row.attrib,pool=pool,pool_name=name,
                               dimensions=[d for d in row.get('dimensions','').split() if d!='Time']))
    nz,solve,total=3,4,6
    dims={'nVertLevels':nz,'nVertLevelsP1':nz+1,'nCells':total,'nCellsSolve':solve}
    layout=SimpleNamespace(pools=set(wanted),fields=fields,dimensions=dims,
                           configs={'config_len_disp':np.float32(12000)})
    paths=(Path(rad.__file__),Path(__file__),a.native_source,a.provider_source,
           a.library,a.library.with_name('abi.json'),registry)
    result={'purpose':'actual native cloud ABI storage and pool ownership diagnostic',
            'process_count':1,'forecast_proven':False,
            'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    try:
        with abi.NativePhysics(a.library,test_only=True) as native:
            original={}
            for row in fields:
                shape=tuple(dims[d] for d in row['dimensions'])
                value=np.arange(np.prod(shape),dtype='f4').reshape(shape)
                native.put_field(row['pool'],row['pool_name'],value)
                original[row['pool'],row['pool_name']]=value.copy()
            profiles={}
            for number,name in enumerate(('qv','qc','qi','qs','t','pres','pres_hyd','rho')):
                value=np.arange(total*(nz+1),dtype='f4').reshape(total,nz+1,1)+np.float32(number*100)
                native.put_variable(name+'_p',value);profiles[name]=value.copy()
            radiation={}
            for number,name in enumerate(('qvrad','qcrad','qirad','qsrad','cldfrac','dx','xland')):
                shape=(total,nz+1,1) if number<5 else (total,1)
                value=np.arange(np.prod(shape),dtype='f4').reshape(shape)+np.float32(1000+number*100)
                native.put_variable(name+'_p',value);radiation[name]=value.copy()
            native.put_variable('f_qc',np.asarray(True));native.put_variable('f_qi',np.asarray(False))
            provider=device.NativeDevicePoolViews(SimpleNamespace(native=native,layout=layout),np)
            call=SlotCall('cloudiness','cld_fraction',Backend.HEX_CUDA,1,0.,10.,StepFlags(),None,{})
            provider.begin(call,{})
            pools=rad.bind_native_pools(provider,call)
            for name,value in profiles.items():
                expected=value[:solve,:nz,0].T
                np.testing.assert_array_equal(pools['prepared'][name].view('u4'),expected.view('u4'))
                assert pools['prepared'][name].flags.c_contiguous
            assert bool(pools['f_qc']) and not bool(pools['f_qi'])
            assert pools['config_len_disp'].tobytes()==layout.configs['config_len_disp'].tobytes()
            shared=pools['radshared'];shared['binding_sentinel']=17
            pools['diag_physics']['cldfrac'][0,0]=np.float32(-0.)
            for value in pools['radiation_prepared'].values():
                value.flat[0]=np.float32(-0.)
            provider.commit()
            expected=original['diag_physics','cldfrac'];expected[0,0]=np.float32(-0.)
            np.testing.assert_array_equal(native.get_field('diag_physics','cldfrac').view('u4'),expected.view('u4'))
            for name,value in profiles.items():
                np.testing.assert_array_equal(native.get_variable(name+'_p')[0].view('u4'),value.view('u4'))
            for name,value in radiation.items():
                value.flat[0]=np.float32(-0.)
                np.testing.assert_array_equal(native.get_variable(name+'_p')[0].view('u4'),value.view('u4'))
            provider.begin(call,{})
            assert rad.bind_native_pools(provider,call)['radshared'] is shared
            assert shared['binding_sentinel']==17
            provider.commit()
            result.update(passed=True,top_workspace_and_halo_words_unchanged=True,
                          signed_zero_writeback_preserved=True,shared_carriers_persist=True,
                          all_native_radiation_carriers_committed=True)
    except Exception as exc:
        result.update(passed=False,failure_type=type(exc).__name__,failure=str(exc))
        raise
    finally:
        a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
