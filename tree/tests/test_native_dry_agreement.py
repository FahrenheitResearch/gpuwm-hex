"""Numerical bound checks and optional live CPU/native one-step comparison."""
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
from hexcore.dycore_agreement import LIMITS, STRICT_IDENTITY_RECIPES, assess, field_report, ulp_distance


def test_six_ulp_boundary_and_near_zero_wind():
    base=np.array([1.0],dtype=np.float32)
    bits=base.view(np.uint32)
    fields={name:field_report(base,base) for name in LIMITS}
    fields['rho']=field_report((bits+6).view(np.float32),base)
    assert assess(fields)['passed']
    fields['rho']=field_report((bits+7).view(np.float32),base)
    assert not assess(fields)['passed']
    fields['rho']=field_report(base,base)
    fields['normal_velocity']=field_report(np.array([1e-4],np.float32),np.array([-1e-6],np.float32))
    assert fields['normal_velocity']['max_ulp']>100000
    assert assess(fields)['passed']
    fields['normal_velocity']=field_report(np.array([3e-4],np.float32),np.zeros(1,np.float32))
    assert not assess(fields)['passed']


def test_a_strict_gnu_native_is_held_to_byte_identity():
    # The dycore claim is 0 differing words against the strict GNU native; one
    # differing word inside the 6 ULP bound passes against an optimised native
    # and fails against the strict one.
    base=np.array([1.0,2.0],dtype=np.float32)
    bits=base.view(np.uint32)
    fields={name:field_report(base,base) for name in LIMITS}
    assert assess(fields,identity=True)['passed']
    fields['rho']=field_report((bits+np.array([1,0],np.uint32)).view(np.float32),base)
    assert assess(fields)['passed']
    assert not assess(fields,identity=True)['passed']
    assert set(STRICT_IDENTITY_RECIPES)=={'gnu-o0','gnu-strict'}


def test_invalid_arrays_and_missing_fields_fail_closed():
    for invalid in (np.array([np.nan],np.float32),np.array([np.inf],np.float32),np.ones(2,np.float32),np.ones(1,np.float64)):
        assert not assess({'rho':field_report(invalid,np.ones(1,np.float32))})['passed']
    assert not assess({})['passed']
    empty=np.empty(0,np.float32)
    assert not assess({name:field_report(empty,empty) for name in LIMITS})['passed']
    assert ulp_distance(np.array([-0.0],np.float32),np.array([0.0],np.float32))[0]==0


def test_live_cpu_native_one_dry_step(tmp_path):
    root=Path(os.environ.get('HEX_NATIVE_DRY_REFERENCE', '.native-dry-reference'))
    init=root/'jw/init/x1.2562.init.nc'
    candidates=[root/f'jw/v8.4.2-{recipe}/rk2-s4' for recipe in ('gnu-o0','ifx-valuesafe','gnu-strict')]
    native=next((p for p in candidates if len(list(p.glob('tracer.*.nc')))>=2),None)
    if not init.is_file() or native is None:
        pytest.skip('native dry reference absent: run gpuwm-hex verify-dycore --backend cpu; set HEX_NATIVE_DRY_REFERENCE to its work root')
    tree=Path(__file__).resolve().parents[1]
    path=tree/'tools/native_dry_reference/run_receipt.py'
    spec=importlib.util.spec_from_file_location('native_case_receipt',path)
    receipt=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receipt)
    build=native.parent.name
    assert receipt.main(['verify-init',str(init.parent),'--standard-init',
        '--expected-executable',str(root/f'native/{build}-init_atmosphere/init_atmosphere_model')])==0
    assert receipt.main(['verify',str(native),'--expected-frames','2',
        '--standard-order','2','--standard-substeps','4',
        '--expected-executable',str(root/f'native/{build}-atmosphere/atmosphere_model'),
        '--expected-init',str(init)])==0
    path=tree/'tools/native_dry_reference/compare_dry_identity.py'
    spec=importlib.util.spec_from_file_location('native_dry_check',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report=tmp_path/'cpu.json'
    assert mod.main(['--tree',str(tree),'--init',str(init),'--native-dir',str(native),
                    '--order','2','--substeps','4','--backend','cpu','--steps','1','--json',str(report)])==0
    import json
    fields=json.loads(report.read_text())['cpu_one_step_from_each_native_output']['steps'][0]['fields']
    recipe=build.split('-',1)[1]
    assert assess(fields,identity=recipe in STRICT_IDENTITY_RECIPES)['passed']
