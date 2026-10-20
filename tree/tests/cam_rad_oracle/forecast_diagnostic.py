"""Measure complete CAM forecasts through the real CLI before table reconcile.

Only this process's callback lookup gains the two requested descriptors. The
native slot preparation, committed CAM backend, model door and output writer
execute unchanged. This diagnostic cannot claim a published seam or landing.
"""
import hashlib
from importlib import resources
import json
import sys


def main(argv=None):
    from hexcore import physics_suites
    from hexcore.mpasphys import device_pools_v842 as pools
    from hexcore.cli import main as door
    source=resources.files('hexcore').joinpath(physics_suites.TABLE_PATH)
    before=source.read_bytes()
    table=json.loads(before)
    added={}
    for slot,scheme in (('radiation_lw','cam_lw'),('radiation_sw','cam_sw')):
        entries=table['mpasphys_v842']['slots'][slot].setdefault('cuda_entries',{})
        descriptor=dict(module='cam_rad_v842',call='run_slot',bind='bind_native_pools')
        if scheme in entries:
            assert entries[scheme]==descriptor,'Owner descriptor differs from this diagnostic'
        else:
            entries[scheme]=descriptor;added[slot]={scheme:descriptor}
    selected=physics_suites.validate_table(table)
    original=pools.load_table
    pools.load_table=lambda:selected
    print('CAM_PROCESS_LOCAL_DIAGNOSTIC',json.dumps(dict(
        production_hook=False,whole_model_acceptance=False,
        original_table_sha256=hashlib.sha256(before).hexdigest(),added=added)),flush=True)
    try:
        return door(sys.argv[1:] if argv is None else argv)
    finally:
        pools.load_table=original
        assert source.read_bytes()==before,'Protected suite table changed during diagnostic'


if __name__=='__main__':
    raise SystemExit(main())
