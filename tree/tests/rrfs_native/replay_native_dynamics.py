"""Run every full native tendency case without a GPU; no port identity claim.

MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5150-6409 at 1d6e370e.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrfs_native.dynamics_fixture import dynamics_fixture, DYNAMICS_CASES
from test_initialization_rrfs_native import call


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    arguments=parser.parse_args()
    path=arguments.oracle.resolve()
    a,(n,*_)=dynamics_fixture(path)
    rows=[]
    for option,cam,visc,vertical,full in DYNAMICS_CASES:
        a.update(mixing_option=option,config_mix_full=int(full),
                 config_v_mom_eddy_visc2=vertical,config_v_theta_eddy_visc2=vertical*.7,
                 config_h_mom_eddy_visc2=.17,config_h_theta_eddy_visc2=.13,
                 config_h_mom_eddy_visc4=.011,config_h_theta_eddy_visc4=.013,
                 config_mpas_cam_coef=cam,config_visc4_2dsmag=visc)
        a['kdiff'][...]=0
        call(path,'atm_compute_dyn_tend_work',a)
        outputs={}
        for field in ('kdiff','tend_u_euler','tend_w_euler','tend_theta_euler','h_divergence','tend_u'):
            array=np.ascontiguousarray(a[field][:,:n])
            outputs[field]=dict(shape=list(array.shape),words=array.size,
                sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                nonfinite_indices_zero_based=np.argwhere(~np.isfinite(array)).tolist(),
                words_hex=[f'{int(word):08x}' for word in array.view('u4').ravel()])
        rows.append(dict(case=[option,cam,visc,vertical,full],fields=outputs))
    receipt=dict(scope='complete standalone native routine; not production binary or port comparison',
                 process_count=1,run_host=os.uname().nodename,
                 oracle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 oracle_source_sha256=hashlib.sha256(path.with_name('oracle.f90').read_bytes()).hexdigest(),
                 cases=rows)
    arguments.receipt.parent.mkdir(parents=True,exist_ok=True)
    arguments.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
