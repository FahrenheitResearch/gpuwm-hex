"""Diagnostic invocation of pristine native YSU on artificial undefined inputs.

Run each call in an isolated process with an external timeout. These are
standalone diagnostics, never evidence that f0's published model does the same.
MPAS v8.4.2 bl_ysu.F90:845-847,1625-1686; module_bl_ysu.F:450-461.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from oracle import Oracle, fixtures, OUTPUTS

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--oracle', type=Path, required=True)
p.add_argument('--fixtures', type=Path, required=True)
p.add_argument('--case', choices=('shallow', 'cloud-top', 'inactive-qc', 'inactive-qi'), required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
_, arrays, flags = next(fixtures(a.fixtures))
wrapper = a.case.startswith('inactive')
if a.case == 'shallow':
    arrays['dz8w2d'].fill(1)
elif a.case == 'cloud-top':
    arrays['tx'] = np.float32(300)*arrays['pi2d']
    arrays['qvx'].fill(.008)
    arrays['qcx'].fill(.002)
    arrays['qix'].fill(0)
    arrays['hfx'].fill(250)
    arrays['br'].fill(-.35)
    flags['ysu_topdown_pblmix'] = 1
else:
    arrays['ctopo'] = np.ones(1, np.float32)
    arrays['ctopo2'] = np.ones(1, np.float32)
    arrays['uox'] = np.zeros(1, np.float32)
    arrays['vox'] = np.zeros(1, np.float32)
    flags['f_qc' if a.case == 'inactive-qc' else 'f_qi'] = 0
out, status = Oracle(a.oracle).run(arrays, flags, native=True, wrapper=wrapper)
record = {'schema':'p11.native-undefined-input-diagnostic/v1', 'case':a.case,
          'processes':1, 'scope':'standalone diagnostic, not published strict model',
          'status':status, 'words':{name:arr.view(np.uint32).ravel().tolist()
                                  for name in OUTPUTS if (arr:=out[name]) is not None}}
a.out.write_text(json.dumps(record,indent=2)+'\n')
