#!/usr/bin/env python3
"""Derive the condensate identity init from an existing qv identity init.

Same result as prepare_real_case.py --moisture condensate (the synthetic
species are a function of the file's own qv, heights and latitude), for a host
that holds the qv init but not the source analysis.
usage: condensate_from_qv_init.py QV_INIT OUT
"""
import shutil, sys
from pathlib import Path
from netCDF4 import Dataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_real_case import add_synthetic_condensate, digest
source, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists(): raise SystemExit('refusing to overwrite an existing prepared input')
shutil.copyfile(source, out)
with Dataset(out, 'r+') as dst:
    dst.set_auto_mask(False)
    add_synthetic_condensate(dst)
print('source_sha256', digest(source)); print('init_sha256', digest(out))
