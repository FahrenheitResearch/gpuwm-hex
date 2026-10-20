#!/usr/bin/env python3
"""Put nearest real-source terrain on a native mesh/static template.

Use this output as GRID_FILE for init case 7 with static/met interpolation
off and vertical-grid generation on. Then prepare_real_case.py remaps the
common dry initial state onto that generated vertical grid.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from prepare_real_case import digest


def xyz(d):
    lat=np.asarray(d['latCell'][:]);lon=np.asarray(d['lonCell'][:])
    return np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','template','out','receipt'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('refusing to overwrite an existing input')
    shutil.copyfile(a.template,a.out)
    with Dataset(a.source) as src,Dataset(a.out,'r+') as dst:
        mapping=cKDTree(xyz(src)).query(xyz(dst))[1]
        terrain=np.asarray(src['zgrid'][:,0])[mapping]
        dst['ter'][:]=terrain
    record={'schema':'gpuwm-hex.common-terrain-template/v1','source_sha256':digest(a.source),
       'template_sha256':digest(a.template),'out_sha256':digest(a.out),
       'terrain_min_m':float(terrain.min()),'terrain_max_m':float(terrain.max()),
       'nonzero_terrain_cells':int(np.count_nonzero(terrain))}
    a.receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
