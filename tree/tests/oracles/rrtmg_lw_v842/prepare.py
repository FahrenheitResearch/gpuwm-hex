"""Copy pinned sources into an owned standalone build directory.
MPAS: module_ra_rrtmg_lw.F:1-12535; mpas_atmphys_rrtmg_lwinit.F:1-1593.
Run compilation through the lane build slot on its assigned host.
"""
import argparse
from pathlib import Path
import shutil
p=argparse.ArgumentParser()
p.add_argument('--mpas',type=Path,required=True)
p.add_argument('--build',type=Path,required=True)
a=p.parse_args();here=Path(__file__).resolve().parent
if a.build.exists(): raise SystemExit('build directory already exists; preserve its results')
shutil.copytree(here,a.build)
root=a.mpas/'src'
for name in ('mpas_kind_types.F','mpas_constants.F'):
 shutil.copy2(root/'framework'/name,a.build/'src'/name)
phys=root/'core_atmosphere'/'physics'
for name in ('module_ra_rrtmg_lw.F','module_ra_rrtmg_vinterp.F'):
 shutil.copy2(phys/'physics_wrf'/name,a.build/'src'/name)
shutil.copy2(phys/'mpas_atmphys_rrtmg_lwinit.F',a.build/'src'/'mpas_atmphys_rrtmg_lwinit.F')
shutil.copy2(phys/'physics_wrf'/'files'/'RRTMG_LW_DATA',a.build/'RRTMG_LW_DATA')
