"""Exercise registration against f4's published registry, without CUDA or mocks.

This checks the unlanded contract only, not real-door or numerical acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def check(seam_tree,receipt):
    # Read f4's implementation; never modify it or substitute a fake registry.
    sys.path.insert(0,str(seam_tree/'src'))
    from hexcore.mpasphys.driver_v842 import SlotRegistry,resolve_namelist,Backend
    path=Path(__file__).parents[2]/'src/hexcore/mpasphys/gf_mpas_v842.py'
    spec=importlib.util.spec_from_file_location('gf_lane_port',path)
    gf=importlib.util.module_from_spec(spec);spec.loader.exec_module(gf)
    registry=SlotRegistry();gf.register(registry)
    configurations=[]
    for deep,shallow in ((0,8),(16,-1),(1,1)):
        text=f"&physics\nconfig_physics_suite='none', config_convection_scheme='cu_grell_freitas', config_gfconv_closure_deep={deep}, config_gfconv_closure_shallow={shallow}\n/"
        config=resolve_namelist(text,backends={'convection':'hex-cuda'})
        assert config.selectors['convection']==gf.SCHEME
        assert config.backends['convection']==Backend.HEX_CUDA
        assert config.options['config_gfconv_closure_deep']==deep
        assert config.options['config_gfconv_closure_shallow']==shallow
        registry.require(config)
        configurations.append(config.receipt())
    driver=seam_tree/'src/hexcore/mpasphys/driver_v842.py'
    report={'result':'Real f4 registry accepts GF hex-cuda callback and resolves native selector/closure options','driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),'port_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'configurations':configurations,'scope':'Published unlanded f4 contract; no GPU, CLI or forecast proof'}
    receipt.write_text(json.dumps(report,indent=2)+'\n');print(report['result'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seam-tree',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();check(a.seam_tree,a.receipt)
