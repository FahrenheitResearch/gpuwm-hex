from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
p=argparse.ArgumentParser(description="Check actual f4 registration only; no numerical/CLI acceptance")
p.add_argument("f4_checkout",type=Path);p.add_argument("receipt",type=Path);args=p.parse_args()
f4=args.f4_checkout.resolve()/'tree/src'
sys.path.insert(0,str(f4))
from hexcore.mpasphys.driver_v842 import SlotRegistry,resolve_namelist
path=Path(__file__).resolve().parents[2]/'src/hexcore/mpasphys/sfclay_mm5_v842.py'
spec=importlib.util.spec_from_file_location('p9_draft_registry_binding',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def provider(call):raise RuntimeError('CPU registration check has no prepared numerical pools')
registry=SlotRegistry();m.register_hex_cuda(registry,provider)
rows=[]
for scheme in m.SCHEMES:
 config=resolve_namelist(f"&physics config_physics_suite='none', config_sfclayer_scheme='{scheme}' /\n&hex_mpasphys config_sfclayer_backend='hex-cuda' /")
 registry.require(config)
 rows.append(config.receipt())
r={'status':'PASS','scope':'Actual draft f4 resolver and SlotRegistry registration/require, no callback, numerical launch, real CLI or forecast','schemes':rows,'module_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'f4_driver_sha256':hashlib.sha256((f4/'hexcore/mpasphys/driver_v842.py').read_bytes()).hexdigest(),'f4_table_sha256':hashlib.sha256((f4/'hexcore/data/profiles/mpas-physics-suites.v1.json').read_bytes()).hexdigest()}
args.receipt.write_text(json.dumps(r,indent=2)+'\n');print('PASS draft resolver/registry, both MM5 schemes; real CLI unproven')
