"""CPU-only CuPy cache preparation; CUDA module loading happens in forecast."""
import hashlib,json,os,sys
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'scratch/noah-corrected-f4/tree/src'))
from hexcore.cuda_backend.aot_build import inventory
from cupy.cuda import compiler
rows=inventory();out=root/'n1out/corrected-forecast-aot';out.mkdir(parents=True,exist_ok=True);cache=out/'cupy';cache.mkdir(exist_ok=True)
os.environ['CUPY_CACHE_DIR']=str(cache)
compiler._kernel_cache_backend=type(compiler._kernel_cache_backend)(str(cache))
original_module=compiler.function.Module;original_selector=compiler._get_arch_for_options_for_nvrtc
class BuildOnlyModule:
 def __init__(self):self.mapping={}
 def load(self,blob):
  assert isinstance(blob,bytes) and blob[:4]==b'\x7fELF'
 def _set_mapping(self,mapping):self.mapping=mapping
# Only the module-loading boundary is omitted during this CPU compiler job.
# The unmodified CuPy compile/cache entry computes its own keys and bytes.
# This class and explicit architecture selection are never used by a forecast.
compiler.function.Module=BuildOnlyModule
compiler._get_arch_for_options_for_nvrtc=lambda arch:('-arch=sm_120','cubin') if str(arch)=='120' else (_ for _ in ()).throw(ValueError(arch))
print('CPU compiling complete production KernelCache inventory:',len(rows),'translation units',flush=True)
try:
 for row in rows:
  compiler._compile_with_cache_cuda(row['source'],tuple(row['options']),'120')
  print('compiled',row['module_key'],flush=True)
finally:
 compiler.function.Module=original_module;compiler._get_arch_for_options_for_nvrtc=original_selector
metadata=[dict(environment=list(k),version_comments=v) for k,v in compiler._empty_file_preprocess_cache.items()]
result=dict(scope='CPU-only exact CuPy cache build; device loading omitted; no numerical proof',modules=len(rows),arch='120',cupy_version=__import__('cupy').__version__,nvrtc_version=compiler._get_nvrtc_version(),empty_preprocess_cache=metadata,artifacts=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(cache.glob('*.cubin'))])
(out/'build.json').write_text(json.dumps(result,indent=2)+'\n');print('CPU cache build completed',len(result['artifacts']),flush=True)

from hexcore import cuda_physics_prep_v841 as prep, cuda_physics_v841 as physics
from hexcore.mpasphys.math_v842 import bind_common_carrier_math
from hexcore.cuda_backend.compile_contract import NvrtcCompileCapture
options=('--std=c++17','--fmad=false','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true')
strict_sources={row['module_key']:bind_common_carrier_math(row['source']) if row['module_key']=='hexcore.cuda_physics_prep_v841' else row['source'] for row in rows}
strict_sources.update(coupling=physics._MPAS_V842_SOURCE,numbers=prep._MPAS_NUMBER_CARRIERS_V842_SOURCE)
compiled=[];selector=compiler._get_arch_for_options_for_nvrtc
compiler._get_arch_for_options_for_nvrtc=lambda arch:('-arch=sm_120','cubin') if str(arch)=='120' else (_ for _ in ()).throw(ValueError(arch))
try:
 for name,source in strict_sources.items():
  digest=hashlib.sha256(source.encode()).hexdigest()
  with NvrtcCompileCapture() as capture:
   binary,_=compiler.compile_using_nvrtc(source,options=options,arch='120',cache_in_memory=True)
  path=out/(name.replace('.','_')+'.cubin');path.write_bytes(binary)
  compiled.append(dict(entry=name,source_sha256=digest,requested_options=list(options),file=path.name,sha256=hashlib.sha256(binary).hexdigest(),effective_compile=capture.evidence(digest)))
finally:compiler._get_arch_for_options_for_nvrtc=selector
(out/'strict-manifest.json').write_text(json.dumps(dict(scope='CPU-only strict native carrier build; no numerical proof',arch='120',cupy_version=__import__('cupy').__version__,nvrtc_version=compiler._get_nvrtc_version(),artifacts=compiled),indent=2)+'\n')
print('strict CPU carrier builds completed',len(compiled),flush=True)
