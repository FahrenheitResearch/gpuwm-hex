"""Build and replay an isolated whole-module candidate; keep live C4 immutable."""
import hashlib, importlib.util, json, os, runpy, shutil, subprocess, sys
from pathlib import Path

root = Path.cwd()
base = root / 'hex/tree/tests/noah_v842_oracle'
source = root / 'hex/tree/src/hexcore/mpasphys/noah_v842.py'
out = root / 'n1out/snowpatch-candidate'
out.mkdir(parents=True, exist_ok=True)
text = source.read_text()
assert hashlib.sha256(text.encode()).hexdigest() == 'c4efdf509ba4c3d3c9ab2b4e942484465bdf33f3f50b937afb7c00c3fb27373c'
assert text.count('r_pow(SNCOVR,SNOEXP)') == 2
candidate = text.replace('r_pow(SNCOVR,SNOEXP)', 'MU(SNCOVR,SNCOVR)')
candidate = candidate.replace('T1 = AD(MU(TFREEZ,noah_max', '// GNU strict folds real PARAMETER SNOEXP=2 to one rounded multiply; module_sf_noahlsm.F:3282.\nT1 = AD(MU(TFREEZ,noah_max', 1)
candidate_source = out / 'noah_v842.py'
candidate_source.write_text(candidate)
spec = importlib.util.spec_from_file_location('hexcore.mpasphys.noah_v842', candidate_source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules[spec.name] = module
import hexcore.mpasphys
hexcore.mpasphys.noah_v842 = module
for name in ('build_host.py', 'helper_entry.cu'):
    shutil.copy2(base / name, out / name)
runpy.run_path(str(out / 'build_host.py'), run_name='__main__')
command = ['g++', '-O3', '-ffp-contract=off', '-fno-tree-vectorize', '-fno-unsafe-math-optimizations', '-fPIC', '-shared', str(out / 'host.cpp'), '-o', str(out / 'libnoah_host.so')]
subprocess.run(command, check=True)
raw_baseline = (out / 'host.cpp').read_text()
assert raw_baseline.count('MU(SNCOVR,SNCOVR)') == 2
raw_baseline = raw_baseline.replace('MU(SNCOVR,SNCOVR)', 'r_pow(SNCOVR,SNOEXP)')
assert raw_baseline.count('r_pow(SNCOVR,SNOEXP)') == 2
(out / 'baseline-host.cpp').write_text(raw_baseline)
baseline_command = ['g++', '-O3', '-ffp-contract=off', '-fno-tree-vectorize', '-fno-unsafe-math-optimizations', '-fno-builtin-powf', '-fPIC', '-shared', str(out / 'baseline-host.cpp'), '-o', str(out / 'libnoah_host_baseline_powf.so')]
subprocess.run(baseline_command, check=True)
rust = ['rustc', '--edition=2024', '-O', str(base / 'pack_dual_driver_input.rs'), '-o', str(out / 'pack_dual_driver_input')]
subprocess.run(rust, check=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
packet = dict(scope='whole-module isolated snowpatch compiler-fold candidate; no forecast acceptance', original_source_sha256=sha(source), candidate_source_sha256=sha(candidate_source), host_source_sha256=sha(out / 'host.cpp'), host_library_sha256=sha(out / 'libnoah_host.so'), baseline_powf_library_sha256=sha(out / 'libnoah_host_baseline_powf.so'), baseline_powf_scope='disable host compiler powf builtin so real exponent2 calls glibc, matching the original CUDA expression path rather than silently folding it', packer_source_sha256=sha(base / 'pack_dual_driver_input.rs'), packer_sha256=sha(out / 'pack_dual_driver_input'), commands=[command, baseline_command, rust], host=os.uname().nodename)
(out / 'build.json').write_text(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2), flush=True)
