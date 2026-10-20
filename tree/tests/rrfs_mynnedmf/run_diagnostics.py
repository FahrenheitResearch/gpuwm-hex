"""Run built whole wrappers and retain truthful build/coverage metadata."""
import argparse,hashlib,json,platform,subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(build,out):
    build=build.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    result={'scope':'standalone synthetic whole-wrapper diagnostics, not CUDA/forecast proof',
            'host':platform.node(),'process_count':1,'build':str(build),
            'fixture_sha256':sha(build/'run.F90'),
            'translation_sha256':sha(build/'translation.cpp'),
            'compiler':subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
            'makefile':(build/'Makefile').read_text(),'runs':{}}
    origin=build/'build-origin.json'
    if origin.exists():result['build_origin']=json.loads(origin.read_text())
    for name,target in [('native','strict'),('host-glibc','host'),('native-repeat','strict'),('native-snan-bounds','diagnostic'),('native-profiled','native')]:
        executable=build/target/'run';words=out/(name+'.words')
        with (out/(name+'.log')).open('w') as log:
            code=subprocess.run([str(executable),str(words)],stdout=log,stderr=subprocess.STDOUT).returncode
        result['runs'][name]={'binary_sha256':sha(executable),'exit_code':code,
                             'words_sha256':sha(words) if words.exists() else None,
                             'words':sum(1 for _ in words.open('rb')) if words.exists() else 0}
    command=['gcov','-b','-c','-j','--stdout','run-module_bl_mynnedmf.gcno','run-module_bl_mynnedmf_driver.gcno']
    cov=subprocess.run(command,cwd=build/'native',capture_output=True,text=True)
    (out/'gcov.json').write_text(cov.stdout);(out/'gcov.log').write_text(cov.stderr)
    result['coverage_exit_code']=cov.returncode
    (out/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'host':result['host'],'runs':{n:r['exit_code'] for n,r in result['runs'].items()},'coverage_exit_code':cov.returncode}))
    return int(any(r['exit_code'] for r in result['runs'].values()) or cov.returncode)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.build,a.out))
