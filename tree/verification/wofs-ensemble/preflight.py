"""CPU binary and case closure before any GPU lease."""
from pathlib import Path
import hashlib,json,os,subprocess
from gpuwm import rustwx
from score import scores
from products import window_frames
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble')); R=W/'prepared'

def main():
    observed=json.loads((R/'observations/nowcast.json').read_text())
    import datetime as dt
    case=json.loads(Path(__file__).with_name('case.json').read_text())
    frames=window_frames(observed,dt.datetime.strptime(case['start_time'],'%Y-%m-%d_%H:%M:%S'),
                         case['cycle_hours'])
    for frame in frames:
        with (R/'observations'/frame['file']).open('rb') as f:
            sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=frame['sha256']: raise ValueError('observation bytes changed')
    renderer=Path(os.environ['GPUWM_RW_WRFBATCH'])
    rustwx.require_ensemble_diagnostic_reducer(renderer)
    tools={}
    for key in ('GPUWM_RW_WRFBATCH','GPUWM_RW_VERIFY','GPUWM_HEX_RW_MPAS_CONVERT','GPUWM_HEX_RW_MPAS_INIT'):
        path=Path(os.environ[key])
        with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        result=subprocess.run([str(path),'--abi'],capture_output=True,text=True)
        tools[key]=dict(path=str(path),sha256=sha,abi=result.stdout.strip(),returncode=result.returncode)
    # Spool coordinates and aggregate render sources use the native writer
    # library, independently of the CLI binaries checked above.
    from gpuwm.io import nc_writer_bridge
    writer_path=nc_writer_bridge.resolve_ncwrite_bridge()
    writer=nc_writer_bridge.load()  # Its loader verifies the actual ABI.
    with writer_path.open('rb') as f: writer_sha=hashlib.file_digest(f,'sha256').hexdigest()
    tools['GPUWM_NCWRITE_BRIDGE']=dict(path=str(writer_path),sha256=writer_sha,
        abi=int(writer.gpuwm_ncwrite_abi_version()),returncode=0)
    engine=subprocess.check_output(['git','-C',os.environ['E'],'rev-parse','HEAD'],text=True).strip()
    from hexcore.cuda_backend.arch_admission import admitted_architecture,nvrtc_target_archs
    targets=nvrtc_target_archs()
    compatibility={}
    for label,compute in [('H100',(9,0)),('B200',(10,0)),('RTX PRO 6000',(12,0))]:
        admission=admitted_architecture(compute,targets=targets)
        compatibility[label]=None if admission is None else admission.as_dict()
    if any(row['returncode'] for row in tools.values()): raise ValueError('native binary ABI failed')
    (W/'evidence/cpu-preflight.json').write_text(json.dumps(dict(observed_frames=len(frames),
        observations_sha256=hashlib.sha256((R/'observations/nowcast.json').read_bytes()).hexdigest(),
        engine_commit=engine,tools=tools,nvrtc_targets=targets,
        hypothetical_architecture_admission=compatibility,
        compatibility_scope='CPU toolchain query only; no H100 or B200 execution',ready=True),indent=2)+'\n')

if __name__=='__main__': main()
