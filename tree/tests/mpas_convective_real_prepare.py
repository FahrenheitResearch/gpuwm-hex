"""Orchestrate typed Rust extraction from sealed f0 native forecast files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from hexcore.mpasdiag.convective_v842 import FIELDS


def prepare(reader, directory, forecasts):
    groups = {}
    provenance = []
    directory.mkdir(parents=True,exist_ok=True)
    for group,forecast in enumerate(forecasts):
        publication = forecast/'PUBLICATION.json'
        manifest = json.loads(publication.read_text())
        hashes = manifest['files']
        paths = []
        for restart in sorted(forecast.glob('restart.*.nc')):
            stamp = restart.name[len('restart.'):-3]
            output = forecast/('output.'+stamp+'.nc')
            diagnostic = forecast/('diagnostics.'+stamp+'.nc')
            if not output.exists() or not diagnostic.exists():
                continue
            destination = directory/(str(group)+'_'+stamp)
            arrays = []
            for name in ('zgrid','w','theta_m','qv','exner','pressure_base','pressure_p',
                         'areaCell','cellsOnVertex','kiteAreasOnVertex',
                         'uReconstructZonal','uReconstructMeridional','vorticity'):
                source = output if name in ('uReconstructZonal','uReconstructMeridional','vorticity') else restart
                arrays.append(dict(name=name,source=str(source),source_sha256=hashes[source.name],
                                   variable=name,dtype='<i4' if name=='cellsOnVertex' else '<f4'))
            native_fields = FIELDS if manifest.get('steps') == 1 else FIELDS[:14]
            for name in native_fields:
                if name not in ('lcl','lfc'):
                    arrays.append(dict(name='native_'+name,source=str(diagnostic),
                                       source_sha256=hashes[diagnostic.name],variable=name,dtype='<f4'))
            request = directory/(destination.name+'.request.json')
            request.write_text(json.dumps(dict(output=str(destination),arrays=arrays),indent=2)+'\n')
            seal = destination/'arrays.json'
            if seal.exists():
                stored = json.loads(seal.read_text())['arrays']
                expected = {a['name']:(a['source_sha256'],a['variable'],a['dtype']) for a in arrays}
                observed = {a['name']:(a['source_sha256'],a['variable'],a['dtype']) for a in stored}
                if observed != expected:
                    raise ValueError('existing bridge belongs to different source words')
                for a in stored:
                    if hashlib.sha256((destination/(a['name']+'.npy')).read_bytes()).hexdigest()!=a['npy_sha256']:
                        raise ValueError('existing replay bridge bytes changed')
            else:
                subprocess.run([str(reader),str(request)],check=True)
            paths.append(destination.name)
        if not paths:
            raise ValueError(f'no matching sealed restart/output/diagnostics frames in {forecast}')
        groups[forecast.name]=paths
        provenance.append(dict(forecast=str(forecast),publication_sha256=hashlib.sha256(publication.read_bytes()).hexdigest(),
                               frames=len(paths)))
    (directory/'groups.json').write_text(json.dumps(groups,indent=2)+'\n')
    (directory/'provenance.json').write_text(json.dumps(dict(reader=str(reader),reader_sha256=hashlib.sha256(reader.read_bytes()).hexdigest(),
                                                          forecasts=provenance),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('forecasts',type=Path,nargs='+')
    a=p.parse_args()
    prepare(a.reader.resolve(),a.directory.resolve(),[f.resolve() for f in a.forecasts])
