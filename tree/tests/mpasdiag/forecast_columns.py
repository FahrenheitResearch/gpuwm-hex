"""Select published same-time restart/history/diagnostic fields for replay.

The Rust word reader does all file decoding and merges the requested fields.
This script only specifies the native pool inputs and chooses artifacts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


def collect(forecast, destination, reader, times):
    from hexcore.mpasdiag.isobaric import FIELDS
    # Case identity is supplied by the publication, never a model default.
    publication = json.loads((forecast/'PUBLICATION.json').read_text())
    restarts = sorted(forecast.glob('restart.*.nc'))
    if not restarts:
        raise ValueError('publication has no restart output')
    for index in times:
        restart = restarts[index]
        timestamp = restart.name.removeprefix('restart.').removesuffix('.nc')
        output, diagnostics = [forecast/f'{kind}.{timestamp}.nc' for kind in ('output','diagnostics')]
        directory = destination/(forecast.name+'-'+timestamp)
        directory.mkdir(parents=True,exist_ok=False)
        requests = [dict(path=str(restart),fields=['theta_m','qv','w','pressure_p','pressure_base','exner','relhum']),
                    dict(path=str(output),fields=['uReconstructZonal','uReconstructMeridional','vorticity']),
                    dict(path=str(diagnostics),fields=[*FIELDS,'t_iso_levels','z_iso_levels'])]
        request = directory/'request.json'
        request.write_text(json.dumps(requests,indent=2)+'\n')
        subprocess.run([str(reader.resolve()),'--request',str(request),str(directory/'columns.json')],check=True)
        subprocess.run([str(reader.resolve()),str(restart),str(directory/'mesh.json'),
                        'zgrid','cellsOnVertex','kiteAreasOnVertex','areaTriangle'],check=True)
        run = dict(host='published f0 strict forecast',case=str(forecast),
                   native_sha256='a71feef594ae020f86aa9450608e21b8e2074415479de521438a7701e5b1d479',
                   case_publication_sha256=hashlib.sha256((forecast/'PUBLICATION.json').read_bytes()).hexdigest(),
                   publication=publication,requests=requests,
                   artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (restart,output,diagnostics)})
        (directory/'run.json').write_text(json.dumps(run,indent=2)+'\n')
        print(directory,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('forecast',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--reader',type=Path,required=True);p.add_argument('--output-indices',type=int,nargs='+',required=True)
    a=p.parse_args();collect(a.forecast,a.destination,a.reader,a.output_indices)
