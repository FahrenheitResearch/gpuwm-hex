"""Command orchestration for generated UGWP science and seekable time metadata."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(cfg['out'])
out.mkdir(parents=True,exist_ok=False)
science=out/'science.nc';stamped=out/'generated.nc'
commands=[
 [cfg['engine'],'--grid',cfg['grid'],'--geog',cfg['geog'],'--out',str(science),'--receipt',str(out/'generation.json')],
 [cfg['engine'],'--compare',cfg['native'],str(science),'--receipt',str(out/'science-comparison.json')],
 [cfg['metadata_engine'],str(science),str(stamped),cfg['valid_time']],
 [cfg['engine'],'--compare',str(science),str(stamped),'--receipt',str(out/'metadata-science-comparison.json')],
]
for i,argv in enumerate(commands):
 with (out/f'command-{i}.log').open('w') as log:
  subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
for name in ('science-comparison','metadata-science-comparison'):
 result=json.loads((out/(name+'.json')).read_text())
 assert result['words']==cfg['words'] and result['unequal_words']==0
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
(out/'preparation.json').write_text(json.dumps({
 'schema':'i4-active-input/v1','passed':True,'run_host':socket.gethostname(),
 'native_process_count':cfg['native_process_count'],'generator_process_count':1,
 'words':cfg['words'],'unequal_words':0,'science_words_preserved_after_metadata':True,
 'valid_time':cfg['valid_time'],'generated_sha256':sha(stamped),
 'engine_sha256':sha(cfg['engine']),'metadata_engine_sha256':sha(cfg['metadata_engine']),
 'commands':commands,
},indent=2)+'\n')
science.unlink()
