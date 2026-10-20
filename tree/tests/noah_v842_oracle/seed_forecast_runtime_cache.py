"""Transport authenticated CPU images to the real forecast's cache paths."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--noah-manifest',type=Path);a=p.parse_args()
root=a.root.resolve();out=a.output.resolve();assert out.is_relative_to(root)
def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
records=[]
for name in ['corrected-forecast-aot','corrected-cupy-cache','actual-cupy-runtime-inventory']:
 manifest=root/'n1out'/name/'build.json';packet=json.loads(manifest.read_text())
 for image in packet['artifacts']:
  source=manifest.parent/'cupy'/image['file'];assert digest(source)==image['sha256']
  for directory in [root/'cache/cupy',out/'scratch/cuda-cache',out/'scratch/cuda-cache/mpasphys']:
   directory.mkdir(parents=True,exist_ok=True);target=directory/image['file']
   if target.exists():assert digest(target)==image['sha256']
   else:
    temporary=target.with_suffix(f'.{os.getpid()}.tmp');shutil.copy2(source,temporary);temporary.replace(target)
   records.append(dict(target=str(target),sha256=image['sha256'],cpu_manifest_sha256=digest(manifest)))
manifest=a.noah_manifest or root/'receipts/cuda-cubins-foundation-sm120.json';packet=json.loads(manifest.read_text())
for image in packet['artifacts']:
 source=manifest.parent/image['file']
 if not source.is_file():source=root/'cache/cupy/hex-noah'/image['file']
 assert digest(source)==image['sha256']
 directory=out/'scratch/cuda-cache/mpasphys/hex-noah';directory.mkdir(parents=True,exist_ok=True);target=directory/image['file']
 if target.exists():assert digest(target)==image['sha256']
 else:
  temporary=target.with_suffix(f'.{os.getpid()}.tmp');shutil.copy2(source,temporary);temporary.replace(target)
 records.append(dict(target=str(target),sha256=image['sha256'],cpu_manifest_sha256=digest(manifest)))
a.receipt.parent.mkdir(parents=True,exist_ok=True)
a.receipt.write_text(json.dumps(dict(scope='authenticated CPU image transport only; no compilation or numerical proof',output=str(out),images=records),indent=2)+'\n')
print('Authenticated runtime cache seeded',out,len(records),flush=True)
