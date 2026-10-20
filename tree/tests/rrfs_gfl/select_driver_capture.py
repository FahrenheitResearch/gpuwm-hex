"""Select complete driver calls from a sealed trace without decoding any word.

All PRE/POST arguments and saved state of each selected call are retained.
Nested call traces remain in the parent artifact; the kernel replays the whole
subsystem at the driver boundary rather than its internal call boundaries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    for name in ('dump','output','inventory','pins','words','oracle'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.dump/'manifest.json').read_text())
    captures={prefix:info for prefix,info in manifest['captures'].items()
              if info['site']=='cu_grell_freitas_li__driver_convection__L670'}
    if not captures:raise ValueError('no complete GFL driver call in sealed trace')
    args.output.mkdir(parents=True)
    for name,info in manifest['files'].items():
        if name.split('.pre.')[0].split('.post.')[0] not in captures:continue
        source=args.dump/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('parent capture changed: '+name)
        shutil.copyfile(source,args.output/name)
    inventory=json.loads(args.inventory.read_text())
    inventory['input_provenance']={'parent_manifest_sha256':hashlib.sha256((args.dump/'manifest.json').read_bytes()).hexdigest(),
        'parent_dump_root_sha256':manifest['dump_root_sha256'],'parent_capture_count':len(manifest['captures']),
        'selected_driver_captures':sorted(captures),
        'scope':'all arguments and saved state at whole GFL driver boundary'}
    selected_inventory=args.output/'inventory.json'
    selected_inventory.write_text(json.dumps(inventory,indent=2)+'\n')
    subprocess.run([sys.executable,str(args.oracle),'--seal','--ref','rrfs',
        '--scheme','cu_grell_freitas_li','--dump',str(args.output),'--inventory',str(selected_inventory),
        '--pins',str(args.pins),'--words',str(args.words),'--recipe',manifest['recipe'],
        '--process-count',str(manifest['process_count'])],check=True)


if __name__=='__main__':main()
