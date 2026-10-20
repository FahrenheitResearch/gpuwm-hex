#!/usr/bin/env python3
"""Subset published native initializations through the engine's existing Rust mesh door.

Explicit rows supply parent_case, parent_init, mesh, region, out and publish.
No Python field processing. The installed rw_mpas_mesh culler implements
MPAS-Limited-Area v2.2 conventions; its receipt records native wrap divergence.
This publication is provenance for native case9, not a new culler identity proof.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from publish_cache import publish, sha
from matrix_support import published, free_memory_bytes


def prepare(row, tool, ready):
    source=Path(row['parent_case'])
    parent=source/row['parent_init']
    publication=json.loads((source/'PUBLICATION.json').read_text())
    if publication['files'].get(parent.name)!=sha(parent):
        raise ValueError('native parent initialization hash differs')
    region=Path(row['region'])
    out=Path(row['out']);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(region,out/'region.json')
    receipt=out/'cull-receipt.json'
    graph=out/(row['mesh']+'.graph.info')
    initial=out/(row['mesh']+'.init.nc')
    with (out/'cull.log').open('w') as log:
        subprocess.run([str(tool),'--cull-parent',str(parent),'--region',str(region),
            '--out',str(initial),'--graph',str(graph),'--receipt',str(receipt)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads(receipt.read_text())
    # Stored zero connectivity must not wrap to an included last parent element;
    # cull.rs documents that precise native create_region divergence.
    if any(result['native_wrap_divergence']):
        raise ValueError('native create_region zero-index wrap would differ on this cut')
    if result['euler_v_minus_e_plus_f']!=1 or not result['mark']['ring_cell_counts'][0]:
        raise ValueError('regional cut has no interior or is not a connected disk')
    if result['output_sha256']!=sha(initial) or result['graph_sha256']!=sha(graph):
        raise ValueError('Rust regional output differs from cull receipt')
    (out/'MANIFEST.json').write_text(json.dumps({
        'schema':'mpas.native-regional-input/v1','row':row,
        'parent_publication_sha256':sha(source/'PUBLICATION.json'),
        'parent_init_sha256':sha(parent),'region_sha256':sha(region),
        'rust_tool':str(tool),'rust_tool_sha256':sha(tool),
        'culler_identity_tier':'existing Rust implementation; native tool not rerun here',
        'files':{p.name:sha(p) for p in (initial,graph,receipt)}},indent=2)+'\n')
    return publish(out,Path(row['publish']),['*.nc','*.graph.info','*.json','*.log'],ready,
                   row['description'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('index',type=Path)
    p.add_argument('--tool',type=Path,required=True)
    p.add_argument('--ready',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();tool=a.tool.resolve(strict=True)
    rows=json.loads(a.index.read_text())['regions']
    if not rows:raise ValueError('empty regional input index')
    pending=list(rows);results=[]
    while pending:
        progress=False
        for row in list(pending):
            cached=published(Path(row['publish']),Path(row['out']))
            if not cached:
                if not (Path(row['parent_case'])/'PUBLICATION.json').is_file():continue
                if free_memory_bytes()<16*1024**3:continue
                cached=prepare(row,tool,a.ready)
            results.append({'region':row,'publication':cached});pending.remove(row)
            a.receipt.write_text(json.dumps({'complete':not pending,'regions':results},indent=2)+'\n')
            print(json.dumps(results[-1]),flush=True);progress=True
        if not progress:time.sleep(20)


if __name__=='__main__':main()
