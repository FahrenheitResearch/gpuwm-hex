#!/usr/bin/env python3
"""Publish a seek-coordinate companion using the pinned p14 Rust utility.

Native authority: Registry.xml:1203-1235 and mpas_stream_manager.F:3775-3780.
The native case7 static science file is preserved. Rust checks every original
science word while adding only xtime; Python schedules the tool and hashes.
"""
import argparse,datetime,json,subprocess
from pathlib import Path
from publish_cache import sha,publish


def prepare(case,out,stamp,tool,source):
    datetime.datetime.strptime(stamp,'%Y-%m-%d_%H:%M:%S')
    files=list(case.glob('*.ugwp_oro_data.nc'))
    if len(files)!=1:raise ValueError('exactly one native UGWP static file required')
    original=files[0];out.mkdir(parents=True,exist_ok=False)
    changed=out/original.name
    proof=json.loads(subprocess.check_output([str(tool.resolve()),str(original.resolve()),str(changed.resolve()),stamp],text=True))
    if proof.get('existing_fields')!=20 or proof.get('differing_words')!=0 or proof.get('added_coordinate')!='xtime':
        raise ValueError('Rust static coordinate tool did not witness unchanged science words')
    proof.update(schema='mpas.native-static-coordinate-copy/v1',source_publication_sha256=sha(case/'PUBLICATION.json'),
                 source_file_sha256=sha(original),result_sha256=sha(changed),tool_sha256=sha(tool),
                 tool_source_sha256=sha(source),valid_time=stamp,owner='p14-ugwp-oro Rust utility; invoked read-only by f0',
                 authority='Registry.xml:1203-1235; mpas_stream_manager.F:3775-3780')
    (out/'MANIFEST.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('case','out','rust-tool','rust-source','publish','ready'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--valid-time',required=True);a=p.parse_args()
    proof=prepare(a.case,a.out,a.valid_time,a.rust_tool,a.rust_source)
    publication=publish(a.out,a.publish,['*.nc','MANIFEST.json'],a.ready,
                        'UGWP static coordinate companion: unchanged twenty native science fields, added seekable run-start xtime; compose with the original init publication')
    print(json.dumps({'coordinate':proof,'publication':publication}))


if __name__=='__main__':main()
