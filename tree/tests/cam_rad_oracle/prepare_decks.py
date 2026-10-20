"""Prepare CAM forecast proof inputs from f0's published G2/G3 decks.

This is a planned-input receipt, never a forecast/verification receipt.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re


def replace_physics(text, options):
    match=re.search(r'(?im)^\s*&physics\b',text)
    if not match:raise ValueError('published deck has no physics namelist')
    end=text.index('/',match.end())
    body=text[match.end():end]
    for key,value in options.items():
        pattern=re.compile(r'(?im)^\s*'+re.escape(key)+r'\s*=.*$')
        statement=f"\n {key} = '{value}'"
        body=pattern.sub(statement,body) if pattern.search(body) else body.rstrip()+statement+'\n '
    return text[:match.end()]+body+text[end:]


def prepare(publication, output, receipt):
    manifest=publication/'PUBLICATION.json'
    published=json.loads(manifest.read_text());records=[]
    for path in sorted(publication.rglob('namelist.atmosphere')):
        relative=path.relative_to(publication)
        selection=path.parent.name
        single=selection in ('radt_lw_scheme-cam_lw','radt_sw_scheme-cam_sw')
        suite=selection.startswith('suite-')
        if not (single or suite):continue
        original=path.read_text();streams=path.with_name('streams.atmosphere')
        for source in (path,streams):
            key=str(source.relative_to(publication))
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            if published['files'].get(key)!=digest:raise ValueError(f'published hash differs: {key}')
        variants=[('published',{})] if single else [
            ('swap-cam-lw',{'config_radt_lw_scheme':'cam_lw'}),
            ('swap-cam-sw',{'config_radt_sw_scheme':'cam_sw'}),
            ('swap-cam-both',{'config_radt_lw_scheme':'cam_lw','config_radt_sw_scheme':'cam_sw'})]
        # Registry.xml:2336-2338 default, plus absorption refresh each rad step.
        for variant,selectors in variants:
            for interval in ('06:00:00','00:30:00'):
                options=selectors|{'config_camrad_abs_update':interval}
                dest=output/relative.parent/(variant+'-abs-'+interval.replace(':',''))
                dest.mkdir(parents=True,exist_ok=True)
                nml=dest/'namelist.atmosphere';stream=dest/'streams.atmosphere'
                nml.write_text(replace_physics(original,options));stream.write_bytes(streams.read_bytes())
                records.append(dict(source=str(relative),variant=variant,options=options,
                    namelist=str(nml),namelist_sha256=hashlib.sha256(nml.read_bytes()).hexdigest(),
                    streams_sha256=hashlib.sha256(stream.read_bytes()).hexdigest(),
                    backend_plan={'radiation_lw':'hex-cuda when cam_lw',
                                  'radiation_sw':'hex-cuda when cam_sw','all_other_slots':'native-cpu'},
                    status='PLANNED: no forecast executed'))
    receipt.write_text(json.dumps(dict(scope='prepared input deck matrix only',
        publication_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        native_acceptance=False,records=records),indent=2)+'\n')
    print(f'CAM prepared forecast inputs={len(records)}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('publication','output','receipt'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.publication,a.output,a.receipt)
