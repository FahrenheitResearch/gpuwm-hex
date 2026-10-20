"""Prepare lane-owned native/CUDA sea-ice proof input pairs from f0 decks.

No simulation or weather-field processing. This does not establish acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def prepare(source, table, output, receipt):
    output.mkdir(parents=True,exist_ok=False)
    slots=json.loads(table.read_text())['mpasphys_v842']['slots']
    keys={n:r['backend_selector'] for n,r in slots.items()}
    assert 'seaice' in keys
    choices={'lsm_scheme-sf_noah','lsm_scheme-sf_noahmp',
             'suite-mesoscale_reference','suite-convection_permitting'}
    rows=[]
    for name in sorted((source/'G2').glob('*/*/*/namelist.atmosphere')):
        if name.parent.name not in choices:continue
        streams=name.with_name('streams.atmosphere').read_text()
        cadence={stream.attrib['name']:stream.attrib['output_interval']
                 for stream in ET.fromstring(streams)
                 if 'output' in stream.attrib.get('type','').split(';')}
        assert cadence, 'native deck has no output streams'
        native=name.read_text()
        assert 'config_frac_seaice' not in native
        for frac in (False,True):
            dest=output/name.parent.relative_to(source/'G2')/('frac-true' if frac else 'frac-false')
            dest.mkdir(parents=True)
            text=native.replace('&physics\n','&physics\n config_frac_seaice = '+('.true.' if frac else '.false.')+'\n')
            assert text!=native
            cuda=text+'&hex_mpasphys\n'+''.join(
                f" {key} = '{'hex-cuda' if slot=='seaice' else 'native-cpu'}'\n" for slot,key in keys.items())+'/\n'
            for filename,value in [('namelist.atmosphere',text),('namelist.hex.atmosphere',cuda),('streams.atmosphere',streams)]:
                (dest/filename).write_text(value)
            rows.append({'deck':str(dest),'native_source':str(name.parent),'fractional':frac,
                'native_stream_cadence':cadence,
                'namelist_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'streams_sha256':hashlib.sha256(streams.encode()).hexdigest(),
                'hex_namelist_sha256':hashlib.sha256(cuda.encode()).hexdigest()})
    assert rows,'no relevant G2 decks'
    receipt.write_text(json.dumps({'kind':'prepared inputs; no execution or sea-ice coverage claimed',
        'publication_sha256':hashlib.sha256((source/'PUBLICATION.json').read_bytes()).hexdigest(),
        'decks':rows},indent=2)+'\n')
    print(len(rows),'native/CUDA deck pairs prepared')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('table',type=Path)
    p.add_argument('output',type=Path);p.add_argument('receipt',type=Path)
    a=p.parse_args();prepare(a.source,a.table,a.output,a.receipt)
