#!/usr/bin/env python3
"""Seal source-bound field metadata beside immutable native references.

Rust reads NetCDF metadata; Python selects source declarations and schedules
publication. Native arrays and existing publications are never rewritten.
"""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from publish_cache import sha, publish


REGISTRY = 'src/core_atmosphere/Registry.xml'


def declarations(source):
    fields = {}
    root = ET.parse(source/REGISTRY).getroot()
    for struct in root.findall('.//var_struct'):
        for field in struct.findall('var'):
            fields[field.get('name')] = dict(field.attrib, structure=struct.get('name'))
        for array in struct.findall('var_array'):
            for field in array.findall('var'):
                fields[field.get('name')] = dict(array.attrib, **field.attrib,
                    structure=struct.get('name'), array=array.get('name'))
    return fields


def create(directory, source, comparator, samples=None):
    manifest = json.loads((directory/'MANIFEST.json').read_text())
    publication = json.loads((directory/'PUBLICATION.json').read_text())
    if publication['files']['MANIFEST.json'] != sha(directory/'MANIFEST.json'):
        raise ValueError('native field inventory has a stale publication')
    if publication['files'].get('streams.atmosphere') != sha(directory/'streams.atmosphere'):
        raise ValueError('native inventory streams differ from sealed publication')
    declared = declarations(source)
    streams = ET.parse(directory/'streams.atmosphere').getroot()
    output_streams = {s.get('name') for s in streams if 'output' in s.get('type', '')}
    # Native Registry state is the atmospheric prognostic authority. The native
    # output views rho/theta are computed in mpas_atm_core.F:956-960.
    prognostic = {'u', 'w', 'rho_zz', 'rho', 'theta_m', 'theta'}
    prognostic.update(name for name, item in declared.items()
                      if item.get('structure') == 'state' and item.get('array') == 'scalars')
    fields = {}; examples = {}
    for name, digest in manifest['outputs'].items():
        stream = name.split('.', 1)[0]
        if stream not in output_streams:
            raise ValueError('native output has no declared stream: '+name)
        if stream in examples:
            continue
        path = (samples or directory)/name
        if sha(path) != digest or publication['files'].get(name) != digest:
            raise ValueError('native inventory sample hash differs: '+name)
        inventory = json.loads(subprocess.check_output([str(comparator), '--inventory', str(path)], text=True))
        rows = []
        for item in inventory['variables']:
            definition = declared.get(item['field'])
            rows.append(dict(item, native_declaration=definition))
        fields[stream] = rows
        examples[stream] = {'file': name, 'sha256': digest, 'stored_xtime': inventory['xtime']}
    if not fields:
        raise ValueError('native inventory has no actual output samples')
    return {'schema': 'mpas.native-source-field-inventory/v1',
            'native_directory': str(directory), 'native_manifest_sha256': sha(directory/'MANIFEST.json'),
            'native_publication_sha256': sha(directory/'PUBLICATION.json'),
            'ranks': manifest['ranks'], 'runtime': manifest['runtime'],
            'source_sha256': {REGISTRY: sha(source/REGISTRY)}, 'reader_sha256': sha(comparator),
            'examples': examples, 'streams': fields,
            'prognostic_fields': {s: sorted(row['field'] for row in rows if row['field'] in prognostic)
                                  for s, rows in fields.items()},
            'prognostic_scope': 'Registry.xml:1683-1748 atmospheric state/scalars, with native rho/theta output views; physics storage declarations retained separately',
            'water_bindings': {s: {row['field']: row['native_declaration'] for row in rows
                if row['native_declaration'] and (row['field'] in ('qv','qc','qr','qi','qs','qg','rho','rho_zz','zgrid','smois','sh2o','snow','canwat','rainc','rainnc','sfcrunoff','udrunoff','qfx','acsnow','acsnom','precipw'))}
                for s, rows in fields.items()},
            'water_integral_complete': False,
            'water_integral_limitation': 'Pinned streams do not store complete per-cell column+soil+snow+canopy mass and cumulative evaporation in kg m^-2; raw units/fields are supplied, no incomplete total advertised',
            'identity_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('directory', 'source', 'comparator', 'work', 'publish', 'ready'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--samples', type=Path,
                        help='restored native samples; each must match the sealed original output hash')
    args = parser.parse_args()
    result = create(args.directory, args.source, args.comparator, args.samples)
    args.work.mkdir(parents=True, exist_ok=False)
    (args.work/'MANIFEST.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(publish(args.work, args.publish, ['MANIFEST.json'], args.ready,
                            'Source-bound native stream/prognostic/water field inventory; no synthesized integrals')))


if __name__ == '__main__':
    main()
