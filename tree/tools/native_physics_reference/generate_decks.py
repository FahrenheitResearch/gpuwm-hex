#!/usr/bin/env python3
"""Generate native suite and single-scheme decks from MPAS's own option tables.

Authority: src/core_atmosphere/Registry.xml:2351-2416 and
src/core_atmosphere/physics/mpas_atmphys_control.F:132-333.
Explicit runtime arguments keep dates and meshes out of generic defaults.
"""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

CONTROL = 'src/core_atmosphere/physics/mpas_atmphys_control.F'
REGISTRY = 'src/core_atmosphere/Registry.xml'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def options(source):
    # control.F validates more options than Registry lists (e.g. revised sfclay
    # and Thompson cloud fraction). Preserve their union and source provenance.
    registry = ET.parse(source / REGISTRY).getroot()
    text = (source / CONTROL).read_text().split(' end subroutine physics_namelist_check')[0]
    result = {}
    for opt in registry.iter('nml_option'):
        name = opt.get('name')
        if name.endswith('_scheme') and name.startswith('config_'):
            if opt.get('type') == 'logical':
                result[name] = ['.false.', '.true.']; continue
            values = set(re.findall(r"[`']([a-z][a-z0-9_]*)[`']", opt.get('possible_values', '')))
            values.update(re.findall(r"\b"+name+r"\s*(?:\.eq\.|==)\s*'([^']+)'", text, re.I))
            values.discard('suite'); result[name] = sorted({v.strip() for v in values})
    suites = {}
    for name in re.findall(r"trim\(config_physics_suite\) == '([^']+)'", text):
        block = text.split("trim(config_physics_suite) == '"+name+"'", 1)[1].split(' else',1)[0]
        suites[name] = dict(re.findall(r"(config_\w+_scheme)\s*=\s*'([^']+)'", block))
    return result, suites

def companions(option, value):
    # Table entries distinguish native control requirements from witnessed
    # driver activation/allocation dependencies; suite defaults stay unchanged.
    table=Path(__file__).with_name('deck_companions.v842.json')
    rows=json.loads(table.read_text())['entries']
    matches=[r['companions'] for r in rows if r['option']==option and r['value']==value]
    if len(matches)>1:raise ValueError('duplicate native companion selection')
    return dict(matches[0]) if matches else {}

def literal(value):
    if isinstance(value, bool):return '.true.' if value else '.false.'
    if isinstance(value, (int,float)):
        number=Decimal(str(value))
        if not number.is_finite():raise ValueError('finite namelist scalar required')
        return str(value)
    if value in ('.true.', '.false.'):
        return value
    if any(c in value for c in "'\n\r"):
        raise ValueError('unsafe namelist value')
    return "'"+value+"'"

def interval_seconds(value):
    match=re.fullmatch(r'(?:(\d+)_)?(\d+):(\d+):(\d+(?:\.\d+)?)',value)
    if not match:raise ValueError('explicit MPAS time interval required')
    days,hours,minutes,seconds=match.groups()
    return Decimal(days or 0)*86400+Decimal(hours)*3600+Decimal(minutes)*60+Decimal(seconds)

def validate_cadence(duration,interval,dt):
    step=Decimal(str(dt));stop=interval_seconds(duration);sample=interval_seconds(interval)
    if step<=0 or stop<=0 or sample<=0:raise ValueError('positive forecast clock intervals required')
    if sample%step or stop%step or stop%sample:
        raise ValueError('native timestep and output interval must divide the requested forecast clock exactly')

def generate(source, out, start, duration, interval, dt, mesh, init, streams, variants=None, stream_lists=None):
    validate_cadence(duration,interval,dt)
    choices, suites = options(source)
    registry={o.get('name'):o for o in ET.parse(source/REGISTRY).getroot().iter('nml_option')}
    if 'config_oml1d' in registry:choices['config_oml1d']=['.false.','.true.']
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    specs = [(f'suite-{s}', s, {}, {}) for s in suites]
    specs += [(f'{o.removeprefix("config_")}-{v.strip(".")}', 'none', {o: v}, companions(o, v))
              for o, values in choices.items() for v in values]
    variant_sha=None
    if variants:
        variant_sha=sha(variants)
        for variant in json.loads(variants.read_text())['variants']:
            for suite in variant['suites']:
                override=variant['override'];companion=variant.get('companions',{}) if suite=='none' else {}
                if any(key not in registry for key in (*override,*companion)):raise ValueError('unknown native variant option')
                specs.append((variant['name']+'-'+suite,suite,override,companion))
    required_lists=sorted({f.get('name') for f in ET.parse(streams).getroot().iter('file')})
    list_directory=stream_lists or streams.parent
    list_hashes={name:sha(list_directory/name) for name in required_lists}
    for name, suite, override, companion in specs:
        settings = {'config_physics_suite': suite, **override, **companion}
        # UGWP diagnostics are explicitly requested for the stream-enabled decks.
        body = ("&nhyd_model\n"
                f" config_start_time = {literal(start)}\n config_run_duration = {literal(duration)}\n config_dt = {dt}\n"
                " config_time_integration_order = 2\n config_number_of_sub_steps = 4\n config_dynamics_split_steps = 3\n/\n"
                f"&decomposition\n config_block_decomp_file_prefix = {literal(mesh+'.graph.info.part.')}\n/\n"
                "&io\n config_pio_num_iotasks = 0\n config_pio_stride = 1\n/\n"
                "&physics\n config_sst_update = false\n config_ugwp_diags = true\n")
        body += ''.join(f' {k} = {literal(v)}\n' for k,v in settings.items())+' /\n'
        deck = out / name; deck.mkdir(exist_ok=False)
        nml = deck / 'namelist.atmosphere'; nml.write_text(body)
        tree = ET.parse(streams)
        original_input = next(s.get('filename_template') for s in tree.getroot()
                              if s.get('name') == 'input')
        original_mesh = original_input.removesuffix('.init.nc')
        for stream in tree.getroot():
            sname = stream.get('name')
            filename = stream.get('filename_template', '')
            if filename.startswith(original_mesh+'.'):
                stream.set('filename_template', mesh+filename[len(original_mesh):])
            if sname == 'input':
                stream.set('filename_template', init)
            if sname == 'surface':
                stream.set('type', 'input;output')
                stream.set('input_interval', 'none')
            if 'output' in stream.get('type', ''):
                stream.set('output_interval', interval)
                stream.set('filename_template', sname+'.$Y-$M-$D_$h.$m.$s.nc')
                stream.set('filename_interval', 'output_interval')
        streamfile = deck / 'streams.atmosphere'; tree.write(streamfile, encoding='unicode')
        for list_name in required_lists:
            if Path(list_name).name!=list_name:raise ValueError('stream list must be a local basename')
            shutil.copy2(list_directory/list_name,deck/list_name)
        entries.append({'deck': name, 'suite': suite, 'override': override, 'companions': companion,
                        'resolved_suite': suites[suite], 'namelist_sha256': sha(nml), 'streams_sha256': sha(streamfile),
                        'stream_list_sha256':list_hashes})
    index = {'schema': 'mpas.native-physics-decks/v1', 'source_sha256': {p: sha(source/p) for p in (REGISTRY, CONTROL)},
             'runtime': dict(start=start, duration=duration, output_interval=interval, dt=dt, mesh=mesh, init=init),
             'option_choices': choices, 'decks': entries,
             'companion_table_sha256':sha(Path(__file__).with_name('deck_companions.v842.json'))}
    if variant_sha:index['variant_table_sha256']=variant_sha
    (out/'INDEX.json').write_text(json.dumps(index, indent=2)+'\n')
    return index

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--streams-template', type=Path, required=True)
    p.add_argument('--variants',type=Path)
    p.add_argument('--stream-list-directory',type=Path)
    for name in ('start', 'duration', 'output-interval', 'mesh', 'init'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--dt', type=float, required=True)
    a=p.parse_args(); r=generate(a.source,a.out,a.start,a.duration,a.output_interval,a.dt,a.mesh,a.init,a.streams_template,a.variants,a.stream_list_directory)
    print(json.dumps({'decks': len(r['decks']), 'options': r['option_choices']}))

if __name__ == '__main__':
    main()
