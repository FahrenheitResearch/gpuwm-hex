"""Add full-wrapper synthetic captures with the MPAS manager's actual bounds.

Original r1 profiles remain byte-for-byte intact. No scientific module or
instrumentation is changed; only synthetic ABI storage gains its held level.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


PROFILES = ('gfl_moist_deep', 'gfl_mynn_shallow_off', 'gfl_stable', 'gfl_subsidence')


def prepare(root):
    make = (root/'synthetic.mk').read_text()
    objects = re.findall(r'native_[0-9a-f]+\.o', make.splitlines()[1])
    if len(objects) != 8:
        raise ValueError('full instrumented native GFL closure must contain eight objects')
    rules = ['include synthetic.mk']
    rows = []
    for profile in PROFILES:
        parent = root/f'capture_{profile}.F90'
        original = parent.read_text()
        level_names = re.findall(r'^\s*real\(real32\) :: (\w+)\(1:1,1:55,1:1(?:,1:1)?\)',
                                 original, flags=re.M)
        if len(level_names) != 34:
            raise ValueError((profile, 'unexpected native boundary arrays', len(level_names)))
        text = original.replace('1:1,1:55,1:1', '1:1,1:56,1:1')
        for dimension in ('kme', 'kde'):
            text, count = re.subn(rf'^(\s*){dimension}=55$', rf'\g<1>{dimension}=56',
                                 text, flags=re.M)
            if count != 1:
                raise ValueError((profile, dimension, count))
        # These allocation words are synthetic ABI sentinels, never native
        # automatic locals. The original 55 active levels are unchanged.
        sentinel = '\n'.join(f'  {name}(1,56,1{",1" if name == "chem3d_in" else ""})=73.25_real32'
                             for name in level_names)
        anchor = "  call oracle_begin('cu_grell_freitas_li__driver_convection__L670',oracle_event)"
        if text.count(anchor) != 1:
            raise ValueError((profile, 'missing original driver capture entry'))
        text = text.replace(anchor, sentinel+'\n'+anchor)
        text = ('! Synthetic bounds from pinned mpas_atmphys_manager.F:685-693.\n'+text)
        target = root/f'capture_{profile}_mpas_bounds.F90'
        if not target.exists() or target.read_text() != text:
            target.write_text(text)
        name = target.stem
        deps = ' '.join(objects)
        rules.extend((f'{name}: {target.name} Makefile mpas_bounds.mk {deps}',
                      f'\t$(FC) $(FLAGS) $< {deps} -o $@'))
        rows.append({'profile':profile+'_mpas_bounds',
                     'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
                     'source_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                     'native_bounds':{'kms':1,'kme':56,'kds':1,'kde':56,'kts':1,'kte':55},
                     'sentinel_arrays':level_names})
    content = '\n'.join(rules)+'\n'
    target = root/'mpas_bounds.mk'
    if not target.exists() or target.read_text() != content:
        target.write_text(content)
    (root/'mpas-bounds-profiles.json').write_text(json.dumps({
        'scope':'full-wrapper synthetic profiles; no real-mesh or forecast proof',
        'authority':'1d6e370e mpas_atmphys_manager.F:685-693', 'profiles':rows}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args().root)
