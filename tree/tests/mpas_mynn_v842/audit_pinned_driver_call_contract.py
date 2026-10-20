"""Bind the complete pinned MYNN caller inventory and optional arguments.

This is source evidence for a coverage audit, not a branch-number mapping.
No native output or uncovered gcov edge is excluded by this receipt.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
from translate_module import parse_modules, split_top, statements


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def named_call(path, name):
    # MPAS's caller has no preprocessor directive within this call. For the
    # WRF adapter use the translator's pinned WRF_CHEM=0 preprocessing.
    if name == 'mynn_bl_driver':
        lines = path.read_text().splitlines()
        matches = [n for n, line in enumerate(lines)
                   if re.match(r'\s*call\s+'+name+r'\s*\(', line, re.I)]
        assert len(matches) == 1
        start = matches[0]
        fragments = []
        for line in lines[start:]:
            assert not line.lstrip().startswith('#')
            fragments.append(line.strip().removesuffix('&').lstrip('&').strip())
            if not line.rstrip().endswith('&'):
                break
        line, text = start+1, ' '.join(fragments).lower()
    else:
        matches = [(n, text) for n, text in statements(path)
                   if re.match(r'call\s+'+name+r'\s*\(', text)]
        assert len(matches) == 1
        line, text = matches[0]
    body = text[text.index('(')+1:]
    assert body.endswith(')')
    arguments = {}
    for item in split_top(body[:-1]):
        match = re.fullmatch(r'(\w+)\s*=\s*(.+)', item)
        assert match and match[1] not in arguments
        arguments[match[1]] = match[2]
    return dict(path=str(path), line=line, sha256=sha(path),
                arguments=arguments, logical_call=text)


def main(a):
    physics = a.native/'src/core_atmosphere/physics'
    wrapper = physics/'physics_wrf/module_bl_mynn.F'
    mmm = physics/'physics_mmm/bl_mynn.F90'
    mpas = physics/'mpas_atmphys_driver_pbl.F'
    inventory = subprocess.check_output(['rg', '-n', '-i',
        r'^\s*call\s+(mynn_bl_driver|bl_mynn_run)\b', str(physics)], text=True)
    rows = inventory.splitlines()
    assert len(rows) == 2
    for path in (wrapper, mpas):
        assert sum(row.startswith(str(path)+':') for row in rows) == 1
    _, routines = parse_modules([wrapper, mmm])
    calls = []
    for path, name in ((mpas, 'mynn_bl_driver'), (wrapper, 'bl_mynn_run')):
        call = named_call(path, name)
        routine = routines[name]
        assert set(call['arguments']).issubset(routine.arguments)
        optional = {name: routine.variables[name] for name in routine.arguments
                    if routine.variables[name].optional}
        missing = set(routine.arguments)-set(call['arguments'])
        assert missing.issubset(optional)
        call['callee'] = dict(name=name, source=routine.source,
                              lines=[routine.start, routine.end])
        call['optional_arguments'] = {
            name: dict(present=name in call['arguments'], declaration_line=v.line,
                       dimensions=v.dimensions, kind=v.kind)
            for name, v in optional.items()}
        calls.append(call)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(scope=__doc__, calls=calls,
        complete_native_caller_inventory=rows, preprocessor_WRF_CHEM=0,
        native_sources={str(p): sha(p) for p in (mpas, wrapper, mmm)},
        gcov_branches_classified=[], excluded_words=[], acceptance_proven=False),
        indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
