"""Prepare a complete host CAM storage experiment, without changing arithmetic.

The observed strict GNU allocation order is alphabetical for automatic arrays;
this experiment uses glibc malloc/free for runtime-bound arrays and reuses the
capture argument buffers. It is a diagnostic, not a native-defect reproduction
claim. Compare every boundary and persistent-state word before drawing one.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import translate as tr


def prepare(source, witness, output):
    output.mkdir(parents=True, exist_ok=False)
    original_parse = tr.parse
    routines = None

    def ordered_parse(paths):
        nonlocal routines
        module, routines = original_parse(paths)
        for routine in routines.values():
            variables = routine.variables
            order = [name for name, v in variables.items() if not v.dims]
            order += sorted(name for name, v in variables.items() if v.dims)
            routine.variables = {name: variables[name] for name in order}
        return module, routines

    tr.parse = ordered_parse
    try:
        core, _, module, _ = tr.generate([source / 'module_ra_cam_support.F', source / 'module_ra_cam.F'])
    finally:
        tr.parse = original_parse
    markers = list(re.finditer(r'// MPAS \S+:\d+-\d+\nCAM_ROUTINE \w+ (\w+)\(', core))
    sections = []
    for index, match in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(core)
        section = core[match.start():end]
        routine = routines[match[1]]
        symbols = module | routine.variables
        for v in routine.variables.values():
            if not v.dims or v.name in routine.args:
                continue
            names = {token for dimension in v.dims for token in re.findall(r'\b[a-z]\w*\b', dimension)}
            dynamic = any(name in symbols and not symbols[name].const for name in names)
            if dynamic:
                old = 'auto ' + v.name + '=arena.array<'
                assert section.count(old) == 1, (routine.name, v.name)
                section = section.replace(old, 'auto ' + v.name + '=arena.native_array<')
        sections.append(section)
    core = core[:markers[0].start()] + ''.join(sections)
    abi = (Path(__file__).parent / 'array_abi.cuh').read_text()
    abi = abi.replace('#define CAM_ROUTINE\n', '#define CAM_ROUTINE __attribute__((noinline))\n')
    abi = abi.replace('char* data;size_t used=0,capacity,peak=0;',
                      'char* data;size_t used=0,capacity,peak=0;\n'
                      'void* allocations[8192]{};size_t active=0;')
    old = '  template<class T,class... B> CAM_HD CamArray<T> array(B... b) {'
    new = '''  template<class T,class... B> CamArray<T> native_array(B... b) {
    auto r=CamArray<T>::contiguous(nullptr,b...);
    CAM_ASSERT(active<8192 && r.count>=0);
    r.data=static_cast<T*>(malloc(r.count*sizeof(T)));
    CAM_ASSERT(r.data!=nullptr);
    allocations[active++]=r.data;return r;
  }
''' + old
    assert abi.count(old) == 1
    abi = abi.replace(old, new)
    abi = abi.replace('CamArena& arena;size_t mark;', 'CamArena& arena;size_t mark,native_mark;')
    abi = abi.replace('mark(a.used) {}', 'mark(a.used),native_mark(a.active) {}')
    abi = abi.replace('arena.used=mark;',
                      'while(arena.active>native_mark) free(arena.allocations[--arena.active]);arena.used=mark;')
    adapter = witness.read_text()
    adapter = adapter.replace('std::vector<std::unique_ptr<char[]>> storage;',
                              'std::vector<std::unique_ptr<char[]>> storage;\n'
                              'std::vector<size_t> capacities;size_t cursor=0;')
    old = '''  auto p=std::make_unique<char[]>(a.count*sizeof(T));
  a.data=reinterpret_cast<T*>(p.get());storage.push_back(std::move(p));'''
    new = '''  size_t bytes=a.count*sizeof(T);
  if(cursor==storage.size()) {storage.push_back(std::make_unique<char[]>(bytes));capacities.push_back(bytes);}
  if(capacities[cursor]<bytes) {storage[cursor]=std::make_unique<char[]>(bytes);capacities[cursor]=bytes;}
  a.data=reinterpret_cast<T*>(storage[cursor++].get());'''
    assert adapter.count(old) == 1
    adapter = adapter.replace(old, new)
    adapter = adapter.replace('while(!inputs.eof()) {', 'while(!inputs.eof()) {inputs.cursor=0;')
    assert adapter.count('inputs.storage.clear();') == 1
    adapter = adapter.replace('inputs.storage.clear();', '')
    files = {'cam_core.cuh': core, 'array_abi.cuh': abi, 'host_witness.cpp': adapter}
    for name, content in files.items():
        (output / name).write_text(content)
    (output / 'Makefile').write_text('CXX = g++\nall: host_witness\n'
        'host_witness: host_witness.cpp cam_core.cuh array_abi.cuh\n'
        '\t$(CXX) -std=c++17 -O3 -g -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $< -o $@\n')
    (output / 'storage-experiment.json').write_text(json.dumps(dict(
        scope='whole CAM glibc automatic-storage experiment; no identity claim',
        native_allocation_evidence='receipts/native-camrad-allocation-order.json',
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'witness', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.witness, args.output)
