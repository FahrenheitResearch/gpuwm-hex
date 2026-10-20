"""Observe and replay CAM's automatic entry storage, without native POST inputs.

MPAS module_ra_cam.F:364-365,678-679 reads LW-only tau allocations it never
wrote. This separate diagnostic makes that read memory an explicit PRE input.
It neither qualifies older incomplete captures nor proves a whole forecast.
The production port and original failures remain unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

FIELDS = ('tauxci', 'tauxcl')


def native(source, output):
    text = source.read_text()
    marker = '#if defined(mpas)\n   USE mpas_atmphys_utilities'
    if text.count(marker) != 1:
        raise ValueError('Expected the native CAM root module import')
    result = text.replace(marker, '   use mpas_oracle_dump, only: oracle_allocation\n' + marker)
    marker = '   nyrm     = yr - yrdata(1) + 1'
    if result.count(marker) != 1:
        raise ValueError('Expected the first native CAM root numerical statement')
    entry = '\n'.join("   call oracle_allocation('module_ra_cam.camrad." + name
                      + "'," + name + ',lower=lbound(' + name + '))' for name in FIELDS)
    result = result.replace(marker, entry + '\n' + marker)
    output.write_text(result)


def cuda(source, output):
    spec = importlib.util.spec_from_file_location('cam_entry_candidate', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fields = list(module.ORACLE_LOCAL_FIELDS)
    code = module.CUDA_SOURCE
    start = code.index('CAM_ROUTINE void camrad(')
    following = re.search(r'\n// MPAS \w+\.F:\d+-\d+\nCAM_ROUTINE ', code[start:])
    if not following:
        raise ValueError('Expected the complete root CAM routine')
    end = start + following.start()
    section = code[start:end]
    for name in FIELDS:
        offset = 18 * len(fields)
        fields.append(('allocation.module_ra_cam.camrad.' + name, 'double', 2))
        pattern = r'auto ' + name + r'=arena\.array<double>\([^;]+;'
        matches = list(re.finditer(pattern, section))
        if len(matches) != 1:
            raise ValueError('Expected one automatic CAM root declaration: ' + name)
        insertion = '''
// MPAS module_ra_cam.F:364-365,678-679: observable automatic PRE storage.
if(oracle_locals && oracle_locals[OFFSET]) {
 auto entry=oracle_array<double>(OFFSET);
 const int column=its-native_chunk_its+entry.low[0];
 for(int k=NAME.low[1];k<NAME.low[1]+NAME.extent[1];++k)
  NAME(NAME.low[0],k)=entry(column,k);
}
'''.replace('OFFSET', str(offset)).replace('NAME', name)
        match = matches[0]
        section = section[:match.end()] + insertion + section[match.end():]
    code = code[:start] + section + code[end:]
    text = source.read_text()
    marker = 'CUDA_SOURCE = r' + chr(34) * 3
    start = text.index(marker) + len(marker)
    end = text.index('\n' + chr(34) * 3, start)
    text = text[:start] + code + text[end:]
    text, count = re.subn(r'(?m)^ORACLE_LOCAL_FIELDS = .+$',
                         lambda _: 'ORACLE_LOCAL_FIELDS = ' + repr(tuple(fields)), text)
    if count != 1:
        raise ValueError('Expected one complete native local-storage inventory')
    marker = 'descriptor=_pack(((name,kind,1),),{name:local_state[name]},device)'
    if text.count(marker) != 1:
        raise ValueError('Expected the native local-storage descriptor launcher')
    # The original SAVE inventory contains only scalars and rank-one arrays.
    # Automatic tau entry storage has rank two; scalar links still use rank one.
    text = text.replace(marker,
        'descriptor=_pack(((name,kind,rank or 1),),{name:local_state[name]},device)')
    output.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('native', 'cuda'))
    for name in ('source', 'output', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    {'native': native, 'cuda': cuda}[args.mode](args.source, args.output)
    args.receipt.write_text(json.dumps(dict(
        scope='explicit automatic PRE storage diagnostic; execution unqualified',
        mode=args.mode, native_citation='module_ra_cam.F:364-365,678-679',
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        native_post_inputs=False, excluded_words=0, whole_model_acceptance=False),
        indent=2) + '\n')
