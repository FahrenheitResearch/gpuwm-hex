"""Grade complete native fixtures through Rust; preserve every output word mask."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args):
    library = ct.CDLL(str(args.bridge.resolve()))
    ptr = ct.POINTER(ct.c_uint32)
    for name, types, result in [('open',[ct.c_char_p],ct.c_void_p),('metadata',[ct.c_void_p],ct.c_char_p),
                               ('words',[ct.c_void_p,ct.c_size_t],ptr),
                               ('compare',[ct.c_void_p,ct.c_size_t,ptr,ct.c_size_t],ct.c_char_p),
                               ('close',[ct.c_void_p],None)]:
        function = getattr(library,'rw_column_fixture_'+name)
        function.argtypes,function.restype = types,result
    handles = [library.rw_column_fixture_open(str(path.resolve()).encode()) for path in [args.reference,args.candidate]]
    try:
        if not all(handles):
            raise ValueError('Rust rejected complete native fixture')
        inventories = [json.loads(library.rw_column_fixture_metadata(h))['fields'] for h in handles]
        if inventories[0] != inventories[1]:
            raise ValueError('Complete native fixture inventories differ')
        rows = []
        for field in inventories[0]:
            if field['phase'] != 'out':
                continue
            candidate = library.rw_column_fixture_words(handles[1],field['index'])
            grade = library.rw_column_fixture_compare(handles[0],field['index'],candidate,field['words'])
            if not grade:
                raise ValueError('Rust rejected full native word grade')
            rows.append(json.loads(grade))
        args.chunks.mkdir(parents=True,exist_ok=False)
        chunks = []
        for offset in range(0,len(rows),5000):
            path = args.chunks/f'grades-{offset:06d}.json'
            part = rows[offset:offset+5000]
            path.write_text(json.dumps(part,indent=2)+'\n')
            chunks.append(dict(path=str(path),sha256=sha(path),records=len(part)))
        result = dict(scope=args.label, process_count=1,excluded_words=[],
                      reference_sha256=sha(args.reference),candidate_sha256=sha(args.candidate),
                      bridge_sha256=sha(args.bridge),grade_chunks=chunks,grade_records=len(rows),
                      words=sum(row['words'] for row in rows),differing_words=sum(row['differing_words'] for row in rows))
        args.receipt.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ['words','differing_words']}))
        return 0
    finally:
        for handle in handles:
            if handle:library.rw_column_fixture_close(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['reference','candidate','bridge','receipt','chunks']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--label',required=True)
    raise SystemExit(run(parser.parse_args()))
