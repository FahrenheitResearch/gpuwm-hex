#!/usr/bin/env python3
"""Publish a native input copy with explicitly selected floating fields set to +0.

The Rust copier changes only caller-named fields. The Rust all-word comparator
then requires every other native variable to remain identical. Native source
and thermodynamic/mesh/terrain fields are unchanged.
"""
import argparse,json,shutil,subprocess
from pathlib import Path
from publish_cache import sha,publish


def prepare(source,work,destination,fields,copier,comparator,ready,description):
    if not fields or len(set(fields))!=len(fields):raise ValueError('nonempty distinct explicit fields required')
    source=source.resolve(strict=True)
    manifest=json.loads((source/'PUBLICATION.json').read_text())
    inputs=[name for name in manifest['files'] if name.endswith('.init.nc')]
    if len(inputs)!=1:raise ValueError('exactly one published native init required')
    work.mkdir(parents=True,exist_ok=False)
    for name,digest in manifest['files'].items():
        p=source/name
        if sha(p)!=digest:raise ValueError('native source publication changed')
        if name!=inputs[0]:shutil.copyfile(p,work/name)
    original=source/inputs[0];target=work/inputs[0]
    subprocess.run([str(copier),str(original),str(target),*fields],check=True)
    result=subprocess.run([str(comparator),str(original),str(target)],text=True,capture_output=True)
    if result.returncode not in (0,1):raise RuntimeError(result.stderr)
    proof=json.loads(result.stdout)
    names={row['field'] for row in proof['fields']}
    if proof.get('errors') or not set(fields).issubset(names) or any(row['differing_words'] for row in proof['fields'] if row['field'] not in fields):
        raise ValueError('dry input copy changed an unselected native field')
    # Selected fields are checked for exact stored +0 by the Rust copier itself.
    (work/'WORD-COPY-WITNESS.json').write_text(json.dumps(proof,indent=2)+'\n')
    provenance={'schema':'mpas.native-explicit-zero-input-copy/v1','source_publication_sha256':sha(source/'PUBLICATION.json'),
      'initial_source_sha256':sha(original),'initial_result_sha256':sha(target),'zero_fields':fields,
      'copier_sha256':sha(copier),'comparator_sha256':sha(comparator),'unselected_fields_differing_words':0,
      'word_witness_sha256':sha(work/'WORD-COPY-WITNESS.json'),'source':str(source)}
    (work/'MANIFEST.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return publish(work,destination,['*'],ready,description)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','work','destination','copier','comparator','ready','receipt'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--field',action='append',required=True);p.add_argument('--description',required=True)
    a=p.parse_args();result=prepare(a.source,a.work,a.destination,a.field,a.copier.resolve(),a.comparator.resolve(),a.ready,a.description)
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
