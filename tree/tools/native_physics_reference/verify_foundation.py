#!/usr/bin/env python3
"""Collect foundation evidence without turning queued work into a DONE claim.

The explicit index names acceptance checks, flag audits, generated deck indexes,
case products with their clock receipts, and forecast catalogs. Identity proofs
are separate: native execution never implies CUDA equivalence or rank identity.
"""
import argparse
import json
from pathlib import Path
from generate_decks import companions, options
from index_forecasts import catalog
from cache_cases import contract
from publish_cache import sha
from artifact_storage import content_sha
from verify_acceptance import validate as validate_acceptance


def read(path):
    return json.loads(Path(path).read_text())


def deck_coverage(directory, source):
    directory=Path(directory);index=read(directory/'INDEX.json')
    choices,suites=options(Path(source))
    if 'config_oml1d' in index['option_choices']:
        choices['config_oml1d']=['.false.','.true.']
    if index['option_choices']!=choices:raise ValueError('generated option coverage differs from MPAS')
    expected={(option,value) for option,values in choices.items() for value in values}
    if 'companion_table_sha256' in index:
        table=Path(__file__).with_name('deck_companions.v842.json')
        if sha(table)!=index['companion_table_sha256']:
            raise ValueError('native companion table hash differs')
    variant_specs={}
    if 'variant_table_sha256' in index:
        table=Path(__file__).with_name('deck_variants.v842.json')
        if sha(table)!=index['variant_table_sha256']:raise ValueError('native variant table hash differs')
        for variant in read(table)['variants']:
            for suite in variant['suites']:
                variant_specs[variant['name']+'-'+suite]=(suite,variant['override'],variant.get('companions',{}) if suite=='none' else {})
    actual=set();suite_names=set();seen_variants=set()
    for row in index['decks']:
        deck=directory/row['deck']
        for name,key in [('namelist.atmosphere','namelist_sha256'),('streams.atmosphere','streams_sha256')]:
            if sha(deck/name)!=row[key]:raise ValueError('generated deck hash differs')
        for name,digest in row.get('stream_list_sha256',{}).items():
            if sha(deck/name)!=digest:raise ValueError('native stream-list hash differs')
        if row['deck'] in variant_specs:
            if (row['suite'],row['override'],row['companions'])!=variant_specs[row['deck']]:
                raise ValueError('native option variant or activation companions differ')
            seen_variants.add(row['deck'])
        elif not row['override']:
            suite_names.add(row['suite'])
        else:
            if row['suite']!='none' or len(row['override'])!=1:
                raise ValueError('single-scheme deck must have suite none and one override')
            option,value=next(iter(row['override'].items()))
            if row['companions']!=companions(option,value):raise ValueError('forced scheme companions differ')
            actual.add((option,value))
        if row['resolved_suite']!=suites[row['suite']]:raise ValueError('resolved native suite differs')
    if (actual!=expected or suite_names!=set(suites) or seen_variants!=set(variant_specs)
        or len(index['decks'])!=len(expected)+len(suites)+len(variant_specs)):
        raise ValueError('generated native decks are incomplete or duplicated')
    for name,digest in index['source_sha256'].items():
        if sha(Path(source)/name)!=digest:raise ValueError('deck authority source hash differs')
    return {'options':len(expected),'suites':len(suites),'decks':len(index['decks']),
            'execution_claim':False,'index_sha256':sha(directory/'INDEX.json')}


def product_clock(row):
    directory=Path(row['directory']);receipt=Path(row['receipt'])
    if not (directory/'PUBLICATION.json').is_file() or not receipt.is_file():return None
    manifest=read(directory/'MANIFEST.json');publication=read(directory/'PUBLICATION.json');clock=read(receipt)
    if 'case' in row and contract(manifest['case'])!=contract(row['case']):
        raise ValueError('native input publication belongs to another requested case')
    digest=sha(directory/'MANIFEST.json');outputs=manifest.get('outputs',manifest.get('files'))
    if (clock['directory']!=str(directory) or clock['manifest_sha256']!=digest
            or publication['files'].get('MANIFEST.json')!=digest):
        raise ValueError('clock or publication belongs to another native manifest')
    if not clock['complete'] or clock['errors']:raise ValueError('native clock witness failed')
    if {file['file']:file['sha256'] for file in clock['files']}!=outputs:
        raise ValueError('native clock witness output coverage differs')
    if not outputs:raise ValueError('native product has no outputs')
    for name,digest in outputs.items():
        if content_sha(directory/name)!=digest or publication['files'].get(name)!=digest:
            raise ValueError('native output hash differs from witnessed publication')
    return {'files':len(outputs),'manifest_sha256':sha(directory/'MANIFEST.json'),
            'receipt_sha256':sha(receipt)}


def collect(index):
    rows=[]
    def check(kind,name,action):
        try:
            evidence=action()
            rows.append({'kind':kind,'name':str(name),'status':'pending' if evidence is None else 'passed',
                         'evidence':evidence})
        except (OSError,ValueError,KeyError,TypeError) as error:
            rows.append({'kind':kind,'name':str(name),'status':'failed','error':str(error)})
    for row in index['acceptance']:
        def acceptance(row=row):
            errors=validate_acceptance(read(row['receipt']),row['hours'],row['cells'],row['levels'],row['init_sha256'])
            if errors:raise ValueError('; '.join(errors))
            return {'receipt_sha256':sha(Path(row['receipt'])),'hours':row['hours']}
        check('hex-dycore-acceptance',row['receipt'],acceptance)
    for path in index['flags']:
        def flags(path=path):
            audit=read(path)
            if not audit['all_ok'] or audit['errors'] or not audit['objects']:
                raise ValueError('native object flag audit is incomplete or failed')
            groups={'physics_mmm/','physics_noaa/UGWP/','physics_noahmp/','physics_wrf/'}
            if (not all(row['flags_ok'] for row in audit['objects']) or not groups.issubset(audit['groups'])
                    or not all(audit['groups'].values())):
                raise ValueError('native external object flag coverage is incomplete')
            return {'recipe':audit['recipe'],'objects':len(audit['objects']),'groups':audit['groups'],
                    'receipt_sha256':sha(Path(path))}
        check('native-flags',path,flags)
    for path in index['decks']:
        check('generated-option-coverage',path,lambda path=path:deck_coverage(path,index['source']))
    for row in index.get('artifacts',[]):
        def artifact(row=row):
            directory=Path(row['directory']);path=directory/'MANIFEST.json'
            if sha(path)!=row['manifest_sha256']:raise ValueError('pinned artifact manifest hash differs')
            files=read(path)['files']
            if not files:raise ValueError('pinned artifact has no files')
            for name,digest in files.items():
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('artifact file escapes directory')
                if sha(directory/relative)!=digest:raise ValueError('pinned artifact file hash differs: '+name)
            return {'files':len(files),'manifest_sha256':sha(path)}
        check('pinned-native-artifact',row['directory'],artifact)
    for row in index['products']:
        check('native-product-clocks',row['directory'],lambda row=row:product_clock(row))
    for row in index.get('forecast_jobs',[]):
        def forecasts(row=row):
            result=catalog(read(row['index'])['jobs'],Path(row['details']))
            if any(entry['status']=='clock-witness-failed' for entry in result['entries']):
                raise ValueError('forecast clock witness failed')
            if not result['complete']:return None
            return {'jobs':len(result['entries']),'job_index_sha256':sha(Path(row['index']))}
        check('native-forecast-contracts',row['index'],forecasts)
    # Identity rows deliberately require the caller to declare their complete,
    # nonempty expected file set and both completed forecast paths.
    for row in index['identity']:
        def identity(row=row):
            path=Path(row['receipt'])
            if not path.is_file():return None
            proof=read(path)
            comparisons=proof.get('comparisons',[proof])
            if len(comparisons)!=1:raise ValueError('expected one full-forecast identity comparison')
            comparison=comparisons[0]
            if [comparison['reference'],comparison['other']]!=row['runs']:
                raise ValueError('identity witness belongs to other forecasts')
            manifests=[read(Path(run)/'MANIFEST.json') for run in row['runs']]
            names=set(comparison['files'])
            if not names or comparison['errors'] or any(names!=set(m['outputs']) for m in manifests):
                raise ValueError('identity witness has incomplete output coverage')
            for name,file in comparison['files'].items():
                if not file['fields'] or file['sha256']!=[m['outputs'][name] for m in manifests]:
                    raise ValueError('identity witness does not match completed native outputs')
            if not comparison['all_identical'] and not comparison.get('first_difference'):
                raise ValueError('measured variation requires the first differing field and word')
            return {'all_fields_identical':comparison['all_identical'],
                    'all_file_bytes_identical':comparison['all_file_bytes_identical'],
                    'first_difference':comparison.get('first_difference'),
                    'receipt_sha256':sha(path),'files':len(names)}
        check('forecast-identity-witness',row['receipt'],identity)
    if not rows:raise ValueError('empty foundation evidence index')
    return {'schema':'mpas.native-foundation-evidence/v1',
            'complete':all(row['status']=='passed' for row in rows),'checks':rows,
            'identity_claim':'only explicitly witnessed comparisons; deck coverage is generation only'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('index',type=Path)
    p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    result=collect(read(a.index));a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':result['complete'],'statuses':{status:sum(row['status']==status for row in result['checks'])
        for status in ('passed','pending','failed')}}))
    return int(not result['complete'])


if __name__=='__main__':raise SystemExit(main())
