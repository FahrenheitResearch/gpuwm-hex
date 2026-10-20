#!/usr/bin/env python3
"""Publish an owned sealed capture under the campaign's shared oracle root.

Publication is local and uses an exclusive cap/accounting lock. Only an owned
source directory is moved; no existing publication is replaced or retired.
Superseded captures may be explicitly retired after their receipts are saved.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import uuid

CAP=100*1000**3


def campaign_inodes(campaign):
    """Account for archived payloads and manifests, deduplicating hard links."""
    seen=set();directories=set();used=0;pending=[campaign]
    while pending:
        directory=pending.pop();stat=directory.stat();identity=(stat.st_dev,stat.st_ino)
        if identity in directories:continue
        directories.add(identity)
        for path in directory.iterdir():
            if path.is_dir():
                # Cross-disk catalogue links point only to sealed stores. Do
                # not traverse arbitrary directory links or directory cycles.
                if not path.is_symlink() or (path/'PUBLICATION.json').is_file():pending.append(path)
                continue
            if not path.is_file() or path.name=='.publish.lock':continue
            stat=path.stat();identity=(stat.st_dev,stat.st_ino)
            if identity not in seen:seen.add(identity);used+=stat.st_size
    return used,seen


def campaign_bytes(campaign):return campaign_inodes(campaign)[0]


def publication_text(receipt,used):
    # The JSON receipt itself occupies campaign space. Its decimal byte count
    # can change its own length, so settle the count before checking the cap.
    receipt['campaign_bytes']=used
    while True:
        encoded=json.dumps(receipt,indent=2)+'\n'
        total=used+len(encoded.encode())
        if receipt['campaign_bytes']==total:break
        receipt['campaign_bytes']=total
    if total>CAP:raise ValueError('campaign oracle cap would exceed 100 GB')
    return encoded


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def publish_archive(index_path,consolidated_path,capture_path,inventory_path,campaign,suite,owner,archive_store=None):
    """Publish an authenticated archive catalogue without decoding weather data.

    The completed native producer already independently restored every member;
    consolidation authenticated the decoded tar stream. Preserve those receipts
    unchanged, and authenticate the actual payload and each invocation manifest.
    Publication never promotes replay or forecast acceptance.
    """
    if not re.fullmatch(r'[A-Za-z0-9_.+-]+',suite):raise ValueError('invalid suite path component')
    if not re.fullmatch(r'[a-z0-9-]+',owner):raise ValueError('invalid owner')
    index=json.loads(index_path.read_text());merged=json.loads(consolidated_path.read_text())
    capture=json.loads(capture_path.read_text());inventory=json.loads(inventory_path.read_text())
    if index.get('schema')!='mpas-native-capture-replay-matrix-v1' or not index.get('completed'):
        raise ValueError('archive publication requires a complete capture index')
    if (capture.get('schema')!='mpas-oracle-passivity-v1' or capture.get('exit')!=0 or
        capture.get('changed_inputs') or capture.get('process_count')!=index['process_count']):
        raise ValueError('archive capture must be a completed unchanged native forecast')
    if (merged.get('schema')!='mpas-native-consolidated-archive-v1' or
        not merged.get('all_decoded_tar_bytes_equal') or
        merged['capture_index_sha256']!=digest(index_path) or
        merged['capture_receipt_sha256']!=digest(capture_path) or
        index['capture_receipt_sha256']!=digest(capture_path) or
        merged['inventory_sha256']!=digest(inventory_path) or
        index['inventory_sha256']!=digest(inventory_path) or
        capture['inventory_sha256']!=digest(inventory_path)):
        raise ValueError('archive catalogue provenance differs')
    source=Path(merged['archive']);campaign=campaign.resolve()
    if source.name!='complete.tar.zst' or digest(source)!=merged['archive_sha256'] or source.stat().st_size!=merged['archive_bytes']:
        raise ValueError('canonical archive payload differs')
    pin=inventory['pin_set_sha256']
    if not re.fullmatch(r'[0-9a-f]{64}',pin):raise ValueError('invalid pin-set hash')
    expected={r['capture']:r['records'] for r in index['captures']}
    rows=merged['invocations']
    if (not expected or len(expected)!=len(index['captures']) or len(rows)!=len(expected) or
        {r['capture']:r['records'] for r in rows}!=expected or sum(expected.values())!=index['records']):
        raise ValueError('archive invocation inventory differs')
    files={'complete.tar.zst':source,'capture-index.json':index_path,
           'consolidated.archive.json':consolidated_path,'capture.json':capture_path,'inventory.json':inventory_path}
    for row in rows:
        prefix=row['capture']
        if not re.fullmatch(r'[A-Za-z0-9_.+-]+',prefix):raise ValueError('invalid capture prefix')
        if (not row['all_restored_member_sha256_equal'] or
            row['capture_receipt_sha256']!=digest(capture_path) or
            row['archive']!=str(source.parent/(prefix+'.tar.gz'))):
            raise ValueError('archive member authentication is incomplete')
        manifest=source.parent/(prefix+'.manifest.json');metadata=source.parent/(prefix+'.archive.json')
        sealed=json.loads(manifest.read_text());original=json.loads(metadata.read_text())
        if (digest(manifest)!=row['manifest_sha256'] or
            any(original.get(k)!=v for k,v in row.items() if k!='manifest_sha256') or
            sealed['dump_root_sha256']!=row['dump_root_sha256'] or
            sealed['inventory_sha256']!=digest(inventory_path) or
            sealed['pin_set_sha256']!=pin or len(sealed['files'])!=row['records'] or
            set(sealed['captures'])!={prefix}):
            raise ValueError('archived invocation manifest differs')
        files[manifest.name]=manifest;files[metadata.name]=metadata
    campaign.mkdir(parents=True,exist_ok=True)
    store=archive_store.resolve() if archive_store else campaign
    store.mkdir(parents=True,exist_ok=True)
    if any(path.stat().st_dev!=store.stat().st_dev for path in files.values()):
        raise ValueError('archive publication requires one filesystem for immutable hard links')
    target=campaign/pin/suite/merged['archive_sha256']
    storage=store/pin/suite/merged['archive_sha256']
    with (campaign/'.publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists() or target.is_symlink() or storage.exists():raise ValueError('refusing existing immutable publication')
        used,seen=campaign_inodes(campaign)
        size=0
        for path in files.values():
            stat=path.stat();identity=(stat.st_dev,stat.st_ino)
            if identity not in seen:seen.add(identity);size+=stat.st_size
        if used+size>CAP:raise ValueError('campaign oracle cap would exceed 100 GB')
        target.parent.mkdir(parents=True,exist_ok=True)
        storage.parent.mkdir(parents=True,exist_ok=True)
        staging=storage.parent/('.partial-'+owner+'-'+uuid.uuid4().hex)
        staging.mkdir()
        try:
            members={}
            for name,path in files.items():
                os.link(path,staging/name)
                members[name]={'sha256':digest(path),'bytes':path.stat().st_size}
            receipt={'schema':'mpas-oracle-archived-publication-v1','owner':owner,'suite':suite,
                     'path':str(target),'source_archive_root':str(source.parent),
                     'storage_path':str(storage),
                     'pin_set_sha256':pin,'reference':inventory['reference'],
                     'records':index['records'],'invocations':len(rows),'members':members,
                     'source_retained':True,'all_member_authentication_receipts_preserved':True,
                     'all_outputs_bitwise_equal':index['all_outputs_bitwise_equal'],
                     'process_count':index['process_count'],'campaign_bytes':used+size,
                     'campaign_cap_bytes':CAP,'forecast_proven':False}
            (staging/'PUBLICATION.json').write_text(publication_text(receipt,used+size))
            staging.rename(storage)
            if storage!=target:
                try:target.symlink_to(storage,target_is_directory=True)
                except BaseException:
                    shutil.rmtree(storage);raise
        except BaseException:
            if staging.exists():shutil.rmtree(staging)
            raise
    return receipt


def publish(source,campaign,suite,owner,keep_source=False):
    if not re.fullmatch(r'[A-Za-z0-9_.+-]+',suite):raise ValueError('invalid suite path component')
    if not re.fullmatch(r'[a-z0-9-]+',owner):raise ValueError('invalid owner')
    source=source.resolve();campaign=campaign.resolve()
    if campaign==source or source in campaign.parents or campaign in source.parents:
        raise ValueError('source and campaign must be separate trees')
    path=Path(__file__).resolve().parents[2]/'column_oracle_replay.py'
    spec=importlib.util.spec_from_file_location('column_oracle_replay',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=module.verify(source)
    target=campaign/manifest['pin_set_sha256']/suite/manifest['dump_root_sha256']
    campaign.mkdir(parents=True,exist_ok=True)
    with (campaign/'.publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists():raise ValueError('refusing existing immutable publication')
        used,seen=campaign_inodes(campaign)
        size=sum(f['bytes'] for f in manifest['files'].values())
        added=0
        for path in source.rglob('*'):
            if not path.is_file():continue
            stat=path.stat()
            if not (keep_source and path.suffix=='.bin' and (stat.st_dev,stat.st_ino) in seen):added+=stat.st_size
        if used+added>CAP:raise ValueError('campaign oracle cap would exceed 100 GB; retire superseded owned captures first')
        target.parent.mkdir(parents=True,exist_ok=True)
        # A cross-device move copies many files before removing the source.
        # Keep that copy outside the immutable name until it is complete.
        staging=target.parent/('.partial-'+owner+'-'+uuid.uuid4().hex)
        if keep_source:
            # Existing scheme lanes may still read the original capture path.
            # Native records are immutable; hard links publish the same bytes
            # without a second payload or a pathname interruption.
            if source.stat().st_dev != campaign.stat().st_dev:
                raise ValueError('keeping the source requires one filesystem; copying would duplicate the raw payload')
            def member_copy(original,destination):
                if Path(original).suffix == '.bin':
                    os.link(original,destination)
                    return destination
                return shutil.copy2(original,destination)
            shutil.copytree(source,staging,copy_function=member_copy)
        else:
            shutil.move(str(source),staging)
        receipt={'schema':'mpas-oracle-local-publication-v1','owner':owner,'suite':suite,
                 'path':str(target),'dump_root_sha256':manifest['dump_root_sha256'],
                 'pin_set_sha256':manifest['pin_set_sha256'],'records':len(manifest['files']),
                 'bytes':size,'campaign_bytes':used+size,'campaign_cap_bytes':CAP,
                 'source_retained':keep_source,'forecast_proven':False}
        try:
            (staging/'PUBLICATION.json').write_text(publication_text(receipt,used+added))
        except BaseException:
            if keep_source:shutil.rmtree(staging)
            else:shutil.move(str(staging),source)
            raise
        staging.rename(target)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--dump',type=Path)
    choice.add_argument('--archive-index',type=Path)
    parser.add_argument('--consolidated-archive',type=Path)
    parser.add_argument('--capture-receipt',type=Path)
    parser.add_argument('--inventory',type=Path)
    parser.add_argument('--archive-store',type=Path,help='owned data-disk store on the archive filesystem; publish a catalogue link across disks')
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--suite',required=True)
    parser.add_argument('--owner',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--keep-source',action='store_true',
                        help='publish immutable records as hard links, retaining active readers at the original path')
    a=parser.parse_args()
    if a.archive_index:
        if not all((a.consolidated_archive,a.capture_receipt,a.inventory)):
            parser.error('archive publication requires --consolidated-archive, --capture-receipt and --inventory')
        result=publish_archive(a.archive_index,a.consolidated_archive,a.capture_receipt,a.inventory,a.campaign_root,a.suite,a.owner,a.archive_store)
    else:result=publish(a.dump,a.campaign_root,a.suite,a.owner,a.keep_source)
    a.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
