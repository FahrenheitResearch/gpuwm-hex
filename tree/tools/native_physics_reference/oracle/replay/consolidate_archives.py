#!/usr/bin/env python3
"""Consolidate complete native invocations into one bounded canonical archive.

GNU gzip/tar retain opaque member bytes; zstd's long window shares repeated
native tables across invocations. Both uncompressed stream SHA256s must match
before original owned archives can be retired. No field data are transformed.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from run_capture_matrix import harness,member_prefix,remote,write,host_command,copy_to_host


def consolidate(a):
    index=json.loads(a.index.read_text())
    if index['schema']!='mpas-native-capture-replay-matrix-v1' or not index['completed']:
        raise ValueError('capture index is incomplete')
    if a.working.exists():raise ValueError('consolidation working directory must be fresh')
    a.working.mkdir(parents=True)
    rows=[]
    for row in index['captures']:
        prefix=row['capture'];member_prefix(prefix+'.pre.check.bin')
        metadata=json.loads(remote(a.host,'cat '+shlex.quote(a.archive_root+'/'+prefix+'.archive.json'),capture_output=True,text=True).stdout)
        target=a.archive_root+'/'+prefix+'.tar.gz'
        if (metadata['schema']!='mpas-native-invocation-archive-v1' or metadata['archive']!=target or
            metadata['capture']!=prefix or metadata['records']!=row['records'] or
            metadata['capture_receipt_sha256']!=index['capture_receipt_sha256'] or
            not metadata['all_restored_member_sha256_equal']):
            raise ValueError('native invocation provenance differs')
        actual=remote(a.host,'sha256sum '+shlex.quote(target),capture_output=True,text=True).stdout.split()[0]
        if actual!=metadata['archive_sha256']:raise ValueError('native invocation archive hash differs')
        metadata['manifest_sha256']=remote(a.host,'sha256sum '+shlex.quote(a.archive_root+'/'+prefix+'.manifest.json'),capture_output=True,text=True).stdout.split()[0]
        rows.append(metadata)
    if not rows or len({r['capture'] for r in rows})!=len(rows):raise ValueError('empty or duplicated invocation index')
    target=a.archive_root+'/complete.tar.zst';partial=target+'.partial'
    remote(a.host,'test ! -e '+shlex.quote(target)+' && test ! -e '+shlex.quote(partial))
    fifo=a.working/'tar-sha.pipe';os.mkfifo(fifo)
    hash_path=a.working/'input-tar.sha256'
    with hash_path.open('wb') as hash_out:
        hasher=subprocess.Popen(['sha256sum',str(fifo)],stdout=hash_out)
        compressor=subprocess.Popen(['zstd','-1','--long=31','-T1','--check','-c'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        sink=subprocess.Popen(host_command(a.host,'cat > '+shlex.quote(partial)),stdin=compressor.stdout)
        compressor.stdout.close()
        tee=subprocess.Popen(['tee',str(fifo)],stdin=subprocess.PIPE,stdout=compressor.stdin)
        compressor.stdin.close()
        try:
            # Concatenated tar archives preserve every uniquely prefixed record.
            # GNU tar --ignore-zeros selects individual invocation members later.
            for row in rows:
                source=subprocess.Popen(host_command(a.host,'cat '+shlex.quote(row['archive'])),stdout=subprocess.PIPE)
                decoder=subprocess.run(['gzip','-dc'],stdin=source.stdout,stdout=tee.stdin)
                source.stdout.close();status=source.wait()
                if status or decoder.returncode:raise ValueError('native archive stream failed')
        finally:
            tee.stdin.close()
            statuses=[tee.wait(),compressor.wait(),sink.wait(),hasher.wait()]
    fifo.unlink()
    if any(statuses):raise ValueError('canonical consolidation pipeline failed: '+str(statuses))
    expected=hash_path.read_text().split()[0]
    # Authenticate the exact decoded tar byte stream independently from gzip.
    source=subprocess.Popen(host_command(a.host,'cat '+shlex.quote(partial)),stdout=subprocess.PIPE)
    decoder=subprocess.Popen(['zstd','-dc','--long=31'],stdin=source.stdout,stdout=subprocess.PIPE)
    source.stdout.close()
    actual=subprocess.run(['sha256sum'],stdin=decoder.stdout,capture_output=True,text=True)
    decoder.stdout.close();statuses=[source.wait(),decoder.wait(),actual.returncode]
    if any(statuses) or actual.stdout.split()[0]!=expected:
        raise ValueError('canonical decoded tar SHA256 differs')
    remote(a.host,'mv '+shlex.quote(partial)+' '+shlex.quote(target))
    archive_hash=remote(a.host,'sha256sum '+shlex.quote(target),capture_output=True,text=True).stdout.split()[0]
    archive_bytes=int(remote(a.host,'stat -c %s '+shlex.quote(target),capture_output=True,text=True).stdout)
    result={'schema':'mpas-native-consolidated-archive-v1','canonical_host':a.host,'archive':target,
        'archive_sha256':archive_hash,'archive_bytes':archive_bytes,'decoded_tar_sha256':expected,
        'all_decoded_tar_bytes_equal':True,'capture_index_sha256':harness.hash_file(a.index),
        'capture_receipt_sha256':index['capture_receipt_sha256'],'inventory_sha256':index['inventory_sha256'],
        'invocations':rows,'runner_sha256':harness.hash_file(Path(__file__)),'forecast_proven':False}
    write(a.receipt,result)
    copy_to_host(a.receipt,a.host,a.archive_root+'/consolidated.archive.json')
    if a.retire_originals:
        for row in rows:remote(a.host,'rm -- '+shlex.quote(row['archive']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('index','working','receipt'):parser.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    for name in ('host','archive-root'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--retire-originals',action='store_true')
    args=parser.parse_args()
    try:
        result=consolidate(args);print(json.dumps({'captures':len(result['invocations']),'archive_bytes':result['archive_bytes'],'all_decoded_tar_bytes_equal':True}));return 0
    except (OSError,ValueError,subprocess.CalledProcessError) as error:
        print('native archive consolidation: '+str(error));return 2

if __name__=='__main__':raise SystemExit(main())
