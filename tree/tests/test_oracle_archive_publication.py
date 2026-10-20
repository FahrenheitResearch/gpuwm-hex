"""Immutable compressed catalogue transport; no numerical acceptance inferred."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
from types import SimpleNamespace

import pytest

import site_support

ORACLE=Path(__file__).resolve().parents[1]/'tools/native_physics_reference/oracle'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


publisher=load('archive_publisher',ORACLE/'publish_dump.py')
matrix=load('archive_matrix',ORACLE/'replay/run_capture_matrix.py')


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n');return path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def catalogue(tmp_path):
    source=tmp_path/'source';source.mkdir()
    records=source/'records';records.mkdir()
    prefix='native.s0000000001.e0000000001.r000000'
    # Opaque transport bytes deliberately are not presented as native physics.
    (records/(prefix+'.pre.q.bin')).write_bytes(b'opaque transport fixture\x00')
    archive=source/'complete.tar.zst'
    site_support.tools_or_skip('tar','zstd',what='packing the compressed catalogue fixture')
    subprocess.run(['tar','--use-compress-program=zstd','-C',str(records),'-cf',str(archive),'.'],check=True)
    inventory=write(source/'inventory.json',{'pin_set_sha256':'a'*64,'reference':'rrfs'})
    capture=write(source/'capture.json',{'schema':'mpas-oracle-passivity-v1','exit':0,'changed_inputs':[],
        'process_count':1,'inventory_sha256':sha(inventory)})
    index=write(source/'capture-index.json',{'schema':'mpas-native-capture-replay-matrix-v1','completed':True,
        'process_count':1,'captures':[{'capture':prefix,'records':1}], 'records':1,
        'capture_receipt_sha256':sha(capture),'inventory_sha256':sha(inventory),'all_outputs_bitwise_equal':False})
    manifest=write(source/(prefix+'.manifest.json'),{'dump_root_sha256':'b'*64,
        'inventory_sha256':sha(inventory),'pin_set_sha256':'a'*64,'files':{'opaque':{}},'captures':{prefix:{}}})
    row={'capture':prefix,'records':1,'archive':str(source/(prefix+'.tar.gz')),
         'all_restored_member_sha256_equal':True,'capture_receipt_sha256':sha(capture),'dump_root_sha256':'b'*64}
    write(source/(prefix+'.archive.json'),row)
    merged=write(source/'consolidated.archive.json',{'schema':'mpas-native-consolidated-archive-v1',
        'all_decoded_tar_bytes_equal':True,'archive':str(archive),'archive_sha256':sha(archive),
        'archive_bytes':archive.stat().st_size,'capture_index_sha256':sha(index),
        'capture_receipt_sha256':sha(capture),'inventory_sha256':sha(inventory),
        'invocations':[dict(row,manifest_sha256=sha(manifest))]})
    return SimpleNamespace(source=source,archive=archive,index=index,merged=merged,capture=capture,
        inventory=inventory,campaign=tmp_path/'campaign',prefix=prefix)


def publish(c,suite='configured-suite'):
    return publisher.publish_archive(c.index,c.merged,c.capture,c.inventory,c.campaign,suite,'test-lane')


def test_archive_publication_relocated_transport_and_truthful_failure(catalogue):
    c=catalogue;r=publish(c);root=Path(r['path'])
    assert not r['all_outputs_bitwise_equal'] and not r['forecast_proven']
    assert r['records']==1 and r['invocations']==1
    assert publisher.campaign_bytes(c.campaign)==r['campaign_bytes']
    for name,row in r['members'].items():
        original=c.source/name
        assert original.stat().st_ino==(root/name).stat().st_ino
        assert sha(root/name)==row['sha256']
    args=SimpleNamespace(archive_publication=root/'PUBLICATION.json',from_archives=True,
        consolidated_archive=root/'consolidated.archive.json',archive_index=root/'capture-index.json',
        capture_receipt=root/'capture.json',inventory=root/'inventory.json',host='local',archive_root=str(root))
    # A producer pathname is provenance, not an I/O dependency after publication.
    c.source.rename(c.source.with_name('retired-original-path'))
    assert matrix.archive_origin(args)==str(c.source)
    unpacked=c.campaign.parent/'restored'
    matrix.unpack(['cat',str(root/'complete.tar.zst')],unpacked,selection=c.prefix)
    assert (unpacked/(c.prefix+'.pre.q.bin')).read_bytes()==b'opaque transport fixture\x00'
    with pytest.raises(ValueError,match='metadata differs'):
        args.inventory=write(c.campaign.parent/'wrong.json',{})
        matrix.archive_origin(args)


def test_archive_publication_cap_counts_metadata_and_shared_inodes(catalogue,monkeypatch):
    c=catalogue;first=publish(c)
    second=publish(c,'another-suite')
    assert publisher.campaign_bytes(c.campaign)==second['campaign_bytes']
    assert second['campaign_bytes']-first['campaign_bytes']==(Path(second['path'])/'PUBLICATION.json').stat().st_size
    monkeypatch.setattr(publisher,'CAP',second['campaign_bytes'])
    with pytest.raises(ValueError,match='cap'):
        publish(c,'third-suite')
    assert not list(c.campaign.rglob('.partial-*'))
    assert publisher.campaign_bytes(c.campaign)==second['campaign_bytes']
    with pytest.raises(ValueError,match='existing immutable'):
        publish(c)


def test_archive_publication_sealed_store_link_is_accounted(catalogue):
    c=catalogue;store=c.campaign.parent/'owned-data-store'
    r=publisher.publish_archive(c.index,c.merged,c.capture,c.inventory,c.campaign,
                                'configured-suite','test-lane',store)
    target=Path(r['path'])
    assert target.is_symlink() and target.resolve()==Path(r['storage_path'])
    assert publisher.campaign_bytes(c.campaign)==r['campaign_bytes']
    assert c.archive.stat().st_ino==(target/'complete.tar.zst').stat().st_ino
    # A second catalogue link and a cycle must not multiply physical bytes.
    (c.campaign/'alias').symlink_to(target,target_is_directory=True)
    (store/'loop').symlink_to(c.campaign,target_is_directory=True)
    assert publisher.campaign_bytes(c.campaign)==r['campaign_bytes']


@pytest.mark.parametrize('corruption',['payload','manifest','index','member-authentication','changed-inputs','inventory'])
def test_archive_publication_rejects_corruption(catalogue,corruption):
    c=catalogue
    if corruption=='payload':c.archive.write_bytes(b'corrupt compressed payload')
    elif corruption=='manifest':(c.source/(c.prefix+'.manifest.json')).write_text('{}')
    elif corruption=='index':c.index.write_text('{}')
    elif corruption=='inventory':c.inventory.write_text('{}')
    elif corruption=='changed-inputs':
        data=json.loads(c.capture.read_text());data['changed_inputs']=['native-input'];write(c.capture,data)
    else:
        data=json.loads(c.merged.read_text());data['invocations'][0]['all_restored_member_sha256_equal']=False;write(c.merged,data)
    with pytest.raises(ValueError):publish(c)
    assert not list(c.campaign.rglob('PUBLICATION.json'))


def test_native_publication_nested_stream_lists_preserve_bytes(tmp_path):
    passivity=load('archive_passivity',ORACLE/'run_passivity.py')
    runtime=tmp_path/'runtime';(runtime/'stream_list').mkdir(parents=True)
    stream=runtime/'stream_list/stream_list.atmosphere.output';stream.write_bytes(b'xtime\nu\n')
    receipt=write(tmp_path/'native.json',{'schema':'mpas.rrfs-native-forecast/v1','recipe':'gnu-strict','completed':True,
        'inputs':{},'resolved_configuration':{'files':{str(stream.relative_to(runtime)):sha(stream)}},
        'exit_status':0,'ranks':1,'binary_sha256':'c'*64,'argv':['native'],'outputs':{}})
    baseline=passivity.native_baseline(receipt,runtime)
    assert baseline['inputs']['stream_list/stream_list.atmosphere.output']=={'source':str(stream),'sha256':sha(stream)}
    data=json.loads(receipt.read_text())
    for name in ('../outside','/outside',''):
        data['resolved_configuration']['files']={name:'d'*64};write(receipt,data)
        with pytest.raises(ValueError,match='unsafe r0 runtime input'):
            passivity.native_baseline(receipt,runtime)


@pytest.mark.parametrize('recipe',['gnu-strict','gnu-o3','coverage'])
def test_replay_recipe_binds_captured_interface_and_actual_artifacts(tmp_path,recipe):
    # These opaque bytes test admission only; numerical identity is established
    # separately by executing the real native programs on captured fork data.
    executables=tmp_path/'replays';(executables/'sources').mkdir(parents=True)
    target='replay_native'
    executable=executables/target;executable.write_bytes(b'owned executable transport fixture')
    source=executables/'sources'/(target+'.F90');source.write_text('native source fixture')
    obj=tmp_path/'native.o';obj.write_bytes(b'owned audited object fixture')
    original={'sites':[{'site':'native','arguments':[{'type':'real(kind=rkind)'}]}],
              'source_sha256':{'scheme.F':'a'*64},'instrumentation_sha256':'b'*64}
    other=dict(original,instrumentation_sha256='c'*64)
    inventory=write(tmp_path/'inventory.json',other)
    manifest=write(executables/'build-manifest.json',{'schema':'mpas-native-framework-replays-v1',
        'recipe':recipe,'inventory_sha256':sha(inventory),'framework_stubs':[],
        'replay_source_sha256':{target:sha(source)}})
    artifacts=write(tmp_path/'artifacts.json',[{'path':str(p),'sha256':sha(p)} for p in (manifest,executable)])
    audit=write(tmp_path/'audit.json',{'recipe':recipe,'all_ok':True,
        'objects':[{'object':'native.o','object_sha256':sha(obj)}]})
    args=SimpleNamespace(replay_recipe=recipe,replay_inventory=inventory,replay_artifacts=artifacts,
                         replay_audit=audit,executable_dir=executables)
    result=matrix.replay_build(args,original)
    assert result['recipe']==recipe and result['explicit_build_binding']
    assert result['build_manifest_sha256']==sha(manifest)
    # A changed declaration cannot inherit another recipe's capture identity.
    changed=json.loads(inventory.read_text());changed['sites'][0]['arguments'][0]['type']='real(kind=8)'
    write(inventory,changed)
    with pytest.raises(ValueError,match='source interface differs'):matrix.replay_build(args,original)
    write(inventory,other)
    executable.write_bytes(b'changed executable')
    with pytest.raises(ValueError,match='executable artifact differs'):matrix.replay_build(args,original)
    executable.write_bytes(b'owned executable transport fixture')
    obj.write_bytes(b'changed object')
    with pytest.raises(ValueError,match='audited native object differs'):matrix.replay_build(args,original)
    obj.write_bytes(b'owned audited object fixture')
    manifest_data=json.loads(manifest.read_text());manifest_data['recipe']='wrong-recipe';write(manifest,manifest_data)
    with pytest.raises(ValueError,match='recipe or inventory differs'):matrix.replay_build(args,original)


def test_alternate_recipe_requires_explicit_build_binding():
    args=SimpleNamespace(replay_recipe='gnu-o3')
    with pytest.raises(ValueError,match='alternate replay recipe requires'):matrix.replay_build(args,{})
