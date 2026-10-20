"""Regression checks for native proof orchestration, not physics substitutes."""
import site_support
import importlib.util
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]/'tools/native_physics_reference'

def load(name):
    spec=importlib.util.spec_from_file_location(name, TOOLS/(name+'.py'))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_cancelled_gpu_reuse_rejects_changed_native_science(tmp_path,monkeypatch):
    import copy,pytest
    monkeypatch.syspath_prepend(str(TOOLS));module=load('reproduce_acceptance')
    mpi=tmp_path/'mpi';mpi.write_bytes(b'original launcher bytes')
    config={'binary_sha256':'native','initial_sha256':'initial','ranks':16,
            'cells':7,'levels':3,'cores':'0-15','mpi':str(mpi),
            'runs':[{'name':'whole','hours':[6,12,18,24],
                     'files':{'namelist.atmosphere':{'path':'old','sha256':'deck'}}}]}
    relocated=copy.deepcopy(config);relocated['runs'][0]['files']['namelist.atmosphere']['path']='sealed'
    module.validate_reuse_contract(relocated,config)
    for change in ('rank','deck','sample'):
        wrong=copy.deepcopy(relocated)
        if change=='rank':wrong['ranks']=1
        elif change=='deck':wrong['runs'][0]['files']['namelist.atmosphere']['sha256']='changed'
        else:wrong['runs'][0]['hours']=[6]
        with pytest.raises(ValueError,match='native reuse'):
            module.validate_reuse_contract(wrong,config)
    other=tmp_path/'other-mpi';other.write_bytes(b'changed launcher bytes')
    relocated['mpi']=str(other)
    with pytest.raises(ValueError,match='MPI executable differs'):
        module.validate_reuse_contract(relocated,config)

def test_audit_rejects_missing_external_groups(tmp_path):
    (tmp_path/'build.log').write_text('')
    assert not load('audit_flags').audit(tmp_path,'gnu-strict')['all_ok']

def test_audit_rejects_one_unstrict_external_object(tmp_path):
    audit=load('audit_flags'); flags=' '.join(audit.STRICT)
    commands=[]
    for group in ('physics_mmm', 'physics_noaa/UGWP', 'physics_noahmp/src', 'physics_wrf'):
        stem=group.split('/')[-1]
        obj=tmp_path/'src/core_atmosphere/physics'/group/(stem+'.o')
        obj.parent.mkdir(parents=True); obj.touch()
        commands.append(f'mpif90 -O3 -DSINGLE_PRECISION {flags if stem != "UGWP" else ""} -c {stem}.F90')
    (tmp_path/'build.log').write_text('\n'.join(commands))
    r=audit.audit(tmp_path,'gnu-strict')
    assert r['errors']==['src/core_atmosphere/physics/physics_noaa/UGWP/UGWP.o']

def test_control_forced_companions():
    companions=load('generate_decks').companions
    assert companions('config_pbl_scheme','bl_mynn')=={'config_sfclayer_scheme':'sf_mynn'}
    assert companions('config_pbl_scheme','bl_ysu')=={'config_sfclayer_scheme':'sf_monin_obukhov_rev'}
    assert companions('config_radt_sw_scheme','cam_sw')=={'config_radt_cld_scheme':'cld_incidence'}
    assert companions('config_lsm_scheme','off')=={}


def test_native_driver_activation_and_allocation_companions():
    companions=load('generate_decks').companions
    ngw=companions('config_ngw_scheme','.true.')
    assert ngw['config_gwdo_scheme']=='bl_ugwp_gwdo'
    assert ngw['config_pbl_scheme']=='bl_ysu'
    for value in ('cld_incidence','cld_fraction','cld_fraction_thompson'):
        cloud=companions('config_radt_cld_scheme',value)
        assert cloud['config_radt_lw_scheme']=='rrtmg_lw'
        assert cloud['config_radt_sw_scheme']=='rrtmg_sw'
    assert companions('config_radt_cld_scheme','cld_fraction_thompson')['config_microp_scheme']=='mp_thompson'
    for value in ('cu_tiedtke','cu_ntiedtke'):
        convection=companions('config_convection_scheme',value)
        assert convection['config_pbl_scheme']=='bl_ysu'
        assert convection['config_sfclayer_scheme']!='off'


def test_missing_outputs_never_pass(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    r=load('compare_runs').compare(a,b,Path('/unused'),'*.nc')
    assert not r['all_identical']
    assert not r['all_file_bytes_identical']


def test_word_comparator_preserves_signed_zero_and_nan_payload(tmp_path):
    import os
    import subprocess
    import json
    import numpy as np
    import pytest
    from netCDF4 import Dataset
    binary=os.environ.get('MPAS_WORD_COMPARE')
    if not binary:
        pytest.skip('set MPAS_WORD_COMPARE to the compiled Rust comparator')
    a=tmp_path/'a.nc';b=tmp_path/'b.nc'
    for path, bits in ((a,[0,0x7fc00001]),(b,[0x80000000,0x7fc00002])):
        with Dataset(path,'w',format='NETCDF3_64BIT_OFFSET') as d:
            d.createDimension('cell',2)
            d.createVariable('f','f4',('cell',))[:]=np.asarray(bits,dtype=np.uint32).view(np.float32)
            d.createVariable('double','f8',('cell',))[:]=[1.0,2.0]
    p=subprocess.run([binary,str(a),str(b)],capture_output=True,text=True)
    assert p.returncode==1
    r=json.loads(p.stdout)
    assert r['differing_words']==2
    assert r['words']==4
    assert next(f for f in r['fields'] if f['field']=='f')['first_difference']['index']==[0]


def test_word_comparator_rejects_missing_variable(tmp_path):
    import os
    import subprocess
    import json
    import pytest
    from netCDF4 import Dataset
    binary=os.environ.get('MPAS_WORD_COMPARE')
    if not binary:
        pytest.skip('set MPAS_WORD_COMPARE to the compiled Rust comparator')
    a=tmp_path/'a.nc';b=tmp_path/'b.nc'
    for path,name in ((a,'f'),(b,'g')):
        with Dataset(path,'w') as d:
            d.createVariable(name,'f4')[...]=1.0
    p=subprocess.run([binary,str(a),str(b)],capture_output=True,text=True)
    assert p.returncode==1
    assert json.loads(p.stdout)['errors']


def test_cache_rejects_unaligned_surface_times_before_native_work(tmp_path,monkeypatch):
    import pytest
    monkeypatch.syspath_prepend(str(TOOLS))
    module=load('cache_cases')
    row={'kind':'global','start':'2023-01-01_00:00:00','stop':'2023-01-02_00:00:00',
         'surface_stop':'2023-01-02_01:00:00','interval_seconds':21600,
         'levels':55,'fg_levels':34,'grid':'missing','graph':'missing','gribs':[]}
    with pytest.raises(ValueError,match='surface_stop must align'):
        module.validate(row,'missing','missing','missing',tmp_path)

def test_native_modis_input_rejects_lake_category_before_initialization(tmp_path,monkeypatch):
    import pytest
    monkeypatch.syspath_prepend(str(TOOLS))
    module=load('cache_cases')
    index=tmp_path/'modis_landuse_20class_30s/index';index.parent.mkdir()
    base='category_min=1\niswater=17\nmminlu="MODIFIED_IGBP_MODIS_NOAH"\n'
    index.write_text(base+'category_max=21\nislake=21\n')
    with pytest.raises(ValueError,match='category21 aborts Noah'):
        module.validate_geography(tmp_path)
    index.write_text(base+'category_max=20\n')
    assert module.validate_geography(tmp_path)=={str(index):module.sha(index)}

def test_case9_matrix_waits_for_regional_grid_and_graph(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    row={'grid':'regional.init.nc','graph':'regional.graph.info','gribs':['analysis.grib'],
         'stages':[9],'ranks':4}
    assert load('cache_matrix').dependencies(row)==[
        'regional.init.nc','regional.graph.info','analysis.grib','regional.graph.info.part.4']

def test_case8_inactive_output_cannot_alias_native_initialization_input(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    module=load('cache_cases')
    for case in [7,8,9]:
        input_name='selected.init.nc' if case==8 else 'selected.grid.nc'
        streams=module.streams_for_case('selected',case,input_name,21600)
        names=[stream.get('filename_template') for stream in streams]
        assert len(set(names))==len(names)
        if case==7:
            assert next(stream.get('filename_template') for stream in streams
                        if stream.get('name')=='output')=='selected.init.nc'

def test_hourly_forecast_refuses_a_timestep_that_skips_sample_clocks():
    import pytest
    validate=load('generate_decks').validate_cadence
    with pytest.raises(ValueError,match='divide the requested forecast clock'):
        validate('1_00:00:00','0_01:00:00',1440)
    validate('1_00:00:00','0_01:00:00',1200)
    validate('5_00:00:00','0_03:00:00',720)

def test_native_product_witness_rejects_stored_clock_hidden_by_filename(tmp_path,monkeypatch):
    import json
    import os
    import numpy as np
    import pytest
    from netCDF4 import Dataset
    monkeypatch.syspath_prepend(str(TOOLS))
    binary=os.environ.get('MPAS_WORD_COMPARE')
    if not binary:pytest.skip('requires Rust inventory door')
    module=load('verify_products');hashes={}
    for minute in (0,6):
        name=f'output.2023-01-01_00.{minute:02}.00.nc'
        clock=f'2023-01-01_00:{(7 if minute else 0):02}:00'
        with Dataset(tmp_path/name,'w',format='NETCDF3_64BIT_OFFSET') as d:
            d.createDimension('Time',None);d.createDimension('StrLen',19)
            d.createVariable('xtime','S1',('Time','StrLen'))[0,:]=np.frombuffer(clock.encode(),dtype='S1')
        hashes[name]=module.sha(tmp_path/name)
    (tmp_path/'streams.atmosphere').write_text('<streams><stream name="output" type="output" '
        'filename_template="output.$Y-$M-$D_$h.$m.$s.nc" output_interval="0_00:06:00"/></streams>')
    (tmp_path/'MANIFEST.json').write_text(json.dumps({'schema':'mpas.native-forecast-cache/v1',
        'outputs':hashes,'runtime':{'start':'2023-01-01_00:00:00','duration':'0_00:06:00'}}))
    result=module.verify(tmp_path,Path(binary),[])
    assert not result['complete']
    assert result['errors']==['stored native clock differs: output.2023-01-01_00.06.00.nc']


def test_generated_streams_use_explicit_mesh_and_enable_surface(tmp_path):
    import xml.etree.ElementTree as ET
    source=tmp_path/'source'
    registry=source/'src/core_atmosphere/Registry.xml';registry.parent.mkdir(parents=True)
    registry.write_text('<registry><nml_option name="config_pbl_scheme" type="character" '
                        'possible_values="\'off\' \'bl_mynn\'"/>'
                        '<nml_option name="config_oml1d" type="logical"/></registry>')
    control=source/'src/core_atmosphere/physics/mpas_atmphys_control.F';control.parent.mkdir()
    control.write_text("if (trim(config_physics_suite) == 'none') then\n"
                       " config_pbl_scheme = 'off'\n else\n end subroutine physics_namelist_check")
    template=tmp_path/'streams'
    template.write_text('<streams><immutable_stream name="input" type="input" '
                        'filename_template="old.init.nc"/><stream name="surface" type="input" '
                        'filename_template="old.sfc_update.nc"/><immutable_stream name="ugwp_oro_data_in" '
                        'type="input" filename_template="old.ugwp_oro_data.nc"/>'
                        '<stream name="output" type="output"><file name="stream_list.atmosphere.output"/></stream></streams>')
    stream_list=tmp_path/'stream_list.atmosphere.output';stream_list.write_text('theta\nqv\n')
    load('generate_decks').generate(source,tmp_path/'decks','2023-01-01_00:00:00','1_00:00:00',
                                   '0_01:00:00',600,'chosen','explicit.init.nc',template)
    streams={s.get('name'):s.attrib for s in ET.parse(tmp_path/'decks/suite-none/streams.atmosphere').getroot()}
    assert streams['input']['filename_template']=='explicit.init.nc'
    assert streams['ugwp_oro_data_in']['filename_template']=='chosen.ugwp_oro_data.nc'
    assert streams['surface']['type']=='input;output'
    assert streams['surface']['output_interval']=='0_01:00:00'
    assert streams['surface']['input_interval']=='none'
    index=json.loads((tmp_path/'decks/INDEX.json').read_text())
    active=next(entry for entry in index['decks'] if entry['deck']=='oml1d-true')
    assert active['companions']=={'config_sfclayer_scheme':'sf_monin_obukhov_rev'}
    assert (tmp_path/'decks/oml1d-true/stream_list.atmosphere.output').read_bytes()==stream_list.read_bytes()
    assert active['stream_list_sha256']['stream_list.atmosphere.output']==load('generate_decks').sha(stream_list)


def test_generated_native_variants_preserve_numeric_options_and_closed_gate(tmp_path):
    tool=load('generate_decks')
    source=site_support.mpas_source()
    if not source.exists():
        import pytest
        pytest.skip('pinned MPAS registry unavailable')
    template=tmp_path/'streams';template.write_text('<streams><immutable_stream name="input" type="input" filename_template="old.init.nc"/></streams>')
    index=tool.generate(source,tmp_path/'decks','2023-01-01_00:00:00','1_00:00:00','0_01:00:00',600,'mesh','mesh.init.nc',template,TOOLS/'deck_variants.v842.json')
    variants=[entry for entry in index['decks'] if entry['deck'].startswith('oml-hml')]
    assert len(variants)==54
    assert {entry['override']['config_oml_hml0'] for entry in variants}=={30.,0.,-1.}
    assert {entry['override']['config_oml_relaxation_time'] for entry in variants}=={0.,864000.}
    body=(tmp_path/'decks/oml-hml0-gamma0.14-relax0-none/namelist.atmosphere').read_text()
    assert 'config_oml1d = .true.' in body and 'config_oml_hml0 = 0.0' in body
    closed=next(entry for entry in index['decks'] if entry['deck']=='oml-driver-gate-closed-none')
    assert closed['companions']=={}
    assert index['companion_table_sha256']==tool.sha(TOOLS/'deck_companions.v842.json')
    for prefix in ('ngw-driver-gate-closed','gwdo-driver-gate-closed-bl_ysu_gwdo','gwdo-driver-gate-closed-bl_ugwp_gwdo'):
        assert next(entry for entry in index['decks'] if entry['deck']==prefix+'-none')['companions']=={}


def test_acceptance_rejects_partial_and_missing_sample_times():
    verify=load('verify_acceptance')
    receipt={'schema':'gpuwm-hex.identity-extend/v1','partial':True,'native_finished':False,
             'all_identical':True,'cells':10,'levels':55,'init_sha256':'pin','rows':[]}
    errors=verify.validate(receipt,[6,12,18,24],10,55,'pin')
    assert 'native comparison incomplete' in errors
    assert 'required output times not compared' in errors


def test_acceptance_rejects_zero_word_comparison():
    verify=load('verify_acceptance')
    receipt={'schema':'gpuwm-hex.identity-extend/v1','native_finished':True,
             'all_identical':True,'cells':10,'levels':55,'init_sha256':'pin',
             'rows':[{'hours':6,'differing_words_total':0,
                      'fields':{name:{'identical':True,'words':0} for name in verify.FIELDS}}]}
    assert 'field comparison failed at 6 h' in verify.validate(receipt,[6],10,55,'pin')


def test_publication_resume_rejects_corrupt_cache(tmp_path, monkeypatch):
    import json
    import pytest
    monkeypatch.syspath_prepend(str(TOOLS))
    support=load('matrix_support')
    source=tmp_path/'scratch';source.mkdir()
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'output.nc').write_bytes(b'original native bytes')
    (cache/'PUBLICATION.json').write_text(json.dumps({
        'source':str(source),'files':{'output.nc':support.sha(cache/'output.nc')}}))
    assert support.published(cache,source)['existing']
    (cache/'output.nc').write_bytes(b'corrupt bytes')
    with pytest.raises(ValueError,match='published file hash differs'):
        support.published(cache,source)


def test_scratch_cleanup_preserves_unmatched_dumps_and_symlink_inputs(tmp_path, monkeypatch):
    import pytest
    monkeypatch.syspath_prepend(str(TOOLS))
    support=load('matrix_support')
    source=tmp_path/'scratch';source.mkdir()
    cache=tmp_path/'cache';cache.mkdir()
    (source/'output.nc').write_bytes(b'native bytes')
    (cache/'output.nc').write_bytes(b'different bytes')
    with pytest.raises(ValueError,match='published dump differs'):
        support.discard_copied_outputs(source,cache,['output.nc'])
    assert (source/'output.nc').exists()
    (cache/'output.nc').write_bytes(b'native bytes')
    (source/'input.nc').symlink_to(cache/'output.nc')
    support.discard_copied_outputs(source,cache,['output.nc','input.nc'])
    assert not (source/'output.nc').exists()
    assert (source/'input.nc').is_symlink()
    assert (cache/'output.nc').read_bytes()==b'native bytes'

def test_word_comparator_detects_integer_and_character_differences_after_batch(tmp_path):
    import os
    import subprocess
    import json
    import numpy as np
    import pytest
    from netCDF4 import Dataset
    binary=os.environ.get('MPAS_WORD_COMPARE')
    if not binary:
        pytest.skip('requires built Rust comparator')
    paths=[tmp_path/'a.nc',tmp_path/'b.nc']
    for index,path in enumerate(paths):
        with Dataset(path,'w',format='NETCDF3_64BIT_OFFSET') as dataset:
            dataset.createDimension('cell',300000)
            dataset.createDimension('character',1)
            data=np.arange(300000,dtype=np.int32)
            if index:data[262145]=-7
            dataset.createVariable('integer','i4',('cell',))[:]=data
            dataset.createVariable('character','S1',('character',))[:]=np.array([b'B' if index else b'A'],dtype='S1')
            dataset.createVariable('unchanged_float','f4')[...]=1.0
    process=subprocess.run([binary,*map(str,paths)],capture_output=True,text=True)
    assert process.returncode==1
    result=json.loads(process.stdout)
    assert result['all_floating_words_identical']
    assert result['differing_words']==2
    integer=next(field for field in result['fields'] if field['field']=='integer')
    assert integer['first_difference']['index']==[262145]
    assert integer['first_difference']['other_bits']=='00000000fffffff9'


def test_forecast_resume_rejects_changed_executable(tmp_path,monkeypatch):
    import pytest
    monkeypatch.syspath_prepend(str(TOOLS))
    module=load('run_matrix')
    build=tmp_path/'build';build.mkdir()
    deck=tmp_path/'deck';deck.mkdir()
    (build/'atmosphere_model').write_bytes(b'pinned executable')
    for name in ('namelist.atmosphere','streams.atmosphere'):
        (deck/name).write_text(name)
    row={'build':str(build),'deck':str(deck),'pin_set_hash':'pin','ranks':1}
    manifest={'pin_set_hash':'pin','ranks':1,'deck':'deck',
        'binary_sha256':module.sha(build/'atmosphere_model'),
        'namelist_sha256':module.sha(deck/'namelist.atmosphere'),
        'streams_sha256':module.sha(deck/'streams.atmosphere')}
    module.validate_manifest(manifest,row)
    (build/'atmosphere_model').write_bytes(b'another executable')
    with pytest.raises(ValueError,match='completed forecast contract differs'):
        module.validate_manifest(manifest,row)


def test_rank_witness_binds_completed_inputs_ranks_and_output_hashes(tmp_path,monkeypatch):
    import json
    import os
    import pytest
    from netCDF4 import Dataset
    monkeypatch.syspath_prepend(str(TOOLS))
    binary=os.environ.get('MPAS_WORD_COMPARE')
    if not binary:pytest.skip('requires built Rust comparator')
    module=load('native_rank_identity');support=load('matrix_support')
    folders=[tmp_path/'rank1',tmp_path/'rank24']
    for folder,ranks in zip(folders,[1,24]):
        folder.mkdir()
        with Dataset(folder/'output.nc','w',format='NETCDF3_64BIT_OFFSET') as dataset:
            dataset.createVariable('field','f4')[...]=1.0
        inputs={'atmosphere_model':'same','input.nc':'same'}
        if ranks>1:inputs[f'mesh.graph.info.part.{ranks}']='partition'
        manifest={'schema':'mpas.native-forecast-cache/v1','pin_set_hash':'pin',
            'deck':'suite','ranks':ranks,'runtime':{'duration':'1_00:00:00'},
            'binary_sha256':'same','namelist_sha256':'nml','streams_sha256':'streams',
            'inputs':inputs,'outputs':{'output.nc':support.sha(folder/'output.nc')}}
        (folder/'MANIFEST.json').write_text(json.dumps(manifest))
    assert module.witness(*folders,Path(binary))['all_identical']
    path=folders[1]/'MANIFEST.json';manifest=json.loads(path.read_text())
    manifest['inputs']['input.nc']='different initialization';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='native inputs differ'):
        module.witness(*folders,Path(binary))
    manifest['inputs']['input.nc']='same';manifest['outputs']['output.nc']='stale hash'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='output differs from native manifest'):
        module.witness(*folders,Path(binary))


def test_native_initialization_witness_rejects_noah_category_and_wrong_levels(tmp_path,monkeypatch):
    import json
    import os
    import numpy as np
    import pytest
    from netCDF4 import Dataset
    monkeypatch.syspath_prepend(str(TOOLS))
    binary=os.environ.get('MPAS_WORD_COMPARE')
    if not binary:pytest.skip('requires Rust inventory/range doors')
    module=load('verify_products');start='2023-01-01_00:00:00'
    static=tmp_path/'mesh.ugwp_oro_data.nc'
    with Dataset(static,'w',format='NETCDF3_64BIT_OFFSET') as dataset:
        dataset.createDimension('nCells',2)
        dataset.createVariable('orography','f4',('nCells',))[:]=[0,0]
    initial=tmp_path/'mesh.init.nc'
    def write_initial(category,levels):
        with Dataset(initial,'w',format='NETCDF3_64BIT_OFFSET') as dataset:
            for dim,size in [('nCells',2),('nVertLevels',levels),('nVertLevelsP1',levels+1),
                             ('nSoilLevels',4),('Time',None),('StrLen',19)]:
                dataset.createDimension(dim,size)
            dataset.createVariable('ivgtyp','i4',('nCells',))[:]=[17,category]
            dataset.createVariable('theta','f4',('Time','nCells','nVertLevels'))[0,:,:]=300
            dataset.createVariable('zgrid','f4',('nCells','nVertLevelsP1'))[:]=0
            dataset.createVariable('soil','f4',('nCells','nSoilLevels'))[:]=0
            dataset.createVariable('xtime','S1',('Time','StrLen'))[0,:]=np.frombuffer(start.encode(),dtype='S1')
        (tmp_path/'MANIFEST.json').write_text(json.dumps({'schema':'mpas.native-input-cache/v1',
            'case':{'mesh':'mesh','start':start,'stop':start,'interval_seconds':21600,
                    'levels':55,'stages':[7]},
            'files':{path.name:module.sha(path) for path in (initial,static)}}))
    write_initial(21,55)
    assert module.verify(tmp_path,Path(binary),[])['errors']==[
        'native Noah vegetation category exceeds20: mesh.init.nc']
    write_initial(20,55)
    assert module.verify(tmp_path,Path(binary),[])['complete']
    write_initial(20,49)
    assert not module.verify(tmp_path,Path(binary),[])['complete']


def test_forecast_catalog_refuses_stale_clock_proof(tmp_path,monkeypatch):
    import hashlib
    import json
    import pytest
    monkeypatch.syspath_prepend(str(TOOLS))
    module=load('index_forecasts');support=load('matrix_support')
    build=tmp_path/'build';build.mkdir();deck=tmp_path/'deck';deck.mkdir()
    output=tmp_path/'published';output.mkdir();details=tmp_path/'clocks';details.mkdir()
    (build/'atmosphere_model').write_bytes(b'fixture executable')
    (build/'MANIFEST.json').write_text(json.dumps({'recipe':'strict-fixture'}))
    for name in ['namelist.atmosphere','streams.atmosphere']:(deck/name).write_text(name)
    runtime={'mesh':'selected','start':'explicit','duration':'declared'}
    (tmp_path/'INDEX.json').write_text(json.dumps({'runtime':runtime}))
    row={'build':str(build),'deck':str(deck),'publish':str(output),'pin_set_hash':'pin','ranks':1}
    manifest={'schema':'mpas.native-forecast-cache/v1','pin_set_hash':'pin','ranks':1,'deck':'deck',
              'binary_sha256':support.sha(build/'atmosphere_model'),'runtime':runtime,
              'namelist_sha256':support.sha(deck/'namelist.atmosphere'),
              'streams_sha256':support.sha(deck/'streams.atmosphere'),'outputs':{'output.nc':'fixture-digest'}}
    (output/'MANIFEST.json').write_text(json.dumps(manifest));digest=support.sha(output/'MANIFEST.json')
    (output/'PUBLICATION.json').write_text(json.dumps({'files':{'MANIFEST.json':digest,**manifest['outputs']}}))
    assert module.entry(row,details)['status']=='awaiting-clock-witness'
    clock=details/(hashlib.sha256(str(output).encode()).hexdigest()[:16]+'.json')
    proof={'directory':str(output),'manifest_sha256':digest,'complete':True,'errors':[],
           'files':[{'file':'output.nc','sha256':'fixture-digest'}]}
    clock.write_text(json.dumps(proof));assert module.entry(row,details)['status']=='native-reference-complete'
    proof['manifest_sha256']='stale proof';clock.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='another manifest'):module.entry(row,details)


def test_foundation_gate_preserves_pending_and_rejects_stale_or_corrupt_product(tmp_path,monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(TOOLS));module=load('verify_foundation')
    directory=tmp_path/'product';directory.mkdir();receipt=tmp_path/'clock.json'
    row={'directory':str(directory),'receipt':str(receipt)}
    index={'acceptance':[],'flags':[],'decks':[],'products':[row],'identity':[]}
    assert module.collect(index)['checks'][0]['status']=='pending'
    data=directory/'output.nc';data.write_bytes(b'opaque native data hashed without Python decoding')
    digest=module.sha(data)
    (directory/'MANIFEST.json').write_text(json.dumps({'files':{'output.nc':digest}}))
    manifest_digest=module.sha(directory/'MANIFEST.json')
    (directory/'PUBLICATION.json').write_text(json.dumps({'files':{'MANIFEST.json':manifest_digest,'output.nc':digest}}))
    clock={'directory':str(directory),'manifest_sha256':manifest_digest,'complete':True,'errors':[],
           'files':[{'file':'output.nc','sha256':digest}]}
    receipt.write_text(json.dumps(clock));assert module.collect(index)['complete']
    clock['manifest_sha256']='stale';receipt.write_text(json.dumps(clock))
    assert module.collect(index)['checks'][0]['status']=='failed'
    clock['manifest_sha256']=manifest_digest;receipt.write_text(json.dumps(clock));data.write_bytes(b'changed data')
    assert not module.collect(index)['complete']


def test_foundation_gate_rejects_empty_identity_and_incomplete_external_audit(tmp_path,monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(TOOLS));module=load('verify_foundation')
    proof=tmp_path/'identity.json';runs=[tmp_path/'rank1',tmp_path/'rank24']
    for run in runs:
        run.mkdir();(run/'MANIFEST.json').write_text(json.dumps({'outputs':{}}))
    proof.write_text(json.dumps({'reference':str(runs[0]),'other':str(runs[1]),
        'files':{},'errors':[],'all_identical':True,'all_file_bytes_identical':True}))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'all_ok':True,'errors':[],
        'objects':[{'flags_ok':True}],'groups':{'physics_mmm/':1}}))
    index={'acceptance':[],'flags':[str(audit)],'decks':[],'products':[],
           'identity':[{'receipt':str(proof),'runs':list(map(str,runs))}]}
    result=module.collect(index)
    assert not result['complete'];assert all(row['status']=='failed' for row in result['checks'])


def test_forecast_controller_can_join_an_inspected_live_run_without_launching(tmp_path,monkeypatch):
    import json
    import sys
    import pytest
    monkeypatch.syspath_prepend(str(TOOLS));module=load('run_matrix')
    case=tmp_path/'case';case.mkdir();(case/'PUBLICATION.json').write_text('{}')
    (case/'input.nc').write_bytes(b'fixture input')
    deck=tmp_path/'deck';deck.mkdir()
    (deck/'streams.atmosphere').write_text('<streams><immutable_stream name="input" filename_template="input.nc"/></streams>')
    out=tmp_path/'active';out.mkdir();sentinel=out/'live-run.log';sentinel.write_text('preserve active native work')
    index=tmp_path/'jobs.json';index.write_text(json.dumps({'jobs':[{'case':str(case),'deck':str(deck),
         'out':str(out),'publish':str(tmp_path/'published'),'build':'unused','tables':'unused',
         'mpi':'unused','ranks':1,'pin_set_hash':'explicit'}]}))
    argv=['run_matrix.py',str(index),'--ready',str(tmp_path/'READY.md'),'--receipt',str(tmp_path/'receipt.json')]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='unfinished forecast needs inspection'):module.main()
    class WaitingForLiveRun(Exception):pass
    def wait(seconds):raise WaitingForLiveRun()
    monkeypatch.setattr(module.time,'sleep',wait)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:pytest.fail('must not duplicate the live forecast'))
    monkeypatch.setattr(sys,'argv',argv+['--join-existing'])
    with pytest.raises(WaitingForLiveRun):module.main()
    assert sentinel.read_text()=='preserve active native work'


def test_shared_surface_table_excludes_pressure_fields_and_keeps_native_inputs(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    source=tmp_path/'Vtable'
    source.write_text('header\ncolumns\nseparator\n 11 | 100 | * | | TT | K | temp |\n 11 | 1 | 0 | | SKINTEMP | K | skin |\n 91 | 1 | 0 | | SEAICE | fraction | ice |\n 81 | 1 | 0 | | LANDSEA | fraction | mask |\n')
    text=load('shared_ungrib').surface_vtable(source)
    assert 'TT' not in text
    assert all(name in text for name in ['SKINTEMP','SEAICE','LANDSEA'])
    source.write_text(source.read_text().replace('SEAICE','OTHER'))
    import pytest
    with pytest.raises(ValueError,match='misses required'):load('shared_ungrib').surface_vtable(source)


def test_shared_decode_requires_completed_hashes_and_source_provenance(tmp_path,monkeypatch):
    import json,pytest
    monkeypatch.syspath_prepend(str(TOOLS));module=load('cache_cases')
    name='GFS:2001-01-01_00';source=tmp_path/name;source.write_bytes(b'native WPS file')
    manifest={'schema':'mpas.shared-ungrib/v1','files':{name:module.sha(source)},'sources':{name:{'sha256':'raw'}}}
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    row={'shared_ungrib':str(tmp_path),'start':'2001-01-01_00:00:00','stop':'2001-01-01_00:00:00','surface_stop':'2001-01-01_00:00:00','interval_seconds':21600}
    assert module.shared_intermediates(row)==[source]
    source.write_bytes(b'changed input')
    with pytest.raises(ValueError,match='changed'):module.shared_intermediates(row)
    source.unlink()
    with pytest.raises(FileNotFoundError):module.shared_intermediates(row)


def test_shared_decode_preserves_unrecognized_interrupted_stage(tmp_path,monkeypatch):
    import pytest
    monkeypatch.syspath_prepend(str(TOOLS));module=load('shared_ungrib')
    source=tmp_path/'source.grib';source.write_bytes(b'encoded-source')
    vtable=tmp_path/'Vtable';vtable.write_text('explicit decoder table')
    directory=tmp_path/'shared';directory.mkdir()
    stage=directory/'.pending-GFS_2001-01-01_00';stage.mkdir()
    sentinel=stage/'unrelated-file';sentinel.write_bytes(b'preserve unrelated bytes')
    with pytest.raises(ValueError,match='unrecognized interrupted native decoder file'):
        module.decode(source,'2001-01-01_00:00:00',directory,'/bin/true',vtable)
    assert sentinel.read_bytes()==b'preserve unrelated bytes'
    assert source.read_bytes()==b'encoded-source'
    assert not (directory/'MANIFEST.json').exists()

def test_native_io_error_cannot_pass_a_completion_banner(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS));module=load('native_completion')
    assert module.completed('WARNING: informational\nFinished running the atmosphere core','atmosphere')
    assert not module.completed('ERROR: SMIOLf_put_var failed with error -5\nFinished running the atmosphere core','atmosphere')
    assert not module.completed('Finished running the atmosphere core','init_atmosphere')


def test_raw_field_digests_preserve_nan_payload_sign_and_multidimensional_order(tmp_path):
    import hashlib,json,os,subprocess
    import numpy as np,pytest
    from netCDF4 import Dataset
    binary=os.environ.get('MPAS_WORD_COMPARE')
    if not binary:pytest.skip('requires built Rust word reader')
    path=tmp_path/'native-words.nc'
    integers=np.arange(3*300005,dtype=np.int32).reshape(3,300005)
    words=np.array([0x00000000,0x80000000,0x7fc00001,0x7fc00002],dtype=np.uint32)
    with Dataset(path,'w',format='NETCDF3_64BIT_OFFSET') as dataset:
        dataset.createDimension('row',3);dataset.createDimension('column',300005);dataset.createDimension('special',4)
        dataset.createVariable('integers','i4',('row','column'))[:]=integers
        dataset.createVariable('floats','f4',('special',))[:]=words.view(np.float32)
    result=json.loads(subprocess.check_output([binary,'--field-digests',str(path)],text=True))
    fields={field['field']:field for field in result['variables']}
    assert fields['integers']['words']==integers.size
    assert fields['integers']['word_sha256']==hashlib.sha256(integers.tobytes(order='C')).hexdigest()
    assert fields['floats']['word_sha256']==hashlib.sha256(words.tobytes()).hexdigest()


def test_disposable_native_output_waits_for_finished_clock_and_preserves_bad_clock(tmp_path,monkeypatch):
    import json,os
    import numpy as np,pytest
    from netCDF4 import Dataset
    import xml.etree.ElementTree as ET
    binary=os.environ.get('MPAS_WORD_COMPARE')
    if not binary:pytest.skip('requires built Rust word reader')
    monkeypatch.syspath_prepend(str(TOOLS));module=load('retire_native_outputs')
    runtime={'start':'2001-01-01_00:00:00','duration':'00_00:20:00'}
    streams=ET.fromstring('<streams><stream name="output" type="output" filename_template="output.$Y-$M-$D_$h.$m.$s.nc" output_interval="00_00:20:00" /></streams>')
    writer=module.RetireOutputs(tmp_path,runtime,streams,Path(binary),[])
    def output(name,clock):
        with Dataset(tmp_path/name,'w',format='NETCDF3_64BIT_OFFSET') as dataset:
            dataset.createDimension('Time',None);dataset.createDimension('StrLen',64)
            dataset.createVariable('xtime','S1',('Time','StrLen'))[0,:]=np.frombuffer(clock.ljust(64).encode(),dtype='S1')
            dataset.createVariable('theta','f4',('Time',))[0]=300
    first='output.2001-01-01_00.00.00.nc';last='output.2001-01-01_00.20.00.nc'
    output(first,runtime['start']);log=tmp_path/'log.atmosphere.0000.out';log.write_text('Begin timestep 2001-01-01_00:00:00\n')
    writer.collect();assert (tmp_path/first).exists() and not writer.receipt.exists()
    log.write_text('Begin timestep 2001-01-01_00:20:00\n');writer.collect()
    assert not (tmp_path/first).exists() and json.loads(writer.receipt.read_text())['files'][first]['inventory']['variables']
    output(last,'2001-01-01_00:10:00')
    with pytest.raises(ValueError,match='stored clocks differ'):writer.collect(finished=True)
    assert (tmp_path/last).exists()


def test_field_inventory_preserves_native_types_units_and_unknown_fields(tmp_path,monkeypatch):
    import os,subprocess
    import numpy as np,pytest
    from netCDF4 import Dataset
    binary=os.environ.get('MPAS_WORD_COMPARE')
    if not binary:pytest.skip('requires built Rust word reader')
    monkeypatch.syspath_prepend(str(TOOLS));module=load('field_inventory')
    source=tmp_path/'source';registry=source/module.REGISTRY;registry.parent.mkdir(parents=True)
    registry.write_text('<registry><var_struct name="state"><var_array name="scalars" type="real" dimensions="nCells nVertLevels Time"><var name="qv" units="kg kg^{-1}"/></var_array></var_struct></registry>')
    directory=tmp_path/'publication';directory.mkdir();samples=tmp_path/'restored';samples.mkdir()
    name='output.2001-01-01_00.00.00.nc';path=samples/name
    with Dataset(path,'w',format='NETCDF3_64BIT_OFFSET') as dataset:
        dataset.createDimension('Time',1);dataset.createDimension('StrLen',64)
        dataset.createVariable('xtime','S1',('Time','StrLen'))[0,:]=np.frombuffer('2001-01-01_00:00:00'.ljust(64).encode(),dtype='S1')
        dataset.createVariable('qv','f4',('Time',))[0]=.01
        dataset.createVariable('unregistered','f4',('Time',))[0]=1
    (directory/'streams.atmosphere').write_text('<streams><stream name="output" type="output"/></streams>')
    manifest={'outputs':{name:module.sha(path)},'ranks':1,'runtime':{'duration':'01_00:00:00'}}
    (directory/'MANIFEST.json').write_text(json.dumps(manifest))
    (directory/'PUBLICATION.json').write_text(json.dumps({'files':{'MANIFEST.json':module.sha(directory/'MANIFEST.json'),'streams.atmosphere':module.sha(directory/'streams.atmosphere'),name:module.sha(path)}}))
    result=module.create(directory,source,Path(binary),samples)
    rows={row['field']:row for row in result['streams']['output']}
    assert rows['qv']['native_declaration']['type']=='real'
    assert rows['qv']['native_declaration']['units']=='kg kg^{-1}'
    assert rows['unregistered']['native_declaration'] is None
    assert result['prognostic_fields']=={'output':['qv']}
    assert not result['water_integral_complete'] and not result['identity_claim']
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='sample hash differs'):
        module.create(directory,source,Path(binary),samples)


def test_frozen_download_requires_exact_public_pin_before_transport(tmp_path,monkeypatch):
    import pytest
    monkeypatch.syspath_prepend(str(TOOLS));module=load('frozen_download')
    pin={'url':'https://example.invalid/data','bytes':1,'sha256':'missing'}
    with pytest.raises(ValueError,match='SHA256 pin required'):
        module.fetch(pin,tmp_path/'out',tmp_path/'stage')
    assert not (tmp_path/'stage').exists() and not (tmp_path/'out').exists()


def test_explicit_zero_copy_preserves_every_other_native_word(tmp_path):
    import os,subprocess
    import numpy as np
    import pytest
    from netCDF4 import Dataset
    copier=os.environ.get('MPAS_ZERO_FIELDS');comparator=os.environ.get('MPAS_WORD_COMPARE')
    if not copier or not comparator:pytest.skip('real Rust input copier/comparator required')
    source=tmp_path/'source.nc';target=tmp_path/'target.nc'
    with Dataset(source,'w',format='NETCDF3_64BIT_OFFSET') as d:
        d.createDimension('n',4)
        d.createVariable('selected','f4',('n',))[:]=np.array([1.,-2.,3.,-4.],dtype='f4')
        d.createVariable('selected64','f8',('n',))[:]=[1.,2.,3.,4.]
        d.createVariable('untouched','f4',('n',))[:]=np.array([0x80000000,0x7fc00123,1,0x3f800000],dtype='u4').view('f4')
        d.createVariable('integer','i4',('n',))[:]=[1,2,3,4]
        d.source='unchanged native header'
    source.chmod(0o444)
    subprocess.run([copier,str(source),str(target),'selected','selected64'],check=True)
    r=json.loads(subprocess.run([comparator,str(source),str(target)],text=True,capture_output=True).stdout)
    assert {f['field'] for f in r['fields'] if f['differing_words']}=={'selected','selected64'}
    with Dataset(target) as d:
        assert d.source=='unchanged native header'
        assert not np.asarray(d['selected'][:]).view('u4').any()
        assert not np.asarray(d['selected64'][:]).view('u8').any()
    bad=tmp_path/'bad.nc'
    p=subprocess.run([copier,str(source),str(bad),'integer'],capture_output=True,text=True)
    assert p.returncode==2 and not bad.exists()


def test_shared_restore_requires_original_encoded_source_before_decoder(tmp_path,monkeypatch):
    import json,pytest
    monkeypatch.syspath_prepend(str(TOOLS));module=load('shared_ungrib')
    source=tmp_path/'source.grib';source.write_bytes(b'changed encoded source')
    vtable=tmp_path/'Vtable';vtable.write_text('explicit decoder table')
    directory=tmp_path/'shared';directory.mkdir();name='GFS:2001-01-01_00'
    manifest={'schema':'mpas.shared-ungrib/v1','ungrib_sha256':module.sha(Path('/bin/true')),
              'vtable_sha256':module.sha(vtable),'files':{name:'original-native-output'},
              'sources':{name:{'sha256':'original-encoded-source','bytes':10}}}
    encoded=json.dumps(manifest);(directory/'MANIFEST.json').write_text(encoded)
    with pytest.raises(ValueError,match='missing or changed'):
        module.decode(source,'2001-01-01_00:00:00',directory,'/bin/true',vtable)
    with pytest.raises(ValueError,match='restoration raw source differs'):
        module.decode(source,'2001-01-01_00:00:00',directory,'/bin/true',vtable,restore_missing=True)
    assert (directory/'MANIFEST.json').read_text()==encoded
    assert not (directory/name).exists()
    assert source.read_bytes()==b'changed encoded source'
