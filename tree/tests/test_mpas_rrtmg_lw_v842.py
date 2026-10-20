"""Pristine strict-GNU MPAS wrapper replay (run under the GPU queue).

Set MPAS_LW_ORACLE_DIR to the standalone Fortran output directory. These
synthetic fixtures supplement the campaign replay/forecast; they do not
replace acceptance. MPAS: module_ra_rrtmg_lw.F:11574-12535.
"""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import struct
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module')
def replay_receipt():
    result={'scope':'strict standalone MPAS columns; not whole-model acceptance',
            'compiler_options':['--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true'],
            'native_process_count':1, 'native_mpi_ranks':None,
            'columns':{},'real_columns':{},'initialization':{},'randoms':{},'mcica':{},'mcica_multi':{},'solver':{},'setcoef':{},'batch':{},'loader':{},'chunks':{},'vinterp':{},'f1_driver':{},'optional':{},'extended_solver':{}}
    yield result
    dest=os.environ.get('MPAS_LW_REPLAY_RECEIPT')
    if dest:
        from hexcore.mpasphys import rrtmg_lw_v842 as port
        result['module_sha256'] = hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest()
        source = (port._solver_source() + '\n' + port._RANDOM_SOURCE + '\n' +
                  port._DRIVER_SOURCE + '\n' + port._INIT_SOURCE)
        result['cuda_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        result['extended_cuda_source_sha256'] = hashlib.sha256(source.replace(
            '#define RLW_MAXLAY 128', '#define RLW_MAXLAY 129').encode()).hexdigest()
        directory=Path(os.environ['MPAS_LW_ORACLE_DIR'])
        result['fixtures']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.glob('native-*.bin')}
        result['total_differing_words']=sum(row['differing_words']
            for group in ('columns','real_columns','initialization','randoms','mcica','mcica_multi','solver','setcoef','batch','loader','chunks','vinterp','f1_driver','optional','extended_solver')
            for case in result[group].values() for row in case.values())
        Path(dest).write_text(json.dumps(result,indent=2)+'\n')


def read_native_fixture(path):
    """Test-only tagged binary reader; production decoding belongs to Rust."""
    data=Path(path).read_bytes();out={};offset=0
    while offset<len(data):
        n=struct.unpack_from('<i',data,offset)[0];offset+=4
        name=data[offset:offset+n].decode('ascii');offset+=n
        dtype,rank=struct.unpack_from('<ii',data,offset);offset+=8
        dims=struct.unpack_from('<'+'i'*rank,data,offset);offset+=4*rank
        count=int(np.prod(dims,dtype=np.int64)) if rank else 1
        array=np.frombuffer(data,dtype='<f4' if dtype==0 else '<i4',count=count,offset=offset)
        offset+=count*4
        out[name]=array.reshape(dims,order='F') if rank else array[0]
    return out


def read_rust_fixture(directory, cp):
    """Upload words already decoded and reordered by fixture_words.rs."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['order'] == 'C'
    result = {}
    for row in manifest['arrays']:
        data = (directory / row['file']).read_bytes()
        assert len(data) == row['words'] * 4
        dtype = '<f4' if row['dtype'] == 0 else '<i4'
        result[row['name']] = cp.asarray(np.frombuffer(data, dtype=dtype).reshape(row['shape']))
    return result


def test_native_short_mcica_continuations(oracle, replay_receipt):
    """MPAS: module_ra_rrtmg_lw.F:2432-2440; no minimum-four-layer refusal.

    The strict standalone boundary continues for all 278 supplied calls.
    Their complete outputs are stable on both hosts and at process counts
    one and two. The ten native STOP cases have no continuation claim.
    """
    path = os.environ.get('MPAS_LW_SHORT_EXPORTS')
    if not path:
        pytest.skip('Rust exports of the measured strict short-column matrix required')
    cp, port, K, _ = oracle
    manifest = json.loads((Path(path) / 'manifest.json').read_text())
    assert len(manifest['mcica']) == 278
    rows = []
    for entry in manifest['mcica']:
        meta = entry['controls']
        inputs = read_rust_fixture(entry['input']['directory'], cp)
        expected = read_rust_fixture(entry['output']['directory'], cp)
        actual = port.mcica_subcol_device(
            cp.ascontiguousarray(inputs['a']), cp.ascontiguousarray(inputs['surface']),
            cp.ascontiguousarray(inputs['tau'].transpose(1, 0, 2)), K,
            icld=meta['overlap'], irng=meta['rng_in'], permuteseed=150,
            idcor=meta['decorrelation'], julday=80 + (meta['scenario'] % 2) * 220)
        arrays = {}
        for name in ('cloud', 'liquid', 'ice', 'snow', 'optical_depth', 'radii'):
            native = expected[name]
            if name != 'radii':
                native = cp.ascontiguousarray(native.transpose(1, 2, 0))
            assert actual[name].shape == native.shape and actual[name].dtype == native.dtype
            differences = int(cp.count_nonzero(actual[name].view(cp.uint32) != native.view(cp.uint32)).item())
            arrays[name] = {'words': int(native.size), 'differing_words': differences}
        arrays['rng_out'] = {'words': 1,
            'differing_words': int(actual['irng'] != int(expected['rng_out'].item()))}
        rows.append({'controls': meta, 'arrays': arrays})
        assert all(a['differing_words'] == 0 for a in arrays.values()), rows[-1]
    replay_receipt['short_mcica_continuations'] = {
        'native_process_counts_measured': [1, 2], 'replay_process_count': 1,
        'cases': rows, 'differing_words': 0,
        'words': sum(a['words'] for row in rows for a in row['arrays'].values())}


def test_native_short_wrapper_continues(oracle, replay_receipt):
    """MPAS: module_ra_rrtmg_lw.F:11810-12305; continue native short calls.

    This test checks continuation and dimensions, and records every output
    difference. Native's one-layer wrapper varies between pristine runs;
    these columns have no bitwise grade and no words are excluded.
    """
    path = os.environ.get('MPAS_LW_SHORT_EXPORTS')
    if not path:
        pytest.skip('Rust exports of the measured strict short-column wrapper required')
    cp, port, K, _ = oracle
    manifest = json.loads((Path(path) / 'manifest.json').read_text())
    columns = read_rust_fixture(manifest['wrapper']['directory'], cp)
    rows = []
    for case in range(300):
        get = lambda name: columns[f'{case:03d}/{name}']
        flags = get('flags').get().tolist()
        actual = port.rrtmg_lwrad_device(
            cp.ascontiguousarray(get('model')[None]),
            cp.ascontiguousarray(get('interfaces')[None]),
            cp.ascontiguousarray(get('surface')[None]), K,
            icloud=flags[0], cldovrlp=flags[1], idcor=flags[2], julday=flags[3],
            has_reqc=flags[4], has_reqi=flags[5], has_reqs=flags[6], o3input=flags[7],
            pin=get('pin'), ozone=get('ozone')[None])
        values = {'flux': cp.stack([actual[name][0] for name in port.FLUX_FIELDS]),
                  'heating': actual['rthratenlw'][0]}
        values.update({name: actual[name][0] for name in ('rre_cloud', 'rre_ice', 'rre_snow')})
        values.update({name: actual['profiles'][0][name][0]
                       for name in ('uflx', 'uflxc', 'dflx', 'dflxc')})
        arrays = {}
        for name, value in values.items():
            expected = get(name)
            assert value.shape == expected.shape and value.dtype == expected.dtype
            unequal = value.view(cp.uint32) != expected.view(cp.uint32)
            arrays[name] = {'words': int(value.size),
                'differing_words': int(cp.count_nonzero(unequal).item()),
                'different_indices_c_zero_based': cp.flatnonzero(unequal.ravel()).get().tolist()}
        rows.append({'case': case, 'arrays': arrays})
    replay_receipt['short_wrapper_continuation_diagnostic'] = {
        'cases': rows, 'native_process_count': 1, 'replay_process_count': 1,
        'bitwise_grade': False, 'words_excluded': False,
        'differing_words': sum(a['differing_words'] for r in rows for a in r['arrays'].values())}


def test_native_subnormal_pressure(oracle, replay_receipt):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:490-492; reject forced FTZ."""
    cp, port, _, _ = oracle
    native = read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR']) /
                                 'native-subnormal-pressure.bin')
    pressure = cp.asarray(native['pres_hyd'])
    actual = cp.empty_like(pressure)
    args = (np.int32(pressure.size), pressure, actual)
    port._gpu_kernel('mpas_pressure_mb')((1,), (32,), args)
    expected = native['pres'].view(np.uint32)
    words = cp.asnumpy(actual).view(np.uint32)
    differing = int(np.count_nonzero(words != expected))
    assert np.any((expected & 0x7fffffff) > 0)
    assert np.any(((expected & 0x7fffffff) > 0) &
                  ((expected & 0x7fffffff) < 0x00800000))
    assert differing == 0
    # Compile the identical production kernel with FTZ enabled to witness
    # that this fixture detects the CuPy RawModule override hazard.
    from gpuwm import nvrtc_ptx_cache
    source = port._DRIVER_SOURCE[port._DRIVER_SOURCE.index(
        'extern "C" __global__ void mpas_pressure_mb'):]
    image, _ = nvrtc_ptx_cache.compile_using_nvrtc(
        source, ('--fmad=false', '--ftz=true', '--prec-div=true',
                 '--prec-sqrt=true'), None, 'mpas_lw_ftz_negative_control.cu')
    module = cp.cuda.function.Module()
    module.load(image.encode() if isinstance(image, str) else image)
    flushed = cp.empty_like(pressure)
    module.get_function('mpas_pressure_mb')((1,), (32,),
        (np.int32(pressure.size), pressure, flushed))
    control_differences = int(np.count_nonzero(
        cp.asnumpy(flushed).view(np.uint32) != expected))
    assert control_differences > 0
    replay_receipt['initialization']['subnormal_pressure'] = {
        'pres': {'words': int(expected.size), 'differing_words': differing}}
    replay_receipt['subnormal_negative_control_differing_words'] = control_differences


def test_rust_raw_table_decode():
    path=os.environ.get('MPAS_LW_ORACLE_DIR')
    if not path:
        pytest.skip('standalone native fixtures required')
    path=Path(path)
    if not (path/'native-raw-tables.bin').exists():
        pytest.skip('Rust raw decoder output required')
    header,payload=(path/'native-raw-tables.bin').read_bytes().split(b'\n',1)
    native=read_native_fixture(path/'native-tables.bin')
    arrays=json.loads(header)['arrays']
    receipt={}
    for field in arrays:
        actual=np.ndarray((field['words'],),dtype='<u4',buffer=payload,offset=field['offset'])
        key=field['module']+'/'+field['name']
        expected=native[key].ravel(order='F').view(np.uint32)
        receipt[key]={'words':actual.size,'differing_words':int(np.count_nonzero(actual!=expected))}
    receipt_path=os.environ.get('MPAS_LW_DECODE_RECEIPT')
    if receipt_path:
        Path(receipt_path).write_text(json.dumps({'arrays':receipt,'total_differing_words':
            sum(row['differing_words'] for row in receipt.values())},indent=2)+'\n')
    assert all(row['differing_words']==0 for row in receipt.values())


@pytest.fixture(scope='module')
def oracle():
    path=os.environ.get('MPAS_LW_ORACLE_DIR')
    if not path:
        pytest.skip('MPAS_LW_ORACLE_DIR not supplied; standalone native fixtures required')
    cp=pytest.importorskip('cupy')
    from hexcore.mpasphys import rrtmg_lw_v842 as port
    raw=read_native_fixture(Path(path)/'native-tables.bin')
    tables={key.replace('rrlw_','',1):value for key,value in raw.items()}
    raw_device={}
    for band,_,source,_,_,_ in port._REDUCTIONS:
        mod=f'rrlw_kg{band:02d}'
        raw_device.setdefault(mod,{})[source]=cp.asarray(
            raw[f'{mod}/{source}'].ravel(order='F'))
    K=port.initialize_device(raw_device)
    columns=read_native_fixture(Path(path)/'native-columns.bin')
    return cp,port,K,columns


def replay_case(oracle,case):
    cp,port,K,columns=oracle
    get=lambda name:columns[f'{case:03d}/{name}']
    flags=get('flags').tolist()
    actual=port.rrtmg_lwrad_device(cp.ascontiguousarray(cp.asarray(get('model')[None])),
        cp.ascontiguousarray(cp.asarray(get('interfaces')[None])),cp.asarray(get('surface')[None]),K,
        icloud=flags[0],cldovrlp=flags[1],idcor=flags[2],julday=flags[3],
        has_reqc=flags[4],has_reqi=flags[5],has_reqs=flags[6],o3input=flags[7],
        pin=cp.asarray(get('pin')),ozone=cp.asarray(get('ozone')[None]))
    return compare_case(cp,port,actual,columns,case)


def compare_case(cp,port,actual,columns,case,row=0):
    get=lambda name:columns[f'{case:03d}/{name}']
    compare={'flux':np.array([cp.asnumpy(actual[name])[row] for name in port.FLUX_FIELDS],dtype=np.float32),
             'heating':cp.asnumpy(actual['rthratenlw'])[row]}
    compare.update({name:cp.asnumpy(actual[name])[row] for name in ('rre_cloud','rre_ice','rre_snow')})
    for name in ('uflx','uflxc','dflx','dflxc'):
        if f'{case:03d}/{name}' in columns:
            compare[name]=cp.asnumpy(actual['profiles'][row][name])[0]
    result={}
    for name,values in compare.items():
        expected=get(name)
        unequal=values.view(np.uint32)!=expected.view(np.uint32)
        result[name]={'words':values.size,'differing_words':int(np.count_nonzero(unequal)),
                      'max_abs':float(np.max(np.abs(values-expected))),
                      'actual_hex':values.view(np.uint32)[unequal].tolist(),
                      'native_hex':expected.view(np.uint32)[unequal].tolist()}
    return result


@pytest.mark.parametrize('case',range(300))
def test_pristine_mpas_column(oracle,replay_receipt,case):
    result=replay_case(oracle,case)
    cp,port,_,_=oracle
    if 'cuda_source_sha256' not in replay_receipt:
        replay_receipt['cuda_source_sha256']=hashlib.sha256((port._solver_source()+
            '\n'+port._RANDOM_SOURCE+'\n'+port._DRIVER_SOURCE+'\n'+port._INIT_SOURCE).encode()).hexdigest()
    replay_receipt['columns'][str(case)]=result
    assert not any(row['differing_words'] for row in result.values()),json.dumps(result)


@pytest.mark.parametrize('column_chunk',[None,2])
def test_pristine_column_batches(oracle,replay_receipt,column_chunk):
    """MPAS: module_ra_rrtmg_lw.F:11810-12305, independent column loop."""
    cp,port,K,columns=oracle
    groups={}
    for case in range(300):
        flags=tuple(columns[f'{case:03d}/flags'].tolist())
        key=(flags,columns[f'{case:03d}/pin'].tobytes())
        groups.setdefault(key,[]).append(case)
    tested=0;layer_counts=set()
    for (flags,_),members in groups.items():
        if len(members)<2:continue
        field=lambda name:cp.ascontiguousarray(cp.asarray(np.stack(
            [columns[f'{case:03d}/{name}'] for case in members])))
        actual=port.rrtmg_lwrad_device(field('model'),field('interfaces'),field('surface'),K,
            icloud=flags[0],cldovrlp=flags[1],idcor=flags[2],julday=flags[3],
            has_reqc=flags[4],has_reqi=flags[5],has_reqs=flags[6],o3input=flags[7],
            pin=cp.asarray(columns[f'{members[0]:03d}/pin']),ozone=field('ozone'),
            column_chunk=column_chunk)
        layer_counts.update(actual['mpas_nlay'].get().tolist())
        for row,case in enumerate(members):
            result=compare_case(cp,port,actual,columns,case,row)
            replay_receipt['batch'][f'{column_chunk}/{case}']=result
            assert not any(row['differing_words'] for row in result.values()),(case,result)
            tested+=1
    assert tested>0 and len(layer_counts)>1


def test_wrapper_default_batch_boundary(oracle,replay_receipt):
    """MPAS: module_ra_rrtmg_lw.F:11810-12185, column-local KISS state."""
    cp,port,K,columns=oracle
    case=0
    get=lambda name:columns[f'{case:03d}/{name}']
    flags=get('flags').tolist()
    # Cross the production wrapper's 256-column boundary with authentic
    # strict-native columns; no expected flux or heating is an input.
    nc=263
    field=lambda name:cp.ascontiguousarray(cp.repeat(cp.asarray(get(name))[None],nc,axis=0))
    actual=port.rrtmg_lwrad_device(field('model'),field('interfaces'),field('surface'),K,
        icloud=flags[0],cldovrlp=flags[1],idcor=flags[2],julday=flags[3],
        has_reqc=flags[4],has_reqi=flags[5],has_reqs=flags[6],o3input=flags[7],
        pin=cp.asarray(get('pin')),ozone=field('ozone'))
    for row in range(nc):
        result=compare_case(cp,port,actual,columns,case,row)
        replay_receipt['batch'][f'boundary/{row}']=result
        assert all(value['differing_words']==0 for value in result.values()),(row,result)


def test_initialization_words(oracle,replay_receipt):
    cp,port,K,_=oracle
    native=read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-tables.bin')
    native.update(read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-diffusivity.bin'))
    actual={key:cp.asnumpy(array) for key,array in K['reduced'].items()}
    expected={key:native['rrlw_'+key].ravel(order='F') for key in actual}
    for name in ('tau_tbl','exp_tbl','tfn_tbl'):
        actual[name]=cp.asnumpy(K[name]);expected[name]=native['rrlw_tbl/'+name]
    actual['rwgt']=cp.asnumpy(K['rwgt']);expected['rwgt']=native['rrlw_wvn/rwgt']
    for name,_,_ in port._STATIC_ARRAYS:
        native_name='chi_mls' if name=='chi' else name
        matches=[key for key in native if key.endswith('/'+native_name)]
        assert len(matches)==1,(name,matches)
        actual['static/'+name]=cp.asnumpy(K[name])
        expected['static/'+name]=native[matches[0]].ravel(order='F')
    for name in port._STATIC_SCALARS:
        matches=[key for key in native if key.endswith('/'+name)]
        assert len(matches)==1,(name,matches)
        actual['scalar/'+name]=np.asarray([K[name]],dtype=np.float32)
        expected['scalar/'+name]=np.asarray([native[matches[0]]],dtype=np.float32)
    result={}
    for key,values in actual.items():
        result[key]={'words':values.size,'differing_words':int(np.count_nonzero(
            values.view(np.uint32)!=expected[key].view(np.uint32)))}
    replay_receipt['initialization']['tables']=result
    assert not any(row['differing_words'] for row in result.values()),json.dumps(result)


@pytest.mark.parametrize('case',range(6))
def test_pristine_random_helpers(oracle,replay_receipt,case):
    """MPAS: module_ra_rrtmg_lw.F:1712-2030, complete int32 RNG state."""
    cp,port,_,_=oracle
    native=read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-randoms.bin')
    get=lambda name:native[f'{case:03d}/{name}']
    seed=cp.asarray(get('seed'))
    n=get('real').size
    actual={name:cp.empty(n,dtype=cp.int32) for name in ('integers','positive')}
    actual['real']=cp.empty(n,dtype=cp.float32)
    actual['state']=cp.empty(625,dtype=cp.int32)
    actual['finalized']=cp.empty(625,dtype=cp.int32)
    port._gpu_kernel('mpas_mt_replay')((1,),(1,),
        (seed,seed.size,int(case in (3,5)),int(case>=4),n,
         actual['integers'],actual['positive'],actual['real'],actual['state'],actual['finalized']))
    result={}
    for name,array in actual.items():
        if case>=4 and name!='real':continue
        values=cp.asnumpy(array)
        result[name]={'words':values.size,'differing_words':int(np.count_nonzero(
            values.view(np.uint32)!=get(name).view(np.uint32)))}
    replay_receipt['randoms'][str(case)]=result
    assert not any(row['differing_words'] for row in result.values()),(case,result)


@pytest.mark.parametrize('case',range(30))
def test_pristine_mcica_helpers(oracle,replay_receipt,case):
    """MPAS: module_ra_rrtmg_lw.F:2099-2737, KISS and MT paths."""
    cp,port,K,_=oracle
    native=read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-mcica.bin')
    get=lambda name:native[f'{case:03d}/{name}']
    overlap,rng,idcor,day,seed=get('flags').tolist()
    a=cp.ascontiguousarray(cp.asarray(get('a')))
    surface=cp.asarray(get('surface'));tau=cp.ascontiguousarray(cp.asarray(get('tau')))
    nl=a.shape[1]
    actual={name:cp.empty((nl,140),dtype=cp.float32)
            for name in ('cloud','liquid','ice','snow','optical_depth')}
    actual['radii']=cp.empty((3,nl),dtype=cp.float32)
    port._gpu_kernel('mpas_mcica_column')((1,),(1,),
        (nl,overlap,idcor,day,surface,a,rng,seed,tau,K['ngb'],actual['optical_depth'],
         actual['cloud'],actual['liquid'],actual['ice'],actual['snow'],actual['radii']))
    result={}
    for name,array in actual.items():
        values=cp.asnumpy(array)
        expected=get(name) if name=='radii' else get(name).T
        result[name]={'words':values.size,'differing_words':int(np.count_nonzero(
            values.view(np.uint32)!=expected.view(np.uint32)))}
    replay_receipt['mcica'][str(case)]=result
    assert not any(row['differing_words'] for row in result.values()),(case,result)


@pytest.mark.parametrize('case',range(30))
def test_pristine_mcica_multi_columns(oracle,replay_receipt,case):
    """MPAS: module_ra_rrtmg_lw.F:2099-2697, global MT draw ordering."""
    cp,port,K,_=oracle
    native=read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-mcica-multi.bin')
    get=lambda name:native[f'{case:03d}/{name}']
    overlap,rng,idcor,day,seed=get('flags').tolist()
    device=lambda array:cp.ascontiguousarray(cp.asarray(array))
    actual=port.mcica_subcol_device(device(get('a')),device(get('surface')),
        device(get('tau').transpose(1,0,2)),K,
        icld=overlap,irng=rng,permuteseed=seed,idcor=idcor,julday=day)
    result={}
    for name in ('cloud','liquid','ice','snow','optical_depth','radii'):
        values=cp.asnumpy(actual[name])
        expected=get(name) if name=='radii' else get(name).transpose(1,2,0)
        unequal=values.view(np.uint32)!=expected.view(np.uint32)
        result[name]={'words':values.size,'differing_words':int(np.count_nonzero(unequal)),
                     'actual_hex':values.view(np.uint32)[unequal].tolist(),
                     'native_hex':expected.view(np.uint32)[unequal].tolist()}
    replay_receipt['mcica_multi'][str(case)]=result
    assert actual['irng']==get('rng_out')
    assert not any(row['differing_words'] for row in result.values()),(case,result)


@pytest.mark.parametrize('case',range(158))
def test_pristine_solver(oracle,replay_receipt,case):
    """MPAS: module_ra_rrtmg_lw.F:10705-11194, direct gas/cloud solver."""
    _replay_solver(oracle, replay_receipt, case, 'native-solver.bin', 'solver')


@pytest.mark.parametrize('case', range(2))
def test_native_extended_solver(oracle, replay_receipt, case):
    """MPAS: module_ra_rrtmg_lw.F:3148-3195; native nlay-sized arrays."""
    _replay_solver(oracle, replay_receipt, case, 'native-extended-solver.bin',
                   'extended_solver')


def _replay_solver(oracle, replay_receipt, case, filename, group):
    cp,port,K,_=oracle
    native=read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR'])/filename)
    get=lambda name:native[f'{case:03d}/{name}']
    device=lambda name:cp.ascontiguousarray(cp.asarray(get(name)))
    gas=get('gas')
    gases=[cp.ascontiguousarray(cp.asarray(gas[i:i+1])) for i in range(10)]
    icloud,inflag,iceflag,liqflag=get('flags').tolist()
    nl=get('play').shape[1]
    actual=port.solve_device(1,nl,icloud,device('play'),device('plev'),device('tlay'),device('tlev'),
        device('tsfc'),*gases,device('emis'),inflag,iceflag,liqflag,
        device('cloud'),device('tau'),device('ice'),device('liquid'),device('snow'),
        device('re_ice'),device('re_liquid'),device('re_snow'),device('aerosol'),K)
    result={}
    for name in ('uflx','dflx','hr','uflxc','dflxc','hrc'):
        values=cp.asnumpy(actual[name]);expected=get(name)
        unequal=values.view(np.uint32)!=expected.view(np.uint32)
        result[name]={'words':values.size,'differing_words':int(np.count_nonzero(unequal)),
                     'actual_hex':values.view(np.uint32)[unequal].tolist(),
                     'native_hex':expected.view(np.uint32)[unequal].tolist()}
    replay_receipt[group][str(case)]=result
    assert not any(row['differing_words'] for row in result.values()),(case,result)


def test_rust_device_table_loader(oracle,replay_receipt):
    """MPAS: mpas_atmphys_rrtmg_lwinit.F:59-1593, actual Rust launch path."""
    asset=os.environ.get('MPAS_LW_TABLE_ASSET');decoder=os.environ.get('MPAS_LW_TABLE_DECODER')
    if not asset or not decoder:pytest.skip('Rust executable and pinned table asset required')
    cp,port,_,columns=oracle
    tables=port.load_tables_device(asset,decoder=decoder)
    result=replay_case((cp,port,tables,columns),1)
    replay_receipt['loader']['column_1']=result
    assert not any(row['differing_words'] for row in result.values()),result


@pytest.mark.parametrize('case',range(14))
def test_native_abort_paths(oracle,replay_receipt,case):
    """MPAS: module_ra_rrtmg_lw.F:2195-2196,2433-2434,2922-3017."""
    cp,port,K,_=oracle
    native=json.loads((Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-fatal-paths.json').read_text())[str(case)]
    assert 'oracle expected' not in native['output']
    if case<2:
        fixture=read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-mcica-multi.bin')
        a=cp.ascontiguousarray(cp.asarray(fixture['000/a']))
        surface=cp.ascontiguousarray(cp.asarray(fixture['000/surface']))
        tau=cp.ascontiguousarray(cp.asarray(fixture['000/tau'].transpose(1,0,2)))
        if case==1:a[:,0,0]=600.
        message='INVALID ICLD' if case==0 else 'KISSVEC SEED GENERATOR'
        assert message in native['output']
        with pytest.raises(ValueError,match=message):
            port.mcica_subcol_device(a,surface,tau,K,icld=6 if case==0 else 1,
                irng=0,permuteseed=150,idcor=0,julday=100)
        verdict={'native_diagnostic':message,'device_verdict':'ValueError'}
    else:
        nl=8;inflag=1 if case==2 else 2;iceflag=3;liqflag=1
        radius_ice=30.;radius_snow=60.;radius_liquid=10.
        if case==3:iceflag=0;radius_ice=9.
        if case in (4,5):iceflag=1;radius_ice=12. if case==4 else 131.
        if case in (6,7):iceflag=2;radius_ice=4. if case==6 else 132.
        if case in (8,9):radius_ice=4. if case==8 else 141.
        if case in (10,11):iceflag=5;radius_snow=4. if case==10 else 141.
        if case in (12,13):radius_liquid=2. if case==12 else 61.
        cloud=cp.ones((1,nl,140),dtype=cp.float32)
        mass=cp.full_like(cloud,10.);tau=cp.full_like(cloud,.1)
        radius=lambda value:cp.full((1,nl),value,dtype=cp.float32)
        flag=cp.zeros(1,dtype=cp.int32);ncb=cp.zeros(1,dtype=cp.int32)
        port._gpu_kernel('rlw_cldprmc_coalesced')((2,),(128,),
            (np.int32(1),np.int32(nl),np.int32(inflag),np.int32(iceflag),np.int32(liqflag),
             cloud,mass,mass,mass,radius(radius_ice),radius(radius_liquid),radius(radius_snow),
             K['absice1'],K['absice2'],K['absice3'],K['absice0'],K['absliq1'],K['absliq0'],
             K['ngb'],tau,ncb,flag))
        expected=(1,2,3,3,4,4,5,5,6,6,7,7)[case-2]
        assert int(flag.item())==expected
        verdict={'native_diagnostic':native['output'].split('Error termination.')[0],
                 'device_abort_code':expected}
    replay_receipt.setdefault('native_abort_tests',{})[str(case)]=verdict


def test_solver_column_chunks(oracle,replay_receipt):
    """MPAS: module_ra_rrtmg_lw.F:11048-11191, five distinct columns."""
    cp,port,K,_=oracle
    native=read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-solver.bin')
    cases=(1,13,25,37,49);nl=8
    get=lambda case,name:native[f'{case:03d}/{name}']
    assert all(np.array_equal(get(case,'flags'),get(1,'flags')) for case in cases)
    field=lambda name,axis=0:cp.ascontiguousarray(cp.concatenate(
        [cp.asarray(get(case,name)) for case in cases],axis=axis))
    gases=[cp.ascontiguousarray(cp.concatenate(
        [cp.asarray(get(case,'gas')[i:i+1]) for case in cases])) for i in range(10)]
    actual=port.solve_device(5,nl,1,field('play'),field('plev'),field('tlay'),field('tlev'),
        field('tsfc'),*gases,field('emis'),2,1,0,
        field('cloud',1),field('tau',1),field('ice',1),field('liquid',1),field('snow',1),
        field('re_ice'),field('re_liquid'),field('re_snow'),field('aerosol'),K,column_chunk=2)
    result={}
    for name in ('uflx','dflx','hr','uflxc','dflxc','hrc'):
        values=cp.asnumpy(actual[name]);expected=np.concatenate([get(case,name) for case in cases])
        unequal=values.view(np.uint32)!=expected.view(np.uint32)
        result[name]={'words':values.size,'differing_words':int(np.count_nonzero(unequal))}
    replay_receipt.setdefault('chunks',{})['5_columns_chunk_2']=result
    assert not any(row['differing_words'] for row in result.values()),result


@pytest.mark.parametrize('dataset',['native-real-columns.bin']+sorted(
    p.name for p in Path(os.environ.get('MPAS_LW_ORACLE_DIR','.')).glob('native-real-columns-set*.bin')))
@pytest.mark.parametrize('case',range(32))
def test_native_init_derived_columns(oracle,replay_receipt,case,dataset):
    """MPAS: module_ra_rrtmg_lw.F:11574-12305, real init thermodynamics.

    Companion radiation settings belong to the harness. This is not an
    f1 driver dump or a whole-model receipt.
    """
    path=Path(os.environ['MPAS_LW_ORACLE_DIR'])/dataset
    if not path.exists():pytest.skip('published native initialization columns required')
    cp,port,K,_=oracle
    columns=read_native_fixture(path)
    result=replay_case((cp,port,K,columns),case)
    replay_receipt['real_columns'][f'{dataset}/{case}']=result
    assert not any(row['differing_words'] for row in result.values()),(case,result)


@pytest.mark.parametrize('case',range(14))
def test_pristine_setcoef_band_selection(oracle,replay_receipt,case):
    """MPAS: module_ra_rrtmg_lw.F:3567-3932, standalone istart 1/16."""
    cp,port,K,_=oracle
    native=read_native_fixture(Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-setcoef.bin')
    get=lambda name:native[f'{case:03d}/{name}']
    nl=get('pavel').size
    fkeys=('colh2o colco2 colo3 coln2o colco colch4 colo2 colbrd fac00 fac01 fac10 fac11 '
           'rat_h2oco2 rat_h2oco2_1 rat_h2oo3 rat_h2oo3_1 rat_h2on2o rat_h2on2o_1 '
           'rat_h2och4 rat_h2och4_1 rat_n2oco2 rat_n2oco2_1 rat_o3co2 rat_o3co2_1 '
           'selffac selffrac forfac forfrac minorfrac scaleminor scaleminorn2').split()
    ikeys='jp jt jt1 indself indfor indminor'.split()
    actual={name:cp.zeros((1,nl),dtype=cp.float32) for name in fkeys}
    actual.update({name:cp.zeros((1,nl),dtype=cp.int32) for name in ikeys})
    actual['laytrop']=cp.zeros(1,dtype=cp.int32)
    for name,shape in [('planklay',(1,nl,16)),('planklev',(1,nl+1,16)),('plankbnd',(1,16))]:
        actual[name]=cp.zeros(shape,dtype=cp.float32)
    output_order=('laytrop jp jt jt1 planklay planklev plankbnd colh2o colco2 colo3 coln2o '
        'colco colch4 colo2 colbrd fac00 fac01 fac10 fac11 rat_h2oco2 rat_h2oco2_1 '
        'rat_h2oo3 rat_h2oo3_1 rat_h2on2o rat_h2on2o_1 rat_h2och4 rat_h2och4_1 '
        'rat_n2oco2 rat_n2oco2_1 rat_o3co2 rat_o3co2_1 selffac selffrac indself '
        'forfac forfrac indfor minorfrac scaleminor scaleminorn2 indminor').split()
    device=lambda name:cp.ascontiguousarray(cp.asarray(np.atleast_1d(get(name))))
    port._gpu_kernel('rlw_setcoef')((1,),(64,),
        (np.int32(1),np.int32(nl),np.int32(get('istart')),device('pavel'),device('tavel'),
         device('tz'),device('tbound'),device('semiss'),device('coldry'),device('wkl'),device('wbroad'),
         K['totplnk'],K['totplk16'],K['preflog'],K['tref'],K['chi'],
         *[actual[name] for name in output_order]))
    result={}
    for name,array in actual.items():
        values=cp.asnumpy(array).ravel();expected=np.atleast_1d(get(name)).ravel()
        unequal=values.view(np.uint32)!=expected.view(np.uint32)
        result[name]={'words':values.size,'differing_words':int(np.count_nonzero(unequal)),
                     'actual_hex':values.view(np.uint32)[unequal].tolist(),
                     'native_hex':expected.view(np.uint32)[unequal].tolist()}
    replay_receipt['setcoef'][str(case)]=result
    assert not any(row['differing_words'] for row in result.values()),(case,result)


@pytest.mark.parametrize('case',range(1,7))
@pytest.mark.parametrize('strided',[False,True])
def test_pristine_multicolumn_vinterp(oracle,replay_receipt,case,strided):
    """MPAS: module_ra_rrtmg_vinterp.F:114-165, common previous index."""
    cp,port,_,_=oracle
    path=Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-vinterp.bin'
    if not path.exists():pytest.skip('native multi-column vinterp fixture required')
    native=read_native_fixture(path)
    get=lambda name:native[f'{case:03d}/{name}']
    # Fortran fixture is (columns,levels); exercise both contiguous copies
    # and the strided views that the real MPAS pool adapter can supply.
    def device(a):
        if not strided:return cp.ascontiguousarray(cp.asarray(a))
        b=cp.empty((a.shape[0],a.shape[1]*2),dtype=cp.float32)
        b[:,::2]=cp.asarray(a)
        return b[:,::2]
    actual=cp.asnumpy(port.vinterp_ozn_device(device(get('pmid')),
        cp.asarray(get('pin')),device(get('ozone'))))
    expected=get('out')
    row={'words':actual.size,'differing_words':int(np.count_nonzero(
        actual.view(np.uint32)!=expected.view(np.uint32)))}
    replay_receipt['vinterp'][f'{case}/{strided}']={'ozone':row}
    assert row['differing_words']==0


def test_published_f1_native_driver(oracle,replay_receipt):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:865-885; native wrapper inputs.

    Expected outputs validate only; all unwritten storage is restored from
    PRE records, including clear/cloud radius words and the extra level.
    """
    cp,port,K,_=oracle
    path=Path(os.environ.get('MPAS_LW_F1_DRIVER_FIXTURE',str(Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-f1-driver.bin')))
    if not path.exists():pytest.skip('published f1 driver sample required')
    native=read_native_fixture(path)
    pre=lambda name:native['pre/'+name]
    scalar=lambda name:int(pre(name))
    nm=scalar('kte')-scalar('kts')+1
    assert scalar('kts')==1 and scalar('jts')==scalar('jte')==1
    field=lambda name:cp.asarray(pre(name))[:,:,0]
    model=cp.stack([field(name)[:,:nm] for name in port.MODEL_FIELDS],axis=1)
    interfaces=cp.stack([field(name)[:,:nm+1] for name in ('p8w','t8w')],axis=1)
    surface=cp.stack([cp.asarray(pre(name))[:,0] for name in port.SURFACE_FIELDS],axis=1)
    names=port.FLUX_FIELDS+('rthratenlw','rre_cloud','rre_ice','rre_snow')
    restored={name:cp.array(pre(name),copy=True) for name in names}
    flags={name:scalar(name) for name in ('icloud','cldovrlp','idcor','julday',
                'has_reqc','has_reqi','has_reqs','o3input')}
    chunk=int(os.environ.get('MPAS_LW_F1_CHUNK_COLUMNS','256'))
    assert chunk>0
    ozone=field('o3clim')
    for first in range(0,model.shape[0],chunk):
        cols=slice(first,min(first+chunk,model.shape[0]))
        actual=port.rrtmg_lwrad_device(cp.ascontiguousarray(model[cols]),
            cp.ascontiguousarray(interfaces[cols]),cp.ascontiguousarray(surface[cols]),K,
            **flags,pin=cp.asarray(pre('pin')),ozone=cp.ascontiguousarray(ozone[cols]))
        for name in names:
            if name in port.FLUX_FIELDS:restored[name][cols,0]=actual[name]
            elif name=='rthratenlw':restored[name][cols,:nm,0]=actual[name]
            elif scalar('icloud')>0:
                mask=field('cldfra3d')[cols,:nm]>0
                restored[name][cols,:nm,0]=cp.where(mask,actual[name],restored[name][cols,:nm,0])
    result={}
    for name in names:
        value=cp.asnumpy(restored[name])
        expected=native['post/'+name]
        unequal=value.view(np.uint32)!=expected.view(np.uint32)
        result[name]={'words':value.size,'differing_words':int(np.count_nonzero(unequal)),
                      'actual_hex':value.view(np.uint32)[unequal].tolist()[:16],
                      'native_hex':expected.view(np.uint32)[unequal].tolist()[:16]}
    replay_receipt['f1_driver'][f'published_columns_{model.shape[0]}']=result
    assert not any(row['differing_words'] for row in result.values()),json.dumps(result)


def test_native_callee_inputs_through_driver_slice(oracle, replay_receipt):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:297-719,847-885.

    Exercise actual packing and writes on the published sampled callee inputs.
    The selected off microphysics route leaves radius diagnostics untouched.
    This isolates the driver slice; it is not a driver-boundary capture replay
    or a forecast. Expected POST words are read only after computation.
    """
    from types import SimpleNamespace
    cp, port, tables, _ = oracle
    path = Path(os.environ['MPAS_LW_ORACLE_DIR']) / 'native-f1-driver.bin'
    if not path.exists():
        pytest.skip('published native callee sample required')
    native = read_native_fixture(path)
    pre = lambda name: native['pre/' + name]
    nm = int(pre('kte'))
    nc = pre('p3d').shape[0]
    assert tuple(int(pre(name)) for name in (
        'kts', 'icloud', 'cldovrlp', 'idcor', 'has_reqc', 'has_reqi',
        'has_reqs', 'o3input')) == (1, 1, 2, 0, 0, 0, 0, 2)
    profile = lambda name: cp.asarray(pre(name))[:, :nm, 0].T
    surface = lambda name: cp.asarray(pre(name))[:, 0]
    prepared = {target: profile(source) for target, source in (
        ('pres_hyd', 'p3d'), ('t', 't3d'), ('dz', 'dz8w'), ('pi', 'pi3d'))}
    prepared.update({target: cp.asarray(pre(source))[:, :nm+1, 0].T
                     for target, source in (('pres2_hyd', 'p8w'), ('t2', 't8w'))})
    prepared['xlat'] = surface('xlat')
    sentinel = np.float32(987.5)
    diagnostic = {('olrtoa' if name == 'olr' else name):
                  cp.full(nc+1, sentinel, dtype=cp.float32)
                  for name in port.FLUX_FIELDS}
    diagnostic.update(cldfrac=profile('cldfra3d'),
                      sfc_emiss=surface('emiss'),
                      o3clim=cp.asarray(pre('o3clim'))[:, :, 0].T,
                      o3vmr=cp.full((nm, nc+1), sentinel, dtype=cp.float32))
    for name in ('rre_cloud', 'rre_ice', 'rre_snow'):
        diagnostic[name] = cp.full((nm, nc+1), sentinel, dtype=cp.float32)
    tendency = cp.full((nm, nc+1), sentinel, dtype=cp.float32)
    pools = dict(nCellsSolve=nc, prepared=prepared, diag_physics=diagnostic,
        radshared={target: profile(source).T for target, source in (
            ('qvrad', 'qv3d'), ('qcrad', 'qc3d'), ('qirad', 'qi3d'),
            ('qsrad', 'qs3d'))},
        sfc_input={target: surface(source) for target, source in (
            ('skintemp', 'tsk'), ('xland', 'xland'), ('xice', 'xice'),
            ('snow', 'snow'))}, atm_input={'pin': cp.asarray(pre('pin'))},
        tend_physics={'rthratenlw': tendency}, rrtmg_lw_tables=tables,
        icloud=int(pre('icloud')), julday=int(pre('julday')), radius_flags=(0, 0, 0))
    options = dict(config_microp_scheme='off', config_microp_re=False,
                   config_radt_cld_overlap='maximum_random',
                   config_radt_cld_dcorrlen='constant', config_o3climatology=True)
    port.radiation_lw_slot(SimpleNamespace(pools=pools, options=options))
    actual = {name: diagnostic['olrtoa' if name == 'olr' else name][:nc]
              for name in port.FLUX_FIELDS}
    actual['rthratenlw'] = tendency[:, :nc].T
    rows = {}
    for name, value in actual.items():
        got = cp.asnumpy(value)
        expected = (native['post/'+name][:, 0] if name in port.FLUX_FIELDS
                    else native['post/'+name][:, :nm, 0])
        rows[name] = {'words': got.size, 'differing_words': int(np.count_nonzero(
            got.view(np.uint32) != expected.view(np.uint32)))}
    # Native to_MPAS writes solve cells only and skips radii for this route.
    for name in port.FLUX_FIELDS:
        assert bool(cp.all(diagnostic['olrtoa' if name == 'olr' else name][nc:] == sentinel))
    assert bool(cp.all(tendency[:, nc:] == sentinel))
    assert bool(cp.all(diagnostic['o3vmr'][:, nc:] == sentinel))
    for name in ('rre_cloud', 'rre_ice', 'rre_snow'):
        assert bool(cp.all(diagnostic[name] == sentinel))
    replay_receipt['f1_driver']['callee_inputs_driver_slice'] = rows
    assert not any(row['differing_words'] for row in rows.values()), json.dumps(rows)


@pytest.mark.parametrize('case',range(20))
def test_pristine_optional_wrapper_arguments(oracle,replay_receipt,case):
    """MPAS: module_ra_rrtmg_lw.F:11789-11811,12217-12247 optional presence."""
    cp,port,K,_=oracle
    path=Path(os.environ['MPAS_LW_ORACLE_DIR'])/'native-optional-columns.bin'
    if not path.exists():pytest.skip('optional native fixtures required')
    native=read_native_fixture(path)
    get=lambda name:native[f'{case:03d}/{name}']
    device=lambda name:cp.ascontiguousarray(cp.asarray(get(name)[None]))
    flags=get('flags').tolist()
    presence=min(case,15)
    actual=port.rrtmg_lwrad_device(device('model'),device('interfaces'),device('surface'),K,
        icloud=flags[0],cldovrlp=flags[1],idcor=flags[2],julday=flags[3],
        has_reqc=flags[4],has_reqi=flags[5],has_reqs=flags[6],o3input=flags[7],
        pin=cp.asarray(get('pin')),ozone=cp.asarray(get('ozone')[None]),
        has_qc3d=bool(presence&1),has_qi3d=bool(presence&2),
        has_qs3d=bool(presence&4),has_cldfra3d=bool(presence&8))
    count=11 if case&8 or case>=16 else 3
    values={'flux':np.array([cp.asnumpy(actual[name])[0] for name in port.FLUX_FIELDS[:count]],dtype=np.float32),
            'heating':cp.asnumpy(actual['rthratenlw'])[0]}
    if case<16:
        values.update({name:cp.asnumpy(actual[name])[0] for name in ('rre_cloud','rre_ice','rre_snow')})
    values.update({name:cp.asnumpy(actual['profiles'][0][name])[0] for name in ('uflx','uflxc','dflx','dflxc')})
    result={}
    for name,words in values.items():
        expected=get(name)[:count] if name=='flux' else get(name)
        unequal=words.view(np.uint32)!=expected.view(np.uint32)
        result[name]={'words':words.size,'differing_words':int(np.count_nonzero(unequal)),
                      'indices':np.flatnonzero(unequal).tolist()[:16],
                      'actual_hex':words.view(np.uint32)[unequal].tolist()[:16],
                      'native_hex':expected.view(np.uint32)[unequal].tolist()[:16]}
    if case<16 and not case&8:
        assert np.all(get('flux')[3:].view(np.uint32)==0)  # native absent outputs preserve PRE
    if case>=16:
        assert all(np.all(get(name).view(np.uint32)==0)
                   for name in ('rre_cloud','rre_ice','rre_snow'))
    replay_receipt.setdefault('optional',{})[str(case)]=result
    assert not any(row['differing_words'] for row in result.values()),(case,result)


def test_complete_native_vinterp_edges(oracle, replay_receipt):
    """MPAS: module_ra_rrtmg_vinterp.F:21-179; full public boundary matrix.

    Replay every strict-native continuation, including one ozone level and
    subnormal inputs. Retain every differing word even if native repeated
    runs vary; no varying words are silently omitted or granted parity.
    """
    path = os.environ.get('MPAS_LW_VINTERP_EDGE_EXPORTS')
    if not path:
        pytest.skip('Rust exports of the complete two-host native matrix required')
    cp, port, _, _ = oracle
    directory = Path(path)
    manifest = json.loads((directory / 'summary.json').read_text())
    assert manifest['case_count'] == 72
    rows = []
    for case in manifest['cases']:
        inputs = read_rust_fixture(directory / 'exports' / case['name'], cp)
        actual = port.vinterp_ozn_device(inputs['pmid'], inputs['pin'], inputs['ozone'])
        expected = inputs['out']
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        unequal = actual.view(cp.uint32) != expected.view(cp.uint32)
        row = {'name': case['name'], 'words': int(expected.size),
               'differing_words': int(cp.count_nonzero(unequal).item()),
               'indices': cp.flatnonzero(unequal).get().tolist(),
               'actual_hex': actual.view(cp.uint32)[unequal].get().tolist(),
               'native_hex': expected.view(cp.uint32)[unequal].get().tolist(),
               'native_repeated_words_stable': case['all_selected_strict_words_stable']}
        rows.append(row)
    replay_receipt['complete_vinterp_edges'] = {
        'cases': rows, 'native_process_counts_measured': [1, 2],
        'replay_process_count': 1, 'words_excluded': False,
        'words': sum(row['words'] for row in rows),
        'differing_words': sum(row['differing_words'] for row in rows)}
    assert not any(row['differing_words'] for row in rows), rows


def test_complete_native_one_level_ozone_wrapper(oracle, replay_receipt):
    """MPAS: module_ra_rrtmg_lw.F:11574-12305; whole one-level ozone wrapper.

    All 300 native calls retain their original cloud, radius, season and
    model-top branches. Rust decodes the untouched inputs and expected
    outputs. Every output word is compared; native POST is never an input.
    """
    path = os.environ.get('MPAS_LW_ONE_LEVEL_OZONE_EXPORTS')
    if not path:
        pytest.skip('Rust exports of the complete one-level native wrapper required')
    cp, port, tables, _ = oracle
    directory = Path(path)
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['columns_per_run'] == 300 and summary['all_selected_words_stable']
    columns = read_rust_fixture(directory / 'exports', cp)
    rows = []
    for case in range(300):
        get = lambda name: columns[f'{case:03d}/{name}']
        flags = get('flags').get().tolist()
        assert flags[7:] == [2, 1]
        actual = port.rrtmg_lwrad_device(
            cp.ascontiguousarray(get('model')[None]),
            cp.ascontiguousarray(get('interfaces')[None]),
            cp.ascontiguousarray(get('surface')[None]), tables,
            icloud=flags[0], cldovrlp=flags[1], idcor=flags[2], julday=flags[3],
            has_reqc=flags[4], has_reqi=flags[5], has_reqs=flags[6],
            o3input=flags[7], pin=get('pin'), ozone=get('ozone')[None])
        values = {'flux': cp.stack([actual[name][0] for name in port.FLUX_FIELDS]),
                  'heating': actual['rthratenlw'][0]}
        values.update({name: actual[name][0] for name in ('rre_cloud', 'rre_ice', 'rre_snow')})
        values.update({name: actual['profiles'][0][name][0]
                       for name in ('uflx', 'uflxc', 'dflx', 'dflxc')})
        arrays = {}
        for name, value in values.items():
            expected = get(name)
            assert value.shape == expected.shape and value.dtype == expected.dtype
            unequal = value.view(cp.uint32) != expected.view(cp.uint32)
            arrays[name] = {'words': int(expected.size),
                'differing_words': int(cp.count_nonzero(unequal).item()),
                'indices': cp.flatnonzero(unequal).get().tolist(),
                'actual_hex': value.view(cp.uint32)[unequal].get().tolist(),
                'native_hex': expected.view(cp.uint32)[unequal].get().tolist()}
        rows.append({'case': case, 'arrays': arrays})
    replay_receipt['one_level_ozone_wrapper'] = {
        'cases': rows, 'native_process_counts_measured': [1, 2],
        'replay_process_count': 1, 'words_excluded': False,
        'words': sum(a['words'] for row in rows for a in row['arrays'].values()),
        'differing_words': sum(a['differing_words'] for row in rows for a in row['arrays'].values())}
    assert not any(a['differing_words'] for row in rows for a in row['arrays'].values()), rows
