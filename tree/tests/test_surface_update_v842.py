"""Whole-module CUDA replay against unchanged MPAS-A v8.4.2 arithmetic.

GPU invocation: the GPU queue run --minutes 10 -- env P25_SURFACE_GPU=1 ... pytest
The prebuilt oracle lives in P25_NATIVE_DIR, prepared by surface_native_v842.py.
"""
import site_support
import ctypes
import hashlib
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import numpy as np
import pytest

from hexcore.mpasphys.surface_update_v842 import (
    CUDA_OPTIONS, MonthlyCalendar, SurfaceAlarms, physics_update_surface, physics_update_sst,
    physics_update_sstskin, physics_update_deepsoiltemp, update_radiation_diagnostics,
    update_precipitation_buckets,
    physics_timetracker_surface,
)
from surface_native_v842 import (REAL, INT, RAD, SURFACE_FIELDS, call,
                                load_real_surface_fields, load_real_land_use, replay_fields,
                                surface_reader)


def receipt_data(**values):
    """Every comparison states its fixed process count and compiler provenance."""
    import hexcore.mpasphys.surface_update_v842 as surface
    return dict(process_count=1, cuda_compiler='direct NVRTC',
                cuda_options=list(CUDA_OPTIONS),
                science_source_sha256=hashlib.sha256(Path(surface.__file__).read_bytes()).hexdigest(),
                **values)


def test_monthly_calendar_boundaries_and_saved_leap():
    c=MonthlyCalendar()
    assert c.weights('2001-01-01_00:00:00') == (11,0,14,17,31)
    assert c.weights('2001-01-15_23:59:59') == (11,0,0,31,31)
    assert c.weights('2001-02-15_00:00:00') == (0,1,0,31,31)
    assert c.weights('2000-02-29_12:00:00') == (1,2,15,14,29)
    assert c.weights('2001-03-01_00:00:00') == (1,2,14,15,29)
    assert c.february==29
    # A native cold restart starts a new Fortran DATA lifetime.
    assert MonthlyCalendar().weights('2001-03-01_00:00:00') == (1,2,14,14,28)


def test_alarm_initial_phase_and_restart():
    cfg={'config_greeness_update':'24:00:00','config_sst_update':True,'config_bucket_update':'6:00:00'}
    a=SurfaceAlarms.create('2001-01-01_00:00:00',cfg,surface_record_interval='6:00:00')
    assert a.rings(datetime(2001,1,1)) == dict(green=True,surface=True,acrain=False,acradt=False)
    assert a.rings(datetime(2001,1,1)) == dict(green=False,surface=False,acrain=False,acradt=False)
    assert a.rings(datetime(2001,1,1,6)) == dict(green=False,surface=True,acrain=True,acradt=True)
    restored=SurfaceAlarms(**vars(a))
    assert restored.rings(datetime(2001,1,2)) == dict(green=True,surface=True,acrain=True,acradt=True)
    assert restored.rings(datetime(2001,1,2)) == dict(green=False,surface=False,acrain=False,acradt=False)


def test_disabled_alarms_and_missing_stream():
    cfg={'config_greeness_update':'1_00:00:00','config_sst_update':False,'config_bucket_update':'none'}
    a=SurfaceAlarms.create('2001-01-01_00:00:00',cfg)
    assert a.rings(datetime(2001,1,1)) == dict(green=True,surface=False,acrain=False,acradt=False)
    cfg['config_sst_update']=True
    with pytest.raises(ValueError,match='record interval'):
        SurfaceAlarms.create('2001-01-01_00:00:00',cfg)


def initial(n=256,ns=4,nl=30):
    rng=np.random.default_rng(125)
    rf=np.asfortranarray(rng.uniform(0,1,(n,len(REAL))).astype('f4'))
    ii=np.zeros((n,len(INT)),dtype='i4',order='F')
    monthly=np.asfortranarray(rng.uniform(0,100,(12,n,2)).astype('f4'))
    soil=np.asfortranarray(rng.uniform(260,300,(ns,n,3)).astype('f4'))
    lag=np.asfortranarray(rng.uniform(260,300,(nl,n)).astype('f4'))
    r={k:rf[:,i] for i,k in enumerate(REAL)}
    i={k:ii[:,j] for j,k in enumerate(INT)}
    for k in ('sst','skintemp','sstsk','tmn','tyear_mean'):r[k][:]=rng.uniform(270,305,n)
    for k in ('glw','gsw','hfx'):r[k][:]=rng.uniform(-200,900,n)
    r['qfx'][:]=rng.uniform(-.00005,.0005,n)
    r['ust'][:]=np.tile([0.,.001,.01,.1,1.,2.,3.,.5],n//8)
    r['sstsk_dtw'][:]=np.tile([0.,.001,.1,1.,3.,4.,-1.,10.],n//8)
    r['xice'][:]=np.tile([0.,.01,.02,.49,.5,.51,.99,1.,0.,0.,.03,.02,.6,0.,.6,.01],n//16)
    r['xicem'][:]=np.tile([0.,.8,.0,.02,.9,0.,1.,.8,0.,.8,.9,.02,.6,0.,.1,.01],n//16)
    r['xland'][:]=np.tile([2.,1.,2.,2.,1.,2.,1.,1.,1.,2.,2.,1.,2.,1.5,1.,1.],n//16)
    r['nsteps_accum'][:]=1200.
    r['tday_accum'][:]=300000.
    r['ndays_accum'][:]=365.
    r['tyear_accum'][:]=100000.
    i['landmask'][:]=np.arange(n)%2
    for k in ('rainc','rainnc',*['ac'+s for s in RAD]):
        r[k][:]=np.tile([0.,99.,100.,101.,200.,301.,-10.,100.],n//8)
    for k in ('cuprec','rainncv',*RAD):r[k][:]=np.tile([0.,0.,0.,0.,1.,0.,-1.,.125],n//8)
    return rf,ii,monthly,soil,lag


@pytest.fixture(scope='module')
def oracle():
    if os.environ.get('P25_SURFACE_GPU')!='1':pytest.skip('GPU replay requires the GPU queue and P25_SURFACE_GPU=1')
    directory=Path(os.environ['P25_NATIVE_DIR'])
    return ctypes.CDLL(str(directory/'oracle.so'))


CASES=[(1,1,120.,100.,.02,100.,'2001-01-01_00:00:00'),
       (1,0,120.,100.,.02,100.,'2001-01-15_00:00:00'),
       (1,1,120.,100.,.02,100.,'2000-02-29_00:00:00'),
       (1,1,120.,100.,.02,100.,'2001-03-01_00:00:00'),
       (1,1,120.,100.,.02,100.,'2001-12-31_00:00:00')]
CASES += [(1,1,120.,100.,.02,100.,date) for date in ['1900-03-01_00:00:00','2004-03-01_00:00:00']]
CASES += [(2,flag,120.,100.,threshold,100.,'2001-01-01_00:00:00') for flag,threshold in [(1,.02),(0,.5)]]
CASES += [(3,1,dt,100.,.02,100.,'2001-01-01_00:00:00') for dt in [1.,120.,1800.]]
CASES += [(4,leap,120.,jul,.02,100.,'2001-01-01_00:00:00')
          for leap in [0,1] for jul in [0.,100.,100.25,100.49999,365.,366.,367.]]
CASES += [(mode,flag,120.,100.,.02,bucket,'2001-01-01_00:00:00')
          for mode in [5,6,7] for flag,bucket in [(0,100.),(1,100.),(1,0.),(1,-1.)]]
calendar=MonthlyCalendar()


def device_pool(arrays):
    import cupy as cp
    rf,ii,monthly,soil,lag=arrays
    s={k:cp.asarray(rf[:,j]) for j,k in enumerate(REAL)}
    s.update({k:cp.asarray(ii[:,j]) for j,k in enumerate(INT)})
    s.update(isice=24,iswater=16)
    for j,k in enumerate(['albedo12m','greenfrac']):s[k]=cp.array(monthly[:,:,j],order='F')
    for j,k in enumerate(['tslb','sh2o','smois']):s[k]=cp.array(soil[:,:,j],order='F')
    s['tlag']=cp.array(lag,order='F')
    return s


def compare_words(native,s):
    import cupy as cp
    arrays=(*[(k,native[0][:,j]) for j,k in enumerate(REAL)],
            *[(k,native[1][:,j]) for j,k in enumerate(INT)],
            *[(k,native[2][:,:,j]) for j,k in enumerate(['albedo12m','greenfrac'])],
            *[(k,native[3][:,:,j]) for j,k in enumerate(['tslb','sh2o','smois'])],('tlag',native[4]))
    failures={}
    words=0
    for k,expected in arrays:
        actual=cp.asnumpy(s[k])
        mask=actual.view('u4')!=expected.view('u4')
        count=int(np.count_nonzero(mask));words+=expected.size
        if count:
            idx=tuple(np.argwhere(mask)[0]);failures[k]=dict(count=count,index=list(map(int,idx)),got=float(actual[idx]),want=float(expected[idx]))
    return words,failures


def load_real_monthly(arrays,path,reader_library,first=0):
    """Rust decodes NetCDF directly into the oracle's replay buffers."""
    reader=ctypes.CDLL(str(reader_library)).read_monthly
    reader.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_size_t]+[ctypes.c_void_p]*3
    reader.restype=ctypes.c_int
    rf,ii,monthly,soil,lag=arrays
    rc=reader(os.fsencode(path),first,rf.shape[0],monthly[:,:,1].ctypes.data,
              monthly[:,:,0].ctypes.data,ii[:,INT.index('landmask')].ctypes.data)
    if rc:raise ValueError(f'Rust monthly input reader failed with NetCDF/schema code {rc}')


@pytest.mark.parametrize('date',['2000-02-29_12:00:00','2001-03-31_12:00:00','2001-12-31_12:00:00'])
def test_real_monthly_words(oracle,date):
    path=os.environ.get('P25_REAL_MONTHLY_INPUT')
    library=os.environ.get('P25_MONTHLY_READER_LIB')
    if not path or not library:pytest.skip('real monthly replay needs the published input and Rust reader')
    native=initial()
    load_real_monthly(native,path,library,first=int(os.environ.get('P25_FIRST_CELL','0')))
    s=device_pool(native)
    mesh=dict(nCells=256,nCellsSolve=251,nSoilLevels=4,nLags=30)
    call(oracle,1,native,date=date,nsolve=251)
    physics_update_surface(date,True,mesh,s,calendar=calendar)
    words,failures=compare_words(native,s)
    import hashlib
    receipt=site_support.receipt_dir('P25_RECEIPTS')
    (receipt/f'real-monthly-{date[:10]}.json').write_text(json.dumps(receipt_data(
        tier='real monthly input CUDA/strict Fortran; other carriers synthetic; no forecast claim',
        input_path=str(path),input_sha256=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),
        first_cell=int(os.environ.get('P25_FIRST_CELL','0')),date=date,words=words,
        differing_words=sum(v['count'] for v in failures.values()),failures=failures),indent=2)+'\n')
    assert not failures,failures


@pytest.mark.parametrize('mode', [1, 2, 3, 4, 5, 6, 7])
def test_complete_real_surface_words(oracle, mode):
    paths = json.loads(os.environ.get('P25_REAL_SURFACE_FILES', '[]'))
    library = os.environ.get('P25_MONTHLY_READER_LIB')
    if not paths or not library:
        pytest.skip('complete real replay needs published native surface/restart/diagnostic files')
    native = initial(ns=int(os.environ.get('P25_REAL_NSOIL', '4')),
                     nl=int(os.environ.get('P25_REAL_NLAGS', '30')))
    first = int(os.environ.get('P25_FIRST_CELL', '0'))
    record = int(os.environ.get('P25_REAL_RECORD', '0'))
    sources = load_real_surface_fields(native, paths, library, mode, first=first, record=record)
    land_use = None
    if mode == 2:
        land_use, land_sources = load_real_land_use(paths, library)
        sources.update(land_sources)
    s = device_pool(native)
    if land_use:
        s.update(isice=land_use['isice_lu'], iswater=land_use['iswater_lu'])
    mesh = dict(nCells=256, nCellsSolve=251, nSoilLevels=native[3].shape[0], nLags=native[4].shape[0])
    # Native metadata is explicitly supplied; no date/mesh defaults in the port.
    date = os.environ['P25_REAL_DATE']
    dt = float(os.environ['P25_REAL_DT'])
    jul = float(os.environ['P25_REAL_JULIAN'])
    leap = int(os.environ['P25_REAL_LEAP'])
    threshold = float(os.environ['P25_REAL_ICE_THRESHOLD'])
    bucket_key = {5:'P25_REAL_BUCKET_RADT',6:'P25_REAL_BUCKET_RAINC',
                  7:'P25_REAL_BUCKET_RAINNC'}.get(mode,'P25_REAL_BUCKET')
    bucket = float(os.environ[bucket_key])
    flag = leap if mode == 4 else 1
    categories = {} if not land_use else dict(isice=land_use['isice_lu'], iswater=land_use['iswater_lu'])
    call(oracle, mode, native, flag, dt, jul, threshold, bucket, date, nsolve=251, **categories)
    if mode == 1: physics_update_surface(date, True, mesh, s, calendar=calendar)
    elif mode == 2: physics_update_sst(True, mesh, s, s, xice_threshold=threshold)
    elif mode == 3: physics_update_sstskin(dt, mesh, s, s)
    elif mode == 4: physics_update_deepsoiltemp(bool(leap), dt, jul, mesh, s, s)
    elif mode == 5:
        update_radiation_diagnostics(dict(config_bucket_radt=bucket), mesh, s, 1, 251,
                                     dt_dyn=dt, l_acradt=True)
    else:
        update_precipitation_buckets(dict(config_bucket_rainc=bucket, config_bucket_rainnc=bucket),
            mesh, s, 1, 251, dt_dyn=dt, l_acrain=True, convection=mode==6, microphysics=mode==7)
    words, failures = compare_words(native, s)
    import hashlib
    hashes = {path: hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()
              for path in set(sources.values())}
    receipt = site_support.receipt_dir('P25_RECEIPTS')
    (receipt / f'complete-real-mode{mode}.json').write_text(json.dumps(receipt_data(
        tier='all touched carriers from native files CUDA/strict Fortran; no forecast claim',
        mode=mode, sources=sources, source_sha256=hashes, first_cell=first, record=record,
        real_carriers=SURFACE_FIELDS[mode], land_use=land_use,
        real_carrier_words=sum(replay_fields(native)[key].size for key in SURFACE_FIELDS[mode]),
        comparison_includes_untouched_synthetic_sentinels=True,
        date=date, dt=dt, julian=jul, leap=leap, threshold=threshold, bucket=bucket,
        words=words, differing_words=sum(v['count'] for v in failures.values()),
        failures=failures), indent=2) + '\n')
    assert not failures, failures


def test_real_surface_stream_words(oracle):
    """Native case8 surface streams contain SST/xice, without cell-ID fields.

    An immutable publication including the identical native initialization is
    the explicit cell-order witness. This tests Rust record decoding and the
    surface-service callback, not production o1 stream handling or a forecast.
    """
    stream = os.environ.get('P25_REAL_STREAM_FILE')
    publication = os.environ.get('P25_REAL_STREAM_PUBLICATION')
    capture_receipt = os.environ.get('P25_REAL_CAPTURE_RECEIPT')
    if not stream or not publication or not capture_receipt:
        pytest.skip('real surface stream requires its publication and native capture provenance')
    import hashlib
    import cupy as cp
    stream = Path(stream)
    pub = json.loads(Path(publication).read_text())
    capture = json.loads(Path(capture_receipt).read_text())
    stream_sha = hashlib.file_digest(stream.open('rb'), 'sha256').hexdigest()
    assert stream_sha == pub['files'][stream.name]
    init_pins = [(Path(path).name, sha) for path, sha in capture['source_sha256'].items()
                 if path.endswith('.init.nc')]
    assert len(init_pins) == 1
    assert pub['files'][init_pins[0][0]] == init_pins[0][1]
    native = initial(ns=int(os.environ['P25_REAL_NSOIL']), nl=int(os.environ['P25_REAL_NLAGS']))
    files = json.loads(os.environ['P25_REAL_SURFACE_FILES'])
    library = os.environ['P25_MONTHLY_READER_LIB']
    first = int(os.environ['P25_FIRST_CELL'])
    # Native streams omit cell IDs. Verify capture order against the exact
    # initialization included in the surface publication before using ordinals.
    init_path = next(path for path in capture['source_sha256'] if path.endswith('.init.nc'))
    sources = load_real_surface_fields(native, files+[init_path], library, 2, first=first)
    land, _ = load_real_land_use(files, library)
    s = device_pool(native)
    s.update(isice=land['isice_lu'], iswater=land['iswater_lu'])
    mesh = dict(nCells=256, nCellsSolve=251, nSoilLevels=native[3].shape[0], nLags=native[4].shape[0])
    configs = dict(config_sst_update=True, config_frac_seaice=True,
                   config_sstdiurn_update=False, config_deepsoiltemp_update=False, config_sfc_albedo=True)
    reader = surface_reader(library)
    time_reader = ctypes.CDLL(library).read_surface_time
    time_reader.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    time_reader.restype = ctypes.c_int
    records = int(os.environ['P25_REAL_STREAM_RECORDS'])
    fields = replay_fields(native)
    rows = []
    previous = None
    for record in range(records):
        stamp = ctypes.create_string_buffer(256)
        length = time_reader(os.fsencode(stream), record, stamp, len(stamp))
        assert length > 0
        date = bytes(stamp[:length]).decode('ascii').rstrip(' \0')
        now = datetime.fromisoformat(date.replace('_', 'T'))
        if previous is not None:
            assert now - previous == timedelta(seconds=int(os.environ['P25_REAL_STREAM_INTERVAL']))
        previous = now
        record_fields = {}
        for key in ('sst', 'xice'):
            output = np.empty(256, dtype='f4')
            assert reader(os.fsencode(stream), key.encode(), 5, None, 1, first, 256,
                          record, output.ctypes.data, output.size) == 0
            record_fields[key] = output
            fields[key][:] = output
        call(oracle, 2, native, 1, date=date, nsolve=251,
             threshold=float(os.environ['P25_REAL_ICE_THRESHOLD']),
             isice=land['isice_lu'], iswater=land['iswater_lu'])
        def install(current_date, pool):
            assert current_date == date
            for key, output in record_fields.items():
                pool[key][...] = cp.asarray(output)
        physics_timetracker_surface(date, 120., 0., False, configs, mesh, s, s,
            flags=dict(green=False, surface=True), calendar=MonthlyCalendar(),
            xice_threshold=float(os.environ['P25_REAL_ICE_THRESHOLD']), read_surface_record=install)
        words, failures = compare_words(native, s)
        rows.append(dict(record=record, date=date, words=words,
                         differing_words=sum(item['count'] for item in failures.values()), failures=failures))
        assert not failures, failures
    stamp = ctypes.create_string_buffer(256)
    assert time_reader(os.fsencode(stream), records, stamp, len(stamp)) == -10004
    receipt = site_support.receipt_dir('P25_RECEIPTS')
    (receipt/'real-surface-stream.json').write_text(json.dumps(receipt_data(
        tier='native surface records through Rust and service callback; no production-stream/forecast claim',
        stream=str(stream), stream_sha256=stream_sha,
        publication_sha256=hashlib.sha256(Path(publication).read_bytes()).hexdigest(),
        identical_initialization_witness=dict(init_pins), initial_carrier_sources=sources,
        records=records, rows=rows, words=sum(row['words'] for row in rows),
        differing_words=sum(row['differing_words'] for row in rows)), indent=2)+'\n')


@pytest.mark.parametrize('mode,flag,dt,jul,threshold,bucket,date',CASES)
def test_native_words(oracle,mode,flag,dt,jul,threshold,bucket,date,n_soil=4,n_lags=30,nsolve=256):
    native=initial(ns=n_soil,nl=n_lags)
    rf,ii,monthly,soil,lag=native
    s=device_pool(native)
    mesh=dict(nCells=rf.shape[0],nCellsSolve=nsolve,nSoilLevels=soil.shape[0],nLags=lag.shape[0])
    cfg=dict(config_bucket_radt=bucket,config_bucket_rainc=bucket,config_bucket_rainnc=bucket)
    call(oracle,mode,native,flag,dt,jul,threshold,bucket,date,nsolve=nsolve)
    if mode==1:physics_update_surface(date,bool(flag),mesh,s,calendar=calendar)
    elif mode==2:physics_update_sst(bool(flag),mesh,s,s,xice_threshold=threshold)
    elif mode==3:physics_update_sstskin(dt,mesh,s,s)
    elif mode==4:physics_update_deepsoiltemp(bool(flag),dt,jul,mesh,s,s)
    elif mode==5:update_radiation_diagnostics(cfg,mesh,s,1,mesh['nCellsSolve'],dt_dyn=dt,l_acradt=bool(flag))
    else:update_precipitation_buckets(cfg,mesh,s,1,mesh['nCellsSolve'],dt_dyn=dt,l_acrain=bool(flag),convection=mode==6,microphysics=mode==7)
    words,failures=compare_words(native,s)
    receipt=site_support.receipt_dir('P25_RECEIPTS')
    receipt.mkdir(exist_ok=True,parents=True)
    name=f'm{mode}-f{flag}-dt{dt}-j{jul}-b{bucket}-{date[:10]}-ns{n_soil}-nl{n_lags}-n{nsolve}'
    (receipt/(name+'.json')).write_text(json.dumps(receipt_data(mode=mode,flag=flag,dt=dt,jul=jul,threshold=threshold,bucket=bucket,date=date,words=words,differing_words=sum(v['count'] for v in failures.values()),failures=failures),indent=2)+'\n')
    assert not failures,failures


@pytest.mark.parametrize('fractional,isice,iswater', [(True,21,17), (False,3,7)])
def test_native_land_use_categories(oracle, fractional, isice, iswater):
    """Native scalar metadata is passed through the pool on ice transitions."""
    native = initial()
    s = device_pool(native)
    s.update(isice=isice, iswater=iswater)
    mesh = dict(nCells=256,nCellsSolve=251,nSoilLevels=4,nLags=30)
    call(oracle,2,native,flag=int(fractional),nsolve=251,isice=isice,iswater=iswater)
    physics_update_sst(fractional,mesh,s,s,xice_threshold=.02)
    # Both transitions must actually use the supplied native categories.
    assert isice in native[1][:251,INT.index('ivgtyp')]
    assert iswater in native[1][:251,INT.index('ivgtyp')]
    words, failures = compare_words(native,s)
    receipt = site_support.receipt_dir('P25_RECEIPTS')
    (receipt/f'land-use-{isice}-{iswater}.json').write_text(json.dumps(receipt_data(
        tier='synthetic land-use metadata CUDA/strict Fortran; no forecast claim',
        fractional=fractional,isice=isice,iswater=iswater,words=words,
        differing_words=sum(v['count'] for v in failures.values()),failures=failures),indent=2)+'\n')
    assert not failures, failures


def test_bucket_alarm_lookahead_and_independent_gates():
    cfg=dict(config_greeness_update='24:00:00',config_sst_update=False,config_bucket_update='6:00:00')
    a=SurfaceAlarms.create('2001-01-01_00:00:00',cfg)
    assert a.bucket_rings(datetime(2001,1,1,5,58),120.,convection_enabled=False)==dict(acrain=False,acradt=True)
    assert a.bucket_rings(datetime(2001,1,1,5,58),120.)==dict(acrain=True,acradt=False)
    assert a.bucket_rings(datetime(2001,1,1,6),120.)==dict(acrain=False,acradt=False)


def test_device_rejects_strided_pool_fields(oracle):
    import cupy as cp
    a=cp.zeros(16,dtype='f4')[::2]
    from hexcore.mpasphys.surface_update_v842 import _array
    with pytest.raises(ValueError,match='order'):
        _array({'field':a},'field',(8,))


@pytest.mark.parametrize('mode',[1,2,3,4,5,6,7])
def test_native_owned_cells_and_variable_levels(oracle,mode):
    test_native_words(oracle,mode,1,120.,365.,.02,100.,'2001-03-01_00:00:00',
                      n_soil=9,n_lags=1,nsolve=251)


def test_native_repeated_updates_and_restart(oracle):
    """Five synthetic days check persistent state at every half-hour step.

    This is a subsystem replay, not the required whole-model forecast proof.
    The strict reference routines run in the manager's order; restart copies
    every resident carrier, calendar state and alarm state halfway through.
    """
    import cupy as cp
    native=initial()
    s=device_pool(native)
    mesh=dict(nCells=256,nCellsSolve=251,nSoilLevels=4,nLags=30)
    cfg=dict(config_greeness_update='24:00:00',config_sst_update=True,
             config_sstdiurn_update=True,config_deepsoiltemp_update=True,
             config_frac_seaice=True,config_sfc_albedo=True,
             config_bucket_update='6:00:00',config_bucket_radt=100.,
             config_bucket_rainc=100.,config_bucket_rainnc=100.)
    start=datetime(2000,12,31)
    alarms=SurfaceAlarms.create(start,cfg,surface_record_interval='6:00:00')
    # Earlier native cases leave the Fortran DATA February at 29.
    clock_calendar=MonthlyCalendar(february=29)
    records=[]
    total_words=0
    for step in range(241):
        now=start+timedelta(seconds=1800*step)
        date=now.strftime('%Y-%m-%d_%H:%M:%S')
        jul=float(np.float32(now.timetuple().tm_yday-1)+np.float32(now.hour+now.minute/60)/np.float32(24))
        leap=now.year%4==0 and (now.year%100!=0 or now.year%400==0)
        flags=alarms.surface_rings(now)
        if flags['surface']:
            pattern=initial()[0]
            native[0][:,REAL.index('xice')]=np.roll(pattern[:,REAL.index('xice')],step//12)
        # Snapshot the input record before the native sea-ice conversion.
        stream_record={key:native[0][:,REAL.index(key)].copy() for key in ('sst','xice')}
        def read_record(stamp,pool):
            for key,value in stream_record.items():pool[key][...] = cp.asarray(value)
        if flags['green']:call(oracle,1,native,date=date,nsolve=251)
        if flags['surface']:call(oracle,2,native,date=date,nsolve=251)
        call(oracle,3,native,dt=1800.,date=date,nsolve=251)
        call(oracle,4,native,flag=int(leap),dt=1800.,jul=jul,date=date,nsolve=251)
        physics_timetracker_surface(date,1800.,jul,leap,cfg,mesh,s,s,flags=flags,
                                   calendar=clock_calendar,xice_threshold=.02,
                                   read_surface_record=read_record)
        buckets=alarms.bucket_rings(now,1800.)
        call(oracle,5,native,flag=int(buckets['acradt']),dt=1800.,date=date,nsolve=251)
        for mode in (6,7):call(oracle,mode,native,flag=int(buckets['acrain']),dt=1800.,date=date,nsolve=251)
        update_radiation_diagnostics(cfg,mesh,s,1,251,dt_dyn=1800.,l_acradt=buckets['acradt'])
        update_precipitation_buckets(cfg,mesh,s,1,251,dt_dyn=1800.,l_acrain=buckets['acrain'])
        words,failures=compare_words(native,s)
        total_words+=words
        records.append(dict(step=step,date=date,flags=flags,buckets=buckets,
                            differing_words=sum(v['count'] for v in failures.values()),failures=failures))
        if failures:break
        if step==120:
            s={key:value.copy(order='F' if value.ndim==2 else 'C') if isinstance(value,cp.ndarray) else value
               for key,value in s.items()}
            alarms=SurfaceAlarms(**vars(alarms))
            clock_calendar=MonthlyCalendar(february=clock_calendar.february)
    receipt=site_support.receipt_dir('P25_RECEIPTS')
    (receipt/'repeated-restart.json').write_text(json.dumps(receipt_data(
        tier='synthetic subsystem CUDA/strict Fortran; no forecast claim',
        words=total_words,records=records),indent=2)+'\n')
    assert not failures,records[-1]
    assert len(records)==241


@pytest.mark.parametrize('mode', [5, 6, 7])
def test_native_subnormal_accumulation_words(oracle, mode):
    """FTZ would erase the strictly nonzero native accumulator words."""
    native = initial()
    fields = replay_fields(native)
    bits = np.array([1, 0x4000, 0x7fffff, 0x800000,
                     0x80000001, 0x80400000, 0x807fffff, 0x80800000], dtype='u4')
    values = np.tile(bits.view('f4'), 32)
    rates = RAD if mode == 5 else ['cuprec'] if mode == 6 else ['rainncv']
    totals = ['ac'+key for key in RAD] if mode == 5 else ['rainc'] if mode == 6 else ['rainnc']
    for key in rates: fields[key][:] = values
    for key in totals: fields[key][:] = 0
    s = device_pool(native)
    mesh = dict(nCells=256, nCellsSolve=251, nSoilLevels=4, nLags=30)
    call(oracle, mode, native, flag=0, dt=1., nsolve=251)
    native_bits = fields[totals[0]][:8].view('u4')
    assert np.array_equal(native_bits, bits), 'native must retain the subnormal fixture'
    cfg = dict(config_bucket_radt=100., config_bucket_rainc=100., config_bucket_rainnc=100.)
    if mode == 5:
        update_radiation_diagnostics(cfg, mesh, s, 1, 251, dt_dyn=1., l_acradt=False)
    else:
        update_precipitation_buckets(cfg, mesh, s, 1, 251, dt_dyn=1., l_acrain=False,
                                    convection=mode==6, microphysics=mode==7)
    words, failures = compare_words(native, s)
    receipt = site_support.receipt_dir('P25_RECEIPTS')
    (receipt/f'subnormal-mode{mode}.json').write_text(json.dumps(receipt_data(
        tier='subnormal strict native/CUDA subsystem; no forecast claim', mode=mode,
        input_bits=bits.tolist(), native_accumulator_bits=native_bits.tolist(),
        words=words, differing_words=sum(row['count'] for row in failures.values()),
        failures=failures), indent=2)+'\n')
    assert not failures, failures
