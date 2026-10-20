"""Exercise the real native writer, including its emitted species schema."""
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def native():
    path = os.environ.get("RW_MPAS_INIT")
    if not path or not Path(path).is_file():
        pytest.skip("RW_MPAS_INIT must name the built native initializer")
    return path


def test_real_native_augmentation_preserves_soil_and_labels(native, tmp_path):
    from netCDF4 import Dataset
    path = tmp_path / "init.nc"
    with Dataset(path, "w", format="NETCDF3_64BIT_DATA") as ds:
        for name, size in [("Time",None),("nCells",2),("nVertLevels",3),("nVertLevelsP1",4),("nSoilLevels",4),("StrLen",64)]:
            ds.createDimension(name,size)
        ds.setncattr("file_id","abcdefghij")
        ds.setncattr("parent_id","parent-one\n")
        ds.setncattr("config_nsoillevels",np.int32(4))
        ds.createVariable("zgrid","f4",("nCells","nVertLevelsP1"))[:] = [[0,50,1000,3000],[2500,2550,3500,5000]]
        ds.createVariable("tslb","f4",("Time","nCells","nSoilLevels"))[0] = np.arange(8,dtype="f4").reshape(2,4)+280
        for name in ("qv","qc","qr"):
            ds.createVariable(name,"f4",("Time","nCells","nVertLevels"))[0] = 0
        ds.createVariable("latCell","f4",("nCells",))[:] = np.deg2rad([35.,35.1])
        ds.createVariable("lonCell","f4",("nCells",))[:] = np.deg2rad([-97.,-97.1])
        ds.createVariable("rho","f4",("Time","nCells","nVertLevels"))[0] = [[1.2,.95,.75]]*2
        ds.createVariable("theta","f4",("Time","nCells","nVertLevels"))[0] = [[300.,305.,315.]]*2
        ds.createVariable("xland","f4",("Time","nCells"))[0] = [1.,2.]
        ds.createVariable("isltyp","i4",("nCells",))[:] = [14,6]
        for name,text in [("xtime","2026-10-07_12:00:00"),("initial_time","2026-10-07_12:00:00"),("mminlu","MODIFIED_IGBP_MODIS_NOAH")]:
            dims=("StrLen",) if name=="initial_time" else ("Time","StrLen")
            var=ds.createVariable(name,"S1",dims)
            payload=np.frombuffer(text.ljust(64).encode(),dtype="S1")
            if len(dims)==1: var[:]=payload
            else: var[0]=payload
    names="qv,qc,qr,qi,qs,qg,ni,nr,nc,nwfa,nifa,ng,volg"
    from hexcore.init_physics import prepare_forecast_init
    from hexcore.physics_suites import resolve_physics
    physics=resolve_physics("rrfs_v2_tempo_candidate",
        {"config_lsm_scheme":"sf_noahmp", "config_frac_seaice":False, "config_gwdo_scheme":"off"}, nominal_dx_m=3000).receipt()
    original=path.read_bytes()
    prepared, automatic=prepare_forecast_init(path,tmp_path/"auto.nc",physics,None)
    assert path.read_bytes()==original
    assert automatic["source"]=="wif-climatology"
    assert automatic["soil_categories_reconciled"] == 2
    with Dataset(prepared) as ds:
        assert len(ds.dimensions["nSoilLevels"])==4
        assert ds.variables["nwfa"][0].min()>0
        np.testing.assert_array_equal(ds.variables["isltyp"][:], [8,14])
    assert prepare_forecast_init(prepared,tmp_path/"unused.nc",physics,"2026-10-07_12:00:00")== (prepared,None)
    ruc=resolve_physics("rrfs_v2_tempo_candidate",{"config_gwdo_scheme":"off"}, nominal_dx_m=3000).receipt()
    with pytest.raises(ValueError,match="no source-analysis provenance"):
        prepare_forecast_init(path,tmp_path/"ruc.nc",ruc,"2026-10-07_12:00:00")
    receipt=json.loads(subprocess.check_output([native,"--augment-init",str(path),"--scalar-names",names],text=True))
    assert receipt["source"]=="synthetic"
    with Dataset(path) as ds:
        assert ds.getncattr("file_id")=="abcdefghij"
        assert ds.getncattr("parent_id")=="parent-one\n"
        assert ds.variables["tslb"][0].tobytes()==(np.arange(8,dtype="f4").reshape(2,4)+280).tobytes()
        for name in names.split(','): assert ds.variables[name].shape==(1,2,3)
        for name in ("nwfa","nifa"):
            field=ds.variables[name][0]
            assert np.isfinite(field).all() and (field>0).all()
            assert field[0,0]==field[0,1]
        assert ds.variables["nwfa"][0,1,0]<ds.variables["nwfa"][0,0,0]
    status=json.loads(subprocess.check_output([native,"--aerosol-status",str(path)],text=True))
    assert status["analysis_pair"] is True
    assert status["soil_category_mismatches"] == 2
    receipt=json.loads(subprocess.check_output([native,"--augment-init",str(path),"--scalar-names",names],text=True))
    assert receipt["source"]=="analysis"
    path.with_name(path.name+'.provenance.json').write_text(json.dumps({'source':'synthetic'}))
    original = path.read_bytes()
    replacement, receipt = prepare_forecast_init(path,tmp_path/'replace-synthetic.nc',physics,None)
    assert receipt['source'] == 'wif-climatology'
    assert path.read_bytes() == original
    with Dataset(path) as old, Dataset(replacement) as new:
        assert old['nwfa'][:].tobytes() != new['nwfa'][:].tobytes()
