import json
from pathlib import Path
import numpy as np
import pytest
from netCDF4 import Dataset
from hexcore.cycle.soil_wetness import source_contract,remap_cold_init
from hexcore.cycle.land_state import context,read,validate_land,LAND_ATTRIBUTE


def pair(tmp_path):
    target=tmp_path/'init.nc'; donor=tmp_path/'donor.nc'
    with Dataset(target,'w',format='NETCDF3_64BIT_DATA') as d:
        for n,size in [('Time',1),('nCells',3),('nSoilLevels',4),('StrLen',64)]:d.createDimension(n,size)
        for n,v in {'isltyp':[1,6,14], 'ivgtyp':[2,13,17], 'xland':[1,1,2], 'xice':[0,0,0], 'latCell':[0,0,0], 'lonCell':[0,0,0]}.items():
            d.createVariable(n,'f4',('nCells',))[:]=v
        d.createVariable('isice_lu','i4').assignValue(15)
        d.createVariable('mminlu','S1',('StrLen',))[:]=np.array(list('MODIFIED_IGBP_MODIS_NOAH'.ljust(64)),dtype='S1')
        d.createVariable('dzs','f4',('nSoilLevels',))[:]=[.1,.3,.6,1]
        for n,vals in {'smois':[.42,.3,1], 'sh2o':[.2,.1,1], 'tslb':[280,281,282]}.items():
            d.createVariable(n,'f4',('Time','nCells','nSoilLevels'))[0]=np.array(vals)[:,None]
    with Dataset(donor,'w') as d:
        for n,size in [('time',1),('y',2),('x',2)]:d.createDimension(n,size)
        t=d.createVariable('time','f8',('time',));t.units='hours since 2023-07-08 12:00:00';t[:]=0
        for n,dim in [('grid_yt','y'),('grid_xt','x')]:d.createVariable(n,'f8',(dim,))[:]=[0,.1]
        fields={'land':1,'vtype':2,'sotyp':6,'veg':50}
        for n in range(1,5):fields.update({f'soilw{n}':.3,f'soill{n}':.15})
        for n,value in fields.items():d.createVariable(n,'f4',('time','y','x'))[:]=value
    return target,donor


def test_cold_state_remaps_hydraulics_and_liquid_fraction_with_receipt(tmp_path):
    target,donor=pair(tmp_path)
    receipt=remap_cold_init(target,donor,source_contract(['NCEP GFS Analysis']),'2023-07-08_12:00:00')
    assert receipt['cold_before_bound_violations']==4
    assert receipt['cold_after_bound_violations']==0 and receipt['clipped_values']==0
    assert receipt['source_rejected_columns']==0
    with Dataset(target) as d:
        fields={n:read(d[n]) for n in ('smois','sh2o','tslb')}
        validate_land(fields,context(d))
        np.testing.assert_allclose(fields['sh2o'][:2],fields['smois'][:2]*.5,rtol=0,atol=0)
        np.testing.assert_array_equal(fields['smois'][2],1)
        np.testing.assert_array_equal(fields['sh2o'][2],1)
        np.testing.assert_array_equal(fields['tslb'],np.broadcast_to([[280],[281],[282]],(3,4)))
        assert not d.ncattrs()  # do not rewrite a large classic header to add provenance
    assert json.loads(Path(str(target)+'.soil-remap.json').read_text())==receipt


@pytest.mark.parametrize('bad',['time','classes','distance','warm'])
def test_bad_donor_or_warm_target_refuses_before_any_write(tmp_path,bad):
    target,donor=pair(tmp_path);contract=source_contract(['NCEP GFS Analysis'])
    if bad=='warm':
        with Dataset(target,'r+') as d:d.setncattr(LAND_ATTRIBUTE,'["smois"]')
    elif bad=='distance':contract['max_donor_distance_km']=-1
    else:
        with Dataset(donor,'r+') as d:d['time' if bad=='time' else 'sotyp'][:]=1 if bad=='time' else 6.2
    before=target.read_bytes()
    with pytest.raises(ValueError):remap_cold_init(target,donor,contract,'2023-07-08_12:00:00')
    assert before==target.read_bytes()
    assert not Path(str(target)+'.soil-remap.json').exists()


def test_init_door_runs_default_remap_before_output_receipt(tmp_path,monkeypatch):
    import argparse
    from hexcore import init_door as door
    target,donor=pair(tmp_path);seed=tmp_path/'seed.nc';target.rename(seed)
    engine=tmp_path/'engine'
    engine.write_text('#!/usr/bin/env python3\nimport sys,shutil\n'
                      f'shutil.copyfile({str(seed)!r},sys.argv[sys.argv.index("--out")+1])\n'
                      'print("{\\"n_cells\\":3,\\"n_vert_levels\\":2}")\n')
    engine.chmod(0o700)
    parser=argparse.ArgumentParser();door.add_init_parser(parser.add_subparsers())
    args=parser.parse_args(['init','--met',str(donor),'--static',str(seed),
        '--capsule',str(seed),'--reference',str(seed),'--out',str(target),
        '--start-time','2023-07-08_12:00:00','--nfglevels','34','--nfgsoillevels','4',
        '--extrap-airtemp','constant','--use-spechumd','no','--theta-adv-order','3',
        '--coef-3rd-order','0.25','--virtual-factor','reproduce-fortran',
        '--deep-soil-moisture','reproduce-fortran','--landuse-table','MODIFIED_IGBP_MODIS_NOAH',
        '--frac-seaice','yes','--tsk-seaice-threshold','100','--oned-underflow','preserve',
        '--engine',str(engine),'--soil-donor',str(donor)])
    monkeypatch.setattr(door,'scan_met_file',lambda *a,**kw:door.MetInventory(
        ('SM000010',),('2023-07-08_12:00:00',),34,True,('NCEP GFS Analysis',)))
    monkeypatch.setattr(door,'prepare_vertical_source',lambda *a:door.VerticalSource(
        mode='compatibility-capsule',capsule=seed,reference=seed,summary={}))
    assert door.run_init_command(args)==0
    receipt=json.loads(Path(str(target)+'.provenance.json').read_text())
    assert receipt['cold_soil_remap']['applied'] is True
    assert receipt['cold_soil_remap']['cold_before_bound_violations']==4
    with Dataset(target) as d:
        validate_land({n:read(d[n]) for n in ('smois','sh2o','tslb')},context(d))
