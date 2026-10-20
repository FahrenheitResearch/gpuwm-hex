"""Case clock and member identity are inputs, not helper constants."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('parity',Path(__file__).parents[1]/'verification/wofs-ensemble/parity.py')
parity=importlib.util.module_from_spec(spec); spec.loader.exec_module(parity)

def test_byte_check_command_uses_manifest_assets_and_clock():
    manifest=dict(mesh='patch',grid='grid.nc',static='static.nc',start_time='2020-05-04_06:00:00',
        members=[dict(init='init.nc',source='historical member',grid='segment.nc')])
    command=parity.parity_command('python',Path('output'),'engine',manifest)
    assert command[command.index('--start-time')+1]==manifest['start_time']
    assert command[command.index('--init-source')+1]=='historical member'
    assert command[command.index('--grid')+1]=='segment.nc'
    assert '--fullstep-local-timestep' not in command
