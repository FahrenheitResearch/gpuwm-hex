"""Apply the committed case's physics roster to prepared manifest rows."""
import os
from pathlib import Path
import json
from hexcore.ensemble import SCHEMA, read_manifest

def apply_roster(data, case):
    if [m['id'] for m in data['members']] != case['members']:
        raise ValueError('prepared member roster differs from case')
    data['schema'] = SCHEMA
    data['forecast_flags'] = list(case['forecast_flags'])
    for member in data['members']:
        member['forecast_flags'] = case.get('member_physics',{}).get(member['id'],[])
    return data

if __name__=='__main__':
    w=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))
    path=w/'prepared/ensemble.json'
    case=json.loads((Path(__file__).parent/'case.json').read_text())
    data=apply_roster(json.loads(path.read_text()),case)
    path.write_text(json.dumps(data,indent=2)+'\n')
    read_manifest(path,8,3,forecast_only=True)
