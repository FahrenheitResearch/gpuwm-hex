"""Check actual closed-member receipts, without reopening model fields."""
import os
from pathlib import Path
import json,sys
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))

def qualify(ensemble,manifest,forecasts,engine_commit):
    rows={r['id']:r for r in manifest['members']}
    jobs=ensemble['jobs']
    if ensemble['status']!='forecast_passed' or len(jobs)!=ensemble['members']:
        raise ValueError('ensemble did not close its complete roster')
    if len({j['id'] for j in jobs})!=len(jobs): raise ValueError('duplicate member receipt')
    devices=[j['segments'][0]['visible_device'] for j in jobs]
    # This showcase was a simultaneous grant. A future waves run may reuse
    # a card, but this receipt must substantiate its eight-card claim.
    if len(set(devices))!=len(jobs): raise ValueError('simultaneous members share a card')
    result=[]
    for job in jobs:
        row=rows[job['id']]; record=forecasts[job['id']]
        driver=record['driver_receipt']; run=driver['forecast']
        if job['returncode'] or record['status']!='passed' or driver['status']!='passed':
            raise ValueError('member did not pass every door')
        if job['resident_devices']!=[0]: raise ValueError('member did not use one logical card')
        lts=run['fullstep_lts']
        if run['refusal'] is not None or not lts['registers_empty_at_end']:
            raise ValueError('LTS did not close its transaction')
        base_dt=lts['base_dt']
        steps=round(ensemble['cycle_hours']*3600/base_dt)
        if run['steps_executed']!=steps or run['steps_requested']!=steps:
            raise ValueError('member stopped short of forecast endpoint')
        if len(run['history_labels'])!=round(ensemble['cycle_hours'])+1:
            raise ValueError('member did not publish every required hour')
        for state in ('before','after'):
            identity=driver['arwen_git'][state]
            if identity['head']!=engine_commit or not identity['clean']:
                raise ValueError('executed engine identity changed')
        expected='mp_wsm6'
        flags=manifest.get('forecast_flags',[])+row.get('forecast_flags',[])
        for flag in flags:
            if flag.startswith('config_microp_scheme='): expected=flag.split('=',1)[1]
        resolution=record['physics']['resolution']
        if resolution is None:
            # The default WSM6 path has no override-resolution receipt.
            # Require the executed backend and constructor decision instead.
            if driver['physics_backend']!='wsm6_column':
                raise ValueError('missing actual microphysics selection')
            selectors=dict(config_microp_scheme='mp_wsm6',
                config_convection_scheme=driver['convection']['scheme'])
        else:
            selectors=resolution['selectors']
        if selectors['config_microp_scheme']!=expected:
            raise ValueError('actual microphysics differs from roster')
        if selectors['config_convection_scheme']!='cu_ntiedtke':
            raise ValueError('case scale-aware convection selection changed')
        gates=run['physical_gates']
        for gate in gates.values():
            if not all(gate[k] for k in ('finite_all_fields','positive_rho_theta_pressure','precipitation_nonnegative')):
                raise ValueError('a published frame failed a physical gate')
        result.append(dict(member=job['id'],visible_device=job['segments'][0]['visible_device'],
            microphysics=selectors['config_microp_scheme'],physics_selectors=selectors,
            base_steps=steps,frames=len(run['history_labels']),
            dry_mass_relative_drift=lts['dry_mass_relative_drift'],
            dry_mass_max_relative_excursion=lts['dry_mass_max_relative_excursion'],
            registers_empty_at_end=True,finite_published_frames=True,
            initial_negative_qv=gates['0']['moisture_negative_counts'].get('qv',0),
            final_negative_moisture=gates[str(steps)]['moisture_negative_counts'],
            grid_sha256=job['segments'][0]['grid_sha256']))
    return dict(qualified_members=len(jobs),members=result,
        scope='receipt consistency, completed clocks and published-frame gates; observations grade forecast skill')

def main(phase):
    manifest=json.loads((W/'prepared/ensemble.json').read_text())
    root=W/'runs'/phase
    receipt=json.loads((root/'ensemble-receipt.json').read_text())
    forecasts={j['id']:json.loads((root/j['id']/'history/forecast-receipt.json').read_text()) for j in receipt['jobs']}
    engine='7008877b6b2271c4bf58945d47e70dc9c7a33063'
    result=qualify(receipt,manifest,forecasts,engine)
    (W/'evidence'/f'{phase}-qualified.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main(sys.argv[1])
