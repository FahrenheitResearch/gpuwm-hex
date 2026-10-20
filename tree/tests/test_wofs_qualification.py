"""Receipt corruption checks, standard library only, CPU only."""
import copy,importlib.util,unittest
from pathlib import Path
V=Path(__file__).parents[1]/'verification/wofs-ensemble'
spec=importlib.util.spec_from_file_location('qualify',V/'qualify.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

def fixture():
    gate=dict(finite_all_fields=True,positive_rho_theta_pressure=True,
        precipitation_nonnegative=True,moisture_negative_counts={'qv':0})
    run=dict(fullstep_lts=dict(base_dt=18.75,registers_empty_at_end=True,
        dry_mass_relative_drift=1e-10,dry_mass_max_relative_excursion=2e-10),
        refusal=None,steps_executed=384,steps_requested=384,
        history_labels={'0':'t0','192':'t1','384':'t2'},physical_gates={'0':gate,'384':gate})
    forecast=dict(status='passed',physics={'resolution':{'selectors':{
        'config_microp_scheme':'mp_wsm6','config_convection_scheme':'cu_ntiedtke'}}},
        driver_receipt=dict(status='passed',forecast=run,arwen_git={
            state:dict(head='engine',clean=True) for state in ('before','after')}))
    jobs=[dict(id=name,returncode=0,resident_devices=[0],segments=[dict(
        visible_device='UUID-'+name,grid_sha256='mesh')]) for name in ('a','b')]
    return dict(status='forecast_passed',members=2,cycle_hours=2,jobs=jobs),dict(
        members=[{'id':'a'},{'id':'b'}]),{name:copy.deepcopy(forecast) for name in ('a','b')}

class QualificationTests(unittest.TestCase):
    def test_actual_receipt_identity_physics_clock_and_gates_are_required(self):
        receipt,manifest,forecasts=fixture()
        self.assertEqual(module.qualify(receipt,manifest,forecasts,'engine')['qualified_members'],2)
        forecasts['a']['physics']['resolution']=None
        forecasts['a']['driver_receipt']['physics_backend']='wsm6_column'
        forecasts['a']['driver_receipt']['convection']={'scheme':'cu_ntiedtke'}
        module.qualify(receipt,manifest,forecasts,'engine')
        forecasts['a']['driver_receipt']['physics_backend']='unknown'
        with self.assertRaisesRegex(ValueError,'missing actual microphysics'):
            module.qualify(receipt,manifest,forecasts,'engine')
        receipt,manifest,forecasts=fixture()
        receipt['jobs'][1]['segments'][0]['visible_device']='UUID-a'
        with self.assertRaisesRegex(ValueError,'share a card'): module.qualify(receipt,manifest,forecasts,'engine')
        receipt,manifest,forecasts=fixture()
        forecasts['a']['driver_receipt']['arwen_git']['after']['head']='other'
        with self.assertRaisesRegex(ValueError,'engine identity'): module.qualify(receipt,manifest,forecasts,'engine')

        receipt,manifest,forecasts=fixture()
        manifest['members'][1]['forecast_flags']=['--physics','config_microp_scheme=mp_thompson']
        with self.assertRaisesRegex(ValueError,'microphysics'): module.qualify(receipt,manifest,forecasts,'engine')
        forecasts['b']['physics']['resolution']['selectors']['config_microp_scheme']='mp_thompson'
        module.qualify(receipt,manifest,forecasts,'engine')
        forecasts['b']['driver_receipt']['forecast']['steps_executed']=376
        with self.assertRaisesRegex(ValueError,'endpoint'): module.qualify(receipt,manifest,forecasts,'engine')

        receipt,manifest,forecasts=fixture()
        forecasts['a']['driver_receipt']['forecast']['fullstep_lts']['registers_empty_at_end']=False
        with self.assertRaisesRegex(ValueError,'transaction'): module.qualify(receipt,manifest,forecasts,'engine')
        receipt,manifest,forecasts=fixture()
        forecasts['a']['driver_receipt']['forecast']['physical_gates']['384']['finite_all_fields']=False
        with self.assertRaisesRegex(ValueError,'physical gate'): module.qualify(receipt,manifest,forecasts,'engine')

if __name__=='__main__': unittest.main()
