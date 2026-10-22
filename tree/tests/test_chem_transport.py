"""Chemistry admission, thermodynamic exclusion and process-safe identity."""
import json
import pytest
from hexcore.chem_transport import ChemistrySpecies, configure, history_units, load_config
from hexcore.species_row import species_row_for_names, AEROSOL_MASS, WSM6_SPECIES_ROW

def config(tmp_path, species=None):
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps({"schema": "gpuwm-hex.chemistry/v1", "species": species or [
        {"name": "smoke", "dry_deposition_m_s": 0.001, "settling_m_s": 0.0001,
         "mass_extinction_m2_kg": 4500.0},
        {"name": "dust_1", "dry_deposition_m_s": 0.002, "settling_m_s": 0.01,
         "mass_extinction_m2_kg": 800.0}]}))
    return path

def test_arbitrary_species_append_transport_without_moisture_loading(tmp_path):
    row = configure(config(tmp_path))
    assert configure(config(tmp_path)) == row
    names = WSM6_SPECIES_ROW.names() + ("smoke", "dust_1")
    declared = species_row_for_names(names)
    assert declared.names() == names
    assert declared.n_mass_loading == 6
    assert declared.declaration("smoke").species_class == AEROSOL_MASS
    assert declared.declaration("smoke").transported
    assert declared.declaration("smoke").units == "kg kg^{-1}"
    assert row.appended_scalar_names == ("smoke", "dust_1")

@pytest.mark.parametrize("field", ["dry_deposition_m_s", "settling_m_s", "mass_extinction_m2_kg"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, 1e300, "bad"])
def test_bad_physics_parameters_refused(field, value):
    args = dict(name="smoke", dry_deposition_m_s=0., settling_m_s=0., mass_extinction_m2_kg=0.)
    args[field] = value
    with pytest.raises(ValueError):
        ChemistrySpecies(**args)

def test_atmospheric_overlap_refused(tmp_path):
    path = config(tmp_path, [{"name": "qv", "dry_deposition_m_s":0.,
                             "settling_m_s":0., "mass_extinction_m2_kg":0.}])
    with pytest.raises(ValueError, match="overlap"):
        configure(path)

def test_duplicate_species_refused(tmp_path):
    path = config(tmp_path)
    raw = json.loads(path.read_text())
    raw["species"].append(raw["species"][0])
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unique"):
        load_config(path)

def test_product_units_are_physical():
    assert history_units("smoke_surface_concentration") == "ug m-3"
    assert history_units("dust_1_column_mass") == "kg m-2"
    assert history_units("sea_salt_3_settled_mass") == "kg m-2"
    assert history_units("aod550") == "1"

def test_complete_initialized_ice_preserved_before_aerosol_storage(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from hexcore.drivers import run_cuda_v841_forecast as forecast
    configure(config(tmp_path))
    atmospheric = WSM6_SPECIES_ROW.names()
    names = atmospheric + ('smoke','dust_1')
    assert forecast.select_initial_scalar_names(dict.fromkeys(atmospheric), names) == atmospheric
    monkeypatch.setattr(forecast.proof, 'N_LEVELS', 3)
    monkeypatch.setattr(forecast.proof, 'N_CELLS', 2)
    values = np.arange(36,dtype=np.float32).reshape(6,3,2)
    before = values.tobytes()
    state=SimpleNamespace(scalars=values)
    receipt=forecast.prepare_initial_scalar_block(state,source_names=atmospheric,model_names=names,negative_qv={})
    assert state.scalars[:6].tobytes() == before
    assert not state.scalars[6:].any()
    assert receipt['source_present'] == list(atmospheric)

@pytest.mark.parametrize('name',['rho','aod550','smoke_column_mass'])
def test_output_name_collisions_refused(tmp_path,name):
    rows=[dict(name='smoke',dry_deposition_m_s=0.,settling_m_s=0.,mass_extinction_m2_kg=0.),
          dict(name=name,dry_deposition_m_s=0.,settling_m_s=0.,mass_extinction_m2_kg=0.)]
    with pytest.raises(ValueError,match='overlap'):
        load_config(config(tmp_path,rows))

def test_named_physics_not_silently_replaced():
    from argparse import Namespace
    from hexcore.chem_transport import admit_arguments
    with pytest.raises(ValueError,match='cannot be silently replaced'):
        admit_arguments(Namespace(chemistry_config='x',physics_backend='mynn_column',physics=None,physics_suite=None))

def test_forecast_refuses_unpopulated_chemistry():
    from types import SimpleNamespace
    from hexcore.chem_transport import attach
    backend = SimpleNamespace(chemistry=SimpleNamespace(raw={}))
    with pytest.raises(ValueError,match='dated initial composition'):
        attach(backend,{},object())

def test_global_door_forwards_chemistry_config(tmp_path,monkeypatch):
    from hexcore.cli import build_parser
    from hexcore import global_door
    monkeypatch.setattr(global_door,'_visible_cards',lambda: [])
    path=config(tmp_path)
    args=build_parser().parse_args(['global','--row','global-120km','--cycle','2026-10-19T00',
          '--work-dir',str(tmp_path/'run'),'--chemistry-config',str(path)])
    plan=global_door.plan(args)
    argv=plan['commands']['forecast']
    assert argv[argv.index('--chemistry-config')+1] == str(path)

def test_column_coupling_uses_endpoint_density(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from hexcore import chem_transport as chem
    from hexcore import cuda_arwen_physics_v841 as base
    from hexcore import cuda_physics_prep_v841 as prep
    endpoint = SimpleNamespace(rho_dry=object(), z_p=object())
    calls = []
    class Base:
        def __init__(self, **kwargs):
            self._prep_geometry = object()
            self._kernel_cache = object()
            self._les_columns = SimpleNamespace(rho_dry=object(), z_p=object())
            self._seam = SimpleNamespace(_driver=SimpleNamespace(fields={'exch_h':np.ones((3,2))}))
        def finish_step(self, **kwargs):
            calls.append(('base',kwargs))
            return 'update'
    def prepare(atmosphere, **kwargs):
        calls.append(('prepare',kwargs))
        return endpoint
    def runtime(*args, **kwargs):
        return SimpleNamespace(nlev=3,ncells=2,step=lambda a,**kw:calls.append(('chem',kw)))
    monkeypatch.setattr(base,'PersistentTwoPhaseCudaPhysicsBackendV841',Base)
    monkeypatch.setattr(prep,'prepare_mpas_to_phys_cuda_v841',prepare)
    monkeypatch.setattr(chem,'ChemistryTransport',runtime)
    row = configure(config(tmp_path))
    backend = chem.build_column_backend(row, constructor=SimpleNamespace(n_levels=3,n_columns=2),
                 prep_geometry=object(),kernel_cache=object(),cell_area_m2=np.ones(2))
    assert backend.finish_step(atmosphere=object(),scalar_names=WSM6_SPECIES_ROW.names(),
                               dt=30.,refl_10cm_due=True) == 'update'
    assert [c[0] for c in calls] == ['base','prepare','chem']
    assert calls[1][1]['post_rk_wsm6'] and calls[1][1]['phase2_only']
    assert calls[2][1]['rho'] is endpoint.rho_dry
    assert calls[2][1]['z'] is endpoint.z_p
