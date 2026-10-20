"""Every published scheme handoff is selectable without changing native peers."""
import pytest
from hexcore.mpasphys import Backend, resolve_namelist
from hexcore.physics_suites import load_table


@pytest.mark.parametrize('slot,scheme', [
    ('microphysics', 'mp_kessler'), ('microphysics', 'mp_wsm6'),
    ('convection', 'cu_kain_fritsch'), ('convection', 'cu_tiedtke'),
    ('convection', 'cu_ntiedtke'), ('convection', 'cu_grell_freitas'),
    ('land_surface', 'sf_noah'), ('pbl', 'bl_ysu'), ('gwdo', 'bl_ysu_gwdo'), ('ngw', 'ugwp_ngw'),
    ('cloudiness', 'cld_incidence'), ('cloudiness', 'cld_fraction'),
    ('cloudiness', 'cld_fraction_thompson'), ('radiation_lw', 'rrtmg_lw'),
])
def test_ready_scheme_selection_has_one_cuda_backend_and_public_descriptor(slot, scheme):
    rows = load_table()['mpasphys_v842']['slots']
    row = rows[slot]
    option = row.get('option_values', {}).get(scheme, scheme)
    value = '.true.' if option is True else repr(option)
    # NGW is embedded in UGWP's driver, as declared by the same suite table.
    companion = 'config_gwdo_scheme="bl_ugwp_gwdo"\n' if slot == 'ngw' else ''
    text = '&physics\nconfig_physics_suite="mesoscale_reference"\n' + companion
    text += row['selector'] + '=' + value + '\n/'
    selected = resolve_namelist(text, backends={slot: 'hex-cuda'})
    native = resolve_namelist(text)
    assert selected.selectors == native.selectors
    assert selected.active(slot) and selected.backends[slot] == Backend.HEX_CUDA
    assert all(backend == Backend.NATIVE_CPU for name, backend in selected.backends.items() if name != slot)
    descriptor = row['cuda_entries'][scheme]
    assert descriptor['module'] and descriptor['call']
