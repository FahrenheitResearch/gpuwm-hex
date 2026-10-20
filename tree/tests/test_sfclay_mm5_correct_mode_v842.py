"""RP3 slot declaration and exact-mode isolation at the MPAS driver boundary."""
from types import SimpleNamespace

import pytest
from hexcore.mpasphys import sfclay_mm5_v842 as mm5
from hexcore.mpasphys.driver_v842 import Backend, resolve_namelist, table_declarations


@pytest.mark.parametrize('scheme', mm5.SCHEMES)
def test_RP3_owned_slot_gap_closes_and_exact_remains_unchanged(scheme):
    text="&physics\nconfig_physics_suite='mesoscale_reference'\nconfig_sfclayer_scheme="+repr(scheme)+"\n/"
    backends={'surface_layer':'hex-cuda'}
    exact=resolve_namelist(text,backends=backends,physics_mode='exact')
    correct=resolve_namelist(text,backends=backends,physics_mode='correct')
    assert exact.corrections==frozenset() and not exact.gaps
    assert table_declarations('surface_layer',scheme,Backend.HEX_CUDA)=={'RP3'}
    assert not any(g.finding=='RP3' and g.slot=='surface_layer' for g in correct.gaps)
    # Other owning slots are independent: never hide an unimplemented seaice fix.
    assert any(g.finding=='RP3' and g.slot=='seaice' for g in correct.gaps)
    assert 'RP3' not in correct.corrections
    def declares(slot, scheme, backend):
        return {'RP3'} if slot=='seaice' else table_declarations(slot,scheme,backend)
    complete=resolve_namelist(text,backends=backends,physics_mode='correct',declarations=declares)
    assert 'RP3' in complete.corrections
    assert not any(g.finding=='RP3' for g in complete.gaps)
    assert mm5._cuda_text(exact.corrections)==mm5.CUDA_SOURCE
    assert mm5._cuda_text(complete.corrections)!=mm5.CUDA_SOURCE


@pytest.mark.parametrize('scheme', mm5.SCHEMES)
@pytest.mark.parametrize('corrections', [(), ('RP3',), ('RP1',)])
def test_slot_passes_only_its_applied_corrections(monkeypatch,scheme,corrections):
    seen=[]
    monkeypatch.setattr(mm5,'run_surface_layer',lambda **kw:seen.append(kw))
    call=SimpleNamespace(scheme=scheme,pools={},options={'config_frac_seaice':True},
                         corrections=corrections)
    mm5.run_slot(call)
    assert seen[0]['corrections']==(('RP3',) if 'RP3' in corrections else ())
    assert seen[0]['scheme']==scheme


def test_changed_source_cannot_silently_drop_RP3(monkeypatch):
    monkeypatch.setattr(mm5,'CUDA_SOURCE',mm5.CUDA_SOURCE.replace('sea[41*n+i]=fmaxf(sst[i],271.4f);',''))
    with pytest.raises(RuntimeError,match='RP3'):
        mm5._cuda_text(('RP3',))


def test_direct_registration_declares_both_owned_slots():
    rows=[]
    registry=SimpleNamespace(register=lambda *args,**kw: rows.append((args,kw)))
    mm5.register_hex_cuda(registry,lambda call:{})
    assert {args[1] for args,kw in rows}==set(mm5.SCHEMES)
    assert all(set(kw['corrections'])=={'RP3'} for args,kw in rows)
