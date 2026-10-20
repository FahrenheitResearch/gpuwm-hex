"""Upper damping contracts against compiled unchanged native source blocks."""
from pathlib import Path
import os
import numpy as np
import pytest
from dataclasses import asdict


def production_case():
    from test_fct_v841 import native_case
    from hexcore.config_v841 import V841UpperDampingDryConfig
    case=native_case.__wrapped__()
    values=asdict(case["config"])
    values.update(config_horiz_mixing="2d_smagorinsky",config_visc4_2dsmag=0.05,
        config_smagorinsky_coef=0.125,config_mpas_cam_coef=2.0,config_rayleigh_damp_u=True,
        config_zd=16000.0,config_number_cam_damping_levels=8)
    case["config"]=V841UpperDampingDryConfig(**values)
    return case


def native_deck():
    folder=os.environ.get("HEXCORE_NATIVE_UPPER_FIXTURE")
    if not folder:pytest.skip("set HEXCORE_NATIVE_UPPER_FIXTURE to the compiled native block deck")
    folder=Path(folder)
    initial=np.fromfile(folder/"upper-input.bin",np.float32).reshape(4,3,59).transpose(0,2,1).copy()
    expected=np.fromfile(folder/"upper-output.bin",np.float32).reshape(2,3,59).transpose(0,2,1).copy()
    return initial,expected


def test_rust_matches_native_cam_and_rayleigh_bits_and_mutation():
    from hexcore.upper_damping_v841 import apply_cam_floor,apply_rayleigh_u
    initial,expected=native_deck()
    viscosity=initial[0].copy();tendency=initial[1].copy()
    apply_cam_floor(viscosity,length=12000,coefficient=2,levels=8)
    apply_rayleigh_u(tendency,initial[2],initial[3],levels=6,days=5,solved=2)
    np.testing.assert_array_equal(viscosity.view(np.uint32),expected[0].view(np.uint32))
    np.testing.assert_array_equal(tendency.view(np.uint32),expected[1].view(np.uint32))
    assert np.count_nonzero(viscosity!=initial[0])>0
    assert np.count_nonzero(tendency!=initial[1])>0
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(initial[0].view(np.uint32),expected[0].view(np.uint32))
    with pytest.raises(ValueError):apply_cam_floor(viscosity,length=12000,coefficient=2,levels=60)


def test_native_upper_settings_reach_the_namelist_and_forecast_builders():
    from hexcore.config import dry_dycore_config_from_namelist
    from hexcore.drivers.run_cuda_v841_forecast import build_forecast_config
    parsed=dry_dycore_config_from_namelist('''&nhyd_model
 config_horiz_mixing="2d_smagorinsky"
 config_visc4_2dsmag=0.05
 config_smagorinsky_coef=0.125
/
&damping
 config_mpas_cam_coef=2.0
 config_number_cam_damping_levels=8
 config_rayleigh_damp_u=.true.
 config_zd=16000.0
/
''',source_release="v8.4.1")
    config=build_forecast_config(dt_seconds=20,time_integration_order=2,acoustic_substeps=4,
        radiation_seconds=900,monotonic=True,cam_coefficient=2,cam_damping_levels=8,
        rayleigh_u=True,damping_start_m=16000)
    config.validate()
    assert parsed.config_number_cam_damping_levels==config.config_number_cam_damping_levels==8
    assert parsed.config_rayleigh_damp_u is config.config_rayleigh_damp_u is True
    # Unset keys take MPAS's Registry defaults, the values the forecast lane runs.
    assert parsed.config_positive_definite is config.config_positive_definite is False
    assert parsed.config_apvm_upwinding == config.config_apvm_upwinding == 0.5


def test_cpu_executes_both_upper_operators_at_the_native_driver_sites():
    from hexcore.driver import DryDycoreDriver
    case=production_case()
    driver=DryDycoreDriver(case["mesh"],case["native"].vertical_grid,case["reference"],case["config"],
        terrain_metrics=case["native"].terrain_metrics,advection_coefficients=case["cpu"].advection_coefficients,
        reference_wind_profiles=case["profiles"])
    result=driver.step(case["state"].copy(),saved_diagnostics=case["saved"]).state
    for field in ("rho","rho_theta","rho_u","rho_w","scalars"):
        assert np.isfinite(getattr(result,field)).all()
    assert result.time_seconds==20


@pytest.mark.gpu
def test_cuda_executes_the_upper_operators_and_repeats_the_real_mesh_step(monkeypatch):
    from hexcore.cuda_driver import CudaDryDycoreDriver
    from hexcore.drivers.run_cuda_v841_full_physics_x4 import attach_inactive_zero_deformation
    import hexcore.cuda_upper_damping_v841 as upper
    case=production_case()
    if "defc_a" not in case["mesh"].arrays:attach_inactive_zero_deformation(case["mesh"])
    launch=upper._launch;calls=[]
    def recording(name,*args,**kwargs):
        calls.append(name);return launch(name,*args,**kwargs)
    monkeypatch.setattr(upper,"_launch",recording)
    outputs=[]
    for _ in range(2):
        calls.clear()
        driver=CudaDryDycoreDriver.from_host(case["mesh"],case["state"].copy(),case["native"].vertical_grid,
            case["reference"],case["config"],saved_diagnostics=case["saved"],
            terrain_metrics=case["native"].terrain_metrics,advection_coefficients=case["cpu"].advection_coefficients,
            reference_wind_profiles=case["profiles"])
        result=driver.step().state
        assert calls.count("upper_cam_f32")==3
        assert calls.count("upper_rayleigh_f32")==9
        outputs.append({field:np.asarray(getattr(result,field)).view(np.uint32)
            for field in ("rho","rho_theta","rho_u","rho_w","scalars")})
    for field in outputs[0]:np.testing.assert_array_equal(outputs[0][field],outputs[1][field])


@pytest.mark.gpu
def test_cuda_matches_the_real_native_upper_block_arrays():
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_upper_damping_v841 import apply_cam_floor_cuda,apply_rayleigh_u_cuda
    initial,expected=native_deck();cache=KernelCache()
    viscosity=cp.asarray(initial[0]);tendency=cp.asarray(initial[1])
    apply_cam_floor_cuda(viscosity,length=12000,coefficient=2,levels=8,cache=cache)
    apply_rayleigh_u_cuda(tendency,cp.asarray(initial[2]),cp.asarray(initial[3]),
        levels=6,days=5,solved=2,cache=cache)
    np.testing.assert_array_equal(cp.asnumpy(viscosity).view(np.uint32),expected[0].view(np.uint32))
    np.testing.assert_array_equal(cp.asnumpy(tendency).view(np.uint32),expected[1].view(np.uint32))
    with pytest.raises((TypeError,ValueError)):
        apply_cam_floor_cuda(viscosity.astype(cp.float64),length=12000,coefficient=2,levels=8,cache=cache)
    with pytest.raises(ValueError):
        apply_rayleigh_u_cuda(tendency,cp.asarray(initial[2][:,:2]),cp.asarray(initial[3]),cache=cache)
    with pytest.raises(ValueError):
        apply_rayleigh_u_cuda(tendency,cp.asarray(initial[2]),cp.asarray(initial[3]),days=0,cache=cache)
