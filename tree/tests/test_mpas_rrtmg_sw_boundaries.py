"""Defined large-ice output and native cloud rejection semantics."""
import json
import hashlib
import os
from pathlib import Path
import numpy as np
import pytest
from mpas_rrtmg_sw_oracle.fatal_boundaries import native_witness
from mpas_rrtmg_sw_oracle.aerosol_failures import native_failures
from mpas_rrtmg_sw_oracle.replay import read_swd, replay, replay_cloud, replay_mcica, replay_reftra, replay_composition

pytestmark = pytest.mark.gpu


def test_native_driver_decorrelation_defaults(tmp_path):
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
    directory = Path(root)
    fixture = directory / "driver_control_options.bin"
    native = np.frombuffer(fixture.read_bytes(), dtype=">i4").reshape(-1, 4)
    dump = read_swd(directory / "tables.swd")
    engine = MPASShortwave(raw_coefficients={k: cp.asarray(v) for k, v in dump.items()
                                           if k.startswith("kg") and v.ndim})
    overlaps = ("none", "random", "maximum_random", "maximum", "exponential", "exponential_random")
    decorators = ("constant", "latitude_varying", "", "unexpected")
    errors = []
    for c, d, overlap, idcor in native:
        configs = dict(config_radt_cld_overlap=overlaps[c-1],
                       config_radt_cld_dcorrlen=decorators[d-1], config_o3climatology=False)
        controls = engine.driver_controls(configs, xtime_s=np.float32(0),
                                          dt_radtsw=np.float32(60), curr_julday=np.float32(90),
                                          degrad=np.float32(.017453292), dpd=np.float32(.98630136))
        differing = int(controls["cldovrlp"] != overlap) + int(controls["idcor"] != idcor)
        if differing:
            errors.append(dict(cloud=int(c),decorrelation=int(d),different_words=differing))
    for cloud in ("", "unexpected"):
        configs = dict(config_radt_cld_overlap=cloud,
                       config_radt_cld_dcorrlen="latitude_varying", config_o3climatology=False)
        controls = engine.driver_controls(configs, xtime_s=np.float32(0),
                                          dt_radtsw=np.float32(60), curr_julday=np.float32(90),
                                          degrad=np.float32(.017453292), dpd=np.float32(.98630136))
        assert controls["cldovrlp"] == 0 and controls["idcor"] == 0
    (tmp_path / "driver-decorrelation-defaults.json").write_text(json.dumps(dict(
        cases=len(native), words=len(native)*2,
        differing_words=sum(row["different_words"] for row in errors), errors=errors,
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        native_process_count=1, whole_model_proven=False), indent=2)+"\n")
    assert not errors


def test_single_layer_mt_overlap(tmp_path):
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
    directory = Path(root)
    dump = read_swd(directory/"tables.swd")
    engine = MPASShortwave(raw_coefficients={k:cp.asarray(v) for k,v in dump.items() if k.startswith("kg") and v.ndim})
    assert replay_mcica(engine, directory/"mcica-single.bin", tmp_path/"mcica-single.json") == 0


def test_native_ozone_extrapolation(tmp_path):
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    directory = Path(root)
    assert replay(directory/"ozone-edges/tables.swd", directory/"ozone-edges/columns.swd",
                  tmp_path/"ozone-edges.json") == 0
    for mode in ("ozone-one-above", "ozone-one-below"):
        assert replay(directory/"tables.swd", directory/mode/"columns.swd",
                      tmp_path/(mode+".json")) == 0


def test_native_aerosol_rejections(tmp_path):
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
    directory = Path(root)
    native = native_failures(directory)
    dump = read_swd(directory/"tables.swd")
    engine = MPASShortwave(raw_coefficients={k:cp.asarray(v) for k,v in dump.items() if k.startswith("kg") and v.ndim})
    inputs = [cp.full((1,4),v,cp.float32) for v in (90000.,280.,.005,300.)]
    options = (
        {"aod_value":-.01}, {"aod_opt":2,"aod5502d":None},
        {"aod_opt":2,"aod5502d":-.01}, {"angexp_opt":2,"angexp2d":None},
        {"ssa_opt":1,"ssa_value":-.01}, {"ssa_opt":1,"ssa_value":1.01},
        {"ssa_opt":2,"aerssa2d":None}, {"ssa_opt":2,"aerssa2d":-.01},
        {"ssa_opt":2,"aerssa2d":1.01}, {"asy_opt":1,"asy_value":-.01},
        {"asy_opt":1,"asy_value":1.01}, {"asy_opt":2,"aerasy2d":None},
        {"asy_opt":2,"aerasy2d":-1.01}, {"asy_opt":2,"aerasy2d":1.01},
        {"aod_opt":0}, {"angexp_opt":0}, {"ssa_opt":0}, {"asy_opt":0},
    )
    for row, override in zip(native["rows"], options):
        kwargs = dict(aod_opt=1,angexp_opt=3,ssa_opt=3,asy_opt=3,
                      aod5502d=.12,angexp2d=1.3,aerssa2d=.85,aerasy2d=.9)
        kwargs.update(override)
        for name in ("aod5502d","angexp2d","aerssa2d","aerasy2d"):
            if kwargs[name] is not None:
                kwargs[name] = cp.full(1,kwargs[name],cp.float32)
        with pytest.raises(ValueError) as error:
            engine.aerosol_optics(*inputs,cp.zeros(1,cp.float32),cp.ones(1,cp.int32),**kwargs)
        row["port_rejection"] = str(error.value)
    (tmp_path/"aerosol-fatal.json").write_text(json.dumps(native,indent=2)+"\n")


def test_large_ice_clamp_and_native_fatal_bounds(tmp_path):
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
    directory = Path(root)
    native = native_witness(directory)
    dump = read_swd(directory/"tables.swd")
    engine = MPASShortwave(raw_coefficients={k:cp.asarray(v) for k,v in dump.items() if k.startswith("kg") and v.ndim})
    assert replay_cloud(engine, directory/"cloud-large-ice.bin", tmp_path/"large-ice.json") == 0
    assert replay_cloud(engine, directory/"cloud-zero-path.bin", tmp_path/"zero-path.json") == 0
    for mode in ("other-ice", "snow-other-ice", "no-ice", "no-liquid", "negative-inflag"):
        assert replay_cloud(engine, directory/f"cloud-{mode}.bin", tmp_path/f"{mode}.json",
                            liqflag=0 if mode == "no-liquid" else 1) == 0
    for row in native["rows"]:
        case = row["case"]
        nc,nl,ng = 1,4,112
        cf,ci,cl = [cp.ones(nl*ng,dtype=cp.float32) for _ in range(3)]
        cs = cp.full(nl*ng, 1. if case in (5,6) else 0., dtype=cp.float32)
        rei = cp.full(nl, {1:4.999,2:131.001,3:4.999,4:140.001}.get(case,40.),dtype=cp.float32)
        rel = cp.full(nl, {7:1.499,8:60.001}.get(case,12.),dtype=cp.float32)
        res = cp.full(nl, {5:4.999,6:140.001}.get(case,60.),dtype=cp.float32)
        tau,ssa,asy,fsf = [cp.full(nl*ng,v,dtype=cp.float32) for v in (.2,.91,.71,.12)]
        taor = cp.empty(nl*ng,dtype=cp.float32)
        error = cp.zeros(1,dtype=cp.int32)
        flags = (nc,nl,1 if case==0 else 2,2 if case in (1,2) else 5 if case in (5,6) else 3,1)
        engine._launch("rsw_cldprmc_b", nl*ng, tuple(np.int32(v) for v in flags)+
                       (engine.tab_gpu,engine.ngb_gpu,cf,ci,cl,cs,rei,rel,res,taor,tau,ssa,asy,fsf,error))
        row["cuda_error"] = int(error.get()[0])
    (tmp_path/"fatal-boundaries.json").write_text(json.dumps(native,indent=2)+"\n")
    assert all(row["cuda_error"] != 0 for row in native["rows"])


def test_native_mixed_column_ozone_fallback(tmp_path):
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE, COMPILE_OPTIONS
    native = read_swd(Path(root)/"ozone-batch.swd")
    # Exercise the production helper verbatim, including its native arithmetic.
    start = CUDA_SOURCE.index("__device__ float mpas_ozone_interp(")
    end = CUDA_SOURCE.index("// MPAS module_ra_rrtmg_sw.F:10197", start)
    code = ("#define FMUL __fmul_rn\n#define FADD __fadd_rn\n"
            "#define FSUB __fsub_rn\n#define FDIV __fdiv_rn\n" + CUDA_SOURCE[start:end] + r'''
extern "C" __global__ void test_ozone(int nc,int nl,int nozn,
 const float* pin,const float* oz,const float* p,float* out) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;int upper=0;
 for(int k=nl-1;k>=0;k--)out[c*nl+k]=mpas_ozone_interp(p[c*nl+k],nozn,pin,oz+c*nozn,upper);
}''')
    from gpuwm import nvrtc_ptx_cache
    ptx, _ = nvrtc_ptx_cache.compile_using_nvrtc(code, COMPILE_OPTIONS, None,
                                               "test_mpas_sw_ozone.cu")
    module = cp.cuda.function.Module()
    module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    kernel = module.get_function("test_ozone")
    pin, oz, p = [cp.ascontiguousarray(cp.asarray(native[n])) for n in ("pin","ozone","pmid")]
    nc,nl = p.shape
    output = cp.empty((nc,nl), dtype=cp.float32)
    kernel((1,),(32,),(np.int32(nc),np.int32(nl),np.int32(pin.size),pin,oz,p,output))
    differing = int(np.count_nonzero(output.get().view(np.uint32) != native["output"].view(np.uint32)))
    (tmp_path/"ozone-batch.json").write_text(json.dumps(dict(
        authority="Pristine MPAS vinterp_ozn; production device helper verbatim",
        words=int(output.size), different_words=differing,
        fixture_sha256=hashlib.sha256((Path(root)/"ozone-batch.swd").read_bytes()).hexdigest(),
        cuda_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
        whole_model_proven=False),indent=2)+"\n")
    assert differing == 0


def test_native_subnormal_shortwave_optics(tmp_path, monkeypatch):
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    import cupy as cp
    directory = Path(root)
    fixture = directory / "reftra-subnormal.bin"
    if not fixture.is_file():
        pytest.fail("Build and run the strict native subnormal optics fixture")
    from hexcore.mpasphys import rrtmg_sw_v842 as port
    MPASShortwave = port.MPASShortwave
    dump = read_swd(directory/"tables.swd")
    engine = MPASShortwave(raw_coefficients={k:cp.asarray(v) for k,v in dump.items() if k.startswith("kg") and v.ndim})
    assert replay_reftra(engine, fixture, tmp_path/"reftra-subnormal.json") == 0
    # A deliberately flushed compile must fail this same native oracle. This
    # proves the fixture detects the compiler hazard, rather than merely using
    # tiny inputs whose effect rounds away in all compared outputs.
    monkeypatch.setattr(port, "COMPILE_OPTIONS", tuple(
        "--ftz=true" if option == "--ftz=false" else option
        for option in port.COMPILE_OPTIONS))
    flushed = MPASShortwave(raw_coefficients={k:cp.asarray(v) for k,v in dump.items() if k.startswith("kg") and v.ndim})
    assert replay_reftra(flushed, fixture, tmp_path/"reftra-ftz-negative-control.json") > 0


def test_native_indeterminate_paths_have_no_port_only_refusal(tmp_path):
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
    dump = read_swd(Path(root)/"tables.swd")
    engine = MPASShortwave(raw_coefficients={k:cp.asarray(v) for k,v in dump.items() if k.startswith("kg") and v.ndim})
    inputs = [cp.full((1,4),v,cp.float32) for v in (90000.,280.,.005,300.)]
    # MPAS aerosol :278-286 continues with an uninitialized spectral slab.
    # This checks continued execution only; every numerical output remains
    # subject to native comparison, and no word exclusion is authorized.
    for exponent in (1,2):
        engine.aerosol_optics(*inputs,cp.zeros(1,cp.float32),cp.ones(1,cp.int32),
            aod_opt=1,angexp_opt=exponent,aod5502d=cp.full(1,.12,cp.float32),
            aod5503d=cp.full((1,4),.03,cp.float32),angexp2d=cp.full(1,1.3,cp.float32))
    # MPAS cloud :2217-2321 and :2392-2421 likewise have no fatal clause
    # when the selector leaves nonzero-path coefficients uninitialized.
    slab = [cp.full(4*112,v,cp.float32) for v in (1.,1.,1.,0.,.2,.91,.71,.12)]
    cf,ci,cl,cs,tau,ssa,asy,fsf = slab
    radii = [cp.full(4,v,cp.float32) for v in (40.,12.,60.)]
    for iceflag,liqflag in ((0,1),(3,0)):
        error = cp.zeros(1,cp.int32)
        engine._launch("rsw_cldprmc_b",4*112,
            tuple(np.int32(v) for v in (1,4,2,iceflag,liqflag))+
            (engine.tab_gpu,engine.ngb_gpu,cf,ci,cl,cs,*radii,cp.empty_like(tau),tau,ssa,asy,fsf,error))
        assert int(error.get()[0]) == 0
    (tmp_path/"native-indeterminate-continuation.json").write_text(json.dumps(dict(
        native_process_count=1,authority="Native source control flow and repeated aerosol witnesses",
        continued=True,bitwise_proven=False,exclusions=[],whole_model_proven=False),indent=2)+"\n")
    # The MPAS driver only routes lower aer_opt=0/3. Diagnose the full module's
    # other unchecked selectors against every native word without converting
    # a continuation check or any numerical difference into a parity claim.
    fixture = Path(root)/"composition-unsupported-aerosol.bin"
    assert fixture.is_file(), "Run the strict native unsupported-selector fixture"
    replay_composition(engine, fixture, tmp_path/"unsupported-aerosol-diagnostic.json")


def test_native_kiss_pressure_order_stop(tmp_path):
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
    witness = json.loads((Path(root)/"kiss-pressure-fatal.json").read_text())
    assert "KISSVEC SEED GENERATOR REQUIRES PMID" in witness["output"]
    assert "UNEXPECTED RETURN" not in witness["output"]
    dump = read_swd(Path(root)/"tables.swd")
    engine = MPASShortwave(raw_coefficients={k:cp.asarray(v) for k,v in dump.items() if k.startswith("kg") and v.ndim})
    pressure = cp.asarray([[100.,200.,300.,400.]], dtype=cp.float32)
    layers = [cp.ones_like(pressure) for _ in range(8)]
    optics = [cp.ones((1,4,14), dtype=cp.float32) for _ in range(4)]
    with pytest.raises(ValueError, match="1742-1744") as error:
        engine.mcica_subcol_sw(pressure, *layers[:7], *optics,
                             layers[7], cp.zeros(1,cp.float32), icld=1, irng=0)
    witness["cuda_rejection"] = str(error.value)
    witness["whole_model_proven"] = False
    (tmp_path/"kiss-pressure-fatal.json").write_text(json.dumps(witness,indent=2)+"\n")


def test_native_short_kiss_and_default_overlap_continue(tmp_path):
    """Standalone native continues; no garbage-value identity or exclusion claim."""
    root = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not root:
        pytest.skip("Strict native SW oracle is required")
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
    directory = Path(root)
    dump = read_swd(directory/"tables.swd")
    engine = MPASShortwave(raw_coefficients={k:cp.asarray(v) for k,v in dump.items() if k.startswith("kg") and v.ndim})
    rows = []
    for nc, nl, overlap, rng in [(c,n,o,r) for c in (1,2) for n in (2,3,4)
                                for o in (1,2,3,4,5) for r in (0,1)] + [
                                    (1,1,0,0),(1,1,2,1),(1,4,-1,0),(1,4,6,0)]:
        pressure = cp.broadcast_to(cp.arange(nl,0,-1,dtype=cp.float32)[None,:]*200, (nc,nl)).copy()
        layers = [cp.full((nc,nl),.4,dtype=cp.float32) for _ in range(7)]
        optics = [cp.full((nc,nl,14),.5,dtype=cp.float32) for _ in range(4)]
        out = engine.mcica_subcol_sw(pressure,*layers[:7],*optics,
                                    cp.zeros((nc,nl),cp.float32),cp.zeros(nc,cp.float32),
                                    icld=overlap,irng=rng)
        cp.cuda.get_current_stream().synchronize()
        assert out["cldfmcl"].shape == (nc,nl,112)
        rows.append(dict(columns=nc,layers=nl,overlap=overlap,rng=rng,continued=True,
                         numeric_identity_measured=False))
    (tmp_path/"mcica-boundary-continuation.json").write_text(json.dumps(dict(
        authority="Pristine MPAS standalone native continued; missing pressure seeds/default CDF remain indeterminate",
        rows=rows,exclusions=[],whole_model_proven=False),indent=2)+"\n")
