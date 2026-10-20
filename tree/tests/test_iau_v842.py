"""IAU control contracts and CUDA vs pristine whole-module native replay."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from hexcore.iau_v842 import (
    IAUConfig, atm_iau_coef, atm_add_tend_anal_incr,
    pre_add_tend_anal_incr, post_add_tend_anal_incr,
    iau_stream_request, bind_iau_increment_arrays,
    iau_kernels,
)


def test_namelist_registry_defaults_and_case_insensitive_names():
    assert IAUConfig.from_namelist("&nhyd_model config_dt=60. /") == IAUConfig()
    config = IAUConfig.from_namelist("&IAU config_IAU_option='on', CONFIG_IAU_WINDOW_LENGTH_S=21600. /")
    assert config == IAUConfig("on",21600.0)
    assert atm_iau_coef(config,360,60).view(np.uint32) == np.float32(1/21600).view(np.uint32)
    assert atm_iau_coef(config,361,60) == 0


@pytest.mark.parametrize("window,step,last", [(2.5,1,3),(3.5,1,4),(2.49,1,2),(0.5,1,1),(0.49,1,0)])
def test_fortran_nint_inclusive_window(window,step,last):
    config=IAUConfig("on",window)
    assert atm_iau_coef(config,last,step)>0
    assert atm_iau_coef(config,last+1,step)==0


@pytest.mark.parametrize("config,itime,dt", [
    (IAUConfig(),1,60), (IAUConfig("ON"),1,60), (IAUConfig("other"),1,60),
    (IAUConfig("on",2),3,1), (IAUConfig("on",2e12),1,2000),
    (IAUConfig("on",0),1,60),
    (IAUConfig("on",-1),1,60),
])
def test_inactive_forcing_does_not_access_pools_or_cuda(config,itime,dt):
    atm_add_tend_anal_incr(config,{},itime,dt,None,None,None,cache=None)
    pre_add_tend_anal_incr(config,{})
    post_add_tend_anal_incr(config,{},None,None,None)


@pytest.mark.parametrize("window,dt,error", [(-2**32,1,ValueError),
    (float("nan"),60,ValueError),(1,0,ValueError),(1,float("inf"),ValueError),(2**32,1,ValueError)])
def test_invalid_intervals_cannot_define_native_finite_forcing(window,dt,error):
    with pytest.raises(error): atm_iau_coef(IAUConfig("on",window),1,dt)


def test_option_trims_fortran_padding_and_integer_step_contract():
    assert atm_iau_coef(IAUConfig("on   "),1,60)>0
    assert atm_iau_coef(IAUConfig(" on"),1,60)==0
    assert atm_iau_coef(IAUConfig("on\t"),1,60)==0
    with pytest.raises(TypeError): atm_iau_coef(IAUConfig("on"),1.5,60)
    with pytest.raises(TypeError): IAUConfig.from_namelist("&iau config_IAU_option=.true. /")
    with pytest.raises(TypeError): IAUConfig.from_namelist("&iau config_IAU_window_length_s='bad' /")


def test_disabled_iau_does_not_access_streams():
    assert iau_stream_request(IAUConfig(),{},start_time=datetime(2000,1,1),
                             initial_time=datetime(2000,1,1)) is None


def test_iau_v842_citations_have_pinned_source_windows():
    manifest=json.loads((Path(__file__).parent/"native_iau_v842/citations.json").read_text())
    source=(Path(__file__).parents[1]/"src/hexcore/iau_v842.py").read_text()
    citations=set()
    for match in re.finditer(r'([A-Za-z_]+\.(?:F|xml)):(\d+)-(\d+)(?:,(\d+)-(\d+))?',source):
        name,a,b,c,d=match.groups()
        citations.add(f"{name}:{a}-{b}")
        if c is not None: citations.add(f"{name}:{c}-{d}")
    assert citations==set(manifest["pins"])
    root=os.environ.get("MPAS_REF")
    if root:
        for pin in manifest["pins"].values():
            lines=(Path(root)/pin["path"]).read_text().splitlines(keepends=True)
            window=''.join(lines[pin["first_line"]-1:pin["last_line"]]).encode()
            assert hashlib.sha256(window).hexdigest()==pin["sha256"]


def test_stream_request_uses_actual_o1_membership_and_filename(tmp_path):
    # The optional source path allows prelanding contract checks without
    # copying another lane's implementation into the delivered code.
    source=os.environ.get("HEXCORE_IAU_STREAM_CONTRACT")
    if source:
        import importlib.util
        import sys
        spec=importlib.util.spec_from_file_location("hexcore._iau_stream_contract",source)
        streams_module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=streams_module
        spec.loader.exec_module(streams_module)
    else:
        streams_module=pytest.importorskip("hexcore.streams_v842",reason="o1-streams-restart not landed")
    path=tmp_path/"streams.atmosphere"
    path.write_text('''<streams><immutable_stream name="iau" type="input"
        filename_template="increments.$Y-$M-$D_$h.$m.$s.nc"
        input_interval="initial_only" filename_interval="none"/></streams>''')
    streams=streams_module.parse_streams(path)
    stamp=datetime(2000,1,1,6)
    config=IAUConfig.from_namelist("&IAU config_IAU_option='on' /")
    request=iau_stream_request(config,streams,start_time=stamp,initial_time=stamp)
    assert request.filename=="increments.2000-01-01_06.00.00.nc"
    assert request.fields==("xtime","u_amb","rho_amb","theta_amb","qv_amb")
    assert request.whence=="nearest" and request.reset_alarms
    moist_request=iau_stream_request(config,streams,start_time=stamp,initial_time=stamp,
                                    active_packages=frozenset({"mp_wsm6_in"}))
    assert set(moist_request.fields)=={"xtime","u_amb","rho_amb","theta_amb",
                                        "qv_amb","qc_amb","qr_amb","qi_amb","qs_amb","qg_amb"}
    # Native activates the stream for every value other than off, although
    # the only nonzero forcing coefficient is the lower-case on value.
    assert iau_stream_request(IAUConfig("ON"),streams,start_time=stamp,initial_time=stamp)==request
    streams["iau"].options["type"]="output"
    with pytest.raises(ValueError,match="input stream"):
        iau_stream_request(config,streams,start_time=stamp,initial_time=stamp)


def _digest(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_native(native_path):
    library=ctypes.CDLL(native_path)
    function=library.run_iau_v2
    pointer=np.ctypeslib.ndpointer(dtype=np.float32,flags="F_CONTIGUOUS")
    function.argtypes=[ctypes.c_int]*11+[ctypes.c_float]*2+[ctypes.c_int]+[pointer]*13+[ctypes.POINTER(ctypes.c_float)]
    function.restype=None
    return function


@pytest.fixture(scope="module")
def replay(request):
    native_path=os.environ.get("HEXCORE_IAU_NATIVE")
    if not native_path:
        pytest.skip("HEXCORE_IAU_NATIVE must name the compiled pristine MPAS IAU library")
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    function=_load_native(native_path)
    cache=KernelCache()
    records=[]
    yield cp,function,cache,records
    destination=os.environ.get("HEXCORE_IAU_RECEIPT")
    if destination:
        destination=Path(destination)
        # Imported fixtures are distinct pytest definitions. Preserve both
        # modules' receipts when synthetic and real captures run together.
        module=request.node.path.stem
        if module != "test_iau_v842":
            destination=destination.with_name(destination.stem+"."+module+destination.suffix)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps({
            "tier":"standalone whole-module CUDA/native replay; not forecast proof",
            "native_execution_host":os.uname().nodename,
            "process_count":1,
            "native_library_sha256":hashlib.sha256(Path(native_path).read_bytes()).hexdigest(),
            "port_source_sha256":hashlib.sha256((Path(__file__).parents[1]/"src/hexcore/iau_v842.py").read_bytes()).hexdigest(),
            "replay_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source":"MPAS-A v8.4.2 f34984b mpas_atm_iau.F:31-292",
            "records":records, "compile_manifest":cache.compile_manifest(),
        },indent=2)+"\n")


def _inputs(seed,nl=11,nc=7,ne=13,ns=8):
    rng=np.random.default_rng(seed)
    def field(shape,lo,hi): return rng.uniform(lo,hi,shape).astype(np.float32)
    return {
        "zz":field((nl,nc),0.1,1.1), "rho_zz":field((nl,nc),0.05,2),
        "theta_m":field((nl,nc),200,400), "scalars":field((ns,nl,nc),0,0.03),
        "rho_edge":field((nl,ne),0.05,2), "u_amb":field((nl,ne),-5,5),
        "rho_amb":field((nl,nc),-0.02,0.02), "theta_amb":field((nl,nc),-2,2),
        "scalars_amb":field((ns,nl,nc),-0.001,0.001),
        "tend_ru":field((nl,ne),-0.001,0.001),
        "tend_rtheta":field((nl,nc),-0.03,0.03),
        "tend_rho":field((nl,nc),-1e-5,1e-5),
        "tend_scalars":field((ns,nl,nc),-1e-6,1e-6),
    }


def _pools(cp,host,ncs,nes,ms,me,qv):
    device={name:cp.asarray(array) for name,array in host.items()}
    nl,nc=host["zz"].shape; ne=host["rho_edge"].shape[1]
    return device,{
        "mesh":{"nVertLevels":nl,"nCells":nc,"nEdges":ne,"nCellsSolve":ncs,"nEdgesSolve":nes,"zz":device["zz"]},
        "state":{"theta_m":device["theta_m"],"rho_zz":device["rho_zz"],"scalars":device["scalars"],
                 "moist_start":ms,"moist_end":me,"index_qv":qv},
        "diag":{"rho_edge":device["rho_edge"]},
        "tend":{"scalars_tend":device["tend_scalars"]},
        "tend_iau":bind_iau_increment_arrays(device),
    }


# seed, option, window, dt, step, cellsSolve, edgesSolve, moistStart/end, qv
DECKS=[
    (1,"on",21600,60,1,5,9,2,7,4),
    (2,"on",21600,60,360,5,9,1,6,1),
    (3,"on",21600,60,361,5,9,1,6,1),
    (4,"off",21600,60,1,5,9,1,6,1),
    (5,"ON",21600,60,1,5,9,1,6,1),
    (6,"other",21600,60,1,5,9,1,6,1),
    (7,"on",2.5,1,3,7,13,2,7,4),
    (8,"on",0.49,1,1,7,13,2,7,4),
    (9,"on",0.5,1,1,0,0,2,7,4),
    (10,"on",2e12,2000,1,5,9,1,6,1),
    (11,"on",60,60,1,5,0,5,4,2), # empty moisture range, qv tendency still enters theta
    (12,"on",60,60,0,0,9,1,8,1),
    (13,"on",21600,60,1,5,9,1,6,1), # subnormal operands/results
    (14,"on",60,60,1,5,9,1,6,1), # zero increments, nonzero incoming qv tendency
    (15,"on",60,60,1,5,9,1,6,1), # dry qv, density and theta coupling
    (16,"on",1e12,1000,1,5,9,1,6,1), # coefficient exactly at the native skip threshold
    (17,"on",0,60,1,5,9,1,6,1), # Registry permits zero; first real timestep is inactive
    (18,"on",21600,600,1,5,9,1,6,1), # state has additional non-moist scalars absent from increments
    (19,"on",-1,60,1,5,9,1,6,1), # finite negative Registry window: normal first step inactive
    (20,"on",-0.5,1,0,5,9,1,6,1), # negative half tie rounds away from zero
    (21,"on",-2.5,1,-3,5,9,1,6,1), # native generic coefficient may be negative, hence no forcing
]


@pytest.mark.parametrize("deck",DECKS,ids=[f"native-deck-{n[0]}" for n in DECKS])
def test_window_weights_against_pristine_native_module_without_cuda(deck):
    native_path=os.environ.get("HEXCORE_IAU_NATIVE")
    if not native_path:
        pytest.skip("HEXCORE_IAU_NATIVE must name the compiled pristine MPAS IAU library")
    native=_load_native(native_path)
    seed,option,window,dt,itime,ncs,nes,ms,me,qv=deck
    host=_inputs(seed)
    if seed==18: host["scalars_amb"]=host["scalars_amb"][:6].copy()
    fields={name:np.asfortranarray(value) for name,value in host.items()}
    nl,nc=host["zz"].shape; ne=host["rho_edge"].shape[1]; ns=host["scalars"].shape[0]
    weight=ctypes.c_float()
    native(nl,nc,ne,ncs,nes,ns,host["scalars_amb"].shape[0],ms,me,qv,itime,dt,window,
           {"off":0,"on":1,"ON":2,"other":3}[option],*fields.values(),ctypes.byref(weight))
    assert np.float32(weight.value).view(np.uint32)==atm_iau_coef(IAUConfig(option,window),itime,dt).view(np.uint32)
    if weight.value<=np.float32(1e-12):
        for name in fields:
            np.testing.assert_array_equal(fields[name].view(np.uint32),host[name].view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("deck",DECKS,ids=[f"deck-{n[0]}" for n in DECKS])
def test_cuda_all_output_words_match_pristine_native_module(replay,deck):
    cp,native,cache,records=replay
    seed,option,window,dt,itime,ncs,nes,ms,me,qv=deck
    host=_inputs(seed)
    if seed==18: host["scalars_amb"]=host["scalars_amb"][:6].copy()
    if seed==13:
        for name in ("u_amb","rho_amb","theta_amb","scalars_amb","tend_scalars","tend_rho","tend_ru","tend_rtheta"):
            host[name]=np.full_like(host[name],np.float32(1e-38))
        host["u_amb"][0,0]=np.float32(-1e-40)
    if seed==14:
        for name in ("u_amb","rho_amb","theta_amb","scalars_amb"): host[name].fill(0)
    if seed==15: host["scalars"][qv-1].fill(0)
    native_fields={name:np.asfortranarray(value) for name,value in host.items()}
    nl,nc=host["zz"].shape; ne=host["rho_edge"].shape[1]; ns=host["scalars"].shape[0]
    weight=ctypes.c_float()
    native(nl,nc,ne,ncs,nes,ns,host["scalars_amb"].shape[0],ms,me,qv,itime,dt,window,
           {"off":0,"on":1,"ON":2,"other":3}[option],*native_fields.values(),ctypes.byref(weight))
    device,pools=_pools(cp,host,ncs,nes,ms,me,qv)
    config=IAUConfig(option,window)
    atm_add_tend_anal_incr(config,pools,itime,dt,device["tend_ru"],device["tend_rtheta"],device["tend_rho"],cache=cache)
    result={name:cp.asnumpy(value) for name,value in device.items()}
    fields={}
    for name,array in result.items():
        expected=native_fields[name]
        words=int(np.count_nonzero(array.view(np.uint32)!=expected.view(np.uint32)))
        mutations=int(np.count_nonzero(array.view(np.uint32)!=host[name].view(np.uint32)))
        fields[name]={"words":array.size,"different_words":words,"mutated_words":mutations,
                      "input_sha256":_digest(host[name]),"native_sha256":_digest(expected),"cuda_sha256":_digest(array)}
    records.append({"deck":list(deck),"fields":fields,"native_weight":weight.value,
                    "cuda_weight":float(atm_iau_coef(config,itime,dt))})
    assert np.float32(weight.value).view(np.uint32)==atm_iau_coef(config,itime,dt).view(np.uint32)
    for name,array in result.items():
        np.testing.assert_array_equal(array.view(np.uint32),native_fields[name].view(np.uint32),err_msg=name)
    active=option=="on" and atm_iau_coef(config,itime,dt)>np.float32(1e-12)
    if active and ncs:
        assert fields["tend_rtheta"]["mutated_words"]>0
    for name in ("tend_rho","tend_rtheta"):
        np.testing.assert_array_equal(result[name][:,ncs:].view(np.uint32),host[name][:,ncs:].view(np.uint32))
    np.testing.assert_array_equal(result["tend_ru"][:,nes:].view(np.uint32),host["tend_ru"][:,nes:].view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("defect",["dtype","missing_moist_storage"])
def test_layout_refusal_happens_before_any_tendency_mutation(replay,defect):
    cp,_,cache,_=replay
    host=_inputs(100)
    device,pools=_pools(cp,host,5,9,1,6,1)
    pools["tend_iau"]["scalars"]=(device["scalars_amb"].astype(cp.float64)
        if defect=="dtype" else device["scalars_amb"][:5].copy())
    with pytest.raises(TypeError if defect=="dtype" else ValueError):
        atm_add_tend_anal_incr(IAUConfig("on"),pools,1,60,device["tend_ru"],device["tend_rtheta"],device["tend_rho"],cache=cache)
    np.testing.assert_array_equal(cp.asnumpy(device["tend_ru"]).view(np.uint32),host["tend_ru"].view(np.uint32))


@pytest.mark.gpu
def test_direct_nvrtc_preserves_subnormal_inputs_without_fallback(replay):
    cp,_,cache,records=replay
    source=cp.asarray(np.array([1,0x007fffff,0x80000001],np.uint32).view(np.float32))
    output=cp.empty_like(source)
    iau_kernels(cache)["iau_subnormal_probe"]((1,),(3,),(source,output))
    words=cp.asnumpy(output).view(np.uint32)
    np.testing.assert_array_equal(words,np.array([2,0x00fffffe,0x80000002],np.uint32))
    records.append({"compiler_subnormal_probe":{"input_bits":[1,0x007fffff,0x80000001],
                    "output_bits":words.tolist(),"different_words":0}})
