"""Strict compiled whole non-LES module replay; never a forecast proof receipt."""
from pathlib import Path
import hashlib
import json
import os
from dataclasses import replace
import numpy as np
import pytest

from hexcore.mixing_v842 import V842MixingConfig
from hexcore.errors import ConfigurationRefusal


def test_fixed_and_perturbation_configuration_domains():
    V842MixingConfig(config_horiz_mixing="2d_fixed",config_mix_full=False,
        config_mix_scalars=True,config_v_mom_eddy_visc2=3.7).validate()
    for name in ("config_h_theta_eddy_visc2","config_h_mom_eddy_visc4",
                 "config_h_theta_eddy_visc4","config_v_mom_eddy_visc2",
                 "config_v_theta_eddy_visc2"):
        for value in (-1.0,np.inf,np.nan):
            with pytest.raises(ConfigurationRefusal,match=name):
                replace(V842MixingConfig(),**{name:value}).validate()
    for name in ("config_mix_full","config_mix_scalars"):
        with pytest.raises(ConfigurationRefusal,match=name):
            replace(V842MixingConfig(),**{name:1}).validate()


def test_nonles_schema_and_native_ignored_orders():
    from dataclasses import fields
    from hexcore.config_v841 import V841DryDycoreConfig
    from hexcore.config_v842 import V842NonlesDycoreConfig
    old = V841DryDycoreConfig()
    new = V842NonlesDycoreConfig()
    new.validate()
    assert all(getattr(old, f.name) == getattr(new, f.name) for f in fields(old))
    fixed = replace(new, config_horiz_mixing="2d_fixed", config_mix_full=False,
        config_mix_scalars=True, config_h_theta_eddy_visc2=37.5,
        config_h_mom_eddy_visc4=9.e8, config_v_mom_eddy_visc2=3.7,
        config_time_integration_order=3)
    fixed.validate()
    knobs = ("config_w_adv_order", "config_theta_adv_order", "config_u_vadv_order",
             "config_w_vadv_order", "config_theta_vadv_order")
    for knob in knobs:
        for order in (2, 3, 4):
            candidate = replace(fixed, **{knob: order})
            candidate.validate()
            assert candidate.mixing_config() == fixed.mixing_config()
        for invalid in (1, 5, True, 3.0):
            with pytest.raises(ConfigurationRefusal, match=knob):
                replace(fixed, **{knob: invalid}).validate()
    assert not replace(new, config_horiz_mixing="2d_smagorinsky").requires_nonles_mixing()
    assert not replace(new, config_mix_full=False).requires_nonles_mixing()
    assert replace(new, config_horiz_mixing="2d_smagorinsky",
                   config_mpas_cam_coef=2.0).requires_nonles_mixing()
    assert fixed.requires_nonles_mixing()


def test_opt_in_namelist_selects_native_options_and_keeps_legacy_door():
    from hexcore.config import dry_dycore_config_from_namelist
    from hexcore.config_v842 import nonles_dycore_config_from_namelist
    text = """&nhyd_model
        config_horiz_mixing='2d_fixed', config_h_theta_eddy_visc2=37.5,
        config_h_mom_eddy_visc4=9.e8, config_v_mom_eddy_visc2=3.7,
        config_mix_full=.false., config_mix_scalars=.true.,
        config_w_adv_order=2, config_theta_adv_order=4,
        config_u_vadv_order=2, config_w_vadv_order=4, config_theta_vadv_order=2,
        config_time_integration_order=3
        /
        &damping config_number_cam_damping_levels=8,
        config_number_rayleigh_damp_u_levels=6 /
    """
    config = nonles_dycore_config_from_namelist(text)
    assert config.requires_nonles_mixing()
    assert config.mixing_config().config_h_theta_eddy_visc2 == 37.5
    assert config.config_mix_full is False and config.config_mix_scalars is True
    assert config.config_theta_adv_order == 4
    assert config.config_time_integration_order == 3
    assert config.config_number_cam_damping_levels == 8
    native_length = nonles_dycore_config_from_namelist(
        "&nhyd_model config_horiz_mixing='2d_fixed', config_len_disp=-1.0 /")
    assert native_length.config_len_disp == -1.0
    with pytest.raises(ConfigurationRefusal):
        dry_dycore_config_from_namelist(text, source_release="v8.4.1")
    for text, knob in (("&physics config_mix_full=.false. /", "config_mix_full"),
        ("&nhyd_model config_w_adv_order=2.0 /", "config_w_adv_order"),
        ("&nhyd_model config_mix_full=1 /", "config_mix_full"),
        ("&nhyd_model config_calendar_type='gregorian' /", "config_calendar_type")):
        with pytest.raises(ConfigurationRefusal, match=knob):
            nonles_dycore_config_from_namelist(text)


class Stream:
    def __init__(self,path):
        self.data=path.read_bytes();self.offset=0
    def array(self,shape,dtype="<f4"):
        count=int(np.prod(shape));dt=np.dtype(dtype)
        a=np.frombuffer(self.data,dtype=dt,count=count,offset=self.offset).copy()
        self.offset+=count*dt.itemsize
        return a.reshape(shape,order="F")
    def done(self):
        assert self.offset==len(self.data)


def read_inputs(folder):
    nl=int((folder/"levels.txt").read_text()) if (folder/"levels.txt").exists() else 8
    f=Stream(folder/"inputs.bin");d={}
    for name,shape,dtype in [
        ("coe",(2,7),"<i4"),("voe",(2,7),"<i4"),("eoc",(3,5),"<i4"),
        ("eov",(3,5),"<i4"),("count",(5,),"<i4"),
        ("ecs",(3,5),"<f4"),("evs",(3,5),"<f4")]:
        d[name]=f.array(shape,dtype)
    for name,size in [("area",5),("at",5),("dc",7),("dv",7),("idc",7),("idv",7),
                       ("d2",7),("d4",7),("angle",7)]:
        d[name]=f.array((size,))
    for name,shape in [("z",(nl+1,5)),("eddy",(nl,5)),("rho",(nl,5)),("re",(nl,7)),
        ("u",(nl,7)),("v",(nl,7)),("w",(nl+1,5)),("theta",(nl,5)),("initial",(nl,5)),
        ("div",(nl,5)),("vort",(nl,5)),("ui",(nl,)),("vi",(nl,)),("zw",(nl,)),("zu",(nl,)),
        ("q",(6,nl,5)),("tq0",(6,nl,5)),("defc2",(3,5)),("defs2",(3,5)),("defcs",(3,5))]:
        d[name]=f.array(shape)
    d["density"]=f.array((5,))
    # Earlier fixtures predate the optional native Euler seed record.
    seed = folder / "u-seed.bin"
    d["u_seed"] = float(np.frombuffer(seed.read_bytes(), dtype="<f4")[0]) if seed.exists() else 1.e-6
    f.done();return d


def expected_output(folder,branch):
    nl=int((folder/"levels.txt").read_text()) if (folder/"levels.txt").exists() else 8
    f=Stream(folder/f"output-{branch}.bin");result={}
    for name,shape,n in [("tend_u_euler",(nl,7),6),("tend_w_euler",(nl+1,5),4),
        ("tend_theta_euler",(nl,5),4),("tend_scalars",(6,nl,5),4),
        ("delsq_u",(nl,7),6),("delsq_divergence",(nl,5),4),
        ("delsq_vorticity",(nl,5),4),("delsq_w",(nl,5),4),("delsq_theta",(nl,5),4),
        ("kdiff",(nl,5),4)]:
        result[name]=np.ascontiguousarray(f.array(shape)[...,:n])
    f.done();return result


@pytest.fixture(scope="module")
def native_d1():
    folder=os.environ.get("HEXCORE_NATIVE_D1_FIXTURE")
    if not folder:
        pytest.skip("set HEXCORE_NATIVE_D1_FIXTURE to tests/dycore_options_native/build.sh output")
    folder=Path(folder);return folder,read_inputs(folder)


@pytest.fixture(scope="module")
def device_d1(native_d1):
    import cupy as cp
    import hexcore.mixing_v842 as port
    import hexcore.cuda_upper_damping_v842 as compiler_port
    from hexcore.cuda_backend import DeviceMesh,TransferStats,KernelCache
    from hexcore.cuda_upper_damping_v842 import direct_cache, CUDA_OPTIONS
    from hexcore.cuda_horizontal_v841 import CudaHorizontalV841
    from hexcore.cuda_v841 import CudaV841Context
    folder,d=native_d1
    nl=d["rho"].shape[0]
    provenance = {
        "mixing_module_sha256": hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest(),
        "cuda_source_sha256": hashlib.sha256(port._CUDA_SOURCE.encode()).hexdigest(),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler_module_sha256": hashlib.sha256(Path(compiler_port.__file__).read_bytes()).hexdigest(),
    }
    companion = folder.parent / "native-zero-seed-negative"
    if (companion / "inputs.bin").exists():
        provenance["negative_fixture"] = {
            "input_sha256": hashlib.sha256((companion / "inputs.bin").read_bytes()).hexdigest(),
            "native_binary_sha256": hashlib.sha256((companion / "deck").read_bytes()).hexdigest(),
        }
    def pack(a):return cp.asarray(np.ascontiguousarray(a))
    coe=d["coe"][:,:6].T-1;voe=d["voe"][:,:6].T-1
    eoc=d["eoc"][:,:4].T-1;eov=d["eov"][:,:4].T-1
    index={"cells_on_edge":coe,"vertices_on_edge":voe,"edges_on_cell":eoc,
        "edges_on_vertex":eov,"n_edges_on_cell":d["count"][:4],
        "cells_on_cell":np.zeros((4,3),np.int32),"edges_on_edge":np.zeros((6,4),np.int32),
        "n_edges_on_edge":np.zeros(6,np.int32),"vertices_on_cell":np.zeros((4,3),np.int32),
        "cells_on_vertex":np.zeros((4,3),np.int32)}
    floats={"edge_sign_on_cell":d["ecs"][:,:4].T,"weights_on_edge":np.zeros((6,4),np.float32),
        "dc_edge":d["dc"][:6],"dv_edge":d["dv"][:6],
        "area_cell":np.float32(1)/d["area"][:4],"area_triangle":np.float32(1)/d["at"][:4],
        "kite_areas_on_vertex":np.ones((4,3),np.float32),
        "lat_cell":np.zeros(4,np.float32),"lon_cell":np.zeros(4,np.float32),
        "lat_edge":np.zeros(6,np.float32),"lon_edge":np.zeros(6,np.float32),
        "angle_edge":d["angle"][:6],"mesh_density":d["density"][:4],
        "defc_a":np.zeros((4,3),np.float32),"defc_b":np.zeros((4,3),np.float32),
        "f_vertex":np.zeros(4,np.float32),"f_edge":np.zeros(6,np.float32),
        "spec_zone_mask_edge":np.zeros(6,np.float32)}
    mesh=DeviceMesh(**{k:pack(v) for k,v in {**index,**floats}.items()},n_cells=4,n_edges=6,
        n_vertices=4,max_edges=3,max_edges2=4,vertex_degree=3,nominal_min_dc=12000.0,
        dtype=np.dtype(np.float32),index_dtype=np.dtype(np.int32),h2d=TransferStats(0,0))
    context=CudaV841Context(inv_area_cell=pack(d["area"][:4]),inv_area_triangle=pack(d["at"][:4]),
        inv_dc_edge=pack(d["idc"][:6]),inv_dv_edge=pack(d["idv"][:6]),
        etp=cp.zeros(nl,cp.float32),etm=cp.zeros(nl,cp.float32),ewp=cp.zeros(nl+1,cp.float32),
        ewm=cp.zeros(nl+1,cp.float32),u_init=pack(d["ui"]),v_init=pack(d["vi"]),h2d=TransferStats(0,0))
    # Use the actual driver's legacy cache: the new boundary itself must route
    # every reused numerical stencil through direct NVRTC without changing it.
    horizontal=CudaHorizontalV841(mesh,nl,context,kernel_cache=KernelCache())
    horizontal._deformation_v841={k:pack(d[n][:,:4].T) for k,n in
        [("coef_c2","defc2"),("coef_s2","defs2"),("coef_cs","defcs")]}
    inputs={k:pack(d[k][:,:n]) for k,n in [("u",6),("v",6),("w",4),("theta",4),
        ("re",6),("div",4),("vort",4),("rho",4),("z",4),("initial",4),("eddy",4)]}
    inputs.update(q=pack(d["q"][:,:,:4]),tq0=pack(d["tq0"][:,:,:4]),
        ui=pack(d["ui"]),vi=pack(d["vi"]),zw=pack(d["zw"]),zu=pack(d["zu"]))
    from hexcore.native_geometry_math import cos,sin
    inputs.update(cos=pack(cos(d["angle"][:6])),sin=pack(sin(d["angle"][:6])))
    real_manifest = folder.parent / "columns" / "real-columns.json"
    if real_manifest.exists():
        provenance["real_columns"] = json.loads(real_manifest.read_text())
    rows=[]
    yield horizontal,inputs,rows
    path=os.environ.get("HEXCORE_D1_RECEIPT")
    if path:
        differences = sum(row["different_words"] for row in rows)
        expected_comparisons = 5126 if (companion / "inputs.bin").exists() else 2566
        complete = len(rows) == expected_comparisons
        Path(path).write_text(json.dumps({"scope":"compiled pristine module replay; no forecast claim",
            "status": "PASS" if complete and differences == 0 else "FAIL" if differences else "INCOMPLETE",
            "expected_comparisons": expected_comparisons, "completed_comparisons": len(rows),
            "words": sum(row["words"] for row in rows), "different_words": differences,
            "excluded_words": [], "cuda_options": list(CUDA_OPTIONS),
            "compiler": "direct NVRTC", "compile_inventory": direct_cache().inventory,
            "native_process_count": 1, "hex_process_count": 1, "mpi": False,
            "port": provenance,
            "native_flags": "-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations",
            "native_binary_sha256": hashlib.sha256((folder/"deck").read_bytes()).hexdigest(),
            "reference_sha256":hashlib.sha256((folder/"mpas_atm_dissipation_models.F").read_bytes()).hexdigest(),
            "input_sha256":hashlib.sha256((folder/"inputs.bin").read_bytes()).hexdigest(),
            "comparisons":rows},indent=2)+"\n")


def compare(rows,label,actual,expected):
    import cupy as cp
    a=cp.asnumpy(actual)
    diff=int(np.count_nonzero(a.view(np.uint32)!=expected.view(np.uint32)))
    rows.append({"branch":label,"words":int(expected.size),"different_words":diff,
                 "cuda_sha256":hashlib.sha256(a.tobytes()).hexdigest(),
                 "native_sha256":hashlib.sha256(expected.tobytes()).hexdigest()})
    np.testing.assert_array_equal(a.view(np.uint32),expected.view(np.uint32),err_msg=label)


@pytest.fixture(scope="module")
def cpu_receipt_rows(native_d1):
    folder, _ = native_d1
    rows = []
    yield rows
    path = os.environ.get("HEXCORE_D1_CPU_RECEIPT")
    if path:
        Path(path).write_text(json.dumps({
            "scope": "Rust CPU versus unchanged extracted GNU-strict native damping blocks; no forecast claim",
            "status": "PASS" if len(rows) == 6 and not any(r["different_words"] for r in rows) else "FAIL",
            "native_process_count": 1, "hex_process_count": 1, "mpi": False,
            "native_binary_sha256": hashlib.sha256((folder / "deck").read_bytes()).hexdigest(),
            "words": sum(r["words"] for r in rows),
            "different_words": sum(r["different_words"] for r in rows),
            "excluded_words": [],
            "comparisons": rows}, indent=2) + "\n")


@pytest.mark.parametrize("branch", (1, 2, 3))
def test_rust_upper_matches_extracted_native_driver_bits(native_d1, cpu_receipt_rows, branch):
    from hexcore.upper_damping_v841 import apply_cam_floor, apply_rayleigh_u
    folder, d = native_d1
    nl=d["rho"].shape[0]
    f = Stream(folder / f"upper-{branch}.bin")
    cam = np.ascontiguousarray(f.array((nl, 5))[:, :4])
    ray = np.ascontiguousarray(f.array((nl, 7))[:, :6]); f.done()
    # The separate native upper-driver decks seed viscosity to 37.5.
    viscosity = np.full((nl, 4), np.float32(37.5))
    tendency = np.ascontiguousarray(d["u"][:, :6])
    coefficient, cam_levels, ray_levels, days, solved = {
        1: (2, 4, 6, 5, 5), 2: (0.00001, 8, 8, 0.03125, 6), 3: (0, 1, 1, 2.7, 0)}[branch]
    apply_cam_floor(viscosity, length=12000, coefficient=coefficient, levels=cam_levels)
    apply_rayleigh_u(tendency, np.ascontiguousarray(d["re"][:, :6]),
        np.ascontiguousarray(d["u"][:, :6]), levels=ray_levels, days=days, solved=solved)
    for name, actual, expected in (("CAM", viscosity, cam), ("Rayleigh", tendency, ray)):
        cpu_receipt_rows.append({"branch": f"{name}:{branch}", "words": int(expected.size),
            "different_words": int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
            "rust_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "native_sha256": hashlib.sha256(expected.tobytes()).hexdigest()})
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("branch",range(1,257))
def test_complete_nonles_module_bits(native_d1,device_d1,branch,label_prefix=""):
    import cupy as cp
    from hexcore.mixing_v842 import compute_nonles_mixing_tendencies_v842
    h,d,rows=device_d1;folder,host=native_d1
    bits=branch-1
    config=V842MixingConfig(config_horiz_mixing="2d_smagorinsky" if bits&32 else "2d_fixed",
        config_h_theta_eddy_visc2=float(host["eddy"][0,0]),config_h_mom_eddy_visc4=0.0 if bits&8 else 9.e8,
        config_h_theta_eddy_visc4=0.0 if bits&8 else 2.e9,
        config_visc4_2dsmag=0.0 if bits&8 else 0.05,
        config_v_mom_eddy_visc2=0.0 if bits&16 else 3.7,
        config_v_theta_eddy_visc2=0.0 if bits&16 else 1.9,
        config_mix_full=bool(bits&1),config_mix_scalars=bool(bits&2),
        config_h_ScaleWithMesh=not bool(bits&64),config_mpas_cam_coef=2.0 if bits&128 else 0.0)
    tq=d["tq0"].copy()
    result=compute_nonles_mixing_tendencies_v842(h,d["u"],d["v"],d["w"],d["theta"],
        d["re"],d["div"],d["vort"],dt=20,config=config,n_cells_solve=3,n_edges_solve=5,
        initial_u_tendency=None if host["u_seed"] == 0 else cp.full((h.nlev,6),host["u_seed"],cp.float32),rho_zz=d["rho"],zgrid=d["z"],
        rdzw=d["zw"],rdzu=d["zu"],u_init=d["ui"],v_init=d["vi"],t_init=d["initial"],
        cos_angle=d["cos"],sin_angle=d["sin"],scalars=d["q"],tend_scalars=tq,
        dynamics_substep=2 if bits&4 else 1)
    assert not h._kernels and not h._v841_kernels
    assert not h.kernel_cache._modules
    expected=expected_output(folder,branch)
    for name in expected:
        compare(rows,f"{label_prefix}{branch}:{name}",tq if name=="tend_scalars" else getattr(result,name),expected[name])


@pytest.mark.gpu
def test_negative_gradient_zero_euler_seed_bits(native_d1, device_d1):
    """Exercise negative zero products in native's Euler accumulation."""
    import cupy as cp
    folder = native_d1[0].parent / "native-zero-seed-negative"
    assert folder.exists(), "build the ./deck 0 0 -1 companion fixture before replay"
    host = read_inputs(folder)
    h, d, rows = device_d1
    # The companion changes only divergence/vorticity and the Euler seed.
    negative = dict(d)
    for name in ("div", "vort"):
        negative[name] = cp.asarray(np.ascontiguousarray(host[name][:, :4]))
    assert host["u_seed"] == 0 and host["eddy"][0, 0] == 0
    # Confirm that negative products reach the zero-viscosity branch and
    # native's initial-zero addition returns positive zero at every edge.
    reference = expected_output(folder, 25)["tend_u_euler"]
    assert np.all(reference.view(np.uint32) == 0)
    assert np.all(expected_output(folder, 25)["delsq_u"] < 0)
    # Witness the defect with the original direct-assignment CUDA stencil.
    # Branch 89 disables mesh scaling as well as fourth/vertical mixing.
    direct = cp.empty_like(d["u"])
    lap = cp.empty_like(d["u"])
    from copy import copy
    from hexcore.cuda_upper_damping_v842 import direct_cache
    witness_launcher = copy(h)
    witness_launcher.kernel_cache = direct_cache()
    witness_launcher._kernels = {}
    witness_launcher._launch("momentum_filter_lap2_f32", h.nedges,
        (d["re"], negative["div"], negative["vort"],
         cp.zeros_like(d["eddy"]), cp.ones(h.nedges, cp.float32),
         h.mesh.cells_on_edge, h.mesh.vertices_on_edge,
         h.mesh.dc_edge, h.mesh.dv_edge, np.int32(h.nlev), np.int32(h.ncells),
         np.int32(h.nedges), np.int32(h.nvertices), lap, direct))
    old = cp.asnumpy(direct)
    native = expected_output(folder, 89)["tend_u_euler"]
    difference = int(np.count_nonzero(old.view(np.uint32) != native.view(np.uint32)))
    assert difference > 0 and np.any(old.view(np.uint32) == 0x80000000)
    receipt = os.environ.get("HEXCORE_D1_RECEIPT")
    if receipt:
        Path(receipt).with_name("signed-zero-witness.json").write_text(json.dumps({
            "scope": "counterfactual original CUDA direct assignment versus strict native zero-seed addition",
            "native_source": "mpas_atm_dissipation_models.F:715-722",
            "branch": 89, "words": int(native.size), "different_words": difference,
            "old_cuda_sha256": hashlib.sha256(old.tobytes()).hexdigest(),
            "native_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
            "negative_input_sha256": hashlib.sha256((folder / "inputs.bin").read_bytes()).hexdigest(),
        }, indent=2) + "\n")
    for branch in range(1, 257):
        test_complete_nonles_module_bits((folder, host), (h, negative, rows),
            branch, label_prefix="negative:")


@pytest.mark.gpu
@pytest.mark.parametrize("branch",(1,2,3))
def test_native_cam_and_rayleigh_bits(native_d1,device_d1,branch):
    from hexcore.cuda_upper_damping_v842 import apply_cam_floor_cuda,apply_rayleigh_u_cuda
    h,d,rows=device_d1;folder,_=native_d1
    nl=h.nlev
    f=Stream(folder/f"upper-{branch}.bin")
    cam=np.ascontiguousarray(f.array((nl,5))[:,:4])
    ray=np.ascontiguousarray(f.array((nl,7))[:,:6]);f.done()
    # Upper-driver inputs are independent of the fixed-mixing coefficient.
    import cupy as cp
    viscosity=cp.full_like(d["eddy"],37.5);tendency=d["u"].copy()
    coefficient,cam_levels,ray_levels,days,solved = {
        1:(2,4,6,5,5), 2:(0.00001,8,8,0.03125,6), 3:(0,1,1,2.7,0)}[branch]
    apply_cam_floor_cuda(viscosity,length=12000,coefficient=coefficient,levels=cam_levels,cache=h.kernel_cache)
    apply_rayleigh_u_cuda(tendency,d["re"],d["u"],levels=ray_levels,days=days,solved=solved,cache=h.kernel_cache)
    compare(rows,f"CAM:{branch}",viscosity,cam)
    compare(rows,f"Rayleigh:{branch}",tendency,ray)
