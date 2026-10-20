#!/usr/bin/env python3
"""Compare every Rust reference to NumPy on the contract's real input bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore.drivers import run_cuda_regional_contract as d
from hexcore.drivers import contract_hostprep as rust
from hexcore.transport import _atmosphere_vertical_flux, _atmosphere_horizontal_edge_values, advance_scalars


def main():
    p = argparse.ArgumentParser()
    for name in ("grid", "init", "lbc-dir", "start-time", "out"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--authority-receipt", help="recheck Rust full-field hashes against a completed NumPy comparison")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    b = d.Bundle(None,None,grid=Path(args.grid),init=Path(args.init),
                 lbc_dir=Path(args.lbc_dir),start_time=args.start_time)
    def digest(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                value.update(block)
        return value.hexdigest()
    input_hashes = {str(path): digest(path) for path in
        [args.grid, args.init, *b.lbc_paths]}
    rows, timings = [], {}
    def timed(name, produce):
        started = time.perf_counter()
        result = produce()
        timings[name] = time.perf_counter()-started
        print(name,timings[name],flush=True)
        return result
    def record(name, actual, expected):
        row = d.compare_bits(name,expected,actual)
        rows.append(row)
        out.write_text(json.dumps({"shape":[b.ncells,b.nlev],"timings":timings,"fields":rows,
            "input_sha256": input_hashes, "rust_binary_sha256": digest(rust.engine()),
            "numpy_version": np.__version__},indent=2))
    if args.authority_receipt:
        authority = json.loads(Path(args.authority_receipt).read_text())
        expected = {row["payload"]: row for row in authority["fields"]}
        if not expected or not all(row["bitwise_equal"] for row in expected.values()):
            raise SystemExit("authority receipt includes a failed reference; it cannot establish bit identity")
        if authority["shape"] != [b.ncells,b.nlev]:
            raise SystemExit("authority input dimensions differ; reference hashes belong to another bundle")
        if authority.get("input_sha256") and authority["input_sha256"] != input_hashes:
            raise SystemExit("authority input bytes differ; reference hashes belong to another bundle")
        def hashed(name,actual):
            row = expected[name]
            actual_hash = d._sha(actual)
            same = actual_hash == row["host_sha256"] and actual.size == row["values"]
            rows.append({"payload":name,"values":int(actual.size),"bitwise_equal":same,
                "host_sha256":row["host_sha256"],"device_sha256":actual_hash})
        prepared = d._acoustic_inputs(b)
        result = timed("rust.acoustic",lambda: rust.acoustic(b,prepared))
        for name in prepared[3].__slots__:
            hashed("acoustic."+name,getattr(result,name))
        flux = timed("rust.vertical",lambda: rust.vertical(b))
        hashed("vertical_flux",flux)
        co = d.build_advection_coefficients(b.mesh,config_scalar_adv_order=3,
            n_vert_levels=b.nlev,source_order_v841=True,allow_regional_sentinels=True)
        edges = timed("rust.horizontal",lambda: rust.horizontal(b,co))
        hashed("horizontal_edge_values",edges)
        result = timed("rust.finish",lambda: rust.finish(b,edges,flux[None,...]))
        hashed("scalars_after_transport",result.scalars)
        hashed("target_density",result.density)
        out.write_text(json.dumps({"shape":[b.ncells,b.nlev],"timings":timings,"fields":rows,
            "input_sha256":input_hashes,"rust_binary_sha256":digest(rust.engine()),
            "numpy_version":np.__version__,"authority_receipt_sha256":digest(args.authority_receipt),
            "authority_inputs_bound":bool(authority.get("input_sha256"))},indent=2))
        ok = len(rows) == len(expected) and all(row["bitwise_equal"] for row in rows)
        print("PASS" if ok else "FAIL",flush=True)
        return 0 if ok else 1
    inputs = d._acoustic_inputs(b)
    off, coef, forcing, state = inputs
    expected = timed("numpy.acoustic", lambda: d.advance_acoustic_step_v841(b.mesh,
        state,forcing,coef,dts=d._DTS,small_step=2,offcentering=off,
        fzm=np.asarray(b.vertical.fzm,"f4"),fzp=np.asarray(b.vertical.fzp,"f4"),
        rdzw=np.asarray(b.vertical.rdzw,"f4"),specified_zone_cell=b.masks.spec_zone_mask_cell,
        specified_zone_edge=b.masks.spec_zone_mask_edge))
    actual = timed("rust.acoustic", lambda: rust.acoustic(b,inputs))
    for name in state.__slots__:
        record("acoustic."+name,getattr(actual,name),getattr(expected,name))
    expected_flux = timed("numpy.vertical",lambda: _atmosphere_vertical_flux(
        b.scalars_stage[0],np.asarray(b.saved.vertical_velocity,"f4"),
        np.asarray(b.vertical.fzm,"f4"),np.asarray(b.vertical.fzp,"f4"),3,0.25))
    actual_flux = timed("rust.vertical",lambda: rust.vertical(b))
    record("vertical_flux",actual_flux,expected_flux)
    coef = d.build_advection_coefficients(b.mesh,config_scalar_adv_order=3,
        n_vert_levels=b.nlev,source_order_v841=True,allow_regional_sentinels=True)
    velocity = np.asarray(b.state.rho_u,"f4")
    expected_edges = timed("numpy.horizontal",lambda: _atmosphere_horizontal_edge_values(
        b.scalars_stage,velocity,b.dv_edge,
        np.where(b.cells_on_edge<0,b.ncells,b.cells_on_edge),
        np.asarray(coef.adv_coefs,"f4"),np.asarray(coef.adv_coefs_3rd,"f4"),
        coef.n_adv_cells_for_edge,coef.adv_cells_for_edge,3,0.25,
        config_apply_lbcs=True,bdy_mask_edge=b.masks.bdy_mask_edge))
    actual_edges = timed("rust.horizontal",lambda: rust.horizontal(b,coef))
    record("horizontal_edge_values",actual_edges,expected_edges)
    old = np.asarray(b.state.scalars,"f4")
    rho = np.asarray(b.state.rho,"f4")
    expected = timed("numpy.finish_composed",lambda: advance_scalars(b.mesh,old,b.scalars_stage,
        rho,rho*np.float32(1.000001),velocity,np.asarray(b.saved.vertical_velocity,"f4"),120,
        coefficients=coef,fzm=np.asarray(b.vertical.fzm,"f4"),fzp=np.asarray(b.vertical.fzp,"f4"),
        rdzw=np.asarray(b.vertical.rdzw,"f4"),rk_step=3,config_coef_3rd_order=0.25,
        config_apply_lbcs=True,bdy_mask_cell=b.masks.bdy_mask_cell,
        bdy_mask_edge=b.masks.bdy_mask_edge,advance_density=True,inv_area_cell=b.inv_area_cell))
    actual = timed("rust.finish",lambda: rust.finish(b,actual_edges,actual_flux[None,...]))
    record("scalars_after_transport",actual.scalars,expected.scalars)
    record("target_density",actual.density,expected.density)
    ok = all(row["bitwise_equal"] for row in rows)
    print("PASS" if ok else "FAIL",flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
