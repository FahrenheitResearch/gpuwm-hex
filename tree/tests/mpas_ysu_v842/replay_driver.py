"""Execute the complete CUDA YSU driver slice against strict Fortran inputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import cupy as cp
import numpy as np
from hexcore.mpasphys import ysu_v842 as ysu
from hexcore.mpasphys.driver_v842 import SlotCall, SlotRegistry, Backend, StepFlags
from oracle import Oracle, constants, fixtures, provenance, cuda_provenance
from replay_cuda import pack


def replay(work, folder):
    os.environ.setdefault("GCOV_PREFIX", str((work.parent / "driver-gcov").resolve()))
    oracle = Oracle(work)
    columns = list(fixtures(folder))
    flags = columns[0][2]
    values = constants()
    header = ysu.shared_device_math_header()
    words = calls = differing = 0
    mismatches = []
    per_array = {}
    for option in (False, True):
        for radiation_mode in ("split", "cancel", "signed-zero"):
            state = pack([c[1] for c in columns], flags)
            carrier_fields = dict(pres_hyd_p="p2d", pres2_hyd_p="p2di", t_p="tx", dz_p="dz8w2d",
                                  pi_p="pi2d", u_p="ux", v_p="vx", qv_p="qvx", qc_p="qcx",
                                  qi_p="qix", psfc_p="psfcpa")
            carriers = {k: state[v] for k, v in carrier_fields.items()}
            # Native prepared storage has kme=kte+1. Poison the unused
            # profile row so a launch using allocation bounds cannot pass.
            # The pressure interfaces already include their upper level.
            for name in carriers:
                if name not in ("pres2_hyd_p", "psfc_p"):
                    extra = cp.full((1, carriers[name].shape[1]), cp.nan, cp.float32)
                    carriers[name] = cp.concatenate((carriers[name], extra), axis=0)
            diag = {k: state[k].copy() for k in ("znt", "ust", "hfx", "qfx", "wspd", "br", "u10", "v10")}
            diag.update(fm=state["psim"], fh=state["psih"])
            for name in ("hpbl", "kpbl", "delta", "wstar"):
                diag[name] = cp.full_like(state["kpbl1d" if name == "kpbl" else name], -999)
            for name in ("kzh", "kzm", "kzq", "exch_h"):
                diag[name] = cp.full_like(state["tx"], -999)
            tendencies = {k: cp.full_like(state["tx"], -999) for k in
                          ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten")}
            source = np.stack([c[1]["rthraten"] for c in columns], axis=-1)
            if radiation_mode == "split":
                lw = np.float32(.5)*source
                sw = np.float32(.5)*source + np.float32(1.e-6)
            elif radiation_mode == "cancel":
                lw = np.full_like(source, .125)
                sw = np.full_like(source, -.125) + source
            else:
                lw = np.full_like(source, -0.0)
                sw = np.full_like(source, 0.0)
            tendencies.update(rthratenlw=cp.asarray(lw), rthratensw=cp.asarray(sw))
            aliases = dict(g="gravity", rovcp="rcp", rovg="rdg", rd="R_d",
                           rv="R_v", ep1="ep_1", ep2="ep_2")
            pools = dict(prepared=carriers, diag_physics=diag,
                         sfc_input={"xland": state["xland"]}, tend_physics=tendencies,
                         constants={aliases.get(k, k): v for k, v in values.items()},
                         scalar_state=dict(f_qc=True, f_qi=True))
            registry = SlotRegistry()
            registry.register("pbl", "bl_ysu", Backend.HEX_CUDA, ysu.launch_slot)
            registry.run(SlotCall("pbl", "bl_ysu", Backend.HEX_CUDA, 1, 0,
                                 float(values["dt"]), StepFlags(), pools,
                                 {"config_ysu_pblmix": option}))
            actual = {**{k: cp.asnumpy(v) for k, v in diag.items()},
                      **{k: cp.asnumpy(tendencies[k]) for k in ysu._WRAPPER_OUTPUTS if k in tendencies},
                      "rthblten": cp.asnumpy(tendencies["rthblten"])}
            mapping = dict(rublten="utnp", rvblten="vtnp", rthblten="ttnp", rqvblten="qvtnp",
                           rqcblten="qctnp", rqiblten="qitnp", hpbl="hpbl", kpbl="kpbl1d",
                           kzh="exch_hx", kzm="exch_mx", kzq="exch_hx", exch_h="diagnostic_exch_h",
                           delta="delta", wstar="wstar", u10="u10", v10="v10")
            for col, (tag, arrays, _) in enumerate(columns):
                expected, _ = oracle.run(arrays, {**flags, "nmix": 0,
                                                 "ysu_topdown_pblmix": int(option), "flag_bep": 0},
                                        driver_radiation=(np.ascontiguousarray(lw[:, col]),
                                                          np.ascontiguousarray(sw[:, col])))
                for name, target in mapping.items():
                    a = actual[name][..., col].reshape(-1).view(np.uint32)
                    b = expected[target].reshape(-1).view(np.uint32)
                    diff = int(np.count_nonzero(a != b))
                    words += a.size
                    differing += diff
                    totals = per_array.setdefault(name, dict(compared_words=0, differing_words=0))
                    totals["compared_words"] += a.size
                    totals["differing_words"] += diff
                    if diff:
                        mismatches.append(dict(column=tag, option=option, radiation=radiation_mode,
                                               array=name, differing_words=diff))
                calls += 1
    assert sum(v["compared_words"] for v in per_array.values()) == words
    assert sum(v["differing_words"] for v in per_array.values()) == differing
    return dict(**provenance(work, folder), **cuda_provenance(), schema="mpas-ysu-v842/driver-replay-v1", host=platform.node(), process_count=1,
                identity_tier="SYNTHETIC-CUDA-DRIVER", compared_calls=calls,
                compared_words=words, differing_words=differing, mismatches=mismatches,
                per_array=per_array,
                cuda_sha256=ysu.CUDA_SHA256, compiler_options=ysu.COMPILER_OPTIONS,
                conversions=["LW+SW before scheme", "Exner temperature-to-theta", "tendency pools",
                             "kzh/kzm/kzq", "zero diagnostic exch_h", "surface u10/v10 preserved"],
                prepared_storage="kme=kte+1; inactive profile row NaN; upper pressure interface retained",
                registration="f4 SlotRegistry -> launch_slot -> complete driver_pbl",
                native_defect_refusals=False,
                not_proven=["real CLI selection", "real columns", "whole forecast", "G0-G2"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.work, args.fixtures)
    args.receipt.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["differing_words"]))
