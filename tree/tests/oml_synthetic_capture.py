#!/usr/bin/env python3
"""Seal synthetic MPAS OML columns in f1 native scalar record schemas.

Native numerics: pristine module_sf_oml.F:9-160 through the strict standalone
oracle. Rust owns encoding; native capture headers supply only the schema.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from test_mpas_oml_v842 import columns, CONSTANTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("template", "output", "transport", "oracle", "tool", "inventory", "pins", "words", "native_replay", "receipt"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--columns",type=int,required=True)
    p.add_argument("--dt",type=float,required=True)
    p.add_argument("--gammas",type=float,nargs="+",required=True)
    p.add_argument("--relaxations",type=float,nargs="+",required=True)
    a=p.parse_args()
    if a.columns<32:raise ValueError("at least 32 columns required for the explicit branch witnesses")
    native=ct.CDLL(str(a.oracle.resolve())).replay
    native.argtypes=[ct.c_int]*5+[np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")]+[ct.c_float]*7
    native.restype=None
    transport=ct.CDLL(str(a.transport.resolve()))
    transport.oml_fixture_write.argtypes=[ct.c_char_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_char_p]
    a.output.mkdir(parents=True,exist_ok=False)
    base=columns(a.columns)
    base[16]*=np.float32(1.e-4)
    rows=[]
    for gamma in a.gammas:
        for relaxation in a.relaxations:
            target=a.output/("combination-%03d"%len(rows));prefix=target/"dumps"
            target.mkdir()
            after=base.copy()
            constants=np.asarray((*CONSTANTS[:2],a.dt,gamma,relaxation),dtype=np.float32)
            native(a.columns,a.columns,0,1,0,after,*constants,0.,0.)
            before=base[:17].copy();post=after[:17].copy()
            status=transport.oml_fixture_write(str(a.template.resolve()).encode(),before.ctypes.data,
                post.ctypes.data,constants.ctypes.data,a.columns,str(prefix.resolve()).encode())
            if status:raise RuntimeError("Rust could not encode synthetic native scalar fixture")
            recipe="Synthetic branch columns; pristine strict OML standalone; no real forecast capture"
            command=["python",str(a.tool.resolve()),"--scheme","oml","--dump",str(prefix),"--words",str(a.words),"--recipe",recipe]
            subprocess.run(command+["--seal","--inventory",str(a.inventory),"--pins",str(a.pins)],check=True,capture_output=True,text=True)
            result=target/"native-replay.json"
            subprocess.run(command+["--backend","native-replay","--executable",str(a.native_replay),"--results",str(target/"native-results"),"--receipt",str(result)],check=True,capture_output=True,text=True)
            proof=json.loads(result.read_text())
            if not proof["all_outputs_bitwise_equal"]:raise RuntimeError("native scalar replay differs from strict standalone")
            rows.append(dict(dump=str(prefix),gamma=gamma,relaxation=relaxation,columns=a.columns,
                dump_root_sha256=proof["dump_root_sha256"],native_replay_receipt=str(result),
                all_outputs_bitwise_equal=True))
    a.receipt.write_text(json.dumps(dict(schema="p19-oml-synthetic-f1-fixtures/v1",forecast_acceptance=False,
        authority="pristine strict Fortran OML standalone vs f1 native scalar replay",combinations=rows,
        oracle_sha256=hashlib.sha256(a.oracle.read_bytes()).hexdigest(),transport_sha256=hashlib.sha256(a.transport.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+"\n")


if __name__=="__main__":main()
