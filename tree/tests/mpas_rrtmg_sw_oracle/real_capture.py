"""Replay f1's native SW call through CUDA; Rust owns typed record I/O.

This is a column diagnostic executable for column_oracle_replay.py. Hidden
carriers that CUDA does not mutate are preserved from PRE, never from POST.
The external oracle compares these carriers too; a native mutation fails.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import numpy as np


def main(prefix, output):
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave, FIELD_NAMES, SURFACE_NAMES
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mpas_rrtmg_sw_oracle.replay import read_swd, port_provenance
    word = os.environ["MPAS_SW_WORDS"]
    prefix, output = Path(prefix), Path(output)
    cache = {}

    def path(name, phase="pre"):
        return prefix.with_name(prefix.name + f".{phase}.{name}.bin")

    def read(name):
        if name not in cache:
            member = path(name)
            header = json.loads(subprocess.check_output([word, "header", str(member), "-"]))
            payload = subprocess.check_output([word, "export", str(member), "-"])
            dtype = {1: "<f4", 2: "<f8", 3: "<i4", 4: "<i4", 5: "u1"}[header["code"]]
            cache[name] = np.frombuffer(payload, dtype=dtype).reshape(header["shape"], order="F").copy()
        return cache[name]

    def scalar(name):
        return read(name).item()

    nz = scalar("kte") - scalar("kts") + 1
    first, last = scalar("its") - scalar("ims"), scalar("ite") - scalar("ims") + 1
    kfirst = scalar("kts") - scalar("kms")
    assert scalar("jts") == scalar("jte") == scalar("jms") == scalar("jme"), "native MPAS j dimension must be singleton"
    fields = {n: read(n)[:, kfirst:kfirst+nz, 0] for n in FIELD_NAMES}
    surface = {n: read(n)[:, 0] for n in SURFACE_NAMES}
    p8w, t8w = (read(n)[:, kfirst:kfirst+nz+1, 0] for n in ("p8w", "t8w"))
    controls = {n: scalar(n) for n in ("xtime", "gmt", "julday", "radt", "degrad", "declin", "solcon",
                "icloud", "cldovrlp", "idcor", "o3input", "has_reqc", "has_reqi", "has_reqs", "aer_opt")}
    pin, ozone = read("pin"), read("o3clim")[:, :, 0]
    aerosol = None
    if path("tauaer3d").exists() and scalar("tauaer3d.present"):
        aerosol = {n: read(n+"3d")[:, kfirst:kfirst+nz, 0, :].transpose(0, 2, 1)
                   for n in ("tauaer", "ssaaer", "asyaer")}
    dump = read_swd(os.environ["MPAS_SW_TABLES"])
    engine = MPASShortwave(raw_coefficients={k: cp.asarray(v) for k, v in dump.items() if k.startswith("kg") and v.ndim})
    names = ("coszr", "gsw", "swcf", "rthratensw", "swupt", "swuptc", "swdnt", "swdntc",
             "swupb", "swupbc", "swdnb", "swdnbc", "swddir", "swddni", "swddif")
    results = {}
    templates = {}
    for name in names:
        templates[name] = path(name) if path(name).exists() else path(name, "post")
        if path(name).exists():
            results[name] = read(name)
        else:
            # INTENT(OUT) has no PRE payload. Read only record metadata;
            # the CUDA result supplies every numerical output word.
            header = json.loads(subprocess.check_output([word, "header", str(templates[name]), "-"]))
            results[name] = np.zeros(header["shape"], dtype=np.float32)
    chunk = int(os.environ.get("MPAS_SW_REPLAY_CHUNK", "512"))
    for start in range(first, last, chunk):
        stop = min(start+chunk, last)
        options = dict(controls, pin=cp.asarray(pin), o3clim=cp.asarray(ozone[start:stop]))
        if aerosol is not None:
            options["aerosol"] = {n: cp.asarray(v[start:stop]) for n, v in aerosol.items()}
        out = engine.rrtmg_swrad({n: cp.asarray(v[start:stop]) for n, v in fields.items()},
            {n: cp.asarray(v[start:stop]) for n, v in surface.items()}, cp.asarray(p8w[start:stop]),
            cp.asarray(t8w[start:stop]), **options)
        for name in names:
            value = out[name].get()
            if results[name].ndim == 3:
                results[name][start:stop, kfirst:kfirst+nz, 0] = value
            else:
                results[name][start:stop, 0] = value
        print(json.dumps({"columns_replayed": stop-first, "columns_total": last-first}), flush=True)
        del out
        cp.get_default_memory_pool().free_all_blocks()
    for name, array in results.items():
        target = output.with_name(output.name + f".post.{name}.bin")
        subprocess.run([word, "import", str(templates[name]), str(target)], input=array.tobytes(order="F"), check=True)
        flag = path(name+".present")
        if not flag.exists():
            flag = path(name+".present", "post")
        if flag.exists():
            shutil.copyfile(flag, output.with_name(output.name+f".post.{name}.present.bin"))
    for member in prefix.parent.glob(prefix.name+".pre.*.bin"):
        name = member.name[len(prefix.name+".pre."):]
        if name.startswith(("state.", "local.")):
            shutil.copyfile(member, output.with_name(output.name+".post."+name))
    print(json.dumps({"port_provenance": port_provenance(), "whole_model_proof": False}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
