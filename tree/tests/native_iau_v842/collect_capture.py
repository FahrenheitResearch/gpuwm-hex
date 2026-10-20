"""Select complete IAU native captures without decoding field data in Python."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil

FIELDS = {
    "zz": "block.structs.mesh.zz", "rho_zz": "block.structs.state.rho_zz.t2",
    "theta_m": "block.structs.state.theta_m.t1", "scalars": "block.structs.state.scalars.t1",
    "rho_edge": "block.structs.diag.rho_edge", "u_amb": "block.structs.tend_iau.u",
    "rho_amb": "block.structs.tend_iau.rho", "theta_amb": "block.structs.tend_iau.theta",
    "scalars_amb": "block.structs.tend_iau.scalars",
    "tend_ru": "argument.tend_ru_physics", "tend_rtheta": "argument.tend_rtheta_physics",
    "tend_rho": "argument.tend_rho_physics", "tend_scalars": "block.structs.tend.scalars_tend",
}
CONTROLS = {
    "nCellsSolve": "block.structs.mesh.nCellsSolve", "nEdgesSolve": "block.structs.mesh.nEdgesSolve",
    "moist_start": "block.structs.state.moist_start", "moist_end": "block.structs.state.moist_end",
    "index_qv": "block.structs.state.index_qv", "itimestep": "argument.itimestep",
    "dt": "argument.dt", "window": "block.configs.config_IAU_window_length_s",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model_log=(args.run/"log.atmosphere.0000.out").read_text()
    if "MPI task count: 1" not in model_log:
        raise ValueError("this capture deck requires exactly one MPI process")
    args.out.mkdir(parents=True, exist_ok=False)
    decoder = ctypes.CDLL(str(args.reader.resolve()))
    read = decoder.iau_oracle_control
    read.argtypes = (ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
                     ctypes.POINTER(ctypes.c_uint32))
    read.restype = ctypes.c_int32
    contexts = set()
    hashes = {}
    metadata = {"option": "on", "fields": {}, "capture": json.loads(
        (args.run/"capture-provenance.json").read_text()), "controls": {}}
    def select(name, phase="pre"):
        found = list((args.run/"dumps").glob(f"boundary__atm_add_tend_anal_incr.*.{phase}.{name}.bin"))
        if len(found) != 1:
            raise ValueError(f"requires exactly one complete pre-call field {name}: {len(found)}")
        file = found[0]
        shutil.copyfile(file, args.out/file.name)
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        return file
    for arg, name in FIELDS.items():
        file = select(name)
        metadata["fields"][arg] = file.name
    metadata["native_post_fields"] = {
        arg: select(FIELDS[arg], "post").name
        for arg in ("tend_ru", "tend_rtheta", "tend_rho", "tend_scalars")
    }
    for arg, name in CONTROLS.items():
        file = select(name)
        kind = 1 if arg in ("dt", "window") else 3
        header = (ctypes.c_int32*21)()
        word = ctypes.c_uint32()
        status = read(os.fsencode(file), kind, header, ctypes.byref(word))
        if status != 0:
            raise ValueError(f"Rust rejected actual native control {name}: {status}")
        contexts.add(tuple(header[13:16]))
        if kind == 1:
            value = ctypes.cast(ctypes.byref(word),ctypes.POINTER(ctypes.c_float)).contents.value
        else:
            value = ctypes.cast(ctypes.byref(word),ctypes.POINTER(ctypes.c_int32)).contents.value
        metadata[arg] = value
        metadata["controls"][arg] = {"file": file.name, "header": list(header)}
    if len(contexts) != 1:
        raise ValueError("native controls mix calls")
    edge_ids = select("block.structs.mesh.indexToEdgeID")
    metadata["native_edge_ids"] = edge_ids.name
    metadata["record_sha256"] = hashes
    (args.out/"manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"records": len(hashes), "context": list(contexts)[0],
                      "manifest": str(args.out/"manifest.json")}))


if __name__ == "__main__":
    main()
