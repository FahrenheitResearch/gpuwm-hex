"""Replay genuine standalone MPAS fixtures; GPU invocation must use the GPU queue on a shared host."""
from pathlib import Path
import argparse
import json
import hashlib
import struct
import numpy as np


def read_swd(path):
    """Test-only stream decoder. Production loading belongs to the Rust seam."""
    records = {}
    with Path(path).open("rb") as f:
        assert f.read(4) == b"SWD1"
        while raw := f.read(4):
            length = struct.unpack(">i", raw)[0]
            name = f.read(length).decode()
            dtype, rank = struct.unpack(">ii", f.read(8))
            dims = struct.unpack(">" + "i" * rank, f.read(4 * rank))
            count = int(np.prod(dims)) if dims else 1
            data = np.frombuffer(f.read(count * 4), dtype=">f4" if dtype == 0 else ">i4")
            records[name] = data.astype(np.float32 if dtype == 0 else np.int32).reshape(dims, order="F")
    return records


def port_provenance():
    from hexcore.mpasphys import rrtmg_sw_v842 as port
    import gpuwm.core.rrtmg_sw as legacy
    kernels=Path(legacy.__file__).parent/"kernels"
    files=[Path(port.__file__),Path(legacy.__file__),
           kernels/"glibc_flt32.cuh",kernels/"glibc_trig_flt32.cuh"]
    return {"compile_options":port.COMPILE_OPTIONS,
            "source_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def replay(tables_path, columns_path, receipt):
    import cupy as cp
    from gpuwm.core.rrtmg_sw import tables_from_dump
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave, FIELD_NAMES, SURFACE_NAMES
    dump = read_swd(tables_path)
    # Production receives the same roster from the Rust binary decoder.
    raw = {k: cp.asarray(v) for k, v in dump.items() if k.startswith("kg") and v.ndim}
    engine = MPASShortwave(raw_coefficients=raw)
    from gpuwm.core.rrtmg_sw import _pack_cuda_tables
    expected_tables, _ = _pack_cuda_tables(tables_from_dump(dump))
    got_tables = engine.tab_gpu.get()
    mismatch = got_tables.view(np.uint32) != expected_tables.view(np.uint32)
    differing = np.flatnonzero(mismatch)
    initialization = {"words": int(expected_tables.size), "differing_words": int(differing.size),
                      "first_indices": differing[:20].tolist(),
                      "native": expected_tables[differing[:20]].tolist(),
                      "cuda": got_tables[differing[:20]].tolist()}
    Path(str(receipt).replace(".json", "-initialization.json")).write_text(json.dumps(initialization, indent=2) + "\n")
    print(json.dumps({"initialization": initialization}), flush=True)
    solar_file = Path(columns_path).parent / "solar.bin"
    solar_words = 0
    if solar_file.exists():
        solar = np.frombuffer(solar_file.read_bytes(), dtype=">f4").astype(np.float32).reshape(-1, 5)
        errors = []
        time_file = Path(columns_path).parent / "driver_times.bin"
        times = np.frombuffer(time_file.read_bytes(), dtype=">f4").astype(np.float32).reshape(-1, 4) if time_file.exists() else None
        control_errors = []
        overlaps = ("none", "random", "maximum_random", "maximum", "exponential", "exponential_random")
        for index, (julian, degrad, dpd, declin, solcon) in enumerate(solar):
            got = engine.radconst(julian, degrad, dpd).get()
            expected = np.asarray((declin, solcon), dtype=np.float32)
            if times is not None:
                seconds, interval, native_minutes, native_interval = times[index]
                overlap = index % 6
                latitude_varying = bool(index % 2)
                cfg = dict(config_radt_cld_overlap=overlaps[overlap],
                           config_radt_cld_dcorrlen="latitude_varying" if latitude_varying else "constant",
                           config_o3climatology=bool(index % 2))
                controls = engine.driver_controls(cfg, xtime_s=seconds, dt_radtsw=interval,
                                                  curr_julday=julian, degrad=degrad, dpd=dpd)
                numeric = np.asarray([controls[n] for n in ("xtime","radt","declin","solcon")],np.float32)
                native = np.asarray((native_minutes,native_interval,declin,solcon),np.float32)
                error = int((numeric.view(np.uint32)!=native.view(np.uint32)).sum())
                error += int(controls["cldovrlp"] != overlap)
                error += int(controls["idcor"] != int(overlap>=4 and latitude_varying))
                error += int(controls["o3input"] != (2 if index % 2 else 0))
                if error:
                    control_errors.append(dict(row=index,differing_words=error,native=native.tolist(),cuda=numeric.tolist()))
            mask = got.view(np.uint32) != expected.view(np.uint32)
            solar_words += int(np.count_nonzero(mask))
            if mask.any():
                errors.append({"row": index, "julian": float(julian), "native": expected.tolist(), "cuda": got.tolist()})
        if times is not None:
            control_result = {"cases": len(times), "differing_words":sum(row["differing_words"] for row in control_errors), "errors":control_errors}
            Path(str(receipt).replace(".json", "-controls.json")).write_text(json.dumps(control_result,indent=2)+"\n")
            solar_words += control_result["differing_words"]
            print(json.dumps({"driver_controls":{k:v for k,v in control_result.items() if k != "errors"}}),flush=True)
        solar_result = {"cases": len(solar), "words": int(solar.shape[0] * 2), "differing_words": solar_words, "errors": errors}
        Path(str(receipt).replace(".json", "-solar.json")).write_text(json.dumps(solar_result, indent=2) + "\n")
        print(json.dumps({k:v for k,v in solar_result.items() if k != "errors"}), flush=True)
    records = read_swd(columns_path)
    cases = sorted({key.split("/")[0] for key in records})
    summary = []
    for case in cases:
        inputs = {k.split("/", 2)[-1]: v for k, v in records.items() if k.startswith(case + "/in/")}
        expected = {k.split("/", 2)[-1]: v for k, v in records.items() if k.startswith(case + "/out/")}
        nz = expected["rthratensw"].size
        fields = {n: cp.asarray(inputs[n][:nz][None]) for n in FIELD_NAMES}
        surface = {n: cp.asarray(inputs[n].reshape(1)) for n in SURFACE_NAMES}
        options = {n: inputs[n].item() for n in (
            "xtime", "gmt", "julday", "radt", "degrad", "declin", "solcon", "icloud",
            "cldovrlp", "idcor", "o3input", "has_reqc", "has_reqi", "has_reqs", "aer_opt")}
        options.update(pin=cp.asarray(inputs["pin"]), o3clim=cp.asarray(inputs["o3clim"][None]))
        if "tauaer" in inputs:
            options["aerosol"] = {n: cp.asarray(inputs[n].T[None]) for n in ("tauaer", "ssaaer", "asyaer")}
        actual = engine.rrtmg_swrad(fields, surface, cp.asarray(inputs["p8w"][None]),
                                  cp.asarray(inputs["t8w"][None]), **options)
        row = {"case": case, "options": {k: v for k, v in options.items() if not isinstance(v, (cp.ndarray, dict))}, "arrays": {}}
        for name, reference in expected.items():
            got = actual[name].get().reshape(reference.shape)
            mismatch = got.view(np.uint32) != reference.view(np.uint32)
            count = int(np.count_nonzero(mismatch))
            result = {"words": int(reference.size), "differing_words": count}
            if count:
                first = int(np.flatnonzero(mismatch)[0])
                result.update(first_index=first, native=float(reference.reshape(-1)[first]),
                              cuda=float(got.reshape(-1)[first]),
                              native_word=int(reference.reshape(-1).view(np.uint32)[first]),
                              cuda_word=int(got.reshape(-1).view(np.uint32)[first]))
            row["arrays"][name] = result
        summary.append(row)
        if "aerosol" in options:
            # Native wrapper ignores any source halo and supplies its own top
            # radiation layer (module_ra_rrtmg_sw.F:10562-10572).
            model_options = dict(options, aerosol={n: a[:, :, :nz].copy()
                                                   for n, a in options["aerosol"].items()})
            model_actual = engine.rrtmg_swrad(fields, surface, cp.asarray(inputs["p8w"][None]),
                                            cp.asarray(inputs["t8w"][None]), **model_options)
            model_arrays = {}
            for name, reference in expected.items():
                got = model_actual[name].get().reshape(reference.shape)
                model_arrays[name] = {"words": int(reference.size), "differing_words":
                                     int(np.count_nonzero(got.view(np.uint32) != reference.view(np.uint32)))}
            summary.append({"case": case + "/model-level-aerosol", "arrays": model_arrays})
    # Group source columns with identical scalar controls; this exercises
    # mixed day/night batches, partial chunks and borrowing the tables on a
    # second CUDA stream against the native oracle rather than a CUDA twin.
    groups = {}
    control_names = ("xtime", "gmt", "julday", "radt", "degrad", "declin", "solcon", "icloud",
                     "cldovrlp", "idcor", "o3input", "has_reqc", "has_reqi", "has_reqs", "aer_opt")
    for case in cases:
        controls = tuple(records[f"{case}/in/{n}"].item() for n in control_names)
        groups.setdefault(controls, []).append(case)
    batch_rows = []
    other_stream = cp.cuda.Stream(non_blocking=True)
    for controls, group in groups.items():
        for chunk in (1, 3):
            with other_stream:
                nz = records[f"{group[0]}/out/rthratensw"].size
                fields = {n: cp.asarray(np.stack([records[f"{c}/in/{n}"][:nz] for c in group])) for n in FIELD_NAMES}
                surface = {n: cp.asarray(np.asarray([records[f"{c}/in/{n}"].item() for c in group], np.float32)) for n in SURFACE_NAMES}
                options = dict(zip(control_names, controls))
                options.update(pin=cp.asarray(records[f"{group[0]}/in/pin"]),
                               o3clim=cp.asarray(np.stack([records[f"{c}/in/o3clim"] for c in group])), column_chunk=chunk)
                if f"{group[0]}/in/tauaer" in records:
                    options["aerosol"] = {n: cp.asarray(np.stack([records[f"{c}/in/{n}"].T for c in group]))
                                          for n in ("tauaer", "ssaaer", "asyaer")}
                actual = engine.rrtmg_swrad(fields, surface, cp.asarray(np.stack([records[f"{c}/in/p8w"] for c in group])),
                    cp.asarray(np.stack([records[f"{c}/in/t8w"] for c in group])), **options)
                for row, case in enumerate(group):
                    arrays = {}
                    for name in actual:
                        reference = records[f"{case}/out/{name}"]
                        got = actual[name][row].get().reshape(reference.shape)
                        mask = got.view(np.uint32) != reference.view(np.uint32)
                        arrays[name] = {"words": int(reference.size), "differing_words": int(mask.sum())}
                    batch_rows.append({"case": case, "chunk": chunk, "batch_columns": len(group), "arrays": arrays})
    total = sum(a["differing_words"] for r in summary for a in r["arrays"].values())
    total += sum(a["differing_words"] for r in batch_rows for a in r["arrays"].values())
    result = {"authority": "pristine MPAS-A v8.4.2 standalone wrapper", "cases": len(cases),
              "differing_words": total, "rows": summary, "batch_rows": batch_rows, "whole_model_proof": False,
              "port_provenance":port_provenance()}
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "batch_rows")}))
    cloud_words = 0
    cloud_file = Path(columns_path).parent / "cloud.bin"
    if cloud_file.exists():
        cloud_words = replay_cloud(engine, cloud_file, str(receipt).replace(".json", "-cloud.json"))
    driver_words = 0
    driver_file = Path(columns_path).parent / "driver.swd"
    if driver_file.exists():
        driver_words = replay_driver(engine, driver_file, str(receipt).replace(".json", "-driver.json"))
    mcica_words = 0
    mcica_file = Path(columns_path).parent / "mcica.bin"
    if mcica_file.exists():
        mcica_words = replay_mcica(engine, mcica_file, str(receipt).replace(".json", "-mcica.json"))
    composition_words = 0
    composition_file = Path(columns_path).parent / "composition.bin"
    if composition_file.exists():
        composition_words = replay_composition(engine, composition_file, str(receipt).replace(".json", "-composition.json"))
    extreme_file = Path(columns_path).parent / "composition-extreme.bin"
    if extreme_file.exists():
        composition_words += replay_composition(engine, extreme_file, str(receipt).replace(".json", "-composition-extreme.json"))
    dense_file = Path(columns_path).parent / "composition-dense.bin"
    if dense_file.exists():
        composition_words += replay_composition(engine, dense_file, str(receipt).replace(".json", "-composition-dense.json"))
    reftra_words = 0
    reftra_file = Path(columns_path).parent / "reftra.bin"
    if reftra_file.exists():
        reftra_words = replay_reftra(engine, reftra_file, str(receipt).replace(".json", "-reftra.json"))
    result["all_interfaces_differing_words"] = (total + int(differing.size) + solar_words +
        cloud_words + driver_words + mcica_words + composition_words + reftra_words)
    result["all_interfaces_bitwise_equal"] = result["all_interfaces_differing_words"] == 0
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    return result["all_interfaces_differing_words"]


def replay_reftra(engine,path,receipt):
    import cupy as cp
    data = Path(path).read_bytes()
    nl,cases = struct.unpack_from(">ii",data)
    offset = 8
    def floats(count):
        nonlocal offset
        out = np.frombuffer(data,dtype=">f4",count=count,offset=offset).astype(np.float32)
        offset += count*4
        return out
    g,tau,ssa = [cp.asarray(floats(nl)) for _ in range(3)]
    check = cp.asarray(np.frombuffer(data,dtype=">i4",count=nl,offset=offset).astype(np.int32))
    offset += nl*4
    rows=[]
    for case in range(cases):
        cosine = floats(1)[0]
        ref = floats(nl*4).reshape(4,nl)
        actual = engine.reftra_sw(g,cosine,tau,ssa,check)
        got = np.stack([actual[n].get() for n in ("pref","prefd","ptra","ptrad")])
        mask = got.view(np.uint32)!=ref.view(np.uint32)
        row = {"case":case,"cosine":float(cosine),"words":int(ref.size),"differing_words":int(mask.sum())}
        if mask.any():
            first=int(np.flatnonzero(mask)[0])
            row.update(first_index=first,native=float(ref.flat[first]),cuda=float(got.flat[first]),
                             native_word=hex(int(ref.view(np.uint32).flat[first])),
                             cuda_word=hex(int(got.view(np.uint32).flat[first])))
        rows.append(row)
    assert offset==len(data)
    result={"cases":cases,"layers_per_case":nl,"differing_words":sum(r["differing_words"] for r in rows),"rows":rows,"whole_model_proof":False}
    Path(receipt).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"reftra":{k:v for k,v in result.items() if k!="rows"}}),flush=True)
    return result["differing_words"]


def replay_composition(engine, path, receipt):
    """Genuine lower module interface, including options fixed by the driver."""
    import cupy as cp
    data = Path(path).read_bytes()
    nc, nl, ng, cases = struct.unpack_from(">iiii", data)
    offset = 16
    def floats(shape):
        nonlocal offset
        count = int(np.prod(shape))
        x = np.frombuffer(data, dtype=">f4", count=count, offset=offset).astype(np.float32)
        offset += count * 4
        return x.reshape(shape, order="F")
    thermodynamic = [cp.asarray(floats((nc, levels))) for levels in (nl,nl+1,nl,nl+1)]
    ts, alb, cosz, adjes = [cp.asarray(floats((nc,))) for _ in range(4)]
    scon = cp.full(nc, floats((1,))[0], cp.float32)
    clouds = [cp.asarray(floats((ng,nc,nl)).transpose(1,2,0).copy()) for _ in range(8)]
    radii = [cp.asarray(floats((nc,nl))) for _ in range(3)]
    tau, ssa, asy = [cp.asarray(floats((nc,nl,14)).transpose(0,2,1).copy()) for _ in range(3)]
    ec = cp.asarray(floats((nc,nl,6)))
    names = ("swuflx","swdflx","swhr","swuflxc","swdflxc","swhrc","swuflxcln","swdflxcln",
             "sibvisdir","sibvisdif","sibnirdir","sibnirdif","swdkdir","swdkdif","swdkdirc")
    rows = []
    for case in range(cases):
        overlap, day, aer, zero = struct.unpack_from(">iiii", data, offset)
        offset += 16
        gases = cp.asarray(floats((nc,nl,6)))
        expected = [floats((nc, nl if name in ("swhr","swhrc") else nl+1)) for name in names]
        actual = engine.rrtmg_sw_batched_device(nc,nl,overlap,*thermodynamic,ts,
            *(cp.ascontiguousarray(gases[:,:,i]) for i in range(6)),alb,alb,alb,alb,cosz,adjes,day,scon,
            2,2,1,*clouds,*radii,aer_opt=aer,mcica_layout="column",aerosol=(tau,asy,ssa),ecaer=ec)
        arrays = {}
        for name, ref in zip(names,expected):
            got = actual[name].get()
            mask = got.view(np.uint32) != ref.view(np.uint32)
            entry = {"words":int(ref.size),"differing_words":int(mask.sum())}
            if mask.any():
                first = int(np.flatnonzero(mask)[0])
                entry.update(first_index=first,native=float(ref.flat[first]),cuda=float(got.flat[first]),
                             native_word=hex(int(ref.view(np.uint32).flat[first])),
                             cuda_word=hex(int(got.view(np.uint32).flat[first])))
            arrays[name] = entry
        rows.append(dict(case=case,overlap=overlap,day=day,aerosol=aer,zero_gases=zero,arrays=arrays))
    assert offset == len(data)
    earth = Path(path).with_name("earth_sun.bin").read_bytes()
    earth_errors = []
    for index in range(len(earth)//8):
        day, reference = struct.unpack_from(">if",earth,index*8)
        got = engine.earth_sun(day).get()
        ref = np.asarray([reference],np.float32)
        if (got.view(np.uint32)!=ref.view(np.uint32)).any():
            earth_errors.append(dict(day=day,native=reference,cuda=float(got[0])))
    total = sum(a["differing_words"] for row in rows for a in row["arrays"].values()) + len(earth_errors)
    result = {"cases":cases,"differing_words":total,"earth_sun_cases":len(earth)//8,
              "earth_sun_errors":earth_errors,"rows":rows,"whole_model_proof":False,
              "port_provenance":port_provenance()}
    Path(receipt).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"composition":{k:v for k,v in result.items() if k not in ("rows","earth_sun_errors")}}),flush=True)
    return total


def replay_mcica(engine, path, receipt):
    import cupy as cp
    data = Path(path).read_bytes()
    nc, nl, ng, cases = struct.unpack_from(">iiii", data)
    seed, juldat, lat = struct.unpack_from(">iif", data, 16)
    offset = 28
    def floats(shape):
        nonlocal offset
        count = int(np.prod(shape))
        x = np.frombuffer(data, dtype=">f4", count=count, offset=offset).astype(np.float32)
        offset += count * 4
        return x.reshape(shape, order="F")
    inputs = [cp.asarray(floats((nc, nl))) for _ in range(9)]
    optics = [cp.asarray(floats((14, nc, nl)).transpose(1, 2, 0).copy()) for _ in range(4)]
    rows = []
    names = ("cldfmcl", "ciwpmcl", "clwpmcl", "cswpmcl", "taucmcl", "ssacmcl", "asmcmcl", "fsfcmcl",
             "reicmcl", "relqmcl", "resnmcl")
    for case in range(cases):
        overlap, irng, idcor, native_irng = struct.unpack_from(">iiii", data, offset)
        offset += 16
        expected = [floats((ng, nc, nl)).transpose(1, 2, 0).copy() for _ in range(8)]
        expected += [floats((nc, nl)) for _ in range(3)]
        actual = engine.mcica_subcol_sw(*inputs[:5], *inputs[6:9], *optics, inputs[5],
            cp.full(nc, lat, cp.float32), icld=overlap, irng=irng, idcor=idcor, permuteseed=seed, juldat=juldat)
        arrays = {}
        for name, ref in zip(names, expected):
            got = actual[name].get()
            mask = got.view(np.uint32) != ref.view(np.uint32)
            entry = {"words": int(ref.size), "differing_words": int(mask.sum())}
            if mask.any():
                first = int(np.flatnonzero(mask)[0])
                entry.update(first_index=first, native=float(ref.flat[first]), cuda=float(got.flat[first]))
            arrays[name] = entry
        rows.append({"case": case, "overlap": overlap, "irng": irng, "native_irng": native_irng,
                     "idcor": idcor, "arrays": arrays})
    assert offset == len(data)
    total = sum(a["differing_words"] for row in rows for a in row["arrays"].values())
    result = {"cases": cases, "differing_words": total, "rows": rows, "whole_model_proof": False, "port_provenance":port_provenance()}
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mcica": {k:v for k,v in result.items() if k != "rows"}}), flush=True)
    return total


def replay_driver(engine, path, receipt):
    """Replay native driver transformations with MPAS (level, cell) views."""
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import FIELD_NAMES, SURFACE_NAMES, SCALAR_OUTPUTS, shortwave_slot
    from types import SimpleNamespace
    records = read_swd(path)
    cases = sorted({key.split("/")[0] for key in records})
    rows = []
    schemes = {1: "mp_thompson", 2: "mp_wsm6", 3: "mp_thompson_aerosols"}
    controls = ("xtime", "gmt", "julday", "radt", "degrad", "declin", "solcon", "icloud",
                "cldovrlp", "idcor", "o3input", "has_reqc", "has_reqi", "has_reqs")
    for case in cases:
        inputs = {k.split("/", 2)[-1]: v for k, v in records.items() if k.startswith(case + "/in/")}
        expected = {k.split("/", 2)[-1]: v for k, v in records.items() if k.startswith(case + "/out/")}
        nz = expected["rthratensw"].size
        fields = {n: cp.asarray(inputs[n][:nz, None]) for n in FIELD_NAMES}
        for name in ("re_cloud", "re_ice", "re_snow"):
            fields[name] = cp.asarray(inputs["raw_" + name][:nz, None])
        surface = {n: cp.asarray(inputs[n].reshape(1)) for n in SURFACE_NAMES}
        options = {n: inputs[n].item() for n in controls}
        options.update(pin=cp.asarray(inputs["pin"]), o3clim=cp.asarray(inputs["o3clim"][:, None]))
        scheme = schemes[int(inputs["microp_case"])]
        actual = engine.radiation_sw_from_MPAS(fields, surface,
            cp.asarray(inputs["p8w"][:, None]), cp.asarray(inputs["t8w"][:, None]),
            lat_cell=cp.asarray(inputs["lat_cell"]), lon_cell=cp.asarray(inputs["lon_cell"]),
            microp_scheme=scheme, config_microp_re=bool(inputs["config_microp_re"]),
            nwfa=cp.asarray(inputs["nwfa"][:nz, None]), nifa=cp.asarray(inputs["nifa"][:nz, None]),
            aerosol_qv=cp.asarray(inputs["aerosol_qv"][:nz, None]), ht=cp.asarray(inputs["ht"].reshape(1)), **options)
        arrays = {}
        for name, reference in expected.items():
            got = actual[name].get().reshape(reference.shape)
            mask = got.view(np.uint32) != reference.view(np.uint32)
            entry = {"words": int(reference.size), "differing_words": int(mask.sum())}
            if mask.any():
                first = int(np.flatnonzero(mask)[0])
                entry.update(first_index=first, native=float(reference.flat[first]), cuda=float(got.flat[first]))
            arrays[name] = entry
        # Execute the f4 callback and verify its native writeback and halos.
        def halo(a):
            shape = (*a.shape[:-1], 2)
            out = cp.full(shape, -987.25, cp.float32)
            out[..., :1] = a
            return out
        sounding = {n: halo(a) for n, a in fields.items()}
        sounding.update(p8w=halo(cp.asarray(inputs["p8w"][:, None])),
                        t8w=halo(cp.asarray(inputs["t8w"][:, None])), controls={n:inputs[n].item() for n in controls},
                        nwfa=halo(cp.asarray(inputs["nwfa"][:nz, None])),
                        nifa=halo(cp.asarray(inputs["nifa"][:nz, None])),
                        aerosol_qv=halo(cp.asarray(inputs["aerosol_qv"][:nz, None])),
                        ht=halo(cp.asarray(inputs["ht"].reshape(1))))
        diag = {n: cp.full(2, -987.25, cp.float32) for n in SCALAR_OUTPUTS}
        diag.update(sfc_albedo=halo(surface["albedo"]), o3clim=halo(cp.asarray(inputs["o3clim"][:, None])),
                    taod5502d=cp.full(2, -987.25, cp.float32), taod5503d=cp.full((nz,2), -987.25, cp.float32))
        pools = dict(nCellsSolve=1, rrtmg_sw_engine=engine, radiation_sw_input=sounding,
                     mesh=dict(latCell=halo(cp.asarray(inputs["lat_cell"])), lonCell=halo(cp.asarray(inputs["lon_cell"]))),
                     sfc_input={n:halo(surface[k]) for n,k in (("skintemp","tsk"),("snow","snow"),("xice","xice"),("xland","xland"))},
                     diag_physics=diag, tend_physics=dict(rthratensw=cp.full((nz,2), -987.25, cp.float32)),
                     atm_input=dict(pin=cp.asarray(inputs["pin"])))
        shortwave_slot(SimpleNamespace(pools=pools, options=dict(config_microp_scheme=scheme,
                       config_microp_re=bool(inputs["config_microp_re"]))))
        targets = dict(diag) | pools["tend_physics"]
        for name in (*SCALAR_OUTPUTS, "rthratensw", *(["taod5502d", "taod5503d"] if scheme=="mp_thompson_aerosols" else [])):
            reference = expected[name]
            got = targets[name][..., :1].get().reshape(reference.shape)
            arrays["slot/"+name] = {"words": int(reference.size), "differing_words":int((got.view(np.uint32)!=reference.view(np.uint32)).sum())}
            remaining = targets[name][..., 1:].get()
            arrays["halo/"+name] = {"words": int(remaining.size), "differing_words":int((remaining != np.float32(-987.25)).sum())}
        rows.append({"case": case, "microp_scheme": scheme,
                     "config_microp_re": bool(inputs["config_microp_re"]), "arrays": arrays})
    total = sum(a["differing_words"] for row in rows for a in row["arrays"].values())
    result = {"cases": len(cases), "differing_words": total, "rows": rows, "whole_model_proof": False, "port_provenance":port_provenance()}
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)
    return total


def replay_cloud(engine, path, receipt, *, liqflag=1):
    import cupy as cp
    data = Path(path).read_bytes()
    nl, ng, cases = struct.unpack(">iii", data[:12])
    offset, rows = 12, []
    def floats(count):
        nonlocal offset
        x = np.frombuffer(data, dtype=">f4", count=count, offset=offset).astype(np.float32)
        offset += count * 4
        return x
    for index in range(cases):
        inflag, iceflag = struct.unpack_from(">ii", data, offset)
        offset += 8
        cf, ci, cl, cs = [cp.asarray(floats(nl * ng)) for _ in range(4)]
        rei, rel, res = [cp.asarray(floats(nl)) for _ in range(3)]
        tau, ssa, asy, fsf = [cp.asarray(floats(nl * ng)) for _ in range(4)]
        expected = [floats(nl * ng) for _ in range(4)]
        taor, error = cp.empty(nl * ng, dtype=cp.float32), cp.zeros(1, dtype=cp.int32)
        engine._launch("rsw_cldprmc_b", nl * ng,
            tuple(np.int32(v) for v in (1, nl, inflag, iceflag, liqflag)) +
            (engine.tab_gpu, engine.ngb_gpu, cf, ci, cl, cs, rei, rel, res, taor, tau, ssa, asy, fsf, error))
        arrays = {}
        for name, got, ref in zip(("taor", "tau", "ssa", "asy"), (taor, tau, ssa, asy), expected):
            got = got.get()
            mask = got.view(np.uint32) != ref.view(np.uint32)
            entry = {"words": int(ref.size), "differing_words": int(mask.sum())}
            if mask.any():
                first = int(np.flatnonzero(mask)[0])
                entry.update(first_index=first, native=float(ref[first]), cuda=float(got[first]))
            arrays[name] = entry
        rows.append({"case": index, "inflag": inflag, "iceflag": iceflag, "liqflag": liqflag,
                     "error": int(error.get()[0]), "arrays": arrays})
    assert offset == len(data)
    total = sum(a["differing_words"] for row in rows for a in row["arrays"].values()) + sum(row["error"] != 0 for row in rows)
    result = {"cases": cases, "differing_words": total, "rows": rows, "whole_model_proof": False, "port_provenance":port_provenance()}
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)
    return total + sum(row["error"] != 0 for row in rows)


def replay_aerosols(engine, directory, receipt):
    import cupy as cp
    directory = Path(directory)
    inputs = {k: cp.asarray(v) for k, v in dict(np.load(directory / "aerosol_inputs.npz")).items()}
    nc, nz = inputs["p"].shape
    def load_native(name, diagnostics=False):
        data = (directory / name).read_bytes()
        offset = 0
        result = {}
        for key, shape, dtype in (
            ("tauaer", (nc, nz, 14), ">f4"), ("ssaaer", (nc, nz, 14), ">f4"),
            ("asyaer", (nc, nz, 14), ">f4"), ("taod5503d", (nc, nz), ">f4"),
            ("taod5502d", (nc,), ">f4"), ("taer_type", (nc,), ">i4")):
            size = int(np.prod(shape))
            arr = np.frombuffer(data, dtype=dtype, count=size, offset=offset).astype(
                np.float32 if dtype == ">f4" else np.int32).reshape(shape, order="F")
            offset += size * 4
            if key in ("tauaer", "ssaaer", "asyaer"):
                arr = arr.transpose(0, 2, 1).copy()
            result[key] = arr
        if diagnostics:
            for key in ("angexp2d", "aerssa2d", "aerasy2d"):
                result[key] = np.frombuffer(data, dtype=">f4", count=nc, offset=offset).astype(np.float32)
                offset += nc * 4
        assert offset == len(data)
        return result
    def compare(label, actual, expected):
        arrays = {}
        for key, reference in expected.items():
            got = actual[key].get()
            if key in ("tauaer", "ssaaer", "asyaer"):
                got = got[:, :, :nz]
            mismatch = got.view(np.uint32) != reference.view(np.uint32)
            count = int(np.count_nonzero(mismatch))
            result = {"words": int(reference.size), "differing_words": count}
            if count:
                pos = int(np.flatnonzero(mismatch)[0])
                result.update(first_index=pos, native=float(reference.flat[pos]), cuda=float(got.flat[pos]))
            arrays[key] = result
        return {"case": label, "arrays": arrays}
    expected = load_native("aer3_mpas_out.bin")
    actual = engine.thompson_aerosols(inputs["p"], inputs["t"], inputs["qv"], inputs["dz"], inputs["nwfa"], inputs["nifa"], inputs["xland"], inputs["ht"])
    rows = [compare("MPAS-driver", actual, expected)]
    cases = json.loads((directory / "aerosol_cases.json").read_text())
    for index, case in enumerate(cases):
        has3d, aod, ang, ssa, asy, typ, aodv, angv, ssav, asyv = case["control"]
        expected = load_native(case["output"], diagnostics=True)
        i = cp.arange(1, nc + 1, dtype=cp.int32)
        actual = engine.aerosol_optics(inputs["p"], inputs["t"], inputs["qv"], inputs["dz"], inputs["ht"],
            cp.full(nc, typ, dtype=cp.int32), aod5502d=cp.asarray(expected["taod5502d"]),
            aod5503d=cp.asarray(expected["taod5503d"]) if has3d else None,
            angexp2d=(i % 7).astype(cp.float32) * cp.float32(.5) - cp.float32(.5),
            aerssa2d=(i % 5).astype(cp.float32) * cp.float32(.25),
            aerasy2d=(i % 5).astype(cp.float32) * cp.float32(.5) - cp.float32(1.),
            aod_opt=aod, angexp_opt=ang, ssa_opt=ssa, asy_opt=asy,
            aod_value=aodv, angexp_value=angv, ssa_value=ssav, asy_value=asyv)
        expected = {k: v for k, v in expected.items() if k not in ("taod5503d", "taer_type")}
        expected["aod5502d"] = expected.pop("taod5502d")
        row = compare(index, actual, expected)
        row["control"] = case["control"]
        rows.append(row)
    total = sum(a["differing_words"] for r in rows for a in r["arrays"].values())
    result = {"authority": "pristine MPAS-A v8.4.2 aerosol module and driver slice",
              "cases": len(rows), "columns": nc, "levels": nz,
              "differing_words": total, "rows": rows, "whole_model_proof": False, "port_provenance":port_provenance()}
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "batch_rows")}))
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tables"); p.add_argument("columns"); p.add_argument("receipt")
    p.add_argument("--aerosol-directory"); p.add_argument("--aerosol-receipt")
    args = p.parse_args()
    total = replay(args.tables, args.columns, args.receipt)
    if args.aerosol_directory:
        from gpuwm.core.rrtmg_sw import tables_from_dump
        from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
        total += replay_aerosols(MPASShortwave(tables_from_dump(read_swd(args.tables))),
                                 args.aerosol_directory, args.aerosol_receipt)
    raise SystemExit(bool(total))
