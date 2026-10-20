//! Test-only MPAS oracle transport and KF argument packing. All weather-field
//! decoding/encoding and array reshaping happen here, never in Python.
//! Oracle stream v1 is f1's ORCL0001 format. Driver ABI:
//! MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:524-552.
use std::{
    env, fs, io,
    path::{Path, PathBuf},
};
const FIXED: usize = 8 + 21 * 4 + 3 * 256;
const FIELDS: [&str; 10] = ["u", "v", "th", "t", "w", "dz8w", "pcps", "pi", "rho", "qv"];
const STATE: [&str; 7] = [
    "w0avg", "rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten",
];
const SURFACE: [&str; 7] = [
    "dxcell", "areacell", "raincv", "pratec", "nca", "cutop", "cubot",
];
const PARAMS: [&str; 16] = [
    "dt",
    "cudt",
    "curr_secs",
    "xlv0",
    "xlv1",
    "xls0",
    "xls1",
    "cp",
    "r",
    "g",
    "ep1",
    "ep2",
    "svp1",
    "svp2",
    "svp3",
    "svpt0",
];
const FLAGS: [&str; 7] = [
    "stepcu",
    "ktau",
    "adapt_step_flag",
    "warm_rain",
    "f_qr",
    "f_qi",
    "f_qs",
];
#[derive(Clone)]
struct Record {
    head: Vec<u8>,
    h: [i32; 21],
    data: Vec<u8>,
}
fn bad(s: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s)
}
fn read(p: &Path) -> io::Result<Record> {
    let bytes = fs::read(p)?;
    if bytes.len() < FIXED || &bytes[..8] != b"ORCL0001" {
        return Err(bad("invalid ORCL0001 header"));
    }
    let mut h = [0; 21];
    for (i, v) in h.iter_mut().enumerate() {
        *v = i32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap());
    }
    if !(0..=5).contains(&h[2])
        || h[1] <= 0
        || h[3..8].iter().any(|v| *v < 0)
        || h[16] < 0
        || h[19] != 256
        || h[20] != 0
    {
        return Err(bad("invalid oracle dimensions"));
    }
    let count = h[3..8]
        .iter()
        .try_fold(1usize, |a, v| a.checked_mul(*v as usize))
        .ok_or_else(|| bad("size overflow"))?;
    let offset = FIXED
        .checked_add((h[16] as usize) * 4)
        .ok_or_else(|| bad("offset overflow"))?;
    let expected = count
        .checked_mul(h[1] as usize)
        .and_then(|n| n.checked_add(offset))
        .ok_or_else(|| bad("size overflow"))?;
    if bytes.len() != expected {
        return Err(bad("truncated/trailing oracle payload"));
    }
    Ok(Record {
        head: bytes[..offset].to_vec(),
        h,
        data: bytes[offset..].to_vec(),
    })
}
fn path(prefix: &Path, phase: &str, name: &str) -> PathBuf {
    PathBuf::from(format!("{}.{}.{}.bin", prefix.display(), phase, name))
}
fn pre(prefix: &Path, name: &str) -> io::Result<Record> {
    read(&path(prefix, "pre", name))
}
fn integer(prefix: &Path, name: &str) -> io::Result<i32> {
    let r = pre(prefix, name)?;
    if r.h[2] != 0 || r.h[1] != 4 {
        return Err(bad("expected scalar word"));
    }
    Ok(i32::from_le_bytes(r.data[..4].try_into().unwrap()))
}
fn scalar(prefix: &Path, name: &str) -> io::Result<Vec<u8>> {
    let r = pre(prefix, name)?;
    if r.h[2] != 0 || r.h[1] != 4 {
        return Err(bad("expected scalar word"));
    }
    Ok(r.data)
}
fn write_record(out: &Path, name: &str, mut r: Record) -> io::Result<()> {
    for (i, v) in r.h.iter().enumerate() {
        r.head[8 + 4 * i..12 + 4 * i].copy_from_slice(&v.to_le_bytes());
    }
    for (start, value) in [(92 + 256, "post"), (92 + 512, name)] {
        r.head[start..start + 256].fill(b' ');
        r.head[start..start + value.len()].copy_from_slice(value.as_bytes());
    }
    let p = path(out, "post", name);
    if p.exists() {
        return Err(bad("output record already exists"));
    }
    r.head.extend(r.data);
    fs::write(p, r.head)
}
struct Tile {
    first: usize,
    nc: usize,
    nz: usize,
}
fn tile(prefix: &Path) -> io::Result<Tile> {
    let pcps = pre(prefix, "pcps")?;
    let its = integer(prefix, "its")?;
    let ite = integer(prefix, "ite")?;
    let nz = integer(prefix, "kte")?;
    if integer(prefix, "kts")? != 1
        || integer(prefix, "jts")? != 1
        || integer(prefix, "jte")? != 1
        || pcps.h[8] != 1
        || pcps.h[9] != 1
        || pcps.h[10] != 1
        || its < 1
        || ite < its
        || ite > pcps.h[3]
        || nz < 1
        || nz > pcps.h[4]
    {
        return Err(bad("capture is outside MPAS's one-row, one-based tile ABI"));
    }
    Ok(Tile {
        first: (its - 1) as usize,
        nc: (ite - its + 1) as usize,
        nz: nz as usize,
    })
}
fn packed(prefix: &Path, names: &[&str], t: &Tile, vertical: bool) -> io::Result<Vec<u8>> {
    let nlev = if vertical { t.nz + 1 } else { 1 };
    let mut result = vec![0; names.len() * nlev * t.nc * 4];
    for (a, name) in names.iter().enumerate() {
        let r = if (*name == "cutop" || *name == "cubot") && !path(prefix, "pre", name).exists() {
            let mut r = pre(prefix, "dxcell")?;
            r.data.fill(0);
            r
        } else {
            pre(prefix, name)?
        };
        if r.h[0] != 1 || r.h[1] != 4 || (r.h[3] as usize) < t.first + t.nc {
            return Err(bad("invalid real32 KF array"));
        }
        let levels = if vertical {
            (r.h[4] as usize).min(nlev)
        } else {
            1
        };
        for k in 0..levels {
            for c in 0..t.nc {
                let i = (k * r.h[3] as usize + t.first + c) * 4;
                let o = ((a * nlev + k) * t.nc + c) * 4;
                result[o..o + 4].copy_from_slice(&r.data[i..i + 4]);
            }
        }
    }
    Ok(result)
}
fn pack(prefix: &Path, dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let t = tile(prefix)?;
    for (name, fields, vertical) in [
        ("fields", &FIELDS[..], true),
        ("state", &STATE[..], true),
        ("surface", &SURFACE[..], false),
    ] {
        fs::write(
            dir.join(format!("{name}.bin")),
            packed(prefix, fields, &t, vertical)?,
        )?;
    }
    let mut params = vec![];
    for n in PARAMS {
        params.extend(scalar(prefix, n)?);
    }
    fs::write(dir.join("params.bin"), params)?;
    let mut flags = vec![];
    for n in FLAGS {
        let v = integer(prefix, n)?;
        flags.extend(v.to_le_bytes());
    }
    fs::write(dir.join("flags.bin"), flags)?;
    let a = pre(prefix, "cu_act_flag")?;
    let active = (t.first..t.first + t.nc)
        .map(|i| u8::from(a.data[4 * i..4 * i + 4] != [0; 4]))
        .collect::<Vec<_>>();
    fs::write(dir.join("active.bin"), active)?;
    let mut tables = vec![];
    for n in ["ttab", "qstab", "the0k", "alu", "rdpr", "rdthk", "plutop"] {
        tables.extend(pre(prefix, &format!("state.module_cu_kfeta.{n}"))?.data);
    }
    if tables.len() != 110423 * 4 {
        return Err(bad("incorrect native LUTAB storage"));
    }
    fs::write(dir.join("tables.bin"), tables)?;
    println!(
        "{{\"nc\":{},\"nz\":{},\"first\":{},\"warm_rain_word\":{},\"ice_enabled\":{}}}",
        t.nc,
        t.nz,
        t.first,
        integer(prefix, "warm_rain")?,
        integer(prefix, "f_qi")? != 0
    );
    Ok(())
}
fn unpack(prefix: &Path, dir: &Path, out: &Path) -> io::Result<()> {
    let t = tile(prefix)?;
    for (file, names, vertical) in [
        ("state", &STATE[..], true),
        ("surface", &SURFACE[..], false),
    ] {
        let b = fs::read(dir.join(format!("{file}-out.bin")))?;
        let nl = if vertical { t.nz + 1 } else { 1 };
        if b.len() != names.len() * nl * t.nc * 4 {
            return Err(bad("incorrect packed output extent"));
        }
        for (a, name) in names.iter().enumerate() {
            if file == "surface" && a < 2 {
                continue;
            }
            let mut r = if (*name == "cutop" || *name == "cubot") && !path(prefix, "pre", name).exists() {
                let mut r = pre(prefix, "dxcell")?;
                r.data.fill(0);
                r
            } else {
                pre(prefix, name)?
            };
            let levels = if vertical {
                (r.h[4] as usize).min(nl)
            } else {
                1
            };
            for k in 0..levels {
                for c in 0..t.nc {
                    let i = (k * r.h[3] as usize + t.first + c) * 4;
                    let o = ((a * nl + k) * t.nc + c) * 4;
                    r.data[i..i + 4].copy_from_slice(&b[o..o + 4]);
                }
            }
            write_record(out, name, r)?;
        }
    }
    let mut r = pre(prefix, "cu_act_flag")?;
    let active = fs::read(dir.join("active-out.bin"))?;
    if active.len() != t.nc {
        return Err(bad("incorrect logical output extent"));
    }
    for (c, a) in active.iter().enumerate() {
        r.data[(t.first + c) * 4..(t.first + c + 1) * 4]
            .copy_from_slice(&i32::from(*a != 0).to_le_bytes());
    }
    write_record(out, "cu_act_flag", r)?;
    for n in [
        "f_qr",
        "f_qi",
        "f_qs",
        "rthcuten.present",
        "rqvcuten.present",
        "rqccuten.present",
        "rqrcuten.present",
        "rqicuten.present",
        "rqscuten.present",
        "f_qr.present",
        "f_qi.present",
        "f_qs.present",
    ] {
        write_record(out, n, pre(prefix, n)?)?;
    }
    // Numerical module tables and DATA constants are unchanged by CPS. lastdt
    // is the only mutable saved scalar: module_cu_kfeta.F:195-196,228.
    let parent = prefix.parent().ok_or_else(|| bad("prefix has no parent"))?;
    let stem = format!("{}.pre.", prefix.file_name().unwrap().to_string_lossy());
    for entry in fs::read_dir(parent)? {
        let entry = entry?;
        let filename = entry.file_name();
        let filename = filename.to_string_lossy();
        let Some(name) = filename
            .strip_prefix(&stem)
            .and_then(|v| v.strip_suffix(".bin"))
        else {
            continue;
        };
        if !name.starts_with("state.") && !name.starts_with("local.") {
            continue;
        }
        let mut r = pre(prefix, name)?;
        if name == "local.module_cu_kfeta.kf_eta_cps.lastdt" {
            r.data = scalar(prefix, "dt")?;
        }
        write_record(out, name, r)?;
    }
    Ok(())
}
fn main() -> io::Result<()> {
    let a = env::args().collect::<Vec<_>>();
    if a.len() < 4 {
        return Err(bad(
            "usage: dump_io pack PREFIX DIRECTORY | unpack PREFIX DIRECTORY OUTPUT_PREFIX",
        ));
    }
    match a[1].as_str() {
        "pack" => pack(Path::new(&a[2]), Path::new(&a[3])),
        "unpack" if a.len() == 5 => unpack(Path::new(&a[2]), Path::new(&a[3]), Path::new(&a[4])),
        _ => Err(bad("invalid operation")),
    }
}
