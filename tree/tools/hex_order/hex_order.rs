//! Reversible entity permutation. Neighbor slots and float bytes never change.
//! Standalone: rustc -O hex_order.rs -o hex_order -l netcdf
//!
//! SPEED WITHOUT NEW BYTES (prep-fast, 2026-10-13). `--apply` on a 14 GB init
//! took 335 s at 21% of ONE core: it copied the file, re-read every permuted
//! variable from disk, permuted it element by element with a div/mod per axis,
//! and wrote it back, and every header change (`regional_garbage_donor_cell`
//! in `--apply`, `cell_order_original` in `--record-donor`) let libnetcdf
//! shift the whole data section on disk again. Now the file is read ONCE into
//! memory on every core, libnetcdf runs the very same call sequence on that
//! memory image (`nc_open_memio`: the same nc3 layout code, header growth and
//! fill; only the I/O layer differs), the permutation copies contiguous
//! blocks on every core, and the result is written ONCE. The output file is
//! byte-for-byte the file the disk path writes -- `--compare` and a whole-file
//! sha256 are the proof, and `HEX_ORDER_IO=file` keeps the disk path for
//! re-proving it. A file too large for the RAM the box has free falls back to
//! the disk path (named breakage: an out-of-memory kill of the prep).
use std::{
    ffi::{c_char, c_int, c_void, CStr, CString},
    fs,
    io::{Read, Write},
    path::Path,
};

#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_inq(id: c_int, nd: *mut c_int, nv: *mut c_int, na: *mut c_int, un: *mut c_int) -> c_int;
    fn nc_inq_dim(id: c_int, dim: c_int, name: *mut c_char, len: *mut usize) -> c_int;
    fn nc_inq_var(
        id: c_int,
        var: c_int,
        name: *mut c_char,
        typ: *mut c_int,
        nd: *mut c_int,
        dims: *mut c_int,
        na: *mut c_int,
    ) -> c_int;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
    fn nc_get_var(id: c_int, var: c_int, data: *mut c_void) -> c_int;
    fn nc_get_var_double(id: c_int, var: c_int, data: *mut f64) -> c_int;
    fn nc_put_var(id: c_int, var: c_int, data: *const c_void) -> c_int;
    fn nc_get_att_int(id: c_int, var: c_int, name: *const c_char, value: *mut c_int) -> c_int;
    fn nc_put_att_int(
        id: c_int,
        var: c_int,
        name: *const c_char,
        typ: c_int,
        len: usize,
        value: *const c_int,
    ) -> c_int;
    fn nc_redef(id: c_int) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
    fn nc_def_var(
        id: c_int,
        name: *const c_char,
        typ: c_int,
        nd: c_int,
        dims: *const c_int,
        var: *mut c_int,
    ) -> c_int;
    fn nc_open_memio(path: *const c_char, mode: c_int, info: *mut NcMemio, id: *mut c_int) -> c_int;
    fn nc_close_memio(id: c_int, info: *mut NcMemio) -> c_int;
}
unsafe extern "C" {
    fn malloc(n: usize) -> *mut c_void;
    fn free(p: *mut c_void);
}
/// `NC_memio` from netcdf_mem.h.
#[repr(C)]
struct NcMemio {
    size: usize,
    memory: *mut c_void,
    flags: c_int,
}
type R<T> = Result<T, String>;
fn check(code: i32) -> R<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(unsafe { CStr::from_ptr(nc_strerror(code)) }
            .to_string_lossy()
            .into_owned())
    }
}
struct Nc(i32);
impl Nc {
    fn open(p: &str, mode: i32) -> R<Self> {
        let p = CString::new(p).map_err(|e| e.to_string())?;
        let mut id = 0;
        check(unsafe { nc_open(p.as_ptr(), mode, &mut id) })?;
        Ok(Self(id))
    }
    fn dims(&self) -> R<Vec<(String, usize)>> {
        let mut n = 0;
        check(unsafe {
            nc_inq(
                self.0,
                &mut n,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        })?;
        (0..n)
            .map(|i| {
                let mut b = [0i8; 257];
                let mut l = 0;
                check(unsafe { nc_inq_dim(self.0, i, b.as_mut_ptr(), &mut l) })?;
                Ok((
                    unsafe { CStr::from_ptr(b.as_ptr()) }
                        .to_string_lossy()
                        .into_owned(),
                    l,
                ))
            })
            .collect()
    }
    fn doubles(&self, name: &str, count: usize) -> R<Vec<f64>> {
        let n = CString::new(name).unwrap();
        let mut var = 0;
        check(unsafe { nc_inq_varid(self.0, n.as_ptr(), &mut var) })?;
        let (_, _, dims) = self.var(var)?;
        let dd = self.dims()?;
        if dims.iter().map(|d| dd[*d].1).product::<usize>() != count {
            return Err(format!("{name}: coordinate length mismatch"));
        }
        let mut a = vec![0.0; count];
        check(unsafe { nc_get_var_double(self.0, var, a.as_mut_ptr()) })?;
        Ok(a)
    }
    fn var(&self, var: i32) -> R<(String, i32, Vec<usize>)> {
        let mut b = [0i8; 257];
        let mut t = 0;
        let mut n = 0;
        let mut ds = [0i32; 1024];
        check(unsafe {
            nc_inq_var(
                self.0,
                var,
                b.as_mut_ptr(),
                &mut t,
                &mut n,
                ds.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;
        Ok((
            unsafe { CStr::from_ptr(b.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            t,
            ds[..n as usize].iter().map(|v| *v as usize).collect(),
        ))
    }
}
impl Drop for Nc {
    fn drop(&mut self) {
        if self.0 >= 0 {
            unsafe { nc_close(self.0) };
        }
    }
}

fn spread(v: u64) -> u64 {
    let mut out = 0;
    for bit in 0..21 {
        out |= ((v >> bit) & 1) << (3 * bit);
    }
    out
}
fn morton(xyz: &[Vec<f64>; 3]) -> R<Vec<usize>> {
    let n = xyz[0].len();
    if xyz
        .iter()
        .any(|v| v.len() != n || v.iter().any(|x| !x.is_finite()))
    {
        return Err("invalid Morton coordinates".into());
    }
    let bounds: Vec<_> = xyz
        .iter()
        .map(|v| {
            (
                v.iter().copied().fold(f64::INFINITY, f64::min),
                v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        })
        .collect();
    let key = |i: usize| {
        let mut key = 0;
        for a in 0..3 {
            let (lo, hi) = bounds[a];
            let v = if hi == lo {
                0
            } else {
                (((xyz[a][i] - lo) / (hi - lo)) * 2097151.0).round() as u64
            };
            key |= spread(v) << a;
        }
        key
    };
    // Keys computed once each, on every core, then sorted. `(key, index)`
    // pairs are all distinct, so this is the one order the comparator sort
    // (which recomputed both keys on every comparison) produced.
    let keys = par_map(n, |i| key(i));
    let mut pairs: Vec<(u64, usize)> = keys.into_iter().zip(0..n).collect();
    pairs.sort_unstable();
    Ok(pairs.into_iter().map(|(_, i)| i).collect())
}
fn inverse(order: &[usize]) -> R<Vec<usize>> {
    let mut inv = vec![usize::MAX; order.len()];
    for (i, &j) in order.iter().enumerate() {
        if j >= inv.len() || inv[j] != usize::MAX {
            return Err("map is not a permutation".into());
        }
        inv[j] = i;
    }
    Ok(inv)
}
fn entity(name: &str) -> Option<usize> {
    match name {
        "nCells" => Some(0),
        "nEdges" => Some(1),
        "nVertices" => Some(2),
        _ => None,
    }
}
fn references(name: &str) -> Option<usize> {
    match name {
        "cellsOnCell" | "cellsOnEdge" | "cellsOnVertex" | "advCellsForEdge"
        | "adv_cells_for_edge" => Some(0),
        "edgesOnCell" | "edgesOnEdge" | "edgesOnVertex" => Some(1),
        "verticesOnCell" | "verticesOnEdge" => Some(2),
        _ => None,
    }
}
fn width(t: i32) -> R<usize> {
    match t {
        1 | 2 | 7 => Ok(1),
        3 | 8 => Ok(2),
        4 | 5 | 9 => Ok(4),
        6 | 10 | 11 => Ok(8),
        _ => Err(format!(
            "unsupported NetCDF type {t}; variable data would be lost"
        )),
    }
}
/// Worker threads: `HEX_ORDER_THREADS`, else every core the process may use.
fn threads() -> usize {
    std::env::var("HEX_ORDER_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&t: &usize| t > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}
/// `(0..n).map(f)` on every core, results in index order.
fn par_map<T: Send + Default + Clone, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let mut out = vec![T::default(); n];
    let t = threads().min(n.max(1));
    let chunk = n.div_ceil(t).max(1);
    std::thread::scope(|s| {
        for (c, part) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (k, slot) in part.iter_mut().enumerate() {
                    *slot = f(c * chunk + k);
                }
            });
        }
    });
    out
}
/// The permutation, as contiguous block copies on every core.
///
/// Same mapping as [`gather_reference`]: output element `i` takes input
/// element `j`, where each entity axis index `k` is replaced by `maps[e][k]`.
/// Every axis after the LAST entity axis is unmapped, so the elements under
/// one index of that axis are a contiguous run in both arrays and move as one
/// `copy_from_slice`. A copy cannot change a byte, so this is the reference's
/// output by construction (`block_gather_is_the_reference` checks it anyway).
fn gather(
    raw: &[u8],
    w: usize,
    shape: &[usize],
    axes: &[Option<usize>],
    maps: &[Vec<usize>; 3],
) -> R<Vec<u8>> {
    let total = shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b).ok_or("extent overflow"))?;
    if raw.len() != total.checked_mul(w).ok_or("byte extent overflow")? {
        return Err("raw byte extent mismatch".into());
    }
    for (a, e) in axes.iter().enumerate() {
        if let Some(e) = e {
            if shape[a] != maps[*e].len() {
                return Err(
                    "entity extent differs from map; sparse boundary fields need a separate id map"
                        .into(),
                );
            }
        }
    }
    let Some(last) = axes.iter().rposition(|a| a.is_some()) else {
        return Ok(raw.to_vec());
    };
    let block = shape[last + 1..].iter().product::<usize>() * w;
    let outer = &shape[..=last];
    let blocks: usize = outer.iter().product();
    if blocks == 0 || block == 0 {
        return Ok(raw.to_vec());
    }
    let mut stride = vec![1usize; outer.len()];
    for a in (0..outer.len().saturating_sub(1)).rev() {
        stride[a] = stride[a + 1] * outer[a + 1];
    }
    let mut out = vec![0u8; raw.len()];
    let t = threads().min(blocks);
    let per = blocks.div_ceil(t);
    std::thread::scope(|s| {
        for (c, part) in out.chunks_mut(per * block).enumerate() {
            let (stride, outer) = (&stride, outer);
            s.spawn(move || {
                for (k, dst) in part.chunks_exact_mut(block).enumerate() {
                    let b = c * per + k;
                    let mut j = 0;
                    for a in 0..outer.len() {
                        let idx = (b / stride[a]) % outer[a];
                        j += axes[a].map_or(idx, |e| maps[e][idx]) * stride[a];
                    }
                    dst.copy_from_slice(&raw[j * block..(j + 1) * block]);
                }
            });
        }
    });
    Ok(out)
}
#[cfg_attr(not(test), allow(dead_code))]
fn gather_reference(
    raw: &[u8],
    w: usize,
    shape: &[usize],
    axes: &[Option<usize>],
    maps: &[Vec<usize>; 3],
) -> R<Vec<u8>> {
    let total = shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b).ok_or("extent overflow"))?;
    if raw.len() != total.checked_mul(w).ok_or("byte extent overflow")? {
        return Err("raw byte extent mismatch".into());
    }
    let mut stride = vec![1; shape.len()];
    for a in (0..shape.len().saturating_sub(1)).rev() {
        stride[a] = stride[a + 1] * shape[a + 1];
    }
    for (a, e) in axes.iter().enumerate() {
        if let Some(e) = e {
            if shape[a] != maps[*e].len() {
                return Err(
                    "entity extent differs from map; sparse boundary fields need a separate id map"
                        .into(),
                );
            }
        }
    }
    let mut out = vec![0; raw.len()];
    for i in 0..total {
        let mut j = 0;
        for a in 0..shape.len() {
            let k = (i / stride[a]) % shape[a];
            j += axes[a].map_or(k, |e| maps[e][k]) * stride[a];
        }
        out[i * w..(i + 1) * w].copy_from_slice(&raw[j * w..(j + 1) * w]);
    }
    Ok(out)
}
/// [`remap_serial`] on every core. The first failing chunk in array order
/// reports, and inside it the first failing element: the element the serial
/// loop would have stopped on.
fn remap(raw: &mut [u8], typ: i32, inv: &[usize]) -> R<()> {
    if typ != 4 {
        return Err("connectivity must be NetCDF int32; refusing silent conversion".into());
    }
    let n = raw.len() / 4;
    let t = threads().min(n.max(1));
    let per = n.div_ceil(t).max(1) * 4;
    let results: Vec<R<()>> = std::thread::scope(|s| {
        let hs: Vec<_> = raw
            .chunks_mut(per)
            .map(|part| s.spawn(move || remap_serial(part, typ, inv)))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    results.into_iter().collect()
}
fn remap_serial(raw: &mut [u8], typ: i32, inv: &[usize]) -> R<()> {
    if typ != 4 {
        return Err("connectivity must be NetCDF int32; refusing silent conversion".into());
    }
    for b in raw.chunks_exact_mut(4) {
        let old = i32::from_ne_bytes(b.try_into().unwrap());
        if old <= 0 {
            continue;
        }
        let i = old as usize - 1;
        if i >= inv.len() {
            return Err(format!("connectivity id {old} is outside map"));
        }
        let v = i32::try_from(inv[i] + 1).map_err(|_| "id overflow")?;
        b.copy_from_slice(&v.to_ne_bytes());
    }
    Ok(())
}
fn write_map(path: &str, maps: &[Vec<usize>; 3]) -> R<()> {
    if Path::new(path).exists() {
        return Err("map output already exists".into());
    }
    let mut f = fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(b"HEXORD01").map_err(|e| e.to_string())?;
    for m in maps {
        f.write_all(&(m.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for i in m {
            f.write_all(&(*i as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
fn read_map(path: &str) -> R<[Vec<usize>; 3]> {
    let mut f = std::io::BufReader::new(fs::File::open(path).map_err(|e| e.to_string())?);
    let mut magic = [0; 8];
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"HEXORD01" {
        return Err("invalid map magic".into());
    }
    let mut maps: [Vec<usize>; 3] = Default::default();
    for m in &mut maps {
        let mut b = [0; 8];
        f.read_exact(&mut b).map_err(|e| e.to_string())?;
        let n = usize::try_from(u64::from_le_bytes(b)).map_err(|_| "extent overflow")?;
        if n == 0 {
            return Err("empty mesh entity map".into());
        }
        for _ in 0..n {
            f.read_exact(&mut b).map_err(|e| e.to_string())?;
            m.push(usize::try_from(u64::from_le_bytes(b)).map_err(|_| "id overflow")?);
        }
        inverse(m)?;
    }
    let mut tail = [0; 1];
    if f.read(&mut tail).map_err(|e| e.to_string())? != 0 {
        return Err("trailing map bytes".into());
    }
    Ok(maps)
}
fn apply(map: &str, input: &str, output: &str, undo: bool) -> R<()> {
    if Path::new(output).exists() {
        return Err("output already exists; refusing overwrite".into());
    }
    let mut maps = read_map(map)?;
    if undo {
        for m in &mut maps {
            *m = inverse(m)?;
        }
    }
    let inv: [Vec<usize>; 3] = [inverse(&maps[0])?, inverse(&maps[1])?, inverse(&maps[2])?];
    let src = Nc::open(input, 0)?;
    let dims = src.dims()?;
    // Validate every variable before creating an output.
    let mut nv = 0;
    check(unsafe {
        nc_inq(
            src.0,
            std::ptr::null_mut(),
            &mut nv,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    })?;
    for var in 0..nv {
        let (name, t, ds) = src.var(var)?;
        width(t)?;
        for d in ds {
            if let Some(e) = entity(&dims[d].0) {
                if dims[d].1 != maps[e].len() {
                    return Err(format!(
                        "{name}: {} extent {} differs from map {}",
                        dims[d].0,
                        dims[d].1,
                        maps[e].len()
                    ));
                }
            }
        }
        if references(&name).is_some() && t != 4 {
            return Err(format!("{name}: connectivity type {t} is not int32"));
        }
    }
    let mem: Option<MemNc> = if use_memory(input)? {
        Some(MemNc::open(input, output)?)
    } else {
        fs::copy(input, output).map_err(|e| e.to_string())?;
        None
    };
    let disk: Option<Nc> = if mem.is_none() {
        Some(Nc::open(output, 1)?)
    } else {
        None
    };
    let dst: &Nc = match (&mem, &disk) {
        (Some(m), _) => &m.nc,
        (None, Some(d)) => d,
        (None, None) => unreachable!("one of the two paths opened the output"),
    };
    // Data is read from the handle it is written back to. Each permuted
    // variable is read before it is written and no write touches another
    // variable's bytes, so this reads exactly what the copy of `input` held.
    let data: &Nc = dst;
    let mut changed = 0;
    for var in 0..nv {
        let (name, t, ds) = src.var(var)?;
        let axes: Vec<_> = ds.iter().map(|d| entity(&dims[*d].0)).collect();
        let reference = references(&name);
        if axes.iter().all(|a| a.is_none()) && reference.is_none() {
            continue;
        }
        let shape: Vec<_> = ds.iter().map(|d| dims[*d].1).collect();
        let w = width(t)?;
        let n = shape
            .iter()
            .try_fold(w, |a, b| a.checked_mul(*b).ok_or("byte extent overflow"))?;
        // u64 storage provides sufficient alignment for all supported atomic types.
        let mut storage = vec![0u64; (n + 7) / 8];
        check(unsafe { nc_get_var(data.0, var, storage.as_mut_ptr().cast()) })?;
        let raw = unsafe { std::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), n) };
        let mut out = gather(raw, w, &shape, &axes, &maps)?;
        drop(storage);
        if let Some(e) = reference {
            remap(&mut out, t, &inv[e])?;
        }
        check(unsafe { nc_put_var(dst.0, var, out.as_ptr().cast()) })?;
        changed += 1;
    }
    if dims.iter().any(|(name, _)| name == "nCells") {
        let name = CString::new("regional_garbage_donor_cell").unwrap();
        let mut donor = i32::try_from(maps[0].len() - 1).map_err(|_| "donor overflow")?;
        let status = unsafe { nc_get_att_int(src.0, -1, name.as_ptr(), &mut donor) };
        if status != 0 && status != -43 {
            check(status)?;
        }
        if donor < 0 || donor as usize >= inv[0].len() {
            return Err("regional donor is outside the real cell extent".into());
        }
        put_donor(
            dst,
            i32::try_from(inv[0][donor as usize]).map_err(|_| "donor overflow")?,
        )?;
    }
    if let Some(m) = &mem {
        m.finish_new(output)?;
    }
    println!("{{\"variables_permuted\":{changed},\"inverse\":{undo}}}");
    Ok(())
}

/// Should this file be worked as a memory image? `HEX_ORDER_IO=file|memory`
/// decides when set; otherwise the image is used when the box has room for
/// it twice over (libnetcdf may grow it by reallocation when the header
/// grows) with a quarter of the free memory to spare.
fn use_memory(path: &str) -> R<bool> {
    match std::env::var("HEX_ORDER_IO").as_deref() {
        Ok("file") => return Ok(false),
        Ok("memory") => return Ok(true),
        Ok(other) if !other.is_empty() => {
            return Err(format!("HEX_ORDER_IO={other} is not a mode; it is file or memory"))
        }
        _ => {}
    }
    let size = fs::metadata(path).map_err(|e| e.to_string())?.len();
    let avail_kib = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        })
        .unwrap_or(0);
    Ok(size.saturating_mul(2) <= avail_kib.saturating_mul(1024) / 4 * 3)
}

/// A netCDF file opened from a memory image of a file on disk.
///
/// The image is read with every core (positional reads into one malloc'd
/// buffer, which libnetcdf owns from `nc_open_memio` on and may reallocate),
/// and written back with every core once the edits are done.
struct MemNc {
    nc: Nc,
    closed: std::cell::Cell<bool>,
}
impl MemNc {
    fn open(path: &str, name: &str) -> R<Self> {
        use std::os::unix::fs::FileExt;
        let f = fs::File::open(path).map_err(|e| e.to_string())?;
        let size = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        let memory = unsafe { malloc(size.max(1)) };
        if memory.is_null() {
            return Err(format!("{path}: cannot allocate a {size}-byte image"));
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(memory.cast::<u8>(), size) };
        let piece = 64 << 20;
        let next = std::sync::atomic::AtomicUsize::new(0);
        let failed: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        let base = buf.as_mut_ptr() as usize;
        std::thread::scope(|s| {
            for _ in 0..threads().min(size.div_ceil(piece).max(1)) {
                s.spawn(|| loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let lo = k * piece;
                    if lo >= size {
                        break;
                    }
                    let hi = (lo + piece).min(size);
                    // Disjoint ranges of one buffer: each index is claimed once.
                    let dst = unsafe { std::slice::from_raw_parts_mut((base + lo) as *mut u8, hi - lo) };
                    if let Err(e) = f.read_exact_at(dst, lo as u64) {
                        *failed.lock().unwrap() = Some(format!("{path}: {e}"));
                        break;
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().unwrap() {
            unsafe { free(memory) };
            return Err(e);
        }
        let mut info = NcMemio {
            size,
            memory,
            flags: 0,
        };
        let cname = CString::new(name).map_err(|e| e.to_string())?;
        let mut id = 0;
        let status = unsafe { nc_open_memio(cname.as_ptr(), 1, &mut info, &mut id) };
        if status != 0 {
            unsafe { free(memory) };
            check(status)?;
        }
        Ok(MemNc {
            nc: Nc(id),
            closed: std::cell::Cell::new(false),
        })
    }
    /// Close the image and hand back its final bytes (malloc'd, ours to free).
    fn close(&self) -> R<(*mut c_void, usize)> {
        let mut info = NcMemio {
            size: 0,
            memory: std::ptr::null_mut(),
            flags: 0,
        };
        self.closed.set(true);
        check(unsafe { nc_close_memio(self.nc.0, &mut info) })?;
        Ok((info.memory, info.size))
    }
    /// Close and write the image as a NEW file at `path`.
    fn finish_new(&self, path: &str) -> R<()> {
        let f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("{path}: {e}"))?;
        self.finish_into(f, path)
    }
    /// Close and write the image over the file at `path`, in place.
    fn finish_over(&self, path: &str) -> R<()> {
        let f = fs::OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(|e| format!("{path}: {e}"))?;
        self.finish_into(f, path)
    }
    fn finish_into(&self, f: fs::File, path: &str) -> R<()> {
        use std::os::unix::fs::FileExt;
        let (memory, size) = self.close()?;
        if memory.is_null() && size > 0 {
            return Err(format!("{path}: libnetcdf returned no image"));
        }
        let result = (|| {
            f.set_len(size as u64).map_err(|e| format!("{path}: {e}"))?;
            if size == 0 {
                return Ok(());
            }
            let buf = unsafe { std::slice::from_raw_parts(memory.cast::<u8>(), size) };
            let piece = 64 << 20;
            let next = std::sync::atomic::AtomicUsize::new(0);
            let failed: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
            std::thread::scope(|s| {
                for _ in 0..threads().min(size.div_ceil(piece).max(1)) {
                    s.spawn(|| loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let lo = k * piece;
                        if lo >= size {
                            break;
                        }
                        let hi = (lo + piece).min(size);
                        if let Err(e) = f.write_all_at(&buf[lo..hi], lo as u64) {
                            *failed.lock().unwrap() = Some(format!("{path}: {e}"));
                            break;
                        }
                    });
                }
            });
            match failed.into_inner().unwrap() {
                Some(e) => Err(e),
                None => Ok(()),
            }
        })();
        unsafe { free(memory) };
        result
    }
}
impl Drop for MemNc {
    fn drop(&mut self) {
        if !self.closed.get() {
            if let Ok((memory, _)) = self.close() {
                unsafe { free(memory) };
            }
        }
        // `nc` must not run nc_close on an id that nc_close_memio released.
        self.nc.0 = -1;
    }
}
fn put_donor(f: &Nc, donor: i32) -> R<()> {
    let n = CString::new("regional_garbage_donor_cell").unwrap();
    check(unsafe { nc_redef(f.0) })?;
    check(unsafe { nc_put_att_int(f.0, -1, n.as_ptr(), 4, 1, &donor) })?;
    check(unsafe { nc_enddef(f.0) })?;
    Ok(())
}
fn record_donor(map: &str, file: &str) -> R<()> {
    let maps = read_map(map)?;
    let donor = inverse(&maps[0])?[maps[0].len() - 1];
    // The same two edits on a memory image: both grow the header, and on a
    // disk file each growth shifts the whole data section in place.
    if use_memory(file)? {
        let mem = MemNc::open(file, file)?;
        put_donor(&mem.nc, i32::try_from(donor).map_err(|_| "donor overflow")?)?;
        put_cell_order(&mem.nc, &maps[0])?;
        mem.finish_over(file)?;
    } else {
        let f = Nc::open(file, 1)?;
        put_donor(&f, i32::try_from(donor).map_err(|_| "donor overflow")?)?;
        put_cell_order(&f, &maps[0])?;
    }
    println!(
        "{{\"original_last_cell\":{},\"regional_garbage_donor_cell\":{donor}}}",
        maps[0].len() - 1
    );
    Ok(())
}
fn put_cell_order(f: &Nc, order: &[usize]) -> R<()> {
    let name = CString::new("cell_order_original").unwrap();
    let mut var = 0;
    // An existing ordinal vector was already gathered with all other fields.
    // Leave that composed mapping intact on repeated or inverse permutations.
    if unsafe { nc_inq_varid(f.0, name.as_ptr(), &mut var) } == 0 {
        return Ok(());
    }
    let dim = f
        .dims()?
        .iter()
        .position(|(name, _)| name == "nCells")
        .ok_or("cell dimension missing")? as i32;
    let values: Vec<i32> = order
        .iter()
        .map(|i| i32::try_from(*i).map_err(|_| "ordinal overflow".to_owned()))
        .collect::<R<_>>()?;
    check(unsafe { nc_redef(f.0) })?;
    check(unsafe { nc_def_var(f.0, name.as_ptr(), 4, 1, &dim, &mut var) })?;
    check(unsafe { nc_enddef(f.0) })?;
    check(unsafe { nc_put_var(f.0, var, values.as_ptr().cast()) })?;
    Ok(())
}
fn compare(left: &str, right: &str) -> R<()> {
    let a = Nc::open(left, 0)?;
    let b = Nc::open(right, 0)?;
    if a.dims()? != b.dims()? {
        return Err("dimension tables differ".into());
    }
    let dd = a.dims()?;
    let mut na = 0;
    let mut nb = 0;
    check(unsafe {
        nc_inq(
            a.0,
            std::ptr::null_mut(),
            &mut na,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    })?;
    check(unsafe {
        nc_inq(
            b.0,
            std::ptr::null_mut(),
            &mut nb,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    })?;
    if na != nb {
        return Err("variable counts differ".into());
    }
    for v in 0..na {
        let va = a.var(v)?;
        let vb = b.var(v)?;
        if va != vb {
            return Err(format!("variable schema differs at {}", va.0));
        }
        let n = va.2.iter().try_fold(width(va.1)?, |n, d| {
            n.checked_mul(dd[*d].1).ok_or("extent overflow")
        })?;
        let mut aa = vec![0u64; (n + 7) / 8];
        let mut bb = vec![0u64; (n + 7) / 8];
        check(unsafe { nc_get_var(a.0, v, aa.as_mut_ptr().cast()) })?;
        check(unsafe { nc_get_var(b.0, v, bb.as_mut_ptr().cast()) })?;
        let ar = unsafe { std::slice::from_raw_parts(aa.as_ptr().cast::<u8>(), n) };
        let br = unsafe { std::slice::from_raw_parts(bb.as_ptr().cast::<u8>(), n) };
        if ar != br {
            let bad = ar.iter().zip(br).filter(|(x, y)| x != y).count();
            return Err(format!("{}: {bad} differing bytes out of {n}", va.0));
        }
    }
    println!("{{\"variables_compared\":{na},\"differing_bytes\":0}}");
    Ok(())
}
fn locality(grid: &str, map: &str) -> R<()> {
    use std::collections::BTreeSet;
    let f = Nc::open(grid, 0)?;
    let maps = read_map(map)?;
    let nc = maps[0].len();
    let ne = maps[1].len();
    let coe = f.doubles("cellsOnEdge", ne * 2)?;
    let inv = inverse(&maps[0])?;
    let score = |order: &[usize], inverse_ids: Option<&[usize]>| -> f64 {
        let mut sectors = 0usize;
        let mut requests = 0usize;
        // One float32 level, separately for each of the two neighboring-cell loads.
        for warp in order.chunks(32) {
            for side in 0..2 {
                let mut lines = BTreeSet::new();
                for &e in warp {
                    let id = coe[2 * e + side] as i64;
                    let c = if id <= 0 {
                        nc
                    } else {
                        let c = id as usize - 1;
                        inverse_ids.map_or(c, |m| m[c])
                    };
                    lines.insert(c / 32);
                }
                sectors += lines.len();
                requests += 1;
            }
        }
        sectors as f64 / requests as f64
    };
    let original: Vec<_> = (0..ne).collect();
    println!("{{\"cells\":{nc},\"edges\":{ne},\"neighbor_128byte_lines_per_warp_before\":{},\"neighbor_128byte_lines_per_warp_after\":{},\"scope\":\"static k0 float32 cellsOnEdge gather census, not measured cache traffic\"}}",score(&original,None),score(&maps[1],Some(&inv)));
    Ok(())
}
fn canonical_array(a: &[String]) -> R<()> {
    if a.len() < 6 {
        return Err("canonical-array needs MAP ENTITY AXIS WIDTH DIM...".into());
    }
    let mut maps = read_map(&a[1])?;
    let e: usize = a[2].parse().map_err(|_| "invalid entity")?;
    let axis: usize = a[3].parse().map_err(|_| "invalid axis")?;
    let w: usize = a[4].parse().map_err(|_| "invalid width")?;
    let shape: Vec<usize> = a[5..]
        .iter()
        .map(|v| v.parse().map_err(|_| "invalid extent".to_owned()))
        .collect::<R<_>>()?;
    if e >= 3 || axis >= shape.len() || ![1, 2, 4, 8].contains(&w) {
        return Err("invalid array descriptor".into());
    }
    maps[e] = inverse(&maps[e])?;
    if shape[axis] == maps[e].len() + 1 {
        maps[e].push(maps[e].len());
    }
    let mut axes = vec![None; shape.len()];
    axes[axis] = Some(e);
    let mut raw = Vec::new();
    std::io::stdin()
        .read_to_end(&mut raw)
        .map_err(|e| e.to_string())?;
    let out = gather(&raw, w, &shape, &axes, &maps)?;
    std::io::stdout()
        .write_all(&out)
        .map_err(|e| e.to_string())?;
    Ok(())
}
fn run() -> R<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str){
    Some("--make-map") if a.len()==3=>{let f=Nc::open(&a[1],0)?;let dims=f.dims()?;let mut maps:[Vec<usize>;3]=Default::default();for(e,kind) in ["Cell","Edge","Vertex"].iter().enumerate(){let count=dims.iter().find(|(n,_)|entity(n)==Some(e)).ok_or("mesh entity dimension missing")?.1;let xyz=[f.doubles(&format!("x{kind}"),count)?,f.doubles(&format!("y{kind}"),count)?,f.doubles(&format!("z{kind}"),count)?];maps[e]=morton(&xyz)?;}write_map(&a[2],&maps)?;println!("{{\"cells\":{},\"edges\":{},\"vertices\":{}}}",maps[0].len(),maps[1].len(),maps[2].len());Ok(())},
    Some("--apply")|Some("--invert") if a.len()==4=>apply(&a[1],&a[2],&a[3],a[0]=="--invert"),
    Some("--compare") if a.len()==3=>compare(&a[1],&a[2]),
    Some("--locality") if a.len()==3=>locality(&a[1],&a[2]),
    Some("--canonical-array")=>canonical_array(&a),
    Some("--record-donor") if a.len()==3=>record_donor(&a[1],&a[2]),
    _=>Err("usage: hex_order --make-map GRID MAP | --apply MAP INPUT OUTPUT | --invert MAP INPUT OUTPUT | --compare LEFT RIGHT".into())}
}
fn main() {
    if let Err(e) = run() {
        eprintln!("hex_order: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    unsafe extern "C" {
        fn nc_create(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
        fn nc_def_dim(id: c_int, name: *const c_char, len: usize, dim: *mut c_int) -> c_int;
        fn nc_def_var(
            id: c_int,
            name: *const c_char,
            typ: c_int,
            nd: c_int,
            dims: *const c_int,
            var: *mut c_int,
        ) -> c_int;
        fn nc_enddef(id: c_int) -> c_int;
    }
    #[test]
    fn reversible_bytes_and_slots() {
        let maps = [vec![2, 0, 1], vec![1, 0], vec![0]];
        let vals = [
            0x80000000u32,
            0x7fc12345,
            0x3f800000,
            0x40800000,
            0xdeadbeef,
            0,
        ];
        let raw: Vec<_> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ordered = gather(&raw, 4, &[3, 2], &[Some(0), None], &maps).unwrap();
        let undo = [
            inverse(&maps[0]).unwrap(),
            inverse(&maps[1]).unwrap(),
            inverse(&maps[2]).unwrap(),
        ];
        assert_eq!(
            gather(&ordered, 4, &[3, 2], &[Some(0), None], &undo).unwrap(),
            raw
        );
    }
    #[test]
    fn ids_and_sentinels_round_trip() {
        let mut raw: Vec<_> = [0i32, -1, 1, 2, 3]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let before = raw.clone();
        let m = vec![2, 0, 1];
        remap(&mut raw, 4, &inverse(&m).unwrap()).unwrap();
        remap(&mut raw, 4, &m).unwrap();
        assert_eq!(raw, before);
    }
    /// The block copy must be the element-wise reference, on every layout the
    /// files carry: entity axis first, last, in the middle, two entity axes,
    /// and none.
    #[test]
    fn block_gather_is_the_reference() {
        let maps = [vec![3, 0, 2, 1, 4], vec![1, 2, 0], vec![1, 0]];
        let cases: Vec<(Vec<usize>, Vec<Option<usize>>, usize)> = vec![
            (vec![5], vec![Some(0)], 4),
            (vec![5, 7], vec![Some(0), None], 4),
            (vec![2, 5, 3], vec![None, Some(0), None], 8),
            (vec![5, 3], vec![Some(0), Some(1)], 4),
            (vec![3, 5], vec![None, Some(0)], 1),
            (vec![2, 3, 2, 6], vec![None, Some(1), Some(2), None], 2),
            (vec![4, 3], vec![None, None], 4),
        ];
        for (shape, axes, w) in cases {
            let n: usize = shape.iter().product::<usize>() * w;
            let raw: Vec<u8> = (0..n).map(|i| (i * 37 % 251) as u8).collect();
            for t in ["1", "3", "64"] {
                std::env::set_var("HEX_ORDER_THREADS", t);
                assert_eq!(
                    gather(&raw, w, &shape, &axes, &maps).unwrap(),
                    gather_reference(&raw, w, &shape, &axes, &maps).unwrap(),
                    "shape {shape:?} axes {axes:?} width {w} threads {t}"
                );
            }
        }
    }
    #[test]
    fn rejects_non_bijection() {
        assert!(inverse(&[1, 1]).is_err());
        assert!(inverse(&[0, 2]).is_err());
    }
    #[test]
    fn stable_morton_ties() {
        let xyz = [vec![1., 0., 0.], vec![1., 0., 0.], vec![1., 0., 0.]];
        assert_eq!(morton(&xyz).unwrap(), vec![1, 2, 0]);
    }
    #[test]
    fn multiple_entity_axes() {
        let maps = [vec![1, 0], vec![2, 0, 1], vec![0]];
        let raw: Vec<_> = (0u8..6).collect();
        assert_eq!(
            gather(&raw, 1, &[2, 3], &[Some(0), Some(1)], &maps).unwrap(),
            vec![5, 3, 4, 2, 0, 1]
        );
    }
    #[test]
    fn real_netcdf_round_trip_and_difference_detection() {
        let root = Path::new(
            &std::env::var("HEX_ORDER_TEST_ROOT")
                .expect("set HEX_ORDER_TEST_ROOT to an owned test directory"),
        )
        .join(format!("roundtrip-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let a = root.join("original.nc");
        let p = CString::new(a.to_str().unwrap()).unwrap();
        let mut id = 0;
        check(unsafe { nc_create(p.as_ptr(), 0, &mut id) }).unwrap();
        let f = Nc(id);
        let mut cd = 0;
        let mut ed = 0;
        let mut vd = 0;
        let mut sd = 0;
        for (name, n, d) in [
            ("nCells", 3, &mut cd),
            ("nEdges", 2, &mut ed),
            ("nVertices", 1, &mut vd),
            ("TWO", 2, &mut sd),
        ] {
            let nme = CString::new(name).unwrap();
            check(unsafe { nc_def_dim(id, nme.as_ptr(), n, d) }).unwrap();
        }
        let mut temp = 0;
        let mut coe = 0;
        let nme = CString::new("temperature").unwrap();
        check(unsafe { nc_def_var(id, nme.as_ptr(), 5, 1, &cd, &mut temp) }).unwrap();
        let nme = CString::new("cellsOnEdge").unwrap();
        check(unsafe { nc_def_var(id, nme.as_ptr(), 4, 2, [ed, sd].as_ptr(), &mut coe) }).unwrap();
        check(unsafe { nc_enddef(id) }).unwrap();
        let values = [
            f32::from_bits(0x80000000),
            f32::from_bits(0x7fc12345),
            273.15,
        ];
        let ids = [1i32, 3, 2, 0];
        check(unsafe { nc_put_var(id, temp, values.as_ptr().cast()) }).unwrap();
        check(unsafe { nc_put_var(id, coe, ids.as_ptr().cast()) }).unwrap();
        drop(f);
        let m = root.join("map.bin");
        write_map(m.to_str().unwrap(), &[vec![2, 0, 1], vec![1, 0], vec![0]]).unwrap();
        let b = root.join("ordered.nc");
        let c = root.join("restored.nc");
        apply(
            m.to_str().unwrap(),
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            false,
        )
        .unwrap();
        assert!(compare(a.to_str().unwrap(), b.to_str().unwrap()).is_err());
        apply(
            m.to_str().unwrap(),
            b.to_str().unwrap(),
            c.to_str().unwrap(),
            true,
        )
        .unwrap();
        compare(a.to_str().unwrap(), c.to_str().unwrap()).unwrap();
        assert!(apply(
            m.to_str().unwrap(),
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            false
        )
        .unwrap_err()
        .contains("already exists"));
        record_donor(m.to_str().unwrap(), b.to_str().unwrap()).unwrap();
        // Both I/O paths, whole files: the memory image must write the bytes
        // the disk path writes.
        let mut written = Vec::new();
        for mode in ["file", "memory"] {
            std::env::set_var("HEX_ORDER_IO", mode);
            let o = root.join(format!("ordered-{mode}.nc"));
            apply(m.to_str().unwrap(), a.to_str().unwrap(), o.to_str().unwrap(), false).unwrap();
            record_donor(m.to_str().unwrap(), o.to_str().unwrap()).unwrap();
            written.push(fs::read(&o).unwrap());
        }
        std::env::remove_var("HEX_ORDER_IO");
        assert_eq!(written[0], written[1], "memory image and disk path wrote different files");
        let recorded = Nc::open(b.to_str().unwrap(), 0).unwrap();
        assert_eq!(
            recorded.doubles("cell_order_original", 3).unwrap(),
            vec![2.0, 0.0, 1.0]
        );
    }
}
