//! Registry pool transport. MPAS: framework/mpas_io_streams.F:925-1023,
//! framework/mpas_pool_routines.F:2110-3143. No scheme arithmetic.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, CString},
    fs,
    io::Write,
    os::raw::{c_char, c_int},
    path::Path,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(p: *const c_char, m: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_strerror(c: c_int) -> *const c_char;
    fn nc_inq_varid(id: c_int, n: *const c_char, v: *mut c_int) -> c_int;
    fn nc_inq_ndims(id: c_int, r: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, v: c_int, r: *mut c_int) -> c_int;
    fn nc_inq_var(
        id: c_int,
        v: c_int,
        n: *mut c_char,
        k: *mut c_int,
        r: *mut c_int,
        d: *mut c_int,
        a: *mut c_int,
    ) -> c_int;
    fn nc_inq_dim(id: c_int, d: c_int, n: *mut c_char, l: *mut usize) -> c_int;
    fn nc_get_vara_float(
        id: c_int,
        v: c_int,
        s: *const usize,
        c: *const usize,
        p: *mut f32,
    ) -> c_int;
    fn nc_get_vara_int(id: c_int, v: c_int, s: *const usize, c: *const usize, p: *mut i32)
    -> c_int;
    fn nc_get_vara_text(
        id: c_int,
        v: c_int,
        s: *const usize,
        c: *const usize,
        p: *mut c_char,
    ) -> c_int;
}
fn check(c: c_int) -> Result<(), String> {
    if c == 0 {
        Ok(())
    } else {
        Err(unsafe { CStr::from_ptr(nc_strerror(c)) }
            .to_string_lossy()
            .into_owned())
    }
}
struct File {
    id: c_int,
    path: String,
}
impl File {
    fn open(path: &str) -> Result<Self, String> {
        let mut id = 0;
        check(unsafe {
            nc_open(
                CString::new(path).map_err(|e| e.to_string())?.as_ptr(),
                0,
                &mut id,
            )
        })?;
        Ok(Self {
            id,
            path: path.into(),
        })
    }
}
impl Drop for File {
    fn drop(&mut self) {
        unsafe {
            nc_close(self.id);
        }
    }
}
#[derive(Deserialize)]
struct Request {
    schema: String,
    sources: Vec<String>,
    fields: Vec<Field>,
}
#[derive(Deserialize)]
struct Field {
    pool: String,
    name: String,
    kind: String,
    dimensions: Vec<String>,
    shape: Vec<usize>,
    #[serde(default)]
    components: Vec<String>,
    #[serde(default)]
    initial: String,
    #[serde(default)]
    required: bool,
    #[serde(default)]
    read_input: bool,
}
#[derive(Serialize)]
struct Receipt {
    schema: &'static str,
    fields: Vec<Record>,
}
#[derive(Serialize)]
struct Record {
    pool: String,
    name: String,
    kind: String,
    dimensions: Vec<String>,
    shape: Vec<usize>,
    payload: Option<String>,
    sha256: Option<String>,
    text: Option<String>,
    origins: Vec<String>,
}
// One netCDF read in the library's own words.  The little-endian encoding is
// done by the caller after the netCDF lock is released (netCDF-C is not
// thread-safe, so only the library call itself is serialised).
enum Words {
    Real(Vec<f32>),
    Integer(Vec<i32>),
    Text(Vec<u8>),
}
impl Words {
    fn into_le_bytes(self) -> Vec<u8> {
        match self {
            Words::Real(b) => {
                let mut bytes = Vec::with_capacity(b.len() * 4);
                for x in b {
                    bytes.extend_from_slice(&x.to_bits().to_le_bytes());
                }
                bytes
            }
            Words::Integer(b) => {
                let mut bytes = Vec::with_capacity(b.len() * 4);
                for x in b {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                bytes
            }
            Words::Text(b) => b,
        }
    }
}
impl Words {
    // The same bytes as into_le_bytes, borrowed when the host is little-endian.
    fn le_bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            Words::Text(b) => std::borrow::Cow::Borrowed(b),
            Words::Real(b) if cfg!(target_endian = "little") => {
                // SAFETY: f32 has no padding; its native bytes are its little-endian bits on this host.
                std::borrow::Cow::Borrowed(unsafe {
                    std::slice::from_raw_parts(b.as_ptr().cast::<u8>(), b.len() * 4)
                })
            }
            Words::Integer(b) if cfg!(target_endian = "little") => {
                // SAFETY: as above, for i32.
                std::borrow::Cow::Borrowed(unsafe {
                    std::slice::from_raw_parts(b.as_ptr().cast::<u8>(), b.len() * 4)
                })
            }
            Words::Real(b) => {
                std::borrow::Cow::Owned(b.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect())
            }
            Words::Integer(b) => {
                std::borrow::Cow::Owned(b.iter().flat_map(|x| x.to_le_bytes()).collect())
            }
        }
    }
}
// Where a variable is and how it is read: the netCDF metadata read() takes before its data call.
struct Located {
    file: usize,
    var: c_int,
    nc_type: c_int,
    // Source axes and lengths with a leading Time removed (what pack sees).
    dims: Vec<String>,
    counts: Vec<usize>,
    // The netCDF count vector (Time, when first, read as one record).
    nc_counts: Vec<usize>,
    time: bool,
    origin: String,
}
fn locate(files: &[File], name: &str, kind: &str) -> Result<Option<Located>, String> {
    for (index, f) in files.iter().enumerate().rev() {
        let mut v = 0;
        let n = CString::new(name).map_err(|e| e.to_string())?;
        if unsafe { nc_inq_varid(f.id, n.as_ptr(), &mut v) } != 0 {
            continue;
        }
        let (mut k, mut rank) = (0, 0);
        check(unsafe { nc_inq_varndims(f.id, v, &mut rank) })?;
        if !(0..=16).contains(&rank) {
            return Err(format!("{name}: invalid rank {rank}"));
        }
        let mut ds = vec![0; rank as usize];
        check(unsafe {
            nc_inq_var(
                f.id,
                v,
                std::ptr::null_mut(),
                &mut k,
                &mut rank,
                ds.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;
        let mut dims = Vec::new();
        let mut counts = Vec::new();
        for d in &ds[..rank as usize] {
            let mut label = [0 as c_char; 257];
            let mut len = 0;
            check(unsafe { nc_inq_dim(f.id, *d, label.as_mut_ptr(), &mut len) })?;
            dims.push(
                unsafe { CStr::from_ptr(label.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
            );
            counts.push(len);
        }
        let time = dims.first().map(String::as_str) == Some("Time");
        if time {
            if counts[0] == 0 {
                return Err(format!("{name}: empty Time"));
            }
            counts[0] = 1;
        }
        match kind {
            "real" if k != 5 && k != 6 => {
                return Err(format!("{name}: expected NC_FLOAT/DOUBLE, got {k}"));
            }
            "integer" if k != 4 => return Err(format!("{name}: expected NC_INT, got {k}")),
            "character" if k != 2 => return Err(format!("{name}: expected NC_CHAR, got {k}")),
            "real" | "integer" | "character" => {}
            _ => return Err(format!("unsupported kind {kind}")),
        }
        let nc_counts = counts.clone();
        if time {
            dims.remove(0);
            counts.remove(0);
        }
        return Ok(Some(Located {
            file: index,
            var: v,
            nc_type: k,
            dims,
            counts,
            nc_counts,
            time,
            origin: format!("{}:{name}", f.path),
        }));
    }
    Ok(None)
}
// The data call of read(), for a variable locate() found.
fn fetch(files: &[File], at: &Located, kind: &str) -> Result<Words, String> {
    let f = &files[at.file];
    let starts = vec![0; at.nc_counts.len()];
    let len: usize = at.nc_counts.iter().product();
    Ok(match kind {
        "real" => {
            let mut b = vec![0f32; len];
            check(unsafe {
                nc_get_vara_float(f.id, at.var, starts.as_ptr(), at.nc_counts.as_ptr(), b.as_mut_ptr())
            })?;
            Words::Real(b)
        }
        "integer" => {
            let mut b = vec![0i32; len];
            check(unsafe {
                nc_get_vara_int(f.id, at.var, starts.as_ptr(), at.nc_counts.as_ptr(), b.as_mut_ptr())
            })?;
            Words::Integer(b)
        }
        "character" => {
            let mut b = vec![0u8; len];
            check(unsafe {
                nc_get_vara_text(
                    f.id,
                    at.var,
                    starts.as_ptr(),
                    at.nc_counts.as_ptr(),
                    b.as_mut_ptr().cast(),
                )
            })?;
            Words::Text(b)
        }
        _ => return Err(format!("unsupported kind {kind}")),
    })
}
fn locked<T>(lock: &Mutex<()>, call: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let _held = lock.lock().map_err(|_| "netCDF lock poisoned".to_string())?;
    call()
}
// read() under the process-wide netCDF lock, encoded to bytes after the lock is released.
fn read_locked(
    files: &[File],
    lock: &Mutex<()>,
    name: &str,
    kind: &str,
) -> Result<Option<(Vec<u8>, Vec<String>, Vec<usize>, String)>, String> {
    let found = locked(lock, || match locate(files, name, kind)? {
        Some(at) => Ok(Some((fetch(files, &at, kind)?, at))),
        None => Ok(None),
    })?;
    Ok(found.map(|(w, at)| (w.into_le_bytes(), at.dims, at.counts, at.origin)))
}

// ---- Classic netCDF (CDF-1, CDF-2, CDF-5) words read straight from the file ----
//
// THE COST THIS REMOVES (NOAA RRFS v2 conus3km, 2,341,819 cells x 59 levels, 8 x RTX PRO 6000,
// V8X6000 2026-10-29): the transport took 34 s at 8, 16 and 32 reader processes alike, held by the two
// 7.2 GB scalar fields (state scalars, lbc_scalars), each one field on one thread: 13 netCDF-C reads
// behind the library lock, a per-word div/mod permutation of 1.8 G words, then one write and one SHA-256.
// A field's words are now read by many threads at once: the netCDF-C metadata calls stay behind the
// lock, and the data of an NC_FLOAT/NC_INT variable in a classic file is read with positioned reads in
// row chunks and byte-swapped from the file's big-endian words, which is what nc_get_vara_float/_int do
// for those types (ncx_getn_float_float / ncx_getn_int_int: a byte swap, no conversion).  Any other case
// (NC_DOUBLE into a real field, netCDF-4 files, a record variable not read as one Time record, a file
// shorter than the variable) still reads through netCDF-C.  Every payload is the same bytes.
struct ClassicVar {
    nc_type: c_int,
    begin: u64,
    record: bool,
    ndims: usize,
}
struct Classic {
    file: fs::File,
    len: u64,
    vars: std::collections::HashMap<String, ClassicVar>,
}
struct Header<'a> {
    b: &'a [u8],
    p: usize,
    wide: bool,
}
impl<'a> Header<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let b = self.b;
        let end = self.p.checked_add(n)?;
        let s = b.get(self.p..end)?;
        self.p = end;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_be_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_be_bytes(self.take(8)?.try_into().ok()?))
    }
    // NON_NEG: 32-bit in CDF-1/2, 64-bit in CDF-5.
    fn count(&mut self) -> Option<u64> {
        if self.wide {
            self.u64()
        } else {
            self.u32().map(u64::from)
        }
    }
    fn padded(&mut self, n: u64) -> Option<&'a [u8]> {
        let n = usize::try_from(n).ok()?;
        let s = self.take(n.checked_add((4 - n % 4) % 4)?)?;
        Some(&s[..n])
    }
    fn name(&mut self) -> Option<String> {
        let n = self.count()?;
        String::from_utf8(self.padded(n)?.to_vec()).ok()
    }
    fn attributes(&mut self) -> Option<()> {
        let (tag, n) = (self.u32()?, self.count()?);
        if tag == 0 && n == 0 {
            return Some(());
        }
        if tag != 0x0C {
            return None;
        }
        for _ in 0..n {
            self.name()?;
            let size: u64 = match self.u32()? {
                1 | 2 | 7 => 1,
                3 | 8 => 2,
                4 | 5 | 9 => 4,
                6 | 10 | 11 => 8,
                _ => return None,
            };
            let count = self.count()?;
            self.padded(count.checked_mul(size)?)?;
        }
        Some(())
    }
}
fn parse_classic(head: &[u8]) -> Option<(bool, Vec<(String, ClassicVar)>)> {
    let version = match head.get(..4)? {
        [b'C', b'D', b'F', v] => *v,
        _ => return None,
    };
    let (wide, offset64) = match version {
        1 => (false, false),
        2 => (false, true),
        5 => (true, true),
        _ => return None,
    };
    let mut h = Header { b: head, p: 4, wide };
    h.count()?; // numrecs
    let mut record_dims = Vec::new();
    let (tag, n) = (h.u32()?, h.count()?);
    if tag == 0x0A {
        for _ in 0..n {
            h.name()?;
            record_dims.push(h.count()? == 0);
        }
    } else if !(tag == 0 && n == 0) {
        return None;
    }
    h.attributes()?;
    let mut vars = Vec::new();
    let (tag, n) = (h.u32()?, h.count()?);
    if tag == 0x0B {
        for _ in 0..n {
            let name = h.name()?;
            let ndims = usize::try_from(h.count()?).ok()?;
            let mut record = false;
            for i in 0..ndims {
                let id = usize::try_from(h.count()?).ok()?;
                record |= i == 0 && *record_dims.get(id)?;
            }
            h.attributes()?;
            let nc_type = c_int::try_from(h.u32()?).ok()?;
            h.count()?; // vsize
            let begin = if offset64 { h.u64()? } else { u64::from(h.u32()?) };
            vars.push((name, ClassicVar { nc_type, begin, record, ndims }));
        }
    } else if !(tag == 0 && n == 0) {
        return None;
    }
    Some((wide, vars))
}
fn open_classic(path: &str) -> Option<Classic> {
    let file = fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    for limit in [1u64 << 20, 64 << 20] {
        let mut head = vec![0u8; len.min(limit) as usize];
        read_at(&file, &mut head, 0).ok()?;
        if let Some((_, vars)) = parse_classic(&head) {
            return Some(Classic { file, len, vars: vars.into_iter().collect() });
        }
        if len <= limit {
            break;
        }
    }
    None
}
#[cfg(unix)]
fn read_at(f: &fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, offset)
}
#[cfg(windows)]
fn read_at(f: &fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, offset)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}
#[cfg(unix)]
fn write_at(f: &fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.write_all_at(buf, offset)
}
#[cfg(windows)]
fn write_at(f: &fs::File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_write(buf, offset)?;
        if n == 0 {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        buf = &buf[n..];
        offset += n as u64;
    }
    Ok(())
}
// The file offset of a located variable's words when they can be read directly (see above).
fn direct<'a>(
    classic: &'a [OnceLock<Option<Classic>>],
    files: &[File],
    at: &Located,
    name: &str,
    kind: &str,
) -> Option<(&'a Classic, u64)> {
    if !((kind == "real" && at.nc_type == 5) || (kind == "integer" && at.nc_type == 4)) {
        return None;
    }
    let c = classic[at.file]
        .get_or_init(|| open_classic(&files[at.file].path))
        .as_ref()?;
    let v = c.vars.get(name)?;
    if v.nc_type != at.nc_type || v.ndims != at.nc_counts.len() || (v.record && !at.time) {
        return None;
    }
    let bytes = u64::try_from(at.counts.iter().product::<usize>()).ok()?.checked_mul(4)?;
    if v.begin.checked_add(bytes)? > c.len {
        return None;
    }
    Some((c, v.begin))
}

// Where each source word goes: the axis permutation and native trailing dummy entity of pack().
struct Plan {
    counts: Vec<usize>,
    // Destination word stride of each source axis.
    steps: Vec<usize>,
    base: usize,
    // Words per source row (the source's fastest axis) and rows.
    inner: usize,
    rows: usize,
    // A source axis lands on the component axis, so components may share words: keep request order.
    overlaps_components: bool,
}
impl Plan {
    fn new(
        dims: &[String],
        counts: &[usize],
        field: &Field,
        component: Option<usize>,
    ) -> Result<Plan, String> {
        let offset = usize::from(component.is_some());
        if dims.len() + offset != field.dimensions.len() {
            return Err(format!("{}: dimension rank differs", field.name));
        }
        let axes: Vec<usize> = dims
            .iter()
            .map(|d| {
                field
                    .dimensions
                    .iter()
                    .position(|x| x == d)
                    .ok_or_else(|| format!("{}: unexpected axis {d}", field.name))
            })
            .collect::<Result<_, _>>()?;
        for (a, n) in axes.iter().zip(counts) {
            let expected = field.shape[*a];
            let padded = matches!(
                field.dimensions[*a].as_str(),
                "nCells" | "nEdges" | "nVertices"
            );
            if *n != expected - usize::from(padded) {
                return Err(format!(
                    "{}: axis {} length {n} != {}",
                    field.name,
                    field.dimensions[*a],
                    expected - usize::from(padded)
                ));
            }
        }
        if axes.iter().collect::<std::collections::HashSet<_>>().len() != axes.len() {
            return Err(format!("{}: repeated source axis", field.name));
        }
        if component.is_some_and(|c| c >= field.shape[0]) {
            return Err("component exceeds selected scalar count".into());
        }
        let mut strides = vec![1; field.shape.len()];
        for a in 1..strides.len() {
            strides[a] = strides[a - 1] * field.shape[a - 1];
        }
        let rank = counts.len();
        let inner = if rank == 0 { 1 } else { counts[rank - 1] };
        let len: usize = counts.iter().product();
        Ok(Plan {
            counts: counts.to_vec(),
            steps: axes.iter().map(|a| strides[*a]).collect(),
            base: component.unwrap_or(0),
            inner,
            rows: if inner == 0 { 0 } else { len / inner },
            overlaps_components: component.is_some() && axes.contains(&0),
        })
    }
}
#[derive(Clone, Copy)]
struct Out(*mut u8, usize);
// SAFETY: threads write disjoint words of one live buffer (Plan maps distinct source words to
// distinct destination words; components land on distinct component offsets unless
// overlaps_components, which runs on one thread).
unsafe impl Send for Out {}
unsafe impl Sync for Out {}
// Place source rows [row0, row0 + input/inner) of one component, word for word as pack's loop does.
fn scatter(plan: &Plan, input: &[u8], row0: usize, out: Out) -> Result<(), String> {
    let inner = plan.inner;
    if inner == 0 {
        return Ok(());
    }
    if input.len() % (inner * 4) != 0 {
        return Err("invalid payload word count".into());
    }
    let rank = plan.counts.len();
    let step = if rank == 0 { 1 } else { plan.steps[rank - 1] };
    for (k, src) in input.chunks_exact(inner * 4).enumerate() {
        let mut r = row0 + k;
        let mut dst = plan.base;
        for a in (0..rank.saturating_sub(1)).rev() {
            dst += (r % plan.counts[a]) * plan.steps[a];
            r /= plan.counts[a];
        }
        let last = dst + (inner - 1) * step;
        if last >= out.1 / 4 {
            return Err("destination word out of range".into());
        }
        // SAFETY: every word written is inside the buffer (checked above), and no other thread
        // writes these words (Out).
        unsafe {
            if step == 1 {
                std::ptr::copy_nonoverlapping(src.as_ptr(), out.0.add(dst * 4), inner * 4);
            } else {
                for i in 0..inner {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(i * 4),
                        out.0.add((dst + i * step) * 4),
                        4,
                    );
                }
            }
        }
    }
    Ok(())
}
// Permute named axes and insert the native trailing dummy entity, byte for byte (one pass of
// scatter; the reference the tests hold the chunked transport to).
#[cfg(test)]
fn pack(
    input: &[u8],
    dims: &[String],
    counts: &[usize],
    field: &Field,
    component: Option<usize>,
    out: &mut [u8],
) -> Result<(), String> {
    let plan = Plan::new(dims, counts, field, component)?;
    let len: usize = counts.iter().product();
    if input.len() != len * 4 {
        return Err("invalid payload word count".into());
    }
    scatter(&plan, input, 0, Out(out.as_mut_ptr(), out.len()))
}
// THE BREAKAGE THIS REMOVES (U-H3, hex 0.4.8, 30 km globe x1.655362 on 4x RTX PRO 6000,
// 2026-10-27): every MPAS-physics member start ran this transport on ONE thread for
// about 30 s (100 % of one core, NLWP 1) while its card waited.  Fields are now taken
// by a pool of workers: only the netCDF-C call is serialised (the library is not
// thread-safe); the byte encoding, the axis permutation, the payload write and its
// SHA-256 run beside it.  Each field is the same bytes at any worker count, and the
// manifest keeps request order.  The count is RAYON_NUM_THREADS when set (hexcore
// sets it to the container CPU quota), else the CPUs this process may use, at most 16.
const MAX_WORKERS: usize = 16;
fn workers() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .clamp(1, MAX_WORKERS)
}
// Threads one large field is split over (its row chunks, its payload write): the CPUs this process
// may use, at most MAX_FIELD_THREADS; GPUWM_HEX_POOL_FIELD_THREADS sets it.  The conus3km scalar
// fields are one field each, so the field-level workers above cannot share them out.
const MAX_FIELD_THREADS: usize = 32;
const CHUNK_BYTES: usize = 16 << 20;
const PARALLEL_WRITE_BYTES: usize = 64 << 20;
fn field_threads() -> usize {
    std::env::var("GPUWM_HEX_POOL_FIELD_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .clamp(1, MAX_FIELD_THREADS)
}
// Run jobs 0..n on up to `threads` threads; the result of each job, in job order.
fn run_jobs<T: Send>(n: usize, threads: usize, job: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        return (0..n).map(&job).collect();
    }
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::AcqRel);
                if i >= n {
                    break;
                }
                let value = job(i);
                *slots[i].lock().unwrap_or_else(|p| p.into_inner()) = Some(value);
            });
        }
    });
    slots
        .into_iter()
        .map(|s| s.into_inner().unwrap_or_else(|p| p.into_inner()).expect("job ran"))
        .collect()
}
enum Source<'a> {
    Direct(&'a Classic, u64),
    NetCdf(Located),
}
struct Part<'a> {
    plan: Plan,
    source: Source<'a>,
}
fn transport(
    index: usize,
    f: &Field,
    files: &[File],
    classic: &[OnceLock<Option<Classic>>],
    lock: &Mutex<()>,
    out: &Path,
) -> Result<Record, String> {
    // MPAS-Model-rrfs/src/framework/mpas_field_routines.F:261-299:
    // zero-sized Registry dimensions allocate associated empty arrays.
    if f.shape.len() != f.dimensions.len() {
        return Err(format!("{}: invalid destination shape", f.name));
    }
    let mut origins = Vec::new();
    let mut text = None;
    let mut payload = None;
    let mut hash = None;
    if f.kind == "character" {
        let b = if f.read_input {
            read_locked(files, lock, &f.name, &f.kind)?
        } else {
            None
        };
        text = Some(if let Some((b, _, _, src)) = b {
            origins.push(src);
            String::from_utf8(b)
                .map_err(|e| e.to_string())?
                .trim_end_matches(['\0', ' '])
                .to_string()
        } else {
            if f.required {
                return Err(format!("required input {} absent", f.name));
            }
            origins.push("registry-initial-value".into());
            f.initial.clone()
        });
    } else {
        let word = if f.kind == "integer" {
            if f.initial.is_empty() {
                0
            } else {
                f.initial
                    .parse::<i32>()
                    .map_err(|e| format!("{}: invalid integer initial value: {e}", f.name))?
            }
            .to_le_bytes()
        } else {
            if f.initial.is_empty() {
                0.
            } else {
                f.initial
                    .replace(['d', 'D'], "e")
                    .parse::<f32>()
                    .map_err(|e| format!("{}: invalid real initial value: {e}", f.name))?
            }
            .to_bits()
            .to_le_bytes()
        };
        let len: usize = f.shape.iter().product();
        let names = if f.components.is_empty() {
            vec![f.name.clone()]
        } else {
            f.components.clone()
        };
        // 1. Every component's metadata, in component order, behind the netCDF lock.  The first
        // refusal stops the walk; data errors of the components before it are reported first,
        // as a serial pass would.
        let mut parts: Vec<Part> = Vec::new();
        let mut refusal = None;
        for (c, n) in names.iter().enumerate() {
            let at = if f.read_input {
                match locked(lock, || locate(files, n, &f.kind)) {
                    Ok(at) => at,
                    Err(e) => {
                        refusal = Some(e);
                        break;
                    }
                }
            } else {
                None
            };
            let Some(at) = at else {
                if f.required {
                    refusal = Some(format!("required input {n} absent"));
                    break;
                }
                origins.push(format!("registry-initial-value:{n}"));
                continue;
            };
            let component = if f.components.is_empty() { None } else { Some(c) };
            let plan = match Plan::new(&at.dims, &at.counts, f, component) {
                Ok(plan) => plan,
                Err(e) => {
                    refusal = Some(e);
                    break;
                }
            };
            origins.push(at.origin.clone());
            let source = match direct(classic, files, &at, n, &f.kind) {
                Some((c, begin)) => Source::Direct(c, begin),
                None => Source::NetCdf(at),
            };
            parts.push(Part { plan, source });
        }
        // 2. The allocation default (words no component writes keep it), then the components' words.
        let mut bytes = if word == [0; 4] {
            vec![0u8; len * 4]
        } else {
            word.repeat(len)
        };
        let threads = if parts.iter().any(|p| p.plan.overlaps_components) {
            1
        } else {
            field_threads()
        };
        // Jobs: row chunks of a direct read; a whole component for a netCDF-C read.
        let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
        for (p, part) in parts.iter().enumerate() {
            let rows = part.plan.rows;
            let step = match part.source {
                Source::Direct(..) => (CHUNK_BYTES / (4 * part.plan.inner.max(1))).max(1),
                Source::NetCdf(_) => rows.max(1),
            };
            let mut r = 0;
            while r < rows {
                jobs.push((p, r, (r + step).min(rows)));
                r += step;
            }
        }
        let target = Out(bytes.as_mut_ptr(), bytes.len());
        let results = run_jobs(jobs.len(), threads, |j| -> Result<(), String> {
            let (p, r0, r1) = jobs[j];
            let part = &parts[p];
            let inner = part.plan.inner;
            match &part.source {
                Source::Direct(c, begin) => {
                    let mut buf = vec![0u8; (r1 - r0) * inner * 4];
                    read_at(&c.file, &mut buf, begin + (r0 * inner * 4) as u64)
                        .map_err(|e| format!("{}: {e}", f.name))?;
                    // The file's big-endian words to little-endian, as ncx_getn_*: a byte swap.
                    for w in buf.chunks_exact_mut(4) {
                        let v = u32::from_be_bytes([w[0], w[1], w[2], w[3]]);
                        w.copy_from_slice(&v.to_le_bytes());
                    }
                    scatter(&part.plan, &buf, r0, target)
                }
                Source::NetCdf(at) => {
                    let words = locked(lock, || fetch(files, at, &f.kind))?;
                    scatter(&part.plan, &words.le_bytes(), r0, target)
                }
            }
        });
        if let Some(e) = results.into_iter().find_map(Result::err) {
            return Err(e);
        }
        if let Some(e) = refusal {
            return Err(e);
        }
        // 3. The payload, and its SHA-256 beside the write.  Retain the allocation default in unread
        // dummy storage.
        let name = format!("field-{index:04}.bin");
        let path = out.join(&name);
        if bytes.len() < PARALLEL_WRITE_BYTES || threads == 1 {
            let mut target = fs::File::create(&path).map_err(|e| e.to_string())?;
            target.write_all(&bytes).map_err(|e| e.to_string())?;
            hash = Some(format!("{:x}", Sha256::digest(&bytes)));
        } else {
            let target = fs::File::create(&path).map_err(|e| e.to_string())?;
            target.set_len(bytes.len() as u64).map_err(|e| e.to_string())?;
            let chunks: Vec<&[u8]> = bytes.chunks(CHUNK_BYTES * 4).collect();
            let (digest, written) = std::thread::scope(|scope| {
                let digest = scope.spawn(|| format!("{:x}", Sha256::digest(&bytes)));
                let written = run_jobs(chunks.len(), threads, |k| {
                    write_at(&target, chunks[k], (k * CHUNK_BYTES * 4) as u64)
                });
                (digest.join().expect("hash thread"), written)
            });
            if let Some(e) = written.into_iter().find_map(Result::err) {
                return Err(e.to_string());
            }
            hash = Some(digest);
        }
        payload = Some(name);
    }
    Ok(Record {
        pool: f.pool.clone(),
        name: f.name.clone(),
        kind: f.kind.clone(),
        dimensions: f.dimensions.clone(),
        shape: f.shape.clone(),
        payload,
        sha256: hash,
        text,
        origins,
    })
}
// GPUWM_HEX_POOL_DIRECT_READS=0 reads every variable through netCDF-C (the same payload bytes).
fn direct_reads() -> bool {
    std::env::var("GPUWM_HEX_POOL_DIRECT_READS").map_or(true, |v| v.trim() != "0")
}
fn run_with(request: &Path, out: &Path, workers: usize) -> Result<(), String> {
    run_with_reads(request, out, workers, direct_reads())
}
fn run_with_reads(request: &Path, out: &Path, workers: usize, direct: bool) -> Result<(), String> {
    let r: Request = serde_json::from_slice(&fs::read(request).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if r.schema != "mpas-native-pool-request-v1" {
        return Err("unsupported request".into());
    }
    let files: Vec<File> = r
        .sources
        .iter()
        .map(|p| File::open(p))
        .collect::<Result<_, _>>()?;
    let classic: Vec<OnceLock<Option<Classic>>> = files
        .iter()
        .map(|_| {
            let lock = OnceLock::new();
            if !direct {
                let _ = lock.set(None);
            }
            lock
        })
        .collect();
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let fields = r.fields;
    let lock = Mutex::new(());
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let slots: Vec<Mutex<Option<Result<Record, String>>>> =
        fields.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..workers.clamp(1, fields.len().max(1)) {
            scope.spawn(|| {
                // Indices are claimed in order and every claimed field finishes, so the
                // claimed set is a prefix and the first error in request order is the one
                // a serial pass would have raised.
                while !failed.load(Ordering::Acquire) {
                    let i = next.fetch_add(1, Ordering::AcqRel);
                    if i >= fields.len() {
                        break;
                    }
                    let result = transport(i, &fields[i], &files, &classic, &lock, out);
                    if result.is_err() {
                        failed.store(true, Ordering::Release);
                    }
                    *slots[i].lock().unwrap_or_else(|p| p.into_inner()) = Some(result);
                }
            });
        }
    });
    let mut records = Vec::with_capacity(fields.len());
    for slot in slots {
        match slot.into_inner().unwrap_or_else(|p| p.into_inner()) {
            Some(Ok(record)) => records.push(record),
            Some(Err(e)) => return Err(e),
            None => return Err("physics pool: a field was never transported".into()),
        }
    }
    let receipt = Receipt {
        schema: "mpas-native-pool-payload-v1",
        fields: records,
    };
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
fn run(request: &Path, out: &Path) -> Result<(), String> {
    run_with(request, out, workers())
}
// File metadata only; Registry allocation remains in the orchestration layer.
fn inspect(path: &str) -> Result<(), String> {
    let f = File::open(path)?;
    let mut rank = 0;
    check(unsafe { nc_inq_ndims(f.id, &mut rank) })?;
    let mut dims = std::collections::BTreeMap::new();
    for d in 0..rank {
        let mut label = [0 as c_char; 257];
        let mut len = 0;
        check(unsafe { nc_inq_dim(f.id, d, label.as_mut_ptr(), &mut len) })?;
        dims.insert(
            unsafe { CStr::from_ptr(label.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            len,
        );
    }
    println!(
        "{}",
        serde_json::to_string(&dims).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    if a.len() == 2 && a[1] == "--abi" {
        println!("gpuwm-rw-mpas-physics-pool-v1");
        return;
    }
    if a.len() == 3 && a[1] == "--inspect" {
        if let Err(e) = inspect(&a[2]) {
            eprintln!("physics pool: {e}");
            std::process::exit(1);
        }
        return;
    }
    if a.len() != 3 {
        eprintln!("usage: rw_mpas_physics_pool REQUEST.json OUTPUT_DIR");
        std::process::exit(2);
    }
    if let Err(e) = run(Path::new(&a[1]), Path::new(&a[2])) {
        eprintln!("physics pool: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn field() -> Field {
        Field {
            pool: "state".into(),
            name: "scalars".into(),
            kind: "real".into(),
            dimensions: vec!["num_scalars".into(), "nVertLevels".into(), "nCells".into()],
            shape: vec![2, 2, 4],
            components: vec!["qv".into(), "qc".into()],
            initial: String::new(),
            required: true,
            read_input: true,
        }
    }
    #[test]
    fn permutes_scalar_words_without_arithmetic() {
        let words = [
            0x80000000u32,
            0x7fc00001,
            0x00000001,
            0x3f800000,
            0xff800000,
            0x7f800000,
        ];
        let input: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut out = vec![0; 64];
        pack(
            &input,
            &["nCells".into(), "nVertLevels".into()],
            &[3, 2],
            &field(),
            Some(1),
            &mut out,
        )
        .unwrap();
        for c in 0..3 {
            for z in 0..2 {
                let dst = 1 + 2 * z + 4 * c;
                assert_eq!(&out[dst * 4..dst * 4 + 4], &words[2 * c + z].to_le_bytes());
            }
        }
        assert_eq!(&out[48..], &[0; 16]);
    }
    #[test]
    fn rejects_extent_and_repeated_axis() {
        let mut out = vec![0; 64];
        assert!(
            pack(
                &[0; 16],
                &["nCells".into(), "nVertLevels".into()],
                &[2, 2],
                &field(),
                Some(0),
                &mut out
            )
            .is_err()
        );
        assert!(
            pack(
                &[0; 16],
                &["nVertLevels".into(), "nVertLevels".into()],
                &[2, 2],
                &field(),
                Some(0),
                &mut out
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod rrfs_empty_pool_tests {
    use super::*;
    #[test]
    fn registry_zero_dimension_produces_associated_empty_payload() {
        let root = std::env::var("RRFS_POOL_TEST_ROOT").unwrap_or_else(|_| format!("{}/target",env!("CARGO_MANIFEST_DIR")));
        let root = Path::new(&root).join(format!("rrfs-empty-transport-{}",std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let request=root.join("request.json");
        let r=serde_json::json!({"schema":"mpas-native-pool-request-v1","sources":[],
            "fields":[{"pool":"diag_physics","name":"ddvel","kind":"real",
            "dimensions":["nChem","nCells"],"shape":[0,4],"initial":"0"},
            {"pool":"diag_physics","name":"empty_int","kind":"integer",
            "dimensions":["nCells","nChem"],"shape":[4,0],"initial":"-1"}]});
        fs::write(&request,serde_json::to_vec(&r).unwrap()).unwrap();
        run(&request,&root.join("output")).unwrap();
        let result:serde_json::Value=serde_json::from_slice(
            &fs::read(root.join("output/manifest.json")).unwrap()).unwrap();
        for record in result["fields"].as_array().unwrap() {
            let bytes=fs::read(root.join("output").join(record["payload"].as_str().unwrap())).unwrap();
            assert!(bytes.is_empty());
            assert_eq!(record["sha256"],format!("{:x}",Sha256::digest(&bytes)));
        }
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod parallel_transport_tests {
    use super::*;
    // The per-word reference loop pack() had before the contiguous-run path (U-H3).
    fn reference(input: &[u8], counts: &[usize], axes: &[usize], shape: &[usize],
                 component: Option<usize>, out: &mut [u8]) {
        let mut strides = vec![1; shape.len()];
        for a in 1..strides.len() {
            strides[a] = strides[a - 1] * shape[a - 1];
        }
        let len: usize = counts.iter().product();
        for i in 0..len {
            let mut r = i;
            let mut dst = component.unwrap_or(0);
            for a in (0..counts.len()).rev() {
                let c = r % counts[a];
                r /= counts[a];
                dst += c * strides[axes[a]];
            }
            out[dst * 4..dst * 4 + 4].copy_from_slice(&input[i * 4..i * 4 + 4]);
        }
    }
    #[test]
    fn contiguous_runs_place_every_word_where_the_word_loop_does() {
        // (Time,) nCells, nVertLevels -> Fortran (nVertLevels, nCells+1): the run path.
        let field = Field {
            pool: "state".into(), name: "theta".into(), kind: "real".into(),
            dimensions: vec!["nVertLevels".into(), "nCells".into()], shape: vec![3, 6],
            components: vec![], initial: String::new(), required: true, read_input: true,
        };
        let input: Vec<u8> = (0u32..15).flat_map(|w| (0x40000000 + w).to_le_bytes()).collect();
        let dims = ["nCells".to_string(), "nVertLevels".to_string()];
        let mut fast = vec![0xAB; 72];
        let mut slow = vec![0xAB; 72];
        pack(&input, &dims, &[5, 3], &field, None, &mut fast).unwrap();
        reference(&input, &[5, 3], &[1, 0], &field.shape, None, &mut slow);
        assert_eq!(fast, slow);
        // The other order goes through the word loop and still matches.
        let dims = ["nVertLevels".to_string(), "nCells".to_string()];
        let mut fast = vec![0xCD; 72];
        let mut slow = vec![0xCD; 72];
        pack(&input, &dims, &[3, 5], &field, None, &mut fast).unwrap();
        reference(&input, &[3, 5], &[0, 1], &field.shape, None, &mut slow);
        assert_eq!(fast, slow);
    }
    fn write_request(root: &Path, fields: serde_json::Value) -> std::path::PathBuf {
        fs::create_dir_all(root).unwrap();
        let request = root.join("request.json");
        let r = serde_json::json!({"schema":"mpas-native-pool-request-v1","sources":[],"fields":fields});
        fs::write(&request, serde_json::to_vec(&r).unwrap()).unwrap();
        request
    }
    fn root(tag: &str) -> std::path::PathBuf {
        let base = std::env::var("RRFS_POOL_TEST_ROOT")
            .unwrap_or_else(|_| format!("{}/target", env!("CARGO_MANIFEST_DIR")));
        Path::new(&base).join(format!("{tag}-{}", std::process::id()))
    }
    #[test]
    fn one_worker_and_many_write_the_same_bytes_and_manifest() {
        let root = root("parallel-transport");
        let fields: Vec<serde_json::Value> = (0..40)
            .map(|i| serde_json::json!({"pool":"diag","name":format!("f{i}"),
                "kind": if i % 3 == 0 {"integer"} else {"real"},
                "dimensions":["nVertLevels","nCells"],"shape":[1 + i % 5, 7 + i],
                "initial": if i % 3 == 0 {format!("{}", i as i32 - 20)} else {format!("{}.5d0", i)}}))
            .collect();
        let request = write_request(&root, serde_json::Value::Array(fields));
        run_with(&request, &root.join("serial"), 1).unwrap();
        run_with(&request, &root.join("parallel"), 8).unwrap();
        let serial = fs::read(root.join("serial/manifest.json")).unwrap();
        assert_eq!(serial, fs::read(root.join("parallel/manifest.json")).unwrap());
        let manifest: serde_json::Value = serde_json::from_slice(&serial).unwrap();
        for (i, record) in manifest["fields"].as_array().unwrap().iter().enumerate() {
            assert_eq!(record["name"], format!("f{i}"));
            let name = record["payload"].as_str().unwrap();
            assert_eq!(fs::read(root.join("serial").join(name)).unwrap(),
                       fs::read(root.join("parallel").join(name)).unwrap());
        }
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn the_first_error_in_request_order_is_the_one_reported() {
        let root = root("parallel-transport-error");
        let mut fields: Vec<serde_json::Value> = (0..30)
            .map(|i| serde_json::json!({"pool":"diag","name":format!("ok{i}"),"kind":"real",
                "dimensions":["nCells"],"shape":[64],"initial":"0"}))
            .collect();
        fields[11] = serde_json::json!({"pool":"diag","name":"first","kind":"real",
            "dimensions":["nCells"],"shape":[4],"initial":"not-a-number"});
        fields[23] = serde_json::json!({"pool":"diag","name":"second","kind":"real",
            "dimensions":["nCells"],"shape":[4, 2],"initial":"0"});
        let request = write_request(&root, serde_json::Value::Array(fields));
        for workers in [1, 4, 16] {
            let e = run_with(&request, &root.join(format!("w{workers}")), workers).unwrap_err();
            assert!(e.starts_with("first: invalid real initial value"), "{workers}: {e}");
        }
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn the_worker_count_follows_the_quota_variable() {
        // Single test touching the variable, so no other test races it.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", "3") };
        assert_eq!(workers(), 3);
        unsafe { std::env::set_var("RAYON_NUM_THREADS", "999") };
        assert_eq!(workers(), MAX_WORKERS);
        unsafe { std::env::remove_var("RAYON_NUM_THREADS") };
        assert!((1..=MAX_WORKERS).contains(&workers()));
    }
}

#[cfg(test)]
mod classic_direct_tests {
    use super::*;
    // A classic file written by hand: Time (record) x nCells x nVertLevels scalars, a fixed real,
    // a fixed integer, attributes whose values need padding.  version 1, 2 or 5.
    fn word(h: &mut Vec<u8>, v: u64, wide: bool) {
        if wide {
            h.extend_from_slice(&v.to_be_bytes());
        } else {
            h.extend_from_slice(&(v as u32).to_be_bytes());
        }
    }
    fn name(h: &mut Vec<u8>, s: &str, wide: bool) {
        word(h, s.len() as u64, wide);
        h.extend_from_slice(s.as_bytes());
        while h.len() % 4 != 0 {
            h.push(0);
        }
    }
    const SPECIAL: [u32; 5] = [0x7fc00001, 0x80000000, 0x00000001, 0xff800000, 0x7f800000];
    fn record_word(v: usize) -> u32 {
        // NaN payloads, signed zero, subnormals and infinities travel as words.
        if v % 7 == 3 { SPECIAL[(v / 7) % 5] } else { (1.0f32 + v as f32 / 8.0).to_bits() }
    }
    fn classic_file(version: u8) -> (Vec<u8>, Vec<u64>) {
        let wide = version == 5;
        let offset64 = version != 1;
        let mut h = b"CDF".to_vec();
        h.push(version);
        word(&mut h, 1, wide); // numrecs
        h.extend_from_slice(&10u32.to_be_bytes());
        word(&mut h, 3, wide);
        for (n, l) in [("Time", 0), ("nCells", 5), ("nVertLevels", 3)] {
            name(&mut h, n, wide);
            word(&mut h, l, wide);
        }
        h.extend_from_slice(&12u32.to_be_bytes()); // one global attribute, padded char data
        word(&mut h, 1, wide);
        name(&mut h, "title", wide);
        h.extend_from_slice(&2u32.to_be_bytes());
        word(&mut h, 3, wide);
        h.extend_from_slice(b"abc\0");
        h.extend_from_slice(&11u32.to_be_bytes());
        word(&mut h, 4, wide);
        let mut patches = Vec::new();
        let vars: [(&str, &[u64], u32, u64, bool); 4] = [
            ("ter", &[1], 5, 20, false),
            ("idx", &[1], 4, 20, true),
            ("qv", &[0, 1, 2], 5, 60, false),
            ("qc", &[0, 1, 2], 5, 60, false),
        ];
        for (n, dims, t, vsize, attr) in vars {
            name(&mut h, n, wide);
            word(&mut h, dims.len() as u64, wide);
            for d in dims {
                word(&mut h, *d, wide);
            }
            if attr {
                h.extend_from_slice(&12u32.to_be_bytes());
                word(&mut h, 1, wide);
                name(&mut h, "units", wide);
                h.extend_from_slice(&3u32.to_be_bytes()); // NC_SHORT, one value + 2 pad bytes
                word(&mut h, 1, wide);
                h.extend_from_slice(&[0, 7, 0, 0]);
            } else {
                h.extend_from_slice(&0u32.to_be_bytes());
                word(&mut h, 0, wide);
            }
            h.extend_from_slice(&t.to_be_bytes());
            word(&mut h, vsize, wide);
            patches.push(h.len());
            word(&mut h, 0, offset64);
        }
        let mut begins = Vec::new();
        let mut at = h.len() as u64;
        for ((_, _, _, vsize, _), p) in vars.iter().zip(&patches) {
            begins.push(at);
            let b = if offset64 { at.to_be_bytes().to_vec() } else { (at as u32).to_be_bytes().to_vec() };
            h[*p..*p + b.len()].copy_from_slice(&b);
            at += vsize;
        }
        for i in 0..5u32 {
            h.extend_from_slice(&(100.5f32 + i as f32).to_bits().to_be_bytes());
        }
        for i in 0..5i32 {
            h.extend_from_slice(&(i * 1000 - 7).to_be_bytes());
        }
        for v in 0..30 {
            h.extend_from_slice(&record_word(v).to_be_bytes());
        }
        (h, begins)
    }
    #[test]
    fn the_header_parse_finds_every_variable_where_it_begins() {
        for version in [1u8, 2, 5] {
            let (file, begins) = classic_file(version);
            let (_, vars) = parse_classic(&file).expect("classic header");
            let names: Vec<_> = vars.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, ["ter", "idx", "qv", "qc"], "CDF-{version}");
            for ((_, v), b) in vars.iter().zip(&begins) {
                assert_eq!(v.begin, *b, "CDF-{version}");
            }
            assert!(!vars[0].1.record && !vars[1].1.record && vars[2].1.record && vars[3].1.record);
            assert_eq!(vars[1].1.nc_type, 4);
            // A truncated header is refused (netCDF-C then reads), never misread.
            assert!(parse_classic(&file[..begins[0] as usize - 3]).is_none());
        }
        assert!(parse_classic(b"\x89HDF\r\n\x1a\n").is_none());
    }
    #[test]
    fn direct_reads_write_the_bytes_netcdf_c_reads() {
        let base = std::env::var("RRFS_POOL_TEST_ROOT")
            .unwrap_or_else(|_| format!("{}/target", env!("CARGO_MANIFEST_DIR")));
        for version in [1u8, 2, 5] {
            let root = Path::new(&base).join(format!("classic-direct-{version}-{}", std::process::id()));
            fs::create_dir_all(&root).unwrap();
            let (file, _) = classic_file(version);
            let source = root.join("in.nc");
            fs::write(&source, &file).unwrap();
            let r = serde_json::json!({"schema":"mpas-native-pool-request-v1",
                "sources":[source.to_string_lossy()],
                "fields":[
                    {"pool":"state","name":"scalars","kind":"real",
                     "dimensions":["num_scalars","nVertLevels","nCells"],"shape":[3,3,6],
                     "components":["qv","qc","qr"],"read_input":true},
                    {"pool":"mesh","name":"ter","kind":"real","dimensions":["nCells"],"shape":[6],
                     "initial":"2.5","read_input":true},
                    {"pool":"mesh","name":"idx","kind":"integer","dimensions":["nCells"],"shape":[6],
                     "initial":"-1","read_input":true},
                    {"pool":"state","name":"qv","kind":"real","dimensions":["nVertLevels","nCells"],
                     "shape":[3,6],"read_input":true}]});
            let request = root.join("request.json");
            fs::write(&request, serde_json::to_vec(&r).unwrap()).unwrap();
            run_with_reads(&request, &root.join("netcdf"), 1, false).unwrap();
            for workers in [1, 4] {
                let direct = root.join(format!("direct-{workers}"));
                run_with_reads(&request, &direct, workers, true).unwrap();
                let a = fs::read(root.join("netcdf/manifest.json")).unwrap();
                assert_eq!(a, fs::read(direct.join("manifest.json")).unwrap(), "CDF-{version}");
                let manifest: serde_json::Value = serde_json::from_slice(&a).unwrap();
                for record in manifest["fields"].as_array().unwrap() {
                    let name = record["payload"].as_str().unwrap();
                    assert_eq!(fs::read(root.join("netcdf").join(name)).unwrap(),
                               fs::read(direct.join(name)).unwrap(), "CDF-{version} {name}");
                }
            }
            // qc (component 1) at level 2 of cell 4 is word 4*3+2 of qc's record (after qv's 15).
            let scalars = fs::read(root.join("netcdf/field-0000.bin")).unwrap();
            let word = |i: usize| u32::from_le_bytes(scalars[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(word(1 + 3 * 2 + 9 * 4), record_word(15 + 4 * 3 + 2));
            assert_eq!(word(2), 0, "qr is absent: its words keep the allocation default");
            assert_eq!(word(9 * 5), 0, "the dummy cell keeps the allocation default");
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn scatter_in_row_chunks_places_words_as_one_pass() {
        let field = Field {
            pool: "state".into(), name: "scalars".into(), kind: "real".into(),
            dimensions: vec!["num_scalars".into(), "nVertLevels".into(), "nCells".into()],
            shape: vec![4, 5, 8], components: vec![], initial: String::new(),
            required: true, read_input: true,
        };
        let dims = ["nCells".to_string(), "nVertLevels".to_string()];
        let input: Vec<u8> = (0u32..35).flat_map(|w| (0x3f000000 + w).to_le_bytes()).collect();
        let plan = Plan::new(&dims, &[7, 5], &field, Some(2)).unwrap();
        let mut whole = vec![0xEE; 160 * 4];
        scatter(&plan, &input, 0, Out(whole.as_mut_ptr(), whole.len())).unwrap();
        let mut chunked = vec![0xEE; 160 * 4];
        for (r0, r1) in [(0, 3), (3, 4), (4, 7)] {
            let out = Out(chunked.as_mut_ptr(), chunked.len());
            scatter(&plan, &input[r0 * 20..r1 * 20], r0, out).unwrap();
        }
        assert_eq!(whole, chunked);
        for c in 0..7 {
            for z in 0..5 {
                let dst = 2 + 4 * z + 20 * c;
                assert_eq!(&whole[dst * 4..dst * 4 + 4], &input[(5 * c + z) * 4..(5 * c + z) * 4 + 4]);
            }
        }
    }
}
