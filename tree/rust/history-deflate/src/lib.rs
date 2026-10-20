//! Parallel deflate for hex history frames, byte for byte the file HDF5's own
//! single-thread deflate filter writes.
//!
//! THE STAGE THIS TAKES OFF ONE THREAD (B4, hex global map 2026-10-16): a
//! history frame is a netCDF-4 file whose variables carry the shuffle and
//! zlib level-1 filters.  HDF5 runs the filter pipeline chunk by chunk on the
//! one thread that called it, so a 15 km frame (11.1 GiB) spent 115 s in zlib
//! on one core.
//!
//! HOW, WITHOUT CHANGING ONE BYTE.  The writer that defines the file and
//! calls netCDF is unchanged.  Before it runs, this library is handed the
//! frame's arrays and the chunk layout netCDF-C picks for them, and worker
//! threads build every chunk exactly as HDF5 will hand it to the deflate
//! filter (transposed to the file's order, edge chunks padded with the fill
//! value, byte-shuffled) and compress it with the same `compress2` from the
//! same zlib the HDF5 library links.  For the length of the frame the
//! shuffle and deflate entries of HDF5's own filter table use cached bytes.
//! The deflate entry points at [`filter`], which looks
//! the shuffled chunk up by content (a sampled hash, then a full byte
//! comparison) and returns the precomputed stream.  Anything it does not hold
//! -- a chunk it never saw, an incompressible chunk, a decompression -- goes
//! to HDF5's original deflate function, so the result never depends on this
//! library having guessed right; only the speed does.
//!
//! HDF5 is never called from a worker thread: the workers touch only the
//! caller's arrays and zlib.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const MARKER: &[u8] = b"hex-history-deflate-v2\0";
const H5Z_FILTER_DEFLATE: c_int = 1;
const H5Z_FILTER_SHUFFLE: c_int = 2;
const H5Z_FLAG_REVERSE: c_uint = 0x0100;
const Z_OK: c_int = 0;
const RTLD_NOW: c_int = 2;
const RTLD_NOLOAD: c_int = 4;

type FilterFn =
    unsafe extern "C" fn(c_uint, usize, *const c_uint, usize, *mut usize, *mut *mut c_void) -> usize;
// zlib: int compress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level)
type Compress2 = unsafe extern "C" fn(*mut u8, *mut std::os::raw::c_ulong, *const u8, std::os::raw::c_ulong, c_int) -> c_int;
type RegisterFn = unsafe extern "C" fn(*const H5ZClass2) -> c_int;
type OpenFn = unsafe extern "C" fn() -> c_int;
type AllocFn = unsafe extern "C" fn(usize, c_uint) -> *mut c_void;
type FreeFn = unsafe extern "C" fn(*mut c_void) -> c_int;

/// HDF5's H5Z_class2_t (H5Zpublic.h, H5Z_CLASS_T_VERS 1).
#[repr(C)]
struct H5ZClass2 {
    version: c_int,
    id: c_int,
    encoder_present: c_uint,
    decoder_present: c_uint,
    name: *const c_char,
    can_apply: *const c_void,
    set_local: *const c_void,
    filter: Option<FilterFn>,
}

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

struct Lib {
    /// HDF5's own deflate class (the exported `H5Z_DEFLATE`), never written.
    deflate: *const H5ZClass2,
    shuffle: *const H5ZClass2,
    register: RegisterFn,
    original: FilterFn,
    original_shuffle: FilterFn,
    alloc: AllocFn,
    free: FreeFn,
    compress2: Compress2,
    hdf5_path: String,
}
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

static LIB: OnceLock<Lib> = OnceLock::new();
static ACTIVE: Mutex<Option<Arc<Frame>>> = Mutex::new(None);
static WORKERS: Mutex<Vec<JoinHandle<()>>> = Mutex::new(Vec::new());

/// One variable of the frame, as the caller lays it out (C ABI).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HexJob {
    /// float32 source, C order: (levels, cells), or (cells,) when levels == 0.
    pub src: *const f32,
    pub levels: u64,
    pub cells: u64,
    /// netCDF-C's chunk for the file variable (1, cells, levels) or (1, cells).
    pub chunk_cells: u64,
    pub chunk_levels: u64,
    pub level: c_int,
    pub shuffle: c_int,
    pub fill_bits: u32,
}
unsafe impl Send for HexJob {}
unsafe impl Sync for HexJob {}

struct Entry {
    level: c_int,
    unshuffled: Option<Vec<u8>>,
    shuffled: Vec<u8>,
    compressed: Vec<u8>,
}

#[derive(Default)]
struct Store {
    map: HashMap<u64, Vec<Arc<Entry>>>,
    shuffle: HashMap<u64, Vec<Weak<Entry>>>,
    bytes: usize,
    peak: usize,
    done_workers: usize,
    blocked: usize,
}

struct Frame {
    jobs: Vec<HexJob>,
    work: Vec<(u32, u64, u64)>,
    next: AtomicUsize,
    store: Mutex<Store>,
    cv: Condvar,
    budget: usize,
    workers: usize,
    cancel: AtomicBool,
    hits: AtomicU64,
    misses: AtomicU64,
    shuffle_hits: AtomicU64,
    shuffle_misses: AtomicU64,
    deferred: AtomicU64,
    hit_bytes: AtomicU64,
    inplace_hits: AtomicU64,
    worker_ns: AtomicU64,
    filter_wait_ns: AtomicU64,
    started: Instant,
}

fn err_out(buf: *mut c_char, len: usize, text: &str) {
    if buf.is_null() || len == 0 {
        return;
    }
    let bytes = text.as_bytes();
    let n = bytes.len().min(len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
}

unsafe fn last_dlerror() -> String {
    let e = dlerror();
    if e.is_null() { "unknown dlerror".into() } else { CStr::from_ptr(e).to_string_lossy().into_owned() }
}

unsafe fn sym(handle: *mut c_void, name: &[u8]) -> Result<*mut c_void, String> {
    let p = dlsym(handle, name.as_ptr() as *const c_char);
    if p.is_null() {
        Err(format!("{} not found: {}", String::from_utf8_lossy(&name[..name.len() - 1]), last_dlerror()))
    } else {
        Ok(p)
    }
}

/// The ABI marker the Python seam checks before it trusts this library.
#[no_mangle]
pub extern "C" fn hex_hd_version() -> *const c_char {
    MARKER.as_ptr() as *const c_char
}

/// Bind to the HDF5 library already loaded in this process (the one netCDF
/// uses, named by path) and to the zlib that library resolves `compress2`
/// from.  Returns 0, or -1 with the reason in `err`.
#[no_mangle]
pub unsafe extern "C" fn hex_hd_init(hdf5_path: *const c_char, err: *mut c_char, errlen: usize) -> c_int {
    if LIB.get().is_some() {
        return 0;
    }
    let path = CStr::from_ptr(hdf5_path).to_string_lossy().into_owned();
    let handle = dlopen(hdf5_path, RTLD_NOW | RTLD_NOLOAD);
    if handle.is_null() {
        err_out(err, errlen, &format!("{path} is not loaded in this process: {}", last_dlerror()));
        return -1;
    }
    let result = (|| -> Result<Lib, String> {
        // H5Z_register is the library's internal registration, which replaces
        // the table entry of an id already registered (the public H5Zregister
        // refuses predefined ids); H5Z_DEFLATE is the exported deflate class.
        let open: OpenFn = std::mem::transmute(sym(handle, b"H5open\0")?);
        let register: RegisterFn = std::mem::transmute(sym(handle, b"H5Z_register\0")?);
        let class = sym(handle, b"H5Z_DEFLATE\0")? as *const H5ZClass2;
        let shuffle = sym(handle, b"H5Z_SHUFFLE\0")? as *const H5ZClass2;
        let alloc: AllocFn = std::mem::transmute(sym(handle, b"H5allocate_memory\0")?);
        let free: FreeFn = std::mem::transmute(sym(handle, b"H5free_memory\0")?);
        // dlsym on the library's handle searches it and then its dependencies,
        // so this is the compress2 HDF5's own deflate filter calls.
        let compress2: Compress2 = std::mem::transmute(sym(handle, b"compress2\0")?);
        if open() < 0 {
            return Err("H5open failed".into());
        }
        let c = &*class;
        if c.version != 1 || c.id != H5Z_FILTER_DEFLATE {
            return Err(format!("deflate class has version {} id {}, not an H5Z_class2_t version 1", c.version, c.id));
        }
        let original = c.filter.ok_or("the deflate class has no filter function")?;
        let s = &*shuffle;
        if s.version != 1 || s.id != H5Z_FILTER_SHUFFLE {
            return Err("shuffle class is not H5Z_class2_t version 1".into());
        }
        let original_shuffle = s.filter.ok_or("the shuffle class has no filter function")?;
        Ok(Lib { deflate: class, shuffle, register, original, original_shuffle,
                 alloc, free, compress2, hdf5_path: path.clone() })
    })();
    match result {
        Ok(lib) => {
            let _ = LIB.set(lib);
            0
        }
        Err(e) => {
            err_out(err, errlen, &e);
            -1
        }
    }
}

fn sample_hash(b: &[u8]) -> u64 {
    let n = b.len();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    if n >= 8 {
        let samples = 256usize;
        let span = n - 8;
        for i in 0..=samples {
            let at = span * i / samples;
            let mut w = [0u8; 8];
            w.copy_from_slice(&b[at..at + 8]);
            h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0x0100_0000_01b3).rotate_left(23);
        }
    } else {
        for &x in b {
            h = (h ^ x as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// The chunk exactly as HDF5's pipeline hands it to deflate: file order
/// (1, cells, levels) row-major, out-of-extent elements set to the fill
/// value, then the shuffle filter's byte planes.
fn build_chunk(job: &HexJob, ci: u64, li: u64) -> (Vec<u8>, Option<Vec<u8>>) {
    let three_d = job.levels > 0;
    let cc = job.chunk_cells as usize;
    let lc = if three_d { job.chunk_levels as usize } else { 1 };
    let n = cc * lc;
    let mut out = vec![0u8; n * 4];
    let mut unshuffled = (job.shuffle != 0).then(|| vec![0u8; n * 4]);
    let c0 = (ci as usize) * cc;
    let l0 = (li as usize) * lc;
    let cells = job.cells as usize;
    let levels = if three_d { job.levels as usize } else { 1 };
    let fill = job.fill_bits.to_le_bytes();
    let shuffle = job.shuffle != 0 && n > 1;
    let mut put = |e: usize, bits: [u8; 4]| {
        if let Some(raw) = &mut unshuffled {
            raw[4 * e..4 * e + 4].copy_from_slice(&bits);
        }
        if shuffle {
            out[e] = bits[0];
            out[n + e] = bits[1];
            out[2 * n + e] = bits[2];
            out[3 * n + e] = bits[3];
        } else {
            out[4 * e..4 * e + 4].copy_from_slice(&bits);
        }
    };
    // Keep a small tile of every level resident. Scanning the full cell axis
    // at each level repeatedly evicts the strided destination cache lines.
    for c_start in (0..cc).step_by(64) {
        for l in 0..lc {
            let gl = l0 + l;
            for c in c_start..(c_start + 64).min(cc) {
                let gc = c0 + c;
                let bits = if gc < cells && gl < levels {
                    let index = if three_d { gl * cells + gc } else { gc };
                    unsafe { (*job.src.add(index)).to_bits().to_le_bytes() }
                } else {
                    fill
                };
                put(c * lc + l, bits);
            }
        }
    }
    (out, unshuffled)
}

fn worker(frame: Arc<Frame>, lib: &'static Lib) {
    loop {
        if frame.cancel.load(Ordering::Relaxed) {
            break;
        }
        let i = frame.next.fetch_add(1, Ordering::Relaxed);
        if i >= frame.work.len() {
            break;
        }
        {
            let mut s = frame.store.lock().unwrap();
            while s.bytes >= frame.budget && !frame.cancel.load(Ordering::Relaxed) {
                s.blocked += 1;
                frame.cv.notify_all();
                s = frame.cv.wait(s).unwrap();
                s.blocked -= 1;
            }
        }
        let started = Instant::now();
        let (j, ci, li) = frame.work[i];
        let job = &frame.jobs[j as usize];
        let (shuffled, unshuffled) = build_chunk(job, ci, li);
        // Destination capacity = the chunk's own size: a stream that fits is
        // the one compress2 writes at any larger capacity too; one that does
        // not is left to HDF5's own function, whatever it does with it.
        let mut compressed = vec![0u8; shuffled.len()];
        let mut dest_len = compressed.len() as std::os::raw::c_ulong;
        let rc = unsafe {
            (lib.compress2)(compressed.as_mut_ptr(), &mut dest_len, shuffled.as_ptr(),
                            shuffled.len() as std::os::raw::c_ulong, job.level)
        };
        if rc != Z_OK {
            frame.worker_ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            frame.deferred.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        compressed.truncate(dest_len as usize);
        compressed.shrink_to_fit();
        frame.worker_ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let key = sample_hash(&shuffled);
        let size = shuffled.len() + compressed.len() + unshuffled.as_ref().map_or(0, Vec::len);
        // A single oversized chunk stays on HDF5's normal compression path.
        // The budget bounds retained entries, excluding worker scratch arrays.
        if size > frame.budget {
            frame.deferred.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let mut s = frame.store.lock().unwrap();
        while s.bytes > frame.budget - size && !frame.cancel.load(Ordering::Relaxed) {
            s.blocked += 1;
            frame.cv.notify_all();
            s = frame.cv.wait(s).unwrap();
            s.blocked -= 1;
        }
        if frame.cancel.load(Ordering::Relaxed) {
            break;
        }
        let entry = Arc::new(Entry { level: job.level, unshuffled, shuffled, compressed });
        if let Some(raw) = &entry.unshuffled {
            s.shuffle.entry(sample_hash(raw)).or_default().push(Arc::downgrade(&entry));
        }
        s.map.entry(key).or_default().push(entry);
        s.bytes += size;
        s.peak = s.peak.max(s.bytes);
        frame.cv.notify_all();
    }
    let mut s = frame.store.lock().unwrap();
    s.done_workers += 1;
    frame.cv.notify_all();
}

/// Shuffle is still a serial byte-interleave pass after parallel deflate
/// precomputation. Replace it with a copy of the worker's exact bytes. Full
/// comparisons include fill padding; every unprepared chunk falls through.
unsafe extern "C" fn shuffle_filter(
    flags: c_uint, cd_nelmts: usize, cd_values: *const c_uint, nbytes: usize,
    buf_size: *mut usize, buf: *mut *mut c_void,
) -> usize {
    let lib = match LIB.get() { Some(lib) => lib, None => return 0 };
    let original = lib.original_shuffle;
    if flags & H5Z_FLAG_REVERSE != 0 || cd_nelmts < 1 || cd_values.is_null()
        || *cd_values != 4 || buf.is_null() || (*buf).is_null()
        || buf_size.is_null() || *buf_size < nbytes {
        return original(flags, cd_nelmts, cd_values, nbytes, buf_size, buf);
    }
    let frame = match ACTIVE.lock().unwrap().clone() {
        Some(f) => f,
        None => return original(flags, cd_nelmts, cd_values, nbytes, buf_size, buf),
    };
    let input = std::slice::from_raw_parts(*buf as *const u8, nbytes);
    let key = sample_hash(input);
    let waited = Instant::now();
    let mut found = None;
    {
        let mut s = frame.store.lock().unwrap();
        loop {
            if let Some(bucket) = s.shuffle.get_mut(&key) {
                bucket.retain(|e| e.strong_count() > 0);
                if let Some((pos, entry)) = bucket.iter().enumerate().find_map(|(pos, weak)| {
                    let entry = weak.upgrade()?;
                    (entry.unshuffled.as_deref() == Some(input)).then_some((pos, entry))
                }) {
                    bucket.swap_remove(pos);
                    found = Some(entry);
                    break;
                }
            }
            let finished = s.done_workers >= frame.workers;
            // Insertion backpressure can block below the hard byte limit.
            // Fall through so the consumer can drain retained chunks.
            let stalled = s.blocked > 0;
            if finished || stalled || waited.elapsed() > Duration::from_secs(120) {
                break;
            }
            s = frame.cv.wait_timeout(s, Duration::from_millis(500)).unwrap().0;
        }
    }
    match found {
        Some(entry) => {
            std::ptr::copy_nonoverlapping(entry.shuffled.as_ptr(), *buf as *mut u8, nbytes);
            frame.shuffle_hits.fetch_add(1, Ordering::Relaxed);
            nbytes
        }
        None => {
            frame.shuffle_misses.fetch_add(1, Ordering::Relaxed);
            original(flags, cd_nelmts, cd_values, nbytes, buf_size, buf)
        }
    }
}

/// The replacement entry of HDF5's deflate filter while a frame is open.
unsafe extern "C" fn filter(
    flags: c_uint, cd_nelmts: usize, cd_values: *const c_uint, nbytes: usize, buf_size: *mut usize,
    buf: *mut *mut c_void,
) -> usize {
    let lib = match LIB.get() {
        Some(lib) => lib,
        None => return 0,
    };
    let original = lib.original;
    if flags & H5Z_FLAG_REVERSE != 0 || cd_nelmts < 1 || buf.is_null() || (*buf).is_null() || buf_size.is_null() {
        return original(flags, cd_nelmts, cd_values, nbytes, buf_size, buf);
    }
    let frame = match ACTIVE.lock().unwrap().clone() {
        Some(f) => f,
        None => return original(flags, cd_nelmts, cd_values, nbytes, buf_size, buf),
    };
    let level = *cd_values as c_int;
    let input = std::slice::from_raw_parts(*buf as *const u8, nbytes);
    let key = sample_hash(input);
    let waited = Instant::now();
    let mut found: Option<Arc<Entry>> = None;
    {
        let mut s = frame.store.lock().unwrap();
        loop {
            if let Some(bucket) = s.map.get_mut(&key) {
                if let Some(pos) = bucket.iter().position(|e| e.level == level && e.shuffled.as_slice() == input) {
                    let entry = bucket.swap_remove(pos);
                    if bucket.is_empty() {
                        s.map.remove(&key);
                    }
                    s.bytes -= entry.shuffled.len() + entry.compressed.len()
                        + entry.unshuffled.as_ref().map_or(0, Vec::len);
                    frame.cv.notify_all();
                    found = Some(entry);
                    break;
                }
            }
            let finished = s.done_workers >= frame.workers;
            // A completed chunk may be waiting for room even below the byte
            // limit. Compress a cache miss normally so the writer can advance
            // and consume the retained entries that are blocking the workers.
            let stalled = s.blocked > 0;
            if finished || stalled || waited.elapsed() > Duration::from_secs(120) {
                break;
            }
            s = frame.cv.wait_timeout(s, Duration::from_millis(500)).unwrap().0;
        }
    }
    frame.filter_wait_ns.fetch_add(waited.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let entry = match found {
        Some(e) => e,
        None => {
            frame.misses.fetch_add(1, Ordering::Relaxed);
            return original(flags, cd_nelmts, cd_values, nbytes, buf_size, buf);
        }
    };
    let len = entry.compressed.len();
    if *buf_size >= len {
        // The matched compressed stream fits in the existing HDF5 allocation.
        // Keep its pointer and capacity; HDF5 consumes only the returned length.
        std::ptr::copy_nonoverlapping(entry.compressed.as_ptr(), *buf as *mut u8, len);
        frame.inplace_hits.fetch_add(1, Ordering::Relaxed);
    } else {
        let out = (lib.alloc)(len, 0) as *mut u8;
        if out.is_null() {
            frame.misses.fetch_add(1, Ordering::Relaxed);
            return original(flags, cd_nelmts, cd_values, nbytes, buf_size, buf);
        }
        std::ptr::copy_nonoverlapping(entry.compressed.as_ptr(), out, len);
        (lib.free)(*buf);
        *buf = out as *mut c_void;
        *buf_size = len;
    }
    frame.hits.fetch_add(1, Ordering::Relaxed);
    frame.hit_bytes.fetch_add(nbytes as u64, Ordering::Relaxed);
    len
}

/// Open a frame: start `threads` workers over `jobs` and point HDF5's
/// shuffle and deflate entries at the exact-byte cache filters.  The arrays must stay alive and unchanged
/// until [`hex_hd_end`] returns.
#[no_mangle]
pub unsafe extern "C" fn hex_hd_begin(
    jobs: *const HexJob, njobs: usize, threads: c_uint, budget_bytes: u64, err: *mut c_char, errlen: usize,
) -> c_int {
    let lib = match LIB.get() {
        Some(lib) => lib,
        None => {
            err_out(err, errlen, "hex_hd_init has not bound an HDF5 library");
            return -1;
        }
    };
    if ACTIVE.lock().unwrap().is_some() {
        err_out(err, errlen, "a frame is already open");
        return -1;
    }
    let jobs: Vec<HexJob> = std::slice::from_raw_parts(jobs, njobs).to_vec();
    let mut work = Vec::new();
    for (j, job) in jobs.iter().enumerate() {
        if job.src.is_null() || job.cells == 0 || job.chunk_cells == 0 || (job.levels > 0 && job.chunk_levels == 0) {
            err_out(err, errlen, &format!("job {j} has an empty array or chunk"));
            return -1;
        }
        let nc = job.cells.div_ceil(job.chunk_cells);
        let nl = if job.levels > 0 { job.levels.div_ceil(job.chunk_levels) } else { 1 };
        for ci in 0..nc {
            for li in 0..nl {
                work.push((j as u32, ci, li));
            }
        }
    }
    let workers = (threads.max(1) as usize).min(work.len().max(1));
    let frame = Arc::new(Frame {
        jobs, work, next: AtomicUsize::new(0), store: Mutex::new(Store::default()), cv: Condvar::new(),
        budget: budget_bytes.max(1 << 26) as usize, workers, cancel: AtomicBool::new(false),
        hits: AtomicU64::new(0), misses: AtomicU64::new(0), deferred: AtomicU64::new(0),
        shuffle_hits: AtomicU64::new(0), shuffle_misses: AtomicU64::new(0),
        hit_bytes: AtomicU64::new(0), inplace_hits: AtomicU64::new(0), worker_ns: AtomicU64::new(0), filter_wait_ns: AtomicU64::new(0),
        started: Instant::now(),
    });
    let mut handles = WORKERS.lock().unwrap();
    for _ in 0..workers {
        let f = frame.clone();
        handles.push(std::thread::spawn(move || worker(f, lib)));
    }
    *ACTIVE.lock().unwrap() = Some(frame);
    let original = &*lib.deflate;
    // Leaked on purpose: the table holds a copy, and this is a few bytes per frame.
    let mine = Box::leak(Box::new(H5ZClass2 {
        version: original.version, id: original.id, encoder_present: original.encoder_present,
        decoder_present: original.decoder_present, name: original.name, can_apply: original.can_apply,
        set_local: original.set_local, filter: Some(filter),
    }));
    if (lib.register)(mine) < 0 {
        let _ = hex_hd_end(std::ptr::null_mut(), 0);
        err_out(err, errlen, "H5Z_register refused the replacement deflate entry");
        return -1;
    }
    let original = &*lib.shuffle;
    let mine = Box::leak(Box::new(H5ZClass2 {
        version: original.version, id: original.id, encoder_present: original.encoder_present,
        decoder_present: original.decoder_present, name: original.name, can_apply: original.can_apply,
        set_local: original.set_local, filter: Some(shuffle_filter),
    }));
    if (lib.register)(mine) < 0 {
        let _ = hex_hd_end(std::ptr::null_mut(), 0);
        err_out(err, errlen, "H5Z_register refused the replacement shuffle entry");
        return -1;
    }
    0
}

/// Close the frame: restore HDF5's shuffle and deflate entries, stop and join the
/// workers, and write the frame's counts as JSON into `out`.
#[no_mangle]
pub unsafe extern "C" fn hex_hd_end(out: *mut c_char, outlen: usize) -> c_int {
    let lib = match LIB.get() {
        Some(lib) => lib,
        None => return -1,
    };
    let restored_deflate = (lib.register)(lib.deflate);
    let restored_shuffle = (lib.register)(lib.shuffle);
    let frame = ACTIVE.lock().unwrap().take();
    let Some(frame) = frame else {
        return -1;
    };
    frame.cancel.store(true, Ordering::Relaxed);
    frame.cv.notify_all();
    for h in WORKERS.lock().unwrap().drain(..) {
        let _ = h.join();
    }
    let s = frame.store.lock().unwrap();
    let left: usize = s.map.values().map(|v| v.len()).sum();
    let text = format!(
        concat!("{{\"schema\":\"gpuwm-hex.history-deflate/v1\",\"hdf5\":\"{}\",\"workers\":{},\"chunks\":{},",
                "\"hits\":{},\"misses\":{},\"deferred\":{},\"unused\":{},\"hit_bytes\":{},",
                "\"inplace_hits\":{},\"shuffle_hits\":{},\"shuffle_misses\":{},",
                "\"worker_seconds\":{:.3},\"filter_wait_seconds\":{:.3},\"frame_seconds\":{:.3},\"peak_bytes\":{},\"budget_bytes\":{},\"restored\":{}}}"),
        lib.hdf5_path.replace('\\', "\\\\").replace('"', "\\\""), frame.workers, frame.work.len(),
        frame.hits.load(Ordering::Relaxed), frame.misses.load(Ordering::Relaxed),
        frame.deferred.load(Ordering::Relaxed), left, frame.hit_bytes.load(Ordering::Relaxed),
        frame.inplace_hits.load(Ordering::Relaxed),
        frame.shuffle_hits.load(Ordering::Relaxed), frame.shuffle_misses.load(Ordering::Relaxed),
        frame.worker_ns.load(Ordering::Relaxed) as f64 * 1e-9,
        frame.filter_wait_ns.load(Ordering::Relaxed) as f64 * 1e-9,
        frame.started.elapsed().as_secs_f64(), s.peak, frame.budget, restored_deflate >= 0 && restored_shuffle >= 0,
    );
    err_out(out, outlen, &text);
    0
}
