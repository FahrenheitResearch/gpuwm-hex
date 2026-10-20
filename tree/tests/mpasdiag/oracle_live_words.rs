//! Rust field decoding and exact comparison for f1 ORCL0001 captures.
//! Format: f1-native-tap tools/native_physics_reference/oracle/FORMAT.md.
//! Scientific words are never interpreted or arithmetically transformed here.
use std::{collections::{HashMap, HashSet}, ffi::{CStr, CString, c_char, c_void}, fs, path::Path};
const FIXED: usize = 8 + 21 * 4 + 3 * 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Metadata {
    header: [i32; 21], strings: [u8; 768], payload: *const u8,
    payload_bytes: usize, ids: *const u8,
}
struct Record { raw: Vec<u8>, metadata: Metadata, text: Option<CString> }

fn decode(path: &Path) -> Result<Record, String> {
    let raw = fs::read(path).map_err(|e| e.to_string())?;
    if raw.len() < FIXED || &raw[..8] != b"ORCL0001" { return Err("invalid ORCL0001 header".into()); }
    let mut h = [0; 21];
    for (i, v) in h.iter_mut().enumerate() {
        *v = i32::from_le_bytes(raw[8 + i * 4..12 + i * 4].try_into().unwrap());
    }
    if !(0..=5).contains(&h[2]) || h[3..8].iter().any(|v| *v < 0) || h[16] < 0 || h[19] != 256 || h[20] != 0 {
        return Err("malformed dimensions/version".into());
    }
    let width = match h[0] { 1 | 3 | 4 => 4, 2 => 8, 5 if h[1] > 0 => h[1], _ => return Err("invalid type/width".into()) };
    if width != h[1] { return Err("type width mismatch".into()); }
    if h[17] < 0 || h[17] > h[2] || h[18] < 0 || h[18] > h[2] || (h[17] > 0 && h[17] == h[18]) {
        return Err("invalid cell/level axes".into());
    }
    let ids = if h[17] > 0 { h[2 + h[17] as usize] } else { 0 };
    if h[16] != ids { return Err("ID count mismatch".into()); }
    if h[0] != 5 && h[3 + h[2] as usize..8].iter().any(|v| *v != 1) {
        return Err("inactive extents must equal one".into());
    }
    let elements = h[3..8].iter().try_fold(1usize, |n, v| n.checked_mul(*v as usize)).ok_or("shape overflow")?;
    let bytes = elements.checked_mul(width as usize).ok_or("payload overflow")?;
    let offset = (ids as usize).checked_mul(4).and_then(|v| FIXED.checked_add(v)).ok_or("ID overflow")?;
    if offset.checked_add(bytes) != Some(raw.len()) { return Err("truncated or trailing payload".into()); }
    let metadata = Metadata { header: h, strings: raw[92..FIXED].try_into().unwrap(),
                              payload: unsafe { raw.as_ptr().add(offset) }, payload_bytes: bytes,
                              ids: unsafe { raw.as_ptr().add(FIXED) } };
    let text = if h[0] == 5 && h[2] == 0 {
        let value = std::str::from_utf8(&raw[offset..]).map_err(|_| "invalid character input")?;
        Some(CString::new(value.trim_end_matches(' ')).map_err(|_| "NUL in character input")?)
    } else { None };
    Ok(Record { raw, metadata, text })
}

// Handles belong to one caller; close only after all borrowed buffers are released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_open(path: *const c_char) -> *mut c_void {
    if path.is_null() || cfg!(target_endian = "big") { return std::ptr::null_mut(); }
    let path = unsafe { CStr::from_ptr(path) };
    let Ok(path) = path.to_str() else { return std::ptr::null_mut(); };
    match decode(Path::new(path)) {
        Ok(record) => Box::into_raw(Box::new(record)).cast(),
        Err(error) => { eprintln!("oracle capture {path}: {error}"); std::ptr::null_mut() }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_metadata(handle: *const c_void, output: *mut Metadata) -> i32 {
    if handle.is_null() || output.is_null() { return -1; }
    unsafe { output.write((*handle.cast::<Record>()).metadata); }
    0
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_text(handle: *const c_void) -> *const c_char {
    if handle.is_null() { return std::ptr::null(); }
    unsafe { &*handle.cast::<Record>() }.text.as_ref().map_or(std::ptr::null(), |s| s.as_ptr())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_compare(handle: *const c_void, payload: *const u8, bytes: usize, different: *mut usize) -> i32 {
    if handle.is_null() || payload.is_null() || different.is_null() { return -1; }
    let record = unsafe { &*handle.cast::<Record>() };
    if bytes != record.metadata.payload_bytes { return -2; }
    let width = record.metadata.header[1] as usize;
    let reference = &record.raw[record.raw.len() - bytes..];
    let actual = unsafe { std::slice::from_raw_parts(payload, bytes) };
    let count = reference.chunks_exact(width).zip(actual.chunks_exact(width)).filter(|(a, b)| a != b).count();
    unsafe { different.write(count); }
    0
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_close(handle: *mut c_void) {
    if !handle.is_null() { drop(unsafe { Box::from_raw(handle.cast::<Record>()) }); }
}

// Compare owned columns by global ID; zero IDs mark excluded halo/dummy cells.
// Array layout movement and byte comparisons belong to Rust, never Python.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_compare_owned(reference: *const c_void, candidate: *const c_void,
    cells: *mut usize, words: *mut usize, different: *mut usize) -> i32 {
    if reference.is_null() || candidate.is_null() || cells.is_null() || words.is_null() || different.is_null() { return -1; }
    let (a, b) = unsafe { (&*reference.cast::<Record>(), &*candidate.cast::<Record>()) };
    let (ha, hb) = (&a.metadata.header, &b.metadata.header);
    if ha[..3] != hb[..3] || ha[17..19] != hb[17..19] || ha[17] == 0 { return -2; }
    let axis = ha[17] as usize - 1;
    for dim in 0..ha[2] as usize {
        if (dim != axis && ha[3 + dim] != hb[3 + dim]) || ha[8 + dim] != hb[8 + dim] { return -2; }
    }
    let ids = |record: &Record| -> Vec<i32> {
        record.raw[FIXED..FIXED + record.metadata.header[16] as usize * 4]
            .chunks_exact(4).map(|v| i32::from_le_bytes(v.try_into().unwrap())).collect()
    };
    let mut index = HashMap::new();
    for (col, id) in ids(a).into_iter().enumerate() {
        if id < 0 || (id > 0 && index.insert(id, col).is_some()) { return -3; }
    }
    let mut seen = HashSet::new();
    let width = ha[1] as usize;
    let inner = ha[3..3 + axis].iter().map(|v| *v as usize).product::<usize>();
    let outer = ha[4 + axis..3 + ha[2] as usize].iter().map(|v| *v as usize).product::<usize>();
    let pa = &a.raw[a.raw.len() - a.metadata.payload_bytes..];
    let pb = &b.raw[b.raw.len() - b.metadata.payload_bytes..];
    let mut count = 0;
    for (col, id) in ids(b).into_iter().enumerate() {
        if id == 0 { continue; }
        if id < 0 || !seen.insert(id) { return -3; }
        let Some(&reference_col) = index.get(&id) else { return -4; };
        for block in 0..outer {
            let oa = (block * ha[3 + axis] as usize + reference_col) * inner * width;
            let ob = (block * hb[3 + axis] as usize + col) * inner * width;
            count += pa[oa..oa + inner * width].chunks_exact(width)
                .zip(pb[ob..ob + inner * width].chunks_exact(width)).filter(|(a,b)| a != b).count();
        }
    }
    if seen.is_empty() { return -5; }
    unsafe { cells.write(seen.len()); words.write(seen.len() * inner * outer); different.write(count); }
    0
}

// Compare live ABI storage on the oracle's positive global IDs. Zero IDs are
// native halo/dummy storage, never physical columns (FORMAT.md).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_compare_live_owned(reference: *const c_void,
    payload: *const u8, bytes: usize, cells: *mut usize, words: *mut usize,
    different: *mut usize) -> i32 {
    if reference.is_null() || payload.is_null() || cells.is_null() || words.is_null() || different.is_null() { return -1; }
    let record = unsafe { &*reference.cast::<Record>() };
    let h = &record.metadata.header;
    if bytes != record.metadata.payload_bytes || h[17] == 0 { return -2; }
    let axis = h[17] as usize - 1;
    let width = h[1] as usize;
    let inner = h[3..3 + axis].iter().map(|v| *v as usize).product::<usize>();
    let outer = h[4 + axis..3 + h[2] as usize].iter().map(|v| *v as usize).product::<usize>();
    let actual = unsafe { std::slice::from_raw_parts(payload, bytes) };
    let expected = &record.raw[record.raw.len() - bytes..];
    let mut seen = HashSet::new();
    let mut count = 0;
    for (col, raw_id) in record.raw[FIXED..FIXED + h[16] as usize * 4].chunks_exact(4).enumerate() {
        let id = i32::from_le_bytes(raw_id.try_into().unwrap());
        if id == 0 { continue; }
        if id < 0 || !seen.insert(id) { return -3; }
        for block in 0..outer {
            let offset = (block * h[3 + axis] as usize + col) * inner * width;
            count += expected[offset..offset + inner * width].chunks_exact(width)
                .zip(actual[offset..offset + inner * width].chunks_exact(width)).filter(|(a,b)| a != b).count();
        }
    }
    unsafe { cells.write(seen.len()); words.write(seen.len() * inner * outer); different.write(count); }
    0
}
