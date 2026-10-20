//! ORCL0001 record I/O for Tiedtke's f1 replay adapter. No weather arithmetic.
use std::{cell::RefCell, ffi::{c_char, CStr, CString}, fs, io::{self, Write}, ptr};
const FIXED: usize = 8 + 21 * 4 + 3 * 256;
pub struct Record { header: [i32; 21], strings: [u8; 768], ids: Vec<i32>, words: Vec<u32> }
thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn bad(s: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, s) }
fn failure(e: impl std::fmt::Display) {
    ERROR.with(|v| *v.borrow_mut() = CString::new(e.to_string().replace('\0', "?")).unwrap());
}
fn validate(h: &[i32; 21], bytes: usize) -> io::Result<(usize, usize)> {
    if !matches!(h[0], 1 | 3 | 4) || h[1] != 4 || !(0..=5).contains(&h[2])
        || h[3..8].iter().any(|v| *v < 0) || h[16] < 0 || h[19] != 256 || h[20] != 0 {
        return Err(bad("Tiedtke requires valid four-byte real/integer/logical records"));
    }
    if h[3 + h[2] as usize..8].iter().any(|v| *v != 1)
        || h[17] < 0 || h[17] > h[2] || h[18] < 0 || h[18] > h[2]
        || (h[17] > 0 && h[17] == h[18]) {
        return Err(bad("invalid record axes or inactive dimensions"));
    }
    let ids = if h[17] == 0 { 0 } else { h[2 + h[17] as usize] as usize };
    if h[16] as usize != ids { return Err(bad("global ID count disagrees with cell dimension")); }
    let words = h[3..8].iter().try_fold(1usize, |a, v| a.checked_mul(*v as usize))
        .ok_or_else(|| bad("record shape overflow"))?;
    let expected = ids.checked_add(words).and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(FIXED)).ok_or_else(|| bad("record size overflow"))?;
    if expected != bytes { return Err(bad("truncated or trailing record payload")); }
    Ok((ids, words))
}
fn decode(raw: &[u8]) -> io::Result<Record> {
    if raw.len() < FIXED || &raw[..8] != b"ORCL0001" { return Err(bad("invalid oracle magic/header")); }
    let mut header = [0; 21];
    for (i, v) in header.iter_mut().enumerate() {
        *v = i32::from_le_bytes(raw[8 + 4*i..12 + 4*i].try_into().unwrap());
    }
    let (ids, _) = validate(&header, raw.len())?;
    let offset = FIXED + 4 * ids;
    Ok(Record { header, strings: raw[92..FIXED].try_into().unwrap(),
        ids: raw[FIXED..offset].chunks_exact(4).map(|v| i32::from_le_bytes(v.try_into().unwrap())).collect(),
        words: raw[offset..].chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect() })
}
fn encode(record: &Record, payload: &[u32], post: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(FIXED + 4 * (record.ids.len() + payload.len()));
    out.extend_from_slice(b"ORCL0001");
    for v in record.header { out.extend_from_slice(&v.to_le_bytes()); }
    let mut strings = record.strings;
    if post { strings[256..512].fill(b' '); strings[256..260].copy_from_slice(b"post"); }
    out.extend_from_slice(&strings);
    for v in &record.ids { out.extend_from_slice(&v.to_le_bytes()); }
    for v in payload { out.extend_from_slice(&v.to_le_bytes()); }
    out
}
unsafe fn path<'a>(p: *const c_char) -> io::Result<&'a str> {
    if p.is_null() { return Err(bad("null path")); }
    CStr::from_ptr(p).to_str().map_err(|_| bad("non-UTF8 path"))
}
#[no_mangle]
pub extern "C" fn tk_error() -> *const c_char { ERROR.with(|v| v.borrow().as_ptr()) }
#[no_mangle]
pub unsafe extern "C" fn tk_load(p: *const c_char) -> *mut Record {
    match path(p).and_then(|p| fs::read(p)).and_then(|r| decode(&r)) {
        Ok(r) => Box::into_raw(Box::new(r)), Err(e) => { failure(e); ptr::null_mut() }
    }
}
#[no_mangle]
pub unsafe extern "C" fn tk_free(p: *mut Record) { if !p.is_null() { drop(Box::from_raw(p)); } }
#[no_mangle]
pub unsafe extern "C" fn tk_meta(p: *const Record, header: *mut i32) {
    ptr::copy_nonoverlapping((*p).header.as_ptr(), header, 21);
}
#[no_mangle]
pub unsafe extern "C" fn tk_data(p: *const Record) -> *const u32 { (*p).words.as_ptr() }
#[no_mangle]
pub unsafe extern "C" fn tk_len(p: *const Record) -> usize { (*p).words.len() }
#[no_mangle]
pub unsafe extern "C" fn tk_scalar(p: *const Record, value: *mut f64) -> i32 {
    let r = &*p;
    if r.header[2] != 0 || r.words.len() != 1 {
        failure("scalar argument has non-scalar schema"); return -1;
    }
    *value = match r.header[0] {
        1 => f32::from_bits(r.words[0]) as f64,
        3 => (r.words[0] as i32) as f64,
        4 if r.words[0] <= 1 => r.words[0] as f64,
        _ => { failure("noncanonical scalar logical"); return -1; }
    };
    0
}
#[no_mangle]
pub unsafe extern "C" fn tk_write_post(p: *const Record, filename: *const c_char,
                                       data: *const u32, words: usize) -> i32 {
    let r = &*p;
    if words != r.words.len() || (words != 0 && data.is_null()) {
        failure("output payload disagrees with input record schema"); return -1;
    }
    let payload = if words == 0 { &[] } else { std::slice::from_raw_parts(data, words) };
    let result = path(filename).and_then(|name| {
        fs::OpenOptions::new().write(true).create_new(true).open(name)
            .and_then(|mut f| f.write_all(&encode(r, payload, true)))
    });
    match result { Ok(_) => 0, Err(e) => { failure(e); -1 } }
}
// Fixture construction still uses the Rust encoder. Caller owns typed arrays.
#[no_mangle]
pub unsafe extern "C" fn tk_create(filename: *const c_char, header: *const i32,
        strings: *const u8, ids: *const i32, data: *const u32, words: usize) -> i32 {
    let h: [i32; 21] = std::slice::from_raw_parts(header, 21).try_into().unwrap();
    let size = (h[16].max(0) as usize).checked_add(words).and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(FIXED));
    let result = size.ok_or_else(|| bad("record size overflow")).and_then(|size| validate(&h, size))
        .and_then(|(nc, nw)| {
            if nw != words { return Err(bad("fixture payload size mismatch")); }
            let r = Record { header: h, strings: std::slice::from_raw_parts(strings, 768).try_into().unwrap(),
                ids: if nc == 0 { vec![] } else { std::slice::from_raw_parts(ids, nc).to_vec() },
                words: if nw == 0 { vec![] } else { std::slice::from_raw_parts(data, nw).to_vec() } };
            path(filename).and_then(|name| fs::OpenOptions::new().write(true).create_new(true)
                .open(name).and_then(|mut f| f.write_all(&encode(&r, &r.words, false))))
        });
    match result { Ok(_) => 0, Err(e) => { failure(e); -1 } }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> Record {
        Record { header: [1,4,2,2,3,1,1,1,7,1,1,1,1,2,1,0,2,1,2,256,0],
                 strings: [b' '; 768], ids: vec![41, 53],
                 words: vec![0, 0x80000000, 0x3f800001, 0xff800000, 0x7fc00123, 0x3f800000] }
    }
    #[test]
    fn words_bounds_ids_and_context_survive() {
        let r = sample(); let raw = encode(&r, &r.words, false); let decoded = decode(&raw).unwrap();
        assert_eq!(decoded.words, r.words); assert_eq!(decoded.header, r.header); assert_eq!(decoded.ids, r.ids);
        assert_eq!(encode(&decoded, &decoded.words, false), raw);
        let post = encode(&decoded, &decoded.words, true);
        assert_eq!(&post[348..352], b"post"); assert_eq!(&post[860..], &raw[860..]);
    }
    #[test]
    fn malformed_records_are_rejected() {
        let r = sample(); let mut raw = encode(&r, &r.words, false);
        raw.pop(); assert!(decode(&raw).is_err()); raw.push(0); raw.push(0); assert!(decode(&raw).is_err());
        let mut raw = encode(&r, &r.words, false); raw[0] = 0; assert!(decode(&raw).is_err());
        let mut r = sample(); r.header[17] = 2; assert!(decode(&encode(&r, &r.words, false)).is_err());
        let mut r = sample(); r.header[3] = -1; assert!(decode(&encode(&r, &r.words, false)).is_err());
    }
}
