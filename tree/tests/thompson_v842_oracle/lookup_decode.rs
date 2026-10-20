//! Byte-preserving test transport for MPAS lookup records and auxiliary arrays.
//! Python supplies dimensions and launches kernels; all file decoding is Rust.
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::fs;
use std::io;
use std::os::raw::c_char;
use std::path::Path;

enum Array { F32(Vec<u32>), F64(Vec<u64>) }
pub struct Tables { arrays: Vec<Array> }
thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn bad(message: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message) }

fn decode(bytes: &[u8], counts: &[u64], widths: &[u32], records: bool, little: bool) -> io::Result<Tables> {
    if counts.len() != widths.len() { return Err(bad("inconsistent lookup metadata")); }
    let mut offset = 0usize;
    let mut arrays = Vec::new();
    for (&count, &width) in counts.iter().zip(widths) {
        if width != 4 && width != 8 { return Err(bad("lookup words must be binary32 or binary64")); }
        let size = count.checked_mul(width as u64).and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| bad("lookup extent overflow"))?;
        if records {
            let end = offset.checked_add(4).ok_or_else(|| bad("record offset overflow"))?;
            let marker: [u8; 4] = bytes.get(offset..end).ok_or_else(|| bad("truncated record marker"))?.try_into().unwrap();
            let length = if little { u32::from_le_bytes(marker) } else { u32::from_be_bytes(marker) };
            if length as usize != size { return Err(bad("lookup record extent differs from MPAS metadata")); }
            offset = end;
        }
        let end = offset.checked_add(size).ok_or_else(|| bad("lookup offset overflow"))?;
        let payload = bytes.get(offset..end).ok_or_else(|| bad("truncated lookup payload"))?;
        arrays.push(if width == 4 {
            Array::F32(payload.chunks_exact(4).map(|b| {
                let word = b.try_into().unwrap();
                if little { u32::from_le_bytes(word) } else { u32::from_be_bytes(word) }
            }).collect())
        } else {
            Array::F64(payload.chunks_exact(8).map(|b| {
                let word = b.try_into().unwrap();
                if little { u64::from_le_bytes(word) } else { u64::from_be_bytes(word) }
            }).collect())
        });
        offset = end;
        if records {
            let end = offset.checked_add(4).ok_or_else(|| bad("record offset overflow"))?;
            let marker: [u8; 4] = bytes.get(offset..end).ok_or_else(|| bad("truncated closing marker"))?.try_into().unwrap();
            let length = if little { u32::from_le_bytes(marker) } else { u32::from_be_bytes(marker) };
            if length as usize != size { return Err(bad("lookup closing record marker mismatch")); }
            offset = end;
        }
    }
    if offset != bytes.len() { return Err(bad("extra bytes after lookup arrays")); }
    Ok(Tables { arrays })
}

/// Caller owns valid filename/metadata pointers; returned buffers live until close.
#[no_mangle]
pub unsafe extern "C" fn thompson_lookup_open(filename: *const c_char, counts: *const u64,
    widths: *const u32, n: usize, records: bool, little: bool) -> *mut Tables {
    let result = (|| {
        let name = CStr::from_ptr(filename).to_str().map_err(|_| bad("lookup path is not UTF-8"))?;
        let bytes = fs::read(Path::new(name))?;
        decode(&bytes, std::slice::from_raw_parts(counts,n), std::slice::from_raw_parts(widths,n), records,little)
    })();
    match result {
        Ok(tables) => Box::into_raw(Box::new(tables)),
        Err(error) => {
            ERROR.with(|e| *e.borrow_mut() = CString::new(error.to_string()).unwrap());
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn thompson_lookup_data(tables: *const Tables, index: usize) -> *const std::ffi::c_void {
    let tables = &*tables;
    match &tables.arrays[index] {
        Array::F32(words) => words.as_ptr().cast(), Array::F64(words) => words.as_ptr().cast()
    }
}
#[no_mangle]
pub unsafe extern "C" fn thompson_lookup_close(tables: *mut Tables) { drop(Box::from_raw(tables)); }
#[no_mangle]
pub extern "C" fn thompson_lookup_error() -> *const c_char { ERROR.with(|e| e.borrow().as_ptr()) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn big_endian_records_preserve_signed_zero_and_nan_payloads() {
        let words = [0x8000_0000_0000_0000u64,0x7ff8_1234_5678_9abc,0x3ff0_0000_0000_0000];
        let mut bytes = (24u32).to_be_bytes().to_vec();
        for word in words { bytes.extend(word.to_be_bytes()); }
        bytes.extend(24u32.to_be_bytes());
        match &decode(&bytes,&[3],&[8],true,false).unwrap().arrays[0] {
            Array::F64(actual) => assert_eq!(actual,&words), _ => panic!()
        }
        let last = bytes.len()-1;bytes[last] ^= 1;
        assert!(decode(&bytes,&[3],&[8],true,false).is_err());
    }
    #[test]
    fn mixed_auxiliary_words_and_exact_extent() {
        let mut bytes = 0x8000_0000_0000_0000u64.to_le_bytes().to_vec();
        bytes.extend(0x7fc1_2345u32.to_le_bytes());
        let arrays=decode(&bytes,&[1,1],&[8,4],false,true).unwrap().arrays;
        match &arrays[1] { Array::F32(actual) => assert_eq!(actual,&[0x7fc1_2345]),_ => panic!() }
        assert!(decode(&bytes[..11],&[1,1],&[8,4],false,true).is_err());
        assert!(decode(&bytes,&[1],&[8],false,true).is_err());
        assert!(decode(&bytes,&[u64::MAX],&[8],false,true).is_err());
    }
}
