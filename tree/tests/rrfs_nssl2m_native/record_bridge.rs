// Generic record bridge adapted from r15; no physics or numeric conversion.
// Appended to the committed oracle_words.rs decoder in a private build directory.
// Only Rust decodes and encodes native oracle records; Python owns launch wiring.
use std::ffi::{c_char, CStr};

struct NsslWords { record: Record, words: Vec<u32> }

#[no_mangle]
pub unsafe extern "C" fn nssl_record_open(path: *const c_char, header: *mut i32,
                                        payload: *mut *const u32) -> *mut std::ffi::c_void {
    let result = (|| -> io::Result<NsslWords> {
        let name = CStr::from_ptr(path).to_str().map_err(|_| bad("invalid path"))?;
        let record = read(Path::new(name))?;
        if record.header[1] != 4 || !matches!(record.header[0], 1 | 3 | 4) {
            return Err(bad("NSSL bridge requires real32, int32 or logical32"));
        }
        let words = record.data.chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        Ok(NsslWords { record, words })
    })();
    match result {
        Ok(value) => {
            std::ptr::copy_nonoverlapping(value.record.header.as_ptr(), header, 21);
            *payload = value.words.as_ptr();
            Box::into_raw(Box::new(value)).cast()
        }
        Err(error) => { eprintln!("NSSL record open: {error}"); std::ptr::null_mut() }
    }
}

#[no_mangle]
pub unsafe extern "C" fn nssl_record_close(handle: *mut std::ffi::c_void) {
    if !handle.is_null() { drop(Box::from_raw(handle.cast::<NsslWords>())); }
}

#[no_mangle]
pub unsafe extern "C" fn nssl_record_post(template: *const c_char, target: *const c_char,
                                        payload: *const u32, count: usize) -> i32 {
    let result = (|| -> io::Result<()> {
        let source = CStr::from_ptr(template).to_str().map_err(|_| bad("invalid source"))?;
        let destination = CStr::from_ptr(target).to_str().map_err(|_| bad("invalid target"))?;
        let mut record = read(Path::new(source))?;
        if record.header[1] != 4 || count != record.data.len() / 4 {
            return Err(bad("replacement payload extent differs"));
        }
        // Metadata comes from PRE, never from expected POST values.
        record.strings[256..512].fill(b' ');
        record.strings[256..260].copy_from_slice(b"post");
        record.data.clear();
        for value in std::slice::from_raw_parts(payload, count) {
            record.data.extend_from_slice(&value.to_le_bytes());
        }
        write(Path::new(destination), &record)
    })();
    match result { Ok(()) => 0, Err(error) => { eprintln!("NSSL record write: {error}"); 1 } }
}
