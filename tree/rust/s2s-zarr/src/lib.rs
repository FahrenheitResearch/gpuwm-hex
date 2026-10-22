//! Validated Blosc transport for ERA5 Zarr chunks. No Python codec or array
//! transformation is involved. Each caller owns its buffers; c-blosc runs one
//! thread per chunk so the orchestration pool controls total parallelism.
use std::ffi::{c_char, c_int, c_void};
#[link(name = "dl")]
extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}
type Validate = unsafe extern "C" fn(*const c_void, usize, *mut usize) -> c_int;
type Decode = unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int) -> c_int;

/// All pointers and capacities belong to the caller and must be valid for the
/// duration of this call. Returns the decoded byte count, or a negative error.
#[no_mangle]
pub unsafe extern "C" fn hex_s2s_decode(path: *const c_char, source: *const u8,
    source_len: usize, destination: *mut u8, destination_len: usize) -> i64 {
    if path.is_null() || source.is_null() || destination.is_null() || source_len < 16 {
        return -1;
    }
    let library = dlopen(path, 2); // RTLD_NOW, private handle, no global symbols.
    if library.is_null() { return -2; }
    let validate = dlsym(library, b"blosc_cbuffer_validate\0".as_ptr().cast());
    let decode = dlsym(library, b"blosc_decompress_ctx\0".as_ptr().cast());
    let result = if validate.is_null() || decode.is_null() { -3 } else {
        let validate: Validate = std::mem::transmute(validate);
        let decode: Decode = std::mem::transmute(decode);
        let mut expected = 0usize;
        if validate(source.cast(), source_len, &mut expected) != 0 || expected != destination_len {
            -4
        } else {
            let count = decode(source.cast(), destination.cast(), destination_len, 1);
            if count <= 0 || count as usize != expected { -5 } else { count as i64 }
        }
    };
    dlclose(library);
    result
}
