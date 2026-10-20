//! Synthetic oracle file transport, deliberately owned by Rust.
use std::{ffi::CStr, fs, io::{BufWriter,Write}, os::raw::{c_char,c_void},slice};
// Native public parameters: RRFS TEMPO/src/module_mp_tempo_params.F90:23-857.
// Rust owns kind promotion and little-endian serialization for CUDA's blob.
#[no_mangle]
pub unsafe extern "C" fn tempo_append_table(path:*const c_char,ptr:*const c_void,count:usize,kind:i32,first:i32)->i32 {
    if (ptr.is_null() && kind!=0) || path.is_null() || !(0..=4).contains(&kind) {return 2}
    let Ok(name)=CStr::from_ptr(path).to_str() else{return 2};
    let Ok(file)=fs::OpenOptions::new().write(true).create_new(first!=0).append(first==0).open(name) else{return 1};
    let mut output=BufWriter::new(file);
    for i in 0..count {
        let value=match kind {
            0=>0.0_f64, // Reserved slots for explicitly unallocated inactive tables.
            1=>*ptr.cast::<f32>().add(i) as f64,
            2=>*ptr.cast::<f64>().add(i),
            3=>*ptr.cast::<i32>().add(i) as f64,
            _=>*ptr.cast::<i64>().add(i) as f64,
        };
        if output.write_all(&value.to_le_bytes()).is_err() {return 1}
    }
    if output.flush().is_err() {1} else {0}
}
#[no_mangle]
pub unsafe extern "C" fn tempo_write(path:*const c_char,ptr:*const c_void,size:usize)->i32 {
    let p=CStr::from_ptr(path).to_str().unwrap();
    match fs::write(p,slice::from_raw_parts(ptr.cast::<u8>(),size)){Ok(())=>0,Err(_)=>1}
}
#[no_mangle]
pub unsafe extern "C" fn tempo_read_f64(path:*const c_char,count:usize)->*mut f64 {
    let Ok(bytes)=fs::read(CStr::from_ptr(path).to_str().unwrap()) else{return std::ptr::null_mut()};
    if bytes.len()!=count*8{return std::ptr::null_mut()}
    let values:Vec<f64>=bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    Box::into_raw(values.into_boxed_slice()).cast::<f64>()
}
#[no_mangle]
pub unsafe extern "C" fn tempo_free_f64(ptr:*mut f64,count:usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr,count)))
}
#[no_mangle]
pub unsafe extern "C" fn tempo_read_f32(path:*const c_char,count:usize)->*mut f32 {
    let Ok(bytes)=fs::read(CStr::from_ptr(path).to_str().unwrap()) else{return std::ptr::null_mut()};
    if bytes.len()!=count*4{return std::ptr::null_mut()}
    let values:Vec<f32>=bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    Box::into_raw(values.into_boxed_slice()).cast::<f32>()
}
#[no_mangle]
pub unsafe extern "C" fn tempo_free_f32(ptr:*mut f32,count:usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr,count)))
}
