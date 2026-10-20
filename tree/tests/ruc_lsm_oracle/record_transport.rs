//! Reuse the landed native-tap decoder, writer and exact-word comparator.
//! This ABI exposes typed owned buffers; Python never decodes weather files.
include!("../../tools/native_physics_reference/oracle/oracle_words.rs");
use std::{ffi::CStr,os::raw::c_char,ptr};
unsafe fn path<'a>(value:*const c_char)-> &'a Path {
    Path::new(CStr::from_ptr(value).to_str().unwrap())
}
#[no_mangle]
pub unsafe extern "C" fn oracle_read(name:*const c_char,h:*mut i32,size:*mut usize)->*mut u8 {
    match read(path(name)) {
        Ok(r)=>{
            ptr::copy_nonoverlapping(r.header.as_ptr(),h,21);
            *size=r.data.len();let data=r.data.into_boxed_slice();
            Box::into_raw(data) as *mut u8
        },
        Err(e)=>{eprintln!("{e}");ptr::null_mut()}
    }
}
#[no_mangle]
pub unsafe extern "C" fn oracle_free(data:*mut u8,size:usize) {
    drop(Box::from_raw(ptr::slice_from_raw_parts_mut(data,size)));
}
#[no_mangle]
pub unsafe extern "C" fn oracle_write(template:*const c_char,dest:*const c_char,data:*const u8,size:usize)->i32 {
    let result=(||{
        let mut r=read(path(template))?;
        if r.data.len()!=size{return Err(bad("output payload size differs from native input template"));}
        r.data=std::slice::from_raw_parts(data,size).to_vec();
        r.strings[256..512].fill(b' ');r.strings[256..260].copy_from_slice(b"post");
        write(path(dest),&r)
    })();
    if let Err(e)=result{eprintln!("{e}");1}else{0}
}

// The translated native call flow supplies invocation counts, one per cell.
// This maps captured PRE entries in original horizontal order. It performs
// record transport only; no weather arithmetic or expected-output lookup.
#[no_mangle]
pub unsafe extern "C" fn oracle_land_entries(prefix:*const c_char,calls:*const i32,n:usize,size:*mut usize)->*mut u8 {
    let result=(||->io::Result<Vec<u8>>{
        let prefix=CStr::from_ptr(prefix).to_str().unwrap();
        let record=|name:&str|{let file=format!("{prefix}.pre.{name}.bin");read(Path::new(&file)).map_err(|e|bad(&format!("{file}: {e}")))};
        let mut ordinal=0;let mut data=Vec::new();
        for &count in std::slice::from_raw_parts(calls,n){
            if count<0 || count>1{return Err(bad("replay mapper supports at most one snowtemp invocation per native cell"));}
            if count==1 {
                ordinal+=1;
                let r=record(&format!("allocation.module_ruc_land.snowtemp.ilnb.entry.L3677.n{ordinal:010}"))?;
                if r.header[0]!=3 || r.data.len()!=4{return Err(bad("snowtemp entry must be one native integer"));}
                data.extend_from_slice(&r.data);
            }else{data.extend_from_slice(&0i32.to_le_bytes());}
        }
        let extra=Path::new(&format!("{prefix}.pre.allocation.module_ruc_land.snowtemp.ilnb.entry.L3677.n{:010}.bin",ordinal+1)).exists();
        if extra{return Err(bad("unmapped snowtemp entry ordinal, native call topology differs"));}
        Ok(data)
    })();
    match result{Ok(data)=>{*size=data.len();Box::into_raw(data.into_boxed_slice()) as *mut u8},Err(e)=>{eprintln!("{e}");ptr::null_mut()}}
}

// Native sea-ice INOUT entry words, same original invocation-order contract.
#[no_mangle]
pub unsafe extern "C" fn oracle_ice_entries(prefix:*const c_char,calls:*const i32,n:usize,size:*mut usize)->*mut u8 {
    let result=(||->io::Result<Vec<u8>>{
        let prefix=CStr::from_ptr(prefix).to_str().unwrap();
        let record=|name:&str|{let file=format!("{prefix}.pre.{name}.bin");read(Path::new(&file)).map_err(|e|bad(&format!("{file}: {e}")))};
        let mut ordinal=0;let mut data=Vec::new();
        for &count in std::slice::from_raw_parts(calls,n){
            if count<0 || count>1{return Err(bad("replay mapper supports at most one snowseaice invocation per native cell"));}
            if count==1 {
                ordinal+=1;
                let r=record(&format!("allocation.module_ruc_ice.snowseaice.ilnb.entry.L1591.n{ordinal:010}"))?;
                if r.header[0]!=3 || r.data.len()!=4{return Err(bad("snowseaice entry must be one native integer"));}
                data.extend_from_slice(&r.data);
            }else{data.extend_from_slice(&0i32.to_le_bytes());}
        }
        let extra=Path::new(&format!("{prefix}.pre.allocation.module_ruc_ice.snowseaice.ilnb.entry.L1591.n{:010}.bin",ordinal+1)).exists();
        if extra{return Err(bad("unmapped snowseaice entry ordinal, native call topology differs"));}
        Ok(data)
    })();
    match result{Ok(data)=>{*size=data.len();Box::into_raw(data.into_boxed_slice()) as *mut u8},Err(e)=>{eprintln!("{e}");ptr::null_mut()}}
}
