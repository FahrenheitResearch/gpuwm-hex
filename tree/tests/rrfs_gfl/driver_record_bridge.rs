// Append to oracle_words.rs for a private cdylib build. No new record codec.
// Fork boundary: MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_todynamics.F:66-245.
use std::ffi::{c_char, c_void, CStr, CString};
struct DriverRecord { record: Record, text: CString }
unsafe fn r4_path<'a>(p: *const c_char) -> io::Result<&'a Path> {
    Ok(Path::new(CStr::from_ptr(p).to_str().map_err(|_|bad("invalid path"))?))
}
#[no_mangle]
pub unsafe extern "C" fn r4_record_open(path: *const c_char, header: *mut i32,
    payload: *mut *const u8, length: *mut usize, name: *mut *const c_char) -> *mut c_void {
    match r4_path(path).and_then(read).and_then(|record| {
        let text=CString::new(String::from_utf8_lossy(&record.strings[512..768]).trim_end())
            .map_err(|_|bad("invalid record name"))?;
        Ok(DriverRecord{record,text})
    }) {
        Ok(value)=>{
            std::ptr::copy_nonoverlapping(value.record.header.as_ptr(),header,21);
            *payload=value.record.data.as_ptr();*length=value.record.data.len();*name=value.text.as_ptr();
            Box::into_raw(Box::new(value)).cast()
        },
        Err(e)=>{eprintln!("RRFS record open: {e}");std::ptr::null_mut()}
    }
}
#[no_mangle]
pub unsafe extern "C" fn r4_record_close(handle:*mut c_void){
    if !handle.is_null(){drop(Box::from_raw(handle.cast::<DriverRecord>()));}
}
#[no_mangle]
pub unsafe extern "C" fn r4_record_text(handle:*mut c_void, result:*mut *const c_char)->i32 {
    let value=&mut *handle.cast::<DriverRecord>();
    if value.record.header[0]!=5{return 1;}
    let decoded=String::from_utf8_lossy(&value.record.data).trim_end().to_string();
    match CString::new(decoded){Ok(s)=>{value.text=s;*result=value.text.as_ptr();0},Err(_)=>1}
}
#[no_mangle]
pub unsafe extern "C" fn r4_record_post(source:*const c_char, target:*const c_char,
    payload:*const u8, length:usize)->i32 {
    let result=(||->io::Result<()>{
        let mut record=read(r4_path(source)?)?;
        if !payload.is_null(){
            if length!=record.data.len(){return Err(bad("replacement extent differs"));}
            record.data=std::slice::from_raw_parts(payload,length).to_vec();
        }
        // Expected POST is deliberately inaccessible to record generation.
        record.strings[256..512].fill(b' ');record.strings[256..260].copy_from_slice(b"post");
        write(r4_path(target)?,&record)
    })();
    match result{Ok(())=>0,Err(e)=>{eprintln!("RRFS record post: {e}");1}}
}
#[no_mangle]
pub unsafe extern "C" fn r4_record_compare(left:*const c_char,right:*const c_char)->*mut c_char {
    let result=(||->io::Result<String>{
        let a=read(r4_path(left)?)?;let b=read(r4_path(right)?)?;
        if a.strings!=b.strings || a.header[13..16]!=b.header[13..16] || a.header[19..]!=b.header[19..] {
            return Ok(format!("{{\"words\":{},\"different_words\":null,\"bitwise_equal\":false,\"comparison_complete\":false,\"schema_mismatch\":[\"boundary_identity\"]}}",a.data.len()/component_width(&a)));
        }
        compare(&a,&b)
    })();
    match result{Ok(s)=>CString::new(s).unwrap().into_raw(),Err(e)=>{eprintln!("RRFS compare: {e}");std::ptr::null_mut()}}
}
#[no_mangle]
pub unsafe extern "C" fn r4_string_free(value:*mut c_char){if !value.is_null(){drop(CString::from_raw(value));}}
