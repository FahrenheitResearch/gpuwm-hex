// Test-only FFI appended to f1's unchanged oracle_words.rs. Its validated
// record decoder/writer owns field I/O; Python only orchestrates array views.
use std::cell::RefCell;
use std::ffi::{CStr,CString};
use std::os::raw::c_char;
pub struct DumpRecord { record:Record, aligned:Vec<u64> }
thread_local! { static FFI_ERROR:RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn ffi_error(error:io::Error) { FFI_ERROR.with(|e|*e.borrow_mut()=CString::new(error.to_string()).unwrap()); }
#[no_mangle]
pub unsafe extern "C" fn dump_open(filename:*const c_char)->*mut DumpRecord {
    let result=(|| {
        if !cfg!(target_endian="little") {return Err(bad("oracle test host must be little endian"));}
        let name=CStr::from_ptr(filename).to_str().map_err(|_|bad("invalid UTF-8 path"))?;
        let record=read(Path::new(name))?;
        let aligned=record.data.chunks(8).map(|b| {let mut raw=[0u8;8];raw[..b.len()].copy_from_slice(b);u64::from_le_bytes(raw)}).collect();
        Ok(DumpRecord {record,aligned})
    })();
    match result {Ok(r)=>Box::into_raw(Box::new(r)),Err(e)=>{ffi_error(e);std::ptr::null_mut()}}
}
#[no_mangle]
pub unsafe extern "C" fn dump_header(r:*const DumpRecord)->*const i32 {(&*r).record.header.as_ptr()}
#[no_mangle]
pub unsafe extern "C" fn dump_data(r:*const DumpRecord)->*const std::ffi::c_void {(&*r).aligned.as_ptr().cast()}
#[no_mangle]
pub unsafe extern "C" fn dump_close(r:*mut DumpRecord) {drop(Box::from_raw(r));}
/// Write candidate data using input schema/IDs. Null data preserves the input
/// bytes for records whose immutability is checked separately by the replay.
#[no_mangle]
pub unsafe extern "C" fn dump_write(r:*const DumpRecord,filename:*const c_char,
    data:*const u8,bytes:usize)->bool {
    let result=(|| {
        let mut record=(&*r).record.clone();
        if !data.is_null() {
            if bytes!=record.data.len() {return Err(bad("candidate byte count differs from captured schema"));}
            record.data.copy_from_slice(std::slice::from_raw_parts(data,bytes));
        }
        record.strings[256..512].fill(b' ');record.strings[256..260].copy_from_slice(b"post");
        let name=CStr::from_ptr(filename).to_str().map_err(|_|bad("invalid UTF-8 output path"))?;
        write(Path::new(name),&record)
    })();
    match result {Ok(())=>true,Err(e)=>{ffi_error(e);false}}
}
#[no_mangle]
pub extern "C" fn dump_error()->*const c_char {FFI_ERROR.with(|e|e.borrow().as_ptr())}
