// Reader/writer copied from f1 oracle_words.rs SHA256
// 5a0fe35409a8bcb3240d73da21c14a2a650df9ce70f0eedc4de377e089b0cd38.
//! Typed MPAS oracle stream decoding, exact word comparison and column extraction.
//! No arithmetic tolerance is introduced. Python delegates all field I/O here.
use std::{fs, io, path::Path};
const FIXED: usize = 8 + 21 * 4 + 3 * 256;
#[derive(Clone)]
struct Record { header: [i32;21], strings: Vec<u8>, ids: Vec<i32>, data: Vec<u8> }
fn bad(message: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message) }
fn read(path: &Path) -> io::Result<Record> {
    let raw = fs::read(path)?;
    if raw.len() < FIXED || &raw[..8] != b"ORCL0001" { return Err(bad("invalid oracle header")); }
    let mut h=[0i32;21];
    for (i,v) in h.iter_mut().enumerate() { *v=i32::from_le_bytes(raw[8+4*i..12+4*i].try_into().unwrap()); }
    if !(0..=5).contains(&h[2]) || h[3..8].iter().any(|v| *v<0) || h[16]<0 || h[19]!=256 || h[20]!=0 {
        return Err(bad("malformed dimensions/version"));
    }
    let width=match h[0] {1|3|4=>4,2|6=>8,5 if h[1]>0=>h[1],_=>return Err(bad("invalid type code/width"))};
    if h[1]!=width { return Err(bad("type width mismatch")); }
    if h[17]<0 || h[17]>h[2] || h[18]<0 || h[18]>h[2] || (h[17]>0 && h[17]==h[18]) {return Err(bad("invalid axes"));}
    let expected_ids=if h[17]>0 {h[2+h[17] as usize]} else {0};
    if h[16]!=expected_ids {return Err(bad("ID count mismatch"));}
    if h[0]!=5 && h[3+h[2] as usize..8].iter().any(|v|*v!=1) {return Err(bad("inactive dimensions must equal one"));}
    let elements=h[3..8].iter().try_fold(1usize,|a,v|a.checked_mul(*v as usize)).ok_or_else(||bad("shape overflow"))?;
    let offset=FIXED.checked_add((h[16] as usize).checked_mul(4).ok_or_else(||bad("ID overflow"))?).ok_or_else(||bad("offset overflow"))?;
    let size=elements.checked_mul(width as usize).and_then(|v|v.checked_add(offset)).ok_or_else(||bad("payload overflow"))?;
    if raw.len()!=size {return Err(bad("truncated or trailing payload"));}
    let ids=raw[FIXED..offset].chunks_exact(4).map(|x|i32::from_le_bytes(x.try_into().unwrap())).collect();
    Ok(Record{header:h, strings:raw[92..FIXED].to_vec(), ids,data:raw[offset..].to_vec()})
}
fn write(path:&Path,r:&Record)->io::Result<()> {
    use std::io::Write;
    let mut file=fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(b"ORCL0001")?;for v in r.header {file.write_all(&v.to_le_bytes())?;}
    file.write_all(&r.strings)?;for v in &r.ids {file.write_all(&v.to_le_bytes())?;}file.write_all(&r.data)
}
// Test-only C ABI around f1's strict oracle record reader/writer. No weather arithmetic.
use std::ffi::{CStr, c_char};
struct Handle { record: Record, words: Vec<u32> }
#[no_mangle]
pub unsafe extern "C" fn ysu_load(path: *const c_char) -> *mut std::ffi::c_void {
    let result = (|| -> io::Result<Handle> {
        let s=CStr::from_ptr(path).to_str().map_err(|_|bad("invalid path"))?;
        let record=read(Path::new(s))?;
        if record.data.len()%4!=0 { return Err(bad("test ABI needs four-byte-aligned record storage")); }
        let words=record.data.chunks_exact(4).map(|x|u32::from_le_bytes(x.try_into().unwrap())).collect();
        Ok(Handle{record,words})
    })();
    match result {Ok(h)=>Box::into_raw(Box::new(h)).cast(),Err(e)=>{eprintln!("YSU oracle I/O: {e}");std::ptr::null_mut()}}
}
#[no_mangle]
pub unsafe extern "C" fn ysu_header(h:*mut std::ffi::c_void)->*const i32 { (*(h as *mut Handle)).record.header.as_ptr() }
#[no_mangle]
pub unsafe extern "C" fn ysu_words(h:*mut std::ffi::c_void)->*const u32 { (*(h as *mut Handle)).words.as_ptr() }
#[no_mangle]
pub unsafe extern "C" fn ysu_count(h:*mut std::ffi::c_void)->usize { (*(h as *mut Handle)).words.len() }
#[no_mangle]
pub unsafe extern "C" fn ysu_free(h:*mut std::ffi::c_void) { if !h.is_null() { drop(Box::from_raw(h as *mut Handle)); } }
#[no_mangle]
pub unsafe extern "C" fn ysu_save(h:*mut std::ffi::c_void,path:*const c_char,data:*const u32,count:usize)->i32 {
    let h=&*(h as *mut Handle);
    if count!=h.words.len() { return 2; }
    let mut r=h.record.clone();
    r.data=std::slice::from_raw_parts(data,count).iter().flat_map(|x|x.to_le_bytes()).collect();
    r.strings[256..512].fill(b' ');r.strings[256..260].copy_from_slice(b"post");
    match CStr::from_ptr(path).to_str().map_err(|_|bad("invalid path")).and_then(|s|write(Path::new(s),&r)) {
        Ok(())=>0,Err(e)=>{eprintln!("YSU oracle output: {e}");1}
    }
}

// MPAS v8.4.2 physics_mmm/bl_ysu.F90:1417-1418: successful control outputs.
#[no_mangle]
pub unsafe extern "C" fn ysu_success_message(h:*mut std::ffi::c_void,path:*const c_char)->i32 {
    let h=&*(h as *mut Handle);
    let mut data=vec![b' ';h.record.data.len()];
    let message=b"bl_ysu_run OK";
    if data.len()<message.len() {return 2;}
    data[..message.len()].copy_from_slice(message);
    let words:Vec<u32>=data.chunks_exact(4).map(|x|u32::from_le_bytes(x.try_into().unwrap())).collect();
    ysu_save((h as *const Handle).cast_mut().cast(),path,words.as_ptr(),words.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    fn path(name:&str)->std::path::PathBuf {
        Path::new("n1out").join(format!("ysu-io-test-{}-{name}",std::process::id()))
    }
    fn fixture()->Record {
        let mut h=[0;21];h[0]=1;h[1]=4;h[2]=2;h[3..8].copy_from_slice(&[2,3,1,1,1]);
        h[16]=2;h[17]=1;h[18]=2;h[19]=256;
        let mut strings=vec![b' ';768];strings[256..259].copy_from_slice(b"pre");
        Record{header:h,strings,ids:vec![91,17],
               data:[0,0x80000000,0x7fc00001,0x00000001,0x3f800000,0xffffffff]
               .iter().flat_map(|x:&u32|x.to_le_bytes()).collect()}
    }
    #[test]
    fn preserves_bits_ids_shape_and_entry_storage() {
        let input=path("pre");let output=path("post");let r=fixture();write(&input,&r).unwrap();
        unsafe {
            let h=ysu_load(CString::new(input.to_str().unwrap()).unwrap().as_ptr());assert!(!h.is_null());
            assert_eq!(ysu_count(h),6);
            let words=std::slice::from_raw_parts(ysu_words(h),6).to_vec();
            assert_eq!(words,[0,0x80000000,0x7fc00001,1,0x3f800000,0xffffffff]);
            let target=CString::new(output.to_str().unwrap()).unwrap();
            assert_eq!(ysu_save(h,target.as_ptr(),words.as_ptr(),5),2);assert!(!output.exists());
            assert_eq!(ysu_save(h,target.as_ptr(),words.as_ptr(),6),0);
            let saved=read(&output).unwrap();assert_eq!(saved.header,r.header);assert_eq!(saved.ids,r.ids);
            assert_eq!(saved.data,r.data);assert_eq!(&saved.strings[256..260],b"post");
            assert_eq!(ysu_save(h,target.as_ptr(),words.as_ptr(),6),1);ysu_free(h);
        }
        fs::remove_file(input).unwrap();fs::remove_file(output).unwrap();
    }
    #[test]
    fn rejects_truncation_and_wrong_id_mapping() {
        let input=path("invalid");let r=fixture();write(&input,&r).unwrap();
        let original=fs::read(&input).unwrap();let mut truncated=original.clone();truncated.pop();
        fs::write(&input,&truncated).unwrap();assert!(read(&input).is_err());
        let mut bad_ids=original;bad_ids[8+16*4..12+16*4].copy_from_slice(&1i32.to_le_bytes());
        fs::write(&input,&bad_ids).unwrap();assert!(read(&input).is_err());fs::remove_file(input).unwrap();
    }
    #[test]
    fn serializes_native_success_message_and_padding() {
        let input=path("message-pre");let output=path("message-post");let mut r=fixture();
        r.header[0]=5;r.header[1]=32;r.header[2]=0;r.header[3..8].fill(1);
        r.header[16]=0;r.header[17]=0;r.header[18]=0;r.ids.clear();r.data=vec![b'?';32];write(&input,&r).unwrap();
        unsafe {
            let h=ysu_load(CString::new(input.to_str().unwrap()).unwrap().as_ptr());assert!(!h.is_null());
            assert_eq!(ysu_success_message(h,CString::new(output.to_str().unwrap()).unwrap().as_ptr()),0);ysu_free(h);
        }
        let saved=read(&output).unwrap();let length=b"bl_ysu_run OK".len();
        assert_eq!(&saved.data[..length],b"bl_ysu_run OK");
        assert!(saved.data[length..].iter().all(|b|*b==b' '));fs::remove_file(input).unwrap();fs::remove_file(output).unwrap();
    }
}
