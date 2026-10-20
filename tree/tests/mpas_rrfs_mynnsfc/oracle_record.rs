//! ORCL0001 transport seeded from f1 native oracle_words.rs.
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

use std::{ffi::{CStr,c_char},ptr,slice};
/// Decode a validated record into caller-owned memory (free with oracle_free).
/// `header` points to 21 i32s; `length` receives byte count.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_read(path:*const c_char,header:*mut i32,length:*mut usize)->*mut u8 {
    unsafe {
    let result=(|| {let path=CStr::from_ptr(path).to_str().map_err(|_|bad("non-UTF8 path"))?;read(Path::new(path))})();
    match result {Ok(r)=>{ptr::copy_nonoverlapping(r.header.as_ptr(),header,21);*length=r.data.len();Box::into_raw(r.data.into_boxed_slice()) as *mut u8},
      Err(e)=>{eprintln!("oracle_read: {e}");*length=0;ptr::null_mut()}}
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_free(data:*mut u8,length:usize) { unsafe { if !data.is_null() {drop(Box::from_raw(ptr::slice_from_raw_parts_mut(data,length)));} } }
/// Preserve all schema/IDs; write new payload using entry-storage template.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_write(template:*const c_char,dest:*const c_char,data:*const u8,length:usize)->i32 {
    unsafe {
    let result=(|| {
      let template=CStr::from_ptr(template).to_str().map_err(|_|bad("non-UTF8 template"))?;
      let dest=CStr::from_ptr(dest).to_str().map_err(|_|bad("non-UTF8 dest"))?;
      let r=read(Path::new(template))?;debug_assert_eq!(r.strings.len(),768);debug_assert_eq!(r.ids.len(),r.header[16] as usize);if r.data.len()!=length {return Err(bad("payload length mismatch"));}
      let mut raw=fs::read(template)?;let offset=raw.len()-length;
      raw[offset..].copy_from_slice(slice::from_raw_parts(data,length));
      raw[348..604].fill(b' ');raw[348..352].copy_from_slice(b"post");
      let mut file=fs::OpenOptions::new().write(true).create_new(true).open(dest)?;
      use std::io::Write;file.write_all(&raw)
    })();
    match result {Ok(())=>0,Err(e)=>{eprintln!("oracle_write: {e}");1}}
    }
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn preserves_words_and_rejects_truncated_records() {
   let root=env_root();fs::create_dir(&root).unwrap();
   let input=root.join("entry.bin");let output=root.join("post.bin");
   let mut h=[0i32;21];h[0]=1;h[1]=4;h[2]=1;h[3..8].fill(1);h[3]=3;
   h[8]=7;h[16]=3;h[17]=1;h[19]=256;
   let mut raw=b"ORCL0001".to_vec();for v in h {raw.extend(v.to_le_bytes());}
   raw.extend([b' ';768]);for id in [12i32,0,19] {raw.extend(id.to_le_bytes());}
   let payload=[0x80000000u32,0x7fc00123,0x3f800000];for v in payload {raw.extend(v.to_le_bytes());}
   fs::write(&input,&raw).unwrap();let r=read(&input).unwrap();
   assert_eq!(r.ids,vec![12,0,19]);assert_eq!(r.header,h);
   use std::ffi::CString;
   let a=CString::new(input.to_str().unwrap()).unwrap();let b=CString::new(output.to_str().unwrap()).unwrap();
   unsafe {let mut header=[0;21];let mut len=0;let ptr=oracle_read(a.as_ptr(),header.as_mut_ptr(),&mut len);
     assert!(!ptr.is_null());assert_eq!(len,12);assert_eq!(header,h);
     assert_eq!(oracle_write(a.as_ptr(),b.as_ptr(),ptr,len),0);oracle_free(ptr,len);}
   assert_eq!(read(&output).unwrap().data,r.data);
   assert_eq!(&fs::read(&output).unwrap()[348..352],b"post");
   raw.pop();fs::write(&input,raw).unwrap();assert!(read(&input).is_err());
   fs::remove_dir_all(root).unwrap();
 }
 fn env_root()->std::path::PathBuf {std::env::current_dir().unwrap().join(format!("oracle-record-test-{}",std::process::id()))}
}
