//! Native ORCL0001 transport and TEMPO table packing. Rust owns field decoding.
use std::{ffi::{CStr,c_char},fs,io::Write,path::Path};
const FIXED:usize=860;
fn record(path:&Path)->Result<([i32;21],Vec<u8>),String>{
    let raw=fs::read(path).map_err(|e|e.to_string())?;
    if raw.len()<FIXED || &raw[..8]!=b"ORCL0001" {return Err("invalid native header".into())}
    let mut h=[0_i32;21];
    for (i,v) in h.iter_mut().enumerate(){*v=i32::from_le_bytes(raw[8+i*4..12+i*4].try_into().unwrap());}
    if !(0..=5).contains(&h[2]) || h[3..8].iter().any(|v|*v<0) || h[16]<0 || h[19]!=256 || h[20]!=0 {return Err("invalid native shape".into())}
    let width=match h[0]{1|3|4=>4,2|6=>8,5 if h[1]>0=>h[1],_=>return Err("unsupported native kind".into())};
    if h[1]!=width || h[17]<0 || h[17]>h[2] || h[18]<0 || h[18]>h[2] || (h[17]>0 && h[17]==h[18]) {return Err("invalid native kind/axes".into())}
    let ids=if h[17]>0{h[2+h[17] as usize]}else{0};
    if ids!=h[16] {return Err("invalid cell map".into())}
    let count=h[3..8].iter().try_fold(1_usize,|a,v|a.checked_mul(*v as usize)).ok_or("shape overflow")?;
    let offset=FIXED.checked_add(ids as usize*4).ok_or("offset overflow")?;
    if count.checked_mul(width as usize).and_then(|v|v.checked_add(offset))!=Some(raw.len()) {return Err("truncated native payload".into())}
    Ok((h,raw[offset..].to_vec()))
}
unsafe fn name<'a>(p:*const c_char)->Result<&'a str,String>{
    if p.is_null(){return Err("null path".into())}
    CStr::from_ptr(p).to_str().map_err(|e|e.to_string())
}
#[no_mangle]
pub unsafe extern "C" fn tempo_capture_read(path:*const c_char,header:*mut i32,size:*mut usize)->*mut u8{
    let result=name(path).and_then(|p|record(Path::new(p)));
    match result {
        Ok((h,raw)) if !header.is_null() && !size.is_null()=>{
            std::ptr::copy_nonoverlapping(h.as_ptr(),header,21);*size=raw.len();
            Box::into_raw(raw.into_boxed_slice()).cast()
        },
        _=>std::ptr::null_mut()
    }
}
#[no_mangle]
pub unsafe extern "C" fn tempo_capture_free(ptr:*mut u8,size:usize){
    if !ptr.is_null(){drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr,size)))}
}
fn pack(base:&Path,index:&Path,out:&Path)->Result<(),String>{
    let mut blob=fs::read(base).map_err(|e|e.to_string())?;
    if blob.len()%8!=0 {return Err("invalid base table bytes".into())}
    let text=fs::read_to_string(index).map_err(|e|e.to_string())?;
    let mut end=0;
    for row in text.lines(){
        let fields:Vec<_>=row.split('\t').collect();
        if fields.len()!=3{return Err("invalid layout row".into())}
        let offset:usize=fields[0].parse().map_err(|_|"invalid offset")?;
        let count:usize=fields[1].parse().map_err(|_|"invalid extent")?;
        if offset!=end || count==0 || (offset+count)*8>blob.len(){return Err("incomplete/overlapping table layout".into())}
        end=offset+count;
        if fields[2]=="SOURCE_CONSTANT"{continue}
        if fields[2]=="ABSENT"{
            blob[offset*8..end*8].fill(0);continue
        }
        let (h,raw)=record(Path::new(fields[2]))?;
        if raw.len()!=count*h[1] as usize{return Err("captured table extent differs".into())}
        for (i,bytes) in raw.chunks_exact(h[1] as usize).enumerate(){
            let value=match h[0]{
                1=>f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
                2=>f64::from_le_bytes(bytes.try_into().unwrap()),
                3=>i32::from_le_bytes(bytes.try_into().unwrap()) as f64,
                6=>i64::from_le_bytes(bytes.try_into().unwrap()) as f64,
                _=>return Err("invalid table kind".into())
            };
            blob[(offset+i)*8..(offset+i+1)*8].copy_from_slice(&value.to_le_bytes());
        }
    }
    if end*8!=blob.len(){return Err("incomplete captured table layout".into())}
    let mut file=fs::OpenOptions::new().write(true).create_new(true).open(out).map_err(|e|e.to_string())?;
    file.write_all(&blob).map_err(|e|e.to_string())
}
#[no_mangle]
pub unsafe extern "C" fn tempo_capture_pack(base:*const c_char,index:*const c_char,out:*const c_char)->i32{
    let result=(||pack(Path::new(name(base)?),Path::new(name(index)?),Path::new(name(out)?)))();
    if let Err(e)=result{eprintln!("TEMPO captured tables: {e}");1}else{0}
}
