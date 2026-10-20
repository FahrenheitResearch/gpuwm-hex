//! Exact changed-word locations in two native NetCDF files, decoded in Rust.
//! Supporting native-defect evidence; never replaces the forecast verifier.
//! Mask bits index complete native variable storage in NetCDF dimension order.
use std::{collections::BTreeMap,env,ffi::{CString,CStr,c_char,c_void},fs::OpenOptions,io::{self,Write}};
#[link(name="netcdf")]
unsafe extern "C" {
 fn nc_open(path:*const c_char,mode:i32,id:*mut i32)->i32;
 fn nc_close(id:i32)->i32;
 fn nc_inq_nvars(id:i32,n:*mut i32)->i32;
 fn nc_inq_var(id:i32,var:i32,name:*mut c_char,kind:*mut i32,rank:*mut i32,dims:*mut i32,atts:*mut i32)->i32;
 fn nc_inq_dimlen(id:i32,dim:i32,len:*mut usize)->i32;
 fn nc_get_var(id:i32,var:i32,out:*mut c_void)->i32;
 fn nc_strerror(status:i32)->*const c_char;
}
fn check(status:i32)->io::Result<()>{if status==0{Ok(())}else{Err(io::Error::other(
 unsafe{CStr::from_ptr(nc_strerror(status))}.to_string_lossy().into_owned()))}}
struct File(i32);
impl File {
 fn open(path:&str)->io::Result<Self>{let path=CString::new(path).map_err(io::Error::other)?;let mut id=0;
  unsafe{check(nc_open(path.as_ptr(),0,&mut id))?;}Ok(Self(id))}
 fn variables(&self)->io::Result<BTreeMap<String,(i32,i32,Vec<usize>)>>{
  let mut n=0;unsafe{check(nc_inq_nvars(self.0,&mut n))?;}
  let mut out=BTreeMap::new();
  for index in 0..n {
   let mut name=[0 as c_char;257];let mut kind=0;let mut rank=0;let mut dims=[0;1024];let mut atts=0;
   unsafe{check(nc_inq_var(self.0,index,name.as_mut_ptr(),&mut kind,&mut rank,dims.as_mut_ptr(),&mut atts))?;}
   if !(0..=1024).contains(&rank){return Err(io::Error::other("invalid NetCDF rank"))}
   let mut shape=Vec::new();for dim in &dims[..rank as usize]{let mut length=0;
    unsafe{check(nc_inq_dimlen(self.0,*dim,&mut length))?;}shape.push(length);}
   let name=unsafe{CStr::from_ptr(name.as_ptr())}.to_str().map_err(io::Error::other)?.to_owned();
   if out.insert(name,(index,kind,shape)).is_some(){return Err(io::Error::other("duplicate NetCDF variable"))}
  }
  Ok(out)
 }
 fn read(&self,index:i32,length:usize)->io::Result<Vec<u8>>{
  let mut words=vec![0u64;length.div_ceil(8)];
  unsafe{check(nc_get_var(self.0,index,words.as_mut_ptr() as *mut _))?;}
  Ok(unsafe{std::slice::from_raw_parts(words.as_ptr() as *const u8,length)}.to_vec())
 }
}
impl Drop for File{fn drop(&mut self){unsafe{nc_close(self.0);}}}
fn width(kind:i32)->io::Result<usize>{Ok(match kind{
 1|2|7=>1,3|8=>2,4|5|9=>4,6|10|11=>8,_=>return Err(io::Error::other("unsupported native variable type"))})}
fn quoted(s:&str)->String{let mut out=String::from("\"");for c in s.chars(){match c{
 '"'=>out.push_str("\\\""),'\\'=>out.push_str("\\\\"),c if c.is_control()=>out.push_str(&format!("\\u{:04x}",c as u32)),c=>out.push(c)}}out.push('"');out}
fn mask(left:&[u8],right:&[u8],width:usize)->io::Result<(Vec<u8>,usize)>{
 if left.len()!=right.len()||left.len()%width!=0{return Err(io::Error::other("native variable word extent mismatch"))}
 let mut mask=vec![0u8;(left.len()/width).div_ceil(8)];let mut changed=0;
 for(index,(l,r))in left.chunks_exact(width).zip(right.chunks_exact(width)).enumerate(){
  if l!=r{mask[index/8]|=1<<(index%8);changed+=1;}}
 Ok((mask,changed))
}
fn main()->io::Result<()>{
 let args:Vec<_>=env::args().collect();if args.len()!=4{return Err(io::Error::other("native_wordsets LEFT.nc RIGHT.nc MASK"))}
 let left=File::open(&args[1])?;let right=File::open(&args[2])?;
 let l=left.variables()?;let r=right.variables()?;
 if l.keys().ne(r.keys()){return Err(io::Error::other("native variable inventories differ"))}
 let mut output=OpenOptions::new().write(true).create_new(true).open(&args[3])?;
 output.write_all(b"NCMASK01")?;let mut offset=8;let mut rows=Vec::new();let mut different=0;
 for(name,(li,kind,shape))in l{
  let(ri,rkind,rshape)=&r[&name];if kind!=*rkind||shape!=*rshape{return Err(io::Error::other("native variable schema differs"))}
  let width=width(kind)?;let count=shape.iter().try_fold(1usize,|a,b|a.checked_mul(*b))
   .ok_or_else(||io::Error::other("native variable shape overflow"))?;
  let length=count.checked_mul(width).ok_or_else(||io::Error::other("native variable byte overflow"))?;
  if length>1024*1024*1024{return Err(io::Error::other("native diagnostic variable exceeds capacity"))}
  let(bits,changed)=mask(&left.read(li,length)?,&right.read(*ri,length)?,width)?;
  output.write_all(&bits)?;
  rows.push(format!("{{\"field\":{},\"type\":{kind},\"word_width\":{width},\"shape\":{shape:?},\"compared_words\":{count},\"different_words\":{changed},\"mask_byte_offset\":{offset},\"mask_bytes\":{}}}",quoted(&name),bits.len()));
  offset+=bits.len();different+=changed;
 }
 println!("{{\"schema\":\"native-netcdf-wordsets-v1\",\"mask_format\":\"NCMASK01; LSB-first one bit per variable word\",\"different_words\":{different},\"fields\":[{}]}}",rows.join(","));Ok(())
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn exact_word_masks_preserve_signed_zero_nan_and_tail_positions(){
  let left:Vec<u8>=[0u32,0x80000000,0x7fc12345,1,2,3,4,5,6].into_iter().flat_map(u32::to_le_bytes).collect();
  let right:Vec<u8>=[0x80000000u32,0x80000000,0x7fc12346,1,2,3,4,5,7].into_iter().flat_map(u32::to_le_bytes).collect();
  let(bits,changed)=mask(&left,&right,4).unwrap();assert_eq!(changed,3);assert_eq!(bits,vec![5,1]);
 }
 #[test]fn malformed_extents_fail(){assert!(mask(&[0],&[0,1],1).is_err());assert!(mask(&[0],&[0],4).is_err());}
}
