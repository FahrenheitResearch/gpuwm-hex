//! Copy native NetCDF bytes and set explicitly named f32/f64 fields to IEEE +0.
//! No interpolation, conversion of other fields, or implicit field selection.
use std::{collections::BTreeSet, ffi::{CString,CStr,c_char,c_int,c_void},path::Path,os::unix::fs::PermissionsExt};
#[link(name="netcdf")]
unsafe extern "C" {
 fn nc_open(p:*const c_char,mode:c_int,id:*mut c_int)->c_int;
 fn nc_close(id:c_int)->c_int;
 fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
 fn nc_inq_var(id:c_int,var:c_int,name:*mut c_char,t:*mut c_int,nd:*mut c_int,d:*mut c_int,a:*mut c_int)->c_int;
 fn nc_inq_dimlen(id:c_int,dim:c_int,n:*mut usize)->c_int;
 fn nc_put_vara(id:c_int,var:c_int,start:*const usize,count:*const usize,data:*const c_void)->c_int;
 fn nc_get_vara(id:c_int,var:c_int,start:*const usize,count:*const usize,data:*mut c_void)->c_int;
 fn nc_strerror(code:c_int)->*const c_char;
}
fn check(code:c_int)->Result<(),String>{if code==0 {Ok(())}else{Err(unsafe{CStr::from_ptr(nc_strerror(code))}.to_string_lossy().into_owned())}}
struct File(c_int);
impl Drop for File{fn drop(&mut self){unsafe{nc_close(self.0);}}}
fn run()->Result<(),String>{
 let args:Vec<String>=std::env::args().collect();
 if args.len()<4{return Err("usage: copy_zero_fields INPUT OUTPUT FIELD...".into());}
 let names:BTreeSet<_>=args[3..].iter().collect();
 if names.len()!=args.len()-3{return Err("duplicate explicit field".into());}
 if Path::new(&args[2]).exists(){return Err("output already exists".into());}
 // Validate all selections before creating an output file.
 let name=CString::new(args[1].as_str()).map_err(|e|e.to_string())?;let mut id=0;
 check(unsafe{nc_open(name.as_ptr(),0,&mut id)})?;let source=File(id);let mut selected=Vec::new();
 for name in names{
  let name=CString::new(name.as_str()).map_err(|e|e.to_string())?;let mut var=0;
  check(unsafe{nc_inq_varid(source.0,name.as_ptr(),&mut var)})?;
  let mut ty=0;let mut nd=0;let mut dims=[0;1024];
  check(unsafe{nc_inq_var(source.0,var,std::ptr::null_mut(),&mut ty,&mut nd,dims.as_mut_ptr(),std::ptr::null_mut())})?;
  if ty!=5 && ty!=6{return Err("selected field is not IEEE float/double".into());}
  let mut shape=Vec::new();for d in &dims[..nd as usize]{let mut n=0;check(unsafe{nc_inq_dimlen(source.0,*d,&mut n)})?;shape.push(n);}
  selected.push((var,shape,ty));
 }
 std::fs::copy(&args[1],&args[2]).map_err(|e|e.to_string())?;
 // fs::copy preserves read-only publication permissions. Only the newly
 // created output must become writable; the immutable source stays untouched.
 let mode=std::fs::metadata(&args[2]).map_err(|e|e.to_string())?.permissions().mode();
 std::fs::set_permissions(&args[2],std::fs::Permissions::from_mode(mode|0o200)).map_err(|e|e.to_string())?;
 let name=CString::new(args[2].as_str()).map_err(|e|e.to_string())?;let mut id=0;
 check(unsafe{nc_open(name.as_ptr(),1,&mut id)})?;let output=File(id);
 for (var,shape,ty) in selected{
  let total=shape.iter().try_fold(1_usize,|a,&b|a.checked_mul(b)).ok_or("shape overflow")?;
  if total==0{continue;}
  let rows=shape.first().copied().unwrap_or(1);let width=total/rows;
  let batch=(262144/width.max(1)).max(1).min(rows);let mut start=vec![0;shape.len()];let mut count=shape.clone();let mut offset=0;
  while offset<rows{
   if !shape.is_empty(){start[0]=offset;count[0]=batch.min(rows-offset);}
   let n=count.iter().product::<usize>();let size=if ty==5{4}else{8};let zero=vec![0_u64;(n*size).div_ceil(8)];
   check(unsafe{nc_put_vara(output.0,var,start.as_ptr(),count.as_ptr(),zero.as_ptr().cast())})?;
   let mut read=vec![u64::MAX;zero.len()];check(unsafe{nc_get_vara(output.0,var,start.as_ptr(),count.as_ptr(),read.as_mut_ptr().cast())})?;
   let bytes=unsafe{std::slice::from_raw_parts(read.as_ptr().cast::<u8>(),n*size)};
   if bytes.iter().any(|&b|b!=0){return Err("stored selected field did not become IEEE +0".into());}
   offset+=batch;
  }
 }
 println!("copied native file and zeroed {} explicitly selected fields",args.len()-3);Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("{e}");std::process::exit(2);}}
