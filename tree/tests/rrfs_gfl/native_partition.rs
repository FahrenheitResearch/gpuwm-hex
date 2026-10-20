//! A fixed balanced partition for the two-rank native defect measurement.
//! This is a diagnostic decomposition, not a canonical forecast proof deck.
//! Fork framework/mpas_block_decomp.F:120-158 at 1d6e370e reads one block ID
//! for each global cell. Rust owns the NetCDF dimension read and file encode.
use std::{env,ffi::{CString,c_char,CStr},fs::OpenOptions,io::{self,Write}};
#[link(name="netcdf")]
unsafe extern "C" {
 fn nc_open(path:*const c_char,mode:i32,id:*mut i32)->i32;
 fn nc_inq_dimid(id:i32,name:*const c_char,dim:*mut i32)->i32;
 fn nc_inq_dimlen(id:i32,dim:i32,len:*mut usize)->i32;
 fn nc_close(id:i32)->i32;
 fn nc_strerror(status:i32)->*const c_char;
}
fn check(status:i32)->io::Result<()> {
 if status!=0 {Err(io::Error::other(unsafe{CStr::from_ptr(nc_strerror(status))}.to_string_lossy().into_owned()))}
 else{Ok(())}
}
fn main()->io::Result<()> {
 let args:Vec<_>=env::args().collect();if args.len()!=4{return Err(io::Error::other("native_partition MESH RANKS OUTPUT"))}
 let ranks:usize=args[2].parse().map_err(io::Error::other)?;
 if ![2,4].contains(&ranks){return Err(io::Error::other("diagnostic supports two/four fixed ranks"))}
 let path=CString::new(args[1].as_str()).map_err(io::Error::other)?;
 let name=CString::new("nCells").unwrap();let mut id=0;let mut dim=0;let mut count=0;
 unsafe {check(nc_open(path.as_ptr(),0,&mut id))?;check(nc_inq_dimid(id,name.as_ptr(),&mut dim))?;
  check(nc_inq_dimlen(id,dim,&mut count))?;check(nc_close(id))?;}
 if count<ranks || count>100000000 {return Err(io::Error::other("invalid native global cell count"))}
 let mut out=OpenOptions::new().write(true).create_new(true).open(&args[3])?;
 for cell in 0..count {writeln!(out,"{}",cell*ranks/count)?;}
 println!("{{\"scope\":\"diagnostic balanced decomposition\",\"nCells\":{count},\"process_count\":{ranks}}}");
 Ok(())
}
