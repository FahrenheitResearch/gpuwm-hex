//! Standalone test fixture: decode initialized surface observations with Rust.
//! This is not a capture of MPAS's prepared lowest-level physics call.
//! Atmospheric inputs use initialized 2m T/q and 10m wind at their observed
//! heights (dz8w=20m); pressure uses surface pressure. Other state is explicit.
use std::{ffi::CString,io::{self,Write},os::raw::{c_char,c_int}};
#[link(name="netcdf")]
unsafe extern "C" {
 fn nc_open(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
 fn nc_close(id:c_int)->c_int;
 fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
 fn nc_inq_varndims(id:c_int,var:c_int,n:*mut c_int)->c_int;
 fn nc_inq_vardimid(id:c_int,var:c_int,dims:*mut c_int)->c_int;
 fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
 fn nc_inq_dimid(id:c_int,name:*const c_char,dim:*mut c_int)->c_int;
 fn nc_get_vara_float(id:c_int,var:c_int,start:*const usize,count:*const usize,out:*mut f32)->c_int;
}
fn check(code:i32) { assert_eq!(code,0,"netCDF error {code}"); }
fn read(id:i32,name:&str,n:usize)->Vec<f32> {
 let key=CString::new(name).unwrap(); let mut var=0;let mut rank=0;
 unsafe {check(nc_inq_varid(id,key.as_ptr(),&mut var));check(nc_inq_varndims(id,var,&mut rank));}
 let mut dims=vec![0;rank as usize];let mut count=vec![1;rank as usize];
 unsafe {check(nc_inq_vardimid(id,var,dims.as_mut_ptr()));}
 let mut cells=0;unsafe{check(nc_inq_dimid(id,c"nCells".as_ptr(),&mut cells));}
 for (i,dim) in dims.iter().enumerate() { if *dim==cells {count[i]=n;} }
 assert_eq!(count.iter().product::<usize>(),n,"unexpected dimensions for {name}");
 let mut out=vec![0.0;n];let start=vec![0;rank as usize];
 unsafe {check(nc_get_vara_float(id,var,start.as_ptr(),count.as_ptr(),out.as_mut_ptr()));}
 out
}
fn main() {
 let path=CString::new(std::env::args().nth(1).expect("case_columns INIT.nc")).unwrap();
 let mut id=0;unsafe{check(nc_open(path.as_ptr(),0,&mut id));}
 let mut dim=0;let mut n=0;unsafe {check(nc_inq_dimid(id,c"nCells".as_ptr(),&mut dim));check(nc_inq_dimlen(id,dim,&mut n));}
 let mut a=vec![0.0f32;48*n];
 for (row,name) in [(34,"u10"),(35,"v10"),(36,"t2m"),(37,"q2"),(38,"surface_pressure"),(40,"surface_pressure"),(41,"skintemp"),(44,"xland")] {
  let v=read(id,name,n);a[row*n..(row+1)*n].copy_from_slice(&v);
 }
 for i in 0..n {
  // Explicit antecedent/LSM fixture values; not guessed native diagnostic state.
  a[i]=if a[44*n+i]>1.5 {0.0001} else {0.1}; // ZNT
  for (row,value) in [(1,0.2),(2,0.2),(3,0.0),(39,20.0),(42,1000.0),(43,1.0),(46,480000.0)] {a[row*n+i]=value;}
  for row in [34,35,36,37,38,40,41,44] {assert!(a[row*n+i].is_finite(),"nonfinite {row} cell {i}");}
  assert!(a[36*n+i]>150.0 && a[38*n+i]>1000.0,"uninitialized surface input at cell {i}");
 }
 unsafe{check(nc_close(id));}
 let mut stdout=io::stdout().lock();stdout.write_all(&(n as u64).to_le_bytes()).unwrap();
 for v in a {stdout.write_all(&v.to_le_bytes()).unwrap();}
}
