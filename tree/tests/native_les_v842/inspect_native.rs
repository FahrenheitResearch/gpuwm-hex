//! Count non-finite stored floating words in a native NetCDF artifact.
//! Diagnostic only; comparison against Hex remains a separate acceptance.
use std::{env,ffi::{c_char,c_int,CString},fs};
#[link(name="netcdf")]
extern "C" {
    fn nc_open(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_inq_nvars(id:c_int,n:*mut c_int)->c_int;
    fn nc_inq_varname(id:c_int,var:c_int,name:*mut c_char)->c_int;
    fn nc_inq_vartype(id:c_int,var:c_int,typ:*mut c_int)->c_int;
    fn nc_inq_varndims(id:c_int,var:c_int,n:*mut c_int)->c_int;
    fn nc_inq_vardimid(id:c_int,var:c_int,dims:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
    fn nc_get_var_float(id:c_int,var:c_int,data:*mut f32)->c_int;
    fn nc_get_var_double(id:c_int,var:c_int,data:*mut f64)->c_int;
}
fn checked(code:c_int){assert_eq!(code,0,"netCDF error {code}");}
fn main(){
    let args:Vec<String>=env::args().collect();assert_eq!(args.len(),3,"inspect-native INPUT.nc RECEIPT.json");
    let(mut id,mut nvars)=(0,0);unsafe {
        checked(nc_open(CString::new(args[1].as_str()).unwrap().as_ptr(),0,&mut id));
        checked(nc_inq_nvars(id,&mut nvars));
    }
    let(mut rows,mut total,mut nonfinite)=(Vec::new(),0usize,0usize);
    for var in 0..nvars {
        let(mut name,mut typ,mut rank)=([0i8;257],0,0);
        unsafe {checked(nc_inq_varname(id,var,name.as_mut_ptr()));checked(nc_inq_vartype(id,var,&mut typ));}
        if typ!=5 && typ!=6 {continue;}
        unsafe {checked(nc_inq_varndims(id,var,&mut rank));}
        let mut dimensions=vec![0;rank as usize];let mut count=1;
        unsafe {
            checked(nc_inq_vardimid(id,var,dimensions.as_mut_ptr()));
            for dim in dimensions {let mut size=0;checked(nc_inq_dimlen(id,dim,&mut size));count*=size;}
        }
        let bad=if typ==5 {
            let mut data=vec![0f32;count];unsafe {checked(nc_get_var_float(id,var,data.as_mut_ptr()));}
            data.iter().filter(|x|!x.is_finite()).count()
        }else{
            let mut data=vec![0f64;count];unsafe {checked(nc_get_var_double(id,var,data.as_mut_ptr()));}
            data.iter().filter(|x|!x.is_finite()).count()
        };
        let name=unsafe {std::ffi::CStr::from_ptr(name.as_ptr())}.to_str().unwrap();
        rows.push(format!("{{\"field\":\"{name}\",\"type\":{typ},\"words\":{count},\"nonfinite\":{bad}}}"));
        total+=count;nonfinite+=bad;
    }
    unsafe {checked(nc_close(id));}
    fs::write(&args[2],format!("{{\"kind\":\"native output finite-word diagnostic; no Hex identity claimed\",\"floating_words\":{total},\"nonfinite\":{nonfinite},\"fields\":[{}]}}\n",rows.join(","))).unwrap();
    println!("floating_words={total} nonfinite={nonfinite}");
    if nonfinite!=0 {std::process::exit(1);}
}
