//! Controlled nonzero IAU input for real-state native capture, not an analysis.
//! Registry.xml:1173-1183,3972-4011 defines the immutable stream fields.
use std::{ffi::{CString, c_char, c_int}, env};

#[link(name="netcdf")]
unsafe extern "C" {
    fn nc_open(path:*const c_char, mode:c_int, id:*mut c_int)->c_int;
    fn nc_create(path:*const c_char, mode:c_int, id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_inq_dimid(id:c_int, name:*const c_char, dim:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int, dim:c_int, len:*mut usize)->c_int;
    fn nc_inq_varid(id:c_int, name:*const c_char, var:*mut c_int)->c_int;
    fn nc_get_vara_text(id:c_int, var:c_int, start:*const usize,
                        count:*const usize, value:*mut c_char)->c_int;
    fn nc_def_dim(id:c_int, name:*const c_char, len:usize, dim:*mut c_int)->c_int;
    fn nc_def_var(id:c_int, name:*const c_char, kind:c_int, rank:c_int,
                  dims:*const c_int, var:*mut c_int)->c_int;
    fn nc_enddef(id:c_int)->c_int;
    fn nc_put_vara_float(id:c_int, var:c_int, start:*const usize,
                         count:*const usize, value:*const f32)->c_int;
    fn nc_put_vara_text(id:c_int, var:c_int, start:*const usize,
                        count:*const usize, value:*const c_char)->c_int;
}
fn c(s:&str)->CString { CString::new(s).unwrap() }
fn check(status:c_int) { assert_eq!(status,0,"netCDF status {status}"); }
fn main() {
    let args:Vec<_>=env::args().collect();
    assert!(args.len()==3 || args.len()==4,
            "increment_fixture <native init> <new output> [earlier diagnostic timestamp]");
    let mut input=0;
    unsafe { check(nc_open(c(&args[1]).as_ptr(),0,&mut input)); }
    let mut lengths=Vec::new();
    for name in ["nVertLevels","nCells","nEdges","StrLen"] {
        let (mut dim,mut len)=(0,0);
        unsafe { check(nc_inq_dimid(input,c(name).as_ptr(),&mut dim));
            check(nc_inq_dimlen(input,dim,&mut len)); }
        lengths.push(len);
    }
    let mut xtime=vec![b' ';lengths[3]];
    let mut var=0;
    unsafe { check(nc_inq_varid(input,c("xtime").as_ptr(),&mut var));
        check(nc_get_vara_text(input,var,[0,0].as_ptr(),[1,lengths[3]].as_ptr(),
                              xtime.as_mut_ptr().cast()));
        check(nc_close(input)); }
    let mut output=0;
    // NC_NOCLOBBER: fixtures may never overwrite a reference or existing run.
    unsafe { check(nc_create(c(&args[2]).as_ptr(),4,&mut output)); }
    let mut dims=[0;5];
    for (i,(name,len)) in ["nVertLevels","nCells","nEdges","StrLen","Time"]
        .into_iter().zip(lengths.iter().copied().chain([0])).enumerate() {
        unsafe { check(nc_def_dim(output,c(name).as_ptr(),len,&mut dims[i])); }
    }
    let fields=[("u_amb",2,0.25f32),("rho_amb",1,0.00001),
        ("theta_amb",1,-0.125),("qv_amb",1,0.000001),("qc_amb",1,0.0000001),
        ("qr_amb",1,-0.0000001),("qi_amb",1,0.0000001),
        ("qs_amb",1,-0.0000001),("qg_amb",1,0.0000001)];
    let mut variables=Vec::new();
    for &(name,entity,_) in &fields {
        let mut v=0;
        unsafe { check(nc_def_var(output,c(name).as_ptr(),5,3,
                                 [dims[4],dims[entity],dims[0]].as_ptr(),&mut v)); }
        variables.push(v);
    }
    let mut time_var=0;
    unsafe { check(nc_def_var(output,c("xtime").as_ptr(),2,2,
                             [dims[4],dims[3]].as_ptr(),&mut time_var));
        check(nc_enddef(output)); }
    let selected_record=usize::from(args.len()==4);
    for (&(_,entity,value),v) in fields.iter().zip(variables) {
        let count=[1,lengths[entity],lengths[0]];
        let data=vec![value;count.iter().product()];
        unsafe { check(nc_put_vara_float(output,v,[selected_record,0,0].as_ptr(),count.as_ptr(),data.as_ptr())); }
        if selected_record==1 {
            let wrong=vec![16.0f32;data.len()];
            unsafe {check(nc_put_vara_float(output,v,[0,0,0].as_ptr(),count.as_ptr(),wrong.as_ptr()));}
        }
    }
    if selected_record==1 {
        assert_eq!(args[3].len(),19,"earlier diagnostic timestamp must be an MPAS timestamp");
        let mut earlier=vec![b' ';lengths[3]];
        earlier[..19].copy_from_slice(args[3].as_bytes());
        unsafe {check(nc_put_vara_text(output,time_var,[0,0].as_ptr(),
                                      [1,lengths[3]].as_ptr(),earlier.as_ptr().cast()));}
    }
    unsafe { check(nc_put_vara_text(output,time_var,[selected_record,0].as_ptr(),
                                  [1,lengths[3]].as_ptr(),xtime.as_ptr().cast()));
        check(nc_close(output)); }
    println!("controlled increments: levels={} cells={} edges={}; not analysis minus forecast",
             lengths[0],lengths[1],lengths[2]);
}
