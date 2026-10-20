//! Decode a stored character field for an exact output mismatch witness.
//! Diagnostic only; this neither changes fields nor excludes native words.
use std::{env, ffi::{c_char, c_int, CString}, fs};
#[link(name = "netcdf")]
extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
    fn nc_inq_vartype(id: c_int, var: c_int, typ: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, var: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_vardimid(id: c_int, var: c_int, dims: *mut c_int) -> c_int;
    fn nc_inq_dimlen(id: c_int, dim: c_int, len: *mut usize) -> c_int;
    fn nc_get_var_text(id: c_int, var: c_int, data: *mut c_char) -> c_int;
}
fn checked(code: c_int) { assert_eq!(code, 0, "netCDF error {code}"); }
fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 4, "inspect-text INPUT.nc FIELD RECEIPT.json");
    let (mut id, mut var, mut typ, mut rank) = (0, 0, 0, 0);
    unsafe {
        checked(nc_open(CString::new(args[1].as_str()).unwrap().as_ptr(), 0, &mut id));
        checked(nc_inq_varid(id, CString::new(args[2].as_str()).unwrap().as_ptr(), &mut var));
        checked(nc_inq_vartype(id, var, &mut typ));
        checked(nc_inq_varndims(id, var, &mut rank));
    }
    assert_eq!(typ, 2, "field must contain NC_CHAR storage");
    let mut dims = vec![0; rank as usize];
    let mut shape = vec![0; rank as usize];
    unsafe {
        checked(nc_inq_vardimid(id, var, dims.as_mut_ptr()));
        for (i, dim) in dims.iter().enumerate() { checked(nc_inq_dimlen(id, *dim, &mut shape[i])); }
    }
    let mut data = vec![0u8; shape.iter().product()];
    unsafe {
        checked(nc_get_var_text(id, var, data.as_mut_ptr().cast()));
        checked(nc_close(id));
    }
    fs::write(&args[3], format!("{{\"schema\":\"d2-les-stored-character-witness-v1\",\"shape\":{shape:?},\"bytes\":{data:?}}}\n")).unwrap();
}
