//! Two-record transport diagnostic. No acceptance analysis is synthesized.
//! Registry.xml:1173-1183,3972-4011 names the actual IAU fields.
use std::{env, ffi::{CString, c_char, c_int}, fs, path::Path};
#[link(name="netcdf")]
unsafe extern "C" {
    fn nc_create(p:*const c_char,m:c_int,id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_def_dim(id:c_int,n:*const c_char,l:usize,d:*mut c_int)->c_int;
    fn nc_def_var(id:c_int,n:*const c_char,k:c_int,r:c_int,d:*const c_int,v:*mut c_int)->c_int;
    fn nc_enddef(id:c_int)->c_int;
    fn nc_put_vara_float(id:c_int,v:c_int,s:*const usize,c:*const usize,p:*const f32)->c_int;
    fn nc_put_vara_int(id:c_int,v:c_int,s:*const usize,c:*const usize,p:*const i32)->c_int;
    fn nc_put_vara_text(id:c_int,v:c_int,s:*const usize,c:*const usize,p:*const c_char)->c_int;
}
fn check(code:c_int) { assert_eq!(code,0,"netCDF fixture status {code}"); }
fn c(s:&str)->CString { CString::new(s).unwrap() }
fn main() {
    let a:Vec<_>=env::args().collect();
    assert_eq!(a.len(),2,"multirecord_fixture OWNED_NEW_DIRECTORY");
    let root=Path::new(&a[1]); fs::create_dir(root).unwrap();
    let path=root.join("increments.nc");
    let mut id=0;
    unsafe {check(nc_create(c(path.to_str().unwrap()).as_ptr(),4,&mut id));}
    let mut d=[0;5];
    for (i,(name,size)) in [("Time",2),("nCells",2),("nEdges",3),("nVertLevels",2),("StrLen",64)].iter().enumerate() {
        unsafe {check(nc_def_dim(id,c(name).as_ptr(),*size,&mut d[i]));}
    }
    let names=["u_amb","rho_amb","theta_amb","qv_amb","qc_amb","qr_amb","qi_amb","qs_amb","qg_amb"];
    let mut vars=Vec::new();
    for (i,name) in names.iter().enumerate() {
        let mut v=0;
        unsafe {check(nc_def_var(id,c(name).as_ptr(),5,3,[d[0],d[if i==0 {2}else{1}],d[3]].as_ptr(),&mut v));}
        vars.push(v);
    }
    let (mut ids,mut xtime)=(0,0);
    unsafe {
        check(nc_def_var(id,c("indexToCellID").as_ptr(),4,2,[d[0],d[1]].as_ptr(),&mut ids));
        check(nc_def_var(id,c("xtime").as_ptr(),2,2,[d[0],d[4]].as_ptr(),&mut xtime));
        check(nc_enddef(id));
    }
    let patterns=[0x00000001u32,0x80000001,0x3f800001,0x007fffff,0x40800000,0x80800000];
    let mut scalar_expected=vec![0u8;6*2*3*4];
    for (i,(&v,name)) in vars.iter().zip(names).enumerate() {
        let entities=if i==0 {3}else{2};
        let wrong=vec![16.0f32;entities*2];
        let words:Vec<_>=(0..entities*2).map(|j|patterns[(j+i)%patterns.len()]).collect();
        let correct:Vec<_>=words.iter().map(|w|f32::from_bits(*w)).collect();
        unsafe {
            check(nc_put_vara_float(id,v,[0,0,0].as_ptr(),[1,entities,2].as_ptr(),wrong.as_ptr()));
            check(nc_put_vara_float(id,v,[1,0,0].as_ptr(),[1,entities,2].as_ptr(),correct.as_ptr()));
        }
        let mut expected=Vec::new();
        for &word in &words {expected.extend(word.to_le_bytes());}
        expected.extend([0u8;8]);
        fs::write(root.join(format!("{name}.expected.bin")),expected).unwrap();
        if i>=3 {
            for (j,word) in words.iter().enumerate() {
                let offset=4*((i-3)+6*j);
                scalar_expected[offset..offset+4].copy_from_slice(&word.to_le_bytes());
            }
        }
    }
    fs::write(root.join("scalars_amb.expected.bin"),scalar_expected).unwrap();
    let wrong_ids=[-11i32,-12]; let right_ids=[7i32,3];
    unsafe {
        check(nc_put_vara_int(id,ids,[0,0].as_ptr(),[1,2].as_ptr(),wrong_ids.as_ptr()));
        check(nc_put_vara_int(id,ids,[1,0].as_ptr(),[1,2].as_ptr(),right_ids.as_ptr()));
    }
    fs::write(root.join("indexToCellID.expected.bin"),[7i32,3,0].into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>()).unwrap();
    for (record,stamp) in ["2001-01-01_00:00:00","2001-01-01_06:00:00"].iter().enumerate() {
        let mut text=[b' ';64];text[..stamp.len()].copy_from_slice(stamp.as_bytes());
        unsafe {check(nc_put_vara_text(id,xtime,[record,0].as_ptr(),[1,64].as_ptr(),text.as_ptr().cast()));}
    }
    unsafe {check(nc_close(id));}
    println!("two actual NetCDF Time records written; record1 preserves subnormal words");
}
