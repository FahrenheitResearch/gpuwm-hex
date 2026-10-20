//! Test-only native init reader. Rust owns netCDF decoding; Python only packs ABI arrays.
use std::{ffi::{CString, CStr}, fs, io::Write, os::raw::{c_char,c_int}, path::Path};

#[link(name="netcdf")]
unsafe extern "C" {
    fn nc_open(path:*const c_char, mode:c_int, id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_strerror(code:c_int)->*const c_char;
    fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
    fn nc_inq_var(id:c_int,var:c_int,name:*mut c_char,kind:*mut c_int,
                  rank:*mut c_int,dims:*mut c_int,attrs:*mut c_int)->c_int;
    fn nc_inq_dim(id:c_int,dim:c_int,name:*mut c_char,len:*mut usize)->c_int;
    fn nc_get_vara_float(id:c_int,var:c_int,start:*const usize,count:*const usize,
                         values:*mut f32)->c_int;
}

fn check(code:c_int) {
    if code!=0 {
        panic!("netCDF: {}",unsafe{CStr::from_ptr(nc_strerror(code))}.to_string_lossy());
    }
}
struct File(c_int);
impl Drop for File {fn drop(&mut self){check(unsafe{nc_close(self.0)});}}

fn field(file:&File,name:&str)->(Vec<f32>,usize,usize) {
    let mut var=0;
    check(unsafe{nc_inq_varid(file.0,CString::new(name).unwrap().as_ptr(),&mut var)});
    let (mut kind,mut rank)=(0,0);
    let mut dims=[0;1024];
    check(unsafe{nc_inq_var(file.0,var,std::ptr::null_mut(),&mut kind,&mut rank,
                          dims.as_mut_ptr(),std::ptr::null_mut())});
    assert_eq!(kind,5,"{name}: expected stored NC_FLOAT");
    assert!(rank==2 || rank==3,"{name}: expected [Time,] nCells, nVertLevels");
    let mut counts=Vec::new();
    let mut names=Vec::new();
    for dim in &dims[..rank as usize] {
        let mut length=0;
        let mut label=[0 as c_char;257];
        check(unsafe{nc_inq_dim(file.0,*dim,label.as_mut_ptr(),&mut length)});
        counts.push(length);
        names.push(unsafe{CStr::from_ptr(label.as_ptr())}.to_str().unwrap().to_owned());
    }
    if rank==3 {
        assert_eq!(names[0],"Time");
        assert!(counts[0]>0);
        counts[0]=1;
    }
    assert_eq!(&names[names.len()-2..],&["nCells","nVertLevels"]);
    let (nc,nl)=(counts[counts.len()-2],counts[counts.len()-1]);
    let starts=vec![0;rank as usize];
    let mut values=vec![0.;nc*nl];
    check(unsafe{nc_get_vara_float(file.0,var,starts.as_ptr(),counts.as_ptr(),values.as_mut_ptr())});
    (values,nc,nl)
}

fn main() {
    let args:Vec<_>=std::env::args().collect();
    assert_eq!(args.len(),3,"usage: reader INIT.nc OUTPUT_DIR");
    let mut id=0;
    check(unsafe{nc_open(CString::new(args[1].as_str()).unwrap().as_ptr(),0,&mut id)});
    let file=File(id);
    let out=Path::new(&args[2]);fs::create_dir_all(out).unwrap();
    let mut shape=None;
    for name in ["rho","zz","theta","qv","qc","qr"] {
        let (values,nc,nl)=field(&file,name);
        if let Some(old)=shape {assert_eq!(old,(nc,nl));} else {shape=Some((nc,nl));}
        let mut target=fs::File::create(out.join(format!("{name}.bin"))).unwrap();
        for value in values {target.write_all(&value.to_bits().to_le_bytes()).unwrap();}
    }
    let (nc,nl)=shape.unwrap();
    fs::write(out.join("shape.json"),format!("{{\"n_cells\":{nc},\"n_levels\":{nl},\"layout\":\"cell-level\",\"dtype\":\"little-endian-f32\"}}\n")).unwrap();
}
