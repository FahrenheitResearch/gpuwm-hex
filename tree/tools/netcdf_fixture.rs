//! Generic reference-fixture population and lossless variable extraction.
use std::{env, ffi::CString, fs, os::raw::{c_char,c_int}, path::Path};
#[link(name="netcdf")]
extern "C" {
    fn nc_open(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
    fn nc_inq_vartype(id:c_int,var:c_int,typ:*mut c_int)->c_int;
    fn nc_inq_varndims(id:c_int,var:c_int,n:*mut c_int)->c_int;
    fn nc_inq_vardimid(id:c_int,var:c_int,dims:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
    fn nc_get_var_float(id:c_int,var:c_int,data:*mut f32)->c_int;
    fn nc_get_var_double(id:c_int,var:c_int,data:*mut f64)->c_int;
    fn nc_put_var_float(id:c_int,var:c_int,data:*const f32)->c_int;
}
fn checked(code:c_int){assert_eq!(code,0,"NetCDF error {code}");}
fn open(path:&str,write:bool)->c_int {let mut id=0;unsafe{checked(nc_open(CString::new(path).unwrap().as_ptr(),if write{1}else{0},&mut id));}id}
fn variable(id:c_int,name:&str)->(c_int,c_int,Vec<usize>){
    let(mut var,mut typ,mut n)=(0,0,0);
    unsafe{checked(nc_inq_varid(id,CString::new(name).unwrap().as_ptr(),&mut var));checked(nc_inq_vartype(id,var,&mut typ));checked(nc_inq_varndims(id,var,&mut n));}
    let mut dims=vec![0;n as usize];let mut shape=Vec::new();
    unsafe{checked(nc_inq_vardimid(id,var,dims.as_mut_ptr()));for d in dims{let mut len=0;checked(nc_inq_dimlen(id,d,&mut len));shape.push(len);}}
    (var,typ,shape)
}
fn main(){
    let args:Vec<String>=env::args().collect();assert!(args.len()>=5,"seed INPUT OUTPUT VARIABLE SCALE, or extract INPUT DIRECTORY VARIABLE...");
    if args[1]=="seed" {
        assert_eq!(args.len(),6,"seed INPUT OUTPUT VARIABLE SCALE");
        fs::copy(&args[2],&args[3]).unwrap();let id=open(&args[3],true);
        let(var,typ,shape)=variable(id,&args[4]);assert_eq!(typ,5,"seed requires native float32");
        let scale:f32=args[5].parse().unwrap();assert!(scale>0.0&&scale.is_finite());
        let total:usize=shape.iter().product();let levels=*shape.last().unwrap();
        let mut values=vec![0.0_f32;total];
        for (index,value) in values.iter_mut().enumerate(){let cell=index/levels;let k=index%levels;
            *value=scale*(0.25+0.75*((cell%17) as f32/16.0))*(0.25+0.75*(k as f32/(levels-1) as f32));}
        unsafe{checked(nc_put_var_float(id,var,values.as_ptr()));checked(nc_close(id));}
    } else {
        assert_eq!(args[1],"extract");let id=open(&args[2],false);let out=Path::new(&args[3]);fs::create_dir_all(out).unwrap();
        let mut metadata=Vec::new();
        for name in &args[4..]{assert!(name.chars().all(|x|x.is_ascii_alphanumeric()||x=='_'));
            let(var,typ,shape)=variable(id,name);let size:usize=shape.iter().product();let mut bytes=Vec::new();
            let dtype=if typ==5{let mut values=vec![0.0_f32;size];unsafe{checked(nc_get_var_float(id,var,values.as_mut_ptr()));}for value in values{bytes.extend_from_slice(&value.to_le_bytes());}"<f4"}
                else{assert_eq!(typ,6,"extract supports floating variables only");let mut values=vec![0.0_f64;size];unsafe{checked(nc_get_var_double(id,var,values.as_mut_ptr()));}for value in values{bytes.extend_from_slice(&value.to_le_bytes());}"<f8"};
            fs::write(out.join(format!("{name}.bin")),bytes).unwrap();metadata.push(format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?}}}"));
        }
        unsafe{checked(nc_close(id));}fs::write(out.join("variables.json"),format!("{{{}}}\n",metadata.join(","))).unwrap();
    }
}
