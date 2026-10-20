//! Lossless real restart inputs and strict native LES buoyancy replay.
//! MPAS v8.4.2 mpas_atm_dissipation_models.F:464-575 (native call).
//! qtot follows mpas_atm_time_integration.F:3235-3245, in Registry moist order.
//! This is a standalone diagnostic; no forecast acceptance is implied.
use std::{env, ffi::{CString, c_char, c_int, c_void}, fs, path::Path};

#[link(name="netcdf")]
extern "C" {
    fn nc_open(path:*const c_char, mode:c_int, id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_inq_varid(id:c_int, name:*const c_char, var:*mut c_int)->c_int;
    fn nc_inq_vartype(id:c_int, var:c_int, typ:*mut c_int)->c_int;
    fn nc_inq_varndims(id:c_int, var:c_int, n:*mut c_int)->c_int;
    fn nc_inq_vardimid(id:c_int, var:c_int, dims:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int, dim:c_int, len:*mut usize)->c_int;
    fn nc_get_var_float(id:c_int, var:c_int, data:*mut f32)->c_int;
}
#[link(name="dl")]
extern "C" {
    fn dlopen(path:*const c_char, mode:c_int)->*mut c_void;
    fn dlsym(handle:*mut c_void, name:*const c_char)->*mut c_void;
    fn dlclose(handle:*mut c_void)->c_int;
}
type Replay = unsafe extern "C" fn(*mut f32,*const f32,*const f32,*const f32,
    *const f32,*const f32,*const f32,*const i32,*const i32,*const f32,
    *const i32,*const i32,*const i32,*const i32,*const i32,*const i32);

fn checked(code:c_int) { assert_eq!(code,0,"NetCDF error {code}"); }
pub fn read(id:c_int, name:&str)->(Vec<f32>,Vec<usize>) {
    let (mut var,mut typ,mut rank)=(0,0,0);
    unsafe {
        checked(nc_inq_varid(id,CString::new(name).unwrap().as_ptr(),&mut var));
        checked(nc_inq_vartype(id,var,&mut typ));
        assert_eq!(typ,5,"{name}: require stored RKIND float32, no conversion");
        checked(nc_inq_varndims(id,var,&mut rank));
    }
    let mut dims=vec![0;rank as usize]; let mut shape=Vec::new();
    unsafe {
        checked(nc_inq_vardimid(id,var,dims.as_mut_ptr()));
        for dim in dims {let mut size=0;checked(nc_inq_dimlen(id,dim,&mut size));shape.push(size);}
    }
    let mut data=vec![0.0;shape.iter().product()];
    unsafe {checked(nc_get_var_float(id,var,data.as_mut_ptr()));}
    (data,shape)
}
fn field(id:c_int,name:&str,nc:usize,nv:usize)->Vec<f32> {
    let(mut data,shape)=read(id,name);
    assert!(shape==[nc,nv] || shape==[1,nc,nv],"{name}: unexpected shape {shape:?}");
    // NetCDF C order (cell,level) is native Fortran order (level,cell).
    // One extra native sentinel column; never sampled as a physical cell.
    data.resize((nc+1)*nv,0.0); data
}
fn write(path:&Path, data:&[f32]) {
    let bytes:Vec<u8>=data.iter().flat_map(|x|x.to_le_bytes()).collect();
    fs::write(path,bytes).unwrap();
}
fn load(path:&Path)->Vec<f32> {
    let bytes=fs::read(path).unwrap();assert_eq!(bytes.len()%4,0);
    bytes.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect()
}
fn main() {
    let args:Vec<String>=env::args().collect();
    assert_eq!(args.len(),5,"extract RESTART.nc OUTPUT_DIR LIBRARY, or replay INPUT_DIR OUTPUT_DIR LIBRARY");
    let out=Path::new(&args[3]);fs::create_dir_all(out).unwrap();
    let (nc,nv,fields)=if args[1]=="extract" {
        let mut id=0;unsafe {checked(nc_open(CString::new(args[2].as_str()).unwrap().as_ptr(),0,&mut id));}
        let (_,shape)=read(id,"theta_m");assert_eq!(shape.len(),3);assert_eq!(shape[0],1);
        let(nc,nv)=(shape[1],shape[2]);assert!(nc>0 && nv>=3);
        let mut scalars=vec![0.0;6*nv*(nc+1)];let mut qtot=vec![0.0;nv*(nc+1)];
        let mut moist=0;
        for (iq,name) in ["qv","qc","qr","qi","qs","qg"].iter().enumerate() {
            let data=field(id,name,nc,nv);
            for j in 0..nv*nc {
                scalars[6*j+iq]=data[j];
                qtot[j]+=data[j]; // Native serial binary32 multiply-free sum.
                if iq==1 && j%nv>0 && j%nv<nv-1 && data[j]>=0.00001 {moist+=1;}
            }
        }
        let fields=vec![field(id,"theta_m",nc,nv),field(id,"exner",nc,nv),
            field(id,"pressure_base",nc,nv),field(id,"pressure_p",nc,nv),
            field(id,"zgrid",nc,nv+1),scalars,qtot];
        unsafe {checked(nc_close(id));}
        fs::write(out.join("dimensions.txt"),format!("{nc} {nv}\n")).unwrap();
        fs::write(out.join("provenance.json"),format!(
            "{{\"kind\":\"real native restart buoyancy inputs\",\"nCells\":{nc},\"nVertLevels\":{nv},\"moist_interior_levels\":{moist},\"dry_interior_levels\":{},\"scalar_order\":[\"qv\",\"qc\",\"qr\",\"qi\",\"qs\",\"qg\"],\"qtot\":\"native serial RKIND sum, mpas_atm_time_integration.F:3235-3245\",\"sentinel_column\":\"zero; output marker 0.913\",\"whole_forecast_proven\":false}}\n",nc*(nv-2)-moist)).unwrap();
        (nc,nv,fields)
    } else {
        assert_eq!(args[1],"replay");let input=Path::new(&args[2]);
        let dims:Vec<usize>=fs::read_to_string(input.join("dimensions.txt")).unwrap().split_whitespace().map(|x|x.parse().unwrap()).collect();
        assert_eq!(dims.len(),2);
        let fields=["theta_m","exner","pressure_b","pp","zgrid","scalars","qtot"].iter().map(|n|load(&input.join(format!("{n}.bin")))).collect();
        fs::copy(input.join("dimensions.txt"),out.join("dimensions.txt")).unwrap();
        fs::copy(input.join("provenance.json"),out.join("provenance.json")).unwrap();
        (dims[0],dims[1],fields)
    };
    for (name,data) in ["theta_m","exner","pressure_b","pp","zgrid","scalars","qtot"].iter().zip(&fields) {
        write(&out.join(format!("{name}.bin")),data);
    }
    let sizes=[nv*(nc+1),nv*(nc+1),nv*(nc+1),nv*(nc+1),(nv+1)*(nc+1),6*nv*(nc+1),nv*(nc+1)];
    for (data,size) in fields.iter().zip(sizes) {assert_eq!(data.len(),size);}
    let handle=unsafe {dlopen(CString::new(args[4].as_str()).unwrap().as_ptr(),2)};
    assert!(!handle.is_null(),"cannot load strict native library");
    let symbol=unsafe {dlsym(handle,CString::new("replay_calculate_n2").unwrap().as_ptr())};
    assert!(!symbol.is_null(),"missing native boundary");
    let replay:Replay=unsafe {std::mem::transmute(symbol)};
    let (nc_i,nv_i,start,maxedges,ns,qv)=(i32::try_from(nc).unwrap(),i32::try_from(nv).unwrap(),1,1,6,1);
    for qc in [0,2] {
        let mut bn2=vec![0.913;nv*(nc+1)];
        unsafe {replay(bn2.as_mut_ptr(),fields[0].as_ptr(),fields[1].as_ptr(),fields[2].as_ptr(),fields[3].as_ptr(),fields[4].as_ptr(),fields[5].as_ptr(),&qv,&qc,fields[6].as_ptr(),&start,&nc_i,&nc_i,&nv_i,&maxedges,&ns);}
        write(&out.join(format!("bn2-qc{qc}.bin")),&bn2);
    }
    unsafe {assert_eq!(dlclose(handle),0);}
}
