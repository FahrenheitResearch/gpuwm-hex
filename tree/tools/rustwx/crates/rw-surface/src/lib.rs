//! Read real MPAS surface inputs into caller-owned replay buffers.
//! NetCDF decoding stays in Rust; no weather data are decoded in Python.
use std::os::raw::{c_char, c_int};
// libnetcdf/HDF5 are not thread safe in every wheel build. Serialize their
// short native reads when multiple resident ranks ring the same alarm.
static NETCDF: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg_attr(feature = "static-netcdf", link(name = "netcdf", kind = "static"))]
#[cfg_attr(not(feature = "static-netcdf"), link(name = "netcdf"))]
extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
    fn nc_inq_vartype(id: c_int, var: c_int, typ: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, var: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_vardimid(id: c_int, var: c_int, dims: *mut c_int) -> c_int;
    fn nc_inq_dimid(id: c_int, name: *const c_char, dim: *mut c_int) -> c_int;
    fn nc_inq_dimlen(id: c_int, dim: c_int, len: *mut usize) -> c_int;
    fn nc_get_vara_float(id: c_int, var: c_int, start: *const usize,
                         count: *const usize, data: *mut f32) -> c_int;
    fn nc_get_vara_int(id: c_int, var: c_int, start: *const usize,
                       count: *const usize, data: *mut c_int) -> c_int;
    fn nc_get_vara_text(id: c_int, var: c_int, start: *const usize,
                       count: *const usize, data: *mut c_char) -> c_int;
}

/// Return the exact native stream extent, or a negative NetCDF error.
#[no_mangle]
pub unsafe extern "C" fn surface_cell_count(path: *const c_char) -> i64 {
    let _guard = NETCDF.lock().unwrap_or_else(|p| p.into_inner());
    if path.is_null() {return -10003;}
    let mut id=0;
    let rc=nc_open(path,0,&mut id);if rc!=0{return rc as i64;}
    let (mut dim,mut count)=(0,0);
    let mut rc=nc_inq_dimid(id,b"nCells\0".as_ptr().cast(),&mut dim);
    if rc==0 {rc=nc_inq_dimlen(id,dim,&mut count);}
    let close=nc_close(id);
    if rc!=0 {rc as i64} else if close!=0 {close as i64} else {count as i64}
}

/// Decode the native surface stream's timestamp (Registry.xml xtime).
/// Return its byte length, or a negative NetCDF/schema error. No NUL is added.
#[no_mangle]
pub unsafe extern "C" fn read_surface_time(path: *const c_char, record: usize,
    output: *mut c_char, capacity: usize) -> c_int {
    let _guard = NETCDF.lock().unwrap_or_else(|p| p.into_inner());
    if path.is_null() || output.is_null() {return -10003;}
    let mut id=0;let rc=nc_open(path,0,&mut id);if rc!=0{return rc;}
    let check=|rc|if rc==0{Ok(())}else{Err(rc)};
    let result=(|| {
        let (mut var,mut typ,mut rank,mut time,mut width)=(0,0,0,0,0);
        check(nc_inq_varid(id,b"xtime\0".as_ptr().cast(),&mut var))?;
        check(nc_inq_vartype(id,var,&mut typ))?;
        check(nc_inq_varndims(id,var,&mut rank))?;
        if typ!=2 || rank!=2{return Err(-10001);}
        let mut dims=[0;2];check(nc_inq_vardimid(id,var,dims.as_mut_ptr()))?;
        check(nc_inq_dimid(id,b"Time\0".as_ptr().cast(),&mut time))?;
        check(nc_inq_dimid(id,b"StrLen\0".as_ptr().cast(),&mut width))?;
        if dims!=[time,width]{return Err(-10002);}
        let (mut records,mut bytes)=(0,0);
        check(nc_inq_dimlen(id,time,&mut records))?;
        check(nc_inq_dimlen(id,width,&mut bytes))?;
        if record>=records{return Err(-10004);}
        if bytes>capacity || bytes>c_int::MAX as usize{return Err(-10005);}
        check(nc_get_vara_text(id,var,[record,0].as_ptr(),[1,bytes].as_ptr(),output))?;
        Ok(bytes as c_int)
    })();
    let close=nc_close(id);
    match result {Ok(bytes)=>if close==0{bytes}else{close},Err(rc)=>rc}
}
unsafe fn read(id: c_int, name: &[u8], typ: c_int, cells: c_int, months: c_int,
               first: usize, n: usize, output: *mut std::ffi::c_void) -> Result<(), c_int> {
    let (mut var, mut actual_type, mut rank) = (0,0,0);
    let check = |rc| if rc == 0 { Ok(()) } else { Err(rc) };
    check(nc_inq_varid(id,name.as_ptr().cast(),&mut var))?;
    check(nc_inq_vartype(id,var,&mut actual_type))?;
    check(nc_inq_varndims(id,var,&mut rank))?;
    let expected_rank=if typ==5 {2} else {1};
    if actual_type!=typ || rank!=expected_rank {return Err(-10001);}
    let mut dims=[0;2];
    check(nc_inq_vardimid(id,var,dims.as_mut_ptr()))?;
    if dims[0]!=cells || (rank==2 && dims[1]!=months) {return Err(-10002);}
    let start=[first,0];let count=[n,12];
    check(if typ==5 {nc_get_vara_float(id,var,start.as_ptr(),count.as_ptr(),output.cast())}
          else {nc_get_vara_int(id,var,start.as_ptr(),count.as_ptr(),output.cast())})
}
/// Outputs hold n*12 float32 words each and n int32 land-mask words.
/// NetCDF (cell, month) layout is directly MPAS Fortran (month, cell).
#[no_mangle]
pub unsafe extern "C" fn read_monthly(path: *const c_char, first: usize, n: usize,
    green: *mut f32, albedo: *mut f32, land: *mut c_int) -> c_int {
    let _guard = NETCDF.lock().unwrap_or_else(|p| p.into_inner());
    if path.is_null() || green.is_null() || albedo.is_null() || land.is_null() || n==0 {return -10003;}
    let mut id=0;let rc=nc_open(path,0,&mut id);if rc!=0{return rc;}
    let result=(|| {
        let (mut cells,mut months,mut ncells,mut nmonths)=(0,0,0,0);
        for (name,dim,len) in [(b"nCells\0".as_slice(),&mut cells,&mut ncells),
                               (b"nMonths\0".as_slice(),&mut months,&mut nmonths)] {
            let rc=nc_inq_dimid(id,name.as_ptr().cast(),dim);if rc!=0{return Err(rc);}
            let rc=nc_inq_dimlen(id,*dim,len);if rc!=0{return Err(rc);}
        }
        if nmonths!=12 || first.checked_add(n).map_or(true,|end|end>ncells){return Err(-10004);}
        read(id,b"greenfrac\0",5,cells,months,first,n,green.cast())?;
        read(id,b"albedo12m\0",5,cells,months,first,n,albedo.cast())?;
        read(id,b"landmask\0",4,cells,months,first,n,land.cast())
    })();
    let close=nc_close(id);
    match result {Ok(())=>close,Err(rc)=>rc}
}

/// Read one static or time-dependent field without type/layout conversion.
/// Registry.xml:2561-2579,3463-3479,3783-3855 (MPAS-A v8.4.2).
/// NetCDF (Time?, cell, level?) is directly Fortran (level?, cell).
/// axis is null for a scalar or cell field, otherwise the Registry dimension.
/// The caller supplies the output capacity in 32-bit words.
#[no_mangle]
pub unsafe extern "C" fn read_surface_field(path: *const c_char, name: *const c_char,
    typ: c_int, axis: *const c_char, levels: usize, first: usize, n: usize,
    record: usize, output: *mut std::ffi::c_void, words: usize) -> c_int {
    let _guard = NETCDF.lock().unwrap_or_else(|p| p.into_inner());
    if path.is_null() || name.is_null() || output.is_null() || n==0 || levels==0 ||
       ![4,5].contains(&typ) || (axis.is_null() && levels!=1) {return -10003;}
    if n.checked_mul(levels).map_or(true,|need|need>words) {return -10005;}
    let mut id=0;let rc=nc_open(path,0,&mut id);if rc!=0{return rc;}
    let check = |rc| if rc==0 {Ok(())} else {Err(rc)};
    let result=(|| {
        let (mut var,mut actual,mut rank,mut cells,mut len)=(0,0,0,0,0);
        check(nc_inq_varid(id,name,&mut var))?;
        check(nc_inq_vartype(id,var,&mut actual))?;
        check(nc_inq_varndims(id,var,&mut rank))?;
        if actual!=typ {return Err(-10001);}
        if rank==0 {
            if !axis.is_null() || first!=0 || n!=1 {return Err(-10004);}
            return check(if typ==5 {
                nc_get_vara_float(id,var,std::ptr::null(),std::ptr::null(),output.cast())
            } else {
                nc_get_vara_int(id,var,std::ptr::null(),std::ptr::null(),output.cast())
            });
        }
        let base=if axis.is_null(){1}else{2};
        if actual!=typ || (rank!=base && rank!=base+1){return Err(-10001);}
        let mut dims=[0;3];check(nc_inq_vardimid(id,var,dims.as_mut_ptr()))?;
        let timed=rank==base+1;let offset=usize::from(timed);
        check(nc_inq_dimid(id,b"nCells\0".as_ptr().cast(),&mut cells))?;
        if dims[offset]!=cells {return Err(-10002);}
        check(nc_inq_dimlen(id,cells,&mut len))?;
        if first.checked_add(n).map_or(true,|end|end>len){return Err(-10004);}
        let mut start=[0;3];let mut count=[1;3];
        start[offset]=first;count[offset]=n;
        if !axis.is_null(){
            let mut dim=0;check(nc_inq_dimid(id,axis,&mut dim))?;
            check(nc_inq_dimlen(id,dim,&mut len))?;
            if dims[offset+1]!=dim || len!=levels{return Err(-10002);}
            count[offset+1]=levels;
        }
        if timed {
            let mut time=0;check(nc_inq_dimid(id,b"Time\0".as_ptr().cast(),&mut time))?;
            if dims[0]!=time{return Err(-10002);}
            check(nc_inq_dimlen(id,time,&mut len))?;
            if record>=len{return Err(-10004);}
            start[0]=record;
        }
        check(if typ==5 {nc_get_vara_float(id,var,start.as_ptr(),count.as_ptr(),output.cast())}
              else {nc_get_vara_int(id,var,start.as_ptr(),count.as_ptr(),output.cast())})
    })();
    let close=nc_close(id);
    match result {Ok(())=>close,Err(rc)=>rc}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    #[link(name="netcdf")]
    extern "C" {
        fn nc_create(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
        fn nc_def_dim(id:c_int,name:*const c_char,len:usize,dim:*mut c_int)->c_int;
        fn nc_def_var(id:c_int,name:*const c_char,typ:c_int,rank:c_int,dims:*const c_int,var:*mut c_int)->c_int;
        fn nc_enddef(id:c_int)->c_int;
        fn nc_put_var_float(id:c_int,var:c_int,data:*const f32)->c_int;
        fn nc_put_var_int(id:c_int,var:c_int,data:*const c_int)->c_int;
        fn nc_put_var_text(id:c_int,var:c_int,data:*const c_char)->c_int;
    }
    #[test]
    fn native_axes_records_words_and_invalid_inputs() {unsafe {
        let dir=std::env::var_os("P25_READER_TEST_DIR").expect("test scratch must be explicit");
        let path=std::path::PathBuf::from(dir).join(format!("reader-{}.nc",std::process::id()));
        assert!(!path.exists());
        let cpath=CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        let mut id=0;assert_eq!(nc_create(cpath.as_ptr(),4,&mut id),0);
        let mut dims=[0;3];
        for (i,(name,len)) in [(b"Time\0".as_slice(),2),(b"nCells\0".as_slice(),4),
                             (b"nSoilLevels\0".as_slice(),2)].iter().enumerate(){
            assert_eq!(nc_def_dim(id,name.as_ptr().cast(),*len,&mut dims[i]),0);
        }
        let (mut width,mut stamp)=(0,0);
        assert_eq!(nc_def_dim(id,b"StrLen\0".as_ptr().cast(),64,&mut width),0);
        assert_eq!(nc_def_var(id,b"xtime\0".as_ptr().cast(),2,2,
            [dims[0],width].as_ptr(),&mut stamp),0);
        let (mut soil,mut land,mut tsk,mut ice)=(0,0,0,0);
        assert_eq!(nc_def_var(id,b"tslb\0".as_ptr().cast(),5,3,dims.as_ptr(),&mut soil),0);
        assert_eq!(nc_def_var(id,b"landmask\0".as_ptr().cast(),4,1,dims[1..].as_ptr(),&mut land),0);
        assert_eq!(nc_def_var(id,b"skintemp\0".as_ptr().cast(),5,2,dims.as_ptr(),&mut tsk),0);
        assert_eq!(nc_def_var(id,b"isice_lu\0".as_ptr().cast(),4,0,std::ptr::null(),&mut ice),0);
        assert_eq!(nc_enddef(id),0);
        let mut values:Vec<f32>=(0..16).map(|v|v as f32).collect();
        values[10]=-0.;values[11]=f32::from_bits(0x7fc12345);
        assert_eq!(nc_put_var_float(id,soil,values.as_ptr()),0);
        assert_eq!(nc_put_var_int(id,land,[0,1,0,1].as_ptr()),0);
        assert_eq!(nc_put_var_float(id,tsk,values.as_ptr()),0);
        assert_eq!(nc_put_var_int(id,ice,[24].as_ptr()),0);
        let mut times=[b' ';128];
        times[..19].copy_from_slice(b"2001-01-01_00:00:00");
        times[64..83].copy_from_slice(b"2001-01-01_06:00:00");
        assert_eq!(nc_put_var_text(id,stamp,times.as_ptr().cast()),0);
        assert_eq!(nc_close(id),0);
        let mut output=[99f32;4];
        let name=b"tslb\0".as_ptr().cast();let axis=b"nSoilLevels\0".as_ptr().cast();
        let read=|typ,axis,levels,first,n,record,words,out|read_surface_field(
            cpath.as_ptr(),name,typ,axis,levels,first,n,record,out,words);
        assert_eq!(read(5,axis,2,1,2,1,4,output.as_mut_ptr().cast()),0);
        assert_eq!(output.map(f32::to_bits),values[10..14].iter().map(|v|v.to_bits()).collect::<Vec<_>>().as_slice());
        let snapshot=output.map(f32::to_bits);
        assert_eq!(read(4,axis,2,1,2,1,4,output.as_mut_ptr().cast()),-10001);
        assert_eq!(read(5,axis,3,1,1,1,4,output.as_mut_ptr().cast()),-10002);
        assert_eq!(read(5,axis,2,1,2,2,4,output.as_mut_ptr().cast()),-10004);
        assert_eq!(read(5,axis,2,3,2,1,4,output.as_mut_ptr().cast()),-10004);
        assert_eq!(read(5,axis,2,1,2,1,3,output.as_mut_ptr().cast()),-10005);
        assert_eq!(read(5,axis,2,usize::MAX,2,1,4,output.as_mut_ptr().cast()),-10004);
        assert_eq!(read(5,axis,2,0,usize::MAX,1,4,output.as_mut_ptr().cast()),-10005);
        assert_eq!(read(5,axis,2,1,2,1,4,std::ptr::null_mut()),-10003);
        assert_eq!(output.map(f32::to_bits),snapshot);
        let mut mask=[-1;2];
        assert_eq!(read_surface_field(cpath.as_ptr(),b"landmask\0".as_ptr().cast(),4,
            std::ptr::null(),1,1,2,99,mask.as_mut_ptr().cast(),2),0);
        assert_eq!(mask,[1,0]);
        assert_eq!(read_surface_field(cpath.as_ptr(),b"isice_lu\0".as_ptr().cast(),4,
            std::ptr::null(),1,0,1,99,mask.as_mut_ptr().cast(),2),0);
        assert_eq!(mask[0],24);
        assert_eq!(read_surface_field(cpath.as_ptr(),b"isice_lu\0".as_ptr().cast(),4,
            std::ptr::null(),1,0,2,99,mask.as_mut_ptr().cast(),2),-10004);
        assert_eq!(read_surface_field(cpath.as_ptr(),b"skintemp\0".as_ptr().cast(),5,
            std::ptr::null(),1,1,2,1,output.as_mut_ptr().cast(),4),0);
        assert_eq!(output[..2],[5.,6.]);
        assert_eq!(read_surface_field(cpath.as_ptr(),b"absent\0".as_ptr().cast(),5,
            std::ptr::null(),1,1,2,1,output.as_mut_ptr().cast(),4),-49);
        let mut time=[0u8;64];
        assert_eq!(read_surface_time(cpath.as_ptr(),1,time.as_mut_ptr().cast(),64),64);
        assert_eq!(time.as_slice(),&times[64..]);
        assert_eq!(read_surface_time(cpath.as_ptr(),2,time.as_mut_ptr().cast(),64),-10004);
        assert_eq!(read_surface_time(cpath.as_ptr(),1,time.as_mut_ptr().cast(),63),-10005);
        assert_eq!(read_surface_time(cpath.as_ptr(),1,std::ptr::null_mut(),64),-10003);
        std::fs::remove_file(path).unwrap();
    }}
}
