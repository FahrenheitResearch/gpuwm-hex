//! Real native initialization fields against complete production Rust bridges.
//! MPAS-Model-rrfs/src/core_init_atmosphere/mpas_atm_advection.F:743-939 and
//! MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1266-1309 at 1d6e370e.
//! Rust owns NetCDF decoding, layout conversion and every word comparison.
use std::{ffi::{CString,c_char,c_int},io::{self,Write},path::Path,
          process::{Command,Stdio}};
#[path="oracle_record.rs"] mod record;
#[link(name="netcdf")]
unsafe extern "C" {
    fn nc_open(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
    fn nc_inq_dimid(id:c_int,name:*const c_char,dim:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
    fn nc_inq_varndims(id:c_int,var:c_int,rank:*mut c_int)->c_int;
    fn nc_inq_vardimid(id:c_int,var:c_int,dims:*mut c_int)->c_int;
    fn nc_inq_vartype(id:c_int,var:c_int,kind:*mut c_int)->c_int;
    fn nc_get_var_float(id:c_int,var:c_int,data:*mut f32)->c_int;
    fn nc_get_var_double(id:c_int,var:c_int,data:*mut f64)->c_int;
    fn nc_get_var_int(id:c_int,var:c_int,data:*mut i32)->c_int;
    fn nc_get_att_float(id:c_int,var:c_int,name:*const c_char,data:*mut f32)->c_int;
    fn nc_inq_attlen(id:c_int,var:c_int,name:*const c_char,len:*mut usize)->c_int;
    fn nc_get_att_text(id:c_int,var:c_int,name:*const c_char,data:*mut c_char)->c_int;
}
fn bad(message:impl ToString)->io::Error {io::Error::other(message.to_string())}
fn cstr(value:&str)->io::Result<CString>{CString::new(value).map_err(bad)}
fn check(code:i32)->io::Result<()> {
    if code==0 {Ok(())}else{Err(bad(format!("NetCDF status {code}")))}
}
struct Dataset(i32);
impl Drop for Dataset {fn drop(&mut self){unsafe{nc_close(self.0);}}}
impl Dataset {
    fn open(path:&str)->io::Result<Self>{let mut id=0;unsafe{check(nc_open(cstr(path)?.as_ptr(),0,&mut id))?;}Ok(Self(id))}
    fn dim(&self,name:&str)->io::Result<usize>{let mut id=0;let mut len=0;
        unsafe{check(nc_inq_dimid(self.0,cstr(name)?.as_ptr(),&mut id))?;
               check(nc_inq_dimlen(self.0,id,&mut len))?;}Ok(len)}
    fn variable(&self,name:&str,shape:&[usize])->io::Result<i32>{
        let mut var=0;let mut rank=0;
        unsafe{check(nc_inq_varid(self.0,cstr(name)?.as_ptr(),&mut var))?;
               check(nc_inq_varndims(self.0,var,&mut rank))?;}
        if rank<0 || rank as usize!=shape.len(){return Err(bad(format!("{name}: unexpected rank")));}
        let mut dims=vec![0;shape.len()];
        unsafe{check(nc_inq_vardimid(self.0,var,dims.as_mut_ptr()))?;}
        for (dim,&expected) in dims.into_iter().zip(shape){let mut len=0;
            unsafe{check(nc_inq_dimlen(self.0,dim,&mut len))?;}
            if len!=expected{return Err(bad(format!("{name}: unexpected extent")));}}
        Ok(var)
    }
    fn floats(&self,name:&str,shape:&[usize])->io::Result<Vec<f32>>{
        let var=self.variable(name,shape)?;let mut data=vec![0.;shape.iter().product()];
        unsafe{check(nc_get_var_float(self.0,var,data.as_mut_ptr()))?;}Ok(data)
    }
    fn indexes(&self,name:&str,shape:&[usize])->io::Result<Vec<i32>>{
        let var=self.variable(name,shape)?;let mut data=vec![0;shape.iter().product()];
        unsafe{check(nc_get_var_int(self.0,var,data.as_mut_ptr()))?;}Ok(data)
    }
    fn coordinate(&self,name:&str,n:usize,payload:&mut Vec<u8>)->io::Result<()> {
        let var=self.variable(name,&[n])?;let mut kind=0;
        unsafe{check(nc_inq_vartype(self.0,var,&mut kind))?;}
        match kind {
            5=>{payload.extend(4i32.to_le_bytes());for value in self.floats(name,&[n])?{payload.extend(value.to_le_bytes());}},
            6=>{payload.extend(8i32.to_le_bytes());let mut data=vec![0f64;n];
                unsafe{check(nc_get_var_double(self.0,var,data.as_mut_ptr()))?;}
                for value in data{payload.extend(value.to_le_bytes());}},
            _=>return Err(bad("real coordinates required")),
        }Ok(())
    }
    fn radius(&self)->io::Result<f32>{let mut value=0.;
        unsafe{check(nc_get_att_float(self.0,-1,cstr("sphere_radius")?.as_ptr(),&mut value))?;}Ok(value)}
    fn sphere(&self)->io::Result<bool>{let name=cstr("on_a_sphere")?;let mut len=0;
        unsafe{check(nc_inq_attlen(self.0,-1,name.as_ptr(),&mut len))?;}
        let mut value=vec![0u8;len];unsafe{check(nc_get_att_text(self.0,-1,name.as_ptr(),value.as_mut_ptr().cast()))?;}
        match std::str::from_utf8(&value).map_err(bad)?.trim(){"YES"=>Ok(true),"NO"=>Ok(false),_=>Err(bad("invalid on_a_sphere"))}
    }
}
fn run_bridge(path:&str,payload:Vec<u8>)->io::Result<Vec<u8>>{
    let mut child=Command::new(path).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    // Drain stdout concurrently with the input writer: large real meshes must
    // not deadlock on pipe capacity after the child consumes its full input.
    let mut input=child.stdin.take().ok_or_else(||bad("bridge stdin absent"))?;
    let writer=std::thread::spawn(move ||input.write_all(&payload));
    let result=child.wait_with_output()?;
    writer.join().map_err(|_|bad("bridge input writer panicked"))??;
    if !result.status.success(){return Err(bad(String::from_utf8_lossy(&result.stderr)));}
    Ok(result.stdout)
}
fn words(data:&[u8])->io::Result<Vec<u32>> {
    if data.len()%4!=0{return Err(bad("incomplete binary32 payload"));}
    Ok(data.chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect())
}
fn comparison(name:&str,actual:&[u32],expected:&[u32])->io::Result<(String,usize)>{
    if actual.len()!=expected.len(){return Err(bad(format!("{name}: output extent mismatch")));}
    let different=actual.iter().zip(expected).filter(|(a,b)|a!=b).count();
    let first=actual.iter().zip(expected).position(|(a,b)|a!=b);
    Ok((format!("{{\"field\":\"{name}\",\"words\":{},\"different_words\":{different},\"first_difference\":{}}}",actual.len(),first.map_or("null".into(),|v|v.to_string())),different))
}
fn native_record(path:&str)->io::Result<([i32;21],Vec<f32>)>{
    let mut header=[0;21];let mut len=0;let path=cstr(path)?;
    unsafe{let pointer=record::oracle_read(path.as_ptr(),header.as_mut_ptr(),&mut len);
        if pointer.is_null(){return Err(bad("invalid native record"));}
        let raw=std::slice::from_raw_parts(pointer,len).to_vec();record::oracle_free(pointer,len);
        if header[0]!=1 || header[1]!=4 || header[2]!=2{return Err(bad("native binary32 matrix required"));}
        Ok((header,words(&raw)?.into_iter().map(f32::from_bits).collect()))}
}
fn deformation(init:&str,bridge:&str)->io::Result<(Vec<String>,usize)> {
    let dataset=Dataset::open(init)?;let nc=dataset.dim("nCells")?;let ne=dataset.dim("nEdges")?;
    let nv=dataset.dim("nVertices")?;let me=dataset.dim("maxEdges")?;
    let mut payload=b"RRFSDEF2".to_vec();for n in [nc,ne,nv,me]{payload.extend((n as u64).to_le_bytes());}
    payload.extend(i32::from(dataset.sphere()?).to_le_bytes());payload.extend(dataset.radius()?.to_le_bytes());
    for (name,shape) in [("nEdgesOnCell",vec![nc]),("edgesOnCell",vec![nc,me]),
                        ("cellsOnCell",vec![nc,me]),("verticesOnCell",vec![nc,me]),("cellsOnEdge",vec![ne,2])]{
        for value in dataset.indexes(name,&shape)?{
            payload.extend((value-if name=="nEdgesOnCell"{0}else{1}).to_le_bytes());}}
    for (suffix,n) in [("Cell",nc),("Vertex",nv)]{
        for axis in ["x","y","z"]{dataset.coordinate(&format!("{axis}{suffix}"),n,&mut payload)?;}}
    let actual=words(&run_bridge(bridge,payload)?)?;
    if actual.len()!=4*nc*me{return Err(bad("deformation bridge extent mismatch"));}
    let mut rows=Vec::new();let mut differences=0;
    for (i,name) in ["defc_a","defc_b","cell_gradient_coef_x","cell_gradient_coef_y"].into_iter().enumerate(){
        let expected:Vec<_>=dataset.floats(name,&[nc,me])?.into_iter().map(f32::to_bits).collect();
        let (row,diff)=comparison(name,&actual[i*nc*me..(i+1)*nc*me],&expected)?;rows.push(row);differences+=diff;}
    Ok((rows,differences))
}
fn damping(init:&str,zgrid:&str,dss:&str,xnutr:f32,zd:f32,bridge:&str)->io::Result<(Vec<String>,usize)> {
    let dataset=Dataset::open(init)?;let nc=dataset.dim("nCells")?;
    let (zh,z)=native_record(zgrid)?;let (dh,d)=native_record(dss)?;
    let nl=dh[3] as usize;let storage=dh[4] as usize;
    if storage<nc || zh[3] as usize!=nl+1 || zh[4] as usize!=storage || dh[8..10]!=[1,1] || zh[8..10]!=[1,1]{return Err(bad("native damping shape/bounds mismatch"));}
    let mut payload=b"RRFSDSS1".to_vec();for n in [nl,nc]{payload.extend((n as u64).to_le_bytes());}
    payload.extend(xnutr.to_le_bytes());payload.extend(zd.to_le_bytes());
    for k in 0..=nl{for c in 0..nc{payload.extend(z[c*(nl+1)+k].to_le_bytes());}}
    for value in dataset.floats("meshDensity",&[nc])?{payload.extend(value.to_le_bytes());}
    let actual=words(&run_bridge(bridge,payload)?)?;
    let expected:Vec<_>=(0..nl).flat_map(|k|(0..nc).map(move|c|(k,c)))
        .map(|(k,c)|d[c*nl+k].to_bits()).collect();
    let (row,diff)=comparison("dss_physical_cells",&actual,&expected)?;
    // Native core.F:1297 clears the complete associated pool, including dummy
    // columns, before the physical nCells loop. The regional constructor pads
    // the returned physical bridge array with the same positive-zero words.
    let padding=vec![0u32;(storage-nc)*nl];
    let native_padding:Vec<_>=d[nc*nl..].iter().map(|value|value.to_bits()).collect();
    let (padding_row,padding_diff)=comparison("dss_dummy_pool",&padding,&native_padding)?;
    Ok((vec![row,padding_row],diff+padding_diff))
}
fn main()->io::Result<()> {
    let args:Vec<_>=std::env::args().collect();
    let (rows,diff,out)=match args.get(1).map(String::as_str){
        Some("deformation") if args.len()==5=>{let (r,d)=deformation(&args[2],&args[3])?;(r,d,&args[4])},
        Some("damping") if args.len()==9=>{let (r,d)=damping(&args[2],&args[3],&args[4],args[5].parse().map_err(bad)?,args[6].parse().map_err(bad)?,&args[7])?;(r,d,&args[8])},
        _=>return Err(bad("deformation INIT BRIDGE OUT or damping INIT ZGRID_RECORD DSS_RECORD XNUTR ZD BRIDGE OUT")),
    };
    std::fs::write(Path::new(out),format!("{{\"scope\":\"complete Rust preparation on original real native initialization fields; damping physical and dummy pool words\",\"native_ranks\":1,\"different_words\":{diff},\"passed\":{},\"comparisons\":[{}]}}\n",diff==0,rows.join(",")))?;
    if diff!=0{return Err(bad(format!("{diff} differing native initialization words")));}Ok(())
}
