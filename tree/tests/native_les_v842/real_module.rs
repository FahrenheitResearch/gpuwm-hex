//! Real 55-level profiles on the synthetic three-cell LES test stencil.
//! Pristine whole-module native RK slice, mpas_atm_dissipation_models.F:208-1606.
//! Geometry/deformation weights are synthetic; this is not mesh/forecast proof.
#[allow(dead_code)] mod real_columns;
include!("real_template.rs");
use std::{collections::HashMap,env,ffi::{c_char,c_int,c_void,CString},fs,path::Path};
extern "C" {
    fn nc_open(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_get_var_int(id:c_int,var:c_int,data:*mut c_int)->c_int;
    fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
    fn dlopen(path:*const c_char,mode:c_int)->*mut c_void;
    fn dlsym(handle:*mut c_void,name:*const c_char)->*mut c_void;
    fn dlclose(handle:*mut c_void)->c_int;
    fn invoke_les_boundary(symbol:*mut c_void,args:*mut *mut c_void);
}
fn extent(text:&str,dims:&HashMap<&str,usize>)->usize {
    if text==":" {return dims["ncells"]+1;}
    if let Ok(n)=text.parse::<usize>() {return n;}
    if let Some((name,add))=text.split_once('+') {return dims[name]+add.parse::<usize>().unwrap();}
    if let Some((name,sub))=text.split_once('-') {return dims[name]-sub.parse::<usize>().unwrap();}
    dims[text]
}
fn floats(data:&[f32])->Vec<u32> {data.iter().map(|x|x.to_bits()).collect()}
fn open(path:&Path)->c_int {
    let mut id=0;unsafe {assert_eq!(nc_open(CString::new(path.to_str().unwrap()).unwrap().as_ptr(),0,&mut id),0);}id
}
fn subset(id:c_int,name:&str,indices:&[usize],nv:usize)->Vec<u32> {
    let(data,shape)=real_columns::read(id,name);
    assert_eq!(*shape.last().unwrap(),nv);assert!(shape.len()==2 || shape.len()==3);
    if shape.len()==3 {assert_eq!(shape[0],1);}
    let mut out=vec![0;nv*(indices.len()+1)];
    for (column,source) in indices.iter().enumerate() {
        for k in 0..nv {out[column*nv+k]=data[source*nv+k].to_bits();}
    }out
}
fn ints(id:c_int,name:&str,size:usize)->Vec<i32> {
    let(mut var,mut out)=(0,vec![0;size]);
    unsafe {
        assert_eq!(nc_inq_varid(id,CString::new(name).unwrap().as_ptr(),&mut var),0);
        assert_eq!(nc_get_var_int(id,var,out.as_mut_ptr()),0);
    }out
}
fn write(path:&Path,words:&[u32]) {fs::write(path,words.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()).unwrap();}
fn main() {
    let args:Vec<String>=env::args().collect();
    assert_eq!(args.len(),5,"real-module RESTART.nc OUTPUT.nc DIRECTORY LIBRARY");
    let id=open(Path::new(&args[1]));let diagnostic=open(Path::new(&args[2]));
    let(_,shape)=real_columns::read(id,"theta_m");let(nc,nv)=(shape[1],shape[2]);
    let(u,ushape)=real_columns::read(id,"u");let ne=ushape[1];
    let cells=[0,nc/3,2*nc/3];let edges=[0,ne/3,2*ne/3];
    let mut real=HashMap::new();
    for (target,source,levels) in [("theta_m","theta_m",nv),("exner","exner",nv),
        ("pressure_b","pressure_base",nv),("pp","pressure_p",nv),("zgrid","zgrid",nv+1),
        ("rho_zz","rho_zz",nv),("w","w",nv+1),("zz","zz",nv),("t_init","t_init",nv)] {
        real.insert(target,subset(id,source,&cells,levels));
    }
    real.insert("u",subset(id,"u",&edges,nv));
    for (target,source) in [("ucell","uReconstructZonal"),("vcell","uReconstructMeridional"),
        ("ureconstructzonal","uReconstructZonal"),("ureconstructmeridional","uReconstructMeridional")] {
        real.insert(target,subset(diagnostic,source,&cells,nv));
    }
    // Native tangential reconstruction: mpas_atm_time_integration.F:7317-7330.
    // Uses the actual mesh's stored weights, connectivity and edge winds.
    let(weights,wshape)=real_columns::read(id,"weightsOnEdge");let width=wshape[1];
    let count=ints(id,"nEdgesOnEdge",ne);let neighbors=ints(id,"edgesOnEdge",ne*width);
    let mut v=vec![0.0f32;nv*4];
    for (column,&edge) in edges.iter().enumerate() {
        for j in 0..count[edge] as usize {
            let source=usize::try_from(neighbors[edge*width+j]-1).unwrap();
            for k in 0..nv {v[column*nv+k]+=weights[edge*width+j]*u[source*nv+k];}
        }
    }
    real.insert("v",floats(&v));
    for name in ["rdzu","rdzw","fzm","fzp","u_init","v_init"] {
        let(data,shape)=real_columns::read(id,name);assert_eq!(shape,[nv]);real.insert(name,floats(&data));
    }
    for name in ["hfx","qfx","ustm"] {
        let(data,shape)=real_columns::read(id,name);assert_eq!(shape,[1,nc]);
        let mut out=vec![0;4];for (j,&cell) in cells.iter().enumerate(){out[j]=data[cell].to_bits();}
        real.insert(name,out);
    }
    let mut scalars=vec![0u32;7*nv*4];let mut qtot=vec![0.0f32;nv*4];
    for (iq,name) in ["qv","qc","qr","qi","qs","qg"].iter().enumerate() {
        let values=subset(id,name,&cells,nv);
        for j in 0..nv*3 {scalars[7*j+iq]=values[j];qtot[j]+=f32::from_bits(values[j]);}
    }
    // LES is inactive in these published suites: TKE is an explicit test seed.
    for j in 0..nv*3 {scalars[7*j+6]=0.1f32.to_bits();}
    real.insert("scalars",scalars);real.insert("qtot",floats(&qtot));
    unsafe {assert_eq!(nc_close(id),0);assert_eq!(nc_close(diagnostic),0);}
    let dims:HashMap<&str,usize>=[("ncells",3),("nedges",3),("nvertices",3),("maxedges",2),
        ("vertexdegree",2),("nvertlevels",nv),("num_scalars",7),("num_scalars_dummy",7)].into_iter().collect();
    let mut prototype=Vec::new();let mut shapes=Vec::new();
    for &(name,integer,expressions,oldshape,seed) in PROTOTYPES {
        let shape:Vec<usize>=expressions.iter().map(|x|extent(x,&dims)).collect();
        let count=shape.iter().product::<usize>().max(1);let mut values=Vec::new();
        for index in 0..count {
            let(mut left,mut source,mut stride)=(index,0,1);
            for (&new,&old) in shape.iter().zip(oldshape) {
                let coordinate=left%new;left/=new;source+=(coordinate%old)*stride;stride*=old;
            }
            values.push(seed[source]);
        }
        if let Some(data)=real.get(name) {assert_eq!(count,data.len(),"{name}");values=data.clone();}
        if expressions.is_empty() && integer {
            if let Some(&value)=dims.get(name) {values[0]=value as u32;}
            if name=="index_tke" {values[0]=7;}
        }
        prototype.push(values);shapes.push(shape);
    }
    let handle=unsafe {dlopen(CString::new(args[4].as_str()).unwrap().as_ptr(),2)};assert!(!handle.is_null());
    let symbol=unsafe {dlsym(handle,CString::new("replay_les_boundary").unwrap().as_ptr())};assert!(!symbol.is_null());
    let out=Path::new(&args[3]);fs::create_dir_all(out).unwrap();let mut rows=Vec::new();
    for model in [1u32,2] {for surface in [0u32,1,2] {for substep in [1u32,2] {for mix in [0u32,1] {
        let row=rows.len();let directory=out.join(row.to_string());fs::create_dir_all(&directory).unwrap();
        let mut values=prototype.clone();let mut metadata=Vec::new();
        for (i,&(name,integer,_,_,_)) in PROTOTYPES.iter().enumerate() {
            for (key,value) in [("les_model_opt",model),("les_surface_opt",surface),("dynamics_substep",substep),("mix_scalars",mix)] {
                if name==key {values[i][0]=value;}
            }
            write(&directory.join(format!("input-{name}.bin")),&values[i]);
            metadata.push(format!("{{\"name\":\"{name}\",\"integer\":{integer},\"shape\":{:?}}}",shapes[i]));
        }
        let mut pointers:Vec<*mut c_void>=values.iter_mut().map(|x|x.as_mut_ptr().cast()).collect();
        unsafe {invoke_les_boundary(symbol,pointers.as_mut_ptr());}
        for (i,&(name,_,_,_,_)) in PROTOTYPES.iter().enumerate() {write(&directory.join(format!("output-{name}.bin")),&values[i]);}
        fs::write(directory.join("frame.json"),format!("[{}]\n",metadata.join(","))).unwrap();
        rows.push(format!("{{\"case\":{row},\"model\":{model},\"surface\":{surface},\"substep\":{substep},\"mix_scalars\":{mix}}}"));
    }}}}
    unsafe {assert_eq!(dlclose(handle),0);}
    fs::write(out.join("profiles.json"),format!("{{\"scope\":\"real restart/diagnostic profiles on synthetic stencil; no forecast or mesh identity\",\"cell_ids_zero_based\":{cells:?},\"edge_ids_zero_based\":{edges:?},\"levels\":{nv},\"tke\":\"explicit 0.1 test seed; source suite has no LES TKE\",\"real_fields\":{:?},\"cases\":[{}]}}\n",{let mut names:Vec<_>=real.keys().copied().collect();names.sort();names},rows.join(","))).unwrap();
}
