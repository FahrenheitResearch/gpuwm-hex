//! Native-defect diagnostic: retain exact varying floating-word index sets.
//! NetCDF field decoding stays in Rust. This does not grade a hex forecast.
//! Output authority: mpas_atmphys_driver_convection.F:1061-1102 (RAINC).
use std::{ffi::{c_char,c_int,c_void,CStr,CString},path::{Path,PathBuf},ptr};
#[link(name="netcdf")]
unsafe extern "C" {
    fn nc_open(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_inq_nvars(id:c_int,count:*mut c_int)->c_int;
    fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
    fn nc_inq_var(id:c_int,var:c_int,name:*mut c_char,ty:*mut c_int,nd:*mut c_int,dims:*mut c_int,attrs:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
    fn nc_get_var(id:c_int,var:c_int,out:*mut c_void)->c_int;
    fn nc_strerror(error:c_int)->*const c_char;
}
fn check(rc:c_int)->Result<(),String>{
    if rc==0 {Ok(())} else {Err(unsafe{CStr::from_ptr(nc_strerror(rc))}.to_string_lossy().into_owned())}
}
fn quote(s:&str)->String{
    let mut out=String::from("\"");
    for c in s.chars(){match c{
        '"'=>out.push_str("\\\""),'\\'=>out.push_str("\\\\"),'\n'=>out.push_str("\\n"),
        '\r'=>out.push_str("\\r"),'\t'=>out.push_str("\\t"),
        c if c<' '=>out.push_str(&format!("\\u{:04x}",c as u32)),c=>out.push(c)}}
    out.push('"');out
}
struct File(c_int);
impl File{
    fn open(path:&Path)->Result<Self,String>{
        let mut id=0;
        let name=CString::new(path.to_str().ok_or("non UTF-8 path")?).map_err(|e|e.to_string())?;
        check(unsafe{nc_open(name.as_ptr(),0,&mut id)})?;Ok(Self(id))
    }
    fn variables(&self)->Result<Vec<(String,c_int,Vec<usize>)>,String>{
        let mut n=0;check(unsafe{nc_inq_nvars(self.0,&mut n)})?;
        let mut out=Vec::new();
        for v in 0..n{
            let mut name=[0 as c_char;257];let mut ty=0;let mut nd=0;let mut dims=[0;1024];
            check(unsafe{nc_inq_var(self.0,v,name.as_mut_ptr(),&mut ty,&mut nd,dims.as_mut_ptr(),ptr::null_mut())})?;
            if ty!=5 && ty!=6{continue;}
            let mut shape=Vec::new();
            for &dim in &dims[..nd as usize]{let mut len=0;check(unsafe{nc_inq_dimlen(self.0,dim,&mut len)})?;shape.push(len);}
            out.push((unsafe{CStr::from_ptr(name.as_ptr())}.to_string_lossy().into_owned(),ty,shape));
        }
        Ok(out)
    }
    fn words(&self,name:&str,ty:c_int,shape:&[usize])->Result<Vec<u64>,String>{
        let mut var=0;let name=CString::new(name).map_err(|e|e.to_string())?;
        check(unsafe{nc_inq_varid(self.0,name.as_ptr(),&mut var)})?;
        let count=shape.iter().try_fold(1usize,|n,&m|n.checked_mul(m).ok_or("shape overflow"))?;
        match ty{
            5=>{let mut data=vec![0u32;count];check(unsafe{nc_get_var(self.0,var,data.as_mut_ptr().cast())})?;Ok(data.into_iter().map(u64::from).collect())},
            6=>{let mut data=vec![0u64;count];check(unsafe{nc_get_var(self.0,var,data.as_mut_ptr().cast())})?;Ok(data)},
            _=>Err("unsupported word type".into())
        }
    }
}
impl Drop for File{fn drop(&mut self){unsafe{nc_close(self.0);}}}
fn ranges(indices:impl Iterator<Item=usize>)->Vec<(usize,usize)>{
    let mut out=Vec::new();let mut pending=None;
    for index in indices{
        match pending{
            Some((lo,hi)) if index==hi=>pending=Some((lo,index+1)),
            Some(pair)=>{out.push(pair);pending=Some((index,index+1));},
            None=>pending=Some((index,index+1)),
        }
    }
    if let Some(pair)=pending{out.push(pair);}out
}
fn range_json(r:&[(usize,usize)])->String{format!("[{}]",r.iter().map(|(a,b)|format!("[{a},{b}]")).collect::<Vec<_>>().join(","))}
fn finite(word:u64,ty:c_int)->bool{if ty==5{f32::from_bits(word as u32).is_finite()}else{f64::from_bits(word).is_finite()}}
fn field(name:&str,ty:c_int,shape:&[usize],a:&[u64],b:&[u64])->String{
    let diffs=ranges(a.iter().zip(b).enumerate().filter_map(|(i,(x,y))|if x!=y{Some(i)}else{None}));
    let count:usize=diffs.iter().map(|(a,b)|b-a).sum();
    let nonfinite=|data:&[u64]|ranges(data.iter().enumerate().filter_map(|(i,&w)|if finite(w,ty){None}else{Some(i)}));
    let first=a.iter().zip(b).enumerate().find(|(_, (x,y))|x!=y).map(|(i,(x,y))|format!("{{\"index\":{i},\"reference_bits\":\"{x:016x}\",\"other_bits\":\"{y:016x}\"}}")).unwrap_or_else(||"null".into());
    format!("{{\"field\":{},\"netcdf_type\":{ty},\"shape\":{shape:?},\"words\":{},\"differing_words\":{count},\"differing_index_ranges\":{},\"reference_nonfinite_ranges\":{},\"other_nonfinite_ranges\":{},\"first_difference\":{first}}}",quote(name),a.len(),range_json(&diffs),range_json(&nonfinite(a)),range_json(&nonfinite(b)))
}
fn names(dir:&Path)->Result<Vec<String>,String>{
    let mut out=Vec::new();
    for path in std::fs::read_dir(dir).map_err(|e|e.to_string())?{
        let path=path.map_err(|e|e.to_string())?.path();
        if path.extension().is_some_and(|e|e=="nc") && !path.is_symlink() && path.is_file(){out.push(path.file_name().unwrap().to_str().ok_or("non UTF-8 filename")?.into());}
    }
    out.sort();Ok(out)
}
fn run()->Result<(),String>{
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()==2 && args[1]=="--self-test"{
        assert_eq!(ranges([0,1,4,6,7,8].into_iter()),vec![(0,2),(4,5),(6,9)]);
        assert_eq!(ranges(std::iter::empty()),Vec::<(usize,usize)>::new());
        let a=[0u64,0x80000001,0x7fc01234,0x7f800000,0x80000000];
        let b=[0u64,0,0x7fc01235,0x7f800000,0];
        let s=field("test",5,&[5],&a,&b);
        assert!(s.contains("\"differing_words\":3"));assert!(s.contains("\"differing_index_ranges\":[[1,3],[4,5]]"));
        assert!(s.contains("\"reference_nonfinite_ranges\":[[2,4]]"));
        assert!(!finite(f64::INFINITY.to_bits(),6));assert!(finite(1,6));
        println!("{{\"self_test_passed\":true,\"negative_subnormal_preserved\":true,\"nan_payload_difference_retained\":true,\"negative_zero_difference_retained\":true}}");return Ok(());
    }
    if args.len()!=3{return Err("usage: native-word-sets REFERENCE_DIR OTHER_DIR | --self-test".into());}
    let a=PathBuf::from(&args[1]);let b=PathBuf::from(&args[2]);let an=names(&a)?;let bn=names(&b)?;
    if an.is_empty(){return Err("reference has no native output files".into());}
    let missing=an.iter().filter(|n|!bn.contains(n)).map(|n|quote(n)).collect::<Vec<_>>();
    let extra=bn.iter().filter(|n|!an.contains(n)).map(|n|quote(n)).collect::<Vec<_>>();let mut files=Vec::new();
    for name in an.iter().filter(|n|bn.contains(n)){
        let fa=File::open(&a.join(name))?;let fb=File::open(&b.join(name))?;let av=fa.variables()?;let bv=fb.variables()?;
        if av!=bv{return Err(format!("floating variable/schema mismatch: {name}"));}
        let mut fields=Vec::new();
        for (var,ty,shape) in av{
            let aw=fa.words(&var,ty,&shape)?;let bw=fb.words(&var,ty,&shape)?;
            if aw!=bw || aw.iter().any(|&w|!finite(w,ty)) || bw.iter().any(|&w|!finite(w,ty)){
                fields.push(field(&var,ty,&shape,&aw,&bw));
            }
        }
        files.push(format!("{{\"file\":{},\"varying_or_nonfinite_fields\":[{}]}}",quote(name),fields.join(",")));
    }
    println!("{{\"schema\":\"mpas-native-varying-word-sets/v1\",\"diagnostic_only\":true,\"forecast_proven\":false,\"range_semantics\":\"flat C storage indexes [begin,end); every differing floating word is included\",\"reference\":{},\"other\":{},\"missing_files\":[{}],\"extra_files\":[{}],\"files\":[{}]}}",quote(&args[1]),quote(&args[2]),missing.join(","),extra.join(","),files.join(","));Ok(())
}
fn main(){if let Err(error)=run(){eprintln!("{error}");std::process::exit(2);}}
