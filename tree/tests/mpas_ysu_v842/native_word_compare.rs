//! Bitwise comparison of every native numeric/character field, including full physics.
//! NetCDF decode stays in Rust/native netCDF; Python only schedules file pairs.
use std::{collections::BTreeMap, ffi::{CString, c_char, c_int, c_void}, ptr};

#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_inq_nvars(id: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_var(id: c_int, v: c_int, name: *mut c_char, t: *mut c_int,
                  nd: *mut c_int, dims: *mut c_int, attrs: *mut c_int) -> c_int;
    fn nc_inq_dim(id: c_int, d: c_int, name: *mut c_char, len: *mut usize) -> c_int;
    fn nc_get_vara(id: c_int, v: c_int, start: *const usize, count: *const usize, data: *mut c_void) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
}
// OpenSSL supplies SHA256; Rust still reads every native word without conversion.
#[link(name = "crypto")]
unsafe extern "C" {
    fn EVP_MD_CTX_new() -> *mut c_void;
    fn EVP_MD_CTX_free(ctx: *mut c_void);
    fn EVP_sha256() -> *const c_void;
    fn EVP_DigestInit_ex(ctx: *mut c_void, md: *const c_void, engine: *mut c_void) -> c_int;
    fn EVP_DigestUpdate(ctx: *mut c_void, data: *const c_void, n: usize) -> c_int;
    fn EVP_DigestFinal_ex(ctx: *mut c_void, bytes: *mut u8, n: *mut u32) -> c_int;
}
struct Digest(*mut c_void);
impl Drop for Digest {fn drop(&mut self) {unsafe {EVP_MD_CTX_free(self.0);}}}
fn field_digest(file: &File, var: &Var) -> Result<(usize,String),String> {
    if cfg!(target_endian="big") {return Err("word digests require the pinned little-endian hosts".into());}
    let total=var.shape.iter().try_fold(1_usize,|n,&d|n.checked_mul(d)).ok_or("shape overflow")?;
    let size=match var.ty {1|2|7=>1,3|8=>2,4|5|9=>4,6|10|11=>8,_=>return Err("unsupported NetCDF digest type".into())};
    let ctx=Digest(unsafe {EVP_MD_CTX_new()});
    if ctx.0.is_null() || unsafe {EVP_DigestInit_ex(ctx.0,EVP_sha256(),ptr::null_mut())}!=1 {return Err("SHA256 init failed".into());}
    if total>0 {
        // Slabs along the first dimension preserve complete netCDF C word order.
        let rows=var.shape.first().copied().unwrap_or(1);let width=total/rows;
        let batch=(262144/width.max(1)).max(1).min(rows);
        let mut start=vec![0;var.shape.len()];let mut count=var.shape.clone();let mut offset=0;
        while offset<rows {
            if !var.shape.is_empty() {start[0]=offset;count[0]=batch.min(rows-offset);}
            let n=count.iter().product::<usize>();let mut words=vec![0_u64;(n*size).div_ceil(8)];
            check(unsafe {nc_get_vara(file.0,var.id,start.as_ptr(),count.as_ptr(),words.as_mut_ptr().cast())})?;
            if unsafe {EVP_DigestUpdate(ctx.0,words.as_ptr().cast(),n*size)}!=1 {return Err("SHA256 update failed".into());}
            offset+=batch;
        }
    }
    let mut bytes=[0_u8;32];let mut n=0;
    if unsafe {EVP_DigestFinal_ex(ctx.0,bytes.as_mut_ptr(),&mut n)}!=1 || n!=32 {return Err("SHA256 final failed".into());}
    Ok((total,bytes.iter().map(|x|format!("{x:02x}")).collect()))
}
fn check(code: c_int) -> Result<(), String> {
    if code == 0 { Ok(()) } else {
        Err(unsafe {std::ffi::CStr::from_ptr(nc_strerror(code))}.to_string_lossy().into_owned())
    }
}
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() { match c {
        '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"), '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
        c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)), c => out.push(c),
    }}
    out.push('"'); out
}
struct File(c_int);
impl File {
    fn open(path: &str) -> Result<Self, String> {
        let p=CString::new(path).map_err(|e| e.to_string())?; let mut id=0;
        check(unsafe {nc_open(p.as_ptr(), 0, &mut id)})?; Ok(Self(id))
    }
    fn vars(&self) -> Result<BTreeMap<String, Var>, String> {
        let mut n=0; check(unsafe {nc_inq_nvars(self.0, &mut n)})?;
        let mut vars=BTreeMap::new();
        for id in 0..n {
            let mut name=[0_i8;257]; let mut ty=0; let mut nd=0;
            let mut dims=[0;1024];
            check(unsafe {nc_inq_var(self.0,id,name.as_mut_ptr(),&mut ty,&mut nd,dims.as_mut_ptr(),ptr::null_mut())})?;
            let name=unsafe {std::ffi::CStr::from_ptr(name.as_ptr())}.to_string_lossy().into_owned();
            let mut shape=Vec::new(); let mut dnames=Vec::new();
            for &dim in &dims[..nd as usize] {
                let mut dname=[0_i8;257]; let mut len=0;
                check(unsafe {nc_inq_dim(self.0,dim,dname.as_mut_ptr(),&mut len)})?;
                shape.push(len);
                dnames.push(unsafe {std::ffi::CStr::from_ptr(dname.as_ptr())}.to_string_lossy().into_owned());
            }
            vars.insert(name,Var{id,ty,shape,dnames});
        }
        Ok(vars)
    }
}
impl Drop for File { fn drop(&mut self) { unsafe {nc_close(self.0);} } }
struct Var {id:c_int,ty:c_int,shape:Vec<usize>,dnames:Vec<String>}

fn compare(a: &File, av: &Var, b: &File, bv: &Var) -> Result<(usize,usize,Option<(Vec<usize>,u64,u64)>),String> {
    let total=av.shape.iter().try_fold(1_usize,|n,&d| n.checked_mul(d)).ok_or("shape overflow")?;
    if total==0 {return Ok((0,0,None));}
    let axis=av.shape.iter().enumerate().max_by_key(|&(_,d)| d).map(|(i,_)|i);
    let mut start=vec![0;av.shape.len()]; let mut count=av.shape.clone();
    let rows=axis.map(|i| av.shape[i]).unwrap_or(1);
    let width=total/rows;
    let batch=(262144/width.max(1)).max(1).min(rows);
    let mut compared=0; let mut different=0; let mut first=None;
    let mut offset=0;
    while offset<rows {
        if let Some(i)=axis {start[i]=offset; count[i]=batch.min(rows-offset);}
        let n=count.iter().product::<usize>();
        let size=match av.ty {1|2|7=>1,3|8=>2,4|5|9=>4,6|10|11=>8,_=>return Err("unsupported NetCDF field type".into())};
        // u64 allocation provides alignment for every native numeric type.
        // nc_get_vara copies values without conversion, preserving all bit patterns.
        let units=(n*size).div_ceil(8);
        let mut aa=vec![0_u64;units]; let mut bb=vec![0_u64;units];
        check(unsafe {nc_get_vara(a.0,av.id,start.as_ptr(),count.as_ptr(),aa.as_mut_ptr().cast())})?;
        check(unsafe {nc_get_vara(b.0,bv.id,start.as_ptr(),count.as_ptr(),bb.as_mut_ptr().cast())})?;
        let aa=unsafe {std::slice::from_raw_parts(aa.as_ptr().cast::<u8>(),n*size)};
        let bb=unsafe {std::slice::from_raw_parts(bb.as_ptr().cast::<u8>(),n*size)};
        let word=|bytes: &[u8]|->u64 {match bytes.len() {
            1=>bytes[0] as u64,
            2=>u16::from_ne_bytes(bytes.try_into().unwrap()) as u64,
            4=>u32::from_ne_bytes(bytes.try_into().unwrap()) as u64,
            8=>u64::from_ne_bytes(bytes.try_into().unwrap()),
            _=>unreachable!(),
        }};
        let bits=aa.chunks_exact(size).zip(bb.chunks_exact(size)).map(|(x,y)|(word(x),word(y)));
        for (flat,(x,y)) in bits.into_iter().enumerate() {
            if x!=y {
                different+=1;
                if first.is_none() {
                    let mut idx=vec![0;count.len()]; let mut k=flat;
                    for i in (0..count.len()).rev() {idx[i]=k%count[i]+start[i]; k/=count[i];}
                    first=Some((idx,x,y));
                }
            }
        }
        compared+=n; offset+=batch;
    }
    Ok((compared,different,first))
}
fn run(args: &[String]) -> Result<bool,String> {
    if args.len()==3 && args[1]=="--inventory" {return inventory(&args[2],false);}
    if args.len()==3 && args[1]=="--field-digests" {return inventory(&args[2],true);}
    if args.len()==4 && args[1]=="--i32-range" {return integer_range(&args[2],&args[3]);}
    if args.len()!=3 {return Err("usage: mpas-native-word-compare REFERENCE.nc OTHER.nc".into());}
    let a=File::open(&args[1])?; let b=File::open(&args[2])?;
    let aa=a.vars()?; let bb=b.vars()?;
    let mut errors=Vec::new(); let mut records=Vec::new(); let mut words=0; let mut diffs=0; let mut floating_words=0; let mut floating_diffs=0;
    for name in aa.keys().chain(bb.keys()) {
        if !aa.contains_key(name)||!bb.contains_key(name) {
            let e=format!("variable set differs: {name}"); if !errors.contains(&e) {errors.push(e);}
        }
    }
    for (name,av) in &aa {
        let Some(bv)=bb.get(name) else {continue};
        if av.ty!=bv.ty||av.shape!=bv.shape||av.dnames!=bv.dnames {
            errors.push(format!("type or dimensions differ: {name}"));continue;
        }
        if !(1..=11).contains(&av.ty) {errors.push(format!("unsupported NetCDF type: {name}"));continue;}
        let (n,d,first)=compare(&a,av,&b,bv)?;words+=n;diffs+=d;
        if av.ty==5||av.ty==6 {floating_words+=n;floating_diffs+=d;}
        let first=match first {
            Some((idx,x,y))=>format!("{{\"index\":{idx:?},\"reference_bits\":\"{x:016x}\",\"other_bits\":\"{y:016x}\"}}"),
            None=>"null".into(),
        };
        records.push(format!("{{\"field\":{},\"netcdf_type\":{},\"words\":{n},\"differing_words\":{d},\"first_difference\":{first}}}",quote(name),av.ty));
    }
    if words==0 {errors.push("no field words compared".into());}
    let ok=errors.is_empty()&&diffs==0;
    let floating_ok=errors.is_empty()&&words>0&&floating_diffs==0;
    println!("{{\"schema\":\"mpas.native-word-compare/v1\",\"reference\":{},\"other\":{},\"all_identical\":{ok},\"words\":{words},\"differing_words\":{diffs},\"floating_words\":{floating_words},\"differing_floating_words\":{floating_diffs},\"all_floating_words_identical\":{floating_ok},\"errors\":[{}],\"fields\":[{}]}}",
             quote(&args[1]),quote(&args[2]),errors.iter().map(|s|quote(s)).collect::<Vec<_>>().join(","),records.join(","));
    Ok(ok)
}
fn integer_range(name: &str, path: &str) -> Result<bool,String> {
    let file=File::open(path)?;let vars=file.vars()?;
    let var=vars.get(name).ok_or("requested integer field is absent")?;
    if var.ty!=4 || var.shape.len()!=1 || var.shape[0]==0 {
        return Err("integer range requires a nonempty one-dimensional NC_INT field".into());
    }
    let mut minimum=i32::MAX;let mut maximum=i32::MIN;let mut offset=0;
    while offset<var.shape[0] {
        let n=262144.min(var.shape[0]-offset);let mut values=vec![0_i32;n];
        check(unsafe {nc_get_vara(file.0,var.id,[offset].as_ptr(),[n].as_ptr(),values.as_mut_ptr().cast())})?;
        for value in values {minimum=minimum.min(value);maximum=maximum.max(value);}
        offset+=n;
    }
    println!("{{\"schema\":\"mpas.native-integer-range/v1\",\"file\":{},\"field\":{},\"words\":{},\"minimum\":{minimum},\"maximum\":{maximum}}}",
        quote(path),quote(name),var.shape[0]);
    Ok(true)
}
fn inventory(path: &str, digests: bool) -> Result<bool,String> {
    let file=File::open(path)?;let vars=file.vars()?;
    if vars.is_empty() {return Err("native file has no variables".into());}
    let mut times=Vec::new();
    if let Some(time)=vars.get("xtime") {
        if time.ty!=2 || time.shape.len()!=2 || time.dnames[0]!="Time" {
            return Err("native xtime layout differs".into());
        }
        let total=time.shape.iter().try_fold(1_usize,|n,&d|n.checked_mul(d)).ok_or("clock shape overflow")?;
        if total==0 || total>1<<20 || time.shape[1]==0 {return Err("native xtime is empty or too large".into());}
        let mut bytes=vec![0_u8;total];let start=[0_usize;2];
        check(unsafe {nc_get_vara(file.0,time.id,start.as_ptr(),time.shape.as_ptr(),bytes.as_mut_ptr().cast())})?;
        for row in bytes.chunks_exact(time.shape[1]) {
            let text=std::str::from_utf8(row).map_err(|e|e.to_string())?.trim_end_matches(['\0',' ']);
            times.push(quote(text));
        }
    }
    let mut headers=Vec::new();
    for (name,v) in &vars {
        let extra=if digests {let (words,digest)=field_digest(&file,v)?;format!(",\"words\":{words},\"word_sha256\":{}",quote(&digest))}else{String::new()};
        headers.push(format!("{{\"field\":{},\"netcdf_type\":{},\"shape\":{:?},\"dimensions\":[{}]{extra}}}",
            quote(name),v.ty,v.shape,v.dnames.iter().map(|s|quote(s)).collect::<Vec<_>>().join(",")));
    }
    println!("{{\"schema\":\"mpas.native-netcdf-inventory/v1\",\"file\":{},\"xtime\":[{}],\"variables\":[{}]}}",
        quote(path),times.join(","),headers.join(","));
    Ok(true)
}
fn main() {
    let args=std::env::args().collect::<Vec<_>>();
    match run(&args) {Ok(true)=>(),Ok(false)=>std::process::exit(1),Err(e)=>{eprintln!("{e}");std::process::exit(2)}}
}
