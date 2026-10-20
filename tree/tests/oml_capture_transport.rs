//! Test-only batch transport for actual f1 scalar OML calls. Rust owns all
//! ORCL0001 decode, packing, output encoding and exact-word comparisons.
//! MPAS module_sf_oml.F:9-160; driver_oml.F:131-136.
#![allow(dead_code)]
// Build includes the pristine f1 reader with its two inner doc comments made
// ordinary comments (no parser statements change).
include!(env!("MPAS_OML_ORCL_SOURCE"));
use std::ffi::{CStr, c_char, c_void};

const INPUTS: [&str;17] = ["tml","t0ml","h","h0","huml","hvml","tsk",
    "hfx","lh","gsw","glw","tmoml","ust","emiss","uair","vair","f"];
const CONSTANTS: [&str;5] = ["stbolt","g","dt","oml_gamma","oml_relaxation_time"];
const OUTPUTS: [usize;5] = [0,2,4,5,6];
struct Batch { prefixes: Vec<String>, inputs: Vec<f32>, expected: Vec<f32>, constants: Vec<f32> }

fn scalar(path: &Path)->io::Result<f32> {
    let record = read(path)?;
    if record.header[..3]!=[1,4,0] || record.data.len()!=4 {
        return Err(bad("expected a native RKIND scalar"));
    }
    Ok(f32::from_bits(u32::from_le_bytes(record.data[..].try_into().unwrap())))
}
fn load(root: &Path, single:Option<&str>)->io::Result<Batch> {
    let mut prefixes:Vec<_> = fs::read_dir(root)?.map(|e| e.map(|e|e.path()))
        .collect::<io::Result<Vec<_>>>()?.into_iter()
        .filter_map(|p|p.to_str()?.strip_suffix(".pre.tml.bin").map(str::to_owned)).collect();
    if let Some(single)=single {prefixes.retain(|p|p==single);}
    prefixes.sort();
    if prefixes.is_empty() {return Err(bad("no OML captures"));}
    // Refuse an incomplete argument set or newly captured hidden state.
    let mut expected=std::collections::HashSet::new();
    for prefix in &prefixes {
        for name in INPUTS.iter().chain(CONSTANTS.iter()) {
            expected.insert(format!("{prefix}.pre.{name}.bin"));
        }
        for i in OUTPUTS {expected.insert(format!("{prefix}.post.{}.bin",INPUTS[i]));}
    }
    let actual:std::collections::HashSet<_>=fs::read_dir(root)?.map(|e|e.map(|e|e.path()))
        .collect::<io::Result<Vec<_>>>()?.into_iter().filter(|p|p.extension().is_some_and(|x|x=="bin"))
        .map(|p|p.to_str().unwrap().to_owned())
        .filter(|p|single.is_none_or(|prefix|p.starts_with(&format!("{prefix}.")))).collect();
    if actual!=expected {return Err(bad("incomplete/unexpected native argument/output/state set"));}
    let n=prefixes.len();
    let mut inputs=vec![0.;INPUTS.len()*n];
    let mut constants=vec![0.;CONSTANTS.len()];
    for (c,prefix) in prefixes.iter().enumerate() {
        for (i,name) in INPUTS.iter().enumerate() {
            inputs[i*n+c]=scalar(Path::new(&format!("{prefix}.pre.{name}.bin")))?;
        }
        for (i,name) in CONSTANTS.iter().enumerate() {
            let value=scalar(Path::new(&format!("{prefix}.pre.{name}.bin")))?;
            if c==0 {constants[i]=value;} else if constants[i].to_bits()!=value.to_bits() {
                return Err(bad("batch constants differ"));
            }
        }
    }
    // Expected outputs stay private to comparison; they never seed replay inputs.
    let mut expected=inputs.clone();
    for (c,prefix) in prefixes.iter().enumerate() {
        for i in OUTPUTS {
            expected[i*n+c]=scalar(Path::new(&format!("{prefix}.post.{}.bin",INPUTS[i])))?;
        }
    }
    Ok(Batch{prefixes,inputs,expected,constants})
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oml_batch_open(root:*const c_char)->*mut c_void {
    if root.is_null() {return std::ptr::null_mut();}
    let Ok(root)=unsafe{CStr::from_ptr(root)}.to_str() else {return std::ptr::null_mut();};
    match load(Path::new(root),None) {
        Ok(b)=>Box::into_raw(Box::new(b)).cast(),
        Err(e)=>{eprintln!("OML capture: {e}");std::ptr::null_mut()}
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oml_batch_inputs(handle:*const c_void,n:*mut usize,constants:*mut *const f32)->*const f32 {
    let b=unsafe{&*handle.cast::<Batch>()};
    unsafe{n.write(b.prefixes.len());constants.write(b.constants.as_ptr());}
    b.inputs.as_ptr()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oml_batch_compare(handle:*const c_void,values:*const f32,counts:*mut usize)->i32 {
    let b=unsafe{&*handle.cast::<Batch>()};let n=b.prefixes.len();
    let values=unsafe{std::slice::from_raw_parts(values,INPUTS.len()*n)};
    let counts=unsafe{std::slice::from_raw_parts_mut(counts,INPUTS.len())};counts.fill(0);
    for c in 0..n {
        for i in 0..INPUTS.len() {
            if b.expected[i*n+c].to_bits()!=values[i*n+c].to_bits(){counts[i]+=1;}
        }
    }
    0
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oml_batch_close(handle:*mut c_void) {
    if !handle.is_null(){drop(unsafe{Box::from_raw(handle.cast::<Batch>())});}
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oml_one_open(prefix:*const c_char)->*mut c_void {
    if prefix.is_null(){return std::ptr::null_mut();}
    let Ok(prefix)=unsafe{CStr::from_ptr(prefix)}.to_str() else{return std::ptr::null_mut();};
    let Some(root)=Path::new(prefix).parent() else{return std::ptr::null_mut();};
    match load(root,Some(prefix)) {
        Ok(b)=>Box::into_raw(Box::new(b)).cast(),
        Err(e)=>{eprintln!("OML capture: {e}");std::ptr::null_mut()}
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oml_one_write(handle:*const c_void,values:*const f32,output:*const c_char)->i32 {
    let b=unsafe{&*handle.cast::<Batch>()};
    if b.prefixes.len()!=1 || output.is_null(){return -1;}
    let Ok(output)=unsafe{CStr::from_ptr(output)}.to_str() else{return -2;};
    let values=unsafe{std::slice::from_raw_parts(values,INPUTS.len())};
    for i in OUTPUTS {
        let result=(||->io::Result<()>{
            let mut record=read(Path::new(&format!("{}.pre.{}.bin",b.prefixes[0],INPUTS[i])))?;
            record.strings[256..512].fill(b' ');
            record.strings[256..260].copy_from_slice(b"post");
            record.data=values[i].to_bits().to_le_bytes().to_vec();
            write(Path::new(&format!("{output}.post.{}.bin",INPUTS[i])),&record)
        })();
        if let Err(e)=result {eprintln!("{e}");return -3;}
    }
    0
}

// Complete captured-call batch output; only independently computed values are
// encoded, using PRE record headers. No reference POST payload is copied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oml_batch_write(handle:*const c_void,values:*const f32,output:*const c_char)->i32 {
    if handle.is_null() || values.is_null() || output.is_null(){return -1;}
    let b=unsafe{&*handle.cast::<Batch>()};
    let Ok(output)=unsafe{CStr::from_ptr(output)}.to_str() else{return -2;};
    let n=b.prefixes.len();
    let values=unsafe{std::slice::from_raw_parts(values,INPUTS.len()*n)};
    let result=(||->io::Result<()>{
        fs::create_dir(output)?;
        for (c,prefix) in b.prefixes.iter().enumerate() {
            let name=Path::new(prefix).file_name().ok_or_else(||bad("invalid capture prefix"))?;
            for i in OUTPUTS {
                let mut record=read(Path::new(&format!("{prefix}.pre.{}.bin",INPUTS[i])))?;
                record.strings[256..512].fill(b' ');
                record.strings[256..260].copy_from_slice(b"post");
                record.data=values[i*n+c].to_bits().to_le_bytes().to_vec();
                write(&Path::new(output).join(format!("{}.post.{}.bin",name.to_string_lossy(),INPUTS[i])),&record)?;
            }
        }
        Ok(())
    })();
    match result {Ok(())=>0,Err(e)=>{eprintln!("OML batch output: {e}");-3}}
}

// Synthetic strict-Fortran fixtures use native scalar record schemas, but all
// payloads come from supplied synthetic inputs and pristine Fortran outputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oml_fixture_write(template:*const c_char,before:*const f32,
    after:*const f32,constants:*const f32,n:usize,output:*const c_char)->i32 {
    if template.is_null() || before.is_null() || after.is_null() || constants.is_null()
        || output.is_null() || n==0 || n>i32::MAX as usize{return -1;}
    let (Ok(template),Ok(output))=(unsafe{CStr::from_ptr(template)}.to_str(),
        unsafe{CStr::from_ptr(output)}.to_str()) else{return -2;};
    let before=unsafe{std::slice::from_raw_parts(before,INPUTS.len()*n)};
    let after=unsafe{std::slice::from_raw_parts(after,INPUTS.len()*n)};
    let constants=unsafe{std::slice::from_raw_parts(constants,CONSTANTS.len())};
    let result=(||->io::Result<()>{
        let mut templates=Vec::new();
        for name in INPUTS.iter().chain(CONSTANTS.iter()) {
            let record=read(Path::new(&format!("{template}.pre.{name}.bin")))?;
            if record.header[..3]!=[1,4,0] || record.data.len()!=4{return Err(bad("fixture schema must be native RKIND scalar"));}
            templates.push(record);
        }
        fs::create_dir(output)?;
        for c in 0..n {
            let site=String::from_utf8_lossy(&templates[0].strings[..256]);
            let prefix=format!("{}.s0000000000.e{:010}.r000000",site.trim_end(),c+1);
            for (i,name) in INPUTS.iter().chain(CONSTANTS.iter()).enumerate() {
                let mut record=templates[i].clone();
                record.header[13]=0;record.header[14]=(c+1) as i32;record.header[15]=0;
                let value=if i<INPUTS.len(){before[i*n+c]}else{constants[i-INPUTS.len()]};
                record.data=value.to_bits().to_le_bytes().to_vec();
                write(&Path::new(output).join(format!("{prefix}.pre.{name}.bin")),&record)?;
                if OUTPUTS.contains(&i) {
                    record.strings[256..512].fill(b' ');record.strings[256..260].copy_from_slice(b"post");
                    record.data=after[i*n+c].to_bits().to_le_bytes().to_vec();
                    write(&Path::new(output).join(format!("{prefix}.post.{name}.bin")),&record)?;
                }
            }
        }
        Ok(())
    })();
    match result {Ok(())=>0,Err(e)=>{eprintln!("OML synthetic fixture: {e}");-3}}
}
