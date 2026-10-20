// Diagnostic transport only: Rust records exact host-libm operands/results.
// No numerical algorithm, conversion of weather fields or replacement libm.
use std::{collections::BTreeMap, ffi::{c_char, c_void, CStr}, fs, io::{self, Write},
          path::Path, sync::{LazyLock, Mutex}};

const FUNCTIONS: [(&str, u32); 14] = [
    ("expf",32),("exp",64),("logf",32),("log",64),("log10f",32),("log10",64),
    ("powf",32),("pow",64),("atanf",32),("atan",64),("sinf",32),("sin",64),
    ("tanhf",32),("tanh",64)];
type Key = (u32, u64, u64, u64);
static TRACE: LazyLock<Mutex<BTreeMap<Key,u64>>> = LazyLock::new(||Mutex::new(BTreeMap::new()));

#[no_mangle]
pub extern "C" fn nssl_trace_reset() { TRACE.lock().unwrap().clear(); }

#[no_mangle]
pub extern "C" fn nssl_trace_record(id:u32, x:u64, y:u64, result:u64) {
    assert!((id as usize)<FUNCTIONS.len());
    *TRACE.lock().unwrap().entry((id,x,y,result)).or_default() += 1;
}

unsafe fn path<'a>(p:*const c_char)-> &'a Path { Path::new(CStr::from_ptr(p).to_str().unwrap()) }

#[no_mangle]
pub unsafe extern "C" fn nssl_trace_flush(directory:*const c_char)->i32 {
    let run = || -> io::Result<()> {
        let directory=path(directory); fs::create_dir(directory)?;
        let trace=TRACE.lock().unwrap(); let mut rows=Vec::new();
        for (id,(name,bits)) in FUNCTIONS.iter().enumerate() {
            let mut file=fs::File::create(directory.join(format!("{name}.words")))?;
            let mut unique=0u64; let mut calls=0u64;
            for (&(function,x,y,result),&count) in trace.iter() {
                if function!=id as u32 { continue; }
                unique+=1; calls+=count;
                for word in [x,y,result,count] { file.write_all(&word.to_le_bytes())?; }
            }
            rows.push(format!("{{\"id\":{id},\"name\":\"{name}\",\"bits\":{bits},\"unique\":{unique},\"calls\":{calls}}}"));
        }
        fs::write(directory.join("inventory.json"),format!("{{\"functions\":[{}]}}\n",rows.join(",")))
    };
    match run() {Ok(())=>0,Err(e)=>{eprintln!("NSSL math trace: {e}");1}}
}

enum Values { F32(Vec<f32>), F64(Vec<f64>) }
impl Values {
    fn from_words(bits:u32, words:impl Iterator<Item=u64>)->Self {
        if bits==32 {Self::F32(words.map(|v|f32::from_bits(v as u32)).collect())}
        else {Self::F64(words.map(f64::from_bits).collect())}
    }
    fn pointer(&self)->*const c_void { match self {Self::F32(v)=>v.as_ptr().cast(),Self::F64(v)=>v.as_ptr().cast()} }
}
struct Bank { id:usize, rows:Vec<[u64;4]>, x:Values, y:Values, expected:Values }

#[no_mangle]
pub unsafe extern "C" fn nssl_trace_load(file:*const c_char,id:usize,count:*mut usize,
    x:*mut *const c_void,y:*mut *const c_void,expected:*mut *const c_void)->*mut c_void {
    let run=||->io::Result<Bank> {
        let bytes=fs::read(path(file))?;
        if id>=FUNCTIONS.len() || bytes.len()%32!=0 {return Err(io::Error::other("invalid math trace bank"));}
        let rows:Vec<[u64;4]>=bytes.chunks_exact(32).map(|row|
            std::array::from_fn(|i|u64::from_le_bytes(row[i*8..i*8+8].try_into().unwrap()))).collect();
        let bits=FUNCTIONS[id].1;
        let x=Values::from_words(bits,rows.iter().map(|r|r[0]));
        let y=Values::from_words(bits,rows.iter().map(|r|r[1]));
        let expected=Values::from_words(bits,rows.iter().map(|r|r[2]));
        Ok(Bank{id,rows,x,y,expected})
    };
    match run() {Ok(bank)=>{*count=bank.rows.len(); *x=bank.x.pointer(); *y=bank.y.pointer();
        *expected=bank.expected.pointer(); Box::into_raw(Box::new(bank)).cast()},
        Err(e)=>{eprintln!("NSSL trace load: {e}"); std::ptr::null_mut()}}
}

#[no_mangle]
pub unsafe extern "C" fn nssl_trace_compare(handle:*mut c_void,actual:*const c_void,out:*const c_char)->i32 {
    let bank=&*handle.cast::<Bank>(); let (name,bits)=FUNCTIONS[bank.id];
    let mut different=0; let mut differing_calls=0u64; let mut witnesses=Vec::new();
    for (i,row) in bank.rows.iter().enumerate() {
        let value=if bits==32 {*actual.cast::<u32>().add(i) as u64} else {*actual.cast::<u64>().add(i)};
        if value!=row[2] {
            different+=1; differing_calls+=row[3];
            if witnesses.len()<16 {witnesses.push(format!("{{\"x\":\"{:016x}\",\"y\":\"{:016x}\",\"host\":\"{:016x}\",\"cuda\":\"{value:016x}\",\"calls\":{}}}",row[0],row[1],row[2],row[3]));}
        }
    }
    let calls:u64=bank.rows.iter().map(|r|r[3]).sum();
    let result=format!("{{\"name\":\"{name}\",\"bits\":{bits},\"unique_operands\":{},\"calls\":{calls},\"differing_words\":{different},\"differing_calls\":{differing_calls},\"witnesses\":[{}]}}\n",bank.rows.len(),witnesses.join(","));
    if let Err(e)=fs::write(path(out),result) {eprintln!("NSSL trace comparison: {e}");return -1;}
    i32::from(different!=0)
}

#[no_mangle]
pub unsafe extern "C" fn nssl_trace_close(handle:*mut c_void) {drop(Box::from_raw(handle.cast::<Bank>()));}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    #[test]
    fn trace_preserves_raw_words_and_comparison_detects_one_bit() {
        let root=std::env::var_os("TMPDIR").expect("lane TMPDIR required");
        let root=Path::new(&root).join(format!("nssl-math-trace-test-{}",std::process::id()));
        let directory=CString::new(root.to_str().unwrap()).unwrap();
        nssl_trace_reset();
        // Signed zero and a noncanonical NaN payload: no floating arithmetic.
        let values=[0x8000_0000u64,0x7ff8_0000_0000_0051u64];
        for (id,&word) in values.iter().enumerate() {
            nssl_trace_record(id as u32,word,0,word);
            nssl_trace_record(id as u32,word,0,word);
        }
        unsafe {assert_eq!(nssl_trace_flush(directory.as_ptr()),0);}
        for (id,&word) in values.iter().enumerate() {
            let file=CString::new(root.join(format!("{}.words",FUNCTIONS[id].0)).to_str().unwrap()).unwrap();
            let out=root.join(format!("comparison-{id}.json"));
            let destination=CString::new(out.to_str().unwrap()).unwrap();
            let mut count=0;let mut x=std::ptr::null();let mut y=std::ptr::null();let mut expected=std::ptr::null();
            unsafe {
                let handle=nssl_trace_load(file.as_ptr(),id,&mut count,&mut x,&mut y,&mut expected);
                assert!(!handle.is_null());assert_eq!(count,1);
                assert_eq!(nssl_trace_compare(handle,expected,destination.as_ptr()),0);
                let wrong=word^1;
                assert_eq!(nssl_trace_compare(handle,(&wrong as *const u64).cast(),destination.as_ptr()),1);
                let result=fs::read_to_string(out).unwrap();
                assert!(result.contains("\"differing_words\":1"));
                assert!(result.contains("\"differing_calls\":2"));
                nssl_trace_close(handle);
            }
        }
        fs::remove_dir_all(root).unwrap();
    }
}
