//! Diagnostic elementary-math words from complete host replay, not a referee.
//! Field-independent fixed records, Rust IO and exact GPU word grading.
use std::{ffi::CStr, fs::File, io::{BufWriter, Write}, sync::{Mutex, OnceLock}};
static OUTPUT: OnceLock<Mutex<BufWriter<File>>> = OnceLock::new();
#[unsafe(no_mangle)]
pub extern "C" fn p12_math_record(column: u64, op: u64, a: u64, b: u64, result: u64) {
    let output = OUTPUT.get_or_init(|| Mutex::new(BufWriter::new(File::create(
        std::env::var_os("P12_MATH_TRACE").expect("P12_MATH_TRACE")).expect("trace output"))));
    let mut out = output.lock().unwrap();
    for word in [column, op, a, b, result] { out.write_all(&word.to_le_bytes()).unwrap(); }
}
#[unsafe(no_mangle)]
pub extern "C" fn p12_math_flush() {
    if let Some(out) = OUTPUT.get() { out.lock().unwrap().flush().unwrap(); }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn p12_trace_open(path: *const i8) -> *mut Vec<u64> {
    let path = unsafe { CStr::from_ptr(path) }.to_str().unwrap();
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 40, 0);
    let words = bytes.chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect();
    Box::into_raw(Box::new(words))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn p12_trace_data(handle: *const Vec<u64>) -> *const u64 {
    unsafe { &*handle }.as_ptr()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn p12_trace_len(handle: *const Vec<u64>) -> usize {
    unsafe { &*handle }.len() / 5
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn p12_trace_grade(handle: *const Vec<u64>, actual: *const u64, path: *const i8) -> usize {
    let words = unsafe { &*handle };
    let actual = unsafe { std::slice::from_raw_parts(actual, words.len()/5) };
    let path = unsafe { CStr::from_ptr(path) }.to_str().unwrap();
    let mut output = BufWriter::new(File::create(path).unwrap());
    write!(output, "{{\"scope\":\"Complete host math arguments replayed on shared CUDA math; diagnostic only\",\"records\":{},\"differences\":[", actual.len()).unwrap();
    let mut count = 0;
    for (index, (record, &gpu)) in words.chunks_exact(5).zip(actual).enumerate() {
        if gpu == record[4] { continue; }
        if count != 0 { write!(output, ",").unwrap(); }
        write!(output, "{{\"index\":{index},\"column\":{},\"op\":{},\"a_bits\":{},\"b_bits\":{},\"native_bits\":{},\"cuda_bits\":{gpu}}}", record[0],record[1],record[2],record[3],record[4]).unwrap();
        count += 1;
    }
    writeln!(output, "],\"differing_words\":{count}}}").unwrap();
    output.flush().unwrap(); count
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn p12_trace_close(handle: *mut Vec<u64>) {
    drop(unsafe { Box::from_raw(handle) });
}
