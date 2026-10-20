//! Lossless primitive-variable copy into CDF-5. No field arithmetic.
use std::{env, ffi::{CStr, CString}, os::raw::{c_char, c_int, c_void}, ptr};

#[link(name = "netcdf")]
extern "C" {
    fn nc_open(p: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_create(p: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_inq(id: c_int, nd: *mut c_int, nv: *mut c_int, na: *mut c_int, unlim: *mut c_int) -> c_int;
    fn nc_inq_dim(id: c_int, dim: c_int, name: *mut c_char, len: *mut usize) -> c_int;
    fn nc_def_dim(id: c_int, name: *const c_char, len: usize, dim: *mut c_int) -> c_int;
    fn nc_inq_var(id: c_int, var: c_int, name: *mut c_char, kind: *mut c_int, nd: *mut c_int, dims: *mut c_int, na: *mut c_int) -> c_int;
    fn nc_def_var(id: c_int, name: *const c_char, kind: c_int, nd: c_int, dims: *const c_int, var: *mut c_int) -> c_int;
    fn nc_inq_attname(id: c_int, var: c_int, index: c_int, name: *mut c_char) -> c_int;
    fn nc_copy_att(src: c_int, var: c_int, name: *const c_char, dst: c_int, dstvar: c_int) -> c_int;
    fn nc_inq_type(id: c_int, kind: c_int, name: *mut c_char, size: *mut usize) -> c_int;
    fn nc_set_fill(id: c_int, fill: c_int, old: *mut c_int) -> c_int;
    fn nc__enddef(id: c_int, header_free: usize, variable_align: usize, variable_free: usize, record_align: usize) -> c_int;
    fn nc_get_vara(id: c_int, var: c_int, start: *const usize, count: *const usize, data: *mut c_void) -> c_int;
    fn nc_put_vara(id: c_int, var: c_int, start: *const usize, count: *const usize, data: *const c_void) -> c_int;
}

fn check(code: c_int, context: &str) -> Result<(), String> {
    if code == 0 { Ok(()) } else {
        Err(format!("{context}: {}", unsafe { CStr::from_ptr(nc_strerror(code)) }.to_string_lossy()))
    }
}
struct File(c_int);
impl File {
    fn finish(mut self) -> Result<(), String> {
        let id = self.0; self.0 = -1;
        check(unsafe { nc_close(id) }, "close completed NetCDF")
    }
}
impl Drop for File { fn drop(&mut self) { if self.0 >= 0 { unsafe { nc_close(self.0); } } } }

unsafe fn attrs(src: c_int, var: c_int, dst: c_int, dstvar: c_int, n: c_int) -> Result<(), String> {
    for i in 0..n {
        let mut name = [0 as c_char; 257];
        check(nc_inq_attname(src, var, i, name.as_mut_ptr()), "attribute name")?;
        check(nc_copy_att(src, var, name.as_ptr(), dst, dstvar), "attribute copy")?;
    }
    Ok(())
}

unsafe fn copy(srcpath: &str, dstpath: &str) -> Result<(), String> {
    let mut id = 0;
    check(nc_open(CString::new(srcpath).map_err(|e| e.to_string())?.as_ptr(), 0, &mut id), "open source")?;
    let src = File(id);
    // NC_64BIT_DATA | NC_NOCLOBBER. Refuse an existing destination.
    check(nc_create(CString::new(dstpath).map_err(|e| e.to_string())?.as_ptr(), 0x20 | 0x04, &mut id), "create CDF5")?;
    let dst = File(id);
    let (mut nd, mut nv, mut na, mut unlim) = (0, 0, 0, -1);
    check(nc_inq(src.0, &mut nd, &mut nv, &mut na, &mut unlim), "source dimensions")?;
    let mut lengths = Vec::new();
    let mut output_dims = Vec::new();
    for d in 0..nd {
        let mut name = [0 as c_char; 257];
        let mut len = 0;
        check(nc_inq_dim(src.0, d, name.as_mut_ptr(), &mut len), "dimension")?;
        let mut output = 0;
        check(nc_def_dim(dst.0, name.as_ptr(), if d == unlim { 0 } else { len }, &mut output), "define dimension")?;
        lengths.push(len);
        output_dims.push(output);
    }
    attrs(src.0, -1, dst.0, -1, na)?;
    let mut variables = Vec::new();
    for v in 0..nv {
        let mut name = [0 as c_char; 257];
        let (mut kind, mut dims_count, mut attributes) = (0, 0, 0);
        let mut dims = vec![0; nd as usize];
        check(nc_inq_var(src.0, v, name.as_mut_ptr(), &mut kind, &mut dims_count, dims.as_mut_ptr(), &mut attributes), "variable")?;
        if !(1..=11).contains(&kind) { return Err(format!("unsupported non-primitive NetCDF type {kind}")); }
        dims.truncate(dims_count as usize);
        let shape: Vec<usize> = dims.iter().map(|&d| lengths[d as usize]).collect();
        let mapped: Vec<c_int> = dims.iter().map(|&d| output_dims[d as usize]).collect();
        let mut output = 0;
        check(nc_def_var(dst.0, name.as_ptr(), kind, dims_count, mapped.as_ptr(), &mut output), "define variable")?;
        attrs(src.0, v, dst.0, output, attributes)?;
        let mut size = 0;
        check(nc_inq_type(src.0, kind, ptr::null_mut(), &mut size), "primitive size")?;
        variables.push((v, output, shape, size, CStr::from_ptr(name.as_ptr()).to_string_lossy().to_string()));
    }
    check(nc_set_fill(dst.0, 0x100, ptr::null_mut()), "disable output prefill")?;
    // The vertical door adds dimensions, variables and provenance attributes.
    // With no header reserve, each classic-format redefinition moves tens
    // of GB: the dense init wrote >975 GB before finishing its 38 GB schema.
    // Spare header bytes change file layout only; stored field bits are copied.
    check(nc__enddef(dst.0, 1024 * 1024, 4, 0, 4), "finish CDF5 header with growth reserve")?;
    for (v, output, shape, size, name) in variables {
        if shape.contains(&0) { continue; }
        let rows = shape.first().copied().unwrap_or(1);
        let row_values = shape.iter().skip(1).try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or("row size overflow")?;
        let row_bytes = row_values.checked_mul(size).ok_or("row bytes overflow")?;
        let slab = (4 * 1024 * 1024 / row_bytes.max(1)).max(1);
        for first in (0..rows).step_by(slab) {
            let n = (rows - first).min(slab);
            let bytes = n.checked_mul(row_bytes).ok_or("slab bytes overflow")?;
            // u64 alignment supports every primitive type carried by CDF5.
            let mut buffer = vec![0u64; bytes.div_ceil(8)];
            let mut start = vec![0usize; shape.len()];
            let mut count = shape.clone();
            if !shape.is_empty() { start[0] = first; count[0] = n; }
            check(nc_get_vara(src.0, v, start.as_ptr(), count.as_ptr(), buffer.as_mut_ptr().cast()), &format!("read {name}"))?;
            check(nc_put_vara(dst.0, output, start.as_ptr(), count.as_ptr(), buffer.as_ptr().cast()), &format!("write {name}"))?;
        }
        println!("COPIED\t{name}\t{}", rows * row_bytes);
    }
    dst.finish()?;
    src.finish()?;
    println!("CDF5\t{dstpath}");
    Ok(())
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 { eprintln!("usage: gpuwm-hex-netcdf-copy INPUT.nc OUTPUT.nc"); std::process::exit(2); }
    if let Err(error) = unsafe { copy(&args[1], &args[2]) } { eprintln!("gpuwm-hex-netcdf-copy: {error}"); std::process::exit(1); }
}
