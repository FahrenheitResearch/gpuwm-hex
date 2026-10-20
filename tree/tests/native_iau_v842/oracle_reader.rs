//! Test-only decoder for f1's ORCL0001 transport. No floating-point arithmetic.
//! MPAS field/time-level selection remains explicit in the replay manifest.
use std::{ffi::CStr, fs, os::raw::c_char, panic, ptr};

const FIXED: usize = 8 + 21 * 4 + 3 * 256;
struct Field<'a> { header: [i32; 21], data: &'a [u8], count: usize }

fn decode(raw: &[u8]) -> Result<Field<'_>, &'static str> {
    decode_typed(raw, 1, 1, 3, b"pre")
}

fn decode_typed<'a>(raw: &'a [u8], kind: i32, min_rank: i32, max_rank: i32, phase: &[u8])
    -> Result<Field<'a>, &'static str> {
    if raw.len() < FIXED || &raw[..8] != b"ORCL0001" { return Err("header"); }
    let mut h = [0; 21];
    for (i, value) in h.iter_mut().enumerate() {
        *value = i32::from_le_bytes(raw[8+4*i..12+4*i].try_into().unwrap());
    }
    if h[0] != kind || h[1] != 4 || !(min_rank..=max_rank).contains(&h[2]) || h[19] != 256 || h[20] != 0 {
        return Err("real32 rank/version");
    }
    let rank = h[2] as usize;
    if h[3..3+rank].iter().any(|v| *v < 0) || h[3+rank..8].iter().any(|v| *v != 1) {
        return Err("dimensions");
    }
    if !(0..=h[2]).contains(&h[17]) || !(0..=h[2]).contains(&h[18])
        || (h[17] > 0 && h[17] == h[18]) {
        return Err("axes");
    }
    let ids = if h[17] > 0 { h[2+h[17] as usize] } else { 0 };
    if h[16] < 0 || h[16] != ids { return Err("ID count"); }
    let count = h[3..3+rank].iter().try_fold(1usize, |a, v| a.checked_mul(*v as usize))
        .ok_or("count overflow")?;
    let offset = (h[16] as usize).checked_mul(4).and_then(|v| FIXED.checked_add(v))
        .ok_or("offset overflow")?;
    let size = count.checked_mul(4).and_then(|v| offset.checked_add(v)).ok_or("size overflow")?;
    if raw.len() != size { return Err("payload length"); }
    if raw[92..348].iter().copied().take_while(|v| *v != b' ').collect::<Vec<_>>()
        != b"boundary__atm_add_tend_anal_incr"
        || raw[348..604].iter().copied().take_while(|v| *v != b' ').collect::<Vec<_>>() != phase {
        return Err("IAU pre-call context");
    }
    Ok(Field { header: h, data: &raw[offset..], count })
}

/// Read an actual native scalar dimension/config/argument word and its context.
/// Supports int32 or real32 only. No value or field is inferred by this adapter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iau_oracle_control(path: *const c_char, kind: i32,
    header: *mut i32, value: *mut u32) -> i32 {
    panic::catch_unwind(|| {
        if path.is_null() || header.is_null() || value.is_null() { return -3; }
        if kind != 1 && kind != 3 { return -1; }
        let Ok(path) = (unsafe { CStr::from_ptr(path) }).to_str() else { return -1; };
        let Ok(raw) = fs::read(path) else { return -1; };
        let Ok(field) = decode_typed(&raw,kind,0,0,b"pre") else { return -1; };
        let word = u32::from_le_bytes(field.data.try_into().unwrap());
        unsafe { ptr::copy_nonoverlapping(field.header.as_ptr(),header,21);
            ptr::write(value,word); }
        0
    }).unwrap_or(-1)
}

fn words(field: &Field<'_>) -> (Vec<u32>, Vec<u32>) {
    let original: Vec<u32> = field.data.chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect();
    let rank = field.header[2] as usize;
    let dims = &field.header[3..3+rank];
    let mut row_major = vec![0; field.count];
    for (c, value) in row_major.iter_mut().enumerate() {
        let mut remaining = c;
        let mut coordinates = [0usize; 3];
        for axis in (0..rank).rev() {
            coordinates[axis] = remaining % dims[axis] as usize;
            remaining /= dims[axis] as usize;
        }
        let mut f = 0;
        let mut stride = 1;
        for axis in 0..rank {
            f += coordinates[axis] * stride;
            stride *= dims[axis] as usize;
        }
        *value = original[f];
    }
    (original, row_major)
}

/// Read a real32 field. Query with null outputs/capacity zero; then supply two
/// buffers of exactly the returned element count. Header has 21 int32 words.
/// Returns element count, or -1 invalid record, -2 capacity, -3 null pointers.
/// Caller owns all pointers/buffers. No allocation crosses this FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iau_oracle_read(path: *const c_char, header: *mut i32,
    original: *mut u32, row_major: *mut u32, capacity: usize) -> isize {
    unsafe { read_field(path,header,original,row_major,capacity,b"pre") }
}

/// Decode post-call words only for comparison; never initialize native inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iau_oracle_read_post(path: *const c_char, header: *mut i32,
    original: *mut u32, row_major: *mut u32, capacity: usize) -> isize {
    unsafe { read_field(path,header,original,row_major,capacity,b"post") }
}

/// Align a boundary field to the forecast's explicitly supplied global IDs.
/// A native MPI1 edge pool can still be in a different local order. The final
/// entity is framework dummy storage and never belongs to the resident mesh.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iau_oracle_read_ids(path: *const c_char, post: i32,
    ids: *const i32, entities: usize, output: *mut u32, capacity: usize) -> isize {
    unsafe { read_ids(path,post,ptr::null(),ids,entities,output,capacity) }
}

/// The native pool writer records cell IDs only. An edge field can instead be
/// aligned from its actual integer indexToEdgeID record at the same boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iau_oracle_read_external_ids(path: *const c_char, post: i32,
    native_ids: *const c_char, ids: *const i32, entities: usize,
    output: *mut u32, capacity: usize) -> isize {
    if native_ids.is_null() { return -3; }
    unsafe { read_ids(path,post,native_ids,ids,entities,output,capacity) }
}

unsafe fn read_ids(path: *const c_char, post: i32, native_ids: *const c_char,
    ids: *const i32, entities: usize, output: *mut u32, capacity: usize) -> isize {
    panic::catch_unwind(|| {
        if path.is_null() || ids.is_null() || output.is_null() { return -3; }
        let Ok(path) = (unsafe { CStr::from_ptr(path) }).to_str() else { return -1; };
        let Ok(raw) = fs::read(path) else { return -1; };
        let phase: &[u8] = if post == 0 { b"pre" } else { b"post" };
        let Ok(field) = decode_typed(&raw,1,1,3,phase) else { return -1; };
        let rank = field.header[2] as usize;
        if field.header[2+rank] as usize != entities+1 { return -1; }
        let external;
        let id_bytes = if native_ids.is_null() {
            if field.header[17] as usize != rank || field.header[16] as usize != entities+1 { return -1; }
            &raw[FIXED..FIXED+4*(entities+1)]
        } else {
            if field.header[16] != 0 || field.header[17] != 0 { return -1; }
            let Ok(path) = (unsafe { CStr::from_ptr(native_ids) }).to_str() else { return -1; };
            let Ok(bytes) = fs::read(path) else { return -1; };
            external = bytes;
            let Ok(record) = decode_typed(&external,3,1,1,b"pre") else { return -1; };
            if record.count != entities+1 || record.header[13..16] != field.header[13..16]
                || external[604..860].iter().copied().take_while(|v| *v != b' ').collect::<Vec<_>>()
                    != b"block.structs.mesh.indexToEdgeID" { return -1; }
            record.data
        };
        let count = field.count/(entities+1)*entities;
        if count != capacity { return -2; }
        let mut local = std::collections::HashMap::new();
        for j in 0..entities {
            let id = i32::from_le_bytes(id_bytes[4*j..4*j+4].try_into().unwrap());
            if id <= 0 || local.insert(id,j).is_some() { return -1; }
        }
        let target = unsafe { std::slice::from_raw_parts(ids,entities) };
        let mut seen = std::collections::HashSet::new();
        let mut order = Vec::with_capacity(entities);
        for id in target {
            if !seen.insert(*id) { return -1; }
            let Some(j) = local.get(id) else { return -1; };
            order.push(*j);
        }
        let (_, c) = words(&field);
        let out = unsafe { std::slice::from_raw_parts_mut(output,capacity) };
        for (i,v) in out.iter_mut().enumerate() {
            *v = c[(i/entities)*(entities+1)+order[i%entities]];
        }
        count as isize
    }).unwrap_or(-1)
}

unsafe fn read_field(path: *const c_char, header: *mut i32, original: *mut u32,
    row_major: *mut u32, capacity: usize, phase: &[u8]) -> isize {
    panic::catch_unwind(|| {
        if path.is_null() || header.is_null() { return -3; }
        let Ok(path) = (unsafe { CStr::from_ptr(path) }).to_str() else { return -1; };
        let Ok(raw) = fs::read(path) else { return -1; };
        let Ok(field) = decode_typed(&raw,1,1,3,phase) else { return -1; };
        if field.count > isize::MAX as usize { return -1; }
        unsafe { ptr::copy_nonoverlapping(field.header.as_ptr(), header, 21); }
        if original.is_null() && row_major.is_null() && capacity == 0 {
            return field.count as isize;
        }
        if capacity != field.count { return -2; }
        if original.is_null() || row_major.is_null() { return -3; }
        let (f, c) = words(&field);
        unsafe {
            ptr::copy_nonoverlapping(f.as_ptr(), original, f.len());
            ptr::copy_nonoverlapping(c.as_ptr(), row_major, c.len());
        }
        field.count as isize
    }).unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aligns_local_entity_ids_and_excludes_only_dummy_storage() {
        let mut raw = fixture(&[2,4], &[30,31,10,11,20,21,0,0]);
        for (i,v) in [(16,4i32),(17,2),(18,1)] {
            raw[8+4*i..12+4*i].copy_from_slice(&v.to_le_bytes());
        }
        let ids: Vec<u8> = [3i32,1,2,0].into_iter().flat_map(i32::to_le_bytes).collect();
        raw.splice(FIXED..FIXED,ids);
        let file = std::path::PathBuf::from(std::env::var("MPAS_IAU_TEST_SCRATCH").unwrap())
            .join("global-id-fixture.bin");
        fs::write(&file,raw).unwrap();
        let path = std::ffi::CString::new(file.to_str().unwrap()).unwrap();
        let desired = [1,2,3];
        let mut out = [0u32;6];
        unsafe {
            assert_eq!(iau_oracle_read_ids(path.as_ptr(),0,desired.as_ptr(),3,out.as_mut_ptr(),6),6);
            assert_eq!(out,[10,20,30,11,21,31]);
            assert_eq!(iau_oracle_read_ids(path.as_ptr(),0,[1,1,3].as_ptr(),3,out.as_mut_ptr(),6),-1);
            assert_eq!(iau_oracle_read_ids(path.as_ptr(),0,desired.as_ptr(),3,out.as_mut_ptr(),5),-2);
        }
        fs::remove_file(file).unwrap();
    }
    #[test]
    fn aligns_actual_external_edge_ids_with_authenticated_boundary_context() {
        let scratch = std::path::PathBuf::from(std::env::var("MPAS_IAU_TEST_SCRATCH").unwrap());
        let field = scratch.join("external-edge-field.bin");
        let native_ids = scratch.join("external-edge-ids.bin");
        fs::write(&field,fixture(&[2,4], &[30,31,10,11,20,21,0,0])).unwrap();
        let mut raw = fixture(&[4], &[3,1,2,0]);
        raw[8..12].copy_from_slice(&3i32.to_le_bytes());
        let name = b"block.structs.mesh.indexToEdgeID";
        raw[604..604+name.len()].copy_from_slice(name);
        fs::write(&native_ids,&raw).unwrap();
        let a = std::ffi::CString::new(field.to_str().unwrap()).unwrap();
        let b = std::ffi::CString::new(native_ids.to_str().unwrap()).unwrap();
        let mut out = [0u32;6];
        unsafe {
            assert_eq!(iau_oracle_read_external_ids(a.as_ptr(),0,b.as_ptr(),[1,2,3].as_ptr(),3,out.as_mut_ptr(),6),6);
            assert_eq!(out,[10,20,30,11,21,31]);
            raw[8+4*13..12+4*13].copy_from_slice(&1i32.to_le_bytes());
            fs::write(&native_ids,raw).unwrap();
            assert_eq!(iau_oracle_read_external_ids(a.as_ptr(),0,b.as_ptr(),[1,2,3].as_ptr(),3,out.as_mut_ptr(),6),-1);
        }
        fs::remove_file(field).unwrap();fs::remove_file(native_ids).unwrap();
    }
    fn fixture(shape: &[i32], values: &[u32]) -> Vec<u8> {
        let mut h = [0; 21];
        h[0] = 1; h[1] = 4; h[2] = shape.len() as i32;
        h[3..8].fill(1); h[8..13].fill(1);
        h[3..3+shape.len()].copy_from_slice(shape); h[19] = 256;
        let mut raw = b"ORCL0001".to_vec();
        for v in h { raw.extend(v.to_le_bytes()); }
        raw.resize(FIXED, b' ');
        let key = b"boundary__atm_add_tend_anal_incr";
        raw[92..92+key.len()].copy_from_slice(key);
        raw[348..351].copy_from_slice(b"pre");
        for v in values { raw.extend(v.to_le_bytes()); }
        raw
    }
    #[test]
    fn words_preserve_signed_zero_nan_and_subnormal_in_both_layouts() {
        let bits = [0, 0x80000000, 1, 0x7fc00123, 0x3f800000, 0xff800000];
        let raw = fixture(&[2, 3], &bits);
        let field = decode(&raw).unwrap();
        let (f, c) = words(&field);
        assert_eq!(f, bits);
        assert_eq!(c, [bits[0], bits[2], bits[4], bits[1], bits[3], bits[5]]);
    }
    #[test]
    fn scalar_level_cell_layout_retains_every_word() {
        let bits: Vec<u32> = (0..24).collect();
        let raw = fixture(&[2, 3, 4], &bits);
        let (_, c) = words(&decode(&raw).unwrap());
        for s in 0..2 { for k in 0..3 { for i in 0..4 {
            assert_eq!(c[(s*3+k)*4+i], bits[s+2*k+6*i]);
        }}}
    }
    #[test]
    fn malformed_records_are_rejected() {
        let mut raw = fixture(&[2, 3], &[0; 6]);
        raw.pop(); assert!(decode(&raw).is_err());
        let mut raw = fixture(&[2, 3], &[0; 6]);
        raw.extend([0]); assert!(decode(&raw).is_err());
        for (index, value) in [(0,2), (1,8), (2,4), (3,-1), (5,2),
                               (16,1), (17,3), (19,128), (20,1)] {
            let mut raw = fixture(&[2, 3], &[0; 6]);
            raw[8+4*index..12+4*index].copy_from_slice(&i32::to_le_bytes(value));
            assert!(decode(&raw).is_err(), "header index {index}");
        }
        let mut raw = fixture(&[2, 3], &[0; 6]);
        raw[348..352].copy_from_slice(b"post");
        assert!(decode(&raw).is_err());
    }
    #[test]
    fn empty_and_one_dimensional_records() {
        let raw = fixture(&[0, 3], &[]);
        let (f, c) = words(&decode(&raw).unwrap());
        assert!(f.is_empty() && c.is_empty());
        let raw = fixture(&[3], &[1,2,3]);
        let (f, c) = words(&decode(&raw).unwrap());
        assert_eq!(f, c);
    }
    #[test]
    fn ffi_controls_preserve_native_type_and_context() {
        let dir = env_dir();
        let file = std::path::Path::new(&dir).join(format!("control-{}.bin", std::process::id()));
        let path = std::ffi::CString::new(file.to_str().unwrap()).unwrap();
        let mut header = [0;21]; let mut value = 0;
        for (kind,word) in [(1,600f32.to_bits()),(3,10242u32)] {
            let mut raw = fixture(&[],&[word]);
            raw[8..12].copy_from_slice(&i32::to_le_bytes(kind));
            raw[60..64].copy_from_slice(&1i32.to_le_bytes());
            fs::write(&file,&raw).unwrap();
            unsafe {
                assert_eq!(iau_oracle_control(path.as_ptr(),kind,header.as_mut_ptr(),&mut value),0);
                assert_eq!(value,word); assert_eq!(header[13],1);
                assert_eq!(iau_oracle_control(path.as_ptr(),if kind==1 {3} else {1},header.as_mut_ptr(),&mut value),-1);
                assert_eq!(iau_oracle_control(path.as_ptr(),kind,header.as_mut_ptr(),ptr::null_mut()),-3);
            }
        }
        fs::remove_file(file).unwrap();
    }
    fn env_dir() -> String {
        std::env::var("MPAS_IAU_TEST_SCRATCH").expect("owned scratch directory required")
    }
    #[test]
    fn ffi_post_storage_is_comparison_only() {
        let file = std::path::Path::new(&env_dir()).join(format!("post-{}.bin",std::process::id()));
        let path = std::ffi::CString::new(file.to_str().unwrap()).unwrap();
        let mut raw = fixture(&[2],&[1,0x80000000]);
        raw[348..352].copy_from_slice(b"post");
        fs::write(&file,&raw).unwrap();
        let mut header = [0;21]; let mut f = [0;2]; let mut c = [0;2];
        unsafe {
            assert_eq!(iau_oracle_read(path.as_ptr(),header.as_mut_ptr(),f.as_mut_ptr(),c.as_mut_ptr(),2),-1);
            assert_eq!(iau_oracle_read_post(path.as_ptr(),header.as_mut_ptr(),f.as_mut_ptr(),c.as_mut_ptr(),2),2);
        }
        assert_eq!(f,[1,0x80000000]); assert_eq!(c,f);
        fs::remove_file(file).unwrap();
    }
    #[test]
    fn ffi_validates_capacity_and_preserves_words() {
        use std::ffi::CString;
        let dir = std::env::var("MPAS_IAU_TEST_SCRATCH").expect("owned scratch directory required");
        let fixture_output = std::env::var("MPAS_IAU_TEST_FIXTURE").ok();
        let file = fixture_output.as_ref().map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(&dir).join(format!("ffi-{}.bin", std::process::id())));
        let bits = [0, 0x80000000, 1, 0x7fc00123, 0x3f800000, 0xff800000];
        fs::write(&file, fixture(&[2, 3], &bits)).unwrap();
        let path = CString::new(file.to_str().unwrap()).unwrap();
        let mut header = [0; 21];
        let mut f = [0; 6]; let mut c = [0; 6];
        unsafe {
            assert_eq!(iau_oracle_read(path.as_ptr(), header.as_mut_ptr(), ptr::null_mut(), ptr::null_mut(), 0), 6);
            assert_eq!(header[3..5], [2, 3]);
            assert_eq!(iau_oracle_read(path.as_ptr(), header.as_mut_ptr(), f.as_mut_ptr(), c.as_mut_ptr(), 5), -2);
            assert_eq!(iau_oracle_read(path.as_ptr(), header.as_mut_ptr(), f.as_mut_ptr(), ptr::null_mut(), 6), -3);
            assert_eq!(iau_oracle_read(path.as_ptr(), header.as_mut_ptr(), f.as_mut_ptr(), c.as_mut_ptr(), 6), 6);
            assert_eq!(iau_oracle_read(ptr::null(), header.as_mut_ptr(), f.as_mut_ptr(), c.as_mut_ptr(), 6), -3);
        }
        assert_eq!(f, bits);
        assert_eq!(c, [bits[0], bits[2], bits[4], bits[1], bits[3], bits[5]]);
        if fixture_output.is_none() { fs::remove_file(file).unwrap(); }
    }
}
