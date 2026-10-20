//! Startup input ownership instrument; no model arithmetic or exclusions.
//! Arguments: native output, initialization, hex output, timed surface file.
//! MPAS ordering: mpas_atm_core.F:198-210,655-724,779-806.
use std::ffi::CString;
#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(path: *const i8, mode: i32, id: *mut i32) -> i32;
    fn nc_close(id: i32) -> i32;
    fn nc_inq_varid(id: i32, name: *const i8, var: *mut i32) -> i32;
    fn nc_inq_vartype(id: i32, var: i32, kind: *mut i32) -> i32;
    fn nc_inq_varndims(id: i32, var: i32, rank: *mut i32) -> i32;
    fn nc_inq_vardimid(id: i32, var: i32, dims: *mut i32) -> i32;
    fn nc_inq_dimid(id: i32, name: *const i8, dim: *mut i32) -> i32;
    fn nc_inq_dimlen(id: i32, dim: i32, len: *mut usize) -> i32;
    fn nc_get_vara_float(id: i32, var: i32, start: *const usize, count: *const usize, data: *mut f32) -> i32;
}
fn load(path: &str) -> Vec<f32> {
    unsafe {
        let mut id = 0;
        assert_eq!(nc_open(CString::new(path).unwrap().as_ptr(), 0, &mut id), 0);
        let (mut var, mut kind, mut rank) = (0, 0, 0);
        assert_eq!(nc_inq_varid(id, CString::new("xice").unwrap().as_ptr(), &mut var), 0);
        assert_eq!(nc_inq_vartype(id, var, &mut kind), 0);
        assert_eq!(kind, 5, "Require NC_FLOAT; no precision conversion");
        assert_eq!(nc_inq_varndims(id, var, &mut rank), 0);
        assert_eq!(rank, 2);
        let (mut time_dim, mut cell_dim) = (0, 0);
        assert_eq!(nc_inq_dimid(id, CString::new("Time").unwrap().as_ptr(), &mut time_dim), 0);
        assert_eq!(nc_inq_dimid(id, CString::new("nCells").unwrap().as_ptr(), &mut cell_dim), 0);
        let mut dims = [0, 0];
        assert_eq!(nc_inq_vardimid(id, var, dims.as_mut_ptr()), 0);
        assert_eq!(dims, [time_dim, cell_dim]);
        let (mut records, mut cells) = (0, 0);
        assert_eq!(nc_inq_dimlen(id, time_dim, &mut records), 0);
        assert!(records > 0);
        assert_eq!(nc_inq_dimlen(id, cell_dim, &mut cells), 0);
        let mut values = vec![0f32; cells];
        assert_eq!(nc_get_vara_float(id, var, [0, 0].as_ptr(), [1, cells].as_ptr(), values.as_mut_ptr()), 0);
        assert_eq!(nc_close(id), 0);
        values
    }
}
fn main() {
    let paths: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(paths.len(), 4);
    let arrays: Vec<_> = paths.iter().map(|path| load(path)).collect();
    let cells = arrays[0].len();
    assert!(arrays.iter().all(|array| array.len() == cells));
    let counts: Vec<_> = arrays.iter().map(|array| array.iter().zip(&arrays[0])
        .filter(|(a, b)| a.to_bits() != b.to_bits()).count()).collect();
    println!("{{\"n_cells\":{},\"native_vs_native_init_hex_surface_counts\":{:?},\"differences\":[", cells, counts);
    let mut first = true;
    for i in 0..cells {
        if arrays[0][i].to_bits() == arrays[2][i].to_bits() { continue; }
        if !first { println!(","); }
        first = false;
        let values: Vec<_> = arrays.iter().map(|array| array[i]).collect();
        assert!(values.iter().all(|value| value.is_finite()));
        print!("{{\"cell_index_zero_based\":{},\"native_bits\":{},\"init_bits\":{},\"hex_bits\":{},\"surface_record0_bits\":{},\"values\":{:?}}}", i,
               values[0].to_bits(), values[1].to_bits(), values[2].to_bits(), values[3].to_bits(), values);
    }
    println!("]}}");
}
