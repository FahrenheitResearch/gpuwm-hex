//! Small checked NetCDF transport, seeded from the in-tree rw-surface reader.
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int},
    path::Path,
    sync::{Mutex, MutexGuard},
};

static NETCDF: Mutex<()> = Mutex::new(());
#[link(name = "netcdf")]
extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, var: c_int, rank: *mut c_int) -> c_int;
    fn nc_inq_vardimid(id: c_int, var: c_int, dims: *mut c_int) -> c_int;
    fn nc_inq_dim(id: c_int, dim: c_int, name: *mut c_char, len: *mut usize) -> c_int;
    fn nc_get_var_double(id: c_int, var: c_int, data: *mut f64) -> c_int;
    fn nc_get_vara_double(id: c_int, var: c_int, start: *const usize, count: *const usize, data: *mut f64) -> c_int;
    fn nc_get_var_text(id: c_int, var: c_int, data: *mut c_char) -> c_int;
    fn nc_inq_attlen(id: c_int, var: c_int, name: *const c_char, len: *mut usize) -> c_int;
    fn nc_get_att_text(id: c_int, var: c_int, name: *const c_char, data: *mut c_char) -> c_int;
    fn nc_get_att_double(id: c_int, var: c_int, name: *const c_char, data: *mut f64) -> c_int;
}

fn check(code: i32) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(unsafe { CStr::from_ptr(nc_strerror(code)) }
            .to_string_lossy()
            .into_owned())
    }
}
pub struct File {
    id: i32,
    _lock: MutexGuard<'static, ()>,
}
pub struct Array {
    pub values: Vec<f64>,
    pub shape: Vec<usize>,
    pub dims: Vec<String>,
    pub units: String,
}
impl File {
    pub fn clock(&self) -> Result<String, String> {
        let mut var = 0;
        let mut rank = 0;
        check(unsafe { nc_inq_varid(self.id, b"xtime\0".as_ptr().cast(), &mut var) })?;
        check(unsafe { nc_inq_varndims(self.id, var, &mut rank) })?;
        if rank != 2 {
            return Err("xtime must have Time,StrLen dimensions".into());
        }
        let mut ids = [0; 2];
        check(unsafe { nc_inq_vardimid(self.id, var, ids.as_mut_ptr()) })?;
        let mut size = 1usize;
        for (at, dim) in ids.iter().enumerate() {
            let mut len = 0;
            let mut label = [0i8; 257];
            check(unsafe { nc_inq_dim(self.id, *dim, label.as_mut_ptr(), &mut len) })?;
            if (at == 0 && len != 1) || len == 0 || len > 256 {
                return Err("xtime must hold exactly one bounded timestamp".into());
            }
            size *= len;
        }
        let mut bytes = vec![0u8; size];
        check(unsafe { nc_get_var_text(self.id, var, bytes.as_mut_ptr().cast()) })?;
        String::from_utf8(bytes)
            .map(|s| s.trim_end_matches([' ', '\0']).to_string())
            .map_err(|e| e.to_string())
    }
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let lock = NETCDF.lock().map_err(|e| e.to_string())?;
        let name =
            CString::new(path.as_ref().to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let mut id = 0;
        check(unsafe { nc_open(name.as_ptr(), 0, &mut id) })
            .map_err(|e| format!("{}: {e}", path.as_ref().display()))?;
        Ok(Self { id, _lock: lock })
    }
    pub fn attr(&self, var: i32, name: &str) -> Result<String, String> {
        let key = CString::new(name).map_err(|e| e.to_string())?;
        let mut len = 0;
        check(unsafe { nc_inq_attlen(self.id, var, key.as_ptr(), &mut len) })?;
        let mut bytes = vec![0u8; len];
        check(unsafe { nc_get_att_text(self.id, var, key.as_ptr(), bytes.as_mut_ptr().cast()) })?;
        String::from_utf8(bytes).map_err(|e| e.to_string())
    }
    fn number(&self, var: i32, name: &str) -> Result<Option<f64>, String> {
        let key = CString::new(name).map_err(|e| e.to_string())?;
        let mut len = 0;
        let code = unsafe { nc_inq_attlen(self.id, var, key.as_ptr(), &mut len) };
        if code == -43 {
            return Ok(None);
        } // NC_ENOTATT
        check(code)?;
        if len != 1 {
            return Err(format!("{name} must be scalar"));
        }
        let mut value = 0.;
        check(unsafe { nc_get_att_double(self.id, var, key.as_ptr(), &mut value) })?;
        Ok(Some(value))
    }
    pub fn read(&self, name: &str) -> Result<Array, String> {
        self.read_selection(name, None)
    }
    pub fn read_selection(&self, name: &str, selection: Option<(usize, usize)>) -> Result<Array, String> {
        let key = CString::new(name).map_err(|e| e.to_string())?;
        let mut var = 0;
        check(unsafe { nc_inq_varid(self.id, key.as_ptr(), &mut var) })
            .map_err(|e| format!("{name}: {e}"))?;
        let mut rank = 0;
        check(unsafe { nc_inq_varndims(self.id, var, &mut rank) })?;
        if !(1..=6).contains(&rank) {
            return Err(format!("{name}: unsupported rank {rank}"));
        }
        let mut ids = vec![0; rank as usize];
        check(unsafe { nc_inq_vardimid(self.id, var, ids.as_mut_ptr()) })?;
        let mut shape = Vec::new();
        let mut dims = Vec::new();
        for dim in ids {
            let mut len = 0;
            let mut label = [0i8; 257];
            check(unsafe { nc_inq_dim(self.id, dim, label.as_mut_ptr(), &mut len) })?;
            shape.push(len);
            dims.push(
                unsafe { CStr::from_ptr(label.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        let mut start = vec![0; shape.len()];
        if let Some((first, count)) = selection {
            if count == 0 || first.checked_add(count).map_or(true, |end| end > shape[0]) {
                return Err(format!("{name}: invalid leading-dimension selection"));
            }
            start[0] = first;
            shape[0] = count;
        }
        let n = shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or("array extent overflow")?;
        if n == 0 || n > 500_000_000 {
            return Err(format!("{name}: invalid extent {n}"));
        }
        let mut values = vec![0.; n];
        check(unsafe {
            if selection.is_some() {
                nc_get_vara_double(self.id, var, start.as_ptr(), shape.as_ptr(), values.as_mut_ptr())
            } else {
                nc_get_var_double(self.id, var, values.as_mut_ptr())
            }
        })?;
        let fill = self.number(var, "_FillValue")?;
        let missing = self.number(var, "missing_value")?;
        let scale = self.number(var, "scale_factor")?.unwrap_or(1.);
        let offset = self.number(var, "add_offset")?.unwrap_or(0.);
        for v in &mut values {
            *v = if fill == Some(*v) || missing == Some(*v) || v.abs() > 9e35 {
                f64::NAN
            } else {
                *v * scale + offset
            };
        }
        Ok(Array {
            values,
            shape,
            dims,
            units: self.attr(var, "units").unwrap_or_default(),
        })
    }
}
impl Drop for File {
    fn drop(&mut self) {
        unsafe {
            nc_close(self.id);
        }
    }
}

impl Array {
    pub fn layout(&self, dims: &[&str], shape: &[usize]) -> Result<(), String> {
        if self.dims.iter().map(String::as_str).collect::<Vec<_>>() != dims || self.shape != shape {
            return Err(format!(
                "dimensions {:?} {:?}, expected {dims:?} {shape:?}",
                self.dims, self.shape
            ));
        }
        Ok(())
    }
    pub fn units(&self, allowed: &[&str]) -> Result<(), String> {
        if allowed.contains(&self.units.as_str()) {
            Ok(())
        } else {
            Err(format!("units {:?}, expected {allowed:?}", self.units))
        }
    }
}
