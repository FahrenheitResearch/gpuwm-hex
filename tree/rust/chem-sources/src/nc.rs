// NetCDF ABI seeded from tree/rust/netcdf-copy (Apache-2.0).
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int},
};
#[link(name = "netcdf")]
extern "C" {
    fn nc_open(p: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_inq_varid(id: c_int, n: *const c_char, v: *mut c_int) -> c_int;
    fn nc_inq_vartype(id: c_int, v: c_int, t: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, v: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_vardimid(id: c_int, v: c_int, d: *mut c_int) -> c_int;
    fn nc_inq_dimlen(id: c_int, d: c_int, n: *mut usize) -> c_int;
    fn nc_inq_dimname(id: c_int, d: c_int, n: *mut c_char) -> c_int;
    fn nc_inq_attlen(id: c_int, v: c_int, n: *const c_char, l: *mut usize) -> c_int;
    fn nc_get_att_text(id: c_int, v: c_int, n: *const c_char, p: *mut c_char) -> c_int;
    fn nc_get_att_double(id: c_int, v: c_int, n: *const c_char, p: *mut f64) -> c_int;
    fn nc_get_vara_text(
        id: c_int,
        v: c_int,
        s: *const usize,
        n: *const usize,
        p: *mut c_char,
    ) -> c_int;
    fn nc_get_vara_double(
        id: c_int,
        v: c_int,
        s: *const usize,
        c: *const usize,
        p: *mut f64,
    ) -> c_int;
    fn nc_create(p: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_def_dim(id: c_int, n: *const c_char, l: usize, d: *mut c_int) -> c_int;
    fn nc_def_var(
        id: c_int,
        n: *const c_char,
        t: c_int,
        nd: c_int,
        ds: *const c_int,
        v: *mut c_int,
    ) -> c_int;
    fn nc_put_att_text(id: c_int, v: c_int, n: *const c_char, l: usize, p: *const c_char) -> c_int;
    #[cfg(test)]
    fn nc_put_att_double(
        id: c_int,
        v: c_int,
        n: *const c_char,
        t: c_int,
        len: usize,
        p: *const f64,
    ) -> c_int;
    #[cfg(test)]
    fn nc_put_var_text(id: c_int, v: c_int, p: *const c_char) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
    fn nc_put_var_float(id: c_int, v: c_int, p: *const f32) -> c_int;
}
fn c(s: &str) -> Result<CString, String> {
    CString::new(s).map_err(|e| e.to_string())
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
pub struct File(i32);
pub struct Field {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub dims: Vec<String>,
}
impl Drop for File {
    fn drop(&mut self) {
        unsafe {
            nc_close(self.0);
        }
    }
}
impl File {
    pub fn open(path: &str) -> Result<Self, String> {
        let mut id = 0;
        check(unsafe { nc_open(c(path)?.as_ptr(), 0, &mut id) })
            .map_err(|e| format!("{path}: {e}"))?;
        Ok(Self(id))
    }
    fn var(&self, name: &str) -> Result<i32, String> {
        let mut v = 0;
        check(unsafe { nc_inq_varid(self.0, c(name)?.as_ptr(), &mut v) })
            .map_err(|e| format!("variable {name}: {e}"))?;
        Ok(v)
    }
    pub fn attr(&self, name: &str, key: &str) -> Option<String> {
        let v = self.var(name).ok()?;
        self.text_attribute(v, key)
    }
    pub fn global_attr(&self, key: &str) -> Option<String> {
        self.text_attribute(-1, key)
    }
    pub fn global_number(&self, key: &str) -> Option<f64> {
        self.number(-1, key)
    }
    fn text_attribute(&self, v: i32, key: &str) -> Option<String> {
        let k = c(key).ok()?;
        let mut len = 0;
        check(unsafe { nc_inq_attlen(self.0, v, k.as_ptr(), &mut len) }).ok()?;
        let mut b = vec![0u8; len];
        check(unsafe { nc_get_att_text(self.0, v, k.as_ptr(), b.as_mut_ptr().cast()) }).ok()?;
        String::from_utf8(b).ok()
    }
    fn number(&self, v: i32, key: &str) -> Option<f64> {
        let k = c(key).ok()?;
        let mut len = 0;
        check(unsafe { nc_inq_attlen(self.0, v, k.as_ptr(), &mut len) }).ok()?;
        if len != 1 {
            return None;
        }
        let mut out = 0.;
        check(unsafe { nc_get_att_double(self.0, v, k.as_ptr(), &mut out) }).ok()?;
        Some(out)
    }
    pub fn read(&self, name: &str, time: usize) -> Result<Field, String> {
        self.read_policy(name, time, None)
    }
    pub fn read_policy(
        &self,
        name: &str,
        time: usize,
        no_source: Option<f64>,
    ) -> Result<Field, String> {
        let v = self.var(name)?;
        let mut nd = 0;
        check(unsafe { nc_inq_varndims(self.0, v, &mut nd) })?;
        let mut ds = vec![0; nd as usize];
        check(unsafe { nc_inq_vardimid(self.0, v, ds.as_mut_ptr()) })?;
        let mut shape = Vec::new();
        let mut dims = Vec::new();
        for &d in &ds {
            let mut len = 0;
            let mut n = [0 as c_char; 257];
            check(unsafe { nc_inq_dimlen(self.0, d, &mut len) })?;
            check(unsafe { nc_inq_dimname(self.0, d, n.as_mut_ptr()) })?;
            shape.push(len);
            dims.push(
                unsafe { CStr::from_ptr(n.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        let mut start = vec![0; shape.len()];
        let mut count = shape.clone();
        let time_axis = dims.iter().position(|d| d.eq_ignore_ascii_case("time"));
        if let Some(i) = time_axis {
            if time >= shape[i] {
                return Err(format!("{name}: time index {time} outside {}", shape[i]));
            }
            start[i] = time;
            count[i] = 1;
        }
        let size = count
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or("field size overflow")?;
        let mut data = vec![0.; size];
        check(unsafe {
            nc_get_vara_double(self.0, v, start.as_ptr(), count.as_ptr(), data.as_mut_ptr())
        })?;
        let mut xtype = 0;
        check(unsafe { nc_inq_vartype(self.0, v, &mut xtype) })?;
        // NetCDF also fills unwritten variables without an explicit _FillValue attribute.
        let default_fill = match xtype {
            1 => Some(-127.),
            3 => Some(-32767.),
            4 => Some(-2147483647.),
            5 => Some(9.96921e36_f32 as f64),
            6 => Some(9.969209968386869e36),
            7 => Some(255.),
            8 => Some(65535.),
            9 => Some(4294967295.),
            10 => Some(-9223372036854775806_i64 as f64),
            11 => Some(18446744073709551614_u64 as f64),
            _ => None,
        };
        let fill = self.number(v, "_FillValue").or(default_fill);
        let missing = self.number(v, "missing_value");
        let scale = self.number(v, "scale_factor").unwrap_or(1.);
        let offset = self.number(v, "add_offset").unwrap_or(0.);
        for x in &mut data {
            if no_source.is_some() && Some(*x) == no_source {
                *x = 0.
            } else if Some(*x) == fill || Some(*x) == missing || !x.is_finite() {
                *x = f64::NAN
            } else {
                *x = *x * scale + offset
            }
        }
        if let Some(i) = time_axis {
            shape.remove(i);
            dims.remove(i);
        }
        Ok(Field { data, shape, dims })
    }
    pub fn record_text(&self, name: &str) -> Result<String, String> {
        self.record_text_at(name, 0)
    }
    pub fn record_text_at(&self, name: &str, index: usize) -> Result<String, String> {
        let v = self.var(name)?;
        let mut nd = 0;
        check(unsafe { nc_inq_varndims(self.0, v, &mut nd) })?;
        if nd != 2 {
            return Err("native timestamp must have Time and StrLen dimensions".into());
        }
        let mut ds = [0; 2];
        check(unsafe { nc_inq_vardimid(self.0, v, ds.as_mut_ptr()) })?;
        let mut width = 0;
        check(unsafe { nc_inq_dimlen(self.0, ds[1], &mut width) })?;
        if width == 0 || width > 1024 {
            return Err("timestamp width outside 1..1024".into());
        }
        let mut b = vec![0u8; width];
        check(unsafe {
            nc_get_vara_text(
                self.0,
                v,
                [index, 0].as_ptr(),
                [1, width].as_ptr(),
                b.as_mut_ptr().cast(),
            )
        })?;
        String::from_utf8(b)
            .map(|s| {
                s.trim_matches(|c: char| c == '\0' || c.is_whitespace())
                    .to_owned()
            })
            .map_err(|e| e.to_string())
    }
    #[cfg(test)]
    pub fn numericattr(&self, v: i32, n: &str, value: f64) -> Result<(), String> {
        check(unsafe { nc_put_att_double(self.0, v, c(n)?.as_ptr(), 5, 1, &value) })
    }
    pub fn close(self) -> Result<(), String> {
        let id = self.0;
        std::mem::forget(self);
        check(unsafe { nc_close(id) })
    }
    pub fn create(path: &str) -> Result<Self, String> {
        let mut id = 0;
        check(unsafe { nc_create(c(path)?.as_ptr(), 0x20 | 4, &mut id) })?;
        Ok(Self(id))
    }
    pub fn dim(&self, n: &str, len: usize) -> Result<i32, String> {
        let mut id = 0;
        check(unsafe { nc_def_dim(self.0, c(n)?.as_ptr(), len, &mut id) })?;
        Ok(id)
    }
    pub fn define(&self, n: &str, ds: &[i32], units: &str) -> Result<i32, String> {
        let mut v = 0;
        check(unsafe {
            nc_def_var(
                self.0,
                c(n)?.as_ptr(),
                5,
                ds.len() as i32,
                ds.as_ptr(),
                &mut v,
            )
        })?;
        self.putattr(v, "units", units)?;
        Ok(v)
    }
    #[cfg(test)]
    pub fn define_text(&self, name: &str, ds: &[i32]) -> Result<i32, String> {
        let mut v = 0;
        check(unsafe {
            nc_def_var(
                self.0,
                c(name)?.as_ptr(),
                2,
                ds.len() as i32,
                ds.as_ptr(),
                &mut v,
            )
        })?;
        Ok(v)
    }
    #[cfg(test)]
    pub fn put_text(&self, v: i32, value: &str) -> Result<(), String> {
        check(unsafe { nc_put_var_text(self.0, v, value.as_ptr().cast()) })
    }
    pub fn putattr(&self, v: i32, n: &str, text: &str) -> Result<(), String> {
        check(unsafe {
            nc_put_att_text(self.0, v, c(n)?.as_ptr(), text.len(), text.as_ptr().cast())
        })
    }
    pub fn end(&self) -> Result<(), String> {
        check(unsafe { nc_enddef(self.0) })
    }
    pub fn put(&self, v: i32, data: &[f32]) -> Result<(), String> {
        check(unsafe { nc_put_var_float(self.0, v, data.as_ptr()) })
    }
}
