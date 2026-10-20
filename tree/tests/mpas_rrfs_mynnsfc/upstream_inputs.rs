//! Read pinned scheme-CI column inputs without a Python field decoder.
use std::ffi::{CStr,c_char};

/// Fill field-major single-precision storage; -1 denotes a malformed fixture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mynn_fixture(path:*const c_char, names:*const *const c_char,
    nf:usize, output:*mut f32, capacity:usize) ->isize {
    let result=std::panic::catch_unwind(|| {
        let path=unsafe{CStr::from_ptr(path)}.to_str().ok()?;
        let input=std::fs::read_to_string(path).ok()?;
        let mut lines=input.lines();let header:Vec<_>=lines.next()?.split_whitespace().collect();
        let records:Vec<Vec<f32>>=lines.filter(|l|!l.trim().is_empty()).map(|line|
            line.split_whitespace().map(str::parse::<f32>).collect::<Result<Vec<_>,_>>()).collect::<Result<_,_>>().ok()?;
        let nc=records.len();if nc>capacity || records.iter().any(|r|r.len()!=header.len()){return None;}
        let out=unsafe{std::slice::from_raw_parts_mut(output,nf*capacity)};out.fill(0.);
        let names=unsafe{std::slice::from_raw_parts(names,nf)};
        for (j,p) in names.iter().enumerate() {
            let name=unsafe{CStr::from_ptr(*p)}.to_str().ok()?;
            let source=match name {
                "u3d"=>"u1d","v3d"=>"v1d","t3d"|"th3d"=>"t1d","qv3d"=>"qv1d",
                "p3d"=>"p1d","rho3d"=>"rho1d","dz8w"=>"dz8w1d","u3d2"=>"u1d2",
                "v3d2"=>"v1d2","dz8w2"=>"dz2w1d","pattern_spp_pbl"=>"rstoch1d",_=>name};
            if let Some(k)=header.iter().position(|h|h.eq_ignore_ascii_case(source)) {
                for i in 0..nc {out[j*capacity+i]=records[i][k];}
            }
        }
        Some(nc as isize)
    });result.ok().flatten().unwrap_or(-1)
}
