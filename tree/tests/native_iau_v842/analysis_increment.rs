//! IAU acceptance input preparation: analysis minus matching native forecast.
//! Registry.xml:3972-4011; no interpolation or missing moisture substitution.
use std::{env, ffi::{CString,c_char,c_int}};
type Result<T> = std::result::Result<T,String>;
#[link(name="netcdf")]
unsafe extern "C" {
    fn nc_open(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_create(path:*const c_char,mode:c_int,id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_inq_dimid(id:c_int,name:*const c_char,dim:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
    fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
    fn nc_inq_vartype(id:c_int,var:c_int,kind:*mut c_int)->c_int;
    fn nc_inq_varndims(id:c_int,var:c_int,rank:*mut c_int)->c_int;
    fn nc_inq_vardimid(id:c_int,var:c_int,dims:*mut c_int)->c_int;
    fn nc_get_vara_float(id:c_int,var:c_int,start:*const usize,count:*const usize,value:*mut f32)->c_int;
    fn nc_get_vara_text(id:c_int,var:c_int,start:*const usize,count:*const usize,value:*mut c_char)->c_int;
    fn nc_get_var_int(id:c_int,var:c_int,value:*mut c_int)->c_int;
    fn nc_def_dim(id:c_int,name:*const c_char,len:usize,dim:*mut c_int)->c_int;
    fn nc_def_var(id:c_int,name:*const c_char,kind:c_int,rank:c_int,dims:*const c_int,var:*mut c_int)->c_int;
    fn nc_enddef(id:c_int)->c_int;
    fn nc_put_vara_float(id:c_int,var:c_int,start:*const usize,count:*const usize,value:*const f32)->c_int;
    fn nc_put_vara_text(id:c_int,var:c_int,start:*const usize,count:*const usize,value:*const c_char)->c_int;
    #[cfg(test)]
    fn nc_put_var_int(id:c_int,var:c_int,value:*const c_int)->c_int;
}
fn c(s:&str)->CString { CString::new(s).unwrap() }
fn check(code:i32)->Result<()> { if code==0 {Ok(())} else {Err(format!("netCDF status {code}"))} }
struct Nc(i32);
impl Drop for Nc { fn drop(&mut self) { if self.0>=0 {unsafe {nc_close(self.0);}} } }
impl Nc {
    fn finish(mut self)->Result<()> { let id=self.0; self.0=-1; unsafe {check(nc_close(id))} }
    fn open(path:&str)->Result<Self> {
        let mut id=0; unsafe {check(nc_open(c(path).as_ptr(),0,&mut id))?;} Ok(Self(id))
    }
    fn dim(&self,name:&str)->Result<(i32,usize)> {
        let (mut id,mut len)=(0,0);
        unsafe {check(nc_inq_dimid(self.0,c(name).as_ptr(),&mut id))?;
            check(nc_inq_dimlen(self.0,id,&mut len))?;} Ok((id,len))
    }
    fn var(&self,name:&str)->Result<i32> {
        let mut id=0; unsafe {check(nc_inq_varid(self.0,c(name).as_ptr(),&mut id))?;} Ok(id)
    }
    fn schema(&self,var:i32,kind:i32,dims:&[i32])->Result<()> {
        let (mut k,mut rank)=(0,0);
        unsafe {check(nc_inq_vartype(self.0,var,&mut k))?;
            check(nc_inq_varndims(self.0,var,&mut rank))?;}
        if k!=kind || rank!=dims.len() as i32 {return Err("native field type/rank differs".into());}
        let mut actual=vec![0;dims.len()];
        unsafe {check(nc_inq_vardimid(self.0,var,actual.as_mut_ptr()))?;}
        if actual!=dims {return Err("native field axes differ".into());} Ok(())
    }
    fn text(&self,name:&str,timed:bool)->Result<Vec<u8>> {
        let record=if timed {Some(self.dim("Time")?.1.checked_sub(1).ok_or("empty native time axis")?)} else {None};
        self.text_at(name,record)
    }
    fn text_at(&self,name:&str,record:Option<usize>)->Result<Vec<u8>> {
        let (str_id,width)=self.dim("StrLen")?;
        let var=self.var(name)?;
        let (start,count,dims)=if let Some(record)=record {
            let (time_id,n)=self.dim("Time")?;
            if record>=n {return Err("native time record out of range".into());}
            (vec![record,0],vec![1,width],vec![time_id,str_id])
        } else {(vec![0],vec![width],vec![str_id])};
        self.schema(var,2,&dims)?;
        let mut value=vec![0;width];
        unsafe {check(nc_get_vara_text(self.0,var,start.as_ptr(),count.as_ptr(),value.as_mut_ptr().cast()))?;} Ok(value)
    }
    fn ids(&self,name:&str,axis:&str)->Result<Vec<i32>> {
        let (dim,n)=self.dim(axis)?; let var=self.var(name)?;
        self.schema(var,4,&[dim])?;
        let mut value=vec![0;n]; unsafe {check(nc_get_var_int(self.0,var,value.as_mut_ptr()))?;} Ok(value)
    }
    fn field(&self,name:&str,entity:&str)->Result<Vec<f32>> {
        self.field_at(name,entity,self.dim("Time")?.1.checked_sub(1).ok_or("empty native time axis")?)
    }
    fn field_at(&self,name:&str,entity:&str,record:usize)->Result<Vec<f32>> {
        let (t,nt)=self.dim("Time")?; let (e,ne)=self.dim(entity)?;
        let (l,nl)=self.dim("nVertLevels")?;
        if record>=nt {return Err("native time record out of range".into());}
        let var=self.var(name)?; self.schema(var,5,&[t,e,l])?;
        let mut value=vec![0.0;ne.checked_mul(nl).ok_or("field size overflow")?];
        unsafe {check(nc_get_vara_float(self.0,var,[record,0,0].as_ptr(),[1,ne,nl].as_ptr(),value.as_mut_ptr()))?;} Ok(value)
    }
}
fn timestamp(raw:&[u8])->Result<i64> {
    let s=std::str::from_utf8(raw.get(..19).ok_or("short native timestamp")?).map_err(|_|"timestamp bytes")?;
    if !s.is_ascii() || &s[4..5]!="-" || &s[7..8]!="-" || &s[10..11]!="_"
        || &s[13..14]!=":" || &s[16..17]!=":" {return Err("native timestamp format".into());}
    let number=|a,b| s[a..b].parse::<i64>().map_err(|_|"native timestamp digits".to_string());
    let (mut y,m,d,h,n,sec)=(number(0,4)?,number(5,7)?,number(8,10)?,number(11,13)?,number(14,16)?,number(17,19)?);
    let leap=y%4==0 && (y%100!=0 || y%400==0);
    let days=[31,if leap {29} else {28},31,30,31,30,31,31,30,31,30,31];
    if y<1 || !(1..=12).contains(&m) || d<1 || d>days[(m-1) as usize]
        || !(0..24).contains(&h) || !(0..60).contains(&n) || !(0..60).contains(&sec) {
        return Err("native timestamp range".into());
    }
    y-=i64::from(m<=2); let era=y.div_euclid(400); let yr=y-era*400;
    let adjusted=m+if m>2 {-3} else {9};
    let day=era*146097+yr*365+yr/4-yr/100+(153*adjusted+2)/5+d-1;
    Ok(day*86400+h*3600+n*60+sec)
}
fn validate_times(a:&[u8],b:&[u8],start:&[u8],age:i64)->Result<()> {
    if age<=0 || timestamp(a)?!=timestamp(b)? {return Err("analysis/background valid times differ or invalid age".into());}
    if timestamp(b)?-timestamp(start)?!=age {return Err("native background forecast age differs".into());} Ok(())
}
fn increments(analysis:&str,background:&str,output:&str,age:i64)->Result<()> {
    let a=Nc::open(analysis)?; let b=Nc::open(background)?;
    let at=a.text("xtime",true)?; let valid=timestamp(&at)?;
    let mut matching=None;
    for record in 0..b.dim("Time")?.1 {
        let time=b.text_at("xtime",Some(record))?;
        if timestamp(&time)?==valid {
            if matching.is_some() {return Err("duplicate matching background valid time".into());}
            matching=Some((record,time));
        }
    }
    let (background_record,bt)=matching.ok_or("analysis/background valid times differ")?;
    validate_times(&at,&bt,&b.text("initial_time",false)?,age)?;
    for axis in ["nCells","nEdges","nVertLevels","StrLen"] {
        if a.dim(axis)?.1!=b.dim(axis)?.1 {return Err(format!("analysis/background {axis} differs"));}
    }
    for (name,axis) in [("indexToCellID","nCells"),("indexToEdgeID","nEdges")] {
        if a.ids(name,axis)?!=b.ids(name,axis)? {return Err(format!("analysis/background {name} ordering differs"));}
    }
    let mut fields=Vec::new();
    for (i,name) in ["u","rho","theta","qv","qc","qr","qi","qs","qg"].iter().enumerate() {
        // Native background allocation defines the active moisture fields.
        if i>=4 && b.var(name).is_err() {continue;}
        let entity=if i==0 {"nEdges"} else {"nCells"};
        let av=a.field(name,entity)?; let bv=b.field_at(name,entity,background_record)?;
        if av.len()!=bv.len() {return Err(format!("analysis/background {name} shape differs"));}
        let values=av.into_iter().zip(bv).map(|(av,bv)|av-bv).collect::<Vec<f32>>();
        fields.push((format!("{name}_amb"),entity,values));
    }
    // All input/schema/time checks precede NC_NOCLOBBER creation.
    let mut id=0; unsafe {check(nc_create(c(output).as_ptr(),4,&mut id))?;} let out=Nc(id);
    let mut dims=Vec::new();
    for name in ["Time","nEdges","nCells","nVertLevels","StrLen"] {
        let len=if name=="Time" {0} else {a.dim(name)?.1}; let mut dim=0;
        unsafe {check(nc_def_dim(out.0,c(name).as_ptr(),len,&mut dim))?;} dims.push(dim);
    }
    let mut variables=Vec::new();
    for (name,entity,_) in &fields {
        let mut var=0; let e=if *entity=="nEdges" {1} else {2};
        unsafe {check(nc_def_var(out.0,c(name).as_ptr(),5,3,[dims[0],dims[e],dims[3]].as_ptr(),&mut var))?;} variables.push(var);
    }
    let mut time=0;
    unsafe {check(nc_def_var(out.0,c("xtime").as_ptr(),2,2,[dims[0],dims[4]].as_ptr(),&mut time))?;
        check(nc_enddef(out.0))?;}
    for ((_,entity,values),var) in fields.iter().zip(variables) {
        let count=[1,a.dim(entity)?.1,a.dim("nVertLevels")?.1];
        unsafe {check(nc_put_vara_float(out.0,var,[0,0,0].as_ptr(),count.as_ptr(),values.as_ptr()))?;}
    }
    unsafe {check(nc_put_vara_text(out.0,time,[0,0].as_ptr(),[1,at.len()].as_ptr(),at.as_ptr().cast()))?;}
    out.finish()?;
    println!("wrote {} increment fields at {}, native background age {} seconds, record {}",fields.len(),std::str::from_utf8(&at[..19]).unwrap(),age,background_record);
    Ok(())
}
fn main() {
    let args:Vec<_>=env::args().collect();
    let result=if args.len()==5 {
        args[4].parse().map_err(|_|"invalid background age".into()).and_then(|age| increments(&args[1],&args[2],&args[3],age))
    } else {Err("usage: analysis_increment ANALYSIS BACKGROUND OUT AGE_SECONDS".into())};
    if let Err(error)=result {eprintln!("{error}");std::process::exit(1);}
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(path:&str,analysis:bool,moist:bool,id_offset:i32)->Result<()> {
        let mut id=0; unsafe {check(nc_create(c(path).as_ptr(),4,&mut id))?;} let file=Nc(id);
        let mut dims=[0;5];
        for (i,(name,len)) in [("Time",0),("nCells",2),("nEdges",3),("nVertLevels",2),("StrLen",64)].iter().enumerate() {
            unsafe {check(nc_def_dim(id,c(name).as_ptr(),*len,&mut dims[i]))?;}
        }
        let mut time=0; let mut initial=0; let mut ids=[0;2]; let mut fields=Vec::new();
        unsafe {
            check(nc_def_var(id,c("xtime").as_ptr(),2,2,[dims[0],dims[4]].as_ptr(),&mut time))?;
            check(nc_def_var(id,c("initial_time").as_ptr(),2,1,[dims[4]].as_ptr(),&mut initial))?;
            for (i,name) in ["indexToCellID","indexToEdgeID"].iter().enumerate() {
                check(nc_def_var(id,c(name).as_ptr(),4,1,[dims[i+1]].as_ptr(),&mut ids[i]))?;
            }
            for (i,name) in ["u","rho","theta","qv","qc","qr","qi","qs","qg"].iter().enumerate() {
                if !moist && i>=4 {continue;}
                let mut var=0; let axis=if i==0 {dims[2]} else {dims[1]};
                check(nc_def_var(id,c(name).as_ptr(),5,3,[dims[0],axis,dims[3]].as_ptr(),&mut var))?;
                fields.push((var,if i==0 {3} else {2},i));
            }
            check(nc_enddef(id))?;
            check(nc_put_var_int(id,ids[0],[1+id_offset,2].as_ptr()))?;
            check(nc_put_var_int(id,ids[1],[3,4,5].as_ptr()))?;
            let mut start=vec![b' ';64]; start[..19].copy_from_slice(b"2000-02-29_21:00:00");
            check(nc_put_vara_text(id,initial,[0].as_ptr(),[64].as_ptr(),start.as_ptr().cast()))?;
            for record in 0..if analysis {2} else {3} {
                let mut valid=start.clone(); if record==1 {valid[..19].copy_from_slice(b"2000-03-01_03:00:00");}
                if record==2 {valid[..19].copy_from_slice(b"2000-03-01_09:00:00");}
                check(nc_put_vara_text(id,time,[record,0].as_ptr(),[1,64].as_ptr(),valid.as_ptr().cast()))?;
                for &(var,entities,i) in &fields {
                    let values=(0..entities*2).map(|j| (j+i+record) as f32+if analysis {0.25} else {0.0}).collect::<Vec<_>>();
                    check(nc_put_vara_float(id,var,[record,0,0].as_ptr(),[1,entities,2].as_ptr(),values.as_ptr()))?;
                }
            }
        }
        file.finish()
    }
    #[test]
    fn native_netcdf_fields_last_record_subtraction_and_atomic_refusals() {
        let root=std::path::PathBuf::from(env::var("IAU_INCREMENT_TEST_DIR").expect("test scratch directory required"))
            .join(format!("analysis-increment-{}",std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let paths=(0..6).map(|i| root.join(format!("{i}.nc")).to_str().unwrap().to_string()).collect::<Vec<_>>();
        fixture(&paths[0],true,true,0).unwrap(); fixture(&paths[1],false,true,0).unwrap();
        increments(&paths[0],&paths[1],&paths[2],21600).unwrap();
        let out=Nc::open(&paths[2]).unwrap();
        for (i,name) in ["u","rho","theta","qv","qc","qr","qi","qs","qg"].iter().enumerate() {
            let values=out.field(&format!("{name}_amb"),if i==0 {"nEdges"} else {"nCells"}).unwrap();
            assert!(values.iter().all(|v|v.to_bits()==0.25f32.to_bits()));
        }
        assert_eq!(&out.text("xtime",true).unwrap()[..19],b"2000-03-01_03:00:00"); drop(out);
        let original=std::fs::read(&paths[2]).unwrap();
        assert!(increments(&paths[0],&paths[1],&paths[2],21600).is_err());
        assert_eq!(original,std::fs::read(&paths[2]).unwrap());
        fixture(&paths[3],true,true,1).unwrap();
        assert!(increments(&paths[3],&paths[1],&paths[5],21600).is_err());
        fixture(&paths[4],true,false,0).unwrap();
        assert!(increments(&paths[4],&paths[1],&paths[5],21600).is_err());
        assert!(increments(&paths[0],&paths[1],&paths[5],1).is_err());
        assert!(!std::path::Path::new(&paths[5]).exists());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn matching_valid_time_and_forecast_age_cross_month_and_year() {
        for (valid,start) in [("2020-03-01_03:00:00","2020-02-29_21:00:00"),
                              ("2001-01-01_03:00:00","2000-12-31_21:00:00")] {
            assert!(validate_times(valid.as_bytes(),valid.as_bytes(),start.as_bytes(),21600).is_ok());
            assert!(validate_times(valid.as_bytes(),start.as_bytes(),start.as_bytes(),21600).is_err());
            assert!(validate_times(valid.as_bytes(),valid.as_bytes(),start.as_bytes(),1).is_err());
        }
        assert!(timestamp(b"2001-02-29_03:00:00").is_err());
        assert!(timestamp(b"2000-01-01_24:00:00").is_err());
    }
}
