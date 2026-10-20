//! Paired initialization evaluation. All field reads, interpolation and scores
//! run in Rust. Input paths and publication selectors are campaign metadata.
use grib_core::grib2::{grid_latlon, unpack_message, Grib2File};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, fs, io::Read};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const LEVELS: [f64; 3] = [85000., 50000., 25000.];
const NAMES: [&str; 4] = ["z", "t", "u", "v"];

fn digest(path: &str) -> Result<String> {
    let mut f=fs::File::open(path)?; let mut h=Sha256::new(); let mut b=vec![0;1<<20];
    loop {let n=f.read(&mut b)?; if n==0 {break} h.update(&b[..n]);}
    Ok(format!("{:x}",h.finalize()))
}
fn read(f: &netcrust::File, n: &str) -> Result<Vec<f64>> { Ok(f.read_f64_first_record_or_all(n)?) }
fn at_pressure(p: &[f64], x: &[f64], target:f64) -> f64 {
    if p.iter().chain(x).any(|v| !v.is_finite()) {return f64::NAN}
    let mut matches=0;let mut value=f64::NAN;
    for k in 0..p.len() {if p[k]==target {matches+=1;value=x[k];}}
    for k in 1..p.len() {
        if (p[k-1]>target && p[k]<target) || (p[k-1]<target && p[k]>target) {
            matches+=1;
            let w=(target.ln()-p[k-1].ln())/(p[k].ln()-p[k-1].ln());
            value=if p[k-1]>p[k] {x[k-1]+w*(x[k]-x[k-1])} else {f64::NAN};
        }
    }
    if matches==1 {value} else {f64::NAN}
}
struct Frame { fields:BTreeMap<String,Vec<f64>>, health:Value }
fn frame(path:&str, init:&str, n:usize) -> Result<Frame> {
    let f=netcrust::File::open(path)?; let g=netcrust::File::open(init)?;
    let p=read(&f,"pressure")?; let th=read(&f,"theta")?;
    let u=read(&f,"u_zonal")?; let v=read(&f,"v_meridional")?;
    let zg=read(&g,"zgrid")?; let nz=p.len()/n;
    if nz<2 || p.len()!=n*nz || [th.len(),u.len(),v.len()].iter().any(|&m|m!=p.len()) || zg.len()!=n*(nz+1) {
        return Err("history and initial vertical shapes differ".into());
    }
    if p.iter().any(|x| !x.is_finite()||*x<=0.) || th.iter().any(|x| !x.is_finite()||*x<=0.) || u.iter().chain(&v).any(|x| !x.is_finite()) {
        return Err("history has nonfinite winds or nonpositive/nonfinite pressure or theta".into());
    }
    let mut health=BTreeMap::new();
    for (key,x) in [("pressure",&p),("theta",&th),("u_zonal",&u),("v_meridional",&v)] {
        health.insert(key,json!({"count":x.len(),"finite":true,"min":x.iter().copied().fold(f64::INFINITY,f64::min),"max":x.iter().copied().fold(f64::NEG_INFINITY,f64::max)}));
    }
    let mut fields=BTreeMap::new();
    let mut reversed_columns=0;let mut max_reversed_dp=0_f64;let mut inversion_examples=Vec::new();
    for level in LEVELS {for key in NAMES {fields.insert(format!("{key}{}",level as i32/100),vec![f64::NAN;n]);}}
    for c in 0..n {
        let pp=&p[c*nz..(c+1)*nz];
        if pp.windows(2).any(|a|a[0]<=a[1]) {
            reversed_columns+=1;
            for (k,a) in pp.windows(2).enumerate().filter(|(_,a)|a[0]<=a[1]) {
                max_reversed_dp=max_reversed_dp.max(a[1]-a[0]);
                if inversion_examples.len()<10 {inversion_examples.push(json!({"cell":c,"lower_level":k,"lower_pressure":a[0],"upper_pressure":a[1]}));}
            }
        }
        let temp:Vec<f64>=(0..nz).map(|k|th[c*nz+k]*(pp[k]/100000.).powf(287.04/1004.5)).collect();
        let z:Vec<f64>=(0..nz).map(|k|0.5*(zg[c*(nz+1)+k]+zg[c*(nz+1)+k+1])).collect();
        for level in LEVELS {
            for (key,values) in [("z",z.as_slice()),("t",temp.as_slice()),("u",&u[c*nz..(c+1)*nz]),("v",&v[c*nz..(c+1)*nz])] {
                fields.get_mut(&format!("{key}{}",level as i32/100)).unwrap()[c]=at_pressure(pp,values,level);
            }
        }
    }
    Ok(Frame{fields,health:json!({"fields":health,"non_monotone_pressure_columns":reversed_columns,"max_reversed_pressure_pa":max_reversed_dp,"inversion_examples":inversion_examples})})
}
struct Grid { nx:usize,ny:usize,lat:f64,lon:f64,dy:f64,dx:f64,values:Vec<f64> }
impl Grid {
    fn open(path:&str) -> Result<Self> {
        let bytes=fs::read(path)?; let file=Grib2File::from_bytes(&bytes)?;
        if file.messages.len()!=1 {return Err("reference GRIB must contain exactly one selected field".into())}
        let m=&file.messages[0]; if m.grid.template!=0 {return Err("reference is not regular latitude/longitude".into())}
        let values=unpack_message(m)?; let (lat,lon)=grid_latlon(&m.grid)?;
        let nx=m.grid.nx as usize; let ny=m.grid.ny as usize;
        if nx<2 || ny<2 || values.len()!=nx*ny {return Err("reference grid shape mismatch".into())}
        let dx=(lon[1]-lon[0]+180.).rem_euclid(360.)-180.; let dy=lat[nx]-lat[0];
        if ((nx as f64)*dx.abs()-360.).abs()>1e-5 {return Err("reference longitude grid does not cover globe".into())}
        Ok(Self{nx,ny,lat:lat[0],lon:lon[0],dx,dy,values})
    }
    fn sample(&self,lat:f64,lon:f64)->f64 {
        let y=(lat-self.lat)/self.dy;
        if y < -1e-6 || y > self.ny as f64-1.+1e-6 {return f64::NAN}
        let y=y.clamp(0.,self.ny as f64-1.);let j=(y.floor() as usize).min(self.ny-2);let fy=y-j as f64;
        let x=((lon-self.lon)/self.dx).rem_euclid(self.nx as f64);let i=x.floor() as usize;let fx=x-i as f64;
        let a=[self.values[j*self.nx+i],self.values[j*self.nx+(i+1)%self.nx],self.values[(j+1)*self.nx+i],self.values[(j+1)*self.nx+(i+1)%self.nx]];
        if a.iter().any(|v| !v.is_finite()) {return f64::NAN}
        (1.-fy)*((1.-fx)*a[0]+fx*a[1])+fy*((1.-fx)*a[2]+fx*a[3])
    }
}
#[derive(Default)]
struct Stat { n:usize,w:f64,e:f64,e2:f64 }
impl Stat {
    fn add(&mut self,e:f64,w:f64) {self.n+=1;self.w+=w;self.e+=w*e;self.e2+=w*e*e;}
    fn value(&self)->Value {if self.w==0. {json!({"count":0})} else {json!({"count":self.n,"bias":self.e/self.w,"rmse":(self.e2/self.w).sqrt(),"weight":self.w})}}
}
#[derive(Clone)]
struct Obs { station:String,valid:String,lat:f64,lon:f64,level:i32,z:f64,t:f64,u:f64,v:f64 }
fn number(s:&str,a:usize,b:usize)->Option<f64> {s.get(a..b)?.trim().parse().ok().filter(|v:&f64|*v!=-9999. && *v!=-8888.)}
fn observations(paths:&Value, valid_times:&[String])->Result<Vec<Obs>> {
    let mut out=Vec::new();
    for path in paths.as_array().ok_or("sonde_files must be an array")? {
        let text=fs::read_to_string(path.as_str().ok_or("sonde path must be text")?)?;
        let mut header:Option<(String,String,f64,f64)>=None;
        for line in text.lines() {
            if line.starts_with('#') {
                header=None;
                let fields:Vec<_>=line.split_whitespace().collect();
                if fields.len()<7 {continue}
                let year:usize=fields[1].parse()?;let month:usize=fields[2].parse()?;let day:usize=fields[3].parse()?;let hour:usize=fields[4].parse()?;
                let date=format!("{year:04}-{month:02}-{day:02}_{hour:02}:00:00");
                if !valid_times.iter().any(|x|x==&date) {continue}
                if let (Some(lat),Some(lon))=(number(line,55,62),number(line,63,71)) {
                    if lat.abs()<=900000. && lon.abs()<=1800000. {header=Some((fields[0][1..].into(),date,lat/10000.,lon/10000.));}
                }
            } else if let Some((station,valid,lat,lon))=&header {
                let Some(p)=number(line,9,15) else {continue};
                if !LEVELS.contains(&p) {continue}
                let z=number(line,16,21).unwrap_or(f64::NAN);
                let t=number(line,22,27).map(|v|v*0.1+273.15).unwrap_or(f64::NAN);
                let direction=number(line,40,45); let speed=number(line,46,51);
                let (u,v)=match (direction,speed) { (Some(d),Some(s)) if (0. ..=360.).contains(&d) && s>=0. =>(-s*0.1*d.to_radians().sin(),-s*0.1*d.to_radians().cos()),_=>(f64::NAN,f64::NAN)};
                out.push(Obs{station:station.clone(),valid:valid.clone(),lat:*lat,lon:*lon,level:p as i32/100,z,t,u,v});
            }
        }
    }
    // Some records duplicate a standard level. Keep the first usable value of
    // each variable in the published record order, filling only missing values.
    let mut unique:BTreeMap<(String,String,i32),Obs>=BTreeMap::new();
    for o in out {
        unique.entry((o.station.clone(),o.valid.clone(),o.level)).and_modify(|old|{
            if !old.z.is_finite() {old.z=o.z} if !old.t.is_finite() {old.t=o.t}
            if !old.u.is_finite() {old.u=o.u;old.v=o.v}
        }).or_insert(o);
    }
    Ok(unique.into_values().collect())
}
fn nearest(lat:f64,lon:f64,la:&[f64],lo:&[f64])->usize {
    let a=lat.to_radians();let b=lon.to_radians();let q=[a.cos()*b.cos(),a.cos()*b.sin(),a.sin()];
    let mut best=(0,f64::NEG_INFINITY);
    for c in 0..la.len() {let dot=q[0]*la[c].cos()*lo[c].cos()+q[1]*la[c].cos()*lo[c].sin()+q[2]*la[c].sin();if dot>best.1 {best=(c,dot)}}
    best.0
}
fn vector_rows(rows:&mut Vec<Value>) {
    let mut additions=Vec::new();
    for u in rows.iter().filter(|r|r["field"].as_str().unwrap_or("").starts_with('u')) {
        let field=u["field"].as_str().unwrap();let vfield=format!("v{}",&field[1..]);
        if let Some(v)=rows.iter().find(|v|v["field"]==vfield && ["lead_hours","arm","reference","region"].iter().all(|k|u[k]==v[k])) {
            if u["stats"]["count"]!=v["stats"]["count"] || u["stats"]["weight"]!=v["stats"]["weight"] {continue}
            if let (Some(ur),Some(vr))=(u["stats"]["rmse"].as_f64(),v["stats"]["rmse"].as_f64()) {
                let mut row=u.clone();row["field"]=json!(format!("wind{}",&field[1..]));
                row["stats"]=json!({"count":u["stats"]["count"],"weight":u["stats"]["weight"],"rmse":(ur*ur+vr*vr).sqrt(),"bias_u":u["stats"]["bias"],"bias_v":v["stats"]["bias"]});
                additions.push(row);
            }
        }
    }
    rows.extend(additions);
}
fn main()->Result<()> {
    let args:Vec<String>=std::env::args().collect();
    if args.get(1).map(String::as_str)==Some("audit") {
        let path=args.get(2).ok_or("audit needs an init path")?;
        let file=netcrust::File::open(path)?;let mut fields=BTreeMap::new();
        for key in ["skintemp","sst","snow","snowh","xice","seaice","tslb","smois","sh2o","qc","qr","qv","tke","w","surface_pressure","theta","u","rho","ter"] {
            if file.variable(key).is_none() {continue}
            let v=read(&file,key)?;let finite=v.iter().filter(|x|x.is_finite()).count();
            let zero=v.iter().filter(|&&x|x==0.).count();let min=v.iter().copied().fold(f64::INFINITY,f64::min);let max=v.iter().copied().fold(f64::NEG_INFINITY,f64::max);
            let mean=v.iter().filter(|x|x.is_finite()).sum::<f64>()/finite as f64;
            fields.insert(key,json!({"count":v.len(),"finite":finite,"zero":zero,"min":min,"max":max,"mean":mean}));
        }
        println!("{}",serde_json::to_string_pretty(&json!({"path":path,"sha256":digest(path)?,"fields":fields}))?);return Ok(())
    }
    if args.get(1).map(String::as_str)==Some("selfcheck") {
        let x=at_pressure(&[100000.,50000.],&[10.,20.],(100000_f64*50000.).sqrt());
        assert!((x-15.).abs()<1e-12);assert!(at_pressure(&[100000.,50000.],&[10.,20.],25000.).is_nan());
        assert!(at_pressure(&[100000.,49000.,51000.,20000.],&[0.,1.,2.,3.],50000.).is_nan());
        assert!(at_pressure(&[100000.,50000.,25000.,26000.,1000.],&[0.,1.,2.,3.,4.],85000.).is_finite());
        assert!(at_pressure(&[100000.,50000.,25000.,26000.,1000.],&[0.,1.,2.,3.,4.],25000.).is_nan());
        let g=Grid{nx:4,ny:2,lat:90.,lon:180.,dx:90.,dy:-180.,values:vec![0.,1.,2.,3.,10.,11.,12.,13.]};
        assert!((g.sample(0.,225.)-5.5).abs()<1e-12);assert!((g.sample(0.,135.)-6.5).abs()<1e-12);
        let mut s=Stat::default();s.add(1.,1.);s.add(-1.,1.);assert_eq!(s.value()["rmse"],1.);assert_eq!(s.value()["bias"],0.);
        println!("pressure brackets, longitude origin and seam, weighted bias/RMSE: passed");return Ok(())
    }
    let path=args.get(1).ok_or("usage: initial-source-skill CONFIG.json | selfcheck")?;
    let cfg:Value=serde_json::from_str(&fs::read_to_string(path)?)?;
    let gridpath=cfg["grid"].as_str().ok_or("grid path missing")?;
    let grid=netcrust::File::open(gridpath)?;let la=read(&grid,"latCell")?;let lo=read(&grid,"lonCell")?;let area=read(&grid,"areaCell")?;let n=la.len();
    if lo.len()!=n || area.len()!=n || area.iter().any(|x| !x.is_finite()||*x<=0.) {return Err("invalid cell area or geometry".into())}
    let valid_times:Vec<String>=cfg["leads"].as_array().ok_or("leads missing")?.iter().map(|x|x["valid"].as_str().unwrap_or("").to_string()).collect();
    let obs=observations(&cfg["sonde_files"],&valid_times)?;
    let mut cells=BTreeMap::new();for o in &obs {cells.entry(o.station.clone()).or_insert_with(||nearest(o.lat,o.lon,&la,&lo));}
    let arms=cfg["arms"].as_array().ok_or("arms missing")?;if arms.len()!=2 {return Err("this instrument needs exactly two matched arms".into())}
    let mut analyses=Vec::new();let mut sondes=Vec::new();let mut samples=Vec::new();let mut frame_health=Vec::new();let mut hashes=BTreeMap::new();hashes.insert(gridpath.to_string(),digest(gridpath)?);
    for lead in cfg["leads"].as_array().ok_or("leads missing")? {
        let valid=lead["valid"].as_str().ok_or("valid missing")?;let hours=lead["hours"].as_u64().ok_or("hours missing")?;
        eprintln!("scoring lead {hours}, {valid}");
        let mut frames=Vec::new();
        for arm in arms {
            let hp=arm["histories"][hours.to_string()].as_str().ok_or("history missing")?;let ip=arm["init"].as_str().ok_or("init missing")?;
            let f=frame(hp,ip,n)?;frame_health.push(json!({"arm":arm["id"],"lead_hours":hours,"fields":f.health}));frames.push(f);hashes.insert(hp.to_string(),digest(hp)?);if !hashes.contains_key(ip) {hashes.insert(ip.to_string(),digest(ip)?);}
        }
        let mut refs:BTreeMap<String,BTreeMap<String,Grid>>=BTreeMap::new();
        for (source,fields) in lead["references"].as_object().ok_or("references missing")? {
            let mut r=BTreeMap::new();for (key,v) in fields.as_object().ok_or("reference fields missing")? {let p=v.as_str().ok_or("reference path missing")?;r.insert(key.clone(),Grid::open(p)?);hashes.insert(p.to_string(),digest(p)?);}
            refs.insert(source.clone(),r);
        }
        for key in frames[0].fields.keys() {
            let target:f64=key[1..].parse::<f64>()?*100.;
            for (refid,r) in &refs {
                for (region,min,max) in [("globe",-90.,90.),("NH",20.,90.),("tropics",-20.,20.),("SH",-90.,-20.)] {
                    let mut stat=[Stat::default(),Stat::default()];
                    for c in 0..n {
                        let lat=la[c].to_degrees();let lon=lo[c].to_degrees();if lat<min || lat>max {continue}
                        if refs.values().any(|rr|{let sp=rr["sp"].sample(lat,lon);!sp.is_finite()||sp<target}) {continue}
                        let an=r[key].sample(lat,lon);let a=frames[0].fields[key][c];let b=frames[1].fields[key][c];
                        if !an.is_finite()||!a.is_finite()||!b.is_finite() {continue}
                        stat[0].add(a-an,area[c]);stat[1].add(b-an,area[c]);
                    }
                    for i in 0..2 {analyses.push(json!({"lead_hours":hours,"reference":refid,"region":region,"field":key,"arm":arms[i]["id"],"stats":stat[i].value()}));}
                }
            }
        }
        let mut stats:BTreeMap<(String,String),Stat>=BTreeMap::new();
        for o in obs.iter().filter(|o|o.valid==valid) {
            let c=cells[&o.station];
            for (name,value) in [("z",o.z),("t",o.t),("u",o.u),("v",o.v)] {
                let key=format!("{name}{}",o.level);let a=frames[0].fields[&key][c];let b=frames[1].fields[&key][c];
                if !value.is_finite()||!a.is_finite()||!b.is_finite() {continue}
                let mut row=json!({"station":o.station,"valid":valid,"lead_hours":hours,"latitude":o.lat,"longitude":o.lon,"cell":c,"field":key,"observation":value});
                for i in 0..2 {let id=arms[i]["id"].as_str().unwrap();let fc=frames[i].fields[&key][c];stats.entry((id.into(),key.clone())).or_default().add(fc-value,1.);row[id]=json!(fc);}
                for (id,r) in &refs {let fc=r[&key].sample(o.lat,o.lon);if fc.is_finite() {stats.entry((id.clone(),key.clone())).or_default().add(fc-value,1.);row[id]=json!(fc);}}
                samples.push(row);
            }
        }
        for ((arm,key),s) in stats {sondes.push(json!({"lead_hours":hours,"arm":arm,"field":key,"stats":s.value()}));}
    }
    vector_rows(&mut analyses);vector_rows(&mut sondes);
    let result=json!({"schema":"hex.initial-source-skill/v1","instrument":"Rust netcrust + grib-core, linear ln-pressure interpolation; analysis bilinear to cells with area weights; sounding nearest cell at nominal time; exact standard pressure records","analyses":analyses,"sondes":sondes,"sonde_samples":samples,"frame_health":frame_health,"sonde_station_count":cells.len(),"input_sha256":hashes,"configuration":cfg});
    fs::write(args.get(2).ok_or("output path missing")?,serde_json::to_string_pretty(&result)?)?;
    Ok(())
}
