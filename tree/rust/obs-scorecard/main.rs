//! Native observation operators. Raw formats use the engine's Rust readers.
//! Sonde selection and paired squared-error diagnostics follow obsref.py.
use chrono::DateTime;
use grib_core::grib2::Grib2File;
use rw_wrfbatch::verification::{paired_fss_rect, paired_grid_errors};
use rw_wrfbatch::verification_io::{GridData, RadarSpec, load_radar, source_identity};
use rustwx_core::{CanonicalField as F, FieldSelector as S};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const LEVELS: &[u16] = &[1000,925,850,700,500,400,300,250,200,150,100,70,50,30,20,10];
fn xyz(lat:f64,lon:f64)->[f64;3] {let (a,b)=(lat.to_radians(),lon.to_radians());[a.cos()*b.cos(),a.cos()*b.sin(),a.sin()]}
fn dot(a:[f64;3],b:[f64;3])->f64 {a.iter().zip(b).map(|(a,b)|a*b).sum()}
fn cross(a:[f64;3],b:[f64;3])->[f64;3] {[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
#[derive(Clone)] struct Node {i:usize,axis:usize,left:Option<Box<Node>>,right:Option<Box<Node>>}
fn tree(ids:&mut [usize],pts:&[[f64;3]],depth:usize)->Option<Box<Node>> {
    if ids.is_empty(){return None} let axis=depth%3;let mid=ids.len()/2;
    ids.select_nth_unstable_by(mid,|&a,&b|pts[a][axis].total_cmp(&pts[b][axis]));
    let (left,rest)=ids.split_at_mut(mid);let (center,right)=rest.split_at_mut(1);
    Some(Box::new(Node{i:center[0],axis,left:tree(left,pts,depth+1),right:tree(right,pts,depth+1)}))
}
fn query(node:&Option<Box<Node>>,pts:&[[f64;3]],p:[f64;3],best:&mut Vec<(f64,usize)>,n:usize) {
    let Some(node)=node else{return}; let d=(0..3).map(|a|(pts[node.i][a]-p[a]).powi(2)).sum::<f64>();
    best.push((d,node.i));best.sort_by(|a,b|a.0.total_cmp(&b.0));best.truncate(n);
    let delta=p[node.axis]-pts[node.i][node.axis]; let (near,far)=if delta<0. {(&node.left,&node.right)}else{(&node.right,&node.left)};
    query(near,pts,p,best,n);if best.len()<n || delta*delta<=best.last().unwrap().0 {query(far,pts,p,best,n)}
}
struct Mesh {pts:Vec<[f64;3]>,node:Option<Box<Node>>}
impl Mesh {
    fn new(lat:&[f64],lon:&[f64])->Self {let pts=lat.iter().zip(lon).map(|(&a,&b)|xyz(a.to_degrees(),b.to_degrees())).collect::<Vec<_>>();
        let node=tree(&mut (0..pts.len()).collect::<Vec<_>>(),&pts,0);Self{pts,node}}
    fn nearest(&self,p:[f64;3],n:usize)->Vec<(f64,usize)> {let mut b=Vec::new();query(&self.node,&self.pts,p,&mut b,n);b}
    fn weights(&self,lat:f64,lon:f64)->Vec<(usize,f64)> {
        let p=xyz(lat,lon);let ids=self.nearest(p,3);if ids.len()<3{return vec![(ids[0].1,1.)]}
        let (a,b,c)=(self.pts[ids[0].1],self.pts[ids[1].1],self.pts[ids[2].1]);let det=dot(a,cross(b,c));
        if det.abs()<1e-12{return vec![(ids[0].1,1.)]}
        let mut w=[dot(p,cross(b,c))/det,dot(a,cross(p,c))/det,dot(a,cross(b,p))/det];let sum=w.iter().sum::<f64>();
        for x in &mut w {*x/=sum} if w.iter().any(|&x|!x.is_finite() || x< -0.5 || x>1.5) {return vec![(ids[0].1,1.)]}
        ids.iter().zip(w).map(|(a,w)|(a.1,w)).collect()
    }
}
#[derive(Clone)] struct Ob {id:String,lat:f64,lon:f64,p:u16,q:String,val:f64,elev:f64}
fn hx(s:&str)->Result<f64> {Ok(f64::from_bits(u64::from_str_radix(s,16)?))}
fn sondes(path:&Path,valid:&str)->Result<Vec<Ob>> {
    let text=std::fs::read_to_string(path)?;if text.starts_with("station_id,") {return neutral_sondes(&text,valid)}let mut rows=Vec::new();let mut kind=String::new();let mut header=String::new();let mut mode=0;let mut base=None;
    for line in text.lines() {let t=line.split_whitespace().collect::<Vec<_>>();if t.is_empty(){continue}
        match t[0] {
            "M"=>{kind=t[1].into();base=t.get(2).map(|x|x.to_string());mode=0},
            "S"=>{mode=0;if kind!="ADPUPA"{continue} if t.len()<9{return Err("short prepBUFR subset".into())}
                let typ=hx(t[6])?;if typ==120. {mode=1}else if typ==220. {mode=2} header=line.into();
                let expected=valid.replace(['-',':','T','Z'],"");if base.as_deref()!=Some(&expected[..10]) {return Err("prepBUFR message cycle differs from forecast valid time".into())}},
            "L" if mode>0=>{
                if t.len()<34{return Err("short prepBUFR level".into())} let h=header.split_whitespace().collect::<Vec<_>>();
                let p=hx(t[2])?;if hx(t[15])?>=4. {continue} let Some(&lev)=LEVELS.iter().find(|&&l|(p-l as f64).abs()<0.05) else{continue};
                let sid=u64::from_str_radix(h[2],16)?.to_le_bytes();let id=String::from_utf8_lossy(&sid).trim().to_string();
                let (mut lon,mut lat)=(hx(h[3])?,hx(h[4])?);let (x,y)=(hx(t[30])?,hx(t[31])?);
                if x.abs()<400. && y.abs()<=90. {(lon,lat)=(x,y)}
                if lat.abs()>90. || lon.abs()>400. {continue} // Time window retained from subset DHR.
                if hx(h[5])?.abs()>1.5 {continue}
                let mut add=|q:&str,value:f64,mark:f64| {if value.is_finite() && value.abs()<1e9 && mark<4. {rows.push(Ob{id:id.clone(),lat,lon,p:lev,q:q.into(),val:value,elev:0.})}};
                if mode==1 {add("height",hx(t[5])?,hx(t[18])?);let mut v=hx(t[4])?+273.15;
                    if t.last()==Some(&"V") {let q=hx(t[3])?;if q.abs()>=1e9 {continue}v/=1.+0.61*q*1e-6} add("temperature",v,hx(t[17])?);
                }else{add("u",hx(t[6])?,hx(t[19])?);add("v",hx(t[7])?,hx(t[19])?)}
            },_=>{}
        }
    }
    let mut seen=BTreeSet::new();rows.retain(|r|seen.insert((r.id.clone(),r.p,r.q.clone())));Ok(rows)
}
/// A verification region is data: a longitude/latitude box and the lattice step the precipitation
/// comparison uses. Absent, the scorer keeps its original CONUS lattice and scores every report it is given.
#[derive(Clone,Copy,Debug,PartialEq)] struct Region {w:f64,s:f64,e:f64,n:f64,step:f64}
const CONUS:Region=Region{w:-125.,s:22.,e:-66.,n:50.,step:0.125};
impl Region {
    fn from_request(request:&Value)->Result<Option<Region>> {
        let r=&request["region"];if r.is_null(){return Ok(None)}
        let g=|k:&str|r[k].as_f64().ok_or_else(||format!("region {k} absent or not a number"));
        let reg=Region{w:g("west")?,s:g("south")?,e:g("east")?,n:g("north")?,step:if r["step_degrees"].is_null(){0.125}else{g("step_degrees")?}};
        if !(reg.n>reg.s && reg.e>reg.w && reg.e-reg.w<=360. && reg.s>= -90. && reg.n<=90. && reg.step>0.) {return Err("region needs south<north within +-90, west<east spanning at most 360 degrees and a positive step".into())}
        Ok(Some(reg))
    }
    fn contains(&self,lat:f64,lon:f64)->bool {lat>=self.s && lat<=self.n && (lon-self.w).rem_euclid(360.)<=self.e-self.w}
    fn shape(&self)->(usize,usize) {(((self.e-self.w)/self.step).round() as usize+1,((self.n-self.s)/self.step).round() as usize+1)}
    fn coordinates(&self)->Vec<(f64,f64)> {let (nx,ny)=self.shape();let r=*self;(0..ny).flat_map(|y|(0..nx).map(move|x|(r.s+y as f64*r.step,r.w+x as f64*r.step))).collect()}
}
/// Neutral radiosonde table (`station_id,latitude_deg,longitude_deg,elevation_m,valid_time,pressure_hpa,variable,value`),
/// the sonde counterpart of the neutral surface table, so any sounding archive is a converter, not a scorer path.
fn neutral_sondes(text:&str,valid:&str)->Result<Vec<Ob>> {
    let target=DateTime::parse_from_rfc3339(valid)?.timestamp();let mut lines=text.lines();
    let headers=lines.next().ok_or("sonde table empty")?.split(',').collect::<Vec<_>>();
    let index=|s:&str|headers.iter().position(|h|*h==s).ok_or_else(||format!("sonde table has no {s}"));
    let (id,la,lo,tm,pr,var,val)=(index("station_id")?,index("latitude_deg")?,index("longitude_deg")?,index("valid_time")?,index("pressure_hpa")?,index("variable")?,index("value")?);
    let mut rows=Vec::new();
    for row in lines {let r=row.split(',').collect::<Vec<_>>();if r.len()!=headers.len(){return Err("neutral sonde table row length differs from its header".into())}
        // Nominal synoptic launch hour; the table carries the nominal time, never the release time.
        if (DateTime::parse_from_rfc3339(r[tm])?.timestamp()-target).abs()>3600 {continue}
        let p:f64=r[pr].parse()?;let Some(&lev)=LEVELS.iter().find(|&&l|(p-l as f64).abs()<0.05) else{continue};
        let q=match r[var] {"height_m"=>"height","temperature_k"=>"temperature","wind_u_m_s"=>"u","wind_v_m_s"=>"v",_=>continue};
        let value:f64=r[val].parse()?;if !value.is_finite(){continue}
        rows.push(Ob{id:r[id].into(),lat:r[la].parse()?,lon:r[lo].parse()?,p:lev,q:q.into(),val:value,elev:0.});
    }
    let mut seen=BTreeSet::new();rows.retain(|r|seen.insert((r.id.clone(),r.p,r.q.clone())));Ok(rows)
}
/// One IGRA v2 station file as text: the `.txt` NCEI serves, or the `.txt.zip` it serves it in (what the engine's
/// `rw_igra2 fetch` keeps), so a fetched archive is decoded where it landed. A zip is walked by its local headers
/// (`PK\x03\x04`), stored (0) and deflate (8) members only, as the engine's rw_igra2 reads the same files.
fn igra_text(path:&Path)->Result<String> {
    let bytes=std::fs::read(path)?;let name=path.to_string_lossy();
    if !bytes.starts_with(b"PK\x03\x04") {return Ok(String::from_utf8(bytes).map_err(|_|format!("{name}: IGRA text is not UTF-8"))?)}
    let (mut at,mut text)=(0usize,String::new());
    let u16le=|b:&[u8],i:usize|u16::from_le_bytes([b[i],b[i+1]]) as usize;
    let u32le=|b:&[u8],i:usize|u32::from_le_bytes([b[i],b[i+1],b[i+2],b[i+3]]) as usize;
    while bytes.len()>=at+30 && bytes[at..at+4]==*b"PK\x03\x04" {
        let (flags,method,size,nlen,xlen)=(u16le(&bytes,at+6),u16le(&bytes,at+8),u32le(&bytes,at+18),u16le(&bytes,at+26),u16le(&bytes,at+28));
        if flags&0x8!=0 {return Err(format!("{name}: a zip member with a trailing data descriptor (sizes after the data) is not read").into())}
        let start=at+30+nlen+xlen;let end=start.checked_add(size).filter(|&e|e<=bytes.len()).ok_or_else(||format!("{name}: a zip member runs past the end of the file"))?;
        let member=String::from_utf8_lossy(&bytes[at+30..at+30+nlen]).to_string();let data=&bytes[start..end];
        let raw=match method {0=>data.to_vec(),8=>{let mut out=Vec::new();std::io::Read::read_to_end(&mut flate2::read::DeflateDecoder::new(data),&mut out).map_err(|e|format!("{name}: member {member} will not inflate: {e}"))?;out},
            m=>return Err(format!("{name}: member {member} uses zip method {m}; only stored (0) and deflate (8) are read").into())};
        if member.ends_with(".txt") {text.push_str(&String::from_utf8(raw).map_err(|_|format!("{name}: member {member} is not UTF-8"))?)}
        at=end;
    }
    if text.is_empty() {return Err(format!("{name}: the zip holds no .txt station file").into())}
    Ok(text)
}
/// IGRA v2 station files (fixed-width text, NOAA NCEI) to the neutral sonde table. Mandatory levels only,
/// QC-removed (-8888) and missing (-9999) values dropped, nominal hour 99 (unknown) skipped.
fn decode_igra(out:&Path,start:&str,end:&str,inputs:&[String])->Result<Value> {
    let (a,b)=(DateTime::parse_from_rfc3339(start)?.timestamp(),DateTime::parse_from_rfc3339(end)?.timestamp());
    let mut text=String::from("station_id,latitude_deg,longitude_deg,elevation_m,valid_time,pressure_hpa,variable,value\n");
    let (mut soundings,mut rows)=(0usize,0usize);let mut sources=Vec::new();
    for input in inputs {
        let body=igra_text(Path::new(input))?;sources.push(source_identity(Path::new(input))?);
        let mut current:Option<(String,f64,f64,String)>=None;
        for line in body.lines() {
            let f=|r:std::ops::Range<usize>|line.get(r).map(str::trim).unwrap_or("");
            if line.starts_with('#') {
                current=None;let hour:u32=match f(24..26).parse(){Ok(h) if h<24=>h,_=>continue};
                let date=chrono::NaiveDate::from_ymd_opt(f(13..17).parse()?,f(18..20).parse()?,f(21..23).parse()?).ok_or("IGRA header date invalid")?;
                let when=date.and_hms_opt(hour,0,0).ok_or("IGRA header hour invalid")?.and_utc();
                if when.timestamp()<a || when.timestamp()>b {continue}
                let (lat,lon)=(f(55..62).parse::<f64>()?/1e4,f(63..71).parse::<f64>()?/1e4);
                if lat.abs()>90. || lon.abs()>180. {continue}
                soundings+=1;current=Some((f(1..12).to_string(),lat,lon,when.format("%Y-%m-%dT%H:%M:%SZ").to_string()));continue
            }
            let Some((id,lat,lon,when))=&current else{continue};
            let press:i64=match f(9..15).parse(){Ok(p)=>p,Err(_)=>continue};if press<=0 || press%100!=0 {continue}
            let p=(press/100) as u16;if !LEVELS.contains(&p){continue}
            let num=|r:std::ops::Range<usize>|->Option<f64>{let v:i64=f(r).parse().ok()?;(v> -8888).then_some(v as f64)};
            let mut push=|var:&str,value:f64|{text.push_str(&format!("{id},{lat},{lon},0,{when},{p},{var},{value}\n"));rows+=1};
            if let Some(z)=num(16..21) {push("height_m",z)}
            if let Some(t)=num(22..27) {push("temperature_k",t/10.+273.15)}
            if let (Some(d),Some(s))=(num(40..45),num(46..51)) {if (0.0..=360.).contains(&d){let (s,r)=(s/10.,d.to_radians());push("wind_u_m_s",-s*r.sin());push("wind_v_m_s",-s*r.cos())}}
        }
    }
    std::fs::write(out,&text)?;
    Ok(json!({"schema":"gpuwm-hex.decoded-igra/v1","window":[start,end],"soundings":soundings,"rows":rows,"inputs":sources,"table":source_identity(out)?,"levels":"mandatory pressure levels the scorer reports"}))
}
fn surface(path:&Path,valid:&str)->Result<Vec<Ob>> {
    let target=DateTime::parse_from_rfc3339(valid)?.timestamp();let text=std::fs::read_to_string(path)?;let mut lines=text.lines();
    let headers=lines.next().ok_or("ASOS table empty")?.split(',').collect::<Vec<_>>();
    let index=|s:&str|headers.iter().position(|h|*h==s).ok_or_else(||format!("ASOS table has no {s}"));
    let (id,la,lo,z,tm,var,val,meas)=(index("station_id")?,index("latitude_deg")?,index("longitude_deg")?,index("elevation_m")?,index("valid_time")?,index("variable")?,index("value")?,index("measurement")?);
    let mut chosen:BTreeMap<(String,String),(i64,Ob)>=BTreeMap::new();
    for row in lines {let r=row.split(',').collect::<Vec<_>>();if r.len()!=headers.len(){return Err("neutral ASOS table row length differs from its header".into())}let dt=(DateTime::parse_from_rfc3339(r[tm])?.timestamp()-target).abs();if dt>600{continue}
        let q=match r[var] {"temperature_k" if r[meas]=="screen_temperature_2m"=>"temperature", "dewpoint_k" if r[meas]=="screen_dewpoint_2m"=>"dewpoint",
            "wind_u_m_s" if r[meas]=="anemometer_wind_10m"=>"u", "wind_v_m_s" if r[meas]=="anemometer_wind_10m"=>"v", "surface_pressure_pa" if r[meas]=="station_pressure_from_altimeter" || r[meas]=="station_pressure"=>"pressure",_=>continue};
        let o=Ob{id:r[id].into(),lat:r[la].parse()?,lon:r[lo].parse()?,p:0,q:q.into(),val:r[val].parse()?,elev:r[z].parse()?};
        let key=(o.id.clone(),o.q.clone());if chosen.get(&key).map(|x|dt<x.0).unwrap_or(true){chosen.insert(key,(dt,o));}
    } Ok(chosen.into_values().map(|r|r.1).collect())
}
struct Hex {mesh:Mesh,init_mesh:Mesh,n:usize,p:Vec<f64>,z:Vec<f64>,nz:usize,fields:BTreeMap<String,Vec<f64>>}
fn nc(file:&netcrust::File,name:&str)->Result<Vec<f64>> {Ok(file.read_array_f64(name)?.into_values())}
fn stream_field(history:&netcrust::File,diagnostics:Option<&netcrust::File>,names:&[&str])->Result<Option<Vec<f64>>> {
    for file in std::iter::once(history).chain(diagnostics) {
        for name in names {if file.variable(name).is_some(){return Ok(Some(nc(file,name)?))}}
    }
    Ok(None)
}
impl Hex {
    fn open(history:&Path,init:&Path,diagnostics:Option<&Path>)->Result<Self> {
        let f=netcrust::File::open(history)?;let i=netcrust::File::open(init)?;
        let d=diagnostics.map(netcrust::File::open).transpose()?;
        let lat=f.read_f64("latCell")?;let lon=f.read_f64("lonCell")?;let n=f.dimension("nVertLevels").ok_or("no nVertLevels")?.len();
        let init_mesh=Mesh::new(&i.read_f64("latCell")?,&i.read_f64("lonCell")?);let mesh=Mesh::new(&lat,&lon);let z=nc(&i,"zgrid")?;
        let p=nc(&f,"pressure")?;if p.len()!=lat.len()*n || z.len()!=init_mesh.pts.len()*(n+1){return Err("pressure/zgrid shape differs from cell and level dimensions".into())}
        let mut fields=BTreeMap::new();
        // MPAS spells the earth-relative winds and screen temperature differently.
        // Its diagnostics stream may hold fields absent from the output stream.
        for (q,names) in [("temperature",&["theta"][..]),("u",&["u_zonal","uReconstructZonal"][..]),
            ("v",&["v_meridional","uReconstructMeridional"][..]),("t2",&["t2","t2m"][..]),
            ("q2",&["q2"][..]),("ps",&["surface_pressure"][..]),("u10",&["u10"][..]),
            ("v10",&["v10"][..]),("rainc",&["rainc"][..]),("rainnc",&["rainnc"][..])] {
            if let Some(values)=stream_field(&f,d.as_ref(),names)? {fields.insert(q.into(),values);}
        }
        Ok(Self{mesh,init_mesh,n,p,z,nz:n+1,fields})
    }
    fn zcell(&self,c:usize)->Option<usize> {let a=self.init_mesh.nearest(self.mesh.pts[c],1)[0];(a.0.sqrt()<1e-6).then_some(a.1)}
    fn sample(&self,o:&Ob)->Option<f64> {
        let mut out=0.;for (c,w) in self.mesh.weights(o.lat,o.lon) {if w.abs()<1e-12{continue}let ic=self.zcell(c)?;let zs=self.z[ic*self.nz];
            let value=if o.p>0 {let col=&self.p[c*self.n..(c+1)*self.n];if col.iter().any(|p|!p.is_finite() || *p<=0.) || col.windows(2).any(|p|p[0]<=p[1]) {return None}
                let (k,a)=rw_isobaric::bracket(col,o.p as f64*100.)?;
                let get=|j:usize|->Option<f64> {if o.q=="height" {Some(0.5*(self.z[ic*self.nz+j]+self.z[ic*self.nz+j+1]))}else{
                    let v=*self.fields.get(&o.q)?.get(c*self.n+j)?;Some(if o.q=="temperature" {v*(col[j]/1e5).powf(287./1004.5)}else{v})}};
                rw_isobaric::lerp(get(k)?,get(k+1)?,a)?
            }else {match o.q.as_str() {
                "temperature"=>*self.fields.get("t2")?.get(c)?,"u"=>*self.fields.get("u10")?.get(c)?,"v"=>*self.fields.get("v10")?.get(c)?,
                "dewpoint"=>{let q=self.fields.get("q2")?[c].max(0.);let ps=self.fields.get("ps")?[c];let e=q*ps/(0.622+q);if e<=0. {return None}let l=(e/611.2).ln();273.15+243.5*l/(17.67-l)},
                "pressure"=>{let dz=zs-o.elev;if dz.abs()>50. {return None}let t=self.fields.get("temperature")?[c*self.n]*(self.p[c*self.n]/1e5).powf(287./1004.5);
                    let zmid=0.5*(self.z[ic*self.nz]+self.z[ic*self.nz+1]);let tl=t+0.0065*(zmid-zs);self.fields.get("ps")?[c]*(9.80665*dz/(287.*(tl+0.00325*dz))).exp()},_=>return None}}
            ;if !value.is_finite() || value.abs()>1e20 {return None}out+=w*value;
        }Some(out)
    }
    fn rain(&self,lat:f64,lon:f64)->Option<f64> {let c=self.mesh.nearest(xyz(lat,lon),1)[0].1;Some(self.fields.get("rainc")?[c]+self.fields.get("rainnc")?[c])}
}
struct Plane {grid:GridData,values:Vec<f64>}
impl Plane {fn sample(&self,lat:f64,lon:f64)->Option<f64> {
    let g=&self.grid;
    if g.nx>=2 && g.ny>=2 {
        let dx=f64::from(g.lon[1]-g.lon[0]);let dy=f64::from(g.lat[g.nx]-g.lat[0]);
        // Global regular grids need a cyclic seed. The engine's bounded-grid
        // local inverse starts at longitude zero and cannot walk through the
        // array edge to a negative-longitude point on a 0..360 grid.
        if (dx.abs()*g.nx as f64-360.).abs()<0.001 && dy.abs()>1e-9
            && (g.lat[g.nx-1]-g.lat[0]).abs()<1e-5
            && (g.lon[g.nx]-g.lon[0]).abs()<1e-5 {
            let x=((lon-f64::from(g.lon[0]))/dx).rem_euclid(g.nx as f64);let y=(lat-f64::from(g.lat[0]))/dy;
            if y< -1e-6 || y>g.ny as f64-1.+1e-6 {return None}
            let y=y.clamp(0.,g.ny as f64-1.);let (ix,iy)=(x.floor() as usize,(y.floor() as usize).min(g.ny-2));let (tx,ty)=(x-ix as f64,y-iy as f64);
            let mut result=0.;for (i,w) in [(iy*g.nx+ix,(1.-tx)*(1.-ty)),(iy*g.nx+(ix+1)%g.nx,tx*(1.-ty)),((iy+1)*g.nx+ix,(1.-tx)*ty),((iy+1)*g.nx+(ix+1)%g.nx,tx*ty)] {
                if w<=1e-12{continue}let v=*self.values.get(i)?;if !v.is_finite() || v.abs()>1e20 {return None}result+=w*v;
            }return Some(result)
        }
    }
    self.grid.sample(&self.values,self.grid.position(lat,lon)?)
} }
struct Reference {label:String,role:String,fields:BTreeMap<String,Plane>,provenance:Value}
fn raw_parameter(g:&Grib2File,table:&Value,quantity:&str)->Result<Option<Plane>> {
    for row in table.as_array().into_iter().flatten().filter(|r|r["quantity"]==quantity) {
        let multiplier=row["multiplier"].as_f64().ok_or("GRIB metadata multiplier missing")?;
        if !multiplier.is_finite() || multiplier<=0. {return Err("GRIB unit multiplier must be finite and positive".into())}
        if row["units"]!=if quantity=="precipitation"{"mm"}else{"m"} {return Err("raw parameter canonical units disagree with the scored quantity".into())}
        let matches=g.messages.iter().filter(|m|m.discipline as u64==row["discipline"].as_u64().unwrap_or(u64::MAX)
            && m.product.parameter_category as u64==row["category"].as_u64().unwrap_or(u64::MAX)
            && m.product.parameter_number as u64==row["parameter"].as_u64().unwrap_or(u64::MAX)
            && m.product.level_type as u64==row["level_type"].as_u64().unwrap_or(u64::MAX)
            && m.product.second_level_type==255
            && (quantity!="terrain" || m.product.statistical_process_type.is_none())
            && row["center_id"].as_u64().map(|v|v==m.identification.center_id as u64).unwrap_or(true)
            && row["local_table_version"].as_u64().map(|v|v==m.identification.local_table_version as u64).unwrap_or(true)).collect::<Vec<_>>();
        if matches.is_empty(){continue}if matches.len()!=1{return Err("raw parameter metadata matches multiple messages".into())}
        let m=matches[0];if m.grid.is_reduced || m.grid.scan_mode&0x30!=0 {return Err("raw parameter needs regular non-alternating grid rows".into())}
        let values=grib_core::grib2::unpack::unpack_message(m)?.into_iter().map(|v|v*multiplier).collect::<Vec<_>>();
        let (lat,lon)=grib_core::grib2::grid::grid_latlon(&m.grid)?;
        if values.len()!=lat.len() || values.len()!=lon.len(){return Err("raw parameter and coordinate grid lengths differ".into())}
        return Ok(Some(Plane{grid:GridData{lat:lat.into_iter().map(|x|x as f32).collect(),lon:lon.into_iter().map(|x|x as f32).collect(),ny:m.grid.ny as usize,nx:m.grid.nx as usize,projection:None},values}));
    }Ok(None)
}
fn raw_isobaric(g:&Grib2File,selector:S,hour:u16)->Result<Option<Plane>> {
    let rustwx_core::VerticalSelector::IsobaricHpa(level)=selector.vertical else{return Ok(None)};
    // WMO parameter metadata, shared by all reference models. The higher-level
    // selector door restricts pressure to 100..1000 hPa; the same GRIB reader
    // can decode the remaining mandatory levels without that product guard.
    let parameter=match selector.field {F::GeopotentialHeight=>(3,5),F::Temperature=>(0,0),F::UWind=>(2,2),F::VWind=>(2,3),_=>return Ok(None)};
    let messages=g.messages.iter().filter(|m|m.discipline==0 && (m.product.parameter_category,m.product.parameter_number)==parameter
        && m.product.level_type==100 && (m.product.level_value-f64::from(level)*100.).abs()<0.01
        && m.product.second_level_type==255 && m.product.statistical_process_type.is_none()
        && m.product.time_range_unit==1 && m.product.forecast_time==u32::from(hour)).collect::<Vec<_>>();
    if messages.is_empty(){return Ok(None)}if messages.len()!=1 {return Err("multiple deterministic messages match one mandatory-level field".into())}
    let m=messages[0];if m.grid.is_reduced || m.grid.scan_mode&0x30!=0 {return Err("mandatory-level raw extraction requires non-alternating regular grid rows".into())}
    let values=grib_core::grib2::unpack::unpack_message(m)?;let (lat,lon)=grib_core::grib2::grid::grid_latlon(&m.grid)?;
    if values.len()!=lat.len() || values.len()!=lon.len() {return Err("raw mandatory-level values differ from their coordinate grid".into())}
    Ok(Some(Plane{grid:GridData{lat:lat.into_iter().map(|x|x as f32).collect(),lon:lon.into_iter().map(|x|x as f32).collect(),ny:m.grid.ny as usize,nx:m.grid.nx as usize,projection:None},values}))
}
fn reference(spec:&Value,levels:&[u16],valid:&str,start:&str,table:&Value)->Result<Reference> {
    let path=Path::new(spec["path"].as_str().ok_or("reference path absent")?);let g=Grib2File::from_bytes(&std::fs::read(path)?)?;
    let hour=spec["hour"].as_u64().ok_or("reference hour absent")? as u16;
    let want=DateTime::parse_from_rfc3339(valid)?.naive_utc();let mut g=g;
    let cycle=DateTime::parse_from_rfc3339(if spec["role"]=="analysis" {valid}else{start})?.naive_utc();
    g.messages.retain(|m|m.reference_time==cycle && m.reference_time+chrono::Duration::hours(i64::from(hour))==want);
    if g.messages.is_empty(){return Err("reference initialization and lead differ from requested valid time".into())}
    let mut selectors=Vec::new();let mut keys=Vec::new();
    for &l in levels {for (q,f) in [("height",F::GeopotentialHeight),("temperature",F::Temperature),("u",F::UWind),("v",F::VWind)] {selectors.push(S::isobaric(f,l));keys.push(format!("{q}:{l}"));}}
    for (q,s) in [("temperature:0",S::height_agl(F::Temperature,2)),("dewpoint:0",S::height_agl(F::Dewpoint,2)),("u:0",S::height_agl(F::UWind,10)),("v:0",S::height_agl(F::VWind,10)),("pressure:0",S::surface(F::Pressure)),("terrain:0",S::surface(F::GeopotentialHeight))] {selectors.push(s);keys.push(q.into())}
    // A selector capability gap must leave one field absent, not erase a
    // whole reference model's otherwise decodable observation scores.
    let mut raw_fields=BTreeMap::new();
    let extracted=loop {match rustwx_io::extract_fields_from_grib2_partial_at_forecast_hour(&g,&selectors,hour) {
        Ok(value)=>break value,
        Err(rustwx_io::IoError::UnsupportedStructuredSelector{selector})=>{
            let index=selectors.iter().position(|s|*s==selector).ok_or("unsupported selector was not in the request")?;
            if let Some(plane)=raw_isobaric(&g,selector,hour)? {raw_fields.insert(keys[index].clone(),plane);}
            selectors.remove(index);keys.remove(index);
        },Err(e)=>return Err(e.into())
    }};let mut fields=raw_fields;
    for field in extracted.extracted {if let Some(index)=selectors.iter().position(|s|*s==field.selector) {fields.insert(keys[index].clone(),Plane{grid:GridData{lat:field.grid.lat_deg,lon:field.grid.lon_deg,ny:field.grid.shape.ny,nx:field.grid.shape.nx,projection:field.projection},values:field.values.into_iter().map(f64::from).collect()});}}
    let mut provenance=source_identity(path)?;
    if !fields.contains_key("terrain:0") {
        if let Some(plane)=raw_parameter(&g,table,"terrain")? {fields.insert("terrain:0".into(),plane);}
        else if let Some(path)=spec["static_path"].as_str() {
            let mut s=Grib2File::from_bytes(&std::fs::read(path)?)?;
            s.messages.retain(|m|m.reference_time==cycle && m.product.forecast_time==0 && m.product.statistical_process_type.is_none());
            if let Some(plane)=raw_parameter(&s,table,"terrain")? {fields.insert("terrain:0".into(),plane);provenance["static_terrain"]=source_identity(Path::new(path))?;}
        }
    }
    Ok(Reference{label:spec["label"].as_str().ok_or("label absent")?.into(),role:spec["role"].as_str().unwrap_or("forecast").into(),fields,provenance})
}
impl Reference {
    fn bracket(&self,q:&str,p:u16)->Option<(u16,u16)> {
        if p==0{return None}let levels=self.fields.keys().filter_map(|key|key.strip_prefix(&format!("{q}:"))?.parse::<u16>().ok()).filter(|&l|l>0).collect::<Vec<_>>();
        Some((*levels.iter().filter(|&&l|l>p).min()?,*levels.iter().filter(|&&l|l<p).max()?))
    }
    fn has_field(&self,q:&str,p:u16)->bool {self.fields.contains_key(&format!("{q}:{p}")) || self.bracket(q,p).is_some()}
    fn sample(&self,o:&Ob)->Option<f64> {let mut value=if let Some(plane)=self.fields.get(&format!("{}:{}",o.q,o.p)){plane.sample(o.lat,o.lon)?}else{
        let (hi,lo)=self.bracket(&o.q,o.p)?;let a=self.fields.get(&format!("{}:{hi}",o.q))?.sample(o.lat,o.lon)?;let b=self.fields.get(&format!("{}:{lo}",o.q))?.sample(o.lat,o.lon)?;
        rw_isobaric::lerp(a,b,((o.p as f64).ln()-(hi as f64).ln())/((lo as f64).ln()-(hi as f64).ln()))?};
        if o.p==0 && o.q=="pressure" {let dz=self.fields.get("terrain:0")?.sample(o.lat,o.lon)?-o.elev;if dz.abs()>50. {return None}let t=self.fields.get("temperature:0")?.sample(o.lat,o.lon)?;value*=(9.80665*dz/(287.*(t+0.00325*dz))).exp()}Some(value)}
}
fn statistics(errors:&[(String,f64,f64)])->Value {if errors.is_empty(){return json!({"n":0,"rmse":null,"bias":null,"status":"NOT_MEASURED"})}
    let n=errors.len() as f64;json!({"n":errors.len(),"rmse":(errors.iter().map(|r|r.1).sum::<f64>()/n).sqrt(),"bias":errors.iter().map(|r|r.2).sum::<f64>()/n,"status":"MEASURED"})}
fn score_observations(source:&str,obs:&[Ob],hex:&Hex,refs:&[Reference])->Value {
    let labels=std::iter::once("hex".to_string()).chain(refs.iter().map(|r|r.label.clone())).collect::<Vec<_>>();let mut metrics=Vec::new();let mut paired=Vec::new();
    for p in if source=="radiosonde" {LEVELS.to_vec()}else{vec![0]} {for q in if source=="radiosonde" {vec!["height","temperature","wind"]}else{vec!["temperature","dewpoint","wind","pressure"]} {
        let selected=obs.iter().filter(|o|o.p==p && (o.q==q || q=="wind" && o.q=="u")).collect::<Vec<_>>();
        let mut all=vec![Vec::new();labels.len()];let mut dropped=0;let mut available=vec![true;labels.len()];
        for (j,r) in refs.iter().enumerate() {available[j+1]=r.has_field(if q=="wind"{"u"}else{q},p) && (q!="wind" || r.has_field("v",p)) && (p!=0 || q!="pressure" || r.fields.contains_key("terrain:0"));}
        for o in &selected {let v=if q=="wind"{obs.iter().find(|v|v.id==o.id && v.p==o.p && v.q=="v")}else{None};if q=="wind" && v.is_none(){dropped+=1;continue}
            let mut samples=Vec::new();samples.push(hex.sample(o).zip(if let Some(v)=v{hex.sample(v)}else{Some(0.)}));
            samples.extend(refs.iter().map(|r|r.sample(o).zip(if let Some(v)=v{r.sample(v)}else{Some(0.)})));
            if samples.iter().zip(&available).any(|(s,a)|*a && s.is_none()) {dropped+=1;continue}
            for (j,s) in samples.iter().enumerate() {if !available[j]{continue}let Some((a,b))=s else{continue};let e=a-o.val;
                let (e2,bias)=if let Some(v)=v{(e*e+(b-v.val).powi(2),a.hypot(*b)-o.val.hypot(v.val))}else{(e*e,e)};
                all[j].push((o.id.clone(),e2,bias));}
        }
        for (j,label) in labels.iter().enumerate() {let mut row=statistics(&all[j]);row["source"]=json!(source);row["quantity"]=json!(q);row["level_hpa"]=json!(p);row["label"]=json!(label);row["units"]=json!(match q {"height"=>"m","temperature"|"dewpoint"=>"K","wind"=>"m s-1",_=>"Pa"});
            row["selected"]=json!(selected.len());row["common_mask_drops"]=json!(dropped);
            if p>0 {if j==0{row["pressure_sampling"]=json!("native-column log-pressure interpolation")}else{let field=if q=="wind"{"u"}else{q};let reference=&refs[j-1];
                row["pressure_sampling"]=json!(if reference.fields.contains_key(&format!("{field}:{p}")){"published pressure level"}else{"log-pressure interpolation between published levels"});
                if !reference.fields.contains_key(&format!("{field}:{p}")){row["bracketing_levels_hpa"]=json!(reference.bracket(field,p));}
            }}
            if all[j].is_empty(){row["reason"]=json!(if !available[j]{"reference field absent"}else{"no matched observations with finite common model support"})}metrics.push(row);}
        for j in 1..labels.len() {if all[j].len()!=all[0].len() || all[j].is_empty(){continue}let mut blocks:BTreeMap<String,Vec<f64>>=BTreeMap::new();
            for (a,b) in all[0].iter().zip(&all[j]) {blocks.entry(a.0.clone()).or_default().push(a.1-b.1)}let d=blocks.values().map(|v|v.iter().sum::<f64>()/v.len() as f64).collect::<Vec<_>>();let n=d.len();let mean=d.iter().sum::<f64>()/n as f64;
            let se=if n>1 {Some((d.iter().map(|x|(x-mean).powi(2)).sum::<f64>()/(n-1) as f64/n as f64).sqrt())}else{None};
            paired.push(json!({"source":source,"quantity":q,"level_hpa":p,"reference":labels[j],"station_blocks":n,"mean_dsq_hex_minus_reference":mean,"se":se,"standardized_difference":se.filter(|&s|s>0.).map(|s|mean/s),"inference":"exploratory station-block diagnostic; one forecast cycle cannot establish independent-event significance"}));}
    }} json!({"metrics":metrics,"paired":paired})
}
fn precip_plane(path:&Path,start:&str,end:&str,table:&Value)->Result<Option<Plane>> {
    let mut g=Grib2File::from_bytes(&std::fs::read(path)?)?;
    let (a,b)=(DateTime::parse_from_rfc3339(start)?.naive_utc(),DateTime::parse_from_rfc3339(end)?.naive_utc());
    let hours=(b-a).num_hours();
    let seconds=|unit:u8,v:u32|->Option<i64>{Some(match unit{0=>i64::from(v)*60,1=>i64::from(v)*3600,2=>i64::from(v)*86400,10=>i64::from(v)*10800,11=>i64::from(v)*21600,12=>i64::from(v)*43200,13=>i64::from(v),_=>return None})};
    g.messages.retain(|m|m.product.statistical_process_type==Some(1) && m.product.statistical_time_range_hours().map(i64::from)==Some(hours)
        && m.product.end_of_interval==Some(b) && seconds(m.product.time_range_unit,m.product.forecast_time).map(|s|m.reference_time+chrono::Duration::seconds(s))==Some(a));
    if g.messages.is_empty(){return Ok(None)}
    let fields=rustwx_io::extract_fields_from_grib2_partial(&g,&[S::surface(F::TotalPrecipitation)])?;
    let Some(f)=fields.extracted.into_iter().next() else{return raw_parameter(&g,table,"precipitation")};
    if f.units!="mm" && f.units!="kg/m^2" {return Err(format!("precipitation has unsupported units {}",f.units).into())}
    Ok(Some(Plane{grid:GridData{lat:f.grid.lat_deg,lon:f.grid.lon_deg,ny:f.grid.shape.ny,nx:f.grid.shape.nx,projection:f.projection},values:f.values.into_iter().map(f64::from).collect()}))
}
fn score_rain(request:&Value,hex:&Hex,valid_references:&BTreeSet<String>)->Result<Value> {
    let prev=request["previous_time"].as_str().ok_or("no previous accumulation boundary")?;let end=request["valid_time"].as_str().unwrap();
    let hours=(DateTime::parse_from_rfc3339(end)?-DateTime::parse_from_rfc3339(prev)?).num_hours();
    let region=Region::from_request(request)?;let lattice=region.unwrap_or(CONUS);
    let (nx,ny)=lattice.shape();let coordinates=lattice.coordinates();
    let source=request["qpe"]["label"].as_str().unwrap_or("mrms").to_string();
    let (observed,valid,provenance)=if request["qpe"].is_object() {
        if hours<1 {return Err("satellite QPE comparison needs a positive accumulation interval".into())}
        qpe_netcdf(&request["qpe"],prev,end,&coordinates)?
    } else {
    let rain=request["rain"].as_array().ok_or("MRMS hourly packs absent")?;
    if hours<1 || rain.len()!=hours as usize {return Err("MRMS packs do not cover every hour of the model interval".into())}
    let mut observed=vec![0.;nx*ny];let mut valid=vec![true;nx*ny];let mut provenance=Vec::new();
    for (index,row) in rain.iter().enumerate() {
        let spec:RadarSpec=serde_json::from_value(row.clone())?;let radar=load_radar(&spec)?;
        let expected=DateTime::parse_from_rfc3339(prev)?+chrono::Duration::hours(index as i64+1);
        let raw_time=if radar.valid_time.len()==19 {format!("{}Z",radar.valid_time)}else{radar.valid_time.clone()};
        if DateTime::parse_from_rfc3339(&raw_time)?!=expected {return Err("MRMS interval end differs from requested hour".into())}
        let values=radar.values.iter().zip(&radar.valid).map(|(&v,&ok)|if ok{v}else{f64::NAN}).collect::<Vec<_>>();
        for (i,&(lat,lon)) in coordinates.iter().enumerate(){match radar.grid.position(lat,lon).and_then(|p|radar.grid.sample(&values,p)){Some(v) if v>=0.=>observed[i]+=v,_=>valid[i]=false}}
        provenance.push(json!({"pack":source_identity(&spec.path)?,"grid":source_identity(&spec.grid_path)?,"end":expected.to_rfc3339(),"source":radar.provenance}));
    }
    (observed,valid,provenance)};
    let previous_file=netcrust::File::open(Path::new(request["previous_history"].as_str().ok_or("previous history absent")?))?;
    let previous_mesh=Mesh::new(&previous_file.read_f64("latCell")?,&previous_file.read_f64("lonCell")?);
    let previous_diag=request["previous_diagnostics"].as_str().map(|p|netcrust::File::open(Path::new(p))).transpose()?;
    let rc=stream_field(&previous_file,previous_diag.as_ref(),&["rainc"])?.ok_or("previous rainc absent")?;
    let rn=stream_field(&previous_file,previous_diag.as_ref(),&["rainnc"])?.ok_or("previous rainnc absent")?;
    let candidate=coordinates.iter().map(|&(lat,lon)| {let c=previous_mesh.nearest(xyz(lat,lon),1)[0].1;
        let now=hex.rain(lat,lon).unwrap_or(f64::NAN);let value=now-rc[c]-rn[c];if value>= -1e-5 {value.max(0.)}else{f64::NAN}}).collect::<Vec<_>>();
    let mut arms=vec![("hex".into(),candidate)];let mut failures=Vec::new();
    for spec in request["references"].as_array().into_iter().flatten().filter(|s|s["role"]=="forecast" && s["label"].as_str().map(|l|valid_references.contains(l)).unwrap_or(false)) {
        let path=Path::new(spec["path"].as_str().unwrap());let planes=(||->Result<(Plane,Option<Plane>)> {
            if let Some(p)=precip_plane(path,prev,end,&request["grib_parameters"])? {return Ok((p,None))}
            let cycle=request["start"].as_str().ok_or("forecast cycle absent")?;
            let p=precip_plane(path,cycle,end,&request["grib_parameters"])?.ok_or("reference has no accumulation ending at the requested time")?;
            let previous=if prev==cycle{None}else{Some(precip_plane(Path::new(spec["previous_path"].as_str().ok_or("previous reference accumulation absent")?),cycle,prev,&request["grib_parameters"])?.ok_or("previous reference interval differs from its cycle")?)};
            Ok((p,previous))})();
        match planes {Ok((a,b))=>{let values=coordinates.iter().map(|&(la,lo)|{let v=a.sample(la,lo).unwrap_or(f64::NAN)-b.as_ref().map(|b|b.sample(la,lo).unwrap_or(f64::NAN)).unwrap_or(0.);if v>= -1e-5{v.max(0.)}else{f64::NAN}}).collect();arms.push((spec["label"].as_str().unwrap().into(),values));},Err(e)=>failures.push(json!({"label":spec["label"],"source":source,"reason":e.to_string()}))}
    }
    // A common fixed lattice (CONUS unless the request names a region) permits the same support across forecast grids.
    // Native readers bilinearly sample the observed QPE and structured references; hex keeps cell values.
    let (dx,dy)=(111.195*lattice.step*((lattice.s+lattice.n)/2.).to_radians().cos(),111.195*lattice.step);
    let mut metrics=Vec::new();
    for threshold in request["thresholds_mm"].as_array().ok_or("thresholds absent")? {for width in request["widths_km"].as_array().ok_or("widths absent")? {
        let result=paired_fss_rect("precipitation", "mm", threshold.as_f64().ok_or("invalid threshold")?,width.as_f64().ok_or("invalid width")?,dx,dy,ny,nx,&observed,&valid,&arms)?;
        for a in &result.arms {metrics.push(json!({"source":source,"quantity":"precipitation","label":a.label,"value":a.fss,"n":a.count,"status":if a.fss.is_some(){"MEASURED"}else{"NOT_MEASURED"},"reason":if a.fss.is_none(){Some("no complete observed-event neighborhoods")}else{None},"threshold_mm":result.threshold,"width_km":result.width_km,"actual_width_km":result.actual_width_km,"actual_height_km":result.actual_height_km,"interval_hours":hours,"common_cells":result.common_cells,"observed_event_cells":result.observed_event_cells}));}
    }}
    let continuous=paired_grid_errors("precipitation","mm",&observed,&valid,&arms)?;
    if region.is_none() {
        return Ok(json!({"metrics":metrics,"failures":failures,"provenance":provenance,"continuous":continuous,"grid":{"west":-125.,"east":-66.,"south":22.,"north":50.,"step_degrees":0.125,"dx_km_at_36N":dx,"dy_km":dy,"resampling":"nearest hex cell, bilinear MRMS and structured references; not a conservative water-budget estimate"}}))
    }
    Ok(json!({"metrics":metrics,"failures":failures,"provenance":provenance,"continuous":continuous,"observed_source":source,"grid":{"west":lattice.w,"east":lattice.e,"south":lattice.s,"north":lattice.n,"step_degrees":lattice.step,"dx_km_at_center_latitude":dx,"dy_km":dy,"resampling":format!("nearest hex cell, bilinear {source} and structured references; not a conservative water-budget estimate")}}))
}
/// Gridded satellite QPE on a regular latitude/longitude grid in NetCDF (CMORPH, IMERG-like layouts): a rate
/// variable with `time_bounds` per slice. Slices wholly inside (previous, valid] are summed as rate x duration;
/// a slice straddling the interval is refused, and the slices must tile the interval exactly.
fn qpe_netcdf(q:&Value,prev:&str,end:&str,coordinates:&[(f64,f64)])->Result<(Vec<f64>,Vec<bool>,Vec<Value>)> {
    let (a,b)=(DateTime::parse_from_rfc3339(prev)?.timestamp(),DateTime::parse_from_rfc3339(end)?.timestamp());
    let var=q["variable"].as_str().ok_or("satellite QPE variable absent")?;
    let units=q["units"].as_str().ok_or("satellite QPE units absent")?;
    let per_hour=match units {"mm/hr"|"mm h-1"|"mm/h"=>true,"mm"=>false,_=>return Err(format!("satellite QPE units {units} are not a supported precipitation rate or depth").into())};
    let mut observed=vec![0.;coordinates.len()];let mut valid=vec![true;coordinates.len()];let mut provenance=Vec::new();let mut covered=0i64;
    for path in q["files"].as_array().ok_or("satellite QPE files absent")?.iter().map(|p|p.as_str().ok_or("satellite QPE file is not a path")) {
        let path=Path::new(path?);let f=netcrust::File::open(path)?;let bounds=nc(&f,"time_bounds")?;
        let slices=(0..bounds.len()/2).map(|i|(i,bounds[2*i] as i64,bounds[2*i+1] as i64+1)).collect::<Vec<_>>();
        if slices.iter().any(|&(_,s,e)|s<b && e>a && !(s>=a && e<=b)) {return Err("satellite QPE slice straddles the model accumulation interval".into())}
        let inside=slices.into_iter().filter(|&(_,s,e)|s>=a && e<=b).collect::<Vec<_>>();if inside.is_empty(){continue}
        let v=f.variable(var).ok_or_else(||format!("{} has no {var}",path.display()))?;
        if v.attribute("units").and_then(|u|u.as_string()).map(|u|u.trim()!=units).unwrap_or(true) {return Err(format!("{} {var} units differ from the declared {units}",path.display()).into())}
        let scale=v.attribute("scale_factor").and_then(|x|x.as_f64()).unwrap_or(1.);let offset=v.attribute("add_offset").and_then(|x|x.as_f64()).unwrap_or(0.);
        let missing=["missing_value","_FillValue"].iter().filter_map(|k|v.attribute(k).and_then(|x|x.as_f64())).collect::<Vec<_>>();
        let lat=f.read_f64("lat")?;let lon=f.read_f64("lon")?;let raw=nc(&f,var)?;let (ny,nx)=(lat.len(),lon.len());
        if raw.len()!=bounds.len()/2*ny*nx || ny<2 || nx<2 {return Err("satellite QPE array shape differs from time x lat x lon".into())}
        let (dlat,dlon)=(lat[1]-lat[0],lon[1]-lon[0]);let cyclic=(dlon*nx as f64-360.).abs()<1e-3;
        for (i,s,e) in inside {covered+=e-s;let depth=if per_hour{(e-s) as f64/3600.}else{1.};let plane=&raw[i*ny*nx..(i+1)*ny*nx];
            for (k,&(la,lo)) in coordinates.iter().enumerate() {if !valid[k]{continue}
                let y=(la-lat[0])/dlat;let x=if cyclic{((lo-lon[0])/dlon).rem_euclid(nx as f64)}else{(lo-lon[0])/dlon};
                if y<0. || y>(ny-1) as f64 || x<0. || (!cyclic && x>(nx-1) as f64) {valid[k]=false;continue}
                let (iy,ix)=((y.floor() as usize).min(ny-2),x.floor() as usize);let (ty,tx)=(y-iy as f64,x-ix as f64);
                let jx=if cyclic{(ix+1)%nx}else{(ix+1).min(nx-1)};let mut sum=0.;
                for (idx,w) in [(iy*nx+ix,(1.-tx)*(1.-ty)),(iy*nx+jx,tx*(1.-ty)),((iy+1)*nx+ix,(1.-tx)*ty),((iy+1)*nx+jx,tx*ty)] {
                    if w<=1e-12 {continue}let r=plane[idx];if !r.is_finite() || missing.iter().any(|m|(r-m).abs()<0.5) {sum=f64::NAN;break}sum+=w*(r*scale+offset);
                }
                if sum.is_finite() && sum>= -1e-9 {observed[k]+=sum.max(0.)*depth}else{valid[k]=false}
            }
        }
        provenance.push(json!({"file":source_identity(path)?,"variable":var,"units":units,"scale_factor":scale,"add_offset":offset}));
    }
    if covered!=b-a {return Err(format!("satellite QPE slices cover {covered} s of the {} s model interval",b-a).into())}
    Ok((observed,valid,provenance))
}
fn run(request:&Value)->Result<Value> {
    let valid=request["valid_time"].as_str().ok_or("valid_time absent")?;let history=Path::new(request["history"].as_str().ok_or("history absent")?);let init=Path::new(request["init"].as_str().ok_or("init absent")?);
    let hex=Hex::open(history,init,request["diagnostics"].as_str().map(Path::new))?;let mut refs=Vec::new();let mut failures=Vec::new();
    let start=request["start"].as_str().ok_or("forecast start absent")?;
    for spec in request["references"].as_array().into_iter().flatten(){match reference(spec,LEVELS,valid,start,&request["grib_parameters"]){Ok(r)=>refs.push(r),Err(e)=>failures.push(json!({"label":spec["label"],"reason":e.to_string()}))}}
    let mut output=json!({"schema":"gpuwm-hex.native-observation-score/v1","valid_time":valid,"metrics":[],"paired":[],"failures":failures,"provenance":{"history":source_identity(history)?,"init":source_identity(init)?,"references":refs.iter().map(|r|json!({"label":r.label,"role":r.role,"artifact":r.provenance})).collect::<Vec<_>>()},"operators":{"hex":"three-nearest gnomonic barycentric, nearest fallback, log-pressure interpolation of zgrid midpoint height, sensible T and earth-relative winds","pressure":"station pressure from ASOS altimeter; hydrostatic height adjustment with terrain difference at most 50 m","common_support":"finite common observations per metric across available arms, no model-error-based observation censoring"}});
    if let Some(path)=request["diagnostics"].as_str(){output["provenance"]["diagnostics"]=source_identity(Path::new(path))?;}
    let region=Region::from_request(request)?;
    if let Some(r)=region {output["region"]=json!({"west":r.w,"east":r.e,"south":r.s,"north":r.n,"step_degrees":r.step,"selection":"radiosonde and surface reports inside the box; precipitation lattice spans the box"});}
    for (source,key) in [("radiosonde","sondes"),("asos","surface")] {
        let rows=if let Some(path)=request[key].as_str(){if source=="radiosonde"{sondes(Path::new(path),valid)}else{surface(Path::new(path),valid)}}else{Ok(Vec::new())};
        let rows=rows.map(|mut rows|{if let Some(r)=region{rows.retain(|o|r.contains(o.lat,o.lon))}rows});
        match rows {Ok(rows)=>{if let Some(path)=request[key].as_str(){output["provenance"][source]=source_identity(Path::new(path))?}let scores=score_observations(source,&rows,&hex,&refs);for key in ["metrics","paired"] {output[key].as_array_mut().unwrap().extend(scores[key].as_array().unwrap().clone())}},Err(e)=>output["failures"].as_array_mut().unwrap().push(json!({"source":source,"reason":e.to_string()}))}
    }
    if request["rain"].is_array() || request["qpe"].is_object() {let rain_source=request["qpe"]["label"].as_str().unwrap_or("mrms").to_string();match score_rain(request,&hex,&refs.iter().map(|r|r.label.clone()).collect()) {
        Ok(r)=>{output["metrics"].as_array_mut().unwrap().extend(r["metrics"].as_array().unwrap().clone());output["failures"].as_array_mut().unwrap().extend(r["failures"].as_array().unwrap().clone());output[rain_source.as_str()]=r;},
        Err(e)=>output["failures"].as_array_mut().unwrap().push(json!({"source":rain_source,"reason":e.to_string()}))}}
    Ok(output)
}
fn main() {let args=std::env::args().collect::<Vec<_>>();let result=(||->Result<()> {
    if args.get(1).map(String::as_str)==Some("--abi"){println!("gpuwm-hex.native-observation-score/v1");return Ok(())}
    if args.get(1).map(String::as_str)==Some("--inventory-grib") && args.len()==3 {
        let file=Grib2File::from_bytes(&std::fs::read(&args[2])?)?;
        let rows=file.messages.iter().filter(|m|m.product.parameter_category==1 || m.product.level_type==1 || m.product.level_type==100 && m.product.level_value<10000.).map(|m|json!({"discipline":m.discipline,"category":m.product.parameter_category,"parameter":m.product.parameter_number,"level_type":m.product.level_type,"level_value":m.product.level_value,"time_unit":m.product.time_range_unit,"forecast_time":m.product.forecast_time,"stat_process":m.product.statistical_process_type,"stat_hours":m.product.statistical_time_range_hours(),"end":m.product.end_of_interval.map(|v|v.to_string()),"product_debug":format!("{:?}",m.product)})).collect::<Vec<_>>();
        println!("{}",json!({"schema":"gpuwm-hex.grib-score-inventory/v1","artifact":source_identity(Path::new(&args[2]))?,"messages":rows}));return Ok(())
    }
    if args.get(1).map(String::as_str)==Some("--decode-sondes") && args.len()==4 {
        let path=Path::new(&args[2]);let bytes=std::fs::read(path)?;
        let mut file=rw_obs::ncep_bufr::read_file(&bytes,"global-sondes")?;
        let keep=file.messages.iter().filter(|m|file.template_of(m).mnemonic=="ADPUPA").map(|m|m.number).collect::<BTreeSet<_>>();
        file.messages.retain(|m|keep.contains(&m.number));
        let listing=rw_obs::prepbufr::dump(&file)?;
        std::fs::write(&args[3],&listing)?;
        println!("{}",json!({"schema":"gpuwm-hex.decoded-sondes/v1","producer":"rw_obs::ncep_bufr and rw_obs::prepbufr::dump","input":source_identity(path)?,"listing":source_identity(Path::new(&args[3]))?,"selected_message_type":"ADPUPA"}));return Ok(())
    }
    if args.get(1).map(String::as_str)==Some("--decode-igra") && args.len()>=6 {
        println!("{}",decode_igra(Path::new(&args[2]),&args[3],&args[4],&args[5..])?);return Ok(())
    }
    if args.get(1).map(String::as_str)==Some("--fetch-input") && args.len()==4 {
        use std::io::Read;
        let output=Path::new(&args[3]);if output.exists(){return Err("fetch destination exists; cannot overwrite a caller-owned observation".into())}
        let temp=output.with_extension("part");let mut response=rw_obs::net::agent().get(&args[2]).call()?;
        let mut writer=std::fs::File::create(&temp)?;let size=std::io::copy(&mut response.body_mut().as_reader().take(512*1024*1024+1),&mut writer)?;
        if size>512*1024*1024 {std::fs::remove_file(&temp)?;return Err("observation object exceeded the 512 MiB bounded transport".into())}
        writer.sync_all()?;drop(writer);std::fs::rename(temp,output)?;
        println!("{}",json!({"schema":"gpuwm-hex.fetched-observation/v1","url":args[2],"artifact":source_identity(output)?}));return Ok(())
    }
    if args.len()!=3 {return Err("usage: hex_obs_scorecard REQUEST.json OUTPUT.json".into())}
    let request=serde_json::from_slice(&std::fs::read(&args[1])?)?;let out=run(&request)?;std::fs::write(&args[2],serde_json::to_vec_pretty(&out)?)?;Ok(())})();
    if let Err(e)=result{eprintln!("hex_obs_scorecard: {e}");std::process::exit(2)}
}
#[cfg(test)] mod tests {use super::*;
    #[test] fn spherical_nearest_wraps_and_barycentric_reproduces_constant(){let la:Vec<f64>=vec![0.,0.,0.1,1.];let lo:Vec<f64>=vec![179.9,-179.9,180.,179.];let m=Mesh::new(&la.iter().map(|v|v.to_radians()).collect::<Vec<_>>(),&lo.iter().map(|v|v.to_radians()).collect::<Vec<_>>());assert_eq!(m.nearest(xyz(0.,-179.91),1)[0].1,1);assert!((m.weights(0.01,179.98).iter().map(|p|p.1).sum::<f64>()-1.).abs()<1e-12);}
    #[test] fn default_region_reproduces_the_original_conus_lattice(){
        assert_eq!(Region::from_request(&json!({})).unwrap(),None);assert_eq!(CONUS.shape(),(473,225));
        let c=CONUS.coordinates();let old=(0..225).flat_map(|y|(0..473).map(move|x|(22.+y as f64*0.125,-125.+x as f64*0.125))).collect::<Vec<_>>();
        assert!(c.iter().zip(&old).all(|(a,b)|a.0.to_bits()==b.0.to_bits() && a.1.to_bits()==b.1.to_bits()) && c.len()==old.len());
        assert_eq!((111.195*CONUS.step*((CONUS.s+CONUS.n)/2.).to_radians().cos()).to_bits(),(111.195*0.125*36_f64.to_radians().cos()).to_bits());
    }
    #[test] fn region_selects_inside_the_box_across_longitude_conventions(){
        let r=Region::from_request(&json!({"region":{"west":116.,"east":127.5,"south":4.,"north":22.}})).unwrap().unwrap();
        assert!(r.contains(14.5,121.)&&r.contains(14.5,121.-360.)&&!r.contains(30.,121.)&&!r.contains(14.5,-97.));
        assert!(Region::from_request(&json!({"region":{"west":10.,"east":5.,"south":0.,"north":1.}})).is_err());
    }
    #[test] fn neutral_sonde_table_keeps_mandatory_levels_at_the_nominal_hour(){
        let t="station_id,latitude_deg,longitude_deg,elevation_m,valid_time,pressure_hpa,variable,value\nA,14.5,121.3,0,2025-11-03T00:00:00Z,500,temperature_k,268.1\nA,14.5,121.3,0,2025-11-03T00:00:00Z,501,temperature_k,268.0\nA,14.5,121.3,0,2025-11-03T12:00:00Z,850,height_m,1500\nA,14.5,121.3,0,2025-11-03T00:00:00Z,850,wind_u_m_s,-3\n";
        let rows=neutral_sondes(t,"2025-11-03T00:00:00Z").unwrap();assert_eq!(rows.len(),2);assert_eq!((rows[0].p,rows[0].q.as_str()),(500,"temperature"));assert_eq!(rows[1].q,"u");
    }
    #[test] fn igra_fixed_width_rows_decode_to_neutral_sondes(){
        let dir=std::env::temp_dir().join(format!("igra-test-{}",std::process::id()));std::fs::create_dir_all(&dir).unwrap();
        let input=dir.join("x.txt");let out=dir.join("x.csv");
        // IGRA v2 data row: LVLTYP 1-2, ETIME 4-8, PRESS 10-15, PFLAG 16, GPH 17-21, ZFLAG 22, TEMP 23-27, TFLAG 28, RH 29-33, DPDP 35-39, WDIR 41-45, WSPD 47-51.
        let row=|p:i64,z:i64,t:i64,d:i64,s:i64|format!("10 {:>5} {:>6}A{:>5}B{:>5}B{:>5} {:>5} {:>5} {:>5}\n",-9999,p,z,t,10,80,d,s);
        let header=|h:&str|format!("#RPM00098433 2025 11 03 {h} 2315    3 ncdc-gts ncdc-gts  145812  1213693\n");
        let text=header("00")+&row(50000,5850,-105,270,150)+&row(50100,5860,-106,270,150)+&row(85000,-8888,-9999,-9999,-9999)+&header("99")+&row(50000,5850,-105,270,150);
        assert_eq!(&text[72+9..72+15]," 50000");assert_eq!(&text[72+46..72+51],"  150");assert_eq!(&text[55..62]," 145812");
        std::fs::write(&input,text).unwrap();
        let r=decode_igra(&out,"2025-11-03T00:00:00Z","2025-11-03T00:00:00Z",&[input.to_string_lossy().into()]).unwrap();
        assert_eq!(r["soundings"],1);let rows=neutral_sondes(&std::fs::read_to_string(&out).unwrap(),"2025-11-03T00:00:00Z").unwrap();
        let get=|q:&str|rows.iter().find(|o|o.q==q&&o.p==500).unwrap().val;
        assert_eq!(get("height"),5850.);assert!((get("temperature")-(262.65)).abs()<1e-9);assert!((get("u")-15.).abs()<1e-9 && get("v").abs()<1e-9);
        assert!(rows.iter().all(|o|o.p!=850));std::fs::remove_dir_all(dir).ok();
    }
    #[test] fn igra_station_zips_decode_like_their_text(){
        // The engine's rw_igra2 fetch keeps NCEI's .txt.zip files; deflated and stored members decode to the same table.
        let dir=std::env::temp_dir().join(format!("igra-zip-test-{}",std::process::id()));std::fs::create_dir_all(&dir).unwrap();
        let row=|p:i64,z:i64,t:i64,d:i64,s:i64|format!("10 {:>5} {:>6}A{:>5}B{:>5}B{:>5} {:>5} {:>5} {:>5}\n",-9999,p,z,t,10,80,d,s);
        let text=format!("#RPM00098433 2025 11 03 00 2315    3 ncdc-gts ncdc-gts  145812  1213693\n")+&row(50000,5850,-105,270,150)+&row(85000,1500,150,180,50);
        let zip=|method:u16,data:&[u8]|{let name=b"RPM00098433-data-beg2025.txt";let mut z=b"PK\x03\x04".to_vec();
            z.extend_from_slice(&20u16.to_le_bytes());z.extend_from_slice(&0u16.to_le_bytes());z.extend_from_slice(&method.to_le_bytes());z.extend_from_slice(&[0;8]);
            z.extend_from_slice(&(data.len() as u32).to_le_bytes());z.extend_from_slice(&(text.len() as u32).to_le_bytes());
            z.extend_from_slice(&(name.len() as u16).to_le_bytes());z.extend_from_slice(&0u16.to_le_bytes());z.extend_from_slice(name);z.extend_from_slice(data);
            z.extend_from_slice(b"PK\x01\x02");z};
        let mut deflated=Vec::new();{use std::io::Write;let mut e=flate2::write::DeflateEncoder::new(&mut deflated,flate2::Compression::default());e.write_all(text.as_bytes()).unwrap();e.finish().unwrap();}
        let (plain,stored,packed)=(dir.join("a.txt"),dir.join("b.txt.zip"),dir.join("c.txt.zip"));
        std::fs::write(&plain,&text).unwrap();std::fs::write(&stored,zip(0,text.as_bytes())).unwrap();std::fs::write(&packed,zip(8,&deflated)).unwrap();
        let table=|input:&Path|{let out=dir.join("t.csv");decode_igra(&out,"2025-11-03T00:00:00Z","2025-11-03T00:00:00Z",&[input.to_string_lossy().into()]).unwrap();std::fs::read_to_string(&out).unwrap()};
        let reference=table(&plain);assert!(reference.lines().count()>4);assert_eq!(table(&stored),reference);assert_eq!(table(&packed),reference);
        std::fs::write(dir.join("d.txt.zip"),zip(9,b"x")).unwrap();assert!(igra_text(&dir.join("d.txt.zip")).is_err());
        std::fs::remove_dir_all(dir).ok();
    }
    #[test] fn empty_statistics_never_emit_nan(){assert_eq!(statistics(&[])["rmse"],Value::Null);let s=statistics(&[("x".into(),4.,-2.)]);assert_eq!(s["rmse"],2.);assert_eq!(s["bias"],-2.);}
    #[test] fn global_bilinear_wraps_both_prime_meridian_and_dateline(){let plane=Plane{grid:GridData{lat:vec![10.,10.,10.,10.,-10.,-10.,-10.,-10.],lon:vec![0.,90.,180.,270.,0.,90.,180.,270.],nx:4,ny:2,projection:None},values:vec![0.,1.,2.,3.,0.,1.,2.,3.]};assert!((plane.sample(0.,-45.).unwrap()-1.5).abs()<1e-12);assert_eq!(plane.sample(0.,-180.),Some(2.));assert_eq!(plane.sample(11.,0.),None);}
    #[test] fn native_fss_uses_complete_support_and_declines_dry_significance(){let obs=vec![0.,0.,0.,0.,5.,0.,0.,0.,0.];let valid=vec![true;9];let arms=vec![("x".into(),obs.clone())];let score=paired_fss_rect("rain","mm",1.,3.,1.,1.,3,3,&obs,&valid,&arms).unwrap();assert_eq!(score.arms[0].fss,Some(1.));let mut masked=valid.clone();masked[0]=false;let score=paired_fss_rect("rain","mm",1.,3.,1.,1.,3,3,&obs,&masked,&arms).unwrap();assert_eq!(score.arms[0].count,0);}
    #[test] fn reference_level_interpolation_is_log_pressure_without_extrapolation(){
        let plane=|v|Plane{grid:GridData{lat:vec![10.,10.,10.,10.,-10.,-10.,-10.,-10.],lon:vec![0.,90.,180.,270.,0.,90.,180.,270.],nx:4,ny:2,projection:None},values:vec![v;8]};
        let reference=Reference{label:"arbitrary".into(),role:"forecast".into(),fields:BTreeMap::from([("temperature:1000".into(),plane(300.)),("temperature:500".into(),plane(250.))]),provenance:json!({})};
        let mut o=Ob{id:"s".into(),lat:0.,lon:-40.,p:750,q:"temperature".into(),val:280.,elev:0.};
        assert!((reference.sample(&o).unwrap()-(300.-50.*(0.75_f64.ln()/0.5_f64.ln()))).abs()<1e-12);o.p=1100;assert_eq!(reference.sample(&o),None);o.p=400;assert_eq!(reference.sample(&o),None);
    }
}
