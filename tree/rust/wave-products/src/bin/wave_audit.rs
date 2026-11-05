//! Native parallel reductions for coupled-wave evidence and coastal sampling.
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{fs, path::Path, error::Error, io::{Read, Seek, SeekFrom}};
type R<T> = Result<T, Box<dyn Error + Send + Sync>>;
fn floats(spec: &Value) -> R<Vec<f32>> {
    let path=spec.as_str().or_else(||spec["path"].as_str()).ok_or("array path required")?;
    let mut file=fs::File::open(path)?;
    file.seek(SeekFrom::Start(spec["offset_bytes"].as_u64().unwrap_or(0)))?;
    let mut bytes=Vec::new();
    file.take(spec["bytes"].as_u64().unwrap_or(u64::MAX)).read_to_end(&mut bytes)?;
    if bytes.len()%4 != 0 { return Err("unaligned float input".into()); }
    Ok(bytes.par_chunks_exact(4).map(|b| if spec["dtype"]=="i4" { i32::from_le_bytes(b.try_into().unwrap()) as f32 } else { f32::from_le_bytes(b.try_into().unwrap()) }).collect())
}
fn xyz(lat: f64, lon: f64) -> [f64;3] {
    let (s,c)=lat.to_radians().sin_cos(); let (y,x)=lon.to_radians().sin_cos(); [c*x,c*y,s]
}
fn distance(a: [f64;3], b: [f64;3]) -> f64 {
    let chord = a.iter().zip(b.iter()).map(|(x,y)|(x-y).powi(2)).sum::<f64>().sqrt();
    12742.0*(chord/2.).min(1.).asin()
}
fn destination(lat:f64,lon:f64,bearing:f64,km:f64)->[f64;3] {
    let p=lat.to_radians();let l=lon.to_radians();let a=bearing.to_radians();let d=km/6371.;
    let q=(p.sin()*d.cos()+p.cos()*d.sin()*a.cos()).clamp(-1.,1.).asin();
    let r=l+(a.sin()*d.sin()*p.cos()).atan2(d.cos()-p.sin()*q.sin());
    xyz(q.to_degrees(),r.to_degrees())
}
#[derive(Clone,Copy)]
struct Point {p:[f64;3], index:usize}
struct Node {point:Point,axis:usize,left:Option<Box<Node>>,right:Option<Box<Node>>}
impl Node {
    fn build(points:&mut[Point],depth:usize)->Option<Box<Self>> {
        if points.is_empty(){return None}
        let axis=depth%3;let middle=points.len()/2;
        points.select_nth_unstable_by(middle,|a,b|a.p[axis].total_cmp(&b.p[axis]));
        let (left,tail)=points.split_at_mut(middle);let (point,right)=tail.split_first_mut().unwrap();
        let point=*point;
        let (left,right)=if left.len()+right.len()>2048 {
            rayon::join(||Self::build(left,depth+1),||Self::build(right,depth+1))
        }else{(Self::build(left,depth+1),Self::build(right,depth+1))};
        Some(Box::new(Self{point,axis,left,right}))
    }
    fn nearest(&self,p:[f64;3],best:&mut(f64,usize)) {
        let q=self.point.p;let distance=(q[0]-p[0]).powi(2)+(q[1]-p[1]).powi(2)+(q[2]-p[2]).powi(2);
        if distance<best.0 {*best=(distance,self.point.index);}
        let delta=p[self.axis]-q[self.axis];
        let (first,last)=if delta<0. {(&self.left,&self.right)}else{(&self.right,&self.left)};
        if let Some(node)=first {node.nearest(p,best);}
        if delta*delta<=best.0 {if let Some(node)=last {node.nearest(p,best);}}
    }
}
fn main()->R<()> {
    let request=std::env::args().nth(1).ok_or("wave_audit REQUEST.json")?;
    let r:Value=serde_json::from_slice(&fs::read(request)?)?;
    let threads=r["threads"].as_u64().unwrap_or(24) as usize;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    let load=|k:&str|->R<Vec<f32>> {floats(&r[k])};
    let lat=load("latitude")?;let lon=load("longitude")?;let wet=load("wet")?;
    let n=lat.len();if n!=lon.len()||n!=wet.len(){return Err("coordinate extents differ".into());}
    let result=match r["operation"].as_str().ok_or("missing operation")? {
        "sample" => {
            let indices=r["indices"].as_array().ok_or("indices required")?;
            let mut sampled=serde_json::Map::new();
            for (key,spec) in r["fields"].as_object().ok_or("fields required")? {
                let values=floats(spec)?;
                if values.len()!=n{return Err("sample field extent differs".into());}
                let rows:Vec<R<Value>>=indices.par_iter().map(|row|{
                    let i=row["cell"].as_u64().ok_or("cell required")? as usize;
                    if i>=n{return Err("sample cell outside mesh".into());}
                    Ok(json!({"label":row["label"],"cell":i,"value":values[i]}))
                }).collect();
                sampled.insert(key.clone(),Value::Array(rows.into_iter().collect::<R<_>>()?));
            }
            json!({"fields":sampled})
        },
        "wind" => {
            let u=load("u")?;let v=load("v")?;
            if u.len()!=n||v.len()!=n{return Err("wind extents differ".into());}
            let bounds=r["bounds"].as_array().ok_or("bounds required")?;
            let b:Vec<f64>=bounds.iter().map(|x|x.as_f64().unwrap()).collect();
            let values:Vec<_>=(0..n).into_par_iter().filter_map(|i|{
                let x=(lon[i] as f64-b[0]).rem_euclid(360.)+b[0];
                (wet[i]>0.5 && x<=b[1] && lat[i] as f64>=b[2] && lat[i] as f64<=b[3]
                    && u[i].is_finite() && v[i].is_finite()).then_some((u[i].hypot(v[i]),i))
            }).collect();
            let &(max,i)=values.iter().max_by(|a,b|a.0.total_cmp(&b.0)).ok_or("empty wind region")?;
            json!({"maximum_m_s":max,"latitude":lat[i],"longitude":lon[i],"cells":values.len(),"bounds":b})
        },
        "stations" => {
            let points:Vec<_>=(0..n).into_par_iter().map(|i|xyz(lat[i] as f64,lon[i] as f64)).collect();
            let stations=r["stations"].as_array().ok_or("stations required")?;
            let mut indexed:Vec<_>=points.iter().enumerate().map(|(index,&p)|Point{p,index}).collect();
            let tree=Node::build(&mut indexed,0).ok_or("empty point tree")?;
            let depth=if r["depth"].is_null(){vec![f32::NAN;n]}else{load("depth")?};
            if depth.len()!=n{return Err("depth extent differs".into());}
            let area=if r["area"].is_null(){None}else{Some(load("area")?)};
            if let Some(ref a)=area {if a.len()!=n || a.iter().any(|x|!x.is_finite() || *x<=0.){return Err("station cell area must be finite positive and match mesh".into());}}
            let direction_data=if r["directions"].is_null(){None}else{Some(load("directions")?)};
            let nt=r["direction_times"].as_u64().unwrap_or(0) as usize;
            if let Some(ref data)=direction_data {if nt==0 || data.len()!=stations.len()*nt{return Err("station direction extent differs".into());}}
            let radius=r["max_distance_km"].as_f64().unwrap_or(150.);
            let rows:Vec<Value>=stations.par_iter().enumerate().map(|(station_index,station)|{
                let origin=xyz(station["lat"].as_f64().unwrap(),station["lon"].as_f64().unwrap());
                let directions:Vec<f64>=if let Some(ref data)=direction_data {
                    data[station_index*nt..(station_index+1)*nt].iter().filter(|x|x.is_finite()).map(|&x|x as f64).collect()
                }else{station["incoming_from_degrees"].as_array().unwrap().iter().filter_map(Value::as_f64).filter(|x|x.is_finite()).collect()};
                let mut candidates:Vec<_>=(0..n).filter(|&i|wet[i]>0.5).map(|i|(distance(origin,points[i]),i))
                    .filter(|(d,_)|*d<=radius).collect();
                candidates.sort_by(|a,b|a.0.total_cmp(&b.0));
                // This tests unresolved coastal shielding over a few local cells.
                // A fixed 300 km ray on a fine mesh instead rejects legitimate
                // finite-fetch seas by reaching an unrelated distant coastline.
                let ray_extent_km=candidates.first().and_then(|(_,i)|area.as_ref().map(|a|
                    (2.5*(a[*i] as f64).sqrt()/1000.).min(300.))).unwrap_or(300.);
                let candidates:Vec<Value>=candidates.iter().map(|&(d,i)|{
                    let mut total=0usize;let mut clear=0usize;
                    for &bearing in &directions { for offset in [-20.,0.,20.] { for fraction in [0.25,0.5,0.75,1.] {
                        let km=ray_extent_km*fraction;
                        let p=destination(lat[i] as f64,lon[i] as f64,bearing+offset,km);
                        let mut nearest=(f64::INFINITY,0);tree.nearest(p,&mut nearest);
                        let nearest=nearest.1;
                        total+=1;if wet[nearest]>0.5 {clear+=1;}
                    }}}
                    json!({"cell":i,"distance_km":d,"lat":lat[i],"lon":lon[i],"depth_m":depth[i],"exposed_samples":clear,"total_samples":total,"exposed_fraction":clear as f64/total.max(1) as f64})
                }).collect();
                let fully_exposed=candidates.iter().find(|c|c["total_samples"].as_u64().unwrap()>0 && c["exposed_samples"]==c["total_samples"]);
                let selected=fully_exposed.or_else(||candidates.iter().max_by(|a,b|{
                    a["exposed_fraction"].as_f64().unwrap().total_cmp(&b["exposed_fraction"].as_f64().unwrap())
                    .then_with(||b["distance_km"].as_f64().unwrap().total_cmp(&a["distance_km"].as_f64().unwrap()))
                }));
                json!({"station":station,"candidates":candidates,"selected":selected,
                    "fully_exposed":fully_exposed.is_some(), "ray_extent_km":ray_extent_km,
                    "rule":"nearest wet cell with every incident ray sample wet; if none, greatest exposed fraction then nearest; prepared reference FROM directions, +/-20 degrees, four equally spaced samples through 2.5 local sqrt(area) cell widths capped at 300 km; no observations used"})
            }).collect();
            json!({"stations":rows,"radius_km":radius})
        },
        _=>return Err("unknown audit operation".into())
    };
    let out=Path::new(r["output"].as_str().ok_or("output required")?);
    fs::write(out,serde_json::to_vec_pretty(&result)?)?;Ok(())
}
