//! Original Apache-2.0. Native cell-constant visualization, never new physics.
//! Rust owns spherical lookup, polygon masks and direction-arrow geometry.
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{error::Error, fs, path::{Path, PathBuf}, time::Instant};
type R<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn text<'a>(v: &'a Value, k: &str) -> R<&'a str> {
    v[k].as_str().ok_or_else(|| format!("missing string {k}").into())
}
fn f32s(p: &Path) -> R<Vec<f32>> {
    let b=fs::read(p)?;
    if b.len()%4!=0 {return Err("unaligned float32 input".into());}
    Ok(b.par_chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect())
}
fn f64s(p: &Path) -> R<Vec<f64>> {
    let b=fs::read(p)?;
    if b.len()%8!=0 {return Err("unaligned float64 input".into());}
    Ok(b.par_chunks_exact(8).map(|x|f64::from_le_bytes(x.try_into().unwrap())).collect())
}
fn i32s(p: &Path) -> R<Vec<i32>> {
    let b=fs::read(p)?;
    if b.len()%4!=0 {return Err("unaligned int32 input".into());}
    Ok(b.par_chunks_exact(4).map(|x|i32::from_le_bytes(x.try_into().unwrap())).collect())
}
fn write(p:&Path, values:&[f32])->R<()> {
    let mut bytes=vec![0_u8;values.len()*4];
    bytes.par_chunks_mut(4).zip(values.par_iter()).for_each(|(b,x)|b.copy_from_slice(&x.to_le_bytes()));
    fs::write(p,bytes)?;Ok(())
}
fn xyz(lat:f64,lon:f64)->[f64;3] {let (s,c)=lat.sin_cos();let (y,x)=lon.sin_cos();[c*x,c*y,s]}
fn dot(a:[f64;3],b:[f64;3])->f64 {a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
fn cross(a:[f64;3],b:[f64;3])->[f64;3] {[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
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
fn destination(lat:f64,lon:f64,bearing:f64,angle:f64)->(f64,f64) {
    let phi=lat.to_radians();let lambda=lon.to_radians();let a=bearing.to_radians();
    let p=(phi.sin()*angle.cos()+phi.cos()*angle.sin()*a.cos()).clamp(-1.,1.).asin();
    let l=lambda+(a.sin()*angle.sin()*phi.cos()).atan2(angle.cos()-phi.sin()*p.sin());
    (p.to_degrees(),(l.to_degrees()+180.).rem_euclid(360.)-180.)
}
fn main()->R<()> {
    let began=Instant::now();
    let request=PathBuf::from(std::env::args().nth(1).ok_or("rw_hex_wave_products REQUEST.json")?);
    let r:Value=serde_json::from_slice(&fs::read(&request)?)?;
    let threads=r["threads"].as_u64().unwrap_or(std::thread::available_parallelism()?.get() as u64) as usize;
    if !(1..=256).contains(&threads){return Err("threads must be 1..256".into());}
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    let root=request.parent().unwrap_or(Path::new("."));
    let path=|key:&str|->R<PathBuf>{Ok(root.join(text(&r,key)?))};
    let lat=f32s(&path("latitude")?)?;let lon=f32s(&path("longitude")?)?;let wet=f32s(&path("wet")?)?;
    let n=lat.len();if n==0||lon.len()!=n||wet.len()!=n{return Err("point array extents differ".into());}
    if lat.par_iter().zip(lon.par_iter()).any(|(&a,&b)|!a.is_finite()||a.abs()>90.||!b.is_finite()){
        return Err("nonfinite or invalid point coordinates".into());}
    let vertices=i32s(&path("vertices_on_cell")?)?;let counts=i32s(&path("edge_counts")?)?;
    let vlat=f64s(&path("vertex_latitude")?)?;let vlon=f64s(&path("vertex_longitude")?)?;
    if counts.len()!=n||vertices.len()%n!=0||vlat.len()!=vlon.len(){return Err("polygon array extents differ".into());}
    let me=vertices.len()/n;
    if counts.iter().any(|&c|c<3||c as usize>me){return Err("invalid polygon edge count".into());}
    let centers:Vec<_>=lat.par_iter().zip(lon.par_iter()).map(|(&a,&b)|xyz((a as f64).to_radians(),(b as f64).to_radians())).collect();
    let nodes:Vec<_>=vlat.par_iter().zip(vlon.par_iter()).map(|(&a,&b)|xyz(a,b)).collect();
    let polygons:Vec<R<Vec<[f64;3]>>>=(0..n).into_par_iter().map(|c|{
        (0..counts[c] as usize).map(|j|{
            let a=vertices[c*me+j]-1;let b=vertices[c*me+(j+1)%counts[c] as usize]-1;
            if a<0||b<0||a as usize>=nodes.len()||b as usize>=nodes.len(){return Err("polygon vertex index outside mesh".into());}
            let mut plane=cross(nodes[a as usize],nodes[b as usize]);
            if dot(plane,centers[c])<0.{plane.iter_mut().for_each(|v|*v = -*v);}
            Ok(plane)
        }).collect()
    }).collect();
    let polygons:Vec<_>=polygons.into_iter().collect::<R<_>>()?;
    let mut points:Vec<_>=centers.par_iter().enumerate().map(|(index,&p)|Point{p,index}).collect();
    let tree=Node::build(&mut points,0).ok_or("empty point tree")?;
    let b:Vec<f64>=if r["bounds"].is_null(){
        let sum=centers.par_iter().copied().reduce(||[0.;3],|a,b|[a[0]+b[0],a[1]+b[1],a[2]+b[2]]);
        let center=sum[1].atan2(sum[0]).to_degrees();
        let range=(0..n).into_par_iter().map(|i|{
            let x=(lon[i] as f64-center+180.).rem_euclid(360.)-180.+center;
            [x,x,lat[i] as f64,lat[i] as f64]
        }).reduce(||[f64::INFINITY,f64::NEG_INFINITY,f64::INFINITY,f64::NEG_INFINITY],
            |a,b|[a[0].min(b[0]),a[1].max(b[1]),a[2].min(b[2]),a[3].max(b[3])]);
        let pad_x=((range[1]-range[0])*0.03).max(0.05);
        let pad_y=((range[3]-range[2])*0.03).max(0.05);
        vec![range[0]-pad_x,range[1]+pad_x,(range[2]-pad_y).max(-89.5),(range[3]+pad_y).min(89.5)]
    }else{
        let bounds=r["bounds"].as_array().ok_or("bounds W,E,S,N required")?;
        if bounds.len()!=4{return Err("bounds need four entries".into());}
        bounds.iter().map(|v|v.as_f64().ok_or("nonnumeric bounds")).collect::<Result<_,_>>()?
    };
    let nx=r["nx"].as_u64().ok_or("nx required")? as usize;
    let ny=r["ny"].as_u64().ok_or("ny required")? as usize;
    if nx<2||ny<2||nx*ny>4_000_000||b.iter().any(|v|!v.is_finite())||b[1]<=b[0]||b[3]<=b[2]||b[2]< -90.||b[3]>90.{
        return Err("invalid product grid".into());}
    // A periodic longitude axis contains one copy of the seam. Including
    // both -180 and +180 makes CF normalization produce duplicate -180
    // columns, a zero first spacing and a blank regular-grid raster.
    let periodic=(b[1]-b[0]-360.).abs()<1.0e-6;
    let dx=(b[1]-b[0])/(if periodic {nx} else {nx-1}) as f64;
    let x0=b[0]+if periodic {0.5*dx} else {0.};
    let dy=(b[3]-b[2])/(ny-1) as f64;
    let index:Vec<Option<usize>>=(0..nx*ny).into_par_iter().map(|i|{
        let a=b[2]+(i/nx) as f64*dy;let o=x0+(i%nx) as f64*dx;let p=xyz(a.to_radians(),o.to_radians());
        let mut best=(f64::INFINITY,0);tree.nearest(p,&mut best);let c=best.1;
        (wet[c]>0.5&&polygons[c].iter().all(|&plane|dot(plane,p)>=-1.0e-12)).then_some(c)
    }).collect();
    let out=root.join(text(&r,"output")?);fs::create_dir_all(&out)?;
    write(&out.join("latitude.bin"),&(0..ny).map(|j|(b[2]+j as f64*dy) as f32).collect::<Vec<_>>())?;
    write(&out.join("longitude.bin"),&(0..nx).map(|j|(x0+j as f64*dx) as f32).collect::<Vec<_>>())?;
    let fields=r["fields"].as_object().ok_or("field metadata required")?;
    let mapped:Vec<R<(String,Value)>>=fields.iter().collect::<Vec<_>>().into_par_iter().map(|(name,source)|{
        let values=f32s(&root.join(source.as_str().ok_or("field path required")?))?;
        if values.len()!=n{return Err("field extent differs from point grid".into());}
        let plane:Vec<_>=index.par_iter().map(|i|i.map_or(f32::NAN,|c|values[c])).collect();
        let leaf=format!("{name}.bin");write(&out.join(&leaf),&plane)?;
        Ok((name.clone(),json!({"path":leaf,"units":r["units"][name]})))
    }).collect();
    let mapped:serde_json::Map<String,Value>=mapped.into_iter().collect::<R<_>>()?;
    let direction=f32s(&root.join(text(&r,"peak_direction")?))?;
    let height=f32s(&root.join(text(&r,"height")?))?;
    if direction.len()!=n||height.len()!=n{return Err("direction/height extent differs".into());}
    let stride=(nx/32).max(1);let stride_y=(ny/20).max(1);
    let mut lines=Vec::new();
    for y in (stride_y/2..ny).step_by(stride_y){for x in (stride/2..nx).step_by(stride){
        let Some(c)=index[y*nx+x] else{continue};if !direction[c].is_finite()||!height[c].is_finite()||height[c]<0.1{continue}
        let a=b[2]+y as f64*dy;let o=x0+x as f64*dx;
        let bearing=(direction[c] as f64+180.).rem_euclid(360.);
        // Longitude spacing contracts toward a pole. Size the arrow against
        // physical separation so it does not span neighbouring meridians.
        let angle=0.28*(stride as f64*dx*a.to_radians().cos().abs()).min(stride_y as f64*dy).to_radians();
        let tip=destination(a,o,bearing,angle);let tail=destination(a,o,bearing+180.,angle);
        let wing1=destination(tip.0,tip.1,bearing+150.,angle*0.65);
        let wing2=destination(tip.0,tip.1,bearing-150.,angle*0.65);
        for p in [vec![tail,tip],vec![wing1,tip,wing2]]{
            if p.windows(2).all(|v|(v[0].1-v[1].1).abs()<180.){lines.push(json!({"points":p,"color":"#222222","width":1,"closed":false}));}
        }
    }}
    fs::write(out.join("directions.json"),serde_json::to_vec(&json!({"lines":lines}))?)?;
    let receipt=json!({"schema":"gpuwm-hex.wave-product-grid/v1","status":"complete","nx":nx,"ny":ny,"bounds":b,
        "fields":mapped,"finite_cells":index.iter().filter(|i|i.is_some()).count(),"arrow_segments":lines.len(),
        "parallelism":"Rayon tree construction, target cells and fields","threads":threads,"wall_seconds":began.elapsed().as_secs_f64(),
        "reconstruction":"native Voronoi cell constant with spherical polygon containment; no smoothing"});
    fs::write(out.join("receipt.json"),serde_json::to_vec_pretty(&receipt)?)?;println!("{receipt}");Ok(())
}
