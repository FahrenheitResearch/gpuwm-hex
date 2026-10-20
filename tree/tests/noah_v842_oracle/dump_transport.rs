//! f1 typed dump -> packed CUDA ABI -> typed output. Numeric I/O stays in Rust.
//! Format authority: f1 native_physics_reference/oracle/oracle_words.rs.
//! Driver layout: MPAS v8.4.2 mpas_atmphys_driver_lsm.F:705-749.
use std::{collections::HashMap,env,fs,path::Path,process::Command};
include!("table_schema.rs");
include!("dump_layout.rs");
const FIXED:usize=8+21*4+3*256;
const MODULE_LOCATIONS:[&str;4]=["state.module_sf_noahlsm.iloc",
    "state.module_sf_noahlsm.jloc","state.module_sf_noahlsm_glacial_only.iloc",
    "state.module_sf_noahlsm_glacial_only.jloc"];
fn store(data:&mut [u8],name:&str,index:usize,value:&str){
    let (offset,count,kind)=field(name);assert!(index<count);
    let bytes=if kind=='i'{value.parse::<i32>().unwrap().to_le_bytes()}else{assert_eq!(kind,'f');value.replace(['d','D'],"E").parse::<f32>().unwrap().to_le_bytes()};
    data[offset+index*4..offset+index*4+4].copy_from_slice(&bytes);
}
#[derive(Clone)]
struct Record { raw:Vec<u8>, h:[i32;21], at:usize }
impl Record {
    fn read(path:&Path)->Self {
        let raw=fs::read(path).unwrap();assert!(raw.len()>=FIXED);assert_eq!(&raw[..8],b"ORCL0001");
        let mut h=[0;21];for (i,v) in h.iter_mut().enumerate(){*v=i32::from_le_bytes(raw[8+i*4..12+i*4].try_into().unwrap());}
        assert!((0..=5).contains(&h[2]) && h[3..8].iter().all(|v|*v>=0));
        assert!(h[16]>=0 && h[19]==256 && h[20]==0);
        let width=match h[0]{1|3|4=>4,2|6=>8,5=>h[1],_=>panic!("unknown type")};assert!(width>0);assert_eq!(h[1],width);
        assert!(h[17]>=0 && h[17]<=h[2] && h[18]>=0 && h[18]<=h[2]);
        let at=FIXED+usize::try_from(h[16]).unwrap()*4;
        let elements=h[3..8].iter().try_fold(1usize,|n,d|n.checked_mul(*d as usize)).unwrap();
        assert_eq!(raw.len(),at+elements.checked_mul(width as usize).unwrap());Self{raw,h,at}
    }
    fn offset(&self,i:i32,j:i32,k:usize)->usize {
        let p=match self.h[2]{
            0=>0,
            1=>k,
            2=>usize::try_from(i-self.h[8]).unwrap()+self.h[3] as usize*usize::try_from(j-self.h[9]).unwrap(),
            3=>usize::try_from(i-self.h[8]).unwrap()+self.h[3] as usize*(usize::try_from(k as i32+1-self.h[9]).unwrap()+self.h[4] as usize*usize::try_from(j-self.h[10]).unwrap()),
            _=>panic!("Noah argument rank unsupported")};
        let at=self.at+p*self.h[1] as usize;assert!(at+4<=self.raw.len());at
    }
    fn word(&self,i:i32,j:i32,k:usize)->[u8;4]{let at=self.offset(i,j,k);self.raw[at..at+4].try_into().unwrap()}
    fn integer(&self)->i32{assert!([3,4].contains(&self.h[0]) && self.h[2]==0);i32::from_le_bytes(self.word(0,0,0))}
}
fn records(prefix:&str)->HashMap<String,Record>{
    let path=Path::new(prefix);let dir=path.parent().unwrap();let stem=path.file_name().unwrap().to_str().unwrap();
    let marker=format!("{stem}.pre.");let mut records=HashMap::new();
    for item in fs::read_dir(dir).unwrap(){let item=item.unwrap();let name=item.file_name();let name=name.to_str().unwrap();
        if let Some(rest)=name.strip_prefix(&marker).and_then(|s|s.strip_suffix(".bin")){records.insert(rest.into(),Record::read(&item.path()));}
    }records
}
fn columns(r:&HashMap<String,Record>)->Vec<(i32,i32)>{
    let get=|name:&str|r[name].integer();
    assert_eq!(get("num_soil_layers"),4,"packed MPAS Noah uses four soil layers");
    assert_eq!(get("kts"),1,"MPAS lowest-layer pressure convention, Noah driver:788-791");
    // These are MPAS compile-time options; refusing a WRF-only capture prevents
    // silently replaying a different algorithm (driver_lsm.F:32-38,705-749).
    for name in ["myj","frpcpn","rdlai2d","ua_phys","fasdas","sf_urban_physics"]{assert_eq!(get(name),0,"capture enables WRF-only option {name}");}
    (get("jts")..=get("jte")).flat_map(|j|(get("its")..=get("ite")).map(move|i|(i,j))).collect()
}
fn main(){
    let a:Vec<_>=env::args().collect();
    if a.get(1).map(String::as_str)==Some("fixture"){fixture(&a);return;}
    assert!(a.len()>=5,"pack INPUT_PREFIX PACKED_PREFIX TABLE_DECODER TABLE_DIRECTORY | unpack INPUT_PREFIX OUTPUT_PREFIX PACKED_PREFIX OUTPUT_NAMES");
    let r=records(&a[2]);let cells=columns(&r);
    match a[1].as_str(){
        "pack"=>{
            assert_eq!(a.len(),6);let mut floats=Vec::new();let mut ints=Vec::new();
            for &(i,j) in &cells{
                for &(name,k) in FLOATS{assert_eq!(r[name].h[0],1);floats.extend_from_slice(&r[name].word(i,j,k));}
                for &(name,k) in INTS{assert!([3,4].contains(&r[name].h[0]));ints.extend_from_slice(&r[name].word(i,j,k));}
            }
            fs::write(format!("{}.f32",a[3]),floats).unwrap();fs::write(format!("{}.i32",a[3]),ints).unwrap();
            let locations:Vec<u8>=cells.iter().flat_map(|&(i,j)|[i,j].into_iter().flat_map(i32::to_le_bytes)).collect();
            fs::write(format!("{}.locations",a[3]),locations).unwrap();
            let state:Vec<u8>=MODULE_LOCATIONS.iter().flat_map(|name|r.get(*name).map_or([0;4],|v|v.word(0,0,0))).collect();
            fs::write(format!("{}.module_locations",a[3]),state).unwrap();
            let landuse=String::from_utf8(r["mminlu"].raw[r["mminlu"].at..].to_vec()).unwrap();
            let decoded=Command::new(&a[4]).args([&a[5],landuse.trim(),"STAS"]).output().unwrap();assert!(decoded.status.success());
            let mut table=decoded.stdout[..TABLE_BYTES].to_vec();let parameters=defaults();
            assert_eq!(parameters.len(),TABLE_BYTES);
            for name in TABLE_FIELDS{
                if let Some(record)=r.get(&format!("state.module_sf_noahlsm.{}",name.to_lowercase())){
                    let (at,n,kind)=field(name);let size=n*if kind=='c'{1}else{4};assert_eq!(record.raw.len()-record.at,size);
                    table[at..at+size].copy_from_slice(&record.raw[record.at..]);
                }
            }
            fs::write(format!("{}.table",a[3]),table).unwrap();println!("{{\"columns\":{}}}",cells.len());
        },
        "unpack"=>{
            assert_eq!(a.len(),6);let floats=fs::read(format!("{}.f32",a[4])).unwrap();let ints=fs::read(format!("{}.i32",a[4])).unwrap();
            assert_eq!(floats.len(),cells.len()*FLOATS.len()*4);assert_eq!(ints.len(),cells.len()*INTS.len()*4);
            for name in fs::read_to_string(&a[5]).unwrap().lines(){
                let mut record=r.get(name).unwrap_or_else(||panic!("post storage has no pre-call template: {name}")).clone();
                if let Some(k)=MODULE_LOCATIONS.iter().position(|v|*v==name){
                    let path=format!("{}.module_locations",a[4]);
                    if Path::new(&path).exists(){
                        let state=fs::read(path).unwrap();assert_eq!(state.len(),16);
                        record.raw[record.at..record.at+4].copy_from_slice(&state[k*4..k*4+4]);
                    }
                }
                for (layout,data) in [(FLOATS,&floats),(INTS,&ints)]{
                    for (k,&(field,level)) in layout.iter().enumerate(){if field!=name{continue;}
                        for (c,&(i,j)) in cells.iter().enumerate(){let at=record.offset(i,j,level);let start=(c*layout.len()+k)*4;
                            record.raw[at..at+4].copy_from_slice(&data[start..start+4]);}
                    }
                }
                record.raw[348..604].fill(b' ');record.raw[348..352].copy_from_slice(b"post");
                let path=format!("{}.post.{name}.bin",a[3]);assert!(!Path::new(&path).exists());fs::write(path,record.raw).unwrap();
            }
        },
        _=>panic!("unknown transport operation")
    }
}

// Test transport from existing native ABI columns; no substitute weather math.
fn fixture(a:&[String]){
    assert!(a.len()==4 || a.len()==5);let x=fs::read(format!("{}.f32",a[2])).unwrap();let ix=fs::read(format!("{}.i32",a[2])).unwrap();
    let count=16;let nx=20;assert!(x.len()>=count*FLOATS.len()*4 && ix.len()>=count*INTS.len()*4);
    let mut names=Vec::new();
    for (layout,data,code) in [(FLOATS,&x,1),(INTS,&ix,3)]{
        for &(name,_) in layout{
            if names.contains(&name){continue;}names.push(name);
            let items:Vec<_>=layout.iter().enumerate().filter(|(_,v)|v.0==name).collect();
            let scalar=["dt","xice_threshold","isurban","isice","itimestep","usemonalb","opt_thcnd"].contains(&name);
            let levels=items.len();let rank=if scalar{0}else if name=="dzs"{1}else if levels==1{2}else{3};
            let dims=match rank{0=>[1,1,1,1,1],1=>[levels as i32,1,1,1,1],2=>[nx,1,1,1,1],3=>[nx,levels as i32,1,1,1],_=>unreachable!()};
            let lower=if rank>=2{[-2,1,1,1,1]}else{[1,1,1,1,1]};
            let mut payload=vec![0xA5;dims.iter().product::<i32>() as usize*4];
            for &(source_index,&(_,level)) in &items{
                for c in 0..if scalar || rank==1{1}else{count}{
                    let at=match rank{0=>0,1=>level,2=>c+3,3=>c+3+nx as usize*level,_=>unreachable!()}*4;
                    let from=(c*layout.len()+source_index)*4;payload[at..at+4].copy_from_slice(&data[from..from+4]);
                }
            }
            emit_fixture(&a[3],name,code,4,rank,dims,lower,payload);
        }
    }
    for (name,value) in [("its",1),("ite",16),("jts",1),("jte",1),("kts",1),("num_soil_layers",4),("myj",0),("frpcpn",0),("rdlai2d",0),("ua_phys",0),("fasdas",0),("sf_urban_physics",0)]{
        emit_fixture(&a[3],name,3,4,0,[1;5],[1;5],i32::to_le_bytes(value).to_vec());
    }
    emit_fixture(&a[3],"mminlu",5,35,0,[1;5],[1;5],format!("{:<35}","MODIFIED_IGBP_MODIS_NOAH").into_bytes());
    if a.len()==5 {
        let table=fs::read(&a[4]).unwrap(); assert_eq!(table.len(),TABLE_BYTES);
        for name in TABLE_FIELDS {
            // This synthetic fixture models readonly parameter snapshots.
            // Mutable diagnostic coordinates are verified by real captures,
            // whose native tile coordinates differ from one-cell ABI calls.
            if ["ILOC","JLOC"].contains(name){continue;}
            let (at,n,kind)=field(name); let character=kind=='c';
            let width=if character{n as i32}else{4};
            let rank=if character || n==1{0}else{1};
            let dims=if rank==0{[1;5]}else{[n as i32,1,1,1,1]};
            let code=if character{5}else if kind=='i'{3}else{assert_eq!(kind,'f');1};
            let size=n*if character{1}else{4};
            emit_fixture(&a[3],&format!("state.module_sf_noahlsm.{}",name.to_lowercase()),
                         code,width,rank,dims,[1;5],table[at..at+size].to_vec());
        }
    }
}
fn emit_fixture(prefix:&str,name:&str,code:i32,width:i32,rank:i32,dims:[i32;5],lower:[i32;5],data:Vec<u8>){
    let mut h=[0;21];h[0]=code;h[1]=width;h[2]=rank;h[3..8].copy_from_slice(&dims);h[8..13].copy_from_slice(&lower);
    if rank>=2{h[16]=dims[0];h[17]=1;if rank==3{h[18]=2;}}h[19]=256;
    let mut raw=b"ORCL0001".to_vec();for v in h{raw.extend_from_slice(&v.to_le_bytes());}
    for text in ["lsm__driver_lsm__L705","pre",name]{raw.extend_from_slice(format!("{text:<256}").as_bytes());}
    for i in 0..h[16]{raw.extend_from_slice(&(i+1).to_le_bytes());}raw.extend(data);
    fs::write(format!("{prefix}.pre.{name}.bin"),raw).unwrap();
}
