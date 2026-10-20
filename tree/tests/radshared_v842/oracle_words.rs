//! Typed MPAS oracle stream decoding, exact word comparison and column extraction.
//! No arithmetic tolerance is introduced. Python delegates all field I/O here.
//! Based on f1's oracle_words.rs; lane additions export validated buffers and
//! replace payloads without interpreting weather fields in Python.
use std::{collections::BTreeMap, env, fs, io, path::Path};
const FIXED: usize = 8 + 21 * 4 + 3 * 256;
#[derive(Clone)]
struct Record { header: [i32;21], strings: Vec<u8>, ids: Vec<i32>, data: Vec<u8> }
fn bad(message: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message) }
fn read(path: &Path) -> io::Result<Record> {
    let raw = fs::read(path)?;
    if raw.len() < FIXED || &raw[..8] != b"ORCL0001" { return Err(bad("invalid oracle header")); }
    let mut h=[0i32;21];
    for (i,v) in h.iter_mut().enumerate() { *v=i32::from_le_bytes(raw[8+4*i..12+4*i].try_into().unwrap()); }
    if !(0..=5).contains(&h[2]) || h[3..8].iter().any(|v| *v<0) || h[16]<0 || h[19]!=256 || h[20]!=0 {
        return Err(bad("malformed dimensions/version"));
    }
    let width=match h[0] {1|3|4=>4,2|6=>8,5 if h[1]>0=>h[1],_=>return Err(bad("invalid type code/width"))};
    if h[1]!=width { return Err(bad("type width mismatch")); }
    if h[17]<0 || h[17]>h[2] || h[18]<0 || h[18]>h[2] || (h[17]>0 && h[17]==h[18]) {return Err(bad("invalid axes"));}
    let expected_ids=if h[17]>0 {h[2+h[17] as usize]} else {0};
    if h[16]!=expected_ids {return Err(bad("ID count mismatch"));}
    if h[0]!=5 && h[3+h[2] as usize..8].iter().any(|v|*v!=1) {return Err(bad("inactive dimensions must equal one"));}
    let elements=h[3..8].iter().try_fold(1usize,|a,v|a.checked_mul(*v as usize)).ok_or_else(||bad("shape overflow"))?;
    let offset=FIXED.checked_add((h[16] as usize).checked_mul(4).ok_or_else(||bad("ID overflow"))?).ok_or_else(||bad("offset overflow"))?;
    let size=elements.checked_mul(width as usize).and_then(|v|v.checked_add(offset)).ok_or_else(||bad("payload overflow"))?;
    if raw.len()!=size {return Err(bad("truncated or trailing payload"));}
    let ids=raw[FIXED..offset].chunks_exact(4).map(|x|i32::from_le_bytes(x.try_into().unwrap())).collect();
    Ok(Record{header:h, strings:raw[92..FIXED].to_vec(), ids,data:raw[offset..].to_vec()})
}
fn escaped(s:&str)->String {
    let mut out=String::from("\"");
    for c in s.chars() { match c {'"'=>out.push_str("\\\""),'\\'=>out.push_str("\\\\"),'\n'=>out.push_str("\\n"),c if c.is_control()=>out.push_str(&format!("\\u{:04x}",c as u32)),c=>out.push(c)} }
    out.push('"');out
}
fn word(bytes:&[u8])->u64 {bytes.iter().enumerate().fold(0,|v,(i,b)|v|((*b as u64)<<(8*i)))}
fn ordered(bits:u64,width:usize)->u64 {
    let sign=1u64<<(width*8-1); let mask=if width==8 {u64::MAX} else {(1u64<<(width*8))-1};
    if bits&sign!=0 {!bits&mask} else {bits|sign}
}
fn finite(bits:u64,code:i32)->bool {if code==1 {bits&0x7f800000!=0x7f800000} else {bits&0x7ff0000000000000!=0x7ff0000000000000}}
fn coords(r:&Record,mut index:usize)->[i32;5] {
    let mut result=[0;5];
    for (i,v) in result.iter_mut().enumerate() { let dim=r.header[3+i] as usize;*v=(index%dim) as i32+r.header[8+i];index/=dim; }
    result
}
fn compare(a:&Record,b:&Record)->io::Result<String> {
    if a.header[..13]!=b.header[..13] || a.header[16..19]!=b.header[16..19] || a.ids!=b.ids {
        let differences=[("type",a.header[0]!=b.header[0]),("width",a.header[1]!=b.header[1]),
            ("rank",a.header[2]!=b.header[2]),("dimensions",a.header[3..8]!=b.header[3..8]),
            ("lower_bounds",a.header[8..13]!=b.header[8..13]),("global_ids",a.ids!=b.ids),
            ("cell_axis",a.header[17]!=b.header[17]),("level_axis",a.header[18]!=b.header[18])]
            .into_iter().filter(|(_,different)|*different).map(|(name,_)|escaped(name)).collect::<Vec<_>>().join(",");
        return Ok(format!("{{\"words\":{},\"other_words\":{},\"different_words\":null,\"max_ulp\":null,\"different_nonfinite_words\":null,\"first_difference\":null,\"bitwise_equal\":false,\"comparison_complete\":false,\"schema_mismatch\":[{differences}]}}",a.data.len()/a.header[1] as usize,b.data.len()/b.header[1] as usize));
    }
    let width=a.header[1] as usize;
    let mut differing=0;let mut max_ulp=0;let mut nonfinite=0;let mut first=None;
    for (i,(l,r)) in a.data.chunks_exact(width).zip(b.data.chunks_exact(width)).enumerate() {
        if l==r {continue;} differing+=1;if first.is_none(){first=Some(i);}
        if a.header[0]==1 || a.header[0]==2 {
            let x=word(l);let y=word(r);
            if !finite(x,a.header[0]) || !finite(y,a.header[0]) {nonfinite+=1;} else {max_ulp=max_ulp.max(ordered(x,width).abs_diff(ordered(y,width)));}
        }
    }
    let location=if let Some(i)=first {
        let c=coords(a,i); let ca=a.header[17]; let la=a.header[18];
        let cell=if ca>0 {let id=a.ids[(c[ca as usize-1]-a.header[7+ca as usize]) as usize];if id>0{id.to_string()}else{"null".into()}} else {"null".into()};
        let level=if la>0 {c[la as usize-1].to_string()} else {"null".into()};
        format!("{{\"word\":{i},\"cellID\":{cell},\"level\":{level},\"indices\":{:?}}}", &c[..a.header[2] as usize])
    } else {"null".into()};
    let ulp=if a.header[0]==1||a.header[0]==2 {max_ulp.to_string()} else {"null".into()};
    Ok(format!("{{\"words\":{},\"different_words\":{differing},\"max_ulp\":{ulp},\"different_nonfinite_words\":{nonfinite},\"first_difference\":{location},\"bitwise_equal\":{}}}",a.data.len()/width,differing==0))
}
fn column(r:&Record,id:i32)->io::Result<Record> {
    let ca=r.header[17];if ca==0{return Ok(r.clone());}
    let axis=ca as usize-1;
    let col=r.ids.iter().position(|v|*v==id).ok_or_else(||bad("requested global cell ID absent"))?;
    let mut out=r.clone();out.header[3+axis]=1;out.header[8+axis]=r.header[8+axis]+col as i32;out.header[16]=1;out.ids=vec![id];out.data.clear();
    let width=r.header[1] as usize;
    for (i,v) in r.data.chunks_exact(width).enumerate() {if (coords(r,i)[axis]-r.header[8+axis]) as usize==col {out.data.extend(v);}}
    Ok(out)
}
fn write(path:&Path,r:&Record)->io::Result<()> {
    use std::io::Write;
    let mut file=fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(b"ORCL0001")?;for v in r.header {file.write_all(&v.to_le_bytes())?;}
    file.write_all(&r.strings)?;for v in &r.ids {file.write_all(&v.to_le_bytes())?;}file.write_all(&r.data)
}
fn validate(root:&Path)->io::Result<String> {
    let mut paths=fs::read_dir(root)?.map(|p|p.map(|p|p.path())).collect::<io::Result<Vec<_>>>()?;
    paths.retain(|p|p.extension().is_some_and(|e|e=="bin"));paths.sort();
    if paths.is_empty(){return Err(bad("no oracle records"));}
    let mut groups:BTreeMap<(String,String,i32),(usize,usize,usize,usize)>=BTreeMap::new();
    for path in &paths {
        let r=read(path).map_err(|e|bad(&format!("{}: {e}",path.display())))?;
        let key=String::from_utf8_lossy(&r.strings[..256]).trim_end().to_string();
        let phase=String::from_utf8_lossy(&r.strings[256..512]).trim_end().to_string();
        let name=String::from_utf8_lossy(&r.strings[512..768]);
        let g=groups.entry((key,phase,r.header[0])).or_default();
        g.0+=1;g.1+=r.data.len();g.2+=usize::from(r.header[17]>0);
        if name.trim_end().ends_with(".present") {
            if r.header[0]!=4 || r.header[2]!=0{return Err(bad("presence flag is not a logical scalar"));}
            g.3+=usize::from(word(&r.data)==0);
        }
    }
    let rows=groups.into_iter().map(|((key,phase,code),(records,bytes,mapped,absent))|
        format!("{{\"key\":{},\"phase\":{},\"code\":{code},\"records\":{records},\"payload_bytes\":{bytes},\"cell_mapped_records\":{mapped},\"absent_presence_records\":{absent}}}",escaped(&key),escaped(&phase))).collect::<Vec<_>>().join(",");
    Ok(format!("{{\"schema\":\"mpas-oracle-transport-validation-v1\",\"all_records_valid\":true,\"records\":{},\"groups\":[{rows}]}}",paths.len()))
}
fn main()->io::Result<()> {
    let args:Vec<String>=env::args().collect();
    if args.len()==4 && args[1]=="compare-raw" {
        let a=fs::read(&args[2])?;let b=fs::read(&args[3])?;
        if a.len()!=b.len() || a.len()%4!=0 {return Err(bad("raw 32-bit output extent differs"));}
        let different=a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(l,r)|l!=r).count();
        println!("{{\"words\":{},\"different_words\":{different},\"bitwise_equal\":{},\"comparison_complete\":true}}",a.len()/4,different==0);
        return Ok(());
    }
    if args.len()==5 && args[1]=="raw-prefix" {
        use std::io::Write;
        let mut data=fs::read(&args[2])?;
        if data.len()%4!=0{return Err(bad("raw float32 array length must be a multiple of four"));}
        let prefix=args[4].split(',').map(|s|s.parse::<u32>().map_err(|_|bad("invalid prefix word")))
            .collect::<io::Result<Vec<_>>>()?;
        if prefix.len()>data.len()/4{return Err(bad("prefix exceeds raw array"));}
        for (i,value) in prefix.into_iter().enumerate(){data[4*i..4*i+4].copy_from_slice(&value.to_le_bytes());}
        let mut output=fs::OpenOptions::new().write(true).create_new(true).open(&args[3])?;
        output.write_all(&data)?;return Ok(());
    }
    if args.len()==4 && args[1]=="table" {
        let mut data=Vec::new();
        for token in fs::read_to_string(&args[2])?.split_whitespace() {
            let value:f32=token.replace(['d','D'],"e").parse().map_err(|_|bad("invalid table float"))?;
            data.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&args[3],data)?;return Ok(());
    }
    if args.len()==3 && args[1]=="validate" {println!("{}",validate(Path::new(&args[2]))?);return Ok(());}
    if args.len()<4 {return Err(bad("usage: oracle_words compare LEFT RIGHT | column INPUT CELLID OUTPUT"));}
    let a=read(Path::new(&args[2]))?;
    match args[1].as_str() {
        "payload"=>fs::write(&args[3],&a.data)?,
        "profile"=>{
            if a.header[0]!=1 || a.header[2]!=3 || a.header[5]!=1 {
                return Err(bad("profile requires float32 [cell,level,1]"));
            }
            let nc=a.header[3] as usize;let nz=a.header[4] as usize;
            let mut data=Vec::with_capacity(a.data.len());
            for c in 0..nc {for k in 0..nz {
                let i=(c+k*nc)*4;data.extend_from_slice(&a.data[i..i+4]);
            }}
            fs::write(&args[3],data)?;
        },
        "replace"=>{
            if args.len()!=5{return Err(bad("replace TEMPLATE PAYLOAD OUTPUT"));}
            let mut out=a.clone();out.data=fs::read(&args[3])?;
            if out.data.len()!=a.data.len(){return Err(bad("replacement size mismatch"));}
            out.strings[256..512].fill(b' ');out.strings[256..260].copy_from_slice(b"post");
            write(Path::new(&args[4]),&out)?;
        },
        "postcopy"=>{
            let mut out=a.clone();out.strings[256..512].fill(b' ');
            out.strings[256..260].copy_from_slice(b"post");write(Path::new(&args[3]),&out)?;
        },
        "compare"=>println!("{}",compare(&a,&read(Path::new(&args[3]))?)?),
        "column"=>{if args.len()!=5{return Err(bad("column requires output path"));}let id=args[3].parse().map_err(|_|bad("invalid cellID"))?;write(Path::new(&args[4]),&column(&a,id)?)?;},
        "header"=>println!("{{\"code\":{},\"width\":{},\"shape\":{:?},\"lower\":{:?},\"global_ids\":{:?},\"key\":{},\"phase\":{},\"name\":{},\"step\":{},\"event\":{},\"mpi_rank\":{},\"cell_axis\":{},\"level_axis\":{}}}",a.header[0],a.header[1],&a.header[3..3+a.header[2] as usize],&a.header[8..8+a.header[2] as usize],a.ids,escaped(String::from_utf8_lossy(&a.strings[..256]).trim_end()),escaped(String::from_utf8_lossy(&a.strings[256..512]).trim_end()),escaped(String::from_utf8_lossy(&a.strings[512..768]).trim_end()),a.header[13],a.header[14],a.header[15],a.header[17],a.header[18]),
        "present"=>{if a.header[0]!=4 || a.header[2]!=0 {return Err(bad("presence requires a scalar logical record"));}println!("{}",word(&a.data)!=0);},
        _=>return Err(bad("unknown operation")),
    }
    Ok(())
}
