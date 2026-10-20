//! Compare every observer word between complete unchanged native forecasts.
//! MPAS module_ra_cam.F:678-679; driver_radiation_lw.F:179-180,539-540.
//! gzip only unwraps receipt storage; word decoding/comparison is Rust.
use std::{collections::BTreeSet,env,fs,path::{Path,PathBuf},process::Command};

fn files(root:&Path)->BTreeSet<String> {
    fs::read_dir(root).unwrap().map(|p|p.unwrap().file_name().into_string().unwrap())
        .filter(|p|p.ends_with(".raw.gz")).collect()
}
fn words(path:PathBuf)->Vec<u32> {
    let output=Command::new("gzip").arg("-dc").arg(path).output().unwrap();
    assert!(output.status.success());
    assert!(!output.stdout.is_empty() && output.stdout.len()%4==0);
    output.stdout.chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect()
}
fn main() {
    let args:Vec<_>=env::args().collect();assert_eq!(args.len(),4);
    let left=Path::new(&args[1]);let right=Path::new(&args[2]);
    let ranks:usize=args[3].parse().unwrap();assert!(ranks==1 || ranks==2);
    let names=files(left);assert_eq!(names,files(right));assert_eq!(names.len(),ranks*96);
    let mut total=0;let mut different=0;let mut sep=false;
    println!("{{\"scope\":\"complete strict-native boundary comparison; no exclusions\",\"process_count\":{},\"excluded_words\":0,\"arrays\":[",ranks);
    for name in names {
        assert!(name.bytes().all(|x|x.is_ascii_alphanumeric() || b"-._".contains(&x)));
        let a=words(left.join(&name));let b=words(right.join(&name));assert_eq!(a.len(),b.len());
        assert_eq!(a.len()%56,0);let physical_end=(a.len()/56)*55;
        let mut diffs=Vec::new();let mut physical=0;
        for (i,(x,y)) in a.iter().zip(&b).enumerate() {
            if x!=y {diffs.push((i,*x,*y));if i<physical_end {physical+=1;}}
        }
        total+=a.len();different+=diffs.len();
        if sep {println!(",");}sep=true;
        print!("{{\"file\":\"{}\",\"words\":{},\"physical_levels_end\":{},\"differing_words\":{},\"physical_differing_words\":{},\"native_left_physical_nonzero\":{},\"native_right_physical_nonzero\":{},\"word_set\":[",
            name,a.len(),physical_end,diffs.len(),physical,
            a[..physical_end].iter().filter(|x|**x!=0).count(),b[..physical_end].iter().filter(|x|**x!=0).count());
        for (i,(index,x,y)) in diffs.iter().enumerate() {
            if i>0 {print!(",");}print!("[{},\"{:08x}\",\"{:08x}\"]",index,x,y);
        }
        print!("]}}");
    }
    println!("],\"words\":{},\"differing_words\":{},\"complete\":true}}",total,different);
}
