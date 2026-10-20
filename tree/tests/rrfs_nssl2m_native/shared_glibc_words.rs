// Supporting audit transport for the unapplied glibc shared-header proposal.
// Rust generates synthetic operands, invokes native libm and compares raw words.
include!("math_trace.rs");

#[link(name="m")]
unsafe extern "C" {
    fn sin(x:f64)->f64;
    fn tanh(x:f64)->f64;
}
#[link(name="dl")]
unsafe extern "C" {
    fn dlopen(path:*const c_char,flags:i32)->*mut c_void;
    fn dlsym(handle:*mut c_void,symbol:*const c_char)->*mut c_void;
    fn dlclose(handle:*mut c_void)->i32;
}

fn create(root:&Path)->io::Result<()> {
    fs::create_dir_all(root)?;
    let mut inputs=vec![0,1,0x8000000000000000,0x8000000000000001,0x0010000000000000,
        0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000051,0xfff8000000000051,
        0x7ff0000000000051,0xfff0000000000051,0x7fefffffffffffff,0xffefffffffffffff];
    for boundary in [0x3e50000000000000u64,0x3feb600000000000,0x400368fd00000000,
        0x419921fb00000000,0x3fc020c49ba5e354,0x3c80000000000000,0x3c90000000000000,
        0x3ff0000000000000,0x4036000000000000,0x3fd62e4200000000,0x3ff0a2b200000000,
        0x4043687a00000000,0x40862e4200000000] {
        for delta in -16i64..=16 {
            let word=boundary.wrapping_add_signed(delta);
            inputs.extend([word,word|0x8000000000000000]);
        }
    }
    let mut random=0x1d6e370e170532a2u64;
    for _ in 0..65536 {
        random^=random<<13; random^=random>>7; random^=random<<17;
        inputs.push(random);
        inputs.push(((random>>11) as f64*(64.0/9007199254740992.0)-32.0).to_bits());
    }
    let mut rows=Vec::new();
    for (id,name,function) in [(11,"sin",sin as unsafe extern "C" fn(f64)->f64),
                               (13,"tanh",tanh as unsafe extern "C" fn(f64)->f64)] {
        let mut file=fs::File::create(root.join(format!("{name}.words")))?;
        for &word in &inputs {
            let result=unsafe{function(f64::from_bits(word))}.to_bits();
            for value in [word,0,result,1] {file.write_all(&value.to_le_bytes())?;}
        }
        rows.push(format!("{{\"id\":{id},\"name\":\"{name}\",\"bits\":64,\"unique\":{},\"calls\":{}}}",inputs.len(),inputs.len()));
    }
    fs::write(root.join("inventory.json"),format!("{{\"functions\":[{}]}}\n",rows.join(",")))
}

fn host(library:&Path,bank:&Path,out:&Path)->io::Result<bool> {
    fs::create_dir_all(out)?;
    let library=std::ffi::CString::new(library.to_str().unwrap()).unwrap();
    let handle=unsafe{dlopen(library.as_ptr(),2)};
    if handle.is_null() {return Err(io::Error::other("cannot load proposed host library"));}
    let mut total=0;let mut different=0;
    for (id,name) in [(11,"sin"),(13,"tanh")] {
        let input=bank.join(format!("{name}.words"));
        if !input.is_file() {continue;}
        let symbol=std::ffi::CString::new(format!("proposed_{name}")).unwrap();
        let pointer=unsafe{dlsym(handle,symbol.as_ptr())};
        if pointer.is_null() {return Err(io::Error::other("missing proposed function"));}
        let function:unsafe extern "C" fn(f64)->f64=unsafe{std::mem::transmute(pointer)};
        let input=std::ffi::CString::new(input.to_str().unwrap()).unwrap();
        let destination=std::ffi::CString::new(out.join(format!("{name}.json")).to_str().unwrap()).unwrap();
        let mut count=0;let mut x=std::ptr::null();let mut y=std::ptr::null();let mut expected=std::ptr::null();
        unsafe {
            let bank=nssl_trace_load(input.as_ptr(),id,&mut count,&mut x,&mut y,&mut expected);
            if bank.is_null() {return Err(io::Error::other("cannot load raw operand bank"));}
            let values=std::slice::from_raw_parts(x.cast::<f64>(),count);
            let actual:Vec<f64>=values.iter().map(|&x|function(x)).collect();
            let rc=nssl_trace_compare(bank,actual.as_ptr().cast(),destination.as_ptr());
            total+=count;different+=i32::from(rc!=0);nssl_trace_close(bank);
        }
    }
    unsafe{dlclose(handle);}
    fs::write(out.join("summary.json"),format!("{{\"words\":{total},\"differing_functions\":{different},\"acceptance\":false}}\n"))?;
    Ok(different==0)
}

fn main()->io::Result<()> {
    let args:Vec<String>=std::env::args().collect();
    let pass=match args[1].as_str() {
        "create"=>{create(Path::new(&args[2]))?;true},
        "host"=>host(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]))?,
        _=>return Err(io::Error::other("usage: create OUT | host LIB BANK OUT"))
    };
    if !pass {std::process::exit(1);}
    Ok(())
}
