fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--abi"] {
        println!("rw-s2s/v1 native-mjo fixed-eof bsiso hovmoller weekly-era5 rmm-bcor rayon");
        return;
    }
    let result = match args.as_slice() {
        [mode, path] if mode == "mjo" => rw_s2s::mjo::run(path),
        [mode, path] if mode == "score" => rw_s2s::scores::run(path),
        [mode, path] if mode == "extract" => rw_s2s::extract::run(path),
        _ => Err("usage: rw_s2s mjo|score|extract PLAN.json".into()),
    };
    if let Err(error) = result {
        if let Some(path) = args.get(1) {
            if let Ok(plan) = rw_s2s::load::<serde_json::Value>(path) {
                if let Some(output) = plan["output"].as_str() {
                    let _ = rw_s2s::save(
                        std::path::Path::new(output).join("receipt.json"),
                        &serde_json::json!({"status":"failed","error":error,"plan":path}),
                    );
                }
            }
        }
        eprintln!("rw_s2s: {error}");
        std::process::exit(2);
    }
}
