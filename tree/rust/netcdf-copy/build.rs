fn main() {
    println!("cargo:rerun-if-env-changed=GPUWM_HEX_NETCDF_CLASSIC_PREFIX");
    println!("cargo:rustc-check-cfg=cfg(netcdf_classic_static)");
    if let Some(prefix) = std::env::var_os("GPUWM_HEX_NETCDF_CLASSIC_PREFIX") {
        let library = std::path::PathBuf::from(prefix).join("lib");
        assert!(library.join("libnetcdf.a").is_file(), "missing static classic netCDF archive");
        println!("cargo:rustc-link-search=native={}", library.display());
        println!("cargo:rustc-cfg=netcdf_classic_static");
    }
}
