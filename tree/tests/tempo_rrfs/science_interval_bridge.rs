// Explicit cross-boundary pairing for the complete real native TEMPO interval.
// Fork: MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:592-646
// at 1d6e370e. Append after the unchanged oracle_words and micro_record_bridge.
// Only boundary/phase/event differ; types, bounds, IDs and every payload word
// remain subject to the original exact Rust comparator.
#[no_mangle]
pub unsafe extern "C" fn r5_interval_compare(left: *const c_char, right: *const c_char) -> *mut c_char {
    let result = (|| -> io::Result<String> {
        let a = read(r4_path(left)?)?;
        let b = read(r4_path(right)?)?;
        let text = |r: &Record, start: usize| -> String {
            String::from_utf8_lossy(&r.strings[start..start + 256]).trim_end().to_string()
        };
        if text(&a, 0) != "boundary__microphysics_to_mpas"
            || text(&b, 0) != "boundary__microphysics_from_mpas"
            || text(&a, 256) != "pre" || text(&b, 256) != "post"
            || a.strings[512..] != b.strings[512..]
            || a.header[13] != 1 || b.header[13] != 0 || a.header[15] != 0 || b.header[15] != 0
            // The producer's file names number these two taps 0 and 1;
            // the authenticated original record headers number them in the
            // step member (1 and 0), with event 0 and MPI rank 0 at both taps.
            || a.header[14] != 0 || b.header[14] != 0
            || a.header[19..] != b.header[19..]
        {
            return Err(bad(&format!("unapproved interval pairing: {} / {}, {} / {}, names {} / {}, headers {:?} / {:?}",
                text(&a, 0), text(&b, 0), text(&a, 256), text(&b, 256), text(&a, 512), text(&b, 512), a.header, b.header)));
        }
        compare(&a, &b)
    })();
    match result {
        Ok(s) => CString::new(s).unwrap().into_raw(),
        Err(e) => { eprintln!("RRFS interval compare: {e}"); std::ptr::null_mut() }
    }
}
