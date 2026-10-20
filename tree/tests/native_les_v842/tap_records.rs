//! f1 transport plus lossless payload transfer for the LES CUDA replay.
mod f1 {
    // Build copies f1's reader verbatim except inner documentation markers.
    include!(env!("D2_TAP_F1"));
    pub fn run() -> io::Result<()> {
        use std::io::{Read, Write};
        let args: Vec<String> = env::args().collect();
        if args.len() != 4 || !matches!(args[1].as_str(), "payload" | "wrap") {
            return main();
        }
        let record = read(Path::new(&args[2]))?;
        if args[1] == "payload" {
            return io::stdout().write_all(&record.data);
        }
        let mut payload = Vec::new();
        io::stdin().read_to_end(&mut payload)?;
        if payload.len() != record.data.len() { return Err(bad("output payload size mismatch")); }
        let mut strings = record.strings;
        strings[256..512].fill(b' ');
        strings[256..260].copy_from_slice(b"post");
        let mut output = Vec::from(&b"ORCL0001"[..]);
        for word in record.header { output.extend_from_slice(&word.to_le_bytes()); }
        output.extend_from_slice(&strings);
        for word in record.ids { output.extend_from_slice(&word.to_le_bytes()); }
        output.extend_from_slice(&payload);
        fs::write(&args[3], output)
    }
}
fn main() -> std::io::Result<()> { f1::run() }
