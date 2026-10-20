# Native surface record reader

This ships the C ABI used by both hex physics routes. Its decode routines
come from `tests/surface_monthly_input_v842.rs`, the existing native MPAS
surface replay reader. The crate adds packaging, stream extent validation,
and a lock around NetCDF calls for concurrent resident ranks. It does not
link or depend on a private reader binary.

`read_surface_time(path, record, output, capacity)` reads `xtime` characters
and returns their length, without a trailing NUL. `surface_cell_count(path)`
returns the native `nCells` extent. `read_surface_field` reads a caller-owned
float32 or int32 buffer after checking types, dimension order, record,
column range and capacity. `read_monthly` retains the existing monthly
replay ABI. All functions return NetCDF error codes or the schema errors
`-10001` (type/rank), `-10002` (axes), `-10003` (arguments), `-10004`
(range), and `-10005` (capacity).

The surface series uses `Time,nCells,StrLen=64`, character
`xtime(Time,StrLen)` containing padded `YYYY-MM-DD_HH:MM:SS`, and float32
`sst(Time,nCells)` and `xice(Time,nCells)`. The shared Python record reader
selects latest-before on the model clock and copies both fields to the GPU
in one transfer. Halo/padded columns remain caller-owned. Monthly native
inputs use float32 `greenfrac/albedo12m(nCells,nMonths=12)` and int32
`landmask(nCells)`.

Build with `cargo build --release --locked --offline -p rw-surface` from
`tools/rustwx`. Source builds need NetCDF-C development files. The wheel
bridge builder uses the `static-netcdf` feature with its pinned combined
NetCDF-C/HDF5/zlib archive, so installed wheels need no system NetCDF.
`setup.py` verifies the binary and Rust source hashes in bridge provenance.

For tests, set `P25_READER_TEST_DIR` to an owned scratch directory and run
`cargo test --locked --offline -p rw-surface`. The test removes its own
small temporary NetCDF after checking native words and malformed inputs.
