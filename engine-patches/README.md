Apply `rw-mpas-stream-ordered-metadata.patch` in the engine repository, then rebuild the `rw-mpas` binaries with `--features system-netcdf`. The hex build helper carries that feature flag.

Base engine commit: `24b6f08c48db3199051280f5a874ef33b41a5762`.

The codec retains dimension, variable and typed attribute order and matches SMIOL's 131072-byte header alignment. Its Rust unit test writes one frame, reads it through the Rust NetCDF reader, and checks every primitive word, timestamp and metadata order. Run with a writable `RW_MPAS_STREAM_TEST_DIRECTORY` and the engine revision stamp set by the engine build.

The stream patch was checked and tested in a standalone single-source Cargo harness. The whole engine workspace build was not run.

`rrtmg-lw-active-gpoints.patch` is a separate engine candidate, not applied by the hex build helper. It edits `gpuwm/core/rrtmg_lw.py` so that, in the per-g-point longwave band kernels, the padded lanes above the band g-point count return before the band coefficient arithmetic. Status: offline compile only (66 to 70 registers, no spills). Real-input words, loaded resources and an H60 wall pair have not been measured, so no build applies it until that qualification runs.
