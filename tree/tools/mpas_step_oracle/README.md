# Frozen full-step trajectory extractor

This lane captures both endpoints of one complete MPAS-A RK3 step.  Its
authority is the frozen v8.2.3 model executable, not a reimplementation of an
operator.  The run is the 15-level x1.2562 Jablonowski-Williamson case with a
600-second time step, six acoustic substeps, split dynamics/transport, and no
physics suite.

The Fortran program reads the model history files through NetCDF-Fortran and
writes their IEEE-754 binary32 payloads without decimal conversion.  Files are
little-endian and have no header; `manifest.json` is the authoritative schema,
shape, hash, and tolerance record.

From WSL:

```bash
bash tools/mpas_step_oracle/extract.sh "$PWD"
python tools/mpas_step_oracle/build_manifest.py \
  oracle/jw-x1.2562-v8.2.3
```

This evidence does **not** claim that the Python port matches the trajectory.
It creates the gate that the Python whole-step driver must pass.
