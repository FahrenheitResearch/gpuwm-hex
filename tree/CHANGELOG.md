# Changelog

## Unreleased

Fixed:
- The registered v15.150.38857 static is rebuilt on the unified 82-variable
  `rw_mpas_static` writer and re-registered (#330). The retired writer's
  drag band sampled terrain 180 degrees of longitude from every cell (the
  archive-origin assumption): measured corr(old, new) for var2d is +0.003
  at the same cell and +0.697 at lon+180, and the field-by-field compare
  shows oa/ol moving full scale on two thirds of cells. The rebuilt static
  matches a native init_atmosphere static for the same mesh at var2d
  corr +0.9999, oa1 +0.9961, land-only con +0.9928, and adds the operator
  tables and soil-composition group the retired writer omitted. The x4 and
  x1.40962 rows were measured to be native-built statics (v8.4.1 and the
  published v8.2.0 artifact) that never carried the band; each registry row
  now names its builder.
- A lake column (MODIS category 21) is folded to open water at the forecast
  loader boundary, the same conversion WRF applies without a lake model.
  The arwen vegetation tables end at category 20, so before the fold any
  generated-mesh run with lakes died with an IndexError inside the Noah-MP
  cold start; the native x4 landuse never exceeds 19, which is why the
  proof path never saw it. The fold count is in every run receipt; on the
  x4 case the mask is empty and every array passes through untouched.
- The GWDO dt guard follows the mesh binding: a registered mesh runs the
  YSU-GWDO kernel at its own Courant-admitted timestep instead of dying
  at step 0 on "requires dt_seconds=120". The kernel takes dt as a runtime
  argument; on the frozen native mesh nothing is rebound and the guard
  still demands exactly 120 s.
- The x4 proof's restart leg is bit-identical again. GF's advective
  forcing pair (rthdynten/rqvdynten) is per-step carried state: each
  step's dynamics forms it and the next step's physics consumes it, and
  it lives outside both the MPAS atmosphere and the Arwen backend
  restart payload. The F030 checkpoint never captured it, so every
  restored run re-entered step 16 with zero forcing lanes while the
  unbroken run fed the real step-15 pair, and the step-16 identity gate
  failed deterministically on every arm (#327, 5/5 red on the reference node, red
  since the forcing lanes landed). Checkpoint schema v3 downloads the
  pair at F030, refuses to write a checkpoint without it, re-seeds it on
  restore in both the fresh-process worker and the in-process
  instrument, and gates the rehydration with its own fingerprint
  identity. A pre-v3 checkpoint is refused by name instead of resuming
  wrong.
