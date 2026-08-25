# Changelog

## 0.1.1 (2026-08-25)

The forecast becomes a front door, and the referee runs.

New:
- gpuwm-hex forecast: a front door that binds the mesh, asks the card
  first against a measured per-card row, refuses by name with numbers,
  and prints the render command when it passes. --preflight gives the
  same answer without spending anything.
- The obs referee ships with its first scorecard: canonical model bundles
  gain a producer, and four metrics that could not score now score.
- The default history stream publishes refl10cm and q2, and
  rw_mpas_convert maps both, so reflectivity products read the model's
  own field instead of the renderer's hydrometeor fallback.
- A generated mesh completes a forecast end to end. A mesh whose Voronoi
  edges collapse is refused at bind, by name, before anything expensive.
- The engine seam pin moves to the gpuwm release line and verifies 16/16
  clean, so an engine checkout at the pinned commit satisfies the pins
  as cut.
- A per-allocation device-memory ledger. Measured 2026-08-24 on an RTX
  5070 Ti: x1.40962 peaks at 5,604.0 MiB with the engine's device-sized
  radiation chunks.

Fixed:
- The registered v15 static is rebuilt on the unified 82-variable writer;
  the retired writer's drag band sampled terrain 180 degrees of longitude
  away. Every registry row now names its builder.
- A lake landuse column folds to open water at the forecast loader
  boundary; a generated mesh with lakes no longer dies in the Noah-MP
  cold start.
- The GWDO dt guard follows the mesh binding; a registered mesh runs at
  its own admitted timestep.
- Restart checkpoints carry GF's advective forcing pair (schema v3);
  restored runs are bit-identical again, and a pre-v3 checkpoint is
  refused by name.
- The forecast door leaves output creation to the driver; an admitted run
  no longer fails on a directory that already exists.

Requires gpuwm 2.5.5 or newer for the seam bytes and the bundled engine
binaries.
