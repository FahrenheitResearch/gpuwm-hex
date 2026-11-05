# Spectral waves on the native mesh

The global, variable-resolution globe, regional and moving doors accept
`--waves on` and `--waves-options '<json>'`. Waves default off. The shared
engine settings table supplies the roughness-feedback default; an explicit
`{"roughness_feedback":false}` retains the atmosphere's existing surface kernel.
The MPAS MYNN, RRFS MYNN-SFC and ArWen routes receive the same device roughness.
An external roughness is inserted before the water layer computes thermal
roughness and fluxes. It is not an after-the-fact diagnostic replacement.

The solver is `gpuwm.ocean.waves:create_solver`, with its `voronoi` backend.
The backend uses native cellsOnEdge, edgesOnCell, dvEdge, dcEdge, areaCell,
angleEdge and spherical cell/edge locations. Compact wet-cell spectra keep
the shared action-density convention. Shared faces carry opposite fluxes;
open edges consume incident spectra on inflow and coasts absorb outgoing action.
Limiter, time integration and dispersion references are in
`gpuwm/ocean/waves/backends/voronoi_SCIENCE.md` in the engine.

One device wave state receives the owner cells of each atmospheric rate class.
It advances only when every owner reaches a coupling boundary. Source winds
are east/north. Both route bindings leave atmospheric arrays read-only when
roughness feedback is false. The completed 192-hour MPAS pair compared every
field in 33 histories and 33 diagnostic files: 3,330,686,997 stored words,
including 3,159,010,701 floating words, with zero differences. Global attributes
were outside that comparison. This proves the recorded MPAS pair, not every
physics route or later source tree; exact run provenance is in the lane's
`receipts/atmosphere-identity.json`.

Prepared input uses the shared Rust `gpuwm.waves.prepared/v1` contract. The
`prepared` option selects an existing verified manifest folder. Automatic mesh
preparation delegates to the data lane's `prepare_mesh` entry point. Native
cell order and coordinates are checked before constructing the solver.
The source `preparation` option is metadata for that shared entry point.

Checkpoints contain action, depth/water-level state, the finite-iteration
surface and shelter carriers, clocks, forcing and station samples. A moving
mesh uses the atmosphere's validated spherical polygon-overlap operator for
action. Direction bins rotate by spherical parallel transport before linear
periodic interpolation; this preserves their frequency-bin integral. Other
cell carriers use positive area weights, with vector components rotated into
the new local frame. New uncovered area uses prepared spectra at the move time.
The packed positive-wind-input mask is a diagnostic rather than an interpolated
integer field. After remapping, the carried forcing is installed and stress
diagnostics rebuild that mask before feedback is refreshed. Undeclared new
core checkpoint arrays refuse a move until an explicit transfer rule exists.
This remapping is first order in geographic space, as in Jones (1999),
*Monthly Weather Review* 127, 2204-2210,
DOI 10.1175/1520-0493(1999)127<2204:FASOCR>2.0.CO;2.

Native wave snapshots and station spectra are written by the Rust NetCDF
writer. Stations retain `[station,time,frequency,direction]` exchange spectra.
The direction coordinate is FROM, clockwise from north; the legacy scalar
toward diagnostic is named `wave_toward_direction` in points.nc to avoid
colliding with the spectral coordinate.
Station metadata records the selected wet cell, coordinates, depth, distance
from the buoy and exposure decision. Coastal selection uses prepared incoming
swell directions and local mesh scale, without observation tuning. Large
offshore offsets on coarse meshes remain a limitation for coastal comparisons.

Products use `rw_hex_wave_products` for parallel spherical polygon lookup and
`rw_wrfbatch` through `gpuwm.rustwx.run_cf_renderer` for drawing. Maps retain
native cell values without smoothing. The six-hour sequence includes Hs with
peak travel-direction arrows, Tp, wind sea and all three swell partitions.
Products and station files are under `<case>/<domain>/waves/`.

Implementation status and qualification are separate. The lane report names
the committed tree, executed commands, missing dependencies and whole-run
evidence. This document is not a scientific skill or release-readiness claim.
