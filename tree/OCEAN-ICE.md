# Prognostic ocean and sea ice

The S2S door selects `--ocean-mode prognostic` by default. It initializes SST and ice fraction once from each member's initialization and evolves them at the ArWen surface cadence. No future SST or ice record is read. `--ocean-mode prescribed` retains the dated surface-series route. A conflicting `--surface-series` in prognostic mode refuses before launch.

For a global forecast outside the S2S door, set `GPUWM_HEX_OCEAN_MODE=prognostic`. The surface service requires Noah-MP and its routed surface-layer hook. Other land schemes refuse this configuration. Existing hydro and stochastic selections remain in force.

`--ocean-options '{"depth_m":25}'` overrides declared entries in `src/hexcore/data/ocean-mixed-layer.v1.json`. The complete parameter table is recorded in the member signature and surface checkpoint. Direct forecasts use the same JSON in `GPUWM_HEX_OCEAN_OPTIONS`. Changed parameters or a changed ocean mode refuse a checkpoint restore.

## Equations and state

Each native Voronoi water column carries mixed-layer temperature and depth, depth-integrated horizontal current, ice volume per grid-cell area, ice area fraction, ice surface temperature and snow water equivalent. These are binary64 GPU arrays in the existing surface restart and rollback pool. Atmosphere-facing surface fields retain their existing binary32 layout.

The stored energy relative to the salinity-dependent freezing temperature is

`E = rho_water * cp_water * H * (T - Tfreeze) - rho_ice * L * ice_volume`.

Open water receives absorbed shortwave plus net longwave minus its own sensible and latent heat fluxes. Ice receives the corresponding ice-tile flux. A zero-heat-capacity ice and snow column solves surface temperature with conductive resistance `h_ice/k_ice + h_snow/k_snow`. Surface melt first consumes snow latent heat. Basal ocean heat melts ice; winter conductive loss grows ice. Ocean supercooling converts to frazil latent heat. Ice growth spreads at the declared new-ice thickness; melt decreases area in proportion to lost volume. Snow water is conserved when area changes. Surface categories, albedo, skin temperature and the fractional flux blend are republished to the atmospheric driver.

Wind mixing uses the WRF public-domain OML bulk potential-energy inversion family with a fixed deep temperature gradient. An exact inertial rotation avoids explicit-Euler current energy growth. Deepening adds thermocline heat explicitly to the budget. Default depth is 50 m, maximum depth 200 m and gradient 0.14 K/m. The per-cell cumulative boundary energy, entrainment energy and numerical closure residual are exported. These diagnostics make budget errors visible rather than correcting the state after the fact.

This is a mixed-layer ocean and zero-layer thermodynamic sea-ice model. It has fixed salinity, no horizontal ocean or ice dynamics and no ice-sheet melt or runoff model. Initial ice thickness is a declared uniform prior, not an observed thickness analysis. A coupled response demonstration does not establish seasonal forecast skill or validated melt/runoff products.

Prepared initializations supplied with `--prepared-init` are hard-linked into the member initialization folder. This keeps the same input inode available through final provenance checks even if the producing run removes its own path. Put the shared input on the work-directory filesystem.

## Response products

The normal S2S products stage includes ocean response products. They can also be produced independently after a completed member:

```
python -m hexcore.ocean_products --work-dir PATH
```

The door reads history through the Rust NetCDF bridge, reduces budgets and remaps on the GPU, writes CF through the Rust bridge, and calls `gpuwm.rustwx.run_cf_renderer`. Apply `../engine-patches/cf-physical-scale.patch` to the engine and rebuild `rw_wrfbatch`. Point `GPUWM_HEX_RW_WRFBATCH` at that renderer. A JSON product table supplies shared signed SST and ice-change ladders, including small changes around zero. Maps cover each weekly endpoint and the final endpoint; a CSV records every history endpoint.

## Source and licence

The new coupling code and renderer patch are original Apache-2.0 code under the repository licence. The wind-mixing equation family is shared with `cap-slab-ocean/gpuwm/core/kernels/oml.cu`, which identifies WRF v4.6.1 `phys/module_sf_oml.F`. WRF's source licence states that it is "in the public domain" and permits use "for any purpose without any fee or charge". The WRF notice already ships in `NOTICE`; upstream source licence: https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/LICENSE.txt . No new external model implementation is vendored.

Qualification results, exact commands, resource measurements, inspected maps and remaining gaps belong to the dated capability report, not this implementation description.

The 2026-10-20 capability run completed one GFS-initialized global member for 30 days on 40,962 cells and 55 atmospheric levels. All 4,320 steps were finite, with no future prescribed surface records. The binary64 state closed the cumulative energy budget to a maximum absolute residual of 0.0401 J/m2 at day 30. Ten SST and ice-change maps were rendered through the Rust renderer and inspected. The delivered capability report contains the source pins, daily budgets, resource samples, stage timings and figures. This qualifies the default prognostic response path; it is not an observational forecast-skill result.
