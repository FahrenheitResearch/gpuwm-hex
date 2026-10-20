# Physics suites

A hex forecast runs one of two kinds of physics. **MPAS-A v8.4.2's own suites**
(`mesoscale_reference`, `convection_permitting`) run through the ported MPAS
physics driver on the MPAS route (`--physics-backend mpasphys`), scheme by
scheme either as native MPAS Fortran or as hex's CUDA port, and are matched
against native MPAS bit for bit: see [the MPAS physics seam](manual/mpas-physics-seam.md)
and the README's *Running an MPAS physics suite*. **ArWen's physics**, the
default, is described on this page: a **named suite** plus optional
**overrides**, read from one table,
`src/hexcore/data/profiles/mpas-physics-suites.v1.json`. The dycore never reads
it. ArWen's physics is graded against observations (MRMS, ASOS), not against
MPAS's physics code.

## The suites

| suite | what runs | seam |
|---|---|---|
| `arwen_mpas_column` (default) | WSM6; hex's convection rule by mesh spacing; YSU over the revised MM5 surface layer; Noah-MP (4 layers); hex's YSU gravity-wave drag; legacy RRTMG with the Xu-Randall cloud fraction | frozen column batch, row `wsm6_column` |
| `arwen_thompson_column` | the default with aerosol-aware Thompson (WRF mp=28) | frozen column batch, row `thompson_aero_column` |
| `rrfs_v2_dev` | shaped after RRFS v2 (`hrrrv5`): aerosol-aware Thompson; no cumulus (New Tiedtke on the 12 km meshes); MYNN-EDMF over the MYNN surface layer; RUC on 9 soil levels; the GSL orographic drag; legacy RRTMG with MYNN's subgrid cloud | column-suite seam, row `mynn_ruc_gsl_column` |

On this ArWen route, MPAS's own suite names (`mesoscale_reference`,
`convection_permitting`, `hrrrv5`) also resolve to ArWen's implementation of
their schemes, exactly as MPAS resolves the selectors; whatever the ArWen
route cannot run in them is refused by name. For MPAS's own schemes, use the
MPAS route above. `rrfs_v2_dev` lists every known difference from
RRFS v2 in its table row, and the run's receipt carries the list.

With no suite flag at all, a run is the run a bare door made before named
suites: same backend row, same convection request, same driver arguments, same
configuration (bitwise on the card: `evidence/physics-suites-20261007/`).

## Front doors

```
gpuwm-hex forecast ... --physics-suite rrfs_v2_dev
gpuwm-hex forecast ... --physics-suite arwen_mpas_column --physics config_microp_scheme=mp_thompson_aerosols
gpuwm-hex forecast ... --physics-namelist namelist.atmosphere
gpuwm-hex physics list [--mesh NAME]
gpuwm-hex physics resolve --physics-suite NAME [--physics KEY=VALUE ...] [--mesh NAME]
gpuwm-hex physics import namelist.atmosphere [--mesh NAME]
```

`--physics KEY=VALUE` takes an MPAS selector (`config_microp_scheme`,
`config_convection_scheme`, `config_pbl_scheme`, `config_sfclayer_scheme`,
`config_lsm_scheme`, `config_gwdo_scheme`, `config_radt_lw_scheme`,
`config_radt_sw_scheme`, `config_radt_cld_scheme`), a WOOF generation selector
(`thompson_version`, `thompson_fork_snow_fall`, `bl_mynn_version`,
`bl_mynn_mixlength`, `mynn_sfclay_variant`, `rrtmg_cloud_optics_form`) or
`config_frac_seaice`. `--convection` given explicitly is the same override as
`--physics config_convection_scheme=...`. A selection behind the column-suite
seam that no profile names is assembled from the option rows and runs as its
own backend row.

Behind the column-suite seam the `auto` cumulus request is decided at the
door from the mesh row's spacing; where it selects New Tiedtke (20 km and
coarser), which that seam does not carry yet, the request is refused before
the card. A preset that holds the surface layer and PBL (`--preset fast`) was
measured on the frozen suite only and is refused with column-suite physics
unless the cadence is named with `--pbl-cadence`; the receipt of a
column-suite run does not repeat the preset row's YSU/Noah-MP suite as though
it ran.

`--physics-namelist` reads a `namelist.atmosphere`. Every option ends in one
disposition: admitted, ignored as MPAS ignores it (a TEMPO knob when the
microphysics is not TEMPO, a stochastic parameter when stochastic physics is
off), handed to the v8.4.1 dycore namelist door, outside (I/O, decomposition,
restart, printout, soundings, assimilation, the calendar), or refused with the
breakage and the follow-up. Any refusal refuses the run; `--physics` on top of
a namelist is read as if the namelist had said it.

An option the namelist leaves out is read at MPAS's Registry default, as MPAS
runs it, and its disposition says so. A namelist with no
`config_physics_suite` is `mesoscale_reference` (New Tiedtke and Noah, so it
refuses until those are overridden), not hex's default; no radiation interval
is MPAS's 30 min, not hex's 600 s; no `config_frac_seaice` is true. The
defaults are those of MPAS-A v8.4.1, or of the RRFS fork (1d6e370e) for a
namelist naming `hrrrv5`; both are rows in the table (`namelist.registry_defaults`).
A namelist naming a hex suite reads no Registry: that suite's own row decides.
MYNN's mixing switches (`config_mynn_mixscalars`, `config_mynn_mixaerosols`,
`config_mynn_mixnumcon`) are held only when the boundary layer is MYNN, the
only scheme MPAS reads them for.

The namelist's dynamics (`&nhyd_model`, `&damping`, `&limited_area`) must be
the run's own.  `config_dt`, `config_time_integration_order`,
`config_number_of_sub_steps` and `config_monotonic` are taken from the
namelist when their flags (`--dt`, `--time-integration-order`,
`--acoustic-substeps`, `--monotonic`) are absent; the door builds the run's
configuration and refuses, in one message, every other dynamics option it
would not execute, naming the flag that carries it (`config_mpas_cam_coef`
and `config_number_cam_damping_levels` are `--cam-coefficient` and
`--cam-damping-levels`, `config_rayleigh_damp_u` is `--rayleigh-u`,
`config_zd` is `--damping-start-m`). RRFS v2's 12 km namelist therefore runs
with `--cam-coefficient 2.0 --cam-damping-levels 8 --rayleigh-u
--damping-start-m 16000`, its schedule coming from the namelist itself. `config_epssm` is read and ignored, as MPAS-A v8.4.1 ignores it. `gpuwm-hex physics import`
prints the verdict per option. RRFS v2's own namelist (rendered) refuses
exactly two options: `mp_tempo` (TEMPO has no port in the engine) and
`config_mynn_mixaerosols = 1` (the column seam has no tendency lanes for the
aerosol numbers).

## Inputs a suite needs

* RUC (`sf_ruc`) integrates 9 soil levels: the init must carry them
  (`init_atmosphere` with `config_nsoillevels = 9`); the run checks the init's
  `zs` against RUC's depths.
* The GSL drag (`bl_ugwp_gwdo`) reads the mesh's `<mesh>.ugwp_oro_data.nc`;
  the door looks beside `--grid`, or takes `--orographic-statistics FILE`.
* Aerosol-aware Thompson reads `freezeH2O.dat` and `qr_acr_qg_V4.dat`, which no
  wheel carries: run `gpuwm fetch-tables` once. The door refuses before the
  card if they are absent.

## Adding a suite

Add a row to `suites` in the table. A new MPAS spelling of a scheme the seams
carry is a row in `options`. A scheme the seams do not carry is a refused row
naming what breaks and what would lift it. No code path keys on a suite name;
`tests/test_physics_suites.py` holds the table to its schema.
