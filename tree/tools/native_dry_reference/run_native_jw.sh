#!/usr/bin/env bash
# Native dry Jablonowski-Williamson runs on the public x1.2562 mesh, one rank, with
# the prognostic and saved-diagnostic fields written after every model step.
#
# usage: run_native_jw.sh WORK_ROOT init BUILD     (BUILD = TAG-RECIPE; makes the one init file)
#        run_native_jw.sh WORK_ROOT run  BUILD ARM ORDER SUBSTEPS
#
# The namelists are those of verification/mpas-842-repin-20261011/native-jw.sh:
# 26 levels, dt 1800 s, 12 steps, no mixing, no physics, monotonic third-order
# transport, three dynamics split steps.
set -euo pipefail
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
W=$1; MODE=$2; BUILD=$3
MESH=${MESH:-x1.2562}; LEVELS=${LEVELS:-26}; START=${START:-2000-01-01_00:00:00}
if [[ $BUILD = *-ifx-* ]]; then
  set +u; source "${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}" >/dev/null 2>&1; set -u
fi
M=$W/mesh; R=$W/jw
if [ ! -f "$M/$MESH.grid.nc" ]; then
  mkdir -p "$M"; cd "$M"
  curl -fsSLO https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes/$MESH.tar.gz
  tar xzf $MESH.tar.gz
fi
if [ "$MESH" = x1.2562 ]; then
  PIN_FILE="$SCRIPT_DIR/REFERENCE-PINS.json"
  GRID_PIN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["grid_sha256_file"])' "$PIN_FILE")
  echo "$GRID_PIN  $M/$MESH.grid.nc" | sha256sum -c -
fi
if [ "$MODE" = init ]; then
  T=$R/${INIT_SUBDIR:-init}; mkdir -p "$T"; cd "$T"
  ln -sfn "${GRID_FILE:-$M/$MESH.grid.nc}" "$MESH.grid.nc"
  ln -sfn "$M/$MESH.graph.info" .
  ln -sfn "$W/native/$BUILD-init_atmosphere/init_atmosphere_model" .
  cat > namelist.init_atmosphere <<NML
&nhyd_model
    config_init_case = ${INIT_CASE:-2}
    config_start_time = "$START"
    config_stop_time = "$START"
    config_theta_adv_order = 3
    config_coef_3rd_order = 0.25
/
&dimensions
    config_nvertlevels = $LEVELS
    config_nsoillevels = 4
    config_nfglevels = 38
    config_nfgsoillevels = 4
/
&data_sources
    config_fg_interval = 21600
    config_use_spechumd = false
/
&vertical_grid
    config_ztop = 30000.0
    config_nsmterrain = 1
    config_smooth_surfaces = true
    config_dzmin = 0.3
    config_nsm = 30
    config_tc_vertical_grid = true
    config_blend_bdy_terrain = false
/
&preproc_stages
    config_static_interp = false
    config_native_gwd_static = false
    config_vertical_grid = ${VERTICAL_GRID:-false}
    config_met_interp = false
    config_input_sst = false
    config_frac_seaice = false
/
&io
    config_pio_num_iotasks = 0
    config_pio_stride = 1
/
&decomposition
    config_block_decomp_file_prefix = "$MESH.graph.info.part."
/
NML
  cat > streams.init_atmosphere <<XML
<streams>
<immutable_stream name="input" type="input" filename_template="$MESH.grid.nc" input_interval="initial_only" />
<immutable_stream name="output" type="output" filename_template="$MESH.init.nc" packages="initial_conds" output_interval="initial_only" />
<immutable_stream name="surface" type="output" filename_template="$MESH.sfc_update.nc" filename_interval="none" packages="sfc_update" output_interval="none" />
<immutable_stream name="lbc" type="output" filename_template="lbc.\$Y-\$M-\$D_\$h.\$m.\$s.nc" filename_interval="output_interval" packages="lbcs" output_interval="none" />
</streams>
XML
  mpirun -np 1 ./init_atmosphere_model > init.stdout.log 2> init.stderr.log
  test -s $MESH.init.nc
  "${PYTHON:-python3}" "$SCRIPT_DIR/run_receipt.py" write-init "$T"
  echo "init made by $BUILD" > MADE-BY.txt; exit 0
fi
ARM=$4; ORDER=$5; SUB=$6; DT=${DT:-1800.0}; DUR=${DUR:-0_06:00:00}; OUT=${OUT:-0_00:30:00}
MIXING=${MIXING:-off}
SMDIV=${SMDIV:-0.0}; XNUTR=${XNUTR:-0.0}
if [ "$MIXING" = off ]; then HORIZ_MIXING=2d_fixed
elif [ "$MIXING" = smag ]; then HORIZ_MIXING=2d_smagorinsky
else echo "unsupported mixing: $MIXING" >&2; exit 2; fi
S=$W/native/$BUILD-atmosphere; T=$R/$BUILD/$ARM
test ! -d "$T" || { echo "Refusing to overwrite existing run: $T" >&2; exit 2; }
mkdir -p "$T"; cd "$T"
ln -s "$S/atmosphere_model" .
ln -s "${INIT_FILE:-$R/init/$MESH.init.nc}" "$MESH.init.nc"
ln -s "$M/$MESH.graph.info" .
RANKS=${RANKS:-1}
if [ "$RANKS" -gt 1 ]; then
  ln -s "$M/$MESH.graph.info.part.$RANKS" .
fi
ln -s "$S/stream_list.atmosphere.surface" . 2>/dev/null || true
cat > namelist.atmosphere <<NML
&nhyd_model
    config_time_integration_order = $ORDER
    config_dt = $DT
    config_start_time = "$START"
    config_run_duration = "$DUR"
    config_split_dynamics_transport = true
    config_number_of_sub_steps = $SUB
    config_dynamics_split_steps = 3
    config_horiz_mixing = "$HORIZ_MIXING"
    config_len_disp = 0.0
    config_visc4_2dsmag = 0.05
    config_smagorinsky_coef = 0.125
    config_del4u_div_factor = 10.0
    config_h_ScaleWithMesh = true
    config_h_theta_eddy_visc2 = 0.0
    config_v_theta_eddy_visc2 = 0.0
    config_h_mom_eddy_visc2 = 0.0
    config_v_mom_eddy_visc2 = 0.0
    config_h_theta_eddy_visc4 = 0.0
    config_h_mom_eddy_visc4 = 0.0
    config_scalar_advection = true
    config_monotonic = true
    config_positive_definite = false
    config_scalar_adv_order = 3
    config_scalar_vadv_order = 3
    config_coef_3rd_order = 0.25
    config_apvm_upwinding = 0.5
    config_smdiv = $SMDIV
/
&damping
    config_zd = 22000.0
    config_xnutr = $XNUTR
/
&limited_area
    config_apply_lbcs = false
/
&io
    config_pio_num_iotasks = 0
    config_pio_stride = 1
/
&decomposition
    config_block_decomp_file_prefix = "$MESH.graph.info.part."
/
&restart
    config_do_restart = false
/
&printout
    config_print_global_minmax_vel = true
    config_print_detailed_minmax_vel = false
/
&IAU
    config_IAU_option = "off"
/
&physics
    config_sst_update = false
    config_sstdiurn_update = false
    config_deepsoiltemp_update = false
    config_bucket_update = "none"
    config_physics_suite = "none"
/
&soundings
    config_sounding_interval = "none"
/
NML
cat > streams.atmosphere <<XML
<streams>
<immutable_stream name="input" type="input" filename_template="$MESH.init.nc" input_interval="initial_only" />
<immutable_stream name="restart" type="input;output" filename_template="restart.\$Y-\$M-\$D_\$h.\$m.\$s.nc" input_interval="initial_only" output_interval="none" />
<stream name="tracer_oracle" type="output" filename_template="tracer.\$Y-\$M-\$D_\$h.\$m.\$s.nc" output_interval="$OUT">
  <var name="rho_zz"/> <var name="theta_m"/> <var name="ru"/> <var name="rw"/> <var name="u"/> <var name="w"/>
  <var name="rho_base"/> <var name="theta_base"/> <var name="rtheta_base"/> <var name="rho_p"/> <var name="rtheta_p"/>
  <var name="exner"/> <var name="exner_base"/> <var name="pressure_p"/> <var name="pressure_base"/>
  <var_array name="scalars"/>
$(for v in ${EXTRA_STREAM_VARS:-}; do echo "  <var name=\"$v\"/>"; done)
</stream>
<stream name="surface" type="input" filename_template="$MESH.sfc_update.nc" filename_interval="none" input_interval="none">
  <file name="stream_list.atmosphere.surface"/>
</stream>
<immutable_stream name="iau" type="input" filename_template="$MESH.AmB.\$Y-\$M-\$D_\$h.\$m.\$s.nc" filename_interval="none" packages="iau" input_interval="initial_only" />
<immutable_stream name="lbc_in" type="input" filename_template="lbc.\$Y-\$M-\$D_\$h.\$m.\$s.nc" filename_interval="input_interval" packages="limited_area" input_interval="3:00:00" />
</streams>
XML
mpirun -np "$RANKS" ./atmosphere_model > atmosphere.stdout.log 2> atmosphere.stderr.log
grep -q "Finished running the atmosphere core" log.atmosphere.0000.out
"${PYTHON:-python3}" "$SCRIPT_DIR/run_receipt.py" write "$T"
echo "$BUILD $ARM ok $(ls tracer.*.nc | wc -l) files"
