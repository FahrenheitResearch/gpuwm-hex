#!/usr/bin/env bash
set -euo pipefail

# Regeneration template for executing the already-audited stock v8.2.3 binary
# on a qv-only mutation of the immutable JW input.  No model source or
# executable is modified.  The selected archived authority namelist omitted
# scalar order and APVM lines and therefore used the same Registry defaults
# (3, 3, and 0.5); this template spells them out.  The fixture manifest hashes
# the actual archived namelist, not this semantically equivalent template.
repo_root="${1:?usage: run_nonzero_tracer_authority.sh REPOSITORY_ROOT INPUT_NC [RUN_ID]}"
input_nc="${2:?missing qv-mutated input NetCDF}"
run_id="${3:-jw-nonzero-tracer-v823-$(date -u +%Y%m%dT%H%M%SZ)}"
# Site-specific roots.  An empty authority root would create the run under
# "/runs/..." at the filesystem root, so refuse by name instead.
authority_root="${MPAS_AUTHORITY_ROOT:?set MPAS_AUTHORITY_ROOT to the MPAS authority checkout root}"
env_setup="${MPAS_ENV_SETUP:?set MPAS_ENV_SETUP to the MPAS dependency environment setup script}"
target="${authority_root}/runs/${run_id}"
mesh_dir="${repo_root}/data/meshes/x1.2562"

test -x "${authority_root}/bin/atmosphere_model"
test -s "${input_nc}"
if [[ -e "${target}" ]]; then
  echo "refusing to overwrite existing run directory ${target}" >&2
  exit 2
fi
mkdir -p "${target}"
ln -s "${authority_root}/bin/atmosphere_model" "${target}/atmosphere_model"
ln -s "${input_nc}" "${target}/x1.2562.init.nc"
ln -s "${mesh_dir}/x1.2562.graph.info" "${target}/x1.2562.graph.info"
ln -s "${authority_root}/source/stream_list.atmosphere.surface" \
  "${target}/stream_list.atmosphere.surface"

cat >"${target}/namelist.atmosphere" <<'EOF'
&nhyd_model
    config_time_integration_order = 3
    config_dt = 600.0
    config_start_time = '2000-01-01_00:00:00'
    config_run_duration = '0_00:10:00'
    config_split_dynamics_transport = true
    config_number_of_sub_steps = 6
    config_dynamics_split_steps = 1
    config_horiz_mixing = '2d_fixed'
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
    config_epssm = 0.1
    config_smdiv = 0.0
/
&damping
    config_zd = 22000.0
    config_xnutr = 0.0
/
&limited_area
    config_apply_lbcs = false
/
&io
    config_pio_num_iotasks = 0
    config_pio_stride = 1
/
&decomposition
    config_block_decomp_file_prefix = 'x1.2562.graph.info.part.'
/
&restart
    config_do_restart = false
/
&printout
    config_print_global_minmax_vel = true
    config_print_detailed_minmax_vel = false
/
&IAU
    config_IAU_option = 'off'
/
&physics
    config_sst_update = false
    config_sstdiurn_update = false
    config_deepsoiltemp_update = false
    config_bucket_update = 'none'
    config_physics_suite = 'none'
/
&soundings
    config_sounding_interval = 'none'
/
EOF

cat >"${target}/streams.atmosphere" <<'EOF'
<streams>
<immutable_stream name="input" type="input" filename_template="x1.2562.init.nc" input_interval="initial_only" />
<immutable_stream name="restart" type="input;output" filename_template="restart.$Y-$M-$D_$h.$m.$s.nc" input_interval="initial_only" output_interval="none" />
<stream name="tracer_oracle" type="output" filename_template="tracer.$Y-$M-$D_$h.$m.$s.nc" output_interval="0_00:10:00">
  <var name="rho_zz"/>
  <var name="theta_m"/>
  <var name="ru"/>
  <var name="rw"/>
  <var name="u"/>
  <var name="w"/>
  <var name="rho_base"/>
  <var name="theta_base"/>
  <var name="rtheta_base"/>
  <var name="rho_p"/>
  <var name="rtheta_p"/>
  <var name="exner"/>
  <var name="exner_base"/>
  <var name="pressure_p"/>
  <var name="pressure_base"/>
  <var_array name="scalars"/>
</stream>
<stream name="surface" type="input" filename_template="x1.2562.sfc_update.nc" filename_interval="none" input_interval="none">
  <file name="stream_list.atmosphere.surface"/>
</stream>
<immutable_stream name="iau" type="input" filename_template="x1.2562.AmB.$Y-$M-$D_$h.$m.$s.nc" filename_interval="none" packages="iau" input_interval="initial_only" />
<immutable_stream name="lbc_in" type="input" filename_template="lbc.$Y-$M-$D_$h.$m.$s.nc" filename_interval="input_interval" packages="limited_area" input_interval="3:00:00" />
</streams>
EOF

# shellcheck disable=SC1091
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${env_setup}"
export LD_LIBRARY_PATH="${PNETCDF}/lib:${PIO}/lib:${NETCDF}/lib:${LD_LIBRARY_PATH:-}"
cd "${target}"
mpirun -np 1 ./atmosphere_model >atmosphere.stdout.log 2>atmosphere.stderr.log
test -s tracer.2000-01-01_00.00.00.nc
test -s tracer.2000-01-01_00.10.00.nc
grep -q "Finished running the atmosphere core" log.atmosphere.0000.out

sha256sum \
  atmosphere_model namelist.atmosphere streams.atmosphere x1.2562.init.nc \
  tracer.2000-01-01_00.00.00.nc tracer.2000-01-01_00.10.00.nc \
  log.atmosphere.0000.out
echo "run_root=${target}"
