#!/usr/bin/env bash
set -euo pipefail

# Run the frozen v8.2.3 model itself on the published x1.2562 mesh using the
# analytic Jablonowski-Williamson perturbation (init case 2).  This requires no
# external meteorological or WPS geography data and is the fastest path to a
# whole-step and first-day Fortran authority artifact.

repo_root="${1:?usage: run_frozen_jw.sh REPOSITORY_ROOT [all|init|step|day|receipt]}"
mode="${2:-all}"
# Site-specific roots.  An empty authority root would place the run under
# "/runs/..." at the filesystem root, so refuse by name instead.
authority_root="${MPAS_AUTHORITY_ROOT:?set MPAS_AUTHORITY_ROOT to the MPAS authority checkout root}"
env_setup="${MPAS_ENV_SETUP:?set MPAS_ENV_SETUP to the MPAS dependency environment setup script}"
run_id="${RUN_ID:-jw-x1.2562-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${authority_root}/runs/${run_id}"
mesh_dir="${repo_root}/data/meshes/x1.2562"
receipt_dir="${repo_root}/receipts/frozen-v8.2.3/${run_id}"
ranks="${MPAS_RANKS:-1}"

case "${mode}" in
  all|init|step|day|receipt) ;;
  *) echo "mode must be all, init, step, day, or receipt" >&2; exit 2 ;;
esac
for executable in init_atmosphere_model atmosphere_model; do
  if [[ ! -x "${authority_root}/bin/${executable}" ]]; then
    echo "missing frozen executable ${authority_root}/bin/${executable}" >&2
    exit 2
  fi
done

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "${env_setup}"
export LD_LIBRARY_PATH="${PNETCDF}/lib:${PIO}/lib:${NETCDF}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${run_root}" "${receipt_dir}"

link_mesh() {
  local target="$1"
  ln -sfn "${mesh_dir}/x1.2562.grid.nc" "${target}/x1.2562.grid.nc"
  ln -sfn "${mesh_dir}/x1.2562.graph.info" "${target}/x1.2562.graph.info"
  local partition
  for partition in 2 4 6 8 12 16; do
    ln -sfn "${mesh_dir}/x1.2562.graph.info.part.${partition}" \
      "${target}/x1.2562.graph.info.part.${partition}"
  done
}

write_init_configuration() {
  local target="$1"
  cat >"${target}/namelist.init_atmosphere" <<'EOF'
&nhyd_model
    config_init_case = 2
    config_start_time = '2000-01-01_00:00:00'
    config_stop_time = '2000-01-01_00:00:00'
    config_theta_adv_order = 3
    config_coef_3rd_order = 0.25
    config_interface_projection = 'linear_interpolation'
/
&dimensions
    config_nvertlevels = 15
    config_nsoillevels = 4
    config_nfglevels = 38
    config_nfgsoillevels = 4
    config_gocartlevels = 30
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
&interpolation_control
    config_extrap_airtemp = 'lapse-rate'
/
&preproc_stages
    config_static_interp = false
    config_native_gwd_static = false
    config_vertical_grid = false
    config_met_interp = false
    config_input_sst = false
    config_frac_seaice = false
/
&io
    config_pio_num_iotasks = 0
    config_pio_stride = 1
/
&decomposition
    config_block_decomp_file_prefix = 'x1.2562.graph.info.part.'
/
EOF
  cat >"${target}/streams.init_atmosphere" <<'EOF'
<streams>
<immutable_stream name="input"
                  type="input"
                  filename_template="x1.2562.grid.nc"
                  input_interval="initial_only" />
<immutable_stream name="output"
                  type="output"
                  filename_template="x1.2562.init.nc"
                  packages="initial_conds"
                  output_interval="initial_only" />
<immutable_stream name="surface"
                  type="output"
                  filename_template="x1.2562.sfc_update.nc"
                  filename_interval="none"
                  packages="sfc_update"
                  output_interval="none" />
<immutable_stream name="lbc"
                  type="output"
                  filename_template="lbc.$Y-$M-$D_$h.$m.$s.nc"
                  filename_interval="output_interval"
                  packages="lbcs"
                  output_interval="none" />
</streams>
EOF
}

run_init() {
  local target="${run_root}/init"
  mkdir -p "${target}"
  link_mesh "${target}"
  ln -sfn "${authority_root}/bin/init_atmosphere_model" "${target}/init_atmosphere_model"
  write_init_configuration "${target}"
  cd "${target}"
  mpirun -np "${ranks}" ./init_atmosphere_model >init.stdout.log 2>init.stderr.log
  test -s x1.2562.init.nc
}

write_atmosphere_configuration() {
  local target="$1"
  local duration="$2"
  local output_interval="$3"
  cat >"${target}/namelist.atmosphere" <<EOF
&nhyd_model
    config_time_integration_order = 3
    config_dt = 600.0
    config_start_time = '2000-01-01_00:00:00'
    config_run_duration = '${duration}'
    config_split_dynamics_transport = true
    config_number_of_sub_steps = 6
    config_dynamics_split_steps = 1
    config_horiz_mixing = '2d_smagorinsky'
    config_visc4_2dsmag = 0.05
    config_scalar_advection = true
    config_monotonic = true
    config_coef_3rd_order = 0.25
    config_epssm = 0.1
    config_smdiv = 0.1
/
&damping
    config_zd = 22000.0
    config_xnutr = 0.2
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
    config_IAU_window_length_s = 21600.0
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
  cat >"${target}/streams.atmosphere" <<EOF
<streams>
<immutable_stream name="input"
                  type="input"
                  filename_template="x1.2562.init.nc"
                  input_interval="initial_only" />
<immutable_stream name="restart"
                  type="input;output"
                  filename_template="restart.\$Y-\$M-\$D_\$h.\$m.\$s.nc"
                  input_interval="initial_only"
                  output_interval="none" />
<stream name="output"
        type="output"
        filename_template="history.\$Y-\$M-\$D_\$h.\$m.\$s.nc"
        output_interval="${output_interval}">
  <file name="stream_list.atmosphere.output"/>
</stream>
<stream name="diagnostics"
        type="output"
        filename_template="diag.\$Y-\$M-\$D_\$h.\$m.\$s.nc"
        output_interval="${output_interval}">
  <file name="stream_list.atmosphere.diagnostics"/>
</stream>
<stream name="surface"
        type="input"
        filename_template="x1.2562.sfc_update.nc"
        filename_interval="none"
        input_interval="none">
  <file name="stream_list.atmosphere.surface"/>
</stream>
<immutable_stream name="iau"
                  type="input"
                  filename_template="x1.2562.AmB.\$Y-\$M-\$D_\$h.\$m.\$s.nc"
                  filename_interval="none"
                  packages="iau"
                  input_interval="initial_only" />
<immutable_stream name="lbc_in"
                  type="input"
                  filename_template="lbc.\$Y-\$M-\$D_\$h.\$m.\$s.nc"
                  filename_interval="input_interval"
                  packages="limited_area"
                  input_interval="3:00:00" />
</streams>
EOF
}

run_atmosphere() {
  local label="$1"
  local duration="$2"
  local output_interval="$3"
  local target="${run_root}/${label}"
  mkdir -p "${target}"
  link_mesh "${target}"
  ln -sfn "${authority_root}/bin/atmosphere_model" "${target}/atmosphere_model"
  ln -sfn "${run_root}/init/x1.2562.init.nc" "${target}/x1.2562.init.nc"
  ln -sfn "${authority_root}/source/stream_list.atmosphere.output" \
    "${target}/stream_list.atmosphere.output"
  ln -sfn "${authority_root}/source/stream_list.atmosphere.diagnostics" \
    "${target}/stream_list.atmosphere.diagnostics"
  ln -sfn "${authority_root}/source/stream_list.atmosphere.surface" \
    "${target}/stream_list.atmosphere.surface"
  write_atmosphere_configuration "${target}" "${duration}" "${output_interval}"
  cd "${target}"
  mpirun -np "${ranks}" ./atmosphere_model >atmosphere.stdout.log 2>atmosphere.stderr.log
}

write_receipt() {
  if ! python3 -c 'import netCDF4, numpy' >/dev/null 2>&1; then
    {
      echo "evidence=frozen MPAS-A v8.2.3 executable"
      echo "run_root=${run_root}"
      echo "netcdf_python_receipt=unavailable (hash/log receipt follows)"
      find "${run_root}" -type f -name '*.nc' -print0 \
        | sort -z | xargs -0 sha256sum
      grep -H "Finished running" "${run_root}"/*/log.*.out || true
      grep -H "Warning messages\|Error messages\|Critical error messages" \
        "${run_root}"/*/log.*.out || true
    } >"${receipt_dir}/run-receipt.txt"
    cat "${receipt_dir}/run-receipt.txt"
    return
  fi
  python3 - "${run_root}" "${receipt_dir}/run-receipt.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
from netCDF4 import Dataset
import numpy as np

run_root = Path(sys.argv[1])
output = Path(sys.argv[2])
record = {"evidence": "frozen MPAS-A v8.2.3 executable", "run_root": str(run_root), "files": {}}
for path in sorted(run_root.rglob("*.nc")):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    item = {"bytes": path.stat().st_size, "sha256": digest, "dimensions": {}, "fields": {}}
    with Dataset(path) as dataset:
        item["dimensions"] = {name: len(dim) for name, dim in dataset.dimensions.items()}
        for name in ("rho", "theta", "u", "w", "pressure", "zgrid", "qv"):
            if name not in dataset.variables:
                continue
            values = np.asarray(dataset.variables[name][:])
            if not np.issubdtype(values.dtype, np.number):
                continue
            finite = np.isfinite(values)
            item["fields"][name] = {
                "shape": list(values.shape),
                "finite": int(finite.sum()),
                "count": int(values.size),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
    item["relative_path"] = str(path.relative_to(run_root))
    record["files"][item["relative_path"]] = item
for log in sorted(run_root.rglob("log.atmosphere.*.out")):
    text = log.read_text(errors="replace")
    record.setdefault("model_logs", {})[str(log.relative_to(run_root))] = {
        "bytes": log.stat().st_size,
        "completed": "Finished running the atmosphere core" in text,
        "critical_count": text.lower().count("critical"),
        "nan_count": text.lower().count("nan"),
        "tail": text.splitlines()[-20:],
    }
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

if [[ "${mode}" == all || "${mode}" == init ]]; then
  run_init
fi
if [[ "${mode}" == all || "${mode}" == step ]]; then
  test -s "${run_root}/init/x1.2562.init.nc" || run_init
  run_atmosphere step '0_00:10:00' '0_00:10:00'
fi
if [[ "${mode}" == all || "${mode}" == day ]]; then
  test -s "${run_root}/init/x1.2562.init.nc" || run_init
  run_atmosphere day '1_00:00:00' '0_06:00:00'
fi
write_receipt
