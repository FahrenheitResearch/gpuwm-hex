#!/bin/bash
# Build one real-terrain identity case on any public MPAS quasi-uniform mesh.
# usage: prepare_identity_case.sh ROOT MESH MESH_DIR MOISTURE [SOURCE_INIT] [START]
#   ROOT     case root (mesh/, native/, jw/ are created under it)
#   MESH     e.g. x1.40962, x1.163842, x1.2621442
#   MESH_DIR directory holding MESH.grid.nc, MESH.graph.info(.part.N)
#   MOISTURE dry | qv | condensate   (prepare_real_case.py --moisture)
# Terrain: the source analysis' lowest zgrid, nearest-column, then native
# init_atmosphere case 7 smoothing (config_nsmterrain=1, config_smooth_surfaces)
# and the 55-level vertical grid -- the dycore-match recipe of prepare-forecast.sh.
set -euo pipefail
ROOT=$1; MESH=$2; MESH_DIR=$3; MOISTURE=$4
SOURCE=${5:?SOURCE_INIT is required: an ordered init.nc whose lowest zgrid gives the terrain}
START=${6:-2023-03-31_12:00:00}
HERE=$(dirname "$(readlink -f "$0")")
NAT=${NATIVE_BUILDS:?set NATIVE_BUILDS to the native reference builds (build_all_references.sh output)}
# HEX_DYCORE_ENV: optional prefix holding venv/ and deps/ (MPI, PnetCDF) for the native tools
if [ -n "${HEX_DYCORE_ENV:-}" ]; then
  export PATH=$HEX_DYCORE_ENV/venv/bin:$HEX_DYCORE_ENV/deps/bin:$PATH
  export LD_LIBRARY_PATH=$HEX_DYCORE_ENV/deps/lib:${LD_LIBRARY_PATH:-}
fi
mkdir -p "$ROOT/mesh" "$ROOT/native"
for f in "$MESH_DIR/$MESH".grid.nc "$MESH_DIR/$MESH".graph.info*; do ln -sfn "$f" "$ROOT/mesh/$(basename "$f")"; done
for b in v8.4.2-gnu-o0-init_atmosphere v8.4.2-gnu-strict-atmosphere v8.4.2-gnu-o3-atmosphere; do
  ln -sfn "$NAT/$b" "$ROOT/native/$b"; done
T="$HERE/run_native_jw.sh"
if [ ! -f "$ROOT/terrain.static.nc" ]; then
  [ -f "$ROOT/jw/terrain-base/$MESH.init.nc" ] || MESH=$MESH INIT_SUBDIR=terrain-base bash "$T" "$ROOT" init v8.4.2-gnu-o0
  python - "$ROOT" "$MESH" "$SOURCE" <<'PY'
import shutil,sys,numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from netCDF4 import Dataset
root,mesh,source=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
tmp=root/'terrain.static.nc.part'
shutil.copyfile(root/f'jw/terrain-base/{mesh}.init.nc',tmp)
with Dataset(source) as src,Dataset(tmp,'r+') as dst:
    def xyz(d):
        lat=np.asarray(d['latCell'][:]);lon=np.asarray(d['lonCell'][:])
        return np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
    mapping=cKDTree(xyz(src)).query(xyz(dst))[1]
    dst['ter'][:]=np.asarray(src['zgrid'][:,0])[mapping]
tmp.rename(root/'terrain.static.nc')
PY
fi
[ -f "$ROOT/jw/template-terrain/$MESH.init.nc" ] || MESH=$MESH LEVELS=55 GRID_FILE="$ROOT/terrain.static.nc" INIT_CASE=7 VERTICAL_GRID=true \
  INIT_SUBDIR=template-terrain START=$START bash "$T" "$ROOT" init v8.4.2-gnu-o0
OUT="$ROOT/real-terrain-$MOISTURE.init.nc"
[ -f "$OUT" ] || python "$HERE/prepare_real_case.py" --source "$SOURCE" \
  --template "$ROOT/jw/template-terrain/$MESH.init.nc" --out "$OUT" \
  --receipt "$ROOT/real-terrain-$MOISTURE-init-receipt.json" --source-time "$START" --moisture "$MOISTURE"
echo "PREPARED $OUT"
