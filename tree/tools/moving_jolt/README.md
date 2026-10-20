# Moving-region jolt experiments

The research scripts behind `docs/moving-jolt-report.md`: synthetic dry experiments, not the
original tropical-cyclone decks. The headline reduced test (the 522-cell refined family, ordinary
and LTS executors, fixed / baseline / balanced arms) is packaged and needs nothing from this folder:

```bash
gpuwm-hex moving check --device gpu --lts --out moving-check-lts
gpuwm-hex moving check --device gpu --out moving-check
gpuwm-hex moving check --device cpu --out moving-check-cpu
```

The scripts here separate the causes (CPU force fixtures, local vs global terrain reconciliation,
rates-only LTS changes, geometry identity of a rebuilt mesh). Run them from a working folder that
holds `experiments/`, `evidence/`, `bridges/` and `tmp/`, with this repository checked out as `hex/`:

```bash
mkdir -p experiments evidence bridges tmp
cp hex/tree/tools/moving_jolt/*.py hex/tree/tools/moving_jolt/setup-env.sh experiments/
```

Copy `rw_mpas_mesh`, `rw_mpas_geometry` and `rw_mpas_hostprep` from the installed gpuwm wheel's
`gpuwm/libexec/bridges/` directory into `bridges/` (they must be executable). The optional CPU
overlap helper:

```bash
g++ -O3 -shared -fPIC hex/tree/tools/spherical_overlap.cpp -o bridges/spherical_overlap.so
source experiments/setup-env.sh
export GPUWM_HEX_SPHERICAL_OVERLAP_LIB="$PWD/bridges/spherical_overlap.so"
```

Build the fixture meshes (132-cell base and displaced family; 522-cell refined family):

```bash
bridges/rw_mpas_mesh --out experiments/grid.nc --background-km 2500 --cells 128 --fit-spacing yes --sweeps 2 --tolerance .1 --clobber
python experiments/regenerate.py
python experiments/refined_family.py
```

CPU numerical experiments:

```bash
python experiments/cpu_force.py > evidence/cpu-force.log
python experiments/regenerated_force.py > evidence/regenerated-force.log
python experiments/regenerated_force.py --refined > evidence/refined-force.log
python experiments/integrate.py --steps 80 --out cpu-three-moves
python experiments/integrate.py --family --steps 80 --out cpu-regenerated-three-moves
```

GPU experiments (one card; on a shared host, reserve it with the host's own queue first):

```bash
python experiments/integrate.py --gpu --out gpu-three-moves
python experiments/integrate.py --gpu --family --out gpu-regenerated-three-moves
python experiments/integrate.py --gpu --refined --out gpu-refined-three-moves
python experiments/integrate.py --gpu --refined --lts --rate-shift .25 --out gpu-refined-lts-three-moves
python experiments/integrate.py --gpu --family --lts --rate-shift .25 --out gpu-lts-three-moves
python experiments/integrate.py --gpu --family --lts --rate-shift .25 --rates-only --arms fixed,rates --out gpu-lts-rates-only
```

Each integration runs fixed, baseline remap and corrected remap arms. Moves occur at 600, 1200 and
1800 s; the run ends at 2400 s. `--family` uses three locally displaced 132-cell grids, `--refined`
the 522-cell moving refinement cap, `--lts` the production two-rate executor with RK2 and scalar
reflux (GPU only). `--rate-shift .25` moves the class boundary at each remesh; `--rates-only --arms
fixed,rates` isolates it. `--arms fixed` produces identity controls: compare `final_hashes`,
`saved_hashes` and `rows` (ignoring wall time) between two checkouts, for example the revision
before the fix and this one. `summarize.py` collects the conservation bounds of the named records.

The production height-remap fix is on by default; `GPUWM_HEX_MOVING_BALANCE=0` reproduces the
previous thermodynamic path. Fixed-region execution does not call the changed moving-remap code.
No terrain smoothing, divergence damping or new time filter is enabled in these fixtures, and the
native split-explicit integrator itself is unchanged.

The metric is the maximum post-move |w| or surface-pressure tendency in the next 600 seconds
divided by the matching fixed window, with 30-second samples. The instantaneous cross-mesh pressure
change is recorded separately, along with that change after subtracting the reference atmosphere's
terrain effect. Neither is silently included in, or substituted for, the post-move metric.
