# gpuwm-hex

MPAS-Atmosphere v8.4.2 on the GPU. gpuwm-hex 0.4.1 ports the whole model to
CUDA: the dynamical core, both MPAS physics suites and their scheme options,
initialization and static fields, the incremental analysis update, limited-area
runs, streams, restart and diagnostics (the scheme list and what is not yet in
are in `tree/README.md`). Alongside it runs ArWen's own physics
(the default) and capabilities MPAS does not have: one-command global
forecasts, local time stepping and an experimental storm-following region. It
runs on one consumer card.

This is the repository. The distribution it builds lives in [`tree/`](tree/),
and its user documentation is [`tree/README.md`](tree/README.md) (the
capability contract: every claim with its scope) and the
[user manual](tree/docs/manual/index.md). This page says what the project is,
what it is matched against and how that is checked.

**Install:** `pip install gpuwm-hex`. It requires the ArWen engine, gpuwm 2.9
(`gpuwm>=2.9.4,<2.10`), and pip resolves it.

---

## What 0.4.1 claims, and how far

Each line is measured, and each carries its scope. Nothing outside the scope
is claimed. [`tree/README.md`](tree/README.md) gives the detail.

Kernels compile with fast numerics by default (FMA contraction, flush-to-zero); `--numerics strict` is the opt-in for bitwise identity to native MPAS-A, and every identity claim here was measured with it.

| Claim | Scope |
|---|---|
| The dynamical core is byte-identical to native MPAS-A v8.4.2 built with the pinned strict recipe (GNU Fortran, single precision, no FP contraction or vectorised reassociation). | `x1.40962`, 55 levels, real terrain, dry, 5 days, two dates; the same mesh with water vapour and with five condensates transported, on 1, 2 and 4 ranks of one card; `x1.163842` with water vapour; one dry step on `x1.2562`. Hex and the reference run on the same host C library. |
| A 24-hour forecast of the `convection_permitting` suite with all ten of its physics schemes on the GPU is bitwise identical to native MPAS-A v8.4.2 in every field hex writes: 545,733,285 values, 0 different. | `x1.2562`, one process, 2023-03-31 12Z, exact mode. Native also writes about 105 million values hex does not write yet; they are counted, not hidden. |
| Every other MPAS-A v8.4.2 physics scheme in this release is bitwise identical to native on real inputs captured from a native run, and a forecast with it on was compared with native, map by map. | Per scheme. A whole-forecast match is claimed only for the forecast above. |
| Correct mode is the default on every route. On the MPAS suites it fixes 5 defects found in native MPAS-A v8.4.2; on the ArWen route it applies the corrections the engine carries (WB4, WB5); `--physics-mode exact` reproduces native bit for bit. | Each fix is checked against its own native referee build. No combined referee is claimed. |
| Real-data initialization and the static builder match native field for field. IAU, limited-area boundaries, streams, restart and the diagnostics match native on real captured inputs. | Per unit, on the cases in `tree/README.md`. A few static words where native reads undefined memory are labelled declared residue, never identical. |
| `gpuwm-hex forecast --analysis` starts a forecast from an MPAS analysis bundle (direct insertion, IAU, cycling restart, remap). | Measured on the ArWen physics route. IAU also runs across a restart, on the explicit MPAS suite route, on same-host multi-card ranks and in member rosters; IAU on multi-card local time stepping is refused by name. |
| `gpuwm-hex global` runs a global forecast in one command. It is the main global model; gpuwm-global remains available as a second (spectral) global model. | Rows are defined at 120, 60, 30 and 15 km, on ArWen physics and on the gpuwm-global suite; the measured runs are the 120 km rows. |
| A storm-following 3 km region, experimental: `gpuwm-hex moving`. | ArWen physics, three 48-hour tropical cyclone runs; the balanced remap between moves is the default. |
| RRFS-MPAS runs end to end from a pip install: the GSL fork's dycore on the GPU, its ported physics on the GPU by default, the rest through the fork's native library, which `physics-resources build` compiles from the fork's source on your machine. | With every physics slot on the native library (`--physics-default-backend native-cpu`), a one-hour `hrrrv5` run on `x1.2562`, one process, is bitwise to the native fork build in every field hex writes; each GPU port is bitwise as a module replay. RRFS initial conditions come from the fork's own init (named gap). |

Named gaps, each with its reason, are listed in
[`tree/README.md`](tree/README.md#not-in-this-release).

---

## The two relationships that define this project

### To native MPAS-Atmosphere v8.4.2: matched wherever a match is reachable

The referee for anything that ports MPAS is native MPAS itself: the pinned
v8.4.2 source (commit `f34984b2`) built with a strict GNU recipe, run on the
same inputs, compared word for word. "Identical" here always means every
stored word, with no tolerance. Where hex cannot match (a value native reads
from undefined memory, a field hex does not write yet), the gap is counted and
named, never averaged away.

The reference is rebuildable from this repository:
[`tree/tools/native_dry_reference/`](tree/tools/native_dry_reference/) builds the
dycore reference, [`tree/tools/native_physics_reference/`](tree/tools/native_physics_reference/)
the physics reference, and [`tree/verification/gates/`](tree/verification/gates/)
holds the gate decks every change to the port must pass before it lands: the
dry dycore, both MPAS suites on the native backend, and the all-GPU forecast
above. A claim whose reference cannot be rebuilt is not made.

Native MPAS has defects of its own. Matching it exactly found real ones.
**Correct mode**, the default on every route, fixes the ones this release
carries; `--physics-mode exact` puts every one of them back. A fix that a
scheme does not carry yet is never a refusal: the run keeps native behaviour
for it and names the gap at start-up and in its receipt.

**ArWen's physics is judged by observations, not by MPAS.** It is a different
physics implementation, so matching MPAS was never its standard; its referee
is skill against MRMS, Stage-IV, ASOS and radiosondes, and its measured
differences from native MPAS are registered in
[`tree/docs/declared-divergences.md`](tree/docs/declared-divergences.md). A
bias that is wrong against observations is still wrong.

### To `gpuwm` (ArWen): a one-way dependency

- **The default physics** is ArWen's, reached through the engine's
  column-batch seam, so gpuwm-hex and `gpuwm` share one implementation of it.
  Sixteen engine files are pinned by SHA-256; `gpuwm-hex doctor` reports
  whether the installed engine still matches.
- **The MPAS physics ports** are this project's own (`tree/src/hexcore/mpasphys/`).
  They compile with device maths headers read from the installed `gpuwm` at
  run time.
- **The Rust field engines** (`rw_mpas_init`, `rw_mpas_static`,
  `rw_mpas_convert`, `rw_mpas_mesh`, `rw_mpas_lbc`, `rw_wrfbatch` and the
  others) are built in `gpuwm`'s Rust workspace and reach a user through its
  bridge bundle (`gpuwm fetch-bridges`). Two of them, `rw_mpas_geometry` and
  `rw_mpas_stream`, ship in the gpuwm-hex wheel instead, pinned by SHA-256,
  because no released gpuwm carries the versions this release needs. Mesh
  generation stays in `gpuwm`.
- `gpuwm` does not import `hexcore`, so the two can be separate repositories.

---

## Layout

```
LICENSE          Apache-2.0
NOTICE           every upstream, its terms and what derives from it
README.md        this page
.github/         the CPU-tier workflow and the publish workflow
tree/            the gpuwm-hex distribution
  src/hexcore/     the port (`import hexcore`)
    mpasphys/        the MPAS-A v8.4.2 physics driver and scheme ports
  tools/           native reference recipes, gate runner, comparators
  tests/           the three-tier battery
  verification/    gate decks, schemas, manifests and vertical specs
  evidence/        see EVIDENCE.md
  docs/            the manual and the door pages
  licenses/        every upstream licence text
```

Every measurement quoted in this repository names the receipt it came from.
The receipts themselves are not carried here; `tree/evidence/EVIDENCE.md` says
why and what an `evidence/...` reference means.

## Running the battery

Three tiers, split by what a machine has to own. Each names what it is missing
rather than failing somewhere unrecognisable. Tier 1 needs nothing but Python:

```sh
cd tree
PYTHONPATH=src python -m pytest -q -m "not gpu and not bigcard and not assets" \
    $(grep -v '^#' tools/battery/cpu_files.txt | grep -v '^$')
```

`tools/battery/README.md` covers all three. `GPUWM_HEX_NO_LOCAL_GPU=1` bans
device contact outright. The gate decks need the native references and a
card; `tree/verification/gates/README.md` says how to lay them out and run
them.

## Licence

Apache-2.0, the same licence `gpuwm` ships under, except where an upstream's
own terms govern a ported portion. MPAS-Atmosphere (LANS and UCAR) is
BSD-3-Clause, and its terms travel with every copy. The physics ports also
carry NCAR MMM-physics (BSD-3-Clause), WRF physics as MPAS distributes it (the
WRF public-domain notice), RRTMG (AER, BSD-3-Clause) and Noah-MP (UCAR's terms,
which are **not** Apache-2.0 and do not allow selling it or a work that
contains it). `NOTICE` names every upstream and its terms,
`tree/THIRD-PARTY-NOTICES.txt` lists the derived files, and `tree/licenses/`
holds the texts, including those of the Rust crates the wheel's compiled
helpers link; all required permissive texts ship in the wheel and sdist.

UCAR Noah-MP v5.0 and UGWP ports are withheld from this public export.
UCAR's Noah-MP terms forbid sale; UGWP remains on licence hold. Their
upstream pins and attribution remain in `NOTICE` sections 4 and 8. The public
wheel contains no Fortran runtime. An opt-in cannot restore an excluded port.
The default ArWen route uses its separate WRF-lineage physics.

This is **not** the version available from LANS and UCAR, and neither they nor
their contributors endorse it. Results from gpuwm-hex are not results from
MPAS-Atmosphere.
