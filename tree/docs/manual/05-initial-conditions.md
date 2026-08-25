# 5. Initial conditions: `gpuwm-hex init`

The init door builds MPAS initial conditions without running the native
Fortran `init_atmosphere_model`. The numerics are the Rust engine
`rw_mpas_init`; the door finds the engine, refuses bad inputs by name before
the engine runs, and writes a provenance receipt beside the init file.
[`docs/init-door.md`](../init-door.md) is the door's own full reference.

## 5.1 The two vertical modes

The init engine does not invent the vertical grid — the arrays that define
where the model's levels sit (`zgrid, zz, fzm, fzp, dzu, rdzw, zb, zb3`)
and the smoothed terrain have to come from somewhere. The door has two
modes for that, selected by which flags you pass, mutually exclusive:

**Native-free mode (`--vertical-spec`) — the normal path.** The door
accepts a versioned JSON vertical declaration
(`gpuwm-hex.vertical-spec/v1`, examples under
`verification/vertical-specs/`), constructs the v8.4.1 vertical contract
itself, and writes a durable vertical artifact the engine consumes. No
native artifact appears anywhere in the mesh's lineage at runtime; the
provenance receipt records `native_runtime_dependency: false`. Proven end
to end on x1.40962 against the native golden: the mint is
schema-complete (all 134 native variables, 0 missing), 75 carried fields
bit-identical, the constructed vertical within millimetres of the native
one (`zgrid` max 3.9 mm over a 30 km column), and the produced init
starts the dycore
[`evidence/native-free-proof-20260824/RECEIPT.md`]. The constructed
vertical is not the native vertical byte for byte; the quantified deltas
live in that receipt.

**Capsule mode (`--capsule` + `--reference`) — the compatibility mode,
and the byte-anchored one.** The vertical arrays are read out of a
**native-minted init-class file** for this exact mesh, and the capsule's
`zgrid` is asserted bit-identical against the reference before a single
level is trusted. This is the mode behind the door's bit-parity proof of
record (92/92 carried fields bit-identical against a native golden —
[`docs/init-door.md`](../init-door.md), *Proof of record*), and the mode
to choose when the run must reproduce a native vertical exactly. It is
the only mode that needs a native `init_atmosphere_model` artifact.

The admission design behind native-free mode is documented in
[`docs/native-free-init-admission.md`](../native-free-init-admission.md).

## 5.2 The invocation

```sh
gpuwm-hex init \
  --met     <WPS-intermediate, or a directory holding exactly one> \
  --grid    assets/x1.40962.grid.nc \
  --static  assets/x1.40962.static.nc \
  --vertical-spec verification/vertical-specs/tc55-v1.json \
  --out     work/x1.40962.init.nc \
  --start-time 2026-08-12_06:00:00 \
  --nfglevels 34 --nfgsoillevels 4 \
  --extrap-airtemp lapse-rate --use-spechumd no \
  --theta-adv-order 3 --coef-3rd-order 0.25 \
  --virtual-factor reproduce-fortran \
  --deep-soil-moisture reproduce-fortran \
  --landuse-table MODIFIED_IGBP_MODIS_NOAH \
  --frac-seaice yes --tsk-seaice-threshold 100.0 \
  --oned-underflow preserve
```

Capsule mode replaces the `--vertical-spec` line with `--capsule` and
`--reference`, both naming the native-minted init-class file; everything
else is identical.

On success: the init file, `<out>.provenance.json` (SHA-256 of every input,
the engine binary, the argv, the engine's own receipt, and the output), a
JSON summary on stdout, exit 0. The capsule proving run built a 372 MB
init for `x1.40962` from a GFS intermediate with 2.7 s in the engine, and
chapter 2's forecast ran from it; the native-free mint of the same mesh
and met spends about 2 s in the engine, minutes in the door's first-mint
geometry solve, and about a minute on a re-mint from the keyed cache.

## 5.3 Every switch is explicit, and why

**Every physics switch is required and has no default.** Each one changes
the numbers in a file that opens cleanly and reads plausibly either way, so
a defaulted switch would be a silent wrong answer. Each refusal prints the
native namelist key, so a captured `namelist.init_atmosphere` transcribes
without guessing:

| flag | native key / meaning |
| --- | --- |
| `--start-time` | `config_start_time` |
| `--nfglevels` | `config_nfglevels` — must cover the met file's distinct first-guess levels; the refusal names the real count |
| `--nfgsoillevels` | `config_nfgsoillevels` |
| `--extrap-airtemp` | `config_extrap_airtemp` (`constant`/`linear`/`lapse-rate`) |
| `--use-spechumd` | `config_use_spechumd` |
| `--theta-adv-order` | `config_theta_adv_order` (must agree with the vertical) |
| `--coef-3rd-order` | `config_coef_3rd_order` |
| `--landuse-table` | `config_landuse_data` |
| `--frac-seaice` | `config_frac_seaice` |
| `--tsk-seaice-threshold` | `config_tsk_seaice_threshold`, K |
| `--virtual-factor` | reference-build property: `reproduce-fortran` or `consistent` |
| `--deep-soil-moisture` | reference-build property: `reproduce-fortran` or `corrected` |
| `--oned-underflow` | reference-build property: `preserve`, or `reproduce-ifx-ftz` to match an `-O3` ifx reference where the modes diverge |

The last three are properties of the reference *build*, not namelist keys;
`reproduce-fortran`/`reproduce-fortran`/`preserve` is the defined-behavior
set.

## 5.4 What the door refuses

Sixteen named refusals fire before the engine runs — a physics switch not
given, a missing input, a directory with zero or several intermediates, a
truncated intermediate, `LANDSEA` absent, no surface-tagged level, more
first-guess levels than declared, no soil layers, a start time not in the
met file, a static missing engine-read variables, a grid/static `nCells`
mismatch, a capsule for another mesh, a reference whose `zgrid` differs.
The full list with exact wording is in
[`docs/init-door.md`](../init-door.md); chapter 8 walks the ones you will
actually meet. Engine-side refusals (exit 1) pass through verbatim, with
one added line stating that the engine refused and nothing was written.

## 5.5 Portability of the produced init

The produced init is drop-in only for a run pinning the same grid/static
bytes the capsule was built from; the forecast driver refuses otherwise,
reporting changed reconstruction-coefficient bytes. That message means
provenance mismatch, not corruption.
