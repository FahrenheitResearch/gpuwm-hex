# 5. Initial conditions: `gpuwm-hex init`

The init door builds MPAS initial conditions without running the native
Fortran `init_atmosphere_model`. The numerics are the Rust engine
`rw_mpas_init`; the door finds the engine, refuses bad inputs by name before
the engine runs, and writes a provenance receipt beside the init file.
[`docs/init-door.md`](../init-door.md) is the door's own full reference.

## 5.1 The two vertical modes, and which one works today

The init engine does not invent the vertical grid — the arrays that define
where the model's levels sit (`zgrid, zz, fzm, fzp, dzu, rdzw, zb, zb3`)
and the smoothed terrain have to come from somewhere. The door has two
modes for that, selected by which flags you pass, mutually exclusive:

**Capsule mode (`--capsule` + `--reference`) — the working path.** The
vertical arrays are read out of a **native-minted init-class file** for
this exact mesh, and the capsule's `zgrid` is asserted bit-identical
against the reference before a single level is trusted. This is the mode
every init in this manual was built with, and the mode behind the door's
proof of record (92/92 carried fields bit-identical against a native
golden; the produced init started the port in a forecast smoke —
[`docs/init-door.md`](../init-door.md), *Proof of record*).

What the capsule means in practice: **the first init for any given mesh
requires one native `init_atmosphere_model` artifact for that mesh.** After
that, gpuwm-hex builds every subsequent init for the mesh from its own door
— new dates, new sources, no native tool again. This is the sharpest
prerequisite in the product; it is stated here rather than discovered as a
runtime error.

**Native-free mode (`--vertical-spec`) — shipped, and not yet working end
to end. Do not plan on it.** The door accepts a versioned JSON vertical
declaration (`gpuwm-hex.vertical-spec/v1`, examples under
`verification/vertical-specs/`), constructs the v8.4.1 vertical contract
itself, writes a durable vertical artifact — and then the engine refuses
every native-free mint, correctly and by name, because the constructed
artifact does not yet carry the 35 init-stream variable slots the engine
derives its output schema from. **As of this manual no native-free init has
ever been emitted end to end**
[`evidence/statics-330-unified-rebuild/RECEIPT.md`, "A third defect"]. The
admission design is documented in
[`docs/native-free-init-admission.md`](../native-free-init-admission.md);
treat that page as the design record of an in-progress lane, and this
paragraph as the current truth.

## 5.2 The invocation

```sh
gpuwm-hex init \
  --met     <WPS-intermediate, or a directory holding exactly one> \
  --grid    assets/x1.40962.grid.nc \
  --static  assets/x1.40962.static.nc \
  --capsule   assets/x1.40962-native-init.nc \
  --reference assets/x1.40962-native-init.nc \
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

On success: the init file, `<out>.provenance.json` (SHA-256 of every input,
the engine binary, the argv, the engine's own receipt, and the output), a
JSON summary on stdout, exit 0. The proving run built a 372 MB init for
`x1.40962` from a GFS intermediate in 2.7 s, and chapter 2's forecast ran
from it.

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
