# gpuwm-hex — open state

What is open, with the measurement that makes it open. Refreshed 2026-08-24,
at the 0.1.0 tree. Every number here was measured; anything that was not is
marked **NOT MEASURED** and says what would settle it.

Measurements are named by the receipt directory they were written to, under
`tree/evidence/`. Those receipt files are not carried in this repository —
`tree/evidence/EVIDENCE.md` says why and what to ask for — so treat an
`evidence/...` path here as the *identity* of a measurement rather than as a
file you can open.

Closed items are not listed. If a defect is not here, either it is fixed or it
was never found — this file is not a substitute for looking. Items this
refresh closed against the 2026-08-21 edition, with where the closure lives:
per-mesh `dt` (#300, `tools/mpas_mesh_binding.py` registry rows + Courant
admission), native-free init admission (#305, `docs/native-free-init-admission.md`,
with its declared oracle gap — §4 below), local time stepping (merged, opt-in,
measured in `README.md` and `evidence/local-timestep/`), the `rw_mpas_static`
host-memory defect (#306, fixed engine-side by the streaming builder), the
generator dislocation quality class (#304, measured mechanism + validate
floors, engine-side), and the missing pin tag (§7 below).

---

## 1. The obs-skill verification HAS RUN — #283's open half, as far as observations reach

Refreshed 2026-08-25 by the obs-referee first-run walk. Four full-physics
forecasts on `x4.163842`, the proving node (RTX 5090), scored against NCEP/EMC Stage-IV
hourly QPE, MRMS composite reflectivity and ASOS surface reports; every archive
byte decoded by a shipped Rust door, the unstructured-to-structured gather by
`rw_mpas_convert` onto its 22 km Lambert CONUS window. Evidence, chain and
every instrument by SHA-256: `tree/evidence/obs-referee-283/` (`RECEIPT.md`
first). Byte-reproducibility is measured, not asserted: the suite was run twice
against the same bundles and all seven output files match.

| case | role | cycle | steps |
| --- | --- | --- | --- |
| `gfs-20260812-divergence` | known divergence | 2026-08-12 06Z | 720/720 |
| `era5-20240521-divergence` | known divergence | 2024-05-21 12Z | **690/720, truncated** |
| `gfs-20250714-independent-control` | independent control | 2025-07-14 12Z | 720/720 |
| `gfs-20250114-weak-convection-control` | weak convection control | 2025-01-14 12Z | 720/720 |

The two controls the manifest had left `pending` are selected and pinned by a
mechanical screen of the Stage-IV archive rather than by eye
(`tree/verification/producers/select_control_cases.py`; rule and numbers in
`docs/obs-referee.md` and each case's `metadata.selection_basis`). The two
divergence cycles are corrected from 00Z to the 06Z and 12Z their initial
conditions carry. The referee also had no producer until this lane: the only
thing that had ever written a canonical bundle was the synthetic fixture, which
a production manifest is forbidden to accept, so the contract named a producer
nobody could run. `tree/verification/producers/` is that producer.

**Divergence 2 is measured, and it went the other way.** The registered claim
that current ArWen has a *negative* precipitation bias against observations is
**DISFAVORED**: the paired case-block interval is entirely positive, **+0.0247
mm/h [+0.0041, +0.0606]** over four cases. Against Stage-IV the port is wet in
every case (+9.8 %, +56.9 %, +2.4 %, +41.5 % of the observed mean), and its
frequency bias at 1 mm/h is above one in three of four. Being a third drier
than native MPAS in `rainc` does not make the port drier than the atmosphere;
that inference is now measured and it is wrong. Neighbourhood placement (FSS,
4 cells, 1 mm/h) is **0.428 / 0.801 / 0.532 / 0.860**.

**Divergence 3's declared referee RUNS now (2026-08-25,
`lane/history-refl-q2`).** The history stream publishes `refl10cm` (computed
inside the due step's own WSM6 call — WRF's `diagflag` arrangement, the point
where native MPAS-A computes it; engine seam pin moved to `6e333822e` on
`0d04db712`, since converged into the release line as `26daaab7e` — §8) and
`q2` (bitwise, `q2_products_allowed = "true"`), default-on,
at a measured **+1.13 %** history-byte cost over a full 24 h case.
`rw_mpas_convert` carries both (`REFL_10CM`, `Q2`, converter lane
`03c2491cc`), so reflectivity renders draw the model's field instead of the
renderer's hydrometeor fallback, and the model bundle scores
`reflectivity_dbz` (column max) and `dewpoint_k` (the engine's own formula on
Q2/PSFC). Re-run and re-scored on `gfs-20260812-divergence` (720/720): CSI
**0.0916**/20 dBZ and **0.0097**/40 dBZ on 643,419 pairs, object referee **86
model vs 54 MRMS 35 dBZ objects**, 8 matches at **110.9 km** median
displacement; `asos-dewpoint-rmse` **3.312 K** (bias −1.257 K, n = 56,667).
The other three cases' bundles predate the fields and re-run mechanically
(~46 min of one card each). Evidence:
`tree/evidence/history-refl-q2-20260825/`. The precipitation-extremum half of
divergence 3 was already measured and is not decisive at 22 km against a
4.8 km pixel; the numbers and the reason are in `declared-divergences.md`.

`asos-pressure-rmse` is the third unscorable metric and is not a port defect:
`rw_asos` reports MSLP, a sea-level reduction, and calling it station pressure
would fabricate the reduction.

**A guardrail measurement worth not rediscovering:** ASOS 10 m wind-speed RMSE
is 1.94-2.29 m/s with a one-signed **+0.46 to +0.64 m/s** high bias across all
four cases; 2 m temperature RMSE is 2.53-3.16 K, correlation 0.89-0.96. Not a
declared divergence, not claimed as one, recorded because it was measured.

**The scorecard's headline `scientific_verdict` reads NOT_MEASURED and that is
not a failed run.** It compares the opt-in `gf-subsidence-experiment` arm
against `arwen-current`; that arm was not run and the verdict stays
NOT_MEASURED until it is. The port's own skill is the per-case records and the
absolute claim, both measured: 28 of `arwen-current`'s 48 case/metric records.

**MEASURED 2026-08-25 (the refl/q2 history walk): the ERA5 divergence case does
not complete 24 h at the previous engine pin either.** Re-run of the same
registered init at `pin/mpas-port-arwen-seam` (`629ddb6f0`, pre-pin-move tree
the proving node): `truncated_by_model_refusal`, the same step 691 at
82,800 s, the same one-step `qv_max` doubling (0.0314 → 0.0678 kg/kg). The
step-691 refusal is a property of the case and configuration, not of the GF
pin move, and this case's "24 h" divergence numbers are 23 h numbers at both
pins. Receipt: `tree/evidence/history-refl-q2-20260825/RECEIPT.md`.

**NOT MEASURED, and named:**

- **forecast skill from a native-free mint.** All four cases were initialized by
  native `init_atmosphere`, so this run says nothing about that path (§4).

The theta drift (#1 there) still needs a profile referee MRMS/ASOS cannot
provide, and remains disqualifying for long-range work: +0.019 K/h above level
45, about +3.2 K extrapolated at 7 days. The run records it `UNRESOLVED` by
rule rather than leaving it blank. Choosing the profile reference is a ruling,
not an agent's call, and this lane did not make it.

## 2. The x4 restart-identity leg — #327 FIXED 2026-08-24, gate green

Root cause found, and the fix is in this tree: the checkpoint schema never carried
`CudaV841GfDynamicsTendencies` — GF's advective-forcing pair
`rthdynten`/`rqvdynten`, formed by each step's dynamics and consumed by the
NEXT step's `begin_step`. Driver-owned per-step state outside both the MPAS
atmosphere and the backend payload, so a restored run re-entered step 16 with
zero forcing lanes while the continuous run fed the real step-15 pair —
deterministic divergence across all 24 fingerprint paths, every arm. The
task-231 forcing-lane seam introduced the carrier after the 08-20 green tree,
which is why the gate was green then; this closes cap308's tree-vs-arwen-pin
suspect split as "both, jointly". Fix: schema v2→v3, the carrier is
downloaded with its own fingerprint and clock check, re-uploaded and re-seeded
on restore, and a pre-v3 checkpoint is refused by name. Gate on the proving
node (RTX 5090): before rc=1 in 592.5 s, after rc=0 in 619.6 s —
`restart_bitwise_identical: true`, the restarted history file byte-identical
(same sha256) to the uninterrupted one. Evidence:
`tree/evidence/restart-step16-327/`.

## 3. The registered generated static carries the antipodal drag band — #330

The unified `rw_mpas_static` oracle found that the **retired** writer computed
its ten GWD fields (`var2d, con, oa1..4, ol1..4`) from terrain at the antipode
of every cell — an archive-origin assumption, every value finite and
plausible. The registered `v15.150.38857` static (`a326fad3…`, registered
2026-08-21) was built by that writer and carries the band. The two
published-mesh statics are unaffected: both were registered from native-made
files, and the x1 registration (2026-08-14) predates any Rust writer
existing.

**This is done.** The statics-330 unified-writer lane merged and is an ancestor
of this tip; the rebuild landed at `420f323`. `v15.150.38857` was rebuilt on
the unified 82-variable `rw_mpas_static` and re-registered:
`tools/mpas_mesh_binding.py:142` now pins `static_sha256`
`199c16ca993edfca9335b9e63b63db0a67e0eb201179d3dd1df1f9510420635f` at
74,304,272 bytes, and every registry row now names its builder. The band is
measured, not asserted: `corr(old, new)` for `var2d` is **+0.003 at the same
cell and +0.697 at lon+180**, and `oa`/`ol` move full scale on two thirds of
cells. Against a native `init_atmosphere` static for the same mesh the
rebuilt file scores `var2d` **+0.9999**, `oa1` **+0.9961**, land-only `con`
**+0.9928**, and it adds the operator tables and soil-composition group the
retired writer omitted. Recorded under Fixed in `tree/CHANGELOG.md`.

## 4. The constructed vertical path — measured against the native oracle

Native-free init admission (#305) is merged, is the normal path, and now has
its native comparison (the proving node, RTX 5090, 2026-08-24,
`tree/evidence/native-free-proof-20260824/`): the x1.40962 native-free mint
is **schema-complete against the native golden — 134/134 variables, 0
missing, 0 dtype/dims mismatches** — the constructed vertical sits within
**3.9 mm of native zgrid** (35/42 computed met-state fields WITHIN the
same-zgrid r3 tolerances; the 7 EXCEEDS are the density/humidity families
amplifying that mm-scale difference, worst relhum 7.7 RH points), the mint
is same-session byte-deterministic, the minted init **ran the dycore**
(receipts in the evidence folder), and capsule mode stayed byte-identical
to the 2026-08-20 evidence output (modulo the self-describing
`gpuwm_provenance` exe-path attribute). The **generated-mesh forecast**
closed on 2026-08-24: `u96.64002` is ours end to end — `rw_mpas_mesh` grid,
`rw_mpas_static` static, native-free init, no native binary at any stage —
and it ran full physics on the proving node (RTX 5070 Ti) at rc 0, `status "passed"`,
`finite: true` at every step
(receipt `tree/evidence/genmesh-dual-edge-20260824/`; see
[`tree/evidence/EVIDENCE.md`](tree/evidence/EVIDENCE.md)).
Still open here: **forecast skill** (§1's obs referee).

## 5. Capacity — the converged stack is the of-record row on the 170 SM card

The measured model of record (2026-08-25, the proving node's RTX 5090 32,607 MiB,
170 SM, hex `7fe514b` + engine `26daaab7e` — the converged release-line
seam pin, #310 chunk sizing and the refl-q2 history stream all in — the
#264 instrument, both meshes in one session,
`tree/evidence/pin-move-335-20260825/node2/`):

```
converged 26daaab7e:  4,339.1 MiB + 103,696 B/cell   (170 SM part)
```

x1.40962 (40,962 cells) peaked at **8,390 MiB** — 3,898 MiB of headroom
inside a 12 GiB budget. x4.163842 (163,842 cells) peaked at
**20,542 MiB** — still a 32 GiB-card configuration under the unchanged
24 GiB `NATIVE_DEVICE_FLOOR`. Projected u96.64002 on this model:
10,668 MiB (NOT MEASURED on this card). The widest launched local frame
is now `wsm6_column` at 7,216 B — the release line's YSU frame work
reached the port through the converged engine, so `ysu_column` (9,232 B)
no longer tops the table; predicted local-memory reservation upper bound
1,797.0 MiB (was 2,298.98 MiB). Peak = this process's `nvidia-smi` row,
same convention as every prior row.

SUPERSEDED (2026-08-24, the proving node's RTX 5090, both meshes at both pins in one
session, `tree/evidence/gf-pin-move-measured-20260824/` — superseded
2026-08-25 by the converged-stack row above; kept as the pin-move-era
record):

```
old pin 629ddb6f0:  9,547.8 MiB + 93,286 B/cell   (x1 peak  9,948 MiB, x4 peak 20,902 MiB)
new pin 0d04db712:  6,296.5 MiB + 93,474 B/cell
```

Against that superseded row the converged stack moves the fixed term
DOWN 1,957.4 MiB (#310 sizes the radiation chunk to the card — on
170 SM only the LW chunk narrows — and the frame-cut wave shrinks the
reservation) and the slope UP 10,222 B/cell (the refl10cm/q2 history
publication and the release line's own seam-file evolution ride in the
per-cell term); net at the meshes measured: x1 −1,558 MiB, x4 −360 MiB.
`gf_gfdrv_stage` and the 4,990 MiB the pre-cut ledger attributed to it
remain gone from the launched set. This supersedes every earlier
footprint model (the 26.4 GiB x4 figure, the 5,018 + 140,916 fit, and
the pre-cut per-card table).

Open, in measurement order:

- **The merged tip is ledgered — on the 70 SM card** (2026-08-24, the proving
  node's RTX 5070 Ti, 16,303 MiB, the merged tip at engine `0d04db712`,
  `evidence/merged-tip-ledger-20260824/`): x1.40962 peaks **6,724.0 MiB**
  and u96.64002 peaks **9,344.0 MiB** — both inside 12 GiB on physical
  hardware. At both mesh sizes the peak instant carries an identical
  **2,832.8 MiB** of mesh-independent RRTMG/McICA chunk transients
  (SW 2048 / LW 4096); engine #310 sizes the chunk to the device instead and, measured in the
  same session, releases 1,120.0 MiB on x1 (peak 5,604.0, no longer
  radiation-set) and 680.0 MiB on u96 (peak 8,664.0) with byte-identical
  output (135/135 surfaces, 31/31 fingerprints, flip-validated) at zero
  wall cost. The 170 SM half of this bullet CLOSED 2026-08-25: the
  converged-stack row above IS the merged-and-converged tip on the
  170 SM part. The 70 SM numbers here predate the seam convergence and
  the refl-q2 slope; a 70 SM re-run at the converged stack is NOT
  MEASURED, and per-card fixed terms still never transfer.
- **The fixed term is per-card.** All numbers above are the 170 SM part. Both
  smaller parts previously measured carried smaller fixed terms in both
  components; per-card numbers are measured rows, never transfers.
- **The x4 admission floor predates the cut.** `NATIVE_DEVICE_FLOOR` holds
  24 GiB free against a measured 20,902 MiB peak. Re-deriving it moves frozen
  proof constants (`MIN_FREE_DEVICE_BYTES`), so it is a deliberate re-proof
  decision, not an edit; until then x4 remains in practice a 32 GiB-card
  configuration.

## 6. The mesh registry and the published-family floors

`tools/mpas_mesh_binding.py` holds three rows, each pinning grid+static by
byte count and SHA-256, each with its own declared `dt` under a versioned
Courant policy (125 m/s, 0.90) admitted against the file's real `dcEdge` —
never `nominalMinDc`, which measured 10.5–39.7 % high as a length across three
meshes:

| row | cells | dt | provenance |
| --- | --- | --- | --- |
| `x4.163842` | 163,842 | 120 s | native authority; **frozen no-op** — binding asserts constants, changes nothing, under the named floors `NATIVE_DEVICE_FLOOR` 24 GiB / `NATIVE_RESTART_FLOOR` 22 GiB |
| `x1.40962` | 40,962 | 120 s | published mesh, native-made static; admission floor scales with cells |
| `v15.150.38857` | 38,857 | 60 s | generated; **refused at bind** since 2026-08-24 — collapsed dual edges, see below; static also carries the §3 drag band pending #330 |
| `u96.64002` | 64,002 | 120 s | generated, Goldberg-seeded; the first generated mesh to complete a full-physics forecast |

The generator's own quality floors (engine-side, `rw-mpas mesh/validate.rs`)
are anchored to the published family's measured readings rather than chosen by
taste, and that anchoring is stated in the refusal text: `min_dv_over_dc`
0.02 sits between the published x4's own 0.0336 — the roughest reading in the
family, which the floor deliberately admits — and the measured dislocation
class at 0.0147 and below. The published x4 passes at 1.68× the floor; nothing
is waived for it silently.

**The same floor is now enforced at BIND, not only at emit** (2026-08-24,
`src/mpas_port/dual_edge_admission.py`). `v15.150.38857` was generated on
2026-08-21, before the generator grew those floors, and nothing downstream
re-checked it: it carries 61 edges under 0.02, worst 1.685e-04 — `dvEdge`
6.514 m at edge 19786 — and the TRiSK tangential terms divide by `dvEdge`, so
the potential-vorticity gradient there is amplified 5,935x. That is the whole
of its step-0 `FloatingPointError`: a per-launch probe puts every runaway
magnitude in the first outer step on that one edge, and the first non-finite
value is `exner` at cell 6461, an immediate neighbour of both its cells. No
timestep is a lever against it, so the bind refuses it by name rather than
letting the run die on a four-byte flag that names nothing. The frozen
`x4.163842` binds unchanged (fingerprint `2ff6c8957e6f0876`) on its own
measured 0.03365.

**What this leaves open engine-side:** Goldberg seeding fixes UNIFORM
requests, and a graded request still takes the density-biased Fibonacci seed
whose dislocation quads relax to near-cocircular. Variable-resolution
generation is therefore refused at both ends today; draining that tail needs
a mechanism Lloyd relaxation does not have.

## 7. Local time stepping — landed, measured, and the stronger form measured to a no-go

Landed with all four gates measured. Opt-in, default OFF. Gate 2 proved the default path untouched: switch-off is
bit-identical to the pinned configuration, and through the full-physics front
door every history frame matches byte for byte. Conservation is flux-matched
at binary32 rounding, measured (dry mass 2.6e-11, passive qv 6.4e-10 against
the 2.0e-8 bound).

**The option does not pay on the published x4.163842 mesh — 0.988x — and the
ceiling says it cannot**: the acoustic loop is 23.5% of a model step and the
released `(1,3,6)` schedule caps the whole-step gain at 1.16x even if every
column were coarse. It ships as a declared divergence with a measured cost,
not as a speedup. See the README's *Local time stepping, opt-in*.

**The full-step form is measured to a no-go too** (2026-08-24,
`tools/probe_lts_fullstep_projection.py`, RTX 5070 Ti,
`evidence/local-timestep/fullstep-projection.json`). Stepping every dycore
kernel per rate class has an arithmetic prize of 1.47x on x4.163842 and 2.73x
on the generated 15 km-in-136 km box mesh, but composing MEASURED per-class
launch costs of the port's own kernels over the real class index lists gives
**1.254x** on x4 (before interface bookkeeping, which the acoustic form
measured at ~7% of a step) and **0.979x — a projected slowdown — on the box
mesh**, whose 38,857 cells sit below the card's occupancy knee, so a
5,562-cell class launch costs nearly what the whole mesh costs. The uniform
mesh is the known-answer row: exactly 1.000x, index-list launches within 0.2%
of the pinned kernels. What could reopen it — meshes an order of magnitude
larger, or concurrent-stream class overlap — is named in the README and
NOT MEASURED.

## 8. Repository and packaging

**`pip install gpuwm-hex` resolves.** The declared floor is `gpuwm>=2.5.5`:
2.5.3 was the first `gpuwm` that stages the four MPAS bridge binaries both
front doors drive, and 2.5.6 is the first whose published bytes match the
sixteen-file seam manifest this port pins (`tree/pyproject.toml` states both
reasons). Measured 2026-08-24: PyPI serves `gpuwm` 2.5.4, 2.5.3, 2.5.2 …
and `gpuwm-data` 2.5.4, 2.5.3, 2.5.2, 2.5.1, 2.5.0, and `gpuwm-hex` 0.1.0 is
itself published. A fresh-venv install of `gpuwm-hex==0.1.0` against the
2.5.4 engine was proven end to end the same day. The floor is a floor rather
than a hold, and nothing in this section is a reason to install from a
checkout.

**The forecast front door exists** (merged 2026-08-24):
`gpuwm-hex forecast` binds a registered mesh against its pinned bytes,
admits against the measured footprint row on the card it is actually on
(`--preflight` answers "will this run?" without integrating), runs the one
integration loop this project has, and hands the render command forward. 17
named refusals, no tracebacks at the door; 45 CPU-testable tests. The
pyproject no longer calls it a door on a room with no floor. Proven against
the artifact on the RTX 3080 (2026-08-24,
`evidence gallery hex-forecast-door-20260824`): the door bound x1.40962,
measured the card, and correctly REFUSED — predicted 9,948.0 MiB against
9,097.0 MiB free of 10,239.5 — which is itself the capacity finding: no
registered mesh fits 10 GiB until the #310 rung lands. Two readings from
that proof carry weight: the fitted footprint row predicts x4's
independently measured peak exactly (20,902 MiB, a point it was not fitted
to), and on Windows/WDDM CuPy and `nvidia-smi` disagree about free memory
by 2,319 MiB in the same process at the same moment — the door reads the
CUDA driver deliberately (that is the allocator the run uses), so Windows
admission is optimistic against `nvidia-smi`; stated in manual 6.2. An
earlier caveat is retired: re-walked 2026-08-24 on the same 3080
(`evidence/small-card-3080-20260824/`) — the native-free mint SUCCEEDS on
the desktop box (`native_runtime_dependency: false`, rc 0, 377.5 s first
mint), so "no init can be minted on that box" is dead. The same walk
found the deeper verdict: with a measured row that admits, the run stops
at `cuda.compute_capability=8.6 is below required 12.0` — the sm_120
execution pin, not VRAM, is what closes a 10 GiB Ampere card, the 3080's
own footprint row is NOT MEASURABLE (the #264 instrument funnels through
the same pin), and preflight, being deliberately CUDA-free, answers
`preflight_passed` on a card the run then refuses. The walk also found
and fixed the door's success leg (it pre-created `--out`/`--scratch`,
which the driver requires absent — every admitted door run on every card
died on `FileExistsError` until the fix landed).

**The four unmerged lanes are adjudicated.** Each was batteried and read;
two landed, two are held with the reason recorded rather than the branch
quietly rotting.

*Merged.*

- The allocation-ledger lane (#264). Documentation, evidence and
  probes; nothing under `src/`. **Its numbers are PRE-GF-frame-cut** — the
  lane measured 2026-08-20 at pin `629ddb6f0`, and section 5 above is the
  account of record. The merged doc carries a supersession header saying so.
  It is kept because it is the BEFORE arm of the pin move's own delta: the
  3,251.3 MiB the cut released is a subtraction against its fixed term. Its
  headline — the local-memory backing store at 7,034.0 MiB, sized by
  `gf_gfdrv_stage`'s 29,264 B frame — is the saving the pin move went and
  TOOK; it is not an opportunity still open, and it must not be quoted as
  one. What survives as current is the method, not the totals.
- The physics-tier-park lane (#261), merged for the
  INSTRUMENTS, not the mode. Parking the physics tier is a measured negative:
  it moves 786.8 MiB and releases 735.4 MiB, and the allocator reservation —
  the number a card actually has to provide — got WORSE, 4055.3 → 4068.7 MiB
  on x1.40962, while every arm's history stayed byte-identical. **These are
  also pre-cut absolutes**, measured 2026-08-20 at pin `629ddb6f0`: the
  lane's unmodified arm reserves 5414.0 MiB, which is exactly the old-x1 pool
  row in the pin move's own verdict table, so the arm is positively
  identified as the old pin. The A/B SIGN is what carries — both arms ran the
  same stack — and the sign is what the verdict rests on. The absolute
  numbers do not transfer to the current pin (`26daaab7e` since
  2026-08-25) and re-running the park there is NOT MEASURED. The park
  lands inert (`physics_park=None`, `park_physics_tier=False`, and
  `--park-physics-tier` exists only on a measurement harness, never on the
  one console script), and the flag's help text carries the regression so
  the reason not to use it travels with the switch. What is worth taking is
  upstream: the three RRTMG chunk constants cost the same at 40,962 cells as
  at 163,842.

*Held, not merged — the reason is the same in both cases and it is a real
finding, not a scheduling excuse.*

- The diagnostics-output lane — **held by its own measurement.**
  Dropping the construction-time diagnostic's device copies is worth 984
  B/cell and the outputs stay byte-identical, but the footprint did not
  follow: pool high-water went 17376.3 → 16268.1 MiB on x4.163842 and
  **5404.5 → 7111.7 MiB on x1.40962**, a rise that reproduced exactly across
  two runs, one on an idle card. Freeing 38.4 MiB early moves the arena
  enough that the fixed 1,745.6 MiB RRTMG-SW chunk workspace stops being
  servable from the free list at the step where radiation and history capture
  coincide. The lane's own final commit says it must not ship until the arena
  is owned, and the hold stands. **But the stated stakes were pre-cut and
  must not be repeated as they were.** Those arms are 2026-08-20 at pin
  `629ddb6f0`. At pin `0d04db712` (2026-08-24) the x1.40962 pool high-water
  is **7,159.6 MiB on its own**, with no lane applied — within 48 MiB of the
  7,111.7 MiB this lane was held for producing — and x1.40962 peaked
  9,948 MiB there; at the converged pin `26daaab7e` (2026-08-25) it peaks
  **8,390 MiB** and fits a 12 GiB budget with room. So "this lane threatens
  the 12 GiB claim" is NOT a supportable statement at the current pin, and
  whether its +1,707.2 MiB survives the frame cut at all is
  **NOT MEASURED**. The hold is now for want of a measurement at the
  current pin, not for a known regression against it.
- The workspace-arenas lane — **held for want of a run.** The three
  allocation removals and the RK1/RK2 aliasing are individually well argued
  (every consumer of the tangential velocity takes it as `const float *`;
  `rw` and `rw_save` are both `const` in `cuda_acoustic_v841.py:278`, and at
  stage 1 their difference was already exactly zero), but the lane re-freezes
  three execution pins — `cuda_driver.py`, `cuda_horizontal.py`,
  `cuda_horizontal_v841.py` — and those pins exist precisely to force every
  affected proof to re-run against the new digest. No proof was re-run. The
  lane carries no receipt, and the tier that would catch a regression needs
  the byte-pinned authority files and a 32 GiB card.

**What the held pair jointly proves is worth more than either lane.** Both
lanes reduce live device bytes; the diagnostics lane MEASURED that doing so
moved the pool high-water the wrong way by 1,707.2 MiB (at the old pin, on
x1.40962). That is the arenas lane's success criterion, and the arenas lane
never measured it. Live bytes are not the footprint. Neither lane should be
judged on allocation counts again — the reservation is the number, and that
conclusion is about METHOD, so the frame cut does not touch it.

Both branches are kept. What unblocks them is one x1.40962 run per arm **at
the current pin (`26daaab7e`, converged 2026-08-25)** on a node holding the
authority files,
comparing pool high-water and history digests, not more reading. Re-using
either lane's recorded absolutes instead of re-running would compare across
the frame cut and produce a confident wrong answer.

**The pin lineage is CONVERGED with the release line (2026-08-25).** `gpuwm` carries annotated tags
`pin/mpas-port-arwen-seam` (`629ddb6f0`), `pin/mpas-port-arwen-seam-v2`
(`0d04db712`, placed 2026-08-23), `pin/mpas-port-arwen-seam-v3`
(`6e333822e`, the refl seam) and `pin/mpas-port-arwen-seam-v4`
(`26daaab7e`) — and v4 is different in kind: it is the seam-converge
MERGE of the pin lineage into the engine release line (at
`613b681d3`), not a commit on a side line. `ARWEN_BUILD_COMMIT` points
there, so the next public engine snapshot cut from the release line
satisfies the port's guards as cut. The reachability hazard the 08-21
edition named is closed.

## 9. The two pin legs that need hardware

Unchanged, stated so a green local battery is not mistaken for a verified pin.

**The native authority — NOT VERIFIED HERE.** 9 entries, ~6.9 GiB, masked
SHA-256 (the random netCDF `file_id` NUL'd), held in triplicate on the proving
hardware, no fetch path, none in this repository.

**The compiled-endpoint fixture — NOT PRESENT.**
`tools/compare_v841_compiled_endpoint.py` defaults to
`tree/oracle/jw-x1.2562-v8.4.1-split3-endpoint-nonclaim`, six files pinned by
SHA-256; the directory is not in this checkout, so the comparator needs
`--fixture` pointed at the bundle wherever it lives.

What **is** verified locally (2026-08-25): the sixteen
`ARWEN_SOURCE_MANIFEST` files hash-match **16/16** at `gpuwm` commit
`26daaab7e` — which now sits ON the release lineage (the seam-converge
merge into the engine release line at `613b681d3`), so the old
"10/16 at the release line" caveat is retired: the release
line and the pin line name the same bytes. Proven by real bind
(`pin_arwen_physics_v841`) and by `verify_arwen_checkout_git` (head
`26daaab7e`, `clean: true`, `dirty_paths: []`);
`tree/evidence/pin-move-335-20260825/`. The anchor remains an anchor:
14/16 at `629ddb6f0` and 9/16 at the earlier port-landing ref still discriminate.
See `README.md`.
