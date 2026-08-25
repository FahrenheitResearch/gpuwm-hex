# Declared divergences

gpuwm-hex runs ArWen's physics, not MPAS's. Whole-model agreement with native
MPAS-Atmosphere therefore stopped being reachable the moment that choice was
made, and on 2026-08-20 physics parity was retired as a goal: **the dynamical
core stays pinned to native v8.4.1 byte-identity — that pin is the correctness
anchor — and the physics is judged by skill against observations, MRMS and
ASOS, not by agreement with another model.**

This document is the register of the known physics-shaped differences against
native. Each one is **declared**: measured, mechanism named, referee named. A
declared divergence is not a blocker and it is not a licence to ignore a
defect — a bias that is wrong against *observations* is still wrong. Retiring
MPAS as the referee changes the referee; it does not clear a finding.

**The status of the referee itself: it has run.** The first real run is
2026-08-25 on the proving node (RTX 5090) — four full-physics forecasts on the
163,842-cell mesh, scored against NCEP/EMC Stage-IV hourly QPE, MRMS composite
reflectivity and ASOS surface reports, every archive byte decoded by the
shipped Rust doors. Three ran the full 24 h; the ERA5 case refused its own step
691 and truncated at 23 h, which is recorded below and in the receipt rather
than smoothed over.
The scorecard, the per-case metric records and the run receipt are in
receipt `../evidence/obs-referee-283/` (see
[`../evidence/EVIDENCE.md`](../evidence/EVIDENCE.md)), and that
directory's `RECEIPT.md` carries the chain and the SHA-256 of every instrument
in it. Two of the four cases are the divergence cases themselves; the two
controls the manifest had left `pending` are now selected against a measured
screen of the observation archive and pinned, with the screen's rule and its
numbers in [`obs-referee.md`](obs-referee.md).

One substitution, recorded here because it changes which instrument holds the
whistle. The declared precipitation referee is MRMS one-hour QPE; the shipped
MRMS door decodes composite reflectivity only and has no QPE quantity, so
precipitation is scored against Stage-IV, NCEP's hourly multi-sensor analysis,
through its own shipped door. Reflectivity is still MRMS.

What each divergence's referee now says is in its own section below. The open
half of #283 is closed as far as MRMS/ASOS can close it, and where they cannot
reach, the reason is named rather than left blank.

## The measurement behind all three

24 h forecasts, two independent weather cases, two mixing regimes,
163,842-cell mesh, cell-aligned against native output with no interpolation.
What makes these three *divergences* rather than chaos is case independence:
chaos between a single-precision CPU model and a CUDA port is symmetric and
case-dependent, and everything else in the comparison is exactly that
(envelope statistics match; peak updraft 11.49 vs 11.26 m/s; domain means
within 0.05 %). These three are one-signed and identical across both cases and
both mixing regimes, which chaos cannot be.

---

## 1. Upper-band potential-temperature drift

**Magnitude.** Above level 45 the port warms relative to native at
**+0.019 K/h, near-linear and one-signed, +0.46 K at 24 h**, identical in
both weather cases and both mixing regimes. Extrapolated — arithmetic on the
measured rate, not a measurement — a 7-day run carries about **+3.2 K** of
stratospheric error, which disqualifies this version for long-range work.

**Mechanism.** Case independence means a code path, not weather. The prime
suspect is the legacy-RRTMG radiation lane (carried for parity with native,
which was itself a parity cost; RRTMGP is free to become the production arm).
Not localized further.

**Referee.** MRMS and ASOS **cannot see this one** — surface stations and
radar-derived precipitation do not verify model theta above level 45, and the
obs-referee's scientific boundary says so explicitly. The referee of record is
a vertical-profile referee: radiosondes or a provenance-pinned analysis
profile. Until one is supplied, this divergence is **UNRESOLVED by
observations** — declared, bounded at 24 h, and disqualifying for long-range
use as stated.

The 2026-08-25 run did not change that and was never going to. It records the
claim `upper-band-theta-drift-cause` as `UNRESOLVED` under the manifest's
`outside_primary_scope` rule, with the reason carried in the scorecard: MRMS
and ASOS do not observe model theta above level 45. That is the boundary
holding, not a gap in the run — the run has no authority here and says so.
Choosing the profile reference is not an agent's call, and this lane did not
make it.

## 2. Convective-to-explicit precipitation repartition

**Magnitude.** The port's Grell-Freitas produces about a third less
convective rain (`rainc` **−36 % / −34 %** in the two cases); explicit
microphysics makes up roughly half of it (`rainnc` **+29 % / +25 %**); net
domain-mean precipitation runs about **15 % dry**.

**Mechanism.** The declared GF generation gap, verified by source count on
both sides: the port carries WRF v4.6.1's **Freitas-2018** GF body; native
v8.4.1's `module_cu_gf.mpas.F` is the **2013 ensemble fork**. Native has zero
occurrences of `dicycle` and zero of `tau_ecmwf`; it carries Fritsch-Chappell
`AA0/1200s` closure members the port does not; it runs `c0=.002` against the
port's temperature-scaled `c0=.004`; its shallow scheme does not precipitate
while the port's folds `prets` into `pratec`. The seam is **not** the gap:
task #231 closed seam-level non-parity (the four auxiliary forcing lanes,
shallow-on, per-cell `dx` all reach the scheme the way native feeds them).
Closing the generation gap means porting native's `cup_gf`/`cup_gf_sh`
bodies — a program, not a seam edit — and whether it *should* close is
exactly what the referee decides: 2018-generation GF against observations may
be better, worse, or a wash versus the 2013 fork.

**Referee.** Hourly precipitation — continuous bias, threshold contingency
statistics, fractions skill score, connected-object displacement, paired
case-block confidence intervals — with ASOS surface variables as the secondary
surface check. This is squarely inside the obs-referee's boundary. Declared as
MRMS; run against Stage-IV, for the reason above.

**Status: MEASURED — and it went the other way.**

The registered claim `current-arwen-convective-rainfall-deficit` says current
ArWen has a *negative* one-hour precipitation bias against observations. The
referee returns **DISFAVORED**: the paired case-block interval is entirely
positive. Estimate **+0.0247 mm/h**, 95 % interval **[+0.0041, +0.0606]**, four
cases, 5,000 replicates. Against Stage-IV the port is **wet**, in every case:

| case | model mean | Stage-IV mean | bias | | paired samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gfs-20260812-divergence` | 0.1260 mm/h | 0.1148 mm/h | **+0.0112** | +9.8 % | 468,864 |
| `era5-20240521-divergence` | 0.2185 mm/h | 0.1392 mm/h | **+0.0793** | +56.9 % | 449,328 |
| `gfs-20250714-independent-control` | 0.1549 mm/h | 0.1512 mm/h | **+0.0037** | +2.4 % | 468,864 |
| `gfs-20250114-weak-convection-control` | 0.0156 mm/h | 0.0110 mm/h | **+0.0046** | +41.5 % | 468,864 |

**What that does and does not overturn.** It does not touch the magnitude
above: `rainc` really is about a third below native's and the generation gap is
real, verified by source count. What it overturns is the inference that was
sitting on top of it — that being drier than native means being drier than the
atmosphere. It does not. On these four cases, over the 22 km CONUS window, the
port rains **more** than the gauge-and-radar analysis, not less, and the
frequency bias at 1 mm/h is above one in three of the four cases (1.59, 1.35,
1.38, 0.77): it rains over too much area, not too little. This does not convert
arithmetically into a statement about native's own bias — the divergence
measured a global domain mean against native and this measures a CONUS window
against observations, which are neither the same domain nor the same
statistic — but the direction of the correction the port needs is now a
measurement rather than an assumption, and it is the opposite of the one the
deficit implied.

**Placement.** Fractions skill score at a 4-cell (about 88 km) neighbourhood,
1 mm/h: **0.428 / 0.801 / 0.532 / 0.860** across the four cases in the order
above. Point-for-point CSI at 1 mm/h is **0.083 / 0.304 / 0.134 / 0.250**, and
that gap between the two is mostly scale, not skill: the alignment compares a
22 km model box against a single 4.8 km Stage-IV pixel, and a box mean is
smoother than a point. The neighbourhood score is the one to read for
placement.

**Surface guardrails, measured on the same runs.** ASOS 2 m temperature RMSE
**3.16 / 2.53 / 2.72 / 3.11 K** (correlation 0.89 to 0.96, bias between −0.69
and +0.62 K) and 10 m wind-speed RMSE **2.03 / 2.29 / 1.94 / 2.05 m/s**, over
52,000 to 57,000 paired station reports per case. The wind bias is one-signed
across all four cases at **+0.46 to +0.64 m/s** — the port is windier at 10 m
than the stations are. That is not one of the three declared divergences and it
is not claimed as one here; it is what the guardrail measured, recorded so it
is not discovered twice.

## 3. Downstream condensate surplus

**Magnitude.** With more rain made explicitly, the port carries **+50 % cloud
water and +62 % rain water** in the domain mean by 24 h, with much heavier
point extrema (max-cell 24 h precipitation 502 vs 308 mm).

**Mechanism.** Downstream consequence of (2): rain the cumulus scheme does not
remove is condensed and precipitated explicitly, and the condensate load rides
along. It should be re-measured after any GF change **before** anyone touches
microphysics — treating it as a microphysics defect while (2) stands would tune
the wrong scheme.

**Referee.** **MRMS reflectivity** (the observable consequence of hydrometeor
loading) and the extreme tail of hourly precipitation. The cloud-water /
rain-water *partition itself* is outside MRMS/ASOS scope — the obs-referee
declares it UNRESOLVED pending a profile/process referee, and the 2026-08-25
run records exactly that for the claim
`condensate-partition-surplus-cause`.

**Status: the reflectivity half RUNS, and has first numbers (2026-08-25).**
The gap the first run named — the history stream carried no `refl10cm` — is
closed default-on: the due step's own WSM6 call computes the field (WRF's
`diagflag` arrangement, the point where native MPAS-A computes it), every
history frame publishes it, `rw_mpas_convert` carries it as `REFL_10CM`, and
the model bundle scores its column maximum against MRMS composite
reflectivity. Re-scored on `gfs-20260812-divergence` (re-run 2026-08-25,
720/720, hex `39fa25e`, engine `6e333822e`): CSI **0.0916** at 20 dBZ,
**0.0097** at 40 dBZ (n = 643,419 pairs), and the object referee — the read
that bears on THIS divergence — counted **86 model 35 dBZ objects against 54
observed** with 8 matches at a median **110.9 km** displacement. An object
surplus points the same direction as the declared condensate surplus, on one
case; the point-CSI values pay the documented 22 km-box-vs-0.01° resolution
cost and should not be read alone. The other three cases' bundles predate the
field and re-run mechanically. Evidence:
`tree/evidence/history-refl-q2-20260825/`.

**Status: the precipitation-extremum half is measured and is not decisive at
this resolution.** The port's 99.9th-percentile hourly rate against Stage-IV's,
per case: **6.09 vs 15.0**, **20.1 vs 13.75**, **8.56 vs 17.38**, **1.80 vs
2.00 mm/h**; frequency bias at 10 mm/h **0.089 / 1.917 / 0.246 / 0.860**. The
port's tail is *weaker* than observed in the two GFS convective cases and
*heavier* in the ERA5 case, and the two GFS cases are where the 22 km box
against a 4.8 km pixel bites hardest — a box mean reaches 10 mm/h far less
often than a point does, which shows up as a probability of detection of 0.001
and 0.011 there. So the "much heavier point extrema" consequence is neither
confirmed nor refuted at this scale, and saying it was would be reading the
grid rather than the model. What would settle it is the reflectivity half, or a
neighbourhood-maximum statistic the metric vocabulary does not currently carry.

One extremum in the ERA5 case is not weather: a single cell reaches **530.7
mm/h** in the last hour before that run refused its own step 691. It is the
runaway column that caused the refusal. It contributes about 1.5 % of that
case's precipitation bias, so it does not explain the wet result above, but it
should not be read as a forecast either.

---

## What the receipts carry

Every full-physics run receipt and every history file carries:

- `gf_native_parity_claim: false` — the fact: no parity with native GF is
  claimed;
- `gf_declared_divergence` — the declaration naming the generation gap and
  pointing here.

Receipts stopped carrying this as a `*_blocker` when parity was retired as a
goal (ruling of 2026-08-20): a blocker names work that must happen before
shipping, and this is not that — it is a property of the product, declared so
a user deciding whether to trust a number has it before the run, not after.

## What would move this document

The first real obs-referee run happened on 2026-08-25 and moved (2) outright:
it is now a skill statement against observations, and it points the opposite
way from the deficit. Three things would move the rest.

1. **`refl10cm` and `q2` in the history stream — DONE, 2026-08-25.** Both
   ship default-on in every history frame (+1.13 % history bytes, measured
   over a full 24 h case), and the four metrics they unblock are scored on
   the re-run divergence case: the three MRMS reflectivity numbers above and
   `asos-dewpoint-rmse` at **3.312 K RMSE, −1.257 K bias** over 56,667
   station reports. What remains of this item is mechanical: re-run the
   other three cases so their bundles carry the fields too.
2. **A profile referee for (1)** — radiosondes or a provenance-pinned analysis
   profile. Choosing the reference is not an automated decision and this lane
   did not make it.
3. **Any GF-lane change re-measures all three**, (3) explicitly only after (2).
   The re-measurement now has a standing target rather than an empty column:
   a GF change that removes the convective rainfall deficit against native
   would, on this evidence, make the port *wetter* against observations than it
   already is, so "closing the generation gap" and "improving obs skill" are
   not the same act and should not be assumed to be.

The measurement the first run owed is made (2026-08-25): the ERA5 divergence
case does **not** complete 24 h at the previous engine pin either. Re-run at
`pin/mpas-port-arwen-seam` (`629ddb6f0`, pre-pin-move tree `ca2a86d`) it
refused the same step 691 at 82,800 s with the same one-step `qv_max`
doubling (0.0314 → 0.0678 kg/kg). The truncation is a property of the case
and configuration, not of the GF pin move, and the 24 h divergence numbers
for this case remain 23 h numbers at both pins.
`tree/evidence/history-refl-q2-20260825/RECEIPT.md` carries the receipt.
