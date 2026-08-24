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

**The status of the referee itself:** the obs-skill machinery exists and is
byte-reproducible ([`obs-referee.md`](obs-referee.md),
`tools/run_obs_referee.py`, manifests under `verification/manifests/`), the
production manifest carries the two divergence cases
(`gfs-20260812-divergence`, `era5-20240521-divergence`) plus two
deliberately-pending control cases — and **the comparison has not been run**.
Until it runs, every verdict below reads NOT MEASURED against observations.
That is the open half of #283.

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

**Referee.** **MRMS one-hour precipitation** — continuous bias, threshold
contingency statistics, fractions skill score, connected-object displacement,
paired case-block confidence intervals — with ASOS surface variables as the
secondary surface check. This is squarely inside the obs-referee's boundary.
Status: **NOT MEASURED** until the referee runs.

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
loading) and the extreme tail of MRMS one-hour precipitation. The
cloud-water/rain-water *partition itself* is outside MRMS/ASOS scope — the
obs-referee declares it UNRESOLVED pending a profile/process referee — but the
reflectivity and precipitation-extremum consequences are scoreable. Status:
**NOT MEASURED** until the referee runs.

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

The first real obs-referee run (`tools/run_obs_referee.py run` against
materialized MRMS/ASOS canonical bundles and all four manifest cases) turns
(2) and (3) from "differences from MPAS" into "skill against observations",
in either direction. A supplied profile referee does the same for (1). Any
GF-lane change re-measures all three — (3) explicitly only after (2).
