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

## 1. The obs-skill verification has not run — the open half of #283

The three physics divergences are **declared** — mechanism, magnitude and
referee registered in [`tree/docs/declared-divergences.md`](tree/docs/declared-divergences.md),
run receipts and history files carry `gf_native_parity_claim: false` beside
`gf_declared_divergence` (no receipt calls them a blocker anymore) — and the
comparison that would judge them has **NOT been run**. The referee machinery
is merged and byte-reproducible (`tree/tools/run_obs_referee.py`); the
production manifest carries the two divergence cases and two
deliberately-pending controls, and a pending case makes the scorecard
NOT_MEASURED. Retiring MPAS as the referee changed the referee; it did not
clear the findings. The theta drift (#1 there) additionally needs a profile
referee MRMS/ASOS cannot provide, and remains disqualifying for long-range
work: +0.019 K/h above level 45, about +3.2 K extrapolated at 7 days.

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
existing. Open act: rebuild `v15.150.38857`'s static with the unified writer
and re-register (new bytes + SHA-256 in `tools/mpas_mesh_binding.py`); until
then the registry row's note names the band. Runs in a parallel lane.

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
`gpuwm_provenance` exe-path attribute). Still open here: **a
generated-mesh (v15-class) forecast** — blocked by the pre-existing v15
step-0 FloatingPointError, which fires with native statics too — and
**forecast skill** (§1's obs referee).

## 5. Capacity — measured at the pin lane, unledgered at the merged tip

The measured model of record (2026-08-24, the proving node's RTX 5090, both meshes at both
pins in one session, `tree/evidence/gf-pin-move-measured-20260824/`):

```
old pin 629ddb6f0:  9,547.8 MiB + 93,286 B/cell
new pin 0d04db712:  6,296.5 MiB + 93,474 B/cell
```

x1.40962 (40,962 cells) peaked at **9,948 MiB** — a 12 GiB budget, measured.
x4.163842 (163,842 cells) peaked at **20,902 MiB**. The widest launched local
frame is now `ysu_column` at 9,232 B; `gf_gfdrv_stage` and the 4,990 MiB the
old ledger attributed to it are gone from the launched set. This supersedes
every earlier footprint model (the 26.4 GiB x4 figure, the 5,018 + 140,916
fit, and the pre-cut per-card table).

Open, in measurement order:

- **This tree is unledgered.** The arms ran a tree that predates the #308
  copy-elision merge. The 0.1.0 tree carries both changes; elision projects
  x1.40962 to ~9,587 MiB (**NOT MEASURED** — one two-mesh ledger session on
  this tree settles it).
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
| `v15.150.38857` | 38,857 | 60 s | generated; static carries the §3 drag band pending #330 |

The generator's own quality floors (engine-side, `rw-mpas mesh/validate.rs`)
are anchored to the published family's measured readings rather than chosen by
taste, and that anchoring is stated in the refusal text: `min_dv_over_dc`
0.02 sits between the published x4's own 0.0336 — the roughest reading in the
family, which the floor deliberately admits — and the measured dislocation
class at 0.0147 and below. The published x4 passes at 1.68× the floor; nothing
is waived for it silently.

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

**`pip install gpuwm-hex` resolves only once `gpuwm>=2.5.3` is on PyPI.** That
floor is deliberate and it is hard: 2.5.3 is the first `gpuwm` that both
carries the physics-seam bytes this port pins by SHA-256 and stages the four
MPAS bridge binaries both front doors drive. A lower floor would let pip build
an install that can open neither door, which is a stranded install rather than
a degraded one. Until then, install from a checkout. `tree/README.md` carries
the matrix.

**Four work lanes are held back, on purpose** — none has had the battery run
against it, so none is in this tree: a diagnostics-output lane (+4 files), a
physics-tier-park lane (#261, +4), an allocation-ledger lane (#264, +3), and a
workspace-arenas lane (+2). Merging #264 in particular needs its own battery
pass. They are named here so that a reader who finds a referenced tool absent
knows it was withheld rather than lost.

**Both pins are tag-anchored now.** `gpuwm` carries annotated tags
`pin/mpas-port-arwen-seam` (`629ddb6f0`) and `pin/mpas-port-arwen-seam-v2`
(`0d04db712`, placed 2026-08-23). A commit that survives only as the tip of a
work branch is one branch deletion away from making the sixteen pins
unverifiable forever; a tag on each closes that.

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

What **is** verified locally: the sixteen `ARWEN_SOURCE_MANIFEST` files
hash-match **16/16** at `gpuwm` commit `0d04db712` (14/16 at `629ddb6f0`,
10/16 at `gpuwm`'s 2.5.0 release line, 9/16 at the earlier port-landing ref —
which is what makes it an anchor rather than a tautology). See `README.md`.
