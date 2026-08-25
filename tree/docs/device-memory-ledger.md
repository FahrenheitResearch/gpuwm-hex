# Device memory ledger — where the CUDA port's VRAM goes

> ## SUPERSEDED AS A CURRENT ACCOUNT — read this first
>
> **Everything below was measured 2026-08-20 at the Arwen seam pin
> `629ddb6f0`, which is BEFORE the Grell-Freitas local-memory frame cut.**
> The pin moved to `0d04db712` (measured 2026-08-24) and then converged with
> the engine release line as `26daaab7e` (measured 2026-08-25). The current
> account of record is the 2026-08-25 converged-stack section immediately
> below, `evidence/pin-move-335-20260825/node2/` and STATE.md section 5, not
> this document's 2026-08-20 body.
>
> | | this document (pre-cut, 08-20) | of record (converged, 08-25, 170 SM) |
> |---|---|---|
> | fixed term | 9,797.8 MiB | **4,339.1 MiB** |
> | per-cell slope | 86,630 B/cell | **103,696 B/cell** |
> | widest launched local frame | `gf_gfdrv_stage`, 29,264 B | **`wsm6_column`, 7,216 B** |
>
> **The saving this document identifies has been TAKEN, not left on the
> table.** Section 1 calls the `gf_gfdrv_stage` frame "the single largest
> available saving in the whole footprint". That was true when written and it
> is why the cut was made: the pin move released **3,251.3 MiB** of fixed
> term. `gf_gfdrv_stage` has since left the launched set entirely. Nobody
> should read section 1 as an opportunity that is still open.
>
> **These measurements are kept, and deliberately.** They are the BEFORE arm
> of the pin move's own delta — the 3,251.3 MiB is a subtraction against the
> numbers on this page, so deleting them would delete the evidence for the
> improvement. Read the whole document in the past tense.
>
> **One apparent contradiction, resolved, because it will otherwise be
> re-discovered as a defect.** `evidence/gf-pin-move-measured-20260824/`'s
> VERDICT-LEDGER reports `ysu_column` as the widest local frame in BOTH the
> old and the new arm, and a local-frame reservation delta of 0.0 MiB. That
> does not contradict this document — it is exactly the instrument limitation
> section 5 of this page documents in advance: a post-run `gc.get_objects()`
> scan cannot see the `gf_gfdrv_stage` frame at all, because
> `gpuwm/core/gf.py:166` rebinds the `RawKernel` on every call and the
> 29,264 B frame is unreachable by the end of a run. The scan understates the
> largest single allocation by 4,990 MiB and reports 9,232 B instead. The
> campaign used that scan-based row, so its 0.0 MiB delta measures the
> instrument, not the cut. The fixed term falling 3,251.3 MiB in the same
> session is the number that reflects the cut.

---

## 2026-08-25 — the converged-stack arm (the proving node, RTX 5090, 170 SM) — OF RECORD

The seam pin converged with the engine release line (#335): hex
**`7fe514b`** + engine **`26daaab7e`** (`pin/mpas-port-arwen-seam-v4`,
the seam-converge merge into the engine release line), which brings
the #310 device-sized radiation chunk, the release line's frame-cut
wave, and the refl10cm/q2 history stream into one stack. Measured
2026-08-25 (UTC 16:46–16:52Z) on **the proving node, RTX 5090
(32,607 MiB, 170 SM)**, the #264 instrument
(`tools/device_memory_ledger/hex_ledger_probe.py`), 6 steps × 120 s
full physics, peak = this process's `nvidia-smi` row. Raw ledgers and
the drive script: `evidence/pin-move-335-20260825/node2/`.

| mesh | cells | peak (process nvidia-smi) |
| --- | --- | --- |
| `x1.40962` | 40,962 | **8,390 MiB** |
| `x4.163842` | 163,842 | **20,542 MiB** |

Two-point fit: **fixed 4,339.1 MiB + 103,696 B/cell** on this card.
Against the superseded 170 SM row (`0d04db712`, 2026-08-24: 6,296.5 MiB
+ 93,474 B/cell; x1 9,948, x4 20,902) the fixed term falls
**1,957.4 MiB** (#310 narrows the LW chunk on a 170 SM part; the frame
wave shrinks the local-memory reservation — widest launched frame is now
`wsm6_column` 7,216 B, predicted reservation upper bound 1,797.0 MiB)
and the slope rises **10,222 B/cell** (the refl10cm/q2 history
publication and the release line's seam-file evolution ride per-cell).
Net: x1 −1,558 MiB, x4 −360 MiB. x1.40962 fits a 12 GiB budget with
3,898 MiB of headroom; x4.163842 remains a 32 GiB-card configuration
under the unchanged 24 GiB `NATIVE_DEVICE_FLOOR`. The fixed term is
per-card: none of this transfers to smaller parts, whose rows are
measured, never derived.

Same-session corruption check on this no-ECC part: the #327 restart
gate ran first on the same card and stack — two independent processes
(uninterrupted parent, fresh-process restart worker) produced
byte-identical history (`118332d5…` both), which is the dual-run byte
comparison this node's hardware history requires.

## 2026-08-24 — the merged-tip arm (the second proving node, RTX 5070 Ti, 70 SM)

The 08-24 pin-move campaign left one open row: the integration tip
carries both the GF frame cut AND the #308 copy-elision merge, and no
ledger had run on that combination. This section is that measurement.
It is a NEW dated arm, not a revision of anything below; the 2026-08-20
content stays untouched as the pre-cut record.

Measured 2026-08-24 (UTC 2026-08-25T04:33–04:58Z) on **the second proving node,
RTX 5070 Ti (16,303 MiB, 70 SM)**, hex tree **`9a87d27`** (the merged
the integration tip), engine pin **`0d04db712`**, the #264
instrument (`tools/device_memory_ledger/hex_ledger_probe.py`, selftest
rc 0), 6 steps × 120 s full physics, peak = this process's `nvidia-smi`
row. Raw readings and the drive scripts:
`evidence/merged-tip-ledger-20260824/`.

| arm | mesh | cells | peak MiB | peak phase | pool MiB | non-pool MiB |
|---|---|---:|---:|---|---:|---:|
| merged tip | `x1.40962` | 40,962 | **6,724.0** | after_step_1 | 5,414.0 | 1,310.0 |
| merged tip | `u96.64002` | 64,002 | **9,344.0** | after_step_6 | 8,126.3 | 1,217.7 |
| + #310 narrowing | `x1.40962` | 40,962 | **5,604.0** | after_step_2 | 4,274.3 | 1,329.7 |
| + #310 narrowing | `u96.64002` | 64,002 | **8,664.0** | after_step_6 | 7,386.6 | 1,277.4 |

Two-mesh affine fit at the merged tip, THIS CARD: 2,066.0 MiB +
119,239 B/cell — with the caveat that the two peaks land at different
instants (x1 in step 1's radiation; u96 at the step-6 radiation +
history coincidence), so the pair does not separate card constant from
slope cleanly. Quote the per-mesh peaks. **The fixed term is per-card:
nothing here moves the 170 SM model of record (6,296.5 MiB +
93,474 B/cell at `0d04db712`, pre-#308), and the merged tip on that
card class remains NOT MEASURED.**

What sits at the peak: **2,832.8 MiB of RRTMG/McICA chunk transients,
identical at both mesh sizes** — the mesh-independent radiation
workspace this document's tier-2 section identified (`rrtmg_sw.py`
spcvmc workspace 1,745.6 MiB and friends), priced by
`legacy_radiation_vram_bytes` at 2,832.4 MiB for SW 2048 / LW 4096.

**#310 took it** (the engine chunk-narrow lane, `gpuwm`
`1a665e3fc`): the chunk width now defaults to the smallest 256-multiple
saturating the device's resident threads, capped at the old constants.
On this 70 SM part that is SW 1024 / LW 768: x1 releases 1,120.0 MiB
(the peak moves off radiation entirely — RRTMG live at the new peak is
0.4 MiB), u96 releases 680.0 MiB (its step-6 pool event shrinks from
+1,745.7 to +872.8 MiB, exactly the halved SW workspace). Output is
byte-identical (135/135 history surfaces, 31/31 per-step fingerprints,
flip-validated comparator) at zero measured wall cost. On a 170 SM part
the fix halves only LW (4096 → 2048; SW's ceiling binds), and no arm
ran there.

**12 GiB, on this card, measured:** x1.40962 6,724.0 → 5,604.0 MiB and
u96.64002 9,344.0 → 8,664.0 MiB. Both meshes fit a 12 GiB budget at the
merged tip with ≥ 2.9 GiB headroom (≥ 3.5 GiB with #310). `x4.163842`
was not attempted — its 24 GiB admission floor exceeds the card.

---

Measured 2026-08-20 on one RTX 5090 (32,607 MiB, 170 SM, sm_120), float32,
nVertLevels=55, full physics, 6 steps, at two mesh sizes so the fixed and
per-cell terms separate by subtraction rather than by assumption.

**Tense warning:** every present-tense claim below describes the pre-cut
stack at pin `629ddb6f0`. It is left in its original wording rather than
rewritten, so that it remains quotable as the before arm.

Runs: `x1.40962` (40,962 cells / 122,880 edges) and `x4.163842`
(163,842 cells / 491,520 edges), both from GFS 2026-08-12 06Z, both rc 0,
`status: passed`.

## The measured model

```
footprint(cells) = 9,798 MiB + 86,630 B/cell        (this card, this build)
```

| tier | fixed MiB | B/cell | what it is |
|---|---:|---:|---|
| non-pool | 7,730.9 | 1,193 | CUDA context, kernel local-memory backing store, module images |
| pool tier 1 — resident | 0.5 | 37,064 | 385 arrays live between steps |
| pool tier 2 — transient | 1,253.0 | 47,728 | step workspace live at the peak instant |
| pool tier 3 — allocator | 813.3 | 645 | pool blocks held above the live peak |
| **total** | **9,797.8** | **86,630** | |

Whole-process anchors (this process's `nvidia-smi` row, not device-wide):
x1 = 13,182.0 MiB, x4 = 23,334.0 MiB.

Prognostic state is 4,628 B/cell — **5.3 %** of the slope. The other 94.7 %
is resident physics tendency storage, step workspace and allocator headroom,
itemised below.

## 1. The fixed term is a property of the CARD, not of the model

Half of a 10 GiB card disappears before the first cell exists, and 7,034.0 MiB
of the 7,777.5 MiB non-pool total is **one allocation**: the CUDA local-memory
backing store.

| class | x1 MiB | x4 MiB | scales with cells | dtype | lifetime | allocated at |
|---|---:|---:|---|---|---|---|
| CUDA context + CuPy/NVRTC baseline | 498.0 | 498.0 | no | — | process | first CUDA touch |
| local-memory frame, `gf_gfdrv_stage` (29,264 B/thread) — **HISTORICAL: cut at pin `0d04db712`, no longer in the launched set** | 4,990.0 | 4,990.0 | no | local | process | `gpuwm/core/gf.py:168` |
| local-memory frame, `ysu_column` (9,232 B/thread) | 1,790.0 | 1,790.0 | no | local | process | `gpuwm/core/ysu.py:104` |
| local-memory frame, `rlw_rtrn_march` (2,048 B/thread) | 254.0 | 254.0 | no | local | process | `gpuwm/core/rrtmg_lw.py:3979` |
| non-pool taken during device-stack construct | 142.4 | 207.3 | yes (554 B/cell) | cubin + | process | `cuda_backend/runtime.py:270` |
| **unattributed residue** | **103.1** | **178.0** | 639 B/cell | | | |
| non-pool total, measured | 7,777.5 | 7,917.3 | 1,193 B/cell | | | |

The three local-memory rows are **increments to one pool**, not three pools.
CUDA sizes a single per-context local backing store to the largest single
kernel demand and never returns it while the context lives:

* launched alone, `gf_gfdrv_stage` reserves **7,034.0 MiB** — exactly
  254.0 + 1,790.0 + 4,990.0, the sum of the three in-run increments;
* launch `gf_gfdrv_stage` first and `rlw_rtrn_march`'s own increment is
  **0.0 MiB**.

Consequences worth acting on:

* The store is sized from the card's resident-thread capacity, not from the
  mesh, so a "fixed term" quoted without naming the card is meaningless. Its
  size on any other card is NOT MEASURED — only this 170 SM part was run.
* It is driven by **one kernel**. `gf_gfdrv_stage` carries a 29,264 B
  per-thread frame; cutting that frame is worth up to 4,990 MiB on this card
  and is the single largest available saving in the whole footprint.
  **TAKEN, 2026-08-24.** This is the finding the pin move acted on. The cut
  landed at `0d04db712` and released 3,251.3 MiB of fixed term (9,547.8 →
  6,296.5 MiB); `gf_gfdrv_stage` is gone from the launched set and the widest
  launched frame is now `ysu_column` at 9,232 B. This bullet is a record of
  why the cut happened, NOT a saving still on offer.
* `gf_deep_stage` (26,880 B) and `gf_shallow_stage` (18,944 B) are compiled
  but never launched from the shipped package — they would cost 6,438.0 and
  4,462.0 MiB if a future call site launched them first.

Module images are not the story: all eight physics modules together load
**4.0 MiB** of device code (`module-images.json`).

## 2. The slope: 86,630 B/cell against 4,628 B/cell of state

### Tier 1 — resident, 37,064 B/cell (8.0 × the prognostic state)

1,448.4 MiB at x1 → 5,791.8 MiB at x4, over 385 named arrays, all `run`
lifetime, and with essentially **zero fixed part** (0.5 MiB) — this tier is
pure per-cell. The largest rows:

| array | x1 MiB | x4 MiB | B/cell | dtype | allocated at |
|---|---:|---:|---:|---|---|
| `atm.terrain.zb3_cell` | 87.5 | 350.0 | 2,240 | float32 | `cuda_backend/containers.py:736` |
| `atm.terrain.zb_cell` | 87.5 | 350.0 | 2,240 | float32 | `cuda_backend/containers.py:735` |
| `atm.state.scalars` | 51.6 | 206.3 | 1,320 | float32 | `cuda_transport_v841.py:612` |
| (unnamed) | 34.4 | 137.5 | 880 | float32 | `cuda_arwen_physics_v841.py:1328` |
| `atm.saved.normal_velocity` | 25.8 | 103.1 | 660 | float32 | `cuda_backend/recovery.py:267` |
| `atm.state.rho_u` | 25.8 | 103.1 | 660 | float32 | `cuda_driver.py:3188` |
| `atm.vertical.zxu` | 25.8 | 103.1 | 660 | float32 | `cuda_backend/containers.py:433` |

The shape of this tier is the finding: after the seven rows above it is a long
flat tail. **104 of the 385 arrays sit at 220 B/cell** — one full
`nVertLevels × nCells` float32 field each, 8.6 MiB at x1 and 34.4 MiB at x4.

Counted by owner at x4:

| owner | arrays | x4 MiB |
|---|---:|---:|
| `gpuwm/core/physics.py` | 174 | 1,273.8 |
| `gpuwm/core/mpas_column_batch.py` | 74 | 1,531.9 |
| `cuda_backend/containers.py` | 50 | 1,231.3 |
| `cuda_gwdo_v841.py` | 19 | 214.4 |
| `gpuwm/core/ysu.py` | 15 | 279.4 |
| `cuda_backend/recovery.py` | 7 | 310.0 |
| `cuda_arwen_physics_v841.py` | 6 | 291.3 |
| `cuda_driver.py` | 6 | 275.6 |
| remainder (7 files) | 34 | 383.9 |

**The ArWen physics seam owns 265 of the 385 arrays and 3,085.4 MiB of the
5,791.7 MiB resident tier — 53.3 %.** The dycore state the port itself holds
is the smaller half. What that seam keeps live between steps is four parallel
tendency sets (`tendencies`, `pbl_tendencies`, `radiation_tendencies`,
`cumulus_tendencies`, `physics.py:1270-1414`), a `last_ysu` result set
(`ysu.py:93`), a `_last_gwdo_result` set (`cuda_gwdo_v841.py:1523-1530`), a
WSM6 scratch set (`mpas_column_batch.py:372`) and the seam's `_in`/`_out`
mirrors (`mpas_column_batch.py:552-562`).

All 385 arrays are in `evidence/device-memory-ledger-264/LEDGER-TABLE.json`
under `resident_arrays`, with dtype, x4 shape, B/cell and site.

### Tier 2 — transient, 47,728 B/cell (the largest slope contributor)

3,117.5 MiB at x1 → 8,710.7 MiB at x4, on top of the resident tier, live
simultaneously at the global peak (both peaks land in `step_integrate`).

The peak lands in a **different scheme at each mesh**, which is itself the
point:

* x1 peak is radiation-bound — `rrtmg_sw.py:3466` alone holds 1,745.6 MiB,
  plus `rrtmg_mcica.py:617` 392.0 and `rrtmg_sw.py:3471` 299.2. These are
  fixed-chunk column batches: they do **not** grow from x1 to x4, so they
  belong to the fixed term in disguise and dominate only small meshes.
* x4 peak is dycore-bound — `cuda_driver.py:3222/3253`, `cuda_horizontal.py:991`
  and `cuda_backend/recovery.py:267` at 412.5 MiB each, then six
  `cuda_horizontal_v841.py` operator temporaries at 309.4 MiB each.

So the slope is set by horizontal-operator and acoustic-substep temporaries,
not by state. Full per-site lists for both peaks are in `LEDGER-TABLE.json`
under `peak_instant_sites` (139 sites at x1, 188 at x4).

### Tier 3 — allocator, 645 B/cell + 813.3 MiB fixed

`pool.total_bytes()` sits 838.6 MiB (x1) / 914.2 MiB (x4) above the live
peak: CuPy best-fit rounding plus free blocks the pool keeps rather than
returns. Nearly mesh-independent, and it is the cheapest MiB in the ledger to
recover — a pool policy change reaches it without touching any physics.

## 3. What is not attributed

**103.1 MiB at x1, 178.0 MiB at x4** of the non-pool term (1.3 % and 2.2 %),
scaling at 639 B/cell. Ruled out by measurement, not by argument:

* module images — all eight physics modules measured at 4.0 MiB total;
* kernel local frames — every kernel in the four physics modules had its
  `local_size_bytes` read from the driver, and every frame > 0 was launched
  and measured; the two zero-local controls reserved exactly 0.0 MiB;
* pool allocations — the hook sees every one and they are in tiers 1-3;
* sibling-process contamination — every number here is this process's own
  `nvidia-smi` row, which another process cannot move.

That it scales with cells says the remainder is a small per-cell allocation
taken outside the CuPy pool. It is not identified.

## Instrument

`tools/device_memory_ledger/`, validated against known answers in both
directions before any number here was believed:

| instrument | validation |
|---|---|
| `kernel_reservation.py` | 512 MiB array → hook, pool and `nvidia-smi` all move by 512 MiB; a view moves nothing; an 8,192 B local frame reserves 1,784.0 MiB by `cudaMemGetInfo` **and** 1,784.0 MiB by this process's `nvidia-smi` row; a zero-local kernel reserves 0; a second launch reserves 0 |
| `reservation_probe.py` | positive control `gf_gfdrv_stage` standalone = 7,034.0 MiB = the sum of the three in-run increments; negative control (zero-local kernel) = 0.0 MiB; reverse-order control drives `rlw_rtrn_march` to 0.0 MiB |
| non-perturbation | traced vs untraced x1 run differ in 1 of 1,935 receipt leaf fields, and that field is `memory_admission/free_bytes` (free VRAM at admission, moved by other lanes on the card) |

Two traps this instrument had to survive, both of which silently produce
wrong ledgers:

1. **`cudaMemGetInfo` is device-wide.** On a shared card a sibling lane's
   allocation lands inside any window measured that way; one contended run
   here reported a 7,052 MiB radiation delta and a *negative* surface-layer
   delta. Every number in this ledger comes from the per-process row instead.
2. **A post-run `gc.get_objects()` kernel scan misses the biggest frame.**
   `RawModule.get_function` returns a fresh `RawKernel` that
   `gpuwm/core/gf.py:166` rebinds on every call, so by the end of the run the
   29,264 B frame is unreachable and a scan reports a maximum local frame of
   9,232 B — understating the largest single allocation in the process by
   4,990 MiB.

A third route stays out of reach by design: RRTMG compiles through
`compile_using_nvrtc` + `cupy.cuda.function.Module` to avoid CuPy's appended
`-ftz=true`. Proxying that class is not an option — CuPy's own Cython
elementwise path constructs and type-checks it, and a proxy kills every cupy
expression in the process with `Cannot convert TracedCudaModule to
cupy.cuda.function.Module`. Those kernels are measured out of band by
`reservation_probe.py` instead.

## Correction to commit ff28ef5

That commit's subject says "the pool sees 89% of the rest". The figure is not
supported by anything measured here and no number in this document produces
it. The measured shares are: the CuPy pool accounts for **98.6 %** of the
slope (85,437 of 86,630 B/cell) and **21.1 %** of the fixed term (2,066.8 of
9,797.8 MiB); as a share of the whole footprint it is 41.0 % at x1 and
66.1 % at x4. Forward commits only, so the subject stands as written and this
note is the correction of record.

Commit f531d04's own body repeats the same fixed-term share as 16.4 %. That
is arithmetic error, not measurement: 2,066.8 / 9,797.8 = 21.1 %. The four
percentages in the paragraph above are the checked ones.
