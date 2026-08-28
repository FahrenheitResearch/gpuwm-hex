# Changelog

*This file accumulated without a version cut through 0.1.0 and 0.1.1, so
entries below carry work from those lines as well as this one, and no honest
boundary can be drawn between them after the fact. The published release note
for each shipped version is the summary of record for what that version
contained. From 0.2.0 forward the file is cut at the release.*

## 0.2.0

New:
- **A limited-area forecast runs the full physics stack.** `gpuwm-hex
  forecast --lbc-dir` integrates a culled regional mesh behind boundary
  files built from its own coarse parent, with WSM6, Grell-Freitas, YSU,
  YSU-GWDO, revised-MO, NoahMP, cloud fraction and RRTMG all attached.
  Measured on a 10 GiB RTX 3080: six hours, 1,080/1,080 steps, 13 history
  frames on 11,020 cells, peak 6,224 MiB, median 0.271 s/step, 343 rendered
  products. Against a global full-physics run over the same ground at t+6 h:
  theta 1.117 K RMS (r = 0.999973), precipitation r = 0.95, reflectivity
  r = 0.82, vertical velocity r = 0.621. Before this, the regional path was a
  dry dycore carrying one passive moisture variable and published no
  renderable weather field at all. `evidence/regional-physics-20260826/`.
- **`gpuwm-hex cull`** cuts a limited-area grid, static and initial condition
  out of a global case in about a second, where a native regional init took
  775 s on the 121,182-cell parent `v4.75.121182`, measured 2026-08-26 with
  `gpuwm-hex init` and NOT against native. Culling that init is the
  supported route into the
  regional lane.
- **`gpuwm-hex swath`** decides where the fine grid goes, from a coarse
  forecast's own fields: detection on sea-level-reduced pressure, a
  declarative threat grammar, ranking that is commensurable across phenomena
  carrying different units, and hysteresis so a placement does not chase
  noise. Four independently placed grids over four different kinds of weather
  each completed 1,080/1,080 full-physics steps on one card
  (`evidence/four-swaths-20260827/`).
- **`gpuwm-hex cycle`** follows weather across cycles: plan, cull, force,
  forecast, render. Two cycles of one real case ran end to end in 1,058 s at
  peak 7,744 MiB, cycle 2 re-detecting six hours on and continuing all four
  slots — three reusing their mesh, one regenerating. Starting a corridor
  from transplanted parent state rather than from the beginning transplants
  in 0.83 s and saves **273.8 s, 43 %**, against a real baseline arm.
  `docs/cycle-door.md`, `evidence/cycling-loop-20260827/`.
- **The regional anchor is keyed to a configuration class, not to a cull's
  own boundary-mask digest.** A re-placed swath was a new digest and therefore
  a new anchor, at 5.5 to 8.7 minutes of card against 4 to 6 minutes for the
  forecast it admitted — roughly three forecasts of permission per forecast
  run, which is what made cycling unaffordable. Five concentric culls of one
  parent were minted independently and returned the same verdict at the same
  Courant margin, so not one input the mint reads distinguished them. The
  class is earned once (two 1,080-step runs, 13/13 masked digests identical);
  the contract deck stays per-geometry because it runs on the cull's own zone
  geometry. Residual per-geometry cost 147.3 / 136.2 s of deck against the
  288-382 s of mint it retired, and anchors now admit by presented content
  rather than by a row in source.

Changed:
- **BREAKING: the import namespace is `hexcore`.** It was `mpas_port` through
  0.1.1. `import mpas_port` no longer resolves and there is deliberately no
  alias shim. Every module path underneath is unchanged, so the migration is
  one token: `from mpas_port.X import Y` becomes `from hexcore.X import Y`.
  The distribution name (`gpuwm-hex`), the console script (`gpuwm-hex ...`)
  and every command surface are untouched. The old name overclaimed: this
  project keeps MPAS-A v8.4.1's **dycore and mesh** byte-identical, pinned as
  a specification, and deliberately does not match that model's physics,
  which is WRF physics run through MPAS's own plumbing. Naming the package
  after another project put that project's name in every user's import line
  for a relationship holding over half the model. `hexcore` names what is
  actually pinned and matches the distribution.
- **The shipped limited-area cut is wider, and the width is measured.** Five
  concentric culls of one parent against a no-boundary control: every field
  improves monotonically with cut width, because slicing at the fine core's
  edge discards the parent's own resolution ramp, which IS the
  intermediate-resolution ladder and is already inside the mesh. The knee is
  **1.35x** (`w` r 0.624 to 0.744, 2 m temperature r 0.578 to 0.852) for
  +27 % cells, +25 s wall, +42 MiB and zero extra forecasts. All nine shipped
  placement rows carry `cull_pad_scale` 1.35. `evidence/nest-ratio-20260827/`.
- **Device-memory admission takes the card's shape.** See the closed items
  below: this replaces the affine row and the flat headroom outright.

Fixed:
- **A fresh `pip install gpuwm-hex` resolved an engine this port's own pin
  refuses.** The dependency was `gpuwm>=2.5.5` with no ceiling, so pip took
  the newest published engine — at the time 2.5.7 — and the forecast lane
  then refused at launch with two SHA-256 digests and no version number while
  `gpuwm-hex doctor` reported the estate healthy and exited 0. A green
  install and a dead run, with no route out by reading. Three changes, all
  default-on: the declared range is now **`gpuwm>=2.5.8,<2.5.9`**, derived by
  `hexcore.engine_pin` from a measured table of every published engine rather
  than typed, with an **exclusive ceiling at the first engine nobody has
  measured** so a future engine cut cannot re-open it; `doctor` hashes the
  pinned files that live in `site-packages` and reports the offending version
  and the fix; and the forecast door's refusal names the version it found, the
  version it wants, which files moved, and the two commands that close the
  gap. `evidence/standalone-20260827/`, `evidence/userwalk-20260827/`.
- **The engine floor is 2.5.8, and it is the only usable engine.** The seam
  manifest was re-pinned on 2026-08-28 and every row of the verdict table
  moved with it, because `moved` is measured against *this port's* sixteen
  files: 2.5.6, the floor of the day before, now reads 4 of 16 moved. 2.5.8 is
  the only published gpuwm whose bytes match, so this port has no fallback
  engine — stated because it is a real exposure, not because it is
  comfortable. The table is spliced from the instrument's JSON, never typed
  (`evidence/repin-258-20260828/engine-verdicts.json`).
- **`doctor` printed two adjacent lines that contradicted each other, and the
  forecast lane refused for a reason that had stopped being true.** On a
  byte-perfect install of the pinned engine the report read `16 of 16 pinned
  files are in this install and all 16 match`, and the next line said an
  installed gpuwm cannot satisfy the pin and that lane needs a source
  checkout. The second was a constant string written when the manifest pinned
  `docs/mpas-seam.md`, which no wheel carried; gpuwm 2.5.8 ships it inside the
  wheel at the manifest's own key. Measured 2026-08-28 in a virtualenv holding
  only the published wheels: `inspect_seam` over the install returns
  `checked=16, matched=16, moved=(), absent=()`, `doctor` exits 0, and the
  forecast door ACCEPTS `--gpuwm-checkout <site-packages>` at its own byte
  check. What still refuses is the driver's `verify_arwen_checkout_git`, and
  for a different reason: it records the checkout's HEAD, tree and dirty paths
  into every receipt so the executed source can be named by commit, and
  `site-packages` is not a git working tree. That refusal was a bare
  `CalledProcessError` at exit 128 and is now named. The guard is KEPT and its
  reason is corrected everywhere it is stated — the driver, `engine_pin`,
  both doors, `doctor`, `tools/battery/gpu_gates.txt`, `pyproject.toml`,
  README and five manual chapters. Retiring it needs a receipt identity that
  does not spell a commit and is a named follow-up in
  `docs/release-checklist-0.2.md`. `evidence/checkout-reason-20260828/`.
- **The precipitation verdict was stated backwards on three user-facing
  pages.** README, the concepts chapter and the troubleshooting chapter all
  carried "net domain-mean precipitation runs about 15 % dry" with no referee
  attached. That figure is a global domain mean against native MPAS-A, the
  referee retired on 2026-08-20. The live referee — skill against
  observations — ran on 2026-08-25 against NCEP/EMC Stage-IV and returned the
  opposite sign in all four cases (+0.0247 mm/h paired, 95 %
  [+0.0041, +0.0606]; frequency bias at 1 mm/h 1.59 / 1.35 / 1.38 / 0.77, so
  it rains over too much area). Each page now names which referee each number
  belongs to and states the obs sample's limits rather than trading one
  overstatement for another: four cases, two of them the divergence cases
  themselves, one truncated at 23 h with the largest bias, one +41.5 % on
  almost no rain, and the two clean complete cases at +9.8 % and +2.4 %. The
  troubleshooting chapter routed a user whose run looked too WET to a page
  saying the model runs dry; its advice is rewritten around what the live
  referee measured. The three declared-divergence magnitudes also carry a
  tense now: they entered the tree already finished on 2026-08-20 with no
  receipt, no card and no run commit, under engine pin `629ddb6f0`, and
  whether any of them survived the three engine pin moves since is NOT
  MEASURED.
- **A shipped capability was documented as impossible.** The concepts chapter
  said divergence 3's reflectivity half "cannot be scored at all against this
  build, because the history stream carries no reflectivity field to
  compare". `refl10cm` has been computed in the due step's own WSM6 call and
  published in every history frame, default on, since `e3caf6c`. The chapter
  now carries what the run returned: on the one case re-run with the field,
  86 model 35 dBZ objects against 54 observed, 8 matched at a median 110.9 km
  displacement, point CSI 0.0916 at 20 dBZ and 0.0097 at 40 dBZ over 643,419
  pairs, with the resolution cost named. `evidence/history-refl-q2-20260825/`.
- **The limited-area lane is reachable from published artefacts.** Through
  2.5.7 it was not: no published `rw_mpas_mesh` carried `--cull-parent` and
  `rw_mpas_lbc` was in no bundle and no published source, so the flagship
  feature of this release could be run only from a source-built engine.
  Measured 2026-08-28 on the pinned engine: `gpuwm fetch-bridges` stages 26 of
  26 artifacts against its packaged pins, `rw_mpas_lbc` included, and
  `gpuwm-hex cull` drove the staged published `rw_mpas_mesh` through two real
  cuts of the 40,962-cell global parent (338 and 606 cells; grid, static and
  init each written; 0.9 s). Not yet measured from published artefacts: a
  boundary set written by the published `rw_mpas_lbc` and a `--lbc-dir`
  forecast behind it.
- **The distribution's own test battery could not pass on the tree that gets
  published.** Thirty tests read measurement receipts under `evidence/`,
  which every published surface holds out on purpose; they raised
  `FileNotFoundError` on paths that never existed on the reader's machine,
  and `ci.yml` runs on push. They now skip with a stated reason when the
  tree does not carry its receipts, and still FAIL when it does carry them
  and the receipt a row names is missing — a held-out record and a row citing
  a measurement nobody can check are different findings and get different
  answers. Two packaging guards required `docs/LANE-BRIEFING.md` to EXIST in
  order to check that it does not ship, which a published tree cannot
  satisfy; they now assert the outcome instead. `setuptools` is declared as
  the test dependency it always was (`ensurepip` stopped seeding it at Python
  3.12, so three packaging gates failed on 3.13 and not on 3.11). And the
  forecast-door tests substitute the card-shape seam, which
  `GPUWM_HEX_NO_LOCAL_GPU` — set by this project's own CI — made refuse.
  Assembled public tree: **32 failed before, 0 after**. Unpacked sdist: **40
  failed before, 0 after**.
- **Three campaign scripts published home-directory paths.**
  `tools/device_memory_ledger/run_arm.sh`, `run_kern.sh` and
  `ledger_table.py` carried 24 absolute paths under one machine's home
  directory — a private working-tree layout and a venv path, in scripts that
  could not run anywhere else anyway. They read `HEX_REPO`, `HEX_WORK`,
  `HEX_ASSETS`, `HEX_PYTHON` and `ARWEN_CHECKOUT` now, and refuse by name
  when one is unset rather than defaulting somewhere wrong.
- **The device-memory gate had the wrong shape, and it failed in both
  directions at once.** The affine row charged card-sized workspace knees —
  Grell-Freitas, YSU and the RRTMG shortwave chunk, the three sites that do
  NOT scale 4x on a 4x mesh — as per-cell growth. Across seventeen recorded
  peaks its error spans **-41.42 % to +27.19 %**, and **six runs exceeded what
  the old gate demanded**: four on the limited-area path the cascade actually
  runs, short by 1,056 / 1,104 / 860 / 716 MiB, plus 96 and 28 MiB on two
  graded globals. The gate now asks
  `device_admission.model_for_card(card, configuration)` for
  `core(card, configuration)` plus a Grell-Freitas workspace at
  `min(cells, SMs x 4 x 64)` plus a YSU workspace at
  `min(cells, SMs x 16 x 32)` plus a per-cell term, and it covers all
  seventeen. The margin is named — that card's shortwave workspace plus
  11.2 MiB of instrument convention — instead of a flat 512 MiB. The retired
  arm stays computable at `device_admission.RETIRED_AFFINE_ROW_20260826` so
  the comparison can be re-made rather than re-argued.
  `evidence/memory-shape-20260827/`.
- **The door reads the live card.** Two registered meshes that the shaped
  gate's predecessor had moved from admitted to refused on a 10 GiB card are
  admitted again, confirmed on the real hardware rather than derived. The
  workaround that reported itself as a workaround is retired with the defect.

Known issue:
- Across four placed variable-resolution meshes at one card, one engine pin
  and one schedule, the peak spans **+3.89 % to -6.76 %**, and one mesh with
  15,343 MORE cells peaked 318 MiB LOWER. The mechanism is named — allocator
  placement of the shortwave block, which stops being servable from the free
  list at the step where radiation and history capture coincide — but it is
  NOT separated by an A/B on those meshes. The limited-area core is an
  envelope over five samples, not a per-allocation fit.
- Pool retention is 20-30 % of the footprint with no arena owning it.
- A cycle is one parent integration read at successive times, not a parent
  regenerated per cycle. Regenerating it is the operational remedy for a
  corridor that has moved far, and it is not built. Two of four admitted
  slots per cycle are skipped as background culls by a measured minimum-edge
  ratio.
- A corridor started from transplanted parent state begins with no cloud ice,
  snow or graupel, because the initial-condition stream carries no slot for
  them. Hour-zero reflectivity does not correlate; one hour on, the
  microphysics has re-formed the ice and r = 0.863. Temperature agrees to
  five decimals throughout.
- No obs-skill score exists for a cycled case. Every limited-area verdict
  above is against a global run over the same ground, not against
  observations.

---

Added:
- Registry row `v16.66.195629` — `v16.66.195630` regenerated from its own spec
  row, unchanged, by a generator that no longer makes four-sided cells (gpuwm
  2026-08-26, `evidence/meshgen-coordination-20260826/`).
  The cause of the old row's blow-up was the graded generator's insertion
  operator placing its new generator on the near-cocircular quad's own
  circumcentre, where its Delaunay ring is exactly the four quad cells — 18 of
  18 and 13 of 13 insertions measured, and readable in the shipped bytes
  themselves: cell 195615's four neighbours lie on a circle of radius
  20.783 km to within 0.10 km and the cell sits 0.564 km from its centre, 2.7 %
  of the radius. The surgery's local polish then PINNED the cell it had just
  damaged, and neither the repair loop nor the emit gate ever read a
  coordination number, because a quadrilateral plus the two heptagons the same
  operation makes leaves `sum(6 - nEdgesOnCell)` at exactly 12. All three
  graded spec rows now regenerate clean: `v16.66.195629` (195,629 cells,
  `{5: 1037, 6: 193568, 7: 1023, 8: 1}`), `v15.60.224210` (unchanged cell
  count, `{5: 1073, 6: 222076, 7: 1061}`, digests moved), and `v20.80.151649`,
  whose regenerated geometry is BIT-IDENTICAL to the registered bytes — every
  cell centre, ring and edge length exactly equal — so its completed 6 h
  forecast still describes what the generator emits.

- Ledger #367 is closed at the producer, and it was never reporting-only
  (`evidence/meshgen-coordination-20260826/LEDGER-367-CLOSEOUT.md`).
  `rw-mpas` `density.rs::polygon_contains` accepted on `|winding| > pi`, which
  cannot tell a point inside a ring from a point whose ANTIPODE is inside it,
  so every polygon region refined a congruent ghost of itself on the far side
  of the globe. That ghost's edge is a step, not a ramp — the signed distance
  jumps from −19,900 km to +19,900 km across it, both ends of a saturated
  `tanh` — so the spacing field fell from the region's spacing to the
  background across one cell and the generator's gradient gate refused every
  swath spec the placement layer emits, at `background/spacing - 1` per cell
  every time (1775 %/cell at 4 km in 75 km). Fixing the containment test alone
  cleared both symptoms: all four emitted specs now clear the gate at every
  spacing tried, `tools/probe_polygon_attainment.py` returns "no defect
  reproduced at this engine build", and the polygon arm tracks its cap control
  to four figures. One correction to the row: the ghost was in the FIELD, not
  only the report — at a 600 km half-width it cost 299,497 cells against the
  equivalent cap's 175,721.

Changed:
- `cell_coordination_admission`'s remedy is re-anchored, not retired. It used
  to hand "regenerate" to the reader as a coin flip and name a generator-side
  follow-up; that follow-up has landed, so the refusal now names the fixed
  generator, the mechanism it fixed, and `v16.66.195629` as this row's
  replacement. The gate itself stays and must: the mesh it refuses already
  exists, is still registered, and is still on disk.
- The `v16.66.195630` row records that it is superseded. It stays registered
  and stays refused, because it is the bytes that measured the cost.
- The device-memory row of record is re-fitted at the merged tip (ruled 2026-08-26): `5,016.5 MiB + 98,748 B/cell` on the 170 SM card,
  replacing the 2026-08-25 row measured at hex `7fe514b`. One #264 session,
  both published meshes, same card, same engine pin, same protocol
  byte-for-byte — `evidence/memory-row-refit-20260826/`. The tip did NOT
  shift the footprint uniformly: `x1.40962` rose 484.0 MiB (8,390.0 ->
  8,874.0) while `x4.163842` FELL 96.0 MiB (20,542.0 -> 20,446.0), so the
  fixed term rose 677.4 MiB, the slope fell 4,948 B/cell, and the two rows
  cross at about 143,554 cells. Quoted at this tip the old row left
  `x1.40962`'s requirement 27.9 MiB above its measured peak — 5 % of the
  shared 512 MiB headroom, on the smallest published mesh. The superseded row
  retires computably (`device_admission.RETIRED_ROW_20260825`,
  `retired_converged_row_floor_bytes`), a test refuses any governing surface
  that quotes it as the footprint (RED at the lane base, 18 offences) and any
  shipped caller of either retired arm, and every `CARD_TIER_ROWS` entry now
  carries `measured_at_pin` / `restated_at_tip` — a per-card row with no pin
  on it is how the 170 SM row outlived its tree. Cells admitted: the 32 GiB
  part 270,915 -> 277,297; the 16 GiB part 111,524 -> 109,919; the 10 GiB
  part 42,934 -> 37,892.
- The RTX 3080 tier row re-borrows its slope from the merged tip
  (2,483.0 MiB + 98,748 B/cell). The fixed term is a property of the card and
  the slope a property of the build, so a BORROWED slope must be the current
  build's; the fixed term is re-derived from that card's own measured
  6,340.5 MiB peak and reproduces it exactly. Arithmetic on one measurement,
  not a new one — the row still declares NOT RE-MEASURED AT THE MERGED TIP.

Known issue *(both entries below are CLOSED at 0.2.0 and are kept because they
are the before-arm of the fix. The first was measured again at its own
protocol and REVERSED SIGN: `v20.80.151649` measures 19,255.25 MiB against a
19,297.8 MiB prediction, so the row OVER-predicts by 0.22 %, the two peak
conventions agree to 0.75 MiB, and the attribution to mesh shape below was
wrong. The second is fixed by the door reading the live card. Do not quote
either as open.)*:
- The re-fitted row still under-predicts the one GRADED mesh measured against
  it, and now does so past the gate. `v20.80.151649` (151,649 cells) peaked
  19,838.0 MiB against a 19,297.8 MiB prediction: +540.2 MiB (+2.80 %), and
  **28.2 MiB above its own `required_free_bytes`** — a card offered exactly
  that requirement would be admitted and then overrun. The re-fit did not
  cause it and did not remove it: both fitted points are quasi-uniform global
  meshes and this row is EXACT on both, which is what makes the excess
  attributable to the mesh shape rather than to a stale term (the 08-25 row
  missed the same point by +2.60 % and missed both uniform meshes as well).
  At most 11.2 MiB is the whole-device sampling convention, measured side by
  side in the same session. Pinned by
  `test_the_graded_point_exceeds_the_shipped_rows_requirement_and_says_so`.
  The remedy is one #264 arm on a graded mesh at this row's own protocol, not
  a wider gate; the drive script is committed at
  `evidence/memory-row-refit-20260826/node2/drive_graded.sh` and has not run.
  Until it does, the 512 MiB headroom is not spendable and graded meshes want
  margin above what the row returns.
- Two registered meshes — `x1.40962` and `v15.150.38857` — move from admitted
  to refused on the DEFAULT row on the 10 GiB desktop card, which the 08-25
  row admitted. The refusal is conservative rather than correct: the default
  row carries the 170 SM card's fixed term, and that card was measured
  running `x1.40962` at a 6,340.5 MiB peak. Its own row admits the mesh with
  2,244 MiB to spare (`--device-fixed-mib 2483.0 --device-bytes-per-cell
  98748`, chapter 6). A door that selects a measured tier row from the
  detected card would make that the default; until then the flag is a
  workaround and is reported as one.

Fixed:
- A global mesh carrying an all-zero `bdyMask` triple is a global mesh. MPAS
  writes that triple all-zero on a sphere and the unified `rw_mpas_static`
  follows the convention, so every static this project generates ships it;
  classifying a mesh as a regional cull on the PRESENCE of the triple made a
  closed sphere a bounded disk and refused it for being a sphere — Euler
  characteristic 2, boundary rings 1..7 empty. Measured on `v20.80.151649`,
  RTX 5090: bound clean, refused at load. Every generated-static row was
  affected; the published statics are native-made and carry no triple, which
  is why no test caught it. The test is now a boundary ZONE, a nonzero mask
  value, and an incomplete triple is still refused on presence.
- `--preflight` answers the timestep question beside the memory one instead
  of exiting on it. A row declaring an unanchored timestep ended the preflight
  before the admission verdict printed, so "will this mesh fit my card?" went
  unanswered for exactly the meshes people ask it about.

Changed:
- The device admission floor is re-proved against hardware (ruled 2026-08-26) and the citations that outlived the constant it replaced are
  retired. The floor itself is unchanged — the measured affine row plus one
  512 MiB headroom, from the single `hexcore.device_admission` surface,
  since 2026-08-25 — but the graded-mesh lane measured its capacity
  boundaries on a base that predated that change and merged the conclusions
  beside it, so three registry rows described a retired linear proxy as
  governing. `device_admission.retired_linear_floor_bytes` is now the one
  place that computes the retired arm, tests refuse any governing surface
  that quotes it as a requirement and any shipped caller of it, and
  `FLOOR_DERIVATION` records the ruling, the finding and the measured
  consequence. Measured on an RTX 5090: the 224,210-cell graded mesh is
  admitted on device memory (26,511.7 MiB predicted against 31,642.6 MiB
  free) where the proxy demanded more than the card holds, the 32 GiB part
  carries 270,915 cells against the proxy's 210,952, and a 1 h full-physics
  `v20.80.151649` forecast ran rc 0 at the shared sum. That run is also the
  row's first out-of-sample point and it came in 2.60 % OVER prediction,
  finishing 9.7 MiB inside its own requirement on the headroom; re-fitting
  the row at the merged tip is a named follow-up.

New:
- **The frozen lane runs at five timesteps, not one.** It was ruled on 2026-08-26 that the v8.4.1 column-physics lane stops being pinned to 120 s,
  and anchors were earned the same day at **100 s, 75 s, 20 s and 5 s** — each
  two forecasts on named hardware, finite at every step, every history frame
  identical between arms, minted on the already-registered `x1.40962` because
  an anchor is a property of the timestep and a Courant limit is an upper
  bound. A timestep with no anchor is still refused by name before anything is
  allocated. Only 120 s carries a native reference and only it ever can, so
  every new row records `native_reference=None` rather than being conflated
  with it. Three registered graded meshes go from refused at bind to runnable:
  `v16.66.195630` at **16.5 km** core spacing, and both 224k-cell rows at 75 s.
  Each anchor's health band is measured against a 120 s control on the same
  card, mesh and init, and reported as a trend as well as a min/max: 100 s and
  75 s track the control within parts in 1e4, while **20 s does not** — the
  vertical-velocity mean climbs monotonically to 5.53 m/s against 1.48 and
  keeps climbing, with `theta_m` max 5.1 K below the control. That run is
  finite at every step and byte-identical across arms, so it is a different
  solution rather than an unstable one; whether the cause is Grell-Freitas
  being called 180 times an hour instead of 30 or resolved dynamics is
  recorded as NOT MEASURED, with what would settle it. Convection-off is not
  covered: the frozen configuration pins `config_convection_scheme`, so the
  fine anchors certify the GF-on configuration and no other.

Fixed:
- **A global mesh was refused as a corrupt regional cull, closing the whole
  published family.** Every forecast on `x1.40962` — at any timestep,
  including the proven 120 s — died with `regional mesh is not a bounded disk:
  nCells-nEdges+nVertices = 2, not 1` and three empty-`bdyMask`-ring findings.
  Both were the proof the mesh is global read as proof of a broken cull: 2 is
  a closed sphere's Euler characteristic, and empty rings are what an all-zero
  mask means. `Mesh.validate` classified on the *presence* of the
  `bdyMaskCell/Edge/Vertex` triple, and native MPAS-A writes that triple into
  a global mesh's static file too, all zero — the published `x1.40962.static.nc`
  carries all three with zero nonzero entries. A cull has a boundary zone, so
  the rule is now the triple **plus a nonempty zone**; an all-zero triple on a
  mesh that is not a closed sphere is refused by name.

- The regional anchor is re-minted at the merged tip, and the NVRTC
  reciprocal defect is measured on a live sm_120 forecast. The #355 fix moved
  the third-order stencil denominator off a source literal in both
  `cuda_driver` and `cuda_transport`, superseding the pre-fix forecast pair;
  the anchor's source-binding check caught it and also showed the check was
  too narrow, so every translation unit the regional step launches through is
  named now. An A/B on the card — merged tip against the same tree with those
  two units reverted, the reverted arm reproducing the superseded digest
  exactly — measures the defect moving 97.8 % of interior `u` values and
  42.5 % of `theta` at three forecast hours, with the specified zone unmoved
  at every field and every lead. The same attribution probe re-run at the
  merged tip gives an unchanged 36,750 of 163,405 `kinetic_energy` values, so
  the reciprocal defect explains none of that divergence and its cause is
  still open; the count is pinned so a later lane cannot adopt the fix as its
  explanation.
- The model timestep is admitted from an earned-anchor registry
  (`hexcore.dt_admission`), on the same pattern as per-architecture and
  regional admission. The frozen v8.4.1 column-physics configuration refused
  any `config_dt` but 120.0 with a literal, and pinned `config_bldt_seconds`
  and `config_cudt_seconds` beside it; the premise of that refusal was
  "unproven at this timestep", not "wrong at this timestep". **The admitted
  set is unchanged** — 120 s holds the only anchor and every other value is
  still refused before anything is allocated — but the refusal now names the
  evidence the anchor rests on and the procedure that mints another, and a
  second anchor is one table row rather than an edit to two files. An anchor
  carries a schedule receipt (host-derivable: physics cadence step counts, the
  Grell-Freitas `cudt == dt` law WRF pins for `cu_physics = 3`, the RK
  schedule's shape against the proven one, the WSM6 minor-loop split, and
  clock closure in binary64), an integration anchor (two byte-identical
  forecasts on named hardware), and a nullable native reference — nullable
  because the one native MPAS-A v8.4.1 integration this program holds was run
  at 120 s and no other timestep can ever have one. `tools/mint_dt_anchor.py`
  mints and verifies; its verifier certifies the registered row and fails all
  six fabricated variants of it, and it refuses to mint at all unless it
  reproduces the archived 120 s stage tables exactly. Registering a second
  anchor moves the frozen lane off its proven timestep and is a ruling, not a
  tool run: `evidence/dt-admission-20260826/RULING-PACKET.md`.
- `gpuwm-hex forecast --preflight` answers the timestep question from the
  registry row alone, with no card and no file, the same way it answers the
  architecture question. A row whose declared timestep holds no anchor is
  refused at argument resolution instead of after the mesh bytes are read.
- The regional (limited-area) forecast runs on the card, and its anchor is
  earned. `hexcore.cuda_regional_forecast_v841` carries the device
  residency, the memory model and the stage sequencing that let the port's
  whole-step CUDA driver run `config_apply_lbcs=true` on a native-culled
  mesh; `mpas-port`'s driver gained a `regional_v841` hook mirroring its
  halo-exchanger hook, guarded at every one of its ten call sites so a
  whole-mesh run is bitwise untouched. Four independent processes ran three
  forecast hours on the 2,971-cell CONUS cull in 21.5 seconds of card time
  and produced masked-digest-identical history at all seven published frames
  while every whole-file digest differed. `ADMITTED_REGIONS` now holds one
  earned row, `conus-x1.2971`, naming L5's contract receipt and this
  forecast pair; every other regional configuration, including the larger x4
  cull of the same region, still refuses at the door by name.
- The device runs native MPAS's own garbage-element memory model. Every
  array carries one padded element per dimension with absent neighbours
  remapped to it, and the native pool value is restored into every garbage
  column after each launch — pad-compute-strip, held resident, because a
  device launch cannot skip its last element when the thread bound and the
  array stride are the same integer. The discipline is armed by an optional
  `KernelCache.post_launch` observer, so it reaches every entrypoint the step
  resolves without one shared CUDA translation unit changing by a byte. All
  twelve recorded division-by-garbage-geometry sites and the
  `divergence_damping_f32` sentinel early-out are retired without touching a
  kernel; two further blockers found by running it — a singular tridiagonal
  denominator at a zero reference state, and a recovered-state validator that
  demands positive density over its whole launch extent — are answered by a
  non-trapping reference pad and by running the identical test over the
  elements native solves.
- The v8.4.1 regional (limited-area) surface is a CUDA translation unit.
  `hexcore.cuda_regional_v841` carries 22 kernels — the lateral-boundary
  pool with its four derived coupled fields and its device-side time
  interpolation, the specified-zone tendency assignment, the relaxation-zone
  Rayleigh and Laplacian stages with their hardwired 50/10-dt coefficients,
  the u/ru specified-zone overwrite and the w hard-zero, the end-of-step
  `reset_speczone_values`, the scalar boundary adjust/set/clamp stages, the
  acoustic specified-zone pressure-gradient masking and implicit-solve skip,
  and the scalar-transport mask-4/5 edge downgrade with its specified-zone
  cell skip. Each kernel mirrors exactly one function of the v8.4.1 CPU
  authority, which is its expected-bits oracle, and the four native quirks
  are replicated at their sites citing that lane's anchors rather than
  re-derived. The kernels live in their own translation unit and under their
  own names so the global lane's sources stay byte-identical and every
  archived compile manifest, FTZ audit count and receipt that pins them
  stays valid. Proved on the 16 GiB proving card (RTX 5070 Ti, sm_120) by
  `tools/run_cuda_regional_contract.py` against the native-culled reference
  mesh: 8 of 8 contract decks bitwise identical over 10,443,332 float32
  values compared as raw bit patterns, 22 of 22 kernels covered with no
  kernel lacking a deck, dual-run stable both within a process and across
  two independent processes (84 of 84 payload digests identical), and every
  deck re-run with a deliberately wrong zone geometry FAILS, so each proof
  is shown to work in both directions. Evidence:
  `evidence/regional-cuda-l5-20260826/`.
- Regional CUDA execution is refused by name until a registered regional
  anchor exists (`hexcore.cuda_backend.regional_admission`), the ruling
  of 2026-08-25, mirroring the per-architecture earned-anchor pattern. An
  anchor is a row naming a contract receipt and a byte-identical forecast
  pair that exist in this repository; adding a region is table work. The
  registry is empty, so every regional configuration refuses, and the
  refusal names the breakage it prevents: a regional forecast that carries a
  receipt nobody could verify. The two CUDA host validations that already
  refused a culled mesh now refuse through that gate instead of declaring
  the lane closed/global, a premise the kernels above retired.

Fixed:
- The dycore's outer step and the frozen physics seam's step come from one
  source. They were two: `bind_mesh` rebound `DT_SECONDS` in the proof and
  forecast modules and in the GWDO guards, and the sealed Arwen constructor
  read that rebound value, but the dycore takes its outer step from
  `config.config_dt` and the configuration was built from its dataclass
  default. MEASURED (2026-08-26, the 32 GiB proving card (RTX 5090)): a mesh row declaring 100 s
  bound clean, allocated 18,820 MiB, spent 285 s and died inside composite
  step 0 with `post-RK candidate time must equal the exact step endpoint:
  120.0 != 100.0`. The forecast host now builds its configuration at the bound
  row's timestep and derives all four seam clocks from that configuration, so
  the two cannot diverge by construction; a coherence gate refuses a divergent
  pair on the host, before device memory is taken, quoting what it prevents.
  The forecast door's step count was a third clock reading the registry row
  while the run stepped at `config_dt`; all three now agree.
- A registered graded row said "Declared dt 90 s" while declaring 75 s. The
  value moved when the radiation-cadence rule landed and the sentence did not.
  The note now states the derivation: 95.84 s Courant limit, and 75 s is the
  largest value at or below it that also divides the 600 s radiation cadence
  exactly and closes the model clock in binary64.
- Regional CUDA kernels no longer divide by a source literal. MEASURED on
  the 16 GiB proving card: NVRTC rewrites `x / <float literal>` as `x * (1/<literal>)`, so
  `mpas_div(x, 5.0f)` returns a value one ulp from the correctly-rounded
  float32 quotient the CPU authority computes, while a runtime divisor and
  `__fdiv_rn` are exact. It cost four kernels their bitwise identity at
  once and the contract deck is what caught it. The hardwired `nRelaxZone`
  denominator is now a runtime argument, and a test greps the translation
  unit so the defect cannot return. The same hazard is measured and
  recorded at eight further sites in shared, frozen-source-pinned
  translation units (`mpas_div(..., 12.0f)`: two in `cuda_transport`, six in
  `cuda_driver`), where a third of float32 arguments take a different value
  than the CPU authority's division — a named cause for the released
  `transport_vertical_flux` differing from the CPU authority at 51,258 of
  166,376 values on the reference cull. Those eight sites are fixed in the
  next entry.
- The eight shared literal divisors are gone, and the rewrite is an
  architecture boundary rather than a property of one stack. MEASURED on the
  desktop RTX 3080 (sm_86, NVRTC 13.0.48 `CL-36260728`, CUDA driver 13030):
  one compiler, one option set, one source — NVRTC emits `div.rn.f32`
  against the literal for every target up to `compute_90` and `mul.rn.f32`
  by the literal's float32 reciprocal from `compute_100` up, which covers
  every card this port runs production work on. A differential compile of
  all fifteen CUDA translation units puts the census at ten rewritten
  instructions over eight source sites: `transport_vertical_flux` in
  `cuda_transport` (inherited by `cuda_transport_v841`), and
  `vertical_u_flux_f32`, `theta_vertical_flux_f32` and `w_vertical_flux_f32`
  in `cuda_driver`. The flux3/flux4 denominator is now the translation-unit
  constant `mpas_third_order_denominator`, which the host can write and the
  compiler therefore may not fold; `mpas_div` still carries the division, so
  its FTZ subnormal guard stays on the path. On targets below the boundary
  every payload digest is byte-unchanged, measured on sm_86 against the CPU
  authority on the reference regional bytes. Evidence:
  `evidence/nvrtc-reciprocal-20260826/`.
- The `transport_vertical_flux` divergence is explained in full, not
  partly. Reproduced independently on the RTX 3080 by running the shipped
  kernel and an instruction-level emulation of the higher target's
  arithmetic on the same reference bytes: the shipped kernel matches the CPU
  authority's `_atmosphere_vertical_flux` at all 166,376 values, and the
  rewrite alone moves exactly 51,258 of them — the whole of the recorded
  divergence, with nothing left over. The other three kernels move 157,710
  of 474,032, 51,639 of 154,492 and 51,318 of 154,492 interior values.
- Two files that supply CUDA bytes to pinned translation units are pinned
  themselves. `cuda_transport_v841` compiles `cuda_transport._CUDA_SOURCE`
  and every unit prepends `cuda_fp32.CUDA_FTZ_HELPERS`, so while those two
  were unpinned an edit to either changed a pinned unit's compiled bytes
  with every pinned digest still matching. This lane's own remedy landed in
  `cuda_transport.py` with the frozen-source proof reporting green, which is
  how the hole was found.

New:
- The v8.4.1 regional (limited-area) runtime runs in the CPU authority
  lane. `hexcore.regional_v841` transcribes the complete surface of
  `mpas_atm_boundaries.F` and the `atm_srk3` regional insertions: the
  7-ring masks and `specZoneMask` derivation, `nearestRelaxationCell`, the
  limited-area admission checks, the two-level LBC value/tendency pool with
  the four derived coupled fields (`lbc_rho_zz`, `lbc_ru`, `lbc_rho_edge`,
  `lbc_rtheta_m`) that `hexcore.lbc` deliberately left to the driver,
  `meshScalingRegional`, the spec/relax-zone tendency stages with their
  hardwired 50/10-dt Rayleigh and Laplacian coefficients, the acoustic
  specified-zone pressure-gradient masking and implicit-solve skip, the
  scalar-transport edge downgrade at masks 4-5, the u/ru specified-zone
  overwrite after recover, the w hard-zero, `bdy_adjust_scalars`,
  `bdy_set_scalars`, `reset_speczone_values`, the moist coefficients
  (`qtot`/`cqw`/`cqu`) and the unconditional end-of-step negative-scalar
  clamp of DO_PHYSICS builds. `config_apply_lbcs=True` is admitted only
  behind real, admitted LBC state — every other absence still refuses by
  name — and the two circular `transport.py` sentinel refusals now name a
  remedy that exists. Four native quirks are REPLICATED, not fixed, each
  documented where it is implemented and each a checked fact in
  `tests/test_regional_runtime.py`: the monotonic copy-back that admits
  only `bdyMaskCell <= nSpecZone` and so excludes relaxation rings 3-5; the
  Fortran operator precedence in the mask-4/5 edge condition, where
  `.and.` binds tighter than `.or.` and the mask-4 half therefore fires
  regardless of `config_apply_lbcs`; ring 1 never being nudged; and the
  `tend_rho` pool, which `atm_compute_dyn_tend_work` writes only at
  `rk_step == 1`, so the regional adjustments persist into RK stages 2 and
  3 and are applied again on top of themselves. Evidence:
  `evidence/regional-cpu-l4-20260825/`.

Fixed:
- `rvord` is the REAL(RKIND) quotient of the float32 constants, not the
  rounded float64 quotient. One ulp; it alone broke frame-0 `theta`
  bitwise identity against the compiled reference.

- One device-memory admission surface (`hexcore.device_admission`), and
  the `NATIVE_DEVICE_FLOOR` re-derived from measurement (ruled 2026-08-25).
  Every free-memory gate — the forecast door, `--preflight`, the mesh
  binding's per-mesh floor, the driver's `MIN_FREE_DEVICE_BYTES` and the
  restart-worker floor — now computes the same sum, the converged-pin
  measured row (4,339.1 MiB + 103,696 B/cell, the 170 SM #264 fit) plus one
  shared 512 MiB headroom, and the door forwards its resolved requirement
  into the driver argv (`--required-free-bytes`) so a card admitted on its
  own measured row cannot be refused downstream on the default model. The
  retired floor (an asserted 24 GiB scaled linearly per cell) refused
  meshes the measured row says fit and admitted x1.40962 below its measured
  peak; both retired breakages are test-pinned facts. Per-card rows at the
  converged stack measured the same day: RTX 5070 Ti two-point fit
  1,774.0 MiB + 115,143 B/cell (x1 6,272 / u96 8,802 MiB, u96 rc 0 under
  the new floor); RTX 3080 x1 6,340.5 MiB device-view, fixed 2,289.6 MiB
  with the slope borrowed and said so — x1.40962 is now admitted on the
  10 GiB desktop card, where the superseded row refused it. No 12 GiB card
  exists in the fleet: the 12 GiB tier figure ships as a DECLARED
  DERIVATION, labeled `DERIVED, NOT MEASURED`, and the label is
  test-pinned. Evidence: `evidence/l6-capacity-20260825/`.
- Regional (limited-area) meshes are admitted. `Mesh.validate()` recognises
  the `bdyMaskCell/Edge/Vertex` triple a native cull adds and validates the
  measured regional contract: absent-neighbour sentinels tolerated in
  exactly the five arrays a cull zeroes (`cellsOnCell`, `cellsOnEdge`,
  `edgesOnEdge` inside the unshrunk row, `cellsOnVertex`, `edgesOnVertex`)
  and only on ring-7 elements; reciprocity exempt only where the absent
  element makes it undefined; Euler characteristic 1 for a disk; edge and
  vertex masks equal to the minimum of their present cells' masks; neighbour
  masks within 1; ring populations growing outward; incidence identities
  corrected by exactly the sentinel counts. Every refusal names the concrete
  breakage. Both native culls of the regional reference mint load through
  `Mesh.from_netcdf` and pass `mesh-check`
  (`evidence/regional-admission-l1-20260825/`). The registry gains regional
  ROW fields — `boundary_zone_width`, `bdy_mask_sha256`
  (`regional_mask_digest`), and a nullable `lbc_source` — cross-examined at
  bind for every row before any constant moves: a regional cull on a global
  row, a digest or width mismatch, and an empty boundary-source slot are
  each refused by name (no boundary stream exists yet to force the zone).
  `mesh-check` prints a `regional` receipt block (zone width, per-ring cell
  counts, the pinned mask digest) and accepts `--grid-only` because a
  culled grid exists before its static does.
- `hexcore.lbc`: the lateral-boundary file reader and the two-level
  value/tendency pool, a transcription of `mpas_atm_boundaries.F` admission
  and timekeeping. `LbcInventory` applies the two stream rules by each
  file's own xtime — LATEST_BEFORE for the first admission,
  EARLIEST_STRICTLY_AFTER for every advance — and a missing interval refuses
  naming the rule, the model time and the timeline it searched. `LbcPool`
  holds the interval-end state and the float32 `(new - old) / dt` tendency,
  and `state_at` interpolates linearly backward from the interval end,
  `mpas_atm_get_bdy_state` verbatim. The reader pins the v8.4.1 lbc stream
  contract (seven float32 full-mesh fields on their measured dimensions) and
  refuses a missing, transposed or widened variable by name. Unit-tested on
  synthetic schema-correct files and on the three real native case-9 files
  of the 2026-08-25 regional oracle (`GPUWM_HEX_LBC_ORACLE_DIR`). Derived
  coupled fields (`lbc_rho_zz`, `lbc_ru`, `lbc_rho_edge`, `lbc_rtheta_m`)
  and driver wiring are deliberately out: they need mesh state and belong to
  the runtime lane; the pool refuses their names and says whose they are.
- Graded (variable-resolution) meshes are registrable and bindable. Four
- **A generated variable-resolution mesh completes a full-physics
  forecast** -- the first in this project. Registry row `v20.80.151649`
  carries 20 km resolution in its core inside an 80 km background at
  151,649 cells, and ran 6 h at dt 120 s on one RTX 5090: 180/180 steps,
  rc 0, finite at every step, 621.6 s wall, 19,226 MiB peak (0.57 % under
  the capacity model's prediction). Run twice, all seven history frames
  byte-identical. The uniform mesh at that resolution would be 1,472,535
  cells and fit no card this project owns; the graded mesh is 9.71x
  smaller and left 8.69 GiB of admission margin.
- Graded (variable-resolution) meshes are registrable and bindable. Five
  rows join the registry from the engine's new hierarchical-Goldberg
  generator, pinned by byte count and SHA-256 like every other row, each
  with its own dt admitted from the file's real `dcEdge` under the
  versioned Courant policy. A second row was produced by moving ONE
  coordinate in the spec JSON with zero code changes and binds through the
  real forecast door -- the arbitrary-acceptance test, passed on the door
  rather than on a diagram. Measured quality across the generated set:
  min `dvEdge/dcEdge` 0.0406-0.1023, 2.0x to 5.1x the admission floor, with
  zero edges under 0.04; the Fibonacci-seeded mesh this registry refuses
  reads 1.685e-4.
- `tools/probe_dv_floor_boundary.py` measures the real post-#311 dvEdge
  load boundary through the actual loader on the actual published pair
  (edited copies at 50/200/1,000/5,000 m dual edges), and
  `tools/audit_donor_padding.py` reports which padding convention a grid or
  static carries -- the #333 donor-readability surface, measured on four
  artifacts and handed over rather than silently changed.

Fixed:
- A mesh row whose dt the frozen v8.4.1 lane cannot step is refused AT BIND
  instead of inside composite step 0. Measured: a row at dt 100 s --
  Courant-admitted, dual-edge admitted, cadence-dividing -- bound clean,
  allocated 18,820 MiB, spent 285 s and died on `post-RK candidate time
  must equal the exact step endpoint: 120.0 != 100.0`, because the dycore
  takes its outer step from `config_dt`, which `V841MpasColumnPhysicsConfig`
  pins to exactly 120.0. The refusal names the frozen constant, the cost of
  not refusing, and the remedy. A companion guard refuses a dt that does not
  divide the 600 s radiation cadence, checked where the row is written.
- The Arwen seam pin leaves the pin-only lineage: `ARWEN_BUILD_COMMIT` moves
  to `26daaab7e`, the engine's seam-converge merge, where the refl10cm seam
  (`6e333822e`) folds into the release line (`613b681d3`). The sixteen-file
  manifest re-freezes on release-line bytes (seven digests move: physics,
  gf, noahmp_runtime, kernels/__init__, config, io/restart, docs/mpas-seam),
  the contract surface and adapter digests move with it, and the proof
  re-pins `cuda_arwen_physics_v841.py`. A checkout of the engine release
  line now verifies 16/16 with no recorded dirt, so the next public engine
  snapshot satisfies the port's pins as cut. No port behavior changes;
  the pinned engine bytes gain the release line's own seam-file evolution
  (per-PBL GF forcing wiring and the task-206/#310-era support files).
- The default history stream publishes `refl10cm` and `q2`, the two fields
  whose absence left four registered obs-referee metrics unscorable on the
  first real run. `refl10cm` is computed inside the due step's own WSM6 call
  from post-call temperature and the unchanged prepared pressure (WRF's
  `diagflag` arrangement, the point where native MPAS-A computes the field),
  carried through the transactional seam, and consumed exactly once per
  frame; `q2` is published bitwise with `q2_products_allowed = "true"` and
  its occasional native-parity negatives preserved. `rw_mpas_convert` maps
  both (`REFL_10CM`, `Q2`), so `rw_wrfbatch` reflectivity products read the
  model's own field instead of the renderer's hydrometeor fallback, and the
  model bundle producer derives `reflectivity_dbz` (column maximum, the
  MRMS-comparable composite) and `dewpoint_k` (the engine's
  dewpoint-from-mixing-ratio on Q2/PSFC, transcribed exactly). The Arwen
  seam pin moves to `6e333822e` (the refl-capable seam on
  `pin/mpas-port-arwen-seam-v2`); snapshot schema v3.

Fixed:
- Stale-guard sweep, hex side (2026-08-25, audit #347 findings 5-10 +
  unknowns). The 2-GPU partition scheduler's floors route through
  `hexcore.device_admission` (the retired 22 GiB linear shape and the
  20 GiB `require_devices` default are gone; the per-partition
  application of the measured row is labeled `DERIVED, NOT MEASURED`
  pending a 2-GPU #264 run). `reservation_probe`'s self-validation
  control is re-pinned from the dead pre-#294 `gf_gfdrv_stage` 7,034 MiB
  premise to the post-cut widest frame (`wsm6_column`, 7,216 B) with a
  device-derived bound, and `module_image_probe` gains the registry
  route so the widest-frame module cannot escape the ledger.
  `copy_elision_accounting`'s "of record" arm is the converged row,
  test-pinned to `FLOOR_DERIVATION`. The FTZ guard-cost timing ceiling
  is per-architecture beside the arch-admission registry (sm_120 keeps
  1.25; sm_86 gets 1.75 from its recorded 1.47-1.57x deviation;
  unregistered architectures refuse by name). Bigcard refusal/marker
  strings compute from `X4_FULL_PHYSICS_BYTES` instead of restating the
  retired 26.4 GiB; README and manual chapter 1 stop asserting the
  superseded row and the un-re-derived floor. The dry-runner 16 GiB
  floor and the x1.163842 nominalMinDc dt rule are adjudicated frozen
  closed-case records with the determination written at the constants;
  STATE.md's declared engine floor reads the enforced 2.5.5.
- `forecast_door.FOOTPRINT_MODEL` no longer quotes the superseded
  6,296.5 MiB + 93,474 B/cell row (pin `0d04db712`): it is the of-record
  converged row, and `tests/test_device_admission.py` re-fits the raw
  evidence ledgers and pins the shipped coefficients to them, so a
  constant drifting from the evidence it cites fails by name (#340).
- The registered v15.150.38857 static is rebuilt on the unified 82-variable
  `rw_mpas_static` writer and re-registered (#330). The retired writer's
  drag band sampled terrain 180 degrees of longitude from every cell (the
  archive-origin assumption): measured corr(old, new) for var2d is +0.003
  at the same cell and +0.697 at lon+180, and the field-by-field compare
  shows oa/ol moving full scale on two thirds of cells. The rebuilt static
  matches a native init_atmosphere static for the same mesh at var2d
  corr +0.9999, oa1 +0.9961, land-only con +0.9928, and adds the operator
  tables and soil-composition group the retired writer omitted. The x4 and
  x1.40962 rows were measured to be native-built statics (v8.4.1 and the
  published v8.2.0 artifact) that never carried the band; each registry row
  now names its builder.
- A lake column (MODIS category 21) is folded to open water at the forecast
  loader boundary, the same conversion WRF applies without a lake model.
  The arwen vegetation tables end at category 20, so before the fold any
  generated-mesh run with lakes died with an IndexError inside the Noah-MP
  cold start; the native x4 landuse never exceeds 19, which is why the
  proof path never saw it. The fold count is in every run receipt; on the
  x4 case the mask is empty and every array passes through untouched.
- The GWDO dt guard follows the mesh binding: a registered mesh runs the
  YSU-GWDO kernel at its own Courant-admitted timestep instead of dying
  at step 0 on "requires dt_seconds=120". The kernel takes dt as a runtime
  argument; on the frozen native mesh nothing is rebound and the guard
  still demands exactly 120 s.
- The x4 proof's restart leg is bit-identical again. GF's advective
  forcing pair (rthdynten/rqvdynten) is per-step carried state: each
  step's dynamics forms it and the next step's physics consumes it, and
  it lives outside both the MPAS atmosphere and the Arwen backend
  restart payload. The F030 checkpoint never captured it, so every
  restored run re-entered step 16 with zero forcing lanes while the
  unbroken run fed the real step-15 pair, and the step-16 identity gate
  failed deterministically on every arm (#327, 5/5 red on the reference node, red
  since the forcing lanes landed). Checkpoint schema v3 downloads the
  pair at F030, refuses to write a checkpoint without it, re-seeds it on
  restore in both the fresh-process worker and the in-process
  instrument, and gates the rehydration with its own fingerprint
  identity. A pre-v3 checkpoint is refused by name instead of resuming
  wrong.

## 0.1.1 (2026-08-25)

The forecast becomes a front door, and the referee runs.

New:
- gpuwm-hex forecast: a front door that binds the mesh, asks the card
  first against a measured per-card row, refuses by name with numbers,
  and prints the render command when it passes. --preflight gives the
  same answer without spending anything.
- The obs referee ships with its first scorecard: canonical model bundles
  gain a producer, and four metrics that could not score now score.
- The default history stream publishes refl10cm and q2, and
  rw_mpas_convert maps both, so reflectivity products read the model's
  own field instead of the renderer's hydrometeor fallback.
- A generated mesh completes a forecast end to end. A mesh whose Voronoi
  edges collapse is refused at bind, by name, before anything expensive.
- The engine seam pin moves to the gpuwm release line and verifies 16/16
  clean, so an engine checkout at the pinned commit satisfies the pins
  as cut.
- A per-allocation device-memory ledger. Measured 2026-08-24 on an RTX
  5070 Ti: x1.40962 peaks at 5,604.0 MiB with the engine's device-sized
  radiation chunks.

Fixed:
- The registered v15 static is rebuilt on the unified 82-variable writer;
  the retired writer's drag band sampled terrain 180 degrees of longitude
  away. Every registry row now names its builder.
- A lake landuse column folds to open water at the forecast loader
  boundary; a generated mesh with lakes no longer dies in the Noah-MP
  cold start.
- The GWDO dt guard follows the mesh binding; a registered mesh runs at
  its own admitted timestep.
- Restart checkpoints carry GF's advective forcing pair (schema v3);
  restored runs are bit-identical again, and a pre-v3 checkpoint is
  refused by name.
- The forecast door leaves output creation to the driver; an admitted run
  no longer fails on a directory that already exists.

Requires gpuwm 2.5.5 or newer for the seam bytes and the bundled engine
binaries.
