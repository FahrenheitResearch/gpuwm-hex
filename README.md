# gpuwm-hex

A GPU-native global variable-resolution atmospheric model core: the dynamical
core of **MPAS-Atmosphere v8.4.1** ported to CUDA, running **ArWen's physics**
rather than MPAS's, on one consumer card.

This is the repository. The distribution it builds lives in [`tree/`](tree/)
and its user-facing documentation is [`tree/README.md`](tree/README.md), which
carries the install matrix, the front doors, the measured limitations and the
assets a user must supply. This page is the repository-level orientation: what
this project is, what it is pinned to, and what it is judged by.

**Install:** `pip install gpuwm-hex`. It resolves only once `gpuwm>=2.5.3` is
on PyPI — that is a hard dependency floor, not a soft one, and the reason is
in [`tree/pyproject.toml`](tree/pyproject.toml) where it is declared. Until
then, install from a checkout. `tree/README.md` carries the full matrix.

---

## The two relationships that define this project

### To `gpuwm` (ArWen) — a one-way dependency across two declared seams

The port owns no physics. Every physics column in the forecast lane runs
through ArWen's column-batch seam, so gpuwm-hex and `gpuwm` share one physics
implementation instead of two that drift.

The dependency is strictly one-directional, and that is measured, not assumed:
`mpas_port` imports `gpuwm` at 5 sites under `tree/src/` and 14 under
`tree/tools/`; `gpuwm` imports `mpas_port` at **zero** sites. There is no cycle,
which is why the two can be separate repositories at all.

Two seams cross the boundary, and neither is a source dependency:

| seam | what crosses | how it refuses |
| --- | --- | --- |
| **Python** | the `gpuwm` distribution, `>=2.5.3`, plus a `gpuwm` source checkout at the pinned commit for the forecast lane | 16 source files pinned by SHA-256; a moved byte refuses by name before a device is touched |
| **Rust binaries** | `rw_mpas_init`, `rw_mpas_convert`, `rw_mpas_mesh`, `rw_mpas_static`, built in `gpuwm`'s `tools/rustwx` workspace | each carries an ABI marker and a `GPUWM_BRIDGE_SOURCE_REV` stamp, so a stale binary is detectable rather than silently wrong |

Mesh generation and the `gpuwm mesh` door stay in `gpuwm`. The `rw-mpas` crate
depends by path on `rw-store`, `static-fields` and `rustwx-core` — 776
first-party files across the closure — and `static-fields` is `gpuwm`'s own WPS
geography stack. Vendoring that closure here would fork it into two repos;
moving it would invert three path dependencies across a repository boundary and
un-ship a `gpuwm` front door. gpuwm-hex consumes the four binaries through the
environment-variable ladder its doors already have.

### To native MPAS-Atmosphere v8.4.1 — pinned in the dycore, divergent in the physics

**The dycore stays pinned.** Byte-identity of the dynamical core against native
MPAS-A v8.4.1 is kept, and it is the only correctness anchor this project has.
It is achievable, it is the thing that was actually ported, and without it there
is no anchor at all. See *The pin* below.

**The physics goes its own way.** The port already runs ArWen's physics, not
MPAS's, so whole-model parity against MPAS stopped being reachable the moment
that choice was made — every remaining "correctness defect" against native is
physics-shaped for exactly that reason. Physics parity is therefore **retired as
a goal**, and the known differences are **declared divergences, not blockers**:

- the Grell-Freitas cumulus scheme is a different generation (2018 formulation
  here, the 2013 ensemble fork in native v8.4.1);
- an upper-band potential-temperature drift, `+0.019 K/h` above level 45,
  reaching `+0.46 K` at 24 h;
- a convective rainfall deficit, `rainc` -36 % / -34 % across two cases;
- a downstream condensate surplus, `+50 %` cloud water and `+62 %` rain water in
  the domain mean at 24 h.

All four are quantified in `tree/README.md`, registered with mechanism,
magnitude and the named observational referee for each in
[`tree/docs/declared-divergences.md`](tree/docs/declared-divergences.md), and
reproduced in `NOTICE` as part of the derivative marking BSD-3-Clause
requires.

**The verification of record for physics is obs-skill** — MRMS and ASOS — not
agreement with MPAS. Matching another model's choices was never evidence of
being right.

**This is not licence to ignore a defect.** A bias that is wrong against
*observations* is still wrong. The upper-band theta drift does not become
acceptable because MPAS is no longer the referee; it changes referee. The
obs-skill verification that would settle it has **not been run** — it is open
work, tracked in [`STATE.md`](STATE.md).

---

## The pin

`tree/src/mpas_port/cuda_arwen_physics_v841.py` pins sixteen `gpuwm` source
files by SHA-256, and `tree/tools/run_cuda_v841_full_physics_x4.py` names the
`gpuwm` commit they were taken from:

```
ARWEN_COMMIT = 0d04db71298d010a61fee3267c07277da3b8b64f
```

That commit is the seam pin `629ddb6f0` — the annotated tag
`pin/mpas-port-arwen-seam` — carrying exactly one cherry-picked `gpuwm` commit:
the Grell-Freitas local-memory frame cut. Two of the sixteen files move with it,
`gpuwm/core/gf.py` and `gpuwm/core/kernels/gf.cu`; the other fourteen, including
the whole seam contract surface, are byte-identical at both commits.

All sixteen hash-match at that commit, and at no other ref checked. Measured
2026-08-23 against the `gpuwm` object store:

| ref | files matching |
| --- | --- |
| `0d04db712` (this pin) | **16 / 16** |
| annotated tag `pin/mpas-port-arwen-seam` (`629ddb6f0`) | 14 / 16 |
| `gpuwm`'s 2.5.0 release line | 10 / 16 |
| the earlier port-landing ref | 9 / 16 |

The anchor is the *reachability of that object*. A commit that survives only as
the tip of one `gpuwm` lane branch is one branch deletion away from being
unreachable — at which point the sixteen pins could never be verified again and
every proof built on them would become unfalsifiable. `629ddb6f0` is safe:
`gpuwm` carries the annotated tag `pin/mpas-port-arwen-seam` on it, a local tag
and explicitly not a release tag.

**`0d04db712` has one too**: the annotated tag `pin/mpas-port-arwen-seam-v2`,
placed 2026-08-23, whose message names exactly this obligation — deleting the
lane branch must not orphan the sixteen pinned objects. Both pins the port has
ever executed are therefore tag-anchored in `gpuwm`.

### Verifying it

`tests/test_proof_guard_pins.py` verifies the manifest against a real `gpuwm`
object store. Point `$GPUWM_OBJECT_STORE` at a `gpuwm` checkout that holds the
pinned commit; with the variable unset the guard has nowhere to look and says
so.

```sh
cd tree
PYTHONPATH=src python -m pytest tests/test_proof_guard_pins.py tests/test_proof_guard_ordering.py -q
```

The two outcomes are deliberately different. A box that **has** a `gpuwm` store
which no longer holds the commit **refuses by name** — that is the failure the
guard exists for. A box with **no** store at all (a CI runner, a fresh clone on
another machine) skips with a message saying which anchor went unverified, so
the skip cannot be read as a pass.

### The two legs that need hardware

The pin has two further legs that no repository can carry:

- **The native authority** — about 6.9 GiB of byte-pinned grid, static, init and
  native-history files, pinned by masked SHA-256 (the random `file_id` netCDF
  attribute is masked out so a bit-exact rerun satisfies the pin while one
  flipped data byte refuses). They ship with no fetch path and live on the
  dedicated nodes.
- **The compiled-endpoint fixture** — `tools/compare_v841_compiled_endpoint.py`
  wants `tree/oracle/jw-x1.2562-v8.4.1-split3-endpoint-nonclaim`, six files
  pinned by SHA-256. That directory is not in this checkout.

Both are stated in [`STATE.md`](STATE.md) as unverified-here rather than
implied to be green.

---

## Layout

```
LICENSE          Apache-2.0
NOTICE           MPAS-Atmosphere BSD-3-Clause verbatim + the derivative marking
README.md        this page
STATE.md         open items, carried forward with their measurements
.github/         ci.yml, the battery; publish.yml, release-triggered only
tree/            the gpuwm-hex distribution
  src/mpas_port/   the port itself (import namespace is still mpas_port)
  tools/           harnesses, proof guards, comparators
  tests/           the three-tier battery
  verification/    schemas, manifests and vertical specs the docs invoke
  evidence/        see EVIDENCE.md
  docs/            the manual, the door pages, the declared divergences
```

Every measurement quoted in this repository names the receipt it came from.
The receipt files themselves are not carried here — `tree/evidence/EVIDENCE.md`
says why and what to ask for.

## Running the battery

Three tiers, split by **what a machine has to own**, each naming what it is
missing rather than dying somewhere unrecognisable. Tier 1 needs nothing but
Python:

```sh
cd tree
PYTHONPATH=src python -m pytest -q -m "not gpu and not bigcard and not assets" \
    $(grep -v '^#' tools/battery/cpu_files.txt | grep -v '^$')
```

`tools/battery/README.md` covers all three. `GPUWM_HEX_NO_LOCAL_GPU=1` bans
device contact outright.

## Licence

Apache-2.0, the same licence `gpuwm` ships under. The upstream it derives from,
MPAS-Atmosphere v8.4.1 (LANL/UCAR), is BSD-3-Clause; those terms govern the
MPAS-derived portions and travel with every copy. `NOTICE` reproduces that
licence in full and carries the marking BSD-3-Clause separately requires of a
derivative work. Both files ship inside the wheel and the sdist, and the built
wheel declares `License-Expression: Apache-2.0` with both under
`dist-info/licenses/`.

This is **not** the version available from LANS and UCAR, and neither they nor
their contributors endorse it. Results from gpuwm-hex are not results from
MPAS-Atmosphere.
