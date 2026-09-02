# gpuwm-hex User Manual

gpuwm-hex runs a global, variable-resolution atmospheric model on one consumer
CUDA GPU. The numerics are a port of the MPAS-Atmosphere v8.4.1 dynamical core;
the physics is the ArWen engine's suite. This manual is for users at every
level of experience: it starts from "what is an atmospheric model" and ends at
the receipts behind every measured number.

gpuwm-hex is a research and educational tool. It is never a substitute for
official forecasts and warnings from your national meteorological service. Do
not use it to make safety decisions.

*(MPAS-Atmosphere is a registered project of NCAR and LANL. gpuwm-hex is an
independent port and is not affiliated with or endorsed by them.)*

## Chapters

1. [What gpuwm-hex is](01-what-this-is.md) — plain-language introduction: what
   it does, what it cannot do, and what hardware it needs.
2. [Quickstart](02-quickstart.md) — install, `gpuwm-hex doctor`, closing every
   gap doctor names, obtaining assets, first initial conditions, first
   forecast, first rendered products.
3. [Concepts](03-concepts.md) — mesh, static, init; the doors; why the
   dynamical core is pinned and the physics is judged by observations; what a
   refusal is and why every refusal names its remedy.
4. [Meshes and static fields](04-meshes-and-static.md) — validate a mesh pair,
   generate a new mesh and build its static, capacity planning with the
   measured footprint model.
5. [Initial conditions](05-initial-conditions.md) — the init door: the
   native-free vertical path, the capsule compatibility mode, and every
   physics switch with its native namelist key.
6. [Running a forecast](06-forecast.md) — the source-checkout requirement,
   registered meshes and timestep admission, local time stepping and its
   measured cost, restart, and what a run receipt claims and does not claim.
7. [Rendering products](07-render.md) — history files to product PNGs through
   the Rust renderer, the delivered layout, and the render manifest.
8. [Troubleshooting](08-troubleshooting.md) — the most common named refusals,
   what each means, and the remedy for each.
9. [Reference](09-reference.md) — doors and flags, environment variables, the
   mesh registry, the test battery, and where receipts live.

## Conventions

**Every command was run before it was printed.** Each command block in this
manual was executed against the real artifact it documents — the installed
0.2.1 wheel for the doors, the repository tools for the forecast lane, a CUDA
node for anything that needs a card — before it was written down. Where a
command's output is shown, it is the output that run produced, with machine
home directories shortened to `~`.

**What a walk of this manual actually reaches.** On 2026-08-27 the printed
commands were executed in printed order, from an empty environment, against
the built 0.2.1 wheel and the *published* engine, by somebody following the
page rather than remembering the code. Install, `doctor`, `mesh-check`, the
init door, the forecast door and the render door all reach real output —
16 min 46 s from an empty machine to the first rendered products. Three
things did not, and all three were against the engine published that day,
gpuwm 2.5.7. Two are closed by the 2.5.8 floor this distribution now declares,
re-measured 2026-08-28 against the published bundle: the **limited-area lane**
of chapter 6.8 and the `cull` door of chapter 9 have the engine capabilities
they need, and the render door's **default window** is understood by the
published converter. The third stands, with a narrower reason than it had: the forecast lane needs
a `gpuwm` **git** checkout at **v2.6.4**, not just whatever pip installs — not
because a pinned file is missing from the wheel (at 2.5.8 all sixteen resolve
from `site-packages`, measured 2026-08-28) but because the run records the
checkout's HEAD, tree and dirty paths into every receipt. Each is written
where you meet it; the 2026-08-27 walk is in
`evidence/userwalk-20260827/RECEIPT.md` and is a record of 2.5.7.

**Citations.** Bare paths (`docs/init-door.md`, `tools/battery/README.md`) are
repository-relative files. Paths under `evidence/` are measurement receipts
committed beside the code they measure. A claim with no bracket is a
definition or a description of behavior, not a measurement.

**Declared divergences.** gpuwm-hex differs from native MPAS-Atmosphere in
three measured, physics-shaped ways. They are stated in chapter 3, quantified
in [`docs/declared-divergences.md`](../declared-divergences.md), and never
hidden behind an option. Read them before trusting a number.

**Version.** This manual describes gpuwm-hex 0.2.3 with the gpuwm 2.6.x
engine line. Where a measurement was taken on specific hardware, the card is
named beside the number.

**Vocabulary.** The *distribution* (what you `pip install`) and the *command*
(what you type) are both `gpuwm-hex`. The *import namespace* is `hexcore`,
renamed from `mpas_port` at 0.2.0 and explained in the README. A *door* is a command a
user can walk through; a *lane* is a capability that today requires a source
checkout. The *engine* is the `gpuwm` distribution that supplies the physics
seam and the Rust binaries.
