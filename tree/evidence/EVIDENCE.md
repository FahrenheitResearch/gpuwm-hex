# Evidence

Every measurement quoted anywhere in this repository — in `README.md`, in
`STATE.md`, in `docs/`, in `CHANGELOG.md` — names the receipt directory it was
written to. Those names look like paths under this directory:

```
evidence/gf-pin-move-measured-20260824/
evidence/native-free-proof-20260824/
evidence/restart-step16-327/
evidence/local-timestep/
```

**The receipt files themselves are not carried in this repository.** Treat an
`evidence/...` reference as the *identity* of a measurement — the thing you
quote when you ask about it — rather than as a file you can open here.

## Why they are not here

A receipt is a record of a run on specific hardware. It carries the absolute
paths the run used, the device it ran on, the scratch directories it wrote
through and the machine it was launched from. That is what makes a receipt
worth having and it is also what makes it the wrong thing to publish: it
describes a private machine, not this software.

Rewriting them to remove that would leave documents that are no longer the
bytes the run produced, which is worse than not shipping them — an
un-reproducible receipt that *looks* reproducible is a claim with nothing
behind it.

## What is here instead

The contracts the receipts are written against **do** ship, because they are
specifications rather than records:

- `../verification/schemas/` — the JSON schemas a receipt must satisfy
  (`normalized-artifact-receipt-v1`, `gf-subsidence-treatment-v1`)
- `../verification/manifests/` — the obs-referee production manifest
- `../verification/vertical-specs/` — the vertical-coordinate specifications
  the init door consumes
- `../verification/README.md`, `CANONICAL-BUNDLE-CONTRACT.md`,
  `GF-HOOK-CONTRACT.md` — what a conforming receipt has to contain

So the *shape* of every measurement is public and checkable, and the harnesses
that produce them are in `../tools/`. What is withheld is one campaign's
particular output files.

## Reproducing rather than reading

Most of what the receipts record is reproducible from this tree. `../README.md`
and `../docs/manual/` state the hardware each tier needs, and
`../tools/battery/` lists the three tiers by what a machine has to own. The
tiers that need a large CUDA device and the byte-pinned mesh/static/init
authority files (about 6.9 GiB, no fetch path) cannot be reproduced without
that hardware and those assets — `../../STATE.md` §9 says so plainly rather
than implying they are green.

## Asking for one

If a specific measurement matters to you — for a review, a comparison, or a
reproduction attempt — open an issue naming the receipt directory and what you
need out of it. Individual figures can be provided with their provenance
stated.
