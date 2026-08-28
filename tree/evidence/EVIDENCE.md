# What an `evidence/...` reference means

Almost every claim in this project's documentation ends with a path like
`evidence/memory-shape-20260827/` or `evidence/restart-step16-327/`. This
file says what those paths are, where they live, and what to do when the one
you want is not beside you.

## The short version

An `evidence/<name>/` path is the **identity of a measurement**, not a
promise that the file is in your hands.

Each one names a receipt: the commands that were run, the machine and the
card they ran on, the exact engine and mesh bytes they ran against, and the
numbers that came out. The name is stable, so a claim made in `README.md`
today and a claim made in a refusal message six months from now can point at
the same measurement and mean it.

## Where they live, and where they do not

| surface | carries receipts? |
|---|---|
| the source repository | **yes** — under `evidence/`, one directory per measurement |
| the sdist (`gpuwm-hex-*.tar.gz`) | **no**, except this file |
| the wheel | **no** |

The distributions carry the *contracts* — `docs/`, `tests/`,
`verification/`, `tools/` — because those are things you run. They do not
carry the receipts, because those are records of runs already made, on
hardware you do not have, and they are large: the receipt set is roughly
89 MB against a 1.9 MB sdist. Shipping them would multiply the download by
forty-something to deliver files that no test reads and no door opens.

So if you installed from PyPI and a document points you at
`evidence/something/`, that is not a missing file. It is a citation. Read
the repository for the receipt itself.

## Reading a receipt

Most directories carry a `RECEIPT.md` that leads with what was measured and
what the limits of the measurement are. Beside it you will usually find the
raw artefacts the receipt quotes: JSON digests, driver logs, the shell
script that drove the run, and any figures.

Two conventions worth knowing before you read one:

- **Numbers carry a tense.** A receipt records what was true on its own
  date, at its own engine pin, on its own card. Later receipts supersede
  earlier ones without editing them, because a record that gets rewritten
  is not a record. When two receipts disagree, the later date and the
  `docs/` surface that quotes it are the current answer.
- **A named limit is part of the result.** Receipts in this tree state what
  they did *not* separate — which arm was not run, which card was not
  available, which candidate the evidence does not distinguish. Those
  sentences are load-bearing and are not hedging.

## The contracts a receipt is written against

The measurements are not free-floating. What they check against ships with
the distribution:

- `verification/manifests/` and `verification/vertical-specs/` — the schemas
  and vertical-level contracts the init and obs-referee legs are graded on.
- `tools/` — the drivers themselves, including the proof harness that
  verifies its own executing modules by SHA-256 before it runs.
- `docs/source-matrix.md` — the per-source verdict table, which reproduces
  every verdict verbatim so the table is readable without the receipts
  behind it.

## Asking for one

If you need a specific receipt and cannot reach the repository, open an
issue naming the exact `evidence/<name>/` path you want and the claim you
are checking. Naming the claim matters: it is usually faster to point you
at the measurement that actually settles your question than at the one the
document happened to cite.
