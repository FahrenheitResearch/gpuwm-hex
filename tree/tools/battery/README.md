# The gpuwm-hex test battery

Three tiers. The split is not by speed, it is by **what a machine has to own**
to run a tier honestly, and each tier that cannot run says which thing is
missing rather than dying somewhere unrecognisable.

The lists in this directory are the source of truth for what a tier covers.
CI reads them; a queue script reads them; nobody retypes a file list into a
workflow, because a retyped list drifts and the drift is invisible until a
gate that was supposed to run has not been running for a month.

---

## Tier 1 — `cpu_files.txt` — runs anywhere

```sh
PYTHONPATH=src python -m pytest -q -m "not gpu and not bigcard and not assets" \
    $(grep -v '^#' tools/battery/cpu_files.txt)
```

Needs nothing but Python and the declared dependencies. This is the tier a
GitHub-hosted runner runs on every push and pull request, and it is the tier
that catches what a public cut can actually break: the packaging declaration,
the front doors' argument surfaces and refusals, the proof-guard ordering and
pin structure, the multi-GPU partition executor's pure logic.

It deliberately does not import CuPy. Any test whose module or function
reaches CuPy is auto-marked `gpu` by `tests/conftest.py` — by AST, not by
trusting an author to remember a decorator — so it cannot leak into this tier
by omission.

## Tier 2 — `asset_gates.txt` — needs the byte-pinned authority files

```sh
PYTHONPATH=src python -m pytest -q -m assets $(grep -v '^#' tools/battery/asset_gates.txt)
```

About 6.9 GiB of mesh, static, init and native-history files, pinned by byte
count and SHA-256 (masked over the netCDF `file_id` nonce for the history
files, which is random per write). **They ship with no fetch path.** They live
relative to the checkout root, on whichever machine holds them, under
`work/v841-vr-static/` and
`work/v841-full-physics-gf-gwdo-native-authority-20260820a/`.

Without them these tests skip with the count and the first missing path
named, because the failure they used to produce — `FileNotFoundError` on a
path inside the checkout that has never existed on this machine — reads as a
broken checkout rather than a missing asset.

## Tier 3 — `gpu_gates.txt` — needs a big card

```sh
PYTHONPATH=src python -m pytest -q -m bigcard        # the capacity preflight
# then the harnesses named in gpu_gates.txt
```

The x4.163842 full-physics tier holds **about 26.4 GiB resident**, so it needs
a device with 32 GiB. On a smaller card it does not refuse a check — it dies
inside a CuPy allocation part-way through a run, after burning the time it
took to get there. `tests/test_device_capacity.py` is the cheap version of
that discovery and runs first.

**This tier is not in GitHub Actions and will not be.** GitHub-hosted runners
have no CUDA device, and a tier that is silently skipped in CI is worse than a
tier that is declared external: the first reports green while proving nothing.
The reason is named here, in `gpu_gates.txt`, and in the CI workflow itself.

Set `GPUWM_HEX_NO_LOCAL_GPU=1` (or `GPUWM_NO_LOCAL_GPU=1`, honoured so a box
configured for the engine behaves the same) to ban device contact outright:
the gates skip by name and `CUDA_VISIBLE_DEVICES` is set to `-1` so that a
transitive allocation the markers cannot see still finds no device.
