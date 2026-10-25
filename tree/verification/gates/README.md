# Public gate index

The decks, ratchet and sha256 pins of the G1 (cached dry dycore and suite-none
physics) and G2 (landed suites and options) gates, with every data path made
relative.  Same deck ids, same pins and same ratchet entries as the
maintainers' gate; only the locations moved.

- `index.json`: gate -> decks (`mpas-gate-index-v1`).
- `decks/`: one verification deck per gate entry.  Pinned inputs are
  `../data/<path>`; data paths in the model command are `{deck_dir}/../data/<path>`.
  The MPAS route's own resources are not data: the native framework library,
  the Registry pool reader, the pinned MPAS-Model tree, the strict Thompson
  tables and the hex-cuda packet are named `{resource:KEY}` and come from
  `gpuwm-hex physics-resources fetch` (the native-referee decks also need the
  site build `physics-resources build --pin-set v8.4.2 --build-native-library`;
  docs/physics-resources.md).
- `ratchet.json`: the recorded-missing inventory each gate is graded against.
- `data-manifest.json`: every data file a deck pins, with sha256, size and how
  to rebuild or fetch it.  None of it is redistributed here.
- `data/`: not in the distribution.  Rebuild the references with
  `tools/native_dry_reference` and `tools/native_physics_reference`, fetch the
  MPAS-Data tables at the pinned commit, and lay the files out as the manifest
  lists (any layout of links works: only sha256 is checked).

Run:

    tools/mpas_gate.sh --gates G1,G2 --index verification/gates/index.json \
        --ratchet verification/gates/ratchet.json --out /tmp/gate

One deck, from a pip install. The headline deck runs every active slot on the
GPU through the ported framework: it needs `gpuwm-hex physics-resources fetch`
and nothing built at the site (no native library, no MPAS-Model tree):

    gpuwm-hex physics-resources fetch
    gpuwm-hex verify-mpas --deck verification/gates/decks/mpas-headline-convection-permitting.json \
        --ratchet verification/gates/ratchet.json --receipt hl-3.json --work-root hl-3-work

The comparator is the installed gpuwm `rw_mpas_static` bridge's `verify` tool
(`--comparator` names another build). No deck needs an environment record:
verify-mpas captures the environment before and after the run and writes it
into the receipt.

Check the layout first, without a card:

    python tools/check_gate_index.py verification/gates --data

Each deck fixes its process count (proof-bar ruling 2), and the claim each
gate makes is the scope stated in `index.json`.
