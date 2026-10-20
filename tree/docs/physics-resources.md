# MPAS physics resources

The MPAS-A v8.4.2 framework route uses the Registry metadata, device maths,
Rust pool reader and table decoders carried by the wheel. It loads no native
MPAS library and needs no C or Fortran compiler. Thompson tables and auxiliary
arrays come from six DATA ZIPs pinned in
`hexcore/data/physics-resource-assets.json`. The release tag is
`physics-resources-v0.4.1`. Every archive and member has size and SHA-256 pins.
Release assets contain no executable, native library or Fortran runtime.

The 16 MPAS-Data files are separate upstream fetches pinned to commit
`c57dbc7be629802c6e848770a9e44b9bc602be41` and SHA-256 per file. They are
never bundled in a release ZIP. The Thompson DATA is generated from MPAS-A
v8.4.2 BSD-3-Clause code and data. The asset release carries
`licenses/LICENSE-MPAS-BSD-3-Clause.txt` beside the ZIPs.

```sh
gpuwm-hex physics-resources fetch
gpuwm-hex physics-resources fetch --from /path/to/offline-mirror
```

An offline mirror holds the six ZIPs and the 16 upstream files by their
manifest basenames. Set `GPUWM_HEX_PHYSICS_RESOURCES_FROM=DIR` for first use.
`GPUWM_HEX_PHYSICS_RESOURCES_URL=BASE` selects an HTTP mirror containing both
kinds of file, using the same packaged pins. `--jobs N` selects 2 to 64
parallel download and extraction workers. Fetch assembles and verifies the
CUDA packet with the installed wheel's helpers. It invokes no source builder.
The DATA is identical for fast and strict numerics.

The cache is `$GPUWM_HEX_CACHE_ROOT/physics-resources`, or
`~/.cache/gpuwm-hex/physics-resources`. `installed.json` records the resource
paths and hashes. Compressed download copies are removed after successful
installation. The offline mirror remains available. Expanded resources and
packet paths are portable when the whole cache moves. Set
`GPUWM_HEX_PHYSICS_RESOURCES_FIRST_USE=off` to require staging in advance.
An explicit forecast resource option takes precedence over the cache.

## RRFS-MPAS and native referee tools

RRFS-MPAS has no native-free framework port yet. Its native library is not
distributed because it needs a Fortran runtime. A public run without a
site-built library refuses before resource fetching or pool hydration.

Sites can explicitly build native referee resources from pinned upstream
sources. This requires `gcc`, `g++`, `gfortran`, `make`, `cpp`, and the
netCDF-Fortran development files. The source builder also prepares its pinned
MPICH and PnetCDF dependencies. These libraries stay in the site's cache.

```sh
gpuwm-hex physics-resources build --pin-set rrfs
gpuwm-hex physics-resources build --pin-set v8.4.2
```

The first command supplies a local RRFS framework for a later run with that
cache, or with explicit `--native-physics-library` and `--mpas-source` paths.
The second supplies native MPAS-A referee resources. Neither command is part
of the default MPAS-A framework route. `--no-native-library` skips native
library construction. `--native-library PATH` adopts an existing MPAS-A
build with its ABI and build receipts. `--mpich PREFIX` selects a local MPI
installation, and `--jobs N` controls the source build workers.

Native source builds stub Noah-MP v5.0 by default. The explicit
`--licence-opt-in ucar-noahmp` option builds that upstream code under its own
terms. The public distribution carries no Noah-MP v5.0 implementation.

Verification decks may name explicitly built referee resources by portable
keys such as `{resource:native-library}` and `{resource:mpas-source}`.
The default framework route binds only `{resource:pool-reader}` and
`{resource:physics-resources}`.

`gpuwm-hex doctor --explain` reports the installed MPAS DATA, packet and pool
reader separately. It verifies installed hashes without fetching or building.
An empty cache shows `gpuwm-hex physics-resources fetch`; a damaged installed
resource is a named gap with that repair command.
