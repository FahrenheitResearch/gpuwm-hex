"""One process-wide lock for netCDF4 / HDF5 calls made from threads.

THE BREAKAGE THIS PREVENTS (gpuwm-hex 0.4.4 candidate, CPU battery on Python 3.13, 2026-10-23): the
products door renders its views in a thread pool, and two views opened their frames with netCDF4 at
once to write the graticule overlays; the netCDF-C and HDF5 libraries netCDF4-python links are not
thread-safe, and the battery died with a segmentation fault inside write_graticule_overlays.  The
boundary series (nest_boundary) hit the same library limit and holds the same lock.
"""

from __future__ import annotations

import threading

NETCDF = threading.RLock()
