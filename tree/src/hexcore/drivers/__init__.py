"""The forecast drivers, shipped inside the package.

``run_cuda_v841_forecast`` is the engineering forecast driver the
``gpuwm-hex forecast`` door runs, ``run_cuda_v841_full_physics_x4`` is the
one-case proof harness it reaches the model through, and
``mpas_mesh_binding`` is the mesh registry both bind against.  They lived in
the repository's ``tools/`` through 0.3.1, which a wheel does not carry, so
the forecast could not run from an install at all; ``tools/`` keeps a
same-named entry for each so scripts that run or import them by path keep
working and reach these modules.

Nothing is imported here: the drivers pull numpy, netCDF4 and (lazily) CuPy,
and the doors that do not forecast must not pay for them.
"""
