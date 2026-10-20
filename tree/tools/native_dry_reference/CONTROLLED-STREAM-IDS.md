The pinned native stream manager seeds its random file IDs from C time().
For a paired output proof, give native and Python the same declared seed before
initialization. Production code and complete header/hash comparisons stay unchanged.

Build the native input shim:

```
cc -shared -fPIC -Wall -Wextra -Werror tools/native_dry_reference/file_id_clock.c -ldl -o /YOUR/PROOF/file-id-clock.so
MPAS_PROOF_FILE_ID_SEED=1792237507 LD_PRELOAD=/YOUR/PROOF/file-id-clock.so ./atmosphere_model
```

Apply the preload only to the native referee. Run the actual Python driver with:

```
python tools/native_dry_reference/seeded_file_id_python.py --seed 1792237507 --module hexcore.cli -- forecast [normal proof arguments]
```

For an observation driver, use `--script path/to/driver.py -- [driver arguments]`.
The Python wrapper changes only the metadata seed lookup before writer creation.
Neither wrapper reads native output or edits any generated file. Native C time()
is controlled; clock_gettime, monotonic timing and Python time.time remain real.
Keep the pinned executable and case, one process, strict numerics, GPU OWNER and
no CUDA graphs. Check complete inventories, ordered typed headers and SHA256 of
every written file, record both digests, then delete raw outputs.
