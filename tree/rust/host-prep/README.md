# Contract host preparation

This crate computes the regional contract's float32 acoustic substep,
horizontal scalar edge values, vertical scalar flux, and transport finish.
The unchanged NumPy functions remain the independent test authorities.
Only the contract driver calls this binary. Forecast numerics do not change.

Build on the CPU box:

```
CARGO_BUILD_JOBS=2 cargo build --release --manifest-path rust/host-prep/Cargo.toml
export GPUWM_HEX_CONTRACT_HOSTPREP=$PWD/rust/host-prep/target/release/hex_contract_hostprep
```

An installed wheel finds the executable through that environment variable
or PATH. A source checkout also checks the crate's release target directory.
An explicit missing path refuses instead of selecting a different executable.

The private streaming protocol begins with `HEXREF01`, mode and six u64
dimensions (cells, edges, levels, cell stencil width, advection stencil
width, tracers), followed by a u64 array count. Each array carries a u64
ASCII name length, name, u8 element width, u64 value count and contiguous
little-endian values. Width 4 is float32; width 8 is int64. Outputs are
contiguous float32 arrays in the order declared by the Python bridge.
Modes 0 through 3 are horizontal, vertical, finish and acoustic.

These modes implement the contract's fixed third-order coefficient 0.25,
outer timestep 120 s, third RK stage transport, and second acoustic substep
with timestep 120/3/6 s. They are not general forecast API entrypoints.
Arithmetic retains the authority's float32 operation order. No fast math
or fused multiply-add is enabled.

The driver loads a CPU bundle once, prepares shared references before CUDA
initialization, and runs eight independent deck processes by default.
Linux workers share the parent bundle through fork. Other platforms use
spawn. `--workers 1` measures serial execution of the same Rust references.
No worker reads a serialized cache from a previous run. The second device
run and the mutated device run both compare against the original host
reference. The transport finish keeps its existing device-flux substitution
and caches separately for each exact flux hash.

Validation:

```
GPUWM_NO_LOCAL_GPU=1 python -m pytest -q tests/test_contract_hostprep.py
python tools/check_contract_hostprep.py --grid GRID --init INIT --lbc-dir LBC \
  --start-time START --out RECEIPT.json
```

The real-input checker retains only timings and full-array hashes. The
arrays stay in process and disappear when it exits. `prep.deck` C1 begin/end
events include deck and kernel names; logging failures cannot change the
verdict. `prep.references` records the shared host-reference preparation.
