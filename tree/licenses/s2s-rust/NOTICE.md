# Native S2S reducer source notices

This directory retains the upstream licence files for the active dependency
closure of rw-s2s. Versions are pinned in tools/rustwx/Cargo.lock. The licence
expressions below were checked in each published crate's Cargo.toml and licence
file. No third-party implementation source is copied into the new crate.

| Crate | Upstream expression | Selected permissive option | Notice files |
| --- | --- | --- | --- |
| autocfg 1.5.1 | Apache-2.0 OR MIT | Apache-2.0 | tree/licenses/s2s-rust/autocfg-1.5.1/LICENSE-MIT, tree/licenses/s2s-rust/autocfg-1.5.1/LICENSE-APACHE |
| chrono 0.4.44 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/chrono-0.4.44/LICENSE.txt |
| crossbeam-deque 0.8.8 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/crossbeam-deque-0.8.8/LICENSE-MIT, tree/licenses/s2s-rust/crossbeam-deque-0.8.8/LICENSE-APACHE |
| crossbeam-epoch 0.9.21 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/crossbeam-epoch-0.9.21/LICENSE-MIT, tree/licenses/s2s-rust/crossbeam-epoch-0.9.21/LICENSE-APACHE |
| crossbeam-utils 0.8.23 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/crossbeam-utils-0.8.23/LICENSE-MIT, tree/licenses/s2s-rust/crossbeam-utils-0.8.23/LICENSE-APACHE |
| either 1.19.0 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/either-1.19.0/LICENSE-MIT, tree/licenses/s2s-rust/either-1.19.0/LICENSE-APACHE |
| itoa 1.0.18 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/itoa-1.0.18/LICENSE-MIT, tree/licenses/s2s-rust/itoa-1.0.18/LICENSE-APACHE |
| memchr 2.8.3 | Unlicense OR MIT | MIT | tree/licenses/s2s-rust/memchr-2.8.3/COPYING, tree/licenses/s2s-rust/memchr-2.8.3/LICENSE-MIT |
| num-traits 0.2.19 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/num-traits-0.2.19/LICENSE-MIT, tree/licenses/s2s-rust/num-traits-0.2.19/LICENSE-APACHE |
| rayon 1.10.0 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/rayon-1.10.0/LICENSE-MIT, tree/licenses/s2s-rust/rayon-1.10.0/LICENSE-APACHE |
| rayon-core 1.13.0 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/rayon-core-1.13.0/LICENSE-MIT, tree/licenses/s2s-rust/rayon-core-1.13.0/LICENSE-APACHE |
| ryu 1.0.23 | Apache-2.0 OR BSL-1.0 | Apache-2.0 | tree/licenses/s2s-rust/ryu-1.0.23/LICENSE-BOOST, tree/licenses/s2s-rust/ryu-1.0.23/LICENSE-APACHE |
| serde 1.0.218 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/serde-1.0.218/LICENSE-MIT, tree/licenses/s2s-rust/serde-1.0.218/LICENSE-APACHE |
| serde_json 1.0.140 | MIT OR Apache-2.0 | Apache-2.0 | tree/licenses/s2s-rust/serde_json-1.0.140/LICENSE-MIT, tree/licenses/s2s-rust/serde_json-1.0.140/LICENSE-APACHE |

The five proc-macro packages recorded by serde's inactive lockfile dependencies
are not compiled by this crate. See the build dependency tree in the delivery
receipt. This reducer avoids adding those compiler dependencies.

The renderer extension uses the installed engine's existing dependency graph.
Its existing MIT notice is retained in src/hexcore/data/licenses/rustwx-LICENSE-MIT.
The new native NetCDF transport is seeded from this repository's rw-surface
transport, under the repository's Apache-2.0 licence. It links the system
NetCDF C library, whose core licence is BSD-3-Clause; no copy of that library
is redistributed by this source patch.

No observation data or operational EOFs are redistributed. Reference inputs
must identify their own licence. Non-permissive inputs require the matching
explicit research opt-in, and are never downloaded by these modules.
