# Moving-region ensembles

The one-card moving multi-class executor supports GPU SPPT/SKEB. One member
spectral clock advances at each synchronized LTS macro start, with the pattern
held over that macro. Each class uses the same global dual-vertex curl through
its halos. Unique owned-edge dissipation integrals feed an area-weighted global
cell reduction at the macro boundary. SKEB uses that completed budget in the next
macro, with a zero budget at startup. Receipts name this one-macro lag.

Checkpoint carriers retain spectral coefficients, counters, the synchronized
boundary and the per-level budget. They do not retain a mesh-specific harmonic
basis. The destination mesh rebuilds its basis on the GPU and restores the same
member process. Splitting one stochastic member across multiple cards still
requires a cross-device budget reduction and is refused. Independent ensemble
members can each use one granted card. Whole moving-ensemble qualification is
pending; the default perturbations remain enabled.

Stage and prepare a moving case once, then run independent moving forecasts:

```bash
gpuwm-hex moving-ensemble --case CASE.json --casedir prepared/ --out ensemble/ --hours 6
```

The default roster has eight equal-weight members. Each has a stable member ID,
independent GPU SPPT/SKEB forcing and private moving-controller, physics and restart
state. The packaged `data/moving-ensemble.v1.json` supplies alternating WSM6 and
Thompson microphysics rows. Existing case physics and forecast defaults continue
to apply; the member's explicit physics flags override the corresponding choices.
This is a research ensemble. The spread has no calibration or forecast-skill claim.

Only the prepared source and initial geometry are shared read-only. Every moving
controller diagnoses its own model storm and chooses its own moves. One forecast
runs per visible card; additional members queue on that card. Grant cards through
the host's resource manager first. `--devices` selects the granted visible device
IDs; the default uses `CUDA_VISIBLE_DEVICES`. `--engine` and `--geog` have the same
meaning as on the single moving-run door.

## Explicit weighted/source roster

Use `--manifest roster.json --count N` for an explicit roster. All N rows run.
There is no implicit truncation, random selection or replication of a heavy row.
Weights affect diagnostic summaries, not forecast count. Source preparation can be
from any init-source table row, including external model inputs already admitted
by that table. New source models use table rows and prepared assets.

For a source with ensemble members, set `init_source` and `init_member` in each
moving case before `moving stage`, for example `"init_source": "gefs"` and
`"init_member": 3`. The source table validates the member and selects its products.
Stage and prepare each source member in a separate `casedir`, then reference those
folders in the roster. A prepared folder cannot be restaged as another source,
cycle or member, which would leave its old initial conditions mislabeled.
Receipts preserve the selected source-member number separately from the stable
stochastic identity, and record which stochastic perturbations are enabled.

```json
{
  "schema": "gpuwm-hex.moving-ensemble-roster/v1",
  "source_members": 2,
  "members": [
    {
      "id": "source-a",
      "weight": 16,
      "casedir": "prepared-a",
      "start_member": "start",
      "physics": "wsm6",
      "forecast_flags": ["--physics", "config_microp_scheme=mp_wsm6"]
    },
    {
      "id": "source-b",
      "weight": 6,
      "casedir": "prepared-b",
      "start_member": "start",
      "physics": "thompson",
      "forecast_flags": ["--physics", "config_microp_scheme=mp_thompson"]
    }
  ]
}
```

Relative source folders resolve beside the manifest. Omitted `casedir` uses the
command's shared prepared case. Source clocks and prepared time steps must match
the moving case. Weights must be finite and positive; they normalize over the
selected rows. Controller-owned path, time, stochastic identity and card flags
cannot be replaced through member flags. The seed is `--seed`; stochastic
identities derive from the IDs and survive reordering. Optional manifest and row
`stochastic` objects override physical forcing parameters, with row precedence.

## Products and receipts

`ensemble-receipt.json` records the whole command, selected rows, normalized
weights, stage/init hashes, per-member commands, logs, status, stage walls,
parallelism and sampled VRAM. The whole-card VRAM sample includes other shared
pool owners and is labelled as such. Per-member samples follow the controller's
descendants, every five seconds; neither is an allocator peak.

`ensemble-products.json` contains every tracker fix and coincident-time
spherical mean positions, weighted RMS track spread and weighted P10/P50/P90
central-pressure and maximum-wind values. Missing fixes and unavailable
intensities remain missing; every lead reports counts and included weights.
Native moving-grid fields are never pooled by cell index.
`track-intensity.png` contains receipt analysis charts: member displacements,
track spread and pressure/wind plumes. `track-plume.png` is the native
`rw_wrfbatch --tc-track-plume` map, driven through the renderer environment in
`gpuwm.rustwx`. Its `native-tracks.json` keeps missing fixes and member weights;
`track-render-receipt.json` pins the executable and command. Weather-field maps use the ordinary
`gpuwm-hex render`/`products` Rust path on the individual segment histories.
