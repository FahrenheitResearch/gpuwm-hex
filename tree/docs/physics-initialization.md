# Soil and aerosol initialization

Pass the forecast's physics suite to the normal initialization door:

```sh
gpuwm-hex init --physics-suite rrfs_v2_tempo_candidate ...
```

The suite supplies the target soil geometry and the complete transported
species row. `--land-surface noahmp` selects four soil layers while retaining
the suite's TEMPO species. Initialization without a suite retains the existing
four-layer Noah path and does not add a species row. On the ordinary forecast
road, selecting a suite completes its initial fields automatically. A soil
geometry change replays the recorded MET source with the original init controls
into a sibling `.physics-init.nc`, and does not alter the caller's input. Missing
or changed analysis or changed initial-state bytes refuse this replay. Aerosol
completion uses the same source decision below. Already complete inputs pass
through unchanged.

The projected intermediate producer retains the original source soil depth
nodes alongside its existing four layer fields. Equal upper and lower WPS
soil bounds identify depth nodes. The RUC initializer interpolates these
samples onto its nine nodes. Layer sources use WRF's skin and deep temperature
anchors and its moisture extrapolation. The initializer does not pad a
four-layer state to nine levels.

A complete, positive analyzed water-friendly and ice-friendly aerosol pair
has priority. Otherwise the door uses the engine's WIF dataset resolver and
Rust-seeded climatology pipeline. A microphysics row requiring climatology,
including TEMPO, acquires the verified monthly dataset automatically when
the probe finds none. Other aerosol-aware rows retain the WRF synthetic
height profile fallback and report it. `--wif-climatology PATH` names the dataset to use when analysis
does not supply a usable pair. The provenance receipt records the source
decision and native augmentation command.

The native writer adds the entire selected species row. This prevents a
partial row from discarding supplied aerosols during cold initialization.
Dataset resolution and process scheduling are Python orchestration. Soil
remapping, the synthetic fallback, geometry conversion and file emission are
Rust.

This initialization change does not qualify forecast skill. Use observations
to grade physics. The MPAS-A v8.4.1 dycore source pins are unchanged.
