# High-top vertical binding

`global-120km-high-top` selects `countries/s2s80-v1.json`. It has 80 layers,
81 interfaces and a 50,000 m lid, approximately 1 hPa. Above 20,000 m the
interface spacing is 750 m. Actual top pressure depends on the initialized
atmosphere and must be measured from the run.

The version 2 declaration also binds the existing forecast controls: vertical
Rayleigh damping from 40,000 m with coefficient 0.2, a 0.2 CAM coefficient in
the top eight layers, horizontal Rayleigh damping in the top six layers with
a five-day timescale, and RRTMG longwave and shortwave radiation every 1,800 s.
RRTMG uses the engine's pressure, latitude and month dependent climatological
ozone. The declaration identifies that existing ozone authority. It does not
introduce prognostic ozone.

Print a complete plan without acquiring data or using a GPU:

```sh
gpuwm-hex s2s --row global-120km-high-top --source cfs \
  --cycle YYYY-MM-DDTHH --days 30 --members 1 --lagged 0 \
  --work-dir RUN --dry-run
```

Remove `--dry-run` to run the door. Both `global` and `s2s` also accept
`--vertical-spec PATH`. An explicit selection overrides the row's declaration.
The selected digest determines the runtime mesh name, registry level count,
init cache location and S2S restart identity. Prepared S2S inits must carry
matching vertical provenance. Published vertical metrics cannot be relabelled
with a different level count or model top.

Version 1 declarations keep their canonical digests and existing runtime
defaults. Version 2 requires all runtime settings, including its ozone and
radiation pair, so a partial declaration cannot silently inherit low-top
settings.

The real-size qualification is [the retained receipt](s2s/receipts/high-top-20180205.json),
with [31 daily diagnostics](s2s/receipts/high-top-20180205-daily.csv).
The 2018-02-05 CFS initialization ran for 30 days on 40,962 cells through
4,320 accepted steps and all products. Sampled own peak was 15,288 MiB on
one RTX PRO 6000, with a 24,576 MiB shared-pool reservation. Wall including
pool wait was 3,534.458 s; the forecast stage was 3,073.248 s. Actual top
midpoint pressures across all daily frames were 0.3913 to 1.0002 hPa.

At 10 hPa, the native-cell area-weighted 60-90N temperature rose from
217.3816 K to 241.5559 K by February 15. The area-weighted 59-61N zonal
wind changed from +44.8285 m/s initially to -3.2042 m/s on February 12,
the first daily reversal. The wind band approximates 60N. This demonstrates
the simulated warming and reversal across the real event. It uses one
member and retrospective analysis SST/ice. Independent observed skill,
ensemble probability skill and multi-member card fit remain unqualified.
Observation coverage was missing; climatology used the explicitly labelled
ensemble-relative fallback. The plan continues to distinguish derived
80-level capacity estimates from measured 55-level rows.

The sub-2 hPa lid also needs the engine clear-buffer correction in
`docs/s2s/engine-patches/0003-rrtmg-high-top-clear-buffer.patch`.
The ordinary WRF 4 hPa extension rounds to zero at that lid, while the
longwave prep requires one layer above the atmosphere. The correction
retains one clear buffer and preserves the ordinary lower-top layer count.
The complete six-source engine helper set is bound by
`engine-2.9.3-high-top-buffer-source`; it does not relax source identity
checks. The patch must land in the engine before an unmodified engine
installation can run this row. The qualification used an isolated engine copy with this correction.
