# `gpuwm-hex cycle` — following weather from one cycle to the next

Everything this project had built was a snapshot. Detection, placement,
culling, a full-physics limited-area forecast and a composite render all
worked once, from one coarse run, at one hour. The plan layer already knew how
to decide whether a slot MOVES or STAYS between cycles — continuation radius,
promotion margin, dwell, regeneration thresholds — and there was nothing to
hand that decision to.

This door is the loop those parts were built for.

## The two legs

```
gpuwm-hex cycle plan   # what every cycle would place, and what a delayed
                       # start would save.  Nothing is cut, no device opened.

gpuwm-hex cycle run    # do it: cull, mid-window init, boundaries, contract
                       # deck, full-physics fine forecast, frame, next cycle
```

## One cycle, in order

| leg | what it does | typical cost |
| --- | --- | --- |
| plan | detect in the coarse forecast, place swaths, hysteresis decides move or stay | under a second |
| cull (grid) | cut the fine grid out of the parent | 1–3 s |
| **resolution check** | is this swath inside the parent's refinement? | free |
| cull (static, init) | the rest of the limited-area case | 1–3 s |
| delayed start | the parent's state at the hour the swath wanted | seconds |
| boundaries | `rw_mpas_lbc` over the coarse parent's own frames | 3–5 s |
| contract deck | the 22 regional kernels against the CPU authority, on THESE rings | ~2 min |
| fine forecast | full physics, through the shipped forecast door | ~6 min per 6 h |
| render | weather fields through `rw_mpas_convert` + `rw_wrfbatch` | minutes |

## Three things the loop refuses, by name

**A swath outside the parent's refinement.** A parent mesh is refined over the
region *some* placement asked for. A later cycle can rank a swath somewhere
else entirely, and a cull taken there is a limited-area domain made of the
parent's BACKGROUND cells. Nothing about that fails loudly: it binds, it
admits, and it integrates 71 km cells at a 20 s timestep, producing a full set
of frames nobody could tell from a real forecast without reading `dcEdge`. The
loop cuts the grid first, measures `min(dcEdge)`, and moves to the next
candidate. An operational cascade answers this by regenerating the parent for
the new placement; a cascade holding one parent has to skip and say so.

**A late swath with no parent stream.** A swath that wants to start after its
parent's init hour needs the parent's state at that hour. Without
`--parent-history` the only initial condition available is the parent's init,
and covering the window would mean integrating the fine grid through hours
nobody placed a grid for. The refusal happens before anything is cut.

**A cull with no contract deck.** The 22 regional kernels are indexed by ring,
so a deck run on another cull's rings measured another cull's zones. The loop
runs the deck and presents its receipt; a cull whose deck did not pass never
reaches a forecast.

## The delayed start

A swath placed for weather that arrives at hour 12 should start at hour 12.
Before this door the only initial condition a cull could have was its parent's
INIT, so covering that threat meant integrating the fine grid from hour 0 and
throwing the first twelve hours away.

The parent's own history stream cannot be an initial condition on its own —
measured, it publishes no `xtime`, no `zgrid`/`zz`, no base state and no
connectivity. But everything it lacks is TIME-INVARIANT and is in the parent's
init. So a mid-window state is **the culled init with its prognostic state
replaced by the parent's state at the hour the swath wanted**, gathered onto
the child's cells by exact coordinate bits — a cull moves no cell centre, so a
child cell's `latCell`/`lonCell` are the parent's float64 bits unchanged, and
**a miss is a refusal, never a nearest neighbour**.

**What it does not carry**, reported per field on every composition:

- the ice-phase hydrometeors (`qi`, `qs`, `qg`): the parent publishes them and
  the init stream has no slot, so a grid started inside an ice cloud re-forms
  it and the first hour is a spin-up;
- land-surface memory the history stream does not publish — snow water
  equivalent, snow depth, snow cover, sea ice, SST — which stays at the
  parent's own init hour. Soil temperature, soil moisture and skin temperature
  ARE carried;
- turbulent kinetic energy, for the same reason.

**An init's clock lives in three places** and all three move together: the
`xtime` variable, `initial_time`, and the `config_start_time` global attribute
the forecast door asserts `--start-time` against.

## Registering a cull nobody wrote a row for

`gpuwm-hex forecast` resolves `--mesh` against the registry and refuses a name
it does not hold. That is right about a mesh somebody downloaded and
unanswerable for a mesh the cascade cut four seconds ago.
`hexcore.cascade_row` writes the row from the cull receipt that made it, and
the registry re-hashes both files at bind regardless, so a row that lies about
its bytes refuses exactly as a hand-written one would. Lineage stops at a row a
person registered: a cull of an unregistered parent is refused by name.

What a cascade row does not have is a person who read it. What stands in for
that is the per-geometry contract deck, which is a stronger statement about the
rings than a reader is.

## No phenomenon appears on this door

There is no `--tropical-cyclone`, no per-threat mode and no branch on
`threat_class` anywhere in the package. A tropical cyclone, a convective area,
a fire-weather region and an atmospheric river are ROWS in
`threat-metrics.v3`; they reach the loop as admitted entries carrying a cull
region, a mesh spec, an ignition hour and a pad, and the loop cannot tell them
apart. `tests/test_cycle_cascade.py` asserts that as a property of the
executable source — identifiers, argument names and live strings, docstrings
excluded.
