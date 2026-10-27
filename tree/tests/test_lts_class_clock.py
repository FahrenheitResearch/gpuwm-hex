"""A default step under full-step local time stepping lands on the class clock.

THE BREAKAGE (multi-zoom run3, 2026-10-26): on a refined mesh whose finest cells take a 20 s default step, the
coarsest LTS class steps max_rate x 20 s, which misses the 1800 s radiation interval, so every bare active-LTS run
was refused. Carried from feat/multi-zoom-door's test_multi_zoom.py with the fix (2c56994f).
"""

from hexcore import mpas_defaults


def test_lts_default_step_lands_on_the_class_clock():
    """3.75 km cells take 20 s by default; 16 x 20 s misses the 1800 s radiation clock, 18.75 s does not."""
    events = {"radiation": 1800.0, "history": 3600.0, "run length": 21600.0}
    default = mpas_defaults.default_dt_seconds(3750.0)
    assert default["dt_seconds"] == 20.0
    moved = mpas_defaults.lts_class_clock_dt(default, max_rate=16, events=events)
    assert moved["dt_seconds"] == 18.75 and moved["source"] == "lts class clock"
    # An explicit step is the user's: the class clock refuses it, the rule does not move it.
    explicit = {"dt_seconds": 20.0, "source": "explicit"}
    assert mpas_defaults.lts_class_clock_dt(explicit, max_rate=16, events=events) is explicit
    # A default already on the clock is untouched.
    on_clock = {"dt_seconds": 18.75, "source": "mesh"}
    assert mpas_defaults.lts_class_clock_dt(on_clock, max_rate=16, events=events) is on_clock
