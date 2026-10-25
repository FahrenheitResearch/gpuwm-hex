"""A source's horizon: its last posted forecast hour, and what a run past it is.

Every value comes from the init-source table row (``forecast.ladders`` per cycle hour, else ``forecast.max_hours``),
so a new source's horizon is table work, never code (the arbitrary acceptance test).

Extending a run: start from the source's LAST hour (``--from-last-hour``) and run on. A global or variable-resolution
mesh needs no boundaries, so the hours past the source's horizon are a free run of our own model: nothing from the
source constrains them, and every receipt says so. A limited area past the horizon takes its boundaries from a
global hex run we make ourselves (``gpuwm-hex regional --parent-*`` on that run), never from the source.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Mapping

FREE_RUN_NOTE = ("free-running beyond the source: past its last forecast hour this is our own model's forecast, "
                 "nothing from the source constrains it")


class SourceHorizonError(ValueError):
    pass


def last_posted_lead(row: Mapping[str, Any], cycle_hour: int) -> int:
    """The last forecast hour the source posts for a cycle at this UTC hour (IFS HRES: 360 h at 00/12Z, 144 h at
    06/18Z; GFS 384 h; AIFS 360 h)."""
    forecast = row.get("forecast") or {}
    if not forecast:
        raise SourceHorizonError(f"source {row.get('id')!r} declares no forecast hours (an analysis source), so it "
                                 "has no last hour to start from")
    for ladder in forecast.get("ladders") or ():
        if ladder.get("cycle_hours") is None or int(cycle_hour) in ladder["cycle_hours"]:
            return int(ladder["steps"][-1][0])
    return int(forecast["max_hours"])


def beyond_source(row: Mapping[str, Any], source_cycle: _dt.datetime, lead_hours: float,
                  hours: float) -> dict[str, Any] | None:
    """The part of a run past the source's horizon, or None when the source covers the whole run."""
    if not row.get("forecast"):
        horizon = 0
    else:
        horizon = last_posted_lead(row, source_cycle.hour)
    end = float(lead_hours) + float(hours)
    if end <= horizon:
        return None
    start = max(float(horizon), float(lead_hours))
    return dict(source=row.get("id"), source_cycle_utc=source_cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
                source_last_lead_hours=horizon,
                free_running_from_utc=(source_cycle + _dt.timedelta(hours=start)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                free_running_hours=end - start, note=FREE_RUN_NOTE)


def resolve_lead(row: Mapping[str, Any], source_cycle: _dt.datetime, lead_hours: int = 0,
                 from_last_hour: bool = False) -> int:
    if from_last_hour:
        if lead_hours:
            raise SourceHorizonError("--lead-hours and --from-last-hour name two different starts; pass one")
        return last_posted_lead(row, source_cycle.hour)
    lead = int(lead_hours or 0)
    if lead < 0:
        raise SourceHorizonError("--lead-hours must be nonnegative")
    return lead


def brand(issued_label: str, extension: Mapping[str, Any] | None) -> str | None:
    """The soundings' top-right text for a run past its source (None when the source covers it)."""
    if not extension:
        return None
    return f"{issued_label} · free run past f{extension['source_last_lead_hours']:03d}"


def label(issued_label: str, extension: Mapping[str, Any] | None) -> str:
    """The run's init label, naming the free run when there is one."""
    if not extension:
        return issued_label
    return (f"{issued_label}; free-running beyond the source after its f{extension['source_last_lead_hours']:03d} "
            f"({extension['free_running_hours']:g} h)")
