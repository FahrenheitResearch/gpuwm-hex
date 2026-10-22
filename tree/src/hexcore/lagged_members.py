"""Clock planning only. Source bytes are fetched and decoded by the source door."""
from datetime import timedelta

from .init_sources import Cycle, InitSourceRefusal, template_context


def parse_lags(spec):
    if spec is None:
        return [0]
    try:
        lags = [int(part.strip()) for part in str(spec).split(",")]
    except ValueError as error:
        raise InitSourceRefusal("--lagged needs comma-separated nonnegative hours") from error
    if not lags or min(lags) < 0 or len(set(lags)) != len(lags):
        raise InitSourceRefusal("--lagged must name each nonnegative lag once")
    return lags


def expand_lags(row, cycle, members, lags, lead_hours=0):
    """Bind source bytes separately from the common forecast window.

    Forecast rows supply their own lead at the common clock. Analysis rows
    supply an earlier analysis at lead zero; the model integrates the gap.
    ``valid_time`` always describes the input bytes, never a relabelled analysis.
    """
    if type(lead_hours) is not int or lead_hours < 0:
        raise InitSourceRefusal("forecast lead must be nonnegative integer hours")
    valid = Cycle(cycle.when + timedelta(hours=lead_hours)).start_time
    result = []
    for lag in lags:
        if type(lag) is not int or lag < 0:
            raise InitSourceRefusal("lags must be nonnegative integer hours")
        source = Cycle(cycle.when - timedelta(hours=lag))
        if (source.when.minute or source.when.second or
                source.when.hour not in row.get("posting", {}).get("cycle_hours", list(range(24)))):
            raise InitSourceRefusal(f"{row['id']}: lag {lag} selects an unpublished cycle")
        for member in members:
            analysis = not row.get("forecast")
            lead = 0 if analysis else lead_hours + lag
            template_context(source, row, member, lead)
            record = dict(source=row["id"], source_cycle=source.start_time,
                          source_member=member, lead_hours=lead, lag_hours=lag,
                          valid_time=source.start_time if analysis else valid)
            if analysis:
                record.update(window_start=valid, spinup_hours=lead_hours + lag)
            result.append(record)
    return result


def forecast_window(provenance, hours):
    """The existing resident roster can integrate private member clocks."""
    spinup = provenance.get("spinup_hours", 0)
    return dict(start_time=provenance["valid_time"], hours=hours + spinup,
                history_from_minutes=60 * spinup)
