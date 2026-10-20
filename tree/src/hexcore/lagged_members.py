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
    """Cycle-major Cartesian expansion. Every member verifies to one common clock."""
    valid = Cycle(cycle.when + timedelta(hours=lead_hours)).start_time
    result = []
    for lag in lags:
        source = Cycle(cycle.when - timedelta(hours=lag))
        if source.when.hour not in row.get("posting", {}).get("cycle_hours", list(range(24))):
            raise InitSourceRefusal(f"{row['id']}: lag {lag} selects an unpublished cycle")
        for member in members:
            lead = lead_hours + lag
            template_context(source, row, member, lead)
            result.append(dict(source=row["id"], source_cycle=source.start_time,
                               source_member=member, lead_hours=lead, lag_hours=lag,
                               valid_time=valid))
    return result
