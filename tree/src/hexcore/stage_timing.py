"""Wall-clock stage receipts; never insert device synchronization."""

from contextlib import contextmanager
from datetime import datetime, timezone
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@contextmanager
def stage(rows, name, *, parallelism, timeline=None):
    """Record host elapsed time, including failures, at existing boundaries.

    GPU stages include their existing waits only. This is not a device timer.
    ``parallelism`` describes the execution requested, not measured occupancy.
    """
    row = dict(stage=name, started_utc=utc_now(), parallelism=parallelism,
               timer="host wall; no added synchronization", status="running")
    rows.append(row)
    began = time.perf_counter()
    if timeline is not None:
        from . import time_ledger
        time_ledger.emit(timeline, "begin", attrs={"parallelism": parallelism})
    try:
        yield row
    except BaseException as error:
        row.update(status="failed", error=type(error).__name__)
        raise
    else:
        row["status"] = "done"
    finally:
        row.update(ended_utc=utc_now(), wall_seconds=time.perf_counter() - began)
        if timeline is not None:
            time_ledger.emit(timeline, "end", attrs={"parallelism": parallelism,
                             "status": row["status"], "wall_seconds": row["wall_seconds"]})
