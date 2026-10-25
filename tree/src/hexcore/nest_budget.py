"""The money and card checks a ``[hex_nests]`` firing must pass, at fire time.

THE BREAKAGE THIS PREVENTS: the forecast loop armed its trigger without the
two checks the design names for a firing -- "the card has room for one more
nest" and "the remaining hold covers one more nest's worst case" -- so trigger
nests could overfill the card and, with the hold never consulted, run past what
the user authorised (shapes 2-3 checker, 2026-10-23).

THE NUMBERS ARE THE SITE'S.  The prices are computed once, by the site's
pricing (``apps/web/src/lib/nest-addon.ts`` ``triggerHold``), and reach the box
in the job; the worker puts them in the ``[hex_nests]`` table as ``hold``, and
the loop writes them to ``<forecast>/nest-hold.json`` when it arms.  This
module only does cents arithmetic on them:

* ``hold_cents``            what arming held from the user's balance;
* ``reserve_cents``         the reclaim reserve inside it (never given to a firing);
* ``per_firing_cents``      one firing's own hold: the most it can be charged;
* ``per_firing_quote_cents`` one firing's quote: the least it is charged;
* ``overrun_factor``        the charge's margin over measured cost (1.25);
* ``output_seconds``        the upload tail the site adds to every nest's cost
                            (nest-addon.ts ``outputSeconds``; 120 s when the
                            table predates the key).

A firing fires only when ``hold - reserve - committed >= per_firing_cents``,
where ``committed`` is each earlier firing's final charge once its nest has
ended (``nest-hold-spent.jsonl``, written by ``add-nest``) and its whole
``per_firing_cents`` while it runs.  The firing row carries ``budget_cents``;
``add-nest`` stops the nest when its measured cost reaches ``budget_cents /
overrun_factor`` less the upload tail that follows the stop (the site's
``nestStopCents``), so its charge (measured cost x overrun, capped at the
budget) is never below its cost and the run's total never passes its hold.
(Before 2026-10-24 the box left the tail out: a trigger nest stopped at its
hold cost up to 186 cents against a 172.8-cent stop, and the box and the site
stopped nests by different rules.)

The card check is a reservation in :mod:`hexcore.card_ledger`, sized by the
nest's own estimate (``<forecast>/nest-fire-budget.json``, written by
``add-nest --watch`` from the engine's estimator for the table's nest), made
atomically so two quick firings cannot both pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .card_ledger import CardLedger

HOLD_NAME = "nest-hold.json"
SPENT_NAME = "nest-hold-spent.jsonl"
FIRE_BUDGET_NAME = "nest-fire-budget.json"
HOLD_SCHEMA = "gpuwm-hex.nest-hold.v1"
#: The site's upload tail per nest (apps/web nest-addon.ts NEST_ADDON.outputSeconds).
UPLOAD_TAIL_SECONDS = 120.0


def stop_cents(hold_cents: float, *, overrun_factor: float, usd_per_hour: float | None,
               output_seconds: float = UPLOAD_TAIL_SECONDS) -> float:
    """nest-addon.ts ``nestStopCents``: the hold over the overrun factor, less the upload tail at the box's rate."""

    tail = (float(output_seconds) / 3600.0) * float(usd_per_hour or 0.0) * 100.0
    return max(0.0, float(hold_cents) / float(overrun_factor) - tail)


def reservation_name(run_dir: Path, index: int) -> str:
    """Stable across processes (the loop claims it, add-nest adopts and releases it)."""

    digest = hashlib.sha1(str(Path(run_dir).resolve()).encode("utf-8")).hexdigest()[:10]
    return f"trigger-{digest}-{int(index)}"


def read_hold(run_dir: Path) -> dict[str, Any] | None:
    try:
        value = json.loads((Path(run_dir) / HOLD_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for key in ("hold_cents", "reserve_cents", "per_firing_cents", "overrun_factor"):
        if not isinstance(value.get(key), (int, float)):
            raise ValueError(f"{HOLD_NAME} lacks a number for {key}")
    return value


def read_spent(run_dir: Path) -> dict[int, int]:
    """Final charge (cents) of each firing whose nest has ended."""

    try:
        text = (Path(run_dir) / SPENT_NAME).read_text(encoding="utf-8")
    except OSError:
        return {}
    spent: dict[int, int] = {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
            spent[int(row["index"])] = int(row["charge_cap_cents"])
        except (ValueError, KeyError, TypeError):
            continue
    return spent


def committed_cents(cost_cents: float, *, budget_cents: float, quote_cents: float,
                    overrun_factor: float, quote_slack: float = 1.1) -> int:
    """What a finished firing will be charged (nest-addon.ts nestAddCharge, capped at its budget).

    The quote while the measured cost is at or under quote / 1.1, else the cost
    x the overrun factor, never past the firing's budget and never below its cost.
    """

    if cost_cents <= quote_cents / quote_slack:
        charge = math.ceil(quote_cents)
    else:
        charge = min(math.ceil(budget_cents), math.ceil(cost_cents * overrun_factor))
    return int(max(charge, math.ceil(cost_cents)))


def write_spent(run_dir: Path, index: int, charge_cap_cents: int) -> None:
    path = Path(run_dir) / SPENT_NAME
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"index": int(index), "charge_cap_cents": int(charge_cap_cents)}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def hold_left_hook(run_dir: Path, firings: Callable[[], list[Mapping[str, Any]]]):
    """``LoopTrigger(hold_left=...)``: refuse a firing the remaining hold cannot cover."""

    def check(firing: dict[str, Any]) -> tuple[bool, str]:
        hold = read_hold(run_dir)
        if hold is None:
            # A run armed without a price (a person's own box, not the site):
            # nothing is held, so nothing caps the firings but max_firings.
            return True, "no hold"
        spent = read_spent(run_dir)
        per = int(hold["per_firing_cents"])
        committed = sum(spent.get(int(f["index"]), int(f.get("budget_cents", per))) for f in firings())
        remaining = int(hold["hold_cents"]) - int(hold["reserve_cents"]) - committed
        if remaining < per:
            return False, (f"{remaining} cents of the hold left after {len(firings())} firing(s); "
                           f"one more nest may cost up to {per}")
        firing["budget_cents"] = per
        firing["stop_after_cents"] = round(stop_cents(
            per, overrun_factor=float(hold["overrun_factor"]), usd_per_hour=hold.get("usd_per_hour"),
            output_seconds=float(hold.get("output_seconds", UPLOAD_TAIL_SECONDS))), 3)
        firing["usd_per_hour"] = hold.get("usd_per_hour")
        return True, f"{remaining} cents left"

    return check


def card_fit_hook(run_dir: Path, ledger: CardLedger, *, owner_pid: int | None = None):
    """``LoopTrigger(fits=...)``: reserve the nest's memory on the card, atomically."""

    def check(firing: dict[str, Any]) -> tuple[bool, str]:
        try:
            budget = json.loads((Path(run_dir) / FIRE_BUDGET_NAME).read_text(encoding="utf-8"))
            nbytes = int(budget["nest_bytes"])
        except (OSError, ValueError, KeyError, TypeError):
            # The watcher has not priced the nest yet; the block stays held and
            # the next look tries again.
            return False, f"the nest's size is not measured yet ({FIRE_BUDGET_NAME})"
        name = reservation_name(run_dir, int(firing["index"]))
        # No nest process exists yet: the parent (this process) only keeps the
        # row alive.  Its own card memory must never count as the nest's
        # allocation (card_ledger module text).
        verdict = ledger.reserve(name, nbytes, [], owner=owner_pid or os.getpid())
        if verdict["fits"] is False:
            return False, (f"needs {verdict['need_bytes'] / 2**30:.1f} GiB, "
                           f"{verdict['available_bytes'] / 2**30:.1f} GiB free after other nests' claims")
        firing["card_reservation"] = name
        firing["card_ledger"] = str(ledger.path)
        return True, "reserved"

    return check
