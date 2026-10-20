"""MPAS clock and calendar words without a Fortran runtime.

Authority: mpas_timekeeping.F; backend/runtime.f90:330-386,437-490.
Physics alarm scheduling remains in PhysicsManager, which uses rational seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import calendar
from datetime import date, timedelta
import re

import numpy as np


def interval_from_rkind(seconds):
    """mpas_timekeeping.F:1365,1452-1492: F*.9 then eight fraction digits."""
    return Fraction(f'{float(np.float32(seconds)):.9f}'[:-1])


@dataclass(frozen=True)
class ModelTime:
    year: int
    month: int
    day: int
    seconds: Fraction = Fraction(0)
    calendar: str = 'gregorian'

    @classmethod
    def parse(cls, stamp, calendar='gregorian'):
        aliases = {'gregorian': 'gregorian', 'gregorian_noleap': 'noleap',
                   'noleap': 'noleap', '360day': '360_day', '360_day': '360_day'}
        kind = aliases[calendar]
        match = re.fullmatch(r'(\d{4})-(\d\d)-(\d\d)[_T ](\d\d):(\d\d):(\d\d)(?:\.(\d+))?', stamp)
        if match is None:
            raise ValueError('invalid MPAS timestamp: ' + stamp)
        y, m, d, h, minute, s = map(int, match.groups()[:6])
        result = cls(y, m, d, Fraction(h*3600+minute*60+s) +
                     Fraction('0.' + ((match[7] or '0')[:8])), kind)
        if not (1 <= m <= 12 and 1 <= d <= result.month_days(y, m)
                and 0 <= h < 24 and 0 <= minute < 60 and 0 <= s < 60):
            raise ValueError('MPAS timestamp is outside its calendar')
        return result

    def month_days(self, year, month):
        if self.calendar == '360_day':
            return 30
        if self.calendar == 'noleap' and month == 2:
            return 28
        return calendar.monthrange(year, month)[1]

    @property
    def day_of_year(self):
        return self.day + sum(self.month_days(self.year, m) for m in range(1, self.month))

    @property
    def leapyear(self):
        # The manager shim calls isLeapYear(year), independent of TheCalendar.
        return calendar.isleap(self.year)

    def add(self, seconds):
        days, second = divmod(self.seconds + Fraction(str(seconds)), 86400)
        y, m, d = self.year, self.month, self.day + int(days)
        if self.calendar == 'gregorian':
            shifted = date(y, m, self.day) + timedelta(days=int(days))
            y, m, d = shifted.year, shifted.month, shifted.day
        else:
            while d > self.month_days(y, m):
                d -= self.month_days(y, m)
                m += 1
                if m == 13:
                    y, m = y + 1, 1
            while d < 1:
                m -= 1
                if m == 0:
                    y, m = y - 1, 12
                d += self.month_days(y, m)
        return ModelTime(y, m, d, second, self.calendar)

    def stamp(self):
        whole = int(self.seconds)
        h, remainder = divmod(whole, 3600)
        minute, second = divmod(remainder, 60)
        # Native dateTimeString uses whole seconds; fractional time remains in
        # the clock and follows the shim's integer s_n/s_d division below.
        return f'{self.year:04d}-{self.month:02d}-{self.day:02d}_{h:02d}:{minute:02d}:{second:02d}'


class PhysicsClock:
    def __init__(self, *, start, stop, dt, calendar='gregorian'):
        self.start_time = ModelTime.parse(start, calendar)
        self.stop_time = ModelTime.parse(stop, calendar)
        self.current_time = self.start_time
        self.dt = interval_from_rkind(dt)
        if self.dt <= 0:
            raise ValueError('MPAS clock requires a positive time step')
        self.elapsed = Fraction(0)

    def advance(self):
        self.current_time = self.current_time.add(self.dt)
        self.elapsed += self.dt

    def calendar_words(self):
        t = self.current_time
        h, rem = divmod(int(t.seconds), 3600)
        minute, s = divmod(rem, 60)
        f = np.float32
        # Preserve each RKIND operation and the native integer fraction divide.
        utc_h = f(f(f(h) + f(f(minute) / f(60))) + f(f(s) / f(3600)))
        julian = f(f(t.day_of_year - 1) + f(utc_h / f(24)))
        return dict(current_date=t.stamp(), julian=julian, julian_in=julian,
                    leapyear=t.leapyear)
