"""Bind o1's live alarm API to diagnostics without duplicating Rust I/O.

The forecast supplies stream_active(stream), field_activity(stream), a live
current_time() and write_stream(stream, time). Field activity must describe
registered members independently of whether the containing stream is active;
using Stream.fields() on a package-inactive stream loses that information.
"""
from .utils import Direction, StreamView


class LiveStreamAdapter:
    """Bridge StreamAlarms.due/reset and the diagnostics output-cycle API.

    Selection authority is MPAS v8.4.2 mpas_atm_diagnostics_utils.F:85-129.
    Alarm/write/reset ordering is owned by OutputDiagnosticsCycle.
    """
    def __init__(self, alarms, current_time, *, stream_active, field_activity,
                 write_stream):
        self.alarms, self.current_time = alarms, current_time
        self.stream_active, self.field_activity = stream_active, field_activity
        self.write_stream = write_stream

    @staticmethod
    def _direction(stream):
        directions = frozenset(stream.options["type"].split(";"))
        return {frozenset(("input",)): Direction.INPUT,
                frozenset(("output",)): Direction.OUTPUT,
                frozenset(("input", "output")): Direction.INPUT_OUTPUT}[directions]

    def __call__(self):
        when = self.current_time()
        return tuple(StreamView(
            name, self._direction(stream), self.stream_active(stream),
            self.alarms.due(name, when, "output"), self.field_activity(stream),
            stream.interval("input"), stream.interval("output"))
            for name, stream in self.alarms.streams.items())

    def ringing_output_alarms(self):
        when = self.current_time()
        return tuple(name for name, stream in self.alarms.streams.items()
                     if self._direction(stream) != Direction.INPUT
                     and self.stream_active(stream)
                     and self.alarms.due(name, when, "output"))

    def write(self):
        when = self.current_time()
        for name in self.ringing_output_alarms():
            self.write_stream(self.alarms.streams[name], when)

    def reset_output_alarms(self, stream_id=None):
        when = self.current_time()
        names = self.alarms.streams if stream_id is None else (stream_id,)
        for name in names:
            stream = self.alarms.streams.get(name)
            if stream is not None and self._direction(stream) != Direction.INPUT:
                self.alarms.reset(name, when, "output")
