"""Native output ordering; the forecast driver owns clock/state advancement.

Stream adapter operations are ringing_output_alarms(), write(), and
reset_output_alarms(stream_id=None). They operate on the live stream manager.
The output_fields callback computes atm_compute_output_diagnostics for every
block from current state level 1, including f1's dump hook at that call.
"""


class OutputDiagnosticsCycle:
    def __init__(self, manager, streams, output_fields):
        self.manager, self.streams, self.output_fields = manager, streams, output_fields

    def initial_output(self, *, restart):
        """MPAS v8.4.2 mpas_atm_core.F:664-719."""
        self.streams.reset_output_alarms("restart")
        if restart:
            self.streams.reset_output_alarms("output")
            self.streams.reset_output_alarms("diagnostics")
        if self.streams.ringing_output_alarms():
            self.output_fields()
        self.manager.reset()
        self.manager.update()
        self.manager.compute()
        self.streams.write()
        self.manager.reset()
        self.streams.reset_output_alarms()

    def after_clock_advance(self):
        """MPAS v8.4.2 mpas_atm_core.F:825-910.

        Call after the completed step is rotated to level 1 and the clock
        advances. Accumulation occurs even without an output alarm.
        """
        if self.streams.ringing_output_alarms():
            self.output_fields()
        self.manager.update()
        self.manager.compute()
        self.streams.write()
        self.manager.reset()
        self.streams.reset_output_alarms()
