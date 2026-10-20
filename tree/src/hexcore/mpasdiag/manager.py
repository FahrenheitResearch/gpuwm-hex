"""MPAS diagnostics lifecycle, with other lanes supplying their subsystems."""
from .utils import DiagnosticsUtils


class DiagnosticTemplate:
    def setup(self, configs, pools, clock):
        """MPAS v8.4.2 mpas_atm_diagnostic_template.F:37-56."""
        self.mesh, self.clock = pools["mesh"], clock

    def update(self):
        """MPAS v8.4.2 mpas_atm_diagnostic_template.F:69-76: no-op."""

    def compute(self):
        """MPAS v8.4.2 mpas_atm_diagnostic_template.F:89-96: no-op."""

    def reset(self):
        """MPAS v8.4.2 mpas_atm_diagnostic_template.F:109-116: no-op."""

    def cleanup(self):
        """MPAS v8.4.2 mpas_atm_diagnostic_template.F:129-135: no-op."""


class DiagnosticsManager:
    """Other diagnostic modules are required; missing ones cannot be skipped.

    Their adapters implement setup/compute and the update/reset/cleanup
    methods that native calls. State pools hold the current native level 1.
    """
    def __init__(self, *, isobaric, cloud, convective, pv, soundings):
        self.utils = DiagnosticsUtils()
        self.template = DiagnosticTemplate()
        self.isobaric, self.cloud = isobaric, cloud
        self.convective, self.pv, self.soundings = convective, pv, soundings

    def setup(self, streams, configs, pools, clock, dminfo):
        """MPAS v8.4.2 mpas_atm_diagnostics_manager.F:31-63."""
        self.utils.init(streams)
        self.template.setup(configs, pools, clock)
        self.isobaric.setup(pools, clock, self.utils)
        self.cloud.setup(pools, clock, self.utils)
        self.convective.setup(pools, clock, self.utils)
        self.pv.setup(pools, clock, self.utils)
        self.soundings.setup(configs, pools, clock, dminfo)

    def update(self):
        """MPAS v8.4.2 mpas_atm_diagnostics_manager.F:76-87."""
        self.template.update()
        self.convective.update()

    def compute(self):
        """MPAS v8.4.2 mpas_atm_diagnostics_manager.F:100-119."""
        self.template.compute()
        self.isobaric.compute()
        self.cloud.compute()
        self.convective.compute()
        self.pv.compute()
        self.soundings.compute()

    def reset(self):
        """MPAS v8.4.2 mpas_atm_diagnostics_manager.F:132-143."""
        self.template.reset()
        self.convective.reset()

    def cleanup(self):
        """MPAS v8.4.2 mpas_atm_diagnostics_manager.F:156-173."""
        self.template.cleanup()
        self.soundings.cleanup()
        self.utils.finalize()
