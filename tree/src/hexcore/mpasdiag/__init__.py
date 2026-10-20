"""In-model diagnostics from the pinned MPAS-A v8.4.2 source.

This package consumes live device pools and stream alarms; history-file
post-products continue to be owned by the Rust render/product door.
This modified CUDA derivative retains the MPAS copyright and license in
_license.py.
"""
from .manager import DiagnosticTemplate, DiagnosticsManager
from .utils import DiagnosticsUtils, Direction, StreamView
from .isobaric import IsobaricDiagnostics, compute_output_diagnostics
from .output_cycle import OutputDiagnosticsCycle
from .streams import LiveStreamAdapter

__all__ = ["DiagnosticTemplate", "DiagnosticsManager", "DiagnosticsUtils",
           "Direction", "StreamView", "IsobaricDiagnostics", "compute_output_diagnostics",
           "OutputDiagnosticsCycle", "LiveStreamAdapter"]
