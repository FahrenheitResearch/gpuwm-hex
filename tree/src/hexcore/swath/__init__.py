"""Where the fine grid goes, and how it moves when the weather does.

This package is the decision layer of the cascade: it reads a coarse
forecast this project produced, finds what is worth resolving, projects
where that thing will BE over the fine level's lead window, and emits the
mesh-spec rows that put resolution there.  It runs on the CPU, it opens no
device, and it writes documents -- the mesh lane, the culler and the
forecast door consume those documents unchanged.

ONE MECHANISM, MANY CONFIGURATIONS.  A tropical cyclone and a
four-ingredient fire-weather region are not two code paths here; they are
two rows of ``threat-metrics``.  The pipeline is the same for both:

    field derivation -> feature detection -> confirmation -> region ->
    track association -> projection -> swath ring -> ranking ->
    admission -> hysteresis -> start time

Adding a phenomenon is a row.  Adding a field is a row.  Adding a source
is a row.  If something ever needs a branch on WHICH phenomenon it is,
that is the defect, not the feature.

AND A COMPOUND PHENOMENON IS ALSO A ROW.  Most threats worth chasing are
conjunctions over quantities that share no unit -- fire weather is hot AND
dry AND windy over dry fuel; organised severe convection is a storm AND
shear AND ascent.  A field row's operands come from the history file or
from OTHER FIELD ROWS, ``threshold_margin`` turns any field into a
dimensionless distance past a threshold, and ``extremum_of`` takes the
weakest of several.  So the conjunction is a composition of rows and it
reaches the same detector a blob of reflectivity does.
"""

from __future__ import annotations

from .errors import (
    SwathCapacityRefusal,
    SwathDocumentError,
    SwathError,
    SwathRefusal,
)

__all__ = [
    "SwathCapacityRefusal",
    "SwathDocumentError",
    "SwathError",
    "SwathRefusal",
]
