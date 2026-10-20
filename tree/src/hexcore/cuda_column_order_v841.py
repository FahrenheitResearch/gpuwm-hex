"""Preserve the legacy column pack's face carriers under entity permutation.

Column physics consumes raw cell momentum rates. Its retained ARW face-u
carrier averages adjacent *stored* columns, so a new memory order changes
that unused carrier. Run the existing face coupler in the original pack
order, and return its cell-aligned v carrier to the active memory order.
No floating-point expression or weather tendency is replaced.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import numpy as np

_ACTIVE = ContextVar("hex_column_pack_order", default=None)


@dataclass
class ColumnOrder:
    # ``forward``/``inverse`` are the limited-area pack: the real cells plus
    # native's garbage column in its own slot.  ``real_*`` are the same
    # permutation without that slot.
    forward: object
    inverse: object
    real_forward: object = None
    real_inverse: object = None

    @classmethod
    def from_metadata(cls, source):
        order = np.asarray(source)
        if order.ndim != 1 or order.dtype.kind not in "iu" or order.size == 0:
            raise ValueError("cell_order_original must be a nonempty integer vector")
        if not np.array_equal(np.sort(order), np.arange(order.size)):
            raise ValueError("cell_order_original must be a bijection; prevents coupling unrelated columns")
        import cupy as cp
        # The real-cell permutation comes from the Rust ordering producer.
        real = cp.asarray(order.astype(np.int32))
        # A limited-area pack is one column wider: native's appended garbage
        # column, which keeps its original allocation slot.  A global pack
        # has no such column (n_columns is nCells), so it takes the real-cell
        # permutation alone.  MEASURED 2026-10-11, x1.163842 reordered, one
        # card, default physics: the padded map alone refused step 1 with
        # "cell momentum shape (55, 1, 163842), ordered columns 163843"; with
        # both maps the hour runs and every cell field of its history equals
        # the unordered run's under the permutation, byte for byte.
        forward = cp.concatenate((real, cp.asarray([order.size], dtype=cp.int32)))
        return cls(forward, cp.argsort(forward).astype(cp.int32),
                   real, cp.argsort(real).astype(cp.int32))

    def maps(self, columns):
        """The (forward, inverse) pair for a pack ``columns`` wide, or None."""
        cells = int(self.forward.size) - 1
        if columns == cells and self.real_forward is not None:
            return self.real_forward, self.real_inverse
        if columns == cells + 1:
            return self.forward, self.inverse
        return None

    def couple(self, original, state, cfg, mass_u, mass_v):
        import cupy as cp
        maps = None if mass_u.ndim != 3 else self.maps(int(mass_u.shape[-1]))
        if state.has_msf or maps is None or mass_u.shape[1] != 1:
            cells = int(self.forward.size) - 1
            raise ValueError(
                "original column pack requires one row and identity map factors; prevents changing physical face coupling "
                f"(has_msf={bool(state.has_msf)}, cell momentum shape {tuple(mass_u.shape)}, ordered cells {cells}: "
                f"a global pack is {cells} columns wide, a limited-area pack {cells + 1})")
        forward, inverse = maps
        u = cp.take(mass_u, inverse, axis=-1)
        v = cp.take(mass_v, inverse, axis=-1)
        ru, rv = original(state, cfg, u, v)
        # Face-u is the original synthetic ARW row's serialized carrier.
        # Face-v is duplicated in y and remains a physical cell column field.
        return ru, cp.take(rv, forward, axis=-1)


def install_coupler():
    from gpuwm.core import physics
    current = physics._couple_momentum_to_faces
    if getattr(current, "_hex_original_pack_order", False):
        return

    def couple(state, cfg, mass_u, mass_v):
        order = _ACTIVE.get()
        if order is None:
            return current(state, cfg, mass_u, mass_v)
        return order.couple(current, state, cfg, mass_u, mass_v)

    couple._hex_original_pack_order = True
    physics._couple_momentum_to_faces = couple


@contextmanager
def original_pack_scope(order):
    install_coupler()
    token = _ACTIVE.set(order)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def attach_column_order(backend, mesh):
    source = getattr(mesh, "arrays", {}).get("cell_order_original")
    if source is None:
        return None
    order = ColumnOrder.from_metadata(source)
    eager = backend.begin_step

    def phase1(*args, **kwargs):
        with original_pack_scope(order):
            return eager(*args, **kwargs)

    backend.begin_step = phase1
    return {"cells":int(np.asarray(source).size),
            "physics_arithmetic":"existing face coupler, original column pack order",
            "field":"cell_order_original"}
