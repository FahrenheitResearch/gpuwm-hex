"""The boundary-zone shell check, and the test it replaced.

A limited-area mesh's relaxation zone is seven rings, and ring ``k`` is staged
against ring ``k-1`` every step.  ``Mesh.validate`` has to refuse a zone whose
rings do not actually sit on one another, because such a zone blends from a
ring that does not reach it and nothing announces the hole.

It used to check that instead by requiring ring POPULATIONS to grow outward.
That is true on a uniform mesh and false on a variable-resolution one -- and
variable resolution is the only kind this program culls.  Measured on the proving RTX 5090
on 2026-08-27: three culls of one parent, one binary, one run, one of them
byte-identical to the registered ``r4.75.11020``, and the old check refused
two of them.  The refused ones are legitimate: their rings grow outward into
the parent's coarsening ramp, so each ring wraps a LONGER perimeter with
FEWER, WIDER cells.

Both directions are asserted here, because both failure modes are silent:

* a zone whose populations shrink outward but whose rings all sit on the ring
  inside them is ADMITTED -- otherwise the culler's own output is refused and
  a lane loses meshes it correctly built;
* a zone with a genuinely detached ring cell is REFUSED by name -- otherwise
  the check is a formality that passes everything.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexcore.mesh import Mesh, MeshValidationError, regional_ring_shell_errors

from test_regional_mesh_admission import cull_regional_mesh, parent_sphere  # noqa: F401


def _neighbour_pairs(neighbours: dict[int, list[int]]):
    """Flatten an adjacency map into the (source, neighbour) pair arrays."""

    source: list[int] = []
    target: list[int] = []
    for cell, others in neighbours.items():
        for other in others:
            source.append(cell)
            target.append(other)
    return np.asarray(source, np.int64), np.asarray(target, np.int64)


def _concentric_zone(populations):
    """A legal seven-ring zone with the given per-ring cell counts.

    Every ring-``k`` cell is wired to at least one ring-``k-1`` cell and to at
    least one ring-``k+1`` cell, which is what a shell IS.  The counts are free,
    so a shell that shrinks outward and a shell that grows outward are the same
    construction with different numbers -- which is exactly the point: the
    populations carry no information about whether the zone is torn.
    """

    mask: list[int] = []
    rings: list[list[int]] = []
    for ring, count in enumerate(populations):
        rings.append(list(range(len(mask), len(mask) + count)))
        mask.extend([ring] * count)
    neighbours: dict[int, list[int]] = {cell: [] for cell in range(len(mask))}
    for ring in range(1, len(rings)):
        inner, outer = rings[ring - 1], rings[ring]
        for position, cell in enumerate(outer):
            partner = inner[position % len(inner)]
            neighbours[cell].append(partner)
            neighbours[partner].append(cell)
    return np.asarray(mask, np.int64), neighbours


SHRINKING = (40, 254, 251, 249, 242, 241, 237, 232)  # d070, measured
GROWING = (40, 240, 245, 251, 257, 263, 269, 275)  # d100, measured


def test_a_zone_whose_populations_shrink_outward_is_admitted():
    """The false-refusal direction, and it is the one that cost meshes.

    These are ``d070``'s own measured ring populations.  Under the old check
    this mesh -- produced by the same binary, from the same parent, in the same
    run as a cull that is byte-identical to the registered ``r4.75.11020`` --
    was refused as "a torn or renumbered zone".
    """

    mask, neighbours = _concentric_zone(SHRINKING)
    source, target = _neighbour_pairs(neighbours)
    assert regional_ring_shell_errors(mask, source, target, 7) == []


def test_a_zone_whose_populations_grow_outward_is_still_admitted():
    """The old check's own passing case has to keep passing.

    Replacing a gate is only correct if it still admits everything the gate
    admitted.  These are ``d100``'s measured populations.
    """

    mask, neighbours = _concentric_zone(GROWING)
    source, target = _neighbour_pairs(neighbours)
    assert regional_ring_shell_errors(mask, source, target, 7) == []


def test_a_detached_ring_cell_is_refused_and_names_its_ring():
    """The false-admission direction: the breakage the gate exists for.

    One ring-4 cell is rewired to touch only ring-4 and ring-5 cells.  It is
    forced from a ring that does not reach it, and every other invariant --
    populations, mask deltas, no empty rings -- is untouched, so nothing else
    in the validator would notice.
    """

    mask, neighbours = _concentric_zone(GROWING)
    detached = int(np.flatnonzero(mask == 4)[0])
    partners = [c for c in neighbours[detached] if mask[c] == 3]
    assert partners, "the fixture did not wire this cell to the ring inside it"
    for partner in partners:
        neighbours[detached].remove(partner)
        neighbours[partner].remove(detached)
    sibling = int(np.flatnonzero(mask == 4)[1])
    neighbours[detached].append(sibling)
    neighbours[sibling].append(detached)

    source, target = _neighbour_pairs(neighbours)
    errors = regional_ring_shell_errors(mask, source, target, 7)
    assert len(errors) == 1, errors
    assert "ring 4 has cells with no ring-3 neighbour" in errors[0]
    assert "torn or renumbered zone" in errors[0]


def test_every_ring_is_checked_not_only_the_first():
    """A loop that stopped early would pass a tear in any outer ring.

    Checked at ring 7 specifically, the outermost, which is the one carrying
    the sentinels and the one a renumbering is most likely to disturb.
    """

    mask, neighbours = _concentric_zone(GROWING)
    detached = int(np.flatnonzero(mask == 7)[0])
    for partner in [c for c in neighbours[detached] if mask[c] == 6]:
        neighbours[detached].remove(partner)
        neighbours[partner].remove(detached)
    sibling = int(np.flatnonzero(mask == 7)[1])
    neighbours[detached].append(sibling)
    neighbours[sibling].append(detached)

    source, target = _neighbour_pairs(neighbours)
    errors = regional_ring_shell_errors(mask, source, target, 7)
    assert len(errors) == 1 and "ring 7 has cells with no ring-6" in errors[0]


def test_the_real_cull_fixture_still_validates_end_to_end(parent_sphere):  # noqa: F811
    """The change is inside ``Mesh.validate``, so the whole path is exercised.

    A unit test on the extracted function cannot see a wiring mistake in the
    caller -- wrong argument order, a stale variable, the pair arrays built
    from the wrong mask.
    """

    mesh = cull_regional_mesh(parent_sphere, seed_cell=0, disk_hops=12)
    assert mesh.validate() is mesh


def test_a_torn_zone_is_refused_through_the_validator(parent_sphere):  # noqa: F811
    """And the caller's refusal reaches a reader through the real door.

    Ring 4 is relabelled to ring 6 on a single cell, which detaches it from
    ring 5.  Relabelling a cell also puts the DERIVED edge and vertex masks
    out of step with it, and those checks fire too -- correctly, and they are
    asserted alongside rather than suppressed, because a validator that
    reported only one consequence of a corruption would send a reader hunting
    the wrong array.
    """

    mesh = cull_regional_mesh(parent_sphere, seed_cell=0, disk_hops=12)
    mask = np.asarray(mesh.arrays["bdyMaskCell"]).copy()
    victim = int(np.flatnonzero(mask == 4)[0])
    mask[victim] = 6
    mesh.arrays["bdyMaskCell"] = mask
    with pytest.raises(MeshValidationError) as failure:
        mesh.validate()
    message = str(failure.value)
    assert "has cells with no ring-" in message
    assert "torn or renumbered zone" in message


def test_the_retired_check_would_have_refused_the_registered_cull_class():
    """The evidence for retiring it, kept where a reader will meet it.

    ``d070``'s measured widths grow 6.31 -> 7.71 km across the seven rings, so
    the shell's perimeter grows 11.6 per cent while its population falls 8.7
    per cent.  Both numbers are in this assertion so nobody has to take the
    receipt's word for which one describes a shell.
    """

    counts = np.asarray(SHRINKING[1:], np.float64)
    widths = np.asarray([6.31, 6.504, 6.715, 6.988, 7.219, 7.476, 7.71])
    perimeter = counts * widths
    assert counts[-1] < counts[0], "the populations shrink"
    assert perimeter[-1] > perimeter[0], "and the shell still grows"
    assert perimeter[-1] / perimeter[0] == pytest.approx(1.116, abs=0.005)
    assert counts[-1] / counts[0] == pytest.approx(0.913, abs=0.005)
