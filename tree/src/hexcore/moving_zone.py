"""Frozen-far-field moving-region members and their changed-zone carry maps.

A move re-centres the refinement regions of a variable-resolution global mesh.
Outside the cells whose requested spacing actually changes, the new member keeps
every generator point of the previous member, so those Voronoi cells, their
static fields, vertical coordinate and state are carried unchanged. Only the
changed band is regenerated (seeded from the previous member's points carried
by the move rotation, then Lloyd-relaxed against the new resolution request
with every outside generator fixed) and only that band is remapped.

The resolution request is the same JSON spec the Rust generator reads; this
module evaluates the identical tanh-ramp spacing field for caps. CPU only.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_229.0  # rw-mpas mesh/geom.rs EARTH_RADIUS_M


def unit_from_deg(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def arc(points, centre):
    """rw-mpas geom::arc: atan2(|a x b|, a.b), vectorised over points."""
    cross = np.cross(points, centre)
    return np.arctan2(np.linalg.norm(cross, axis=-1), points @ centre)


def spec_spacing_m(spec, points):
    """rw-mpas PreparedSpec::spacing_m for cap regions (metres)."""
    points = np.asarray(points, dtype=np.float64)
    back = 1.0 / (float(spec['background_km']) * 1000.0)
    inv = np.full(points.shape[0], back)
    for region in spec.get('regions', []):
        shape = region['shape']
        if shape.get('kind') != 'cap':
            raise ValueError(f"moving-zone spacing evaluates cap regions only, not {shape.get('kind')!r}")
        centre = unit_from_deg(*shape['center_deg'])
        radius = float(shape['radius_km']) * 1000.0 / EARTH_RADIUS_M
        if 'transition_km' in region:
            width = float(region['transition_km']) * 1000.0 / EARTH_RADIUS_M
        else:
            width = float(region['transition_cells']) * float(region['spacing_km']) * 1000.0 / EARTH_RADIUS_M
        fine = 1.0 / (float(region['spacing_km']) * 1000.0)
        s = arc(points, centre) - radius
        blend = 0.5 * (1.0 - np.tanh(s / width))
        np.maximum(inv, back + (fine - back) * blend, out=inv)
    return 1.0 / inv


def spec_finest_m(spec):
    return 1000.0 * min([float(spec['background_km'])] +
                        [float(r['spacing_km']) for r in spec.get('regions', [])])


def spec_density(spec, points):
    return (spec_finest_m(spec) / spec_spacing_m(spec, points)) ** 4


def spec_anchor(spec):
    """Centre of the finest cap: the point the move rotation carries."""
    regions = [r for r in spec.get('regions', []) if r['shape'].get('kind') == 'cap']
    if not regions:
        raise ValueError('moving-zone spec has no cap region to anchor the move')
    finest = min(regions, key=lambda r: float(r['spacing_km']))
    return unit_from_deg(*finest['shape']['center_deg'])


def rotation_between_vectors(a, b):
    c = float(a @ b)
    if c < -1 + 1e-10:
        raise ValueError('antipodal anchors have no unique minimal rotation')
    v = np.cross(a, b)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _stereographic(points, centre):
    """Circle-preserving projection about `centre`; spherical Delaunay = planar Delaunay."""
    helper = np.array([0.0, 0.0, 1.0]) if abs(centre[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(helper, centre))
    e2 = np.cross(centre, e1)
    denominator = 1.0 + points @ centre
    if np.any(denominator < 0.2):
        raise ValueError('changed band reaches too close to the projection antipode')
    return np.stack((2 * (points @ e1) / denominator, 2 * (points @ e2) / denominator), axis=-1)


def _voronoi_centroids(points, free, spec):
    """Density-weighted Voronoi centroids of `points[free]` (chordal sub-triangle quadrature)."""
    from scipy.spatial import Delaunay
    centre = _unit(points[free].mean(axis=0))
    triangles = Delaunay(_stereographic(points, centre)).simplices
    a, b, c = (points[triangles[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    orient = np.sign(np.einsum('ij,ij->i', normal, a + b + c))
    circum = _unit(normal * orient[:, None])
    count = len(points)
    weight = np.zeros(count)
    moment = np.zeros((count, 3))
    for k in range(3):
        owner = triangles[:, k]
        p = points[owner]
        q = points[triangles[:, (k + 1) % 3]]
        r = points[triangles[:, (k + 2) % 3]]
        mq, mr = _unit(p + q), _unit(p + r)
        for x, y in ((mq, circum), (circum, mr)):
            area = 0.5 * orient * np.einsum('ij,ij->i', np.cross(x - p, y - p), _unit(p + x + y))
            g = (p + x + y) / 3.0
            w = spec_density(spec, _unit(g)) * area
            weight += np.bincount(owner, weights=w, minlength=count)
            for axis in range(3):
                moment[:, axis] += np.bincount(owner, weights=w * g[:, axis], minlength=count)
    ids = np.flatnonzero(free)
    if np.any(weight[ids] <= 0):
        raise ValueError('a relaxed generator has a non-positive Voronoi weight')
    return _unit(moment[ids] / weight[ids, None])


def _circumcentres(points, simplices):
    a, b, c = (points[simplices[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    return _unit(normal * np.sign(np.einsum('ij,ij->i', normal, a + b + c))[:, None])


def _quad_flags(points, free, floor):
    """Near-cocircular Delaunay quads (dvEdge/dcEdge < floor) and under-coordinated band generators."""
    from scipy.spatial import Delaunay
    tri = Delaunay(_stereographic(points, _unit(points[free].mean(axis=0))))
    simplices, neighbours = tri.simplices, tri.neighbors
    circum = _circumcentres(points, simplices)
    every = np.arange(len(simplices))
    quads, qs, touched = [], [], []
    for m in range(3):
        u = neighbours[:, m]
        ok = u > every
        t, u = every[ok], u[ok]
        i, j, k = simplices[t, (m + 1) % 3], simplices[t, (m + 2) % 3], simplices[t, m]
        l = simplices[u].sum(axis=1) - i - j
        q = np.linalg.norm(circum[t] - circum[u], axis=1) / np.linalg.norm(points[i] - points[j], axis=1)
        touch = free[i] | free[j] | free[k] | free[l]
        touched.append(q[touch])
        sel = touch & (q < floor)
        quads.append(np.stack((i[sel], j[sel], k[sel], l[sel]), axis=1))
        qs.append(q[sel])
    quads, qs, touched = np.concatenate(quads), np.concatenate(qs), np.concatenate(touched)
    order = np.argsort(qs)
    edges = np.unique(np.sort(np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]]),
                              axis=1), axis=0)
    degree = np.bincount(edges.ravel(), minlength=len(points))
    # Generators adjacent to a band generator have complete local rings too.
    near = free.copy()
    touching = free[edges[:, 0]] | free[edges[:, 1]]
    near[edges[touching].ravel()] = True
    # Voronoi cell areas (chordal sub-triangles), for the local fill ratio.
    area = np.zeros(len(points))
    a, b, c = (points[simplices[:, k]] for k in range(3))
    orient = np.sign(np.einsum('ij,ij->i', np.cross(b - a, c - a), a + b + c))
    for k in range(3):
        owner = simplices[:, k]
        p, q2, r = points[owner], points[simplices[:, (k + 1) % 3]], points[simplices[:, (k + 2) % 3]]
        for x, y in ((_unit(p + q2), circum), (circum, _unit(p + r))):
            part = 0.5 * orient * np.einsum('ij,ij->i', np.cross(x - p, y - p), _unit(p + x + y))
            area += np.bincount(owner, weights=part, minlength=len(points))
    return dict(quads=quads[order], q=qs[order], min_q=float(touched.min()) if len(touched) else 1.0,
                low_coordination=np.flatnonzero(near & (degree < 5)), area=area)


def _apply_surgery(points, free, origin, flags, spec, memory=None):
    """Count edits on near-cocircular quads (after rw-mpas mesh/surgery.rs S1/S2, simplified):
    overdense quads (local fill ratio < 1) lose their most crowded band generator, underdense
    quads gain one at the midpoint of their long diagonal. Only band generators are removed."""
    delete, insert, claimed = set(), [], []
    items = [tuple(int(v) for v in quad) for quad in flags['quads']]
    items += [(int(v),) for v in flags['low_coordination']]
    for quad in items:
        if len(quad) == 1 and not free[quad[0]]:
            # A carried generator left under-coordinated by the band joins the band
            # and is deleted like a band defect (rw-mpas coordination clause).
            free[quad[0]] = True
        movable = [v for v in quad if free[v]]
        if not movable:
            continue
        centre = _unit(points[list(quad)].mean(axis=0))
        h = float(spec_spacing_m(spec, centre[None, :])[0]) / EARTH_RADIUS_M
        if any(np.linalg.norm(centre - other) < 3 * h for other in claimed):
            continue
        claimed.append(centre)
        if len(quad) == 1:
            delete.add(quad[0])
            continue
        # Local fill ratio phi = mean(A_c / A_target) over the quad's cells (rw-mpas surgery.rs).
        h_cells = spec_spacing_m(spec, points[list(quad)]) / EARTH_RADIUS_M
        ratios = flags['area'][list(quad)] / (np.sqrt(3) / 2 * h_cells ** 2)
        kind = 'delete' if ratios.mean() < 1.0 else 'insert'
        if memory is not None:
            # A site that two same-kind ops did not cure gets the other op (rw-mpas forced swap).
            site = next((m for m in memory if np.linalg.norm(m['centre'] - centre) < 2 * h), None)
            if site is None:
                site = dict(centre=centre, kinds=[])
                memory.append(site)
            if len(site['kinds']) >= 2 and site['kinds'][-1] == site['kinds'][-2] == kind:
                kind = 'insert' if kind == 'delete' else 'delete'
            site['kinds'].append(kind)
        if kind == 'delete':
            crowd = sorted((ratios[n], v) for n, v in enumerate(quad) if free[v])
            delete.add(crowd[0][1])
        else:
            i, j, k, l = (points[v] for v in quad)
            # (i, j) is the collapsed dual edge's own (short) diagonal; insert on the long one.
            insert.append(_unit(k + l))
    keep = np.ones(len(points), dtype=bool)
    keep[list(delete)] = False
    released = [int(origin[v]) for v in delete if origin[v] >= 0]
    # A carried generator that was offered but not deleted stays carried and fixed.
    free[(origin >= 0) & ~np.isin(np.arange(len(points)), list(delete))] = False
    points, free, origin = points[keep], free[keep], origin[keep]
    if insert:
        points = np.concatenate((np.asarray(insert), points), axis=0)
        free = np.concatenate((np.ones(len(insert), dtype=bool), free))
        origin = np.concatenate((np.full(len(insert), -1, dtype=np.int64), origin))
    return points, free, origin, dict(deleted=len(delete), inserted=len(insert),
                                      released_carried=released), np.asarray(claimed)


@dataclass
class MemberCentres:
    centres: np.ndarray        # (n, 3) unit vectors, carried cells first in source order
    density: np.ndarray        # meshDensity for each centre
    carried_source: np.ndarray  # source cell index for carried centres (len = n_carried)
    receipt: dict


def build_member_centres(prev_centres, prev_density, cells_on_cell, n_edges_on_cell, prev_spec, new_spec, *,
                         spacing_tolerance=5e-3, halo_rings=2, fixed_rings=3, tolerance=1e-3,
                         sweeps=800, min_sweeps=40, omega=1.4, quality_floor=0.04,
                         surgery_rounds=20, surgery_sweeps=8, admission_floor=0.02,
                         progress=None):
    """Generator points of the next member: unchanged outside the changed band."""
    began = time.perf_counter()
    prev = _unit(np.asarray(prev_centres, dtype=np.float64))
    n = len(prev)
    h_old = spec_spacing_m(prev_spec, prev)
    h_new = spec_spacing_m(new_spec, prev)
    changed = np.abs(h_new / h_old - 1.0) > spacing_tolerance
    neighbours = np.asarray(cells_on_cell, dtype=np.int64)
    counts = np.asarray(n_edges_on_cell, dtype=np.int64)
    if neighbours.min() >= 1 and neighbours.max() == n:
        neighbours = neighbours - 1
    slot = np.arange(neighbours.shape[1])[None, :] < counts[:, None]

    def dilate(mask, rings):
        mask = mask.copy()
        for _ in range(rings):
            grown = mask.copy()
            hit = mask[np.where(slot, neighbours, 0)] & slot
            grown |= hit.any(axis=1)
            mask = grown
        return mask

    band = dilate(changed, halo_rings)
    if not band.any():
        raise ValueError('the move changes no requested spacing beyond the tolerance; nothing to rebuild')
    fixed = dilate(band, fixed_rings) & ~band
    rotation = rotation_between_vectors(spec_anchor(prev_spec), spec_anchor(new_spec))
    from scipy.spatial import cKDTree
    tree = cKDTree(prev)
    moved = prev @ rotation.T
    _, owner = tree.query(moved)
    seeds = moved[band[owner]]
    # Never seed on top of a fixed generator: drop seeds closer than 0.35 h to one.
    fixed_ids = np.flatnonzero(fixed | ~band)
    fixed_tree = cKDTree(prev[fixed_ids])
    distance, _ = fixed_tree.query(seeds)
    h_seed = spec_spacing_m(new_spec, seeds) / EARTH_RADIUS_M
    seeds = seeds[distance > 0.35 * h_seed]
    expected_ratio = float(np.mean((h_old[band] / h_new[band]) ** 2))
    # The rotation overlaps fixed generators on the trailing seam (dropped above) and opens holes
    # on the leading seam. Refill the holes from the previous band generators, largest hole first,
    # so the band keeps its expected count (otherwise every move drains ~0.1 % of the band).
    deficit = int(round(band.sum() * expected_ratio)) - len(seeds)
    filled = 0
    if deficit > 0:
        occupied = cKDTree(np.concatenate((seeds, prev[fixed_ids]), axis=0))
        candidates = prev[band]
        gap, _ = occupied.query(candidates)
        h_candidate = spec_spacing_m(new_spec, candidates) / EARTH_RADIUS_M
        score = gap / h_candidate
        chosen = []
        for index in np.argsort(-score):
            if score[index] < 0.5 or len(chosen) >= deficit:
                break
            point = candidates[index]
            if chosen and np.min(np.linalg.norm(np.asarray(chosen) - point, axis=1)) < 0.5 * h_candidate[index]:
                continue
            chosen.append(point)
        if chosen:
            seeds = np.concatenate((seeds, np.asarray(chosen)), axis=0)
            filled = len(chosen)
    # Baseline: the previous member's own band generators, measured by this same
    # centroid estimator against the previous request. The new band stops when it
    # is at least as relaxed as the band it replaces (or meets `tolerance`).
    old_points = np.concatenate((prev[band], prev[fixed]), axis=0)
    old_free = np.zeros(len(old_points), dtype=bool)
    old_free[:int(band.sum())] = True
    baseline_delta = np.linalg.norm(_voronoi_centroids(old_points, old_free, prev_spec) - prev[band], axis=1) / (
        h_old[band] / EARTH_RADIUS_M)
    baseline = float(baseline_delta.mean())
    baseline_max = float(baseline_delta.max())
    stop = max(float(tolerance), baseline)
    stop_max = max(4 * float(tolerance), baseline_max)
    if progress is not None:
        progress(dict(previous_band_mean_delta_over_h=baseline, previous_band_max_delta_over_h=baseline_max,
                      band_cells=int(band.sum()), seeds=int(len(seeds))))
    points = np.concatenate((seeds, prev[fixed]), axis=0)
    origin = np.concatenate((np.full(len(seeds), -1, dtype=np.int64), np.flatnonzero(fixed)))
    free = np.zeros(len(points), dtype=bool)
    free[:len(seeds)] = True
    history = []
    surgery = []

    def relax(points, free, minimum):
        for sweep in range(int(sweeps)):
            h_free = spec_spacing_m(new_spec, points[free]) / EARTH_RADIUS_M
            centroid = _voronoi_centroids(points, free, new_spec)
            current = points[free]
            delta = np.linalg.norm(centroid - current, axis=1) / h_free
            history.append((float(delta.mean()), float(delta.max())))
            if progress is not None and len(history) % 10 == 1:
                progress(dict(sweep=len(history) - 1, mean_delta_over_h=history[-1][0],
                              max_delta_over_h=history[-1][1], seconds=time.perf_counter() - began))
            if sweep >= minimum and delta.mean() <= stop and delta.max() <= stop_max:
                return points
            points[free] = _unit(current + omega * (centroid - current))
        raise ValueError(f'changed-band relaxation did not reach mean delta/h {stop} in {sweeps} sweeps '
                         f'(last {history[-1][0]:.3g}, max {history[-1][1]:.3g})')

    points = relax(points, free, min_sweeps)
    memory = []
    for round_index in range(int(surgery_rounds) + 1):
        flags = _quad_flags(points, free, quality_floor)
        surgery.append(dict(round=round_index, flagged_quads=int(len(flags['q'])), min_q=flags['min_q'],
                            low_coordination=int(len(flags['low_coordination']))))
        if progress is not None:
            progress(dict(surgery=surgery[-1]))
        if not len(flags['q']) and not len(flags['low_coordination']):
            break
        if round_index == int(surgery_rounds) and flags['min_q'] >= admission_floor and \
                not len(flags['low_coordination']):
            # Rounds spent with every edge above the model's admission floor: accept, recorded.
            surgery[-1]['accepted_above_admission_floor'] = admission_floor
            break
        if round_index == int(surgery_rounds):
            raise ValueError(f'changed-band surgery left {len(flags["q"])} near-cocircular quads '
                             f'(min dv/dc {flags["min_q"]:.3g}) after {surgery_rounds} rounds')
        points, free, origin, edits, sites = _apply_surgery(points, free, origin, flags, new_spec, memory)
        for index in edits['released_carried']:
            band[index] = True
        surgery[-1].update(edits)
        if len(sites):
            # Local polish (plain Lloyd) of band generators within four spacings of an edit.
            near_distance, _ = cKDTree(sites).query(points)
            h_points = spec_spacing_m(new_spec, points) / EARTH_RADIUS_M
            local = free & (near_distance < 4 * h_points)
            for _ in range(int(surgery_sweeps)):
                points[local] = _voronoi_centroids(points, local, new_spec)
    keep = np.flatnonzero(~band)
    new_points = points[free]
    h_free = spec_spacing_m(new_spec, new_points) / EARTH_RADIUS_M
    # Spatially coherent order for the regenerated generators.
    lat = np.arcsin(np.clip(new_points[:, 2], -1, 1))
    lon = np.arctan2(new_points[:, 1], new_points[:, 0])
    order = np.lexsort((lon, np.round(lat / (np.median(h_free) * 8))))
    new_points = new_points[order]
    centres = np.concatenate((prev[keep], new_points), axis=0)
    density = np.concatenate((np.asarray(prev_density, dtype=np.float64)[keep],
                              spec_density(new_spec, new_points)))
    receipt = dict(
        method='frozen-far-field member: previous generators kept outside the changed band; band seeded by '
               'the anchor rotation and Lloyd-relaxed against the new spec with outside generators fixed',
        spacing_tolerance=spacing_tolerance, halo_rings=halo_rings, fixed_rings=fixed_rings,
        source_cells=n, changed_cells=int(changed.sum()), band_cells_removed=int(band.sum()),
        band_cells_generated=int(len(new_points)), carried_cells=int(len(keep)), cells=int(len(centres)),
        band_count_ratio_generated_over_removed=float(len(new_points) / band.sum()), seam_holes_refilled=filled,
        band_expected_count_ratio=expected_ratio,
        rotation_anchor_arc_km=float(arc(spec_anchor(prev_spec)[None, :], spec_anchor(new_spec))[0] *
                                     EARTH_RADIUS_M / 1000),
        relaxation_sweeps=len(history), relaxation_mean_delta_over_h=history[-1][0],
        relaxation_max_delta_over_h=history[-1][1], relaxation_tolerance=tolerance, omega=omega,
        relaxation_stop=stop, relaxation_stop_max=stop_max, previous_band_mean_delta_over_h=baseline,
        previous_band_max_delta_over_h=baseline_max, surgery=surgery, quality_floor=quality_floor,
        band_spacing_km_min=float(h_new[band].min() / 1000), band_spacing_km_max=float(h_new[band].max() / 1000),
        seconds=time.perf_counter() - began)
    return MemberCentres(centres, density, keep, receipt)


def corridor_refusal(member_spec, anchor, outer_radius_km, fixed_spacing_km, *, tolerance=0.06, samples=72):
    """None when the stitched region about `anchor` lies inside the fixed corridor, else the reason.

    Guard for the breakage it prevents: a region stitched in where the fixed rings have already
    ramped toward the background would put its outermost (fixed-ring spacing) cells against
    coarser frame cells with no ramp between them, a spacing jump the seam relaxation cannot
    grade (the defect class of the 2026-10-12 far-field spike, where the changed zone reached a
    coarse ramp over steep terrain).
    """
    centre = np.asarray(anchor, dtype=np.float64)
    helper = np.array([0.0, 0.0, 1.0]) if abs(centre[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(helper, centre))
    e2 = np.cross(centre, e1)
    angle = float(outer_radius_km) * 1000.0 / EARTH_RADIUS_M
    theta = np.linspace(0.0, 2 * np.pi, int(samples), endpoint=False)
    ring = (np.cos(angle) * centre[None, :] +
            np.sin(angle) * (np.cos(theta)[:, None] * e1[None, :] + np.sin(theta)[:, None] * e2[None, :]))
    spacing = spec_spacing_m(member_spec, ring) / 1000.0
    worst = float(spacing.max())
    if worst > float(fixed_spacing_km) * (1.0 + tolerance):
        return (f'region centre is outside the fixed corridor: requested spacing reaches {worst:.2f} km on the '
                f'{outer_radius_km:.0f} km stitch circle (fixed ring {fixed_spacing_km:g} km, tolerance {tolerance:g})')
    return None


def build_insert_member(frame_centres, template_centres, template_anchor, new_anchor, member_spec, *,
                        inner_radius_km, outer_radius_km, fixed_spacing_km, guard_rows=4,
                        tolerance=2e-3, max_tolerance=2e-2, accept_tolerance=1e-2, sweeps=300, min_sweeps=10,
                        omega=1.4, quality_floor=0.04, surgery_rounds=20, surgery_sweeps=8,
                        admission_floor=0.02, progress=None):
    """Generator points of a member stitched from two generator meshes.

    FRAME is the generator's mesh of the fixed corridor rings alone (no fine region); TEMPLATE is the
    generator's mesh of the full request about `template_anchor`. The member keeps every frame
    generator farther than `outer_radius_km` from `new_anchor`, takes the template's generators
    within `inner_radius_km` of the template anchor as a RIGID rotation onto `new_anchor` (so every
    fine cell is the generator's own cell, edge for edge), and relaxes only the seam annulus between
    them, which lies wholly in the fixed ring's uniform spacing. No member is built from another
    member, so edge quality cannot degrade along a chain, and every member shares the frame's far
    field exactly.
    """
    began = time.perf_counter()
    from scipy.spatial import cKDTree
    frame = _unit(np.asarray(frame_centres, dtype=np.float64))
    template = _unit(np.asarray(template_centres, dtype=np.float64))
    source = np.asarray(template_anchor, dtype=np.float64)
    target = np.asarray(new_anchor, dtype=np.float64)
    refusal = corridor_refusal(member_spec, target, outer_radius_km, fixed_spacing_km)
    if refusal:
        raise ValueError(refusal)
    r_in = float(inner_radius_km) * 1000.0 / EARTH_RADIUS_M
    r_out = float(outer_radius_km) * 1000.0 / EARTH_RADIUS_M
    h = float(fixed_spacing_km) * 1000.0 / EARTH_RADIUS_M
    template_arc = arc(template, source)
    inside = np.flatnonzero(template_arc <= r_in)
    rotation = rotation_between_vectors(source, target)
    insert = _unit(template[inside] @ rotation.T)
    insert_arc = template_arc[inside]
    frame_arc = arc(frame, target)
    keep = np.flatnonzero(frame_arc > r_out)
    seeds = frame[(frame_arc > r_in) & (frame_arc <= r_out)]
    if len(seeds):
        distance, _ = cKDTree(insert).query(seeds)
        seeds = seeds[distance > 0.6 * h]
    if not len(seeds):
        raise ValueError('the seam annulus holds no frame generator; widen it')
    fixed_insert = np.flatnonzero(insert_arc > r_in - guard_rows * h)
    fixed_frame = np.flatnonzero(frame_arc[keep] <= r_out + guard_rows * h)
    # origin: -1 a seam generator; i < len(insert) a fixed insert generator; len(insert) + j a fixed frame one
    points = np.concatenate((seeds, insert[fixed_insert], frame[keep][fixed_frame]), axis=0)
    origin = np.concatenate((np.full(len(seeds), -1, dtype=np.int64), fixed_insert.astype(np.int64),
                             len(insert) + fixed_frame.astype(np.int64)))
    free = np.zeros(len(points), dtype=bool)
    free[:len(seeds)] = True
    history = []
    accepted_loose = False
    for sweep in range(int(sweeps)):
        h_free = spec_spacing_m(member_spec, points[free]) / EARTH_RADIUS_M
        centroid = _voronoi_centroids(points, free, member_spec)
        current = points[free]
        delta = np.linalg.norm(centroid - current, axis=1) / h_free
        history.append((float(delta.mean()), float(delta.max())))
        if progress is not None and len(history) % 10 == 1:
            progress(dict(sweep=len(history) - 1, mean_delta_over_h=history[-1][0],
                          max_delta_over_h=history[-1][1], seconds=time.perf_counter() - began))
        if sweep >= min_sweeps and delta.mean() <= tolerance and delta.max() <= max_tolerance:
            break
        points[free] = _unit(current + omega * (centroid - current))
    else:
        if history[-1][0] > accept_tolerance:
            raise ValueError(f'seam relaxation did not reach mean delta/h {accept_tolerance} in {sweeps} sweeps '
                             f'(last {history[-1][0]:.3g}, max {history[-1][1]:.3g})')
        accepted_loose = True
    surgery, memory, released_insert, released_frame = [], [], [], []
    for round_index in range(int(surgery_rounds) + 1):
        flags = _quad_flags(points, free, quality_floor)
        surgery.append(dict(round=round_index, flagged_quads=int(len(flags['q'])), min_q=flags['min_q'],
                            low_coordination=int(len(flags['low_coordination']))))
        if progress is not None:
            progress(dict(surgery=surgery[-1]))
        if not len(flags['q']) and not len(flags['low_coordination']):
            break
        if round_index == int(surgery_rounds) and flags['min_q'] >= admission_floor and \
                not len(flags['low_coordination']):
            surgery[-1]['accepted_above_admission_floor'] = admission_floor
            break
        if round_index == int(surgery_rounds):
            raise ValueError(f'seam surgery left {len(flags["q"])} near-cocircular quads '
                             f'(min dv/dc {flags["min_q"]:.3g}) after {surgery_rounds} rounds')
        points, free, origin, edits, sites = _apply_surgery(points, free, origin, flags, member_spec, memory)
        for index in edits['released_carried']:
            if index < len(insert):
                released_insert.append(index)
            else:
                released_frame.append(index - len(insert))
        surgery[-1].update(edits)
        if len(sites):
            near_distance, _ = cKDTree(sites).query(points)
            h_points = spec_spacing_m(member_spec, points) / EARTH_RADIUS_M
            local = free & (near_distance < 4 * h_points)
            for _ in range(int(surgery_sweeps)):
                points[local] = _voronoi_centroids(points, local, member_spec)
    insert_keep = np.ones(len(insert), dtype=bool)
    insert_keep[released_insert] = False
    frame_keep = np.ones(len(keep), dtype=bool)
    frame_keep[released_frame] = False
    seam = points[free]
    lat = np.arcsin(np.clip(seam[:, 2], -1, 1))
    lon = np.arctan2(seam[:, 1], seam[:, 0])
    seam = seam[np.lexsort((lon, np.round(lat / (8 * h))))]
    centres = np.concatenate((frame[keep][frame_keep], insert[insert_keep], seam), axis=0)
    density = spec_density(member_spec, centres)
    receipt = dict(
        method="stitched member: frame generators kept outside the stitch circle, the template's fine region "
               "rigidly rotated onto the member centre, seam annulus Lloyd-relaxed between fixed neighbours",
        inner_radius_km=float(inner_radius_km), outer_radius_km=float(outer_radius_km),
        fixed_spacing_km=float(fixed_spacing_km), guard_rows=guard_rows,
        frame_cells=int(len(frame)), template_cells=int(len(template)),
        frame_cells_kept=int(frame_keep.sum()), insert_cells=int(insert_keep.sum()), seam_cells=int(len(seam)),
        seam_seeds=int(len(seeds)), released_insert=len(released_insert), released_frame=len(released_frame),
        cells=int(len(centres)),
        rotation_anchor_arc_km=float(arc(source[None, :], target)[0] * EARTH_RADIUS_M / 1000),
        relaxation_sweeps=len(history), relaxation_mean_delta_over_h=history[-1][0],
        relaxation_max_delta_over_h=history[-1][1], relaxation_tolerance=tolerance,
        relaxation_accepted_above_tolerance=accepted_loose, omega=omega, surgery=surgery,
        quality_floor=quality_floor, seconds=time.perf_counter() - began)
    return MemberCentres(centres, density, np.arange(int(frame_keep.sum())), receipt)


def write_centres(path, member, template_grid, *, spec=None):
    """Write the --from-centres request in the template's netCDF data model."""
    from netCDF4 import Dataset
    with Dataset(template_grid) as source, Dataset(path, 'w', format=source.data_model) as out:
        out.createDimension('nCells', len(member.centres))
        for index, name in enumerate(('xCell', 'yCell', 'zCell')):
            variable = out.createVariable(name, 'f8', ('nCells',))
            variable[:] = member.centres[:, index]
        variable = out.createVariable('meshDensity', 'f8', ('nCells',))
        variable[:] = member.density
        variable = out.createVariable('nominalMinDc', source.variables['nominalMinDc'].dtype, ())
        variable.assignValue(source.variables['nominalMinDc'][...])
        for name in source.ncattrs():
            if name.startswith('rw_mesh_') or name == 'file_id':
                continue
            out.setncattr(name, source.getncattr(name))
        out.setncattr('moving_zone_member', json.dumps(member.receipt, sort_keys=True))
        if spec is not None:
            out.setncattr('moving_zone_spec', json.dumps(spec, sort_keys=True))


# ---------------------------------------------------------------------------
# Changed-zone carry map between two members (controller side, CPU only).
# ---------------------------------------------------------------------------

def _grid_topology(path):
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        centres = np.stack([np.asarray(ds.variables[k][:], dtype=np.float64) for k in ('xCell', 'yCell', 'zCell')], -1)
        out = dict(centres=_unit(centres),
                   area=np.asarray(ds.variables['areaCell'][:], dtype=np.float64),
                   counts=np.asarray(ds.variables['nEdgesOnCell'][:], dtype=np.int64),
                   cells_on_cell=np.asarray(ds.variables['cellsOnCell'][:], dtype=np.int64) - 1,
                   cells_on_edge=np.asarray(ds.variables['cellsOnEdge'][:], dtype=np.int64) - 1,
                   edges_on_cell=np.asarray(ds.variables['edgesOnCell'][:], dtype=np.int64) - 1,
                   dv_edge=np.asarray(ds.variables['dvEdge'][:], dtype=np.float64),
                   dc_edge=np.asarray(ds.variables['dcEdge'][:], dtype=np.float64))
    return out


def _cell_fields(path, n_cells, names=None):
    """Every per-cell variable of a static/init file (first dimension nCells, optionally after Time)."""
    from netCDF4 import Dataset
    fields = {}
    with Dataset(path) as ds:
        for name, variable in ds.variables.items():
            if names is not None and name not in names:
                continue
            dims = variable.dimensions
            if dims and dims[0] == 'Time':
                if len(dims) < 2 or dims[1] != 'nCells':
                    continue
                values = np.asarray(variable[0])
            elif dims and dims[0] == 'nCells':
                values = np.asarray(variable[:])
            else:
                continue
            if values.shape[0] == n_cells and values.dtype.kind in 'fiu':
                fields[name] = values
    return fields


ZONE_INIT_FIELDS = ('zgrid', 'zz', 'zxu', 'dss', 'hx', 'rho_base', 'theta_base', 'ter',
                    'xland', 'landmask', 'ivgtyp', 'isltyp')


@dataclass
class ZoneMap:
    """Identity carry outside the changed zone; a closed conservative overlap inside it."""
    carried_target: np.ndarray   # target cell ids carried exactly
    carried_source: np.ndarray   # matching source cell ids
    zone_target: np.ndarray      # target cell ids remapped
    zone_source: np.ndarray      # source cell ids that donate to the zone
    edge_target: np.ndarray      # target edges carried exactly (faces of carried cells)
    edge_source: np.ndarray
    edge_sign: np.ndarray        # +1/-1: source normal orientation relative to target
    zone_overlap: object         # ConservativeMap over (zone_source -> zone_target), sub-indexed
    n_source: int
    n_target: int
    n_source_edges: int
    n_target_edges: int
    receipt: dict
    column_target: np.ndarray = None   # same polygon, changed vertical coordinate: vertical remap only
    column_source: np.ndarray = None

    @classmethod
    def build(cls, source_grid, target_grid, *, source_static=None, target_static=None,
              source_init=None, target_init=None, zgrid_tolerance_m=1e-3, progress=None):
        from scipy.spatial import cKDTree
        from .moving_remap import SphericalCells, ConservativeMap
        began = time.perf_counter()
        s, t = _grid_topology(source_grid), _grid_topology(target_grid)
        ns, nt = len(s['area']), len(t['area'])
        distance, match = cKDTree(s['centres']).query(t['centres'])
        matched = distance < 1e-10
        source_of = np.where(matched, match, -1)
        if len(np.unique(source_of[matched])) != int(matched.sum()):
            raise ValueError('two target generators match one source generator')
        # A carried cell keeps its generator, its whole neighbour set and its area:
        # its Voronoi polygon is then the same polygon.
        same = matched.copy()
        ids = np.flatnonzero(matched)
        src = source_of[ids]
        tslot = np.arange(t['cells_on_cell'].shape[1])[None, :] < t['counts'][ids][:, None]
        sslot = np.arange(s['cells_on_cell'].shape[1])[None, :] < s['counts'][src][:, None]
        neighbour_t = np.where(tslot, source_of[np.where(tslot, t['cells_on_cell'][ids], 0)], -2)
        neighbour_s = np.where(sslot, s['cells_on_cell'][src], -2)
        width = min(neighbour_t.shape[1], neighbour_s.shape[1])
        if neighbour_t.shape[1] != neighbour_s.shape[1]:
            extra_t = (neighbour_t[:, width:] != -2).any(axis=1) if neighbour_t.shape[1] > width else False
            extra_s = (neighbour_s[:, width:] != -2).any(axis=1) if neighbour_s.shape[1] > width else False
            same[ids] &= ~(extra_t | extra_s)
            neighbour_t, neighbour_s = neighbour_t[:, :width], neighbour_s[:, :width]
        same[ids] &= s['counts'][src] == t['counts'][ids]
        same[ids] &= np.all(np.sort(neighbour_t, axis=1) == np.sort(neighbour_s, axis=1), axis=1)
        # Generator and --from-centres members agree on areas to ~1e-9 (rounding); the model holds FP32.
        same[ids] &= np.abs(t['area'][ids] / s['area'][src] - 1) < 1e-8
        geometry_identical = int(same.sum())
        # Same polygon: the state is the same polygon mean. A changed vertical coordinate
        # (terrain smoothing reaches a few rings past the band) makes the column a COLUMN
        # cell: horizontal identity plus a vertical remap. Static differences on same
        # polygons are recorded (neighbour-stencil operators, sub-grid orography windows).
        column = np.zeros(nt, dtype=bool)
        static_differences = {}
        zgrid_shift = 0.0
        for label, (a_path, b_path), names in (('static', (source_static, target_static), None),
                                               ('init', (source_init, target_init), ZONE_INIT_FIELDS)):
            if a_path is None or b_path is None:
                continue
            a, b = _cell_fields(a_path, ns, names), _cell_fields(b_path, nt, names)
            ids = np.flatnonzero(same)
            for name in sorted(set(a) & set(b)):
                x, y = a[name][source_of[ids]], b[name][ids]
                if x.shape[1:] != y.shape[1:]:
                    continue
                difference = np.abs(x.astype(np.float64) - y.astype(np.float64)).reshape(len(ids), -1)
                difference = np.where(np.isnan(difference), 0.0, difference).max(axis=1)
                if name == 'zgrid' and label == 'init':
                    moved = difference > zgrid_tolerance_m
                    column[ids[moved]] = True
                    zgrid_shift = float(difference.max()) if len(difference) else 0.0
                if (difference > 0).any():
                    static_differences[f'{label}/{name}'] = dict(cells=int((difference > 0).sum()),
                                                                 abs_max=float(difference.max()))
        polygon_same = same.copy()
        same &= ~column
        column_t = np.flatnonzero(column)
        column_s = source_of[column_t]
        carried_t = np.flatnonzero(same)
        carried_s = source_of[carried_t]
        zone_t = np.flatnonzero(~polygon_same)
        mask_s = np.ones(ns, dtype=bool)
        mask_s[source_of[np.flatnonzero(polygon_same)]] = False
        zone_s = np.flatnonzero(mask_s)
        # Edges: every face of a carried cell is the same face in both members.
        key_s = np.sort(s['cells_on_edge'], axis=1)
        flat_s = key_s[:, 0] * ns + key_s[:, 1]
        order = np.argsort(flat_s)
        cot = t['cells_on_edge']
        a_t, b_t = cot[:, 0], cot[:, 1]
        carried_mask = polygon_same
        face = (carried_mask[a_t] | carried_mask[b_t]) & (source_of[a_t] >= 0) & (source_of[b_t] >= 0)
        if np.any((carried_mask[a_t] | carried_mask[b_t]) & ~face):
            raise ValueError('a carried cell has a face toward an unmatched generator')
        e_t = np.flatnonzero(face)
        sa, sb = source_of[a_t[e_t]], source_of[b_t[e_t]]
        key = np.minimum(sa, sb) * ns + np.maximum(sa, sb)
        position = np.minimum(np.searchsorted(flat_s, key, sorter=order), len(order) - 1)
        e_s = order[position]
        found = flat_s[e_s] == key
        if not found.all():
            raise ValueError(f'{int((~found).sum())} faces of carried cells have no source edge')
        sign = np.where(s['cells_on_edge'][e_s, 0] == sa, 1, -1).astype(np.int8)
        # Rounding-level geometry (1e-12 on the unit sphere): measure dv against the face's dc.
        dv_error = float(np.max(np.abs(t['dv_edge'][e_t] - s['dv_edge'][e_s]) / s['dc_edge'][e_s])) if len(e_t) else 0.0
        dc_error = float(np.max(np.abs(t['dc_edge'][e_t] / s['dc_edge'][e_s] - 1))) if len(e_t) else 0.0
        if dv_error > 1e-8 or dc_error > 1e-10:
            raise ValueError(f'carried faces differ in length: dv {dv_error}, dc {dc_error}')
        # Closed conservative overlap on the changed zone only.
        source_cells, target_cells = SphericalCells.read(source_grid), SphericalCells.read(target_grid)

        def subset(cells, chosen):
            return SphericalCells(cells.centers[chosen], cells.vertices[chosen], cells.counts[chosen],
                                  cells.area[chosen], cells.radius_m)
        zone_map = ConservativeMap.build(subset(source_cells, zone_s), subset(target_cells, zone_t),
                                         progress=progress)
        zone_area_t = float(target_cells.area[zone_t].sum())
        receipt = dict(
            method='identity carry of unchanged cells and of every face of a carried cell; closed first-order '
                   'spherical overlap on the changed zone only',
            source_cells=ns, target_cells=nt, matched_generators=int(matched.sum()),
            geometry_identical_cells=geometry_identical, same_polygon_field_differences=static_differences,
            column_cells=int(len(column_t)), column_zgrid_shift_abs_max_m=zgrid_shift,
            zgrid_tolerance_m=zgrid_tolerance_m,
            carried_cells=int(len(carried_t)), zone_target_cells=int(len(zone_t)),
            zone_source_cells=int(len(zone_s)), carried_faces=int(len(e_t)),
            zone_area_fraction=zone_area_t / float(target_cells.area.sum()),
            zone_area_relative_difference=float(source_cells.area[zone_s].sum() / zone_area_t - 1),
            carried_face_dv_relative_max=dv_error, carried_face_dc_relative_max=dc_error,
            zone_operator=zone_map.receipt, seconds=time.perf_counter() - began,
            source_grid=str(source_grid), target_grid=str(target_grid),
            source_sha256=source_cells.fingerprint(), target_sha256=target_cells.fingerprint())
        return cls(carried_t, carried_s, zone_t, zone_s, e_t, e_s, sign, zone_map, ns, nt,
                   len(s['cells_on_edge']), len(cot), receipt, column_t, column_s)

    def save(self, path):
        m = self.zone_overlap.overlap
        with open(path, 'wb') as stream:
            np.savez(stream, carried_target=self.carried_target, carried_source=self.carried_source,
                     zone_target=self.zone_target, zone_source=self.zone_source, edge_target=self.edge_target,
                     edge_source=self.edge_source, edge_sign=self.edge_sign, data=m.data, indices=m.indices,
                     indptr=m.indptr, shape=np.asarray(m.shape), source_area=self.zone_overlap.source_area,
                     target_area=self.zone_overlap.target_area,
                     sizes=np.asarray([self.n_source, self.n_target, self.n_source_edges, self.n_target_edges]),
                     column_target=self.column_target, column_source=self.column_source,
                     operator_receipt=json.dumps(self.zone_overlap.receipt, sort_keys=True),
                     receipt=json.dumps(self.receipt, sort_keys=True))

    @classmethod
    def load(cls, path):
        from scipy.sparse import csr_matrix
        from .moving_remap import ConservativeMap
        with np.load(path, allow_pickle=False) as d:
            overlap = ConservativeMap(csr_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape'])),
                                      d['source_area'], d['target_area'], json.loads(str(d['operator_receipt'])))
            sizes = [int(v) for v in d['sizes']]
            return cls(d['carried_target'], d['carried_source'], d['zone_target'], d['zone_source'],
                       d['edge_target'], d['edge_source'], d['edge_sign'], overlap, *sizes,
                       json.loads(str(d['receipt'])), d['column_target'], d['column_source'])

    @property
    def remapped_target(self):
        return np.concatenate((self.zone_target, self.column_target))

    @property
    def remapped_source(self):
        return np.concatenate((self.zone_source, self.column_source))

    def remapped_operator(self, zone_physical, source_area, target_area):
        """Horizontal operator on zone + column cells: zone overlap block plus column identity."""
        from scipy.sparse import csr_matrix, bmat
        from .moving_remap import ConservativeMap
        nz_t, nz_s, nc = len(self.zone_target), len(self.zone_source), len(self.column_target)
        identity = csr_matrix((np.asarray(target_area, dtype=np.float64)[self.column_target],
                               (np.arange(nc), np.arange(nc))), shape=(nc, nc))
        overlap = bmat([[zone_physical.overlap, None], [None, identity]], format='csr')
        source = np.concatenate((zone_physical.source_area,
                                 np.asarray(source_area, dtype=np.float64)[self.column_source]))
        target = np.concatenate((zone_physical.target_area,
                                 np.asarray(target_area, dtype=np.float64)[self.column_target]))
        receipt = dict(zone_physical.receipt, column_identity_cells=nc,
                       column_area_relative_max=float(np.abs(target[nz_t:] / source[nz_s:] - 1).max()) if nc else 0.0)
        return ConservativeMap(overlap, source, target, receipt)


class CompositeMap:
    """Global operator view for column physics: exact copy outside, overlap inside the zone."""

    def __init__(self, zone, zone_physical, source_area, target_area):
        from scipy.sparse import csr_matrix
        self.zone = zone
        self.sub = zone_physical
        self.source_area = np.asarray(source_area, dtype=np.float64)
        self.target_area = np.asarray(target_area, dtype=np.float64)
        coo = zone_physical.overlap.tocoo()
        rows = np.concatenate((zone.carried_target, zone.remapped_target[coo.row]))
        cols = np.concatenate((zone.carried_source, zone.remapped_source[coo.col]))
        data = np.concatenate((self.target_area[zone.carried_target], coo.data))
        self.overlap = csr_matrix((data, (rows, cols)), shape=(zone.n_target, zone.n_source))
        self.receipt = dict(zone.receipt, zone_operator=zone_physical.receipt)

    def intensive(self, values):
        a = np.asarray(values)
        if a.shape[-1] != self.zone.n_source or not np.all(np.isfinite(a)):
            raise ValueError('source cell extent or finite-value gate failed')
        out = np.empty((*a.shape[:-1], self.zone.n_target), dtype=np.float64)
        out[..., self.zone.carried_target] = a[..., self.zone.carried_source]
        out[..., self.zone.remapped_target] = self.sub.intensive(a[..., self.zone.remapped_source])
        return out
