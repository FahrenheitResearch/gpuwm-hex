"""First-order conservative remapping between convex spherical MPAS cells.

Overlap polygons have great-circle edges. No longitude projection, nearest
interpolation, global scalar rescaling, or damping is used for conserved fields.
The sparse operator is checked against both meshes' declared areas before use.
This module is CPU-only and does not import a CUDA runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
import sys
import datetime
from contextlib import contextmanager

import numpy as np
from scipy.sparse import csr_matrix, diags, bmat
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

DONOR_AREA_RTOL = 2e-14
# Row closure controls constant-field reproduction, while donor closure
# controls every conserved integral. Keep row error below one thousandth
# of FP32 relative precision, independently of the raw geometry gate.
RECIPIENT_AREA_RTOL = min(1e-10, np.finfo(np.float32).eps / 1024)


def _accelerator():
    path = os.environ.get('GPUWM_HEX_SPHERICAL_OVERLAP_LIB')
    if not path:
        return None
    import ctypes
    from numpy.ctypeslib import ndpointer
    library = ctypes.CDLL(path)
    f64 = ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    i64 = ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    function = library.spherical_overlaps
    function.argtypes = [f64, ctypes.c_int, i64, i64, ctypes.c_int,
                         f64, ctypes.c_int, f64, ctypes.c_double, f64]
    function.restype = ctypes.c_int
    return function


def sphere_xyz(lat, lon):
    lat, lon = np.asarray(lat), np.asarray(lon)
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def spherical_area(vertices):
    """Unit-sphere area by signed, stable spherical triangle excess."""
    if len(vertices) < 3:
        return 0.0
    a = vertices[0]
    b, c = vertices[1:-1], vertices[2:]
    # det(a,b,c) == det(a,b-a,c-a). At LES scales the original cross
    # subtracts O(1) products to recover O(dx/radius)^2 area. Chord
    # differences retain the small triangle before multiplying it out.
    determinant = np.einsum('ij,ij->i', np.cross(b-a, c-a), a[None, :])
    denominator = 1.0 + b @ a + c @ a + np.einsum('ij,ij->i', b, c)
    return abs(float(np.sum(2.0 * np.arctan2(determinant, denominator))))


def overlap_polygon(subject, clip, center):
    """Clip a convex unit-sphere polygon against oriented great-circle planes."""
    poly = np.asarray(subject, dtype=np.float64)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        normal = np.cross(a, b-a)
        normal /= np.linalg.norm(normal)
        if np.dot(normal, center) < 0:
            normal = -normal
        if len(poly) == 0:
            break
        dots = poly @ normal
        output = []
        for i in range(len(poly)):
            p, q = poly[i-1], poly[i]
            dp, dq = dots[i-1], dots[i]
            pin, qin = dp >= 0.0, dq >= 0.0
            if pin != qin:
                # A linear chord/plane intersection normalizes to the exact
                # great-circle arc/plane intersection for these small cells.
                x = (dq * p - dp * q) / (dq - dp)
                x /= np.linalg.norm(x)
                output.append(x)
            if qin:
                output.append(q)
        poly = np.asarray(output, dtype=np.float64).reshape(-1, 3)
    return poly


@dataclass
class SphericalCells:
    centers: np.ndarray
    vertices: np.ndarray
    counts: np.ndarray
    area: np.ndarray
    radius_m: float

    @classmethod
    def read(cls, path):
        from netCDF4 import Dataset
        with Dataset(path) as ds:
            def xyz(suffix):
                a = np.stack([np.asarray(ds.variables[n + suffix][:], dtype=np.float64) for n in 'xyz'], axis=-1)
                return a / np.linalg.norm(a, axis=-1, keepdims=True)
            centers, vertex = xyz('Cell'), xyz('Vertex')
            indices = np.asarray(ds.variables['verticesOnCell'][:], dtype=np.int64) - 1
            counts = np.asarray(ds.variables['nEdgesOnCell'][:], dtype=np.int64)
            area = np.asarray(ds.variables['areaCell'][:], dtype=np.float64)
            radius = float(ds.getncattr('sphere_radius')) if 'sphere_radius' in ds.ncattrs() else 6371220.0
            return cls(centers, vertex[np.maximum(indices, 0)], counts, area, radius)

    def validate(self):
        if self.centers.shape != (len(self.area), 3) or self.vertices.shape[0] != len(self.area):
            raise ValueError('cell geometry shapes differ')
        if not np.all(np.isfinite(self.area)) or np.any(self.area <= 0) or np.any(self.counts < 3):
            raise ValueError('invalid cell area or vertex count')
        for i, n in enumerate(self.counts):
            p = self.vertices[i, :n]
            normals = np.cross(p, np.roll(p, -1, axis=0))
            sign = np.sign(normals @ self.centers[i])
            if np.any(sign == 0) or np.any(sign != sign[0]):
                raise ValueError(f'cell {i} is not consistently ordered and convex')
        return self

    def angular_radius(self):
        angles = np.arccos(np.clip(np.einsum('ijk,ik->ij', self.vertices, self.centers), -1, 1))
        angles[np.arange(self.vertices.shape[1])[None, :] >= self.counts[:, None]] = 0
        return angles.max(axis=1)

    def fingerprint(self):
        digest = hashlib.sha256()
        for value in (self.centers, self.vertices, self.counts, self.area):
            digest.update(str((value.shape, str(value.dtype))).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(repr(self.radius_m).encode())
        return digest.hexdigest()


def _overlap_rows(source, target, sr, tr, groups, accelerator, j0, j1):
    source_vertices = np.ascontiguousarray(source.vertices, dtype=np.float64)
    source_counts = np.ascontiguousarray(source.counts, dtype=np.int64)
    rows, cols, values = [], [], []
    tested = 0
    for j in range(j0, j1):
        q = target.vertices[j, :target.counts[j]]
        for ids, tree, bound in groups:
            radius = 2 * math.sin(min(math.pi, tr[j] + bound + 1e-13) / 2)
            candidates = ids[tree.query_ball_point(target.centers[j], radius)]
            chords = np.linalg.norm(source.centers[candidates]-target.centers[j], axis=1)
            candidates = candidates[chords <= 2*np.sin(np.minimum(math.pi, sr[candidates]+tr[j]+1e-13)/2)]
            if accelerator is not None:
                candidates = np.ascontiguousarray(candidates, dtype=np.int64)
                areas = np.empty(len(candidates), dtype=np.float64)
                rc = accelerator(source_vertices, source_vertices.shape[1], source_counts,
                                 candidates, len(candidates), np.ascontiguousarray(q), len(q),
                                 np.ascontiguousarray(target.centers[j]), source.radius_m, areas)
                if rc:
                    raise ValueError(f'spherical overlap accelerator refused geometry: {rc}')
                keep = areas > np.minimum(source.area[candidates], target.area[j])*1e-14
                rows.extend([j]*int(keep.sum())); cols.extend(candidates[keep].tolist()); values.extend(areas[keep].tolist())
                tested += len(candidates)
                continue
            for i in candidates:
                chord = np.linalg.norm(source.centers[i] - target.centers[j])
                if chord > 2 * math.sin(min(math.pi, sr[i] + tr[j] + 1e-13) / 2):
                    continue
                tested += 1
                p = overlap_polygon(source.vertices[i, :source.counts[i]], q, target.centers[j])
                area = spherical_area(p) * source.radius_m ** 2
                if area > min(source.area[i], target.area[j]) * 1e-14:
                    rows.append(j); cols.append(i); values.append(area)
    return rows, cols, values, tested


def _overlap_groups(source, sr):
    bins = np.floor(np.log2(np.maximum(sr, 1e-15))).astype(int)
    groups = []
    for b in np.unique(bins):
        ids = np.flatnonzero(bins == b)
        groups.append((ids, cKDTree(source.centers[ids]), float(sr[ids].max())))
    return groups


# Spawned CPU workers reopen read-only mappings instead of receiving a full
# geometry copy per task or inheriting a parent's CUDA/BLAS state through fork.
def _init_overlap_worker(directory, radius):
    global _overlap_worker_state
    def cells(prefix):
        arrays = [np.load(Path(directory) / f'{prefix}-{name}.npy', mmap_mode='r')
                  for name in ('centers', 'vertices', 'counts', 'area')]
        return SphericalCells(*arrays, radius)
    source, target = cells('source'), cells('target')
    sr = np.load(Path(directory) / 'sr.npy', mmap_mode='r')
    tr = np.load(Path(directory) / 'tr.npy', mmap_mode='r')
    _overlap_worker_state = source, target, sr, tr, _overlap_groups(source, sr), _accelerator()


def _overlap_worker(bounds):
    began = time.process_time()
    result = _overlap_rows(*_overlap_worker_state, *bounds)
    return result, time.process_time() - began


@contextmanager
def _overlap_executor(source, target, sr, tr, workers, backend):
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    if backend == 'process':
        import multiprocessing
        import tempfile
        with tempfile.TemporaryDirectory(prefix='hex-overlap-') as directory:
            for prefix, cells in [('source', source), ('target', target)]:
                for name in ('centers', 'vertices', 'counts', 'area'):
                    np.save(Path(directory) / f'{prefix}-{name}.npy', getattr(cells, name))
            np.save(Path(directory) / 'sr.npy', sr)
            np.save(Path(directory) / 'tr.npy', tr)
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=_init_overlap_worker,
                    initargs=(directory, source.radius_m)) as pool:
                yield lambda bounds: pool.map(_overlap_worker, bounds)
        return
    state = source, target, sr, tr, _overlap_groups(source, sr), _accelerator()
    def run(bounds):
        began = time.thread_time()
        return _overlap_rows(*state, *bounds), time.thread_time() - began
    if backend == 'serial':
        yield lambda bounds: map(run, bounds)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield lambda bounds: pool.map(run, bounds)


def component_area_diagnostics(matrix, source_area, target_area):
    """Measure incompatible declared marginal totals in disconnected overlaps."""
    count, labels = connected_components(bmat([[None, matrix], [matrix.T, None]], format='csr'),
                                         directed=False)
    rows, columns = labels[:matrix.shape[0]], labels[matrix.shape[0]:]
    source_total = np.bincount(columns, weights=source_area, minlength=count)
    target_total = np.bincount(rows, weights=target_area, minlength=count)
    if np.any(source_total <= 0) or np.any(target_total <= 0):
        raise ValueError('overlap component lacks a donor or recipient')
    factor = source_total / target_total
    worst = int(np.argmax(abs(factor-1)))
    source_counts = np.bincount(columns, minlength=count)
    target_counts = np.bincount(rows, minlength=count)
    diagnostics = dict(components=int(count),
                       declared_component_area_relative_floor=float(abs(factor-1).max()),
                       worst_component_source_cells=int(source_counts[worst]),
                       worst_component_target_cells=int(target_counts[worst]),
                       worst_component_source_declared_area=float(source_total[worst]),
                       worst_component_target_declared_area=float(target_total[worst]),
                       worst_component_declared_area_difference=float(source_total[worst]-target_total[worst]))
    return diagnostics


@dataclass
class ConservativeMap:
    overlap: csr_matrix   # target rows, source columns, m2
    source_area: np.ndarray
    target_area: np.ndarray
    receipt: dict

    @classmethod
    def build(cls, source, target, *, coverage_tolerance=1e-8, progress=None,
              reconciliation_limit=16384, target_subdomain=False):
        if isinstance(reconciliation_limit, bool) or int(reconciliation_limit) != reconciliation_limit or reconciliation_limit < 1:
            raise ValueError('reconciliation limit must be a positive integer')
        began = time.perf_counter()
        source.validate(); target.validate()
        if source.radius_m != target.radius_m:
            raise ValueError('sphere radii differ')
        total_error = abs(source.area.sum() / target.area.sum() - 1.0)
        if not target_subdomain and total_error > 2e-13:
            raise ValueError(f'mesh total declared areas differ by {total_error:.17g}')
        sr, tr = source.angular_radius(), target.angular_radius()
        # Radius buckets avoid querying every fine cell against the largest
        # coarse cell's bound. Each bucket still has a complete cap bound.
        n_target = len(target.area)
        workers = int(os.environ.get('GPUWM_HEX_OVERLAP_THREADS', '0')) or len(os.sched_getaffinity(0))
        workers = max(1, min(workers, n_target))
        gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)()
        backend = os.environ.get('GPUWM_HEX_OVERLAP_EXECUTOR', 'auto')
        if backend not in ('auto', 'serial', 'thread', 'process'):
            raise ValueError('overlap executor must be auto, serial, thread or process')
        if backend == 'auto':
            # Small operators do not amortize interpreter startup. Real moving
            # meshes use processes on GIL builds, threads on free-threaded builds.
            backend = 'serial' if n_target < 64 or workers == 1 else ('process' if gil_enabled else 'thread')
        if backend == 'serial':
            workers = 1
        bounds = np.linspace(0, n_target, min(n_target, workers * 8) + 1).astype(int)
        rows, cols, values, tested = [], [], [], 0
        overlap_started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        overlap_began = time.perf_counter()
        worker_cpu_seconds = 0.0
        with _overlap_executor(source, target, sr, tr, workers, backend) as execute:
            for block, ((r, c, v, t), cpu_seconds) in enumerate(execute(list(zip(bounds[:-1], bounds[1:])))):
                rows.extend(r); cols.extend(c); values.extend(v); tested += t
                worker_cpu_seconds += cpu_seconds
                if progress is not None:
                    progress(dict(target_cells=int(bounds[block + 1]), overlaps=len(values),
                                  seconds=time.perf_counter()-began, threads=workers,
                                  executor=backend))
        overlap_execution = dict(started_utc=overlap_started,
            ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            wall_seconds=time.perf_counter()-overlap_began, worker_cpu_seconds=worker_cpu_seconds,
            executor=backend, workers=workers, gil_enabled=gil_enabled,
            parallelism=f'CPU {workers} {"processes" if backend == "process" else "threads"}',
            cpu_scope='overlap tasks only; initialization and coordinator excluded')
        matrix = csr_matrix((values, (rows, cols)), shape=(len(target.area), len(source.area)))
        raw_rows = np.asarray(matrix.sum(axis=1)).ravel()
        raw_cols = np.asarray(matrix.sum(axis=0)).ravel()
        if target_subdomain:
            # Every recipient must be covered. A donor can lie partly or
            # wholly outside the nest; its covered area owns the integral.
            row_error = float(np.max(abs(raw_rows / target.area - 1)))
            donor_excess = float(np.max(np.maximum(raw_cols / source.area - 1, 0)))
            # Check polygon coverage against polygon areas, independently of
            # stored native metrics. Fine meshes can have cancellation in the
            # generator's declared area, even though their polygons tile.
            source_geometric = np.array([spherical_area(v[:n])*source.radius_m**2
                                         for v,n in zip(source.vertices,source.counts)])
            target_geometric = np.array([spherical_area(v[:n])*target.radius_m**2
                                         for v,n in zip(target.vertices,target.counts)])
            metric_error = max(float(np.max(abs(source_geometric/source.area-1))),
                               float(np.max(abs(target_geometric/target.area-1))))
            if metric_error > np.finfo(np.float32).eps:
                raise ValueError(f'declared spherical areas differ beyond FP32 metric rounding: {metric_error}')
            geometric_coverage = float(np.max(abs(raw_rows/target_geometric-1)))
            geometric_excess = float(np.max(np.maximum(raw_cols/source_geometric-1,0)))
            if (not np.all(raw_rows > 0) or
                    max(geometric_coverage, geometric_excess) > coverage_tolerance):
                raise ValueError(f'subdomain overlap coverage residual {geometric_coverage:.17g}; '
                                 f'donor excess {geometric_excess:.17g}')
            matrix.data *= np.repeat(target.area / raw_rows, np.diff(matrix.indptr))
            covered = np.asarray(matrix.sum(axis=0)).ravel()
            return cls(matrix, covered, target.area.copy(), dict(
                method='convex great-circle spherical cell overlap', order=1,
                target_subdomain=True, integral_scope='parent area covered by target cells',
                raw_coverage_relative_max=row_error, raw_donor_excess_relative_max=donor_excess,
                geometric_coverage_relative_max=geometric_coverage,
                geometric_donor_excess_relative_max=geometric_excess,
                declared_metric_relative_max=metric_error,
                source_sha256=source.fingerprint(), target_sha256=target.fingerprint(),
                source_cells=len(source.area), target_cells=len(target.area),
                covered_source_cells=int(np.count_nonzero(covered)),
                covered_parent_area_m2=float(covered.sum()),
                source_declared_area_m2=float(source.area.sum()),
                candidate_pairs=tested, overlaps=matrix.nnz, overlap_execution=overlap_execution, seconds=time.perf_counter()-began))
        raw_error = max(float(np.max(abs(raw_rows / target.area - 1))), float(np.max(abs(raw_cols / source.area - 1))))
        geometry_receipt = {}
        coverage_error = raw_error
        if raw_error > coverage_tolerance:
            # Fine-grid declared metrics can differ from their actual spherical
            # polygons by FP32 rounding. Validate the geometry independently,
            # as the subdomain path does; never accept a gap by relaxing its
            # coverage tolerance. Reconciliation below still uses declared areas.
            source_geometric = np.array([spherical_area(v[:n])*source.radius_m**2
                                         for v,n in zip(source.vertices,source.counts)])
            target_geometric = np.array([spherical_area(v[:n])*target.radius_m**2
                                         for v,n in zip(target.vertices,target.counts)])
            metric_error = max(float(np.max(abs(source_geometric/source.area-1))),
                               float(np.max(abs(target_geometric/target.area-1))))
            if not np.isfinite(metric_error) or metric_error > np.finfo(np.float32).eps:
                raise ValueError(f'declared spherical areas differ beyond FP32 metric rounding: {metric_error}')
            coverage_error = max(float(np.max(abs(raw_rows/target_geometric-1))),
                                 float(np.max(abs(raw_cols/source_geometric-1))))
            geometry_receipt = dict(geometric_coverage_relative_max=coverage_error,
                                    declared_metric_relative_max=metric_error)
        if not np.all(raw_rows > 0) or not np.all(raw_cols > 0) or coverage_error > coverage_tolerance:
            raise ValueError(f'spherical overlap coverage residual {coverage_error:.17g} exceeds {coverage_tolerance}')
        if progress is not None:
            diagnostics = component_area_diagnostics(matrix, source.area, target.area)
            progress(dict(phase='overlap-area-components', **diagnostics))
        # Geometry/declared metric differences must first pass the independent
        # coverage gate. Sparse marginal reconciliation corrects rounding only.
        balanced = matrix.copy()
        residual = raw_error
        column_error = float(np.max(abs(raw_cols / source.area - 1)))
        iterations = 0
        for iteration in range(int(reconciliation_limit)):
            if residual <= RECIPIENT_AREA_RTOL and column_error <= DONOR_AREA_RTOL:
                break
            # Scale existing CSR entries directly, retaining the exact overlap
            # support. Each entry has one row factor and one donor factor;
            # sparse matrix products with diagonal matrices are unnecessary.
            row_factor = target.area / np.asarray(balanced.sum(axis=1)).ravel()
            balanced.data *= np.repeat(row_factor, np.diff(balanced.indptr))
            column_factor = source.area / np.asarray(balanced.sum(axis=0)).ravel()
            balanced.data *= column_factor[balanced.indices]
            column_error = float(np.max(abs(np.asarray(balanced.sum(axis=0)).ravel()/source.area-1)))
            residual = max(float(np.max(abs(np.asarray(balanced.sum(axis=1)).ravel()/target.area-1))), column_error)
            iterations = iteration + 1
        # Column sums govern conservation. Row sums govern constant-field
        # reproduction. Ill-conditioned tiny overlaps can converge slowly;
        # accept only a measured row error far below FP32 state rounding,
        # while retaining FP64 conservation of each donor's entire area.
        column_error = float(np.max(abs(np.asarray(balanced.sum(axis=0)).ravel()/source.area-1)))
        if residual > RECIPIENT_AREA_RTOL or column_error > DONOR_AREA_RTOL:
            raise ValueError(f'overlap area marginals failed to converge: {residual}')
        return cls(balanced.tocsr(), source.area.copy(), target.area.copy(),
                   dict(method='convex great-circle spherical cell overlap', order=1,
                        raw_coverage_relative_max=raw_error, reconciled_relative_max=residual,
                        donor_area_relative_max=column_error,
                        donor_area_relative_tolerance=DONOR_AREA_RTOL,
                        recipient_area_relative_tolerance=RECIPIENT_AREA_RTOL,
                        fp32_constant_reproduction_abs_max=float(abs(
                            (np.asarray(balanced.sum(axis=1)).ravel()/target.area).astype(np.float32)-np.float32(1)).max()),
                        source_sha256=source.fingerprint(), target_sha256=target.fingerprint(),
                        reconciliation_iterations=iterations,reconciliation_iteration_limit=int(reconciliation_limit),
                        source_cells=len(source.area), target_cells=len(target.area),
                        candidate_pairs=tested, overlaps=matrix.nnz, overlap_execution=overlap_execution, seconds=time.perf_counter()-began,
                        **geometry_receipt))

    def with_physical_areas(self, source_area, target_area, *, source_geometry_area=None):
        """Reconcile only prepared metric rounding, conserving active areas."""
        source_area=np.asarray(source_area,dtype=np.float64)
        target_area=np.asarray(target_area,dtype=np.float64)
        if self.receipt.get('target_subdomain', False):
            geometry = np.asarray(source_geometry_area, dtype=np.float64)
            if (geometry.shape != source_area.shape or np.any(geometry <= 0)
                    or source_area.shape != self.source_area.shape
                    or target_area.shape != self.target_area.shape):
                raise ValueError('subdomain prepared metrics require complete source geometry areas')
            metric_error = max(float(np.max(abs(source_area / geometry - 1))),
                               float(np.max(abs(target_area / self.target_area - 1))))
            if not np.isfinite(metric_error) or metric_error > np.finfo(np.float32).eps:
                raise ValueError('prepared physical areas differ beyond FP32 metric rounding')
            matrix = self.overlap.copy()
            matrix.data *= np.repeat(target_area / np.asarray(matrix.sum(axis=1)).ravel(),
                                     np.diff(matrix.indptr))
            covered = np.asarray(matrix.sum(axis=0)).ravel()
            return ConservativeMap(matrix, covered, target_area, dict(self.receipt,
                prepared_metric_relative_difference=metric_error,
                covered_parent_area_m2=float(covered.sum()),
                integral_area_authority='prepared target areaCell and covered parent overlap'))
        metric_error=max(float(np.max(abs(source_area/self.source_area-1))),
                         float(np.max(abs(target_area/self.target_area-1))))
        if metric_error>np.finfo(np.float32).eps:
            raise ValueError('prepared physical areas differ beyond FP32 metric rounding')
        total_difference=float(source_area.sum()-target_area.sum())
        arithmetic=16*np.finfo(float).eps*float(self.source_area.sum())
        cast_envelope=float(abs(source_area-self.source_area).sum()+
                            abs(target_area-self.target_area).sum())
        if abs(total_difference)>arithmetic:
            # Different cell counts have different FP32 area sums. Admit only
            # the exact cast of each independently validated geometry metric.
            # Do not rescale an atmospheric field or substitute equal totals.
            exact_cast=(np.array_equal(source_area,self.source_area.astype(np.float32).astype(float)) and
                        np.array_equal(target_area,self.target_area.astype(np.float32).astype(float)))
            if not exact_cast or abs(total_difference)>cast_envelope+arithmetic:
                raise ValueError('prepared source and target global areas disagree beyond exact metric casts')
        matrix=self.overlap.copy()
        # Each sparse diagonal product only scales the existing CSR entries.
        # Avoid rebuilding two sparse matrices per iteration. Canonical CSR
        # has one entry per donor, so the old products return this same order
        # and perform exactly these two multiplications. Keep the general
        # sparse-product behavior for duplicate or unordered input entries.
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import nullcontext
        from .host_parallel import threads, column_blocks
        direct=(matrix.format=="csr" and matrix.dtype==np.dtype(np.float64) and matrix.has_canonical_format)
        owners=np.repeat(np.arange(matrix.shape[0]),np.diff(matrix.indptr)) if direct else None
        workers=min(threads(),max(1,matrix.nnz//32768)) if direct else 1
        blocks=column_blocks(matrix.nnz,workers)
        # Keep each row/column's additions in SciPy's original CSR order.
        # Splitting a reduction over rows would reassociate donor sums. Split
        # by output columns instead, and retain every donor in original order.
        row_parts=[];column_parts=[]
        if workers>1:
            bounds=np.linspace(0,matrix.shape[0],workers+1,dtype=np.int64)
            for start,stop in zip(bounds[:-1],bounds[1:]):
                part=matrix[start:stop]
                part.data=matrix.data[matrix.indptr[start]:matrix.indptr[stop]]
                row_parts.append((slice(start,stop),part))
            positions=csr_matrix((np.arange(matrix.nnz,dtype=np.int64),matrix.indices,matrix.indptr),
                                 shape=matrix.shape)
            bounds=np.linspace(0,matrix.shape[1],workers+1,dtype=np.int64)
            for start,stop in zip(bounds[:-1],bounds[1:]):
                part=positions[:,start:stop]
                column_parts.append((slice(start,stop),part.data.copy(),
                                     csr_matrix((np.empty(part.nnz),part.indices,part.indptr),shape=part.shape)))
        with (ThreadPoolExecutor(max_workers=workers) if workers>1 else nullcontext(None)) as pool:
            def totals(axis):
                if pool is None:return np.asarray(matrix.sum(axis=axis)).ravel()
                result=np.empty(matrix.shape[1-axis],dtype=np.float64)
                if axis==1:
                    def row_sum(item):
                        block,part=item
                        result[block]=np.asarray(part.sum(axis=1)).ravel()
                    list(pool.map(row_sum,row_parts))
                else:
                    def column_sum(item):
                        block,indexes,part=item
                        part.data=matrix.data[indexes]
                        result[block]=np.asarray(part.sum(axis=0)).ravel()
                    list(pool.map(column_sum,column_parts))
                return result
            def scale(factors,indexes):
                def block_scale(block):
                    matrix.data[block]*=factors[indexes[block]]
                if pool is None:
                    block_scale(slice(0,matrix.nnz))
                else:
                    list(pool.map(block_scale,blocks))
            for iteration in range(256):
                if direct:
                    scale(target_area/totals(1),owners)
                    scale(source_area/totals(0),matrix.indices)
                else:
                    matrix=diags(target_area/np.asarray(matrix.sum(axis=1)).ravel())@matrix
                    matrix=matrix@diags(source_area/np.asarray(matrix.sum(axis=0)).ravel())
                row_error=float(np.max(abs(totals(1)/target_area-1)))
                if row_error<=1e-12:break
        column_error=float(np.max(abs(np.asarray(matrix.sum(axis=0)).ravel()/source_area-1)))
        if row_error>2*np.finfo(np.float32).eps or column_error>2e-14:
            raise ValueError('prepared-area overlap reconciliation failed')
        return ConservativeMap(matrix.tocsr(),source_area,target_area,dict(self.receipt,
            prepared_metric_relative_difference=metric_error,prepared_row_relative_error=row_error,
            prepared_donor_relative_error=column_error,prepared_reconciliation_iterations=iteration+1,
            prepared_global_area_difference_m2=total_difference,
            prepared_global_area_cast_envelope_m2=cast_envelope,
            integral_area_authority='prepared dynamics areaCell, cast to FP64 for summation'))

    def intensive(self, values):
        a = np.asarray(values)
        if a.shape[-1] != len(self.source_area) or not np.all(np.isfinite(a)):
            raise ValueError('source cell extent or finite-value gate failed')
        flat = np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1])
        out = (self.overlap @ flat.T).T / self.target_area
        return out.reshape((*a.shape[:-1], len(self.target_area)))

    def conserved(self, values, *, dtype=np.float32):
        exact = self.intensive(values)
        result = exact.astype(dtype)
        before = np.sum(np.asarray(values, dtype=np.float64) * self.source_area, axis=-1)
        after_exact = np.sum(exact * self.target_area, axis=-1)
        after = np.sum(result.astype(np.float64) * self.target_area, axis=-1)
        # Actual cast error envelope, not a loose user-selected tolerance.
        envelope = np.sum(abs(result.astype(np.float64) - exact) * self.target_area, axis=-1)
        arithmetic = 128 * np.finfo(np.float64).eps * np.sum(abs(exact) * self.target_area, axis=-1)
        if np.any(abs(after-before) > envelope + arithmetic):
            raise ValueError('conserved integral exceeds measured state-cast and FP64 arithmetic envelope')
        return result, dict(before=np.asarray(before).tolist(), after=np.asarray(after).tolist(),
                            fp64_error=np.asarray(after_exact-before).tolist(),
                            rounding_envelope=np.asarray(envelope).tolist(), absolute_error=np.asarray(after-before).tolist())

    def save(self, path):
        a = self.overlap
        np.savez_compressed(path, data=a.data, indices=a.indices, indptr=a.indptr, shape=a.shape,
                            source_area=self.source_area, target_area=self.target_area,
                            receipt=json.dumps(self.receipt, sort_keys=True))

    @classmethod
    def load(cls, path, *, source=None, target=None):
        with np.load(path, allow_pickle=False) as d:
            a = csr_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
            receipt = json.loads(str(d['receipt']))
            for role, mesh in (('source', source), ('target', target)):
                if mesh is not None and receipt[role+'_sha256'] != mesh.fingerprint():
                    raise ValueError(f'{role} geometry differs from saved remap operator')
            return cls(a, d['source_area'], d['target_area'], receipt)
