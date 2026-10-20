"""Real CUDA packing, precision authority and finite-check regressions.

Can also run with ``python -m unittest discover -s tree/tests
-p test_device_field_upload.py`` on a CUDA host without pytest.
"""
import os
from contextlib import ExitStack
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from hexcore.cuda_backend import containers as c


def atmosphere_fixture(cells=17, levels=3):
    """Shape-valid synthetic topology and finite fields, no physical forecast."""
    edges, vertices, width, edge_width = 3 * cells, 2 * cells, 10, 20
    mesh = {}
    for name, shape in {
        'cellsOnEdge': (edges, 2), 'edgesOnCell': (cells, width),
        'nEdgesOnCell': (cells,), 'cellsOnCell': (cells, width),
        'verticesOnEdge': (edges, 2), 'edgesOnEdge': (edges, edge_width),
        'nEdgesOnEdge': (edges,), 'verticesOnCell': (cells, width),
        'edgesOnVertex': (vertices, 3), 'cellsOnVertex': (vertices, 3),
    }.items():
        mesh[name] = np.zeros(shape, dtype=np.int32)
    mesh['nEdgesOnCell'].fill(6)
    mesh['nEdgesOnEdge'].fill(10)
    mesh['edgesOnCell'][:, 6:] = -1
    mesh['edgesOnEdge'][:, 10:] = -1
    for name, shape in {
        'weightsOnEdge': (edges, edge_width), 'dcEdge': (edges,),
        'dvEdge': (edges,), 'areaCell': (cells,), 'areaTriangle': (vertices,),
        'kiteAreasOnVertex': (vertices, 3), 'latCell': (cells,),
        'lonCell': (cells,), 'latEdge': (edges,), 'lonEdge': (edges,),
        'angleEdge': (edges,), 'meshDensity': (cells,),
        'defc_a': (cells, width), 'defc_b': (cells, width),
        'fVertex': (vertices,), 'fEdge': (edges,),
    }.items():
        mesh[name] = np.ones(shape, dtype=np.float32)
    mesh['weightsOnEdge'][:, 10:] = -0.0
    host_mesh = SimpleNamespace(arrays=mesh, dimensions=dict(
        nCells=cells, nEdges=edges, nVertices=vertices, maxEdges=width,
        maxEdges2=edge_width, vertexDegree=3), nominalMinDc=1.)

    def field(shape, seed, dtype=np.float32, order='C'):
        a = np.full(shape, seed, dtype=dtype, order=order)
        a.flat[0] = -0.0
        if a.size > 1:
            a.flat[1] = np.float32(1e-45)
        return a

    vertical = {name: field((levels + 1 if name in ('zw', 'ah') else levels,), i + 1,
                            np.float64) for i, name in enumerate(
        ('zw', 'dzw', 'rdzw', 'zu', 'dzu', 'rdzu', 'rdzwp', 'rdzwm', 'fzp', 'fzm', 'ah'))}
    for i, (name, shape) in enumerate({
        'hx': (levels + 1, cells), 'zgrid': (levels + 1, cells),
        'zz': (levels, cells), 'zxu': (levels, edges), 'dss': (levels, cells),
    }.items()):
        vertical[name] = field(shape, i + 11, np.float64, 'F')
    vertical.update(cf1=1., cf2=-0., cf3=3., first_height_level=0)
    reference = {name: field((levels, cells), i + 21, np.float64, 'F')
                 for i, name in enumerate(('rho_base', 'rho_theta_base', 'pressure_base', 'exner_base'))}
    state = {name: field(shape, i + 31) for i, (name, shape) in enumerate({
        'rho': (levels, cells), 'rho_theta': (levels, cells),
        'rho_u': (levels, edges), 'rho_w': (levels + 1, cells),
        'scalars': (6, levels, cells),
    }.items())}
    state['time_seconds'] = 120.
    saved = {name: field((levels + 1 if name == 'vertical_velocity' else levels,
                         edges if name == 'normal_velocity' else cells), i + 41)
             for i, name in enumerate(('theta_m', 'exner', 'density_perturbation',
                 'rho_theta_perturbation', 'pressure_perturbation', 'normal_velocity', 'vertical_velocity'))}
    terrain = {name: field((levels + 1, cells, width), i + 51)
               for i, name in enumerate(('zb_cell', 'zb3_cell'))}
    for a in terrain.values():
        a[:, :, 6:] = -0.0
    return (host_mesh, *(SimpleNamespace(**fields) for fields in
                        (state, vertical, reference, saved, terrain)))


@unittest.skipIf(cp is None, "CuPy is required for the real device gate")
class DeviceFieldUpload(unittest.TestCase):
    def test_atmosphere_uploads_overlap_and_return_ready_exact_arrays(self):
        sources = atmosphere_fixture()
        barrier = threading.Barrier(3)
        observations = []
        with ExitStack() as stack:
            for cls in (c.DeviceMesh, c.DevicePrognosticState, c.DeviceVerticalGrid,
                        c.DeviceReferenceState, c.DeviceSavedDiagnostics, c.DeviceTerrainMetrics):
                original, name = cls.from_host, cls.__name__

                def build(*args, original=original, name=name, **kwargs):
                    observations.append((name, threading.get_ident(),
                                         cp.cuda.runtime.getDevice(),
                                         cp.cuda.get_current_stream().ptr))
                    if name in ('DeviceMesh', 'DevicePrognosticState', 'DeviceVerticalGrid'):
                        barrier.wait(timeout=10)
                    return original(*args, **kwargs)

                stack.enter_context(patch.object(cls, 'from_host', side_effect=build))
            with cp.cuda.Stream(non_blocking=True) as caller:
                device_id = cp.cuda.runtime.getDevice()
                actual = c.DeviceAtmosphere.from_host(*sources, dtype=np.float32)
                self.assertEqual(cp.cuda.get_current_stream().ptr, caller.ptr)
                self.assertTrue(all(row[2] == device_id and row[3] != caller.ptr
                                    for row in observations))
                self.assertEqual(len({row[1] for row in observations[:3]}), 3)
                for name, source in zip(('state', 'vertical', 'reference', 'saved'), sources[1:5]):
                    value = getattr(actual, name)
                    for field, host in vars(source).items():
                        if isinstance(host, np.ndarray):
                            expected = np.ascontiguousarray(host, dtype=np.float32)
                            self.assertEqual(cp.asnumpy(getattr(value, field)).tobytes(),
                                             expected.tobytes())
                for field, array in zip(('zb_cell', 'zb3_cell'), actual.terrain.full_layout()):
                    self.assertEqual(cp.asnumpy(array).tobytes(),
                                     getattr(sources[-1], field).tobytes())
                mesh = actual.mesh.to_host()
                for target, source in (('cells_on_edge', 'cellsOnEdge'),
                                       ('edges_on_edge', 'edgesOnEdge'),
                                       ('weights_on_edge', 'weightsOnEdge')):
                    self.assertEqual(mesh[target].tobytes(), sources[0].arrays[source].tobytes())
                actual.validate()

    def test_atmosphere_optional_components_and_error_priority(self):
        sources = atmosphere_fixture()
        actual = c.DeviceAtmosphere.from_host(*sources[:4], None, dtype=np.float32)
        self.assertIsNone(actual.terrain)
        self.assertEqual(actual.saved.h2d.bytes, 0)
        actual.validate()
        with patch.object(c.DeviceMesh, 'from_host', side_effect=ValueError('first mesh')), \
                patch.object(c.DevicePrognosticState, 'from_host', side_effect=ValueError('second state')):
            with self.assertRaisesRegex(ValueError, 'first mesh'):
                c.DeviceAtmosphere.from_host(*sources, dtype=np.float32)
        # Refuse after the first state field has already been queued, while a
        # later component also fails. Worker cleanup must finish queued copies
        # and preserve the earlier state's named exception.
        sources[1].rho_theta = sources[1].rho_theta.astype(np.float64)
        sources[3].pressure_base.fill(np.nan)
        with self.assertRaisesRegex(TypeError, 'rho_theta must have dtype'):
            c.DeviceAtmosphere.from_host(*sources, dtype=np.float32)

    def test_terrain_restore_and_dropped_tail_validation(self):
        from hexcore import host_parallel

        words = np.array([0, 0x80000000, 1, 0x007fffff, 0x3f800001], dtype=np.uint32)
        base = np.resize(words.view(np.float32), (7, 31, 10)).astype(np.float64)
        base[:, :, 6:] = -0.0
        with patch.object(host_parallel, "MIN_PARALLEL_ELEMENTS", 1), \
                patch.object(host_parallel, "MIN_BLOCK_COLUMNS", 2), \
                patch.dict(os.environ, {host_parallel.ENVIRONMENT: "4"}):
            for source in (base, np.asfortranarray(base), base[::-1, ::-1, :]):
                terrain = SimpleNamespace(zb_cell=source, zb3_cell=source.copy())
                value = c.DeviceTerrainMetrics.from_host(terrain, dtype=np.float32,
                                                        slot_stride=6)
                value.validate(n_vert_levels=6, n_cells=31, max_edges=10)
                self.assertEqual(value.slot_stride, 6)
                self.assertTrue(np.signbit(value.slot_fill))
                expected = np.ascontiguousarray(source, dtype=np.float32)
                for actual in value.full_layout():
                    self.assertEqual(cp.asnumpy(actual).tobytes(), expected.tobytes())
            for bad in (np.nan, np.inf, 1e300):
                source = base.copy()
                source[-1, -1, -1] = bad
                with self.assertRaisesRegex(ValueError, "zb_cell contains non-finite"):
                    c.DeviceTerrainMetrics.from_host(
                        SimpleNamespace(zb_cell=source, zb3_cell=base),
                        dtype=np.float32, slot_stride=6)
            second = base.copy()
            second[:, :, 6:] = 0.0
            value = c.DeviceTerrainMetrics.from_host(
                SimpleNamespace(zb_cell=base, zb3_cell=second),
                dtype=np.float32, slot_stride=6)
            self.assertEqual(value.slot_stride, 10)

    def test_device_words_match_numpy_packing_and_conversion(self):
        words = np.array([0, 0x80000000, 0x00000001, 0x007fffff,
                          0x3f800001, 0x7fc01234, 0xff800000], dtype=np.uint32)
        for dtype in (np.float32, np.float64):
            base = np.resize(words.view(np.float32), (13, 17)).astype(dtype)
            for source in (base, np.asfortranarray(base), base[::-1, ::2]):
                for target in (np.float32, np.float64):
                    with self.subTest(dtype=dtype, strides=source.strides, target=target):
                        actual = c._device_float_fields(
                            cp, SimpleNamespace(field=source), ("field",),
                            np.dtype(target), finite=False,
                        )["field"]
                        expected = np.ascontiguousarray(source, dtype=target)
                        self.assertTrue(actual.flags.c_contiguous)
                        self.assertEqual(cp.asnumpy(actual).tobytes(), expected.tobytes())

    def test_dtype_authority_and_finite_failures_are_retained(self):
        for cls, field in ((c.DevicePrognosticState, "rho"),
                           (c.DeviceSavedDiagnostics, "theta_m")):
            with self.assertRaisesRegex(TypeError, field + " must have dtype"):
                cls.from_host(SimpleNamespace(**{field: np.ones(2, dtype=np.float64)}),
                              dtype=np.float32)
        for value in (np.nan, np.inf, -np.inf, 1e300):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "pressure_base contains non-finite"):
                reference = SimpleNamespace(**{
                    name: np.array([value if name == "pressure_base" else 1], dtype=np.float64)
                    for name in ("rho_base", "rho_theta_base", "pressure_base", "exner_base")
                })
                c.DeviceReferenceState.from_host(reference, dtype=np.float32)

    def test_vertical_lazy_and_resident_hx_keep_exact_words(self):
        nlev, cells, edges = 3, 7, 13
        values = {}
        for name in ("zw", "ah"):
            values[name] = np.arange(nlev + 1, dtype=np.float64)
        for name in ("dzw", "rdzw", "zu", "dzu", "rdzu", "rdzwp", "rdzwm", "fzp", "fzm"):
            values[name] = np.arange(nlev, dtype=np.float64)
        for name, shape in {"hx": (nlev + 1, cells), "zgrid": (nlev + 1, cells),
                            "zz": (nlev, cells), "zxu": (nlev, edges), "dss": (nlev, cells)}.items():
            values[name] = np.arange(np.prod(shape), dtype=np.float64).reshape(shape, order="F")
        source = SimpleNamespace(**values, cf1=1., cf2=-0., cf3=3., first_height_level=0)
        for lazy in (False, True):
            with self.subTest(lazy=lazy), patch.dict(
                    os.environ, {"GPUWM_HEX_MESH_COMPACT": "1",
                                 "GPUWM_HEX_MESH_COMPACT_VERTICAL_HX": str(int(lazy))}):
                actual = c.DeviceVerticalGrid.from_host(source, dtype=np.float32)
                actual.validate(n_cells=cells, n_edges=edges)
                self.assertEqual(actual.hx_resident, not lazy)
                for name, value in values.items():
                    expected = np.ascontiguousarray(value, dtype=np.float32)
                    self.assertEqual(cp.asnumpy(getattr(actual, name)).tobytes(), expected.tobytes())
                self.assertTrue(actual.hx_resident)

    def test_finite_word_scan_covers_empty_boundaries_and_grid_stride(self):
        for dtype, integer, bad_words, finite_words in (
            (np.float32, np.uint32,
             (0x7f800000, 0xff800000, 0x7fc01234, 0x7f800001),
             (0, 0x80000000, 1, 0x007fffff, 0x7f7fffff)),
            (np.float64, np.uint64,
             (0x7ff0000000000000, 0xfff0000000000000,
              0x7ff8000000001234, 0x7ff0000000000001),
             (0, 0x8000000000000000, 1, 0x000fffffffffffff,
              0x7fefffffffffffff)),
        ):
            for count in (0, 1, 255, 256, 257, 65535 * 256 + 19):
                with self.subTest(dtype=dtype, count=count):
                    source = np.resize(np.array(finite_words, dtype=integer), count).view(dtype)
                    c._device_float_fields(cp, SimpleNamespace(field=source),
                                           ("field",), np.dtype(dtype))
                    if count:
                        source.view(integer)[-1] = bad_words[-1]
                        with self.assertRaisesRegex(ValueError, "field contains non-finite"):
                            c._device_float_fields(cp, SimpleNamespace(field=source),
                                                   ("field",), np.dtype(dtype))
            for bad in bad_words:
                with self.subTest(dtype=dtype, bad=bad):
                    source = np.array([0, bad, 0], dtype=integer).view(dtype)
                    with self.assertRaisesRegex(ValueError, "second contains non-finite"):
                        c._device_float_fields(
                            cp, SimpleNamespace(first=np.zeros(1, dtype=dtype), second=source),
                            ("first", "second"), np.dtype(dtype),
                        )


if __name__ == "__main__":
    unittest.main()
