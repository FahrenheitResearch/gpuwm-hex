"""Blocked host copies write exactly the bytes of the one-call NumPy forms."""

from __future__ import annotations

import numpy as np
import pytest

from hexcore import host_parallel


@pytest.fixture
def forced_threads(monkeypatch):
    """Split even small arrays into many blocks on several threads."""

    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 7)
    monkeypatch.setenv(host_parallel.ENVIRONMENT, "4")


@pytest.mark.parametrize("dtype", ["<f4", "<f8", ">f4", "<i8"])
@pytest.mark.parametrize("axes", [(1, 0), (0, 1)])
@pytest.mark.parametrize("shape", [(1003, 55), (55, 1003)])
def test_transposed_copy_matches_numpy_bytes(forced_threads, dtype, axes, shape):
    rng = np.random.default_rng(841)
    value = rng.standard_normal(shape).astype(dtype)
    expected = np.transpose(value, axes).copy()
    actual = host_parallel.transposed_copy(value, axes)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()


def test_blocks_cover_every_column_once_in_order():
    for columns in (0, 1, 7, 8192, 100_003):
        blocks = host_parallel.column_blocks(columns, 6)
        covered = [index for block in blocks for index in range(columns)[block]]
        assert covered == list(range(columns))


def test_run_blocks_reaches_every_column(forced_threads):
    seen = np.zeros(1000, dtype=np.int64)

    def mark(block):
        seen[block] += 1

    host_parallel.run_blocks(mark, seen.size, elements=seen.size)
    assert np.all(seen == 1)


def test_thread_count_refuses_a_non_positive_setting(monkeypatch):
    monkeypatch.setenv(host_parallel.ENVIRONMENT, "0")
    with pytest.raises(ValueError, match="ambiguous host copy schedule"):
        host_parallel.threads()


def test_coupled_ru_blocks_match_the_whole_array_expression(forced_threads):
    """The driver's edge-blocked ru gather against the one-call expression."""

    rng = np.random.default_rng(7)
    rho = rng.uniform(0.1, 1.3, size=(55, 400)).astype(np.float32)
    u = rng.standard_normal((55, 1201)).astype(np.float32)
    cells = rng.integers(0, 400, size=(1201, 2)).astype(np.int64)
    half = rho.dtype.type(0.5)
    expected = (half * u) * (rho[:, cells[:, 0]] + rho[:, cells[:, 1]])
    actual = np.empty_like(expected)

    def fill(block):
        actual[:, block] = (half * u[:, block]) * (rho[:, cells[block, 0]] + rho[:, cells[block, 1]])

    host_parallel.run_blocks(fill, cells.shape[0], elements=actual.size)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("reverse", [False, True])
def test_contiguous_copy_preserves_all_float_words(forced_threads, width, reverse):
    """NaN payloads, infinities, signed zeros and subnormals are only copied."""
    special = ([0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000,
                0x7fc12345, 0x7f812345] if width == 4 else
               [0, 0x8000000000000000, 1, 0x8000000000000001,
                0x7ff0000000000000, 0xfff0000000000000,
                0x7ff8000000012345, 0x7ff0000000012345])
    words = np.resize(np.asarray(special, dtype=f'u{width}'), (101, 2, 7))
    view = words.view(f'f{width}').transpose(2, 1, 0)
    if reverse:
        view = view[::-1, :, ::-1]
    view.flags.writeable = False
    actual = host_parallel.contiguous_copy(view, dtype=view.dtype)
    expected = np.ascontiguousarray(view)
    assert actual.flags.c_contiguous and actual.flags.writeable
    assert not np.shares_memory(actual, view)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", ['scalar', 'empty', 'contiguous', 'small', 'list', 'object'])
def test_contiguous_copy_fallbacks_keep_numpy_contract(monkeypatch, kind):
    value = np.arange(30, dtype=np.float32).reshape(10, 3).T
    dtype = np.dtype('f4')
    if kind == 'scalar': value = np.array(1, dtype=dtype)
    elif kind == 'empty': value = value[:, :0]
    elif kind == 'contiguous': value = value.copy(); value.flags.writeable = False
    elif kind == 'list': value = [[1, 2], [3, 4]]
    elif kind == 'object': value = value.astype(object); dtype = np.dtype(object)
    monkeypatch.setattr(host_parallel, 'MIN_PARALLEL_ELEMENTS', 100 if kind == 'small' else 0)
    def forbidden(*args, **kwargs):
        pytest.fail('fallback dispatched workers')
    monkeypatch.setattr(host_parallel, 'run_blocks', forbidden)
    actual = host_parallel.contiguous_copy(value, dtype=dtype)
    expected = np.ascontiguousarray(value, dtype=dtype)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.writeable == expected.flags.writeable
    if kind == 'object': np.testing.assert_array_equal(actual, expected)
    else: assert actual.tobytes() == expected.tobytes()
    if kind == 'contiguous': assert actual is value


@pytest.mark.parametrize('dtype', ['f4', 'f8', '>f4'])
@pytest.mark.parametrize('reverse', [False, True])
def test_float_conversion_preserves_subnormals_nan_payloads_and_rounding(forced_threads, dtype, reverse):
    words = np.array([0, 0x80000000, 1, 0x007fffff, 0x3f800001,
                      0x7fc12345, 0x7f812345, 0xff800000], dtype=np.uint32)
    with np.errstate(invalid='ignore'):
        source = np.resize(words.view(np.float32), (101, 2, 7)).astype(np.float64)
    source = source.transpose(2, 1, 0)
    if reverse:
        source = source[::-1, :, ::-1]
    source.flags.writeable = False
    with np.errstate(invalid='ignore'):
        actual = host_parallel.contiguous_copy(source, dtype=np.dtype(dtype))
        expected = np.ascontiguousarray(source, dtype=dtype)
    assert actual.flags.c_contiguous and actual.flags.writeable
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('workers', [1, 2, 32])
def test_contiguous_copy_caps_workers_and_honors_lower_setting(forced_threads, monkeypatch, workers):
    monkeypatch.setenv(host_parallel.ENVIRONMENT, str(workers))
    real = host_parallel.ThreadPoolExecutor
    seen = []
    def executor(*args, **kwargs):
        seen.append(kwargs['max_workers'])
        return real(*args, **kwargs)
    monkeypatch.setattr(host_parallel, 'ThreadPoolExecutor', executor)
    value = np.arange(101*2*7, dtype=np.int64).reshape(101,2,7).transpose(2,1,0)
    result = host_parallel.contiguous_copy(value, dtype=value.dtype)
    assert result.tobytes() == np.ascontiguousarray(value).tobytes()
    assert seen == ([] if workers == 1 else [min(workers, 4)])


def test_entity_major_vertical_view_splits_its_long_axis(forced_threads, monkeypatch):
    source = np.arange(56 * 2 * 101, dtype="f4").reshape(56, 2, 101)
    view = source.transpose(2, 1, 0)
    run = host_parallel.run_blocks
    columns = []

    def observe(function, count, **kwargs):
        columns.append(count)
        return run(function, count, **kwargs)

    monkeypatch.setattr(host_parallel, "run_blocks", observe)
    actual = host_parallel.contiguous_copy(view, dtype=view.dtype)
    assert columns == [101]
    assert actual.tobytes() == view.tobytes(order="C")
@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, 0.0])
@pytest.mark.parametrize("position", [0, 7, 999])
def test_parallel_finite_scan_preserves_refusal(forced_threads, special, position):
    values = np.ones(1000, dtype=np.float32)
    values[position] = special
    before = values.tobytes()
    assert host_parallel.all_finite(values) == bool(np.all(np.isfinite(values)))
    assert values.tobytes() == before


@pytest.mark.parametrize("layout", ["fortran", "transpose", "negative", "gapped", "broadcast"])
@pytest.mark.parametrize("dtype", ["<f4", ">f8", "<c8"])
@pytest.mark.parametrize("special", [0.0, np.nan, np.inf, -np.inf])
def test_strided_finite_scan_keeps_words_and_result(forced_threads, layout, dtype, special):
    values = np.ones((101, 2, 7), dtype=dtype)
    values[99, 1, 6] = special
    if layout == "fortran": values = np.asfortranarray(values)
    elif layout == "transpose": values = values.transpose(2, 1, 0)
    elif layout == "negative": values = values[::-1, :, ::-1]
    elif layout == "gapped": values = values[1::2, :, ::2]
    elif layout == "broadcast": values = np.broadcast_to(values[99:100], (101, 2, 7))
    values.flags.writeable = False
    before = values.tobytes()
    assert host_parallel.all_finite(values) == bool(np.all(np.isfinite(values)))
    assert values.tobytes() == before


def test_terrain_finite_scan_uses_multiple_threads_and_keeps_refusals(forced_threads, monkeypatch):
    import threading
    from hexcore.terrain import _float_array
    real = np.isfinite
    barrier = threading.Barrier(4, timeout=5)
    seen = set()
    lock = threading.Lock()

    def scan(value):
        with lock:
            first = threading.get_ident() not in seen
            seen.add(threading.get_ident())
        if first: barrier.wait()
        return real(value)

    values = np.ones((101, 2, 7), dtype=np.float32).transpose(2, 1, 0)
    monkeypatch.setattr(np, "isfinite", scan)
    assert _float_array("zb", values, ndim=3) is values
    assert len(seen) == 4
    monkeypatch.setattr(np, "isfinite", real)
    values[-1, -1, -1] = np.nan
    with pytest.raises(ValueError, match="^zb contains non-finite values$"):
        _float_array("zb", values, ndim=3)
    with pytest.raises(ValueError, match="^zb must be 2-dimensional"):
        _float_array("zb", values, ndim=2)
    with pytest.raises(TypeError, match="^zb must have dtype float32 or float64"):
        _float_array("zb", np.ones(values.shape, dtype=np.int32), ndim=3)


@pytest.mark.parametrize("bad_active", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_active_stencils_keep_padding_exempt(forced_threads, bad_active, dtype):
    counts = np.arange(1001) % 8
    cells = np.full((1001, 7), 100, dtype=dtype)
    for row, count in enumerate(counts):
        cells[row, :count] = 3
    if bad_active:
        cells[997, 0] = 100
    expected = all(not np.any((cells[i, :n] < 0) | (cells[i, :n] >= 10))
                   for i, n in enumerate(counts))
    assert host_parallel.active_indices_in_range(cells, counts, 10) == expected


def test_compaction_copy_splits_long_entity_axis(forced_threads):
    from hexcore.cuda_backend.compaction import compact_host
    bits = np.resize(np.array([0, 0x80000000, 0x7fc12345, 0x7f812345], np.uint32),
                     (3, 1001, 10))
    values = bits.view(np.float32)
    actual = compact_host(values, 6)
    assert actual.tobytes() == np.ascontiguousarray(values[..., :6]).tobytes()


@pytest.mark.parametrize("defect", ["none", "nonfinite", "count", "active", "dtype"])
def test_cuda_coefficient_admission_retains_refusals(forced_threads, defect):
    from types import SimpleNamespace
    from hexcore.cuda_driver import _validate_v841_host_coefficients
    data = SimpleNamespace(adv_coefs=np.ones((1001, 7), np.float32),
                           adv_coefs_3rd=np.ones((1001, 7), np.float32),
                           n_adv_cells_for_edge=np.full(1001, 4, np.int32),
                           adv_cells_for_edge=np.full((1001, 7), -1, np.int32))
    data.adv_cells_for_edge[:, :4] = 2
    if defect == "nonfinite": data.adv_coefs_3rd[-1, -1] = np.nan
    elif defect == "count": data.n_adv_cells_for_edge[-1] = 8
    elif defect == "active": data.adv_cells_for_edge[-1, 3] = 10
    elif defect == "dtype": data.adv_coefs = data.adv_coefs.astype(np.float64)
    if defect == "none":
        _validate_v841_host_coefficients(data, n_cells=10, n_edges=1001)
    else:
        with pytest.raises(TypeError if defect == "dtype" else ValueError):
            _validate_v841_host_coefficients(data, n_cells=10, n_edges=1001)


@pytest.mark.parametrize("source_dtype, wire_dtype", [
    ("f4", "<f4"), ("f8", "<f8"), (">f4", "<f8"),
    (">f8", "<f8"), ("i4", "<f8"), ("i8", "<f8"),
])
@pytest.mark.parametrize("layout", ["native", "transpose", "negative"])
def test_vertical_v2_float_wire_words_and_aliases(
        forced_threads, source_dtype, wire_dtype, layout):
    """The raw vertical protocol retains dtype selection and C-order bytes."""
    from hexcore.vertical import _v2_array

    words = np.array([0, 0x80000000, 1, 0x007fffff, 0x3f800001,
                      0x7fc12345, 0x7f812345, 0xff800000], dtype=np.uint32)
    if np.dtype(source_dtype).kind == "f":
        with np.errstate(invalid="ignore"):
            values = words.view(np.float32).astype(source_dtype)
    else:
        values = np.array([0, -1, 1, 12345, -12345, 2**30, -2**30, 7],
                          dtype=source_dtype)
    with np.errstate(invalid="ignore"):
        value = np.resize(values, (401, 2, 8)).astype(source_dtype, copy=False)
    if layout == "transpose": value = value.transpose(2, 1, 0)
    elif layout == "negative": value = value[::-1, :, ::-1]
    value.flags.writeable = False
    original = value.tobytes()
    with np.errstate(invalid="ignore"):
        expected = np.ascontiguousarray(value, dtype=wire_dtype)
        actual = _v2_array("zgrid", value)
    assert actual.dtype == np.dtype(wire_dtype)
    assert actual.shape == value.shape and actual.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()
    assert value.tobytes() == original
    if value.flags.c_contiguous and value.dtype == np.dtype(wire_dtype):
        assert actual is value and not actual.flags.writeable


@pytest.mark.parametrize("source_dtype, wire_dtype", [
    ("i4", "<i4"), ("i8", "<i8"), (">i4", "<i8"), ("u4", "<i8"),
])
def test_vertical_v2_integer_wire_keeps_width_and_inactive_sentinels(
        forced_threads, source_dtype, wire_dtype):
    from hexcore.vertical import _v2_array

    values = np.array([0, 1, 17, 2**30], dtype=source_dtype)
    if values.dtype.kind == "i": values[0] = -1
    value = np.resize(values, (401, 2, 8)).astype(source_dtype, copy=False)
    value = value.transpose(2, 1, 0)
    value.flags.writeable = False
    actual = _v2_array("cells_on_edge", value)
    assert actual.dtype == np.dtype(wire_dtype)
    assert actual.tobytes() == np.ascontiguousarray(value, dtype=wire_dtype).tobytes()


def test_vertical_v2_conversion_retains_caller_error_policy(forced_threads):
    from hexcore.vertical import _v2_array

    value = np.full((401, 2, 8), 0x7f812345, dtype=">u4").view(">f4")
    with np.errstate(invalid="raise"):
        with pytest.raises(FloatingPointError) as old:
            np.ascontiguousarray(value, dtype="<f8")
        with pytest.raises(FloatingPointError) as new:
            _v2_array("zgrid", value)
    assert str(new.value) == str(old.value)
