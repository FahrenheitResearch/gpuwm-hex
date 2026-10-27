"""MPAS-suite LTS frames: deposit and owner assembly run on the writer workers, never on a class step thread.

THE BREAKAGE THIS PREVENTS (HEX-HOST-BOUND-2026-10-27 'What remains'): every rate class waited for its frame
download and copied its deposit on its step thread, and class 0 assembled the global frame there, so at every
frame the classes parked late and the coordinator (which starts a macro only when every class is parked) held
the card about 2 s a frame on the 96/3 km storm mesh.  Now ``PhysicsStreams.write`` takes the writer-thread
capture for the class streams too: the step thread only queues the download.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys.lts_v842 import LtsAborted, LtsClassExecution, _Board, _FrameReady
from hexcore.mpasphys.streams_output_v842 import PhysicsStreams
from hexcore.output import HistoryField


def _classes():
    # Two classes of a 5-cell mesh: class 0 owns global cells 3, 0 (then a halo cell), class 1 owns 1, 4, 2.
    parts = [SimpleNamespace(cell_l2g=np.array([3, 0, 1]), n_owned_cells=2, edge_l2g=np.arange(1), n_owned_edges=1,
                             vertex_l2g=np.arange(1), n_owned_vertices=1),
             SimpleNamespace(cell_l2g=np.array([1, 4, 2, 0]), n_owned_cells=3, edge_l2g=np.arange(1),
                             n_owned_edges=1, vertex_l2g=np.arange(1), n_owned_vertices=1)]
    board = _Board(2)
    layout = SimpleNamespace(dimensions=dict(nCells=5, nEdges=1, nVertices=1, nVertLevels=2))
    rows = {'theta': (dict(name='theta', pool='diag', pool_name='theta', kind='real'), None)}
    writers = []
    for index in range(2):
        execution = SimpleNamespace(board=board, index=index, partitions=parts, partition=parts[index],
                                    global_layout=layout)
        cls = LtsClassExecution._streams_type(execution)
        writer = cls.__new__(cls)
        writer.rows = rows
        writers.append(writer)
    return board, parts, writers


def _frame(part, key):
    owned = part.n_owned_cells
    words = np.stack([part.cell_l2g[:owned] * 10.0, part.cell_l2g[:owned] * 10.0 + 1]).astype(np.float32)
    return {'theta': HistoryField(words, ('nVertLevels', 'nCells'), {})}


def test_the_class_streams_take_the_writer_thread_capture():
    board, _, writers = _classes()
    cls = type(writers[0])
    assert cls.WRITER_THREAD_CAPTURE and cls._snapshot is not PhysicsStreams._snapshot
    assert PhysicsStreams.WRITER_THREAD_CAPTURE is False


def test_class_zero_assembles_on_its_worker_and_the_others_hold_their_words_until_it_has():
    board, parts, writers = _classes()
    key = ('2024-10-07 18:30:00', ('output',))
    results, done = {}, {}

    def worker(index):
        results[index] = writers[index]._assemble_captured(_FrameReady(lambda: None, key, None),
                                                           _frame(parts[index], key))
        done[index] = time.monotonic()

    late = threading.Thread(target=worker, args=(1,))
    late.start()
    late.join(timeout=0.3)
    # Class 1 deposited its pinned words and holds them (its frame stays in flight) until class 0 has read them.
    assert late.is_alive() and 1 in board.deposits[key]
    first = threading.Thread(target=worker, args=(0,))
    first.start()
    first.join(timeout=30)
    late.join(timeout=30)
    assert results[1] == {}
    theta = results[0]['theta'].values
    np.testing.assert_array_equal(theta, np.stack([np.arange(5) * 10.0, np.arange(5) * 10.0 + 1]))
    assert key in board.assembled and key not in board.deposits


def test_a_failed_assembly_stops_every_class_instead_of_leaving_them_waiting():
    board, parts, writers = _classes()
    key = ('2024-10-07 19:00:00', ('output',))
    broken = _frame(parts[1], key)
    broken['theta'] = HistoryField(broken['theta'].values[:, :1], ('nVertLevels', 'nCells'), {})   # too few words
    errors = {}

    def worker(index, frame):
        try:
            writers[index]._assemble_captured(_FrameReady(lambda: None, key, None), frame)
        except BaseException as error:   # noqa: BLE001 - the test records which error each class saw
            errors[index] = error

    threads = [threading.Thread(target=worker, args=(1, broken)),
               threading.Thread(target=worker, args=(0, _frame(parts[0], key)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    assert isinstance(errors[0], ValueError) and 'fewer words' in str(errors[0])
    assert isinstance(errors[1], LtsAborted)
    with pytest.raises(LtsAborted):
        board.wait(lambda: False, 'anything')
