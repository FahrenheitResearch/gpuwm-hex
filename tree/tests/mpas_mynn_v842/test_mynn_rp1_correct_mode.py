"""RP1 correct mode of the hex-cuda MYNN moisture conversion (bl_mynn_pre.F / bl_mynn_post.F).

Exact mode must stay the pinned native conversion word for word; correct mode must equal the native
correct-mode build (MPAS v8.4.2 plus the upstream-format correct-mode patch, gfortran strict single
precision) word for word, and keep condensate unchanged under zero MYNN tendencies.
The two native word files come from that build's harness (the correct-mode evidence of 2026-10-13,
mpas-rad-pbl-lsm-seaice/evidence/harnesses: RP1-pristine/words.bin and RP1-patched-true/words.bin).
"""
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

from hexcore.mpasphys.bl_mynn_v842 import MynnDriverAdapter

FIXTURES = Path(__file__).with_name("fixtures")
ITE, KTE = 8, 3


def f32(x):
    return np.float32(x)


def matrix():
    """The harness inputs, computed in float32 in the harness's operation order."""
    qv = np.empty((ITE, KTE), np.float32); qc = np.empty_like(qv); qi = np.empty_like(qv); qs = np.empty_like(qv)
    for k in range(1, KTE + 1):
        for i in range(1, ITE + 1):
            qv[i - 1, k - 1] = f32(0.005) * f32(i) + f32(0.001) * f32(k)
            qc[i - 1, k - 1] = f32(1.e-3) * f32(k)
            qi[i - 1, k - 1] = f32(3.e-4) * f32(i)
            qs[i - 1, k - 1] = f32(7.e-4)
    return qv, qc, qi, qs


def device(a):
    return cp.asfortranarray(cp.asarray(np.asfortranarray(a)))


def convert(adapter, exact_inverse, qv, qc, qi, qs, d, dt):
    mixing = {"qv": device(qv), "qc": device(qc), "qi": device(qi), "qs": device(qs)}
    specific = {n: cp.empty((ITE, KTE), cp.float32, order="F") for n in ("sqv", "sqc", "sqi", "sqs")}
    adapter.pre(mixing, specific, (True, True, True), exact_inverse=exact_inverse)
    tend = {n: device(v) for n, v in zip(("dqv", "dqc", "dqi", "dqs"), d)}
    adapter.post(mixing, tend, (True, True, True), dt, exact_inverse=exact_inverse)
    return specific, tend


@pytest.mark.parametrize("exact_inverse,fixture", [(False, "rp1_native_exact_words.bin"),
                                                   (True, "rp1_native_correct_words.bin")])
def test_conversion_words_equal_the_native_build(exact_inverse, fixture):
    adapter = MynnDriverAdapter()
    qv, qc, qi, qs = matrix()
    d = [np.full((ITE, KTE), v, np.float32) for v in (1.e-6, -2.5e-7, 1.e-7, 1.5e-7)]
    specific, tend = convert(adapter, exact_inverse, qv, qc, qi, qs, d, f32(20.))
    got = np.concatenate([np.asfortranarray(a.get()).ravel(order="F") for a in
                          (*(specific[n] for n in ("sqv", "sqc", "sqi", "sqs")),
                           *(tend[n] for n in ("dqv", "dqc", "dqi", "dqs")))])
    # The native harness writes big-endian stream words (MPAS's -fconvert=big-endian recipe).
    want = np.frombuffer((FIXTURES / fixture).read_bytes(), dtype=">f4").astype(np.float32)
    assert got.view(np.uint32).tolist() == want.view(np.uint32).tolist()


def drift(exact_inverse):
    adapter = MynnDriverAdapter()
    qv = np.empty((ITE, KTE), np.float32)
    for i in range(ITE):
        qv[i, :] = f32(0.005) * f32(i + 1)
    qc = np.full((ITE, KTE), 1.e-3, np.float32); qi = qc.copy(); qs = qc.copy()
    start = (qc.copy(), qi.copy(), qs.copy())
    dt = f32(20.)
    for _ in range(45):                       # 900 s of 20 s MYNN calls, all MYNN tendencies zero
        zero = [np.zeros((ITE, KTE), np.float32) for _ in range(4)]
        _, tend = convert(adapter, exact_inverse, qv, qc, qi, qs, zero, dt)
        qv = qv + dt * tend["dqv"].get(); qc = qc + dt * tend["dqc"].get()
        qi = qi + dt * tend["dqi"].get(); qs = qs + dt * tend["dqs"].get()
    return max(float(np.max(np.abs(a - b) / b)) for a, b in zip((qc, qi, qs), start)), float((qc[3, 0] - 1.e-3) / 1.e-3)


def test_RP1_zero_tendencies_keep_condensate_only_in_correct_mode():
    worst_exact, cloud_exact = drift(False)
    worst_correct, _ = drift(True)
    assert cloud_exact < -0.017               # matched MPAS: about 1.7 % of cloud water lost at qv = 0.02
    assert worst_exact > 1.e-2
    assert worst_correct < 2.e-5               # correct mode: round-off only
