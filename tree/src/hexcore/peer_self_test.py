"""Startup peer-copy self-test: does a card pair's direct peer route move the right bytes?

Standalone on purpose: it imports nothing from hexcore, takes the array module (CuPy) as an
argument, and returns plain records, so the engine's tilestream transport and Global's NCCL
setup can call or vendor the same check verbatim and read the same verdict.

THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-05, a rented 8x RTX PRO 6000
container): deviceCanAccessPeer reported 1 both ways, peer access enabled, and
cudaMemcpyPeerAsync completed with success at 17-42 GB/s, yet every copied word but the
first was wrong, at 4 KiB to 512 MiB, in both directions; a kernel reading peer memory
directly was equally wrong.  Same-host hex halos over that route made every N-rank forecast
non-finite at step 1, and NCCL's P2P transport there never completed its first broadcast.
On a healthy host of the same model the same test moves every word exactly.
Success codes do not prove the bytes, so the bytes are checked.

Verdict per ordered pair: ``passed`` (every word exact), ``wrong_data`` (copy finished,
words differ) or ``timeout`` (copy did not finish inside the bound; the pair may hang).
A caller stages, or disables P2P for, any pair that did not pass, and records why.
"""
from __future__ import annotations

import time
from typing import Any, Iterable

SCHEMA = "peer-copy-self-test/v1"
DEFAULT_WORDS = 1 << 16  # 256 KiB: large enough to cross pages, small enough to cost nothing
DEFAULT_TIMEOUT_S = 10.0


def _pattern(xp: Any, words: int) -> Any:
    return (xp.arange(words, dtype=xp.uint32) * xp.uint32(2654435761)) ^ xp.uint32(0x5A5A5A5A)


def check_pair(xp: Any, src: int, dst: int, *, words: int = DEFAULT_WORDS,
               timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Copy a known pattern src -> dst over the peer route and compare every word.

    Peer access must already be enabled for the pair (or the driver's own fallback is what
    gets tested).  The copy is asynchronous on a fresh non-blocking stream and polled, so a
    hanging route returns ``timeout`` instead of blocking the caller forever.
    """
    rt = xp.cuda.runtime
    nbytes = int(words) * 4
    with xp.cuda.Device(src):
        source = _pattern(xp, words)
        xp.cuda.Device(src).synchronize()
    with xp.cuda.Device(dst):
        received = xp.zeros(words, dtype=xp.uint32)
        stream = xp.cuda.Stream(non_blocking=True)
        done = xp.cuda.Event(disable_timing=True)
        started = time.monotonic()
        rt.memcpyPeerAsync(received.data.ptr, dst, source.data.ptr, src, nbytes, stream.ptr)
        done.record(stream)
        while not done.done:
            if time.monotonic() - started > timeout_s:
                return {"schema": SCHEMA, "source_device": int(src), "destination_device": int(dst),
                        "bytes": nbytes, "verdict": "timeout", "wrong_words": None,
                        "seconds": time.monotonic() - started, "timeout_s": float(timeout_s)}
            time.sleep(0.001)
        seconds = time.monotonic() - started
        wrong = int((received != _pattern(xp, words)).sum())
    return {"schema": SCHEMA, "source_device": int(src), "destination_device": int(dst),
            "bytes": nbytes, "verdict": "passed" if wrong == 0 else "wrong_data",
            "wrong_words": wrong, "seconds": seconds}


def check_pairs(xp: Any, pairs: Iterable[tuple[int, int]], *, enable_access: bool = True,
                words: int = DEFAULT_WORDS, timeout_s: float = DEFAULT_TIMEOUT_S) -> list[dict[str, Any]]:
    """Check every ordered pair; optionally enable peer access first (already-enabled is fine)."""
    records = []
    for src, dst in pairs:
        src, dst = int(src), int(dst)
        if src == dst:
            continue
        if enable_access:
            with xp.cuda.Device(src):
                try:
                    xp.cuda.runtime.deviceEnablePeerAccess(dst)
                except xp.cuda.runtime.CUDARuntimeError as error:
                    if error.status != 704:  # cudaErrorPeerAccessAlreadyEnabled
                        raise
        records.append(check_pair(xp, src, dst, words=words, timeout_s=timeout_s))
    return records


def failed_pairs(records: Iterable[dict[str, Any]]) -> set[tuple[int, int]]:
    return {(r["source_device"], r["destination_device"]) for r in records if r["verdict"] != "passed"}
