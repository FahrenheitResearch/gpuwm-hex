"""Blocking TCP peer link for the two-rank v8.4.1 executor (multi-GPU v1).

Design source: ``MPAS-MULTIGPU-DESIGN-2026-08-17.md`` D3.  The measured 25 GbE
link moves 3.09 GB/s on a single plain TCP stream and does not degrade under
load, so v1 is one persistent socket pair, ``TCP_NODELAY``, length-prefixed
frames, and nothing else.  The frame sequence number doubles as the cross-node
barrier: both sides must issue the same round in the same order or the link
refuses.

Failure policy (design D5): fail-stop with a receipt.  Every receive carries a
timeout; on timeout or a short read the exchange raises ``NetPeerError`` and
the caller aborts the run (truth advances only at completed rounds, so the
last committed step is intact).  There is no silent continuation and no
single-GPU fallback mid-run.

HOST MOVES BYTES ONLY: this module never touches field values.  Payloads are
opaque bytes packed and unpacked by the executor.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from typing import Any

__all__ = [
    "NET_VERSION",
    "NetPeerError",
    "PeerLink",
    "connect_peer",
]

NET_VERSION = "partition-net-v841/1"

_MAGIC = 0x4D504731  # "MPG1"
# magic u32, seq u64, tag 8 bytes (ascii, NUL padded), payload length u64
_HEADER = struct.Struct("!IQ8sQ")
_SOCKET_BUFFER_BYTES = 64 * (1 << 20)
_MAX_FRAME_BYTES = 4 * (1 << 30)


class NetPeerError(RuntimeError):
    """The peer link failed or the two ranks disagreed about the schedule."""


def _encode_tag(tag: str) -> bytes:
    raw = tag.encode("ascii")
    if not raw or len(raw) > 8:
        raise NetPeerError(f"frame tag must be 1..8 ascii bytes: {tag!r}")
    return raw.ljust(8, b"\x00")


class PeerLink:
    """One persistent, sequence-checked, full-duplex frame link.

    ``exchange`` sends this rank's frame from a dedicated sender thread while
    the calling thread receives the peer's frame, so both directions run
    concurrently (the measured link is full duplex at line rate).  The send
    queue is bounded at one frame: the caller cannot run ahead of the barrier.
    """

    def __init__(self, sock: socket.socket, *, timeout_seconds: float = 300.0) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SOCKET_BUFFER_BYTES)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCKET_BUFFER_BYTES)
        except OSError:
            pass  # kernel caps are advisory; throughput was measured without them
        sock.settimeout(float(timeout_seconds))
        self._sock = sock
        self._seq = 0
        self._closed = False
        self._send_queue: "queue.Queue[bytes | None]" = queue.Queue(maxsize=1)
        self._send_done: "queue.Queue[BaseException | None]" = queue.Queue(maxsize=1)
        self._sender = threading.Thread(
            target=self._sender_main, name="v841-net-sender", daemon=True
        )
        self._sender.start()
        self.bytes_sent = 0
        self.bytes_received = 0
        self.frames = 0
        self.wire_seconds = 0.0

    # -- internals ---------------------------------------------------------

    def _sender_main(self) -> None:
        while True:
            frame = self._send_queue.get()
            if frame is None:
                return
            try:
                self._sock.sendall(frame)
            except BaseException as exc:  # noqa: BLE001 - handed to the caller
                self._send_done.put(exc)
            else:
                self._send_done.put(None)

    def _recv_exact(self, count: int) -> bytes:
        chunks: list[bytes] = []
        remaining = int(count)
        while remaining > 0:
            try:
                chunk = self._sock.recv(min(remaining, 1 << 24))
            except socket.timeout as exc:
                raise NetPeerError(
                    f"peer receive timed out with {remaining} of {count} bytes pending"
                ) from exc
            except OSError as exc:
                raise NetPeerError(f"peer receive failed: {exc}") from exc
            if not chunk:
                raise NetPeerError(
                    f"peer closed the link with {remaining} of {count} bytes pending"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    # -- the exchange ------------------------------------------------------

    def exchange(self, tag: str, payload: bytes) -> bytes:
        """Send ``payload`` under ``tag`` and receive the peer's same-round frame.

        Refuses unless the peer's frame carries the identical sequence number
        and tag -- the deterministic-round-schedule barrier.
        """

        if self._closed:
            raise NetPeerError("exchange on a closed peer link")
        seq = self._seq
        self._seq += 1
        started = time.perf_counter()
        header = _HEADER.pack(_MAGIC, seq, _encode_tag(tag), len(payload))
        self._send_queue.put(header + bytes(payload))

        raw = self._recv_exact(_HEADER.size)
        magic, peer_seq, peer_tag_raw, length = _HEADER.unpack(raw)
        if magic != _MAGIC:
            raise NetPeerError(f"bad frame magic 0x{magic:08x}")
        if length > _MAX_FRAME_BYTES:
            raise NetPeerError(f"peer frame of {length} bytes exceeds the frame cap")
        data = self._recv_exact(int(length))

        send_error = self._send_done.get()
        if send_error is not None:
            raise NetPeerError(f"peer send failed: {send_error}") from send_error
        peer_tag = peer_tag_raw.rstrip(b"\x00").decode("ascii", "replace")
        if int(peer_seq) != seq or peer_tag != tag:
            raise NetPeerError(
                "round schedule diverged between the ranks: local "
                f"(seq={seq}, tag={tag!r}) != peer (seq={int(peer_seq)}, "
                f"tag={peer_tag!r})"
            )
        self.frames += 1
        self.bytes_sent += _HEADER.size + len(payload)
        self.bytes_received += _HEADER.size + len(data)
        self.wire_seconds += time.perf_counter() - started
        return data

    def exchange_json(self, tag: str, value: Any) -> Any:
        """Barrier-exchange one canonical-JSON document (handshakes, verdicts)."""

        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        received = self.exchange(tag, payload)
        try:
            return json.loads(received.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NetPeerError(f"peer sent non-JSON under tag {tag!r}") from exc

    def receipt(self) -> dict[str, Any]:
        return {
            "net_version": NET_VERSION,
            "frames": int(self.frames),
            "bytes_sent": int(self.bytes_sent),
            "bytes_received": int(self.bytes_received),
            "wire_seconds": float(self.wire_seconds),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def __enter__(self) -> "PeerLink":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def connect_peer(
    *,
    role: str,
    host: str,
    port: int,
    rendezvous_seconds: float = 1800.0,
    timeout_seconds: float = 300.0,
) -> PeerLink:
    """Rendezvous the two ranks.

    ``role='server'`` binds ``host:port`` and accepts exactly one peer;
    ``role='client'`` retries the connect until the rendezvous window closes
    (the two ranks construct their device stacks at different speeds, so the
    window is generous; once linked, the per-frame timeout takes over).
    """

    deadline = time.monotonic() + float(rendezvous_seconds)
    if role == "server":
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, int(port)))
            listener.listen(1)
            listener.settimeout(max(1.0, deadline - time.monotonic()))
            try:
                accepted, _address = listener.accept()
            except socket.timeout as exc:
                raise NetPeerError(
                    f"no peer connected to {host}:{port} within the "
                    f"{rendezvous_seconds:.0f}s rendezvous window"
                ) from exc
        finally:
            listener.close()
        return PeerLink(accepted, timeout_seconds=timeout_seconds)
    if role == "client":
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            try:
                sock.connect((host, int(port)))
            except OSError as exc:
                last_error = exc
                sock.close()
                time.sleep(2.0)
                continue
            return PeerLink(sock, timeout_seconds=timeout_seconds)
        raise NetPeerError(
            f"could not reach the server rank at {host}:{port} within the "
            f"{rendezvous_seconds:.0f}s rendezvous window: {last_error}"
        )
    raise ValueError(f"role must be 'server' or 'client', not {role!r}")
