"""Optional schedule-slot transport with one producer rendezvous per round.

Packets belong to an immutable schedule slot, never to a shared field schema.
The coordinator must bracket a stopped rank group with begin_step/end_step;
end_step drains producers and transfers before any slot can be reused. This
is an eager transport optimization, not a CUDA graph implementation.
"""

from __future__ import annotations

import math
import operator
import time
from typing import Any, Sequence

from .partition_samehost_transport_v841 import (
    SameHostTransport, SameHostTransportError, _Buffers, _Round, _Submission,
)


class StepSlotTransport(SameHostTransport):
    """Byte-only exchange with private packets and independent tagged rounds.

    begin_step(step, rounds) accepts a round count or a sequence of round
    labels. Tags use the existing ``step:position:label`` wire format. Every
    rank must issue positions in order. The first step fixes all slot schemas.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._step_id = None
        self._last_step = None
        self._expected_rounds = None
        self._expected_labels = None
        self._slot_schemas: dict[int, tuple[Any, ...]] = {}
        self._slot_labels: dict[int, str] = {}
        self._slot_rounds: dict[int, _Round] = {}
        self._positions = [0] * self.n_ranks
        self._active_exchanges = 0
        self._ending = False
        self._producer_streams: dict[tuple[int, int], Any] = {}
        self.pipeline_stats = {
            "steps_started": 0, "steps_completed": 0,
            "producer_barrier_wait_calls": 0, "producer_wait_seconds": 0.0,
            "consumer_wait_events": 0, "second_barrier_wait_calls": 0,
        }

    def _live(self) -> None:
        if self._error is not None:
            raise self._error
        if self._closed or self._stopping:
            raise SameHostTransportError("schedule-slot transport is closed")

    def abort(self, reason: Any) -> None:
        """Keep every incomplete round owned while waking producer waiters."""
        with self._condition:
            if self._error is None:
                self._error = SameHostTransportError(str(reason))
                self._failed_rounds.extend(self._slot_rounds.values())
            self._condition.notify_all()

    def _refuse(self, error: BaseException) -> None:
        self.abort(f"schedule-slot exchange failed: {error}")
        if isinstance(error, SameHostTransportError):
            raise error
        raise self._error from error

    def begin_step(self, step: int, rounds: int | Sequence[str]) -> None:
        """Open a group only after the previous end_step completed its drain."""
        try:
            step = operator.index(step)
            if isinstance(rounds, int):
                count, labels = operator.index(rounds), None
            else:
                labels = tuple(rounds)
                count = len(labels)
                if any(not isinstance(label, str) or not label or ":" in label for label in labels):
                    raise SameHostTransportError("round labels must be nonempty strings without colons")
            if count < 1:
                raise SameHostTransportError("a schedule-slot step requires at least one round")
            with self._condition:
                self._live()
                if self._step_id is not None or self._ending or self._active_exchanges:
                    raise SameHostTransportError("previous step has not completed end_step and GPU drain")
                if self._last_step is not None and step != self._last_step + 1:
                    raise SameHostTransportError("schedule-slot step clock must advance by one")
                if self._expected_rounds is not None and count != self._expected_rounds:
                    raise SameHostTransportError("schedule-slot round count changed after preparation")
                if self._expected_labels is not None and labels != self._expected_labels:
                    raise SameHostTransportError("schedule-slot round labels changed after preparation")
                self._expected_rounds, self._expected_labels = count, labels
                self._step_id = step
                self._positions = [0] * self.n_ranks
                self._slot_rounds.clear()
                self.pipeline_stats["steps_started"] += 1
        except BaseException as error:
            self._refuse(error)

    def _tag(self, tag: Any) -> tuple[int, int, str]:
        if not isinstance(tag, str):
            raise SameHostTransportError("schedule-slot tag must be step:position:label")
        parts = tag.split(":")
        if len(parts) != 3 or not parts[2]:
            raise SameHostTransportError("schedule-slot tag must be step:position:label")
        try:
            step, position = int(parts[0]), int(parts[1])
        except ValueError as error:
            raise SameHostTransportError("schedule-slot tag has a noninteger step or position") from error
        if parts[0] != str(step) or parts[1] != str(position):
            raise SameHostTransportError("schedule-slot tag step and position must use canonical integers")
        return step, position, parts[2]

    def _slot_buffers(self, rank: int, position: int, schema: Any, fields: Any) -> _Buffers:
        key = (position, schema)
        buffers = self._cache[rank].get(key)
        if buffers is None:
            buffers = _Buffers(
                {peer: self._make_packet(fields, table, "send", rank=rank) for peer, table in self._peers[rank].items()},
                {peer: self._make_packet(fields, table, "recv", rank=rank) for peer, table in self._peers[rank].items()},
            )
            self._cache[rank][key] = buffers
        return buffers

    def exchange(self, rank: int, tag: Any, fields: Any, *, interior_work=None) -> dict[str, int | float]:
        started = time.perf_counter()
        entered = False
        try:
            rank = operator.index(rank)
            if not 0 <= rank < self.n_ranks:
                raise SameHostTransportError(f"invalid rank {rank}")
            step, position, label = self._tag(tag)
            fields = tuple(fields)
            schema = self._schema(rank, fields)
            with self._condition:
                self._live()
                if self._ending or self._step_id is None or step != self._step_id:
                    raise SameHostTransportError(f"rank {rank} exchange tag is outside the open step")
                if not 0 <= position < self._expected_rounds:
                    raise SameHostTransportError(f"rank {rank} exchange position is outside the declared schedule")
                if position != self._positions[rank]:
                    raise SameHostTransportError(f"rank {rank} expected slot {self._positions[rank]}, got {position}")
                expected_label = (self._expected_labels[position] if self._expected_labels is not None
                                  else self._slot_labels.get(position, label))
                if label != expected_label:
                    raise SameHostTransportError(f"slot {position} round label changed")
                if position in self._slot_schemas and schema != self._slot_schemas[position]:
                    raise SameHostTransportError(f"slot {position} field schema changed")
                self._slot_schemas.setdefault(position, schema)
                self._slot_labels.setdefault(position, label)
                round_ = self._slot_rounds.get(position)
                if round_ is None:
                    round_ = _Round(self.pipeline_stats["steps_started"] * self._expected_rounds + position,
                                    time.monotonic() + self.timeout, tag=tag, schema=schema)
                    self._slot_rounds[position] = round_
                self._check_live(round_)
                if round_.tag != tag or round_.schema != schema:
                    raise SameHostTransportError(f"slot {position} rank {rank} disagrees on tag or schema")
                if rank in round_.entered:
                    raise SameHostTransportError(f"slot {position} rank {rank} entered twice")
                round_.entered.add(rank)
                round_.reserved_fields.append(fields)
                self._active_exchanges += 1
                entered = True
            if self._cpu:
                stream = None
                buffers = self._slot_buffers(rank, position, schema, fields)
                self._copy_packets(fields, buffers.send.values(), buffers.send_batch, scatter=False)
            else:
                with self.xp.cuda.Device(self.devices[rank]):
                    stream = self.xp.cuda.get_current_stream()
                    # A descriptor upload or pack may enqueue successfully
                    # before a later API fails. Retain its stream before the
                    # first enqueue so direct failure drain can fence it.
                    with self._condition:
                        self._check_live(round_)
                        self._producer_streams[self.devices[rank], int(stream.ptr)] = stream
                    buffers = self._slot_buffers(rank, position, schema, fields)
                    self._copy_packets(fields, buffers.send.values(), buffers.send_batch, scatter=False, stream=stream)
            submission = _Submission(fields, buffers, stream)
            if not self._cpu:
                with self.xp.cuda.Device(self.devices[rank]):
                    submission.owners.extend((buffers.send_batch.descriptor, *buffers.send_batch.owners))
                    submission.produced = self.xp.cuda.Event(disable_timing=True)
                    submission.produced.record(stream)
            with self._condition:
                self._check_live(round_)
                round_.submissions[rank] = submission
                self._condition.notify_all()
                waited = time.perf_counter()
                while len(round_.submissions) != self.n_ranks:
                    self._check_live(round_)
                    self.pipeline_stats["producer_barrier_wait_calls"] += 1
                    self._condition.wait(max(0.0, round_.deadline - time.monotonic()))
                self.pipeline_stats["producer_wait_seconds"] += time.perf_counter() - waited
            self._receive(rank, round_, submission)
            if interior_work is not None:
                interior_work()
            if not self._cpu:
                # Private slot packets need no other rank's received event.
                # This rank still must consume its own halos after unpack.
                self._fence_consumers(rank, round_, submission, all_ranks=False)
                if self.release_fields_after_fence:
                    self._release_rank_fields(round_, submission, fields)
            sent = sum(int(packet.buffer.nbytes) for packet in buffers.send.values())
            received = sum(int(packet.buffer.nbytes) for packet in buffers.recv.values())
            with self._condition:
                self._check_live(round_)
                if not self._cpu:
                    self.pipeline_stats["consumer_wait_events"] += 1
                round_.completed.add(rank)
                self._positions[rank] += 1
                if len(round_.completed) == self.n_ranks:
                    moved = 0
                    for source_rank, source in round_.submissions.items():
                        for peer, packet in source.buffers.send.items():
                            size = int(packet.buffer.nbytes)
                            moved += size
                            self.stats[f"{self.paths[self.devices[source_rank], self.devices[peer]]}_bytes"] += size
                    self.stats["rounds"] += 1
                    self.stats["bytes"] += moved
                    self.stats["seconds"] += max(0.0, time.monotonic() - (round_.deadline - self.timeout))
                    self._queue_completion(round_)
                    round_.done = True
                    self._condition.notify_all()
            return {"bytes_sent": sent, "bytes_received": received,
                    "seconds": time.perf_counter() - started}
        except BaseException as error:
            self._refuse(error)
        finally:
            if entered:
                with self._condition:
                    self._active_exchanges -= 1
                    self._condition.notify_all()

    def end_step(self, step: int) -> None:
        """Check a stopped complete group, then drain before allowing reuse."""
        try:
            step = operator.index(step)
            with self._condition:
                self._live()
                if self._ending or step != self._step_id or self._active_exchanges:
                    raise SameHostTransportError("end_step requires the open step and stopped rank workers")
                if (len(self._slot_rounds) != self._expected_rounds
                        or any(position != self._expected_rounds for position in self._positions)
                        or any(not round_.done or len(round_.completed) != self.n_ranks
                               for round_ in self._slot_rounds.values())):
                    raise SameHostTransportError("end_step refused a partial schedule-slot rank group")
                self._ending = True
            self.synchronize()
            with self._condition:
                self._live()
                self._last_step, self._step_id = step, None
                self._slot_rounds.clear()
                self._ending = False
                self.pipeline_stats["steps_completed"] += 1
        except BaseException as error:
            self._refuse(error)

    def synchronize(self, *, timeout: float | None = None) -> None:
        """Drain transfers and every producer/consumer stream without reuse."""
        seconds = self.timeout if timeout is None else float(timeout)
        if not math.isfinite(seconds) or seconds <= 0:
            raise SameHostTransportError("drain timeout must be finite and positive")
        deadline = time.monotonic() + seconds
        with self._condition:
            if self._active_exchanges:
                raise SameHostTransportError("cannot drain while schedule-slot exchanges are being issued")
            if self._step_id is not None and self._error is None and (
                    len(self._slot_rounds) != self._expected_rounds
                    or any(position != self._expected_rounds for position in self._positions)
                    or any(not round_.done or len(round_.completed) != self.n_ranks
                           for round_ in self._slot_rounds.values())):
                error = SameHostTransportError("cannot drain a partial schedule-slot rank group")
                self.abort(error)
                raise error
            streams = tuple(self._producer_streams.items())
        fences = []
        if not self._cpu:
            for (device, _), stream in streams:
                with self.xp.cuda.Device(device):
                    event = self.xp.cuda.Event(disable_timing=True)
                    event.record(stream)
                    fences.append((device, event))
        super().synchronize(timeout=max(1e-9, deadline - time.monotonic()))
        for device, event in fences:
            while True:
                with self.xp.cuda.Device(device):
                    if event.done:
                        break
                if time.monotonic() >= deadline:
                    raise SameHostTransportError("schedule-slot stream drain timed out; ownership is retained")
                time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))

    def close(self, *, timeout: float | None = None) -> None:
        incomplete = None
        with self._condition:
            if self._step_id is not None and self._error is None:
                incomplete = SameHostTransportError("close requires checked end_step for the open rank group")
                self.abort(incomplete)
        super().close(timeout=timeout)
        if incomplete is not None:
            raise incomplete

    def receipt(self) -> dict[str, Any]:
        with self._condition:
            receipt = super().receipt()
            slot_bytes = []
            pinned = 0
            for cache in self._cache:
                rank_bytes = {}
                for (position, _), buffers in cache.items():
                    packets = tuple(buffers.send.values()) + tuple(buffers.recv.values())
                    rank_bytes[str(position)] = sum(int(packet.buffer.nbytes) for packet in packets)
                    pinned += sum(int(packet.buffer.nbytes) for packet in packets if packet.host is not None)
                slot_bytes.append(rank_bytes)
            receipt.update({
                "transport_version": "partition-samehost-slot-pipeline-v841/1",
                "pipeline": dict(self.pipeline_stats), "open_step": self._step_id,
                "expected_rounds": self._expected_rounds, "prepared_slots": len(self._slot_schemas),
                "packet_cache_law": "private schedule slots; immutable schema; GPU drain before step reuse",
                "slot_packet_bytes_by_rank": slot_bytes,
                "retained_slot_packet_bytes": sum(sum(rank.values()) for rank in slot_bytes),
                "retained_pinned_packet_bytes": pinned,
                "consumer_fences": "own receive only; producer and consumer streams drain at end_step",
            })
            return receipt


__all__ = ["StepSlotTransport"]
