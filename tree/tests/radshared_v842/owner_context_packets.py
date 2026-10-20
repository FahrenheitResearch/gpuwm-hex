"""Lossless, hash-bound packaging of unapplied owner-review metadata."""
import hashlib
import json
from pathlib import Path

LIMIT = 10_000_000
SET_SCHEMA = 'p21-owner-context-packet-set/v1'
PART_SCHEMA = 'p21-owner-context-packet-part/v1'


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def load_packet(path, resolve=Path):
    path = Path(path)
    assert path.stat().st_size < LIMIT
    value = json.loads(path.read_text())
    if value['schema'] != SET_SCHEMA:
        return value
    packet = dict(value['packet_metadata'])
    assert packet['proposal_only'] and not packet['owner_approved']
    assert not packet['usable_as_baseline']
    assert 'completed_contexts' not in packet
    contexts = []
    paths = set()
    for item in value['parts']:
        part_path = resolve(item['path'])
        assert str(part_path) not in paths
        paths.add(str(part_path))
        content = part_path.read_bytes()
        assert len(content) < LIMIT
        assert hashlib.sha256(content).hexdigest() == item['sha256']
        part = json.loads(content)
        assert set(part) == {'schema', 'completed_contexts'}
        assert part['schema'] == PART_SCHEMA
        assert len(part['completed_contexts']) == item['completed_contexts'] > 0
        contexts.extend(part['completed_contexts'])
    indices = [row['request_index'] for row in contexts]
    assert indices == sorted(set(indices))
    assert len(contexts) == value['completed_contexts']
    packet['completed_contexts'] = contexts
    if 'packet_key_order' in value:
        order = value['packet_key_order']
        assert len(order) == len(set(order)) == len(packet)
        assert set(order) == set(packet)
        packet = {key: packet[key] for key in order}
    return packet


def write_packet(packet, out, split=False, limit=LIMIT):
    assert 1_000 <= limit <= LIMIT
    out = Path(out)
    assert not out.exists()
    content = encoded(packet)
    if not split:
        assert len(content) < limit
        out.write_bytes(content)
        return
    metadata = {key: value for key, value in packet.items()
                if key != 'completed_contexts'}
    groups = []
    group = []
    for row in packet['completed_contexts']:
        candidate = group + [row]
        if len(encoded(dict(schema=PART_SCHEMA, completed_contexts=candidate))) >= limit:
            assert group, 'A single complete context exceeds the receipt limit'
            groups.append(group)
            group = [row]
        else:
            group = candidate
    if group:
        groups.append(group)
    assert groups
    files = []
    parts = []
    for index, group in enumerate(groups, 1):
        path = out.with_name(out.stem + f'.part-{index:03d}.json')
        content = encoded(dict(schema=PART_SCHEMA, completed_contexts=group))
        assert len(content) < limit and not path.exists()
        files.append((path, content))
        parts.append(dict(path=str(path.resolve()),
                          sha256=hashlib.sha256(content).hexdigest(),
                          completed_contexts=len(group)))
    manifest = encoded(dict(schema=SET_SCHEMA, packet_metadata=metadata,
                            packet_key_order=list(packet),
                            completed_contexts=len(packet['completed_contexts']),
                            parts=parts))
    assert len(manifest) < limit
    # Validate the complete output set before publishing any part.
    for path, content in files:
        path.write_bytes(content)
    out.write_bytes(manifest)
