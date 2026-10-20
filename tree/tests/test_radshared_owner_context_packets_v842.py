"""Review packet integrity; synthetic metadata never counts as a forecast."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('p21_owner_packets',
    Path(__file__).parent / 'radshared_v842' / 'owner_context_packets.py')
packets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packets)


@pytest.fixture
def packet_set(tmp_path):
    packet = dict(schema='fixture-review', proposal_only=True, owner_approved=False,
        usable_as_baseline=False, completed_contexts=[
            dict(request_index=index, fixture_metadata='x' * 600)
            for index in range(4)], process_count=1)
    out = tmp_path / 'review.json'
    packets.write_packet(packet, out, split=True, limit=1500)
    return out, packet


def test_packet_reconstructs_original_bytes(packet_set):
    out, packet = packet_set
    assert packets.encoded(packets.load_packet(out)) == packets.encoded(packet)
    assert all(Path(part['path']).stat().st_size < 1500
               for part in json.loads(out.read_text())['parts'])


@pytest.mark.parametrize('fault', ['part_hash', 'part_order', 'part_duplicate',
    'context_count', 'approval', 'key_order'])
def test_corrupt_review_set_refuses(packet_set, fault):
    out, _ = packet_set
    manifest = json.loads(out.read_text())
    if fault == 'part_hash':
        part = Path(manifest['parts'][0]['path'])
        part.write_text(part.read_text() + ' ')
    elif fault == 'part_order':
        manifest['parts'].reverse()
    elif fault == 'part_duplicate':
        manifest['parts'].append(manifest['parts'][0])
    elif fault == 'context_count':
        manifest['completed_contexts'] += 1
    elif fault == 'approval':
        manifest['packet_metadata']['owner_approved'] = True
    else:
        manifest['packet_key_order'].append(manifest['packet_key_order'][0])
    out.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        packets.load_packet(out)
