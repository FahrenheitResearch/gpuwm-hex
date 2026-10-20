"""Replay image custody against an actual supporting Rust sheet packet.

The external scheme packet is validator evidence only, never an i4 forecast.
"""
import argparse
import json
from pathlib import Path
import subprocess

from sheet_evidence import digest, qualify, replay_faults


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('packet', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    expected = json.loads((args.packet / 'expected.json').read_text())
    measured = qualify(args.packet, expected)
    controls = replay_faults(args.packet, expected)
    source = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'schema': 'i4-actual-sheet-validator-replay/v1', 'passed': True,
        'review_revision': revision, 'source_clean': True,
        'supporting_other_scheme_packet_only': True, 'new_forecast_execution': False,
        'i4_images_rendered_or_inspected': False,
        'source_authority_sha256': digest(args.packet / 'authority.json'),
        'expected_sha256': digest(args.packet / 'expected.json'),
        'validator_sha256': digest(Path(__file__).with_name('sheet_evidence.py')),
        'valid_packet': measured, 'corrupt_evidence_controls_refused': controls,
    }, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
