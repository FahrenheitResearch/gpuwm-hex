"""Authenticate full forecast controls without inferring candidate word identity."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def summarize(args):
    cuda_deck = json.loads(args.cuda_deck.read_text())
    native_deck = json.loads(args.native_deck.read_text())
    expected = copy.deepcopy(cuda_deck)
    command = expected['hex_command']
    overlay = "config_convection_backend='hex-cuda'"
    index = command.index(overlay)
    if command[index - 1] != '--physics' or command.count(overlay) != 1:
        raise ValueError('control must remove exactly one convection backend overlay')
    del command[index - 1:index + 1]
    expected['slot_backends']['convection'] = 'native-cpu'
    expected['id'] = native_deck['id']
    if expected != native_deck:
        raise ValueError('control changes scientific inputs, providers or other options')
    cuda = json.loads(args.cuda_receipt.read_text())
    native = json.loads(args.native_receipt.read_text())
    for result in (cuda, native):
        if (result.get('evidence_kind') != 'FORECAST'
                or result.get('process_count') != cuda_deck['process_count']
                or result.get('failed_model_exit') is not None):
            raise ValueError('control needs actual forecasts at the same process count')
    fields = native['comparison']['fields']
    computed = [row for row in fields if row['status'] == 'compared']
    if not computed:
        raise ValueError('control has no computed fields')
    differences = [row for row in computed if row['differing_words']]
    result = dict(
        complete=True, diagnostic_only=True, acceptance_proven=False,
        pairwise_word_identity_proven=False,
        native_process_count=cuda_deck['process_count'],
        cuda_deck=pin(args.cuda_deck), native_deck=pin(args.native_deck),
        cuda_receipt=pin(args.cuda_receipt), all_native_receipt=pin(args.native_receipt),
        all_native_computed_words=sum(row['compared_words'] for row in computed),
        all_native_differing_words=sum(row['differing_words'] for row in computed),
        all_native_missing_samples=len(fields) - len(computed),
        first_all_native_difference=differences[0] if differences else None,
        all_field_diff_metadata_identical=fields == cuda['comparison']['fields'],
        scope='Both full-deck forecasts use the same scientific inputs and native '
              'reference. Equal field difference metadata does not prove pairwise '
              'word identity of the two candidate outputs.')
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cuda-deck', 'native-deck', 'cuda-receipt', 'native-receipt', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args())))
