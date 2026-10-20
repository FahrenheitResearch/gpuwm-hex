"""Prepare a whole-module fresh-allocation and arena-partition diagnostic.

MPAS module_ra_cam.F:364-365,678-679 leaves LW-only tau scratch unwritten.
The pinned full binary's physical words were zero in both-host/count/repeat
witnesses. This candidate tests that initial allocation only; it never resets
an existing arena, supplies native POST storage, or claims other unassigned
native words are zero. Complete word comparisons decide qualification.
The optional fixed stride keeps a shorter final batch from relocating its
column arenas onto storage used by earlier batches.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, output, receipt, optimized=False, fixed_stride=False):
    text = source.read_text()
    before = "if workspace is None:\n            workspace=cp.empty(needed,dtype=cp.uint8)"
    after = "if workspace is None:\n            workspace=cp.zeros(needed,dtype=cp.uint8)"
    if text.count(before) != 1:
        raise ValueError('Expected exactly one boundary arena allocation')
    candidate = text.replace(before, after)
    if optimized:
        option = ", '--Ofast-compile=max'"
        if candidate.count(option) != 1:
            raise ValueError('Expected one compile-time shortcut option')
        candidate = candidate.replace(option, '')
    if fixed_stride:
        before_stride = '(meta,global_meta,workspace,np.uint64(workspace.nbytes),error,np.int32(start)'
        after_stride = ('(meta,global_meta,workspace,\n'
                        '                    # MPAS module_ra_cam.F:364-365,678-679: preserve arena addresses.\n'
                        '                    np.uint64(((workspace.nbytes//min(1024,columns))&~7)*min(1024,columns-start)),\n'
                        '                    error,np.int32(start)')
        if candidate.count(before_stride) != 1:
            raise ValueError('Expected one parallel column arena partition')
        candidate = candidate.replace(before_stride, after_stride)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(candidate)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(dict(
        scope='whole-module initial-allocation candidate; not yet qualified',
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        native_citation='module_ra_cam.F:364-365,678-679',
        transformations=1+int(fixed_stride), cuda_changed=False,
        fixed_stride=fixed_stride, optimized_compile=optimized,
        existing_arena_reset=False,
        native_post_inputs=False, excluded_words=0,
        whole_model_acceptance=False), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--fixed-stride', action='store_true')
    args = parser.parse_args()
    prepare(args.source, args.output, args.receipt, args.optimized, args.fixed_stride)
