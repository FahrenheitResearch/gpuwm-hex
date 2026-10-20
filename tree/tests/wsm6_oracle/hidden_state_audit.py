"""Measure native hidden-state changes with f1's Rust typed-word comparator.

This audits native captures only; it does not replay or validate port state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--words', type=Path, required=True)
    parser.add_argument('--capture', nargs=2, action='append', required=True, metavar=('ROOT', 'PREFIX'))
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    captures = []
    for root, prefix in args.capture:
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        sealed = manifest.pop('dump_root_sha256')
        actual = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert sealed == actual
        rows = {}
        for name in sorted(manifest['files']):
            if not name.startswith(prefix + '.post.state.'):
                continue
            post = root / name
            pre = root / name.replace('.post.', '.pre.', 1)
            for path in (pre, post):
                assert path.resolve().parent == root.resolve() and not path.is_symlink()
                assert digest(path) == manifest['files'][path.name]['sha256']
            comparison = json.loads(subprocess.check_output(
                [str(args.words.resolve()), 'compare', str(pre), str(post)], text=True))
            rows[name[len(prefix + '.post.'):]] = dict(
                comparison=comparison, pre_sha256=digest(pre), post_sha256=digest(post))
        assert rows
        captures.append(dict(capture=prefix, dump_root_sha256=sealed, records=rows,
                             changed_records=sum(not r['comparison']['bitwise_equal'] for r in rows.values())))
    result = dict(scope=__doc__.strip(), helper_sha256=digest(args.words), captures=captures)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print('native hidden-state audit:', len(captures), 'captures;',
          sum(r['changed_records'] for r in captures), 'changed records')


if __name__ == '__main__':
    main()
