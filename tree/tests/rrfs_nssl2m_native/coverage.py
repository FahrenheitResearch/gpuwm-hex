"""Summarize native gcov branches and name every branch not reached by replay."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gcov", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replay", type=Path, help="Authenticate real native Fortran replay for this coverage run")
    a = p.parse_args()
    payload = json.loads(gzip.decompress(a.gcov.read_bytes()))
    source = a.source.read_text().splitlines()
    rows = [f for f in payload["files"] if f["file"].endswith("module_mp_nssl_2mom.F")]
    assert len(rows) == 1
    missed, branches, reached, lines, lines_reached = [], 0, 0, 0, 0
    for row in rows[0]["lines"]:
        lines += 1
        lines_reached += int(row["count"] > 0)
        for index, branch in enumerate(row.get("branches", [])):
            branches += 1
            reached += int(branch["count"] > 0)
            if not branch["count"]:
                number = row["line_number"]
                missed.append(dict(function=row.get("function_name"), line=number, branch=index,
                                   fallthrough=branch.get("fallthrough"), source=source[number-1].strip()))
    result = dict(scope="strict native gcov on synthetic complete-call replay; no forecast/r1 coverage claim",
                  branches=branches, branches_reached=reached, lines=lines, lines_reached=lines_reached,
                  unreached=missed)
    if a.replay:
        import hashlib
        replay=json.loads(a.replay.read_text())
        assert replay['backend']=='native-fortran' and replay['passed']
        assert replay['rows'] and all(row['comparison_complete'] and row['differing_words']==0 for row in replay['rows'])
        result.update(scope='Strict pristine native gcov on the complete authenticated real first-call replay; not CUDA or forecast acceptance',
                      replay_sha256=hashlib.sha256(a.replay.read_bytes()).hexdigest(),
                      capture_member_manifest_sha256=replay['capture_member_manifest_sha256'],
                      source_commit=replay['source_commit'],
                      compared_words=sum(row['compared_words'] for row in replay['rows']),
                      differing_words=0)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Native branches: {reached}/{branches}; lines: {lines_reached}/{lines}")


if __name__ == "__main__":
    main()
