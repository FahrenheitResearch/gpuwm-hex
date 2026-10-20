"""Summarise hex_stack_sampler output: where wall time went, per thread group and window.

Usage: python sampler_report.py SAMPLES.jsonl.gz [--from T] [--to T] [--epoch] [--thread REGEX]
                                [--top N] [--depth D] [--match SUBSTR]
Times are monotonic seconds (or epoch with --epoch).  Prints, for busy (non-idle)
samples: inclusive seconds per frame (file:function), and the hottest D-deep stack tails.
"""
import argparse
import collections
import gzip
import json
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples")
    p.add_argument("--from", dest="t0", type=float, default=None)
    p.add_argument("--to", dest="t1", type=float, default=None)
    p.add_argument("--epoch", action="store_true")
    p.add_argument("--thread", default=None)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--match", default=None)
    p.add_argument("--idle", action="store_true", help="include idle samples")
    a = p.parse_args()
    rows, stacks = [], {}
    with gzip.open(a.samples, "rt") as stream:
        try:
            for line in stream:
                try:
                    d = json.loads(line)
                except ValueError:
                    break  # a running sampler's last line may be partial
                if "d" in d:
                    stacks[d["d"][0]] = d["d"][1]
                elif "stacks" in d:  # first-version files
                    stacks.update(enumerate(d["stacks"]))
                else:
                    rows.append(d)
        except (EOFError, OSError):
            pass  # a running sampler's gzip stream has no trailer yet
    key = "u" if a.epoch else "t"
    if len(rows) < 2:
        raise SystemExit("too few samples")
    dt = (rows[-1][key] - rows[0][key]) / (len(rows) - 1)
    inclusive, tails, threads = collections.Counter(), collections.Counter(), collections.Counter()
    pattern = re.compile(a.thread) if a.thread else None
    n = 0
    for row in rows:
        t = row[key]
        if (a.t0 is not None and t < a.t0) or (a.t1 is not None and t > a.t1):
            continue
        n += 1
        for name, sid in row["s"]:
            if pattern and not pattern.search(name):
                continue
            stack = stacks[sid]
            if stack.startswith("IDLE;") and not a.idle:
                continue
            if a.match and a.match not in stack:
                continue
            frames = stack.split(";")
            threads[re.sub(r"\d+", "#", name)] += dt
            seen = set()
            for frame in frames:
                fn = frame.rsplit(":", 1)[0]
                if fn not in seen:
                    inclusive[fn] += dt
                    seen.add(fn)
            tails[";".join(f.rsplit(":", 1)[0] if i < len(frames[-a.depth:]) - 1 else f
                           for i, f in enumerate(frames[-a.depth:]))] += dt
    print(f"window samples={n} interval~{dt:.3f}s")
    print("busy thread-seconds by thread group:")
    for name, sec in threads.most_common(15):
        print(f"  {sec:9.1f}  {name}")
    print("inclusive busy seconds by frame:")
    for fn, sec in inclusive.most_common(a.top):
        print(f"  {sec:9.1f}  {fn}")
    print(f"hottest {a.depth}-deep tails:")
    for tail, sec in tails.most_common(a.top // 2):
        print(f"  {sec:9.1f}  {tail}")


if __name__ == "__main__":
    main()
