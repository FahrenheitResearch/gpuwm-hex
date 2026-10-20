"""Seal word comparisons of complete native/transcription column executions."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(native, translated, receipt, metadata):
    words, differing, samples = Counter(), Counter(), []
    decks, steps, fields = set(), set(), set()
    with native.open() as left, translated.open() as right:
        for lineno, (a,b) in enumerate(itertools.zip_longest(left,right),1):
            if a is None or b is None:
                raise ValueError(f"Word stream length differs at record {lineno}")
            x,y = a.split(),b.split()
            if x[:-1] != y[:-1]:
                raise ValueError(f"Word address differs at record {lineno}: {x[:-1]} / {y[:-1]}")
            field,deck,step,column,level,bits = x
            key = f"{field}/deck{deck}"
            words[key] += 1
            decks.add(int(deck));steps.add(int(step));fields.add(field)
            if bits != y[-1]:
                differing[key] += 1
                if len(samples) < 32:
                    samples.append({"field":field,"deck":int(deck),"call":int(step),
                                    "column":int(column),"level":int(level),
                                    "native":bits,"translated":y[-1]})
    result = {"scope":"complete MMM host-glibc transcription against strict standalone Fortran; no CUDA or forecast acceptance",
              "native_sha256":digest(native),"translated_sha256":digest(translated),
              "words":sum(words.values()),"differing_words":sum(differing.values()),
              "decks":sorted(decks),"calls":sorted(steps),"arrays":sorted(fields),
              "by_array_deck":{key:{"words":n,"differing_words":differing[key]} for key,n in sorted(words.items())},
              "first_differences":samples,"excluded_words":[],
              "native_defect_policy":"Standalone equality does not substitute for required repeated real native executable measurements.",
              "build":json.loads(metadata.read_text())}
    receipt.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"words":result["words"],"differing_words":result["differing_words"]}))
    return bool(result["differing_words"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--translated",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--metadata",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(compare(args.native,args.translated,args.receipt,args.metadata))
