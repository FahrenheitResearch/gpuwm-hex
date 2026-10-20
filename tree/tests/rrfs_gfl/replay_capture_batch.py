#!/usr/bin/env python3
"""Run a complete sealed GFL capture inside one GPU allocation/compilation."""
from pathlib import Path
import sys
from replay_capture import main


if __name__ == '__main__':
    dump, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True)
    captures = sorted(dump.glob('*.pre.itimestep.bin'))
    if not captures:
        raise ValueError('GFL dump has no complete wrapper calls')
    for record in captures:
        prefix = record.name.removesuffix('.pre.itimestep.bin')
        sys.argv = [sys.argv[0], str(dump / prefix), str(output / prefix)]
        main()
