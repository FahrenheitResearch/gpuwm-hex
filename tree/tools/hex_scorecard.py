"""Checkout wrapper for the installed gpuwm-hex score command."""
from hexcore.cli import main
import sys
if __name__ == "__main__":
    raise SystemExit(main(["score", *sys.argv[1:]]))
