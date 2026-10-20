#!/usr/bin/env python3
"""Measure the unchanged contract's CPU references on its actual inputs."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore.drivers import run_cuda_regional_contract as deck
from hexcore.transport import _atmosphere_vertical_flux, advance_scalars


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--lbc-dir", required=True)
    p.add_argument("--start-time", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profile = cProfile.Profile()
    profile.enable()
    started = time.perf_counter()
    b = deck.Bundle(None, None, grid=Path(args.grid), init=Path(args.init),
                    lbc_dir=Path(args.lbc_dir), start_time=args.start_time)
    times = {"bundle": time.perf_counter() - started}
    started = time.perf_counter()
    off, coefficients, forcing, state = deck._acoustic_inputs(b)
    result = deck.advance_acoustic_step_v841(b.mesh, state, forcing, coefficients,
        dts=deck._DTS, small_step=2, offcentering=off,
        fzm=b.vertical.fzm, fzp=b.vertical.fzp, rdzw=b.vertical.rdzw,
        specified_zone_edge=b.masks.spec_zone_mask_edge,
        specified_zone_cell=b.masks.spec_zone_mask_cell)
    times["acoustic"] = time.perf_counter() - started
    started = time.perf_counter()
    flux = _atmosphere_vertical_flux(b.scalars_stage[0], b.saved.vertical_velocity,
        b.vertical.fzm, b.vertical.fzp, 3, 0.25)
    times["vertical_flux"] = time.perf_counter() - started
    profile.disable()
    pstats.Stats(profile, stream=(out / "profile.txt").open("w")).sort_stats("cumulative").print_stats(60)
    (out / "cpu.json").write_text(json.dumps({"times": times, "acoustic_hashes": {
        name: deck._sha(getattr(result, name)) for name in result.__slots__},
        "vertical_flux_hash": deck._sha(flux), "shape": [b.ncells, b.nlev]}, indent=2))
    print(json.dumps(times), flush=True)


if __name__ == "__main__":
    main()
