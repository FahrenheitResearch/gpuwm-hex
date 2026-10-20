#!/usr/bin/env python3
"""Capture this FCT row from a real forecast and replay it on the same device.

Run through the assigned GPU pool helper. Pass the shape's ordinary forecast
arguments after --. Capture stops before the first fused launch; device input
copies stay in memory and are released on exit. No raw state is written.

Run strict and fast in separate processes. For a separate nsys run, select
--profile baseline or fused and use capture-range=cudaProfilerApi. The trace
contains row kernels and input reset copies; reduce only the named row kernels.
This replay's repeat count is not a forecast step count or a whole-step pair.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FUSED = "fct_horizontal_low_order_scale_elem_v841"
BASELINE = ("fct_horizontal_low_order_elem_v841", "fct_scale_elem_v841")


class CaptureDone(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerics", choices=("strict", "fast"), required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--profile", choices=("baseline", "fused"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("forecast", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forecast = args.forecast[1:] if args.forecast[:1] == ["--"] else args.forecast
    if not forecast or args.repeats < 1:
        parser.error("forecast arguments and a positive repeat count are required")
    from hexcore import numerics
    numerics.set_mode(args.numerics)
    from hexcore import cuda_transport_v841 as transport
    from hexcore.cli import _select_numerics, build_parser, main as forecast_main

    # The forecast flag or namelist can override the process environment.
    # Resolve the same selection as the door before spending a GPU slot.
    forecast_options = build_parser().parse_args(forecast)
    _select_numerics(forecast_options)
    if numerics.mode() != args.numerics:
        parser.error(f"forecast selects {numerics.mode()} numerics; replay requests {args.numerics}")

    cp = transport._cupy()
    captured = {}
    original = transport._launch

    def capture(name, count, arguments, cache):
        if name == FUSED:
            captured.update(count=count, cache=cache,
                            arguments=tuple(v.copy() if isinstance(v, cp.ndarray) else v
                                            for v in arguments))
            raise CaptureDone()
        return original(name, count, arguments, cache)

    began = time.monotonic()
    transport._launch = capture
    try:
        forecast_main(forecast)
    except CaptureDone:
        pass
    finally:
        transport._launch = original
    if not captured:
        raise RuntimeError("forecast did not launch the fused FCT row; check its mesh and limiter config")
    cp.cuda.get_current_stream().synchronize()
    capture_seconds = time.monotonic() - began
    data, count, cache = captured["arguments"], captured["count"], captured["cache"]
    nt, nl, nc, ne = (int(v) for v in data[:4])
    if args.shape == "H60" and (nc, nl) != (163842, 55):
        raise RuntimeError(f"H60 capture has cells={nc}, levels={nl}")
    if args.shape == "H15" and (nc, nl) != (2621442, 55):
        raise RuntimeError(f"H15 capture has cells={nc}, levels={nl}")
    outputs = tuple(v.copy() for v in data[11:14])

    def reset():
        for target, initial in zip(outputs, data[11:14]):
            cp.copyto(target, initial)

    def run(which):
        if which == "fused":
            original(FUSED, count, (*data[:11], *outputs, *data[14:]), cache)
        else:
            original(BASELINE[0], count, (*data[:11], *outputs), cache)
            original(BASELINE[1], count, (*data[:3], *data[14:], *outputs), cache)

    # Collect compile resources before executing either replay variant.
    resources = {}
    driver = ctypes.CDLL("libcuda.so.1")
    occupancy = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
    occupancy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
                         ctypes.c_int, ctypes.c_size_t]
    occupancy.restype = ctypes.c_int
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    for name in (*BASELINE, FUSED):
        kernel = transport._kernel(name, cache)
        # An owner replay uses a plain fixed-step cache, with no launch observer.
        attributes = dict(kernel.attributes)
        blocks = ctypes.c_int()
        rc = occupancy(ctypes.byref(blocks), ctypes.c_void_p(kernel.kernel.ptr), 128, 0)
        if rc:
            raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor({name}): {rc}")
        resources[name] = dict(attributes, block_threads=128, blocks_per_sm=blocks.value,
                               theoretical_occupancy=blocks.value * 128 / props["maxThreadsPerMultiProcessor"])
    # Warm both sides before the separate profile capture.
    reset()
    run("baseline")
    expected = tuple(v.copy() for v in outputs)
    reset()
    run("fused")
    differences = [int(cp.count_nonzero(a.view(cp.uint32) != b.view(cp.uint32)))
                   for a, b in zip(outputs, expected)]
    if args.profile:
        cp.cuda.get_current_stream().synchronize()
        cp.cuda.runtime.profilerStart()
        for _ in range(args.repeats):
            reset()
            run(args.profile)
        cp.cuda.get_current_stream().synchronize()
        cp.cuda.runtime.profilerStop()
    report = dict(shape=args.shape, numerics=args.numerics,
                  forecast_arguments=forecast, cells=nc, levels=nl, edges=ne,
                  species_this_pass=nt, capture_seconds=capture_seconds,
                  differing_words=dict(zip(("mass", "scale_in", "scale_out"), differences)),
                  words_compared=sum(v.size for v in outputs), resources=resources,
                  row_launches_per_pass=dict(baseline=2, fused=1),
                  profile_variant=args.profile, profile_repeats=args.repeats if args.profile else 0,
                  whole_step_measured=False)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return int(any(differences))


if __name__ == "__main__":
    raise SystemExit(main())
