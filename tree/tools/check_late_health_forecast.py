#!/usr/bin/env python3
"""Run the real forecast door with a reproducible health refusal injection."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refuse-step", type=int, default=0)
    parser.add_argument("--injection-log", type=Path, required=True)
    parser.add_argument("--profile-first", type=int, default=0)
    parser.add_argument("--profile-count", type=int, default=30)
    parser.add_argument("forecast_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from hexcore.drivers import run_cuda_v841_forecast as forecast
    from hexcore.cli import main as door
    original = forecast.step_health_gate

    def gate(stack, step, cp, **kwargs):
        with args.injection_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"checked_step": step,
                "live_seconds": float(stack["driver"].atmosphere.state.time_seconds)}) + "\n")
        if step == args.refuse_step:
            raise FloatingPointError(f"injected health refusal at step {step}")
        return original(stack, step, cp, **kwargs)

    forecast.step_health_gate = gate
    if args.profile_first:
        execute = forecast.proof.execute_composite_step
        step_index = 0

        def profiled_step(**kwargs):
            nonlocal step_index
            import cupy as cp
            step_index += 1
            if step_index == args.profile_first:
                cp.cuda.runtime.profilerStart()
            active = args.profile_first <= step_index < args.profile_first + args.profile_count
            if active:
                cp.cuda.nvtx.RangePush(f"hex.step.{step_index}")
            try:
                return execute(**kwargs)
            finally:
                if active:
                    cp.cuda.nvtx.RangePop()
                if step_index == args.profile_first + args.profile_count - 1:
                    cp.cuda.runtime.profilerStop()

        forecast.proof.execute_composite_step = profiled_step
    argv = args.forecast_args
    if argv[:1] == ["--"]:
        argv = argv[1:]
    return door(["forecast", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
