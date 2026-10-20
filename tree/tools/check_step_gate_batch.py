#!/usr/bin/env python3
"""Instrument the real forecast door for same-step gate failure and nsys checks.

This tool changes fields or a gate word only when --failure is named. It is
never imported by the product. Run it through the card queue with the same
sealed forecast arguments as the measured arms.
"""
import argparse
import json
from pathlib import Path
import sys
import statistics
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure", choices=("coupling", "clamp", "recovery", "final",
                        "rho_nan", "rho_zero", "theta_zero", "exner_zero", "velocity", "hydrometeor"))
    parser.add_argument("--profile-window", help="inclusive step range, for cudaProfilerApi capture")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("forecast_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.forecast_args
    if argv[:1] == ["--"]:
        argv = argv[1:]
    import cupy as cp
    from hexcore import cli, cuda_driver, cuda_physics_v841 as physics
    from hexcore.drivers import run_cuda_v841_forecast as forecast
    proof = forecast.proof
    injected = []
    def install_function(name, mutate):
        original = getattr(physics, name)
        def wrapped(*values, **kwargs):
            mutate(values, kwargs)
            injected.append(name)
            return original(*values, **kwargs)
        setattr(physics, name, wrapped)
    if args.failure == "coupling":
        install_function("couple_raw_column_physics_v841",
                         lambda values, kw: values[0].dtheta.__setitem__((0, 0), cp.nan))
    elif args.failure == "clamp":
        install_function("clamp_wsm6_scalars_in_place_v841",
                         lambda values, kw: values[0].__setitem__((0, 0, 0), cp.nan))
    elif args.failure == "recovery":
        install_function("recover_post_rk_wsm6_state_v841",
                         lambda values, kw: values[1].theta.__setitem__((0, 0), cp.nan))
    elif args.failure:
        original = cuda_driver.accumulate_finite_array_cuda_v841
        def at_commit(*values, **kwargs):
            result = original(*values, **kwargs)
            frame = sys._getframe(1)
            if frame.f_code.co_name != "commit_post_wsm6_candidate":
                return result
            atmosphere = frame.f_locals["committed"]
            if args.failure == "final":
                kwargs["invalid_flag"].fill(1)
            elif args.failure == "rho_nan":
                atmosphere.state.rho[0, 0] = cp.nan
            elif args.failure == "rho_zero":
                atmosphere.state.rho[0, 0] = 0
            elif args.failure == "theta_zero":
                atmosphere.saved.theta_m[0, 0] = 0
            elif args.failure == "exner_zero":
                atmosphere.saved.exner[0, 0] = 0
            elif args.failure == "velocity":
                atmosphere.saved.vertical_velocity[0, 0] = 201
            elif args.failure == "hydrometeor":
                atmosphere.state.scalars[1, 0, 0] = -1
            injected.append(args.failure)
            return result
        cuda_driver.accumulate_finite_array_cuda_v841 = at_commit
    first = last = None
    if args.profile_window:
        first, last = map(int, args.profile_window.split(":"))
        if not 1 <= first <= last:
            raise ValueError("profile window must ascend from step one or later")
    original_composite = proof.execute_composite_step
    original_health = forecast.step_health_gate
    count = 0
    composite_seconds = 0
    step_seconds = []
    def composite(*values, **kwargs):
        nonlocal count, composite_seconds
        count += 1
        if count == first:
            cp.cuda.get_current_stream().synchronize()
            cp.cuda.runtime.profilerStart()
        if last is not None and count == last + 1:
            cp.cuda.get_current_stream().synchronize()
            cp.cuda.runtime.profilerStop()
        if args.profile_window:
            cp.cuda.nvtx.RangePush(f"hex.step.{count}")
        started = time.perf_counter()
        try:
            return original_composite(*values, **kwargs)
        finally:
            composite_seconds = time.perf_counter() - started
            if args.profile_window:
                cp.cuda.nvtx.RangePop()
    def health(*values, **kwargs):
        started = time.perf_counter()
        try:
            return original_health(*values, **kwargs)
        finally:
            step_seconds.append(composite_seconds + time.perf_counter() - started)
            if not hasattr(forecast, "install_step_gate_batch"):
                forecast.time_ledger.emit("forecast.step", attrs={
                    "step": count, "host_step_seconds": step_seconds[-1],
                    "basis": "matched observation of composite plus health"})
    proof.execute_composite_step = composite
    forecast.step_health_gate = health
    rc = cli.main(argv)
    output = Path(argv[argv.index("--out") + 1])
    receipt_path = output / "cuda-v841-forecast-receipt.json"
    report = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    record = {"returncode": rc, "argv": argv, "failure": args.failure,
              "profile_window": args.profile_window, "injected": injected,
              "host_step_median_seconds": statistics.median(step_seconds) if step_seconds else None,
              "host_step_seconds": step_seconds,
              "forecast": report.get("forecast"),
              "published_history": [p.name for p in sorted(output.glob("*.nc"))],
              "restart_paths": [str(p.relative_to(output)) for p in output.rglob("*restart*")]}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if args.failure and not injected:
        raise RuntimeError("failure injection was never reached")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
