"""Replay the whole module in fresh processes across native option branches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def scenarios():
    rows = [dict(name="default-2m", moments=2), dict(name="default-3m", moments=3),
            dict(name="diagnostics-off", options=dict(diagnose=0, dbz_due=0, reqc=0, reqi=0, reqs=0)),
            dict(name="subcycles", options=dict(nmicro=3, dt=30.0)),
            dict(name="sedimentation-only", overrides=dict(isedonly=1)),
            dict(name="accurate-sedimentation", overrides=dict(do_accurate_sedimentation=True, interval_sedi_vt=2)),
            dict(name="bolton-saturation", overrides=dict(iqvsopt=1)),
            dict(name="old-limiter", overrides=dict(ioldlimiter=1)),
            dict(name="new-limiter", overrides=dict(ioldlimiter=0)),
            dict(name="bigg-old-melting", overrides=dict(ibiggopt=1, ihmlt=1)),
            dict(name="crystal-disk", overrides=dict(ixtaltype=2)),
            dict(name="snow-melt-diameter", overrides=dict(snowmeltdia=0.0002)),
            dict(name="ccn-restore", overrides=dict(restoreccn=True, ccntimeconst=10.0)),
            dict(name="third-moment-rescale-off", moments=3,
                 overrides=dict(rescale_low_alpha=False, rescale_low_alphar=False,
                                rescale_low_alphah=False, rescale_low_alphahl=False, rescale_high_alpha=False))]
    for option in (2,3,4,10,-1,-2,-10,-11):
        rows.append(dict(name=f"ice-nuc-{option}", overrides=dict(icenucopt=option)))
    for option in (0,1,2,4,6):
        rows.append(dict(name=f"drag-{option}", overrides=dict(icdx=option,icdxhl=option)))
    for option in (1,3):
        rows.append(dict(name=f"snow-fall-{option}", overrides=dict(isnowfall=option)))
    for option in (1,2):
        rows.append(dict(name=f"cloud-shape-{option}", overrides=dict(idiagnosecnu=option)))
    for profile in ("dry","warm_rain","mixed_ice"):
        rows.append(dict(name="r1-"+profile,moments=3,r1_profile=profile))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "host", "source", "out", "scratch"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in scenarios():
        name = scenario["name"]
        overrides = a.out / (name + ".namelist.json")
        options = a.out / (name + ".options.json")
        receipt = a.out / (name + ".json")
        overrides.write_text(json.dumps(scenario.get("overrides", {})))
        options.write_text(json.dumps(scenario.get("options", {})))
        command = [sys.executable, str(ROOT / "replay.py"), "--native", str(a.native.resolve()),
                   "--host", str(a.host.resolve()), "--source", str(a.source.resolve()),
                   "--moments", str(scenario.get("moments", 2)),
                   "--overrides", str(overrides.resolve()), "--options", str(options.resolve()),
                   "--receipt", str(receipt.resolve()), "--scratch", str((a.scratch/name).resolve())]
        if scenario.get("r1_profile"):
            command += ["--r1-profile", scenario["r1_profile"]]
        with (a.out / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        rows.append(dict(scenario=scenario, exit_code=result.returncode, receipt=str(receipt)))
        (a.out / "matrix.json").write_text(json.dumps(dict(scope="synthetic whole-module host replay; no CUDA/forecast claim", runs=rows), indent=2) + "\n")
        print(name, result.returncode, flush=True)
    return int(any(row["exit_code"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
