"""Summarize measured ensemble receipts without reading simulation data."""
import argparse
import json
from pathlib import Path
from hexcore.ensemble import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    data = json.loads((a.run / "ensemble-receipt.json").read_text())
    peak, baseline = {}, {}
    for sample in data.get("memory_samples", []):
        for text in sample["gpu_rows"]:
            uuid, used, utilization = (part.strip() for part in text.split(","))
            if uuid not in [row["visible_token"] for row in data["cards"]]:
                continue
            baseline.setdefault(uuid, int(used))
            peak[uuid] = max(peak.get(uuid,0), int(used))
    result = {"status": data["status"], "members": data["members"],
              "granted_cards": len(data["cards"]), "slots":data["slots"],
              "forecast_wall_seconds":data.get("forecast_wall_seconds"),
              "total_wall_seconds":data.get("total_wall_seconds"),
              "export_and_spread_wall_seconds":data.get("export_and_spread_wall_seconds"),
              "baseline_mib":baseline,"whole_card_peak_mib":peak,
              "member_wall_seconds":[row["wall_seconds"] for row in data.get("jobs",[])],
              "output_count":len(data.get("outputs",[])),
              "jedi_open_checked":data["jedi_open_checked"],
              "receipt":str((a.run / "ensemble-receipt.json").resolve())}
    growth = a.run / "spread-growth.json"
    if growth.exists():
        result["growth"] = [{"variable":r["variable"],"units":r["units"],
                              "initial_mean_level_spread":sum(r["initial"])/len(r["initial"]),
                              "one_hour_mean_level_spread":sum(r["one_hour"])/len(r["one_hour"]),
                              "levels_with_growth":r["levels_with_growth"], "levels":r["levels"]}
                             for r in json.loads(growth.read_text())["variables"]]
    if a.out:
        write_json(a.out, result)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
