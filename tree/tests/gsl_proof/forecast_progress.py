"""Retain real model progress after verify-mpas retires compared outputs.

Only orchestration receipts are decoded here; weather fields stay in Rust.
The actual forecast CLI prints its mpas-native-forecast-v1 receipt to hex.log.
"""
import json
from pathlib import Path


def retain_forecast_progress(forecast, destination):
    forecast = Path(forecast)
    records = []
    model_log = forecast / 'hex.log'
    if model_log.is_file():
        for line in model_log.read_text().splitlines():
            if not line.startswith('{'):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get('schema') == 'mpas-native-forecast-v1':
                records.append(record)
    disk_receipt = forecast / 'hex-output/native-physics-receipt.json'
    if disk_receipt.is_file():
        record = json.loads(disk_receipt.read_text())
        assert record.get('schema') == 'mpas-native-forecast-v1', 'unexpected model receipt schema'
        records.append(record)
    assert records, 'no actual forecast progress receipt in model log or output'
    assert all(record == records[0] for record in records), 'conflicting actual forecast receipts'
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records[0], indent=2) + '\n')
    return records[0]
