"""The preset instrument resolves executing package bytes without a checkout."""
from hexcore.drivers import forecast_receipt
from hexcore.shipped_sources import resolve


def test_snapshot_hashes_installed_sources():
    snapshot = forecast_receipt.source_snapshot()
    assert snapshot
    assert all("absent" not in record for record in snapshot.values())
    name = "src/hexcore/drivers/forecast_receipt.py"
    assert snapshot[name] == forecast_receipt.file_record(resolve(name))


def test_serial_policy_resolves_beside_installed_driver():
    policy = forecast_receipt._load_serial_io_policy()
    assert callable(policy)
