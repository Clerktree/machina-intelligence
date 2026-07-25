from datetime import datetime, timezone

from machina_harness.platform import Asset, PlatformStore, TelemetryBatch


def test_sqlite_store_persists_platform_counts(tmp_path):
    db_path = tmp_path / "machina.db"
    store = PlatformStore(str(db_path))
    store.register_asset(Asset(asset_id="m1", name="Motor", asset_type="motor"))
    store.ingest_telemetry(TelemetryBatch(
        asset_id="m1", timestamp=datetime.now(timezone.utc), values={"rpm": 1200},
    ))
    assert store.snapshot()["assets"] == 1
    reopened = PlatformStore(str(db_path))
    assert reopened.snapshot()["assets"] == 1
    assert reopened.snapshot()["telemetry_batches"] == 1
    reopened.ingest_telemetry(TelemetryBatch(
        asset_id="m1", timestamp=datetime.now(timezone.utc), values={"rpm": 1210},
    ))
    assert reopened.snapshot()["telemetry_batches"] == 2
