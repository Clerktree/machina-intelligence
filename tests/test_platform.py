from datetime import datetime, timezone

import pytest

from machina_harness.platform import Asset, KnowledgeDocument, KnowledgeSearchRequest, PlatformStore, TelemetryBatch


def test_platform_requires_registered_assets():
    store = PlatformStore()
    with pytest.raises(KeyError):
        store.ingest_telemetry(TelemetryBatch(
            asset_id="missing",
            timestamp=datetime.now(timezone.utc),
            values={"temperature_c": 60},
        ))


def test_platform_tracks_asset_and_telemetry():
    store = PlatformStore()
    store.register_asset(Asset(asset_id="m1", name="Motor 1", asset_type="motor"))
    store.ingest_telemetry(TelemetryBatch(
        asset_id="m1",
        timestamp=datetime.now(timezone.utc),
        values={"temperature_c": 60},
    ))
    assert store.snapshot()["assets"] == 1
    assert store.snapshot()["telemetry_batches"] == 1


def test_knowledge_search_returns_relevant_manual():
    store = PlatformStore()
    store.index_document(KnowledgeDocument(
        document_id="manual-1", title="Pump lubrication guide",
        text="Inspect the bearing and apply approved grease every 1000 hours.",
        asset_type="pump",
    ))
    results = store.search_knowledge(KnowledgeSearchRequest(query="bearing grease", asset_type="pump"))
    assert results[0].document_id == "manual-1"
