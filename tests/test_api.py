from fastapi.testclient import TestClient

from machina_harness.api import app


def test_capabilities_and_platform_model_registry():
    client = TestClient(app)
    capabilities = client.get("/v1/capabilities")
    assert capabilities.status_code == 200
    assert {item["id"] for item in capabilities.json()} >= {"fault_diagnosis", "remaining_useful_life"}
    snapshot = client.get("/v1/platform/snapshot").json()
    assert snapshot["models"] >= 2
    assert len(client.get("/v1/models").json()) >= 2


def test_model_health_reports_bundled_plugins():
    client = TestClient(app)
    health = {item["capability"]: item for item in client.get("/v1/model-health").json()}
    assert health["fault_diagnosis"]["available"] is True
    assert health["remaining_useful_life"]["available"] is True
    assert health["quality_prediction"]["available"] is True
    assert health["fault_diagnosis"]["source"] == "bundled"
    assert len(health["fault_diagnosis"]["sha256"]) == 64


def test_readiness_and_request_trace():
    client = TestClient(app)
    response = client.get("/ready", headers={"X-Request-ID": "pilot-check-001"})
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.headers["X-Request-ID"] == "pilot-check-001"


def test_inference_is_audited_without_storing_sensor_values():
    client = TestClient(app)
    response = client.post("/v1/analyze", headers={"X-Request-ID": "audit-check-001"}, json={
        "machine_id": "audit-pump",
        "sample_rate_hz": 1000,
        "sensors": {"vibration": [1, 1, 1, 1, 1, 12]},
    })
    assert response.status_code == 200
    audits = client.get("/v1/audit/inferences?limit=1").json()
    assert audits[0]["request_id"] == "audit-check-001"
    assert audits[0]["sensor_summary"] == {"vibration": 6}
    assert "values" not in audits[0]


def test_sensor_window_rejects_non_finite_values():
    client = TestClient(app)
    response = client.post("/v1/analyze", json={
        "machine_id": "bad-input",
        "sample_rate_hz": 1000,
        "sensors": {"vibration": [1, 2, "NaN"]},
    })
    assert response.status_code == 422


def test_asset_telemetry_and_event_flow():
    client = TestClient(app)
    asset = client.post("/v1/assets", json={
        "asset_id": "test-pump-1", "name": "Test pump", "asset_type": "pump",
    })
    assert asset.status_code == 200
    telemetry = client.post("/v1/telemetry", json={
        "asset_id": "test-pump-1", "timestamp": "2026-07-25T08:00:00Z",
        "values": {"temperature_c": 62.1},
    })
    assert telemetry.status_code == 200
    event = client.post("/v1/maintenance-events", json={
        "asset_id": "test-pump-1", "timestamp": "2026-07-25T08:01:00Z",
        "event_type": "inspection", "description": "Baseline inspection",
    })
    assert event.status_code == 200
    assert client.get("/v1/assets").json()[0]["asset_id"] == "test-pump-1"


def test_knowledge_index_and_search():
    client = TestClient(app)
    indexed = client.post("/v1/knowledge/documents", json={
        "document_id": "sop-1", "title": "Motor inspection SOP",
        "text": "Inspect vibration and temperature before restarting the motor.",
        "asset_type": "motor",
    })
    assert indexed.status_code == 200
    results = client.post("/v1/knowledge/search", json={
        "query": "vibration temperature", "asset_type": "motor",
    })
    assert results.status_code == 200
    assert results.json()[0]["document_id"] == "sop-1"


def test_maintenance_brief_is_grounded():
    client = TestClient(app)
    client.post("/v1/assets", json={
        "asset_id": "brief-motor", "name": "Brief motor", "asset_type": "motor",
    })
    client.post("/v1/knowledge/documents", json={
        "document_id": "brief-sop", "title": "Motor restart SOP",
        "text": "Inspect vibration before restarting the motor.", "asset_type": "motor",
    })
    response = client.post("/v1/copilot/maintenance-brief", json={
        "asset_id": "brief-motor", "question": "vibration restart", "asset_type": "motor",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["asset"]["asset_id"] == "brief-motor"
    assert body["evidence"][0]["document_id"] == "brief-sop"
    assert "machina-cwru-bearing-fault" in body["model_plugins"]
