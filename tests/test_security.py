from fastapi.testclient import TestClient

from machina_harness.api import app


def test_health_is_public_and_api_key_is_optional(monkeypatch):
    monkeypatch.setenv("MACHINA_API_KEY", "test-secret")
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/capabilities").status_code == 401
    assert client.get("/v1/capabilities", headers={"X-Machina-API-Key": "test-secret"}).status_code == 200
    monkeypatch.delenv("MACHINA_API_KEY")

