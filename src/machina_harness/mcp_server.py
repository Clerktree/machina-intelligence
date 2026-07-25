"""MCP adapter for Grok Build.

Run this from the repository root with the `mcp` extra installed. It uses the
same inference function as the HTTP API, so the agent cannot silently use a
different model path.
"""

from .anomaly import analyze_window
from .capabilities import MACHINA_CAPABILITIES
from .platform import Asset, KnowledgeDocument, KnowledgeSearchRequest, MaintenanceEvent, TelemetryBatch, store
from .rul import predict
from .schemas import RULRequest
from .energy import EnergyRequest, EnergySample, analyze_energy
from .quality import QualityRequest, predict as predict_quality
from .copilot import MaintenanceBriefRequest, build_maintenance_brief
from .schemas import SensorWindow


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("Install the MCP adapter with: pip install -e '.[mcp]'") from exc

    server = FastMCP("Machina")

    @server.tool()
    def analyze_machine_window(
        machine_id: str,
        sensors: dict[str, list[float]],
        sample_rate_hz: float,
        asset_type: str = "rotating_equipment",
    ) -> dict:
        """Analyze a timestamp-ordered sensor window for machine anomalies."""
        result = analyze_window(SensorWindow(
            machine_id=machine_id,
            sensors=sensors,
            sample_rate_hz=sample_rate_hz,
            asset_type=asset_type,
        ))
        return result.model_dump()

    @server.tool()
    def machine_harness_health() -> dict[str, str]:
        """Return the active Machina Harness service and model version."""
        return {"status": "ok", "model_version": "machina-baseline-0.1.0"}

    @server.tool()
    def list_machine_intelligence_capabilities() -> list[dict]:
        """List Machina's available and planned machine-intelligence skills."""
        return [capability.model_dump() for capability in MACHINA_CAPABILITIES]

    @server.tool()
    def register_machine_asset(asset_id: str, name: str, asset_type: str, site: str | None = None) -> dict:
        """Register an asset before telemetry or maintenance events are ingested."""
        return store.register_asset(Asset(
            asset_id=asset_id, name=name, asset_type=asset_type, site=site,
        )).model_dump()

    @server.tool()
    def record_machine_telemetry(asset_id: str, values: dict[str, float], timestamp: str, quality: str = "good") -> dict:
        """Record one timestamped machine telemetry batch."""
        return store.ingest_telemetry(TelemetryBatch(
            asset_id=asset_id, values=values, timestamp=timestamp, quality=quality,
        )).model_dump(mode="json")

    @server.tool()
    def record_maintenance_event(asset_id: str, event_type: str, description: str, timestamp: str) -> dict:
        """Record an inspection, repair, replacement, lubrication, or failure event."""
        return store.add_event(MaintenanceEvent(
            asset_id=asset_id, event_type=event_type, description=description, timestamp=timestamp,
        )).model_dump(mode="json")

    @server.tool()
    def machina_platform_snapshot() -> dict:
        """Return counts of registered assets, telemetry, events, and models."""
        return store.snapshot()

    @server.tool()
    def list_machine_assets() -> list[dict]:
        """List assets currently known to Machina."""
        return [asset.model_dump() for asset in store.list_assets()]

    @server.tool()
    def list_registered_models() -> list[dict]:
        """List model plugins registered with Machina."""
        return [model.model_dump() for model in store.list_models()]

    @server.tool()
    def index_maintenance_document(document_id: str, title: str, text: str, asset_type: str | None = None, source_uri: str | None = None) -> dict:
        """Index a manual, SOP, maintenance note, or other engineering document."""
        return store.index_document(KnowledgeDocument(
            document_id=document_id, title=title, text=text, asset_type=asset_type, source_uri=source_uri,
        )).model_dump()

    @server.tool()
    def search_machine_knowledge(query: str, asset_type: str | None = None, limit: int = 5) -> list[dict]:
        """Search indexed machine manuals and maintenance knowledge."""
        return [result.model_dump() for result in store.search_knowledge(KnowledgeSearchRequest(
            query=query, asset_type=asset_type, limit=limit,
        ))]

    @server.tool()
    def estimate_remaining_useful_life(asset_id: str, cycle: float, max_observed_cycle: float, sensors: dict[str, float]) -> dict:
        """Estimate remaining cycles with the configured RUL plugin."""
        result = predict(RULRequest(
            asset_id=asset_id, cycle=cycle, max_observed_cycle=max_observed_cycle, sensors=sensors,
        ))
        if result is None:
            return {"status": "unavailable", "reason": "RUL model is not configured or required sensors are missing"}
        return result.model_dump()

    @server.tool()
    def analyze_machine_energy(asset_id: str, history: list[dict[str, float]], current: dict[str, float]) -> dict:
        """Analyze energy per unit output against the asset's recent baseline."""
        result = analyze_energy(EnergyRequest(
            asset_id=asset_id,
            history=[EnergySample(**sample) for sample in history],
            current=EnergySample(**current),
        ))
        return result.model_dump()

    @server.tool()
    def predict_process_quality(asset_id: str, machine_type: str, air_temperature_k: float, process_temperature_k: float, rotational_speed_rpm: float, torque_nm: float, tool_wear_min: float) -> dict:
        """Predict a process failure mode from manufacturing operating conditions."""
        result = predict_quality(QualityRequest(
            asset_id=asset_id, machine_type=machine_type,
            air_temperature_k=air_temperature_k, process_temperature_k=process_temperature_k,
            rotational_speed_rpm=rotational_speed_rpm, torque_nm=torque_nm, tool_wear_min=tool_wear_min,
        ))
        if result is None:
            return {"status": "unavailable", "reason": "Quality model is not configured"}
        return result.model_dump()

    @server.tool()
    def prepare_maintenance_brief(asset_id: str, question: str, asset_type: str | None = None, limit: int = 5) -> dict:
        """Prepare grounded machine evidence for a maintenance answer."""
        return build_maintenance_brief(MaintenanceBriefRequest(
            asset_id=asset_id, question=question, asset_type=asset_type, limit=limit,
        )).model_dump()

    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
