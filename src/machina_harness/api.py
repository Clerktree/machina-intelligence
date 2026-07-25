import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .anomaly import analyze_window
from .capabilities import MACHINA_CAPABILITIES, Capability
from .platform import (
    Asset, KnowledgeDocument, KnowledgeSearchRequest, KnowledgeSearchResult,
    MaintenanceEvent, ModelDescriptor, TelemetryBatch, store,
)
from .rul import predict
from .schemas import RULPrediction, RULRequest
from .energy import EnergyFinding, EnergyRequest, analyze_energy
from .quality import QualityPrediction, QualityRequest, predict as predict_quality
from .copilot import MaintenanceBrief, MaintenanceBriefRequest, build_maintenance_brief
from .schemas import Finding, SensorWindow

app = FastAPI(title="Machina Harness", version="0.1.0")


@app.middleware("http")
async def optional_api_key(request: Request, call_next):
    """Require X-Machina-API-Key whenever MACHINA_API_KEY is configured."""
    configured_key = os.getenv("MACHINA_API_KEY")
    if configured_key and request.url.path != "/health":
        if request.headers.get("x-machina-api-key") != configured_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing Machina API key"})
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "machina-harness"}


@app.get("/v1/capabilities", response_model=list[Capability])
def capabilities() -> list[Capability]:
    """Return the platform capability registry for dashboards and agents."""
    return MACHINA_CAPABILITIES


@app.get("/v1/platform/snapshot")
def platform_snapshot() -> dict:
    return store.snapshot()


@app.post("/v1/assets", response_model=Asset)
def register_asset(asset: Asset) -> Asset:
    return store.register_asset(asset)


@app.get("/v1/assets", response_model=list[Asset])
def list_assets() -> list[Asset]:
    return store.list_assets()


@app.post("/v1/telemetry", response_model=TelemetryBatch)
def ingest_telemetry(batch: TelemetryBatch) -> TelemetryBatch:
    return store.ingest_telemetry(batch)


@app.post("/v1/maintenance-events", response_model=MaintenanceEvent)
def add_maintenance_event(event: MaintenanceEvent) -> MaintenanceEvent:
    return store.add_event(event)


@app.post("/v1/models", response_model=ModelDescriptor)
def register_model(model: ModelDescriptor) -> ModelDescriptor:
    return store.register_model(model)


@app.get("/v1/models", response_model=list[ModelDescriptor])
def list_models() -> list[ModelDescriptor]:
    return store.list_models()


@app.post("/v1/knowledge/documents", response_model=KnowledgeDocument)
def index_knowledge_document(document: KnowledgeDocument) -> KnowledgeDocument:
    return store.index_document(document)


@app.post("/v1/knowledge/search", response_model=list[KnowledgeSearchResult])
def search_knowledge(request: KnowledgeSearchRequest) -> list[KnowledgeSearchResult]:
    return store.search_knowledge(request)


@app.post("/v1/rul", response_model=RULPrediction)
def estimate_rul(request: RULRequest) -> RULPrediction:
    result = predict(request)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="RUL model is not configured or required sensors are missing")
    return result


@app.post("/v1/energy/analyze", response_model=EnergyFinding)
def analyze_energy_endpoint(request: EnergyRequest) -> EnergyFinding:
    return analyze_energy(request)


@app.post("/v1/quality/predict", response_model=QualityPrediction)
def predict_quality_endpoint(request: QualityRequest) -> QualityPrediction:
    result = predict_quality(request)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Quality model is not configured")
    return result


@app.post("/v1/copilot/maintenance-brief", response_model=MaintenanceBrief)
def maintenance_brief(request: MaintenanceBriefRequest) -> MaintenanceBrief:
    return build_maintenance_brief(request)


@app.post("/v1/analyze", response_model=Finding)
def analyze(window: SensorWindow) -> Finding:
    return analyze_window(window)
