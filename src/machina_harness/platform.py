"""Domain-neutral Machina platform contracts and an in-memory reference store."""

from datetime import datetime, timezone
import os
import sqlite3
import threading
from typing import Literal

from pydantic import BaseModel, Field


class Asset(BaseModel):
    asset_id: str
    name: str
    asset_type: str
    site: str | None = None
    parent_asset_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class TelemetryBatch(BaseModel):
    asset_id: str
    timestamp: datetime
    values: dict[str, float]
    quality: Literal["good", "suspect", "bad"] = "good"
    source: str = "api"


class MaintenanceEvent(BaseModel):
    asset_id: str
    timestamp: datetime
    event_type: Literal["inspection", "repair", "replacement", "lubrication", "failure", "other"]
    description: str
    component: str | None = None
    work_order_id: str | None = None


class ModelDescriptor(BaseModel):
    model_id: str
    version: str
    capability: str
    asset_types: list[str]
    input_schema: list[str]
    output_schema: list[str]
    status: Literal["experimental", "validated", "production"]


class KnowledgeDocument(BaseModel):
    document_id: str
    title: str
    text: str
    asset_type: str | None = None
    source_uri: str | None = None
    revision: str | None = None


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=2)
    asset_type: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchResult(BaseModel):
    document_id: str
    title: str
    excerpt: str
    score: float
    source_uri: str | None = None


class PlatformStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path
        self._connection = sqlite3.connect(db_path, check_same_thread=False) if db_path else None
        self._lock = threading.RLock()
        self.assets: dict[str, Asset] = {}
        self.telemetry: list[TelemetryBatch] = []
        self.events: list[MaintenanceEvent] = []
        self.models: dict[str, ModelDescriptor] = {}
        if self._connection:
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS maintenance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    text TEXT NOT NULL, asset_type TEXT, source_uri TEXT,
                    revision TEXT
                );
            """)
            self._connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(document_id UNINDEXED, title, text, asset_type)"
            )
            self._connection.commit()
            self.assets = {
                asset_id: Asset.model_validate_json(payload)
                for asset_id, payload in self._connection.execute("SELECT asset_id, payload FROM assets")
            }
            self.models = {
                model_id: ModelDescriptor.model_validate_json(payload)
                for model_id, payload in self._connection.execute("SELECT model_id, payload FROM models")
            }

    def register_asset(self, asset: Asset) -> Asset:
        self.assets[asset.asset_id] = asset
        if self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO assets(asset_id, payload) VALUES (?, ?)",
                (asset.asset_id, asset.model_dump_json()),
            )
            self._connection.commit()
        return asset

    def index_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        if not self._connection:
            # Keep a minimal ephemeral knowledge index on the object itself.
            if not hasattr(self, "knowledge_documents"):
                self.knowledge_documents = {}
            self.knowledge_documents[document.document_id] = document
            return document
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO knowledge_documents(document_id, title, text, asset_type, source_uri, revision) VALUES (?, ?, ?, ?, ?, ?)",
                (document.document_id, document.title, document.text, document.asset_type, document.source_uri, document.revision),
            )
            self._connection.execute("DELETE FROM knowledge_fts WHERE document_id = ?", (document.document_id,))
            self._connection.execute(
                "INSERT INTO knowledge_fts(document_id, title, text, asset_type) VALUES (?, ?, ?, ?)",
                (document.document_id, document.title, document.text, document.asset_type or ""),
            )
            self._connection.commit()
        return document

    def search_knowledge(self, request: KnowledgeSearchRequest) -> list[KnowledgeSearchResult]:
        if self._connection:
            query = "SELECT d.document_id, d.title, d.text, d.source_uri, bm25(knowledge_fts) AS score FROM knowledge_fts JOIN knowledge_documents d USING(document_id) WHERE knowledge_fts MATCH ?"
            params: list[str | int] = [request.query]
            if request.asset_type:
                query += " AND (d.asset_type = ? OR d.asset_type IS NULL)"
                params.append(request.asset_type)
            query += " ORDER BY score LIMIT ?"
            params.append(request.limit)
            rows = self._connection.execute(query, params).fetchall()
            return [KnowledgeSearchResult(
                document_id=row[0], title=row[1], excerpt=row[2][:500], score=float(-row[4]), source_uri=row[3],
            ) for row in rows]
        documents = getattr(self, "knowledge_documents", {}).values()
        terms = request.query.lower().split()
        matches = []
        for document in documents:
            if request.asset_type and document.asset_type not in {None, request.asset_type}:
                continue
            haystack = f"{document.title} {document.text}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                matches.append(KnowledgeSearchResult(
                    document_id=document.document_id, title=document.title,
                    excerpt=document.text[:500], score=float(score), source_uri=document.source_uri,
                ))
        return sorted(matches, key=lambda result: result.score, reverse=True)[:request.limit]

    def ingest_telemetry(self, batch: TelemetryBatch) -> TelemetryBatch:
        if batch.asset_id not in self.assets:
            raise KeyError(f"Unknown asset: {batch.asset_id}")
        self.telemetry.append(batch)
        if self._connection:
            self._connection.execute(
                "INSERT INTO telemetry(asset_id, timestamp, payload) VALUES (?, ?, ?)",
                (batch.asset_id, batch.timestamp.isoformat(), batch.model_dump_json()),
            )
            self._connection.commit()
        return batch

    def add_event(self, event: MaintenanceEvent) -> MaintenanceEvent:
        if event.asset_id not in self.assets:
            raise KeyError(f"Unknown asset: {event.asset_id}")
        self.events.append(event)
        if self._connection:
            self._connection.execute(
                "INSERT INTO maintenance_events(asset_id, timestamp, payload) VALUES (?, ?, ?)",
                (event.asset_id, event.timestamp.isoformat(), event.model_dump_json()),
            )
            self._connection.commit()
        return event

    def register_model(self, model: ModelDescriptor) -> ModelDescriptor:
        self.models[model.model_id] = model
        if self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO models(model_id, payload) VALUES (?, ?)",
                (model.model_id, model.model_dump_json()),
            )
            self._connection.commit()
        return model

    def list_assets(self) -> list[Asset]:
        if self._connection:
            self.assets = {
                asset_id: Asset.model_validate_json(payload)
                for asset_id, payload in self._connection.execute("SELECT asset_id, payload FROM assets")
            }
        return list(self.assets.values())

    def list_models(self) -> list[ModelDescriptor]:
        if self._connection:
            self.models = {
                model_id: ModelDescriptor.model_validate_json(payload)
                for model_id, payload in self._connection.execute("SELECT model_id, payload FROM models")
            }
        return list(self.models.values())

    def snapshot(self) -> dict[str, int | str]:
        if self._connection:
            counts = {
                "assets": self._connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0],
                "telemetry_batches": self._connection.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0],
                "maintenance_events": self._connection.execute("SELECT COUNT(*) FROM maintenance_events").fetchone()[0],
                "models": self._connection.execute("SELECT COUNT(*) FROM models").fetchone()[0],
            }
        else:
            counts = {
                "assets": len(self.assets),
                "telemetry_batches": len(self.telemetry),
                "maintenance_events": len(self.events),
                "models": len(self.models),
            }
        return {
            **counts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


store = PlatformStore(os.getenv("MACHINA_DB_PATH"))

# Built-in public baselines. They remain experimental until site-specific
# validation promotes them through the registry.
store.register_model(ModelDescriptor(
    model_id="machina-cwru-bearing-fault",
    version="0.1.0",
    capability="fault_diagnosis",
    asset_types=["bearing", "motor", "rotating_equipment"],
    input_schema=["vibration window"],
    output_schema=["fault class", "confidence"],
    status="experimental",
))
store.register_model(ModelDescriptor(
    model_id="machina-ai4i-quality",
    version="0.1.0",
    capability="quality_prediction",
    asset_types=["machine", "production_line", "manufacturing_equipment"],
    input_schema=["machine type", "temperature", "speed", "torque", "tool wear"],
    output_schema=["failure mode", "failure probability"],
    status="experimental",
))
store.register_model(ModelDescriptor(
    model_id="machina-cmapss-rul",
    version="0.2.1",
    capability="remaining_useful_life",
    asset_types=["engine", "rotating_equipment"],
    input_schema=["cycle", "operating context", "sensor channels"],
    output_schema=["remaining cycles", "uncertainty interval"],
    status="experimental",
))
