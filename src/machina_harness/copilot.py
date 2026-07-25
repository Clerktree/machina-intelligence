"""Grounded maintenance context for Grok or another reasoning layer."""

from pydantic import BaseModel, Field

from .platform import KnowledgeSearchRequest, KnowledgeSearchResult, store


class MaintenanceBriefRequest(BaseModel):
    asset_id: str
    question: str = Field(min_length=2)
    asset_type: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class MaintenanceBrief(BaseModel):
    asset_id: str
    question: str
    asset: dict | None
    evidence: list[KnowledgeSearchResult]
    model_plugins: list[str]
    instructions: list[str]


def build_maintenance_brief(request: MaintenanceBriefRequest) -> MaintenanceBrief:
    asset = next((item.model_dump() for item in store.list_assets() if item.asset_id == request.asset_id), None)
    evidence = store.search_knowledge(KnowledgeSearchRequest(
        query=request.question, asset_type=request.asset_type, limit=request.limit,
    ))
    return MaintenanceBrief(
        asset_id=request.asset_id,
        question=request.question,
        asset=asset,
        evidence=evidence,
        model_plugins=[model.model_id for model in store.list_models()],
        instructions=[
            "Use only the returned evidence and model outputs as factual support.",
            "Separate observed signals from hypotheses.",
            "Recommend qualified human inspection for safety-critical decisions.",
        ],
    )

