from typing import Literal

from pydantic import BaseModel, Field


class SensorWindow(BaseModel):
    machine_id: str = Field(min_length=1)
    sensors: dict[str, list[float]]
    sample_rate_hz: float = Field(gt=0)
    asset_type: str = "rotating_equipment"


class Finding(BaseModel):
    machine_id: str
    status: Literal["normal", "watch", "critical"]
    anomaly_score: float = Field(ge=0, le=1)
    contributing_sensors: list[str]
    recommendation: str
    model_version: str
    predicted_fault: str | None = None
    fault_confidence: float | None = Field(default=None, ge=0, le=1)


class RULRequest(BaseModel):
    asset_id: str
    cycle: float = Field(ge=0)
    max_observed_cycle: float = Field(gt=0)
    sensors: dict[str, float]


class RULPrediction(BaseModel):
    asset_id: str
    predicted_cycles: float = Field(ge=0)
    lower_cycles: float = Field(ge=0)
    upper_cycles: float = Field(ge=0)
    model_version: str
    warning: str
