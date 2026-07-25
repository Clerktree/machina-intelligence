import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SensorWindow(BaseModel):
    machine_id: str = Field(min_length=1)
    sensors: dict[str, list[float]]
    sample_rate_hz: float = Field(gt=0)
    asset_type: str = "rotating_equipment"

    @model_validator(mode="after")
    def validate_sensor_window(self) -> "SensorWindow":
        if not self.sensors:
            raise ValueError("at least one sensor series is required")
        for name, values in self.sensors.items():
            if not name.strip():
                raise ValueError("sensor names cannot be empty")
            if len(values) < 3:
                raise ValueError(f"sensor '{name}' needs at least 3 samples")
            if len(values) > 1_000_000:
                raise ValueError(f"sensor '{name}' exceeds the 1,000,000 sample limit")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"sensor '{name}' contains a non-finite value")
        return self


class Finding(BaseModel):
    machine_id: str
    status: Literal["normal", "watch", "critical"]
    anomaly_score: float = Field(ge=0, le=1)
    contributing_sensors: list[str]
    recommendation: str
    model_version: str
    predicted_fault: str | None = None
    fault_confidence: float | None = Field(default=None, ge=0, le=1)
    abstained: bool = False
    data_quality: Literal["valid", "insufficient"] = "valid"
    human_review_required: bool = True


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
