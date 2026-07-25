"""Optional process-failure/quality model plugin."""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .model_runtime import model_path
from pydantic import BaseModel, Field


class QualityRequest(BaseModel):
    asset_id: str
    machine_type: str = "M"
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float


class QualityPrediction(BaseModel):
    asset_id: str
    predicted_failure_mode: str
    failure_probability: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_version: str
    warning: str


@lru_cache(maxsize=1)
def _load():
    path = model_path("quality_prediction")
    if not path:
        return None
    artifact_path = Path(path)
    try:
        import joblib
        return joblib.load(artifact_path), json.loads((artifact_path.parent / "metadata.json").read_text())
    except (FileNotFoundError, ImportError, ValueError, json.JSONDecodeError):
        return None


def predict(request: QualityRequest) -> QualityPrediction | None:
    loaded = _load()
    if loaded is None:
        return None
    model, metadata = loaded
    type_code = {"L": 0, "M": 1, "H": 2}.get(request.machine_type.upper(), 1)
    values = [[type_code, request.air_temperature_k, request.process_temperature_k,
               request.rotational_speed_rpm, request.torque_nm, request.tool_wear_min]]
    probabilities = model.predict_proba(values)[0]
    classes = [str(value) for value in model.classes_]
    index = int(np.argmax(probabilities))
    normal_probability = float(probabilities[classes.index("normal")]) if "normal" in classes else 0.0
    return QualityPrediction(
        asset_id=request.asset_id,
        predicted_failure_mode=classes[index],
        failure_probability=round(1.0 - normal_probability, 4),
        probabilities={label: round(float(probability), 4) for label, probability in zip(classes, probabilities)},
        model_version=metadata["model_version"],
        warning="The AI4I benchmark is synthetic; validate on real process quality data before deployment.",
    )
