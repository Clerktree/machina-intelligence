"""Optional C-MAPSS RUL plugin loaded by an explicit model path."""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .schemas import RULPrediction, RULRequest


@lru_cache(maxsize=1)
def _load():
    path = os.getenv("MACHINA_RUL_MODEL_PATH")
    if not path:
        return None
    model_path = Path(path)
    try:
        import joblib
        model = joblib.load(model_path)
        metadata = json.loads((model_path.parent / "metadata.json").read_text())
        return model, metadata
    except (FileNotFoundError, ImportError, ValueError, json.JSONDecodeError):
        return None


def predict(request: RULRequest) -> RULPrediction | None:
    loaded = _load()
    if loaded is None:
        return None
    model, metadata = loaded
    values = {
        "cycle": request.cycle,
        "cycle_fraction": request.cycle / request.max_observed_cycle,
        **request.sensors,
    }
    features = metadata["features"]
    if any(name not in values for name in features):
        return None
    prediction = max(0.0, float(model.predict([[values[name] for name in features]])[0]))
    # RF dispersion is a practical, transparent uncertainty proxy for this baseline.
    trees = np.asarray([estimator.predict([[values[name] for name in features]])[0] for estimator in model.estimators_])
    spread = max(1.0, float(np.std(trees) * 1.96))
    return RULPrediction(
        asset_id=request.asset_id,
        predicted_cycles=round(prediction, 2),
        lower_cycles=round(max(0.0, prediction - spread), 2),
        upper_cycles=round(prediction + spread, 2),
        model_version=metadata["model_version"],
        warning="Research baseline; validate on the target asset and operating regime.",
    )

