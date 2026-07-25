"""Resolve bundled model artifacts and report their runtime availability."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_CONFIG = {
    "fault_diagnosis": ("MACHINA_CLASSIFIER_PATH", "artifacts/cwru-enhanced/model.joblib", "machina-cwru-enhanced-et-0.2.0"),
    "remaining_useful_life": ("MACHINA_RUL_MODEL_PATH", "artifacts/rul-cmapss/model.joblib", "machina-cmapss-rul-et-0.2.1"),
    "quality_prediction": ("MACHINA_QUALITY_MODEL_PATH", "artifacts/ai4i-quality/model.joblib", "machina-ai4i-quality-rf-0.1.0"),
}


def model_path(capability: str) -> Path | None:
    """Return an explicitly configured path or a bundled artifact path."""
    env_name, bundled_path, _ = MODEL_CONFIG[capability]
    configured = os.getenv(env_name)
    path = Path(configured) if configured else PROJECT_ROOT / bundled_path
    return path if path.is_file() else None


def model_health() -> list[dict[str, str | bool]]:
    """Expose safe, path-free model availability for dashboards and operators."""
    result = []
    for capability, (env_name, _, version) in MODEL_CONFIG.items():
        path = model_path(capability)
        result.append({
            "capability": capability,
            "model_version": version,
            "available": path is not None,
            "source": "configured" if os.getenv(env_name) else "bundled",
        })
    return result
