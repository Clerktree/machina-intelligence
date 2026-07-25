"""Optional trained fault classifier loaded through an explicit environment path."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np


def _features(signal: list[float]) -> list[float]:
    values = np.asarray(signal, dtype=float)
    values = values[np.isfinite(values)]
    centered = values - values.mean()
    rms = float(np.sqrt(np.mean(centered ** 2)))
    peak = float(np.max(np.abs(centered)))
    variance = float(np.mean(centered ** 2))
    kurtosis = float(np.mean(centered ** 4) / (variance ** 2 + 1e-12))
    crest = peak / (rms + 1e-12)
    return [rms, peak, kurtosis, crest]


@lru_cache(maxsize=1)
def _load():
    path = os.getenv("MACHINA_CLASSIFIER_PATH")
    if not path:
        return None
    try:
        import joblib
        return joblib.load(Path(path))
    except (FileNotFoundError, ImportError, ValueError):
        return None


def classify(signal: list[float]) -> tuple[str, float, str] | None:
    model = _load()
    if model is None or len(signal) < 32:
        return None
    probabilities = model.predict_proba([_features(signal)])[0]
    index = int(np.argmax(probabilities))
    return str(model.classes_[index]), float(probabilities[index]), "machina-cwru-rf-0.1.0"

