"""Optional trained fault classifier loaded through an explicit environment path."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .model_runtime import model_path


def _enhanced_features(signal: list[float], sample_rate: float = 12000.0) -> list[float]:
    """Match scripts/train_cwru_enhanced.py for the published model bundle."""
    from scipy.signal import hilbert

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    centered = values - np.mean(values)
    absolute = np.abs(centered)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    peak = float(np.max(absolute))
    mean_abs = float(np.mean(absolute))
    variance = float(np.mean(centered ** 2))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / sample_rate)
    spectrum[0] = 0.0
    power = spectrum ** 2
    total_power = float(np.sum(power) + 1e-12)
    normalized_power = power / total_power
    edges = np.asarray([0, 500, 1000, 2000, 3000, 4000, 5000, sample_rate / 2.0 + 1])
    band_energy = [float(np.sum(power[(frequencies >= low) & (frequencies < high)]) / total_power)
                   for low, high in zip(edges[:-1], edges[1:])]
    envelope = np.abs(hilbert(centered))
    envelope_spectrum = np.abs(np.fft.rfft((envelope - envelope.mean()) * np.hanning(envelope.size)))
    envelope_spectrum[0] = 0.0
    return [
        rms, peak, mean_abs, float(np.mean(centered ** 4) / (variance ** 2 + 1e-12)),
        peak / (rms + 1e-12), peak / (mean_abs + 1e-12), rms / (mean_abs + 1e-12),
        float(np.std(centered)), float(np.min(centered)), float(np.max(centered)),
        float(frequencies[int(np.argmax(spectrum))]),
        float(np.sum(frequencies * power) / total_power),
        float(-np.sum(normalized_power * np.log(normalized_power + 1e-12))),
        float(frequencies[int(np.argmax(envelope_spectrum))]), *band_energy,
    ]


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
    path = model_path("fault_diagnosis")
    if not path:
        return None
    try:
        import joblib
        return joblib.load(path)
    except (FileNotFoundError, ImportError, ValueError):
        return None


def classify(signal: list[float]) -> tuple[str, float, str] | None:
    model = _load()
    if model is None or len(signal) < 32:
        return None
    if isinstance(model, dict) and model.get("feature_version") == "cwru-signal-v2":
        estimator = model["model"]
        row = _enhanced_features(signal)
        version = model.get("model_version", "machina-cwru-calibrated-et-0.3.0")
    else:
        estimator = model
        row = _features(signal)
        version = "machina-cwru-rf-0.1.0"
    probabilities = estimator.predict_proba([row])[0]
    index = int(np.argmax(probabilities))
    return str(estimator.classes_[index]), float(probabilities[index]), version
