"""Small, explainable baseline. Replace this scorer with trained weights later."""

from math import isfinite, sqrt

from .schemas import Finding, SensorWindow
from .classifier import classify

MODEL_VERSION = "machina-baseline-0.1.0"


def _robust_score(values: list[float]) -> float:
    clean = [float(v) for v in values if isfinite(float(v))]
    if len(clean) < 3:
        return 0.0
    ordered = sorted(clean)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    deviations = sorted(abs(value - median) for value in clean)
    mad_middle = len(deviations) // 2
    mad = deviations[mad_middle] if len(deviations) % 2 else (deviations[mad_middle - 1] + deviations[mad_middle]) / 2
    # A zero MAD is common for quantized or idle sensors. Any distinct tail
    # value is then a meaningful anomaly rather than a divide-by-zero case.
    if mad == 0:
        return 1.0 if clean[-1] != median else 0.0
    tail = abs(clean[-1] - median) / (1.4826 * mad)
    return min(1.0, max(0.0, (tail - 1.5) / 4.5))


def analyze_window(window: SensorWindow) -> Finding:
    scores = {name: _robust_score(values) for name, values in window.sensors.items()}
    ordered = sorted(scores, key=scores.get, reverse=True)
    score = max(scores.values(), default=0.0)
    status = "critical" if score >= 0.75 else "watch" if score >= 0.35 else "normal"
    recommendation = {
        "normal": "Continue monitoring; no immediate maintenance action indicated.",
        "watch": "Inspect the leading sensor trend and compare with the last maintenance record.",
        "critical": "Stop relying on automated diagnosis; perform a qualified inspection before continued operation.",
    }[status]
    vibration = window.sensors.get("vibration") or window.sensors.get("vibration_de")
    classification = classify(vibration) if vibration else None
    return Finding(
        machine_id=window.machine_id,
        status=status,
        anomaly_score=round(score, 4),
        contributing_sensors=[name for name in ordered if scores[name] > 0][:3],
        recommendation=recommendation,
        model_version=classification[2] if classification else MODEL_VERSION,
        predicted_fault=classification[0] if classification else None,
        fault_confidence=round(classification[1], 4) if classification else None,
    )
