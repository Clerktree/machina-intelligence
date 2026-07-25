"""Explainable energy-efficiency analytics baseline."""

from statistics import median

from pydantic import BaseModel, Field


class EnergySample(BaseModel):
    power_kw: float = Field(ge=0)
    output_rate: float = Field(gt=0)


class EnergyRequest(BaseModel):
    asset_id: str
    history: list[EnergySample] = Field(min_length=3)
    current: EnergySample


class EnergyFinding(BaseModel):
    asset_id: str
    current_energy_per_output: float
    baseline_energy_per_output: float
    efficiency_anomaly_score: float = Field(ge=0, le=1)
    status: str
    recommendation: str
    model_version: str = "machina-energy-analytics-0.1.0"


def analyze_energy(request: EnergyRequest) -> EnergyFinding:
    history_ratios = [sample.power_kw / sample.output_rate for sample in request.history]
    baseline = median(history_ratios)
    current = request.current.power_kw / request.current.output_rate
    ratio = current / (baseline + 1e-12)
    score = min(1.0, max(0.0, (ratio - 1.05) / 0.95))
    status = "critical" if score >= 0.55 else "watch" if score >= 0.15 else "normal"
    recommendation = {
        "normal": "Continue monitoring energy per unit output.",
        "watch": "Inspect operating conditions, idle time, and recent maintenance for efficiency drift.",
        "critical": "Investigate abnormal energy consumption before treating the asset as healthy.",
    }[status]
    return EnergyFinding(
        asset_id=request.asset_id,
        current_energy_per_output=round(current, 6),
        baseline_energy_per_output=round(baseline, 6),
        efficiency_anomaly_score=round(score, 4),
        status=status,
        recommendation=recommendation,
    )
