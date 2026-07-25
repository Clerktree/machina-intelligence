from machina_harness.energy import EnergyRequest, EnergySample, analyze_energy


def test_energy_drift_is_flagged():
    result = analyze_energy(EnergyRequest(
        asset_id="compressor-1",
        history=[EnergySample(power_kw=100, output_rate=10) for _ in range(4)],
        current=EnergySample(power_kw=160, output_rate=10),
    ))
    assert result.status == "critical"
    assert result.efficiency_anomaly_score > 0.55


def test_energy_baseline_is_normal():
    result = analyze_energy(EnergyRequest(
        asset_id="compressor-1",
        history=[EnergySample(power_kw=100, output_rate=10) for _ in range(4)],
        current=EnergySample(power_kw=102, output_rate=10),
    ))
    assert result.status == "normal"
