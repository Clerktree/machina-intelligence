from machina_harness.anomaly import analyze_window
from machina_harness.schemas import SensorWindow


def test_normal_window_is_explainable():
    result = analyze_window(SensorWindow(
        machine_id="m1",
        sample_rate_hz=10,
        sensors={"vibration": [1, 1.01, 0.99, 1.0, 1.02]},
    ))
    assert result.status == "normal"
    assert result.model_version.startswith("machina-baseline-")


def test_spike_is_flagged():
    result = analyze_window(SensorWindow(
        machine_id="m2",
        sample_rate_hz=10,
        sensors={"vibration": [1, 1, 1, 1, 10]},
    ))
    assert result.status in {"watch", "critical"}
    assert "vibration" in result.contributing_sensors

