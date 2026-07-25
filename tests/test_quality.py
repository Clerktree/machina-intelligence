import json

import joblib
import numpy as np

from machina_harness.quality import _load, predict
from machina_harness.quality import QualityRequest


class FakeQualityModel:
    classes_ = np.array(["normal", "power"])

    def predict_proba(self, values):
        return np.array([[0.96, 0.04]])


def test_quality_failure_probability_is_not_predicted_class_probability(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    joblib.dump(FakeQualityModel(), model_path)
    (tmp_path / "metadata.json").write_text(json.dumps({"model_version": "test-quality"}))
    monkeypatch.setenv("MACHINA_QUALITY_MODEL_PATH", str(model_path))
    _load.cache_clear()
    result = predict(QualityRequest(
        asset_id="m1", machine_type="M", air_temperature_k=300,
        process_temperature_k=310, rotational_speed_rpm=1500,
        torque_nm=50, tool_wear_min=100,
    ))
    assert result.failure_probability == 0.04
    assert result.predicted_failure_mode == "normal"
    _load.cache_clear()

