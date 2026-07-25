---
license: apache-2.0
tags:
- industrial-ai
- predictive-maintenance
- bearing-fault-diagnosis
- vibration-analysis
- tabular-classification
pipeline_tag: tabular-classification
---

# Machina CWRU Bearing Fault Classifier

This is the first public baseline for Machina Harness. It is a scikit-learn
Random Forest trained from vibration windows derived from the Case Western
Reserve University bearing-fault dataset.

## Classes

`normal`, `ball`, `inner_race`, and `outer_race`.

## Features

The model expects four features in this order:

1. RMS
2. Peak amplitude
3. Kurtosis
4. Crest factor

See `metadata.json` for the grouped evaluation report. The reported test
macro-F1 is 0.5870 and accuracy is 0.6555. This is a research baseline, not a
certified safety or maintenance-control model. Validate against the target
machine, sensor placement, sampling rate, and operating conditions.

Raw CWRU data is not redistributed in this repository. Users must obtain it
under the dataset's applicable terms.

## Local inference

```python
import joblib
import numpy as np

model = joblib.load("model.joblib")
features = np.asarray([[rms, peak, kurtosis, crest_factor]])
print(model.predict(features))
print(model.predict_proba(features))
```

