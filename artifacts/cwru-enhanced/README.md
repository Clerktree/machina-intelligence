---
license: apache-2.0
tags:
- industrial-ai
- predictive-maintenance
- bearing-fault-diagnosis
- vibration-analysis
pipeline_tag: tabular-classification
---

# Machina CWRU Enhanced Bearing Classifier

This experimental plugin uses 21 time-domain, spectral, and envelope features
from 4,096-sample vibration windows and an ExtraTrees classifier. Training used
64 12 kHz drive-end files from the public Case Western Reserve University
bearing benchmark, with source-file grouping and leave-one-RPM-out checks.

The benchmark is useful for reproducibility, but it is not evidence of
generalization to customer machinery. The reported score is a research
benchmark only; validate on target assets and operating regimes before use.
