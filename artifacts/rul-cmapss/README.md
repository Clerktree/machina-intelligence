---
license: apache-2.0
tags:
- industrial-ai
- predictive-maintenance
- remaining-useful-life
- time-series
pipeline_tag: tabular-regression
---

# Machina C-MAPSS RUL Baseline

ExtraTrees remaining-useful-life estimator trained on NASA C-MAPSS FD001.
The model expects the feature order recorded in `metadata.json`, including
engine cycle, normalized cycle fraction, and non-constant sensor channels.

This is a research baseline. Official FD001 test MAE is recorded in
`metadata.json`; validate on the target equipment before using it for
maintenance decisions.
