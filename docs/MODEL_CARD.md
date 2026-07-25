---
language:
- en
license: apache-2.0
tags:
- industrial-ai
- predictive-maintenance
- anomaly-detection
- vibration-analysis
- time-series
pipeline_tag: tabular-classification
---

# Machina Fault Detector

## Summary

This model is intended to classify faults in rotating equipment from labelled
sensor-window features. This initial release is a baseline for reproducible
experiments, not a certified safety system.

## Intended use

Use it to prioritize inspection and maintenance review. Do not use it as the
sole control signal for machinery shutdown, personnel safety, or regulatory
compliance.

## Training and evaluation

Version `machina-cwru-rf-0.1.0` was trained from 161 CWRU MATLAB files using
RMS, peak, kurtosis, and crest-factor features from vibration windows. The
evaluation split was grouped by source file and included all four classes in
both folds. Accuracy was 0.6555 and macro-F1 was 0.5870. Normal-bearing recall
was 0.25, so this checkpoint is a research baseline and must not be used as a
sole maintenance or safety decision system.

Never report random-row validation as the primary result.

## Limitations

Performance depends on sensor placement, sampling rate, machine type, load,
and operating environment. Site-specific validation is required.
