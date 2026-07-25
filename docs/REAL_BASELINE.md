# Real baseline release

## Dataset

The first benchmark uses the Case Western Reserve University Bearing Fault
Dataset. The project source describes normal, inner-race, outer-race, and ball
fault recordings with drive-end and fan-end vibration signals at 12 kHz and
48 kHz. The raw data is intentionally kept outside this repository.

Source: <https://csegroups.case.edu/bearingdatacenter/pages/download-data-file>

## Method

- 161 source MATLAB files
- 1,288 vibration windows
- 4,096-sample windows
- RMS, peak, kurtosis, and crest-factor features
- Random Forest with 300 trees
- Grouped file-level split with all classes represented in train and test

## Results

| Metric | Value |
|---|---:|
| Accuracy | 0.6555 |
| Macro-F1 | 0.5870 |
| Normal recall | 0.2500 |
| Ball recall | 0.7250 |
| Inner-race recall | 0.6500 |
| Outer-race recall | 0.6438 |

The normal recall is inadequate for production. The next iteration should use
operating-condition normalization, balanced machine-level sampling, and a
raw-signal 1D CNN or time-series encoder. These metrics are a transparent
starting point, not a performance claim for unseen industrial equipment.

## Enhanced signal baseline

The next reproducible experiment uses 64 12 kHz drive-end files, 1,024
source-grouped windows, and 21 time-domain, spectral, and envelope features.
The selected sigmoid-calibrated ExtraTrees model is published locally as
`artifacts/cwru-enhanced` and uses a leave-one-RPM-out robustness check.

| Metric | Value |
|---|---:|
| Grouped split macro-F1 | 0.9963 |
| Leave-one-RPM-out mean macro-F1 | 0.9915 |
| Leave-one-RPM-out minimum macro-F1 | 0.9844 |

The calibrated model is selected despite the small benchmark-score tradeoff so
that fault confidence can support conservative abstention behavior.

These unusually strong benchmark numbers are not industrial generalization.
CWRU is a controlled laboratory dataset; the next acceptance test is a held-out
machine or site dataset with real operating variation.
