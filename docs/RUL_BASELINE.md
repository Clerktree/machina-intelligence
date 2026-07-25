# Remaining-useful-life baseline

Machina's first RUL plugin uses the NASA C-MAPSS FD001 run-to-failure engine
simulation. It trains a Random Forest regressor on engine cycle, normalized
cycle position, and non-constant sensor channels. The target is capped at 125
cycles, a common C-MAPSS baseline convention.

Evaluation is reported twice: an engine-level holdout from the training fleet,
and the official FD001 test fleet with NASA's supplied RUL targets. This keeps
the result auditable and prevents rows from the same engine being randomly
split across folds.

Run it on the lab with:

```bash
python scripts/train_rul.py /tmp/cmapss --subset FD001
```

The result is a baseline plugin, not a universal RUL estimator. It must be
retrained or adapted for the target asset family and operating conditions.

The improved FD001 baseline currently reports 62.36-cycle MAE and 71.72-cycle RMSE on
the official test fleet. This is a starting point for model improvement, not
a production-quality result.
