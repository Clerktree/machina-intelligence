# Training contract

The first real model should be evaluated by machine, not by random rows. A
random row split leaks the operating signature of the same machine into both
sets and produces misleading results.

## Data contract

Create one row per labelled sensor window. Include:

- `machine_id`: stable asset identifier, used for group splits
- `timestamp`: UTC timestamp for traceability
- `label`: `normal`, `imbalance`, `misalignment`, `bearing_fault`, etc.
- numeric features derived from the window: RMS, peak, crest factor,
  kurtosis, temperature, load, RPM, and frequency-band energy

Keep raw data private if it contains customer or site information. Publish a
sanitized sample dataset and a dataset card alongside the model.

## Lab run

The RTX 4500 Ada is more than sufficient for the initial Random Forest and
small 1D CNN/Transformer experiments. Before downloading public datasets or
model weights, free several gigabytes on the lab home filesystem; it currently
has less than 3 GB available.

```bash
python scripts/train_baseline.py data/windows.csv --output artifacts/baseline
```

The baseline is a reference point. The next model should compare against it
using macro-F1, per-class recall, false-alarm rate, and calibration. For a
maintenance product, missed-fault recall and false alarms matter more than
raw accuracy.

