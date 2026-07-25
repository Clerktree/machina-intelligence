# Process-quality baseline

The first quality plugin uses the UCI AI4I 2020 Predictive Maintenance
Dataset. It predicts normal operation or one of the labeled failure modes from
machine type, air/process temperature, rotational speed, torque, and tool
wear. The dataset is explicitly synthetic, so the resulting model is a
contract and integration baseline rather than a deployment claim.

Source: <https://archive.ics.uci.edu/dataset/601/ai4i>

Run on the lab:

```bash
python scripts/train_quality.py /tmp/ai4i/ai4i2020.csv
```

