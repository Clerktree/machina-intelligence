"""Generate a small labelled dataset for smoke-testing the training pipeline."""

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/demo-windows.csv"))
    args = parser.parse_args()
    random.seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["machine_id", "label", "vibration_rms", "temperature_c", "motor_current_a"]
    labels = ["normal", "imbalance", "bearing_fault"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for machine_index in range(18):
            label = labels[machine_index // 6]
            for _ in range(20):
                values = {
                    "normal": (0.12, 61.0, 15.0),
                    "imbalance": (0.42, 63.0, 16.5),
                    "bearing_fault": (0.72, 68.0, 17.5),
                }[label]
                writer.writerow({
                    "machine_id": f"demo-motor-{machine_index:03d}",
                    "label": label,
                    "vibration_rms": round(random.gauss(values[0], 0.02), 5),
                    "temperature_c": round(random.gauss(values[1], 0.5), 3),
                    "motor_current_a": round(random.gauss(values[2], 0.2), 3),
                })
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

