"""Train a process/failure-mode classifier on the UCI AI4I dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ai4i-quality"))
    args = parser.parse_args()
    try:
        import joblib
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    frame = pd.read_csv(args.csv)
    frame["failure_mode"] = np.select(
        [frame["TWF"] == 1, frame["HDF"] == 1, frame["PWF"] == 1, frame["OSF"] == 1, frame["RNF"] == 1],
        ["tool_wear", "heat_dissipation", "power", "overstrain", "random"],
        default="normal",
    )
    type_code = frame["Type"].map({"L": 0, "M": 1, "H": 2}).astype(float)
    feature_columns = ["type_code", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    X = frame[["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]].copy()
    X.insert(0, "type_code", type_code)
    y = frame["failure_mode"]
    train_index, test_index = train_test_split(np.arange(len(frame)), test_size=0.25, random_state=42, stratify=y)
    model = ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_index], y.iloc[train_index])
    prediction = model.predict(X.iloc[test_index])
    labels = sorted(y.unique())
    report = classification_report(y.iloc[test_index], prediction, labels=labels, output_dict=True, zero_division=0)
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "model.joblib")
    metadata = {
        "model_version": "machina-ai4i-quality-et-0.1.0",
        "model_family": "ExtraTreesClassifier",
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "synthetic_dataset": True,
        "features": feature_columns,
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y.iloc[test_index], prediction, labels=labels).tolist(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

