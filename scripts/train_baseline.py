"""Train a first tabular baseline from a CSV of labelled sensor windows.

Expected columns: machine_id, label, and numeric sensor feature columns.
This is intentionally separate from the inference API so the production
service never retrains itself from unverified data.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    args = parser.parse_args()

    try:
        import joblib
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    frame = pd.read_csv(args.csv)
    required = {"machine_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    features = [column for column in frame.columns if column not in required]
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(frame[features], frame["label"], groups=frame["machine_id"]))
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(frame.iloc[train_idx][features], frame.iloc[train_idx]["label"])
    accuracy = model.score(frame.iloc[test_idx][features], frame.iloc[test_idx]["label"])
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "model.joblib")
    (args.output / "metadata.json").write_text(json.dumps({
        "model_version": "machina-rf-0.1.0",
        "features": features,
        "labels": list(model.classes_),
        "group_split_accuracy": accuracy,
    }, indent=2) + "\n")
    print(f"saved {args.output}; group-split accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()

