"""Train and evaluate a C-MAPSS remaining-useful-life baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_cmapss(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")
    frame.columns = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [
        f"sensor_{index}" for index in range(1, frame.shape[1] - 4)
    ]
    return frame


def add_features(frame: pd.DataFrame, sensor_columns: list[str]) -> pd.DataFrame:
    result = frame[["unit", "cycle", *sensor_columns]].copy()
    max_cycles = result.groupby("unit")["cycle"].transform("max")
    result["cycle_fraction"] = result["cycle"] / max_cycles
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--output", type=Path, default=Path("artifacts/rul-cmapss"))
    args = parser.parse_args()

    try:
        import joblib
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    train = read_cmapss(args.data_root / f"train_{args.subset}.txt")
    test = read_cmapss(args.data_root / f"test_{args.subset}.txt")
    rul = np.loadtxt(args.data_root / f"RUL_{args.subset}.txt")
    sensor_columns = [column for column in train.columns if column.startswith("sensor_")]
    # Remove constant channels; they add no information and destabilize small baselines.
    sensor_columns = [column for column in sensor_columns if train[column].nunique() > 1]
    train["rul"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    train["rul"] = train["rul"].clip(upper=125)
    train_features = add_features(train, sensor_columns)
    test_last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).copy()
    test_features = add_features(test_last, sensor_columns)
    feature_columns = ["cycle", "cycle_fraction", *sensor_columns]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, validation_index = next(splitter.split(train_features, train["rul"], groups=train["unit"]))
    model = ExtraTreesRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_features.iloc[train_index][feature_columns], train.iloc[train_index]["rul"])
    validation_prediction = np.maximum(0, model.predict(train_features.iloc[validation_index][feature_columns]))
    validation_target = train.iloc[validation_index]["rul"].to_numpy()
    official_prediction = np.maximum(0, model.predict(test_features[feature_columns]))
    official_target = rul[:len(official_prediction)]

    def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
        return {
            "mae_cycles": float(mean_absolute_error(target, prediction)),
            "rmse_cycles": float(np.sqrt(mean_squared_error(target, prediction))),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "model.joblib")
    metadata = {
        "model_version": "machina-cmapss-rul-et-0.2.1",
        "model_family": "ExtraTreesRegressor",
        "dataset": f"NASA C-MAPSS {args.subset}",
        "features": feature_columns,
        "validation_engines": int(len(set(train.iloc[validation_index]["unit"]))),
        "validation": metrics(validation_target, validation_prediction),
        "official_test_engines": int(len(official_prediction)),
        "official_test": metrics(official_target, official_prediction),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
