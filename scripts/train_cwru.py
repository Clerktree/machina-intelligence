"""Train a reproducible CWRU bearing-fault baseline from MATLAB files.

The split is by source file, not by randomly mixed windows, to avoid leaking
near-identical adjacent samples between train and test.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def label_for(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "normal" in parts:
        return "normal"
    for label in ("ir", "or", "b"):
        if label in parts:
            return {"ir": "inner_race", "or": "outer_race", "b": "ball"}[label]
    raise ValueError(f"Cannot infer fault label from {path}")


def features(signal: np.ndarray) -> list[float]:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    signal = signal[np.isfinite(signal)]
    if signal.size < 32:
        raise ValueError("signal window too short")
    centered = signal - np.mean(signal)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    peak = float(np.max(np.abs(centered)))
    variance = float(np.mean(centered ** 2))
    kurtosis = float(np.mean(centered ** 4) / (variance ** 2 + 1e-12))
    crest = peak / (rms + 1e-12)
    return [rms, peak, kurtosis, crest]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cwru-baseline"))
    parser.add_argument("--windows-per-file", type=int, default=8)
    args = parser.parse_args()

    try:
        import joblib
        from scipy.io import loadmat
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    files = sorted(args.data_root.rglob("*.mat"))
    if not files:
        raise SystemExit(f"No .mat files found below {args.data_root}")
    rows, groups, labels = [], [], []
    for path in files:
        label = label_for(path)
        mat = loadmat(path)
        signal_keys = sorted(key for key in mat if re.search(r"_(DE|FE)_time$", key))
        if not signal_keys:
            continue
        signal = np.asarray(mat[signal_keys[0]]).reshape(-1)
        starts = np.linspace(0, max(0, signal.size - 4096), args.windows_per_file, dtype=int)
        for start in np.unique(starts):
            window = signal[start:start + 4096]
            if window.size >= 1024:
                rows.append(features(window))
                groups.append(path.name)
                labels.append(label)
    if len(set(groups)) < 5:
        raise SystemExit("Not enough source files for a file-level split")

    from sklearn.model_selection import GroupShuffleSplit
    X = np.asarray(rows)
    labels = np.asarray(labels)
    split = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=42)
    all_labels = set(labels)
    train_idx = test_idx = None
    for candidate_train, candidate_test in split.split(X, labels, groups=groups):
        if set(labels[candidate_train]) == all_labels and set(labels[candidate_test]) == all_labels:
            train_idx, test_idx = candidate_train, candidate_test
            break
    if train_idx is None:
        raise SystemExit("Could not find a group split containing every class in both folds")
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X[train_idx], labels[train_idx])
    prediction = model.predict(X[test_idx])
    report = classification_report(labels[test_idx], prediction, output_dict=True, zero_division=0)
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "model.joblib")
    metadata = {
        "model_version": "machina-cwru-rf-0.1.0",
        "dataset": "CWRU Bearing Fault Dataset",
        "source_files": len(set(groups)),
        "windows": len(rows),
        "features": ["rms", "peak", "kurtosis", "crest_factor"],
        "labels": sorted(set(labels)),
        "classification_report": report,
        "confusion_matrix_labels": sorted(set(labels)),
        "confusion_matrix": confusion_matrix(labels[test_idx], prediction, labels=sorted(set(labels))).tolist(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
