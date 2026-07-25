"""Train an auditable CWRU bearing classifier with time/frequency features.

The split is grouped by source file. Windows from one recording never appear in
both train and test, which avoids the common CWRU leakage failure mode.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


FEATURE_VERSION = "cwru-signal-v2"


def label_for(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "normal" in parts:
        return "normal"
    for label in ("ir", "or", "b"):
        if label in parts:
            return {"ir": "inner_race", "or": "outer_race", "b": "ball"}[label]
    raise ValueError(f"Cannot infer fault label from {path}")


def _safe_kurtosis(values: np.ndarray) -> float:
    variance = float(np.mean(values ** 2))
    return float(np.mean(values ** 4) / (variance ** 2 + 1e-12))


def features(signal: np.ndarray, sample_rate: float = 12000.0) -> list[float]:
    """Extract stable time, spectrum, and envelope features from one window."""
    from scipy.signal import hilbert

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 1024:
        raise ValueError("signal window too short")
    centered = values - np.mean(values)
    absolute = np.abs(centered)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    peak = float(np.max(absolute))
    mean_abs = float(np.mean(absolute))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / sample_rate)
    spectrum[0] = 0.0
    power = spectrum ** 2
    total_power = float(np.sum(power) + 1e-12)
    normalized_power = power / total_power
    nyquist = sample_rate / 2.0
    edges = np.asarray([0, 500, 1000, 2000, 3000, 4000, 5000, nyquist + 1])
    band_energy = [float(np.sum(power[(frequencies >= low) & (frequencies < high)]) / total_power)
                   for low, high in zip(edges[:-1], edges[1:])]
    envelope = np.abs(hilbert(centered))
    envelope_spectrum = np.abs(np.fft.rfft((envelope - envelope.mean()) * np.hanning(envelope.size)))
    envelope_spectrum[0] = 0.0
    envelope_frequency = frequencies[int(np.argmax(envelope_spectrum))]
    dominant_frequency = frequencies[int(np.argmax(spectrum))]
    spectral_centroid = float(np.sum(frequencies * power) / total_power)
    spectral_entropy = float(-np.sum(normalized_power * np.log(normalized_power + 1e-12)))
    crest = peak / (rms + 1e-12)
    impulse = peak / (mean_abs + 1e-12)
    shape = rms / (mean_abs + 1e-12)
    return [
        rms, peak, mean_abs, _safe_kurtosis(centered), crest, impulse, shape,
        float(np.std(centered)), float(np.min(centered)), float(np.max(centered)),
        dominant_frequency, spectral_centroid, spectral_entropy, envelope_frequency,
        *band_energy,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cwru-enhanced"))
    parser.add_argument("--windows-per-file", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=4096)
    args = parser.parse_args()

    try:
        import joblib
        from scipy.io import loadmat
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        from sklearn.model_selection import GroupShuffleSplit
        import sklearn
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    files = sorted(args.data_root.rglob("*.mat"))
    rows, groups, labels, rpms = [], [], [], []
    for path in files:
        label = label_for(path)
        mat = loadmat(path)
        signal_keys = sorted(key for key in mat if re.search(r"_(DE|FE)_time$", key))
        if not signal_keys:
            continue
        signal = np.asarray(mat[signal_keys[0]]).reshape(-1)
        starts = np.linspace(0, max(0, signal.size - args.window_size), args.windows_per_file, dtype=int)
        for start in np.unique(starts):
            window = signal[start:start + args.window_size]
            if window.size >= args.window_size:
                rows.append(features(window))
                groups.append(path.name)
                labels.append(label)
                rpms.append(path.stem.rsplit("_", 1)[-1])
    if len(set(groups)) < 8 or len(set(labels)) < 4:
        raise SystemExit("Need at least 8 source files and all four classes")

    X = np.asarray(rows)
    labels = np.asarray(labels)
    rpms = np.asarray(rpms)
    split = GroupShuffleSplit(n_splits=200, test_size=0.25, random_state=42)
    all_labels = set(labels)
    train_idx = test_idx = None
    for candidate_train, candidate_test in split.split(X, labels, groups=groups):
        if set(labels[candidate_train]) == all_labels and set(labels[candidate_test]) == all_labels:
            train_idx, test_idx = candidate_train, candidate_test
            break
    if train_idx is None:
        raise SystemExit("Could not find a grouped split containing every class")

    def make_candidates():
        return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600, class_weight="balanced", max_features="sqrt",
            min_samples_leaf=1, random_state=42, n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600, class_weight="balanced", max_features="sqrt",
            min_samples_leaf=1, random_state=42, n_jobs=-1,
        ),
        }

    candidates = make_candidates()
    results = {}
    best_name = None
    best_score = -1.0
    for name, model in candidates.items():
        model.fit(X[train_idx], labels[train_idx])
        prediction = model.predict(X[test_idx])
        score = float(f1_score(labels[test_idx], prediction, average="macro"))
        results[name] = {
            "macro_f1": score,
            "accuracy": float(np.mean(prediction == labels[test_idx])),
            "classification_report": classification_report(labels[test_idx], prediction, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels[test_idx], prediction, labels=sorted(all_labels)).tolist(),
        }
        if score > best_score:
            best_name, best_score = name, score

    # Harder robustness check: hold out every operating speed in turn. A
    # random file split can still make the benchmark look too easy when the
    # same speeds and fault severities occur on both sides of the split.
    rpm_results = {}
    for name in candidates:
        scores = []
        for held_out_rpm in sorted(set(rpms)):
            train_mask = rpms != held_out_rpm
            test_mask = rpms == held_out_rpm
            model = make_candidates()[name]
            model.fit(X[train_mask], labels[train_mask])
            prediction = model.predict(X[test_mask])
            scores.append(float(f1_score(labels[test_mask], prediction, average="macro")))
        rpm_results[name] = {
            "per_rpm_macro_f1": dict(zip(sorted(set(rpms)), scores)),
            "mean_macro_f1": float(np.mean(scores)),
            "min_macro_f1": float(np.min(scores)),
        }
    best_name = max(rpm_results, key=lambda name: rpm_results[name]["mean_macro_f1"])
    best_model = make_candidates()[best_name]
    best_model.fit(X, labels)
    args.output.mkdir(parents=True, exist_ok=True)
    model_version = "machina-cwru-enhanced-et-0.2.0" if best_name == "extra_trees" else "machina-cwru-enhanced-rf-0.2.0"
    joblib.dump({"model": best_model, "feature_version": FEATURE_VERSION, "model_version": model_version}, args.output / "model.joblib")
    metadata = {
        "model_version": model_version,
        "model_family": best_name,
        "feature_version": FEATURE_VERSION,
        "dataset": "CWRU Bearing Fault Dataset, 12 kHz drive-end files",
        "source_files": len(set(groups)),
        "windows": len(rows),
        "window_size": args.window_size,
        "features": int(X.shape[1]),
        "labels": sorted(all_labels),
        "split": "GroupShuffleSplit by source file, random_state=42",
        "results": results,
        "leave_one_rpm_out": rpm_results,
        "selected_model": best_name,
        "training_sklearn_version": sklearn.__version__,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
