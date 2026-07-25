"""Audit all prepared model directories before a Hugging Face upload."""

import hashlib
import json
import sys
from pathlib import Path


REQUIRED_FILES = ("README.md", "metadata.json", "model.joblib")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path) -> list[dict]:
    results = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
        if missing:
            raise SystemExit(f"{directory}: missing {missing}")
        metadata = json.loads((directory / "metadata.json").read_text())
        for field in ("model_version", "dataset", "features"):
            if field not in metadata:
                raise SystemExit(f"{directory}: metadata missing {field}")
        if "labels" not in metadata and "official_test" not in metadata and "classification_report" not in metadata:
            raise SystemExit(f"{directory}: metadata has no evaluation shape (labels or regression metrics)")
        if directory.name == "cwru-enhanced":
            rpm_check = metadata.get("leave_one_rpm_out", {}).get("extra_trees", {})
            if rpm_check.get("min_macro_f1", 0) < 0.90:
                raise SystemExit(f"{directory}: leave-one-RPM minimum macro-F1 is below the release gate")
            model_card = " ".join((directory / "README.md").read_text().lower().split())
            if "not evidence of generalization" not in model_card and "not industrial generalization" not in model_card:
                raise SystemExit(f"{directory}: model card must state the benchmark limitation")
        results.append({
            "directory": directory.name,
            "model_version": metadata["model_version"],
            "model_bytes": (directory / "model.joblib").stat().st_size,
            "model_sha256": sha256(directory / "model.joblib"),
        })
    if len(results) < 3:
        raise SystemExit("Expected at least three model release directories")
    return results


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
    print(json.dumps(audit(root), indent=2))
