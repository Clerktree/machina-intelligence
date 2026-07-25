"""Publish a prepared model directory to Hugging Face Hub.

Authentication is intentionally external: run `hf auth login` or provide
`HF_TOKEN` in the environment. The script never stores credentials.
"""

import os
from pathlib import Path


def main() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install publishing dependencies with: pip install -e '.[train]'") from exc
    repo_id = os.getenv("HF_REPO_ID")
    if not repo_id:
        raise SystemExit("Set HF_REPO_ID, e.g. your-org/machina-cwru-bearing-classifier")
    source = Path(os.getenv("HF_MODEL_DIR", "artifacts/cwru-baseline"))
    if not (source / "model.joblib").exists():
        raise SystemExit(f"Missing model artifact: {source / 'model.joblib'}")
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(source), repo_id=repo_id, repo_type="model")
    print(f"published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

