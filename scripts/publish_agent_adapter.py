"""Publish a trained Machina LoRA adapter directory to Hugging Face Hub."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/machina-agent-mistral-7b-lora"))
    parser.add_argument("--repo-id", default="clerktree/machina-agent-mistral-7b-lora")
    args = parser.parse_args()

    required = ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"]
    missing = [name for name in required if not (args.source / name).exists()]
    if missing:
        raise SystemExit(f"Missing adapter files in {args.source}: {', '.join(missing)}")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(args.source), repo_id=args.repo_id, repo_type="model")
    print(json.dumps({"repo_id": args.repo_id, "source": str(args.source)}))


if __name__ == "__main__":
    main()
