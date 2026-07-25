# Hugging Face release plan

Machina publishes separate model repositories so users can download only the
capability they need:

- `machina-cwru-bearing-fault`
- `machina-cmapss-rul`
- `machina-ai4i-quality`

Each artifact directory contains `model.joblib`, `metadata.json`, and a model
card. Raw benchmark datasets are not redistributed. The AI4I model card marks
the source dataset as synthetic; the CWRU and C-MAPSS cards identify their
benchmark provenance and evaluation limits.

## Publish

Authenticate outside the repository and publish one directory at a time:

```bash
hf auth login

HF_REPO_ID=your-org/machina-cwru-bearing-fault \
HF_MODEL_DIR=artifacts/cwru-baseline \
python scripts/publish_hf.py

HF_REPO_ID=your-org/machina-cmapss-rul \
HF_MODEL_DIR=artifacts/rul-cmapss \
python scripts/publish_hf.py

HF_REPO_ID=your-org/machina-ai4i-quality \
HF_MODEL_DIR=artifacts/ai4i-quality \
python scripts/publish_hf.py
```

The script accepts `HF_TOKEN` for CI, but credentials must never be committed
to the repository or baked into the Docker image.

For a repeatable multi-model release, add an `HF_TOKEN` repository secret and
run the manual `Publish Machina models` GitHub Actions workflow. It audits all
three artifact directories before starting the parallel uploads.

The model files use Git LFS because the RUL checkpoint is over 100 MB. Install
Git LFS before committing the artifacts:

```bash
git lfs install
git add artifacts .gitattributes
git commit -m "Add Machina model release artifacts"
```
