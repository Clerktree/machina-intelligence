# Release audit

Run this before uploading model directories to Hugging Face:

```bash
python scripts/verify_release.py artifacts
```

The audit requires every model release to contain a model card, metadata, and
weights, and prints a SHA-256 hash for reproducibility. It does not claim that
benchmark performance transfers to a customer machine; each model card keeps
that limitation explicit.

