# Machina Intelligence — Harness

Machina is an open-source, all-around machine-intelligence platform. It is
designed to understand machines across their lifecycle: what they are doing,
whether they are healthy, why they are drifting, what will fail next, how much
energy they use, how they affect quality, and what an engineer should do.

The first MVP targets rotating equipment, but the bearing model is only the
first capability plugin. The platform contract is deliberately broader so
future models can cover pumps, motors, compressors, turbines, production
lines, robots, and other equipment without renaming the product.

## Current capability registry

- Available: anomaly detection
- Available: bearing fault diagnosis
- Available: remaining useful life (C-MAPSS baseline)
- Available: energy-efficiency analytics baseline
- Available: process-quality/failure prediction baseline (AI4I synthetic data)
- Planned: energy intelligence
- Planned: quality prediction
- Planned: maintenance copilot
- Planned: machine knowledge graph

Query `/v1/capabilities` for the machine-readable registry.

Check whether the bundled model plugins are actually available at runtime:

```bash
curl http://127.0.0.1:8000/v1/model-health
```

Local source-tree runs automatically use the versioned artifacts in
`artifacts/`. Deployments can override them with `MACHINA_CLASSIFIER_PATH`,
`MACHINA_RUL_MODEL_PATH`, and `MACHINA_QUALITY_MODEL_PATH`.

## MVP scope

- Accept timestamped sensor windows as JSON or CSV-derived records.
- Produce an anomaly score, severity, contributing sensors, and a safe
  maintenance recommendation.
- Keep the model interface independent from the LLM interface.
- Run locally on CPU; use the lab GPU for training larger time-series models.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn machina_harness.api:app --reload
```

Then open `http://127.0.0.1:8000/docs` or call:

```bash
curl -X POST http://127.0.0.1:8000/v1/analyze \
  -H 'content-type: application/json' \
  -d @configs/sample-window.json
```

For a reproducible packaged runtime with the bundled models:

```bash
docker compose up --build -d
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for persistent storage and
production hardening guidance.

See [docs/INDUSTRY_READINESS.md](docs/INDUSTRY_READINESS.md) for the evidence
and standards-alignment gate required before a production deployment.

Model artifacts and Hugging Face publication instructions are in
[docs/HUGGINGFACE_RELEASE.md](docs/HUGGINGFACE_RELEASE.md).
Run [scripts/verify_release.py](scripts/verify_release.py) before publishing.

## Clertree release hub

- [Machina Intelligence landing Space](https://huggingface.co/spaces/clerktree/machina-intelligence-hub)
- [Enhanced bearing fault model](https://huggingface.co/clerktree/machina-cwru-bearing-fault)
- [Remaining useful life model](https://huggingface.co/clerktree/machina-cmapss-rul)
- [Process quality model](https://huggingface.co/clerktree/machina-ai4i-quality)
- Agent adapter: `clerktree/machina-agent-mistral-7b-lora` after the lab fine-tune is published.

The static landing-page source is in [hf-space](hf-space).

## Machina agent

Machina uses a two-layer model design. The specialist models produce grounded
machine outputs for faults, RUL, energy, and quality; the agent is a
Mistral-based Clertree adapter that routes requests to those tools and writes
operator-ready responses.

Train the adapter with [scripts/train_machina_agent.py](scripts/train_machina_agent.py)
and run a local prompt after publishing or copying an adapter:

```bash
pip install -e '.[agent]'
python scripts/run_machina_agent.py \
  --adapter clerktree/machina-agent-mistral-7b-lora \
  "which model plugins are installed?"
```

The agent runner emits the model text for the next turn. In production, execute
requested tool calls against the MCP server, append the tool result, and ask the
agent for the final grounded answer.

## Roadmap

1. Establish a reproducible baseline with public bearing datasets.
2. Train and evaluate a fault classifier and remaining-useful-life estimator.
3. Add connectors for MQTT, OPC-UA, REST, and uploaded CSV files.
4. Connect the API to the Grok Build agent as a machinery diagnostic tool.
5. Publish versioned models, datasets, model cards, and a browser demo on
   Hugging Face.

The model is decision support, not a safety controller. Human inspection and
site-specific validation are required before maintenance action.
