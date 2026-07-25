# Machina agent fine-tune

Machina has two model layers:

1. Specialist models operate on machine signals: bearing faults, RUL, energy, and process quality.
2. The Machina agent routes natural-language requests to those specialists and to the MCP platform tools.

The agent is a QLoRA adapter over `mistralai/Mistral-7B-Instruct-v0.3`. The base model is Apache-2.0 and already supports function calling; the Clertree adapter teaches the machine-intelligence vocabulary, conservative engineering behavior, and Machina tool names. This is a fine-tune, not a claim that Clertree trained a foundation model from scratch.

## Train on the lab GPU

```bash
python scripts/build_agent_dataset.py --repeats 120
python scripts/train_machina_agent.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/machina-agent-sft.jsonl \
  --output artifacts/machina-agent-mistral-7b-lora \
  --epochs 2
```

The run uses 4-bit NF4 quantization, bf16 compute, gradient checkpointing, and LoRA. The training examples teach tool selection and final answer style; live tool results remain authoritative at inference time. Validate the adapter with held-out routing examples and MCP integration tests before calling it production-ready.

## Inference

The runtime helper loads the base model plus the Clertree adapter and renders
the same MCP tool schemas used during training:

```bash
pip install -e '.[agent]'
python scripts/run_machina_agent.py \
  --adapter clerktree/machina-agent-mistral-7b-lora \
  "prepare a maintenance brief for pump-07"
```

The first generation should either request the smallest useful MCP tool or give
a short grounded response when enough context is already present. The MCP tool
result is still the source of truth; the language model should not fabricate
telemetry, asset state, or safety conclusions.

For European edge deployments, merge the adapter into the base model and export a quantized GGUF or AWQ artifact for a local runtime. Keep the specialist regressors separate: the small language model is the orchestrator, not a replacement for signal models.
