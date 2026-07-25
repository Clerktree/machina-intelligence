---
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
pipeline_tag: text-generation
tags:
  - machina
  - clerktree
  - mistral
  - qlora
  - tool-calling
  - predictive-maintenance
  - edge-ai
---

# Machina Agent Mistral 7B LoRA

Machina Agent is a Clertree fine-tuned adapter for machine-intelligence
workflows. It is trained to route natural-language maintenance and operations
questions to Machina MCP tools for fault diagnosis, remaining useful life,
energy analytics, process quality, asset state, and maintenance knowledge.

This repository contains a PEFT LoRA adapter, not a standalone foundation
model. Load it over `mistralai/Mistral-7B-Instruct-v0.3`.

## Intended Use

- Route machine questions to the smallest useful Machina MCP tool.
- Keep specialist model outputs separate from language-model reasoning.
- Produce concise engineering responses that separate observations from
  hypotheses.
- Support local, sovereign, and edge-oriented deployments after quantization.

## Training

- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Method: QLoRA, NF4 4-bit loading, bf16 compute
- Adapter: LoRA on attention and MLP projection modules
- Dataset: synthetic-but-grounded Machina MCP routing examples generated from
  the open-source harness tool schemas

The training examples teach tool selection and Clertree response style. Runtime
facts must still come from telemetry, registered model plugins, maintenance
knowledge, and MCP tool results.

## Quick Start

```python
from machina_harness.agent import generate_agent_turn, load_agent

tokenizer, model = load_agent(adapter="clerktree/machina-agent-mistral-7b-lora")
response = generate_agent_turn("which model plugins are installed?", tokenizer, model)
print(response.text)
print(response.tool_calls)
```

## Safety

Machina is decision support for engineers and operators. It is not a safety
controller. Validate outputs against site procedures, calibrated sensors, and
qualified human inspection before maintenance action.
