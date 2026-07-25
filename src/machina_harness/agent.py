"""Runtime helpers for the Machina Mistral tool-calling agent."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "clerktree/machina-agent-mistral-7b-lora"

SYSTEM_PROMPT = (
    "You are Machina, Clertree's sovereign machine-intelligence agent. "
    "You operate as a careful engineering copilot: inspect before acting, "
    "select the smallest useful MCP tool, never invent sensor readings, and "
    "clearly separate observed tool results from hypotheses. Be concise, calm, "
    "practical, and safety-aware. For safety-critical decisions require "
    "qualified human inspection."
)


MCP_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "machine_harness_health",
            "description": "Check service and active model health.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_machine_intelligence_capabilities",
            "description": "List available and planned machine intelligence skills.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "machina_platform_snapshot",
            "description": "Return asset, telemetry, event, and model counts.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_machine_assets",
            "description": "List registered machine assets.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_registered_models",
            "description": "List installed model plugins and versions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_machine_knowledge",
            "description": "Search indexed maintenance manuals and engineering notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "asset_type": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_maintenance_brief",
            "description": "Assemble grounded evidence for a maintenance answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "question": {"type": "string"},
                    "asset_type": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["asset_id", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_remaining_useful_life",
            "description": "Estimate remaining useful life from degradation signals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "cycle": {"type": "number"},
                    "max_observed_cycle": {"type": "number"},
                    "sensors": {"type": "object"},
                },
                "required": ["asset_id", "cycle", "max_observed_cycle", "sensors"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_machine_energy",
            "description": "Compare current energy intensity with recent history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "history": {"type": "array", "items": {"type": "object"}},
                    "current": {"type": "object"},
                },
                "required": ["asset_id", "history", "current"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_process_quality",
            "description": "Predict process failure risk from operating conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "machine_type": {"type": "string"},
                    "air_temperature_k": {"type": "number"},
                    "process_temperature_k": {"type": "number"},
                    "rotational_speed_rpm": {"type": "number"},
                    "torque_nm": {"type": "number"},
                    "tool_wear_min": {"type": "number"},
                },
                "required": [
                    "asset_id",
                    "machine_type",
                    "air_temperature_k",
                    "process_temperature_k",
                    "rotational_speed_rpm",
                    "torque_nm",
                    "tool_wear_min",
                ],
            },
        },
    },
]


@dataclass(frozen=True)
class AgentResponse:
    """Generated text plus any parsed tool call metadata exposed by Transformers."""

    text: str
    tool_calls: list[dict[str, Any]]


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Mistral tool-call generations when they are emitted as JSON."""
    match = re.search(r"\[TOOL_CALLS\]\s*(\[.*\])", text, flags=re.DOTALL)
    payload = match.group(1).strip() if match else text.strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def load_agent(base_model: str = DEFAULT_BASE_MODEL, adapter: str = DEFAULT_ADAPTER):
    """Load the Machina adapter over its Mistral base model.

    Heavy ML imports stay inside this function so the API/MCP server can run
    without agent dependencies installed.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def generate_agent_turn(
    prompt: str,
    tokenizer: Any,
    model: Any,
    *,
    max_new_tokens: int = 256,
) -> AgentResponse:
    """Generate one Machina agent turn with the MCP tool schema in context."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=MCP_TOOL_SPECS,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(encoded, device=model.device)
    with torch.no_grad():
        output = model.generate(
            encoded,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, encoded.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return AgentResponse(text=text, tool_calls=parse_tool_calls(text))
