"""Build a synthetic-but-grounded SFT set for the Machina tool router.

The dataset teaches routing and response style, not machine facts. Runtime
facts must still come from MCP tool results and the registered model plugins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TOOLS = [
    ("machine_harness_health", "Check service and active model health.", {}, "{\"status\":\"ok\",\"model_version\":\"machina-baseline-0.1.0\"}"),
    ("list_machine_intelligence_capabilities", "List available and planned machine intelligence skills.", {}, "[{\"name\":\"anomaly_detection\",\"status\":\"available\"}]"),
    ("machina_platform_snapshot", "Return asset, telemetry, event, and model counts.", {}, "{\"assets\":3,\"telemetry_records\":12840,\"maintenance_events\":21,\"models\":3}"),
    ("list_machine_assets", "List registered machine assets.", {}, "[{\"asset_id\":\"pump-07\",\"name\":\"Cooling pump 07\",\"asset_type\":\"pump\"}]"),
    ("list_registered_models", "List installed model plugins and versions.", {}, "[{\"model_id\":\"machina-cwru-rf-0.1.0\",\"task\":\"bearing_fault_classification\"}]"),
    ("search_machine_knowledge", "Search indexed maintenance manuals and engineering notes.", {"query": "What should be checked when pump vibration rises?", "asset_type": "pump", "limit": 5}, "[{\"title\":\"Pump maintenance SOP\",\"snippet\":\"Inspect coupling, alignment, lubrication, and bearing temperature.\"}]"),
    ("prepare_maintenance_brief", "Assemble grounded evidence for a maintenance answer.", {"asset_id": "pump-07", "question": "Why did vibration increase this shift?", "asset_type": "pump", "limit": 5}, "{\"asset_id\":\"pump-07\",\"evidence\":[{\"title\":\"Pump maintenance SOP\"}],\"instructions\":[\"Separate observed signals from hypotheses.\"]}"),
    ("estimate_remaining_useful_life", "Estimate remaining useful life from degradation signals.", {"asset_id": "engine-03", "cycle": 180, "max_observed_cycle": 200, "sensors": {"setting_1": 0.2, "sensor_2": 642.1, "sensor_3": 1580.4}}, "{\"asset_id\":\"engine-03\",\"predicted_rul_cycles\":42.7,\"model_version\":\"machina-cmapss-rul-et-0.2.1\"}"),
    ("analyze_machine_energy", "Compare current energy intensity with recent history.", {"asset_id": "compressor-02", "history": [{"energy_kwh": 10.2, "output_units": 100}, {"energy_kwh": 10.4, "output_units": 100}], "current": {"energy_kwh": 12.8, "output_units": 100}}, "{\"asset_id\":\"compressor-02\",\"energy_per_unit\":0.128,\"baseline_energy_per_unit\":0.103,\"status\":\"degraded\"}"),
    ("predict_process_quality", "Predict process failure risk from operating conditions.", {"asset_id": "line-a-12", "machine_type": "M", "air_temperature_k": 298.1, "process_temperature_k": 308.6, "rotational_speed_rpm": 1450, "torque_nm": 42.1, "tool_wear_min": 120}, "{\"asset_id\":\"line-a-12\",\"failure_probability\":0.31,\"predicted_failure_mode\":\"tool_wear\",\"model_version\":\"machina-ai4i-quality-et-0.1.0\"}"),
]

SYSTEM = ("You are Machina, Clertree's sovereign machine-intelligence agent. "
          "You operate as a careful engineering copilot: inspect before acting, "
          "select the smallest useful MCP tool, never invent sensor readings, "
          "and clearly separate observed tool results from hypotheses. Be concise, "
          "calm, practical, and safety-aware. For safety-critical decisions require "
          "qualified human inspection. Available tools are supplied with each example.")


def tool_specs() -> list[dict]:
    return [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in args}, "required": list(args)}}} for n, d, args, _ in TOOLS]


def examples(repeats: int) -> list[dict]:
    rows: list[dict] = []
    prefixes = ["", "Please ", "Can you ", "Machina, ", "I need you to "]
    for repeat in range(repeats):
        for index, (name, description, arguments, result) in enumerate(TOOLS):
            prompt = {
                "machine_harness_health": "check whether the machine intelligence service is healthy",
                "list_machine_intelligence_capabilities": "show me what Machina can do right now",
                "machina_platform_snapshot": "give me a platform snapshot",
                "list_machine_assets": "which machines are registered?",
                "list_registered_models": "which model plugins are installed?",
                "search_machine_knowledge": "search the maintenance knowledge for pump vibration checks",
                "prepare_maintenance_brief": "prepare a grounded maintenance brief for pump-07",
                "estimate_remaining_useful_life": "estimate remaining useful life for engine-03 from its latest cycle data",
                "analyze_machine_energy": "is compressor-02 using more energy per unit than its baseline?",
                "predict_process_quality": "check quality risk on line-a-12 with the current process conditions",
            }[name]
            user = prefixes[(repeat + index) % len(prefixes)] + prompt
            # Mistral's v0.3 chat template requires exactly 9 alphanumeric ID characters.
            call = {"id": f"c{repeat:04d}{index:04d}", "type": "function", "function": {"name": name, "arguments": arguments}}
            final = f"I used `{name}`. Observed result: {result} Next step: validate this against the asset context and follow the site's maintenance procedure; escalate to a qualified engineer if the signal is safety-critical."
            rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}, {"role": "assistant", "tool_calls": [call]}, {"role": "tool", "tool_call_id": call["id"], "content": result}, {"role": "assistant", "content": final}], "tools": tool_specs()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/machina-agent-sft.jsonl"))
    parser.add_argument("--repeats", type=int, default=120)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = examples(args.repeats)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "examples": len(rows), "tools": len(TOOLS)}))


if __name__ == "__main__":
    main()
