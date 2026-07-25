"""Run one prompt through the Machina Mistral LoRA adapter."""
from __future__ import annotations

import argparse
import json

from machina_harness.agent import DEFAULT_ADAPTER, DEFAULT_BASE_MODEL, generate_agent_turn, load_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer, model = load_agent(base_model=args.base_model, adapter=args.adapter)
    response = generate_agent_turn(args.prompt, tokenizer, model, max_new_tokens=args.max_new_tokens)
    print(json.dumps({"text": response.text, "tool_calls": response.tool_calls}, indent=2))


if __name__ == "__main__":
    main()
