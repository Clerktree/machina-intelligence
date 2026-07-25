"""QLoRA fine-tuning for the Machina MCP tool-calling agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dataset", type=Path, default=Path("data/machina-agent-sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/machina-agent-mistral-7b-lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=1280)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the QLoRA run")
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    def render(row: dict) -> dict:
        """Train only the first assistant tool-call turn.

        The prompt contains the system/user messages plus available tools. Labels
        start at the assistant response so the model is not rewarded for copying
        the prompt or the tool schema.
        """
        prompt = tokenizer.apply_chat_template(
            row["messages"][:2],
            tools=row["tools"],
            tokenize=False,
            add_generation_prompt=True,
        )
        calls = []
        for call in row["messages"][2]["tool_calls"]:
            calls.append({
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
                "id": call["id"],
            })
        target = "[TOOL_CALLS] " + json.dumps(calls, ensure_ascii=False) + tokenizer.eos_token
        completion = prompt + target
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(completion, truncation=True, max_length=args.max_length, add_special_tokens=False)["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    rendered = [render(row) for row in rows]
    supervised_tokens = sum(1 for row in rendered for label in row["labels"] if label != -100)
    if supervised_tokens == 0:
        raise SystemExit("No supervised assistant tokens were found; check the chat-template rendering")
    print(json.dumps({"examples": len(rows), "supervised_tokens": supervised_tokens}))
    dataset = Dataset.from_list(rendered).train_test_split(test_size=0.08, seed=42)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map="auto", torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    def collator(features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [tokenizer.pad_token_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    training = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, gradient_checkpointing=True, learning_rate=2e-4, warmup_ratio=0.05, lr_scheduler_type="cosine", logging_steps=5, eval_steps=50, save_steps=50, eval_strategy="steps", save_total_limit=2, bf16=True, optim="paged_adamw_8bit", report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, train_dataset=dataset["train"], eval_dataset=dataset["test"], data_collator=collator)
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "machina-training.json").write_text(json.dumps({"base_model": args.model, "method": "QLoRA", "examples": len(rows), "epochs": args.epochs, "max_length": args.max_length}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(rows)}))


if __name__ == "__main__":
    main()
