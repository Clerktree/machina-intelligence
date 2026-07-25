"""QLoRA fine-tuning for the Machina MCP tool-calling agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dataset", type=Path, default=Path("data/machina-agent-sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/machina-agent-mistral-7b-lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the QLoRA run")
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    def render(row: dict) -> str:
        return tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False, add_generation_prompt=False)

    rendered = [{"text": render(row)} for row in rows]
    dataset = Dataset.from_list(rendered).train_test_split(test_size=0.08, seed=42)

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map="auto", torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, gradient_checkpointing=True, learning_rate=2e-4, warmup_ratio=0.05, lr_scheduler_type="cosine", logging_steps=5, eval_steps=50, save_steps=50, eval_strategy="steps", save_total_limit=2, bf16=True, optim="paged_adamw_8bit", report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, train_dataset=tokenized["train"], eval_dataset=tokenized["test"], data_collator=collator)
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "machina-training.json").write_text(json.dumps({"base_model": args.model, "method": "QLoRA", "examples": len(rows), "epochs": args.epochs, "max_length": args.max_length}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(rows)}))


if __name__ == "__main__":
    main()
