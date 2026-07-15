from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class SftDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        skipped = 0
        for sample in samples:
            messages = sample["messages"]
            prompt_text = build_prompt(tokenizer, messages)
            target_text = messages[-1]["content"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + list(target_ids)
            if len(input_ids) > max_length:
                # 截断证据侧（prompt 头部保留系统消息之外从中间截断过于复杂，
                # 这里直接从左侧截断 prompt，保证监督目标完整）。
                overflow = len(input_ids) - max_length
                input_ids = input_ids[overflow:]
                labels = labels[overflow:]
                skipped += 1
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        self.truncated = skipped

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(
                torch.cat([item["input_ids"], torch.full((pad,), self.pad_token_id)])
            )
            labels.append(torch.cat([item["labels"], torch.full((pad,), -100)]))
            attention_mask.append(
                torch.cat(
                    [
                        torch.ones(len(item["input_ids"]), dtype=torch.long),
                        torch.zeros(pad, dtype=torch.long),
                    ]
                )
            )
        return {
            "input_ids": torch.stack(input_ids).long(),
            "labels": torch.stack(labels).long(),
            "attention_mask": torch.stack(attention_mask).long(),
        }


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    model_cfg = config["model"]
    model_path = model_cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=getattr(
            torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    return model, tokenizer


def attach_lora(model: Any, config: dict[str, Any]) -> Any:
    lora_cfg = config["lora"]
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def save_environment_snapshot(output_dir: Path, config: dict[str, Any]) -> None:
    import bitsandbytes
    import peft
    import transformers

    snapshot = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": config["training"]["seed"],
    }
    (output_dir / "environment.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "training_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT training for Qwen3-8B")
    parser.add_argument("--config", type=Path, default=Path("configs/qlora_qwen3_8b.yaml"))
    parser.add_argument("--max-samples", type=int, default=None, help="dry-run 采样条数")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    set_seed(int(training_cfg["seed"]))

    output_dir = args.output_dir or Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs if args.epochs is not None else float(training_cfg["epochs"])

    model, tokenizer = load_model_and_tokenizer(config)
    model = attach_lora(model, config)
    if int(os.environ.get("RANK", 0)) == 0:
        save_environment_snapshot(output_dir, config)

    train_samples = load_jsonl(Path(config["data"]["train_path"]))
    eval_samples = load_jsonl(Path(config["data"]["validation_path"]))
    if args.max_samples is not None:
        train_samples = train_samples[: args.max_samples]
        eval_samples = eval_samples[: max(2, args.max_samples // 5)]

    max_length = int(training_cfg["max_length"])
    train_dataset = SftDataset(train_samples, tokenizer, max_length)
    eval_dataset = SftDataset(eval_samples, tokenizer, max_length)
    print(
        f"train={len(train_dataset)} (truncated={train_dataset.truncated}) "
        f"eval={len(eval_dataset)} (truncated={eval_dataset.truncated})"
    )

    # 多卡 DDP 时按 world size 折算梯度累积，保持总有效 batch 不变
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = max(1, int(training_cfg["gradient_accumulation_steps"]) // world_size)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=int(training_cfg["batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(training_cfg.get("logging_steps", 1)),
        logging_dir=str(output_dir / "logs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=int(training_cfg["seed"]),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    print(f"train_loss={result.training_loss:.4f}")

    if trainer.is_world_process_zero():
        adapter_dir = output_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        log_path = output_dir / "logs" / "loss_history.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            for entry in trainer.state.log_history:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"adapter_saved={adapter_dir}")


if __name__ == "__main__":
    main()
