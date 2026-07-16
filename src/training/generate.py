from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions with a LoRA adapter")
    parser.add_argument("--config", type=Path, default=Path("configs/qlora_qwen3_8b.yaml"))
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter 目录；不传则使用基座模型")
    parser.add_argument("--input", type=Path, required=True, help="SFT 格式 jsonl")
    parser.add_argument("--output", type=Path, required=True, help="预测结果 jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    model_path = model_cfg["model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    samples = load_jsonl(args.input)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for index, sample in enumerate(samples, start=1):
            messages = sample["messages"][:-1]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            record = {
                "sample_id": sample.get("sample_id"),
                "company": sample.get("metadata", {}).get("company"),
                "stock_code": sample.get("metadata", {}).get("stock_code"),
                "report_year": sample.get("metadata", {}).get("report_year"),
                "reference": sample["messages"][-1]["content"],
                "prediction": generated,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(samples)}] {record['sample_id']}")

    print(f"predictions_saved={args.output}")


if __name__ == "__main__":
    main()
