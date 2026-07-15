from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training/eval loss curves")
    parser.add_argument("--log", type=Path, required=True, help="loss_history.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="输出 PNG 路径")
    args = parser.parse_args()

    train_steps, train_loss, eval_steps, eval_loss = [], [], [], []
    with args.log.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            step = entry.get("step") or entry.get("epoch")
            if "loss" in entry:
                train_steps.append(float(step))
                train_loss.append(float(entry["loss"]))
            if "eval_loss" in entry:
                eval_steps.append(float(step))
                eval_loss.append(float(entry["eval_loss"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_steps, train_loss, marker="o", label="train loss")
    if eval_loss:
        ax.plot(eval_steps, eval_loss, marker="s", label="eval loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("Qwen3-8B QLoRA SFT loss")
    ax.legend()
    ax.grid(alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"loss_curve_saved={args.output}")


if __name__ == "__main__":
    main()
