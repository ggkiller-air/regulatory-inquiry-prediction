"""自动指标：字符级 ROUGE-L（宏平均）与 BERTScore-F1（宏平均）。

清洗规则（预测与参考完全一致）：
1. 去除 <think>...</think> 段（若存在）；
2. 全角标点归一为半角，去掉所有空白字符（空格/换行/制表）；
3. 中文文本按单字符序列计算 ROUGE-L。

BERTScore：bert-base-chinese（本地 models/bert-base-chinese），默认层，
不使用 baseline rescaling，取全部测试样本 F1 的宏平均。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

THINK_RE = re.compile(r"<think>.*?</think>", re.S)

FULL2HALF = str.maketrans(
    "，。！？；：（）【】《》“”‘’、０１２３４５６７８９％",
    ",.!?;:()[]<>\"\"'',0123456789%",
)


def clean_text(text: str) -> str:
    text = THINK_RE.sub("", str(text))
    text = text.translate(FULL2HALF)
    text = re.sub(r"[*#]", "", text)  # markdown 装饰符，所有方法统一去除
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^(监管问询子问题|问询子问题|监管问询问题)[::]?", "", text)
    return text.strip()


def lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if ca == cb else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred, ref = clean_text(pred), clean_text(ref)
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bert-model", default="models/bert-base-chinese")
    args = parser.parse_args()

    import bert_score
    import torch

    results = []
    per_sample_dir = args.output.parent / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    for pred_path in args.predictions:
        rows = load_jsonl(pred_path)
        preds = [clean_text(r["prediction"]) for r in rows]
        refs = [clean_text(r["reference"]) for r in rows]
        rouge = [rouge_l_f1(r["prediction"], r["reference"]) for r in rows]

        p, rcl, f1 = bert_score.score(
            preds,
            refs,
            model_type=args.bert_model,
            num_layers=12,
            lang="zh",
            rescale_with_baseline=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=16,
        )
        f1_list = [float(x) for x in f1]

        name = pred_path.stem
        per_rows = [
            {
                "sample_id": r["sample_id"],
                "rouge_l_f1": round(rg, 6),
                "bertscore_f1": round(bs, 6),
            }
            for r, rg, bs in zip(rows, rouge, f1_list)
        ]
        with (per_sample_dir / f"{name}_metrics.jsonl").open("w", encoding="utf-8") as f:
            for row in per_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        results.append(
            {
                "config": name,
                "prediction_file": str(pred_path),
                "n_samples": len(rows),
                "rouge_l_f1_macro": sum(rouge) / len(rouge),
                "bertscore_f1_macro": sum(f1_list) / len(f1_list),
            }
        )
        print(results[-1])

    meta = {
        "bertscore_model": args.bert_model,
        "bertscore_model_source": "google-bert/bert-base-chinese",
        "bertscore_num_layers": 12,
        "rescale_with_baseline": False,
        "bert_score_version": bert_score.__version__,
        "rouge_l": "character-level LCS F1, macro-averaged",
        "cleaning": "remove <think> blocks, full-width->half-width punctuation, strip all whitespace",
    }
    args.output.write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
