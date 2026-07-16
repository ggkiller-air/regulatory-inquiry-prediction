"""Multi-reference 评测：按相同模型输入合并测试子问题为输入组。

同一（公司，风险主题）下的子问题在无泄漏设定中模型输入完全相同（含证据），
贪心解码输出也相同。因此按输入分组，每组一条预测、多条参考问询：
ROUGE-L / BERTScore-F1 取该预测与组内各参考的最高匹配，再对组宏平均。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .auto_metrics import clean_text, rouge_l_f1

INPUT_FILES = {
    "zero_shot": "outputs/eval_inputs/test_no_evidence.jsonl",
    "base_evidence": "data/training_leakage_free/test.jsonl",
    "qlora_no_evidence": "outputs/eval_inputs/test_no_evidence.jsonl",
    "full_model": "data/training_leakage_free/test.jsonl",
    "wo_company_year": "outputs/eval_inputs/test_global_evidence.jsonl",
}


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    import bert_score
    import torch

    results = []
    group_detail: dict[str, list[dict]] = {}

    for config, input_path in INPUT_FILES.items():
        inputs = load_jsonl(input_path)
        preds = {r["sample_id"]: r for r in load_jsonl(f"outputs/eval_predictions/{config}.jsonl")}

        groups: dict[str, dict] = {}
        for s in inputs:
            key = hashlib.sha256(s["messages"][1]["content"].encode()).hexdigest()
            g = groups.setdefault(
                key, {"sample_ids": [], "references": [], "predictions": set(), "meta": s["metadata"]}
            )
            g["sample_ids"].append(s["sample_id"])
            g["references"].append(preds[s["sample_id"]]["reference"])
            g["predictions"].add(preds[s["sample_id"]]["prediction"])

        # 同组输入相同 + 贪心解码 → 预测必须唯一
        for g in groups.values():
            assert len(g["predictions"]) == 1, f"{config}: multiple predictions in one input group"

        # BERTScore：展开为 (pred, ref) 全部配对一次算完
        pair_preds, pair_refs, pair_group = [], [], []
        for key, g in groups.items():
            p = clean_text(next(iter(g["predictions"])))
            for ref in g["references"]:
                pair_preds.append(p)
                pair_refs.append(clean_text(ref))
                pair_group.append(key)
        _, _, f1 = bert_score.score(
            pair_preds,
            pair_refs,
            model_type="models/bert-base-chinese",
            num_layers=12,
            lang="zh",
            rescale_with_baseline=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=32,
        )
        bert_by_group: dict[str, float] = {}
        for key, score in zip(pair_group, [float(x) for x in f1]):
            bert_by_group[key] = max(bert_by_group.get(key, 0.0), score)

        rouge_scores, detail = [], []
        for key, g in groups.items():
            pred = next(iter(g["predictions"]))
            best_rouge = max(rouge_l_f1(pred, ref) for ref in g["references"])
            rouge_scores.append(best_rouge)
            detail.append(
                {
                    "group_key": key[:12],
                    "company": g["meta"]["company"],
                    "risk_topic": g["meta"]["risk_topic"],
                    "n_references": len(g["references"]),
                    "sample_ids": g["sample_ids"],
                    "rouge_l_best": round(best_rouge, 6),
                    "bertscore_f1_best": round(bert_by_group[key], 6),
                }
            )
        group_detail[config] = detail

        n_groups = len(groups)
        results.append(
            {
                "config": config,
                "n_groups": n_groups,
                "n_samples": len(inputs),
                "rouge_l_f1_macro": sum(rouge_scores) / n_groups,
                "bertscore_f1_macro": sum(bert_by_group.values()) / n_groups,
            }
        )
        print(results[-1])

    out = {
        "meta": {
            "protocol": "multi-reference by identical model input; best match per group, macro over groups",
            "bertscore_model": "models/bert-base-chinese (google-bert/bert-base-chinese)",
            "bertscore_num_layers": 12,
            "rescale_with_baseline": False,
            "rouge_l": "character-level LCS F1, best over group references, macro over groups",
            "cleaning": "same as auto_metrics.clean_text",
        },
        "results": results,
        "groups": group_detail,
    }
    Path("outputs/eval_predictions/group_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved=outputs/eval_predictions/group_metrics.json")


if __name__ == "__main__":
    main()
