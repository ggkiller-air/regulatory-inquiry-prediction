"""汇总自动指标到主表/消融表 CSV+MD，并做一致性检查。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PENDING = "待人工复核"

MAIN_ROWS = [
    ("Qwen3-8B Zero-shot", "zero_shot"),
    ("Qwen3-8B + Evidence", "base_evidence"),
    ("Qwen3-8B QLoRA", "qlora_no_evidence"),
    ("Full Model（本文）", "full_model"),
]

ABLATION_ROWS = [
    ("Full Model", "full_model"),
    ("w/o Evidence", "qlora_no_evidence"),
    ("w/o Company-Year Constraint", "wo_company_year"),
    ("w/o QLoRA", "base_evidence"),
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def consistency_checks(pred_dir: Path, test_path: Path) -> list[str]:
    notes = []
    test = load_jsonl(test_path)
    test_ids = [r["sample_id"] for r in test]
    gold = {r["sample_id"]: r["messages"][-1]["content"] for r in test}
    configs = ["zero_shot", "base_evidence", "qlora_no_evidence", "full_model", "wo_company_year"]
    inputs = {
        "zero_shot": Path("outputs/eval_inputs/test_no_evidence.jsonl"),
        "base_evidence": Path("data/training_leakage_free/test.jsonl"),
        "qlora_no_evidence": Path("outputs/eval_inputs/test_no_evidence.jsonl"),
        "full_model": Path("data/training_leakage_free/test.jsonl"),
        "wo_company_year": Path("outputs/eval_inputs/test_global_evidence.jsonl"),
    }
    for cfg in configs:
        rows = load_jsonl(pred_dir / f"{cfg}.jsonl")
        ids = [r["sample_id"] for r in rows]
        assert len(ids) == len(set(ids)), f"{cfg}: duplicate sample_ids"
        assert ids == test_ids, f"{cfg}: sample_id set/order differs from test set"
        for r in rows:
            assert r["reference"] == gold[r["sample_id"]], f"{cfg}: reference mismatch {r['sample_id']}"
        # 参考问询不得出现在模型输入中
        inp = {s["sample_id"]: s["messages"][1]["content"] for s in load_jsonl(inputs[cfg])}
        for sid, g in gold.items():
            windows = [g[i : i + 15] for i in range(0, max(1, len(g) - 15), 5)]
            assert not any(w in inp[sid] for w in windows), f"{cfg}: gold leaked into input {sid}"
        notes.append(f"{cfg}: n={len(ids)}, ids match test set, references consistent, no gold text in input")
    return notes


FOOTNOTE = (
    "\n评测协议：按相同模型输入将 23 条测试子问题合并为 7 个输入组（multi-reference），"
    "每组取预测与组内各参考问询的最高 ROUGE-L / BERTScore-F1，再对 7 组宏平均。\n"
    "关键问点F1、证据支持率、幻觉率待人工复核完成后填写"
    "（复核表：reports/key_point_annotation.xlsx、reports/factuality_annotation.xlsx）。\n"
)


def write_table(rows, metrics_by_config, out_csv: Path, out_md: Path, first_col: str) -> None:
    header = [first_col, "ROUGE-L", "BERTScore-F1", "关键问点F1", "证据支持率", "幻觉率"]
    csv_rows, md_lines = [], []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] + ["---:"] * 5) + " |")
    for label, cfg in rows:
        m = metrics_by_config[cfg]
        r = f"{m['rouge_l_f1_macro']:.4f}"
        b = f"{m['bertscore_f1_macro']:.4f}"
        csv_rows.append([label, r, b, PENDING, PENDING, PENDING])
        md_lines.append(f"| {label} | {r} | {b} | {PENDING} | {PENDING} | {PENDING} |")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(csv_rows)
    out_md.write_text("\n".join(md_lines) + "\n" + FOOTNOTE, encoding="utf-8")


def main() -> None:
    pred_dir = Path("outputs/eval_predictions")
    data = json.loads(Path("outputs/eval_predictions/group_metrics.json").read_text(encoding="utf-8"))
    metrics = {r["config"]: r for r in data["results"]}

    notes = consistency_checks(pred_dir, Path("data/training_leakage_free/test.jsonl"))
    for n in notes:
        print("CHECK:", n)

    reports = Path("reports")
    write_table(MAIN_ROWS, metrics, reports / "main_results.csv", reports / "main_results.md", "方法")
    write_table(
        ABLATION_ROWS, metrics, reports / "ablation_results.csv", reports / "ablation_results.md", "模型变体"
    )
    Path("reports/consistency_checks.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("tables written")


if __name__ == "__main__":
    main()
