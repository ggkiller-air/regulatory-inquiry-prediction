from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import write_jsonl
from .sft_export import build_sft_sample, load_jsonl

EVIDENCE_TOP_K = 3

# 公司级划分，禁止同一公司跨集合。
# 划分依据：
# - test 保留南京新百（唯一有原始问询函金标的公司）以获得最高质量评测；
# - validation 与 test 的样本量各约占 10%-15%；
# - 其余 7 家公司进入 train。
SPLIT_ASSIGNMENT: dict[str, tuple[str, ...]] = {
    "train": (
        "600373",  # 中文传媒
        "600678",  # 四川金顶
        "600822",  # 上海物贸
        "688363",  # 华熙生物
        "688685",  # 迈信林
        "688793",  # 倍轻松
        "834033",  # 康普化学
    ),
    "validation": (
        "603466",  # 风语筑
        "603922",  # 金鸿顺
    ),
    "test": (
        "600682",  # 南京新百
        "688611",  # 杭州柯林
    ),
}


def load_training_target_ids(processed_dir: Path) -> set[str]:
    questions = load_jsonl(processed_dir / "regulatory_questions_clean.jsonl")
    return {
        row["question_id"]
        for row in questions
        if row.get("is_training_target") and not row.get("exclude_from_training")
    }


def split_for_stock_code(stock_code: str) -> str:
    for split_name, codes in SPLIT_ASSIGNMENT.items():
        if stock_code in codes:
            return split_name
    raise ValueError(f"stock code {stock_code} is not assigned to any split")


def export_sft_splits(root: Path, top_k: int = EVIDENCE_TOP_K) -> dict[str, Any]:
    processed_dir = root / "data" / "processed"
    training_dir = root / "data" / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    target_ids = load_training_target_ids(processed_dir)
    candidates = load_jsonl(processed_dir / "evidence_candidates.jsonl")
    selected = [
        row
        for row in candidates
        if row["question_id"] in target_ids
        and row.get("candidate_count", 0) > 0
        and str(row.get("cleaned_question_text", "")).strip()
    ]

    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_ASSIGNMENT}
    for row in selected:
        sample = build_sft_sample(row, top_k=top_k)
        splits[split_for_stock_code(row["stock_code"])].append(sample)

    validate_company_disjointness(splits)

    output_paths: dict[str, Path] = {}
    for split_name, samples in splits.items():
        path = training_dir / f"{split_name}.jsonl"
        write_jsonl(path, samples)
        output_paths[split_name] = path

    write_split_summary(root / "reports" / "sft_split_summary.md", splits, top_k=top_k)

    return {
        "total_samples": sum(len(samples) for samples in splits.values()),
        "evidence_top_k": top_k,
        **{f"{name}_samples": len(samples) for name, samples in splits.items()},
        **{f"{name}_path": str(path) for name, path in output_paths.items()},
    }


def validate_company_disjointness(splits: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split_name, samples in splits.items():
        for sample in samples:
            stock_code = sample["metadata"]["stock_code"]
            previous = seen.setdefault(stock_code, split_name)
            if previous != split_name:
                raise ValueError(
                    f"company {stock_code} appears in both {previous} and {split_name}"
                )


def write_split_summary(
    path: Path, splits: dict[str, list[dict[str, Any]]], top_k: int
) -> None:
    lines = [
        "# SFT Split Summary",
        "",
        f"- Evidence per sample: BM25 top-{top_k} annual-report chunks",
        "- Split unit: company (no company appears in more than one split)",
        "",
    ]
    for split_name, samples in splits.items():
        companies = sorted(
            {
                (s["metadata"]["stock_code"], s["metadata"]["company"])
                for s in samples
            }
        )
        lines.append(f"## {split_name} ({len(samples)} samples)")
        lines.append("")
        for stock_code, company in companies:
            count = sum(
                1 for s in samples if s["metadata"]["stock_code"] == stock_code
            )
            lines.append(f"- {stock_code} {company}: {count}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def company_split_table() -> dict[str, str]:
    return {
        code: split_name
        for split_name, codes in SPLIT_ASSIGNMENT.items()
        for code in codes
    }


def load_split_samples(root: Path, split_name: str) -> list[dict[str, Any]]:
    path = root / "data" / "training" / f"{split_name}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
