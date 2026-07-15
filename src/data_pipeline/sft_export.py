from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import write_jsonl


SYSTEM_PROMPT = (
    "你是一名熟悉中国上市公司年度报告和交易所信息披露监管规则的财经文本分析助手。"
    "请仅根据给定年度报告证据，生成可能被监管机构提出的一个具体问询子问题。"
)

USER_TEMPLATE = """请根据以下同公司、同年度的年度报告证据，生成一个监管问询子问题。

公司：{company}
证券代码：{stock_code}
报告年度：{report_year}

年度报告证据：
{evidence_text}

要求：
1. 问题应聚焦一个明确的信息披露风险点。
2. 使用监管问询风格，包含“说明、披露、列示、结合、是否”等要求。
3. 不要回答问题，不要编造证据之外的事实。
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_page_range(candidate: dict[str, Any]) -> str:
    start = candidate.get("page_start")
    end = candidate.get("page_end")
    if start == end:
        return f"p{start}"
    return f"p{start}-{end}"


def build_evidence_text(candidates: list[dict[str, Any]], top_k: int) -> str:
    lines: list[str] = []
    for index, candidate in enumerate(candidates[:top_k], start=1):
        lines.append(
            f"【证据{index}｜年报页 {format_page_range(candidate)}｜chunk {candidate['chunk_id']}】"
        )
        lines.append(str(candidate["text"]).strip())
    return "\n".join(lines)


def build_sft_sample(row: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    evidence_text = build_evidence_text(row.get("candidates", []), top_k)
    user_content = USER_TEMPLATE.format(
        company=row["company"],
        stock_code=row["stock_code"],
        report_year=row["report_year"],
        evidence_text=evidence_text,
    )
    used_candidates = row.get("candidates", [])[:top_k]

    return {
        "sample_id": row["question_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["cleaned_question_text"]},
        ],
        "metadata": {
            "question_id": row["question_id"],
            "company": row["company"],
            "stock_code": row["stock_code"],
            "report_year": int(row["report_year"]),
            "topic_title": row.get("topic_title", ""),
            "original_question_number": row.get("original_question_number"),
            "subquestion_number": row.get("subquestion_number"),
            "evidence_top_k": len(used_candidates),
            "evidence_chunk_ids": [candidate["chunk_id"] for candidate in used_candidates],
            "evidence_page_ranges": [
                format_page_range(candidate) for candidate in used_candidates
            ],
            "source": f"annual_report_bm25_top{len(used_candidates)}",
        },
    }


def export_sft_dataset(root: Path, top_k: int = 5, dry_run_size: int = 10) -> dict[str, Any]:
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    candidates = load_jsonl(processed_dir / "evidence_candidates.jsonl")
    selected = [
        row
        for row in candidates
        if row.get("candidate_count", 0) > 0
        and str(row.get("cleaned_question_text", "")).strip()
    ]
    samples = [build_sft_sample(row, top_k=top_k) for row in selected]
    dry_run_samples = samples[:dry_run_size]

    train_path = processed_dir / "sft_train.jsonl"
    dry_run_path = processed_dir / "sft_dry_run_10.jsonl"
    write_jsonl(train_path, samples)
    write_jsonl(dry_run_path, dry_run_samples)
    write_sft_summary(
        reports_dir / "sft_export_summary.md",
        train_path=train_path,
        dry_run_path=dry_run_path,
        samples=samples,
        top_k=top_k,
    )

    return {
        "sft_samples": len(samples),
        "dry_run_samples": len(dry_run_samples),
        "evidence_top_k": top_k,
        "train_path": str(train_path),
        "dry_run_path": str(dry_run_path),
    }


def write_sft_summary(
    path: Path,
    train_path: Path,
    dry_run_path: Path,
    samples: list[dict[str, Any]],
    top_k: int,
) -> None:
    companies = sorted({sample["metadata"]["company"] for sample in samples})
    lines = [
        "# SFT Export Summary",
        "",
        "## Outputs",
        "",
        f"- `{train_path}`",
        f"- `{dry_run_path}`",
        "",
        "## Scope",
        "",
        f"- Samples: {len(samples)}",
        f"- Evidence per sample: top-{top_k} BM25 annual-report chunks",
        "- Input evidence uses only annual report chunks from the same company and same year.",
        "- Target output is the cleaned formal regulatory subquestion.",
        "- Inquiry, reply and supporting documents are not included in the model input.",
        "",
        "## Companies",
        "",
    ]
    lines.extend(f"- {company}" for company in companies)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
