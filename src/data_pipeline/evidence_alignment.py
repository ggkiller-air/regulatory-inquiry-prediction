from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import write_csv, write_jsonl


CHUNK_SIZE = 750
CHUNK_OVERLAP = 100
TOP_K = 10
REVIEW_TOP_K = 5

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
ALNUM_RE = re.compile(r"[a-zA-Z0-9]+(?:[.,][a-zA-Z0-9]+)*%?")
REQUEST_BOUNDARY_RE = re.compile(r"请(?:公司|你公司)[:：]?|请补充披露|请结合|请说明")


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    page_number: int


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_report_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def page_range_for_offsets(
    spans: list[TextSpan],
    start: int,
    end: int,
    full_text: str,
) -> tuple[int | None, int | None]:
    pages: list[int] = []
    for span in spans:
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        if overlap_start >= overlap_end:
            continue
        if full_text[overlap_start:overlap_end].strip():
            pages.append(span.page_number)
    if not pages:
        return None, None
    return min(pages), max(pages)


def build_annual_report_chunks(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    annual_pages = [page for page in pages if page.get("document_type") == "annual_report"]
    by_report: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for page in annual_pages:
        by_report[(str(page["stock_code"]), int(page["report_year"]))].append(page)

    chunks: list[dict[str, Any]] = []
    for (stock_code, report_year), report_pages in sorted(by_report.items()):
        report_pages = sorted(report_pages, key=lambda item: int(item["page_number"]))
        company = str(report_pages[0]["company"])
        source_file = str(report_pages[0]["source_file"])
        text_parts: list[str] = []
        spans: list[TextSpan] = []
        offset = 0

        for page in report_pages:
            page_text = normalize_report_text(str(page.get("text", "")))
            if not page_text:
                continue
            if text_parts:
                text_parts.append("\n")
                offset += 1
            start = offset
            text_parts.append(page_text)
            offset += len(page_text)
            spans.append(
                TextSpan(
                    start=start,
                    end=offset,
                    page_number=int(page["page_number"]),
                )
            )

        full_text = "".join(text_parts)
        if not full_text:
            continue

        start = 0
        chunk_index = 1
        step = max(1, chunk_size - overlap)
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunk_text = full_text[start:end].strip()
            if len(chunk_text) >= 50:
                page_start, page_end = page_range_for_offsets(spans, start, end, full_text)
                chunks.append(
                    {
                        "chunk_id": f"{stock_code}_{report_year}_ar_chunk_{chunk_index:04d}",
                        "company": company,
                        "stock_code": stock_code,
                        "report_year": report_year,
                        "source_file": source_file,
                        "page_start": page_start,
                        "page_end": page_end,
                        "text": chunk_text,
                    }
                )
                chunk_index += 1
            if end == len(full_text):
                break
            start += step

    return chunks


def tokenize(text: str) -> list[str]:
    text = normalize_report_text(text).lower()
    tokens: list[str] = []
    for match in CJK_RE.finditer(text):
        sequence = match.group(0)
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    for match in ALNUM_RE.finditer(text):
        token = match.group(0).replace(",", "")
        tokens.append(token)
        if token.endswith("%"):
            tokens.append(token[:-1])
    return [token for token in tokens if token]


class BM25Index:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for document in documents:
            term_freq = Counter(tokenize(str(document.get("text", ""))))
            self.term_freqs.append(term_freq)
            doc_length = sum(term_freq.values())
            self.doc_lengths.append(doc_length)
            document_frequency.update(term_freq.keys())

        self.document_count = len(documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.document_count if self.document_count else 0.0
        )
        self.idf = {
            term: math.log(1 + (self.document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def top_n(self, query: str, n: int = TOP_K) -> list[tuple[dict[str, Any], float]]:
        if not self.documents:
            return []

        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []

        scores: list[tuple[int, float]] = []
        k1 = 1.5
        b = 0.75
        avgdl = self.avg_doc_length or 1.0

        for doc_index, term_freq in enumerate(self.term_freqs):
            doc_length = self.doc_lengths[doc_index] or 1
            score = 0.0
            for term, query_freq in query_terms.items():
                freq = term_freq.get(term, 0)
                if not freq:
                    continue
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_length / avgdl)
                query_weight = 1 + math.log(query_freq)
                score += self.idf.get(term, 0.0) * numerator / denominator * query_weight
            scores.append((doc_index, score))

        scores.sort(key=lambda item: (-item[1], self.documents[item[0]]["chunk_id"]))
        return [(self.documents[index], score) for index, score in scores[:n]]


def select_training_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("question_level") == "subquestion"
        and row.get("is_training_target") is True
        and row.get("exclude_from_training") is False
    ]


def parent_background(parent_question_text: str) -> str:
    match = REQUEST_BOUNDARY_RE.search(parent_question_text)
    if not match:
        return parent_question_text
    return parent_question_text[: match.start()]


def build_query(question: dict[str, Any]) -> str:
    parts = [
        str(question.get("topic_title", "")),
        parent_background(str(question.get("parent_question_text", ""))),
        str(question.get("cleaned_question_text", "")),
    ]
    return normalize_report_text("。".join(part for part in parts if part))


def build_indexes(chunks: list[dict[str, Any]]) -> dict[tuple[str, int], BM25Index]:
    by_report: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_report[(str(chunk["stock_code"]), int(chunk["report_year"]))].append(chunk)
    return {key: BM25Index(value) for key, value in by_report.items()}


def retrieve_candidates(
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexes = build_indexes(chunks)
    rows: list[dict[str, Any]] = []

    for question in questions:
        report_key = (str(question["stock_code"]), int(question["report_year"]))
        query = build_query(question)
        index = indexes.get(report_key)
        candidates: list[dict[str, Any]] = []

        if index is not None:
            for rank, (chunk, score) in enumerate(index.top_n(query, TOP_K), start=1):
                candidates.append(
                    {
                        "rank": rank,
                        "chunk_id": chunk["chunk_id"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "score": round(score, 6),
                        "text": chunk["text"],
                    }
                )

        rows.append(
            {
                "question_id": question["question_id"],
                "company": question["company"],
                "stock_code": str(question["stock_code"]),
                "report_year": int(question["report_year"]),
                "topic_title": question.get("topic_title", ""),
                "original_question_number": question.get("original_question_number"),
                "subquestion_number": question.get("subquestion_number"),
                "cleaned_question_text": question.get("cleaned_question_text", ""),
                "query": query,
                "candidate_count": len(candidates),
                "max_score": candidates[0]["score"] if candidates else 0.0,
                "candidates": candidates,
            }
        )

    return rows


def ranges_intersect(
    left_start: int | None,
    left_end: int | None,
    right_start: int | None,
    right_end: int | None,
) -> bool:
    if None in {left_start, left_end, right_start, right_end}:
        return False
    return int(left_start) <= int(right_end) and int(right_start) <= int(left_end)


def load_alignment_gold(workbook_path: Path) -> list[dict[str, Any]]:
    df = pd.read_excel(workbook_path, sheet_name="03_年报对齐")
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sample_id = str(row.get("sample_id", "")).strip()
        match = re.search(r"_Q(\d+)", sample_id)
        if not match:
            continue
        if pd.isna(row.get("证据起始页")) or pd.isna(row.get("证据结束页")):
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "company": str(row.get("公司", "")).strip(),
                "topic_title": str(row.get("问题标题", "")).strip(),
                "question_number": int(match.group(1)),
                "gold_page_start": int(row["证据起始页"]),
                "gold_page_end": int(row["证据结束页"]),
                "gold_text": str(row.get("年报证据文本（模型输入）", "")).strip(),
            }
        )
    return rows


def evaluate_retrieval(
    candidates: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    workbook_path: Path,
) -> dict[str, Any]:
    gold_rows = load_alignment_gold(workbook_path)
    candidates_by_question = {row["question_id"]: row for row in candidates}
    questions_by_parent: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_parent[
            (str(question["company"]), int(question["original_question_number"]))
        ].append(question)

    validation_rows: list[dict[str, Any]] = []
    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0}

    for gold in gold_rows:
        mapped_questions = questions_by_parent.get((gold["company"], gold["question_number"]), [])
        mapped_ids = [question["question_id"] for question in mapped_questions]
        row_hits: dict[int, bool] = {}

        for k in hit_counts:
            hit = False
            for question_id in mapped_ids:
                question_candidates = candidates_by_question.get(question_id, {}).get(
                    "candidates",
                    [],
                )
                for candidate in question_candidates[:k]:
                    if ranges_intersect(
                        candidate.get("page_start"),
                        candidate.get("page_end"),
                        gold["gold_page_start"],
                        gold["gold_page_end"],
                    ):
                        hit = True
                        break
                if hit:
                    break
            row_hits[k] = hit
            if hit:
                hit_counts[k] += 1

        validation_rows.append(
            {
                **gold,
                "mapped_question_count": len(mapped_ids),
                "mapped_question_ids": ";".join(mapped_ids),
                "hit_at_1": row_hits[1],
                "hit_at_3": row_hits[3],
                "hit_at_5": row_hits[5],
                "hit_at_10": row_hits[10],
            }
        )

    denominator = len(validation_rows) or 1
    metrics = {f"recall_at_{k}": hit_counts[k] / denominator for k in hit_counts}
    return {
        "gold_rows": validation_rows,
        "metrics": metrics,
        "validation_count": len(validation_rows),
        "mapped_validation_count": sum(
            1 for row in validation_rows if row["mapped_question_count"] > 0
        ),
    }


def validation_lookup(validation_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["company"]), int(row["question_number"])): row
        for row in validation_rows
    }


def write_evidence_review(
    path: Path,
    candidates: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> None:
    lookup = validation_lookup(evaluation["gold_rows"])
    review_rows: list[dict[str, Any]] = []
    for row in candidates:
        gold = lookup.get((str(row["company"]), int(row["original_question_number"])))
        for candidate in row["candidates"][:REVIEW_TOP_K]:
            review_rows.append(
                {
                    "question_id": row["question_id"],
                    "company": row["company"],
                    "stock_code": row["stock_code"],
                    "report_year": row["report_year"],
                    "topic_title": row["topic_title"],
                    "original_question_number": row["original_question_number"],
                    "subquestion_number": row["subquestion_number"],
                    "cleaned_question_text": row["cleaned_question_text"],
                    "rank": candidate["rank"],
                    "页码": f"{candidate['page_start']}-{candidate['page_end']}",
                    "证据文本": candidate["text"],
                    "BM25分数": candidate["score"],
                    "chunk_id": candidate["chunk_id"],
                    "validation_sample_id": gold["sample_id"] if gold else "",
                    "gold_pages": (
                        f"{gold['gold_page_start']}-{gold['gold_page_end']}" if gold else ""
                    ),
                    "hit_gold_range": (
                        ranges_intersect(
                            candidate["page_start"],
                            candidate["page_end"],
                            gold["gold_page_start"],
                            gold["gold_page_end"],
                        )
                        if gold
                        else ""
                    ),
                    "人工确认": "",
                }
            )

    metrics_rows = [
        {"metric": key, "value": value}
        for key, value in sorted(evaluation["metrics"].items())
    ]
    validation_rows = evaluation["gold_rows"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(review_rows).to_excel(writer, index=False, sheet_name="top5_candidates")
        pd.DataFrame(metrics_rows).to_excel(writer, index=False, sheet_name="metrics")
        pd.DataFrame(validation_rows).to_excel(writer, index=False, sheet_name="validation_gold")


def build_ocr_priority_rows(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        if page.get("document_type") != "annual_report":
            continue
        status = str(page.get("extraction_status", ""))
        char_count = int(page.get("char_count", 0))
        if status != "needs_ocr" and char_count >= 50:
            continue
        rows.append(
            {
                "company": page.get("company", ""),
                "stock_code": page.get("stock_code", ""),
                "report_year": page.get("report_year", ""),
                "source_file": page.get("source_file", ""),
                "page_number": page.get("page_number", ""),
                "char_count": char_count,
                "extraction_status": status,
                "priority_reason": "needs_ocr" if status == "needs_ocr" else "sparse_text",
            }
        )
    rows.sort(key=lambda item: (str(item["stock_code"]), int(item["page_number"])))
    return rows


def write_retrieval_summary(
    path: Path,
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    evaluation: dict[str, Any],
    ocr_rows: list[dict[str, Any]],
) -> None:
    no_valid = [row for row in candidates if not row["candidates"] or row["max_score"] <= 0]
    chunk_counts = Counter((chunk["stock_code"], chunk["company"]) for chunk in chunks)
    ocr_counts = Counter((row["stock_code"], row["company"]) for row in ocr_rows)
    metrics = evaluation["metrics"]

    lines = [
        "# Retrieval Summary",
        "",
        "## Outputs",
        "",
        "- `data/processed/annual_report_chunks.jsonl`",
        "- `data/processed/evidence_candidates.jsonl`",
        "- `reports/evidence_review.xlsx`",
        "- `reports/ocr_priority_pages.csv`",
        "",
        "## Scope",
        "",
        "- Queries use only formal clean subquestions with `is_training_target=true` and `exclude_from_training=false`.",
        "- Evidence corpus uses only same-company, same-year `annual_report` pages.",
        "- Inquiry, reply and supporting documents are not indexed as evidence.",
        "",
        "## Counts",
        "",
        f"- Processed training subquestions: {len(questions)}",
        f"- Annual report chunks: {len(chunks)}",
        f"- Evidence candidate rows: {len(candidates)}",
        f"- Questions with no effective candidate score: {len(no_valid)}",
        f"- OCR priority annual-report pages: {len(ocr_rows)}",
        "",
        "## Retrieval Validation",
        "",
        f"- Human-aligned parent records: {evaluation['validation_count']}",
        f"- Mapped validation records: {evaluation['mapped_validation_count']}",
        f"- Recall@1: {metrics['recall_at_1']:.4f}",
        f"- Recall@3: {metrics['recall_at_3']:.4f}",
        f"- Recall@5: {metrics['recall_at_5']:.4f}",
        f"- Recall@10: {metrics['recall_at_10']:.4f}",
        "",
        "Validation note: the Excel workbook contains 14 parent-level annual-report alignments. "
        "Each row is mapped by company and original question number to the corresponding formal "
        "subquestions; a hit is counted when any mapped subquestion candidate page range intersects "
        "the gold page range.",
        "",
        "## Chunk Counts By Company",
        "",
        "| Stock code | Company | Chunks |",
        "| --- | --- | ---: |",
    ]
    for (stock_code, company), count in sorted(chunk_counts.items()):
        lines.append(f"| {stock_code} | {company} | {count} |")

    lines.extend(["", "## No Effective Candidates", ""])
    if no_valid:
        for row in no_valid:
            lines.append(f"- `{row['question_id']}` {row['company']} {row['topic_title']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## OCR Priority Pages", ""])
    if ocr_counts:
        lines.extend(["| Stock code | Company | Pages |", "| --- | --- | ---: |"])
        for (stock_code, company), count in sorted(ocr_counts.items()):
            lines.append(f"| {stock_code} | {company} | {count} |")
    else:
        lines.append("- None.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evidence_alignment(root: Path) -> dict[str, Any]:
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    workbook_path = root / "01_metadata" / "监管问询预测_项目工作簿_年报对齐完成.xlsx"

    pages = load_jsonl(processed_dir / "document_pages.jsonl")
    question_rows = load_jsonl(processed_dir / "regulatory_questions_clean.jsonl")
    questions = select_training_questions(question_rows)
    chunks = build_annual_report_chunks(pages)
    candidates = retrieve_candidates(questions, chunks)
    evaluation = evaluate_retrieval(candidates, questions, workbook_path)
    ocr_rows = build_ocr_priority_rows(pages)

    write_jsonl(processed_dir / "annual_report_chunks.jsonl", chunks)
    write_jsonl(processed_dir / "evidence_candidates.jsonl", candidates)
    write_evidence_review(reports_dir / "evidence_review.xlsx", candidates, evaluation)
    write_csv(
        reports_dir / "ocr_priority_pages.csv",
        ocr_rows,
        [
            "company",
            "stock_code",
            "report_year",
            "source_file",
            "page_number",
            "char_count",
            "extraction_status",
            "priority_reason",
        ],
    )
    write_retrieval_summary(
        reports_dir / "retrieval_summary.md",
        questions,
        chunks,
        candidates,
        evaluation,
        ocr_rows,
    )

    no_valid = [row for row in candidates if not row["candidates"] or row["max_score"] <= 0]
    return {
        "training_subquestions": len(questions),
        "annual_report_chunks": len(chunks),
        "candidate_rows": len(candidates),
        "recall_at_1": evaluation["metrics"]["recall_at_1"],
        "recall_at_3": evaluation["metrics"]["recall_at_3"],
        "recall_at_5": evaluation["metrics"]["recall_at_5"],
        "recall_at_10": evaluation["metrics"]["recall_at_10"],
        "no_effective_candidates": len(no_valid),
        "ocr_priority_pages": len(ocr_rows),
    }
