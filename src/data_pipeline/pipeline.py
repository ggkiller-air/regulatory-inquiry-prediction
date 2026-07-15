from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .io_utils import write_csv, write_jsonl
from .manifest import scan_raw_documents
from .models import DocumentRecord, PageRecord, ParseFailure, QuestionRecord, ReviewIssue
from .pdf_extract import extract_pdf_pages
from .quality import review_questions
from .questions import extract_questions


PROJECT_ROOT = Path(".")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


MANIFEST_FIELDS = [
    "document_id",
    "company",
    "stock_code",
    "report_year",
    "document_type",
    "supporting_subtype",
    "relative_path",
    "filename",
    "file_size",
    "is_missing",
    "parse_status",
    "page_count",
    "notes",
]

PARSE_FAILURE_FIELDS = [
    "document_id",
    "company",
    "stock_code",
    "document_type",
    "source_file",
    "page_number",
    "failure_type",
    "reason",
]

REVIEW_FIELDS = [
    "question_id",
    "company",
    "stock_code",
    "source_document_type",
    "source_file",
    "question_level",
    "source_page_start",
    "source_page_end",
    "issue_type",
    "question_text",
]


def run_all(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    processed_dir = project_root / PROCESSED_DIR
    reports_dir = project_root / REPORTS_DIR

    documents = scan_raw_documents(project_root / "02_raw_pdf")
    updated_documents: list[DocumentRecord] = []
    all_pages: list[PageRecord] = []
    failures: list[ParseFailure] = []

    for document in documents:
        updated, pages, doc_failures = extract_pdf_pages(document, project_root)
        updated_documents.append(updated)
        all_pages.extend(pages)
        failures.extend(doc_failures)

    questions, question_failures = extract_questions(updated_documents, all_pages)
    failures.extend(question_failures)
    review_issues = review_questions(questions)

    write_csv(processed_dir / "manifest.csv", updated_documents, MANIFEST_FIELDS)
    write_jsonl(processed_dir / "document_pages.jsonl", all_pages)
    write_jsonl(processed_dir / "regulatory_questions.jsonl", questions)
    write_csv(reports_dir / "parse_failures.csv", failures, PARSE_FAILURE_FIELDS)
    write_csv(reports_dir / "questions_for_review.csv", review_issues, REVIEW_FIELDS)
    write_extraction_summary(
        reports_dir / "extraction_summary.md",
        documents=updated_documents,
        pages=all_pages,
        questions=questions,
        failures=failures,
        review_issues=review_issues,
    )

    return {
        "documents": updated_documents,
        "pages": all_pages,
        "questions": questions,
        "failures": failures,
        "review_issues": review_issues,
    }


def write_extraction_summary(
    path: Path,
    documents: list[DocumentRecord],
    pages: list[PageRecord],
    questions: list[QuestionRecord],
    failures: list[ParseFailure],
    review_issues: list[ReviewIssue],
) -> None:
    pdf_documents = [doc for doc in documents if doc.filename.lower().endswith(".pdf")]
    successful_pdfs = [doc for doc in pdf_documents if doc.parse_status != "failed"]
    needs_ocr_pages = [failure for failure in failures if failure.failure_type == "needs_ocr"]
    needs_ocr_docs = sorted({failure.source_file for failure in needs_ocr_pages})
    source_counts = Counter(question.source_document_type for question in questions)

    parent_by_company: dict[str, int] = defaultdict(int)
    child_by_company: dict[str, int] = defaultdict(int)
    for question in questions:
        key = f"{question.stock_code}_{question.company}"
        if question.question_level == "parent":
            parent_by_company[key] += 1
        else:
            child_by_company[key] += 1

    parse_failure_counts = Counter(failure.failure_type for failure in failures)

    lines = [
        "# Extraction Summary",
        "",
        "生成日期：2026-07-15",
        "",
        "## 输出文件",
        "",
        "- `data/processed/manifest.csv`",
        "- `data/processed/document_pages.jsonl`",
        "- `data/processed/regulatory_questions.jsonl`",
        "- `reports/questions_for_review.csv`",
        "- `reports/parse_failures.csv`",
        "",
        "## PDF 解析概况",
        "",
        f"- PDF 文件数：{len(pdf_documents)}",
        f"- 成功解析 PDF 数量：{len(successful_pdfs)}",
        f"- 逐页记录数：{len(pages)}",
        f"- needs_ocr 页面数：{len(needs_ocr_pages)}",
        f"- needs_ocr 文件数：{len(needs_ocr_docs)}",
        "",
    ]
    if needs_ocr_docs:
        lines.append("needs_ocr 文件：")
        lines.append("")
        for source_file in needs_ocr_docs:
            count = sum(1 for failure in needs_ocr_pages if failure.source_file == source_file)
            lines.append(f"- `{source_file}`：{count} 页")
        lines.append("")

    lines.extend(
        [
            "## 监管问题抽取概况",
            "",
            f"- 问题总记录数：{len(questions)}",
            f"- 一级复合问题数：{sum(1 for question in questions if question.question_level == 'parent')}",
            f"- 子问题数：{sum(1 for question in questions if question.question_level == 'subquestion')}",
            f"- 需要人工复核的问题记录数：{sum(1 for question in questions if question.needs_review)}",
            "",
            "按来源：",
            "",
        ]
    )
    for source, count in sorted(source_counts.items()):
        lines.append(f"- {source}: {count}")

    lines.extend(["", "按公司：", "", "| 公司 | 一级问题 | 子问题 |", "| --- | ---: | ---: |"])
    for company in sorted(set(parent_by_company) | set(child_by_company)):
        lines.append(f"| {company} | {parent_by_company[company]} | {child_by_company[company]} |")

    lines.extend(["", "## 质量检查", ""])
    lines.append(f"- questions_for_review 记录数：{len(review_issues)}")
    review_counts = Counter(issue.issue_type for issue in review_issues)
    for issue_type, count in sorted(review_counts.items()):
        lines.append(f"- {issue_type}: {count}")

    lines.extend(["", "## 失败与告警", ""])
    if parse_failure_counts:
        for failure_type, count in sorted(parse_failure_counts.items()):
            lines.append(f"- {failure_type}: {count}")
    else:
        lines.append("- 未记录解析失败或 OCR 告警。")

    question_failures = [failure for failure in failures if failure.failure_type == "question_extraction_empty"]
    if question_failures:
        lines.extend(["", "问题抽取为空的来源：", ""])
        for failure in question_failures:
            lines.append(f"- `{failure.source_file}`：{failure.reason}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
