from __future__ import annotations

import re
from pathlib import Path

import fitz

from .models import DocumentRecord, PageRecord, ParseFailure


NEEDS_OCR_CHAR_THRESHOLD = 40


def clean_page_text(text: str) -> str:
    text = (
        text.replace("\u3000", " ")
        .replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", line)
        line = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。；：！？、）】])", "", line)
        line = re.sub(r"(?<=[（【])\s+(?=[\u4e00-\u9fff])", "", line)
        lines.append(line)
    return "\n".join(lines)


def page_extraction_status(text: str, threshold: int = NEEDS_OCR_CHAR_THRESHOLD) -> str:
    return "needs_ocr" if len(text.strip()) < threshold else "success"


def extract_pdf_pages(
    document: DocumentRecord,
    project_root: Path,
    needs_ocr_threshold: int = NEEDS_OCR_CHAR_THRESHOLD,
) -> tuple[DocumentRecord, list[PageRecord], list[ParseFailure]]:
    path = project_root / document.relative_path
    pages: list[PageRecord] = []
    failures: list[ParseFailure] = []

    if document.is_missing:
        return document, pages, failures
    if path.suffix.lower() != ".pdf":
        updated = document.model_copy(update={"parse_status": "skipped_non_pdf", "page_count": 0})
        return updated, pages, failures

    try:
        pdf = fitz.open(path)
    except Exception as exc:
        failures.append(
            ParseFailure(
                document_id=document.document_id,
                company=document.company,
                stock_code=document.stock_code,
                document_type=document.document_type,
                source_file=document.relative_path,
                failure_type="open_failed",
                reason=str(exc),
            )
        )
        return document.model_copy(update={"parse_status": "failed", "notes": str(exc)}), pages, failures

    page_count = pdf.page_count
    needs_ocr_pages = 0
    failed_pages = 0

    for index in range(page_count):
        page_number = index + 1
        try:
            raw_text = pdf.load_page(index).get_text("text")
            cleaned = clean_page_text(raw_text)
            status = page_extraction_status(cleaned, needs_ocr_threshold)
            if status == "needs_ocr":
                needs_ocr_pages += 1
                failures.append(
                    ParseFailure(
                        document_id=document.document_id,
                        company=document.company,
                        stock_code=document.stock_code,
                        document_type=document.document_type,
                        source_file=document.relative_path,
                        page_number=page_number,
                        failure_type="needs_ocr",
                        reason=f"extracted text has fewer than {needs_ocr_threshold} characters",
                    )
                )
            pages.append(
                PageRecord(
                    document_id=document.document_id,
                    company=document.company,
                    stock_code=document.stock_code,
                    report_year=document.report_year,
                    document_type=document.document_type,
                    supporting_subtype=document.supporting_subtype,
                    source_file=document.relative_path,
                    page_number=page_number,
                    text=cleaned,
                    char_count=len(cleaned),
                    extraction_status=status,
                )
            )
        except Exception as exc:
            failed_pages += 1
            failures.append(
                ParseFailure(
                    document_id=document.document_id,
                    company=document.company,
                    stock_code=document.stock_code,
                    document_type=document.document_type,
                    source_file=document.relative_path,
                    page_number=page_number,
                    failure_type="page_extract_failed",
                    reason=str(exc),
                )
            )

    if failed_pages == page_count and page_count:
        parse_status = "failed"
    elif needs_ocr_pages == page_count and page_count:
        parse_status = "needs_ocr"
    elif failed_pages or needs_ocr_pages:
        parse_status = "partial_needs_ocr"
    else:
        parse_status = "success"

    notes = document.notes
    extras = []
    if needs_ocr_pages:
        extras.append(f"needs_ocr_pages={needs_ocr_pages}")
    if failed_pages:
        extras.append(f"failed_pages={failed_pages}")
    if extras:
        notes = "; ".join(part for part in [notes, *extras] if part)

    return document.model_copy(
        update={"parse_status": parse_status, "page_count": page_count, "notes": notes}
    ), pages, failures

