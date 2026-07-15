from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import DocumentRecord, DocumentType


RAW_ROOT = Path("02_raw_pdf")


def infer_report_year(filename: str) -> str:
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else "2024"


def _document_id(
    stock_code: str,
    report_year: str,
    document_type: DocumentType,
    relative_path: str,
    missing_role: str,
    supporting_subtype: str,
) -> str:
    rel_hash = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    parts = [stock_code, report_year, document_type]
    if document_type == "missing_notice" and missing_role:
        parts.append(missing_role)
    if document_type == "supporting" and supporting_subtype:
        parts.append(supporting_subtype)
    parts.append(rel_hash)
    return "_".join(part for part in parts if part)


def _classify(path: Path, raw_root: Path) -> tuple[DocumentType, str, str]:
    rel_parts = path.relative_to(raw_root).parts
    role = rel_parts[1] if len(rel_parts) > 1 else ""
    is_missing = "MISSING" in path.name.upper()

    if is_missing:
        return "missing_notice", "", role
    if role in {"annual_report", "inquiry", "reply"}:
        return role, "", role
    if role == "supporting":
        supporting_subtype = ""
        if len(rel_parts) > 3:
            supporting_subtype = rel_parts[2]
        return "supporting", supporting_subtype, role
    return "unknown", "", role


def scan_raw_documents(raw_root: Path = RAW_ROOT) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    for path in sorted(raw_root.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file():
            continue

        rel_parts = path.relative_to(raw_root).parts
        company_dir = rel_parts[0] if rel_parts else ""
        stock_code, _, company = company_dir.partition("_")
        document_type, supporting_subtype, missing_role = _classify(path, raw_root)
        relative_path = path.as_posix()
        is_missing = document_type == "missing_notice"
        report_year = infer_report_year(path.name)
        notes = ""
        if is_missing:
            notes = f"missing marker under {missing_role or 'unknown'}"

        documents.append(
            DocumentRecord(
                document_id=_document_id(
                    stock_code=stock_code,
                    report_year=report_year,
                    document_type=document_type,
                    relative_path=relative_path,
                    missing_role=missing_role,
                    supporting_subtype=supporting_subtype,
                ),
                company=company,
                stock_code=stock_code,
                report_year=report_year,
                document_type=document_type,
                supporting_subtype=supporting_subtype,
                relative_path=relative_path,
                filename=path.name,
                file_size=path.stat().st_size,
                is_missing=is_missing,
                parse_status="skipped_missing" if is_missing else "pending",
                page_count=0,
                notes=notes,
            )
        )
    return documents

