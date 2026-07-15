from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DocumentType = Literal[
    "annual_report",
    "inquiry",
    "reply",
    "supporting",
    "missing_notice",
    "unknown",
]


class DocumentRecord(BaseModel):
    document_id: str
    company: str
    stock_code: str
    report_year: str
    document_type: DocumentType
    supporting_subtype: str = ""
    relative_path: str
    filename: str
    file_size: int
    is_missing: bool
    parse_status: str = "pending"
    page_count: int = 0
    notes: str = ""


class PageRecord(BaseModel):
    document_id: str
    company: str
    stock_code: str
    report_year: str
    document_type: DocumentType
    supporting_subtype: str = ""
    source_file: str
    page_number: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)
    extraction_status: str


class QuestionRecord(BaseModel):
    question_id: str
    company: str
    stock_code: str
    report_year: str
    topic_title: str
    parent_question_text: str
    question_text: str
    question_level: Literal["parent", "subquestion"]
    source_document_type: DocumentType
    source_file: str
    source_page_start: int | None
    source_page_end: int | None
    extraction_method: str
    needs_review: bool


class ParseFailure(BaseModel):
    document_id: str
    company: str
    stock_code: str
    document_type: DocumentType
    source_file: str
    page_number: int | None = None
    failure_type: str
    reason: str


class ReviewIssue(BaseModel):
    question_id: str
    company: str
    stock_code: str
    source_document_type: DocumentType
    source_file: str
    question_level: str
    source_page_start: int | None
    source_page_end: int | None
    issue_type: str
    question_text: str

