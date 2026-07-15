from __future__ import annotations

import re
from collections import Counter, defaultdict

from .models import QuestionRecord, ReviewIssue


RESPONSE_CONTAMINATION_RE = re.compile(
    r"公司回复|公司说明|会计师回复|年审会计师回复|回复如下|核查意见|我们执行了|我们认为|经核查"
)


def normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def review_questions(questions: list[QuestionRecord]) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    id_counts = Counter(question.question_id for question in questions)
    text_sources: dict[str, set[str]] = defaultdict(set)
    text_counts = Counter(normalize_question_text(question.question_text) for question in questions)

    for question in questions:
        text_key = normalize_question_text(question.question_text)
        text_sources[text_key].add(question.source_file)

    for question in questions:
        question_issues: list[str] = []
        text = question.question_text.strip()
        text_key = normalize_question_text(text)

        if not text:
            question_issues.append("empty_question")
        if id_counts[question.question_id] > 1:
            question_issues.append("duplicate_question_id")
        if text_key and text_counts[text_key] > 1:
            question_issues.append("duplicate_question_text")
        if question.question_level == "parent" and not (30 <= len(text) <= 5000):
            question_issues.append("abnormal_parent_length")
        if question.question_level == "subquestion" and not (8 <= len(text) <= 1800):
            question_issues.append("abnormal_subquestion_length")
        if RESPONSE_CONTAMINATION_RE.search(text):
            question_issues.append("possible_response_contamination")
        if question.source_page_start is None or question.source_page_end is None:
            question_issues.append("missing_source_page")
        if len(text_sources[text_key]) > 1:
            question_issues.append("same_question_from_multiple_files")
        if question.needs_review:
            question_issues.append("source_needs_manual_review")

        for issue in question_issues:
            issues.append(
                ReviewIssue(
                    question_id=question.question_id,
                    company=question.company,
                    stock_code=question.stock_code,
                    source_document_type=question.source_document_type,
                    source_file=question.source_file,
                    question_level=question.question_level,
                    source_page_start=question.source_page_start,
                    source_page_end=question.source_page_end,
                    issue_type=issue,
                    question_text=text[:1000],
                )
            )
    return issues

