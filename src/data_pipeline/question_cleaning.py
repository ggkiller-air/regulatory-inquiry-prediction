from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import write_csv, write_jsonl
from .models import PageRecord
from .questions import SourceText, build_source_text, compact_text


NANJING_STOCK_CODE = "600682"
SUPPORTING_STOCK_CODES = {"603629", "835174"}

REQUEST_TRIGGER_RE = re.compile(
    r"请(?:公司|你公司)[:：]?|请补充披露|请结合|请说明"
)
REQUEST_VERB_RE = re.compile(
    r"说明|披露|列示|分析|核实|解释|测算|补充|结合|是否|原因|合理性|匹配"
)
SUBQUESTION_MARKER_RE = re.compile(
    r"[（(]\s*(?P<label>\d+|[一二三四五六七八九十]+)\s*[）)]"
)
QUESTION_REFERENCE_RE = re.compile(
    r"问题(?:\s*[（(]\s*(?:\d+|[一二三四五六七八九十]+)\s*[）)])+"
)
OMISSION_RE = re.compile(
    r"……|…|省略|从略|[（(]\s*略\s*[）)]|略[。；;，,]|\.{3,}|_{3,}|待补充|缺失"
)
ANSWER_CONTAMINATION_RE = re.compile(
    r"公司回复|回复如下|公司说明|会计师回复|核查意见|经核查"
)
FOOTER_BOUNDARY_RE = re.compile(
    r"针对前述问题|请你公司收到本函件后|特此公告|董事会\s*\d{4}\s*年"
)
SECTION_TITLE_RE = re.compile(r"([一二三四五六七八九十]+、\s*关于[^\n。；：:]{1,80})")
QUESTION_NUMBER_RE = re.compile(
    r"^(?:问题\s*)?(?P<num>\d+|[一二三四五六七八九十]+)[、.．]"
)

PROFESSIONAL_PATTERNS = [
    re.compile(
        r"请(?:年审)?会计师[^。；;]*?"
        r"(?:发表(?:明确)?意见|进行核查并发表(?:明确)?意见|核查并发表(?:明确)?意见|"
        r"说明[^。；;]*?(?:审计程序|审计过程)[^。；;]*)"
        r"(?:[。；;])?"
    ),
    re.compile(
        r"请独立董事[^。；;]*?(?:发表(?:明确)?意见|进行核查[^。；;]*|核查[^。；;]*)"
        r"(?:[。；;])?"
    ),
    re.compile(
        r"请(?:保荐机构|券商|主办券商|保荐人)[^。；;]*?"
        r"(?:发表(?:明确)?意见|进行核查[^。；;]*|核查[^。；;]*|说明[^。；;]*)"
        r"(?:[。；;])?"
    ),
    re.compile(r"[（(]\s*问询函第[一二三四五六七八九十\d]+条\s*[）)]"),
]


@dataclass(frozen=True)
class DerivedSubquestion:
    raw_text: str
    label: str
    method: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_raw_backup(processed_dir: Path) -> Path:
    source = processed_dir / "regulatory_questions.jsonl"
    target = processed_dir / "regulatory_questions_raw.jsonl"
    if not target.exists():
        shutil.copy2(source, target)
    return target


def merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    merged: list[tuple[int, int, str]] = []
    for start, end, value in sorted(spans, key=lambda item: (item[0], item[1])):
        if not merged or start >= merged[-1][1]:
            merged.append((start, end, value))
            continue
        prev_start, prev_end, prev_value = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end), f"{prev_value}；{value}")
    return merged


def professional_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for pattern in PROFESSIONAL_PATTERNS:
        for match in pattern.finditer(text):
            value = compact_text(match.group(0)).strip("。；; ")
            if value:
                spans.append((match.start(), match.end(), value))
    return merge_spans(spans)


def in_spans(offset: int, spans: list[tuple[int, int, str]]) -> bool:
    return any(start <= offset < end for start, end, _value in spans)


def strip_professional_requests(text: str) -> tuple[str, str]:
    spans = professional_spans(text)
    if not spans:
        return compact_text(text), ""

    pieces: list[str] = []
    requests: list[str] = []
    cursor = 0
    for start, end, value in spans:
        pieces.append(text[cursor:start])
        requests.append(value)
        cursor = end
    pieces.append(text[cursor:])

    cleaned = compact_text("".join(pieces))
    cleaned = re.sub(r"[。；;]{2,}", "。", cleaned)
    cleaned = cleaned.strip(" ，,。；;")

    unique_requests = list(dict.fromkeys(requests))
    return cleaned, "；".join(unique_requests)


def strip_leading_requester(text: str) -> str:
    text = re.sub(r"^请(?:公司|你公司)[:：]?", "", text).strip()
    text = re.sub(r"^请(?=补充披露|结合|说明|列示|分析|核实)", "", text).strip()
    return text.strip(" ，,。；;")


def is_meaningful_request(text: str) -> bool:
    cleaned, _request = strip_professional_requests(text)
    cleaned = strip_leading_requester(cleaned)
    return len(cleaned) >= 8 and bool(REQUEST_VERB_RE.search(cleaned))


def subquestion_markers(text: str) -> list[re.Match[str]]:
    trigger = REQUEST_TRIGGER_RE.search(text)
    search_start = trigger.start() if trigger else 0
    professional = professional_spans(text)
    references = [(match.start(), match.end(), "") for match in QUESTION_REFERENCE_RE.finditer(text)]

    return [
        match
        for match in SUBQUESTION_MARKER_RE.finditer(text, search_start)
        if not in_spans(match.start(), professional)
        and not in_spans(match.start(), references)
    ]


def split_unlabeled_requests(text: str) -> list[DerivedSubquestion]:
    trigger = REQUEST_TRIGGER_RE.search(text)
    if not trigger:
        return []

    segment = compact_text(text[trigger.start() :])
    if not is_meaningful_request(segment):
        return []

    parts = [compact_text(part) for part in re.split(r"[；;]", segment)]
    parts = [part for part in parts if part]
    if len(parts) > 1 and all(is_meaningful_request(part) for part in parts):
        return [
            DerivedSubquestion(raw_text=part, label=str(index), method="semicolon_request_split")
            for index, part in enumerate(parts, start=1)
        ]

    return [
        DerivedSubquestion(
            raw_text=segment,
            label="1",
            method="unlabeled_request_fallback",
        )
    ]


def derive_subquestions(parent_text: str) -> list[DerivedSubquestion]:
    text = compact_text(parent_text)
    markers = subquestion_markers(text)
    if not markers:
        return split_unlabeled_requests(text)

    children: list[DerivedSubquestion] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        raw_child = compact_text(text[start:end])
        if not raw_child:
            continue
        children.append(
            DerivedSubquestion(
                raw_text=raw_child,
                label=marker.group("label"),
                method="numbered_subquestion_split",
            )
        )
    return children


def expected_subquestion_count(parent_text: str) -> int:
    markers = subquestion_markers(compact_text(parent_text))
    if markers:
        return len(markers)
    return 1 if split_unlabeled_requests(parent_text) else 0


def parse_report_year(value: Any) -> int:
    match = re.search(r"\d{4}", str(value))
    if not match:
        return 0
    return int(match.group(0))


def chinese_number_to_int(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    mapping = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if value in mapping:
        return mapping[value]
    if value.startswith("十") and len(value) == 2:
        return 10 + mapping.get(value[1], 0)
    if value.endswith("十") and len(value) == 2:
        return mapping.get(value[0], 0) * 10
    if "十" in value:
        left, right = value.split("十", 1)
        return mapping.get(left, 1) * 10 + mapping.get(right, 0)
    return None


def parse_original_question_number(row: dict[str, Any], fallback: int) -> int:
    text = compact_text(str(row.get("parent_question_text") or row.get("question_text") or ""))
    match = QUESTION_NUMBER_RE.search(text)
    if match:
        parsed = chinese_number_to_int(match.group("num"))
        if parsed is not None:
            return parsed
    match = re.search(r"_q(\d+)", str(row.get("question_id", "")))
    if match:
        return int(match.group(1))
    return fallback


def clean_question_fields(
    row: dict[str, Any],
    raw_text: str,
    question_level: str,
    original_question_number: int,
    subquestion_number: int | None,
    force_exclude: bool,
) -> dict[str, Any]:
    raw_text = compact_text(raw_text)
    cleaned_text, professional_request = strip_professional_requests(raw_text)
    if question_level == "subquestion":
        cleaned_text = strip_leading_requester(cleaned_text)

    is_complete = bool(cleaned_text) and not OMISSION_RE.search(cleaned_text)
    exclude = question_level == "parent" or force_exclude or not is_complete

    cleaned = dict(row)
    cleaned["report_year"] = parse_report_year(row.get("report_year"))
    cleaned["question_level"] = question_level
    cleaned["raw_question_text"] = raw_text
    cleaned["cleaned_question_text"] = cleaned_text
    cleaned["question_text"] = cleaned_text
    cleaned["professional_opinion_request"] = professional_request
    cleaned["is_complete"] = is_complete
    cleaned["exclude_from_training"] = exclude
    cleaned["is_training_target"] = question_level == "subquestion" and not exclude
    cleaned["original_question_number"] = original_question_number
    cleaned["subquestion_number"] = subquestion_number
    return cleaned


def process_parent_group(
    parent: dict[str, Any],
    existing_children: list[dict[str, Any]],
    parent_sequence: int,
    force_exclude: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    parent_text = compact_text(str(parent.get("question_text") or ""))
    original_number = parse_original_question_number(parent, parent_sequence)

    parent_row = clean_question_fields(
        parent,
        parent_text,
        "parent",
        original_number,
        None,
        force_exclude,
    )
    rows.append(parent_row)

    derived_children = derive_subquestions(parent_text)
    marker_count = len(subquestion_markers(parent_text))
    expected_count: int | str = marker_count if marker_count else ">=1"
    status = "pass"
    if marker_count and len(derived_children) != marker_count:
        status = "fail"
    elif not marker_count and not derived_children:
        expected_count = 0
        status = "no_request_detected"

    validation.append(
        {
            "company": parent.get("company", ""),
            "stock_code": parent.get("stock_code", ""),
            "question_id": parent.get("question_id", ""),
            "source_document_type": parent.get("source_document_type", ""),
            "validation_type": "subquestion_count_alignment",
            "expected_subquestions": expected_count,
            "actual_subquestions": len(derived_children),
            "status": status,
            "details": "",
        }
    )

    for child_index, child in enumerate(derived_children, start=1):
        existing = existing_children[child_index - 1] if child_index <= len(existing_children) else {}
        child_row = dict(parent)
        child_row.update(existing)
        child_row["question_id"] = f"{parent['question_id']}_sub{child_index:02d}"
        child_row["question_level"] = "subquestion"
        child_row["parent_question_text"] = parent_text
        child_row["source_page_start"] = existing.get(
            "source_page_start", parent.get("source_page_start")
        )
        child_row["source_page_end"] = existing.get(
            "source_page_end", parent.get("source_page_end")
        )
        child_row["extraction_method"] = f"clean_{child.method}"

        rows.append(
            clean_question_fields(
                child_row,
                child.raw_text,
                "subquestion",
                original_number,
                child_index,
                force_exclude,
            )
        )

    if not existing_children and derived_children:
        validation.append(
            {
                "company": parent.get("company", ""),
                "stock_code": parent.get("stock_code", ""),
                "question_id": parent.get("question_id", ""),
                "source_document_type": parent.get("source_document_type", ""),
                "validation_type": "generated_missing_subquestions",
                "expected_subquestions": expected_count,
                "actual_subquestions": len(derived_children),
                "status": "generated",
                "details": "parent had no raw child records; generated from parent request text",
            }
        )

    return rows, validation


def infer_topic_title(source: SourceText, start: int, block: str) -> str:
    own_topic = re.search(r"^\d+\.\s*(关于[^。；：:]{1,60})", block)
    if own_topic:
        return own_topic.group(1).strip()

    prior_sections = [
        match.group(1)
        for match in SECTION_TITLE_RE.finditer(source.text)
        if match.start() < start
    ]
    if prior_sections:
        return compact_text(prior_sections[-1]).split("、", 1)[-1]
    return ""


def content_page_range(source: SourceText, start: int, end: int) -> tuple[int | None, int | None]:
    pages: list[int] = []
    for span in source.spans:
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        if overlap_start >= overlap_end:
            continue
        segment = source.text[overlap_start:overlap_end]
        if compact_text(segment):
            pages.append(span.page_number)

    if pages:
        return min(pages), max(pages)
    return source.page_range(start, max(start, end - 1))


def reparse_nanjing_inquiry(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    page_models = [
        PageRecord.model_validate(page)
        for page in pages
        if page.get("stock_code") == NANJING_STOCK_CODE
        and page.get("document_type") == "inquiry"
    ]
    if not page_models:
        return []

    source = build_source_text(page_models)
    matches = list(re.finditer(r"(?m)^\s*(?P<num>[1-7])\.", source.text))
    rows: list[dict[str, Any]] = []
    first_page = page_models[0]

    for index, match in enumerate(matches):
        number = int(match.group("num"))
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source.text)
        next_section = SECTION_TITLE_RE.search(source.text, match.end(), end)
        if next_section:
            end = next_section.start()
        footer = FOOTER_BOUNDARY_RE.search(source.text, match.end(), end)
        if footer:
            end = footer.start()

        block = compact_text(source.text[start:end])
        block = FOOTER_BOUNDARY_RE.split(block)[0].strip(" ，,。；;")
        page_start, page_end = content_page_range(source, start, end)
        source_file = first_page.source_file

        rows.append(
            {
                "question_id": f"{NANJING_STOCK_CODE}_{parse_report_year(first_page.report_year)}_q{number:02d}",
                "company": first_page.company,
                "stock_code": first_page.stock_code,
                "report_year": parse_report_year(first_page.report_year),
                "topic_title": infer_topic_title(source, start, block),
                "parent_question_text": block,
                "question_text": block,
                "question_level": "parent",
                "source_document_type": "inquiry",
                "source_file": source_file,
                "source_page_start": page_start,
                "source_page_end": page_end,
                "extraction_method": "nanjing_inquiry_reparse",
                "needs_review": False,
            }
        )

    return rows


def group_children(raw_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("question_level") != "subquestion":
            continue
        parent_id = str(row.get("question_id", "")).rsplit("_sub", 1)[0]
        children[parent_id].append(row)
    return children


def build_clean_rows(
    raw_rows: list[dict[str, Any]],
    pages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    children_by_parent = group_children(raw_rows)
    clean_rows: list[dict[str, Any]] = []
    auxiliary_rows: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    inserted_nanjing = False
    parent_sequence_by_company: Counter[str] = Counter()

    def append_group(parent: dict[str, Any], existing_children: list[dict[str, Any]]) -> None:
        stock_code = str(parent.get("stock_code", ""))
        parent_sequence_by_company[stock_code] += 1
        force_exclude = parent.get("source_document_type") == "supporting"
        rows, checks = process_parent_group(
            parent,
            existing_children,
            parent_sequence_by_company[stock_code],
            force_exclude,
        )
        validation.extend(checks)
        if force_exclude:
            auxiliary_rows.extend(rows)
            validation.append(
                {
                    "company": parent.get("company", ""),
                    "stock_code": stock_code,
                    "question_id": parent.get("question_id", ""),
                    "source_document_type": parent.get("source_document_type", ""),
                    "validation_type": "supporting_moved_to_auxiliary",
                    "expected_subquestions": "",
                    "actual_subquestions": "",
                    "status": "moved",
                    "details": "supporting-source questions excluded from formal gold data",
                }
            )
        else:
            clean_rows.extend(rows)

    for row in raw_rows:
        stock_code = str(row.get("stock_code", ""))
        if stock_code == NANJING_STOCK_CODE:
            if not inserted_nanjing:
                for nanjing_parent in reparse_nanjing_inquiry(pages):
                    append_group(nanjing_parent, [])
                inserted_nanjing = True
            continue

        if row.get("question_level") != "parent":
            continue
        append_group(row, children_by_parent.get(str(row.get("question_id")), []))

    validation.extend(validate_records(clean_rows, "formal"))
    validation.extend(validate_records(auxiliary_rows, "auxiliary"))
    return clean_rows, auxiliary_rows, validation


def validate_records(rows: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    validation: list[dict[str, Any]] = []
    id_counts = Counter(str(row.get("question_id", "")) for row in rows)
    text_sources: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        cleaned_text = str(row.get("cleaned_question_text", ""))
        question_id = str(row.get("question_id", ""))
        text_key = (str(row.get("stock_code", "")), cleaned_text)
        if cleaned_text:
            text_sources[text_key].add(str(row.get("source_file", "")))

        checks = [
            (not cleaned_text, "empty_question", "cleaned question text is empty"),
            (
                row.get("question_level") == "subquestion" and len(cleaned_text) < 8,
                "question_too_short",
                "subquestion text shorter than 8 chars",
            ),
            (
                len(cleaned_text) > 1800,
                "question_too_long",
                "cleaned question text longer than 1800 chars",
            ),
            (
                bool(ANSWER_CONTAMINATION_RE.search(cleaned_text)),
                "suspected_answer_contamination",
                "cleaned text still contains answer-like phrase",
            ),
            (
                row.get("source_page_start") in ("", None)
                or row.get("source_page_end") in ("", None),
                "missing_source_page",
                "source page start/end is missing",
            ),
            (
                id_counts[question_id] > 1,
                "duplicate_question_id",
                "question_id appears more than once",
            ),
            (
                not row.get("is_complete", True),
                "incomplete_question",
                "question contains omission marker or obvious missing content",
            ),
        ]
        for failed, issue_type, details in checks:
            if not failed:
                continue
            validation.append(
                {
                    "company": row.get("company", ""),
                    "stock_code": row.get("stock_code", ""),
                    "question_id": question_id,
                    "source_document_type": row.get("source_document_type", ""),
                    "validation_type": f"{dataset}_{issue_type}",
                    "expected_subquestions": "",
                    "actual_subquestions": "",
                    "status": "warn",
                    "details": details,
                }
            )

    for (stock_code, cleaned_text), sources in text_sources.items():
        if len(sources) <= 1:
            continue
        affected = [row for row in rows if row.get("stock_code") == stock_code and row.get("cleaned_question_text") == cleaned_text]
        for row in affected:
            validation.append(
                {
                    "company": row.get("company", ""),
                    "stock_code": row.get("stock_code", ""),
                    "question_id": row.get("question_id", ""),
                    "source_document_type": row.get("source_document_type", ""),
                    "validation_type": f"{dataset}_duplicate_question_across_files",
                    "expected_subquestions": "",
                    "actual_subquestions": "",
                    "status": "warn",
                    "details": "same cleaned question text appears in multiple source files",
                }
            )

    return validation


def write_quality_summary(
    path: Path,
    raw_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    auxiliary_rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:
    formal_levels = Counter(row["question_level"] for row in clean_rows)
    auxiliary_levels = Counter(row["question_level"] for row in auxiliary_rows)
    source_counts = Counter(row["source_document_type"] for row in clean_rows)
    excluded = sum(1 for row in clean_rows if row["exclude_from_training"])
    trainable = sum(1 for row in clean_rows if row["is_training_target"])
    incomplete = sum(1 for row in clean_rows if not row["is_complete"])
    professional = sum(1 for row in clean_rows if row["professional_opinion_request"])
    validation_counts = Counter(row["validation_type"] for row in validation)

    company_lines = []
    by_company: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        by_company[(row["stock_code"], row["company"])].append(row)
    for (stock_code, company), rows in sorted(by_company.items()):
        levels = Counter(row["question_level"] for row in rows)
        trainable_company = sum(1 for row in rows if row["is_training_target"])
        company_lines.append(
            f"| {stock_code} | {company} | {levels['parent']} | {levels['subquestion']} | {trainable_company} |"
        )

    validation_lines = [
        f"| {issue_type} | {count} |" for issue_type, count in sorted(validation_counts.items())
    ]

    content = "\n".join(
        [
            "# Question Quality Summary",
            "",
            "## Outputs",
            "",
            "- Preserved raw extraction: `data/processed/regulatory_questions_raw.jsonl`",
            "- Formal cleaned questions: `data/processed/regulatory_questions_clean.jsonl`",
            "- Auxiliary supporting-source questions: `data/processed/auxiliary_questions.jsonl`",
            "- Validation table: `reports/question_extraction_validation.csv`",
            "",
            "## Overall Counts",
            "",
            f"- Raw records: {len(raw_rows)}",
            f"- Formal cleaned records: {len(clean_rows)}",
            f"- Formal parent questions retained as context: {formal_levels['parent']}",
            f"- Formal subquestions: {formal_levels['subquestion']}",
            f"- Formal trainable subquestions: {trainable}",
            f"- Formal excluded records: {excluded}",
            f"- Formal incomplete records: {incomplete}",
            f"- Formal records with separated professional/source-note requests: {professional}",
            f"- Auxiliary records moved out of gold data: {len(auxiliary_rows)}",
            f"- Auxiliary parent questions: {auxiliary_levels['parent']}",
            f"- Auxiliary subquestions: {auxiliary_levels['subquestion']}",
            "",
            "## Formal Source Distribution",
            "",
            f"- inquiry: {source_counts['inquiry']}",
            f"- reply: {source_counts['reply']}",
            f"- supporting: {source_counts['supporting']}",
            "",
            "## Company Counts",
            "",
            "| Stock code | Company | Parent | Subquestion | Trainable subquestion |",
            "| --- | --- | ---: | ---: | ---: |",
            *company_lines,
            "",
            "## Repair Notes",
            "",
            "- Reparsed 南京新百 inquiry from `document_pages.jsonl`; restored original questions 1-7.",
            "- Restored 南京新百 original question 1 and its 3 numbered subquestions.",
            "- Trimmed 南京新百 question 7 before reply deadline, disclosure-media, risk-warning, announcement and board-date footer text.",
            "- Moved 利通电子 and 五新隧装 supporting-source records to auxiliary output and excluded them from formal gold data.",
            "- Added fallback subquestions for parent questions that had no numbered children but contained clear regulatory requests.",
            "- Separated auditor, independent-director, sponsor/broker opinion requests and inquiry-source notes into `professional_opinion_request`.",
            "",
            "## Validation Issue Counts",
            "",
            "| Validation type | Count |",
            "| --- | ---: |",
            *validation_lines,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_question_cleaning(root: Path) -> dict[str, Any]:
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    raw_backup = ensure_raw_backup(processed_dir)

    raw_rows = load_jsonl(raw_backup)
    pages = load_jsonl(processed_dir / "document_pages.jsonl")
    clean_rows, auxiliary_rows, validation = build_clean_rows(raw_rows, pages)

    write_jsonl(processed_dir / "regulatory_questions_clean.jsonl", clean_rows)
    write_jsonl(processed_dir / "auxiliary_questions.jsonl", auxiliary_rows)
    write_csv(
        reports_dir / "question_extraction_validation.csv",
        validation,
        [
            "company",
            "stock_code",
            "question_id",
            "source_document_type",
            "validation_type",
            "expected_subquestions",
            "actual_subquestions",
            "status",
            "details",
        ],
    )
    write_quality_summary(
        reports_dir / "question_quality_summary.md",
        raw_rows,
        clean_rows,
        auxiliary_rows,
        validation,
    )

    return {
        "raw_records": len(raw_rows),
        "formal_records": len(clean_rows),
        "auxiliary_records": len(auxiliary_rows),
        "formal_parent_questions": sum(
            1 for row in clean_rows if row["question_level"] == "parent"
        ),
        "formal_subquestions": sum(
            1 for row in clean_rows if row["question_level"] == "subquestion"
        ),
        "trainable_subquestions": sum(
            1 for row in clean_rows if row["is_training_target"]
        ),
        "excluded_records": sum(1 for row in clean_rows if row["exclude_from_training"]),
        "validation_records": len(validation),
    }
