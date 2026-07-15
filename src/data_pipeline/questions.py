from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .models import DocumentRecord, PageRecord, ParseFailure, QuestionRecord


QUESTION_TRIGGER_RE = re.compile(
    r"请(?:公司|你公司|年审会计师|会计师|保荐机构|独立董事)|请补充披露|请结合|请说明"
)

PARENT_START_RE = re.compile(
    r"(?m)^(?P<label>"
    r"问题\s*[一二三四五六七八九十\d]+[、.．]?\s*(?:关于[^\n。；：:]{1,80})?[。:：]?"
    r"|[一二三四五六七八九十]+、\s*关于[^\n]{1,100}"
    r"|\d+\s*[\.、]\s*(?:关于[^\n。；：:]{1,80}|年报披露|年报及前期公告|你公司|本期|报告期)[^\n]{0,120}"
    r")"
)

SECTION_HEADING_RE = re.compile(r"(?m)^([一二三四五六七八九十]+、\s*关于[^\n。；：:]{1,80})")

ANSWER_BOUNDARY_RE = re.compile(
    r"(?m)^\s*(?:"
    r"回复[:：]"
    r"|公司回复[:：]"
    r"|【公司回复】"
    r"|【回复】"
    r"|公司说明[:：]?"
    r"|[一二三四五六七八九十]+、公司说明"
    r"|[（(]一[）)]\s*公司回复"
    r"|年审会计师回复[:：]"
    r"|会计师回复[:：]"
    r"|保荐机构回复[:：]"
    r")"
)

INLINE_ANSWER_BOUNDARY_RE = re.compile(
    r"(?:【公司回复】|【回复】|一、公司回复|公司回复[:：]?|公司说明[:：]?|回复[:：])"
)

SUBQUESTION_RE = re.compile(r"[（(]\s*(?P<label>\d+|[一二三四五六七八九十]+)\s*[）)]")

SOURCE_PRIORITY = {
    "inquiry": 0,
    "reply": 1,
    "auditor_reply": 2,
    "sponsor_or_broker": 3,
    "independent_director": 4,
    "": 5,
}


@dataclass(frozen=True)
class PageSpan:
    start: int
    end: int
    page_number: int


@dataclass(frozen=True)
class SourceText:
    text: str
    spans: list[PageSpan]

    def page_for_offset(self, offset: int) -> int | None:
        for span in self.spans:
            if span.start <= offset <= span.end:
                return span.page_number
        return self.spans[-1].page_number if self.spans else None

    def page_range(self, start: int, end: int) -> tuple[int | None, int | None]:
        pages = [
            span.page_number
            for span in self.spans
            if span.start <= end and span.end >= start
        ]
        if not pages:
            return self.page_for_offset(start), self.page_for_offset(end)
        return min(pages), max(pages)


def compact_text(text: str) -> str:
    text = re.sub(r"本公司董事会及全体董事保证.{0,260}?法律责任。", "", text, flags=re.S)
    text = re.sub(r"\[\[PAGE:\d+]]", "", text)
    text = re.sub(r"(?m)^\s*第\s*\d+\s*页\s*共\s*\d+\s*页\s*$", "", text)
    text = re.sub(r"(?m)^\s*-\s*\d+\s*-\s*$", "", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*(?:证券代码[:：].*|证券简称[:：].*|公告编号[:：].*)$", "", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text.strip(" \n\t；;")


def build_source_text(pages: list[PageRecord]) -> SourceText:
    chunks: list[str] = []
    spans: list[PageSpan] = []
    offset = 0
    for page in sorted(pages, key=lambda item: item.page_number):
        marker = f"\n[[PAGE:{page.page_number}]]\n"
        chunk = marker + page.text + "\n"
        text_start = offset + len(marker)
        text_end = text_start + len(page.text)
        chunks.append(chunk)
        spans.append(PageSpan(start=text_start, end=text_end, page_number=page.page_number))
        offset += len(chunk)
    return SourceText(text="".join(chunks), spans=spans)


def cut_before_answer(block: str) -> str:
    boundary = ANSWER_BOUNDARY_RE.search(block)
    boundary_start = boundary.start() if boundary else len(block)
    inline_boundary = INLINE_ANSWER_BOUNDARY_RE.search(block)
    if inline_boundary:
        boundary_start = min(boundary_start, inline_boundary.start())

    chinese_answer_section = re.search(r"(?m)^\s*[（(][一二三四五六七八九十]+[）)]\s*", block)
    if chinese_answer_section and re.search(
        r"请\s*(?:年审)?\s*会计师|请\s*独立董事|请\s*保荐机构",
        block[: chinese_answer_section.start()],
    ):
        boundary_start = min(boundary_start, chinese_answer_section.start())

    return block[:boundary_start].strip()


def extract_topic_title(label: str, block: str, section_title: str) -> str:
    candidates = [label, block[:120], section_title]
    for candidate in candidates:
        match = re.search(r"关于[^。；：:\n]{1,60}", candidate)
        if match:
            return match.group(0).rstrip("。；：: ")
    return section_title.rstrip("。；：: ")


def split_subquestions(parent_block: str) -> list[tuple[int, int, str]]:
    trigger_match = QUESTION_TRIGGER_RE.search(parent_block)
    search_start = trigger_match.start() if trigger_match else 0
    reference_spans = [
        (match.start(), match.end())
        for match in re.finditer(
            r"问题(?:\s*[（(]\s*(?:\d+|[一二三四五六七八九十]+)\s*[）)])+",
            parent_block,
        )
    ]
    matches = [
        match
        for match in SUBQUESTION_RE.finditer(parent_block, search_start)
        if not any(start <= match.start() < end for start, end in reference_spans)
    ]
    children: list[tuple[int, int, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(parent_block)
        child = compact_text(parent_block[start:end])
        if len(child) < 8:
            continue
        if not re.search(r"说明|披露|列示|分析|补充|结合|核实|解释|测算", child):
            continue
        children.append((match.start(), end, child))
    return children


def find_parent_blocks(source_text: SourceText) -> list[tuple[int, int, str, str, str]]:
    matches = list(PARENT_START_RE.finditer(source_text.text))
    blocks: list[tuple[int, int, str, str, str]] = []
    current_section = ""
    section_matches = list(SECTION_HEADING_RE.finditer(source_text.text))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source_text.text)
        raw_block = source_text.text[start:end]
        question_block = cut_before_answer(raw_block)
        if not QUESTION_TRIGGER_RE.search(question_block):
            continue

        prior_sections = [section for section in section_matches if section.start() <= start]
        if prior_sections:
            current_section = compact_text(prior_sections[-1].group(1))

        label = match.group("label")
        topic = extract_topic_title(label=label, block=question_block, section_title=current_section)
        blocks.append((start, start + len(question_block), label, topic, question_block))
    return blocks


def select_question_source_documents(documents: list[DocumentRecord]) -> dict[str, list[DocumentRecord]]:
    by_company: dict[str, list[DocumentRecord]] = defaultdict(list)
    for document in documents:
        if document.is_missing or document.parse_status == "failed":
            continue
        if document.document_type in {"inquiry", "reply", "supporting"}:
            by_company[document.stock_code].append(document)

    selected: dict[str, list[DocumentRecord]] = {}
    for stock_code, company_docs in by_company.items():
        inquiry_docs = [doc for doc in company_docs if doc.document_type == "inquiry"]
        reply_docs = [doc for doc in company_docs if doc.document_type == "reply"]
        supporting_docs = [doc for doc in company_docs if doc.document_type == "supporting"]
        if inquiry_docs:
            selected[stock_code] = sorted(inquiry_docs, key=lambda doc: doc.relative_path)
        elif reply_docs:
            selected[stock_code] = sorted(reply_docs, key=lambda doc: doc.relative_path)
        else:
            selected[stock_code] = sorted(
                supporting_docs,
                key=lambda doc: (
                    SOURCE_PRIORITY.get(doc.supporting_subtype, 9),
                    doc.relative_path,
                ),
            )[:1]
    return selected


def extract_questions(
    documents: list[DocumentRecord],
    pages: list[PageRecord],
) -> tuple[list[QuestionRecord], list[ParseFailure]]:
    pages_by_doc: dict[str, list[PageRecord]] = defaultdict(list)
    for page in pages:
        pages_by_doc[page.document_id].append(page)

    selected = select_question_source_documents(documents)
    questions: list[QuestionRecord] = []
    failures: list[ParseFailure] = []

    for company_docs in selected.values():
        for document in company_docs:
            doc_pages = pages_by_doc.get(document.document_id, [])
            source = build_source_text(doc_pages)
            blocks = find_parent_blocks(source)
            if not blocks:
                failures.append(
                    ParseFailure(
                        document_id=document.document_id,
                        company=document.company,
                        stock_code=document.stock_code,
                        document_type=document.document_type,
                        source_file=document.relative_path,
                        failure_type="question_extraction_empty",
                        reason="no regulatory question block matched selected source",
                    )
                )
                continue

            needs_review = document.document_type == "supporting"
            for parent_index, (start, end, _label, topic, raw_block) in enumerate(blocks, start=1):
                parent_text = compact_text(raw_block)
                page_start, page_end = source.page_range(start, end)
                parent_id = f"{document.stock_code}_{document.report_year}_q{parent_index:02d}"
                questions.append(
                    QuestionRecord(
                        question_id=parent_id,
                        company=document.company,
                        stock_code=document.stock_code,
                        report_year=document.report_year,
                        topic_title=topic,
                        parent_question_text=parent_text,
                        question_text=parent_text,
                        question_level="parent",
                        source_document_type=document.document_type,
                        source_file=document.relative_path,
                        source_page_start=page_start,
                        source_page_end=page_end,
                        extraction_method="regex_parent_block",
                        needs_review=needs_review,
                    )
                )

                for child_index, (child_start, child_end, child_text) in enumerate(
                    split_subquestions(raw_block),
                    start=1,
                ):
                    child_page_start, child_page_end = source.page_range(
                        start + child_start,
                        start + child_end,
                    )
                    questions.append(
                        QuestionRecord(
                            question_id=f"{parent_id}_sub{child_index:02d}",
                            company=document.company,
                            stock_code=document.stock_code,
                            report_year=document.report_year,
                            topic_title=topic,
                            parent_question_text=parent_text,
                            question_text=child_text,
                            question_level="subquestion",
                            source_document_type=document.document_type,
                            source_file=document.relative_path,
                            source_page_start=child_page_start,
                            source_page_end=child_page_end,
                            extraction_method="regex_subquestion_split",
                            needs_review=needs_review,
                        )
                    )
    return questions, failures
