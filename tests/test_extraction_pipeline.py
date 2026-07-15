from __future__ import annotations

import json

from src.data_pipeline.models import DocumentRecord, PageRecord, QuestionRecord
from src.data_pipeline.pdf_extract import page_extraction_status
from src.data_pipeline.questions import (
    build_source_text,
    extract_questions,
    find_parent_blocks,
    select_question_source_documents,
    split_subquestions,
)


def make_doc(
    stock_code: str,
    company: str,
    document_type: str,
    document_id: str,
    relative_path: str,
    supporting_subtype: str = "",
    is_missing: bool = False,
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        company=company,
        stock_code=stock_code,
        report_year="2024",
        document_type=document_type,
        supporting_subtype=supporting_subtype,
        relative_path=relative_path,
        filename=relative_path.rsplit("/", 1)[-1],
        file_size=100,
        is_missing=is_missing,
        parse_status="skipped_missing" if is_missing else "success",
        page_count=1,
        notes="",
    )


def make_page(document: DocumentRecord, text: str) -> PageRecord:
    return PageRecord(
        document_id=document.document_id,
        company=document.company,
        stock_code=document.stock_code,
        report_year=document.report_year,
        document_type=document.document_type,
        supporting_subtype=document.supporting_subtype,
        source_file=document.relative_path,
        page_number=1,
        text=text,
        char_count=len(text),
        extraction_status="success",
    )


def test_reply_question_stops_before_answer_boundary() -> None:
    text = (
        "问题1.关于经营业绩。\n"
        "年报披露，公司收入下降。请公司：（1）说明收入下降原因；（2）说明毛利率变化原因。"
        "请年审会计师发表意见。\n"
        "回复：\n"
        "一、公司说明\n"
        "这里是公司回答，不应进入问题。"
    )
    source = build_source_text(
        [
            PageRecord(
                document_id="doc",
                company="测试公司",
                stock_code="000001",
                report_year="2024",
                document_type="reply",
                source_file="reply.pdf",
                page_number=1,
                text=text,
                char_count=len(text),
                extraction_status="success",
            )
        ]
    )

    blocks = find_parent_blocks(source)

    assert len(blocks) == 1
    assert "公司回答" not in blocks[0][4]
    assert "回复：" not in blocks[0][4]


def test_parent_question_and_subquestion_split() -> None:
    parent = (
        "问题1.关于经营业绩。年报披露，公司收入下降。"
        "请公司：（1）说明收入下降原因；（2）补充披露主要客户情况；"
        "（3）结合成本说明毛利率变化原因。请年审会计师发表意见。"
    )

    children = split_subquestions(parent)

    assert len(children) == 3
    assert "收入下降原因" in children[0][2]
    assert "主要客户" in children[1][2]
    assert "毛利率" in children[2][2]


def test_inquiry_priority_over_reply() -> None:
    inquiry = make_doc("600682", "南京新百", "inquiry", "inquiry_doc", "inquiry/a.pdf")
    reply = make_doc("600682", "南京新百", "reply", "reply_doc", "reply/a.pdf")

    selected = select_question_source_documents([reply, inquiry])

    assert selected["600682"] == [inquiry]


def test_supporting_fallback_sets_needs_review() -> None:
    supporting = make_doc(
        "603629",
        "利通电子",
        "supporting",
        "supporting_doc",
        "supporting/auditor_reply/a.pdf",
        supporting_subtype="auditor_reply",
    )
    page = make_page(
        supporting,
        (
            "一、关于算力业务收入\n"
            "年报显示，公司算力业务增长。请公司补充披露：（1）说明收入确认政策；"
            "（2）说明毛利率较高的原因。请年审会计师发表意见。\n"
            "(一) 算力租赁和技服维保业务的经营模式\n"
            "公司回复：回答正文。"
        ),
    )

    questions, failures = extract_questions([supporting], [page])

    assert not failures
    assert questions
    assert all(question.needs_review for question in questions)
    assert {question.question_level for question in questions} == {"parent", "subquestion"}
    assert all("公司回复" not in question.question_text for question in questions)


def test_needs_ocr_status_for_sparse_text() -> None:
    assert page_extraction_status("封面") == "needs_ocr"
    assert page_extraction_status("这是一个足够长的页面文本。" * 5) == "success"


def test_question_jsonl_schema_validation() -> None:
    question = QuestionRecord(
        question_id="600000_2024_q01",
        company="测试公司",
        stock_code="600000",
        report_year="2024",
        topic_title="关于经营业绩",
        parent_question_text="问题1.关于经营业绩。请公司说明收入下降原因。",
        question_text="请公司说明收入下降原因。",
        question_level="subquestion",
        source_document_type="reply",
        source_file="reply.pdf",
        source_page_start=1,
        source_page_end=1,
        extraction_method="regex_subquestion_split",
        needs_review=False,
    )
    payload = json.dumps(question.model_dump(), ensure_ascii=False)

    loaded = QuestionRecord.model_validate_json(payload)

    assert loaded.question_id == question.question_id
