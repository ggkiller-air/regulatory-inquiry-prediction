from __future__ import annotations

from src.data_pipeline.question_cleaning import (
    build_clean_rows,
    derive_subquestions,
    process_parent_group,
    reparse_nanjing_inquiry,
    strip_professional_requests,
)


def make_parent(
    question_id: str = "600000_2024_q01",
    stock_code: str = "600000",
    source_document_type: str = "reply",
    text: str = "问题1.关于收入。请公司说明收入下降原因。",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "company": "测试公司",
        "stock_code": stock_code,
        "report_year": "2024",
        "topic_title": "关于收入",
        "parent_question_text": text,
        "question_text": text,
        "question_level": "parent",
        "source_document_type": source_document_type,
        "source_file": "reply.pdf",
        "source_page_start": 1,
        "source_page_end": 1,
        "extraction_method": "regex_parent_block",
        "needs_review": False,
    }


def make_page(page_number: int, text: str) -> dict[str, object]:
    return {
        "document_id": "600682_2024_inquiry_test",
        "company": "南京新百",
        "stock_code": "600682",
        "report_year": "2024",
        "document_type": "inquiry",
        "supporting_subtype": "",
        "source_file": "nanjing_inquiry.pdf",
        "page_number": page_number,
        "text": text,
        "char_count": len(text),
        "extraction_status": "success",
    }


def test_nanjing_reparse_recovers_question_one_and_trims_footer() -> None:
    pages = [
        make_page(
            1,
            "一、关于公司经营情况\n"
            "1.年报披露，期间费用增长。请公司：（1）分业务板块列示费用构成；"
            "（2）列示聘请中介机构费和咨询费对象；（3）结合主营业务说明支出增长合理性。\n",
        ),
        make_page(
            2,
            "2.年报披露，子公司持有货币资金。请公司：（1）补充披露现金分红；"
            "（2）结合利润分配政策说明资金调配。\n"
            "二、关于公司财务信息\n"
            "3.关于商誉减值。请公司：（1）列示减值测试过程；（2）说明计提是否充分。\n"
            "4.关于应收账款。请公司：（1）说明坏账参数；（2）披露前五大客户。\n"
            "5.关于其他应收款。请公司补充披露交易对手方，并说明挂账原因。\n"
            "6.关于货币资金。请公司说明存在大额资金且贷款逾期的原因。\n"
            "7.关于存货。请公司补充披露原材料内容和库龄。请年审会计师说明审计程序。\n"
            "针对前述问题，公司应当书面回复。\n特此公告。\n董事会\n2025年7月5日",
        ),
    ]

    parents = reparse_nanjing_inquiry(pages)
    first_children = derive_subquestions(parents[0]["question_text"])

    assert [row["question_id"] for row in parents] == [
        "600682_2024_q01",
        "600682_2024_q02",
        "600682_2024_q03",
        "600682_2024_q04",
        "600682_2024_q05",
        "600682_2024_q06",
        "600682_2024_q07",
    ]
    assert len(first_children) == 3
    assert "针对前述问题" not in parents[-1]["question_text"]
    assert "特此公告" not in parents[-1]["question_text"]


def test_professional_requests_and_source_notes_are_separated() -> None:
    text = (
        "说明毛利率下降的原因。请年审会计师就上述问题进行核查并发表明确意见。"
        "（问询函第一条）"
    )

    cleaned, professional_request = strip_professional_requests(text)

    assert cleaned == "说明毛利率下降的原因"
    assert "年审会计师" in professional_request
    assert "问询函第一条" in professional_request


def test_parent_generates_unlabeled_subquestion_and_subquestion_is_target() -> None:
    parent = make_parent(
        text="问题1.关于收入。年报披露收入下降。请公司说明收入下降原因及合理性。"
    )

    rows, _validation = process_parent_group(parent, [], 1, False)

    assert rows[0]["question_level"] == "parent"
    assert rows[0]["exclude_from_training"] is True
    assert rows[0]["is_training_target"] is False
    assert rows[1]["question_level"] == "subquestion"
    assert rows[1]["is_training_target"] is True
    assert rows[1]["report_year"] == 2024


def test_supporting_rows_are_moved_to_auxiliary() -> None:
    raw_rows = [
        make_parent(
            question_id="603629_2024_q01",
            stock_code="603629",
            source_document_type="supporting",
            text="一、关于算力业务收入。请公司说明收入增长原因。",
        )
    ]

    clean_rows, auxiliary_rows, _validation = build_clean_rows(raw_rows, [])

    assert clean_rows == []
    assert len(auxiliary_rows) == 2
    assert all(row["source_document_type"] == "supporting" for row in auxiliary_rows)
    assert all(row["exclude_from_training"] is True for row in auxiliary_rows)


def test_numbered_subquestions_ignore_professional_reference_markers() -> None:
    parent = (
        "问题1.关于费用。请公司：（1）说明销售费用增长原因；"
        "（2）说明管理费用增长原因。请独立董事对问题（1）（2）发表意见。"
    )

    children = derive_subquestions(parent)

    assert len(children) == 2
    assert "独立董事" in children[1].raw_text
