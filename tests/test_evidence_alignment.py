from __future__ import annotations

from src.data_pipeline.evidence_alignment import (
    BM25Index,
    build_annual_report_chunks,
    build_query,
    ranges_intersect,
    retrieve_candidates,
    select_training_questions,
)


def make_page(
    stock_code: str,
    company: str,
    page_number: int,
    text: str,
    document_type: str = "annual_report",
) -> dict[str, object]:
    return {
        "document_id": f"{stock_code}_2024_{document_type}",
        "company": company,
        "stock_code": stock_code,
        "report_year": "2024",
        "document_type": document_type,
        "supporting_subtype": "",
        "source_file": f"{stock_code}_{document_type}.pdf",
        "page_number": page_number,
        "text": text,
        "char_count": len(text),
        "extraction_status": "success",
    }


def make_question(stock_code: str = "600000", company: str = "测试公司") -> dict[str, object]:
    return {
        "question_id": f"{stock_code}_2024_q01_sub01",
        "company": company,
        "stock_code": stock_code,
        "report_year": 2024,
        "topic_title": "关于存货",
        "parent_question_text": "问题1.关于存货。年报显示存货大幅增长。请公司说明原因。",
        "cleaned_question_text": "说明存货大幅增长的原因及跌价准备计提是否充分",
        "question_level": "subquestion",
        "is_training_target": True,
        "exclude_from_training": False,
        "original_question_number": 1,
        "subquestion_number": 1,
    }


def test_build_chunks_uses_only_annual_report_pages_and_keeps_page_range() -> None:
    pages = [
        make_page("600000", "测试公司", 1, "存货增长。" * 80),
        make_page("600000", "测试公司", 2, "跌价准备。" * 80),
        make_page("600000", "测试公司", 1, "问询函文本。" * 80, "reply"),
    ]

    chunks = build_annual_report_chunks(pages, chunk_size=120, overlap=20)

    assert chunks
    assert all(chunk["stock_code"] == "600000" for chunk in chunks)
    assert all("问询函文本" not in chunk["text"] for chunk in chunks)
    assert chunks[0]["page_start"] == 1
    assert chunks[-1]["page_end"] == 2


def test_bm25_retrieves_matching_chunk() -> None:
    documents = [
        {"chunk_id": "a", "text": "货币资金 利息收入 受限资金"},
        {"chunk_id": "b", "text": "存货 跌价准备 可变现净值"},
    ]
    index = BM25Index(documents)

    results = index.top_n("存货跌价准备是否充分", 1)

    assert results[0][0]["chunk_id"] == "b"
    assert results[0][1] > 0


def test_retrieve_candidates_isolated_by_company_and_year() -> None:
    question = make_question("600000", "测试公司")
    chunks = [
        {
            "chunk_id": "600000_2024_ar_chunk_0001",
            "company": "测试公司",
            "stock_code": "600000",
            "report_year": 2024,
            "source_file": "a.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "存货大幅增长，跌价准备计提情况。",
        },
        {
            "chunk_id": "600001_2024_ar_chunk_0001",
            "company": "其他公司",
            "stock_code": "600001",
            "report_year": 2024,
            "source_file": "b.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "存货大幅增长，跌价准备计提情况。完全匹配。",
        },
    ]

    rows = retrieve_candidates([question], chunks)

    assert rows[0]["candidates"]
    assert rows[0]["candidates"][0]["chunk_id"].startswith("600000_2024")


def test_training_question_filter_and_query_background() -> None:
    valid = make_question()
    parent = {**valid, "question_level": "parent", "is_training_target": False}
    excluded = {**valid, "question_id": "excluded", "exclude_from_training": True}

    selected = select_training_questions([valid, parent, excluded])
    query = build_query(valid)

    assert selected == [valid]
    assert "年报显示存货大幅增长" in query
    assert "请公司说明原因" not in query
    assert "跌价准备计提是否充分" in query


def test_page_range_intersection() -> None:
    assert ranges_intersect(10, 12, 12, 15)
    assert ranges_intersect(10, 12, 8, 10)
    assert not ranges_intersect(10, 12, 13, 15)
    assert not ranges_intersect(None, 12, 10, 12)
