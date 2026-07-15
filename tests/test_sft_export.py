from __future__ import annotations

from src.data_pipeline.sft_export import build_sft_sample


def test_sft_sample_uses_evidence_as_input_and_question_as_output() -> None:
    row = {
        "question_id": "600000_2024_q01_sub01",
        "company": "测试公司",
        "stock_code": "600000",
        "report_year": 2024,
        "topic_title": "关于存货",
        "original_question_number": 1,
        "subquestion_number": 1,
        "cleaned_question_text": "说明存货增长原因及跌价准备是否充分",
        "candidates": [
            {
                "chunk_id": "600000_2024_ar_chunk_0001",
                "page_start": 10,
                "page_end": 10,
                "text": "年报披露，存货账面价值同比增长，已计提跌价准备。",
            }
        ],
    }

    sample = build_sft_sample(row)
    user_content = sample["messages"][1]["content"]

    assert "年报披露，存货账面价值同比增长" in user_content
    assert "说明存货增长原因及跌价准备是否充分" not in user_content
    assert sample["messages"][2]["content"] == "说明存货增长原因及跌价准备是否充分"
    assert sample["metadata"]["evidence_chunk_ids"] == ["600000_2024_ar_chunk_0001"]
