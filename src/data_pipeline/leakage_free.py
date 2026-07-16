"""无泄漏数据重建：标准化风险主题 + 去泄漏证据检索 + 新 SFT 导出。

检索 query 只使用标准化风险主题与通用财经风险关键词；
公司与报告年度仅用于限定检索语料范围（或在全局消融中不限定）。
禁止使用真实监管子问题、父问题事实描述、回复公告内容及金标中的具体事实。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .evidence_alignment import BM25Index
from .io_utils import write_jsonl

# ---------------------------------------------------------------------------
# 1. 风险主题标准化
# ---------------------------------------------------------------------------

STANDARD_TOPICS = [
    "收入确认与经营真实性",
    "应收账款与信用风险",
    "资产减值与商誉",
    "关联交易与资金占用",
    "成本费用与盈利质量",
    "现金流与持续经营",
    "公司治理与信息披露",
    "其他财务风险",
]

# topic_title（44 个去重值）→ 粗粒度标准主题。
# 映射只依据 topic_title 字面所属的会计/监管领域，不引入任何公司事实。
TOPIC_TITLE_MAP = {
    "关于经营业绩": "收入确认与经营真实性",
    "关于公司经营情况": "收入确认与经营真实性",
    "关于公司主营业务和经营业绩": "收入确认与经营真实性",
    "关于主营业务": "收入确认与经营真实性",
    "关于营业收入": "收入确认与经营真实性",
    "关于业绩波动": "收入确认与经营真实性",
    "关于电网业务": "收入确认与经营真实性",
    "关于储能业务": "收入确认与经营真实性",
    "关于算力服务及销售业务": "收入确认与经营真实性",
    "关于境外收入": "收入确认与经营真实性",
    "关于线下销售收入": "收入确认与经营真实性",
    "关于收入确认和销售退回": "收入确认与经营真实性",
    "关于航空航天及民用零部件业务": "收入确认与经营真实性",
    "关于毛利率": "收入确认与经营真实性",
    "关于客户和供应商": "收入确认与经营真实性",
    "关于未完工项目": "收入确认与经营真实性",
    "关于应收账款": "应收账款与信用风险",
    "关于应收账款和长期应收款": "应收账款与信用风险",
    "关于其他应收款和预付款项": "应收账款与信用风险",
    "关于应收和预付款项": "应收账款与信用风险",
    "关于其他应收款": "应收账款与信用风险",
    "关于预付款项": "应收账款与信用风险",
    "关于存货": "资产减值与商誉",
    "关于商誉": "资产减值与商誉",
    "关于商誉减值": "资产减值与商誉",
    "关于固定资产与在建工程": "资产减值与商誉",
    "关于固定资产及在建工程": "资产减值与商誉",
    "关于在建工程": "资产减值与商誉",
    "关于长期股权投资": "资产减值与商誉",
    "关于关联交易": "关联交易与资金占用",
    "关于往来款项与对外担保": "关联交易与资金占用",
    "关于往来款项": "关联交易与资金占用",
    "关于往来款和内部控制": "关联交易与资金占用",
    "关于销售费用": "成本费用与盈利质量",
    "关于期间费用": "成本费用与盈利质量",
    "关于货币资金": "现金流与持续经营",
    "关于货币资金和借款": "现金流与持续经营",
    "关于委托理财与货币资金": "现金流与持续经营",
    "关于偿债能力": "现金流与持续经营",
    "关于投资理财": "现金流与持续经营",
    "关于对外投资": "现金流与持续经营",
    "关于募投项目": "现金流与持续经营",
    "关于会计差错更正": "公司治理与信息披露",
    "关于其他": "其他财务风险",
}

# 每个标准主题的通用财经风险关键词（仅通用会计/监管术语，不含公司事实）。
TOPIC_KEYWORDS = {
    "收入确认与经营真实性": "营业收入 收入确认 主营业务 毛利率 经营业绩 销售收入 客户 供应商 收入真实性 业绩波动",
    "应收账款与信用风险": "应收账款 其他应收款 预付款项 坏账准备 账龄 信用风险 回款 减值 期后回款",
    "资产减值与商誉": "存货 存货跌价准备 商誉 减值测试 固定资产 在建工程 长期股权投资 资产减值 可收回金额",
    "关联交易与资金占用": "关联交易 关联方 资金占用 对外担保 往来款项 内部控制 非经营性资金往来",
    "成本费用与盈利质量": "销售费用 管理费用 研发费用 财务费用 期间费用 营业成本 盈利质量",
    "现金流与持续经营": "货币资金 现金流量 借款 有息负债 偿债能力 委托理财 投资理财 募集资金 流动性 持续经营",
    "公司治理与信息披露": "会计差错更正 前期会计差错 会计政策 会计估计 信息披露 公司治理 内部控制",
    "其他财务风险": "财务风险 会计处理 信息披露 合规",
}

SYSTEM_PROMPT = (
    "你是一名熟悉中国上市公司年度报告和交易所信息披露监管规则的财经文本分析助手。"
    "请根据给定的公司信息、风险主题以及可能提供的年度报告证据，"
    "生成可能被监管机构提出的一个具体监管问询子问题。"
)

USER_TEMPLATE_WITH_EVIDENCE = """请围绕给定风险主题，根据以下同公司、同年度的年度报告证据，生成一个监管问询子问题。

公司：{company}
证券代码：{stock_code}
报告年度：{report_year}
风险主题：{risk_topic}

年度报告证据：
{evidence_text}

要求：
1. 问题应聚焦一个明确的信息披露风险点。
2. 使用监管问询风格，包含“说明、披露、列示、结合、是否”等要求。
3. 不要回答问题，不要编造与年度报告无关的事实。
"""

USER_TEMPLATE_NO_EVIDENCE = """请围绕给定风险主题，针对该公司该年度的年度报告，生成一个监管问询子问题。

公司：{company}
证券代码：{stock_code}
报告年度：{report_year}
风险主题：{risk_topic}

要求：
1. 问题应聚焦一个明确的信息披露风险点。
2. 使用监管问询风格，包含“说明、披露、列示、结合、是否”等要求。
3. 不要回答问题，不要编造与年度报告无关的事实。
"""


def build_query(risk_topic: str) -> str:
    return f"{risk_topic} {TOPIC_KEYWORDS[risk_topic]}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_page_range(chunk: dict[str, Any]) -> str:
    start, end = chunk.get("page_start"), chunk.get("page_end")
    if start == end:
        return f"p{start}"
    return f"p{start}-{end}"


def build_evidence_text(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"【证据{index}｜年报页 {format_page_range(chunk)}｜chunk {chunk['chunk_id']}】"
        )
        lines.append(str(chunk["text"]).strip())
    return "\n".join(lines)


def build_sample(
    old: dict[str, Any],
    risk_topic: str,
    evidence_chunks: list[dict[str, Any]],
    source_tag: str,
) -> dict[str, Any]:
    meta = old["metadata"]
    common = {
        "company": meta["company"],
        "stock_code": meta["stock_code"],
        "report_year": meta["report_year"],
        "risk_topic": risk_topic,
    }
    if evidence_chunks:
        user_content = USER_TEMPLATE_WITH_EVIDENCE.format(
            evidence_text=build_evidence_text(evidence_chunks), **common
        )
    else:
        user_content = USER_TEMPLATE_NO_EVIDENCE.format(**common)
    new_meta = dict(meta)
    new_meta.update(
        {
            "risk_topic": risk_topic,
            "retrieval_query": build_query(risk_topic),
            "evidence_top_k": len(evidence_chunks),
            "evidence_chunk_ids": [c["chunk_id"] for c in evidence_chunks],
            "evidence_page_ranges": [format_page_range(c) for c in evidence_chunks],
            "evidence_companies": sorted({c["stock_code"] for c in evidence_chunks}),
            "source": source_tag,
        }
    )
    return {
        "sample_id": old["sample_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": old["messages"][-1]["content"]},
        ],
        "metadata": new_meta,
    }


def main() -> None:
    root = Path(".")
    chunks = load_jsonl(root / "data/processed/annual_report_chunks.jsonl")

    per_report: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for chunk in chunks:
        per_report.setdefault((str(chunk["stock_code"]), int(chunk["report_year"])), []).append(chunk)
    indexes = {key: BM25Index(value) for key, value in per_report.items()}
    global_index = BM25Index(chunks)

    out_dir = root / "data/training_leakage_free"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = root / "outputs/eval_inputs"
    eval_dir.mkdir(parents=True, exist_ok=True)

    topic_counts: Counter[str] = Counter()
    unmapped: set[str] = set()
    stats: dict[str, Any] = {}

    for split in ["train", "validation", "test"]:
        samples = load_jsonl(root / f"data/training/{split}.jsonl")
        constrained_out: list[dict[str, Any]] = []
        no_evidence_out: list[dict[str, Any]] = []
        global_out: list[dict[str, Any]] = []
        for old in samples:
            meta = old["metadata"]
            title = meta.get("topic_title", "")
            risk_topic = TOPIC_TITLE_MAP.get(title)
            if risk_topic is None:
                unmapped.add(title)
                risk_topic = "其他财务风险"
            topic_counts[risk_topic] += 1
            query = build_query(risk_topic)

            key = (str(meta["stock_code"]), int(meta["report_year"]))
            top3 = [c for c, _ in indexes[key].top_n(query, 3)]
            constrained_out.append(
                build_sample(old, risk_topic, top3, "leakage_free_bm25_top3")
            )
            if split == "test":
                no_evidence_out.append(build_sample(old, risk_topic, [], "no_evidence"))
                gtop3 = [c for c, _ in global_index.top_n(query, 3)]
                global_out.append(
                    build_sample(old, risk_topic, gtop3, "leakage_free_bm25_top3_global")
                )
        write_jsonl(out_dir / f"{split}.jsonl", constrained_out)
        stats[split] = len(constrained_out)
        if split == "test":
            write_jsonl(eval_dir / "test_no_evidence.jsonl", no_evidence_out)
            write_jsonl(eval_dir / "test_global_evidence.jsonl", global_out)

    print("split_sizes:", stats)
    print("topic_counts:", dict(topic_counts))
    if unmapped:
        print("UNMAPPED topic_titles:", sorted(unmapped))
    else:
        print("all topic_titles mapped")


if __name__ == "__main__":
    main()
