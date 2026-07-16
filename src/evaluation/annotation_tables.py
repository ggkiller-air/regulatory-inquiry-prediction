"""第三阶段：关键问点与事实性人工复核表（机器预标注版）。

所有行 reviewer_status=pending_review：本脚本产生的匹配与判断只是规则预标注，
用于辅助人工复核，不构成人工评审结果。

关键问点五个维度：
1. 财务事项或监管对象（fin_item / entity）
2. 监管动作（action）
3. 具体核查问题 + 风险判断或披露要求（risk/whether 短语）
4. 时间、主体或范围限定（time/scope）
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .auto_metrics import clean_text

# ---------------------------------------------------------------------------
# 词典与抽取规则
# ---------------------------------------------------------------------------

ACTIONS = [
    "补充披露", "披露", "说明", "列示", "核实", "核查", "自查", "量化分析", "论证", "对比分析",
]

FIN_ITEMS = [
    "营业收入", "主营业务收入", "收入确认", "销售退回", "境外收入", "线下销售收入", "毛利率",
    "净利润", "扣非净利润", "经营业绩", "业绩波动", "盈利质量",
    "销售费用", "管理费用", "研发费用", "财务费用", "期间费用", "营业成本", "中介机构费", "咨询费",
    "应收账款", "长期应收款", "其他应收款", "预付款项", "应收款项", "坏账准备", "账龄", "期后回款",
    "存货", "跌价准备", "商誉", "减值测试", "减值准备", "资产减值", "可收回金额",
    "固定资产", "在建工程", "长期股权投资", "无形资产", "开发支出",
    "货币资金", "受限资金", "现金流", "经营活动现金流", "借款", "有息负债", "偿债能力", "流动性",
    "委托理财", "投资理财", "理财产品", "募集资金", "募投项目", "对外投资",
    "关联交易", "关联方", "资金占用", "对外担保", "往来款", "非经营性资金往来", "内部控制",
    "会计差错", "会计政策", "会计估计", "前期差错", "审计程序", "关键审计事项",
    "分红", "现金分红", "未分配利润", "留存收益", "利润分配",
    "客户", "供应商", "存储费", "检测制备费", "商业实质", "持续经营",
]

TIME_SCOPE = [
    "报告期", "近三年", "近两年", "期末", "期初", "分季度", "分月度", "分业务", "分板块",
    "分产品", "分行业", "分地区", "同行业可比公司", "前五大", "前五名",
]

WHETHER_RE = re.compile(r"是否[^，。;；、？,.;?()<>\s]{2,28}")

# 风险判断问点的证据锚定词干（比 FIN_ITEMS 更宽松，仅用于支持率预标注）
RISK_ANCHOR_STEMS = [
    "收入", "成本", "费用", "毛利", "利润", "资金", "存货", "商誉", "减值", "应收",
    "预付", "担保", "关联", "理财", "借款", "分红", "现金", "客户", "供应商", "会计",
]
RATIONALE_RE = re.compile(
    r"(?:合理性|真实性|准确性|充分性|必要性|公允性|可回收性|可实现性|谨慎性)"
)
YEAR_RE = re.compile(r"20\d{2}")
AMOUNT_RE = re.compile(r"\d[\d,，.]*\s*(?:万元|亿元|万股|亿股|%|个百分点)")
ENTITY_RE = re.compile(r"[一-鿿]{2,12}(?:股份有限公司|有限公司|集团|干细胞|香港)")


def extract_key_points(text: str, company: str) -> dict[str, list[str]]:
    """返回 {维度: [问点]}，问点为去重后的短语。"""
    t = clean_text(text)
    points: dict[str, list[str]] = {
        "fin_item": [],
        "action": [],
        "risk_or_disclosure": [],
        "time_scope": [],
        "entity": [],
    }
    for item in FIN_ITEMS:
        if item in t:
            points["fin_item"].append(item)
    for action in ACTIONS:
        if action in t:
            points["action"].append(action)
    points["risk_or_disclosure"] = list(dict.fromkeys(WHETHER_RE.findall(t)))
    points["risk_or_disclosure"] += list(dict.fromkeys(RATIONALE_RE.findall(t)))
    for scope in TIME_SCOPE:
        if scope in t:
            points["time_scope"].append(scope)
    points["time_scope"] += list(dict.fromkeys(YEAR_RE.findall(t)))
    entities = [e for e in dict.fromkeys(ENTITY_RE.findall(t))]
    if company and company in t:
        entities.append(company)
    points["entity"] = entities
    # "披露"包含于"补充披露"时去重
    if "补充披露" in points["action"] and "披露" in points["action"]:
        points["action"].remove("披露")
    return {k: list(dict.fromkeys(v)) for k, v in points.items()}


def flatten(points: dict[str, list[str]]) -> list[str]:
    return [f"{dim}:{p}" for dim, plist in points.items() for p in plist]


def match_points(ref: list[str], pred: list[str]) -> list[str]:
    """预标注匹配：同维度下相同或互为子串即算匹配（供人工修正）。"""
    matched = []
    for rp in ref:
        rdim, rtext = rp.split(":", 1)
        for pp in pred:
            pdim, ptext = pp.split(":", 1)
            if rdim != pdim:
                continue
            if rtext == ptext or rtext in ptext or ptext in rtext:
                matched.append(rp)
                break
    return matched


def prf(ref: list[str], pred: list[str], matched: list[str]) -> tuple[float, float, float]:
    if not pred and not ref:
        return 1.0, 1.0, 1.0
    matched_pred = 0
    for pp in pred:
        pdim, ptext = pp.split(":", 1)
        for rp in ref:
            rdim, rtext = rp.split(":", 1)
            if pdim == rdim and (rtext == ptext or rtext in ptext or ptext in rtext):
                matched_pred += 1
                break
    precision = matched_pred / len(pred) if pred else 0.0
    recall = len(matched) / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------

METHODS = {
    "zero_shot": "Qwen3-8B Zero-shot",
    "base_evidence": "Qwen3-8B + Evidence",
    "qlora_no_evidence": "Qwen3-8B QLoRA",
    "full_model": "Full Model",
    "wo_company_year": "w/o Company-Year Constraint",
}

INPUT_FILES = {
    "zero_shot": "outputs/eval_inputs/test_no_evidence.jsonl",
    "base_evidence": "data/training_leakage_free/test.jsonl",
    "qlora_no_evidence": "outputs/eval_inputs/test_no_evidence.jsonl",
    "full_model": "data/training_leakage_free/test.jsonl",
    "wo_company_year": "outputs/eval_inputs/test_global_evidence.jsonl",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    chunks = load_jsonl("data/processed/annual_report_chunks.jsonl")
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    corpus_by_report: dict[tuple[str, int], str] = {}
    for c in chunks:
        key = (str(c["stock_code"]), int(c["report_year"]))
        corpus_by_report[key] = corpus_by_report.get(key, "") + clean_text(c["text"])

    kp_rows: list[dict[str, Any]] = []
    fact_rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}

    for method, method_label in METHODS.items():
        preds = load_jsonl(f"outputs/eval_predictions/{method}.jsonl")
        inputs = {s["sample_id"]: s for s in load_jsonl(INPUT_FILES[method])}
        f1s, support_num, support_den, halluc_samples = [], 0, 0, 0

        for r in preds:
            sid = r["sample_id"]
            meta = inputs[sid]["metadata"]
            company, year = r["company"], int(r["report_year"])
            risk_topic = meta["risk_topic"]
            has_evidence_input = meta["evidence_top_k"] > 0
            evidence_ids = meta["evidence_chunk_ids"]
            evidence_pages = meta["evidence_page_ranges"]

            ref_points = flatten(extract_key_points(r["reference"], company))
            pred_points = flatten(extract_key_points(r["prediction"], company))
            matched = match_points(ref_points, pred_points)
            precision, recall, f1 = prf(ref_points, pred_points, matched)
            f1s.append(f1)

            kp_rows.append(
                {
                    "method": method_label,
                    "sample_id": sid,
                    "company": company,
                    "report_year": year,
                    "risk_topic": risk_topic,
                    "reference_question": r["reference"],
                    "predicted_question": r["prediction"],
                    "reference_key_points": " ；".join(ref_points),
                    "predicted_key_points": " ；".join(pred_points),
                    "matched_key_points": " ；".join(matched),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "reviewer_status": "pending_review",
                    "reviewer_notes": "",
                }
            )

            # ---------------- 事实性 / 证据支持 ----------------
            # 支持判断依据：有证据输入 → 输入证据文本；无证据输入 → 该公司该年度全部年报文本（外部核验）。
            evidence_text = (
                "".join(clean_text(chunk_by_id[cid]["text"]) for cid in evidence_ids)
                if has_evidence_input
                else corpus_by_report[(str(r["stock_code"]), year)]
            )
            verify_corpus = corpus_by_report.get((str(r["stock_code"]), year), "")

            # 支持率预标注规则：
            # - action（说明/披露等监管动作动词）不计入需证据支持的问点；
            # - fin_item / time_scope：术语在证据中出现即支持；
            # - entity：已知实体词表中的名称在证据中出现即支持；
            # - risk_or_disclosure：风险判断本身是模型推断，只要求其锚定的
            #   财务事项在证据中出现（宽松预标注，最终以人工复核为准）。
            entity_lexicon = [e for e in ENTITY_RE.findall(verify_corpus)] + [company]
            supported, unsupported = [], []
            for pp in pred_points:
                dim, ptext = pp.split(":", 1)
                if dim == "action":
                    continue
                if dim == "risk_or_disclosure":
                    anchors = [it for it in FIN_ITEMS if it in ptext]
                    stems = [s for s in RISK_ANCHOR_STEMS if s in ptext]
                    ok = any(a in evidence_text for a in anchors) or any(
                        s in evidence_text for s in stems
                    )
                elif dim == "entity":
                    known = [e for e in entity_lexicon if e in ptext]
                    ok = any(e in evidence_text for e in known)
                else:
                    ok = ptext in evidence_text
                (supported if ok else unsupported).append(pp)
            n_points = len(supported) + len(unsupported)
            support_num += len(supported)
            support_den += n_points
            rate = len(supported) / n_points if n_points else None

            # 幻觉预检：金额 / 年份 / 主体名是否能在该公司年报全文中找到
            pred_clean = clean_text(r["prediction"])
            halluc_items, halluc_types = [], set()
            for amt in AMOUNT_RE.findall(pred_clean):
                if amt.replace("，", ",") not in verify_corpus:
                    halluc_items.append(amt)
                    halluc_types.add("amount")
            for yr in set(YEAR_RE.findall(pred_clean)):
                if yr not in verify_corpus and int(yr) != year:
                    halluc_items.append(yr)
                    halluc_types.add("year")
            for ent in set(ENTITY_RE.findall(pred_clean)):
                # 贪婪前缀可能把动词短语并入实体，只要匹配串内含任一已知实体即不算幻觉
                if ent in verify_corpus or company in ent:
                    continue
                if any(known in ent for known in entity_lexicon):
                    continue
                halluc_items.append(ent)
                halluc_types.add("entity")
            contains_halluc = bool(halluc_items)
            halluc_samples += contains_halluc

            fact_rows.append(
                {
                    "method": method_label,
                    "sample_id": sid,
                    "company": company,
                    "report_year": year,
                    "risk_topic": risk_topic,
                    "evidence_input_received": has_evidence_input,
                    "evidence_chunks": " ；".join(evidence_ids) if has_evidence_input else "(无证据输入；核验依据=该公司该年度年报全文)",
                    "evidence_page_ranges": " ；".join(evidence_pages) if has_evidence_input else "",
                    "predicted_question": r["prediction"],
                    "generated_key_points": " ；".join(pred_points),
                    "supported_key_points": " ；".join(supported),
                    "unsupported_key_points": " ；".join(unsupported),
                    "evidence_support_rate": round(rate, 4) if rate is not None else "",
                    "contains_hallucination": contains_halluc,
                    "hallucination_type": ",".join(sorted(halluc_types)),
                    "hallucinated_text": " ；".join(halluc_items),
                    "factuality_basis": "输入Top-3证据" if has_evidence_input else "公司—年度年报全文(外部核验)",
                    "reviewer_status": "pending_review",
                    "reviewer_notes": "",
                }
            )

        summary[method] = {
            "method": method_label,
            "n_samples": len(preds),
            "key_point_macro_f1": sum(f1s) / len(f1s),
            "evidence_support_rate": support_num / support_den if support_den else None,
            "hallucination_rate": halluc_samples / len(preds),
        }

    pd.DataFrame(kp_rows).to_excel("reports/key_point_annotation.xlsx", index=False)
    pd.DataFrame(fact_rows).to_excel("reports/factuality_annotation.xlsx", index=False)
    Path("outputs/eval_predictions/preliminary_manual_metrics.json").write_text(
        json.dumps(
            {
                "status": "Preliminary — pending human verification",
                "note": "规则预标注（子串匹配+词典抽取），未经人工确认，不得作为最终论文数值",
                "results": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for m, s in summary.items():
        print(
            f"{m:20s} KP-MacroF1={s['key_point_macro_f1']:.4f} "
            f"Support={s['evidence_support_rate']:.4f} Halluc={s['hallucination_rate']:.4f}"
        )
    print(f"kp_rows={len(kp_rows)} fact_rows={len(fact_rows)}")


if __name__ == "__main__":
    main()
