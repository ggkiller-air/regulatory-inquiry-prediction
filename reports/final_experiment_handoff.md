# 论文实验交接报告（一作参考版）

生成日期：2026-07-17
用途：为论文实验部分提供可直接引用的设置说明、结果表、指标定义与人工复核材料。
**重要**：关键问点 F1、证据支持率、幻觉率尚未人工确认，结果表中一律标注"待人工复核"，不呈现任何未经确认的数值；复核候选标注见两个 xlsx 复核表。

---

## 1. 数据集与训练设置摘要

- 数据来源：13 家 A 股上市公司 2024 年年度报告及交易所监管问询/回复公告（41 个 PDF）。清洗后得到 157 条正式监管问询子问题（`reports/question_quality_summary.md`）。
- 划分（公司级，无公司跨集合，测试公司不在训练集中）：train 118 条 / 7 家，validation 16 条 / 2 家，test 23 条 / 2 家（南京新百 12、杭州柯林 11）。
- 任务形式：输入 = 公司信息 + 标准化风险主题 +（可选）Top-3 年报证据；输出 = 一条监管问询子问题；金标 = 真实问询子问题。
- 微调：Qwen3-8B，4-bit NF4 QLoRA（rank=16，alpha=32，dropout=0.05，7 组投影层），seed=42，3 epochs，lr=2e-4 cosine，有效 batch 16，max_length=2560，4×A800 DDP。adapter：`outputs/leakage_free_run/adapter/`；eval loss 2.27→2.11。
- 推理（五组配置完全一致）：贪心解码（do_sample=False），max_new_tokens=256，enable_thinking=False，同一 system prompt 与生成要求。

## 2. 无泄漏检索方法

- 早期版本的证据检索曾以真实问询文本作 query（存在标签泄漏），本轮已全部重建。
- 新检索 query = 标准化风险主题名 + 该主题通用财经风险关键词（词表见 `src/data_pipeline/leakage_free.py:TOPIC_KEYWORDS`，只含通用会计/监管术语）。
- 检索器：自实现 BM25（k1=1.5，b=0.75，中文 bigram），按（公司，年度）独立建索引，取 Top-3；消融 `w/o Company-Year Constraint` 在全部 3732 个年报 chunk 上全局检索。
- 程序化验证：train/validation/test 全部样本的模型输入中不存在金标问询的任何 15 字符片段（滑窗步长 5），见 `reports/consistency_checks.txt`。

## 3. 风险主题标准化方法

44 个原始问询主题标题按会计/监管领域映射为 8 类粗粒度主题（收入确认与经营真实性 55、资产减值与商誉 32、应收账款与信用风险 27、现金流与持续经营 26、关联交易与资金占用 9、成本费用与盈利质量 5、公司治理与信息披露 2、其他财务风险 1）。标准主题不含具体事实、金额、主体或监管措辞。
**论文需说明的残余假设**：标准主题由真实问询的主题标题映射而来，等价于假设"测试时已知该公司该年度的粗粒度风险方向"。

## 4. 评测协议（multi-reference）

在无泄漏设定下，同一（公司，风险主题）的多条子问题模型输入完全相同，贪心解码输出也相同。因此 23 条测试子问题按相同模型输入合并为 **7 个输入组**（南京新百 4 组、杭州柯林 3 组，每组 1–5 条参考问询）。每组取该组预测与组内各参考问询的最高 ROUGE-L / BERTScore-F1，再对 7 组宏平均。分组明细见 `outputs/eval_predictions/group_metrics.json`。

### 主结果表（7 组 multi-reference，覆盖 23 条子问题）

| 方法 | ROUGE-L | BERTScore-F1 | 关键问点F1 | 证据支持率 | 幻觉率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B Zero-shot | 0.2118 | 0.7271 | 待人工复核 | 待人工复核 | 待人工复核 |
| Qwen3-8B + Evidence | 0.2030 | 0.7208 | 待人工复核 | 待人工复核 | 待人工复核 |
| Qwen3-8B QLoRA | 0.2587 | **0.7520** | 待人工复核 | 待人工复核 | 待人工复核 |
| Full Model（本文） | **0.2592** | 0.7473 | 待人工复核 | 待人工复核 | 待人工复核 |

## 5. 消融实验表（同一协议）

| 模型变体 | ROUGE-L | BERTScore-F1 | 关键问点F1 | 证据支持率 | 幻觉率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full Model | **0.2592** | 0.7473 | 待人工复核 | 待人工复核 | 待人工复核 |
| w/o Evidence | 0.2587 | **0.7520** | 待人工复核 | 待人工复核 | 待人工复核 |
| w/o Company-Year Constraint | 0.1964 | 0.7257 | 待人工复核 | 待人工复核 | 待人工复核 |
| w/o QLoRA | 0.2030 | 0.7208 | 待人工复核 | 待人工复核 | 待人工复核 |

`w/o Evidence` 与主表 `Qwen3-8B QLoRA`、`w/o QLoRA` 与主表 `Qwen3-8B + Evidence` 为同一预测文件，数值完全一致（未重复推理）。

## 6. 指标定义与计算方式

| 指标 | 定义 | 状态 |
| --- | --- | --- |
| ROUGE-L | 字符级 LCS F1；multi-reference：每输入组取预测与组内各参考的最高值，再对 7 组宏平均。预测与参考经同一清洗（去 `<think>`、全角转半角、去 markdown 符、去空白、去开头标签） | ✅ 自动，已确认 |
| BERTScore-F1 | `google-bert/bert-base-chinese`（本地权重），num_layers=12，无 baseline rescaling，bert-score 0.3.12；multi-reference 组内最高匹配后对 7 组宏平均 | ✅ 自动，已确认 |
| 关键问点F1 | 五维问点（财务事项/监管对象、监管动作、核查问题与风险判断、时间主体范围限定、实体）的 Macro-F1。复核表已给出规则候选标注，**数值未经人工确认，本报告不呈现** | 待人工复核 |
| 证据支持率 | 有证据支撑的生成问点数 / 全部生成问点数（监管动作动词不计入分母）。有证据输入的方法核验其输入 Top-3 证据；无证据输入的方法按约定核验该公司该年度年报全文，复核表 `evidence_input_received` 列区分两种情形，**跨方法核验依据不同，复核时注意** | 待人工复核 |
| 幻觉率 | 含至少一项无证据事实的样本数 / 测试样本数。复核表已含金额/年份/主体三类机械预检结果；"与年报方向相反的表述"必须人工判断。问点方向偏移不计为幻觉 | 待人工复核 |

## 7. 主结果表客观分析（3–5 条）

1. Full Model 取得最高 ROUGE-L（0.2592），但与不加证据的 QLoRA 变体（0.2587）基本持平（差距 0.0005）；BERTScore-F1 上 QLoRA（0.7520）反而略高于 Full Model（0.7473）。在 multi-reference 协议下，证据输入对微调模型的增益不明显。
2. QLoRA 领域微调是最主要的性能来源：两个微调配置（0.2587/0.2592）明显高于全部未微调配置（≤0.2118），BERTScore 同趋势。
3. 对未微调基座，直接加入证据未产生增益：ROUGE-L 由 0.2118 降至 0.2030，BERTScore 由 0.7271 降至 0.7208，基座倾向复述证据细节而偏离监管问询文体。
4. 两项自动指标只衡量与参考问询的表面/语义相似度；证据是否改善问点的事实锚定与可支持性，需以人工复核后的证据支持率和幻觉率判断，目前不应下结论。
5. 测试集仅 23 条子问题、合并后 7 个输入组、2 家公司，样本量很小，结论应使用"表明""显示出一定优势"等审慎表述。

## 8. 消融表客观分析（3–5 条）

1. 移除 QLoRA 的降幅最大（ROUGE-L −0.0562，BERTScore −0.0265），表明领域微调是当前方法最主要的性能来源。
2. 移除证据后 ROUGE-L 基本不变（−0.0005），BERTScore 反而 +0.0047：在 multi-reference 协议下，公司—年度约束证据对自动指标的边际贡献很小，其价值需结合人工复核的事实性指标评估。
3. 取消公司—年度约束后 ROUGE-L 降至 0.1964，为全部变体最低，实测部分测试组 Top-3 证据全部来自其他公司，显示检索范围控制对财经长文档场景较为重要。
4. w/o Company-Year Constraint 低于 w/o Evidence（0.1964 vs 0.2587），表明错误公司的证据比没有证据更有害。
5. 由于 7 个输入组的样本量极小，上述差异均未做显著性检验，论文中应避免过强表述。

## 9. 论文实验部分中文结果描述草稿（自动指标部分，可直接改写引用）

> 由于同一公司同一风险主题下的多条真实问询子问题对应完全相同的模型输入，我们将 23 条测试子问题按输入合并为 7 个评测组，每组以组内全部真实子问题为多参考（multi-reference），取最高匹配后对组宏平均。表 X 给出了各方法的结果。经 QLoRA 领域微调的两个配置明显优于未微调基线：Full Model 取得最高的 ROUGE-L（0.2592，较 Zero-shot 提升 0.0474），QLoRA（无证据）取得最高的 BERTScore-F1（0.7520）。两个微调配置之间差距很小，表明在当前小规模测试集上，领域微调是最主要的性能来源，而检索证据对自动指标的边际贡献有限。对未经微调的基座模型，直接输入证据反而使 ROUGE-L 由 0.2118 降至 0.2030，其输出倾向于复述证据细节而偏离监管问询的文体与关注方式，显示证据利用能力需要通过领域微调获得。消融实验（表 Y）显示，取消检索的公司—年度约束后 ROUGE-L 降至 0.1964，为全部变体最低且低于完全不使用证据的变体，表明在财经长文档场景下控制检索范围比单纯提供更多文本更为重要。由于测试集规模有限（7 组、23 条子问题、2 家公司），上述结果主要反映方法的相对趋势。
>
> （关键问点 F1、证据支持率与幻觉率待人工复核完成后补充；证据输入的实际价值需结合这三项事实性指标综合评估。）

## 10. 数值确认状态

| 内容 | 状态 |
| --- | --- |
| ROUGE-L、BERTScore-F1（multi-reference，全部 5 组） | ✅ 自动指标，已确认，可直接用于论文 |
| 关键问点 F1 | 未确认。复核表 `reports/key_point_annotation.xlsx` 已含规则候选标注（115 行全部 `pending_review`），数值不呈现在任何结果表中 |
| 证据支持率 | 未确认。复核表同上，核验依据跨方法不同，复核时须结合 `evidence_input_received` 列 |
| 幻觉率 | 未确认。复核表 `reports/factuality_annotation.xlsx` 已含金额/年份/主体机械预检（未发现该三类幻觉），方向相反类陈述未覆盖，须人工判断 |
| 检索 Recall@K =1.0（14 条人工对齐） | 基于旧版（含泄漏 query）检索的历史指标，论文引用需注明或按新检索重新验证 |

人工复核流程建议：在两个 xlsx 中逐行修正 `matched_key_points` / `supported_key_points` / `contains_hallucination` 等列，将 `reviewer_status` 改为 `confirmed`，然后重新计算宏平均即可（每方法 23 行；由于每方法仅 7 个不同输出，实际需精读的预测文本约 7×5=35 条）。

## 11. 文件路径清单

| 类别 | 路径 |
| --- | --- |
| 测试集（无泄漏，含风险主题） | `data/training_leakage_free/test.jsonl`（无证据/全局证据输入：`outputs/eval_inputs/test_no_evidence.jsonl`、`outputs/eval_inputs/test_global_evidence.jsonl`） |
| 训练/验证集 | `data/training_leakage_free/{train,validation}.jsonl` |
| 新 QLoRA adapter | `outputs/leakage_free_run/adapter/`（训练日志 `outputs/leakage_free_run/logs/`） |
| 五组预测 | `outputs/eval_predictions/{zero_shot,base_evidence,qlora_no_evidence,full_model,wo_company_year}.jsonl` |
| Multi-reference 分组指标（论文所用） | `outputs/eval_predictions/group_metrics.json`（含 7 组的公司、主题、sample_id、逐组最高分） |
| 逐样本单参考指标（参考用，非论文口径） | `outputs/eval_predictions/per_sample/*_metrics.jsonl`；汇总 `outputs/eval_predictions/auto_metrics.json` |
| 规则预标注汇总（仅供复核参考，非结果） | `outputs/eval_predictions/preliminary_manual_metrics.json` |
| 人工复核表 | `reports/key_point_annotation.xlsx`（115 行）、`reports/factuality_annotation.xlsx`（115 行） |
| 结果表 | `reports/main_results.{md,csv}`、`reports/ablation_results.{md,csv}` |
| 一致性检查 | `reports/consistency_checks.txt` |
| 阶段报告 | `reports/experiment_artifact_audit.md`（产物审计）、`reports/experiment_metrics_summary.md`（第二阶段自动指标） |
| 评估脚本 | `src/evaluation/{auto_metrics,annotation_tables,build_tables}.py`；数据重建 `src/data_pipeline/leakage_free.py` |

旧版产物（`outputs/full_run/`、`data/training/`、原始 processed 数据）全部保留未动。
