# 实验产物审计报告（Experiment Artifact Audit）

生成日期：2026-07-16
审计范围：仅检查现有文件与配置，未运行模型、未重新训练、未计算任何新指标。

---

## 1. 数据集文件

| 文件 | 条数 | 说明 |
| --- | ---: | --- |
| `data/training/train.jsonl` | 118 | SFT 训练集，7 家公司（600373/600678/600822/688363/688685/688793/834033） |
| `data/training/validation.jsonl` | 16 | 验证集，2 家公司（603466 风语筑、603922 金鸿顺） |
| `data/training/test.jsonl` | 23 | **测试集**，2 家公司（600682 南京新百 12 条、688611 杭州柯林 11 条） |
| `data/processed/sft_train.jsonl` | 157 | 旧版全量 SFT 导出（Top-5 证据），仅作参考，未用于正式训练 |
| `data/processed/regulatory_questions_clean.jsonl` | 221 | 清洗后正式问询记录（157 条可训练子问题 + 64 条父问题上下文） |
| `data/processed/evidence_candidates.jsonl` | 157 | 每条子问题的 BM25 Top-10 候选证据（含 chunk_id、页码、得分、原文） |
| `data/processed/annual_report_chunks.jsonl` | 3732 | 13 家公司年报分块语料（chunk_size=750，overlap=100，含页码） |
| `data/processed/document_pages.jsonl` | 4610 页 | 全部 41 个 PDF 的逐页抽取文本 |

**公司级划分检查（已验证）**：train/validation/test 三个集合公司完全互斥，`train∩test = ∅`，`val∩test = ∅`。测试公司未出现在训练集中，满足要求。

**样本格式**：每条为 `{sample_id, messages(system/user/assistant), metadata}`。
- `messages[-1]`（assistant）= 真实监管问询子问题（金标参考文本）；
- `metadata` 含 `company`、`stock_code`、`report_year`、`topic_title`（风险主题）、`evidence_chunk_ids`、`evidence_page_ranges`（Top-3）、`source: annual_report_bm25_top3`。

**注意**：原始 PDF 目录（`00_project_docs/`、`01_metadata/`、`02_raw_pdf/`）**不在本机**，只有处理后的 `document_pages.jsonl` 与 `annual_report_chunks.jsonl`。所有重检索只能基于现有 chunk 文件进行（可行，无需原始 PDF）。

---

## 2. 测试集真实监管问询文本

- 位置一：`data/training/test.jsonl` 每条 `messages[-1].content`；
- 位置二：`outputs/full_run/predictions/test_predictions.jsonl` 的 `reference` 字段（两者一致，逐条含 `sample_id`）。

---

## 3. Prompt 模板与输入构成

模板定义在 `src/data_pipeline/sft_export.py`：

- system：固定财经监管助手指令；
- user：`公司 + 证券代码 + 报告年度 + Top-3 年报证据（含页码与 chunk_id）+ 3 条生成要求`；
- assistant：清洗后的金标子问题。

**关键事实：当前 prompt 中不包含风险主题（`topic_title`）**。`topic_title` 只存在于 `metadata`，训练和推理输入均未使用。

> 影响：论文定义的 "Full Model =QLoRA + Top-3 证据 + 风险主题" 与已训练模型的实际输入（QLoRA + Top-3 证据，无风险主题）不一致。若主表 Full Model 要包含风险主题输入，该配置**没有对应的现成预测结果**，且模型训练时也未见过该字段；消融项 "w/o Risk Topic" 与已有 `full_run` 预测才是同一配置。需在第二阶段决策（见 §9）。

---

## 4. 证据检索（`src/data_pipeline/evidence_alignment.py`）

- 方法：自实现 BM25（k1=1.5，b=0.75），中文 bigram + 数字/字母 token；
- 索引：**按（公司，年度）分别建索引**，即检索天然带公司—年度约束；
- Top-K：候选保留 Top-10，训练/测试输入使用 Top-3；
- 检索质量已有指标（`reports/retrieval_summary.md`，基于 14 条人工对齐父问题）：Recall@1/3/5/10 均为 1.0000。

**重大风险点（必须在第二阶段处理）**：检索 query 由
`topic_title + 父问题背景（"请说明"前的叙述部分）+ 金标子问题全文` 拼接而成（`build_query()`，`evidence_alignment.py:221`）。
即**测试集的 Top-3 证据是用真实监管问询文本检索出来的**。虽然问询文本本身没有直接进入模型输入，但证据选择过程泄漏了金标信息，违反"不得将真实监管问询作为模型输入"的实验约束精神。若严格执行，需用不含金标子问题的 query（如仅 topic_title + 公司年度信息，或父问题背景）对测试集重新检索并重新生成所有含证据的实验。

---

## 5. 模型权重

| 产物 | 位置 | 状态 |
| --- | --- | --- |
| 基座 Qwen3-8B | `models/Qwen3-8B/`（16 GB，5 个 safetensors 分片完整，含 tokenizer/config） | ✅ 可用 |
| QLoRA adapter（正式） | `outputs/full_run/adapter/`（rank=16，alpha=32，目标 7 组投影层） | ✅ 可用，**不得覆盖** |
| QLoRA checkpoints | `outputs/full_run/checkpoints/checkpoint-16, checkpoint-24` | ✅ 保留 |
| dry-run adapter | `outputs/dry_run/adapter/`（10 样本 1 epoch，仅验证流程） | 仅流程验证，不用于论文 |

---

## 6. 已有预测结果（与主表/消融表配置对照）

| 论文配置 | 对应现有产物 | 状态 |
| --- | --- | --- |
| 主表 1：Qwen3-8B Zero-shot（无微调、无证据） | 无 | ❌ 需第二阶段生成 |
| 主表 2：Qwen3-8B + Evidence（无微调、Top-3 证据） | 无 | ❌ 需第二阶段生成 |
| 主表 3：Qwen3-8B QLoRA（微调、无证据） | 无 | ❌ 需第二阶段生成 |
| 主表 4：Full Model（QLoRA + Top-3 证据；如含风险主题则更无对应产物） | `outputs/full_run/predictions/test_predictions.jsonl`（23 条，QLoRA + Top-3 证据，**无风险主题**） | ⚠️ 部分对应，取决于 Full Model 定义与泄漏处理决策 |
| 消融 w/o Evidence | 无（同主表 3 或需另生成，视风险主题决策） | ❌ |
| 消融 w/o Risk Topic | = 现有 `full_run` 预测（若 Full Model 定义含风险主题） | ⚠️ 可复用 |
| 消融 w/o Company-Year Constraint | 无（需在全库 3732 chunks 上重建全局 BM25 并重新生成） | ❌ |
| 消融 w/o QLoRA | = 主表 2（配置完全相同，可直接复用） | ❌（随主表 2 生成） |

现有预测文件字段：`{sample_id, company, stock_code, report_year, reference, prediction}`，23 条，sample_id 无重复无缺失（与 test.jsonl 一一对应）。

另有 `outputs/dry_run/predictions/validation_dry_run.jsonl`（2 条），仅流程验证，不用于论文。

---

## 7. 已有指标与报告

| 报告 | 内容 | 可否直接用于论文表 |
| --- | --- | --- |
| `reports/retrieval_summary.md` | 检索 Recall@1/3/5/10 = 1.0（14 条人工对齐） | 可引用，但非主表指标 |
| `reports/full_training_summary.md` | train loss 5.79→1.85，eval loss 2.21→2.07，训练设置 | 训练细节可引用 |
| `outputs/full_run/logs/loss_history.jsonl` + `loss_curve.png` | 完整 loss 日志 | 同上 |
| `reports/question_quality_summary.md` | 数据清洗统计（157 条可训练子问题，来源 inquiry 19 / reply 202） | 数据集描述可引用 |
| `reports/evidence_review.xlsx` | 14 条父问题级人工证据对齐 | 可作证据支持率复核起点 |
| ROUGE-L / BERTScore / 关键问点 F1 / 证据支持率 / 幻觉率 | **全部不存在** | ❌ 需第二/三阶段计算 |

---

## 8. 随机种子、解码参数、环境

- 训练：seed=42，epochs=3，lr=2e-4 cosine，有效 batch 16，max_length=2560，4-bit NF4 QLoRA（`configs/qlora_qwen3_8b.yaml`，`outputs/full_run/training_config.yaml` 一致）。
- 推理（`src/training/generate.py`）：**贪心解码**（`do_sample=False`，temperature/top_p/top_k=None），`max_new_tokens=256`，`enable_thinking=False`，chat template 生成。现有 `full_run` 预测即用此设置。
- 环境：Python 3.11.15，torch 2.8.0+cu128，transformers 5.13.1，peft 0.19.1，bitsandbytes 0.49.2；4 × A800-80GB 当前空闲。
- **评测依赖缺失**：`rouge_score`、`bert_score`、`jieba` 未安装（第二阶段需经 uv 安装；注意本机代理/镜像约束）。
- `generate.py` 现在**强制要求 --adapter**，跑无微调基线需要小改脚本（增加可选 adapter），不影响已有权重。

---

## 9. 第二阶段前需要确认的决策点

1. **检索泄漏**：测试集 Top-3 证据由金标问询文本检索而来（§4）。是否用去泄漏 query（建议：仅 `topic_title` + 公司年度，不含金标子问题文本）对测试集重新检索，并以此重跑所有含证据实验？（严格做法=是；但这会使 Full Model 输入与训练时的证据分布略有差异。）
2. **Full Model 是否含风险主题输入**：已训练模型 prompt 不含 `topic_title`（§3）。两种方案：
   - A. 定义 Full Model = QLoRA + Top-3 证据（不含风险主题），消融表删除/重定义 "w/o Risk Topic"；
   - B. 推理时在 prompt 中加入 `topic_title`（模型训练时未见过该格式，可复用现有 QLoRA 权重，但 Full Model 需新生成预测，"w/o Risk Topic" 复用与否取决于泄漏决策）。
   注意：`topic_title`（如"关于公司经营情况"）本身抽取自真实问询函标题，将其作为测试输入同样存在轻度金标信息泄漏，需在论文中说明或改用独立标注的风险主题。
3. **Zero-shot / 无证据 prompt 统一**：需确定去掉证据后的统一模板（保留公司、代码、年度与任务指令，其余不变），所有无证据配置共用。

## 10. 结论

- 可直接复用：测试集（23 条）、QLoRA adapter、基座模型、年报 chunk 语料、BM25 检索代码、贪心解码设置、`full_run` 测试集预测（作为某一配置的结果）。
- 必须新生成（推理，不训练）：主表 1/2/3 及消融 w/o Company-Year Constraint；Full Model 与 w/o Risk Topic 视 §9 决策。
- 必须新计算：全部五项指标；其中关键问点 F1、证据支持率、幻觉率需第三阶段人工复核表支撑。
