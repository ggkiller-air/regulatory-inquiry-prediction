# 实验自动指标汇总（第二阶段）

生成日期：2026-07-17
本阶段完成：无泄漏数据重建、新 QLoRA 训练、主实验与消融实验推理、ROUGE-L 与 BERTScore-F1 计算、一致性检查。
关键问点 F1、证据支持率、幻觉率留待第三阶段人工复核，本文件中不给出任何未验证数值。

---

## 1. 风险主题标准化规则

- 8 类标准主题：收入确认与经营真实性 / 应收账款与信用风险 / 资产减值与商誉 / 关联交易与资金占用 / 成本费用与盈利质量 / 现金流与持续经营 / 公司治理与信息披露 / 其他财务风险。
- 44 个原始 `topic_title` 全部按其所属会计/监管领域映射（映射表硬编码于 `src/data_pipeline/leakage_free.py:TOPIC_TITLE_MAP`），不含任何公司事实、金额、主体或监管措辞。
- 全量 157 条样本主题分布：收入确认与经营真实性 55、资产减值与商誉 32、应收账款与信用风险 27、现金流与持续经营 26、关联交易与资金占用 9、成本费用与盈利质量 5、公司治理与信息披露 2、其他财务风险 1。

## 2. 无泄漏检索 query 定义

```
query = 标准主题名 + 该主题的通用财经风险关键词表（见 leakage_free.py:TOPIC_KEYWORDS）
```

- 公司与报告年度仅用于限定 BM25 语料范围（每公司—年度独立索引）；
- 消融 `w/o Company-Year Constraint` 用同一 query 在全部 3732 个年报 chunk 上全局检索（可检索到其他公司年报，实测 test 首条样本 Top-3 全部来自其他公司）；
- query 中不含真实监管子问题、父问题事实描述、回复公告内容及金标事实；
- 已程序化验证：train/validation/test 所有样本的模型输入中不存在金标问询的任何 15 字符片段（滑窗步长 5）。

## 3. 数据与模型产物

| 产物 | 路径 |
| --- | --- |
| 无泄漏 SFT 数据（含风险主题，Top-3 证据） | `data/training_leakage_free/{train,validation,test}.jsonl`（118/16/23，公司级划分与原版一致） |
| 无证据测试输入 | `outputs/eval_inputs/test_no_evidence.jsonl` |
| 全局检索测试输入 | `outputs/eval_inputs/test_global_evidence.jsonl` |
| 新 QLoRA adapter | `outputs/leakage_free_run/adapter/` |
| 训练配置 | `configs/qlora_qwen3_8b_leakage_free.yaml`（与原正式训练超参完全一致：seed=42、3 epochs、lr=2e-4 cosine、有效 batch 16、rank=16/alpha=32、NF4 4-bit、max_length=2560） |
| 训练日志 | `outputs/leakage_free_run/logs/loss_history.jsonl`（train_loss 终值 2.78；eval loss 2.27→2.14→2.11） |

原有 `outputs/full_run/`、`data/training/`、全部旧报告与模型权重均未修改、未覆盖。

## 4. 推理设置（五组实验完全一致）

- 脚本：`src/training/generate.py`（本阶段仅将 `--adapter` 改为可选以支持无微调基线）；
- 解码：贪心（`do_sample=False`），`max_new_tokens=256`，`enable_thinking=False`，Qwen3 chat template；
- system prompt 与用户模板的"要求"部分五组完全一致（定义于 `leakage_free.py`）；
- 测试集：`data/training_leakage_free/test.jsonl` 的 23 条（南京新百 12、杭州柯林 11），与训练公司无交集。

| 实验配置 | adapter | 证据 | 预测文件 |
| --- | --- | --- | --- |
| Qwen3-8B Zero-shot | 无 | 无 | `outputs/eval_predictions/zero_shot.jsonl` |
| Qwen3-8B + Evidence（= w/o QLoRA） | 无 | 无泄漏 Top-3 | `outputs/eval_predictions/base_evidence.jsonl` |
| Qwen3-8B QLoRA（= w/o Evidence） | leakage_free_run | 无 | `outputs/eval_predictions/qlora_no_evidence.jsonl` |
| Full Model | leakage_free_run | 无泄漏 Top-3 | `outputs/eval_predictions/full_model.jsonl` |
| w/o Company-Year Constraint | leakage_free_run | 全局 Top-3 | `outputs/eval_predictions/wo_company_year.jsonl` |

所有配置均含标准化风险主题与公司/代码/年度信息输入。

## 5. 指标计算方式

- **ROUGE-L**：字符级 LCS F1，逐样本计算后宏平均（`src/evaluation/auto_metrics.py`）。
- **BERTScore-F1**：模型 `google-bert/bert-base-chinese`（本地 `models/bert-base-chinese`），bert-score 0.3.12（pip 包版本号 0.3.13），num_layers=12，**不使用 baseline rescaling**，全部样本 F1 宏平均。
- 统一清洗（预测与参考相同）：去除 `<think>` 段、全角标点转半角、去除 markdown 装饰符（`*`/`#`）、去除全部空白、去除开头"监管问询子问题："类标签。
- 逐样本得分：`outputs/eval_predictions/per_sample/*_metrics.jsonl`；汇总：`outputs/eval_predictions/auto_metrics.json`。

## 6. 自动指标结果（n=23，宏平均）

### 主结果表（自动指标部分）

| 方法 | ROUGE-L | BERTScore-F1 |
| --- | ---: | ---: |
| Qwen3-8B Zero-shot | 0.1672 | 0.6937 |
| Qwen3-8B + Evidence | 0.1588 | 0.6885 |
| Qwen3-8B QLoRA | 0.1807 | 0.7103 |
| **Full Model（本文）** | **0.1933** | **0.7188** |

### 消融表（自动指标部分）

| 模型变体 | ROUGE-L | BERTScore-F1 |
| --- | ---: | ---: |
| Full Model | 0.1933 | 0.7188 |
| w/o Evidence | 0.1807 | 0.7103 |
| w/o Company-Year Constraint | 0.1574 | 0.6910 |
| w/o QLoRA | 0.1588 | 0.6885 |

复用规则：`w/o Evidence` = 主表 `Qwen3-8B QLoRA`；`w/o QLoRA` = 主表 `Qwen3-8B + Evidence`（同一预测文件，未重复推理）。
完整表格（含待复核列占位）：`reports/main_results.{csv,md}`、`reports/ablation_results.{csv,md}`。

## 7. 一致性检查（`reports/consistency_checks.txt`）

对全部五组配置程序化验证通过：

- 每组均为 23 条，sample_id 与测试集完全一致且无重复、无缺失；
- 每组 `reference` 与测试集金标逐条一致（同一版参考文本）；
- 金标问询的任何 15 字符片段均未出现在任何配置的模型输入中；
- 五组使用同一解码参数、同一 system prompt、同一清洗规则、同一 BERTScore 模型。

## 8. 残余泄漏与可复现性说明

1. **已消除**：证据检索不再使用真实问询文本；训练数据同步重建并重新训练。
2. **残余（需在论文中说明）**：标准化风险主题由真实问询的 `topic_title` 映射而来，等价于假设"测试时已知该公司该年度的粗粒度风险方向"。主题为 8 类通用类别，不含具体事实，但严格意义上仍源自金标信号。
3. **可复现性**：全部配置可复现（seed=42、贪心解码、脚本与配置入库）。唯一注意点：QLoRA 训练在 4×A800 DDP 下进行，换卡数会改变梯度累积折算（脚本自动保持有效 batch=16，但浮点顺序差异可能导致极小的数值抖动）；`bert-score` 包元数据版本 0.3.13 与模块 `__version__` 0.3.12 不一致，属上游打包问题，已如实记录。
4. 无任何实验配置无法严格复现。
