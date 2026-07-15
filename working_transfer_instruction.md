# 面向监管问询预测的财经文本大模型微调项目交接说明

## 1. 项目目标

本项目按课程大作业/约稿项目的标准推进，目标是完成一套可复现的低资源财经文本微调流程。

核心任务：

> 输入上市公司年度报告中的相关财务与经营证据，输出监管机构可能提出的监管问询问题。

最终计划采用：

- 基座模型：Qwen3-8B
- 微调方式：LoRA / 4-bit QLoRA
- 数据来源：上市公司年度报告、监管问询函、问询函回复公告及辅助核查材料
- 正式模型输入：年度报告证据
- 监督目标：监管问询问题
- 不允许作为模型输入：公司回复正文、会计师回复正文、券商核查意见正文

---

## 2. 当前项目目录

项目根目录：

```text
面向监管问询预测的财经文本大模型微调方法研究/
├── 00_project_docs/
├── 01_metadata/
├── 02_raw_pdf/
├── src/
├── configs/
├── data/
│   └── processed/
├── reports/
├── tests/
└── models/
    └── Qwen3-8B/        # 后续下载，本目录应加入 .gitignore
```

原始数据目录 `02_raw_pdf/` 下共有 13 家公司，每家公司通常包含：

```text
公司目录/
├── annual_report/       # 年度报告
├── inquiry/             # 原始监管问询函，部分缺失
├── reply/               # 公司回复公告，部分缺失
└── supporting/          # 会计师、券商、独董等辅助材料
```

原始文件不得移动、覆盖、删除或改写。

---

## 3. 原始数据概况

当前原始数据统计：

- 公司数量：13
- PDF 数量：41
- 年度报告：13 份
- 独立监管问询函：1 份
- 公司回复公告：11 份
- supporting PDF：16 份
- 缺失说明文件：14 个

13 家公司包括：

- 600373 中文传媒
- 600678 四川金顶
- 600682 南京新百
- 600822 上海物贸
- 603466 风语筑
- 603629 利通电子
- 603922 金鸿顺
- 688363 华熙生物
- 688611 杭州柯林
- 688685 迈信林
- 688793 倍轻松
- 834033 康普化学
- 835174 五新隧装

其中：

- 南京新百有独立原始问询函；
- 其余大多数公司从回复公告中抽取监管问题；
- 利通电子、五新隧装缺少公司回复公告，只能从 supporting 材料中恢复部分问题，因此暂不作为正式训练金标。

---

## 4. 已完成工作

### 4.1 第一阶段：数据审计与工程初始化

已完成：

- 递归扫描全部原始目录；
- 统计所有 PDF、缺失文件和 supporting 类型；
- 检查文件命名、目录层级和重复文件；
- 初始化 Python 3.11 + uv 工程；
- 建立 `src/`、`configs/`、`data/processed/`、`reports/`、`tests/`；
- 添加 `ruff` 和 `pytest`；
- 未发现 PDF 内容哈希重复。

主要报告：

```text
reports/initial_data_audit.md
```

### 4.2 第二阶段：PDF 文本提取与监管问题抽取

已完成：

- 41 / 41 个 PDF 成功打开并解析；
- 按页提取 4610 条文本记录；
- 生成原始文件清单；
- 从 inquiry、reply、supporting 中抽取监管问询；
- 区分 parent 一级问题和 subquestion 子问题；
- 标记 OCR 需求页面；
- 未执行 OCR。

主要输出：

```text
data/processed/manifest.csv
data/processed/document_pages.jsonl
data/processed/regulatory_questions.jsonl
reports/extraction_summary.md
reports/questions_for_review.csv
reports/parse_failures.csv
```

初始抽取结果：

- 监管问题总记录：231
- 一级问题：72
- 子问题：159
- 来源分布：
  - inquiry：12
  - reply：195
  - supporting：24
- needs_ocr：15 个文件、251 页
- 硬解析失败：0

### 4.3 第二阶段修复：监管问题清洗

初始问题抽取存在以下问题：

- 南京新百漏掉原始问询函第 1 题；
- 南京新百最后一题混入公告尾部；
- 部分子问题混入“请年审会计师发表意见”等专业机构要求；
- 利通电子、五新隧装 supporting 问题不完整；
- parent 和 subquestion 不能同时作为训练目标。

现已修复并生成：

```text
data/processed/regulatory_questions_raw.jsonl
data/processed/regulatory_questions_clean.jsonl
data/processed/auxiliary_questions.jsonl
reports/question_quality_summary.md
reports/question_extraction_validation.csv
```

正式 clean 数据统计：

- 一级问题：64 条
  - 只作为背景上下文
  - 不作为训练目标
- 子问题：157 条
  - 均为正式训练目标
- 正式数据来源：
  - inquiry：19
  - reply：202
  - supporting：0
- 不完整正式记录：0
- `report_year` 已统一为整数

辅助数据：

- `auxiliary_questions.jsonl`：32 条
- 其中：
  - parent：9 条
  - subquestion：23 条
  - 3 条含“……”或明确缺项
- 这些数据当前不进入正式训练集

关键字段包括：

```text
raw_question_text
cleaned_question_text
professional_opinion_request
is_complete
exclude_from_training
is_training_target
```

### 4.4 第三阶段：监管问题与年报证据检索对齐

已完成：

- 将 13 份年报切分为 3732 个 chunk；
- 每个 chunk 约 500–800 中文字符；
- chunk 间有一定重叠；
- 保留公司、股票代码、页码范围和原文；
- 使用 BM25 在同公司、同年度年报中检索；
- 为 157 条正式子问题各生成 Top-10 候选证据；
- 无有效候选问题：0。

主要输出：

```text
data/processed/annual_report_chunks.jsonl
data/processed/evidence_candidates.jsonl
reports/evidence_review.xlsx
reports/retrieval_summary.md
reports/ocr_priority_pages.csv
```

当前检索验证结果：

```text
Recall@1  = 1.0000
Recall@3  = 1.0000
Recall@5  = 1.0000
Recall@10 = 1.0000
```

注意：该结果只说明 BM25 能覆盖已有 14 条 parent 级人工页码金标，不能解释为每个 Top-1 候选都完全正确。当前项目定位为课程大作业/约稿，因此决定不继续做大规模教师模型筛选，直接进入训练主线。

建议 OCR 页面共 21 页，但目前不影响主线：

- 正式训练中暂不执行 OCR；
- 五新隧装不属于正式训练数据，其大量 OCR 页面可暂时忽略。

---

## 5. 当前可用于训练的数据

正式训练目标共 157 条。

每条训练样本计划构造为：

```json
{
  "instruction": "根据年度报告证据生成监管机构可能提出的问询问题。",
  "input": {
    "company": "公司名称",
    "report_year": 2024,
    "topic_title": "关于存货",
    "evidence": [
      {
        "page_start": 10,
        "page_end": 10,
        "text": "年度报告证据文本"
      }
    ]
  },
  "output": "清洗后的监管子问题"
}
```

证据输入策略：

- 每条子问题取 BM25 Top-3 chunk；
- 只使用同公司、同年度的 annual_report；
- 不使用 inquiry、reply、supporting 作为输入；
- parent 只作为背景信息，不作为独立输出目标；
- subquestion 作为真正的 SFT 输出。

---

## 6. 下一步工作

### 6.1 导出 SFT 数据

需要从以下文件构造训练样本：

```text
data/processed/regulatory_questions_clean.jsonl
data/processed/evidence_candidates.jsonl
```

筛选条件：

```text
is_training_target = true
exclude_from_training = false
```

每条样本使用 BM25 Top-3 证据。

输出建议：

```text
data/training/train.jsonl
data/training/validation.jsonl
data/training/test.jsonl
```

必须按公司划分，禁止同一公司跨 train / validation / test。

### 6.2 检查服务器环境

训练前检查：

```bash
nvidia-smi
df -h .
```

需要确认：

- GPU 型号
- GPU 数量
- 单卡显存
- 空闲显存
- CUDA 版本
- 当前目录剩余磁盘空间
- PyTorch 是否能识别 CUDA
- 服务器是否可访问 Hugging Face 或 ModelScope

### 6.3 下载 Qwen3-8B

模型不要下载到服务器全局缓存，统一放在当前项目目录：

```text
models/Qwen3-8B/
```

建议命令：

```bash
mkdir -p models/Qwen3-8B

uv run hf download Qwen/Qwen3-8B \
  --local-dir models/Qwen3-8B
```

同时：

- 将 `models/` 加入 `.gitignore`；
- 训练配置中的 `model_name_or_path` 设置为：

```yaml
model_name_or_path: models/Qwen3-8B
```

下载完成后检查：

```text
config.json
tokenizer 配置
*.safetensors
```

### 6.4 QLoRA dry-run

先用 10 条训练样本做 dry-run，确认：

- 模型可以加载；
- 4-bit 量化正常；
- forward / backward 正常；
- loss 能计算；
- adapter 能保存；
- 推理脚本能重新加载 adapter；
- 输出 JSONL 正常。

初始建议参数：

```yaml
model:
  model_name_or_path: models/Qwen3-8B
  load_in_4bit: true

training:
  seed: 42
  epochs: 3
  learning_rate: 2.0e-4
  batch_size: 1
  gradient_accumulation_steps: 16
  max_length: 2048

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
```

具体参数应根据 GPU 显存调整。

### 6.5 正式训练

dry-run 成功后再启动完整训练。

正式训练至少保存：

```text
outputs/
├── checkpoints/
├── adapter/
├── logs/
├── predictions/
└── configs/
```

需要保存：

- LoRA adapter；
- tokenizer；
- 完整训练配置；
- 随机种子；
- 环境版本；
- loss 日志；
- 测试集生成结果；
- 每条输出对应的 sample_id 和公司信息。

---

## 7. 当前技术路线

```text
原始 PDF
→ PDF 按页文本提取
→ 监管问题抽取与清洗
→ 年报文本切块
→ BM25 检索相关证据
→ 构造 157 条 SFT 样本
→ 按公司划分数据
→ Qwen3-8B 4-bit QLoRA
→ 测试集生成结果
```

---

## 8. 当前边界与注意事项

1. 当前数据量较小，项目定位应为低资源微调实验，不应夸大泛化能力。
2. 正式训练数据只有 11 家公司的 157 条子问题。
3. 利通电子、五新隧装当前仅作为 auxiliary，不进入正式训练与测试。
4. 不允许把公司回复正文作为模型输入，否则会产生标签泄漏。
5. 不允许随机按单条问题切分数据，必须按公司划分。
6. 不建议 parent 与 subquestion 同时作为独立训练输出。
7. 当前 BM25 Top-3 直接作为输入即可，不再继续拖延做复杂证据筛选。
8. OCR、GPT 教师筛选和数据扩增都可以等第一版训练结果出来后再决定。
9. 如果后续需要数据扩增，必须先划分 train / validation / test，再只对训练集扩增。
10. 原始 PDF 和 `regulatory_questions_raw.jsonl` 必须永久保留，任何清洗都生成新文件。

---

## 9. 当前工程验证状态

已完成验证：

```text
uv run ruff check .
uv run pytest
```

最近一次结果：

```text
ruff 通过
pytest 17 passed
```

前序问题清洗阶段：

```text
pytest 12 passed
```

后续新增训练代码后，需要继续保持：

```bash
uv run ruff check .
uv run pytest
```

全部通过后才能启动正式训练。

---

## 10. 交接后第一条建议指令

可以直接给 Codex：

```text
进入模型训练阶段，不再继续扩展数据清洗流程。

1. 使用 data/processed/regulatory_questions_clean.jsonl 中：
   - is_training_target=true
   - exclude_from_training=false
   的157条正式子问题。

2. 每条问题从 data/processed/evidence_candidates.jsonl 中取 BM25 Top-3 chunk。

3. 构造 SFT 数据：
   - instruction：根据年度报告证据生成监管问询问题
   - input：公司、年度、topic_title、Top-3年报证据及页码
   - output：cleaned_question_text

4. 按公司划分 train / validation / test，禁止同公司跨集合。

5. 输出：
   - data/training/train.jsonl
   - data/training/validation.jsonl
   - data/training/test.jsonl

6. 检查 GPU、CUDA、磁盘和 PyTorch 环境。

7. 不使用服务器全局 Hugging Face 缓存。
   将 Qwen/Qwen3-8B 下载到：
   models/Qwen3-8B

8. 训练配置中的 model_name_or_path 设为：
   models/Qwen3-8B

9. 使用 4-bit QLoRA：
   - rank=16
   - alpha=32
   - dropout=0.05
   - learning_rate=2e-4
   - epochs=3
   - batch_size=1
   - gradient_accumulation_steps=16
   - max_length=2048

10. 先只用10条样本做 dry-run。
    确认模型加载、前向、反向、loss、adapter保存和推理均正常后，
    停止并汇报，不要直接启动完整训练。
```

---

## 11. 一句话项目状态

> 数据审计、PDF 解析、监管问题抽取清洗和年报证据检索已经完成；当前已有 157 条正式 SFT 目标和对应 BM25 Top-3 年报证据，下一步直接导出训练集、下载本地 Qwen3-8B，并完成 QLoRA dry-run。
