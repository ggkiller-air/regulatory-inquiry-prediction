# QLoRA Dry-Run Summary (2026-07-16)

## 环境

- GPU：4 × NVIDIA A800-SXM4-80GB（空闲），驱动 575.57.08 / CUDA 12.9
- Python 3.11.15（uv 管理），torch 2.8.0+cu128，`torch.cuda.is_available()=True`
- transformers / peft / bitsandbytes / accelerate 已通过 `uv sync` 安装
- 磁盘剩余约 700G

## 模型

- Qwen/Qwen3-8B 已下载到 `models/Qwen3-8B/`（16.4GB，5 个 safetensors 分片齐全，config/tokenizer 完整）
- 经 hf-mirror 下载，需设置 `HF_HUB_DISABLE_XET=1`（镜像不支持 Xet CAS）
- `models/` 已在 `.gitignore` 中

## SFT 数据

- 正式训练目标 157 条（is_training_target=true 且 exclude_from_training=false）
- 每条使用 BM25 Top-3 年报证据（交接文档要求 Top-3；旧 `sft_train.jsonl` 为 Top-5，仅保留作参考）
- 公司级划分（无公司跨集合，见 `reports/sft_split_summary.md`）：
  - train：118 条 / 7 家公司
  - validation：16 条 / 2 家（风语筑、金鸿顺）
  - test：23 条 / 2 家（南京新百、杭州柯林——南京新百含唯一原始问询函金标，保留作最高质量评测）
- 实测 token 长度 max=2242，`max_length` 由 2048 放宽至 2560，避免截断 42 条样本

## Dry-run 结果（10 条样本，1 epoch）

命令：

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python -m src.training.train_qlora --max-samples 10 --epochs 1 --output-dir outputs/dry_run
```

验证项全部通过：

- [x] 4-bit NF4 量化加载正常（trainable 43.6M / 8.23B ≈ 0.53%）
- [x] forward / backward 正常，loss=5.916，grad_norm=11.54
- [x] eval loss 可计算（6.522）
- [x] adapter 保存至 `outputs/dry_run/adapter/`（含 tokenizer、训练配置、环境快照、loss 日志）
- [x] 推理脚本重新加载 adapter 成功，输出 JSONL 正常（`outputs/dry_run/predictions/validation_dry_run.jsonl`）
- [x] `uv run ruff check .` 通过，`uv run pytest` 22 passed

Dry-run 生成内容仍为通用监管问询模板（未学到样本分布），属预期——仅 1 个 optimizer step。

## 正式训练入口（尚未执行）

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python -m src.training.train_qlora --output-dir outputs/full_run
```

配置见 `configs/qlora_qwen3_8b.yaml`（seed=42，epochs=3，lr=2e-4，rank=16，alpha=32，dropout=0.05，
batch_size=1，grad_accum=16，max_length=2560）。
118 条 × 3 epochs ≈ 22 个 optimizer step，单卡预计几分钟级别。

测试集生成：

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python -m src.training.generate \
  --adapter outputs/full_run/adapter \
  --input data/training/test.jsonl \
  --output outputs/full_run/predictions/test_predictions.jsonl
```
