# Qwen3-8B QLoRA 正式训练总结 (2026-07-16)

## 训练设置

- 4 × A800-80GB DDP(`torchrun --nproc_per_node=4`),4-bit NF4 QLoRA
- 总有效 batch 保持 16(每卡 batch_size=1 × grad_accum=4 × 4 卡)
- 超参:seed=42, epochs=3, lr=2e-4 cosine, rank=16, alpha=32, dropout=0.05, max_length=2560
- 数据:train 118 / validation 16 / test 23(公司级划分,无泄漏)
- 训练耗时:约 100 秒(24 个 optimizer step)

## 结果

- train loss:5.8 → 1.85
- eval loss:2.21(epoch1)→ 2.09(epoch2)→ 2.07(epoch3),未见过拟合
- test 集 23 条全部生成成功,输出为监管问询风格、贴合证据主题

## 产物位置

```text
outputs/full_run/
├── adapter/                          # LoRA adapter + tokenizer
├── checkpoints/                      # 按 epoch 保存(最近 2 个)
├── logs/loss_history.jsonl           # 完整 loss 日志
├── logs/loss_curve.png               # loss 曲线图
├── predictions/test_predictions.jsonl  # 测试集 23 条生成结果(含 sample_id/公司/参考答案)
├── environment.json                  # 环境版本快照
└── training_config.yaml              # 完整训练配置
```

## 观察与下一步候选

- 模型已学会监管问询文体和"结合……说明……是否存在……"句式;
  但具体问点与金标存在偏差(如金标问期间费用构成,模型问投资理财),
  受限于 118 条训练样本,属低资源微调的预期表现。
- 可选后续:BLEU/ROUGE/人工评分量化评测、对比未微调基线、
  训练集数据扩增(仅对 train)、OCR 补充证据。
