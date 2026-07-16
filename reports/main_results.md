| 方法 | ROUGE-L | BERTScore-F1 | 关键问点F1 | 证据支持率 | 幻觉率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B Zero-shot | 0.2118 | 0.7271 | 待人工复核 | 待人工复核 | 待人工复核 |
| Qwen3-8B + Evidence | 0.2030 | 0.7208 | 待人工复核 | 待人工复核 | 待人工复核 |
| Qwen3-8B QLoRA | 0.2587 | 0.7520 | 待人工复核 | 待人工复核 | 待人工复核 |
| Full Model（本文） | 0.2592 | 0.7473 | 待人工复核 | 待人工复核 | 待人工复核 |

评测协议：按相同模型输入将 23 条测试子问题合并为 7 个输入组（multi-reference），每组取预测与组内各参考问询的最高 ROUGE-L / BERTScore-F1，再对 7 组宏平均。
关键问点F1、证据支持率、幻觉率待人工复核完成后填写（复核表：reports/key_point_annotation.xlsx、reports/factuality_annotation.xlsx）。
