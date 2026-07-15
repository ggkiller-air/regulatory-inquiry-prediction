# Retrieval Summary

## Outputs

- `data/processed/annual_report_chunks.jsonl`
- `data/processed/evidence_candidates.jsonl`
- `reports/evidence_review.xlsx`
- `reports/ocr_priority_pages.csv`

## Scope

- Queries use only formal clean subquestions with `is_training_target=true` and `exclude_from_training=false`.
- Evidence corpus uses only same-company, same-year `annual_report` pages.
- Inquiry, reply and supporting documents are not indexed as evidence.

## Counts

- Processed training subquestions: 157
- Annual report chunks: 3732
- Evidence candidate rows: 157
- Questions with no effective candidate score: 0
- OCR priority annual-report pages: 21

## Retrieval Validation

- Human-aligned parent records: 14
- Mapped validation records: 14
- Recall@1: 1.0000
- Recall@3: 1.0000
- Recall@5: 1.0000
- Recall@10: 1.0000

Validation note: the Excel workbook contains 14 parent-level annual-report alignments. Each row is mapped by company and original question number to the corresponding formal subquestions; a hit is counted when any mapped subquestion candidate page range intersects the gold page range.

## Chunk Counts By Company

| Stock code | Company | Chunks |
| --- | --- | ---: |
| 600373 | 中文传媒 | 375 |
| 600678 | 四川金顶 | 249 |
| 600682 | 南京新百 | 399 |
| 600822 | 上海物贸 | 217 |
| 603466 | 风语筑 | 230 |
| 603629 | 利通电子 | 284 |
| 603922 | 金鸿顺 | 237 |
| 688363 | 华熙生物 | 450 |
| 688611 | 杭州柯林 | 274 |
| 688685 | 迈信林 | 306 |
| 688793 | 倍轻松 | 289 |
| 834033 | 康普化学 | 202 |
| 835174 | 五新隧装 | 220 |

## No Effective Candidates

- None.

## OCR Priority Pages

| Stock code | Company | Pages |
| --- | --- | ---: |
| 600373 | 中文传媒 | 2 |
| 600678 | 四川金顶 | 1 |
| 600682 | 南京新百 | 2 |
| 688363 | 华熙生物 | 1 |
| 834033 | 康普化学 | 1 |
| 835174 | 五新隧装 | 14 |
