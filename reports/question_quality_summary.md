# Question Quality Summary

## Outputs

- Preserved raw extraction: `data/processed/regulatory_questions_raw.jsonl`
- Formal cleaned questions: `data/processed/regulatory_questions_clean.jsonl`
- Auxiliary supporting-source questions: `data/processed/auxiliary_questions.jsonl`
- Validation table: `reports/question_extraction_validation.csv`

## Overall Counts

- Raw records: 231
- Formal cleaned records: 221
- Formal parent questions retained as context: 64
- Formal subquestions: 157
- Formal trainable subquestions: 157
- Formal excluded records: 64
- Formal incomplete records: 0
- Formal records with separated professional/source-note requests: 88
- Auxiliary records moved out of gold data: 32
- Auxiliary parent questions: 9
- Auxiliary subquestions: 23

## Formal Source Distribution

- inquiry: 19
- reply: 202
- supporting: 0

## Company Counts

| Stock code | Company | Parent | Subquestion | Trainable subquestion |
| --- | --- | ---: | ---: | ---: |
| 600373 | 中文传媒 | 8 | 27 | 27 |
| 600678 | 四川金顶 | 4 | 14 | 14 |
| 600682 | 南京新百 | 7 | 12 | 12 |
| 600822 | 上海物贸 | 5 | 12 | 12 |
| 603466 | 风语筑 | 5 | 11 | 11 |
| 603922 | 金鸿顺 | 2 | 5 | 5 |
| 688363 | 华熙生物 | 4 | 9 | 9 |
| 688611 | 杭州柯林 | 4 | 11 | 11 |
| 688685 | 迈信林 | 8 | 14 | 14 |
| 688793 | 倍轻松 | 10 | 27 | 27 |
| 834033 | 康普化学 | 7 | 15 | 15 |

## Repair Notes

- Reparsed 南京新百 inquiry from `document_pages.jsonl`; restored original questions 1-7.
- Restored 南京新百 original question 1 and its 3 numbered subquestions.
- Trimmed 南京新百 question 7 before reply deadline, disclosure-media, risk-warning, announcement and board-date footer text.
- Moved 利通电子 and 五新隧装 supporting-source records to auxiliary output and excluded them from formal gold data.
- Added fallback subquestions for parent questions that had no numbered children but contained clear regulatory requests.
- Separated auditor, independent-director, sponsor/broker opinion requests and inquiry-source notes into `professional_opinion_request`.

## Validation Issue Counts

| Validation type | Count |
| --- | ---: |
| auxiliary_incomplete_question | 3 |
| auxiliary_question_too_short | 2 |
| generated_missing_subquestions | 17 |
| subquestion_count_alignment | 73 |
| supporting_moved_to_auxiliary | 9 |
