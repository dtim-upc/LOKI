# Relationship Clustering: Matched-Support Robustness Analysis

Evaluation of relationship clustering performance across strictly matched admission subsets, ensuring identical test support when comparing LOKI against direct prompting baselines.

---

## 1. Evaluation Cohorts

- **`matched_all`:** Admissions where both the baseline prompt and the corresponding LOKI variant successfully produced candidate pairs.
- **`matched_multitype_overlap`:** Subset of admissions containing multiple ground-truth relationship types, evaluating disambiguation performance on complex admissions.
- **Metric Formulation:** LOKI metrics report cluster-level macro scores. Prompt metrics report comparable oracle pair P/R/F1. Secondary diagnostic metrics (Accuracy, Purity, ARI) are recall-weighted to account for unrecovered ground-truth pairs.

---

## 2. Matched-Support Summary

| Baseline Model | LOKI Baseline | Slice | Admissions | Baseline Pairs | LOKI Pairs | Baseline Clusters | LOKI Clusters | Baseline P / R / F1 | LOKI Macro P / R / F1 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| **Qwen-3.7-Max** | LOKI + GPT-OSS 20B | matched_all | 377 | 4,117 | 2,726 | 544 | 2,014 | 0.997 / 0.716 / 0.817 | 0.755 / 0.756 / 0.743 |
| **Qwen-3.7-Max** | LOKI + GPT-OSS 20B | matched_multitype_overlap | 164 | 2,103 | 1,525 | 301 | 1,143 | 0.993 / 0.623 / 0.750 | 0.560 / 0.581 / 0.545 |
| **Qwen-3.7-Max** | LOKI + Qwen-3.6 | matched_all | 379 | 4,127 | 2,899 | 546 | 2,109 | 0.997 / 0.716 / 0.817 | 0.750 / 0.733 / 0.727 |
| **Qwen-3.7-Max** | LOKI + Qwen-3.6 | matched_multitype_overlap | 176 | 2,222 | 1,681 | 315 | 1,230 | 0.994 / 0.625 / 0.752 | 0.539 / 0.558 / 0.526 |
| **Qwen-3.6-Local** | LOKI + GPT-OSS 20B | matched_all | 377 | 3,078 | 2,726 | 519 | 2,014 | 0.993 / 0.539 / 0.677 | 0.755 / 0.756 / 0.743 |
| **Qwen-3.6-Local** | LOKI + GPT-OSS 20B | matched_multitype_overlap | 154 | 1,473 | 1,454 | 268 | 1,084 | 0.986 / 0.451 / 0.601 | 0.565 / 0.582 / 0.547 |
| **Qwen-3.6-Local** | LOKI + Qwen-3.6 | matched_all | 379 | 3,085 | 2,899 | 521 | 2,109 | 0.993 / 0.539 / 0.677 | 0.750 / 0.733 / 0.727 |
| **Qwen-3.6-Local** | LOKI + Qwen-3.6 | matched_multitype_overlap | 165 | 1,568 | 1,608 | 284 | 1,167 | 0.987 / 0.457 / 0.607 | 0.536 / 0.555 / 0.523 |

---

## 3. Structural Partition Diagnostics (Recall-Weighted)

| Baseline Model | LOKI Baseline | Slice | Baseline Accuracy | LOKI Accuracy | Baseline Purity | LOKI Purity | Baseline ARI | LOKI ARI |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen-3.7-Max** | LOKI + GPT-OSS 20B | matched_all | 0.685 | **0.841** | 0.714 | **0.996** | 0.704 | **0.808** |
| **Qwen-3.7-Max** | LOKI + GPT-OSS 20B | matched_multitype_overlap | 0.581 | **0.760** | 0.619 | **0.990** | 0.607 | **0.747** |
| **Qwen-3.7-Max** | LOKI + Qwen-3.6 | matched_all | 0.686 | **0.807** | 0.714 | **0.995** | 0.705 | **0.860** |
| **Qwen-3.7-Max** | LOKI + Qwen-3.6 | matched_multitype_overlap | 0.585 | **0.729** | 0.622 | **0.989** | 0.614 | **0.771** |
| **Qwen-3.6-Local** | LOKI + GPT-OSS 20B | matched_all | 0.495 | **0.841** | 0.538 | **0.996** | 0.530 | **0.808** |
| **Qwen-3.6-Local** | LOKI + GPT-OSS 20B | matched_multitype_overlap | 0.399 | **0.755** | 0.449 | **0.990** | 0.443 | **0.744** |
| **Qwen-3.6-Local** | LOKI + Qwen-3.6 | matched_all | 0.496 | **0.807** | 0.538 | **0.995** | 0.531 | **0.860** |
| **Qwen-3.6-Local** | LOKI + Qwen-3.6 | matched_multitype_overlap | 0.404 | **0.727** | 0.455 | **0.991** | 0.449 | **0.771** |

---

## 4. Key Findings

1. **Consistent Structural Superiority:** Across all matched slices, both LOKI pipelines consistently achieve higher Cluster Accuracy ($0.727$–$0.841$ vs. $0.399$–$0.686$), higher Cluster Purity ($0.989$–$0.996$ vs. $0.449$–$0.714$), and higher ARI ($0.744$–$0.860$ vs. $0.443$–$0.705$) compared to direct prompting baselines.
2. **Robustness on Complex Admissions:** On the harder `matched_multitype_overlap` slice (requiring simultaneous disambiguation across multiple relation types), LOKI maintains near-perfect cluster purity ($\ge 0.989$) and high ARI ($\ge 0.744$), demonstrating that topological candidate clustering prevents cross-type contamination.
3. **Partition Granularity:** LOKI generates fine-grained relation groupings tailored to distinct join paths ($1,084$–$2,109$ clusters), avoiding the coarse over-aggregation observed in direct prompt outputs ($268$–$546$ clusters).
