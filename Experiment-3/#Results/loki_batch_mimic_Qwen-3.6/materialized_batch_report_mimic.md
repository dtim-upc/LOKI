# Batch Materialization Summary: MIMIC-IV (LOKI + Qwen-3.6)

- **Evaluation Profile:** default
- **Admissions Evaluated:** 382
- **Total Predicted Pairs:** 36,840
- **Total Ground Truth Pairs:** 5,456
- **Mean Cluster-Level Macro P / R / F1:** 0.7459 / 0.7285 / 0.7221
- **Mean Oracle Macro P / R / F1:** 0.8341 / 0.5115 / 0.6005
- **Mean Raw Pair-Cluster Purity:** 0.9949
- **Mean Cluster ARI:** 0.8577
- **Mean Cluster Silhouette:** 0.5336

---

## Mean Stage Execution Timers

| Pipeline Phase | Stage Component | Execution Time (s) |
|---|---|---:|
| **Encoding & Representation** | Phase C: Joint Contextual Encoding | 6.5338 |
| | Phase D5: Cross-Encoder Reranking | 2.6154 |
| | Phase D6: Cross-Encoder Pair Filtering | 0.0006 |
| **Path Extraction & Filtering** | Phase D: Join Path Extraction | 0.0249 |
| | Phase D: Heuristic Pair Filtering | 0.0116 |
| **Clustering** | Phase E: HDBSCAN Clustering | 0.0990 |
| | Phase E: Cluster Silhouette Computation | 0.0257 |
| | Phase E: Cluster Tail Filtering | 0.0025 |
| | Phase E: Negative Cluster Suppression | 0.0003 |
| **Semantic Materialization** | Phase E: Cluster Labeling (LLM) | 382.0610 |
| | Phase E: Pair Label Refinement | 1,450.9627 |
| | Phase E: Total Semantic Materialization | 1,833.1540 |
| **Evaluation** | Phase F: Evaluation & Scoring | 0.0039 |
| **Total Pipeline** | **End-to-End Execution Time** | **1,842.3442** |

---

## High-Performing Admissions

- **Best Relaxed Pair F1:** Admission 25471024 (0.75)
- **Best Pair Average Precision:** Admission 27676611 (1.00)

---

## Artifact References

- **Per-Admission Detail:** `materialized_batch_results_mimic.csv`
- **Aggregate Metrics:** `materialized_batch_summary_mimic.csv`
