# Batch Materialization Summary: MIMIC-IV (LOKI + GPT-OSS 20B)

- **Evaluation Profile:** default
- **Admissions Evaluated:** 382
- **Total Predicted Pairs:** 23,444
- **Total Ground Truth Pairs:** 5,456
- **Mean Cluster-Level Macro P / R / F1:** 0.7469 / 0.7470 / 0.7344
- **Mean Oracle Macro P / R / F1:** 0.8249 / 0.4850 / 0.5769
- **Mean Raw Pair-Cluster Purity:** 0.9956
- **Mean Cluster ARI:** 0.8062
- **Mean Cluster Silhouette:** 0.5012

---

## Runtime Breakdown

- **Mean End-to-End Runtime / Admission:** 179.83 s
- **Mean Join-Path Representation & Clustering:** 9.28 s (9.18 s representation + 0.10 s HDBSCAN)
- **First-Pass Labeling Time:** 35.55 s
- **Total First-Pass Pipeline Runtime:** 44.83 s

---

## High-Performing Admissions

- **Best Relaxed Pair F1:** Admission 25471024 (0.80)
- **Best Pair Average Precision:** Admission 27676611 (1.00)

---

## Artifact References

- **Per-Admission Detail:** `materialized_batch_results_mimic.csv`
- **Aggregate Metrics:** `materialized_batch_summary_mimic.csv`
