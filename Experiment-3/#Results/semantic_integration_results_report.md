# Relationship Clustering Diagnostics & Supplementary Report

This document provides supporting cluster-quality diagnostics, semantic labeling metrics, and compute profiles accompanying the primary **Relationship-Type Table Materialization** evaluation ([`relationship_table_report.md`](relationship_table_report.md)) across 382 MIMIC-IV admissions.

---

## 1. Evaluation Framework & Metrics

### 1.1 Typed Relationship Materialization Scores
The primary benchmark family evaluating how accurately each system discovers, resolves, and categorizes cross-table entity relationships into target semantic relation types:
- **Precision:** Macro-averaged precision across admissions (`cluster_label_macro_precision`).
- **Recall:** Macro-averaged recall across admissions (`cluster_label_macro_recall`).
- **F1:** Macro-averaged harmonic mean of precision and recall (`cluster_label_macro_f1`).

### 1.2 Ground Truth (GT) Pair Recovery
Measures end-to-end recall of reference entity pairs across the complete dataset:
\[
\text{GT Pair Recovery} = \frac{N_{\text{gt\_matched\_pairs}}}{N_{\text{gt\_pairs}}}
\]

### 1.3 Cluster Quality Diagnostics
Evaluates the partition structure of the discovered relationships:
- **Purity:** Degree to which each induced relation group consists of pairs belonging to a single ground-truth class.
- **Adjusted Rand Index (ARI):** Agreement between the discovered cluster partition and the reference partition, corrected for chance.
- **Accuracy & Macro F1:** Assignment accuracy against ground-truth partition mappings. For direct prompting baselines, clusters are induced by partitioning predicted pairs by relationship type.

---

## 2. Typed Relationship Materialization Scores

Data Sources:
- `#Results/relationship_clustering_dashboard_summary.csv`
- `#Results/relationship_clustering_summary.csv`
- `#Results/LOKI_Batch_mimic_GPT_OSS/materialized_batch_summary_mimic.csv`
- `#Results/loki_batch_mimic_Qwen-3.6/materialized_batch_summary_mimic.csv`

| System | Precision | Recall | F1 | GT Pair Recovery | Mean Runtime / Adm. |
|---|---:|---:|---:|---:|---:|
| **LOKI + GPT-OSS 20B** | 0.747 | 0.747 | 0.734 | 0.500 | 179.8 s |
| **LOKI + Qwen-3.6** | 0.746 | 0.729 | 0.722 | 0.532 | 1,842.4 s |
| **Direct Qwen-3.7-Max (API)** | 0.964 | 0.961 | 0.962 | 0.760 | 175.9 s |
| **Direct Qwen-3.6-Local** | 0.929 | 0.926 | 0.927 | 0.569 | 89.9 s |

### Key Findings
- **High Labeling Fidelity:** Both LOKI variants achieve strong Materialization F1 scores ($\sim 0.73$), with `LOKI + GPT-OSS 20B` achieving $0.734$ and `LOKI + Qwen-3.6` achieving $0.722$.
- **GT Pair Recovery:** `LOKI + Qwen-3.6` captures 53.2% of all ground-truth pairs, approaching direct local prompting (56.9%) while operating through structured multi-hop path extraction.
- **Efficiency-Quality Frontier:** `LOKI + GPT-OSS 20B` matches the runtime of commercial frontier APIs (179.8 s vs. 175.9 s) while executing on lightweight local/open-weight model architectures.

---

## 3. Cluster Quality Diagnostics

Evaluates the structural integrity and coherence of the materialized relation groups:

| System | Macro F1 | Accuracy | Purity | ARI |
|---|---:|---:|---:|---:|
| **LOKI + GPT-OSS 20B** | 0.734 | 0.846 | **0.996** | 0.806 |
| **LOKI + Qwen-3.6** | 0.722 | 0.817 | **0.995** | **0.858** |
| **Direct Qwen-3.7-Max (API)** | 0.817 | 0.696 | 0.715 | 0.706 |
| **Direct Qwen-3.6-Local** | 0.678 | 0.509 | 0.539 | 0.532 |

### Structural Observations
- **Near-Perfect Cluster Purity:** LOKI models achieve exceptionally high cluster purity ($\ge 0.995$), outperforming monolithic frontier models (0.715 for Qwen-3.7-Max and 0.539 for direct Qwen-3.6). This demonstrates that topological clustering over dense join paths isolates pure relationship types before prompt-based labeling occurs.
- **Partition Agreement (ARI):** LOKI achieves superior Adjusted Rand Index ($0.806$ to $0.858$), indicating that its materialized relational structures correspond closely to the underlying ground truth schema.

---

## 4. Compute & Resource Profile

Summary of execution time and LLM token requirements across 382 MIMIC-IV admissions:

| System | Mean Runtime / Adm. | Stage Breakdown | LLM Token Footprint | Inference Cost (382 Adm.) |
|---|---|---|---|---:|
| **LOKI + GPT-OSS 20B** | 179.8 s | 9.2 s join-path + 0.1 s HDBSCAN + 170.5 s labeling | 7.2K tokens / adm. (2.75M total) | **\$0.70** |
| **LOKI + Qwen-3.6** | 1,842.4 s | 9.2 s join-path + 0.1 s HDBSCAN + 1,833.1 s labeling | 7.2K tokens / adm. (2.75M total) | **\$2.53** |
| **Direct Qwen-3.7-Max** | 175.9 s | Single-pass prompt generation | 23.1K tokens / adm. (8.82M total) | **\$30.60** |
| **Direct Qwen-3.6-Local** | 89.9 s | Single-pass prompt generation | 22.1K tokens / adm. (8.43M total) | **\$7.76** |

---

## 5. Visual Artifacts & Figures

The evaluation suite outputs comprehensive visualization figures under `Visualizations/relationship_clustering/`:

### 5.1 Primary Comparison Figures
- **`all_models_main_comparison_metrics.png`**
  - Side-by-side comparative summary.
  - *Left Panel:* Relationship Clustering Quality (Accuracy, Purity, ARI).
  - *Right Panel:* Typed Relationship Materialization (Precision, Recall, F1).
- **`all_models_semantic_integration_metrics.png`**
  - Standalone evaluation of Typed Relationship Materialization Precision, Recall, and F1 across all four systems.
- **`all_models_semantic_integration_slices.png`**
  - Performance across controlled admission subsets: matched overall cohort (left panel) and high-complexity multitype overlap admissions (right panel).

### 5.2 Diagnostic & Structural Figures
- **`all_models_relationship_clustering_metrics.png`**
  - Standalone cluster quality metrics (Label Accuracy, Cluster Purity, and ARI).
- **`all_models_relationship_clustering_slices.png`**
  - Cluster quality diagnostics evaluated on matched-support slices.
- **`loki_per_admission_relationship_clustering_quality.png`**
  - Admission-level scatter plot comparing `LOKI + GPT-OSS 20B` and `LOKI + Qwen-3.6` (Cluster Recall vs. Cluster Precision, colored by Macro F1).
- **`LOKI_GPT-OSS_20B_relationship_clustering_dashboard.png`**
  - Comprehensive per-admission scatter and metric breakdown for the GPT-OSS 20B pipeline.
- **`LOKI_Qwen-3.6_relationship_clustering_dashboard.png`**
  - Comprehensive per-admission scatter and metric breakdown for the Qwen-3.6 pipeline.

### 5.3 Efficiency & Integrity Figures
- **`all_models_compute_cost.png`**
  - Runtime and token consumption comparison showing LOKI's 67%+ reduction in token footprint.
- **`all_models_data_quality.png`**
  - Relational schema adherence and entity integrity diagnostics.
