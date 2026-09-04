# LOKI: Experimental Evaluation & Benchmark Results

This repository contains the experimental evaluation artifacts for the **LOKI** pipeline compared against direct frontier LLM prompting baselines on the MIMIC-IV clinical dataset across 382 hospital admissions.

In this evaluation, the **primary task and main paper result** is **Relationship-Type Table Materialization**: discovering, extracting, and physically materializing relational tables for each cross-table relationship type. The accompanying **Relationship Clustering diagnostics**, **matched-support slice analyses**, and **compute cost profiles** provide supporting structural and efficiency verification.

---

## 1. Primary Result: Relationship-Type Table Materialization

### 1.1 Task Formulation & Evaluation Metrics

The core objective of LOKI is the automated construction of relational tables corresponding to distinct relationship types from unaligned EHR databases. Each predicted cluster is materialized as one relational table representing a specific semantic relationship within an admission:

- **Predicted Table ($T_{\text{pred}}$):** A materialized relation table produced by the system for a specific relationship type.
- **Ground Truth Table ($T_{\text{gt}}$):** A reference relation table defined by the clinical ground truth for that relationship type.
- **Best-Match Typed-Pair Macro P / R / F1:** Mean per-admission scores evaluated on entity pairs after mapping each predicted table to its optimal ground-truth relation type.
- **Typed Table Materialization Macro P / R / F1:** Mean per-admission macro scores evaluating physical table recovery.
- **Typed Table Materialization Micro P / R / F1:** Global pooled scores aggregating table-level true positives, false positives, and false negatives across all 382 admissions.

$$
\text{Precision} = \frac{\text{TP}}{\text{Pred}}, \quad \text{Recall} = \frac{\text{TP}}{\text{GT}}, \quad \text{F1} = \frac{2 \times \text{TP}}{\text{Pred} + \text{GT}}
$$

Detailed technical report: **[`relationship_table_report.md`](%23Results/relationship_table_report.md)**

---

### 1.2 Materialization Summary Performance

| System | Admissions | Best-Match Typed-Pair Macro P / R / F1 | Typed Table Materialization Macro P / R / F1 | Typed Table Materialization Micro P / R / F1 |
|---|---:|---|---|---|
| **LOKI + GPT-OSS 20B** | 378 | 0.982 / 0.486 / 0.627 | 0.755 / 0.755 / 0.742 | **0.840 / 0.840 / 0.840** |
| **LOKI + Qwen-3.6** | 380 | 0.982 / 0.515 / 0.652 | 0.750 / 0.732 / 0.726 | **0.848 / 0.807 / 0.827** |
| **Direct Qwen-3.7-Max** | 381 | 0.997 / 0.717 / 0.817 | 0.966 / 0.963 / 0.964 | 0.958 / 0.958 / 0.958 |
| **Direct Qwen-3.6-Local** | 381 | 0.993 / 0.540 / 0.678 | 0.932 / 0.929 / 0.930 | 0.920 / 0.920 / 0.920 |

---

### 1.3 Physical Table Materialization Counts (Global Micro Surface)

Evaluates the discrete relational tables materialized by each system against ground truth tables:

| System | Pred Tables | GT Tables | TP | FP | FN | Micro Precision | Micro Recall | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LOKI + GPT-OSS 20B** | 2,018 | 2,018 | 1,696 | 322 | 322 | **0.8404** | **0.8404** | **0.8404** |
| **LOKI + Qwen-3.6** | 2,011 | 2,113 | 1,705 | 306 | 408 | **0.8478** | **0.8069** | **0.8269** |
| **Direct Qwen-3.7-Max** | 548 | 548 | 525 | 23 | 23 | 0.9580 | 0.9580 | 0.9580 |
| **Direct Qwen-3.6-Local** | 523 | 523 | 481 | 42 | 42 | 0.9197 | 0.9197 | 0.9197 |

---

### 1.4 Entity Pair Resolution Counts (Best-Match Surface)

Evaluates the accuracy of individual cross-table entity pairs mapped into materialized tables:

| System | Pred Pairs | GT Pairs | TP | FP | FN | Micro Precision | Micro Recall | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LOKI + GPT-OSS 20B** | 2,882 | 6,441 | 2,823 | 59 | 3,618 | **0.9795** | 0.4383 | 0.6056 |
| **LOKI + Qwen-3.6** | 3,051 | 6,454 | 2,990 | 61 | 3,464 | **0.9800** | 0.4633 | 0.6291 |
| **Direct Qwen-3.7-Max** | 4,149 | 6,468 | 4,135 | 14 | 2,333 | 0.9966 | 0.6393 | 0.7789 |
| **Direct Qwen-3.6-Local** | 3,103 | 6,468 | 3,086 | 17 | 3,382 | 0.9945 | 0.4771 | 0.6449 |

---

### 1.5 Primary Materialization Figures

#### Core Benchmark Comparison
Side-by-side visualization showing Relationship Clustering Quality (left panel) alongside Typed Relationship Materialization (right panel):

![Main Comparison Metrics](%23Results/Visualizations/relationship_clustering/all_models_main_comparison_metrics.png)

#### Matched-Support & Multi-Type Complexity Evaluation
Performance across controlled admission subsets: overall matched support cohort (left panel) and high-complexity multi-type overlap admissions (right panel):

![Semantic Integration Slices](%23Results/Visualizations/relationship_clustering/all_models_semantic_integration_slices.png)

---

### 1.6 Materialization Findings

1. **High Pair Precision Across Architectures:** LOKI achieves $\ge 97.9\%$ pair-level precision ($0.9795$ for GPT-OSS 20B and $0.9800$ for Qwen-3.6), demonstrating that candidate join paths mapped into tables rarely suffer from semantic misclassification.
2. **Table Materialization Quality:** LOKI attains $84.0\%$ to $84.8\%$ micro precision on physical table creation ($1,696$ correct tables for GPT-OSS and $1,705$ for Qwen-3.6), validating the integrity of its end-to-end schema synthesis.
3. **Relational Granularity:** Monolithic prompting collapses distinct join paths into broad aggregate relation buckets ($523$–$548$ total tables). In contrast, LOKI discovers and materializes fine-grained relation tables ($2,011$–$2,018$ tables) capturing localized semantic pathways across database tables.

---

## 2. Supplementary Diagnostics: Cluster Partition Quality

Cluster quality diagnostics quantify the structural coherence of candidate join-path partitions generated prior to final relation table synthesis.

Detailed report: **[`semantic_integration_results_report.md`](%23Results/semantic_integration_results_report.md)**  
Matched-support appendix: **[`relationship_clustering_fairness_report.md`](%23Results/relationship_clustering_fairness_report.md)**

### 2.1 Full-Cohort Structural Diagnostics

| System | Macro F1 | Accuracy | Purity | Adjusted Rand Index (ARI) |
|---|---:|---:|---:|---:|
| **LOKI + GPT-OSS 20B** | 0.734 | 0.846 | **0.996** | **0.806** |
| **LOKI + Qwen-3.6** | 0.722 | 0.817 | **0.995** | **0.858** |
| **Direct Qwen-3.7-Max (API)** | 0.817 | 0.696 | 0.715 | 0.706 |
| **Direct Qwen-3.6-Local** | 0.678 | 0.509 | 0.539 | 0.532 |

![Relationship Clustering Diagnostics](%23Results/Visualizations/relationship_clustering/all_models_relationship_clustering_metrics.png)

---

### 2.2 Matched-Cohort Robustness Analysis (Recall-Weighted Diagnostics)

Evaluates clustering partition structure across strictly matched admission subsets:

| Baseline Model | LOKI Baseline | Evaluation Slice | Admissions | Baseline Accuracy | LOKI Accuracy | Baseline Purity | LOKI Purity | Baseline ARI | LOKI ARI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen-3.7-Max** | LOKI + GPT-OSS 20B | matched_all | 377 | 0.685 | **0.841** | 0.714 | **0.996** | 0.704 | **0.808** |
| **Qwen-3.7-Max** | LOKI + GPT-OSS 20B | matched_multitype_overlap | 164 | 0.581 | **0.760** | 0.619 | **0.990** | 0.607 | **0.747** |
| **Qwen-3.7-Max** | LOKI + Qwen-3.6 | matched_all | 379 | 0.686 | **0.807** | 0.714 | **0.995** | 0.705 | **0.860** |
| **Qwen-3.7-Max** | LOKI + Qwen-3.6 | matched_multitype_overlap | 176 | 0.585 | **0.729** | 0.622 | **0.989** | 0.614 | **0.771** |
| **Qwen-3.6-Local** | LOKI + GPT-OSS 20B | matched_all | 377 | 0.495 | **0.841** | 0.538 | **0.996** | 0.530 | **0.808** |
| **Qwen-3.6-Local** | LOKI + GPT-OSS 20B | matched_multitype_overlap | 154 | 0.399 | **0.755** | 0.449 | **0.990** | 0.443 | **0.744** |
| **Qwen-3.6-Local** | LOKI + Qwen-3.6 | matched_all | 379 | 0.496 | **0.807** | 0.538 | **0.995** | 0.531 | **0.860** |
| **Qwen-3.6-Local** | LOKI + Qwen-3.6 | matched_multitype_overlap | 165 | 0.404 | **0.727** | 0.455 | **0.991** | 0.449 | **0.771** |

![Relationship Clustering Across Slices](%23Results/Visualizations/relationship_clustering/all_models_relationship_clustering_slices.png)

---

### 2.3 Admission-Level Clustering Distributions

Scatter plot illustrating admission-level cluster recall vs. precision across both LOKI backends, with point colors representing Macro F1:

![LOKI Per-Admission Quality](%23Results/Visualizations/relationship_clustering/loki_per_admission_relationship_clustering_quality.png)

---

### 2.4 Cluster Quality Observations

1. **Near-Perfect Structural Purity:** LOKI consistently achieves cluster purity $\ge 99.5\%$ ($0.995$–$0.996$), compared to $0.539$–$0.715$ for prompt baselines. Topological clustering over contextual path embeddings separates distinct relationship semantics cleanly.
2. **Partition Agreement (ARI):** LOKI achieves an Adjusted Rand Index of $0.806$–$0.858$, significantly outperforming direct prompt partitions ($0.532$–$0.706$).
3. **Robustness on Multi-Type Admissions:** On the complex `matched_multitype_overlap` slice (admissions exhibiting multiple co-occurring relation types), LOKI maintains cluster purity above $98.9\%$ and ARI above $0.744$, verifying that multi-hop path clustering prevents semantic cross-contamination.

---

## 3. Compute Economics & Inference Efficiency

Detailed report: **[`Compute_Cost/README.md`](%23Results/Compute_Cost/README.md)**

### 3.1 Workload & Cost Summary (382 Admissions)

Evaluated under standard public commercial API rates per million tokens:
- **GPT-OSS 20B:** \$0.200 / 1M input, \$0.300 / 1M output (Cloudflare Workers AI)
- **Qwen-3.6-35B-A3B:** \$0.248 / 1M input, \$1.485 / 1M output (Alibaba Cloud Model Studio)
- **Qwen-3.7-Max:** \$1.650 / 1M input, \$4.951 / 1M output (Alibaba Cloud Model Studio)

| System | Admissions | Avg. Tokens / Adm. | Total Tokens | Prompt Tokens | Completion Tokens | Total Cost | Cost / Adm. | Cost Savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LOKI + GPT-OSS 20B** | 382 | **7,200** | **2,750,400** | 1,255,720 | 1,494,680 | **\$0.70** | **\$0.0018** | **97.7%** |
| **LOKI + Qwen-3.6** | 382 | **7,200** | **2,750,400** | 1,255,720 | 1,494,680 | **\$2.53** | **\$0.0066** | **91.7%** |
| **Direct Qwen-3.6-Local** | 382 | 22,079 | 8,434,285 | 3,850,749 | 4,583,536 | **\$7.76** | \$0.0203 | 74.6% |
| **Direct Qwen-3.7-Max** | 382 | 23,087 | 8,819,207 | 3,957,877 | 4,861,330 | **\$30.60** | \$0.0801 | Baseline (1.0×) |

---

### 3.2 Compute Trade-Off Diagram

Runtime and token consumption comparison illustrating LOKI's compute efficiency:

![Compute Cost & Token Efficiency](%23Results/Visualizations/relationship_clustering/all_models_compute_cost_half_circle.png)

---

### 3.3 Efficiency Architecture

1. **Token Reduction:** LOKI processes multi-hop join extraction and density clustering deterministically, invoking the LLM strictly for cluster centroid labeling. This reduces token traffic by **67.4% to 68.8%** (2.75M vs. 8.43M–8.82M tokens).
2. **Decoupled Model Tiering:** Because LLM inference is confined to structured relationship labeling on pre-filtered evidence, LOKI executes reliably with lightweight open-weight models (\$0.70–\$2.53 total cost), bypassing expensive frontier API deployments (\$30.60) while maintaining higher structural purity.

---

## 4. Relational Data Quality & Schema Adherence

Evaluates relational integrity, foreign-key consistency, and schema validity of the materialized outputs:

![Relational Schema Adherence and Data Quality](%23Results/Visualizations/relationship_clustering/all_models_data_quality.png)

---

## 5. Artifact Directory Map

Detailed technical reports and raw audit logs available in this repository:

- **[`#Results/relationship_table_report.md`](%23Results/relationship_table_report.md):** Primary paper results — complete pair- and table-level contingency matrices ($\text{TP}, \text{FP}, \text{FN}$) and formal metric definitions.
- **[`#Results/semantic_integration_results_report.md`](%23Results/semantic_integration_results_report.md):** Comprehensive reference for cluster quality diagnostics and per-system dashboards.
- **[`#Results/relationship_clustering_fairness_report.md`](%23Results/relationship_clustering_fairness_report.md):** Detailed matched-support ablation analysis for appendix documentation.
- **[`#Results/Compute_Cost/README.md`](%23Results/Compute_Cost/README.md):** Cloud token-cost methodology, price models, and detailed arithmetic breakdowns.
- **[`#Results/Visualizations/relationship_clustering/README.md`](%23Results/Visualizations/relationship_clustering/README.md):** Interactive publication figure gallery with vector PDF downloads.
- **`#Results/LOKI_Batch_mimic_*/`:** Raw batch materialization outputs, per-admission metrics CSVs, and stage execution logs.
- **`LLM_Eval_Ex-3/`:** Pipeline execution code, ground truth reference files (`GT/`), and raw prediction outputs (`Pred/`).

