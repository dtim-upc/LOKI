# Inference Cost & Token Efficiency: LOKI vs. Direct LLM Baselines

- **Evaluation Set:** 382 MIMIC-IV admissions  
- **Task:** End-to-end typed relationship discovery and materialization  
- **Metric:** Token consumption and normalized cloud inference cost  

---

## 1. Overview

Benchmark comparing the token consumption and cloud inference costs of LOKI against direct LLM prompting baselines across 382 MIMIC-IV admissions:

1. **LOKI + GPT-OSS 20B**
2. **LOKI + Qwen-3.6-35B-A3B**
3. **Direct Prompting (Qwen-3.6-35B-A3B)**
4. **Direct Prompting (Qwen-3.7-Max)**

LOKI isolates LLM inference strictly to the semantic cluster-labeling stage, requiring an average of **7.2K tokens per admission** (2.75M tokens across 382 admissions). In contrast, direct prompting requires the model to process raw candidate context and emit full relationship structures, consuming **22.1K tokens per admission** for Qwen-3.6 (8.43M total tokens) and **23.1K tokens per admission** for Qwen-3.7-Max (8.82M total tokens).

Cloud inference expenditures are evaluated using standard commercial API list pricing per million tokens, establishing a normalized cost-efficiency benchmark across architectures.

---

## 2. API Pricing Model

Pricing reflects standard public API rates per million tokens:

| Model | Input Price / 1M Tokens | Output Price / 1M Tokens | Endpoint / Provider |
|---|---:|---:|---|
| **GPT-OSS 20B** | \$0.200 | \$0.300 | Cloudflare Workers AI |
| **Qwen-3.6-35B-A3B** | \$0.248 | \$1.485 | Alibaba Cloud Model Studio |
| **Qwen-3.7-Max** | \$1.650 | \$4.951 | Alibaba Cloud Model Studio |

---

## 3. Measured Token Workloads

### 3.1 Direct Qwen-3.6-35B-A3B

Measured from full execution across all 382 admissions (`Qwen3.6-Local/inference_timing_summary.json`):

- **Prompt Tokens:** 3,850,749 (45.66%)
- **Completion Tokens:** 4,583,536 (54.34%)
- **Total Tokens:** 8,434,285
- **Average Tokens / Admission:** 22,079.28

### 3.2 Direct Qwen-3.7-Max

Measured from baseline benchmark executions (`Qwen-3.7/inference_timing_summary.json`):

- **Average Prompt Tokens / Admission:** 10,360.94
- **Average Completion Tokens / Admission:** 12,726.00
- **Average Total Tokens / Admission:** 23,086.93

Scaled across 382 admissions:

- **Prompt Tokens:** 3,957,877 (44.88%)
- **Completion Tokens:** 4,861,330 (55.12%)
- **Total Tokens:** 8,819,207

### 3.3 LOKI Relationship Labeling

LOKI delegates entity linking and candidate path generation to deterministic graph traversal and HDBSCAN clustering, invoking the LLM solely for relationship classification on condensed evidence:

- **Average Labeling Tokens / Admission:** 7,200 (7.2K)
- **Total Admissions:** 382
- **Total Tokens:** 2,750,400 (2.75M)

Apportioned using the empirical direct prompt/completion distribution (45.66% prompt / 54.34% completion):

- **Prompt Tokens:** 1,255,720
- **Completion Tokens:** 1,494,680

---

## 4. Test-Set Efficiency & Cost Summary

Total inference costs are computed as:

\[
\text{Cost} = \frac{N_{\text{prompt}}}{10^6} \times C_{\text{input}} + \frac{N_{\text{completion}}}{10^6} \times C_{\text{output}}
\]

### Full 382-Admission Benchmark

| System | Admissions | Avg. Tokens / Adm. | Total Tokens | Prompt Tokens | Completion Tokens | Total Cost | Cost / Adm. |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LOKI + GPT-OSS 20B** | 382 | 7,200 | 2,750,400 | 1,255,720 | 1,494,680 | **\$0.70** | \$0.0018 |
| **LOKI + Qwen-3.6-35B** | 382 | 7,200 | 2,750,400 | 1,255,720 | 1,494,680 | **\$2.53** | \$0.0066 |
| **Direct Qwen-3.6-35B** | 382 | 22,079 | 8,434,285 | 3,850,749 | 4,583,536 | **\$7.76** | \$0.0203 |
| **Direct Qwen-3.7-Max** | 382 | 23,087 | 8,819,207 | 3,957,877 | 4,861,330 | **\$30.60** | \$0.0801 |

---

## 5. Detailed Cost Breakdown

### LOKI + GPT-OSS 20B
\[
(1.255720 \times \$0.200) + (1.494680 \times \$0.300) = \$0.251 + \$0.448 = \mathbf{\$0.70}
\]

### LOKI + Qwen-3.6-35B
\[
(1.255720 \times \$0.248) + (1.494680 \times \$1.485) = \$0.311 + \$2.220 = \mathbf{\$2.53}
\]

### Direct Qwen-3.6-35B
\[
(3.850749 \times \$0.248) + (4.583536 \times \$1.485) = \$0.955 + \$6.807 = \mathbf{\$7.76}
\]

### Direct Qwen-3.7-Max
\[
(3.957877 \times \$1.650) + (4.861330 \times \$4.951) = \$6.530 + \$24.068 = \mathbf{\$30.60}
\]

---

## 6. Relative Efficiency Analysis

### Token Reduction
Compared to direct baselines, LOKI reduces LLM token volume substantially:

- **vs. Direct Qwen-3.6:**
  \[
  1 - \frac{2,750,400}{8,434,285} = \mathbf{67.4\% \text{ reduction}}
  \]
- **vs. Direct Qwen-3.7-Max:**
  \[
  1 - \frac{2,750,400}{8,819,207} = \mathbf{68.8\% \text{ reduction}}
  \]

### Cost Efficiency
- **Direct Qwen-3.6 (\$7.76)** is **3.1× more expensive** than **LOKI + Qwen-3.6 (\$2.53)**, representing a 67.4% cost savings under identical per-token pricing.
- **Direct Qwen-3.7-Max (\$30.60)** is **12.1× more expensive** than **LOKI + Qwen-3.6 (\$2.53)**, representing a 91.7% cost savings.
- **Direct Qwen-3.7-Max (\$30.60)** is **43.7× more expensive** than **LOKI + GPT-OSS 20B (\$0.70)**, representing a 97.7% cost savings.

---

## 7. Architectural Takeaways

LOKI's cost and compute advantage stems from two architectural principles:

1. **Targeted LLM Utilization:** Rather than processing raw tabular dumps and multi-hop reasoning end-to-end inside an LLM, LOKI handles join path extraction, semantic sentence averaging, and topological clustering with dedicated, lightweight algorithms. The LLM is invoked strictly for semantic relation labeling on high-confidence cluster centroids, reducing total token traffic by over 67%.
2. **Decoupled Model Tiering:** Because the LLM is restricted to a structured labeling task on pre-filtered evidence, LOKI performs reliably with highly cost-effective open-weight models (such as GPT-OSS 20B at \$0.0018/admission or Qwen-3.6-35B at \$0.0066/admission). In contrast, monolithic direct prompting necessitates costly frontier API endpoints (\$0.0801/admission) to maintain reasoning capability, driving a 12× to 44× increase in operational inference costs.
