# Relationship-Type Table Materialization Evaluation

Each predicted cluster represents one materialized table corresponding to a distinct relationship type. Evaluation is performed at both the individual entity-pair level and the aggregated table level across the MIMIC-IV evaluation set.

- **Predicted Table:** One materialized same-type table within an admission.
- **Ground Truth (GT) Table:** One reference same-type table within an admission.
- **Best-Match Typed-Pair Macro P/R/F1:** Mean per-admission pair scores obtained after mapping each predicted table to its optimal ground-truth relationship type.
- **Best-Match Typed-Pair Micro P/R/F1:** Global pooled pair scores across all admissions on the best-match surface.
- **Typed Table Materialization Macro P/R/F1:** Unweighted mean of per-admission table-level scores.
- **Typed Table Materialization Micro P/R/F1:** Global pooled scores from cumulative typed table counts across all admissions.

---

## Metric Formulations

### 1. Best-Match Typed-Pair Metrics
- **Pred pairs:** Number of predicted typed entity pairs on the best-match surface.
- **GT pairs:** Number of ground-truth typed entity pairs on that surface.
- **TP:** Predicted typed pairs whose oracle-assigned relationship type matches the reference pair type.
- **FP:** Predicted typed pairs that do not match a reference pair of that type.
- **FN:** Reference typed pairs not recovered by any oracle-assigned predicted typed pair.

\[
\text{Precision} = \frac{\text{TP}}{\text{Pred pairs}}, \quad \text{Recall} = \frac{\text{TP}}{\text{GT pairs}}, \quad \text{F1} = \frac{2\text{TP}}{\text{Pred pairs} + \text{GT pairs}}
\]

### 2. Typed Table Materialization Metrics
- **Pred tables:** Total predicted typed table objects.
- **GT tables:** Total reference typed table objects.
- **TP:** Predicted tables whose assigned relationship type exactly matches the reference table type.
- **FP:** Predicted tables whose assigned relationship type does not match any reference table of that type.
- **FN:** Reference tables not matched by any predicted table of that type.

\[
\text{Precision} = \frac{\text{TP}}{\text{Pred tables}}, \quad \text{Recall} = \frac{\text{TP}}{\text{GT tables}}, \quad \text{F1} = \frac{2\text{TP}}{\text{Pred tables} + \text{GT tables}}
\]

---

## Summary Results

| System | Scope | Admissions | Best-Match Typed-Pair Macro P/R/F1 | Typed Table Materialization Macro P/R/F1 | Typed Table Materialization Micro P/R/F1 |
| --- | --- | ---: | --- | --- | --- |
| **LOKI + GPT-OSS 20B** | Full | 378 | 0.982 / 0.486 / 0.627 | 0.755 / 0.755 / 0.742 | 0.840 / 0.840 / 0.840 |
| **LOKI + Qwen-3.6** | Full | 380 | 0.982 / 0.515 / 0.652 | 0.750 / 0.732 / 0.726 | 0.848 / 0.807 / 0.827 |
| **Direct Qwen-3.7-Max** | Full | 381 | 0.997 / 0.717 / 0.817 | 0.966 / 0.963 / 0.964 | 0.958 / 0.958 / 0.958 |
| **Direct Qwen-3.6-Local** | Full | 381 | 0.993 / 0.540 / 0.678 | 0.932 / 0.929 / 0.930 | 0.920 / 0.920 / 0.920 |

---

## Raw Pair-Level Counts (Best-Match Surface)

| System | Scope | Pred pairs | GT pairs | TP | FP | FN | Micro P | Micro R | Micro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LOKI + GPT-OSS 20B** | Full | 2,882 | 6,441 | 2,823 | 59 | 3,618 | 0.9795 | 0.4383 | 0.6056 |
| **LOKI + Qwen-3.6** | Full | 3,051 | 6,454 | 2,990 | 61 | 3,464 | 0.9800 | 0.4633 | 0.6291 |
| **Direct Qwen-3.7-Max** | Full | 4,149 | 6,468 | 4,135 | 14 | 2,333 | 0.9966 | 0.6393 | 0.7789 |
| **Direct Qwen-3.6-Local** | Full | 3,103 | 6,468 | 3,086 | 17 | 3,382 | 0.9945 | 0.4771 | 0.6449 |

---

## Raw Table-Level Counts (Typed Materialization)

| System | Scope | Pred tables | GT tables | TP | FP | FN | Micro P | Micro R | Micro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LOKI + GPT-OSS 20B** | Full | 2,018 | 2,018 | 1,696 | 322 | 322 | 0.8404 | 0.8404 | 0.8404 |
| **LOKI + Qwen-3.6** | Full | 2,011 | 2,113 | 1,705 | 306 | 408 | 0.8478 | 0.8069 | 0.8269 |
| **Direct Qwen-3.7-Max** | Full | 548 | 548 | 525 | 23 | 23 | 0.9580 | 0.9580 | 0.9580 |
| **Direct Qwen-3.6-Local** | Full | 523 | 523 | 481 | 42 | 42 | 0.9197 | 0.9197 | 0.9197 |

---

## Technical Characteristics & Analysis

1. **Precision Dominance Across Systems:** All evaluated pipelines achieve very high pair precision ($\ge 97.9\%$), demonstrating that once a candidate pair is selected into a relationship cluster, semantic misclassification is rare.
2. **Table-Level Quality:** LOKI achieves $84.0\%$ to $84.8\%$ micro precision on materialized tables, confirming robust cluster purity when grouping discovered join paths into relational tables.
3. **Macro vs. Micro Aggregations:** Macro scores reflect the unweighted mean across admissions, whereas micro scores aggregate raw instances globally across the entire test set. For balanced admission distributions, macro and micro values align closely.
4. **Error Symmetry:** In balanced configurations where the number of predicted tables matches ground truth tables, false positives and false negatives are equal ($FP = FN = 322$ for LOKI + GPT-OSS), as each table misclassification simultaneously introduces one false positive under the predicted label and one false negative under the target label.