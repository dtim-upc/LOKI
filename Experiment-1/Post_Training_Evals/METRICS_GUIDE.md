# Metrics Guide for Post-Training Evaluation

This document explains all key metrics used in this codebase, with small examples and interpretation guidance.

## 1) What Constitutes One Test Example?

In this project, one test example is one anchored clinical instance containing:

- an `anchor_id` (links example to annotation/admission mapping),
- a table (or table type) with multiple rows (query side),
- a `primary_positive` clinical note represented as a set of sentences (document side),
- annotation-derived ground-truth row-sentence links for that instance.

Note:
- Diagnosis and medication tables are intentionally kept separate for the same admission. This enables table-type-specific evaluation now and supports later integration/relationship-level evaluation across table types.

Evaluation is performed **within that example**:

- candidate pairs = all row-sentence combinations from that example only,
- ground truth = annotated positive pairs for that same example.

So the row-sentence ranking metrics are not corpus-wide note retrieval metrics; they are per-example grounding metrics and then aggregated across examples.

## 2) Core Setup: What Is a "Pair"?

For one test example:

- We have `num_rows` table rows.
- We have `num_sentences` note sentences.
- The model produces a score matrix of shape `[num_rows, num_sentences]`.
- Each cell `(row_i, sent_j)` is one candidate row-sentence pair.

So if there are 4 rows and 5 sentences, total candidate pairs = `4 * 5 = 20`.

Ground truth consists of annotated positive pairs, e.g.:
- `GT = {(0,1), (2,4), (3,0)}`

---

## 3) Ranking Metrics (@K)

These are computed on the flattened list of pair scores, sorted descending.

### 2.1 Precision@K

Definition (per example):
- `Precision@K = TP@K / K`

Where:
- `TP@K` = number of ground-truth pairs appearing in top-K predictions.

Example:
- Top-5 contains 3 true pairs.
- `Precision@5 = 3/5 = 0.60`.

### 2.2 Recall@K

Definition (per example):
- `Recall@K = TP@K / |GT|`

Example:
- `|GT| = 8`, and top-5 contains 3 true pairs.
- `Recall@5 = 3/8 = 0.375`.

### 2.3 F1@K

Definition (per example):
- `F1@K = 2 * Precision@K * Recall@K / (Precision@K + Recall@K)`

Example:
- `Precision@5 = 0.60`, `Recall@5 = 0.375`
- `F1@5 = 2*(0.6*0.375)/(0.6+0.375) = 0.4615`.

### 2.4 NDCG@K

Intuition:
- Rewards placing true pairs high in ranking, with logarithmic discount by rank.

For binary relevance:
- `DCG@K = sum(1/log2(rank+1))` over relevant items in top-K.
- `NDCG@K = DCG@K / IDCG@K` where `IDCG@K` is ideal DCG.

Range: `[0, 1]`, higher is better.

### 2.5 MRR@K (Optional in Paper)

Definition:
- Reciprocal rank of first relevant item within top-K; 0 if none.

Example:
- First correct pair at rank 2 => `MRR = 1/2 = 0.5`.

Note:
- MRR can look similar across models when all models already place at least one correct pair very early.

### 2.6 Mean Rank

Definition:
- Average rank position of all ground-truth pairs in the ranked list.

Range:
- Not bounded to `[0,1]`. Lower is better.

Interpretation:
- If average candidates/query is ~140 and Mean Rank is 70, GT pairs are on average in the top half.

---

## 4) How Test-Set Aggregation Works

In this codebase, ranking metrics are computed per example, then macro-averaged:

- `Final Precision@K = mean over examples of Precision@K`
- same for Recall@K, F1@K, NDCG@K, MRR@K

Mean Rank is averaged across examples (ignoring `inf` when no GT present).

This is **macro averaging** over queries/examples.

---

## 5) Raw Ranking Counts (for Paper Transparency)

### 4.1 `hits@K`

Total number of GT pairs retrieved in top-K across all queries.

### 4.2 `max_hits@K`

Theoretical maximum retrievable GT at K:
- `max_hits@K = sum_q min(|GT_q|, K)`

### 4.3 Why `hits@K / max_hits@K` != plotted Precision@K

- `hits@K / max_hits@K` is a **ceiling-normalized hit coverage**.
- Plotted `Precision@K` is macro query precision.

They answer different questions and should not be mixed.

---

## 6) Pair Classification Metrics (Dynamic Threshold)

These are **not** ranking@K metrics. They are threshold-based classification metrics over all candidate pairs.

### 5.1 Dynamic Threshold Rule

For each example:

1. Collect GT pair scores: `GT_scores`.
2. Collect non-GT pair scores: `NonGT_scores`.
3. Set threshold:
   - `threshold = (mean(GT_scores) + mean(NonGT_scores)) / 2`
4. Predict positive if `score >= threshold`.
5. Compute TP/FP/FN, then precision/recall/F1.

Fallback:
- If one side is empty, threshold falls back to median of all pair scores.

### 5.2 Why Precision Can Be Low with High TP

Because candidate space is large and sparse:
- Candidate pairs can be tens of thousands.
- GT pairs are much fewer.
- Dynamic threshold may produce many predicted positives to keep recall high.
- This can increase FP and reduce precision.

So high TP and low precision can coexist.

---

## 7) Curve Metrics

### 6.1 ROC-AUC

- Uses all pair labels/scores pooled across examples.
- Measures discrimination ability independent of threshold.
- Range `[0,1]`, higher is better.

### 6.2 Average Precision (AP)

- Area under Precision-Recall curve.
- Also threshold-free and robust for class imbalance.
- Commonly used as primary metric in retrieval-like settings.

---

## 8) Diagnosis vs Medication Metrics

The evaluator computes metrics separately for:
- `diagnosis`
- `medication`

Then macro-averages these table types for overall summary values.

Useful raw fields:
- `diagnosis_prediction_breakdown`: TP/FP/FN
- `medication_prediction_breakdown`: TP/FP/FN

---

## 9) Recommended Reporting for VLDB-Style Results

For main result table:

- Ranking: `P@1, P@5, P@10, R@1, R@5, R@10, F1@1, F1@5, F1@10, NDCG@1, NDCG@5, NDCG@10`
- Support: `Mean Rank (raw, lower better)`
- Transparency: `queries, ground-truth pairs, candidate pairs, hits@K/max_hits@K`

For grounding/classification table:

- `Dynamic F1`, `AP`, `ROC-AUC`, and TP/FP/FN (overall + diagnosis + medication)

For dataset table:

- split sizes, row/sentence counts, sentence length stats, annotation coverage.

---

## 10) Quick Terminology Map

- **Candidate pair**: any `(row, sentence)` pair scored by model.
- **Ground-truth pair**: annotated positive `(row, sentence)`.
- **Non-GT pair**: candidate pair not in GT set.
- **Query/example**: one admission/table instance with its row and sentence set.
- **Flattened ranking**: converting score matrix into one sorted list of pairs.

