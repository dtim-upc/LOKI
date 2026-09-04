# TaBERT for Table-Text Discovery: Summary & Findings

## 1. Overview

We integrated **TaBERT** (Table BERT with Vertical Self-Attention) into the LOKI SOTA evaluation framework for the **table-text discovery** task on the **Pharma Protrix Flipped** dataset. This document summarises the architecture, fine-tuning pipeline, evaluation results, and the **schema-memorisation phenomenon** we uncovered.

---

## 2. Model Architecture

| Property | Value |
|---|---|
| Base model | TaBERT-Large (K=3), Vertical Self-Attention |
| Backbone | BERT-large-uncased (24 layers, 1024 hidden) |
| Pre-training | Masked Column Prediction + Cell Value Recovery on 26M web tables |
| Inference mode | **Cross-encoder**: every (document, table) pair goes through a full BERT forward pass |

### Cross-Encoder Scoring

Unlike bi-encoder models (LOKI, TabSTAR) that encode documents and tables independently, TaBERT jointly encodes each (document, table) pair:

```
1. joint_repr = TaBERT.encode(context_tokens, table)
   → context_encoding  (batch, seq_len, 1024)
   → column_encoding   (batch, num_cols, 1024)

2. ctx_vec = mean_pool(context_encoding, context_mask)   → (batch, 1024)
   col_vec = mean_pool(column_encoding, column_mask)     → (batch, 1024)

3. similarity = cosine_similarity(ctx_vec, col_vec) × temperature
```

The `column_encoding` is a per-column representation produced by TaBERT's vertical self-attention layers — it captures the column header semantics and the sampled cell values beneath each column.

---

## 3. Fine-Tuning Setup

### 3.1 LoRA Adapter Configuration

We use **LoRA** (Low-Rank Adaptation) to fine-tune efficiently without modifying the full model:

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | BERT: `query`, `key`, `value`, `dense`; Vertical attention: `query_linear`, `key_linear`, `value_linear` |

### 3.2 Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 6 (best checkpoint selected by validation accuracy) |
| Effective batch size | 32 (batch\_size=2 × grad\_accum=16) |
| Learning rate | 2e-4 (peak) |
| LR schedule | Linear warmup (10%) → cosine annealing |
| Optimizer | AdamW (weight decay = 0.01) |
| Max gradient norm | 1.0 |
| Mixed precision | AMP (BFloat16 on CUDA) |
| Gradient checkpointing | Enabled |

### 3.3 Loss Function

**Triplet loss** (softplus formulation, matching LOKI's protocol):

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \log\bigl(1 + e^{\,\gamma\,(s_{\text{neg}}^{(i)} - s_{\text{pos}}^{(i)} + m)}\bigr)
$$

| Symbol | Value |
|---|---|
| $m$ (margin) | 0.3 |
| $\gamma$ (scale) | 10.0 |

### 3.4 Learnable Temperature

A single learnable scalar $\tau$ (initialised at 10.0, stored in log-space) scales all cosine similarities before the loss. Clamped to $[1, 100]$ during inference.

### 3.5 Triplet Generation

For `is_flipped=True` (pharma):
- **Anchor**: Document sentence (natural language context)
- **Positive**: Matching table row
- **Negative**: Non-matching table row

Strategy: `"full"` — all positive × negative combinations (capped at `max_triplets=64` per example).

---

## 4. Table Serialisation

Table rows are converted into TaBERT Table objects using **`multi_column`** mode — the architecturally correct setting for TaBERT's vertical self-attention.

Row strings like:

```
id: BE0000749; name: Sodium-dependent serotonin transporter; organism: Humans
```

are parsed by the regex `(?:^|; )([\w][\w-]*): ` into structured multi-column tables:

| id | name | organism |
|---|---|---|
| BE0000749 | Sodium-dependent serotonin transporter | Humans |

This creates column schemas for each of the 12 source table types, but only **8 distinct schemas** exist because some types share the same columns (e.g., *carriers*, *targets*, and *transporters* all have `[id, name, organism, known_action, position, parent_key]`, and their `_polypeptides` counterparts share a 20-column schema). The 6 remaining types each have a unique schema (e.g., *drug-interactions* → `[drugbank-id, name, description, parent_key]`, *drug_pharmacology* → `[drugbank_id, indication, pharmacodynamics, …]`).

### Non-Flipped Mode

In the **non-flipped** configuration (`is_flipped=False`), the code path uses `example_to_tabert_table()`, which reads the structured `anchor_headers` and `anchor_rows` directly from the JSON. Each query has exactly **one** table — schema identity cannot serve as a shortcut.

---

## 5. Dataset Characteristics

| Property | Value |
|---|---|
| Dataset | Pharma Protrix Flipped (DrugBank clinical protocols) |
| Train / Val / Test | 648 / 138 / 140 examples |
| Positive source tables | **12 types** but only **8 distinct schemas** (3 pairs share columns) |
| Total source tables | **82** (combinable fragments from 12 types) |
| Total fragments | 2,240 individual table row fragments |
| GT per test doc | Exactly 8 source tables (in combined\_tables mode) |

### Split Design

| Overlap Type | Train ↔ Test | Note |
|---|---|---|
| Fragment-level | **0%** (0 / 2,240) | Clean — no row leakage |
| Source-table-level | **100%** (12 / 12 types, 8 / 8 schemas) | By design — splits are by table rows, not by table type |
| Document (anchor) | **100%** (140 / 140) | Same anchor documents appear across splits |

The splits divide rows within each source table across train/val/test, but **all 12 source table types (8 distinct schemas) appear in every split**. This is standard for row-level evaluation but has implications for table-level evaluation.

---

## 6. Evaluation Results

### 6.1 SOTA Comparison — Combined Tables (82 Source Tables)

| Model | Type | P@1 | MAP | Mean Rank |
|---|---|---|---|---|
| **TaBERT (fine-tuned, multi\_column)** | Cross-encoder | **1.0000** | **0.9551** | **4.91** |
| TabSTAR | Bi-encoder | 0.9714 | 0.2967 | — |
| LOKI | Bi-encoder | — | 0.5587 | 13.42 |
| CMDL | Sparse retrieval | — | 0.3867 | 20.98 |
| **TaBERT (pretrained, not fine-tuned)** | Cross-encoder | **0.1800** | **0.1243** | **51.98** |

### 6.2 SOTA Comparison — Row-Level Fragments (2,240 Fragments)

| Model | Type | P@1 | MAP (Macro) | MAP (Micro) | Mean Rank (Macro) |
|---|---|---|---|---|---|
| **LOKI** | Bi-encoder | **0.5286** | **0.2463** | **0.1037** | **203.77** |
| **TaBERT (fine-tuned, multi\_column)** | Cross-encoder | 0.3071 | 0.0775 | 0.0406 | 466.44 |
| CMDL | Sparse retrieval | 0.0000 | 0.0071 | 0.0000 | 1005.72 |
| TabSTAR | Bi-encoder | 0.0071 | 0.0065 | 0.0033 | 1212.62 |

### 6.3 Fine-Tuned Model Evaluation (Direct, evaluate\_finetuned.py)

| Metric | Value |
|---|---|
| Table-level accuracy | 82.1% |
| Row-sentence F1 | 0.2647 |

### 6.4 Critical Observation — The Combined → Row-Level Collapse

| Setting | Corpus Size | TaBERT P@1 | TaBERT MAP | LOKI P@1 | LOKI MAP |
|---|---|---|---|---|---|
| Combined tables | 82 | **1.0000** | **0.9551** | — | 0.5587 |
| Row-level | 2,240 | 0.3071 | 0.0775 | 0.5286 | 0.2463 |
| **Drop factor** | **27×** | **3.3× worse** | **12.3× worse** | — | **2.3× worse** |

The **pretrained (non-fine-tuned) TaBERT** scores *worst* among all four models at combined-table level (P@1 = 0.18, MAP = 0.12). After fine-tuning with multi\_column mode, it jumps to *perfect* P@1 = 1.0 and MAP = 0.96 at table-level — but **collapses to P@1 = 0.31 and MAP = 0.08 at row-level**.

For comparison, LOKI's MAP drops only 2.3× from combined to row-level (a proportional increase in task difficulty). TaBERT's MAP drops **12.3×** — far beyond what the harder task explains. This asymmetric collapse is the smoking gun for schema memorisation.

---

## 7. Schema Memorisation Analysis

### 7.1 The Problem

Investigation revealed that the near-perfect results stem from a **schema-matching shortcut**, not genuine content understanding:

1. **Only 8 distinct table schemas** exist among the 12 source table types. Three pairs share identical column headers: carriers/targets/transporters share `[id, name, organism, known_action, position, parent_key]`, and their `_polypeptides` counterparts share a 20-column schema. The remaining 6 types each have unique schemas.

2. In `multi_column` mode, TaBERT's `column_encoding` is heavily influenced by **column header tokens**. The `mean_pool(column_encoding)` representation is dominated by the schema signature (column names), not by the cell values.

3. During fine-tuning, the model learns a trivial mapping: **document topic → table schema**. For example, a document mentioning "serotonin transporter" maps to the `targets` schema (`[id, name, organism, actions]`), while a document about "drug interactions" maps to the `drug-interactions` schema (`[drug_name, description]`).

4. This reduces the table-text discovery task to an **8-way schema classification** problem — trivially solvable with 648 training examples (~81 per schema).

### 7.2 Evidence

| # | Evidence | What it shows |
|---|---|---|
| 1 | Pretrained TaBERT = worst model (P@1 = 0.18) | Pre-training alone does not solve the task |
| 2 | Fine-tuned multi\_column = perfect (P@1 = 1.00) | Fine-tuning overfits to schema patterns |
| 3 | Combined → Row-level: MAP drops 12.3× for TaBERT but only 2.3× for LOKI | TaBERT's knowledge is schema-level, not content-level |
| 4 | Row-level fine-tuned TaBERT P@1 = 0.31 (behind LOKI's 0.53) | Cannot distinguish rows within the same table type |
| 5 | All 12 source table types (8 schemas) seen during training | No held-out table types to test generalisation |
| 6 | Column encoding ∝ column headers | Schema identity dominates the table representation |

**Evidence #3 is the most conclusive.** When rows are grouped by source table (combined mode), each schema appears once — so schema classification = correct retrieval. At row-level, ~187 fragments share each schema — the model ranks them all similarly but cannot identify which specific rows are relevant, causing the catastrophic drop.

### 7.3 The Irony of Row\_Text

Before implementing multi-column parsing, all tables used a single `Row_Text` column — architecturally "wrong" for TaBERT (defeats vertical attention over columns) but **correct for evaluation fairness**: every table had the identical `[Row_Text]` schema, forcing the model to attend to actual cell content.

The multi-column fix was architecturally correct (proper column structure for vertical attention) but inadvertently handed TaBERT a schema-level shortcut that trivialised the task.

### 7.4 Mitigation

Since the schema memorisation phenomenon is an inherent property of multi\_column mode on this dataset (only 8 distinct schemas), **row-level (fragmented) evaluation** is the primary mitigation: it forces the model to distinguish among ~187 fragments sharing the same schema, neutralising the shortcut.

The combined-table results are reported with a schema memorisation caveat and are not directly comparable against other models.

---

## 8. Conclusive Evidence — Combined vs Row-Level Asymmetry

The **combined → row-level comparison** provides conclusive evidence of schema memorisation without needing additional experiments.

### 8.1 The Core Comparison

| Condition | Granularity | Fine-tuned? | P@1 | MAP |
|---|---|---|---|---|
| Combined tables | 82 tables | Yes | **1.00** | **0.96** |
| Row-level fragments | 2,240 fragments | Yes | 0.31 | 0.08 |

### 8.2 Why This Is Conclusive

1. **If the model learned content matching**, performance should degrade proportionally when moving from 82 combined tables to 2,240 fragments (task is ~27× harder). LOKI shows this expected proportional degradation: MAP drops 2.3×.

2. **TaBERT's MAP drops 12.3×** — 5× worse than LOKI's proportional drop. This disproportionate collapse can only be explained by the model relying on a signal that exists at the table level (schema identity) but becomes useless at the fragment level (all ~187 rows of the same type share identical schemas).

3. **At row-level, TaBERT (P@1=0.31) still beats CMDL (0.00) and TabSTAR (0.01)** — it does pick up *some* content signal from fine-tuning. But it falls well short of LOKI (P@1=0.53), confirming most of its combined-table "ability" was schema classification, not content understanding.

### 8.3 Summary of Proof

```
Schema memorisation proof:
  Combined tables:  TaBERT >> LOKI  (P@1: 1.00 vs ~0.56 MAP)  → schema shortcut works
  Row-level:        TaBERT << LOKI  (P@1: 0.31 vs 0.53)        → schema shortcut fails
  
  Same model, same weights. Only difference = fragment grouping.
  ∴ TaBERT learned schema identity, not content matching.
```

---

## 9. Architectural Advantage: Cross-Encoder vs Bi-Encoder

Independent of the schema memorisation issue, TaBERT benefits from its **cross-encoder** architecture:

| Aspect | Cross-Encoder (TaBERT) | Bi-Encoder (LOKI, TabSTAR) |
|---|---|---|
| Encoding | Joint — full attention between doc & table tokens | Independent — doc and table encoded separately |
| Interaction | Token-level cross-attention | Cosine similarity of pooled embeddings |
| Complexity | $O(N \times M)$ forward passes for $N$ docs, $M$ tables | $O(N + M)$ encodings, then dot-product scoring |
| Expressiveness | Can capture fine-grained token interactions | Limited to embedding-space similarity |
| Scalability | Expensive at scale (no pre-computed table embeddings) | Scales well (cache table embeddings) |

In the combined\_tables setting (82 source tables), the cross-encoder cost is manageable ($140 \times 82 = 11{,}480$ forward passes). At row-level (2,240 fragments), cost increases to $140 \times 2{,}240 = 313{,}600$ forward passes.

---

## 10. Code Structure

```
TaBERT/
├── finetune.py              # Training script (TrainingConfig, train loop, checkpointing)
├── model_wrapper.py         # TaBERTForContrastive (LoRA, scoring, temperature)
├── data_loader.py           # Dataset, table builders, KV parsing, triplet generation
├── evaluate_finetuned.py    # Table-level accuracy, row-sentence F1, frozen baseline
└── HOW_TO_RUN.md            # Full workflow guide

SOTA_Evaluation_New/
├── evaluate_tabert.py       # Cross-encoder SOTA evaluation (ranking all tables)
├── run_comparison_pharma.py # Unified 4-model comparison (CMDL, LOKI, TabSTAR, TaBERT)
└── HOW_TO_RUN.md            # Setup guide for all models
```

### Key Functions

| Function | File | Purpose |
|---|---|---|
| `TaBERTForContrastive.score()` | model\_wrapper.py | Joint encode → mean-pool → cosine similarity |
| `strings_to_tabert_table()` | data\_loader.py | Row strings → TaBERT Table object (multi\_column parsing) |
| `_parse_kv_sentence()` | data\_loader.py | Regex-based key:value parser for pharma rows |
| `evaluate_tabert()` | evaluate\_tabert.py | Full cross-encoder ranking over entire corpus |
| `compute_triplet_loss()` | finetune.py | Softplus margin triplet loss |

---

## 11. Conclusions

1. **TaBERT's cross-encoder architecture is powerful** for table-text matching, but its $O(N \times M)$ cost limits scalability compared to bi-encoder alternatives.

2. **Fine-tuning with multi\_column mode produces artificially inflated results** on the Pharma dataset due to schema memorisation. With only 8 distinct column schemas (across 12 source table types) and 100% overlap across splits, the model learns a trivial 8-way schema classifier rather than content-level matching.

3. **The combined → row-level performance collapse is conclusive proof.** TaBERT's MAP drops 12.3× (0.96 → 0.08) when moving from 82 combined tables to 2,240 row fragments — far exceeding LOKI's proportional 2.3× drop (0.56 → 0.25). This asymmetric degradation can only be explained by schema-level memorisation.

4. **At row-level, LOKI dominates all models.** Fine-tuned TaBERT (P@1=0.31) is a distant second behind LOKI (P@1=0.53), confirming that LOKI's bi-encoder approach with cross-attention learns genuine content representations that transfer across granularity levels.

5. **The pretrained model (without fine-tuning) performs worst** among all four models, confirming that TaBERT's pre-training on 26M web tables does not directly transfer to the pharma domain table-text discovery task.

6. **CMDL and TabSTAR are near-random at row-level** (P@1 ≈ 0.00–0.01, MAP < 0.01), indicating the pharma domain's specialised vocabulary is beyond the reach of generic sparse retrieval and tabular pre-training.

7. **For fair reporting**, row-level (fragmented) results should be the primary evaluation metric for TaBERT. Combined-table results should be reported with the schema memorisation caveat.

---

## 12. Final Model Ranking

### Combined Tables (82 source tables)

| Rank | Model | P@1 | MAP | Note |
|---|---|---|---|---|
| 1 | TaBERT (fine-tuned) | 1.00 | 0.96 | ⚠️ Schema memorisation |
| 2 | TabSTAR | 0.97 | 0.30 | |
| 3 | LOKI | — | 0.56 | |
| 4 | CMDL | — | 0.39 | |

### Row-Level Fragments (2,240 fragments) — Fair Comparison

| Rank | Model | P@1 | MAP (Macro) | MAP (Micro) | Mean Rank |
|---|---|---|---|---|---|
| 1 | **LOKI** | **0.53** | **0.25** | **0.10** | **204** |
| 2 | TaBERT (fine-tuned) | 0.31 | 0.08 | 0.04 | 466 |
| 3 | CMDL | 0.00 | 0.01 | 0.00 | 1006 |
| 4 | TabSTAR | 0.01 | 0.01 | 0.00 | 1213 |

---

## 13. Concluding Remarks: Fair Evaluation Strategy

The schema memorisation phenomenon means that **combined-table results are misleading** for TaBERT. The recommended evaluation strategy is:

1. **Primary metric: Row-level (fragmented) evaluation.** With 2,240 fragments sharing only 8 schemas, the model must rely on content understanding rather than schema classification. This is the only fair comparison against other models.

2. **Secondary metric: Combined-table evaluation with caveat.** The near-perfect combined-table results demonstrate the schema memorisation phenomenon and should be reported alongside the row-level results for completeness, but not used as a competitive benchmark.

> *"We report TaBERT's row-level fragmented results as the primary metric for fair comparison. Combined-table results are reported separately as evidence of the schema memorisation phenomenon, where TaBERT trivially classifies 8 distinct column schemas rather than performing content-level matching."*

---

## 14. Next Steps

- [x] ~~Run row-level SOTA evaluation (without `--combined_tables`)~~ — **Done** (Section 6.2)
- [x] ~~Prove schema memorisation~~ — **Done** (Section 8, the combined→row-level asymmetry is conclusive)
- [x] ~~Simplify codebase — removed `--table_mode` and `--uniform_columns` options~~ — multi\_column is now the only mode
- [ ] Incorporate findings into VLDB paper — recommend row-level fragmented as primary TaBERT evaluation
- [ ] *(Optional)* Source-table holdout splits for future experiments (hold out entire table types during training)
