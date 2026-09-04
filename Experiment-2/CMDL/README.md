# CMDL (Experiment-2: Comparative Evaluation)
[Cross-Modal Data Discovery over Structured and Unstructured Data Lakes](https://www.vldb.org/pvldb/vol16/p3377-eltabakh.pdf)  
Published at Very Large Databses (VLDB) 2023

### Entry points (LOKI Pipeline Evaluation):
- `run_pharma_cmdl.py`: The main full-pipeline script for document-to-table discovery. It handles data loading from `pharma_flipped_structured` JSON, table/text indexing, feature building, triplet-loss model training, and Annoy-based KNN retrieval evaluation in one execution. Table columns are extracted directly from structured `headers` + `rows[].content` arrays (no formatted-string parsing).

- `evaluate_cmdl.py` (in `../SOTA_Evaluation`): A standalone evaluation script for pre-trained models.

### Entry points (Legacy CMDL Notebooks):
- `trainer/pretrain-text.ipynb`: Fine tuning a language model on text corpus to learn text embeddings
- `trainer/pretrain-tables.ipynb`: Fine tuning a language model on table collection to learn tuple embeddings
- `trainer/column_text_joint_training.ipynb`: training a baseline connecting text to table columns
- `compare_gt.py`: accuracy measurement of search based baselines and similarity sketches on text->table relation discovery using the ground truth provided

# Instruction to Run on Pharma Data

## What Was Created

`run_pharma_cmdl.py` — CMDL training for **Doc-to-Table** on Pharma (PubMed + DrugBank).

## Pipeline

| Phase | Description |
|---|---|
| **0: Split & Copy** | Reads GT, splits text IDs 70/15/15, copies files into `pharma_data/{split}/` |
| **1: Build Features** | Profiles texts/tables, creates fastText WEM features (cached) |
| **2: Train** | CMDL joint training with TripletLoss, dual DataLoaders |
| **3: Evaluate** | P@K / R@K / F1@K on val and test splits |

## How to Run

```bash
# Full pipeline (abstract + keywords)
python run_pharma_cmdl.py

# Abstract-only mode (no MeSH keywords — fairer comparison)
python run_pharma_cmdl.py --abstract_only --force_split

# Eval/test only (requires prior training)
python run_pharma_cmdl.py --eval_only
python run_pharma_cmdl.py --test_only
```

### Abstract-Only Mode

Each PubMed Target file has:
- **Line 1**: The abstract text
- **Lines 2+**: MeSH keywords (one per line, e.g. "Cerebral Cortex", "Clomipramine")

The `--abstract_only` flag strips keywords during data splitting, keeping only the abstract. This tests the model without keyword metadata, which could otherwise provide conceptual shortcuts. Results are saved to separate directories (`pharma_data_abstract_only/`, `output_cmdl_pharma_abstract_only/`) for side-by-side comparison.

## Evaluation Methodology

Matches the original CMDL codebase exactly:

1. **Column-level ANN search** — Annoy index over all 416 column embeddings; for each text query, retrieve `2×K` nearest columns
2. **Table-level aggregation** — group retrieved columns by parent table, sum similarity scores (`1/(1+distance)`) per table
3. **Top-K tables** — rank by aggregated score, return top K
4. **Metrics** — micro-averaged P@K, R@K, F1@K against table-level GT

## 500-Epoch Results (Full Text)

### Test Set (140 texts)

| K | P@K | R@K | F1@K |
|:---:|:---:|:---:|:---:|
| 1 | 0.7929 | 0.0989 | 0.1759 |
| 3 | 0.7887 | 0.2861 | 0.4199 |
| 5 | 0.7976 | 0.4144 | 0.5455 |
| 10 | 0.7764 | 0.5508 | 0.6444 |

Full results report: `output_cmdl_pharma/CMDL_Pharma_Evaluation_Report.md`

## Output Structure

```
pharma_data/                          # Full text data
pharma_data_abstract_only/            # Abstract-only data
output_cmdl_pharma/                   # Full text results
output_cmdl_pharma_abstract_only/     # Abstract-only results
├── features/{split}/                 # Cached features
├── models/                           # Saved model checkpoints
├── val_results.json
├── test_results.json
└── pharma_cmdl_results_summary.txt
```


# Running CMDL on MIMIC Data (Doc-to-Table)

## Prerequisites

1. **Python environment** with dependencies:
   ```
   pip install torch numpy gensim annoy tqdm spacy
   python -m spacy download en_core_web_sm
   ```

2. **FastText model** — download the Facebook Common Crawl model:
   ```
   mkdir -p resources/fasttext/cc
   # Download from: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
   # Extract cc.en.300.bin into resources/fasttext/cc/
   ```
   On first run, the pipeline auto-converts this to gensim format (`.gensim`) for faster future loads.

3. **MIMIC data** — place LOKI-format JSON files in `mimic_data/`:
   ```
   CMDL/
   ├── mimic_data/
   │   ├── train_row_level.json
   │   ├── val_row_level.json
   │   └── test_row_level.json
   ```

---

## Quick Start

### Recommended: CMDL-Style Pipeline (faithful joint-training mechanics)
Uses original CMDL column-text joint training mechanics (independent text/column mini-batches + label sub-matrix triplet training), while replacing Snorkel weak labels with MIMIC ground-truth links.
```bash
python run_mimic_cmdl_original.py --epochs 500 --text_batch_size 32 --col_batch_size 32 --steps_per_epoch 26
```

### Ensure all examples are touched each epoch (recommended for sanity checks)
This runs enough steps so every text and column example is seen at least once per epoch (the shorter side may repeat).
```bash
python run_mimic_cmdl_original.py --epochs 20 --ensure_all_examples_per_epoch
```

### Paper-style mini-batch sizing (8% of DEs) + paper margin
This follows the paper defaults more closely for mini-batch sizing and margin. It can be memory-heavy on large datasets.
```bash
python run_mimic_cmdl_original.py --epochs 500 --ensure_all_examples_per_epoch --batch_fraction 0.08 --margin 0.2
```

### With dataset-size controls (train/val/test)
```bash
python run_mimic_cmdl_original.py --max_train_examples 10000 --max_eval_examples 1000 --max_test_examples 1000 --epochs 500
```

### Full Dataset (no split subsampling)
```bash
python run_mimic_cmdl_original.py --epochs 500 --max_train_examples 0 --max_eval_examples 0 --max_test_examples 0
```

---

## Modes (`run_mimic_cmdl_original.py`)

### 1. Full Pipeline (default)
Runs all phases: data conversion → feature building → training → evaluation on val + test.
```bash
python run_mimic_cmdl_original.py --epochs 200
```

### 2. Eval Only (`--eval_only`)
Loads a previously saved model and evaluates on val + test (skips training).
```bash
python run_mimic_cmdl_original.py --eval_only --max_test_examples 100
```

### 3. Test Only (`--test_only`)
Loads a previously saved model and evaluates **only on test** (skips train/val entirely).
```bash
python run_mimic_cmdl_original.py --test_only --max_test_examples 50
```

---

## Key Arguments (`run_mimic_cmdl_original.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_train_examples` | 10000 | Subsample training set |
| `--max_eval_examples` | 1000 | Subsample validation set |
| `--max_test_examples` | 0 (all) | Subsample test set |
| `--epochs` | 500 | Max training epochs |
| `--text_batch_size` | 32 | Text mini-batch size |
| `--col_batch_size` | 32 | Column mini-batch size |
| `--batch_fraction` | 0.08 | If `>0`, auto-set mini-batch sizes as fraction of train DEs (paper-style `m,n`) |
| `--steps_per_epoch` | 26 | CMDL notebook-style fixed steps/epoch (`<=0` auto) |
| `--ensure_all_examples_per_epoch` | true | Force enough steps so each text/column example is seen at least once per epoch |
| `--disable_ensure_all_examples_per_epoch` | false | Disable full per-epoch example coverage |
| `--lr` | 0.0001 | Learning rate |
| `--hidden_size` | 200 | Encoder hidden layer size |
| `--output_size` | 100 | Embedding output dimension |
| `--margin` | 0.2 | TripletLoss margin |
| `--neg_weight` | 1.0 | TripletLoss negative term weight |
| `--train_text_fraction` | 1.0 | Fraction of train text features used |
| `--train_col_fraction` | 1.0 | Fraction of train column features used |
| `--seed` | 42 | Random seed for reproducibility |
| `--force_conversion` | false | Force re-convert data even if cached |
| `--output_dir` | `output_cmdl_mimic_original` | Output directory |

---

## Pipeline Phases (CMDL-Style Runner)

### Phase 1: Data Conversion
Converts LOKI JSON → CMDL native format (CSV tables + TXT files + ground truth).
- Output: `<output_dir>/data/{train,val,test}/` (default: `output_cmdl_mimic_original`)
- Skipped automatically if data already exists (use `--force_conversion` to override)

### Phase 2: Feature Building
Profiles texts/tables and builds FastText (WEM) feature vectors.
- Text features: 300-dim WEM embeddings
- Column features: 900-dim (3 × 300) concatenated WEM embeddings (table name, column name, values)
- Output: `<output_dir>/features/{train,val,test}/` (default: `output_cmdl_mimic_original`)

### Phase 3: Training
CMDL-style mini-batch training with TripletLoss on text-column pairs.
- Independent text and column loaders (batch pair per step)
- Label sub-matrix is sliced from full GT label matrix at each step
- Mask keeps anchor rows with both positive and negative columns in-batch
- Model: two EncoderNets (text → embedding, column → embedding)
- Saved: `output_cmdl_mimic_original/models/{text_enet_best.pt, col_enet_best.pt}`

### Phase 4: Evaluation
Builds Annoy index from trained column embeddings, retrieves top-K tables per text query.
- Metrics: P@K, R@K, F1@K at K = 1, 3, 5, 10
- Query side: document embeddings (MIMIC notes)
- Retrieval side: table/column embeddings
- Results: `output_cmdl_mimic_original/{val,test}_results.json`

---

## Output Structure (default)

```
output_cmdl_mimic_original/
├── data/
│   ├── train/          # Converted CSV tables, TXT texts, GT file
│   ├── val/
│   └── test/
├── features/
│   ├── train/          # WEM feature tensors (.npy)
│   ├── val/
│   └── test/
├── models/
│   ├── text_enet_best.pt
│   └── col_enet_best.pt
├── val_results.json
└── test_results.json
```

---

## Notes

- **GPU**: If OOM, lower `--text_batch_size` and/or `--col_batch_size`.
- **Dataset size control**: use `--max_train_examples`, `--max_eval_examples`, and `--max_test_examples` independently.
- **Coverage checks**: use `--ensure_all_examples_per_epoch` when you want full per-epoch pass coverage over supplied train examples.
- **Paper defaults**: try `--batch_fraction 0.08 --margin 0.2` (from the paper), but expect higher memory/time.
- **Subsampling**: Deterministic (seed-based) — same seed always gives same subset.
- **First run**: WEM model conversion (`.bin` → `.gensim`) takes a few minutes but only happens once.
- **Re-runs**: Use `--force_conversion` to regenerate data, or `--skip_features` to reuse cached features.


# CMDL Cross-Dataset Analysis: Pharma vs MIMIC

## Overview

We evaluated CMDL (Cross-Modal Data Lake) on two datasets for the **Doc-to-Table** retrieval task. The model achieves strong results on the Pharma dataset (its native benchmark) but performs near random on the MIMIC dataset. This document analyzes why.

## Results Comparison

### Pharma (PubMed + DrugBank) — Test Set

| K | P@K | R@K | F1@K |
|:---:|:---:|:---:|:---:|
| 1 | **0.7929** | 0.0989 | 0.1759 |
| 3 | **0.7887** | 0.2861 | 0.4199 |
| 5 | **0.7976** | 0.4144 | 0.5455 |
| 10 | **0.7764** | 0.5508 | **0.6444** |

### MIMIC (Clinical Notes + Patient Tables) — Test Set

| K | P@K | R@K | F1@K |
|:---:|:---:|:---:|:---:|
| 1 | 0.2222 | 0.0833 | 0.1212 |
| 3 | 0.1481 | 0.1667 | 0.1569 |
| 5 | 0.1444 | 0.2708 | 0.1884 |
| 10 | 0.0889 | 0.3333 | 0.1404 |

## Dataset Comparison

| Property | Pharma | MIMIC |
|---|---|---|
| **Text source** | PubMed abstracts (biomedical literature) | MIMIC-III clinical discharge summaries |
| **Table source** | DrugBank relational database (82 CSVs) | Per-patient structured tables (1 diagnostic + 1 medication per note) |
| **Text length** | Short abstracts (~150-300 words) + MeSH keywords | Long clinical notes (~500-2000+ words), sparse narrative |
| **Vocabulary** | Standard biomedical English | Clinical abbreviations (e.g., "BID", "prn", "h/o", "c/o") |
| **Table reuse** | Same 12 tables shared across all 926 texts | Each note has its own unique tables (no sharing) |
| **Texts per table** | ~926 texts → same 12 tables (~77:1 ratio) | 1 text → 2 unique tables (1:2 ratio) |
| **Total columns** | 416 (from 82 tables) | ~thousands (all unique per example) |
| **GT density** | Dense: ~8 positive tables per text out of 82 | Sparse: 2 positive tables per text out of thousands |
| **Train texts / tables** | 648 / 82 (shared) | ~5,000 unique notes / ~10,000 tables |
| **Val texts / tables** | 138 / 82 (shared) | ~500 notes / ~1,000 tables |
| **Test texts / tables** | 140 / 82 (shared) | 18 notes / 36 tables |

## Analysis: Why CMDL Fails on MIMIC

### 1. Shared vs. Unique Tables (Fundamental Architecture Mismatch)

CMDL is designed for the **data lake scenario**: a fixed collection of database tables queried by many documents. The Pharma dataset perfectly embodies this — DrugBank is a single relational database, and every PubMed paper maps to the same small set of tables (drug, targets, pharmacology, etc.).

MIMIC breaks this assumption entirely. Each clinical encounter produces its **own unique table** — a patient's specific diagnoses, medications, and lab values. There are no shared table patterns for the model to learn. In CMDL's embedding space:

- **Pharma**: The column `drugbank-drug.csv,name` receives consistent training signal from ~926 texts → robust, well-positioned embedding
- **MIMIC**: The column `patient_001.csv,diagnosis` receives signal from exactly 1 text → essentially noise, no generalization possible

Notably, despite training on a large corpus (~5,000 unique notes and ~10,000 tables), the model still fails on even a small test set of 18 documents. The sheer volume of training data cannot compensate for the lack of shared table structure.

In a real-world data lake, it is unrealistic to assume that all query documents will map to the same fixed set of tables. Datasets where each document has distinct table associations (like MIMIC) represent a more challenging and arguably more realistic retrieval scenario.

### 2. Text Quality and Domain Mismatch

**Pharma texts** are well-structured PubMed abstracts written in standard biomedical English, supplemented with curated MeSH keywords (e.g., "Cerebral Cortex", "Clomipramine", "Receptors, Serotonin"). These keywords provide strong conceptual overlap with DrugBank table content.

**MIMIC texts** are clinical discharge summaries filled with:
- **Abbreviations**: "BID" (twice daily), "prn" (as needed), "h/o" (history of), "c/o" (complaining of)
- **Telegraphic style**: Sentence fragments, lists, shorthand
- **Highly specialized jargon** not typically found in general biomedical literature

The fastText WEM features at the core of CMDL's pipeline are trained on general English text. Clinical abbreviations and shorthand are likely **out-of-vocabulary or poorly represented**, leading to degraded text embeddings for MIMIC documents.

### 3. Pre-trained Model Bias

CMDL's advanced pipeline (the tuple encoder branch in `pharma-text2tuple.json`) uses `allenai/biomed_roberta_base` — a transformer model **pre-trained specifically on PubMed and biomedical text**. This gives the Pharma pipeline a significant head start:

- The tokenizer and model already understand biomedical vocabulary (drug names, gene symbols, medical terminology)
- PubMed abstracts closely match the model's pre-training distribution
- DrugBank table content (drug descriptions, pharmacology) shares this vocabulary

For MIMIC, even if BiomedRoBERTa were used, clinical notes represent a **domain shift** from the biomedical literature it was trained on. The model would not natively understand clinical abbreviations, nursing shorthand, or the telegraphic documentation style common in EHR data.

### 4. Label Matrix Sparsity

The training signal quality differs dramatically:

- **Pharma**: 648 texts × 416 columns, with **42,820 positive pairs** (dense label matrix). Each text has ~66 positive columns on average. The model receives rich, reinforced training signal.
- **MIMIC**: ~5,000 texts each mapping to columns from 2 unique tables. With ~10,000 tables and thousands of columns, each text has only a handful of positive columns out of thousands. The label matrix is **extremely sparse** (<0.1% positive). Most mini-batches contain no positive pairs, yielding zero-loss steps that produce no learning.

## Summary

CMDL's strong Pharma results reflect a model operating within its designed assumptions: a fixed data lake of shared tables, clean biomedical text, and domain-matched pre-trained features. The poor MIMIC results expose fundamental limitations when these assumptions are violated — unique per-document tables, noisy clinical text, and domain mismatch. This suggests that more robust approaches are needed for cross-modal retrieval in settings where table structures are not shared across queries.



