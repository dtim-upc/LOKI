# HOW TO RUN — Experiment-2 (SOTA Evaluation on Data Discovery)

Unified evaluation framework for comparing **CMDL**, **LOKI**, **TabSTAR**, and **TaBERT** on the Pharma Protocol (Flipped) table-text discovery task.

---

## 1. Prerequisites

### Python environment

All four models share the same environment. Key packages:

```
torch >= 2.0
transformers
sentence-transformers
numpy, tqdm, scikit-learn
matplotlib          # for plots
openpyxl            # for Excel export
annoy               # required by CMDL
spacy + en_core_web_sm  # required by CMDL text profiler
gensim              # required by CMDL (FastText WEM)
```

### External code repositories

The following sibling directories must exist alongside `SOTA_Evaluation_New/`:

```
<project_root>/
├── CMDL/                    # CMDL source code (profiler, featurizer, WEM, converter)
├── TaBERT/                  # TaBERT source code (table_bert package)
├── TabSTAR/                 # TabSTAR source code (src/tabstar package)
└── SOTA_Evaluation_New/     # ← this directory
```

The LOKI model runtime is bundled in `SOTA_Evaluation_New/loki_runtime/`; no
external LOKI source directory is required.

### Test data

The default test file is configured in `config.py` via `TEST_DATA_FILE`.
For the pharma flipped structured dataset, point it at:

```
../Datasets/pharma_flipped_structured/test_row_level.json
```

Override with `--test_file` if your data is elsewhere.

---

## 2. Model Setup

All model weights live under `SOTA_Evaluation_New/models/`. Each subfolder
must be populated before running the corresponding evaluator.

### Automatic Model Download (Recommended)

All published evaluation models can be downloaded automatically from Hugging Face:
[https://huggingface.co/shaoncsecu/LOKI/tree/main/Exp-2/models](https://huggingface.co/shaoncsecu/LOKI/tree/main/Exp-2/models)

From inside `SOTA_Evaluation_New/`, run:

```bash
python download_models.py
```

This single command automatically downloads and arranges all 4 models (`CMDL`, `LOKI`, `TaBERT`, `TabSTAR`) into `models/` and ensures the required base sentence encoder (`sentence-transformers/embeddinggemma-300m-medical`) is cached in `models/hf_assets/`.

#### Useful options:

- **Download specific models only:**
  ```bash
  python download_models.py --models LOKI TabSTAR
  ```
- **Force re-download / overwrite:**
  ```bash
  python download_models.py --force
  ```
- **Custom destination directory:**
  ```bash
  python download_models.py --destination path/to/models
  ```

---

### Model Directory Structure

From the `SOTA_Evaluation_New/` directory, the complete model layout populated by the download script is:

```
models/
├── CMDL/
│   ├── text_enet_best.pt
│   ├── col_enet_best.pt
│   └── resources/fasttext/cc/
│       └── cc.en.300.gensim
├── LOKI/
│   ├── args.json
│   └── embeddinggemma-300m-medical_best.pt
├── TaBERT/
│   └── tabert_large_k3/
│       ├── model.bin
│       └── tb_config.json
└── TabSTAR/
    ├── tabstar_weights/
    │   ├── config.json
    │   └── model.safetensors
    └── e5_small_v2/
        ├── config.json
        └── model.safetensors
```

### CMDL

```
models/CMDL/
├── text_enet_best.pt                      # trained text EncoderNet
├── col_enet_best.pt                       # trained column EncoderNet
└── resources/fasttext/cc/
    ├── cc.en.300.gensim                   # FastText word-embedding model (gensim format)
    ├── cc.en.300.gensim.vectors_ngrams.npy
    └── cc.en.300.gensim.vectors_vocab.npy
```

Copy `text_enet_best.pt` and `col_enet_best.pt` from your CMDL training
output. The FastText model (`cc.en.300`) can be downloaded from
[Facebook fastText](https://fasttext.cc/docs/en/crawl-vectors.html) and
converted to gensim format.

### LOKI

```
models/LOKI/
├── args.json                              # training hyperparameters
└── embeddinggemma-300m-medical_best.pt    # default LOKI checkpoint
```

Only one checkpoint is required. The bundled `*_best.pt` checkpoint is
auto-discovered as `best_model` and is selected by default through
`config.py` → `LOKI_ACTIVE_MODEL`.

Additional checkpoints, such as `best_test_avg_precision_*/model.pt` or
`best_test_f1_*/model.pt`, are optional. Keep them only when explicitly
comparing checkpoint-selection criteria with `--loki_model best_test_ap` or
`--loki_model best_test_acc`; the standard comparison and scalability scripts
load one selected checkpoint, never all checkpoints together.

If the checkpoint was trained with header conditioning enabled,
run SOTA evaluation with:

```bash
--use_schema_aware_loki
```

This enables the bundled schema-aware scorer path. In schema-aware
checkpoints, schema conditioning means per-column schema sketches
(`Column {header}. Example values: ...`) are encoded separately from
value-only row text and used to steer table-side Q/K routing. By default,
the SOTA scripts auto-detect this from the checkpoint `args.json`, so the
flag is mainly useful to force the schema-aware path explicitly.

### TaBERT

```
models/TaBERT/
└── tabert_large_k3/
    ├── model.bin          # TaBERT-Large (K=3) pretrained weights
    └── tb_config.json
```

Download from the [TaBERT GitHub releases](https://github.com/facebookresearch/TaBERT)
or copy from your local TaBERT checkout.

**Using a fine-tuned model:** If you have a fine-tuned TaBERT checkpoint,
replace `model.bin` with your fine-tuned weights (keeping the same filename
`model.bin`) and leave all other files (`tb_config.json`, etc.) unchanged.

### TabSTAR

```
models/TabSTAR/
├── tabstar_weights/       # full TabSTAR model (alana89/TabSTAR)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer*.json, vocab.txt
└── e5_small_v2/           # E5-small-v2 sentence encoder (intfloat/e5-small-v2)
    ├── config.json
    ├── model.safetensors
    └── tokenizer*.json, vocab.txt
```

**Automatic download:** TabSTAR weights are automatically fetched by `python download_models.py` (or specifically `python download_models.py --models TabSTAR`). Alternatively, you can use the standalone script `python download_tabstar_weights.py`.

---

## 3. Running Evaluations

All commands are run from the `SOTA_Evaluation_New/` directory.

### Run all 4 models (default)

```bash
python run_comparison_pharma.py
```

### Run all 4 models on row-level fragments

By default, evaluations use individual row-level fragments (~2240). The
following explicit flag is equivalent to the default:

```bash
python run_comparison_pharma.py --no-combined_tables
```

### Run a single model

Skip the other three with `--skip_*` flags:

```bash
# LOKI only
python run_comparison_pharma.py --skip_cmdl --skip_tabstar --skip_tabert

# LOKI only, explicitly force the schema-aware scorer path
python run_comparison_pharma.py --skip_cmdl --skip_tabstar --skip_tabert --use_schema_aware_loki

# CMDL only
python run_comparison_pharma.py --skip_loki --skip_tabstar --skip_tabert

# TabSTAR only
python run_comparison_pharma.py --skip_cmdl --skip_loki --skip_tabert

# TaBERT only
python run_comparison_pharma.py --skip_cmdl --skip_loki --skip_tabstar
```

### Run a subset of models

Combine skip flags as needed:

```bash
# LOKI + CMDL only (skip the SOTA baselines)
python run_comparison_pharma.py --skip_tabstar --skip_tabert

# All three neural models, skip CMDL
python run_comparison_pharma.py --skip_cmdl
```

### Quick sanity check (small subsample)

```bash
python run_comparison_pharma.py --max_test_examples 10
python run_comparison_pharma.py --max_test_examples 50 --no-combined_tables
```

### Standalone model scripts

Each evaluator can also be run independently:

```bash
python evaluate_loki.py
python evaluate_cmdl.py
python evaluate_tabstar.py
python evaluate_tabert.py
```

To force structured LOKI scoring explicitly:

```bash
python evaluate_loki.py --use_schema_aware_loki
```

This path is now auto-selected for checkpoints whose `args.json` enables either
`use_header_conditioning` or `use_cell_level_matching`.

They use the defaults from `config.py` and accept the same `--test_file`,
`--max_test_examples`, `--seed`, and `--output_dir` flags. Schema-aware
LOKI outputs are also tagged with a schema-representation version, so old
pooled-schema caches/results are not reused accidentally.

---

## 4. Interactive Result Reuse

When previous result files are detected, the script prompts:

```
Choose what to do:
  1) Re-run ALL models (fresh evaluation)
  2) Re-run CMDL only
  3) Re-run LOKI only
  4) Re-run TabSTAR only
  5) Re-run TaBERT only
  6) Skip running, just generate comparison + plots
```

Option **6** loads all existing results from disk and regenerates the
comparison tables, plots, and Excel export without re-running any model.

You can also bypass the prompt by providing pre-computed result files:

```bash
python run_comparison_pharma.py \
  --cmdl_results   results/CMDL_pharma_combined_results.json \
  --loki_results   results/LOKI_pharma_combined_results.json \
  --tabstar_results results/TabSTAR_pharma_combined_results.json \
  --tabert_results  results/TaBERT_pharma_combined_results.json
```

---

## 5. Configuration

Edit `config.py` to change defaults without CLI flags:

| Setting | Location | Purpose |
|---------|----------|---------|
| `MAX_TEST_EXAMPLES` | `config.py` | Default subsample size (0 = full test set) |
| `SEED` | `config.py` | Random seed for deterministic subsampling |
| `K_VALUES` | `config.py` | List of K values for @K metrics (default: [1, 2, 4, 8, 16, 32]) |
| `SCALABILITY_SIZES` | `config.py` | Candidate pool sizes for `run_scalability_pharma.py` (0 = full pool) |
| `LOKI_ACTIVE_MODEL` | `config.py` | Which LOKI checkpoint to use by default |
| `LOKI_AGGREGATION_METHOD` | `config.py` | Score aggregation (default: `top_k_pairs`) |
| `OUTPUT_DIR` | `config.py` | Default output directory |

---

## 6. CLI Reference — `run_comparison_pharma.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--test_file` | (from `config.py`) | Path to test JSON |
| `--max_test_examples` | `0` (all) | Subsample size, shared across all models |
| `--seed` | `42` | Random seed for subsampling |
| `--output_dir` | `results/` | Output directory |
| `--combined_tables` / `--no-combined_tables` | off | Row-level fragments are the supported evaluation mode. `--no-combined_tables` is explicit but equivalent to the default; combined-table evaluation is not implemented by the unified evaluators. |
| `--skip_cmdl` | off | Skip CMDL evaluation |
| `--skip_loki` | off | Skip LOKI evaluation |
| `--skip_tabstar` | off | Skip TabSTAR evaluation |
| `--skip_tabert` | off | Skip TaBERT evaluation |
| `--loki_model` | `best_model` | LOKI checkpoint key. `best_test_ap` and `best_test_acc` are available only when their optional checkpoints are present. |
| `--use_schema_aware_loki` | auto-detect | Use the bundled schema-aware LOKI scorer path. For structured checkpoints this means per-column schema sketches are encoded separately from value-only rows. By default the evaluator auto-detects this from the checkpoint `args.json`. |
| `--encode_batch_size` | `64` | Batch size for LOKI sentence encoding |
| `--eval_row_chunk_size` | `0` | Row chunking for LOKI micro pass (0 = no chunking) |
| `--cache_table_embeddings` | on | Pre-encode tables on GPU (use `--no-cache_table_embeddings` to disable) |
| `--cache_doc_embeddings` | off | Pre-encode documents on GPU (use `--cache_doc_embeddings` to enable) |
| `--bf16` / `--no-bf16` | on | BFloat16 inference for TaBERT |
| `--torch_compile` / `--no-torch_compile` | on | torch.compile() for TaBERT |
| `--cmdl_results` | — | Load pre-computed CMDL results JSON |
| `--loki_results` | — | Load pre-computed LOKI results JSON |
| `--tabstar_results` | — | Load pre-computed TabSTAR results JSON |
| `--tabert_results` | — | Load pre-computed TaBERT results JSON |
| `--is_flipped` | `True` | *(Deprecated — use `--task` direction instead.)* Flipped dataset mode (queries = docs, corpus = table rows) |

---

## 7. Output Artifacts

All outputs are saved to `results/` (or `--output_dir`):

| File | Description |
|------|-------------|
| `CMDL_pharma_results.json` | CMDL macro + micro metrics |
| `LOKI_pharma_results.json` | LOKI macro + micro metrics |
| `TabSTAR_pharma_results.json` | TabSTAR macro + micro metrics |
| `TaBERT_pharma_results.json` | TaBERT macro + micro metrics |
| `combined_pharma_results.json` | All models side-by-side |
| `pharma_macro_plot.png` | Macro comparison: P/R/F1/NDCG/MRR @K, AP bars, Mean Rank, PR curve |
| `pharma_micro_plot.png` | Micro comparison: P/R/F1 @K |
| `Result_Ranking.xlsx` | Per-K ranking metrics for all models |
| `Result_Summary.xlsx` | Condensed paper-style result table |

Results are written for row-level evaluation without a `_combined` suffix.

---

## 8. Script Overview

| Script | Purpose |
|--------|---------|
| `run_comparison_pharma.py` | **Main entry point.** Runs all models, prints comparisons, generates plots, exports Excel. |
| `run_scalability_pharma.py` | **Search-space scalability study:** score each model once on the full corpus, then re-rank under restricted candidate pools; writes JSON caches, Excel, and classic scalability plots. |
| `plot_scalability.py` | Regenerates **classic** scalability figures from existing `*_pharma_scalability.json` only (no model re-run). |
| `plot_scalability_vldb.py` | **Extra publication plots:** per-metric panels (facets, heatmaps, paper_v1–v3 bands, radar, global MAP/AP/rank). |
| `plot_scalability_composite_row.py` | **Single-row composite** macro and micro figures (PNG + PDF) for papers. |
| `evaluate_cmdl.py` | CMDL evaluation engine (standalone or called by comparison script). |
| `evaluate_loki.py` | LOKI evaluation engine with Macro + Micro (standalone or called). |
| `evaluate_tabstar.py` | TabSTAR evaluation engine (standalone or called). |
| `evaluate_tabert.py` | TaBERT evaluation engine (standalone or called). Uses structured `headers`+`content` via `extract_structured_tables()` when available; falls back to legacy string parsing for other datasets. |
| `config.py` | Central configuration (model paths, K values, output dir). |
| `metrics.py` | Shared ranking metrics (P@K, R@K, F1@K, NDCG@K, MRR@K, MAP, Score AP, Mean Rank). |
| `export_excel.py` | Exports results to Excel workbooks. |
| `download_tabstar_weights.py` | Downloads TabSTAR + E5-small-v2 weights from HuggingFace. |

---

## 9. Scalability study and plotting (recommended order)

Use this when you want figures that show how retrieval changes as the **per-query candidate pool** grows (e.g. 50 → Full), using the Pharma flipped protocol. All commands assume the working directory is `SOTA_Evaluation_New/`.

### Step 1 — Produce metrics and baseline plots

Run the scalability driver. It scores each selected model **once** on the full table pool, restricts candidates per pool size (always keeping ground truth + hard negatives), recomputes metrics, then calls `plot_scalability.py` internally and exports Excel.

```bash
python run_scalability_pharma.py
```

Useful flags (see the script docstring for the full list):

| Flag | Role |
|------|------|
| `--output_dir` | Where to write JSON caches, plots, and Excel (default: under `results/scalability/`). |
| `--sizes` | Candidate pool sizes; `0` means full pool (`config.SCALABILITY_SIZES` by default). |
| `--focal_k` | K for the main 2×3 scalability grid in `plot_scalability.py` (default: 8). |
| `--skip_cmdl` / `--skip_loki` / … | Same model toggles as other drivers. |
| `--no-combined_tables` | Explicitly use the default row-level fragment evaluation. |
| `--force_rerun` | Ignore cached `*_full_scores.json` and re-score. |

**Artifacts you should see** under your `--output_dir` (example names without `_combined`):

| Artifact | Description |
|----------|-------------|
| `{MODEL}_pharma_full_scores.json` | Cached full similarity structure for reuse. |
| `{MODEL}_pharma_scalability.json` | Per-pool macro + micro metrics (input to all plotting scripts below). |
| `scalability_macro_K{focal_k}.png`, `scalability_micro_K{focal_k}.png` | Classic multi-metric vs pool size. |
| `scalability_macro_per_K.png` | Macro metrics with **subplots per K**. |
| `Scalability_Results.xlsx` | Tabular export (**macro** columns for P/R/F1/NDCG/MRR/All@K, MAP, AP, mean rank). |

### Step 2 — (Optional) Regenerate classic plots only

If the `*_pharma_scalability.json` files already exist and you only want to refresh the **original** `plot_scalability` figures (e.g. after changing `--focal_k` or editing `config.K_VALUES` / pool list alignment):

```bash
python plot_scalability.py --results_dir path/to/scalability --focal_k 8
```

`--results_dir` must contain `CMDL_pharma_scalability.json`, `LOKI_pharma_scalability.json`, etc. Pool sizes on the x-axis come from **`config.SCALABILITY_SIZES`**; they must match the keys present in the JSON.

### Step 3 — (Optional) VLDB-style / supplementary plots

`plot_scalability_vldb.py` reads the same `*_pharma_scalability.json` files and writes under `<results_dir>/vldb_plots/` (or `--output_dir`), grouped in `macro/` and `micro/` subfolders.

```bash
# Paper-style bands only (skip the large MAP/AP/rank figure)
python plot_scalability_vldb.py --results_dir path/to/scalability ^
  --styles paper_v1 paper_v2 paper_v3 --metric_type macro --skip_global
```

On Linux/macOS, replace `^` with `\` for line continuation.

| Flag | Role |
|------|------|
| `--styles` | Choose any of: `facet_k`, `facet_pool`, `heatmap`, `radar`, `paper_v1`, `paper_v2`, `paper_v3`. |
| `--metric_type macro` / `micro` | Which block to plot (can pass both). |
| `--focal_k` | Solid line K for `paper_v1`. |
| `--band_ks` | e.g. `4,8,16` — K set for the min–max band in `paper_v1` (empty = all K in JSON). |
| `--skip_global` | Omit the 1×3 MAP / Score AP / Mean Rank figure. |
| `--suffix` | Filename suffix (e.g. `_combined`). |

### Step 4 — (Optional) One-row composite figures (PNG + PDF)

For a **single horizontal strip** (macro and micro) with P/R/F1/NDCG/MRR bands, AP, and All@K:

```bash
python plot_scalability_composite_row.py --results_dir path/to/scalability
```

Defaults write to `path/to/scalability/composite_row/`:

- `scalability_composite_macro_row.png` / `.pdf`
- `scalability_composite_micro_row.png` / `.pdf`

| Flag | Role |
|------|------|
| `--output_dir` | Override composite output directory. |
| `--band_ks` | K values for the min–max band (default `4,8,16`; empty = all K in JSON). |
| `--focal_k` | K for the solid line inside the band (default 8). |
| `--all_k_hit` | K for the All@K panel (default 32). |
| `--figure_title` / `--figure_title_macro` / `--figure_title_micro` | Custom suptitles. |
| `--x_label_candidates` | Shared x-axis label (default `Candidate pool size`). |
| `--marker_size`, `--loki_star_scale` | Marker sizing (LOKI uses a star). |

**Note:** On this Pharma split, **macro and micro P/R/F1@K match** when every query has the same \|GT\|; MAP/AP/mean rank still differ. The composite **micro** row includes placeholder panels for NDCG/MRR (not defined in micro JSON) and uses **macro** `All@K` for the hit-rate column.

### Summary flow

1. `run_scalability_pharma.py` → JSON + Excel + **classic** plots.  
2. `plot_scalability.py` → redo **classic** plots only if needed.  
3. `plot_scalability_vldb.py` → extra per-metric / paper variants.  
4. `plot_scalability_composite_row.py` → **macro + micro** composite PNG/PDF for the paper.

Steps 2–4 never re-run GPU inference as long as the `*_pharma_scalability.json` files are present.
