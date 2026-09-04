# LOKI Pipeline — Run Instructions

## TL;DR — Run the pipeline with best defaults

The current best validated configuration is baked into the argparse defaults: LMStudio with HDBSCAN-backed grouping, `contextual_sentence_average` pair embeddings, CE rerank + combined CE pair filter, semantic pair refinement, and gated same-pair path splitting. **No flags are required.**

> **Requirement:** LM Studio must be running at `http://192.168.1.128:1234` with model `openai/gpt-oss-20b` loaded, unless you override the server with `--llm_base_url`.

**Single admission (one click):**

```powershell
Set-Location 'LOKI'
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363
```
→ All artifacts written to `Batch_Materialization\loki_run_20393363\` (JSON, CSV, audit MD, all visualization PNGs plus matching PDF copies).

**Whole-dataset batch (one click):**

```powershell
Set-Location 'LOKI'
python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1
```
→ All artifacts written to `Batch_Materialization\loki_batch_mimic\` (batch CSVs, summary, report MD, dashboard PNGs plus matching PDF copies). Embedding projection comparison plots are now **off by default** and only appear when `--batch_projection` is enabled.

If LM Studio connectivity drops during batch mode, the run now retries each LM Studio request up to 5 times, then stops cleanly without silently switching to keyword labeling. Completed admissions remain saved in the batch CSV/report.

**Batch subset (first N admissions):**

```powershell
Set-Location 'LOKI'
python materialize_joins.py --dataset mimic --run_all_admissions --max_admissions 20 --batch_progress_every 1
```
→ Useful for smoke tests, development runs, and partial reruns when you do not want to process the full dataset.

**Resume an interrupted batch after fixing LM Studio:**

```powershell
Set-Location 'LOKI'
python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1 --resume
```
→ Reloads the existing `materialized_batch_results_mimic.csv`, skips admissions that already completed, and continues from the first unfinished admission.

**Resume with a different LM Studio retry budget:**

```powershell
Set-Location 'LOKI'
python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1 --resume --llm_retry_attempts 8
```
→ Useful if the server is reachable but occasionally slow or briefly unstable.

**Optional: turn on batch embedding projection diagnostics:**

```powershell
Set-Location 'LOKI'
python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1 --batch_projection
```
→ Adds the optional pair-embedding projection comparison PNGs for exploratory analysis. These are diagnostic figures, not the main evaluation dashboards.

**Regenerate the CSV-backed batch dashboards only (no inference):**

```powershell
Set-Location 'LOKI'
python materialize_joins.py --regenerate_batch_diagrams_from_results_csv ..\Batch_Materialization\loki_batch_mimic\materialized_batch_results_mimic.csv
```
→ Rebuilds `materialized_batch_metrics_mimic.png` and `materialized_batch_representation_dashboard_mimic.png` in the same folder as the supplied `materialized_batch_results_*.csv`, without re-running model inference.

Both commands resolve to this fully-explicit invocation (every flag matches the current default):

```powershell
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 `
  --cluster_label_backend lmstudio `
  --llm_model openai/gpt-oss-20b `
  --llm_hdbscan `
  --llm_no_agglomerative `
  --pair_embedding_mode contextual_sentence_average `
  --hdbscan_min_cluster_size 4 `
  --no_shared_pair_merge `
  --llm_max_evidence_sents 3 `
  --cluster_refine_min_pairs 5 `
  --cluster_refine_semantic_subsplit --cluster_refine_semantic_distance 0.20 `
  --cluster_refine_llm_per_path_vote `
  --cluster_refine_path_subsplit `
  --cluster_refine_path_subsplit_min_mass 0.25 `
  --cluster_refine_path_subsplit_min_share 0.30 `
  --cluster_refine_path_subsplit_max_gap 0.12 `
  --suppress_negative_clusters `
  --use_cross_encoder `
  --ce_pair_filter_mode combined --ce_pair_filter_quantile 0.25 `
  --show_typed_metrics
```

---

This document covers how to run the main materialization pipeline in this project:

- **`LOKI/materialize_joins.py`** — the LOKI pipeline itself (path extraction, cluster labeling, evaluation, and direct model/profile comparisons)

---

## Prerequisites

**Always activate the THOR conda environment and navigate to the LOKI folder first:**

```powershell
Set-Location 'LOKI'
```

All commands below assume your working directory is `LOKI`.

Use the full Python path to ensure the correct environment:
```
c:/Users/SHAON/anaconda3/envs/THOR/python.exe
```

Or activate the environment first:
```powershell
conda activate THOR
python materialize_joins.py ...
```

---

## Part 1 — `materialize_joins.py`

This is the main LOKI pipeline. It takes a hospital admission's diagnosis table, medication table, and clinical notes, then extracts join paths and assigns relationship labels (TREATS, ADVERSE_EFFECT, DISCONTINUED, CONTRAINDICATED, NEGATIVE when the annotation inventory includes it).

There are two modes:
- **Single-admission** — run on one specific patient admission; saves visualization PNGs plus same-basename PDF copies and a detailed audit report under `Batch_Materialization/loki_run_<id>/`
- **Batch** — run on all admissions in a dataset; saves aggregate CSV metrics and comparison charts under `Batch_Materialization/loki_batch_<dataset>/`

---

### 1.1 — Quick Start (one-click defaults)

See the **TL;DR** at the top of this document for the two one-click commands. The defaults bake in the current validated best-quality pipeline (Phase D + D.5 + D.6 + E + F):

| Stage | Default | Purpose |
|---|---|---|
| Cluster labeler | `--cluster_label_backend lmstudio` | Local LLM labeling via LM Studio |
| Local model | `--llm_model openai/gpt-oss-20b` | LMStudio model name |
| Grouping mode | `--llm_hdbscan` (default) | Preserve HDBSCAN structure instead of pair-identity grouping |
| LLM label mode | `--llm_no_agglomerative` (default) | Faster per-cluster closed-label LMStudio labeling |
| Pair embedding mode | `--pair_embedding_mode contextual_sentence_average` | Best validated batch profile with stronger recovery and clustering quality |
| HDBSCAN min size | `--hdbscan_min_cluster_size 4` | Stable fine-grained pair clusters |
| Shared-pair merge | `--no_shared_pair_merge` (default) | Avoid merging semantically different relation clusters through shared pairs |
| Pair-label semantic split | `--cluster_refine_semantic_subsplit` (on) | Split same-label pair groups semantically before rebuilding clusters |
| Pair-label per-path vote | `--cluster_refine_llm_per_path_vote` (on) | Use LMStudio per-path evidence votes when refining pairs |
| Gated path split | `--cluster_refine_path_subsplit` (on) | Split mixed same-pair evidence only when support is strong enough |
| NEGATIVE suppression | `--suppress_negative_clusters` (on) | Drop non-annotated NEGATIVE clusters after refinement |
| Evidence sentences | `--llm_max_evidence_sents 3` | Top-3 CE-ranked sentences per cluster/pair |
| CE rerank | `--use_cross_encoder` (on) | Phase D.5 — cross-encoder per-pair sentence rerank |
| Option D filter | `--ce_pair_filter_mode combined` @ `0.25` | Phase D.6 — drop only pairs weak by BOTH CE *and* LOKI signals |

**Latest validated results on admission 20393363** (current defaults):

| Metric | Value |
|---|---|
| Exact triple F1 | **0.210** |
| Relaxed pair F1 | **0.400** |
| Automatic typed pair F1 | **0.356** |
| Oracle pair F1 | **0.571** |
| Multi-rel recall | **10 / 13** |
| Cluster purity / ARI | **0.935 / 0.803** |

**Common opt-outs:**

```powershell
# Disable Option D (Phase D.6) only — keeps CE rerank
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --ce_pair_filter_mode off

# Disable cross-encoder rerank (Phase D.5) — also implicitly disables Option D
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --no_cross_encoder

# Use GLiNER2 instead of LMStudio (no LM Studio needed, faster)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2

# Revert to historical pair-identity LMStudio grouping
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --llm_no_hdbscan

# Disable same-pair path splitting inside pair-label refinement
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --no_cluster_refine_path_subsplit

# Re-enable shared-pair merge (off by default in the current profile)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --shared_pair_merge
```

**Optional batch-only diagnostics to play with:**

| Flag | Default | What it changes |
|---|---|---|
| `--batch_projection` | off | Enables the optional batch embedding projection figures (2D, 3D, semantic). |
| `--batch_projection_points_per_type` | `300` | Caps how many sampled GT-backed predicted pairs per relation type are plotted. Lower it for lighter, cleaner exploratory figures. |
| `--batch_projection_max_points` | `1800` | Global cap on sampled plotted points across relation types. Lower it when the batch corpus is large or the figure looks too dense. |

Recommended practical settings:

```powershell
# Conservative exploratory view
python materialize_joins.py --dataset mimic --run_all_admissions --batch_projection --batch_projection_points_per_type 100 --batch_projection_max_points 600

# Richer diagnostic view
python materialize_joins.py --dataset mimic --run_all_admissions --batch_projection --batch_projection_points_per_type 250 --batch_projection_max_points 1500
```

There is currently no separate CLI switch for “2D only” versus “3D only”; enabling `--batch_projection` generates the whole projection bundle.

---

### 1.1.1 — Legacy Quick Start (pre-Option D)

---

### 1.2 — Single Admission: Dataset & Admission Selection

| Flag | Values | Description |
|------|--------|-------------|
| `--dataset` | `mimic` (default), `mimic_small` | Which dataset split to use. `mimic_small` is faster for testing. |
| `--single_admission` | flag | Run one admission instead of the default full-dataset batch mode |
| `--admission_id` | e.g. `20393363` | The specific admission to run. Omit to use the built-in default. |

```powershell
# Run a specific admission from the full mimic dataset
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363

# Run a specific admission from the smaller development dataset
python materialize_joins.py --single_admission --dataset mimic_small --admission_id 20393363

# Run a different admission
python materialize_joins.py --single_admission --dataset mimic --admission_id 20301031
```

---

### 1.3 — Single Admission: Cluster Labeling Backends

The cluster labeling backend decides how each discovered relationship cluster is assigned a type label.

| Backend | Flag | What it does |
|---------|------|-------------|
| `gliner2` | `--cluster_label_backend gliner2` | Faster non-LLM alternative using a small NER model to classify clusters from clinical text evidence. |
| `keyword` | `--cluster_label_backend keyword` | Fast lexical fallback. Scans evidence text for keywords associated with each relation type. No model needed. |
| `lmstudio` | `--cluster_label_backend lmstudio` | **Default.** Uses a local LLM running in LM Studio for the current best validated profile. Requires LM Studio to be running. |
| `oracle` | `--cluster_label_backend oracle` | **Research only.** Assigns GT labels directly (ground-truth majority vote). Shows the theoretical maximum performance achievable with perfect labeling. |

---

#### 1.3.1 — GLiNER2 Backend

GLiNER2 has two independent axes to configure:

**Axis A: `--pair_embedding_mode`** — how pairs are embedded for HDBSCAN clustering

| Mode | Flag | Description |
|------|------|-------------|
| `contextual_sentence_average` | `--pair_embedding_mode contextual_sentence_average` | Score-weighted average of refined sentence embeddings. Current global default. |
| `signature` | `--pair_embedding_mode signature` | Encode the full evidence text of the pair as a string. |
| `semantic_signature` | `--pair_embedding_mode semantic_signature` | Encode TF-IDF top terms extracted from the evidence. |
| `row_pair_hybrid` | `--pair_embedding_mode row_pair_hybrid` | Concatenation of `[diag_embedding ‖ avg_sent_embedding ‖ med_embedding]`. Useful ablation when you want stronger row-identity structure. |

**Axis B: `--gliner2_label_input_mode`** — what text is fed to GLiNER2 for labeling

| Mode | Flag | Description |
|------|------|-------------|
| `sentence_evidence` | (default) | Feed raw clinical sentences from the cluster as evidence text. |
| `semantic_signature` | `--gliner2_label_input_mode semantic_signature` | Feed TF-IDF semantic keywords extracted from the cluster's pair signatures. |

> These two axes are **independent** — any combination of `pair_embedding_mode` × `gliner2_label_input_mode` is valid. The full 4×2 = 8-combo grid is covered by the ablation study.

**Full 8-combo single-admission GLiNER2 grid:**

```powershell
# 1. signature + sentence_evidence
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode signature --gliner2_label_input_mode sentence_evidence

# 2. signature + semantic_signature
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode signature --gliner2_label_input_mode semantic_signature

# 3. contextual_sentence_average + sentence_evidence  (historical GLiNER2 baseline)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode contextual_sentence_average --gliner2_label_input_mode sentence_evidence

# 4. contextual_sentence_average + semantic_signature
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode contextual_sentence_average --gliner2_label_input_mode semantic_signature

# 5. row_pair_hybrid + sentence_evidence
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode row_pair_hybrid --gliner2_label_input_mode sentence_evidence

# 6. row_pair_hybrid + semantic_signature
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode row_pair_hybrid --gliner2_label_input_mode semantic_signature

# 7. semantic_signature + sentence_evidence
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode semantic_signature --gliner2_label_input_mode sentence_evidence

# 8. semantic_signature + semantic_signature
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode semantic_signature --gliner2_label_input_mode semantic_signature
```

**Extra GLiNER2 options:**

```powershell
# Enable per-sentence voting (reduces hub-sentence noise; helps ADVERSE_EFFECT precision)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend gliner2 \
  --pair_embedding_mode contextual_sentence_average \
  --gliner2_label_input_mode sentence_evidence \
  --gliner2_per_sentence_vote

# Show detailed per-label typed metrics in the console output
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend gliner2 --show_typed_metrics
```

---

#### 1.3.2 — Keyword Backend (fast baseline)

No model required. Uses keyword matching only:

```powershell
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend keyword
```

---

#### 1.3.3 — LMStudio Backend (local LLM)

> **Requirement:** LM Studio must be running with a model loaded before you run these commands.
> Default server address: `http://192.168.1.128:1234`

**Available LLM options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--llm_model` | `openai/gpt-oss-20b` | Model name as it appears in LM Studio |
| `--llm_base_url` | `http://192.168.1.128:1234` | LM Studio server address |
| `--llm_temperature` | `0.0` | Sampling temperature. `0.0` = fully deterministic. |
| `--llm_timeout` | `90` | Seconds to wait per API call before timeout. |
| `--llm_retry_attempts` | `5` | Retries LM Studio requests this many times before giving up. In batch mode, exhaustion aborts the current admission and the batch exits cleanly; otherwise the pipeline falls back to keyword labeling. |
| `--llm_max_evidence_sents` | `3` | How many top evidence sentences to include in the prompt per cluster. Now CE-aware — sentences are ranked by `ce_score` first when Phase D.5 is enabled. |
| `--llm_no_agglomerative` | **(default: on)** | Use fast per-cluster closed-label mode instead of agglomerative re-labeling. |
| `--llm_agglomerative` | off | Opt in to agglomerative re-labeling instead of the default per-cluster mode. |
| `--llm_agglom_distance` | `0.25` | Cosine distance threshold for the agglomerative grouping phase. Used only when `--llm_agglomerative` is enabled. |
| `--llm_path_vote` | flag | Per-path vote mode (HDBSCAN-backed): one closed-label LLM call per path, results aggregated back to HDBSCAN clusters via path_score-weighted vote. Bypasses the 4-phase agglomerative pipeline while preserving HDBSCAN structural grouping. |
| `--llm_hdbscan` | **(default: on)** | Preserve HDBSCAN-backed cluster grouping for LMStudio labeling. |
| `--llm_no_hdbscan` | off | Historical pair-identity grouping alternative that bypasses HDBSCAN entirely. Generates `llm_vs_hdbscan_<id>.png`. |
| `--llm_agglom_encoder` | `bge` | Encoder for agglom Phase 2 free-form phrase embeddings. `bge` = BAAI/bge-large-en-v1.5; `medembed` = loaded LOKI model; `minilm` = sentence-transformers/all-MiniLM-L6-v2. Used only with `--llm_agglomerative`. |
| `--pair_embedding_mode` | `contextual_sentence_average` | Current default pair representation for HDBSCAN clustering. |
| `--hdbscan_min_cluster_size` | `4` | Current default HDBSCAN minimum cluster size. Pass `0` to restore auto-calibration. |
| `--no_shared_pair_merge` | **(default: on)** | Disable shared-pair cluster merging in the current default profile. |
| `--shared_pair_merge` | off | Re-enable shared-pair must-link merging. |
| `--cluster_refine_min_pairs` | `5` | Only refine raw clusters with at least this many unique diagnosis-medication pairs. |
| `--cluster_refine_semantic_subsplit` | **(default: on)** | Semantically subcluster same-label pair buckets during refinement. |
| `--cluster_refine_llm_per_path_vote` | **(default: on)** | Use LMStudio per-path voting when refining pair labels. |
| `--cluster_refine_path_subsplit` | **(default: on)** | Gated same-pair path split for mixed-evidence pairs. |
| `--cluster_refine_path_subsplit_min_mass` | `0.25` | Minimum vote mass for a path-split child to survive. |
| `--cluster_refine_path_subsplit_min_share` | `0.30` | Minimum within-pair vote share for a path-split child to survive. |
| `--cluster_refine_path_subsplit_max_gap` | `0.12` | Maximum dominant-minus-child vote-mass gap allowed for a path-split child. |
| `--suppress_negative_clusters` | **(default: on)** | Suppress non-annotated NEGATIVE clusters after refinement. |
| `--use_cross_encoder` | **(default: on)** | Phase D.5 cross-encoder per-pair sentence rerank using `Alibaba-NLP/gte-reranker-modernbert-base`. Writes `ce_score` onto every surviving path and makes downstream signature/cluster-prompt builders CE-aware. |
| `--no_cross_encoder` | flag | Opt-out for `--use_cross_encoder`. Disables Phase D.5 (also implicitly disables Option D). |
| `--ce_pair_filter_mode` | `combined` | Phase D.6 Option D pair-level filter. `combined` (default) drops a (diag, med) pair only when BOTH its max LOKI score AND its max CE score are in the bottom quantile. `absolute` drops pairs whose max CE score is below `--ce_pair_filter_threshold`. `quantile` uses CE quantile only. `off` disables. |
| `--ce_pair_filter_quantile` | `0.25` | Bottom-quantile cutoff used by `combined` and `quantile` modes. |
| `--ce_pair_filter_threshold` | `0.05` | Absolute CE-score cutoff used by `absolute` mode. |
| `--threshold` | auto (μ+σ) | Fixed γ threshold for path extraction. Lower values (e.g. `0.27`) improve recall at the cost of precision. **Do not override** — adaptive threshold is well-calibrated; lowering to 0.25 collapses precision (295 pred vs 22 GT). |
| `--llm_per_path_vote` | flag | One LLM call per path, then aggregate by path_score-weighted vote. Slower but more fine-grained. |

**Examples with Qwen3:**

```powershell
# Per-cluster mode (fastest LLM option, no agglomerative grouping)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_no_agglomerative

# Agglomerative mode, distance = 0.25 (most fine-grained grouping)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_agglom_distance 0.25

# Agglomerative mode, distance = 0.25 (alternative manual mode)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_agglom_distance 0.25

# Agglomerative mode, distance = 0.50 (more coarse grouping)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_agglom_distance 0.50

# Per-path vote (slower but more fine-grained per-cluster context)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_agglom_distance 0.25 \
  --llm_per_path_vote

# Per-path vote mode (HDBSCAN-backed; bypasses agglom, aggregates per HDBSCAN cluster)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_path_vote

# No-HDBSCAN pair mode (each unique (diag,med) pair = its own cluster; generates llm_vs_hdbscan_*.png)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_no_hdbscan

# Agglom with bge-large encoder in Phase 2 (better short-phrase grouping than MedEmbed)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_agglom_distance 0.25 \
  --llm_agglom_encoder bge

# Agglom with MiniLM encoder in Phase 2 (fastest STS encoder option)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_agglom_distance 0.25 \
  --llm_agglom_encoder minilm
```

**Examples with GPT-OSS-20B:**

```powershell
# GPT-OSS per-cluster (no agglomerative)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model openai/gpt-oss-20b \
  --llm_no_agglomerative

# GPT-OSS agglomerative d=0.25  (alternative manual mode)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model openai/gpt-oss-20b \
  --llm_agglom_distance 0.25 \
  --llm_timeout 90

# GPT-OSS agglomerative d=0.25 + lower gamma for improved recall
# (use when adaptive threshold is near a cliff edge, e.g. γ≈0.30)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model openai/gpt-oss-20b \
  --llm_agglom_distance 0.25 \
  --llm_timeout 90 \
  --threshold 0.27

# GPT-OSS agglomerative d=0.25 (most fine-grained)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model openai/gpt-oss-20b \
  --llm_agglom_distance 0.25 \
  --llm_timeout 90

# GPT-OSS per-path vote mode (HDBSCAN-backed; bypasses agglom)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model openai/gpt-oss-20b \
  --llm_path_vote \
  --llm_timeout 90

# GPT-OSS no-HDBSCAN pair mode (pair-identity grouping; generates llm_vs_hdbscan_*.png)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model openai/gpt-oss-20b \
  --llm_no_hdbscan \
  --llm_timeout 90

# GPT-OSS agglom with bge encoder for Phase 2 (tighter short-phrase grouping)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model openai/gpt-oss-20b \
  --llm_agglom_distance 0.25 \
  --llm_agglom_encoder bge \
  --llm_timeout 90

# GPT-OSS with custom server URL (if running on a different port)
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend lmstudio \
  --llm_model openai/gpt-oss-20b \
  --llm_base_url http://127.0.0.1:1234 \
  --llm_agglom_distance 0.25 \
  --llm_timeout 90
```

---

#### 1.3.4 — Oracle Backend (research upper bound)

Assigns ground-truth labels directly to clusters — shows the ceiling achievable if labeling were perfect:

```powershell
python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend oracle --show_typed_metrics
```

---

### 1.4 — Batch Mode (all admissions)

Run the pipeline over every annotated admission in the dataset. Use `--max_admissions N` when you want to cap the batch to the first `N` eligible admissions. Results are saved under `Batch_Materialization/loki_batch_<dataset>/`.

When `--cluster_label_backend lmstudio` is active in batch mode, LM Studio transport failures are fail-closed: the run retries up to `--llm_retry_attempts` times per request, then stops the batch at the current unfinished admission, preserves the partial CSV/report/failure log, and prints a `--resume` command.

```powershell
# Batch with current best defaults (lmstudio + HDBSCAN-backed refinement profile; requires LM Studio running)
python materialize_joins.py --dataset mimic --run_all_admissions

# Batch on the smaller development set (faster, for testing)
python materialize_joins.py --dataset mimic_small --run_all_admissions

# Limit to first N admissions (useful during development)
python materialize_joins.py --dataset mimic --run_all_admissions --max_admissions 20

# Show batch progress every admission instead of every 25
python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1

# Resume after an LM Studio interruption using the existing batch CSV
python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1 --resume

# Resume with a custom LM Studio retry budget
python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1 --resume --llm_retry_attempts 8

# Batch with GLiNER2 (no LM Studio needed — faster but lower quality)
python materialize_joins.py --dataset mimic --run_all_admissions \
  --cluster_label_backend gliner2

# Batch with historical pair-identity LMStudio grouping
python materialize_joins.py --dataset mimic --run_all_admissions --llm_no_hdbscan

# Suppress the per-type metrics table (speeds up large batch output)
python materialize_joins.py --dataset mimic --run_all_admissions --no_typed_metrics
```

**Full batch 8-combo GLiNER2 grid:**

```powershell
python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode signature                   --gliner2_label_input_mode sentence_evidence   --batch_progress_every 1
python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode signature                   --gliner2_label_input_mode semantic_signature  --batch_progress_every 1
python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode contextual_sentence_average  --gliner2_label_input_mode sentence_evidence   --batch_progress_every 1
python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode contextual_sentence_average  --gliner2_label_input_mode semantic_signature  --batch_progress_every 1
python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode row_pair_hybrid              --gliner2_label_input_mode sentence_evidence   --batch_progress_every 1
python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode row_pair_hybrid              --gliner2_label_input_mode semantic_signature  --batch_progress_every 1
python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode semantic_signature           --gliner2_label_input_mode sentence_evidence   --batch_progress_every 1
python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode semantic_signature           --gliner2_label_input_mode semantic_signature  --batch_progress_every 1
```

**Batch with LMStudio — Qwen3:**

```powershell
# Per-cluster (fastest)
python materialize_joins.py --dataset mimic_small --run_all_admissions --cluster_label_backend lmstudio --llm_model medgemma-27b-text-it --llm_no_agglomerative --batch_progress_every 1

# Agglomerative d=0.35
python materialize_joins.py --dataset mimic_small --run_all_admissions --cluster_label_backend lmstudio --llm_model medgemma-27b-text-it --llm_agglom_distance 0.35 --batch_progress_every 1
```

**Batch with LMStudio — MedGemma-27B:**

```powershell
# MedGemma agglomerative d=0.35
python materialize_joins.py --dataset mimic_small --run_all_admissions \
  --cluster_label_backend lmstudio \
  --llm_model medgemma-27b-text-it \
  --llm_agglom_distance 0.35 \
  --batch_progress_every 1
```

**Batch with Oracle (upper bound):**

```powershell
python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend oracle --batch_progress_every 1
```

---

### 1.5 — Advanced Options Reference

These flags are available for both single-admission and batch modes.

#### Clustering

| Flag | Default | Description |
|------|---------|-------------|
| `--hdbscan_min_cluster_size` | `4` | HDBSCAN minimum cluster size in the current default profile. Pass `0` to calibrate automatically. |
| `--max_clusters` | `0` (disabled) | Cap on number of clusters. If exceeded, clusters are merged via meta-clustering. |
| `--enable_meta_clustering` | flag | Enable hierarchical meta-clustering to merge fragment clusters by Jaccard sentence/pair overlap. |
| `--no_shared_pair_merge` | **(default: on)** | Disable must-link merging of clusters that share a (diag, med) pair. |
| `--shared_pair_merge` | off | Re-enable must-link merging of clusters that share a (diag, med) pair. |
| `--no_cluster_tail_filter` | flag | Disable tail-pair trimming inside large clusters. |
| `--cluster_tail_mode` | `adaptive_std` | Tail filtering mode: `legacy`, `conservative`, `soft_weight`, `adaptive_std`, `adaptive_percentile`. |

#### Stage 5 Path Extraction Tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | auto (μ+σ) | Fixed γ score threshold. Leave unset for adaptive threshold. |
| `--stage5_top_k` | auto | Row-side top-k for Stage 5 extraction. |
| `--stage5_sent_diag_top_k` | `8` | Max sentences per diagnosis row. |
| `--stage5_sent_med_top_k` | `12` | Max sentences per medication row. |
| `--stage5_max_pairs_per_sentence` | `12` | Max (diag, med) pairs contributed by one sentence. |
| `--stage5_max_sentences_per_pair` | `3` | Max mediating sentences kept per pair. |
| `--no_pair_filter` | flag | Disable weak singleton pair filter. |

#### Diagnostics & Debugging

| Flag | Description |
|------|-------------|
| `--show_typed_metrics` | Print per-label typed F1 scores in addition to the default output. **(default: on)** |
| `--no_typed_metrics` | Suppress the per-type metrics table (opt-out from the default `--show_typed_metrics`). |
| `--enable_pair_recovery_diagnostics` | Record which GT pairs failed and at which pipeline stage. |
| `--debug_recall_cascade` | Verbose per-pair recall diagnostics printed to console. |
| `--diagnostics_output_dir <path>` | Write diagnostic artifacts to a directory. |
| `--skip_visualizations` | Skip PNG generation (useful for faster repeated comparisons). |

#### Cross-Encoder Per-Pair Sentence Rerank (Phase D.5, Option C)

Default-on zero-shot cross-encoder that **rewrites the within-pair evidence ordering** seen by Phase E. For every `(diag, med)` pair that survives Phase D + pair_filter, each of its 1–3 mediating sentences is scored by the CE (`query = "Diagnosis: … | Medication: …"`, `passage = "[section] sentence"`), and a `ce_score` field is attached to every path record. Phase E's signature builders (used by LMStudio, agglom, and GLiNER2 labelers) then prefer `ce_score` over LOKI's `path_score` when choosing the representative sentence(s) for each pair.

**What this changes:** the *evidence text* shown to the labeling backend per surviving pair. The materialized table preview's representative sentence per pair may also change.

**What this does NOT change:** the set of `(diag, med)` pairs that reach Phase E (CE never adds, drops, or pair_filters anything). Pair recall, pair precision at the candidate level, and the gamma threshold are all unaffected. Differences in F1 vs. a non-CE run reflect labeling quality only.

**Cost:** ~1–3 CE forward passes per surviving pair (typically 100–400 pairs after pair_filter), so the rerank adds ≤ 5 seconds even on a 3060 — orders of magnitude cheaper than the prior "all non-zero triples" diagnostic.

| Flag | Default | Description |
|------|---------|-------------|
| `--use_cross_encoder` | on | Enable per-pair sentence rerank. |
| `--no_cross_encoder` | off | Disable the default per-pair sentence rerank. |
| `--cross_encoder_model` | `Alibaba-NLP/gte-reranker-modernbert-base` | Any `sentence-transformers.CrossEncoder`-compatible HF checkpoint. |
| `--cross_encoder_device` | auto | Override device (`cuda`, `cpu`). |
| `--cross_encoder_batch_size` | `32` | Scoring batch size. |
| `--cross_encoder_max_length` | `512` | Tokenizer max length. |
| `--cross_encoder_no_fp16` | flag | Disable fp16 inference. |
| `--cross_encoder_no_section_prefix` | flag | Drop the `[section_name]` prefix from the passage. |
| `--cross_encoder_no_normalize` | flag | Return raw logits instead of sigmoid-normalized scores. |

**Smoke test (standalone, no LOKI required):**
```powershell
c:/Users/SHAON/anaconda3/envs/THOR/python.exe cross_encoder_rerank.py --smoke_test
```

**Recommended: full pipeline with current defaults** — the proper A/B against your baseline LOKI run:
```powershell
c:/Users/SHAON/anaconda3/envs/THOR/python.exe materialize_joins.py --single_admission --dataset mimic --admission_id 20393363
```

**Other backend pairings (CE rerank is orthogonal to the labeling backend):**
```powershell
# GLiNER2 backend + CE rerank
c:/Users/SHAON/anaconda3/envs/THOR/python.exe materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --use_cross_encoder

# Per-cluster LMStudio (no agglom) + CE rerank
c:/Users/SHAON/anaconda3/envs/THOR/python.exe materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend lmstudio --use_cross_encoder

# Oracle backend + CE rerank — labeling upper bound, isolates evidence-selection effect
c:/Users/SHAON/anaconda3/envs/THOR/python.exe materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend oracle --use_cross_encoder
```

**How to read the Phase D.5 console block:**
- `Reranking sentences within N surviving (diag, med) pairs (M total path-sentences) using ...` — sizes the rerank.
- `CE score distribution: n=… p50=… p90=… mean=…` — sanity check that CE returned non-degenerate scores.
- `Top-1 evidence sentence changed in X/Y multi-sentence pairs (Z%)` — fraction of pairs where CE picked a different best sentence than LOKI. Z = 0% means CE agreed with LOKI everywhere (rerank had no effect); high Z means downstream metrics will diverge from a no-CE run.
- `GT triple separation on surviving candidates: AUC-ROC=… AUC-PR=…` — diagnostic-only AUC over the surviving candidate set (not the full 208k cross-product). Higher is better.

**Notes:**
- The pair-level recall, gamma threshold, pair_filter survivors, and cluster-tail filter all run *before* the CE rerank, so pipeline pair counts are independent of `--use_cross_encoder`. Only what Phase E *sees per pair* changes.
- First run downloads the model to `%USERPROFILE%\.cache\huggingface\hub\`.
- `cross-encoder/ettin-reranker-400m-v1` requires `transformers>=5.2.0`. The current THOR env pins to 4.57.6 (Unsloth dependency); ettin will fail to load until the env is upgraded.

---

## Part 2 — Direct Comparison Workflows

`ablation.py` has been removed. To compare models or clustering profiles, run `materialize_joins.py` directly and archive each output folder before the next run so the next command does not overwrite the previous artifacts.

---

### 2.1 — Compare Two LMStudio Models on One Admission

```powershell
Set-Location 'LOKI'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --llm_model openai/gpt-oss-20b
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_gpt_oss_20b'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --llm_model <SECOND_MODEL>
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_second_model'
```

Swap `<SECOND_MODEL>` for any LM Studio model you have loaded. After each run, compare the archived folders' JSON, CSV, audit Markdown, and PNGs side by side.

---

### 2.2 — Compare LMStudio vs GLiNER2 on One Admission

```powershell
Set-Location 'LOKI'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --llm_model openai/gpt-oss-20b
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_lmstudio'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend gliner2 \
  --pair_embedding_mode contextual_sentence_average \
  --gliner2_label_input_mode sentence_evidence
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_gliner2'
```

The GLiNER2 example uses a historically stronger fallback profile. Keep the admission id fixed so the audits are directly comparable.

---

### 2.3 — Compare Two Models on the Full Dataset

Batch mode already reuses the current parser defaults. The only extra step is archiving the batch output directory between runs.

```powershell
Set-Location 'LOKI'

python materialize_joins.py --dataset mimic --llm_model openai/gpt-oss-20b
Rename-Item '..\Batch_Materialization\loki_batch_mimic' 'loki_batch_mimic_gpt_oss_20b'

python materialize_joins.py --dataset mimic --llm_model <SECOND_MODEL>
Rename-Item '..\Batch_Materialization\loki_batch_mimic' 'loki_batch_mimic_second_model'
```

Each archived batch folder contains the aggregate CSVs, Markdown report, and dashboard PNGs for that model.

---

### 2.4 — Compare the Current Default Profile Against Historical `--llm_no_hdbscan`

```powershell
Set-Location 'LOKI'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --llm_model openai/gpt-oss-20b
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_default_profile'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --llm_model openai/gpt-oss-20b --llm_no_hdbscan
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_no_hdbscan'
```

Use this when you want to check whether the newer HDBSCAN-backed refinement stack still beats the historical pair-identity grouping on the same admission.

---

## Part 3 — Output Files Reference

### `materialize_joins.py` outputs (single-admission)

Written to `Batch_Materialization\loki_run_<id>\` (all per-run artifacts collected in a single folder):

Every visualization listed below is written twice: the documented `.png` file and a sibling `.pdf` with the same basename.

| File | Description |
|------|-------------|
| `materialized_joins_<id>.json` | Full join paths with cluster labels and scores |
| `materialized_table_<id>.csv` | Integrated relationship table (diag × med × label) |
| `cluster_audit_<id>.md` | Cluster-by-cluster label audit report |
| `embedding_space_<id>.png` | 2D UMAP/PCA projection of cluster embeddings |
| `join_topic_map_<id>.png` | Join-path topic map visualization |
| `clusters_tsne_<id>.png` | t-SNE of all candidate row-pair embeddings colored by HDBSCAN cluster |
| `semantic_cluster_projection_<id>.png` | Semantic grouping of predicted join paths |
| `clusters_tsne_full_<id>.png` | t-SNE of all sentence embeddings |
| `classwise_typed_metrics_<id>.png` | Per-class auto vs oracle F1 bar chart |
| `agglom_recluster_<id>.png` | **LMStudio agglom only.** Two-panel PCA/t-SNE showing LLM agglomerative re-groups (left) vs original HDBSCAN clusters (right). Useful for diagnosing whether agglom re-clustering is semantically coherent. |
| `llm_vs_hdbscan_<id>.png` | **`--llm_no_hdbscan` only.** Two-panel BGE-large/TF-IDF embedding showing LLM pair-identity groups (left, coloured by predicted label) vs original HDBSCAN clusters (right). Use this to judge whether bypassing HDBSCAN produces more coherent semantic groupings. |

### `materialize_joins.py` outputs (batch)

Written to `Batch_Materialization\loki_batch_<dataset>\`:

Every batch dashboard listed below is written as both `.png` and same-basename `.pdf`.

| File | Description |
|------|-------------|
| `materialized_batch_results_<dataset>.csv` | Per-admission metrics (one row per admission) |
| `materialized_batch_summary_<dataset>.csv` | Aggregate summary statistics |
| `materialized_batch_report_<dataset>.md` | Human-readable batch report |
| `materialized_batch_metrics_<dataset>.png` | Metric distribution plots |
| `materialized_batch_classwise_metrics_<dataset>.png` | Per-class F1 bar charts |

---

## Part 4 — Common Workflows

### "I want to quickly try the pipeline on one admission"

> Make sure LM Studio is running with a model loaded (for example `openai/gpt-oss-20b`) — the default backend is now `lmstudio`.

```powershell
Set-Location 'LOKI'
python materialize_joins.py --single_admission
```

To run without LM Studio (GLiNER2 fallback):

```powershell
python materialize_joins.py --single_admission --cluster_label_backend gliner2
```

### "I want to compare two LMStudio models on one admission"

```powershell
Set-Location 'LOKI'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --llm_model openai/gpt-oss-20b
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_gpt_oss_20b'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --llm_model <SECOND_MODEL>
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_second_model'
```

### "I want to compare LMStudio vs GLiNER2 on one admission"

```powershell
Set-Location 'LOKI'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --llm_model openai/gpt-oss-20b
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_lmstudio'

python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
  --cluster_label_backend gliner2 \
  --pair_embedding_mode contextual_sentence_average \
  --gliner2_label_input_mode sentence_evidence
Rename-Item '..\Batch_Materialization\loki_run_20393363' 'loki_run_20393363_gliner2'
```

### "I want to compare two models on the full mimic dataset"

```powershell
Set-Location 'LOKI'

python materialize_joins.py --dataset mimic --llm_model openai/gpt-oss-20b
Rename-Item '..\Batch_Materialization\loki_batch_mimic' 'loki_batch_mimic_gpt_oss_20b'

python materialize_joins.py --dataset mimic --llm_model <SECOND_MODEL>
Rename-Item '..\Batch_Materialization\loki_batch_mimic' 'loki_batch_mimic_second_model'
```

### "I want to understand why a specific admission is mislabeled"

```powershell
Set-Location 'LOKI'
python materialize_joins.py --single_admission --dataset mimic \
  --admission_id <ADMISSION_ID> \
  --cluster_label_backend gliner2 \
  --pair_embedding_mode contextual_sentence_average \
  --gliner2_label_input_mode sentence_evidence \
  --show_typed_metrics \
  --enable_pair_recovery_diagnostics \
  --debug_recall_cascade
```
