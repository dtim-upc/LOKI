#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
materialize_joins.py
====================

Stage 5 + 6 of the LOKI pipeline: cross-table join path extraction and
semantic materialization for a single MIMIC hospital admission or a batch
of admissions.

Stage 5 - Joint encoding of T_A U T_B with document D
  1. Concatenate diagnosis rows + medication rows -> joint R [n_A+n_B, d]
  2. Encode note sentences -> S [189, d]
  3. Single LOKI bidirectional forward pass -> pair score matrix P [n_A+n_B, 189]
     and refined sentence embeddings S̃ [189, d]
  4. Extract atomic links J_A (diag->sent) and J_B (med->sent) above threshold gamma
  5. Transitive join on shared mediating sentences -> candidate (diag, sent, med) paths

Stage 6 - Semantic materialization
  6. Cluster mediating S̃_j embeddings (HDBSCAN) to discover relationship types
    7. Label each cluster via GLiNER2 anchored entity + relation inference
  8. Assemble and output the integrated table T_integrated

Requirements
------------
    Activate the THOR conda environment before running these commands.

Single-Admission Quick Start (current best config - defaults match this command)
--------------------------------------------------------------------------------
    cd f:\\#LOKI_JOIN\\LOKI
    conda activate THOR
    python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363

        Equivalent fully-explicit invocation (all flags shown match defaults):
      python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 \
                    --cluster_label_backend lmstudio --llm_hdbscan --llm_no_agglomerative \
                                        --pair_embedding_mode contextual_sentence_average --hdbscan_min_cluster_size 4 \
                    --no_shared_pair_merge --cluster_refine_min_pairs 5 \
                    --cluster_refine_semantic_subsplit --cluster_refine_semantic_distance 0.20 \
                    --cluster_refine_llm_per_path_vote --cluster_refine_path_subsplit \
                    --cluster_refine_path_subsplit_min_mass 0.25 \
                    --cluster_refine_path_subsplit_min_share 0.30 \
                    --cluster_refine_path_subsplit_max_gap 0.12 \
                    --suppress_negative_clusters --use_cross_encoder \
                    --ce_pair_filter_mode combined --ce_pair_filter_quantile 0.25

    To disable Option D (CE pair-level filter) only:
      python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --ce_pair_filter_mode off

    To revert to legacy GLiNER2 labeling:
      python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode signature --gliner2_label_input_mode sentence_evidence

Single-Admission Variants
-------------------------
    python LOKI\\materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --show_typed_metrics --max_clusters 8
    python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode semantic_signature --gliner2_label_input_mode sentence_evidence
    python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode signature --gliner2_label_input_mode semantic_signature
    python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode semantic_signature --gliner2_label_input_mode semantic_signature
    python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode contextual_sentence_average --gliner2_label_input_mode sentence_evidence
    python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode contextual_sentence_average --gliner2_label_input_mode semantic_signature
    python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode row_pair_hybrid --gliner2_label_input_mode sentence_evidence
    python materialize_joins.py --single_admission --dataset mimic --admission_id 20393363 --cluster_label_backend gliner2 --pair_embedding_mode row_pair_hybrid --gliner2_label_input_mode semantic_signature
    python materialize_joins.py --single_admission --dataset mimic --admission_id 20301031 --cluster_label_backend gliner2 --pair_embedding_mode signature --gliner2_label_input_mode sentence_evidence

All-Admission Batch Runs
------------------------
    # Default (current best): LMStudio HDBSCAN + contextual_sentence_average + CE rerank +
    # pair-label semantic/path refinement + NEGATIVE suppression.
    # No explicit flags required - all defaults match the validated best config.
    python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1

    # Resume an interrupted LMStudio batch after connectivity is restored.
    # Completed admissions are loaded from the saved batch CSV and skipped.
    python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1 --resume

    # Same as above, but override the default 5 LMStudio retry attempts.
    python materialize_joins.py --dataset mimic --run_all_admissions --batch_progress_every 1 --resume --llm_retry_attempts 8

    # Legacy GLiNER2 batch run:
    python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode signature --gliner2_label_input_mode sentence_evidence --batch_progress_every 1
    python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode signature --gliner2_label_input_mode semantic_signature --batch_progress_every 1
    python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode semantic_signature --gliner2_label_input_mode sentence_evidence --batch_progress_every 1
    python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode semantic_signature --gliner2_label_input_mode semantic_signature --batch_progress_every 1
    python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode row_pair_hybrid --gliner2_label_input_mode sentence_evidence --batch_progress_every 1
    python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode row_pair_hybrid --gliner2_label_input_mode semantic_signature --batch_progress_every 1
    python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode contextual_sentence_average --gliner2_label_input_mode sentence_evidence --batch_progress_every 1
    python materialize_joins.py --dataset mimic --run_all_admissions --cluster_label_backend gliner2 --pair_embedding_mode contextual_sentence_average --gliner2_label_input_mode semantic_signature --batch_progress_every 1

Limited Batch Size Examples
---------------------------
    python materialize_joins.py --dataset mimic --run_all_admissions --max_admissions 12
    python materialize_joins.py --dataset mimic --run_all_admissions --max_admissions 50 --cluster_label_backend gliner2 --pair_embedding_mode signature --gliner2_label_input_mode sentence_evidence

Notes
-----
    - Running with no arguments defaults to full-dataset batch inference on mimic.
    - For single-admission runs, set --admission_id explicitly when you want an admission other than the built-in example.
    - Four pair_embedding_mode options: signature (encode full evidence text), semantic_signature (encode TF-IDF top-terms), contextual_sentence_average (score-weighted average of refined S̃_j - current default), row_pair_hybrid (concat [r̃_A || s̄_p || r̃_B] mirroring the join path triple).
    - Use gliner2_label_input_mode=sentence_evidence for default GLiNER2 naming evidence and gliner2_label_input_mode=semantic_signature to name clusters from pair semantic signatures.
    - The GLiNER2 naming input mode is independent from pair_embedding_mode; run the full 4x2 grid as an ablation when comparing settings.
    - If --cluster_label_backend keyword is selected, --gliner2_label_input_mode is ignored.
    - In batch mode with LMStudio labeling, transport failures retry 5 times by default, then stop the current run cleanly so it can be resumed with --resume after connectivity is fixed.
    - Batch runs reuse the same summary/report/result filenames for a given dataset profile; use --resume to continue the same run, or rename/copy artifacts if you want to preserve separate batch snapshots.
    - Single-admission artifacts are written under Batch_Materialization/loki_run_<admission_id>/.
"""

from __future__ import annotations

import sys
import os
import json
import csv
import argparse
import re
import io
import time
import contextlib
import hashlib
import warnings
import numpy as np
import torch

# UMAP emits a UserWarning every call when both `random_state` is set and the
# default `n_jobs` would have enabled parallelism - UMAP forces n_jobs=1 for
# reproducibility and warns about it. We deliberately want reproducible
# projections, so silence this specific recurring warning to keep batch logs
# readable. Filter once at import time so subprocess loggers inherit it.
warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value .* overridden to 1 by setting random_state.*",
    category=UserWarning,
    module=r"umap\..*",
)
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# -- project root on sys.path (allows importing models, initialization, etc.) -
sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer
from models import BidirectionalTableTextModel
from hf_model_resolver import (
    bootstrap_hf_model_snapshots,
    ensure_repo_local_hf_snapshot,
    get_configured_hf_cache_folder,
    get_repo_local_hf_cache_folder,
    load_hf_model_with_cache_fallback,
)

# -- configurable paths --------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).parent.parent
DEFAULT_DATASET_NAME = "mimic"
DATASET_CONFIGS: Dict[str, Dict[str, Path]] = {
    "mimic_ex3": {
        "data_file": WORKSPACE_ROOT / "Datasets/mimic_ex3/test_row_level.json",
        "annot_file": WORKSPACE_ROOT / "Datasets/mimic_ex3/Annotated_Test.json",
    },
    "mimic_balanced_top3": {
        "data_file": WORKSPACE_ROOT / "Datasets/mimic_balanced_top3/test_row_level.json",
        "annot_file": WORKSPACE_ROOT / "Datasets/mimic_balanced_top3/Annotated_Test.json",
    },
    "mimic_small": {
        "data_file": WORKSPACE_ROOT / "Datasets/mimic_small/test_row_level.json",
        "annot_file": WORKSPACE_ROOT / "Datasets/mimic_small/Annotated_Test.json",
    },
    "mimic": {
        "data_file": WORKSPACE_ROOT / "Datasets/mimic/test_row_level.json",
        "annot_file": WORKSPACE_ROOT / "Datasets/mimic/Annotated_Test.json",
    },
}
DATA_FILE      = DATASET_CONFIGS[DEFAULT_DATASET_NAME]["data_file"]
ANNOT_FILE     = DATASET_CONFIGS[DEFAULT_DATASET_NAME]["annot_file"]
MODEL_DIR      = WORKSPACE_ROOT / "model"

DEFAULT_CKPT = MODEL_DIR / "abhinand/MedEmbed-large-v0.1/best_test_avg_precision_epoch_16/model.pt"

ARGS_FILE      = MODEL_DIR / "args.json"
ENCODER_NAME   = "abhinand/MedEmbed-large-v0.1"
DEFAULT_BGE_ENCODER_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_MINILM_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ADMISSION_ID   = "20393363"
TARGET_PATIENT = "10155915"
DEFAULT_ADMISSION_ID = ADMISSION_ID
DEFAULT_TARGET_PATIENT = TARGET_PATIENT
DEFAULT_SINGLE_ADMISSION_DATASET = "mimic"

GLINER2_MODEL = "fastino/gliner2-base-v1"
DEFAULT_CLUSTER_LABEL_BACKEND = "gliner2"
DEFAULT_GLINER2_LABEL_INPUT_MODE = "sentence_evidence"

_GLINER2_MODEL_CACHE: Dict[str, object] = {}

# -- LMStudio (local LLM) cluster labeling constants ---------------------
LMSTUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1234"
# LMSTUDIO_DEFAULT_BASE_URL = "http://192.168.1.128:1234"
LMSTUDIO_DEFAULT_MODEL = "qwen3.6-35b-a3b-mtp"
LMSTUDIO_DEFAULT_TEMPERATURE = 0.0
LMSTUDIO_DEFAULT_TIMEOUT_SECS = 3600
LMSTUDIO_DEFAULT_RETRY_ATTEMPTS = 5
LMSTUDIO_DEFAULT_MAX_EVIDENCE_SENTS = 0
LMSTUDIO_DEFAULT_AGGLOM_DISTANCE = 0.25
_LMSTUDIO_LABEL_CACHE: Dict[str, str] = {}
_AGGLOM_ENCODER_CACHE: Dict[str, Any] = {}
_LMSTUDIO_FAIL_CLOSED = False
_LMSTUDIO_RETRY_ATTEMPTS = LMSTUDIO_DEFAULT_RETRY_ATTEMPTS

# All per-run artifacts (materialized JSON/CSV, cluster audit MD, Stage-5/6
# visualization plots written as PNG plus sibling PDF) are collected under
# VIS_DIR so the workspace root stays
# uncluttered. Per-run and per-batch folders are themselves nested under a
# single top-level BATCH_MATERIALIZATION_DIR to keep the workspace root clean.
# These defaults are overridden per-run inside configure_runtime_context()
# once the actual run tag is known.
BATCH_MATERIALIZATION_DIR = WORKSPACE_ROOT / "Batch_Materialization"
VIS_DIR = BATCH_MATERIALIZATION_DIR / f"loki_run_{ADMISSION_ID}"
OUT_JSON = VIS_DIR / f"materialized_joins_{ADMISSION_ID}.json"
OUT_CSV  = VIS_DIR / f"materialized_table_{ADMISSION_ID}.csv"
OUT_AUDIT = VIS_DIR / f"cluster_audit_{ADMISSION_ID}.md"
OUT_EMBEDDING = VIS_DIR / f"embedding_space_{ADMISSION_ID}.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PairEmbeddingTensor = torch.Tensor
PairEmbeddingCacheKey = Tuple[int, int, int, int, int, str]
PairEmbeddingCache = Dict[PairEmbeddingCacheKey, Tuple[List[Tuple[int, int]], PairEmbeddingTensor]]
_PAIR_EMBEDDING_CACHE: PairEmbeddingCache = {}


def _to_numpy_array(values: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(dtype, copy=False) if dtype is not None else values
    if torch.is_tensor(values):
        array = values.detach().cpu().numpy()
        return array.astype(dtype, copy=False) if dtype is not None else array
    return np.asarray(values, dtype=dtype)


def _save_figure_outputs(
    fig: Any,
    out_path: str | Path,
    **savefig_kwargs: Any,
) -> None:
    target_path = Path(out_path)
    fig.savefig(target_path, **savefig_kwargs)
    if target_path.suffix.lower() == ".png":
        fig.savefig(target_path.with_suffix(".pdf"), **savefig_kwargs)


def _pair_embedding_device(
    refined_sentences: Optional[torch.Tensor],
    refined_rows: Optional[torch.Tensor] = None,
) -> torch.device:
    if torch.is_tensor(refined_sentences):
        return refined_sentences.device
    if torch.is_tensor(refined_rows):
        return refined_rows.device
    return DEVICE


def _ensure_embedding_tensor(
    values: Any,
    *,
    device: Optional[torch.device] = None,
    normalize: bool = False,
) -> torch.Tensor:
    if torch.is_tensor(values):
        tensor = values.detach()
    else:
        tensor = torch.as_tensor(values)
    target_device = device if device is not None else tensor.device
    tensor = tensor.to(device=target_device, dtype=torch.float32)
    if normalize:
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=-1)
    return tensor


def _build_sentence_transformer(model_name: str, **kwargs) -> Tuple[SentenceTransformer, str, str]:
    sentence_encoder, resolved_model_name, model_source = load_hf_model_with_cache_fallback(
        SentenceTransformer,
        model_name,
        **kwargs,
    )
    return sentence_encoder, resolved_model_name, model_source


def _required_hf_models_for_run(cli: argparse.Namespace) -> List[str]:
    required_models: List[str] = [ENCODER_NAME]

    if bool(getattr(cli, "use_cross_encoder", False)):
        required_models.append(str(getattr(cli, "cross_encoder_model", "")).strip())

    cluster_label_backend = str(getattr(cli, "cluster_label_backend", "")).strip().lower()
    if cluster_label_backend == "gliner2":
        required_models.append(str(getattr(cli, "gliner2_model", "")).strip())

    if cluster_label_backend == "lmstudio":
        llm_agglom_encoder = str(getattr(cli, "llm_agglom_encoder", "")).strip().lower()
        if bool(getattr(cli, "llm_agglomerative", False)):
            if llm_agglom_encoder == "bge":
                required_models.append(DEFAULT_BGE_ENCODER_NAME)
            elif llm_agglom_encoder == "minilm":
                required_models.append(DEFAULT_MINILM_ENCODER_NAME)

        if bool(getattr(cli, "llm_no_hdbscan", False)) and not bool(getattr(cli, "skip_visualizations", False)):
            required_models.append(DEFAULT_BGE_ENCODER_NAME)

    deduped_models: List[str] = []
    seen: Set[str] = set()
    for raw_model_name in required_models:
        model_name = str(raw_model_name or "").strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        deduped_models.append(model_name)
    return deduped_models


def _bootstrap_hf_assets_for_run(cli: argparse.Namespace) -> None:
    required_models = _required_hf_models_for_run(cli)
    if not required_models:
        return

    print("\n-- Phase 0: Hugging Face Asset Bootstrap -------------------------")
    managed_cache_folder = get_repo_local_hf_cache_folder()
    print(f"  Managed repo-local cache: {managed_cache_folder}")

    external_cache_folder = get_configured_hf_cache_folder()
    if external_cache_folder:
        print(f"  External cache source: {external_cache_folder}")

    bootstrap_records = bootstrap_hf_model_snapshots(required_models, allow_online=True)
    for record in bootstrap_records:
        print(
            f"  {record['model_name']} -> {record['source']}\n"
            f"    {record['resolved_path']}"
        )


# =============================================================================
# Phase A - Data loading
# =============================================================================

def load_admission_data() -> Tuple[List[str], List[str], List[str], Dict[int, str]]:
    """
    Parse test_row_level_v2.json and return (diag_rows, med_rows, sent_texts, sent_meta)
    for admission ADMISSION_ID.
    """
    with open(DATA_FILE, encoding="utf-8") as f:
        examples = json.load(f)

    diag_rows: Optional[List[str]] = None
    med_rows:  Optional[List[str]] = None
    sent_texts: Optional[List[str]] = None
    sent_meta:  Optional[Dict[int, str]] = None

    for ex in examples:
        if ex.get("patient_id") != TARGET_PATIENT or ex.get("admission_id") != ADMISSION_ID:
            continue

        tables = ex.get("tables", {})

        if "diagnosis" in tables:
            diag_rows = [
                r["formatted"]
                for r in sorted(tables["diagnosis"]["rows"], key=lambda r: r["row_idx"])
            ]
            # Sentences are stored under primary_positive.sentences
            primary_pos = ex.get("primary_positive", ex)  # fall back to top-level
            sentences   = primary_pos.get("sentences", {})
            if not sentences:
                # second fallback: top-level key
                sentences = ex.get("sentences", {})
            sent_texts = [sentences[k]["text"] for k in sorted(sentences.keys(), key=int)]
            sent_meta  = {int(k): sentences[k]["section_name"] for k in sentences}

        elif "medication" in tables:
            med_rows = [
                r["formatted"]
                for r in sorted(tables["medication"]["rows"], key=lambda r: r["row_idx"])
            ]

    assert diag_rows  is not None, "Diagnosis rows not found for target admission"
    assert med_rows   is not None, "Medication rows not found for target admission"
    assert sent_texts is not None, "Note sentences not found for target admission"

    print(
        f"  Loaded {len(diag_rows)} diagnosis rows, "
        f"{len(med_rows)} medication rows, "
        f"{len(sent_texts)} note sentences"
    )
    return diag_rows, med_rows, sent_texts, sent_meta


def load_ground_truth():
    """
    Load Annotated_Test.json and return:
      gt_relationships : list of dicts {diag_idx, drug_idx, rel_type, evidence_sents}
                         (0-based row indices; evidence_sents are 0-based sent indices)
      gt_diag          : {0-based diag_idx: [sent_idxs]} from row_grounding (for row-recall)
      gt_med           : {0-based med_idx:  [sent_idxs]} from row_grounding (for row-recall)
      multi_pairs      : set of (diag_idx, drug_idx) pairs that carry >1 relationship type
    """
    with open(ANNOT_FILE, encoding="utf-8") as f:
        annots = json.load(f)

    entry = annots[ADMISSION_ID]

    # Row-level grounding (used only for per-table row-recall metric)
    rg = entry["row_grounding"]
    gt_diag = {int(k) - 1: v["sentences"] for k, v in rg["diagnosis"].items()}
    gt_med  = {int(k) - 1: v["sentences"] for k, v in rg["medication"].items()}

    # Primary GT: the annotated relationships (drug_row / diagnosis_row are 1-based)
    gt_relationships = []
    for rel in entry["relationships"]:
        d_idx = rel["diagnosis_row"] - 1
        m_idx = rel["drug_row"] - 1
        gt_relationships.append({
            "diag_idx"      : d_idx,
            "drug_idx"      : m_idx,
            "rel_type"      : _normalize_rel_type(rel["relationship_type"]),
            "evidence_sents": rel["evidence_sentences"],  # already 0-based
        })

    # Multi-relationship flags: each flag entry independently asserts all its
    # relationship types for the (diag, drug) pair.  These are counted as separate
    # GT annotations alongside the main relationships array (no deduplication).
    multi_pairs: set = set()
    for flag in entry.get("multi_relationship_flags", []):
        d_idx = flag["diagnosis_row"] - 1
        m_idx = flag["drug_row"]      - 1
        multi_pairs.add((d_idx, m_idx))
        for rtype in flag["relationship_types"]:
            gt_relationships.append({
                "diag_idx"      : d_idx,
                "drug_idx"      : m_idx,
                "rel_type"      : _normalize_rel_type(rtype),
                "evidence_sents": [],
            })

    from collections import Counter as _Counter
    active_rel_types, rel_type_sources = _resolve_rel_types_from_annotation_corpus()
    type_counts = _Counter(r["rel_type"] for r in gt_relationships)
    n_unique_pairs = len({(r["diag_idx"], r["drug_idx"]) for r in gt_relationships})
    print(
        f"  Ground truth: {len(gt_relationships)} relationships "
        f"({', '.join(f'{v} {k}' for k, v in sorted(type_counts.items(), key=lambda item: _rel_type_sort_key(item[0])))}), "
        f"{n_unique_pairs} unique (diag, drug) pairs, "
        f"{len(multi_pairs)} multi-relationship pairs"
    )
    source_labels = ", ".join(
        str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
        for path in rel_type_sources
    )
    print(f"  Active relationship types (annotation corpus): {', '.join(active_rel_types)}")
    print(f"  Relationship inventory sources: {source_labels}")
    print(
        f"  Row coverage: {len(gt_diag)} diagnosis rows, "
        f"{len(gt_med)} medication rows annotated"
    )
    return gt_relationships, gt_diag, gt_med, multi_pairs


def get_dataset_paths(dataset_name: str) -> Tuple[Path, Path]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of {sorted(DATASET_CONFIGS)}")
    config = DATASET_CONFIGS[dataset_name]
    return config["data_file"], config["annot_file"]


def load_dataset_examples(data_file: Path) -> Dict[str, Dict]:
    with open(data_file, encoding="utf-8") as f:
        examples = json.load(f)

    admission_index: Dict[str, Dict] = {}
    for ex in examples:
        admission_id = str(ex.get("admission_id", "")).strip()
        patient_id = str(ex.get("patient_id", "")).strip()
        if not admission_id:
            continue

        record = admission_index.setdefault(
            admission_id,
            {
                "admission_id": admission_id,
                "patient_id": patient_id,
                "diagnosis_example": None,
                "medication_example": None,
            },
        )
        if record["patient_id"] and patient_id and record["patient_id"] != patient_id:
            raise ValueError(
                f"Admission {admission_id} maps to multiple patient ids: "
                f"{record['patient_id']} vs {patient_id}"
            )
        if patient_id:
            record["patient_id"] = patient_id

        tables = ex.get("tables", {})
        if "diagnosis" in tables:
            record["diagnosis_example"] = ex
        if "medication" in tables:
            record["medication_example"] = ex

    return admission_index


def load_annotation_entries(annot_file: Path) -> Dict[str, Dict]:
    with open(annot_file, encoding="utf-8") as f:
        annots = json.load(f)
    return {str(admission_id): entry for admission_id, entry in annots.items()}


def _extract_negative_pairs_from_annotation_entry(entry: Dict) -> set[Tuple[int, int]]:
    negative_pairs: set[Tuple[int, int]] = set()
    for record in entry.get("negative_relationships", []):
        diag_row = record.get("diagnosis_row")
        drug_row = record.get("drug_row")
        if diag_row is None or drug_row is None:
            continue
        negative_pairs.add((int(diag_row) - 1, int(drug_row) - 1))
    return negative_pairs


EVALUATION_PROFILE_DEFAULT = "default"
EVALUATION_PROFILE_AE_DIS_CLEAN = "ae_dis_clean"
AE_DIS_CLEAN_LABELS = frozenset({"ADVERSE_EFFECT", "DISCONTINUED"})
AE_DIS_CLEAN_CANDIDATE_LABELS = frozenset({"ADVERSE_EFFECT", "DISCONTINUED", "NEGATIVE"})


def _normalize_evaluation_profile(profile: str) -> str:
    resolved = str(profile or EVALUATION_PROFILE_DEFAULT).strip().lower()
    return resolved or EVALUATION_PROFILE_DEFAULT


def _resolved_output_dataset_name(dataset_name: str, evaluation_profile: str) -> str:
    resolved_profile = _normalize_evaluation_profile(evaluation_profile)
    if resolved_profile == EVALUATION_PROFILE_DEFAULT:
        return dataset_name
    return f"{dataset_name}_{resolved_profile}"


def _resolve_candidate_labels(candidate_labels: Optional[List[str]] = None) -> List[str]:
    normalized_labels: List[str] = []
    seen: Set[str] = set()
    for raw_label in candidate_labels or _preferred_rel_type_order():
        normalized_label = _normalize_rel_type(raw_label)
        if not normalized_label or normalized_label in seen:
            continue
        normalized_labels.append(normalized_label)
        seen.add(normalized_label)
    return _preferred_rel_type_order(normalized_labels or _preferred_rel_type_order())


def _candidate_labels_for_evaluation_profile(evaluation_profile: str) -> List[str]:
    resolved_profile = _normalize_evaluation_profile(evaluation_profile)
    if resolved_profile == EVALUATION_PROFILE_AE_DIS_CLEAN:
        return _resolve_candidate_labels(list(AE_DIS_CLEAN_CANDIDATE_LABELS))
    return _resolve_candidate_labels()


def _annotation_pair_label_sets(entry: Dict) -> Dict[Tuple[int, int], Set[str]]:
    pair_label_sets: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

    for rel in entry.get("relationships", []):
        pair_label_sets[(int(rel["diagnosis_row"]) - 1, int(rel["drug_row"]) - 1)].add(
            _normalize_rel_type(rel.get("relationship_type", ""))
        )

    for flag in entry.get("multi_relationship_flags", []):
        pair_key = (int(flag["diagnosis_row"]) - 1, int(flag["drug_row"]) - 1)
        for rel_type in flag.get("relationship_types", []):
            pair_label_sets[pair_key].add(_normalize_rel_type(rel_type))

    return pair_label_sets


def _clean_ae_dis_target_pairs_from_annotation_entry(
    entry: Dict,
) -> Tuple[Set[Tuple[int, int]], Dict[str, int]]:
    pair_label_sets = _annotation_pair_label_sets(entry)
    target_pairs: Set[Tuple[int, int]] = set()
    n_ae_only = 0
    n_dis_only = 0

    for pair_key, labels in pair_label_sets.items():
        if labels == {"ADVERSE_EFFECT"}:
            target_pairs.add(pair_key)
            n_ae_only += 1
        elif labels == {"DISCONTINUED"}:
            target_pairs.add(pair_key)
            n_dis_only += 1

    return target_pairs, {
        "n_target_pairs": len(target_pairs),
        "n_ae_only_pairs": n_ae_only,
        "n_dis_only_pairs": n_dis_only,
        "has_clean_ae": 1 if n_ae_only > 0 else 0,
        "has_clean_dis": 1 if n_dis_only > 0 else 0,
    }


def _admission_matches_evaluation_profile(entry: Dict, evaluation_profile: str) -> bool:
    resolved_profile = _normalize_evaluation_profile(evaluation_profile)
    if resolved_profile == EVALUATION_PROFILE_DEFAULT:
        return True
    if resolved_profile != EVALUATION_PROFILE_AE_DIS_CLEAN:
        raise ValueError(f"Unsupported evaluation profile: {evaluation_profile}")

    _target_pairs, stats = _clean_ae_dis_target_pairs_from_annotation_entry(entry)
    return bool(stats.get("has_clean_ae")) and bool(stats.get("has_clean_dis"))


def _filter_ground_truth_for_target_pairs(
    gt_relationships: List[Dict],
    gt_diag: Dict[int, List[int]],
    gt_med: Dict[int, List[int]],
    multi_pairs: Set[Tuple[int, int]],
    target_pairs: Optional[Set[Tuple[int, int]]],
) -> Tuple[List[Dict], Dict[int, List[int]], Dict[int, List[int]], Set[Tuple[int, int]]]:
    if not target_pairs:
        return [], {}, {}, set()

    filtered_relationships = [
        rel
        for rel in gt_relationships
        if (int(rel["diag_idx"]), int(rel["drug_idx"])) in target_pairs
    ]
    kept_diag_rows = {int(rel["diag_idx"]) for rel in filtered_relationships}
    kept_med_rows = {int(rel["drug_idx"]) for rel in filtered_relationships}

    filtered_gt_diag = {
        row_idx: sentences
        for row_idx, sentences in gt_diag.items()
        if int(row_idx) in kept_diag_rows
    }
    filtered_gt_med = {
        row_idx: sentences
        for row_idx, sentences in gt_med.items()
        if int(row_idx) in kept_med_rows
    }
    filtered_multi_pairs = {
        pair for pair in multi_pairs
        if pair in target_pairs
    }
    return filtered_relationships, filtered_gt_diag, filtered_gt_med, filtered_multi_pairs


def _filter_paths_for_target_pairs(
    paths: Optional[List[Dict]],
    target_pairs: Optional[Set[Tuple[int, int]]],
) -> Optional[List[Dict]]:
    if paths is None or target_pairs is None:
        return paths
    return [
        path for path in paths
        if (int(path["diag_row_idx"]), int(path["med_row_idx"])) in target_pairs
    ]


def load_admission_data_from_examples(admission_record: Dict) -> Tuple[List[str], List[str], List[str], Dict[int, str]]:
    diag_example = admission_record.get("diagnosis_example")
    med_example = admission_record.get("medication_example")
    admission_id = admission_record.get("admission_id", "")

    assert diag_example is not None, f"Diagnosis rows not found for admission {admission_id}"
    assert med_example is not None, f"Medication rows not found for admission {admission_id}"

    diag_rows = [
        row["formatted"]
        for row in sorted(diag_example["tables"]["diagnosis"]["rows"], key=lambda item: item["row_idx"])
    ]
    med_rows = [
        row["formatted"]
        for row in sorted(med_example["tables"]["medication"]["rows"], key=lambda item: item["row_idx"])
    ]

    primary_pos = diag_example.get("primary_positive", diag_example)
    sentences = primary_pos.get("sentences", {}) or diag_example.get("sentences", {})
    sent_texts = [sentences[key]["text"] for key in sorted(sentences.keys(), key=int)]
    sent_meta = {int(key): sentences[key]["section_name"] for key in sentences}

    return diag_rows, med_rows, sent_texts, sent_meta


def load_ground_truth_for_admission(
    admission_id: str,
    annotation_entries: Dict[str, Dict],
    annotation_paths: Optional[List[Path]] = None,
    resolve_rel_inventory: bool = True,
) -> Tuple[List[Dict], Dict[int, List[int]], Dict[int, List[int]], set]:
    if admission_id not in annotation_entries:
        raise KeyError(f"Admission {admission_id} not found in annotation entries")

    entry = annotation_entries[admission_id]

    rg = entry["row_grounding"]
    gt_diag = {int(k) - 1: v["sentences"] for k, v in rg["diagnosis"].items()}
    gt_med = {int(k) - 1: v["sentences"] for k, v in rg["medication"].items()}

    gt_relationships: List[Dict] = []
    for rel in entry["relationships"]:
        gt_relationships.append({
            "diag_idx": rel["diagnosis_row"] - 1,
            "drug_idx": rel["drug_row"] - 1,
            "rel_type": _normalize_rel_type(rel["relationship_type"]),
            "evidence_sents": rel["evidence_sentences"],
        })

    multi_pairs: set = set()
    for flag in entry.get("multi_relationship_flags", []):
        d_idx = flag["diagnosis_row"] - 1
        m_idx = flag["drug_row"] - 1
        multi_pairs.add((d_idx, m_idx))
        for rel_type in flag["relationship_types"]:
            gt_relationships.append({
                "diag_idx": d_idx,
                "drug_idx": m_idx,
                "rel_type": _normalize_rel_type(rel_type),
                "evidence_sents": [],
            })

    if resolve_rel_inventory:
        _resolve_rel_types_from_annotation_corpus(annotation_paths=annotation_paths)
    return gt_relationships, gt_diag, gt_med, multi_pairs


def _sanitize_ground_truth_indices(
    gt_relationships: List[Dict],
    gt_diag: Dict[int, List[int]],
    gt_med: Dict[int, List[int]],
    multi_pairs: set,
    n_diag_rows: int,
    n_med_rows: int,
    n_sentences: int,
) -> Tuple[List[Dict], Dict[int, List[int]], Dict[int, List[int]], set, Dict[str, int]]:
    def _filter_sent_ids(sent_ids: List[int]) -> Tuple[List[int], int]:
        filtered = [
            int(sent_idx)
            for sent_idx in sent_ids
            if 0 <= int(sent_idx) < n_sentences
        ]
        return filtered, max(len(sent_ids) - len(filtered), 0)

    stats = {
        "dropped_relationships": 0,
        "dropped_multi_pairs": 0,
        "dropped_diag_rows": 0,
        "dropped_med_rows": 0,
        "dropped_relationship_sentence_refs": 0,
        "dropped_row_grounding_sentence_refs": 0,
    }

    sanitized_relationships: List[Dict] = []
    for rel in gt_relationships:
        diag_idx = int(rel["diag_idx"])
        med_idx = int(rel["drug_idx"])
        if not (0 <= diag_idx < n_diag_rows and 0 <= med_idx < n_med_rows):
            stats["dropped_relationships"] += 1
            continue
        filtered_evidence_sents, n_dropped_sent_ids = _filter_sent_ids(list(rel.get("evidence_sents", [])))
        stats["dropped_relationship_sentence_refs"] += n_dropped_sent_ids
        sanitized_rel = dict(rel)
        sanitized_rel["evidence_sents"] = filtered_evidence_sents
        sanitized_relationships.append(sanitized_rel)

    sanitized_gt_diag: Dict[int, List[int]] = {}
    for row_idx, sent_ids in gt_diag.items():
        if not (0 <= int(row_idx) < n_diag_rows):
            stats["dropped_diag_rows"] += 1
            continue
        filtered_sent_ids, n_dropped_sent_ids = _filter_sent_ids(list(sent_ids))
        stats["dropped_row_grounding_sentence_refs"] += n_dropped_sent_ids
        sanitized_gt_diag[int(row_idx)] = filtered_sent_ids

    sanitized_gt_med: Dict[int, List[int]] = {}
    for row_idx, sent_ids in gt_med.items():
        if not (0 <= int(row_idx) < n_med_rows):
            stats["dropped_med_rows"] += 1
            continue
        filtered_sent_ids, n_dropped_sent_ids = _filter_sent_ids(list(sent_ids))
        stats["dropped_row_grounding_sentence_refs"] += n_dropped_sent_ids
        sanitized_gt_med[int(row_idx)] = filtered_sent_ids

    sanitized_multi_pairs = {
        (int(diag_idx), int(med_idx))
        for diag_idx, med_idx in multi_pairs
        if 0 <= int(diag_idx) < n_diag_rows and 0 <= int(med_idx) < n_med_rows
    }
    stats["dropped_multi_pairs"] = len(multi_pairs) - len(sanitized_multi_pairs)

    return sanitized_relationships, sanitized_gt_diag, sanitized_gt_med, sanitized_multi_pairs, stats


# =============================================================================
# Phase B - Model reconstruction and loading
# =============================================================================

def load_model_args() -> Dict:
    with open(ARGS_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_model(args: Dict) -> BidirectionalTableTextModel:
    """Reconstruct BidirectionalTableTextModel from the training args."""
    print(f"\n  Loading encoder: {ENCODER_NAME}")
    sentence_encoder, resolved_encoder_name, encoder_source = _build_sentence_transformer(ENCODER_NAME)
    if encoder_source != "repo_id":
        print(f"  Using local snapshot: {resolved_encoder_name}")
    # sentence-transformers >=5 renamed get_sentence_embedding_dimension to
    # get_embedding_dimension; prefer the new name when available to silence
    # the FutureWarning while staying compatible with older installs.
    _dim_fn = getattr(
        sentence_encoder,
        "get_embedding_dimension",
        sentence_encoder.get_sentence_embedding_dimension,
    )
    native_dim = _dim_fn() or 1024
    override_dim = args.get("override_embedding_dim", 0)
    embedding_dim = override_dim if override_dim else native_dim

    # Match training: try Matryoshka truncation first; fall back to learned projection.
    uses_matryoshka = False
    if native_dim != embedding_dim:
        try:
            _test = sentence_encoder.encode(
                ["test"], truncate_dim=embedding_dim,
                convert_to_numpy=True, show_progress_bar=False,
            )
            if _test.shape[-1] == embedding_dim:
                uses_matryoshka = True
                sentence_encoder.truncate_dim = embedding_dim
        except Exception:
            pass

    dim_mode = "matryoshka" if uses_matryoshka else ("projected" if native_dim != embedding_dim else "exact")
    print(f"  Embedding dim: {embedding_dim} (native={native_dim}, mode={dim_mode})")

    model = BidirectionalTableTextModel(
        sentence_encoder       = sentence_encoder,
        embedding_dim          = embedding_dim,
        native_embedding_dim   = native_dim if (native_dim != embedding_dim and not uses_matryoshka) else None,
        trainable_encoder      = False,
        use_cross_attention_lora = args.get("use_cross_attention_lora", False),
        lora_rank              = args.get("lora_rank", 128),
        lora_alpha             = args.get("lora_alpha", 512),
        lora_dropout           = args.get("lora_dropout", 0.1),
        top_k                  = args.get("top_k", 5),
        pair_score_method      = args.get("pair_score_method", "cosine"),
        share_weights          = args.get("share_attention_weights", True),
        use_refinement         = args.get("use_refinement", False),
        use_self_attention     = args.get("use_self_attention", False),
        self_attention_heads   = args.get("self_attention_heads", 1),
        self_attention_dropout = args.get("self_attention_dropout", 0.1),
        init_method            = args.get("init_method", "zeros"),
        init_method_params     = args.get("init_method_params", {"bias_value": 0.0}),
        attention_type         = args.get("attention_type", "top_k_sparse"),
        sparse_top_k           = args.get("sparse_top_k", 5),
        window_size            = args.get("window_size", 5),
        threshold_base         = args.get("threshold_base", 0.3),
        norm_type              = args.get("norm_type", "rmsnorm"),
        use_qk_rmsnorm         = args.get("use_qk_rmsnorm", False),
        use_latent_bottleneck  = args.get("use_latent_bottleneck", False),
        latent_num             = args.get("latent_num", 64),
        latent_dropout         = args.get("latent_dropout", 0.0),
        use_gated_attention       = args.get("use_gated_attention", True),
        # NOTE: the trained checkpoint contains *both* outer gates
        # (bidirectional_attention.{forward,reverse}_output_gate) and inner gates
        # (bidirectional_attention.{forward,reverse}_attention.attention_output_gate).
        # Older training code coupled both to --use_gated_attention; the refactor
        # split the inner one onto its own --use_inner_gate flag, which is not
        # saved in args.json. Re-couple them here so inner-gate weights actually
        # load (otherwise sparse attention runs un-gated and collapses to hubs).
        use_inner_gate            = args.get("use_inner_gate", args.get("use_gated_attention", True)),
        gated_attention_mode      = args.get("gated_attention_mode", "vector"),
        gated_attention_hidden_dim = args.get("gated_attention_hidden_dim", 0),
        gated_attention_dropout   = args.get("gated_attention_dropout", 0.0),
        gated_attention_init_bias = args.get("gated_attention_init_bias", 6.0),
        use_header_conditioning   = args.get("use_header_conditioning", False),
        use_cell_level_matching   = args.get("use_cell_level_matching", False),
        cell_matching_weight      = args.get("cell_matching_weight", 0.35),
        cell_matching_pooling     = args.get("cell_matching_pooling", "max"),
        cell_row_fusion_weight    = args.get("cell_row_fusion_weight", 0.15),
        disable_temperature       = args.get("disable_temperature", True),
        verbose                   = False,
    )

    # Set attention activation attributes used dynamically in _apply_attention
    # (only relevant for standard attention; top_k_sparse reads these via getattr)
    model.bidirectional_attention.attention_activation = args.get("attention_activation", "softmax")
    model.bidirectional_attention.attention_alpha      = args.get("attention_alpha", 1.5)

    return model


def load_checkpoint(model: BidirectionalTableTextModel, ckpt_path: Path) -> None:
    """
    Load model.pt weights. Uses strict=False and reports any key mismatches
    so the user can see if cross-attention keys loaded correctly.
    """
    print(f"  Loading checkpoint: {ckpt_path}")
    # weights_only=False is required when the checkpoint may contain
    # non-tensor objects (e.g. Python scalars stored as state).
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Automatically remap auto_model prefix to model prefix to prevent noisy SentenceTransformer key mismatches
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("sentence_encoder.0.auto_model", "sentence_encoder.0.model")
        new_state_dict[new_k] = v
    state_dict = new_state_dict

    # -- Asymmetric gate-mode compatibility shim -----------------------------
    # The trained checkpoint may have been built with vector OUTER gates but
    # SCALAR INNER gates (output shape [1, D] instead of [D, D]). The current
    # models.py forwards a single `gated_attention_mode` to both, so build_model
    # may have created vector inner gates that don't match the checkpoint.
    # Detect any *.attention_output_gate.net.<i>.weight whose checkpoint shape
    # is [1, D] and rebuild the matching submodule's final Linear as (D -> 1).
    _patched_gates = []
    for ck_key, ck_tensor in state_dict.items():
        if not ck_key.endswith(".weight"):
            continue
        if "attention_output_gate.net." not in ck_key:
            continue
        try:
            current_param = model.get_parameter(ck_key)
        except AttributeError:
            continue
        if tuple(ck_tensor.shape) == tuple(current_param.shape):
            continue
        # Shape mismatch: replace the final Linear in this gate's `.net`.
        # Key format: "<module path>.net.<idx>.weight"
        gate_module_path, _, tail = ck_key.rpartition(".net.")
        linear_idx_str, _, _ = tail.partition(".")
        try:
            gate_module = model.get_submodule(gate_module_path)
            old_linear = gate_module.net[int(linear_idx_str)]
        except (AttributeError, ValueError, IndexError):
            continue
        if not isinstance(old_linear, torch.nn.Linear):
            continue
        ck_out, ck_in = int(ck_tensor.shape[0]), int(ck_tensor.shape[1])
        if (old_linear.in_features, old_linear.out_features) == (ck_in, ck_out):
            continue
        new_linear = torch.nn.Linear(
            ck_in, ck_out,
            bias=old_linear.bias is not None,
            device=old_linear.weight.device,
            dtype=old_linear.weight.dtype,
        )
        gate_module.net[int(linear_idx_str)] = new_linear
        if hasattr(gate_module, "mode") and ck_out == 1:
            gate_module.mode = "scalar"
        _patched_gates.append((gate_module_path, ck_out))
    if _patched_gates:
        print(
            f"  Gate-shape shim: rebuilt {len(_patched_gates)} inner gate layer(s) "
            f"to match checkpoint (scalar inner / vector outer asymmetry)."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    ca_loaded = sum(1 for k in state_dict if "bidirectional_attention" in k)
    enc_loaded = sum(1 for k in state_dict if "sentence_encoder" in k)
    print(f"  Keys loaded - cross-attention: {ca_loaded}, encoder: {enc_loaded}")

    if missing:
        print(f"  Missing  keys ({len(missing)}): {missing[:4]}"
              f"{'...' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:4]}"
              f"{'...' if len(unexpected) > 4 else ''}")
    if not missing and not unexpected:
        print("  Checkpoint loaded with no key mismatches.")


# =============================================================================
# Phase C - Joint encoding (Stage 5, Algorithm step 1)
# =============================================================================

def joint_encode(
    model: BidirectionalTableTextModel,
    diag_rows: List[str],
    med_rows: List[str],
    sent_texts: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Concatenate T_A U T_B and run a single LOKI bidirectional forward pass.

    Returns
    -------
    pair_scores       : FloatTensor [n_diag+n_med, n_sent]
    raw_rows          : FloatTensor [n_diag+n_med, d]   - encoder row embeddings before cross-attention
    raw_sentences     : FloatTensor [n_sent, d]         - encoder sentence embeddings before cross-attention
    refined_rows      : FloatTensor [n_diag+n_med, d]   - contextualized row embeddings after cross-attention
    refined_sentences : FloatTensor [n_sent, d]   - S̃_j for Stage 6 clustering
    forward_attn      : FloatTensor [n_diag+n_med, n_sent]
    """
    joint_rows = diag_rows + med_rows
    print(f"\n  Encoding {len(joint_rows)} rows ({len(diag_rows)} diag + {len(med_rows)} med) "
          f"and {len(sent_texts)} sentences...")

    with torch.no_grad():
        rows_emb = model.encode_sentences(joint_rows, batch_size=32).to(DEVICE)   # [N, d]
        sent_emb = model.encode_sentences(sent_texts, batch_size=32).to(DEVICE)   # [M, d]
        raw_rows = rows_emb.float()
        raw_sentences = sent_emb.float()

        # Attention layers are bfloat16; cast embeddings to match
        rows_emb = rows_emb.to(torch.bfloat16)
        sent_emb = sent_emb.to(torch.bfloat16)

        # Call BidirectionalCrossAttention directly to obtain refined_sentences (S̃)
        # model.forward() would discard refined_sentences; we need them for Stage 6
        result = model.bidirectional_attention(
            rows_emb.unsqueeze(0),  # [1, N, d]
            sent_emb.unsqueeze(0),  # [1, M, d]
        )
        pair_scores_b, refined_rows_b, refined_sents_b, fwd_attn_b, _rev_attn_b = result

    pair_scores       = pair_scores_b.squeeze(0).float()     # [N, M]
    refined_rows      = refined_rows_b.squeeze(0).float()    # [N, d]
    refined_sentences = refined_sents_b.squeeze(0).float()   # [M, d]
    forward_attn      = fwd_attn_b.squeeze(0).float()        # [N, M]

    print(f"  pair_scores:       {tuple(pair_scores.shape)}, "
          f"range [{float(pair_scores.min()):.3f}, {float(pair_scores.max()):.3f}]")
    print(f"  refined_rows:      {tuple(refined_rows.shape)}")
    print(f"  refined_sentences: {tuple(refined_sentences.shape)}")
    return pair_scores, raw_rows, raw_sentences, refined_rows, refined_sentences, forward_attn


# =============================================================================
# Phase D - Atomic link extraction + transitive join (Stage 5, steps 2-18)
# =============================================================================

def compute_threshold(
    pair_scores: torch.Tensor,
    floor: float = 0.15,
    adaptive_cap: Optional[float] = None,
    adaptive_gap_limit: Optional[float] = None,
    adaptive_force_legacy_max: Optional[float] = None,
) -> float:
    P = pair_scores.detach().cpu().numpy()
    # Calibrate on the top-1 sentence score per row rather than all (row x sent)
    # scores.  The full-matrix μ+2σ inflates σ because the majority of cells are
    # irrelevant (diag-row <-> unrelated sentence) and score near zero.  For hard
    # admissions those low scores dominate, driving gamma above what GT pairs
    # can achieve.  Using per-row maximums focuses the distribution on "the
    # strongest achievable link per row", which is what actually matters.
    top1_per_row = P.max(axis=1)          # [n_rows]  - best sentence per row
    mu_all    = float(P.mean())
    sigma_all = float(P.std())
    p75 = float(np.percentile(top1_per_row, 75))
    p50 = float(np.percentile(top1_per_row, 50))
    # Use the 75th-percentile of per-row top-1 scores as the primary signal;
    # fall back to max(floor, legacy μ+2σ) so gamma never exceeds what the
    # old formula would have produced (avoids regressions on easy admissions).
    legacy_gamma = max(mu_all + 2 * sigma_all, floor)
    gamma = max(min(p75, legacy_gamma), floor)
    adaptive_cap_applied = False
    adaptive_apply_reason = "gap_ok"
    adaptive_skip_reason = "p75_limited"
    adaptive_gap = p75 - legacy_gamma
    if adaptive_cap is not None:
        capped_gamma = min(gamma, max(float(adaptive_cap), floor))
        # Only cap admissions where the legacy μ+2σ branch is the active limiter.
        # When p75 is already the limiting term, pushing gamma down further tended
        # to hurt strong mimic_small cases without helping the threshold-heavy ones.
        if legacy_gamma <= p75:
            if adaptive_gap_limit is None or adaptive_gap <= float(adaptive_gap_limit):
                gamma = capped_gamma
                adaptive_cap_applied = True
            elif (
                adaptive_force_legacy_max is not None
                and legacy_gamma <= float(adaptive_force_legacy_max)
            ):
                gamma = capped_gamma
                adaptive_cap_applied = True
                adaptive_apply_reason = (
                    f"force_legacy({legacy_gamma:.3f}<={float(adaptive_force_legacy_max):.3f})"
                )
            else:
                adaptive_skip_reason = (
                    f"gap_limited({adaptive_gap:.3f}>{float(adaptive_gap_limit):.3f})"
                )
    print(
        f"\n  Adaptive gamma:"
        f"  p50_top1={p50:.3f}  p75_top1={p75:.3f}  legacy(mu+2s)={legacy_gamma:.3f}"
        f"  -> gamma=max(min(p75,legacy),floor)={gamma:.3f}"
        + (
            ""
            if adaptive_cap is None
            else (
                f"  (adaptive_cap={float(adaptive_cap):.3f}"
                + (
                    ""
                    if adaptive_gap_limit is None
                    else f", gap_limit={float(adaptive_gap_limit):.3f}"
                )
                + (
                    ""
                    if adaptive_force_legacy_max is None
                    else f", force_legacy_max={float(adaptive_force_legacy_max):.3f}"
                )
                + (
                    f", applied:{adaptive_apply_reason})"
                    if adaptive_cap_applied
                    else f", skipped:{adaptive_skip_reason})"
                )
            )
        )
    )
    return gamma


def _select_cutoff_band_items(
    ranked_items: List[Any],
    base_limit: int,
    score_getter: Callable[[Any], float],
    overflow_margin: float = 0.0,
    overflow_limit: int = 0,
    plateau_margin: float = 0.0,
    plateau_min_extra: int = 0,
    plateau_max_extra: int = 0,
) -> Tuple[List[Any], Dict[str, object]]:
    if not ranked_items:
        return [], {
            "base_limit": max(int(base_limit), 0),
            "candidate_count": 0,
            "cutoff_score": None,
            "selected_count": 0,
            "overflow_eligible_count": 0,
            "overflow_added_count": 0,
            "plateau_eligible_count": 0,
            "plateau_added_count": 0,
            "plateau_triggered": False,
        }

    resolved_limit = min(max(int(base_limit), 1), len(ranked_items))
    cutoff_score = float(score_getter(ranked_items[resolved_limit - 1]))
    selected_indices = set(range(resolved_limit))

    def _eligible_tail(margin: float) -> List[int]:
        if margin <= 0.0:
            return []
        eligible_indices: List[int] = []
        min_score = cutoff_score - float(margin)
        for idx in range(resolved_limit, len(ranked_items)):
            item_score = float(score_getter(ranked_items[idx]))
            if item_score < min_score:
                break
            eligible_indices.append(idx)
        return eligible_indices

    overflow_candidates = _eligible_tail(max(float(overflow_margin), 0.0))
    overflow_selected = set(overflow_candidates[:max(int(overflow_limit), 0)])
    selected_indices.update(overflow_selected)

    plateau_candidates = _eligible_tail(max(float(plateau_margin), 0.0))
    plateau_selected: set[int] = set()
    plateau_triggered = False
    if plateau_candidates and len(plateau_candidates) >= max(int(plateau_min_extra), 1):
        plateau_triggered = max(int(plateau_max_extra), 0) > 0
        for idx in plateau_candidates[:max(int(plateau_max_extra), 0)]:
            if idx not in selected_indices:
                selected_indices.add(idx)
                plateau_selected.add(idx)

    selected_items = [ranked_items[idx] for idx in sorted(selected_indices)]
    return selected_items, {
        "base_limit": resolved_limit,
        "candidate_count": len(ranked_items),
        "cutoff_score": cutoff_score,
        "selected_count": len(selected_items),
        "overflow_eligible_count": len(overflow_candidates),
        "overflow_added_count": len(overflow_selected),
        "plateau_eligible_count": len(plateau_candidates),
        "plateau_added_count": len(plateau_selected),
        "plateau_triggered": plateau_triggered,
    }


def _stage5_multi_sentence_threshold_rescue(
    candidate_paths_by_pair: Dict[Tuple[int, int], List[Dict]],
    threshold_rescue_min_sentences: int,
    max_sentences_per_pair: int,
) -> List[Dict]:
    if threshold_rescue_min_sentences <= 1 or not candidate_paths_by_pair:
        return []

    rescued_paths: List[Dict] = []
    for pair_paths in candidate_paths_by_pair.values():
        unique_sent_ids = {int(path["sent_idx"]) for path in pair_paths}
        if len(unique_sent_ids) < threshold_rescue_min_sentences:
            continue

        ranked_pair_paths = sorted(
            pair_paths,
            key=lambda item: (
                float(item.get("path_score", 0.0)),
                float(item.get("score_diag", 0.0)),
                float(item.get("score_med", 0.0)),
            ),
            reverse=True,
        )
        for path in ranked_pair_paths[:max_sentences_per_pair]:
            path["stage5_threshold_rescued"] = True
            rescued_paths.append(path)

    return rescued_paths


def _stage5_diag_row_sibling_threshold_rescue(
    candidate_paths_by_pair: Dict[Tuple[int, int], List[Dict]],
    admitted_paths: List[Dict],
    max_sentences_per_pair: int,
) -> List[Dict]:
    if not candidate_paths_by_pair or not admitted_paths:
        return []

    admitted_sent_ids_by_diag: Dict[int, Set[int]] = defaultdict(set)
    for path in admitted_paths:
        try:
            admitted_sent_ids_by_diag[int(path["diag_row_idx"])].add(int(path["sent_idx"]))
        except (KeyError, TypeError, ValueError):
            continue

    rescued_paths: List[Dict] = []
    for (diag_idx, _med_idx), pair_paths in candidate_paths_by_pair.items():
        candidate_sent_ids = {int(path.get("sent_idx", -1)) for path in pair_paths}
        if len(candidate_sent_ids) < 2:
            continue

        anchor_path = pair_paths[0] if pair_paths else {}
        medication_anchor = _extract_row_field(str(anchor_path.get("med_row_text", "")), "drug") or str(anchor_path.get("med_row_text", ""))
        if not medication_anchor:
            continue
        if not any(
            _text_matches_anchor(
                path.get("sent_text", ""),
                medication_anchor,
                normalization_mode="clinical_light",
            )
            for path in pair_paths
        ):
            continue

        support_sent_ids = admitted_sent_ids_by_diag.get(int(diag_idx), set())
        if not support_sent_ids:
            continue

        overlap_paths = [
            path for path in pair_paths
            if int(path.get("sent_idx", -1)) in support_sent_ids
            and (
                str(path.get("section_name", "")).strip().lower() != "medications on admission"
                or any(
                    re.search(pattern, str(path.get("sent_text", "")), flags=re.IGNORECASE)
                    for pattern in _EXPLICIT_DISCONTINUE_PATTERNS
                )
            )
        ]
        if not overlap_paths:
            continue

        ranked_pair_paths = sorted(
            overlap_paths,
            key=lambda item: (
                float(item.get("path_score", 0.0)),
                float(item.get("score_diag", 0.0)),
                float(item.get("score_med", 0.0)),
            ),
            reverse=True,
        )
        support_overlap_sent_ids = sorted({int(path["sent_idx"]) for path in overlap_paths})
        for path in ranked_pair_paths[:max_sentences_per_pair]:
            path["stage5_threshold_rescued"] = True
            path["stage5_diag_row_sibling_threshold_rescued"] = True
            path["stage5_diag_row_sibling_support_sent_ids"] = support_overlap_sent_ids
            rescued_paths.append(path)

    return rescued_paths


def _stage5_med_row_stopcue_threshold_rescue(
    candidate_paths_by_pair: Dict[Tuple[int, int], List[Dict]],
    admitted_paths: List[Dict],
    max_sentences_per_pair: int,
) -> List[Dict]:
    if not candidate_paths_by_pair or not admitted_paths:
        return []

    admitted_sent_ids_by_med: Dict[int, Set[int]] = defaultdict(set)
    for path in admitted_paths:
        try:
            admitted_sent_ids_by_med[int(path["med_row_idx"])].add(int(path["sent_idx"]))
        except (KeyError, TypeError, ValueError):
            continue

    rescued_paths: List[Dict] = []
    for (_diag_idx, med_idx), pair_paths in candidate_paths_by_pair.items():
        candidate_sent_ids = {int(path.get("sent_idx", -1)) for path in pair_paths}
        if not candidate_sent_ids:
            continue

        anchor_path = pair_paths[0] if pair_paths else {}
        medication_anchor = _extract_row_field(str(anchor_path.get("med_row_text", "")), "drug") or str(anchor_path.get("med_row_text", ""))
        if not medication_anchor:
            continue

        stopcue_anchor_paths = [
            path for path in pair_paths
            if _text_matches_anchor(
                path.get("sent_text", ""),
                medication_anchor,
                normalization_mode="clinical_light",
            )
            and str(path.get("section_name", "")).strip().lower() == "medications on admission"
            and any(
                re.search(pattern, str(path.get("sent_text", "")), flags=re.IGNORECASE)
                for pattern in _EXPLICIT_DISCONTINUE_PATTERNS
            )
        ]
        if not stopcue_anchor_paths:
            continue

        support_sent_ids = admitted_sent_ids_by_med.get(int(med_idx), set())
        if not support_sent_ids:
            continue

        overlap_paths = [
            path for path in stopcue_anchor_paths
            if int(path.get("sent_idx", -1)) in support_sent_ids
        ]
        if not overlap_paths:
            continue

        ranked_pair_paths = sorted(
            overlap_paths,
            key=lambda item: (
                float(item.get("path_score", 0.0)),
                float(item.get("score_diag", 0.0)),
                float(item.get("score_med", 0.0)),
            ),
            reverse=True,
        )
        support_overlap_sent_ids = sorted({int(path["sent_idx"]) for path in overlap_paths})
        for path in ranked_pair_paths[:max_sentences_per_pair]:
            path["stage5_threshold_rescued"] = True
            path["stage5_med_row_stopcue_threshold_rescued"] = True
            path["stage5_med_row_stopcue_support_sent_ids"] = support_overlap_sent_ids
            rescued_paths.append(path)

    return rescued_paths


def _stage5_diag_stopcue_row_rescue(
    ranked_candidates: List[Tuple[float, float, int]],
    selected_candidates: List[Tuple[float, float, int]],
    stopcue_sentence_ids: Optional[Set[int]],
    link_floor: float,
    max_extra: int = 1,
) -> Tuple[List[Tuple[float, float, int]], int]:
    resolved_stopcue_sentence_ids = stopcue_sentence_ids or set()
    resolved_max_extra = max(int(max_extra), 0)
    if not resolved_stopcue_sentence_ids or resolved_max_extra <= 0 or not ranked_candidates:
        return selected_candidates, 0

    selected_sent_ids = {int(sent_idx) for _rank_score, _raw_score, sent_idx in selected_candidates}
    if selected_sent_ids & resolved_stopcue_sentence_ids:
        return selected_candidates, 0

    rescued_candidates: List[Tuple[float, float, int]] = []
    min_raw_score = max(float(link_floor), 0.20)
    for rank_score, raw_score, sent_idx in ranked_candidates:
        if int(sent_idx) in selected_sent_ids or int(sent_idx) not in resolved_stopcue_sentence_ids:
            continue
        if float(raw_score) < min_raw_score:
            continue
        rescued_candidates.append((float(rank_score), float(raw_score), int(sent_idx)))
        if len(rescued_candidates) >= resolved_max_extra:
            break

    if not rescued_candidates:
        return selected_candidates, 0

    return sorted(selected_candidates + rescued_candidates, reverse=True), len(rescued_candidates)


def _stage5_stopcue_single_sentence_threshold_rescue(
    candidate_paths_by_pair: Dict[Tuple[int, int], List[Dict]],
    stopcue_sentence_ids: Optional[Set[int]],
    max_sentences_per_pair: int,
) -> List[Dict]:
    resolved_stopcue_sentence_ids = stopcue_sentence_ids or set()
    if not resolved_stopcue_sentence_ids or not candidate_paths_by_pair:
        return []

    rescued_paths: List[Dict] = []
    for pair_paths in candidate_paths_by_pair.values():
        stopcue_paths = [
            path for path in pair_paths
            if int(path.get("sent_idx", -1)) in resolved_stopcue_sentence_ids
        ]
        if not stopcue_paths:
            continue

        ranked_pair_paths = sorted(
            stopcue_paths,
            key=lambda item: (
                float(item.get("path_score", 0.0)),
                float(item.get("score_diag", 0.0)),
                float(item.get("score_med", 0.0)),
            ),
            reverse=True,
        )
        for path in ranked_pair_paths[:max_sentences_per_pair]:
            path["stage5_threshold_rescued"] = True
            path["stage5_stopcue_threshold_rescued"] = True
            rescued_paths.append(path)

    return rescued_paths


def extract_cross_table_join_paths(
    pair_scores:  torch.Tensor,
    n_diag:       int,
    diag_rows:    List[str],
    med_rows:     List[str],
    sent_texts:   List[str],
    sent_meta:    Dict[int, str],
    gamma:        float,
    top_k:        int = 5,
    diag_row_top_k: Optional[int] = None,
    med_row_top_k: Optional[int] = None,
    row_plateau_margin: float = 0.0,
    row_plateau_min_extra: int = 0,
    row_plateau_max_extra: int = 0,
    sent_diag_top_k: int = 3,
    sent_med_top_k: int = 4,
    max_pairs_per_sentence: int = 8,
    max_sentences_per_pair: int = 2,
    sentence_specificity_alpha: float = 0.0,
    section_priors: Optional[Dict[str, float]] = None,
    sentence_overflow_margin: float = 0.0,
    sentence_overflow_limit: int = 0,
    sentence_plateau_margin: float = 0.0,
    sentence_plateau_min_extra: int = 0,
    sentence_plateau_max_extra: int = 0,
    stopcue_diag_sentence_top_k: int = 0,
    pair_plateau_margin: float = 0.0,
    pair_plateau_min_extra: int = 0,
    pair_plateau_max_extra: int = 0,
    threshold_rescue_margin: float = 0.0,
    threshold_rescue_min_sentences: int = 2,
    diag_row_sibling_rescue_margin: float = 0.0,
    med_row_stopcue_rescue_margin: float = 0.0,
) -> List[Dict]:
    """
    Extract transitive (diagnosis_row, sentence, medication_row) join paths.

     Algorithm:
      1. Build atomic link sets J_A and J_B using a low floor with row-side top_k.
          When enabled, sentence-specificity and section priors affect how
          atomic row-sentence links are ranked before the top-k gates.
         Optionally, expand the per-row sentence shortlist when the row-side
         cutoff sits on a plateau of near-tied sentence scores.
     2. Apply a sentence-side top-k gate separately for diagnosis rows and
         medication rows. This approximates the proposal's mutual top-k filter and
         suppresses sentence hubs before the transitive cross-product.
        Optionally, keep a bounded overflow band or a larger plateau band of
        near-tied rows per sentence.
     3. Transitive join on shared sentences.
     4. Filter triples where path_score (average of both sides) >= gamma.
     5. Cap the number of (diag, med) pairs contributed by one sentence and the
         number of mediating sentences retained for one (diag, med) pair.
        Optionally, keep a plateau band of near-tied sentence-local row pairs.
          Optionally, rescue repeated near-threshold pairs across multiple
          mediator sentences instead of lowering gamma globally.
     6. Sort by path_score descending.

    Returns a list of path dicts (all indices 0-based).
    """
    P = _to_numpy_array(pair_scores, dtype=np.float32)
    # Link floor: each individual side must score at least this to contribute to
    # a join path.  Previously gamma/√2, which compounded the adaptive-gamma problem:
    # when gamma rises for hard admissions, the floor also rises, killing most GT
    # candidates before they can even reach the gamma check.  We now use a gentler
    # formula - max(gamma * 0.5, 0.15) - so a single strong side (e.g. score 0.55)
    # can partially compensate for a weaker side (0.30) and still average to gamma.
    LINK_FLOOR = max(gamma * 0.5, 0.15)

    diag_row_top_k = max(1, int(diag_row_top_k if diag_row_top_k is not None else top_k))
    med_row_top_k = max(1, int(med_row_top_k if med_row_top_k is not None else top_k))
    row_plateau_margin = max(float(row_plateau_margin), 0.0)
    row_plateau_min_extra = max(int(row_plateau_min_extra), 0)
    row_plateau_max_extra = max(int(row_plateau_max_extra), 0)
    sent_diag_top_k = max(1, min(sent_diag_top_k, diag_row_top_k))
    sent_med_top_k = max(1, min(sent_med_top_k, med_row_top_k))
    weighting_active = float(sentence_specificity_alpha) > 0.0 or bool(section_priors)
    overflow_margin = max(float(sentence_overflow_margin), 0.0)
    overflow_limit = max(int(sentence_overflow_limit), 0)
    sentence_plateau_margin = max(float(sentence_plateau_margin), 0.0)
    sentence_plateau_min_extra = max(int(sentence_plateau_min_extra), 0)
    sentence_plateau_max_extra = max(int(sentence_plateau_max_extra), 0)
    stopcue_diag_sentence_top_k = max(int(stopcue_diag_sentence_top_k), 0)
    pair_plateau_margin = max(float(pair_plateau_margin), 0.0)
    pair_plateau_min_extra = max(int(pair_plateau_min_extra), 0)
    pair_plateau_max_extra = max(int(pair_plateau_max_extra), 0)
    threshold_rescue_margin = max(float(threshold_rescue_margin), 0.0)
    threshold_rescue_min_sentences = max(int(threshold_rescue_min_sentences), 2)
    diag_row_sibling_rescue_margin = max(float(diag_row_sibling_rescue_margin), 0.0)
    med_row_stopcue_rescue_margin = max(float(med_row_stopcue_rescue_margin), 0.0)
    stopcue_sentence_ids = {
        sent_idx
        for sent_idx, sent_text in enumerate(sent_texts)
        if any(re.search(pattern, str(sent_text), flags=re.IGNORECASE) for pattern in _EXPLICIT_DISCONTINUE_PATTERNS)
    }
    stopcue_boost = 1.25 if stopcue_sentence_ids else 1.0
    stopcue_threshold_rescue_margin = 0.005 if stopcue_sentence_ids else 0.0
    stopcue_pair_cap = max(max_pairs_per_sentence, 36)
    if stopcue_diag_sentence_top_k > sent_diag_top_k:
        stopcue_pair_cap = max(stopcue_pair_cap, 3 * stopcue_diag_sentence_top_k)
    diag_stopcue_row_rescue_stats = {
        "rows_touched": 0,
        "candidates_added": 0,
    }

    def _build_atomic_links(
        row_offset: int,
        row_count: int,
        row_top_k: int,
        sentence_top_k: int,
        stopcue_sentence_top_k: int = 0,
    ) -> Dict[int, List[Tuple[int, float]]]:
        row_floor_candidates: Dict[int, List[Tuple[int, float]]] = {}
        sentence_support_counts: Dict[int, int] = defaultdict(int)

        for local_idx in range(row_count):
            row_idx = row_offset + local_idx
            floor_candidates: List[Tuple[int, float]] = []
            for sent_idx in range(P.shape[1]):
                raw_score = float(P[row_idx, sent_idx])
                if raw_score < LINK_FLOOR:
                    continue
                floor_candidates.append((sent_idx, raw_score))
                sentence_support_counts[sent_idx] += 1
            row_floor_candidates[local_idx] = floor_candidates

        sentence_rank_weights = {
            sent_idx: _stage5_sentence_rank_weight(
                sent_idx,
                sentence_support=sentence_support_counts[sent_idx],
                sent_meta=sent_meta,
                sentence_specificity_alpha=sentence_specificity_alpha,
                section_priors=section_priors,
                stopcue_sentence_ids=stopcue_sentence_ids,
                stopcue_boost=stopcue_boost,
            )
            for sent_idx in sentence_support_counts
        }

        row_top_links: Dict[int, Dict[int, float]] = {}
        sent_candidates: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

        for local_idx, floor_candidates in row_floor_candidates.items():
            ranked_candidates = [
                (
                    float(raw_score) * float(sentence_rank_weights.get(sent_idx, 1.0)),
                    float(raw_score),
                    int(sent_idx),
                )
                for sent_idx, raw_score in floor_candidates
            ]
            ranked_candidates.sort(reverse=True)
            selected_candidates, _selection_stats = _select_cutoff_band_items(
                ranked_candidates,
                base_limit=row_top_k,
                score_getter=lambda item: float(item[0]),
                plateau_margin=row_plateau_margin,
                plateau_min_extra=row_plateau_min_extra,
                plateau_max_extra=row_plateau_max_extra,
            )
            if row_offset == 0:
                selected_candidates, rescued_count = _stage5_diag_stopcue_row_rescue(
                    ranked_candidates,
                    selected_candidates,
                    stopcue_sentence_ids=stopcue_sentence_ids,
                    link_floor=LINK_FLOOR,
                    max_extra=1,
                )
                if rescued_count > 0:
                    diag_stopcue_row_rescue_stats["rows_touched"] += 1
                    diag_stopcue_row_rescue_stats["candidates_added"] += rescued_count
            row_top_links[local_idx] = {
                sent_idx: raw_score
                for _rank_score, raw_score, sent_idx in selected_candidates
            }
            for rank_score, raw_score, sent_idx in selected_candidates:
                sent_candidates[sent_idx].append((local_idx, raw_score, rank_score))

        links_by_sent: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for sent_idx, candidates in sent_candidates.items():
            candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
            effective_sentence_top_k = sentence_top_k
            if (
                row_offset == 0
                and stopcue_sentence_top_k > effective_sentence_top_k
                and sent_idx in stopcue_sentence_ids
            ):
                effective_sentence_top_k = stopcue_sentence_top_k
            selected_candidates, _selection_stats = _select_cutoff_band_items(
                candidates,
                base_limit=effective_sentence_top_k,
                score_getter=lambda item: float(item[1]),
                overflow_margin=overflow_margin,
                overflow_limit=overflow_limit,
                plateau_margin=sentence_plateau_margin,
                plateau_min_extra=sentence_plateau_min_extra,
                plateau_max_extra=sentence_plateau_max_extra,
            )

            for local_idx, score, _rank_score in selected_candidates:
                if sent_idx in row_top_links.get(local_idx, {}):
                    links_by_sent[sent_idx].append((local_idx, score))

        return links_by_sent

    # --- Step 1: Build atomic link sets with row-side and sentence-side gates ---
    A_by_sent = _build_atomic_links(
        0,
        n_diag,
        diag_row_top_k,
        sent_diag_top_k,
        stopcue_sentence_top_k=stopcue_diag_sentence_top_k,
    )
    B_by_sent = _build_atomic_links(n_diag, len(med_rows), med_row_top_k, sent_med_top_k)

    shared_sents = set(A_by_sent.keys()) & set(B_by_sent.keys())
    n_J_A = sum(len(v) for v in A_by_sent.values())
    n_J_B = sum(len(v) for v in B_by_sent.values())
    print(
        f"  Atomic links after mutual-style gating: J_A={n_J_A}, J_B={n_J_B}, "
        f"shared sentences={len(shared_sents)}"
    )
    print(
        f"  Row-side top-k: diag<= {diag_row_top_k}, med<= {med_row_top_k}"
    )
    if row_plateau_max_extra > 0 and row_plateau_margin > 0.0:
        print(
            f"  Stage 5 row plateau: margin={row_plateau_margin:.4f}, "
            f"min_extra={row_plateau_min_extra}, max_extra={row_plateau_max_extra}"
        )
    print(
        f"  Sentence caps: diag<= {sent_diag_top_k}, med<= {sent_med_top_k}, "
        f"pairs/sentence<= {max_pairs_per_sentence}, evidence/pair<= {max_sentences_per_pair}"
    )
    if weighting_active:
        print(
            f"  Stage 5 ranking weights: sentence_specificity_alpha={float(sentence_specificity_alpha):.3f}, "
            f"section_priors={len(section_priors or {})}"
        )
    if stopcue_sentence_ids:
        print(
            f"  Stage 5 stop-cue sentence boost: {len(stopcue_sentence_ids)} sentences x{stopcue_boost:.2f}"
        )
        if stopcue_diag_sentence_top_k > sent_diag_top_k:
            print(
                f"  Stage 5 stop-cue diag sentence cap: expanding diagnosis-side sentence cap to {stopcue_diag_sentence_top_k}"
            )
        if stopcue_pair_cap > max_pairs_per_sentence:
            print(
                f"  Stage 5 stop-cue pair cap: expanding explicit discontinue sentences to {stopcue_pair_cap} pairs"
            )
        if diag_stopcue_row_rescue_stats["candidates_added"] > 0:
            print(
                f"  Stage 5 stop-cue diag-row rescue: added {diag_stopcue_row_rescue_stats['candidates_added']} "
                f"sentence links across {diag_stopcue_row_rescue_stats['rows_touched']} diagnosis rows"
            )
    if overflow_limit > 0 and overflow_margin > 0.0:
        print(
            f"  Stage 5 sentence overflow: limit={overflow_limit}, margin={overflow_margin:.4f}"
        )
    if sentence_plateau_max_extra > 0 and sentence_plateau_margin > 0.0:
        print(
            f"  Stage 5 sentence plateau: margin={sentence_plateau_margin:.4f}, "
            f"min_extra={sentence_plateau_min_extra}, max_extra={sentence_plateau_max_extra}"
        )
    if pair_plateau_max_extra > 0 and pair_plateau_margin > 0.0:
        print(
            f"  Stage 5 pair plateau: margin={pair_plateau_margin:.4f}, "
            f"min_extra={pair_plateau_min_extra}, max_extra={pair_plateau_max_extra}"
        )
    if threshold_rescue_margin > 0.0:
        print(
            f"  Stage 5 threshold rescue: margin={threshold_rescue_margin:.4f}, "
            f"min_sentences={threshold_rescue_min_sentences}"
        )
    elif stopcue_threshold_rescue_margin > 0.0:
        print(
            f"  Stage 5 stop-cue threshold rescue: margin={stopcue_threshold_rescue_margin:.4f}, single-sentence enabled"
        )
    if med_row_stopcue_rescue_margin > 0.0:
        print(
            f"  Stage 5 med-row stop-cue rescue: margin={med_row_stopcue_rescue_margin:.4f}, multi-sentence anchored explicit-stop overlap"
        )

    # --- Steps 2-4: Transitive join, filter on path_score >= gamma ---
    # Key: we gate on the AVERAGE score (join confidence), not each side
    # independently.  A strong med->sent link can compensate for a weaker
    # diag->sent link as long as the join path is overall confident.
    seen_triples: set = set()
    all_paths: List[Dict] = []
    near_threshold_candidates_by_pair: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    diag_row_sibling_candidates_by_pair: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    med_row_stopcue_candidates_by_pair: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)

    for j in shared_sents:
        sent_paths: List[Dict] = []
        sent_near_threshold_paths: List[Dict] = []
        for i_A, score_A in A_by_sent[j]:
            for i_B, score_B in B_by_sent[j]:
                triple_key = (i_A, j, i_B)
                if triple_key in seen_triples:
                    continue
                seen_triples.add(triple_key)
                path_score = (float(score_A) + float(score_B)) / 2.0
                path_record = {
                    "diag_row_idx"  : i_A,
                    "med_row_idx"   : i_B,
                    "sent_idx"      : j,
                    "score_diag"    : round(float(score_A), 4),
                    "score_med"     : round(float(score_B), 4),
                    "path_score"    : round(path_score, 4),
                    "diag_row_text" : diag_rows[i_A],
                    "med_row_text"  : med_rows[i_B],
                    "sent_text"     : sent_texts[j],
                    "section_name"  : sent_meta.get(j, ""),
                    "cluster_id"    : -1,
                    "relationship"  : "",
                }
                if path_score >= gamma:
                    sent_paths.append(path_record)
                else:
                    if threshold_rescue_margin > 0.0 and path_score >= (gamma - threshold_rescue_margin):
                        sent_near_threshold_paths.append(path_record)
                    elif (
                        stopcue_threshold_rescue_margin > 0.0
                        and j in stopcue_sentence_ids
                        and path_score >= (gamma - stopcue_threshold_rescue_margin)
                    ):
                        path_record["stage5_stopcue_near_threshold"] = True
                        sent_near_threshold_paths.append(path_record)
                    if (
                        diag_row_sibling_rescue_margin > 0.0
                        and path_score >= (gamma - diag_row_sibling_rescue_margin)
                    ):
                        diag_row_sibling_candidates_by_pair[(path_record["diag_row_idx"], path_record["med_row_idx"])].append(path_record)
                    if (
                        med_row_stopcue_rescue_margin > 0.0
                        and path_score >= (gamma - med_row_stopcue_rescue_margin)
                    ):
                        med_row_stopcue_candidates_by_pair[(path_record["diag_row_idx"], path_record["med_row_idx"])].append(path_record)

        sent_paths.sort(key=lambda item: item["path_score"], reverse=True)
        effective_max_pairs_per_sentence = max_pairs_per_sentence if j not in stopcue_sentence_ids else stopcue_pair_cap
        selected_sent_paths, _selection_stats = _select_cutoff_band_items(
            sent_paths,
            base_limit=effective_max_pairs_per_sentence,
            score_getter=lambda item: float(item["path_score"]),
            plateau_margin=pair_plateau_margin,
            plateau_min_extra=pair_plateau_min_extra,
            plateau_max_extra=pair_plateau_max_extra,
        )
        all_paths.extend(selected_sent_paths)

        if sent_near_threshold_paths:
            sent_near_threshold_paths.sort(key=lambda item: item["path_score"], reverse=True)
            selected_near_threshold_paths, _selection_stats = _select_cutoff_band_items(
                sent_near_threshold_paths,
                base_limit=stopcue_pair_cap if j in stopcue_sentence_ids else effective_max_pairs_per_sentence,
                score_getter=lambda item: float(item["path_score"]),
                plateau_margin=pair_plateau_margin,
                plateau_min_extra=pair_plateau_min_extra,
                plateau_max_extra=pair_plateau_max_extra,
            )
            for path in selected_near_threshold_paths:
                near_threshold_candidates_by_pair[(path["diag_row_idx"], path["med_row_idx"])].append(path)

    if threshold_rescue_margin > 0.0 and near_threshold_candidates_by_pair:
        existing_pairs = {
            (path["diag_row_idx"], path["med_row_idx"])
            for path in all_paths
        }
        rescued_paths = _stage5_multi_sentence_threshold_rescue(
            {
                pair: pair_paths
                for pair, pair_paths in near_threshold_candidates_by_pair.items()
                if pair not in existing_pairs
            },
            threshold_rescue_min_sentences=threshold_rescue_min_sentences,
            max_sentences_per_pair=max_sentences_per_pair,
        )
        if rescued_paths:
            rescued_pairs = len({(path["diag_row_idx"], path["med_row_idx"]) for path in rescued_paths})
            print(
                f"  Stage 5 threshold rescue admitted {rescued_pairs} near-threshold pairs "
                f"({len(rescued_paths)} paths)"
            )
            all_paths.extend(rescued_paths)

    if stopcue_sentence_ids and near_threshold_candidates_by_pair:
        existing_pairs = {
            (path["diag_row_idx"], path["med_row_idx"])
            for path in all_paths
        }
        rescued_paths = _stage5_stopcue_single_sentence_threshold_rescue(
            {
                pair: pair_paths
                for pair, pair_paths in near_threshold_candidates_by_pair.items()
                if pair not in existing_pairs
            },
            stopcue_sentence_ids=stopcue_sentence_ids,
            max_sentences_per_pair=max_sentences_per_pair,
        )
        if rescued_paths:
            rescued_pairs = len({(path["diag_row_idx"], path["med_row_idx"]) for path in rescued_paths})
            print(
                f"  Stage 5 stop-cue threshold rescue admitted {rescued_pairs} near-threshold pairs "
                f"({len(rescued_paths)} paths)"
            )
            all_paths.extend(rescued_paths)

    if diag_row_sibling_rescue_margin > 0.0 and diag_row_sibling_candidates_by_pair:
        existing_pairs = {
            (path["diag_row_idx"], path["med_row_idx"])
            for path in all_paths
        }
        rescued_paths = _stage5_diag_row_sibling_threshold_rescue(
            {
                pair: pair_paths
                for pair, pair_paths in diag_row_sibling_candidates_by_pair.items()
                if pair not in existing_pairs
            },
            admitted_paths=all_paths,
            max_sentences_per_pair=max_sentences_per_pair,
        )
        if rescued_paths:
            rescued_pairs = len({(path["diag_row_idx"], path["med_row_idx"]) for path in rescued_paths})
            print(
                f"  Stage 5 diag-row sibling rescue admitted {rescued_pairs} near-threshold pairs "
                f"({len(rescued_paths)} paths)"
            )
            all_paths.extend(rescued_paths)

    if med_row_stopcue_rescue_margin > 0.0 and med_row_stopcue_candidates_by_pair:
        existing_pairs = {
            (path["diag_row_idx"], path["med_row_idx"])
            for path in all_paths
        }
        rescued_paths = _stage5_med_row_stopcue_threshold_rescue(
            {
                pair: pair_paths
                for pair, pair_paths in med_row_stopcue_candidates_by_pair.items()
                if pair not in existing_pairs
            },
            admitted_paths=all_paths,
            max_sentences_per_pair=max_sentences_per_pair,
        )
        if rescued_paths:
            rescued_pairs = len({(path["diag_row_idx"], path["med_row_idx"]) for path in rescued_paths})
            print(
                f"  Stage 5 med-row stop-cue rescue admitted {rescued_pairs} near-threshold pairs "
                f"({len(rescued_paths)} paths)"
            )
            all_paths.extend(rescued_paths)

    # --- Step 5: Limit mediating evidence per predicted (diag, med) pair ---
    pair_buckets: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for path in all_paths:
        pair_buckets[(path["diag_row_idx"], path["med_row_idx"])].append(path)

    limited_paths: List[Dict] = []
    for pair_paths in pair_buckets.values():
        pair_paths.sort(key=lambda item: (item["path_score"], item["score_diag"], item["score_med"]), reverse=True)
        limited_paths.extend(pair_paths[:max_sentences_per_pair])

    # --- Step 6: Sort ---
    result = sorted(limited_paths, key=lambda x: x["path_score"], reverse=True)
    n_unique_pairs = len({(p["diag_row_idx"], p["med_row_idx"]) for p in result})
    print(f"  Discovered {len(result)} (diag, sent, med) triples "
          f"covering {n_unique_pairs} unique (diag, med) pairs")

    if result:
        print("\n  Top-5 paths:")
        for p in result[:5]:
            print(f"    diag[{p['diag_row_idx']+1:2d}] x med[{p['med_row_idx']+1:2d}]  "
                  f"path_score={p['path_score']:.4f}  via sent[{p['sent_idx']}]  "
                  f"({p['section_name']})")
            print(f"    -> {p['sent_text'][:100]}...")

    return result


def _l2_normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return array / norms


def _bucket_paths_by_pair(paths: List[Dict]) -> Dict[Tuple[int, int], List[Dict]]:
    pair_buckets: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for path in paths:
        pair_buckets[(path["diag_row_idx"], path["med_row_idx"])].append(path)
    return pair_buckets


def _pair_recovery_stage_snapshot(
    paths: List[Dict],
    labels: Optional[np.ndarray] = None,
    cluster_name_map: Optional[Dict[int, str]] = None,
    cluster_label_details: Optional[Dict[int, Dict[str, object]]] = None,
) -> Dict[str, object]:
    pair_buckets = _bucket_paths_by_pair(paths)
    sent_pair_members: Dict[int, set] = defaultdict(set)
    for pair, pair_paths in pair_buckets.items():
        for sent_idx in {int(path["sent_idx"]) for path in pair_paths}:
            sent_pair_members[sent_idx].add(pair)

    pair_cluster_votes: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    if labels is not None and len(labels) == len(paths):
        for path, lbl in zip(paths, labels):
            pair_cluster_votes[(int(path["diag_row_idx"]), int(path["med_row_idx"]))][int(lbl)] += 1

    pair_records: List[Dict[str, object]] = []
    for (diag_idx, med_idx), pair_paths in sorted(pair_buckets.items()):
        unique_sent_ids = sorted({int(path["sent_idx"]) for path in pair_paths})
        unique_sections = sorted({
            str(path.get("section_name", "")).strip()
            for path in pair_paths
            if str(path.get("section_name", "")).strip()
        })
        best_path = max(
            pair_paths,
            key=lambda path: (
                float(path.get("path_score", 0.0)),
                float(path.get("score_diag", 0.0)),
                float(path.get("score_med", 0.0)),
            ),
        )
        max_sentence_fanout = max(
            (len(sent_pair_members[sent_idx]) for sent_idx in unique_sent_ids),
            default=0,
        )
        pair_record = {
            "diag_row_idx": diag_idx,
            "med_row_idx": med_idx,
            "n_paths": len(pair_paths),
            "n_unique_sentences": len(unique_sent_ids),
            "n_unique_sections": len(unique_sections),
            "unique_sent_ids": unique_sent_ids,
            "unique_sections": unique_sections,
            "best_path_score": round(float(best_path.get("path_score", 0.0)), 4),
            "best_score_diag": round(float(best_path.get("score_diag", 0.0)), 4),
            "best_score_med": round(float(best_path.get("score_med", 0.0)), 4),
            "max_sentence_fanout": max_sentence_fanout,
        }
        cluster_votes = pair_cluster_votes.get((diag_idx, med_idx), {})
        if cluster_votes:
            cluster_id = max(cluster_votes.items(), key=lambda item: (item[1], -item[0]))[0]
            pair_record["cluster_id"] = cluster_id
            cluster_label = _normalize_rel_type(str((cluster_name_map or {}).get(int(cluster_id), "")))
            if cluster_label:
                pair_record["cluster_label"] = cluster_label
            cluster_detail = (cluster_label_details or {}).get(int(cluster_id), {}) or {}
            if cluster_detail:
                backend_name = str(cluster_detail.get("backend", "")).strip()
                if backend_name:
                    pair_record["cluster_backend"] = backend_name
                label_source = str(cluster_detail.get("label_source", "")).strip()
                if label_source:
                    pair_record["cluster_label_source"] = label_source
                refinement_parent_cluster_id = cluster_detail.get("refinement_parent_cluster_id")
                if refinement_parent_cluster_id is not None:
                    try:
                        pair_record["refinement_parent_cluster_id"] = int(refinement_parent_cluster_id)
                    except (TypeError, ValueError):
                        pass
                split_mode = str(cluster_detail.get("pair_label_refinement_split_mode", "")).strip().lower()
                if split_mode:
                    pair_record["pair_label_refinement_split_mode"] = split_mode
                low_signal_rescue_mode = str(cluster_detail.get("low_signal_rescue_mode", "")).strip().lower()
                if low_signal_rescue_mode:
                    pair_record["low_signal_rescue_mode"] = low_signal_rescue_mode
        pair_records.append(pair_record)

    return {
        "n_pairs": len(pair_buckets),
        "n_paths": len(paths),
        "pairs": pair_records,
    }


def _pair_lookup_from_stage_snapshot(stage_snapshot: object) -> Dict[Tuple[int, int], Dict[str, object]]:
    if not isinstance(stage_snapshot, dict):
        return {}

    pair_records = stage_snapshot.get("pairs")
    if not isinstance(pair_records, list):
        return {}

    lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
    for pair_record in pair_records:
        if not isinstance(pair_record, dict):
            continue
        try:
            pair = (int(pair_record["diag_row_idx"]), int(pair_record["med_row_idx"]))
        except (KeyError, TypeError, ValueError):
            continue
        lookup[pair] = pair_record
    return lookup


def _pair_decision_lookup(decision_records: object) -> Dict[Tuple[int, int], Dict[str, object]]:
    if not isinstance(decision_records, list):
        return {}

    lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
    for decision in decision_records:
        if not isinstance(decision, dict):
            continue
        try:
            pair = (int(decision["diag_row_idx"]), int(decision["med_row_idx"]))
        except (KeyError, TypeError, ValueError):
            continue
        lookup[pair] = decision
    return lookup


def _stage5_atomic_link_diagnostic_state(
    score_matrix: np.ndarray,
    row_offset: int,
    row_count: int,
    row_top_k: int,
    sentence_top_k: int,
    link_floor: float,
    row_plateau_margin: float = 0.0,
    row_plateau_min_extra: int = 0,
    row_plateau_max_extra: int = 0,
    sent_meta: Optional[Dict[int, str]] = None,
    sentence_specificity_alpha: float = 0.0,
    section_priors: Optional[Dict[str, float]] = None,
    sentence_overflow_margin: float = 0.0,
    sentence_overflow_limit: int = 0,
    sentence_plateau_margin: float = 0.0,
    sentence_plateau_min_extra: int = 0,
    sentence_plateau_max_extra: int = 0,
    stopcue_sentence_ids: Optional[Set[int]] = None,
    stopcue_boost: float = 1.0,
) -> Dict[str, object]:
    row_floor_candidates: Dict[int, List[Tuple[int, float]]] = {}
    sentence_support_counts: Dict[int, int] = defaultdict(int)

    for local_idx in range(row_count):
        row_idx = row_offset + local_idx
        floor_candidates: List[Tuple[int, float]] = []
        for sent_idx in range(score_matrix.shape[1]):
            raw_score = float(score_matrix[row_idx, sent_idx])
            if raw_score < link_floor:
                continue
            floor_candidates.append((sent_idx, raw_score))
            sentence_support_counts[sent_idx] += 1
        row_floor_candidates[local_idx] = floor_candidates

    sentence_rank_weights = {
        sent_idx: _stage5_sentence_rank_weight(
            sent_idx,
            sentence_support=sentence_support_counts[sent_idx],
            sent_meta=sent_meta,
            sentence_specificity_alpha=sentence_specificity_alpha,
            section_priors=section_priors,
            stopcue_sentence_ids=stopcue_sentence_ids,
            stopcue_boost=stopcue_boost,
        )
        for sent_idx in sentence_support_counts
    }
    row_plateau_margin = max(float(row_plateau_margin), 0.0)
    row_plateau_min_extra = max(int(row_plateau_min_extra), 0)
    row_plateau_max_extra = max(int(row_plateau_max_extra), 0)
    overflow_margin = max(float(sentence_overflow_margin), 0.0)
    overflow_limit = max(int(sentence_overflow_limit), 0)
    sentence_plateau_margin = max(float(sentence_plateau_margin), 0.0)
    sentence_plateau_min_extra = max(int(sentence_plateau_min_extra), 0)
    sentence_plateau_max_extra = max(int(sentence_plateau_max_extra), 0)

    row_top_links: Dict[int, Dict[int, float]] = {}
    sent_candidates: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for local_idx, floor_candidates in row_floor_candidates.items():
        ranked_candidates = [
            (
                float(raw_score) * float(sentence_rank_weights.get(sent_idx, 1.0)),
                float(raw_score),
                int(sent_idx),
            )
            for sent_idx, raw_score in floor_candidates
        ]
        ranked_candidates.sort(reverse=True)
        selected_candidates, _selection_stats = _select_cutoff_band_items(
            ranked_candidates,
            base_limit=row_top_k,
            score_getter=lambda item: float(item[0]),
            plateau_margin=row_plateau_margin,
            plateau_min_extra=row_plateau_min_extra,
            plateau_max_extra=row_plateau_max_extra,
        )
        if row_offset == 0:
            selected_candidates, _rescued_count = _stage5_diag_stopcue_row_rescue(
                ranked_candidates,
                selected_candidates,
                stopcue_sentence_ids=stopcue_sentence_ids,
                link_floor=link_floor,
                max_extra=1,
            )
        row_top_links[local_idx] = {
            sent_idx: raw_score
            for _rank_score, raw_score, sent_idx in selected_candidates
        }
        for rank_score, raw_score, sent_idx in selected_candidates:
            sent_candidates[sent_idx].append((local_idx, raw_score, rank_score))

    links_by_sent: Dict[int, List[Tuple[int, float]]] = {}
    kept_rows_by_sent: Dict[int, set] = {}
    ranked_rows_by_sent: Dict[int, List[Tuple[int, float]]] = {}
    selection_stats_by_sent: Dict[int, Dict[str, object]] = {}
    for sent_idx, candidates in sent_candidates.items():
        candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
        ranked_rows_by_sent[sent_idx] = [
            (local_idx, score)
            for local_idx, score, _rank_score in candidates
        ]
        selected_candidates, selection_stats = _select_cutoff_band_items(
            candidates,
            base_limit=sentence_top_k,
            score_getter=lambda item: float(item[1]),
            overflow_margin=overflow_margin,
            overflow_limit=overflow_limit,
            plateau_margin=sentence_plateau_margin,
            plateau_min_extra=sentence_plateau_min_extra,
            plateau_max_extra=sentence_plateau_max_extra,
        )
        selection_stats_by_sent[sent_idx] = selection_stats
        kept = [
            (local_idx, score)
            for local_idx, score, _rank_score in selected_candidates
            if sent_idx in row_top_links.get(local_idx, {})
        ]
        links_by_sent[sent_idx] = kept
        kept_rows_by_sent[sent_idx] = {local_idx for local_idx, _score in kept}

    return {
        "row_top_links": row_top_links,
        "links_by_sent": links_by_sent,
        "kept_rows_by_sent": kept_rows_by_sent,
        "ranked_rows_by_sent": ranked_rows_by_sent,
        "selection_stats_by_sent": selection_stats_by_sent,
    }


def _stage5_pair_rankings_by_sentence(
    diag_state: Dict[str, object],
    med_state: Dict[str, object],
    gamma: float,
    max_pairs_per_sentence: int,
    pair_plateau_margin: float = 0.0,
    pair_plateau_min_extra: int = 0,
    pair_plateau_max_extra: int = 0,
) -> Dict[int, Dict[str, object]]:
    diag_links_by_sent = diag_state.get("links_by_sent", {})
    med_links_by_sent = med_state.get("links_by_sent", {})
    if not isinstance(diag_links_by_sent, dict) or not isinstance(med_links_by_sent, dict):
        return {}

    rankings: Dict[int, Dict[str, object]] = {}
    shared_sents = set(diag_links_by_sent) & set(med_links_by_sent)
    for sent_idx in shared_sents:
        sent_pairs: List[Tuple[float, float, float, int, int]] = []
        for diag_idx, score_diag in diag_links_by_sent.get(sent_idx, []):
            for med_idx, score_med in med_links_by_sent.get(sent_idx, []):
                path_score = (float(score_diag) + float(score_med)) / 2.0
                if path_score < gamma:
                    continue
                sent_pairs.append((path_score, float(score_diag), float(score_med), int(diag_idx), int(med_idx)))

        sent_pairs.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected_pairs, selection_stats = _select_cutoff_band_items(
            sent_pairs,
            base_limit=max_pairs_per_sentence,
            score_getter=lambda item: float(item[0]),
            plateau_margin=pair_plateau_margin,
            plateau_min_extra=pair_plateau_min_extra,
            plateau_max_extra=pair_plateau_max_extra,
        )
        selected_pair_keys = {
            (diag_idx, med_idx)
            for _path_score, _score_diag, _score_med, diag_idx, med_idx in selected_pairs
        }
        rankings[sent_idx] = {
            "pairs": {
                (diag_idx, med_idx): {
                    "rank": float(rank),
                    "path_score": round(path_score, 4),
                    "score_diag": round(score_diag, 4),
                    "score_med": round(score_med, 4),
                    "selected": (diag_idx, med_idx) in selected_pair_keys,
                }
                for rank, (path_score, score_diag, score_med, diag_idx, med_idx) in enumerate(sent_pairs, start=1)
            },
            "selection_stats": selection_stats,
        }

    return rankings


def _diagnose_stage5_pair_extraction_failure(
    diag_idx: int,
    med_idx: int,
    score_matrix: np.ndarray,
    n_diag: int,
    gamma: float,
    max_pairs_per_sentence: int,
    link_floor: float,
    diag_state: Dict[str, object],
    med_state: Dict[str, object],
    pair_rankings_by_sentence: Dict[int, Dict[str, object]],
    focus_sent_ids: Optional[Set[int]] = None,
) -> Dict[str, object]:
    diag_scores = score_matrix[diag_idx]
    med_scores = score_matrix[n_diag + med_idx]
    path_scores = (diag_scores + med_scores) / 2.0
    candidate_sent_ids = list(range(score_matrix.shape[1]))
    diagnostic_focus = "all_sentences"
    if focus_sent_ids:
        filtered_focus_sent_ids = sorted(
            {
                int(sent_idx)
                for sent_idx in focus_sent_ids
                if 0 <= int(sent_idx) < score_matrix.shape[1]
            }
        )
        if filtered_focus_sent_ids:
            candidate_sent_ids = filtered_focus_sent_ids
            diagnostic_focus = "gt_evidence"

    best_sent_idx = (
        max(
            candidate_sent_ids,
            key=lambda sent_idx: (
                float(path_scores[sent_idx]),
                float(diag_scores[sent_idx]),
                float(med_scores[sent_idx]),
            ),
        )
        if candidate_sent_ids else -1
    )
    best_path_score = float(path_scores[best_sent_idx]) if best_sent_idx >= 0 else 0.0
    best_score_diag = float(diag_scores[best_sent_idx]) if best_sent_idx >= 0 else 0.0
    best_score_med = float(med_scores[best_sent_idx]) if best_sent_idx >= 0 else 0.0

    floor_sentences = [
        sent_idx
        for sent_idx in candidate_sent_ids
        if float(diag_scores[sent_idx]) >= link_floor and float(med_scores[sent_idx]) >= link_floor
    ]

    row_top_links_diag = diag_state.get("row_top_links", {})
    row_top_links_med = med_state.get("row_top_links", {})
    kept_rows_diag = diag_state.get("kept_rows_by_sent", {})
    kept_rows_med = med_state.get("kept_rows_by_sent", {})
    ranked_rows_diag = diag_state.get("ranked_rows_by_sent", {})
    ranked_rows_med = med_state.get("ranked_rows_by_sent", {})
    selection_stats_diag = diag_state.get("selection_stats_by_sent", {})
    selection_stats_med = med_state.get("selection_stats_by_sent", {})

    def _row_cutoff_summary(
        ranked_rows_by_sent: object,
        selection_stats_by_sent: object,
        sent_idx: int,
        row_idx: int,
        row_score: float,
    ) -> Dict[str, object]:
        if not isinstance(ranked_rows_by_sent, dict) or not isinstance(selection_stats_by_sent, dict):
            return {
                "rank": None,
                "cutoff": None,
                "delta_to_cutoff": None,
                "candidate_count": 0,
                "selected_count": 0,
            }
        candidates = ranked_rows_by_sent.get(sent_idx, [])
        rank = None
        if isinstance(candidates, list):
            rank = next(
                (
                    position
                    for position, item in enumerate(candidates, start=1)
                    if isinstance(item, tuple) and len(item) >= 1 and int(item[0]) == row_idx
                ),
                None,
            )
        selection_stats = selection_stats_by_sent.get(sent_idx, {}) if isinstance(selection_stats_by_sent, dict) else {}
        cutoff_score = None
        if isinstance(selection_stats, dict) and selection_stats.get("cutoff_score") is not None:
            cutoff_score = float(selection_stats["cutoff_score"])
        return {
            "rank": rank,
            "cutoff": round(cutoff_score, 4) if cutoff_score is not None else None,
            "delta_to_cutoff": round(cutoff_score - float(row_score), 4) if cutoff_score is not None else None,
            "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
            "selected_count": int(selection_stats.get("selected_count", 0)) if isinstance(selection_stats, dict) else 0,
        }

    row_top_sentences = [
        sent_idx
        for sent_idx in floor_sentences
        if sent_idx in row_top_links_diag.get(diag_idx, {}) and sent_idx in row_top_links_med.get(med_idx, {})
    ]
    sentence_top_sentences = [
        sent_idx
        for sent_idx in row_top_sentences
        if diag_idx in kept_rows_diag.get(sent_idx, set()) and med_idx in kept_rows_med.get(sent_idx, set())
    ]
    row_top_above_gamma_sentences = [
        sent_idx
        for sent_idx in row_top_sentences
        if float(path_scores[sent_idx]) >= gamma
    ]
    above_gamma_sentences = [
        sent_idx
        for sent_idx in sentence_top_sentences
        if float(path_scores[sent_idx]) >= gamma
    ]

    retained_sentences: List[int] = []
    best_sentence_rank: Optional[int] = None
    for sent_idx in above_gamma_sentences:
        pair_payload = pair_rankings_by_sentence.get(sent_idx, {})
        ranking_lookup = pair_payload.get("pairs", {}) if isinstance(pair_payload, dict) else {}
        ranking = ranking_lookup.get((diag_idx, med_idx)) if isinstance(ranking_lookup, dict) else None
        if not ranking:
            continue
        rank = int(ranking.get("rank", max_pairs_per_sentence + 1))
        if best_sentence_rank is None or rank < best_sentence_rank:
            best_sentence_rank = rank
        if bool(ranking.get("selected", False)):
            retained_sentences.append(sent_idx)

    best_sentence_diag_summary = _row_cutoff_summary(
        ranked_rows_diag,
        selection_stats_diag,
        best_sent_idx,
        diag_idx,
        best_score_diag,
    ) if best_sent_idx >= 0 else {"rank": None, "cutoff": None, "delta_to_cutoff": None, "candidate_count": 0, "selected_count": 0}
    best_sentence_med_summary = _row_cutoff_summary(
        ranked_rows_med,
        selection_stats_med,
        best_sent_idx,
        med_idx,
        best_score_med,
    ) if best_sent_idx >= 0 else {"rank": None, "cutoff": None, "delta_to_cutoff": None, "candidate_count": 0, "selected_count": 0}

    best_sentence_pair_payload = pair_rankings_by_sentence.get(best_sent_idx, {}) if best_sent_idx >= 0 else {}
    best_sentence_pair_lookup = best_sentence_pair_payload.get("pairs", {}) if isinstance(best_sentence_pair_payload, dict) else {}
    best_sentence_pair_ranking = best_sentence_pair_lookup.get((diag_idx, med_idx), {}) if isinstance(best_sentence_pair_lookup, dict) else {}
    best_sentence_pair_selection = best_sentence_pair_payload.get("selection_stats", {}) if isinstance(best_sentence_pair_payload, dict) else {}
    pair_cutoff_score = None
    if isinstance(best_sentence_pair_selection, dict) and best_sentence_pair_selection.get("cutoff_score") is not None:
        pair_cutoff_score = float(best_sentence_pair_selection["cutoff_score"])

    failure_stage = "STAGE5_UNATTRIBUTED"
    if not floor_sentences:
        failure_stage = "LINK_FLOOR"
    elif not row_top_sentences:
        failure_stage = "ROW_SIDE_TOP_K"
    elif not sentence_top_sentences and row_top_above_gamma_sentences:
        failure_stage = "SENTENCE_SIDE_TOP_K"
    elif row_top_above_gamma_sentences and not above_gamma_sentences:
        failure_stage = "SENTENCE_SIDE_TOP_K"
    elif not above_gamma_sentences:
        failure_stage = "TRANSITIVE_JOIN_THRESHOLD"
    elif not retained_sentences:
        failure_stage = "MAX_PAIRS_PER_SENTENCE"

    return {
        "failure_stage": failure_stage,
        "diagnostic_focus": diagnostic_focus,
        "n_focus_sentences": len(candidate_sent_ids),
        "best_sentence_idx": best_sent_idx,
        "best_path_score": round(best_path_score, 4),
        "best_score_diag": round(best_score_diag, 4),
        "best_score_med": round(best_score_med, 4),
        "link_floor": round(float(link_floor), 4),
        "n_floor_sentences": len(floor_sentences),
        "n_row_top_sentences": len(row_top_sentences),
        "n_sentence_top_sentences": len(sentence_top_sentences),
        "n_row_top_above_gamma_sentences": len(row_top_above_gamma_sentences),
        "n_above_gamma_sentences": len(above_gamma_sentences),
        "n_retained_sentences": len(retained_sentences),
        "best_sentence_rank": best_sentence_rank,
        "best_sentence_diag_rank": best_sentence_diag_summary.get("rank"),
        "best_sentence_med_rank": best_sentence_med_summary.get("rank"),
        "best_sentence_diag_cutoff": best_sentence_diag_summary.get("cutoff"),
        "best_sentence_med_cutoff": best_sentence_med_summary.get("cutoff"),
        "best_sentence_diag_delta_to_cutoff": best_sentence_diag_summary.get("delta_to_cutoff"),
        "best_sentence_med_delta_to_cutoff": best_sentence_med_summary.get("delta_to_cutoff"),
        "best_sentence_diag_candidate_count": best_sentence_diag_summary.get("candidate_count"),
        "best_sentence_med_candidate_count": best_sentence_med_summary.get("candidate_count"),
        "best_sentence_diag_selected_count": best_sentence_diag_summary.get("selected_count"),
        "best_sentence_med_selected_count": best_sentence_med_summary.get("selected_count"),
        "best_sentence_pair_rank": int(best_sentence_pair_ranking.get("rank", 0)) if best_sentence_pair_ranking else None,
        "best_sentence_pair_cutoff": round(pair_cutoff_score, 4) if pair_cutoff_score is not None else None,
        "best_sentence_pair_delta_to_cutoff": round(pair_cutoff_score - best_path_score, 4) if pair_cutoff_score is not None else None,
        "best_sentence_pair_candidate_count": int(best_sentence_pair_selection.get("candidate_count", 0)) if isinstance(best_sentence_pair_selection, dict) else 0,
        "best_sentence_pair_selected_count": int(best_sentence_pair_selection.get("selected_count", 0)) if isinstance(best_sentence_pair_selection, dict) else 0,
    }


def _build_gt_pair_recovery_diagnostics(
    gt_relationships: List[Dict],
    pair_scores: torch.Tensor,
    n_diag: int,
    sent_texts: List[str],
    gamma: float,
    diag_row_top_k: int,
    med_row_top_k: int,
    row_plateau_margin: float,
    row_plateau_min_extra: int,
    row_plateau_max_extra: int,
    sent_diag_top_k: int,
    sent_med_top_k: int,
    max_pairs_per_sentence: int,
    sent_meta: Optional[Dict[int, str]],
    sentence_specificity_alpha: float,
    section_priors: Optional[Dict[str, float]],
    sentence_overflow_margin: float,
    sentence_overflow_limit: int,
    sentence_plateau_margin: float,
    sentence_plateau_min_extra: int,
    sentence_plateau_max_extra: int,
    pair_plateau_margin: float,
    pair_plateau_min_extra: int,
    pair_plateau_max_extra: int,
    pair_recovery_diagnostics: Dict[str, object],
    pair_filter_stats: Dict,
    ce_pair_filter_stats: Dict,
    cluster_tail_filter_stats: Dict,
) -> Dict[str, object]:
    gt_pair_types: Dict[Tuple[int, int], set] = defaultdict(set)
    gt_pair_evidence_sents: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for rel in gt_relationships:
        pair = (int(rel["diag_idx"]), int(rel["drug_idx"]))
        gt_pair_types[pair].add(str(rel["rel_type"]))
        for sent_idx in rel.get("evidence_sents", []) or []:
            try:
                gt_pair_evidence_sents[pair].add(int(sent_idx))
            except (TypeError, ValueError):
                continue

    if not gt_pair_types:
        return {
            "summary": {
                "n_gt_pairs": 0,
                "n_recovered_pairs": 0,
                "n_missed_pairs": 0,
                "by_stage": {},
            },
            "pairs": [],
        }

    score_matrix = pair_scores.detach().cpu().numpy()
    n_med = score_matrix.shape[0] - n_diag
    link_floor = max(float(gamma) / (2 ** 0.5), 0.15)
    stopcue_sentence_ids = {
        sent_idx
        for sent_idx, sent_text in enumerate(sent_texts)
        if any(re.search(pattern, str(sent_text), flags=re.IGNORECASE) for pattern in _EXPLICIT_DISCONTINUE_PATTERNS)
    }
    stopcue_boost = 1.25 if stopcue_sentence_ids else 1.0
    diag_state = _stage5_atomic_link_diagnostic_state(
        score_matrix,
        row_offset=0,
        row_count=n_diag,
        row_top_k=diag_row_top_k,
        sentence_top_k=sent_diag_top_k,
        link_floor=link_floor,
        row_plateau_margin=row_plateau_margin,
        row_plateau_min_extra=row_plateau_min_extra,
        row_plateau_max_extra=row_plateau_max_extra,
        sent_meta=sent_meta,
        sentence_specificity_alpha=sentence_specificity_alpha,
        section_priors=section_priors,
        sentence_overflow_margin=sentence_overflow_margin,
        sentence_overflow_limit=sentence_overflow_limit,
        sentence_plateau_margin=sentence_plateau_margin,
        sentence_plateau_min_extra=sentence_plateau_min_extra,
        sentence_plateau_max_extra=sentence_plateau_max_extra,
        stopcue_sentence_ids=stopcue_sentence_ids,
        stopcue_boost=stopcue_boost,
    )
    med_state = _stage5_atomic_link_diagnostic_state(
        score_matrix,
        row_offset=n_diag,
        row_count=n_med,
        row_top_k=med_row_top_k,
        sentence_top_k=sent_med_top_k,
        link_floor=link_floor,
        row_plateau_margin=row_plateau_margin,
        row_plateau_min_extra=row_plateau_min_extra,
        row_plateau_max_extra=row_plateau_max_extra,
        sent_meta=sent_meta,
        sentence_specificity_alpha=sentence_specificity_alpha,
        section_priors=section_priors,
        sentence_overflow_margin=sentence_overflow_margin,
        sentence_overflow_limit=sentence_overflow_limit,
        sentence_plateau_margin=sentence_plateau_margin,
        sentence_plateau_min_extra=sentence_plateau_min_extra,
        sentence_plateau_max_extra=sentence_plateau_max_extra,
        stopcue_sentence_ids=stopcue_sentence_ids,
        stopcue_boost=stopcue_boost,
    )
    pair_rankings_by_sentence = _stage5_pair_rankings_by_sentence(
        diag_state,
        med_state,
        gamma=float(gamma),
        max_pairs_per_sentence=max_pairs_per_sentence,
        pair_plateau_margin=pair_plateau_margin,
        pair_plateau_min_extra=pair_plateau_min_extra,
        pair_plateau_max_extra=pair_plateau_max_extra,
    )

    stage5_lookup = _pair_lookup_from_stage_snapshot(pair_recovery_diagnostics.get("after_stage5_extraction"))
    pair_filter_lookup = _pair_lookup_from_stage_snapshot(pair_recovery_diagnostics.get("after_pair_filter"))
    ce_pair_filter_enabled = bool(ce_pair_filter_stats.get("enabled")) and str(ce_pair_filter_stats.get("mode", "off")).strip().lower() != "off"
    ce_pair_lookup = (
        _pair_lookup_from_stage_snapshot(pair_recovery_diagnostics.get("after_ce_pair_filter"))
        if ce_pair_filter_enabled
        else pair_filter_lookup
    )
    cluster_tail_lookup = _pair_lookup_from_stage_snapshot(pair_recovery_diagnostics.get("after_cluster_tail_filter"))
    final_lookup = _pair_lookup_from_stage_snapshot(pair_recovery_diagnostics.get("after_low_signal_cluster_filter"))

    pair_filter_decisions = _pair_decision_lookup(pair_filter_stats.get("pair_decisions"))
    ce_pair_decisions = _pair_decision_lookup(ce_pair_filter_stats.get("pair_decisions"))
    cluster_tail_decisions = _pair_decision_lookup(cluster_tail_filter_stats.get("pair_decisions"))
    cluster_signal_lookup = {
        int(record["cluster_id"]): record
        for record in pair_recovery_diagnostics.get("cluster_signal_filter", [])
        if isinstance(record, dict) and "cluster_id" in record
    }

    pair_records: List[Dict[str, object]] = []
    stage_counts: Dict[str, int] = defaultdict(int)
    for pair in sorted(gt_pair_types):
        diag_idx, med_idx = pair
        record: Dict[str, object] = {
            "diag_row_idx": diag_idx,
            "med_row_idx": med_idx,
            "gt_rel_types": sorted(gt_pair_types[pair]),
        }

        final_record = final_lookup.get(pair)
        if final_record is not None:
            record.update({
                "status": "recovered",
                "failure_stage": "RECOVERED",
                "final_n_paths": int(final_record.get("n_paths", 0)),
                "final_n_unique_sentences": int(final_record.get("n_unique_sentences", 0)),
                "best_path_score": float(final_record.get("best_path_score", 0.0)),
            })
            if "cluster_id" in final_record:
                record["cluster_id"] = int(final_record["cluster_id"])
        elif pair not in stage5_lookup:
            record.update({
                "status": "missed",
                **_diagnose_stage5_pair_extraction_failure(
                    diag_idx,
                    med_idx,
                    score_matrix,
                    n_diag=n_diag,
                    gamma=float(gamma),
                    max_pairs_per_sentence=max_pairs_per_sentence,
                    link_floor=link_floor,
                    diag_state=diag_state,
                    med_state=med_state,
                    pair_rankings_by_sentence=pair_rankings_by_sentence,
                    focus_sent_ids=gt_pair_evidence_sents.get(pair),
                ),
            })
        elif pair not in pair_filter_lookup:
            decision = pair_filter_decisions.get(pair, {})
            record.update({
                "status": "missed",
                "failure_stage": "PAIR_FILTER",
                "filter_reason": str(decision.get("reason", "pair_filter_drop")),
                "best_path_score": float(decision.get("best_score", 0.0)),
                "support_count": int(decision.get("support_count", 0)),
                "max_sentence_fanout": int(decision.get("max_sentence_fanout", 0)),
                "diag_rank": int(decision.get("diag_rank", 0)),
                "med_rank": int(decision.get("med_rank", 0)),
            })
        elif ce_pair_filter_enabled and pair not in ce_pair_lookup:
            decision = ce_pair_decisions.get(pair, {})
            record.update({
                "status": "missed",
                "failure_stage": "CE_PAIR_FILTER",
                "filter_reason": str(decision.get("reason", "ce_pair_filter_drop")),
                "best_path_score": float(decision.get("best_loki_score", 0.0)),
                "best_ce_score": float(decision.get("best_ce_score", 0.0)),
                "ce_delta_to_cutoff": float(decision.get("ce_delta_to_cutoff", 0.0)),
                "loki_delta_to_cutoff": float(decision.get("loki_delta_to_cutoff", 0.0)),
                "support_count": int(decision.get("support_count", 0)),
            })
        elif pair not in cluster_tail_lookup:
            decision = cluster_tail_decisions.get(pair, {})
            record.update({
                "status": "missed",
                "failure_stage": "CLUSTER_TAIL_FILTER",
                "filter_reason": str(decision.get("reason", "cluster_tail_drop")),
                "best_path_score": float(decision.get("best_score", 0.0)),
                "cluster_id": int(decision.get("cluster_id", -1)),
                "rank_within_cluster": int(decision.get("rank", 0)),
                "support_count": int(decision.get("support_count", 0)),
            })
        else:
            pre_signal_record = cluster_tail_lookup.get(pair, {})
            cluster_id = pre_signal_record.get("cluster_id")
            record.update({
                "status": "missed",
                "failure_stage": "LOW_SIGNAL_CLUSTER_FILTER",
                "best_path_score": float(pre_signal_record.get("best_path_score", 0.0)),
            })
            if cluster_id is not None:
                record["cluster_id"] = int(cluster_id)
                signal_record = cluster_signal_lookup.get(int(cluster_id), {})
                if signal_record:
                    record["cluster_connection_score"] = float(signal_record.get("connection_score", 0.0))
                    record["cluster_threshold"] = float(signal_record.get("threshold", 0.0))
                    record["cluster_n_pairs"] = int(signal_record.get("n_pairs", 0))
                    record["cluster_n_paths"] = int(signal_record.get("n_paths", 0))

        stage_counts[str(record["failure_stage"])] += 1
        pair_records.append(record)

    n_recovered_pairs = stage_counts.get("RECOVERED", 0)
    return {
        "summary": {
            "n_gt_pairs": len(gt_pair_types),
            "n_recovered_pairs": n_recovered_pairs,
            "n_missed_pairs": len(gt_pair_types) - n_recovered_pairs,
            "by_stage": dict(sorted(stage_counts.items())),
        },
        "pairs": pair_records,
    }


def _gt_failure_stage_sort_key(stage_name: object) -> Tuple[int, str]:
    resolved = str(stage_name or "").strip().upper()
    order = {
        "RECOVERED": 0,
        "SENTENCE_SIDE_TOP_K": 1,
        "MAX_PAIRS_PER_SENTENCE": 2,
        "TRANSITIVE_JOIN_THRESHOLD": 3,
        "PAIR_FILTER": 4,
        "CE_PAIR_FILTER": 5,
        "CLUSTER_TAIL_FILTER": 6,
        "LOW_SIGNAL_CLUSTER_FILTER": 7,
        "ROW_SIDE_TOP_K": 8,
        "LINK_FLOOR": 9,
        "STAGE5_UNATTRIBUTED": 10,
    }
    return (order.get(resolved, 100), resolved)


def _gt_failure_focus_priority(record: Dict[str, object]) -> Tuple[int, float, float, int, int]:
    stage = str(record.get("failure_stage", "")).strip().upper()
    if stage == "SENTENCE_SIDE_TOP_K":
        deltas = [
            value
            for value in (
                _to_float_or_none(record.get("best_sentence_diag_delta_to_cutoff")),
                _to_float_or_none(record.get("best_sentence_med_delta_to_cutoff")),
            )
            if value is not None
        ]
        delta = min((max(value, 0.0) for value in deltas), default=999.0)
        return (0, delta, -float(record.get("best_path_score", 0.0)), int(record.get("diag_row_idx", -1)), int(record.get("med_row_idx", -1)))
    if stage == "MAX_PAIRS_PER_SENTENCE":
        pair_delta = _to_float_or_none(record.get("best_sentence_pair_delta_to_cutoff"))
        return (1, max(pair_delta or 999.0, 0.0), -float(record.get("best_path_score", 0.0)), int(record.get("diag_row_idx", -1)), int(record.get("med_row_idx", -1)))
    if stage == "TRANSITIVE_JOIN_THRESHOLD":
        return (2, -float(record.get("best_path_score", 0.0)), 0.0, int(record.get("diag_row_idx", -1)), int(record.get("med_row_idx", -1)))
    return (3, _gt_failure_stage_sort_key(stage)[0], -float(record.get("best_path_score", 0.0)), int(record.get("diag_row_idx", -1)), int(record.get("med_row_idx", -1)))


def _summarize_gt_pair_failures(
    gt_pair_recovery: object,
    focus_limit: int = 4,
) -> Optional[Dict[str, object]]:
    if not isinstance(gt_pair_recovery, dict):
        return None

    summary = gt_pair_recovery.get("summary", {})
    pair_records = gt_pair_recovery.get("pairs", [])
    if not isinstance(summary, dict) or not isinstance(pair_records, list):
        return None

    by_stage = summary.get("by_stage", {}) if isinstance(summary.get("by_stage", {}), dict) else {}
    missed_records = [
        record for record in pair_records
        if isinstance(record, dict) and str(record.get("failure_stage", "")).upper() != "RECOVERED"
    ]
    missed_records.sort(key=_gt_failure_focus_priority)

    focus_records: List[Dict[str, object]] = []
    for record in missed_records[:max(int(focus_limit), 0)]:
        focus_record = {
            "diag_row_idx": int(record.get("diag_row_idx", -1)),
            "med_row_idx": int(record.get("med_row_idx", -1)),
            "failure_stage": str(record.get("failure_stage", "")),
            "best_path_score": _to_float_or_none(record.get("best_path_score")),
            "best_sentence_idx": record.get("best_sentence_idx"),
            "gt_rel_types": list(record.get("gt_rel_types", [])) if isinstance(record.get("gt_rel_types", []), list) else [],
        }
        for key in (
            "best_sentence_diag_rank",
            "best_sentence_med_rank",
            "best_sentence_pair_rank",
            "best_sentence_diag_delta_to_cutoff",
            "best_sentence_med_delta_to_cutoff",
            "best_sentence_pair_delta_to_cutoff",
        ):
            if key in record:
                focus_record[key] = record.get(key)
        focus_records.append(focus_record)

    return {
        "n_gt_pairs": int(summary.get("n_gt_pairs", 0)),
        "n_recovered_pairs": int(summary.get("n_recovered_pairs", 0)),
        "n_missed_pairs": int(summary.get("n_missed_pairs", 0)),
        "by_stage": dict(sorted(((str(stage), int(count)) for stage, count in by_stage.items()), key=lambda item: _gt_failure_stage_sort_key(item[0]))),
        "focus_misses": focus_records,
    }


def _print_gt_pair_failure_report(gt_pair_failure_report: object) -> None:
    if not isinstance(gt_pair_failure_report, dict):
        return

    n_gt_pairs = int(gt_pair_failure_report.get("n_gt_pairs", 0))
    n_recovered_pairs = int(gt_pair_failure_report.get("n_recovered_pairs", 0))
    n_missed_pairs = int(gt_pair_failure_report.get("n_missed_pairs", 0))
    by_stage = gt_pair_failure_report.get("by_stage", {}) if isinstance(gt_pair_failure_report.get("by_stage", {}), dict) else {}
    focus_misses = gt_pair_failure_report.get("focus_misses", []) if isinstance(gt_pair_failure_report.get("focus_misses", []), list) else []

    print("  GT pair recovery:")
    print(f"    recovered={n_recovered_pairs}/{n_gt_pairs}  missed={n_missed_pairs}")
    if by_stage:
        stage_text = ", ".join(f"{stage}={count}" for stage, count in by_stage.items())
        print(f"    by_stage: {stage_text}")
    if focus_misses:
        print("    Closest remaining misses:")
        for record in focus_misses:
            diag_idx = int(record.get("diag_row_idx", -1)) + 1
            med_idx = int(record.get("med_row_idx", -1)) + 1
            stage_name = str(record.get("failure_stage", ""))
            best_path_score = _to_float_or_none(record.get("best_path_score"))
            parts = [
                f"diag[{diag_idx}] x med[{med_idx}]",
                stage_name,
            ]
            if best_path_score is not None:
                parts.append(f"best_path={best_path_score:.4f}")
            for label, key in (
                ("diag_rank", "best_sentence_diag_rank"),
                ("med_rank", "best_sentence_med_rank"),
                ("pair_rank", "best_sentence_pair_rank"),
                ("diag_delta", "best_sentence_diag_delta_to_cutoff"),
                ("med_delta", "best_sentence_med_delta_to_cutoff"),
                ("pair_delta", "best_sentence_pair_delta_to_cutoff"),
            ):
                value = record.get(key)
                numeric_value = _to_float_or_none(value)
                if numeric_value is None:
                    continue
                if label.endswith("_delta"):
                    parts.append(f"{label}={numeric_value:.4f}")
                else:
                    parts.append(f"{label}={int(numeric_value)}")
            rel_types = record.get("gt_rel_types", [])
            if isinstance(rel_types, list) and rel_types:
                parts.append(f"types={'/'.join(str(rel_type) for rel_type in rel_types)}")
            print(f"      {'  '.join(parts)}")


def _gt_failure_report_stage_count(
    gt_pair_failure_report: object,
    stage_name: str,
) -> Optional[int]:
    if not isinstance(gt_pair_failure_report, dict):
        return None

    by_stage = gt_pair_failure_report.get("by_stage", {})
    if not isinstance(by_stage, dict):
        return None

    raw_value = by_stage.get(stage_name, 0)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _compute_pair_embeddings(
    paths: List[Dict],
    refined_sentences: torch.Tensor,
    refined_rows: Optional[torch.Tensor] = None,
    n_diag: Optional[int] = None,
    sentence_encoder: Optional[SentenceTransformer] = None,
    embedding_mode: str = "contextual_sentence_average",
    verbose: bool = True,
) -> Tuple[List[Tuple[int, int]], PairEmbeddingTensor]:
    pair_buckets = _bucket_paths_by_pair(paths)
    pair_keys = sorted(pair_buckets)
    if not pair_keys:
        return [], torch.empty((0, 0), device=_pair_embedding_device(refined_sentences, refined_rows), dtype=torch.float32)

    resolved_mode = (embedding_mode or "signature").strip().lower()
    if resolved_mode not in {"signature", "semantic_signature", "contextual_sentence_average", "row_pair_hybrid"}:
        raise ValueError(f"Unsupported pair embedding mode: {embedding_mode}")

    cache_key: PairEmbeddingCacheKey = (
        id(paths),
        id(refined_sentences),
        id(refined_rows) if refined_rows is not None else 0,
        len(paths),
        int(n_diag) if n_diag is not None else -1,
        resolved_mode,
    )
    cached_result = _PAIR_EMBEDDING_CACHE.get(cache_key)
    if cached_result is not None:
        return cached_result

    embedding_device = _pair_embedding_device(refined_sentences, refined_rows)
    refined_sentences_tensor = _ensure_embedding_tensor(refined_sentences, device=embedding_device)

    def _weighted_sentence_average(pair_paths: List[Dict]) -> torch.Tensor:
        sentence_ids = torch.as_tensor(
            [int(path["sent_idx"]) for path in pair_paths],
            device=embedding_device,
            dtype=torch.long,
        )
        weights = torch.as_tensor(
            [max(float(path.get("path_score", 0.0)), 1e-4) for path in pair_paths],
            device=embedding_device,
            dtype=torch.float32,
        )
        sent_vectors = refined_sentences_tensor.index_select(0, sentence_ids)
        weights = weights / weights.sum().clamp_min(1e-6)
        return torch.sum(sent_vectors * weights.unsqueeze(1), dim=0)

    embs_norm: Optional[torch.Tensor] = None
    if resolved_mode in {"signature", "semantic_signature"} and sentence_encoder is not None:
        try:
            if resolved_mode == "semantic_signature":
                pair_signatures = [_build_pair_semantic_signature(pair_buckets[pair]) for pair in pair_keys]
                mode_label = "condensed semantic signatures"
            else:
                pair_signatures = [_build_pair_signature(pair_buckets[pair]) for pair in pair_keys]
                mode_label = "aggregated evidence signatures"
            encoded = sentence_encoder.encode(
                pair_signatures,
                batch_size=min(32, len(pair_signatures)),
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            embs_norm = _ensure_embedding_tensor(encoded, device=embedding_device, normalize=True)
            if verbose:
                print(f"  Clustering {len(pair_keys)} candidate row pairs using {mode_label}")
        except Exception as exc:
            if verbose:
                print(f"  Warning: {resolved_mode} pair encoding failed ({exc}). Falling back to contextual sentence aggregation.")
            sentence_encoder = None

    if (
        embs_norm is None
        and resolved_mode == "row_pair_hybrid"
        and refined_rows is not None
        and n_diag is not None
        and refined_rows.shape[0] > (n_diag + max(pair[1] for pair in pair_keys))
    ):
        pair_vectors: List[torch.Tensor] = []
        refined_rows_tensor = _ensure_embedding_tensor(refined_rows, device=embedding_device)
        for diag_idx, med_idx in pair_keys:
            pair_paths = pair_buckets[(diag_idx, med_idx)]
            sentence_avg = _weighted_sentence_average(pair_paths)
            component_vectors = torch.stack([
                refined_rows_tensor[int(diag_idx)],
                sentence_avg,
                refined_rows_tensor[int(n_diag + med_idx)],
            ], dim=0)
            component_vectors = torch.nn.functional.normalize(component_vectors, p=2, dim=-1)
            pair_vectors.append(component_vectors.reshape(-1))

        embs_norm = torch.nn.functional.normalize(torch.stack(pair_vectors, dim=0), p=2, dim=-1)
        if verbose:
            print(f"  Clustering {len(pair_keys)} candidate row pairs using row-pair hybrid embeddings")

    if embs_norm is None:
        pair_vectors = []
        for pair in pair_keys:
            pair_paths = pair_buckets[pair]
            pair_vectors.append(_weighted_sentence_average(pair_paths))

        embs_norm = torch.nn.functional.normalize(torch.stack(pair_vectors, dim=0), p=2, dim=-1)
        if verbose:
            print(f"  Clustering {len(pair_keys)} candidate row pairs using weighted contextual sentence averages")

    _PAIR_EMBEDDING_CACHE[cache_key] = (pair_keys, embs_norm)
    return pair_keys, embs_norm


def _cluster_pair_embeddings(
    pair_embeddings: PairEmbeddingTensor,
    verbose: bool = True,
    min_cluster_size: int = 2,
) -> Tuple[np.ndarray, int]:
    if pair_embeddings.shape[0] == 0:
        return np.array([], dtype=int), 0
    if pair_embeddings.shape[0] == 1:
        return np.zeros(1, dtype=int), 1

    effective_min_cluster_size = max(2, min_cluster_size)
    pair_labels, cluster_backend = _fit_hdbscan_labels(
        pair_embeddings,
        min_cluster_size=effective_min_cluster_size,
        metric="euclidean",
        verbose=verbose,
        context="pair clustering",
    )
    n_clusters_real = len(set(pair_labels) - {-1})
    n_noise = int((pair_labels == -1).sum())
    if verbose:
        print(f"  HDBSCAN ({cluster_backend}): {n_clusters_real} clusters, {n_noise} noise points")

    unique_labels: np.ndarray = pair_labels.copy()
    next_id = int(unique_labels.max()) + 1 if len(unique_labels) > 0 else 0
    for idx, lbl in enumerate(pair_labels):
        if lbl == -1:
            unique_labels[idx] = next_id
            next_id += 1

    return unique_labels.astype(int), len(set(unique_labels))


def cluster_mediating_sentences(
    paths: List[Dict],
    refined_sentences: torch.Tensor,
    refined_rows: Optional[torch.Tensor] = None,
    n_diag: Optional[int] = None,
    sentence_encoder: Optional[SentenceTransformer] = None,
    embedding_mode: str = "contextual_sentence_average",
    max_clusters: int = 4,
    hdbscan_min_cluster_size: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Cluster diagnosis-medication row pairs using aggregated evidence signatures.

    Returns (labels aligned to paths list, n_clusters).

    The primary representation is a per-pair evidence signature built from the
    diagnosis row, medication row, and the top mediating sentences for that pair.
    This makes the clustering objective explicitly about whether row pairs from the
    two tables behave like the same relationship, rather than about sentence-level
    mediator semantics alone. If signature encoding is unavailable, fall back to a
    weighted average of the pair's contextualized sentence embeddings.
    """
    if not paths:
        return np.array([], dtype=int), 0

    pair_keys, pair_embeddings = _compute_pair_embeddings(
        paths,
        refined_sentences,
        refined_rows=refined_rows,
        n_diag=n_diag,
        sentence_encoder=sentence_encoder,
        embedding_mode=embedding_mode,
        verbose=True,
    )
    if len(pair_keys) == 1:
        return np.zeros(len(paths), dtype=int), 1

    # Phase 5: auto-calibrate min_cluster_size from pair/rel-type ratio if hdbscan_min_cluster_size==0
    _n_rel_types = max(len(_preferred_rel_type_order()), 1)
    _eff_min_cs = (
        max(2, len(pair_keys) // (3 * _n_rel_types))
        if hdbscan_min_cluster_size <= 0
        else max(2, hdbscan_min_cluster_size)
    )
    unique_labels, n_total_clusters = _cluster_pair_embeddings(pair_embeddings, verbose=True, min_cluster_size=_eff_min_cs)

    pair_to_label = {
        pair: int(unique_labels[idx])
        for idx, pair in enumerate(pair_keys)
    }
    labels = np.array([
        pair_to_label[(p["diag_row_idx"], p["med_row_idx"])]
        for p in paths
    ], dtype=int)
    return labels, n_total_clusters


def _cluster_labels_to_numpy(raw_labels: Any) -> np.ndarray:
    if isinstance(raw_labels, np.ndarray):
        return raw_labels.astype(int, copy=False)
    to_numpy = getattr(raw_labels, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy(), dtype=int)
    to_pandas = getattr(raw_labels, "to_pandas", None)
    if callable(to_pandas):
        return np.asarray(to_pandas().to_numpy(), dtype=int)
    get_method = getattr(raw_labels, "get", None)
    if callable(get_method):
        return np.asarray(get_method(), dtype=int)
    return np.asarray(raw_labels, dtype=int)


def _fit_hdbscan_labels(
    embeddings: Any,
    *,
    min_cluster_size: int,
    metric: str = "euclidean",
    verbose: bool = True,
    context: str = "clustering",
) -> Tuple[np.ndarray, str]:
    gpu_error: Optional[BaseException] = None
    try:
        from cuml.cluster import HDBSCAN as CuMLHDBSCAN  # type: ignore

        gpu_embeddings: Any = embeddings
        try:
            import cupy as cp  # type: ignore

            if torch.is_tensor(embeddings):
                tensor_embeddings = embeddings.detach().contiguous()
                if tensor_embeddings.is_cuda:
                    gpu_embeddings = cp.from_dlpack(tensor_embeddings)
                else:
                    gpu_embeddings = cp.asarray(tensor_embeddings.cpu().numpy(), dtype=cp.float32)
            else:
                gpu_embeddings = cp.asarray(_to_numpy_array(embeddings, dtype=np.float32), dtype=cp.float32)
        except Exception:
            gpu_embeddings = _to_numpy_array(embeddings, dtype=np.float32)

        clusterer = CuMLHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric=metric,
        )
        return _cluster_labels_to_numpy(clusterer.fit_predict(gpu_embeddings)), "rapids_gpu"
    except Exception as exc:
        gpu_error = exc

    try:
        import hdbscan  # type: ignore

        if verbose:
            print(
                f"  [WARNING] GPU HDBSCAN unavailable for {context} ({gpu_error}). "
                "Falling back to CPU HDBSCAN."
            )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric=metric,
        )
        return _cluster_labels_to_numpy(clusterer.fit_predict(_to_numpy_array(embeddings, dtype=np.float32))), "cpu"
    except Exception as cpu_exc:
        raise RuntimeError(
            f"HDBSCAN unavailable for {context}. GPU attempt failed ({gpu_error}); "
            f"CPU HDBSCAN failed ({cpu_exc}). Install RAPIDS cuML or hdbscan."
        ) from cpu_exc


# Active relationship types are derived from the available annotation corpus.
DEFAULT_REL_TYPES = ["TREATS", "ADVERSE_EFFECT", "DISCONTINUED"]
REL_TYPES = list(DEFAULT_REL_TYPES)
_KNOWN_REL_TYPE_PRIORITY = {
    "TREATS": 0,
    "ADVERSE_EFFECT": 1,
    "DISCONTINUED": 2,
    "CONTRAINDICATED": 3,
    "NEGATIVE": 4,
    "OTHER": 5,
}

_LABEL_DEFINITIONS = {
    "TREATS": "the medication is used, prescribed, started, or continued to treat or manage the diagnosis.",
    "ADVERSE_EFFECT": "the medication caused, worsened, or is suspected to cause the diagnosis or symptom.",
    "DISCONTINUED": "the medication was stopped, held, avoided, or switched away from.",
    "CONTRAINDICATED": "the medication should not be used because it is contraindicated, unsafe, or inappropriate for the diagnosis or patient context.",
    "NEGATIVE": "the evidence explicitly indicates there is no supported diagnosis-medication relationship for this admission.",
    "OTHER": "the evidence is not primarily treatment; it instead reflects either an adverse effect or a medication being stopped, held, or switched.",
}

# Clinical keyword signals per type (keyword-based fallback classifier)
_REL_KEYWORDS: Dict[str, List[str]] = {
    "TREATS":         ["treat", "prescri", "start", "receiv", "continu", "given",
                       "therapy", "regimen", "initiat"],
    "ADVERSE_EFFECT": ["adverse", "side effect", "cause", "suspect", "culprit",
                       "concern", "toxic", "induc", "hallucinat", "intolera",
                       "drop", "platelet", "thrombocytopenia", "edema"],
    "DISCONTINUED":   ["discontinu", "held", "stopp", "stop", "d/c", "dc'd",
                       "withdrawn", "no longer", "switch", "avoid", "held/discontinued"],
    "CONTRAINDICATED": ["contraindicat", "contraindication", "unsafe", "cannot use",
                       "not candidate", "avoid due to", "allergy"],
    "NEGATIVE":       ["unrelated", "not linked", "not related", "not mentioned",
                       "no evidence", "no relationship", "not for", "does not treat",
                       "does not cause", "prophylaxis"],
    "OTHER":          [],
}


def _normalize_rel_type(rel_type: str) -> str:
    normalized = re.sub(r"\s+", "_", str(rel_type).strip().upper())
    if normalized == "CONTEXT":
        return "NEGATIVE"
    return normalized


def _rel_type_sort_key(rel_type: str) -> Tuple[int, str]:
    normalized = _normalize_rel_type(rel_type)
    return (_KNOWN_REL_TYPE_PRIORITY.get(normalized, len(_KNOWN_REL_TYPE_PRIORITY)), normalized)


def _set_active_rel_types(rel_types: List[str]) -> List[str]:
    global REL_TYPES

    normalized: List[str] = []
    seen: set[str] = set()
    for rel_type in rel_types:
        normalized_type = _normalize_rel_type(rel_type)
        if not normalized_type or normalized_type in seen:
            continue
        normalized.append(normalized_type)
        seen.add(normalized_type)

    if not normalized:
        normalized = list(DEFAULT_REL_TYPES)

    REL_TYPES = sorted(normalized, key=_rel_type_sort_key)
    return REL_TYPES


def _resolve_rel_types(gt_relationships: List[Dict]) -> List[str]:
    discovered = [
        _normalize_rel_type(rel.get("rel_type", ""))
        for rel in gt_relationships
        if str(rel.get("rel_type", "")).strip()
    ]
    return _set_active_rel_types(discovered or DEFAULT_REL_TYPES)


def _annotation_inventory_files(annotation_paths: Optional[List[Path]] = None) -> List[Path]:
    candidates: List[Path] = list(annotation_paths or [ANNOT_FILE])

    unique_files: List[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        unique_files.append(path)
        seen.add(resolved)
    return unique_files


def _resolve_rel_types_from_annotation_corpus(
    annotation_paths: Optional[List[Path]] = None,
) -> Tuple[List[str], List[Path]]:
    discovered: List[str] = []
    source_files = _annotation_inventory_files(annotation_paths=annotation_paths)

    for annot_path in source_files:
        with open(annot_path, encoding="utf-8") as f:
            annots = json.load(f)

        for entry in annots.values():
            for rel in entry.get("relationships", []):
                rel_type = _normalize_rel_type(rel.get("relationship_type", ""))
                if rel_type:
                    discovered.append(rel_type)
            for flag in entry.get("multi_relationship_flags", []):
                for rel_type in flag.get("relationship_types", []):
                    normalized = _normalize_rel_type(rel_type)
                    if normalized:
                        discovered.append(normalized)

    return _set_active_rel_types(discovered or DEFAULT_REL_TYPES), source_files


def _preferred_rel_type_order(rel_types: Optional[List[str]] = None) -> List[str]:
    return sorted(list(rel_types or REL_TYPES or DEFAULT_REL_TYPES), key=_rel_type_sort_key)


def _label_display_text(label: str) -> str:
    return _normalize_rel_type(label).replace("_", " ").lower()


def _label_definition(label: str) -> str:
    normalized = _normalize_rel_type(label)
    return _LABEL_DEFINITIONS.get(
        normalized,
        f"the evidence best matches the clinical relationship type '{_label_display_text(normalized)}'.",
    )

_EXPLICIT_DISCONTINUE_PATTERNS = [
    r"\bdiscontinu(?:e|ed|ation|ing)?\b",
    r"\bheld\b",
    r"\bstopp(?:ed|ing)?\b",
    r"\bstop\b",
    r"\bd/c\b",
    r"\bdc'd\b",
    r"\bwithdrawn\b",
    r"\bno longer\b",
    r"\bswitch(?:ed|ing)?\b",
    r"\bavoid(?:ed|ing)?\b",
]

_GLINER2_ACTION_PATTERNS: Dict[str, List[str]] = {
    "TREATS": [
        "start",
        "started",
        "continue",
        "continued",
        "given",
        "administered",
        "prescribed",
        "for control",
        "for pain",
        "for nausea",
        "therapy",
        "regimen",
    ],
    "ADVERSE_EFFECT": [
        "could not tolerate",
        "intoler",
        "adverse",
        "side effect",
        "medication effect",
        "caused",
        "worsen",
        "hallucination",
        "edema",
        "insomnia",
        "toxic",
    ],
    "DISCONTINUED": [
        "d/c",
        "dc'd",
        "discontinu",
        "stopped",
        "stop",
        "held",
        "withdrawn",
        "switched",
        "avoid",
    ],
    "CONTRAINDICATED": [
        "contraindicat",
        "unsafe",
        "allergy",
        "not candidate",
        "avoid due to",
    ],
    "NEGATIVE": [
        "unrelated",
        "not mentioned",
        "not linked",
        "not for",
        "no evidence",
    ],
}

_CLINICAL_NORMALIZATION_REWRITES: Dict[str, str] = {
    "af": "atrial fibrillation",
    "afib": "atrial fibrillation",
    "aki": "acute kidney injury",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "ckd": "chronic kidney disease",
    "copd": "chronic obstructive pulmonary disease",
    "dm": "diabetes mellitus",
    "htn": "hypertension",
    "mi": "myocardial infarction",
    "sob": "shortness of breath",
    "t2dm": "type 2 diabetes mellitus",
}


def _extract_row_field(row_text: str, field_name: str) -> str:
    field_name = field_name.strip().lower()
    for part in row_text.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        if key.strip().lower() == field_name:
            return value.strip().rstrip(".")
    return ""


def _build_path_signature(path: Dict) -> str:
    diagnosis = _extract_row_field(path.get("diag_row_text", ""), "diagnosis") or "Unknown diagnosis"
    medication = _extract_row_field(path.get("med_row_text", ""), "drug") or "Unknown medication"
    section = str(path.get("section_name", "Unknown section")).strip() or "Unknown section"
    sentence = " ".join(str(path.get("sent_text", "")).split()) or "No evidence sentence"
    return (
        f"Diagnosis: {diagnosis}. "
        f"Medication: {medication}. "
        f"Section: {section}. "
        f"Evidence: {sentence}"
    )


def _build_pair_signature(
    pair_paths: List[Dict],
    max_evidence: int = 3,
    max_chars_per_sentence: int = 220,
) -> str:
    # Prefer ce_score (Option C, per-pair sentence rerank) when present; fall
    # back to LOKI path_score otherwise. ce_score is None for paths that have
    # not been CE-scored, which sorts last under the float() cast below.
    ranked_paths = sorted(
        pair_paths,
        key=lambda item: (
            float(item.get("ce_score", float("-inf"))) if item.get("ce_score") is not None else float("-inf"),
            float(item.get("path_score", 0.0)),
            float(item.get("score_diag", 0.0)),
            float(item.get("score_med", 0.0)),
        ),
        reverse=True,
    )
    anchor = ranked_paths[0]
    evidence_lines = []
    seen_sentences: set[str] = set()
    for idx, path in enumerate(ranked_paths, start=1):
        sentence = " ".join(str(path.get("sent_text", "")).split())[:max_chars_per_sentence]
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        evidence_lines.append(
            f"Evidence {len(evidence_lines) + 1}: section={path.get('section_name', '')}; "
            f"sentence={sentence}; score={float(path.get('path_score', 0.0)):.4f}"
        )
        if len(evidence_lines) >= max_evidence:
            break

    evidence_block = " ".join(evidence_lines) if evidence_lines else "No evidence sentences available."
    return (
        f"Diagnosis row: {anchor.get('diag_row_text', '')}. "
        f"Medication row: {anchor.get('med_row_text', '')}. "
        f"{evidence_block}"
    )


def _build_pair_semantic_signature(
    pair_paths: List[Dict],
    max_terms: int = 6,
    max_sentences: int = 3,
    max_chars_per_sentence: int = 180,
) -> str:
    # Prefer ce_score (Option C) when present so the semantic signature is
    # built from CE-preferred sentences. Falls back to LOKI path_score.
    ranked_paths = sorted(
        pair_paths,
        key=lambda item: (
            float(item.get("ce_score", float("-inf"))) if item.get("ce_score") is not None else float("-inf"),
            float(item.get("path_score", 0.0)),
            float(item.get("score_diag", 0.0)),
            float(item.get("score_med", 0.0)),
        ),
        reverse=True,
    )

    texts: List[str] = []
    weights: List[float] = []
    seen_sentences: set[str] = set()
    for path in ranked_paths:
        sentence = " ".join(str(path.get("sent_text", "")).split())[:max_chars_per_sentence]
        if not sentence or sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        texts.append(sentence)
        weights.append(max(float(path.get("path_score", 0.0)), 0.25))
        if len(texts) >= max_sentences:
            break

    signature_terms = _extract_signature_terms_from_texts(
        texts,
        max_terms=max_terms,
        weights=weights,
    )
    if signature_terms:
        return "Semantic signature: " + "; ".join(signature_terms)
    if texts:
        return "Semantic evidence: " + " ".join(texts)
    return _build_pair_signature(
        pair_paths,
        max_evidence=min(max_sentences, 2),
        max_chars_per_sentence=max_chars_per_sentence,
    )


def _representative_cluster_sentences(
    cluster_paths: List[Dict],
    max_sentences: int = 4,
    max_chars_per_sentence: int = 220,
) -> List[str]:
    sentence_stats: Dict[int, Dict[str, float | str]] = {}
    for path in cluster_paths:
        sent_idx = int(path["sent_idx"])
        stats = sentence_stats.setdefault(
            sent_idx,
            {
                "text": str(path["sent_text"]).strip(),
                "best_score": 0.0,
                "count": 0.0,
            },
        )
        stats["count"] = float(stats["count"]) + 1.0
        stats["best_score"] = max(float(stats["best_score"]), float(path.get("path_score", 0.0)))

    ranked = sorted(
        sentence_stats.items(),
        key=lambda item: (-float(item[1]["best_score"]), -float(item[1]["count"]), item[0]),
    )
    return [
        str(stats["text"])[:max_chars_per_sentence]
        for _sent_idx, stats in ranked[:max_sentences]
    ]


def _build_gliner2_occurrence_text(path: Dict) -> str:
    medication = _extract_row_field(path.get("med_row_text", ""), "drug")
    diagnosis = _extract_row_field(path.get("diag_row_text", ""), "diagnosis")
    if not medication:
        medication = " ".join(str(path.get("med_row_text", "")).split())
    if not diagnosis:
        diagnosis = " ".join(str(path.get("diag_row_text", "")).split())

    section = " ".join(str(path.get("section_name", "")).split()) or "Unknown"
    sentence = " ".join(str(path.get("sent_text", "")).split())
    return (
        "Clinical note evidence for medication-diagnosis relationship typing.\n"
        f"Medication: {medication}\n"
        f"Diagnosis: {diagnosis}\n"
        f"Section: {section}\n"
        f"Evidence sentence: {sentence}"
    )


def _build_gliner2_semantic_signature_occurrence_text(path: Dict, pair_paths: List[Dict]) -> str:
    medication = _extract_row_field(path.get("med_row_text", ""), "drug")
    diagnosis = _extract_row_field(path.get("diag_row_text", ""), "diagnosis")
    if not medication:
        medication = " ".join(str(path.get("med_row_text", "")).split())
    if not diagnosis:
        diagnosis = " ".join(str(path.get("diag_row_text", "")).split())

    signature = _build_pair_semantic_signature(pair_paths)
    sections = sorted(
        {
            " ".join(str(item.get("section_name", "")).split())
            for item in pair_paths
            if str(item.get("section_name", "")).strip()
        }
    )
    section_text = ", ".join(sections[:3]) if sections else "Unknown"
    return (
        "Clinical semantic evidence for medication-diagnosis relationship typing.\n"
        f"Medication: {medication}\n"
        f"Diagnosis: {diagnosis}\n"
        f"Sections: {section_text}\n"
        f"Semantic evidence: {signature}"
    )


def _build_gliner2_occurrence_text_by_mode(
    path: Dict,
    pair_paths: List[Dict],
    label_input_mode: str,
) -> str:
    resolved_mode = (label_input_mode or DEFAULT_GLINER2_LABEL_INPUT_MODE).strip().lower()
    if resolved_mode == "semantic_signature":
        return _build_gliner2_semantic_signature_occurrence_text(path, pair_paths)
    return _build_gliner2_occurrence_text(path)


def _to_float_or_none(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_match_text(text: object, mode: str = "legacy") -> str:
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    resolved_mode = (mode or "legacy").strip().lower()
    if resolved_mode == "clinical_light":
        expanded_tokens: List[str] = []
        for token in tokens:
            rewrite = _CLINICAL_NORMALIZATION_REWRITES.get(token, token)
            expanded_tokens.extend(re.findall(r"[a-z0-9]+", rewrite.lower()))
        tokens = expanded_tokens
    return " ".join(tokens)


def _text_matches_anchor(text: object, anchor: object, normalization_mode: str = "legacy") -> bool:
    normalized_text = _normalize_match_text(text, mode=normalization_mode)
    normalized_anchor = _normalize_match_text(anchor, mode=normalization_mode)
    if not normalized_text or not normalized_anchor:
        return False
    if normalized_anchor in normalized_text or normalized_text in normalized_anchor:
        return True

    text_tokens = set(normalized_text.split())
    anchor_tokens = set(normalized_anchor.split())
    if not text_tokens or not anchor_tokens:
        return False

    overlap = text_tokens & anchor_tokens
    if not overlap:
        return False
    return len(overlap) >= max(1, min(len(text_tokens), len(anchor_tokens)))


def _extract_gliner2_entities(result: object, entity_type: str) -> List[Dict[str, object]]:
    if not isinstance(result, dict):
        return []
    entity_payload = result.get("entities", {})
    if not isinstance(entity_payload, dict):
        return []
    values = entity_payload.get(entity_type, [])
    if isinstance(values, str):
        return [{"text": values}]
    if not isinstance(values, list):
        return []

    entities: List[Dict[str, object]] = []
    for value in values:
        if isinstance(value, str):
            entities.append({"text": value})
        elif isinstance(value, dict):
            entities.append(value)
    return entities


def _extract_gliner2_relation_records(result: object) -> List[Tuple[str, Dict[str, object], Dict[str, object]]]:
    if not isinstance(result, dict):
        return []
    relation_payload = result.get("relation_extraction", {})
    if not isinstance(relation_payload, dict):
        return []

    records: List[Tuple[str, Dict[str, object], Dict[str, object]]] = []
    for raw_rel_type, values in relation_payload.items():
        rel_type = _normalize_rel_type(raw_rel_type)
        if isinstance(values, tuple) and len(values) == 2:
            head_text, tail_text = values
            records.append((rel_type, {"text": head_text}, {"text": tail_text}))
            continue
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, tuple) and len(value) == 2:
                head_text, tail_text = value
                records.append((rel_type, {"text": head_text}, {"text": tail_text}))
            elif isinstance(value, dict):
                head = value.get("head", {})
                tail = value.get("tail", {})
                if isinstance(head, dict) and isinstance(tail, dict):
                    records.append((rel_type, head, tail))
    return records


def _relation_record_confidence(head: Dict[str, object], tail: Dict[str, object]) -> float:
    head_conf = _to_float_or_none(head.get("confidence"))
    tail_conf = _to_float_or_none(tail.get("confidence"))
    valid = [value for value in (head_conf, tail_conf) if value is not None]
    if not valid:
        return 1.0
    return max(0.0, min(valid))


def _gliner2_action_signal_scores(action_entities: List[Dict[str, object]]) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for entity in action_entities:
        action_text = _normalize_match_text(entity.get("text", ""))
        if not action_text:
            continue
        confidence = _to_float_or_none(entity.get("confidence"))
        confidence_value = confidence if confidence is not None else 1.0
        for rel_type, patterns in _GLINER2_ACTION_PATTERNS.items():
            if any(pattern in action_text for pattern in patterns):
                scores[rel_type] += confidence_value
    return dict(scores)


def _sentence_signal_scores(sentence_text: str, candidate_labels: List[str]) -> Dict[str, float]:
    normalized_sentence = str(sentence_text).lower()
    scores: Dict[str, float] = {label: 0.0 for label in candidate_labels}
    for rel_type in candidate_labels:
        if rel_type == "DISCONTINUED":
            scores[rel_type] += float(sum(
                len(re.findall(pattern, normalized_sentence, flags=re.IGNORECASE))
                for pattern in _EXPLICIT_DISCONTINUE_PATTERNS
            ))
        for keyword in _REL_KEYWORDS.get(rel_type, []):
            scores[rel_type] += float(normalized_sentence.count(keyword))
    return scores


def _score_gliner2_hybrid_occurrence(
    path: Dict,
    entity_result: object,
    relation_result: object,
    candidate_labels: List[str],
    anchor_normalization_mode: str = "legacy",
) -> Tuple[Optional[str], Dict[str, float], Dict[str, object]]:
    medication_anchor = _extract_row_field(path.get("med_row_text", ""), "drug") or str(path.get("med_row_text", ""))
    diagnosis_anchor = _extract_row_field(path.get("diag_row_text", ""), "diagnosis") or str(path.get("diag_row_text", ""))
    sentence_text = " ".join(str(path.get("sent_text", "")).split())

    medication_entities = _extract_gliner2_entities(entity_result, "medication")
    diagnosis_entities = _extract_gliner2_entities(entity_result, "diagnosis")
    action_entities = _extract_gliner2_entities(entity_result, "action")

    matched_medications = [
        entity for entity in medication_entities
        if _text_matches_anchor(entity.get("text", ""), medication_anchor, normalization_mode=anchor_normalization_mode)
    ]
    matched_diagnoses = [
        entity for entity in diagnosis_entities
        if _text_matches_anchor(entity.get("text", ""), diagnosis_anchor, normalization_mode=anchor_normalization_mode)
    ]
    medication_matched = bool(matched_medications) or _text_matches_anchor(
        sentence_text,
        medication_anchor,
        normalization_mode=anchor_normalization_mode,
    )
    diagnosis_matched = bool(matched_diagnoses) or _text_matches_anchor(
        sentence_text,
        diagnosis_anchor,
        normalization_mode=anchor_normalization_mode,
    )

    scores: Dict[str, float] = {label: 0.0 for label in candidate_labels}
    relation_hits: List[Dict[str, object]] = []
    for rel_type, head, tail in _extract_gliner2_relation_records(relation_result):
        if rel_type not in candidate_labels:
            continue
        head_text = str(head.get("text", ""))
        tail_text = str(tail.get("text", ""))
        med_match = _text_matches_anchor(
            head_text,
            medication_anchor,
            normalization_mode=anchor_normalization_mode,
        ) or _text_matches_anchor(
            tail_text,
            medication_anchor,
            normalization_mode=anchor_normalization_mode,
        )
        diag_match = _text_matches_anchor(
            head_text,
            diagnosis_anchor,
            normalization_mode=anchor_normalization_mode,
        ) or _text_matches_anchor(
            tail_text,
            diagnosis_anchor,
            normalization_mode=anchor_normalization_mode,
        )
        if not (med_match and diag_match):
            continue

        confidence = _relation_record_confidence(head, tail)
        relation_weight = confidence
        if rel_type == "ADVERSE_EFFECT":
            relation_weight *= 1.35
        elif rel_type == "DISCONTINUED":
            relation_weight *= 1.2
        scores[rel_type] += relation_weight
        relation_hits.append({
            "type": rel_type,
            "head": head_text,
            "tail": tail_text,
            "confidence": confidence,
        })

    action_scores = _gliner2_action_signal_scores(action_entities)
    for rel_type, action_score in action_scores.items():
        if rel_type not in scores:
            continue
        if rel_type == "TREATS":
            scores[rel_type] += 0.7 * action_score
        elif rel_type == "DISCONTINUED":
            scores[rel_type] += 1.3 * action_score
        else:
            scores[rel_type] += 1.15 * action_score

    lexical_scores = _sentence_signal_scores(sentence_text, candidate_labels)
    for rel_type, lexical_score in lexical_scores.items():
        if rel_type not in scores or lexical_score <= 0.0:
            continue
        if rel_type == "TREATS":
            scores[rel_type] += 0.2 * lexical_score
        elif rel_type == "DISCONTINUED":
            scores[rel_type] += 0.45 * lexical_score
        else:
            scores[rel_type] += 0.35 * lexical_score

    adverse_or_stop_support = scores.get("ADVERSE_EFFECT", 0.0) + scores.get("DISCONTINUED", 0.0)
    if adverse_or_stop_support > 0.0 and "TREATS" in scores:
        strong_negative_signal = (
            action_scores.get("ADVERSE_EFFECT", 0.0) > 0.0
            or action_scores.get("DISCONTINUED", 0.0) > 0.0
            or lexical_scores.get("ADVERSE_EFFECT", 0.0) > 0.0
            or lexical_scores.get("DISCONTINUED", 0.0) > 0.0
        )
        if strong_negative_signal:
            scores["TREATS"] *= 0.25

    anchor_labels = ("TREATS", "ADVERSE_EFFECT", "DISCONTINUED", "CONTRAINDICATED")
    if not medication_matched:
        for rel_type in anchor_labels:
            if rel_type in scores:
                scores[rel_type] *= 0.5
    if not diagnosis_matched:
        for rel_type in anchor_labels:
            if rel_type in scores:
                scores[rel_type] *= 0.5

    positive_scores = {label: score for label, score in scores.items() if score > 0.0}
    if not positive_scores:
        return None, scores, {
            "medication_matched": medication_matched,
            "diagnosis_matched": diagnosis_matched,
            "action_signals": action_scores,
            "lexical_signals": lexical_scores,
            "relation_hits": relation_hits,
        }

    tie_break_order = _preferred_rel_type_order(list(scores))
    predicted_label = min(
        tie_break_order,
        key=lambda rel_type: (-scores.get(rel_type, 0.0), tie_break_order.index(rel_type)),
    )
    return predicted_label, scores, {
        "medication_matched": medication_matched,
        "diagnosis_matched": diagnosis_matched,
        "action_signals": action_scores,
        "lexical_signals": lexical_scores,
        "relation_hits": relation_hits,
        "matched_medications": [str(entity.get("text", "")) for entity in matched_medications],
        "matched_diagnoses": [str(entity.get("text", "")) for entity in matched_diagnoses],
    }


def _build_supporting_evidence(
    cluster_paths: List[Dict],
    evidence_records: Optional[List[Dict[str, object]]] = None,
    max_items: int = 3,
) -> List[Dict[str, object]]:
    if evidence_records:
        ranked_records = sorted(
            evidence_records,
            key=lambda record: (
                -float(record.get("vote_weight", 0.0)),
                -float(record.get("path_score", 0.0)),
                int(record.get("sent_idx", -1)),
            ),
        )
        trimmed_records: List[Dict[str, object]] = []
        for record in ranked_records[:max_items]:
            trimmed_records.append({
                "sent_idx": int(record.get("sent_idx", -1)),
                "section_name": str(record.get("section_name", "")),
                "sentence": str(record.get("sentence", ""))[:220],
                "label": str(record.get("label", "")),
                "confidence": _to_float_or_none(record.get("confidence")),
                "vote_weight": float(record.get("vote_weight", 0.0)),
                "path_score": float(record.get("path_score", 0.0)),
            })
        return trimmed_records

    ranked_paths = sorted(
        cluster_paths,
        key=lambda path: (
            float(path.get("path_score", 0.0)),
            float(path.get("score_diag", 0.0)),
            float(path.get("score_med", 0.0)),
        ),
        reverse=True,
    )
    trimmed_paths: List[Dict[str, object]] = []
    for path in ranked_paths[:max_items]:
        trimmed_paths.append({
            "sent_idx": int(path.get("sent_idx", -1)),
            "section_name": str(path.get("section_name", "")),
            "sentence": " ".join(str(path.get("sent_text", "")).split())[:220],
            "path_score": float(path.get("path_score", 0.0)),
        })
    return trimmed_paths


def _load_gliner2_model(model_name: str):
    cached_model = _GLINER2_MODEL_CACHE.get(model_name)
    if cached_model is not None:
        return cached_model

    from gliner2 import GLiNER2  # type: ignore

    resolved_model_name, _model_source = ensure_repo_local_hf_snapshot(model_name, allow_online=True)
    model = GLiNER2.from_pretrained(resolved_model_name, map_location=DEVICE.type)
    _GLINER2_MODEL_CACHE[model_name] = model
    return model


def _keyword_scores(
    cluster_paths: List[Dict],
    candidate_labels: Optional[List[str]] = None,
) -> Dict[str, int]:
    combined = " ".join(sent.lower() for sent in _representative_cluster_sentences(
        cluster_paths,
        max_sentences=6,
        max_chars_per_sentence=300,
    ))
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    return {
        rel_type: sum(combined.count(keyword) for keyword in _REL_KEYWORDS.get(rel_type, []))
        for rel_type in resolved_candidate_labels
    }


def _best_scored_label(scores: Dict[str, int], tie_break_order: List[str]) -> str:
    if not scores:
        raise ValueError("scores must be non-empty")
    best_score = max(scores.values())
    tied = {label for label, score in scores.items() if score == best_score}
    for label in tie_break_order:
        if label in tied:
            return label
    return next(iter(scores))


def _normalize_section_key(section_name: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(section_name).lower()))


def _resolve_optional_input_path(path_text: str) -> Optional[Path]:
    raw_path = str(path_text or "").strip()
    if not raw_path:
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate

    search_roots = [Path.cwd(), WORKSPACE_ROOT, Path(__file__).parent]
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return search_roots[0] / candidate


def _load_section_priors(priors_file: str) -> Tuple[Dict[str, float], Optional[Path]]:
    resolved_path = _resolve_optional_input_path(priors_file)
    if resolved_path is None:
        return {}, None
    if not resolved_path.exists():
        raise FileNotFoundError(f"Section priors file not found: {resolved_path}")

    with open(resolved_path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("section_priors"), dict):
        payload = payload["section_priors"]
    if not isinstance(payload, dict):
        raise ValueError("Section priors JSON must be an object mapping section names to numeric weights")

    section_priors: Dict[str, float] = {}
    for raw_section_name, raw_weight in payload.items():
        section_key = _normalize_section_key(raw_section_name)
        if not section_key:
            continue
        section_priors[section_key] = max(0.0, float(raw_weight))

    return section_priors, resolved_path


def _section_prior_weight(section_name: object, section_priors: Optional[Dict[str, float]] = None) -> float:
    if not section_priors:
        return 1.0
    section_key = _normalize_section_key(section_name)
    if not section_key:
        return 1.0
    return max(0.0, float(section_priors.get(section_key, 1.0)))


def _sentence_specificity_weight(sentence_fanout: int, alpha: float) -> float:
    resolved_alpha = max(float(alpha), 0.0)
    resolved_fanout = max(int(sentence_fanout), 1)
    if resolved_alpha <= 0.0:
        return 1.0
    return 1.0 / (1.0 + resolved_alpha * max(resolved_fanout - 1, 0))


def _stage5_sentence_rank_weight(
    sent_idx: int,
    sentence_support: int,
    sent_meta: Optional[Dict[int, str]] = None,
    sentence_specificity_alpha: float = 0.0,
    section_priors: Optional[Dict[str, float]] = None,
    stopcue_sentence_ids: Optional[Set[int]] = None,
    stopcue_boost: float = 1.0,
) -> float:
    resolved_alpha = max(float(sentence_specificity_alpha), 0.0)
    if resolved_alpha <= 0.0:
        specificity_weight = 1.0
    else:
        # Stage 5 ranking should prefer more specific sentences without overwhelming
        # the raw row-sentence score that still carries most of the retrieval signal.
        specificity_target = _sentence_specificity_weight(sentence_support, 1.0)
        blend = resolved_alpha / (1.0 + resolved_alpha)
        specificity_weight = (1.0 - blend) + (blend * specificity_target)
    section_weight = _section_prior_weight((sent_meta or {}).get(int(sent_idx), ""), section_priors)
    stopcue_weight = max(float(stopcue_boost), 1.0) if int(sent_idx) in (stopcue_sentence_ids or set()) else 1.0
    return specificity_weight * section_weight * stopcue_weight


def _support_weighted_pair_connection_score(
    pair_paths: List[Dict],
    sentence_pair_members: Optional[Dict[int, set]] = None,
    sentence_specificity_alpha: float = 0.0,
    section_priors: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    best_path_by_sentence: Dict[int, Dict] = {}
    for path in pair_paths:
        sent_idx = int(path["sent_idx"])
        current_best = best_path_by_sentence.get(sent_idx)
        if current_best is None or float(path.get("path_score", 0.0)) > float(current_best.get("path_score", 0.0)):
            best_path_by_sentence[sent_idx] = path

    connection_score = 0.0
    best_path_score = 0.0
    max_sentence_fanout = 0
    for sent_idx, path in best_path_by_sentence.items():
        path_score = max(float(path.get("path_score", 0.0)), 0.0)
        sentence_fanout = 1
        if sentence_pair_members is not None:
            sentence_fanout = max(len(sentence_pair_members.get(sent_idx, set())), 1)
        specificity_weight = _sentence_specificity_weight(sentence_fanout, sentence_specificity_alpha)
        section_weight = _section_prior_weight(path.get("section_name", ""), section_priors)
        connection_score += path_score * specificity_weight * section_weight
        best_path_score = max(best_path_score, path_score)
        max_sentence_fanout = max(max_sentence_fanout, sentence_fanout)

    return {
        "connection_score": connection_score,
        "best_path_score": best_path_score,
        "n_unique_sentences": len(best_path_by_sentence),
        "max_sentence_fanout": max_sentence_fanout,
    }


def _cluster_signal_strength(cluster_paths: List[Dict]) -> int:
    scores = _keyword_scores(cluster_paths)
    strongest = max(scores.values()) if scores else 0
    if strongest > 0:
        return strongest

    if _explicit_discontinue_hits(cluster_paths) > 0:
        return 1

    unique_pairs = len({(p["diag_row_idx"], p["med_row_idx"]) for p in cluster_paths})
    unique_sentences = len({int(p["sent_idx"]) for p in cluster_paths})
    return 1 if unique_pairs >= 2 or unique_sentences >= 2 else 0


def _build_low_signal_bundle_rescue_detail(
    label: str,
    cluster_paths: List[Dict],
    member_cluster_ids: List[int],
    cluster_label_details: Optional[Dict[int, Dict[str, object]]] = None,
    candidate_labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    label_scores: Dict[str, float] = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
    label_counts: Dict[str, float] = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
    backends: List[str] = []
    supporting_evidence_records: List[Dict[str, object]] = []
    n_scored_occurrences = 0

    for cluster_id in member_cluster_ids:
        detail = (cluster_label_details or {}).get(int(cluster_id), {}) or {}
        backend_name = str(detail.get("backend", "")).strip()
        if backend_name and backend_name not in backends:
            backends.append(backend_name)

        raw_scores = detail.get("label_scores")
        if isinstance(raw_scores, dict):
            for rel_type, raw_score in raw_scores.items():
                rel_key = _normalize_rel_type(str(rel_type))
                if rel_key not in label_scores:
                    continue
                try:
                    label_scores[rel_key] += float(raw_score)
                except (TypeError, ValueError):
                    continue

        raw_counts = detail.get("label_counts")
        if isinstance(raw_counts, dict):
            for rel_type, raw_count in raw_counts.items():
                rel_key = _normalize_rel_type(str(rel_type))
                if rel_key not in label_counts:
                    continue
                try:
                    label_counts[rel_key] += float(raw_count)
                except (TypeError, ValueError):
                    continue

        supporting_evidence = detail.get("supporting_evidence")
        if isinstance(supporting_evidence, list):
            supporting_evidence_records.extend(
                record for record in supporting_evidence if isinstance(record, dict)
            )

        try:
            n_scored_occurrences += int(detail.get("n_scored_occurrences", 0) or 0)
        except (TypeError, ValueError):
            pass

    normalized_label = _normalize_rel_type(str(label)) or str(label)
    if normalized_label in label_scores and max(label_scores.values(), default=0.0) <= 0.0:
        label_scores[normalized_label] = float(max(len(member_cluster_ids), 1))
    if normalized_label in label_counts and max(label_counts.values(), default=0.0) <= 0.0:
        label_counts[normalized_label] = float(max(len(member_cluster_ids), 1))

    return {
        "backend": "+".join(backends) if backends else "low_signal_bundle_rescue",
        "label_source": "low_signal_bundle_rescue",
        "score_type": "cluster_bundle_votes",
        "label_input_mode": "cluster_bundle",
        "label_scores": {rel_type: round(score, 4) for rel_type, score in label_scores.items()},
        "label_counts": {rel_type: round(count, 4) for rel_type, count in label_counts.items()},
        "fallback_reason": None,
        "n_occurrences": len(cluster_paths),
        "n_unique_sentences": len({int(path["sent_idx"]) for path in cluster_paths}),
        "n_scored_occurrences": max(n_scored_occurrences, len(cluster_paths)),
        "supporting_evidence": _build_supporting_evidence(
            cluster_paths,
            evidence_records=supporting_evidence_records,
        ),
        "low_signal_rescue_mode": "sibling_bundle",
        "low_signal_rescue_member_cluster_ids": [int(cluster_id) for cluster_id in member_cluster_ids],
    }


def _rescue_low_signal_cluster_bundles(
    paths: List[Dict],
    labels: np.ndarray,
    clusters: Dict[int, List[Dict]],
    cluster_name_map: Dict[int, str],
    cluster_label_details: Dict[int, Dict[str, object]],
    cluster_connection_signals: Dict[int, Dict[str, object]],
    cluster_pair_label_refinement_stats: Optional[Dict[str, object]] = None,
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[int, List[Dict]], Dict[int, str], Dict[int, Dict[str, object]], Dict[str, object]]:
    stats: Dict[str, object] = {
        "enabled": True,
        "candidate_singletons": 0,
        "vetoed_clusters": [],
        "bundle_groups": [],
        "rescued_cluster_ids": [],
        "rescued_member_clusters": [],
        "reason": "no_dropped_clusters",
    }
    if not paths or not clusters or len(labels) != len(paths):
        stats["reason"] = "no_paths"
        return labels, clusters, cluster_name_map, cluster_label_details, stats

    dropped_cluster_ids = [
        int(cluster_id)
        for cluster_id, signal in sorted(cluster_connection_signals.items())
        if not bool(signal.get("keep", False))
    ]
    if not dropped_cluster_ids:
        return labels, clusters, cluster_name_map, cluster_label_details, stats

    bundle_groups: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)
    vetoed_cluster_records: List[Dict[str, object]] = []
    negative_dominant_parent_labels: Dict[int, str] = {}
    split_clusters = (cluster_pair_label_refinement_stats or {}).get("split_clusters")
    if isinstance(split_clusters, list):
        for split_record in split_clusters:
            if not isinstance(split_record, dict):
                continue
            try:
                parent_cluster_id = int(split_record.get("parent_cluster_id", -1))
            except (TypeError, ValueError):
                continue
            dominant_label = ""
            children = split_record.get("children")
            if isinstance(children, list):
                for child in children:
                    if not isinstance(child, dict):
                        continue
                    try:
                        child_cluster_id = int(child.get("cluster_id", -1))
                    except (TypeError, ValueError):
                        continue
                    if child_cluster_id == parent_cluster_id:
                        dominant_label = _normalize_rel_type(str(child.get("label", "")))
                        break
            if dominant_label:
                negative_dominant_parent_labels[parent_cluster_id] = dominant_label

    for cluster_id in dropped_cluster_ids:
        cluster_paths = list(clusters.get(int(cluster_id), []) or [])
        signal = cluster_connection_signals.get(int(cluster_id), {}) or {}
        if len(cluster_paths) != 1:
            continue
        if int(signal.get("n_pairs", 0) or 0) != 1:
            continue
        if int(signal.get("n_paths", 0) or 0) != 1:
            continue
        if int(signal.get("n_unique_sentences", 0) or 0) != 1:
            continue

        stats["candidate_singletons"] = int(stats.get("candidate_singletons", 0)) + 1
        label = _normalize_rel_type(str(cluster_name_map.get(int(cluster_id), "")))
        if not label or label == "NEGATIVE":
            continue

        detail = cluster_label_details.get(int(cluster_id), {}) or {}
        parent_cluster_id = detail.get("refinement_parent_cluster_id")
        split_mode = str(detail.get("pair_label_refinement_split_mode", "")).strip().lower()
        if label == "TREATS" and parent_cluster_id is not None and split_mode in {"pair_label", "path_label"}:
            try:
                resolved_parent_cluster_id = int(parent_cluster_id)
            except (TypeError, ValueError):
                resolved_parent_cluster_id = None
            if resolved_parent_cluster_id is not None:
                parent_detail = cluster_label_details.get(resolved_parent_cluster_id, {}) or {}
                parent_dominant_label = _normalize_rel_type(str(detail.get("pair_label_refinement_path_dominant_label", "")))
                if not parent_dominant_label:
                    parent_dominant_label = _normalize_rel_type(
                        str(
                            parent_detail.get(
                                "pair_label_refinement_dominant_label",
                                cluster_name_map.get(resolved_parent_cluster_id, ""),
                            )
                        )
                    )
                if not parent_dominant_label:
                    parent_dominant_label = _normalize_rel_type(
                        str(negative_dominant_parent_labels.get(resolved_parent_cluster_id, ""))
                    )
                if parent_dominant_label == "NEGATIVE":
                    vetoed_cluster_records.append({
                        "cluster_id": int(cluster_id),
                        "reason": "negative_dominant_parent_treats_singleton",
                        "parent_cluster_id": int(resolved_parent_cluster_id),
                    })
                    continue

        path = cluster_paths[0]
        group_key = (
            int(path.get("diag_row_idx", -1)),
            int(path.get("sent_idx", -1)),
            label,
        )
        bundle_groups[group_key].append(int(cluster_id))

    stats["vetoed_clusters"] = vetoed_cluster_records
    rescue_groups = [
        (group_key, sorted(member_cluster_ids))
        for group_key, member_cluster_ids in sorted(bundle_groups.items())
        if len(member_cluster_ids) >= 2
    ]
    if not rescue_groups:
        stats["reason"] = "no_bundle_groups"
        return labels, clusters, cluster_name_map, cluster_label_details, stats

    member_to_rescued_cluster: Dict[int, int] = {}
    rescue_group_members: Dict[int, List[int]] = {}
    rescued_cluster_names: Dict[int, str] = {}
    rescued_cluster_details: Dict[int, Dict[str, object]] = {}
    bundle_group_records: List[Dict[str, object]] = []
    next_cluster_id = (max(int(cluster_id) for cluster_id in clusters) + 1) if clusters else 0

    for group_key, member_cluster_ids in rescue_groups:
        rescued_cluster_id = next_cluster_id
        next_cluster_id += 1
        diag_row_idx, sent_idx, label = group_key
        rescued_cluster_paths: List[Dict] = []
        rescue_group_members[int(rescued_cluster_id)] = [int(cluster_id) for cluster_id in member_cluster_ids]
        for member_cluster_id in member_cluster_ids:
            member_to_rescued_cluster[int(member_cluster_id)] = int(rescued_cluster_id)
            rescued_cluster_paths.extend(list(clusters.get(int(member_cluster_id), []) or []))
        rescued_cluster_names[int(rescued_cluster_id)] = str(label)
        rescued_cluster_details[int(rescued_cluster_id)] = _build_low_signal_bundle_rescue_detail(
            str(label),
            rescued_cluster_paths,
            member_cluster_ids,
            cluster_label_details=cluster_label_details,
            candidate_labels=candidate_labels,
        )
        rescued_cluster_details[int(rescued_cluster_id)]["low_signal_rescue_group_key"] = {
            "diag_row_idx": int(diag_row_idx),
            "sent_idx": int(sent_idx),
            "label": str(label),
        }
        bundle_group_records.append({
            "rescued_cluster_id": int(rescued_cluster_id),
            "member_cluster_ids": [int(cluster_id) for cluster_id in member_cluster_ids],
            "diag_row_idx": int(diag_row_idx),
            "sent_idx": int(sent_idx),
            "label": str(label),
        })

    updated_labels: List[int] = []
    for path, lbl in zip(paths, labels):
        original_cluster_id = int(lbl)
        rescued_cluster_id = member_to_rescued_cluster.get(original_cluster_id, original_cluster_id)
        updated_labels.append(int(rescued_cluster_id))
        if rescued_cluster_id != original_cluster_id:
            path["raw_cluster_id"] = int(rescued_cluster_id)
            path["low_signal_bundle_rescued"] = True
            path["low_signal_bundle_member_cluster_ids"] = list(rescue_group_members.get(int(rescued_cluster_id), []))

    labels = np.asarray(updated_labels, dtype=int)
    updated_clusters: Dict[int, List[Dict]] = defaultdict(list)
    for path, lbl in zip(paths, labels):
        updated_clusters[int(lbl)].append(path)

    updated_cluster_name_map = {
        int(cluster_id): name
        for cluster_id, name in cluster_name_map.items()
        if int(cluster_id) not in member_to_rescued_cluster
    }
    updated_cluster_label_details = {
        int(cluster_id): detail
        for cluster_id, detail in cluster_label_details.items()
        if int(cluster_id) not in member_to_rescued_cluster
    }
    updated_cluster_name_map.update(rescued_cluster_names)
    updated_cluster_label_details.update(rescued_cluster_details)

    stats["bundle_groups"] = bundle_group_records
    stats["rescued_cluster_ids"] = [int(record["rescued_cluster_id"]) for record in bundle_group_records]
    stats["rescued_member_clusters"] = sorted(int(cluster_id) for cluster_id in member_to_rescued_cluster)
    stats["reason"] = "applied"
    return labels, dict(updated_clusters), updated_cluster_name_map, updated_cluster_label_details, stats


def _cluster_connection_filter_signal(
    cluster_paths: List[Dict],
    gamma: float,
    mode: str = "legacy",
    sentence_specificity_alpha: float = 0.0,
    section_priors: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    resolved_mode = (mode or "legacy").strip().lower()
    if resolved_mode not in {"legacy", "support_weighted"}:
        raise ValueError(f"Unsupported pair connection mode: {mode}")

    unique_pairs = len({(p["diag_row_idx"], p["med_row_idx"]) for p in cluster_paths})
    unique_sentences = len({int(p["sent_idx"]) for p in cluster_paths})
    base_signal = {
        "mode": resolved_mode,
        "n_pairs": unique_pairs,
        "n_paths": len(cluster_paths),
        "n_unique_sentences": unique_sentences,
    }
    if not cluster_paths:
        return {
            **base_signal,
            "keep": False,
            "connection_score": 0.0,
            "threshold": round(float(gamma), 4),
        }

    if resolved_mode == "legacy":
        signal_strength = _cluster_signal_strength(cluster_paths)
        return {
            **base_signal,
            "keep": signal_strength > 0,
            "signal_strength": signal_strength,
            "connection_score": float(signal_strength),
            "threshold": 0.0,
        }

    pair_buckets = _bucket_paths_by_pair(cluster_paths)
    sentence_pair_members: Dict[int, set] = defaultdict(set)
    for pair, pair_paths in pair_buckets.items():
        for sent_idx in {int(path["sent_idx"]) for path in pair_paths}:
            sentence_pair_members[sent_idx].add(pair)

    best_pair: Optional[Tuple[int, int]] = None
    best_pair_details: Optional[Dict[str, object]] = None
    for pair, pair_paths in pair_buckets.items():
        details = _support_weighted_pair_connection_score(
            pair_paths,
            sentence_pair_members=sentence_pair_members,
            sentence_specificity_alpha=sentence_specificity_alpha,
            section_priors=section_priors,
        )
        if best_pair_details is None:
            best_pair = pair
            best_pair_details = details
            continue
        candidate_key = (
            float(details.get("connection_score", 0.0)),
            float(details.get("best_path_score", 0.0)),
            int(details.get("n_unique_sentences", 0)),
        )
        best_key = (
            float(best_pair_details.get("connection_score", 0.0)),
            float(best_pair_details.get("best_path_score", 0.0)),
            int(best_pair_details.get("n_unique_sentences", 0)),
        )
        if candidate_key > best_key:
            best_pair = pair
            best_pair_details = details

    resolved_best_pair = best_pair if best_pair is not None else (-1, -1)
    resolved_details = best_pair_details or {
        "connection_score": 0.0,
        "best_path_score": 0.0,
        "n_unique_sentences": 0,
        "max_sentence_fanout": 0,
    }
    connection_score = float(resolved_details.get("connection_score", 0.0))
    return {
        **base_signal,
        "keep": connection_score >= float(gamma),
        "connection_score": round(connection_score, 4),
        "threshold": round(float(gamma), 4),
        "sentence_specificity_alpha": round(max(float(sentence_specificity_alpha), 0.0), 4),
        "best_pair": {
            "diag_row_idx": int(resolved_best_pair[0]),
            "med_row_idx": int(resolved_best_pair[1]),
            "connection_score": round(connection_score, 4),
            "best_path_score": round(float(resolved_details.get("best_path_score", 0.0)), 4),
            "n_unique_sentences": int(resolved_details.get("n_unique_sentences", 0)),
            "max_sentence_fanout": int(resolved_details.get("max_sentence_fanout", 0)),
        },
    }


def _keyword_classify(
    cluster_paths: List[Dict],
    candidate_labels: Optional[List[str]] = None,
) -> str:
    """Classify cluster into one of REL_TYPES using keyword matching on sentences."""
    tie_break_order = _resolve_candidate_labels(candidate_labels)
    scores = _keyword_scores(cluster_paths, candidate_labels=tie_break_order)
    best = _best_scored_label(scores, tie_break_order)
    return best if scores.get(best, 0) > 0 else tie_break_order[0]


def _cluster_evidence_text(
    cluster_paths: List[Dict],
    max_sentences: int = 8,
    max_chars_per_sentence: int = 320,
) -> str:
    evidence_sentences = _representative_cluster_sentences(
        cluster_paths,
        max_sentences=max_sentences,
        max_chars_per_sentence=max_chars_per_sentence,
    )
    return " ".join(sentence.lower() for sentence in evidence_sentences)


def _explicit_discontinue_hits(cluster_paths: List[Dict]) -> int:
    text = _cluster_evidence_text(cluster_paths)
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in _EXPLICIT_DISCONTINUE_PATTERNS)


def label_clusters_with_keyword(
    clusters: Dict[int, List[Dict]],
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    if not clusters:
        return {}, {}

    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    cluster_labels: Dict[int, str] = {}
    cluster_details: Dict[int, Dict[str, object]] = {}

    print("  Using keyword cluster labeler")
    for cid, cpaths in sorted(clusters.items()):
        unique_sent_count = len({int(path["sent_idx"]) for path in cpaths})
        keyword_scores = {
            label: float(score)
            for label, score in _keyword_scores(cpaths, candidate_labels=resolved_candidate_labels).items()
        }
        label = _keyword_classify(cpaths, candidate_labels=resolved_candidate_labels)
        cluster_labels[cid] = label
        cluster_details[cid] = {
            "backend": "keyword",
            "label_source": "keyword",
            "score_type": "keyword_counts",
            "label_scores": {
                rel_type: float(keyword_scores.get(rel_type, 0.0))
                for rel_type in resolved_candidate_labels
            },
            "n_occurrences": len(cpaths),
            "n_unique_sentences": unique_sent_count,
            "supporting_evidence": _build_supporting_evidence(cpaths),
        }
        print(
            f"  Cluster {cid:3d}  ({unique_sent_count:2d} sents / {len(cpaths):3d} paths)  "
            f"-> {label} [keyword]"
        )

    return cluster_labels, cluster_details


def _build_cluster_evidence_pool(
    cluster_paths: List[Dict],
    all_paths: List[Dict],
    hub_fanout_threshold: float = 0.3,
    max_pool_size: int = 12,
    sent_meta: Optional[Dict[int, str]] = None,
    section_priors: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Return up to max_pool_size hub-filtered paths from cluster_paths.

    Hub sentences are those mediating more than max(2, threshold * n_cluster_pairs)
    distinct row-pairs globally across all_paths. Filtering them prevents generic
    discharge notes from dominating cluster label votes. Survivors are ranked by
    path_score * sentence rank weight; top max_pool_size are returned.
    """
    if not cluster_paths:
        return cluster_paths

    # Global per-sentence fanout: distinct (diag, med) pairs across all paths
    global_fanout: Dict[int, int] = {}
    if all_paths:
        _sent_pairs: Dict[int, set] = defaultdict(set)
        for path in all_paths:
            _sent_pairs[int(path["sent_idx"])].add(
                (int(path["diag_row_idx"]), int(path["med_row_idx"]))
            )
        global_fanout = {s: len(pairs) for s, pairs in _sent_pairs.items()}

    n_cluster_pairs = len(
        {(int(p["diag_row_idx"]), int(p["med_row_idx"])) for p in cluster_paths}
    )
    hub_threshold = max(2, int(hub_fanout_threshold * max(n_cluster_pairs, 1)))

    filtered = [
        path for path in cluster_paths
        if global_fanout.get(int(path["sent_idx"]), 0) <= hub_threshold
    ] or cluster_paths  # fallback: never drop everything

    def _pool_weight(path: Dict) -> float:
        sent_idx = int(path["sent_idx"])
        fanout = global_fanout.get(sent_idx, 1)
        rank_w = _stage5_sentence_rank_weight(sent_idx, fanout, sent_meta, 0.0, section_priors)
        return float(path.get("path_score", 0.0)) * rank_w

    return sorted(filtered, key=_pool_weight, reverse=True)[:max_pool_size]


# =============================================================================
# LMStudio (local LLM) cluster labeling backend
# =============================================================================

def _build_lmstudio_system_prompt(candidate_labels: List[str]) -> str:
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    label_list = " | ".join(resolved_candidate_labels)
    label_definitions = "\n".join(
        f"- {label}: {_label_definition(label)}"
        for label in resolved_candidate_labels
    )

    if "TREATS" in resolved_candidate_labels:
        decision_rules = (
            "3. Default assumption: if a medication appears in a patient's record alongside a diagnosis, "
            "assume it is related (TREATS) unless the evidence clearly says otherwise. "
            "Discharge medication lists and co-occurrence in the same note are sufficient grounds for TREATS.\n"
            "4. Use NEGATIVE ONLY when the evidence explicitly documents that the medication is for a "
            "completely different, named condition unrelated to this diagnosis. "
            "Absence of explicit mention is NOT grounds for NEGATIVE - use TREATS when uncertain.\n"
            "5. Do not output any label that is not in the valid label list above.\n\n"
        )
    else:
        decision_rules = (
            "3. Focus ONLY on whether the evidence documents an adverse effect or an explicit "
            "stop/hold/avoid/switch decision.\n"
            "4. Medication lists, continuation, routine co-occurrence, or general treatment use are NOT "
            "evidence for ADVERSE_EFFECT or DISCONTINUED.\n"
            "5. If NEGATIVE is in the valid label list and the evidence does not clearly support one of the "
            "positive labels, choose NEGATIVE.\n"
            "6. Do not output TREATS or CONTRAINDICATED unless they appear in the valid label list above.\n\n"
        )

    examples: List[str] = []
    if "TREATS" in resolved_candidate_labels:
        examples.extend([
            "Diagnosis: essential hypertension\n"
            "Evidence: Lisinopril was started for blood pressure control. Patient continued on ACE inhibitor therapy.\n"
            "Medication: lisinopril 10 mg daily\n"
            "Reasoning: The medication is prescribed to treat the patient's hypertension. <LABEL>TREATS</LABEL>",
            "Diagnosis: COPD\n"
            "Evidence: Fluticasone-Salmeterol 250-50 mcg Inhalation BID. Medications on Admission.\n"
            "Medication: fluticasone-salmeterol inhaler\n"
            "Reasoning: Inhaled corticosteroid/LABA combination is a standard COPD maintenance therapy; co-occurrence in discharge list is sufficient. <LABEL>TREATS</LABEL>",
        ])
    if "ADVERSE_EFFECT" in resolved_candidate_labels:
        examples.append(
            "Diagnosis: acute kidney injury\n"
            "Evidence: Renal function declined after NSAID administration. NSAIDs held due to nephrotoxicity.\n"
            "Medication: ibuprofen 400 mg PRN\n"
            "Reasoning: NSAID use caused the kidney injury as a documented adverse effect. <LABEL>ADVERSE_EFFECT</LABEL>"
        )
    if "DISCONTINUED" in resolved_candidate_labels:
        examples.append(
            "Diagnosis: hypertension\n"
            "Evidence: Metoprolol was discontinued secondary to bradycardia. Medication stopped on admission.\n"
            "Medication: metoprolol succinate 25 mg\n"
            "Reasoning: The medication was explicitly stopped during this admission. <LABEL>DISCONTINUED</LABEL>"
        )
    if "NEGATIVE" in resolved_candidate_labels:
        if "TREATS" in resolved_candidate_labels:
            examples.append(
                "Diagnosis: vaginal disorder\n"
                "Evidence: Duloxetine used for chronic pain management. Continued at discharge.\n"
                "Medication: duloxetine 60 mg daily\n"
                "Reasoning: The note explicitly states duloxetine is for chronic pain, a different named condition; no link to the vaginal disorder. <LABEL>NEGATIVE</LABEL>"
            )
        else:
            examples.append(
                "Diagnosis: acute kidney injury\n"
                "Evidence: Home medications reviewed. Lisinopril 10 mg daily was continued on admission. No side effect, hold, stop, or avoidance was documented.\n"
                "Medication: lisinopril 10 mg daily\n"
                "Reasoning: The evidence only shows routine medication continuation and does not document an adverse effect or discontinuation. <LABEL>NEGATIVE</LABEL>"
            )

    examples_text = "\n---\n".join(examples)
    if examples_text:
        examples_text = f"Examples:\n---\n{examples_text}\n---"

    return (
        "You are a clinical NLP assistant specializing in medication-diagnosis relationship "
        "typing in hospital discharge records.\n"
        "Task: Given a diagnosis, evidence sentences from a clinical note, and a medication, "
        "choose the single most appropriate relationship label from the closed set below. "
        "You MUST choose exactly one label from this set and output it inside <LABEL> tags.\n\n"
        f"Valid labels: {label_list}\n"
        f"Label definitions:\n{label_definitions}\n\n"
        "Instructions:\n"
        "1. Write one sentence of clinical reasoning.\n"
        "2. Output the chosen label inside <LABEL> tags on the same or next line.\n"
        f"{decision_rules}"
        f"{examples_text}"
    )


def _build_lmstudio_freeform_system_prompt(candidate_labels: Optional[List[str]] = None) -> str:
    """System prompt for per-path free-form relationship description (Phase 1 of agglomerative mode)."""
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    if "TREATS" not in resolved_candidate_labels:
        return (
            "You are a clinical NLP assistant. Given a diagnosis, a sentence from a clinical note, "
            "and a medication, describe the clinical relationship between them in a short phrase of "
            "3 to 8 words. Focus on whether the medication caused the problem, or was stopped, held, "
            "avoided, or switched away from. If neither is supported, say 'no adverse effect or discontinuation evidence'.\n"
            "Examples: 'caused acute kidney injury', 'discontinued due to severe rash', "
            "'held for hypotension', 'no adverse effect or discontinuation evidence'.\n"
            "Output ONLY the phrase, nothing else."
        )
    return (
        "You are a clinical NLP assistant. Given a diagnosis, a sentence from a clinical note, "
        "and a medication, describe the clinical relationship between them in a short phrase of "
        "3 to 8 words. Be specific and clinical. Avoid generic phrases like 'related to' or "
        "'associated with'. Examples: 'prescribed to treat hypertension', "
        "'caused acute kidney injury', 'discontinued due to severe rash', "
        "'contraindicated in renal failure'.\n"
        "Output ONLY the phrase, nothing else."
    )


def _build_lmstudio_freeform_user_message(path: Dict) -> str:
    """Build the user message for free-form per-path description."""
    diag = _extract_row_field(path.get("diag_row_text", ""), "diagnosis") or " ".join(
        str(path.get("diag_row_text", "")).split()
    )
    med = _extract_row_field(path.get("med_row_text", ""), "drug") or " ".join(
        str(path.get("med_row_text", "")).split()
    )
    sent = str(path.get("sent_text", "")).strip()[:300]
    return f"Diagnosis: {diag}\nEvidence: {sent}\nMedication: {med}"


def _plot_agglom_recluster(
    phrase_embeddings: np.ndarray,
    agglom_ids: List[int],
    path_freeform: List[str],
    agglom_taxonomy: Dict[int, str],
    all_paths_flat: List[Tuple[int, int, Dict]],
    out_path: str,
) -> None:
    """Two-panel projection of LLM agglomerative re-cluster quality.

    Left:  points colored by agglom group; centroid text box = taxonomy label.
    Right: same coordinates colored by original HDBSCAN cluster ID.
    Comparing the two panels shows whether the LLM re-grouping crosses HDBSCAN
    cluster boundaries in a semantically meaningful way.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        print(f"  Agglom re-cluster visualization skipped (missing library): {exc}")
        return

    n = len(agglom_ids)
    if n < 2:
        print("  Agglom re-cluster visualization skipped (< 2 paths).")
        return

    # -- Dimensionality reduction --------------------------------------------
    if n <= 30:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(phrase_embeddings)
        var_label = f"PCA  ({100 * reducer.explained_variance_ratio_.sum():.1f}% var)"
    else:
        try:
            perplexity = min(18, max(4, n // 5), n - 1)
            coords = TSNE(
                n_components=2, perplexity=perplexity,
                random_state=42, init="pca", max_iter=1500,
            ).fit_transform(phrase_embeddings)
            var_label = f"t-SNE  (perplexity={perplexity})"
        except Exception:
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(phrase_embeddings)
            var_label = f"PCA  ({100 * reducer.explained_variance_ratio_.sum():.1f}% var)"

    # -- Color palettes ------------------------------------------------------
    n_agglom = max(agglom_ids) + 1
    agglom_cmap = plt.colormaps.get_cmap("tab20" if n_agglom > 10 else "tab10")
    agglom_palette = {gid: agglom_cmap(gid / max(n_agglom - 1, 1)) for gid in range(n_agglom)}
    hdbscan_cids = [cid for cid, _, _ in all_paths_flat]
    unique_hdbscan = sorted(set(hdbscan_cids))
    hdbscan_cmap = plt.colormaps.get_cmap("tab20b" if len(unique_hdbscan) > 10 else "Set1")
    hdbscan_palette = {
        cid: hdbscan_cmap(i / max(len(unique_hdbscan) - 1, 1))
        for i, cid in enumerate(unique_hdbscan)
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")
    ax_left, ax_right = axes

    # -- Left panel: agglom group coloring ----------------------------------
    for flat_idx, (aid, (_, _, _)) in enumerate(zip(agglom_ids, all_paths_flat)):
        ax_left.scatter(
            coords[flat_idx, 0], coords[flat_idx, 1],
            color=agglom_palette[aid], s=70, alpha=0.85,
            edgecolors="white", linewidths=0.6, zorder=3,
        )
    if n <= 20:
        for flat_idx, phrase in enumerate(path_freeform):
            ax_left.annotate(
                phrase[:30], xy=(coords[flat_idx, 0], coords[flat_idx, 1]),
                fontsize=5.5, ha="center", va="top",
                xytext=(0, -7), textcoords="offset points", color="#374151",
            )
    for gid in range(n_agglom):
        mask = [i for i, a in enumerate(agglom_ids) if a == gid]
        if not mask:
            continue
        cx = float(np.mean([coords[i, 0] for i in mask]))
        cy = float(np.mean([coords[i, 1] for i in mask]))
        tax = agglom_taxonomy.get(gid, "?").replace("_", " ")
        ax_left.text(
            cx, cy, f"G{gid}: {tax}",
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": agglom_palette[gid],
                  "alpha": 0.85, "edgecolor": "white"},
            zorder=5,
        )
    left_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=agglom_palette[gid], markersize=8,
               label=f"G{gid} \u00b7 {agglom_taxonomy.get(gid, '?').replace('_', ' ')}")
        for gid in range(n_agglom)
    ]
    ax_left.legend(handles=left_handles, fontsize=7, loc="upper left",
                   framealpha=0.9, facecolor="white", edgecolor="#d1d5db",
                   title="Agglom Group", title_fontsize=7)
    ax_left.set_title(
        f"LLM Agglomerative Re-Clusters\n{n} paths \u2192 {n_agglom} groups",
        fontsize=11, color="#111827", pad=8,
    )

    # -- Right panel: HDBSCAN cluster coloring ------------------------------
    for flat_idx, (hcid, _, _) in enumerate(all_paths_flat):
        ax_right.scatter(
            coords[flat_idx, 0], coords[flat_idx, 1],
            color=hdbscan_palette[hcid], s=70, alpha=0.85,
            edgecolors="white", linewidths=0.6, zorder=3,
        )
    for hcid in unique_hdbscan:
        mask = [i for i, (c, _, _) in enumerate(all_paths_flat) if c == hcid]
        cx = float(np.mean([coords[i, 0] for i in mask]))
        cy = float(np.mean([coords[i, 1] for i in mask]))
        ax_right.text(
            cx, cy, f"C{hcid}",
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": hdbscan_palette[hcid],
                  "alpha": 0.85, "edgecolor": "white"},
            zorder=5,
        )
    right_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=hdbscan_palette[cid], markersize=8,
               label=f"HDBSCAN C{cid}")
        for cid in unique_hdbscan
    ]
    ax_right.legend(handles=right_handles, fontsize=7, loc="upper left",
                    framealpha=0.9, facecolor="white", edgecolor="#d1d5db",
                    title="HDBSCAN Cluster", title_fontsize=7)
    ax_right.set_title(
        f"Original HDBSCAN Clusters\n{len(unique_hdbscan)} clusters",
        fontsize=11, color="#111827", pad=8,
    )

    for ax in axes:
        ax.set_xlabel("dim 1", color="#111827")
    ax_left.set_ylabel("dim 2", color="#111827")
    fig.suptitle(
        f"LOKI \u2014 LLM Agglomerative Re-Cluster Quality  \u00b7  {var_label}",
        fontsize=12, color="#111827", y=1.02,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved agglom re-cluster plot: {out_path}")


def _plot_llm_vs_hdbscan(
    all_paths: List[Dict],
    pair_final_labels: Dict[Tuple[int, int], str],
    path_hdbscan_cids: List[int],
    out_path: str,
) -> None:
    """Two-panel comparison: LLM direct-pair labels (left) vs original HDBSCAN clusters (right).

    Both panels share the same 2D embedding space (BGE-large -> PCA/t-SNE; TF-IDF fallback).
    Left:  each path coloured by its (diag, med) pair's LLM-predicted relation type.
    Right: same coordinates coloured by original HDBSCAN cluster membership.
    Comparing the two panels reveals whether the LLM produces semantically coherent groups
    that are independent of (or better than) the HDBSCAN structural assignment.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from sklearn.decomposition import PCA
    except ImportError as exc:
        print(f"  LLM vs HDBSCAN visualization skipped (missing library): {exc}")
        return
    n = len(all_paths)
    if n < 2:
        print("  LLM vs HDBSCAN visualization skipped (< 2 paths).")
        return

    # Build text representation for each path
    path_texts = []
    for path in all_paths:
        diag_text = _extract_row_field(path.get("diag_row_text", ""), "diagnosis") or ""
        med_text = _extract_row_field(path.get("med_row_text", ""), "drug") or ""
        sent_text = str(path.get("sent_text", "")).strip()
        path_texts.append(f"{diag_text} {sent_text} {med_text}".strip())

    # Embed paths - try BGE first, fall back to TF-IDF
    embeddings = None
    embed_method = "TF-IDF"
    try:
        if "bge" in _AGGLOM_ENCODER_CACHE:
            st_model = _AGGLOM_ENCODER_CACHE["bge"]
        else:
            _AGGLOM_ENCODER_CACHE["bge"] = _build_sentence_transformer(DEFAULT_BGE_ENCODER_NAME)[0]
            st_model = _AGGLOM_ENCODER_CACHE["bge"]
        embeddings = st_model.encode(
            path_texts, normalize_embeddings=True, show_progress_bar=False
        )
        embed_method = "BGE-large"
    except Exception:
        pass

    if embeddings is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(max_features=512, sublinear_tf=True)
            embeddings = tfidf.fit_transform(path_texts).toarray()
        except Exception as exc:
            print(f"  LLM vs HDBSCAN visualization skipped (embedding failed): {exc}")
            return

    # 2D reduction
    if n <= 30:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        var_label = f"PCA  ({embed_method})"
    else:
        try:
            from sklearn.manifold import TSNE
            perplexity = min(18, max(4, n // 5), n - 1)
            coords = TSNE(
                n_components=2, perplexity=perplexity,
                random_state=42, init="pca", max_iter=1500,
            ).fit_transform(embeddings)
            var_label = f"t-SNE  ({embed_method}, perplexity={perplexity})"
        except Exception:
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
            var_label = f"PCA  ({embed_method})"

    # Colour palettes
    REL_COLORS: Dict[str, str] = {
        "TREATS": "#22c55e",
        "ADVERSE_EFFECT": "#ef4444",
        "DISCONTINUED": "#f59e0b",
        "CONTRAINDICATED": "#8b5cf6",
    }
    DEFAULT_LLM_COLOR = "#6b7280"
    unique_hdbscan = sorted(set(path_hdbscan_cids))
    hdbscan_cmap = plt.colormaps.get_cmap("tab20b" if len(unique_hdbscan) > 10 else "Set1")
    hdbscan_palette: Dict[int, Any] = {
        cid: hdbscan_cmap(i / max(len(unique_hdbscan) - 1, 1))
        for i, cid in enumerate(unique_hdbscan)
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")
    ax_left, ax_right = axes

    # Left panel: LLM-predicted label per (diag, med) pair
    # Typed pairs (TREATS / ADVERSE_EFFECT / DISCONTINUED / CONTRAINDICATED) are drawn
    # opaque and on top; unlabeled / suppressed pairs are rendered as faint background
    # context so reviewers can see the typed structure clearly without losing the global
    # distribution. The legend explicitly calls out the "Other / Unlabeled" class.
    has_unlabeled = False
    for idx, path in enumerate(all_paths):
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        lbl = pair_final_labels.get(pair_key, "")
        if lbl in REL_COLORS:
            ax_left.scatter(
                coords[idx, 0], coords[idx, 1],
                color=REL_COLORS[lbl], s=80, alpha=0.92,
                edgecolors="white", linewidths=0.7, zorder=4,
            )
        else:
            has_unlabeled = True
            ax_left.scatter(
                coords[idx, 0], coords[idx, 1],
                color=DEFAULT_LLM_COLOR, s=24, alpha=0.18,
                edgecolors="none", zorder=2,
            )
    left_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=REL_COLORS[lbl], markersize=9, label=lbl)
        for lbl in _preferred_rel_type_order()
        if lbl in REL_COLORS
    ]
    if has_unlabeled:
        left_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=DEFAULT_LLM_COLOR, markeredgecolor="none",
                   alpha=0.35, markersize=8,
                   label="Other / Unlabeled"),
        )
    ax_left.legend(
        handles=left_handles, fontsize=8, loc="best",
        framealpha=0.9, facecolor="white", edgecolor="#d1d5db",
    )
    n_pairs = len(pair_final_labels)
    ax_left.set_title(
        f"LLM Direct Labels  \u2014  No HDBSCAN\n{n_pairs} (diag, med) pairs",
        fontsize=11, color="#111827", pad=8,
    )

    # Right panel: original HDBSCAN cluster colours
    for idx, cid in enumerate(path_hdbscan_cids):
        ax_right.scatter(
            coords[idx, 0], coords[idx, 1],
            color=hdbscan_palette[cid], s=70, alpha=0.85,
            edgecolors="white", linewidths=0.6, zorder=3,
        )
    for cid in unique_hdbscan:
        mask = [i for i, c in enumerate(path_hdbscan_cids) if c == cid]
        cx = float(np.mean([coords[i, 0] for i in mask]))
        cy = float(np.mean([coords[i, 1] for i in mask]))
        ax_right.text(
            cx, cy, f"C{cid}", ha="center", va="center",
            fontsize=8, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": hdbscan_palette[cid],
                  "alpha": 0.85, "edgecolor": "white"},
            zorder=5,
        )
    right_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=hdbscan_palette[cid], markersize=8,
               label=f"HDBSCAN C{cid}")
        for cid in unique_hdbscan
    ]
    ax_right.legend(
        handles=right_handles, fontsize=7, loc="upper left",
        framealpha=0.9, facecolor="white", edgecolor="#d1d5db",
        title="HDBSCAN Cluster", title_fontsize=7,
        ncol=2 if len(unique_hdbscan) > 6 else 1,
    )
    ax_right.set_title(
        f"Original HDBSCAN Clusters\n{len(unique_hdbscan)} clusters",
        fontsize=11, color="#111827", pad=8,
    )
    for ax in axes:
        ax.set_xlabel("dim 1", color="#111827")
    ax_left.set_ylabel("dim 2", color="#111827")
    fig.suptitle(
        f"LOKI \u2014 LLM Direct Groups vs HDBSCAN  \u00b7  {var_label}  \u00b7  {n} paths",
        fontsize=12, color="#111827", y=1.02,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved LLM vs HDBSCAN comparison: {out_path}")

    # -- VLDB companion plots ------------------------------------------------
    # All reuse the precomputed `embeddings`, `path_texts`, and the LLM/HDBSCAN
    # label assignments to render publication-quality views that complement the
    # main 2D scatter. Each view is independent and skipped silently on error so
    # one failure does not block the others.
    def _companion_path(suffix: str) -> str:
        return str(Path(out_path).with_name(
            Path(out_path).stem.replace("llm_vs_hdbscan", suffix) + ".png"
        ))

    # 1. 3D - typed clusters across PCA / t-SNE / UMAP
    try:
        _plot_llm_clusters_3d(
            all_paths=all_paths, pair_final_labels=pair_final_labels,
            embeddings=embeddings, rel_colors=REL_COLORS,
            embed_method=embed_method, out_path=_companion_path("llm_clusters_3d"),
        )
    except Exception as exc:
        print(f"  LLM 3D cluster visualization skipped: {exc}")

    # 2. Supervised LDA 2D + 1σ covariance ellipses (best class-separation view)
    try:
        _plot_llm_clusters_lda(
            all_paths=all_paths, pair_final_labels=pair_final_labels,
            embeddings=embeddings, rel_colors=REL_COLORS,
            embed_method=embed_method, out_path=_companion_path("llm_lda_ellipses"),
        )
    except Exception as exc:
        print(f"  LLM LDA-ellipses visualization skipped: {exc}")

    # 3. Sankey: HDBSCAN cluster -> LLM type (shows what the LLM step contributes)
    try:
        _plot_hdbscan_to_llm_sankey(
            all_paths=all_paths, pair_final_labels=pair_final_labels,
            path_hdbscan_cids=path_hdbscan_cids,
            rel_colors=REL_COLORS, default_color=DEFAULT_LLM_COLOR,
            hdbscan_palette=hdbscan_palette,
            out_path=_companion_path("hdbscan_to_llm_sankey"),
        )
    except Exception as exc:
        print(f"  HDBSCAN->LLM Sankey skipped: {exc}")

    # 4. HDBSCAN x LLM contingency heatmap (compact quantitative companion to Sankey)
    try:
        _plot_hdbscan_llm_heatmap(
            all_paths=all_paths, pair_final_labels=pair_final_labels,
            path_hdbscan_cids=path_hdbscan_cids,
            rel_colors=REL_COLORS,
            out_path=_companion_path("hdbscan_llm_heatmap"),
        )
    except Exception as exc:
        print(f"  HDBSCANxLLM heatmap skipped: {exc}")

    # 5. Small multiples (1x4 facets, one per type highlighted) on the same 2D coords
    try:
        _plot_llm_clusters_facets(
            all_paths=all_paths, pair_final_labels=pair_final_labels,
            coords=coords, rel_colors=REL_COLORS,
            var_label=var_label, out_path=_companion_path("llm_clusters_facets"),
        )
    except Exception as exc:
        print(f"  LLM small-multiples facets skipped: {exc}")

    # 6. Per-type top TF-IDF tokens (what the LLM is keying on)
    try:
        _plot_llm_type_top_tokens(
            path_texts=path_texts, all_paths=all_paths,
            pair_final_labels=pair_final_labels, rel_colors=REL_COLORS,
            out_path=_companion_path("llm_type_top_tokens"),
        )
    except Exception as exc:
        print(f"  LLM top-tokens chart skipped: {exc}")

    # 7. Score distributions by LLM type (LOKI path_score & CE ce_score)
    try:
        _plot_llm_type_score_distributions(
            all_paths=all_paths, pair_final_labels=pair_final_labels,
            rel_colors=REL_COLORS, out_path=_companion_path("llm_score_distributions"),
        )
    except Exception as exc:
        print(f"  LLM score-distribution chart skipped: {exc}")


def _plot_llm_clusters_3d(
    all_paths: List[Dict],
    pair_final_labels: Dict[Tuple[int, int], str],
    embeddings: np.ndarray,
    rel_colors: Dict[str, str],
    embed_method: str,
    out_path: str,
) -> None:
    """Three-panel 3D visualization of LLM-typed clusters using PCA, t-SNE, and UMAP.

    Reuses the path embeddings already computed by ``_plot_llm_vs_hdbscan`` and shows
    ONLY the typed pairs (TREATS / ADVERSE_EFFECT / DISCONTINUED / CONTRAINDICATED) so
    inter-class separation is easier to read. Designed for the VLDB visualization -
    moving from 2D to 3D typically separates DISCONTINUED from ADVERSE_EFFECT, which
    overlap heavily in 2D projections because both rely on negation / temporal cues.

    Falls back gracefully when UMAP isn't installed (uses ICA as third projection).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        print(f"  LLM 3D cluster visualization skipped (missing library): {exc}")
        return

    # Restrict to typed-labelled paths only
    typed_indices: List[int] = []
    typed_labels: List[str] = []
    for idx, path in enumerate(all_paths):
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        lbl = pair_final_labels.get(pair_key, "")
        if lbl in rel_colors:
            typed_indices.append(idx)
            typed_labels.append(lbl)

    if len(typed_indices) < 4:
        print(f"  LLM 3D cluster visualization skipped (< 4 typed paths: {len(typed_indices)}).")
        return

    sub_emb = np.asarray(embeddings)[typed_indices]
    n_typed = sub_emb.shape[0]

    # -- Projection 1: PCA --
    try:
        pca_coords = PCA(n_components=3, random_state=42).fit_transform(sub_emb)
    except Exception as exc:
        print(f"  3D PCA failed: {exc}")
        return

    # -- Projection 2: t-SNE --
    perplexity = min(18, max(4, n_typed // 4), max(n_typed - 1, 4))
    try:
        tsne_coords = TSNE(
            n_components=3, perplexity=perplexity,
            random_state=42, init="pca", max_iter=1500,
        ).fit_transform(sub_emb)
        tsne_title = f"t-SNE 3D  (perplexity={perplexity})"
    except Exception as exc:
        print(f"  3D t-SNE failed, reusing PCA: {exc}")
        tsne_coords = pca_coords
        tsne_title = "t-SNE 3D  (fallback: PCA)"

    # -- Projection 3: UMAP (preferred) or FastICA fallback --
    third_coords = None
    third_title = ""
    try:
        import umap  # type: ignore
        n_neighbors = min(15, max(2, n_typed - 1))
        third_coords = umap.UMAP(
            n_components=3, n_neighbors=n_neighbors,
            min_dist=0.1, random_state=42,
        ).fit_transform(sub_emb)
        third_title = f"UMAP 3D  (n_neighbors={n_neighbors})"
    except Exception:
        try:
            from sklearn.decomposition import FastICA
            third_coords = FastICA(
                n_components=3, random_state=42, max_iter=500,
            ).fit_transform(sub_emb)
            third_title = "FastICA 3D  (UMAP not installed)"
        except Exception as exc:
            print(f"  3D UMAP/ICA both failed, reusing PCA for third panel: {exc}")
            third_coords = pca_coords
            third_title = "PCA 3D  (fallback)"

    fig = plt.figure(figsize=(21, 7))
    fig.patch.set_facecolor("white")
    panels = [
        (pca_coords, f"PCA 3D  ({embed_method})"),
        (tsne_coords, f"{tsne_title}  ·  {embed_method}"),
        (third_coords, f"{third_title}  ·  {embed_method}"),
    ]

    point_colors = [rel_colors[lbl] for lbl in typed_labels]
    for panel_idx, (coords3d, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, panel_idx, projection="3d")
        ax.set_facecolor("white")
        ax.scatter(
            coords3d[:, 0], coords3d[:, 1], coords3d[:, 2],
            c=point_colors, s=55, alpha=0.92,
            edgecolors="white", linewidths=0.5, depthshade=True,
        )
        ax.set_title(title, fontsize=11, color="#111827", pad=8)
        ax.set_xlabel("dim 1", color="#111827", fontsize=9)
        ax.set_ylabel("dim 2", color="#111827", fontsize=9)
        ax.set_zlabel("dim 3", color="#111827", fontsize=9)
        ax.tick_params(labelsize=7, colors="#374151")
        # softer panes
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("#d1d5db")
            pane.set_alpha(0.05)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=rel_colors[lbl], markersize=10, label=lbl)
        for lbl in _preferred_rel_type_order()
        if lbl in rel_colors
    ]
    fig.legend(
        handles=legend_handles, fontsize=10,
        loc="lower center", ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.95, facecolor="white", edgecolor="#d1d5db",
    )
    fig.suptitle(
        f"LOKI - LLM Typed Cluster Geometry (3D)  ·  {n_typed} typed paths",
        fontsize=13, color="#111827", y=0.99,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved 3D LLM cluster visualization: {out_path}")


# =============================================================================
# VLDB-quality companion plots - supervised projection, contingency, distributions
# =============================================================================

def _typed_subset(
    all_paths: List[Dict],
    pair_final_labels: Dict[Tuple[int, int], str],
    rel_colors: Dict[str, str],
) -> Tuple[List[int], List[str]]:
    """Return indices and labels of paths whose pair received one of the four
    canonical LLM-typed relationship labels in ``rel_colors``."""
    typed_indices: List[int] = []
    typed_labels: List[str] = []
    for idx, path in enumerate(all_paths):
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        lbl = pair_final_labels.get(pair_key, "")
        if lbl in rel_colors:
            typed_indices.append(idx)
            typed_labels.append(lbl)
    return typed_indices, typed_labels


def _plot_llm_clusters_lda(
    all_paths: List[Dict],
    pair_final_labels: Dict[Tuple[int, int], str],
    embeddings: np.ndarray,
    rel_colors: Dict[str, str],
    embed_method: str,
    out_path: str,
) -> None:
    """Supervised 2D LDA projection with class centroids and 1σ covariance ellipses.

    Unlike PCA / t-SNE / UMAP, LDA explicitly finds axes that maximize the
    between-class to within-class scatter ratio for the LLM-predicted types,
    so this is the cleanest "do the embeddings actually encode the predicate?"
    view for VLDB readers.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Ellipse
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.decomposition import PCA
    except ImportError as exc:
        print(f"  LDA-ellipses visualization skipped (missing library): {exc}")
        return

    typed_indices, typed_labels = _typed_subset(all_paths, pair_final_labels, rel_colors)
    if len(typed_indices) < 6:
        print(f"  LDA-ellipses skipped (< 6 typed paths: {len(typed_indices)}).")
        return

    sub_emb = np.asarray(embeddings)[typed_indices]
    label_arr = np.asarray(typed_labels)
    unique_labels = sorted(set(typed_labels), key=lambda l: _preferred_rel_type_order().index(l)
                           if l in _preferred_rel_type_order() else 99)

    # LDA needs >= 2 classes with >= 2 samples each
    per_class_counts = {lbl: int((label_arr == lbl).sum()) for lbl in unique_labels}
    valid_classes = [lbl for lbl, c in per_class_counts.items() if c >= 2]
    if len(valid_classes) < 2:
        print(f"  LDA-ellipses skipped (need >=2 classes with >=2 samples; got {per_class_counts}).")
        return

    used_method = "LDA"
    try:
        n_comp = min(2, len(valid_classes) - 1)
        lda = LinearDiscriminantAnalysis(n_components=max(n_comp, 1))
        coords_lda = lda.fit_transform(sub_emb, label_arr)
        if coords_lda.shape[1] == 1:
            # 2-class case: LDA gives only 1 axis; pad with PCA orthogonal axis for readability
            pca_axis = PCA(n_components=2, random_state=42).fit_transform(sub_emb)[:, 1:2]
            coords2d = np.hstack([coords_lda, pca_axis])
            used_method = "LDA (1D) + PCA y-axis"
        else:
            coords2d = coords_lda[:, :2]
    except Exception as exc:
        print(f"  LDA failed, falling back to PCA: {exc}")
        coords2d = PCA(n_components=2, random_state=42).fit_transform(sub_emb)
        used_method = "PCA (LDA fallback)"

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#d1d5db")

    for lbl in unique_labels:
        mask = label_arr == lbl
        pts = coords2d[mask]
        color = rel_colors[lbl]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            color=color, s=70, alpha=0.85,
            edgecolors="white", linewidths=0.6, zorder=4,
            label=f"{lbl} (n={int(mask.sum())})",
        )
        if pts.shape[0] >= 2:
            centroid = pts.mean(axis=0)
            ax.scatter(
                centroid[0], centroid[1],
                marker="X", color=color, s=240,
                edgecolors="#111827", linewidths=1.2, zorder=6,
            )
            # 1-σ covariance ellipse (no scipy dependency)
            try:
                cov = np.cov(pts.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = eigvals.argsort()[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
                width, height = 2.0 * np.sqrt(np.maximum(eigvals, 1e-9))
                ellipse = Ellipse(
                    xy=(float(centroid[0]), float(centroid[1])),
                    width=float(width), height=float(height),
                    angle=angle, facecolor=color, edgecolor=color,
                    alpha=0.12, linewidth=1.6, zorder=3,
                )
                ax.add_patch(ellipse)
            except Exception:
                pass

    ax.legend(
        fontsize=9, loc="best", framealpha=0.95,
        facecolor="white", edgecolor="#d1d5db",
        title="LLM relation type", title_fontsize=9,
    )
    ax.set_xlabel("LDA axis 1", color="#111827")
    ax.set_ylabel("LDA axis 2", color="#111827")
    ax.set_title(
        f"LOKI - Supervised Class Geometry ({used_method})  ·  {embed_method}\n"
        f"Centroids (x) and 1σ covariance ellipses  ·  {len(typed_indices)} typed paths",
        fontsize=11.5, color="#111827", pad=8,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved LDA + ellipses visualization: {out_path}")


def _plot_hdbscan_to_llm_sankey(
    all_paths: List[Dict],
    pair_final_labels: Dict[Tuple[int, int], str],
    path_hdbscan_cids: List[int],
    rel_colors: Dict[str, str],
    default_color: str,
    hdbscan_palette: Dict[int, Any],
    out_path: str,
) -> None:
    """Sankey / alluvial: HDBSCAN cluster (left) -> LLM relation type (right).

    Bar heights are proportional to path counts. Ribbon thickness is proportional
    to the number of paths flowing from a given HDBSCAN cluster into a given LLM
    type. Tells the system-design story: "what does the LLM step contribute on
    top of pure structural clustering?"
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch, Rectangle
        from matplotlib.path import Path as MplPath
    except ImportError as exc:
        print(f"  Sankey visualization skipped (missing library): {exc}")
        return

    n = len(all_paths)
    if n < 2 or len(path_hdbscan_cids) != n:
        print("  Sankey skipped (insufficient paths / mismatched HDBSCAN labels).")
        return

    # Build contingency - paper-ready Sankey shows only typed LLM relations and
    # only the HDBSCAN clusters that actually contributed at least one typed
    # (non-"Other / Unlabeled") path. We deliberately drop the "Other" type
    # bar and any HDBSCAN cluster that routed only to "Other"; those add noise
    # without telling the system-design story.
    rel_order = [lbl for lbl in _preferred_rel_type_order() if lbl in rel_colors]
    flow: Dict[Tuple[int, str], int] = defaultdict(int)
    left_totals: Dict[int, int] = defaultdict(int)
    right_totals: Dict[str, int] = defaultdict(int)
    for idx, path in enumerate(all_paths):
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        lbl_raw = pair_final_labels.get(pair_key, "")
        if lbl_raw not in rel_colors:
            continue  # skip "Other / Unlabeled" (negative type)
        cid = int(path_hdbscan_cids[idx])
        if cid < 0:
            continue  # skip HDBSCAN noise (negative cluster)
        flow[(cid, lbl_raw)] += 1
        left_totals[cid] += 1
        right_totals[lbl_raw] += 1

    rel_order = [lbl for lbl in rel_order if right_totals.get(lbl, 0) > 0]
    if not rel_order or not left_totals:
        print("  Sankey skipped (no typed edges after filtering Other / Unlabeled).")
        return

    # Collapse small HDBSCAN clusters into two combined rows so the figure stays
    # compact for VLDB layout. n=1 -> "Singletons"; 2 <= n <= 3 -> "Small clusters
    # (n<=3)". Each bucket gets a sentinel id and a neutral gray bar plus an
    # explicit label so reviewers can see exactly what was aggregated.
    SINGLETON_CID = -2
    SMALL_CID = -3
    SMALL_MAX_N = 3  # collapse any cluster with path count <= this (and >= 2)
    singleton_cids = [cid for cid, tot in left_totals.items() if tot == 1]
    small_cids = [cid for cid, tot in left_totals.items() if 2 <= tot <= SMALL_MAX_N]
    n_singleton_clusters = len(singleton_cids)
    n_small_clusters = len(small_cids)
    n_singleton_paths = sum(left_totals[c] for c in singleton_cids)
    n_small_paths = sum(left_totals[c] for c in small_cids)
    if n_singleton_clusters >= 2:
        for scid in singleton_cids:
            for lbl in list(rel_order):
                cnt = flow.pop((scid, lbl), 0)
                if cnt > 0:
                    flow[(SINGLETON_CID, lbl)] += cnt
            del left_totals[scid]
        left_totals[SINGLETON_CID] = n_singleton_paths
    if n_small_clusters >= 2:
        for scid in small_cids:
            for lbl in list(rel_order):
                cnt = flow.pop((scid, lbl), 0)
                if cnt > 0:
                    flow[(SMALL_CID, lbl)] += cnt
            del left_totals[scid]
        left_totals[SMALL_CID] = n_small_paths

    # Sort: real clusters first by id, small + singleton buckets pinned to the bottom.
    sentinel_keys = {SINGLETON_CID, SMALL_CID}
    real_keys = sorted(cid for cid in left_totals if cid not in sentinel_keys)
    tail = []
    if SMALL_CID in left_totals:
        tail.append(SMALL_CID)
    if SINGLETON_CID in left_totals:
        tail.append(SINGLETON_CID)
    left_keys = real_keys + tail

    typed_total = sum(right_totals.values())

    # Layout
    fig, ax = plt.subplots(figsize=(12, max(5.5, 0.35 * max(len(left_keys), len(rel_order)) + 4)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axis_off()

    bar_x_left, bar_x_right = 0.05, 0.95
    bar_width = 0.04
    gap = 0.012  # vertical gap between stacked bars on a side

    total_paths = float(typed_total)
    left_height = 1.0 - gap * max(len(left_keys) - 1, 0)
    right_height = 1.0 - gap * max(len(rel_order) - 1, 0)

    left_positions: Dict[int, Tuple[float, float]] = {}
    cursor = 1.0
    for cid in left_keys:
        h = left_totals[cid] / total_paths * left_height
        top, bot = cursor, cursor - h
        left_positions[cid] = (bot, top)
        if cid == SINGLETON_CID:
            color = "#9ca3af"
            row_label = f"Singletons\n(k={n_singleton_clusters} clusters,\n n={left_totals[cid]} paths)"
        elif cid == SMALL_CID:
            color = "#b5bcc7"
            row_label = f"Small clusters (n<={SMALL_MAX_N})\n(k={n_small_clusters} clusters,\n n={left_totals[cid]} paths)"
        else:
            color = hdbscan_palette.get(cid, "#9ca3af")
            row_label = f"HDBSCAN C{cid}\n(n={left_totals[cid]})"
        ax.add_patch(Rectangle(
            (bar_x_left, bot), bar_width, h,
            facecolor=color, edgecolor="white", linewidth=0.8,
        ))
        ax.text(
            bar_x_left - 0.008, (top + bot) / 2.0,
            row_label,
            ha="right", va="center", fontsize=8.5, color="#111827",
            fontweight=("bold" if cid in sentinel_keys else "normal"),
        )
        cursor = bot - gap

    right_positions: Dict[str, Tuple[float, float]] = {}
    cursor = 1.0
    for lbl in rel_order:
        h = right_totals[lbl] / total_paths * right_height
        top, bot = cursor, cursor - h
        right_positions[lbl] = (bot, top)
        color = rel_colors.get(lbl, default_color)
        ax.add_patch(Rectangle(
            (bar_x_right - bar_width, bot), bar_width, h,
            facecolor=color, edgecolor="white", linewidth=0.8,
        ))
        ax.text(
            bar_x_right + 0.008, (top + bot) / 2.0,
            f"{lbl}\n(n={right_totals[lbl]})",
            ha="left", va="center", fontsize=9, color="#111827", fontweight="bold",
        )
        cursor = bot - gap

    # Draw ribbons (sorted by left top-down, then right top-down for readability)
    left_cursors = {cid: top for cid, (_, top) in left_positions.items()}
    right_cursors = {lbl: top for lbl, (_, top) in right_positions.items()}
    edges = [(cid, lbl, cnt) for (cid, lbl), cnt in flow.items() if cnt > 0]
    edges.sort(key=lambda e: (left_keys.index(e[0]), rel_order.index(e[1])))

    for cid, lbl, cnt in edges:
        h = cnt / total_paths * left_height  # left and right scaling identical (both use total_paths)
        l_top = left_cursors[cid]
        l_bot = l_top - h
        left_cursors[cid] = l_bot
        r_top = right_cursors[lbl]
        r_bot = r_top - (cnt / total_paths * right_height)
        right_cursors[lbl] = r_bot

        x0, x1 = bar_x_left + bar_width, bar_x_right - bar_width
        ctrl = (x0 + x1) / 2.0
        verts = [
            (x0, l_top),
            (ctrl, l_top), (ctrl, r_top), (x1, r_top),
            (x1, r_bot),
            (ctrl, r_bot), (ctrl, l_bot), (x0, l_bot),
            (x0, l_top),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ribbon_color = rel_colors.get(lbl, default_color)
        ax.add_patch(PathPatch(
            MplPath(verts, codes),
            facecolor=ribbon_color, edgecolor="none", alpha=0.35,
        ))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    n_real_clusters = len(real_keys)
    collapsed_parts = []
    if n_small_clusters >= 2:
        collapsed_parts.append(f"{n_small_clusters} small (n<={SMALL_MAX_N}) collapsed")
    if n_singleton_clusters >= 2:
        collapsed_parts.append(f"{n_singleton_clusters} singletons (n=1) collapsed")
    cluster_summary = (
        f"{n_real_clusters} large HDBSCAN clusters (n>{SMALL_MAX_N})"
        + (f" + {' + '.join(collapsed_parts)}" if collapsed_parts else "")
    )
    ax.set_title(
        f"LOKI - HDBSCAN Clusters -> LLM Relation Types  ·  {typed_total} typed paths\n"
        f"Ribbon thickness ∝ path count  ·  {cluster_summary} -> {len(rel_order)} LLM types"
        + ("\n(Negative cluster id=−1 and \"Other / Unlabeled\" type excluded)"),
        fontsize=12, color="#111827", pad=10,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved HDBSCAN->LLM Sankey: {out_path}")


def _plot_hdbscan_llm_heatmap(
    all_paths: List[Dict],
    pair_final_labels: Dict[Tuple[int, int], str],
    path_hdbscan_cids: List[int],
    rel_colors: Dict[str, str],
    out_path: str,
) -> None:
    """Compact row-normalized contingency heatmap of HDBSCAN cluster x LLM type.

    Each row sums to 1.0 (fraction of that HDBSCAN cluster routed to each LLM
    type). Cell annotations show raw counts in parentheses.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  HDBSCANxLLM heatmap skipped (missing library): {exc}")
        return

    n = len(all_paths)
    if n < 2 or len(path_hdbscan_cids) != n:
        print("  Heatmap skipped (insufficient paths).")
        return

    rel_order = [lbl for lbl in _preferred_rel_type_order() if lbl in rel_colors]
    rel_order.append("Other")
    left_keys = sorted(set(path_hdbscan_cids))
    counts = np.zeros((len(left_keys), len(rel_order)), dtype=np.int64)
    for idx, path in enumerate(all_paths):
        cid = int(path_hdbscan_cids[idx])
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        lbl_raw = pair_final_labels.get(pair_key, "")
        lbl = lbl_raw if lbl_raw in rel_colors else "Other"
        i_row = left_keys.index(cid)
        i_col = rel_order.index(lbl)
        counts[i_row, i_col] += 1

    row_sums = counts.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0.0] = 1.0
    normed = counts / row_sums

    fig, ax = plt.subplots(figsize=(max(6.5, 1.0 * len(rel_order) + 3), max(3.5, 0.45 * len(left_keys) + 1.5)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(normed, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(rel_order)))
    ax.set_xticklabels(rel_order, rotation=20, ha="right", fontsize=9, color="#111827")
    ax.set_yticks(range(len(left_keys)))
    ax.set_yticklabels([f"C{cid}  (n={int(counts[i].sum())})" for i, cid in enumerate(left_keys)],
                       fontsize=9, color="#111827")
    ax.set_xlabel("LLM relation type", color="#111827")
    ax.set_ylabel("HDBSCAN cluster", color="#111827")

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            cnt = int(counts[i, j])
            if cnt == 0:
                continue
            frac = normed[i, j]
            txt_color = "white" if frac > 0.55 else "#111827"
            ax.text(
                j, i, f"{frac:.2f}\n({cnt})",
                ha="center", va="center", fontsize=8.5, color=txt_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Row-normalized share", color="#111827")
    ax.set_title(
        f"LOKI - HDBSCAN x LLM Contingency  ·  {n} paths\n"
        "Rows sum to 1.0  ·  raw count in parentheses",
        fontsize=11.5, color="#111827", pad=8,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved HDBSCANxLLM heatmap: {out_path}")


def _plot_llm_clusters_facets(
    all_paths: List[Dict],
    pair_final_labels: Dict[Tuple[int, int], str],
    coords: np.ndarray,
    rel_colors: Dict[str, str],
    var_label: str,
    out_path: str,
) -> None:
    """Small-multiples view: one panel per LLM type, highlighted against a faded backdrop.

    Same 2D coordinates as the main scatter; reviewers can scan all four classes
    in a single glance without colour-clash overload.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Small-multiples facets skipped (missing library): {exc}")
        return

    rel_order = [lbl for lbl in _preferred_rel_type_order() if lbl in rel_colors]
    if not rel_order:
        return

    # Map each path to its label (or "Other")
    path_labels: List[str] = []
    for path in all_paths:
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        lbl_raw = pair_final_labels.get(pair_key, "")
        path_labels.append(lbl_raw if lbl_raw in rel_colors else "Other")

    n_panels = len(rel_order)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 5.0), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, lbl in zip(axes, rel_order):
        ax.set_facecolor("white")
        ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")

        # backdrop
        mask_other = np.array([pl != lbl for pl in path_labels])
        ax.scatter(
            coords[mask_other, 0], coords[mask_other, 1],
            color="#9ca3af", s=18, alpha=0.18,
            edgecolors="none", zorder=2,
        )
        # highlighted class
        mask_class = ~mask_other
        n_class = int(mask_class.sum())
        ax.scatter(
            coords[mask_class, 0], coords[mask_class, 1],
            color=rel_colors[lbl], s=80, alpha=0.95,
            edgecolors="white", linewidths=0.7, zorder=4,
        )
        ax.set_title(f"{lbl}  (n={n_class})", fontsize=11, color="#111827", pad=6)
        ax.set_xlabel("dim 1", color="#111827", fontsize=9)
    axes[0].set_ylabel("dim 2", color="#111827", fontsize=9)
    fig.suptitle(
        f"LOKI - LLM Relation Types (small multiples)  ·  {var_label}  ·  {len(all_paths)} paths",
        fontsize=12.5, color="#111827", y=1.02,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved LLM small-multiples facets: {out_path}")


def _plot_llm_type_top_tokens(
    path_texts: List[str],
    all_paths: List[Dict],
    pair_final_labels: Dict[Tuple[int, int], str],
    rel_colors: Dict[str, str],
    out_path: str,
    top_k_tokens: int = 10,
) -> None:
    """Per-type top TF-IDF tokens as a 1xK horizontal bar chart.

    Shows qualitative lexical evidence of what each LLM-typed cluster is keying
    on - useful for the paper to back up the supervised LDA separation with
    interpretable terms.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        print(f"  Top-tokens chart skipped (missing library): {exc}")
        return

    rel_order = [lbl for lbl in _preferred_rel_type_order() if lbl in rel_colors]
    if not rel_order or not path_texts:
        return

    # Bucket texts by LLM type
    buckets: Dict[str, List[str]] = {lbl: [] for lbl in rel_order}
    for idx, path in enumerate(all_paths):
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        lbl = pair_final_labels.get(pair_key, "")
        if lbl in buckets and idx < len(path_texts):
            buckets[lbl].append(path_texts[idx])

    rel_present = [lbl for lbl in rel_order if buckets[lbl]]
    if not rel_present:
        print("  Top-tokens chart skipped (no typed paths).")
        return

    # Fit TF-IDF over all path texts; compute per-class mean tf-idf vector
    try:
        vec = TfidfVectorizer(
            max_features=2000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True, min_df=1,
        )
        X = vec.fit_transform(path_texts)
        vocab = np.array(vec.get_feature_names_out())
    except Exception as exc:
        print(f"  Top-tokens TF-IDF failed: {exc}")
        return

    # Per-class mean (TF-IDF) minus background mean -> discriminative tokens
    bg_mean = np.asarray(X.mean(axis=0)).ravel()
    fig, axes = plt.subplots(1, len(rel_present), figsize=(4.6 * len(rel_present), 5.0))
    if len(rel_present) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, lbl in zip(axes, rel_present):
        ax.set_facecolor("white")
        ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.7, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")

        idxs_for_class = [i for i, p in enumerate(all_paths)
                          if pair_final_labels.get((int(p["diag_row_idx"]), int(p["med_row_idx"])), "") == lbl
                          and i < X.shape[0]]
        if not idxs_for_class:
            ax.text(0.5, 0.5, "(no paths)", ha="center", va="center",
                    transform=ax.transAxes, color="#6b7280")
            ax.set_title(f"{lbl}  (n=0)", fontsize=11, color="#111827")
            continue

        cls_mean = np.asarray(X[idxs_for_class].mean(axis=0)).ravel()
        discriminative = cls_mean - bg_mean
        top_idx = np.argsort(-discriminative)[:top_k_tokens]
        top_idx = [i for i in top_idx if discriminative[i] > 0][:top_k_tokens]
        if not top_idx:
            top_idx = list(np.argsort(-cls_mean)[:top_k_tokens])

        tokens = vocab[top_idx][::-1]
        weights = cls_mean[top_idx][::-1]
        ax.barh(
            range(len(tokens)), weights,
            color=rel_colors[lbl], alpha=0.85, edgecolor="white", linewidth=0.6,
        )
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=9, color="#111827")
        ax.set_xlabel("mean TF-IDF in class", fontsize=9, color="#111827")
        ax.set_title(f"{lbl}  (n={len(idxs_for_class)})",
                     fontsize=11, color="#111827", pad=6)

    fig.suptitle(
        "LOKI - Discriminative tokens per LLM relation type  ·  TF-IDF (1-2 grams)",
        fontsize=12.5, color="#111827", y=1.02,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved per-type top-tokens chart: {out_path}")


def _plot_llm_type_score_distributions(
    all_paths: List[Dict],
    pair_final_labels: Dict[Tuple[int, int], str],
    rel_colors: Dict[str, str],
    out_path: str,
) -> None:
    """Two-panel violin/strip plot of LOKI ``path_score`` and CE ``ce_score`` per LLM type.

    Visualises whether the underlying join confidences (LOKI) and the
    cross-encoder reranker scores stratify by LLM-inferred relationship type -
    e.g. TREATS paths typically score higher than ADVERSE_EFFECT paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Score-distribution chart skipped (missing library): {exc}")
        return

    rel_order = [lbl for lbl in _preferred_rel_type_order() if lbl in rel_colors]
    rel_order.append("Other")
    palette = {**rel_colors, "Other": "#9ca3af"}

    loki_buckets: Dict[str, List[float]] = {lbl: [] for lbl in rel_order}
    ce_buckets: Dict[str, List[float]] = {lbl: [] for lbl in rel_order}
    for path in all_paths:
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        lbl_raw = pair_final_labels.get(pair_key, "")
        lbl = lbl_raw if lbl_raw in rel_colors else "Other"
        try:
            loki_buckets[lbl].append(float(path.get("path_score", 0.0)))
        except (TypeError, ValueError):
            pass
        ce_val = path.get("ce_score")
        if ce_val is not None:
            try:
                ce_buckets[lbl].append(float(ce_val))
            except (TypeError, ValueError):
                pass

    present = [lbl for lbl in rel_order if loki_buckets[lbl]]
    if not present:
        print("  Score-distribution chart skipped (no scored paths).")
        return

    ce_has_data = any(ce_buckets[lbl] for lbl in present)
    n_panels = 2 if ce_has_data else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 5.0))
    if n_panels == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    def _draw_violin(ax, buckets: Dict[str, List[float]], title: str, ylabel: str):
        ax.set_facecolor("white")
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.7, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")

        labels_drawn = [lbl for lbl in present if buckets[lbl]]
        data = [buckets[lbl] for lbl in labels_drawn]
        positions = list(range(1, len(labels_drawn) + 1))
        if not data:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="#6b7280")
            return

        # Only types with >=2 samples get a violin; singletons get a marker only
        violin_positions = [p for p, d in zip(positions, data) if len(d) >= 2]
        violin_data = [d for d in data if len(d) >= 2]
        if violin_data:
            parts = ax.violinplot(
                violin_data, positions=violin_positions,
                widths=0.7, showmeans=False, showmedians=True, showextrema=False,
            )
            for body, lbl in zip(parts["bodies"], [l for l, d in zip(labels_drawn, data) if len(d) >= 2]):
                body.set_facecolor(palette[lbl])
                body.set_edgecolor(palette[lbl])
                body.set_alpha(0.45)
            if "cmedians" in parts:
                parts["cmedians"].set_color("#111827")
                parts["cmedians"].set_linewidth(1.4)

        # Strip / jitter overlay
        rng = np.random.default_rng(42)
        for pos, lbl, d in zip(positions, labels_drawn, data):
            if not d:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(d))
            ax.scatter(
                np.full(len(d), pos) + jitter, d,
                color=palette[lbl], s=24, alpha=0.75,
                edgecolors="white", linewidths=0.4, zorder=4,
            )
            ax.scatter(
                [pos], [float(np.mean(d))],
                marker="D", color=palette[lbl], s=70,
                edgecolors="#111827", linewidths=1.0, zorder=6,
            )

        ax.set_xticks(positions)
        # Rotate slightly + right-align so long type names (e.g. "CONTRAINDICATED",
        # "DISCONTINUED") never collide with adjacent labels on narrow axes.
        ax.set_xticklabels(
            [f"{lbl}\n(n={len(d)})" for lbl, d in zip(labels_drawn, data)],
            fontsize=8.5, color="#111827", rotation=18, ha="right",
            rotation_mode="anchor",
        )
        ax.set_ylabel(ylabel, color="#111827")
        ax.set_title(title, fontsize=11.5, color="#111827", pad=6)

    _draw_violin(axes[0], loki_buckets,
                 "LOKI path_score by LLM type", "path_score (avg of diag+med side)")
    if ce_has_data:
        _draw_violin(axes[1], ce_buckets,
                     "Cross-encoder ce_score by LLM type", "ce_score ∈ [0, 1]")

    fig.suptitle(
        "LOKI - Confidence stratification by LLM relation type",
        fontsize=12.5, color="#111827", y=1.02,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved per-type score distribution: {out_path}")


def _label_clusters_agglomerative(
    clusters: Dict[int, List[Dict]],
    base_url: str,
    model: str,
    candidate_labels: List[str],
    temperature: float,
    timeout_secs: int,
    agglom_distance_threshold: float,
    agglom_encoder: str = "medembed",
    encoder_model: Optional[Any] = None,
    vis_out_path: Optional[str] = None,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    """Agglomerative re-labeling pipeline.

    Phase 1 - Per path, ask the LLM for a free-form relationship description
               (3-8 words; unconstrained).
    Phase 2 - Embed all descriptions with the MedEmbed encoder (falls back to
               TF-IDF when no encoder is available) and run AgglomerativeClustering
               so semantically similar paths are grouped regardless of which HDBSCAN
               cluster they were originally assigned to.
    Phase 3 - One closed-label LLM call per agglomerative group.
    Phase 4 - Map HDBSCAN clusters back via path_score-weighted majority vote.
    """
    from collections import defaultdict as _dd
    from sklearn.cluster import AgglomerativeClustering as _AggClust  # type: ignore

    freeform_sys = _build_lmstudio_freeform_system_prompt(candidate_labels)
    taxonomy_sys = _build_lmstudio_system_prompt(candidate_labels)
    tie_break_order = list(candidate_labels)

    # -- Phase 1: free-form label per path ----------------------------------
    all_paths_flat: List[Tuple[int, int, Dict]] = []  # (cid, pidx_in_cluster, path)
    for cid, cpaths in sorted(clusters.items()):
        for pidx, path in enumerate(cpaths):
            all_paths_flat.append((cid, pidx, path))

    total = len(all_paths_flat)
    print(f"  [agglom] Phase 1: generating free-form labels for {total} paths ...")
    path_freeform: List[str] = []
    for _cid, _pidx, path in all_paths_flat:
        user_msg = _build_lmstudio_freeform_user_message(path)
        cache_key = _lmstudio_cache_key(freeform_sys, user_msg)
        if cache_key in _LMSTUDIO_LABEL_CACHE:
            phrase = _LMSTUDIO_LABEL_CACHE[cache_key]
        else:
            try:
                raw = _call_lmstudio(base_url, model, freeform_sys, user_msg, temperature, timeout_secs)
                phrase = raw.strip().strip('"\' ') or "clinical relationship"
            except Exception as exc:
                if _should_abort_lmstudio_fallback(exc):
                    raise
                phrase = "clinical relationship"
            _LMSTUDIO_LABEL_CACHE[cache_key] = phrase
        path_freeform.append(phrase)
    print(f"  [agglom] Phase 1 done. Sample: {path_freeform[:4]}")

    # -- Phase 2: Embed descriptions + AgglomerativeClustering --------------
    n = len(path_freeform)
    if n == 0:
        return {}, {}
    phrase_embeddings: Optional[np.ndarray] = None
    if n == 1:
        agglom_ids: List[int] = [0]
    else:
        try:
            if agglom_encoder == "medembed" and encoder_model is not None:
                embed_mode = "MedEmbed encoder"
                with torch.no_grad():
                    emb = encoder_model.encode_sentences(
                        path_freeform, batch_size=32, normalize=True
                    )
                phrase_embeddings = emb.cpu().numpy()  # [n, d]
            elif agglom_encoder in ("bge", "minilm"):
                _MODEL_NAME_MAP = {
                    "bge": DEFAULT_BGE_ENCODER_NAME,
                    "minilm": DEFAULT_MINILM_ENCODER_NAME,
                }
                if agglom_encoder not in _AGGLOM_ENCODER_CACHE:
                    _AGGLOM_ENCODER_CACHE[agglom_encoder] = _build_sentence_transformer(
                        _MODEL_NAME_MAP[agglom_encoder]
                    )[0]
                st_model = _AGGLOM_ENCODER_CACHE[agglom_encoder]
                phrase_embeddings = st_model.encode(path_freeform, normalize_embeddings=True)
                embed_mode = f"{agglom_encoder} ({_MODEL_NAME_MAP[agglom_encoder]})"
            else:
                embed_mode = "TF-IDF (no encoder available)"
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                from sklearn.preprocessing import normalize as _normalize  # type: ignore
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
                phrase_embeddings = _normalize(
                    vectorizer.fit_transform(path_freeform), norm="l2"
                ).toarray()
            print(f"  [agglom] Phase 2 embedding: {embed_mode}")
            agg = _AggClust(
                n_clusters=None,
                distance_threshold=agglom_distance_threshold,
                metric="cosine",
                linkage="average",
            )
            agglom_ids = agg.fit_predict(phrase_embeddings).tolist()
        except Exception as exc:
            print(f"  [agglom] Clustering failed ({exc}); treating all paths as one group.")
            agglom_ids = [0] * n

    n_agglom = max(agglom_ids) + 1
    print(f"  [agglom] Phase 2: {n} paths -> {n_agglom} agglomerative groups")

    # -- Phase 3: taxonomy label per agglom group ---------------------------
    agglom_phrases: Dict[int, List[str]] = _dd(list)
    for flat_idx, aid in enumerate(agglom_ids):
        agglom_phrases[int(aid)].append(path_freeform[flat_idx])

    agglom_taxonomy: Dict[int, str] = {}
    for agid, phrases in sorted(agglom_phrases.items()):
        combined = "\n".join(f"- {p}" for p in phrases[:12])
        user_msg = combined
        cache_key = _lmstudio_cache_key(taxonomy_sys, user_msg)
        if cache_key in _LMSTUDIO_LABEL_CACHE:
            tax_label: Optional[str] = _LMSTUDIO_LABEL_CACHE[cache_key]
        else:
            try:
                raw = _call_lmstudio(base_url, model, taxonomy_sys, user_msg, temperature, timeout_secs)
                tax_label = _parse_lmstudio_label(raw, candidate_labels)
            except Exception as exc:
                if _should_abort_lmstudio_fallback(exc):
                    raise
                tax_label = None
            if tax_label is None:
                tax_label = _keyword_classify(
                    [{"sent_text": " ".join(phrases), "sent_idx": 0}],
                    candidate_labels=candidate_labels,
                )
            _LMSTUDIO_LABEL_CACHE[cache_key] = tax_label  # type: ignore[assignment]
        agglom_taxonomy[agid] = tax_label or tie_break_order[0]  # type: ignore[assignment]
    print(f"  [agglom] Phase 3 taxonomy: {dict(sorted(agglom_taxonomy.items()))}")

    # -- Phase 4: map HDBSCAN clusters via path_score-weighted vote ---------
    hdbscan_votes: Dict[int, Dict[str, float]] = _dd(lambda: {lbl: 0.0 for lbl in candidate_labels})
    hdbscan_freeform: Dict[int, List[str]] = _dd(list)
    for flat_idx, (cid, _pidx, path) in enumerate(all_paths_flat):
        aid = agglom_ids[flat_idx]
        tax_label_f = agglom_taxonomy.get(aid, tie_break_order[0])
        score = max(float(path.get("path_score", 0.0)), 0.0)
        hdbscan_votes[cid][tax_label_f] += score if score > 0 else 1.0
        hdbscan_freeform[cid].append(path_freeform[flat_idx])

    cluster_labels: Dict[int, str] = {}
    cluster_details: Dict[int, Dict[str, object]] = {}
    for cid, vote_map in sorted(hdbscan_votes.items()):
        best_label = min(
            tie_break_order,
            key=lambda lbl: (-vote_map.get(lbl, 0.0), tie_break_order.index(lbl)),
        )
        cpaths = clusters[cid]
        unique_sent_count = len({int(p["sent_idx"]) for p in cpaths})
        keyword_scores_map = {lbl: float(s) for lbl, s in _keyword_scores(cpaths).items()}
        cluster_labels[cid] = best_label
        cluster_details[cid] = {
            "backend": "llm_agglomerative",
            "label_source": "lmstudio_agglom_vote",
            "score_type": "path_score_weighted_vote",
            "label_input_mode": "freeform_agglomerative",
            "label_scores": {lbl: round(v, 4) for lbl, v in vote_map.items()},
            "label_counts": {},
            "fallback_reason": None,
            "n_occurrences": len(cpaths),
            "n_unique_sentences": unique_sent_count,
            "n_scored_occurrences": len(cpaths),
            "supporting_evidence": _build_supporting_evidence(cpaths, evidence_records=[]),
            "keyword_scores": keyword_scores_map,
            "agglom_freeform_sample": hdbscan_freeform[cid][:5],
        }
        print(
            f"  Cluster {cid:3d}  ({unique_sent_count:2d} sents / {len(cpaths):3d} paths)  "
            f"-> {best_label} [agglom_vote]"
        )

    # -- Visualization: agglom re-cluster projection --------------------------
    if vis_out_path is not None and phrase_embeddings is not None:
        _plot_agglom_recluster(
            phrase_embeddings, agglom_ids, path_freeform,
            agglom_taxonomy, all_paths_flat, vis_out_path,
        )

    return cluster_labels, cluster_details


def _build_lmstudio_cluster_user_message(
    cluster_paths: List[Dict],
    max_sentences: int = LMSTUDIO_DEFAULT_MAX_EVIDENCE_SENTS,
) -> str:
    best_path = max(cluster_paths, key=lambda p: float(p.get("path_score", 0.0)))
    medication = _extract_row_field(best_path.get("med_row_text", ""), "drug")
    diagnosis = _extract_row_field(best_path.get("diag_row_text", ""), "diagnosis")
    if not medication:
        medication = " ".join(str(best_path.get("med_row_text", "")).split())
    if not diagnosis:
        diagnosis = " ".join(str(best_path.get("diag_row_text", "")).split())
    # Prefer ce_score (Option C, per-pair sentence rerank) when present so the
    # LLM sees CE-preferred evidence first and within the top-N window. Fall
    # back to LOKI path_score otherwise. We track both best_ce and best_path
    # per sentence so unscored sentences (ce_score is None) sort last but the
    # rest of the LOKI ranking is preserved.
    sent_stats: Dict[int, Dict] = {}
    for path in cluster_paths:
        idx = int(path["sent_idx"])
        stats = sent_stats.setdefault(
            idx,
            {"text": str(path.get("sent_text", "")).strip(), "best_score": 0.0, "best_ce": float("-inf")},
        )
        stats["best_score"] = max(stats["best_score"], float(path.get("path_score", 0.0)))
        ce_val = path.get("ce_score")
        if ce_val is not None:
            stats["best_ce"] = max(stats["best_ce"], float(ce_val))
    ranked_sents = sorted(
        sent_stats.values(),
        key=lambda s: (s["best_ce"], s["best_score"]),
        reverse=True,
    )
    resolved_max_sentences = len(ranked_sents) if int(max_sentences) <= 0 else int(max_sentences)
    evidence = " ".join(s["text"][:200] for s in ranked_sents[:resolved_max_sentences])
    return (
        f"Diagnosis: {diagnosis}\n"
        f"Evidence: {evidence}\n"
        f"Medication: {medication}"
    )


def _lmstudio_cache_key(system_prompt: str, user_message: str) -> str:
    content = f"{system_prompt}\n\n{user_message}"
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _parse_lmstudio_label(response_text: str, candidate_labels: List[str]) -> Optional[str]:
    """Extract <LABEL>...</LABEL> from LLM response and validate against candidate_labels."""
    match = re.search(r"<LABEL>\s*(.*?)\s*</LABEL>", response_text, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip().upper().replace(" ", "_")
    if raw in candidate_labels:
        return raw
    # Partial prefix match (e.g. "TREAT" -> "TREATS", "ADVERSE" -> "ADVERSE_EFFECT")
    for label in candidate_labels:
        if len(raw) >= 4 and (raw in label or label.startswith(raw[:4])):
            return label
    return None


class LMStudioUnavailableError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, last_error: Optional[BaseException] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


def _configure_lmstudio_runtime(*, fail_closed: bool, retry_attempts: int = LMSTUDIO_DEFAULT_RETRY_ATTEMPTS) -> None:
    global _LMSTUDIO_FAIL_CLOSED, _LMSTUDIO_RETRY_ATTEMPTS
    _LMSTUDIO_FAIL_CLOSED = bool(fail_closed)
    _LMSTUDIO_RETRY_ATTEMPTS = max(int(retry_attempts), 1)


def _current_lmstudio_retry_attempts() -> int:
    return max(int(_LMSTUDIO_RETRY_ATTEMPTS), 1)


def _should_abort_lmstudio_fallback(exc: BaseException) -> bool:
    return _LMSTUDIO_FAIL_CLOSED and isinstance(exc, LMStudioUnavailableError)


def _probe_lmstudio_ready(base_url: str, timeout_secs: int = 5) -> bool:
    try:
        import requests as _requests  # type: ignore
    except ImportError as exc:
        if _LMSTUDIO_FAIL_CLOSED:
            raise LMStudioUnavailableError(
                "requests package not installed. Run: pip install requests",
                attempts=0,
                last_error=exc,
            ) from exc
        return False

    url = f"{base_url.rstrip('/')}/api/v1/models"
    attempts = _current_lmstudio_retry_attempts()
    last_error: Optional[BaseException] = None
    for _attempt in range(1, attempts + 1):
        try:
            probe = _requests.get(url, timeout=timeout_secs)
            probe.raise_for_status()
            return True
        except Exception as exc:
            last_error = exc

    if _LMSTUDIO_FAIL_CLOSED:
        raise LMStudioUnavailableError(
            f"Cannot reach LMStudio after {attempts} attempts ({url}): {last_error}",
            attempts=attempts,
            last_error=last_error,
        ) from last_error
    return False


def _call_lmstudio(
    base_url: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    timeout_secs: int,
) -> str:
    """Call LM Studio's native /api/v1/chat endpoint.

    Combines system_prompt and user_message into the `input` field.
    Returns the `output` string from the response, or falls back to
    OpenAI-style choices if the server is running in compatibility mode.
    Set ANTHROPIC_AUTH_TOKEN in the environment if the server requires auth.
    """
    try:
        import requests as _requests  # type: ignore
    except ImportError as exc:
        raise LMStudioUnavailableError(
            "requests package not installed. Run: pip install requests",
            attempts=0,
            last_error=exc,
        ) from exc

    full_input = (
        f"{system_prompt}\n\n---\n\n{user_message}" if system_prompt else user_message
    )
    api_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    url = f"{base_url.rstrip('/')}/api/v1/chat"
    payload = {
        "model": model,
        "input": full_input,
        "temperature": temperature,
        "store": False
        }
        
    attempts = _current_lmstudio_retry_attempts()
    last_error: Optional[BaseException] = None
    data: Dict[str, Any]
    for attempt in range(1, attempts + 1):
        try:
            resp = _requests.post(url, headers=headers, json=payload, timeout=timeout_secs)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                raise LMStudioUnavailableError(
                    f"LMStudio request failed after {attempts} attempts ({url}): {exc}",
                    attempts=attempts,
                    last_error=exc,
                ) from exc
    else:
        raise LMStudioUnavailableError(
            f"LMStudio request failed after {attempts} attempts ({url}): {last_error}",
            attempts=attempts,
            last_error=last_error,
        ) from last_error

    if "output" in data:
        out = data["output"]
        if isinstance(out, list):
            # /api/v1/chat sometimes returns a list of message dicts or strings
            parts: List[str] = []
            for item in out:
                if isinstance(item, dict):
                    parts.append(str(item.get("content") or item.get("text") or ""))
                else:
                    parts.append(str(item))
            return " ".join(p for p in parts if p)
        return str(out)
    if "choices" in data and data["choices"]:
        return data["choices"][0].get("message", {}).get("content", "")
    raise ValueError(f"Unexpected LM Studio response structure: {list(data.keys())}")


def label_clusters_with_llm_path_vote(
    clusters: Dict[int, List[Dict]],
    base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
    model: str = LMSTUDIO_DEFAULT_MODEL,
    temperature: float = LMSTUDIO_DEFAULT_TEMPERATURE,
    timeout_secs: int = LMSTUDIO_DEFAULT_TIMEOUT_SECS,
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    """Label HDBSCAN clusters via flat per-path LLM calls, bypassing the 4-phase
    agglomerative pipeline entirely.

    Each path gets its own closed-label LMStudio call. Results are aggregated
    back to HDBSCAN clusters via path_score-weighted majority vote.
    HDBSCAN structural grouping is preserved; only the relationship type label
    is decided per-path rather than per agglom-group.
    """
    if not clusters:
        return {}, {}

    print(f"  Loading LMStudio labeler (path-vote mode): {base_url}  model={model}")
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    tie_break_order = list(resolved_candidate_labels)

    try:
        if not _probe_lmstudio_ready(base_url, timeout_secs=5):
            print("  Warning: LMStudio is unavailable. Falling back to keyword classifier.")
            return label_clusters_with_keyword(clusters, candidate_labels=resolved_candidate_labels)
        print("  LMStudio ready")
    except Exception as exc:
        if _should_abort_lmstudio_fallback(exc):
            raise
        print(f"  Warning: Cannot reach LMStudio ({exc}). Falling back to keyword classifier.")
        return label_clusters_with_keyword(clusters, candidate_labels=resolved_candidate_labels)

    system_prompt = _build_lmstudio_system_prompt(resolved_candidate_labels)

    # Flatten all paths across all clusters
    all_paths_flat: List[Tuple[int, Dict]] = []
    for cid, cpaths in sorted(clusters.items()):
        for path in cpaths:
            all_paths_flat.append((cid, path))

    total = len(all_paths_flat)
    print(f"  [llm_path_vote] Labeling {total} paths across {len(clusters)} clusters ...")

    # Per-path closed-label call
    path_labels: List[Tuple[int, str, float]] = []  # (cid, label, path_score)
    for cid, path in all_paths_flat:
        diag_text = (
            _extract_row_field(path.get("diag_row_text", ""), "diagnosis")
            or " ".join(str(path.get("diag_row_text", "")).split())
        )
        med_text = (
            _extract_row_field(path.get("med_row_text", ""), "drug")
            or " ".join(str(path.get("med_row_text", "")).split())
        )
        user_msg = (
            f"Diagnosis: {diag_text}\n"
            f"Evidence: {' '.join(str(path.get('sent_text', '')).split())}\n"
            f"Medication: {med_text}\n"
            "Output:"
        )
        cache_key = _lmstudio_cache_key(system_prompt, user_msg)
        if cache_key in _LMSTUDIO_LABEL_CACHE:
            parsed = _LMSTUDIO_LABEL_CACHE[cache_key]
        else:
            try:
                raw = _call_lmstudio(base_url, model, system_prompt, user_msg, temperature, timeout_secs)
                parsed = _parse_lmstudio_label(raw, resolved_candidate_labels) or ""
            except Exception as exc:
                if _should_abort_lmstudio_fallback(exc):
                    raise
                parsed = ""
            _LMSTUDIO_LABEL_CACHE[cache_key] = parsed
        if not parsed:
            parsed = _keyword_classify([path], candidate_labels=resolved_candidate_labels)
        path_score = max(float(path.get("path_score", 0.0)), 0.0)
        path_labels.append((cid, parsed, path_score))

    # Aggregate per HDBSCAN cluster via path_score-weighted vote
    cluster_labels: Dict[int, str] = {}
    cluster_details: Dict[int, Dict[str, object]] = {}
    for cid, cpaths in sorted(clusters.items()):
        vote_scores: Dict[str, float] = {lbl: 0.0 for lbl in resolved_candidate_labels}
        vote_counts: Dict[str, float] = {lbl: 0.0 for lbl in resolved_candidate_labels}
        n_scored = 0
        for pcid, plabel, pscore in path_labels:
            if pcid == cid and plabel in vote_scores:
                vote_scores[plabel] += pscore
                vote_counts[plabel] += 1.0
                n_scored += 1
        unique_sent_count = len({int(p["sent_idx"]) for p in cpaths})
        if max(vote_scores.values(), default=0.0) > 0.0:
            label = min(
                tie_break_order,
                key=lambda lbl: (-vote_scores.get(lbl, 0.0), tie_break_order.index(lbl)),
            )
            label_source = "lmstudio_path_vote"
        elif max(vote_counts.values(), default=0.0) > 0.0:
            label = min(
                tie_break_order,
                key=lambda lbl: (-vote_counts.get(lbl, 0.0), tie_break_order.index(lbl)),
            )
            label_source = "lmstudio_path_vote_count"
        else:
            label = _keyword_classify(cpaths, candidate_labels=resolved_candidate_labels)
            label_source = "keyword_fallback"
        cluster_labels[cid] = label
        cluster_details[cid] = {
            "backend": "llm_path_vote",
            "label_source": label_source,
            "score_type": "lmstudio_path_vote_weight",
            "label_input_mode": "per_path_direct",
            "label_scores": vote_scores,
            "label_counts": vote_counts,
            "fallback_reason": None,
            "n_occurrences": len(cpaths),
            "n_unique_sentences": unique_sent_count,
            "n_scored_occurrences": n_scored,
            "supporting_evidence": _build_supporting_evidence(cpaths, evidence_records=[]),
        }
        print(
            f"  Cluster {cid:3d}  ({unique_sent_count:2d} sents / {len(cpaths):3d} paths)  "
            f"-> {label} [{label_source}]"
        )

    return cluster_labels, cluster_details


def label_pairs_with_llm_no_hdbscan(
    clusters: Dict[int, List[Dict]],
    base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
    model: str = LMSTUDIO_DEFAULT_MODEL,
    temperature: float = LMSTUDIO_DEFAULT_TEMPERATURE,
    timeout_secs: int = LMSTUDIO_DEFAULT_TIMEOUT_SECS,
    vis_out_path: Optional[str] = None,
    max_evidence_sents: int = LMSTUDIO_DEFAULT_MAX_EVIDENCE_SENTS,
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    """Label (diag, med) pairs with one LLM call per pair, bypassing HDBSCAN grouping.

    All paths belonging to the same (diag_row_idx, med_row_idx) pair are grouped first;
    then a single closed-label LMStudio call is made per pair, presenting the top-scored
    evidence sentences together via _build_lmstudio_cluster_user_message(). This gives the
    LLM the full multi-sentence context for each pair rather than one sentence at a time.

    Synthetic cluster IDs are assigned to each unique pair in sorted order, and
    path["raw_cluster_id"] is mutated in-place so the downstream pipeline can rebuild
    clusters/labels consistently (see run_materialization_pipeline's post-label_clusters block).

    Generates a two-panel comparison visualisation (LLM pair groups vs HDBSCAN clusters)
    using BGE-large embeddings (falls back to TF-IDF when sentence_transformers unavailable).
    """
    if not clusters:
        return {}, {}

    print(f"  Loading LMStudio labeler (no-HDBSCAN pair mode): {base_url}  model={model}")
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    tie_break_order = list(resolved_candidate_labels)

    try:
        if not _probe_lmstudio_ready(base_url, timeout_secs=5):
            print("  Warning: LMStudio is unavailable. Falling back to keyword classifier.")
            return label_clusters_with_keyword(clusters, candidate_labels=resolved_candidate_labels)
        print("  LMStudio ready")
    except Exception as exc:
        if _should_abort_lmstudio_fallback(exc):
            raise
        print(f"  Warning: Cannot reach LMStudio ({exc}). Falling back to keyword classifier.")
        return label_clusters_with_keyword(clusters, candidate_labels=resolved_candidate_labels)

    system_prompt = _build_lmstudio_system_prompt(resolved_candidate_labels)

    # Flatten all paths, preserving original HDBSCAN cluster IDs for visualisation
    all_paths_flat: List[Tuple[int, Dict]] = []  # (original_hdbscan_cid, path)
    for cid, cpaths in sorted(clusters.items()):
        for path in cpaths:
            all_paths_flat.append((cid, path))

    # Preserve the structural HDBSCAN assignment before this mode rewrites
    # raw_cluster_id into one synthetic cluster per (diag, med) pair.
    for cid, path in all_paths_flat:
        path["hdbscan_cluster_id"] = int(path.get("raw_cluster_id", cid))

    total = len(all_paths_flat)
    print(f"  [llm_no_hdbscan] Labeling {total} paths across {len(clusters)} HDBSCAN clusters ...")

    # -- One LLM call per (diag, med) pair - all evidence sentences together ----------
    # Group all paths by pair so the LLM receives the full multi-sentence context per pair.
    pair_paths_map: Dict[Tuple[int, int], List[Dict]] = {}
    for _, path in all_paths_flat:
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        pair_paths_map.setdefault(pair_key, []).append(path)

    # Collect: (diag_row_idx, med_row_idx, label, aggregate_path_score)
    path_llm_labels: List[Tuple[int, int, str, float]] = []
    for (d_idx, m_idx), cpaths in sorted(pair_paths_map.items()):
        user_msg = _build_lmstudio_cluster_user_message(cpaths, max_sentences=max_evidence_sents)
        cache_key = _lmstudio_cache_key(system_prompt, user_msg)
        if cache_key in _LMSTUDIO_LABEL_CACHE:
            parsed = _LMSTUDIO_LABEL_CACHE[cache_key]
        else:
            try:
                raw = _call_lmstudio(base_url, model, system_prompt, user_msg, temperature, timeout_secs)
                parsed = _parse_lmstudio_label(raw, resolved_candidate_labels) or ""
            except Exception as exc:
                if _should_abort_lmstudio_fallback(exc):
                    raise
                parsed = ""
            _LMSTUDIO_LABEL_CACHE[cache_key] = parsed
        if not parsed:
            parsed = _keyword_classify(cpaths, candidate_labels=resolved_candidate_labels)
        agg_score = sum(max(float(p.get("path_score", 0.0)), 0.0) for p in cpaths)
        path_llm_labels.append((d_idx, m_idx, parsed, agg_score))

    # Aggregate per (diag_row_idx, med_row_idx) pair
    from collections import defaultdict as _dd
    pair_vote_scores: Dict[Tuple[int, int], Dict[str, float]] = _dd(
        lambda: {lbl: 0.0 for lbl in resolved_candidate_labels}
    )
    pair_vote_counts: Dict[Tuple[int, int], Dict[str, float]] = _dd(
        lambda: {lbl: 0.0 for lbl in resolved_candidate_labels}
    )
    for d_idx, m_idx, lbl, pscore in path_llm_labels:
        pair_key = (d_idx, m_idx)
        if lbl in pair_vote_scores[pair_key]:
            pair_vote_scores[pair_key][lbl] += pscore
            pair_vote_counts[pair_key][lbl] += 1.0

    # Assign synthetic cluster IDs from sorted unique (diag, med) pairs
    unique_pairs = sorted(pair_vote_scores.keys())
    pair_to_cid: Dict[Tuple[int, int], int] = {p: i for i, p in enumerate(unique_pairs)}

    # Determine final label per pair
    pair_final_labels: Dict[Tuple[int, int], str] = {}
    cluster_labels: Dict[int, str] = {}
    cluster_details: Dict[int, Dict[str, object]] = {}
    for pair_key in unique_pairs:
        syn_cid = pair_to_cid[pair_key]
        vote_scores = dict(pair_vote_scores[pair_key])
        vote_counts = dict(pair_vote_counts[pair_key])
        cpaths = [
            path for _, path in all_paths_flat
            if int(path["diag_row_idx"]) == pair_key[0]
            and int(path["med_row_idx"]) == pair_key[1]
        ]
        if max(vote_scores.values(), default=0.0) > 0.0:
            label = min(
                tie_break_order,
                key=lambda lbl: (-vote_scores.get(lbl, 0.0), tie_break_order.index(lbl)),
            )
            label_source = "lmstudio_no_hdbscan_vote"
        elif max(vote_counts.values(), default=0.0) > 0.0:
            label = min(
                tie_break_order,
                key=lambda lbl: (-vote_counts.get(lbl, 0.0), tie_break_order.index(lbl)),
            )
            label_source = "lmstudio_no_hdbscan_vote_count"
        else:
            label = _keyword_classify(cpaths, candidate_labels=resolved_candidate_labels)
            label_source = "keyword_fallback"
        pair_final_labels[pair_key] = label
        cluster_labels[syn_cid] = label
        unique_sent_count = len({int(p["sent_idx"]) for p in cpaths})
        cluster_details[syn_cid] = {
            "backend": "llm_no_hdbscan",
            "label_source": label_source,
            "score_type": "lmstudio_pair_vote_weight",
            "label_input_mode": "per_path_direct",
            "label_scores": vote_scores,
            "label_counts": vote_counts,
            "fallback_reason": None,
            "n_occurrences": len(cpaths),
            "n_unique_sentences": unique_sent_count,
            "n_scored_occurrences": len(cpaths),
            "supporting_evidence": _build_supporting_evidence(cpaths, evidence_records=[]),
        }
        print(
            f"  Pair ({pair_key[0]:2d},{pair_key[1]:2d})  syn_cid={syn_cid:3d}  "
            f"({unique_sent_count:2d} sents / {len(cpaths):3d} paths)  -> {label} [{label_source}]"
        )

    # Mutate raw_cluster_id on every path to the synthetic pair cluster ID.
    # Pairs labeled NEGATIVE get raw_cluster_id = -1 and are suppressed downstream.
    _negative_pairs = {pk for pk, lbl in pair_final_labels.items() if lbl == "NEGATIVE"}
    for _, path in all_paths_flat:
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        path["raw_cluster_id"] = -1 if pair_key in _negative_pairs else pair_to_cid[pair_key]

    n_suppressed = len(_negative_pairs)
    n_kept = len(cluster_labels) - n_suppressed
    if n_suppressed:
        print(f"  [llm_no_hdbscan] Suppressed {n_suppressed} NEGATIVE-labeled pairs (no relation)")
    print(
        f"  [llm_no_hdbscan] {n_kept} active pair-clusters from "
        f"{len(clusters)} HDBSCAN clusters  ({total} paths)"
    )

    # Comparison visualisation (LLM groups vs HDBSCAN)
    if vis_out_path is not None:
        _plot_llm_vs_hdbscan(
            [path for _, path in all_paths_flat],
            pair_final_labels,
            [hcid for hcid, _ in all_paths_flat],
            vis_out_path,
        )

    return cluster_labels, cluster_details


def label_clusters_with_lmstudio(
    clusters: Dict[int, List[Dict]],
    base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
    model: str = LMSTUDIO_DEFAULT_MODEL,
    temperature: float = LMSTUDIO_DEFAULT_TEMPERATURE,
    timeout_secs: int = LMSTUDIO_DEFAULT_TIMEOUT_SECS,
    max_evidence_sents: int = LMSTUDIO_DEFAULT_MAX_EVIDENCE_SENTS,
    per_path_vote: bool = False,
    all_paths: Optional[List[Dict]] = None,
    hub_fanout_threshold: float = 0.3,
    max_pool_sentences: int = 12,
    use_agglomerative: bool = True,
    agglom_distance_threshold: float = LMSTUDIO_DEFAULT_AGGLOM_DISTANCE,
    encoder_model: Optional[Any] = None,
    vis_out_path: Optional[str] = None,
    llm_path_vote: bool = False,
    agglom_encoder: str = "medembed",
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    """Label clusters using a local LLM via LMStudio's OpenAI-compatible API.

    Agglomerative mode (default): per-path free-form LLM description ->
    MedEmbed encoder + AgglomerativeClustering -> one closed-label call per
    agglom group -> path_score-weighted vote maps back to HDBSCAN clusters.
    Handles noisy HDBSCAN groupings by re-clustering on semantic similarity.

    Per-cluster mode (--llm_no_agglomerative): one closed-label LLM call
    per HDBSCAN cluster presenting the top-scored evidence sentences.

    Per-path vote mode (--llm_per_path_vote): one closed-label call per
    path, aggregated by path_score-weighted vote.

    Falls back to the keyword classifier on parse failures and on connectivity
    failures when strict batch abort mode is disabled. In batch resume mode,
    LMStudio transport failures stop the batch so the unfinished admission can
    be retried cleanly with --resume.
    """
    if not clusters:
        return {}, {}

    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    if llm_path_vote:
        return label_clusters_with_llm_path_vote(
            clusters,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout_secs=timeout_secs,
            candidate_labels=resolved_candidate_labels,
        )

    print(f"  Loading LMStudio labeler: {base_url}  model={model}")
    tie_break_order = list(resolved_candidate_labels)

    try:
        if not _probe_lmstudio_ready(base_url, timeout_secs=5):
            print("  Warning: LMStudio is unavailable. Falling back to keyword classifier.")
            return label_clusters_with_keyword(clusters, candidate_labels=resolved_candidate_labels)
        print("  LMStudio ready")
    except Exception as exc:
        if _should_abort_lmstudio_fallback(exc):
            raise
        print(f"  Warning: Cannot reach LMStudio ({exc}). Falling back to keyword classifier.")
        return label_clusters_with_keyword(clusters, candidate_labels=resolved_candidate_labels)

    # -- Agglomerative mode (default) --------------------------------------
    if use_agglomerative:
        return _label_clusters_agglomerative(
            clusters,
            base_url=base_url,
            model=model,
            candidate_labels=resolved_candidate_labels,
            temperature=temperature,
            timeout_secs=timeout_secs,
            agglom_distance_threshold=agglom_distance_threshold,
            encoder_model=encoder_model,
            vis_out_path=vis_out_path,
            agglom_encoder=agglom_encoder,
        )

    system_prompt = _build_lmstudio_system_prompt(resolved_candidate_labels)
    cluster_labels: Dict[int, str] = {}
    cluster_details: Dict[int, Dict[str, object]] = {}

    for cid, cpaths in sorted(clusters.items()):
        unique_sent_count = len({int(path["sent_idx"]) for path in cpaths})
        keyword_scores_map = {
            lbl: float(score) for lbl, score in _keyword_scores(cpaths, candidate_labels=resolved_candidate_labels).items()
        }
        label = _keyword_classify(cpaths, candidate_labels=resolved_candidate_labels)
        label_source = "keyword_fallback"
        fallback_reason: Optional[str] = None
        vote_scores: Dict[str, float] = {lbl: 0.0 for lbl in resolved_candidate_labels}
        vote_counts: Dict[str, float] = {lbl: 0.0 for lbl in resolved_candidate_labels}
        n_scored = 0

        try:
            if per_path_vote:
                # One LLM call per path; aggregate by path_score-weighted vote
                scoring_paths = (
                    _build_cluster_evidence_pool(
                        cpaths, all_paths or list(cpaths),
                        hub_fanout_threshold=hub_fanout_threshold,
                        max_pool_size=max_pool_sentences,
                    )
                    if all_paths is not None
                    else cpaths
                )
                for path in scoring_paths:
                    diag_text = (
                        _extract_row_field(path.get("diag_row_text", ""), "diagnosis")
                        or " ".join(str(path.get("diag_row_text", "")).split())
                    )
                    med_text = (
                        _extract_row_field(path.get("med_row_text", ""), "drug")
                        or " ".join(str(path.get("med_row_text", "")).split())
                    )
                    user_msg = (
                        f"Diagnosis: {diag_text}\n"
                        f"Evidence: {' '.join(str(path.get('sent_text', '')).split())}\n"
                        f"Medication: {med_text}\n"
                        "Output:"
                    )
                    cache_key = _lmstudio_cache_key(system_prompt, user_msg)
                    if cache_key in _LMSTUDIO_LABEL_CACHE:
                        parsed = _LMSTUDIO_LABEL_CACHE[cache_key]
                    else:
                        raw_response = _call_lmstudio(
                            base_url, model, system_prompt, user_msg, temperature, timeout_secs
                        )
                        parsed = _parse_lmstudio_label(raw_response, resolved_candidate_labels) or ""
                        _LMSTUDIO_LABEL_CACHE[cache_key] = parsed
                    if parsed:
                        path_score = max(float(path.get("path_score", 0.0)), 0.0)
                        vote_scores[parsed] += path_score
                        vote_counts[parsed] += 1.0
                        n_scored += 1

                if max(vote_scores.values(), default=0.0) > 0.0:
                    label = min(
                        tie_break_order,
                        key=lambda lbl: (-vote_scores.get(lbl, 0.0), tie_break_order.index(lbl)),
                    )
                    label_source = "llm_per_path_vote"
                elif max(vote_counts.values(), default=0.0) > 0.0:
                    label = min(
                        tie_break_order,
                        key=lambda lbl: (-vote_counts.get(lbl, 0.0), tie_break_order.index(lbl)),
                    )
                    label_source = "llm_per_path_vote_count"
                else:
                    fallback_reason = "no_lmstudio_votes"
            else:
                # Per-cluster mode: one LLM call presenting top evidence sentences
                user_msg = _build_lmstudio_cluster_user_message(
                    cpaths, max_sentences=max_evidence_sents
                )
                cache_key = _lmstudio_cache_key(system_prompt, user_msg)
                if cache_key in _LMSTUDIO_LABEL_CACHE:
                    parsed_label = _LMSTUDIO_LABEL_CACHE[cache_key]
                else:
                    raw_response = _call_lmstudio(
                        base_url, model, system_prompt, user_msg, temperature, timeout_secs
                    )
                    parsed_label = _parse_lmstudio_label(raw_response, resolved_candidate_labels) or ""
                    _LMSTUDIO_LABEL_CACHE[cache_key] = parsed_label

                if parsed_label:
                    label = parsed_label
                    vote_scores[parsed_label] = 1.0
                    label_source = "lmstudio_cluster"
                    n_scored = 1
                else:
                    fallback_reason = "no_label_parsed"

        except Exception as exc:
            if _should_abort_lmstudio_fallback(exc):
                raise
            fallback_reason = str(exc)
            print(f"  LMStudio labeling failed for cluster {cid}: {exc}")

        cluster_labels[cid] = label
        cluster_details[cid] = {
            "backend": "lmstudio" if label_source.startswith("lmstudio") else "keyword",
            "label_source": label_source,
            "score_type": "lmstudio_vote_weight" if label_source.startswith("lmstudio") else "keyword_counts",
            "label_input_mode": "per_path_vote" if per_path_vote else "cluster_evidence",
            "label_scores": vote_scores if label_source.startswith("lmstudio") else {
                lbl: float(keyword_scores_map.get(lbl, 0.0)) for lbl in resolved_candidate_labels
            },
            "label_counts": vote_counts if per_path_vote else None,
            "fallback_reason": fallback_reason,
            "n_occurrences": len(cpaths),
            "n_unique_sentences": unique_sent_count,
            "n_scored_occurrences": n_scored,
            "supporting_evidence": _build_supporting_evidence(cpaths, evidence_records=[]),
        }
        print(
            f"  Cluster {cid:3d}  ({unique_sent_count:2d} sents / {len(cpaths):3d} paths)  "
            f"-> {label} [{label_source}]"
        )

    return cluster_labels, cluster_details


def label_clusters_with_gliner2(
    clusters: Dict[int, List[Dict]],
    gliner2_model: str = GLINER2_MODEL,
    batch_size: int = 8,
    threshold: float = 0.5,
    max_len: Optional[int] = 384,
    anchor_normalization_mode: str = "legacy",
    label_input_mode: str = DEFAULT_GLINER2_LABEL_INPUT_MODE,
    per_sentence_vote: bool = False,
    all_paths: Optional[List[Dict]] = None,
    hub_fanout_threshold: float = 0.3,
    max_pool_sentences: int = 12,
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    if not clusters:
        return {}, {}

    print(f"  Loading GLiNER2 labeler: {gliner2_model}")
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    gliner = None
    try:
        gliner = _load_gliner2_model(gliner2_model)
        print("  GLiNER2 ready")
    except Exception as exc:
        print(f"  Warning: Could not load GLiNER2 labeler ({exc}). Falling back to keyword classifier.")

    if gliner is None:
        return label_clusters_with_keyword(clusters, candidate_labels=resolved_candidate_labels)

    cluster_labels: Dict[int, str] = {}
    cluster_details: Dict[int, Dict[str, object]] = {}
    tie_break_order = list(resolved_candidate_labels)
    entity_schema = {
        "medication": "Drug or medication name",
        "diagnosis": "Diagnosis, symptom, side effect, contraindication, or clinical problem",
        "action": "Treatment action such as started, continued, stopped, discontinued, held, avoided, or could not tolerate",
    }
    relation_types = [_normalize_rel_type(label).lower() for label in resolved_candidate_labels]
    resolved_label_input_mode = (label_input_mode or DEFAULT_GLINER2_LABEL_INPUT_MODE).strip().lower()
    if resolved_label_input_mode not in {"sentence_evidence", "semantic_signature"}:
        raise ValueError(f"Unsupported GLiNER2 label input mode: {label_input_mode}")

    for cid, cpaths in sorted(clusters.items()):
        unique_sent_count = len({int(path["sent_idx"]) for path in cpaths})
        keyword_scores = {
            label: float(score)
            for label, score in _keyword_scores(cpaths, candidate_labels=resolved_candidate_labels).items()
        }
        label = _keyword_classify(cpaths, candidate_labels=resolved_candidate_labels)
        label_source = "keyword_fallback"
        fallback_reason: Optional[str] = None
        vote_scores = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
        vote_counts = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
        evidence_records: List[Dict[str, object]] = []

        try:
            # Phase 3+4: use hub-filtered evidence pool when per_sentence_vote is enabled
            scoring_paths = (
                _build_cluster_evidence_pool(
                    cpaths, all_paths or list(cpaths),
                    hub_fanout_threshold=hub_fanout_threshold,
                    max_pool_size=max_pool_sentences,
                )
                if (per_sentence_vote and all_paths is not None)
                else cpaths
            )
            pair_buckets = _bucket_paths_by_pair(scoring_paths)
            occurrence_texts = [
                _build_gliner2_occurrence_text_by_mode(
                    path,
                    pair_buckets.get((int(path["diag_row_idx"]), int(path["med_row_idx"])), [path]),
                    resolved_label_input_mode,
                )
                for path in scoring_paths
            ]
            batch_size_resolved = max(1, min(batch_size, len(occurrence_texts)))
            entity_results = gliner.batch_extract_entities(
                occurrence_texts,
                entity_schema,
                batch_size=batch_size_resolved,
                threshold=threshold,
                include_confidence=True,
                max_len=max_len,
            )
            relation_results = gliner.batch_extract_relations(
                occurrence_texts,
                relation_types,
                batch_size=batch_size_resolved,
                threshold=threshold,
                include_confidence=True,
                max_len=max_len,
            )

            for path, entity_result, relation_result in zip(scoring_paths, entity_results, relation_results):
                predicted_label, occurrence_scores, occurrence_meta = _score_gliner2_hybrid_occurrence(
                    path,
                    entity_result,
                    relation_result,
                    candidate_labels=candidate_labels,
                    anchor_normalization_mode=anchor_normalization_mode,
                )
                if predicted_label is None:
                    continue

                path_score = max(float(path.get("path_score", 0.0)), 0.0)
                for rel_type, raw_score in occurrence_scores.items():
                    vote_scores[rel_type] += path_score * max(float(raw_score), 0.0)

                predicted_score = max(float(occurrence_scores.get(predicted_label, 0.0)), 0.0)
                total_score = sum(max(float(score), 0.0) for score in occurrence_scores.values())
                normalized_support = predicted_score / total_score if total_score > 0.0 else 0.0
                vote_weight = path_score * predicted_score
                vote_counts[predicted_label] += 1.0
                evidence_records.append({
                    "sent_idx": int(path.get("sent_idx", -1)),
                    "section_name": str(path.get("section_name", "")),
                    "sentence": " ".join(str(path.get("sent_text", "")).split()),
                    "label": predicted_label,
                    "confidence": normalized_support,
                    "vote_weight": vote_weight,
                    "path_score": path_score,
                    "hybrid_scores": occurrence_scores,
                    "relation_hits": occurrence_meta.get("relation_hits", []),
                    "action_signals": occurrence_meta.get("action_signals", {}),
                    "lexical_signals": occurrence_meta.get("lexical_signals", {}),
                    "label_input_mode": resolved_label_input_mode,
                })

            if max(vote_scores.values(), default=0.0) > 0.0:
                label = min(
                    tie_break_order,
                    key=lambda rel_type: (-vote_scores.get(rel_type, 0.0), tie_break_order.index(rel_type)),
                )
                label_source = "gliner2_per_sentence_vote" if per_sentence_vote else "gliner2_hybrid"
            elif max(vote_counts.values(), default=0.0) > 0.0:
                label = min(
                    tie_break_order,
                    key=lambda rel_type: (-vote_counts.get(rel_type, 0.0), tie_break_order.index(rel_type)),
                )
                label_source = "gliner2_per_sentence_vote_count" if per_sentence_vote else "gliner2_hybrid_count_fallback"
            else:
                fallback_reason = "no_gliner2_votes"
        except Exception as exc:
            fallback_reason = str(exc)
            print(f"  GLiNER2 labeling failed for cluster {cid}: {exc}")

        cluster_labels[cid] = label
        cluster_details[cid] = {
            "backend": "gliner2" if label_source.startswith("gliner2") else "keyword",
            "label_source": label_source,
            "score_type": "hybrid_weighted_votes" if label_source.startswith("gliner2") else "keyword_counts",
            "label_input_mode": resolved_label_input_mode,
            "label_scores": vote_scores if label_source.startswith("gliner2") else {
                rel_type: float(keyword_scores.get(rel_type, 0.0))
                for rel_type in resolved_candidate_labels
            },
            "label_counts": vote_counts if label_source.startswith("gliner2") else None,
            "fallback_reason": fallback_reason,
            "n_occurrences": len(cpaths),
            "n_unique_sentences": unique_sent_count,
            "n_scored_occurrences": len(evidence_records),
            "supporting_evidence": _build_supporting_evidence(cpaths, evidence_records=evidence_records),
        }
        print(
            f"  Cluster {cid:3d}  ({unique_sent_count:2d} sents / {len(cpaths):3d} paths)  "
            f"-> {label} [{label_source}]"
        )

    return cluster_labels, cluster_details


def label_clusters(
    clusters: Dict[int, List[Dict]],
    backend: str = DEFAULT_CLUSTER_LABEL_BACKEND,
    gliner2_model: str = GLINER2_MODEL,
    gliner2_batch_size: int = 8,
    gliner2_threshold: float = 0.5,
    gliner2_max_len: Optional[int] = 384,
    anchor_normalization_mode: str = "legacy",
    gliner2_label_input_mode: str = DEFAULT_GLINER2_LABEL_INPUT_MODE,
    per_sentence_vote: bool = False,
    all_paths: Optional[List[Dict]] = None,
    hub_fanout_threshold: float = 0.3,
    max_pool_sentences: int = 12,
    llm_base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
    llm_model: str = LMSTUDIO_DEFAULT_MODEL,
    llm_temperature: float = LMSTUDIO_DEFAULT_TEMPERATURE,
    llm_timeout_secs: int = LMSTUDIO_DEFAULT_TIMEOUT_SECS,
    llm_max_evidence_sents: int = LMSTUDIO_DEFAULT_MAX_EVIDENCE_SENTS,
    llm_per_path_vote: bool = False,
    llm_agglomerative: bool = True,
    llm_agglom_distance: float = LMSTUDIO_DEFAULT_AGGLOM_DISTANCE,
    gt_relationships: Optional[List[Dict]] = None,
    encoder_model: Optional[Any] = None,
    llm_agglom_vis_path: Optional[str] = None,
    llm_path_vote: bool = False,
    llm_no_hdbscan: bool = False,
    llm_no_hdbscan_vis_path: Optional[str] = None,
    llm_agglom_encoder: str = "medembed",
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    resolved_backend = (backend or DEFAULT_CLUSTER_LABEL_BACKEND).strip().lower()
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    if resolved_backend == "keyword":
        return label_clusters_with_keyword(clusters, candidate_labels=resolved_candidate_labels)
    if resolved_backend == "oracle":
        if not gt_relationships:
            print("  Warning: oracle backend requested but no gt_relationships provided; falling back to keyword.")
            return label_clusters_with_keyword(clusters, candidate_labels=resolved_candidate_labels)
        return label_clusters_with_oracle(clusters, gt_relationships, candidate_labels=resolved_candidate_labels)
    if resolved_backend == "gliner2":
        return label_clusters_with_gliner2(
            clusters,
            gliner2_model=gliner2_model,
            batch_size=gliner2_batch_size,
            threshold=gliner2_threshold,
            max_len=gliner2_max_len,
            anchor_normalization_mode=anchor_normalization_mode,
            label_input_mode=gliner2_label_input_mode,
            per_sentence_vote=per_sentence_vote,
            all_paths=all_paths,
            hub_fanout_threshold=hub_fanout_threshold,
            max_pool_sentences=max_pool_sentences,
            candidate_labels=resolved_candidate_labels,
        )
    if resolved_backend == "lmstudio":
        if llm_no_hdbscan:
            return label_pairs_with_llm_no_hdbscan(
                clusters,
                base_url=llm_base_url,
                model=llm_model,
                temperature=llm_temperature,
                timeout_secs=llm_timeout_secs,
                vis_out_path=llm_no_hdbscan_vis_path,
                max_evidence_sents=llm_max_evidence_sents,
                candidate_labels=resolved_candidate_labels,
            )
        if llm_path_vote:
            return label_clusters_with_llm_path_vote(
                clusters,
                base_url=llm_base_url,
                model=llm_model,
                temperature=llm_temperature,
                timeout_secs=llm_timeout_secs,
                candidate_labels=resolved_candidate_labels,
            )
        return label_clusters_with_lmstudio(
            clusters,
            base_url=llm_base_url,
            model=llm_model,
            temperature=llm_temperature,
            timeout_secs=llm_timeout_secs,
            max_evidence_sents=llm_max_evidence_sents,
            per_path_vote=llm_per_path_vote,
            all_paths=all_paths,
            hub_fanout_threshold=hub_fanout_threshold,
            max_pool_sentences=max_pool_sentences,
            use_agglomerative=llm_agglomerative,
            agglom_distance_threshold=llm_agglom_distance,
            encoder_model=encoder_model,
            vis_out_path=llm_agglom_vis_path,
            llm_path_vote=False,
            agglom_encoder=llm_agglom_encoder,
            candidate_labels=resolved_candidate_labels,
        )
    raise ValueError(f"Unsupported cluster label backend: {backend}")


def label_clusters_with_oracle(
    clusters: Dict[int, List[Dict]],
    gt_relationships: List[Dict],
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    """Oracle upper-bound labeler: assign GT labels to each cluster via
    path_score-weighted majority vote from ground-truth (diag, med) pairs.
    Clusters with no GT-matched paths receive the highest-priority fallback label.
    This gives the labeling ceiling achievable when path extraction is fixed."""
    gt_pair_labels: Dict[Tuple[int, int], str] = {}
    for rel in gt_relationships:
        key = (int(rel["diag_idx"]), int(rel["drug_idx"]))
        gt_pair_labels[key] = _normalize_rel_type(str(rel.get("rel_type", "")))

    tie_break_order = _resolve_candidate_labels(candidate_labels)
    cluster_labels: Dict[int, str] = {}
    cluster_details: Dict[int, Dict[str, object]] = {}

    print("  Using oracle cluster labeler (GT majority-vote per cluster)")
    for cid, cpaths in sorted(clusters.items()):
        votes: Dict[str, float] = defaultdict(float)
        for path in cpaths:
            key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
            gt_lbl = gt_pair_labels.get(key)
            if gt_lbl and gt_lbl in tie_break_order:
                votes[gt_lbl] += float(path.get("path_score", 1.0))

        if votes:
            label = max(
                votes,
                key=lambda lbl: (
                    votes[lbl],
                    -tie_break_order.index(lbl) if lbl in tie_break_order else 0,
                ),
            )
        else:
            label = tie_break_order[0]

        unique_sent_count = len({int(p["sent_idx"]) for p in cpaths})
        cluster_labels[cid] = label
        cluster_details[cid] = {
            "backend": "oracle",
            "label_source": "gt_majority_vote",
            "gt_vote_weights": {k: round(v, 4) for k, v in votes.items()},
            "n_occurrences": len(cpaths),
            "n_unique_sentences": unique_sent_count,
            "supporting_evidence": _build_supporting_evidence(cpaths),
        }
        print(
            f"  Cluster {cid:3d}  ({unique_sent_count:2d} sents / {len(cpaths):3d} paths)  "
            f"-> {label} [oracle]  votes={dict(votes)}"
        )

    return cluster_labels, cluster_details


def promote_discontinued_cluster_labels(
    clusters: Dict[int, List[Dict]],
    cluster_name_map: Dict[int, str],
) -> Tuple[Dict[int, str], Dict[int, str]]:
    promoted = dict(cluster_name_map)
    reasons: Dict[int, str] = {}

    for cid, cpaths in sorted(clusters.items()):
        current = promoted.get(cid, "")
        scores = _keyword_scores(cpaths)
        discontinue_score = scores.get("DISCONTINUED", 0)
        adverse_score = scores.get("ADVERSE_EFFECT", 0)
        treat_score = scores.get("TREATS", 0)
        explicit_stop_hits = _explicit_discontinue_hits(cpaths)

        if explicit_stop_hits == 0 and discontinue_score == 0:
            continue

        should_promote = False
        if current == "TREATS":
            should_promote = (
                explicit_stop_hits >= 2
                and discontinue_score >= max(1, treat_score - 1)
                and discontinue_score >= adverse_score
            ) or (
                explicit_stop_hits >= 1
                and discontinue_score >= max(1, treat_score)
                and adverse_score == 0
            )
        elif current == "ADVERSE_EFFECT":
            should_promote = (
                explicit_stop_hits >= 2
                and discontinue_score >= adverse_score
            )

        if should_promote:
            promoted[cid] = "DISCONTINUED"
            reasons[cid] = (
                f"explicit_stop_hits={explicit_stop_hits}, discontinue={discontinue_score}, "
                f"adverse={adverse_score}, treats={treat_score}"
            )

    return promoted, reasons


def _aggregate_pair_label_refined_detail(
    label: str,
    cluster_paths: List[Dict],
    pair_keys: List[Tuple[int, int]],
    pair_label_details: Dict[Tuple[int, int], Dict[str, object]],
    candidate_labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    label_scores: Dict[str, float] = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
    label_counts: Dict[str, float] = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
    backends: List[str] = []
    supporting_evidence_records: List[Dict[str, object]] = []
    n_scored_occurrences = 0

    for pair in pair_keys:
        detail = pair_label_details.get(pair) or {}
        backend_name = str(detail.get("backend", "")).strip()
        if backend_name and backend_name not in backends:
            backends.append(backend_name)

        raw_scores = detail.get("label_scores")
        if isinstance(raw_scores, dict):
            for rel_type, raw_score in raw_scores.items():
                rel_key = _normalize_rel_type(str(rel_type))
                if rel_key not in label_scores:
                    continue
                try:
                    label_scores[rel_key] += float(raw_score)
                except (TypeError, ValueError):
                    continue

        raw_counts = detail.get("label_counts")
        if isinstance(raw_counts, dict):
            for rel_type, raw_count in raw_counts.items():
                rel_key = _normalize_rel_type(str(rel_type))
                if rel_key not in label_counts:
                    continue
                try:
                    label_counts[rel_key] += float(raw_count)
                except (TypeError, ValueError):
                    continue
        elif label in label_counts:
            label_counts[label] += 1.0

        supporting_evidence = detail.get("supporting_evidence")
        if isinstance(supporting_evidence, list):
            supporting_evidence_records.extend(
                record for record in supporting_evidence if isinstance(record, dict)
            )

        try:
            n_scored_occurrences += int(detail.get("n_scored_occurrences", 0) or 0)
        except (TypeError, ValueError):
            pass

    if label in label_scores and max(label_scores.values(), default=0.0) <= 0.0:
        label_scores[label] = float(max(len(pair_keys), 1))
    if label in label_counts and max(label_counts.values(), default=0.0) <= 0.0:
        label_counts[label] = float(len(pair_keys))

    return {
        "backend": "+".join(backends) if backends else "pair_label_refinement",
        "label_source": "pair_label_refine_split",
        "score_type": "pair_label_aggregated_votes",
        "label_input_mode": "pair_evidence",
        "label_scores": {rel_type: round(score, 4) for rel_type, score in label_scores.items()},
        "label_counts": {rel_type: round(count, 4) for rel_type, count in label_counts.items()},
        "fallback_reason": None,
        "n_occurrences": len(cluster_paths),
        "n_unique_sentences": len({int(path["sent_idx"]) for path in cluster_paths}),
        "n_scored_occurrences": max(n_scored_occurrences, len(pair_keys)),
        "supporting_evidence": _build_supporting_evidence(
            cluster_paths,
            evidence_records=supporting_evidence_records,
        ),
    }


def _path_identity_key(path: Dict) -> Tuple[int, int, int]:
    return (
        int(path.get("diag_row_idx", -1)),
        int(path.get("sent_idx", -1)),
        int(path.get("med_row_idx", -1)),
    )


def _label_path_with_lmstudio(
    path: Dict,
    base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
    model: str = LMSTUDIO_DEFAULT_MODEL,
    temperature: float = LMSTUDIO_DEFAULT_TEMPERATURE,
    timeout_secs: int = LMSTUDIO_DEFAULT_TIMEOUT_SECS,
    candidate_labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    resolved_candidate_labels = list(candidate_labels or _preferred_rel_type_order())
    system_prompt = _build_lmstudio_system_prompt(resolved_candidate_labels)
    diag_text = (
        _extract_row_field(path.get("diag_row_text", ""), "diagnosis")
        or " ".join(str(path.get("diag_row_text", "")).split())
    )
    med_text = (
        _extract_row_field(path.get("med_row_text", ""), "drug")
        or " ".join(str(path.get("med_row_text", "")).split())
    )
    user_msg = (
        f"Diagnosis: {diag_text}\n"
        f"Evidence: {' '.join(str(path.get('sent_text', '')).split())}\n"
        f"Medication: {med_text}\n"
        "Output:"
    )

    cache_key = _lmstudio_cache_key(system_prompt, user_msg)
    fallback_reason = None
    if cache_key in _LMSTUDIO_LABEL_CACHE:
        parsed_label = _LMSTUDIO_LABEL_CACHE[cache_key]
    else:
        try:
            raw_response = _call_lmstudio(
                base_url,
                model,
                system_prompt,
                user_msg,
                temperature,
                timeout_secs,
            )
            parsed_label = _parse_lmstudio_label(raw_response, resolved_candidate_labels) or ""
        except Exception as exc:
            if _should_abort_lmstudio_fallback(exc):
                raise
            parsed_label = ""
            fallback_reason = str(exc)
        _LMSTUDIO_LABEL_CACHE[cache_key] = parsed_label

    if parsed_label:
        label = parsed_label
        label_source = "lmstudio_path_direct"
        backend_name = "lmstudio"
    else:
        label = _keyword_classify([path])
        label_source = "keyword_fallback"
        backend_name = "keyword"
        if fallback_reason is None:
            fallback_reason = "no_label_parsed"

    path_score = max(float(path.get("path_score", 0.0)), 0.0)
    vote_weight = path_score if path_score > 0.0 else 1.0
    return {
        "label": label,
        "label_source": label_source,
        "backend": backend_name,
        "fallback_reason": fallback_reason,
        "sent_idx": int(path.get("sent_idx", -1)),
        "section_name": str(path.get("section_name", "")),
        "sentence": " ".join(str(path.get("sent_text", "")).split())[:220],
        "vote_weight": vote_weight,
        "path_score": path_score,
        "confidence": None,
    }


def _aggregate_path_label_refined_detail(
    label: str,
    cluster_paths: List[Dict],
    path_label_details: Dict[Tuple[int, int, int], Dict[str, object]],
    candidate_labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    label_scores: Dict[str, float] = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
    label_counts: Dict[str, float] = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
    backends: List[str] = []
    supporting_evidence_records: List[Dict[str, object]] = []
    n_scored_occurrences = 0

    for path in cluster_paths:
        detail = path_label_details.get(_path_identity_key(path)) or {}
        backend_name = str(detail.get("backend", "")).strip()
        if backend_name and backend_name not in backends:
            backends.append(backend_name)

        path_label = _normalize_rel_type(str(detail.get("label", label))) or label
        vote_weight = _to_float_or_none(detail.get("vote_weight"))
        if vote_weight is None:
            vote_weight = max(float(path.get("path_score", 0.0)), 1.0)
        if path_label in label_scores:
            label_scores[path_label] += float(vote_weight)
            label_counts[path_label] += 1.0

        supporting_evidence_records.append({
            "sent_idx": int(detail.get("sent_idx", path.get("sent_idx", -1))),
            "section_name": str(detail.get("section_name", path.get("section_name", ""))),
            "sentence": str(detail.get("sentence", " ".join(str(path.get("sent_text", "")).split())[:220])),
            "label": path_label,
            "confidence": _to_float_or_none(detail.get("confidence")),
            "vote_weight": float(vote_weight),
            "path_score": float(detail.get("path_score", path.get("path_score", 0.0)) or 0.0),
        })
        n_scored_occurrences += 1

    if label in label_scores and max(label_scores.values(), default=0.0) <= 0.0:
        label_scores[label] = float(max(len(cluster_paths), 1))
    if label in label_counts and max(label_counts.values(), default=0.0) <= 0.0:
        label_counts[label] = float(len(cluster_paths))

    return {
        "backend": "+".join(backends) if backends else "pair_path_refinement",
        "label_source": "pair_path_refine_split",
        "score_type": "path_label_votes",
        "label_input_mode": "per_path_direct",
        "label_scores": {rel_type: round(score, 4) for rel_type, score in label_scores.items()},
        "label_counts": {rel_type: round(count, 4) for rel_type, count in label_counts.items()},
        "fallback_reason": None,
        "n_occurrences": len(cluster_paths),
        "n_unique_sentences": len({int(path["sent_idx"]) for path in cluster_paths}),
        "n_scored_occurrences": max(n_scored_occurrences, len(cluster_paths)),
        "supporting_evidence": _build_supporting_evidence(
            cluster_paths,
            evidence_records=supporting_evidence_records,
        ),
    }


def _semantic_subcluster_pair_keys(
    pair_keys: List[Tuple[int, int]],
    pair_buckets: Dict[Tuple[int, int], List[Dict]],
    refined_sentences: Optional[torch.Tensor],
    refined_rows: Optional[torch.Tensor] = None,
    n_diag: Optional[int] = None,
    sentence_encoder: Optional[SentenceTransformer] = None,
    embedding_mode: str = "contextual_sentence_average",
    distance_threshold: float = 0.20,
) -> List[List[Tuple[int, int]]]:
    resolved_pair_keys = [
        pair_key
        for pair_key in sorted({(int(diag_idx), int(med_idx)) for diag_idx, med_idx in pair_keys})
        if pair_key in pair_buckets
    ]
    if len(resolved_pair_keys) <= 1 or refined_sentences is None:
        return [resolved_pair_keys]

    threshold = max(float(distance_threshold), 0.0)
    if threshold <= 0.0:
        return [resolved_pair_keys]

    semantic_paths: List[Dict] = []
    for pair_key in resolved_pair_keys:
        semantic_paths.extend(pair_buckets[pair_key])

    try:
        ordered_pair_keys, pair_embeddings = _compute_pair_embeddings(
            semantic_paths,
            refined_sentences,
            refined_rows=refined_rows,
            n_diag=n_diag,
            sentence_encoder=sentence_encoder,
            embedding_mode=embedding_mode,
            verbose=False,
        )
        if len(ordered_pair_keys) <= 1 or pair_embeddings.shape[0] <= 1:
            return [resolved_pair_keys]

        from sklearn.cluster import AgglomerativeClustering as _AggClust  # type: ignore

        agg = _AggClust(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        )
        semantic_group_ids = agg.fit_predict(_to_numpy_array(pair_embeddings, dtype=np.float32)).tolist()
    except Exception:
        return [resolved_pair_keys]

    semantic_groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for pair_key, semantic_group_id in zip(ordered_pair_keys, semantic_group_ids):
        semantic_groups[int(semantic_group_id)].append(pair_key)

    if len(semantic_groups) <= 1:
        return [resolved_pair_keys]

    return [
        list(group_pair_keys)
        for _, group_pair_keys in sorted(
            semantic_groups.items(),
            key=lambda item: (-len(item[1]), tuple(item[1])),
        )
    ]


def _refine_clusters_by_pair_labels(
    paths: List[Dict],
    clusters: Dict[int, List[Dict]],
    labels: np.ndarray,
    backend: str = DEFAULT_CLUSTER_LABEL_BACKEND,
    current_cluster_name_map: Optional[Dict[int, str]] = None,
    current_cluster_label_details: Optional[Dict[int, Dict[str, object]]] = None,
    min_cluster_pairs: int = 5,
    refined_sentences: Optional[torch.Tensor] = None,
    refined_rows: Optional[torch.Tensor] = None,
    n_diag: Optional[int] = None,
    pair_embedding_mode: str = "contextual_sentence_average",
    semantic_subsplit: bool = False,
    semantic_distance_threshold: float = 0.20,
    path_subsplit: bool = False,
    path_subsplit_min_score_mass: float = 0.25,
    path_subsplit_min_score_share: float = 0.30,
    path_subsplit_max_dominant_gap: float = 0.12,
    gliner2_model: str = GLINER2_MODEL,
    gliner2_batch_size: int = 8,
    gliner2_threshold: float = 0.5,
    gliner2_max_len: Optional[int] = 384,
    anchor_normalization_mode: str = "legacy",
    gliner2_label_input_mode: str = DEFAULT_GLINER2_LABEL_INPUT_MODE,
    per_sentence_vote: bool = False,
    all_paths: Optional[List[Dict]] = None,
    hub_fanout_threshold: float = 0.3,
    max_pool_sentences: int = 12,
    llm_base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
    llm_model: str = LMSTUDIO_DEFAULT_MODEL,
    llm_temperature: float = LMSTUDIO_DEFAULT_TEMPERATURE,
    llm_timeout_secs: int = LMSTUDIO_DEFAULT_TIMEOUT_SECS,
    llm_max_evidence_sents: int = LMSTUDIO_DEFAULT_MAX_EVIDENCE_SENTS,
    llm_per_path_vote: bool = False,
    gt_relationships: Optional[List[Dict]] = None,
    encoder_model: Optional[Any] = None,
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[Dict[int, List[Dict]], np.ndarray, Dict[int, str], Dict[int, Dict[str, object]], Dict[str, object]]:
    resolved_backend = (backend or DEFAULT_CLUSTER_LABEL_BACKEND).strip().lower()
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    semantic_subsplit_enabled = bool(semantic_subsplit)
    path_subsplit_enabled = bool(path_subsplit) and resolved_backend == "lmstudio" and bool(llm_per_path_vote)
    resolved_path_subsplit_min_score_mass = max(float(path_subsplit_min_score_mass), 0.0)
    resolved_path_subsplit_min_score_share = min(max(float(path_subsplit_min_score_share), 0.0), 1.0)
    resolved_path_subsplit_max_dominant_gap = max(float(path_subsplit_max_dominant_gap), 0.0)
    refined_name_map = dict(current_cluster_name_map or {})
    refined_label_details = {
        int(cluster_id): dict(detail)
        for cluster_id, detail in (current_cluster_label_details or {}).items()
    }
    stats: Dict[str, object] = {
        "enabled": True,
        "backend": resolved_backend,
        "min_cluster_pairs": max(int(min_cluster_pairs), 2),
        "llm_per_path_vote": bool(llm_per_path_vote) if resolved_backend == "lmstudio" else False,
        "semantic_subsplit": semantic_subsplit_enabled,
        "semantic_distance_threshold": round(max(float(semantic_distance_threshold), 0.0), 4),
        "path_subsplit": path_subsplit_enabled,
        "path_subsplit_min_score_mass": round(resolved_path_subsplit_min_score_mass, 4),
        "path_subsplit_min_score_share": round(resolved_path_subsplit_min_score_share, 4),
        "path_subsplit_max_dominant_gap": round(resolved_path_subsplit_max_dominant_gap, 4),
        "parent_clusters_considered": 0,
        "parent_clusters_split": 0,
        "semantic_parent_clusters_split": 0,
        "path_parent_clusters_split": 0,
        "child_clusters_added": 0,
        "semantic_child_clusters_added": 0,
        "path_child_clusters_added": 0,
        "path_split_candidates": 0,
        "path_split_rejected": 0,
        "path_split_folded_paths": 0,
        "pairs_scored": 0,
        "pairs_reassigned": 0,
        "paths_reassigned": 0,
        "reason": "no_candidate_clusters",
        "split_clusters": [],
    }

    if not paths or not clusters:
        stats["reason"] = "no_paths"
        return clusters, labels, refined_name_map, refined_label_details, stats

    candidate_cluster_ids = [
        cid
        for cid, cpaths in sorted(clusters.items())
        if len(_bucket_paths_by_pair(cpaths)) >= max(int(min_cluster_pairs), 2)
    ]
    if not candidate_cluster_ids:
        return clusters, labels, refined_name_map, refined_label_details, stats

    pair_clusters: Dict[int, List[Dict]] = {}
    pair_cluster_lookup: Dict[int, Tuple[int, int]] = {}
    pair_to_parent_cluster: Dict[Tuple[int, int], int] = {}
    temp_cluster_id = 0
    for cid in candidate_cluster_ids:
        for pair_key, pair_paths in sorted(_bucket_paths_by_pair(clusters[cid]).items()):
            pair_clusters[temp_cluster_id] = list(pair_paths)
            pair_cluster_lookup[temp_cluster_id] = pair_key
            pair_to_parent_cluster[pair_key] = cid
            temp_cluster_id += 1

    if not pair_clusters:
        return clusters, labels, refined_name_map, refined_label_details, stats

    stats["parent_clusters_considered"] = len(candidate_cluster_ids)
    stats["pairs_scored"] = len(pair_clusters)
    print(
        f"  Pair-label refinement: scoring {len(pair_clusters)} pairs from "
        f"{len(candidate_cluster_ids)} large raw clusters using {resolved_backend} evidence labels"
    )
    if resolved_backend == "lmstudio" and llm_per_path_vote:
        print("  Pair-label refinement: using LMStudio per-path voting within each pair cluster")
    if path_subsplit_enabled:
        print(
            "  Pair-label refinement: splitting mixed-evidence same-pair paths by per-path labels "
            f"(min_mass={resolved_path_subsplit_min_score_mass:.2f}, "
            f"min_share={resolved_path_subsplit_min_score_share:.2f}, "
            f"max_gap={resolved_path_subsplit_max_dominant_gap:.2f})"
        )

    preferred_order = list(resolved_candidate_labels)
    pair_label_map: Dict[Tuple[int, int], str] = {}
    pair_label_details: Dict[Tuple[int, int], Dict[str, object]] = {}
    path_level_label_records: Dict[Tuple[int, int, int], Dict[str, object]] = {}

    if current_cluster_name_map is None and resolved_backend == "lmstudio":
        # Single-pass path-level labeling: label all paths in all clusters
        print(f"  Single-pass path-level labeling: labeling all paths via LMStudio...")
        for cid, cpaths in sorted(clusters.items()):
            for path in cpaths:
                path_key = _path_identity_key(path)
                if path_key not in path_level_label_records:
                    detail = _label_path_with_lmstudio(
                        path,
                        base_url=llm_base_url,
                        model=llm_model,
                        temperature=llm_temperature,
                        timeout_secs=llm_timeout_secs,
                        candidate_labels=preferred_order,
                    )
                    path_level_label_records[path_key] = detail

        # Build pair labels by aggregating path-level labels
        for temp_id, pair_key in pair_cluster_lookup.items():
            pair_paths = pair_clusters[temp_id]
            vote_scores = {lbl: 0.0 for lbl in preferred_order}
            vote_counts = {lbl: 0.0 for lbl in preferred_order}
            n_scored = 0
            supporting_evidence_records = []
            backends = []
            
            for path in pair_paths:
                path_key = _path_identity_key(path)
                detail = path_level_label_records[path_key]
                plabel = _normalize_rel_type(str(detail.get("label", ""))) or _keyword_classify([path], candidate_labels=preferred_order)
                pscore = max(float(path.get("path_score", 0.0)), 0.0)
                
                if plabel in vote_scores:
                    vote_scores[plabel] += pscore
                    vote_counts[plabel] += 1.0
                    n_scored += 1
                
                backend_name = str(detail.get("backend", "")).strip()
                if backend_name and backend_name not in backends:
                    backends.append(backend_name)
                supporting_evidence_records.append({
                    "sent_idx": int(detail.get("sent_idx", path.get("sent_idx", -1))),
                    "section_name": str(detail.get("section_name", path.get("section_name", ""))),
                    "sentence": str(detail.get("sentence", " ".join(str(path.get("sent_text", "")).split())[:220])),
                    "label": plabel,
                    "confidence": _to_float_or_none(detail.get("confidence")),
                    "vote_weight": float(detail.get("vote_weight", pscore)),
                    "path_score": float(detail.get("path_score", path.get("path_score", 0.0)) or 0.0),
                })
                
            unique_sent_count = len({int(p["sent_idx"]) for p in pair_paths})
            if max(vote_scores.values(), default=0.0) > 0.0:
                label = min(
                    preferred_order,
                    key=lambda lbl: (-vote_scores.get(lbl, 0.0), preferred_order.index(lbl)),
                )
                label_source = "lmstudio_path_vote"
            elif max(vote_counts.values(), default=0.0) > 0.0:
                label = min(
                    preferred_order,
                    key=lambda lbl: (-vote_counts.get(lbl, 0.0), preferred_order.index(lbl)),
                )
                label_source = "lmstudio_path_vote_count"
            else:
                label = _keyword_classify(pair_paths, candidate_labels=preferred_order)
                label_source = "keyword_fallback"
                
            pair_label_map[pair_key] = label
            pair_label_details[pair_key] = {
                "backend": "+".join(backends) if backends else "lmstudio",
                "label_source": label_source,
                "score_type": "lmstudio_pair_vote_weight",
                "label_input_mode": "per_path_direct",
                "label_scores": vote_scores,
                "label_counts": vote_counts,
                "fallback_reason": None,
                "n_occurrences": len(pair_paths),
                "n_unique_sentences": unique_sent_count,
                "n_scored_occurrences": n_scored,
                "supporting_evidence": _build_supporting_evidence(pair_paths, evidence_records=supporting_evidence_records),
            }
    else:
        # Legacy path: Round 1 was executed and provided current_cluster_name_map
        pair_cluster_name_map, pair_cluster_label_details = label_clusters(
            pair_clusters,
            backend=resolved_backend,
            gliner2_model=gliner2_model,
            gliner2_batch_size=gliner2_batch_size,
            gliner2_threshold=gliner2_threshold,
            gliner2_max_len=gliner2_max_len,
            anchor_normalization_mode=anchor_normalization_mode,
            gliner2_label_input_mode=gliner2_label_input_mode,
            per_sentence_vote=per_sentence_vote,
            all_paths=all_paths,
            hub_fanout_threshold=hub_fanout_threshold,
            max_pool_sentences=max_pool_sentences,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_timeout_secs=llm_timeout_secs,
            llm_max_evidence_sents=llm_max_evidence_sents,
            llm_per_path_vote=bool(llm_per_path_vote) if resolved_backend == "lmstudio" else False,
            llm_agglomerative=False,
            gt_relationships=gt_relationships,
            encoder_model=encoder_model,
            llm_agglom_vis_path=None,
            llm_path_vote=False,
            llm_no_hdbscan=False,
            llm_no_hdbscan_vis_path=None,
            llm_agglom_encoder="bge",
            candidate_labels=resolved_candidate_labels,
        )
        
        for temp_id, pair_key in pair_cluster_lookup.items():
            pair_label_map[pair_key] = pair_cluster_name_map.get(
                temp_id,
                _keyword_classify(pair_clusters[temp_id], candidate_labels=resolved_candidate_labels),
            )
            pair_label_details[pair_key] = dict(pair_cluster_label_details.get(temp_id, {}))

    sentence_encoder = getattr(encoder_model, "sentence_encoder", None)
    next_cluster_id = (max(clusters) + 1) if clusters else 0
    pair_to_final_cluster: Dict[Tuple[int, int], int] = {}
    path_to_final_cluster: Dict[Tuple[int, int, int], int] = {}
    refined_clusters: Dict[int, List[Dict]] = defaultdict(list)

    def _bucket_pair_paths_by_path_label(pair_paths: List[Dict]) -> Dict[str, List[Dict]]:
        label_to_paths: Dict[str, List[Dict]] = defaultdict(list)
        for path in pair_paths:
            path_key = _path_identity_key(path)
            detail = path_level_label_records.get(path_key)
            if detail is None:
                detail = _label_path_with_lmstudio(
                    path,
                    base_url=llm_base_url,
                    model=llm_model,
                    temperature=llm_temperature,
                    timeout_secs=llm_timeout_secs,
                    candidate_labels=preferred_order,
                )
                path_level_label_records[path_key] = detail
            label_to_paths[
                _normalize_rel_type(str(detail.get("label", "")))
                or _keyword_classify([path], candidate_labels=preferred_order)
            ].append(path)
        return label_to_paths

    for cid, cpaths in sorted(clusters.items()):
        pair_buckets = _bucket_paths_by_pair(cpaths)
        if cid not in candidate_cluster_ids or len(pair_buckets) < 2:
            refined_clusters[cid].extend(cpaths)
            for pair_key in pair_buckets:
                pair_to_final_cluster[pair_key] = cid
            for path in cpaths:
                path_to_final_cluster[_path_identity_key(path)] = cid
                
            if current_cluster_name_map is None:
                # Compute label for this non-candidate cluster
                vote_scores = {lbl: 0.0 for lbl in preferred_order}
                vote_counts = {lbl: 0.0 for lbl in preferred_order}
                n_scored = 0
                supporting_evidence_records = []
                backends = []
                
                for path in cpaths:
                    path_key = _path_identity_key(path)
                    detail = path_level_label_records.get(path_key)
                    if detail is None:
                        detail = _label_path_with_lmstudio(
                            path,
                            base_url=llm_base_url,
                            model=llm_model,
                            temperature=llm_temperature,
                            timeout_secs=llm_timeout_secs,
                            candidate_labels=preferred_order,
                        )
                        path_level_label_records[path_key] = detail
                    plabel = _normalize_rel_type(str(detail.get("label", ""))) or _keyword_classify([path], candidate_labels=preferred_order)
                    pscore = max(float(path.get("path_score", 0.0)), 0.0)
                    
                    if plabel in vote_scores:
                        vote_scores[plabel] += pscore
                        vote_counts[plabel] += 1.0
                        n_scored += 1
                    
                    backend_name = str(detail.get("backend", "")).strip()
                    if backend_name and backend_name not in backends:
                        backends.append(backend_name)
                    supporting_evidence_records.append({
                        "sent_idx": int(detail.get("sent_idx", path.get("sent_idx", -1))),
                        "section_name": str(detail.get("section_name", path.get("section_name", ""))),
                        "sentence": str(detail.get("sentence", " ".join(str(path.get("sent_text", "")).split())[:220])),
                        "label": plabel,
                        "confidence": _to_float_or_none(detail.get("confidence")),
                        "vote_weight": float(detail.get("vote_weight", pscore)),
                        "path_score": float(detail.get("path_score", path.get("path_score", 0.0)) or 0.0),
                    })
                
                unique_sent_count = len({int(p["sent_idx"]) for p in cpaths})
                if max(vote_scores.values(), default=0.0) > 0.0:
                    label = min(
                        preferred_order,
                        key=lambda lbl: (-vote_scores.get(lbl, 0.0), preferred_order.index(lbl)),
                    )
                    label_source = "lmstudio_path_vote"
                elif max(vote_counts.values(), default=0.0) > 0.0:
                    label = min(
                        preferred_order,
                        key=lambda lbl: (-vote_counts.get(lbl, 0.0), preferred_order.index(lbl)),
                    )
                    label_source = "lmstudio_path_vote_count"
                else:
                    label = _keyword_classify(cpaths, candidate_labels=preferred_order)
                    label_source = "keyword_fallback"
                
                refined_name_map[cid] = label
                refined_label_details[cid] = {
                    "backend": "+".join(backends) if backends else "lmstudio",
                    "label_source": label_source,
                    "score_type": "lmstudio_vote_weight",
                    "label_input_mode": "per_path_direct",
                    "label_scores": vote_scores,
                    "label_counts": vote_counts,
                    "fallback_reason": None,
                    "n_occurrences": len(cpaths),
                    "n_unique_sentences": unique_sent_count,
                    "n_scored_occurrences": n_scored,
                    "supporting_evidence": _build_supporting_evidence(cpaths, evidence_records=supporting_evidence_records),
                }
            continue

        label_to_pairs: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for pair_key in sorted(pair_buckets):
            pair_label = pair_label_map.get(pair_key)
            if not pair_label:
                pair_label = refined_name_map.get(
                    cid,
                    _keyword_classify(pair_buckets[pair_key], candidate_labels=preferred_order),
                )
            label_to_pairs[pair_label].append(pair_key)

        semantic_group_specs: List[Tuple[str, List[Tuple[int, int]], int, int]] = []
        parent_semantic_split = False
        semantic_children_added = 0
        for rel_type, rel_pair_keys in sorted(
            label_to_pairs.items(),
            key=lambda item: (
                -len(item[1]),
                preferred_order.index(item[0]) if item[0] in preferred_order else len(preferred_order),
            ),
        ):
            semantic_groups = [sorted(rel_pair_keys)]
            if semantic_subsplit_enabled and len(rel_pair_keys) >= 2:
                semantic_groups = _semantic_subcluster_pair_keys(
                    rel_pair_keys,
                    pair_buckets,
                    refined_sentences,
                    refined_rows=refined_rows,
                    n_diag=n_diag,
                    sentence_encoder=sentence_encoder,
                    embedding_mode=pair_embedding_mode,
                    distance_threshold=semantic_distance_threshold,
                )
                if len(semantic_groups) > 1:
                    parent_semantic_split = True
                    semantic_children_added += len(semantic_groups) - 1
                    print(
                        f"    Semantic sub-split: raw cluster {cid} label {rel_type} -> "
                        f"{len(semantic_groups)} groups from {len(rel_pair_keys)} pairs"
                    )

            for semantic_group_index, group_pair_keys in enumerate(semantic_groups):
                semantic_group_specs.append((
                    rel_type,
                    group_pair_keys,
                    semantic_group_index,
                    len(semantic_groups),
                ))

        ordered_groups = sorted(
            semantic_group_specs,
            key=lambda item: (
                -len(item[1]),
                preferred_order.index(item[0]) if item[0] in preferred_order else len(preferred_order),
                item[2],
            ),
        )
        parent_children: List[Dict[str, object]] = []
        parent_path_split = False
        path_children_added = 0
        parent_paths_reassigned = 0

        for rel_type, child_pair_keys, semantic_group_index, semantic_group_count in ordered_groups:
            primary_child_paths: List[Dict] = []
            primary_child_pair_keys: List[Tuple[int, int]] = []
            deferred_path_children: List[Dict[str, object]] = []

            for pair_key in child_pair_keys:
                pair_paths = list(pair_buckets[pair_key])
                if path_subsplit_enabled and len(pair_paths) >= 2:
                    path_label_buckets = _bucket_pair_paths_by_path_label(pair_paths)
                    if len(path_label_buckets) > 1:
                        path_label_scores = {
                            split_label: sum(
                                max(
                                    float(
                                        (_to_float_or_none((path_level_label_records.get(_path_identity_key(path)) or {}).get("vote_weight"))
                                         if path_level_label_records.get(_path_identity_key(path)) is not None else None)
                                        or float(path.get("path_score", 0.0))
                                    ),
                                    1e-4,
                                )
                                for path in split_paths
                            )
                            for split_label, split_paths in path_label_buckets.items()
                        }
                        total_path_label_score = max(float(sum(path_label_scores.values())), 1e-6)
                        dominant_path_label = (
                            rel_type
                            if rel_type in path_label_buckets
                            else min(
                                preferred_order,
                                key=lambda split_label: (
                                    -path_label_scores.get(split_label, 0.0),
                                    preferred_order.index(split_label) if split_label in preferred_order else len(preferred_order),
                                ),
                            )
                        )
                        dominant_score = float(path_label_scores.get(dominant_path_label, 0.0))
                        dominant_paths = list(path_label_buckets.get(dominant_path_label, []))
                        for split_label, split_paths in sorted(
                            path_label_buckets.items(),
                            key=lambda item: (
                                -path_label_scores.get(item[0], 0.0),
                                preferred_order.index(item[0]) if item[0] in preferred_order else len(preferred_order),
                            ),
                        ):
                            if split_label == dominant_path_label:
                                continue
                            stats["path_split_candidates"] = int(stats["path_split_candidates"]) + 1
                            child_score = float(path_label_scores.get(split_label, 0.0))
                            child_share = child_score / total_path_label_score
                            dominant_gap = max(dominant_score - child_score, 0.0)
                            keep_path_child = (
                                (child_score >= resolved_path_subsplit_min_score_mass
                                 or child_share >= resolved_path_subsplit_min_score_share)
                                and dominant_gap <= resolved_path_subsplit_max_dominant_gap
                            )
                            if not keep_path_child:
                                stats["path_split_rejected"] = int(stats["path_split_rejected"]) + 1
                                stats["path_split_folded_paths"] = int(stats["path_split_folded_paths"]) + len(split_paths)
                                dominant_paths.extend(split_paths)
                                continue
                            parent_path_split = True
                            path_children_added += 1
                            parent_paths_reassigned += len(split_paths)
                            deferred_path_children.append({
                                "label": split_label,
                                "pair_key": pair_key,
                                "paths": list(split_paths),
                                "path_score_mass": round(child_score, 4),
                                "path_score_share": round(child_share, 4),
                                "dominant_gap": round(dominant_gap, 4),
                                "dominant_label": dominant_path_label,
                            })
                        pair_paths = dominant_paths or pair_paths

                if pair_paths:
                    primary_child_paths.extend(pair_paths)
                    primary_child_pair_keys.append(pair_key)

            if primary_child_paths:
                parent_children.append({
                    "label": rel_type,
                    "paths": primary_child_paths,
                    "pair_keys": primary_child_pair_keys,
                    "semantic_group_index": int(semantic_group_index),
                    "semantic_group_count": int(semantic_group_count),
                    "split_mode": "pair_label",
                })

            for path_child in deferred_path_children:
                parent_children.append({
                    "label": str(path_child.get("label", rel_type)),
                    "paths": list(path_child.get("paths", [])),
                    "pair_keys": [path_child.get("pair_key")],
                    "semantic_group_index": int(semantic_group_index),
                    "semantic_group_count": int(semantic_group_count),
                    "split_mode": "path_label",
                    "source_pair_key": path_child.get("pair_key"),
                    "path_split_score_mass": path_child.get("path_score_mass"),
                    "path_split_score_share": path_child.get("path_score_share"),
                    "path_split_dominant_gap": path_child.get("dominant_gap"),
                    "path_split_dominant_label": path_child.get("dominant_label"),
                })

        if len(parent_children) <= 1:
            refined_clusters[cid].extend(cpaths)
            for pair_key in pair_buckets:
                pair_to_final_cluster[pair_key] = cid
            for path in cpaths:
                path_to_final_cluster[_path_identity_key(path)] = cid
            continue

        ordered_children = sorted(
            parent_children,
            key=lambda child: (
                -len(child["paths"]),
                preferred_order.index(child["label"]) if child["label"] in preferred_order else len(preferred_order),
                0 if child.get("split_mode") == "pair_label" else 1,
                int(child.get("semantic_group_index", -1)),
            ),
        )
        dominant_label = str(ordered_children[0]["label"])
        split_summary: List[Dict[str, object]] = []

        for index, child in enumerate(ordered_children):
            child_cluster_id = cid if index == 0 else next_cluster_id
            if index > 0:
                next_cluster_id += 1

            child_label = str(child["label"])
            child_paths = list(child["paths"])
            child_pair_keys = list(child.get("pair_keys", []))
            split_mode = str(child.get("split_mode", "pair_label"))

            refined_clusters[child_cluster_id].extend(child_paths)
            refined_name_map[child_cluster_id] = child_label

            if split_mode == "path_label":
                child_detail = _aggregate_path_label_refined_detail(
                    child_label,
                    child_paths,
                    path_level_label_records,
                    candidate_labels=preferred_order,
                )
                source_pair_key = child.get("source_pair_key")
                if isinstance(source_pair_key, tuple) and len(source_pair_key) == 2:
                    child_detail["pair_label_refinement_source_pair"] = [
                        int(source_pair_key[0]),
                        int(source_pair_key[1]),
                    ]
            else:
                child_detail = _aggregate_pair_label_refined_detail(
                    child_label,
                    child_paths,
                    child_pair_keys,
                    pair_label_details,
                    candidate_labels=preferred_order,
                )

            child_detail["refinement_parent_cluster_id"] = cid
            child_detail["pair_label_refinement_split_mode"] = split_mode
            child_detail["pair_label_refinement_semantic_group_index"] = int(child.get("semantic_group_index", -1))
            child_detail["pair_label_refinement_semantic_group_count"] = int(child.get("semantic_group_count", 0))
            child_detail["pair_label_refinement_semantic_distance"] = (
                round(max(float(semantic_distance_threshold), 0.0), 4)
                if semantic_subsplit_enabled
                else None
            )
            if split_mode == "path_label":
                child_detail["pair_label_refinement_path_score_mass"] = _to_float_or_none(child.get("path_split_score_mass"))
                child_detail["pair_label_refinement_path_score_share"] = _to_float_or_none(child.get("path_split_score_share"))
                child_detail["pair_label_refinement_path_dominant_gap"] = _to_float_or_none(child.get("path_split_dominant_gap"))
                child_detail["pair_label_refinement_path_dominant_label"] = str(child.get("path_split_dominant_label", ""))
            refined_label_details[child_cluster_id] = child_detail

            split_record: Dict[str, object] = {
                "cluster_id": child_cluster_id,
                "label": child_label,
                "n_pairs": len({(int(diag_idx), int(med_idx)) for diag_idx, med_idx in child_pair_keys}),
                "n_paths": len(child_paths),
                "semantic_group_index": int(child.get("semantic_group_index", -1)),
                "semantic_group_count": int(child.get("semantic_group_count", 0)),
                "split_mode": split_mode,
            }
            if split_mode == "path_label":
                source_pair_key = child.get("source_pair_key")
                if isinstance(source_pair_key, tuple) and len(source_pair_key) == 2:
                    split_record["source_pair"] = [
                        int(source_pair_key[0]),
                        int(source_pair_key[1]),
                    ]
                split_record["path_score_mass"] = _to_float_or_none(child.get("path_split_score_mass"))
                split_record["path_score_share"] = _to_float_or_none(child.get("path_split_score_share"))
                split_record["path_dominant_gap"] = _to_float_or_none(child.get("path_split_dominant_gap"))
                split_record["path_dominant_label"] = str(child.get("path_split_dominant_label", ""))
            split_summary.append(split_record)

            if split_mode != "path_label":
                for pair_key in child_pair_keys:
                    pair_to_final_cluster[pair_key] = child_cluster_id
                    if child_cluster_id != cid:
                        try:
                            pair_to_parent_cluster[pair_key] = cid
                        except Exception:
                            pass

            for path in child_paths:
                path_key = _path_identity_key(path)
                path_to_final_cluster[path_key] = child_cluster_id
                path["pair_label_refine_parent_cluster_id"] = int(cid)
                path["pair_label_refine_label"] = child_label
                path["pair_label_refine_semantic_group_index"] = int(child.get("semantic_group_index", -1))
                path["pair_label_refine_semantic_group_count"] = int(child.get("semantic_group_count", 0))
                path["pair_label_refine_path_split"] = (split_mode == "path_label")
                if split_mode == "path_label":
                    path_detail = path_level_label_records.get(path_key) or {}
                    path["pair_label_refine_path_label"] = _normalize_rel_type(str(path_detail.get("label", child_label))) or child_label
                    path["pair_label_refine_path_label_source"] = str(path_detail.get("label_source", ""))

        dominant_detail = refined_label_details.get(cid, {})
        dominant_detail["pair_label_refinement_children"] = split_summary
        dominant_detail["pair_label_refinement_dominant_label"] = dominant_label
        dominant_detail["pair_label_refinement_semantic_subsplit"] = parent_semantic_split
        dominant_detail["pair_label_refinement_path_subsplit"] = parent_path_split
        refined_label_details[cid] = dominant_detail

        stats["parent_clusters_split"] = int(stats["parent_clusters_split"]) + 1
        if parent_semantic_split:
            stats["semantic_parent_clusters_split"] = int(stats["semantic_parent_clusters_split"]) + 1
            stats["semantic_child_clusters_added"] = int(stats["semantic_child_clusters_added"]) + semantic_children_added
        if parent_path_split:
            stats["path_parent_clusters_split"] = int(stats["path_parent_clusters_split"]) + 1
            stats["path_child_clusters_added"] = int(stats["path_child_clusters_added"]) + path_children_added
            stats["paths_reassigned"] = int(stats["paths_reassigned"]) + parent_paths_reassigned
        stats["child_clusters_added"] = int(stats["child_clusters_added"]) + max(len(ordered_children) - 1, 0)
        stats["pairs_reassigned"] = int(stats["pairs_reassigned"]) + sum(
            len(child.get("pair_keys", []))
            for child in ordered_children[1:]
            if str(child.get("split_mode", "pair_label")) != "path_label"
        )
        split_clusters_records = stats.get("split_clusters")
        if isinstance(split_clusters_records, list):
            split_clusters_records.append({
                "parent_cluster_id": cid,
                "n_pairs": len(pair_buckets),
                "semantic_subsplit": parent_semantic_split,
                "path_subsplit": parent_path_split,
                "children": split_summary,
            })

    if int(stats["parent_clusters_split"]) == 0:
        stats["reason"] = "no_mixed_clusters"
        return clusters, labels, refined_name_map, refined_label_details, stats

    refined_labels = np.asarray(
        [
            path_to_final_cluster.get(
                _path_identity_key(path),
                pair_to_final_cluster.get((int(path["diag_row_idx"]), int(path["med_row_idx"])), int(label)),
            )
            for path, label in zip(paths, labels)
        ],
        dtype=int,
    )
    stats["reason"] = "applied"
    print(
        f"  Pair-label refinement: split {int(stats['parent_clusters_split'])} raw clusters into "
        f"{int(stats['child_clusters_added']) + int(stats['parent_clusters_split'])} label/semantic/path-homogeneous groups"
    )
    return dict(refined_clusters), refined_labels, refined_name_map, refined_label_details, stats


def _build_negative_refinement_child_rescue_detail(
    label: str,
    cluster_paths: List[Dict],
    parent_cluster_id: int,
    support_child_cluster_ids: List[int],
    cluster_label_details: Optional[Dict[int, Dict[str, object]]] = None,
    candidate_labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    resolved_candidate_labels = _resolve_candidate_labels(candidate_labels)
    label_scores: Dict[str, float] = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
    label_counts: Dict[str, float] = {rel_type: 0.0 for rel_type in resolved_candidate_labels}
    backends: List[str] = []
    supporting_evidence_records: List[Dict[str, object]] = []
    n_scored_occurrences = 0

    for cluster_id in support_child_cluster_ids:
        detail = (cluster_label_details or {}).get(int(cluster_id), {}) or {}
        backend_name = str(detail.get("backend", "")).strip()
        if backend_name and backend_name not in backends:
            backends.append(backend_name)

        raw_scores = detail.get("label_scores")
        if isinstance(raw_scores, dict):
            for rel_type, raw_score in raw_scores.items():
                rel_key = _normalize_rel_type(str(rel_type))
                if rel_key not in label_scores:
                    continue
                try:
                    label_scores[rel_key] += float(raw_score)
                except (TypeError, ValueError):
                    continue

        raw_counts = detail.get("label_counts")
        if isinstance(raw_counts, dict):
            for rel_type, raw_count in raw_counts.items():
                rel_key = _normalize_rel_type(str(rel_type))
                if rel_key not in label_counts:
                    continue
                try:
                    label_counts[rel_key] += float(raw_count)
                except (TypeError, ValueError):
                    continue

        supporting_evidence = detail.get("supporting_evidence")
        if isinstance(supporting_evidence, list):
            supporting_evidence_records.extend(
                record for record in supporting_evidence if isinstance(record, dict)
            )

        try:
            n_scored_occurrences += int(detail.get("n_scored_occurrences", 0) or 0)
        except (TypeError, ValueError):
            pass

    normalized_label = _normalize_rel_type(str(label)) or str(label)
    if normalized_label in label_scores and max(label_scores.values(), default=0.0) <= 0.0:
        label_scores[normalized_label] = float(max(len(cluster_paths), 1))
    if normalized_label in label_counts and max(label_counts.values(), default=0.0) <= 0.0:
        label_counts[normalized_label] = float(max(len(cluster_paths), 1))

    return {
        "backend": "+".join(backends) if backends else "negative_refinement_child_rescue",
        "label_source": "negative_refinement_child_rescue",
        "score_type": "refinement_child_support",
        "label_input_mode": "refinement_child_support",
        "label_scores": {rel_type: round(score, 4) for rel_type, score in label_scores.items()},
        "label_counts": {rel_type: round(count, 4) for rel_type, count in label_counts.items()},
        "fallback_reason": None,
        "n_occurrences": len(cluster_paths),
        "n_unique_sentences": len({int(path["sent_idx"]) for path in cluster_paths}),
        "n_scored_occurrences": max(n_scored_occurrences, len(cluster_paths)),
        "supporting_evidence": _build_supporting_evidence(
            cluster_paths,
            evidence_records=supporting_evidence_records,
        ),
        "low_signal_rescue_mode": "refinement_child_support",
        "low_signal_rescue_parent_cluster_id": int(parent_cluster_id),
        "low_signal_rescue_support_child_cluster_ids": [
            int(cluster_id) for cluster_id in support_child_cluster_ids
        ],
    }


def _rescue_negative_clusters_from_refinement_children(
    paths: List[Dict],
    labels: np.ndarray,
    clusters: Dict[int, List[Dict]],
    cluster_name_map: Dict[int, str],
    cluster_label_details: Dict[int, Dict[str, object]],
    cluster_pair_label_refinement_stats: Optional[Dict[str, object]] = None,
    candidate_labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[int, List[Dict]], Dict[int, str], Dict[int, Dict[str, object]], Dict[str, object]]:
    stats: Dict[str, object] = {
        "enabled": True,
        "support_child_groups_considered": 0,
        "rescued_groups": [],
        "rescued_cluster_ids": [],
        "rescued_pairs": [],
        "rescued_parent_clusters": [],
        "reason": "no_negative_refinement_children",
    }
    if not paths or not clusters or len(labels) != len(paths):
        stats["reason"] = "no_paths"
        return labels, clusters, cluster_name_map, cluster_label_details, stats

    split_clusters = (cluster_pair_label_refinement_stats or {}).get("split_clusters")
    if not isinstance(split_clusters, list) or not split_clusters:
        return labels, clusters, cluster_name_map, cluster_label_details, stats

    pair_snapshot = _pair_recovery_stage_snapshot(
        paths,
        labels=labels,
        cluster_name_map=cluster_name_map,
        cluster_label_details=cluster_label_details,
    )
    pair_records = pair_snapshot.get("pairs") if isinstance(pair_snapshot, dict) else None
    if not isinstance(pair_records, list) or not pair_records:
        stats["reason"] = "no_refined_pair_snapshot"
        return labels, clusters, cluster_name_map, cluster_label_details, stats

    support_children_by_med: Dict[int, Dict[str, object]] = {}
    for pair_record in pair_records:
        if not isinstance(pair_record, dict):
            continue
        try:
            resolved_cluster_id = int(pair_record.get("cluster_id", -1))
            med_row_idx = int(pair_record.get("med_row_idx", -1))
        except (TypeError, ValueError):
            continue
        if resolved_cluster_id < 0 or med_row_idx < 0:
            continue

        cluster_detail = cluster_label_details.get(int(resolved_cluster_id), {}) or {}
        try:
            refinement_parent_cluster_id = cluster_detail.get("refinement_parent_cluster_id")
        except (TypeError, ValueError):
            refinement_parent_cluster_id = None
        if refinement_parent_cluster_id is None:
            continue
        split_mode = str(
            pair_record.get(
                "pair_label_refinement_split_mode",
                cluster_detail.get("pair_label_refinement_split_mode", ""),
            )
        ).strip().lower()
        if split_mode not in {"pair_label", "path_label"}:
            continue
        cluster_label = _normalize_rel_type(
            str(pair_record.get("cluster_label", cluster_name_map.get(resolved_cluster_id, "")))
        )
        if cluster_label != "DISCONTINUED":
            continue

        label_scores = cluster_detail.get("label_scores") if isinstance(cluster_detail.get("label_scores"), dict) else {}
        label_counts = cluster_detail.get("label_counts") if isinstance(cluster_detail.get("label_counts"), dict) else {}
        dis_score = _to_float_or_none(label_scores.get("DISCONTINUED")) or 0.0
        treat_score = _to_float_or_none(label_scores.get("TREATS")) or 0.0
        negative_count = _to_float_or_none(label_counts.get("NEGATIVE")) or 0.0
        if dis_score <= 0.0 or treat_score > 0.0 or negative_count > 0.0:
            continue

        if split_mode == "path_label":
            child_share = _to_float_or_none(cluster_detail.get("pair_label_refinement_path_score_share"))
            child_gap = _to_float_or_none(cluster_detail.get("pair_label_refinement_path_dominant_gap"))
            if child_share is None or child_gap is None:
                continue
            if child_share < 0.45 or child_gap > 0.03:
                continue

        if int(cluster_detail.get("n_scored_occurrences", 0) or 0) <= 0:
            continue

        support_sent_ids = {
            int(record.get("sent_idx", -1))
            for record in cluster_detail.get("supporting_evidence", [])
            if isinstance(record, dict) and int(record.get("sent_idx", -1)) >= 0
        }
        if not support_sent_ids:
            continue

        med_support = support_children_by_med.setdefault(med_row_idx, {
            "med_row_idx": int(med_row_idx),
            "label": "DISCONTINUED",
            "support_child_cluster_ids": [],
            "support_sent_ids": set(),
        })
        support_child_cluster_ids = med_support["support_child_cluster_ids"]
        if resolved_cluster_id not in support_child_cluster_ids:
            support_child_cluster_ids.append(int(resolved_cluster_id))
        med_support_sent_ids = med_support["support_sent_ids"]
        if isinstance(med_support_sent_ids, set):
            med_support_sent_ids.update(support_sent_ids)

    if not support_children_by_med:
        stats["reason"] = "no_dis_support_children"
        return labels, clusters, cluster_name_map, cluster_label_details, stats

    stats["support_child_groups_considered"] = len(support_children_by_med)
    next_cluster_id = (max(int(cluster_id) for cluster_id in clusters) + 1) if clusters else 0
    pair_to_rescued_cluster: Dict[Tuple[int, Tuple[int, int]], int] = {}
    pair_rescue_metadata: Dict[Tuple[int, Tuple[int, int]], Dict[str, object]] = {}
    rescued_cluster_names: Dict[int, str] = {}
    rescued_cluster_details: Dict[int, Dict[str, object]] = {}
    rescue_group_records: List[Dict[str, object]] = []

    for parent_cluster_id, parent_paths in sorted(clusters.items()):
        if _normalize_rel_type(str(cluster_name_map.get(parent_cluster_id, ""))) != "NEGATIVE":
            continue
        parent_detail = cluster_label_details.get(int(parent_cluster_id), {}) or {}
        if parent_detail.get("refinement_parent_cluster_id") is None:
            continue
        parent_split_mode = str(parent_detail.get("pair_label_refinement_split_mode", "")).strip().lower()
        if parent_split_mode not in {"pair_label", "path_label"}:
            continue

        resolved_parent_paths = list(parent_paths or [])
        if not resolved_parent_paths:
            continue
        parent_pair_buckets = _bucket_paths_by_pair(resolved_parent_paths)
        for pair_key, pair_paths in sorted(parent_pair_buckets.items()):
            resolved_pair_key = (int(pair_key[0]), int(pair_key[1]))
            med_support = support_children_by_med.get(int(resolved_pair_key[1]))
            if not isinstance(med_support, dict):
                continue
            support_sent_ids = set(med_support.get("support_sent_ids") or set())
            if not support_sent_ids:
                continue

            pair_sent_ids = {
                int(path.get("sent_idx", -1))
                for path in pair_paths
                if int(path.get("sent_idx", -1)) >= 0
            }
            support_overlap_sent_ids = sorted(pair_sent_ids & support_sent_ids)
            if not support_overlap_sent_ids:
                continue

            pair_sections = {
                str(path.get("section_name", "")).strip()
                for path in pair_paths
                if str(path.get("section_name", "")).strip()
            }
            if len(pair_paths) < 3:
                continue
            if len(pair_sent_ids) < 3:
                continue
            if len(pair_sections) < 2:
                continue

            rescued_cluster_id = next_cluster_id
            next_cluster_id += 1
            rescued_cluster_names[int(rescued_cluster_id)] = "DISCONTINUED"
            rescued_cluster_details[int(rescued_cluster_id)] = _build_negative_refinement_child_rescue_detail(
                "DISCONTINUED",
                list(pair_paths),
                int(parent_cluster_id),
                list(med_support.get("support_child_cluster_ids") or []),
                cluster_label_details=cluster_label_details,
                candidate_labels=candidate_labels,
            )
            rescued_cluster_details[int(rescued_cluster_id)]["low_signal_rescue_group_key"] = {
                "parent_cluster_id": int(parent_cluster_id),
                "med_row_idx": int(resolved_pair_key[1]),
                "label": "DISCONTINUED",
            }
            rescued_cluster_details[int(rescued_cluster_id)]["low_signal_rescue_support_overlap_sent_ids"] = list(
                support_overlap_sent_ids
            )

            rescue_group_records.append({
                "rescued_cluster_id": int(rescued_cluster_id),
                "parent_cluster_id": int(parent_cluster_id),
                "med_row_idx": int(resolved_pair_key[1]),
                "label": "DISCONTINUED",
                "support_child_cluster_ids": [
                    int(cluster_id) for cluster_id in list(med_support.get("support_child_cluster_ids") or [])
                ],
                "support_sent_ids": sorted(int(sent_idx) for sent_idx in support_sent_ids),
                "support_overlap_sent_ids": list(support_overlap_sent_ids),
                "rescued_pairs": [{
                    "diag_row_idx": int(resolved_pair_key[0]),
                    "med_row_idx": int(resolved_pair_key[1]),
                }],
            })
            pair_to_rescued_cluster[(int(parent_cluster_id), resolved_pair_key)] = int(rescued_cluster_id)
            pair_rescue_metadata[(int(parent_cluster_id), resolved_pair_key)] = {
                "rescued_cluster_id": int(rescued_cluster_id),
                "parent_cluster_id": int(parent_cluster_id),
                "support_child_cluster_ids": [
                    int(cluster_id) for cluster_id in list(med_support.get("support_child_cluster_ids") or [])
                ],
            }

    if not pair_to_rescued_cluster:
        stats["reason"] = "no_matching_negative_pairs"
        return labels, clusters, cluster_name_map, cluster_label_details, stats

    updated_labels: List[int] = []
    for path, lbl in zip(paths, labels):
        original_cluster_id = int(lbl)
        pair_key = (int(path.get("diag_row_idx", -1)), int(path.get("med_row_idx", -1)))
        metadata = pair_rescue_metadata.get((original_cluster_id, pair_key))
        rescued_cluster_id = int(metadata.get("rescued_cluster_id")) if metadata else original_cluster_id
        updated_labels.append(int(rescued_cluster_id))
        if metadata:
            path["raw_cluster_id"] = int(rescued_cluster_id)
            path["negative_refinement_child_rescued"] = True
            path["negative_refinement_parent_cluster_id"] = int(metadata.get("parent_cluster_id", -1))
            path["negative_refinement_support_child_cluster_ids"] = list(
                metadata.get("support_child_cluster_ids") or []
            )

    labels = np.asarray(updated_labels, dtype=int)
    updated_clusters: Dict[int, List[Dict]] = defaultdict(list)
    for path, lbl in zip(paths, labels):
        updated_clusters[int(lbl)].append(path)

    updated_cluster_name_map = {int(cluster_id): name for cluster_id, name in cluster_name_map.items()}
    updated_cluster_label_details = {
        int(cluster_id): detail for cluster_id, detail in cluster_label_details.items()
    }
    updated_cluster_name_map.update(rescued_cluster_names)
    updated_cluster_label_details.update(rescued_cluster_details)

    stats["rescued_groups"] = rescue_group_records
    stats["rescued_cluster_ids"] = [int(record["rescued_cluster_id"]) for record in rescue_group_records]
    stats["rescued_pairs"] = [
        pair_record
        for record in rescue_group_records
        for pair_record in list(record.get("rescued_pairs") or [])
    ]
    stats["rescued_parent_clusters"] = sorted(
        {int(record["parent_cluster_id"]) for record in rescue_group_records}
    )
    stats["reason"] = "applied"
    return labels, dict(updated_clusters), updated_cluster_name_map, updated_cluster_label_details, stats


def _suppress_negative_labeled_clusters(
    paths: List[Dict],
    clusters: Dict[int, List[Dict]],
    labels: np.ndarray,
    current_cluster_name_map: Optional[Dict[int, str]] = None,
    current_cluster_label_details: Optional[Dict[int, Dict[str, object]]] = None,
    negative_pairs: Optional[set[Tuple[int, int]]] = None,
    keep_annotated_negative_clusters: bool = True,
    cluster_pair_label_refinement_stats: Optional[Dict[str, object]] = None,
    candidate_labels: Optional[List[str]] = None,
    enable_refinement_child_rescue: bool = False,
) -> Tuple[List[Dict], Dict[int, List[Dict]], np.ndarray, Dict[int, str], Dict[int, Dict[str, object]], Dict[str, object]]:
    filtered_name_map = dict(current_cluster_name_map or {})
    filtered_label_details = {
        int(cluster_id): dict(detail)
        for cluster_id, detail in (current_cluster_label_details or {}).items()
    }
    stats: Dict[str, object] = {
        "enabled": True,
        "keep_annotated_negative_clusters": bool(keep_annotated_negative_clusters),
        "negative_clusters_considered": 0,
        "negative_clusters_suppressed": 0,
        "negative_clusters_kept": 0,
        "annotated_negative_clusters_kept": 0,
        "suppressed_pairs": 0,
        "suppressed_paths": 0,
        "refinement_child_rescue_enabled": bool(enable_refinement_child_rescue),
        "refinement_child_rescue_groups": [],
        "refinement_child_rescue_cluster_ids": [],
        "refinement_child_rescue_pairs": [],
        "refinement_child_rescue_parent_clusters": [],
        "refinement_child_rescue_reason": "disabled" if not enable_refinement_child_rescue else "no_negative_refinement_children",
        "reason": "no_paths",
        "suppressed_cluster_ids": [],
    }

    if not paths or not clusters:
        return paths, clusters, labels, filtered_name_map, filtered_label_details, stats

    if enable_refinement_child_rescue:
        labels, clusters, filtered_name_map, filtered_label_details, refinement_child_rescue_stats = _rescue_negative_clusters_from_refinement_children(
            paths,
            labels,
            clusters,
            filtered_name_map,
            filtered_label_details,
            cluster_pair_label_refinement_stats=cluster_pair_label_refinement_stats,
            candidate_labels=candidate_labels,
        )
        stats["refinement_child_rescue_groups"] = list(
            refinement_child_rescue_stats.get("rescued_groups") or []
        )
        stats["refinement_child_rescue_cluster_ids"] = list(
            refinement_child_rescue_stats.get("rescued_cluster_ids") or []
        )
        stats["refinement_child_rescue_pairs"] = list(
            refinement_child_rescue_stats.get("rescued_pairs") or []
        )
        stats["refinement_child_rescue_parent_clusters"] = list(
            refinement_child_rescue_stats.get("rescued_parent_clusters") or []
        )
        stats["refinement_child_rescue_reason"] = str(
            refinement_child_rescue_stats.get("reason", "")
        )

    negative_pair_set = set(negative_pairs or set())
    negative_cluster_ids: List[int] = []
    annotated_negative_cluster_ids: set[int] = set()
    suppressed_cluster_ids: List[int] = []

    for cid, cpaths in sorted(clusters.items()):
        cluster_label = _normalize_rel_type(filtered_name_map.get(cid, ""))
        if cluster_label != "NEGATIVE":
            continue
        negative_cluster_ids.append(int(cid))
        if not negative_pair_set or not keep_annotated_negative_clusters:
            continue
        cluster_pairs = set(_bucket_paths_by_pair(cpaths))
        if cluster_pairs & negative_pair_set:
            annotated_negative_cluster_ids.add(int(cid))

    stats["negative_clusters_considered"] = len(negative_cluster_ids)
    if not negative_cluster_ids:
        stats["reason"] = "no_negative_clusters"
        return paths, clusters, labels, filtered_name_map, filtered_label_details, stats

    for cid in negative_cluster_ids:
        if cid in annotated_negative_cluster_ids:
            continue
        suppressed_cluster_ids.append(cid)

    if not suppressed_cluster_ids:
        stats["negative_clusters_kept"] = len(negative_cluster_ids)
        stats["annotated_negative_clusters_kept"] = len(annotated_negative_cluster_ids)
        stats["reason"] = "only_annotated_negative_clusters"
        return paths, clusters, labels, filtered_name_map, filtered_label_details, stats

    suppressed_cluster_id_set = set(suppressed_cluster_ids)
    filtered_paths: List[Dict] = []
    filtered_labels: List[int] = []
    for path, lbl in zip(paths, labels):
        cluster_id = int(lbl)
        if cluster_id in suppressed_cluster_id_set:
            path["negative_cluster_suppressed"] = True
            path["raw_cluster_id"] = -1
            continue
        filtered_paths.append(path)
        filtered_labels.append(cluster_id)

    filtered_clusters = {
        int(cid): list(cpaths)
        for cid, cpaths in clusters.items()
        if int(cid) not in suppressed_cluster_id_set
    }
    for cid in suppressed_cluster_ids:
        filtered_name_map.pop(int(cid), None)
        filtered_label_details.pop(int(cid), None)

    stats["negative_clusters_suppressed"] = len(suppressed_cluster_ids)
    stats["negative_clusters_kept"] = len(negative_cluster_ids) - len(suppressed_cluster_ids)
    stats["annotated_negative_clusters_kept"] = len(annotated_negative_cluster_ids)
    stats["suppressed_pairs"] = sum(
        len(_bucket_paths_by_pair(clusters[cid]))
        for cid in suppressed_cluster_ids
    )
    stats["suppressed_paths"] = sum(len(clusters[cid]) for cid in suppressed_cluster_ids)
    stats["suppressed_cluster_ids"] = suppressed_cluster_ids
    stats["reason"] = "applied"
    print(
        f"  Negative-cluster suppression: dropped {len(suppressed_cluster_ids)} NEGATIVE clusters "
        f"({int(stats['suppressed_pairs'])} pairs / {int(stats['suppressed_paths'])} paths)"
    )
    if annotated_negative_cluster_ids:
        print(
            f"  Negative-cluster suppression: kept {len(annotated_negative_cluster_ids)} annotated negative clusters"
        )

    return (
        filtered_paths,
        filtered_clusters,
        np.asarray(filtered_labels, dtype=int),
        filtered_name_map,
        filtered_label_details,
        stats,
    )


def filter_candidate_pairs(
    paths: List[Dict],
    gamma: float,
    diag_top_k: int = 8,
    med_top_k: int = 8,
    score_margin: float = 0.03,
    hub_fanout: int = 6,
    mode: str = "legacy",
    collect_details: bool = False,
) -> Tuple[List[Dict], Dict]:
    resolved_mode = (mode or "legacy").strip().lower()
    if resolved_mode not in {"legacy", "weak_only"}:
        raise ValueError(f"Unsupported pair filter mode: {mode}")

    if not paths:
        return paths, {
            "enabled": True,
            "n_pairs_before": 0,
            "n_pairs_after": 0,
            "dropped_pairs": 0,
            "dropped_paths": 0,
            "mode": resolved_mode,
            "reason": "no_paths",
        }

    pair_buckets = _bucket_paths_by_pair(paths)
    sent_pair_members: Dict[int, set] = defaultdict(set)
    for pair, pair_paths in pair_buckets.items():
        for sent_idx in {int(path["sent_idx"]) for path in pair_paths}:
            sent_pair_members[sent_idx].add(pair)

    pair_best_score: Dict[Tuple[int, int], float] = {
        pair: max(float(path.get("path_score", 0.0)) for path in pair_paths)
        for pair, pair_paths in pair_buckets.items()
    }

    diag_ranks: Dict[Tuple[int, int], int] = {}
    med_ranks: Dict[Tuple[int, int], int] = {}
    diag_groups: Dict[int, List[Tuple[Tuple[int, int], float]]] = defaultdict(list)
    med_groups: Dict[int, List[Tuple[Tuple[int, int], float]]] = defaultdict(list)
    for pair, best_score in pair_best_score.items():
        diag_groups[pair[0]].append((pair, best_score))
        med_groups[pair[1]].append((pair, best_score))

    for items in diag_groups.values():
        items.sort(key=lambda item: (-item[1], item[0][1]))
        for rank, (pair, _score) in enumerate(items, start=1):
            diag_ranks[pair] = rank

    for items in med_groups.values():
        items.sort(key=lambda item: (-item[1], item[0][0]))
        for rank, (pair, _score) in enumerate(items, start=1):
            med_ranks[pair] = rank

    kept_pairs: set = set()
    dropped_pairs: List[Tuple[int, int]] = []
    pair_decisions: List[Dict[str, object]] = []
    for pair, pair_paths in pair_buckets.items():
        support_count = len({int(path["sent_idx"]) for path in pair_paths})
        signal_strength = _cluster_signal_strength(pair_paths)
        best_score = pair_best_score[pair]
        max_sentence_fanout = max(len(sent_pair_members[int(path["sent_idx"])]) for path in pair_paths)
        diag_rank = diag_ranks.get(pair, diag_top_k + 1)
        med_rank = med_ranks.get(pair, med_top_k + 1)
        in_rank = diag_rank <= diag_top_k or med_rank <= med_top_k
        strong_score = best_score >= (gamma + score_margin)

        should_drop = support_count == 1 and not in_rank and not strong_score and max_sentence_fanout >= hub_fanout
        if resolved_mode == "legacy":
            should_drop = should_drop and signal_strength == 0

        keep = not should_drop

        if keep:
            kept_pairs.add(pair)
        else:
            dropped_pairs.append(pair)

        if collect_details:
            if not keep:
                decision_reason = "weak_single_sentence_hub"
            elif support_count >= 2:
                decision_reason = "multi_sentence_support"
            elif signal_strength > 0:
                decision_reason = "signal_strength"
            elif in_rank:
                decision_reason = "top_rank"
            elif strong_score:
                decision_reason = "strong_score"
            elif resolved_mode == "weak_only":
                decision_reason = "weak_only_keep"
            else:
                decision_reason = "legacy_keep"
            pair_decisions.append({
                "diag_row_idx": pair[0],
                "med_row_idx": pair[1],
                "kept": keep,
                "reason": decision_reason,
                "support_count": support_count,
                "signal_strength": signal_strength,
                "best_score": round(best_score, 4),
                "diag_rank": diag_rank,
                "med_rank": med_rank,
                "in_rank": in_rank,
                "strong_score": strong_score,
                "max_sentence_fanout": max_sentence_fanout,
            })

    filtered_paths = [
        path for path in paths
        if (path["diag_row_idx"], path["med_row_idx"]) in kept_pairs
    ]

    dropped_path_count = len(paths) - len(filtered_paths)
    print(
        f"  Pair filter: kept {len(kept_pairs)}/{len(pair_buckets)} candidate pairs "
        f"(dropped {len(dropped_pairs)} weak single-sentence hub pairs; {dropped_path_count} paths removed)"
    )

    stats = {
        "enabled": True,
        "mode": resolved_mode,
        "n_pairs_before": len(pair_buckets),
        "n_pairs_after": len(kept_pairs),
        "dropped_pairs": len(dropped_pairs),
        "dropped_paths": dropped_path_count,
        "diag_top_k": diag_top_k,
        "med_top_k": med_top_k,
        "score_margin": score_margin,
        "hub_fanout": hub_fanout,
    }
    if collect_details:
        stats["pair_decisions"] = pair_decisions

    return filtered_paths, stats


def filter_cluster_pair_tails(
    paths: List[Dict],
    labels: np.ndarray,
    keep_rank: int = 2,
    score_margin: float = 0.01,
    mode: str = "legacy",
    collect_details: bool = False,
    adaptive_lambda: float = 0.5,
    adaptive_percentile: float = 25.0,
    rescue_unique_evidence: bool = True,
) -> Tuple[List[Dict], np.ndarray, Dict]:
    resolved_mode = (mode or "legacy").strip().lower()
    if resolved_mode not in {"legacy", "conservative", "soft_weight", "adaptive_std", "adaptive_percentile"}:
        raise ValueError(f"Unsupported cluster tail mode: {mode}")

    effective_keep_rank = keep_rank
    effective_score_margin = score_margin
    keep_all_pairs = False
    if resolved_mode == "conservative":
        effective_keep_rank = max(keep_rank, 3)
        effective_score_margin = max(score_margin, 0.02)
    elif resolved_mode == "soft_weight":
        keep_all_pairs = True

    if not paths:
        return paths, labels, {
            "enabled": True,
            "dropped_pairs": 0,
            "dropped_paths": 0,
            "clusters_touched": 0,
            "mode": resolved_mode,
            "keep_rank": effective_keep_rank,
            "score_margin": effective_score_margin,
            "reason": "no_paths",
        }

    pair_buckets = _bucket_paths_by_pair(paths)
    pair_best_score = {
        pair: max(float(path.get("path_score", 0.0)) for path in pair_paths)
        for pair, pair_paths in pair_buckets.items()
    }
    pair_support = {
        pair: len({int(path["sent_idx"]) for path in pair_paths})
        for pair, pair_paths in pair_buckets.items()
    }

    cluster_pairs: Dict[int, set] = defaultdict(set)
    pair_cluster_id: Dict[Tuple[int, int], int] = {}
    for path, lbl in zip(paths, labels):
        pair = (path["diag_row_idx"], path["med_row_idx"])
        resolved_label = int(lbl)
        cluster_pairs[resolved_label].add(pair)
        pair_cluster_id[pair] = resolved_label

    kept_pairs: set = set()
    dropped_pairs: set = set()
    clusters_touched = 0
    pair_decisions: List[Dict[str, object]] = []
    cluster_leader_scores: Dict[int, float] = {}
    cluster_lexical_gap_allowance: Dict[int, float] = {}

    for cid, members in cluster_pairs.items():
        ranked_members = sorted(
            members,
            key=lambda pair: (
                pair_best_score[pair],
                pair_support[pair],
                -pair[0],
                -pair[1],
            ),
            reverse=True,
        )
        if len(ranked_members) <= keep_rank:
            kept_pairs.update(ranked_members)
            continue

        leader_score = pair_best_score[ranked_members[0]]
        # Per-cluster adaptive margin computation
        cluster_effective_score_margin = effective_score_margin
        cluster_effective_keep_rank = effective_keep_rank
        if resolved_mode == "adaptive_std":
            _cls = np.array([pair_best_score[p] for p in ranked_members])
            cluster_effective_score_margin = max(0.005, adaptive_lambda * float(np.std(_cls)))
        elif resolved_mode == "adaptive_percentile":
            _cls = np.array([pair_best_score[p] for p in ranked_members])
            _pct = float(np.percentile(_cls, adaptive_percentile))
            cluster_effective_score_margin = max(0.0, leader_score - _pct)
            cluster_effective_keep_rank = max(keep_rank, 2)
        cluster_leader_scores[cid] = leader_score
        cluster_lexical_gap_allowance[cid] = max(0.02, 2.0 * cluster_effective_score_margin)
        cluster_dropped = False
        for rank, pair in enumerate(ranked_members, start=1):
            within_keep_rank = rank <= cluster_effective_keep_rank
            within_margin = pair_best_score[pair] >= (leader_score - cluster_effective_score_margin)
            repeated_support = pair_support[pair] >= 2
            keep = keep_all_pairs or within_keep_rank or within_margin or repeated_support
            if keep:
                kept_pairs.add(pair)
            else:
                dropped_pairs.add(pair)
                cluster_dropped = True
            if collect_details:
                if keep_all_pairs and not (within_keep_rank or within_margin or repeated_support):
                    decision_reason = "soft_weight_no_drop"
                elif within_keep_rank:
                    decision_reason = "keep_rank"
                elif within_margin:
                    decision_reason = "leader_margin"
                elif repeated_support:
                    decision_reason = "repeated_support"
                else:
                    decision_reason = "tail_drop"
                pair_decisions.append({
                    "cluster_id": cid,
                    "diag_row_idx": pair[0],
                    "med_row_idx": pair[1],
                    "rank": rank,
                    "kept": keep,
                    "reason": decision_reason,
                    "best_score": round(pair_best_score[pair], 4),
                    "leader_score": round(leader_score, 4),
                    "score_gap": round(max(0.0, leader_score - pair_best_score[pair]), 4),
                    "support_count": pair_support[pair],
                })
        if cluster_dropped:
            clusters_touched += 1

    # Unique-evidence rescue: if a dropped pair introduces at least one sentence
    # not already covered by any kept pair in the cluster, keep it regardless of
    # score.  This is type-agnostic - it does not assume which relationship type
    # the pair represents; it only preserves sentence-level diversity so the
    # downstream labeler sees all distinct evidence.
    unique_sent_rescued: set = set()
    if rescue_unique_evidence and dropped_pairs:
        kept_sentences: set = set()
        for kp in kept_pairs:
            for path in pair_buckets.get(kp, []):
                sidx = int(path.get("sent_idx", -1))
                if sidx >= 0:
                    kept_sentences.add(sidx)
        for pair in list(dropped_pairs):
            pair_sents = {
                int(p.get("sent_idx", -1))
                for p in pair_buckets.get(pair, [])
                if int(p.get("sent_idx", -1)) >= 0
            }
            if pair_sents and not pair_sents.issubset(kept_sentences):
                kept_pairs.add(pair)
                dropped_pairs.discard(pair)
                unique_sent_rescued.add(pair)
                kept_sentences |= pair_sents
                if collect_details:
                    cid = pair_cluster_id.get(pair, -1)
                    leader_score = float(cluster_leader_scores.get(cid, pair_best_score[pair]))
                    pair_decisions.append({
                        "cluster_id": cid,
                        "diag_row_idx": pair[0],
                        "med_row_idx": pair[1],
                        "rank": None,
                        "kept": True,
                        "reason": "unique_evidence_rescue",
                        "best_score": round(pair_best_score[pair], 4),
                        "leader_score": round(leader_score, 4),
                        "score_gap": round(max(0.0, leader_score - pair_best_score[pair]), 4),
                        "support_count": pair_support[pair],
                    })
        if unique_sent_rescued:
            print(f"  Unique-evidence rescue: recovered {len(unique_sent_rescued)} dropped pairs with uncovered evidence sentences")

    lexical_cue_rescued: set = set()
    if dropped_pairs:
        rescue_labels = ["TREATS", "ADVERSE_EFFECT", "DISCONTINUED"]
        for pair in list(dropped_pairs):
            cid = pair_cluster_id.get(pair)
            if cid is None:
                continue
            leader_score = float(cluster_leader_scores.get(cid, pair_best_score[pair]))
            gap_allowance = float(cluster_lexical_gap_allowance.get(cid, max(0.02, 2.0 * effective_score_margin)))
            score_gap = max(0.0, leader_score - pair_best_score[pair])
            if score_gap > gap_allowance:
                continue

            pair_paths = pair_buckets.get(pair, [])
            if not pair_paths:
                continue
            cue_scores = {
                label: float(score)
                for label, score in _keyword_scores(pair_paths, candidate_labels=rescue_labels).items()
            }
            treats_score = cue_scores.get("TREATS", 0.0)
            adverse_score = cue_scores.get("ADVERSE_EFFECT", 0.0)
            discontinue_score = cue_scores.get("DISCONTINUED", 0.0)
            explicit_stop_hits = _explicit_discontinue_hits(pair_paths)
            non_treat_best = max(adverse_score, discontinue_score)
            if explicit_stop_hits <= 0 and not (non_treat_best > 0.0 and non_treat_best > treats_score):
                continue

            kept_pairs.add(pair)
            dropped_pairs.discard(pair)
            lexical_cue_rescued.add(pair)
            if collect_details:
                pair_decisions.append({
                    "cluster_id": cid,
                    "diag_row_idx": pair[0],
                    "med_row_idx": pair[1],
                    "rank": None,
                    "kept": True,
                    "reason": "lexical_relation_cue_rescue",
                    "best_score": round(pair_best_score[pair], 4),
                    "leader_score": round(leader_score, 4),
                    "score_gap": round(score_gap, 4),
                    "support_count": pair_support[pair],
                    "cue_scores": {
                        label: round(float(cue_scores.get(label, 0.0)), 2)
                        for label in rescue_labels
                    },
                    "explicit_stop_hits": int(explicit_stop_hits),
                })
        if lexical_cue_rescued:
            print(
                f"  Lexical-cue rescue: recovered {len(lexical_cue_rescued)} dropped pairs "
                f"with explicit ADVERSE_EFFECT/DISCONTINUED evidence"
            )

    filtered_paths: List[Dict] = []
    filtered_labels: List[int] = []
    for path, lbl in zip(paths, labels):
        pair = (path["diag_row_idx"], path["med_row_idx"])
        if pair in kept_pairs:
            filtered_paths.append(path)
            filtered_labels.append(int(lbl))

    dropped_path_count = len(paths) - len(filtered_paths)
    print(
        f"  Cluster-tail filter: kept {len(kept_pairs)}/{len(pair_buckets)} clustered pairs "
        f"(dropped {len(dropped_pairs)} low-score tail pairs across {clusters_touched} clusters; "
        f"{dropped_path_count} paths removed)"
    )

    stats = {
        "enabled": True,
        "mode": resolved_mode,
        "dropped_pairs": len(dropped_pairs),
        "dropped_paths": dropped_path_count,
        "clusters_touched": clusters_touched,
        "keep_rank": effective_keep_rank,
        "score_margin": effective_score_margin,
        "unique_sent_rescued": len(unique_sent_rescued),
        "lexical_cue_rescued": len(lexical_cue_rescued),
    }
    if collect_details:
        stats["pair_decisions"] = pair_decisions

    return filtered_paths, np.asarray(filtered_labels, dtype=int), stats


# =============================================================================
# Phase F - Evaluation against Annotated_Test.json ground truth
# =============================================================================

def build_gt_path_set(gt_relationships: List[Dict]) -> Tuple[set, set]:
    """
    Build ground-truth sets directly from the annotated relationships:

      gt_triples : {(diag_idx, sent_j, drug_idx)} - one triple per evidence sentence
                   per relationship entry (typed or untyped, all 24 entries contribute)
      gt_pairs   : {(diag_idx, drug_idx)} - 22 unique pairs (relationship-type agnostic)

    Note: multi_relationship_flags pairs are already subsumed in gt_pairs because both
    relationship entries for the same (diag, drug) pair contribute the same pair key.
    """
    gt_triples: set = set()
    gt_pairs:   set = set()

    for rel in gt_relationships:
        i_A = rel["diag_idx"]
        i_B = rel["drug_idx"]
        gt_pairs.add((i_A, i_B))
        for j in rel["evidence_sents"]:
            gt_triples.add((i_A, j, i_B))

    print(f"\n  GT: {len(gt_relationships)} relationships -> "
          f"{len(gt_pairs)} unique pairs, {len(gt_triples)} (diag, sent, drug) triples")
    return gt_triples, gt_pairs


def _score_prediction(pred: set, gt: set) -> Tuple[int, float, float, float]:
    tp = len(pred & gt)
    p  = tp / max(len(pred), 1)
    r  = tp / max(len(gt), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return tp, p, r, f1


def _prf1(pred: set, gt: set) -> Tuple[float, float, float]:
    _, p, r, f1 = _score_prediction(pred, gt)
    return round(p, 4), round(r, 4), round(f1, 4)


def _pair_sentence_index(items: set) -> Dict[Tuple[Any, ...], Set[int]]:
    """Index predicted/GT sentence ids by pair key.

    Supports both untyped triples ``(diag, sent, med)`` and typed triples
    ``(diag, sent, med, rel_type)``.
    """
    sentence_index: Dict[Tuple[Any, ...], Set[int]] = defaultdict(set)
    for item in items:
        if len(item) == 4:
            diag_idx, sent_idx, med_idx, rel_type = item
            pair_key: Tuple[Any, ...] = (diag_idx, med_idx, rel_type)
        elif len(item) == 3:
            diag_idx, sent_idx, med_idx = item
            pair_key = (diag_idx, med_idx)
        else:
            continue
        sentence_index[pair_key].add(int(sent_idx))
    return sentence_index


def _score_evidence_pairs(
    pred_pairs: set,
    pred_triples: set,
    gt_triples: set,
) -> Tuple[int, float, float, float, int, int]:
    """Score evidence support once per pair, not once per sentence.

    This intentionally differs from strict triple overlap. A predicted pair counts
    as a TP iff at least one predicted sentence matches one GT evidence sentence
    for the same pair key. Otherwise that entire predicted pair counts once as an
    FP, even if it generated many predicted sentences.
    """
    pred_pair_keys = set(pred_pairs)
    pred_sentence_index = _pair_sentence_index(pred_triples)
    gt_sentence_index = _pair_sentence_index(gt_triples)
    gt_pair_keys = set(gt_sentence_index.keys())

    tp = 0
    for pair_key in pred_pair_keys & gt_pair_keys:
        if pred_sentence_index.get(pair_key, set()) & gt_sentence_index.get(pair_key, set()):
            tp += 1

    p = tp / max(len(pred_pair_keys), 1)
    r = tp / max(len(gt_pair_keys), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return tp, p, r, f1, len(pred_pair_keys), len(gt_pair_keys)


def _evidence_pair_prf1(
    pred_pairs: set,
    pred_triples: set,
    gt_triples: set,
) -> Tuple[float, float, float, int, int]:
    _, p, r, f1, n_pred, n_gt = _score_evidence_pairs(pred_pairs, pred_triples, gt_triples)
    return round(p, 4), round(r, 4), round(f1, 4), n_pred, n_gt


def _best_rel_type_evidence_match(
    pred_pairs: set,
    pred_triples: set,
    gt_triples_by_type: Dict[str, set],
) -> Tuple[str, Dict[str, float]]:
    """Choose the best type under the any-evidence-match scoring rule."""
    best_type = REL_TYPES[0]
    best_score: Optional[Tuple[float, int, float, float]] = None
    best_metrics: Dict[str, float] = {
        "tp": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "n_pred": len(pred_pairs),
        "n_gt": 0,
    }

    for rel_type in REL_TYPES:
        gt_items = gt_triples_by_type.get(rel_type, set())
        tp, p, r, f1, n_pred, n_gt = _score_evidence_pairs(pred_pairs, pred_triples, gt_items)
        score = (f1, tp, p, r)
        if best_score is None or score > best_score:
            best_score = score
            best_type = rel_type
            best_metrics = {
                "tp": tp,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "n_pred": n_pred,
                "n_gt": n_gt,
            }

    return best_type, best_metrics


def _build_typed_gt_sets(
    gt_relationships: List[Dict],
) -> Tuple[Dict[str, set], Dict[str, set], Dict[str, int], set, set]:
    gt_pairs_by_type: Dict[str, set] = defaultdict(set)
    gt_triples_by_type: Dict[str, set] = defaultdict(set)
    gt_pair_counts: Dict[str, int] = defaultdict(int)
    gt_typed_pairs: set = set()
    gt_typed_triples: set = set()

    for rel in gt_relationships:
        rel_type = rel["rel_type"]
        diag_idx = rel["diag_idx"]
        drug_idx = rel["drug_idx"]
        pair = (diag_idx, drug_idx)

        gt_pairs_by_type[rel_type].add(pair)
        gt_pair_counts[rel_type] += 1
        gt_typed_pairs.add((diag_idx, drug_idx, rel_type))

        for sent_idx in rel["evidence_sents"]:
            triple = (diag_idx, sent_idx, drug_idx)
            gt_triples_by_type[rel_type].add(triple)
            gt_typed_triples.add((diag_idx, sent_idx, drug_idx, rel_type))

    return (
        gt_pairs_by_type,
        gt_triples_by_type,
        gt_pair_counts,
        gt_typed_pairs,
        gt_typed_triples,
    )


def _typed_prediction_sets(paths: List[Dict]) -> Tuple[set, set]:
    pred_typed_pairs: set = set()
    pred_typed_triples: set = set()

    for p in paths:
        rel_type = p.get("relationship")
        if rel_type not in REL_TYPES:
            continue

        diag_idx = p["diag_row_idx"]
        sent_idx = p["sent_idx"]
        med_idx = p["med_row_idx"]
        pred_typed_pairs.add((diag_idx, med_idx, rel_type))
        pred_typed_triples.add((diag_idx, sent_idx, med_idx, rel_type))

    return pred_typed_pairs, pred_typed_triples


def _build_multi_valid(
    gt_relationships: List[Dict],
    multi_pairs: set,
) -> Dict[Tuple[int, int], Set[str]]:
    """
    For each (diag_idx, drug_idx) pair listed in multi_relationship_flags, return the
    set of ALL valid relationship types for that pair (as loaded into gt_relationships).
    Pairs where only one unique type is present are excluded (no adjustment needed).
    """
    multi_valid: Dict[Tuple[int, int], Set[str]] = {}
    for pair in multi_pairs:
        d, m = pair
        types = {
            rel["rel_type"]
            for rel in gt_relationships
            if rel["diag_idx"] == d and rel["drug_idx"] == m
        }
        if len(types) > 1:
            multi_valid[pair] = types
    return multi_valid


def _adjust_typed_sets_multilabel(
    pred_typed_pairs: set,
    gt_typed_pairs: set,
    pred_typed_triples: set,
    gt_typed_triples: set,
    multi_valid: Dict[Tuple[int, int], Set[str]],
) -> Tuple[set, set, set, set]:
    """
    Any-match relaxation for multi-label pairs (those in multi_relationship_flags).

    For each (diag_idx, drug_idx) pair with multiple valid relationship types:
      - The GT is collapsed to one canonical entry (alphabetically-first valid type
        that is present in gt_typed_pairs), so the pair counts as ONE unit in the
        recall denominator.
      - Any prediction whose type is among the valid types is remapped to the same
        canonical entry -> TP.  Predicting a single valid type is sufficient; no
        penalty is applied for the other valid types of that pair.
      - Predictions with an invalid type remain FP as normal.
      - If no valid prediction is made for the pair, one FN entry is retained.

    Applies identically to typed-triple sets, remapping the type dimension while
    preserving the sentence index.
    """
    if not multi_valid:
        return pred_typed_pairs, gt_typed_pairs, pred_typed_triples, gt_typed_triples

    adj_pred_tp = set(pred_typed_pairs)
    adj_gt_tp   = set(gt_typed_pairs)
    adj_pred_tt = set(pred_typed_triples)
    adj_gt_tt   = set(gt_typed_triples)

    for (d, m), valid_types in multi_valid.items():
        # ---- typed pairs ----
        pair_gt = {(d2, m2, t) for (d2, m2, t) in adj_gt_tp if d2 == d and m2 == m}
        gt_types_present = {t for _, _, t in pair_gt}
        # Canonical = alphabetically-first type that is both valid and present in GT
        overlap = valid_types & gt_types_present
        canonical_type = min(overlap) if overlap else min(valid_types)
        canonical_p = (d, m, canonical_type)

        adj_gt_tp -= pair_gt
        adj_gt_tp.add(canonical_p)

        valid_pred_p = {
            (d2, m2, t) for (d2, m2, t) in adj_pred_tp
            if d2 == d and m2 == m and t in valid_types
        }
        if valid_pred_p:
            adj_pred_tp -= valid_pred_p
            adj_pred_tp.add(canonical_p)

        # ---- typed triples ----
        pair_gt_t = {(d2, s, m2, t) for (d2, s, m2, t) in adj_gt_tt if d2 == d and m2 == m}
        adj_gt_tt -= pair_gt_t
        for (d2, s, m2, _) in pair_gt_t:
            adj_gt_tt.add((d2, s, m2, canonical_type))

        valid_pred_t = {
            (d2, s, m2, t) for (d2, s, m2, t) in adj_pred_tt
            if d2 == d and m2 == m and t in valid_types
        }
        if valid_pred_t:
            adj_pred_tt -= valid_pred_t
            for (d2, s, m2, _) in valid_pred_t:
                adj_pred_tt.add((d2, s, m2, canonical_type))

    return adj_pred_tp, adj_gt_tp, adj_pred_tt, adj_gt_tt


def _select_supported_pair_label(
    label_scores: Dict[str, float],
    label_counts: Dict[str, int],
) -> str:
    if not label_scores:
        return ""
    return min(
        label_scores,
        key=lambda label: (
            -float(label_scores.get(label, 0.0)),
            -int(label_counts.get(label, 0)),
            _rel_type_sort_key(label),
        ),
    )


def _build_batch_pair_label_records(
    admission_id: str,
    reporting_paths: List[Dict],
    gt_relationships: List[Dict],
    multi_pairs: set,
) -> List[Dict[str, object]]:
    gt_pair_types = _build_gt_pair_type_lookup(gt_relationships)
    multi_valid = _build_multi_valid(gt_relationships, multi_pairs)
    pair_label_scores: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pair_label_counts: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path in reporting_paths:
        pair_key = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        if pair_key not in gt_pair_types:
            continue
        predicted_label = _normalize_rel_type(str(path.get("relationship", "") or ""))
        if not predicted_label:
            continue
        try:
            path_score = float(path.get("path_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            path_score = 0.0
        pair_label_scores[pair_key][predicted_label] += path_score
        pair_label_counts[pair_key][predicted_label] += 1

    records: List[Dict[str, object]] = []
    for pair_key in sorted(pair_label_scores):
        valid_gt_types = list(gt_pair_types.get(pair_key, ()))
        if not valid_gt_types:
            continue
        canonical_gt_label = valid_gt_types[0]
        predicted_label_raw = _select_supported_pair_label(
            pair_label_scores[pair_key],
            pair_label_counts[pair_key],
        )
        if not predicted_label_raw:
            continue
        is_multilabel_gt = len(valid_gt_types) > 1
        predicted_label = (
            canonical_gt_label
            if is_multilabel_gt and predicted_label_raw in multi_valid.get(pair_key, set())
            else predicted_label_raw
        )
        records.append({
            "admission_id": admission_id,
            "diag_row_idx": pair_key[0],
            "med_row_idx": pair_key[1],
            "predicted_label": predicted_label,
            "predicted_label_raw": predicted_label_raw,
            "gt_label": canonical_gt_label,
            "gt_valid_labels": valid_gt_types,
            "is_multilabel_gt": is_multilabel_gt,
        })
    return records


def _best_rel_type_match(pred_items: set, gt_by_type: Dict[str, set]) -> Tuple[str, Dict[str, float]]:
    best_type = REL_TYPES[0]
    best_score: Optional[Tuple[float, int, float, float]] = None
    best_metrics: Dict[str, float] = {
        "tp": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "n_pred": len(pred_items),
        "n_gt": 0,
    }

    for rel_type in REL_TYPES:
        gt_items = gt_by_type.get(rel_type, set())
        tp, p, r, f1 = _score_prediction(pred_items, gt_items)
        score = (f1, tp, p, r)
        if best_score is None or score > best_score:
            best_score = score
            best_type = rel_type
            best_metrics = {
                "tp": tp,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "n_pred": len(pred_items),
                "n_gt": len(gt_items),
            }

    return best_type, best_metrics


def _oracle_cluster_type_metrics(
    paths: List[Dict],
    gt_pairs_by_type: Dict[str, set],
    gt_triples_by_type: Dict[str, set],
    gt_typed_pairs: set,
    gt_typed_triples: set,
    adj_gt_pairs_by_type: Optional[Dict[str, set]] = None,
    adj_gt_triples_by_type: Optional[Dict[str, set]] = None,
) -> Dict:
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    for path in paths:
        cluster_id = path.get("cluster_id")
        if cluster_id is None:
            continue
        clusters[int(cluster_id)].append(path)

    oracle_pair_preds: set = set()
    oracle_triple_preds: set = set()
    oracle_triple_pred_triples: set = set()
    oracle_pair_by_type: Dict[str, set] = defaultdict(set)
    oracle_triple_by_type: Dict[str, set] = defaultdict(set)
    oracle_triple_triples_by_type: Dict[str, set] = defaultdict(set)
    pair_assignments: List[Dict] = []
    triple_assignments: List[Dict] = []

    for cluster_id in sorted(clusters):
        cluster_paths = clusters[cluster_id]
        cluster_pairs = {
            (p["diag_row_idx"], p["med_row_idx"])
            for p in cluster_paths
        }
        cluster_triples = {
            (p["diag_row_idx"], p["sent_idx"], p["med_row_idx"])
            for p in cluster_paths
        }

        pair_type, pair_metrics = _best_rel_type_match(cluster_pairs, gt_pairs_by_type)
        triple_type, triple_metrics = _best_rel_type_evidence_match(cluster_pairs, cluster_triples, gt_triples_by_type)

        oracle_pair_by_type[pair_type].update(cluster_pairs)
        oracle_triple_by_type[triple_type].update(cluster_pairs)
        oracle_triple_triples_by_type[triple_type].update(cluster_triples)
        oracle_pair_preds.update({(*pair, pair_type) for pair in cluster_pairs})
        oracle_triple_preds.update({(*pair, triple_type) for pair in cluster_pairs})
        oracle_triple_pred_triples.update({(*triple, triple_type) for triple in cluster_triples})

        pair_assignments.append({
            "cluster_id": cluster_id,
            "assigned_type": pair_type,
            "n_paths": len(cluster_paths),
            "n_pairs": len(cluster_pairs),
            **pair_metrics,
        })
        triple_assignments.append({
            "cluster_id": cluster_id,
            "assigned_type": triple_type,
            "n_paths": len(cluster_paths),
            "n_pairs": len(cluster_pairs),
            "n_triples": len(cluster_triples),
            **triple_metrics,
        })

    # Use adjusted per-type GT (multi-label aware) when provided, else fall back to original.
    _eval_gt_pairs_by_type   = adj_gt_pairs_by_type   if adj_gt_pairs_by_type   is not None else gt_pairs_by_type
    _eval_gt_triples_by_type = adj_gt_triples_by_type if adj_gt_triples_by_type is not None else gt_triples_by_type

    oracle_pair_type_metrics: Dict[str, Dict] = {}
    oracle_triple_type_metrics: Dict[str, Dict] = {}
    for rel_type in REL_TYPES:
        pred_pairs = oracle_pair_by_type.get(rel_type, set())
        gt_pairs = _eval_gt_pairs_by_type.get(rel_type, set())
        pair_p, pair_r, pair_f1 = _prf1(pred_pairs, gt_pairs)
        oracle_pair_type_metrics[rel_type] = {
            "precision": pair_p,
            "recall": pair_r,
            "f1": pair_f1,
            "n_pred": len(pred_pairs),
            "n_gt": len(gt_pairs),
        }

        pred_triples = oracle_triple_triples_by_type.get(rel_type, set())
        pred_triple_pairs = oracle_triple_by_type.get(rel_type, set())
        gt_triples = _eval_gt_triples_by_type.get(rel_type, set())
        triple_p, triple_r, triple_f1, triple_n_pred, triple_n_gt = _evidence_pair_prf1(
            pred_triple_pairs,
            pred_triples,
            gt_triples,
        )
        oracle_triple_type_metrics[rel_type] = {
            "precision": triple_p,
            "recall": triple_r,
            "f1": triple_f1,
            "n_pred": triple_n_pred,
            "n_gt": triple_n_gt,
        }

    pair_p, pair_r, pair_f1 = _prf1(oracle_pair_preds, gt_typed_pairs)
    triple_p, triple_r, triple_f1, triple_n_pred, triple_n_gt = _evidence_pair_prf1(
        oracle_triple_preds,
        oracle_triple_pred_triples,
        gt_typed_triples,
    )

    return {
        "typed_pair": {
            "precision": pair_p,
            "recall": pair_r,
            "f1": pair_f1,
            "n_pred": len(oracle_pair_preds),
            "n_gt": len(gt_typed_pairs),
        },
        "typed_triple": {
            "precision": triple_p,
            "recall": triple_r,
            "f1": triple_f1,
            "n_pred": triple_n_pred,
            "n_gt": triple_n_gt,
        },
        "per_type_pair": oracle_pair_type_metrics,
        "per_type_triple": oracle_triple_type_metrics,
        "pair_assignments": pair_assignments,
        "triple_assignments": triple_assignments,
    }


def _cluster_label_vs_oracle_metrics(
    paths: List[Dict],
    oracle_pair_assignments: List[Dict],
) -> Dict:
    """Compare each final cluster label against its oracle cluster type."""
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    for path in paths:
        cluster_id = path.get("cluster_id")
        if cluster_id is None:
            continue
        clusters[int(cluster_id)].append(path)

    oracle_by_cluster: Dict[int, Dict[str, object]] = {}
    for assignment in oracle_pair_assignments or []:
        try:
            cluster_id = int(assignment.get("cluster_id"))
        except Exception:
            continue
        oracle_type = _normalize_rel_type(str(assignment.get("assigned_type", "")))
        oracle_tp = int(assignment.get("tp", 0) or 0)
        if not oracle_type or oracle_tp <= 0:
            continue
        oracle_by_cluster[cluster_id] = dict(assignment)

    pred_typed_clusters: set = set()
    gt_typed_clusters: set = set()
    pred_by_type: Dict[str, set] = defaultdict(set)
    gt_by_type: Dict[str, set] = defaultdict(set)
    cluster_assignments: List[Dict[str, object]] = []

    for cluster_id in sorted(clusters):
        oracle_assignment = oracle_by_cluster.get(cluster_id)
        if not oracle_assignment:
            continue

        oracle_type = _normalize_rel_type(str(oracle_assignment.get("assigned_type", "")))
        if oracle_type not in REL_TYPES:
            continue

        cluster_paths = clusters[cluster_id]
        label_votes: Dict[str, float] = defaultdict(float)
        for path in cluster_paths:
            predicted_type = _normalize_rel_type(str(path.get("relationship", "")))
            if not predicted_type:
                continue
            label_votes[predicted_type] += float(path.get("path_score", 1.0))

        predicted_type = ""
        if label_votes:
            predicted_type = max(
                label_votes.items(),
                key=lambda item: (item[1], -_rel_type_sort_key(item[0])[0], item[0]),
            )[0]

        gt_typed_clusters.add((cluster_id, oracle_type))
        gt_by_type[oracle_type].add(cluster_id)
        if predicted_type in REL_TYPES:
            pred_typed_clusters.add((cluster_id, predicted_type))
            pred_by_type[predicted_type].add(cluster_id)

        cluster_pairs = {
            (p["diag_row_idx"], p["med_row_idx"])
            for p in cluster_paths
        }
        cluster_assignments.append({
            "cluster_id": cluster_id,
            "predicted_type": predicted_type,
            "oracle_type": oracle_type,
            "correct": predicted_type == oracle_type,
            "n_paths": len(cluster_paths),
            "n_pairs": len(cluster_pairs),
            "oracle_tp": oracle_assignment.get("tp"),
            "oracle_precision": oracle_assignment.get("precision"),
            "oracle_recall": oracle_assignment.get("recall"),
            "oracle_f1": oracle_assignment.get("f1"),
        })

    overall_p, overall_r, overall_f1 = _prf1(pred_typed_clusters, gt_typed_clusters)
    per_type_metrics: Dict[str, Dict] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    macro_types: List[str] = []
    for rel_type in REL_TYPES:
        pred_clusters = pred_by_type.get(rel_type, set())
        gt_clusters = gt_by_type.get(rel_type, set())
        type_p, type_r, type_f1 = _prf1(pred_clusters, gt_clusters)
        per_type_metrics[rel_type] = {
            "precision": type_p,
            "recall": type_r,
            "f1": type_f1,
            "n_pred": len(pred_clusters),
            "n_gt": len(gt_clusters),
        }
        if pred_clusters or gt_clusters:
            macro_types.append(rel_type)
            macro_precision += type_p
            macro_recall += type_r
            macro_f1 += type_f1

    n_types = max(len(macro_types), 1)
    n_correct = sum(1 for assignment in cluster_assignments if bool(assignment.get("correct")))
    accuracy = (
        n_correct / len(cluster_assignments)
        if cluster_assignments else 0.0
    )
    return {
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "accuracy": round(accuracy, 4),
        "n_pred": len(pred_typed_clusters),
        "n_gt": len(gt_typed_clusters),
        "n_evaluated": len(cluster_assignments),
        "n_correct": n_correct,
        "macro_precision": round(macro_precision / n_types, 4),
        "macro_recall": round(macro_recall / n_types, 4),
        "macro_f1": round(macro_f1 / n_types, 4),
        "per_type": per_type_metrics,
        "assignments": cluster_assignments,
    }


def _compute_per_type_typed_metrics(
    pred_typed_pairs: set,
    pred_typed_triples: set,
    gt_pairs_by_type: Dict[str, set],
    gt_triples_by_type: Dict[str, set],
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    per_type_pair: Dict[str, Dict] = {}
    per_type_triple: Dict[str, Dict] = {}

    for rel_type in REL_TYPES:
        pred_pairs = {
            (diag_idx, med_idx)
            for diag_idx, med_idx, typed_rel in pred_typed_pairs
            if typed_rel == rel_type
        }
        gt_pairs = gt_pairs_by_type.get(rel_type, set())
        pair_p, pair_r, pair_f1 = _prf1(pred_pairs, gt_pairs)
        per_type_pair[rel_type] = {
            "precision": pair_p,
            "recall": pair_r,
            "f1": pair_f1,
            "n_pred": len(pred_pairs),
            "n_gt": len(gt_pairs),
        }

        pred_triples = {
            (diag_idx, sent_idx, med_idx)
            for diag_idx, sent_idx, med_idx, typed_rel in pred_typed_triples
            if typed_rel == rel_type
        }
        gt_triples = gt_triples_by_type.get(rel_type, set())
        triple_p, triple_r, triple_f1, triple_n_pred, triple_n_gt = _evidence_pair_prf1(
            pred_pairs,
            pred_triples,
            gt_triples,
        )
        per_type_triple[rel_type] = {
            "precision": triple_p,
            "recall": triple_r,
            "f1": triple_f1,
            "n_pred": triple_n_pred,
            "n_gt": triple_n_gt,
        }

    return per_type_pair, per_type_triple


def _compute_per_type_untyped_pair_metrics(
    pred_pairs: set,
    gt_pairs_by_type: Dict[str, set],
) -> Dict[str, Dict]:
    """
    Per-class *untyped* pair metrics: for each relationship type, evaluate whether
    the predicted (diag, med) pairs cover the GT pairs of that type - without
    requiring LOKI to have assigned the correct type label.  This is the per-class
    decomposition of the overall untyped oracle pair F1, and represents the
    retrieval-level ceiling available to the labeling step.

    Precision is computed as:
        |pred ∩ GT_k| / |pred ∩ GT_any|
    i.e., "of the predicted pairs that are any GT pair, what fraction are type k?"
    This conditional denominator ensures that untyped F1 >= oracle typed F1 for every
    class, since it removes unrelated (FP vs all classes) noise from the denominator.
    Recall is standard: |pred ∩ GT_k| / |GT_k|.
    """
    all_gt_pairs: set = set()
    for pairs in gt_pairs_by_type.values():
        all_gt_pairs.update(pairs)
    n_relevant = len(pred_pairs & all_gt_pairs)

    per_type: Dict[str, Dict] = {}
    for rel_type in REL_TYPES:
        gt_pairs = gt_pairs_by_type.get(rel_type, set())
        tp = len(pred_pairs & gt_pairs)
        precision = tp / n_relevant if n_relevant > 0 else 0.0
        recall = tp / len(gt_pairs) if gt_pairs else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_type[rel_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_pred": len(pred_pairs),
            "n_gt": len(gt_pairs),
        }
    return per_type


def _print_classwise_typed_metric_table(
    classwise_metrics: Dict[str, Dict[str, Dict]],
    title: str = "  Classwise typed metrics (automatic and oracle):",
) -> None:
    print(title)
    print(
        "    "
        f"{'Type':<16} "
        f"{'UntypedPair P/R/F1':<22} "
        f"{'AutoPair P/R/F1':<20} "
        f"{'AutoTriple P/R/F1':<20} "
        f"{'OraclePair P/R/F1':<20} "
        f"{'OracleTriple P/R/F1':<20}"
    )
    for rel_type in REL_TYPES:
        untyped_pair = classwise_metrics.get("untyped_pair", {}).get(rel_type, {})
        auto_pair = classwise_metrics.get("auto_pair", {}).get(rel_type, {})
        auto_triple = classwise_metrics.get("auto_triple", {}).get(rel_type, {})
        oracle_pair = classwise_metrics.get("oracle_pair", {}).get(rel_type, {})
        oracle_triple = classwise_metrics.get("oracle_triple", {}).get(rel_type, {})

        def _fmt(metric: Dict[str, object]) -> str:
            return (
                f"{float(metric.get('precision', 0.0)):.3f}/"
                f"{float(metric.get('recall', 0.0)):.3f}/"
                f"{float(metric.get('f1', 0.0)):.3f}"
            )

        print(
            "    "
            f"{rel_type:<16} "
            f"{_fmt(untyped_pair):<22} "
            f"{_fmt(auto_pair):<20} "
            f"{_fmt(auto_triple):<20} "
            f"{_fmt(oracle_pair):<20} "
            f"{_fmt(oracle_triple):<20}"
        )


def _cluster_quality(
    paths: List[Dict],
    gt_relationships: List[Dict],
) -> Dict:
    """
    Compute cluster purity and ARI for paths whose (diag, drug) pair has a GT entry.
    Only paths with a valid REL_TYPES label are included.
    """
    from collections import Counter as _Counter
    gt_pair_types: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for rel in gt_relationships:
        gt_pair_types[(rel["diag_idx"], rel["drug_idx"])].append(rel["rel_type"])

    pred_labels: List[str] = []
    true_labels: List[str] = []
    for p in paths:
        key = (p["diag_row_idx"], p["med_row_idx"])
        pred_type = p.get("relationship", "")
        if pred_type not in REL_TYPES or key not in gt_pair_types:
            continue
        gt_types = gt_pair_types[key]
        true_type = pred_type if pred_type in gt_types else gt_types[0]
        pred_labels.append(pred_type)
        true_labels.append(true_type)

    if not pred_labels:
        return {"purity": None, "ari": None, "n_evaluated": 0}

    type_buckets: Dict[str, List[str]] = defaultdict(list)
    for p, t in zip(pred_labels, true_labels):
        type_buckets[p].append(t)
    purity = (
        sum(_Counter(v).most_common(1)[0][1] for v in type_buckets.values())
        / len(pred_labels)
    )

    try:
        from sklearn.metrics import adjusted_rand_score  # type: ignore
        all_types = sorted(set(pred_labels + true_labels))
        t2i = {t: i for i, t in enumerate(all_types)}
        ari = adjusted_rand_score(
            [t2i[t] for t in true_labels],
            [t2i[p] for p in pred_labels],
        )
    except Exception:
        ari = None

    return {
        "purity": round(purity, 4),
        "ari":    round(ari, 4) if ari is not None else None,
        "n_evaluated": len(pred_labels),
    }


def _raw_pair_cluster_quality(
    paths: List[Dict],
    gt_relationships: List[Dict],
    cluster_key: str = "cluster_id",
) -> Dict:
    gt_pair_types: Dict[Tuple[int, int], set] = defaultdict(set)
    gt_by_type: Dict[str, set] = defaultdict(set)
    for rel in gt_relationships:
        pair = (rel["diag_idx"], rel["drug_idx"])
        gt_pair_types[pair].add(rel["rel_type"])
        gt_by_type[rel["rel_type"]].add(pair)

    cluster_votes: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for path in paths:
        pair = (path["diag_row_idx"], path["med_row_idx"])
        cluster_votes[pair][int(path.get(cluster_key, path.get("cluster_id", -1)))] += 1

    pair_to_cluster: Dict[Tuple[int, int], int] = {}
    split_pairs = 0
    for pair, votes in cluster_votes.items():
        if len(votes) > 1:
            split_pairs += 1
        pair_to_cluster[pair] = max(votes.items(), key=lambda item: (item[1], -item[0]))[0]

    gt_matched_pairs = {
        pair: cluster_id
        for pair, cluster_id in pair_to_cluster.items()
        if pair in gt_pair_types
    }
    if not gt_matched_pairs:
        return {
            "purity": None,
            "n_clusters": 0,
            "n_pred_pairs": len(pair_to_cluster),
            "n_gt_matched_pairs": 0,
            "split_pairs": split_pairs,
            "oracle_pair": {"precision": None, "recall": None, "f1": None, "n_pred": 0, "n_gt": 0},
            "per_type_oracle": {},
        }

    cluster_members: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for pair, cluster_id in gt_matched_pairs.items():
        cluster_members[cluster_id].append(pair)

    dominant_sum = 0
    for members in cluster_members.values():
        type_support = {
            rel_type: sum(1 for pair in members if rel_type in gt_pair_types[pair])
            for rel_type in REL_TYPES
        }
        dominant_sum += max(type_support.values()) if type_support else 0
    purity = dominant_sum / max(len(gt_matched_pairs), 1)

    pred_by_type: Dict[str, set] = defaultdict(set)
    for cluster_id, members in cluster_members.items():
        best_type = REL_TYPES[0]
        best_score: Optional[Tuple[float, int, float, float]] = None
        member_set = set(members)
        for rel_type in REL_TYPES:
            gt_items = gt_by_type.get(rel_type, set())
            tp, p, r, f1 = _score_prediction(member_set, gt_items)
            score = (f1, tp, p, r)
            if best_score is None or score > best_score:
                best_score = score
                best_type = rel_type
        pred_by_type[best_type].update(member_set)

    pred_typed_pairs = {
        (*pair, rel_type)
        for rel_type, members in pred_by_type.items()
        for pair in members
    }
    gt_typed_pairs = {
        (*pair, rel_type)
        for pair, rel_types in gt_pair_types.items()
        for rel_type in rel_types
    }
    oracle_p, oracle_r, oracle_f1 = _prf1(pred_typed_pairs, gt_typed_pairs)

    per_type_oracle: Dict[str, Dict] = {}
    for rel_type in REL_TYPES:
        pred_pairs = pred_by_type.get(rel_type, set())
        gt_pairs = gt_by_type.get(rel_type, set())
        pair_p, pair_r, pair_f1 = _prf1(pred_pairs, gt_pairs)
        per_type_oracle[rel_type] = {
            "precision": pair_p,
            "recall": pair_r,
            "f1": pair_f1,
            "n_pred": len(pred_pairs),
            "n_gt": len(gt_pairs),
        }

    return {
        "purity": round(purity, 4),
        "n_clusters": len(set(gt_matched_pairs.values())),
        "n_pred_pairs": len(pair_to_cluster),
        "n_gt_matched_pairs": len(gt_matched_pairs),
        "split_pairs": split_pairs,
        "oracle_pair": {
            "precision": oracle_p,
            "recall": oracle_r,
            "f1": oracle_f1,
            "n_pred": len(pred_typed_pairs),
            "n_gt": len(gt_typed_pairs),
        },
        "per_type_oracle": per_type_oracle,
    }


def _raw_cluster_pair_mix(
    paths: List[Dict],
    gt_relationships: List[Dict],
    cluster_key: str = "cluster_id",
) -> List[Dict]:
    gt_pair_types: Dict[Tuple[int, int], set] = defaultdict(set)
    for rel in gt_relationships:
        gt_pair_types[(rel["diag_idx"], rel["drug_idx"])].add(rel["rel_type"])

    cluster_pairs: Dict[int, set] = defaultdict(set)
    for path in paths:
        cluster_id = int(path.get(cluster_key, path.get("cluster_id", -1)))
        cluster_pairs[cluster_id].add((path["diag_row_idx"], path["med_row_idx"]))

    summaries: List[Dict] = []
    for cluster_id, members in cluster_pairs.items():
        type_counts: Dict[str, int] = defaultdict(int)
        gt_pair_count = 0
        for pair in members:
            rel_types = gt_pair_types.get(pair, set())
            if rel_types:
                gt_pair_count += 1
            for rel_type in rel_types:
                type_counts[rel_type] += 1
        summaries.append({
            "cluster_id": cluster_id,
            "n_pairs": len(members),
            "n_gt_pairs": gt_pair_count,
            "gt_type_counts": {
                rel_type: type_counts.get(rel_type, 0)
                for rel_type in REL_TYPES
                if type_counts.get(rel_type, 0) > 0
            },
        })

    summaries.sort(key=lambda item: (-item["n_gt_pairs"], -item["n_pairs"], item["cluster_id"]))
    return summaries


def evaluate(
    paths:            List[Dict],
    gt_relationships: List[Dict],
    gt_diag:          Dict[int, List[int]],
    gt_med:           Dict[int, List[int]],
    gt_triples:       set,
    gt_pairs:         set,
    multi_pairs:      set,
    raw_cluster_paths: Optional[List[Dict]] = None,
    show_typed_metrics: bool = False,
) -> Dict:
    pred_triples = {(p["diag_row_idx"], p["sent_idx"], p["med_row_idx"]) for p in paths}
    pred_pairs   = {(p["diag_row_idx"], p["med_row_idx"]) for p in paths}
    pred_diag    = {p["diag_row_idx"] for p in paths}
    pred_med     = {p["med_row_idx"]  for p in paths}

    p_t, r_t, f1_t = _prf1(pred_triples, gt_triples)
    p_p, r_p, f1_p = _prf1(pred_pairs,   gt_pairs)

    diag_covered = len(pred_diag & set(gt_diag.keys()))
    med_covered  = len(pred_med  & set(gt_med.keys()))
    diag_recall  = diag_covered / max(len(gt_diag), 1)
    med_recall   = med_covered  / max(len(gt_med), 1)

    pred_tp_pairs = pred_pairs & gt_pairs
    multi_tp      = len(pred_tp_pairs & multi_pairs)

    (
        gt_by_type_set,
        gt_triples_by_type,
        gt_by_type_cnt,
        gt_typed_pairs,
        gt_typed_triples,
    ) = _build_typed_gt_sets(gt_relationships)

    pred_typed_pairs, pred_typed_triples = _typed_prediction_sets(paths)

    # Multi-label any-match: for pairs flagged in multi_relationship_flags, predicting
    # ANY one valid type counts as a full correct prediction.  Collapse each such pair
    # to a single canonical GT entry and remap valid-type predictions to that entry.
    multi_valid = _build_multi_valid(gt_relationships, multi_pairs)
    adj_pred_tp, adj_gt_tp, adj_pred_tt, adj_gt_tt = _adjust_typed_sets_multilabel(
        pred_typed_pairs, gt_typed_pairs, pred_typed_triples, gt_typed_triples, multi_valid
    )
    # Rebuild per-type GT dicts from the adjusted typed-pair / typed-triple sets.
    adj_gt_pairs_by_type: Dict[str, set] = defaultdict(set)
    for (_d, _m, _t) in adj_gt_tp:
        adj_gt_pairs_by_type[_t].add((_d, _m))
    adj_gt_triples_by_type: Dict[str, set] = defaultdict(set)
    for (_d, _s, _m, _t) in adj_gt_tt:
        adj_gt_triples_by_type[_t].add((_d, _s, _m))

    typed_pair_p, typed_pair_r, typed_pair_f1 = _prf1(adj_pred_tp, adj_gt_tp)
    typed_triple_p, typed_triple_r, typed_triple_f1, typed_triple_n_pred, typed_triple_n_gt = _evidence_pair_prf1(
        adj_pred_tp,
        adj_pred_tt,
        adj_gt_tt,
    )

    # Per-type typed P/R/F1 (automatic predictions at pair and triple levels).
    auto_pair_type_metrics, auto_triple_type_metrics = _compute_per_type_typed_metrics(
        adj_pred_tp,
        adj_pred_tt,
        adj_gt_pairs_by_type,
        adj_gt_triples_by_type,
    )
    # Keep backward-compatible per_type payload name for pair-level automatic metrics.
    type_metrics: Dict[str, Dict] = auto_pair_type_metrics

    raw_cluster_source = raw_cluster_paths if raw_cluster_paths is not None else paths
    raw_pair_clusters = _raw_pair_cluster_quality(
        raw_cluster_source,
        gt_relationships,
        cluster_key="raw_cluster_id" if raw_cluster_paths is not None else "cluster_id",
    )
    cluster_mix = _raw_cluster_pair_mix(
        raw_cluster_source,
        gt_relationships,
        cluster_key="raw_cluster_id" if raw_cluster_paths is not None else "cluster_id",
    )
    cq = _cluster_quality(paths, gt_relationships)
    oracle_metrics = _oracle_cluster_type_metrics(
        paths,
        gt_by_type_set,
        gt_triples_by_type,
        adj_gt_tp,
        adj_gt_tt,
        adj_gt_pairs_by_type,
        adj_gt_triples_by_type,
    )
    cluster_label_metrics = _cluster_label_vs_oracle_metrics(
        paths,
        oracle_metrics.get("pair_assignments", []),
    )
    classwise_typed_metrics = {
        "untyped_pair": _compute_per_type_untyped_pair_metrics(pred_pairs, gt_by_type_set),
        "auto_pair": auto_pair_type_metrics,
        "auto_triple": auto_triple_type_metrics,
        "oracle_pair": oracle_metrics["per_type_pair"],
        "oracle_triple": oracle_metrics["per_type_triple"],
    }

    sep = "-" * 66
    print(f"\n{sep}")
    print("  Evaluation Results")
    print(sep)
    print(f"  Exact-triple    P={p_t:.3f}  R={r_t:.3f}  F1={f1_t:.3f}  "
          f"(pred={len(pred_triples)}, gt={len(gt_triples)})")
    print(f"  Relaxed-pair    P={p_p:.3f}  R={r_p:.3f}  F1={f1_p:.3f}  "
          f"(pred={len(pred_pairs)}, gt={len(gt_pairs)})")
    print(f"  Multi-rel recall: {multi_tp}/{len(multi_pairs)} multi-rel pairs recovered")
    print(f"  Diag  row recall: {diag_recall:.3f}  "
          f"({diag_covered}/{len(gt_diag)} annotated rows reached)")
    print(f"  Med   row recall: {med_recall:.3f}  "
          f"({med_covered}/{len(gt_med)} annotated rows reached)")
    if raw_pair_clusters["purity"] is not None:
        oracle_pair = raw_pair_clusters["oracle_pair"]
        print("  Raw pair-cluster quality:")
        print(f"    Pair purity      {raw_pair_clusters['purity']:.3f}  "
              f"(clusters={raw_pair_clusters['n_clusters']}, gt-matched pairs={raw_pair_clusters['n_gt_matched_pairs']}, split_pairs={raw_pair_clusters['split_pairs']})")
        print(f"    Oracle pair F1   P={oracle_pair['precision']:.3f}  R={oracle_pair['recall']:.3f}  "
              f"F1={oracle_pair['f1']:.3f}  (pred={oracle_pair['n_pred']}, gt={oracle_pair['n_gt']})")
        print("    Per-type oracle pair breakdown:")
        for rel_type in REL_TYPES:
            metric = raw_pair_clusters["per_type_oracle"][rel_type]
            print(f"      {rel_type:<16}  P={metric['precision']:.3f}  R={metric['recall']:.3f}  "
                  f"F1={metric['f1']:.3f}  (pred={metric['n_pred']}, gt={metric['n_gt']})")
        if cluster_label_metrics["n_evaluated"] > 0:
            print("  Cluster label vs oracle cluster type:")
            print(
                f"    Cluster label P={cluster_label_metrics['precision']:.3f}  "
                f"R={cluster_label_metrics['recall']:.3f}  "
                f"F1={cluster_label_metrics['f1']:.3f}  "
                f"Acc={cluster_label_metrics['accuracy']:.3f}  "
                f"(pred={cluster_label_metrics['n_pred']}, gt={cluster_label_metrics['n_gt']})"
            )
        if cluster_mix:
            print("    Cluster GT mix:")
            for summary in cluster_mix:
                mix = summary["gt_type_counts"]
                mix_str = ", ".join(f"{rel}:{count}" for rel, count in mix.items()) if mix else "no GT-matched pairs"
                print(f"      C{summary['cluster_id']:02d}  pairs={summary['n_pairs']:2d}  gt_pairs={summary['n_gt_pairs']:2d}  {mix_str}")
    if show_typed_metrics and cq["n_evaluated"] > 0:
        ari_str = f"{cq['ari']:.3f}" if cq["ari"] is not None else "n/a"
        print(f"  Cluster purity:   {cq['purity']:.3f}  ARI: {ari_str}  "
              f"(over {cq['n_evaluated']} GT-matched paths)")
        print("  Typed-label overall (secondary to clustering):")
        print(f"    Automatic pair    P={typed_pair_p:.3f}  R={typed_pair_r:.3f}  "
            f"F1={typed_pair_f1:.3f}  (pred={len(adj_pred_tp)}, gt={len(adj_gt_tp)})")
        print(f"    Automatic evidence P={typed_triple_p:.3f}  R={typed_triple_r:.3f}  "
            f"F1={typed_triple_f1:.3f}  (pred={typed_triple_n_pred}, gt={typed_triple_n_gt})")
        print(f"    Oracle pair       P={oracle_metrics['typed_pair']['precision']:.3f}  "
            f"R={oracle_metrics['typed_pair']['recall']:.3f}  "
            f"F1={oracle_metrics['typed_pair']['f1']:.3f}  "
            f"(pred={oracle_metrics['typed_pair']['n_pred']}, gt={oracle_metrics['typed_pair']['n_gt']})")
        print(f"    Oracle evidence   P={oracle_metrics['typed_triple']['precision']:.3f}  "
            f"R={oracle_metrics['typed_triple']['recall']:.3f}  "
            f"F1={oracle_metrics['typed_triple']['f1']:.3f}  "
            f"(pred={oracle_metrics['typed_triple']['n_pred']}, gt={oracle_metrics['typed_triple']['n_gt']})")
        print(f"  Per-type pair breakdown (automatic vs oracle):")
        for t in REL_TYPES:
            auto = type_metrics[t]
            oracle = oracle_metrics["per_type_pair"][t]
            print(f"    {t:<16}  auto F1={auto['f1']:.3f}  "
                  f"oracle F1={oracle['f1']:.3f}  "
                  f"(auto pred={auto['n_pred']}, oracle pred={oracle['n_pred']}, gt={auto['n_gt']})")
        if cluster_label_metrics["n_evaluated"] > 0:
            print("  Per-type cluster-label breakdown (predicted cluster label vs oracle cluster type):")
            for rel_type in REL_TYPES:
                metric = cluster_label_metrics["per_type"][rel_type]
                print(
                    f"    {rel_type:<16}  P={metric['precision']:.3f}  R={metric['recall']:.3f}  "
                    f"F1={metric['f1']:.3f}  (pred={metric['n_pred']}, gt={metric['n_gt']})"
                )
    _print_classwise_typed_metric_table(classwise_typed_metrics)
    print(sep)

    return {
        "exact_triple":    {"precision": p_t, "recall": r_t, "f1": f1_t,
                            "n_pred": len(pred_triples), "n_gt": len(gt_triples)},
        "relaxed_pair":    {"precision": p_p, "recall": r_p, "f1": f1_p,
                            "n_pred": len(pred_pairs), "n_gt": len(gt_pairs)},
          "typed_pair":      {"precision": typed_pair_p, "recall": typed_pair_r,
                        "f1": typed_pair_f1, "n_pred": len(adj_pred_tp),
                        "n_gt": len(adj_gt_tp)},
          "typed_triple":    {"precision": typed_triple_p, "recall": typed_triple_r,
                        "f1": typed_triple_f1, "n_pred": typed_triple_n_pred,
                        "n_gt": typed_triple_n_gt},
        "raw_pair_clusters": raw_pair_clusters,
                "raw_cluster_mix": cluster_mix,
          "oracle_cluster_remap": oracle_metrics,
                    "cluster_label": cluster_label_metrics,
        "per_type":        type_metrics,
                "per_type_typed_pair": auto_pair_type_metrics,
                "per_type_typed_triple": auto_triple_type_metrics,
                "classwise_typed_metrics": classwise_typed_metrics,
        "cluster_quality": cq,
        "multi_rel_pair_recall": {"recovered": multi_tp, "total": len(multi_pairs)},
        "diag_row_recall": round(diag_recall, 4),
        "med_row_recall":  round(med_recall, 4),
    }


# =============================================================================
# Output
# =============================================================================

def diagnose_gt_coverage(
    pair_scores: torch.Tensor,
    n_diag: int,
    gt_relationships: List[Dict],
) -> None:
    """
    For every GT (diag, med) pair, find the best achievable path score
    across ALL 189 sentences regardless of threshold.  Shows exactly where
    recall is being lost and at what threshold each GT pair becomes recoverable.
    """
    P = _to_numpy_array(pair_scores, dtype=np.float32)          # (n_diag+n_med, n_sents)
    n_sents = P.shape[1]

    # Deduplicate GT pairs, keeping all rel_types per pair
    gt_by_pair: Dict[tuple, List[str]] = {}
    for rel in gt_relationships:
        key = (rel["diag_idx"], rel["drug_idx"])
        gt_by_pair.setdefault(key, []).append(rel["rel_type"])

    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    recovered_at: Dict[float, int] = {t: 0 for t in thresholds}

    rows = []
    for (d_idx, m_idx), rel_types in sorted(gt_by_pair.items()):
        d_scores = P[d_idx]                  # (n_sents,) - diag row scores
        m_scores = P[n_diag + m_idx]         # (n_sents,) - med row scores
        path_scores = (d_scores + m_scores) / 2.0
        best_j   = int(path_scores.argmax())
        best_score = float(path_scores[best_j])
        best_diag  = float(d_scores[best_j])
        best_med   = float(m_scores[best_j])
        rows.append((d_idx + 1, m_idx + 1, best_score, best_diag, best_med,
                     "/".join(rel_types)))
        for t in thresholds:
            if best_diag >= t and best_med >= t:
                recovered_at[t] += 1

    print("\n-- GT Coverage Diagnostic ----------------------------------------")
    print(f"  {'Diag':>5}  {'Med':>5}  {'BestPath':>9}  {'DiagScore':>9}  {'MedScore':>9}  GT Type")
    print("  " + "-" * 64)
    for d, m, bp, bd, bm, rt in sorted(rows, key=lambda x: -x[2]):
        flag = "[OK]" if bd >= 0.344 and bm >= 0.344 else ("~" if bp >= 0.25 else "✗")
        print(f"  {d:>5}  {m:>5}  {bp:>9.4f}  {bd:>9.4f}  {bm:>9.4f}  {rt:<20} {flag}")

    print()
    print("  Recoverable GT pairs at each threshold (no top_k limit):")
    for t in thresholds:
        print(f"    gamma={t:.2f}  ->  {recovered_at[t]:2d} / {len(gt_by_pair)} GT pairs  "
              f"(recall {recovered_at[t]/len(gt_by_pair):.1%})")
    print()


def _write_diagnostics_artifacts(
    diagnostics_output_dir: str,
    metrics: Dict,
    pair_recovery_diagnostics: Optional[Dict[str, object]] = None,
    cluster_name_map: Optional[Dict[int, str]] = None,
    cluster_label_details: Optional[Dict[int, Dict[str, object]]] = None,
    cluster_label_backend: Optional[str] = None,
    cluster_label_input_mode: Optional[str] = None,
) -> None:
    resolved_dir = _resolve_optional_input_path(diagnostics_output_dir)
    if resolved_dir is None:
        return

    resolved_dir.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_dir / f"diagnostics_summary_{ADMISSION_ID}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "admission_id": ADMISSION_ID,
            "patient_id": TARGET_PATIENT,
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved diagnostics summary: {summary_path}")

    if pair_recovery_diagnostics:
        pair_recovery_path = resolved_dir / f"pair_recovery_diagnostics_{ADMISSION_ID}.json"
        with open(pair_recovery_path, "w", encoding="utf-8") as f:
            json.dump(pair_recovery_diagnostics, f, indent=2, ensure_ascii=False)
        print(f"  Saved pair-recovery diagnostics: {pair_recovery_path}")

    if cluster_name_map is not None or cluster_label_details is not None:
        cluster_labeling_path = resolved_dir / f"cluster_labeling_{ADMISSION_ID}.json"
        with open(cluster_labeling_path, "w", encoding="utf-8") as f:
            json.dump({
                "backend": cluster_label_backend,
                "label_input_mode": cluster_label_input_mode,
                "cluster_name_map": cluster_name_map or {},
                "cluster_label_details": cluster_label_details or {},
            }, f, indent=2, ensure_ascii=False)
        print(f"  Saved cluster-label diagnostics: {cluster_labeling_path}")


def save_outputs(
    paths: List[Dict],
    metrics: Dict,
    gt_relationships: List[Dict] = None,
    cluster_name_map: Optional[Dict[int, str]] = None,
    cluster_label_details: Optional[Dict[int, Dict[str, object]]] = None,
    cluster_label_backend: Optional[str] = None,
    cluster_label_input_mode: Optional[str] = None,
    pair_recovery_diagnostics: Optional[Dict[str, object]] = None,
    diagnostics_output_dir: str = "",
) -> None:
    output = {
        "admission_id": ADMISSION_ID,
        "patient_id":   TARGET_PATIENT,
        "metrics":      metrics,
        "paths":        paths,
    }
    if cluster_name_map is not None or cluster_label_details is not None:
        output["cluster_labeling"] = {
            "backend": cluster_label_backend,
            "label_input_mode": cluster_label_input_mode,
            "cluster_name_map": cluster_name_map or {},
            "cluster_label_details": cluster_label_details or {},
        }
    if pair_recovery_diagnostics:
        output["pair_recovery_diagnostics"] = pair_recovery_diagnostics
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved JSON : {OUT_JSON}")
    if diagnostics_output_dir:
        _write_diagnostics_artifacts(
            diagnostics_output_dir,
            metrics,
            pair_recovery_diagnostics=pair_recovery_diagnostics,
            cluster_name_map=cluster_name_map,
            cluster_label_details=cluster_label_details,
            cluster_label_backend=cluster_label_backend,
            cluster_label_input_mode=cluster_label_input_mode,
        )

    if paths:
        # Build GT lookup: (diag_idx, drug_idx) -> list of rel_types
        gt_lookup: Dict[tuple, List[str]] = {}
        if gt_relationships:
            for rel in gt_relationships:
                key = (rel["diag_idx"], rel["drug_idx"])
                gt_lookup.setdefault(key, []).append(rel["rel_type"])

        fieldnames = ["Med Row", "Diag Row", "Cluster",
                      "Predicted Relationship", "GT Relationship"]
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            seen: set = set()
            for p in sorted(paths, key=lambda x: -x["path_score"]):
                key = (p["diag_row_idx"], p["med_row_idx"])
                if key in seen:
                    continue
                seen.add(key)
                gt_rels = gt_lookup.get(key, [])
                writer.writerow({
                    "Med Row":                p["med_row_idx"] + 1,
                    "Diag Row":               p["diag_row_idx"] + 1,
                    "Cluster":                p.get("cluster_id", ""),
                    "Predicted Relationship": p.get("relationship", ""),
                    "GT Relationship":        " / ".join(gt_rels) if gt_rels else "",
                })
        print(f"  Saved CSV  : {OUT_CSV}")


def save_cluster_audit(
    raw_cluster_paths: List[Dict],
    gt_relationships: List[Dict],
    cluster_name_map: Dict[int, str],
    kept_cluster_ids: set[int],
    cluster_label_details: Optional[Dict[int, Dict[str, object]]] = None,
) -> None:
    if not raw_cluster_paths:
        return

    gt_lookup: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for rel in gt_relationships:
        gt_lookup[(rel["diag_idx"], rel["drug_idx"])] .append(rel["rel_type"])

    cluster_paths: Dict[int, List[Dict]] = defaultdict(list)
    for path in raw_cluster_paths:
        cluster_paths[int(path.get("raw_cluster_id", path.get("cluster_id", -1)))].append(path)

    lines: List[str] = [
        f"# Cluster Audit: admission {ADMISSION_ID}",
        "",
        "Small inspection report over raw clusters before low-signal filtering.",
        "",
    ]

    for cid in sorted(cluster_paths):
        cpaths = cluster_paths[cid]
        pair_buckets = _bucket_paths_by_pair(cpaths)
        signal_scores = _keyword_scores(cpaths)
        explicit_stop_hits = _explicit_discontinue_hits(cpaths)
        kept = cid in kept_cluster_ids
        label = cluster_name_map.get(cid, f"cluster_{cid}")
        label_detail = (cluster_label_details or {}).get(cid, {})
        label_backend = str(label_detail.get("backend", ""))
        label_source = str(label_detail.get("label_source", ""))
        score_type = str(label_detail.get("score_type", ""))
        label_scores = label_detail.get("label_scores")
        label_counts = label_detail.get("label_counts")
        n_occurrences = int(label_detail.get("n_occurrences", len(cpaths)))
        n_unique_sentences = int(label_detail.get("n_unique_sentences", len({int(path['sent_idx']) for path in cpaths})))
        fallback_reason = str(label_detail.get("fallback_reason", "") or "")
        supporting_evidence = label_detail.get("supporting_evidence") or []
        refinement_children = label_detail.get("pair_label_refinement_children") or []
        kept_refinement_children = []
        if isinstance(refinement_children, list):
            kept_refinement_children = [
                child for child in refinement_children
                if int(child.get("cluster_id", -1)) in kept_cluster_ids
            ]

        if refinement_children:
            if kept_refinement_children and len(kept_refinement_children) == len(refinement_children):
                status_text = "retained via pair-label split"
            elif kept_refinement_children:
                status_text = "partially retained via pair-label split"
            else:
                status_text = "dropped after pair-label split"
        else:
            status_text = "retained" if kept else "dropped low-signal"

        lines.append(f"## Cluster {cid}")
        lines.append("")
        lines.append(f"- Status: {status_text}")
        lines.append(f"- Final label: {label}")
        if label_backend:
            lines.append(f"- Label backend: {label_backend}")
        if label_source:
            lines.append(f"- Label source: {label_source}")
        lines.append(f"- Paths: {len(cpaths)}")
        lines.append(f"- Sentence occurrences scored: {n_occurrences}")
        lines.append(f"- Unique sentences: {n_unique_sentences}")
        lines.append(f"- Unique pairs: {len(pair_buckets)}")
        keyword_text = ", ".join(
            f"{rel_type}={int(signal_scores.get(rel_type, 0))}"
            for rel_type in _preferred_rel_type_order(list(signal_scores))
        )
        lines.append(f"- Keyword scores: {keyword_text}")
        lines.append(f"- Explicit discontinue hits: {explicit_stop_hits}")
        if isinstance(label_scores, dict) and label_scores:
            ordered_score_labels = _preferred_rel_type_order(list(label_scores))
            if score_type == "keyword_counts":
                score_text = ", ".join(
                    f"{rel_type}={int(round(float(label_scores.get(rel_type, 0.0))))}"
                    for rel_type in ordered_score_labels
                )
                lines.append(f"- Keyword label scores: {score_text}")
            else:
                score_title = "Vote totals" if score_type == "weighted_votes" else "Label scores"
                score_text = ", ".join(
                    f"{rel_type}={float(label_scores.get(rel_type, 0.0)):.4f}"
                    for rel_type in ordered_score_labels
                )
                lines.append(f"- {score_title}: {score_text}")
        if isinstance(label_counts, dict) and label_counts:
            count_text = ", ".join(
                f"{rel_type}={int(round(float(label_counts.get(rel_type, 0.0))))}"
                for rel_type in _preferred_rel_type_order(list(label_counts))
            )
            lines.append(f"- Label counts: {count_text}")
        if fallback_reason:
            lines.append(f"- Fallback reason: {fallback_reason}")
        if isinstance(refinement_children, list) and refinement_children:
            refinement_text = ", ".join(
                f"C{int(child.get('cluster_id', -1)):02d}:{child.get('label', 'UNKNOWN')} ({int(child.get('n_pairs', 0))} pairs)"
                for child in refinement_children
            )
            lines.append(f"- Pair-label refinement: {refinement_text}")
        if supporting_evidence:
            lines.append("- Top labeling evidence:")
            for record in supporting_evidence[:3]:
                sentence = " ".join(str(record.get("sentence", "")).split())[:220]
                sent_idx = record.get("sent_idx", "?")
                section_name = str(record.get("section_name", "")).strip()
                metadata: List[str] = []
                if record.get("label"):
                    metadata.append(f"label={record['label']}")
                confidence = _to_float_or_none(record.get("confidence"))
                if confidence is not None:
                    metadata.append(f"conf={confidence:.3f}")
                vote_weight = _to_float_or_none(record.get("vote_weight"))
                if vote_weight is not None:
                    metadata.append(f"vote={vote_weight:.4f}")
                path_score = _to_float_or_none(record.get("path_score"))
                if path_score is not None:
                    metadata.append(f"path={path_score:.4f}")
                meta_text = f" | {' | '.join(metadata)}" if metadata else ""
                lines.append(
                    f"  - sent[{sent_idx}] {section_name}: {sentence}{meta_text}"
                )
        lines.append("")
        lines.append("Top pairs:")

        ranked_pairs = sorted(
            pair_buckets.items(),
            key=lambda item: max(float(path.get("path_score", 0.0)) for path in item[1]),
            reverse=True,
        )

        for (diag_idx, med_idx), pair_paths in ranked_pairs[:5]:
            best_score = max(float(path.get("path_score", 0.0)) for path in pair_paths)
            gt_types = sorted(set(gt_lookup.get((diag_idx, med_idx), [])))
            gt_text = ", ".join(gt_types) if gt_types else "none"
            lines.append(
                f"- Pair diag[{diag_idx + 1}] x med[{med_idx + 1}] | best_score={best_score:.4f} | GT={gt_text}"
            )

            seen_sentences: set[str] = set()
            trigger_lines = sorted(
                pair_paths,
                key=lambda path: (
                    float(path.get("path_score", 0.0)),
                    float(path.get("score_diag", 0.0)),
                    float(path.get("score_med", 0.0)),
                ),
                reverse=True,
            )
            emitted = 0
            for path in trigger_lines:
                sentence = " ".join(str(path.get("sent_text", "")).split())
                if sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                lines.append(
                    f"  - sent[{path['sent_idx']}] {path.get('section_name', '')}: {sentence[:220]}"
                )
                emitted += 1
                if emitted >= 2:
                    break
        lines.append("")

    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_AUDIT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved audit: {OUT_AUDIT}")


def print_table_preview(paths: List[Dict], n: int = 12) -> None:
    if not paths:
        print("  (no paths discovered)")
        return
    print(f"\n{'DiagRow':>7}  {'MedRow':>6}  {'Score':>6}  {'Relationship':<28}  Mediating Sentence")
    print("-" * 110)
    for p in paths[:n]:
        rel = p.get("relationship", "")[:27]
        sent = p["sent_text"][:55]
        print(
            f"  [{p['diag_row_idx']+1:2d}]     [{p['med_row_idx']+1:2d}]  "
            f"{p['path_score']:>6.4f}  {rel:<28}  {sent}..."
        )


# Color palette for relationship types in plots
_REL_COLORS = {
    "TREATS":          "#5b9bd5",  # paper blue
    "ADVERSE_EFFECT":  "#e15759",  # paper red
    "NEGATIVE":        "#f28e2b",  # paper orange
    "DISCONTINUED":    "#b39ddb",  # paper purple
    "CONTRAINDICATED": "#76b7b2",  # teal fallback
    "OTHER":           "#64748b",  # slate fallback
}
_REL_MARKERS = {
    "TREATS": "o",
    "ADVERSE_EFFECT": "^",
    "NEGATIVE": "P",
    "DISCONTINUED": "s",
    "OTHER": "D",
}
_DYNAMIC_REL_COLORS = [
    "#5b9bd5",
    "#e15759",
    "#f28e2b",
    "#b39ddb",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#4e79a7",
]
_DYNAMIC_REL_MARKERS = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "*"]


def _rel_visual_index(rel_type: str) -> int:
    normalized = _normalize_rel_type(rel_type)
    return sum(ord(ch) for ch in normalized)


def _rel_color(rel_type: str) -> str:
    normalized = _normalize_rel_type(rel_type)
    if normalized in _REL_COLORS:
        return _REL_COLORS[normalized]
    return _DYNAMIC_REL_COLORS[_rel_visual_index(normalized) % len(_DYNAMIC_REL_COLORS)]


def _hex_to_rgb(color: str) -> Tuple[float, float, float]:
    resolved = str(color).strip().lstrip("#")
    if len(resolved) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {color}")
    return tuple(int(resolved[index:index + 2], 16) / 255.0 for index in (0, 2, 4))


def _blend_hex_color(
    color: str,
    blend_with: str = "#ffffff",
    blend_ratio: float = 0.82,
) -> Tuple[float, float, float]:
    base = _hex_to_rgb(color)
    overlay = _hex_to_rgb(blend_with)
    ratio = min(max(float(blend_ratio), 0.0), 1.0)
    return tuple(
        ((1.0 - ratio) * base[channel]) + (ratio * overlay[channel])
        for channel in range(3)
    )


def _rel_marker(rel_type: str) -> str:
    normalized = _normalize_rel_type(rel_type)
    if normalized in _REL_MARKERS:
        return _REL_MARKERS[normalized]
    return _DYNAMIC_REL_MARKERS[_rel_visual_index(normalized) % len(_DYNAMIC_REL_MARKERS)]


def _rel_offset(rel_type: str, x_span: float, y_span: float) -> Tuple[float, float]:
    normalized = _normalize_rel_type(rel_type)
    known_offsets = {
        "TREATS": (-0.020 * x_span, 0.018 * y_span),
        "ADVERSE_EFFECT": (0.022 * x_span, 0.016 * y_span),
        "NEGATIVE": (-0.024 * x_span, -0.014 * y_span),
        "DISCONTINUED": (0.000 * x_span, -0.024 * y_span),
        "OTHER": (0.018 * x_span, -0.018 * y_span),
    }
    if normalized in known_offsets:
        return known_offsets[normalized]

    ordered = _preferred_rel_type_order()
    rel_index = ordered.index(normalized) if normalized in ordered else 0
    angle = (2.0 * np.pi * rel_index) / max(len(ordered), 1)
    return (0.024 * x_span * float(np.cos(angle)), 0.024 * y_span * float(np.sin(angle)))


def _project_embeddings_for_display(embeddings: np.ndarray) -> Tuple[np.ndarray, str]:
    if embeddings.size == 0:
        return np.empty((0, 2), dtype=np.float32), "none"

    embeddings = _l2_normalize_rows(np.asarray(embeddings, dtype=np.float32))
    n_points = embeddings.shape[0]
    if n_points == 1:
        return np.zeros((1, 2), dtype=np.float32), "identity"

    from sklearn.decomposition import PCA

    if n_points <= 4:
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        return coords.astype(np.float32), "PCA"

    try:
        from umap import UMAP  # type: ignore

        n_neighbors = min(18, max(6, n_points // 18), n_points - 1)
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.28,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
        return np.asarray(coords, dtype=np.float32), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE

        pca_dims = min(24, embeddings.shape[1], n_points - 1)
        tsne_input = embeddings
        if pca_dims >= 2:
            tsne_input = PCA(n_components=pca_dims, random_state=42).fit_transform(embeddings)

        perplexity = min(30, max(8, n_points // 18), n_points - 1)
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="pca",
            max_iter=1500,
        ).fit_transform(tsne_input)
        return np.asarray(coords, dtype=np.float32), "t-SNE"


def _project_cluster_map_embeddings(embeddings: np.ndarray) -> Tuple[np.ndarray, str]:
    if embeddings.size == 0:
        return np.empty((0, 2), dtype=np.float32), "none"

    embeddings = _l2_normalize_rows(np.asarray(embeddings, dtype=np.float32))
    n_points = embeddings.shape[0]
    if n_points == 1:
        return np.zeros((1, 2), dtype=np.float32), "identity"

    from sklearn.decomposition import PCA

    if n_points <= 4:
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        return coords.astype(np.float32), "PCA"

    try:
        from umap import UMAP  # type: ignore

        reducer = UMAP(
            n_components=2,
            n_neighbors=min(12, max(4, n_points // 8), n_points - 1),
            min_dist=0.05,
            spread=1.25,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
        return np.asarray(coords, dtype=np.float32), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE

        pca_dims = min(16, embeddings.shape[1], n_points - 1)
        tsne_input = embeddings
        if pca_dims >= 2:
            tsne_input = PCA(n_components=pca_dims, random_state=42).fit_transform(embeddings)

        perplexity = min(12, max(4, n_points // 6), n_points - 1)
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="pca",
            max_iter=1500,
        ).fit_transform(tsne_input)
        return np.asarray(coords, dtype=np.float32), "t-SNE"


def _truncate_label(text: str, max_chars: int = 26) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _format_index_group(indices: List[int], one_based: bool = False, max_items: int = 4) -> str:
    values = sorted({int(idx) + (1 if one_based else 0) for idx in indices})
    if not values:
        return "-"
    if len(values) <= max_items:
        return ",".join(str(value) for value in values)
    visible = ",".join(str(value) for value in values[:max_items])
    return f"{visible},+{len(values) - max_items}"


def _format_cluster_index_label(
    diag_ids: List[int],
    sent_ids: List[int],
    med_ids: List[int],
) -> str:
    diag_text = _format_index_group(diag_ids, one_based=True, max_items=3)
    sent_text = _format_index_group(sent_ids, one_based=False, max_items=4)
    med_text = _format_index_group(med_ids, one_based=True, max_items=3)
    return f"(diag[{diag_text}], sent[{sent_text}], med[{med_text}])"


def _format_single_triple_label(path: Dict) -> str:
    return (
        f"(diag[{int(path['diag_row_idx']) + 1}], "
        f"sent[{int(path['sent_idx'])}], "
        f"med[{int(path['med_row_idx']) + 1}])"
    )


def _build_gt_pair_type_lookup(gt_relationships: List[Dict]) -> Dict[Tuple[int, int], Tuple[str, ...]]:
    pair_types: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for rel in gt_relationships:
        pair_types[(int(rel["diag_idx"]), int(rel["drug_idx"]))].add(str(rel["rel_type"]))
    return {
        pair: tuple(sorted(types))
        for pair, types in pair_types.items()
    }


def _resolve_cluster_semantic_type(
    cluster_paths: List[Dict],
    gt_pair_types: Dict[Tuple[int, int], Tuple[str, ...]],
    negative_pairs: Optional[set[Tuple[int, int]]] = None,
) -> str:
    support_counts: Dict[str, int] = defaultdict(int)
    cluster_pairs = {
        (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        for path in cluster_paths
    }

    for pair in cluster_pairs:
        for rel_type in gt_pair_types.get(pair, ()):
            support_counts[_normalize_rel_type(rel_type)] += 1
        if negative_pairs and pair in negative_pairs:
            support_counts["NEGATIVE"] += 1

    if support_counts:
        negative_count = int(support_counts.get("NEGATIVE", 0))
        positive_support = sum(
            count
            for rel_type, count in support_counts.items()
            if rel_type != "NEGATIVE"
        )
        if negative_count > 0 and positive_support == 0:
            return "NEGATIVE"

        discontinue_count = int(support_counts.get("DISCONTINUED", 0))
        treats_count = int(support_counts.get("TREATS", 0))
        adverse_count = int(support_counts.get("ADVERSE_EFFECT", 0))
        if discontinue_count > 0 and _explicit_discontinue_hits(cluster_paths) > 0:
            if discontinue_count >= adverse_count and discontinue_count >= max(treats_count - 1, 1):
                return "DISCONTINUED"

        return min(
            support_counts,
            key=lambda rel_type: (-support_counts[rel_type], _rel_type_sort_key(rel_type)),
        )

    predicted_counts: Dict[str, int] = defaultdict(int)
    for path in cluster_paths:
        rel_type = _normalize_rel_type(path.get("relationship", ""))
        if rel_type:
            predicted_counts[rel_type] += 1
    if predicted_counts:
        return min(
            predicted_counts,
            key=lambda rel_type: (-predicted_counts[rel_type], _rel_type_sort_key(rel_type)),
        )

    return "OTHER"


def _resolve_path_semantic_type(
    path: Dict,
    cluster_semantic_type: str,
    gt_pair_types: Dict[Tuple[int, int], Tuple[str, ...]],
    negative_pairs: Optional[set[Tuple[int, int]]] = None,
) -> str:
    pair = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
    gt_types = tuple(_normalize_rel_type(rel_type) for rel_type in gt_pair_types.get(pair, ()))
    if gt_types:
        if cluster_semantic_type in gt_types:
            return cluster_semantic_type
        if "DISCONTINUED" in gt_types and _explicit_discontinue_hits([path]) > 0:
            return "DISCONTINUED"
        if "DISCONTINUED" in gt_types and "TREATS" in gt_types:
            return "DISCONTINUED"
        return gt_types[0]

    if negative_pairs and pair in negative_pairs:
        return "NEGATIVE"

    predicted_type = _normalize_rel_type(path.get("relationship", ""))
    if predicted_type and predicted_type != "OTHER":
        return predicted_type

    return cluster_semantic_type or "OTHER"


def _signature_source_text(record: Dict[str, object]) -> str:
    sentence_text = str(record.get("sentence_text", "") or "").strip()
    if sentence_text:
        return re.sub(r"\s+", " ", sentence_text).lower()

    path = record.get("path")
    if isinstance(path, dict):
        sent_text = str(path.get("sent_text", "") or "").strip()
        if sent_text:
            return re.sub(r"\s+", " ", sent_text).lower()
    return ""


def _is_signature_candidate(term: str, stop_words: set[str]) -> bool:
    tokens = [token for token in re.findall(r"[a-zA-Z]+", term.lower()) if token]
    if not tokens:
        return False
    if all(token in stop_words for token in tokens):
        return False
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return False
    return True


def _select_signature_terms(
    feature_names: np.ndarray,
    scores: np.ndarray,
    stop_words: set[str],
    max_terms: int = 4,
) -> Tuple[str, ...]:
    if feature_names.size == 0 or scores.size == 0:
        return tuple()

    ranked_indices = np.argsort(scores)[::-1]
    candidates: List[Tuple[str, Tuple[str, ...]]] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0.0:
            break
        term = str(feature_names[index]).strip()
        if not _is_signature_candidate(term, stop_words):
            continue
        tokens = tuple(token for token in re.findall(r"[a-zA-Z]+", term.lower()) if token)
        if not tokens:
            continue
        candidates.append((term, tokens))
        if len(candidates) >= max_terms * 12:
            break

    if not candidates:
        return tuple()

    selected: List[str] = []
    selected_tokens: List[set[str]] = []

    def add_candidate(term: str, tokens: Tuple[str, ...]) -> bool:
        token_set = set(tokens)
        for existing in selected_tokens:
            if token_set.issubset(existing) or existing.issubset(token_set):
                return False
        selected.append(term)
        selected_tokens.append(token_set)
        return True

    for require_multi_word in (True, False, None):
        for term, tokens in candidates:
            is_multi_word = len(tokens) > 1
            if require_multi_word is True and not is_multi_word:
                continue
            if require_multi_word is False and is_multi_word:
                continue
            if add_candidate(term, tokens) and len(selected) >= max_terms:
                return tuple(selected)

    return tuple(selected[:max_terms])


def _semantic_signature_stop_words() -> set[str]:
    preserved_tokens = {"for", "to", "by", "due", "no", "not", "with", "without"}
    signature_noise_tokens = {
        "admission", "admissions", "bedtime", "bid", "capsule", "capsules", "daily",
        "day", "days", "discussion", "discharge", "dose", "doses", "g", "home",
        "hospital", "im", "iv", "kg", "mcg", "medication", "medications", "mg",
        "ml", "nightly", "oral", "patient", "patients", "per", "po", "prn",
        "qam", "qday", "qhs", "qid", "qpm", "sc", "scheduled", "status", "subq",
        "tab", "tabs", "tablet", "tablets", "tid", "via", "week", "weekly",
    }
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return (set(ENGLISH_STOP_WORDS) - preserved_tokens) | signature_noise_tokens
    except Exception:
        return ({
            "a", "an", "and", "are", "as", "at", "be", "but", "for", "from",
            "had", "has", "have", "in", "is", "it", "of", "on", "or", "the",
            "to", "was", "were", "with",
        } - preserved_tokens) | signature_noise_tokens


def _extract_signature_terms_from_texts(
    texts: List[str],
    max_terms: int = 4,
    weights: Optional[List[float]] = None,
) -> Tuple[str, ...]:
    stop_words = _semantic_signature_stop_words()
    token_scores: Dict[str, float] = defaultdict(float)

    for index, text in enumerate(texts):
        weight = 1.0
        if weights is not None and index < len(weights):
            weight = max(float(weights[index]), 0.25)
        tokens = [
            token
            for token in re.findall(r"[a-zA-Z]+", text.lower())
            if token and token not in stop_words and len(token) >= 3
        ]
        for token in tokens:
            token_scores[token] += 1.0 * weight
        for token_index in range(len(tokens) - 1):
            bigram = f"{tokens[token_index]} {tokens[token_index + 1]}"
            if _is_signature_candidate(bigram, stop_words):
                token_scores[bigram] += 1.35 * weight

    if not token_scores:
        return tuple()

    feature_names = np.asarray(list(token_scores.keys()))
    feature_scores = np.asarray(list(token_scores.values()), dtype=np.float32)
    return _select_signature_terms(
        feature_names,
        feature_scores,
        stop_words,
        max_terms=max_terms,
    )


def _derive_semantic_signatures(
    records_by_label: Dict[str, List[Dict[str, object]]],
    max_terms: int = 4,
) -> Dict[str, Tuple[str, ...]]:
    stop_words = _semantic_signature_stop_words()

    label_texts: Dict[str, List[str]] = {}
    for label, records in records_by_label.items():
        texts = [text for text in (_signature_source_text(record) for record in records) if text]
        label_texts[label] = texts

    documents: List[str] = []
    document_labels: List[str] = []
    for label, texts in label_texts.items():
        for text in texts:
            documents.append(text)
            document_labels.append(label)

    if not documents:
        return {label: tuple() for label in records_by_label}

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
            ngram_range=(1, 2),
            stop_words=sorted(stop_words),
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(documents)
        feature_names = np.asarray(vectorizer.get_feature_names_out())

        signatures: Dict[str, Tuple[str, ...]] = {}
        for label in records_by_label:
            label_indices = [index for index, doc_label in enumerate(document_labels) if doc_label == label]
            if not label_indices:
                signatures[label] = tuple()
                continue
            label_scores = np.asarray(matrix[label_indices].sum(axis=0)).ravel()
            signatures[label] = _select_signature_terms(
                feature_names,
                label_scores,
                stop_words,
                max_terms=max_terms,
            )
        return signatures
    except Exception:
        signatures = {}
        for label, texts in label_texts.items():
            signatures[label] = _extract_signature_terms_from_texts(
                texts,
                max_terms=max_terms,
            )
        return signatures


def _semantic_signature_text(signature_terms: Tuple[str, ...]) -> str:
    terms = tuple(term for term in signature_terms if term)
    if not terms:
        terms = ("evidence", "signal")
    first_line = ", ".join(terms[:2])
    second_line = ", ".join(terms[2:])
    if second_line:
        body = f"{{{first_line},\n{second_line}}}"
    else:
        body = f"{{{first_line}}}"
    return "Semantic Signature:\n" + body


def _format_single_triple_label(
    path: Dict,
    gt_pair_types: Optional[Dict[Tuple[int, int], Tuple[str, ...]]] = None,
) -> str:
    base = (
        f"(diag[{int(path['diag_row_idx']) + 1}], "
        f"sent[{int(path['sent_idx'])}], "
        f"med[{int(path['med_row_idx']) + 1}])"
    )
    if gt_pair_types is None:
        return base

    pair = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
    rel_types = gt_pair_types.get(pair, ())
    if not rel_types:
        return f"{base} | pred-only"
    return f"{base} | GT-pair:{'/'.join(rel_types)}"


def _top_cluster_triple_labels(
    cluster_paths: List[Dict],
    max_triples: int = 3,
    gt_pair_types: Optional[Dict[Tuple[int, int], Tuple[str, ...]]] = None,
) -> List[str]:
    best_by_triple: Dict[Tuple[int, int, int], Dict] = {}
    for path in cluster_paths:
        triple = (int(path["diag_row_idx"]), int(path["sent_idx"]), int(path["med_row_idx"]))
        best = best_by_triple.get(triple)
        if best is None or float(path.get("path_score", 0.0)) > float(best.get("path_score", 0.0)):
            best_by_triple[triple] = path

    ranked = sorted(
        best_by_triple.values(),
        key=lambda item: (
            1.0 if gt_pair_types and (int(item["diag_row_idx"]), int(item["med_row_idx"])) in gt_pair_types else 0.0,
            float(item.get("path_score", 0.0)),
            float(item.get("score_diag", 0.0)),
            float(item.get("score_med", 0.0)),
        ),
        reverse=True,
    )
    if not ranked:
        return []

    lines = [
        _format_single_triple_label(path, gt_pair_types=gt_pair_types)
        for path in ranked[:max_triples]
    ]
    if len(ranked) > max_triples:
        lines.append(f"(+{len(ranked) - max_triples} more triples)")
    return lines


def visualize_embedding_space(
    diag_rows: List[str],
    med_rows: List[str],
    raw_rows: torch.Tensor,
    raw_sentences: torch.Tensor,
    refined_rows: torch.Tensor,
    refined_sentences: torch.Tensor,
    paths: List[Dict],
    gt_relationships: List[Dict],
    out_path: str,
    sentence_encoder: Optional[SentenceTransformer] = None,
    cluster_key: str = "cluster_id",
    max_clusters: Optional[int] = None,
    label_top_k: int = 12,
    include_cluster_ids: Optional[set[int]] = None,
    show_cluster_numbers: bool = True,
    triples_per_label: int = 3,
    pair_embedding_mode: str = "contextual_sentence_average",
) -> None:
    """
    Topic-oriented join map of the final materialized space.

    Each island corresponds to one retained relationship cluster. Diagnosis rows,
    medication rows, and mediating sentences are laid out inside the island so the
    figure reads as a join-topic map rather than as a generic manifold scatter.

    max_clusters limits the number of rendered clusters using a relation-diverse
    ranking: seed one GT-backed cluster per relation type when available, then
    fill the remainder by cluster size so the map stays legible even when the
    predicted cluster label differs.
    include_cluster_ids can be used to render a specific subset of cluster ids.
    label_top_k controls how many rendered clusters receive text labels.
    triples_per_label controls how many single triples are listed inside each
    cluster label before collapsing the remainder.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
        from matplotlib.patches import Ellipse
    except ImportError as exc:
        print(f"  Embedding atlas skipped (missing library): {exc}")
        return

    del raw_rows, raw_sentences

    n_diag = len(diag_rows)
    n_med = len(med_rows)
    n_rows = n_diag + n_med
    if refined_rows.shape[0] != n_rows:
        print("  Join-topic map skipped (row embedding shape mismatch).")
        return
    if not paths:
        print("  Join-topic map skipped (no retained clustered paths).")
        return

    refined_rows_np = _l2_normalize_rows(refined_rows.float().cpu().numpy())
    refined_sentences_np = _l2_normalize_rows(refined_sentences.float().cpu().numpy())

    clusters: Dict[int, List[Dict]] = defaultdict(list)
    cluster_names: Dict[int, str] = {}
    for path in paths:
        cluster_id = int(path.get(cluster_key, path.get("cluster_id", -1)))
        if cluster_id < 0:
            continue
        clusters[cluster_id].append(path)
        cluster_names.setdefault(cluster_id, str(path.get("relationship", "")).replace("_", " "))
    if not clusters:
        print("  Join-topic map skipped (no cluster ids available on retained paths).")
        return

    pair_keys, pair_embeddings = _compute_pair_embeddings(
        paths,
        refined_sentences,
        refined_rows=refined_rows,
        n_diag=n_diag,
        sentence_encoder=sentence_encoder,
        embedding_mode=pair_embedding_mode,
        verbose=False,
    )
    if len(pair_keys) == 0:
        print("  Join-topic map skipped (no pair embeddings available).")
        return
    pair_embeddings_np = _to_numpy_array(pair_embeddings, dtype=np.float32)

    cluster_votes: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for path in paths:
        cluster_votes[(path["diag_row_idx"], path["med_row_idx"])][int(path.get(cluster_key, -1))] += 1
    pair_to_cluster = {
        pair: max(votes.items(), key=lambda item: (item[1], -item[0]))[0]
        for pair, votes in cluster_votes.items()
    }

    cluster_pair_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
    for index, pair in enumerate(pair_keys):
        cluster_id = pair_to_cluster.get(pair)
        if cluster_id is not None and cluster_id >= 0:
            cluster_pair_embeddings[int(cluster_id)].append(pair_embeddings_np[index])

    gt_pair_types = _build_gt_pair_type_lookup(gt_relationships)
    cluster_gt_pair_counts: Dict[int, int] = {}
    cluster_gt_type_match_counts: Dict[int, int] = {}
    cluster_gt_type_support_counts: Dict[int, Dict[str, int]] = {}
    cluster_gt_representative_types: Dict[int, str] = {}
    cluster_relation_keys: Dict[int, str] = {}
    for cluster_id, cpaths in clusters.items():
        cluster_pairs = {
            (int(path["diag_row_idx"]), int(path["med_row_idx"]))
            for path in cpaths
        }
        cluster_gt_pair_counts[cluster_id] = sum(
            1 for pair in cluster_pairs
            if pair in gt_pair_types
        )
        relationship = str(cluster_names.get(cluster_id, "")).upper().replace(" ", "_")
        cluster_relation_keys[cluster_id] = relationship
        type_support_counts = {
            rel_type: sum(
                1 for pair in cluster_pairs
                if rel_type in gt_pair_types.get(pair, ())
            )
            for rel_type in REL_TYPES
        }
        cluster_gt_type_support_counts[cluster_id] = type_support_counts
        best_gt_type = max(
            REL_TYPES,
            key=lambda rel_type: (type_support_counts.get(rel_type, 0), -_rel_type_sort_key(rel_type)[0]),
        ) if REL_TYPES else ""
        if type_support_counts.get(best_gt_type, 0) > 0:
            cluster_gt_representative_types[cluster_id] = best_gt_type
        cluster_gt_type_match_counts[cluster_id] = (
            type_support_counts.get(relationship, 0)
            if relationship in REL_TYPES else 0
        )

    ranked_cluster_ids = sorted(
        clusters,
        key=lambda cluster_id: (
            -int(cluster_gt_type_match_counts.get(cluster_id, 0) > 0),
            -cluster_gt_type_match_counts.get(cluster_id, 0),
            -int(cluster_gt_pair_counts.get(cluster_id, 0) > 0),
            -cluster_gt_pair_counts.get(cluster_id, 0),
            -len({(path["diag_row_idx"], path["med_row_idx"]) for path in clusters[cluster_id]}),
            -max(float(path.get("path_score", 0.0)) for path in clusters[cluster_id]),
            cluster_id,
        ),
    )

    selected_cluster_ids = list(ranked_cluster_ids)
    seeded_relation_by_cluster: Dict[int, str] = {}
    if include_cluster_ids:
        include_set = {int(cid) for cid in include_cluster_ids if int(cid) in clusters}
        selected_cluster_ids = [cluster_id for cluster_id in ranked_cluster_ids if cluster_id in include_set]
    elif max_clusters is not None and max_clusters > 0:
        selected_cluster_ids = []
        seen_cluster_ids = set(selected_cluster_ids)
        for rel_type in REL_TYPES:
            if len(selected_cluster_ids) >= max_clusters:
                break
            relation_candidates = [
                cluster_id
                for cluster_id in ranked_cluster_ids
                if cluster_id not in seen_cluster_ids
                and cluster_gt_type_support_counts.get(cluster_id, {}).get(rel_type, 0) > 0
            ]
            if not relation_candidates:
                continue
            best_cluster_id = min(
                relation_candidates,
                key=lambda cluster_id: (
                    -cluster_gt_type_support_counts.get(cluster_id, {}).get(rel_type, 0),
                    -int(cluster_gt_pair_counts.get(cluster_id, 0) > 0),
                    -cluster_gt_pair_counts.get(cluster_id, 0),
                    -len({(path["diag_row_idx"], path["med_row_idx"]) for path in clusters[cluster_id]}),
                    -max(float(path.get("path_score", 0.0)) for path in clusters[cluster_id]),
                    cluster_id,
                ),
            )
            selected_cluster_ids.append(best_cluster_id)
            seeded_relation_by_cluster[best_cluster_id] = rel_type
            seen_cluster_ids.add(best_cluster_id)

        seen_relation_keys = {
            cluster_relation_keys.get(cluster_id, "")
            for cluster_id in selected_cluster_ids
            if cluster_relation_keys.get(cluster_id, "")
        }
        for cluster_id in ranked_cluster_ids:
            if len(selected_cluster_ids) >= max_clusters:
                break
            if cluster_id in seen_cluster_ids:
                continue
            relation_key = cluster_relation_keys.get(cluster_id, "")
            if relation_key and relation_key not in seen_relation_keys:
                selected_cluster_ids.append(cluster_id)
                seen_cluster_ids.add(cluster_id)
                seen_relation_keys.add(relation_key)

        for cluster_id in ranked_cluster_ids:
            if len(selected_cluster_ids) >= max_clusters:
                break
            if cluster_id in seen_cluster_ids:
                continue
            selected_cluster_ids.append(cluster_id)
            seen_cluster_ids.add(cluster_id)

    if not selected_cluster_ids:
        print("  Join-topic map skipped (cluster selection removed all clusters).")
        return

    cluster_display_relation_keys = {
        cluster_id: seeded_relation_by_cluster.get(cluster_id, cluster_relation_keys.get(cluster_id, ""))
        for cluster_id in selected_cluster_ids
    }

    selected_paths = [
        path for path in paths
        if int(path.get(cluster_key, path.get("cluster_id", -1))) in set(selected_cluster_ids)
    ]
    if not selected_paths:
        print("  Join-topic map skipped (no paths left after cluster selection).")
        return

    def _cluster_node_center(cluster_id: int) -> np.ndarray:
        node_embs = []
        for path in clusters[cluster_id]:
            node_embs.append(refined_rows_np[int(path["diag_row_idx"])])
            node_embs.append(refined_rows_np[n_diag + int(path["med_row_idx"])])
            node_embs.append(refined_sentences_np[int(path["sent_idx"])])
        return np.mean(np.stack(node_embs, axis=0), axis=0).astype(np.float32)

    center_ids: List[int] = []
    center_embeddings: List[np.ndarray] = []
    use_pair_centers = True
    pair_center_dims = set()
    for cluster_id in selected_cluster_ids:
        pair_embs = cluster_pair_embeddings.get(cluster_id)
        if not pair_embs:
            use_pair_centers = False
            break
        pair_center_dims.add(int(np.asarray(pair_embs[0]).reshape(-1).shape[0]))
        if len(pair_center_dims) > 1:
            use_pair_centers = False
            break

    if not use_pair_centers:
        print(
            "  Join-topic map: using node-space centroids because pair embeddings are incomplete "
            "or inconsistent across rendered clusters."
        )

    for cluster_id in selected_cluster_ids:
        pair_embs = cluster_pair_embeddings.get(cluster_id)
        if use_pair_centers and pair_embs:
            center_embeddings.append(np.mean(np.stack(pair_embs, axis=0), axis=0).astype(np.float32))
        else:
            center_embeddings.append(_cluster_node_center(cluster_id))
        center_ids.append(cluster_id)

    center_coords, method = _project_embeddings_for_display(np.stack(center_embeddings, axis=0))
    center_coords = center_coords.astype(np.float32)
    if center_coords.shape[0] > 1:
        center_coords = center_coords - center_coords.mean(axis=0, keepdims=True)
        x_scale = max(float(center_coords[:, 0].std()), 1e-3)
        y_scale = max(float(center_coords[:, 1].std()), 1e-3)
        center_coords[:, 0] = center_coords[:, 0] / x_scale * 4.2
        center_coords[:, 1] = center_coords[:, 1] / y_scale * 3.1
    print(
        f"  Running {method} on {len(center_ids)} retained join-topic centroids "
        f"(rendered clusters: {', '.join(f'C{cluster_id:02d}' for cluster_id in selected_cluster_ids)}) ..."
    )

    gt_sent_idx = {
        int(sent_idx)
        for rel in gt_relationships
        for sent_idx in rel.get("evidence_sents", [])
    }

    fig, ax = plt.subplots(figsize=(18.0, 11.0))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, color="#e2e8f0", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")
        spine.set_linewidth(1.0)

    palette = plt.colormaps.get_cmap("turbo")
    cluster_colors: Dict[int, object] = {}
    for index, cluster_id in enumerate(center_ids):
        display_relation_key = cluster_display_relation_keys.get(cluster_id, "")
        if display_relation_key:
            cluster_colors[cluster_id] = _rel_color(display_relation_key)
        else:
            cluster_colors[cluster_id] = palette((index + 0.5) / max(len(center_ids), 1))

    from sklearn.decomposition import PCA

    labeled_cluster_ids = set(selected_cluster_ids[: min(max(label_top_k, 0), len(selected_cluster_ids))])

    def local_layout(node_embeddings: np.ndarray, node_types: List[str]) -> np.ndarray:
        def _order_within_group(group_embeddings: np.ndarray) -> List[int]:
            if group_embeddings.shape[0] <= 1:
                return list(range(group_embeddings.shape[0]))
            axis = PCA(n_components=1, random_state=42).fit_transform(group_embeddings).reshape(-1)
            return [idx for idx, _value in sorted(enumerate(axis.tolist()), key=lambda item: item[1])]

        def _fan_slots(count: int, x_center: float, x_wave: float, y_span: float) -> List[Tuple[float, float]]:
            if count <= 0:
                return []
            if count == 1:
                return [(x_center, 0.0)]
            ys = np.linspace(-y_span, y_span, count, dtype=np.float32)
            middle = 0.5 * (count - 1)
            slots: List[Tuple[float, float]] = []
            for idx, y_value in enumerate(ys.tolist()):
                offset = float(idx - middle)
                x_value = x_center + x_wave * (1.0 - min(abs(offset) / max(middle, 1.0), 1.0))
                slots.append((x_value, float(y_value)))
            return slots

        coords = np.zeros((len(node_types), 2), dtype=np.float32)
        for node_type, x_center, x_wave, y_span in (
            ("diag", -0.64, -0.06, 0.34),
            ("sent", 0.00, 0.10, 0.54),
            ("med", 0.64, 0.06, 0.34),
        ):
            group_indices = [idx for idx, kind in enumerate(node_types) if kind == node_type]
            if not group_indices:
                continue
            group_embs = node_embeddings[group_indices]
            ordered_local = _order_within_group(group_embs)
            slots = _fan_slots(len(group_indices), x_center=x_center, x_wave=x_wave, y_span=y_span)
            for slot_idx, local_idx in enumerate(ordered_local):
                coords[group_indices[local_idx]] = np.asarray(slots[slot_idx], dtype=np.float32)
        return coords

    center_lookup = {cluster_id: center_coords[index] for index, cluster_id in enumerate(center_ids)}
    all_positions: List[np.ndarray] = []

    for cluster_id in selected_cluster_ids:
        cpaths = clusters[cluster_id]
        center = center_lookup[cluster_id]
        color = cluster_colors[cluster_id]
        fill_color = _blend_hex_color(str(color), blend_ratio=0.84) if isinstance(color, str) else color

        diag_ids = sorted({int(path["diag_row_idx"]) for path in cpaths})
        med_ids = sorted({int(path["med_row_idx"]) for path in cpaths})
        sent_ids = sorted({int(path["sent_idx"]) for path in cpaths})

        node_keys: List[Tuple[str, int]] = []
        node_types: List[str] = []
        node_embs: List[np.ndarray] = []
        for diag_idx in diag_ids:
            node_keys.append(("diag", diag_idx))
            node_types.append("diag")
            node_embs.append(refined_rows_np[diag_idx])
        for med_idx in med_ids:
            node_keys.append(("med", med_idx))
            node_types.append("med")
            node_embs.append(refined_rows_np[n_diag + med_idx])
        for sent_idx in sent_ids:
            node_keys.append(("sent", sent_idx))
            node_types.append("sent")
            node_embs.append(refined_sentences_np[sent_idx])

        local_coords = local_layout(np.stack(node_embs, axis=0), node_types)
        cluster_radius = 0.92 + 0.08 * np.sqrt(max(len(node_keys), 1))
        positions: Dict[Tuple[str, int], np.ndarray] = {}
        for node_key, local_coord in zip(node_keys, local_coords):
            absolute = center + cluster_radius * local_coord
            positions[node_key] = absolute
            all_positions.append(absolute)

        blob = Ellipse(
            xy=(float(center[0]), float(center[1])),
            width=cluster_radius * 2.55,
            height=cluster_radius * 1.85,
            facecolor=fill_color,
            edgecolor=color,
            linewidth=1.5,
            alpha=0.92,
            zorder=1,
        )
        ax.add_patch(blob)

        best_pair_paths = [
            max(pair_paths, key=lambda path: float(path.get("path_score", 0.0)))
            for _pair, pair_paths in _bucket_paths_by_pair(cpaths).items()
        ]
        best_pair_paths = sorted(best_pair_paths, key=lambda path: float(path.get("path_score", 0.0)), reverse=True)
        for path in best_pair_paths[: min(10, len(best_pair_paths))]:
            dpos = positions[("diag", int(path["diag_row_idx"]))]
            spos = positions[("sent", int(path["sent_idx"]))]
            mpos = positions[("med", int(path["med_row_idx"]))]
            ax.plot(
                [dpos[0], spos[0], mpos[0]],
                [dpos[1], spos[1], mpos[1]],
                color=color,
                linewidth=1.15,
                alpha=0.30,
                solid_capstyle="round",
                zorder=2,
            )

        diag_strength = defaultdict(int)
        med_strength = defaultdict(int)
        sent_strength = defaultdict(int)
        for path in cpaths:
            diag_strength[int(path["diag_row_idx"])] += 1
            med_strength[int(path["med_row_idx"])] += 1
            sent_strength[int(path["sent_idx"])] += 1

        diag_coords = np.asarray([positions[("diag", diag_idx)] for diag_idx in diag_ids], dtype=np.float32)
        med_coords = np.asarray([positions[("med", med_idx)] for med_idx in med_ids], dtype=np.float32)
        sent_coords = np.asarray([positions[("sent", sent_idx)] for sent_idx in sent_ids], dtype=np.float32)

        ax.scatter(
            sent_coords[:, 0], sent_coords[:, 1],
            s=[44 + 11 * sent_strength[sent_idx] for sent_idx in sent_ids],
            c=[color], marker="o", alpha=0.88,
            linewidths=[1.2 if sent_idx in gt_sent_idx else 0.0 for sent_idx in sent_ids],
            edgecolors=["#ffffff" if sent_idx in gt_sent_idx else color for sent_idx in sent_ids],
            zorder=4,
        )
        ax.scatter(
            diag_coords[:, 0], diag_coords[:, 1],
            s=[120 + 26 * diag_strength[diag_idx] for diag_idx in diag_ids],
            c=[color], marker="D", alpha=0.97,
            linewidths=1.2, edgecolors="#0f172a", zorder=5,
        )
        ax.scatter(
            med_coords[:, 0], med_coords[:, 1],
            s=[124 + 26 * med_strength[med_idx] for med_idx in med_ids],
            c=[color], marker="h", alpha=0.97,
            linewidths=1.2, edgecolors="#0f172a", zorder=5,
        )

        if cluster_id in labeled_cluster_ids:
            predicted_relation_key = cluster_relation_keys.get(cluster_id, "")
            display_relation_key = cluster_display_relation_keys.get(cluster_id, predicted_relation_key)
            cluster_name = (display_relation_key or cluster_names.get(cluster_id, "topic").upper()).replace("_", " ")
            if predicted_relation_key and display_relation_key and predicted_relation_key != display_relation_key:
                cluster_name = f"{cluster_name} [pred:{predicted_relation_key.replace('_', ' ')}]"
            triple_lines = _top_cluster_triple_labels(
                cpaths,
                max_triples=max(1, triples_per_label),
                gt_pair_types=gt_pair_types,
            )
            header = f"{cluster_name}" if not show_cluster_numbers else f"C{cluster_id:02d}  {cluster_name}"
            label_text = "\n".join([header] + triple_lines) if triple_lines else header
            ax.text(
                float(center[0]),
                float(center[1] + cluster_radius * 0.86),
                label_text,
                ha="center",
                va="bottom",
                fontsize=8.7,
                color="#0f172a",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 1.2,
                    "alpha": 0.96,
                },
                zorder=7,
            )

    if all_positions:
        points = np.stack(all_positions, axis=0)
        x_span = max(float(points[:, 0].max() - points[:, 0].min()), 1.0)
        y_span = max(float(points[:, 1].max() - points[:, 1].min()), 1.0)
        ax.set_xlim(float(points[:, 0].min()) - 0.10 * x_span, float(points[:, 0].max()) + 0.10 * x_span)
        ax.set_ylim(float(points[:, 1].min()) - 0.10 * y_span, float(points[:, 1].max()) + 0.12 * y_span)

    legend_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#64748b",
               markeredgecolor="#0f172a", markersize=9, label="Diagnosis row"),
        Line2D([0], [0], marker="h", color="none", markerfacecolor="#64748b",
               markeredgecolor="#0f172a", markersize=10, label="Medication row"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748b",
               markeredgecolor="#64748b", markersize=7, label="Mediating sentence"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748b",
               markeredgecolor="#ffffff", markersize=7, label="GT evidence sentence"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#cbd5e1",
    )
    ax.set_title(
        f"LOKI Join Topic Map - Admission {ADMISSION_ID}\n"
        f"Diagnosis rows, medication rows, and evidence sentences grouped by retained materialization cluster"
        f"  |  shown={len(selected_cluster_ids)}",
        fontsize=15,
        color="#0f172a",
        pad=16,
    )
    fig.text(
        0.5,
        0.025,
        f"Cluster islands are positioned by retained pair-level topic embeddings using {method}; internal node layouts separate diagnosis rows, medications, and mediating sentences.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(rect=(0.02, 0.05, 0.98, 0.96))
    _save_figure_outputs(fig, out_path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved join-topic map: {out_path}")


def visualize_clusters_tsne(
    paths: List[Dict],
    refined_rows: torch.Tensor,
    n_diag: int,
    refined_sentences: torch.Tensor,
    gt_relationships: List[Dict],
    out_path: str,
    sentence_encoder: Optional[SentenceTransformer] = None,
    cluster_key: str = "raw_cluster_id",
    pair_embedding_mode: str = "contextual_sentence_average",
) -> None:
    """
    Two-panel projection of the actual pair-clustering space.

    Left: predicted diagnosis-medication row pairs colored by raw cluster id.
    Right: the same coordinates with GT relationship overlays, including
    explicit DISCONTINUED markers so they remain visible even if the automatic
    cluster labels have not converged yet.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        print(f"  t-SNE visualization skipped (missing library): {exc}")
        return

    pair_keys, pair_embeddings = _compute_pair_embeddings(
        paths,
        refined_sentences,
        refined_rows=refined_rows,
        n_diag=n_diag,
        sentence_encoder=sentence_encoder,
        embedding_mode=pair_embedding_mode,
        verbose=False,
    )
    if not pair_keys:
        print("  No row-pair embeddings available - skipping pair-cluster visualization.")
        return
    pair_embeddings_np = _to_numpy_array(pair_embeddings, dtype=np.float32)

    n_pairs = len(pair_keys)
    print(f"  Running t-SNE on {n_pairs} candidate row-pair embeddings ...")
    if n_pairs == 1:
        coords = np.zeros((1, 2), dtype=np.float32)
    elif n_pairs <= 4:
        coords = PCA(n_components=2, random_state=42).fit_transform(pair_embeddings_np)
    else:
        perplexity = min(18, max(4, n_pairs // 5), n_pairs - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=42, init="pca", max_iter=1500)
        coords = tsne.fit_transform(pair_embeddings_np)

    cluster_votes: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    cluster_names: Dict[int, str] = {}
    for path in paths:
        pair = (path["diag_row_idx"], path["med_row_idx"])
        cluster_id = int(path.get(cluster_key, path.get("cluster_id", -1)))
        cluster_votes[pair][cluster_id] += 1
        if cluster_id not in cluster_names and path.get("relationship"):
            cluster_names[cluster_id] = str(path.get("relationship", ""))

    pair_to_cluster = {
        pair: max(votes.items(), key=lambda item: (item[1], -item[0]))[0]
        for pair, votes in cluster_votes.items()
    }

    gt_pair_types: Dict[Tuple[int, int], set] = defaultdict(set)
    for rel in gt_relationships:
        gt_pair_types[(rel["diag_idx"], rel["drug_idx"])].add(rel["rel_type"])

    unique_clusters = sorted({pair_to_cluster[pair] for pair in pair_keys})
    cmap = plt.colormaps.get_cmap("tab20")
    cluster_palette = {
        cid: cmap(idx / max(len(unique_clusters) - 1, 1))
        for idx, cid in enumerate(unique_clusters)
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.8)
        ax.tick_params(colors="#111827")
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")

    ax_left, ax_right = axes

    for cid in unique_clusters:
        idx = [i for i, pair in enumerate(pair_keys) if pair_to_cluster.get(pair) == cid]
        if not idx:
            continue
        matched_idx = [i for i in idx if pair_keys[i] in gt_pair_types]
        unmatched_idx = [i for i in idx if pair_keys[i] not in gt_pair_types]
        color = cluster_palette[cid]
        if unmatched_idx:
            ax_left.scatter(coords[unmatched_idx, 0], coords[unmatched_idx, 1],
                            s=48, c=[color], marker="x", alpha=0.75,
                            linewidths=1.2, zorder=2)
        if matched_idx:
            ax_left.scatter(coords[matched_idx, 0], coords[matched_idx, 1],
                            s=112, c=[color], marker="o", alpha=0.92,
                            linewidths=0.9, edgecolors="white", zorder=3)

        centroid = coords[idx].mean(axis=0)
        cluster_name = cluster_names.get(cid, "")
        label = f"C{cid:02d}"
        if cluster_name:
            label = f"{label}\n{cluster_name.replace('_', ' ').title()}"
        ax_left.text(
            centroid[0], centroid[1], label,
            ha="center", va="center", fontsize=8.5, color="#111827",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.92},
            zorder=4,
        )

    left_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6b7280",
               markeredgecolor="white", markersize=9, label="GT-matched predicted pair"),
        Line2D([0], [0], marker="x", color="#6b7280", markersize=8,
               linestyle="None", label="Predicted-only pair"),
    ]
    ax_left.legend(handles=left_handles, loc="upper left", fontsize=8.5,
                   framealpha=0.92, facecolor="white", edgecolor="#d1d5db")
    ax_left.set_title(
        f"Predicted Row-Pair Clusters\n{n_pairs} candidate pairs  |  colors = raw cluster ids",
        fontsize=11.5, color="#111827", pad=10,
    )

    ax_right.scatter(coords[:, 0], coords[:, 1], s=24, c="#d1d5db", alpha=0.6,
                     linewidths=0, zorder=1)
    x_span = max(float(coords[:, 0].max() - coords[:, 0].min()), 1.0)
    y_span = max(float(coords[:, 1].max() - coords[:, 1].min()), 1.0)

    for i, pair in enumerate(pair_keys):
        rel_types = gt_pair_types.get(pair, set())
        if len(rel_types) > 1:
            ax_right.scatter(coords[i, 0], coords[i, 1], s=150, facecolors="none",
                             edgecolors="#111827", linewidths=1.2, zorder=2)

    for rel_type in REL_TYPES:
        rel_points = []
        for i, pair in enumerate(pair_keys):
            if rel_type not in gt_pair_types.get(pair, set()):
                continue
            dx, dy = _rel_offset(rel_type, x_span, y_span)
            rel_points.append((coords[i, 0] + dx, coords[i, 1] + dy))
        if not rel_points:
            continue
        rel_arr = np.asarray(rel_points, dtype=np.float32)
        ax_right.scatter(rel_arr[:, 0], rel_arr[:, 1],
                         s=110, c=[_rel_color(rel_type)], marker=_rel_marker(rel_type),
                         alpha=0.96, linewidths=0.9, edgecolors="white", zorder=3)

    right_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d1d5db",
               markeredgecolor="#d1d5db", markersize=7, label="All predicted pairs"),
    ]
    for rel_type in REL_TYPES:
        right_handles.append(
            Line2D([0], [0], marker=_rel_marker(rel_type), color="none",
                   markerfacecolor=_rel_color(rel_type), markeredgecolor="white",
                   markersize=9, label=f"GT {rel_type}")
        )
    right_handles.append(
        Line2D([0], [0], marker="o", color="#111827", markerfacecolor="none",
               markersize=9, linestyle="None", label="Multi-label GT pair")
    )
    ax_right.legend(handles=right_handles, loc="upper left", fontsize=8.5,
                    framealpha=0.92, facecolor="white", edgecolor="#d1d5db")
    ax_right.set_title(
        "Ground-Truth Relation Overlay\nMarkers show GT relationship evidence",
        fontsize=11.5, color="#111827", pad=10,
    )

    for ax in axes:
        ax.set_xlabel("projection dim 1", color="#111827")
    ax_left.set_ylabel("projection dim 2", color="#111827")
    fig.suptitle(
        f"LOKI - Pair-Cluster Embedding View (t-SNE)\nAdmission {ADMISSION_ID}",
        fontsize=13, color="#111827", y=1.02,
    )

    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved t-SNE plot: {out_path}")


def visualize_semantic_cluster_projection(
    paths: List[Dict],
    refined_rows: torch.Tensor,
    n_diag: int,
    refined_sentences: torch.Tensor,
    gt_relationships: List[Dict],
    out_path: str,
    sentence_encoder: Optional[SentenceTransformer] = None,
    cluster_key: str = "cluster_id",
    pair_embedding_mode: str = "contextual_sentence_average",
    negative_pairs: Optional[set[Tuple[int, int]]] = None,
) -> None:
    """
    Paper-style semantic relationship map over all predicted join paths.

    The reference figure is a 4-group semantic projection, not a micro-cluster
    audit. To match that structure, every predicted join path is assigned to a
    high-level semantic type and then projected inside a fixed quadrant anchor
    for that type. This preserves all predicted paths while making the semantic
    grouping legible in the same visual language as the paper mockup.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        print(f"  Semantic cluster projection skipped (missing library): {exc}")
        return

    pair_keys, pair_embeddings = _compute_pair_embeddings(
        paths,
        refined_sentences,
        refined_rows=refined_rows,
        n_diag=n_diag,
        sentence_encoder=sentence_encoder,
        embedding_mode=pair_embedding_mode,
        verbose=False,
    )
    if not pair_keys:
        print("  Semantic cluster projection skipped (no pair embeddings available).")
        return

    pair_to_index = {pair: index for index, pair in enumerate(pair_keys)}
    pair_embeddings_np = _l2_normalize_rows(_to_numpy_array(pair_embeddings, dtype=np.float32))
    refined_sentences_np = _l2_normalize_rows(refined_sentences.float().cpu().numpy())

    cluster_paths: Dict[int, List[Dict]] = defaultdict(list)
    for path in paths:
        cluster_id = int(path.get(cluster_key, path.get("cluster_id", -1)))
        if cluster_id < 0:
            continue
        cluster_paths[cluster_id].append(path)

    if not cluster_paths:
        print("  Semantic cluster projection skipped (no cluster ids available on retained paths).")
        return

    gt_pair_types = _build_gt_pair_type_lookup(gt_relationships)
    negative_pair_set = set(negative_pairs or set())
    cluster_semantic_types = {
        cluster_id: _resolve_cluster_semantic_type(cpaths, gt_pair_types, negative_pair_set)
        for cluster_id, cpaths in cluster_paths.items()
    }

    semantic_paths: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    semantic_counts: Dict[str, int] = defaultdict(int)
    primary_types = {"TREATS", "ADVERSE_EFFECT", "NEGATIVE", "DISCONTINUED", "CONTRAINDICATED"}
    for path in paths:
        pair = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        pair_index = pair_to_index.get(pair)
        if pair_index is None:
            continue

        cluster_id = int(path.get(cluster_key, path.get("cluster_id", -1)))
        cluster_semantic_type = cluster_semantic_types.get(cluster_id, "OTHER")
        semantic_type = _resolve_path_semantic_type(
            path,
            cluster_semantic_type,
            gt_pair_types,
            negative_pairs=negative_pair_set,
        )
        if semantic_type not in primary_types:
            semantic_type = "OTHER"

        pair_embedding = pair_embeddings_np[pair_index]
        sent_idx = int(path.get("sent_idx", -1))
        if 0 <= sent_idx < refined_sentences_np.shape[0] and pair_embedding.shape == refined_sentences_np.shape[1:]:
            path_embedding = (0.72 * pair_embedding) + (0.28 * refined_sentences_np[sent_idx])
            norm = float(np.linalg.norm(path_embedding))
            if norm > 0.0:
                path_embedding = path_embedding / norm
        else:
            path_embedding = pair_embedding

        semantic_paths[semantic_type].append({
            "embedding": np.asarray(path_embedding, dtype=np.float32),
            "path": path,
        })
        semantic_counts[semantic_type] += 1

    if not semantic_paths:
        print("  Semantic cluster projection skipped (no semantic path assignments available).")
        return

    semantic_signatures = _derive_semantic_signatures(semantic_paths)

    display_order = [
        rel_type
        for rel_type in ("TREATS", "ADVERSE_EFFECT", "NEGATIVE", "DISCONTINUED", "CONTRAINDICATED")
        if semantic_paths.get(rel_type)
    ]
    if semantic_paths.get("OTHER"):
        display_order.append("OTHER")

    semantic_summary = ", ".join(
        f"{rel_type}={semantic_counts.get(rel_type, 0)}"
        for rel_type in display_order
    )
    print(f"  Semantic type counts: {semantic_summary}")

    anchor_positions: Dict[str, np.ndarray] = {
        "TREATS": np.asarray([-2.35, 1.55], dtype=np.float32),
        "ADVERSE_EFFECT": np.asarray([2.35, 1.55], dtype=np.float32),
        "NEGATIVE": np.asarray([-2.35, -1.55], dtype=np.float32),
        "DISCONTINUED": np.asarray([2.35, -1.55], dtype=np.float32),
        "CONTRAINDICATED": np.asarray([0.0, 0.0], dtype=np.float32),
        "OTHER": np.asarray([0.0, -2.3], dtype=np.float32),
    }
    callout_positions: Dict[str, Tuple[float, float]] = {
        "TREATS": (-2.55, 2.45),
        "ADVERSE_EFFECT": (2.55, 2.45),
        "NEGATIVE": (-2.55, -2.62),
        "DISCONTINUED": (2.55, -2.62),
        "CONTRAINDICATED": (0.0, 2.9),
        "OTHER": (0.0, -3.05),
    }

    fig, ax = plt.subplots(figsize=(13.6, 10.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#d1d5db")
        spine.set_linewidth(1.0)

    group_scales: Dict[str, np.ndarray] = {
        "TREATS": np.asarray([1.05, 0.82], dtype=np.float32),
        "ADVERSE_EFFECT": np.asarray([1.05, 0.82], dtype=np.float32),
        "NEGATIVE": np.asarray([0.95, 0.78], dtype=np.float32),
        "DISCONTINUED": np.asarray([0.95, 0.78], dtype=np.float32),
        "CONTRAINDICATED": np.asarray([0.78, 0.68], dtype=np.float32),
        "OTHER": np.asarray([0.78, 0.64], dtype=np.float32),
    }

    for rel_type in display_order:
        records = semantic_paths.get(rel_type, [])
        if not records:
            continue

        embeddings = np.stack(
            [np.asarray(record["embedding"], dtype=np.float32) for record in records],
            axis=0,
        )
        local_coords, method = _project_cluster_map_embeddings(embeddings)
        if local_coords.shape[0] > 1:
            local_coords = local_coords - local_coords.mean(axis=0, keepdims=True)
            denom = max(float(np.max(np.abs(local_coords))), 1e-6)
            if denom <= 1e-5:
                angles = np.linspace(0.0, 2.0 * np.pi, local_coords.shape[0], endpoint=False, dtype=np.float32)
                local_coords = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.24
            else:
                local_coords = local_coords / denom
        else:
            local_coords = np.zeros((1, 2), dtype=np.float32)

        anchor = anchor_positions.get(rel_type, anchor_positions["OTHER"])
        scaled_coords = anchor + (local_coords * group_scales.get(rel_type, group_scales["OTHER"]))
        color = _rel_color(rel_type)
        ax.scatter(
            scaled_coords[:, 0], scaled_coords[:, 1],
            s=90, c=[color], marker="o", alpha=0.92,
            linewidths=1.15, edgecolors="white", zorder=3,
        )

        callout_x, callout_y = callout_positions.get(rel_type, callout_positions["OTHER"])
        ax.text(
            callout_x,
            callout_y,
            _semantic_signature_text(semantic_signatures.get(rel_type, tuple())),
            ha="center",
            va="center",
            fontsize=12.5,
            color="#4b5563",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#a3a3a3",
                "linewidth": 1.4,
                "alpha": 0.98,
            },
            zorder=4,
        )

    print(
        f"  Running semantic grouping on {len(paths)} predicted join paths "
        f"across {len(display_order)} semantic relationship groups ..."
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_rel_color(rel_type),
            markeredgecolor="white",
            markersize=10.5,
            label=f"R{index}: {rel_type}",
        )
        for index, rel_type in enumerate(display_order, start=1)
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(legend_handles),
        title="Latent Relationships",
        frameon=True,
        framealpha=0.98,
        facecolor="white",
        edgecolor="#a3a3a3",
        fontsize=12.5,
        title_fontsize=13.5,
    )

    ax.set_title(
        "Fig X: Semantic Projection of Discovered Join Paths",
        fontsize=17.0,
        color="#111827",
        pad=56,
    )
    ax.set_xlabel("Latent Dimension 1", color="#111827", fontsize=12.5)
    ax.set_ylabel("Latent Dimension 2", color="#111827", fontsize=12.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-4.05, 4.05)
    ax.set_ylim(-3.25, 3.25)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved semantic cluster projection: {out_path}")



def visualize_dataset_semantic_projection(
    dataset_name: str,
    records: List[Dict[str, object]],
    out_path: Path,
    max_points_per_type: int = 300,
    max_total_points: int = 1800,
) -> None:
    if len(records) < 4:
        print("  Batch semantic projection skipped (not enough semantic join-path records).")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        print(f"  Batch semantic projection skipped (missing library): {exc}")
        return

    sampled_records = _sample_projection_points(
        records,
        max_points_per_type=max_points_per_type,
        max_total_points=max_total_points,
    )
    grouped_sampled: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    grouped_all: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in sampled_records:
        grouped_sampled[str(record["label"])].append(record)
    for record in records:
        grouped_all[str(record["label"])].append(record)

    display_order = [
        rel_type
        for rel_type in ("TREATS", "ADVERSE_EFFECT", "NEGATIVE", "DISCONTINUED", "CONTRAINDICATED")
        if grouped_sampled.get(rel_type)
    ]
    if not display_order:
        print("  Batch semantic projection skipped (no primary semantic groups retained).")
        return

    semantic_signatures = _derive_semantic_signatures({
        rel_type: grouped_all.get(rel_type, [])
        for rel_type in display_order
    })
    admissions = sorted({str(record.get("admission_id", "")) for record in records if record.get("admission_id")})
    semantic_summary = ", ".join(
        f"{rel_type}={len(grouped_all.get(rel_type, []))}"
        for rel_type in display_order
    )
    print(f"  Batch semantic type counts: {semantic_summary}")

    anchor_positions: Dict[str, np.ndarray] = {
        "TREATS": np.asarray([-2.45, 1.6], dtype=np.float32),
        "ADVERSE_EFFECT": np.asarray([2.45, 1.6], dtype=np.float32),
        "NEGATIVE": np.asarray([-2.45, -1.6], dtype=np.float32),
        "DISCONTINUED": np.asarray([2.45, -1.6], dtype=np.float32),
        "CONTRAINDICATED": np.asarray([0.0, 0.0], dtype=np.float32),
    }
    callout_positions: Dict[str, Tuple[float, float]] = {
        "TREATS": (-2.65, 2.55),
        "ADVERSE_EFFECT": (2.65, 2.55),
        "NEGATIVE": (-2.65, -2.72),
        "DISCONTINUED": (2.65, -2.72),
        "CONTRAINDICATED": (0.0, 2.95),
    }
    group_scales: Dict[str, np.ndarray] = {
        "TREATS": np.asarray([1.08, 0.86], dtype=np.float32),
        "ADVERSE_EFFECT": np.asarray([1.08, 0.86], dtype=np.float32),
        "NEGATIVE": np.asarray([1.0, 0.82], dtype=np.float32),
        "DISCONTINUED": np.asarray([1.0, 0.82], dtype=np.float32),
        "CONTRAINDICATED": np.asarray([0.86, 0.72], dtype=np.float32),
    }

    fig, ax = plt.subplots(figsize=(14.0, 10.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#d1d5db")
        spine.set_linewidth(1.0)

    for rel_type in display_order:
        items = grouped_sampled.get(rel_type, [])
        if not items:
            continue

        embeddings = np.stack(
            [np.asarray(item["embedding"], dtype=np.float32) for item in items],
            axis=0,
        )
        local_coords, _method = _project_cluster_map_embeddings(embeddings)
        if local_coords.shape[0] > 1:
            local_coords = local_coords - local_coords.mean(axis=0, keepdims=True)
            denom = max(float(np.max(np.abs(local_coords))), 1e-6)
            if denom <= 1e-5:
                angles = np.linspace(0.0, 2.0 * np.pi, local_coords.shape[0], endpoint=False, dtype=np.float32)
                local_coords = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.24
            else:
                local_coords = local_coords / denom
        else:
            local_coords = np.zeros((1, 2), dtype=np.float32)

        scaled_coords = anchor_positions[rel_type] + (local_coords * group_scales[rel_type])
        ax.scatter(
            scaled_coords[:, 0],
            scaled_coords[:, 1],
            s=75,
            c=[_rel_color(rel_type)],
            marker="o",
            alpha=0.9,
            linewidths=1.0,
            edgecolors="white",
            zorder=3,
        )

        callout_x, callout_y = callout_positions[rel_type]
        ax.text(
            callout_x,
            callout_y,
            _semantic_signature_text(semantic_signatures.get(rel_type, tuple())),
            ha="center",
            va="center",
            fontsize=12.5,
            color="#4b5563",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#a3a3a3",
                "linewidth": 1.35,
                "alpha": 0.98,
            },
            zorder=4,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_rel_color(rel_type),
            markeredgecolor="white",
            markersize=10.5,
            label=f"R{index}: {rel_type}",
        )
        for index, rel_type in enumerate(display_order, start=1)
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(display_order),
        title="Latent Relationships",
        frameon=True,
        framealpha=0.98,
        facecolor="white",
        edgecolor="#a3a3a3",
        fontsize=12.5,
        title_fontsize=13.5,
    )

    ax.set_title(
        f"Semantic Projection of Discovered Join Paths - {dataset_name}\n"
        f"Sampled predicted join paths across {len(admissions)} admissions (n={len(sampled_records)})",
        fontsize=16.0,
        color="#111827",
        pad=62,
    )
    ax.set_xlabel("Latent Dimension 1", color="#111827", fontsize=12.5)
    ax.set_ylabel("Latent Dimension 2", color="#111827", fontsize=12.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-3.4, 3.4)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved batch semantic projection: {out_path}")


def visualize_all_sentences_tsne(
    refined_sentences: torch.Tensor,
    paths: List[Dict],
    gt_relationships: List[Dict],
    out_path: str,
    min_cluster_size: int = 8,
) -> None:
    """Project all refined sentence embeddings with mediator and GT overlays."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import normalize
    except ImportError as exc:
        print(f"  Full t-SNE visualization skipped (missing library): {exc}")
        return

    n_sents = refined_sentences.shape[0]
    emb = refined_sentences.float().cpu().numpy()
    emb_normed = normalize(emb, norm="l2")

    # --- cluster ALL sentences with HDBSCAN ---
    cluster_labels, cluster_backend = _fit_hdbscan_labels(
        emb_normed,
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        verbose=True,
        context="full-sentence visualization clustering",
    )

    unique_clusters = sorted(set(cluster_labels))
    n_real = sum(1 for c in unique_clusters if c >= 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"  Full-sentence HDBSCAN ({cluster_backend}): {n_real} clusters, {n_noise} noise points")

    # --- t-SNE ---
    print(f"  Running t-SNE on {n_sents} sentence embeddings ...")
    perplexity = min(30, max(5, n_sents // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, init="pca", max_iter=1000)
    coords = tsne.fit_transform(emb_normed)

    # --- color palette for clusters ---
    palette = plt.colormaps.get_cmap("tab20")
    cluster_color = {
        c: palette(i / max(n_real - 1, 1))
        for i, c in enumerate(c for c in unique_clusters if c >= 0)
    }
    cluster_color[-1] = "#d1d5db"

    # --- metadata ---
    med_indices: set = {p["sent_idx"] for p in paths}
    gt_sents: set = {j for rel in gt_relationships for j in rel.get("evidence_sents", [])}

    # --- plot ---
    fig, ax = plt.subplots(figsize=(14, 9.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.8)

    noise_idx = [j for j in range(n_sents) if cluster_labels[j] == -1 and j not in med_indices and j not in gt_sents]
    if noise_idx:
        ax.scatter(coords[noise_idx, 0], coords[noise_idx, 1],
                   s=12, c=[cluster_color[-1]], alpha=0.28, linewidths=0,
                   label="Noise/background", zorder=1)

    # 1. Clustered sentences (background layer)
    for cid in unique_clusters:
        if cid < 0:
            continue
        idx = [j for j in range(n_sents)
               if cluster_labels[j] == cid and j not in med_indices and j not in gt_sents]
        if not idx:
            continue
        color = cluster_color[cid]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   s=26, c=[color], alpha=0.60, linewidths=0, zorder=2)
        centroid = coords[idx].mean(axis=0)
        ax.text(
            centroid[0], centroid[1], f"S{cid}",
            ha="center", va="center", fontsize=8.5, color="#111827",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
            zorder=6,
        )

    # 2. GT evidence sentences (white diamond outline, cluster color fill)
    gt_only = [j for j in gt_sents if j not in med_indices]
    if gt_only:
        colors_gt = [cluster_color[cluster_labels[j]] for j in gt_only]
        ax.scatter(coords[gt_only, 0], coords[gt_only, 1],
                   s=100, c=colors_gt, marker="D", alpha=0.9,
                   linewidths=1.1, edgecolors="#111827", label="GT evidence", zorder=4)

    # 3. Mediating sentences (LOKI selected) - black star outline, cluster color fill
    med_list = sorted(med_indices)
    if med_list:
        colors_med = [cluster_color[cluster_labels[j]] for j in med_list]
        is_also_gt = [j in gt_sents for j in med_list]
        # non-GT mediators
        ngt = [j for j, g in zip(med_list, is_also_gt) if not g]
        if ngt:
            ax.scatter(coords[ngt, 0], coords[ngt, 1],
                       s=200, c=[cluster_color[cluster_labels[j]] for j in ngt],
                       marker="o", alpha=1.0, linewidths=1.2, edgecolors="black",
                       label="Mediating (LOKI)", zorder=5)
        # GT + mediating
        gtm = [j for j, g in zip(med_list, is_also_gt) if g]
        if gtm:
            ax.scatter(coords[gtm, 0], coords[gtm, 1],
                       s=320, c=[cluster_color[cluster_labels[j]] for j in gtm],
                       marker="*", alpha=1.0, linewidths=1.2, edgecolors="black",
                       label="Mediating + GT", zorder=6)

    ax.set_title(
        "LOKI - Sentence Embedding Neighborhoods (t-SNE)\n"
        f"Admission {ADMISSION_ID}  |  {n_sents} sentences  "
        f"|  {n_real} clusters  |  {n_noise} noise",
        color="#111827", fontsize=11.5,
    )
    ax.set_xlabel("t-SNE dim 1", color="#111827")
    ax.set_ylabel("t-SNE dim 2", color="#111827")
    ax.tick_params(colors="#4b5563")
    for spine in ax.spines.values():
        spine.set_edgecolor("#d1d5db")

    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles, leg_labels, loc="upper right", fontsize=8,
              framealpha=0.92, facecolor="white", edgecolor="#d1d5db",
              labelcolor="#111827")

    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved full-cluster t-SNE plot: {out_path}")


# =============================================================================
# Main
# =============================================================================

def compute_pair_average_precision(
    pair_scores: torch.Tensor,
    n_diag: int,
    gt_pairs: set,
) -> Optional[float]:
    if pair_scores.numel() == 0:
        return None

    try:
        from sklearn.metrics import average_precision_score  # type: ignore
    except Exception:
        return None

    score_matrix = pair_scores.detach().cpu().numpy()
    n_med = score_matrix.shape[0] - n_diag
    if n_med <= 0 or not gt_pairs:
        return None

    y_true: List[int] = []
    y_score: List[float] = []
    for diag_idx in range(n_diag):
        diag_scores = score_matrix[diag_idx]
        for med_idx in range(n_med):
            med_scores = score_matrix[n_diag + med_idx]
            best_score = float(np.max((diag_scores + med_scores) / 2.0))
            y_true.append(1 if (diag_idx, med_idx) in gt_pairs else 0)
            y_score.append(best_score)

    if len(set(y_true)) < 2:
        return None

    return round(float(average_precision_score(y_true, y_score)), 4)


def _compute_pair_level_cluster_silhouette(
    paths: List[Dict],
    refined_sentences: torch.Tensor,
    refined_rows: Optional[torch.Tensor],
    n_diag: int,
    sentence_encoder: Optional[SentenceTransformer],
    embedding_mode: str,
    cluster_key: str = "cluster_id",
) -> Optional[float]:
    if not paths:
        return None

    try:
        pair_keys, pair_embeddings = _compute_pair_embeddings(
            paths,
            refined_sentences,
            refined_rows=refined_rows,
            n_diag=n_diag,
            sentence_encoder=sentence_encoder,
            embedding_mode=embedding_mode,
            verbose=False,
        )
    except Exception:
        return None

    if not pair_keys:
        return None

    pair_cluster_votes: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for path in paths:
        raw_label = path.get(cluster_key)
        if raw_label is None:
            continue
        try:
            cluster_id = int(raw_label)
        except (TypeError, ValueError):
            continue
        pair = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        pair_cluster_votes[pair][cluster_id] += 1

    labels_for_pairs: List[int] = []
    kept_embeddings: List[np.ndarray] = []
    for pair, embedding in zip(pair_keys, pair_embeddings):
        votes = pair_cluster_votes.get(pair)
        if not votes:
            continue
        cluster_id = max(votes.items(), key=lambda item: (item[1], -item[0]))[0]
        # Ignore noise / suppressed labels when computing structural separation.
        if cluster_id < 0:
            continue
        labels_for_pairs.append(cluster_id)
        kept_embeddings.append(_to_numpy_array(embedding, dtype=np.float32))

    if len(kept_embeddings) < 2 or len(set(labels_for_pairs)) < 2:
        return None

    try:
        from sklearn.metrics import silhouette_score  # type: ignore

        emb_np = _l2_normalize_rows(np.asarray(kept_embeddings, dtype=np.float32))
        return round(float(silhouette_score(emb_np, np.asarray(labels_for_pairs, dtype=int))), 4)
    except Exception:
        return None


def merge_clusters_by_shared_pairs(
    clusters: Dict[int, List[Dict]],
    labels: np.ndarray,
    paths: List[Dict],
) -> Tuple[Dict[int, List[Dict]], np.ndarray, Dict[int, int]]:
    """Merge any two HDBSCAN clusters that share a (diag_idx, med_idx) pair.

    Uses union-find: two clusters sharing at least one row pair are merged into
    a single cluster with the minimum cluster id as canonical id. This is
    zero-purity-risk because shared row-pairs must belong to one relationship.

    Returns:
        merged_clusters: Dict mapping new canonical cluster id -> list of paths
        remapped_labels:  np.ndarray aligned to `paths` with updated cluster ids
        merge_map:        Dict[old_id, new_canonical_id] (empty if no merges)
    """
    if not clusters or len(clusters) <= 1:
        return clusters, labels, {}

    # Build pair -> set-of-cluster-ids index
    pair_to_clusters: Dict[Tuple[int, int], set] = defaultdict(set)
    for cid, cpaths in clusters.items():
        for path in cpaths:
            pair_to_clusters[(int(path["diag_row_idx"]), int(path["med_row_idx"]))].add(cid)

    # Union-Find
    parent: Dict[int, int] = {cid: cid for cid in clusters}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    for cids in pair_to_clusters.values():
        cid_list = sorted(cids)
        for i in range(1, len(cid_list)):
            _union(cid_list[0], cid_list[i])

    merge_map: Dict[int, int] = {cid: _find(cid) for cid in clusters}

    if len(set(merge_map.values())) == len(clusters):
        return clusters, labels, {}  # nothing to merge

    n_before = len(clusters)
    n_after = len(set(merge_map.values()))
    print(
        f"  Shared-pair merge: {n_before} \u2192 {n_after} clusters "
        f"({n_before - n_after} merges)"
    )

    merged_clusters: Dict[int, List[Dict]] = defaultdict(list)
    for cid, cpaths in clusters.items():
        merged_clusters[merge_map[cid]].extend(cpaths)

    remapped_labels = np.array(
        [merge_map.get(int(lbl), int(lbl)) for lbl in labels], dtype=int
    )
    return dict(merged_clusters), remapped_labels, merge_map


def _build_cluster_distance_matrix(
    clusters: Dict[int, List[Dict]],
    alpha_sent: float = 0.6,
    alpha_pair: float = 0.4,
) -> Tuple[np.ndarray, List[int]]:
    """Build condensed pairwise distance vector between clusters for scipy linkage.

    Distance = alpha_sent * Jaccard-sentence + alpha_pair * Jaccard-pair.
    Returns (condensed_distance_vector, ordered_cluster_id_list).
    """
    cids = sorted(clusters.keys())
    n = len(cids)
    cluster_sents: Dict[int, frozenset] = {
        cid: frozenset(int(p["sent_idx"]) for p in cpaths)
        for cid, cpaths in clusters.items()
    }
    cluster_pairs: Dict[int, frozenset] = {
        cid: frozenset((int(p["diag_row_idx"]), int(p["med_row_idx"])) for p in cpaths)
        for cid, cpaths in clusters.items()
    }
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = cids[i], cids[j]
            si, sj = cluster_sents[ci], cluster_sents[cj]
            j_sent = 1.0 - len(si & sj) / max(len(si | sj), 1)
            pi, pj = cluster_pairs[ci], cluster_pairs[cj]
            j_pair = 1.0 - len(pi & pj) / max(len(pi | pj), 1)
            D[i, j] = D[j, i] = alpha_sent * j_sent + alpha_pair * j_pair
    condensed = D[np.triu_indices(n, k=1)]
    return condensed, cids


def meta_cluster_hierarchical(
    clusters: Dict[int, List[Dict]],
    labels: np.ndarray,
    paths: List[Dict],
    n_meta_clusters: int = 0,
    alpha_sent: float = 0.6,
    alpha_pair: float = 0.4,
) -> Tuple[Dict[int, List[Dict]], np.ndarray, Dict]:
    """Hierarchically merge HDBSCAN clusters by Jaccard sentence/pair distance.

    n_meta_clusters=0: silhouette-optimal cut via sklearn.
    n_meta_clusters>0: cut to exactly that many clusters.
    """
    n = len(clusters)
    if n <= 1:
        return clusters, labels, {"n_before": n, "n_after": n, "skipped": True}

    try:
        from scipy.cluster.hierarchy import linkage, fcluster  # type: ignore
        from scipy.spatial.distance import squareform  # type: ignore
    except ImportError:
        print("  Warning: scipy not available \u2014 meta_cluster_hierarchical skipped")
        return clusters, labels, {"skipped": True, "reason": "scipy_unavailable"}

    condensed, cids = _build_cluster_distance_matrix(
        clusters, alpha_sent=alpha_sent, alpha_pair=alpha_pair
    )
    if n <= 2:
        return clusters, labels, {"n_before": n, "n_after": n, "skipped": True, "reason": "too_few"}

    Z = linkage(condensed, method="average")

    if n_meta_clusters > 0:
        flat = fcluster(Z, t=max(1, min(n_meta_clusters, n - 1)), criterion="maxclust")
    else:
        try:
            from sklearn.metrics import silhouette_score  # type: ignore
            D_full = squareform(condensed)
            best_k, best_score = 2, -1.0
            for k in range(2, n):
                flat_k = fcluster(Z, t=k, criterion="maxclust")
                if len(set(flat_k)) < 2:
                    continue
                sc = float(silhouette_score(D_full, flat_k, metric="precomputed"))
                if sc > best_score:
                    best_k, best_score = k, sc
            flat = fcluster(Z, t=best_k, criterion="maxclust")
        except Exception:
            flat = fcluster(Z, t=max(2, n // 2), criterion="maxclust")

    merge_map = {cids[i]: int(flat[i]) for i in range(n)}
    n_after = len(set(merge_map.values()))
    if n_after == n:
        return clusters, labels, {"n_before": n, "n_after": n, "merged": 0}

    print(f"  Meta-cluster merge: {n} \u2192 {n_after} clusters ({n - n_after} merges)")
    merged_clusters: Dict[int, List[Dict]] = defaultdict(list)
    for cid, cpaths in clusters.items():
        merged_clusters[merge_map[cid]].extend(cpaths)
    remapped_labels = np.array(
        [merge_map.get(int(lbl), int(lbl)) for lbl in labels], dtype=int
    )
    return dict(merged_clusters), remapped_labels, {"n_before": n, "n_after": n_after, "merged": n - n_after}


def _ce_format_query(diag_text: str, med_text: str) -> str:
    return f"Diagnosis: {diag_text} | Medication: {med_text}"


def _ce_format_passage(sent_text: str, section_name: str, prefix_section: bool) -> str:
    if prefix_section and section_name:
        return f"[{section_name}] {sent_text}"
    return sent_text


def _percentile_summary(scores: np.ndarray) -> str:
    if scores.size == 0:
        return "n=0"
    qs = np.percentile(scores, [0, 10, 25, 50, 75, 90, 100])
    return (
        f"n={scores.size}  min={qs[0]:.4f}  p10={qs[1]:.4f}  p25={qs[2]:.4f}  "
        f"p50={qs[3]:.4f}  p75={qs[4]:.4f}  p90={qs[5]:.4f}  max={qs[6]:.4f}  "
        f"mean={scores.mean():.4f}  std={scores.std():.4f}"
    )


def _compute_auc_pr_roc(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Return (AUC-ROC, AUC-PR) for binary `labels` using `scores`. Returns
    (nan, nan) if labels are degenerate."""
    if scores.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan"), float("nan")
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore
        return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
    except Exception:
        return float("nan"), float("nan")


def rerank_pair_sentences_with_cross_encoder(
    paths: List[Dict],
    diag_rows: List[str],
    med_rows: List[str],
    sent_texts: List[str],
    sent_meta: Dict[int, str],
    gt_relationships: List[Dict],
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
    fp16: bool = True,
    section_prefix: bool = True,
    normalize: bool = True,
) -> Dict[str, object]:
    """Option C - Per-pair sentence reranker. For every surviving (diag, med)
    pair in ``paths``, score each of its mediating sentences with a zero-shot
    cross-encoder and write a ``ce_score`` field onto every path record. The
    downstream signature builders prefer ``ce_score`` when present, so Phase E
    receives the CE-preferred evidence sentence per pair without any path being
    added, dropped, or re-pair-filtered. Pipeline pair counts / pair recall are
    therefore unchanged; only the *evidence ordering within each pair* shifts.

    Returns a small summary dict (sizes, timings, optional GT separation AUCs).
    """
    from cross_encoder_rerank import build_reranker  # local import: keep CE optional

    if not paths:
        return {"n_pairs": 0, "n_sentences": 0, "model": model_name}

    # Group surviving paths by (diag_row_idx, med_row_idx).
    pair_buckets: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for p in paths:
        pair_buckets[(int(p["diag_row_idx"]), int(p["med_row_idx"]))].append(p)

    n_pairs = len(pair_buckets)
    n_sent_total = sum(len(v) for v in pair_buckets.values())
    print(
        f"  Reranking sentences within {n_pairs} surviving (diag, med) pairs "
        f"({n_sent_total} total path-sentences) using {model_name}"
    )

    t_load_start = time.time()
    reranker = build_reranker(
        model_name=model_name,
        device=device,
        max_length=max_length,
        fp16=fp16,
        normalize=normalize,
    )
    t_load = time.time() - t_load_start
    print(f"  Model loaded in {t_load:.1f}s  (backend={getattr(reranker, 'name', model_name)})")

    # Build the full (query, passage) batch in deterministic order. Each entry
    # corresponds to exactly one path record we will later annotate.
    ce_inputs: List[Tuple[str, str]] = []
    flat_paths: List[Dict] = []
    for (i_a, i_b), pair_paths in pair_buckets.items():
        query = _ce_format_query(diag_rows[i_a], med_rows[i_b])
        for path in pair_paths:
            sent_idx = int(path["sent_idx"])
            passage = _ce_format_passage(
                sent_texts[sent_idx], sent_meta.get(sent_idx, ""), section_prefix
            )
            ce_inputs.append((query, passage))
            flat_paths.append(path)

    t_score_start = time.time()
    scores = reranker.score(ce_inputs, batch_size=batch_size)
    t_score = time.time() - t_score_start
    rate = len(ce_inputs) / max(t_score, 1e-6)
    print(f"  Scored {len(ce_inputs)} (query, sentence) pairs in {t_score:.1f}s  ({rate:.1f} pairs/sec)")

    scores_np = np.asarray(scores, dtype=np.float64)
    for path, s in zip(flat_paths, scores_np):
        path["ce_score"] = float(s)

    # Brief score distribution.
    print(f"  CE score distribution: {_percentile_summary(scores_np)}")

    # How often did CE pick a different top-1 sentence than LOKI?
    n_reordered = 0
    n_multi_sent_pairs = 0
    for pair_paths in pair_buckets.values():
        if len(pair_paths) < 2:
            continue
        n_multi_sent_pairs += 1
        loki_top = max(pair_paths, key=lambda p: float(p["path_score"]))
        ce_top = max(pair_paths, key=lambda p: float(p["ce_score"]))
        if int(loki_top["sent_idx"]) != int(ce_top["sent_idx"]):
            n_reordered += 1
    if n_multi_sent_pairs:
        print(
            f"  Top-1 evidence sentence changed in {n_reordered}/{n_multi_sent_pairs} "
            f"multi-sentence pairs ({n_reordered / n_multi_sent_pairs * 100:.1f}%)"
        )

    # Optional GT diagnostic: AUC for "is this triple in GT?" using CE scores,
    # restricted to the surviving candidate set. Useful for comparing CE
    # variants without re-running Phase E.
    summary: Dict[str, object] = {
        "n_pairs": n_pairs,
        "n_sentences": n_sent_total,
        "model": model_name,
        "load_seconds": round(t_load, 2),
        "score_seconds": round(t_score, 2),
        "top1_reorder_rate": (n_reordered / n_multi_sent_pairs) if n_multi_sent_pairs else 0.0,
        "n_multi_sentence_pairs": n_multi_sent_pairs,
    }
    if gt_relationships:
        gt_triple_set: Set[Tuple[int, int, int]] = set()
        for r in gt_relationships:
            d_idx = int(r["diag_idx"])
            m_idx = int(r["drug_idx"])
            for s in r.get("evidence_sents", []) or []:
                gt_triple_set.add((d_idx, int(s) - 1, m_idx))
        triple_labels = np.array(
            [
                1 if (int(p["diag_row_idx"]), int(p["sent_idx"]), int(p["med_row_idx"])) in gt_triple_set else 0
                for p in flat_paths
            ],
            dtype=np.int32,
        )
        roc_trip, ap_trip = _compute_auc_pr_roc(scores_np, triple_labels)
        print(
            f"  GT triple separation on surviving candidates: "
            f"AUC-ROC={roc_trip:.4f}  AUC-PR={ap_trip:.4f}  "
            f"(positives={int(triple_labels.sum())}/{triple_labels.size})"
        )
        summary["auc_roc_triple_on_candidates"] = roc_trip
        summary["auc_pr_triple_on_candidates"] = ap_trip

    return summary


def filter_pairs_by_cross_encoder(
    paths: List[Dict],
    mode: str = "combined",
    threshold: float = 0.05,
    quantile: float = 0.25,
    collect_details: bool = False,
) -> Tuple[List[Dict], Dict[str, object]]:
    """Option D - pair-level filter that uses CE scores written by Phase D.5.

    Operates on the per-pair max ``ce_score``. Three modes:

      - ``absolute``  drop pairs whose max ce_score is < ``threshold``.
      - ``quantile``  drop pairs whose max ce_score is in the bottom ``quantile``
                      of the CE distribution.
      - ``combined``  (conservative; recommended) drop a pair only when BOTH its
                      max LOKI ``path_score`` AND its max ``ce_score`` fall in
                      the bottom ``quantile`` of their respective distributions.
                      Never drops a pair that looks strong by either signal.

    Returns (kept_paths, stats_dict). ``stats_dict`` is suitable for inclusion
    in the run summary.
    """
    base_stats = {
        "enabled": mode != "off",
        "mode": mode,
        "threshold": threshold if mode == "absolute" else None,
        "quantile": quantile if mode in ("quantile", "combined") else None,
        "n_pairs_before": 0,
        "n_pairs_after": 0,
        "dropped_pairs": 0,
        "dropped_paths": 0,
    }
    if not paths or mode == "off":
        return paths, base_stats

    pair_buckets: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for p in paths:
        pair_buckets[(int(p["diag_row_idx"]), int(p["med_row_idx"]))].append(p)

    n_before = len(pair_buckets)
    base_stats["n_pairs_before"] = n_before
    base_stats["n_pairs_after"] = n_before

    pair_ce_max: Dict[Tuple[int, int], float] = {}
    for pair, pp in pair_buckets.items():
        ce_vals = [float(p["ce_score"]) for p in pp if p.get("ce_score") is not None]
        pair_ce_max[pair] = max(ce_vals) if ce_vals else float("-inf")
    pair_loki_max: Dict[Tuple[int, int], float] = {
        pair: max(float(p.get("path_score", 0.0)) for p in pp)
        for pair, pp in pair_buckets.items()
    }
    pair_support: Dict[Tuple[int, int], int] = {
        pair: len({int(p.get("sent_idx", -1)) for p in pp if int(p.get("sent_idx", -1)) >= 0})
        for pair, pp in pair_buckets.items()
    }

    have_ce = [v for v in pair_ce_max.values() if v != float("-inf")]
    if not have_ce:
        base_stats["reason"] = "no_ce_scores"
        print("  Option D CE pair filter skipped - no ce_score values on candidate paths.")
        return paths, base_stats

    drop_pairs: set = set()
    ce_cutoff: Optional[float] = None
    loki_cutoff: Optional[float] = None
    if mode == "absolute":
        ce_cutoff = float(threshold)
        for pair, ce in pair_ce_max.items():
            if ce != float("-inf") and ce < ce_cutoff:
                drop_pairs.add(pair)
        base_stats["cutoff_ce"] = ce_cutoff
    elif mode == "quantile":
        ce_cutoff = float(np.quantile(have_ce, quantile))
        for pair, ce in pair_ce_max.items():
            if ce != float("-inf") and ce < ce_cutoff:
                drop_pairs.add(pair)
        base_stats["cutoff_ce"] = ce_cutoff
    elif mode == "combined":
        ce_cutoff = float(np.quantile(have_ce, quantile))
        loki_cutoff = float(np.quantile(list(pair_loki_max.values()), quantile))
        for pair in pair_buckets:
            if pair_ce_max[pair] < ce_cutoff and pair_loki_max[pair] < loki_cutoff:
                drop_pairs.add(pair)
        base_stats["cutoff_ce"] = ce_cutoff
        base_stats["cutoff_loki"] = loki_cutoff
    else:
        raise ValueError(f"Unknown ce_pair_filter mode: {mode}")

    if collect_details:
        pair_decisions: List[Dict[str, object]] = []
        for pair in sorted(pair_buckets):
            best_ce = pair_ce_max[pair]
            best_loki = pair_loki_max[pair]
            kept = pair not in drop_pairs
            if kept:
                decision_reason = "kept"
            elif mode == "combined":
                decision_reason = "below_both_cutoffs"
            elif mode == "absolute":
                decision_reason = "below_ce_threshold"
            else:
                decision_reason = "below_ce_quantile"
            pair_decisions.append({
                "diag_row_idx": pair[0],
                "med_row_idx": pair[1],
                "kept": kept,
                "reason": decision_reason,
                "best_ce_score": None if best_ce == float("-inf") else round(float(best_ce), 4),
                "best_loki_score": round(float(best_loki), 4),
                "cutoff_ce": None if ce_cutoff is None else round(float(ce_cutoff), 4),
                "cutoff_loki": None if loki_cutoff is None else round(float(loki_cutoff), 4),
                "ce_delta_to_cutoff": None if ce_cutoff is None or best_ce == float("-inf") else round(float(ce_cutoff) - float(best_ce), 4),
                "loki_delta_to_cutoff": None if loki_cutoff is None else round(float(loki_cutoff) - float(best_loki), 4),
                "support_count": int(pair_support.get(pair, 0)),
            })
        base_stats["pair_decisions"] = pair_decisions

    if not drop_pairs:
        print(f"  Option D CE pair filter [{mode}]: kept all {n_before} pairs (no pair below cutoff).")
        return paths, base_stats

    kept_paths = [
        p for p in paths
        if (int(p["diag_row_idx"]), int(p["med_row_idx"])) not in drop_pairs
    ]
    n_dropped_paths = len(paths) - len(kept_paths)
    print(
        f"  Option D CE pair filter [{mode}]: dropped {len(drop_pairs)}/{n_before} pairs "
        f"({n_dropped_paths} paths)"
    )
    base_stats["n_pairs_after"] = n_before - len(drop_pairs)
    base_stats["dropped_pairs"] = len(drop_pairs)
    base_stats["dropped_paths"] = n_dropped_paths
    return kept_paths, base_stats


def run_materialization_pipeline(
    cli: argparse.Namespace,
    model: BidirectionalTableTextModel,
    model_args: Dict,
    diag_rows: List[str],
    med_rows: List[str],
    sent_texts: List[str],
    sent_meta: Dict[int, str],
    gt_relationships: List[Dict],
    gt_diag: Dict[int, List[int]],
    gt_med: Dict[int, List[int]],
    multi_pairs: set,
    negative_pairs: Optional[set[Tuple[int, int]]] = None,
    evaluation_target_pairs: Optional[Set[Tuple[int, int]]] = None,
    evaluation_profile: str = EVALUATION_PROFILE_DEFAULT,
) -> Dict:
    _PAIR_EMBEDDING_CACHE.clear()
    stage_timers: Dict[str, float] = {}
    pipeline_started = time.perf_counter()

    def _mark_stage(stage_name: str, started_at: float) -> None:
        stage_timers[stage_name] = round(time.perf_counter() - started_at, 4)

    resolved_evaluation_profile = _normalize_evaluation_profile(evaluation_profile)
    evaluation_candidate_labels = _candidate_labels_for_evaluation_profile(resolved_evaluation_profile)
    evaluation_target_pair_set = set(evaluation_target_pairs) if evaluation_target_pairs is not None else None
    section_priors, resolved_section_priors_path = _load_section_priors(cli.section_priors_file)
    rollout_config = {
        "pair_connection_mode": cli.pair_connection_mode,
        "low_signal_bundle_rescue": bool(getattr(cli, "low_signal_bundle_rescue", False)),
        "pair_embedding_mode": cli.pair_embedding_mode,
        "gliner2_label_input_mode": cli.gliner2_label_input_mode,
        "anchor_normalization_mode": cli.anchor_normalization_mode,
        "cluster_refine_by_pair_label": cli.cluster_refine_by_pair_label,
        "cluster_refine_min_pairs": cli.cluster_refine_min_pairs,
        "cluster_refine_semantic_subsplit": cli.cluster_refine_semantic_subsplit,
        "cluster_refine_semantic_distance": cli.cluster_refine_semantic_distance,
        "cluster_refine_llm_per_path_vote": cli.cluster_refine_llm_per_path_vote,
        "cluster_refine_path_subsplit": cli.cluster_refine_path_subsplit,
        "cluster_refine_path_subsplit_min_mass": cli.cluster_refine_path_subsplit_min_mass,
        "cluster_refine_path_subsplit_min_share": cli.cluster_refine_path_subsplit_min_share,
        "cluster_refine_path_subsplit_max_gap": cli.cluster_refine_path_subsplit_max_gap,
        "suppress_negative_clusters": cli.suppress_negative_clusters,
        "sentence_specificity_alpha": cli.sentence_specificity_alpha,
        "stage5_sentence_overflow_margin": cli.stage5_sentence_overflow_margin,
        "stage5_sentence_overflow_limit": cli.stage5_sentence_overflow_limit,
        "stage5_row_plateau_margin": cli.stage5_row_plateau_margin,
        "stage5_row_plateau_min_extra": cli.stage5_row_plateau_min_extra,
        "stage5_row_plateau_max_extra": cli.stage5_row_plateau_max_extra,
        "stage5_sentence_plateau_margin": cli.stage5_sentence_plateau_margin,
        "stage5_sentence_plateau_min_extra": cli.stage5_sentence_plateau_min_extra,
        "stage5_sentence_plateau_max_extra": cli.stage5_sentence_plateau_max_extra,
        "stage5_pair_plateau_margin": cli.stage5_pair_plateau_margin,
        "stage5_pair_plateau_min_extra": cli.stage5_pair_plateau_min_extra,
        "stage5_pair_plateau_max_extra": cli.stage5_pair_plateau_max_extra,
        "stage5_threshold_rescue_margin": cli.stage5_threshold_rescue_margin,
        "stage5_threshold_rescue_min_sentences": cli.stage5_threshold_rescue_min_sentences,
        "section_priors_file": cli.section_priors_file,
        "stage5_diag_row_top_k": cli.stage5_diag_row_top_k,
        "stage5_med_row_top_k": cli.stage5_med_row_top_k,
        "pair_filter_mode": cli.pair_filter_mode,
        "cluster_tail_mode": cli.cluster_tail_mode,
        "enable_pair_recovery_diagnostics": cli.enable_pair_recovery_diagnostics,
        "debug_recall_cascade": cli.debug_recall_cascade,
        "diagnostics_output_dir": cli.diagnostics_output_dir,
    }
    collect_pair_recovery_diagnostics = cli.enable_pair_recovery_diagnostics or cli.debug_recall_cascade

    n_diag = len(diag_rows)
    gt_relationships, gt_diag, gt_med, multi_pairs, gt_sanitize_stats = _sanitize_ground_truth_indices(
        gt_relationships,
        gt_diag,
        gt_med,
        multi_pairs,
        n_diag_rows=n_diag,
        n_med_rows=len(med_rows),
        n_sentences=len(sent_texts),
    )
    if any(gt_sanitize_stats.values()):
        print(
            "  GT sanitization: "
            f"dropped {gt_sanitize_stats['dropped_relationships']} invalid relationships, "
            f"{gt_sanitize_stats['dropped_multi_pairs']} invalid multi-pairs, "
            f"{gt_sanitize_stats['dropped_diag_rows']} invalid diagnosis grounding rows, "
            f"{gt_sanitize_stats['dropped_med_rows']} invalid medication grounding rows, "
            f"{gt_sanitize_stats['dropped_relationship_sentence_refs']} invalid relationship sentence refs, "
            f"{gt_sanitize_stats['dropped_row_grounding_sentence_refs']} invalid row-grounding sentence refs"
        )
    if resolved_section_priors_path is not None:
        print(
            f"  Loaded {len(section_priors)} section priors from {resolved_section_priors_path}"
        )

    print("\n-- Phase C: Joint Encoding ---------------------------------------")
    phase_c_started = time.perf_counter()
    pair_scores, raw_rows, raw_sentences, refined_rows, refined_sentences, _fwd_attn = joint_encode(
        model, diag_rows, med_rows, sent_texts
    )
    _mark_stage("phase_c_joint_encoding_sec", phase_c_started)
    expected_shape = (n_diag + len(med_rows), len(sent_texts))
    assert pair_scores.shape == expected_shape, \
        f"pair_scores shape mismatch: got {tuple(pair_scores.shape)}, expected {expected_shape}"

    print("\n-- Phase D: Join Path Extraction ---------------------------------")
    phase_d_started = time.perf_counter()
    gamma = (
        cli.threshold
        if cli.threshold is not None
        else compute_threshold(
            pair_scores,
            floor=model_args.get("join_path_threshold", 0.15),
            adaptive_cap=cli.adaptive_threshold_cap,
            adaptive_gap_limit=cli.adaptive_threshold_gap_limit,
            adaptive_force_legacy_max=cli.adaptive_threshold_force_legacy_max,
        )
    )
    inference_top_k = cli.stage5_top_k if cli.stage5_top_k is not None else 32
    inference_diag_row_top_k = cli.stage5_diag_row_top_k if cli.stage5_diag_row_top_k is not None else inference_top_k
    inference_med_row_top_k = cli.stage5_med_row_top_k if cli.stage5_med_row_top_k is not None else inference_top_k
    inference_sent_diag_top_k = cli.stage5_sent_diag_top_k
    inference_sent_med_top_k = cli.stage5_sent_med_top_k
    inference_max_pairs_per_sentence = cli.stage5_max_pairs_per_sentence
    inference_max_sentences_per_pair = cli.stage5_max_sentences_per_pair
    paths = extract_cross_table_join_paths(
        pair_scores, n_diag, diag_rows, med_rows, sent_texts, sent_meta, gamma,
        top_k=inference_top_k,
        diag_row_top_k=inference_diag_row_top_k,
        med_row_top_k=inference_med_row_top_k,
        row_plateau_margin=cli.stage5_row_plateau_margin,
        row_plateau_min_extra=cli.stage5_row_plateau_min_extra,
        row_plateau_max_extra=cli.stage5_row_plateau_max_extra,
        sent_diag_top_k=inference_sent_diag_top_k,
        sent_med_top_k=inference_sent_med_top_k,
        max_pairs_per_sentence=inference_max_pairs_per_sentence,
        max_sentences_per_pair=inference_max_sentences_per_pair,
        sentence_specificity_alpha=cli.sentence_specificity_alpha,
        section_priors=section_priors,
        sentence_overflow_margin=cli.stage5_sentence_overflow_margin,
        sentence_overflow_limit=cli.stage5_sentence_overflow_limit,
        sentence_plateau_margin=cli.stage5_sentence_plateau_margin,
        sentence_plateau_min_extra=cli.stage5_sentence_plateau_min_extra,
        sentence_plateau_max_extra=cli.stage5_sentence_plateau_max_extra,
        stopcue_diag_sentence_top_k=cli.stage5_stopcue_diag_sentence_top_k,
        pair_plateau_margin=cli.stage5_pair_plateau_margin,
        pair_plateau_min_extra=cli.stage5_pair_plateau_min_extra,
        pair_plateau_max_extra=cli.stage5_pair_plateau_max_extra,
        threshold_rescue_margin=cli.stage5_threshold_rescue_margin,
        threshold_rescue_min_sentences=cli.stage5_threshold_rescue_min_sentences,
        diag_row_sibling_rescue_margin=cli.stage5_diag_row_sibling_rescue_margin,
        med_row_stopcue_rescue_margin=cli.stage5_med_row_stopcue_rescue_margin,
    )
    _mark_stage("phase_d_join_path_extraction_sec", phase_d_started)
    pair_recovery_diagnostics: Dict[str, object] = {}
    if collect_pair_recovery_diagnostics:
        pair_recovery_diagnostics["after_stage5_extraction"] = _pair_recovery_stage_snapshot(paths)
    pair_filter_started = time.perf_counter()
    if paths and not cli.no_pair_filter and cli.pair_filter_mode != "off":
        paths, pair_filter_stats = filter_candidate_pairs(
            paths,
            gamma=gamma,
            diag_top_k=cli.pair_filter_diag_top_k,
            med_top_k=cli.pair_filter_med_top_k,
            score_margin=cli.pair_filter_margin,
            hub_fanout=cli.pair_filter_hub_fanout,
            mode=cli.pair_filter_mode,
            collect_details=collect_pair_recovery_diagnostics,
        )
    else:
        pair_filter_stats = {
            "enabled": not cli.no_pair_filter,
            "mode": cli.pair_filter_mode,
            "n_pairs_before": len({(p["diag_row_idx"], p["med_row_idx"]) for p in paths}),
            "n_pairs_after": len({(p["diag_row_idx"], p["med_row_idx"]) for p in paths}),
            "dropped_pairs": 0,
            "dropped_paths": 0,
            "reason": "disabled" if cli.no_pair_filter or cli.pair_filter_mode == "off" else "no_paths",
        }
    _mark_stage("phase_d_pair_filter_sec", pair_filter_started)
    if collect_pair_recovery_diagnostics:
        pair_recovery_diagnostics["after_pair_filter"] = _pair_recovery_stage_snapshot(paths)

    cross_encoder_rerank_summary: Dict[str, object] = {}
    cross_encoder_started = time.perf_counter()
    if getattr(cli, "use_cross_encoder", False) and paths:
        print("\n-- Phase D.5: Cross-Encoder Per-Pair Sentence Rerank (Option C) --")
        try:
            cross_encoder_rerank_summary = rerank_pair_sentences_with_cross_encoder(
                paths=paths,
                diag_rows=diag_rows,
                med_rows=med_rows,
                sent_texts=sent_texts,
                sent_meta=sent_meta,
                gt_relationships=gt_relationships,
                model_name=cli.cross_encoder_model,
                device=(cli.cross_encoder_device or None),
                batch_size=cli.cross_encoder_batch_size,
                max_length=cli.cross_encoder_max_length,
                fp16=not cli.cross_encoder_no_fp16,
                section_prefix=not cli.cross_encoder_no_section_prefix,
                normalize=not cli.cross_encoder_no_normalize,
            )
        except Exception as exc:  # rerank failure must never break the main pipeline
            print(f"  Cross-encoder rerank failed: {exc!r}  - continuing with LOKI-only ordering")
            cross_encoder_rerank_summary = {"error": repr(exc)}
    _mark_stage("phase_d5_cross_encoder_rerank_sec", cross_encoder_started)

    ce_pair_filter_stats: Dict[str, object] = {"enabled": False, "mode": "off"}
    ce_pair_filter_mode = (getattr(cli, "ce_pair_filter_mode", "off") or "off").strip().lower()
    ce_pair_filter_started = time.perf_counter()
    if (
        ce_pair_filter_mode != "off"
        and getattr(cli, "use_cross_encoder", False)
        and paths
        and not cross_encoder_rerank_summary.get("error")
    ):
        print("\n-- Phase D.6: Cross-Encoder Pair-Level Filter (Option D) --------")
        paths, ce_pair_filter_stats = filter_pairs_by_cross_encoder(
            paths,
            mode=ce_pair_filter_mode,
            threshold=float(getattr(cli, "ce_pair_filter_threshold", 0.05)),
            quantile=float(getattr(cli, "ce_pair_filter_quantile", 0.25)),
            collect_details=collect_pair_recovery_diagnostics,
        )
        if collect_pair_recovery_diagnostics:
            pair_recovery_diagnostics["after_ce_pair_filter"] = _pair_recovery_stage_snapshot(paths)
    _mark_stage("phase_d6_cross_encoder_pair_filter_sec", ce_pair_filter_started)

    print("\n-- Phase E: Semantic Materialization -----------------------------")
    phase_e_started = time.perf_counter()
    clustered_paths: Optional[List[Dict]] = None
    cluster_name_map: Dict[int, str] = {}
    cluster_label_details: Dict[int, Dict[str, object]] = {}
    cluster_label_backend = (cli.cluster_label_backend or DEFAULT_CLUSTER_LABEL_BACKEND).strip().lower()
    gliner2_label_input_mode = (cli.gliner2_label_input_mode or DEFAULT_GLINER2_LABEL_INPUT_MODE).strip().lower()
    if gliner2_label_input_mode not in {"sentence_evidence", "semantic_signature"}:
        raise ValueError(f"Unsupported GLiNER2 label input mode: {cli.gliner2_label_input_mode}")
    if cluster_label_backend == "keyword" and gliner2_label_input_mode != DEFAULT_GLINER2_LABEL_INPUT_MODE:
        print(
            "  Warning: --gliner2_label_input_mode is ignored when "
            "--cluster_label_backend keyword is selected."
        )
    kept_cluster_ids: set[int] = set()
    cluster_tail_filter_stats = {
        "enabled": not cli.no_cluster_tail_filter,
        "dropped_pairs": 0,
        "dropped_paths": 0,
        "clusters_touched": 0,
        "keep_rank": cli.cluster_tail_keep_rank,
        "score_margin": cli.cluster_tail_margin,
        "reason": "no_paths",
    }
    cluster_pair_label_refinement_stats = {
        "enabled": bool(cli.cluster_refine_by_pair_label),
        "backend": cluster_label_backend,
        "min_cluster_pairs": int(cli.cluster_refine_min_pairs),
        "llm_per_path_vote": bool(cli.cluster_refine_llm_per_path_vote),
        "semantic_subsplit": bool(cli.cluster_refine_semantic_subsplit),
        "semantic_distance_threshold": round(max(float(cli.cluster_refine_semantic_distance), 0.0), 4),
        "path_subsplit": bool(cli.cluster_refine_path_subsplit),
        "path_subsplit_min_score_mass": round(max(float(cli.cluster_refine_path_subsplit_min_mass), 0.0), 4),
        "path_subsplit_min_score_share": round(min(max(float(cli.cluster_refine_path_subsplit_min_share), 0.0), 1.0), 4),
        "path_subsplit_max_dominant_gap": round(max(float(cli.cluster_refine_path_subsplit_max_gap), 0.0), 4),
        "parent_clusters_considered": 0,
        "parent_clusters_split": 0,
        "semantic_parent_clusters_split": 0,
        "path_parent_clusters_split": 0,
        "child_clusters_added": 0,
        "semantic_child_clusters_added": 0,
        "path_child_clusters_added": 0,
        "path_split_candidates": 0,
        "path_split_rejected": 0,
        "path_split_folded_paths": 0,
        "pairs_scored": 0,
        "pairs_reassigned": 0,
        "paths_reassigned": 0,
        "reason": "no_paths",
    }
    negative_cluster_suppression_stats = {
        "enabled": bool(cli.suppress_negative_clusters),
        "keep_annotated_negative_clusters": True,
        "negative_clusters_considered": 0,
        "negative_clusters_suppressed": 0,
        "negative_clusters_kept": 0,
        "annotated_negative_clusters_kept": 0,
        "suppressed_pairs": 0,
        "suppressed_paths": 0,
        "reason": "no_paths",
    }
    cluster_connection_filter_stats = {
        "mode": cli.pair_connection_mode,
        "low_signal_bundle_rescue": bool(getattr(cli, "low_signal_bundle_rescue", False)),
        "n_clusters_before": 0,
        "n_clusters_after": 0,
        "dropped_clusters": 0,
        "threshold": round(float(gamma), 4),
        "sentence_specificity_alpha": round(max(float(cli.sentence_specificity_alpha), 0.0), 4),
        "section_priors_file": str(resolved_section_priors_path) if resolved_section_priors_path is not None else "",
        "reason": "no_paths",
    }
    # Initialize silhouette to ensure it's always available below
    cluster_silhouette = None
    if paths:
        clustering_started = time.perf_counter()
        labels, n_clusters = cluster_mediating_sentences(
            paths,
            refined_sentences,
            refined_rows=refined_rows,
            n_diag=n_diag,
            sentence_encoder=getattr(model, "sentence_encoder", None),
            embedding_mode=cli.pair_embedding_mode,
            hdbscan_min_cluster_size=cli.hdbscan_min_cluster_size,
        )
        _mark_stage("phase_e_hdbscan_clustering_sec", clustering_started)
        print(f"  {n_clusters} relationship clusters discovered")

        for path, lbl in zip(paths, labels):
            path["raw_cluster_id"] = int(lbl)

        clusters: Dict[int, List[Dict]] = defaultdict(list)
        for path, lbl in zip(paths, labels):
            clusters[int(lbl)].append(path)

        cluster_tail_started = time.perf_counter()
        if not cli.no_cluster_tail_filter:
            paths, labels, cluster_tail_filter_stats = filter_cluster_pair_tails(
                paths,
                labels,
                keep_rank=cli.cluster_tail_keep_rank,
                score_margin=cli.cluster_tail_margin,
                mode=cli.cluster_tail_mode,
                collect_details=collect_pair_recovery_diagnostics,
                adaptive_lambda=cli.cluster_tail_adaptive_lambda,
                adaptive_percentile=cli.cluster_tail_adaptive_percentile,
                rescue_unique_evidence=not cli.no_rescue_unique_evidence,
            )
            clusters = defaultdict(list)
            for path, lbl in zip(paths, labels):
                clusters[int(lbl)].append(path)
        else:
            cluster_tail_filter_stats["reason"] = "disabled"
        _mark_stage("phase_e_cluster_tail_filter_sec", cluster_tail_started)
        if collect_pair_recovery_diagnostics:
            pair_recovery_diagnostics["after_cluster_tail_filter"] = _pair_recovery_stage_snapshot(paths, labels=labels)

        # Phase 2: Shared-pair must-link pre-merge
        shared_merge_started = time.perf_counter()
        if not cli.no_shared_pair_merge:
            clusters, labels, _shared_merge_map = merge_clusters_by_shared_pairs(clusters, labels, paths)
            for path, lbl in zip(paths, labels):
                path["raw_cluster_id"] = int(lbl)
        _mark_stage("phase_e_shared_pair_merge_sec", shared_merge_started)

        # Phase 6: Hierarchical meta-clustering (optional)
        meta_clustering_started = time.perf_counter()
        _run_meta = cli.enable_meta_clustering or (cli.max_clusters > 0 and len(clusters) > cli.max_clusters)
        _meta_n = cli.meta_cluster_n if cli.enable_meta_clustering else (cli.max_clusters if cli.max_clusters > 0 else 0)
        if _run_meta:
            clusters, labels, _meta_stats = meta_cluster_hierarchical(
                clusters, labels, paths,
                n_meta_clusters=_meta_n,
                alpha_sent=cli.meta_cluster_alpha_sent,
                alpha_pair=cli.meta_cluster_alpha_pair,
            )
            for path, lbl in zip(paths, labels):
                path["raw_cluster_id"] = int(lbl)
        _mark_stage("phase_e_meta_clustering_sec", meta_clustering_started)

        print(f"  Cluster labeling backend: {cluster_label_backend}")
        if cluster_label_backend == "gliner2":
            print(f"  GLiNER2 label input mode: {gliner2_label_input_mode}")
        if cluster_label_backend == "lmstudio":
            print(f"  LLM model: {cli.llm_model}  url: {cli.llm_base_url}")
        if getattr(cli, "llm_path_vote", False) and cli.llm_agglomerative:
            print("  [warn] --llm_path_vote is set; --llm_agglomerative will be ignored.")
        if getattr(cli, "llm_no_hdbscan", False) and cli.llm_agglomerative:
            print("  [warn] --llm_no_hdbscan is set; HDBSCAN grouping and --llm_agglomerative will be bypassed.")
        _agglom_vis_path = (
            str(VIS_DIR / f"agglom_recluster_{ADMISSION_ID}.png")
            if cluster_label_backend == "lmstudio" and cli.llm_agglomerative
               and not getattr(cli, "llm_path_vote", False)
               and not getattr(cli, "llm_no_hdbscan", False)
               and not cli.skip_visualizations
            else None
        )
        _no_hdbscan_vis_path = (
            str(VIS_DIR / f"llm_vs_hdbscan_{ADMISSION_ID}.png")
            if cluster_label_backend == "lmstudio" and getattr(cli, "llm_no_hdbscan", False)
               and not cli.skip_visualizations
            else None
        )
        use_single_pass_opt = (
            cluster_label_backend == "lmstudio"
            and getattr(cli, "cluster_refine_by_pair_label", False)
            and not getattr(cli, "llm_no_hdbscan", False)
            and not getattr(cli, "llm_agglomerative", False)
        )
        if use_single_pass_opt:
            print("  Pair-label refinement: merging Round 1 and Round 2 into a single path-level pass.")
            cluster_name_map = None
            cluster_label_details = None
            _mark_stage("phase_e_cluster_labeling_sec", time.perf_counter())
        else:
            cluster_labeling_started = time.perf_counter()
            cluster_name_map, cluster_label_details = label_clusters(
                clusters,
                backend=cluster_label_backend,
                gliner2_model=cli.gliner2_model,
                gliner2_batch_size=cli.gliner2_batch_size,
                gliner2_threshold=cli.gliner2_threshold,
                gliner2_max_len=cli.gliner2_max_len,
                anchor_normalization_mode=cli.anchor_normalization_mode,
                gliner2_label_input_mode=gliner2_label_input_mode,
                per_sentence_vote=cli.gliner2_per_sentence_vote,
                all_paths=paths,
                hub_fanout_threshold=cli.gliner2_hub_fanout_threshold,
                max_pool_sentences=cli.gliner2_max_pool_sentences,
                llm_base_url=cli.llm_base_url,
                llm_model=cli.llm_model,
                llm_temperature=cli.llm_temperature,
                llm_timeout_secs=cli.llm_timeout,
                llm_max_evidence_sents=cli.llm_max_evidence_sents,
                llm_per_path_vote=cli.llm_per_path_vote,
                llm_agglomerative=cli.llm_agglomerative,
                llm_agglom_distance=cli.llm_agglom_distance,
                gt_relationships=gt_relationships,
                encoder_model=model,
                llm_agglom_vis_path=_agglom_vis_path,
                llm_path_vote=cli.llm_path_vote,
                llm_no_hdbscan=cli.llm_no_hdbscan,
                llm_no_hdbscan_vis_path=_no_hdbscan_vis_path,
                llm_agglom_encoder=cli.llm_agglom_encoder,
                candidate_labels=evaluation_candidate_labels,
            )
            _mark_stage("phase_e_cluster_labeling_sec", cluster_labeling_started)

        using_pair_identity_mode = (
            cluster_label_backend == "lmstudio"
            and getattr(cli, "llm_no_hdbscan", False)
        )
        pair_label_refine_started = time.perf_counter()
        if using_pair_identity_mode:
            cluster_pair_label_refinement_stats["reason"] = "llm_no_hdbscan"
        elif not getattr(cli, "cluster_refine_by_pair_label", False):
            cluster_pair_label_refinement_stats["reason"] = "disabled"
        else:
            clusters, labels, cluster_name_map, cluster_label_details, cluster_pair_label_refinement_stats = _refine_clusters_by_pair_labels(
                paths,
                clusters,
                labels,
                backend=cluster_label_backend,
                current_cluster_name_map=cluster_name_map,
                current_cluster_label_details=cluster_label_details,
                min_cluster_pairs=cli.cluster_refine_min_pairs,
                refined_sentences=refined_sentences,
                refined_rows=refined_rows,
                n_diag=n_diag,
                pair_embedding_mode=cli.pair_embedding_mode,
                semantic_subsplit=cli.cluster_refine_semantic_subsplit,
                semantic_distance_threshold=cli.cluster_refine_semantic_distance,
                path_subsplit=cli.cluster_refine_path_subsplit,
                path_subsplit_min_score_mass=cli.cluster_refine_path_subsplit_min_mass,
                path_subsplit_min_score_share=cli.cluster_refine_path_subsplit_min_share,
                path_subsplit_max_dominant_gap=cli.cluster_refine_path_subsplit_max_gap,
                gliner2_model=cli.gliner2_model,
                gliner2_batch_size=cli.gliner2_batch_size,
                gliner2_threshold=cli.gliner2_threshold,
                gliner2_max_len=cli.gliner2_max_len,
                anchor_normalization_mode=cli.anchor_normalization_mode,
                gliner2_label_input_mode=gliner2_label_input_mode,
                per_sentence_vote=cli.gliner2_per_sentence_vote,
                all_paths=paths,
                hub_fanout_threshold=cli.gliner2_hub_fanout_threshold,
                max_pool_sentences=cli.gliner2_max_pool_sentences,
                llm_base_url=cli.llm_base_url,
                llm_model=cli.llm_model,
                llm_temperature=cli.llm_temperature,
                llm_timeout_secs=cli.llm_timeout,
                llm_max_evidence_sents=cli.llm_max_evidence_sents,
                llm_per_path_vote=cli.cluster_refine_llm_per_path_vote,
                gt_relationships=gt_relationships,
                encoder_model=model,
                candidate_labels=evaluation_candidate_labels,
            )
        _mark_stage("phase_e_pair_label_refinement_sec", pair_label_refine_started)
        if collect_pair_recovery_diagnostics:
            pair_recovery_diagnostics["after_pair_label_refinement"] = _pair_recovery_stage_snapshot(
                paths,
                labels=labels,
                cluster_name_map=cluster_name_map,
                cluster_label_details=cluster_label_details,
            )

        negative_suppression_started = time.perf_counter()
        if using_pair_identity_mode:
            negative_cluster_suppression_stats["reason"] = "llm_no_hdbscan"
        elif cluster_label_backend != "lmstudio":
            negative_cluster_suppression_stats["reason"] = "non_lmstudio_backend"
        elif not getattr(cli, "suppress_negative_clusters", True):
            negative_cluster_suppression_stats["reason"] = "disabled"
        else:
            paths, clusters, labels, cluster_name_map, cluster_label_details, negative_cluster_suppression_stats = _suppress_negative_labeled_clusters(
                paths,
                clusters,
                labels,
                current_cluster_name_map=cluster_name_map,
                current_cluster_label_details=cluster_label_details,
                negative_pairs=negative_pairs,
                keep_annotated_negative_clusters=True,
                cluster_pair_label_refinement_stats=cluster_pair_label_refinement_stats,
                candidate_labels=evaluation_candidate_labels,
                enable_refinement_child_rescue=bool(getattr(cli, "low_signal_bundle_rescue", False)),
            )
        _mark_stage("phase_e_negative_cluster_suppression_sec", negative_suppression_started)
        if collect_pair_recovery_diagnostics:
            pair_recovery_diagnostics["after_negative_cluster_suppression"] = _pair_recovery_stage_snapshot(
                paths,
                labels=labels,
                cluster_name_map=cluster_name_map,
                cluster_label_details=cluster_label_details,
            )

        # For --llm_no_hdbscan, label_pairs_with_llm_no_hdbscan() has already mutated
        # path["raw_cluster_id"] on every path to synthetic pair-cluster IDs.  Rebuild
        # `clusters` and `labels` from those mutated IDs so that the downstream connection
        # filter, kept_cluster_ids, and relationship assignment all use consistent synthetic IDs.
        if using_pair_identity_mode:
            _syn_clusters: Dict[int, List[Dict]] = {}
            _new_labels: List[int] = []
            _kept_paths: List[Dict] = []
            for path in paths:
                _syn_cid = int(path["raw_cluster_id"])
                if _syn_cid == -1:  # NEGATIVE-labeled pair - drop from materialized output
                    continue
                _syn_clusters.setdefault(_syn_cid, []).append(path)
                _new_labels.append(_syn_cid)
                _kept_paths.append(path)
            paths = _kept_paths
            clusters = _syn_clusters
            labels = np.asarray(_new_labels, dtype=int)
        else:
            for path, lbl in zip(paths, labels):
                path["raw_cluster_id"] = int(lbl)

        clustered_paths = list(paths)

        cluster_connection_signals = {
            cid: _cluster_connection_filter_signal(
                cpaths,
                gamma=gamma,
                mode=cli.pair_connection_mode,
                sentence_specificity_alpha=cli.sentence_specificity_alpha,
                section_priors=section_priors,
            )
            for cid, cpaths in sorted(clusters.items())
        }
        low_signal_bundle_rescue_stats: Dict[str, object] = {
            "enabled": bool(getattr(cli, "low_signal_bundle_rescue", False)),
            "candidate_singletons": 0,
            "vetoed_clusters": [],
            "bundle_groups": [],
            "rescued_cluster_ids": [],
            "rescued_member_clusters": [],
            "reason": "disabled" if not getattr(cli, "low_signal_bundle_rescue", False) else "no_dropped_clusters",
        }
        bundle_rescue_started = time.perf_counter()
        if getattr(cli, "low_signal_bundle_rescue", False):
            labels, clusters, cluster_name_map, cluster_label_details, low_signal_bundle_rescue_stats = _rescue_low_signal_cluster_bundles(
                paths,
                labels,
                clusters,
                cluster_name_map,
                cluster_label_details,
                cluster_connection_signals,
                cluster_pair_label_refinement_stats=cluster_pair_label_refinement_stats,
                candidate_labels=evaluation_candidate_labels,
            )
            if list(low_signal_bundle_rescue_stats.get("rescued_cluster_ids") or []):
                rescued_cluster_ids = list(low_signal_bundle_rescue_stats.get("rescued_cluster_ids") or [])
                rescued_member_clusters = list(low_signal_bundle_rescue_stats.get("rescued_member_clusters") or [])
                print(
                    f"  Low-signal bundle rescue merged {len(rescued_member_clusters)} dropped singleton clusters "
                    f"into {len(rescued_cluster_ids)} rescued clusters: {rescued_cluster_ids}"
                )
                cluster_connection_signals = {
                    cid: _cluster_connection_filter_signal(
                        cpaths,
                        gamma=gamma,
                        mode=cli.pair_connection_mode,
                        sentence_specificity_alpha=cli.sentence_specificity_alpha,
                        section_priors=section_priors,
                    )
                    for cid, cpaths in sorted(clusters.items())
                }
            _mark_stage("phase_e_low_signal_bundle_rescue_sec", bundle_rescue_started)
        kept_cluster_ids = {
            cid for cid, signal in cluster_connection_signals.items()
            if bool(signal.get("keep", False))
        }
        dropped_cluster_ids = sorted(set(clusters) - kept_cluster_ids)
        cluster_connection_filter_stats.update({
            "n_clusters_before": len(clusters),
            "n_clusters_after": len(kept_cluster_ids),
            "dropped_clusters": len(dropped_cluster_ids),
            "bundle_rescue_candidate_singletons": int(low_signal_bundle_rescue_stats.get("candidate_singletons", 0) or 0),
            "bundle_rescue_vetoed_clusters": list(low_signal_bundle_rescue_stats.get("vetoed_clusters") or []),
            "bundle_rescue_groups": list(low_signal_bundle_rescue_stats.get("bundle_groups") or []),
            "bundle_rescue_cluster_ids": list(low_signal_bundle_rescue_stats.get("rescued_cluster_ids") or []),
            "bundle_rescue_member_clusters": list(low_signal_bundle_rescue_stats.get("rescued_member_clusters") or []),
            "bundle_rescue_reason": str(low_signal_bundle_rescue_stats.get("reason", "")),
            "reason": "applied",
        })
        if dropped_cluster_ids:
            drop_reason = "low-support clusters under support-weighted connection scoring" if cli.pair_connection_mode == "support_weighted" else "low-signal clusters with no lexical relation cues"
            print(
                f"  Dropping {len(dropped_cluster_ids)} {drop_reason}: "
                f"{dropped_cluster_ids}"
            )
            filtered_paths: List[Dict] = []
            filtered_labels: List[int] = []
            for path, lbl in zip(paths, labels):
                if int(lbl) in kept_cluster_ids:
                    filtered_paths.append(path)
                    filtered_labels.append(int(lbl))
            paths = filtered_paths
            labels = np.asarray(filtered_labels, dtype=int)
        print(f"  Retained {len(set(labels))} clusters after low-signal filtering")
        if collect_pair_recovery_diagnostics:
            pair_recovery_diagnostics["after_low_signal_cluster_filter"] = _pair_recovery_stage_snapshot(paths, labels=labels)
            pair_recovery_diagnostics["low_signal_bundle_rescue"] = low_signal_bundle_rescue_stats
            pair_recovery_diagnostics["cluster_signal_filter"] = [
                {
                    "cluster_id": cid,
                    **cluster_connection_signals[cid],
                }
                for cid in sorted(cluster_connection_signals)
            ]

        for path, lbl in zip(paths, labels):
            path["cluster_id"] = int(lbl)
            path["relationship"] = cluster_name_map.get(int(lbl), f"cluster_{lbl}")

        paths_for_eval = _filter_paths_for_target_pairs(paths, evaluation_target_pair_set) or []
        # --- Compute cluster silhouette (pair-level) for diagnostic reporting ---
        # In llm_no_hdbscan mode the final cluster_id is one singleton per pair, so
        # use the preserved structural HDBSCAN assignment instead.
        cluster_silhouette_started = time.perf_counter()
        cluster_silhouette = _compute_pair_level_cluster_silhouette(
            paths_for_eval,
            refined_sentences,
            refined_rows=refined_rows,
            n_diag=n_diag,
            sentence_encoder=getattr(model, "sentence_encoder", None),
            embedding_mode=cli.pair_embedding_mode,
            cluster_key="cluster_id",
        )
        if cluster_silhouette is None and getattr(cli, "llm_no_hdbscan", False):
            cluster_silhouette = _compute_pair_level_cluster_silhouette(
                paths_for_eval,
                refined_sentences,
                refined_rows=refined_rows,
                n_diag=n_diag,
                sentence_encoder=getattr(model, "sentence_encoder", None),
                embedding_mode=cli.pair_embedding_mode,
                cluster_key="hdbscan_cluster_id",
            )
        _mark_stage("phase_e_cluster_silhouette_sec", cluster_silhouette_started)
    else:
        _mark_stage("phase_e_hdbscan_clustering_sec", phase_e_started)
        stage_timers.setdefault("phase_e_cluster_tail_filter_sec", 0.0)
        stage_timers.setdefault("phase_e_shared_pair_merge_sec", 0.0)
        stage_timers.setdefault("phase_e_meta_clustering_sec", 0.0)
        stage_timers.setdefault("phase_e_cluster_labeling_sec", 0.0)
        stage_timers.setdefault("phase_e_pair_label_refinement_sec", 0.0)
        stage_timers.setdefault("phase_e_negative_cluster_suppression_sec", 0.0)
        stage_timers.setdefault("phase_e_low_signal_bundle_rescue_sec", 0.0)
        stage_timers.setdefault("phase_e_cluster_silhouette_sec", 0.0)
        print("  No paths found - nothing to cluster.")
    _mark_stage("phase_e_semantic_materialization_sec", phase_e_started)

    print("\n-- Phase F: Evaluation -------------------------------------------")
    phase_f_started = time.perf_counter()
    eval_paths = _filter_paths_for_target_pairs(paths, evaluation_target_pair_set) or []
    eval_clustered_paths = _filter_paths_for_target_pairs(clustered_paths, evaluation_target_pair_set)
    gt_triples, gt_pairs = build_gt_path_set(gt_relationships)
    metrics = evaluate(
        eval_paths,
        gt_relationships,
        gt_diag,
        gt_med,
        gt_triples,
        gt_pairs,
        multi_pairs,
        raw_cluster_paths=eval_clustered_paths,
        show_typed_metrics=cli.show_typed_metrics,
    )
    _mark_stage("phase_f_evaluation_sec", phase_f_started)
    # Attach computed silhouette (may be None)
    metrics["cluster_silhouette"] = cluster_silhouette
    metrics["stage5_config"] = {
        "top_k": inference_top_k,
        "sent_diag_top_k": inference_sent_diag_top_k,
        "sent_med_top_k": inference_sent_med_top_k,
        "max_pairs_per_sentence": inference_max_pairs_per_sentence,
        "max_sentences_per_pair": inference_max_sentences_per_pair,
        "gamma": round(float(gamma), 4),
        "diag_row_top_k": inference_diag_row_top_k,
        "med_row_top_k": inference_med_row_top_k,
        "row_plateau_margin": round(max(float(cli.stage5_row_plateau_margin), 0.0), 4),
        "row_plateau_min_extra": max(int(cli.stage5_row_plateau_min_extra), 0),
        "row_plateau_max_extra": max(int(cli.stage5_row_plateau_max_extra), 0),
        "sentence_specificity_alpha": round(max(float(cli.sentence_specificity_alpha), 0.0), 4),
        "sentence_overflow_margin": round(max(float(cli.stage5_sentence_overflow_margin), 0.0), 4),
        "sentence_overflow_limit": max(int(cli.stage5_sentence_overflow_limit), 0),
        "sentence_plateau_margin": round(max(float(cli.stage5_sentence_plateau_margin), 0.0), 4),
        "sentence_plateau_min_extra": max(int(cli.stage5_sentence_plateau_min_extra), 0),
        "sentence_plateau_max_extra": max(int(cli.stage5_sentence_plateau_max_extra), 0),
        "stopcue_diag_sentence_top_k": max(int(cli.stage5_stopcue_diag_sentence_top_k), 0),
        "pair_plateau_margin": round(max(float(cli.stage5_pair_plateau_margin), 0.0), 4),
        "pair_plateau_min_extra": max(int(cli.stage5_pair_plateau_min_extra), 0),
        "pair_plateau_max_extra": max(int(cli.stage5_pair_plateau_max_extra), 0),
        "threshold_rescue_margin": round(max(float(cli.stage5_threshold_rescue_margin), 0.0), 4),
        "threshold_rescue_min_sentences": max(int(cli.stage5_threshold_rescue_min_sentences), 2),
        "diag_row_sibling_rescue_margin": round(max(float(cli.stage5_diag_row_sibling_rescue_margin), 0.0), 4),
        "med_row_stopcue_rescue_margin": round(max(float(cli.stage5_med_row_stopcue_rescue_margin), 0.0), 4),
        "section_priors_file": str(resolved_section_priors_path) if resolved_section_priors_path is not None else "",
    }
    metrics["pair_filter"] = pair_filter_stats
    metrics["ce_pair_filter"] = ce_pair_filter_stats
    metrics["cluster_tail_filter"] = cluster_tail_filter_stats
    metrics["cluster_pair_label_refinement"] = cluster_pair_label_refinement_stats
    metrics["negative_cluster_suppression"] = negative_cluster_suppression_stats
    metrics["cluster_connection_filter"] = cluster_connection_filter_stats
    metrics["rollout_config"] = rollout_config
    metrics["evaluation_profile"] = resolved_evaluation_profile
    metrics["evaluation_target_pair_count"] = len(evaluation_target_pair_set) if evaluation_target_pair_set is not None else None
    metrics["evaluation_target_labels"] = sorted(AE_DIS_CLEAN_LABELS) if resolved_evaluation_profile == EVALUATION_PROFILE_AE_DIS_CLEAN else None
    _mark_stage("pipeline_total_sec", pipeline_started)
    metrics["stage_timers"] = dict(sorted(stage_timers.items()))
    print("  Stage timers (sec):")
    for stage_name, elapsed in metrics["stage_timers"].items():
        print(f"    {stage_name}: {elapsed:.4f}")
    if collect_pair_recovery_diagnostics:
        gt_pair_recovery = _build_gt_pair_recovery_diagnostics(
            gt_relationships,
            pair_scores,
            n_diag=n_diag,
            sent_texts=sent_texts,
            gamma=float(gamma),
            diag_row_top_k=inference_diag_row_top_k,
            med_row_top_k=inference_med_row_top_k,
            row_plateau_margin=float(cli.stage5_row_plateau_margin),
            row_plateau_min_extra=int(cli.stage5_row_plateau_min_extra),
            row_plateau_max_extra=int(cli.stage5_row_plateau_max_extra),
            sent_diag_top_k=inference_sent_diag_top_k,
            sent_med_top_k=inference_sent_med_top_k,
            max_pairs_per_sentence=inference_max_pairs_per_sentence,
            sent_meta=sent_meta,
            sentence_specificity_alpha=float(cli.sentence_specificity_alpha),
            section_priors=section_priors,
            sentence_overflow_margin=float(cli.stage5_sentence_overflow_margin),
            sentence_overflow_limit=int(cli.stage5_sentence_overflow_limit),
            sentence_plateau_margin=float(cli.stage5_sentence_plateau_margin),
            sentence_plateau_min_extra=int(cli.stage5_sentence_plateau_min_extra),
            sentence_plateau_max_extra=int(cli.stage5_sentence_plateau_max_extra),
            pair_plateau_margin=float(cli.stage5_pair_plateau_margin),
            pair_plateau_min_extra=int(cli.stage5_pair_plateau_min_extra),
            pair_plateau_max_extra=int(cli.stage5_pair_plateau_max_extra),
            pair_recovery_diagnostics=pair_recovery_diagnostics,
            pair_filter_stats=pair_filter_stats,
            ce_pair_filter_stats=ce_pair_filter_stats,
            cluster_tail_filter_stats=cluster_tail_filter_stats,
        )
        pair_recovery_diagnostics["gt_pair_recovery"] = gt_pair_recovery
        metrics["pair_recovery_diagnostics_summary"] = {
            stage_name: {
                "n_pairs": int(stage_snapshot.get("n_pairs", 0)),
                "n_paths": int(stage_snapshot.get("n_paths", 0)),
            }
            for stage_name, stage_snapshot in pair_recovery_diagnostics.items()
            if isinstance(stage_snapshot, dict) and "n_pairs" in stage_snapshot
        }
        metrics["gt_pair_failure_summary"] = gt_pair_recovery.get("summary", {})
        metrics["gt_pair_failure_report"] = _summarize_gt_pair_failures(gt_pair_recovery)
        if metrics["gt_pair_failure_report"] is not None:
            _print_gt_pair_failure_report(metrics["gt_pair_failure_report"])
    metrics["pair_average_precision"] = compute_pair_average_precision(pair_scores, n_diag, gt_pairs)
    metrics["evaluation_pred_path_count"] = len(eval_paths)
    metrics["n_final_clusters"] = len({int(path.get("cluster_id", -1)) for path in eval_paths}) if eval_paths else 0
    metrics["cluster_label_backend"] = cluster_label_backend if eval_paths else None
    metrics["gliner2_label_input_mode"] = (
        gliner2_label_input_mode
        if eval_paths and cluster_label_backend == "gliner2"
        else None
    )

    if cli.debug_recall_cascade and pair_recovery_diagnostics:
        print("  Pair-recovery cascade diagnostics:")
        for stage_name in (
            "after_stage5_extraction",
            "after_pair_filter",
            "after_cluster_tail_filter",
            "after_pair_label_refinement",
            "after_negative_cluster_suppression",
            "after_low_signal_cluster_filter",
        ):
            stage_snapshot = pair_recovery_diagnostics.get(stage_name)
            if not isinstance(stage_snapshot, dict):
                continue
            print(
                f"    {stage_name}: pairs={int(stage_snapshot.get('n_pairs', 0))}, "
                f"paths={int(stage_snapshot.get('n_paths', 0))}"
            )
        gt_pair_summary = pair_recovery_diagnostics.get("gt_pair_recovery", {}).get("summary", {})
        by_stage = gt_pair_summary.get("by_stage", {}) if isinstance(gt_pair_summary, dict) else {}
        if by_stage:
            stage_text = ", ".join(f"{stage}={count}" for stage, count in sorted(by_stage.items()))
            print(f"    gt_pair_failure_stages: {stage_text}")

    return {
        "n_diag": n_diag,
        "gt_relationships": gt_relationships,
        "pair_scores": pair_scores,
        "raw_rows": raw_rows,
        "raw_sentences": raw_sentences,
        "refined_rows": refined_rows,
        "refined_sentences": refined_sentences,
        "paths": paths,
        "clustered_paths": clustered_paths,
        "cluster_name_map": cluster_name_map,
        "cluster_label_details": cluster_label_details,
        "cluster_label_backend": cluster_label_backend if paths else None,
        "cluster_label_input_mode": gliner2_label_input_mode if (paths and cluster_label_backend == "gliner2") else None,
        "kept_cluster_ids": kept_cluster_ids,
        "pair_recovery_diagnostics": pair_recovery_diagnostics,
        "cross_encoder_rerank_summary": cross_encoder_rerank_summary,
        "ce_pair_filter_stats": ce_pair_filter_stats,
        "metrics": metrics,
    }


def build_batch_result_row(
    dataset_name: str,
    admission_id: str,
    patient_id: str,
    runtime_sec: float,
    diag_rows: List[str],
    med_rows: List[str],
    sent_texts: List[str],
    result: Dict,
) -> Dict[str, object]:
    metrics = result["metrics"]
    stage_timers = metrics.get("stage_timers", {})
    raw_pair_clusters = metrics.get("raw_pair_clusters", {})
    oracle_pair = raw_pair_clusters.get("oracle_pair", {})
    oracle_per_type = raw_pair_clusters.get("per_type_oracle", {})
    cluster_label = metrics.get("cluster_label", {})
    cluster_quality = metrics.get("cluster_quality", {})
    multi_rel = metrics.get("multi_rel_pair_recall", {})
    multi_rel_total = int(multi_rel.get("total", 0) or 0)
    multi_rel_ratio = None if multi_rel_total == 0 else round(float(multi_rel.get("recovered", 0)) / multi_rel_total, 4)
    gt_pair_failure_report = metrics.get("gt_pair_failure_report")
    gt_pairs_recovered = None
    gt_pairs_missed = None
    gt_pair_recovery_ratio = None
    if isinstance(gt_pair_failure_report, dict):
        gt_pairs_recovered = int(gt_pair_failure_report.get("n_recovered_pairs", 0))
        gt_pairs_missed = int(gt_pair_failure_report.get("n_missed_pairs", 0))
        gt_pair_total = int(gt_pair_failure_report.get("n_gt_pairs", 0))
        gt_pair_recovery_ratio = None if gt_pair_total == 0 else round(gt_pairs_recovered / gt_pair_total, 4)

    oracle_macro_precision = None
    oracle_macro_recall = None
    oracle_macro_f1 = None
    if isinstance(oracle_per_type, dict) and oracle_per_type:
        precision_values: List[float] = []
        recall_values: List[float] = []
        f1_values: List[float] = []
        for rel_type in REL_TYPES:
            metric = oracle_per_type.get(rel_type, {}) or {}
            n_pred = int(metric.get("n_pred", 0) or 0)
            n_gt = int(metric.get("n_gt", 0) or 0)
            if n_pred <= 0 and n_gt <= 0:
                continue
            precision_values.append(float(metric.get("precision", 0.0) or 0.0))
            recall_values.append(float(metric.get("recall", 0.0) or 0.0))
            f1_values.append(float(metric.get("f1", 0.0) or 0.0))
        if precision_values:
            oracle_macro_precision = round(sum(precision_values) / len(precision_values), 4)
            oracle_macro_recall = round(sum(recall_values) / len(recall_values), 4)
            oracle_macro_f1 = round(sum(f1_values) / len(f1_values), 4)

    row = {
        "dataset": dataset_name,
        "evaluation_profile": metrics.get("evaluation_profile", EVALUATION_PROFILE_DEFAULT),
        "admission_id": admission_id,
        "patient_id": patient_id,
        "runtime_sec": round(float(runtime_sec), 3),
        "n_diag_rows": len(diag_rows),
        "n_med_rows": len(med_rows),
        "n_sentences": len(sent_texts),
        "n_paths": int(metrics.get("evaluation_pred_path_count", len(result["paths"]))),
        "n_pred_pairs": metrics["relaxed_pair"]["n_pred"],
        "n_gt_pairs": metrics["relaxed_pair"]["n_gt"],
        "n_final_clusters": metrics.get("n_final_clusters", 0),
        "cluster_label_backend": metrics.get("cluster_label_backend"),
        "gliner2_label_input_mode": metrics.get("gliner2_label_input_mode"),
        "pair_average_precision": metrics.get("pair_average_precision"),
        "exact_triple_precision": metrics["exact_triple"]["precision"],
        "exact_triple_recall": metrics["exact_triple"]["recall"],
        "exact_triple_f1": metrics["exact_triple"]["f1"],
        "relaxed_pair_precision": metrics["relaxed_pair"]["precision"],
        "relaxed_pair_recall": metrics["relaxed_pair"]["recall"],
        "relaxed_pair_f1": metrics["relaxed_pair"]["f1"],
        "cluster_label_macro_precision": cluster_label.get("macro_precision"),
        "cluster_label_macro_recall": cluster_label.get("macro_recall"),
        "cluster_label_macro_f1": cluster_label.get("macro_f1"),
        "cluster_label_precision": cluster_label.get("precision"),
        "cluster_label_recall": cluster_label.get("recall"),
        "cluster_label_f1": cluster_label.get("f1"),
        "cluster_label_accuracy": cluster_label.get("accuracy"),
        "cluster_label_n_evaluated": cluster_label.get("n_evaluated"),
        "cluster_label_n_correct": cluster_label.get("n_correct"),
        "oracle_macro_precision": oracle_macro_precision,
        "oracle_macro_recall": oracle_macro_recall,
        "oracle_macro_f1": oracle_macro_f1,
        "typed_pair_precision": metrics["typed_pair"]["precision"],
        "typed_pair_recall": metrics["typed_pair"]["recall"],
        "typed_pair_f1": metrics["typed_pair"]["f1"],
        "typed_triple_precision": metrics["typed_triple"]["precision"],
        "typed_triple_recall": metrics["typed_triple"]["recall"],
        "typed_triple_f1": metrics["typed_triple"]["f1"],
        "diag_row_recall": metrics["diag_row_recall"],
        "med_row_recall": metrics["med_row_recall"],
        "multi_rel_pair_recall": multi_rel_ratio,
        "gt_pair_recovery_ratio": gt_pair_recovery_ratio,
        "gt_pairs_recovered": gt_pairs_recovered,
        "gt_pairs_missed": gt_pairs_missed,
        "gt_fail_sentence_side_top_k": _gt_failure_report_stage_count(gt_pair_failure_report, "SENTENCE_SIDE_TOP_K"),
        "gt_fail_max_pairs_per_sentence": _gt_failure_report_stage_count(gt_pair_failure_report, "MAX_PAIRS_PER_SENTENCE"),
        "gt_fail_transitive_join_threshold": _gt_failure_report_stage_count(gt_pair_failure_report, "TRANSITIVE_JOIN_THRESHOLD"),
        "gt_fail_row_side_top_k": _gt_failure_report_stage_count(gt_pair_failure_report, "ROW_SIDE_TOP_K"),
        "gt_fail_link_floor": _gt_failure_report_stage_count(gt_pair_failure_report, "LINK_FLOOR"),
        "raw_pair_cluster_purity": raw_pair_clusters.get("purity"),
        "raw_pair_oracle_precision": oracle_pair.get("precision"),
        "raw_pair_oracle_recall": oracle_pair.get("recall"),
        "raw_pair_oracle_f1": oracle_pair.get("f1"),
        "cluster_purity": cluster_quality.get("purity"),
        "cluster_ari": cluster_quality.get("ari"),
        "cluster_silhouette": metrics.get("cluster_silhouette"),
    }
    if isinstance(stage_timers, dict):
        for stage_name, elapsed in sorted(stage_timers.items()):
            if not isinstance(stage_name, str):
                continue
            if stage_name != "pipeline_total_sec" and not stage_name.startswith("phase_"):
                continue
            if elapsed is None:
                row[stage_name] = None
                continue
            row[stage_name] = round(float(elapsed), 4)
    return row


def _mean_metric(rows: List[Dict[str, object]], key: str) -> Optional[float]:
    values = []
    for row in rows:
        value = _to_float_or_none(row.get(key))
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def summarize_batch_rows(dataset_name: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    def _sum_int_metric(key: str) -> int:
        total = 0
        for row in rows:
            value = _to_float_or_none(row.get(key))
            if value is not None:
                total += int(value)
        return total

    def _sum_optional_int_metric(key: str) -> Optional[int]:
        total = 0
        found = False
        for row in rows:
            value = _to_float_or_none(row.get(key))
            if value is None:
                continue
            total += int(round(value))
            found = True
        return total if found else None

    def _sum_gt_pairs_recovered() -> int:
        total = 0
        for row in rows:
            explicit_value = _to_float_or_none(row.get("gt_pairs_recovered"))
            if explicit_value is not None:
                total += int(round(explicit_value))
                continue
            gt_pairs = _to_float_or_none(row.get("n_gt_pairs"))
            relaxed_recall = _to_float_or_none(row.get("relaxed_pair_recall"))
            if gt_pairs is None or relaxed_recall is None:
                continue
            total += int(round(float(gt_pairs) * float(relaxed_recall)))
        return total

    def _sum_gt_pairs_missed(recovered_total: int) -> int:
        explicit_total = _sum_int_metric("gt_pairs_missed")
        if explicit_total > 0:
            return explicit_total
        return max(_sum_int_metric("n_gt_pairs") - recovered_total, 0)

    metric_keys = [
        "runtime_sec",
        "pair_average_precision",
        "exact_triple_precision",
        "exact_triple_recall",
        "exact_triple_f1",
        "relaxed_pair_precision",
        "relaxed_pair_recall",
        "relaxed_pair_f1",
        "cluster_label_macro_precision",
        "cluster_label_macro_recall",
        "cluster_label_macro_f1",
        "cluster_label_precision",
        "cluster_label_recall",
        "cluster_label_f1",
        "cluster_label_accuracy",
        "oracle_macro_precision",
        "oracle_macro_recall",
        "oracle_macro_f1",
        "typed_pair_precision",
        "typed_pair_recall",
        "typed_pair_f1",
        "typed_triple_precision",
        "typed_triple_recall",
        "typed_triple_f1",
        "diag_row_recall",
        "med_row_recall",
        "multi_rel_pair_recall",
        "gt_pair_recovery_ratio",
        "gt_pairs_recovered",
        "gt_pairs_missed",
        "gt_fail_sentence_side_top_k",
        "gt_fail_max_pairs_per_sentence",
        "gt_fail_transitive_join_threshold",
        "gt_fail_row_side_top_k",
        "gt_fail_link_floor",
        "raw_pair_cluster_purity",
        "raw_pair_oracle_precision",
        "raw_pair_oracle_recall",
        "raw_pair_oracle_f1",
        "cluster_purity",
        "cluster_silhouette",
        "cluster_ari",
    ]
    stage_timer_keys = sorted({
        key
        for row in rows
        for key in row.keys()
        if isinstance(key, str) and (key == "pipeline_total_sec" or key.startswith("phase_")) and key.endswith("_sec")
    })
    metric_keys.extend(stage_timer_keys)
    averages = {key: _mean_metric(rows, key) for key in metric_keys}
    gt_pairs_recovered_total = _sum_gt_pairs_recovered()
    totals = {
        "n_admissions": len(rows),
        "n_paths": _sum_int_metric("n_paths"),
        "n_pred_pairs": _sum_int_metric("n_pred_pairs"),
        "n_gt_pairs": _sum_int_metric("n_gt_pairs"),
        "n_final_clusters": _sum_int_metric("n_final_clusters"),
        "cluster_label_n_evaluated": _sum_optional_int_metric("cluster_label_n_evaluated"),
        "cluster_label_n_correct": _sum_optional_int_metric("cluster_label_n_correct"),
        "gt_pairs_recovered": gt_pairs_recovered_total,
        "gt_pairs_missed": _sum_gt_pairs_missed(gt_pairs_recovered_total),
    }
    return {
        "dataset": dataset_name,
        "totals": totals,
        "averages": averages,
        "best_relaxed_pair_f1": max(rows, key=lambda row: float(row["relaxed_pair_f1"])),
        "best_pair_average_precision": max(rows, key=lambda row: float(row["pair_average_precision"] or -1.0)),
    }


def _batch_materialization_dir(dataset_name: str) -> Path:
    return BATCH_MATERIALIZATION_DIR / f"loki_batch_{dataset_name}"


def _batch_results_csv_path(dataset_name: str) -> Path:
    return _batch_materialization_dir(dataset_name) / f"materialized_batch_results_{dataset_name}.csv"


def _batch_summary_csv_path(dataset_name: str) -> Path:
    return _batch_materialization_dir(dataset_name) / f"materialized_batch_summary_{dataset_name}.csv"


def _batch_report_md_path(dataset_name: str) -> Path:
    return _batch_materialization_dir(dataset_name) / f"materialized_batch_report_{dataset_name}.md"


def _batch_failures_csv_path(dataset_name: str) -> Path:
    return _batch_materialization_dir(dataset_name) / f"materialized_batch_failures_{dataset_name}.csv"


def _batch_resume_state_path(dataset_name: str) -> Path:
    return _batch_materialization_dir(dataset_name) / f"materialized_batch_resume_state_{dataset_name}.json"


def _load_saved_batch_rows(dataset_name: str) -> List[Dict[str, object]]:
    results_path = _batch_results_csv_path(dataset_name)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Cannot resume batch run because the results CSV does not exist: {results_path}"
        )

    with open(results_path, newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]

    if not rows:
        raise RuntimeError(f"Cannot resume batch run because the results CSV is empty: {results_path}")
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_batch_resume_state(dataset_name: str) -> List[Dict[str, Any]]:
    state_path = _batch_resume_state_path(dataset_name)
    if not state_path.exists():
        return []
    with open(state_path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        return []
    return [record for record in payload if isinstance(record, dict)]


def save_batch_resume_state(dataset_name: str, records: List[Dict[str, Any]]) -> Path:
    state_path = _batch_resume_state_path(dataset_name)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(records), f, ensure_ascii=False, indent=2)
    return state_path


def _upsert_batch_resume_record(records: List[Dict[str, Any]], record: Dict[str, Any]) -> None:
    admission_id = str(record.get("admission_id", "")).strip()
    if not admission_id:
        return
    for index, existing in enumerate(records):
        if str(existing.get("admission_id", "")).strip() == admission_id:
            records[index] = record
            return
    records.append(record)


def _restore_batch_resume_artifacts(
    rows: List[Dict[str, object]],
    state_records: List[Dict[str, Any]],
) -> Tuple[bool, List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    state_by_admission: Dict[str, Dict[str, Any]] = {}
    for record in state_records:
        admission_id = str(record.get("admission_id", "")).strip()
        if admission_id:
            state_by_admission[admission_id] = record

    restored_metrics_payloads: List[Dict[str, object]] = []
    restored_pipeline_funnel_rows: List[Dict[str, object]] = []
    restored_pair_label_records: List[Dict[str, object]] = []
    restored_cluster_label_records: List[Dict[str, object]] = []
    state_complete = True

    for row in rows:
        admission_id = str(row.get("admission_id", "")).strip()
        if not admission_id:
            state_complete = False
            continue

        record = state_by_admission.get(admission_id)
        if record is None:
            state_complete = False
            continue

        metrics_payload = record.get("metrics_payload")
        pipeline_row = record.get("pipeline_funnel_row")
        pair_records = record.get("pair_label_records")
        cluster_records = record.get("cluster_label_records")

        if isinstance(metrics_payload, dict):
            restored_metrics_payloads.append(metrics_payload)
        else:
            state_complete = False

        if isinstance(pipeline_row, dict):
            restored_pipeline_funnel_rows.append(pipeline_row)
        else:
            state_complete = False

        if isinstance(pair_records, list):
            restored_pair_label_records.extend(
                pair_record for pair_record in pair_records if isinstance(pair_record, dict)
            )
        else:
            state_complete = False

        if isinstance(cluster_records, list):
            restored_cluster_label_records.extend(
                cluster_record for cluster_record in cluster_records if isinstance(cluster_record, dict)
            )
        else:
            state_complete = False

    return (
        state_complete,
        restored_metrics_payloads,
        restored_pipeline_funnel_rows,
        restored_pair_label_records,
        restored_cluster_label_records,
    )


def _clear_batch_failures_file(dataset_name: str) -> None:
    failed_path = _batch_failures_csv_path(dataset_name)
    if failed_path.exists():
        failed_path.unlink()


def save_batch_results(dataset_name: str, rows: List[Dict[str, object]]) -> Tuple[Path, Path, Path]:
    batch_dir = _batch_materialization_dir(dataset_name)
    batch_dir.mkdir(parents=True, exist_ok=True)
    results_path = _batch_results_csv_path(dataset_name)
    summary_path = _batch_summary_csv_path(dataset_name)
    report_path = _batch_report_md_path(dataset_name)

    summary = summarize_batch_rows(dataset_name, rows)

    fieldnames: List[str] = []
    seen_fieldnames: Set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen_fieldnames:
                continue
            seen_fieldnames.add(key)
            fieldnames.append(key)

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_row = {"dataset": dataset_name, **summary["totals"], **summary["averages"]}
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    lines = [
        f"# Batch Materialization Summary: {dataset_name}",
        "",
        f"- Evaluation profile: {rows[0].get('evaluation_profile', EVALUATION_PROFILE_DEFAULT)}",
        f"- Admissions evaluated: {summary['totals']['n_admissions']}",
        f"- Total predicted pairs: {summary['totals']['n_pred_pairs']}",
        f"- Total GT pairs: {summary['totals']['n_gt_pairs']}",
        f"- Mean pair average precision: {summary['averages']['pair_average_precision']}",
        f"- Mean relaxed pair F1: {summary['averages']['relaxed_pair_f1']}",
        f"- Mean exact triple F1: {summary['averages']['exact_triple_f1']}",
        f"- Mean cluster-level macro P/R/F1: {summary['averages']['cluster_label_macro_precision']} / {summary['averages']['cluster_label_macro_recall']} / {summary['averages']['cluster_label_macro_f1']}",
        f"- Mean oracle macro P/R/F1: {summary['averages']['oracle_macro_precision']} / {summary['averages']['oracle_macro_recall']} / {summary['averages']['oracle_macro_f1']}",
        f"- Mean raw pair-cluster purity: {summary['averages']['raw_pair_cluster_purity']}",
        f"- Mean cluster ARI: {summary['averages'].get('cluster_ari') if summary['averages'].get('cluster_ari') is not None else 'N/A'}",
        f"- Mean cluster silhouette: {summary['averages'].get('cluster_silhouette') if summary['averages'].get('cluster_silhouette') is not None else 'N/A'}",
        "",
    ]
    mean_stage_timers = [
        (key, summary["averages"].get(key))
        for key in sorted(summary["averages"].keys())
        if isinstance(key, str) and (key == "pipeline_total_sec" or key.startswith("phase_")) and key.endswith("_sec")
    ]
    if mean_stage_timers:
        lines.extend([
            "## Mean Stage Timers (sec)",
            "",
        ])
        for stage_name, elapsed in mean_stage_timers:
            elapsed_text = f"{elapsed:.4f}" if isinstance(elapsed, (int, float)) else "N/A"
            lines.append(f"- {stage_name}: {elapsed_text}")
        lines.append("")
    if summary["averages"].get("gt_pair_recovery_ratio") is not None:
        lines.extend([
            "## GT Pair Recovery",
            "",
            f"- Mean GT pair recovery ratio: {summary['averages']['gt_pair_recovery_ratio']}",
            f"- Total GT pairs recovered: {summary['totals']['gt_pairs_recovered']}",
            f"- Total GT pairs missed: {summary['totals']['gt_pairs_missed']}",
            f"- Mean SENTENCE_SIDE_TOP_K misses: {summary['averages']['gt_fail_sentence_side_top_k']}",
            f"- Mean MAX_PAIRS_PER_SENTENCE misses: {summary['averages']['gt_fail_max_pairs_per_sentence']}",
            f"- Mean TRANSITIVE_JOIN_THRESHOLD misses: {summary['averages']['gt_fail_transitive_join_threshold']}",
            "",
        ])

    lines.extend([
        "## Best Admissions",
        "",
        f"- Best relaxed pair F1: admission {summary['best_relaxed_pair_f1']['admission_id']} ({summary['best_relaxed_pair_f1']['relaxed_pair_f1']})",
        f"- Best pair average precision: admission {summary['best_pair_average_precision']['admission_id']} ({summary['best_pair_average_precision']['pair_average_precision']})",
        "",
        f"Per-admission CSV: {results_path.name}",
        f"Aggregate CSV: {summary_path.name}",
    ])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Saved batch results CSV : {results_path}")
    print(f"  Saved batch summary CSV : {summary_path}")
    print(f"  Saved batch report MD   : {report_path}")
    return results_path, summary_path, report_path


def save_batch_failures(dataset_name: str, failed_admissions: List[Dict[str, str]]) -> Path:
    batch_dir = _batch_materialization_dir(dataset_name)
    batch_dir.mkdir(parents=True, exist_ok=True)
    failed_path = _batch_failures_csv_path(dataset_name)
    with open(failed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["admission_id", "patient_id", "error"])
        writer.writeheader()
        writer.writerows(failed_admissions)
    return failed_path


def collect_dataset_projection_points(
    admission_id: str,
    paths: List[Dict],
    refined_rows: torch.Tensor,
    n_diag: int,
    refined_sentences: torch.Tensor,
    gt_relationships: List[Dict],
    sentence_encoder: Optional[SentenceTransformer],
    pair_embedding_mode: str = "contextual_sentence_average",
) -> List[Dict[str, object]]:
    if not paths:
        return []

    pair_keys, pair_embeddings = _compute_pair_embeddings(
        paths,
        refined_sentences,
        refined_rows=refined_rows,
        n_diag=n_diag,
        sentence_encoder=sentence_encoder,
        embedding_mode=pair_embedding_mode,
        verbose=False,
    )
    gt_pair_types = _build_gt_pair_type_lookup(gt_relationships)
    points: List[Dict[str, object]] = []
    for pair, embedding in zip(pair_keys, pair_embeddings):
        rel_types = gt_pair_types.get(pair, ())
        if not rel_types:
            continue
        points.append({
            "admission_id": admission_id,
            "label": rel_types[0],
            "multi_label": len(rel_types) > 1,
            "embedding": _to_numpy_array(embedding, dtype=np.float32),
        })
    return points


def collect_dataset_semantic_projection_records(
    admission_id: str,
    paths: List[Dict],
    refined_rows: torch.Tensor,
    n_diag: int,
    refined_sentences: torch.Tensor,
    gt_relationships: List[Dict],
    sentence_encoder: Optional[SentenceTransformer],
    pair_embedding_mode: str = "contextual_sentence_average",
    negative_pairs: Optional[set[Tuple[int, int]]] = None,
    cluster_key: str = "cluster_id",
) -> List[Dict[str, object]]:
    if not paths:
        return []

    pair_keys, pair_embeddings = _compute_pair_embeddings(
        paths,
        refined_sentences,
        refined_rows=refined_rows,
        n_diag=n_diag,
        sentence_encoder=sentence_encoder,
        embedding_mode=pair_embedding_mode,
        verbose=False,
    )
    if not pair_keys:
        return []

    pair_to_index = {pair: index for index, pair in enumerate(pair_keys)}
    pair_embeddings_np = _l2_normalize_rows(_to_numpy_array(pair_embeddings, dtype=np.float32))
    refined_sentences_np = _l2_normalize_rows(refined_sentences.float().cpu().numpy())

    cluster_paths: Dict[int, List[Dict]] = defaultdict(list)
    for path in paths:
        cluster_id = int(path.get(cluster_key, path.get("cluster_id", -1)))
        if cluster_id < 0:
            continue
        cluster_paths[cluster_id].append(path)
    if not cluster_paths:
        return []

    gt_pair_types = _build_gt_pair_type_lookup(gt_relationships)
    negative_pair_set = set(negative_pairs or set())
    cluster_semantic_types = {
        cluster_id: _resolve_cluster_semantic_type(cpaths, gt_pair_types, negative_pair_set)
        for cluster_id, cpaths in cluster_paths.items()
    }

    primary_types = {"TREATS", "ADVERSE_EFFECT", "NEGATIVE", "DISCONTINUED", "CONTRAINDICATED"}
    records: List[Dict[str, object]] = []
    for path in paths:
        pair = (int(path["diag_row_idx"]), int(path["med_row_idx"]))
        pair_index = pair_to_index.get(pair)
        if pair_index is None:
            continue

        cluster_id = int(path.get(cluster_key, path.get("cluster_id", -1)))
        cluster_semantic_type = cluster_semantic_types.get(cluster_id, "OTHER")
        semantic_type = _resolve_path_semantic_type(
            path,
            cluster_semantic_type,
            gt_pair_types,
            negative_pairs=negative_pair_set,
        )
        if semantic_type not in primary_types:
            continue

        pair_embedding = pair_embeddings_np[pair_index]
        sent_idx = int(path.get("sent_idx", -1))
        if 0 <= sent_idx < refined_sentences_np.shape[0] and pair_embedding.shape == refined_sentences_np.shape[1:]:
            path_embedding = (0.72 * pair_embedding) + (0.28 * refined_sentences_np[sent_idx])
            norm = float(np.linalg.norm(path_embedding))
            if norm > 0.0:
                path_embedding = path_embedding / norm
        else:
            path_embedding = pair_embedding

        records.append({
            "admission_id": admission_id,
            "label": semantic_type,
            "embedding": np.asarray(path_embedding, dtype=np.float32),
            "sentence_text": str(path.get("sent_text", "") or "").strip(),
        })
    return records


def _sample_projection_points(
    points: List[Dict[str, object]],
    max_points_per_type: int,
    max_total_points: int,
) -> List[Dict[str, object]]:
    if len(points) <= max_total_points:
        return points

    rng = np.random.default_rng(42)
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for point in points:
        grouped[str(point["label"])].append(point)

    sampled: List[Dict[str, object]] = []
    for label in _preferred_rel_type_order(sorted(grouped)):
        bucket = grouped.get(label, [])
        if len(bucket) <= max_points_per_type:
            sampled.extend(bucket)
            continue
        indices = rng.choice(len(bucket), size=max_points_per_type, replace=False)
        sampled.extend(bucket[int(idx)] for idx in sorted(indices))

    if len(sampled) <= max_total_points:
        return sampled

    indices = rng.choice(len(sampled), size=max_total_points, replace=False)
    return [sampled[int(idx)] for idx in sorted(indices)]


def visualize_dataset_projection_benchmark(
    dataset_name: str,
    points: List[Dict[str, object]],
    out_path: Path,
    max_points_per_type: int = 300,
    max_total_points: int = 1800,
) -> None:
    if len(points) < 3:
        print("  Dataset projection plot skipped (not enough GT-backed pair points).")
        return

    sampled = _sample_projection_points(points, max_points_per_type=max_points_per_type, max_total_points=max_total_points)
    embeddings = _l2_normalize_rows(np.stack([np.asarray(point["embedding"], dtype=np.float32) for point in sampled], axis=0))
    labels = [str(point["label"]) for point in sampled]
    multi_flags = [bool(point["multi_label"]) for point in sampled]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE, MDS
    except ImportError as exc:
        print(f"  Dataset projection plot skipped (missing library): {exc}")
        return

    coords_pca = PCA(n_components=2, random_state=42).fit_transform(embeddings).astype(np.float32)
    perplexity = min(35, max(6, len(sampled) // 12), len(sampled) - 1)
    coords_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", max_iter=1000).fit_transform(embeddings).astype(np.float32)

    projection_name = "UMAP"
    try:
        from umap import UMAP  # type: ignore

        reducer = UMAP(
            n_components=2,
            n_neighbors=min(20, max(6, len(sampled) // 25), len(sampled) - 1),
            min_dist=0.2,
            metric="cosine",
            random_state=42,
        )
        coords_umap = reducer.fit_transform(embeddings).astype(np.float32)
    except Exception:
        projection_name = "MDS"
        coords_umap = MDS(
            n_components=2,
            random_state=42,
            normalized_stress="auto",
            n_init=4,
            init="random",
        ).fit_transform(embeddings).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    fig.patch.set_facecolor("white")

    projections = [
        ("PCA", coords_pca),
        ("t-SNE", coords_tsne),
        (projection_name, coords_umap),
    ]
    ordered_labels = _preferred_rel_type_order(sorted(set(labels)))

    for ax, (title, coords) in zip(axes, projections):
        ax.set_facecolor("white")
        ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.85)
        for rel_type in ordered_labels:
            indices = [idx for idx, label in enumerate(labels) if label == rel_type]
            if not indices:
                continue
            single_idx = [idx for idx in indices if not multi_flags[idx]]
            multi_idx = [idx for idx in indices if multi_flags[idx]]
            if single_idx:
                ax.scatter(coords[single_idx, 0], coords[single_idx, 1],
                           s=28, c=[_rel_color(rel_type)], marker=_rel_marker(rel_type),
                           alpha=0.78, linewidths=0, label=rel_type)
            if multi_idx:
                ax.scatter(coords[multi_idx, 0], coords[multi_idx, 1],
                           s=42, c=[_rel_color(rel_type)], marker=_rel_marker(rel_type),
                           alpha=0.9, linewidths=1.0, edgecolors="#111827")

        ax.set_title(title, color="#111827", fontsize=12)
        ax.set_xlabel("Component 1", color="#111827")
        ax.set_ylabel("Component 2", color="#111827")
        ax.tick_params(colors="#4b5563")
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")

    handles = [
        plt.Line2D([0], [0], marker=_rel_marker(rel_type), color="none",
                   markerfacecolor=_rel_color(rel_type), markeredgecolor="white",
                   markersize=8, label=rel_type)
        for rel_type in ordered_labels
    ]
    handles.append(
        plt.Line2D([0], [0], marker="o", color="#111827", markerfacecolor="white",
                   linestyle="None", markersize=8, label="Multi-label pair")
    )
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"LOKI Pair-Embedding Projection Comparison - {dataset_name}\n"
        f"Sampled GT-backed predicted pairs (n={len(sampled)})",
        color="#111827", fontsize=14, y=1.02,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved dataset projection plot: {out_path}")


def visualize_dataset_projection_benchmark_3d(
    dataset_name: str,
    points: List[Dict[str, object]],
    out_path: Path,
    max_points_per_type: int = 300,
    max_total_points: int = 1800,
) -> None:
    if len(points) < 4:
        print("  Dataset 3D projection plot skipped (not enough GT-backed pair points).")
        return

    sampled = _sample_projection_points(points, max_points_per_type=max_points_per_type, max_total_points=max_total_points)
    embeddings = _l2_normalize_rows(np.stack([np.asarray(point["embedding"], dtype=np.float32) for point in sampled], axis=0))
    labels = [str(point["label"]) for point in sampled]
    multi_flags = [bool(point["multi_label"]) for point in sampled]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE, MDS
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:
        print(f"  Dataset 3D projection plot skipped (missing library): {exc}")
        return

    coords_pca = PCA(n_components=3, random_state=42).fit_transform(embeddings).astype(np.float32)
    perplexity = min(35, max(6, len(sampled) // 12), len(sampled) - 1)
    coords_tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        max_iter=1000,
    ).fit_transform(embeddings).astype(np.float32)

    projection_name = "UMAP"
    try:
        from umap import UMAP  # type: ignore

        reducer = UMAP(
            n_components=3,
            n_neighbors=min(20, max(6, len(sampled) // 25), len(sampled) - 1),
            min_dist=0.2,
            metric="cosine",
            random_state=42,
        )
        coords_umap = reducer.fit_transform(embeddings).astype(np.float32)
    except Exception:
        projection_name = "MDS"
        coords_umap = MDS(
            n_components=3,
            random_state=42,
            normalized_stress="auto",
            n_init=4,
            init="random",
        ).fit_transform(embeddings).astype(np.float32)

    fig = plt.figure(figsize=(18, 6.8))
    fig.patch.set_facecolor("white")
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]

    projections = [
        ("PCA 3D", coords_pca),
        ("t-SNE 3D", coords_tsne),
        (f"{projection_name} 3D", coords_umap),
    ]
    ordered_labels = _preferred_rel_type_order(sorted(set(labels)))

    for ax, (title, coords) in zip(axes, projections):
        ax.set_facecolor("white")
        ax.view_init(elev=22, azim=42)
        ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.7)
        for rel_type in ordered_labels:
            indices = [idx for idx, label in enumerate(labels) if label == rel_type]
            if not indices:
                continue
            single_idx = [idx for idx in indices if not multi_flags[idx]]
            multi_idx = [idx for idx in indices if multi_flags[idx]]
            if single_idx:
                ax.scatter(
                    coords[single_idx, 0], coords[single_idx, 1], coords[single_idx, 2],
                    s=18, c=[_rel_color(rel_type)], marker=_rel_marker(rel_type),
                    alpha=0.55, linewidths=0, depthshade=False,
                )
            if multi_idx:
                ax.scatter(
                    coords[multi_idx, 0], coords[multi_idx, 1], coords[multi_idx, 2],
                    s=30, c=[_rel_color(rel_type)], marker=_rel_marker(rel_type),
                    alpha=0.9, linewidths=0.8, edgecolors="#111827", depthshade=False,
                )

        ax.set_title(title, color="#111827", fontsize=12, pad=12)
        ax.set_xlabel("Component 1", color="#111827", labelpad=8)
        ax.set_ylabel("Component 2", color="#111827", labelpad=8)
        ax.set_zlabel("Component 3", color="#111827", labelpad=8)
        ax.tick_params(colors="#4b5563", labelsize=8)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.line.set_color("#d1d5db")
                axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
                axis.pane.set_edgecolor((0.88, 0.90, 0.92, 1.0))
            except Exception:
                pass

    handles = [
        plt.Line2D([0], [0], marker=_rel_marker(rel_type), color="none",
                   markerfacecolor=_rel_color(rel_type), markeredgecolor="white",
                   markersize=8, label=rel_type)
        for rel_type in ordered_labels
    ]
    handles.append(
        plt.Line2D([0], [0], marker="o", color="#111827", markerfacecolor="white",
                   linestyle="None", markersize=8, label="Multi-label pair")
    )
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        f"LOKI Pair-Embedding 3D Projection Comparison - {dataset_name}\n"
        f"Sampled GT-backed predicted pairs (n={len(sampled)})",
        color="#111827", fontsize=14, y=1.03,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved dataset 3D projection plot: {out_path}")


def visualize_batch_metric_overview(
    dataset_name: str,
    rows: List[Dict[str, object]],
    out_path: Path,
) -> None:
    if not rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Batch metric dashboard skipped (missing library): {exc}")
        return

    summary = summarize_batch_rows(dataset_name, rows)

    # Helper to safely convert per-row metric values to a numeric numpy array,
    # replacing None/NaN/unparseable values with `default` so points are not
    # dropped by matplotlib when coordinates are NaN.
    def _safe_row_float(rows_list, key, default=0.0):
        vals = []
        for r in rows_list:
            v = r.get(key)
            if v is None:
                vals.append(default)
                continue
            try:
                vals.append(float(v))
            except Exception:
                vals.append(default)
        arr = np.asarray(vals, dtype=np.float32)
        return np.nan_to_num(arr, nan=default)

    # Relaxed (end-to-end) metrics
    relaxed_precisions = _safe_row_float(rows, "relaxed_pair_precision", default=0.0)
    relaxed_recalls = _safe_row_float(rows, "relaxed_pair_recall", default=0.0)
    # Oracle (cluster-remapped) metrics (diagnostic)
    oracle_precisions = _safe_row_float(rows, "raw_pair_oracle_precision", default=0.0)
    oracle_recalls = _safe_row_float(rows, "raw_pair_oracle_recall", default=0.0)
    pair_ap = _safe_row_float(rows, "pair_average_precision", default=0.0)

    # Sizes: keep the same visual mapping used in materialize_joins
    n_gt_vals = []
    for r in rows:
        try:
            n = int(r.get("n_gt_pairs") or 0)
        except Exception:
            try:
                n = int(float(r.get("n_gt_pairs") or 0))
            except Exception:
                n = 0
        n_gt_vals.append(n)
    sizes = np.asarray([max(36, 9 * int(n)) for n in n_gt_vals], dtype=np.float32)

    # Layout: Relaxed scatter | Oracle scatter | Macro bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    fig.patch.set_facecolor("white")
    ax_relaxed, ax_oracle, ax_bar = axes

    # Styling sizes
    title_fs = 14
    label_fs = 12
    tick_fs = 10
    value_fs = 9
    suptitle_fs = 16

    # --- Relaxed scatter ---
    ax_relaxed.set_facecolor("white")
    ax_relaxed.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.85)
    scatter_relaxed = ax_relaxed.scatter(
        relaxed_recalls,
        relaxed_precisions,
        c=pair_ap,
        s=sizes,
        cmap="viridis",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
    )
    for target_f1 in (0.2, 0.4, 0.6):
        f1_x = np.linspace(max(target_f1 / 2.0 + 1e-3, 0.01), 1.0, 200)
        f1_y = (target_f1 * f1_x) / np.maximum(2.0 * f1_x - target_f1, 1e-6)
        valid = (f1_y >= 0.0) & (f1_y <= 1.0)
        ax_relaxed.plot(f1_x[valid], f1_y[valid], linestyle="--", linewidth=1.0, color="#cbd5e1")
        if np.any(valid):
            x_text = min(float(f1_x[valid][-1]), 0.78)
            ax_relaxed.text(x_text, float(f1_y[valid][-1]), f"F1={target_f1:.1f}", fontsize=value_fs, color="#64748b")

    top_relaxed = sorted(rows, key=lambda row: float(row["relaxed_pair_f1"]), reverse=True)[:5]
    for row in top_relaxed:
        ax_relaxed.text(
            float(row["relaxed_pair_recall"]),
            float(row["relaxed_pair_precision"]),
            str(row["admission_id"]),
            fontsize=value_fs,
            color="#111827",
        )

    ax_relaxed.set_title("Admission-Level Relaxed Pair Precision / Recall", color="#111827", fontsize=title_fs)
    ax_relaxed.set_xlabel("Relaxed pair recall", color="#111827", fontsize=label_fs)
    ax_relaxed.set_ylabel("Relaxed pair precision", color="#111827", fontsize=label_fs)
    ax_relaxed.tick_params(colors="#4b5563", labelsize=tick_fs)
    for spine in ax_relaxed.spines.values():
        spine.set_edgecolor("#d1d5db")
    cbar1 = fig.colorbar(scatter_relaxed, ax=ax_relaxed, fraction=0.046, pad=0.04)
    cbar1.set_label("Pair average precision", color="#111827", fontsize=value_fs)

    # --- Oracle scatter (diagnostic) ---
    ax_oracle.set_facecolor("white")
    ax_oracle.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.85)
    scatter_oracle = ax_oracle.scatter(
        oracle_recalls,
        oracle_precisions,
        c=pair_ap,
        s=sizes,
        cmap="viridis",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
    )
    for target_f1 in (0.2, 0.4, 0.6):
        f1_x = np.linspace(max(target_f1 / 2.0 + 1e-3, 0.01), 1.0, 200)
        f1_y = (target_f1 * f1_x) / np.maximum(2.0 * f1_x - target_f1, 1e-6)
        valid = (f1_y >= 0.0) & (f1_y <= 1.0)
        ax_oracle.plot(f1_x[valid], f1_y[valid], linestyle="--", linewidth=1.0, color="#cbd5e1")
        if np.any(valid):
            x_text = min(float(f1_x[valid][-1]), 0.78)
            ax_oracle.text(x_text, float(f1_y[valid][-1]), f"F1={target_f1:.1f}", fontsize=value_fs, color="#64748b")

    # Admission labels for Oracle panel intentionally omitted (diagnostic view)

    ax_oracle.set_title("Admission-Level Oracle Pair Precision / Recall", color="#111827", fontsize=title_fs)
    ax_oracle.set_xlabel("Oracle pair recall", color="#111827", fontsize=label_fs)
    ax_oracle.set_ylabel("Oracle pair precision", color="#111827", fontsize=label_fs)
    ax_oracle.tick_params(colors="#4b5563", labelsize=tick_fs)
    for spine in ax_oracle.spines.values():
        spine.set_edgecolor("#d1d5db")
    cbar2 = fig.colorbar(scatter_oracle, ax=ax_oracle, fraction=0.046, pad=0.04)
    cbar2.set_label("Pair average precision", color="#111827", fontsize=value_fs)

    # --- Macro-average bar chart (condensed bars) ---
    metric_names = [
        "Pair AP",
        "Relaxed Pair F1",
        "Exact Triple F1",
        "Typed Pair F1",
        "Oracle Pair F1",
    ]
    metric_values = [
        summary["averages"].get("pair_average_precision") or 0.0,
        summary["averages"].get("relaxed_pair_f1") or 0.0,
        summary["averages"].get("exact_triple_f1") or 0.0,
        summary["averages"].get("typed_pair_f1") or 0.0,
        summary["averages"].get("raw_pair_oracle_f1") or 0.0,
    ]
    bar_colors = ["#0ea5e9", "#22c55e", "#f97316", "#ef4444", "#8b5cf6"]
    positions = np.arange(len(metric_names))
    bar_width = 0.6
    # Replace None with 0.0 for plotting; track which are N/A for annotation
    plot_values = [v if v is not None else 0.0 for v in metric_values]
    ax_bar.bar(positions, plot_values, width=bar_width, color=bar_colors, alpha=0.9)
    # Allow negative silhouette values to display; otherwise keep lower bound at 0.0
    min_val = min(plot_values)
    max_val = max(plot_values)
    if min_val < 0.0:
        y_min = min(-1.0, min_val - 0.03)
    else:
        y_min = 0.0
    y_max = max(1.0, max_val + 0.03)
    ax_bar.set_ylim(y_min, y_max)
    ax_bar.set_title("Macro-Average Batch Metrics", color="#111827", fontsize=title_fs)
    ax_bar.set_ylabel("Score", color="#111827", fontsize=label_fs)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(metric_names, fontsize=tick_fs, rotation=18)
    ax_bar.tick_params(axis="y", colors="#4b5563", labelsize=tick_fs)
    ax_bar.grid(True, axis="y", color="#e5e7eb", linewidth=0.8, alpha=0.85)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#d1d5db")
    for index, value in enumerate(metric_values):
        if value is None:
            ax_bar.text(index, 0.02, "N/A", ha="center", va="bottom", fontsize=value_fs, color="#6b7280", style="italic")
            continue
        # Position label above positive bars, below negative bars
        if value < 0:
            ax_bar.text(index, value - 0.02, f"{value:.3f}", ha="center", va="top", fontsize=value_fs, color="#111827")
        else:
            ax_bar.text(index, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=value_fs, color="#111827")

    fig.suptitle(
        f"LOKI Batch Evaluation Dashboard - {dataset_name}\n"
        f"Admissions={summary['totals']['n_admissions']}  Pred pairs={summary['totals']['n_pred_pairs']}  GT pairs={summary['totals']['n_gt_pairs']}  TP={summary['totals']['gt_pairs_recovered']}",
        color="#111827",
        fontsize=suptitle_fs,
        y=1.02,
    )
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved batch metric dashboard: {out_path}")


def visualize_batch_representation_overview(
    dataset_name: str,
    rows: List[Dict[str, object]],
    out_path: Path,
) -> None:
    """Paper-facing batch dashboard focused on representation and clustering quality.

    Shows the admission-level cluster macro precision/recall spread and the
    headline relationship clustering metrics used in the paper-facing dashboard.
    """
    if not rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Batch representation dashboard skipped (missing library): {exc}")
        return

    summary = summarize_batch_rows(dataset_name, rows)

    def _safe_row_float(rows_list, key, default=0.0):
        vals = []
        for r in rows_list:
            v = r.get(key)
            if v is None:
                vals.append(default)
                continue
            try:
                vals.append(float(v))
            except Exception:
                vals.append(default)
        arr = np.asarray(vals, dtype=np.float32)
        return np.nan_to_num(arr, nan=default)

    def _spread_overlapping_points(x_vals, y_vals, radius=0.025):
        spread_x = np.asarray(x_vals, dtype=np.float32).copy()
        spread_y = np.asarray(y_vals, dtype=np.float32).copy()
        duplicate_groups: Dict[Tuple[float, float], List[int]] = defaultdict(list)
        for idx, (x_val, y_val) in enumerate(zip(spread_x, spread_y)):
            duplicate_groups[(round(float(x_val), 4), round(float(y_val), 4))].append(idx)

        for (_key_x, _key_y), indices in duplicate_groups.items():
            if len(indices) <= 1:
                continue
            group_radius = min(radius, 0.01 + 0.004 * len(indices))
            base_x = min(max(float(spread_x[indices[0]]), group_radius), 1.0 - group_radius)
            base_y = min(max(float(spread_y[indices[0]]), group_radius), 1.0 - group_radius)
            angles = np.linspace(0.0, 2.0 * np.pi, len(indices), endpoint=False)
            for offset_idx, point_idx in enumerate(indices):
                spread_x[point_idx] = base_x + group_radius * np.cos(float(angles[offset_idx]))
                spread_y[point_idx] = base_y + group_radius * np.sin(float(angles[offset_idx]))
        return spread_x, spread_y

    cluster_label_precisions = _safe_row_float(rows, "cluster_label_macro_precision", default=0.0)
    cluster_label_recalls = _safe_row_float(rows, "cluster_label_macro_recall", default=0.0)
    pair_ap = _safe_row_float(rows, "pair_average_precision", default=0.0)
    scatter_recalls, scatter_precisions = _spread_overlapping_points(
        cluster_label_recalls,
        cluster_label_precisions,
    )

    n_gt_vals = []
    for r in rows:
        try:
            n = int(r.get("n_gt_pairs") or 0)
        except Exception:
            try:
                n = int(float(r.get("n_gt_pairs") or 0))
            except Exception:
                n = 0
        n_gt_vals.append(n)
    sizes = np.asarray([max(36, 9 * int(n)) for n in n_gt_vals], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.4))
    fig.patch.set_facecolor("white")
    ax_auto, ax_bar = axes

    title_fs = 14
    label_fs = 12
    tick_fs = 10
    value_fs = 9
    suptitle_fs = 16

    ax_auto.set_facecolor("white")
    ax_auto.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.85)
    scatter_auto = ax_auto.scatter(
        scatter_recalls,
        scatter_precisions,
        c=pair_ap,
        s=sizes,
        cmap="viridis",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
    )
    for target_f1 in (0.2, 0.4, 0.6):
        f1_x = np.linspace(max(target_f1 / 2.0 + 1e-3, 0.01), 1.0, 200)
        f1_y = (target_f1 * f1_x) / np.maximum(2.0 * f1_x - target_f1, 1e-6)
        valid = (f1_y >= 0.0) & (f1_y <= 1.0)
        ax_auto.plot(f1_x[valid], f1_y[valid], linestyle="--", linewidth=1.0, color="#cbd5e1")
        if np.any(valid):
            x_text = min(float(f1_x[valid][-1]), 0.78)
            ax_auto.text(x_text, float(f1_y[valid][-1]), f"F1={target_f1:.1f}", fontsize=value_fs, color="#64748b")

    ax_auto.set_title("P/R (Macro) Per Admission for Relationship Clustering", color="#111827", fontsize=title_fs)
    ax_auto.set_xlabel("Cluster macro recall", color="#111827", fontsize=label_fs)
    ax_auto.set_ylabel("Cluster macro precision", color="#111827", fontsize=label_fs)
    ax_auto.tick_params(colors="#4b5563", labelsize=tick_fs)
    for spine in ax_auto.spines.values():
        spine.set_edgecolor("#d1d5db")

    cbar = fig.colorbar(scatter_auto, ax=ax_auto, fraction=0.046, pad=0.04)
    cbar.set_label("Pair average precision", color="#111827", fontsize=value_fs)

    metric_names = [
        "Macro\nP",
        "Macro\nR",
        "Macro\nF1",
        "Mean\nAccuracy",
        "Cluster\nPurity",
        "Cluster\nARI",
        "Cluster\nSilhouette",
    ]
    metric_values = [
        summary["averages"].get("cluster_label_macro_precision") or 0.0,
        summary["averages"].get("cluster_label_macro_recall") or 0.0,
        summary["averages"].get("cluster_label_macro_f1") or 0.0,
        summary["averages"].get("cluster_label_accuracy") or 0.0,
        summary["averages"].get("raw_pair_cluster_purity") or 0.0,
        summary["averages"].get("cluster_ari") if summary["averages"].get("cluster_ari") is not None else None,
        summary["averages"].get("cluster_silhouette") if summary["averages"].get("cluster_silhouette") is not None else None,
    ]
    bar_colors = ["#ef4444", "#dc2626", "#b91c1c", "#fb923c", "#14b8a6", "#334155", "#6366f1"]
    positions = np.arange(len(metric_names))
    plot_values = [v if v is not None else 0.0 for v in metric_values]

    ax_bar.set_facecolor("white")
    ax_bar.bar(positions, plot_values, width=0.62, color=bar_colors, alpha=0.9)
    min_val = min(plot_values)
    max_val = max(plot_values)
    if min_val < 0.0:
        y_min = min(-1.0, min_val - 0.03)
    else:
        y_min = 0.0
    y_max = max(1.08, max_val + 0.08)
    ax_bar.set_ylim(y_min, y_max)
    ax_bar.set_title("Relationship Clustering Metrics", color="#111827", fontsize=title_fs)
    ax_bar.set_ylabel("Score", color="#111827", fontsize=label_fs)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(metric_names, fontsize=tick_fs, rotation=0, ha="center")
    ax_bar.tick_params(axis="y", colors="#4b5563", labelsize=tick_fs)
    ax_bar.grid(True, axis="y", color="#e5e7eb", linewidth=0.8, alpha=0.85)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#d1d5db")
    y_range = max(y_max - y_min, 1e-6)
    label_margin = 0.05 * y_range
    for index, value in enumerate(metric_values):
        if value is None:
            ax_bar.text(index, 0.02, "N/A", ha="center", va="bottom", fontsize=value_fs, color="#6b7280", style="italic")
            continue
        if value < 0:
            label_y = max(value - 0.02, y_min + label_margin)
            ax_bar.text(index, label_y, f"{value:.3f}", ha="center", va="top", fontsize=value_fs, color="#111827")
        else:
            label_y = min(value + 0.02, y_max - label_margin)
            ax_bar.text(index, label_y, f"{value:.3f}", ha="center", va="bottom", fontsize=value_fs, color="#111827")

    title_counts = [f"Admissions={summary['totals']['n_admissions']}"]
    evaluated_clusters = summary["totals"].get("cluster_label_n_evaluated")
    correctly_labeled_clusters = summary["totals"].get("cluster_label_n_correct")
    if evaluated_clusters is not None:
        title_counts.append(f"Evaluated clusters={evaluated_clusters}")
    if correctly_labeled_clusters is not None:
        title_counts.append(f"Correctly labeled clusters={correctly_labeled_clusters}")
    title_counts.append(f"Final predicted clusters={summary['totals']['n_final_clusters']}")

    fig.suptitle(
        f"LOKI Relationship Clustering Dashboard - {dataset_name}\n"
        + "  ".join(title_counts),
        color="#111827",
        fontsize=suptitle_fs,
        y=1.02,
    )
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.28)
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved batch representation dashboard: {out_path}")


# =============================================================================
# VLDB-grade aggregate batch visualizations
# =============================================================================

def visualize_batch_pipeline_funnel(
    dataset_name: str,
    funnel_rows: List[Dict[str, object]],
    out_path: Path,
) -> None:
    """Corpus-aggregated pipeline funnel.

    Sums per-admission counts across every stage of the LOKI materialization
    pipeline so the reader can see, in one glance, how much each filter
    contributes to precision and how many GT pairs survive each gate.
    """
    if not funnel_rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Batch pipeline funnel skipped (missing library): {exc}")
        return

    def _sum(key: str) -> int:
        total = 0
        for r in funnel_rows:
            v = r.get(key)
            try:
                total += int(v) if v is not None else 0
            except Exception:
                continue
        return total

    stages = [
        ("Stage-5\ncandidate pairs", _sum("stage5_candidate_pairs"), "#1d4ed8"),
        ("After deterministic\npair filter",   _sum("after_pair_filter_pairs"), "#0ea5e9"),
        ("After CE\npair filter (Option D)", _sum("after_ce_filter_pairs"), "#14b8a6"),
        ("After cluster-tail\nfilter (= predicted)", _sum("after_cluster_tail_pairs"), "#22c55e"),
        ("GT pairs\nrecovered",  _sum("gt_pairs_recovered"), "#f59e0b"),
        ("Total GT pairs\n(annotation)",       _sum("n_gt_pairs"), "#9ca3af"),
    ]
    # Filter out any zero stages (e.g. CE filter disabled) but always keep the
    # first/last anchors so the visual baseline stays interpretable.
    pruned = [stages[0]] + [s for s in stages[1:-1] if s[1] > 0] + [stages[-1]]
    if stages[-2][1] > 0 and stages[-2] not in pruned:
        pruned.insert(-1, stages[-2])

    labels   = [s[0] for s in pruned]
    counts   = [s[1] for s in pruned]
    colors   = [s[2] for s in pruned]
    n_admissions = len(funnel_rows)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8, alpha=0.85)

    y_pos = np.arange(len(labels))[::-1]
    bars = ax.barh(y_pos, counts, color=colors, edgecolor="white", linewidth=1.2, alpha=0.92)

    base_count = counts[0] if counts and counts[0] > 0 else 1
    for bar, count, label in zip(bars, counts, labels):
        pct = 100.0 * count / base_count
        ax.text(
            bar.get_width() + max(counts) * 0.012,
            bar.get_y() + bar.get_height() / 2.0,
            f"{count:,}   ({pct:.1f}% of input)",
            va="center", ha="left",
            fontsize=10, color="#111827",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, color="#1f2937")
    ax.set_xlabel("Pair count (summed over corpus)", fontsize=11, color="#1f2937")
    ax.set_xlim(0, max(counts) * 1.22 if counts else 1)
    ax.set_title(
        f"LOKI Pipeline Funnel - {dataset_name}\n"
        f"Aggregated across {n_admissions} admissions",
        fontsize=13, color="#111827",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved batch pipeline funnel: {out_path}")


def visualize_batch_gamma_vs_f1(
    dataset_name: str,
    batch_rows: List[Dict[str, object]],
    batch_metrics_payloads: List[Dict[str, object]],
    out_path: Path,
) -> None:
    """Per-admission gamma vs relaxed pair F1 scatter, coloured by GT pair count.

    Empirically justifies the adaptive-gamma choice: shows the operating-region
    plateau where F1 is robust to gamma, and surfaces hard admissions where gamma
    drifts into a regime where F1 collapses.
    """
    if not batch_rows or not batch_metrics_payloads:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Batch gamma-vs-F1 plot skipped (missing library): {exc}")
        return

    gammas: List[float] = []
    f1s: List[float] = []
    n_gts: List[int] = []
    n_paths: List[int] = []
    n_payloads = len(batch_metrics_payloads)
    for idx, row in enumerate(batch_rows):
        if idx >= n_payloads:
            break
        payload = batch_metrics_payloads[idx] or {}
        stage5_cfg = payload.get("stage5_config") or {}
        try:
            gamma = float(stage5_cfg.get("gamma"))
        except Exception:
            continue
        try:
            f1 = float(row.get("relaxed_pair_f1") or 0.0)
        except Exception:
            f1 = 0.0
        try:
            n_gt = int(row.get("n_gt_pairs") or 0)
        except Exception:
            n_gt = 0
        try:
            n_path = int(row.get("n_paths") or 0)
        except Exception:
            n_path = 0
        gammas.append(gamma)
        f1s.append(f1)
        n_gts.append(n_gt)
        n_paths.append(n_path)

    if not gammas:
        print("  Batch gamma-vs-F1 plot skipped (no gamma values captured).")
        return

    gammas_arr = np.asarray(gammas, dtype=np.float32)
    f1_arr = np.asarray(f1s, dtype=np.float32)
    n_gts_arr = np.asarray(n_gts, dtype=np.float32)
    sizes = np.asarray([max(24, 6 * n) for n in n_paths], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 6.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.85)

    sc = ax.scatter(
        gammas_arr, f1_arr,
        c=n_gts_arr, s=sizes,
        cmap="plasma", alpha=0.78,
        edgecolors="white", linewidths=0.6,
    )

    # Mean gamma and median F1 reference lines
    mean_gamma = float(np.mean(gammas_arr))
    median_f1 = float(np.median(f1_arr))
    ax.axvline(mean_gamma, linestyle="--", linewidth=1.0, color="#94a3b8")
    ax.axhline(median_f1, linestyle="--", linewidth=1.0, color="#94a3b8")
    ax.text(mean_gamma, ax.get_ylim()[1] * 0.97, f" mean gamma={mean_gamma:.3f}",
            fontsize=9, color="#475569", va="top")
    ax.text(ax.get_xlim()[1] * 0.99, median_f1, f"median F1={median_f1:.3f} ",
            fontsize=9, color="#475569", ha="right", va="bottom")

    cbar = plt.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("# GT pairs per admission", fontsize=10, color="#1f2937")

    ax.set_xlabel("Adaptive gamma threshold (per admission)", fontsize=11, color="#1f2937")
    ax.set_ylabel("Relaxed pair F1", fontsize=11, color="#1f2937")
    ax.set_title(
        f"Per-Admission gamma vs Relaxed Pair F1 - {dataset_name}\n"
        f"{len(gammas)} admissions, point size ∝ candidate-triple count",
        fontsize=13, color="#111827",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved batch gamma-vs-F1 scatter: {out_path}")


def visualize_batch_confusion_matrix(
    dataset_name: str,
    pair_label_records: List[Dict[str, object]],
    out_path: Path,
    normalize: bool = True,
) -> None:
    """Dataset-scale predicted x GT relation-label confusion matrix.

    Rows are GT labels only; unmatched predicted pairs are excluded so the
    figure reflects label confusion over GT-matched pairs rather than corpus-
    level false positives. Columns still include ``UNLABELED`` for clusters
    that received no semantic name. Row-normalised so the matrix reads as
    per-class recall by default.
    """
    if not pair_label_records:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Batch confusion matrix skipped (missing library): {exc}")
        return

    def _clean(label: object, fallback: str) -> str:
        s = str(label or "").strip()
        if not s:
            return fallback
        return _normalize_rel_type(s)

    gt_labels_seen: set = set()
    pred_labels_seen: set = set()
    cleaned: List[Tuple[str, str]] = []
    filtered_records: List[Dict[str, object]] = []
    for rec in pair_label_records:
        gt = _clean(rec.get("gt_label"), "NONE")
        pr = _clean(rec.get("predicted_label"), "UNLABELED")
        if gt == "NONE":
            continue
        gt_labels_seen.add(gt)
        pred_labels_seen.add(pr)
        cleaned.append((gt, pr))
        filtered_records.append(rec)

    if not cleaned:
        print("  Batch confusion matrix skipped (no GT-matched predicted pairs after filtering false positives).")
        return

    canonical = [t for t in REL_TYPES if t in (gt_labels_seen | pred_labels_seen)]
    gt_order = list(canonical)
    pred_order = list(canonical)
    for extra in sorted(pred_labels_seen - set(pred_order)):
        pred_order.append(extra)
    # Bubble UNLABELED to the end if present
    if "UNLABELED" in pred_order:
        pred_order = [p for p in pred_order if p != "UNLABELED"] + ["UNLABELED"]

    gt_idx = {lbl: i for i, lbl in enumerate(gt_order)}
    pred_idx = {lbl: j for j, lbl in enumerate(pred_order)}

    matrix = np.zeros((len(gt_order), len(pred_order)), dtype=np.int64)
    for gt, pr in cleaned:
        if gt not in gt_idx or pr not in pred_idx:
            continue
        matrix[gt_idx[gt], pred_idx[pr]] += 1

    row_totals = matrix.sum(axis=1, keepdims=True)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            display = np.where(row_totals > 0, matrix / np.maximum(row_totals, 1), 0.0)
    else:
        display = matrix.astype(np.float32)

    fig_w = max(7.5, 0.85 * len(pred_order) + 4.0)
    fig_h = max(5.5, 0.65 * len(gt_order) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    im = ax.imshow(display, cmap="Blues", aspect="auto", vmin=0.0,
                   vmax=(1.0 if normalize else None))

    ax.set_xticks(np.arange(len(pred_order)))
    ax.set_xticklabels(pred_order, rotation=35, ha="right", fontsize=9, color="#1f2937")
    ax.set_yticks(np.arange(len(gt_order)))
    ax.set_yticklabels([f"{lbl}  (n={int(row_totals[i, 0])})" for i, lbl in enumerate(gt_order)],
                       fontsize=9, color="#1f2937")
    ax.set_xlabel("Predicted relation label", fontsize=11, color="#1f2937")
    ax.set_ylabel("Ground-truth relation label", fontsize=11, color="#1f2937")

    # Cell annotations
    max_val = float(display.max()) if display.size else 0.0
    for i in range(len(gt_order)):
        for j in range(len(pred_order)):
            count = int(matrix[i, j])
            if count == 0:
                continue
            cell = float(display[i, j])
            color = "white" if cell > max_val * 0.55 else "#1f2937"
            if normalize:
                txt = f"{cell:.2f}\n({count})"
            else:
                txt = f"{count}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label("Row-normalised fraction" if normalize else "Pair count",
                   fontsize=10, color="#1f2937")

    n_admissions = len({rec.get("admission_id") for rec in filtered_records})
    ax.set_title(
        f"Predicted vs GT Relation Labels - {dataset_name}\n"
        f"{len(filtered_records):,} GT-matched predicted pairs across {n_admissions} admissions"
        + ("  (row-normalised -> recall)" if normalize else ""),
        fontsize=13, color="#111827",
    )

    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved batch confusion matrix: {out_path}")


def visualize_batch_cluster_confusion_matrix(
    dataset_name: str,
    cluster_label_records: List[Dict[str, object]],
    out_path: Path,
    normalize: bool = True,
) -> None:
    """Dataset-scale predicted x oracle cluster-label confusion matrix.

    Evaluates only the GT-anchored clusters that receive an oracle relation type,
    matching the cluster-level macro P/R/F1/accuracy support shown elsewhere.
    Rows are oracle cluster types; columns are predicted cluster types, with an
    optional ``UNLABELED`` column for clusters with no valid semantic label.
    """
    if not cluster_label_records:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Batch cluster confusion matrix skipped (missing library): {exc}")
        return

    def _clean(label: object, fallback: str) -> str:
        s = str(label or "").strip()
        if not s:
            return fallback
        return _normalize_rel_type(s)

    gt_labels_seen: set = set()
    pred_labels_seen: set = set()
    cleaned: List[Tuple[str, str]] = []
    filtered_records: List[Dict[str, object]] = []
    for rec in cluster_label_records:
        gt = _clean(rec.get("gt_label"), "NONE")
        pr = _clean(rec.get("predicted_label"), "UNLABELED")
        if gt == "NONE":
            continue
        gt_labels_seen.add(gt)
        pred_labels_seen.add(pr)
        cleaned.append((gt, pr))
        filtered_records.append(rec)

    if not cleaned:
        print("  Batch cluster confusion matrix skipped (no evaluated clusters after filtering).")
        return

    canonical = [t for t in REL_TYPES if t in (gt_labels_seen | pred_labels_seen)]
    gt_order = list(canonical)
    pred_order = list(canonical)
    for extra in sorted(pred_labels_seen - set(pred_order)):
        pred_order.append(extra)
    if "UNLABELED" in pred_order:
        pred_order = [p for p in pred_order if p != "UNLABELED"] + ["UNLABELED"]

    gt_idx = {lbl: i for i, lbl in enumerate(gt_order)}
    pred_idx = {lbl: j for j, lbl in enumerate(pred_order)}

    matrix = np.zeros((len(gt_order), len(pred_order)), dtype=np.int64)
    for gt, pr in cleaned:
        if gt not in gt_idx or pr not in pred_idx:
            continue
        matrix[gt_idx[gt], pred_idx[pr]] += 1

    row_totals = matrix.sum(axis=1, keepdims=True)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            display = np.where(row_totals > 0, matrix / np.maximum(row_totals, 1), 0.0)
    else:
        display = matrix.astype(np.float32)

    fig_w = max(7.5, 0.85 * len(pred_order) + 4.0)
    fig_h = max(5.5, 0.65 * len(gt_order) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    im = ax.imshow(display, cmap="Blues", aspect="auto", vmin=0.0,
                   vmax=(1.0 if normalize else None))

    ax.set_xticks(np.arange(len(pred_order)))
    ax.set_xticklabels(pred_order, rotation=35, ha="right", fontsize=9, color="#1f2937")
    ax.set_yticks(np.arange(len(gt_order)))
    ax.set_yticklabels([f"{lbl}  (n={int(row_totals[i, 0])})" for i, lbl in enumerate(gt_order)],
                       fontsize=9, color="#1f2937")
    ax.set_xlabel("Predicted cluster label", fontsize=11, color="#1f2937")
    ax.set_ylabel("Oracle cluster label", fontsize=11, color="#1f2937")

    max_val = float(display.max()) if display.size else 0.0
    for i in range(len(gt_order)):
        for j in range(len(pred_order)):
            count = int(matrix[i, j])
            if count == 0:
                continue
            cell = float(display[i, j])
            color = "white" if cell > max_val * 0.55 else "#1f2937"
            if normalize:
                txt = f"{cell:.2f}\n({count})"
            else:
                txt = f"{count}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label("Row-normalised fraction" if normalize else "Cluster count",
                   fontsize=10, color="#1f2937")

    n_admissions = len({rec.get("admission_id") for rec in filtered_records})
    n_correct = sum(1 for rec in filtered_records if bool(rec.get("correct")))
    ax.set_title(
        f"Predicted vs Oracle Cluster Labels - {dataset_name}\n"
        f"{len(filtered_records):,} evaluated clusters across {n_admissions} admissions  "
        f"Correctly labeled={n_correct:,}"
        + ("  (row-normalised -> recall)" if normalize else ""),
        fontsize=13, color="#111827",
    )

    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved batch cluster confusion matrix: {out_path}")


def summarize_batch_classwise_typed_metrics(
    metrics_by_admission: List[Dict[str, object]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    families = ("untyped_pair", "auto_pair", "auto_triple", "oracle_pair", "oracle_triple")
    totals: Dict[str, Dict[str, Dict[str, float]]] = {
        family: {
            rel_type: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "_count": 0.0}
            for rel_type in REL_TYPES
        }
        for family in families
    }

    for metrics in metrics_by_admission:
        classwise = metrics.get("classwise_typed_metrics", {})
        if not isinstance(classwise, dict):
            continue
        for family in families:
            family_payload = classwise.get(family, {})
            if not isinstance(family_payload, dict):
                continue
            for rel_type in REL_TYPES:
                rel_payload = family_payload.get(rel_type, {})
                if not isinstance(rel_payload, dict):
                    continue
                for metric_name in ("precision", "recall", "f1"):
                    value = _to_float_or_none(rel_payload.get(metric_name))
                    if value is None:
                        continue
                    totals[family][rel_type][metric_name] += float(value)
                totals[family][rel_type]["_count"] += 1.0

    summary: Dict[str, Dict[str, Dict[str, float]]] = {family: {} for family in families}
    for family in families:
        for rel_type in REL_TYPES:
            payload = totals[family][rel_type]
            count = max(payload.get("_count", 0.0), 1.0)
            summary[family][rel_type] = {
                "precision": round(payload["precision"] / count, 4),
                "recall": round(payload["recall"] / count, 4),
                "f1": round(payload["f1"] / count, 4),
            }

    return summary


def visualize_classwise_typed_metrics(
    classwise_metrics: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
    title: str,
) -> None:
    if not classwise_metrics:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Classwise metric figure skipped (missing library): {exc}")
        return

    families = [
        ("untyped_pair", "Untyped Pair", "#6b7280"),
        ("auto_pair", "Auto Pair", "#2563eb"),
        ("auto_triple", "Auto Triple", "#0ea5e9"),
        ("oracle_pair", "Oracle Pair", "#f97316"),
        ("oracle_triple", "Oracle Triple", "#ef4444"),
    ]
    metric_specs = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]
    x = np.arange(len(REL_TYPES), dtype=np.float32)
    width = 0.15

    fig, axes = plt.subplots(1, 3, figsize=(18.2, 6.6), sharey=True)
    fig.patch.set_facecolor("white")

    x_labels = [rel_type.replace("_", " ") for rel_type in REL_TYPES]
    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        ax.set_facecolor("white")
        for family_idx, (family_key, family_label, color) in enumerate(families):
            offsets = x + (family_idx - (len(families) - 1) / 2.0) * width
            values = [
                float(classwise_metrics.get(family_key, {}).get(rel_type, {}).get(metric_key, 0.0))
                for rel_type in REL_TYPES
            ]
            ax.bar(
                offsets,
                values,
                width=width,
                color=color,
                alpha=0.86,
                label=family_label if metric_key == "precision" else None,
            )

        ax.set_title(metric_label, color="#111827", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=22, ha="right", color="#4b5563")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8, alpha=0.85)
        ax.tick_params(axis="y", colors="#4b5563")
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")

    axes[0].set_ylabel("Score", color="#111827")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, color="#111827", fontsize=14, y=1.07)
    plt.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved classwise typed metrics figure: {out_path}")


def configure_runtime_context(
    dataset_name: str,
    data_file: Path,
    annot_file: Path,
    admission_id: str,
    patient_id: str,
    create_dir: bool = True,
) -> None:
    global DATA_FILE, ANNOT_FILE, ADMISSION_ID, TARGET_PATIENT, OUT_JSON, OUT_CSV, OUT_AUDIT, OUT_EMBEDDING, VIS_DIR

    DATA_FILE = data_file
    ANNOT_FILE = annot_file
    ADMISSION_ID = admission_id
    TARGET_PATIENT = patient_id

    if dataset_name == DEFAULT_DATASET_NAME and admission_id == DEFAULT_ADMISSION_ID:
        run_tag = admission_id
    else:
        run_tag = f"{dataset_name}_{admission_id}"

    VIS_DIR = BATCH_MATERIALIZATION_DIR / f"loki_run_{run_tag}"
    # Defer mkdir: in batch mode no per-admission file is written, so creating
    # the folder eagerly produces N empty directories. Callers that actually
    # plan to write per-admission artifacts (single-admission runs) leave
    # create_dir=True and any single-admission file writer can also call
    # VIS_DIR.mkdir on demand.
    if create_dir:
        VIS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON = VIS_DIR / f"materialized_joins_{run_tag}.json"
    OUT_CSV = VIS_DIR / f"materialized_table_{run_tag}.csv"
    OUT_AUDIT = VIS_DIR / f"cluster_audit_{run_tag}.md"
    OUT_EMBEDDING = VIS_DIR / f"embedding_space_{run_tag}.png"


def _infer_dataset_name_from_results_csv(results_csv_path: Path) -> str:
    stem = results_csv_path.stem
    prefix = "materialized_batch_results_"
    if stem.startswith(prefix):
        suffix = stem[len(prefix):].strip()
        if suffix:
            return suffix
    return DEFAULT_DATASET_NAME


def regenerate_batch_diagrams_from_results_csv(results_csv_path: Path) -> None:
    resolved_csv = Path(results_csv_path).expanduser().resolve()
    if not resolved_csv.exists():
        raise FileNotFoundError(f"Batch results CSV not found: {resolved_csv}")

    with open(resolved_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Batch results CSV is empty: {resolved_csv}")

    required_cluster_label_columns = {
        "cluster_label_macro_precision",
        "cluster_label_macro_recall",
        "cluster_label_macro_f1",
        "cluster_label_accuracy",
        "pair_average_precision",
        "raw_pair_cluster_purity",
        "cluster_ari",
        "cluster_silhouette",
    }
    header = set(rows[0].keys())
    missing_cluster_label_columns = sorted(required_cluster_label_columns - header)
    if missing_cluster_label_columns:
        missing_str = ", ".join(missing_cluster_label_columns)
        raise ValueError(
            "Batch results CSV is too old to regenerate the current relationship-clustering dashboard. "
            f"Missing macro dashboard columns: {missing_str}. "
            "These macro cluster/oracle scores are not present in the old aggregate CSV; rerun the batch to write a "
            "fresh materialized_batch_results_*.csv first."
        )

    dataset_name = ""
    for row in rows:
        dataset_name = str(row.get("dataset") or "").strip()
        if dataset_name:
            break
    if not dataset_name:
        dataset_name = _infer_dataset_name_from_results_csv(resolved_csv)

    out_dir = resolved_csv.parent
    metrics_dashboard_out = out_dir / f"materialized_batch_metrics_{dataset_name}.png"
    representation_dashboard_out = out_dir / f"materialized_batch_representation_dashboard_{dataset_name}.png"

    print("=" * 66)
    print("  LOKI - Regenerate Batch Diagrams From Results CSV")
    print(f"  CSV: {resolved_csv}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Admissions: {len(rows)}")
    print("=" * 66)

    visualize_batch_metric_overview(dataset_name, rows, metrics_dashboard_out)
    visualize_batch_representation_overview(dataset_name, rows, representation_dashboard_out)

    print("\n  Regenerated batch diagrams:")
    print(f"    {metrics_dashboard_out}")
    print(f"    {representation_dashboard_out}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOKI cross-table join path extraction and semantic materialization"
    )
    parser.set_defaults(
        run_all_admissions=True,
        llm_no_hdbscan=False,     # default to the current HDBSCAN-backed LMStudio separation profile
        llm_agglomerative=False,
        cluster_refine_semantic_subsplit=True,
        cluster_refine_llm_per_path_vote=True,
        cluster_refine_path_subsplit=True,
        no_shared_pair_merge=True,
        show_typed_metrics=True,  # always show per-type breakdown
    )
    parser.add_argument(
        "--dataset", type=str, choices=sorted(DATASET_CONFIGS), default=DEFAULT_DATASET_NAME,
        help=f"Dataset split to run (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--run_all_admissions", action="store_true",
        help="Run inference over every annotated admission in the selected dataset",
    )
    parser.add_argument(
        "--single_admission", dest="run_all_admissions", action="store_false",
        help="Run a single admission instead of the default full-dataset batch inference",
    )
    parser.add_argument(
        "--admission_id", type=str, default=None,
        help="Admission id for single-admission inference (defaults to the original paper example for mimic)",
    )
    parser.add_argument(
        "--patient_id", type=str, default=None,
        help="Patient id for single-admission inference when multiple records are available",
    )
    parser.add_argument(
        "--max_admissions", type=int, default=None,
        help="Optional cap for batch inference over the selected dataset",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Batch mode only. Resume from the saved batch CSV for the selected dataset and "
            "skip admissions that already completed."
        ),
    )
    parser.add_argument(
        "--evaluation_profile",
        type=str,
        choices=[EVALUATION_PROFILE_DEFAULT, EVALUATION_PROFILE_AE_DIS_CLEAN],
        default=EVALUATION_PROFILE_DEFAULT,
        help=(
            "Evaluation scope. 'default' scores the full annotation inventory. "
            f"'{EVALUATION_PROFILE_AE_DIS_CLEAN}' restricts evaluation to admissions that contain at least one clean "
            "ADVERSE_EFFECT-only pair and one clean DISCONTINUED-only pair, and scores only those clean pairs."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CKPT),
        help=f"Path to model.pt (default: {DEFAULT_CKPT})",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed gamma threshold (default: adaptive μ+σ, floor 0.15)",
    )
    parser.add_argument(
        "--adaptive_threshold_cap", type=float, default=0.295,
        help="Optional upper cap applied only to adaptive gamma; ignored when --threshold is set.",
    )
    parser.add_argument(
        "--adaptive_threshold_gap_limit", type=float, default=0.05,
        help="Optional max (p75_top1 - legacy μ+2σ) gap allowed before adaptive gamma capping is skipped; ignored when --threshold is set.",
    )
    parser.add_argument(
        "--adaptive_threshold_force_legacy_max", type=float, default=0.30,
        help="Optional legacy μ+2σ cutoff that still forces adaptive gamma capping even when the gap limit would skip it; ignored when --threshold is set.",
    )
    parser.add_argument(
        "--cluster_label_backend", type=str, choices=["gliner2", "keyword", "lmstudio", "oracle"], default="lmstudio",
        help=(
            "Cluster naming backend. gliner2 (default): anchored entity+relation scoring. "
            "lmstudio: local LLM via LMStudio OpenAI-compatible API (requires LMStudio running). "
            "keyword: fast lexical fallback."
        ),
    )
    parser.add_argument(
        "--gliner2_model", type=str, default=GLINER2_MODEL,
        help=f"GLiNER2 model for cluster labeling when --cluster_label_backend gliner2 (default: {GLINER2_MODEL})",
    )
    parser.add_argument(
        "--gliner2_batch_size", type=int, default=8,
        help="Batch size for GLiNER2 sentence-occurrence classification (default: 8)",
    )
    parser.add_argument(
        "--gliner2_threshold", type=float, default=0.5,
        help="Confidence threshold used by GLiNER2 classification (default: 0.5)",
    )
    parser.add_argument(
        "--gliner2_max_len", type=int, default=384,
        help="Maximum token length for each GLiNER2 sentence-occurrence input (default: 384)",
    )
    parser.add_argument(
        "--stage5_top_k", type=int, default=None,
        help="Row-side top-k used during Stage 5 extraction (default: recall-oriented auto setting)",
    )
    parser.add_argument(
        "--stage5_diag_row_top_k", type=int, default=None,
        help="Optional diagnosis-row top-k override for Stage 5 ablations (default: uses --stage5_top_k)",
    )
    parser.add_argument(
        "--stage5_med_row_top_k", type=int, default=None,
        help="Optional medication-row top-k override for Stage 5 ablations (default: uses --stage5_top_k)",
    )
    parser.add_argument(
        "--stage5_sent_diag_top_k", type=int, default=8,
        help="Sentence-side diagnosis cap during Stage 5 extraction (default: 8)",
    )
    parser.add_argument(
        "--stage5_sent_med_top_k", type=int, default=12,
        help="Sentence-side medication cap during Stage 5 extraction (default: 12)",
    )
    parser.add_argument(
        "--stage5_max_pairs_per_sentence", type=int, default=12,
        help="Maximum diagnosis-medication pairs contributed by one sentence (default: 12)",
    )
    parser.add_argument(
        "--stage5_max_sentences_per_pair", type=int, default=3,
        help="Maximum mediating sentences retained per diagnosis-medication pair (default: 3)",
    )
    parser.add_argument(
        "--no_pair_filter", action="store_true",
        help="Disable the post-recovery pair filter for weak hub-driven singleton pairs",
    )
    parser.add_argument(
        "--pair_filter_diag_top_k", type=int, default=8,
        help="Keep weak singleton pairs if they are within this diagnosis-row rank (default: 8)",
    )
    parser.add_argument(
        "--pair_filter_med_top_k", type=int, default=8,
        help="Keep weak singleton pairs if they are within this medication-row rank (default: 8)",
    )
    parser.add_argument(
        "--pair_filter_margin", type=float, default=0.03,
        help="Extra path-score margin above gamma that keeps a singleton pair (default: 0.03)",
    )
    parser.add_argument(
        "--pair_filter_hub_fanout", type=int, default=6,
        help="Sentence fanout threshold used by the weak singleton pair filter (default: 6)",
    )
    parser.add_argument(
        "--pair_filter_mode", type=str, choices=["legacy", "weak_only", "off"], default="legacy",
        help="Pair filter rollout mode for ablations (default: legacy)",
    )
    parser.add_argument(
        "--pair_connection_mode", type=str, choices=["legacy", "support_weighted"], default="legacy",
        help="How untyped pair connection strength is computed in experimental rollouts (default: legacy)",
    )
    parser.add_argument(
        "--low_signal_bundle_rescue", action="store_true",
        help="Experimental: enable the Phase E low-signal rescue bundle. This merges dropped singleton sibling bundles that share diagnosis row, mediator sentence, and predicted label, vetoes singleton TREATS children split from NEGATIVE-dominant parents, and can peel a methadone-style DISCONTINUED pair back out of NEGATIVE suppression when a pure retained DISCONTINUED refinement child provides same-medication support.",
    )
    parser.add_argument(
        "--sentence_specificity_alpha", type=float, default=0.0,
        help="Continuous sentence-specificity penalty strength for Stage 5 ablations (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--stage5_sentence_overflow_margin", type=float, default=0.0,
        help="Keep a bounded overflow band of near-tied rows per sentence when they fall within this raw-score margin of the sentence cap cutoff (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--stage5_sentence_overflow_limit", type=int, default=0,
        help="Maximum extra rows per sentence admitted by the Stage 5 overflow band (default: 0 = disabled)",
    )
    parser.add_argument(
        "--stage5_row_plateau_margin", type=float, default=0.0,
        help="Adaptive Stage 5 row-side sentence-cap expansion margin around the weighted row cutoff for plateaued rows (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--stage5_row_plateau_min_extra", type=int, default=0,
        help="Minimum number of extra row-side sentence candidates within the Stage 5 row plateau margin before adaptive row-side expansion triggers (default: 0 = disabled)",
    )
    parser.add_argument(
        "--stage5_row_plateau_max_extra", type=int, default=0,
        help="Maximum extra sentences admitted per row by adaptive Stage 5 row-side plateau expansion (default: 0 = disabled)",
    )
    parser.add_argument(
        "--stage5_sentence_plateau_margin", type=float, default=0.004,
        help="Adaptive Stage 5 sentence-side cap expansion margin around the raw-score cutoff for plateaued sentences (default: 0.004; set to 0.0 to disable)",
    )
    parser.add_argument(
        "--stage5_sentence_plateau_min_extra", type=int, default=8,
        help="Minimum number of extra rows within the Stage 5 sentence plateau margin before adaptive sentence-cap expansion triggers (default: 8)",
    )
    parser.add_argument(
        "--stage5_sentence_plateau_max_extra", type=int, default=12,
        help="Maximum extra rows per sentence admitted by adaptive Stage 5 sentence-side plateau expansion (default: 12; set to 0 to disable)",
    )
    parser.add_argument(
        "--stage5_stopcue_diag_sentence_top_k", type=int, default=0,
        help="Experimental: expand the diagnosis-side Stage 5 sentence cap to this value for explicit discontinue sentences only (default: 0 = disabled)",
    )
    parser.add_argument(
        "--stage5_pair_plateau_margin", type=float, default=0.002,
        help="Adaptive Stage 5 per-sentence pair-cap expansion margin around the path-score cutoff for plateaued sentences (default: 0.002; set to 0.0 to disable)",
    )
    parser.add_argument(
        "--stage5_pair_plateau_min_extra", type=int, default=8,
        help="Minimum number of extra sentence-local row pairs within the Stage 5 pair plateau margin before adaptive pair-cap expansion triggers (default: 8)",
    )
    parser.add_argument(
        "--stage5_pair_plateau_max_extra", type=int, default=12,
        help="Maximum extra diagnosis-medication pairs admitted by adaptive Stage 5 pair plateau expansion per sentence (default: 12; set to 0 to disable)",
    )
    parser.add_argument(
        "--stage5_threshold_rescue_margin", type=float, default=0.0,
        help="Rescue near-threshold diagnosis-medication pairs only when their sentence-local path scores land within this margin below gamma across multiple mediator sentences (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--stage5_threshold_rescue_min_sentences", type=int, default=2,
        help="Minimum number of unique mediator sentences required by the Stage 5 near-threshold rescue rule (default: 2)",
    )
    parser.add_argument(
        "--stage5_diag_row_sibling_rescue_margin", type=float, default=0.0,
        help="Rescue near-threshold diagnosis-medication pairs only when they stay within this margin below gamma, retain multi-sentence evidence, include at least one medication-anchored sentence, and share an allowed support sentence with an already-admitted pair on the same diagnosis row (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--stage5_med_row_stopcue_rescue_margin", type=float, default=0.0,
        help="Rescue near-threshold diagnosis-medication pairs only when they stay within this margin below gamma, retain multi-sentence medication-anchored evidence, contain explicit discontinue language, and share a support sentence with an already-admitted pair on the same medication row (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--section_priors_file", type=str, default="",
        help="Optional JSON file mapping section names to reliability priors for Stage 5/typing ablations",
    )
    parser.add_argument(
        "--pair_embedding_mode", type=str, choices=["signature", "semantic_signature", "contextual_sentence_average", "row_pair_hybrid"], default="contextual_sentence_average",
        help="Pair embedding representation used for clustering ablations (default: contextual_sentence_average - current best validated batch profile)",
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size", type=int, default=4,
        help="HDBSCAN min_cluster_size override (default: 4). Set 0 to auto-calibrate from pair/rel-type ratio.",
    )
    parser.add_argument(
        "--gliner2_label_input_mode", type=str, choices=["sentence_evidence", "semantic_signature"], default=DEFAULT_GLINER2_LABEL_INPUT_MODE,
        help="GLiNER2 cluster-naming input text mode; independent from --pair_embedding_mode (default: sentence_evidence)",
    )
    parser.add_argument(
        "--anchor_normalization_mode", type=str, choices=["legacy", "clinical_light"], default="legacy",
        help="Anchor normalization strategy for GLiNER2 evidence matching ablations (default: legacy)",
    )
    parser.add_argument(
        "--gliner2_per_sentence_vote", action="store_true",
        help="Enable hub-filtered per-sentence voting for GLiNER2 cluster labeling (reduces hub-sentence contamination, improves ADVERSE_EFFECT precision)",
    )
    parser.add_argument(
        "--gliner2_hub_fanout_threshold", type=float, default=0.3,
        help="Hub filter fraction: sentences mediating > max(2, threshold * n_cluster_pairs) global pairs are excluded from per-sentence voting (default: 0.3)",
    )
    parser.add_argument(
        "--gliner2_max_pool_sentences", type=int, default=12,
        help="Max paths to score per cluster in per_sentence_vote mode (default: 12)",
    )
    parser.add_argument(
        "--llm_base_url", type=str, default=LMSTUDIO_DEFAULT_BASE_URL,
        help=f"Local LLM server OpenAI-compatible API base URL (default: {LMSTUDIO_DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--llm_model", type=str, default=LMSTUDIO_DEFAULT_MODEL,
        help=f"Model identifier as shown in the local LLM server (default: {LMSTUDIO_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--llm_temperature", type=float, default=LMSTUDIO_DEFAULT_TEMPERATURE,
        help=f"Sampling temperature for LLM inference (default: {LMSTUDIO_DEFAULT_TEMPERATURE}; 0.0 = deterministic)",
    )
    parser.add_argument(
        "--llm_timeout", type=int, default=LMSTUDIO_DEFAULT_TIMEOUT_SECS,
        help=f"Per-request timeout in seconds for LLM API calls (default: {LMSTUDIO_DEFAULT_TIMEOUT_SECS})",
    )
    parser.add_argument(
        "--llm_retry_attempts", type=int, default=LMSTUDIO_DEFAULT_RETRY_ATTEMPTS,
        help=(
            "Number of LMStudio request attempts before giving up on a request "
            f"(default: {LMSTUDIO_DEFAULT_RETRY_ATTEMPTS}). In batch mode, exhausting this "
            "budget aborts the current admission and stops the batch; otherwise the pipeline "
            "falls back to keyword labeling."
        ),
    )
    parser.add_argument(
        "--llm_max_evidence_sents", type=int, default=LMSTUDIO_DEFAULT_MAX_EVIDENCE_SENTS,
        help=(
            "Max top-scored evidence sentences presented per cluster in per-cluster mode "
            f"(default: {LMSTUDIO_DEFAULT_MAX_EVIDENCE_SENTS} = all unique evidence sentences)"
        ),
    )
    parser.add_argument(
        "--llm_per_path_vote", action="store_true",
        help="Enable per-path LLM voting (one call per path, aggregated by path_score-weighted vote; slower but more fine-grained than the default per-cluster call)",
    )
    parser.add_argument(
        "--llm_no_agglomerative", dest="llm_agglomerative",
        action="store_false",
        help="Disable agglomerative re-labeling and use the faster per-cluster closed-label mode instead (default).",
    )
    parser.add_argument(
        "--llm_agglomerative", dest="llm_agglomerative",
        action="store_true",
        help="Enable agglomerative re-labeling in LMStudio mode instead of the default per-cluster closed-label mode.",
    )
    parser.add_argument(
        "--llm_agglom_distance", type=float, default=LMSTUDIO_DEFAULT_AGGLOM_DISTANCE,
        help=f"Cosine distance threshold for AgglomerativeClustering in free-form label space (default: {LMSTUDIO_DEFAULT_AGGLOM_DISTANCE}). Lower = more fine-grained groups.",
    )
    parser.add_argument(
        "--llm_path_vote", action="store_true",
        help="Per-path LLM mode (HDBSCAN-backed): each path gets its own closed-label LMStudio "
             "call; results are aggregated back to HDBSCAN clusters via path_score-weighted vote. "
             "Bypasses the 4-phase agglomerative pipeline while preserving HDBSCAN structural "
             "grouping.",
    )
    parser.add_argument(
        "--llm_no_hdbscan", action="store_true",
        help="Per-path LLM mode that completely bypasses HDBSCAN grouping: groups paths by "
             "(diag, med) pair identity instead. Each unique (diag_row_idx, med_row_idx) pair "
             "receives its own synthetic cluster with a label derived from per-path LLM votes. "
             "Generates a two-panel comparison visualisation (LLM groups vs HDBSCAN clusters). "
             "Use this to bypass the default HDBSCAN-backed cluster grouping.",
    )
    parser.add_argument(
        "--llm_hdbscan", dest="llm_no_hdbscan", action="store_false",
        help="Use HDBSCAN-backed cluster grouping for LLM labeling (default).",
    )
    parser.add_argument(
        "--cluster_refine_by_pair_label", dest="cluster_refine_by_pair_label",
        action="store_true", default=True,
        help="For HDBSCAN-backed runs, re-label pairs inside larger raw clusters and split mixed clusters by pair-level evidence label before low-signal filtering (default: enabled).",
    )
    parser.add_argument(
        "--no_cluster_refine_by_pair_label", dest="cluster_refine_by_pair_label",
        action="store_false",
        help="Disable the pair-label refinement split on HDBSCAN-backed runs.",
    )
    parser.add_argument(
        "--cluster_refine_min_pairs", type=int, default=5,
        help="Only attempt pair-label refinement on raw clusters with at least this many unique diagnosis-medication pairs (default: 5).",
    )
    parser.add_argument(
        "--cluster_refine_semantic_subsplit", dest="cluster_refine_semantic_subsplit", action="store_true",
        help="Experimental: after pair-label refinement, semantically subcluster same-label pair buckets using pair embeddings before rebuilding refined clusters (default: enabled).",
    )
    parser.add_argument(
        "--no_cluster_refine_semantic_subsplit", dest="cluster_refine_semantic_subsplit", action="store_false",
        help="Disable semantic same-label subclustering inside pair-label refinement.",
    )
    parser.add_argument(
        "--cluster_refine_semantic_distance", type=float, default=0.20,
        help="Cosine distance threshold for --cluster_refine_semantic_subsplit (default: 0.20). Lower = more fine-grained same-label splits.",
    )
    parser.add_argument(
        "--cluster_refine_llm_per_path_vote", dest="cluster_refine_llm_per_path_vote", action="store_true",
        help="Experimental: during LMStudio pair-label refinement, label each supporting path separately and aggregate votes per pair instead of one closed-label call per pair (default: enabled).",
    )
    parser.add_argument(
        "--no_cluster_refine_llm_per_path_vote", dest="cluster_refine_llm_per_path_vote", action="store_false",
        help="Disable LMStudio per-path voting inside pair-label refinement.",
    )
    parser.add_argument(
        "--cluster_refine_path_subsplit", dest="cluster_refine_path_subsplit", action="store_true",
        help="Experimental: after pair-level refinement, split mixed-evidence paths from the same diagnosis-medication pair into separate refined child clusters using per-path LMStudio labels (default: enabled). Requires --cluster_refine_llm_per_path_vote.",
    )
    parser.add_argument(
        "--no_cluster_refine_path_subsplit", dest="cluster_refine_path_subsplit", action="store_false",
        help="Disable same-pair path-level refinement splitting.",
    )
    parser.add_argument(
        "--cluster_refine_path_subsplit_min_mass", type=float, default=0.25,
        help="Minimum total per-path vote mass required for a same-pair path-split child to survive refinement (default: 0.25). Higher = more conservative path splitting.",
    )
    parser.add_argument(
        "--cluster_refine_path_subsplit_min_share", type=float, default=0.30,
        help="Minimum within-pair vote share that can also justify a same-pair path-split child (default: 0.30). Higher = more conservative path splitting.",
    )
    parser.add_argument(
        "--cluster_refine_path_subsplit_max_gap", type=float, default=0.12,
        help="Maximum dominant-minus-child vote-mass gap allowed for a same-pair path-split child (default: 0.12). Lower = only near-tied evidence gets split.",
    )
    parser.add_argument(
        "--suppress_negative_clusters", dest="suppress_negative_clusters",
        action="store_true", default=True,
        help="For HDBSCAN-backed LMStudio runs, drop NEGATIVE-labeled clusters from materialized output after pair-label refinement. If annotated negative pairs are available, clusters containing them are kept (default: enabled).",
    )
    parser.add_argument(
        "--keep_negative_clusters", dest="suppress_negative_clusters",
        action="store_false",
        help="Keep NEGATIVE-labeled HDBSCAN clusters in the materialized output.",
    )
    parser.add_argument(
        "--llm_agglom_encoder", type=str, default="bge",
        choices=["medembed", "bge", "minilm"],
        help="Sentence encoder for agglom Phase 2 embedding of free-form phrases. "
             "'bge' = BAAI/bge-large-en-v1.5 (default; best empirical results). "
             "'medembed' = loaded LOKI model. "
             "'minilm' = sentence-transformers/all-MiniLM-L6-v2.",
    )
    parser.add_argument(
        "--show_typed_metrics", action="store_true",
        help="Print label-based typed metrics in addition to the clustering-first report (default: enabled)",
    )
    parser.add_argument(
        "--no_typed_metrics", dest="show_typed_metrics", action="store_false",
        help="Suppress the per-type metrics table (opt-out from the default --show_typed_metrics)",
    )
    parser.add_argument(
        "--no_cluster_tail_filter", action="store_true",
        help="Disable the post-clustering filter that trims low-score tail pairs inside large raw clusters",
    )
    parser.add_argument(
        "--cluster_tail_keep_rank", type=int, default=2,
        help="Always keep at least this many top-scoring pairs per raw cluster (default: 2)",
    )
    parser.add_argument(
        "--cluster_tail_margin", type=float, default=0.01,
        help="Keep pairs whose score is within this margin of the raw-cluster leader score (default: 0.01)",
    )
    parser.add_argument(
        "--cluster_tail_mode", type=str,
        choices=["legacy", "conservative", "soft_weight", "adaptive_std", "adaptive_percentile"],
        default="adaptive_std",
        help="Cluster-tail filtering mode (default: adaptive_std). adaptive_std scales margin to per-cluster score std; adaptive_percentile drops below pN of cluster score distribution.",
    )
    parser.add_argument(
        "--cluster_tail_adaptive_lambda", type=float, default=0.5,
        help="Lambda multiplier for adaptive_std mode: margin = max(0.005, lambda * std(cluster_scores)) (default: 0.5)",
    )
    parser.add_argument(
        "--cluster_tail_adaptive_percentile", type=float, default=25.0,
        help="Percentile floor for adaptive_percentile mode: drop pairs below this percentile of cluster score distribution (default: 25.0)",
    )
    parser.add_argument(
        "--no_rescue_unique_evidence", action="store_true",
        help="Disable unique-evidence rescue: by default, tail-dropped pairs that introduce an evidence sentence not covered by any kept pair are rescued to preserve sentence diversity.",
    )
    parser.add_argument(
        "--no_shared_pair_merge", dest="no_shared_pair_merge", action="store_true",
        help="Disable shared-pair must-link cluster merging (default).",
    )
    parser.add_argument(
        "--shared_pair_merge", dest="no_shared_pair_merge", action="store_false",
        help="Enable shared-pair must-link cluster merging for HDBSCAN clusters that share a diagnosis-medication row pair.",
    )
    parser.add_argument(
        "--max_clusters", type=int, default=0,
        help="Post-hoc cap: if cluster count exceeds this, merge down to this many via hierarchical meta-clustering (default: 0 = disabled)",
    )
    parser.add_argument(
        "--enable_meta_clustering", action="store_true",
        help="Enable hierarchical meta-clustering to merge HDBSCAN fragment clusters by Jaccard sentence/pair overlap",
    )
    parser.add_argument(
        "--meta_cluster_n", type=int, default=0,
        help="Target number of meta-clusters (default: 0 = silhouette-optimal cut)",
    )
    parser.add_argument(
        "--meta_cluster_alpha_sent", type=float, default=0.6,
        help="Sentence Jaccard distance weight in meta-cluster distance (default: 0.6)",
    )
    parser.add_argument(
        "--meta_cluster_alpha_pair", type=float, default=0.4,
        help="Pair Jaccard distance weight in meta-cluster distance (default: 0.4)",
    )
    parser.add_argument(
        "--enable_pair_recovery_diagnostics", action="store_true",
        help="Record pair-recovery diagnostics and failure-stage metadata for debugging recall",
    )
    parser.add_argument(
        "--debug_recall_cascade", action="store_true",
        help="Print verbose pair-recovery cascade diagnostics during inference",
    )
    parser.add_argument(
        "--diagnostics_output_dir", type=str, default="",
        help="Optional directory for writing recall and debugging diagnostic artifacts",
    )
    parser.add_argument(
        "--skip_visualizations", action="store_true",
        help="Skip single-admission visualization plot generation (PNG and sibling PDF); useful for faster repeated comparisons",
    )
    parser.add_argument(
        "--topic_map_max_clusters", type=int, default=8,
        help="Maximum number of relation-diverse retained clusters to draw in the join-topic map; the default selection seeds GT-backed relation types before filling remaining slots by cluster rank (default: 8)",
    )
    parser.add_argument(
        "--topic_map_label_top_k", type=int, default=8,
        help="Number of drawn clusters that receive text labels in the join-topic map (default: 8)",
    )
    parser.add_argument(
        "--topic_map_cluster_ids", type=str, default="",
        help="Comma-separated retained cluster ids to draw in the join-topic map; overrides --topic_map_max_clusters",
    )
    parser.add_argument(
        "--hide_topic_map_cluster_numbers", action="store_true",
        help="Hide the Cxx cluster id prefix in join-topic map labels",
    )
    parser.add_argument(
        "--topic_map_triples_per_label", type=int, default=4,
        help="Number of single triples to show in each join-topic map label before collapsing the remainder (default: 4)",
    )
    parser.set_defaults(batch_projection=False)
    parser.add_argument(
        "--batch_projection", dest="batch_projection", action="store_true",
        help="Enable the optional sampled dataset-level PCA/t-SNE/UMAP pair-embedding projection figures during batch inference",
    )
    parser.add_argument(
        "--no_batch_projection", dest="batch_projection", action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch_projection_points_per_type", type=int, default=300,
        help="Maximum sampled GT-backed predicted pairs per relation type in the batch projection figure (default: 300)",
    )
    parser.add_argument(
        "--batch_projection_max_points", type=int, default=1800,
        help="Global cap on sampled GT-backed predicted pairs in the batch projection figure (default: 1800)",
    )
    parser.add_argument(
        "--batch_progress_every", type=int, default=25,
        help="How often to print batch progress updates (default: every 25 admissions)",
    )
    parser.add_argument(
        "--regenerate_batch_diagrams_from_results_csv",
        type=str,
        default="",
        help="Skip inference and regenerate the CSV-backed batch dashboard plots (PNG plus sibling PDF) from an existing materialized_batch_results_*.csv file.",
    )

    # -- Cross-encoder rerank (Phase D.5, Option C) - on by default ------------
    parser.add_argument(
        "--use_cross_encoder", dest="use_cross_encoder",
        action="store_true", default=True,
        help="Run cross-encoder per-pair sentence rerank (Option C). Writes ce_score onto every "
             "surviving path; downstream signature/cluster-prompt builders are CE-aware. "
             "Enabled by default - improves typed pair F1 on admission 20393363 by ~+0.03.",
    )
    parser.add_argument(
        "--no_cross_encoder", dest="use_cross_encoder", action="store_false",
        help="Disable the Phase D.5 cross-encoder rerank (opt-out of the default --use_cross_encoder).",
    )
    parser.add_argument(
        "--cross_encoder_model", type=str,
        default="Alibaba-NLP/gte-reranker-modernbert-base",
           help="HF cross-encoder model name or local snapshot path (default: Alibaba-NLP/gte-reranker-modernbert-base; "
               "the bundled local cache is preferred automatically when present). Any sentence-transformers "
               "CrossEncoder-compatible checkpoint works.",
    )
    parser.add_argument(
        "--cross_encoder_device", type=str, default="",
        help="Override device for the cross-encoder (e.g. cuda, cpu). Empty = auto.",
    )
    parser.add_argument(
        "--cross_encoder_batch_size", type=int, default=32,
        help="Cross-encoder scoring batch size (default: 32)",
    )
    parser.add_argument(
        "--cross_encoder_max_length", type=int, default=512,
        help="Cross-encoder max sequence length (default: 512)",
    )
    parser.add_argument(
        "--cross_encoder_no_fp16", action="store_true",
        help="Disable fp16 inference for the cross-encoder",
    )
    parser.add_argument(
        "--cross_encoder_no_section_prefix", action="store_true",
        help="Do not prepend [section_name] to the passage when scoring",
    )
    parser.add_argument(
        "--cross_encoder_no_normalize", action="store_true",
        help="Return raw logits instead of sigmoid-normalized scores",
    )

    # -- Option D - CE-based pair-level filter (after Phase D.5) --------------
    parser.add_argument(
        "--ce_pair_filter_mode", type=str,
        choices=["off", "absolute", "quantile", "combined"],
        default="combined",
        help="Option D pair-level filter. 'absolute' drops pairs whose max ce_score is below "
             "--ce_pair_filter_threshold. 'quantile' drops pairs whose max ce_score is in the "
             "bottom --ce_pair_filter_quantile of the CE distribution. 'combined' (default; "
             "recommended) drops a pair only if BOTH its max LOKI path_score AND its max ce_score "
             "are in the bottom --ce_pair_filter_quantile of their respective distributions "
             "(conservative; never drops pairs that look strong by either signal). Requires "
             "--use_cross_encoder. Pass 'off' to disable.",
    )
    parser.add_argument(
        "--ce_pair_filter_threshold", type=float, default=0.05,
        help="Absolute ce_score threshold for --ce_pair_filter_mode=absolute (default: 0.05; assumes "
             "sigmoid-normalized CE scores in [0, 1]).",
    )
    parser.add_argument(
        "--ce_pair_filter_quantile", type=float, default=0.25,
        help="Quantile floor for --ce_pair_filter_mode=quantile|combined (default: 0.25 = drop "
             "pairs in the bottom 25%% of the CE/LOKI score distributions).",
    )

    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.regenerate_batch_diagrams_from_results_csv:
        regenerate_batch_diagrams_from_results_csv(Path(cli.regenerate_batch_diagrams_from_results_csv))
        return

    if cli.resume and not cli.run_all_admissions:
        raise ValueError("--resume is only supported with --run_all_admissions.")

    _configure_lmstudio_runtime(
        fail_closed=bool(cli.run_all_admissions),
        retry_attempts=max(int(cli.llm_retry_attempts), 1),
    )

    _bootstrap_hf_assets_for_run(cli)

    resolved_evaluation_profile = _normalize_evaluation_profile(cli.evaluation_profile)
    data_file, annot_file = get_dataset_paths(cli.dataset)
    output_dataset_name = _resolved_output_dataset_name(cli.dataset, resolved_evaluation_profile)
    admission_index = load_dataset_examples(data_file)
    annotation_entries = load_annotation_entries(annot_file)
    active_rel_types, _rel_type_sources = _resolve_rel_types_from_annotation_corpus(annotation_paths=[annot_file])

    if cli.run_all_admissions:
        n_eligible_before_eval_filter = 0
        eligible_admission_ids = [
            admission_id
            for admission_id in sorted(annotation_entries)
            if admission_id in admission_index
            and admission_index[admission_id].get("diagnosis_example") is not None
            and admission_index[admission_id].get("medication_example") is not None
        ]
        n_eligible_before_eval_filter = len(eligible_admission_ids)
        if resolved_evaluation_profile != EVALUATION_PROFILE_DEFAULT:
            eligible_admission_ids = [
                admission_id
                for admission_id in eligible_admission_ids
                if _admission_matches_evaluation_profile(
                    annotation_entries[admission_id],
                    resolved_evaluation_profile,
                )
            ]
        if cli.max_admissions is not None:
            eligible_admission_ids = eligible_admission_ids[:cli.max_admissions]

        print("=" * 66)
        print("  LOKI - Batch Cross-Table Materialization")
        print(f"  Dataset: {cli.dataset}  |  Device: {DEVICE}")
        if resolved_evaluation_profile != EVALUATION_PROFILE_DEFAULT:
            print(f"  Evaluation profile: {resolved_evaluation_profile}")
        print(f"  Admissions: {len(eligible_admission_ids)}")
        print(f"  Active relationship types: {', '.join(active_rel_types)}")
        print("=" * 66)

        print("\n-- Phase A: Dataset Indexing -------------------------------------")
        print(f"  Data file : {data_file}")
        print(f"  Annot file: {annot_file}")
        print(f"  Indexed admissions with tables: {len(admission_index)}")
        print(f"  Eligible annotated admissions : {n_eligible_before_eval_filter}")
        if resolved_evaluation_profile != EVALUATION_PROFILE_DEFAULT:
            print(f"  Profile-qualified admissions : {len(eligible_admission_ids)}")

        print("\n-- Phase B: Model Loading ----------------------------------------")
        model_args = load_model_args()
        model = build_model(model_args)
        model.to(DEVICE)
        model.eval()
        load_checkpoint(model, Path(cli.checkpoint))
        params = model.count_parameters()
        print(f"  Parameters: {params['total_parameters']:,} total, {params['trainable_parameters']:,} trainable")

        batch_rows: List[Dict[str, object]] = []
        batch_metrics_payloads: List[Dict[str, object]] = []
        projection_points: List[Dict[str, object]] = []
        semantic_projection_records: List[Dict[str, object]] = []
        batch_resume_records: List[Dict[str, Any]] = []
        # VLDB-grade aggregate-plot inputs collected during the batch loop:
        #   pipeline_funnel_rows: per-admission pair counts at each pipeline
        #     gate (Stage-5 -> pair filter -> CE filter -> cluster-tail -> GT).
        #   pair_label_records:   (predicted_label, gt_label) for every
        #     materialised GT-matched pair across the corpus, for the
        #     dataset-scale relation-label confusion matrix.
        #   cluster_label_records: (predicted_label, gt_label) for every
        #     evaluated final cluster across the corpus, for the dataset-scale
        #     cluster-label confusion matrix.
        pipeline_funnel_rows: List[Dict[str, object]] = []
        pair_label_records: List[Dict[str, object]] = []
        cluster_label_records: List[Dict[str, object]] = []
        failed_admissions: List[Dict[str, str]] = []
        completed_admission_ids: Set[str] = set()
        can_render_stateful_batch_artifacts = True
        can_render_projection_outputs = True
        progress_every = max(1, cli.batch_progress_every)
        sentence_encoder = getattr(model, "sentence_encoder", None)

        if cli.resume:
            batch_rows = _load_saved_batch_rows(output_dataset_name)
            completed_admission_ids = {
                str(row.get("admission_id", "")).strip()
                for row in batch_rows
                if str(row.get("admission_id", "")).strip()
            }
            batch_resume_records = load_batch_resume_state(output_dataset_name)
            (
                can_render_stateful_batch_artifacts,
                restored_metrics_payloads,
                restored_pipeline_funnel_rows,
                restored_pair_label_records,
                restored_cluster_label_records,
            ) = _restore_batch_resume_artifacts(batch_rows, batch_resume_records)
            batch_metrics_payloads.extend(restored_metrics_payloads)
            pipeline_funnel_rows.extend(restored_pipeline_funnel_rows)
            pair_label_records.extend(restored_pair_label_records)
            cluster_label_records.extend(restored_cluster_label_records)
            eligible_admission_ids = [
                admission_id
                for admission_id in eligible_admission_ids
                if admission_id not in completed_admission_ids
            ]
            can_render_projection_outputs = not completed_admission_ids

        # In batch mode we only want the aggregate dataset-level plots (produced
        # below from the collected batch_rows / projection_points); per-admission
        # PNGs are noisy, slow, and produce ~13 files * N_admissions = thousands
        # of images. Force-skip them here regardless of the CLI flag. The
        # aggregate batch visualizations are gated separately on
        # --batch_projection; the main dashboards stay enabled either way.
        if not cli.skip_visualizations:
            print("  (Per-admission visualizations skipped in batch mode; only dataset-level batch plots are produced.)")
            cli.skip_visualizations = True

        print("\n-- Phase C-F: Batch Inference ------------------------------------")
        if cli.resume:
            print(
                f"  Resume mode: loaded {len(completed_admission_ids)} completed admissions from "
                f"{_batch_results_csv_path(output_dataset_name)}"
            )
            print(f"  Remaining admissions      : {len(eligible_admission_ids)}")
            if not can_render_stateful_batch_artifacts:
                print(
                    "  Resume note: existing results predate resume-state tracking; CSV-backed summaries will "
                    "still update, but stateful batch figures will be skipped for this resumed run."
                )
            if cli.batch_projection and not can_render_projection_outputs:
                print(
                    "  Resume note: batch projection plots are skipped on resumed runs because historical "
                    "projection embeddings are not reconstructed from the saved CSV."
                )

        interrupted_lmstudio_error: Optional[LMStudioUnavailableError] = None
        interrupted_admission_id: Optional[str] = None
        for index, admission_id in enumerate(eligible_admission_ids, start=1):
            record = admission_index[admission_id]
            patient_id = str(record.get("patient_id", ""))
            annotation_entry = annotation_entries[admission_id]
            # Refresh per-admission output context so VIS_DIR / OUT_JSON / OUT_CSV /
            # OUT_AUDIT point at this admission's own folder. Pass create_dir=False
            # because batch mode never writes per-admission artifacts (per-admission
            # viz is force-skipped, JSON/CSV/audit are only written in single-admission
            # runs) - eager mkdir would litter 382 empty loki_run_<id> folders.
            configure_runtime_context(
                output_dataset_name, data_file, annot_file, admission_id, patient_id,
                create_dir=False,
            )
            diag_rows, med_rows, sent_texts, sent_meta = load_admission_data_from_examples(record)
            gt_relationships, gt_diag, gt_med, multi_pairs = load_ground_truth_for_admission(
                admission_id,
                annotation_entries,
                resolve_rel_inventory=False,
            )
            negative_pairs = _extract_negative_pairs_from_annotation_entry(annotation_entry)
            evaluation_target_pairs: Optional[Set[Tuple[int, int]]] = None
            if resolved_evaluation_profile == EVALUATION_PROFILE_AE_DIS_CLEAN:
                evaluation_target_pairs, evaluation_stats = _clean_ae_dis_target_pairs_from_annotation_entry(annotation_entry)
                if not (evaluation_stats["has_clean_ae"] and evaluation_stats["has_clean_dis"]):
                    raise ValueError(
                        f"Admission {admission_id} does not satisfy evaluation profile {resolved_evaluation_profile}"
                    )
                gt_relationships, gt_diag, gt_med, multi_pairs = _filter_ground_truth_for_target_pairs(
                    gt_relationships,
                    gt_diag,
                    gt_med,
                    multi_pairs,
                    evaluation_target_pairs,
                )

            started = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_materialization_pipeline(
                        cli,
                        model,
                        model_args,
                        diag_rows,
                        med_rows,
                        sent_texts,
                        sent_meta,
                        gt_relationships,
                        gt_diag,
                        gt_med,
                        multi_pairs,
                        negative_pairs=negative_pairs,
                        evaluation_target_pairs=evaluation_target_pairs,
                        evaluation_profile=resolved_evaluation_profile,
                    )
            except LMStudioUnavailableError as exc:
                interrupted_lmstudio_error = exc
                interrupted_admission_id = admission_id
                failed_admissions.append({
                    "admission_id": admission_id,
                    "patient_id": patient_id,
                    "error": str(exc),
                })
                failed_path = save_batch_failures(output_dataset_name, failed_admissions)
                print(f"  [{index}/{len(eligible_admission_ids)}] admission {admission_id} interrupted: {exc}")
                print(f"  Saved batch failures CSV: {failed_path}")
                break
            except Exception as exc:
                failed_admissions.append({
                    "admission_id": admission_id,
                    "patient_id": patient_id,
                    "error": str(exc),
                })
                save_batch_failures(output_dataset_name, failed_admissions)
                print(f"  [{index}/{len(eligible_admission_ids)}] admission {admission_id} failed: {exc}")
                continue

            runtime_sec = time.perf_counter() - started
            row = build_batch_result_row(
                output_dataset_name,
                admission_id,
                patient_id,
                runtime_sec,
                diag_rows,
                med_rows,
                sent_texts,
                result,
            )
            batch_rows.append(row)
            batch_metrics_payloads.append(result["metrics"])
            reporting_paths = (
                result["paths"]
                if evaluation_target_pairs is None
                else (_filter_paths_for_target_pairs(result["paths"], evaluation_target_pairs) or [])
            )

            # -- Capture corpus-level inputs for VLDB aggregate plots --------
            _m = result["metrics"]
            _pair_filter = _m.get("pair_filter") or {}
            _ce_filter = result.get("ce_pair_filter_stats") or {}
            _stage5_candidate_pairs = int(_pair_filter.get("n_pairs_before") or 0)
            _after_pair_filter_pairs = int(_pair_filter.get("n_pairs_after") or _stage5_candidate_pairs)
            if _ce_filter.get("enabled"):
                _after_ce_pairs = int(_ce_filter.get("n_pairs_after") or _after_pair_filter_pairs)
            else:
                _after_ce_pairs = _after_pair_filter_pairs
            _after_cluster_tail_pairs = int(row.get("n_pred_pairs") or 0)
            admission_pipeline_funnel_row = {
                "admission_id": admission_id,
                "stage5_candidate_pairs": _stage5_candidate_pairs,
                "after_pair_filter_pairs": _after_pair_filter_pairs,
                "after_ce_filter_pairs": _after_ce_pairs,
                "after_cluster_tail_pairs": _after_cluster_tail_pairs,
                "gt_pairs_recovered": int(row.get("gt_pairs_recovered") or 0),
                "n_gt_pairs": int(row.get("n_gt_pairs") or 0),
            }
            pipeline_funnel_rows.append(admission_pipeline_funnel_row)

            # Per-pair (predicted_label, gt_label) collection for the
            # dataset-scale confusion matrix. We keep only GT-matched pairs so
            # the matrix reflects label confusion, not unmatched false positives.
            admission_pair_label_records = _build_batch_pair_label_records(
                admission_id,
                reporting_paths,
                gt_relationships,
                multi_pairs,
            )
            pair_label_records.extend(admission_pair_label_records)

            admission_cluster_label_records: List[Dict[str, object]] = []
            cluster_label_metrics = (result.get("metrics", {}) or {}).get("cluster_label", {}) or {}
            cluster_assignments = cluster_label_metrics.get("assignments", []) or []
            if isinstance(cluster_assignments, list):
                for assignment in cluster_assignments:
                    if not isinstance(assignment, dict):
                        continue
                    oracle_type = _normalize_rel_type(str(assignment.get("oracle_type", "") or ""))
                    if not oracle_type:
                        continue
                    predicted_type = _normalize_rel_type(str(assignment.get("predicted_type", "") or ""))
                    cluster_label_record = {
                        "admission_id": admission_id,
                        "cluster_id": assignment.get("cluster_id"),
                        "predicted_label": predicted_type,
                        "gt_label": oracle_type,
                        "correct": bool(assignment.get("correct")),
                    }
                    admission_cluster_label_records.append(cluster_label_record)
                    cluster_label_records.append(cluster_label_record)

            _upsert_batch_resume_record(
                batch_resume_records,
                {
                    "admission_id": admission_id,
                    "patient_id": patient_id,
                    "batch_row": row,
                    "metrics_payload": {
                        "classwise_typed_metrics": result["metrics"].get("classwise_typed_metrics", {}),
                        "stage5_config": result["metrics"].get("stage5_config", {}),
                    },
                    "pipeline_funnel_row": admission_pipeline_funnel_row,
                    "pair_label_records": admission_pair_label_records,
                    "cluster_label_records": admission_cluster_label_records,
                },
            )
            save_batch_resume_state(output_dataset_name, batch_resume_records)
            save_batch_results(output_dataset_name, batch_rows)

            if cli.batch_projection:
                projection_points.extend(
                    collect_dataset_projection_points(
                        admission_id,
                        reporting_paths,
                        result["refined_rows"],
                        result["n_diag"],
                        result["refined_sentences"],
                        gt_relationships,
                        sentence_encoder,
                        pair_embedding_mode=cli.pair_embedding_mode,
                    )
                )
                semantic_projection_records.extend(
                    collect_dataset_semantic_projection_records(
                        admission_id,
                        reporting_paths,
                        result["refined_rows"],
                        result["n_diag"],
                        result["refined_sentences"],
                        gt_relationships,
                        sentence_encoder,
                        pair_embedding_mode=cli.pair_embedding_mode,
                        negative_pairs=negative_pairs,
                    )
                )

            if index == 1 or index % progress_every == 0 or index == len(eligible_admission_ids):
                ap_value = row["pair_average_precision"]
                ap_str = f"{float(ap_value):.3f}" if ap_value is not None else "n/a"
                # relaxed pair precision / recall (may be missing for some admissions)
                relaxed_p = row.get("relaxed_pair_precision")
                relaxed_r = row.get("relaxed_pair_recall")
                p_str = f"{float(relaxed_p):.3f}" if relaxed_p is not None else "n/a"
                r_str = f"{float(relaxed_r):.3f}" if relaxed_r is not None else "n/a"
                gt_pair_bits = ""
                if row.get("gt_pairs_recovered") is not None:
                    gt_pair_bits = (
                        f"  GTpairs={int(row['gt_pairs_recovered'])}/{int(row['n_gt_pairs'])}"
                        f"  STK={int(row.get('gt_fail_sentence_side_top_k') or 0)}"
                        f"  THR={int(row.get('gt_fail_transitive_join_threshold') or 0)}"
                    )
                print(
                    f"  [{index}/{len(eligible_admission_ids)}] admission {admission_id}  "
                    f"P={p_str}  R={r_str}  F1={float(row.get('relaxed_pair_f1', 0.0)):.3f}  AP={ap_str}  "
                    f"clusters={int(row['n_final_clusters'])}{gt_pair_bits}"
                )

        if interrupted_lmstudio_error is not None:
            if batch_rows:
                save_batch_results(output_dataset_name, batch_rows)
            print("\n-- Batch Interrupted -------------------------------------------")
            if batch_rows:
                print(f"  Completed admissions kept : {len(batch_rows)}")
                print(f"  Results CSV              : {_batch_results_csv_path(output_dataset_name)}")
                print(f"  Summary CSV              : {_batch_summary_csv_path(output_dataset_name)}")
                print(f"  Report MD                : {_batch_report_md_path(output_dataset_name)}")
            else:
                print("  Completed admissions kept : 0")
            print(f"  Failure log              : {_batch_failures_csv_path(output_dataset_name)}")
            if interrupted_admission_id is not None:
                print(f"  Evaluation stopped before admission {interrupted_admission_id} completed.")
            print("  Fix LM Studio connectivity, then rerun the same batch command with --resume.")
            print("  --resume reloads the existing batch CSV and skips already completed admissions.")
            return

        if not batch_rows:
            raise RuntimeError("Batch inference did not complete successfully for any admission.")

        save_batch_results(output_dataset_name, batch_rows)
        batch_vis_dir = _batch_materialization_dir(output_dataset_name)
        batch_vis_dir.mkdir(parents=True, exist_ok=True)
        metrics_dashboard_out = batch_vis_dir / f"materialized_batch_metrics_{output_dataset_name}.png"
        visualize_batch_metric_overview(output_dataset_name, batch_rows, metrics_dashboard_out)
        representation_dashboard_out = batch_vis_dir / f"materialized_batch_representation_dashboard_{output_dataset_name}.png"
        visualize_batch_representation_overview(output_dataset_name, batch_rows, representation_dashboard_out)
        if can_render_stateful_batch_artifacts:
            batch_classwise_summary = summarize_batch_classwise_typed_metrics(batch_metrics_payloads)
            _print_classwise_typed_metric_table(
                batch_classwise_summary,
                title="  Mean classwise typed metrics across admissions:",
            )
            batch_classwise_out = batch_vis_dir / f"materialized_batch_classwise_metrics_{output_dataset_name}.png"
            visualize_classwise_typed_metrics(
                batch_classwise_summary,
                batch_classwise_out,
                title=(
                    f"LOKI Batch Classwise Typed Metrics - {output_dataset_name}\n"
                    f"Averaged over {len(batch_rows)} admissions"
                ),
            )

            # -- VLDB-grade aggregate plots ----------------------------------
            funnel_out = batch_vis_dir / f"materialized_batch_pipeline_funnel_{output_dataset_name}.png"
            visualize_batch_pipeline_funnel(output_dataset_name, pipeline_funnel_rows, funnel_out)
            gamma_f1_out = batch_vis_dir / f"materialized_batch_gamma_vs_f1_{output_dataset_name}.png"
            visualize_batch_gamma_vs_f1(output_dataset_name, batch_rows, batch_metrics_payloads, gamma_f1_out)
            confusion_out = batch_vis_dir / f"materialized_batch_confusion_matrix_{output_dataset_name}.png"
            visualize_batch_confusion_matrix(output_dataset_name, pair_label_records, confusion_out, normalize=True)
            cluster_confusion_out = batch_vis_dir / f"materialized_batch_cluster_confusion_matrix_{output_dataset_name}.png"
            visualize_batch_cluster_confusion_matrix(
                output_dataset_name,
                cluster_label_records,
                cluster_confusion_out,
                normalize=True,
            )
        else:
            print(
                "  Skipped classwise/confusion/funnel/gamma batch diagrams because resume-state data is "
                "missing for one or more already completed admissions."
            )

        if failed_admissions:
            failed_path = save_batch_failures(output_dataset_name, failed_admissions)
            print(f"  Saved batch failures CSV: {failed_path}")
        else:
            _clear_batch_failures_file(output_dataset_name)

        if cli.batch_projection:
            if can_render_projection_outputs:
                projection_out = batch_vis_dir / f"materialized_batch_projections_{output_dataset_name}.png"
                visualize_dataset_projection_benchmark(
                    output_dataset_name,
                    projection_points,
                    projection_out,
                    max_points_per_type=cli.batch_projection_points_per_type,
                    max_total_points=cli.batch_projection_max_points,
                )
                projection_3d_out = batch_vis_dir / f"materialized_batch_projections_3d_{output_dataset_name}.png"
                visualize_dataset_projection_benchmark_3d(
                    output_dataset_name,
                    projection_points,
                    projection_3d_out,
                    max_points_per_type=cli.batch_projection_points_per_type,
                    max_total_points=cli.batch_projection_max_points,
                )
                semantic_projection_out = batch_vis_dir / f"materialized_batch_semantic_projection_{output_dataset_name}.png"
                visualize_dataset_semantic_projection(
                    output_dataset_name,
                    semantic_projection_records,
                    semantic_projection_out,
                    max_points_per_type=cli.batch_projection_points_per_type,
                    max_total_points=cli.batch_projection_max_points,
                )
            else:
                print(
                    "  Skipped batch projection plots because resumed runs do not reconstruct historical "
                    "projection embeddings from the saved CSV."
                )

        summary = summarize_batch_rows(output_dataset_name, batch_rows)
        print("\n-- Batch Summary ------------------------------------------------")
        print(f"  Mean pair average precision: {summary['averages']['pair_average_precision']}")
        print(f"  Mean relaxed pair F1      : {summary['averages']['relaxed_pair_f1']}")
        print(f"  Mean exact triple F1      : {summary['averages']['exact_triple_f1']}")
        print(
            "  Mean cluster macro       : "
            f"P={summary['averages']['cluster_label_macro_precision']}  "
            f"R={summary['averages']['cluster_label_macro_recall']}  "
            f"F1={summary['averages']['cluster_label_macro_f1']}"
        )
        print(
            "  Mean oracle macro        : "
            f"P={summary['averages']['oracle_macro_precision']}  "
            f"R={summary['averages']['oracle_macro_recall']}  "
            f"F1={summary['averages']['oracle_macro_f1']}"
        )
        _ari = summary["averages"].get("cluster_ari")
        print(f"  Mean cluster ARI          : {_ari if _ari is not None else 'N/A'}")
        _sil = summary["averages"].get("cluster_silhouette")
        print(f"  Mean cluster silhouette   : {_sil if _sil is not None else 'N/A'}")
        mean_stage_timers = [
            (key, summary["averages"].get(key))
            for key in sorted(summary["averages"].keys())
            if isinstance(key, str) and (key == "pipeline_total_sec" or key.startswith("phase_")) and key.endswith("_sec")
        ]
        if mean_stage_timers:
            print("  Mean stage timers (sec):")
            for stage_name, elapsed in mean_stage_timers:
                elapsed_text = f"{elapsed:.4f}" if isinstance(elapsed, (int, float)) else "N/A"
                print(f"    {stage_name}: {elapsed_text}")
        if summary["averages"].get("gt_pair_recovery_ratio") is not None:
            print(f"  Mean GT pair recovery     : {summary['averages']['gt_pair_recovery_ratio']}")
            print(f"  Mean GT miss STK / THR    : {summary['averages']['gt_fail_sentence_side_top_k']} / {summary['averages']['gt_fail_transitive_join_threshold']}")
        print(f"  Admissions completed      : {summary['totals']['n_admissions']}")
        print(f"  Admissions failed         : {len(failed_admissions)}")
        print("\n[OK] Done.")
        return

    admission_id = cli.admission_id or (
        DEFAULT_ADMISSION_ID if cli.dataset == DEFAULT_SINGLE_ADMISSION_DATASET else None
    )
    if admission_id is None:
        raise ValueError(
            f"No default admission is configured for dataset '{cli.dataset}'. "
            "Pass --admission_id or use --run_all_admissions."
        )
    if admission_id not in admission_index:
        raise KeyError(f"Admission {admission_id} not found in dataset file {data_file}")

    record = admission_index[admission_id]
    record_patient_id = str(record.get("patient_id", ""))
    patient_id = cli.patient_id or record_patient_id or DEFAULT_TARGET_PATIENT
    if cli.patient_id and record_patient_id and cli.patient_id != record_patient_id:
        raise ValueError(
            f"Patient id mismatch for admission {admission_id}: dataset has {record_patient_id}, got {cli.patient_id}"
        )

    configure_runtime_context(output_dataset_name, data_file, annot_file, admission_id, patient_id)

    print("=" * 66)
    print("  LOKI - Cross-Table Join Path Extraction & Materialization")
    print(f"  Admission: {ADMISSION_ID}  |  Patient: {TARGET_PATIENT}")
    print(f"  Dataset: {cli.dataset}  |  Device: {DEVICE}")
    print("=" * 66)

    print("\n-- Phase A: Data Loading -----------------------------------------")
    diag_rows, med_rows, sent_texts, sent_meta = load_admission_data_from_examples(record)
    print(
        f"  Loaded {len(diag_rows)} diagnosis rows, {len(med_rows)} medication rows, {len(sent_texts)} note sentences"
    )
    gt_relationships, gt_diag, gt_med, multi_pairs = load_ground_truth_for_admission(
        admission_id,
        annotation_entries,
        resolve_rel_inventory=False,
    )
    annotation_entry = annotation_entries[admission_id]
    negative_pairs = _extract_negative_pairs_from_annotation_entry(annotation_entry)
    evaluation_target_pairs: Optional[Set[Tuple[int, int]]] = None
    if resolved_evaluation_profile == EVALUATION_PROFILE_AE_DIS_CLEAN:
        evaluation_target_pairs, evaluation_stats = _clean_ae_dis_target_pairs_from_annotation_entry(annotation_entry)
        if not (evaluation_stats["has_clean_ae"] and evaluation_stats["has_clean_dis"]):
            raise ValueError(
                f"Admission {admission_id} does not satisfy evaluation profile {resolved_evaluation_profile}"
            )
        gt_relationships, gt_diag, gt_med, multi_pairs = _filter_ground_truth_for_target_pairs(
            gt_relationships,
            gt_diag,
            gt_med,
            multi_pairs,
            evaluation_target_pairs,
        )
    from collections import Counter as _Counter
    type_counts = _Counter(rel["rel_type"] for rel in gt_relationships)
    n_unique_pairs = len({(rel["diag_idx"], rel["drug_idx"]) for rel in gt_relationships})
    annot_label = str(annot_file.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    print(
        f"  Ground truth: {len(gt_relationships)} relationships "
        f"({', '.join(f'{v} {k}' for k, v in sorted(type_counts.items(), key=lambda item: _rel_type_sort_key(item[0])))}), "
        f"{n_unique_pairs} unique (diag, drug) pairs, {len(multi_pairs)} multi-relationship pairs"
    )
    print(f"  Active relationship types (annotation corpus): {', '.join(active_rel_types)}")
    print(f"  Relationship inventory source: {annot_label}")
    if resolved_evaluation_profile != EVALUATION_PROFILE_DEFAULT:
        print(f"  Evaluation profile: {resolved_evaluation_profile}")
    if negative_pairs:
        print(f"  Negative relationship pairs: {len(negative_pairs)}")
    print(f"  Row coverage: {len(gt_diag)} diagnosis rows, {len(gt_med)} medication rows annotated")

    print("\n-- Phase B: Model Loading ----------------------------------------")
    model_args = load_model_args()
    model = build_model(model_args)
    model.to(DEVICE)
    model.eval()
    load_checkpoint(model, Path(cli.checkpoint))
    params = model.count_parameters()
    print(f"  Parameters: {params['total_parameters']:,} total, {params['trainable_parameters']:,} trainable")

    result = run_materialization_pipeline(
        cli,
        model,
        model_args,
        diag_rows,
        med_rows,
        sent_texts,
        sent_meta,
        gt_relationships,
        gt_diag,
        gt_med,
        multi_pairs,
        negative_pairs=negative_pairs,
        evaluation_target_pairs=evaluation_target_pairs,
        evaluation_profile=resolved_evaluation_profile,
    )

    paths = (
        result["paths"]
        if evaluation_target_pairs is None
        else (_filter_paths_for_target_pairs(result["paths"], evaluation_target_pairs) or [])
    )
    metrics = result["metrics"]
    clustered_paths = (
        result["clustered_paths"]
        if evaluation_target_pairs is None
        else _filter_paths_for_target_pairs(result["clustered_paths"], evaluation_target_pairs)
    )
    cluster_name_map = result["cluster_name_map"]
    cluster_label_details = result["cluster_label_details"]
    cluster_label_backend = result["cluster_label_backend"]
    cluster_label_input_mode = result["cluster_label_input_mode"]
    kept_cluster_ids = result["kept_cluster_ids"]
    pair_recovery_diagnostics = result["pair_recovery_diagnostics"]

    print("\n-- Materialized Table Preview ------------------------------------")
    print_table_preview(paths, n=12)

    sanitized_gt_relationships = result.get("gt_relationships", gt_relationships)

    save_outputs(
        paths,
        metrics,
        sanitized_gt_relationships,
        cluster_name_map=cluster_name_map,
        cluster_label_details=cluster_label_details,
        cluster_label_backend=cluster_label_backend,
        cluster_label_input_mode=cluster_label_input_mode,
        pair_recovery_diagnostics=pair_recovery_diagnostics,
        diagnostics_output_dir=cli.diagnostics_output_dir,
    )
    if clustered_paths:
        save_cluster_audit(
            clustered_paths,
            sanitized_gt_relationships,
            cluster_name_map,
            kept_cluster_ids,
            cluster_label_details=cluster_label_details,
        )

    diagnose_gt_coverage(result["pair_scores"], result["n_diag"], sanitized_gt_relationships)

    if cli.skip_visualizations:
        print("\n-- Visualizations -------------------------------------------------")
        print("  Skipped (--skip_visualizations)")
    else:
        print("\n-- Visualizations -------------------------------------------------")
        selected_topic_map_clusters: Optional[set[int]] = None
        if cli.topic_map_cluster_ids.strip():
            selected_topic_map_clusters = {
                int(part.strip())
                for part in cli.topic_map_cluster_ids.split(",")
                if part.strip()
            }
        visualize_embedding_space(
            diag_rows,
            med_rows,
            result["raw_rows"],
            result["raw_sentences"],
            result["refined_rows"],
            result["refined_sentences"],
            paths,
            sanitized_gt_relationships,
            str(OUT_EMBEDDING),
            sentence_encoder=getattr(model, "sentence_encoder", None),
            cluster_key="cluster_id",
            max_clusters=cli.topic_map_max_clusters,
            label_top_k=cli.topic_map_label_top_k,
            include_cluster_ids=selected_topic_map_clusters,
            show_cluster_numbers=not cli.hide_topic_map_cluster_numbers,
            triples_per_label=cli.topic_map_triples_per_label,
            pair_embedding_mode=cli.pair_embedding_mode,
        )

        viz_paths = clustered_paths if clustered_paths is not None else paths
        tsne_out = str(VIS_DIR / f"clusters_tsne_{ADMISSION_ID}.png")
        visualize_clusters_tsne(
            viz_paths,
            result["refined_rows"],
            result["n_diag"],
            result["refined_sentences"],
            sanitized_gt_relationships,
            tsne_out,
            sentence_encoder=getattr(model, "sentence_encoder", None),
            cluster_key="raw_cluster_id" if clustered_paths is not None else "cluster_id",
            pair_embedding_mode=cli.pair_embedding_mode,
        )

        semantic_cluster_out = str(VIS_DIR / f"semantic_cluster_projection_{ADMISSION_ID}.png")
        visualize_semantic_cluster_projection(
            paths,
            result["refined_rows"],
            result["n_diag"],
            result["refined_sentences"],
            sanitized_gt_relationships,
            semantic_cluster_out,
            sentence_encoder=getattr(model, "sentence_encoder", None),
            cluster_key="cluster_id",
            pair_embedding_mode=cli.pair_embedding_mode,
            negative_pairs=negative_pairs,
        )

        classwise_metrics_out = VIS_DIR / f"classwise_typed_metrics_{ADMISSION_ID}.png"
        visualize_classwise_typed_metrics(
            metrics.get("classwise_typed_metrics", {}),
            classwise_metrics_out,
            title=f"Classwise Typed Metrics - admission {ADMISSION_ID} ({cli.dataset})",
        )

        tsne_full_out = str(VIS_DIR / f"clusters_tsne_full_{ADMISSION_ID}.png")
        visualize_all_sentences_tsne(result["refined_sentences"], viz_paths, sanitized_gt_relationships, tsne_full_out)

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
