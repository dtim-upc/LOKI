"""
Post-Training Evaluation Script (Comprehensive Version)

This script loads saved models from disk and evaluates them using the exact same
data loaders, model architectures, and evaluation functions used during training.
This ensures consistency between training-time and post-training metrics.

Features:
- Loads training configuration from args.json
- Supports multiple model types: best (validation), best_test_overall_acc, best_test_avg_precision
- Runs FULL 3-stage evaluation (Stage 0, 2, 3) using original codebase functions
- Saves detailed results including:
  - Per-table metrics (diagnosis/medication)
  - ROC-AUC scores
  - Pair scores data for every row-sentence pair
  - Prediction breakdown (TP, FP, FN)
- Generates comprehensive visualizations:
  - ROC/PR curves
  - 3-stage comparison charts

Usage:
    # Evaluate a specific training run with full 3-stage analysis
    python post_training_evaluation.py --output_dir ./output_cross_attention_cache/bidirectional_xxx_20260120_123456/

    # Quick mode: only evaluate trained models (skip Stage 2)
    python post_training_evaluation.py --output_dir ... --quick

    # Specify model type explicitly
    python post_training_evaluation.py --output_dir ... --model_type best_test_avg_precision
"""
# =============================================================================
# FIX: OpenMP duplicate library error on Windows
# This MUST be set before importing numpy, torch, or any other library that uses OpenMP
# =============================================================================
import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from loki_path import ensure_loki_on_path, load_loki_module
from model_download import download_input_models

ensure_loki_on_path()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import Unsloth from the shared LOKI codebase if available.
try:
    _loki_unsloth_encoder = load_loki_module("unsloth_encoder")
    UNSLOTH_AVAILABLE = _loki_unsloth_encoder.UNSLOTH_AVAILABLE
    FAST_SENTENCE_TRANSFORMER_AVAILABLE = _loki_unsloth_encoder.FAST_SENTENCE_TRANSFORMER_AVAILABLE
    create_unsloth_sentence_encoder = _loki_unsloth_encoder.create_unsloth_sentence_encoder
    get_model_max_seq_length = _loki_unsloth_encoder.get_model_max_seq_length  # Auto-detect from model config
except (ImportError, AttributeError):
    UNSLOTH_AVAILABLE = False
    FAST_SENTENCE_TRANSFORMER_AVAILABLE = False
    def create_unsloth_sentence_encoder(*args, **kwargs):
        raise ImportError("Unsloth not available")
    def get_model_max_seq_length(model_name: str, default: int = 512) -> int:
        return default


import torch
import numpy as np
from tqdm import tqdm

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

# Import from shared LOKI codebase
from data import load_row_level_dataset, _extract_sentences_robust  # pyright: ignore[reportMissingImports]
from evaluate_mimic_row_sent import (  # pyright: ignore[reportMissingImports]
    load_mimic_test_data_and_annotations,
    get_anchor_id_to_admission_mapping,
    extract_mimic_row_sentence_pairs,
    calculate_mimic_grounding_metrics
)
from models import BidirectionalTableTextModel, TableTextEmbeddingModel  # pyright: ignore[reportMissingImports]
from row_sentence_eval import safe_tensor_to_numpy  # pyright: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer
from hf_model_resolver import ensure_repo_local_hf_snapshot  # pyright: ignore[reportMissingImports]

# sklearn metrics
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from decimal import Decimal, ROUND_HALF_UP
from utils import save_plot_multi_format  # pyright: ignore[reportMissingImports]


def round_half_up(value: float, decimals: int = 2) -> float:
    """
    Round a value using ROUND_HALF_UP (0.555 -> 0.56, not banker's rounding).
    
    Args:
        value: The float value to round
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Rounded float value
    """
    d = Decimal(str(value))
    rounded = d.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP)
    return float(rounded)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = REPO_ROOT / "Datasets"
DATASET_PATH_KEYS = [
    "train_file",
    "eval_file",
    "test_file",
    "row_sent_test_file",
    "row_sent_annotation_file",
]
DATASET_DIR_ALIASES = {
    "mimic": "mimic",
    "mimic_data": "mimic",
    "mimic_small": "mimic",
    "mimic_flipped": "mimic_flipped",
    "protrix": "protrix",
    "protrix_data": "protrix",
    "totto": "totto",
    "totto_data": "totto",
    "multihiertt": "multihiertt",
    "multihiertt_data": "multihiertt",
    "pharma_flipped": "pharma_flipped_structured",
    "pharma_flipped_structured": "pharma_flipped_structured",
}


def _canonicalize_dataset_dir_name(name: Optional[str]) -> Optional[str]:
    """Map historical dataset folder names to the repo-root dataset layout."""
    if not name:
        return None
    return DATASET_DIR_ALIASES.get(name.lower(), name)


def resolve_dataset_path(path_value: Optional[str], args: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Resolve dataset paths against the repo-root Datasets directory.

    Handles:
    - legacy Linux absolute paths ending in `/Datasets/<dataset_name>/...`
    - historical relative paths like `./mimic_data/train_row_level.json`
    - current repo-root layout under `Datasets/<dataset_name>/...`
    """
    if not path_value:
        return path_value

    raw_path = str(path_value)
    direct_path = Path(raw_path)
    if direct_path.exists():
        return str(direct_path)

    normalized = raw_path.replace("\\", "/").strip()
    relative_candidate = (REPO_ROOT / normalized.lstrip("./")).resolve()
    if relative_candidate.exists():
        return str(relative_candidate)

    parts = [part for part in normalized.split("/") if part and part != "."]
    basename = Path(normalized).name
    candidates: List[Path] = []
    dataset_hints: List[str] = []

    dataset_format = _canonicalize_dataset_dir_name((args or {}).get("dataset_format"))
    if dataset_format:
        dataset_hints.append(dataset_format)

    for part in parts:
        canonical = _canonicalize_dataset_dir_name(part)
        if canonical and canonical not in dataset_hints:
            dataset_hints.append(canonical)

    dataset_relative_parts: Optional[List[str]] = None
    if "Datasets" in parts:
        datasets_idx = parts.index("Datasets")
        if datasets_idx + 1 < len(parts):
            dataset_relative_parts = parts[datasets_idx + 1 :]
    elif len(parts) >= 2 and _canonicalize_dataset_dir_name(parts[0]):
        dataset_relative_parts = parts

    if dataset_relative_parts:
        dataset_dir = _canonicalize_dataset_dir_name(dataset_relative_parts[0])
        remainder = dataset_relative_parts[1:]
        if dataset_dir:
            candidates.append(DATASETS_ROOT / dataset_dir / Path(*remainder))
            if dataset_dir not in dataset_hints:
                dataset_hints.insert(0, dataset_dir)

    for hint in dataset_hints:
        candidates.append(DATASETS_ROOT / hint / basename)

    seen = set()
    deduped_candidates: List[Path] = []
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key not in seen:
            seen.add(candidate_key)
            deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate.exists():
            return str(candidate)

    if dataset_hints:
        matched = []
        for hint in dataset_hints:
            dataset_dir = DATASETS_ROOT / hint
            if dataset_dir.exists():
                matched.extend(sorted(dataset_dir.glob(basename)))
        if len(matched) == 1:
            return str(matched[0])

    return raw_path


def normalize_dataset_paths(args: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve dataset-related paths in loaded training args."""
    resolved_args = dict(args)
    for key in DATASET_PATH_KEYS:
        original_value = resolved_args.get(key)
        resolved_value = resolve_dataset_path(original_value, resolved_args)
        if resolved_value and resolved_value != original_value:
            resolved_args[key] = resolved_value
            print(f"[INFO] Resolved {key}: {original_value} -> {resolved_value}")
    return resolved_args


def parse_unsloth_target_modules(raw_value: Any) -> Any:
    """
    Normalize saved Unsloth target-module config for evaluation.

    Returns either:
    - "auto" to let the shared LOKI helper detect the correct modules, or
    - a cleaned list of module names
    """
    if raw_value is None:
        return "auto"

    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value or value.lower() == "auto":
            return "auto"
        modules = [m.strip() for m in value.split(",") if m.strip()]
        return modules or "auto"

    if isinstance(raw_value, (list, tuple, set)):
        modules = [str(m).strip() for m in raw_value if str(m).strip()]
        if not modules:
            return "auto"
        if len(modules) == 1 and modules[0].lower() == "auto":
            return "auto"
        return modules

    return "auto"


def is_unsloth_target_module_error(exc: Exception) -> bool:
    """Return True when Unsloth/PEFT failed because the requested target modules do not exist."""
    message = str(exc)
    return (
        "Target modules" in message
        and "not found in the base model" in message
    )


def get_row_sent_f1_metric(metrics: Dict[str, Any]) -> float:
    """Return the row-sentence F1 metric from a metrics payload."""
    return metrics.get("row_sent_f1", metrics.get("dynamic_f1", metrics.get("overall_accuracy", 0.0)))


def maybe_subsample_row_sent_eval_data(
    test_examples: List[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    max_examples: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Deterministically subsample row-sentence evaluation data for quick smoke tests.

    This mirrors the training-side `--row_sent_max_examples` behavior closely:
    1. prefer examples that can be matched to annotations
    2. sample deterministically using stable example keys
    3. filter annotations to the selected subset
    """
    if max_examples is None or max_examples <= 0 or len(test_examples) <= max_examples:
        return test_examples, annotations

    annotation_keys = {str(key) for key in annotations.keys()}
    anchor_to_admission = get_anchor_id_to_admission_mapping(annotations, test_examples)
    pre_filter_count = len(test_examples)

    def get_stable_key(example: Dict[str, Any]) -> str:
        return str(example.get("anchor_id") or example.get("id") or "")

    filtered_examples = []
    for example in test_examples:
        example_keys = {
            get_stable_key(example),
            str(example.get("admission_id", "")),
        }
        mapped_admission_id = anchor_to_admission.get(example.get("anchor_id"))
        if mapped_admission_id is not None:
            example_keys.add(str(mapped_admission_id))

        if any(key and key in annotation_keys for key in example_keys):
            filtered_examples.append(example)

    if filtered_examples and len(filtered_examples) < pre_filter_count:
        print(
            f"🔍 Pre-filtered row-sentence smoke-test set to {len(filtered_examples)}/{pre_filter_count} "
            f"examples with annotations"
        )

    working_examples = filtered_examples or test_examples

    if len(working_examples) > max_examples:
        working_examples_sorted = sorted(working_examples, key=get_stable_key)
        sampling_rng = random.Random(42)
        selected_examples = sampling_rng.sample(working_examples_sorted, max_examples)
        print(f"⚡ Subsampled row-sentence evaluation to {len(selected_examples)} examples (deterministic)")
    else:
        selected_examples = working_examples
        print(f"⚡ Using {len(selected_examples)} row-sentence examples for smoke test")

    selected_keys = {get_stable_key(example) for example in selected_examples if get_stable_key(example)}
    selected_admission_ids = {
        str(example.get("admission_id"))
        for example in selected_examples
        if example.get("admission_id") is not None
    }
    for example in selected_examples:
        mapped_admission_id = anchor_to_admission.get(example.get("anchor_id"))
        if mapped_admission_id is not None:
            selected_admission_ids.add(str(mapped_admission_id))

    selected_annotations = {
        key: value
        for key, value in annotations.items()
        if str(key) in selected_keys or str(key) in selected_admission_ids
    }

    return selected_examples, selected_annotations or annotations


def checkpoint_uses_legacy_sentence_encoder_keys(state_dict: Dict[str, Any]) -> bool:
    """Return True when a checkpoint uses the older SentenceTransformer `auto_model` key prefix."""
    return any(
        str(key).startswith("sentence_encoder.0.auto_model.")
        for key in state_dict.keys()
    )


def checkpoint_uses_legacy_inner_gate_keys(state_dict: Dict[str, Any]) -> bool:
    """Return True when a checkpoint stores the old inner-gate parameter names."""
    legacy_prefixes = (
        "bidirectional_attention.forward_attention.attention_output_gate.",
        "bidirectional_attention.reverse_attention.attention_output_gate.",
    )
    return any(str(key).startswith(legacy_prefixes) for key in state_dict.keys())


def get_checkpoint_compatible_model_args(
    args: Dict[str, Any],
    state_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize saved training args for loading older checkpoints with the current shared LOKI code.

    Older bidirectional runs used `use_gated_attention` to enable both the outer gate and the
    inner attention gate. In current shared code those are split, so if the saved args predate
    `use_inner_gate` we infer it from the checkpoint layout.
    """
    compatible_args = dict(args)

    if (
        compatible_args.get("use_bidirectional", True)
        and "use_inner_gate" not in compatible_args
        and state_dict is not None
        and checkpoint_uses_legacy_inner_gate_keys(state_dict)
    ):
        compatible_args["use_inner_gate"] = bool(compatible_args.get("use_gated_attention", False))
        print("[INFO] Enabling legacy inner-gate compatibility for this checkpoint.")

    return compatible_args


def remap_checkpoint_state_dict_for_current_libraries(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap legacy checkpoint keys to match the current library stack.

    Newer `sentence_transformers` versions register the encoder backbone under `.model`, while
    older checkpoints in this project were saved under `.auto_model`.
    """
    if not checkpoint_uses_legacy_sentence_encoder_keys(state_dict):
        return state_dict

    remapped_state_dict = {}
    for key, value in state_dict.items():
        remapped_key = str(key).replace(
            "sentence_encoder.0.auto_model.",
            "sentence_encoder.0.model.",
        )
        remapped_state_dict[remapped_key] = value

    print("[INFO] Remapped legacy SentenceTransformer checkpoint keys for current library compatibility.")
    return remapped_state_dict


def align_checkpoint_sentence_encoder_keys(
    state_dict: Dict[str, Any],
    model_state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Match SentenceTransformer encoder key prefixes to the current retry model."""
    encoder_prefixes = (
        "sentence_encoder.0.auto_model.",
        "sentence_encoder.0.model.",
    )
    checkpoint_prefix = next(
        (prefix for prefix in encoder_prefixes if any(key.startswith(prefix) for key in state_dict)),
        None,
    )
    model_prefix = next(
        (prefix for prefix in encoder_prefixes if any(key.startswith(prefix) for key in model_state_dict)),
        None,
    )

    if not checkpoint_prefix or not model_prefix or checkpoint_prefix == model_prefix:
        return state_dict

    print(
        "[INFO] Remapped SentenceTransformer checkpoint keys "
        f"from {checkpoint_prefix} to {model_prefix}."
    )
    return {
        key.replace(checkpoint_prefix, model_prefix, 1)
        if key.startswith(checkpoint_prefix) else key: value
        for key, value in state_dict.items()
    }


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


UNI_DIRECTION_DISPLAY_NAMES = {
    "row_to_sentence": "Uni (R⟶S)",
    "sentence_to_row": "Uni (S⟶R)",
}

UNI_DIRECTION_RESULTS_DIRS = {
    "row_to_sentence": "Uni (R-S)",
    "sentence_to_row": "Uni (S-R)",
}

DISPLAY_NAME_ALIASES = {
    "Uni-cross": "Uni (R⟶S)",
    "Uni (R-S)": "Uni (R⟶S)",
    "Uni (S-R)": "Uni (S⟶R)",
    "Uni (R→S)": "Uni (R⟶S)",
    "Uni (S→R)": "Uni (S⟶R)",
}

KNOWN_MODEL_RESULTS_DIRS = {
    "LOKI",
    "FT-Encoder",
    "Uni-cross",
    "Uni (R-S)",
    "Uni (S-R)",
    "Uni (R→S)",
    "Uni (S→R)",
    "Uni (R⟶S)",
    "Uni (S⟶R)",
}


def _to_display_model_name(results_dir_name: str) -> str:
    """Convert a filesystem-friendly model folder name into a display label."""
    return DISPLAY_NAME_ALIASES.get(results_dir_name, results_dir_name)


def normalize_model_results_identity(output_dir: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer a stable results subdirectory and display label for the evaluated model.

    Unidirectional checkpoints are normalized into dedicated variant folders so the
    two attention directions do not overwrite each other or get collapsed into the
    legacy Uni-cross bucket.
    """
    output_path = Path(output_dir)
    use_bidirectional = args.get("use_bidirectional", True)
    attention_direction = args.get("attention_direction", "row_to_sentence")

    if not use_bidirectional:
        results_dir_name = UNI_DIRECTION_RESULTS_DIRS.get(attention_direction, output_path.name)
        display_name = UNI_DIRECTION_DISPLAY_NAMES.get(
            attention_direction,
            _to_display_model_name(results_dir_name)
        )
        return {
            "results_dir_name": results_dir_name,
            "display_name": display_name,
            "family": "unidirectional",
            "attention_direction": attention_direction,
            "use_bidirectional": False,
        }

    results_dir_name = next(
        (
            candidate for candidate in [output_path.name, output_path.parent.name]
            if candidate in KNOWN_MODEL_RESULTS_DIRS
        ),
        output_path.name
    )
    return {
        "results_dir_name": results_dir_name,
        "display_name": _to_display_model_name(results_dir_name),
        "family": "bidirectional",
        "attention_direction": attention_direction,
        "use_bidirectional": True,
    }


def _extract_rows_from_example(example: Dict[str, Any], table_type: Optional[str] = None) -> List[str]:
    """Extract row texts from either Protrix-style or MIMIC-style example payloads."""
    rows: List[str] = []
    raw_rows = example.get("anchor_rows", [])

    if raw_rows:
        for row in raw_rows:
            if isinstance(row, dict):
                formatted = row.get("formatted", "")
                if formatted:
                    rows.append(formatted)
            elif isinstance(row, str) and row:
                rows.append(row)
        return rows

    tables = example.get("tables", {})
    if not isinstance(tables, dict):
        return rows

    if table_type and table_type in tables:
        table_values = [tables.get(table_type, {})]
    else:
        table_values = list(tables.values())

    for table_data in table_values:
        if not isinstance(table_data, dict):
            continue
        for row in table_data.get("rows", []):
            if isinstance(row, dict):
                formatted = row.get("formatted", "")
                if formatted:
                    rows.append(formatted)
            elif isinstance(row, str) and row:
                rows.append(row)
    return rows


def _extract_primary_sentences_from_example(example: Dict[str, Any]) -> List[str]:
    """Extract primary-positive note sentences in a format-agnostic way."""
    primary_positive = example.get("primary_positive", {})
    raw_sentences = primary_positive.get("sentences", [])
    return _extract_sentences_robust(raw_sentences)


def _compute_split_statistics(examples: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
    """Compute dataset descriptive statistics for one split."""
    total_rows = 0
    total_primary_sentences = 0
    total_additional_positive_sentences = 0
    total_negative_sentences = 0
    row_counts: List[int] = []
    primary_sentence_counts: List[int] = []
    sentence_char_lens: List[int] = []
    sentence_token_lens: List[int] = []

    for example in examples:
        rows = _extract_rows_from_example(example)
        row_count = len(rows)
        total_rows += row_count
        row_counts.append(row_count)

        primary_sentences = _extract_primary_sentences_from_example(example)
        primary_count = len(primary_sentences)
        total_primary_sentences += primary_count
        primary_sentence_counts.append(primary_count)

        additional_positives = example.get("additional_positives", [])
        for item in additional_positives:
            sents = _extract_sentences_robust(item.get("sentences", []))
            total_additional_positive_sentences += len(sents)

        negatives = example.get("negatives", [])
        for item in negatives:
            sents = _extract_sentences_robust(item.get("sentences", []))
            total_negative_sentences += len(sents)

        for sent in primary_sentences:
            sent = sent.strip()
            if not sent:
                continue
            sentence_char_lens.append(len(sent))
            sentence_token_lens.append(len(sent.split()))

    def _safe_stats(values: List[int]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0}
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "split_name": split_name,
        "num_examples": len(examples),
        "total_rows": total_rows,
        "avg_rows_per_example": float(np.mean(row_counts)) if row_counts else 0.0,
        "total_primary_sentences": total_primary_sentences,
        "avg_primary_sentences_per_example": float(np.mean(primary_sentence_counts)) if primary_sentence_counts else 0.0,
        "total_additional_positive_sentences": total_additional_positive_sentences,
        "total_negative_sentences": total_negative_sentences,
        "sentence_length_chars": _safe_stats(sentence_char_lens),
        "sentence_length_tokens": _safe_stats(sentence_token_lens),
    }


# =============================================================================
# RANKING METRICS FUNCTIONS
# =============================================================================

def calculate_ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Model prediction scores
        k: Number of top predictions to consider (use len(y_true) for "all")
    
    Returns:
        NDCG@K score
    """
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    
    # Get top k indices sorted by score (descending)
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    
    # Calculate DCG@K
    dcg = 0.0
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG@K (ideal DCG)
    num_relevant = int(min(k, np.sum(y_true)))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_recall_f1_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> Tuple[float, float, float]:
    """
    Calculate Precision@K, Recall@K, and F1@K.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Model prediction scores
        k: Number of top predictions to consider
    
    Returns:
        Tuple of (precision@k, recall@k, f1@k)
    """
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    
    sorted_indices = np.argsort(y_scores)[::-1]
    top_k_indices = set(sorted_indices[:k])
    actual_positive = set(np.where(y_true == 1)[0])
    
    if len(actual_positive) == 0:
        return 0.0, 0.0, 0.0
    
    tp = len(top_k_indices & actual_positive)
    fp = len(top_k_indices - actual_positive)
    fn = len(actual_positive - top_k_indices)
    
    precision_k = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_k = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
    
    return precision_k, recall_k, f1_k


def calculate_mrr_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Calculate MRR@K (Mean Reciprocal Rank at K).
    
    MRR@K is the reciprocal of the rank of the first relevant item within top K.
    If no relevant item is found in top K, returns 0.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Model prediction scores
        k: Number of top predictions to consider
    
    Returns:
        MRR@K score (1/rank of first relevant item, or 0 if none in top k)
    """
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    
    # Find the rank of the first relevant item
    for rank, idx in enumerate(sorted_indices, 1):
        if y_true[idx] == 1:
            return 1.0 / rank
    
    return 0.0  # No relevant item found in top k


def calculate_mean_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate mean rank of all ground truth items.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Model prediction scores
    
    Returns:
        Mean rank of ground truth items (lower is better)
    """
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return float('inf')
    
    sorted_indices = np.argsort(y_scores)[::-1]
    gt_indices = set(np.where(y_true == 1)[0])
    
    ranks = []
    for rank, idx in enumerate(sorted_indices, 1):
        if idx in gt_indices:
            ranks.append(rank)
    
    return np.mean(ranks) if ranks else float('inf')


def calculate_pair_ranking_metrics(
    pair_scores: np.ndarray,
    row_sentence_pairs: List[Tuple[int, int]],
    num_rows: int,
    num_sentences: int,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """
    Calculate comprehensive ranking metrics for row-sentence pairs.
    
    This flattens the pair score matrix and treats it as a ranking problem.
    
    Args:
        pair_scores: [num_rows, num_sentences] matrix of scores
        row_sentence_pairs: List of ground truth (row_idx, sent_idx) pairs
        num_rows: Number of rows
        num_sentences: Number of sentences
        k_values: List of K values for @K metrics
    
    Returns:
        Dictionary with all ranking metrics
    """
    if not row_sentence_pairs or pair_scores.size == 0:
        return {
            "precision_at_k": {k: 0.0 for k in k_values},
            "recall_at_k": {k: 0.0 for k in k_values},
            "f1_at_k": {k: 0.0 for k in k_values},
            "ndcg_at_k": {k: 0.0 for k in k_values},
            "mrr_at_k": {k: 0.0 for k in k_values},
            "mean_rank": float('inf'),
            "num_ground_truth_pairs": 0
        }
    
    # Create binary ground truth matrix
    gt_matrix = np.zeros((num_rows, num_sentences), dtype=int)
    for row_idx, sent_idx in row_sentence_pairs:
        if 0 <= row_idx < num_rows and 0 <= sent_idx < num_sentences:
            gt_matrix[row_idx, sent_idx] = 1
    
    # Flatten for ranking evaluation
    gt_flat = gt_matrix.flatten()
    scores_flat = pair_scores.flatten()
    
    # Calculate metrics at different K values
    precision_at_k = {}
    recall_at_k = {}
    f1_at_k = {}
    ndcg_at_k = {}
    mrr_at_k = {}
    
    for k in k_values:
        actual_k = min(k, len(scores_flat))
        prec, rec, f1 = calculate_precision_recall_f1_at_k(gt_flat, scores_flat, actual_k)
        precision_at_k[k] = prec
        recall_at_k[k] = rec
        f1_at_k[k] = f1
        ndcg_at_k[k] = calculate_ndcg_at_k(gt_flat, scores_flat, actual_k)
        mrr_at_k[k] = calculate_mrr_at_k(gt_flat, scores_flat, actual_k)
    
    # Also calculate @all
    all_k = len(scores_flat)
    prec_all, rec_all, f1_all = calculate_precision_recall_f1_at_k(gt_flat, scores_flat, all_k)
    precision_at_k["all"] = prec_all
    recall_at_k["all"] = rec_all
    f1_at_k["all"] = f1_all
    ndcg_at_k["all"] = calculate_ndcg_at_k(gt_flat, scores_flat, all_k)
    mrr_at_k["all"] = calculate_mrr_at_k(gt_flat, scores_flat, all_k)
    
    # Calculate mean rank
    mean_rank = calculate_mean_rank(gt_flat, scores_flat)
    
    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "f1_at_k": f1_at_k,
        "ndcg_at_k": ndcg_at_k,
        "mrr_at_k": mrr_at_k,
        "mean_rank": mean_rank,
        "num_ground_truth_pairs": len(row_sentence_pairs)
    }


def calculate_dynamic_binary_accuracy_for_pairs(
    pair_scores: np.ndarray,
    row_sentence_pairs: List[Tuple[int, int]],
    num_rows: int,
    num_sentences: int
) -> float:
    """
    Calculate binary accuracy for pair classification using dynamic threshold.

    Dynamic threshold follows the same rule used elsewhere in this script:
    threshold = (mean(GT scores) + mean(non-GT scores)) / 2, with median fallback.
    """
    if pair_scores.size == 0 or num_rows <= 0 or num_sentences <= 0:
        return 0.0

    gt_pairs = set(row_sentence_pairs)
    gt_scores = [
        pair_scores[i, j] for i, j in row_sentence_pairs
        if 0 <= i < num_rows and 0 <= j < num_sentences
    ]
    non_gt_scores = [
        pair_scores[i, j] for i in range(num_rows) for j in range(num_sentences)
        if (i, j) not in gt_pairs
    ]

    if gt_scores and non_gt_scores:
        threshold = (np.mean(gt_scores) + np.mean(non_gt_scores)) / 2.0
    else:
        threshold = np.median(pair_scores.flatten())

    # Build binary labels over all candidate pairs.
    y_true = np.zeros((num_rows, num_sentences), dtype=int)
    for i, j in row_sentence_pairs:
        if 0 <= i < num_rows and 0 <= j < num_sentences:
            y_true[i, j] = 1
    y_true_flat = y_true.flatten()
    y_pred_flat = (pair_scores.flatten() >= threshold).astype(int)

    if len(y_true_flat) == 0:
        return 0.0
    return float(np.mean(y_true_flat == y_pred_flat))


def load_training_args(output_dir: str) -> Dict[str, Any]:
    """Load training arguments from args.json in the output directory."""
    args_path = Path(output_dir) / "args.json"
    
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found in {output_dir}")
    
    with open(args_path, 'r', encoding='utf-8') as f:
        args = json.load(f)

    args = normalize_dataset_paths(args)
    
    print(f"[INFO] Loaded training args from {args_path}")
    return args


def find_model_checkpoints(output_dir: str) -> Dict[str, Path]:
    """Find all available model checkpoints in the output directory."""
    output_path = Path(output_dir)
    checkpoints = {}
    
    # Search for model.pt files recursively
    model_files = list(output_path.rglob("model.pt"))
    
    for model_file in model_files:
        parent_dir = model_file.parent.name
        
        if parent_dir.startswith("best_model_epoch_"):
            checkpoints["best"] = model_file
        elif parent_dir.startswith("best_test_overall_acc_epoch_"):
            checkpoints["best_test_overall_acc"] = model_file
        elif parent_dir.startswith("best_test_avg_precision_epoch_"):
            checkpoints["best_test_avg_precision"] = model_file
    
    # Also check for direct legacy model files (*_best.pt)
    if not checkpoints:
        legacy_matches = list(output_path.glob("*_best.pt"))
        if legacy_matches:
            checkpoints["best"] = legacy_matches[0]
    
    return checkpoints


def create_sentence_encoder(args: Dict[str, Any], device: str = "cuda") -> SentenceTransformer:
    """Create sentence encoder using the same settings as training.
    
    IMPORTANT: Uses torch.bfloat16 precision to match training-time encoder loading.
    This ensures numerical consistency between training and post-training evaluation.
    """
    model_name = args.get("model_name", "abhinand/MedEmbed-small-v0.1")
    use_unsloth = args.get("use_unsloth", False) and UNSLOTH_AVAILABLE
    
    # Get max_seq_length: prefer saved value from training args, else auto-detect
    saved_max_seq_length = args.get("max_seq_length")
    if saved_max_seq_length is not None:
        max_seq_length = saved_max_seq_length
        print(f"[INFO] Using max_seq_length from training args: {max_seq_length}")
    else:
        # Auto-detect from model config
        max_seq_length = get_model_max_seq_length(model_name, default=512)
    
    if use_unsloth:
        print(f"[INFO] Loading encoder with Unsloth: {model_name}")
        if FAST_SENTENCE_TRANSFORMER_AVAILABLE:
            print(f"[INFO] Using FastSentenceTransformer API (specialized for embeddings)")
        else:
            print(f"[INFO] Using FastModel API (fallback)")
        
        pooling_mode = args.get("unsloth_pooling_mode", "mean")
        use_encoder_qlora = args.get("unsloth_qlora", False)
        target_modules = parse_unsloth_target_modules(args.get("unsloth_target_modules", "auto"))

        def _build_unsloth_encoder(curr_use_qlora: bool, curr_target_modules: Any):
            return create_unsloth_sentence_encoder(
                model_name=model_name,
                device=device,
                max_seq_length=max_seq_length,
                # Unsloth configuration
                use_unsloth=True,
                load_in_4bit=args.get("unsloth_4bit", False),
                dtype=torch.bfloat16,
                full_finetuning=False,  # For evaluation, we don't need full finetuning
                # LoRA configuration (match training settings when possible)
                use_qlora=curr_use_qlora,
                lora_rank=args.get("unsloth_qlora_rank", 32),
                lora_alpha=args.get("unsloth_qlora_alpha", 64.0),
                lora_dropout=0.0,
                target_modules=curr_target_modules if curr_use_qlora else None,
                # Pooling configuration
                pooling_mode=pooling_mode,
                normalize_embeddings=True,
            )

        try:
            sentence_encoder = _build_unsloth_encoder(use_encoder_qlora, target_modules)
        except Exception as exc:
            if not (use_encoder_qlora and is_unsloth_target_module_error(exc)):
                raise

            if target_modules != "auto":
                print("[WARNING] Saved Unsloth target modules are incompatible with this encoder.")
                print("[INFO] Retrying encoder load with auto-detected target modules...")
                try:
                    sentence_encoder = _build_unsloth_encoder(True, "auto")
                except Exception as retry_exc:
                    if not is_unsloth_target_module_error(retry_exc):
                        raise
                    print("[WARNING] Auto-detected encoder LoRA target modules also failed.")
                    print("[INFO] Falling back to evaluation without encoder-side QLoRA wrappers...")
                    sentence_encoder = _build_unsloth_encoder(False, None)
            else:
                print("[WARNING] Auto-detected encoder LoRA target modules are incompatible with this model.")
                print("[INFO] Falling back to evaluation without encoder-side QLoRA wrappers...")
                sentence_encoder = _build_unsloth_encoder(False, None)
    else:
        print(f"[INFO] Loading encoder with SentenceTransformers: {model_name}")
        resolved_model_name, model_source = ensure_repo_local_hf_snapshot(model_name)
        print(f"[HF] Snapshot ready for {model_name}: {resolved_model_name} ({model_source})")
        # Use bfloat16 precision to match training-time encoder loading (run_cross_attention.py)
        # This ensures numerical consistency between training and evaluation
        try:
            sentence_encoder = SentenceTransformer(
                resolved_model_name,
                device=device,
                model_kwargs={"dtype": torch.bfloat16},
                trust_remote_code=True,
                local_files_only=True,
            )
            print(f"[INFO] Loaded encoder with torch.bfloat16 precision")
        except Exception as e:
            print(f"[WARNING] Failed to load with bfloat16: {e}")
            print(f"[INFO] Falling back to default precision")
            sentence_encoder = SentenceTransformer(
                resolved_model_name,
                device=device,
                trust_remote_code=True,
                local_files_only=True,
            )
    
    return sentence_encoder


def load_model_from_checkpoint(
    checkpoint_path: Path,
    args: Dict[str, Any],
    device: str = "cuda"
) -> BidirectionalTableTextModel:
    """Load model from checkpoint using saved training arguments."""
    print(f"[INFO] Loading model from: {checkpoint_path}")

    def _build_model(curr_args: Dict[str, Any]):
        sentence_encoder = create_sentence_encoder(curr_args, device)

        embedding_dim = curr_args.get("embedding_dim")
        if embedding_dim is None:
            try:
                embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
            except Exception:
                embedding_dim = 768

        use_bidirectional = curr_args.get("use_bidirectional", True)
        if use_bidirectional:
            return BidirectionalTableTextModel(
                sentence_encoder=sentence_encoder,
                embedding_dim=embedding_dim,
                top_k=curr_args.get("top_k", 5),
                use_self_attention=curr_args.get("use_self_attention", False),
                self_attention_heads=curr_args.get("self_attention_heads", 1),
                self_attention_dropout=curr_args.get("self_attention_dropout", 0.1),
                attention_type=curr_args.get("attention_type", "top_k_sparse"),
                use_gated_attention=curr_args.get("use_gated_attention", True),
                gated_attention_mode=curr_args.get("gated_attention_mode", "vector"),
                gated_attention_hidden_dim=curr_args.get("gated_attention_hidden_dim", 0),
                gated_attention_dropout=curr_args.get("gated_attention_dropout", 0.0),
                gated_attention_init_bias=curr_args.get("gated_attention_init_bias", 2.0),
                sparse_top_k=curr_args.get("sparse_top_k", 5),
                window_size=curr_args.get("window_size", 5),
                threshold_base=curr_args.get("threshold_base", 0.1),
                use_refinement=curr_args.get("use_refinement", False),
                norm_type=curr_args.get("norm_type", "rmsnorm"),
                use_qk_rmsnorm=curr_args.get("use_qk_rmsnorm", False),
                share_weights=curr_args.get("share_weights", curr_args.get("share_attention_weights", True)),
                use_cross_attention_lora=curr_args.get("use_cross_attention_lora", False),
                lora_rank=curr_args.get("lora_rank", 128),
                lora_alpha=curr_args.get("lora_alpha", 512),
                lora_dropout=curr_args.get("lora_dropout", 0.1),
                use_latent_bottleneck=curr_args.get("use_latent_bottleneck", False),
                latent_num=curr_args.get("latent_num", 64),
                latent_dropout=curr_args.get("latent_dropout", 0.0),
                use_inner_gate=curr_args.get("use_inner_gate", False),
                use_header_conditioning=curr_args.get("use_header_conditioning", False),
                use_cell_level_matching=curr_args.get("use_cell_level_matching", False),
                cell_matching_weight=curr_args.get("cell_matching_weight", 0.35),
                cell_matching_pooling=curr_args.get("cell_matching_pooling", "max"),
                cell_row_fusion_weight=curr_args.get("cell_row_fusion_weight", 0.15),
                disable_temperature=curr_args.get("disable_temperature", False),
                init_method=curr_args.get("init_method", "orthogonal"),
                init_method_params=curr_args.get("init_method_params", None),
                verbose=False
            )
        return TableTextEmbeddingModel(
            sentence_encoder=sentence_encoder,
            embedding_dim=embedding_dim,
            top_k=curr_args.get("top_k", 5),
            attention_type=curr_args.get("attention_type", "standard"),
            sparse_top_k=curr_args.get("sparse_top_k", 5),
            window_size=curr_args.get("window_size", 5),
            threshold_base=curr_args.get("threshold_base", 0.1),
            init_method=curr_args.get("init_method", "xavier_uniform"),
            init_method_params=curr_args.get("init_method_params", None),
            norm_type=curr_args.get("norm_type", "layernorm"),
            attention_direction=curr_args.get("attention_direction", "row_to_sentence"),
            use_latent_bottleneck=curr_args.get("use_latent_bottleneck", False),
            latent_num=curr_args.get("latent_num", 64),
            latent_dropout=curr_args.get("latent_dropout", 0.0),
            use_gated_attention=curr_args.get("use_gated_attention", False),
            gated_attention_mode=curr_args.get("gated_attention_mode", "scalar"),
            gated_attention_hidden_dim=curr_args.get("gated_attention_hidden_dim", 0),
            gated_attention_dropout=curr_args.get("gated_attention_dropout", 0.0),
            gated_attention_init_bias=curr_args.get("gated_attention_init_bias", 2.0),
            disable_temperature=curr_args.get("disable_temperature", False),
            skip_ffn=curr_args.get("skip_ffn", False),
            use_cross_attention_lora=curr_args.get("use_cross_attention_lora", False),
            lora_rank=curr_args.get("lora_rank", 16),
            lora_alpha=curr_args.get("lora_alpha", 32.0),
            lora_dropout=curr_args.get("lora_dropout", 0.1),
            verbose=False
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base_args = get_checkpoint_compatible_model_args(args, state_dict)
    remapped_state_dict = remap_checkpoint_state_dict_for_current_libraries(state_dict)
    use_remapped_state_dict = remapped_state_dict is not state_dict

    candidates: List[Tuple[str, Dict[str, Any], bool]] = []
    seen_signatures = set()

    def _add_candidate(description: str, candidate_args: Dict[str, Any], use_remapped: bool) -> None:
        signature = (
            bool(candidate_args.get("use_unsloth", False)),
            bool(candidate_args.get("unsloth_qlora", False)),
            bool(candidate_args.get("use_cross_attention_lora", False)),
            bool(candidate_args.get("use_inner_gate", False)),
            bool(candidate_args.get("use_gated_attention", False)),
            candidate_args.get("attention_type"),
            use_remapped,
        )
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        candidates.append((description, candidate_args, use_remapped))

    _add_candidate("saved training args", base_args, False)

    retry_no_adapters_args = dict(base_args)
    retry_no_adapters_args["unsloth_qlora"] = False
    retry_no_adapters_args["use_cross_attention_lora"] = False
    _add_candidate("without Unsloth QLoRA / cross-attention LoRA wrappers", retry_no_adapters_args, False)

    retry_no_unsloth_args = dict(retry_no_adapters_args)
    retry_no_unsloth_args["use_unsloth"] = False
    _add_candidate(
        "without Unsloth encoder wrappers",
        retry_no_unsloth_args,
        use_remapped_state_dict,
    )

    last_error: Optional[RuntimeError] = None
    model = None

    for attempt_idx, (description, candidate_args, use_remapped) in enumerate(candidates, start=1):
        if attempt_idx > 1:
            print(f"[INFO] Retrying load {description}...")

        candidate_state_dict = remapped_state_dict if use_remapped else state_dict
        model = _build_model(candidate_args)

        # Patch scalar-to-vector mode mismatches for older checkpoints
        model_sd = model.state_dict()
        candidate_state_dict = align_checkpoint_sentence_encoder_keys(
            candidate_state_dict,
            model_sd,
        )
        patched = False
        for k, v in candidate_state_dict.items():
            if "gate.net.1" in k and k in model_sd:
                m_shape = model_sd[k].shape
                if len(m_shape) == 2 and len(v.shape) == 2 and m_shape[0] > 1 and v.shape[0] == 1 and m_shape[1] == v.shape[1]:
                    if not patched:
                        candidate_state_dict = dict(candidate_state_dict)
                        patched = True
                    candidate_state_dict[k] = v.repeat(m_shape[0], 1)
                elif len(m_shape) == 1 and len(v.shape) == 1 and m_shape[0] > 1 and v.shape[0] == 1:
                    if not patched:
                        candidate_state_dict = dict(candidate_state_dict)
                        patched = True
                    candidate_state_dict[k] = v.repeat(m_shape[0])

        try:
            model.load_state_dict(candidate_state_dict)
            break
        except RuntimeError as e:
            last_error = e
            if attempt_idx == 1:
                print("[WARNING] Initial checkpoint load failed.")
            del model
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    else:
        assert last_error is not None
        raise last_error

    assert model is not None
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully")
    return model



def create_sophisticated_pretrain_model(
    sentence_encoder: SentenceTransformer,
    args: Dict[str, Any],
    device: str = "cuda"
) -> BidirectionalTableTextModel:
    """
    Create Stage 2 sophisticated model with training architecture but random init.
    This matches what was used during training but before any weight updates.
    """
    embedding_dim = args.get("embedding_dim")
    if embedding_dim is None:
        try:
            embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
        except:
            embedding_dim = 768
    
    # Stage 2: Same architecture as training but fresh weights
    model = BidirectionalTableTextModel(
        sentence_encoder=sentence_encoder,
        embedding_dim=embedding_dim,
        top_k=args.get("top_k", 5),
        use_self_attention=args.get("use_self_attention", False),
        self_attention_heads=args.get("self_attention_heads", 1),
        self_attention_dropout=args.get("self_attention_dropout", 0.1),
        attention_type=args.get("attention_type", "top_k_sparse"),
        use_gated_attention=args.get("use_gated_attention", True),
        gated_attention_mode=args.get("gated_attention_mode", "vector"),
        gated_attention_hidden_dim=args.get("gated_attention_hidden_dim", 0),
        gated_attention_dropout=args.get("gated_attention_dropout", 0.0),
        gated_attention_init_bias=args.get("gated_attention_init_bias", 2.0),
        sparse_top_k=args.get("sparse_top_k", 5),
        window_size=args.get("window_size", 5),
        threshold_base=args.get("threshold_base", 0.1),
        use_refinement=args.get("use_refinement", False),
        norm_type=args.get("norm_type", "rmsnorm"),
        use_qk_rmsnorm=args.get("use_qk_rmsnorm", False),
        share_weights=args.get("share_weights", args.get("share_attention_weights", True)),
        use_cross_attention_lora=args.get("use_cross_attention_lora", False),
        lora_rank=args.get("lora_rank", 128),
        lora_alpha=args.get("lora_alpha", 512),
        lora_dropout=args.get("lora_dropout", 0.1),
        use_latent_bottleneck=args.get("use_latent_bottleneck", False),
        latent_num=args.get("latent_num", 64),
        latent_dropout=args.get("latent_dropout", 0.0),
        use_inner_gate=args.get("use_inner_gate", False),
        use_header_conditioning=args.get("use_header_conditioning", False),
        use_cell_level_matching=args.get("use_cell_level_matching", False),
        cell_matching_weight=args.get("cell_matching_weight", 0.35),
        cell_matching_pooling=args.get("cell_matching_pooling", "max"),
        cell_row_fusion_weight=args.get("cell_row_fusion_weight", 0.15),
        disable_temperature=args.get("disable_temperature", False),
        init_method=args.get("init_method", "orthogonal"),
        init_method_params=args.get("init_method_params", None),
        verbose=False
    )
    model.to(device)
    model.eval()
    return model


def evaluate_frozen_encoder_comprehensive(
    sentence_encoder,
    test_examples: List[Dict],
    annotations: Dict,
    args: Dict[str, Any],
    batch_size: int = 64,
    collect_pair_scores: bool = True,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Stage 0: Evaluate using ONLY the frozen sentence encoder with cosine similarity.
    
    This does NOT use our cross-attention model at all. It simply:
    1. Encodes rows and sentences with the frozen encoder
    2. Computes cosine similarity matrix
    3. Calculates metrics including ROC-AUC
    
    Returns:
        Dictionary with full metrics comparable to other stages
    """
    print(f"\n[Stage 0] Frozen Encoder Baseline (Pure Cosine Similarity)")
    print(f"  - No cross-attention, just raw encoder similarity scores")
    
    # Build anchor_id -> admission mapping
    anchor_to_admission = get_anchor_id_to_admission_mapping(annotations, test_examples)
    
    # Aggregation tracking
    diagnosis_metrics = []
    medication_metrics = []
    diagnosis_dynamic_binary_accuracy = []
    medication_dynamic_binary_accuracy = []
    all_pair_scores_data = []
    all_y_true = []
    all_y_scores = []
    
    total_tp, total_fp, total_fn = 0, 0, 0
    diagnosis_tp, diagnosis_fp, diagnosis_fn = 0, 0, 0
    medication_tp, medication_fp, medication_fn = 0, 0, 0
    
    # Ranking metrics collection
    k_values = [1, 3, 5, 10]
    all_ranking_metrics = []
    ranking_raw_counts = {
        "queries_evaluated": 0,
        "total_rows": 0,
        "total_documents": 0,
        "total_candidate_pairs": 0,
        "total_ground_truth_pairs": 0,
        "hits_at_k": {str(k): 0 for k in k_values},
        "max_possible_hits_at_k": {str(k): 0 for k in k_values}
    }
    
    with torch.no_grad():
        for example_idx, example in enumerate(tqdm(test_examples, desc="Stage 0: Frozen Encoder")):
            anchor_id = example.get("anchor_id")
            if anchor_id is None or anchor_id not in anchor_to_admission:
                continue
            
            admission_id, table_type = anchor_to_admission[anchor_id]
            annotation = annotations.get(admission_id, {})
            row_grounding = annotation.get("row_grounding", {})
            
            # Get ground truth pairs
            _, row_sentence_pairs = extract_mimic_row_sentence_pairs(row_grounding, table_type)
            if not row_sentence_pairs:
                continue
            
            # Extract rows/sentences from example
            rows = _extract_rows_from_example(example, table_type=table_type)
            
            if not rows:
                continue
            
            sentences = _extract_primary_sentences_from_example(example)
            
            if not sentences:
                continue
            
            try:
                # Encode with frozen encoder
                row_embeddings = sentence_encoder.encode(
                    rows, batch_size=batch_size, convert_to_tensor=True, 
                    normalize_embeddings=True, device=device
                )
                sentence_embeddings = sentence_encoder.encode(
                    sentences, batch_size=batch_size, convert_to_tensor=True,
                    normalize_embeddings=True, device=device
                )
                
                # Compute cosine similarity (already normalized, so matmul = cosine sim)
                # row_embeddings: [num_rows, dim]
                # sentence_embeddings: [num_sentences, dim]
                pair_scores = torch.matmul(row_embeddings, sentence_embeddings.t())
                pair_scores_np = safe_tensor_to_numpy(pair_scores.detach())
                
                num_rows, num_sentences = pair_scores_np.shape
                
                # Calculate metrics for this example
                metrics = calculate_mimic_grounding_metrics(pair_scores_np, row_sentence_pairs, num_rows, num_sentences)
                dynamic_binary_accuracy = calculate_dynamic_binary_accuracy_for_pairs(
                    pair_scores_np, row_sentence_pairs, num_rows, num_sentences
                )
                
                # Collect pair scores data for ROC-AUC
                if collect_pair_scores:
                    gt_set = set(row_sentence_pairs)
                    for i in range(num_rows):
                        for j in range(num_sentences):
                            is_gt = (i, j) in gt_set
                            all_pair_scores_data.append([i, j, float(pair_scores_np[i, j]), is_gt])
                            all_y_true.append(1 if is_gt else 0)
                            all_y_scores.append(float(pair_scores_np[i, j]))
                
                # Update prediction counts using dynamic threshold
                valid_gt_pairs = [(i, j) for i, j in row_sentence_pairs if 0 <= i < num_rows and 0 <= j < num_sentences]
                gt_pairs_set = set(valid_gt_pairs)
                gt_scores = [pair_scores_np[i, j] for i, j in valid_gt_pairs]
                non_gt_scores = [pair_scores_np[i, j] for i in range(num_rows) for j in range(num_sentences) if (i, j) not in gt_pairs_set]
                
                if gt_scores and non_gt_scores:
                    threshold = (np.mean(gt_scores) + np.mean(non_gt_scores)) / 2
                else:
                    threshold = np.median(pair_scores_np.flatten())
                
                predicted_pairs = set()
                for i in range(num_rows):
                    for j in range(num_sentences):
                        if pair_scores_np[i, j] >= threshold:
                            predicted_pairs.add((i, j))
                
                gt_pairs = gt_pairs_set
                example_tp = len(gt_pairs & predicted_pairs)
                example_fp = len(predicted_pairs - gt_pairs)
                example_fn = len(gt_pairs - predicted_pairs)

                total_tp += example_tp
                total_fp += example_fp
                total_fn += example_fn

                if table_type == "diagnosis":
                    diagnosis_tp += example_tp
                    diagnosis_fp += example_fp
                    diagnosis_fn += example_fn
                else:
                    medication_tp += example_tp
                    medication_fp += example_fp
                    medication_fn += example_fn

                # Ranking raw counts for transparent reporting in papers/tables
                ranking_raw_counts["queries_evaluated"] += 1
                ranking_raw_counts["total_rows"] += num_rows
                ranking_raw_counts["total_documents"] += num_sentences
                ranking_raw_counts["total_candidate_pairs"] += (num_rows * num_sentences)
                ranking_raw_counts["total_ground_truth_pairs"] += len(valid_gt_pairs)

                gt_flat = np.zeros(num_rows * num_sentences, dtype=int)
                for i, j in valid_gt_pairs:
                    gt_flat[i * num_sentences + j] = 1
                scores_flat = pair_scores_np.flatten()
                sorted_indices = np.argsort(scores_flat)[::-1]
                for k in k_values:
                    actual_k = min(k, len(scores_flat))
                    top_k_indices = sorted_indices[:actual_k]
                    hits = int(np.sum(gt_flat[top_k_indices]))
                    ranking_raw_counts["hits_at_k"][str(k)] += hits
                    ranking_raw_counts["max_possible_hits_at_k"][str(k)] += min(len(valid_gt_pairs), actual_k)
                
                # Calculate ranking metrics for this example
                example_ranking_metrics = calculate_pair_ranking_metrics(
                    pair_scores_np, row_sentence_pairs, num_rows, num_sentences, k_values
                )
                all_ranking_metrics.append(example_ranking_metrics)
                
                # Store by table type
                if table_type == "diagnosis":
                    diagnosis_metrics.append(metrics)
                    diagnosis_dynamic_binary_accuracy.append(dynamic_binary_accuracy)
                else:
                    medication_metrics.append(metrics)
                    medication_dynamic_binary_accuracy.append(dynamic_binary_accuracy)
                    
            except Exception as e:
                print(f"[WARNING] Stage 0 evaluation error for example {example_idx}: {str(e)[:100]}...")
                continue
    
    # Aggregate per-table metrics
    def aggregate(m_list):
        if not m_list:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "average_precision": 0.0}
        return {
            "precision": np.mean([m["precision"] for m in m_list]),
            "recall": np.mean([m["recall"] for m in m_list]),
            "f1": np.mean([m["f1"] for m in m_list]),
            "jaccard": np.mean([m["jaccard"] for m in m_list]),
            "average_precision": np.mean([m.get("average_precision", 0.0) for m in m_list]),
            "examples": len(m_list)
        }
    
    diag_agg = aggregate(diagnosis_metrics)
    med_agg = aggregate(medication_metrics)
    
    # Compute ROC-AUC
    roc_auc = 0.0
    if all_y_true and len(set(all_y_true)) > 1:
        try:
            roc_auc = roc_auc_score(all_y_true, all_y_scores)
        except ValueError:
            pass
    
    # Aggregate ranking metrics across all examples
    aggregated_ranking = {
        "precision_at_k": {},
        "recall_at_k": {},
        "f1_at_k": {},
        "ndcg_at_k": {},
        "mrr_at_k": {},
        "mean_rank": 0.0
    }
    
    if all_ranking_metrics:
        for k in list(k_values) + ["all"]:
            k_key = str(k) if isinstance(k, int) else k
            aggregated_ranking["precision_at_k"][k_key] = np.mean([
                m["precision_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
            aggregated_ranking["recall_at_k"][k_key] = np.mean([
                m["recall_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
            aggregated_ranking["f1_at_k"][k_key] = np.mean([
                m["f1_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
            aggregated_ranking["ndcg_at_k"][k_key] = np.mean([
                m["ndcg_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
            aggregated_ranking["mrr_at_k"][k_key] = np.mean([
                m["mrr_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
        
        valid_ranks = [m["mean_rank"] for m in all_ranking_metrics if m["mean_rank"] != float('inf')]
        aggregated_ranking["mean_rank"] = np.mean(valid_ranks) if valid_ranks else float('inf')
    
    # Compute overall metrics
    avg_ap = (diag_agg["average_precision"] + med_agg["average_precision"]) / 2
    avg_f1 = (diag_agg["f1"] + med_agg["f1"]) / 2
    diag_dyn_bin_acc = np.mean(diagnosis_dynamic_binary_accuracy) if diagnosis_dynamic_binary_accuracy else 0.0
    med_dyn_bin_acc = np.mean(medication_dynamic_binary_accuracy) if medication_dynamic_binary_accuracy else 0.0
    dynamic_binary_accuracy = (diag_dyn_bin_acc + med_dyn_bin_acc) / 2
    
    result = {
        "model_name": "STAGE 0: Frozen Encoder (Cosine Similarity)",
        "diagnosis": diag_agg,
        "medication": med_agg,
        "average_precision": avg_ap,
        "row_sent_f1": avg_f1,
        "overall_accuracy": avg_f1,
        "dynamic_f1": avg_f1,
        "dynamic_binary_accuracy": dynamic_binary_accuracy,
        "roc_auc": roc_auc,
        "examples_evaluated": len(diagnosis_metrics) + len(medication_metrics),
        "diagnosis_examples": len(diagnosis_metrics),
        "medication_examples": len(medication_metrics),
        "diagnosis_dynamic_binary_accuracy": diag_dyn_bin_acc,
        "medication_dynamic_binary_accuracy": med_dyn_bin_acc,
        "prediction_breakdown": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "total_ground_truth_positives": total_tp + total_fn
        },
        "diagnosis_prediction_breakdown": {
            "tp": diagnosis_tp,
            "fp": diagnosis_fp,
            "fn": diagnosis_fn,
            "total_ground_truth_positives": diagnosis_tp + diagnosis_fn
        },
        "medication_prediction_breakdown": {
            "tp": medication_tp,
            "fp": medication_fp,
            "fn": medication_fn,
            "total_ground_truth_positives": medication_tp + medication_fn
        },
        # Ranking metrics
        "ranking_metrics": aggregated_ranking,
        "ranking_raw_counts": ranking_raw_counts,
        "precision_at_k": aggregated_ranking["precision_at_k"],
        "recall_at_k": aggregated_ranking["recall_at_k"],
        "f1_at_k": aggregated_ranking["f1_at_k"],
        "ndcg_at_k": aggregated_ranking["ndcg_at_k"],
        "mrr_at_k": aggregated_ranking["mrr_at_k"],
        "mean_rank": aggregated_ranking["mean_rank"]
    }
    
    if collect_pair_scores:
        result["pair_scores_data"] = all_pair_scores_data
    
    # Print summary
    print(f"   Diagnosis - P: {round_half_up(diag_agg['precision'], 2):.2f}, R: {round_half_up(diag_agg['recall'], 2):.2f}, F1: {round_half_up(diag_agg['f1'], 2):.2f}, AP: {round_half_up(diag_agg['average_precision'], 2):.2f}")
    print(f"   Medication - P: {round_half_up(med_agg['precision'], 2):.2f}, R: {round_half_up(med_agg['recall'], 2):.2f}, F1: {round_half_up(med_agg['f1'], 2):.2f}, AP: {round_half_up(med_agg['average_precision'], 2):.2f}")
    print(f"   Overall - AP: {round_half_up(avg_ap, 2):.2f}, F1: {round_half_up(avg_f1, 2):.2f}, DynBinAcc: {round_half_up(dynamic_binary_accuracy, 2):.2f}, ROC-AUC: {round_half_up(roc_auc, 2):.2f}")
    
    # Print ranking metrics summary
    if aggregated_ranking["precision_at_k"]:
        print(f"   Ranking - P@5: {round_half_up(aggregated_ranking['precision_at_k'].get('5', 0), 2):.2f}, "
              f"R@5: {round_half_up(aggregated_ranking['recall_at_k'].get('5', 0), 2):.2f}, "
              f"NDCG@5: {round_half_up(aggregated_ranking['ndcg_at_k'].get('5', 0), 2):.2f}, "
              f"MRR@5: {round_half_up(aggregated_ranking['mrr_at_k'].get('5', 0), 2):.2f}, "
              f"MeanRank: {aggregated_ranking['mean_rank']:.1f}")
    
    return result


def comprehensive_stage_evaluation(
    model,
    test_examples: List[Dict],
    annotations: Dict,
    args: Dict[str, Any],
    stage_name: str,
    batch_size: int = 1,
    collect_pair_scores: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation for a single stage with full metrics.
    
    Returns:
        Dictionary with diagnosis/medication breakdown, ROC-AUC, pair_scores_data
    """
    print(f"\n{'='*60}")
    print(f"📊 {stage_name}")
    print("="*60)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Build anchor_id -> admission mapping
    anchor_to_admission = get_anchor_id_to_admission_mapping(annotations, test_examples)
    
    # Aggregation method
    aggregation_method = args.get("aggregation_method", "top_k_pairs")
    
    # Per-table metrics collection
    diagnosis_metrics = []
    medication_metrics = []
    diagnosis_dynamic_binary_accuracy = []
    medication_dynamic_binary_accuracy = []
    all_pair_scores_data = []
    
    # Prediction tracking
    total_tp, total_fp, total_fn = 0, 0, 0
    diagnosis_tp, diagnosis_fp, diagnosis_fn = 0, 0, 0
    medication_tp, medication_fp, medication_fn = 0, 0, 0
    all_y_true = []
    all_y_scores = []
    
    # Ranking metrics collection (per-example, then aggregated)
    k_values = [1, 3, 5, 10]
    all_ranking_metrics = []  # Store per-example ranking metrics
    ranking_raw_counts = {
        "queries_evaluated": 0,
        "total_rows": 0,
        "total_documents": 0,
        "total_candidate_pairs": 0,
        "total_ground_truth_pairs": 0,
        "hits_at_k": {str(k): 0 for k in k_values},
        "max_possible_hits_at_k": {str(k): 0 for k in k_values}
    }
    
    with torch.no_grad():
        for example_idx, example in enumerate(tqdm(test_examples, desc=f"Evaluating {stage_name}")):
            anchor_id = example.get("anchor_id")
            if anchor_id is None or anchor_id not in anchor_to_admission:
                continue
            
            admission_id, table_type = anchor_to_admission[anchor_id]
            annotation = annotations.get(admission_id, {})
            row_grounding = annotation.get("row_grounding", {})
            
            # Get ground truth pairs
            _, row_sentence_pairs = extract_mimic_row_sentence_pairs(row_grounding, table_type)
            if not row_sentence_pairs:
                continue
            
            # Extract rows/sentences from example
            rows = _extract_rows_from_example(example, table_type=table_type)
            
            if not rows:
                continue
            
            sentences = _extract_primary_sentences_from_example(example)
            
            if not sentences:
                continue
            
            try:
                # Get embeddings and pair scores
                row_embeddings = model.encode_sentences(rows, batch_size=batch_size)
                sentence_embeddings = model.encode_sentences(sentences, batch_size=batch_size)
                
                row_tensor = row_embeddings.unsqueeze(0).to(device)
                sentence_tensor = sentence_embeddings.unsqueeze(0).to(device)
                
                # Get pair scores from model
                if isinstance(model, BidirectionalTableTextModel):
                    _, pair_scores = model(row_tensor, sentence_tensor, aggregation_method=aggregation_method)
                    pair_scores_np = safe_tensor_to_numpy(pair_scores.squeeze(0).detach())
                else:
                    # FIXED: For unidirectional model, use CONTEXTUALIZED embeddings
                    # The get_contextualized_pair_scores() method computes:
                    # 1. Contextualized row embeddings (after cross-attention + FFN)
                    # 2. Cosine similarity between contextualized rows and original sentences
                    pair_scores = model.get_contextualized_pair_scores(row_tensor, sentence_tensor)
                    pair_scores_np = safe_tensor_to_numpy(pair_scores.squeeze(0).detach())
                
                num_rows, num_sentences = pair_scores_np.shape
                
                # Calculate metrics for this example
                metrics = calculate_mimic_grounding_metrics(pair_scores_np, row_sentence_pairs, num_rows, num_sentences)
                dynamic_binary_accuracy = calculate_dynamic_binary_accuracy_for_pairs(
                    pair_scores_np, row_sentence_pairs, num_rows, num_sentences
                )
                
                # Collect pair scores data
                if collect_pair_scores:
                    gt_set = set(row_sentence_pairs)
                    for i in range(num_rows):
                        for j in range(num_sentences):
                            is_gt = (i, j) in gt_set
                            all_pair_scores_data.append([i, j, float(pair_scores_np[i, j]), is_gt])
                            all_y_true.append(1 if is_gt else 0)
                            all_y_scores.append(float(pair_scores_np[i, j]))
                
                # Update prediction counts
                # Use dynamic threshold (mean of GT and non-GT scores)
                valid_gt_pairs = [(i, j) for i, j in row_sentence_pairs if 0 <= i < num_rows and 0 <= j < num_sentences]
                gt_pairs_set = set(valid_gt_pairs)
                gt_scores = [pair_scores_np[i, j] for i, j in valid_gt_pairs]
                non_gt_scores = [pair_scores_np[i, j] for i in range(num_rows) for j in range(num_sentences) if (i, j) not in gt_pairs_set]
                
                if gt_scores and non_gt_scores:
                    threshold = (np.mean(gt_scores) + np.mean(non_gt_scores)) / 2
                else:
                    threshold = np.median(pair_scores_np.flatten())
                
                predicted_pairs = set()
                for i in range(num_rows):
                    for j in range(num_sentences):
                        if pair_scores_np[i, j] >= threshold:
                            predicted_pairs.add((i, j))
                
                gt_pairs = gt_pairs_set
                example_tp = len(gt_pairs & predicted_pairs)
                example_fp = len(predicted_pairs - gt_pairs)
                example_fn = len(gt_pairs - predicted_pairs)

                total_tp += example_tp
                total_fp += example_fp
                total_fn += example_fn

                if table_type == "diagnosis":
                    diagnosis_tp += example_tp
                    diagnosis_fp += example_fp
                    diagnosis_fn += example_fn
                else:
                    medication_tp += example_tp
                    medication_fp += example_fp
                    medication_fn += example_fn

                ranking_raw_counts["queries_evaluated"] += 1
                ranking_raw_counts["total_rows"] += num_rows
                ranking_raw_counts["total_documents"] += num_sentences
                ranking_raw_counts["total_candidate_pairs"] += (num_rows * num_sentences)
                ranking_raw_counts["total_ground_truth_pairs"] += len(valid_gt_pairs)

                gt_flat = np.zeros(num_rows * num_sentences, dtype=int)
                for i, j in valid_gt_pairs:
                    gt_flat[i * num_sentences + j] = 1
                scores_flat = pair_scores_np.flatten()
                sorted_indices = np.argsort(scores_flat)[::-1]
                for k in k_values:
                    actual_k = min(k, len(scores_flat))
                    top_k_indices = sorted_indices[:actual_k]
                    hits = int(np.sum(gt_flat[top_k_indices]))
                    ranking_raw_counts["hits_at_k"][str(k)] += hits
                    ranking_raw_counts["max_possible_hits_at_k"][str(k)] += min(len(valid_gt_pairs), actual_k)
                
                # Calculate ranking metrics for this example
                example_ranking_metrics = calculate_pair_ranking_metrics(
                    pair_scores_np, row_sentence_pairs, num_rows, num_sentences, k_values
                )
                all_ranking_metrics.append(example_ranking_metrics)
                
                # Store by table type
                if table_type == "diagnosis":
                    diagnosis_metrics.append(metrics)
                    diagnosis_dynamic_binary_accuracy.append(dynamic_binary_accuracy)
                else:
                    medication_metrics.append(metrics)
                    medication_dynamic_binary_accuracy.append(dynamic_binary_accuracy)
                    
            except Exception as e:
                print(f"[WARNING] Evaluation error for example {example_idx}: {str(e)[:100]}...")
                continue
    
    # Aggregate per-table metrics
    def aggregate(m_list):
        if not m_list:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "average_precision": 0.0}
        return {
            "precision": np.mean([m["precision"] for m in m_list]),
            "recall": np.mean([m["recall"] for m in m_list]),
            "f1": np.mean([m["f1"] for m in m_list]),
            "jaccard": np.mean([m["jaccard"] for m in m_list]),
            "average_precision": np.mean([m.get("average_precision", 0.0) for m in m_list]),
            "examples": len(m_list)
        }
    
    diag_agg = aggregate(diagnosis_metrics)
    med_agg = aggregate(medication_metrics)
    
    # Compute ROC-AUC if we have data
    roc_auc = 0.0
    if all_y_true and len(set(all_y_true)) > 1:
        try:
            roc_auc = roc_auc_score(all_y_true, all_y_scores)
        except ValueError:
            pass
    
    # Aggregate ranking metrics across all examples
    aggregated_ranking = {
        "precision_at_k": {},
        "recall_at_k": {},
        "f1_at_k": {},
        "ndcg_at_k": {},
        "mrr_at_k": {},
        "mean_rank": 0.0
    }
    
    if all_ranking_metrics:
        # Aggregate each K value
        for k in list(k_values) + ["all"]:
            k_key = str(k) if isinstance(k, int) else k
            aggregated_ranking["precision_at_k"][k_key] = np.mean([
                m["precision_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
            aggregated_ranking["recall_at_k"][k_key] = np.mean([
                m["recall_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
            aggregated_ranking["f1_at_k"][k_key] = np.mean([
                m["f1_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
            aggregated_ranking["ndcg_at_k"][k_key] = np.mean([
                m["ndcg_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
            aggregated_ranking["mrr_at_k"][k_key] = np.mean([
                m["mrr_at_k"].get(k, 0.0) for m in all_ranking_metrics
            ])
        
        # Mean rank (filter out inf values)
        valid_ranks = [m["mean_rank"] for m in all_ranking_metrics if m["mean_rank"] != float('inf')]
        aggregated_ranking["mean_rank"] = np.mean(valid_ranks) if valid_ranks else float('inf')
    
    # Compute overall metrics
    avg_ap = (diag_agg["average_precision"] + med_agg["average_precision"]) / 2
    avg_f1 = (diag_agg["f1"] + med_agg["f1"]) / 2
    diag_dyn_bin_acc = np.mean(diagnosis_dynamic_binary_accuracy) if diagnosis_dynamic_binary_accuracy else 0.0
    med_dyn_bin_acc = np.mean(medication_dynamic_binary_accuracy) if medication_dynamic_binary_accuracy else 0.0
    dynamic_binary_accuracy = (diag_dyn_bin_acc + med_dyn_bin_acc) / 2
    
    result = {
        "model_name": stage_name,
        "diagnosis": diag_agg,
        "medication": med_agg,
        "average_precision": avg_ap,
        "row_sent_f1": avg_f1,
        "overall_accuracy": avg_f1,
        "dynamic_f1": avg_f1,
        "dynamic_binary_accuracy": dynamic_binary_accuracy,
        "roc_auc": roc_auc,
        "examples_evaluated": len(diagnosis_metrics) + len(medication_metrics),
        "diagnosis_examples": len(diagnosis_metrics),
        "medication_examples": len(medication_metrics),
        "diagnosis_dynamic_binary_accuracy": diag_dyn_bin_acc,
        "medication_dynamic_binary_accuracy": med_dyn_bin_acc,
        "prediction_breakdown": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "total_ground_truth_positives": total_tp + total_fn
        },
        "diagnosis_prediction_breakdown": {
            "tp": diagnosis_tp,
            "fp": diagnosis_fp,
            "fn": diagnosis_fn,
            "total_ground_truth_positives": diagnosis_tp + diagnosis_fn
        },
        "medication_prediction_breakdown": {
            "tp": medication_tp,
            "fp": medication_fp,
            "fn": medication_fn,
            "total_ground_truth_positives": medication_tp + medication_fn
        },
        # Ranking metrics
        "ranking_metrics": aggregated_ranking,
        "ranking_raw_counts": ranking_raw_counts,
        "precision_at_k": aggregated_ranking["precision_at_k"],
        "recall_at_k": aggregated_ranking["recall_at_k"],
        "f1_at_k": aggregated_ranking["f1_at_k"],
        "ndcg_at_k": aggregated_ranking["ndcg_at_k"],
        "mrr_at_k": aggregated_ranking["mrr_at_k"],
        "mean_rank": aggregated_ranking["mean_rank"]
    }
    
    if collect_pair_scores:
        result["pair_scores_data"] = all_pair_scores_data
    
    # Print summary
    print(f"   Diagnosis - P: {round_half_up(diag_agg['precision'], 2):.2f}, R: {round_half_up(diag_agg['recall'], 2):.2f}, F1: {round_half_up(diag_agg['f1'], 2):.2f}, AP: {round_half_up(diag_agg['average_precision'], 2):.2f}")
    print(f"   Medication - P: {round_half_up(med_agg['precision'], 2):.2f}, R: {round_half_up(med_agg['recall'], 2):.2f}, F1: {round_half_up(med_agg['f1'], 2):.2f}, AP: {round_half_up(med_agg['average_precision'], 2):.2f}")
    print(f"   Overall - AP: {round_half_up(avg_ap, 2):.2f}, F1: {round_half_up(avg_f1, 2):.2f}, DynBinAcc: {round_half_up(dynamic_binary_accuracy, 2):.2f}, ROC-AUC: {round_half_up(roc_auc, 2):.2f}")
    
    # Print ranking metrics summary
    if aggregated_ranking["precision_at_k"]:
        print(f"   Ranking - P@5: {round_half_up(aggregated_ranking['precision_at_k'].get('5', 0), 2):.2f}, "
              f"R@5: {round_half_up(aggregated_ranking['recall_at_k'].get('5', 0), 2):.2f}, "
              f"NDCG@5: {round_half_up(aggregated_ranking['ndcg_at_k'].get('5', 0), 2):.2f}, "
              f"MRR@5: {round_half_up(aggregated_ranking['mrr_at_k'].get('5', 0), 2):.2f}, "
              f"MeanRank: {aggregated_ranking['mean_rank']:.1f}")
    
    return result


def create_comprehensive_three_stage_visualizations(
    frozen_metrics, sophisticated_metrics, trained_metrics, output_dir
):
    """Create comprehensive visualizations comparing all 3 evaluation stages with ranking metrics."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARNING] matplotlib/seaborn not available, skipping visualizations")
        return
    
    print("Creating 3-stage comprehensive comparison visualization with ranking metrics...")
    
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create 3x3 figure for comprehensive metrics
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle('Complete 3-Stage Model Evolution Analysis (MIMIC)', fontsize=16, fontweight='bold')
    
    stage_names = ['Stage 0: Frozen', 'Stage 2: Sophisticated', 'Stage 3: Trained']
    stage_short = ['S0: Frozen', 'S2: Soph.', 'S3: Trained']
    stage_colors = ['#ff7f0e', '#d62728', '#1f77b4']  # Orange, Red, Blue
    metrics_list = [frozen_metrics, sophisticated_metrics, trained_metrics]
    
    # Helper to safely get ranking metric values
    def get_ranking_values(metric_key, k_values):
        values = []
        for m in metrics_list:
            k_dict = m.get(metric_key, {})
            k_vals = []
            for k in k_values:
                k_str = str(k)
                val = k_dict.get(k_str, k_dict.get(k, 0.0))
                k_vals.append(val if val is not None else 0.0)
            values.append(k_vals)
        return values
    
    k_values = [1, 3, 5, 10]
    
    # Row 0: Precision@K, Recall@K, NDCG@K line plots
    # 0,0 - Precision@K
    precision_values = get_ranking_values('precision_at_k', k_values)
    for idx, (prec_vals, name, color) in enumerate(zip(precision_values, stage_names, stage_colors)):
        marker = ['o', 's', '^', 'D'][idx]
        axes[0, 0].plot(k_values, prec_vals, f'{marker}-', label=name, linewidth=2.5, markersize=8, color=color)
    axes[0, 0].set_title('Precision@K Progression', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('K')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticks(k_values)
    
    # 0,1 - Recall@K
    recall_values = get_ranking_values('recall_at_k', k_values)
    for idx, (rec_vals, name, color) in enumerate(zip(recall_values, stage_names, stage_colors)):
        marker = ['o', 's', '^', 'D'][idx]
        axes[0, 1].plot(k_values, rec_vals, f'{marker}-', label=name, linewidth=2.5, markersize=8, color=color)
    axes[0, 1].set_title('Recall@K Progression', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('K')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticks(k_values)
    
    # 0,2 - NDCG@K
    ndcg_values = get_ranking_values('ndcg_at_k', k_values)
    for idx, (ndcg_vals, name, color) in enumerate(zip(ndcg_values, stage_names, stage_colors)):
        marker = ['o', 's', '^', 'D'][idx]
        axes[0, 2].plot(k_values, ndcg_vals, f'{marker}-', label=name, linewidth=2.5, markersize=8, color=color)
    axes[0, 2].set_title('NDCG@K Progression', fontweight='bold', fontsize=12)
    axes[0, 2].set_xlabel('K')
    axes[0, 2].set_ylabel('NDCG')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_xticks(k_values)
    
    # Row 1: MRR@K, F1@K, Key Metrics Bar Chart
    # 1,0 - MRR@K
    mrr_values = get_ranking_values('mrr_at_k', k_values)
    for idx, (mrr_vals, name, color) in enumerate(zip(mrr_values, stage_names, stage_colors)):
        marker = ['o', 's', '^', 'D'][idx]
        axes[1, 0].plot(k_values, mrr_vals, f'{marker}-', label=name, linewidth=2.5, markersize=8, color=color)
    axes[1, 0].set_title('MRR@K (Mean Reciprocal Rank)', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('K')
    axes[1, 0].set_ylabel('MRR')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xticks(k_values)
    
    # 1,1 - F1@K
    f1_values = get_ranking_values('f1_at_k', k_values)
    for idx, (f1_vals, name, color) in enumerate(zip(f1_values, stage_names, stage_colors)):
        marker = ['o', 's', '^', 'D'][idx]
        axes[1, 1].plot(k_values, f1_vals, f'{marker}-', label=name, linewidth=2.5, markersize=8, color=color)
    axes[1, 1].set_title('F1@K Progression', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('K')
    axes[1, 1].set_ylabel('F1')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks(k_values)
    
    # 1,2 - Key Metrics Bar Chart (AP, F1, ROC-AUC, NDCG@5)
    ap_values = [m.get('average_precision', 0) for m in metrics_list]
    f1_overall = [get_row_sent_f1_metric(m) for m in metrics_list]
    roc_values = [m.get('roc_auc', 0) for m in metrics_list]
    ndcg5_values = [m.get('ndcg_at_k', {}).get('5', m.get('ndcg_at_k', {}).get(5, 0)) for m in metrics_list]
    
    x = np.arange(len(stage_short))
    width = 0.2
    
    axes[1, 2].bar(x - 1.5*width, ap_values, width, label='Avg Precision', color='#1f77b4', alpha=0.8)
    axes[1, 2].bar(x - 0.5*width, f1_overall, width, label='F1', color='#ff7f0e', alpha=0.8)
    axes[1, 2].bar(x + 0.5*width, roc_values, width, label='ROC-AUC', color='#2ca02c', alpha=0.8)
    axes[1, 2].bar(x + 1.5*width, ndcg5_values, width, label='NDCG@5', color='#d62728', alpha=0.8)
    axes[1, 2].set_title('Key Metrics Comparison', fontweight='bold', fontsize=12)
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(stage_short, fontsize=9)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].set_ylim(0, 1)
    
    # Row 2: Diagnosis/Medication, Performance Evolution, Improvement
    # 2,0 - Diagnosis vs Medication Breakdown
    diag_f1 = [m.get('diagnosis', {}).get('f1', 0) for m in metrics_list]
    med_f1 = [m.get('medication', {}).get('f1', 0) for m in metrics_list]
    
    x = np.arange(len(stage_short))
    width = 0.35
    
    axes[2, 0].bar(x - width/2, diag_f1, width, label='Diagnosis', color='#1f77b4', alpha=0.8)
    axes[2, 0].bar(x + width/2, med_f1, width, label='Medication', color='#ff7f0e', alpha=0.8)
    axes[2, 0].set_title('F1 by Table Type', fontweight='bold', fontsize=12)
    axes[2, 0].set_ylabel('F1 Score')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(stage_short, fontsize=9)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    axes[2, 0].set_ylim(0, 1)
    
    # 2,1 - Performance Evolution Line Plot
    stages = [0, 1, 2]
    mean_ranks = [m.get('mean_rank', 999) for m in metrics_list]
    # Normalize mean rank for visualization (lower is better, so invert)
    max_rank = max([r for r in mean_ranks if r != float('inf')] or [100])
    norm_ranks = [1 - min(r, max_rank) / max_rank for r in mean_ranks]
    
    axes[2, 1].plot(stages, ap_values, 'o-', label='Avg Precision', linewidth=2.5, markersize=8)
    axes[2, 1].plot(stages, f1_overall, 's-', label='F1', linewidth=2.5, markersize=8)
    axes[2, 1].plot(stages, roc_values, '^-', label='ROC-AUC', linewidth=2.5, markersize=8)
    axes[2, 1].plot(stages, norm_ranks, 'D-', label='Rank Score (1-norm)', linewidth=2.5, markersize=8)
    axes[2, 1].set_title('Performance Evolution', fontweight='bold', fontsize=12)
    axes[2, 1].set_xlabel('Stage')
    axes[2, 1].set_ylabel('Score')
    axes[2, 1].set_xticks(stages)
    axes[2, 1].set_xticklabels(['S0', 'S1', 'S2'])
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim(0, 1)
    
    # 2,2 - Improvement from Baseline (AP)
    # Calculate difference first, then round the improvement
    improvements = [0, round_half_up(ap_values[1] - ap_values[0], 2), 
                   round_half_up(ap_values[2] - ap_values[0], 2)]
    colors = ['gray' if v == 0 else ('green' if v > 0 else 'red') for v in improvements]
    
    bars = axes[2, 2].bar(stage_short, improvements, color=colors, alpha=0.7)
    axes[2, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2, 2].set_title('AP Improvement vs Baseline', fontweight='bold', fontsize=12)
    axes[2, 2].set_ylabel('Δ Average Precision')
    axes[2, 2].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, improvements):
        if v != 0:
            height = bar.get_height()
            axes[2, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01 if v > 0 else height - 0.02,
                           f'{v:+.2f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "comprehensive_three_stage_comparison.png"
    save_plot_multi_format(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"Saved 3-stage comprehensive comparison: {plot_path}")
    plt.close()


def create_three_stage_roc_pr_curves(
    frozen_metrics, sophisticated_metrics, trained_metrics, output_dir
):
    """Create ROC and PR curves comparing all 3 stages."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available, skipping ROC/PR curves")
        return
    
    print("Creating 3-stage ROC and PR curve analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Complete 3-Stage ROC and PR Curve Evolution (MIMIC)', fontsize=16, fontweight='bold')
    
    stage_names = ['Stage 0: Frozen', 'Stage 2: Sophisticated', 'Stage 3: Trained']
    stage_colors = ['#ff7f0e', '#d62728', '#1f77b4']
    stage_linestyles = ['-', '--', '-.']
    
    stages_data = [frozen_metrics, sophisticated_metrics, trained_metrics]
    
    try:
        # ROC Curves
        ax1 = axes[0]
        for stage_data, name, color, ls in zip(stages_data, stage_names, stage_colors, stage_linestyles):
            if stage_data and 'pair_scores_data' in stage_data and stage_data['pair_scores_data']:
                pair_data = stage_data['pair_scores_data']
                labels = [1 if item[3] else 0 for item in pair_data]
                scores = [item[2] for item in pair_data]
                
                if len(set(labels)) > 1:
                    fpr, tpr, _ = roc_curve(labels, scores)
                    auc_score = stage_data.get('roc_auc', 0)
                    ax1.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                            label=f"{name} (AUC = {auc_score:.2f})")
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR Curves
        ax2 = axes[1]
        for stage_data, name, color, ls in zip(stages_data, stage_names, stage_colors, stage_linestyles):
            if stage_data and 'pair_scores_data' in stage_data and stage_data['pair_scores_data']:
                pair_data = stage_data['pair_scores_data']
                labels = [1 if item[3] else 0 for item in pair_data]
                scores = [item[2] for item in pair_data]
                
                if len(set(labels)) > 1:
                    precision, recall, _ = precision_recall_curve(labels, scores)
                    ap_score = stage_data.get('average_precision', 0)
                    ax2.plot(recall, precision, color=color, linestyle=ls, linewidth=2,
                            label=f"{name} (AP = {ap_score:.2f})")
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"[WARNING] Could not create ROC/PR curves: {e}")
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "three_stage_roc_pr_curves.png"
    save_plot_multi_format(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"Saved 3-stage ROC/PR curves: {plot_path}")
    plt.close()


def create_stage_progression_analysis(
    frozen_metrics, sophisticated_metrics, trained_metrics, output_dir
):
    """Create detailed stage progression analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available, skipping progression analysis")
        return
    
    print("Creating detailed stage progression analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed 3-Stage Progression Analysis (MIMIC)', fontsize=16, fontweight='bold')
    
    stage_names = ['Stage 0\nFrozen', 'Stage 2\nSophisticated', 'Stage 3\nTrained']
    stages = [0, 1, 2]
    metrics_list = [frozen_metrics, sophisticated_metrics, trained_metrics]
    
    # 1. Average Precision
    ap_values = [m.get('average_precision', 0) if m else 0 for m in metrics_list]
    axes[0, 0].plot(stages, ap_values, 'o-', linewidth=3, markersize=10, color='darkblue')
    axes[0, 0].fill_between(stages, ap_values, alpha=0.3, color='lightblue')
    for i in range(1, len(stages)):
        if metrics_list[i]:
            imp = round_half_up(ap_values[i] - ap_values[i-1], 2)
            axes[0, 0].annotate(f'+{imp:.2f}', xy=(stages[i], ap_values[i]),
                               xytext=(stages[i], ap_values[i] + 0.05),
                               ha='center', fontsize=9, fontweight='bold',
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    axes[0, 0].set_title('Average Precision Progression', fontweight='bold')
    axes[0, 0].set_xticks(stages)
    axes[0, 0].set_xticklabels(stage_names)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. F1 Score
    f1_values = [get_row_sent_f1_metric(m) if m else 0 for m in metrics_list]
    axes[0, 1].plot(stages, f1_values, 's-', linewidth=3, markersize=10, color='darkgreen')
    axes[0, 1].fill_between(stages, f1_values, alpha=0.3, color='lightgreen')
    for i in range(1, len(stages)):
        if metrics_list[i]:
            imp = round_half_up(f1_values[i] - f1_values[i-1], 2)
            axes[0, 1].annotate(f'+{imp:.2f}', xy=(stages[i], f1_values[i]),
                               xytext=(stages[i], f1_values[i] + 0.05),
                               ha='center', fontsize=9, fontweight='bold',
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    axes[0, 1].set_title('Dynamic F1 Progression', fontweight='bold')
    axes[0, 1].set_xticks(stages)
    axes[0, 1].set_xticklabels(stage_names)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ROC-AUC
    roc_values = [m.get('roc_auc', 0) if m else 0 for m in metrics_list]
    axes[1, 0].plot(stages, roc_values, '^-', linewidth=3, markersize=10, color='darkorange')
    axes[1, 0].fill_between(stages, roc_values, alpha=0.3, color='moccasin')
    for i in range(1, len(stages)):
        if metrics_list[i]:
            imp = round_half_up(roc_values[i] - roc_values[i-1], 2)
            axes[1, 0].annotate(f'+{imp:.2f}', xy=(stages[i], roc_values[i]),
                               xytext=(stages[i], roc_values[i] + 0.05),
                               ha='center', fontsize=9, fontweight='bold',
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    axes[1, 0].set_title('ROC-AUC Progression', fontweight='bold')
    axes[1, 0].set_xticks(stages)
    axes[1, 0].set_xticklabels(stage_names)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Diagnosis vs Medication Comparison
    diag_ap = [m.get('diagnosis', {}).get('average_precision', 0) if m else 0 for m in metrics_list]
    med_ap = [m.get('medication', {}).get('average_precision', 0) if m else 0 for m in metrics_list]
    
    x = np.arange(len(stages))
    width = 0.35
    axes[1, 1].bar(x - width/2, diag_ap, width, label='Diagnosis AP', color='#1f77b4', alpha=0.8)
    axes[1, 1].bar(x + width/2, med_ap, width, label='Medication AP', color='#ff7f0e', alpha=0.8)
    axes[1, 1].set_title('Table-Specific Average Precision', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(stage_names)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "detailed_stage_progression_analysis.png"
    save_plot_multi_format(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"Saved detailed progression analysis: {plot_path}")
    plt.close()


def generate_visualizations(results: Dict[str, Any], output_dir: str):
    """Generate comprehensive visualizations like MIMIC_Protrix_Evaluation."""
    evaluations = results.get("evaluations", {})
    
    # Extract metrics for each stage
    frozen_metrics = evaluations.get("stage_0_frozen_encoder")
    sophisticated_metrics = evaluations.get("stage_2_sophisticated_pretrain")
    
    # For trained models, use the first available one
    trained_metrics = None
    for key in ["stage_3_best_test_avg_precision", "stage_3_best_test_overall_acc", "stage_3_best"]:
        if key in evaluations:
            trained_metrics = evaluations[key]
            break
    
    # If we have full 3-stage data, create comprehensive visualizations
    if frozen_metrics and sophisticated_metrics and trained_metrics:
        create_comprehensive_three_stage_visualizations(
            frozen_metrics, sophisticated_metrics, trained_metrics, output_dir
        )
        create_three_stage_roc_pr_curves(
            frozen_metrics, sophisticated_metrics, trained_metrics, output_dir
        )
        create_stage_progression_analysis(
            frozen_metrics, sophisticated_metrics, trained_metrics, output_dir
        )
    else:
        # Fallback: create simple comparison chart with whatever data we have
        print("[INFO] Creating simplified visualization (not all stages available)")
        _create_simple_comparison_chart(results, output_dir)


def _create_simple_comparison_chart(results: Dict[str, Any], output_dir: str):
    """Fallback simple visualization when not all stages are available."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARNING] matplotlib/seaborn not available, skipping visualizations")
        return
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    viz_dir = Path(output_dir)
    
    # Collect data for plotting
    stage_names = []
    stage_aps = []
    stage_f1s = []
    stage_roc_aucs = []
    
    for stage_key, stage_data in results.get("evaluations", {}).items():
        if isinstance(stage_data, dict) and "average_precision" in stage_data:
            stage_names.append(stage_key)
            stage_aps.append(stage_data.get("average_precision", 0))
            stage_f1s.append(get_row_sent_f1_metric(stage_data))
            stage_roc_aucs.append(stage_data.get("roc_auc", 0))
    
    if not stage_names:
        print("[WARNING] No stage data for visualization")
        return
    
    # Create comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(stage_names))
    width = 0.6
    
    # Average Precision
    axes[0].bar(x, stage_aps, width, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Average Precision')
    axes[0].set_title('Average Precision by Stage')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stage_names, rotation=45, ha='right')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(stage_aps):
        axes[0].text(i, v + 0.02, f'{round_half_up(v, 2):.2f}', ha='center', fontsize=9)
    
    # F1 Score
    axes[1].bar(x, stage_f1s, width, color='forestgreen', alpha=0.8)
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score by Stage')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stage_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(stage_f1s):
        axes[1].text(i, v + 0.02, f'{round_half_up(v, 2):.2f}', ha='center', fontsize=9)
    
    # ROC-AUC
    axes[2].bar(x, stage_roc_aucs, width, color='coral', alpha=0.8)
    axes[2].set_ylabel('ROC-AUC')
    axes[2].set_title('ROC-AUC by Stage')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(stage_names, rotation=45, ha='right')
    axes[2].set_ylim(0, 1)
    for i, v in enumerate(stage_roc_aucs):
        axes[2].text(i, v + 0.02, f'{round_half_up(v, 2):.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    save_plot_multi_format(str(viz_dir / "comprehensive_stage_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved visualization to: {viz_dir / 'comprehensive_stage_comparison.png'}")




def run_post_training_evaluation(
    output_dir: str,
    results_dir: str = "Post_Training_Results",
    model_types: Optional[List[str]] = None,
    run_full_stages: bool = True,
    collect_pair_scores: bool = True,
    generate_plots: bool = True,
    row_sent_max_examples: Optional[int] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run complete post-training evaluation with all 4 stages and comprehensive metrics.
    
    Args:
        output_dir: Path to training output directory containing args.json and model checkpoints
        results_dir: Base directory to save evaluation results (default: Post_Training_Results)
        model_types: Which model checkpoints to evaluate
        run_full_stages: Whether to run full 4-stage evaluation
        collect_pair_scores: Whether to collect pair scores data
        generate_plots: Whether to generate visualization plots
        row_sent_max_examples: Optional cap for row-sentence evaluation examples
        device: Device to use for evaluation
    """
    args = load_training_args(output_dir)
    model_identity = normalize_model_results_identity(output_dir, args)
    run_folder_name = model_identity["results_dir_name"]
    run_results_dir = Path(results_dir) / run_folder_name
    run_results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🎯 POST-TRAINING EVALUATION (Comprehensive)")
    print("="*70)
    print(f"Model directory: {output_dir}")
    print(f"Model label: {model_identity['display_name']}")
    print(f"Results directory: {run_results_dir}")
    print(f"Device: {device}")
    print(f"Full 4-stage evaluation: {run_full_stages}")
    print(f"Collect pair scores: {collect_pair_scores}")
    print(f"Row-sentence max examples: {row_sent_max_examples if row_sent_max_examples else 'all'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    # Find checkpoints
    checkpoints = find_model_checkpoints(output_dir)
    
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints found in {output_dir}")
    
    print(f"\n[INFO] Found checkpoints: {list(checkpoints.keys())}")
    
    # Filter model types
    if model_types is None or "all" in model_types:
        model_types = list(checkpoints.keys())
    else:
        model_types = [mt for mt in model_types if mt in checkpoints]
    
    if not model_types:
        raise ValueError(f"No matching checkpoints")
    
    print(f"[INFO] Will evaluate: {model_types}")

    model_build_args = dict(args)
    try:
        reference_checkpoint = checkpoints[model_types[0]]
        reference_state_dict = torch.load(reference_checkpoint, map_location="cpu", weights_only=True)
        model_build_args = get_checkpoint_compatible_model_args(args, reference_state_dict)
    except Exception as exc:
        print(f"[WARNING] Could not infer checkpoint architecture compatibility up front: {exc}")
    finally:
        if "reference_state_dict" in locals():
            del reference_state_dict
    
    # Load test data
    default_test_file = str(DATASETS_ROOT / "mimic" / "test_row_level.json")
    default_annotation_file = str(DATASETS_ROOT / "mimic" / "Annotated_Test.json")
    test_file = resolve_dataset_path(
        args.get("row_sent_test_file") or args.get("test_file") or default_test_file,
        args,
    )
    annotation_file = resolve_dataset_path(
        args.get("row_sent_annotation_file") or default_annotation_file,
        args,
    )
    
    print(f"\n[INFO] Loading test data from: {test_file}")
    print(f"[INFO] Loading annotations from: {annotation_file}")
    
    test_examples, annotations = load_mimic_test_data_and_annotations(test_file, annotation_file)
    
    if not test_examples or not annotations:
        raise ValueError("Failed to load test data or annotations")

    full_test_examples = test_examples
    full_annotations = annotations
    loaded_test_example_count = len(test_examples)
    loaded_annotation_count = len(annotations)

    if row_sent_max_examples is not None and row_sent_max_examples > 0:
        test_examples, annotations = maybe_subsample_row_sent_eval_data(
            test_examples=test_examples,
            annotations=annotations,
            max_examples=row_sent_max_examples,
        )
        if not test_examples or not annotations:
            raise ValueError("Row-sentence smoke-test subsampling produced no evaluable examples")

    # Dataset statistics (useful for raw-count reporting in paper tables)
    train_file = args.get("train_file")
    val_file = args.get("eval_file")
    dataset_statistics = {}
    split_to_path = {
        "train": train_file,
        "val": val_file,
        "test": test_file,
    }
    for split_name, split_path in split_to_path.items():
        if not split_path:
            continue
        try:
            split_examples = load_row_level_dataset(split_path)
            dataset_statistics[split_name] = _compute_split_statistics(split_examples, split_name)
            dataset_statistics[split_name]["source_file"] = split_path
        except Exception as e:
            print(f"[WARNING] Could not compute {split_name} split statistics from {split_path}: {e}")

    # Test split matching stats against the full source annotations/test set
    anchor_to_admission = get_anchor_id_to_admission_mapping(full_annotations, full_test_examples)
    matched_test_examples = sum(1 for ex in full_test_examples if ex.get("anchor_id") in anchor_to_admission)
    if "test" in dataset_statistics:
        dataset_statistics["test"]["matched_to_annotations"] = matched_test_examples
        dataset_statistics["test"]["annotation_coverage_ratio"] = (
            matched_test_examples / len(full_test_examples) if full_test_examples else 0.0
        )
    eval_anchor_to_admission = get_anchor_id_to_admission_mapping(annotations, test_examples)
    matched_eval_examples = sum(1 for ex in test_examples if ex.get("anchor_id") in eval_anchor_to_admission)
    dataset_statistics["row_sent_eval"] = {
        "max_examples_requested": row_sent_max_examples,
        "examples_loaded": loaded_test_example_count,
        "examples_selected": len(test_examples),
        "annotations_loaded": loaded_annotation_count,
        "annotations_selected": len(annotations),
        "matched_to_annotations": matched_eval_examples,
        "annotation_coverage_ratio": (
            matched_eval_examples / len(test_examples) if test_examples else 0.0
        ),
    }
    dataset_statistics["annotation_summary"] = {
        "num_admissions": len(full_annotations),
        "num_anchor_mappings": len(anchor_to_admission),
        "num_admissions_selected_for_eval": len(annotations),
        "num_anchor_mappings_selected_for_eval": len(eval_anchor_to_admission),
    }
    
    # Results dictionary
    results = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "test_file": test_file,
            "annotation_file": annotation_file,
            "model_name": args.get("model_name", "unknown"),
            "model_display_name": model_identity["display_name"],
            "results_subdir_name": model_identity["results_dir_name"],
            "model_family": model_identity["family"],
            "use_bidirectional": model_identity["use_bidirectional"],
            "attention_direction": model_identity["attention_direction"],
            "output_dir_name": Path(output_dir).name,
            "row_sent_max_examples": row_sent_max_examples,
            "row_sent_examples_loaded": loaded_test_example_count,
            "row_sent_examples_selected": len(test_examples),
            "row_sent_annotations_loaded": loaded_annotation_count,
            "row_sent_annotations_selected": len(annotations),
            "device": device
        },
        "training_args": args,
        "dataset_statistics": dataset_statistics,
        "evaluations": {}
    }
    
    # Create sentence encoder (shared across stages)
    sentence_encoder = create_sentence_encoder(args, device)
    
    if run_full_stages:
        # Stage 0: Frozen Encoder (Pure Cosine Similarity - no cross-attention)
        print("\n" + "="*60)
        print("🔥 STAGE 0: Frozen Encoder Baseline (Cosine Similarity)")
        print("="*60)
        
        stage_0_metrics = evaluate_frozen_encoder_comprehensive(
            sentence_encoder=sentence_encoder,
            test_examples=test_examples,
            annotations=annotations,
            args=args,
            batch_size=args.get("eval_batch_size", 64),
            collect_pair_scores=collect_pair_scores,
            device=device
        )
        results["evaluations"]["stage_0_frozen_encoder"] = stage_0_metrics
        
        # Stage 2: Sophisticated Pre-training
        stage_2_model = create_sophisticated_pretrain_model(sentence_encoder, model_build_args, device)
        stage_2_metrics = comprehensive_stage_evaluation(
            model=stage_2_model,
            test_examples=test_examples,
            annotations=annotations,
            args=args,
            stage_name="STAGE 2: Sophisticated (Pre-training)",
            collect_pair_scores=collect_pair_scores
        )
        results["evaluations"]["stage_2_sophisticated_pretrain"] = stage_2_metrics
        del stage_2_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Clean up shared encoder
    del sentence_encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Stage 3: Trained Model(s)
    for model_type in model_types:
        checkpoint_path = checkpoints[model_type]
        
        model = load_model_from_checkpoint(checkpoint_path, model_build_args, device)
        
        stage_3_metrics = comprehensive_stage_evaluation(
            model=model,
            test_examples=test_examples,
            annotations=annotations,
            args=args,
            stage_name=f"STAGE 3: Trained ({model_type})",
            collect_pair_scores=collect_pair_scores
        )
        results["evaluations"][f"stage_3_{model_type}"] = stage_3_metrics
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Record total time and results directory info
    results["evaluation_info"]["total_time"] = (datetime.now() - start_time).total_seconds()
    results["evaluation_info"]["results_directory"] = str(run_results_dir)
    results["evaluation_info"]["model_directory"] = str(output_dir)
    
    # Save results to the run-specific results directory
    results_path = run_results_dir / "results_post_training_eval.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"\n{'='*70}")
    print("✅ POST-TRAINING EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"📁 Results saved to: {run_results_dir}")
    print(f"⏱️  Total time: {results['evaluation_info']['total_time']:.1f}s")
    
    # Generate visualizations in the run-specific results directory
    if generate_plots:
        generate_visualizations(results, str(run_results_dir))
    
    # Print summary
    print("\n📊 SUMMARY:")
    for eval_name, metrics in results["evaluations"].items():
        if isinstance(metrics, dict):
            ap = metrics.get('average_precision', 0.0)
            f1 = get_row_sent_f1_metric(metrics)
            roc = metrics.get('roc_auc', 0.0)
            print(f"   {eval_name}: AP={round_half_up(ap, 2):.2f}, F1={round_half_up(f1, 2):.2f}, ROC-AUC={round_half_up(roc, 2):.2f}")
    
    return results


def main():
    """Main entry point for post-training evaluation."""
    parser = argparse.ArgumentParser(
        description="Post-Training Evaluation Script (Comprehensive) - Full 4-stage evaluation with detailed metrics"
    )
    
    parser.add_argument(
        "--output_dir", type=str, 
        default="Input_Models/LOKI",
        help="Path to training output directory containing args.json and model checkpoints"
    )
    
    parser.add_argument(
        "--results_dir", type=str, default="Post_Training_Results",
        help="Base directory to save evaluation results. Unidirectional runs are normalized into 'Uni (R-S)' or 'Uni (S-R)' subfolders."
    )
    
    parser.add_argument(
        "--model_type", type=str, nargs="+", default=["all"],
        choices=["all", "best", "best_test_overall_acc", "best_test_avg_precision"],
        help="Which model checkpoint(s) to evaluate. Default: all available"
    )
    
    parser.add_argument(
        "--quick", action="store_true", default=False,
        help="Quick mode: only evaluate trained checkpoints (skips Stage 0 and Stage 2)"
    )
    
    parser.add_argument(
        "--skip_pair_scores", action="store_true", default=False,
        help="Skip collecting pair scores data (reduces output file size)"
    )
    
    parser.add_argument(
        "--row_sent_max_examples", type=int, default=None,
        help="Maximum number of row-sentence test examples to evaluate (default: all examples)"
    )

    parser.add_argument(
        "--no_plots", action="store_true", default=False,
        help="Skip generating visualization plots"
    )
    
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run evaluation on (cuda or cpu)"
    )
    parser.add_argument(
        "--download_models", action="store_true", default=False,
        help="Download the requested published model folder from Hugging Face when it is missing locally"
    )
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.download_models and not Path(args.output_dir).is_dir():
        download_input_models(
            destination=str(Path(args.output_dir).parent),
            model_names=[Path(args.output_dir).name],
        )
    
    # Run evaluation
    try:
        results = run_post_training_evaluation(
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            model_types=args.model_type,
            run_full_stages=not args.quick,
            collect_pair_scores=not args.skip_pair_scores,
            generate_plots=not args.no_plots,
            row_sent_max_examples=args.row_sent_max_examples,
            device=args.device
        )
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
