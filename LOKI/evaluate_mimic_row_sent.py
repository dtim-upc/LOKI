"""
MIMIC Row-Sentence Evaluation Module

This module provides row-sentence level evaluation capabilities for MIMIC data.
Unlike Protrix (which uses highlighted_cells), MIMIC annotations use a richer format
with separate diagnosis/medication row grounding and mention types.

Annotation format (from Annotated_Test.json):
{
  "admission_id": {
    "row_grounding": {
      "diagnosis": {"row_idx": {"sentences": [...], "mention_types": [...], "_vote_count": N}},
      "medication": {"row_idx": {"sentences": [...], "mention_types": [...], "_vote_count": N}}
    },
    ...
  }
}

Evaluation metrics:
- Level A (Row Grounding): precision, recall, f1, jaccard per table type (diagnosis/medication)
- Level B (Relationships): TODO - to be implemented later
"""

# Import optimization utilities for faster evaluation
try:
    from unsloth_encoder import (
        UNSLOTH_AVAILABLE,
        FAST_SENTENCE_TRANSFORMER_AVAILABLE,
        TORCH_COMPILE_AVAILABLE,
        optimize_model_for_inference,
    )
except (ImportError, NotImplementedError, Exception):
    UNSLOTH_AVAILABLE = False
    FAST_SENTENCE_TRANSFORMER_AVAILABLE = False
    TORCH_COMPILE_AVAILABLE = False
    def optimize_model_for_inference(model, **kwargs):
        return model
        
import torch
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from sklearn.metrics import average_precision_score

from models import TableTextEmbeddingModel, BidirectionalTableTextModel

# Import BFloat16-safe conversion function
from row_sentence_eval import safe_tensor_to_numpy
from data import _extract_rows_robust, _extract_sentences_robust, _extract_table_rows_for_model


def load_mimic_annotations(annotation_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load MIMIC annotations from Annotated_Test.json.
    
    Args:
        annotation_file: Path to the MIMIC annotation file
        
    Returns:
        Dictionary mapping admission_id to annotation data, including:
        - row_grounding.diagnosis: {row_idx: {sentences: [...], mention_types: [...]}}
        - row_grounding.medication: {row_idx: {sentences: [...], mention_types: [...]}}
        - diagnosis_anchor_id, medication_anchor_id
    """
    if not Path(annotation_file).exists():
        print(f"[WARNING] MIMIC annotation file not found: {annotation_file}")
        return {}
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # The format is {admission_id: {...annotation_data...}}
        annotations = {}
        for admission_id, annotation in data.items():
            annotations[admission_id] = annotation
        
        # Count statistics
        total_diagnosis_rows = 0
        total_medication_rows = 0
        for admission_id, ann in annotations.items():
            row_grounding = ann.get("row_grounding", {})
            total_diagnosis_rows += len(row_grounding.get("diagnosis", {}))
            total_medication_rows += len(row_grounding.get("medication", {}))
        
        print(f"[INFO] Loaded {len(annotations)} MIMIC annotations")
        print(f"       Diagnosis row groundings: {total_diagnosis_rows}")
        print(f"       Medication row groundings: {total_medication_rows}")
        
        return annotations
        
    except Exception as e:
        print(f"[ERROR] Error loading MIMIC annotations: {e}")
        return {}


def get_anchor_id_to_admission_mapping(annotations: Dict[str, Dict], test_examples: List[Dict]) -> Dict[int, Tuple[str, str]]:
    """
    Build a mapping from anchor_id to (admission_id, table_type).
    
    This is needed because test_examples use anchor_id, but annotations use admission_id.
    
    Args:
        annotations: MIMIC annotations keyed by admission_id
        test_examples: List of test examples with anchor_id
        
    Returns:
        Dictionary mapping anchor_id -> (admission_id, table_type)
    """
    anchor_to_admission = {}
    
    for admission_id, ann in annotations.items():
        # Map diagnosis anchor
        diag_anchor = ann.get("diagnosis_anchor_id")
        if diag_anchor is not None:
            anchor_to_admission[diag_anchor] = (admission_id, "diagnosis")
        
        # Map medication anchor
        med_anchor = ann.get("medication_anchor_id")
        if med_anchor is not None:
            anchor_to_admission[med_anchor] = (admission_id, "medication")
    
    return anchor_to_admission


def extract_mimic_row_sentence_pairs(
    row_grounding: Dict[str, Dict],
    table_type: str
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Extract row indices and row-sentence pairs from MIMIC row_grounding.
    
    Args:
        row_grounding: {"diagnosis": {...}, "medication": {...}}
        table_type: "diagnosis" or "medication"
        
    Returns:
        Tuple of (unique_row_indices, row_sentence_pairs)
        Note: row indices are converted to 0-based (annotation uses 1-based row_idx keys)
    """
    table_grounding = row_grounding.get(table_type, {})
    
    if not table_grounding:
        return [], []
    
    row_indices = []
    pairs = []
    
    for row_idx_str, grounding_info in table_grounding.items():
        # Convert 1-based row_idx to 0-based
        row_idx = int(row_idx_str) - 1
        row_indices.append(row_idx)
        
        sentences = grounding_info.get("sentences", [])
        for sent_idx in sentences:
            # Note: sentence indices in annotations are 0-based already
            pairs.append((row_idx, sent_idx))
    
    return sorted(set(row_indices)), pairs


def calculate_mimic_grounding_metrics(
    pair_scores: np.ndarray,
    row_sentence_pairs: List[Tuple[int, int]],
    num_rows: int,
    num_sentences: int
) -> Dict[str, float]:
    """
    Calculate grounding metrics for MIMIC row-sentence evaluation.
    
    Returns:
        Dictionary with precision, recall, f1, jaccard metrics
    """
    if not row_sentence_pairs or pair_scores.size == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'jaccard': 0.0,
            'gt_score_mean': 0.0,
            'non_gt_score_mean': 0.0,
            'score_separation': 0.0,
            'average_precision': 0.0,
            'oob_pairs_filtered': 0
        }
    
    # Filter ground truth pairs to only those within score matrix bounds.
    # Annotations may reference sentence/row indices beyond what the test data contains
    # (e.g., annotations created against a different data version with more sentences).
    valid_pairs = [(i, j) for i, j in row_sentence_pairs
                   if 0 <= i < num_rows and 0 <= j < num_sentences]
    oob_count = len(row_sentence_pairs) - len(valid_pairs)
    
    if not valid_pairs:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'jaccard': 0.0,
            'gt_score_mean': 0.0,
            'non_gt_score_mean': 0.0,
            'score_separation': 0.0,
            'average_precision': 0.0,
            'oob_pairs_filtered': oob_count
        }
    
    # Create ground truth set from validated in-bounds pairs only
    gt_pairs = set(valid_pairs)
    
    # Get scores for ground truth and non-ground truth pairs
    gt_scores = []
    for i, j in valid_pairs:
        gt_scores.append(pair_scores[i, j])
    
    non_gt_scores = []
    for i in range(num_rows):
        for j in range(num_sentences):
            if (i, j) not in gt_pairs:
                non_gt_scores.append(pair_scores[i, j])
    
    # Calculate score statistics
    gt_score_mean = np.mean(gt_scores) if gt_scores else 0.0
    non_gt_score_mean = np.mean(non_gt_scores) if non_gt_scores else 0.0
    score_separation = gt_score_mean - non_gt_score_mean
    
    # Use threshold-based classification
    if gt_scores and non_gt_scores:
        threshold = (gt_score_mean + non_gt_score_mean) / 2.0
    else:
        threshold = np.median(pair_scores.flatten())
    
    # Create predictions based on threshold
    predicted_pairs = set()
    for i in range(num_rows):
        for j in range(num_sentences):
            if pair_scores[i, j] >= threshold:
                predicted_pairs.add((i, j))
    
    # Calculate precision, recall, F1, Jaccard
    tp = len(gt_pairs & predicted_pairs)
    fp = len(predicted_pairs - gt_pairs)
    fn = len(gt_pairs - predicted_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Jaccard similarity
    union = len(gt_pairs | predicted_pairs)
    jaccard = tp / union if union > 0 else 0.0

    # Calculate Average Precision
    y_true = np.zeros((num_rows, num_sentences), dtype=int)
    for i, j in valid_pairs:
        y_true[i, j] = 1
    
    y_scores = pair_scores.flatten()
    y_true_flat = y_true.flatten()
    
    # Avoid AP error if all y_true are 0 or 1
    if len(np.unique(y_true_flat)) > 1:
        avg_precision = average_precision_score(y_true_flat, y_scores)
    else:
        avg_precision = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
        'average_precision': avg_precision,
        'gt_score_mean': gt_score_mean,
        'non_gt_score_mean': non_gt_score_mean,
        'score_separation': score_separation,
        'oob_pairs_filtered': oob_count
    }


def evaluate_frozen_encoder_mimic(
    sentence_encoder,
    examples: List[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    batch_size: int = 1,
    max_examples: Optional[int] = None,
    test_cache = None,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluates MIMIC data using ONLY the frozen sentence encoder (Stage 0 Baseline).
    
    Args:
        sentence_encoder: The sentence transformer encoder
        examples: Test examples
        annotations: MIMIC annotations
        
    Returns:
        Dict with 'average_precision', 'f1' (F1)
    """
    
    class FrozenModelWrapper:
        def __init__(self, encoder):
            self.sentence_encoder = encoder
            # Pre-compute device to avoid generator issues
            self._device = self._detect_device()
        
        def _detect_device(self):
            """Detect the device of the encoder."""
            try:
                return self.sentence_encoder[0].auto_model.embeddings.word_embeddings.weight.device
            except:
                if hasattr(self.sentence_encoder, "device"):
                    return self.sentence_encoder.device
                return torch.device("cpu")
        
        def parameters(self):
            # Return a list with a single dummy parameter on the correct device
            # This avoids generator lifecycle issues with next()
            dummy = torch.tensor(0.0, device=self._device, requires_grad=False)
            return iter([dummy])
            
        def eval(self):
            pass
            
        def __call__(self, row_tensor, sentence_tensor, aggregation_method=None):
            # Compute cosine similarity
            # row_tensor: [1, num_rows, dim]
            # sentence_tensor: [1, num_sents, dim]
            
            rows = row_tensor.squeeze(0)  # [num_rows, dim]
            sents = sentence_tensor.squeeze(0)  # [num_sents, dim]
            
            # Normalize
            rows = torch.nn.functional.normalize(rows, p=2, dim=1)
            sents = torch.nn.functional.normalize(sents, p=2, dim=1)
            
            # Cosine sim: [num_rows, num_sents]
            sim_matrix = torch.matmul(rows, sents.t())
            
            # Return score (overall) and pair scores
            # Frozen encoder has no overall score meaningful here, just return 0.0
            return torch.tensor(0.0), sim_matrix
            
        def encode_rows(self, rows, batch_size=16):
            return self.sentence_encoder.encode(rows, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
            
        def encode_sentences(self, sents, batch_size=16):
            return self.sentence_encoder.encode(sents, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
        
        def get_contextualized_pair_scores(self, row_tensor, sentence_tensor):
            """For frozen encoder, return raw cosine similarity (no contextualization)."""
            # Compute cosine similarity directly
            rows = row_tensor.squeeze(0)  # [num_rows, dim]
            sents = sentence_tensor.squeeze(0)  # [num_sents, dim]
            
            # Normalize
            rows = torch.nn.functional.normalize(rows, p=2, dim=1)
            sents = torch.nn.functional.normalize(sents, p=2, dim=1)
            
            # Cosine sim: [num_rows, num_sents]
            sim_matrix = torch.matmul(rows, sents.t())
            
            # Return with batch dimension: [1, num_rows, num_sents]
            return sim_matrix.unsqueeze(0)

    # Use the wrapper with the standard evaluation function
    wrapper = FrozenModelWrapper(sentence_encoder)
    
    metrics = evaluate_mimic_row_grounding(
        model=wrapper,
        examples=examples,
        annotations=annotations,
        batch_size=batch_size,
        max_examples=max_examples,
        test_cache=test_cache,
        aggregation_method="frozen"
    )
    
    # Aggregate AP and F1 across all tables
    level_a = metrics.get("level_a_row_grounding", {})
    diag_metrics = level_a.get("diagnosis", {})
    med_metrics = level_a.get("medication", {})
    
    avg_ap = (diag_metrics.get("average_precision", 0.0) + med_metrics.get("average_precision", 0.0)) / 2
    avg_f1 = (diag_metrics.get("f1", 0.0) + med_metrics.get("f1", 0.0)) / 2
    
    return {
        'average_precision': avg_ap,
        'f1': avg_f1,  # Use F1 as proxy for "accuracy" in summary table
        'examples_evaluated': metrics.get('examples_evaluated', 0)
    }


def evaluate_mimic_with_model(
    model,
    test_examples: List[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    max_examples: Optional[int] = None,
    test_cache = None,
    verbose: bool = False,  # NEW: Control debug output
    **kwargs
) -> Dict[str, float]:
    """
    Wrapper for Stage 1/2 evaluation calls from train.py
    """
    metrics = evaluate_mimic_row_grounding(
        model=model,
        examples=test_examples,
        annotations=annotations,
        max_examples=max_examples,
        test_cache=test_cache,
        verbose=verbose
    )
    
    level_a = metrics.get("level_a_row_grounding", {})
    diag_metrics = level_a.get("diagnosis", {})
    med_metrics = level_a.get("medication", {})
    
    avg_ap = (diag_metrics.get("average_precision", 0.0) + med_metrics.get("average_precision", 0.0)) / 2
    avg_f1 = (diag_metrics.get("f1", 0.0) + med_metrics.get("f1", 0.0)) / 2
    
    return {
        'average_precision': avg_ap,
        'f1': avg_f1,
        'examples_evaluated': metrics.get('examples_evaluated', 0)
    }


def evaluate_mimic_row_grounding(
    model,
    examples: List[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    batch_size: int = 1,
    max_examples: Optional[int] = None,
    aggregation_method: str = None,
    test_cache = None,
    verbose: bool = False  # NEW: Control debug output
) -> Dict[str, Any]:
    """
    Evaluate Level A (Row Grounding) for MIMIC data.
    
    This evaluates how well the model grounds table rows to sentences,
    separately for diagnosis and medication tables.
    
    Args:
        model: The model to evaluate
        examples: List of test examples (from test_row_level_v2.json)
        annotations: MIMIC annotations (from Annotated_Test.json)
        batch_size: Batch size for encoding
        max_examples: Maximum examples to evaluate (None = all)
        aggregation_method: Aggregation method to use
        test_cache: Optional embedding cache
        verbose: Whether to print debug output for first few examples
        
    Returns:
        Dictionary with metrics:
        {
            "level_a_row_grounding": {
                "diagnosis": {"precision": ..., "recall": ..., "f1": ..., "jaccard": ...},
                "medication": {"precision": ..., "recall": ..., "f1": ..., "jaccard": ...}
            },
            "examples_evaluated": N
        }
    """
    model.eval()
    device = next(model.parameters()).device
    
    # CRITICAL: Detect model dtype to ensure embeddings match model component dtypes
    model_dtype = next(model.parameters()).dtype
    
    # Build anchor_id -> admission mapping
    anchor_to_admission = get_anchor_id_to_admission_mapping(annotations, examples)
    
    # Determine aggregation method
    if aggregation_method is None:
        if isinstance(model, BidirectionalTableTextModel):
            aggregation_method = "top_k_pairs"
        else:
            aggregation_method = "entropy_regularized"
    
    # Collect metrics per table type
    diagnosis_metrics = []
    medication_metrics = []
    examples_processed = 0
    examples_skipped_no_valid_pairs = 0
    total_oob_pairs = 0
    
    with torch.no_grad():
        for example_idx, example in enumerate(examples):
            if max_examples is not None and examples_processed >= max_examples:
                break
            
            anchor_id = example.get("anchor_id")
            if anchor_id is None or anchor_id not in anchor_to_admission:
                continue
            
            admission_id, table_type = anchor_to_admission[anchor_id]
            annotation = annotations.get(admission_id, {})
            row_grounding = annotation.get("row_grounding", {})
            
            # Get ground truth pairs for this table type
            _, row_sentence_pairs = extract_mimic_row_sentence_pairs(row_grounding, table_type)
            
            if not row_sentence_pairs:
                continue
            
            # 1. Try unified extraction first
            rows = _extract_rows_robust(example)
            
            # 2. If empty or missing, try MIMIC 'tables' format using the known table_type
            if not rows and "tables" in example:
                tables = example.get("tables", {})
                target_table = tables.get(table_type, {})
                if "rows" in target_table:
                    for row in target_table["rows"]:
                        if isinstance(row, dict):
                            formatted_text = row.get("formatted", "")
                            if formatted_text:
                                rows.append(formatted_text)
                        elif isinstance(row, str) and row:
                            rows.append(row)
            
            if not rows:
                continue
            
            # Extract sentences from primary_positive
            # MIMIC format uses dict: {"0": {"text": "..."}, "1": {"text": "..."}}
            # Protrix format uses list: ["sentence 1", "sentence 2"]
            primary_positive = example.get("primary_positive", {})
            raw_sentences = primary_positive.get("sentences", [])
            
            sentences = []
            if isinstance(raw_sentences, dict):
                # MIMIC format - extract text from dict entries
                try:
                    sorted_keys = sorted(raw_sentences.keys(), key=lambda k: int(k))
                    for k in sorted_keys:
                        item = raw_sentences[k]
                        if isinstance(item, dict):
                            text = item.get("text", "")
                            if text:
                                sentences.append(text)
                        elif isinstance(item, str) and item:
                            sentences.append(item)
                except (ValueError, TypeError):
                    # Fallback for non-integer keys
                    for item in raw_sentences.values():
                        if isinstance(item, dict):
                            text = item.get("text", "")
                            if text:
                                sentences.append(text)
                        elif isinstance(item, str) and item:
                            sentences.append(item)
            elif isinstance(raw_sentences, list):
                # Protrix format - list of strings or dicts
                for item in raw_sentences:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            sentences.append(text)
                    elif isinstance(item, str) and item:
                        sentences.append(item)
            
            if not sentences:
                continue
            
            try:
                # Get embeddings
                if test_cache is not None:
                    row_embeddings = test_cache.get_table_embeddings(anchor_id)
                    if row_embeddings is None:
                        row_embeddings = model.encode_sentences(rows, batch_size=batch_size)
                    
                    context_id = primary_positive.get("id")
                    sentence_embeddings = None
                    if context_id is not None:
                        sentence_embeddings = test_cache.get_context_embeddings(context_id)
                    if sentence_embeddings is None:
                        sentence_embeddings = model.encode_sentences(sentences, batch_size=batch_size)
                else:
                    row_embeddings = model.encode_sentences(rows, batch_size=batch_size)
                    sentence_embeddings = model.encode_sentences(sentences, batch_size=batch_size)
                
                # Add batch dimension and convert to model dtype
                row_tensor = row_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
                sentence_tensor = sentence_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
                
                # Get pair scores
                if isinstance(model, BidirectionalTableTextModel):
                    _, pair_scores = model(row_tensor, sentence_tensor, aggregation_method=aggregation_method)
                    pair_scores_np = safe_tensor_to_numpy(pair_scores.squeeze(0).detach())
                    
                    # DEBUG: Also compute raw cosine similarity for comparison (Stage 0 style)
                    if verbose and examples_processed < 3:  # Only log for first 3 examples when verbose
                        raw_rows = row_tensor.squeeze(0)
                        raw_sents = sentence_tensor.squeeze(0)
                        raw_rows_norm = torch.nn.functional.normalize(raw_rows, p=2, dim=1)
                        raw_sents_norm = torch.nn.functional.normalize(raw_sents, p=2, dim=1)
                        raw_sim = torch.matmul(raw_rows_norm, raw_sents_norm.t())
                        raw_sim_np = safe_tensor_to_numpy(raw_sim.detach())
                        
                        # Compare score separations
                        gt_pairs_set = set(row_sentence_pairs)
                        # Raw similarities
                        raw_gt_scores = [raw_sim_np[i, j] for i, j in row_sentence_pairs if 0 <= i < raw_sim_np.shape[0] and 0 <= j < raw_sim_np.shape[1]]
                        raw_non_gt = [raw_sim_np[i, j] for i in range(raw_sim_np.shape[0]) for j in range(raw_sim_np.shape[1]) if (i, j) not in gt_pairs_set]
                        raw_sep = np.mean(raw_gt_scores) - np.mean(raw_non_gt) if raw_gt_scores and raw_non_gt else 0.0
                        
                        # Refined pair scores
                        ref_gt_scores = [pair_scores_np[i, j] for i, j in row_sentence_pairs if 0 <= i < pair_scores_np.shape[0] and 0 <= j < pair_scores_np.shape[1]]
                        ref_non_gt = [pair_scores_np[i, j] for i in range(pair_scores_np.shape[0]) for j in range(pair_scores_np.shape[1]) if (i, j) not in gt_pairs_set]
                        ref_sep = np.mean(ref_gt_scores) - np.mean(ref_non_gt) if ref_gt_scores and ref_non_gt else 0.0
                        
                        print(f"    Example {example_idx} ({table_type}): Raw sep={raw_sep:.3f}, Refined sep={ref_sep:.3f}, Delta={ref_sep - raw_sep:+.3f}")
                else:
                    # FIXED: For unidirectional model, use CONTEXTUALIZED embeddings
                    # The get_contextualized_pair_scores() method computes:
                    # 1. Contextualized row embeddings (after cross-attention + FFN)
                    # 2. Cosine similarity between contextualized rows and original sentences
                    # This captures what the model has learned, unlike raw frozen embeddings
                    pair_scores = model.get_contextualized_pair_scores(row_tensor, sentence_tensor)
                    pair_scores_np = safe_tensor_to_numpy(pair_scores.squeeze(0).detach())
                
                num_rows, num_sentences = pair_scores_np.shape
                
                # Filter GT pairs to score matrix bounds before computing metrics
                valid_pairs = [(r, s) for r, s in row_sentence_pairs
                               if 0 <= r < num_rows and 0 <= s < num_sentences]
                oob = len(row_sentence_pairs) - len(valid_pairs)
                total_oob_pairs += oob
                
                if not valid_pairs:
                    examples_skipped_no_valid_pairs += 1
                    continue
                
                # Calculate metrics
                metrics = calculate_mimic_grounding_metrics(
                    pair_scores_np, valid_pairs, num_rows, num_sentences
                )
                
                # Store by table type
                if table_type == "diagnosis":
                    diagnosis_metrics.append(metrics)
                else:
                    medication_metrics.append(metrics)
                
                examples_processed += 1
                
            except Exception as e:
                print(f"[WARNING] MIMIC eval error for example {example_idx}: {str(e)[:100]}...")
                continue
    
    # Aggregate metrics
    def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
        if not metrics_list:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "average_precision": 0.0}
        
        return {
            "precision": np.mean([m["precision"] for m in metrics_list]),
            "recall": np.mean([m["recall"] for m in metrics_list]),
            "f1": np.mean([m["f1"] for m in metrics_list]),
            "jaccard": np.mean([m["jaccard"] for m in metrics_list]),
            "average_precision": np.mean([m.get("average_precision", 0.0) for m in metrics_list])
        }
    
    result = {
        "level_a_row_grounding": {
            "diagnosis": aggregate_metrics(diagnosis_metrics),
            "medication": aggregate_metrics(medication_metrics)
        },
        "examples_evaluated": examples_processed,
        "diagnosis_examples": len(diagnosis_metrics),
        "medication_examples": len(medication_metrics)
    }
    
    print(f"[INFO] MIMIC Row Grounding evaluation: {examples_processed} examples")
    print(f"       Diagnosis: {len(diagnosis_metrics)} examples, F1={result['level_a_row_grounding']['diagnosis']['f1']:.3f}")
    print(f"       Medication: {len(medication_metrics)} examples, F1={result['level_a_row_grounding']['medication']['f1']:.3f}")
    if examples_skipped_no_valid_pairs > 0:
        print(f"       [WARN]  Skipped {examples_skipped_no_valid_pairs} examples (all GT pairs out-of-bounds)")
    if total_oob_pairs > 0:
        print(f"       [WARN]  Filtered {total_oob_pairs} out-of-bounds GT pairs across all examples")
    
    return result


def quick_mimic_row_grounding_eval(
    model,
    test_examples: List[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    max_examples: Optional[int] = None,
    aggregation_method: str = None,
    test_cache = None,
    verbose: bool = False  # NEW: Control debug output (default False for training)
) -> Dict[str, Any]:
    """
    Quick MIMIC row grounding evaluation for use during training.
    
    This is the main entry point for training-time evaluation.
    
    Args:
        model: The model to evaluate
        test_examples: List of test examples
        annotations: MIMIC annotation dictionary
        max_examples: Maximum examples to evaluate (None = all)
        aggregation_method: Aggregation method
        test_cache: Optional embedding cache
        verbose: Whether to print debug output
        
    Returns:
        Dictionary with row grounding metrics for diagnosis and medication
    """
    return evaluate_mimic_row_grounding(
        model=model,
        examples=test_examples,
        annotations=annotations,
        batch_size=1,
        max_examples=max_examples,
        aggregation_method=aggregation_method,
        test_cache=test_cache,
        verbose=verbose
    )


def load_mimic_test_data_and_annotations(
    test_file: str,
    annotation_file: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Load MIMIC test data and annotations for row grounding evaluation.
    
    Args:
        test_file: Path to test dataset (test_row_level_v2.json)
        annotation_file: Path to annotations (Annotated_Test.json)
        
    Returns:
        Tuple of (test_examples, annotations)
    """
    try:
        from data import load_row_level_dataset
        test_examples = load_row_level_dataset(test_file)
        annotations = load_mimic_annotations(annotation_file)
        
        # Build mapping and check consistency
        anchor_to_admission = get_anchor_id_to_admission_mapping(annotations, test_examples)
        
        matched_examples = sum(1 for ex in test_examples if ex.get("anchor_id") in anchor_to_admission)
        
        print(f"[INFO] MIMIC evaluation data loaded:")
        print(f"       Test examples: {len(test_examples)}")
        print(f"       Annotations: {len(annotations)} admissions")
        print(f"       Matched examples: {matched_examples}")
        
        return test_examples, annotations
        
    except Exception as e:
        print(f"[ERROR] Error loading MIMIC test data: {e}")
        return [], {}


# ============================================================================
# TODO: Level B (Relationship) Evaluation - To be implemented later
# ============================================================================
# 
# def evaluate_mimic_relationships(model, examples, annotations) -> Dict:
#     """
#     Level B: Drug-Diagnosis relationship evaluation.
#     
#     Ground truth: relationships array with:
#     - drug_row, diagnosis_row
#     - relationship_type: TREATS, ADVERSE_EFFECT, CONTRAINDICATED, DISCONTINUED
#     - evidence_sentences, evidence_scope, confidence
#     
#     Metrics:
#     - Pair identification: precision, recall, f1
#     - Type classification: per-type f1, accuracy, Cohen's kappa
#     - Evidence quality: Jaccard on evidence_sentences
#     """
#     pass


# ============================================================================
# MIMIC-Flipped (DOC_TO_TABLE) Row-Sentence Evaluation
# ============================================================================
# Annotation format (Annotated_Test.json) - flat list:
#   [{"anchor_id": int, "primary_table_id": int, "medication_table_id": int,
#     "highlighted_cells_diagnosis":  [[sent_idx, row_idx], ...],
#     "highlighted_cells_medication": [[sent_idx, row_idx], ...]}]
#
# Score matrix orientation (DOC_TO_TABLE):
#   model(sent_tensor[1,S,d], table_tensor[1,R,d]) -> pair_scores[1,S,R]
#   pair_scores_np[sent_idx, row_idx] indexes directly into highlighted_cells_*
# ============================================================================

def load_mimic_flipped_annotations(annotation_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load mimic_flipped Annotated_Test.json.

    mimic_flipped uses the same annotation format as mimic: a dict keyed by
    admission_id with row_grounding.  admission_id is the stable join key
    present in both the TABLE_TO_DOC and DOC_TO_TABLE test examples.

    Returns:
        {admission_id: {"row_grounding": {"diagnosis": {...}, "medication": {...}},
                        "diagnosis_anchor_id": ..., "medication_anchor_id": ...}}
    """
    return load_mimic_annotations(annotation_file)


def load_mimic_flipped_test_data_and_annotations(
    test_file: str,
    annotation_file: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load mimic_flipped test examples and annotations."""
    try:
        from data import load_row_level_dataset
        test_examples  = load_row_level_dataset(test_file)
        annotations    = load_mimic_flipped_annotations(annotation_file)

        matched = sum(1 for ex in test_examples if ex.get("admission_id") in annotations)
        print(f"[INFO] mimic_flipped evaluation data loaded:")
        print(f"       Test examples : {len(test_examples)}")
        print(f"       Annotations   : {len(annotations)}")
        print(f"       Matched       : {matched}")
        return test_examples, annotations
    except Exception as e:
        print(f"[ERROR] Error loading mimic_flipped test data: {e}")
        return [], {}


def evaluate_mimic_flipped_row_grounding(
    model,
    examples: List[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    batch_size: int = 1,
    max_examples: Optional[int] = None,
    aggregation_method: str = None,
    test_cache=None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate Level A row grounding for mimic_flipped (DOC_TO_TABLE direction).

    For each example the anchor is the clinical note (doc) and the two candidates
    are the diagnosis and medication tables.  We compute a separate grounding score
    matrix for each table type and compare against highlighted_cells_*.

    Score matrix shape: pair_scores_np[sent_idx, row_idx]  (S x R)
    Ground truth pairs from highlighted_cells_*: [[sent_idx, row_idx], ...]
    """
    model.eval()
    device     = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    if aggregation_method is None:
        aggregation_method = (
            "top_k_pairs" if isinstance(model, BidirectionalTableTextModel)
            else "entropy_regularized"
        )

    diagnosis_metrics:  List[Dict] = []
    medication_metrics: List[Dict] = []
    examples_processed = 0
    examples_skipped_no_valid_pairs = 0
    total_oob_pairs = 0

    with torch.no_grad():
        for example in examples:
            if max_examples is not None and examples_processed >= max_examples:
                break

            anchor_id    = example.get("anchor_id")
            admission_id = example.get("admission_id")
            annotation   = annotations.get(admission_id)
            if annotation is None:
                continue

            # Extract (sent_idx, row_idx) pairs from row_grounding
            # extract_mimic_row_sentence_pairs returns (row_idx, sent_idx); swap for DOC_TO_TABLE
            row_grounding = annotation.get("row_grounding", {})
            _, diag_pairs_rs = extract_mimic_row_sentence_pairs(row_grounding, "diagnosis")
            _, med_pairs_rs  = extract_mimic_row_sentence_pairs(row_grounding, "medication")
            hc_diag = [(s, r) for r, s in diag_pairs_rs]  # (sent_idx, row_idx)
            hc_med  = [(s, r) for r, s in med_pairs_rs]

            # -- Note sentence embeddings (context side) ----------------------
            if test_cache is not None:
                sent_embeddings = test_cache.get_context_embeddings(anchor_id)
                if sent_embeddings is None:
                    continue
            else:
                sentences = _extract_sentences_robust(example.get("anchor_sentences", []))
                if not sentences:
                    continue
                sent_embeddings = model.encode_sentences(sentences, batch_size=batch_size)

            sent_tensor = sent_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
            num_sents   = sent_embeddings.shape[0]

            def _get_pair_scores(table_item, table_id):
                """Return pair_scores_np [num_sents, num_table_rows] or None."""
                if test_cache is not None:
                    tbl_emb = test_cache.get_table_embeddings(table_id)
                    if tbl_emb is None:
                        return None
                else:
                    rows = _extract_table_rows_for_model(table_item)
                    if not rows:
                        return None
                    tbl_emb = model.encode_sentences(rows, batch_size=batch_size)

                tbl_tensor = tbl_emb.unsqueeze(0).to(device=device, dtype=model_dtype)

                if isinstance(model, BidirectionalTableTextModel):
                    _, ps = model(sent_tensor, tbl_tensor, aggregation_method=aggregation_method)
                else:
                    ps = model.get_contextualized_pair_scores(sent_tensor, tbl_tensor)

                return safe_tensor_to_numpy(ps.squeeze(0).detach())  # [num_sents, num_table_rows]

            # -- Diagnosis ----------------------------------------------------
            diag_item = example.get("primary_positive", {})
            diag_id   = diag_item.get("id")
            if diag_id is not None and hc_diag:
                ps_np = _get_pair_scores(diag_item, diag_id)
                if ps_np is not None:
                    num_diag_rows = ps_np.shape[1]
                    pairs = [(int(s), int(r)) for s, r in hc_diag]
                    # Filter to in-bounds pairs
                    valid = [(s, r) for s, r in pairs if 0 <= s < num_sents and 0 <= r < num_diag_rows]
                    oob = len(pairs) - len(valid)
                    total_oob_pairs += oob
                    if not valid:
                        examples_skipped_no_valid_pairs += 1
                    else:
                        m = calculate_mimic_grounding_metrics(ps_np, valid, num_sents, num_diag_rows)
                        diagnosis_metrics.append(m)
                        if verbose and examples_processed < 3:
                            print(f"    example diag: sep={m['score_separation']:.3f} f1={m['f1']:.3f}")

            # -- Medication ---------------------------------------------------
            aps     = example.get("additional_positives", [])
            med_item = aps[0] if aps else {}
            med_id   = med_item.get("id")
            if med_id is not None and hc_med:
                ps_np = _get_pair_scores(med_item, med_id)
                if ps_np is not None:
                    num_med_rows = ps_np.shape[1]
                    pairs = [(int(s), int(r)) for s, r in hc_med]
                    # Filter to in-bounds pairs
                    valid = [(s, r) for s, r in pairs if 0 <= s < num_sents and 0 <= r < num_med_rows]
                    oob = len(pairs) - len(valid)
                    total_oob_pairs += oob
                    if not valid:
                        examples_skipped_no_valid_pairs += 1
                    else:
                        m = calculate_mimic_grounding_metrics(ps_np, valid, num_sents, num_med_rows)
                        medication_metrics.append(m)
                        if verbose and examples_processed < 3:
                            print(f"    example med : sep={m['score_separation']:.3f} f1={m['f1']:.3f}")

            examples_processed += 1

    def _agg(mlist):
        if not mlist:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "average_precision": 0.0}
        return {
            "precision":         float(np.mean([m["precision"]         for m in mlist])),
            "recall":            float(np.mean([m["recall"]            for m in mlist])),
            "f1":                float(np.mean([m["f1"]                for m in mlist])),
            "jaccard":           float(np.mean([m["jaccard"]           for m in mlist])),
            "average_precision": float(np.mean([m.get("average_precision", 0.0) for m in mlist])),
        }

    result = {
        "level_a_row_grounding": {
            "diagnosis":  _agg(diagnosis_metrics),
            "medication": _agg(medication_metrics),
        },
        "examples_evaluated":  examples_processed,
        "diagnosis_examples":  len(diagnosis_metrics),
        "medication_examples": len(medication_metrics),
    }
    print(f"[INFO] mimic_flipped Row Grounding: {examples_processed} examples")
    print(f"       Diagnosis  F1={result['level_a_row_grounding']['diagnosis']['f1']:.3f}  "
          f"AP={result['level_a_row_grounding']['diagnosis']['average_precision']:.3f}")
    print(f"       Medication F1={result['level_a_row_grounding']['medication']['f1']:.3f}  "
          f"AP={result['level_a_row_grounding']['medication']['average_precision']:.3f}")
    if examples_skipped_no_valid_pairs > 0:
        print(f"       [WARN]  Skipped {examples_skipped_no_valid_pairs} table evaluations (all GT pairs out-of-bounds)")
    if total_oob_pairs > 0:
        print(f"       [WARN]  Filtered {total_oob_pairs} out-of-bounds GT pairs across all examples")
    return result


def evaluate_mimic_flipped_with_model(
    model,
    test_examples: List[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    max_examples: Optional[int] = None,
    test_cache=None,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, float]:
    """Thin wrapper for Stage 1/2 calls from train.py (mirrors evaluate_mimic_with_model)."""
    metrics  = evaluate_mimic_flipped_row_grounding(
        model=model,
        examples=test_examples,
        annotations=annotations,
        max_examples=max_examples,
        test_cache=test_cache,
        verbose=verbose,
    )
    level_a   = metrics.get("level_a_row_grounding", {})
    diag_m    = level_a.get("diagnosis",  {})
    med_m     = level_a.get("medication", {})
    avg_ap    = (diag_m.get("average_precision", 0.0) + med_m.get("average_precision", 0.0)) / 2
    avg_f1    = (diag_m.get("f1", 0.0)               + med_m.get("f1", 0.0))               / 2
    return {
        "average_precision": avg_ap,
        "f1":                avg_f1,
        "examples_evaluated": metrics.get("examples_evaluated", 0),
    }


def quick_mimic_flipped_row_grounding_eval(
    model,
    test_examples: List[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    max_examples: Optional[int] = None,
    aggregation_method: str = None,
    test_cache=None,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Training-time quick evaluation wrapper for mimic_flipped."""
    return evaluate_mimic_flipped_row_grounding(
        model=model,
        examples=test_examples,
        annotations=annotations,
        batch_size=1,
        max_examples=max_examples,
        aggregation_method=aggregation_method,
        test_cache=test_cache,
        verbose=verbose,
    )


def evaluate_frozen_encoder_mimic_flipped(
    sentence_encoder,
    examples: List[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    batch_size: int = 1,
    max_examples: Optional[int] = None,
    test_cache=None,
    **kwargs,
) -> Dict[str, float]:
    """
    Stage 0 baseline for mimic_flipped using only the frozen sentence encoder.
    Wraps the encoder in a FrozenModelWrapper and delegates to
    evaluate_mimic_flipped_row_grounding.
    """
    class _FrozenWrapper:
        def __init__(self, encoder):
            self.sentence_encoder = encoder
            self._device = self._find_device()

        def _find_device(self):
            try:
                return self.sentence_encoder[0].auto_model.embeddings.word_embeddings.weight.device
            except Exception:
                return getattr(self.sentence_encoder, "device", torch.device("cpu"))

        def parameters(self):
            dummy = torch.tensor(0.0, device=self._device, requires_grad=False)
            return iter([dummy])

        def eval(self): pass

        def __call__(self, sent_tensor, table_tensor, aggregation_method=None):
            sents  = torch.nn.functional.normalize(sent_tensor.squeeze(0),  p=2, dim=1)
            tables = torch.nn.functional.normalize(table_tensor.squeeze(0), p=2, dim=1)
            sim    = torch.matmul(sents, tables.t())  # [num_sents, num_table_rows]
            return torch.tensor(0.0), sim

        def get_contextualized_pair_scores(self, sent_tensor, table_tensor):
            sents  = torch.nn.functional.normalize(sent_tensor.squeeze(0),  p=2, dim=1)
            tables = torch.nn.functional.normalize(table_tensor.squeeze(0), p=2, dim=1)
            return torch.matmul(sents, tables.t()).unsqueeze(0)

        def encode_sentences(self, texts, batch_size=16):
            return self.sentence_encoder.encode(
                texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True
            )

    wrapper = _FrozenWrapper(sentence_encoder)
    metrics = evaluate_mimic_flipped_row_grounding(
        model=wrapper,
        examples=examples,
        annotations=annotations,
        batch_size=batch_size,
        max_examples=max_examples,
        test_cache=test_cache,
    )
    level_a = metrics.get("level_a_row_grounding", {})
    diag_m  = level_a.get("diagnosis",  {})
    med_m   = level_a.get("medication", {})
    return {
        "average_precision": (diag_m.get("average_precision", 0.0) + med_m.get("average_precision", 0.0)) / 2,
        "f1":                (diag_m.get("f1", 0.0)               + med_m.get("f1", 0.0))               / 2,
        "examples_evaluated": metrics.get("examples_evaluated", 0),
    }

