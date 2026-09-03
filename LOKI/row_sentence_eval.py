"""
Row-Sentence Evaluation Module

This module provides row-sentence level evaluation capabilities for measuring
how well models trained on global table-paragraph alignment perform on the
fine-grained row-sentence alignment task.

Extracted and adapted from evaluate_protrix_row_sent.py for integration
into the training pipeline.
"""

import torch
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from pathlib import Path

from models import TableTextEmbeddingModel, BidirectionalTableTextModel
from data import IdBasedEmbeddingCache, _extract_sentences_robust, _extract_rows_robust

def safe_tensor_to_numpy(tensor):
    """Safely convert PyTorch tensor to numpy, handling BFloat16 conversion."""
    if hasattr(tensor, 'dtype') and tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.cpu().numpy()

def extract_model_config(model) -> Dict[str, Any]:
    """
    Extract model configuration from a trained model's attributes.
    
    Args:
        model: The model instance (TableTextEmbeddingModel or BidirectionalTableTextModel)
        
    Returns:
        Dictionary with model configuration parameters
    """
    config = {}
    
    # Common attributes for both model types
    common_attrs = [
        'embedding_dim', 'top_k', 'attention_type', 'sparse_top_k', 
        'window_size', 'threshold_base', 'init_method', 'init_method_params',
        # NEW: activation config
        'attention_activation', 'attention_alpha'
    ]
    
    # Bidirectional-specific attributes
    bidirectional_attrs = [
        'pair_score_method', 'share_weights', 'use_refinement', 
        'use_self_attention', 'self_attention_heads', 'self_attention_dropout'
    ]
    
    # LoRA attributes (check if they exist in the model or attention module)
    lora_attrs = [
        'use_cross_attention_lora', 'lora_rank', 'lora_alpha', 'lora_dropout'
    ]
    
    # Extract common attributes
    for attr in common_attrs:
        if hasattr(model, attr):
            config[attr] = getattr(model, attr)
    
    # Extract bidirectional-specific attributes
    if isinstance(model, BidirectionalTableTextModel):
        for attr in bidirectional_attrs:
            if hasattr(model, attr):
                config[attr] = getattr(model, attr)
        
        # Also check bidirectional_attention module for additional parameters
        if hasattr(model, 'bidirectional_attention'):
            attention_module = model.bidirectional_attention
            for attr in bidirectional_attrs + lora_attrs:
                if hasattr(attention_module, attr):
                    config[attr] = getattr(attention_module, attr)
    
    # Extract LoRA attributes (can be in main model or attention modules)
    for attr in lora_attrs:
        if hasattr(model, attr):
            config[attr] = getattr(model, attr)
    
    # Check cross_attention module for unidirectional models
    if hasattr(model, 'cross_attention'):
        attention_module = model.cross_attention
        for attr in common_attrs + lora_attrs:
            if hasattr(attention_module, attr) and attr not in config:
                config[attr] = getattr(attention_module, attr)

    # Try to capture norm_type if available on model or attention modules
    for comp in [model, getattr(model, 'bidirectional_attention', None), getattr(model, 'cross_attention', None)]:
        if comp is not None and hasattr(comp, 'norm_type') and 'norm_type' not in config:
            config['norm_type'] = getattr(comp, 'norm_type')
    
    return config


def validate_aggregation_method(model, aggregation_method: str) -> bool:
    """
    Validate that the aggregation method is compatible with the model type.
    
    Args:
        model: The model instance
        aggregation_method: The aggregation method to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Unidirectional model aggregation methods
    unidirectional_methods = {
        "mean", "top_k_sum", "top_k_mean", "weighted_top_k", 
        "max", "attention_weighted", "sparse_top_k", "entropy_regularized"
    }
    
    # Bidirectional model aggregation methods  
    bidirectional_methods = {
        "top_k_pairs", "max_pairs", "mean_pairs", "weighted_pairs", "sparse_pairs",
        # Some unidirectional methods also work with bidirectional
        "entropy_regularized"  
    }
    
    if isinstance(model, BidirectionalTableTextModel):
        return aggregation_method in bidirectional_methods
    else:
        return aggregation_method in unidirectional_methods


def get_annotation_id_candidates(example: Dict[str, Any]) -> List[str]:
    """Return candidate identifier values that may be used to match an example to annotations."""
    candidates: List[str] = []
    for field in ("anchor_id", "id", "admission_id"):
        value = example.get(field)
        if value is None:
            continue
        candidates.append(str(value))
    return candidates


def example_has_annotation_match(example: Dict[str, Any], annotation_keys: Any) -> bool:
    """Check whether an example can be matched to a set of annotation keys across supported formats."""
    if not annotation_keys:
        return False
    key_strings = {str(k) for k in annotation_keys}
    return any(candidate in key_strings for candidate in get_annotation_id_candidates(example))


def load_protrix_annotations(annotation_file: str) -> Dict[int, List[List[int]]]:
    """Load annotations mapping anchor_id to highlighted_cells.
    
    Supports list-based (Protrix) and dict-based (Totto) serialization.
    Note: If the file is in MIMIC format (dict-based without highlighted_cells), 
    returns empty dict and lets MIMIC evaluation handle it instead.
    """
    if not Path(annotation_file).exists():
        print(f"[WARN]  Row-sentence annotation file not found: {annotation_file}")
        return {}
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = {}
        if isinstance(data, list):
            for entry in data:
                anchor_id = entry.get("anchor_id")
                if anchor_id is None:
                    continue
                annotations[anchor_id] = entry.get("highlighted_cells", []) or []
        elif isinstance(data, dict):
            # Defensive fallback for TOTTO dict format, ignoring MIMIC format
            for key, value in data.items():
                try:
                    anchor_id = int(key)
                except Exception:
                    continue
                if isinstance(value, dict) and "highlighted_cells" in value:
                    annotations[anchor_id] = value.get("highlighted_cells", []) or []
                elif isinstance(value, list):
                    annotations[anchor_id] = value
        
        # If it was actually MIMIC, annotations will be empty, which is the correct inherited behavior
        if not annotations and isinstance(data, dict):
            # MIMIC format detected - skip loading, let MIMIC eval handle it
            return {}
        
        print(f"[OK] Loaded {len(annotations)} row-sentence annotations")
        return annotations
        
    except Exception as e:
        print(f"[ERROR] Error loading row-sentence annotations: {e}")
        return {}


def extract_row_sentence_pairs(highlighted_cells: List[List[int]]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Extract row indices and row-sentence pairs from highlighted_cells.
    
    Args:
        highlighted_cells: List of [row_idx, sent_idx] pairs (1-based rows, 0-based sentences)
    
    Returns:
        Tuple of (unique_row_indices, row_sentence_pairs)
    """
    if not highlighted_cells:
        return [], []
    
    # Extract unique row indices (convert to 0-based for internal processing)
    row_indices = list(set([cell[0] - 1 for cell in highlighted_cells]))  # Convert to 0-based
    
    # Extract row-sentence pairs (keep original format for analysis)
    pairs = [(cell[0] - 1, cell[1]) for cell in highlighted_cells]  # Convert row to 0-based
    
    return sorted(row_indices), pairs


def calculate_f1_for_pairs(pair_scores: np.ndarray, 
                                       row_sentence_pairs: List[Tuple[int, int]],
                                       num_rows: int, 
                                       num_sentences: int) -> float:
    """
    Calculate F1-based accuracy for row-sentence pairs (NO RANKING ASSUMPTION).
    
    Since all ground truth pairs are equally valid (no ranking), this treats it as
    a multi-label classification problem using optimal threshold to separate
    ground truth pairs from non-ground truth pairs.
    
    Args:
        pair_scores: NxM matrix of row-sentence similarities
        row_sentence_pairs: List of (row_idx, sent_idx) ground truth pairs
        num_rows: Number of rows
        num_sentences: Number of sentences
    
    Returns:
        F1 score using optimal threshold (better training feedback than strict accuracy)
    """
    if not row_sentence_pairs or pair_scores.size == 0:
        return 0.0
    
    # Create ground truth set
    gt_pairs = set(row_sentence_pairs)
    
    # Flatten scores for threshold fallback
    flat_scores = pair_scores.flatten()
    
    # Find optimal threshold using ground truth scores (with bound checking to prevent crashes)
    gt_scores = [pair_scores[i, j] for i, j in row_sentence_pairs if i < num_rows and j < num_sentences]
    non_gt_scores = [pair_scores[i, j] for i in range(num_rows) for j in range(num_sentences) 
                     if (i, j) not in gt_pairs]
    
    if gt_scores and non_gt_scores:
        # Use threshold between mean of GT scores and mean of non-GT scores
        threshold = (np.mean(gt_scores) + np.mean(non_gt_scores)) / 2.0
    else:
        # Fallback to median of all scores
        threshold = np.median(flat_scores)
    
    # Create predictions based on threshold
    predicted_pairs = set()
    for i in range(num_rows):
        for j in range(num_sentences):
            if pair_scores[i, j] >= threshold:
                predicted_pairs.add((i, j))
    
    # Calculate precision, recall, F1
    # This automatically penalizes for out-of-bounds GTs because they are in gt_pairs but not in predicted_pairs
    tp = len(gt_pairs & predicted_pairs)
    fp = len(predicted_pairs - gt_pairs)
    fn = len(gt_pairs - predicted_pairs)
    
    if tp + fp == 0:  # No predictions
        precision = 0.0
    else:
        precision = tp / (tp + fp)
        
    if tp + fn == 0:  # No ground truth
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        return 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score


def calculate_average_precision_for_pairs(pair_scores: np.ndarray, 
                                        row_sentence_pairs: List[Tuple[int, int]],
                                        num_rows: int, 
                                        num_sentences: int) -> float:
    """
    Calculate Average Precision for row-sentence pair prediction.
    
    This metric measures the quality of ranking - how well the model ranks
    ground truth pairs above non-ground truth pairs.
    
    Args:
        pair_scores: NxM matrix of row-sentence similarities
        row_sentence_pairs: List of (row_idx, sent_idx) ground truth pairs
        num_rows: Number of rows
        num_sentences: Number of sentences
    
    Returns:
        Average precision score
    """
    if not row_sentence_pairs or pair_scores.size == 0:
        return 0.0
    
    # Create ground truth set
    gt_pairs = set(row_sentence_pairs)
    
    # Flatten scores and create binary labels
    flat_scores = pair_scores.flatten()
    all_pairs = [(i, j) for i in range(num_rows) for j in range(num_sentences)]
    
    # Create binary labels (1 for ground truth pairs, 0 for others)
    y_true = np.array([1 if pair in gt_pairs else 0 for pair in all_pairs])
    
    # Handle edge case where all labels are 0 or all are 1
    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return 0.0
    
    try:
        # Calculate average precision
        avg_precision = average_precision_score(y_true, flat_scores)
        return avg_precision
    except Exception:
        return 0.0


def evaluate_row_sentence_metrics(model, examples: List[Dict[str, Any]], 
                                annotations: Dict[int, List[List[int]]],
                                batch_size: int = 1,
                                max_examples: Optional[int] = None,
                                aggregation_method: str = None,
                                model_config: Dict[str, Any] = None,
                                test_cache = None) -> Dict[str, float]:
    """
    Evaluate row-sentence metrics for a model (FIXED for better training feedback).
    
    IMPROVEMENTS:
    - FIXED: No ranking assumption among ground truth pairs (all GT pairs equally valid)
    - Overall accuracy now uses F1-based threshold approach for better training feedback
    - FIXED: Supports ALL aggregation methods that the model was trained with
    - FIXED: Dynamically extracts model configuration (attention_type, top_k, pair_score_method, etc.) - NO HARDCODING!
    - Uses model's learned attention weights instead of raw similarities
    - Includes training-friendly metrics (precision, recall, score separation, etc.)
    
    SUPPORTED AGGREGATION METHODS:
    
    Unidirectional Models (TableTextEmbeddingModel):
    - "mean": Mean of all row scores (problematic, not recommended)
    - "top_k_sum": Sum of top-k row scores (recommended)
    - "top_k_mean": Mean of top-k row scores
    - "weighted_top_k": Weighted combination of top-k scores
    - "max": Maximum row score only
    - "attention_weighted": Attention-weighted row scores
    - "sparse_top_k": Sparse top-k with zero masking
    - "entropy_regularized": Top-k with entropy bonus (default)
    
    Bidirectional Models (BidirectionalTableTextModel):
    - "top_k_pairs": Sum of top-k pair scores (default)
    - "max_pairs": Maximum pair score
    - "mean_pairs": Mean of all pair scores
    - "weighted_pairs": Attention-weighted pair scores
    - "sparse_pairs": Sparse top-k pairs
    - "entropy_regularized": Also supported for bidirectional
    
    Args:
        model: The model to evaluate (TableTextEmbeddingModel or BidirectionalTableTextModel)
        examples: List of test examples
        annotations: Dictionary mapping anchor_id to highlighted_cells
        batch_size: Batch size for evaluation
        max_examples: Maximum number of examples to evaluate (for speed)
        aggregation_method: Aggregation method to use (if None, uses model's default)
        model_config: Optional model configuration dict (if None, extracts from model attributes)
        test_cache: Optional test cache for efficient embedding lookup (avoids re-encoding)
    
    Returns:
        Dictionary with improved metrics (NO RANKING ASSUMPTION):
        - row_sent_f1: F1-based threshold accuracy
        - row_sent_avg_precision: Average precision (separates GT from non-GT pairs)
        - row_sent_precision: Precision using optimal threshold
        - row_sent_recall: Recall using optimal threshold
        - row_sent_f1_score: F1 score using optimal threshold
        - row_sent_gt_score_mean: Mean score of ground truth pairs
        - row_sent_non_gt_score_mean: Mean score of non-ground truth pairs
        - row_sent_score_separation: Difference between GT and non-GT means (key training signal)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # CRITICAL: Detect model dtype to ensure embeddings match model component dtypes
    # Get dtype from custom components which are guaranteed to be converted properly
    if hasattr(model, 'bidirectional_attention'):
        model_dtype = next(model.bidirectional_attention.parameters()).dtype
    elif hasattr(model, 'cross_attention'):
        model_dtype = next(model.cross_attention.parameters()).dtype
    else:
        model_dtype = next(model.parameters()).dtype
    
    # FIXED: Extract model configuration dynamically (no hardcoding!)
    if model_config is None:
        model_config = extract_model_config(model)
        # print(f"[INFO] Extracted model configuration: {model_config}")  # Commented out - too verbose for training
    else:
        # print(f"[INFO] Using provided model configuration: {model_config}")  # Commented out - too verbose for training
        pass
    
    # FIXED: Determine appropriate aggregation method based on model type and training
    if aggregation_method is None:
        if isinstance(model, BidirectionalTableTextModel):
            # Default for bidirectional models (pair-level aggregation)
            aggregation_method = "top_k_pairs"
        else:
            # Default for unidirectional models (row-level aggregation)
            aggregation_method = "entropy_regularized"
        # print(f"[INFO] Using default aggregation method: {aggregation_method}")  # Commented out - too verbose for training
        pass
    else:
        # Validate the specified aggregation method
        if not validate_aggregation_method(model, aggregation_method):
            model_type = "bidirectional" if isinstance(model, BidirectionalTableTextModel) else "unidirectional"
            print(f"[WARN]  Warning: {aggregation_method} may not be compatible with {model_type} model")
        # print(f"[INFO] Using specified aggregation method: {aggregation_method}")  # Commented out - too verbose for training
    
    overall_accuracies = []
    avg_precisions = []
    # UPDATED: Storage for training-friendly metrics (no ranking assumption)
    precisions = []
    recalls = []
    f1_scores = []
    gt_score_means = []
    non_gt_score_means = []
    score_separations = []
    examples_processed = 0
    
    with torch.no_grad():
        for example_idx, example in enumerate(examples):
            # Limit number of examples if specified
            if max_examples is not None and examples_processed >= max_examples:
                break
                
            anchor_id = example.get("anchor_id")
            
            if anchor_id is None or anchor_id not in annotations:
                continue
            
            highlighted_cells = annotations[anchor_id]
            
            # Skip examples without annotations
            if not highlighted_cells:
                continue
            
            # Extract tables/documents (unified through robust extraction)
            rows = _extract_rows_robust(example)
            
            if not rows:
                continue
            
            # Extract texts from primary_positive (target candidates)
            primary_positive = example.get("primary_positive", {})
            sentences = _extract_sentences_robust(primary_positive.get("sentences", []))
            
            # Fallback for Flipped schema: primary_positive contains rows (it's a table)
            if not sentences:
                sentences = _extract_rows_robust(primary_positive)
            
            if not sentences:
                continue
            
            # Extract row-sentence pairs from annotations
            _, row_sentence_pairs = extract_row_sentence_pairs(highlighted_cells)
            
            if not row_sentence_pairs:
                continue
            
            is_flipped = "anchor_sentences" in example
            
            try:
                # Get embeddings from cache or encode fresh
                if test_cache is not None:
                    primary_positive = example.get("primary_positive", {})
                    context_id = primary_positive.get("id")
                    
                    if not is_flipped:
                        # Standard TABLE_TO_DOC
                        row_embeddings = test_cache.get_table_embeddings(anchor_id)
                        sentence_embeddings = test_cache.get_context_embeddings(context_id) if context_id else None
                    else:
                        # Flipped DOC_TO_TABLE
                        # 'rows' variable holds Document sentences (anchor)
                        row_embeddings = test_cache.get_context_embeddings(anchor_id)
                        # 'sentences' variable holds Table rows (target)
                        sentence_embeddings = test_cache.get_table_embeddings(context_id) if context_id else None
                        
                    if row_embeddings is None:
                        # Fallback to fresh encoding if not in cache
                        row_embeddings = model.encode_sentences(rows, batch_size=batch_size)
                    
                    if sentence_embeddings is None:
                        # Fallback to fresh encoding if not in cache
                        sentence_embeddings = model.encode_sentences(sentences, batch_size=batch_size)
                else:
                    # No cache - encode fresh (original behavior)
                    row_embeddings = model.encode_sentences(rows, batch_size=batch_size)
                    sentence_embeddings = model.encode_sentences(sentences, batch_size=batch_size)
                
                # Add batch dimension and convert to model dtype
                row_tensor = row_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
                sentence_tensor = sentence_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
                
                if isinstance(model, BidirectionalTableTextModel):
                    # For bidirectional models, get pair scores directly
                    global_similarity, pair_scores = model(row_tensor, sentence_tensor, aggregation_method=aggregation_method)
                    # Use safe conversion to handle BFloat16
                    pair_scores_np = safe_tensor_to_numpy(pair_scores.squeeze(0).detach())
                else:
                    # For unidirectional models, use CONTEXTUALIZED embeddings
                    pair_scores = model.get_contextualized_pair_scores(row_tensor, sentence_tensor)
                    pair_scores_np = safe_tensor_to_numpy(pair_scores.squeeze(0).detach())
                
                # TRANSPOSE SCORES FOR FLIPPED SCHEMA
                # If flipped, pair_scores_np is [num_docs_sentences, num_table_rows]
                # We need [num_table_rows, num_docs_sentences] for metrics calculation
                if is_flipped:
                    pair_scores_np = pair_scores_np.T
                
                num_rows, num_sentences = pair_scores_np.shape
                
                # Calculate metrics
                overall_acc = calculate_f1_for_pairs(
                    pair_scores_np, row_sentence_pairs, num_rows, num_sentences
                )
                avg_prec = calculate_average_precision_for_pairs(
                    pair_scores_np, row_sentence_pairs, num_rows, num_sentences
                )
                
                # ADDED: Calculate training-friendly metrics for better feedback
                training_metrics = calculate_training_friendly_metrics(
                    pair_scores_np, row_sentence_pairs, num_rows, num_sentences
                )
                
                overall_accuracies.append(overall_acc)
                avg_precisions.append(avg_prec)
                
                # UPDATED: Store training-friendly metrics (no ranking assumption)
                precisions.append(training_metrics['precision'])
                recalls.append(training_metrics['recall'])
                f1_scores.append(training_metrics['f1_score'])
                gt_score_means.append(training_metrics['gt_score_mean'])
                non_gt_score_means.append(training_metrics['non_gt_score_mean'])
                score_separations.append(training_metrics['score_separation'])
                
                examples_processed += 1
                
            except Exception as e:
                # Skip examples that cause errors, but log for debugging
                print(f"[WARN]  Row-sentence eval error for example {example_idx} (anchor_id={anchor_id}): {str(e)[:100]}...")
                continue
    
    # Compute aggregate metrics
    if overall_accuracies:
        mean_f1 = np.mean(overall_accuracies)  # Now F1-based
        mean_avg_precision = np.mean(avg_precisions)
        
        # UPDATED: Aggregate training-friendly metrics (no ranking assumption)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)
        mean_gt_score_mean = np.mean(gt_score_means)
        mean_non_gt_score_mean = np.mean(non_gt_score_means)
        mean_score_separation = np.mean(score_separations)
    else:
        mean_f1 = 0.0
        mean_avg_precision = 0.0
        mean_precision = 0.0
        mean_recall = 0.0
        mean_f1_score = 0.0
        mean_gt_score_mean = 0.0
        mean_non_gt_score_mean = 0.0
        mean_score_separation = 0.0
    
    # Add debugging info about processing results
    print(f"[INFO] Row-sentence evaluation completed: {examples_processed}/{len(examples)} examples processed")
    if examples_processed == 0:
        print("[WARN]  WARNING: No examples were successfully processed in row-sentence evaluation!")
    
    return {
        # Original metrics (FIXED: no ranking assumption)
        'row_sent_f1': mean_f1,  # Now F1-based threshold accuracy
        'row_sent_avg_precision': mean_avg_precision,  # Still valid (separates GT from non-GT)
        
        # UPDATED: Training-friendly metrics for better monitoring (no ranking assumption)
        'row_sent_precision': mean_precision,
        'row_sent_recall': mean_recall,
        'row_sent_f1_score': mean_f1_score,
        'row_sent_gt_score_mean': mean_gt_score_mean,
        'row_sent_non_gt_score_mean': mean_non_gt_score_mean,
        'row_sent_score_separation': mean_score_separation,  # Key metric: higher = better learning
        
        'examples_evaluated': examples_processed
    }


def load_test_data_and_annotations(test_file: str, annotation_file: str) -> Tuple[List[Dict[str, Any]], Dict[int, List[List[int]]]]:
    """
    Load test data and annotations for row-sentence evaluation.
    
    Args:
        test_file: Path to test dataset file
        annotation_file: Path to annotation file
    
    Returns:
        Tuple of (test_examples, annotations)
    """
    try:
        from data import load_row_level_dataset
        test_examples = load_row_level_dataset(test_file)
        annotations = load_protrix_annotations(annotation_file)
        
        # Keep only examples that can be matched to annotations across supported identifier fields.
        if annotations:
            matched_examples = [
                ex for ex in test_examples
                if example_has_annotation_match(ex, annotations.keys())
            ]
            if len(matched_examples) < len(test_examples):
                print(f"[INFO] Filtered row-sentence test set to {len(matched_examples)}/{len(test_examples)} examples with annotation matches")
            test_examples = matched_examples
        
        # Check consistency using the same compatibility logic
        annotation_ids = set(annotations.keys())
        annotation_id_strings = {str(k) for k in annotation_ids}
        examples_with_annotations = sum(
            1
            for ex in test_examples
            if any(candidate in annotation_id_strings for candidate in get_annotation_id_candidates(ex))
        )
        
        print(f"[INFO] Row-sentence evaluation data:")
        print(f"   Test examples: {len(test_examples)}")
        print(f"   Annotations: {len(annotations)}")
        print(f"   Examples with annotations: {examples_with_annotations}")
        
        return test_examples, annotations
        
    except Exception as e:
        print(f"[ERROR] Error loading test data or annotations: {e}")
        return [], {}


def calculate_training_friendly_metrics(pair_scores: np.ndarray, 
                                       row_sentence_pairs: List[Tuple[int, int]],
                                       num_rows: int, 
                                       num_sentences: int) -> Dict[str, float]:
    """
    Calculate training-friendly metrics (NO RANKING ASSUMPTION).
    
    Since all ground truth pairs are equally valid, focuses on threshold-based
    and classification metrics rather than ranking metrics.
    """
    if not row_sentence_pairs or pair_scores.size == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'gt_score_mean': 0.0,
            'non_gt_score_mean': 0.0,
            'score_separation': 0.0
        }
    
    # Create ground truth set
    gt_pairs = set(row_sentence_pairs)
    
    # Get scores for ground truth and non-ground truth pairs (with bound checking)
    gt_scores = [pair_scores[i, j] for i, j in row_sentence_pairs if i < num_rows and j < num_sentences]
    non_gt_scores = [pair_scores[i, j] for i in range(num_rows) for j in range(num_sentences) 
                     if (i, j) not in gt_pairs]
    
    # Calculate score statistics
    gt_score_mean = np.mean(gt_scores) if gt_scores else 0.0
    non_gt_score_mean = np.mean(non_gt_scores) if non_gt_scores else 0.0
    score_separation = gt_score_mean - non_gt_score_mean  # Higher is better
    
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
    
    # Calculate precision, recall, F1
    # This automatically penalizes for out-of-bounds GTs
    tp = len(gt_pairs & predicted_pairs)
    fp = len(predicted_pairs - gt_pairs)
    fn = len(gt_pairs - predicted_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'gt_score_mean': gt_score_mean,
        'non_gt_score_mean': non_gt_score_mean,
        'score_separation': score_separation  # Key metric: how well model separates GT from non-GT
    }


def quick_row_sentence_eval(model, test_examples: List[Dict[str, Any]], 
                          annotations: Dict[int, List[List[int]]],
                          max_examples: Optional[int] = None,
                          aggregation_method: str = None,
                          model_config: Dict[str, Any] = None,
                          test_cache = None) -> Dict[str, float]:
    """
    Quick row-sentence evaluation for use during training.
    
    Evaluates on all or a subset of examples during training epochs.
    
    Args:
        model: The model to evaluate
        test_examples: List of test examples
        annotations: Annotation dictionary
        max_examples: Maximum number of examples to evaluate (None = all examples)
        aggregation_method: Aggregation method to use (if None, uses model's default)
        model_config: Optional model configuration dict (if None, extracts from model attributes)
        test_cache: Optional test cache for efficient embedding lookup (avoids re-encoding)
    
    Returns:
        Dictionary with row-sentence metrics
    """
    return evaluate_row_sentence_metrics(
        model=model,
        examples=test_examples,
        annotations=annotations,
        batch_size=1,
        max_examples=max_examples,
        aggregation_method=aggregation_method,
        model_config=model_config,
        test_cache=test_cache
    )


def evaluate_frozen_encoder_only(
    sentence_encoder,
    examples: List[Dict[str, Any]],
    annotations: Dict[int, List[List[int]]],
    batch_size: int = 1,
    max_examples: Optional[int] = None,
    test_cache=None,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluates row-sentence alignment using ONLY the frozen sentence encoder (Stage 0 Baseline).
    This function wraps the sentence_encoder in a dummy model interface and delegates
    to the standard `evaluate_row_sentence_metrics` loop.
    """
    class FrozenModelWrapper:
        def __init__(self, encoder):
            self.sentence_encoder = encoder
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
            # Return a single dummy parameter on the correct device
            # This avoids generator lifecycle issues with next()
            dummy = torch.tensor(0.0, device=self._device, requires_grad=False)
            return iter([dummy])
            
        def eval(self):
            pass
            
        def __call__(self, row_tensor, sentence_tensor, aggregation_method=None):
            rows = row_tensor.squeeze(0)  # [num_rows, dim]
            sents = sentence_tensor.squeeze(0)  # [num_sents, dim]
            
            # Normalize
            rows = torch.nn.functional.normalize(rows, p=2, dim=1)
            sents = torch.nn.functional.normalize(sents, p=2, dim=1)
            
            # Cosine sim: [num_rows, num_sents]
            sim_matrix = torch.matmul(rows, sents.t())
            
            return torch.tensor(0.0), sim_matrix.unsqueeze(0)
            
        def encode_rows(self, rows, batch_size=16):
            return self.sentence_encoder.encode(rows, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
            
        def encode_sentences(self, sents, batch_size=16):
            return self.sentence_encoder.encode(sents, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
        
        def get_contextualized_pair_scores(self, row_tensor, sentence_tensor):
            """For frozen encoder, return raw cosine similarity."""
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
    
    return evaluate_row_sentence_metrics(
        model=wrapper,
        examples=examples,
        annotations=annotations,
        batch_size=batch_size,
        max_examples=max_examples,
        test_cache=test_cache,
        aggregation_method="frozen"
    )
