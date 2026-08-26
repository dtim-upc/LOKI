import os
import gc
import datetime
import json
import time
import random
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm.auto import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
import numpy as np

# Optional wandb import for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from models import TableTextEmbeddingModel, BidirectionalTableTextModel
from losses import IdBasedCachedTripletLoss
from data import IdBasedEmbeddingCache
from evaluate import evaluate_with_cache, evaluate_with_id_cache
from utils import GPUMemoryManager
from training_curves import TrainingCurves
from row_sentence_eval import load_test_data_and_annotations, quick_row_sentence_eval
from evaluate_mimic_row_sent import (
    load_mimic_test_data_and_annotations, quick_mimic_row_grounding_eval,
    load_mimic_flipped_test_data_and_annotations,
    evaluate_mimic_flipped_with_model, evaluate_frozen_encoder_mimic_flipped,
    quick_mimic_flipped_row_grounding_eval,
)
# Old totto module is unified into row_sentence_eval

# Export only the ID-based training function
__all__ = ['train_with_id_based_triplets']

def ensure_requires_grad(tensor_or_list, device=None):
    """
    Ensure that a tensor or list of tensors requires gradients.
    If not, create a clone that requires gradients.
    
    Args:
        tensor_or_list: A tensor or list of tensors
        device: Optional device to place the tensor on
        
    Returns:
        The same tensor or list of tensors, but ensuring all require gradients
    """
    if isinstance(tensor_or_list, list):
        return [ensure_requires_grad(t, device) for t in tensor_or_list]
    elif hasattr(tensor_or_list, 'requires_grad'):
        if tensor_or_list.requires_grad:
            return tensor_or_list
        else:
            return tensor_or_list.detach().clone().requires_grad_(True)
    else:
        # If it's not a tensor or has no requires_grad attribute, return as is
        return tensor_or_list


def create_frozen_encoder_baseline(sentence_encoder):
    """
    Create a simple function that mimics model evaluation using only the frozen encoder
    with basic cosine similarity. This is the TRUE baseline with no learned components.
    """
    return sentence_encoder


def _aggregate_similarity_scores(similarity_matrix: torch.Tensor, 
                                  aggregation_method: str, 
                                  top_k: int) -> float:
    """
    Aggregate a similarity matrix into a single score using the specified method.
    This matches the aggregation logic used in BidirectionalTableTextModel for fair comparison.
    
    Args:
        similarity_matrix: [num_rows, num_sentences] cosine similarity matrix
        aggregation_method: Method to use for aggregation
        top_k: Number of top pairs for top_k methods
        
    Returns:
        Aggregated similarity score (float)
    """
    num_rows, num_sentences = similarity_matrix.shape
    flat_scores = similarity_matrix.view(-1)  # Flatten to [N*M]
    
    if aggregation_method == "mean_pairs" or aggregation_method == "mean":
        # Mean of all pair scores
        return flat_scores.mean().item()
    
    elif aggregation_method == "top_k_pairs" or aggregation_method == "top_k_sum":
        # Sum of top-k pair scores (matches BidirectionalTableTextModel)
        k = min(top_k, num_rows * num_sentences)
        top_k_scores, _ = torch.topk(flat_scores, k=k)
        return top_k_scores.sum().item()
    
    elif aggregation_method == "max_pairs" or aggregation_method == "max":
        # Maximum pair score
        return flat_scores.max().item()
    
    elif aggregation_method == "weighted_pairs" or aggregation_method == "weighted_top_k":
        # Weighted by softmax attention (approximate)
        weights = torch.nn.functional.softmax(flat_scores, dim=0)
        return (flat_scores * weights).sum().item()
    
    elif aggregation_method == "sparse_pairs" or aggregation_method == "sparse_top_k":
        # Top-k with mean (sparse attention style)
        k = min(top_k, num_rows * num_sentences)
        top_k_scores, _ = torch.topk(flat_scores, k=k)
        return top_k_scores.mean().item()
    
    elif aggregation_method == "entropy_regularized":
        # Top-k mean with entropy weighting (simplified version)
        k = min(top_k, num_rows * num_sentences)
        top_k_scores, _ = torch.topk(flat_scores, k=k)
        return top_k_scores.mean().item()
    
    elif aggregation_method == "top_k_mean":
        # Mean of top-k scores
        k = min(top_k, num_rows * num_sentences)
        top_k_scores, _ = torch.topk(flat_scores, k=k)
        return top_k_scores.mean().item()
    
    else:
        # Fallback to mean for unknown methods
        print(f"⚠️  Unknown aggregation method '{aggregation_method}', falling back to mean")
        return flat_scores.mean().item()


def evaluate_frozen_encoder_baseline(sentence_encoder, eval_examples, batch_size=16, 
                                      aggregation_method="top_k_pairs", top_k=5,
                                      eval_cache: Optional['IdBasedEmbeddingCache'] = None,
                                      task_direction: str = "TABLE_TO_DOC",
                                      native_direction: str = "TABLE_TO_DOC"):
    """
    Evaluate using only the frozen sentence encoder with basic cosine similarity.
    This is the true baseline - no cross-attention, no learned components.
    
    PERFORMANCE NOTE: This function can be VERY slow without a cache because it 
    encodes every row, positive, and negative from scratch for each example.
    Pass an eval_cache to use pre-computed embeddings for ~100x speedup.
    
    Args:
        sentence_encoder: The sentence encoder to use
        eval_examples: List of evaluation examples
        batch_size: Batch size for encoding
        aggregation_method: Aggregation method to use (matches Stage 2 for fair comparison)
            - "mean_pairs": Mean of all pair scores
            - "top_k_pairs": Sum of top-k pair scores (default)
            - "max_pairs": Maximum pair score
        top_k: Number of top pairs for top_k methods
        eval_cache: Optional pre-built IdBasedEmbeddingCache for fast evaluation.
                   If provided, uses cached embeddings instead of encoding on-the-fly.
    """
    import data
    from sentence_transformers import util
    
    use_cache = eval_cache is not None
    if use_cache:
        print(f"🔥 Evaluating with FROZEN ENCODER ONLY (using CACHE, aggregation={aggregation_method})")
    else:
        print(f"🔥 Evaluating with FROZEN ENCODER ONLY (no cache - may be slow, aggregation={aggregation_method})")
    
    correct_predictions = 0
    total_comparisons = 0
    
    # Helper to extract valid rows from a list (defined once outside loop)
    def extract_valid_rows(row_list):
        valid_rows = []
        for row in row_list:
            if isinstance(row, dict):
                formatted_text = row.get("formatted", "")
                if formatted_text:
                    valid_rows.append(formatted_text)
            elif isinstance(row, str) and row:
                valid_rows.append(row)
        return valid_rows
    
    # Helper to robustly extract sentences from MIMIC format (dict or list)
    def extract_sentences_from_mimic_format(sentences_data):
        extracted_texts = []
        if isinstance(sentences_data, dict):
            try:
                sorted_keys = sorted(sentences_data.keys(), key=lambda k: int(k))
                for k in sorted_keys:
                    item = sentences_data[k]
                    if isinstance(item, dict):
                        extracted_texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        extracted_texts.append(item)
            except ValueError:
                for item in sentences_data.values():
                    if isinstance(item, dict):
                        extracted_texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        extracted_texts.append(item)
        elif isinstance(sentences_data, list):
            for item in sentences_data:
                if isinstance(item, dict):
                    extracted_texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    extracted_texts.append(item)
        return [t for t in extracted_texts if t]
    
    with torch.no_grad():
        for example in tqdm(eval_examples, desc="Evaluating frozen encoder baseline"):
            anchor_id = example.get("anchor_id")
            if anchor_id is None:
                continue
                # 1. Get query embeddings - from cache or encode on-the-fly
            if use_cache:
                if native_direction == "TABLE_TO_DOC":
                    query_embeddings = eval_cache.get_table_embeddings(anchor_id)
                else:
                    query_embeddings = eval_cache.get_context_embeddings(anchor_id)
                
                if query_embeddings is None:
                    continue
                query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
            else:
                # Direct extraction for query
                if native_direction == "TABLE_TO_DOC":
                    query_texts = data._extract_rows_robust(example)
                else: # DOC_TO_TABLE (native)
                    query_texts = data._extract_sentences_robust(example.get("anchor_sentences", []))
                
                if not query_texts:
                    continue
                query_embeddings = sentence_encoder.encode(query_texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
            
            # Helper for candidate extraction
            def get_candidate_embeddings(candidate):
                cid = candidate.get("id")
                if not cid: return None
                
                if use_cache:
                    if native_direction == "TABLE_TO_DOC":
                        # Candidates are Docs
                        ce = eval_cache.get_context_embeddings(cid)
                    else:
                        # Candidates are Tables
                        ce = eval_cache.get_table_embeddings(cid)
                    if ce is not None:
                        return torch.nn.functional.normalize(ce, p=2, dim=-1)
                    return None
                else:
                    if native_direction == "TABLE_TO_DOC":
                        texts = data._extract_sentences_robust(candidate.get("sentences", []))
                    else:
                        texts = data._extract_rows_robust(candidate)
                    if not texts: return None
                    return sentence_encoder.encode(texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
            
            # Collect all positive contexts
            all_positives = []
            primary_positive = example.get("primary_positive", {})
            if primary_positive.get("id") is not None:
                all_positives.append(primary_positive)
            
            additional_positives = example.get("additional_positives", [])
            if additional_positives:
                for add_pos in additional_positives:
                    if add_pos.get("id") is not None:
                        all_positives.append(add_pos)
            
            if not all_positives:
                continue

            # Process all positive contexts against all negatives
            for positive in all_positives:
                positive_embeddings = get_candidate_embeddings(positive)
                if positive_embeddings is None:
                    continue
                
                # Compute cosine similarity matrix [num_query, num_candidate]
                similarity_matrix = util.cos_sim(query_embeddings, positive_embeddings)
                
                # Apply aggregation method
                positive_similarity = _aggregate_similarity_scores(
                    similarity_matrix, aggregation_method, top_k
                )
                
                # Process negatives
                for negative in example["negatives"]:
                    negative_embeddings = get_candidate_embeddings(negative)
                    if negative_embeddings is None:
                        continue
                    
                    # Compute cosine similarity matrix for negative
                    similarity_matrix = util.cos_sim(query_embeddings, negative_embeddings)
                    
                    # Apply same aggregation method
                    negative_similarity = _aggregate_similarity_scores(
                        similarity_matrix, aggregation_method, top_k
                    )
                    
                    # Simple comparison: positive should be more similar than negative
                    total_comparisons += 1
                    if positive_similarity > negative_similarity:
                        correct_predictions += 1
    
    if total_comparisons == 0:
        accuracy = 0.0
    else:
        accuracy = correct_predictions / total_comparisons
    
    return {'accuracy': accuracy, 'total_comparisons': total_comparisons}


def _extract_sentence_texts_for_attention_diagnostic(primary_positive: Dict[str, Any]) -> List[str]:
    raw_sentences = primary_positive.get("sentences", [])
    sentences: List[str] = []
    if isinstance(raw_sentences, dict):
        try:
            items = [raw_sentences[k] for k in sorted(raw_sentences.keys(), key=lambda key: int(key))]
        except (TypeError, ValueError):
            items = list(raw_sentences.values())
    else:
        items = raw_sentences if isinstance(raw_sentences, list) else []

    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "")
            if text:
                sentences.append(text)
        elif isinstance(item, str) and item:
            sentences.append(item)
    return sentences


def _compute_attention_collapse_diagnostics(
    model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel],
    examples: List[Dict[str, Any]],
    id_cache: Optional[IdBasedEmbeddingCache] = None,
    max_examples: int = 3,
    batch_size: int = 32,
    aggregation_method: str = "top_k_pairs",
) -> Dict[str, float]:
    """Lightweight diagnostic for real bidirectional sparse attention patterns."""
    if not isinstance(model, BidirectionalTableTextModel) or max_examples <= 0:
        return {}

    import data

    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    summaries: List[Dict[str, float]] = []

    with torch.no_grad():
        for example in examples:
            if len(summaries) >= max_examples:
                break

            rows = data._extract_rows_robust(example)
            primary_positive = example.get("primary_positive", {})
            sentences = _extract_sentence_texts_for_attention_diagnostic(primary_positive)
            if not rows or not sentences:
                continue

            anchor_id = example.get("anchor_id")
            context_id = primary_positive.get("id")

            row_embeddings = None
            sentence_embeddings = None
            if id_cache is not None:
                if anchor_id is not None:
                    row_embeddings = id_cache.get_table_embeddings(anchor_id)
                if context_id is not None:
                    sentence_embeddings = id_cache.get_context_embeddings(context_id)

            if row_embeddings is None:
                row_embeddings = model.encode_sentences(rows, batch_size=batch_size)
            if sentence_embeddings is None:
                sentence_embeddings = model.encode_sentences(sentences, batch_size=batch_size)

            row_tensor = row_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
            sentence_tensor = sentence_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
            try:
                _scores, _pair_scores, forward_attn, reverse_attn = model(
                    row_tensor,
                    sentence_tensor,
                    aggregation_method=aggregation_method,
                    return_attention_weights=True,
                )
            except Exception as exc:
                print(f"⚠️  Attention diagnostic skipped one example: {str(exc)[:120]}")
                continue

            def summarize(attn: torch.Tensor) -> Tuple[int, int, float, float, float]:
                attn_2d = attn.squeeze(0).float()
                num_queries, num_keys = attn_2d.shape
                if num_queries == 0 or num_keys == 0:
                    return 0, 0, 0.0, 0.0, 0.0
                k = min(int(getattr(model, "top_k", 5)), num_keys)
                topk = torch.topk(attn_2d, k=max(k, 1), dim=-1).indices.cpu().tolist()
                unique_patterns = len({tuple(row) for row in topk})
                col_mass = attn_2d.sum(dim=0)
                total_mass = col_mass.sum().clamp_min(1e-8)
                max_hub_fraction = float((col_mass.max() / total_mass).item())
                row_variance = float(attn_2d.var(dim=0, unbiased=False).mean().item()) if num_queries > 1 else 0.0
                entropy = float((-(attn_2d * torch.log(attn_2d.clamp_min(1e-10))).sum(dim=-1)).mean().item())
                return unique_patterns, num_queries, max_hub_fraction, row_variance, entropy

            f_unique, f_queries, f_hub, f_var, f_entropy = summarize(forward_attn)
            r_unique, r_queries, r_hub, r_var, r_entropy = summarize(reverse_attn)
            summaries.append({
                "forward_unique": f_unique,
                "forward_queries": f_queries,
                "forward_unique_ratio": f_unique / max(f_queries, 1),
                "forward_hub_fraction": f_hub,
                "forward_row_variance": f_var,
                "forward_entropy": f_entropy,
                "reverse_unique": r_unique,
                "reverse_queries": r_queries,
                "reverse_unique_ratio": r_unique / max(r_queries, 1),
                "reverse_hub_fraction": r_hub,
                "reverse_row_variance": r_var,
                "reverse_entropy": r_entropy,
            })

    if not summaries:
        return {}

    metrics = {
        key: float(np.mean([summary[key] for summary in summaries]))
        for key in summaries[0]
    }
    metrics["examples_evaluated"] = float(len(summaries))
    print(
        "🔎 Attention diagnostics: "
        f"forward unique_topk≈{metrics['forward_unique']:.1f}/{metrics['forward_queries']:.1f} "
        f"(ratio={metrics['forward_unique_ratio']:.3f}), "
        f"hub_fraction={metrics['forward_hub_fraction']:.3f}, "
        f"row_var={metrics['forward_row_variance']:.6f}, "
        f"entropy={metrics['forward_entropy']:.3f}"
    )
    print(
        "   Reverse attention: "
        f"unique_topk≈{metrics['reverse_unique']:.1f}/{metrics['reverse_queries']:.1f} "
        f"(ratio={metrics['reverse_unique_ratio']:.3f}), "
        f"hub_fraction={metrics['reverse_hub_fraction']:.3f}, "
        f"row_var={metrics['reverse_row_variance']:.6f}, "
        f"entropy={metrics['reverse_entropy']:.3f}"
    )
    return metrics


def train_with_id_based_triplets(model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel],
                    train_examples: List[Dict[str, Any]],
                    eval_examples: List[Dict[str, Any]],
                    output_path: Path,
                    run_name: str,
                    learning_rate: float = 5e-5,
                    encoder_learning_rate: Optional[float] = None,
                    epochs: int = 3,
                    train_batch_size: int = 16,
                    eval_batch_size: int = 16,
                    weight_decay: float = 0.01,
                    warmup_ratio: float = 0.1,
                    lr_scheduler_type: str = "cosine",
                    min_lr_ratio: float = 0.0,
                    lr_num_cycles: int = 3,
                    max_grad_norm: float = 2.0,
                    gradient_accumulation_steps: int = 1,
                    enable_checkpointing: bool = True,
                    use_cache: bool = True,
                    loss_type: str = "enhanced_triplet",
                    aggregation_method: str = "entropy_regularized",
                    top_k: int = 5,  # Number of top pairs for aggregation methods
                    triplet_weight: float = 0.7,  # NEW: Triplet loss weight
                    attention_loss_weight: float = 0.2,
                    diversity_weight: float = 0.1,
                    direct_attention_loss_weight: float = 0.1,
                    direct_attention_diversity_weight: float = 1.0,
                    direct_attention_hub_weight: float = 0.5,
                    direct_attention_entropy_weight: float = 0.1,
                    direct_attention_entropy_floor_ratio: float = 0.5,
                    forward_attention_loss_weight: float = 0.0,
                    pair_mil_loss_weight: float = 0.0,
                    pair_mil_positive_margin: float = 0.2,
                    pair_mil_negative_margin: float = 0.05,
                    pair_mil_sparsity_weight: float = 0.0,
                    pair_mil_hub_weight: float = 0.0,
                    margin: float = 0.5,
                    margin_end: Optional[float] = None,
                    margin_schedule: str = "linear",  # "none" or "linear"
                    scale: float = 10.0,
                    pair_loss_weight: float = 0.1,
                    pair_margin: float = 0.1,
                    pair_score_method: str = "cosine",
                    # NEW: Attention distillation parameters
                    use_attention_distillation: bool = False,
                    distillation_weight: float = 0.2,
                    teacher_temperature: float = 0.1,
                    student_temperature: float = 0.1,
                    distillation_loss_type: str = "kl_div",
                    teacher_hub_centering: bool = True,
                    share_attention_weights: bool = False,
                    extract_join_paths: bool = False,
                    join_path_threshold: float = 0.1,
                    # NEW: SIGReg and Sinkhorn structural regularization
                    sigreg_weight: float = 0.0,
                    sigreg_target_std: float = 1.0,
                    sigreg_num_proj: int = 1024,
                    sigreg_knots: int = 17,
                    sinkhorn_weight: float = 0.0,
                    mix_examples: bool = True,
                    triplet_strategy: str = "limited",
                    max_triplets_per_example: int = 10,
                    # Hard negative mining
                    use_hard_negative_mining: bool = True,
                    hard_negative_topk: int = 4,
                    # Training curves parameters
                    enable_training_curves: bool = True,
                    track_batch_losses: bool = True,
                    track_val_loss: bool = False,
                    auto_plot_curves: bool = True,
                    # Row-sentence evaluation parameters
                    enable_row_sent_eval: bool = False,
                    row_sent_test_file: str = "protrix_data/test_row_level.json",
                    row_sent_annotation_file: str = "protrix_data/Annotated_Test.json",
                    dataset_format: str = "other",
                    row_sent_max_examples: Optional[int] = None,
                    enable_attention_diagnostics: bool = True,
                    attention_diagnostic_examples: int = 3,
                    save_best_by_test_metrics: bool = True,  # NEW: Save best model by test metrics
                    # Early stopping
                    early_stopping_patience: int = 2,
                    early_stopping_min_epochs: int = 3,
                    # Initialization parameters (NEW - for Stage 1 evaluation)
                    init_method: str = "xavier_uniform",
                    init_method_params: dict = None,
                    # NEW: Training stage control
                    start_training_from_stage: int = 1,
                    encoder_only_training: bool = False,
                    stage0_cache_mode: str = "auto",
                    # Encoder tuning mode
                    encoder_tuning_mode: str = "full",
                    gradual_unfreeze_initial_layers: int = 1,
                    gradual_unfreeze_every: int = 1,
                    gradual_unfreeze_max_layers: int = 0,
                    gradual_unfreeze_include_pooler: bool = True,
                    # Visualization control
                    skip_four_stage_viz: bool = False,
                    # Verbosity control
                    verbose: bool = True,
                    # Weights & Biases logging
                    use_wandb: bool = False,
                    # Task-Aware Dataset flags
                    task_direction: str = "DOC_TO_TABLE",
                    native_direction: str = "DOC_TO_TABLE",
                    # Encoding batch size for cache building
                    encoding_batch_size: int = 256) -> Union[TableTextEmbeddingModel, BidirectionalTableTextModel]:
    """
    Train the model using ID-based triplets with optional caching.
    
    Args:
        model: The TableTextEmbeddingModel or BidirectionalTableTextModel to train
        train_examples: List of training examples
        eval_examples: List of evaluation examples
        output_path: Path to save model checkpoints
        run_name: Name for this training run
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        weight_decay: Weight decay for AdamW optimizer
        warmup_ratio: Ratio of warmup steps
        max_grad_norm: Maximum gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        enable_checkpointing: Whether to enable gradient checkpointing
        use_cache: Whether to use embedding cache
        loss_type: Type of loss function ("enhanced_triplet", "id_cached_triplet", or "bidirectional_triplet")
        aggregation_method: Method for aggregating row scores (e.g., "top_k_sum", "max", "mean") or pair scores for bidirectional
        attention_loss_weight: Weight for attention regularization loss
        margin: Margin for triplet loss (distance between positive and negative similarities)
        scale: Scale factor for triplet loss similarities (amplifies differences between positive/negative scores)
        pair_loss_weight: Weight for pair-wise contrastive loss (bidirectional only)
        pair_margin: Margin for pair-wise contrastive loss (how much positive pairs should exceed negative pairs)
        pair_score_method: Method for computing pair scores (e.g., "cosine", "dot")
        share_attention_weights: Whether to share attention weights between the two models
        extract_join_paths: Whether to extract join paths
        join_path_threshold: Threshold for join path extraction
        mix_examples: If True, mix triplets from different examples in batches (better for TOTTO).
                     If False, keep triplets from same example together (better for Protrix).
        triplet_strategy: Strategy for generating triplets ('full', 'limited', 'random', 'balanced', 'primary_only')
        max_triplets_per_example: Maximum triplets per example for 'limited' or 'random' strategies
        enable_training_curves: Whether to enable training curves tracking and visualization
        track_batch_losses: Whether to track individual batch losses for detailed analysis
        track_val_loss: Whether to compute and track validation loss (requires additional compute)
        auto_plot_curves: Whether to automatically generate and save plots after each epoch
        enable_row_sent_eval: Whether to enable row-sentence level evaluation during training
        row_sent_test_file: Path to test dataset for row-sentence evaluation
        row_sent_annotation_file: Path to row-sentence annotation file
        row_sent_max_examples: Maximum number of test examples to evaluate (None = all examples)
        start_training_from_stage: Which stage to start training from: 0 (Encoder-only) or 1 (Sophisticated Model)
        
    Returns:
        The trained model (sophisticated model with optional encoder fine-tuning)
    """
    import data
    
    print("\nStarting training with ID-based triplet batches...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    use_header_conditioning = bool(getattr(model, 'use_header_conditioning', False))
    use_cell_level_matching = bool(getattr(model, 'use_cell_level_matching', False))
    
    # Initialize training curves tracker
    training_curves = None
    if enable_training_curves:
        training_curves = TrainingCurves(
            output_dir=str(output_path),
            run_name=run_name,
            track_batch_losses=track_batch_losses,
            track_val_loss=track_val_loss,
            track_row_sent_metrics=enable_row_sent_eval,
            auto_save=True,
            auto_plot=auto_plot_curves
        )
        print("🎯 Training curves tracking enabled")
        if save_best_by_test_metrics and enable_row_sent_eval:
            print("🏆 Test-based model saving enabled - will save best models by test metrics")
        # Set dynamic Stage 3 label for curves/summary
        try:
            training_curves.trained_stage_label = f"Trained Stage {start_training_from_stage}"
        except Exception:
            pass
    
    # ================================
    # IMPORTANT: Stage 0 cache logic MUST run BEFORE building any caches
    # to ensure consistent pre-training evaluation regardless of start_training_from_stage
    # ================================
    # Stage 0 cache logic: Determine intended training mode and cache behavior
    if start_training_from_stage == 0 and use_cache:
        # Predict what Stage 0 training mode will be based on user settings
        model_encoder_trainable = getattr(model, 'trainable_encoder', True)

        # Optional override via stage0_cache_mode
        override = str(stage0_cache_mode).lower() if isinstance(stage0_cache_mode, str) else "auto"
        if override not in {"auto", "on", "off"}:
            override = "auto"

        if override == "on":
            print("🔧 Stage 0 cache override: FORCING caches ON (use_cache=True)")
            print("   Note: If encoder parameters are trainable, cached embeddings will become stale and may harm training quality.")
            use_cache = True
        elif override == "off":
            print("🔧 Stage 0 cache override: FORCING caches OFF (use_cache=False)")
            use_cache = False
        else:
            # Determine intended training mode:
            # - If encoder_only_training=True: train encoder only (need to disable cache)
            # - If model has frozen encoder AND user wants cache: cross-attention only training (keep cache)
            # - Otherwise: full model training (disable cache)
            if encoder_only_training:
                print("⚠️  Stage 0 encoder-only training: disabling caches so encoder updates reflect in training/eval")
                use_cache = False
            elif not model_encoder_trainable and use_cache:
                print("✅ Stage 0 cross-attention only training: keeping caches enabled as requested")
                print("   Encoder frozen, training only cross-attention heads - perfect for component analysis!")
            else:
                print("⚠️  Stage 0 full model training: disabling caches so encoder updates reflect in training/eval")
                use_cache = False
    
    # ================================
    # NOW load evaluation data and build caches (with corrected use_cache value)
    # ================================
    
    # Load row-sentence evaluation data if enabled
    row_sent_test_examples = []
    row_sent_annotations = {}
    row_sent_test_cache = None
    row_sent_eval_format = (dataset_format or "other").lower()
    if row_sent_eval_format not in {"mimic", "other"}:
        print(f"⚠️  Unknown dataset_format='{dataset_format}', falling back to 'other'")
        row_sent_eval_format = "other"

    if enable_row_sent_eval:
        print(f"\n🔍 Loading row-sentence evaluation data (backend={row_sent_eval_format})...")
        try:
            _is_mimic_flipped_load = (row_sent_eval_format == "mimic"
                                      and native_direction.upper() == "DOC_TO_TABLE")
            if _is_mimic_flipped_load:
                row_sent_test_examples, row_sent_annotations = load_mimic_flipped_test_data_and_annotations(
                    row_sent_test_file, row_sent_annotation_file
                )
            elif row_sent_eval_format == "mimic":
                row_sent_test_examples, row_sent_annotations = load_mimic_test_data_and_annotations(
                    row_sent_test_file, row_sent_annotation_file
                )
            else:
                row_sent_test_examples, row_sent_annotations = load_test_data_and_annotations(
                    row_sent_test_file, row_sent_annotation_file
                )

            if row_sent_test_examples and row_sent_annotations:
                # Pre-filter to only examples that have matching annotations.
                # The eval loop skips unannotated examples anyway, so encoding
                # them into the cache is pure waste (~80% for big test sets).
                annotation_keys = set(row_sent_annotations.keys())
                pre_filter_count = len(row_sent_test_examples)
                row_sent_test_examples = [
                    ex for ex in row_sent_test_examples
                    if ex.get("admission_id") in annotation_keys
                ]
                if len(row_sent_test_examples) < pre_filter_count:
                    print(f"🔍 Pre-filtered test set to {len(row_sent_test_examples)}/{pre_filter_count} "
                          f"examples with annotations (skipping {pre_filter_count - len(row_sent_test_examples)} unannotated)")

                # Optional further subsampling (only relevant if annotation set is very large)
                if row_sent_max_examples is not None and row_sent_max_examples > 0 and len(row_sent_test_examples) > row_sent_max_examples:
                    # Deterministic sampling using anchor_id or id as staple keys
                    def get_stable_key(ex):
                        # Ensure we use the SAME key that annotations mapping uses (usually anchor_id/id)
                        return str(ex.get('anchor_id') or ex.get('id') or '')
                    
                    row_sent_test_examples_sorted = sorted(row_sent_test_examples, key=get_stable_key)
                    sampling_rng = random.Random(42)  # Fixed seed for consistency
                    row_sent_test_examples = sampling_rng.sample(row_sent_test_examples_sorted, row_sent_max_examples)
                    
                    # Filter annotations to match sampled examples.
                    # mimic_flipped annotations are keyed by admission_id (str), not anchor_id,
                    # so build a secondary key set from admission_id when available.
                    sampled_keys = {get_stable_key(ex) for ex in row_sent_test_examples}
                    sampled_admission_ids = {str(ex.get("admission_id", "")) for ex in row_sent_test_examples if ex.get("admission_id")}
                    row_sent_annotations = {
                        k: v for k, v in row_sent_annotations.items()
                        if str(k) in sampled_keys or str(k) in sampled_admission_ids
                    }
                    
                    print(f"⚡ Subsampled row-sentence evaluation to {len(row_sent_test_examples)} examples (deterministic)")

                if row_sent_max_examples is None or row_sent_max_examples == 0:
                    print(f"✅ Row-sentence evaluation enabled (all {len(row_sent_test_examples)} test examples)")
                else:
                    print(f"✅ Row-sentence evaluation enabled ({len(row_sent_test_examples)} test examples selected)")

                if use_cache:
                    print("🚀 Building test cache for row-sentence evaluation...")
                    from encoding import build_id_based_embedding_cache
                    row_sent_test_cache = build_id_based_embedding_cache(
                        examples=row_sent_test_examples,
                        sentence_encoder_model=model.sentence_encoder,
                        batch_size=eval_batch_size,
                        device=device,
                        split_name="test_row_sent",
                        verbose=verbose,
                        super_batch_size=encoding_batch_size,
                        task_direction=task_direction,
                        native_direction=native_direction,
                        use_header_conditioning=use_header_conditioning,
                        use_cell_level_matching=use_cell_level_matching,
                    )
                    print(f"✅ Test cache built successfully: {row_sent_test_cache.stats()}")
                else:
                    row_sent_test_cache = None
                    print("🔄 Cache disabled for training: row-sentence test will be re-encoded each epoch")
            else:
                print("⚠️  Row-sentence evaluation data not found, disabling evaluation")
                enable_row_sent_eval = False
        except Exception as e:
            print(f"⚠️  Error loading row-sentence evaluation data: {e}")
            print("   Continuing training without row-sentence evaluation")
            enable_row_sent_eval = False
    
    # Row-sentence backend is now selected explicitly via `row_sent_eval_backend`.
    
    # Prepare triplet batches for training
    print("Preparing triplet batches...")
    # Adaptive drop_last: For strategies that might generate many triplets globally
    # we may want to drop incomplete final batches. However, when using
    # isolated per-example batching (mix_examples=False) forcing drop_last for
    # the "full" strategy will drop small per-example batches and can result
    # in zero training batches. Only enable drop_last for full strategy when
    # triplets are mixed across examples to avoid accidentally discarding
    # per-example triplets.
    adaptive_drop_last = (triplet_strategy == "full") and mix_examples
    
    train_batches = data.prepare_triplet_batches(
        examples=train_examples,
        batch_size=train_batch_size,
        shuffle_triplets=True,
        drop_last=adaptive_drop_last,
        mix_examples=mix_examples,
        triplet_strategy=triplet_strategy,
        max_triplets_per_example=max_triplets_per_example,
        task_direction=task_direction,
        native_direction=native_direction,
        use_header_conditioning=use_header_conditioning,
    )
    
    # Validation batches (mirror training strategy so positive/negative distribution matches)
    val_batches = data.prepare_triplet_batches(
        examples=eval_examples,
        batch_size=eval_batch_size,
        shuffle_triplets=False,
        drop_last=False,
        mix_examples=mix_examples,
        triplet_strategy=triplet_strategy,
        max_triplets_per_example=max_triplets_per_example,
        task_direction=task_direction,
        native_direction=native_direction,
        use_header_conditioning=use_header_conditioning,
    )
    
    if adaptive_drop_last:
        print(f"Created {len(train_batches)} triplet batches for training (incomplete batches dropped)")
    else:
        print(f"Created {len(train_batches)} triplet batches for training (all triplets included, adaptive batching)")
    
    # Safety check: Ensure we have batches to train on
    if len(train_batches) == 0:
        raise ValueError(
            f"❌ No training batches were created! This usually means:\n"
            f"   1. Examples generated 0 triplets (check your data)\n"
            f"   2. All batches were dropped (total_triplets < batch_size with drop_last=True)\n"
            f"   Current config: strategy={triplet_strategy}, max_triplets={max_triplets_per_example}, "
            f"batch_size={train_batch_size}, drop_last={adaptive_drop_last}\n"
            f"   Try: reducing batch_size, or using primary_only strategy"
        )
    
    # Initialize embedding cache if requested
    train_cache = None
    eval_cache = None
    if use_cache:
        print("Initializing ID-based embedding caches...")
        from encoding import build_id_based_embedding_cache
        
        # Create separate cache for training examples
        if verbose:
            print("Building training cache...")
        train_cache = build_id_based_embedding_cache(
            examples=train_examples,
            sentence_encoder_model=model.sentence_encoder,
            batch_size=train_batch_size,
            device=device,
            split_name="train",
            verbose=verbose,
            super_batch_size=encoding_batch_size,
            task_direction=task_direction,
            native_direction=native_direction,
            use_header_conditioning=use_header_conditioning,
            use_cell_level_matching=use_cell_level_matching,
        )
        print(f"Training cache stats: {train_cache.stats()}")
        
        # Create separate cache for evaluation examples
        if verbose:
            print("\nBuilding evaluation cache...")
        eval_cache = build_id_based_embedding_cache(
            examples=eval_examples,
            sentence_encoder_model=model.sentence_encoder,
            batch_size=eval_batch_size,
            device=device,
            split_name="eval",
            verbose=verbose,
            super_batch_size=encoding_batch_size,
            task_direction=task_direction,
            native_direction=native_direction,
            use_header_conditioning=use_header_conditioning,
            use_cell_level_matching=use_cell_level_matching,
        )
        print(f"Evaluation cache stats: {eval_cache.stats()}")
    

    
    # Calculate total steps and warmup steps
    num_updates_per_epoch = len(train_batches) // gradient_accumulation_steps
    if len(train_batches) % gradient_accumulation_steps != 0:
        num_updates_per_epoch += 1  # Account for the remaining gradients step
    
    # Use real total steps without artificial inflation to keep schedule faithful
    total_steps = num_updates_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"Scheduler configured for {total_steps} steps with {warmup_steps} warmup steps")
    
    # Create model directory
    output_path = Path(output_path)
    model_dir = output_path / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize variables to track best model
    best_accuracy = 0.0
    best_epoch = 0
    best_state_dict = None
    
    # NEW: Variables to track best test metrics
    best_test_f1 = 0.0
    best_test_epoch = 0
    best_test_avg_precision = 0.0
    best_test_precision_epoch = 0
    best_test_state_dict = None
    best_test_precision_state_dict = None
    
    # Variables to track training progress
    last_loss = float('inf')
    patience = 0
    max_patience = 1  # Number of epochs with no improvement before resetting LR
    # Early stopping state
    early_stop_best_acc = 0.0
    epochs_without_improve = 0
    
    # ================================
    # STAGE 0: FROZEN ENCODER ONLY (TRUE BASELINE)
    # ================================
    print("\n🔥 STAGE 0: FROZEN ENCODER ONLY EVALUATION...")
    print("Evaluating with ONLY the frozen sentence encoder (no cross-attention at all)...")
    
    # For Stage 0 evaluation, we always want to use cached embeddings for speed.
    # If use_cache=False (e.g., for encoder training), we still build a temporary 
    # cache just for Stage 0 baseline evaluation since the encoder is frozen during eval.
    stage0_eval_cache = eval_cache
    if stage0_eval_cache is None:
        print("⚡ Building temporary cache for Stage 0 evaluation (encoder frozen during eval)...")
        from encoding import build_id_based_embedding_cache
        stage0_eval_cache = build_id_based_embedding_cache(
            examples=eval_examples,
            sentence_encoder_model=model.sentence_encoder,
            batch_size=eval_batch_size,
            device=device,
            split_name="stage0_eval",
            verbose=verbose,
            super_batch_size=encoding_batch_size,
            task_direction=task_direction,
            native_direction=native_direction,
            use_header_conditioning=use_header_conditioning,
            use_cell_level_matching=use_cell_level_matching,
        )
        print(f"Temporary Stage 0 cache stats: {stage0_eval_cache.stats()}")
    
    frozen_encoder_metrics = evaluate_frozen_encoder_baseline(
        sentence_encoder=model.sentence_encoder,
        eval_examples=eval_examples,
        batch_size=eval_batch_size,
        aggregation_method=aggregation_method,
        top_k=top_k,
        eval_cache=stage0_eval_cache,  # Use cached embeddings for ~100x speedup
        task_direction=task_direction,
        native_direction=native_direction
    )
    
    print(f"🔥 FROZEN ENCODER ONLY Accuracy: {frozen_encoder_metrics['accuracy']:.3f}")
    print(f"   (Total comparisons: {frozen_encoder_metrics['total_comparisons']})")
    
    # NEW: Row-sentence evaluation for Stage 0 if enabled
    stage_0_row_sent_metrics = {}
    stage0_row_sent_cache = None  # Initialize outside to reuse for Stage 1
    
    # DEBUG: Check state before Stage 0 row-sent evaluation
    print(f"\n🔍 DEBUG Stage 0 Row-Sent Eval Pre-check:")
    print(f"   enable_row_sent_eval = {enable_row_sent_eval}")
    print(f"   row_sent_test_examples count = {len(row_sent_test_examples) if row_sent_test_examples else 0}")
    print(f"   row_sent_annotations count = {len(row_sent_annotations) if row_sent_annotations else 0}")
    print(f"   Condition result = {bool(enable_row_sent_eval and row_sent_test_examples and row_sent_annotations)}")
    
    if enable_row_sent_eval and row_sent_test_examples and row_sent_annotations:
        print("🔍 Stage 0: Evaluating row-sentence alignment...")
        try:
            # Build a cache for Stage 0 row-sent eval if none exists (for speed)
            stage0_row_sent_cache = row_sent_test_cache
            if stage0_row_sent_cache is None:
                print("⚡ Building temporary cache for Stage 0 row-sentence evaluation...")
                from encoding import build_id_based_embedding_cache
                stage0_row_sent_cache = build_id_based_embedding_cache(
                    examples=row_sent_test_examples,
                    sentence_encoder_model=model.sentence_encoder,
                    batch_size=eval_batch_size,
                    device=device,
                    split_name="stage0_row_sent",
                    verbose=verbose,
                    super_batch_size=encoding_batch_size,
                    use_header_conditioning=use_header_conditioning,
                    use_cell_level_matching=use_cell_level_matching,
                )
            
            # Select evaluator backend for Stage 0.
            is_mimic = row_sent_eval_format == "mimic"
            is_mimic_flipped = is_mimic and native_direction.upper() == "DOC_TO_TABLE"

            if is_mimic_flipped:
                print("ℹ️  Using MIMIC-Flipped evaluator (DOC_TO_TABLE)...")
                stage_0_row_sent_metrics = evaluate_frozen_encoder_mimic_flipped(
                    sentence_encoder=model.sentence_encoder,
                    examples=row_sent_test_examples[:row_sent_max_examples] if row_sent_max_examples else row_sent_test_examples,
                    annotations=row_sent_annotations,
                    batch_size=1,
                    max_examples=row_sent_max_examples,
                    test_cache=stage0_row_sent_cache,
                )
                if isinstance(stage_0_row_sent_metrics, dict):
                    stage_0_row_sent_metrics['row_sent_f1'] = stage_0_row_sent_metrics.get('f1', stage_0_row_sent_metrics.get('row_sent_f1', 0.0))
                    stage_0_row_sent_metrics['row_sent_avg_precision'] = stage_0_row_sent_metrics.get('average_precision', stage_0_row_sent_metrics.get('row_sent_avg_precision', 0.0))
            elif is_mimic:
                print("ℹ️  Using MIMIC evaluator...")
                from evaluate_mimic_row_sent import evaluate_frozen_encoder_mimic
                stage_0_row_sent_metrics = evaluate_frozen_encoder_mimic(
                    sentence_encoder=model.sentence_encoder,
                    examples=row_sent_test_examples[:row_sent_max_examples] if row_sent_max_examples else row_sent_test_examples,
                    annotations=row_sent_annotations,
                    batch_size=1,
                    max_examples=row_sent_max_examples,
                    test_cache=stage0_row_sent_cache
                )
                if isinstance(stage_0_row_sent_metrics, dict):
                    stage_0_row_sent_metrics['row_sent_f1'] = stage_0_row_sent_metrics.get('f1', stage_0_row_sent_metrics.get('row_sent_f1', 0.0))
                    stage_0_row_sent_metrics['row_sent_avg_precision'] = stage_0_row_sent_metrics.get('average_precision', stage_0_row_sent_metrics.get('row_sent_avg_precision', 0.0))
            else:
                print("ℹ️  Using Unified Row-Sentence evaluator...")
                from row_sentence_eval import evaluate_frozen_encoder_only
                stage_0_row_sent_metrics = evaluate_frozen_encoder_only(
                    sentence_encoder=model.sentence_encoder,
                    examples=row_sent_test_examples[:row_sent_max_examples] if row_sent_max_examples else row_sent_test_examples,
                    annotations=row_sent_annotations,
                    batch_size=1,
                    test_cache=stage0_row_sent_cache
                )
                # Normalize keys for consistency with later stages
                if isinstance(stage_0_row_sent_metrics, dict):
                    stage_0_row_sent_metrics['row_sent_f1'] = stage_0_row_sent_metrics.get('f1', stage_0_row_sent_metrics.get('row_sent_f1', 0.0))
                    stage_0_row_sent_metrics['row_sent_avg_precision'] = stage_0_row_sent_metrics.get('average_precision', stage_0_row_sent_metrics.get('row_sent_avg_precision', 0.0))
                    
            print(f"🔥 Stage 0 Row-Sent Avg Precision: {stage_0_row_sent_metrics.get('row_sent_avg_precision', 0.0):.3f}")
            print(f"🔥 Stage 0 Row-Sent F1: {stage_0_row_sent_metrics.get('row_sent_f1', 0.0):.3f}")
        except Exception as e:
            print(f"⚠️  Stage 0 row-sentence evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            stage_0_row_sent_metrics = {}
    
    # ================================
    # STAGE 1: SOPHISTICATED MODEL (PRE-TRAINING)
    # ================================
    print("\n🚀 STAGE 1: SOPHISTICATED MODEL EVALUATION (PRE-TRAINING)...")
    print("Evaluating sophisticated model BEFORE training...")
    model.eval()
    
    # Always try CUDA first, then fall back to the model's current device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA device for evaluation")
        model.to(device)
    else:
        device = next(model.parameters()).device
        print(f"💻 Using device: {device} for evaluation")
    
    # NEW: Determine which model to train based on start_training_from_stage
    if start_training_from_stage == 0:
        print(f"\n🎯 TRAINING CONFIGURATION: Starting from Stage 0")
        
        # Check model's encoder trainability and LoRA settings to determine training strategy
        encoder_trainable = getattr(model, 'trainable_encoder', True)
        use_lora = getattr(model, 'use_cross_attention_lora', False) if hasattr(model, 'cross_attention') else False
        
        # NEW: Detect if encoder is wrapped with PEFT/LoRA (Unsloth QLoRA)
        encoder_has_peft = False
        if hasattr(model, 'sentence_encoder'):
            # Check for PEFT wrapper
            try:
                # Check for actual PEFT wrappers, not just attributes that might be None
                encoder_has_peft = any('lora' in name.lower() for name, _ in model.sentence_encoder.named_parameters())
            except Exception as e:
                print(f"   ⚠️ Error checking for PEFT/QLoRA on encoder: {e}")
                encoder_has_peft = False # Ensure it's false if an error occurs
        
        print(f"   📋 Configuration detected:")
        print(f"      Encoder trainable: {encoder_trainable}")
        print(f"      Use cache: {use_cache}")
        print(f"      Use LoRA: {use_lora}")
        print(f"      Encoder-only training: {encoder_only_training}")
        print(f"      Encoder has PEFT/QLoRA: {'✅ Yes' if encoder_has_peft else '❌ No'}")
        
        if encoder_only_training:
            print(f"   Mode: Encoder-only training (cross-attention heads frozen)")
            training_model = model
            training_model_type = "Encoder Only (No Heads)"
            
            # =====================================================================
            # CRITICAL: Handle PEFT vs non-PEFT encoder differently
            # =====================================================================
            if encoder_has_peft:
                # With PEFT/QLoRA: DON'T touch sentence_encoder params - PEFT manages them!
                # Only freeze non-encoder params (cross-attention heads, etc.)
                for name, p in training_model.named_parameters():
                    if "sentence_encoder" not in name:
                        p.requires_grad = False
                print("   🔒 Non-encoder params frozen; encoder params managed by PEFT")
            else:
                # Without PEFT: Enable all encoder params, freeze everything else
                for name, p in training_model.named_parameters():
                    p.requires_grad = ("sentence_encoder" in name)
                print("   🔒 Non-encoder params frozen; encoder params enabled for training")

            # =====================================================================
            # IMPORTANT: Skip gradual unfreezing when PEFT/QLoRA is active
            # =====================================================================
            # With PEFT/QLoRA, the base model weights are already frozen by PEFT, and only
            # LoRA adapter weights are trainable. Gradual unfreezing has no effect because:
            # 1. Base weights are frozen by PEFT regardless of requires_grad
            # 2. LoRA adapters are all trainable from the start
            # 3. The layer structure changes when wrapped by PEFT
            # =====================================================================
            if encoder_has_peft:
                print("   ⚠️ PEFT/QLoRA detected on encoder - skipping gradual unfreezing")
                print("   📝 With QLoRA, only LoRA adapter weights are trainable (base weights frozen by PEFT)")
                
                # Count trainable LoRA parameters vs frozen base parameters
                encoder_total = sum(p.numel() for p in training_model.sentence_encoder.parameters())
                encoder_trainable = sum(p.numel() for p in training_model.sentence_encoder.parameters() if p.requires_grad)
                encoder_frozen = encoder_total - encoder_trainable
                lora_params = sum(p.numel() for name, p in training_model.sentence_encoder.named_parameters() 
                                 if 'lora' in name.lower())
                
                print(f"   📊 Encoder parameter breakdown:")
                print(f"      Total encoder params: {encoder_total:,}")
                print(f"      LoRA adapters (trainable): {encoder_trainable:,} parameters")
                print(f"      Base model (frozen by PEFT): {encoder_frozen:,} parameters")
                if encoder_total > 0:
                    print(f"      Training only {encoder_trainable/encoder_total*100:.2f}% of encoder parameters")
                
                # Verify PEFT is actually working
                if encoder_trainable == encoder_total:
                    print(f"\n   ❌ ERROR: PEFT did NOT freeze base weights!")
                    print(f"      All {encoder_total:,} encoder parameters are trainable.")
                    print(f"      This is likely a bug - QLoRA should freeze ~95% of params.")
                elif encoder_trainable < encoder_total * 0.1:
                    print(f"   ✅ QLoRA verified: Only {encoder_trainable/encoder_total*100:.2f}% params trainable")
                    
            # Apply encoder tuning policy (full vs gradual) - ONLY if not using PEFT
            elif encoder_tuning_mode.lower() == "gradual":
                print("   Encoder tuning mode: gradual unfreezing")
                # Helper: detect encoder layer IDs by parsing parameter names
                import re
                layer_to_params: Dict[int, list] = {}
                pooler_params: list = []
                total_params = 0
                for name, p in training_model.sentence_encoder.named_parameters():
                    total_params += p.numel()
                    # Common patterns across BERT/Roberta/ModernBERT style encoders
                    match = re.search(r"(?:encoder\.|transformer\.|model\.)?(?:layer|layers|h)\.(\d+)\.", name)
                    if match:
                        layer_idx = int(match.group(1))
                        layer_to_params.setdefault(layer_idx, []).append(p)
                        continue
                    if "pooler" in name or "projection" in name:
                        pooler_params.append(p)
                        continue
                    # Embedding matrices or other shared components fall through (left frozen)

                if not layer_to_params:
                    print("   ⚠️ Could not detect encoder layers automatically; falling back to full training for encoder")
                else:
                    max_layer = max(layer_to_params.keys())
                    num_layers = max_layer + 1
                    # Start with all encoder params frozen
                    for p in training_model.sentence_encoder.parameters():
                        p.requires_grad = False
                    # Optionally keep pooler trainable
                    if gradual_unfreeze_include_pooler:
                        for p in pooler_params:
                            p.requires_grad = True
                    # Compute currently unfrozen layers for epoch 0
                    initial_k = max(1, int(gradual_unfreeze_initial_layers))
                    if gradual_unfreeze_max_layers and initial_k > gradual_unfreeze_max_layers:
                        initial_k = gradual_unfreeze_max_layers
                    start_from = max(0, num_layers - initial_k)
                    unfrozen_layers = list(range(start_from, num_layers))
                    for lid in unfrozen_layers:
                        for p in layer_to_params.get(lid, []):
                            p.requires_grad = True
                    print(f"   🔓 Gradual unfreezing initialized: {len(unfrozen_layers)}/{num_layers} top layers trainable (layers {start_from}..{num_layers-1})")
                    # Stash schedule for later epochs
                    training_model._gradual_unfreeze = {
                        "layer_to_params": layer_to_params,
                        "num_layers": num_layers,
                        "per_epoch_layers": max(1, int(gradual_unfreeze_every)),
                        "max_layers": int(gradual_unfreeze_max_layers) if gradual_unfreeze_max_layers else num_layers,
                        "include_pooler": bool(gradual_unfreeze_include_pooler)
                    }
                    # Safety: ensure at least one encoder parameter remains trainable
                    if not any(p.requires_grad for p in training_model.sentence_encoder.parameters()):
                        print("   🔒 Safety: No encoder params trainable after initialization; unfreezing top layer")
                        top_lid = num_layers - 1
                        for p in layer_to_params.get(top_lid, []):
                            p.requires_grad = True
        elif not encoder_trainable and use_cache:
            print(f"   Mode: Cross-attention only training (encoder frozen, using cache)")
            print(f"   This tests if cross-attention heads alone provide the performance gains")
            training_model = model
            training_model_type = "Cross-Attention Only (Frozen Encoder)"
            # Freeze encoder parameters, train only cross-attention heads
            encoder_params = 0
            cross_attention_params = 0
            for name, p in training_model.named_parameters():
                if "sentence_encoder" in name:
                    p.requires_grad = False
                    encoder_params += p.numel()
                else:
                    p.requires_grad = True  # Cross-attention heads and other components
                    cross_attention_params += p.numel()
            
            print(f"   📊 Parameter breakdown:")
            print(f"      Encoder (frozen): {encoder_params:,} parameters")
            print(f"      Cross-attention (trainable): {cross_attention_params:,} parameters")
            print(f"      Training only {cross_attention_params/(encoder_params+cross_attention_params)*100:.1f}% of total parameters")
        else:
            print(f"   Mode: Full model training (encoder + cross-attention heads)")
            training_model = model
            training_model_type = "Full Model (Encoder + Heads)"
            # Train all parameters (classic Stage 0 behavior)
            for p in training_model.parameters():
                p.requires_grad = True
    else:
        # Default: start_training_from_stage == 1 (Sophisticated Model)
        print(f"\n🚀 TRAINING CONFIGURATION: Starting from Stage 1 (Sophisticated Model) - Default behavior")
        print(f"   Will train the sophisticated model as usual")
        
        # Use the sophisticated model for training
        training_model = model
        training_model_type = "Sophisticated Model"
    
    # Enable gradient checkpointing if requested (for memory efficiency)
    if enable_checkpointing and hasattr(training_model.sentence_encoder, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing for memory efficiency")
        training_model.sentence_encoder.gradient_checkpointing_enable()
    
    # Set up optimizer and loss function for the training model
    training_model = training_model.to(device)
    # Parameter groups: lower LR for encoder when fine-tuning to avoid representation drift
    # Always include all encoder params in the optimizer (even if currently frozen) to support gradual unfreezing.
    encoder_params_all = [p for n, p in training_model.named_parameters() if "sentence_encoder" in n]
    head_params = [p for n, p in training_model.named_parameters() if "sentence_encoder" not in n and p.requires_grad]

    effective_encoder_lr = encoder_learning_rate if encoder_learning_rate is not None else max(learning_rate * 0.25, 1e-6)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": learning_rate, "base_lr": learning_rate, "min_lr": learning_rate * max(min_lr_ratio, 0.0)})
    if encoder_params_all:
        param_groups.append({"params": encoder_params_all, "lr": effective_encoder_lr, "base_lr": effective_encoder_lr, "min_lr": effective_encoder_lr * max(min_lr_ratio, 0.0)})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    print(f"Optimizer groups configured: head={len(head_params)} @ {learning_rate:.2e}, encoder={len(encoder_params_all)} @ {effective_encoder_lr:.2e}")
    
    # Create the appropriate loss function
    if encoder_only_training:
        from losses import EncoderOnlyTripletLoss
        if train_cache is not None:
            print("⚠️  Ignoring training cache for encoder-only; encoder is being updated each step")
        ranking_loss_type = getattr(model, 'ranking_loss_type', 'softplus')
        infonce_tau = getattr(model, 'infonce_tau', 0.7)
        loss_fn = EncoderOnlyTripletLoss(
            sentence_encoder=training_model.sentence_encoder,
            device=device,
            margin=margin,
            scale=scale,
            ranking_loss_type=ranking_loss_type,
            infonce_tau=infonce_tau,
            use_hard_negative_mining=use_hard_negative_mining,
            hard_negative_topk=hard_negative_topk
        )
        print(f"Initialized EncoderOnlyTripletLoss with margin={margin}, scale={scale}")
    elif loss_type == "enhanced_triplet":
        from losses import EnhancedTripletLoss
        loss_fn = EnhancedTripletLoss(
            model=training_model, 
            cache=train_cache,
            aggregation_method=aggregation_method,
            triplet_weight=triplet_weight,  # FIXED: Use normalized weights from arguments
            attention_weight=attention_loss_weight,  # Use normalized weights from arguments
            margin=margin,
            scale=scale,
            use_hard_negative_mining=use_hard_negative_mining,
            hard_negative_topk=hard_negative_topk
        )
        print(f"Initialized EnhancedTripletLoss with margin={loss_fn.margin}, scale={loss_fn.scale}")
        print(f"Using Enhanced triplet loss with aggregation: {aggregation_method}")
        print(f"  Normalized weights - Triplet: {loss_fn.triplet_weight:.3f}, Attention: {loss_fn.attention_weight:.3f}")
    elif loss_type == "bidirectional_triplet":
        from losses import BidirectionalTripletLoss
        loss_fn = BidirectionalTripletLoss(
            model=training_model, 
            cache=train_cache,
            aggregation_method=aggregation_method,
            triplet_weight=triplet_weight,  # FIXED: Use normalized weights from arguments
            attention_weight=attention_loss_weight,  # Use normalized weights from arguments
            diversity_weight=diversity_weight,
            pair_weight=pair_loss_weight,  # Use normalized weights from arguments
            pair_margin=pair_margin,
            direct_attention_weight=direct_attention_loss_weight,
            direct_attention_diversity_weight=direct_attention_diversity_weight,
            direct_attention_hub_weight=direct_attention_hub_weight,
            direct_attention_entropy_weight=direct_attention_entropy_weight,
            direct_attention_entropy_floor_ratio=direct_attention_entropy_floor_ratio,
            forward_attention_weight=forward_attention_loss_weight,
            pair_mil_weight=pair_mil_loss_weight,
            pair_mil_positive_margin=pair_mil_positive_margin,
            pair_mil_negative_margin=pair_mil_negative_margin,
            pair_mil_sparsity_weight=pair_mil_sparsity_weight,
            pair_mil_hub_weight=pair_mil_hub_weight,
            margin=margin,
            scale=scale,
            # NEW: Attention distillation parameters
            use_attention_distillation=use_attention_distillation,
            distillation_weight=distillation_weight,
            teacher_temperature=teacher_temperature,
            student_temperature=student_temperature,
            distillation_loss_type=distillation_loss_type,
            teacher_hub_centering=teacher_hub_centering,
            # NEW: SIGReg and Sinkhorn structural regularization
            sigreg_weight=sigreg_weight,
            sigreg_target_std=sigreg_target_std,
            sigreg_num_proj=sigreg_num_proj,
            sigreg_knots=sigreg_knots,
            sinkhorn_weight=sinkhorn_weight,
        )
        print(f"Initialized BidirectionalTripletLoss with margin={loss_fn.margin}, scale={loss_fn.scale}")
        print(f"Using Bidirectional triplet loss with aggregation: {aggregation_method}")
        weight_parts = [f"Triplet: {loss_fn.triplet_weight:.3f}", f"Attention: {loss_fn.attention_weight:.3f}",
                        f"Diversity: {loss_fn.diversity_weight:.3f}", f"DirectAttn: {loss_fn.direct_attention_weight:.3f}",
                        f"ForwardAttn: {loss_fn.forward_attention_weight:.3f}", f"PairMIL: {loss_fn.pair_mil_weight:.3f}",
                        f"Pair: {loss_fn.pair_weight:.3f}"]
        if use_attention_distillation:
            weight_parts.append(f"Distillation: {loss_fn.distillation_weight:.3f}")
            print(f"  Attention distillation enabled: teacher_temp={teacher_temperature}, student_temp={student_temperature}, loss_type={distillation_loss_type}, hub_centering={teacher_hub_centering}")
        if sigreg_weight > 0:
            weight_parts.append(f"SIGReg: {loss_fn.sigreg_weight:.3f}")
        if sinkhorn_weight > 0:
            weight_parts.append(f"Sinkhorn: {loss_fn.sinkhorn_weight:.3f}")
        print(f"  Normalized weights - {', '.join(weight_parts)}")
    elif loss_type == "id_cached_triplet":
        loss_fn = IdBasedCachedTripletLoss(model=training_model, cache=train_cache, margin=margin, scale=scale,
                                           use_hard_negative_mining=use_hard_negative_mining,
                                           hard_negative_topk=hard_negative_topk)
        print(f"Initialized IdBasedCachedTripletLoss with margin={loss_fn.margin}, scale={loss_fn.scale}")
        print("Using standard ID-based cached triplet loss")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose from ['enhanced_triplet', 'bidirectional_triplet', 'id_cached_triplet']")
    
    # Define learning rate scheduler (moved here after optimizer is defined)
    if lr_scheduler_type.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print("Using linear LR schedule with warmup")
    elif lr_scheduler_type.lower() == "cosine_restart":
        # Use hard restarts; cycles approximate T_0 schedule
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=max(1, int(lr_num_cycles))
        )
        print(f"Using cosine-with-restarts LR schedule with warmup (cycles={lr_num_cycles})")
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print("Using cosine LR schedule with warmup")
    
    # For Stage 1 evaluation (pre-training), we can safely use cached embeddings since
    # no training has happened yet. REUSE Stage 0 cache if available - same encoder state!
    stage1_eval_cache = eval_cache
    if stage1_eval_cache is None and stage0_eval_cache is not None:
        print("⚡ Reusing Stage 0 cache for Stage 1 pre-training evaluation (same encoder state)")
        stage1_eval_cache = stage0_eval_cache
    elif stage1_eval_cache is None:
        print("⚡ Building temporary cache for Stage 1 pre-training evaluation...")
        from encoding import build_id_based_embedding_cache
        stage1_eval_cache = build_id_based_embedding_cache(
            examples=eval_examples,
            sentence_encoder_model=model.sentence_encoder,
            batch_size=eval_batch_size,
            device=device,
            split_name="stage1_eval",
            verbose=verbose,
            super_batch_size=encoding_batch_size,
            task_direction=task_direction,
            native_direction=native_direction,
            use_header_conditioning=use_header_conditioning,
            use_cell_level_matching=use_cell_level_matching,
        )
        print(f"Temporary Stage 1 cache stats: {stage1_eval_cache.stats()}")
    
    # Use appropriate evaluation function based on model type
    if isinstance(model, BidirectionalTableTextModel):
        # Import bidirectional evaluation function
        try:
            from run_cross_attention import evaluate_bidirectional_with_join_paths
            print("🔧 Using bidirectional evaluation for bidirectional model")
            initial_metrics = evaluate_bidirectional_with_join_paths(
                model=model,
                examples=eval_examples,
                id_cache=stage1_eval_cache,  # Use cache for ~100x speedup
                batch_size=eval_batch_size,
                aggregation_method=aggregation_method,
                evaluation_margin=0.0
            )
        except ImportError:
            print("⚠️ Could not import bidirectional evaluation, falling back to standard evaluation")
            initial_metrics = evaluate_with_id_cache(model, eval_examples, stage1_eval_cache, eval_batch_size, aggregation_method, allow_cache_build=True)
    else:
        print("🔧 Using standard evaluation for unidirectional model")
        initial_metrics = evaluate_with_id_cache(model, eval_examples, stage1_eval_cache, eval_batch_size, aggregation_method, allow_cache_build=True)
    
    print(f"🚀 SOPHISTICATED MODEL Accuracy (Pre-training): {initial_metrics['accuracy']:.3f}")
    
    # NEW: Row-sentence evaluation for Stage 1 (Sophisticated Untrained) if enabled
    stage_1_row_sent_metrics = {}
    if enable_row_sent_eval and row_sent_test_examples and row_sent_annotations:
        print("🔍 Stage 1: Evaluating row-sentence alignment (Sophisticated Untrained)...")
        try:
            # REUSE Stage 0 row-sent cache if available - same encoder state before training!
            stage1_row_sent_cache = row_sent_test_cache
            if stage1_row_sent_cache is None and stage0_row_sent_cache is not None:
                print("⚡ Reusing Stage 0 row-sent cache for Stage 1 (same encoder state)")
                stage1_row_sent_cache = stage0_row_sent_cache
            elif stage1_row_sent_cache is None:
                print("⚡ Building temporary cache for Stage 1 row-sentence evaluation...")
                from encoding import build_id_based_embedding_cache
                stage1_row_sent_cache = build_id_based_embedding_cache(
                    examples=row_sent_test_examples,
                    sentence_encoder_model=model.sentence_encoder,
                    batch_size=eval_batch_size,
                    device=device,
                    split_name="stage1_row_sent",
                    verbose=verbose,
                    super_batch_size=encoding_batch_size,
                    use_header_conditioning=use_header_conditioning,
                    use_cell_level_matching=use_cell_level_matching,
                )
            
            # Detect evaluation backend for Stage 1.
            is_mimic_stage1 = row_sent_eval_format == "mimic"
            is_mimic_flipped_stage1 = is_mimic_stage1 and native_direction.upper() == "DOC_TO_TABLE"
            is_totto_stage1 = row_sent_eval_format == "totto"
            if not is_mimic_stage1 and not is_totto_stage1:
                if "tables" in row_sent_test_examples[0]:
                    is_mimic_stage1 = True
                elif row_sent_annotations and isinstance(row_sent_annotations, dict):
                    sample_key = next(iter(row_sent_annotations.keys()), None)
                    if sample_key is not None and isinstance(sample_key, str):
                        is_mimic_stage1 = True

            if is_mimic_flipped_stage1:
                stage_1_row_sent_metrics = evaluate_mimic_flipped_with_model(
                    model=model,
                    test_examples=row_sent_test_examples,
                    annotations=row_sent_annotations,
                    max_examples=row_sent_max_examples,
                    test_cache=stage1_row_sent_cache,
                )
                stage_1_row_sent_metrics['row_sent_f1'] = stage_1_row_sent_metrics.get('f1', stage_1_row_sent_metrics.get('row_sent_f1', 0.0))
                stage_1_row_sent_metrics['row_sent_avg_precision'] = stage_1_row_sent_metrics.get('average_precision', stage_1_row_sent_metrics.get('row_sent_avg_precision', 0.0))
            elif is_mimic_stage1:
                from evaluate_mimic_row_sent import evaluate_mimic_with_model
                stage_1_row_sent_metrics = evaluate_mimic_with_model(
                    model=model,
                    test_examples=row_sent_test_examples,
                    annotations=row_sent_annotations,
                    max_examples=row_sent_max_examples,
                    test_cache=stage1_row_sent_cache
                )
                # Map MIMIC keys to standard row_sent keys
                stage_1_row_sent_metrics['row_sent_f1'] = stage_1_row_sent_metrics.get('f1', stage_1_row_sent_metrics.get('row_sent_f1', 0.0))
                stage_1_row_sent_metrics['row_sent_avg_precision'] = stage_1_row_sent_metrics.get('average_precision', stage_1_row_sent_metrics.get('row_sent_avg_precision', 0.0))
            elif is_totto_stage1:
                stage_1_row_sent_metrics = quick_mimic_row_grounding_eval(
                    model=model,
                    test_examples=row_sent_test_examples,
                    annotations=row_sent_annotations,
                    max_examples=row_sent_max_examples,
                    test_cache=row_sent_test_cache
                )
            else:
                stage_1_row_sent_metrics = quick_row_sentence_eval(
                    model=model,
                    test_examples=row_sent_test_examples,
                    annotations=row_sent_annotations,
                    max_examples=row_sent_max_examples,
                    test_cache=row_sent_test_cache
                )          
            print(f"🚀 Stage 1 Row-Sent Avg Precision: {stage_1_row_sent_metrics.get('row_sent_avg_precision', 0.0):.3f}")
            print(f"🚀 Stage 1 Row-Sent F1: {stage_1_row_sent_metrics.get('row_sent_f1', 0.0):.3f}")
            if enable_attention_diagnostics and isinstance(model, BidirectionalTableTextModel):
                _compute_attention_collapse_diagnostics(
                    model=model,
                    examples=row_sent_test_examples,
                    id_cache=stage1_row_sent_cache,
                    max_examples=attention_diagnostic_examples,
                    batch_size=eval_batch_size,
                    aggregation_method=aggregation_method,
                )
        except Exception as e:
            print(f"⚠️  Stage 1 row-sentence evaluation failed: {e}")
            stage_1_row_sent_metrics = {}
    
    # Show the progressive benefits (Stage 0 -> Stage 1 only now)
    architecture_benefit = initial_metrics['accuracy'] - frozen_encoder_metrics['accuracy']
    
    print(f"\n📊 ARCHITECTURE BENEFIT:")
    print(f"   🔥→🚀 Sophisticated model benefit: +{architecture_benefit:.3f} ({architecture_benefit*100:.1f}%)")
    
    # NEW: Show row-sentence progression if enabled
    if enable_row_sent_eval and stage_0_row_sent_metrics and stage_1_row_sent_metrics:
        print(f"\n📈 ROW-SENTENCE PROGRESSION (Avg Precision):")
        stage_0_ap = stage_0_row_sent_metrics.get('row_sent_avg_precision', stage_0_row_sent_metrics.get('average_precision', 0.0))
        stage_1_ap = stage_1_row_sent_metrics.get('row_sent_avg_precision', stage_1_row_sent_metrics.get('average_precision', 0.0))
        print(f"   🔥 Stage 0 (Frozen): {stage_0_ap:.3f}")
        print(f"   🚀 Stage 1 (Sophisticated): {stage_1_ap:.3f} (+{stage_1_ap - stage_0_ap:.3f})")
    
    if architecture_benefit < 0.01:
        print("⚠️  WARNING: Sophisticated model barely improves over frozen encoder!")
        print("   Consider if the architecture complexity is necessary for this task.")
    
    # ================================
    # WANDB LOGGING - Log baseline/initial metrics (Epoch 0)
    # ================================
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        print("📊 Logging initial stage metrics to wandb...")
        # Log Stage 0 (frozen encoder) and Stage 1 (untrained model) baselines
        wandb_baseline_metrics = {
            "epoch": 0,
            # Stage 0 - Frozen Encoder Baseline
            "baseline/stage0_frozen_encoder_accuracy": frozen_encoder_metrics['accuracy'],
            "baseline/stage0_total_comparisons": frozen_encoder_metrics.get('total_comparisons', 0),
            # Stage 1 - Sophisticated Model (Untrained)
            "baseline/stage1_untrained_accuracy": initial_metrics['accuracy'],
            "baseline/architecture_benefit": architecture_benefit,
            # Initial val accuracy (same as Stage 1 for epoch 0)
            "val/accuracy": initial_metrics['accuracy'],
            "val/best_accuracy": initial_metrics['accuracy'],
            "val/best_epoch": 0,
            # Training metrics (epoch 0 has no training)
            "train/loss_mean": 0.0,
            "train/learning_rate": learning_rate,
        }
        
        # Add Stage 0 row-sentence metrics if available
        if stage_0_row_sent_metrics:
            wandb_baseline_metrics.update({
                "baseline/stage0_row_sent_f1": stage_0_row_sent_metrics.get('row_sent_f1', stage_0_row_sent_metrics.get('f1', 0.0)),
                "baseline/stage0_row_sent_avg_precision": stage_0_row_sent_metrics.get('row_sent_avg_precision', stage_0_row_sent_metrics.get('average_precision', 0.0)),
            })
        
        # Add Stage 1 row-sentence metrics if available
        if stage_1_row_sent_metrics:
            wandb_baseline_metrics.update({
                "baseline/stage1_row_sent_f1": stage_1_row_sent_metrics.get('row_sent_f1', 0.0),
                "baseline/stage1_row_sent_avg_precision": stage_1_row_sent_metrics.get('row_sent_avg_precision', 0.0),
                # Also set as initial test metrics
                "test/f1": stage_1_row_sent_metrics.get('row_sent_f1', 0.0),
                "test/avg_precision": stage_1_row_sent_metrics.get('row_sent_avg_precision', 0.0),
                "test/best_f1": stage_1_row_sent_metrics.get('row_sent_f1', 0.0),
                "test/best_avg_precision": stage_1_row_sent_metrics.get('row_sent_avg_precision', 0.0),
            })
        
        wandb.log(wandb_baseline_metrics, step=0)
        print("✅ Baseline metrics logged to wandb (epoch 0)")
    
    # NEW: Add initial stage metrics to training curves if enabled
    if training_curves is not None:
        print("\n📊 Adding initial stage metrics to training curves...")
        # Note: Stage 0 metrics use 'row_sent_avg_precision'/'row_sent_f1' keys (set in lines 781-782, 797-798)
        # Also try 'average_precision'/'f1' as fallback for compatibility
        stage_0_ap = 0.0
        stage_0_acc = 0.0
        if stage_0_row_sent_metrics:
            stage_0_ap = stage_0_row_sent_metrics.get('row_sent_avg_precision', 
                         stage_0_row_sent_metrics.get('average_precision', 0.0))
            stage_0_acc = stage_0_row_sent_metrics.get('row_sent_f1',
                          stage_0_row_sent_metrics.get('f1', 0.0))
            print(f"   DEBUG: Stage 0 row-sent metrics keys: {list(stage_0_row_sent_metrics.keys())}")
            print(f"   DEBUG: Stage 0 AP = {stage_0_ap:.4f}, Acc = {stage_0_acc:.4f}")
        else:
            print("   DEBUG: stage_0_row_sent_metrics is empty or None")
        
        training_curves.add_initial_stage_metrics(
            stage_0_accuracy=frozen_encoder_metrics['accuracy'],
            stage_1_accuracy=initial_metrics['accuracy'],
            stage_2_accuracy=0.0,  # Will be updated after training
            stage_0_row_sent_ap=stage_0_ap,
            stage_1_row_sent_ap=stage_1_row_sent_metrics.get('row_sent_avg_precision', 0.0) if stage_1_row_sent_metrics else 0.0,
            stage_2_row_sent_ap=0.0,  # Will be updated after training
            stage_0_row_sent_acc=stage_0_acc,
            stage_1_row_sent_acc=stage_1_row_sent_metrics.get('row_sent_f1', 0.0) if stage_1_row_sent_metrics else 0.0,
            stage_2_row_sent_acc=0.0  # Will be updated after training
        )
    
    # Add Epoch 0 data to training curves if enabled
    if training_curves is not None:
        print("📊 Adding Epoch 0 (untrained model) baseline to training curves...")
        
        # Use initial_metrics for validation accuracy (this represents the model being trained)
        epoch_0_val_accuracy = initial_metrics['accuracy']
        
        # For row-sentence metrics, use the sophisticated model's pre-training metrics (Stage 1)
        if enable_row_sent_eval:
            if start_training_from_stage == 1:
                # Training sophisticated model - use Stage 1 (Sophisticated untrained) row-sent metrics
                epoch_0_row_sent_f1 = stage_1_row_sent_metrics.get('row_sent_f1')
                epoch_0_row_sent_ap = stage_1_row_sent_metrics.get('row_sent_avg_precision')
            else:
                # Training from Stage 0 - use Stage 0 row-sent metrics
                epoch_0_row_sent_f1 = stage_0_row_sent_metrics.get('row_sent_f1', stage_0_row_sent_metrics.get('f1')) if stage_0_row_sent_metrics else None
                epoch_0_row_sent_ap = stage_0_row_sent_metrics.get('row_sent_avg_precision', stage_0_row_sent_metrics.get('average_precision')) if stage_0_row_sent_metrics else None
        else:
            epoch_0_row_sent_f1 = None
            epoch_0_row_sent_ap = None
        
        training_curves.add_epoch_0_data(
            val_accuracy=epoch_0_val_accuracy,
            row_sent_f1=epoch_0_row_sent_f1,
            row_sent_avg_precision=epoch_0_row_sent_ap
        )
    
    # Training loop
    print(f"\n🎓 STARTING TRAINING OF {training_model_type.upper()}")
    print(f"   Training model type: {training_model_type}")
    print(f"   Training from Stage: {start_training_from_stage}")
    
    for epoch in range(epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{epochs} {'='*20}")
        
        # Start timing this epoch
        epoch_start_time = time.time()
        
        # Apply gradual unfreezing schedule at the START of each epoch (after epoch 0 initialization)
        if (start_training_from_stage == 0 
            and encoder_only_training 
            and 'encoder_tuning_mode' in locals() 
            and str(encoder_tuning_mode).lower() == 'gradual' 
            and hasattr(training_model, '_gradual_unfreeze')):
            gu = training_model._gradual_unfreeze
            current = gu.get('current_unfrozen', 0)
            max_layers = int(gu.get('max_layers', gu.get('num_layers', 0)))
            per_epoch = int(gu.get('per_epoch_layers', 1))
            num_layers = int(gu.get('num_layers', 0))
            layer_to_params = gu.get('layer_to_params', {})
            if current == 0:
                # When initialized earlier, set it based on actual trainable layers
                # Count currently trainable layers among top layers
                trainable_layers = []
                for lid in range(num_layers):
                    any_trainable = any(p.requires_grad for p in layer_to_params.get(lid, []))
                    if any_trainable:
                        trainable_layers.append(lid)
                current = len(trainable_layers)
            # Determine new target number of unfrozen layers
            target = min(max_layers, max(current, 0) + (per_epoch if epoch > 0 else 0))
            if target > current and num_layers > 0:
                start_from = max(0, num_layers - target)
                newly_unfrozen = []
                for lid in range(num_layers - current - 1, start_from - 1, -1):
                    # Unfreeze from top down
                    for p in layer_to_params.get(lid, []):
                        p.requires_grad = True
                    newly_unfrozen.append(lid)
                training_model._gradual_unfreeze['current_unfrozen'] = target
                if newly_unfrozen:
                    low = min(newly_unfrozen)
                    high = max(newly_unfrozen)
                    print(f"   🔓 Gradual unfreezing: enabled layers {low}..{high} (total {target}/{num_layers})")
            # Safety: ensure at least one encoder parameter is trainable
            if not any(p.requires_grad for p in training_model.sentence_encoder.parameters()):
                print("   🔒 Safety: No encoder params trainable after schedule; unfreezing top layer")
                top_lid = num_layers - 1
                for p in layer_to_params.get(top_lid, []):
                    p.requires_grad = True
        
        # Training phase
        training_model.train()
        total_loss = 0.0
        epoch_losses = []
        
        # Quick parameter check (only once at the beginning of training)
        if epoch == 0:
            total_params = sum(p.numel() for p in training_model.parameters())
            trainable_params = sum(p.numel() for p in training_model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            print(f"Training model parameter summary:")
            print(f"   Total: {total_params:,}")
            print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
            print(f"   Frozen: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
            
            # Extra detail for encoder if it exists
            if hasattr(training_model, 'sentence_encoder'):
                enc_total = sum(p.numel() for p in training_model.sentence_encoder.parameters())
                enc_trainable = sum(p.numel() for p in training_model.sentence_encoder.parameters() if p.requires_grad)
                enc_lora = sum(p.numel() for name, p in training_model.sentence_encoder.named_parameters() 
                              if 'lora' in name.lower())
                print(f"   Encoder: {enc_trainable:,}/{enc_total:,} trainable ({enc_trainable/enc_total*100:.2f}%)")
                if enc_lora > 0:
                    print(f"   LoRA params: {enc_lora:,}")
        
        # Process batches
        progress_bar = tqdm(train_batches, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()  # Zero gradients once before accumulation loop
        
        # Update margin schedule per epoch
        if margin_schedule != "none" and margin_end is not None and epochs > 1:
            mix = epoch / max(epochs - 1, 1)
            current_margin = float(margin + (margin_end - margin) * mix)
            if hasattr(loss_fn, 'margin'):
                loss_fn.margin = current_margin
        else:
            current_margin = getattr(loss_fn, 'margin', margin)
        
        for i, batch in enumerate(progress_bar):
            # Process the batch with our loss function
            loss = loss_fn(batch) / gradient_accumulation_steps
            # Stage 0 safety: ensure loss has grad path (skip if PEFT manages the encoder)
            encoder_has_peft_active = (hasattr(training_model, 'sentence_encoder') and 
                                       (hasattr(training_model.sentence_encoder, 'peft_config') or 
                                        hasattr(training_model.sentence_encoder, 'active_adapters') or
                                        any('lora' in n.lower() for n, _ in training_model.sentence_encoder.named_parameters())))
            
            if (start_training_from_stage == 0 and encoder_only_training 
                and not loss.requires_grad and not encoder_has_peft_active):
                print("   ⚠️  Stage 0: loss has no grad_fn; ensuring encoder has trainable params and re-anchoring loss")
                # Ensure at least one encoder param is trainable
                enc_params = list(training_model.sentence_encoder.parameters())
                if not any(p.requires_grad for p in enc_params):
                    # Unfreeze top layer if we can detect layers
                    if hasattr(training_model, '_gradual_unfreeze'):
                        gu = training_model._gradual_unfreeze
                        layer_to_params = gu.get('layer_to_params', {})
                        num_layers = int(gu.get('num_layers', 0))
                        top_lid = max(0, num_layers - 1)
                        for p in layer_to_params.get(top_lid, []):
                            p.requires_grad = True
                    else:
                        # Fallback: unfreeze all encoder params
                        for p in enc_params:
                            p.requires_grad = True
                # Re-anchor loss to a trainable encoder param to restore grad path
                try:
                    anchor_param = next(p for p in enc_params if p.requires_grad)
                    loss = loss + 0.0 * anchor_param.view(-1)[0]
                except StopIteration:
                    pass
            
            # Backward pass
            loss.backward()
            
            # Update metrics (store the actual loss value, not divided by accumulation steps)
            actual_loss = loss.item() * gradient_accumulation_steps
            total_loss += actual_loss
            epoch_losses.append(actual_loss)
            
            # Update progress bar
            progress_bar.set_postfix({'batch_loss': f'{loss.item():.3f}'})
            
            # Print gradient stats occasionally
            if i == 0 or i % 50 == 0:
                total_grad_norm = 0
                for name, param in training_model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        total_grad_norm += grad_norm ** 2
                
                total_grad_norm = total_grad_norm ** 0.5
                # print(f"Step {i}, Total gradient norm: {total_grad_norm:.4f}")
            
            # Perform optimizer step after accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                # Enforce LR floors per param group
                for pg in optimizer.param_groups:
                    if 'min_lr' in pg:
                        pg['lr'] = max(pg['lr'], pg['min_lr'])
                optimizer.zero_grad()
        
        # Final step for any remaining gradients
        if len(train_batches) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            for pg in optimizer.param_groups:
                if 'min_lr' in pg:
                    pg['lr'] = max(pg['lr'], pg['min_lr'])
        
        # Print current learning rates for all parameter groups
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Current learning rates: {', '.join([f'{lr:.2e}' for lr in current_lrs])} | margin={current_margin:.3f}")
        
        # Print loss statistics
        if epoch_losses:
            mean_loss = np.mean(epoch_losses)
            min_loss = np.min(epoch_losses)
            max_loss = np.max(epoch_losses)
            print(f"Epoch {epoch+1} loss statistics - Mean: {mean_loss:.3f}, Min: {min_loss:.3f}, Max: {max_loss:.3f}")
            
            # Print encoding optimization stats if available (for Stage 0 PEFT training)
            if hasattr(loss_fn, 'get_encoding_stats'):
                enc_stats = loss_fn.get_encoding_stats()
                if enc_stats.get('forward_passes', 0) > 0:
                    print(f"   ⚡ Encoding optimization stats:")
                    print(f"      Total texts: {enc_stats.get('total_texts', 0):,}")
                    print(f"      Unique texts (after dedup): {enc_stats.get('unique_texts', 0):,}")
                    print(f"      Forward passes: {enc_stats.get('forward_passes', 0):,}")
                    print(f"      Dedup ratio: {enc_stats.get('dedup_ratio', 1.0):.2%}")
                    print(f"      Avg texts/forward: {enc_stats.get('avg_texts_per_forward', 0):.1f}")
                # Reset stats for next epoch
                if hasattr(loss_fn, 'reset_encoding_stats'):
                    loss_fn.reset_encoding_stats()
            
            # Check if we're making progress
            if mean_loss >= last_loss:
                patience += 1
                print(f"Loss not decreasing. Patience: {patience}/{max_patience}")
                if patience >= max_patience:
                    # Reset learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate * 0.5  # Reduce learning rate
                    print(f"Reset learning rate to {optimizer.param_groups[0]['lr']:.2e}")
                    patience = 0  # Reset patience counter
            else:
                patience = 0  # Reset patience if we're making progress
                
            last_loss = mean_loss  # Update the last loss
        
        # Evaluate after epoch
        print("Evaluating after epoch...")
        training_model.eval()
        
        # Stage-aware validation evaluation
        if start_training_from_stage == 0 and encoder_only_training:
            # Evaluate Stage 0 (encoder-only) using frozen-encoder baseline logic (cosine on embeddings)
            # Rebuild cache each epoch to reflect encoder updates (PEFT/LoRA changes the effective embeddings)
            print("🔎 Validation evaluator: Stage 0 (frozen encoder baseline)")
            print("⚡ Rebuilding eval cache to reflect encoder updates...")
            from encoding import build_id_based_embedding_cache
            epoch_eval_cache = build_id_based_embedding_cache(
                examples=eval_examples,
                sentence_encoder_model=training_model.sentence_encoder,
                batch_size=eval_batch_size,
                device=device,
                split_name=f"eval_epoch{epoch+1}",
                verbose=False,
                super_batch_size=encoding_batch_size,
                task_direction=task_direction,
                native_direction=native_direction,
                use_header_conditioning=use_header_conditioning,
                use_cell_level_matching=use_cell_level_matching,
            )
            eval_metrics = evaluate_frozen_encoder_baseline(
                sentence_encoder=training_model.sentence_encoder,
                eval_examples=eval_examples,
                batch_size=eval_batch_size,
                aggregation_method=aggregation_method,
                top_k=top_k,
                eval_cache=epoch_eval_cache,
                task_direction=task_direction,
                native_direction=native_direction
            )
        else:
            # Use model forward-based evaluation for Stage 2 (Sophisticated Model)
            print("🔎 Validation evaluator: Stage 2 (sophisticated model forward)")
            if isinstance(training_model, BidirectionalTableTextModel):
                # Import bidirectional evaluation function
                try:
                    from run_cross_attention import evaluate_bidirectional_with_join_paths
                    eval_metrics = evaluate_bidirectional_with_join_paths(
                        model=training_model,
                        examples=eval_examples,
                        id_cache=eval_cache if use_cache else None,
                        batch_size=eval_batch_size,
                        aggregation_method=aggregation_method,
                        evaluation_margin=0.0
                    )
                except ImportError:
                    print("⚠️ Could not import bidirectional evaluation, falling back to standard evaluation")
                    eval_metrics = evaluate_with_id_cache(training_model, eval_examples, eval_cache if use_cache else None, eval_batch_size, aggregation_method, allow_cache_build=use_cache)
            else:
                eval_metrics = evaluate_with_id_cache(training_model, eval_examples, eval_cache if use_cache else None, eval_batch_size, aggregation_method, allow_cache_build=use_cache)
        
        print(f"Epoch {epoch+1} Accuracy: {eval_metrics['accuracy']:.3f}")

        # Early stopping based on validation accuracy
        if epoch + 1 >= early_stopping_min_epochs:
            if eval_metrics['accuracy'] > early_stop_best_acc + 1e-6:
                early_stop_best_acc = eval_metrics['accuracy']
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best val acc: {early_stop_best_acc:.3f}")
                    # Break out of training loop early
                    # Best weights are tracked separately below
                    break
        
        # Row-sentence evaluation if enabled
        row_sent_metrics = {}
        if enable_row_sent_eval and row_sent_test_examples and row_sent_annotations:
            # print("🔍 Evaluating row-sentence alignment...")  # Commented out - redundant with epoch summary
            try:
                # Rebuild test cache per epoch when caching is disabled; otherwise reuse
                effective_test_cache = row_sent_test_cache
                if not use_cache:
                    from encoding import build_id_based_embedding_cache
                    effective_test_cache = build_id_based_embedding_cache(
                        examples=row_sent_test_examples,
                        sentence_encoder_model=training_model.sentence_encoder,
                        batch_size=eval_batch_size,
                        device=device,
                        split_name="test_row_sent_epoch",
                        verbose=False,
                        super_batch_size=encoding_batch_size,
                        use_header_conditioning=use_header_conditioning,
                        use_cell_level_matching=use_cell_level_matching,
                    )
                # Detect evaluation backend for Stage 2 loop.
                is_mimic_format = row_sent_eval_format == "mimic"
                is_mimic_flipped_format = is_mimic_format and native_direction.upper() == "DOC_TO_TABLE"
                if not is_mimic_format and row_sent_test_examples and len(row_sent_test_examples) > 0:
                    if "tables" in row_sent_test_examples[0]:
                        is_mimic_format = True
                    if row_sent_annotations and isinstance(row_sent_annotations, dict):
                        sample_key = next(iter(row_sent_annotations.keys()), None)
                        if sample_key is not None and isinstance(sample_key, str):
                            is_mimic_format = True

                # Stage-aware row-sentence evaluation
                if start_training_from_stage == 0 and encoder_only_training:
                    print("🔎 Row-sent evaluator: Stage 0 (frozen encoder baseline)")
                    if is_mimic_flipped_format:
                        print("ℹ️  Using MIMIC-Flipped evaluator (DOC_TO_TABLE)...")
                        row_sent_metrics = evaluate_frozen_encoder_mimic_flipped(
                            sentence_encoder=training_model.sentence_encoder,
                            examples=row_sent_test_examples[:row_sent_max_examples] if row_sent_max_examples else row_sent_test_examples,
                            annotations=row_sent_annotations,
                            batch_size=eval_batch_size,
                            max_examples=row_sent_max_examples,
                            test_cache=effective_test_cache,
                        )
                    elif is_mimic_format:
                        print("ℹ️  Using MIMIC evaluator...")
                        from evaluate_mimic_row_sent import evaluate_frozen_encoder_mimic
                        row_sent_metrics = evaluate_frozen_encoder_mimic(
                            sentence_encoder=training_model.sentence_encoder,
                            examples=row_sent_test_examples[:row_sent_max_examples] if row_sent_max_examples else row_sent_test_examples,
                            annotations=row_sent_annotations,
                            batch_size=eval_batch_size,
                            max_examples=row_sent_max_examples,
                            test_cache=effective_test_cache
                        )
                    else:
                        print("ℹ️  Using Unified Row-Sentence evaluator...")
                        from row_sentence_eval import evaluate_frozen_encoder_only
                        row_sent_metrics = evaluate_frozen_encoder_only(
                            sentence_encoder=training_model.sentence_encoder,
                            examples=row_sent_test_examples[:row_sent_max_examples] if row_sent_max_examples else row_sent_test_examples,
                            annotations=row_sent_annotations,
                            batch_size=eval_batch_size,
                            test_cache=effective_test_cache
                        )
                    # Normalize keys to align with Stage 2 metric names
                    if isinstance(row_sent_metrics, dict):
                        row_sent_metrics['row_sent_f1'] = row_sent_metrics.get('f1', row_sent_metrics.get('row_sent_f1', 0.0))
                        row_sent_metrics['row_sent_avg_precision'] = row_sent_metrics.get('average_precision', row_sent_metrics.get('row_sent_avg_precision', 0.0))
                else:
                    # Stage 2: Sophisticated model evaluation
                    print("🔎 Row-sent evaluator: Stage 2 (sophisticated model forward)")

                    if is_mimic_flipped_format:
                        row_sent_metrics = evaluate_mimic_flipped_with_model(
                            model=training_model,
                            test_examples=row_sent_test_examples,
                            annotations=row_sent_annotations,
                            max_examples=row_sent_max_examples,
                            test_cache=effective_test_cache,
                        )
                        if isinstance(row_sent_metrics, dict):
                            row_sent_metrics['row_sent_f1'] = row_sent_metrics.get('f1', row_sent_metrics.get('row_sent_f1', 0.0))
                            row_sent_metrics['row_sent_avg_precision'] = row_sent_metrics.get('average_precision', row_sent_metrics.get('row_sent_avg_precision', 0.0))
                            row_sent_metrics['examples_evaluated'] = row_sent_metrics.get('examples_evaluated', 0)
                    elif is_mimic_format:
                        from evaluate_mimic_row_sent import evaluate_mimic_with_model
                        row_sent_metrics = evaluate_mimic_with_model(
                            model=training_model,
                            test_examples=row_sent_test_examples,
                            annotations=row_sent_annotations,
                            max_examples=row_sent_max_examples,
                            test_cache=effective_test_cache
                        )
                        # Map MIMIC keys to standard row_sent keys
                        if isinstance(row_sent_metrics, dict):
                            row_sent_metrics['row_sent_f1'] = row_sent_metrics.get('f1', row_sent_metrics.get('row_sent_f1', 0.0))
                            row_sent_metrics['row_sent_avg_precision'] = row_sent_metrics.get('average_precision', row_sent_metrics.get('row_sent_avg_precision', 0.0))
                            row_sent_metrics['examples_evaluated'] = row_sent_metrics.get('examples_evaluated', 0)
                    else:
                        row_sent_metrics = quick_row_sentence_eval(
                            model=training_model,
                            test_examples=row_sent_test_examples,
                            annotations=row_sent_annotations,
                            max_examples=row_sent_max_examples,
                            test_cache=effective_test_cache
                        )
                print(f"   Row-Sent F1: {row_sent_metrics.get('row_sent_f1', 0.0):.3f}")
                print(f"   Row-Sent Avg Precision: {row_sent_metrics.get('row_sent_avg_precision', 0.0):.3f}")
                print(f"   Examples evaluated: {row_sent_metrics.get('examples_evaluated', 0)}")
                if enable_attention_diagnostics and isinstance(training_model, BidirectionalTableTextModel):
                    _compute_attention_collapse_diagnostics(
                        model=training_model,
                        examples=row_sent_test_examples,
                        id_cache=effective_test_cache,
                        max_examples=attention_diagnostic_examples,
                        batch_size=eval_batch_size,
                        aggregation_method=aggregation_method,
                    )
            except Exception as e:
                print(f"⚠️  Row-sentence evaluation failed: {e}")
                row_sent_metrics = {}
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # ================================
        # WANDB LOGGING - Log all metrics at each epoch
        # ================================
        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            # Prepare wandb metrics dict
            wandb_metrics = {
                "epoch": epoch + 1,
                # Training metrics
                "train/loss_mean": np.mean(epoch_losses) if epoch_losses else 0.0,
                "train/loss_min": np.min(epoch_losses) if epoch_losses else 0.0,
                "train/loss_max": np.max(epoch_losses) if epoch_losses else 0.0,
                "train/loss_std": np.std(epoch_losses) if epoch_losses else 0.0,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/epoch_time_seconds": epoch_time,
                "train/margin": current_margin,
                # Validation metrics
                "val/accuracy": eval_metrics['accuracy'],
                "val/total_comparisons": eval_metrics.get('total_comparisons', 0),
            }
            
            # Add test metrics (row-sentence evaluation) if available
            if enable_row_sent_eval and row_sent_metrics:
                wandb_metrics.update({
                    "test/f1": row_sent_metrics.get('row_sent_f1', 0.0),
                    "test/avg_precision": row_sent_metrics.get('row_sent_avg_precision', 0.0),
                    "test/examples_evaluated": row_sent_metrics.get('examples_evaluated', 0),
                })
                # Also track best test metrics for easy comparison
                wandb_metrics.update({
                    "test/best_f1": best_test_f1,
                    "test/best_avg_precision": best_test_avg_precision,
                })
            
            # Track best validation accuracy
            wandb_metrics["val/best_accuracy"] = best_accuracy
            wandb_metrics["val/best_epoch"] = best_epoch
            
            # Log all metrics to wandb
            wandb.log(wandb_metrics, step=epoch + 1)
        
        # Add data to training curves tracker
        if training_curves is not None:
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Add epoch data (val_loss could be computed here if track_val_loss is True)
            val_loss = None
            if track_val_loss:
                # TODO: Implement validation loss computation if needed
                # This would require running the model on validation set with loss function
                pass
            
            training_curves.add_epoch_data(
                epoch=epoch + 1,
                train_losses=epoch_losses,
                val_accuracy=eval_metrics['accuracy'],
                learning_rate=current_lr,
                epoch_time=epoch_time,
                val_loss=val_loss,
                row_sent_f1=row_sent_metrics.get('row_sent_f1'),
                row_sent_avg_precision=row_sent_metrics.get('row_sent_avg_precision')
            )
            # Log Stage 0 dynamics explicitly
            if start_training_from_stage == 0:
                print(f"   Stage 0 note: encoder fine-tuning active; eval uses on-the-fly embeddings (no cache)")
        
        # Save the model if it's the best so far
        if eval_metrics['accuracy'] > best_accuracy:
            best_accuracy = eval_metrics['accuracy']
            best_epoch = epoch + 1
            best_state_dict = {
                name: param.clone().detach().cpu()
                for name, param in training_model.state_dict().items()
            }
            print(f"New best model! Accuracy: {best_accuracy:.3f}")
        
        # NEW: Track best test metrics and optionally save models
        if enable_row_sent_eval and row_sent_metrics:
            current_test_acc = row_sent_metrics.get('row_sent_f1', 0.0)
            current_test_prec = row_sent_metrics.get('row_sent_avg_precision', 0.0)
            
            # Track best test F1
            if current_test_acc > best_test_f1:
                best_test_f1 = current_test_acc
                best_test_epoch = epoch + 1
                best_test_state_dict = {
                    name: param.clone().detach().cpu()
                    for name, param in training_model.state_dict().items()
                }
                print(f"New best test F1! {best_test_f1:.3f} (Epoch {best_test_epoch})")
            
            # Track best test average precision
            if current_test_prec > best_test_avg_precision:
                best_test_avg_precision = current_test_prec
                best_test_precision_epoch = epoch + 1
                best_test_precision_state_dict = {
                    name: param.clone().detach().cpu()
                    for name, param in training_model.state_dict().items()
                }
                print(f"New best test average precision! {best_test_avg_precision:.3f} (Epoch {best_test_precision_epoch})")
    
    # Load the best model
    if best_state_dict is not None:
        print(f"\nLoading best model from epoch {best_epoch} with accuracy {best_accuracy:.3f}")
        training_model.load_state_dict(best_state_dict)
        
        # Save model
        best_model_path = model_dir / f"best_model_epoch_{best_epoch}"
        best_model_path.mkdir(parents=True, exist_ok=True)
        
        model_path = best_model_path / "model.pt"
        torch.save(training_model.state_dict(), model_path)
        print(f"Best model saved to {model_path}")
    
    # NEW: Save best test-based models if requested
    if save_best_by_test_metrics and enable_row_sent_eval:
        print(f"\n🎯 SAVING BEST TEST-BASED MODELS:")
        
        # Save best test F1 model
        if best_test_state_dict is not None:
            print(f"Loading best test F1 model from epoch {best_test_epoch} ({best_test_f1:.3f})")
            training_model.load_state_dict(best_test_state_dict)
            
            best_test_model_path = model_dir / f"best_test_f1_epoch_{best_test_epoch}"
            best_test_model_path.mkdir(parents=True, exist_ok=True)
            
            test_model_path = best_test_model_path / "model.pt"
            torch.save(training_model.state_dict(), test_model_path)
            print(f"Best test F1 model saved to {test_model_path}")
        
        # Save best test average precision model
        if best_test_precision_state_dict is not None:
            print(f"Loading best test average precision model from epoch {best_test_precision_epoch} ({best_test_avg_precision:.3f})")
            training_model.load_state_dict(best_test_precision_state_dict)
            
            best_test_prec_model_path = model_dir / f"best_test_avg_precision_epoch_{best_test_precision_epoch}"
            best_test_prec_model_path.mkdir(parents=True, exist_ok=True)
            
            test_prec_model_path = best_test_prec_model_path / "model.pt"
            torch.save(training_model.state_dict(), test_prec_model_path)
            print(f"Best test average precision model saved to {test_prec_model_path}")
        
        # Reload the validation-based best model for final analysis
        if best_state_dict is not None:
            training_model.load_state_dict(best_state_dict)
            print(f"Reloaded validation-based best model for final analysis")

    if enable_attention_diagnostics and isinstance(training_model, BidirectionalTableTextModel) and row_sent_test_examples:
        print("\n🔎 FINAL ATTENTION COLLAPSE DIAGNOSTIC:")
        _compute_attention_collapse_diagnostics(
            model=training_model,
            examples=row_sent_test_examples,
            id_cache=row_sent_test_cache,
            max_examples=attention_diagnostic_examples,
            batch_size=eval_batch_size,
            aggregation_method=aggregation_method,
        )
    
    # ================================
    # FINAL 3-STAGE IMPACT ANALYSIS
    # ================================
    print("\n" + "="*90)
    print("🎯 COMPREHENSIVE 3-STAGE IMPACT ANALYSIS")
    print("="*90)
    
    # Calculate all stage improvements (3 stages now)
    architecture_benefit = initial_metrics['accuracy'] - frozen_encoder_metrics['accuracy']
    training_improvement = best_accuracy - initial_metrics['accuracy']
    total_improvement = best_accuracy - frozen_encoder_metrics['accuracy']
    
    print(f"📊 COMPLETE PERFORMANCE BREAKDOWN:")
    print(f"   🔥 Stage 0 - Frozen Encoder Only:     {frozen_encoder_metrics['accuracy']:.3f}")
    print(f"   🚀 Stage 1 - Sophisticated (Pre):     {initial_metrics['accuracy']:.3f} (+{architecture_benefit:.3f})")
    print(f"   🏆 Stage 2 - Trained Model:           {best_accuracy:.3f} (+{training_improvement:.3f})")
    print(f"   📈 Total Improvement:                 +{total_improvement:.3f} ({total_improvement*100:.1f}%)")
    
    # NEW: Add test metrics summary if available
    if enable_row_sent_eval and (best_test_f1 > 0 or best_test_avg_precision > 0):
        print(f"\n🏆 TEST METRICS SUMMARY:")
        print(f"   🎯 Best Test F1:     {best_test_f1:.3f} (Epoch {best_test_epoch})")
        print(f"   🎯 Best Test Average Precision:    {best_test_avg_precision:.3f} (Epoch {best_test_precision_epoch})")
        print(f"   📊 Validation vs Test Performance:")
        print(f"      - Validation Best: {best_accuracy:.3f} (Epoch {best_epoch})")
        print(f"      - Test F1: {best_test_f1:.3f} (Epoch {best_test_epoch})")
        print(f"      - Test Avg Precision: {best_test_avg_precision:.3f} (Epoch {best_test_precision_epoch})")
        
        # Check if best test metrics occurred at different epochs than validation
        if best_epoch != best_test_epoch:
            print(f"   ⚠️  NOTE: Best test F1 occurred at epoch {best_test_epoch}, not at best validation epoch {best_epoch}")
        if best_epoch != best_test_precision_epoch:
            print(f"   ⚠️  NOTE: Best test average precision occurred at epoch {best_test_precision_epoch}, not at best validation epoch {best_epoch}")
    
    print(f"\n🔍 DETAILED IMPACT ATTRIBUTION:")
    if total_improvement > 0:
        arch_pct = (architecture_benefit/total_improvement*100)
        tr_pct = (training_improvement/total_improvement*100)
        print(f"   🚀 Architecture Contribution:      {architecture_benefit:.3f} ({arch_pct:.1f}% of total)")
        print(f"   🎓 Training Contribution:          {training_improvement:.3f} ({tr_pct:.1f}% of total)")
    else:
        print(f"   🚀 Architecture Contribution:      {architecture_benefit:.3f}")
        print(f"   🎓 Training Contribution:          {training_improvement:.3f}")
    
    print(f"\n💡 COMPONENT ASSESSMENT:")
    # Architecture assessment
    if architecture_benefit > 0.05:
        print("✅ ARCHITECTURE: Significant benefit - sophisticated model is effective")
    elif architecture_benefit > 0.02:
        print("✅ ARCHITECTURE: Moderate benefit - cross-attention is helping")
    elif architecture_benefit > 0.005:
        print("⚠️  ARCHITECTURE: Small benefit - consider if complexity is worth it")
    else:
        print("❌ ARCHITECTURE: Minimal benefit - consider simpler approaches")
    
    # Training assessment
    if training_improvement > architecture_benefit:
        print("✅ TRAINING: Primary driver of performance - excellent!")
    elif training_improvement > 0.02:
        print("✅ TRAINING: Effective improvement - training is working")
    elif training_improvement > 0.005:
        print("⚠️  TRAINING: Modest improvement - consider longer training")
    else:
        print("❌ TRAINING: Minimal improvement - check hyperparameters/data")
    
    print("="*90)
    
    # ================================
    # WANDB LOGGING - Log final summary metrics
    # ================================
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        print("📊 Logging final summary metrics to wandb...")
        wandb_summary = {
            # Final stage breakdown
            "summary/stage0_frozen_encoder_accuracy": frozen_encoder_metrics['accuracy'],
            "summary/stage1_untrained_accuracy": initial_metrics['accuracy'],
            "summary/stage2_trained_best_accuracy": best_accuracy,
            "summary/architecture_benefit": architecture_benefit,
            "summary/training_improvement": training_improvement,
            "summary/total_improvement": total_improvement,
            # Best model info
            "summary/best_val_accuracy": best_accuracy,
            "summary/best_val_epoch": best_epoch,
            # Test metrics summary
            "summary/best_test_f1": best_test_f1,
            "summary/best_test_epoch": best_test_epoch,
            "summary/best_test_avg_precision": best_test_avg_precision,
            "summary/best_test_precision_epoch": best_test_precision_epoch,
            # Training info
            "summary/total_epochs_trained": epoch + 1,
            "summary/early_stopped": epochs_without_improve >= early_stopping_patience,
        }
        
        # Log to wandb summary (persisted across runs for comparison)
        for key, value in wandb_summary.items():
            wandb.run.summary[key] = value
        
        # Also log as final step
        wandb.log(wandb_summary, step=epochs + 1)
        print("✅ Final summary logged to wandb")
    
    # ================================
    # AUTOMATIC 3-STAGE VISUALIZATION  
    # ================================
    if skip_four_stage_viz:
        print("\n🎨 Skipping 3-STAGE example visualizations (skip_four_stage_viz=True)")
    else:
        print("\n🎨 GENERATING 3-STAGE VISUALIZATION...")
        try:
            from visualize_attention import create_complete_four_stage_analysis
            
            # Use a subset of evaluation examples for visualization
            viz_examples = eval_examples[:3]  # First 3 examples for visualization
            viz_output_dir = model_dir / "three_stage_analysis"
            
            create_complete_four_stage_analysis(
                trained_model=training_model,
                examples=viz_examples,
                output_dir=str(viz_output_dir),
                example_indices="0,1,2",
                base_model_name="all-roberta-large-v1",
                init_method=init_method,
                init_method_params=init_method_params,
                stage_3_label="Stage 2 - Trained"
            )
            
            print(f"✅ 3-stage visualizations saved to: {viz_output_dir}")
            
        except Exception as e:
            print(f"⚠️ Could not generate 3-stage visualization: {e}")
            print("   You can run visualization manually using demo_four_stage_visualization.py")
    
    print("="*90)
    
    # Final training curves summary and cleanup
    if training_curves is not None:
        print("\n" + "="*70)
        print("🎯 TRAINING CURVES SUMMARY")
        print("="*70)
        
        # Print comprehensive summary
        training_curves.print_summary()
        
        # Generate final plots (will be auto-saved)
        training_curves.plot_curves(filename=f"{run_name}_final_training_curves.png")
        
        # Generate batch-level analysis if enabled
        if track_batch_losses:
            print("📈 Generating batch-level analysis...")
            training_curves.plot_batch_losses()  # Plot all epochs heatmap
            # Plot specific epochs of interest
            if len(training_curves.epochs) > 0:
                training_curves.plot_batch_losses(epoch=1)  # First epoch
                if len(training_curves.epochs) > 1:
                    training_curves.plot_batch_losses(epoch=len(training_curves.epochs))  # Last epoch
        
        print("✅ Training curves analysis complete!")
        print(f"📁 All plots saved to: {training_curves.plots_dir}")
        print(f"💾 Training data saved to: {training_curves.data_dir}")
    
    print(f"\n🎓 TRAINING COMPLETED: Returning trained {training_model_type}")
    return training_model 