#!/usr/bin/env python3
"""
Protrix Row-Sentence Evaluation Script

This script evaluates table-text models on the Protrix test dataset with manual annotations.
Unlike the original evaluation that used highlighted_cells as row indices only, this version
uses the precise row-sentence pair annotations in the format [row_index, sentence_index].

Features:
- 4-stage evaluation framework (frozen encoder → simple → sophisticated → trained)
- Protrix-specific row-sentence pair evaluation
- Standard IR metrics: Precision, Recall, F1, NDCG @K=1,3,5 + "all"
- ROC-AUC and Average Precision for row-sentence pairs
- Comprehensive visualizations and analysis
- Proper TP/FP/FN calculations for F1 score
- Handles both empty and annotated examples

Annotation Format:
- highlighted_cells: [[row_idx, sent_idx], ...] where:
  - row_idx: 1-based row number from table
  - sent_idx: 0-based sentence index from text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os
import warnings
import time
import hashlib
import shutil
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from loki_path import ensure_loki_on_path

ensure_loki_on_path()

# Import model and data utilities
from data import load_row_level_dataset, IdBasedEmbeddingCache
from encoding import build_id_based_embedding_cache
from models import BidirectionalTableTextModel, TableTextEmbeddingModel
from sentence_transformers import SentenceTransformer
from utils import save_plot_multi_format

# Import optimization utilities for faster evaluation
try:
    from unsloth_encoder import (
        UNSLOTH_AVAILABLE,
        create_unsloth_sentence_encoder,
        TORCH_COMPILE_AVAILABLE,
        optimize_model_for_inference,
        print_optimization_status,
    )
except (ImportError, NotImplementedError, Exception):
    UNSLOTH_AVAILABLE = False
    TORCH_COMPILE_AVAILABLE = False
    def optimize_model_for_inference(model, **kwargs):
        return model
    def print_optimization_status():
        print("⚠️ Optimization module not available")
    def create_unsloth_sentence_encoder(*args, **kwargs):
        raise ImportError("Unsloth encoder not available")

# Set environment variables and disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

class RandomCrossAttentionWrapper(nn.Module):
    """
    Stage 0: Frozen encoder baseline.
    
    Uses the same cross-attention structure as Stage 1 but with specified initialization.
    This creates a fair baseline that shows the value of training (not initialization).
    
    Key principle: Same architecture as cross-attention models, with same initialization,
    but weights are never trained, creating a structurally comparable baseline.
    """
    
    def __init__(self, sentence_encoder: SentenceTransformer, embedding_dim: int = None, 
                 init_method: str = "xavier_uniform", init_method_params: dict = None):
        super(RandomCrossAttentionWrapper, self).__init__()
        self.sentence_encoder = sentence_encoder
        
        # Get embedding dimension from sentence encoder if not provided
        if embedding_dim is None:
            self.embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = embedding_dim
            
        self.init_method = init_method
        self.init_method_params = init_method_params or {}
        
        # Simple cross-attention layers (same structure as Stage 1)
        self.W_Q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.W_K = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.W_V = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        
        # Initialize weights in FP32 on CPU first (avoids BF16 CPU QR issues), then cast
        self._init_attention_weights()

        # Match dtype to sentence encoder AFTER initialization
        try:
            encoder_dtype = next(self.sentence_encoder.parameters()).dtype
            self.W_Q.to(dtype=encoder_dtype)
            self.W_K.to(dtype=encoder_dtype)
            self.W_V.to(dtype=encoder_dtype)
            print(f"🔧 Matched linear layer dtype to encoder: {encoder_dtype}")
        except Exception:
            print("🔧 Could not detect encoder dtype, using default")
        
        # Freeze the sentence encoder completely
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False
            
        # Freeze attention weights too - this baseline never learns
        for param in [self.W_Q, self.W_K, self.W_V]:
            for p in param.parameters():
                p.requires_grad = False
        
        print(f"🔧 RandomCrossAttentionWrapper initialized - frozen encoder baseline")
    
    def _init_attention_weights(self):
        """Initialize with specified method for consistent baseline."""
        layers = [self.W_Q, self.W_K, self.W_V]
        print(f"🎯 Initializing Stage 0 baseline with method: {self.init_method}")
        
        from initialization import initialize_attention_weights
        initialize_attention_weights(
            layers=layers,
            attention_dim=self.embedding_dim,
            method=self.init_method,
            method_params=self.init_method_params
        )
        print(f"✅ Successfully applied {self.init_method} initialization to Stage 0 baseline")
    
    def encode_sentences(self, sentences: List[str], batch_size: int = 32, normalize: bool = True) -> torch.Tensor:
        """
        Encode sentences using the sentence transformer (same as other stages).
        This ensures we use the computed embeddings, not waste them.
        """
        # Use sentence encoder (same as all other stages)
        embeddings = self.sentence_encoder.encode(
            sentences,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings
    
    def forward(self, row_embeddings: torch.Tensor, sentence_embeddings: torch.Tensor, 
                aggregation_method: str = "top_k_pairs") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Frozen encoder baseline forward pass: same structure as Stage 1.
        
        This provides a structurally comparable baseline that
        demonstrates the value of proper initialization and training.
        
        Args:
            row_embeddings: [batch_size, num_rows, embedding_dim]
            sentence_embeddings: [batch_size, num_sentences, embedding_dim]
            aggregation_method: Method for aggregation (ignored, for compatibility)
            
        Returns:
            Tuple of (global_similarity, pair_scores)
        """
        batch_size, num_rows, embedding_dim = row_embeddings.shape
        batch_size, num_sentences, embedding_dim = sentence_embeddings.shape
        
        # Cross-attention: rows attend to sentences (same structure as Stage 1)
        Q = self.W_Q(row_embeddings)  # [batch, num_rows, embedding_dim]
        K = self.W_K(sentence_embeddings)  # [batch, num_sentences, embedding_dim]
        V = self.W_V(sentence_embeddings)  # [batch, num_sentences, embedding_dim]
        
        # Attention scores (same computation as Stage 1)
        attention_scores = torch.bmm(Q, K.transpose(-2, -1))  # [batch, num_rows, num_sentences]
        attention_scores = attention_scores / (embedding_dim ** 0.5)  # Scale
        
        # Softmax attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.bmm(attention_weights, V)  # [batch, num_rows, embedding_dim]
        
        # Compute pair scores: similarity between rows and sentences via attention
        # The attention already gives us row-to-sentence alignments, so use attention weights as pair scores
        pair_scores = attention_weights  # [batch, num_rows, num_sentences]
        
        # Global similarity (max across all pairs)
        global_similarity = torch.max(pair_scores.view(batch_size, -1), dim=1)[0]
        
        return global_similarity, pair_scores
    
    def get_contextualized_pair_scores(self, row_embeddings: torch.Tensor, 
                                       sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        For frozen encoder baseline, return attention-based pair scores.
        This provides compatibility with the unidirectional model evaluation path.
        """
        # Use the forward pass to get pair_scores (attention weights)
        _, pair_scores = self.forward(row_embeddings, sentence_embeddings)
        return pair_scores

# Utils from original evaluation
def safe_tensor_to_numpy(tensor):
    """Safely convert PyTorch tensor to numpy, handling BFloat16 conversion."""
    if hasattr(tensor, 'dtype') and tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.cpu().numpy()

def safe_tensor_to_scalar(tensor):
    """Safely convert PyTorch tensor to Python scalar, handling BFloat16 conversion."""
    if hasattr(tensor, 'dtype') and tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.item()

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def load_protrix_annotations(annotation_file: str = "protrix_test/Annotated_Test.json") -> Dict[int, List[List[int]]]:
    """Load Protrix annotations mapping anchor_id to highlighted_cells."""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = {}
    for entry in data:
        anchor_id = entry["anchor_id"]
        highlighted_cells = entry["highlighted_cells"]
        annotations[anchor_id] = highlighted_cells
    
    return annotations

def create_output_directory(base_name="protrix_evaluation"):
    """Create timestamped output directory for evaluation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def copy_training_config_to_output(config_path: str, output_dir: str) -> bool:
    """
    Copy the training configuration file to the output directory for tracking.
    
    Args:
        config_path: Path to the training configuration file
        output_dir: Output directory where to copy the config
        
    Returns:
        bool: True if successfully copied, False otherwise
    """
    try:
        # Check if source file exists
        if not os.path.exists(config_path):
            print(f"⚠️ Training config not found at: {config_path}")
            return False
        
        # Create destination path
        config_filename = os.path.basename(config_path)
        dest_path = os.path.join(output_dir, config_filename)
        
        # Copy the file
        shutil.copy2(config_path, dest_path)
        print(f"✅ Training config copied to: {dest_path}")
        
        # Also copy the model file if it exists (for complete tracking)
        # Try to find the model file in the same directory as config
        config_dir = os.path.dirname(config_path)
        model_files = [f for f in os.listdir(config_dir) if f.endswith('.pt')] if os.path.exists(config_dir) else []
        
        if model_files:
            # Copy the first model file found
            model_path = os.path.join(config_dir, model_files[0])
            model_filename = os.path.basename(model_path)
            model_dest_path = os.path.join(output_dir, model_filename)
            shutil.copy2(model_path, model_dest_path)
            print(f"✅ Model file copied to: {model_dest_path}")
        
        # Also copy args.json if it exists
        args_path = os.path.join(config_dir, "args.json")
        if os.path.exists(args_path):
            args_filename = os.path.basename(args_path)
            args_dest_path = os.path.join(output_dir, args_filename)
            shutil.copy2(args_path, args_dest_path)
            print(f"✅ Args file copied to: {args_dest_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error copying training config: {e}")
        return False

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

def calculate_ndcg_at_k(y_true, y_scores, k):
    """Calculate NDCG@K (Normalized Discounted Cumulative Gain)."""
    if len(y_true) == 0 or sum(y_true) == 0:
        return 0.0
    
    # Handle "all" case
    if k == "all":
        k = len(y_true)
    
    # Get top k indices
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    
    # Calculate DCG@K
    dcg = 0.0
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG@K (ideal DCG)
    # Convert sum to int to avoid numpy.float64 in range()
    num_relevant = int(min(k, sum(y_true)))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_precision_recall_f1_at_k(y_true, y_scores, k):
    """
    Calculate Precision, Recall, and F1 at rank K with proper TP/FP/FN handling.
    
    This prevents models from getting high scores by predicting everything as positive.
    """
    # Handle "all" case
    if k == "all":
        k = len(y_true)
    
    sorted_indices = np.argsort(y_scores)[::-1]
    top_k_indices = sorted_indices[:k]
    
    # Calculate TP, FP, FN
    predicted_positive = set(top_k_indices)
    actual_positive = set(np.where(y_true == 1)[0])
    
    tp = len(predicted_positive & actual_positive)  # True Positives
    fp = len(predicted_positive - actual_positive)  # False Positives  
    fn = len(actual_positive - predicted_positive)  # False Negatives
    
    # Calculate metrics with proper TP/FP/FN
    precision_k = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_k = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
    
    return precision_k, recall_k, f1_k

def calculate_overall_accuracy_for_pairs(pair_scores: np.ndarray, 
                                       row_sentence_pairs: List[Tuple[int, int]],
                                       num_rows: int, 
                                       num_sentences: int) -> float:
    """
    Calculate F1-based accuracy for row-sentence pairs (NO RANKING ASSUMPTION).
    
    Since all ground truth pairs are equally valid (no ranking), this treats it as
    a multi-label classification problem using optimal threshold to separate
    ground truth pairs from non-ground truth pairs.
    
    Args:
        pair_scores: N×M matrix of row-sentence similarities
        row_sentence_pairs: List of (row_idx, sent_idx) ground truth pairs
        num_rows: Number of rows
        num_sentences: Number of sentences
    
    Returns:
        F1 score using optimal threshold (better training feedback than strict accuracy)
    """
    if not row_sentence_pairs or pair_scores.size == 0:
        return 0.0
    
    # Filter to in-bounds pairs only
    valid_pairs = [(i, j) for i, j in row_sentence_pairs
                   if 0 <= i < num_rows and 0 <= j < num_sentences]
    if not valid_pairs:
        return 0.0
    
    # Create ground truth set and binary labels
    gt_pairs = set(valid_pairs)
    
    # Flatten scores and create binary labels for ALL possible pairs
    flat_scores = pair_scores.flatten()
    all_pairs = [(i, j) for i in range(num_rows) for j in range(num_sentences)]
    y_true = np.array([1 if pair in gt_pairs else 0 for pair in all_pairs])
    
    # Find optimal threshold using ground truth scores
    gt_scores = [pair_scores[i, j] for i, j in valid_pairs]
    non_gt_scores = [pair_scores[i, j] for i in range(num_rows) for j in range(num_sentences) 
                     if (i, j) not in gt_pairs]
    
    if gt_scores and non_gt_scores:
        # Use threshold between mean of GT scores and mean of non-GT scores
        threshold = (np.mean(gt_scores) + np.mean(non_gt_scores)) / 2.0
    else:
        # Fallback to median of all scores
        threshold = np.median(flat_scores)
    
    # Predict based on threshold
    y_pred = (flat_scores >= threshold).astype(int)
    
    # Calculate F1 score (balances precision and recall)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
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
    
    # Filter to in-bounds pairs only
    valid_pairs = [(i, j) for i, j in row_sentence_pairs
                   if 0 <= i < num_rows and 0 <= j < num_sentences]
    if not valid_pairs:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'gt_score_mean': 0.0,
            'non_gt_score_mean': 0.0,
            'score_separation': 0.0
        }
    
    # Create ground truth set and binary labels
    gt_pairs = set(valid_pairs)
    
    # Get scores for ground truth and non-ground truth pairs
    gt_scores = [pair_scores[i, j] for i, j in valid_pairs]
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
    
    # Create binary predictions
    flat_scores = pair_scores.flatten()
    all_pairs = [(i, j) for i in range(num_rows) for j in range(num_sentences)]
    y_true = np.array([1 if pair in gt_pairs else 0 for pair in all_pairs])
    y_pred = (flat_scores >= threshold).astype(int)
    
    # Calculate precision, recall, F1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
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

def calculate_dynamic_threshold_metrics(y_true, y_scores):
    """
    Calculate F1, Precision, Recall using dynamic threshold: (max + min) / 2
    
    This addresses the issue where F1@all was always the same across models
    by using each model's own confidence distribution to set the threshold.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Model confidence scores
    
    Returns:
        Dictionary with dynamic_f1, dynamic_precision, dynamic_recall, threshold_used
    """
    if len(y_true) == 0 or len(y_scores) == 0:
        return {
            'dynamic_f1': 0.0,
            'dynamic_precision': 0.0, 
            'dynamic_recall': 0.0,
            'threshold_used': 0.0,
            'num_predictions': 0
        }
    
    # Calculate dynamic threshold: midpoint between max and min scores
    min_score = np.min(y_scores)
    max_score = np.max(y_scores)
    dynamic_threshold = (max_score + min_score) / 2.0
    
    # Make binary predictions based on dynamic threshold
    y_pred = (y_scores >= dynamic_threshold).astype(int)
    
    # Calculate TP, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    dynamic_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    dynamic_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dynamic_f1 = 2 * (dynamic_precision * dynamic_recall) / (dynamic_precision + dynamic_recall) if (dynamic_precision + dynamic_recall) > 0 else 0.0
    
    return {
        'dynamic_f1': dynamic_f1,
        'dynamic_precision': dynamic_precision,
        'dynamic_recall': dynamic_recall,
        'threshold_used': dynamic_threshold,
        'num_predictions': int(np.sum(y_pred)),
        'score_range': max_score - min_score
    }

def calculate_dynamic_binary_accuracy(y_true, y_scores):
    """
    Calculate binary accuracy using dynamic threshold.
    
    This is different from F1-based Overall Accuracy:
    - F1-based Overall Accuracy: threshold-based, uses optimal threshold for pair-level classification
    - Dynamic Binary Accuracy: threshold-based, requires correct binary classification
    """
    if len(y_true) == 0:
        return 0.0
    
    # Calculate dynamic threshold
    min_score = np.min(y_scores)
    max_score = np.max(y_scores)
    dynamic_threshold = (max_score + min_score) / 2.0
    
    # Make binary predictions
    y_pred = (y_scores >= dynamic_threshold).astype(int)
    
    # Calculate accuracy
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    
    return correct_predictions / total_predictions

def calculate_pair_level_tp_fp_fn(pair_scores: np.ndarray, 
                                  row_sentence_pairs: List[Tuple[int, int]],
                                  num_rows: int, 
                                  num_sentences: int,
                                  threshold: float = None,
                                  debug_model_name: str = None) -> Dict[str, int]:
    """
    FIXED: Calculate TP/FP/FN correctly for pair-level evaluation.
    
    Previous version calculated FP across ALL possible pairs, leading to mathematically 
    impossible results (TP+FN >> total ground truth pairs).
    
    New version uses top-k evaluation approach: only count FP among top-predicted pairs.
    
    Args:
        pair_scores: N×M matrix of row-sentence similarities
        row_sentence_pairs: List of (row_idx, sent_idx) ground truth pairs  
        num_rows: Number of rows
        num_sentences: Number of sentences
        threshold: Threshold for predictions (if None, uses dynamic approach)
        debug_model_name: Model name for debugging
    
    Returns:
        Dictionary with tp, fp, fn, total_ground_truth_pairs
    """
    if not row_sentence_pairs or pair_scores.size == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'total_ground_truth_pairs': 0
        }
    
    # Filter to in-bounds pairs, then create ground truth set
    valid_pairs = [(i, j) for i, j in row_sentence_pairs
                   if 0 <= i < num_rows and 0 <= j < num_sentences]
    gt_pairs = set(valid_pairs)
    num_gt_pairs = len(valid_pairs)
    
    # Flatten scores and get all pair positions
    flat_scores = pair_scores.flatten()
    all_pairs = [(i, j) for i in range(num_rows) for j in range(num_sentences)]
    
    # Sort pairs by score (descending) 
    sorted_indices = np.argsort(flat_scores)[::-1]
    sorted_pairs = [all_pairs[idx] for idx in sorted_indices]
    
    # FIXED APPROACH: Use adaptive top-k evaluation
    # Take top N predictions where N is reasonable (e.g., equal to ground truth pairs)
    k = max(num_gt_pairs, min(num_gt_pairs, len(all_pairs)))
    top_k_pairs = sorted_pairs[:k]
    
    # Calculate TP and FP among top-k predictions
    tp = 0
    fp = 0
    for pair in top_k_pairs:
        if pair in gt_pairs:
            tp += 1
        else:
            fp += 1
    
    # FN = ground truth pairs not found in top-k
    fn = num_gt_pairs - tp
    
    # Calculate threshold for reference (but not used in fixed evaluation)
    if threshold is None:
        min_score = np.min(pair_scores)
        max_score = np.max(pair_scores)
        threshold = (max_score + min_score) / 2.0
    
    # DEBUG: Check for suspicious results
    if debug_model_name and (tp + fn != num_gt_pairs):
        print(f"🚨 DEBUG: TP/FN math error for {debug_model_name}")
        print(f"   TP={tp}, FN={fn}, Total GT={num_gt_pairs}")
        print(f"   TP+FN={tp+fn} (should equal {num_gt_pairs})")
    
    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'total_ground_truth_pairs': num_gt_pairs,
        'threshold_used': threshold,
        'k_used': k
    }

def calculate_row_sentence_pair_metrics(pair_scores: np.ndarray, 
                                      row_sentence_pairs: List[Tuple[int, int]],
                                      num_rows: int, 
                                      num_sentences: int) -> Dict[str, float]:
    """
    Calculate metrics for row-sentence pair prediction.
    
    Args:
        pair_scores: N×M matrix of row-sentence similarities
        row_sentence_pairs: List of (row_idx, sent_idx) ground truth pairs
        num_rows: Number of rows
        num_sentences: Number of sentences
    
    Returns:
        Dictionary with pair-level metrics
    """
    if not row_sentence_pairs or pair_scores.size == 0:
        return {
            'pair_precision_at_k': [0.0, 0.0, 0.0],  # @1, @3, @5
            'pair_recall_at_k': [0.0, 0.0, 0.0],
            'pair_f1_at_k': [0.0, 0.0, 0.0],
            'pair_roc_auc': 0.0,
            'pair_avg_precision': 0.0,
            'pair_mean_rank': float('inf')
        }
    
    # Create binary ground truth matrix
    gt_matrix = np.zeros((num_rows, num_sentences), dtype=int)
    for row_idx, sent_idx in row_sentence_pairs:
        if 0 <= row_idx < num_rows and 0 <= sent_idx < num_sentences:
            gt_matrix[row_idx, sent_idx] = 1
    
    # Flatten for binary classification metrics
    gt_flat = gt_matrix.flatten()
    scores_flat = pair_scores.flatten()
    
    # Calculate ROC-AUC and Average Precision
    try:
        pair_roc_auc = roc_auc_score(gt_flat, scores_flat) if len(np.unique(gt_flat)) > 1 else 0.0
    except ValueError:
        pair_roc_auc = 0.0
    
    try:
        pair_avg_precision = average_precision_score(gt_flat, scores_flat) if len(np.unique(gt_flat)) > 1 else 0.0
    except ValueError:
        pair_avg_precision = 0.0
    
    # Calculate ranking metrics for true pairs
    pair_ranks = []
    all_pairs = [(i, j, pair_scores[i, j]) for i in range(num_rows) for j in range(num_sentences)]
    all_pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by score, descending
    
    for row_idx, sent_idx in row_sentence_pairs:
        if 0 <= row_idx < num_rows and 0 <= sent_idx < num_sentences:
            # Find rank of this pair
            for rank, (r, s, score) in enumerate(all_pairs, 1):
                if r == row_idx and s == sent_idx:
                    pair_ranks.append(rank)
                    break
    
    pair_mean_rank = np.mean(pair_ranks) if pair_ranks else float('inf')
    
    # Calculate Precision@K, Recall@K, F1@K for pairs
    pair_precision_at_k = []
    pair_recall_at_k = []
    pair_f1_at_k = []
    
    for k in [1, 3, 5]:
        top_k_pairs = all_pairs[:k]
        predicted_pairs = set((r, s) for r, s, score in top_k_pairs)
        true_pairs = set(row_sentence_pairs)
        
        tp = len(predicted_pairs & true_pairs)
        precision_k = tp / k if k > 0 else 0.0
        recall_k = tp / len(true_pairs) if true_pairs else 0.0
        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
        
        pair_precision_at_k.append(precision_k)
        pair_recall_at_k.append(recall_k)
        pair_f1_at_k.append(f1_k)
    
    return {
        'pair_precision_at_k': pair_precision_at_k,
        'pair_recall_at_k': pair_recall_at_k,
        'pair_f1_at_k': pair_f1_at_k,
        'pair_roc_auc': pair_roc_auc,
        'pair_avg_precision': pair_avg_precision,
        'pair_mean_rank': pair_mean_rank
    }

def comprehensive_evaluation_single_model(model, examples, annotations, model_name, config, batch_size=1, show_examples=False):
    """
    Comprehensive evaluation combining IR metrics and row-sentence pair analysis for Protrix data.
    
    This evaluation uses the precise row-sentence annotations instead of just row indices.
    Enhanced with proper accuracy metrics, F1 score prominence, and Average Precision as main metric.
    """
    print(f"\n🔍 Comprehensive evaluation of {model_name} model...")
    print("📋 Using Protrix row-sentence pair annotations")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Detect model dtype for consistent evaluation
    try:
        model_dtype = next(model.parameters()).dtype
        print(f"🔧 Model dtype detected: {model_dtype}")
    except:
        model_dtype = torch.bfloat16
        print("🔧 Could not detect model dtype, defaulting to bfloat16")
    
    # Results storage
    all_results = []
    examples_processed = 0
    
    # Standard IR metrics storage (row-level) - Enhanced with "all"
    all_precision_1, all_recall_1, all_f1_1, all_ndcg_1 = [], [], [], []
    all_precision_3, all_recall_3, all_f1_3, all_ndcg_3 = [], [], [], []
    all_precision_5, all_recall_5, all_f1_5, all_ndcg_5 = [], [], [], []
    all_precision_10, all_recall_10, all_f1_10, all_ndcg_10 = [], [], [], []  # NEW: K=10 metrics
    all_precision_all, all_recall_all, all_f1_all, all_ndcg_all = [], [], [], []  # NEW: "all" metrics
    
    # Enhanced accuracy tracking - UPDATED: Remove top-1, focus on overall
    all_overall_accuracy = []  # Perfect accuracy: ALL ground truths in top positions (renamed from perfect_accuracy)
    
    # NEW: Dynamic threshold-based metrics storage
    all_dynamic_f1 = []
    all_dynamic_precision = []
    all_dynamic_recall = []
    all_dynamic_binary_accuracy = []
    all_dynamic_thresholds = []
    
    # Row-sentence pair metrics storage
    all_pair_metrics = {
        'precision_at_k': [[], [], []],  # @1, @3, @5
        'recall_at_k': [[], [], []],
        'f1_at_k': [[], [], []],
        'roc_auc': [],
        'avg_precision': [],
        'mean_rank': []
    }
    
    # Row-level metrics storage for backward compatibility and visualizations
    all_binary_labels = []
    pair_scores_list = []
    all_pair_scores_data = []  # For visualization: (example_idx, row_idx, score, is_highlighted)
    
    # For TP/FP/FN visualization
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_ground_truth_positives = 0


    with torch.no_grad():
        for example_idx, example in enumerate(tqdm(examples, desc=f"Evaluating {model_name}")):
            anchor_id = example.get("anchor_id")
            
            if anchor_id is None or anchor_id not in annotations:
                continue
            
            highlighted_cells = annotations[anchor_id]
            
            # Skip examples without annotations for now (could be included as negative examples)
            if not highlighted_cells:
                continue
            
            # Extract table rows
            anchor_rows = example.get("anchor_rows", [])
            rows = []
            for row in anchor_rows:
                if isinstance(row, dict):
                    formatted_text = row.get("formatted", "")
                    if formatted_text:
                        rows.append(formatted_text)
                elif isinstance(row, str) and row:
                    rows.append(row)
            
            if not rows:
                continue
            
            examples_processed += 1
            
            # Encode rows using the same method as training
            row_embeddings = model.encode_sentences(rows, batch_size=batch_size)
            row_tensor = row_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
            
            # Process primary positive
            primary_positive = example.get("primary_positive", {})
            if primary_positive.get("id") is not None:
                positive_sentences = primary_positive.get("sentences", [])
                
                if positive_sentences:
                    # Encode positive sentences
                    positive_embeddings = model.encode_sentences(positive_sentences, batch_size=batch_size)
                    positive_tensor = positive_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
                    
                    if hasattr(model, 'bidirectional_attention'):
                        # Bidirectional model - get pair scores directly
                        # Use the SAME aggregation method as training (from config)
                        aggregation_method = config.get("aggregation_method", "top_k_pairs")
                        if example_idx == 0:  # Print once for verification
                            print(f"🔧 Using aggregation method: {aggregation_method} (from training config)")
                        global_similarity, pair_scores = model(
                            row_tensor, positive_tensor, aggregation_method=aggregation_method
                        )
                        pair_scores_np = safe_tensor_to_numpy(pair_scores.squeeze(0))  # [N_rows, M_sentences]
                    else:
                        # FIXED: For unidirectional model, use CONTEXTUALIZED embeddings
                        # The get_contextualized_pair_scores() method computes:
                        # 1. Contextualized row embeddings (after cross-attention + FFN)
                        # 2. Cosine similarity between contextualized rows and original sentences
                        pair_scores = model.get_contextualized_pair_scores(row_tensor, positive_tensor)
                        pair_scores_np = safe_tensor_to_numpy(pair_scores.squeeze(0).detach())  # [N_rows, M_sentences]
                    
                    # Extract ground truth
                    highlighted_row_indices, row_sentence_pairs = extract_row_sentence_pairs(highlighted_cells)
                    
                    # Calculate row-level metrics (for backward compatibility)
                    row_scores = np.max(pair_scores_np, axis=1)  # Max score per row
                    y_true_rows = np.zeros(len(rows), dtype=int)  # Use int dtype to avoid float64 issues
                    for row_idx in highlighted_row_indices:
                        if 0 <= row_idx < len(rows):
                            y_true_rows[row_idx] = 1
                    
                    # Calculate standard IR metrics at different K values + "all"
                    k_values = [1, 3, 5, 10, "all"]
                    metric_lists = [
                        (all_precision_1, all_recall_1, all_f1_1, all_ndcg_1),
                        (all_precision_3, all_recall_3, all_f1_3, all_ndcg_3),
                        (all_precision_5, all_recall_5, all_f1_5, all_ndcg_5),
                        (all_precision_10, all_recall_10, all_f1_10, all_ndcg_10),
                        (all_precision_all, all_recall_all, all_f1_all, all_ndcg_all)
                    ]
                    
                    for k, (prec_list, rec_list, f1_list, ndcg_list) in zip(k_values, metric_lists):
                        prec_k, rec_k, f1_k = calculate_precision_recall_f1_at_k(y_true_rows, row_scores, k)
                        ndcg_k = calculate_ndcg_at_k(y_true_rows, row_scores, k)
                        
                        prec_list.append(prec_k)
                        rec_list.append(rec_k)
                        f1_list.append(f1_k)
                        ndcg_list.append(ndcg_k)
                    
                    # UPDATED: Use pair-level F1-based accuracy (homogenized with training script)
                    overall_accuracy = calculate_overall_accuracy_for_pairs(
                        pair_scores_np, row_sentence_pairs, len(rows), len(positive_sentences)
                    )
                    all_overall_accuracy.append(overall_accuracy)
                    
                    # NEW: Calculate dynamic threshold-based metrics
                    dynamic_metrics = calculate_dynamic_threshold_metrics(y_true_rows, row_scores)
                    dynamic_binary_accuracy = calculate_dynamic_binary_accuracy(y_true_rows, row_scores)
                    
                    # Store dynamic metrics
                    all_dynamic_f1.append(dynamic_metrics['dynamic_f1'])
                    all_dynamic_precision.append(dynamic_metrics['dynamic_precision'])
                    all_dynamic_recall.append(dynamic_metrics['dynamic_recall'])
                    all_dynamic_binary_accuracy.append(dynamic_binary_accuracy)
                    all_dynamic_thresholds.append(dynamic_metrics['threshold_used'])

                    # FIXED: Calculate pair-level TP/FP/FN (not row-level)
                    pair_level_counts = calculate_pair_level_tp_fp_fn(
                        pair_scores_np, row_sentence_pairs, len(rows), len(positive_sentences),
                        debug_model_name=model_name
                    )
                    
                    total_tp += pair_level_counts['tp']
                    total_fp += pair_level_counts['fp'] 
                    total_fn += pair_level_counts['fn']
                    total_ground_truth_positives += pair_level_counts['total_ground_truth_pairs']
                    
                    # Keep row-level calculations for backward compatibility
                    min_score = np.min(row_scores)
                    max_score = np.max(row_scores)
                    dynamic_threshold = (max_score + min_score) / 2.0
                    y_pred_rows = (row_scores >= dynamic_threshold).astype(int)

                    
                    # Calculate row-sentence pair metrics (NEW for Protrix)
                    pair_metrics = calculate_row_sentence_pair_metrics(
                        pair_scores_np, row_sentence_pairs, len(rows), len(positive_sentences)
                    )
                    
                    # Store pair metrics
                    for k_idx in range(3):  # @1, @3, @5
                        all_pair_metrics['precision_at_k'][k_idx].append(pair_metrics['pair_precision_at_k'][k_idx])
                        all_pair_metrics['recall_at_k'][k_idx].append(pair_metrics['pair_recall_at_k'][k_idx])
                        all_pair_metrics['f1_at_k'][k_idx].append(pair_metrics['pair_f1_at_k'][k_idx])
                    
                    all_pair_metrics['roc_auc'].append(pair_metrics['pair_roc_auc'])
                    all_pair_metrics['avg_precision'].append(pair_metrics['pair_avg_precision'])
                    all_pair_metrics['mean_rank'].append(pair_metrics['pair_mean_rank'])
                    
                    # Store for backward compatibility and visualizations
                    for row_idx, (score, label) in enumerate(zip(row_scores, y_true_rows)):
                        all_binary_labels.append(label)
                        pair_scores_list.append(score)
                        all_pair_scores_data.append((example_idx, row_idx, score, bool(label)))
                    
                    # Store detailed results
                    sorted_indices = np.argsort(row_scores)[::-1]
                    example_result = {
                        'example_idx': example_idx,
                        'anchor_id': anchor_id,
                        'num_rows': len(rows),
                        'num_sentences': len(positive_sentences),
                        'ground_truth_row_indices': highlighted_row_indices,
                        'ground_truth_row_sentence_pairs': row_sentence_pairs,
                        'row_scores': row_scores.tolist(),
                        'sorted_predictions': sorted_indices.tolist(),
                        'global_similarity': safe_tensor_to_scalar(global_similarity),
                        'binary_labels': y_true_rows.tolist(),
                        'pair_metrics': pair_metrics,
                        'overall_accuracy': overall_accuracy  # UPDATED: Only overall accuracy
                    }
                    
                    all_results.append(example_result)
                    
                    # Print detailed example (for first few examples if show_examples is True)
                    if show_examples and example_idx < 3:
                        print(f"\n" + "="*50)
                        print(f"📊 {model_name.upper()} - EXAMPLE {anchor_id}")
                        print(f"="*50)
                        
                        print(f"🎯 GROUND TRUTH PAIRS: {row_sentence_pairs}")
                        print(f"📋 ROWS: {len(rows)}, SENTENCES: {len(positive_sentences)}")
                        
                        print(f"🔍 ROW-SENTENCE PAIR SCORES (top 5):")
                        # Show top scoring pairs
                        all_pairs = [(i, j, pair_scores_np[i, j]) for i in range(len(rows)) for j in range(len(positive_sentences))]
                        all_pairs.sort(key=lambda x: x[2], reverse=True)
                        for rank, (r, s, score) in enumerate(all_pairs[:5], 1):
                            is_correct = "✅" if (r, s) in row_sentence_pairs else "❌"
                            print(f"  Rank {rank}: Row {r+1} ↔ Sentence {s} (score: {score:.3f}) {is_correct}")
                        
                        print(f"📊 ACCURACY METRICS:")
                        print(f"  Overall Accuracy (all GT found): {overall_accuracy:.3f}")
    
    # Calculate overall metrics
    if not all_results:
        print("❌ No valid examples found")
        return {}
    
    def get_metrics_at_k(prec_list, rec_list, f1_list, ndcg_list):
        return {
            'precision': np.mean(prec_list),
            'recall': np.mean(rec_list),
            'f1': np.mean(f1_list),
            'ndcg': np.mean(ndcg_list),
            'precision_std': np.std(prec_list),
            'recall_std': np.std(rec_list),
            'f1_std': np.std(f1_list),
            'ndcg_std': np.std(ndcg_list)
        }
    
    # Calculate IR metrics for each K + "all"
    metrics_1 = get_metrics_at_k(all_precision_1, all_recall_1, all_f1_1, all_ndcg_1)
    metrics_3 = get_metrics_at_k(all_precision_3, all_recall_3, all_f1_3, all_ndcg_3)
    metrics_5 = get_metrics_at_k(all_precision_5, all_recall_5, all_f1_5, all_ndcg_5)
    metrics_10 = get_metrics_at_k(all_precision_10, all_recall_10, all_f1_10, all_ndcg_10)
    metrics_all = get_metrics_at_k(all_precision_all, all_recall_all, all_f1_all, all_ndcg_all)
    
    # Enhanced accuracy calculations - UPDATED: Only overall accuracy
    overall_accuracy_final = np.mean(all_overall_accuracy)
    
    # NEW: Calculate average dynamic threshold-based metrics
    dynamic_f1_final = np.mean(all_dynamic_f1) if all_dynamic_f1 else 0.0
    dynamic_precision_final = np.mean(all_dynamic_precision) if all_dynamic_precision else 0.0
    dynamic_recall_final = np.mean(all_dynamic_recall) if all_dynamic_recall else 0.0
    dynamic_binary_accuracy_final = np.mean(all_dynamic_binary_accuracy) if all_dynamic_binary_accuracy else 0.0
    average_threshold_used = np.mean(all_dynamic_thresholds) if all_dynamic_thresholds else 0.0
    
    # Calculate row-level pair metrics (backward compatibility)
    all_binary_labels = np.array(all_binary_labels)
    pair_scores_array = np.array(pair_scores_list)
    
    try:
        roc_auc = roc_auc_score(all_binary_labels, pair_scores_array) if len(np.unique(all_binary_labels)) > 1 else 0.0
    except ValueError:
        roc_auc = 0.0
    
    # Calculate pair-level metrics (NEW for Protrix)
    pair_level_metrics = {}
    for k_idx, k in enumerate([1, 3, 5]):
        pair_level_metrics[f'pair_precision_at_{k}'] = np.mean(all_pair_metrics['precision_at_k'][k_idx])
        pair_level_metrics[f'pair_recall_at_{k}'] = np.mean(all_pair_metrics['recall_at_k'][k_idx])
        pair_level_metrics[f'pair_f1_at_{k}'] = np.mean(all_pair_metrics['f1_at_k'][k_idx])
    
    pair_level_metrics['pair_roc_auc'] = np.mean(all_pair_metrics['roc_auc'])
    pair_level_metrics['pair_avg_precision'] = np.mean(all_pair_metrics['avg_precision'])
    pair_level_metrics['pair_mean_rank'] = np.mean(all_pair_metrics['mean_rank'])
    
    # UPDATED: Use pair-level average precision (homogenized with training script)
    avg_precision = pair_level_metrics['pair_avg_precision']
    
    # UPDATED: New result summary emphasizing Avg Precision first, F1 second, Overall Accuracy third
    print()
    print(f"📊 {model_name.upper()} RESULTS:")
    print(f"   ⭐ Average Precision: {avg_precision:.3f} (PRIMARY METRIC)")
    print(f"   📈 Dynamic F1: {dynamic_f1_final:.3f} (SECONDARY METRIC)")
    print(f"   🎯 Overall Accuracy (F1-based): {overall_accuracy_final:.3f} (Pair-level F1)")
    print(f"   📋 NDCG@1: {metrics_1['ndcg']:.3f}, NDCG@3: {metrics_3['ndcg']:.3f}, NDCG@10: {metrics_10['ndcg']:.3f}")
    print(f"   🔗 Pair-level Avg Precision: {pair_level_metrics['pair_avg_precision']:.3f}")
    print(f"   📊 Pair-level Mean Rank: {pair_level_metrics['pair_mean_rank']:.1f}")
    
    return {
        'model_name': model_name,
        'examples_processed': examples_processed,
        'overall_accuracy': overall_accuracy_final,  # UPDATED: Only overall accuracy
        'metrics_at_1': metrics_1,
        'metrics_at_3': metrics_3,
        'metrics_at_5': metrics_5,
        'metrics_at_10': metrics_10,
        'metrics_at_all': metrics_all,  # NEW: "all" metrics
        'roc_auc': roc_auc,
        'average_precision': avg_precision,  # PRIMARY METRIC
        'f1_score_at_5': metrics_5['f1'],  # NEW: Prominent F1 score
        'f1_score_at_all': metrics_all['f1'],  # NEW: Prominent F1 score
        # NEW: Dynamic threshold-based metrics
        'dynamic_f1': dynamic_f1_final,  # SECONDARY METRIC
        'dynamic_precision': dynamic_precision_final,
        'dynamic_recall': dynamic_recall_final,
        'dynamic_binary_accuracy': dynamic_binary_accuracy_final,
        'average_threshold_used': average_threshold_used,
        'pair_level_metrics': pair_level_metrics,
        'pair_scores_data': all_pair_scores_data,  # For visualizations
        'detailed_results': all_results,
        # Add backward compatibility fields
        'mean_highlighted_rank': pair_level_metrics['pair_mean_rank'],
        'prediction_breakdown': {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'total_ground_truth_positives': total_ground_truth_positives
        }
    }








def load_training_config(config_path="./output_cross_attention_cache/test_model/training_config.json"):
    """Load training configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded training config from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️ Training config not found at {config_path}")
        # Return default config
        return {
            "sentence_encoder": "all-roberta-large-v1",
            "embedding_dim": None,  # Will be determined from sentence encoder
            "use_bidirectional": True,
            "use_cross_attention_lora": True,
            "lora_rank": 16,
            "trainable_encoder": True
        }

def create_simple_model(config, device="cuda", model_name="all-roberta-large-v1", 
                        use_unsloth=False, use_compile=True):
    """Create a simple cross-attention model for stage 1 evaluation.
    
    Args:
        config: Training configuration dictionary
        device: Device to use (cuda/cpu)
        model_name: Name of the sentence encoder model
        use_unsloth: Whether to use Unsloth for faster encoder loading
        use_compile: Whether to apply torch.compile for faster inference
    """
    # Load encoder - optionally with Unsloth optimizations
    if use_unsloth and UNSLOTH_AVAILABLE:
        print("🦥 Using Unsloth-optimized encoder for evaluation")
        sentence_encoder = create_unsloth_sentence_encoder(
            model_name=model_name,
            device=device,
            use_unsloth=True,
            load_in_4bit=False,  # Full precision for evaluation
            dtype=torch.bfloat16,
        )
    else:
        sentence_encoder = SentenceTransformer(
            model_name, 
            model_kwargs={"dtype": torch.bfloat16},
            trust_remote_code=True, 
            device=device
        )
    
    # Extract initialization parameters from training config
    init_method = config.get("init_method", "xavier_uniform")
    init_method_params = config.get("init_method_params", {})
    
    print(f"🔧 Creating Stage 1 simple model with initialization: {init_method}")
    if init_method_params:
        print(f"   Init parameters: {init_method_params}")
    
    # Get embedding dimension from sentence encoder
    embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
    
    model = TableTextEmbeddingModel(
        sentence_encoder=sentence_encoder,
        embedding_dim=embedding_dim,
        trainable_encoder=False,  # ✅ FIXED: Always use frozen encoder
        use_cross_attention_lora=False,  # Simple model without LoRA
        top_k=config.get("top_k", 3),
        # ✅ FIX: Use initialization method from training config
        init_method=init_method,
        init_method_params=init_method_params,
        norm_type=config.get("norm_type", "layernorm")
    )
    
    model.to(device)
    
    # Apply torch.compile for faster inference
    if use_compile and TORCH_COMPILE_AVAILABLE:
        print("⚡ Applying torch.compile() for faster evaluation")
        model = optimize_model_for_inference(model, use_compile=True, compile_mode="reduce-overhead")
    
    return model

def create_sophisticated_model(config, device="cuda", model_name="all-roberta-large-v1",
                               use_unsloth=False, use_compile=True):
    """Create a sophisticated bidirectional model for stage 2/3 evaluation using DYNAMIC config loading.
    
    Args:
        config: Training configuration dictionary
        device: Device to use (cuda/cpu)
        model_name: Name of the sentence encoder model
        use_unsloth: Whether to use Unsloth for faster encoder loading
        use_compile: Whether to apply torch.compile for faster inference
    """
    # Load encoder - optionally with Unsloth optimizations
    if use_unsloth and UNSLOTH_AVAILABLE:
        print("🦥 Using Unsloth-optimized encoder for evaluation")
        sentence_encoder = create_unsloth_sentence_encoder(
            model_name=model_name,
            device=device,
            use_unsloth=True,
            load_in_4bit=False,  # Full precision for evaluation
            dtype=torch.bfloat16,
        )
    else:
        sentence_encoder = SentenceTransformer(
            model_name, 
            model_kwargs={"dtype": torch.bfloat16},
            trust_remote_code=True, 
            device=device
        )
    
    # Load ALL parameters dynamically from config file - no hard-coded defaults
    # This ensures the model matches exactly what was used during training
    print(f"🔧 Creating model with dynamic config loading...")
    print(f"   - Model architecture: {config.get('architecture', 'bidirectional')}")
    print(f"   - Attention type: {config.get('attention_type', 'standard')}")
    print(f"   - Initialization method: {config.get('init_method', 'xavier_uniform')}")
    print(f"   - Use LoRA: {config.get('use_cross_attention_lora', False)}")
    print(f"   - Top-K: {config.get('top_k', 3)}")
    
    # Get embedding dimension from sentence encoder
    embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
    
    model = BidirectionalTableTextModel(
        sentence_encoder=sentence_encoder,
        # Core architecture parameters
        embedding_dim=embedding_dim,
        trainable_encoder=False,  # ✅ FIXED: Always use frozen encoder
        
        # LoRA parameters
        use_cross_attention_lora=config.get("use_cross_attention_lora", False),
        lora_rank=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32.0),
        lora_dropout=config.get("lora_dropout", 0.1),
        
        # Attention parameters
        top_k=config.get("top_k", 3),
        pair_score_method=config.get("pair_score_method", "cosine"),
        share_weights=config.get("share_attention_weights", False),
        
        # Advanced architecture parameters - ENABLE self-attention to prevent collapse
        use_refinement=config.get("use_refinement", True),
        use_self_attention=config.get("use_self_attention", True),  # Changed from False
        self_attention_heads=config.get("self_attention_heads", 8),
        self_attention_dropout=config.get("self_attention_dropout", 0.1),
        
        # ✅ FIX: Use initialization parameters directly from training config (no hardcoded fallbacks)
        init_method=config.get("init_method", "xavier_uniform"),  # Use training config value
        init_method_params=config.get("init_method_params", {}),  # Use training config params
        
        # Attention mechanism parameters
        attention_type=config.get("attention_type", "standard"),
        sparse_top_k=config.get("sparse_top_k", 3),
        window_size=config.get("window_size", 5),
        threshold_base=config.get("threshold_base", 0.1),
        norm_type=config.get("norm_type", "layernorm"),
        use_qk_rmsnorm=config.get("use_qk_rmsnorm", True)
    )
    # Propagate activation/ranking switches so evaluation mirrors training behavior
    setattr(model, 'attention_activation', config.get('attention_activation', 'softmax'))
    setattr(model, 'attention_alpha', config.get('attention_alpha', 1.5))
    setattr(model, 'ranking_loss_type', config.get('ranking_loss_type', 'softplus'))
    setattr(model, 'infonce_tau', config.get('infonce_tau', 0.7))
    setattr(model, 'pair_topk_mask', config.get('pair_topk_mask', False))
    setattr(model, 'pair_topk_k', config.get('pair_topk_k', 0))
    
    model.to(device)
    
    # Apply torch.compile for faster inference
    if use_compile and TORCH_COMPILE_AVAILABLE:
        print("⚡ Applying torch.compile() for faster evaluation")
        model = optimize_model_for_inference(model, use_compile=True, compile_mode="reduce-overhead")
    
    return model

def load_trained_model(config, device="cuda", model_name="all-roberta-large-v1", 
                       model_path="./output_cross_attention_cache/test_model/all-roberta-large-v1_best.pt",
                       use_unsloth=False, use_compile=True):
    """Load the trained sophisticated model for stage 3 evaluation.
    
    Args:
        config: Training configuration dictionary
        device: Device to use (cuda/cpu)
        model_name: Name of the sentence encoder model
        model_path: Path to trained model weights
        use_unsloth: Whether to use Unsloth for faster encoder loading
        use_compile: Whether to apply torch.compile for faster inference
    """
    # Note: We don't apply compile here since we need to load weights first
    model = create_sophisticated_model(config, device, model_name, 
                                        use_unsloth=use_unsloth, use_compile=False)
    trained_weights_loaded = False
    
    # Try to load trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Try loading with strict=True first
        try:
            model.load_state_dict(checkpoint, strict=True)
            print(f"✅ Loaded trained model from: {model_path}")
            trained_weights_loaded = True
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"⚠️ State dict mismatch detected, attempting flexible loading...")
                
                # Try loading with strict=False to ignore mismatched keys
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                    if missing_keys:
                        print(f"⚠️ Missing keys: {len(missing_keys)} (will use initialized weights)")
                    if unexpected_keys:
                        print(f"⚠️ Unexpected keys: {len(unexpected_keys)} (will be ignored)")
                    print(f"✅ Partially loaded trained model from: {model_path}")
                    trained_weights_loaded = True
                except Exception as load_error:
                    print(f"❌ Failed to load trained weights: {load_error}")
                    print(f"🔄 Using untrained sophisticated model instead")
            else:
                raise e
                
    except FileNotFoundError:
        print(f"❌ CRITICAL: Trained model not found at {model_path}")
        print(f"🔄 Using untrained sophisticated model instead")
        print(f"⚠️ WARNING: Stage 3 will be identical to Stage 2!")
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        print(f"🔄 Using untrained sophisticated model instead")
        print(f"⚠️ WARNING: Stage 3 will be identical to Stage 2!")
    
    # Mark the model with loading status for later checks
    model._trained_weights_loaded = trained_weights_loaded
    
    # Apply torch.compile after weights are loaded
    if use_compile and TORCH_COMPILE_AVAILABLE:
        print("⚡ Applying torch.compile() for faster evaluation")
        model = optimize_model_for_inference(model, use_compile=True, compile_mode="reduce-overhead")
    
    return model

def evaluate_frozen_encoder_only(sentence_encoder, examples, annotations, batch_size=1, 
                                 init_method="xavier_uniform", init_method_params=None, test_cache=None):
    """
    Evaluate using frozen encoder baseline.
    Stage 0: Frozen encoder baseline.
    
    Enhanced with comprehensive metrics calculation like other stages.
    This provides a structurally comparable baseline that demonstrates
    the value of proper initialization and architectural sophistication.
    
    UPDATED: Now uses RandomCrossAttentionWrapper for fair structural comparison
    while maintaining performance to properly initialized models.
    
    Args:
        sentence_encoder: The frozen sentence encoder
        examples: List of test examples
        annotations: Dictionary mapping anchor_id to highlighted_cells
        batch_size: Batch size for evaluation
        init_method: Initialization method for the wrapper
        init_method_params: Parameters for initialization method
        test_cache: Optional test cache for efficient embedding lookup (avoids re-encoding)
    """
    print("🔥 Stage 0: Evaluating FROZEN ENCODER BASELINE...")
    
    # Create frozen encoder wrapper for fair structural baseline
    # Get embedding dimension from sentence encoder instead of hardcoding
    embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
    frozen_model = RandomCrossAttentionWrapper(sentence_encoder, embedding_dim, 
                                             init_method=init_method, 
                                             init_method_params=init_method_params)
    
    # Move to same device and dtype as sentence encoder
    encoder_param = next(sentence_encoder.parameters())
    device = encoder_param.device
    dtype = encoder_param.dtype
    frozen_model.to(device=device, dtype=dtype)
    
    # Detect model dtype for consistent evaluation
    try:
        model_dtype = next(frozen_model.parameters()).dtype
        print(f"🔧 Model dtype detected: {model_dtype}")
    except:
        model_dtype = torch.bfloat16
        print("🔧 Could not detect model dtype, defaulting to bfloat16")
    
    # Results storage - same structure as other stages
    all_results = []
    examples_processed = 0
    
    # Standard IR metrics storage - Enhanced with "all"
    all_precision_1, all_recall_1, all_f1_1, all_ndcg_1 = [], [], [], []
    all_precision_3, all_recall_3, all_f1_3, all_ndcg_3 = [], [], [], []
    all_precision_5, all_recall_5, all_f1_5, all_ndcg_5 = [], [], [], []
    all_precision_10, all_recall_10, all_f1_10, all_ndcg_10 = [], [], [], []  # NEW: K=10 metrics
    all_precision_all, all_recall_all, all_f1_all, all_ndcg_all = [], [], [], []  # NEW: "all" metrics
    
    # Enhanced accuracy tracking - UPDATED: Only overall accuracy
    all_overall_accuracy = []
    
    # NEW: Dynamic threshold-based metrics storage
    all_dynamic_f1 = []
    all_dynamic_precision = []
    all_dynamic_recall = []
    all_dynamic_binary_accuracy = []
    all_dynamic_thresholds = []
    
    # Row-sentence pair metrics storage
    all_pair_metrics = {
        'precision_at_k': [[], [], []],  # @1, @3, @5
        'recall_at_k': [[], [], []],
        'f1_at_k': [[], [], []],
        'roc_auc': [],
        'avg_precision': [],
        'mean_rank': []
    }
    
    # For visualization compatibility
    all_binary_labels = []
    pair_scores_list = []
    all_pair_scores_data = []

    # For TP/FP/FN visualization
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_ground_truth_positives = 0
    
    with torch.no_grad():
        for example_idx, example in enumerate(tqdm(examples, desc="Frozen encoder baseline evaluation")):
            anchor_id = example.get("anchor_id")
            
            if anchor_id is None or anchor_id not in annotations:
                continue
            
            highlighted_cells = annotations[anchor_id]
            if not highlighted_cells:
                continue
            
            # Extract table rows - IDENTICAL to other stages
            anchor_rows = example.get("anchor_rows", [])
            rows = []
            for row in anchor_rows:
                if isinstance(row, dict):
                    formatted_text = row.get("formatted", "")
                    if formatted_text:
                        rows.append(formatted_text)
                elif isinstance(row, str) and row:
                    rows.append(row)
            
            if not rows:
                continue
            
            examples_processed += 1
            
            # Get embeddings from cache or encode fresh
            if test_cache is not None:
                # Try to get cached table embeddings
                row_embeddings = test_cache.get_table_embeddings(anchor_id)
                if row_embeddings is None:
                    # Fallback to fresh encoding if not in cache
                    row_embeddings = frozen_model.encode_sentences(rows, batch_size=batch_size)
            else:
                # No cache - encode fresh (original behavior)
                row_embeddings = frozen_model.encode_sentences(rows, batch_size=batch_size)
            
            row_tensor = row_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
            
            # Extract sentences using the SAME method as training (data.py)
            primary_positive = example.get("primary_positive", {})
            if primary_positive.get("id") is None:
                continue
                
            positive_sentences = primary_positive.get("sentences", [])
            if not positive_sentences:
                continue
            
            # Get embeddings from cache or encode fresh
            if test_cache is not None:
                # For sentences, get from primary_positive context_id in cache
                context_id = primary_positive.get("id")
                sentence_embeddings = None
                if context_id is not None:
                    sentence_embeddings = test_cache.get_context_embeddings(context_id)
                
                if sentence_embeddings is None:
                    # Fallback to fresh encoding if not in cache
                    sentence_embeddings = frozen_model.encode_sentences(positive_sentences, batch_size=batch_size)
            else:
                # No cache - encode fresh (original behavior)
                sentence_embeddings = frozen_model.encode_sentences(positive_sentences, batch_size=batch_size)
            
            sentence_tensor = sentence_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
            
            # UPDATED: Use wrapper's forward method (same interface as cross-attention models)
            global_similarity, pair_scores = frozen_model(row_tensor, sentence_tensor, aggregation_method="top_k_pairs")
            
            # Convert to numpy and aggregate (same as cross-attention models)
            pair_scores_np = safe_tensor_to_numpy(pair_scores.squeeze(0))  # [N_rows, M_sentences]
            row_scores = np.max(pair_scores_np, axis=1)  # Take max per row
            
            # Extract ground truth
            highlighted_row_indices, row_sentence_pairs = extract_row_sentence_pairs(highlighted_cells)
            
            # Create binary labels (ground truth) - same as other stages
            y_true = np.zeros(len(rows), dtype=int)
            for highlighted_row_idx in highlighted_row_indices:
                if highlighted_row_idx < len(rows):
                    y_true[highlighted_row_idx] = 1
            
            # Calculate standard IR metrics at different K values + "all" - SAME AS OTHER STAGES
            k_values = [1, 3, 5, 10, "all"]
            metric_lists = [
                (all_precision_1, all_recall_1, all_f1_1, all_ndcg_1),
                (all_precision_3, all_recall_3, all_f1_3, all_ndcg_3),
                (all_precision_5, all_recall_5, all_f1_5, all_ndcg_5),
                (all_precision_10, all_recall_10, all_f1_10, all_ndcg_10),
                (all_precision_all, all_recall_all, all_f1_all, all_ndcg_all)
            ]
            
            for k, (prec_list, rec_list, f1_list, ndcg_list) in zip(k_values, metric_lists):
                prec_k, rec_k, f1_k = calculate_precision_recall_f1_at_k(y_true, row_scores, k)
                ndcg_k = calculate_ndcg_at_k(y_true, row_scores, k)
                
                prec_list.append(prec_k)
                rec_list.append(rec_k)
                f1_list.append(f1_k)
                ndcg_list.append(ndcg_k)
            
            # UPDATED: Use pair-level F1-based accuracy (homogenized with training script)
            overall_accuracy = calculate_overall_accuracy_for_pairs(
                pair_scores_np, row_sentence_pairs, len(rows), len(positive_sentences)
            )
            all_overall_accuracy.append(overall_accuracy)
            
            # NEW: Calculate dynamic threshold-based metrics
            dynamic_metrics = calculate_dynamic_threshold_metrics(y_true, row_scores)
            dynamic_binary_accuracy = calculate_dynamic_binary_accuracy(y_true, row_scores)
            
            # Store dynamic metrics
            all_dynamic_f1.append(dynamic_metrics['dynamic_f1'])
            all_dynamic_precision.append(dynamic_metrics['dynamic_precision'])
            all_dynamic_recall.append(dynamic_metrics['dynamic_recall'])
            all_dynamic_binary_accuracy.append(dynamic_binary_accuracy)
            all_dynamic_thresholds.append(dynamic_metrics['threshold_used'])

            # FIXED: Calculate pair-level TP/FP/FN (not row-level)
            pair_level_counts = calculate_pair_level_tp_fp_fn(
                pair_scores_np, row_sentence_pairs, len(rows), len(positive_sentences),
                debug_model_name="frozen_encoder"
            )
            
            total_tp += pair_level_counts['tp']
            total_fp += pair_level_counts['fp']
            total_fn += pair_level_counts['fn'] 
            total_ground_truth_positives += pair_level_counts['total_ground_truth_pairs']
            
            # Keep row-level calculations for backward compatibility  
            min_score = np.min(row_scores)
            max_score = np.max(row_scores)
            dynamic_threshold = (max_score + min_score) / 2.0
            y_pred_rows = (row_scores >= dynamic_threshold).astype(int)
            
            # Calculate row-sentence pair metrics
            pair_metrics = calculate_row_sentence_pair_metrics(
                pair_scores_np, row_sentence_pairs, len(rows), len(positive_sentences)
            )
            
            # Store pair metrics
            for k_idx in range(3):  # @1, @3, @5
                all_pair_metrics['precision_at_k'][k_idx].append(pair_metrics['pair_precision_at_k'][k_idx])
                all_pair_metrics['recall_at_k'][k_idx].append(pair_metrics['pair_recall_at_k'][k_idx])
                all_pair_metrics['f1_at_k'][k_idx].append(pair_metrics['pair_f1_at_k'][k_idx])
            
            all_pair_metrics['roc_auc'].append(pair_metrics['pair_roc_auc'])
            all_pair_metrics['avg_precision'].append(pair_metrics['pair_avg_precision'])
            all_pair_metrics['mean_rank'].append(pair_metrics['pair_mean_rank'])
            
            # Store for visualization compatibility
            for row_idx, (score, label) in enumerate(zip(row_scores, y_true)):
                all_binary_labels.append(label)
                pair_scores_list.append(score)
                all_pair_scores_data.append((example_idx, row_idx, score, bool(label)))
            
            # Store detailed results
            sorted_indices = np.argsort(row_scores)[::-1]
            example_result = {
                'example_idx': example_idx,
                'anchor_id': anchor_id,
                'num_rows': len(rows),
                'num_sentences': len(positive_sentences),
                'ground_truth_row_indices': highlighted_row_indices,
                'ground_truth_row_sentence_pairs': row_sentence_pairs,
                'row_scores': row_scores.tolist(),
                'sorted_predictions': sorted_indices.tolist(),
                'global_similarity': safe_tensor_to_scalar(global_similarity),  # Now uses proper tensor
                'binary_labels': y_true.tolist(),
                'pair_metrics': pair_metrics,
                'overall_accuracy': overall_accuracy  # UPDATED: Only overall accuracy
            }
            
            all_results.append(example_result)
    
    # Calculate overall metrics - SAME AS OTHER STAGES
    if not all_results:
        print("❌ No valid examples found")
        return {}
    
    def get_metrics_at_k(prec_list, rec_list, f1_list, ndcg_list):
        return {
            'precision': np.mean(prec_list),
            'recall': np.mean(rec_list),
            'f1': np.mean(f1_list),
            'ndcg': np.mean(ndcg_list),
            'precision_std': np.std(prec_list),
            'recall_std': np.std(rec_list),
            'f1_std': np.std(f1_list),
            'ndcg_std': np.std(ndcg_list)
        }
    
    # Calculate IR metrics for each K + "all"
    metrics_1 = get_metrics_at_k(all_precision_1, all_recall_1, all_f1_1, all_ndcg_1)
    metrics_3 = get_metrics_at_k(all_precision_3, all_recall_3, all_f1_3, all_ndcg_3)
    metrics_5 = get_metrics_at_k(all_precision_5, all_recall_5, all_f1_5, all_ndcg_5)
    metrics_10 = get_metrics_at_k(all_precision_10, all_recall_10, all_f1_10, all_ndcg_10)
    metrics_all = get_metrics_at_k(all_precision_all, all_recall_all, all_f1_all, all_ndcg_all)
    
    # Enhanced accuracy calculations - UPDATED: Only overall accuracy
    overall_accuracy_final = np.mean(all_overall_accuracy)
    
    # NEW: Calculate average dynamic threshold-based metrics
    dynamic_f1_final = np.mean(all_dynamic_f1) if all_dynamic_f1 else 0.0
    dynamic_precision_final = np.mean(all_dynamic_precision) if all_dynamic_precision else 0.0
    dynamic_recall_final = np.mean(all_dynamic_recall) if all_dynamic_recall else 0.0
    dynamic_binary_accuracy_final = np.mean(all_dynamic_binary_accuracy) if all_dynamic_binary_accuracy else 0.0
    average_threshold_used = np.mean(all_dynamic_thresholds) if all_dynamic_thresholds else 0.0
    
    # Calculate row-level pair metrics (backward compatibility)
    all_binary_labels = np.array(all_binary_labels)
    pair_scores_array = np.array(pair_scores_list)
    
    try:
        roc_auc = roc_auc_score(all_binary_labels, pair_scores_array) if len(np.unique(all_binary_labels)) > 1 else 0.0
    except ValueError:
        roc_auc = 0.0
    
    # Calculate pair-level metrics
    pair_level_metrics = {}
    for k_idx, k in enumerate([1, 3, 5]):
        pair_level_metrics[f'pair_precision_at_{k}'] = np.mean(all_pair_metrics['precision_at_k'][k_idx])
        pair_level_metrics[f'pair_recall_at_{k}'] = np.mean(all_pair_metrics['recall_at_k'][k_idx])
        pair_level_metrics[f'pair_f1_at_{k}'] = np.mean(all_pair_metrics['f1_at_k'][k_idx])
    
    pair_level_metrics['pair_roc_auc'] = np.mean(all_pair_metrics['roc_auc'])
    pair_level_metrics['pair_avg_precision'] = np.mean(all_pair_metrics['avg_precision'])
    pair_level_metrics['pair_mean_rank'] = np.mean(all_pair_metrics['mean_rank'])
    
    # UPDATED: Use pair-level average precision (homogenized with training script)
    avg_precision = pair_level_metrics['pair_avg_precision']
    
    # UPDATED: New result summary emphasizing Avg Precision first, F1 second, Overall Accuracy third
    print(f"📊 FROZEN ENCODER RESULTS:")
    print(f"   ⭐ Average Precision: {avg_precision:.3f} (PRIMARY METRIC)")
    print(f"   📈 Dynamic F1: {dynamic_f1_final:.3f} (SECONDARY METRIC)")
    print(f"   🎯 Overall Accuracy (F1-based): {overall_accuracy_final:.3f} (Pair-level F1)")
    print(f"   📋 NDCG@1: {metrics_1['ndcg']:.3f}, NDCG@3: {metrics_3['ndcg']:.3f}, NDCG@10: {metrics_10['ndcg']:.3f}")
    print(f"   🔗 Pair-level Avg Precision: {pair_level_metrics['pair_avg_precision']:.3f}")
    print(f"   📊 Pair-level Mean Rank: {pair_level_metrics['pair_mean_rank']:.1f}")
    
    return {
        'model_name': 'frozen_encoder',
        'examples_processed': examples_processed,
        'overall_accuracy': overall_accuracy_final,  # UPDATED: Only overall accuracy
        'accuracy': overall_accuracy_final,  # For backward compatibility
        'total_comparisons': len(all_results),  # For backward compatibility
        'correct_predictions': int(overall_accuracy_final * len(all_results)),  # For backward compatibility
        'metrics_at_1': metrics_1,
        'metrics_at_3': metrics_3,
        'metrics_at_5': metrics_5,
        'metrics_at_10': metrics_10,
        'metrics_at_all': metrics_all,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,  # PRIMARY METRIC
        'f1_score_at_5': metrics_5['f1'],  # NEW: Prominent F1 score
        'f1_score_at_all': metrics_all['f1'],  # NEW: Prominent F1 score
        # NEW: Dynamic threshold-based metrics
        'dynamic_f1': dynamic_f1_final,  # SECONDARY METRIC
        'dynamic_precision': dynamic_precision_final,
        'dynamic_recall': dynamic_recall_final,
        'dynamic_binary_accuracy': dynamic_binary_accuracy_final,
        'average_threshold_used': average_threshold_used,
        'pair_level_metrics': pair_level_metrics,
        'pair_scores_data': all_pair_scores_data,
        'detailed_results': all_results,
        'mean_highlighted_rank': pair_level_metrics['pair_mean_rank'],
        'prediction_breakdown': {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'total_ground_truth_positives': total_ground_truth_positives
        }
    }



def print_detailed_four_stage_comparison(frozen_encoder_metrics, simple_metrics, sophisticated_metrics, trained_metrics):
    """Print detailed comparison of all 4 evaluation stages with Average Precision as primary metric."""
    print(f"\n" + "="*120)
    print(f"📊 COMPREHENSIVE 4-STAGE PROTRIX EVALUATION ANALYSIS")
    print(f"🔥 Stage 0: Frozen Encoder Only → 🎯 Stage 1: +Simple Cross-Attention → 🚀 Stage 2: +Sophisticated Architecture → 🏆 Stage 3: +Training")
    print(f"⭐ PRIMARY METRIC: Average Precision | 📈 SECONDARY METRICS: F1 Score, Overall Accuracy")
    
    # Check if Stage 2 and Stage 3 results are suspiciously similar
    stage2_ap = sophisticated_metrics.get('average_precision', 0.0)
    stage3_ap = trained_metrics.get('average_precision', 0.0) 
    if abs(stage2_ap - stage3_ap) < 0.01:  # Less than 1% difference
        print(f"⚠️ WARNING: Stage 2 and Stage 3 Average Precision are nearly identical!")
        print(f"⚠️ This indicates trained weights were not loaded for Stage 3.")
        print(f"⚠️ Stage 3 comparison results below may be misleading.")
    
    print(f"="*120)
    
    # Extract PRIMARY METRIC values (Average Precision)
    frozen_ap = frozen_encoder_metrics['average_precision']
    simple_ap = simple_metrics['average_precision']
    sophisticated_ap = sophisticated_metrics['average_precision']
    trained_ap = trained_metrics['average_precision']
    
    # Extract secondary metric values
    frozen_acc = frozen_encoder_metrics['overall_accuracy']
    simple_acc = simple_metrics['overall_accuracy']
    sophisticated_acc = sophisticated_metrics['overall_accuracy']
    trained_acc = trained_metrics['overall_accuracy']
    
    # Extract F1 scores (using Dynamic F1 as primary F1 metric)
    frozen_f1 = frozen_encoder_metrics['dynamic_f1']
    simple_f1 = simple_metrics['dynamic_f1']
    sophisticated_f1 = sophisticated_metrics['dynamic_f1']
    trained_f1 = trained_metrics['dynamic_f1']

    # Calculate improvements based on PRIMARY METRIC (Average Precision)
    cross_attention_benefit = simple_ap - frozen_ap
    advanced_features_benefit = sophisticated_ap - simple_ap
    training_benefit = trained_ap - sophisticated_ap
    total_benefit = trained_ap - frozen_ap
    
    print(f"\n🎯 COMPLETE 4-STAGE PROGRESSION ANALYSIS (PRIMARY METRIC: Average Precision):")
    print(f"{'Stage':<35} {'Avg Precision':<15} {'Perfect Acc':<15} {'Dynamic F1':<12} {'AP Improvement':<15} {'% of Total':<12}")
    print(f"-" * 110)
    print(f"{'🔥 Stage 0: Frozen Encoder Only':<35} {frozen_ap:<15.4f} {frozen_acc:<15.4f} {frozen_f1:<12.4f} {'-':<15} {'-':<12}")
    print(f"{'🎯 Stage 1: Untrained Simple':<35} {simple_ap:<15.4f} {simple_acc:<15.4f} {simple_f1:<12.4f} {f'+{cross_attention_benefit:.4f}':<15} {f'{(cross_attention_benefit/total_benefit*100):.1f}%' if total_benefit > 0 else 'N/A':<12}")
    print(f"{'🚀 Stage 2: Untrained Sophisticated':<35} {sophisticated_ap:<15.4f} {sophisticated_acc:<15.4f} {sophisticated_f1:<12.4f} {f'+{advanced_features_benefit:.4f}':<15} {f'{(advanced_features_benefit/total_benefit*100):.1f}%' if total_benefit > 0 else 'N/A':<12}")
    print(f"{'🏆 Stage 3: Trained Sophisticated':<35} {trained_ap:<15.4f} {trained_acc:<15.4f} {trained_f1:<12.4f} {f'+{training_benefit:.4f}':<15} {f'{(training_benefit/total_benefit*100):.1f}%' if total_benefit > 0 else 'N/A':<12}")
    print(f"{'📈 Total Improvement':<35} {'-':<15} {'-':<15} {'-':<12} {f'+{total_benefit:.4f}':<15} {'100.0%':<12}")
    
    print(f"\n🔍 DETAILED IMPACT ATTRIBUTION (Average Precision):")
    if total_benefit > 0:
        ca_pct = (cross_attention_benefit/total_benefit*100)
        af_pct = (advanced_features_benefit/total_benefit*100)
        tr_pct = (training_benefit/total_benefit*100)
        print(f"   🎯 Simple Cross-Attention (Frozen Encoder): {cross_attention_benefit:.4f} ({ca_pct:.1f}% of total)")
        print(f"   🚀 Sophisticated Architecture:             {advanced_features_benefit:.4f} ({af_pct:.1f}% of total)")
        print(f"   🎓 Training Benefits:                      {training_benefit:.4f} ({tr_pct:.1f}% of total)")
    else:
        print(f"   🎯 Simple Cross-Attention (Frozen Encoder): {cross_attention_benefit:.4f}")
        print(f"   🚀 Sophisticated Architecture:             {advanced_features_benefit:.4f}")
        print(f"   🎓 Training Benefits:                      {training_benefit:.4f}")

    print(f"\n📋 DETAILED METRIC COMPARISON:")
    print(f"{'Metric':<25} {'Stage 0':<12} {'Stage 1':<12} {'Stage 2':<12} {'Stage 3':<12} {'Best Stage':<12}")
    print(f"-" * 97)
    
    # Compare key metrics across all 4 stages
    metrics_to_compare = [
        ("⭐ Average Precision", "average_precision", "average_precision", "average_precision", "average_precision"),
        ("📈 Dynamic F1", "dynamic_f1", "dynamic_f1", "dynamic_f1", "dynamic_f1"),
        ("📊 Dynamic Binary Acc", "dynamic_binary_accuracy", "dynamic_binary_accuracy", "dynamic_binary_accuracy", "dynamic_binary_accuracy"),
        ("📈 F1@5", "f1_score_at_5", "f1_score_at_5", "f1_score_at_5", "f1_score_at_5"),
        ("🎯 Overall Accuracy", "overall_accuracy", "overall_accuracy", "overall_accuracy", "overall_accuracy"),
        ("NDCG@1", "metrics_at_1.ndcg", "metrics_at_1.ndcg", "metrics_at_1.ndcg", "metrics_at_1.ndcg"),
        ("NDCG@3", "metrics_at_3.ndcg", "metrics_at_3.ndcg", "metrics_at_3.ndcg", "metrics_at_3.ndcg"),
        ("NDCG@10", "metrics_at_10.ndcg", "metrics_at_10.ndcg", "metrics_at_10.ndcg", "metrics_at_10.ndcg"),
        ("NDCG@all", "metrics_at_all.ndcg", "metrics_at_all.ndcg", "metrics_at_all.ndcg", "metrics_at_all.ndcg"),
        ("ROC-AUC", "roc_auc", "roc_auc", "roc_auc", "roc_auc"),
        ("Pair ROC-AUC", "pair_level_metrics.pair_roc_auc", "pair_level_metrics.pair_roc_auc", "pair_level_metrics.pair_roc_auc", "pair_level_metrics.pair_roc_auc"),
        ("Pair Mean Rank", "mean_highlighted_rank", "mean_highlighted_rank", "mean_highlighted_rank", "mean_highlighted_rank")
    ]
    
    for metric_name, frozen_key, simple_key, soph_key, trained_key in metrics_to_compare:
        # Extract values using nested key access
        def get_nested_value(data, key_path):
            keys = key_path.split('.')
            value = data
            for key in keys:
                value = value[key]
            return value
        
        frozen_val = get_nested_value(frozen_encoder_metrics, frozen_key)
        simple_val = get_nested_value(simple_metrics, simple_key)
        soph_val = get_nested_value(sophisticated_metrics, soph_key)
        trained_val = get_nested_value(trained_metrics, trained_key)
        
        # Determine best stage (for rank, lower is better)
        all_vals = [frozen_val, simple_val, soph_val, trained_val]
        if "Rank" in metric_name:
            best_val = min(all_vals)
            if best_val == frozen_val: best_stage = "Stage 0"
            elif best_val == simple_val: best_stage = "Stage 1"
            elif best_val == soph_val: best_stage = "Stage 2"
            else: best_stage = "Stage 3"
        else:
            best_val = max(all_vals)
            if best_val == frozen_val: best_stage = "Stage 0"
            elif best_val == simple_val: best_stage = "Stage 1"
            elif best_val == soph_val: best_stage = "Stage 2"
            else: best_stage = "Stage 3"
        
        print(f"{metric_name:<25} {frozen_val:<12.3f} {simple_val:<12.3f} {soph_val:<12.3f} {trained_val:<12.3f} {best_stage:<12}")
    
    # Component assessment based on Average Precision
    print(f"\n💡 COMPONENT ASSESSMENT (Based on Average Precision):")
    
    # Cross-attention + trainable encoder assessment
    if cross_attention_benefit > 0.05:
        print("✅ CROSS-ATTENTION + TRAINABLE ENCODER: Significant benefit - architecture worth it")
    elif cross_attention_benefit > 0.02:
        print("✅ CROSS-ATTENTION + TRAINABLE ENCODER: Moderate benefit - reasonable addition")
    elif cross_attention_benefit > 0.005:
        print("⚠️  CROSS-ATTENTION + TRAINABLE ENCODER: Small benefit - consider if worth complexity")
    else:
        print("❌ CROSS-ATTENTION + TRAINABLE ENCODER: Minimal benefit - architecture may not suit task")
    
    # Sophisticated architecture assessment
    if advanced_features_benefit > 0.05:
        print("✅ SOPHISTICATED ARCHITECTURE: Significant benefit - LoRA/advanced attention worth it")
    elif advanced_features_benefit > 0.02:
        print("✅ SOPHISTICATED ARCHITECTURE: Moderate benefit - features are helping")
    elif advanced_features_benefit > 0.005:
        print("⚠️  SOPHISTICATED ARCHITECTURE: Small benefit - consider simplifying")
    else:
        print("❌ SOPHISTICATED ARCHITECTURE: Minimal benefit - use simple architecture")
    
    # Training assessment
    if training_benefit > max(cross_attention_benefit, advanced_features_benefit):
        print("✅ TRAINING: Primary driver of performance - excellent!")
    elif training_benefit > 0.02:
        print("✅ TRAINING: Effective improvement - training is working")
    elif training_benefit > 0.005:
        print("⚠️  TRAINING: Modest improvement - consider longer training")
    else:
        print("❌ TRAINING: Minimal improvement - check hyperparameters/data")
        
    print("="*120)

def create_comprehensive_four_stage_visualizations(frozen_metrics, simple_metrics, sophisticated_metrics, trained_metrics, output_dir):
    """Create comprehensive visualizations comparing all 4 evaluation stages."""
    print("\nCreating comprehensive 4-stage comparison visualizations...")
    
    plt.style.use('default')
    sns.set_palette("Set2")  # Better colors for 4 categories
    
    # 1. Four-Stage IR Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Complete 4-Stage Model Evolution Analysis (Protrix)', fontsize=16, fontweight='bold')
    
    k_values = [1, 3, 5, 10]
    stage_names = ['Stage 0: Frozen Encoder', 'Stage 1: Untrained Simple', 'Stage 2: Untrained Sophisticated', 'Stage 3: Trained Sophisticated']
    stage_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']  # Orange, Green, Red, Blue
    
    # Precision@K comparison across all 4 stages
    frozen_precision = [frozen_metrics[f'metrics_at_{k}']['precision'] for k in k_values]
    simple_precision = [simple_metrics[f'metrics_at_{k}']['precision'] for k in k_values]
    sophisticated_precision = [sophisticated_metrics[f'metrics_at_{k}']['precision'] for k in k_values]
    trained_precision = [trained_metrics[f'metrics_at_{k}']['precision'] for k in k_values]
    
    axes[0, 0].plot(k_values, frozen_precision, 'o-', label=stage_names[0], linewidth=3, markersize=8, color=stage_colors[0])
    axes[0, 0].plot(k_values, simple_precision, 's-', label=stage_names[1], linewidth=3, markersize=8, color=stage_colors[1])
    axes[0, 0].plot(k_values, sophisticated_precision, '^-', label=stage_names[2], linewidth=3, markersize=8, color=stage_colors[2])
    axes[0, 0].plot(k_values, trained_precision, 'D-', label=stage_names[3], linewidth=3, markersize=8, color=stage_colors[3])
    axes[0, 0].set_title('Precision@K Progression', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('K')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticks(k_values)
    axes[0, 0].set_xticklabels([str(k) for k in k_values])
    
    # Recall@K comparison across all 4 stages
    frozen_recall = [frozen_metrics[f'metrics_at_{k}']['recall'] for k in k_values]
    simple_recall = [simple_metrics[f'metrics_at_{k}']['recall'] for k in k_values]
    sophisticated_recall = [sophisticated_metrics[f'metrics_at_{k}']['recall'] for k in k_values]
    trained_recall = [trained_metrics[f'metrics_at_{k}']['recall'] for k in k_values]
    
    axes[0, 1].plot(k_values, frozen_recall, 'o-', label=stage_names[0], linewidth=3, markersize=8, color=stage_colors[0])
    axes[0, 1].plot(k_values, simple_recall, 's-', label=stage_names[1], linewidth=3, markersize=8, color=stage_colors[1])
    axes[0, 1].plot(k_values, sophisticated_recall, '^-', label=stage_names[2], linewidth=3, markersize=8, color=stage_colors[2])
    axes[0, 1].plot(k_values, trained_recall, 'D-', label=stage_names[3], linewidth=3, markersize=8, color=stage_colors[3])
    axes[0, 1].set_title('Recall@K Progression', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('K')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticks(k_values)
    axes[0, 1].set_xticklabels([str(k) for k in k_values])
    
    # NDCG@K comparison across all 4 stages
    frozen_ndcg = [frozen_metrics[f'metrics_at_{k}']['ndcg'] for k in k_values]
    simple_ndcg = [simple_metrics[f'metrics_at_{k}']['ndcg'] for k in k_values]
    sophisticated_ndcg = [sophisticated_metrics[f'metrics_at_{k}']['ndcg'] for k in k_values]
    trained_ndcg = [trained_metrics[f'metrics_at_{k}']['ndcg'] for k in k_values]
    
    axes[0, 2].plot(k_values, frozen_ndcg, 'o-', label=stage_names[0], linewidth=3, markersize=8, color=stage_colors[0])
    axes[0, 2].plot(k_values, simple_ndcg, 's-', label=stage_names[1], linewidth=3, markersize=8, color=stage_colors[1])
    axes[0, 2].plot(k_values, sophisticated_ndcg, '^-', label=stage_names[2], linewidth=3, markersize=8, color=stage_colors[2])
    axes[0, 2].plot(k_values, trained_ndcg, 'D-', label=stage_names[3], linewidth=3, markersize=8, color=stage_colors[3])
    axes[0, 2].set_title('NDCG@K Progression', fontweight='bold', fontsize=14)
    axes[0, 2].set_xlabel('K')
    axes[0, 2].set_ylabel('NDCG')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_xticks(k_values)
    axes[0, 2].set_xticklabels([str(k) for k in k_values])
    
    # Row-Sentence Pair Metrics across all 4 stages - UPDATED to focus on key metrics
    pair_metrics = ['⭐ Avg Precision', '📈 Dynamic F1', '📊 Dynamic Bin Acc', 'Overall Accuracy', 'NDCG@10']
    frozen_pair_values = [
        frozen_metrics['average_precision'],
        frozen_metrics['dynamic_f1'],
        frozen_metrics.get('dynamic_binary_accuracy', 0.0),
        frozen_metrics['overall_accuracy'],
        frozen_metrics['metrics_at_10']['ndcg']
    ]
    simple_pair_values = [
        simple_metrics['average_precision'],
        simple_metrics['dynamic_f1'],
        simple_metrics.get('dynamic_binary_accuracy', 0.0),
        simple_metrics['overall_accuracy'],
        simple_metrics['metrics_at_10']['ndcg']
    ]
    sophisticated_pair_values = [
        sophisticated_metrics['average_precision'],
        sophisticated_metrics['dynamic_f1'],
        sophisticated_metrics.get('dynamic_binary_accuracy', 0.0),
        sophisticated_metrics['overall_accuracy'],
        sophisticated_metrics['metrics_at_10']['ndcg']
    ]
    trained_pair_values = [
        trained_metrics['average_precision'],
        trained_metrics['dynamic_f1'],
        trained_metrics.get('dynamic_binary_accuracy', 0.0),
        trained_metrics['overall_accuracy'],
        trained_metrics['metrics_at_10']['ndcg']
    ]
    
    x = np.arange(len(pair_metrics))
    width = 0.2
    
    axes[1, 0].bar(x - 1.5*width, frozen_pair_values, width, label=stage_names[0], alpha=0.8, color=stage_colors[0])
    axes[1, 0].bar(x - 0.5*width, simple_pair_values, width, label=stage_names[1], alpha=0.8, color=stage_colors[1])
    axes[1, 0].bar(x + 0.5*width, sophisticated_pair_values, width, label=stage_names[2], alpha=0.8, color=stage_colors[2])
    axes[1, 0].bar(x + 1.5*width, trained_pair_values, width, label=stage_names[3], alpha=0.8, color=stage_colors[3])
    axes[1, 0].set_title('Key Metrics 4-Stage Progression', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(pair_metrics, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim(0, 1)
    
    # Overall performance progression - UPDATED to focus on primary metrics
    overall_metrics = ['⭐ Avg Precision', '📈 Dynamic F1', '📊 Dynamic Bin Acc', '📈 F1@5', 'Overall Acc', 'NDCG@10']
    frozen_overall = [
        frozen_metrics['average_precision'],
        frozen_metrics['dynamic_f1'],
        frozen_metrics.get('dynamic_binary_accuracy', 0.0),
        frozen_metrics['f1_score_at_5'],
        frozen_metrics['overall_accuracy'],
        frozen_metrics['metrics_at_10']['ndcg']
    ]
    simple_overall = [
        simple_metrics['average_precision'],
        simple_metrics['dynamic_f1'],
        simple_metrics.get('dynamic_binary_accuracy', 0.0),
        simple_metrics['f1_score_at_5'],
        simple_metrics['overall_accuracy'],
        simple_metrics['metrics_at_10']['ndcg']
    ]
    sophisticated_overall = [
        sophisticated_metrics['average_precision'],
        sophisticated_metrics['dynamic_f1'],
        sophisticated_metrics.get('dynamic_binary_accuracy', 0.0),
        sophisticated_metrics['f1_score_at_5'],
        sophisticated_metrics['overall_accuracy'],
        sophisticated_metrics['metrics_at_10']['ndcg']
    ]
    trained_overall = [
        trained_metrics['average_precision'],
        trained_metrics['dynamic_f1'],
        trained_metrics.get('dynamic_binary_accuracy', 0.0),
        trained_metrics['f1_score_at_5'],
        trained_metrics['overall_accuracy'],
        trained_metrics['metrics_at_10']['ndcg']
    ]
    
    metric_indices = np.arange(len(overall_metrics))
    axes[1, 1].plot(metric_indices, frozen_overall, 'o-', label=stage_names[0], linewidth=3, markersize=8, color=stage_colors[0])
    axes[1, 1].plot(metric_indices, simple_overall, 's-', label=stage_names[1], linewidth=3, markersize=8, color=stage_colors[1])
    axes[1, 1].plot(metric_indices, sophisticated_overall, '^-', label=stage_names[2], linewidth=3, markersize=8, color=stage_colors[2])
    axes[1, 1].plot(metric_indices, trained_overall, 'D-', label=stage_names[3], linewidth=3, markersize=8, color=stage_colors[3])
    axes[1, 1].set_title('Performance Evolution Across Key Metrics', fontweight='bold', fontsize=14)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(metric_indices)
    axes[1, 1].set_xticklabels(overall_metrics, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    # Component contribution analysis - NOW using improvement from baseline (Stage 0)
    frozen_ap = frozen_metrics['average_precision']
    simple_ap = simple_metrics['average_precision']
    sophisticated_ap = sophisticated_metrics['average_precision'] 
    trained_ap = trained_metrics['average_precision']

    # Compute improvement from baseline (Stage 0) for each stage
    stage1_delta = simple_ap - frozen_ap
    stage2_delta = sophisticated_ap - frozen_ap
    stage3_delta = trained_ap - frozen_ap

    components = [
        'Stage 1: Untrained Simple\nvs Baseline',
        'Stage 2: Untrained Sophisticated\nvs Baseline',
        'Stage 3: Trained Sophisticated\nvs Baseline'
    ]
    benefits = [stage1_delta, stage2_delta, stage3_delta]
    benefit_colors = ['green' if b > 0 else 'red' for b in benefits]

    bars = axes[1, 2].bar(components, benefits, color=benefit_colors, alpha=0.7)
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 2].set_title('Component Contribution Analysis\n(Average Precision vs Baseline)', fontweight='bold', fontsize=14)
    axes[1, 2].set_xlabel('Model Components')
    axes[1, 2].set_ylabel('Average Precision Improvement (vs Baseline)')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, value in zip(bars, benefits):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{value:+.3f}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    comprehensive_plot_path = os.path.join(output_dir, "comprehensive_four_stage_comparison.png")
    save_plot_multi_format(comprehensive_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved 4-stage comprehensive comparison: {comprehensive_plot_path}")
    plt.close()

def create_four_stage_roc_pr_curves(frozen_metrics, simple_metrics, sophisticated_metrics, trained_metrics, output_dir):
    """Create comprehensive ROC and PR curves comparing all 4 evaluation stages."""
    print("Creating 4-stage ROC and PR curve analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Complete 4-Stage ROC and PR Curve Evolution (Protrix)', fontsize=16, fontweight='bold')
    
    stage_names = ['Stage 0: Frozen Encoder', 'Stage 1: Untrained Simple', 'Stage 2: Untrained Sophisticated', 'Stage 3: Trained Sophisticated']
    stage_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']  # Orange, Green, Red, Blue
    stage_linestyles = ['-', '--', '-.', ':']
    stage_markers = ['o', 's', '^', 'D']
    
    # Check if we have pair scores data for all stages
    stages_data = [frozen_metrics, simple_metrics, sophisticated_metrics, trained_metrics]
    
    try:
        # ROC Curves
        ax1 = axes[0]
        for i, (stage_data, name, color, linestyle, marker) in enumerate(zip(
            stages_data, stage_names, stage_colors, stage_linestyles, stage_markers
        )):
            if 'pair_scores_data' in stage_data and stage_data['pair_scores_data']:
                # Extract labels and scores from the list of tuples
                # Format: (example_idx, row_idx, score, is_highlighted)
                pair_data = stage_data['pair_scores_data']
                labels = [item[3] for item in pair_data]  # is_highlighted (True/False -> 1/0)
                scores = [item[2] for item in pair_data]  # score values
                
                # Convert boolean labels to integers
                labels = [1 if label else 0 for label in labels]
                
                if len(set(labels)) > 1:  # Need both positive and negative examples
                    fpr, tpr, _ = roc_curve(labels, scores)
                    auc_score = stage_data['roc_auc']
                    
                    ax1.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=3, 
                            label=f"{name} (AUC = {auc_score:.3f})", 
                            marker=marker, markersize=4, markevery=max(1, len(fpr)//10))
                else:
                    print(f"⚠️ Insufficient label variety for {name}")
            else:
                print(f"⚠️ No pair scores data available for {name}")
        
        # Diagonal line for random classifier
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curves: 4-Stage Evolution', fontweight='bold', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # PR Curves
        ax2 = axes[1]
        for i, (stage_data, name, color, linestyle, marker) in enumerate(zip(
            stages_data, stage_names, stage_colors, stage_linestyles, stage_markers
        )):
            if 'pair_scores_data' in stage_data and stage_data['pair_scores_data']:
                # Extract labels and scores from the list of tuples
                pair_data = stage_data['pair_scores_data']
                labels = [1 if item[3] else 0 for item in pair_data]  # is_highlighted
                scores = [item[2] for item in pair_data]  # score values
                
                if len(set(labels)) > 1:  # Need both positive and negative examples
                    precision, recall, _ = precision_recall_curve(labels, scores)
                    ap_score = stage_data['average_precision']
                    
                    ax2.plot(recall, precision, color=color, linestyle=linestyle, linewidth=3,
                            label=f"{name} (AP = {ap_score:.3f})",
                            marker=marker, markersize=4, markevery=max(1, len(recall)//10))
        
        # Baseline for random classifier (proportion of positive class)
        if 'pair_scores_data' in trained_metrics and trained_metrics['pair_scores_data']:
            pair_data = trained_metrics['pair_scores_data']
            labels = [1 if item[3] else 0 for item in pair_data]
            pos_ratio = np.mean(labels)
            ax2.axhline(y=pos_ratio, color='k', linestyle='--', alpha=0.5, linewidth=1,
                       label=f'Random Classifier (AP = {pos_ratio:.3f})')
        
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curves: 4-Stage Evolution', fontweight='bold', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
    except Exception as e:
        print(f"⚠️ Could not create ROC/PR curves: {e}")
        import traceback
        traceback.print_exc()
        for ax in axes:
            ax.text(0.5, 0.5, f"Curve data unavailable\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    plt.tight_layout()
    curve_plot_path = os.path.join(output_dir, "four_stage_roc_pr_curves.png")
    save_plot_multi_format(curve_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved 4-stage ROC/PR curves: {curve_plot_path}")
    plt.close()

def create_stage_progression_analysis(frozen_metrics, simple_metrics, sophisticated_metrics, trained_metrics, output_dir):
    """Create detailed stage progression analysis showing step-by-step improvements."""
    print("Creating detailed stage progression analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed 4-Stage Progression Analysis (Protrix)', fontsize=16, fontweight='bold')
    
    stage_names = ['Stage 0\nFrozen Encoder', 'Stage 1\nUntrained Simple', 'Stage 2\nUntrained Sophisticated', 'Stage 3\nTrained']
    stage_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
    stages = [0, 1, 2, 3]
    
    # 1. Overall Accuracy Progression (F1-based Pair-level Accuracy)
    accuracies = [
        frozen_metrics['overall_accuracy'],
        simple_metrics['overall_accuracy'], 
        sophisticated_metrics['overall_accuracy'],
        trained_metrics['overall_accuracy']
    ]
    
    axes[0, 0].plot(stages, accuracies, 'o-', linewidth=4, markersize=12, color='darkblue')
    axes[0, 0].fill_between(stages, accuracies, alpha=0.3, color='lightblue')
    
    # Add improvement annotations
    for i in range(1, len(stages)):
        improvement = accuracies[i] - accuracies[i-1]
        axes[0, 0].annotate(f'+{improvement:.3f}', 
                           xy=(stages[i], accuracies[i]), 
                           xytext=(stages[i], accuracies[i] + 0.05),
                           ha='center', fontsize=10, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    axes[0, 0].set_title('Overall Accuracy Progression (F1-based Pair-level)', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Evaluation Stage')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(stages)
    axes[0, 0].set_xticklabels(stage_names)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Average Precision Progression (PRIMARY METRIC)
    avg_precisions = [
        frozen_metrics['average_precision'],
        simple_metrics['average_precision'],
        sophisticated_metrics['average_precision'],
        trained_metrics['average_precision']
    ]
    
    axes[0, 1].plot(stages, avg_precisions, 's-', linewidth=4, markersize=12, color='darkgreen')
    axes[0, 1].fill_between(stages, avg_precisions, alpha=0.3, color='lightgreen')
    
    # Add improvement annotations
    for i in range(1, len(stages)):
        improvement = avg_precisions[i] - avg_precisions[i-1]
        axes[0, 1].annotate(f'+{improvement:.3f}', 
                           xy=(stages[i], avg_precisions[i]), 
                           xytext=(stages[i], avg_precisions[i] + 0.05),
                           ha='center', fontsize=10, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    axes[0, 1].set_title('Average Precision Progression (PRIMARY)', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Evaluation Stage')
    axes[0, 1].set_ylabel('Average Precision')
    axes[0, 1].set_xticks(stages)
    axes[0, 1].set_xticklabels(stage_names)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # 3. ROC-AUC Progression
    roc_scores = [
        frozen_metrics['roc_auc'],
        simple_metrics['roc_auc'],
        sophisticated_metrics['roc_auc'],
        trained_metrics['roc_auc']
    ]
    
    axes[1, 0].plot(stages, roc_scores, '^-', linewidth=4, markersize=12, color='darkorange')
    axes[1, 0].fill_between(stages, roc_scores, alpha=0.3, color='moccasin')
    
    # Add improvement annotations
    for i in range(1, len(stages)):
        improvement = roc_scores[i] - roc_scores[i-1]
        axes[1, 0].annotate(f'+{improvement:.3f}', 
                           xy=(stages[i], roc_scores[i]), 
                           xytext=(stages[i], roc_scores[i] + 0.05),
                           ha='center', fontsize=10, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    axes[1, 0].set_title('ROC-AUC Progression', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Evaluation Stage')
    axes[1, 0].set_ylabel('ROC-AUC')
    axes[1, 0].set_xticks(stages)
    axes[1, 0].set_xticklabels(stage_names)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Dynamic F1 and Dynamic Binary Accuracy Progression (SECONDARY METRICS)
    f1_scores = [
        frozen_metrics['dynamic_f1'],
        simple_metrics['dynamic_f1'],
        sophisticated_metrics['dynamic_f1'],
        trained_metrics['dynamic_f1']
    ]
    dyn_bin_acc_scores = [
        frozen_metrics.get('dynamic_binary_accuracy', 0.0),
        simple_metrics.get('dynamic_binary_accuracy', 0.0),
        sophisticated_metrics.get('dynamic_binary_accuracy', 0.0),
        trained_metrics.get('dynamic_binary_accuracy', 0.0)
    ]
    
    axes[1, 1].plot(stages, f1_scores, 'D-', linewidth=4, markersize=12, color='darkred', label='Dynamic F1')
    axes[1, 1].plot(stages, dyn_bin_acc_scores, 'o--', linewidth=3, markersize=10, color='teal', label='Dynamic Binary Acc')
    axes[1, 1].fill_between(stages, f1_scores, alpha=0.2, color='mistyrose')
    
    # Add improvement annotations
    for i in range(1, len(stages)):
        improvement = f1_scores[i] - f1_scores[i-1]
        axes[1, 1].annotate(f'+{improvement:.3f}', 
                           xy=(stages[i], f1_scores[i]), 
                           xytext=(stages[i], f1_scores[i] + 0.05),
                           ha='center', fontsize=10, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    axes[1, 1].set_title('Dynamic F1 + Dynamic Binary Accuracy', fontweight='bold', fontsize=14)
    axes[1, 1].set_xlabel('Evaluation Stage')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(stages)
    axes[1, 1].set_xticklabels(stage_names)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    
    plt.tight_layout()
    progression_plot_path = os.path.join(output_dir, "detailed_stage_progression_analysis.png")
    save_plot_multi_format(progression_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed progression analysis: {progression_plot_path}")
    plt.close()

def create_prediction_breakdown_plot(frozen_metrics, simple_metrics, sophisticated_metrics, trained_metrics, output_dir):
    """Create a stacked bar chart showing the breakdown of TP, FP, and FN for each stage."""
    print("Creating prediction breakdown plot...")
    
    stage_names = ['Stage 0: Frozen', 'Stage 1: Simple', 'Stage 2: Sophisticated', 'Stage 3: Trained']
    
    # Extract prediction breakdown data
    breakdowns = [
        frozen_metrics['prediction_breakdown'],
        simple_metrics['prediction_breakdown'],
        sophisticated_metrics['prediction_breakdown'],
        trained_metrics['prediction_breakdown']
    ]
    
    tps = [b['tp'] for b in breakdowns]
    fps = [b['fp'] for b in breakdowns]
    fns = [b['fn'] for b in breakdowns]
    total_positives = [b['total_ground_truth_positives'] for b in breakdowns]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Stacked bar chart
    ax.bar(stage_names, tps, label='True Positives (TP)', color='#2ca02c')
    ax.bar(stage_names, fps, bottom=tps, label='False Positives (FP)', color='#d62728')
    ax.bar(stage_names, fns, bottom=np.array(tps) + np.array(fps), label='False Negatives (FN)', color='#ff7f0e')
    
    # Ground truth line
    ax.axhline(y=total_positives[0], color='black', linestyle='--', label=f'Total Ground Truth Pairs ({total_positives[0]})')
    
    # Add labels and title
    ax.set_ylabel('Number of Row-Sentence Pairs')
    ax.set_title('Row-Sentence Pair Prediction Breakdown by Evaluation Stage', fontsize=16, fontweight='bold')
    ax.legend()
    
    # Add value labels to bars
    for i, (tp, fp, fn) in enumerate(zip(tps, fps, fns)):
        ax.text(i, tp / 2, str(tp), ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, tp + fp / 2, str(fp), ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, tp + fp + fn / 2, str(fn), ha='center', va='center', color='white', fontweight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "prediction_breakdown.png")
    save_plot_multi_format(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved prediction breakdown plot: {plot_path}")
    plt.close()

def main():
    """Main evaluation function for Protrix test data."""
    import argparse
    
    # Define model shortname mappings
    AVAILABLE_MODELS = {
        "roberta-large": "all-roberta-large-v1",
        "modernbert-base": "answerdotai/ModernBERT-base",
        "nomic-embed-v2": "nomic-ai/nomic-embed-text-v2-moe",
        "nomic-modernbert": "nomic-ai/modernbert-embed-base",
        "modernbert-large": "lightonai/modernbert-embed-large",
        "jina-v3": "jinaai/jina-embeddings-v3",
        "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Protrix Row-Sentence Evaluation")
    parser.add_argument("--model_key", type=str, default="roberta-large",
                      choices=list(AVAILABLE_MODELS.keys()),
                      help=f"Key to select model from available models: {list(AVAILABLE_MODELS.keys())}")
    
    parser.add_argument("--test_file", type=str, default="protrix_data/test_row_level.json",
                      help="Path to the Protrix test dataset")
    parser.add_argument("--annotation_file", type=str, default="protrix_data/Annotated_Test.json",
                      help="Path to the Protrix annotations")
    parser.add_argument("--model_path", type=str, default="./output_cross_attention_cache/test_model/all-roberta-large-v1_best.pt",
                      help="Path to the trained model checkpoint")
    parser.add_argument("--config_path", type=str, default="./output_cross_attention_cache/test_model/training_config.json",
                      help="Path to the training configuration file")
    parser.add_argument("--show_examples", action="store_true", default=False,
                      help="Show detailed examples during evaluation")
    # Normalization override
    parser.add_argument("--norm_type", type=str, default=None, choices=["layernorm", "rmsnorm"],
                      help="Override normalization type (defaults to training config if absent)")
    parser.add_argument("--use_qk_rmsnorm", action="store_true", default=None,
                      help="Override Q/K RMSNorm setting; if omitted, uses training config value")
    
    args = parser.parse_args()
    
    # Get the actual model name from the dictionary
    model_name = AVAILABLE_MODELS[args.model_key]
    
    print("🧪 Protrix Row-Sentence Evaluation")
    print("Evaluating on Protrix test dataset with manual row-sentence annotations")
    print(f"🤖 Model: {model_name}")
    print(f"📁 Test file: {args.test_file}")
    print(f"📋 Annotation file: {args.annotation_file}")
    print(f"🏆 Trained model path: {args.model_path}")
    print(f"⚙️ Config path: {args.config_path}")
    print("=" * 70)
    
    # Create output directory
    output_dir = create_output_directory("protrix_evaluation")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Load datasets
        print("\n📂 Loading datasets...")
        test_examples = load_row_level_dataset(args.test_file)
        annotations = load_protrix_annotations(args.annotation_file)
        
        print(f"✅ Loaded {len(test_examples)} test examples")
        print(f"✅ Loaded {len(annotations)} annotations")
        
        # Check consistency
        test_ids = set(ex.get("anchor_id") for ex in test_examples if ex.get("anchor_id") is not None)
        annotation_ids = set(annotations.keys())
        common_ids = test_ids & annotation_ids
        
        examples_with_annotations = sum(1 for aid in common_ids if annotations[aid])
        print(f"📊 Examples with annotations: {examples_with_annotations}")
        
        if len(common_ids) != len(test_ids) or len(common_ids) != len(annotation_ids):
            print(f"⚠️ ID mismatch: {len(test_ids)} test, {len(annotation_ids)} annotations, {len(common_ids)} common")
        
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {device}")
        
        # Load training configuration
        config = load_training_config(args.config_path)
        if args.norm_type is not None:
            config["norm_type"] = args.norm_type
        if args.use_qk_rmsnorm is not None:
            config["use_qk_rmsnorm"] = True
        
        # Copy training configuration files to output directory for tracking
        print(f"\n📋 Copying training configuration files to output directory...")
        copy_success = copy_training_config_to_output(args.config_path, output_dir)
        if copy_success:
            print(f"✅ Training configuration tracking enabled")
        else:
            print(f"⚠️ Training configuration tracking failed - continuing without config copy")
        
        # Enhanced 4-Stage Evaluation with Always-Frozen Encoder
        print(f"\n{'='*60}")
        print(f"🎯 4-STAGE PROTRIX EVALUATION (FROZEN ENCODER THROUGHOUT)")
        print(f"Stage 0: Frozen Encoder Baseline | Stage 1: +Proper Initialization")
        print(f"Stage 2: +Sophisticated Architecture | Stage 3: +Training")
        print(f"{'='*60}")
        
        total_start_time = time.time()
        
        # Stage 0: Frozen Encoder Baseline
        print(f"\n🔥 STAGE 0: FROZEN ENCODER BASELINE")
        stage_0_start = time.time()
        frozen_sentence_encoder = SentenceTransformer(
            model_name, 
            model_kwargs={"dtype": torch.bfloat16},
            trust_remote_code=True, 
            device=device
        )
        frozen_encoder_metrics = evaluate_frozen_encoder_only(
            frozen_sentence_encoder, test_examples, annotations, batch_size=1,
            init_method=config.get("init_method", "xavier_uniform"),
            init_method_params=config.get("init_method_params", {})
        )
        stage_0_time = time.time() - stage_0_start
        print(f"🔥 Stage 0 completed in {stage_0_time:.1f}s")
        
        # Stage 1: Frozen Encoder + Untrained Simple Cross-Attention
        print(f"\n🎯 STAGE 1: FROZEN ENCODER + UNTRAINED SIMPLE CROSS-ATTENTION")
        stage_1_start = time.time()
        simple_model = create_simple_model(config, device, model_name)
        simple_metrics = comprehensive_evaluation_single_model(
            simple_model, test_examples, annotations, "untrained_simple", config,
            batch_size=1, show_examples=args.show_examples
        )
        stage_1_time = time.time() - stage_1_start
        print(f"🎯 Stage 1 completed in {stage_1_time:.1f}s")
        
        # Stage 2: Frozen Encoder + Untrained Sophisticated
        print(f"\n🚀 STAGE 2: FROZEN ENCODER + UNTRAINED SOPHISTICATED (BIDIRECTIONAL)")
        stage_2_start = time.time()
        sophisticated_model = create_sophisticated_model(config, device, model_name)
        print("🔍 VERIFICATION: Stage 2 uses UNTRAINED sophisticated model")
        sophisticated_metrics = comprehensive_evaluation_single_model(
            sophisticated_model, test_examples, annotations, "untrained_sophisticated", config,
            batch_size=1, show_examples=args.show_examples
        )
        stage_2_time = time.time() - stage_2_start
        print(f"🚀 Stage 2 completed in {stage_2_time:.1f}s")
        
        # Stage 3: Frozen Encoder + Trained Sophisticated
        print(f"\n🏆 STAGE 3: FROZEN ENCODER + TRAINED SOPHISTICATED")
        stage_3_start = time.time()
        trained_model = load_trained_model(config, device, model_name, args.model_path)
        print("🔍 VERIFICATION: Stage 3 should use TRAINED sophisticated model")
        
        # Check if trained weights were actually loaded
        trained_weights_loaded = getattr(trained_model, '_trained_weights_loaded', False)
        
        # Quick parameter comparison to verify models are different
        sophisticated_param_sum = sum(p.sum().item() for p in sophisticated_model.parameters())
        trained_param_sum = sum(p.sum().item() for p in trained_model.parameters())
        print(f"🔍 Parameter sum comparison:")
        print(f"   Stage 2 (untrained): {sophisticated_param_sum:.6f}")
        print(f"   Stage 3 (trained):   {trained_param_sum:.6f}")
        
        params_identical = abs(sophisticated_param_sum - trained_param_sum) < 1e-6
        
        if params_identical and not trained_weights_loaded:
            print("❌ CONFIRMED: No trained weights loaded - Stage 3 is identical to Stage 2!")
            print("💡 SOLUTION: Provide a valid trained model file or skip Stage 3 evaluation")
            print("📁 Expected model path:", args.model_path)
            print("⚠️ This explains why Stage 3 performance might be worse than Stage 2")
        elif params_identical:
            print("⚠️ WARNING: Parameters are nearly identical despite successful loading!")
            print("⚠️ Stage 3 results may be unreliable")
        elif trained_weights_loaded:
            print("✅ Parameters are different - trained weights successfully loaded")
        else:
            print("⚠️ WARNING: Parameters differ but trained weights loading failed")
            print("⚠️ Stage 3 may show unexpected results")
            
        trained_metrics = comprehensive_evaluation_single_model(
            trained_model, test_examples, annotations, "trained_sophisticated", config,
            batch_size=1, show_examples=args.show_examples
        )
        stage_3_time = time.time() - stage_3_start
        print(f"🏆 Stage 3 completed in {stage_3_time:.1f}s")
        
        # Enhanced comparison printing - ORIGINAL
        print_detailed_four_stage_comparison(frozen_encoder_metrics, simple_metrics, sophisticated_metrics, trained_metrics)
        

        
        # Summary
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"📊 PROTRIX EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        # ORIGINAL METRICS SUMMARY
        print(f"🔥 Stage 0 (Frozen): AP={frozen_encoder_metrics['average_precision']:.3f}, Acc={frozen_encoder_metrics['overall_accuracy']:.3f}")
        print(f"🎯 Stage 1 (Simple): AP={simple_metrics['average_precision']:.3f}, Acc={simple_metrics['overall_accuracy']:.3f}")
        print(f"🚀 Stage 2 (Sophisticated): AP={sophisticated_metrics['average_precision']:.3f}, Acc={sophisticated_metrics['overall_accuracy']:.3f}")
        print(f"🏆 Stage 3 (Trained): AP={trained_metrics['average_precision']:.3f}, Acc={trained_metrics['overall_accuracy']:.3f}")
        
        print(f"\n📈 IMPROVEMENTS (PRIMARY METRIC: Average Precision):")
        cross_attention_benefit = simple_metrics['average_precision'] - frozen_encoder_metrics['average_precision']
        architecture_benefit = sophisticated_metrics['average_precision'] - simple_metrics['average_precision']
        training_benefit = trained_metrics['average_precision'] - sophisticated_metrics['average_precision']
        total_benefit = trained_metrics['average_precision'] - frozen_encoder_metrics['average_precision']
        
        print(f"   🔥→🎯 Cross-Attention benefit: +{cross_attention_benefit:.3f}")
        print(f"   🎯→🚀 Architecture benefit: +{architecture_benefit:.3f}")
        print(f"   🚀→🏆 Training benefit: +{training_benefit:.3f}")
        print(f"   🔥→🏆 Total benefit: +{total_benefit:.3f}")
        
        print(f"\n📈 F1 SCORE PROGRESSION:")
        print(f"   🔥→🎯 Dynamic F1 improvement: +{simple_metrics['dynamic_f1'] - frozen_encoder_metrics['dynamic_f1']:.3f}")
        print(f"   🎯→🚀 Dynamic F1 improvement: +{sophisticated_metrics['dynamic_f1'] - simple_metrics['dynamic_f1']:.3f}")
        print(f"   🚀→🏆 Dynamic F1 improvement: +{trained_metrics['dynamic_f1'] - sophisticated_metrics['dynamic_f1']:.3f}")
        print(f"   🔥→🏆 Total Dynamic F1 improvement: +{trained_metrics['dynamic_f1'] - frozen_encoder_metrics['dynamic_f1']:.3f}")
        

        
        # Create comprehensive visualizations - RESTORED
        print(f"\n{'='*60}")
        print(f"📊 CREATING COMPREHENSIVE VISUALIZATIONS...")
        print(f"{'='*60}")
        viz_start = time.time()
        with tqdm(total=4, desc="Creating visualizations") as pbar:
            pbar.set_description("📈 Four-stage comparison")
            create_comprehensive_four_stage_visualizations(frozen_encoder_metrics, simple_metrics, sophisticated_metrics, trained_metrics, output_dir)
            pbar.update(1)
            
            pbar.set_description("📊 ROC/PR curves")
            create_four_stage_roc_pr_curves(frozen_encoder_metrics, simple_metrics, sophisticated_metrics, trained_metrics, output_dir)
            pbar.update(1)
            
            pbar.set_description("📈 Stage progression")
            create_stage_progression_analysis(frozen_encoder_metrics, simple_metrics, sophisticated_metrics, trained_metrics, output_dir)
            pbar.update(1)

            pbar.set_description("📊 Prediction breakdown")
            create_prediction_breakdown_plot(frozen_encoder_metrics, simple_metrics, sophisticated_metrics, trained_metrics, output_dir)
            pbar.update(1)

        viz_time = time.time() - viz_start
        print(f"📊 Visualizations created successfully (⏱️  {viz_time:.1f}s)")
        
        # Save results
        results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'test_file': args.test_file,
                'annotation_file': args.annotation_file,
                'model_name': model_name,
                'device': str(device),
                'total_time': total_time,
                'evaluation_type': 'enhanced_4_stage_protrix_analysis',
                'features': [
                    'frozen_encoder_baseline',
                    'enhanced_accuracy_metrics',
                    'proper_tp_fp_fn_calculations',
                    'top_all_metrics',
                    'comprehensive_visualizations'
                ]
            },
            'stage_0_frozen_encoder': frozen_encoder_metrics,
            'stage_1_simple': simple_metrics,
            'stage_2_sophisticated': sophisticated_metrics,
            'stage_3_trained': trained_metrics,
            'improvements': {
                'cross_attention_benefit': cross_attention_benefit,
                'architecture_benefit': architecture_benefit,
                'training_benefit': training_benefit,
                'total_benefit': total_benefit,
                # Dynamic F1 score improvements
                'f1_cross_attention_benefit': simple_metrics['dynamic_f1'] - frozen_encoder_metrics['dynamic_f1'],
                'f1_architecture_benefit': sophisticated_metrics['dynamic_f1'] - simple_metrics['dynamic_f1'],
                'f1_training_benefit': trained_metrics['dynamic_f1'] - sophisticated_metrics['dynamic_f1'],
                'f1_total_benefit': trained_metrics['dynamic_f1'] - frozen_encoder_metrics['dynamic_f1']
            },
            'enhanced_metrics': {
                'average_precision_progression': [
                    frozen_encoder_metrics['average_precision'],
                    simple_metrics['average_precision'],
                    sophisticated_metrics['average_precision'],
                    trained_metrics['average_precision']
                ],
                'overall_accuracy_progression': [
                    frozen_encoder_metrics['overall_accuracy'],
                    simple_metrics['overall_accuracy'],
                    sophisticated_metrics['overall_accuracy'],
                    trained_metrics['overall_accuracy']
                ],
                'f1_5_progression': [
                    frozen_encoder_metrics['f1_score_at_5'],
                    simple_metrics['f1_score_at_5'],
                    sophisticated_metrics['f1_score_at_5'],
                    trained_metrics['f1_score_at_5']
                ],
                'ndcg_all_progression': [
                    frozen_encoder_metrics['metrics_at_all']['ndcg'],
                    simple_metrics['metrics_at_all']['ndcg'],
                    sophisticated_metrics['metrics_at_all']['ndcg'],
                    trained_metrics['metrics_at_all']['ndcg']
                ],
                'dynamic_f1_progression': [
                    frozen_encoder_metrics['dynamic_f1'],
                    simple_metrics['dynamic_f1'],
                    sophisticated_metrics['dynamic_f1'],
                    trained_metrics['dynamic_f1']
                ],
            }
        }
        
        # Save to JSON
        results_file = os.path.join(output_dir, "protrix_evaluation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"\n✅ Results saved to: {results_file}")
        print(f"⏱️ Total evaluation time: {total_time:.1f}s")
        
        # Print timing breakdown
        print(f"\n⏱️  TIMING SUMMARY:")
        print(f"   Stage 0 (Frozen): {stage_0_time:.1f}s")
        print(f"   Stage 1 (Simple): {stage_1_time:.1f}s")
        print(f"   Stage 2 (Sophisticated): {stage_2_time:.1f}s")
        print(f"   Stage 3 (Trained): {stage_3_time:.1f}s")
        print(f"   Visualization Creation: {viz_time:.1f}s")
        print(f"   📊 TOTAL EVALUATION TIME: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        # Enhanced conclusion
        print(f"\n📋 EXECUTIVE SUMMARY:")
        effectiveness_score = sum([
            cross_attention_benefit > 0.02,  # AP improvement > 2%
            architecture_benefit > 0.02,     # AP improvement > 2%
            training_benefit > 0.02,          # AP improvement > 2%
            total_benefit > 0.05              # Total AP improvement > 5%
        ])
        print(f"🎯 Training Effectiveness: {effectiveness_score}/4 key improvements achieved (Average Precision)")
        
        # Determine best component based on AP
        best_component = "Cross-Attention" if cross_attention_benefit == max(cross_attention_benefit, architecture_benefit, training_benefit) else "Architecture" if architecture_benefit == max(cross_attention_benefit, architecture_benefit, training_benefit) else "Training"
        print(f"📈 Best Component: {best_component} (Average Precision improvement)")
        print(f"🔍 4-Stage Progression: Frozen→Simple→Sophisticated→Trained")
        
        # Assessment based on Average Precision
        if total_benefit > 0.05:
            print("✅ Conclusion: Significant improvements achieved through model development (AP > 5%)")
        elif total_benefit > 0.02:
            print("✅ Conclusion: Moderate improvements - training is beneficial (AP > 2%)")
        else:
            print("⚠️ Conclusion: Limited improvements - requires further optimization (AP < 2%)")
        
        print(f"\n📁 All evaluation outputs saved to: {output_dir}/")
        print("   Files created:")
        print("   - protrix_evaluation_results.json (complete metrics)")
        print("   - comprehensive_four_stage_comparison.png (4-stage comparison charts)")
        print("   - four_stage_roc_pr_curves.png (ROC/PR curve analysis)")
        print("   - detailed_stage_progression_analysis.png (step-by-step analysis)")
        print("   - prediction_breakdown.png (TP/FP/FN breakdown)")
        if copy_success:
            print("   - training_config.json (model configuration)")
            print("   - [model].pt (model weights)")
            print("   - args.json (training arguments)")
            print("   📋 Configuration tracking enabled for reproducibility")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
