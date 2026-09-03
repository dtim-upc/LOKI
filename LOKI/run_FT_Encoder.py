# Import Unsloth encoder integration and torch.compile for faster training/inference
from unsloth_encoder import (
    # Unsloth for encoder optimization
    UNSLOTH_AVAILABLE,
    FAST_SENTENCE_TRANSFORMER_AVAILABLE,  # Flag for preferred API
    create_unsloth_sentence_encoder,
    get_unsloth_status,
    print_unsloth_status,
    get_model_max_seq_length,  # Auto-detect max_seq_length from model config
    # torch.compile for custom module optimization
    TORCH_COMPILE_AVAILABLE,
    compile_custom_modules,
    optimize_model_for_inference,
    optimize_model_for_training,
    print_optimization_status,
)

import os
import sys

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import datetime
import warnings
import traceback
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import random
import contextlib
import numpy as np

import torch

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

from sentence_transformers import SentenceTransformer
from models import TableTextEmbeddingModel, BidirectionalTableTextModel
from data import (
    load_row_level_dataset,
    IdBasedEmbeddingCache,
    _extract_rows_robust,
    _extract_sentences_robust,
    _extract_table_cell_texts,
    _extract_table_rows_for_model,
    _extract_table_schema_text,
    _normalize_schema_texts,
)
from train import train_with_id_based_triplets
from evaluate import evaluate_with_cache, save_evaluation_results, evaluate_model
from utils import GPUMemoryManager
from losses import IdBasedCachedTripletLoss, EnhancedTripletLoss, BidirectionalTripletLoss

# Import visualization functions from visualize_attention.py
from visualize_attention import visualize_models_with_three_way_comparison, visualize_attention_matrix
from encoding import build_id_based_embedding_cache
# Import clean visualization system
from new_visualization import run_clean_analysis_for_examples
# Import initialization system
from initialization import get_available_methods, get_method_description, get_recommended_method_params

# Set environment variables and disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')


# 1. Tell Python's core warning system to ignore everything
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# 2. Block native C/C++ logging warnings from Hugging Face/PyTorch backends
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"  # Silences bitsandbytes startup text


# Repo root (parent of LOKI/). Dataset defaults resolve here, not from process CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASETS_DIR = _PROJECT_ROOT / "Datasets"


def setup_output_dir(args, model_name, embedding_dim=None, max_seq_length=None) -> str:
    """Setup output directory for model checkpoint and logs."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = model_name.split('/')[-1]  # Get the model name without path

    # Add architecture type to output directory name
    arch_suffix = "bidirectional" if args.use_bidirectional else "unidirectional"
    output_dir = Path(args.output_dir) / f"{arch_suffix}_{model_name_short}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of args and model_name (will be updated with embedding info later)
    config_dict = vars(args).copy()
    config_dict['model_name'] = model_name
    config_dict['architecture'] = arch_suffix

    # Add embedding info if available
    if embedding_dim is not None:
        config_dict['embedding_dim'] = embedding_dim
    if max_seq_length is not None:
        config_dict['max_seq_length'] = max_seq_length

    with open(output_dir / "args.json", "w", encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    return str(output_dir)

def _batch_schema_embedding(schema_embedding: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if schema_embedding is None:
        return None
    if schema_embedding.dim() == 1:
        schema_embedding = schema_embedding.unsqueeze(0)
    if schema_embedding.dim() == 2:
        schema_embedding = schema_embedding.unsqueeze(0)
    return schema_embedding.to(device=device, dtype=dtype)


def _batch_cell_embedding(cell_embedding: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if cell_embedding is None:
        return None
    if cell_embedding.dim() == 3:
        cell_embedding = cell_embedding.unsqueeze(0)
    return cell_embedding.to(device=device, dtype=dtype)


def _encode_schema_texts(
    model: BidirectionalTableTextModel,
    schema_texts: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    normalized_texts = _normalize_schema_texts(schema_texts)
    if not normalized_texts:
        return None
    schema_embeddings = model.encode_sentences(
        normalized_texts,
        batch_size=min(batch_size, len(normalized_texts)),
    )
    return schema_embeddings.to(device=device, dtype=dtype)


def _encode_cell_text_rows(
    model: BidirectionalTableTextModel,
    cell_text_rows: List[List[str]],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if not cell_text_rows:
        return None

    num_rows = len(cell_text_rows)
    max_cols = max((len(row) for row in cell_text_rows), default=0)
    if max_cols == 0:
        return None

    flat_texts: List[str] = []
    flat_positions = []
    for row_index, row_cells in enumerate(cell_text_rows):
        padded_cells = list(row_cells) + [""] * max(0, max_cols - len(row_cells))
        for col_index, cell_text in enumerate(padded_cells[:max_cols]):
            normalized_text = str(cell_text).strip()
            if not normalized_text:
                continue
            flat_texts.append(normalized_text)
            flat_positions.append((row_index, col_index))

    if not flat_texts:
        return None

    flat_embeddings = model.encode_sentences(
        flat_texts,
        batch_size=min(batch_size, len(flat_texts)),
    ).to(device=device, dtype=dtype)
    cell_grid = torch.zeros(num_rows, max_cols, flat_embeddings.shape[-1], device=device, dtype=dtype)
    for embedding_index, (row_index, col_index) in enumerate(flat_positions):
        cell_grid[row_index, col_index] = flat_embeddings[embedding_index]
    return cell_grid

def _forward_bidirectional_with_optional_structure(
    model: BidirectionalTableTextModel,
    row_tensor: torch.Tensor,
    candidate_tensor: torch.Tensor,
    aggregation_method: str,
    row_schema_tensor: Optional[torch.Tensor] = None,
    candidate_schema_tensor: Optional[torch.Tensor] = None,
    row_cell_tensor: Optional[torch.Tensor] = None,
    candidate_cell_tensor: Optional[torch.Tensor] = None,
):
    model_kwargs = {
        'aggregation_method': aggregation_method,
    }
    if getattr(model, 'use_header_conditioning', False):
        model_kwargs['rows_schema_embeddings'] = row_schema_tensor
        model_kwargs['sentences_schema_embeddings'] = candidate_schema_tensor
    if getattr(model, 'use_cell_level_matching', False):
        model_kwargs['rows_cell_embeddings'] = row_cell_tensor
        model_kwargs['sentences_cell_embeddings'] = candidate_cell_tensor
    return model(row_tensor, candidate_tensor, **model_kwargs)


def evaluate_bidirectional_with_join_paths(model: BidirectionalTableTextModel,
                                           examples: List[Dict[str, Any]],
                                           id_cache: Optional[IdBasedEmbeddingCache] = None,
                                           batch_size: int = 16,
                                           aggregation_method: str = "top_k_pairs",
                                           join_path_threshold: float = 0.1,
                                           save_join_paths: bool = False,
                                           output_dir: str = None,
                                           evaluation_margin: float = 0.0) -> Dict[str, float]:
    """
    Evaluate the bidirectional model with join path extraction capabilities.
    """
    print("Evaluating bidirectional model with join path extraction...")
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    use_header_conditioning = bool(getattr(model, 'use_header_conditioning', False))
    use_cell_level_matching = bool(getattr(model, 'use_cell_level_matching', False))

    # Note: Cache building is now handled by the caller to respect cache settings
    # The function can work with or without cache

    total_comparisons = 0
    correct_predictions = 0
    join_paths_extracted = []

    with torch.no_grad():
        # Process each example
        for example_idx, example in enumerate(tqdm(examples, desc="Evaluating examples")):
            anchor_id = example.get("anchor_id")
            if anchor_id is None:
                continue

            is_flipped = "anchor_sentences" in example
            
            if is_flipped:
                rows = _extract_sentences_robust(example.get("anchor_sentences", []))
                row_schema_text = None
                row_cell_text_rows = None
            else:
                rows = _extract_table_rows_for_model(example, use_header_conditioning=use_header_conditioning)
                row_schema_text = _extract_table_schema_text(example) if use_header_conditioning else None
                row_cell_text_rows = _extract_table_cell_texts(example) if use_cell_level_matching else None

            if not rows:
                continue  # Skip this example if no valid rows found

            # Get row embeddings from cache or compute on-the-fly
            if id_cache is not None:
                if not is_flipped:
                    row_embeddings = id_cache.get_table_embeddings(anchor_id)
                    row_schema = id_cache.get_table_schema_embedding(anchor_id) if use_header_conditioning else None
                    row_cells = id_cache.get_table_cell_embeddings(anchor_id) if use_cell_level_matching else None
                else:
                    row_embeddings = id_cache.get_context_embeddings(anchor_id)
                    row_schema = None
                    row_cells = None
                    
                if row_embeddings is None:
                    continue
            else:
                # Compute embeddings on-the-fly when cache is disabled
                row_embeddings = model.encode_sentences(rows, batch_size=batch_size)
                row_schema = _encode_schema_texts(model, row_schema_text, batch_size, device, model_dtype)
                row_cells = _encode_cell_text_rows(model, row_cell_text_rows or [], batch_size, device, model_dtype)

            # Add batch dimension for rows
            row_tensor = row_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
            row_schema_tensor = _batch_schema_embedding(row_schema, device, model_dtype)
            row_cell_tensor = _batch_cell_embedding(row_cells, device, model_dtype)

            # Collect all positive contexts (primary + additional)
            all_positives = []

            # Add primary positive
            primary_positive = example.get("primary_positive", {})
            if primary_positive.get("id") is not None:
                all_positives.append(primary_positive)

            # Check if additional_positives exists and is not empty
            additional_positives = example.get("additional_positives", [])
            if additional_positives:  # Only process if not empty
                for add_pos in additional_positives:
                    if add_pos.get("id") is not None:
                        all_positives.append(add_pos)

            if not all_positives:
                continue

            # Process all positive contexts against all negatives
            for positive in all_positives:
                positive_id = positive.get("id")
                if is_flipped:
                    positive_sentences = _extract_table_rows_for_model(positive, use_header_conditioning=use_header_conditioning)
                    positive_schema_text = _extract_table_schema_text(positive) if use_header_conditioning else None
                    positive_cell_text_rows = _extract_table_cell_texts(positive) if use_cell_level_matching else None
                else:
                    positive_sentences = _extract_sentences_robust(positive.get("sentences", []))
                    if not positive_sentences:
                        positive_sentences = _extract_rows_robust(positive)
                    positive_schema_text = None
                    positive_cell_text_rows = None

                if positive_id is None or not positive_sentences:
                    continue

                # Get positive embeddings from cache or compute on-the-fly
                if id_cache is not None:
                    if not is_flipped:
                        positive_embeddings = id_cache.get_context_embeddings(positive_id)
                        positive_schema = None
                        positive_cells = None
                    else:
                        positive_embeddings = id_cache.get_table_embeddings(positive_id)
                        positive_schema = id_cache.get_table_schema_embedding(positive_id) if use_header_conditioning else None
                        positive_cells = id_cache.get_table_cell_embeddings(positive_id) if use_cell_level_matching else None
                        
                    if positive_embeddings is None:
                        continue
                else:
                    positive_embeddings = model.encode_sentences(positive_sentences, batch_size=batch_size)
                    positive_schema = _encode_schema_texts(model, positive_schema_text, batch_size, device, model_dtype)
                    positive_cells = _encode_cell_text_rows(model, positive_cell_text_rows or [], batch_size, device, model_dtype)

                # Add batch dimension
                positive_tensor = positive_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
                positive_schema_tensor = _batch_schema_embedding(positive_schema, device, model_dtype)
                positive_cell_tensor = _batch_cell_embedding(positive_cells, device, model_dtype)

                # Get similarity score and pair scores for positive
                positive_similarity, positive_pair_scores = _forward_bidirectional_with_optional_structure(
                    model,
                    row_tensor,
                    positive_tensor,
                    aggregation_method,
                    row_schema_tensor=row_schema_tensor,
                    candidate_schema_tensor=positive_schema_tensor,
                    row_cell_tensor=row_cell_tensor,
                    candidate_cell_tensor=positive_cell_tensor,
                )
                positive_similarity = positive_similarity.item()

                # Extract join paths for the first positive only (to avoid duplication)
                if positive == all_positives[0] and save_join_paths:
                    join_paths = model.extract_join_paths(
                        positive_pair_scores, rows, positive_sentences,
                        threshold=join_path_threshold, top_k=model.top_k
                    )

                    if join_paths:
                        join_paths_extracted.append({
                            'example_idx': example_idx,
                            'anchor_id': anchor_id,
                            'positive_id': positive_id,
                            'join_paths': join_paths,
                            'rows': rows,
                            'sentences': positive_sentences
                        })

                # Process each negative
                for negative in example["negatives"]:
                    negative_id = negative.get("id")
                    if is_flipped:
                        negative_sentences = _extract_table_rows_for_model(negative, use_header_conditioning=use_header_conditioning)
                        negative_schema_text = _extract_table_schema_text(negative) if use_header_conditioning else None
                        negative_cell_text_rows = _extract_table_cell_texts(negative) if use_cell_level_matching else None
                    else:
                        negative_sentences = _extract_sentences_robust(negative.get("sentences", []))
                        if not negative_sentences:
                            negative_sentences = _extract_rows_robust(negative)
                        negative_schema_text = None
                        negative_cell_text_rows = None

                    if negative_id is None or not negative_sentences:
                        continue

                    # Get negative embeddings from cache or compute on-the-fly
                    if id_cache is not None:
                        if not is_flipped:
                            negative_embeddings = id_cache.get_context_embeddings(negative_id)
                            negative_schema = None
                            negative_cells = None
                        else:
                            negative_embeddings = id_cache.get_table_embeddings(negative_id)
                            negative_schema = id_cache.get_table_schema_embedding(negative_id) if use_header_conditioning else None
                            negative_cells = id_cache.get_table_cell_embeddings(negative_id) if use_cell_level_matching else None
                            
                        if negative_embeddings is None:
                            continue
                    else:
                        negative_embeddings = model.encode_sentences(negative_sentences, batch_size=batch_size)
                        negative_schema = _encode_schema_texts(model, negative_schema_text, batch_size, device, model_dtype)
                        negative_cells = _encode_cell_text_rows(model, negative_cell_text_rows or [], batch_size, device, model_dtype)

                    negative_tensor = negative_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
                    negative_schema_tensor = _batch_schema_embedding(negative_schema, device, model_dtype)
                    negative_cell_tensor = _batch_cell_embedding(negative_cells, device, model_dtype)

                    # Get similarity score for negative
                    negative_similarity, _ = _forward_bidirectional_with_optional_structure(
                        model,
                        row_tensor,
                        negative_tensor,
                        aggregation_method,
                        row_schema_tensor=row_schema_tensor,
                        candidate_schema_tensor=negative_schema_tensor,
                        row_cell_tensor=row_cell_tensor,
                        candidate_cell_tensor=negative_cell_tensor,
                    )
                    negative_similarity = negative_similarity.item()

                    # Check if positive similarity is higher than negative (with optional margin)
                    # With margin: pos_score > neg_score + margin (more conservative)
                    # Without margin (0.0): pos_score > neg_score (current behavior)
                    correct_predictions += 1 if positive_similarity > (negative_similarity + evaluation_margin) else 0
                    total_comparisons += 1

    # Save join paths if requested
    if save_join_paths and join_paths_extracted and output_dir:
        # Save join paths to visualizations folder if it exists, otherwise output_dir
        viz_dir = Path(output_dir) / "visualizations"
        if viz_dir.exists():
            join_paths_file = viz_dir / "extracted_join_paths.json"
        else:
            join_paths_file = Path(output_dir) / "extracted_join_paths.json"
        with open(join_paths_file, 'w', encoding='utf-8') as f:
            json.dump(join_paths_extracted, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(join_paths_extracted)} join path examples to {join_paths_file}")

    # Calculate metrics
    accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0

    return {
        'accuracy': accuracy,
        'total_comparisons': total_comparisons,
        'join_paths_extracted': len(join_paths_extracted)
    }


def main():
    """Main function to parse arguments and run the cross-attention model."""
    AVAILABLE_MODELS = {
        "roberta-large": "sentence-transformers/all-roberta-large-v1",
        "modernbert-base": "unsloth/ModernBERT-base",
        "modernbert-large": "unsloth/ModernBERT-large",
        "nomic-embed-v2": "nomic-ai/nomic-embed-text-v2-moe",
        "nomic-modernbert": "nomic-ai/modernbert-embed-base",
        "jina-v5": "jinaai/jina-embeddings-v5-text-small",
        "MiniLM": "unsloth/all-MiniLM-L6-v2",
        "embedding-gemma": "unsloth/embeddinggemma-300m",
        "qwen3": "unsloth/Qwen3-Embedding-0.6B",
        # Medical Embedding
        "qwen3-med": "luluw/Qwen3-MedEmbed-0.6B",
        "google": "sentence-transformers/embeddinggemma-300m-medical",
        "medembed-small": "abhinand/MedEmbed-small-v0.1",
        "medembed-large": "abhinand/MedEmbed-large-v0.1",
        "google-med-gemma": "vectorranger/embeddinggemma-300m-medical-300k",
        # PubMed Embeddings - for CMDL comparison
        "pubmed-base": "neuml/pubmedbert-base-embeddings",
        "cmdl_model": "allenai/biomed_roberta_base"
    }


    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Cross-Attention Model for Join Path Discovery")
    parser.add_argument("--model_key", type=str, default="medembed-large",
                        choices=list(AVAILABLE_MODELS.keys()),
                        help=f"Key to select model from available models: {list(AVAILABLE_MODELS.keys())}")

    # NEW: Architecture selection
    parser.add_argument("--use_bidirectional", action="store_true", default=True,
                        help="Use bidirectional cross-attention architecture for direct join path discovery")
    parser.add_argument("--no_bidirectional", dest="use_bidirectional", action="store_false",
                        help="Use unidirectional cross-attention architecture (disable bidirectional mode)")

    # Attention direction for unidirectional model
    parser.add_argument("--attention_direction", type=str, default="row_to_sentence",
                        choices=["row_to_sentence", "sentence_to_row"],
                        help="Attention direction for unidirectional model: row_to_sentence (rows query sentences) or sentence_to_row (sentences query rows)")

    parser.add_argument("--use_flash_attention", action="store_true", default=False,
                        help="Whether to use Flash Attention for better performance")

    # Unsloth integration for 2x faster trainings
    parser.add_argument("--use_unsloth", action="store_true", default=True,
                        help="Use Unsloth for 2x faster training with optimized kernels. Requires unsloth package.")
    parser.add_argument("--no_unsloth", dest="use_unsloth", action="store_false",
                        help="Disable Unsloth optimizations (use standard SentenceTransformer loading)")
    # Keep this True if you are using full fine-tuning of the Encoder.
    parser.add_argument("--unsloth_full_finetuning", action="store_true", default=False,
                        help="Enable full fine-tuning of the Unsloth encoder (overrides QLoRA if both are set)")
    parser.add_argument("--unsloth_4bit", action="store_true", default=False,
                        help="Load encoder in 4-bit quantization for memory efficiency (QLoRA)")
    parser.add_argument("--no_unsloth_4bit", dest="unsloth_4bit", action="store_false",
                        help="Disable 4-bit quantization (use full precision)")
    # Keep this False if you are using full fine-tuning
    parser.add_argument("--unsloth_qlora", action="store_true", default=False,
                        help="Apply QLoRA adapters to encoder via Unsloth (efficient encoder fine-tuning)")
    parser.add_argument("--no_unsloth_qlora", dest="unsloth_qlora", action="store_false",
                        help="Disable QLoRA adapters (keep encoder fully frozen)")
    parser.add_argument("--unsloth_qlora_rank", type=int, default=32,
                        help="Rank for Unsloth LoRA adapters (default: 32, as per tutorial)")
    parser.add_argument("--unsloth_qlora_alpha", type=float, default=64.0,
                        help="Alpha for Unsloth LoRA adapters (rule of thumb: 2x rank, default: 64)")
    parser.add_argument("--unsloth_target_modules", type=str, default="auto",
                        help="Comma-separated list of modules to apply LoRA to, or 'auto' for automatic detection. "
                             "Auto-detection uses: BERT/RoBERTa: query,key,value,dense; "
                             "ModernBERT: Wqkv,Wo,Wi; LLaMA-style: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--unsloth_pooling_mode", type=str, default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling mode for Unsloth encoder (default: mean)")

    # Dimension and sequence length overrides for OOM control
    parser.add_argument("--override_embedding_dim", type=int, default=0,
                        help="Override the auto-detected embedding dimension. Uses Matryoshka truncation "
                             "if the model supports it, otherwise adds a linear projection layer. "
                             "Common Matryoshka sizes: 64, 128, 256, 512, 768. 0 = use native dimension (default).")
    parser.add_argument("--override_max_seq_length", type=int, default=0,
                        help="Override the auto-detected maximum sequence length. "
                             "Lower values reduce memory usage. 0 = use native value (default).")

    # Training stage control
    parser.add_argument("--start_training_from_stage", type=int, default=0, choices=[0, 1],
                        help="Choose which stage to start training from: 0 (Encoder-only fine-tuning) or 1 (Sophisticated Model, default).")

    # Encoder training control - add both positive and negative options for clarity
    encoder_group = parser.add_mutually_exclusive_group()
    encoder_group.add_argument("--enable_lora", action="store_true", default=True,
                               help="Fine-tune the encoder (full fine-tuning). Note: training cache will be disabled when enabled so encoder updates take effect.")
    encoder_group.add_argument("--disable_lora", action="store_true", default=False,
                               help="Explicitly disable encoder fine-tuning (keep encoder frozen). Useful for cross-attention only training.")

    # Encoder-only training (baseline without cross-attention, for comparison when the above is 0)
    parser.add_argument("--encoder_only_training", action="store_true", default=True,
                        help="Train only the base sentence encoder using encoder-only triplet objective (no cross-attention)")
    parser.add_argument("--no_encoder_only_training", dest="encoder_only_training", action="store_false",
                        help="Skip encoder-only Stage 0 training (start directly with sophisticated model)")

    # Caching control
    parser.add_argument("--use_cache", action="store_true", default=False,
                        help="Enable embedding caching for improved performance (significantly faster training). "
                             "Automatically disabled if --enable_lora is set (encoder fine-tuning).")

    # Stage 0 cache control (override when --use_cache=True)
    parser.add_argument("--stage0_cache_mode", type=str, choices=["auto", "on", "off"], default="off",
                        help="Override Stage 0 cache behavior when --use_cache=True: 'auto' (current logic), 'on' (force keep caches), 'off' (force disable caches)")

    # Encoder tuning mode (full vs gradual unfreezing) - This will be used to fine-tune the encoder during Stage 0 encoder-only training.
    parser.add_argument("--encoder_tuning_mode", type=str, default="gradual", choices=["full", "gradual"],
                        help="How to fine-tune the encoder during Stage 0 encoder-only training: 'full' trains all layers; 'gradual' performs layer-wise unfreezing from top to bottom.")
    parser.add_argument("--gradual_unfreeze_initial_layers", type=int, default=2,
                        help="Number of top encoder layers to train at the start when using --encoder_tuning_mode gradual")
    parser.add_argument("--gradual_unfreeze_every", type=int, default=1,
                        help="Unfreeze this many additional top layers every N epochs when using gradual unfreezing")
    parser.add_argument("--gradual_unfreeze_max_layers", type=int, default=6,
                        help="Maximum number of top layers to unfreeze (0 = all detected layers)")
    parser.add_argument("--gradual_unfreeze_include_pooler", action="store_true", default=True,
                        help="Also keep the pooler/head of the encoder trainable during gradual unfreezing if present")

    # Dataset paths (under repo Datasets/<name>/; anchored to project root, not CWD)
    parser.add_argument("--train_file", type=str, default=str(_DATASETS_DIR / "feverous" / "train_row_level.json"),
                        help="Path to the training dataset")
    parser.add_argument("--eval_file", type=str, default=str(_DATASETS_DIR / "feverous" / "val_row_level.json"),
                        help="Path to the evaluation dataset")
    parser.add_argument("--test_file", type=str, default=str(_DATASETS_DIR / "feverous" / "test_row_level.json"),
                        help="Path to the test dataset (optional)")
    parser.add_argument("--output_dir", type=str, default="./output_feverous_model/",
                        help="Directory to save the output")
    
    # NEW: Task-Aware Dataset flags
    parser.add_argument("--task_direction", type=str, default="TABLE_TO_DOC",
                        choices=["TABLE_TO_DOC", "DOC_TO_TABLE"],
                        help="The task the model is performing (default: DOC_TO_TABLE for Pharma)")
    parser.add_argument("--native_direction", type=str, default="TABLE_TO_DOC",
                        choices=["TABLE_TO_DOC", "DOC_TO_TABLE"],
                        help="The native direction of the source file (default: DOC_TO_TABLE for Flipped Pharma)")
    parser.add_argument("--dataset_format", type=str, default="other",
                        choices=["mimic", "other"],
                        help="Parsing format for tables/rows (default: other)")

    # Large dataset optimization
    parser.add_argument("--max_train_examples", type=int, default=0,
                        help="Maximum number of training examples to use (0 = use all). "
                             "Use for faster iteration on large datasets (e.g., 5000 for quick experiments)")
    parser.add_argument("--max_eval_examples", type=int, default=0,
                        help="Maximum number of evaluation examples to use (0 = use all). "
                             "Use for faster evaluation on large val sets")
    parser.add_argument("--eval_every_n_steps", type=int, default=0,
                        help="Evaluate every N training steps instead of every epoch (0 = eval per epoch). "
                             "Useful for large datasets where per-epoch eval is too slow")

    # Epochs and Early stopping and checkpointings
    # Note: With 5000 examples and batch_size=64, need ~10 epochs for convergence
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (10 is good for 5k examples, reduce to 5 for 20k+)")
    parser.add_argument("--early_stopping_patience", type=int, default=20,
                        help="Epochs without validation improvement before early stopping")
    parser.add_argument("--early_stopping_min_epochs", type=int, default=20,
                        help="Do not early-stop before this many epochs")
    parser.add_argument("--enable_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing for memory efficiency")

    # Learning rate and scheduler: Fine-tuning stability controls
    # Note: Larger datasets can handle higher learning rates
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (1e-4 is good for 20k+ examples)")
    parser.add_argument("--encoder_lr", type=float, default=1e-5,
                        help="Encoder learning rate (keep lower than main LR)")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["cosine", "linear", "cosine_restart"],
                        help="Learning rate scheduler type (default: cosine)")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of cycles for cosine-with-restarts")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum LR as a fraction of base LR; enforced as LR floor")

    # Batch size and optimizer - Keep the batch size 10x10 = 100 for Protrix dataset and 8x8 = 64 for Pharma Data.
    # Note: Larger batches = better gradient estimates with more data
    parser.add_argument("--train_batch_size", type=int, default=100,
                        help="Training batch size (increase for larger datasets)")
    parser.add_argument("--eval_batch_size", type=int, default=100,
                        help="Evaluation batch size")
    parser.add_argument("--encoding_batch_size", type=int, default=100,
                        help="Batch size for SentenceTransformer.encode() when building embedding caches. "
                             "Higher values saturate the GPU better (256-512 for 24GB VRAM). "
                             "Reduce if you hit OOM during cache building.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps (0.1 = 10%% warmup for stable training with larger batches)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps (effective batch = 32)")

    # Triplet batching control
    parser.add_argument("--mix_examples", action="store_true", default=False,
                        help="Mix triplets from different examples in batches (better for TOTTO). If False, keep triplets from same example together (better for Protrix)")
    parser.add_argument("--do_mix_examples", dest="mix_examples", action="store_true",
                        help="Explicitly enable mixing examples")

    # Triplet sampling strategy (for memory management)
    parser.add_argument("--triplet_strategy", type=str, default="full",
                        choices=["full", "limited", "random", "balanced", "primary_only"],
                        help="Strategy for generating triplets: 'full' (all posxneg combinations), "
                             "'limited' (round-robin over positives, N total), "
                             "'balanced' (partition negatives equally per positive domain, interleaved), "
                             "'random' (sample N triplets), "
                             "'primary_only' (only 1 pos x 1 neg per example)")
    parser.add_argument("--max_triplets_per_example", type=int, default=10,
                        help="Maximum triplets per example when using 'limited' or 'random' strategy")

    # Default to enhanced_triplet for unidirectional; script will switch to bidirectional_triplet when needed
    parser.add_argument("--loss_type", type=str, default="bidirectional_triplet",
                        choices=['id_cached_triplet', 'enhanced_triplet', 'bidirectional_triplet'],
                        help="Type of loss function to use for training")

    # Optional wandb logging
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Whether to use Weights & Biases for logging")

    # LoRA-specific arguments for cross-attention
    parser.add_argument("--use_cross_attention_lora", action="store_true", default=False,
                        help="Whether to use LoRA for cross-attention layers")
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="Rank for LoRA adaptation (lower = more efficient, higher = more expressive)")
    parser.add_argument("--lora_alpha", type=float, default=512,
                        help="LoRA alpha parameter (scaling factor, typically 2x rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="Dropout rate for LoRA layers")

    # Visualization arguments
    parser.add_argument("--do_visualize", action="store_true", default=True,
                        help="Whether to visualize attention matrices after training")
    parser.add_argument("--do_clean_analysis", action="store_true", default=False,
                        help="Whether to run clean step-by-step analysis to fix inconsistencies")
    parser.add_argument("--visualize_examples", type=str, default="1",
                        help="Comma-separated indices of examples to visualize (default: first example)")
    parser.add_argument("--normalize_attention", action="store_true", default=True,
                        help="Whether to normalize attention scores with softmax (default: False)")
    parser.add_argument("--skip_four_stage_viz", action="store_true", default=True,
                        help="Skip the 4-stage example visualizations (heatmaps for examples 0,1,2). Useful to speed up runs.")

    # Top-K Aggregation arguments
    parser.add_argument("--aggregation_method", type=str, default="top_k_pairs",
                        choices=["mean", "top_k_sum", "top_k_mean", "weighted_top_k",
                                 "max", "attention_weighted", "sparse_top_k", "entropy_regularized",
                                 # NEW: Bidirectional aggregation methods
                                 "top_k_pairs", "max_pairs", "mean_pairs", "weighted_pairs", "sparse_pairs"],
                        help="Aggregation method for row scores (or pair scores for bidirectional)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top elements to consider in top-k aggregation methods (default: 5)")

    # Normalization type
    parser.add_argument("--norm_type", type=str, default="rmsnorm",
                        choices=["layernorm", "rmsnorm"],
                        help="Normalization type to use in attention/FFN blocks")

    # Q/K RMSNorm (Qwen-style) toggle for bidirectional attention
    parser.add_argument("--use_qk_rmsnorm", action="store_true", default=False,
                        help="Apply RMSNorm to Q and K projections (forward and reverse) inside attention")

    # Latent bottleneck (Perceiver-style) arguments
    parser.add_argument("--use_latent_bottleneck", action="store_true", default=False,
                        help="Enable Perceiver-style latent bottleneck before cross-attention to reduce uniformity")
    parser.add_argument("--latent_num", type=int, default=64,
                        help="Number of learnable latents (32-128 recommended)")
    parser.add_argument("--latent_dropout", type=float, default=0.0,
                        help="Dropout to apply on latent attention weights")

    parser.add_argument("--pair_topk_mask", action="store_true", default=False,
                        help="Compute pair contrast only on top-k pairs (bidirectional only)")
    parser.add_argument("--pair_topk_k", type=int, default=0,
                        help="Top-k pairs to use for pair contrastive loss (0 = use --top_k)")
    # Hard negative mining
    parser.add_argument("--use_hard_negative_mining", action="store_true", default=True,
                        help="Enable simple in-batch hard negative mining")
    parser.add_argument("--hard_negative_topk", type=int, default=10,
                        help="Top-k hardest negatives to consider (1 uses single hardest)")

    # Loss function parameters
    # Note: With larger datasets, can use slightly higher margins and scale
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Margin for triplet loss (distance between positive and negative similarities)")
    parser.add_argument("--scale", type=float, default=10.0,
                        help="Scale factor for triplet loss similarities (amplifies differences between positive/negative scores)")
    parser.add_argument("--pair_margin", type=float, default=0.3,
                        help="Margin for pair-wise contrastive loss (how much positive pairs should exceed negative pairs)")
    # Margin scheduling
    parser.add_argument("--margin_end", type=float, default=0.5,
                        help="Target margin at last epoch (linear schedule)")
    parser.add_argument("--margin_schedule", type=str, default="none", choices=["linear", "none"],
                        help="Schedule for margin across epochs")

    parser.add_argument("--ranking_loss_type", type=str, default="infonce",
                        choices=["softplus", "infonce"],
                        help="Ranking loss for triplet component: softplus (smooth hinge) or infonce (NCE)")
    # Note: With larger datasets, slightly higher tau (0.2) provides smoother gradients
    parser.add_argument("--infonce_tau", type=float, default=0.2,
                        help="Temperature for InfoNCE ranking loss")

    # Loss component weights (grouped together for easy tuning)
    # OPTIMIZED: Balanced weights with distillation enabled for better row-sentence alignment
    parser.add_argument("--triplet_weight", type=float, default=0.5,
                        help="Relative weight for triplet loss component (will be normalized with other weights)")
    parser.add_argument("--attention_loss_weight", type=float, default=0.0,
                        help="Relative weight for attention regularization loss (will be normalized with other weights)")
    parser.add_argument("--pair_loss_weight", type=float, default=0.3,
                        help="Relative weight for pair-wise contrastive loss (bidirectional only, will be normalized with other weights)")

    # Attention Distillation arguments (preserves zero-shot row-sentence alignment quality)
    # OPTIMIZED: Enabled by default with 0.2 weight to preserve local alignments during training
    parser.add_argument("--use_attention_distillation", action="store_true", default=True,
                        help="Enable attention distillation from frozen encoder to preserve zero-shot row-sentence alignments during training")
    parser.add_argument("--no_attention_distillation", dest="use_attention_distillation", action="store_false",
                        help="Disable attention distillation")
    parser.add_argument("--distillation_weight", type=float, default=0.2,
                        help="Relative weight for attention distillation loss (will be normalized with other loss weights)")
    parser.add_argument("--teacher_temperature", type=float, default=0.1,
                        help="Temperature for teacher (frozen encoder) pair similarities. Lower = sharper distribution (0.05-0.2 recommended)")
    parser.add_argument("--student_temperature", type=float, default=0.1,
                        help="Temperature for student (LOKI) pair scores. Should match teacher_temperature for consistent KL")
    parser.add_argument("--distillation_loss_type", type=str, default="js_div",
                        choices=["kl_div", "mse", "cosine", "js_div"],
                        help="Type of distillation loss: kl_div (KL divergence), mse (mean squared error), cosine (1 - cosine sim), js_div (Jensen-Shannon)")

    parser.add_argument("--pair_score_method", type=str, default="cosine",
                        choices=["cosine", "dot", "mlp"],
                        help="Method for computing pair scores in bidirectional model: cosine (normalized [-1,1]), dot (raw similarity), or mlp (with attention weights)")
    parser.add_argument("--share_attention_weights", action="store_true", default=True,
                        help="Share weights between forward and reverse attention to prevent attention collapse (bidirectional only)")
    parser.add_argument("--extract_join_paths", action="store_true", default=True,
                        help="Whether to extract join paths during evaluation (bidirectional only)")
    parser.add_argument("--join_path_threshold", type=float, default=0.15,
                        help="Threshold for join path extraction (bidirectional only)")

    parser.add_argument("--use_refinement", action="store_true", default=False,
                        help="Whether to use the refinement (FFN) step after attention in the bidirectional model.")

    # Unidirectional model FFN skip option
    parser.add_argument("--skip_ffn", action="store_true", default=True,
                        help="Skip the FFN layer in unidirectional model - use raw cross-attention output directly")

    # Self-attention arguments
    parser.add_argument("--use_self_attention", action="store_true", default=False,
                        help="Whether to apply self-attention before cross-attention for better attention collapse prevention")
    parser.add_argument("--self_attention_heads", type=int, default=1,
                        help="Number of attention heads for self-attention blocks")
    parser.add_argument("--self_attention_dropout", type=float, default=0.1,
                        help="Dropout rate for self-attention blocks")

    # Attention mechanism type arguments
    parser.add_argument("--attention_type", type=str, default="top_k_sparse",
                        choices=["standard", "top_k_sparse", "windowed", "threshold", "latent_cross"],
                        help="Type of attention mechanism: standard (regular), top_k_sparse (sparse attention), windowed (local attention), threshold (semantic filtering)")
    
    # Optional table-side schema routing: per-column schema sketches (header + representative values)
    # condition table Q/K via gating, while V still comes from value-only row embeddings (bidirectional only).
    parser.add_argument("--use_header_conditioning", action="store_true", default=False,
                        help="Enable table schema conditioning for bidirectional attention. Table headers are encoded separately and used to gate table-side query/key routing while keeping values row-content-driven.")
    parser.add_argument("--no_header_conditioning", dest="use_header_conditioning", action="store_false",
                        help="Disable table schema conditioning (default).")
    parser.add_argument("--use_cell_level_matching", action="store_true", default=False,
                        help="Enable hybrid row-plus-cell matching. Table rows keep their row-level embeddings while per-cell header:value embeddings provide additional support during pair scoring.")
    parser.add_argument("--cell_matching_weight", type=float, default=0.35,
                        help="Interpolation weight for pooled cell support when blending it into the row-level pair matrix.")
    parser.add_argument("--cell_matching_pooling", type=str, default="max", choices=["max", "mean"],
                        help="How to pool per-cell sentence support within a row: max focuses on the strongest field, mean averages all fields.")
    parser.add_argument("--cell_row_fusion_weight", type=float, default=0.15,
                        help="Weight used to fuse pooled cell embeddings back into the table-side row embeddings before cross-attention.")

    # Gated attention overlay (post-SDPA gating), inspired by Gated Attention paper
    parser.add_argument("--use_gated_attention", action="store_true", default=True,
                        help="Enable query-dependent gating on attention outputs (post-SDPA), layered on top of any attention type")
    parser.add_argument("--no_gated_attention", dest="use_gated_attention", action="store_false",
                        help="Disable gated attention overlay")
    parser.add_argument("--gated_attention_mode", type=str, default="vector",
                        choices=["scalar", "vector"],
                        help="Gate shape: scalar (one gate per query) or vector (one gate per feature per query)")
    parser.add_argument("--gated_attention_hidden_dim", type=int, default=0,
                        help="Hidden dim for 2-layer gating MLP (0 => single linear)")
    parser.add_argument("--gated_attention_dropout", type=float, default=0.0,
                        help="Dropout on gate values (after sigmoid)")
    parser.add_argument("--gated_attention_init_bias", type=float, default=6.0,
                        help="Initial bias for gate logits (sigmoid(init_bias) close to 1 for pass-through init)")

    # Temperature scaling control (we'll force-disable when gated attention is enabled)
    parser.add_argument("--disable_temperature", action="store_true", default=True,
                        help="Disable temperature scaling inside attention score computation (useful for clean ablations)")

    parser.add_argument("--attention_activation", type=str, default="softmax",
                        choices=["softmax", "entmax15", "alpha_entmax"],
                        help="Activation for attention probabilities")
    parser.add_argument("--attention_alpha", type=float, default=1.5,
                        help="Alpha parameter for alpha-entmax (ignored for softmax/entmax15)")
    parser.add_argument("--sparse_top_k", type=int, default=5,
                        help="Number of top connections to keep in sparse attention (prevents uniform attention)")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Window size for windowed attention mechanism")
    parser.add_argument("--threshold_base", type=float, default=0.3,
                        help="Base threshold for threshold-based attention filtering")

    # Initialization arguments
    parser.add_argument("--init_method", type=str, default="orthogonal",
                        choices=["xavier_uniform", "kaiming_uniform", "orthogonal", "attention_specific", "t5_style",
                                 "diverse_attention", "multiscale",
                                 "zeros", "ones", "diagonal", "identity_preserving", "sparse_random", "scaled_uniform",
                                 "asymmetric_forward_reverse", "tiny_random"],
                        help="Initialization method for attention weights")
    parser.add_argument("--init_method_params", type=str, default=None,
                        help="JSON string of initialization method parameters (e.g., '{\"weight_value\": 0.01}' for ones method)")
    parser.add_argument("--init_show_descriptions", action="store_true", default=False,
                        help="Show detailed descriptions and parameters of all available initialization methods and exit")

    # Training curves tracking arguments
    parser.add_argument("--enable_training_curves", action="store_true", default=True,
                        help="Enable training curves tracking and visualization (default: True)")
    parser.add_argument("--no_training_curves", dest="enable_training_curves", action="store_false",
                        help="Disable training curves tracking")
    parser.add_argument("--track_batch_losses", action="store_true", default=True,
                        help="Track individual batch losses for detailed analysis (default: True)")
    parser.add_argument("--no_batch_losses", dest="track_batch_losses", action="store_false",
                        help="Disable batch-level loss tracking")
    parser.add_argument("--track_val_loss", action="store_true", default=True,
                        help="Compute and track validation loss (requires additional compute, default: False)")
    parser.add_argument("--auto_plot_curves", action="store_true", default=False,
                        help="Automatically generate and save plots after each epoch (default: False)")
    parser.add_argument("--no_auto_plot", dest="auto_plot_curves", action="store_false",
                        help="Disable automatic plot generation")

    # Row-sentence evaluation arguments
    parser.add_argument("--enable_row_sent_eval", action="store_true", default=True,
                        help="Enable row-sentence level evaluation during training (default: False)")
    parser.add_argument("--row_sent_test_file", type=str, default=str(_DATASETS_DIR / "feverous" / "test_row_level.json"),
                        help="Path to test dataset for row-sentence evaluation")
    parser.add_argument("--row_sent_annotation_file", type=str, default=str(_DATASETS_DIR / "feverous" / "Annotated_Test.json"),
                        help="Path to row-sentence annotation file")
    parser.add_argument("--row_sent_max_examples", type=int, default=None,
                        help="Maximum number of test examples to evaluate per epoch (default: None = all examples)")
    # Save best-by-test metrics checkpoints (requires row-sentence eval)
    parser.add_argument("--save_best_by_test_metrics", action="store_true", default=True,
                        help="Also save checkpoints for best test F1 and best test average precision")

    # Verbosity control
    parser.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level: 0=quiet (results only), 1=normal (default), 2=verbose (all details)")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Shortcut for --verbosity 0 (quiet mode, results only)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Shortcut for --verbosity 2 (verbose mode, all details)")

    # torch.compile optimization for custom modules (BidirectionalCrossAttention, etc.)
    # NOTE: Disabled by default due to CUDA graph memory issues with some PyTorch/GPU configurations
    # Enable with --use_compile if you want to test it on your setup
    parser.add_argument("--use_compile", action="store_true", default=False,
                        help="Use torch.compile() to optimize custom attention modules (experimental, may cause CUDA errors)")
    parser.add_argument("--no_compile", dest="use_compile", action="store_false",
                        help="Disable torch.compile() optimizations (default)")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode: 'default' (balanced), 'reduce-overhead' (fast compile), 'max-autotune' (best runtime)")

    args = parser.parse_args()

    # Process verbosity arguments
    if args.quiet:
        args.verbosity = 0
    elif args.verbose:
        args.verbosity = 2
    # Set verbose flag based on verbosity level (used for model/module initialization)
    args.verbose_flag = args.verbosity >= 1  # Normal or verbose mode
    args.extra_verbose = args.verbosity >= 2  # Only verbose mode

    # Parse JSON initialization method parameters
    if args.init_method_params:
        try:
            args.init_method_params = json.loads(args.init_method_params)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Error parsing init_method_params JSON: {e}")
            print(f"   Provided: {args.init_method_params}")
            print(f"   Example: '{{\"weight_value\": 0.01}}' for ones method")
            sys.exit(1)
    else:
        # Use recommended default parameters for the method
        args.init_method_params = get_recommended_method_params(args.init_method)

    # Show initialization method descriptions if requested
    if args.init_show_descriptions:
        print("\n[INFO] Available Initialization Methods:")
        print("=" * 60)
        for method in get_available_methods():
            description = get_method_description(method)
            print(f"\n{method}:")
            print(f"  Description: {description}")
            params = get_recommended_method_params(method)
            if params:
                print(f"  Default parameters: {params}")
        print("\n" + "=" * 60)
        return

    # NOTE: MIMIC data preprocessing is now done separately using annotation_pipeline/preprocess_mimic.py

    if args.use_header_conditioning and not args.use_bidirectional:
        print("[ERROR] --use_header_conditioning currently supports bidirectional LOKI only.")
        sys.exit(1)
    if args.use_cell_level_matching and not args.use_bidirectional:
        print("[ERROR] --use_cell_level_matching currently supports bidirectional LOKI only.")
        sys.exit(1)

    # Set default aggregation method based on architecture
    if args.use_bidirectional and args.aggregation_method in ["mean", "top_k_sum", "top_k_mean", "weighted_top_k",
                                                              "max", "attention_weighted", "sparse_top_k",
                                                              "entropy_regularized"]:
        print(
            f"[WARN]  Switching aggregation method from '{args.aggregation_method}' to 'top_k_pairs' for bidirectional architecture")
        args.aggregation_method = "top_k_pairs"

    # Set default loss type based on architecture
    if args.use_bidirectional and args.loss_type in ["id_cached_triplet", "enhanced_triplet"]:
        print(
            f"[WARN]  Switching loss type from '{args.loss_type}' to 'bidirectional_triplet' for bidirectional architecture")
        args.loss_type = "bidirectional_triplet"

    # Resolve encoder training setting from mutually exclusive arguments.
    # These entrypoints use different parser defaults, so the positive flag is
    # the only reliable signal that encoder fine-tuning was requested.
    enable_encoder_training = bool(args.enable_lora)

    # =====================================================================
    # IMPORTANT: Auto-adjust incompatible 4-bit encoder fine-tuning configs
    # =====================================================================
    # A 4-bit Unsloth encoder cannot be fine-tuned by directly toggling the
    # quantized base weights trainable. If encoder training is requested while
    # staying in 4-bit mode, we must route through QLoRA adapters instead.
    # =====================================================================
    if (args.use_unsloth and args.unsloth_4bit and enable_encoder_training
            and not args.unsloth_qlora and not args.unsloth_full_finetuning):
        print("\n" + "=" * 70)
        print("[WARN]  CONFIGURATION NOTE: 4-bit Unsloth + Encoder Fine-Tuning")
        print("=" * 70)
        print("   You requested encoder fine-tuning while keeping the Unsloth encoder")
        print("   in 4-bit mode, but quantized base weights cannot be trained directly.")
        print("")
        print("   Auto-adjusting configuration:")
        print("   - Keeping --unsloth_4bit enabled")
        print("   - Keeping --unsloth_full_finetuning disabled")
        print("   - Enabling --unsloth_qlora so trainable LoRA adapters are used")
        print("     instead of trying to backprop through quantized base weights")
        print("=" * 70 + "\n")
        args.unsloth_qlora = True

    # If gated attention is enabled, force-disable temperature scaling for clean ablation
    if args.use_gated_attention and not args.disable_temperature:
        print("[INFO] Gated attention enabled -> forcing --disable_temperature for clean ablation (no temperature scaling)")
        args.disable_temperature = True

    # =====================================================================
    # IMPORTANT: Warn about QLoRA + Gradual Unfreezing conflict
    # =====================================================================
    # Unsloth QLoRA freezes the base model weights via PEFT, making gradual
    # unfreezing ineffective. Only LoRA adapter weights are trainable.
    # =====================================================================
    if (args.use_unsloth and args.unsloth_qlora and not args.unsloth_full_finetuning
            and args.encoder_tuning_mode == "gradual" and args.encoder_only_training):
        print("\n" + "=" * 70)
        print("[WARN]  CONFIGURATION NOTE: QLoRA + Gradual Unfreezing")
        print("=" * 70)
        print("   You have both --unsloth_qlora and --encoder_tuning_mode gradual enabled.")
        print("   With QLoRA, the base encoder weights are FROZEN by PEFT, so gradual")
        print("   unfreezing has NO EFFECT. Only LoRA adapter weights are trainable.")
        print("")
        print("   Auto-adjusting configuration:")
        print("   - Keeping --unsloth_qlora enabled")
        print("   - Keeping --unsloth_full_finetuning disabled")
        print("   - Switching --encoder_tuning_mode from 'gradual' to 'full'")
        print("     so no gradual-unfreeze schedule is applied to PEFT-managed weights")
        print("=" * 70 + "\n")
        args.encoder_tuning_mode = "full"

    # Simple cache control logic
    use_cache = args.use_cache
    # If encoder fine-tuning is enabled, disable cache during training so encoder updates affect embeddings
    if enable_encoder_training and use_cache:
        print(
            "[WARN]  Encoder fine-tuning (--enable_lora) detected: disabling cache for training so encoder updates affect embeddings")
        use_cache = False
    # Stage 0 cache logic is now handled in train.py for more sophisticated mode detection
    if use_cache:
        print("[OK] Cache enabled with --use_cache flag")
    else:
        print("[INFO] Cache disabled for training")

    try:
        # Get the actual model name from the dictionary using the key
        model_name = AVAILABLE_MODELS[args.model_key]

        # Initialize wandb if requested
        if args.use_wandb:
            if wandb is None:
                raise RuntimeError(
                    "W&B logging requested (--use_wandb) but `wandb` is not installed in this environment. "
                    "Install it (pip install wandb) or run without --use_wandb."
                )
            arch_name = "Bidirectional" if args.use_bidirectional else "CrossAttention"
            wandb.init(
                # Team Name
                entity="DTIM-UPC",
                # Project Name
                project="LOKI",
                name=f"{model_name.split('/')[-1]}-{arch_name}-{args.epochs}",
                config=vars(args)
            )

        # Initialize memory manager
        memory_manager = GPUMemoryManager()
        memory_manager.clear_memory()
        memory_manager.log_memory_stats("Initial")

        # Initialize model
        architecture_name = "bidirectional cross-attention" if args.use_bidirectional else "unidirectional cross-attention"
        print(f"\nInitializing {architecture_name} model {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ================================================================
        # ENCODER LOADING: Unsloth (fast) or Standard SentenceTransformer
        # ================================================================
        use_unsloth_loading = args.use_unsloth and UNSLOTH_AVAILABLE

        # Resolve model name for Unsloth (SentenceTransformer default namespace)
        unsloth_model_name = model_name
        if use_unsloth_loading and "/" not in model_name and not os.path.exists(model_name):
            unsloth_model_name = f"sentence-transformers/{model_name}"
            print(f"   [INFO] Auto-resolving short model name to '{unsloth_model_name}' for Unsloth")

        if use_unsloth_loading:
            # Use Unsloth for 2x faster training
            print("\n" + "=" * 60)
            print("[INFO] UNSLOTH MODE: Loading encoder with optimized kernels")
            print("=" * 60)
            print_unsloth_status()

            # Show which API will be used
            if FAST_SENTENCE_TRANSFORMER_AVAILABLE:
                print("   API: FastSentenceTransformer (specialized for embeddings) [OK]")
            else:
                print("   API: FastModel (generic LLM loader, fallback)")

            # Parse target modules from comma-separated string, or pass "auto" for auto-detection
            if args.unsloth_target_modules.lower() == "auto":
                target_modules = None  # Will trigger auto-detection in create_unsloth_sentence_encoder
                print("   Target modules: AUTO-DETECT (will be determined based on model architecture)")
            else:
                target_modules = [m.strip() for m in args.unsloth_target_modules.split(",") if m.strip()]
                print(f"   Target modules (user-specified): {target_modules}")

            # Auto-detect max_seq_length from model config (instead of hardcoding 512)
            detected_max_seq_length = get_model_max_seq_length(unsloth_model_name, default=512)

            # Apply user override if specified
            if args.override_max_seq_length > 0:
                print(f"[INFO] Overriding max_seq_length: {detected_max_seq_length} -> {args.override_max_seq_length}")
                detected_max_seq_length = args.override_max_seq_length

            try:
                sentence_encoder = create_unsloth_sentence_encoder(
                    model_name=unsloth_model_name,
                    device=device,
                    max_seq_length=detected_max_seq_length,
                    # Unsloth configuration
                    use_unsloth=True,
                    load_in_4bit=args.unsloth_4bit,
                    dtype=torch.bfloat16,
                    full_finetuning=args.unsloth_full_finetuning,  # Use argument instead of hardcoded False
                    # LoRA configuration (encoder fine-tuning via Unsloth)
                    use_qlora=args.unsloth_qlora,
                    lora_rank=args.unsloth_qlora_rank,
                    lora_alpha=args.unsloth_qlora_alpha,
                    lora_dropout=0.0,  # Unsloth recommends 0.0
                    target_modules=target_modules if args.unsloth_qlora else None,
                    # Pooling configuration (only for FastModel fallback)
                    pooling_mode=args.unsloth_pooling_mode,
                    normalize_embeddings=True,
                )
                print("[OK] Unsloth encoder loaded successfully!")

                if hasattr(sentence_encoder, 'peft_config') or hasattr(sentence_encoder, 'active_adapters'):
                    if getattr(sentence_encoder, 'peft_config', None) or getattr(sentence_encoder, 'active_adapters',
                                                                                 None):
                        print("   PEFT wrappers identified correctly.")

                if args.unsloth_qlora and not args.unsloth_full_finetuning:
                    print(
                        f"   LoRA adapters attached (rank={args.unsloth_qlora_rank}, alpha={args.unsloth_qlora_alpha})")
                    # Override encoder training flag since LoRA means we're fine-tuning
                    enable_encoder_training = True
                    print("   Encoder fine-tuning enabled via Unsloth LoRA")

                    # =====================================================================
                    # DIAGNOSTIC: Check actual PEFT parameter breakdown for encoder
                    # =====================================================================
                    print("\n   [INFO] Encoder PEFT/LoRA Parameter Breakdown:")
                    encoder_total = sum(p.numel() for p in sentence_encoder.parameters())
                    encoder_trainable = sum(p.numel() for p in sentence_encoder.parameters() if p.requires_grad)
                    encoder_frozen = encoder_total - encoder_trainable

                    # Check for LoRA-specific parameters
                    lora_params = sum(p.numel() for name, p in sentence_encoder.named_parameters()
                                      if 'lora' in name.lower())
                    base_params = encoder_total - lora_params

                    print(f"      Total encoder params: {encoder_total:,}")
                    print(f"      Trainable (requires_grad=True): {encoder_trainable:,}")
                    print(f"      Frozen (requires_grad=False): {encoder_frozen:,}")
                    print(f"      LoRA adapter params: {lora_params:,}")
                    print(f"      Base model params: {base_params:,}")

                    if encoder_trainable == encoder_total:
                        print("\n   [WARN] WARNING: All encoder params are trainable!")
                        print("      PEFT may not have frozen the base weights correctly.")
                        print("      This could mean full fine-tuning instead of QLoRA.")

                        # Try to detect if this is a PEFT model
                        has_peft = any('lora' in name.lower() for name, _ in sentence_encoder.named_parameters())
                        print(f"      PEFT model detected: {has_peft}")

                        if has_peft and lora_params == 0:
                            print("      Issue: PEFT wrapper exists but no LoRA params found")
                    elif encoder_trainable < encoder_total * 0.1:
                        print(
                            f"\n   [OK] QLoRA working correctly! Only {encoder_trainable / encoder_total * 100:.2f}% params trainable")

                elif args.unsloth_full_finetuning:
                    print("   Full fine-tuning enabled via Unsloth")
                    enable_encoder_training = True

            except Exception as e:
                print(f"[WARN] Unsloth loading failed: {e}")
                print("   Falling back to standard SentenceTransformer loading...")
                if enable_encoder_training:
                    print(
                        "   [WARN] WARNING: QLoRA adapters NOT attached - encoder will be fully trainable (100%) due to enable_lora flag")
                else:
                    print("   [INFO] Note: Encoder will be properly frozen (0% trainable) during standard loading.")
                use_unsloth_loading = False

        if not use_unsloth_loading:
            # Standard SentenceTransformer loading (fallback or explicit)
            if args.use_unsloth and not UNSLOTH_AVAILABLE:
                print("[WARN] Unsloth requested but not available. Using standard loading.")

            # Warn that QLoRA is not being used
            if args.unsloth_qlora and args.use_unsloth:
                print("\n" + "=" * 70)
                print("[WARN]  IMPORTANT: QLoRA was requested but Unsloth loading failed!")
                print("=" * 70)
                print("   The encoder is loaded WITHOUT QLoRA adapters.")
                if enable_encoder_training:
                    print("   This means 100% of encoder parameters will be trainable (full fine-tuning).")
                    print("   This uses significantly more memory than QLoRA (~1-5% params).")
                else:
                    print("   The encoder will remain completely frozen as fine-tuning is disabled.")
                print("")
                print("   To fix this, check the error message above and ensure:")
                print("   1. Unsloth is properly installed: pip install unsloth")
                print("   2. The target modules are correct for your model architecture")
                print("   3. The model is compatible with Unsloth's FastSentenceTransformer")
                print("=" * 70 + "\n")

            # Set up model_kwargs based on flags
            model_kwargs = {"dtype": torch.bfloat16}
            if "jina" in model_name.lower():
                model_kwargs["default_task"] = "retrieval"

            # Add Flash Attention if requested
            if args.use_flash_attention:
                model_kwargs.update({"attn_implementation": "flash_attention_2", "device_map": "auto"})

            # Encoder fine-tuning flag
            if enable_encoder_training:
                print(
                    "Encoder fine-tuning enabled (trainable_encoder=True). No encoder LoRA adapters are attached by default.")

            try:
                sentence_encoder = SentenceTransformer(
                    model_name,
                    model_kwargs=model_kwargs,
                    trust_remote_code=True,
                    device=device,
                    tokenizer_kwargs={"padding_side": "left"}
                )

                if args.use_flash_attention:
                    print("Initialized model with Flash Attention")
                else:
                    print("Initialized model without Flash Attention")

            except Exception as e:
                print(f"Warning: Failed to initialize with requested configuration: {e}")
                # Create new model_kwargs for fallback (exclude unsupported kwargs)
                fallback_model_kwargs = {}
                if enable_encoder_training:
                    print("Proceeding with encoder fine-tuning; no special model kwargs passed to SentenceTransformer")

                try:
                    # Always include bfloat16 for consistency
                    fallback_model_kwargs = fallback_model_kwargs or {}
                    fallback_model_kwargs["dtype"] = torch.bfloat16

                    sentence_encoder = SentenceTransformer(
                        model_name,
                        model_kwargs=fallback_model_kwargs,
                        trust_remote_code=True,
                        device=device,
                        tokenizer_kwargs={"padding_side": "left"}
                    )
                    print("Successfully initialized with fallback configuration (with bfloat16)")
                except Exception as e2:
                    print(f"Warning: Fallback initialization also failed: {e2}")
                    # Final fallback with bfloat16
                    try:
                        final_kwargs = {"dtype": torch.bfloat16}

                        sentence_encoder = SentenceTransformer(
                            model_name,
                            model_kwargs=final_kwargs,
                            trust_remote_code=True,
                            device=device
                        )
                        print("Final fallback with bfloat16 only")
                    except Exception as e3:
                        print(f"Warning: Final fallback also failed: {e3}")
                        super_final_kwargs = {}

                        sentence_encoder = SentenceTransformer(
                            model_name,
                            model_kwargs=super_final_kwargs if super_final_kwargs else None,
                            trust_remote_code=True,
                            device=device
                        )
                        print(
                            "Falling back to default model initialization without any custom parameters (no bfloat16)")

        # Apply max_seq_length override for standard (non-Unsloth) path
        if not use_unsloth_loading and args.override_max_seq_length > 0:
            native_max_seq = getattr(sentence_encoder, 'max_seq_length', 'unknown')
            sentence_encoder.max_seq_length = args.override_max_seq_length
            print(f"[INFO] Overriding max_seq_length: {native_max_seq} -> {args.override_max_seq_length}")

        # Get embedding dimension and sequence length
        native_embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
        if native_embedding_dim is None:
            print(
                "Warning: get_sentence_embedding_dimension() returned None. Doing dummy forward pass to extract dimension.")
            try:
                dummy_emb = sentence_encoder.encode(["Test"])
                native_embedding_dim = dummy_emb.shape[-1]
            except Exception as e:
                print(f"Error extracting embedding dimension: {e}. Defaulting to 768.")
                native_embedding_dim = 768

        embedding_dim = native_embedding_dim  # Will be overridden below if user specified

        # Apply embedding dimension override (with Matryoshka support)
        uses_matryoshka = False
        if args.override_embedding_dim > 0:
            target_dim = args.override_embedding_dim

            if target_dim >= native_embedding_dim:
                print(f"[WARN] override_embedding_dim ({target_dim}) >= native dim ({native_embedding_dim}), ignoring override.")
            else:
                # Try Matryoshka truncation first (SentenceTransformer built-in)
                sentence_encoder.truncate_dim = target_dim
                try:
                    test_emb = sentence_encoder.encode(["test"], convert_to_tensor=True)
                    actual_dim = test_emb.shape[-1]
                    if actual_dim == target_dim:
                        embedding_dim = target_dim
                        uses_matryoshka = True
                        print(f"[OK] Matryoshka truncation: {native_embedding_dim} -> {target_dim} (model supports it)")
                    else:
                        # Model didn't truncate - clear and use projection instead
                        sentence_encoder.truncate_dim = None
                        embedding_dim = target_dim
                        print(f"[INFO] Model does not support Matryoshka at dim={target_dim}. "
                              f"A linear projection layer ({native_embedding_dim} -> {target_dim}) will be added.")
                except Exception as e:
                    sentence_encoder.truncate_dim = None
                    embedding_dim = target_dim
                    print(f"[INFO] Matryoshka test failed ({e}). Using projection layer: {native_embedding_dim} -> {target_dim}")

        max_seq_length = getattr(sentence_encoder, 'max_seq_length', 512)  # Default to 512 if not available
        print(f"Embedding dimension: {embedding_dim}" + (f" (native: {native_embedding_dim}, Matryoshka)" if uses_matryoshka else (f" (native: {native_embedding_dim}, projected)" if embedding_dim != native_embedding_dim else "")))
        print(f"Max sequence length: {max_seq_length}")

        # Load datasets
        print("\nLoading datasets...")
        train_examples = load_row_level_dataset(args.train_file)
        print(f"Loaded {len(train_examples)} training examples")

        # Optionally sample a subset for faster training on large datasets
        # Uses DETERMINISTIC sampling with isolated random state for reproducibility
        if args.max_train_examples > 0 and len(train_examples) > args.max_train_examples:
            # Create an isolated random instance to not affect global state
            sampling_rng = random.Random(args.seed if hasattr(args, 'seed') else 42)

            # Sort by stable key (example_id or anchor_id) to ensure deterministic ordering
            # before sampling. This guarantees same subset across runs.
            def get_stable_key(ex):
                return ex.get('example_id', '') or str(ex.get('anchor_id', ''))

            train_examples_sorted = sorted(train_examples, key=get_stable_key)
            train_examples = sampling_rng.sample(train_examples_sorted, args.max_train_examples)
            print(
                f"[INFO] Sampled {len(train_examples)} training examples (deterministic, seed={args.seed if hasattr(args, 'seed') else 42})")

        eval_examples = load_row_level_dataset(args.eval_file)
        print(f"Loaded {len(eval_examples)} evaluation examples")

        # Determine eval sample size:
        # 1. If --max_eval_examples is explicitly set, use that
        # 2. If --max_train_examples is set but --max_eval_examples is not, auto-scale to 10% proportion
        # 3. Otherwise, use all eval examples
        effective_max_eval = args.max_eval_examples
        if effective_max_eval == 0 and args.max_train_examples > 0:
            # Auto-calculate: validation should be ~10% of training subset (matching typical 80/10/10 split)
            effective_max_eval = max(args.max_train_examples // 10, 50)  # At least 50 for stable metrics
            print(f"[INFO] Auto-scaling validation set to {effective_max_eval} examples (10% of training subset)")

        # Sample eval examples if needed
        if effective_max_eval > 0 and len(eval_examples) > effective_max_eval:
            # Use a DIFFERENT seed offset for eval to avoid overlap patterns
            eval_sampling_rng = random.Random((args.seed if hasattr(args, 'seed') else 42) + 1000)

            def get_stable_key(ex):
                return ex.get('example_id', '') or str(ex.get('anchor_id', ''))

            eval_examples_sorted = sorted(eval_examples, key=get_stable_key)
            eval_examples = eval_sampling_rng.sample(eval_examples_sorted, effective_max_eval)
            print(f"[INFO] Sampled {len(eval_examples)} evaluation examples (deterministic)")
    
        # NOTE: row_sent_max_examples is intentionally NOT linked to max_eval_examples.
        # Validation subsampling (max_eval_examples) controls the global accuracy eval set.
        # Row-sentence evaluation uses a separate test set with sparse annotations;
        # auto-limiting it would discard most annotated examples.  Default None = use all.

        # Create the model with LoRA options
        if args.use_bidirectional:
            model = BidirectionalTableTextModel(
                sentence_encoder=sentence_encoder,
                embedding_dim=embedding_dim,
                native_embedding_dim=native_embedding_dim if not uses_matryoshka else None,
                trainable_encoder=enable_encoder_training,
                use_cross_attention_lora=args.use_cross_attention_lora,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                top_k=args.top_k,
                pair_score_method=args.pair_score_method,
                share_weights=args.share_attention_weights,
                use_refinement=args.use_refinement,
                use_self_attention=args.use_self_attention,
                self_attention_heads=args.self_attention_heads,
                self_attention_dropout=args.self_attention_dropout,
                init_method=args.init_method,
                init_method_params=args.init_method_params,
                # **NEW**: Attention mechanism parameters
                attention_type=args.attention_type,
                sparse_top_k=args.sparse_top_k,
                window_size=args.window_size,
                threshold_base=args.threshold_base,
                norm_type=args.norm_type,
                use_qk_rmsnorm=args.use_qk_rmsnorm,
                # Latent bottleneck
                use_latent_bottleneck=args.use_latent_bottleneck,
                latent_num=args.latent_num,
                latent_dropout=args.latent_dropout,
                # Gated attention overlay + temperature disable
                use_gated_attention=args.use_gated_attention,
                gated_attention_mode=args.gated_attention_mode,
                gated_attention_hidden_dim=args.gated_attention_hidden_dim,
                gated_attention_dropout=args.gated_attention_dropout,
                gated_attention_init_bias=args.gated_attention_init_bias,
                use_header_conditioning=args.use_header_conditioning,
                use_cell_level_matching=args.use_cell_level_matching,
                cell_matching_weight=args.cell_matching_weight,
                cell_matching_pooling=args.cell_matching_pooling,
                cell_row_fusion_weight=args.cell_row_fusion_weight,
                disable_temperature=args.disable_temperature,
                # Verbosity control
                verbose=args.verbose_flag,
            )
        else:
            model = TableTextEmbeddingModel(
                sentence_encoder=sentence_encoder,
                embedding_dim=embedding_dim,
                native_embedding_dim=native_embedding_dim if not uses_matryoshka else None,
                trainable_encoder=enable_encoder_training,
                use_cross_attention_lora=args.use_cross_attention_lora,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                top_k=args.top_k,
                # **NEW**: Pass attention mechanism parameters to unidirectional model
                attention_type=args.attention_type,
                sparse_top_k=args.sparse_top_k,
                window_size=args.window_size,
                threshold_base=args.threshold_base,
                init_method=args.init_method,
                init_method_params=get_recommended_method_params(args.init_method),
                norm_type=args.norm_type,
                # **NEW**: Attention direction for unidirectional model
                attention_direction=args.attention_direction,
                # Latent bottleneck
                use_latent_bottleneck=args.use_latent_bottleneck,
                latent_num=args.latent_num,
                latent_dropout=args.latent_dropout,
                # Gated attention overlay + temperature disable
                use_gated_attention=args.use_gated_attention,
                gated_attention_mode=args.gated_attention_mode,
                gated_attention_hidden_dim=args.gated_attention_hidden_dim,
                gated_attention_dropout=args.gated_attention_dropout,
                gated_attention_init_bias=args.gated_attention_init_bias,
                disable_temperature=args.disable_temperature,
                # Skip FFN option
                skip_ffn=args.skip_ffn,
                # Verbosity control
                verbose=args.verbose_flag,
            )

        # Now that the model exists, propagate training/loss/attention switches to it
        setattr(model, 'ranking_loss_type', args.ranking_loss_type)
        setattr(model, 'infonce_tau', args.infonce_tau)
        setattr(model, 'pair_topk_mask', args.pair_topk_mask)
        setattr(model, 'pair_topk_k', args.pair_topk_k)
        setattr(model, 'attention_activation', args.attention_activation)
        setattr(model, 'attention_alpha', args.attention_alpha)

        model.to(device)

        # ================================================================
        # TORCH.COMPILE OPTIMIZATION FOR CUSTOM MODULES
        # ================================================================
        # Apply torch.compile() to custom attention modules for 2x faster execution
        if args.use_compile and TORCH_COMPILE_AVAILABLE:
            print("\n" + "=" * 60)
            print("[INFO] TORCH.COMPILE: Optimizing custom attention modules")
            print("=" * 60)
            model = optimize_model_for_training(
                model,
                use_compile=True,
                compile_mode=args.compile_mode,
                use_gradient_checkpointing=False,  # Can be enabled for memory efficiency
            )
            print("=" * 60 + "\n")
        elif args.use_compile and not TORCH_COMPILE_AVAILABLE:
            print("[WARN] torch.compile() requested but not available (requires PyTorch 2.0+)")

        if args.use_cross_attention_lora:
            print(f"Model initialized with Cross-Attention LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
        else:
            print(f"Model initialized with standard cross-attention layers")

        # Print parameter counts to verify LoRA setup
        param_counts = model.count_parameters()
        print(f"\n[INFO] Model Parameter Statistics (Initial):")
        print(f"  Total parameters: {param_counts['total_parameters']:,}")
        print(f"  Trainable parameters: {param_counts['trainable_parameters']:,}")
        print(f"  Frozen parameters: {param_counts['frozen_parameters']:,}")
        print(f"  Trainable percentage: {param_counts['trainable_percentage']:.2f}%")

        # Add encoder-specific info if QLoRA is active
        if args.use_unsloth and args.unsloth_qlora and not args.unsloth_full_finetuning:
            enc_total = sum(p.numel() for p in model.sentence_encoder.parameters())
            enc_trainable = sum(p.numel() for p in model.sentence_encoder.parameters() if p.requires_grad)
            enc_lora = sum(p.numel() for name, p in model.sentence_encoder.named_parameters()
                           if 'lora' in name.lower())
            print(f"  [INFO] Encoder QLoRA Status:")
            print(f"     Encoder total: {enc_total:,}")
            print(f"     Encoder trainable (LoRA): {enc_trainable:,} ({enc_trainable / enc_total * 100:.2f}%)")
            if enc_lora > 0:
                print(f"     LoRA adapter params: {enc_lora:,}")
            print(f"  [INFO]  Note: Final trainable % will be set by train.py based on training mode")

        if args.use_cross_attention_lora:
            lora_params = param_counts.get('lora_parameters', 0)
            if lora_params > 0:
                print(f"  LoRA parameters: {lora_params:,}")
                print(
                    f"  LoRA efficiency: {(lora_params / param_counts['total_parameters']) * 100:.4f}% of total parameters")

        # Print Loss and Aggregation configuration
        print(f"\n[INFO] Loss and Aggregation Configuration:")
        print(f"  Architecture: {'Bidirectional' if args.use_bidirectional else 'Unidirectional'}")
        print(f"  Loss type: {args.loss_type}")
        print(f"  Aggregation method: {args.aggregation_method}")
        print(f"  Top-k value: {args.top_k}")
        print(f"  Norm type: {args.norm_type}")
        print(f"  Q/K RMSNorm: {args.use_qk_rmsnorm}")
        print(f"  Header conditioning: {args.use_header_conditioning}")
        print(f"  Embedding caching: {'[OK] Enabled' if use_cache else '[ERROR] Disabled'}")
        print(f"  Triplet batching: {'[INFO] Mixed examples' if args.mix_examples else '[INFO] Isolated examples'}")
        print(f"  Triplet strategy: {args.triplet_strategy.upper()}")
        if args.triplet_strategy in ["limited", "random", "balanced"]:
            print(f"  Max triplets/example: {args.max_triplets_per_example}")
        print(f"\n  [INFO] Initialization Configuration:")
        print(f"    Method: {args.init_method}")
        print(f"    Description: {get_method_description(args.init_method)}")
        init_params = get_recommended_method_params(args.init_method)
        if init_params:
            print(f"    Parameters: {init_params}")
        print(f"\n  [INFO] Loss Component Weights (will be normalized):")
        print(f"    Triplet weight: {args.triplet_weight}")
        print(f"    Attention weight: {args.attention_loss_weight}")
        if args.use_bidirectional:
            print(f"    Pair weight: {args.pair_loss_weight}")
        print(f"\n  [INFO] Loss Parameters:")
        print(f"    Margin: {args.margin}")
        print(f"    Scale: {args.scale}")
        if args.use_bidirectional:
            print(f"    Pair margin: {args.pair_margin}")
            print(f"    Pair score method: {args.pair_score_method}")
            print(f"    Share attention weights: {args.share_attention_weights}")
            print(f"    Join path extraction: {args.extract_join_paths}")
            if args.extract_join_paths:
                print(f"    Join path threshold: {args.join_path_threshold}")
            print(f"    Self-attention enabled: {args.use_self_attention}")
            if args.use_self_attention:
                print(f"    Self-attention heads: {args.self_attention_heads}")
                print(f"    Self-attention dropout: {args.self_attention_dropout}")
            print(f"    Attention mechanism: {args.attention_type}")
            if args.attention_type == "top_k_sparse":
                print(f"    Sparse top-k: {args.sparse_top_k}")
            elif args.attention_type == "windowed":
                print(f"    Window size: {args.window_size}")
            elif args.attention_type == "threshold":
                print(f"    Threshold base: {args.threshold_base}")

        # **NEW**: Show attention mechanism details for both architectures
        print(f"\n  [INFO] Attention Mechanism Configuration:")
        print(f"    Attention type: {args.attention_type}")
        if args.attention_type == "top_k_sparse":
            print(f"    Sparse top-k: {args.sparse_top_k}")
        elif args.attention_type == "windowed":
            print(f"    Window size: {args.window_size}")
        elif args.attention_type == "threshold":
            print(f"    Threshold base: {args.threshold_base}")
        print(f"    Initialization method: {args.init_method}")

        if args.use_bidirectional:
            print(f"\n  [INFO] Bidirectional-Specific Configuration:")
            print(f"    Pair margin: {args.pair_margin}")
            print(f"    Pair score method: {args.pair_score_method}")
            print(f"    Share attention weights: {args.share_attention_weights}")
            print(f"    Join path extraction: {args.extract_join_paths}")
            if args.extract_join_paths:
                print(f"    Join path threshold: {args.join_path_threshold}")
            print(f"    Self-attention enabled: {args.use_self_attention}")
            if args.use_self_attention:
                print(f"    Self-attention heads: {args.self_attention_heads}")
                print(f"    Self-attention dropout: {args.self_attention_dropout}")

        # Setup output directory with timestamp
        output_dir = setup_output_dir(args, model_name, embedding_dim, max_seq_length)
        print(f"Output will be saved to: {output_dir}")

        # Save training configuration
        config_path = Path(output_dir) / "training_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a copy of args to save, and add the model_name and architecture info for reference
        config_dict = vars(args).copy()
        config_dict['model_name'] = model_name
        config_dict['embedding_dim'] = embedding_dim
        config_dict['max_seq_length'] = max_seq_length
        config_dict['architecture'] = "bidirectional" if args.use_bidirectional else "unidirectional"
        config_dict['loss_type'] = args.loss_type
        config_dict['aggregation_method'] = args.aggregation_method
        config_dict['triplet_weight'] = args.triplet_weight
        config_dict['attention_loss_weight'] = args.attention_loss_weight
        config_dict['pair_loss_weight'] = args.pair_loss_weight if args.use_bidirectional else 0.0
        # Loss/Ranking switches for reproducibility
        config_dict['ranking_loss_type'] = args.ranking_loss_type
        config_dict['infonce_tau'] = args.infonce_tau
        config_dict['pair_topk_mask'] = args.pair_topk_mask
        config_dict['pair_topk_k'] = args.pair_topk_k
        config_dict['margin'] = args.margin
        config_dict['scale'] = args.scale
        config_dict['init_method'] = args.init_method
        config_dict['init_method_params'] = args.init_method_params
        config_dict['use_self_attention'] = args.use_self_attention
        config_dict['self_attention_heads'] = args.self_attention_heads
        config_dict['self_attention_dropout'] = args.self_attention_dropout
        config_dict['attention_type'] = args.attention_type
        config_dict['attention_activation'] = args.attention_activation
        config_dict['attention_alpha'] = args.attention_alpha
        config_dict['sparse_top_k'] = args.sparse_top_k
        config_dict['window_size'] = args.window_size
        config_dict['threshold_base'] = args.threshold_base
        config_dict['norm_type'] = args.norm_type
        config_dict['use_qk_rmsnorm'] = args.use_qk_rmsnorm
        # Gated attention overlay + temperature disable
        config_dict['use_gated_attention'] = args.use_gated_attention
        config_dict['gated_attention_mode'] = args.gated_attention_mode
        config_dict['gated_attention_hidden_dim'] = args.gated_attention_hidden_dim
        config_dict['gated_attention_dropout'] = args.gated_attention_dropout
        config_dict['gated_attention_init_bias'] = args.gated_attention_init_bias
        config_dict['disable_temperature'] = args.disable_temperature
        config_dict['use_cache'] = use_cache
        config_dict['stage0_cache_mode'] = args.stage0_cache_mode
        config_dict['mix_examples'] = args.mix_examples
        config_dict['triplet_strategy'] = args.triplet_strategy
        config_dict['max_triplets_per_example'] = args.max_triplets_per_example
        config_dict['encoder_only_training'] = args.encoder_only_training
        # Encoder tuning configuration
        config_dict['encoder_tuning_mode'] = args.encoder_tuning_mode
        config_dict['gradual_unfreeze_initial_layers'] = args.gradual_unfreeze_initial_layers
        config_dict['gradual_unfreeze_every'] = args.gradual_unfreeze_every
        config_dict['gradual_unfreeze_max_layers'] = args.gradual_unfreeze_max_layers
        config_dict['gradual_unfreeze_include_pooler'] = args.gradual_unfreeze_include_pooler
        # Save new training controls
        config_dict['encoder_lr'] = args.encoder_lr
        config_dict['lr_scheduler_type'] = args.lr_scheduler_type
        config_dict['min_lr_ratio'] = args.min_lr_ratio
        config_dict['early_stopping_patience'] = args.early_stopping_patience
        config_dict['early_stopping_min_epochs'] = args.early_stopping_min_epochs
        config_dict['verbosity'] = args.verbosity

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"\nTraining configuration saved to {config_path}")

        # Train the model
        print("\nStarting Model Training...")
        print("=" * 50)

        # Use ID-based triplet training with caching
        trained_model = train_with_id_based_triplets(
            model=model,
            train_examples=train_examples,
            eval_examples=eval_examples,
            output_path=output_dir,
            run_name=model_name,
            learning_rate=args.lr,
            encoder_learning_rate=args.encoder_lr,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            lr_num_cycles=args.lr_num_cycles,
            min_lr_ratio=args.min_lr_ratio,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            enable_checkpointing=args.enable_checkpointing,
            use_cache=use_cache,
            loss_type=args.loss_type,
            aggregation_method=args.aggregation_method,
            top_k=args.top_k,  # Number of top pairs for aggregation methods
            triplet_weight=args.triplet_weight,  # NEW: Pass triplet weight argument
            attention_loss_weight=args.attention_loss_weight,
            margin=args.margin,
            margin_end=args.margin_end,
            margin_schedule=args.margin_schedule,
            scale=args.scale,
            # NEW: Bidirectional-specific parameters
            pair_loss_weight=args.pair_loss_weight if args.use_bidirectional else 0.0,
            pair_margin=args.pair_margin if args.use_bidirectional else 0.1,
            pair_score_method=args.pair_score_method if args.use_bidirectional else "cosine",
            # NEW: Attention distillation parameters
            use_attention_distillation=args.use_attention_distillation if args.use_bidirectional else False,
            distillation_weight=args.distillation_weight if args.use_bidirectional else 0.0,
            teacher_temperature=args.teacher_temperature if args.use_bidirectional else 0.1,
            student_temperature=args.student_temperature if args.use_bidirectional else 0.1,
            distillation_loss_type=args.distillation_loss_type if args.use_bidirectional else "kl_div",
            share_attention_weights=args.share_attention_weights if args.use_bidirectional else False,
            extract_join_paths=args.extract_join_paths if args.use_bidirectional else False,
            join_path_threshold=args.join_path_threshold if args.use_bidirectional else 0.1,
            mix_examples=args.mix_examples,  # Control triplet batching behavior
            triplet_strategy=args.triplet_strategy,  # NEW: Triplet sampling strategy
            max_triplets_per_example=args.max_triplets_per_example,  # NEW: Max triplets for limited/random
            use_hard_negative_mining=args.use_hard_negative_mining,
            hard_negative_topk=args.hard_negative_topk,
            # Training curves parameters
            enable_training_curves=args.enable_training_curves,
            track_batch_losses=args.track_batch_losses,
            track_val_loss=args.track_val_loss,
            auto_plot_curves=args.auto_plot_curves,
            # Early stopping
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_epochs=args.early_stopping_min_epochs,
            # Row-sentence evaluation parameters
            enable_row_sent_eval=args.enable_row_sent_eval,
            row_sent_test_file=args.row_sent_test_file,
            row_sent_annotation_file=args.row_sent_annotation_file,
            dataset_format=args.dataset_format,
            row_sent_max_examples=args.row_sent_max_examples,
            # Initialization parameters (NEW - for Stage 1 evaluation)
            init_method=args.init_method,
            init_method_params=args.init_method_params,
            # NEW: Training stage control
            start_training_from_stage=args.start_training_from_stage,
            encoder_only_training=args.encoder_only_training,
            stage0_cache_mode=args.stage0_cache_mode,
            encoder_tuning_mode=args.encoder_tuning_mode,
            gradual_unfreeze_initial_layers=args.gradual_unfreeze_initial_layers,
            gradual_unfreeze_every=args.gradual_unfreeze_every,
            gradual_unfreeze_max_layers=args.gradual_unfreeze_max_layers,
            gradual_unfreeze_include_pooler=args.gradual_unfreeze_include_pooler,
            save_best_by_test_metrics=args.save_best_by_test_metrics,
            # Visualization control
            skip_four_stage_viz=args.skip_four_stage_viz,
            # Weights & Biases logging
            use_wandb=args.use_wandb,
            # Task-Aware Dataset flags
            task_direction=args.task_direction,
            native_direction=args.native_direction,
            # Encoding batch size for cache building
            encoding_batch_size=args.encoding_batch_size,
        )

        print(f"\n[OK] Training completed! The returned model is the BEST performing model from training.")
        print(f"[INFO] Check the training logs above to see which epoch achieved the highest validation accuracy.")

        # Save the best model (training function returns the best performing model)
        # Note: train_with_id_based_triplets() automatically loads and returns the best model based on validation accuracy
        best_model_path = Path(output_dir) / f"{model_name.split('/')[-1]}_best.pt"
        torch.save(trained_model.state_dict(), best_model_path)
        print(f"Saved best model to {best_model_path}")

        # Evaluate on test set if provided
        test_examples = None
        if args.test_file:
            print("\nEvaluating on Test Set...")
            test_examples = load_row_level_dataset(args.test_file)
            
            # Subsample test evaluation to match max_eval_examples setting
            if args.max_eval_examples > 0 and len(test_examples) > args.max_eval_examples:
                # Use deterministic sampling
                random.seed(42)
                test_examples = random.sample(test_examples, args.max_eval_examples)
                print(f"[INFO] Sampled {args.max_eval_examples} test examples for final evaluation (deterministic)")
                
            print(f"Loaded {len(test_examples)} test examples")

            # Use specialized evaluation for bidirectional models if join path extraction is enabled
            if args.use_bidirectional and args.extract_join_paths:
                # Build cache only if caching is enabled
                test_cache = None
                if use_cache:
                    print(f"Building test cache for evaluation...")
                    test_cache = build_id_based_embedding_cache(
                        examples=test_examples,
                        sentence_encoder_model=trained_model.sentence_encoder,
                        batch_size=args.eval_batch_size,
                        device=device,
                        split_name="test",
                        super_batch_size=args.encoding_batch_size,
                        task_direction=args.task_direction,
                        native_direction=args.native_direction,
                        use_header_conditioning=args.use_header_conditioning,
                        use_cell_level_matching=args.use_cell_level_matching,
                    )
                    print(f"Test cache stats: {test_cache.stats()}")
                else:
                    print("Cache disabled - evaluation will compute embeddings on-the-fly")

                test_metrics = evaluate_bidirectional_with_join_paths(
                    model=trained_model,
                    examples=test_examples,
                    id_cache=test_cache,
                    batch_size=args.eval_batch_size,
                    aggregation_method=args.aggregation_method,
                    join_path_threshold=args.join_path_threshold,
                    save_join_paths=True,
                    output_dir=output_dir
                )
            else:
                if use_cache:
                    test_metrics = evaluate_with_cache(trained_model, test_examples, batch_size=args.eval_batch_size,
                                                       aggregation_method=args.aggregation_method)
                else:
                    print("Cache disabled - using direct evaluation")
                    test_metrics = evaluate_model(trained_model, test_examples, batch_size=args.eval_batch_size,
                                                  device=device)

            print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
            if args.use_bidirectional and 'join_paths_extracted' in test_metrics:
                print(f"Join Paths Extracted: {test_metrics['join_paths_extracted']}")

            # Save test metrics
            test_metrics_path = Path(output_dir) / "test_metrics.json"
            save_evaluation_results(test_metrics, test_metrics_path)

        # Run visualizations if requested (using the best model from training)
        if args.do_visualize and test_examples:
            print(f"\n[INFO] Generating visualizations using the BEST model (highest validation accuracy)...")
            print(f"[OK] CONFIRMED: Using BEST model checkpoint from training")
            print(f"[INFO] Best model path: {best_model_path}")
            print(f"[INFO] Using aggregation method: {args.aggregation_method}")

            # Verify the best model file exists
            if not best_model_path.exists():
                print(f"[ERROR] ERROR: Best model file not found at {best_model_path}")
                return

            print(f"[OK] Best model file confirmed to exist")

            # Create comprehensive analysis for specified examples using the in-memory trained model
            print(f"\n[INFO] Creating comprehensive model analysis...")
            from visualize_attention import visualize_comprehensive_model_analysis

            # Parse example indices
            if args.visualize_examples.strip():
                try:
                    example_indices = [int(x.strip()) for x in args.visualize_examples.split(',')]
                except ValueError:
                    print(f"Warning: Invalid example indices '{args.visualize_examples}', using first example")
                    example_indices = [0]
            else:
                example_indices = [0]

            # Create comprehensive analysis for each specified example
            for idx in example_indices:
                if idx < len(test_examples):
                    print(f"[INFO] Creating comprehensive analysis for example {idx} using BEST model...")
                    visualize_comprehensive_model_analysis(
                        trained_model=trained_model,  # Use the same trained model instance
                        example=test_examples[idx],
                        example_idx=idx,
                        output_dir=output_dir,
                        aggregation_method=args.aggregation_method,
                        base_model_name=model_name
                    )

                    # === STEP-BY-STEP DIAGNOSTICS HEATMAPS ===
                    # Use the same model instance as comprehensive analysis to ensure consistency
                    from visualize_attention import save_diagnostics_heatmaps, extract_rows_and_sentences

                    # Extract rows and sentences using the robust extraction function
                    rows, sentences = extract_rows_and_sentences(test_examples[idx], idx)

                    if rows and sentences:
                        print(f"[INFO] Generating step-by-step diagnostics for example {idx}...")
                        # Use the same model that was used for comprehensive analysis
                        # This ensures exact consistency between comprehensive and step-by-step results
                        save_diagnostics_heatmaps(
                            model=trained_model,  # Use the same trained model instance
                            rows=rows,
                            sentences=sentences,
                            example_idx=idx,
                            output_dir=output_dir,
                            use_refinement=True  # Use the same refinement setting as the trained model
                        )
                    else:
                        print(f"  [WARN] Could not extract rows/sentences for example {idx}")
                        continue

            # === 4-STAGE EVALUATION INTEGRITY ANALYSIS ===
            if args.skip_four_stage_viz:
                print(f"\n[INFO] Skipping 4-STAGE example visualizations (--skip_four_stage_viz is set)")
            else:
                print(f"\n[INFO] Generating 4-STAGE EVALUATION INTEGRITY ANALYSIS...")
                print(f"[OK] This shows Stage 0->1->2->3 progression for research integrity")
                try:
                    from visualize_attention import create_complete_four_stage_analysis

                    # Use first few test examples for 4-stage analysis
                    viz_examples = test_examples[:3]  # First 3 examples
                    four_stage_output_dir = Path(output_dir) / "four_stage_analysis"

                    create_complete_four_stage_analysis(
                        trained_model=trained_model,
                        examples=viz_examples,
                        output_dir=str(four_stage_output_dir),
                        example_indices=args.visualize_examples,
                        base_model_name=model_name,
                        init_method=args.init_method,
                        init_method_params=args.init_method_params
                    )

                    print(f"[OK] 4-stage analysis completed and saved to: {four_stage_output_dir}")
                    print(f"[INFO] Generated files:")
                    print(f"   - four_stage_comparison_example_X.png (2x2 heatmap comparison)")
                    print(f"   - Stage_X_similarities_example_X.npy (similarity matrices)")

                except Exception as e:
                    print(f"[WARN] Could not generate 4-stage analysis: {e}")
                print("   You can run it manually using demo_four_stage_visualization.py")

        # === CLEAN STEP-BY-STEP ANALYSIS (NEW SYSTEM) ===
        if args.do_clean_analysis and test_examples:
            print(f"\n[INFO] Running CLEAN step-by-step analysis to fix inconsistencies...")
            print(f"[OK] This system computes everything from scratch using single forward pass")
            print(f"[INFO] Using refinement setting: {args.use_refinement}")

            # Parse example indices
            if args.visualize_examples.strip():
                try:
                    example_indices = [int(x.strip()) for x in args.visualize_examples.split(',')]
                except ValueError:
                    print(f"Warning: Invalid example indices '{args.visualize_examples}', using first example")
                    example_indices = [0]
            else:
                example_indices = [0]

            # Run clean analysis for specified examples
            clean_results = run_clean_analysis_for_examples(
                model=trained_model,
                examples=test_examples,
                example_indices=example_indices,
                output_dir=output_dir,
                use_refinement=args.use_refinement
            )

            # Print summary of clean analysis results
            print(f"\n[INFO] Clean Analysis Summary:")
            for idx, results in clean_results.items():
                if results:
                    consistency = results.get('consistency_status', 'UNKNOWN')
                    print(f"  Example {idx}: Consistency = {consistency}")

                    if not args.use_refinement and 'comparison_status' in results:
                        comparison = results['comparison_status']
                        print(f"               Contextualized vs Final = {comparison}")
                        if comparison == "DIFFERENT":
                            print(f"               [WARN]  BUG DETECTED: Should be identical when refinement=False")
                else:
                    print(f"  Example {idx}: Skipped (missing data)")

            print(f"[INFO] Detailed results saved in: {output_dir}/clean_diagnostics/")
            print(f"[INFO] Check individual heatmaps and summary.json files for each example")

        memory_manager.clear_memory()
        if args.use_wandb:
            if wandb is not None:
                wandb.finish()

        print("\nTraining, evaluation, and visualization completed successfully!")

    except Exception as e:
        import traceback
        print(f"\nError during training or evaluation: {str(e)}")
        traceback.print_exc()
        traceback.print_exc()
        if args.use_wandb:
            if wandb is not None:
                wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()