import torch
import json
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
from pathlib import Path
import random

from models import TableTextEmbeddingModel
from data import (
    IdBasedEmbeddingCache,
    _extract_sentences_robust,
    _extract_rows_robust,
    _extract_table_cell_texts,
    _extract_table_rows_for_model,
    _extract_table_schema_text,
    _normalize_schema_texts,
)
from encoding import build_id_based_embedding_cache

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
    model: TableTextEmbeddingModel,
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
    model: TableTextEmbeddingModel,
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

def _forward_with_optional_structure(
    model: TableTextEmbeddingModel,
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

def evaluate_with_id_cache(model: TableTextEmbeddingModel,
                         examples: List[Dict[str, Any]],
                         id_cache: Optional[IdBasedEmbeddingCache] = None,
                         batch_size: int = 16,
                         aggregation_method: str = "entropy_regularized",
                         allow_cache_build: bool = True,
                         evaluation_margin: float = 0.0) -> Dict[str, float]:
    """
    Evaluate the model using the ID-based cache for better efficiency.
    
    Args:
        model: The model to evaluate
        examples: List of processed examples
        id_cache: Optional ID-based cache of embeddings
        batch_size: Batch size for encoding
        aggregation_method: Method for aggregating scores
        allow_cache_build: Whether to build cache if none provided (default: True)
        evaluation_margin: Margin for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Starting ID-based evaluation...")
    model.eval()
    device = next(model.parameters()).device
    
    # CRITICAL: Detect model dtype to ensure embeddings match model component dtypes
    model_dtype = next(model.parameters()).dtype
    use_header_conditioning = bool(getattr(model, 'use_header_conditioning', False))
    use_cell_level_matching = bool(getattr(model, 'use_cell_level_matching', False))
    
    # Build ID-based cache if not provided and allowed
    if id_cache is None:
        if allow_cache_build:
            print("Building ID-based cache for evaluation...")
            id_cache = build_id_based_embedding_cache(
                examples=examples,
                sentence_encoder_model=model.sentence_encoder,
                batch_size=batch_size,
                device=device,
                split_name="eval",
                use_header_conditioning=use_header_conditioning,
                use_cell_level_matching=use_cell_level_matching,
            )
            print(f"Cache stats: {id_cache.stats()}")
        else:
            print("Cache building disabled - falling back to direct evaluation")
            return evaluate_model(model, examples, batch_size, device.type if hasattr(device, 'type') else str(device))
    
    total_comparisons = 0
    correct_predictions = 0
    
    with torch.no_grad():
        # Process each example
        for example in tqdm(examples, desc=f"Evaluating examples"):
            anchor_id = example.get("anchor_id")
            if anchor_id is None:
                continue
            
            is_flipped = "anchor_sentences" in example
            
            # Get row embeddings from cache
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
            
            # Add batch dimension and convert to model dtype
            row_tensor = row_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)  # [1, num_rows, dim]
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
                if positive_id is None:
                    continue
                
                # Get positive embeddings from cache
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
                
                # Add batch dimension and convert to model dtype
                positive_tensor = positive_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)  # [1, num_sentences, dim]
                positive_schema_tensor = _batch_schema_embedding(positive_schema, device, model_dtype)
                positive_cell_tensor = _batch_cell_embedding(positive_cells, device, model_dtype)
                
                # Get similarity score for positive using aggregation method
                positive_similarity, _ = _forward_with_optional_structure(
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
                
                # Process each negative
                for negative in example["negatives"]:
                    negative_id = negative.get("id")
                    if negative_id is None:
                        continue
                    
                    # Get negative embeddings from cache
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
                    
                    # Add batch dimension and convert to model dtype
                    negative_tensor = negative_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)  # [1, num_sentences, dim]
                    negative_schema_tensor = _batch_schema_embedding(negative_schema, device, model_dtype)
                    negative_cell_tensor = _batch_cell_embedding(negative_cells, device, model_dtype)
                    
                    # Get similarity score for negative using aggregation method
                    negative_similarity, _ = _forward_with_optional_structure(
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
    
    # Calculate metrics
    accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0
    
    return {
        'accuracy': accuracy,
        'total_comparisons': total_comparisons
    }

def evaluate_with_cache(model: TableTextEmbeddingModel, 
                         examples: List[Dict[str, Any]], 
                         batch_size: int = 16,
                         aggregation_method: str = "entropy_regularized",
                         evaluation_margin: float = 0.0) -> Dict[str, float]:
    """
    Simplified evaluation function that doesn't depend on EmbeddingCache.
    
    Args:
        model: The model to evaluate
        examples: List of processed examples
        batch_size: Batch size for encoding
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Use the ID-based evaluation when possible
    try:
        return evaluate_with_id_cache(model, examples, None, batch_size, aggregation_method, allow_cache_build=True, evaluation_margin=evaluation_margin)
    except Exception as e:
        print(f"ID-based evaluation failed: {e}. Falling back to direct evaluation.")
    
    print("Starting direct evaluation...")
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    use_header_conditioning = bool(getattr(model, 'use_header_conditioning', False))
    use_cell_level_matching = bool(getattr(model, 'use_cell_level_matching', False))
    
    total_comparisons = 0
    correct_predictions = 0
    
    with torch.no_grad():
        # Process each example
        for example in tqdm(examples, desc=f"Evaluating examples"):
            is_flipped = "anchor_sentences" in example
            
            if is_flipped:
                anchor_rows = _extract_sentences_robust(example.get("anchor_sentences", []))
                anchor_schema_text = None
                anchor_cell_text_rows = None
            else:
                anchor_rows = _extract_table_rows_for_model(example, use_header_conditioning=use_header_conditioning)
                anchor_schema_text = _extract_table_schema_text(example) if use_header_conditioning else None
                anchor_cell_text_rows = _extract_table_cell_texts(example) if use_cell_level_matching else None
            
            if not anchor_rows:
                continue  # Skip this example if no valid rows found
            
            num_rows = len(anchor_rows)
            
            # Skip examples with no rows
            if num_rows == 0:
                continue
            
            try:
                # Encode rows once for this example
                row_embeddings = model.encode_sentences(anchor_rows, batch_size=batch_size)
                row_tensor = row_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)  # [1, num_rows, dim]
                row_schema_tensor = _batch_schema_embedding(
                    _encode_schema_texts(model, anchor_schema_text, batch_size, device, model_dtype),
                    device,
                    model_dtype,
                )
                row_cell_tensor = _batch_cell_embedding(
                    _encode_cell_text_rows(model, anchor_cell_text_rows or [], batch_size, device, model_dtype),
                    device,
                    model_dtype,
                )
                
                # Collect all positive contexts (primary + additional)
                all_positives = []
                
                # Add primary positive
                primary_positive = example.get("primary_positive", {})
                if primary_positive:
                    all_positives.append(primary_positive)
                
                # Check if additional_positives exists and is not empty
                additional_positives = example.get("additional_positives", [])
                if additional_positives:  # Only process if not empty
                    for add_pos in additional_positives:
                        if add_pos:
                            all_positives.append(add_pos)
                
                if not all_positives:
                    continue
                
                # Process all positive contexts against all negatives
                for positive in all_positives:
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
                        
                    if not positive_sentences:
                        continue
                    
                    # Encode positive sentences
                    positive_embeddings = model.encode_sentences(positive_sentences, batch_size=batch_size)
                    positive_tensor = positive_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)  # [1, num_sentences, dim]
                    positive_schema_tensor = _batch_schema_embedding(
                        _encode_schema_texts(model, positive_schema_text, batch_size, device, model_dtype),
                        device,
                        model_dtype,
                    )
                    positive_cell_tensor = _batch_cell_embedding(
                        _encode_cell_text_rows(model, positive_cell_text_rows or [], batch_size, device, model_dtype),
                        device,
                        model_dtype,
                    )
                    
                    # Get similarity score for positive using aggregation method
                    positive_similarity, _ = _forward_with_optional_structure(
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
                    
                    # Process each negative
                    for negative in example["negatives"]:
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
                        
                        # Encode negative sentences
                        if not negative_sentences:
                            continue
                        
                        negative_embeddings = model.encode_sentences(negative_sentences, batch_size=batch_size)
                        negative_tensor = negative_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)  # [1, num_sentences, dim]
                        negative_schema_tensor = _batch_schema_embedding(
                            _encode_schema_texts(model, negative_schema_text, batch_size, device, model_dtype),
                            device,
                            model_dtype,
                        )
                        negative_cell_tensor = _batch_cell_embedding(
                            _encode_cell_text_rows(model, negative_cell_text_rows or [], batch_size, device, model_dtype),
                            device,
                            model_dtype,
                        )
                        
                        # Get similarity score for negative using aggregation method
                        negative_similarity, _ = _forward_with_optional_structure(
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
                
            except Exception as e:
                print(f"Error evaluating example (ID: {example.get('anchor_id', 'unknown')}): {e}")
    
    # Calculate metrics
    accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0
    
    return {
        'accuracy': accuracy,
        'total_comparisons': total_comparisons
    }

def get_augmented_negatives(examples: List[Dict[str, Any]], num_augmented: int = 1) -> List[Dict[str, Any]]:
    """
    Augment examples with additional negatives by randomly sampling from other examples.
    
    Args:
        examples: List of processed examples
        num_augmented: Number of augmented negatives to add per example
        
    Returns:
        List of examples with augmented negatives
    """
    # Create a copy of examples to avoid modifying the original
    augmented_examples = []
    
    # Create a pool of all negative sentences for sampling
    all_negatives = []
    for example in examples:
        for negative in example["negatives"]:
            if negative["sentences"]:  # Only add if there are sentences
                all_negatives.append({
                    "id": negative["id"],
                    "sentences": negative["sentences"]
                })
    
    # Only proceed if we have negatives to sample from
    if not all_negatives:
        print("Warning: No negatives available for augmentation")
        return examples
    
    print(f"Augmenting examples with {num_augmented} random negatives each")
    
    # Augment each example
    for example in examples:
        # Create a deep copy of the example
        augmented_example = {key: value for key, value in example.items()}
        augmented_example["negatives"] = list(example["negatives"])  # Copy the negatives list
        
        # Add augmented negatives
        for _ in range(num_augmented):
            # Sample a random negative that isn't from this example
            candidates = [neg for neg in all_negatives 
                         if neg["id"] not in [n["id"] for n in example["negatives"]]]
            
            if candidates:
                sampled_negative = random.choice(candidates)
                
                # Add as a new negative
                augmented_example["negatives"].append({
                    "id": f"augmented_{sampled_negative['id']}",
                    "distance": 0.0,  # Placeholder distance
                    "sentences": sampled_negative["sentences"]
                })
        
        augmented_examples.append(augmented_example)
    
    return augmented_examples

def evaluate_model(model: TableTextEmbeddingModel,
                  examples: List[Dict[str, Any]],
                  batch_size: int = 16,
                  device: str = "cuda") -> Dict[str, float]:
    """
    Simplified evaluate_model function that doesn't depend on caching.
    
    Args:
        model: The model to evaluate
        examples: List of processed examples
        batch_size: Batch size for encoding
        device: Device to use for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate the model
    metrics = evaluate_with_cache(model, examples, batch_size)
    
    return metrics

def save_evaluation_results(metrics: Dict[str, Any], output_path: str):
    """
    Save evaluation results to a JSON file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_path: Path to save the results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to {output_path}") 