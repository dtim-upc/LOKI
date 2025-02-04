import os
import gc
import datetime
import traceback
import sys
import argparse
import json
import time
import wandb
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.training_args import BatchSamplers
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances

import torch
import torch.cuda
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    evaluation
)
from sentence_transformers.evaluation import TripletEvaluator
from transformers.trainer_callback import TrainerCallback
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
    CosineSimilarityLoss,
    MSELoss,
    TripletLoss,
    MultipleNegativesRankingLoss
)

# This will disable the loggings from Triton
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

def format_table(table_content: List[List[str]], table_title: str = "", caption: str = "") -> str:
    """Format table content into a structured representation with proper punctuation.
    
    Args:
        table_content: List of lists containing table data (first row is headers)
        table_title: Optional table title
        caption: Optional table caption
        
    Returns:
        str: Structured representation of the table
    """
    parts = []
    
    # Add title if present
    if table_title := table_title.strip():
        parts.append(f"Title: {table_title}.")
    
    # Add caption if present and not null
    if caption := caption.strip():
        if caption.lower() != "null":
            parts.append(f"Caption: {caption}.")
    
    if table_content and len(table_content) > 0:
        headers = table_content[0]
        # Headers as a list
        parts.append("Columns: " + ", ".join(map(str, headers)) + ".")
        
        # Each row as key-value pairs
        for i, row in enumerate(table_content[1:], 1):
            # Ensure row length matches headers
            row = (row + [''] * len(headers))[:len(headers)]
            # Create key-value pairs
            row_content = "; ".join(f"{h}: {c}" for h, c in zip(headers, row) if c and str(c).strip())
            if row_content:
                parts.append(f"Row {i}: {row_content}.")
    
    return "\n".join(parts)

def prepare_dataset(file_path: str, model) -> (Dataset, list):  # type: ignore
    """
    Load and prepare the dataset from a JSON file.
    
    Returns a tuple:
      (training_dataset, metadata_list)
    
    - training_dataset: a HuggingFace Dataset with only the keys needed for training:
         'anchor': the formatted table text,
         'positive': the processed positive context (primary or additional),
         'negatives': a list of negative contexts.
         
    - metadata_list: a list of dictionaries (one per training pair) containing full metadata:
         'anchor_id', 'anchor_text',
         'positive_id', 'positive_text', 'positive_type', 'positive_distance',
         'negative_ids', 'negative_texts', 'negative_distances'
    """
    print(f"\nProcessing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise
        
    print(f"Found {len(data)} examples in {file_path}")
    
    # Tracking variables for statistics.
    total_positives = 0
    total_negatives = 0
    max_lengths = {
        'table_chars': 0,
        'table_tokens': 0,
        'context_chars': 0,
        'context_tokens': 0
    }
    tokenizer = model.tokenizer
    
    training_samples = []  # For training: only text fields.
    metadata_list = []     # Full metadata including texts.
    
    def process_sentence_context(context_data, idx, context_type=""):
        """
        Process the sentence_context field.
        If the value is a list of strings, join them with spaces.
        Otherwise, print a warning and convert the value to a string.
        """
        if isinstance(context_data, list):
            return " ".join(context_data)
        else:
            print(f"Warning: sentence_context in example {idx} ({context_type}) is not a list.")
            return str(context_data)
    
    for idx, item in enumerate(tqdm(data, desc="Processing examples", unit="example")):
        try:
            # Format the complete table as one unit.
            table_text = format_table(
                item['anchor']['table_content'],
                item['anchor']['table_title'],
                item['anchor']['caption']
            )
            
            # Update table lengths.
            max_lengths['table_chars'] = max(max_lengths['table_chars'], len(table_text))
            max_lengths['table_tokens'] = max(max_lengths['table_tokens'], len(tokenizer.encode(table_text)))
            
            # Get anchor id.
            anchor_id = item['anchor'].get('id', None)
            
            # Process primary positive.
            primary_context = process_sentence_context(item['primary_positive']['sentence_context'], idx, "primary")
            primary_id = item['primary_positive'].get('id', None)
            primary_distance = item['primary_positive'].get('distance', None)
            
            # Update context lengths for primary positive.
            max_lengths['context_chars'] = max(max_lengths['context_chars'], len(primary_context))
            max_lengths['context_tokens'] = max(max_lengths['context_tokens'], len(tokenizer.encode(primary_context)))
                
            # Process additional positives.
            additional_positives = []
            for pos in item.get('additional_positives', []):
                context = process_sentence_context(pos['sentence_context'], idx, "additional")
                additional_positives.append({
                    "text": context,
                    "id": pos.get("id", None),
                    "distance": pos.get("distance", None)
                })
                
            # Process all negatives.
            negatives = []
            negative_ids = []
            negative_distances = []
            for neg in item.get('negatives', []):
                context = process_sentence_context(neg['sentence_context'], idx, "negative")
                negatives.append(context)
                negative_ids.append(neg.get("negative_id", None))
                negative_distances.append(neg.get("distance", None))
            
            if not negatives:
                print(f"Warning: No valid negatives found for example {idx}")
                continue
                
            # Create a training pair for the primary positive.
            training_pair = {
                'anchor': table_text,
                'positive': primary_context,
                'negatives': negatives
            }
            metadata = {
                'anchor_id': anchor_id,
                'anchor_text': table_text,
                'positive_id': primary_id,
                'positive_text': primary_context,
                'positive_type': 'primary',
                'positive_distance': primary_distance,
                'negative_ids': negative_ids,
                'negative_texts': negatives,
                'negative_distances': negative_distances
            }
            training_samples.append(training_pair)
            metadata_list.append(metadata)
            total_positives += 1
            total_negatives += len(negatives)
            
            # Create a training pair for each additional positive.
            for pos_item in additional_positives:
                training_pair = {
                    'anchor': table_text,
                    'positive': pos_item["text"],
                    'negatives': negatives
                }
                metadata = {
                    'anchor_id': anchor_id,
                    'anchor_text': table_text,
                    'positive_id': pos_item["id"],
                    'positive_text': pos_item["text"],
                    'positive_type': 'additional',
                    'positive_distance': pos_item.get("distance", None),
                    'negative_ids': negative_ids,
                    'negative_texts': negatives,
                    'negative_distances': negative_distances
                }
                training_samples.append(training_pair)
                metadata_list.append(metadata)
                total_positives += 1

        except Exception as e:
            print(f"Warning: Error processing example {idx}: {e}")
            continue
            
    print(f"\nDataset Statistics:")
    print(f"Total training pairs processed: {len(training_samples)}")
    if len(data) > 0:
        print(f"Average positives per original example: {total_positives/len(data):.2f}")
        print(f"Average negatives per original example: {total_negatives/len(data):.2f}")
    print("\nLength Statistics:")
    print(f"Table representations:")
    print(f"  Max characters: {max_lengths['table_chars']}")
    print(f"  Max tokens: {max_lengths['table_tokens']}")
    print(f"Sentence contexts:")
    print(f"  Max characters: {max_lengths['context_chars']}")
    print(f"  Max tokens: {max_lengths['context_tokens']}")
    print(f"\nSuccessfully processed {len(training_samples)} training pairs")
    
    return Dataset.from_list(training_samples), metadata_list

def get_loss_function(loss_type: str, model: SentenceTransformer):
    """Get the specified loss function."""
    loss_types = {
        'mnr': lambda: MultipleNegativesRankingLoss(model=model),
        'cached_mnr': lambda: CachedMultipleNegativesRankingLoss(model=model),
        'cosine': lambda: CosineSimilarityLoss(model=model),
        'mse': lambda: MSELoss(model=model),
        'triplet': lambda: TripletLoss(model=model)
    }
    
    if loss_type not in loss_types:
        raise ValueError(f"Unsupported loss type. Choose from: {', '.join(loss_types.keys())}")
    
    if loss_type in ['mnr', 'cached_mnr']:
        print(f"Using {loss_type} with multiple negatives per anchor-positive pair")
    else:
        print(f"Warning: {loss_type} does not support multiple negatives, will use only first negative")
    
    return loss_types[loss_type]()

class ComprehensiveEvaluator(SentenceEvaluator):
    """Evaluator that focuses on accuracy metrics."""
    
    def __init__(self, eval_dataset: Dataset, name: str = '', batch_size: int = 32,
                 unique_texts: Optional[Dict[str, str]] = None):
        """
        Initialize the evaluator.
        
        Args:
            eval_dataset: Dataset containing anchor, positive, and negative examples.
            name: Name of the evaluator.
            batch_size: Batch size for embedding computation.
            unique_texts: Optional pre-computed unique texts dictionary.
        """
        self.eval_dataset = eval_dataset
        self.name = name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use provided unique texts or collect them.
        if unique_texts is not None:
            self.unique_texts_map = unique_texts
            self.texts_to_encode = list(unique_texts.items())
        else:
            self.unique_texts_map = {}
            self.texts_to_encode = self._collect_unique_texts()

    @staticmethod
    def collect_unique_texts(dataset: Dataset) -> Dict[str, str]:
        """
        Static method to collect unique texts from a dataset more efficiently.
        Returns a dictionary mapping unique keys to their text content.
        """
        print("\nCollecting unique texts...")
        unique_texts = {}
        anchor_count = 0
        pos_count = 0
        neg_count = 0
        
        total_items = len(dataset)
        total_negatives = sum(len(example['negatives']) for example in dataset)
        total_operations = total_items * 2 + total_negatives  # anchors + positives + negatives
        
        with tqdm(total=total_operations, desc="Processing texts") as pbar:
            for example in dataset:
                # Cache anchor
                anchor_key = f"anchor_{example['anchor']}"
                if anchor_key not in unique_texts:
                    unique_texts[anchor_key] = example['anchor']
                    anchor_count += 1
                pbar.update(1)
                
                # Cache positive
                pos_key = f"pos_{example['positive']}"
                if pos_key not in unique_texts:
                    unique_texts[pos_key] = example['positive']
                    pos_count += 1
                pbar.update(1)
                
                # Cache negatives
                for neg in example['negatives']:
                    neg_key = f"neg_{neg}"
                    if neg_key not in unique_texts:
                        unique_texts[neg_key] = neg
                        neg_count += 1
                    pbar.update(1)
        
        print(f"\nUnique texts collected:")
        print(f"- Unique anchors: {anchor_count}")
        print(f"- Unique positives: {pos_count}")
        print(f"- Unique negatives: {neg_count}")
        print(f"Total unique texts: {len(unique_texts)}")
        
        total_chars = sum(len(text) for text in unique_texts.values())
        memory_mb = total_chars * 2 / 1024 / 1024  # Rough estimate: 2 bytes per character
        print(f"Approximate memory usage: {memory_mb:.2f} MB")
        
        return unique_texts

    def _collect_unique_texts(self) -> List[Tuple[str, str]]:
        """Internal method to collect unique texts if not provided."""
        self.unique_texts_map = self.collect_unique_texts(self.eval_dataset)
        return list(self.unique_texts_map.items())

    def create_embeddings_cache(self, model: SentenceTransformer) -> Dict[str, torch.Tensor]:
        """
        Create cache of embeddings for all unique texts with batched processing.
        """
        cache = {}
        start_time = time.time()
        
        print(f"\nComputing embeddings for {len(self.texts_to_encode)} unique texts...")
        
        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(self.texts_to_encode), self.batch_size), desc="Computing embeddings"):
                    batch = self.texts_to_encode[i:i + self.batch_size]
                    keys, texts = zip(*batch)
                    
                    try:
                        embeddings = model.encode(
                            list(texts),
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            batch_size=self.batch_size,
                            show_progress_bar=False  # Disable internal progress bar
                        )
                        
                        for j, key in enumerate(keys):
                            cache[key] = embeddings[j].to(self.device)
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"\nWarning: Out of memory with batch size {self.batch_size}")
                            for j, (key, text) in enumerate(batch):
                                try:
                                    embedding = model.encode(
                                        [text],
                                        convert_to_tensor=True,
                                        normalize_embeddings=True,
                                        batch_size=1
                                    )
                                    cache[key] = embedding[0].to(self.device)
                                except Exception as e2:
                                    print(f"Error processing individual item {j}: {str(e2)}")
                        else:
                            raise e
                    
                    if torch.cuda.is_available():
                        if i % (5 * self.batch_size) == 0:
                            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error during embedding computation: {str(e)}")
            raise
            
        computation_time = time.time() - start_time
        print(f"\nEmbedding computation completed in {computation_time:.2f} seconds")
        print(f"Cache size: {len(cache)} embeddings")
        
        return cache

    def compute_similarity_matrix(self, 
                                  anchor_emb: torch.Tensor,
                                  pos_emb: torch.Tensor,
                                  neg_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity scores between anchor-positive and anchor-negative pairs efficiently.
        
        Args:
            anchor_emb: Anchor embedding [1, embed_dim] or [embed_dim].
            pos_emb: Positive embedding [1, embed_dim] or [embed_dim].
            neg_embeddings: Negative embeddings [n_neg, embed_dim].
        
        Returns:
            Tuple of (positive similarity score, negative similarity scores).
        """
        if anchor_emb.dim() == 1:
            anchor_emb = anchor_emb.unsqueeze(0)
        if pos_emb.dim() == 1:
            pos_emb = pos_emb.unsqueeze(0)
            
        pos_sim = torch.nn.functional.cosine_similarity(anchor_emb.view(1, -1), pos_emb.view(1, -1)).squeeze()
        neg_sims = torch.nn.functional.cosine_similarity(anchor_emb.view(1, -1), neg_embeddings, dim=1)
        
        return pos_sim, neg_sims

    def __call__(self, model: SentenceTransformer, output_path: Optional[str] = None,
                epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        """Evaluate the model using accuracy metrics."""
        if len(self.eval_dataset) == 0:
            return {}
            
        eval_start_time = time.time()
        model.to(self.device)
        model.eval()
        
        embeddings_cache = self.create_embeddings_cache(model)
        
        total_comparisons = sum(len(example['negatives']) for example in self.eval_dataset)
        correct_predictions = 0
        
        with torch.no_grad():
            for example in tqdm(self.eval_dataset, desc="Computing metrics"):
                anchor_emb = embeddings_cache[f"anchor_{example['anchor']}"]
                pos_emb = embeddings_cache[f"pos_{example['positive']}"]
                neg_embeddings = torch.stack([
                    embeddings_cache[f"neg_{neg}"]
                    for neg in example['negatives']
                ])
                
                pos_sim, neg_sims = self.compute_similarity_matrix(anchor_emb, pos_emb, neg_embeddings)
                correct_predictions += torch.sum(pos_sim > neg_sims).item()
                
                if torch.cuda.is_available() and total_comparisons % 1000 == 0:
                    torch.cuda.empty_cache()
        
        accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0
        eval_time = time.time() - eval_start_time
        samples_per_sec = len(self.eval_dataset) / eval_time if eval_time > 0 else 0
        
        metrics = {
            'eval_accuracy': accuracy,
            'eval_total_comparisons': total_comparisons,
            'eval_runtime': eval_time,
            'eval_steps_per_second': samples_per_sec,
            'epoch': epoch if epoch != -1 else 0
        }
        
        print(f"\nEvaluation completed in {eval_time:.2f} seconds ({samples_per_sec:.1f} samples/sec)")
        print(f"Accuracy: {accuracy:.4f}")
        
        return metrics

    def save_best_results(self, model: SentenceTransformer, metadata_list: list, output_file: str) -> None:
        """
        Perform full evaluation using the provided metadata list and save a consolidated ranking result.
        
        This method groups the metadata by anchor so that for each anchor, all positive candidates
        (primary and additional) and negatives are merged. For each anchor, it:
        1. Computes the anchor embedding (once).
        2. Computes cosine similarity scores for each positive candidate and each negative candidate.
        3. Builds a predictions list containing all positive candidates (with relevance 2 for primary and 1 for additional)
            and all negatives (relevance 0), sorted in descending order by cosine similarity.
        4. Builds a ground truth list from the stored distance values (for both positives and negatives),
            sorted in ascending order (with missing distances treated as infinity).
        5. Saves the final consolidated output (one record per anchor) as a JSON file.
        
        Args:
            model: The SentenceTransformer model.
            metadata_list: A list of metadata dictionaries, where each dictionary contains:
                - "anchor_id", "anchor_text"
                - "positive_id", "positive_text", "positive_type", "positive_distance"
                - "negative_ids", "negative_texts", "negative_distances"
            output_file: The path to the JSON file to save the final evaluation output.
        """

        # Group metadata by anchor_id.
        grouped = {}
        for meta in metadata_list:
            anchor_id = meta.get("anchor_id")
            if anchor_id is None:
                continue  # Skip if no anchor_id is available.
            if anchor_id not in grouped:
                # Initialize a new group.
                grouped[anchor_id] = {
                    "anchor_text": meta.get("anchor_text"),
                    "positives": [],
                    "negative_ids": meta.get("negative_ids", []),
                    "negative_texts": meta.get("negative_texts", []),
                    "negative_distances": meta.get("negative_distances", [])
                }
            # Append the positive candidate info.
            grouped[anchor_id]["positives"].append({
                "positive_id": meta.get("positive_id"),
                "positive_text": meta.get("positive_text"),
                "positive_type": meta.get("positive_type"),
                "positive_distance": meta.get("positive_distance")
            })

        final_output = []
        # Process each anchor group.
        for anchor_id, group in tqdm(grouped.items(), desc="Full Evaluation with Metadata"):
            anchor_text = group.get("anchor_text")
            if anchor_text is None:
                continue  # Skip if no anchor_text is present.

            # Compute the anchor embedding (once).
            anchor_emb = model.encode(anchor_text, convert_to_tensor=True, normalize_embeddings=True)
            
            predictions = []
            ground_truth = []
            # Process all positive candidates for this anchor.
            for pos in group["positives"]:
                pos_text = pos.get("positive_text")
                if pos_text is None:
                    continue
                positive_emb = model.encode(pos_text, convert_to_tensor=True, normalize_embeddings=True)
                pos_cos_sim = torch.nn.functional.cosine_similarity(anchor_emb.unsqueeze(0), positive_emb.unsqueeze(0)).item()
                # Determine relevance: 2 for primary, 1 for additional.
                relevance = 2 if pos.get("positive_type") == "primary" else 1
                predictions.append({
                    "candidate_id": pos.get("positive_id"),
                    "cosine_similarity": pos_cos_sim,
                    "relevance": relevance
                })
                ground_truth.append({
                    "candidate_id": pos.get("positive_id"),
                    "distance": pos.get("positive_distance"),
                    "relevance": relevance
                })
            
            # Process negatives (negatives are shared across positives for a given anchor).
            neg_texts = group.get("negative_texts", [])
            for neg_id, neg_text, neg_distance in zip(group.get("negative_ids", []),
                                                    neg_texts,
                                                    group.get("negative_distances", [])):
                negative_emb = model.encode(neg_text, convert_to_tensor=True, normalize_embeddings=True)
                neg_cos_sim = torch.nn.functional.cosine_similarity(anchor_emb.unsqueeze(0), negative_emb.unsqueeze(0)).item()
                predictions.append({
                    "candidate_id": neg_id,
                    "cosine_similarity": neg_cos_sim,
                    "relevance": 0
                })
                ground_truth.append({
                    "candidate_id": neg_id,
                    "distance": neg_distance,
                    "relevance": 0
                })
            
            # Sort predictions by cosine similarity (descending order).
            predictions_sorted = sorted(predictions, key=lambda x: x["cosine_similarity"], reverse=True)
            # Sort ground truth by distance (ascending order; treat None as infinity).
            ground_truth_sorted = sorted(ground_truth, key=lambda x: x["distance"] if x["distance"] is not None else float('inf'))
            
            record = {
                "anchor_id": anchor_id,
                # "anchor_text": anchor_text,
                "predictions": predictions_sorted,
                "ground_truth": ground_truth_sorted
            }
            final_output.append(record)
        
        # Save the final output to the specified JSON file.
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)
        print(f"Full evaluation output saved to {output_file}")


class GPUMemoryManager:
    """Manages GPU memory during training."""
    @staticmethod
    def clear_memory():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2  # MB
        }
    
    @staticmethod
    def log_memory_stats(prefix: str = ""):
        """Log current GPU memory statistics."""
        if not torch.cuda.is_available():
            return
        
        stats = GPUMemoryManager.get_memory_stats()
        print(f"{prefix} GPU Memory Stats:")
        print(f"  Allocated: {stats['allocated']:.2f} MB")
        print(f"  Cached: {stats['cached']:.2f} MB")
        print(f"  Max Allocated: {stats['max_allocated']:.2f} MB")

class TrainingCallback(TrainerCallback):
    """Callback for tracking training progress with accuracy metrics."""
    
    def __init__(self, evaluator, loss_type: str, total_steps: int):
        self.evaluator = evaluator
        self.loss_type = loss_type
        self.current_step = 0
        self.total_steps = total_steps
        self.last_log_time = time.time()
        self.log_interval = 10
        
        # Best model tracking
        self.best_metrics = {
            'accuracy': {'value': float('-inf'), 'epoch': 0, 'state_dict': None},
        }
        self.epoch = 0
        
        # Store latest evaluation results
        self.latest_eval_results = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the start of each epoch."""
        self.epoch += 1
        print(f"\n{'='*20} Epoch {self.epoch} {'='*20}")
        self.current_step = 0
        self.latest_eval_results = None
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each step."""
        self.current_step += 1
            
        current_time = time.time()
        if self.current_step % self.log_interval == 0 and state.log_history:
            try:
                current_loss = state.log_history[-1].get('loss', None)
                if current_loss is not None:
                    print(f"\rStep {self.current_step}/{self.total_steps} - Loss: {current_loss:.4f}", end="")
            except (IndexError, KeyError):
                pass
            self.last_log_time = current_time
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if not metrics:
            return control
            
        self.latest_eval_results = metrics
        
        print(f"\n{'-'*20} Epoch {self.epoch} Summary {'-'*20}")
        print(f"Loss Function: {self.loss_type}")
        
        # Extract current metrics
        current_accuracy = metrics['eval_accuracy']
        
        # Track improvements for each metric
        improvements = {
            'accuracy': current_accuracy > self.best_metrics['accuracy']['value'],
        }
        
        # Update best metrics and save model state if improved
        if 'model' in kwargs:
            current_state = {
                name: param.detach().cpu().clone()
                for name, param in kwargs['model'].state_dict().items()
            }
            
            if improvements['accuracy']:
                self.best_metrics['accuracy'].update({
                    'value': current_accuracy,
                    'epoch': self.epoch,
                    'state_dict': current_state
                })
        
        # Print current scores
        print("\nCurrent Scores:")
        print(f"Accuracy: {current_accuracy:.4f}")
        
        # Print best scores
        print("\nBest Scores:")
        print(f"Best Accuracy: {self.best_metrics['accuracy']['value']:.4f} (Epoch {self.best_metrics['accuracy']['epoch']})")
        
        if any(improvements.values()):
            print("\nNew best score(s) achieved!")
            
        print("-" * 60)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        return control

    def get_best_model_state(self, metric: str = 'accuracy') -> Optional[Dict]:
        """
        Get the best model state for a specific metric.
        
        Args:
            metric: Which metric to use ('accuracy', 'precision', 'recall', 'f1' etc)
            
        Returns:
            Best model state dict or None if no state is available
        """
        if metric not in self.best_metrics:
            print(f"Warning: Unknown metric '{metric}'. Using 'accuracy' instead.")
            metric = 'accuracy'
            
        best_state = self.best_metrics[metric].get('state_dict')
        if best_state is None:
            print(f"Warning: No model state available for metric '{metric}'")
            return None
            
        return best_state

    def get_best_model(self, model: SentenceTransformer, metric: str = 'accuracy') -> SentenceTransformer:
        """
        Load the best model state for a specific metric into the provided model.
        
        Args:
            model: The model to load the state into
            metric: Which metric to use ('accuracy', 'precision', 'recall', 'f1' etc)
            
        Returns:
            Model with the best state loaded
        """
        best_state = self.get_best_model_state(metric)
        if best_state is not None:
            try:
                print(f"\nLoading best model for {metric}:")
                print(f"Epoch: {self.best_metrics[metric]['epoch']}")
                print(f"Score: {self.best_metrics[metric]['value']:.4f}")
                model.load_state_dict(best_state)
                print("Successfully loaded best model state")
            except Exception as e:
                print(f"Error loading best model: {str(e)}")
                
        return model

def train_model(
    model: SentenceTransformer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_path: Path,
    run_name: str,
    initial_metrics: Dict[str, float],
    unique_texts: Dict[str, str],
    learning_rate: float = 8e-5,
    epochs: int = 3,
    train_batch_size: int = 64,
    eval_batch_size: int = 64,
    loss_type: str = 'cached_mnr',
    warmup_ratio: float = 0.05,
    gradient_accumulation_steps: int = 8,
    max_grad_norm: float = 1.0,
    enable_checkpointing: bool = True,
) -> SentenceTransformer:
    """Train the model with memory management and best model tracking."""
    memory_manager = GPUMemoryManager()
    memory_manager.clear_memory()
    memory_manager.log_memory_stats("Initial")
    
    try:
        # Define loss function
        loss = get_loss_function(loss_type, model)
        
        # Calculate total steps
        total_steps_per_epoch = len(train_dataset) // (train_batch_size * gradient_accumulation_steps)
        if len(train_dataset) % (train_batch_size * gradient_accumulation_steps) != 0:
            total_steps_per_epoch += 1
        
        # Create evaluator with shared unique texts - now reusing unique_texts throughout training
        evaluator = ComprehensiveEvaluator(
            eval_dataset=eval_dataset,
            name="table-text-eval",
            batch_size=eval_batch_size,
            unique_texts=unique_texts  # Pass the pre-computed unique texts
        )
        
        # Set up directories
        output_path = Path(output_path)
        model_dir = output_path / run_name
        best_model_dir = model_dir / "best_models"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Define training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(model_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            fp16=True,
            bf16=False,
            learning_rate=learning_rate,
            save_strategy="no",
            eval_strategy="epoch",
            load_best_model_at_end=False,
            save_only_model=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            logging_dir=str(model_dir / "logs"),
            report_to=['wandb'],
            run_name=run_name,
            gradient_checkpointing=enable_checkpointing,
            logging_strategy="steps",
            logging_steps=100
        )

        # Create callback for progress reporting and model tracking
        progress_callback = TrainingCallback(
            evaluator=evaluator,
            loss_type=loss_type,
            total_steps=total_steps_per_epoch,
        )

        # Create trainer
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
            callbacks=[progress_callback]
        )

        # Print training setup once
        print("\nStarting Training")
        print("="*50)
        print(f"Total epochs: {epochs}")
        print(f"Steps per epoch: {total_steps_per_epoch}")
        print(f"Total training steps: {total_steps_per_epoch * epochs}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {train_batch_size * gradient_accumulation_steps}")
        
        # Train the model
        try:
            train_output = trainer.train()
            
            print("\nTraining complete. Processing best model states...")
            
            # Save best models
            metrics_to_save = ['accuracy']
            saved_metrics = {}
            
            for metric in metrics_to_save:
                if progress_callback.get_best_model_state(metric) is not None:
                    metric_dir = best_model_dir / metric
                    os.makedirs(metric_dir, exist_ok=True)
                    
                    # Load and save best model for this metric
                    best_model = progress_callback.get_best_model(model, metric)
                    best_model.save(str(metric_dir))
                    
                    # Verify the saved model using the same evaluator instance
                    # This reuses the cached embeddings and pairs
                    loaded_model = SentenceTransformer(str(metric_dir))
                    verification_metrics = evaluator(loaded_model)
                    
                    saved_metrics[metric] = {
                        'epoch': progress_callback.best_metrics[metric]['epoch'],
                        'value': progress_callback.best_metrics[metric]['value'],
                        'verification': verification_metrics
                    }
                    
                    print(f"\nSaved best model for {metric}:")
                    print(f"Epoch: {saved_metrics[metric]['epoch']}")
                    print(f"Score: {saved_metrics[metric]['value']:.4f}")
                    print(f"Verification accuracy: {verification_metrics['eval_accuracy']:.4f}")
            
            # Save comprehensive metrics
            metrics = {
                'initial_metrics': initial_metrics,
                'best_metrics': saved_metrics,
                'training_params': {
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'train_batch_size': train_batch_size,
                    'gradient_accumulation_steps': gradient_accumulation_steps,
                    'effective_batch_size': train_batch_size * gradient_accumulation_steps,
                    'warmup_ratio': warmup_ratio,
                    'max_grad_norm': max_grad_norm,
                    'loss_type': loss_type
                },
                'memory_stats': memory_manager.get_memory_stats()
            }
            
            metrics_path = model_dir / 'training_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            print(f"\nMetrics saved to {metrics_path}")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

        memory_manager.clear_memory()
        memory_manager.log_memory_stats("Final")
        
        # Return the best overall model (using accuracy metric)
        return progress_callback.get_best_model(model, 'accuracy')
        
    except Exception as e:
        memory_manager.clear_memory()
        print(f"\nError during training: {str(e)}")
        raise

def main():
    warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

    parser = argparse.ArgumentParser(description='Train a sentence transformer model with comprehensive evaluation')
    
    # Model and data arguments
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base",
                        help="Name or path of the base model to use")
    parser.add_argument("--train_file", type=str, default="data/train_simple.json",
                        help="Path to training data JSON file")
    parser.add_argument("--eval_file", type=str, default="data/val_simple.json",
                        help="Path to evaluation data JSON file")
    parser.add_argument("--test_file", type=str, default="data/test_simple.json",
                        help="Path to test data JSON file (optional)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save model outputs")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Evaluation batch size")
    parser.add_argument("--loss_type", type=str, default="cached_mnr",
                        choices=['mnr', 'cached_mnr', 'cosine', 'mse', 'triplet'],
                        help="Type of loss function to use for training")
    
    # Additional training options
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=2.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps")
    parser.add_argument("--enable_checkpointing", type=bool, default=True,
                        help="Enable gradient checkpointing for memory efficiency")
    
    args = parser.parse_args()

    try:
        # Initialize wandb
        wandb.init(
            project="LOKI",
            name=f"{args.model_name.split('/')[-1]}-TableText-{args.epochs}",
            config=vars(args)
        )

        # Initialize memory manager
        memory_manager = GPUMemoryManager()
        memory_manager.clear_memory()
        memory_manager.log_memory_stats("Initial")

        # Initialize model
        print(f"\nInitializing model {args.model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = SentenceTransformer(
                args.model_name,
                model_kwargs={"attn_implementation": "flash_attention_2"},
                device=device
            )
            print("Initialized model with Flash Attention")
        except Exception as e:
            model = SentenceTransformer(args.model_name, device=device)
            print("Initialized model without Flash Attention")
        model.to(device)
        print("Model initialized successfully")

        # Prepare datasets (training and evaluation)
        print("\nPreparing datasets...")
        train_dataset, train_metadata = prepare_dataset(args.train_file, model)
        print(f"Training examples: {len(train_dataset)}")
        eval_dataset, eval_metadata = prepare_dataset(args.eval_file, model)
        # print(eval_metadata[0]["positive_text"])
        print(f"Evaluation examples: {len(eval_dataset)}")

        # Setup output directory with timestamp.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_shortname = args.model_name.split("/")[-1]
        run_name = f"{model_shortname}-TableText-{args.lr}-{timestamp}"
        output_path = Path(args.output_dir) / model_shortname / run_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training configuration.
        config_path = output_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"\nTraining configuration saved to {config_path}")

        # Print initial configuration.
        print("\n" + "="*50)
        print("Training Configuration")
        print("="*50)
        print(f"Model: {args.model_name}")
        print(f"Training examples: {len(train_dataset)}")
        print(f"Evaluation examples: {len(eval_dataset)}")
        steps_per_epoch = len(train_dataset) // (args.train_batch_size * args.gradient_accumulation_steps)
        total_training_steps = steps_per_epoch * args.epochs
        print(f"Total epochs: {args.epochs}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_training_steps}")
        print(f"Batch size: {args.train_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.train_batch_size * args.gradient_accumulation_steps}")
        print("="*50 + "\n")

        # Collect unique texts from evaluation set.
        print("\nCollecting unique texts from validation set...")
        unique_texts = ComprehensiveEvaluator.collect_unique_texts(eval_dataset)

        # Initial evaluation on validation set.
        print("\nPerforming Initial Evaluation on Validation Set...")
        print("-"*40)
        initial_evaluator = ComprehensiveEvaluator(
            eval_dataset=eval_dataset,
            name="initial-eval",
            batch_size=args.eval_batch_size,
            unique_texts=unique_texts
        )
        initial_metrics = initial_evaluator(model)
        print("\nInitial Scores (Validation):")
        print(f"Accuracy: {initial_metrics['eval_accuracy']:.4f}")
        print("-"*30 + "\n")

        # Test set evaluation if provided.
        initial_test_metrics = None
        test_dataset = None
        test_metadata = None
        test_evaluator = None
        test_unique_texts = None
        if args.test_file:
            print("\nPerforming Initial Test Set Evaluation...")
            print("-"*40)
            test_dataset, test_metadata = prepare_dataset(args.test_file, model)
            print(f"Test examples: {len(test_dataset)}")
            print("\nCollecting unique texts for test set...")
            test_unique_texts = ComprehensiveEvaluator.collect_unique_texts(test_dataset)
            test_evaluator = ComprehensiveEvaluator(
                eval_dataset=test_dataset,
                name="test-eval",
                batch_size=args.eval_batch_size,
                unique_texts=test_unique_texts
            )
            initial_test_metrics = test_evaluator(model)
            print("\nInitial Test Scores:")
            print(f"Accuracy: {initial_test_metrics['eval_accuracy']:.4f}")
            print("-"*30 + "\n")

        # Train the model.
        print("\nStarting Model Training...")
        print("="*50)
        trained_model = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=output_path,
            run_name=run_name,
            initial_metrics=initial_metrics,
            unique_texts=unique_texts,
            learning_rate=args.lr,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            loss_type=args.loss_type,
            warmup_ratio=args.warmup_ratio,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            enable_checkpointing=args.enable_checkpointing
        )

        # Load best model once from disk.
        best_model_dir = output_path / run_name / "best_models" / "accuracy"
        if best_model_dir.exists():
            print("\nLoading best model from directory for final evaluation (Validation and Test Sets)...")
            best_model = SentenceTransformer(str(best_model_dir))
        else:
            print("Best model directory not found. Using trained model for final evaluation.")
            best_model = trained_model

        # Final evaluation on the validation set using the best model.
        print("\nPerforming Final Evaluation on Validation Set...")
        print("="*50)
        final_evaluator = ComprehensiveEvaluator(
            eval_dataset=eval_dataset,
            name="final-eval",
            batch_size=args.eval_batch_size,
            unique_texts=unique_texts
        )
        final_metrics = final_evaluator(best_model)
        print("\nResults Summary (Validation):")
        print("="*50)
        print("Evaluation Set Metrics:")
        print("-"*30)
        accuracy_improvement = final_metrics['eval_accuracy'] - initial_metrics['eval_accuracy']
        rel_accuracy_improvement = (accuracy_improvement / initial_metrics['eval_accuracy'] * 100) if initial_metrics['eval_accuracy'] != 0 else 0
        print("\nAccuracy:")
        print(f"  Initial: {initial_metrics['eval_accuracy']:.4f}")
        print(f"  Final:   {final_metrics['eval_accuracy']:.4f}")
        print(f"  Absolute Improvement: {accuracy_improvement:+.4f}")
        print(f"  Relative Improvement: {rel_accuracy_improvement:+.2f}%")
        print("-"*30)
        print(f"Total Comparisons: {final_metrics['eval_total_comparisons']}")
        print("\n" + "="*50)

        # Final evaluation on the test set using the same best model.
        if args.test_file and initial_test_metrics is not None:
            print("\nPerforming Final Test Set Evaluation...")
            print("="*50)
            print("\nEvaluating best model on test set...")
            best_model_metrics = test_evaluator(best_model)
            test_accuracy_improvement = best_model_metrics['eval_accuracy'] - initial_test_metrics['eval_accuracy']
            test_rel_accuracy_improvement = (test_accuracy_improvement / initial_test_metrics['eval_accuracy'] * 100) if initial_test_metrics['eval_accuracy'] != 0 else 0
            print("\nTest Set Results for Best Model:")
            print("\nAccuracy:")
            print(f"  Initial: {initial_test_metrics['eval_accuracy']:.4f}")
            print(f"  Final:   {best_model_metrics['eval_accuracy']:.4f}")
            print(f"  Absolute Improvement: {test_accuracy_improvement:+.4f}")
            print(f"  Relative Improvement: {test_rel_accuracy_improvement:+.2f}%")
            print("-"*30)
            
            metrics_path = output_path / run_name / 'test_metrics.json'
            test_metrics = {
                'initial_metrics': initial_test_metrics,
                'best_model_metrics': best_model_metrics
            }
            with open(metrics_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            print(f"\nTest metrics saved to {metrics_path}")

        # Full detailed evaluation using metadata (for ranking evaluation) using the best model.
        print("\nSaving Full Detailed Results with Metadata for Ranked Evaluation...")
        final_val_output_path = output_path / run_name / "full_validation_output.json"
        final_evaluator.save_best_results(best_model, eval_metadata, str(final_val_output_path))
        if args.test_file and test_metadata is not None:
            final_test_output_path = output_path / run_name / "full_test_output.json"
            test_evaluator.save_best_results(best_model, test_metadata, str(final_test_output_path))

        memory_manager.clear_memory()
        wandb.finish()

    except Exception as e:
        print(f"\nError during training or evaluation: {str(e)}")
        traceback.print_exc()
        wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()
