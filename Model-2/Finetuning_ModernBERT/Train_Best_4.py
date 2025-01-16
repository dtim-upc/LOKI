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

import torch
import torch.cuda
from datasets import Dataset
from tqdm.auto import tqdm
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

def format_table(table_content: List[List[str]], table_title: str = "", caption: str = "") -> str:
    """Format table content into a single string representation."""
    parts = []
    
    # Add title and caption if present and not empty
    if table_title and table_title.strip():
        parts.append(f"Title: {table_title}")
    if caption and caption.strip() and caption.lower() != "null":
        parts.append(f"Caption: {caption}")
    
    if table_content and len(table_content) > 0:
        headers = table_content[0]
        # Add table content
        parts.append("Table Content:")
        
        # Add header row
        parts.append("Headers: " + " | ".join(str(h) for h in headers))
        
        # Process data rows
        for i, row in enumerate(table_content[1:], 1):
            # Ensure row and headers have same length
            if len(row) > len(headers):
                row = row[:len(headers)]  # Truncate row if too long
            elif len(row) < len(headers):
                row = row + [''] * (len(headers) - len(row))  # Pad row if too short
            
            # Create row representation
            try:
                row_data = [f"{headers[j]}: {cell}" for j, cell in enumerate(row)]
                parts.append(f"Row {i}: " + " | ".join(row_data))
            except Exception as e:
                print(f"Warning: Error processing row {i}: {e}")
                print(f"Headers: {headers}")
                print(f"Row: {row}")
                continue
    
    return "\n".join(parts)

def prepare_dataset(file_path: str) -> Dataset:
    """Load and prepare the dataset from JSON file."""
    print(f"\nProcessing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise
        
    print(f"Found {len(data)} examples in {file_path}")
    
    processed_data = []
    total_pairs = 0
    for idx, item in enumerate(tqdm(data, desc="Processing examples", unit="example")):
        try:
            # Format the complete table as one unit
            table_text = format_table(
                item['anchor']['table_content'],
                item['anchor']['table_title'],
                item['anchor']['caption']
            )
            
            # Process primary positive
            if isinstance(item['primary_positive']['sentence_context'], str):
                primary_context = item['primary_positive']['sentence_context']
            else:
                try:
                    sentences = eval(item['primary_positive']['sentence_context'])
                    primary_context = " ".join(sentences)
                except Exception as e:
                    print(f"Warning: Error processing primary context in example {idx}: {e}")
                    continue
                
            # Process additional positives
            additional_positives = []
            for pos in item['additional_positives']:
                try:
                    if isinstance(pos['sentence_context'], str):
                        context = pos['sentence_context']
                    else:
                        sentences = eval(pos['sentence_context'])
                        context = " ".join(sentences)
                    additional_positives.append(context)
                except Exception as e:
                    print(f"Warning: Error processing additional positive in example {idx}: {e}")
                    continue
                
            # Process all negatives
            negatives = []
            for neg in item['negatives']:
                try:
                    if isinstance(neg['sentence_context'], str):
                        context = neg['sentence_context']
                    else:
                        sentences = eval(neg['sentence_context'])
                        context = " ".join(sentences)
                    negatives.append(context)
                except Exception as e:
                    print(f"Warning: Error processing negative in example {idx}: {e}")
                    continue
            
            if not negatives:
                print(f"Warning: No valid negatives found for example {idx}")
                continue
                
            # Create a pair with primary positive and all negatives
            processed_data.append({
                'anchor': table_text,
                'positive': primary_context,
                'negatives': negatives  # Include all negatives
            })
            total_pairs += 1
            
            # Create pairs with additional positives and all negatives
            for pos_text in additional_positives:
                processed_data.append({
                    'anchor': table_text,
                    'positive': pos_text,
                    'negatives': negatives  # Include all negatives
                })
                total_pairs += 1
                
        except Exception as e:
            print(f"Warning: Error processing example {idx}: {e}")
            continue
            
    print(f"Successfully processed {len(processed_data)} training pairs")
    print(f"Average negatives per pair: {sum(len(item['negatives']) for item in processed_data) / total_pairs:.2f}")
    return Dataset.from_list(processed_data)

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
    """Evaluator that efficiently computes accuracy using cached embeddings."""
    
    def __init__(self, eval_dataset: Dataset, name: str = '', batch_size: int = 32, 
                 unique_texts: Optional[Dict[str, str]] = None):
        """
        Initialize the evaluator.
        
        Args:
            eval_dataset: Dataset containing anchor, positive, and negative examples
            name: Name of the evaluator
            batch_size: Batch size for embedding computation
            unique_texts: Optional pre-computed unique texts dictionary
        """
        self.eval_dataset = eval_dataset
        self.name = name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use provided unique texts or collect them
        if unique_texts is not None:
            self.unique_texts_map = unique_texts
            self.texts_to_encode = list(unique_texts.items())
        else:
            self.unique_texts_map = {}
            self.texts_to_encode = self._collect_unique_texts()

    @staticmethod
    def collect_unique_texts(dataset: Dataset) -> Dict[str, str]:
        """
        Static method to collect unique texts from a dataset.
        
        Args:
            dataset: The dataset to collect unique texts from
            
        Returns:
            Dictionary mapping text identifiers to their content
        """
        print("\nCollecting unique texts...")
        unique_texts = {}
        for example in tqdm(dataset, desc="Processing examples"):
            # Cache anchor
            anchor_key = f"anchor_{example['anchor']}"
            if anchor_key not in unique_texts:
                unique_texts[anchor_key] = example['anchor']
            
            # Cache positive
            pos_key = f"pos_{example['positive']}"
            if pos_key not in unique_texts:
                unique_texts[pos_key] = example['positive']
            
            # Cache negatives
            for neg in example['negatives']:
                neg_key = f"neg_{neg}"
                if neg_key not in unique_texts:
                    unique_texts[neg_key] = neg
        
        print(f"Found {len(unique_texts)} unique texts")
        return unique_texts

    def _collect_unique_texts(self) -> List[Tuple[str, str]]:
        """Internal method to collect unique texts if not provided."""
        self.unique_texts_map = self.collect_unique_texts(self.eval_dataset)
        return list(self.unique_texts_map.items())
        
    def create_embeddings_cache(self, model: SentenceTransformer) -> Dict[str, torch.Tensor]:
        """
        Create cache of embeddings for all unique texts with batched processing.
        
        Args:
            model: The SentenceTransformer model
            
        Returns:
            Dict mapping text identifiers to their embeddings
        """
        cache = {}
        start_time = time.time()
        
        # Generate embeddings in batches using pre-collected texts
        print(f"\nComputing embeddings for {len(self.texts_to_encode)} unique texts...")
        with torch.no_grad():
            for i in tqdm(range(0, len(self.texts_to_encode), self.batch_size), desc="Computing embeddings"):
                batch = self.texts_to_encode[i:i + self.batch_size]
                keys, texts = zip(*batch)
                
                # Compute embeddings for batch
                embeddings = model.encode(
                    list(texts),
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    batch_size=self.batch_size
                )
                
                # Store in cache
                for j, key in enumerate(keys):
                    cache[key] = embeddings[j].to(self.device)
                
                # Clear GPU memory if needed
                if torch.cuda.is_available() and i % (5 * self.batch_size) == 0:
                    torch.cuda.empty_cache()
        
        computation_time = time.time() - start_time
        print(f"\nEmbedding computation completed in {computation_time:.2f} seconds")
        return cache

    def compute_similarity_matrix(self, 
                              anchor_emb: torch.Tensor,
                              pos_emb: torch.Tensor,
                              neg_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity scores between anchor-positive and anchor-negative pairs.
        
        Args:
            anchor_emb: Anchor embedding
            pos_emb: Positive example embedding
            neg_embeddings: Tensor of negative example embeddings
            
        Returns:
            Tuple of (positive similarity score, negative similarity scores)
        """
        # Ensure embeddings have correct dimensions
        if anchor_emb.dim() == 1:
            anchor_emb = anchor_emb.unsqueeze(0)  # [1, embed_dim]
            
        if pos_emb.dim() == 1:
            pos_emb = pos_emb.unsqueeze(0)  # [1, embed_dim]
            
        # Calculate positive similarity
        pos_sim = torch.nn.functional.cosine_similarity(anchor_emb, pos_emb, dim=1)  # [1]
        
        # Calculate negative similarities
        neg_sims = torch.nn.functional.cosine_similarity(
            anchor_emb.expand(neg_embeddings.size(0), -1),  # [n_neg, embed_dim]
            neg_embeddings,  # [n_neg, embed_dim]
            dim=1  # compute similarity along embedding dimension
        )  # [n_neg]
        
        return pos_sim, neg_sims

    def __call__(self, 
                 model: SentenceTransformer, 
                 output_path: Optional[str] = None,
                 epoch: int = -1,
                 steps: int = -1) -> Dict[str, float]:
        """
        Evaluate the model using cached embeddings.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if len(self.eval_dataset) == 0:
            return {}
            
        eval_start_time = time.time()
        model.to(self.device)
        model.eval()
        
        # Create embeddings cache using pre-collected texts
        embeddings_cache = self.create_embeddings_cache(model)
        
        correct_predictions = 0
        total_comparisons = 0
        
        # Evaluate using cached embeddings
        with torch.no_grad():
            for example in tqdm(self.eval_dataset, desc="Evaluating"):
                # Get embeddings from cache
                anchor_emb = embeddings_cache[f"anchor_{example['anchor']}"]
                pos_emb = embeddings_cache[f"pos_{example['positive']}"]
                
                # Get all negative embeddings for this example
                neg_embeddings = torch.stack([
                    embeddings_cache[f"neg_{neg}"] 
                    for neg in example['negatives']
                ])
                
                # Compute similarities
                pos_sim, neg_sims = self.compute_similarity_matrix(anchor_emb, pos_emb, neg_embeddings)
                
                # Compare similarities (positive should be higher than all negatives)
                correct_predictions += torch.sum(pos_sim > neg_sims).item()
                total_comparisons += len(example['negatives'])
                
                # Clear memory periodically
                if torch.cuda.is_available() and total_comparisons % 1000 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate metrics
        accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0
        eval_time = time.time() - eval_start_time
        samples_per_sec = len(self.eval_dataset) / eval_time if eval_time > 0 else 0
        
        # Return only the essential metrics
        return {
            'eval_accuracy': accuracy,
            'eval_total_comparisons': total_comparisons,
            'eval_runtime': eval_time,
            'eval_steps_per_second': samples_per_sec,
            'epoch': epoch if epoch != -1 else 0
        }

    def compute_metrics(self, model: SentenceTransformer) -> Dict[str, float]:
        """Compute metrics wrapper."""
        return self.__call__(model)

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
    def __init__(self, evaluator, loss_type: str, total_steps: int):
        self.evaluator = evaluator
        self.loss_type = loss_type
        self.current_step = 0
        self.total_steps = total_steps
        self.progress_bar = None
        self.last_log_time = time.time()
        self.log_interval = 10
        
        # Best model tracking
        self.best_accuracy = float('-inf')
        self.best_model_state = None
        self.best_epoch = 0
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
        if metrics:
            self.latest_eval_results = metrics
            
            print(f"\n{'-'*20} Epoch {self.epoch} Summary {'-'*20}")
            print(f"Loss Function: {self.loss_type}")
            print("\nCurrent Scores:")
            print(f"eval_accuracy: {metrics['eval_accuracy']:.4f}")
            print(f"eval_total_comparisons: {metrics['eval_total_comparisons']:.4f}")
            
            # Update best model tracking
            current_accuracy = metrics['eval_accuracy']
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.best_epoch = self.epoch
                if 'model' in kwargs:
                    self.best_model_state = {
                        'state_dict': {
                            name: param.detach().cpu().clone() 
                            for name, param in kwargs['model'].state_dict().items()
                        },
                        'accuracy': current_accuracy
                    }
                    print(f"\nFound new best model with accuracy: {current_accuracy:.4f}")
            
            print("\nBest Model Info:")
            print(f"Best Epoch: {self.best_epoch}")
            print(f"Best Accuracy: {self.best_accuracy:.4f}")
            print("-" * 60)
            
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch."""
        return control

    def get_best_model(self, model: SentenceTransformer) -> SentenceTransformer:
        """Return the best model encountered during training."""
        if self.best_model_state is not None:
            try:
                print(f"Loading best model from epoch {self.best_epoch}...")
                model.load_state_dict(self.best_model_state['state_dict'])
                print("Successfully loaded best model state")
                return model
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
    """Train the model with enhanced memory management and best model tracking."""
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
        
        # Create evaluator with shared unique texts
        evaluator = ComprehensiveEvaluator(
            eval_dataset=eval_dataset,
            name="table-text-eval",
            batch_size=eval_batch_size,
            unique_texts=unique_texts
        )
        
        # Set up directories
        output_path = Path(output_path)
        best_model_dir = output_path / run_name / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Define training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(output_path / run_name),
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
            logging_dir=str(output_path / run_name / "logs"),
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
            
            print("\nTraining complete. Processing best model...")
            if progress_callback.best_model_state is not None:
                # Load the best state we tracked during training
                model = progress_callback.get_best_model(model)
                
                # Save best model
                final_save_path = str(best_model_dir)
                os.makedirs(final_save_path, exist_ok=True)
                print(f"\nSaving best model to {final_save_path}...")
                model.save(final_save_path)
                
                # Final verification of the saved model
                print("\nPerforming final verification of saved model...")
                loaded_model = SentenceTransformer(final_save_path)
                verification_metrics = evaluator(loaded_model)
                
                print(f"\nBest model metrics:")
                print(f"Best Epoch: {progress_callback.best_epoch}")
                print(f"Best Accuracy: {progress_callback.best_accuracy:.4f}")
                print(f"Final verified accuracy: {verification_metrics['eval_accuracy']:.4f}")
                
                # Save metrics
                metrics = {
                    'initial_metrics': initial_metrics,
                    'best_metrics': {
                        'epoch': progress_callback.best_epoch,
                        'accuracy': progress_callback.best_accuracy,
                        'total_comparisons': verification_metrics['eval_total_comparisons']
                    },
                    'verification_metrics': verification_metrics,
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
                
                metrics_path = output_path / run_name / 'training_metrics.json'
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                    
                print(f"\nMetrics saved to {metrics_path}")
            else:
                print("\nWarning: No best model state found!")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

        memory_manager.clear_memory()
        memory_manager.log_memory_stats("Final")
        
        return model
        
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
    parser.add_argument("--train_file", type=str, default="small/train_simple.json",
                       help="Path to training data JSON file")
    parser.add_argument("--eval_file", type=str, default="small/val_simple.json",
                       help="Path to evaluation data JSON file")
    parser.add_argument("--test_file", type=str, default="small/test_simple.json",
                       help="Path to test data JSON file (optional)")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory to save model outputs")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2,
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
        model = SentenceTransformer(args.model_name)
        print("Model initialized successfully")

        # Prepare datasets
        print("\nPreparing datasets...")
        train_dataset = prepare_dataset(args.train_file)
        print(f"Training examples: {len(train_dataset)}")
        
        eval_dataset = prepare_dataset(args.eval_file)
        print(f"Evaluation examples: {len(eval_dataset)}")

        # Setup output directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_shortname = args.model_name.split("/")[-1]
        run_name = f"{model_shortname}-TableText-{args.lr}-{timestamp}"
        output_path = Path(args.output_dir) / model_shortname / run_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training configuration
        config_path = output_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"\nTraining configuration saved to {config_path}")

        # Print initial configuration
        print("\n" + "="*50)
        print("Training Configuration")
        print("="*50)
        print(f"Model: {args.model_name}")
        print(f"Training examples: {len(train_dataset)}")
        print(f"Evaluation examples: {len(eval_dataset)}")
        print(f"Total epochs: {args.epochs}")
        print(f"Steps per epoch: {len(train_dataset) // (args.train_batch_size * args.gradient_accumulation_steps)}")
        print(f"Total training steps: {(len(train_dataset) // (args.train_batch_size * args.gradient_accumulation_steps)) * args.epochs}")
        print(f"Batch size: {args.train_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.train_batch_size * args.gradient_accumulation_steps}")
        print("="*50 + "\n")

        # Collect unique texts for validation set
        unique_texts = ComprehensiveEvaluator.collect_unique_texts(eval_dataset)

        # Initial model evaluation with shared unique texts
        print("\nPerforming Initial Evaluation...")
        print("-"*40)
        initial_evaluator = ComprehensiveEvaluator(
            eval_dataset=eval_dataset,
            name="initial-eval",
            batch_size=args.eval_batch_size,
            unique_texts=unique_texts
        )
        initial_metrics = initial_evaluator(model)
        
        print("\nInitial Scores:")
        print(f"eval_accuracy: {initial_metrics['eval_accuracy']:.4f}")
        print(f"eval_total_comparisons: {initial_metrics['eval_total_comparisons']}")
        print("-"*30 + "\n")

        # Initialize test-related variables
        initial_test_metrics = None
        test_dataset = None
        test_evaluator = None
        test_unique_texts = None

        # If test file is provided, evaluate initial model on test set before training
        if args.test_file:
            print("\nPerforming Initial Test Set Evaluation...")
            print("-"*40)
            
            # Load and prepare test dataset
            print(f"\nLoading test dataset from {args.test_file}...")
            test_dataset = prepare_dataset(args.test_file)
            print(f"Test examples: {len(test_dataset)}")
            
            # Compute unique texts for test set
            print("\nCollecting unique texts for test set...")
            test_unique_texts = ComprehensiveEvaluator.collect_unique_texts(test_dataset)
            
            # Create test evaluator
            test_evaluator = ComprehensiveEvaluator(
                eval_dataset=test_dataset,
                name="test-eval",
                batch_size=args.eval_batch_size,
                unique_texts=test_unique_texts
            )
            
            # Evaluate initial model on test set
            print("\nEvaluating initial model on test set...")
            initial_test_metrics = test_evaluator(model)
            
            print("\nInitial Test Scores:")
            print(f"Initial test accuracy: {initial_test_metrics['eval_accuracy']:.4f}")
            print(f"Total test comparisons: {initial_test_metrics['eval_total_comparisons']}")
            print("-"*30 + "\n")

        # Train the model with shared unique texts
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
            max_grad_norm=args.max_grad_norm
        )

        # Final evaluation using same unique texts
        print("\nPerforming Final Evaluation...")
        print("="*50)
        final_evaluator = ComprehensiveEvaluator(
            eval_dataset=eval_dataset,
            name="final-eval",
            batch_size=args.eval_batch_size,
            unique_texts=unique_texts
        )
        final_metrics = final_evaluator(trained_model)
        
        # Training summary
        print("\nResults Summary:")
        print("="*50)
        print("Accuracy Improvement on Evaluation Set:")
        print("-"*30)
        
        initial_accuracy = initial_metrics.get('eval_accuracy', 0)
        final_accuracy = final_metrics.get('eval_accuracy', 0)
        accuracy_improvement = final_accuracy - initial_accuracy
        rel_improvement = (accuracy_improvement / initial_accuracy * 100 
                         if initial_accuracy != 0 else 0)
        
        print(f"Initial: {initial_accuracy:.4f}")
        print(f"Final:   {final_accuracy:.4f}")
        print(f"Absolute Improvement: {accuracy_improvement:+.4f}")
        print(f"Relative Improvement: {rel_improvement:+.2f}%")
        print("-"*30)
        print(f"Total Comparisons: {final_metrics.get('eval_total_comparisons', 0)}")
        print("\n" + "="*50)

        # Test set final evaluation (only if test file was provided)
        if args.test_file and initial_test_metrics is not None:
            print("\nPerforming Final Test Set Evaluation...")
            print("="*50)
            
            # Load best model from disk
            best_model_path = output_path / run_name / "best_model"
            print(f"\nLoading best model from {best_model_path}...")
            best_model = SentenceTransformer(str(best_model_path))
            
            print("\nEvaluating best model on test set...")
            final_test_metrics = test_evaluator(best_model)
            
            # Calculate improvements
            test_initial_accuracy = initial_test_metrics['eval_accuracy']
            test_final_accuracy = final_test_metrics['eval_accuracy']
            test_accuracy_improvement = test_final_accuracy - test_initial_accuracy
            test_rel_improvement = (test_accuracy_improvement / test_initial_accuracy * 100 
                                  if test_initial_accuracy != 0 else 0)
            
            print("\nTest Set Results:")
            print("="*50)
            print("Accuracy Results:")
            print("-"*30)
            print(f"Initial: {test_initial_accuracy:.4f}")
            print(f"Final:   {test_final_accuracy:.4f}")
            print(f"Absolute Improvement: {test_accuracy_improvement:+.4f}")
            print(f"Relative Improvement: {test_rel_improvement:+.2f}%")
            print("-"*30)
            print(f"Total Test Comparisons: {final_test_metrics['eval_total_comparisons']}")
            print(f"Test Runtime: {final_test_metrics['eval_runtime']:.2f}s")
            
            # Save metrics
            metrics_path = output_path / run_name / 'training_metrics.json'
            try:
                with open(metrics_path, 'r') as f:
                    all_metrics = json.load(f)
            except FileNotFoundError:
                all_metrics = {}
                
            all_metrics['test_metrics'] = {
                'initial_accuracy': test_initial_accuracy,
                'final_accuracy': test_final_accuracy,
                'absolute_improvement': test_accuracy_improvement,
                'relative_improvement': test_rel_improvement,
                'total_comparisons': final_test_metrics['eval_total_comparisons'],
                'runtime': final_test_metrics['eval_runtime']
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
                
            print(f"\nTest metrics saved to {metrics_path}")

        memory_manager.clear_memory()
        wandb.finish()

    except Exception as e:
        print(f"\nError during training or evaluation: {str(e)}")
        traceback.print_exc()
        wandb.finish()
        sys.exit(1)

if __name__ == "__main__":
    main()
