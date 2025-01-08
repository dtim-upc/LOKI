import os
import gc
import datetime
import traceback
import sys
import argparse
import json
import time
import wandb
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers.evaluation import SentenceEvaluator

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

class TrainingCallback(TrainerCallback):
    """Callback to print training progress and results."""
    def __init__(self, evaluator, loss_type: str, total_steps: int):
        self.evaluator = evaluator
        self.best_scores = None
        self.epoch = 0
        self.loss_type = loss_type
        self.current_step = 0
        self.total_steps = total_steps
        self.progress_bar = None
        self.last_log_time = time.time()
        self.log_interval = 10  # Show metrics every 10 steps

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the start of each epoch."""
        self.epoch += 1
        print(f"\n{'='*20} Epoch {self.epoch} {'='*20}")
        # Initialize progress bar for this epoch
        self.progress_bar = tqdm(total=self.total_steps, desc=f"Training Epoch {self.epoch}")
        self.current_step = 0
        return control
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each step."""
        self.current_step += 1
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        
            # Show metrics every log_interval steps
            current_time = time.time()
            if self.current_step % self.log_interval == 0 and state.log_history:
                # Get current loss values safely
                try:
                    current_loss = state.log_history[-1].get('loss', None)
                    if current_loss is not None:
                        self.progress_bar.set_postfix({
                            'loss': f'{current_loss:.4f}',
                            'steps': f'{self.current_step}/{self.total_steps}'
                        })
                except (IndexError, KeyError):
                    # If we can't get the loss, just show step progress
                    self.progress_bar.set_postfix({
                        'steps': f'{self.current_step}/{self.total_steps}'
                    })
                self.last_log_time = current_time
        return control
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch."""
        # Close the progress bar
        if self.progress_bar is not None:
            self.progress_bar.close()
        
        # Get metrics safely
        train_loss = None
        if state.log_history:
            try:
                # Try to get the most recent training loss
                for log in reversed(state.log_history):
                    if 'loss' in log:
                        train_loss = log['loss']
                        break
            except Exception as e:
                print(f"Warning: Could not retrieve training loss: {str(e)}")
        
        # Get evaluation scores
        scores = None
        try:
            scores = self.evaluator(model) if model is not None else None
        except Exception as e:
            print(f"Warning: Error during evaluation: {str(e)}")
            scores = {}
        
        # Update best scores
        if scores:
            if self.best_scores is None:
                self.best_scores = scores.copy()
            else:
                # Update if f1 improved
                if scores.get('eval_f1', 0) > self.best_scores.get('eval_f1', 0):
                    self.best_scores = scores.copy()
        
        # Print epoch summary
        print(f"\n{'-'*20} Epoch {self.epoch} Summary {'-'*20}")
        print(f"Loss Function: {self.loss_type}")
        if train_loss is not None:
            print(f"Training Loss: {train_loss:.4f}")
        
        if scores:
            print("\nCurrent Scores:")
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
        
        if self.best_scores:
            print("\nBest Scores:")
            for metric, value in self.best_scores.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
        
        print("-"*60)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        return control

class ComprehensiveEvaluator(SentenceEvaluator):
    """
    Enhanced evaluator that properly handles loss computation and metrics.
    """
    def __init__(self, eval_dataset: Dataset, name: str = ''):
        self.eval_dataset = eval_dataset
        self.name = name
        
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        """
        Evaluates the model and returns metrics without computing loss.
        Loss computation is handled by the trainer separately.
        """
        if len(self.eval_dataset) == 0:
            return {}
            
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        
        # Process each example
        for example in tqdm(self.eval_dataset, desc="Evaluating"):
            # Get embeddings
            anchor_embedding = model.encode([example['anchor']], convert_to_tensor=True)
            positive_embedding = model.encode([example['positive']], convert_to_tensor=True)
            negative_embeddings = model.encode(example['negatives'], convert_to_tensor=True)
            
            # Calculate similarity scores
            pos_sim = torch.nn.functional.cosine_similarity(anchor_embedding, positive_embedding)
            neg_sims = torch.nn.functional.cosine_similarity(
                anchor_embedding.repeat(len(example['negatives']), 1),
                negative_embeddings
            )
            
            # Use dynamic thresholding
            threshold = (pos_sim + neg_sims.mean()) / 2
            
            # Update metrics
            if pos_sim > threshold:
                total_tp += 1
            else:
                total_fn += 1
                
            total_tn += (neg_sims <= threshold).sum().item()
            total_fp += (neg_sims > threshold).sum().item()
        
        # Calculate final metrics
        total_pairs = total_tp + total_tn + total_fp + total_fn
        
        accuracy = (total_tp + total_tn) / total_pairs if total_pairs > 0 else 0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate additional metrics
        avg_neg_per_anchor = sum(len(example['negatives']) for example in self.eval_dataset) / len(self.eval_dataset)
        
        metrics = {
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
            'eval_avg_negatives_per_anchor': avg_neg_per_anchor,
            'eval_total_evaluated_pairs': total_pairs
        }
        
        return metrics

    def compute_metrics(self, model) -> Dict[str, float]:
        """Compute metrics for evaluation results."""
        return self.__call__(model)
    
class EarlyStopping:
    """Early stopping handler to prevent overfitting."""
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, current_value: float, model: SentenceTransformer) -> bool:
        if self.best_loss is None:
            self.best_loss = current_value
            self.best_model = model
        elif current_value < self.best_loss - self.min_delta:
            self.best_loss = current_value
            self.counter = 0
            self.best_model = model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop

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

def train_model(
    model: SentenceTransformer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_path: Path,
    run_name: str,
    initial_metrics: Dict[str, float],  # Initial Results Before Training
    learning_rate: float = 8e-5,
    epochs: int = 3,
    train_batch_size: int = 64,
    eval_batch_size: int = 64,
    loss_type: str = 'cached_mnr',
    warmup_ratio: float = 0.05,
    gradient_accumulation_steps: int = 8,
    max_grad_norm: float = 1.0,
    early_stopping_patience: int = 3,
    enable_checkpointing: bool = True,
) -> SentenceTransformer:
    """
    Train the model with enhanced memory management and early stopping.
    """
    # Initialize memory manager and clear GPU memory
    memory_manager = GPUMemoryManager()
    memory_manager.clear_memory()
    memory_manager.log_memory_stats("Initial")
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    try:
        # Define loss function
        loss = get_loss_function(loss_type, model)
        
        # Calculate total steps
        total_steps_per_epoch = len(train_dataset) // (train_batch_size * gradient_accumulation_steps)
        if len(train_dataset) % (train_batch_size * gradient_accumulation_steps) != 0:
            total_steps_per_epoch += 1
        
        # Configure progress bar display
        tqdm.pandas()
        
        # Create evaluator
        evaluator = ComprehensiveEvaluator(
            eval_dataset=eval_dataset,
            name="table-text-eval"
        )
        
        # Set up directories
        output_path = Path(output_path)
        checkpoint_dir = output_path / run_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_dir = output_path / run_name / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)

        # Define training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            fp16=True,
            bf16=False,
            learning_rate=learning_rate,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            save_only_model=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            logging_dir=str(output_path / run_name / "logs"),
            report_to=['wandb'],
            run_name=run_name,
            gradient_checkpointing=True,  # Enable gradient checkpointing
            use_flash_attention_2=True,  # Add this line to enable flash attention
        )

        # Create callback for progress reporting
        progress_callback = TrainingCallback(
            evaluator=evaluator,
            loss_type=loss_type,
            total_steps=total_steps_per_epoch
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

        best_metrics = initial_metrics.copy()
        # best_f1 = initial_metrics['eval_f1']
        
        print("\nInitial Metrics:")
        for metric, value in initial_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

        # Train the model
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        print(f"Total epochs: {epochs}")
        print(f"Steps per epoch: {total_steps_per_epoch}")
        print(f"Total training steps: {total_steps_per_epoch * epochs}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {train_batch_size * gradient_accumulation_steps}")
        
        # Train the model
        trainer.train()
        
        # Final cleanup and memory check
        memory_manager.clear_memory()
        memory_manager.log_memory_stats("Final")
        
        # Load best model and save
        if early_stopping.best_model is not None:
            model = early_stopping.best_model
        
        # Save best model
        print("\nSaving best model...")
        final_save_path = str(best_model_dir)
        os.makedirs(final_save_path, exist_ok=True)
        model.save(final_save_path)
        
        # Verify saved model
        print("\nVerifying saved model...")
        try:
            loaded_model = SentenceTransformer(final_save_path)
            verification_metrics = evaluator(loaded_model)
            
            print("\nMetrics from loaded model:")
            for metric, value in verification_metrics.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
            
            metrics_match = all(
                abs(best_metrics[k] - verification_metrics[k]) < 1e-6 
                for k in best_metrics 
                if isinstance(best_metrics[k], (int, float))
            )
            
            if metrics_match:
                print("\nVerification successful: Saved model produces identical results")
            else:
                print("\nWarning: Saved model metrics differ from best model metrics!")
                
        except Exception as e:
            print(f"\nError during model verification: {str(e)}")
        
        # Save all metrics
        metrics = {
            'initial_metrics': initial_metrics,
            'best_metrics': best_metrics,
            'verification_metrics': verification_metrics if 'verification_metrics' in locals() else None,
            'training_params': {
                'learning_rate': learning_rate,
                'epochs': epochs,
                'train_batch_size': train_batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': train_batch_size * gradient_accumulation_steps,
                'warmup_ratio': warmup_ratio,
                'max_grad_norm': max_grad_norm,
                'loss_type': loss_type,
                'early_stopping_patience': early_stopping_patience
            },
            'memory_stats': memory_manager.get_memory_stats(),
            'verification_successful': metrics_match if 'metrics_match' in locals() else False
        }
        
        metrics_path = output_path / run_name / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"\nBest model saved to {final_save_path}")
        print(f"Metrics saved to {metrics_path}")
        
        return model
        
    except Exception as e:
        memory_manager.clear_memory()
        print(f"\nError during training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train a sentence transformer model with comprehensive evaluation')
    
    # Model and data arguments
    parser.add_argument("--lr", type=float, default=8e-5,
                       help="Learning rate for training")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base",
                       help="Name or path of the base model to use")
    parser.add_argument("--train_file", type=str, default="data/train_simple.json",
                       help="Path to training data JSON file")
    parser.add_argument("--eval_file", type=str, default="data/eval_simple.json",
                       help="Path to evaluation data JSON file")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory to save model outputs")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--loss_type", type=str, default="cached_mnr",
                       choices=['mnr', 'cached_mnr', 'cosine', 'mse', 'triplet'],
                       help="Type of loss function to use for training")
    
    # Additional training options
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                       help="Ratio of warmup steps")
    
    # Arguments for early stopping and checkpointing
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--enable_checkpointing", type=bool, default=True,
                       help="Enable gradient checkpointing for memory efficiency")
    
    args = parser.parse_args()

    try:
        # Initialize wandb
        wandb.init(
            project="LOKI",
            name=f"{args.model_name.split('/')[-1]}-TableText-{args.epochs}",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "model_name": args.model_name,
                "loss_type": args.loss_type,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_grad_norm": args.max_grad_norm,
                "warmup_ratio": args.warmup_ratio,
                "early_stopping_patience": args.early_stopping_patience,
                "enable_checkpointing": args.enable_checkpointing
            }
        )

        # Initialize memory manager
        memory_manager = GPUMemoryManager()
        memory_manager.clear_memory()
        memory_manager.log_memory_stats("Initial")

        # Initialize model
        print(f"\nInitializing model {args.model_name}...")
        model = SentenceTransformer(args.model_name)
        model.config.use_flash_attention_2 = True  # Utilize flash attention
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

        # Initial model evaluation
        print("\nPerforming initial model evaluation...")
        initial_evaluator = ComprehensiveEvaluator(
            eval_dataset=eval_dataset,
            name="initial-eval"
        )
        initial_metrics = initial_evaluator(model)
        
        print("\nInitial Model Metrics:")
        print("-" * 30)
        for metric, value in initial_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Train the model
        print("\nStarting model training...")
        trained_model = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=output_path,
            run_name=run_name,
            initial_metrics=initial_metrics,
            learning_rate=args.lr,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            loss_type=args.loss_type,
            warmup_ratio=args.warmup_ratio,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm
        )

        # Final evaluation
        print("\nPerforming final evaluation...")
        final_evaluator = ComprehensiveEvaluator(
            eval_dataset=eval_dataset,
            name="final-eval"
        )
        final_metrics = final_evaluator(trained_model)
        
        # Calculate and display improvements
        print("\nTraining Results Summary:")
        print("=" * 50)
        print("\nFinal Metrics:")
        for metric in final_metrics:
            if metric in initial_metrics:
                improvement = final_metrics[metric] - initial_metrics[metric]
                rel_improvement = (improvement / initial_metrics[metric] * 100 
                                 if initial_metrics[metric] != 0 else 0)
                print(f"\n{metric}:")
                print(f"  Initial: {initial_metrics[metric]:.4f}")
                print(f"  Final: {final_metrics[metric]:.4f}")
                print(f"  Absolute Improvement: {improvement:+.4f}")
                print(f"  Relative Improvement: {rel_improvement:+.2f}%")
            else:
                print(f"\n{metric}: {final_metrics[metric]:.4f}")

        # Save final metrics
        metrics_path = output_path / "final_metrics.json"
        final_results = {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "training_params": vars(args)
        }
        with open(metrics_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nFinal metrics saved to {metrics_path}")
        
        print(f"\nTraining completed successfully!")
        print(f"Model and results saved to: {output_path}")
        
        # Add memory cleanup at the end
        memory_manager.clear_memory()
        memory_manager.log_memory_stats("Final")
        
        wandb.finish()

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        traceback.print_exc()
        wandb.finish()
        sys.exit(1)

if __name__ == "__main__":
    main()