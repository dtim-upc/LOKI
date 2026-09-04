"""
Fine-tuning script for TaBERT on the pharma_flipped_structured contrastive dataset.

Usage:
    python TaBERT/finetune.py [--data_dir Datasets/pharma_flipped_structured] [--epochs 3] [--batch_size 2] ...

Trains TaBERT with LoRA + softplus margin triplet loss + mixed precision.
"""

import os
import sys
import gc
import json
import math
import time
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from data_loader import (
    TaBERTContrastiveDataset,
    collate_triplets,
    load_row_level_dataset,
    _extract_sentences,
)
from model_wrapper import TaBERTForContrastive, _resolve_path, _resolve_model_path


@dataclass
class TrainingConfig:
    # Data
    data_dir: str = "../Datasets/pharma_flipped_structured"
    model_path: str = "pretrained/tabert_large_k3/model.bin"
    output_dir: str = "finetuned_tabert_pharma"

    # Training
    epochs: int = 5
    batch_size: int = 16
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Loss
    margin: float = 0.3
    scale: float = 10.0

    # Dataset format - for pharam_protrix_flipped, change it to False for other datasets.
    is_flipped: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_lora: bool = True

    # Model
    sample_row_num: int = 3
    max_context_len: int = 256
    gradient_checkpointing: bool = True
    initial_temperature: float = 10.0

    # Data generation
    triplet_strategy: str = "full"
    max_triplets_per_example: int = 64
    max_train_examples: Optional[int] = None
    max_val_examples: Optional[int] = None

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Misc
    seed: int = 42
    log_every: int = 10
    eval_every_epoch: bool = True
    save_merged: bool = True
    num_workers: int = 0


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_dtype(config: TrainingConfig) -> torch.dtype:
    if config.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def compute_triplet_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float,
    scale: float,
) -> torch.Tensor:
    """Softplus margin triplet loss matching LOKI's protocol."""
    diff = neg_scores - pos_scores + margin
    loss = F.softplus(diff * scale)
    return loss.mean()


def evaluate(model, val_dataset, config, device):
    """Run validation and return accuracy + mean loss."""
    model.eval()
    total_correct = 0
    total_pairs = 0
    total_loss = 0.0
    num_batches = 0

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_triplets,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    amp_dtype = get_amp_dtype(config)

    with torch.no_grad():
        for batch in val_loader:
            try:
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=config.use_amp):
                    if config.is_flipped:
                        pos_scores = model.score(batch['contexts'], batch['pos_tables'])
                        neg_scores = model.score(batch['contexts'], batch['neg_tables'])
                    else:
                        pos_scores = model.score(batch['pos_contexts'], batch['tables'])
                        neg_scores = model.score(batch['neg_contexts'], batch['tables'])
                    loss = compute_triplet_loss(
                        pos_scores, neg_scores, config.margin, config.scale
                    )
            except Exception as e:
                print(f"  Eval batch error: {e}")
                continue

            total_loss += loss.item()
            num_batches += 1
            total_correct += (pos_scores > neg_scores).sum().item()
            total_pairs += pos_scores.size(0)

    accuracy = total_correct / max(total_pairs, 1)
    avg_loss = total_loss / max(num_batches, 1)

    model.train()
    return accuracy, avg_loss


def train(config: TrainingConfig):
    # All paths resolve relative to the TaBERT script directory
    config.data_dir = _resolve_model_path(config.data_dir)
    config.model_path = _resolve_model_path(config.model_path)
    config.output_dir = _resolve_model_path(config.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    set_seed(config.seed)

    # ── Model ──
    model = TaBERTForContrastive(
        model_path=config.model_path,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_lora=config.use_lora,
        gradient_checkpointing=config.gradient_checkpointing,
        initial_temperature=config.initial_temperature,
    )
    model.to(device)

    tokenizer = model.tokenizer

    # ── Data ──
    train_path = os.path.join(config.data_dir, "train_row_level.json")
    val_path = os.path.join(config.data_dir, "val_row_level.json")

    print(f"\nLoading training data from {train_path}")
    train_dataset = TaBERTContrastiveDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        is_flipped=config.is_flipped,
        sample_row_num=config.sample_row_num,
        triplet_strategy=config.triplet_strategy,
        max_triplets_per_example=config.max_triplets_per_example,
        max_context_len=config.max_context_len,
        max_examples=config.max_train_examples,
    )

    val_dataset = None
    if os.path.exists(val_path):
        print(f"Loading validation data from {val_path}")
        val_dataset = TaBERTContrastiveDataset(
            data_path=val_path,
            tokenizer=tokenizer,
            is_flipped=config.is_flipped,
            sample_row_num=config.sample_row_num,
            triplet_strategy="primary_only",
            max_triplets_per_example=1,
            max_context_len=config.max_context_len,
            max_examples=config.max_val_examples,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_triplets,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # ── Optimizer & Scheduler ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    amp_dtype = get_amp_dtype(config)
    scaler = GradScaler('cuda', enabled=(config.use_amp and amp_dtype == torch.float16))

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"Training config:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size} x {config.grad_accum_steps} accum = {config.batch_size * config.grad_accum_steps} effective")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Margin: {config.margin}, Scale: {config.scale}")
    print(f"  AMP: {config.use_amp} ({config.amp_dtype})")
    print(f"  Train triplets: {len(train_dataset)}")
    if val_dataset:
        print(f"  Val triplets: {len(val_dataset)}")
    print(f"{'='*60}\n")

    best_val_acc = 0.0
    global_step = 0
    model.train()

    training_log = []

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_batches = 0
        optimizer.zero_grad()

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            try:
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=config.use_amp):
                    if config.is_flipped:
                        pos_scores = model.score(batch['contexts'], batch['pos_tables'])
                        neg_scores = model.score(batch['contexts'], batch['neg_tables'])
                    else:
                        pos_scores = model.score(batch['pos_contexts'], batch['tables'])
                        neg_scores = model.score(batch['neg_contexts'], batch['tables'])
                    loss = compute_triplet_loss(
                        pos_scores, neg_scores, config.margin, config.scale
                    )
                    loss = loss / config.grad_accum_steps

                scaler.scale(loss).backward()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at batch {batch_idx}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad()
                    continue
                raise

            # Track metrics
            with torch.no_grad():
                epoch_loss += loss.item() * config.grad_accum_steps
                epoch_correct += (pos_scores > neg_scores).sum().item()
                epoch_total += pos_scores.size(0)
            epoch_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % config.log_every == 0:
                    avg_loss = epoch_loss / epoch_batches
                    acc = epoch_correct / max(epoch_total, 1)
                    lr = scheduler.get_last_lr()[0]
                    temp = model.temperature.item()
                    elapsed = time.time() - start_time

                    if torch.cuda.is_available():
                        vram = torch.cuda.max_memory_allocated() / 1e9
                        print(
                            f"  [Epoch {epoch+1}/{config.epochs}] "
                            f"Step {global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | Acc: {acc:.3f} | "
                            f"LR: {lr:.2e} | Temp: {temp:.2f} | "
                            f"VRAM: {vram:.1f}GB | "
                            f"Time: {elapsed:.0f}s"
                        )
                    else:
                        print(
                            f"  [Epoch {epoch+1}/{config.epochs}] "
                            f"Step {global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | Acc: {acc:.3f} | "
                            f"LR: {lr:.2e} | Temp: {temp:.2f}"
                        )

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        epoch_acc = epoch_correct / max(epoch_total, 1)
        epoch_time = time.time() - start_time

        print(f"\n  Epoch {epoch+1} complete: Loss={avg_epoch_loss:.4f}, "
              f"TrainAcc={epoch_acc:.3f}, Time={epoch_time:.0f}s")

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_epoch_loss,
            'train_acc': epoch_acc,
            'time': epoch_time,
        }

        # Validation
        if config.eval_every_epoch and val_dataset is not None:
            print("  Running validation...")
            val_acc, val_loss = evaluate(model, val_dataset, config, device)
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
            epoch_log['val_loss'] = val_loss
            epoch_log['val_acc'] = val_acc

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  New best validation accuracy: {best_val_acc:.3f}")
                model.save_pretrained(
                    os.path.join(config.output_dir, "best"),
                    save_merged=config.save_merged
                )

        training_log.append(epoch_log)

        # Save checkpoint every epoch
        model.save_pretrained(
            os.path.join(config.output_dir, f"epoch_{epoch+1}"),
            save_merged=False,
        )

    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(
        os.path.join(config.output_dir, "final"),
        save_merged=config.save_merged
    )

    # Save training log
    log_path = os.path.join(config.output_dir, "training_log.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump({
            'config': asdict(config),
            'log': training_log,
            'best_val_acc': best_val_acc,
        }, f, indent=2)
    print(f"Training log saved to {log_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    return model


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Fine-tune TaBERT on pharma contrastive dataset")

    config = TrainingConfig()
    for fld in config.__dataclass_fields__.values():
        if fld.type == bool:
            parser.add_argument(f"--{fld.name}", type=lambda x: x.lower() in ('true', '1', 'yes'),
                                default=fld.default)
        elif fld.type == Optional[int]:
            parser.add_argument(f"--{fld.name}", type=int, default=fld.default)
        else:
            parser.add_argument(f"--{fld.name}", type=fld.type, default=fld.default)

    args = parser.parse_args()
    return TrainingConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train(config)
