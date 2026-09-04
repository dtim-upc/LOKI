"""
Model wrapper for fine-tuning TaBERT with PEFT LoRA.

Wraps the pretrained VerticalAttentionTableBert with:
  - LoRA adapters on BERT's attention layers + vertical attention layers
  - Gradient checkpointing for memory efficiency
  - A score() method that computes context-column cosine similarity
"""

import sys
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType

from table_bert import TableBertModel
from table_bert.table import Table



_SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(p: str) -> str:
    """If path is relative, resolve it against the current working directory."""
    path = Path(p)
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path)


def _resolve_model_path(p: str) -> str:
    """If path is relative, resolve it against the TaBERT script directory."""
    path = Path(p)
    if not path.is_absolute():
        path = _SCRIPT_DIR / path
    return str(path)


class TaBERTForContrastive(nn.Module):
    """
    TaBERT wrapped for contrastive fine-tuning with LoRA.
    
    Scoring: cosine_similarity(mean_pool(context_enc), mean_pool(column_enc)) * temperature
    """

    def __init__(
        self,
        model_path: str = 'pretrained/tabert_large_k3/model.bin',
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_lora: bool = True,
        gradient_checkpointing: bool = True,
        initial_temperature: float = 10.0,
    ):
        super().__init__()

        # Load pretrained TaBERT (auto-detects VerticalAttention vs Vanilla)
        model_path = _resolve_path(model_path)
        print(f"Loading pretrained TaBERT from {model_path}...")
        self.tabert = TableBertModel.from_pretrained(model_path)
        print(f"Model type: {type(self.tabert).__name__}")

        # Learnable temperature for scaling cosine similarity
        self.log_temperature = nn.Parameter(
            torch.tensor(float(initial_temperature)).log()
        )

        # Enable gradient checkpointing on the BERT backbone
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Apply LoRA
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)

        self._print_trainable_params()

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1.0, max=100.0)

    @property  
    def tokenizer(self):
        return self.tabert.tokenizer

    @property
    def device(self):
        return next(self.parameters()).device

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on the BERT encoder layers."""
        bert = self.tabert.bert
        if hasattr(bert, 'encoder') and hasattr(bert.encoder, 'layer'):
            bert.encoder.layer.gradient_checkpointing = True
            # For HuggingFace transformers BERT
            if hasattr(bert, 'gradient_checkpointing_enable'):
                bert.gradient_checkpointing_enable()
            else:
                # Manual fallback: set flag that forward() checks
                for layer in bert.encoder.layer:
                    layer.gradient_checkpointing = True
            print("Gradient checkpointing enabled on BERT encoder")

    def _apply_lora(self, r: int, alpha: int, dropout: float):
        """Apply LoRA to BERT attention layers and vertical attention layers."""
        # Identify LoRA target modules in the BERT backbone
        # Standard BERT attention layers use: query, key, value, dense
        # Vertical attention layers use: query_linear, key_linear, value_linear
        target_modules = []
        for name, module in self.tabert.named_modules():
            if isinstance(module, nn.Linear):
                # Match BERT attention projections
                short_name = name.split('.')[-1]
                if short_name in ('query', 'key', 'value', 'dense',
                                  'query_linear', 'key_linear', 'value_linear'):
                    target_modules.append(name)

        if not target_modules:
            print("WARNING: No LoRA target modules found, falling back to pattern matching")
            target_modules = ['query', 'key', 'value', 'dense']

        # Deduplicate and use module name patterns
        # PEFT needs short patterns that match across layers
        unique_short_names = list(set(n.split('.')[-1] for n in target_modules))
        print(f"LoRA target modules (patterns): {unique_short_names}")
        print(f"LoRA config: r={r}, alpha={alpha}, dropout={dropout}")

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=unique_short_names,
            bias="none",
            modules_to_save=None,
        )

        # Wrap the TaBERT model with PEFT
        self.tabert = get_peft_model(self.tabert, lora_config)
        # Ensure embedding outputs require grad so gradient checkpointing works
        # with frozen base weights. Equivalent to enable_input_require_grads()
        # which is unavailable in this PEFT version.
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        self.tabert.base_model.model.bert.embeddings.register_forward_hook(
            _make_inputs_require_grad
        )

    def _print_trainable_params(self):
        """Print trainable vs total parameter count."""
        trainable = 0
        total = 0
        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"Trainable parameters: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}%)")

    def score(
        self,
        contexts: List[List[str]],
        tables: List[Table],
    ) -> torch.Tensor:
        """
        Compute similarity scores for (context, table) pairs.
        
        Args:
            contexts: List of tokenized context strings (one per batch element)
            tables: List of TaBERT Table objects (one per batch element)
            
        Returns:
            scores: Tensor of shape (batch_size,) with similarity scores
        """
        # TaBERT encode returns:
        #   context_encoding: (batch, context_len, hidden_size)
        #   column_encoding:  (batch, num_cols, hidden_size)
        context_encoding, column_encoding, info = self.tabert.encode(
            contexts=contexts,
            tables=tables
        )

        # Get masks for proper mean pooling
        tensor_dict = info['tensor_dict']

        # Context mask handling
        if 'context_token_mask' in tensor_dict:
            ctx_mask = tensor_dict['context_token_mask']  # (batch, context_len)
            if ctx_mask.dim() == 1:
                ctx_mask = ctx_mask.unsqueeze(0)
            ctx_mask = ctx_mask.unsqueeze(-1)  # (batch, context_len, 1)
            # Masked mean pool over context tokens
            ctx_sum = (context_encoding * ctx_mask).sum(dim=1)
            ctx_count = ctx_mask.sum(dim=1).clamp(min=1)
            ctx_vec = ctx_sum / ctx_count  # (batch, hidden_size)
        else:
            ctx_vec = context_encoding.mean(dim=1)

        # Column mask handling
        if 'column_mask' in tensor_dict:
            col_mask = tensor_dict['column_mask']  # (batch, num_cols)
            if col_mask.dim() == 1:
                col_mask = col_mask.unsqueeze(0)
            col_mask = col_mask.unsqueeze(-1)  # (batch, num_cols, 1)
            col_sum = (column_encoding * col_mask).sum(dim=1)
            col_count = col_mask.sum(dim=1).clamp(min=1)
            col_vec = col_sum / col_count  # (batch, hidden_size)
        else:
            col_vec = column_encoding.mean(dim=1)

        # Cosine similarity scaled by temperature
        scores = F.cosine_similarity(ctx_vec, col_vec, dim=-1) * self.temperature

        return scores

    def score_single(self, context_tokens: List[str], table: Table) -> torch.Tensor:
        """Score a single (context, table) pair. Returns a scalar tensor."""
        return self.score([context_tokens], [table])[0]

    def save_pretrained(self, save_dir: str, save_merged: bool = True):
        """
        Save the fine-tuned model.
        
        Args:
            save_dir: Directory to save to
            save_merged: If True, also save a merged (LoRA folded into base) model
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        if hasattr(self.tabert, 'save_pretrained'):
            self.tabert.save_pretrained(str(save_path / 'lora_adapter'))
            print(f"LoRA adapter saved to {save_path / 'lora_adapter'}")

        # Save temperature
        torch.save({
            'log_temperature': self.log_temperature.data,
        }, str(save_path / 'extra_params.pt'))

        # Save merged model (produces a standalone model.bin + tb_config.json)
        if save_merged and hasattr(self.tabert, 'merge_and_unload'):
            import copy
            # Deep copy to avoid destroying LoRA adapters on the live model
            tabert_copy = copy.deepcopy(self.tabert)
            merged_model = tabert_copy.merge_and_unload()
            torch.save(merged_model.state_dict(), str(save_path / 'model_merged.bin'))
            del tabert_copy, merged_model
            print(f"Merged model saved to {save_path / 'model_merged.bin'}")

    @classmethod
    def load_finetuned(
        cls,
        save_dir: str,
        base_model_path: str = 'pretrained/tabert_large_k3/model.bin',
    ) -> 'TaBERTForContrastive':
        """Load a fine-tuned model from checkpoint."""
        from peft import PeftModel

        save_path = Path(_resolve_path(save_dir))

        # Load base model without LoRA
        wrapper = cls(
            model_path=_resolve_path(base_model_path),
            use_lora=False,
            gradient_checkpointing=False,
        )

        # Load LoRA adapter
        adapter_path = save_path / 'lora_adapter'
        if adapter_path.exists():
            wrapper.tabert = PeftModel.from_pretrained(wrapper.tabert, str(adapter_path))
            print(f"LoRA adapter loaded from {adapter_path}")

        # Load extra parameters
        extra_path = save_path / 'extra_params.pt'
        if extra_path.exists():
            extra = torch.load(str(extra_path), map_location='cpu', weights_only=True)
            wrapper.log_temperature.data = extra['log_temperature']

        return wrapper
