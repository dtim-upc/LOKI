"""
evaluate_loki.py  —  Evaluate the LOKI model on Table-Text *discovery*.

Adapts LOKI from its native example-level evaluation to CMDL-style
dataset-wide table retrieval:

  For each query document in the test set, score it against ALL tables
  in the test set, rank the tables, and compute P@K / R@K / F1@K / AP /
  NDCG@K / MRR@K.

Supports switching between 3 LOKI best-model checkpoints via --loki_model:
  - best_model        (epoch 16)
  - best_test_ap      (best test Average Precision, epoch 3)
  - best_test_acc     (best test Overall Accuracy, epoch 4)

Uses the SAME subsampled test set as CMDL (via MAX_TEST_EXAMPLES + SEED).

Usage:
  python evaluate_loki.py
  python evaluate_loki.py --loki_model best_test_ap
  python evaluate_loki.py --max_test_examples 100
"""

import os
import sys
import json
import argparse
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from loki_runtime.models import BidirectionalTableTextModel, TableTextEmbeddingModel

from config import (
    LOKI_MODELS, LOKI_ACTIVE_MODEL, LOKI_ARGS_PATH,
    TEST_DATA_FILE, K_VALUES, OUTPUT_DIR,
    LOKI_AGGREGATION_METHOD,
    LOKI_USE_SCHEMA_AWARE_SCORER,
    LOKI_SCHEMA_AWARE_REPRESENTATION,
    LOKI_CELL_LEVEL_MATCHING_REPRESENTATION,
    MAX_TEST_EXAMPLES, MAX_QUERIES, SEED,
)
from metrics import evaluate_retrieval, print_results_table

# ===========================================================================
# Shared subsampling (same logic as evaluate_cmdl.py)
# ===========================================================================

def load_loki_json(path):
    """Load a LOKI JSON dataset file."""
    print("[LOKI] Loading JSON from %s ..." % path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("  Loaded %d examples" % len(data))
    return data


def subsample_deterministic(examples, max_examples, seed):
    """Deterministic subsampling matching LOKI's approach."""
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples

    def get_stable_key(ex):
        return ex.get("example_id", "") or str(ex.get("anchor_id", ""))

    rng = random.Random(seed)
    sorted_examples = sorted(examples, key=get_stable_key)
    sampled = rng.sample(sorted_examples, max_examples)
    print("  -> Subsampled %d examples (seed=%d)" % (len(sampled), seed))
    return sampled


# ===========================================================================
# Data extraction (from subsampled examples)
# ===========================================================================

def _extract_sentences_robust(sentences_data):
    """Extract sentence texts from MIMIC format (dict or list)."""
    texts = []
    if isinstance(sentences_data, dict):
        try:
            sorted_keys = sorted(sentences_data.keys(), key=lambda k: int(k))
        except ValueError:
            sorted_keys = sorted(sentences_data.keys())
        for k in sorted_keys:
            item = sentences_data[k]
            if isinstance(item, dict):
                t = item.get("text", "")
            elif isinstance(item, str):
                t = item
            else:
                t = ""
            if t:
                texts.append(t)
    elif isinstance(sentences_data, list):
        for item in sentences_data:
            if isinstance(item, dict):
                t = item.get("text", "")
            elif isinstance(item, str):
                t = item
            else:
                t = ""
            if t:
                texts.append(t)
    return texts


from unified_data import (
    extract_tables_and_docs_unified as _extract_tables_and_docs_unified,
    extract_tables_docs_and_schemas_unified as _extract_tables_docs_and_schemas_unified,
    extract_tables_docs_and_structures_unified as _extract_tables_docs_and_structures_unified,
    subsample_queries,
)

def extract_tables_and_docs(examples, task="DOC_TO_TABLE", dataset_format="other", native_direction="DOC_TO_TABLE"):
    return _extract_tables_and_docs_unified(examples, task=task, dataset_format=dataset_format, native_direction=native_direction)


def extract_tables_docs_and_schemas(examples, task="DOC_TO_TABLE", dataset_format="other", native_direction="DOC_TO_TABLE"):
    return _extract_tables_docs_and_schemas_unified(
        examples,
        task=task,
        dataset_format=dataset_format,
        native_direction=native_direction,
    )


def extract_tables_docs_and_structures(
    examples,
    task="DOC_TO_TABLE",
    dataset_format="other",
    native_direction="DOC_TO_TABLE",
    use_header_conditioning=False,
    use_cell_level_matching=False,
):
    return _extract_tables_docs_and_structures_unified(
        examples,
        task=task,
        dataset_format=dataset_format,
        native_direction=native_direction,
        use_header_conditioning=use_header_conditioning,
        use_cell_level_matching=use_cell_level_matching,
    )


def _runtime_backend_label(use_schema_aware_loki: bool) -> str:
    return "bundled_schema_aware" if use_schema_aware_loki else "bundled_legacy"


def _loki_schema_representation(
    use_schema_aware_loki: bool,
    checkpoint_uses_header_conditioning: bool,
    checkpoint_uses_cell_level_matching: bool,
) -> str:
    if use_schema_aware_loki:
        representation_parts = []
        if checkpoint_uses_header_conditioning:
            representation_parts.append(LOKI_SCHEMA_AWARE_REPRESENTATION)
        if checkpoint_uses_cell_level_matching:
            representation_parts.append(LOKI_CELL_LEVEL_MATCHING_REPRESENTATION)
        if representation_parts:
            return "+".join(representation_parts)
    return "legacy"


def _resolve_schema_aware_loki_setting(
    requested_use_schema_aware_loki: Optional[bool],
    checkpoint_uses_header_conditioning: bool,
    checkpoint_uses_cell_level_matching: bool,
) -> bool:
    checkpoint_requires_rewind = bool(
        checkpoint_uses_header_conditioning or checkpoint_uses_cell_level_matching
    )
    if requested_use_schema_aware_loki is None:
        return checkpoint_requires_rewind

    if checkpoint_requires_rewind and requested_use_schema_aware_loki is False:
        raise ValueError(
            "This checkpoint uses structured table features (header conditioning and/or cell-level matching). "
            "Re-run SOTA evaluation with --use_schema_aware_loki."
        )

    return bool(requested_use_schema_aware_loki)


def _normalize_schema_text_piece(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() == "nan":
        return ""
    return text


def _normalize_schema_texts(schema_texts: Any) -> List[str]:
    if schema_texts is None:
        return []
    if isinstance(schema_texts, str):
        normalized_text = _normalize_schema_text_piece(schema_texts)
        return [normalized_text] if normalized_text else []
    if isinstance(schema_texts, list):
        normalized_texts = [_normalize_schema_text_piece(text) for text in schema_texts]
        return [text for text in normalized_texts if text]
    return []


def _batch_schema_embedding(
    schema_embedding: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if schema_embedding is None:
        return None
    if schema_embedding.dim() == 1:
        schema_embedding = schema_embedding.unsqueeze(0)
    if schema_embedding.dim() == 2:
        schema_embedding = schema_embedding.unsqueeze(0)
    return schema_embedding.to(device=device, dtype=dtype)


def _batch_cell_embedding(
    cell_embedding: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if cell_embedding is None:
        return None
    if cell_embedding.dim() == 3:
        cell_embedding = cell_embedding.unsqueeze(0)
    return cell_embedding.to(device=device, dtype=dtype)


def _encode_schema_texts(
    sentence_encoder,
    schema_texts: Any,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    normalized_texts = _normalize_schema_texts(schema_texts)
    if not normalized_texts:
        return None
    schema_embeddings = sentence_encoder.encode(
        normalized_texts,
        batch_size=min(batch_size, len(normalized_texts)),
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    )
    return schema_embeddings.to(dtype=dtype)


def _encode_cell_text_rows(
    sentence_encoder,
    cell_text_rows: List[List[str]],
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if not cell_text_rows:
        return None

    num_rows = len(cell_text_rows)
    max_cols = max((len(row) for row in cell_text_rows), default=0)
    if max_cols == 0:
        return None

    flat_texts = []
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

    flat_embeddings = sentence_encoder.encode(
        flat_texts,
        batch_size=min(batch_size, len(flat_texts)),
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    ).to(dtype=dtype)
    cell_grid = torch.zeros(num_rows, max_cols, flat_embeddings.shape[-1], device=flat_embeddings.device, dtype=dtype)
    for embedding_index, (row_index, col_index) in enumerate(flat_positions):
        cell_grid[row_index, col_index] = flat_embeddings[embedding_index]
    return cell_grid


def _forward_loki_pair(
    model,
    table_embeddings: torch.Tensor,
    document_embeddings: torch.Tensor,
    aggregation_method: str,
    device: torch.device,
    model_dtype: torch.dtype,
    table_schema_embedding: Optional[torch.Tensor] = None,
    table_cell_embeddings: Optional[torch.Tensor] = None,
) -> float:
    row_batch = table_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
    sent_batch = document_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)

    model_kwargs = {
        "aggregation_method": aggregation_method,
    }
    if getattr(model, "use_header_conditioning", False):
        model_kwargs["rows_schema_embeddings"] = _batch_schema_embedding(table_schema_embedding, device, model_dtype)
        model_kwargs["sentences_schema_embeddings"] = None
    if getattr(model, "use_cell_level_matching", False):
        model_kwargs["rows_cell_embeddings"] = _batch_cell_embedding(table_cell_embeddings, device, model_dtype)
        model_kwargs["sentences_cell_embeddings"] = None

    global_sim, _pair_scores = model(
        row_batch,
        sent_batch,
        **model_kwargs,
    )

    return float(global_sim.item())


def _score_loki_pair_with_optional_chunking(
    model,
    table_embeddings: torch.Tensor,
    document_embeddings: torch.Tensor,
    aggregation_method: str,
    device: torch.device,
    model_dtype: torch.dtype,
    eval_row_chunk_size: int,
    model_top_k: int,
    table_schema_embedding: Optional[torch.Tensor] = None,
    table_cell_embeddings: Optional[torch.Tensor] = None,
) -> float:
    if eval_row_chunk_size > 0 and table_embeddings.size(0) > eval_row_chunk_size:
        safe_chunk_size = max(eval_row_chunk_size, model_top_k)

        if table_embeddings.size(0) <= safe_chunk_size:
            return _forward_loki_pair(
                model,
                table_embeddings,
                document_embeddings,
                aggregation_method,
                device,
                model_dtype,
                table_schema_embedding=table_schema_embedding,
                table_cell_embeddings=table_cell_embeddings,
            )

        chunk_sims = []
        total_rows = table_embeddings.size(0)
        idx = 0
        while idx < total_rows:
            start_idx = idx
            end_idx = idx + safe_chunk_size
            remainder = total_rows - end_idx
            if remainder > 0 and remainder < model_top_k:
                end_idx = total_rows

            row_chunk = table_embeddings[start_idx:end_idx]
            cell_chunk = None
            if table_cell_embeddings is not None:
                cell_chunk = table_cell_embeddings[start_idx:end_idx]
            chunk_sims.append(
                _forward_loki_pair(
                    model,
                    row_chunk,
                    document_embeddings,
                    aggregation_method,
                    device,
                    model_dtype,
                    table_schema_embedding=table_schema_embedding,
                    table_cell_embeddings=cell_chunk,
                )
            )
            idx = end_idx

        return max(chunk_sims)

    return _forward_loki_pair(
        model,
        table_embeddings,
        document_embeddings,
        aggregation_method,
        device,
        model_dtype,
        table_schema_embedding=table_schema_embedding,
        table_cell_embeddings=table_cell_embeddings,
    )


def _extract_eval_view(
    examples,
    task,
    dataset_format,
    native_direction,
    use_schema_aware_loki: bool,
    checkpoint_uses_header_conditioning: bool,
    checkpoint_uses_cell_level_matching: bool,
):
    if use_schema_aware_loki and (checkpoint_uses_header_conditioning or checkpoint_uses_cell_level_matching):
        return extract_tables_docs_and_structures(
            examples,
            task=task,
            dataset_format=dataset_format,
            native_direction=native_direction,
            use_header_conditioning=checkpoint_uses_header_conditioning,
            use_cell_level_matching=checkpoint_uses_cell_level_matching,
        )

    tables_dict, docs_dict, gt_map = extract_tables_and_docs(
        examples,
        task=task,
        dataset_format=dataset_format,
        native_direction=native_direction,
    )
    return tables_dict, docs_dict, gt_map, {}, {}

# ===========================================================================
# Model loading
# ===========================================================================

def load_loki_model(checkpoint_path, args_json_path, device="cuda", use_schema_aware_loki: Optional[bool] = None):
    """Load the trained LOKI BidirectionalTableTextModel."""
    with open(args_json_path, "r", encoding="utf-8") as f:
        args = json.load(f)

    checkpoint_uses_header_conditioning = bool(args.get("use_header_conditioning", False))
    checkpoint_uses_cell_level_matching = bool(args.get("use_cell_level_matching", False))
    resolved_use_schema_aware_loki = _resolve_schema_aware_loki_setting(
        use_schema_aware_loki,
        checkpoint_uses_header_conditioning,
        checkpoint_uses_cell_level_matching,
    )
    from sentence_transformers import SentenceTransformer
    from hf_model_resolver import ensure_repo_local_hf_snapshot

    model_name = args.get("model_name", "abhinand/MedEmbed-small-v0.1")
    resolved_model_name, model_source = ensure_repo_local_hf_snapshot(model_name, allow_online=True)
    print(f"[LOKI] Snapshot ready for {model_name}: {resolved_model_name} ({model_source})")

    # Model-specific kwargs (e.g., Jina embeddings require a task)
    model_kwargs = {}
    if "jina" in model_name.lower():
        model_kwargs["default_task"] = "retrieval"

    is_local = os.path.isdir(resolved_model_name)

    try:
        sentence_encoder = SentenceTransformer(
            resolved_model_name,
            device=device,
            model_kwargs={**model_kwargs, "dtype": torch.bfloat16},
            trust_remote_code=True,
            token=False,
            local_files_only=is_local,
        )
        print("[LOKI] Loaded encoder %s (bfloat16)" % model_name)
    except Exception as e:
        print(f"[LOKI] Failed to load with bfloat16 ({e}); falling back to default precision")
        try:
            sentence_encoder = SentenceTransformer(
                resolved_model_name,
                device=device,
                model_kwargs=model_kwargs if model_kwargs else None,
                trust_remote_code=True,
                token=False,
                local_files_only=is_local,
            )
            print("[LOKI] Loaded encoder %s (default precision)" % model_name)
        except Exception:
            sentence_encoder = SentenceTransformer(
                resolved_model_name,
                device=device,
                model_kwargs=model_kwargs if model_kwargs else None,
                trust_remote_code=True,
                token=False,
            )
            print("[LOKI] Loaded encoder %s (online/relaxed fallback)" % model_name)

    embedding_dim = args.get("embedding_dim")
    if embedding_dim is None:
        try:
            embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
        except Exception:
            embedding_dim = 768

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Legacy inner-gate compatibility: older checkpoints stored inner attention gates
    # under forward_attention.attention_output_gate / reverse_attention.attention_output_gate
    # controlled by use_gated_attention. Current code splits this into use_inner_gate.
    legacy_inner_gate_prefixes = (
        "bidirectional_attention.forward_attention.attention_output_gate.",
        "bidirectional_attention.reverse_attention.attention_output_gate.",
    )
    if "use_inner_gate" not in args and any(str(k).startswith(legacy_inner_gate_prefixes) for k in state_dict.keys()):
        args["use_inner_gate"] = bool(args.get("use_gated_attention", False))
        print("[LOKI] Enabling legacy inner-gate compatibility for this checkpoint.")

    use_bidirectional = args.get("use_bidirectional", True)
    if use_bidirectional:
        bidirectional_kwargs = dict(
            sentence_encoder=sentence_encoder,
            embedding_dim=embedding_dim,
            native_embedding_dim=args.get("native_embedding_dim"),
            top_k=args.get("top_k", 5),
            use_self_attention=args.get("use_self_attention", False),
            self_attention_heads=args.get("self_attention_heads", 1),
            self_attention_dropout=args.get("self_attention_dropout", 0.1),
            attention_type=args.get("attention_type", "top_k_sparse"),
            use_gated_attention=args.get("use_gated_attention", True),
            gated_attention_mode=args.get("gated_attention_mode", "vector"),
            gated_attention_hidden_dim=args.get("gated_attention_hidden_dim", 0),
            gated_attention_dropout=args.get("gated_attention_dropout", 0.0),
            gated_attention_init_bias=args.get("gated_attention_init_bias", 2.0),
            use_inner_gate=args.get("use_inner_gate", False),
            sparse_top_k=args.get("sparse_top_k", 5),
            window_size=args.get("window_size", 5),
            threshold_base=args.get("threshold_base", 0.1),
            use_refinement=args.get("use_refinement", False),
            norm_type=args.get("norm_type", "rmsnorm"),
            use_qk_rmsnorm=args.get("use_qk_rmsnorm", False),
            share_weights=args.get("share_attention_weights", True),
            use_cross_attention_lora=args.get("use_cross_attention_lora", False),
            lora_rank=args.get("lora_rank", 128),
            lora_alpha=args.get("lora_alpha", 512),
            lora_dropout=args.get("lora_dropout", 0.1),
            use_latent_bottleneck=args.get("use_latent_bottleneck", False),
            latent_num=args.get("latent_num", 64),
            latent_dropout=args.get("latent_dropout", 0.0),
            disable_temperature=args.get("disable_temperature", False),
            init_method=args.get("init_method", "orthogonal"),
            init_method_params=args.get("init_method_params", None),
            verbose=False,
        )
        if resolved_use_schema_aware_loki:
            bidirectional_kwargs.update(
                use_header_conditioning=checkpoint_uses_header_conditioning,
                use_cell_level_matching=checkpoint_uses_cell_level_matching,
                cell_matching_weight=args.get("cell_matching_weight", 0.35),
                cell_matching_pooling=args.get("cell_matching_pooling", "max"),
                cell_row_fusion_weight=args.get("cell_row_fusion_weight", 0.15),
            )
        # Filter kwargs to only those accepted by the bundled LOKI runtime.
        import inspect as _inspect
        _bidir_sig = _inspect.signature(BidirectionalTableTextModel.__init__)
        _accepted = set(_bidir_sig.parameters.keys()) - {"self"}
        _unknown = set(bidirectional_kwargs) - _accepted
        if _unknown:
            print(f"[LOKI][WARN] Dropping unsupported kwargs for this BidirectionalTableTextModel: {sorted(_unknown)}")
        bidirectional_kwargs = {k: v for k, v in bidirectional_kwargs.items() if k in _accepted}
        model = BidirectionalTableTextModel(**bidirectional_kwargs)
    else:
        model = TableTextEmbeddingModel(
            sentence_encoder=sentence_encoder,
            embedding_dim=embedding_dim,
            top_k=args.get("top_k", 5),
            attention_type=args.get("attention_type", "standard"),
            sparse_top_k=args.get("sparse_top_k", 5),
            window_size=args.get("window_size", 5),
            threshold_base=args.get("threshold_base", 0.1),
            init_method=args.get("init_method", "xavier_uniform"),
            init_method_params=args.get("init_method_params", None),
            norm_type=args.get("norm_type", "layernorm"),
            attention_direction=args.get("attention_direction", "row_to_sentence"),
            use_latent_bottleneck=args.get("use_latent_bottleneck", False),
            latent_num=args.get("latent_num", 64),
            latent_dropout=args.get("latent_dropout", 0.0),
            use_gated_attention=args.get("use_gated_attention", False),
            gated_attention_mode=args.get("gated_attention_mode", "scalar"),
            gated_attention_hidden_dim=args.get("gated_attention_hidden_dim", 0),
            gated_attention_dropout=args.get("gated_attention_dropout", 0.0),
            gated_attention_init_bias=args.get("gated_attention_init_bias", 2.0),
            disable_temperature=args.get("disable_temperature", False),
            skip_ffn=args.get("skip_ffn", False),
            use_cross_attention_lora=args.get("use_cross_attention_lora", False),
            lora_rank=args.get("lora_rank", 16),
            lora_alpha=args.get("lora_alpha", 32.0),
            lora_dropout=args.get("lora_dropout", 0.1),
            verbose=False,
        )

    if hasattr(model, "bidirectional_attention"):
        setattr(model.bidirectional_attention, "attention_activation", args.get("attention_activation", "softmax"))
        setattr(model.bidirectional_attention, "attention_alpha", args.get("attention_alpha", 1.5))
    if hasattr(model, "cross_attention"):
        setattr(model.cross_attention, "attention_activation", args.get("attention_activation", "softmax"))
        setattr(model.cross_attention, "attention_alpha", args.get("attention_alpha", 1.5))

    # Remap PEFT/LoRA-wrapped keys if needed.
    # When training with Unsloth/PEFT, the saved state dict has keys like:
    #   sentence_encoder.0.auto_model.base_model.model.layers.0...
    # but SentenceTransformer at eval time expects:
    #   sentence_encoder.0.auto_model.layers.0...
    # Detect and strip the extra "base_model.model." prefix automatically.
    PEFT_PREFIX = "sentence_encoder.0.auto_model.base_model.model."
    EXPECTED_PREFIX = "sentence_encoder.0.auto_model."
    needs_remap = any(k.startswith(PEFT_PREFIX) for k in state_dict.keys())
    if needs_remap:
        remapped = {}
        num_remapped = 0
        for key, value in state_dict.items():
            if key.startswith(PEFT_PREFIX):
                new_key = EXPECTED_PREFIX + key[len(PEFT_PREFIX):]
                remapped[new_key] = value
                num_remapped += 1
            else:
                remapped[key] = value
        state_dict = remapped
        print("[LOKI] Remapped %d PEFT-wrapped state_dict keys (stripped 'base_model.model.' prefix)" % num_remapped)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("[LOKI] Model loaded from %s" % checkpoint_path)
    print("[LOKI] Runtime backend: %s" % _runtime_backend_label(resolved_use_schema_aware_loki))

    args["_resolved_use_schema_aware_loki"] = resolved_use_schema_aware_loki

    return model, sentence_encoder, args


# ===========================================================================
# Core evaluation
# ===========================================================================

def evaluate_loki_macro(
    test_file=None,
    max_test_examples=None,
    seed=None,
    loki_model_key=None,
    checkpoint_path=None,
    args_json_path=None,
    k_values=None,
    aggregation_method=None,
    encode_batch_size=64,
    return_predictions=False,
    aggregate_to_global_tables=False,
    task="DOC_TO_TABLE",
    dataset_format="other",
    native_direction="DOC_TO_TABLE",
    cache_table_embeddings=True,
    cache_doc_embeddings=False,
    return_scores=False,
    max_queries=None,
    use_schema_aware_loki: Optional[bool] = LOKI_USE_SCHEMA_AWARE_SCORER,
):
    """Run LOKI table-level discovery evaluation (Macro mode, no chunking).

    Uses cross-attention retrieval only: score every (doc, table) pair through
    the full model. Accurate but O(D×T).
    """

    test_file = test_file or TEST_DATA_FILE
    max_test_examples = max_test_examples if max_test_examples is not None else MAX_TEST_EXAMPLES
    max_queries = max_queries if max_queries is not None else MAX_QUERIES
    seed = seed if seed is not None else SEED
    loki_model_key = loki_model_key or LOKI_ACTIVE_MODEL
    args_json_path = args_json_path or LOKI_ARGS_PATH
    k_values = k_values or K_VALUES
    aggregation_method = aggregation_method or LOKI_AGGREGATION_METHOD

    # Resolve checkpoint
    if checkpoint_path is None:
        if loki_model_key not in LOKI_MODELS:
            print("[ERROR] Unknown LOKI model key: %s" % loki_model_key)
            print("  Available: %s" % ", ".join(LOKI_MODELS.keys()))
            sys.exit(1)
        checkpoint_path = LOKI_MODELS[loki_model_key]

    print("[LOKI] Using model: %s  |  retrieval_mode: cross_attention" % loki_model_key)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 0. Load model first so extraction can follow checkpoint semantics
    # ------------------------------------------------------------------
    model, sentence_encoder, train_args = load_loki_model(
        checkpoint_path,
        args_json_path,
        device,
        use_schema_aware_loki=use_schema_aware_loki,
    )
    checkpoint_uses_header_conditioning = bool(train_args.get("use_header_conditioning", False))
    checkpoint_uses_cell_level_matching = bool(train_args.get("use_cell_level_matching", False))
    resolved_use_schema_aware_loki = bool(train_args.get("_resolved_use_schema_aware_loki", False))
    loki_schema_representation = _loki_schema_representation(
        resolved_use_schema_aware_loki,
        checkpoint_uses_header_conditioning,
        checkpoint_uses_cell_level_matching,
    )
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # ------------------------------------------------------------------
    # 1. Load & subsample (SAME logic as CMDL)
    # ------------------------------------------------------------------
    examples = load_loki_json(test_file)
    examples = subsample_deterministic(examples, max_test_examples, seed)

    # ------------------------------------------------------------------
    # 2. Extract tables, documents, ground truth
    # ------------------------------------------------------------------
    tables_dict, docs_dict, gt_map, table_schemas_dict, table_cells_dict = _extract_eval_view(
        examples,
        task,
        dataset_format,
        native_direction,
        resolved_use_schema_aware_loki,
        checkpoint_uses_header_conditioning,
        checkpoint_uses_cell_level_matching,
    )

    if max_queries and max_queries > 0:
        gt_map = subsample_queries(gt_map, max_queries, seed)

    # No source aggregation performed.

    fragment_to_global = {}
    if aggregate_to_global_tables:
        for ex in examples:
            anchor_id = ex.get("anchor_id")
            if anchor_id is None:
                continue
            metadata = ex.get("anchor_metadata", "")
            # e.g., "Title: drugbank-drugs_links.csv" -> "drugbank-drugs_links.csv"
            if metadata.startswith("Title: "):
                global_table = metadata[7:].strip()
            else:
                global_table = metadata.strip()
            fragment_to_global[str(anchor_id)] = global_table
            
        print("[LOKI] Global Table Aggregation is ENABLED.")
        print("  Mapped %d table fragments to %d unique global tables." % (
            len(fragment_to_global), len(set(fragment_to_global.values()))
        ))

    print("  Unique tables:    %d" % len(tables_dict))
    print("  Unique documents: %d" % len(docs_dict))
    print("  GT queries:       %d" % len(gt_map))

    # ------------------------------------------------------------------
    # 4. Pre-encode tables and documents (conditionally cached)
    # ------------------------------------------------------------------
    table_ids = sorted(tables_dict.keys(), key=str)
    table_embeddings = {}
    table_schema_embeddings = {}
    table_cell_embeddings = {}

    if cache_table_embeddings:
        print("[LOKI] Pre-encoding all tables (rows) — cached on GPU ...")
        for tid in tqdm(table_ids, desc="Encoding tables"):
            rows = tables_dict[tid]
            emb = sentence_encoder.encode(
                rows, batch_size=encode_batch_size,
                convert_to_tensor=True, normalize_embeddings=True,
                device=device,
            )
            table_embeddings[tid] = emb
    else:
        print("[LOKI] Table embeddings will be encoded on-the-fly (not cached).")

    if checkpoint_uses_header_conditioning:
        print("[LOKI] Pre-encoding all table schemas ...")
        for tid in tqdm(table_ids, desc="Encoding table schemas"):
            schema_emb = _encode_schema_texts(
                sentence_encoder,
                table_schemas_dict.get(tid, []),
                encode_batch_size,
                device,
                model_dtype,
            )
            if schema_emb is not None:
                table_schema_embeddings[tid] = schema_emb

    if checkpoint_uses_cell_level_matching:
        if cache_table_embeddings:
            print("[LOKI] Pre-encoding all table cell grids ...")
            for tid in tqdm(table_ids, desc="Encoding table cells"):
                cell_emb = _encode_cell_text_rows(
                    sentence_encoder,
                    table_cells_dict.get(tid, []),
                    encode_batch_size,
                    device,
                    model_dtype,
                )
                if cell_emb is not None:
                    table_cell_embeddings[tid] = cell_emb
        else:
            print("[LOKI] Table cell grids will be encoded on-the-fly (not cached).")

    doc_ids = sorted(docs_dict.keys(), key=str)
    doc_embeddings = {}
    # Choose which docs to pre-encode: for TABLE_TO_DOC we need all candidate docs;
    # for DOC_TO_TABLE we may only pre-encode query docs (docs that are queries).
    if task.upper() == "TABLE_TO_DOC":
        docs_to_encode = doc_ids
    else:
        docs_to_encode = [did for did in doc_ids if did in gt_map]

    if cache_doc_embeddings:
        print("[LOKI] Pre-encoding documents (sentences) — cached on GPU ...")
        for did in tqdm(docs_to_encode, desc="Encoding documents"):
            sents = docs_dict[did]
            emb = sentence_encoder.encode(
                sents, batch_size=encode_batch_size,
                convert_to_tensor=True, normalize_embeddings=True,
                device=device,
            )
            doc_embeddings[did] = emb
    else:
        print("[LOKI] Document embeddings will be encoded on-the-fly (not cached).")

    # ------------------------------------------------------------------
    # 5. Score queries against candidates (task-dependent)
    # ------------------------------------------------------------------
    predictions_map = {}
    scores_map = {}

    if task.upper() == "DOC_TO_TABLE":
        query_ids = [did for did in doc_ids if did in gt_map]
        print("[LOKI] Cross-attention mode — scoring %d documents × %d tables ..." % (
            len(query_ids), len(table_ids)))

        with torch.no_grad():
            for did in tqdm(query_ids, desc="Scoring"):
                sent_emb = doc_embeddings.get(did)
                if sent_emb is None:
                    sents = docs_dict[did]
                    sent_emb = sentence_encoder.encode(
                        sents, batch_size=encode_batch_size,
                        convert_to_tensor=True, normalize_embeddings=True,
                        device=device,
                    )

                doc_scores = {}
                for tid in table_ids:
                    row_emb = table_embeddings.get(tid)
                    if row_emb is None:
                        rows = tables_dict[tid]
                        row_emb = sentence_encoder.encode(
                            rows, batch_size=encode_batch_size,
                            convert_to_tensor=True, normalize_embeddings=True,
                            device=device,
                        )
                    table_cell_emb = table_cell_embeddings.get(tid)
                    if checkpoint_uses_cell_level_matching and table_cell_emb is None:
                        table_cell_emb = _encode_cell_text_rows(
                            sentence_encoder,
                            table_cells_dict.get(tid, []),
                            encode_batch_size,
                            device,
                            model_dtype,
                        )
                    doc_scores[tid] = _forward_loki_pair(
                        model,
                        row_emb,
                        sent_emb,
                        aggregation_method,
                        model_device,
                        model_dtype,
                        table_schema_embedding=table_schema_embeddings.get(tid),
                        table_cell_embeddings=table_cell_emb,
                    )

                ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                predictions_map[did] = [tid for tid, _ in ranked]
                scores_map[did] = doc_scores
    else:
        # TABLE_TO_DOC: queries are tables, candidates are all docs
        query_ids = [tid for tid in table_ids if tid in gt_map]
        candidate_docs = doc_ids
        print("[LOKI] Cross-attention mode — scoring %d tables × %d documents ..." % (
            len(query_ids), len(candidate_docs)))

        with torch.no_grad():
            for tid in tqdm(query_ids, desc="Scoring"):
                row_emb = table_embeddings.get(tid)
                if row_emb is None:
                    rows = tables_dict[tid]
                    row_emb = sentence_encoder.encode(
                        rows, batch_size=encode_batch_size,
                        convert_to_tensor=True, normalize_embeddings=True,
                        device=device,
                    )

                table_schema_emb = table_schema_embeddings.get(tid)
                table_cell_emb = table_cell_embeddings.get(tid)
                if checkpoint_uses_cell_level_matching and table_cell_emb is None:
                    table_cell_emb = _encode_cell_text_rows(
                        sentence_encoder,
                        table_cells_dict.get(tid, []),
                        encode_batch_size,
                        device,
                        model_dtype,
                    )

                table_scores = {}
                for did in candidate_docs:
                    sent_emb = doc_embeddings.get(did)
                    if sent_emb is None:
                        sents = docs_dict[did]
                        sent_emb = sentence_encoder.encode(
                            sents, batch_size=encode_batch_size,
                            convert_to_tensor=True, normalize_embeddings=True,
                            device=device,
                        )
                    table_scores[did] = _forward_loki_pair(
                        model,
                        row_emb,
                        sent_emb,
                        aggregation_method,
                        model_device,
                        model_dtype,
                        table_schema_embedding=table_schema_emb,
                        table_cell_embeddings=table_cell_emb,
                    )

                ranked = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                predictions_map[tid] = [did for did, _ in ranked]
                scores_map[tid] = table_scores

    if aggregate_to_global_tables:
        print("[LOKI] Aggregating fragment scores to global tables ...")
        global_scores_map = {}
        for did, f_scores in scores_map.items():
            g_scores = {}
            for fid, score in f_scores.items():
                gid = fragment_to_global.get(fid, fid)
                g_scores[gid] = max(g_scores.get(gid, float('-inf')), score)
            global_scores_map[did] = g_scores
            ranked = sorted(g_scores.items(), key=lambda x: x[1], reverse=True)
            predictions_map[did] = [gid for gid, _ in ranked]
        
        scores_map = global_scores_map

        new_gt_map = {}
        for did, fids in gt_map.items():
            g_fids = list(set(fragment_to_global.get(fid, fid) for fid in fids))
            new_gt_map[did] = g_fids
        gt_map = new_gt_map

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    results = evaluate_retrieval(gt_map, predictions_map, k_values, scores_map=scores_map)
    results["num_examples"] = len(examples)
    results["max_test_examples"] = max_test_examples
    results["loki_model"] = loki_model_key
    results["retrieval_mode"] = "cross_attention"
    results["loki_runtime_backend"] = _runtime_backend_label(resolved_use_schema_aware_loki)
    results["use_schema_aware_loki"] = resolved_use_schema_aware_loki
    results["checkpoint_uses_header_conditioning"] = checkpoint_uses_header_conditioning
    results["checkpoint_uses_cell_level_matching"] = checkpoint_uses_cell_level_matching
    results["loki_schema_representation"] = loki_schema_representation

    if return_scores:
        return results, scores_map, gt_map
    if return_predictions:
        return results, predictions_map, gt_map
    return results


def evaluate_loki_micro(
    test_file=None,
    max_test_examples=None,
    seed=None,
    loki_model_key=None,
    checkpoint_path=None,
    args_json_path=None,
    k_values=None,
    aggregation_method=None,
    encode_batch_size=64,
    eval_row_chunk_size=50,
    return_predictions=False,
    task="DOC_TO_TABLE",
    dataset_format="other",
    native_direction="DOC_TO_TABLE",
    cache_table_embeddings=True,
    cache_doc_embeddings=False,
    max_queries=None,
    use_schema_aware_loki: Optional[bool] = LOKI_USE_SCHEMA_AWARE_SCORER,
):
    """
    Evaluates LOKI for Micro metrics using exactly the logic from evaluate_loki_native.py, 
    including `eval_row_chunk_size` limit and caching rules, while supporting JSON datasets directly.
    """
    test_file = test_file or TEST_DATA_FILE
    max_test_examples = max_test_examples if max_test_examples is not None else MAX_TEST_EXAMPLES
    max_queries = max_queries if max_queries is not None else MAX_QUERIES
    seed = seed if seed is not None else SEED
    loki_model_key = loki_model_key or LOKI_ACTIVE_MODEL
    args_json_path = args_json_path or LOKI_ARGS_PATH
    k_values = k_values or K_VALUES
    aggregation_method = aggregation_method or LOKI_AGGREGATION_METHOD
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is None:
        if loki_model_key not in LOKI_MODELS:
            sys.exit(1)
        checkpoint_path = LOKI_MODELS[loki_model_key]

    model, sentence_encoder, args = load_loki_model(
        checkpoint_path,
        args_json_path,
        device,
        use_schema_aware_loki=use_schema_aware_loki,
    )
    model_top_k = args.get("top_k", 5)
    checkpoint_uses_header_conditioning = bool(args.get("use_header_conditioning", False))
    checkpoint_uses_cell_level_matching = bool(args.get("use_cell_level_matching", False))
    resolved_use_schema_aware_loki = bool(args.get("_resolved_use_schema_aware_loki", False))
    loki_schema_representation = _loki_schema_representation(
        resolved_use_schema_aware_loki,
        checkpoint_uses_header_conditioning,
        checkpoint_uses_cell_level_matching,
    )
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    examples = load_loki_json(test_file)
    examples = subsample_deterministic(examples, max_test_examples, seed)
    tables_dict, docs_dict, gt_map, table_schemas_dict, table_cells_dict = _extract_eval_view(
        examples,
        task,
        dataset_format,
        native_direction,
        resolved_use_schema_aware_loki,
        checkpoint_uses_header_conditioning,
        checkpoint_uses_cell_level_matching,
    )

    if max_queries and max_queries > 0:
        gt_map = subsample_queries(gt_map, max_queries, seed)

    table_ids = list(tables_dict.keys())
    doc_ids = list(docs_dict.keys())

    table_embeddings = {}
    table_cell_embeddings = {}
    if cache_table_embeddings:
        print("[LOKI-MICRO] Pre-encoding all tables (rows) — cached on GPU ...")
        for tid in tqdm(table_ids, desc="Encoding tables (micro pass)"):
            rows = tables_dict[tid]
            emb = sentence_encoder.encode(rows, batch_size=encode_batch_size, convert_to_tensor=True, normalize_embeddings=True, device=device)
            table_embeddings[tid] = emb
    else:
        print("[LOKI-MICRO] Table embeddings will be encoded on-the-fly (not cached).")

    table_schema_embeddings = {}
    if checkpoint_uses_header_conditioning:
        print("[LOKI-MICRO] Pre-encoding all table schemas ...")
        for tid in tqdm(table_ids, desc="Encoding table schemas (micro pass)"):
            schema_emb = _encode_schema_texts(
                sentence_encoder,
                table_schemas_dict.get(tid, []),
                encode_batch_size,
                device,
                model_dtype,
            )
            if schema_emb is not None:
                table_schema_embeddings[tid] = schema_emb

    if checkpoint_uses_cell_level_matching:
        if cache_table_embeddings:
            print("[LOKI-MICRO] Pre-encoding all table cell grids ...")
            for tid in tqdm(table_ids, desc="Encoding table cells (micro pass)"):
                cell_emb = _encode_cell_text_rows(
                    sentence_encoder,
                    table_cells_dict.get(tid, []),
                    encode_batch_size,
                    device,
                    model_dtype,
                )
                if cell_emb is not None:
                    table_cell_embeddings[tid] = cell_emb
        else:
            print("[LOKI-MICRO] Table cell grids will be encoded on-the-fly (not cached).")

    doc_embeddings = {}
    if task.upper() == "DOC_TO_TABLE":
        doc_ids_to_encode = [did for did in doc_ids if did in gt_map]
    else:
        doc_ids_to_encode = doc_ids

    if cache_doc_embeddings:
        print("[LOKI-MICRO] Pre-encoding all documents (sentences) — cached on GPU ...")
        for did in tqdm(doc_ids_to_encode, desc="Encoding documents (micro pass)"):
            sents = docs_dict[did]
            emb = sentence_encoder.encode(sents, batch_size=encode_batch_size, convert_to_tensor=True, normalize_embeddings=True, device=device)
            doc_embeddings[did] = emb
    else:
        print("[LOKI-MICRO] Document embeddings will be encoded on-the-fly (not cached).")

    predictions_map = {}
    scores_map = {}

    with torch.no_grad():
        if task.upper() == "DOC_TO_TABLE":
            query_ids = [did for did in doc_ids if did in gt_map]
            print(
                f"[LOKI-MICRO] Scoring {len(query_ids)} docs against {len(table_ids)} tables "
                f"(chunk size = {eval_row_chunk_size})..."
            )

            for did in tqdm(query_ids, desc="Scoring (micro pass)"):
                sent_emb = doc_embeddings.get(did)
                if sent_emb is None:
                    sents = docs_dict[did]
                    sent_emb = sentence_encoder.encode(
                        sents,
                        batch_size=encode_batch_size,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        device=device,
                    )

                doc_scores = {}
                for tid in table_ids:
                    row_emb = table_embeddings.get(tid)
                    if row_emb is None:
                        rows = tables_dict[tid]
                        row_emb = sentence_encoder.encode(
                            rows,
                            batch_size=encode_batch_size,
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            device=device,
                        )
                    table_cell_emb = table_cell_embeddings.get(tid)
                    if checkpoint_uses_cell_level_matching and table_cell_emb is None:
                        table_cell_emb = _encode_cell_text_rows(
                            sentence_encoder,
                            table_cells_dict.get(tid, []),
                            encode_batch_size,
                            device,
                            model_dtype,
                        )

                    doc_scores[tid] = _score_loki_pair_with_optional_chunking(
                        model,
                        row_emb,
                        sent_emb,
                        aggregation_method,
                        model_device,
                        model_dtype,
                        eval_row_chunk_size,
                        model_top_k,
                        table_schema_embedding=table_schema_embeddings.get(tid),
                        table_cell_embeddings=table_cell_emb,
                    )

                ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                predictions_map[did] = [tid for tid, _ in ranked]
                scores_map[did] = doc_scores
        else:
            query_ids = [tid for tid in table_ids if tid in gt_map]
            print(
                f"[LOKI-MICRO] Scoring {len(query_ids)} tables against {len(doc_ids)} documents "
                f"(chunk size = {eval_row_chunk_size})..."
            )

            for tid in tqdm(query_ids, desc="Scoring (micro pass)"):
                row_emb = table_embeddings.get(tid)
                if row_emb is None:
                    rows = tables_dict[tid]
                    row_emb = sentence_encoder.encode(
                        rows,
                        batch_size=encode_batch_size,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        device=device,
                    )
                table_cell_emb = table_cell_embeddings.get(tid)
                if checkpoint_uses_cell_level_matching and table_cell_emb is None:
                    table_cell_emb = _encode_cell_text_rows(
                        sentence_encoder,
                        table_cells_dict.get(tid, []),
                        encode_batch_size,
                        device,
                        model_dtype,
                    )

                table_scores = {}
                for did in doc_ids:
                    sent_emb = doc_embeddings.get(did)
                    if sent_emb is None:
                        sents = docs_dict[did]
                        sent_emb = sentence_encoder.encode(
                            sents,
                            batch_size=encode_batch_size,
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            device=device,
                        )

                    table_scores[did] = _score_loki_pair_with_optional_chunking(
                        model,
                        row_emb,
                        sent_emb,
                        aggregation_method,
                        model_device,
                        model_dtype,
                        eval_row_chunk_size,
                        model_top_k,
                        table_schema_embedding=table_schema_embeddings.get(tid),
                        table_cell_embeddings=table_cell_emb,
                    )

                ranked = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                predictions_map[tid] = [did for did, _ in ranked]
                scores_map[tid] = table_scores

    from metrics import evaluate_retrieval_micro
    micro_results = evaluate_retrieval_micro(gt_map, predictions_map, k_values, scores_map=scores_map)
    micro_results["num_examples"] = len(query_ids)
    micro_results["loki_model"] = loki_model_key
    micro_results["retrieval_mode"] = "cross_attention_chunked"
    micro_results["loki_runtime_backend"] = _runtime_backend_label(resolved_use_schema_aware_loki)
    micro_results["use_schema_aware_loki"] = resolved_use_schema_aware_loki
    micro_results["checkpoint_uses_header_conditioning"] = checkpoint_uses_header_conditioning
    micro_results["checkpoint_uses_cell_level_matching"] = checkpoint_uses_cell_level_matching
    micro_results["loki_schema_representation"] = loki_schema_representation

    if return_predictions:
        return micro_results, predictions_map, gt_map
    return micro_results


def evaluate_loki(
    test_file=None,
    max_test_examples=None,
    seed=None,
    loki_model_key=None,
    checkpoint_path=None,
    args_json_path=None,
    k_values=None,
    aggregation_method=None,
    encode_batch_size=64,
    return_predictions=False,
    aggregate_to_global_tables=False,
    task="DOC_TO_TABLE",
    dataset_format="other",
    native_direction="DOC_TO_TABLE",
    return_micro=False,
    eval_row_chunk_size=50,
    cache_table_embeddings=True,
    cache_doc_embeddings=False,
    return_scores=False,
    max_queries=None,
    use_schema_aware_loki: Optional[bool] = LOKI_USE_SCHEMA_AWARE_SCORER,
):
    """
    Wrapper that executes the macro pass (no chunking) by default.
    If return_micro is True, it runs TWO completely isolated evaluation passes 
    to exactly simulate the old disparate script behaviors.
    """
    macro_results = evaluate_loki_macro(
        test_file=test_file,
        max_test_examples=max_test_examples,
        seed=seed,
        loki_model_key=loki_model_key,
        checkpoint_path=checkpoint_path,
        args_json_path=args_json_path,
        k_values=k_values,
        aggregation_method=aggregation_method,
        encode_batch_size=encode_batch_size,
        return_predictions=return_predictions,
        aggregate_to_global_tables=aggregate_to_global_tables,
        task=task,
        dataset_format=dataset_format,
        native_direction=native_direction,
        cache_table_embeddings=cache_table_embeddings,
        cache_doc_embeddings=cache_doc_embeddings,
        return_scores=return_scores,
        max_queries=max_queries,
        use_schema_aware_loki=use_schema_aware_loki,
    )
    
    if return_scores:
        return macro_results  # Already a (results, scores_map, gt_map) tuple
    
    if not return_micro:
        return macro_results
        
    print("\n" + "=" * 60)
    print("  [LOKI] Initiating separate MICRO pass with chunking...")
    print("=" * 60 + "\n")
    
    micro_results = evaluate_loki_micro(
        test_file=test_file,
        max_test_examples=max_test_examples,
        seed=seed,
        loki_model_key=loki_model_key,
        checkpoint_path=checkpoint_path,
        args_json_path=args_json_path,
        k_values=k_values,
        aggregation_method=aggregation_method,
        encode_batch_size=encode_batch_size,
        eval_row_chunk_size=eval_row_chunk_size,
        return_predictions=return_predictions,
        task=task,
        dataset_format=dataset_format,
        native_direction=native_direction,
        cache_table_embeddings=cache_table_embeddings,
        cache_doc_embeddings=cache_doc_embeddings,
        max_queries=max_queries,
        use_schema_aware_loki=use_schema_aware_loki,
    )
    
    # Strip predictions to fit output sig
    # if return_predictions is True, each results branch will be a tuple
    if return_predictions:
        return macro_results[0], micro_results[0]
    return macro_results, micro_results


# ===========================================================================
# CLI
# ===========================================================================
def main():
    loki_model_names = ", ".join(LOKI_MODELS.keys())

    parser = argparse.ArgumentParser(description="LOKI Table-Text Discovery Evaluation")
    parser.add_argument("--test_file", type=str, default=TEST_DATA_FILE,
                        help="Path to test JSON (default: %s)" % TEST_DATA_FILE)
    parser.add_argument("--max_test_examples", type=int, default=MAX_TEST_EXAMPLES,
                        help="Max test examples (pool subset), 0=all (default: %d)" % MAX_TEST_EXAMPLES)
    parser.add_argument("--max_queries", type=int, default=MAX_QUERIES,
                        help="Max queries to evaluate, 0=all (default: %d)" % MAX_QUERIES)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for subsampling (default: %d)" % SEED)
    parser.add_argument("--task_direction", type=str, default="DOC_TO_TABLE",
                        choices=["DOC_TO_TABLE", "TABLE_TO_DOC"],
                        help="Direction of the retrieval task (default: DOC_TO_TABLE for Pharma)")
    parser.add_argument("--native_direction", type=str, default="DOC_TO_TABLE",
                        choices=["DOC_TO_TABLE", "TABLE_TO_DOC"],
                        help="The native direction of the source JSON (default: DOC_TO_TABLE for Flipped Pharma)")
    parser.add_argument("--dataset_format", type=str, default="other",
                        choices=["mimic", "other"],
                        help="Dataset structure format (mimic for MIMIC-IV, other for pharma/protrix)")
    parser.add_argument("--loki_model", type=str, default=LOKI_ACTIVE_MODEL,
                        choices=list(LOKI_MODELS.keys()),
                        help="Which LOKI checkpoint to use. "
                             "Options: [%s] (default: %s)" % (loki_model_names, LOKI_ACTIVE_MODEL))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Direct path to model.pt (overrides --loki_model)")
    parser.add_argument("--args_json", type=str, default=LOKI_ARGS_PATH,
                        help="Path to LOKI training args.json (default: %s)" % LOKI_ARGS_PATH)
    parser.add_argument("--aggregation_method", type=str, default=LOKI_AGGREGATION_METHOD,
                        help="Aggregation method (default: %s)" % LOKI_AGGREGATION_METHOD)
    parser.add_argument("--use_schema_aware_loki", action=argparse.BooleanOptionalAction,
                        default=LOKI_USE_SCHEMA_AWARE_SCORER,
                        help="Use structured LOKI scoring. Default auto-detects header conditioning and cell-level matching from the checkpoint args.json.")
    parser.add_argument("--encode_batch_size", type=int, default=64,
                        help="Batch size for sentence encoding (default: 64)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory (default: %s)" % OUTPUT_DIR)
    parser.add_argument("--aggregate_to_global_tables", action="store_true",
                        help="Map LOKI's row-fragment predictions to their parent global tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = evaluate_loki(
        test_file=args.test_file,
        max_test_examples=args.max_test_examples,
        max_queries=args.max_queries,
        seed=args.seed,
        task=args.task_direction,
        dataset_format=args.dataset_format,
        native_direction=args.native_direction,
        loki_model_key=args.loki_model,
        checkpoint_path=args.checkpoint,
        args_json_path=args.args_json,
        aggregation_method=args.aggregation_method,
        encode_batch_size=args.encode_batch_size,
        aggregate_to_global_tables=args.aggregate_to_global_tables,
        use_schema_aware_loki=args.use_schema_aware_loki,
    )

    print_results_table(results, "LOKI")

    out_path = os.path.join(args.output_dir, "LOKI_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[LOKI] Results saved to %s" % out_path)


if __name__ == "__main__":
    main()
