"""
Unsloth Integration Module for LOKI

This module provides Unsloth-accelerated encoder loading and QLoRA patching
for faster training of embedding models.

Key Features:
1. Fast encoder loading with Unsloth's optimizations (2x faster training)
2. QLoRA adapter support for memory-efficient fine-tuning
3. Seamless integration with SentenceTransformer API
4. Compatible with BERT, RoBERTa, ModernBERT, and other encoder architectures

Usage:
    from unsloth_encoder import create_unsloth_sentence_encoder
    
    sentence_encoder = create_unsloth_sentence_encoder(
        model_name="answerdotai/ModernBERT-base",
        use_qlora=True,
        lora_rank=16,
        load_in_4bit=True
    )
"""

import torch
from typing import Optional, List, Dict, Any, Tuple, Union
import warnings

# Track whether unsloth is available
UNSLOTH_AVAILABLE = False
FAST_SENTENCE_TRANSFORMER_AVAILABLE = False
try:
    # Import unsloth first to let it patch optimizations
    import unsloth
    # Prefer FastSentenceTransformer for embedding models (specialized API)
    try:
        from unsloth import FastSentenceTransformer, is_bf16_supported
        FAST_SENTENCE_TRANSFORMER_AVAILABLE = True
        print("[Unsloth] FastSentenceTransformer imported - optimized for embeddings!")
    except ImportError:
        from unsloth import FastModel
        # Fallback to FastModel for older Unsloth versions
        print("[Unsloth] FastModel imported (FastSentenceTransformer not available)")
    UNSLOTH_AVAILABLE = True
    print("[Unsloth] Successfully imported - 2x faster training enabled!")
except ImportError as e:
    warnings.warn(f"🦥 Unsloth not available: {e}. Falling back to standard implementation.")

# Import other dependencies after unsloth
from peft import TaskType
import sentence_transformers
from sentence_transformers import SentenceTransformer, SimilarityFunction
from hf_model_resolver import ensure_repo_local_hf_snapshot

# Patch SentenceTransformer to ignore 'MaxSim' from ColBERT models
try:
    original_to_sim = SimilarityFunction.to_similarity_fn
    
    @classmethod
    def patched_to_sim(cls, value):
        if str(value).lower() in ["maxsim", "max_sim"]:
            # Fall back to cosine so SentenceTransformer doesn't crash on init
            return SimilarityFunction.COSINE
        return original_to_sim(value)
        
    SimilarityFunction.to_similarity_fn = patched_to_sim
    
    # Also patch pairwise similarity fn if present
    if hasattr(SimilarityFunction, "to_similarity_pairwise_fn"):
        original_to_sim_pairwise = SimilarityFunction.to_similarity_pairwise_fn
        
        @classmethod
        def patched_to_sim_pairwise(cls, value):
            if str(value).lower() in ["maxsim", "max_sim"]:
                return SimilarityFunction.COSINE
            return original_to_sim_pairwise(value)
            
        SimilarityFunction.to_similarity_pairwise_fn = patched_to_sim_pairwise
except Exception as e:
    print(f"Warning: Failed to patch MaxSim SimilarityFunction for ColBERT compatibility: {e}")


# Model class mapping for different encoder architectures
MODEL_CLASS_MAPPING = {
    # BERT-family models
    "bert": "transformers.BertModel",
    "roberta": "transformers.RobertaModel",
    "modernbert": "transformers.AutoModel",  # ModernBERT uses AutoModel
    "nomic": "transformers.AutoModel",
    "jina": "transformers.AutoModel",
    "qwen": "transformers.AutoModel",
    "gemma": "transformers.AutoModel",
    # Default fallback
    "default": "transformers.AutoModel",
}


def _resolve_hf_snapshot(
    model_name: str,
    *,
    allow_online: bool = True,
    verbose: bool = True,
) -> Tuple[str, str]:
    resolved_model_name, model_source = ensure_repo_local_hf_snapshot(
        model_name,
        allow_online=allow_online,
    )
    if verbose:
        print(
            f"[HF] Snapshot ready for {model_name}: {resolved_model_name} "
            f"({model_source})"
        )
    return resolved_model_name, model_source


def get_model_class(model_name: str):
    """
    Get the appropriate HuggingFace model class for the given model name.
    
    Args:
        model_name: HuggingFace model name/path
        
    Returns:
        The appropriate model class for Unsloth's auto_model parameter
    """
    model_name_lower = model_name.lower()
    
    # Determine model type from name
    if "bert" in model_name_lower and "modern" in model_name_lower:
        from transformers import AutoModel
        return AutoModel
    elif "roberta" in model_name_lower:
        from transformers import RobertaModel
        return RobertaModel
    elif "bert" in model_name_lower:
        from transformers import BertModel
        return BertModel
    elif "nomic" in model_name_lower:
        from transformers import AutoModel
        return AutoModel
    elif "jina" in model_name_lower:
        from transformers import AutoModel
        return AutoModel
    elif "qwen" in model_name_lower:
        from transformers import AutoModel
        return AutoModel
    elif "gemma" in model_name_lower:
        from transformers import AutoModel
        return AutoModel
    else:
        # Default to AutoModel for unknown architectures
        from transformers import AutoModel
        return AutoModel


def get_model_max_seq_length(model_name: str, default: int = 512) -> int:
    """
    Auto-detect the maximum sequence length for a given model from its config.
    
    This reads the model's config from HuggingFace Hub without downloading the full model.
    
    Args:
        model_name: HuggingFace model name/path
        default: Default value if auto-detection fails
        
    Returns:
        Maximum sequence length for the model
    """
    try:
        from transformers import AutoConfig
        resolved_model_name, _model_source = _resolve_hf_snapshot(
            model_name,
            allow_online=True,
            verbose=False,
        )
        
        config = AutoConfig.from_pretrained(
            resolved_model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        
        # Try different config attributes (models store this differently)
        max_len = None
        
        # Check common attribute names in order of preference
        for attr in ['max_position_embeddings', 'n_positions', 'max_seq_length', 
                     'model_max_length', 'seq_length', 'max_length']:
            if hasattr(config, attr):
                max_len = getattr(config, attr)
                if max_len is not None and isinstance(max_len, int) and max_len > 0:
                    break
                max_len = None
        
        if max_len is not None:
            # Models like ModernBERT advertise max_position_embeddings=8192 as their absolute
            # positional limit, not as a practical training sequence length.  Silently loading
            # with 8192 causes immediate OOM.  Cap to a safe training default and tell the user.
            _PRACTICAL_CAP = 512
            if max_len > _PRACTICAL_CAP:
                print(
                    f"[INFO] Auto-detected max_seq_length for {model_name}: {max_len} — "
                    f"this is the model's absolute positional limit and will cause OOM during training. "
                    f"Capping to {_PRACTICAL_CAP}. Use --override_max_seq_length to choose a different value."
                )
                max_len = _PRACTICAL_CAP
            else:
                print(f"[INFO] Auto-detected max_seq_length for {model_name}: {max_len}")
            return max_len
        else:
            print(f"[INFO] Could not auto-detect max_seq_length for {model_name}, using default {default}")
            return default
        
    except Exception as e:
        print(f"[WARNING] Error detecting max_seq_length for {model_name}: {e}")
        return default


# ============================================================================
# TARGET MODULE MAPPINGS FOR DIFFERENT MODEL ARCHITECTURES
# ============================================================================
# Different transformer architectures use different naming conventions for their
# linear layers. This mapping provides the correct target modules for LoRA.

# Architecture-specific target modules for LoRA
ARCHITECTURE_TARGET_MODULES = {
    # BERT-family models (BERT, RoBERTa, DistilBERT, etc.)
    # Layers: query, key, value (attention) + dense (FFN and attention output)
    "bert": ["query", "key", "value", "dense"],
    "roberta": ["query", "key", "value", "dense"],
    "distilbert": ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
    "electra": ["query", "key", "value", "dense"],
    "albert": ["query", "key", "value", "dense"],
    
    # ModernBERT (different naming convention)
    # Layers: Wqkv (fused QKV), Wo (output), Wi (FFN intermediate)
    "modernbert": ["Wqkv", "Wo", "Wi"],
    
    # GPT-style models (GPT2, GPT-Neo, etc.)
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neo": ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    
    # LLaMA-style models (LLaMA, Mistral, Qwen, etc.)
    # These use the q_proj, k_proj, v_proj naming convention
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # T5-style models
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi", "wo"],
    
    # Other architectures
    "xlnet": ["q", "k", "v", "o", "ff.layer_1", "ff.layer_2"],
    "deberta": ["query_proj", "key_proj", "value_proj", "dense"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    
    # Nomic/Jina embedding models (often BERT-based)
    "nomic_bert": ["Wqkv", "Wo", "Wi"],  # May use ModernBERT-style
}

# Default fallback for unknown architectures (tries common patterns)
DEFAULT_TARGET_MODULES = ["query", "key", "value", "dense"]


def detect_target_modules_for_model(model_name: str, verbose: bool = True) -> List[str]:
    """
    Auto-detect the correct LoRA target modules for a given model.
    
    This function inspects the model's architecture and returns the appropriate
    target module names for LoRA injection.
    
    Args:
        model_name: HuggingFace model name/path
        verbose: Whether to print detection information
        
    Returns:
        List of target module names for LoRA
        
    Example:
        >>> detect_target_modules_for_model("abhinand/MedEmbed-large-v0.1")
        ['query', 'key', 'value', 'dense']
        
        >>> detect_target_modules_for_model("unsloth/ModernBERT-base")
        ['Wqkv', 'Wo', 'Wi']
    """
    try:
        from transformers import AutoConfig
        resolved_model_name, _model_source = _resolve_hf_snapshot(
            model_name,
            allow_online=True,
            verbose=False,
        )
        
        config = AutoConfig.from_pretrained(
            resolved_model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_type = getattr(config, 'model_type', 'unknown').lower()
        
        if verbose:
            print(f"[INFO] Detecting target modules for {model_name} (type: {model_type})")
        
        # Check if we have a specific mapping for this architecture
        if model_type in ARCHITECTURE_TARGET_MODULES:
            target_modules = ARCHITECTURE_TARGET_MODULES[model_type]
            if verbose:
                print(f"[INFO] Using architecture-specific modules for '{model_type}': {target_modules}")
            return target_modules
        
        # Try to infer from model name if model_type doesn't match
        model_name_lower = model_name.lower()
        
        # Check for common patterns in the model name
        name_patterns = [
            ("modernbert", "modernbert"),
            ("bert", "bert"),  # This catches BERT, RoBERTa (roberta includes 'bert')
            ("roberta", "roberta"),
            ("distilbert", "distilbert"),
            ("electra", "electra"),
            ("albert", "albert"),
            ("llama", "llama"),
            ("mistral", "mistral"),
            ("qwen", "qwen"),
            ("gemma", "gemma"),
            ("phi", "phi"),
            ("t5", "t5"),
            ("nomic", "nomic_bert"),
            ("jina", "bert"),  # Jina models are often BERT-based
        ]
        
        for pattern, arch_key in name_patterns:
            if pattern in model_name_lower:
                if arch_key in ARCHITECTURE_TARGET_MODULES:
                    target_modules = ARCHITECTURE_TARGET_MODULES[arch_key]
                    if verbose:
                        print(f"[INFO] Inferred architecture '{arch_key}' from name, modules: {target_modules}")
                    return target_modules
        
        # Fallback: try to detect by loading the model and inspecting layer names
        if verbose:
            print(f"[INFO] Unknown architecture '{model_type}', attempting layer inspection...")
        
        target_modules = _detect_target_modules_by_inspection(model_name, verbose)
        return target_modules
        
    except Exception as e:
        if verbose:
            print(f"[WARNING] Error detecting target modules for {model_name}: {e}")
            print(f"[WARNING] Falling back to default BERT-style modules: {DEFAULT_TARGET_MODULES}")
        return DEFAULT_TARGET_MODULES


def _detect_target_modules_by_inspection(model_name: str, verbose: bool = True) -> List[str]:
    """
    Detect target modules by inspecting the actual model layer names.
    
    This is a fallback for unknown architectures - it loads the model config
    or a small portion and inspects the linear layer naming convention.
    
    Args:
        model_name: HuggingFace model name/path
        verbose: Whether to print detection information
        
    Returns:
        List of target module names for LoRA
    """
    try:
        from transformers import AutoModel
        resolved_model_name, _model_source = _resolve_hf_snapshot(
            model_name,
            allow_online=True,
            verbose=False,
        )
        
        # Load the model (this might be slow for large models)
        model = AutoModel.from_pretrained(
            resolved_model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        
        # Collect all linear layer name suffixes
        linear_suffixes = set()
        for name, module in model.named_modules():
            if 'Linear' in type(module).__name__:
                parts = name.split('.')
                if parts:
                    linear_suffixes.add(parts[-1])
        
        # Clean up
        del model
        
        if verbose:
            print(f"[INFO] Detected linear layer suffixes: {sorted(linear_suffixes)}")
        
        # Try to match known patterns
        # Priority: attention layers first, then FFN layers
        detected_modules = []
        
        # BERT-style attention
        bert_attention = ["query", "key", "value"]
        if all(m in linear_suffixes for m in bert_attention):
            detected_modules.extend(bert_attention)
        
        # LLaMA-style attention  
        llama_attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if all(m in linear_suffixes for m in llama_attention):
            detected_modules.extend(llama_attention)
        
        # ModernBERT-style (fused QKV)
        if "Wqkv" in linear_suffixes:
            detected_modules.append("Wqkv")
            if "Wo" in linear_suffixes:
                detected_modules.append("Wo")
        
        # FFN layers
        if "dense" in linear_suffixes:
            detected_modules.append("dense")
        if "Wi" in linear_suffixes:
            detected_modules.append("Wi")
        for ffn in ["gate_proj", "up_proj", "down_proj"]:
            if ffn in linear_suffixes:
                detected_modules.append(ffn)
        
        if detected_modules:
            if verbose:
                print(f"[INFO] Auto-detected target modules: {detected_modules}")
            return detected_modules
        
        # Ultimate fallback
        if verbose:
            print(f"[WARNING] Could not auto-detect modules, using defaults: {DEFAULT_TARGET_MODULES}")
        return DEFAULT_TARGET_MODULES
        
    except Exception as e:
        if verbose:
            print(f"[WARNING] Layer inspection failed: {e}")
        return DEFAULT_TARGET_MODULES


def get_target_modules_for_model(
    model_name: str,
    user_specified_modules: Optional[List[str]] = None,
    auto_detect: bool = True,
    verbose: bool = True
) -> List[str]:
    """
    Get the target modules for LoRA, with support for auto-detection.
    
    This is the main entry point for getting target modules. It handles:
    1. User-specified modules (highest priority)
    2. Auto-detection based on architecture (if enabled)
    3. Fallback to defaults
    
    Args:
        model_name: HuggingFace model name/path
        user_specified_modules: User-provided module list (None or empty for auto)
        auto_detect: Whether to auto-detect modules (default: True)
        verbose: Whether to print detection information
        
    Returns:
        List of target module names for LoRA
    """
    # If user specified modules (and not empty/None/"auto"), use those
    if user_specified_modules:
        if isinstance(user_specified_modules, str):
            # Handle "auto" string
            if user_specified_modules.lower() == "auto":
                pass  # Will auto-detect below
            else:
                # Parse comma-separated string
                modules = [m.strip() for m in user_specified_modules.split(",") if m.strip()]
                if modules:
                    if verbose:
                        print(f"[INFO] Using user-specified target modules: {modules}")
                    return modules
        elif isinstance(user_specified_modules, list) and len(user_specified_modules) > 0:
            # Check if it's not just ["auto"]
            if not (len(user_specified_modules) == 1 and user_specified_modules[0].lower() == "auto"):
                if verbose:
                    print(f"[INFO] Using user-specified target modules: {user_specified_modules}")
                return user_specified_modules
    
    # Auto-detect if enabled
    if auto_detect:
        return detect_target_modules_for_model(model_name, verbose=verbose)
    
    # Fallback to defaults
    if verbose:
        print(f"[INFO] Using default target modules: {DEFAULT_TARGET_MODULES}")
    return DEFAULT_TARGET_MODULES



def create_unsloth_model(
    model_name: str,
    max_seq_length: int = 512,
    load_in_4bit: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: str = "cuda",
    full_finetuning: bool = False,
) -> Tuple[Any, Any]:
    """
    Load a model with Unsloth optimizations for faster training.
    
    Uses FastSentenceTransformer (specialized for embeddings) when available,
    otherwise falls back to FastModel.
    
    Args:
        model_name: HuggingFace model name/path
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to use 4-bit quantization (QLoRA)
        dtype: Model dtype (None for auto-detect)
        device: Device to load model on
        full_finetuning: Whether to enable full finetuning (vs LoRA only)
        
    Returns:
        Tuple of (model, tokenizer) for FastModel, or just model for FastSentenceTransformer
    """
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError(
            "Unsloth is not available. Please install it with: pip install unsloth\n"
            "Or use the standard SentenceTransformer loading instead."
        )

    resolved_model_name, model_source = _resolve_hf_snapshot(
        model_name,
        allow_online=True,
        verbose=True,
    )
    
    print(f"🦥 Loading {model_name} with Unsloth optimizations...")
    print(f"   HF snapshot source: {model_source}")
    
    if FAST_SENTENCE_TRANSFORMER_AVAILABLE:
        # Preferred path: Use FastSentenceTransformer (optimized for embeddings)
        print("   Using FastSentenceTransformer API (specialized for embeddings)")
        
        model = FastSentenceTransformer.from_pretrained(
            model_name=resolved_model_name,
            max_seq_length=max_seq_length,
            full_finetuning=full_finetuning,
            # Note: FastSentenceTransformer handles dtype automatically
        )
        
        print(f"✅ Loaded {model_name} with FastSentenceTransformer")
        # FastSentenceTransformer returns just the model (it's already a SentenceTransformer)
        return model, None  # tokenizer is internal to the model
    else:
        # Fallback: Use FastModel (generic LLM loader)
        print("   Using FastModel API (generic, falling back from FastSentenceTransformer)")
        
        # Get the appropriate model class
        model_class = get_model_class(model_name)
        print(f"   Using model class: {model_class.__name__}")
        
        # Load with Unsloth's FastModel
        model, tokenizer = FastModel.from_pretrained(
            model_name=resolved_model_name,
            auto_model=model_class,
            max_seq_length=max_seq_length,
            dtype=dtype,  # Auto-detects BF16/FP16
            load_in_4bit=load_in_4bit,
        )
        
        print(f"✅ Loaded {model_name} with FastModel")
        return model, tokenizer


def apply_qlora_adapters(
    model: Any,
    lora_rank: int = 32,  # Tutorial default: 32
    lora_alpha: float = 64.0,  # Tutorial default: 64 (2x rank)
    lora_dropout: float = 0.0,  # Unsloth recommends 0.0
    target_modules: Optional[List[str]] = None,
    model_name: Optional[str] = None,  # For auto-detection of target modules
    exclude_modules: Optional[List[str]] = None,
    use_rslora: bool = False,
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 3407,  # Tutorial default seed
) -> Any:
    """
    Apply LoRA/QLoRA adapters to the model for memory-efficient fine-tuning.
    
    Uses FastSentenceTransformer.get_peft_model when available (preferred),
    otherwise falls back to FastModel.get_peft_model.
    
    Target modules are auto-detected based on model architecture if not specified:
    - BERT/RoBERTa: query, key, value, dense
    - ModernBERT: Wqkv, Wo, Wi
    - LLaMA/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    
    Args:
        model: The Unsloth-loaded model (FastSentenceTransformer or FastModel)
        lora_rank: Rank of LoRA matrices (higher = more expressive but slower)
        lora_alpha: LoRA scaling factor (rule of thumb: 2 * rank)
        lora_dropout: Dropout for LoRA layers (0.0 is recommended by Unsloth)
        target_modules: List of modules to apply LoRA to (None for auto-detection)
        model_name: Model name for auto-detection (required if target_modules is None)
        exclude_modules: List of modules to exclude from LoRA
        use_rslora: Whether to use Rank-Stabilized LoRA
        use_gradient_checkpointing: Gradient checkpointing mode ("unsloth" for optimized)
        random_state: Random seed for LoRA initialization (default: 3407 as in tutorial)
        
    Returns:
        Model with LoRA adapters attached
    """
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth is not available for LoRA patching.")
    
    # Auto-detect target modules if not provided
    if target_modules is None:
        if model_name is not None:
            # Use the new auto-detection function
            target_modules = get_target_modules_for_model(
                model_name=model_name,
                user_specified_modules=None,
                auto_detect=True,
                verbose=True
            )
        else:
            # Fallback to BERT-style defaults (most common for embedding models)
            print("[WARNING] No model_name provided for auto-detection, using BERT-style defaults")
            target_modules = DEFAULT_TARGET_MODULES
    
    if exclude_modules is None:
        exclude_modules = []
    
    print(f"🎯 Attaching LoRA adapters (rank={lora_rank}, alpha={lora_alpha})...")
    print(f"   Target modules: {target_modules}")
    print(f"   Random state: {random_state}")
    
    if FAST_SENTENCE_TRANSFORMER_AVAILABLE:
        # Preferred: Use FastSentenceTransformer.get_peft_model
        print("   Using FastSentenceTransformer.get_peft_model API")
        
        model = FastSentenceTransformer.get_peft_model(
            model,
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",  # Optimized setting
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=None,
            task_type="FEATURE_EXTRACTION",  # Important for embeddings!
        )
    else:
        # Fallback: Use FastModel.get_peft_model
        print("   Using FastModel.get_peft_model API (fallback)")
        
        model = FastModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            use_rslora=use_rslora,
            bias="none",
            use_gradient_checkpointing=use_gradient_checkpointing,
            modules_to_save=None,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    
    # Print trainable parameters
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    
    print("✅ LoRA adapters attached successfully")
    return model


def wrap_unsloth_in_sentence_transformer(
    model: Any,
    tokenizer: Any,
    base_model_id: str,
    max_seq_length: int = 512,
    pooling_mode: str = "mean",
    normalize_embeddings: bool = True,
) -> SentenceTransformer:
    """
    Wrap an Unsloth-patched model in a SentenceTransformer for training.
    
    This creates a SentenceTransformer that uses the Unsloth-optimized model
    internally, enabling the use of SentenceTransformer's training utilities
    with Unsloth's speed optimizations.
    
    Args:
        model: The Unsloth-patched model (with or without QLoRA)
        tokenizer: The tokenizer from Unsloth
        base_model_id: Original model name (for config reference)
        max_seq_length: Maximum sequence length
        pooling_mode: Pooling strategy ("mean", "cls", "max")
        normalize_embeddings: Whether to normalize output embeddings
        
    Returns:
        SentenceTransformer instance wrapping the Unsloth model
    """
    print("🔧 Wrapping Unsloth model in SentenceTransformer...")
    resolved_base_model_id, model_source = _resolve_hf_snapshot(
        base_model_id,
        allow_online=True,
        verbose=False,
    )
    print(f"   Transformer snapshot source: {model_source}")
    
    # 1. Create the Transformer module instance
    transformer_module = sentence_transformers.models.Transformer(
        model_name_or_path=resolved_base_model_id,
        max_seq_length=max_seq_length,
    )
    
    # 2. Replace the internal HuggingFace model with our Unsloth-patched model
    transformer_module.auto_model = model
    transformer_module.tokenizer = tokenizer
    # Re-apply max_seq_length to the replaced tokenizer.
    # The Unsloth tokenizer carries the model's native model_max_length (e.g. 8192 for
    # ModernBERT/ModernColBERT).  Newer sentence_transformers versions read max_seq_length
    # from tokenizer.model_max_length, so without this correction the tokenizer would silently
    # expand truncation back to the native limit and cause OOM.
    if hasattr(transformer_module.tokenizer, 'model_max_length'):
        transformer_module.tokenizer.model_max_length = max_seq_length
    print(f"   Patched Transformer module with Unsloth model")
    
    # 3. Create the Pooling module
    hidden_size = model.config.hidden_size
    pooling_module = sentence_transformers.models.Pooling(
        embedding_dimension=hidden_size,
        pooling_mode=pooling_mode,
    )
    print(f"   Created Pooling module (mode={pooling_mode}, dim={hidden_size})")
    
    # 4. Add Normalize module if requested
    modules = [transformer_module, pooling_module]
    if normalize_embeddings:
        normalize_module = sentence_transformers.models.Normalize()
        modules.append(normalize_module)
        print("   Added normalization layer")
    
    # 5. Initialize SentenceTransformer with custom modules
    sbert_model = SentenceTransformer(modules=modules)
    
    print("✅ SentenceTransformer wrapper created successfully")
    return sbert_model


def create_unsloth_sentence_encoder(
    model_name: str,
    device: str = "cuda",
    max_seq_length: int = 512,
    # Unsloth configuration
    use_unsloth: bool = True,
    load_in_4bit: bool = True,
    dtype: Optional[torch.dtype] = None,
    full_finetuning: bool = False,
    # LoRA configuration
    use_qlora: bool = False,
    lora_rank: int = 32,  # Tutorial default
    lora_alpha: float = 64.0,  # Tutorial default (2x rank)
    lora_dropout: float = 0.0,  # Unsloth recommends 0.0
    target_modules: Optional[List[str]] = None,  # None or "auto" for auto-detection
    # Pooling configuration (only used for FastModel fallback)
    pooling_mode: str = "mean",
    normalize_embeddings: bool = True,
) -> SentenceTransformer:
    """
    Create a SentenceTransformer with Unsloth optimizations for faster training.
    
    This is the main entry point for creating an Unsloth-accelerated encoder.
    
    When FastSentenceTransformer is available (preferred):
    - Uses Unsloth's specialized embedding model API
    - Returns a SentenceTransformer directly (no manual wrapping needed)
    - Supports full_finetuning parameter
    
    When only FastModel is available (fallback):
    - Uses generic LLM loader
    - Manually wraps in SentenceTransformer
    
    Target modules for LoRA are automatically detected based on the model architecture:
    - BERT/RoBERTa models: query, key, value, dense
    - ModernBERT models: Wqkv, Wo, Wi
    - LLaMA/Mistral/Qwen models: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    
    Args:
        model_name: HuggingFace model name/path
        device: Device to load model on ("cuda" or "cpu")
        max_seq_length: Maximum sequence length
        use_unsloth: Whether to use Unsloth optimizations (if available)
        load_in_4bit: Whether to use 4-bit quantization (FastModel fallback only)
        dtype: Model dtype (None for auto-detect)
        full_finetuning: Whether to enable full finetuning (vs LoRA only)
        use_qlora: Whether to apply LoRA adapters
        lora_rank: Rank of LoRA matrices (default: 32 as in tutorial)
        lora_alpha: LoRA scaling factor (default: 64, 2x rank as in tutorial)
        lora_dropout: Dropout for LoRA layers (0.0 recommended)
        target_modules: Modules to apply LoRA to (None or "auto" for auto-detection)
        pooling_mode: Pooling strategy ("mean", "cls", "max") - FastModel fallback only
        normalize_embeddings: Whether to normalize output embeddings - FastModel fallback only
        
    Returns:
        SentenceTransformer instance with Unsloth optimizations
        
    Example:
        >>> encoder = create_unsloth_sentence_encoder(
        ...     model_name="abhinand/MedEmbed-large-v0.1",  # BERT-based
        ...     use_qlora=True,
        ...     lora_rank=32
        ... )
        >>> # Target modules auto-detected: ['query', 'key', 'value', 'dense']
        >>> embeddings = encoder.encode(["Hello world"])
    """
    # Check if Unsloth is available and requested
    if use_unsloth and not UNSLOTH_AVAILABLE:
        print("⚠️ Unsloth requested but not available. Falling back to standard loading.")
        use_unsloth = False
    
    # Auto-detect target modules if not explicitly specified
    if use_qlora and not full_finetuning:
        resolved_target_modules = get_target_modules_for_model(
            model_name=model_name,
            user_specified_modules=target_modules,
            auto_detect=True,
            verbose=True
        )
    else:
        resolved_target_modules = target_modules

    resolved_model_name, model_source = _resolve_hf_snapshot(
        model_name,
        allow_online=True,
        verbose=True,
    )
    
    if use_unsloth:
        print(f"\n{'='*60}")
        print("🦥 UNSLOTH MODE: Creating optimized sentence encoder")
        print(f"{'='*60}")
        
        # Force FastModel fallback for ColBERT models to ensure an explicit Pooling layer is added
        if FAST_SENTENCE_TRANSFORMER_AVAILABLE and "colbert" not in model_name.lower():
            # ============================================================
            # PREFERRED PATH: Use FastSentenceTransformer
            # This is optimized specifically for embedding models
            # ============================================================
            print("   API: FastSentenceTransformer (specialized for embeddings)")
            print(f"   Model: {model_name}")
            print(f"   Max seq length: {max_seq_length}")
            print(f"   Full finetuning: {full_finetuning}")
            print(f"   HF snapshot source: {model_source}")
            
            # Step 1: Load model with FastSentenceTransformer
            sentence_encoder = FastSentenceTransformer.from_pretrained(
                model_name=resolved_model_name,
                max_seq_length=max_seq_length,
                full_finetuning=full_finetuning,
            )
            
            # Step 2: Apply LoRA adapters if requested
            if use_qlora and not full_finetuning:
                print(f"   LoRA target modules (auto-detected): {resolved_target_modules}")
                sentence_encoder = FastSentenceTransformer.get_peft_model(
                    sentence_encoder,
                    r=lora_rank,
                    target_modules=resolved_target_modules,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,  # Tutorial default
                    use_rslora=False,
                    loftq_config=None,
                    task_type="FEATURE_EXTRACTION",
                )
                print(f"   LoRA: rank={lora_rank}, alpha={lora_alpha}")
                
                # =====================================================================
                # DIAGNOSTIC: Verify PEFT actually froze base weights
                # =====================================================================
                total_params = sum(p.numel() for p in sentence_encoder.parameters())
                trainable_params = sum(p.numel() for p in sentence_encoder.parameters() if p.requires_grad)
                frozen_params = total_params - trainable_params
                
                # Count LoRA-specific params
                lora_params = sum(p.numel() for name, p in sentence_encoder.named_parameters() 
                                 if 'lora' in name.lower())
                
                print(f"\n   📊 PEFT Status After get_peft_model():")
                print(f"      Total params: {total_params:,}")
                print(f"      Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
                print(f"      Frozen: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
                print(f"      LoRA adapter params: {lora_params:,}")
                
                # Check if PEFT is working correctly
                if trainable_params == total_params:
                    print(f"\n   ⚠️ WARNING: PEFT did NOT freeze base weights!")
                    print(f"      All {total_params:,} parameters are trainable.")
                    print(f"      This means you're doing FULL fine-tuning, NOT QLoRA.")
                    
                    # Try to manually freeze base weights
                    print(f"\n   🔧 Attempting to manually freeze base weights...")
                    for name, param in sentence_encoder.named_parameters():
                        # Keep LoRA params trainable, freeze everything else
                        if 'lora' not in name.lower():
                            param.requires_grad = False
                    
                    # Re-check after manual freeze
                    trainable_after = sum(p.numel() for p in sentence_encoder.parameters() if p.requires_grad)
                    frozen_after = total_params - trainable_after
                    print(f"      After manual freeze:")
                    print(f"      Trainable: {trainable_after:,} ({trainable_after/total_params*100:.2f}%)")
                    print(f"      Frozen: {frozen_after:,} ({frozen_after/total_params*100:.2f}%)")
                    
                    if trainable_after == total_params:
                        print(f"\n   ❌ CRITICAL: Manual freeze failed!")
                        print(f"      No LoRA params found. Check if target_modules are correct.")
                        print(f"      Available param names (first 10):")
                        for i, (name, _) in enumerate(sentence_encoder.named_parameters()):
                            if i < 10:
                                print(f"         {name}")
                    else:
                        print(f"   ✅ Manual freeze succeeded! QLoRA is now active.")
                else:
                    pct = trainable_params / total_params * 100
                    print(f"\n   ✅ PEFT is working! Only {pct:.2f}% params trainable (QLoRA active)")
                
                # Try to use PEFT's built-in method if available
                if hasattr(sentence_encoder, 'print_trainable_parameters'):
                    print("\n   📋 PEFT print_trainable_parameters():")
                    sentence_encoder.print_trainable_parameters()
                
                # Check if the underlying model has PEFT
                underlying_model = None
                if hasattr(sentence_encoder, '_modules'):
                    for module_name, module in sentence_encoder._modules.items():
                        if hasattr(module, 'print_trainable_parameters'):
                            print(f"\n   📋 PEFT on {module_name}.print_trainable_parameters():")
                            module.print_trainable_parameters()
                            underlying_model = module
                            break
                        # Also check for nested model attribute
                        if hasattr(module, 'model') and hasattr(module.model, 'print_trainable_parameters'):
                            print(f"\n   📋 PEFT on {module_name}.model.print_trainable_parameters():")
                            module.model.print_trainable_parameters()
                            underlying_model = module.model
                            break
                    
            elif use_qlora and full_finetuning:
                print("   ⚠️ Ignoring use_qlora since full_finetuning=True")
            
            print(f"\n✅ FastSentenceTransformer encoder ready!")
            print(f"{'='*60}\n")
            
        else:
            # ============================================================
            # FALLBACK PATH: Use FastModel (for older Unsloth versions)
            # ============================================================
            print("   API: FastModel (generic LLM loader, fallback)")
            
            # Step 1: Load model with FastModel
            model, tokenizer = create_unsloth_model(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                dtype=dtype,
                device=device,
                full_finetuning=full_finetuning,
            )
            
            # Step 2: Optionally apply LoRA adapters
            if use_qlora and not full_finetuning:
                print(f"   LoRA target modules (auto-detected): {resolved_target_modules}")
                model = apply_qlora_adapters(
                    model=model,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=resolved_target_modules,
                    model_name=model_name,
                )
            elif use_qlora and full_finetuning:
                print("   ⚠️ Ignoring use_qlora since full_finetuning=True")
            
            # Step 3: Wrap in SentenceTransformer
            sentence_encoder = wrap_unsloth_in_sentence_transformer(
                model=model,
                tokenizer=tokenizer,
                base_model_id=model_name,
                max_seq_length=max_seq_length,
                pooling_mode=pooling_mode,
                normalize_embeddings=normalize_embeddings,
            )
            
            print(f"\nFastModel encoder ready (wrapped in SentenceTransformer)!")
            print(f"{'='*60}\n")
        
    else:
        # Fallback to standard SentenceTransformer loading
        print(f"\n📦 Loading {model_name} with standard SentenceTransformer...")
        
        model_kwargs = {}
        if device == "cuda" and torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        
        sentence_encoder = SentenceTransformer(
            resolved_model_name,
            trust_remote_code=True,
            device=device,
            model_kwargs=model_kwargs,
        )
        
        print(f"✅ Standard encoder loaded")
    
    return sentence_encoder


def get_unsloth_status() -> Dict[str, Any]:
    """
    Get current Unsloth status and capabilities.
    
    Returns:
        Dictionary with Unsloth status information
    """
    status = {
        "available": UNSLOTH_AVAILABLE,
        "fast_sentence_transformer": FAST_SENTENCE_TRANSFORMER_AVAILABLE,
        "version": None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_memory": None,
    }
    
    if UNSLOTH_AVAILABLE:
        try:
            status["version"] = unsloth.__version__
        except AttributeError:
            status["version"] = "unknown"
    
    if torch.cuda.is_available():
        status["gpu_name"] = torch.cuda.get_device_name(0)
        status["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    
    return status


def print_unsloth_status():
    """Print Unsloth status information."""
    status = get_unsloth_status()
    
    print("\n🦥 Unsloth Status:")
    print(f"   Available: {'✅ Yes' if status['available'] else '❌ No'}")
    if status['available']:
        print(f"   FastSentenceTransformer: {'✅ Yes (preferred)' if status['fast_sentence_transformer'] else '❌ No (using FastModel fallback)'}")
    if status['version']:
        print(f"   Version: {status['version']}")
    print(f"   CUDA: {'✅ Yes' if status['cuda_available'] else '❌ No'}")
    if status['gpu_name']:
        print(f"   GPU: {status['gpu_name']}")
        print(f"   GPU Memory: {status['gpu_memory']}")
    print()


# ============================================================================
# TORCH.COMPILE OPTIMIZATION FOR CUSTOM MODULES
# ============================================================================
# Unsloth can only optimize HuggingFace transformers. For custom modules like
# BidirectionalCrossAttention, CrossAttentionModule, and LoRALinear, we use
# torch.compile() (PyTorch 2.0+) for JIT compilation speedups.

# Check torch.compile availability
TORCH_COMPILE_AVAILABLE = False
try:
    import torch
    if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
        TORCH_COMPILE_AVAILABLE = True
except Exception:
    pass


def compile_custom_modules(
    model: torch.nn.Module,
    mode: str = "reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
    fullgraph: bool = False,
    dynamic: bool = True,
    backend: str = "inductor",
) -> torch.nn.Module:
    """
    Apply torch.compile() to custom attention modules for faster execution.
    
    This optimizes:
    - BidirectionalCrossAttention
    - CrossAttentionModule
    - LoRALinear
    - SelfAttentionBlock
    - Any other custom nn.Module
    
    Args:
        model: The model containing custom modules to compile
        mode: Compilation mode:
            - "default": Good balance of compile time and speedup
            - "reduce-overhead": Faster compile, good for small batches
            - "max-autotune": Slowest compile, best runtime performance
        fullgraph: Whether to compile as a single graph (more restrictions)
        dynamic: Whether to allow dynamic shapes
        backend: Compilation backend ("inductor" recommended for CUDA)
        
    Returns:
        Model with compiled modules (in-place modification)
        
    Note:
        torch.compile() is a PyTorch 2.0+ feature. Falls back gracefully
        on older PyTorch versions.
    """
    if not TORCH_COMPILE_AVAILABLE:
        print("⚠️ torch.compile() not available (requires PyTorch 2.0+)")
        return model
    
    print(f"\n🔧 Applying torch.compile() optimizations (mode={mode})...")
    
    compiled_count = 0
    
    # Target module types for compilation
    target_module_names = [
        "BidirectionalCrossAttention",
        "CrossAttentionModule", 
        "LoRALinear",
        "SelfAttentionBlock",
        "LatentCrossAttention",
        "LatentBottleneck",
        "FeedForward",
    ]
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if module_type in target_module_names:
            try:
                # Compile the module's forward method
                compiled_forward = torch.compile(
                    module.forward,
                    mode=mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                    backend=backend,
                )
                module.forward = compiled_forward
                compiled_count += 1
            except Exception as e:
                print(f"   ⚠️ Could not compile {name} ({module_type}): {e}")
    
    if compiled_count > 0:
        print(f"✅ Compiled {compiled_count} custom modules with torch.compile()")
    else:
        print("   No custom modules found to compile")
        
    return model


def optimize_model_for_inference(
    model: torch.nn.Module,
    use_compile: bool = True,
    compile_mode: str = "reduce-overhead",
) -> torch.nn.Module:
    """
    Optimize a model for fast inference (evaluation).
    
    Applies:
    1. torch.compile() for JIT compilation
    2. Sets model to eval mode
    3. Disables gradient computation hints
    
    Args:
        model: The model to optimize
        use_compile: Whether to use torch.compile()
        compile_mode: Compilation mode for torch.compile()
        
    Returns:
        Optimized model
    """
    model.eval()
    
    if use_compile:
        model = compile_custom_modules(model, mode=compile_mode)
    
    # Disable gradient checkpointing if present (not needed for inference)
    if hasattr(model, 'gradient_checkpointing_disable'):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    
    return model


def optimize_model_for_training(
    model: torch.nn.Module,
    use_compile: bool = True,
    compile_mode: str = "default",  # More conservative for training
    use_gradient_checkpointing: bool = False,
) -> torch.nn.Module:
    """
    Optimize a model for fast training.
    
    Applies:
    1. torch.compile() for JIT compilation (with training-safe settings)
    2. Optional gradient checkpointing for memory efficiency
    
    Args:
        model: The model to optimize
        use_compile: Whether to use torch.compile()
        compile_mode: Compilation mode for torch.compile()
        use_gradient_checkpointing: Whether to enable gradient checkpointing
        
    Returns:
        Optimized model
    """
    model.train()
    
    if use_compile:
        model = compile_custom_modules(model, mode=compile_mode, dynamic=True)
    
    # Enable gradient checkpointing if requested (for memory efficiency)
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        except Exception as e:
            print(f"⚠️ Could not enable gradient checkpointing: {e}")
    
    return model


def get_optimization_status() -> Dict[str, Any]:
    """
    Get current optimization capabilities status.
    
    Returns:
        Dictionary with optimization status information
    """
    status = {
        "unsloth_available": UNSLOTH_AVAILABLE,
        "torch_compile_available": TORCH_COMPILE_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "gpu_name": None,
        "gpu_memory": None,
    }
    
    if torch.cuda.is_available():
        status["gpu_name"] = torch.cuda.get_device_name(0)
        status["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    
    return status


def print_optimization_status():
    """Print optimization capabilities status."""
    status = get_optimization_status()
    
    print("\n⚡ Optimization Status:")
    print(f"   Unsloth (Encoder): {'✅ Yes' if status['unsloth_available'] else '❌ No'}")
    print(f"   torch.compile() (Custom Modules): {'✅ Yes' if status['torch_compile_available'] else '❌ No'}")
    print(f"   CUDA: {'✅ Yes' if status['cuda_available'] else '❌ No'}")
    print(f"   PyTorch: {status['pytorch_version']}")
    if status['gpu_name']:
        print(f"   GPU: {status['gpu_name']} ({status['gpu_memory']})")
    print()


# Export public API
__all__ = [
    # Unsloth functions
    "UNSLOTH_AVAILABLE",
    "FAST_SENTENCE_TRANSFORMER_AVAILABLE",  # NEW: Flag for preferred API
    "create_unsloth_sentence_encoder",
    "create_unsloth_model",
    "apply_qlora_adapters",
    "wrap_unsloth_in_sentence_transformer",
    "get_unsloth_status",
    "print_unsloth_status",
    "get_model_max_seq_length",  # Auto-detect max_seq_length from model config
    # Target module auto-detection
    "ARCHITECTURE_TARGET_MODULES",
    "DEFAULT_TARGET_MODULES",
    "detect_target_modules_for_model",
    "get_target_modules_for_model",
    # torch.compile functions
    "TORCH_COMPILE_AVAILABLE",
    "compile_custom_modules",
    "optimize_model_for_inference",
    "optimize_model_for_training",
    "get_optimization_status",
    "print_optimization_status",
]


if __name__ == "__main__":
    # Test the module
    print_unsloth_status()
    print_optimization_status()
    
    if UNSLOTH_AVAILABLE:
        print("Testing Unsloth encoder creation...")
        try:
            # Use tutorial-recommended model and settings
            encoder = create_unsloth_sentence_encoder(
                model_name="unsloth/embeddinggemma-300m",  # Tutorial model
                use_unsloth=True,
                use_qlora=True,
                lora_rank=32,  # Tutorial default
                lora_alpha=64.0,  # Tutorial default (2x rank)
            )
            
            # Test encoding
            test_sentences = ["This is a test sentence.", "Another test."]
            embeddings = encoder.encode(test_sentences)
            print(f"✅ Test embeddings shape: {embeddings.shape}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping Unsloth test (not available)")
