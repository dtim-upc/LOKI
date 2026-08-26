import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import contextlib
import sys
import io
import pandas as pd
import warnings
import math
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
from models import TableTextEmbeddingModel, BidirectionalTableTextModel
from data import load_row_level_dataset, IdBasedEmbeddingCache, _extract_rows_robust, _extract_sentences_robust
from encoding import build_id_based_embedding_cache
from utils import save_plot_multi_format
from hf_model_resolver import bootstrap_hf_model_snapshots, ensure_repo_local_hf_snapshot

# Set up seaborn for better visualizations
plt.style.use('default')
sns.set_style("whitegrid")
warnings.filterwarnings("ignore")

def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a PyTorch tensor to numpy, handling BFloat16 conversion.
    
    Args:
        tensor: PyTorch tensor to convert
        
    Returns:
        NumPy array
    """
    # Convert BFloat16 to Float32 before converting to numpy (BFloat16 not supported by numpy)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.cpu().numpy()

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def _bootstrap_visualization_hf_assets(model_name: str) -> None:
    bootstrap_records = bootstrap_hf_model_snapshots([model_name], allow_online=True)
    if not bootstrap_records:
        return

    print("\n=== Phase 0: Hugging Face Asset Bootstrap ===")
    for record in bootstrap_records:
        print(
            f"   - {record['model_name']} -> {record['resolved_path']} "
            f"({record['source']})"
        )


def _load_sentence_transformer_encoder(
    base_model_name: str,
    *,
    device: Optional[Union[str, torch.device]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    suppress_output: bool = False,
) -> SentenceTransformer:
    resolved_model_name, model_source = ensure_repo_local_hf_snapshot(
        base_model_name,
        allow_online=True,
    )
    print(
        f"   HF snapshot ready for '{base_model_name}': {resolved_model_name} "
        f"({model_source})"
    )

    sentence_transformer_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    if device is not None:
        sentence_transformer_kwargs["device"] = device
    if model_kwargs is not None:
        sentence_transformer_kwargs["model_kwargs"] = model_kwargs
    if tokenizer_kwargs is not None:
        sentence_transformer_kwargs["tokenizer_kwargs"] = tokenizer_kwargs

    if suppress_output:
        with suppress_stdout():
            return SentenceTransformer(resolved_model_name, **sentence_transformer_kwargs)

    return SentenceTransformer(resolved_model_name, **sentence_transformer_kwargs)

def create_dynamic_sentence_encoder(embedding_dim: int, device="cuda"):
    """
    Create a SentenceTransformer-compatible encoder with the specified embedding dimension.
    This is completely dynamic and doesn't depend on any hardcoded models.
    """
    print(f"Creating dynamic encoder for {embedding_dim} dimensions on {device}")
    
    class DynamicSentenceEncoder(torch.nn.Module):
        def __init__(self, embedding_dim, device="cuda"):
            super().__init__()
            self.embedding_dim = embedding_dim
            self._device = device
            # Add a dummy parameter so the model has parameters() method and proper device/dtype handling
            self.dummy_param = torch.nn.Parameter(torch.randn(1, device=device))
            
        def encode(self, sentences, batch_size=32, convert_to_tensor=True, normalize_embeddings=True, **kwargs):
            """
            Generate placeholder embeddings with the correct dimensions.
            Note: These will be replaced by the actual trained cross-attention weights.
            """
            import torch
            if isinstance(sentences, str):
                sentences = [sentences]
            
            embeddings = torch.randn(
                len(sentences), 
                self.embedding_dim, 
                dtype=self.dummy_param.dtype, 
                device=self.dummy_param.device
            )
            
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings
            
        def get_sentence_embedding_dimension(self):
            return self.embedding_dim
            
        def to(self, device):
            """Ensure proper device handling"""
            self._device = device
            super().to(device)
            return self
            
        def eval(self):
            """Ensure proper eval mode"""
            super().eval()
            return self
            
        @property
        def device(self):
            return self._device
    
    return DynamicSentenceEncoder(embedding_dim, device)

def load_model(model_path: Optional[str] = None, base_model_name: str = "answerdotai/ModernBERT-base", device: Optional[str] = None) -> Union[TableTextEmbeddingModel, BidirectionalTableTextModel]:
    """
    Load either a trained model from checkpoint or create a fresh untrained model.
    Uses completely dynamic dimension detection from checkpoint.
    
    Args:
        model_path: Path to a trained model checkpoint (.pt file). If None, creates an untrained model.
        base_model_name: Name of the base sentence transformer model to use (only used if no checkpoint)
        device: Device to load the model on (cuda/cpu). If None, auto-detects.
        
    Returns:
        Loaded TableTextEmbeddingModel (trained or untrained)
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sentence_encoder = None
    embedding_dim = None
    is_bidirectional = False  # Initialize outside the if block
    # Initialize attention parameters with defaults
    attention_type = "standard"
    sparse_top_k = 3
    window_size = 5
    threshold_base = 0.1
    # Gated attention overlay defaults
    use_gated_attention = False
    gated_attention_mode = "scalar"
    gated_attention_hidden_dim = 0
    gated_attention_dropout = 0.0
    gated_attention_init_bias = 2.0
    disable_temperature = False
    use_header_conditioning = False
    
    # If we have a model path, read configuration files first (much more reliable!)
    if model_path and os.path.exists(model_path):
        print(f"📁 Loading TRAINED model from checkpoint: {model_path}")
        
        # Check if this looks like a "best" model path to confirm we're using the best model
        if "_best.pt" in model_path or "best_model" in model_path:
            print(f"✅ CONFIRMED: Loading BEST model checkpoint (highest validation accuracy)")
        
        # Step 1: Try to read from configuration files (preferred approach)
        model_dir = os.path.dirname(model_path)
        config_files = ['args.json', 'training_config.json', 'config.json']
        config_data = None
        
        for config_file in config_files:
            config_path = os.path.join(model_dir, config_file)
            if os.path.exists(config_path):
                print(f"📄 Found configuration file: {config_file}")
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    break
                except Exception as e:
                    print(f"⚠️ Failed to read {config_file}: {e}")
                    continue
        
        if config_data:
            # Extract model configuration from saved config
            is_bidirectional = config_data.get('use_bidirectional', False)
            embedding_dim = config_data.get('embedding_dim')
            
            # Extract attention configuration
            attention_type = config_data.get('attention_type', 'standard')
            sparse_top_k = config_data.get('sparse_top_k', 3)
            window_size = config_data.get('window_size', 5)
            threshold_base = config_data.get('threshold_base', 0.1)
            # Extract gated attention configuration (if present)
            use_gated_attention = config_data.get('use_gated_attention', False)
            gated_attention_mode = config_data.get('gated_attention_mode', 'scalar')
            gated_attention_hidden_dim = config_data.get('gated_attention_hidden_dim', 0)
            gated_attention_dropout = config_data.get('gated_attention_dropout', 0.0)
            gated_attention_init_bias = config_data.get('gated_attention_init_bias', 2.0)
            disable_temperature = config_data.get('disable_temperature', False)
            use_header_conditioning = config_data.get('use_header_conditioning', False)
            
            # Also check for architecture field as backup
            if not is_bidirectional:
                architecture = config_data.get('architecture', '')
                is_bidirectional = (architecture == 'bidirectional')
            
            print(f"✅ Loaded from config: model_type={'Bidirectional' if is_bidirectional else 'Unidirectional'}, embedding_dim={embedding_dim}, attention_type={attention_type}")
            
            if embedding_dim:
                # Create a dynamic sentence encoder that matches the detected dimensions
                sentence_encoder = create_dynamic_sentence_encoder(embedding_dim, device)
            else:
                print("⚠️ Config found but embedding_dim not specified, falling back to checkpoint analysis...")
                config_data = None  # Force fallback
        
        # Step 2: Fallback to checkpoint analysis if config unavailable or incomplete
        if config_data is None or embedding_dim is None:
            print("🔍 Analyzing checkpoint weights for architecture detection...")
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                
                # Detect model type from checkpoint structure
                for key in state_dict.keys():
                    if any(bidirectional_key in key for bidirectional_key in [
                        'bidirectional_attention.', 'cross_attention.forward_W_', 
                        'cross_attention.reverse_W_', 'cross_attention.row_attention_norm', 
                        'cross_attention.sentence_attention_norm'
                    ]):
                        is_bidirectional = True
                        break
                
                print(f"🔍 Detected model type: {'Bidirectional' if is_bidirectional else 'Unidirectional'}")
                
                # Dynamic embedding dimension detection from sentence encoder
                detected_embedding_dim = None
                for key in state_dict.keys():
                    if 'embed_tokens.weight' in key:
                        detected_embedding_dim = state_dict[key].shape[1]  # [vocab_size, embedding_dim]
                        print(f"🎯 Detected embedding_dim from sentence encoder: {detected_embedding_dim}")
                        break
                
                # Fallback to cross-attention layers if sentence encoder not found
                if detected_embedding_dim is None:
                    # Look for consistent dimensions across related layers
                    dimension_candidates = {}
                    
                    for key in state_dict.keys():
                        if ('cross_attention.' in key or 'bidirectional_attention.' in key) and 'weight' in key:
                            shape = state_dict[key].shape
                            if len(shape) == 2:  # Linear layer
                                # Count occurrences of each dimension
                                for dim in shape:
                                    dimension_candidates[dim] = dimension_candidates.get(dim, 0) + 1
                    
                    # Find the most common dimension (likely to be embedding_dim)
                    if dimension_candidates:
                        # Get the dimension that appears most frequently
                        most_common_dim = max(dimension_candidates.items(), key=lambda x: x[1])
                        detected_embedding_dim = most_common_dim[0]
                        print(f"🎯 Inferred embedding_dim from frequency analysis: {detected_embedding_dim} (appears {most_common_dim[1]} times)")
                    
                    # Additional fallback: look for layer norm weights (they have embedding_dim size)
                    if detected_embedding_dim is None:
                        for key in state_dict.keys():
                            if 'norm' in key and 'weight' in key and len(state_dict[key].shape) == 1:
                                detected_embedding_dim = state_dict[key].shape[0]
                                print(f"🎯 Inferred embedding_dim from LayerNorm: {detected_embedding_dim}")
                                break
                
                if detected_embedding_dim is None:
                    print("🔍 Available keys in checkpoint (first 10):")
                    for key in list(state_dict.keys())[:10]:
                        shape_info = state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])
                        print(f"  - {key}: {shape_info}")
                    if len(state_dict.keys()) > 10:
                        print(f"  ... and {len(state_dict.keys()) - 10} more keys")
                    raise ValueError("❌ Could not detect embedding dimension from checkpoint weights")
                
                embedding_dim = detected_embedding_dim
                print(f"✅ Detected model architecture: embedding_dim={embedding_dim}")
                
                # Create a dynamic sentence encoder that matches the detected dimensions
                sentence_encoder = create_dynamic_sentence_encoder(embedding_dim, device)
                
            except Exception as e:
                print(f"❌ Error analyzing checkpoint: {e}")
                raise e
    
    # If no checkpoint or detection failed, use the provided base model
    if sentence_encoder is None:
        print(f"Loading sentence encoder: {base_model_name}")
        
        # For trained model loading, try to preserve Flash Attention if possible
        # For untrained models, we'll avoid Flash Attention in the comparison functions
        try:
            # Try with Flash Attention first if on CUDA (to match training setup)
            if device == "cuda":
                model_kwargs = {
                    "attn_implementation": "flash_attention_2", 
                    "device_map": "auto", 
                    "dtype": torch.bfloat16
                }
                try:
                    sentence_encoder = _load_sentence_transformer_encoder(
                        base_model_name,
                        device=device,
                        model_kwargs=model_kwargs,
                        suppress_output=True,
                    )
                    print(f"✅ Loaded base model with Flash Attention (matching training setup)")
                except Exception as flash_e:
                    print(f"⚠️  Flash Attention failed: {flash_e}")
                    # Fallback without Flash Attention but with bfloat16
                    sentence_encoder = _load_sentence_transformer_encoder(
                        base_model_name,
                        device=device,
                        model_kwargs={"dtype": torch.bfloat16},
                        suppress_output=True,
                    )
                    print(f"✅ Loaded base model without Flash Attention (with bfloat16)")
            else:
                # On CPU, avoid Flash Attention but still use bfloat16
                sentence_encoder = _load_sentence_transformer_encoder(
                    base_model_name,
                    device=device,
                    model_kwargs={"dtype": torch.bfloat16},
                    suppress_output=True,
                )
                print(f"✅ Loaded base model without Flash Attention (CPU mode, with bfloat16)")
                
        except Exception as e:
            print(f"❌ Failed to load base model: {e}")
            # Final fallback to dynamic encoder
            print(f"Using dynamic fallback encoder for dimension 768")
            sentence_encoder = create_dynamic_sentence_encoder(768, device)
            
        embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
    
    print(f"📐 Final embedding dimension: {embedding_dim}")
    
    # Create the model with detected/specified dimensions
    # Check if we detected a bidirectional model from the checkpoint
    if is_bidirectional:
        print("Creating BidirectionalTableTextModel...")
        from models import BidirectionalTableTextModel
        print(f"🚀 Using {attention_type} attention mechanism")
        if attention_type == "top_k_sparse":
            print(f"📋 Using top-{sparse_top_k} sparse attention")
        elif attention_type == "windowed":
            print(f"📋 Using windowed attention with window_size={window_size}")  
        elif attention_type == "threshold":
            print(f"📋 Using threshold attention with base_threshold={threshold_base}")
        else:
            print(f"📋 Using standard scaled dot-product attention")
            
        model = BidirectionalTableTextModel(
            sentence_encoder, 
            embedding_dim, 
            top_k=3,
            attention_type=attention_type,
            sparse_top_k=sparse_top_k,
            window_size=window_size,
            threshold_base=threshold_base,
            use_gated_attention=use_gated_attention,
            gated_attention_mode=gated_attention_mode,
            gated_attention_hidden_dim=gated_attention_hidden_dim,
            gated_attention_dropout=gated_attention_dropout,
            gated_attention_init_bias=gated_attention_init_bias,
            use_header_conditioning=use_header_conditioning,
            disable_temperature=disable_temperature,
        )
    else:
        print("Creating TableTextEmbeddingModel...")
        print(f"🚀 Using {attention_type} attention mechanism in unidirectional model")
        if attention_type == "top_k_sparse":
            print(f"📋 Using top-{sparse_top_k} sparse attention")
        elif attention_type == "windowed":
            print(f"📋 Using windowed attention with window_size={window_size}")  
        elif attention_type == "threshold":
            print(f"📋 Using threshold attention with base_threshold={threshold_base}")
        else:
            print(f"📋 Using standard scaled dot-product attention")
            
        model = TableTextEmbeddingModel(
            sentence_encoder, 
            embedding_dim, 
            top_k=3,
            # **NEW**: Pass attention mechanism parameters to unidirectional model
            attention_type=attention_type,
            sparse_top_k=sparse_top_k,
            window_size=window_size,
            threshold_base=threshold_base,
            use_gated_attention=use_gated_attention,
            gated_attention_mode=gated_attention_mode,
            gated_attention_hidden_dim=gated_attention_hidden_dim,
            gated_attention_dropout=gated_attention_dropout,
            gated_attention_init_bias=gated_attention_init_bias,
            disable_temperature=disable_temperature,
        )
    
    # Move model to specified device
    model = model.to(device)
    
    # Load trained weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Filter out sentence encoder weights and only load custom layer weights
            model_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('sentence_encoder.'):
                    model_state_dict[key] = value
                    
            # Load only the custom layers, skip sentence encoder
            model.load_state_dict(model_state_dict, strict=False)
            print("✅ Trained weights loaded successfully (custom layers only)")
            
            # Final verification that we loaded the best model
            if "_best.pt" in model_path or "best_model" in model_path:
                print("🏆 VERIFICATION: Best model weights are now active for visualization")
            
        except Exception as e:
            print(f"❌ Failed to load trained weights: {e}")
            import traceback
            traceback.print_exc()
            raise e
    else:
        if model_path:
            print(f"⚠️ Warning: Model path {model_path} not found, using untrained model")
        else:
            print("Using untrained model")
    
    model.eval()
    
    # Final status report
    model_status = "BEST TRAINED" if (model_path and ("_best.pt" in model_path or "best_model" in model_path)) else "UNTRAINED" if not model_path else "TRAINED"
    print(f"🎯 Model ready for visualization: {model_status} model on {device}")
    
    return model

def extract_rows_and_sentences(example: Dict[str, Any], example_idx: int) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Robustly extract anchor texts (rows or docs) and target texts (sentences or tables) from an example.
    
    Args:
        example: Example dictionary
        example_idx: Index of the example for error reporting
        
    Returns:
        Tuple of (rows, primary_sentences) or (None, None) if extraction fails
    """
    try:
        is_flipped = "anchor_sentences" in example
        
        if not is_flipped:
            # Traditional TABLE_TO_DOC format
            rows = _extract_rows_robust(example)
            primary_sentences = []
            
            primary_positive = example.get("primary_positive", {})
            if primary_positive:
                primary_sentences = _extract_sentences_robust(primary_positive.get("sentences", []))
            
            if not rows or not primary_sentences:
                print(f"Skipping example {example_idx}: missing rows or sentences")
                return None, None
                
            return rows, primary_sentences
        else:
            # Flipped DOC_TO_TABLE format
            # In flipped mode: 'rows' (Anchor) contains document sentences, 
            # 'sentences' (Target) contains table rows. 
            # This maintains conceptual compatibility with the visualization pipeline.
            doc_sentences = _extract_sentences_robust(example.get("anchor_sentences", []))
            
            table_rows = []
            primary_positive = example.get("primary_positive", {})
            if primary_positive:
                table_rows = _extract_rows_robust(primary_positive)
                
            if not doc_sentences or not table_rows:
                print(f"Skipping example {example_idx}: missing doc sentences or target table rows in Flipped format")
                return None, None
                
            # For visualization of flipped: target rows go into the "sentence" slot so the heatmap renders.
            return doc_sentences, table_rows
            
    except Exception as e:
        print(f"Skipping example {example_idx}: error extracting data - {e}")
        return None, None

def detect_model_type(model) -> str:
    """
    Detect whether the model is unidirectional or bidirectional.
    
    Args:
        model: The model to inspect
        
    Returns:
        "unidirectional" or "bidirectional"
    """
    if isinstance(model, BidirectionalTableTextModel):
        return "bidirectional"
    elif isinstance(model, TableTextEmbeddingModel):
        return "unidirectional"
    else:
        # Try to detect by attributes
        if hasattr(model, 'bidirectional_attention'):
            return "bidirectional"
        elif hasattr(model, 'cross_attention'):
            return "unidirectional"
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

def compute_attention_matrix(model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel], 
                            rows: List[str], 
                            sentences: List[str], 
                            cache: Optional[IdBasedEmbeddingCache] = None,
                            normalize: bool = True,
                            return_type: str = "attention") -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute attention matrix or pair scores from the model.
    
    Args:
        model: The cross-attention model (unidirectional or bidirectional)
        rows: List of table row strings
        sentences: List of sentence strings 
        cache: Optional embedding cache for efficiency
        normalize: Whether to normalize embeddings
        return_type: "attention" for attention weights, "pairs" for pair scores, "both" for both
        
    Returns:
        For unidirectional: attention matrix [num_rows, num_sentences]
        For bidirectional: dict with 'forward_attention', 'reverse_attention', 'pair_scores'
    """
    model_type = detect_model_type(model)
    device = next(model.parameters()).device
    
    # Encode rows and sentences
    if cache:
        # Try to get embeddings from cache first (not implemented in this function)
        pass
    
    # Encode directly
    with torch.no_grad():
        rows_embeddings = model.encode_sentences(rows, normalize=normalize)
        sentences_embeddings = model.encode_sentences(sentences, normalize=normalize)
        
        # Get the target dtype from the model's parameters
        if model_type == "bidirectional":
            target_dtype = next(model.bidirectional_attention.parameters()).dtype
        else:
            target_dtype = next(model.parameters()).dtype
        
        # Add batch dimension, move to device, and cast to target dtype
        rows_tensor = rows_embeddings.unsqueeze(0).to(device=device, dtype=target_dtype)  # [1, num_rows, embedding_dim]
        sentences_tensor = sentences_embeddings.unsqueeze(0).to(device=device, dtype=target_dtype)  # [1, num_sentences, embedding_dim]
        
        if model_type == "unidirectional":
            # **NEW**: Handle different attention interfaces for unidirectional model
            if hasattr(model, 'attention_type') and model.attention_type != "standard":
                # Advanced attention modules interface: (queries_emb, keys_emb, values_emb) -> (context_vectors, attention_weights)
                context_vectors, attention_weights = model.cross_attention(
                    queries_emb=rows_tensor,
                    keys_emb=sentences_tensor,
                    values_emb=sentences_tensor
                )
            else:
                # Original CrossAttentionModule interface: (rows_embeddings, sentences_embeddings) -> (attention_weights, context_vectors)
                attention_weights, context_vectors = model.cross_attention(rows_tensor, sentences_tensor)
                
            attention_matrix = attention_weights[0].float().cpu().numpy()  # [num_rows, num_sentences]
            
            if return_type == "attention":
                return attention_matrix
            elif return_type == "both":
                return {"attention": attention_matrix}
            else:
                return attention_matrix
                
        elif model_type == "bidirectional":
            # Bidirectional model
            pair_scores, refined_rows, refined_sentences, forward_attn, reverse_attn = \
                model.bidirectional_attention(rows_tensor, sentences_tensor)
            
            result = {
                'pair_scores': pair_scores[0].float().cpu().numpy(),  # [num_rows, num_sentences]
                'forward_attention': forward_attn[0].float().cpu().numpy(),  # [num_rows, num_sentences] 
                'reverse_attention': reverse_attn[0].float().cpu().numpy(),  # [num_sentences, num_rows]
            }
            
            if return_type == "attention":
                # Return forward attention as default
                return result['forward_attention']
            elif return_type == "pairs":
                return result['pair_scores']
            elif return_type == "both":
                return result
            else:
                return result

def get_top_k_pairs(attention_matrix: np.ndarray, 
                   rows: List[str], 
                   sentences: List[str], 
                   k: int = 5) -> List[Dict[str, Any]]:
    """
    Get the top-k row-sentence pairs based on attention scores.
    
    Args:
        attention_matrix: Matrix of attention scores
        rows: List of row texts
        sentences: List of sentence texts
        k: Number of top pairs to return
        
    Returns:
        List of dictionaries containing the top pairs
    """
    num_rows, num_sentences = attention_matrix.shape
    
    # Flatten the matrix and get indices of top-k values
    flat_indices = np.argsort(attention_matrix.flatten())[::-1][:k]
    
    # Convert flat indices to 2D indices
    row_indices = flat_indices // num_sentences
    sentence_indices = flat_indices % num_sentences
    
    # Create list of top pairs
    top_pairs = []
    for i in range(min(k, len(flat_indices))):
        row_idx = row_indices[i]
        sentence_idx = sentence_indices[i]
        
        top_pairs.append({
            'row_idx': int(row_idx),
            'sentence_idx': int(sentence_idx),
            'row': rows[row_idx],
            'sentence': sentences[sentence_idx],
            'score': float(attention_matrix[row_idx, sentence_idx])
        })
    
    return top_pairs

def calculate_smart_range(data: np.ndarray, round_to: float = 0.05) -> Tuple[float, float]:
    """
    Calculate smart range for heatmap visualization.
    
    Args:
        data: Input data array
        round_to: Rounding increment (default: 0.05)
        
    Returns:
        Tuple of (vmin, vmax) with smart rounding
    """
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    
    # Round min down to nearest round_to
    vmin = math.floor(data_min / round_to) * round_to
    
    # Round max up to nearest round_to  
    vmax = math.ceil(data_max / round_to) * round_to
    
    return vmin, vmax

def get_consistent_colormap_and_range(data: np.ndarray, matrix_type: str) -> Tuple[str, float, float, Optional[float]]:
    """
    Get consistent colormap and range for different matrix types.
    Always ensures blue-to-red spectrum is visible.
    
    Args:
        data: Input data array
        matrix_type: Type of matrix ('attention', 'similarity', 'difference')
        
    Returns:
        Tuple of (colormap, vmin, vmax, center)
    """
    if matrix_type == 'attention':
        # Attention matrices: always 0 to 1, but use smart range within that
        vmin, vmax = calculate_smart_range(data)
        # Ensure we don't go below 0 or above 1 for attention
        vmin = max(0.0, vmin)
        vmax = min(1.0, vmax)
        # For attention, use center at middle of range to show full blue-to-red spectrum
        center = (vmin + vmax) / 2.0
        return "coolwarm", vmin, vmax, center
        
    elif matrix_type == 'similarity':
        # Similarity matrices: use smart range with full blue-to-red spectrum
        vmin, vmax = calculate_smart_range(data)
        # Set center at middle of range to ensure full spectrum visibility
        center = (vmin + vmax) / 2.0
        return "coolwarm", vmin, vmax, center
        
    elif matrix_type == 'difference':
        # Difference matrices: symmetric around 0
        abs_max = max(abs(np.min(data)), abs(np.max(data)))
        abs_max = math.ceil(abs_max / 0.05) * 0.05  # Round up to nearest 0.05
        return "coolwarm", -abs_max, abs_max, 0.0
        
    else:
        # Default: smart range with full blue-to-red spectrum
        vmin, vmax = calculate_smart_range(data)
        center = (vmin + vmax) / 2.0
        return "coolwarm", vmin, vmax, center

def visualize_attention_matrix(attention_matrix: np.ndarray, 
                             rows: List[str], 
                             sentences: List[str],
                             title: str = "Attention Matrix",
                             output_file: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 10),
                             show_values: bool = False) -> None:
    """
    Create a heatmap visualization of an attention matrix.
    
    Args:
        attention_matrix: N×M matrix where N is number of rows, M is number of sentences
        rows: List of row descriptions (will be simplified for display)
        sentences: List of sentence descriptions (will be simplified for display)
        title: Title for the plot
        output_file: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        show_values: Whether to show numerical values in each cell
    """
    plt.figure(figsize=figsize)
    
    # Create simplified labels
    row_labels = [f"Row {i+1}" for i in range(len(rows))]
    sentence_labels = [f"Sent {i+1}" for i in range(len(sentences))]
    
    # Determine matrix type and get consistent colors
    matrix_type = "attention" if "attention" in title.lower() else "similarity"
    cmap, vmin, vmax, center = get_consistent_colormap_and_range(attention_matrix, matrix_type)
    
    # Create the heatmap
    sns.heatmap(attention_matrix, 
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=show_values, 
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                cbar_kws={'label': 'Attention Score' if matrix_type == 'attention' else 'Similarity Score'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sentences', fontsize=12)
    plt.ylabel('Rows', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_file:
        save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def compare_attention_matrices(untrained_matrix: np.ndarray,
                              trained_matrix: np.ndarray,
                              rows: List[str], 
                              sentences: List[str],
                              title: str = "Attention Matrix Comparison",
                              output_file: Optional[str] = None,
                              figsize: Tuple[int, int] = (18, 10),
                              show_values: bool = False) -> None:
    """
    Create side-by-side comparison of untrained vs trained attention matrices.
    
    Args:
        untrained_matrix: Attention matrix from untrained model
        trained_matrix: Attention matrix from trained model
        rows: List of row descriptions
        sentences: List of sentence descriptions
        title: Title for the plot
        output_file: Path to save the plot
        figsize: Figure size
        show_values: Whether to show values in cells
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Create simplified labels
    row_labels = [f"Row {i+1}" for i in range(len(rows))]
    sentence_labels = [f"Sent {i+1}" for i in range(len(sentences))]
    
    # Get consistent range across both matrices
    combined_data = np.concatenate([untrained_matrix.flatten(), trained_matrix.flatten()])
    matrix_type = "attention" if "attention" in title.lower() else "similarity"
    cmap, vmin, vmax, center = get_consistent_colormap_and_range(combined_data, matrix_type)
    
    # Untrained matrix
    sns.heatmap(untrained_matrix, 
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=show_values,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                ax=ax1,
                cbar=False)
    ax1.set_title("Untrained Model", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sentences')
    ax1.set_ylabel('Rows')
    
    # Trained matrix
    sns.heatmap(trained_matrix, 
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=show_values,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                ax=ax2,
                cbar_kws={'label': 'Attention Score' if matrix_type == 'attention' else 'Similarity Score'})
    ax2.set_title("Trained Model", fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sentences')
    ax2.set_ylabel('')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_top_pairs(top_pairs: List[Dict[str, Any]], 
                       title: str = "Top-5 Attention Pairs",
                       output_file: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualize the top-k row-sentence pairs using a bar chart.
    
    Args:
        top_pairs: List of top pairs
        title: Title for the plot
        output_file: Path to save the visualization
        figsize: Figure size
    """
    # Extract scores and labels
    scores = [pair['score'] for pair in top_pairs]
    labels = [f"Row {pair['row_idx']+1} - Sent {pair['sentence_idx']+1}" for pair in top_pairs]
    
    # Create bar chart
    plt.figure(figsize=figsize)
    bars = plt.barh(labels, scores, color='skyblue')
    
    # Add score values at the end of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{scores[i]:.6f}', va='center')
    
    # Set title and labels
    plt.title(title)
    plt.xlabel("Attention Score")
    plt.ylabel("Row-Sentence Pair")
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_plot_multi_format(output_file, bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close()

def compare_top_pairs(untrained_pairs: List[Dict[str, Any]],
                     trained_pairs: List[Dict[str, Any]],
                     title: str = "Top-5 Attention Pairs Comparison",
                     output_file: Optional[str] = None,
                     figsize: Tuple[int, int] = (15, 6)) -> None:
    """
    Create a side-by-side comparison of top pairs from untrained and trained models.
    
    Args:
        untrained_pairs: Top pairs from untrained model
        trained_pairs: Top pairs from trained model
        title: Title for the overall plot
        output_file: Path to save the visualization
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    
    # Extract scores and labels for untrained model
    untrained_scores = [pair['score'] for pair in untrained_pairs]
    untrained_labels = [f"Row {pair['row_idx']+1} - Sent {pair['sentence_idx']+1}" for pair in untrained_pairs]
    
    # Extract scores and labels for trained model
    trained_scores = [pair['score'] for pair in trained_pairs]
    trained_labels = [f"Row {pair['row_idx']+1} - Sent {pair['sentence_idx']+1}" for pair in trained_pairs]
    
    # Calculate overall max for consistent bar scaling
    score_max = max(max(untrained_scores), max(trained_scores))
    
    # Create untrained model bar chart
    bars1 = axes[0].barh(untrained_labels, untrained_scores, color='lightblue')
    axes[0].set_title("Untrained Model")
    axes[0].set_xlabel("Attention Score")
    axes[0].set_ylabel("Row-Sentence Pair")
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    axes[0].set_xlim(0, score_max * 1.1)  # Same scale for both plots
    
    # Add score values at the end of each bar
    for i, bar in enumerate(bars1):
        axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{untrained_scores[i]:.6f}', va='center')
    
    # Create trained model bar chart
    bars2 = axes[1].barh(trained_labels, trained_scores, color='coral')
    axes[1].set_title("Trained Model")
    axes[1].set_xlabel("Attention Score")
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    axes[1].set_xlim(0, score_max * 1.1)  # Same scale for both plots
    
    # Add score values at the end of each bar
    for i, bar in enumerate(bars2):
        axes[1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{trained_scores[i]:.6f}', va='center')
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Make room for the suptitle
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_plot_multi_format(output_file, bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close()

def print_top_pair_details(top_pairs: List[Dict[str, Any]], model_type: str = "") -> None:
    """
    Print detailed information about the top pairs.
    
    Args:
        top_pairs: List of top pairs
        model_type: Type of model (e.g., "Trained", "Untrained")
    """
    print(f"\n{model_type} Model - Top-5 Pairs by Attention Score:")
    print("="*50)
    
    for i, pair in enumerate(top_pairs):
        print(f"Pair {i+1} (Score: {pair['score']:.6f}):")
        print(f"  Row: {pair['row']}")
        print(f"  Sentence: {pair['sentence']}")
        print()

# process_example_comparison function removed - it was generating old visualization files

def process_example_single(model: TableTextEmbeddingModel,
                         example: Dict[str, Any],
                         example_idx: int,
                         output_dir: str,
                         cache: Optional[IdBasedEmbeddingCache] = None,
                         save_visualizations: bool = True,
                         model_type: str = "") -> None:
    """
    Process a single example with a single model.
    
    Args:
        model: Model to use
        example: Example to process
        example_idx: Index of the example
        output_dir: Directory to save visualizations
        cache: Cache for embeddings
        save_visualizations: Whether to save visualizations
        model_type: Type of model (e.g., "Trained", "Untrained")
    """
    # Create output directory
    output_path = Path(output_dir) / f"example_{example_idx}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get anchor rows and primary sentences with robust extraction
    rows, primary_sentences = extract_rows_and_sentences(example, example_idx)
    if rows is None or primary_sentences is None:
        return
    
    print(f"\nProcessing Example {example_idx} with {model_type} model")
    print(f"Number of Rows: {len(rows)}")
    
    # Process primary positive context
    print(f"Primary Positive Context: {len(primary_sentences)} sentences")
    
    if primary_sentences:
        # Compute attention matrix
        attention_matrix = compute_attention_matrix(model, rows, primary_sentences, cache, normalize=False)
        
        if save_visualizations:
            # Visualize attention matrix
            matrix_file = output_path / f"primary_attention_{model_type.lower()}.pdf"
            visualize_attention_matrix(
                attention_matrix, 
                rows, 
                primary_sentences,
                title=f"Example {example_idx}: Primary Positive Attention ({model_type})",
                output_file=matrix_file,
                show_values=True
            )
        
        # Get top pairs
        top_pairs = get_top_k_pairs(attention_matrix, rows, primary_sentences, k=5)
        
        if save_visualizations:
            # Visualize top pairs
            pairs_file = output_path / f"primary_top_pairs_{model_type.lower()}.pdf"
            visualize_top_pairs(
                top_pairs,
                title=f"Example {example_idx}: Top-5 Primary Positive Pairs ({model_type})",
                output_file=pairs_file
            )
        
        # Save top pair details
        details_file = output_path / f"primary_top_pairs_{model_type.lower()}.txt"
        with open(details_file, 'w', encoding='utf-8') as f:
            f.write(f"{model_type} Model - Top-5 Pairs by Attention Score:\n")
            f.write("="*50 + "\n\n")
            
            for i, pair in enumerate(top_pairs):
                f.write(f"Pair {i+1} (Score: {pair['score']:.6f}):\n")
                f.write(f"  Row: {pair['row']}\n")
                f.write(f"  Sentence: {pair['sentence']}\n\n")
    
    # Process negative contexts (first negative only for brevity)
    if example["negatives"]:
        negative = example["negatives"][0]
        negative_sentences = negative["sentences"]
        
        if negative_sentences:
            print(f"\nNegative Context: {len(negative_sentences)} sentences")
            
            # Compute attention matrix
            attention_matrix = compute_attention_matrix(model, rows, negative_sentences, cache, normalize=False)
            
            if save_visualizations:
                # Visualize attention matrix
                matrix_file = output_path / f"negative_attention_{model_type.lower()}.pdf"
                visualize_attention_matrix(
                    attention_matrix, 
                    rows, 
                    negative_sentences,
                    title=f"Example {example_idx}: Negative Attention ({model_type})",
                    output_file=matrix_file,
                    show_values=True
                )
            
            # Get top pairs
            top_pairs = get_top_k_pairs(attention_matrix, rows, negative_sentences, k=5)
            
            if save_visualizations:
                # Visualize top pairs
                pairs_file = output_path / f"negative_top_pairs_{model_type.lower()}.pdf"
                visualize_top_pairs(
                    top_pairs,
                    title=f"Example {example_idx}: Top-5 Negative Pairs ({model_type})",
                    output_file=pairs_file
                )
            
            # Save top pair details
            details_file = output_path / f"negative_top_pairs_{model_type.lower()}.txt"
            with open(details_file, 'w', encoding='utf-8') as f:
                f.write(f"{model_type} Model - Top-5 Pairs by Attention Score:\n")
                f.write("="*50 + "\n\n")
                
                for i, pair in enumerate(top_pairs):
                    f.write(f"Pair {i+1} (Score: {pair['score']:.6f}):\n")
                    f.write(f"  Row: {pair['row']}\n")
                    f.write(f"  Sentence: {pair['sentence']}\n\n")

def compute_raw_transformer_similarities(model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel], 
                                        rows: List[str], 
                                        sentences: List[str],
                                        base_model_name: Optional[str] = None) -> np.ndarray:
    """
    Compute the raw similarity matrix using the model's own sentence encoder,
    without any cross-attention or refinement layers.
    
    Args:
        model: The model whose sentence encoder will be used
        rows: List of row texts
        sentences: List of sentence texts
        base_model_name: Unused, kept for API compatibility
        
    Returns:
        NumPy array of shape [num_rows, num_sentences] containing similarity scores
    """
    from sentence_transformers import util
    
    with torch.no_grad():
        row_embeddings = model.sentence_encoder.encode(rows, convert_to_tensor=True, normalize_embeddings=True)
        sentence_embeddings = model.sentence_encoder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
        similarity_matrix = util.cos_sim(row_embeddings, sentence_embeddings).float().cpu().numpy()
    
    return similarity_matrix

def compute_comprehensive_similarities(model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel], 
                                     rows: List[str], 
                                     sentences: List[str],
                                     cache: Optional[IdBasedEmbeddingCache] = None,
                                     aggregation_method: str = "entropy_regularized") -> Dict[str, np.ndarray]:
    """
    Compute multiple types of similarities from the trained model to understand what it learned.
    
    Args:
        model: The trained model to analyze (unidirectional or bidirectional)
        rows: List of row texts
        sentences: List of sentence texts
        cache: Optional cache for embeddings
        aggregation_method: Aggregation method to use for full model similarities
        
    Returns:
        Dictionary containing different similarity matrices:
        - 'raw_embeddings': Cosine similarities between raw sentence encoder embeddings
        - 'model_similarities': Similarities from the full model forward pass
        - 'cross_attention': Cross-attention weights from the attention mechanism
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Detect model type
    model_type = detect_model_type(model)
    
    results = {}
    
    with torch.no_grad():
        # 1. Raw embedding similarities (using model's own sentence encoder)
        print("  Computing raw embedding similarities...")
        from sentence_transformers import util
        row_embeddings = model.sentence_encoder.encode(rows, convert_to_tensor=True, normalize_embeddings=True)
        sentence_embeddings = model.sentence_encoder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
        raw_similarities = util.cos_sim(row_embeddings, sentence_embeddings).float().cpu().numpy()
        results['raw_embeddings'] = raw_similarities
        
        # 2. Full model similarities (through the complete forward pass with specified aggregation)
        print(f"  Computing full model similarities using {aggregation_method}...")
        # Add batch dimension
        rows_tensor = row_embeddings.unsqueeze(0).to(device)  # [1, num_rows, embedding_dim]
        sentences_tensor = sentence_embeddings.unsqueeze(0).to(device)  # [1, num_sentences, embedding_dim]
        
        # Get model similarities for each row-sentence pair using specified aggregation
        model_similarities = np.zeros((len(rows), len(sentences)))
        for i in range(len(rows)):
            for j in range(len(sentences)):
                # Create single-row and single-sentence tensors
                single_row = rows_tensor[:, i:i+1, :]  # [1, 1, embedding_dim]
                single_sentence = sentences_tensor[:, j:j+1, :]  # [1, 1, embedding_dim]
                
                # Get similarity through full model with specified aggregation method
                similarity_score, _ = model(single_row, single_sentence, aggregation_method=aggregation_method)
                model_similarities[i, j] = similarity_score.item()
        
        results['model_similarities'] = model_similarities
        
        # 3. Cross-attention weights
        print("  Computing cross-attention weights...")
        if model_type == "bidirectional":
            # For bidirectional models, use forward attention weights
            _, _, _, forward_attn, _ = model.bidirectional_attention(rows_tensor, sentences_tensor)
            cross_attention = forward_attn[0].float().cpu().numpy()  # [num_rows, num_sentences]
        else:
            # For unidirectional models, use cross-attention weights - handle different interfaces
            if hasattr(model, 'attention_type') and model.attention_type != "standard":
                # Advanced attention modules use (queries_emb, keys_emb, values_emb) interface
                _, attention_weights = model.cross_attention(
                    queries_emb=rows_tensor,
                    keys_emb=sentences_tensor,
                    values_emb=sentences_tensor
                )
            else:
                # Original CrossAttentionModule uses (rows_embeddings, sentences_embeddings) interface
                attention_weights, _ = model.cross_attention(rows_tensor, sentences_tensor)
            cross_attention = attention_weights[0].float().cpu().numpy()  # [num_rows, num_sentences]
        results['cross_attention'] = cross_attention
    
    return results

def visualize_three_way_comparison(
    raw_similarities: np.ndarray,
    initial_attention: np.ndarray,
    trained_attention: np.ndarray,
    rows: List[str], 
    sentences: List[str],
    title: str = "Three-way Similarity Comparison",
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 10)) -> None:
    """
    Create a three-panel comparison of raw similarities, initial attention, and trained attention.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Create simplified labels
    row_labels = [f"Row {i+1}" for i in range(len(rows))]
    sentence_labels = [f"Sent {i+1}" for i in range(len(sentences))]
    
    # Get consistent range across all three matrices
    combined_data = np.concatenate([
        raw_similarities.flatten(), 
        initial_attention.flatten(), 
        trained_attention.flatten()
    ])
    cmap, vmin, vmax, center = get_consistent_colormap_and_range(combined_data, 'similarity')
    
    # Panel 1: Raw Similarities
    sns.heatmap(raw_similarities,
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                ax=axes[0],
                cbar=False)
    axes[0].set_title("Raw Transformer\nSimilarities", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Sentences')
    axes[0].set_ylabel('Rows')
    
    # Panel 2: Initial Attention
    sns.heatmap(initial_attention,
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                ax=axes[1],
                cbar=False)
    axes[1].set_title("Initial Cross-Attention\n(Untrained)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sentences')
    axes[1].set_ylabel('')
    
    # Panel 3: Trained Attention
    sns.heatmap(trained_attention,
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                ax=axes[2],
                cbar_kws={'label': 'Similarity Score'})
    axes[2].set_title("Trained Cross-Attention\n(After Training)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Sentences')
    axes[2].set_ylabel('')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_file:
        save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved three-way comparison to {output_file}")
    else:
        plt.show()
    
    plt.close()

def compare_top_k_similarities(
    raw_similarities: np.ndarray,
    initial_attention: np.ndarray,
    trained_attention: np.ndarray,
    rows: List[str], 
    sentences: List[str],
    k: int = 5,
    title: str = "Top-5 Similarity Pairs Comparison",
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 8)) -> None:
    """
    Compare the top-k pairs from all three similarity matrices.
    
    Args:
        raw_similarities: Similarity matrix from raw sentence transformer
        initial_attention: Attention matrix from untrained model
        trained_attention: Attention matrix from trained model
        rows: List of row texts
        sentences: List of sentence texts
        k: Number of top pairs to compare
        title: Title for the overall plot
        output_file: Path to save the visualization
        figsize: Figure size
    """
    # Get top pairs for each method
    raw_pairs = get_top_k_pairs(raw_similarities, rows, sentences, k)
    initial_pairs = get_top_k_pairs(initial_attention, rows, sentences, k)
    trained_pairs = get_top_k_pairs(trained_attention, rows, sentences, k)
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # For raw similarities
    labels1 = [f"Row {p['row_idx']+1} - Sent {p['sentence_idx']+1}" for p in raw_pairs]
    scores1 = [p['score'] for p in raw_pairs]
    bars1 = axes[0].barh(labels1, scores1, color='lightblue')
    axes[0].set_title("Raw Transformer Similarities")
    axes[0].set_ylabel("Row-Sentence Pair")
    for i, bar in enumerate(bars1):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{scores1[i]:.4f}', va='center')
    
    # For initial attention
    labels2 = [f"Row {p['row_idx']+1} - Sent {p['sentence_idx']+1}" for p in initial_pairs]
    scores2 = [p['score'] for p in initial_pairs]
    bars2 = axes[1].barh(labels2, scores2, color='lightgreen')
    axes[1].set_title("Initial Cross-Attention")
    axes[1].set_ylabel("Row-Sentence Pair")
    for i, bar in enumerate(bars2):
        axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{scores2[i]:.4f}', va='center')
    
    # For trained attention
    labels3 = [f"Row {p['row_idx']+1} - Sent {p['sentence_idx']+1}" for p in trained_pairs]
    scores3 = [p['score'] for p in trained_pairs]
    bars3 = axes[2].barh(labels3, scores3, color='coral')
    axes[2].set_title("Trained Cross-Attention")
    axes[2].set_xlabel("Similarity/Attention Score")
    axes[2].set_ylabel("Row-Sentence Pair")
    for i, bar in enumerate(bars3):
        axes[2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{scores3[i]:.4f}', va='center')
    
    # Set common properties
    for ax in axes:
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Find max score to set consistent x limits
    max_score = max(max(scores1), max(scores2), max(scores3))
    for ax in axes:
        ax.set_xlim(0, max_score * 1.1)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Make room for the suptitle
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_plot_multi_format(output_file, bbox_inches='tight')
        print(f"Saved top-k similarities comparison to {output_file}")
    
    # Close the figure to free memory
    plt.close()

def process_example_three_way_comparison(
    untrained_model: TableTextEmbeddingModel,
    trained_model: TableTextEmbeddingModel,
    example: Dict[str, Any],
    example_idx: int,
    output_dir: str,
    base_model_name: str = "answerdotai/ModernBERT-base",
    untrained_cache: Optional[IdBasedEmbeddingCache] = None,
    trained_cache: Optional[IdBasedEmbeddingCache] = None,
    save_visualizations: bool = True) -> None:
    """
    Process a single example and generate three-way comparisons between raw similarities,
    initial cross-attention, and trained cross-attention.
    
    Args:
        untrained_model: Untrained model
        trained_model: Trained model
        example: Example to process
        example_idx: Index of the example
        output_dir: Directory to save visualizations
        base_model_name: Name of the base model for raw similarity computation
        untrained_cache: Optional cache for untrained model
        trained_cache: Optional cache for trained model
        save_visualizations: Whether to save visualizations to disk
    """
    # Extract data from example using robust extraction
    anchor_id = example.get("anchor_id", f"example_{example_idx}")
    rows, sentences = extract_rows_and_sentences(example, example_idx)
    if rows is None or sentences is None:
        return
    
    print(f"\nProcessing example {example_idx} (ID: {anchor_id}):")
    print(f"  {len(rows)} rows, {len(sentences)} sentences")
    
    # Compute raw transformer similarities
    print("Computing raw transformer similarities...")
    raw_similarities = compute_raw_transformer_similarities(untrained_model, rows, sentences, base_model_name)
    
    # Compute initial cross-attention
    print("Computing initial cross-attention...")
    initial_attention = compute_attention_matrix(untrained_model, rows, sentences, untrained_cache)
    
    # Compute trained cross-attention
    print("Computing trained cross-attention...")
    trained_attention = compute_attention_matrix(trained_model, rows, sentences, trained_cache)
    
    # Generate three-way comparison
    if save_visualizations:
        print("Generating three-way comparison...")
        output_file = f"{output_dir}/example_{example_idx}_three_way_comparison.png"
        visualize_three_way_comparison(
            raw_similarities=raw_similarities,
            initial_attention=initial_attention,
            trained_attention=trained_attention,
            rows=rows,
            sentences=sentences,
            title=f"Example {example_idx} - Three-way Similarity Comparison",
            output_file=output_file
        )
        
        # Generate top-k comparison
        print("Generating top-k similarities comparison...")
        top_k_output_file = f"{output_dir}/example_{example_idx}_top_k_comparison.png"
        compare_top_k_similarities(
            raw_similarities=raw_similarities,
            initial_attention=initial_attention,
            trained_attention=trained_attention,
            rows=rows,
            sentences=sentences,
            k=5,
            title=f"Example {example_idx} - Top-5 Similarity Pairs",
            output_file=top_k_output_file
        )
    
    # Get top pairs for each method
    raw_pairs = get_top_k_pairs(raw_similarities, rows, sentences, k=5)
    initial_pairs = get_top_k_pairs(initial_attention, rows, sentences, k=5)
    trained_pairs = get_top_k_pairs(trained_attention, rows, sentences, k=5)
    
    # Print details
    print_top_pair_details(raw_pairs, model_type="Raw Transformer")
    print_top_pair_details(initial_pairs, model_type="Initial Cross-Attention")
    print_top_pair_details(trained_pairs, model_type="Trained Cross-Attention")

def save_top_pair_details(top_pairs, model_type, output_file):
    """
    Save top pair details to a text file instead of printing them.
    
    Args:
        top_pairs: List of top pairs
        model_type: Type of model (e.g., "Trained", "Untrained")
        output_file: Path to save the details
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{model_type} Model - Top-5 Pairs by Attention Score:\n")
        f.write("="*50 + "\n\n")
        
        for i, pair in enumerate(top_pairs):
            f.write(f"Pair {i+1} (Score: {pair['score']:.6f}):\n")
            f.write(f"  Row {pair['row_idx']+1}: {pair['row']}\n")
            f.write(f"  Sentence {pair['sentence_idx']+1}: {pair['sentence']}\n\n")

def compare_all_attention_matrices(untrained_primary_matrix, 
                                  trained_primary_matrix,
                                  untrained_negative_matrix,
                                  trained_negative_matrix,
                                  rows, 
                                  primary_sentences,
                                  negative_sentences,
                                  title="All Attention Matrix Comparison",
                                  output_file=None,
                                  figsize=(20, 15),
                                  show_values=False):
    """
    Create a comprehensive comparison of attention matrices from untrained and trained models,
    showing both primary positive and negative contexts in one visualization.
    
    Args:
        untrained_primary_matrix: Primary attention matrix from untrained model
        trained_primary_matrix: Primary attention matrix from trained model
        untrained_negative_matrix: Negative attention matrix from untrained model
        trained_negative_matrix: Negative attention matrix from trained model
        rows: List of row texts
        primary_sentences: List of primary context sentence texts
        negative_sentences: List of negative context sentence texts
        title: Title for the overall plot
        output_file: Path to save the visualization
        figsize: Figure size
        show_values: Whether to show attention values in cells
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Prepare row and sentence labels
    row_labels = [f"Row {i+1}" for i in range(len(rows))]
    primary_sentence_labels = [f"P-Sent {i+1}" for i in range(len(primary_sentences))]
    negative_sentence_labels = [f"N-Sent {i+1}" for i in range(len(negative_sentences))]
    
    # Calculate overall min and max for consistent color scaling across all matrices
    all_matrices = [
        untrained_primary_matrix, trained_primary_matrix,
        untrained_negative_matrix, trained_negative_matrix
    ]
    vmin = min(matrix.min() for matrix in all_matrices)
    vmax = max(matrix.max() for matrix in all_matrices)
    
    # Create untrained model primary heatmap (top-left)
    cmap, vmin_adj, vmax_adj, center = get_consistent_colormap_and_range(untrained_primary_matrix, "attention")
    sns.heatmap(
        untrained_primary_matrix,
        annot=show_values,
        fmt=".4f" if show_values else None,
        cmap=cmap,
        xticklabels=primary_sentence_labels,
        yticklabels=row_labels,
        ax=axes[0, 0],
        vmin=vmin_adj,
        vmax=vmax_adj,
        center=center,
        annot_kws={"size": 6} if show_values else {},
        cbar_kws={"label": "Attention Score"}
    )
    axes[0, 0].set_title("Untrained Model - Primary Context")
    axes[0, 0].set_xlabel("Primary Sentences")
    axes[0, 0].set_ylabel("Table Rows")
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
    
    # Create trained model primary heatmap (top-right)
    sns.heatmap(
        trained_primary_matrix,
        annot=show_values,
        fmt=".4f" if show_values else None,
        cmap=cmap,
        xticklabels=primary_sentence_labels,
        yticklabels=row_labels,  # Show row labels on all plots
        ax=axes[0, 1],
        vmin=vmin_adj,
        vmax=vmax_adj,
        center=center,
        annot_kws={"size": 6} if show_values else {},
        cbar_kws={"label": "Attention Score"}
    )
    axes[0, 1].set_title("Trained Model - Primary Context")
    axes[0, 1].set_xlabel("Primary Sentences")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
    
    # Create untrained model negative heatmap (bottom-left)
    sns.heatmap(
        untrained_negative_matrix,
        annot=show_values,
        fmt=".4f" if show_values else None,
        cmap=cmap,
        xticklabels=negative_sentence_labels,
        yticklabels=row_labels,
        ax=axes[1, 0],
        vmin=vmin_adj,
        vmax=vmax_adj,
        center=center,
        annot_kws={"size": 6} if show_values else {},
        cbar_kws={"label": "Attention Score"}
    )
    axes[1, 0].set_title("Untrained Model - Negative Context")
    axes[1, 0].set_xlabel("Negative Sentences")
    axes[1, 0].set_ylabel("Table Rows")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
    
    # Create trained model negative heatmap (bottom-right)
    sns.heatmap(
        trained_negative_matrix,
        annot=show_values,
        fmt=".4f" if show_values else None,
        cmap=cmap,
        xticklabels=negative_sentence_labels,
        yticklabels=row_labels,  # Show row labels on all plots
        ax=axes[1, 1],
        vmin=vmin_adj,
        vmax=vmax_adj,
        center=center,
        annot_kws={"size": 6} if show_values else {},
        cbar_kws={"label": "Attention Score"}
    )
    axes[1, 1].set_title("Trained Model - Negative Context")
    axes[1, 1].set_xlabel("Negative Sentences")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Make room for the suptitle
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_plot_multi_format(output_file, bbox_inches='tight')
        print(f"Saved combined attention matrix comparison to {output_file}")
    
    # Close the figure to free memory
    plt.close()

# process_example_visualization function removed - it was generating old visualization files

def visualize_models_with_three_way_comparison(
    trained_model=None,
    trained_model_path=None, 
    examples=None, 
    output_dir=None,
    base_model_name="answerdotai/ModernBERT-base",
    example_indices="0",
    normalize_attention=True,
    enable_lora=False,
    aggregation_method="entropy_regularized",
    device=None
):
    """
    Comprehensive visualization function for trained model analysis.
    
    Args:
        trained_model: The trained model to visualize (optional if trained_model_path provided)
        trained_model_path: Path to the trained model checkpoint (optional if trained_model provided)
        examples: List of examples to visualize
        output_dir: Base output directory for visualizations
        base_model_name: Name of the base model (used for dynamic loading)
        example_indices: Comma-separated indices of examples to visualize
        normalize_attention: Whether to normalize attention scores with softmax (legacy parameter)
        enable_lora: Whether to enable LoRA training (legacy parameter)
        aggregation_method: Aggregation method to use for model similarities
        device: Device to use for computation (defaults to model's device)
    """
    print(f"\nRunning comprehensive visualizations using {aggregation_method}...")
    
    # Load trained model if path provided
    if trained_model is None and trained_model_path is not None:
        print(f"🔍 Loading trained model from: {trained_model_path}")
        # Determine device for visualization
        viz_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        trained_model = load_model(trained_model_path, base_model_name, device=viz_device)
        print(f"✅ Trained model loaded with dynamic dimension detection on {viz_device}")
    elif trained_model is None:
        raise ValueError("Either trained_model or trained_model_path must be provided")
    
    # Create visualizations directory
    visualization_output_dir = Path(output_dir) / "visualizations"
    visualization_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if device is None:
        device = next(trained_model.parameters()).device
    
    # Parse example indices
    indices = [int(idx) for idx in example_indices.split(",")]
    selected_examples = []
    for idx in indices:
        if 0 <= idx < len(examples):
            selected_examples.append(examples[idx])
        else:
            print(f"Warning: Example index {idx} is out of range (0-{len(examples)-1})")
    
    if not selected_examples:
        print("No valid examples selected. Skipping visualization.")
        return
    
    # Process each selected example
    for i, example in enumerate(selected_examples):
        # Get the original index for output file naming
        idx = indices[i]
        
        # Extract data from example using robust extraction
        anchor_id = example.get("anchor_id", f"example_{idx}")
        rows, sentences = extract_rows_and_sentences(example, idx)
        if rows is None or sentences is None:
            continue
        
        print(f"\nProcessing example {idx} (ID: {anchor_id}):")
        print(f"  {len(rows)} rows, {len(sentences)} sentences")
        print(f"  Using aggregation method: {aggregation_method}")
        
        # Generate comprehensive analysis visualization directly in visualizations folder
        print("\nGenerating comprehensive 4-panel analysis...")
        visualize_comprehensive_model_analysis(
            trained_model=trained_model,
            trained_model_path=None,
            example=example,
            example_idx=idx,
            output_dir=output_dir,
            aggregation_method=aggregation_method
        )
    
    print(f"\nAll visualizations saved to {visualization_output_dir}")
    return visualization_output_dir

def main():
    """Main visualization script that supports both unidirectional and bidirectional models."""
    parser = argparse.ArgumentParser(description="Visualize cross-attention patterns")
    parser.add_argument("--trained_model", type=str, required=True, 
                        help="Path to trained model checkpoint")
    parser.add_argument("--base_model", type=str, default="answerdotai/ModernBERT-base",
                        help="Base sentence transformer model")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str, default="./visualizations", 
                        help="Output directory for visualizations")
    parser.add_argument("--example_indices", type=str, default="0",
                        help="Comma-separated example indices to visualize")
    parser.add_argument("--normalize_attention", action="store_true",
                        help="Normalize attention with softmax")
    parser.add_argument("--enable_lora", action="store_true",
                        help="Enable LoRA training for the untrained model")
    parser.add_argument("--aggregation_method", type=str, default="entropy_regularized",
                        choices=["mean", "top_k_sum", "top_k_mean", "weighted_top_k", 
                                "max", "attention_weighted", "sparse_top_k", "entropy_regularized",
                                "top_k_pairs", "max_pairs", "mean_pairs", "weighted_pairs", "sparse_pairs"],
                        help="Aggregation method for comprehensive analysis")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu, defaults to auto-detect)")
    parser.add_argument("--model_type", type=str, choices=["auto", "unidirectional", "bidirectional"], default="auto",
                        help="Force specific model type (auto-detects by default)")
    parser.add_argument("--use_cache", action="store_true",
                        help="Use caching for embeddings")
    parser.add_argument("--init_method", type=str, default="xavier_uniform",
                        choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", "normal", "uniform"],
                        help="Initialization method for untrained baseline models")
    parser.add_argument("--init_method_params", type=str, default="{}",
                        help="JSON string of initialization method parameters")
    
    args = parser.parse_args()
    _bootstrap_visualization_hf_assets(args.base_model)
    
    # Parse init_method_params JSON
    try:
        init_method_params = json.loads(args.init_method_params) if args.init_method_params != "{}" else None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON for init_method_params: {args.init_method_params}, using None")
        init_method_params = None
    
    # Load the dataset
    print(f"Loading dataset from {args.data_file}")
    dataset = load_row_level_dataset(args.data_file)
    
    if not dataset:
        print("No data loaded, exiting")
        return
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Auto-detect device if not specified
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Try to load the trained model and detect its type
    print(f"Loading trained model from {args.trained_model}")
    
    # First, try to detect model type from the checkpoint
    model_type = args.model_type
    if model_type == "auto":
        try:
            checkpoint = torch.load(args.trained_model, map_location='cpu')
            
            # Look for bidirectional-specific keys
            bidirectional_keys = [
                'bidirectional_attention', 'forward_projection', 'reverse_projection',
                'refinement_layers', 'forward_norm', 'reverse_norm'
            ]
            
            has_bidirectional = any(
                any(key in param_name for param_name in checkpoint.keys())
                for key in bidirectional_keys
            )
            
            model_type = "bidirectional" if has_bidirectional else "unidirectional"
            print(f"Auto-detected model type: {model_type}")
            
        except Exception as e:
            print(f"Could not auto-detect model type from checkpoint: {e}")
            model_type = "unidirectional"  # Default fallback
    
    # Load the appropriate model
    if model_type == "bidirectional":
        print("Loading bidirectional model...")
        use_header_conditioning = False

        model_dir = os.path.dirname(args.trained_model)
        for config_file in ['args.json', 'training_config.json', 'config.json']:
            config_path = os.path.join(model_dir, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    use_header_conditioning = config_data.get('use_header_conditioning', False)
                    break
                except Exception:
                    pass
        
        # Initialize sentence encoder
        sentence_encoder = _load_sentence_transformer_encoder(
            args.base_model,
            device=device,
            model_kwargs={"dtype": torch.bfloat16},
        )
        embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
        
        # Create bidirectional model and load weights
        trained_model = BidirectionalTableTextModel(
            sentence_encoder=sentence_encoder,
            embedding_dim=embedding_dim,
            top_k=5,  # Default, can be overridden
            use_header_conditioning=use_header_conditioning,
        ).to(device)
        
        trained_model.load_state_dict(torch.load(args.trained_model, map_location=device))
        trained_model.eval()
        
        print(f"Bidirectional model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in trained_model.parameters()):,}")
        
        # Run bidirectional-specific visualizations
        visualize_bidirectional_models(
            trained_model=trained_model,
            examples=dataset,
            output_dir=args.output_dir,
            base_model_name=args.base_model,
            example_indices=args.example_indices,
            device=device,
            init_method=args.init_method,
            init_method_params=init_method_params
        )
        
    else:  # unidirectional
        print("Loading unidirectional model...")
        
        # Use existing load_model function
        trained_model = load_model(args.trained_model, args.base_model)
        trained_model.to(device)
        
        print(f"Unidirectional model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in trained_model.parameters()):,}")
        
        # Run standard visualizations
        visualize_models_with_three_way_comparison(
            trained_model=trained_model,
            examples=dataset,
            output_dir=args.output_dir,
            base_model_name=args.base_model,
            example_indices=args.example_indices,
            normalize_attention=args.normalize_attention,
            enable_lora=args.enable_lora,
            aggregation_method=args.aggregation_method,
            device=device
        )

def visualize_bidirectional_models(trained_model: BidirectionalTableTextModel,
                                 examples: List[Dict[str, Any]],
                                 output_dir: str,
                                 base_model_name: str,
                                 example_indices: str = "0",
                                 device: Optional[torch.device] = None,
                                 init_method: str = "xavier_uniform",
                                 init_method_params: dict = None) -> None:
    """
    Main visualization pipeline for bidirectional models.
    
    Args:
        trained_model: Trained bidirectional model
        examples: Dataset examples
        output_dir: Output directory for visualizations
        base_model_name: Base model name for creating untrained comparison
        example_indices: Comma-separated indices of examples to visualize
        device: Device for computation
        init_method: Initialization method for untrained baseline model
        init_method_params: Parameters for initialization method
    """
    print("\nRunning bidirectional model visualizations...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse example indices
    indices = [int(idx) for idx in example_indices.split(",")]
    selected_examples = []
    for idx in indices:
        if 0 <= idx < len(examples):
            selected_examples.append((examples[idx], idx))
        else:
            print(f"Warning: Example index {idx} is out of range (0-{len(examples)-1})")
    
    if not selected_examples:
        print("No valid examples selected. Skipping visualization.")
        return
    
    # Create SIMPLE untrained bidirectional model for fair comparison
    print("🎯 Creating SIMPLE untrained bidirectional baseline model for fair comparison...")
    if device is None:
        device = next(trained_model.parameters()).device
    
    sentence_encoder = _load_sentence_transformer_encoder(
        base_model_name,
        device=device,
        model_kwargs={"dtype": torch.bfloat16},
    )
    embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
    
    # Create simple baseline model with default parameters (not same as trained model)
    untrained_model = BidirectionalTableTextModel(
        sentence_encoder=sentence_encoder,
        embedding_dim=embedding_dim,
        trainable_encoder=False,
        use_cross_attention_lora=False,  # No LoRA for baseline
        lora_rank=16,  # Default values
        lora_alpha=32.0,
        lora_dropout=0.1,
        top_k=3,  # Simpler top-k
        pair_score_method="cosine",
        share_weights=False,  # No weight sharing for baseline  
        use_refinement=False,
        use_self_attention=False,
        attention_type="standard",  # Standard attention, not advanced
        sparse_top_k=3,
        window_size=5,
        threshold_base=0.1,
        use_header_conditioning=getattr(trained_model, 'use_header_conditioning', False),
        init_method=init_method,  # Use passed initialization method
        init_method_params=init_method_params
    ).to(device)
    untrained_model.eval()
    
    # Process each example
    for example, idx in selected_examples:
        print(f"\nProcessing example {idx}...")
        
        # Extract data using robust extraction
        rows, sentences = extract_rows_and_sentences(example, idx)
        if rows is None or sentences is None:
            continue
        
        print(f"  {len(rows)} rows, {len(sentences)} sentences")
        
        # Create example-specific directory
        example_dir = output_path / f"example_{idx}"
        example_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Individual model visualizations
        print("  Creating individual model visualizations...")
        process_bidirectional_example(
            untrained_model, example, idx, str(example_dir), "untrained"
        )
        process_bidirectional_example(
            trained_model, example, idx, str(example_dir), "trained"
        )
        
        # 2. Side-by-side comparison
        print("  Creating model comparison...")
        compare_bidirectional_models(
            untrained_model, trained_model, rows, sentences,
            title=f"Example {idx} - Bidirectional Model Comparison",
            output_file=str(example_dir / f"example_{idx}_bidirectional_comparison.png")
        )
        
        print(f"  Completed example {idx}")
    
    print("\nBidirectional model visualization complete!")
    print(f"All outputs saved to: {output_path}")

def visualize_comprehensive_model_analysis(
    trained_model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel] = None,
    trained_model_path: str = None,
    example: Dict[str, Any] = None,
    example_idx: int = 0,
    output_dir: str = "",
    aggregation_method: str = "entropy_regularized",
    base_model_name: str = "answerdotai/ModernBERT-base",
    figsize: Tuple[int, int] = (28, 8)  # Wider to accommodate 4 panels
) -> None:
    """
    Create comprehensive analysis including baseline, contextualized similarities, and attention weights.
    
    Args:
        trained_model: The trained model to analyze (optional if trained_model_path provided)
        trained_model_path: Path to the trained model checkpoint (optional if trained_model provided)
        example: Example data containing rows and sentences
        example_idx: Index of the example being analyzed
        output_dir: Directory to save output files
        aggregation_method: Aggregation method to use for model similarities
        base_model_name: Name of the base model (used for dynamic loading)
        figsize: Figure size for the visualization
    """
    # Load trained model if path provided
    if trained_model is None and trained_model_path is not None:
        print(f"🔍 Loading BEST model for comprehensive analysis from: {trained_model_path}")
        # Auto-detect device for comprehensive analysis
        viz_device = "cuda" if torch.cuda.is_available() else "cpu"
        trained_model = load_model(trained_model_path, base_model_name, device=viz_device)
        print(f"✅ BEST trained model loaded and ready for comprehensive analysis on {viz_device}")
    elif trained_model is None:
        raise ValueError("Either trained_model or trained_model_path must be provided")
    
    # Extract data with robust handling of different data formats
    # Extract data using the robust extraction function
    rows, primary_sentences = extract_rows_and_sentences(example, example_idx)
    if rows is None or primary_sentences is None:
        return
    
    print(f"\n🔍 Creating comprehensive analysis for Example {example_idx}...")
    print(f"  Found {len(rows)} rows and {len(primary_sentences)} sentences")
    
    # Detect model type
    model_type = detect_model_type(trained_model)
    is_bidirectional = model_type == "bidirectional"
    
    print("Computing comprehensive similarities...")
    
    # Compute all similarity types using the SAME method as step-by-step diagnostics
    device = next(trained_model.parameters()).device
    
    with torch.no_grad():
        print("  Computing raw embedding similarities...")
        # Use the model's own encoder (same as diagnostics Step 1)
        row_embeddings = trained_model.encode_sentences(rows, normalize=True)
        sentence_embeddings = trained_model.encode_sentences(primary_sentences, normalize=True)
        
        # Get the target dtype from the model's parameters
        if is_bidirectional:
            target_dtype = next(trained_model.bidirectional_attention.parameters()).dtype
        else:
            target_dtype = next(trained_model.parameters()).dtype
        
        # Add batch dimension and cast to target dtype
        rows_tensor = row_embeddings.unsqueeze(0).to(device=device, dtype=target_dtype)
        sentences_tensor = sentence_embeddings.unsqueeze(0).to(device=device, dtype=target_dtype)
        
        # Compute raw similarities (Step 1) - same as diagnostics
        raw_embeddings = _compute_cosine_similarity_matrix(row_embeddings, sentence_embeddings)
        
        if is_bidirectional:
            print("  Computing bidirectional attention and contextualized similarities...")
            # Get diagnostics from model (same as diagnostics)
            try:
                result = trained_model.bidirectional_attention.forward(rows_tensor, sentences_tensor, diagnostics=True)
                if isinstance(result, tuple) and len(result) == 2:
                    pair_scores, diagnostics = result
                    pair_scores = safe_tensor_to_numpy(pair_scores[0])
                    forward_attn = diagnostics['forward_attention']
                    reverse_attn = diagnostics['reverse_attention']
                else:
                    # Fallback
                    pair_scores, _, _, forward_attn, reverse_attn = trained_model.bidirectional_attention(rows_tensor, sentences_tensor)
                    pair_scores = safe_tensor_to_numpy(pair_scores[0])
                    forward_attn = safe_tensor_to_numpy(forward_attn[0])
                    reverse_attn = safe_tensor_to_numpy(reverse_attn[0])
            except:
                # Fallback
                pair_scores, _, _, forward_attn, reverse_attn = trained_model.bidirectional_attention(rows_tensor, sentences_tensor)
                pair_scores = safe_tensor_to_numpy(pair_scores[0])
                forward_attn = safe_tensor_to_numpy(forward_attn[0])
                reverse_attn = safe_tensor_to_numpy(reverse_attn[0])
            
            # Compute contextualized similarities (Step 4 - same as diagnostics)
            print("  Computing contextualized similarities...")
            # Forward contextualization: rows attend to sentences
            # Ensure consistent dtypes - match the model's tensor dtype
            target_dtype = sentences_tensor.dtype
            forward_attn_tensor = torch.tensor(forward_attn, device=device, dtype=target_dtype).unsqueeze(0)
            forward_context = torch.bmm(forward_attn_tensor, sentences_tensor)
            contextualized_rows = rows_tensor + forward_context
            
            # Reverse contextualization: sentences attend to rows  
            reverse_attn_tensor = torch.tensor(reverse_attn, device=device, dtype=target_dtype).unsqueeze(0)
            reverse_context = torch.bmm(reverse_attn_tensor, rows_tensor)
            contextualized_sentences = sentences_tensor + reverse_context
            
            # Compute cosine similarity between contextualized vectors
            contextualized_similarities = _compute_cosine_similarity_matrix(contextualized_rows[0], contextualized_sentences[0])
            
            # Use forward attention for display
            attention_weights = forward_attn
            
        else:
            print("  Computing unidirectional attention and contextualized similarities...")
            # For unidirectional models - handle different attention interfaces
            if hasattr(trained_model, 'attention_type') and trained_model.attention_type != "standard":
                # Advanced attention modules use (queries_emb, keys_emb, values_emb) interface
                context_vectors, attention_weights = trained_model.cross_attention(
                    queries_emb=rows_tensor,
                    keys_emb=sentences_tensor,
                    values_emb=sentences_tensor
                )
            else:
                # Original CrossAttentionModule uses (rows_embeddings, sentences_embeddings) interface
                attention_weights, context_vectors = trained_model.cross_attention(rows_tensor, sentences_tensor)
            
            attention_weights = safe_tensor_to_numpy(attention_weights[0])
            
            # Compute contextualized similarities (after attention, before refinement)
            contextualized_rows = rows_tensor + context_vectors
            contextualized_similarities = _compute_cosine_similarity_matrix(contextualized_rows[0], sentences_tensor[0])
            
            # Compute final model similarities through full forward pass
            pair_scores = np.zeros((len(rows), len(primary_sentences)))
            for i in range(len(rows)):
                for j in range(len(primary_sentences)):
                    single_row = rows_tensor[:, i:i+1, :]
                    single_sentence = sentences_tensor[:, j:j+1, :]
                    score, _ = trained_model(single_row, single_sentence)
                    pair_scores[i, j] = score.item()
    
    # Create visualization
    if is_bidirectional and pair_scores is not None:
        # Enhanced 5-panel visualization for bidirectional models (matching step-by-step diagnostics)
        fig, axes = plt.subplots(1, 5, figsize=(35, 8))
        
        # Panel 1: Raw embeddings (Step 1) - similarities
        cmap1, vmin1, vmax1, center1 = get_consistent_colormap_and_range(raw_embeddings, "similarity")
        sns.heatmap(raw_embeddings, annot=True, fmt='.3f', ax=axes[0],
                    cmap=cmap1, vmin=vmin1, vmax=vmax1, center=center1, cbar_kws={'label': 'Similarity Score'})
        axes[0].set_title("Step 1: Raw Embeddings\n(Frozen Sentence Encoder)")
        axes[0].set_xlabel("Sentences")
        axes[0].set_ylabel("Table Rows")
        
        # Panel 2: Forward attention weights (Step 2) - attention
        cmap2, vmin2, vmax2, center2 = get_consistent_colormap_and_range(attention_weights, "attention")
        sns.heatmap(attention_weights, annot=True, fmt='.3f', ax=axes[1],
                    cmap=cmap2, vmin=vmin2, vmax=vmax2, center=center2, cbar_kws={'label': 'Attention Score'})
        axes[1].set_title("Step 2: Forward Attention\n(Rows → Sentences)")
        axes[1].set_xlabel("Sentences")
        axes[1].set_ylabel("")
        
        # Panel 3: Contextualized similarities (Step 4) - similarities
        cmap3, vmin3, vmax3, center3 = get_consistent_colormap_and_range(contextualized_similarities, "similarity")
        sns.heatmap(contextualized_similarities, annot=True, fmt='.3f', ax=axes[2],
                    cmap=cmap3, vmin=vmin3, vmax=vmax3, center=center3, cbar_kws={'label': 'Similarity Score'})
        axes[2].set_title("Step 4: Contextualized Similarities\n(After Bidirectional Attention)")
        axes[2].set_xlabel("Sentences")
        axes[2].set_ylabel("")
        
        # Panel 4: Final pair scores (Step 6) - similarities
        cmap4, vmin4, vmax4, center4 = get_consistent_colormap_and_range(pair_scores, "similarity")
        sns.heatmap(pair_scores, annot=True, fmt='.3f', ax=axes[3],
                    cmap=cmap4, vmin=vmin4, vmax=vmax4, center=center4, cbar_kws={'label': 'Similarity Score'})
        method_name = getattr(trained_model, 'pair_score_method', 'cosine')
        axes[3].set_title(f"Step 6: Final Pair Scores\n({method_name} similarity)")
        axes[3].set_xlabel("Sentences")
        axes[3].set_ylabel("")
        
        # Panel 5: Learning effect (final vs raw)
        difference = pair_scores - raw_embeddings
        cmap5, vmin5, vmax5, center5 = get_consistent_colormap_and_range(difference, "difference")
        sns.heatmap(difference, annot=True, fmt='.3f', ax=axes[4],
                    cmap=cmap5, vmin=vmin5, vmax=vmax5, center=center5,
                    cbar_kws={'label': 'Difference'})
        axes[4].set_title("Learning Effect\n(Final - Raw)")
        axes[4].set_xlabel("Sentences")
        axes[4].set_ylabel("")
        
        plt.suptitle(f"Comprehensive Bidirectional Analysis - Example {example_idx}\n"
                    f"Method: {method_name} | Matching Step-by-Step Diagnostics", 
                    fontsize=16)
        
    else:
        # Enhanced 5-panel visualization for unidirectional models (matching step-by-step diagnostics)
        fig, axes = plt.subplots(1, 5, figsize=(35, 8))
        
        # Panel 1: Raw embeddings (Step 1) - similarities
        cmap1, vmin1, vmax1, center1 = get_consistent_colormap_and_range(raw_embeddings, "similarity")
        sns.heatmap(raw_embeddings, annot=True, fmt='.3f', ax=axes[0],
                    cmap=cmap1, vmin=vmin1, vmax=vmax1, center=center1, cbar_kws={'label': 'Similarity Score'})
        axes[0].set_title("Step 1: Raw Embeddings\n(Model's Own Encoder)")
        axes[0].set_xlabel("Sentences")
        axes[0].set_ylabel("Table Rows")
        
        # Panel 2: Cross-attention weights (Step 2) - attention
        cmap2, vmin2, vmax2, center2 = get_consistent_colormap_and_range(attention_weights, "attention")
        sns.heatmap(attention_weights, annot=True, fmt='.3f', ax=axes[1],
                    cmap=cmap2, vmin=vmin2, vmax=vmax2, center=center2, cbar_kws={'label': 'Attention Score'})
        axes[1].set_title("Step 2: Cross-Attention Weights\n(Rows → Sentences)")
        axes[1].set_xlabel("Sentences")
        axes[1].set_ylabel("")
        
        # Panel 3: Contextualized similarities (Step 3) - similarities
        cmap3, vmin3, vmax3, center3 = get_consistent_colormap_and_range(contextualized_similarities, "similarity")
        sns.heatmap(contextualized_similarities, annot=True, fmt='.3f', ax=axes[2],
                    cmap=cmap3, vmin=vmin3, vmax=vmax3, center=center3, cbar_kws={'label': 'Similarity Score'})
        axes[2].set_title("Step 3: Contextualized Similarities\n(After Cross-Attention)")
        axes[2].set_xlabel("Sentences")
        axes[2].set_ylabel("")
        
        # Panel 4: Final model similarities (Step 5) - similarities
        cmap4, vmin4, vmax4, center4 = get_consistent_colormap_and_range(pair_scores, "similarity")
        sns.heatmap(pair_scores, annot=True, fmt='.3f', ax=axes[3],
                    cmap=cmap4, vmin=vmin4, vmax=vmax4, center=center4, cbar_kws={'label': 'Similarity Score'})
        axes[3].set_title("Step 5: Final Model Similarities\n(Full Forward Pass)")
        axes[3].set_xlabel("Sentences")
        axes[3].set_ylabel("")
        
        # Panel 5: Learning effect (final vs raw)
        difference = pair_scores - raw_embeddings
        cmap5, vmin5, vmax5, center5 = get_consistent_colormap_and_range(difference, "difference")
        sns.heatmap(difference, annot=True, fmt='.3f', ax=axes[4],
                    cmap=cmap5, vmin=vmin5, vmax=vmax5, center=center5,
                    cbar_kws={'label': 'Difference'})
        axes[4].set_title("Learning Effect\n(Final - Raw)")
        axes[4].set_xlabel("Sentences")
        axes[4].set_ylabel("")
        
        plt.suptitle(f"Comprehensive Unidirectional Analysis - Example {example_idx}\n"
                    f"Matching Step-by-Step Diagnostics", 
                    fontsize=16)
    
    plt.tight_layout()
    
    # Save visualization
    suffix = "bidirectional" if is_bidirectional else "unidirectional"
    # Save to visualizations subdirectory
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_file = viz_dir / f"comprehensive_analysis_example_{example_idx}.png"
    save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved comprehensive analysis to {output_file}")
    plt.close()
    
    # Print detailed analysis summary (like the user expects)
    print(f"\n📊 Analysis Summary for Example {example_idx}:")
    
    # Raw embeddings statistics
    raw_min, raw_max = raw_embeddings.min(), raw_embeddings.max()
    print(f"  Raw Embeddings - Range: [{raw_min:.3f}, {raw_max:.3f}]")
    
    # Contextualized similarities statistics
    ctx_min, ctx_max = contextualized_similarities.min(), contextualized_similarities.max()
    print(f"  Contextualized Similarities - Range: [{ctx_min:.3f}, {ctx_max:.3f}]")
    
    # Final model similarities statistics
    pair_min, pair_max = pair_scores.min(), pair_scores.max()
    print(f"  Final Model Similarities - Range: [{pair_min:.3f}, {pair_max:.3f}]")
    
    # Attention weights statistics
    attn_min, attn_max = attention_weights.min(), attention_weights.max()
    print(f"  Cross-Attention - Range: [{attn_min:.3f}, {attn_max:.3f}]")
    
    # Top 3 pairs analysis
    def get_top_pairs(matrix, name):
        flat_indices = np.argsort(matrix.flatten())[::-1][:3]
        pairs = []
        for flat_idx in flat_indices:
            row_idx = flat_idx // matrix.shape[1]
            sent_idx = flat_idx % matrix.shape[1]
            score = matrix[row_idx, sent_idx]
            pairs.append((row_idx + 1, sent_idx + 1, score))
        return pairs
    
    print(f"\n🔝 Top 3 pairs by Raw Embeddings:")
    for i, (row_idx, sent_idx, score) in enumerate(get_top_pairs(raw_embeddings, "Raw")):
        print(f"  {i+1}. Row {row_idx} - Sentence {sent_idx}: {score:.3f}")
    
    print(f"\n🔝 Top 3 pairs by Contextualized Similarities:")
    for i, (row_idx, sent_idx, score) in enumerate(get_top_pairs(contextualized_similarities, "Contextualized")):
        print(f"  {i+1}. Row {row_idx} - Sentence {sent_idx}: {score:.3f}")
    
    print(f"\n🔝 Top 3 pairs by Final Model Similarities:")
    for i, (row_idx, sent_idx, score) in enumerate(get_top_pairs(pair_scores, "Final")):
        print(f"  {i+1}. Row {row_idx} - Sentence {sent_idx}: {score:.3f}")
    
    print(f"\n🔝 Top 3 pairs by Cross-Attention:")
    for i, (row_idx, sent_idx, score) in enumerate(get_top_pairs(attention_weights, "Attention")):
        print(f"  {i+1}. Row {row_idx} - Sentence {sent_idx}: {score:.3f}")
    
    # Save detailed text analysis (enhanced for bidirectional)
    # Save text analysis to visualizations subdirectory
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    text_file = viz_dir / f"example_{example_idx}_similarity_analysis.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Comprehensive Similarity Analysis - Example {example_idx} ===\n")
        f.write(f"Model Type: {'Bidirectional' if is_bidirectional else 'Unidirectional'}\n")
        f.write(f"Aggregation Method: {aggregation_method}\n")
        if is_bidirectional:
            method_name = getattr(trained_model, 'pair_score_method', 'cosine')
            f.write(f"Pair Score Method: {method_name}\n")
        f.write("\n")
        
        f.write("TABLE ROWS:\n")
        for i, row in enumerate(rows):
            f.write(f"Row {i+1}: {row}\n")
        
        f.write(f"\nCONTEXT SENTENCES:\n")
        for i, sent in enumerate(primary_sentences):
            f.write(f"Sent {i+1}: {sent}\n")
        
        f.write(f"\n=== SIMILARITY MATRICES ===\n")
        
        f.write(f"\nRAW EMBEDDINGS (Sentence Encoder):\n")
        f.write(f"{raw_embeddings}\n")
        
        f.write(f"\nCONTEXTUALIZED SIMILARITIES (After Attention):\n")
        f.write(f"{contextualized_similarities}\n")
        
        f.write(f"\nFINAL MODEL SIMILARITIES (After Training):\n")
        f.write(f"{pair_scores}\n")
        
        f.write(f"\nCROSS-ATTENTION WEIGHTS:\n")
        f.write(f"{attention_weights}\n")
        
        f.write(f"\nLEARNING EFFECT (Final - Raw):\n")
        difference = pair_scores - raw_embeddings
        f.write(f"{difference}\n")
    
    print(f"📄 Detailed analysis saved to {text_file}")
    print()

def process_example_comprehensive_analysis(
    untrained_model: TableTextEmbeddingModel,
    trained_model: TableTextEmbeddingModel,
    example: Dict[str, Any],
    example_idx: int,
    output_dir: str,
    untrained_cache: Optional[IdBasedEmbeddingCache] = None,
    trained_cache: Optional[IdBasedEmbeddingCache] = None,
    save_visualizations: bool = True
) -> None:
    """
    Process a single example and generate comprehensive analysis visualizations.
    
    Args:
        untrained_model: Untrained model
        trained_model: Trained model
        example: Example to process
        example_idx: Index of the example
        output_dir: Directory to save visualizations
        untrained_cache: Cache for untrained model
        trained_cache: Cache for trained model
        save_visualizations: Whether to save visualizations
    """
    # Create output directory
    output_path = Path(output_dir) / f"example_{example_idx}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get anchor rows and primary sentences with robust extraction
    rows, primary_sentences = extract_rows_and_sentences(example, example_idx)
    if rows is None or primary_sentences is None:
        return
    
    print(f"\nProcessing Example {example_idx}")
    print(f"Number of Rows: {len(rows)}")
    
    # Process primary positive context
    print(f"Primary Positive Context: {len(primary_sentences)} sentences")
    
    if primary_sentences:
        # Compute attention matrices
        untrained_attention = compute_attention_matrix(untrained_model, rows, primary_sentences, untrained_cache, normalize=False)
        trained_attention = compute_attention_matrix(trained_model, rows, primary_sentences, trained_cache, normalize=False)
        
        if save_visualizations:
            # Visualize comparison
            matrix_file = output_path / "primary_attention_comparison.pdf"
            compare_attention_matrices(
                untrained_attention,
                trained_attention,
                rows, 
                primary_sentences,
                title=f"Example {example_idx}: Primary Positive Attention Comparison",
                output_file=matrix_file,
                show_values=True
            )
            
            # Individual visualizations
            untrained_matrix_file = output_path / "primary_attention_untrained.pdf"
            trained_matrix_file = output_path / "primary_attention_trained.pdf"
            
            visualize_attention_matrix(
                untrained_attention, 
                rows, 
                primary_sentences,
                title=f"Example {example_idx}: Primary Positive Attention (Untrained)",
                output_file=untrained_matrix_file,
                show_values=True
            )
            
            visualize_attention_matrix(
                trained_attention, 
                rows, 
                primary_sentences,
                title=f"Example {example_idx}: Primary Positive Attention (Trained)",
                output_file=trained_matrix_file,
                show_values=True
            )
        
        # Get top pairs
        untrained_top_pairs = get_top_k_pairs(untrained_attention, rows, primary_sentences, k=5)
        trained_top_pairs = get_top_k_pairs(trained_attention, rows, primary_sentences, k=5)
        
        if save_visualizations:
            # Compare top pairs
            pairs_file = output_path / "primary_top_pairs_comparison.pdf"
            compare_top_pairs(
                untrained_top_pairs,
                trained_top_pairs,
                title=f"Example {example_idx}: Top-5 Primary Positive Pairs Comparison",
                output_file=pairs_file
            )
            
            # Individual visualizations
            untrained_pairs_file = output_path / "primary_top_pairs_untrained.pdf"
            trained_pairs_file = output_path / "primary_top_pairs_trained.pdf"
            
            visualize_top_pairs(
                untrained_top_pairs,
                title=f"Example {example_idx}: Top-5 Primary Positive Pairs (Untrained)",
                output_file=untrained_pairs_file
            )
            
            visualize_top_pairs(
                trained_top_pairs,
                title=f"Example {example_idx}: Top-5 Primary Positive Pairs (Trained)",
                output_file=trained_pairs_file
            )
        
        # Save top pair details
        untrained_details_file = output_path / "primary_top_pairs_untrained.txt"
        trained_details_file = output_path / "primary_top_pairs_trained.txt"
        
        with open(untrained_details_file, 'w', encoding='utf-8') as f:
            f.write(f"Untrained Model - Top-5 Pairs by Attention Score:\n")
            f.write("="*50 + "\n\n")
            
            for i, pair in enumerate(untrained_top_pairs):
                f.write(f"Pair {i+1} (Score: {pair['score']:.6f}):\n")
                f.write(f"  Row: {pair['row']}\n")
                f.write(f"  Sentence: {pair['sentence']}\n\n")
        
        with open(trained_details_file, 'w', encoding='utf-8') as f:
            f.write(f"Trained Model - Top-5 Pairs by Attention Score:\n")
            f.write("="*50 + "\n\n")
            
            for i, pair in enumerate(trained_top_pairs):
                f.write(f"Pair {i+1} (Score: {pair['score']:.6f}):\n")
                f.write(f"  Row: {pair['row']}\n")
                f.write(f"  Sentence: {pair['sentence']}\n\n")
    
    # Process negative contexts (first negative only for brevity)
    if example["negatives"]:
        negative = example["negatives"][0]
        negative_sentences = negative["sentences"]
        
        if negative_sentences:
            print(f"\nNegative Context: {len(negative_sentences)} sentences")
            
            # Compute attention matrices
            untrained_attention = compute_attention_matrix(untrained_model, rows, negative_sentences, untrained_cache, normalize=False)
            trained_attention = compute_attention_matrix(trained_model, rows, negative_sentences, trained_cache, normalize=False)
            
            if save_visualizations:
                # Visualize comparison
                matrix_file = output_path / "negative_attention_comparison.pdf"
                compare_attention_matrices(
                    untrained_attention,
                    trained_attention,
                    rows, 
                    negative_sentences,
                    title=f"Example {example_idx}: Negative Attention Comparison",
                    output_file=matrix_file,
                    show_values=True
                )
            
            # Get top pairs
            untrained_top_pairs = get_top_k_pairs(untrained_attention, rows, negative_sentences, k=5)
            trained_top_pairs = get_top_k_pairs(trained_attention, rows, negative_sentences, k=5)
            
            if save_visualizations:
                # Compare top pairs
                pairs_file = output_path / "negative_top_pairs_comparison.pdf"
                compare_top_pairs(
                    untrained_top_pairs,
                    trained_top_pairs,
                    title=f"Example {example_idx}: Top-5 Negative Pairs Comparison",
                    output_file=pairs_file
                )
            
            # Save top pair details
            untrained_details_file = output_path / "negative_top_pairs_untrained.txt"
            trained_details_file = output_path / "negative_top_pairs_trained.txt"
            
            with open(untrained_details_file, 'w', encoding='utf-8') as f:
                f.write(f"Untrained Model - Top-5 Pairs by Attention Score:\n")
                f.write("="*50 + "\n\n")
                
                for i, pair in enumerate(untrained_top_pairs):
                    f.write(f"Pair {i+1} (Score: {pair['score']:.6f}):\n")
                    f.write(f"  Row: {pair['row']}\n")
                    f.write(f"  Sentence: {pair['sentence']}\n\n")
            
            with open(trained_details_file, 'w', encoding='utf-8') as f:
                f.write(f"Trained Model - Top-5 Pairs by Attention Score:\n")
                f.write("="*50 + "\n\n")
                
                for i, pair in enumerate(trained_top_pairs):
                    f.write(f"Pair {i+1} (Score: {pair['score']:.6f}):\n")
                    f.write(f"  Row: {pair['row']}\n")
                    f.write(f"  Sentence: {pair['sentence']}\n\n")
    
    # Generate comprehensive analysis visualization
    visualize_comprehensive_model_analysis(
        trained_model,
        example,
        example_idx,
        output_dir
    )

def compute_contextualized_pairwise_similarities(model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel], 
                                               rows: List[str], 
                                               sentences: List[str],
                                               method: str = "final_cosine") -> np.ndarray:
    """
    Compute pairwise similarities between contextualized rows and sentences.
    This shows how much each sentence contributes to each row's final representation.
    
    Args:
        model: The trained model (unidirectional or bidirectional)
        rows: List of table row texts
        sentences: List of sentence texts
        method: Method to compute similarities
            - "final_cosine": Cosine similarity between final row embeddings and original sentences
            - "bidirectional_pairs": Direct pair scores from bidirectional model
            - "attention_weighted": Attention weights as direct contribution measure
            - "decomposed_contribution": Decompose context vectors into sentence contributions
            - "gradient_attribution": Gradient-based attribution of sentences to rows
            
    Returns:
        Matrix of shape [num_rows, num_sentences] showing pairwise similarities
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Detect model type
    model_type = detect_model_type(model)
    
    with torch.no_grad():
        # Encode inputs
        rows_embeddings = model.encode_sentences(rows, batch_size=8, normalize=True)
        sentences_embeddings = model.encode_sentences(sentences, batch_size=8, normalize=True)
        
        # Get the target dtype from the model's parameters
        target_dtype = next(model.parameters()).dtype
        
        # Add batch dimension and cast to target dtype
        rows_tensor = rows_embeddings.unsqueeze(0).to(device=device, dtype=target_dtype)  # [1, num_rows, embedding_dim]
        sentences_tensor = sentences_embeddings.unsqueeze(0).to(device=device, dtype=target_dtype)  # [1, num_sentences, embedding_dim]
        
        if method == "bidirectional_pairs":
            if model_type == "bidirectional":
                # Get direct pair scores from bidirectional model
                pair_scores, _, _, _, _ = model.bidirectional_attention(rows_tensor, sentences_tensor)
                return pair_scores[0].float().cpu().numpy()
            else:
                # Fallback to final_cosine for unidirectional models
                method = "final_cosine"
        
        if method == "final_cosine":
            if model_type == "bidirectional":
                # Get pair scores and attention weights from bidirectional model
                pair_scores, forward_cross_attn, reverse_cross_attn, forward_attn, reverse_attn = model.bidirectional_attention(
                    rows_tensor, sentences_tensor
                )
                # Use pair scores as similarities for bidirectional models
                return pair_scores[0].float().cpu().numpy()
            else:
                # For unidirectional models, compute pairwise similarities using full model forward pass
                batch_size, num_rows, embedding_dim = rows_tensor.shape
                _, num_sentences, _ = sentences_tensor.shape
                
                similarities = torch.zeros(num_rows, num_sentences, device=device)
                
                # Compute similarity for each row-sentence pair using the full model
                for i in range(num_rows):
                    for j in range(num_sentences):
                        # Create single-row and single-sentence tensors
                        single_row = rows_tensor[:, i:i+1, :]  # [1, 1, embedding_dim]
                        single_sentence = sentences_tensor[:, j:j+1, :]  # [1, 1, embedding_dim]
                        
                        # Get similarity through full model forward pass
                        similarity_score, _ = model(single_row, single_sentence)
                        similarities[i, j] = similarity_score.item()
                
                return similarities.float().cpu().numpy()
            
        elif method == "attention_weighted":
            if model_type == "bidirectional":
                # For bidirectional models, use forward attention weights
                _, _, _, forward_attn, _ = model.bidirectional_attention(rows_tensor, sentences_tensor)
                return forward_attn[0].float().cpu().numpy()
            else:
                # Use attention weights as direct contribution measure - handle different interfaces
                if hasattr(model, 'attention_type') and model.attention_type != "standard":
                    # Advanced attention modules use (queries_emb, keys_emb, values_emb) interface
                    _, attention_weights = model.cross_attention(
                        queries_emb=rows_tensor,
                        keys_emb=sentences_tensor,
                        values_emb=sentences_tensor
                    )
                else:
                    # Original CrossAttentionModule uses (rows_embeddings, sentences_embeddings) interface
                    attention_weights, _ = model.cross_attention(rows_tensor, sentences_tensor)
                return attention_weights[0].float().cpu().numpy()
            
        elif method == "decomposed_contribution":
            if model_type == "bidirectional":
                # For bidirectional models, use pair scores as contribution measure
                pair_scores, _, _, _, _ = model.bidirectional_attention(rows_tensor, sentences_tensor)
                return pair_scores[0].float().cpu().numpy()
            else:
                # Decompose context vectors into sentence contributions - handle different interfaces
                if hasattr(model, 'attention_type') and model.attention_type != "standard":
                    # Advanced attention modules use (queries_emb, keys_emb, values_emb) interface
                    context_vectors, attention_weights = model.cross_attention(
                        queries_emb=rows_tensor,
                        keys_emb=sentences_tensor,
                        values_emb=sentences_tensor
                    )
                else:
                    # Original CrossAttentionModule uses (rows_embeddings, sentences_embeddings) interface
                    attention_weights, context_vectors = model.cross_attention(rows_tensor, sentences_tensor)
                
                # context_vectors = attention_weights @ sentences_tensor
                # We can compute how much each sentence contributes to each context vector
                contributions = torch.zeros(1, rows_tensor.size(1), sentences_tensor.size(1), device=device)
                
                for row_idx in range(rows_tensor.size(1)):
                    for sent_idx in range(sentences_tensor.size(1)):
                        # Contribution = attention_weight * cosine_similarity(context_vector, sentence)
                        attention_weight = attention_weights[0, row_idx, sent_idx]
                        context_vec = context_vectors[0, row_idx]  # [embedding_dim]
                        sentence_vec = sentences_tensor[0, sent_idx]  # [embedding_dim]
                        
                        # Cosine similarity between final context and original sentence
                        cosine_sim = torch.cosine_similarity(context_vec, sentence_vec, dim=0)
                        
                        # Weight by attention to get contribution score
                        contribution = attention_weight * cosine_sim
                        contributions[0, row_idx, sent_idx] = contribution
                
                return contributions[0].float().cpu().numpy()
            
        elif method == "gradient_attribution":
            # This requires gradients - more complex implementation
            return compute_gradient_attribution(model, rows_tensor, sentences_tensor)
            
        else:
            raise ValueError(f"Unknown method: {method}")

def compute_gradient_attribution(model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel], 
                               rows_tensor: torch.Tensor,
                               sentences_tensor: torch.Tensor) -> np.ndarray:
    """
    Use gradients to measure how much each sentence affects each row's final representation.
    """
    model.eval()
    attribution_matrix = torch.zeros(rows_tensor.size(1), sentences_tensor.size(1))
    
    # Detect model type
    model_type = detect_model_type(model)
    
    # Make sentences_tensor require gradients
    sentences_tensor.requires_grad_(True)
    
    for row_idx in range(rows_tensor.size(1)):
        if model_type == "bidirectional":
            # For bidirectional models, use pair scores
            pair_scores, _, _, _, _ = model.bidirectional_attention(rows_tensor, sentences_tensor)
            # Focus on this specific row's pair scores
            row_scores = pair_scores[0, row_idx]  # [num_sentences]
            
            # Compute gradient of row scores w.r.t. each sentence
            for sent_idx in range(sentences_tensor.size(1)):
                if sentences_tensor.grad is not None:
                    sentences_tensor.grad.zero_()
                    
                # Compute how much this sentence affects this row's score
                score = row_scores[sent_idx]
                score.backward(retain_graph=True)
                
                # Get gradient for this sentence
                if sentences_tensor.grad is not None:
                    sentence_grad = sentences_tensor.grad[0, sent_idx].norm().item()
                    attribution_matrix[row_idx, sent_idx] = sentence_grad
        else:
            # Get the final embedding for this specific row (unidirectional model) - handle different interfaces
            if hasattr(model, 'attention_type') and model.attention_type != "standard":
                # Advanced attention modules use (queries_emb, keys_emb, values_emb) interface
                context_vectors, attention_weights = model.cross_attention(
                    queries_emb=rows_tensor,
                    keys_emb=sentences_tensor,
                    values_emb=sentences_tensor
                )
            else:
                # Original CrossAttentionModule uses (rows_embeddings, sentences_embeddings) interface
                attention_weights, context_vectors = model.cross_attention(rows_tensor, sentences_tensor)
            norm_context = model.norm1(context_vectors + rows_tensor)
            ff_output = model.feed_forward(norm_context)
            final_context = model.norm2(ff_output + norm_context)
            
            # Focus on this specific row's final representation
            row_embedding = final_context[0, row_idx]  # [embedding_dim]
            
            # Compute gradient of row embedding norm w.r.t. each sentence
            for sent_idx in range(sentences_tensor.size(1)):
                if sentences_tensor.grad is not None:
                    sentences_tensor.grad.zero_()
                    
                # Compute how much this sentence affects this row's final embedding
                row_norm = torch.norm(row_embedding)
                row_norm.backward(retain_graph=True)
                
                # Get gradient for this sentence
                if sentences_tensor.grad is not None:
                    sentence_grad = sentences_tensor.grad[0, sent_idx].norm().item()
                    attribution_matrix[row_idx, sent_idx] = sentence_grad
    
    return attribution_matrix.numpy()

def visualize_contextualized_similarities(rows: List[str], 
                                        sentences: List[str],
                                        similarities: np.ndarray,
                                        method: str = "final_cosine",
                                        title: str = "Contextualized Row-Sentence Similarities",
                                        output_file: Optional[str] = None,
                                        figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Visualize contextualized similarities between rows and sentences.
    """
    plt.figure(figsize=figsize)
    
    # Create simplified labels
    row_labels = [f"Row {i+1}" for i in range(len(rows))]
    sentence_labels = [f"Sent {i+1}" for i in range(len(sentences))]
    
    # Get consistent colormap and range
    cmap, vmin, vmax, center = get_consistent_colormap_and_range(similarities, 'similarity')
    
    # Create heatmap
    sns.heatmap(similarities,
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                cbar_kws={'label': f'Similarity Score ({method})'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sentences', fontsize=12)
    plt.ylabel('Rows', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_file:
        save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved contextualized similarities to {output_file}")
    else:
        plt.show()
    
    plt.close()

def compare_baseline_vs_contextualized(model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel],
                                     rows: List[str], 
                                     sentences: List[str],
                                     base_model_name: str = "answerdotai/ModernBERT-base",
                                     title: str = "Baseline vs Contextualized Similarities",
                                     output_file: Optional[str] = None,
                                     figsize: Tuple[int, int] = (20, 8)) -> Dict[str, np.ndarray]:
    """
    Compare baseline transformer similarities with contextualized similarities.
    """
    # Compute similarities
    baseline_similarities = compute_raw_transformer_similarities(model, rows, sentences, base_model_name)
    contextualized_similarities = compute_contextualized_pairwise_similarities(model, rows, sentences)
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Create simplified labels
    row_labels = [f"Row {i+1}" for i in range(len(rows))]
    sentence_labels = [f"Sent {i+1}" for i in range(len(sentences))]
    
    # Get consistent range across both matrices
    combined_data = np.concatenate([baseline_similarities.flatten(), contextualized_similarities.flatten()])
    cmap, vmin, vmax, center = get_consistent_colormap_and_range(combined_data, 'similarity')
    
    # Baseline similarities
    sns.heatmap(baseline_similarities,
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                ax=ax1,
                cbar=False)
    ax1.set_title("Baseline Transformer\nSimilarities", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sentences')
    ax1.set_ylabel('Rows')
    
    # Contextualized similarities
    sns.heatmap(contextualized_similarities,
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                ax=ax2,
                cbar_kws={'label': 'Similarity Score'})
    ax2.set_title("Contextualized\nSimilarities", fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sentences')
    ax2.set_ylabel('')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved baseline vs contextualized comparison to {output_file}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'baseline_similarities': baseline_similarities,
        'contextualized_similarities': contextualized_similarities
    }

def visualize_bidirectional_attention(pair_scores: np.ndarray,
                                     forward_attention: np.ndarray,
                                     reverse_attention: np.ndarray,
                                     rows: List[str],
                                     sentences: List[str],
                                     title: str = "Bidirectional Cross-Attention Analysis",
                                     output_file: Optional[str] = None,
                                     figsize: Tuple[int, int] = (24, 8)) -> None:
    """
    Create a comprehensive visualization of bidirectional attention mechanisms.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Create simplified labels
    row_labels = [f"Row {i+1}" for i in range(len(rows))]
    sentence_labels = [f"Sent {i+1}" for i in range(len(sentences))]
    
    # Panel 1: Pair Scores
    cmap1, vmin1, vmax1, center1 = get_consistent_colormap_and_range(pair_scores, 'similarity')
    sns.heatmap(pair_scores,
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=True,
                fmt='.3f',
                cmap=cmap1,
                vmin=vmin1,
                vmax=vmax1,
                center=center1,
                ax=axes[0],
                cbar_kws={'label': 'Pair Score'})
    axes[0].set_title("Final Pair Scores\n(Direct Similarity)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Sentences')
    axes[0].set_ylabel('Rows')
    
    # Panel 2: Forward Attention (rows → sentences)
    cmap2, vmin2, vmax2, center2 = get_consistent_colormap_and_range(forward_attention, 'attention')
    sns.heatmap(forward_attention,
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=True,
                fmt='.3f',
                cmap=cmap2,
                vmin=vmin2,
                vmax=vmax2,
                center=center2,
                ax=axes[1],
                cbar_kws={'label': 'Attention Weight'})
    axes[1].set_title("Forward Attention\n(Rows → Sentences)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sentences')
    axes[1].set_ylabel('')
    
    # Panel 3: Reverse Attention (sentences → rows)
    # Note: reverse_attention is [M, N], so we transpose for consistent visualization
    reverse_attention_viz = reverse_attention.T if reverse_attention.shape[0] != len(rows) else reverse_attention
    cmap3, vmin3, vmax3, center3 = get_consistent_colormap_and_range(reverse_attention_viz, 'attention')
    sns.heatmap(reverse_attention_viz,
                xticklabels=sentence_labels,
                yticklabels=row_labels,
                annot=True,
                fmt='.3f',
                cmap=cmap3,
                vmin=vmin3,
                vmax=vmax3,
                center=center3,
                ax=axes[2],
                cbar_kws={'label': 'Attention Weight'})
    axes[2].set_title("Reverse Attention\n(Sentences → Rows)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Sentences')
    axes[2].set_ylabel('')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_file:
        save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved bidirectional attention visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()

def compare_bidirectional_models(untrained_model: BidirectionalTableTextModel,
                               trained_model: BidirectionalTableTextModel,
                               rows: List[str],
                               sentences: List[str],
                               title: str = "Bidirectional Model Comparison",
                               output_file: Optional[str] = None,
                               figsize: Tuple[int, int] = (24, 16)) -> None:
    """
    Compare untrained vs trained bidirectional models.
    
    Args:
        untrained_model: Untrained bidirectional model
        trained_model: Trained bidirectional model
        rows: List of row texts
        sentences: List of sentence texts
        title: Overall title
        output_file: Path to save visualization
        figsize: Figure size
    """
    # Get results from both models
    untrained_results = compute_attention_matrix(
        untrained_model, rows, sentences, return_type="both"
    )
    trained_results = compute_attention_matrix(
        trained_model, rows, sentences, return_type="both"
    )
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Prepare labels
    row_labels = [f"Row {i+1}" for i in range(len(rows))]
    sentence_labels = [f"Sent {i+1}" for i in range(len(sentences))]
    
    # Get consistent ranges for each type of matrix
    all_pair_scores = np.concatenate([
        untrained_results['pair_scores'].flatten(), 
        trained_results['pair_scores'].flatten()
    ])
    all_forward_attn = np.concatenate([
        untrained_results['forward_attention'].flatten(), 
        trained_results['forward_attention'].flatten()
    ])
    all_reverse_attn = np.concatenate([
        untrained_results['reverse_attention'].flatten(), 
        trained_results['reverse_attention'].flatten()
    ])
    
    pair_cmap, pair_vmin, pair_vmax, pair_center = get_consistent_colormap_and_range(all_pair_scores, 'similarity')
    attn_cmap, attn_vmin, attn_vmax, attn_center = get_consistent_colormap_and_range(all_forward_attn, 'attention')
    
    # Row 1: Untrained model
    sns.heatmap(untrained_results['pair_scores'], annot=True, fmt=".3f", cmap=pair_cmap,
                xticklabels=sentence_labels, yticklabels=row_labels, ax=axes[0,0],
                vmin=pair_vmin, vmax=pair_vmax, center=pair_center, cbar=False)
    axes[0,0].set_title("Untrained: Pair Scores", fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel("Table Rows")
    
    sns.heatmap(untrained_results['forward_attention'], annot=True, fmt=".3f", cmap=attn_cmap,
                xticklabels=sentence_labels, yticklabels=row_labels, ax=axes[0,1],
                vmin=attn_vmin, vmax=attn_vmax, center=attn_center, cbar=False)
    axes[0,1].set_title("Untrained: Forward Attention", fontsize=12, fontweight='bold')
    
    sns.heatmap(untrained_results['reverse_attention'], annot=True, fmt=".3f", cmap=attn_cmap,
                xticklabels=row_labels, yticklabels=sentence_labels, ax=axes[0,2],
                vmin=attn_vmin, vmax=attn_vmax, center=attn_center, cbar=False)
    axes[0,2].set_title("Untrained: Reverse Attention", fontsize=12, fontweight='bold')
    
    # Row 2: Trained model  
    sns.heatmap(trained_results['pair_scores'], annot=True, fmt=".3f", cmap=pair_cmap,
                xticklabels=sentence_labels, yticklabels=row_labels, ax=axes[1,0],
                vmin=pair_vmin, vmax=pair_vmax, center=pair_center, 
                cbar_kws={"label": "Similarity Score"})
    axes[1,0].set_title("Trained: Pair Scores", fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel("Sentences")
    axes[1,0].set_ylabel("Table Rows")
    
    sns.heatmap(trained_results['forward_attention'], annot=True, fmt=".3f", cmap=attn_cmap,
                xticklabels=sentence_labels, yticklabels=row_labels, ax=axes[1,1],
                vmin=attn_vmin, vmax=attn_vmax, center=attn_center, 
                cbar_kws={"label": "Attention Score"})
    axes[1,1].set_title("Trained: Forward Attention", fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel("Sentences")
    
    sns.heatmap(trained_results['reverse_attention'], annot=True, fmt=".3f", cmap=attn_cmap,
                xticklabels=row_labels, yticklabels=sentence_labels, ax=axes[1,2],
                vmin=attn_vmin, vmax=attn_vmax, center=attn_center, 
                cbar_kws={"label": "Attention Score"})
    axes[1,2].set_title("Trained: Reverse Attention", fontsize=12, fontweight='bold')
    axes[1,2].set_xlabel("Table Rows")
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved bidirectional comparison to {output_file}")
    
    plt.close()

def visualize_join_paths(pair_scores: np.ndarray,
                        rows: List[str],
                        sentences: List[str],
                        join_paths: List[Tuple[int, int, float]],
                        title: str = "Join Path Discovery",
                        output_file: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize the pair scores matrix with join paths highlighted.
    
    Args:
        pair_scores: Pair score matrix [num_rows, num_sentences]
        rows: List of row texts
        sentences: List of sentence texts  
        join_paths: List of (row_idx, sentence_idx, score) tuples
        title: Plot title
        output_file: Path to save visualization
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create base heatmap
    row_labels = [f"Row {i+1}" for i in range(len(rows))]
    sentence_labels = [f"Sent {i+1}" for i in range(len(sentences))]
    
    # Get consistent colormap and range
    cmap, vmin, vmax, center = get_consistent_colormap_and_range(pair_scores, 'similarity')
    
    sns.heatmap(
        pair_scores,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        xticklabels=sentence_labels,
        yticklabels=row_labels,
        ax=ax,
        cbar_kws={"label": "Similarity Score"}
    )
    
    # Highlight join paths
    for path_idx, (row_idx, sent_idx, score) in enumerate(join_paths):
        # Add a border around the cell
        ax.add_patch(plt.Rectangle((sent_idx, row_idx), 1, 1, 
                                  fill=False, edgecolor='red', lw=3))
        # Add path number
        ax.text(sent_idx + 0.5, row_idx + 0.8, f"#{path_idx+1}", 
                ha='center', va='center', color='red', fontweight='bold', fontsize=10)
    
    ax.set_title(f"{title}\n{len(join_paths)} join paths extracted")
    ax.set_xlabel("Sentences")
    ax.set_ylabel("Table Rows")
    
    # Add legend
    legend_text = "\n".join([
        f"Path {i+1}: Row {row_idx+1} ↔ Sent {sent_idx+1} (score: {score:.3f})"
        for i, (row_idx, sent_idx, score) in enumerate(join_paths[:5])  # Show top 5
    ])
    if len(join_paths) > 5:
        legend_text += f"\n... and {len(join_paths) - 5} more paths"
    
    ax.text(1.02, 0.5, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
    
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved join path visualization to {output_file}")
    
    plt.close()

def process_bidirectional_example(model: BidirectionalTableTextModel,
                                 example: Dict[str, Any],
                                 example_idx: int,
                                 output_dir: str,
                                 model_type: str = "trained") -> None:
    """
    Process a single example with a bidirectional model and create visualizations.
    
    Args:
        model: Bidirectional model to analyze
        example: Dataset example
        example_idx: Index of the example
        output_dir: Directory to save outputs
        model_type: "trained" or "untrained" for file naming
    """
    # Extract data with robust handling
    rows, sentences = extract_rows_and_sentences(example, example_idx)
    if rows is None or sentences is None:
        return
    
    # Get model outputs
    results = compute_attention_matrix(model, rows, sentences, return_type="both")
    pair_scores = results['pair_scores']
    forward_attention = results['forward_attention']
    reverse_attention = results['reverse_attention']
    
    # Create visualizations
    base_filename = f"example_{example_idx}_{model_type}_bidirectional"
    
    # 1. Bidirectional attention visualization
    visualize_bidirectional_attention(
        pair_scores, forward_attention, reverse_attention,
        rows, sentences,
        title=f"Example {example_idx} - {model_type.title()} Bidirectional Analysis",
        output_file=os.path.join(output_dir, f"{base_filename}_attention.png")
    )
    
    # 2. Extract and visualize join paths
    join_paths = model.extract_join_paths(
        torch.tensor(pair_scores).unsqueeze(0), rows, sentences, 
        threshold=0.1, top_k=model.top_k
    )
    
    if join_paths:
        visualize_join_paths(
            pair_scores, rows, sentences, join_paths,
            title=f"Example {example_idx} - {model_type.title()} Join Paths",
            output_file=os.path.join(output_dir, f"{base_filename}_join_paths.png")
        )
        
        # Save join paths details
        join_paths_file = os.path.join(output_dir, f"{base_filename}_join_paths.txt")
        with open(join_paths_file, 'w', encoding='utf-8') as f:
            f.write(f"Join Paths for Example {example_idx} ({model_type} model)\n")
            f.write("="*50 + "\n\n")
            for i, (row_idx, sent_idx, score) in enumerate(join_paths):
                f.write(f"Path {i+1}: Row {row_idx+1} ↔ Sentence {sent_idx+1}\n")
                f.write(f"  Score: {score:.4f}\n")
                f.write(f"  Row: {rows[row_idx]}\n")
                f.write(f"  Sentence: {sentences[sent_idx]}\n\n")
        
        print(f"Extracted {len(join_paths)} join paths for example {example_idx}")
    else:
        print(f"No join paths found for example {example_idx}")

def save_diagnostics_heatmaps(model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel],
                             rows: List[str],
                             sentences: List[str],
                             example_idx: int,
                             output_dir: str,
                             use_refinement: bool = True,
                             base_model_name: Optional[str] = None) -> None:
    """
    Generate comprehensive step-by-step diagnostic heatmaps and analysis.
    
    Args:
        model: The model to analyze (unidirectional or bidirectional)
        rows: List of table row texts
        sentences: List of sentence texts
        example_idx: Index of the example being analyzed
        output_dir: Directory to save diagnostics
        use_refinement: Whether refinement step is enabled in the model
        base_model_name: Name of the base model (unused, kept for API compatibility)
    """
    print(f"🔬 Generating step-by-step diagnostics for example {example_idx}...")
    
    # Create diagnostics subdirectory
    # Save diagnostics to visualizations subdirectory
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = viz_dir / f"diagnostics_example_{example_idx}"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect model type
    model_type = detect_model_type(model)
    is_bidirectional = model_type == "bidirectional"
    
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        # Step 1: Raw embedding similarities (baseline)
        raw_similarities = compute_raw_transformer_similarities(model, rows, sentences)
        
        # Save raw similarities
        np.save(diag_dir / "step1_raw_similarities.npy", raw_similarities)
        visualize_attention_matrix(
            raw_similarities, rows, sentences,
            title=f"Step 1: Raw Embedding Similarities (Example {example_idx})",
            output_file=str(diag_dir / "step1_raw_similarities.png"),
            show_values=True
        )
        
        # Step 2-6: Model-specific diagnostics
        if is_bidirectional:
            _save_bidirectional_diagnostics(model, rows, sentences, example_idx, diag_dir, use_refinement)
        else:
            _save_unidirectional_diagnostics(model, rows, sentences, example_idx, diag_dir, use_refinement)
        
        # Generate validation report
        _generate_validation_report(model, rows, sentences, example_idx, diag_dir, raw_similarities)
        


def _save_bidirectional_diagnostics(model: BidirectionalTableTextModel,
                                   rows: List[str],
                                   sentences: List[str],
                                   example_idx: int,
                                   diag_dir: Path,
                                   use_refinement: bool) -> None:
    """Save diagnostics specific to bidirectional models."""
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Encode inputs
    rows_embeddings = model.encode_sentences(rows, normalize=True)
    sentences_embeddings = model.encode_sentences(sentences, normalize=True)
    
    # Add batch dimension and align dtype
    rows_tensor = rows_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
    sentences_tensor = sentences_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
    
    # Get diagnostics from model
    if hasattr(model, 'bidirectional_attention') and hasattr(model.bidirectional_attention, 'forward'):
        # Call with diagnostics=True if supported
        try:
            result = model.bidirectional_attention.forward(rows_tensor, sentences_tensor, diagnostics=True)
            # BidirectionalCrossAttention.forward(diagnostics=True) returns a 6-tuple:
            # (pair_scores, refined_rows, refined_sentences, fwd_attn, rev_attn, diag_dict)
            # where diag_dict contains 'contextualized_rows', 'contextualized_sentences', etc.
            if isinstance(result, tuple) and len(result) == 6:
                pair_scores_t, _, _, forward_attn_t, reverse_attn_t, model_diag = result
                pair_scores = safe_tensor_to_numpy(pair_scores_t[0])
                forward_attn = safe_tensor_to_numpy(forward_attn_t[0])
                reverse_attn = safe_tensor_to_numpy(reverse_attn_t[0])

                # Compute contextualized similarities from the model's actual contextualized
                # embeddings (post-residual, before optional refinement). This is correct
                # because it uses the full W_V projection path, unlike the old manual approach
                # that bypassed W_V by using raw attention weights × raw embeddings.
                ctx_rows_t = model_diag.get('contextualized_rows')    # [batch, num_rows, dim] cpu
                ctx_sents_t = model_diag.get('contextualized_sentences')  # [batch, num_sents, dim] cpu
                if ctx_rows_t is not None and ctx_sents_t is not None:
                    contextualized_sims = _compute_cosine_similarity_matrix(ctx_rows_t[0], ctx_sents_t[0])
                else:
                    contextualized_sims = None

                diagnostics = {
                    'forward_attention': forward_attn,
                    'reverse_attention': reverse_attn,
                    'contextualized_similarities': contextualized_sims,
                    'pair_scores': pair_scores,
                }
            elif isinstance(result, tuple) and len(result) == 2:
                # Legacy 2-tuple return: (pair_scores, diagnostics_dict)
                pair_scores, diagnostics = result
                pair_scores = safe_tensor_to_numpy(pair_scores[0])
                forward_attn = diagnostics['forward_attention']
                reverse_attn = diagnostics['reverse_attention']
            else:
                raise ValueError(f"Unexpected diagnostics return length: {len(result)}")
        except Exception:
            # Fallback: standard forward pass without diagnostics flag.
            # refined_rows / refined_sentences are the actual model outputs (post W_V + residual
            # + optional refinement), so use them directly for Step 4 similarities.
            pair_scores_t, refined_rows_t, refined_sents_t, forward_attn_t, reverse_attn_t = model.bidirectional_attention(rows_tensor, sentences_tensor)
            pair_scores = safe_tensor_to_numpy(pair_scores_t[0])
            forward_attn = safe_tensor_to_numpy(forward_attn_t[0])
            reverse_attn = safe_tensor_to_numpy(reverse_attn_t[0])
            contextualized_sims = _compute_cosine_similarity_matrix(refined_rows_t[0], refined_sents_t[0])

            diagnostics = {
                'forward_attention': forward_attn,
                'reverse_attention': reverse_attn,
                'contextualized_similarities': contextualized_sims,
                'pair_scores': pair_scores,
            }
    else:
        # Fallback for older model structure
        pair_scores_t, refined_rows_t, refined_sents_t, forward_attn_t, reverse_attn_t = model.bidirectional_attention(rows_tensor, sentences_tensor)
        pair_scores = safe_tensor_to_numpy(pair_scores_t[0])
        forward_attn = safe_tensor_to_numpy(forward_attn_t[0])
        reverse_attn = safe_tensor_to_numpy(reverse_attn_t[0])
        contextualized_sims = _compute_cosine_similarity_matrix(refined_rows_t[0], refined_sents_t[0])

        diagnostics = {
            'forward_attention': forward_attn,
            'reverse_attention': reverse_attn,
            'contextualized_similarities': contextualized_sims,
            'pair_scores': pair_scores,
        }
    
    # Step 2: Forward attention weights
    forward_attn = diagnostics['forward_attention']
    np.save(diag_dir / "step2_forward_attention.npy", forward_attn)
    visualize_attention_matrix(
        forward_attn, rows, sentences,
        title=f"Step 2: Forward Attention Weights (Example {example_idx})",
        output_file=str(diag_dir / "step2_forward_attention.png"),
        show_values=True
    )
    
    # Step 3: Reverse attention weights (transposed for visualization)
    reverse_attn = diagnostics['reverse_attention']
    np.save(diag_dir / "step3_reverse_attention.npy", reverse_attn)
    # Transpose for consistent visualization (rows on y-axis, sentences on x-axis)
    reverse_attn_transposed = reverse_attn.T
    visualize_attention_matrix(
        reverse_attn_transposed, rows, sentences,
        title=f"Step 3: Reverse Attention Weights (Example {example_idx})",
        output_file=str(diag_dir / "step3_reverse_attention.png"),
        show_values=True
    )
    
    # Step 4: Contextualized similarities (before refinement)
    if diagnostics.get('contextualized_similarities') is not None:
        contextualized_sims = diagnostics['contextualized_similarities']
    else:
        # Last-resort fallback: call the model without diagnostics and use the returned
        # contextualized (post-W_V + residual) tensors directly.
        # This is correct — the old approach of doing `attn_weights @ raw_embeddings` was
        # wrong because it skipped the W_V projection entirely.
        with torch.no_grad():
            _ps, ctx_rows_fb, ctx_sents_fb, _, _ = model.bidirectional_attention(rows_tensor, sentences_tensor)
        contextualized_sims = _compute_cosine_similarity_matrix(ctx_rows_fb[0], ctx_sents_fb[0])
    
    np.save(diag_dir / "step4_contextualized_similarities.npy", contextualized_sims)
    visualize_attention_matrix(
        contextualized_sims, rows, sentences,
        title=f"Step 4: Contextualized Similarities (Example {example_idx})",
        output_file=str(diag_dir / "step4_contextualized_similarities.png"),
        show_values=True
    )
    
    # Step 5: Refined similarities (after refinement, if enabled)
    if use_refinement:
        if diagnostics.get('refined_similarities') is not None:
            refined_sims = diagnostics['refined_similarities']
        else:
            # Apply refinement (LayerNorm + FFN) to contextualized vectors
            # This is a simplified version - the actual model may have different refinement layers
            try:
                # Try to access the refinement layers from the model
                if hasattr(model, 'bidirectional_attention'):
                    bidirectional_attn = model.bidirectional_attention
                    if hasattr(bidirectional_attn, 'row_refinement') and hasattr(bidirectional_attn, 'sentence_refinement'):
                        # Apply refinement to contextualized vectors
                        refined_rows = bidirectional_attn.row_refinement(contextualized_rows)
                        refined_sentences = bidirectional_attn.sentence_refinement(contextualized_sentences)
                        refined_sims = _compute_cosine_similarity_matrix(refined_rows[0], refined_sentences[0])
                    else:
                        # Fallback: use contextualized similarities
                        refined_sims = contextualized_sims
                else:
                    refined_sims = contextualized_sims
            except Exception as e:
                refined_sims = contextualized_sims
        
        np.save(diag_dir / "step5_refined_similarities.npy", refined_sims)
        visualize_attention_matrix(
            refined_sims, rows, sentences,
            title=f"Step 5: Refined Similarities (Example {example_idx})",
            output_file=str(diag_dir / "step5_refined_similarities.png"),
            show_values=True
        )
    else:
        refined_sims = contextualized_sims
    
    # Step 6: Final pair scores
    final_scores = diagnostics['pair_scores']
    np.save(diag_dir / "step6_final_pair_scores.npy", final_scores)
    visualize_attention_matrix(
        final_scores, rows, sentences,
        title=f"Step 6: Final Pair Scores (Example {example_idx})",
        output_file=str(diag_dir / "step6_final_pair_scores.png"),
        show_values=True
    )
    
    # Validation: Compare with comprehensive analysis
    try:
        # Use the same model and computation path as comprehensive analysis
        device = next(model.parameters()).device
        
        # Encode using the same method as comprehensive analysis
        comp_row_embeddings = model.encode_sentences(rows, normalize=True)
        comp_sentence_embeddings = model.encode_sentences(sentences, normalize=True)
        comp_rows_tensor = comp_row_embeddings.unsqueeze(0).to(device)
        comp_sentences_tensor = comp_sentence_embeddings.unsqueeze(0).to(device)
        
        # Get comprehensive analysis results using the same computation
        comp_pair_scores, _, _, _, _ = model.bidirectional_attention(comp_rows_tensor, comp_sentences_tensor)
        comp_pair_scores = safe_tensor_to_numpy(comp_pair_scores[0])
        
        # Check if they match (within tolerance)
        if np.allclose(final_scores, comp_pair_scores, atol=1e-6):
            pass  # Validation successful
        else:
            # Save comparison matrix
            diff_matrix = np.abs(final_scores - comp_pair_scores)
            np.save(diag_dir / "validation_difference.npy", diff_matrix)
            visualize_attention_matrix(
                diff_matrix, rows, sentences,
                title=f"Validation: Absolute Difference (Example {example_idx})",
                output_file=str(diag_dir / "validation_difference.png"),
                show_values=True
            )
    except Exception as e:
        pass  # Validation failed silently

def _save_unidirectional_diagnostics(model: TableTextEmbeddingModel,
                                    rows: List[str],
                                    sentences: List[str],
                                    example_idx: int,
                                    diag_dir: Path,
                                    use_refinement: bool) -> None:
    """Save diagnostics specific to unidirectional models."""
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Encode inputs
    rows_embeddings = model.encode_sentences(rows, normalize=True)
    sentences_embeddings = model.encode_sentences(sentences, normalize=True)
    
    # Add batch dimension and align dtype
    rows_tensor = rows_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
    sentences_tensor = sentences_embeddings.unsqueeze(0).to(device=device, dtype=model_dtype)
    
    # Step 2: Cross-attention weights
    print("  Step 2: Cross-attention weights...")
    # Handle different attention interfaces
    if hasattr(model, 'attention_type') and model.attention_type != "standard":
        # Advanced attention modules use (queries_emb, keys_emb, values_emb) interface
        context_vectors, attention_weights = model.cross_attention(
            queries_emb=rows_tensor,
            keys_emb=sentences_tensor,
            values_emb=sentences_tensor
        )
    else:
        # Original CrossAttentionModule uses (rows_embeddings, sentences_embeddings) interface
        attention_weights, context_vectors = model.cross_attention(rows_tensor, sentences_tensor)
    attention_weights = safe_tensor_to_numpy(attention_weights[0])
    
    np.save(diag_dir / "step2_attention_weights.npy", attention_weights)
    visualize_attention_matrix(
        attention_weights, rows, sentences,
        title=f"Step 2: Cross-Attention Weights (Example {example_idx})",
        output_file=str(diag_dir / "step2_attention_weights.png"),
        show_values=True
    )
    
    # Step 3: Contextualized similarities (after attention, before refinement)
    print("  Step 3: Contextualized similarities (after attention)...")
    # Compute residual connection: rows + context
    contextualized_rows = rows_tensor + context_vectors
    contextualized_sims = _compute_cosine_similarity_matrix(contextualized_rows[0], sentences_tensor[0])
    
    np.save(diag_dir / "step3_contextualized_similarities.npy", contextualized_sims)
    visualize_attention_matrix(
        contextualized_sims, rows, sentences,
        title=f"Step 3: Contextualized Similarities (Example {example_idx})",
        output_file=str(diag_dir / "step3_contextualized_similarities.png"),
        show_values=True
    )
    
    # Step 4: Refined similarities (after feed-forward, if enabled)
    if use_refinement:
        print("  Step 4: Refined similarities (after feed-forward)...")
        # Apply layer norm and feed-forward
        norm_context = model.norm1(contextualized_rows)
        ff_output = model.feed_forward(norm_context)
        refined_rows = model.norm2(ff_output + norm_context)
        
        refined_sims = _compute_cosine_similarity_matrix(refined_rows[0], sentences_tensor[0])
        
        np.save(diag_dir / "step4_refined_similarities.npy", refined_sims)
        visualize_attention_matrix(
            refined_sims, rows, sentences,
            title=f"Step 4: Refined Similarities (Example {example_idx})",
            output_file=str(diag_dir / "step4_refined_similarities.png"),
            show_values=True
        )
    else:
        print("  Step 4: Skipped (refinement disabled)")
        refined_sims = contextualized_sims
    
    # Step 5: Final pair scores (using model's aggregation method)
    print("  Step 5: Final pair scores...")
    final_scores = np.zeros((len(rows), len(sentences)))
    for i in range(len(rows)):
        for j in range(len(sentences)):
            single_row = rows_tensor[:, i:i+1, :]
            single_sentence = sentences_tensor[:, j:j+1, :]
            score, _ = model(single_row, single_sentence)
            final_scores[i, j] = score.item()
    
    np.save(diag_dir / "step5_final_pair_scores.npy", final_scores)
    visualize_attention_matrix(
        final_scores, rows, sentences,
        title=f"Step 5: Final Pair Scores (Example {example_idx})",
        output_file=str(diag_dir / "step5_final_pair_scores.png"),
        show_values=True
    )

def _compute_cosine_similarity_matrix(tensor1: torch.Tensor, tensor2: torch.Tensor) -> np.ndarray:
    """Compute cosine similarity matrix between two tensors."""
    # Normalize tensors
    tensor1_norm = torch.nn.functional.normalize(tensor1, p=2, dim=1)
    tensor2_norm = torch.nn.functional.normalize(tensor2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity_matrix = torch.mm(tensor1_norm, tensor2_norm.t())
    
    # Convert BFloat16 to Float32 before converting to numpy (BFloat16 not supported by numpy)
    if similarity_matrix.dtype == torch.bfloat16:
        similarity_matrix = similarity_matrix.float()
    
    return safe_tensor_to_numpy(similarity_matrix)

def _generate_validation_report(model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel],
                               rows: List[str],
                               sentences: List[str],
                               example_idx: int,
                               diag_dir: Path,
                               raw_similarities: np.ndarray) -> None:
    """Generate comprehensive validation and analysis report."""
    print("  Generating validation report...")
    
    model_type = detect_model_type(model)
    is_bidirectional = model_type == "bidirectional"
    
    report_file = diag_dir / f"validation_report_example_{example_idx}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"=== DIAGNOSTIC VALIDATION REPORT ===\n")
        f.write(f"Example: {example_idx}\n")
        f.write(f"Model Type: {'Bidirectional' if is_bidirectional else 'Unidirectional'}\n")
        f.write(f"Rows: {len(rows)}, Sentences: {len(sentences)}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        # Load saved matrices for analysis
        raw_sims = raw_similarities
        
        if is_bidirectional:
            forward_attn = np.load(diag_dir / "step2_forward_attention.npy")
            reverse_attn = np.load(diag_dir / "step3_reverse_attention.npy")
            final_scores = np.load(diag_dir / "step6_final_pair_scores.npy")
            
            # Validation checks
            f.write("=== ATTENTION NORMALIZATION CHECKS ===\n")
            forward_row_sums = forward_attn.sum(axis=1)
            reverse_col_sums = reverse_attn.sum(axis=0)
            
            f.write(f"Forward attention row sums (should be ~1.0):\n")
            for i, sum_val in enumerate(forward_row_sums):
                f.write(f"  Row {i+1}: {sum_val:.6f}\n")
            
            f.write(f"\nReverse attention column sums (should be ~1.0):\n")
            for j, sum_val in enumerate(reverse_col_sums):
                f.write(f"  Sentence {j+1}: {sum_val:.6f}\n")
            
            f.write(f"\n=== SCORE RANGE VALIDATION ===\n")
            f.write(f"Raw similarities: [{raw_sims.min():.6f}, {raw_sims.max():.6f}]\n")
            f.write(f"Forward attention: [{forward_attn.min():.6f}, {forward_attn.max():.6f}]\n")
            f.write(f"Reverse attention: [{reverse_attn.min():.6f}, {reverse_attn.max():.6f}]\n")
            f.write(f"Final pair scores: [{final_scores.min():.6f}, {final_scores.max():.6f}]\n")
            
        else:
            attention_weights = np.load(diag_dir / "step2_attention_weights.npy")
            final_scores = np.load(diag_dir / "step5_final_pair_scores.npy")
            
            # Validation checks
            f.write("=== ATTENTION NORMALIZATION CHECKS ===\n")
            attention_row_sums = attention_weights.sum(axis=1)
            
            f.write(f"Attention row sums (should be ~1.0):\n")
            for i, sum_val in enumerate(attention_row_sums):
                f.write(f"  Row {i+1}: {sum_val:.6f}\n")
            
            f.write(f"\n=== SCORE RANGE VALIDATION ===\n")
            f.write(f"Raw similarities: [{raw_sims.min():.6f}, {raw_sims.max():.6f}]\n")
            f.write(f"Attention weights: [{attention_weights.min():.6f}, {attention_weights.max():.6f}]\n")
            f.write(f"Final pair scores: [{final_scores.min():.6f}, {final_scores.max():.6f}]\n")
        
        # Top 3 pairs analysis with actual content
        f.write(f"\n=== TOP 3 PAIRS ANALYSIS ===\n")
        
        # Raw similarities top pairs
        raw_top_pairs = get_top_k_pairs(raw_sims, rows, sentences, k=3)
        f.write(f"\nTop 3 by Raw Similarities:\n")
        for i, pair in enumerate(raw_top_pairs):
            f.write(f"  {i+1}. Row {pair['row_idx']+1} - Sentence {pair['sentence_idx']+1} (Score: {pair['score']:.6f})\n")
            f.write(f"     Row: {pair['row'][:100]}{'...' if len(pair['row']) > 100 else ''}\n")
            f.write(f"     Sentence: {pair['sentence'][:100]}{'...' if len(pair['sentence']) > 100 else ''}\n\n")
        
        # Final scores top pairs
        final_top_pairs = get_top_k_pairs(final_scores, rows, sentences, k=3)
        f.write(f"Top 3 by Final Model Scores:\n")
        for i, pair in enumerate(final_top_pairs):
            f.write(f"  {i+1}. Row {pair['row_idx']+1} - Sentence {pair['sentence_idx']+1} (Score: {pair['score']:.6f})\n")
            f.write(f"     Row: {pair['row'][:100]}{'...' if len(pair['row']) > 100 else ''}\n")
            f.write(f"     Sentence: {pair['sentence'][:100]}{'...' if len(pair['sentence']) > 100 else ''}\n\n")
        
        # Matrix statistics
        f.write(f"=== MATRIX STATISTICS ===\n")
        f.write(f"Raw similarities - Mean: {raw_sims.mean():.6f}, Std: {raw_sims.std():.6f}\n")
        f.write(f"Final scores - Mean: {final_scores.mean():.6f}, Std: {final_scores.std():.6f}\n")
        f.write(f"Learning effect (Final - Raw) - Mean: {(final_scores - raw_sims).mean():.6f}\n")
        
        # Correlation analysis
        correlation = np.corrcoef(raw_sims.flatten(), final_scores.flatten())[0, 1]
        f.write(f"Correlation between raw and final scores: {correlation:.6f}\n")
        
        f.write(f"\n=== ACTUAL CONTENT ===\n")
        f.write(f"Table Rows:\n")
        for i, row in enumerate(rows):
            f.write(f"  Row {i+1}: {row}\n")
        
        f.write(f"\nSentences:\n")
        for j, sentence in enumerate(sentences):
            f.write(f"  Sentence {j+1}: {sentence}\n")
    
    print(f"  Validation report saved to {report_file}")

def visualize_four_stage_comparison(
    trained_model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel],
    rows: List[str],
    sentences: List[str],
    title: str = "Complete 4-Stage Model Comparison",
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (28, 12),
    base_model_name: str = "all-roberta-large-v1",
    init_method: str = "xavier_uniform",
    init_method_params: dict = None,
    stage_3_label: str = "Stage 3: Trained"
) -> Dict[str, np.ndarray]:
    """
    Create comprehensive 4-stage comparison visualization.
    
    Stage 0: Frozen Encoder Only (basic cosine similarity)
    Stage 1: Simple Cross-Attention (specified init, standard attention, no LoRA)
    Stage 2: Sophisticated Pre-training (same architecture as trained model, but untrained with specified init)
    Stage 3: Sophisticated Post-training (the actual trained model)
    
    Args:
        trained_model: The trained model to use as Stage 3
        rows: List of row strings
        sentences: List of sentence strings
        title: Title for the visualization
        output_file: Optional output file path
        figsize: Figure size tuple
        base_model_name: Base model name for creating untrained models
        init_method: Initialization method to use for untrained models (Stages 1 & 2)
        init_method_params: Optional parameters for the initialization method
    """
    print("🎯 Creating 4-stage comparison visualization...")
    
    device = next(trained_model.parameters()).device
    similarities = {}
    
    with torch.no_grad():
        # Stage 0: Frozen Encoder Only
        print("  🔥 Stage 0: Computing frozen encoder similarities...")
        
        # Use the trained model's sentence encoder for consistency
        # This avoids dimension mismatches and uses the same encoder that was actually used
        row_embeddings = trained_model.encode_sentences(rows, normalize=True)
        sentence_embeddings = trained_model.encode_sentences(sentences, normalize=True)
        
        from sentence_transformers import util
        stage0_similarities = util.cos_sim(row_embeddings, sentence_embeddings).float().cpu().numpy()
        similarities['Stage 0: Frozen Encoder'] = stage0_similarities
        
        # Stage 1: Simple Cross-Attention
        print("  🎯 Stage 1: Creating simple cross-attention model...")
        model_type = "bidirectional" if isinstance(trained_model, BidirectionalTableTextModel) else "unidirectional"
        
        if model_type == "bidirectional":
            simple_model = BidirectionalTableTextModel(
                sentence_encoder=trained_model.sentence_encoder,
                embedding_dim=trained_model.embedding_dim,
                trainable_encoder=False,
                use_cross_attention_lora=False,  # No LoRA
                lora_rank=16,
                lora_alpha=32.0,
                lora_dropout=0.1,
                top_k=3,  # Simple top-k
                pair_score_method="cosine",
                share_weights=False,  # No weight sharing
                use_refinement=True,  # NEW: Enable refinement for Stage 1 as requested
                use_self_attention=False,
                attention_type="standard",  # Standard attention
                sparse_top_k=3,
                window_size=5,
                threshold_base=0.1,
                use_header_conditioning=getattr(trained_model, 'use_header_conditioning', False),
                init_method=init_method,  # Use passed initialization method
                init_method_params=init_method_params,
                verbose=False  # Suppress init logs during visualization
            )
        else:
            simple_model = TableTextEmbeddingModel(
                sentence_encoder=trained_model.sentence_encoder,
                embedding_dim=trained_model.embedding_dim,
                trainable_encoder=False,
                use_cross_attention_lora=False,  # No LoRA
                lora_rank=16,
                lora_alpha=32.0,
                lora_dropout=0.1,
                top_k=3,  # Simple top-k
                attention_type="standard",  # Standard attention
                sparse_top_k=3,
                window_size=5,
                threshold_base=0.1,
                init_method=init_method,  # Use passed initialization method
                init_method_params=init_method_params
            )
        
        simple_model.to(device)
        simple_model.eval()
        
        # Compute Stage 1 similarities
        print("  🎯 Stage 1: Computing simple cross-attention similarities...")
        rows_tensor = row_embeddings.unsqueeze(0).to(device)
        sentences_tensor = sentence_embeddings.unsqueeze(0).to(device)
        
        if model_type == "bidirectional":
            # For bidirectional models, we want the pair_scores matrix, not the global similarity
            global_sim, stage1_similarities = simple_model(rows_tensor, sentences_tensor, aggregation_method="top_k_pairs")
            stage1_similarities = safe_tensor_to_numpy(stage1_similarities[0])  # Remove batch dimension
        else:
            stage1_similarities = compute_attention_matrix(simple_model, rows, sentences, return_type="similarities")
        
        similarities['Stage 1: Simple Cross-Attention'] = stage1_similarities
        
        # Clean up simple model
        del simple_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Stage 2: Sophisticated Pre-training (create untrained version of trained model)
        print("  🚀 Stage 2: Creating sophisticated untrained model...")
        if model_type == "bidirectional":
            sophisticated_untrained = BidirectionalTableTextModel(
                sentence_encoder=trained_model.sentence_encoder,
                embedding_dim=trained_model.embedding_dim,
                trainable_encoder=trained_model.trainable_encoder,
                use_cross_attention_lora=trained_model.bidirectional_attention.use_lora,
                lora_rank=trained_model.bidirectional_attention.lora_rank if hasattr(trained_model.bidirectional_attention, 'lora_rank') else 128,
                lora_alpha=trained_model.bidirectional_attention.lora_alpha if hasattr(trained_model.bidirectional_attention, 'lora_alpha') else 512,
                lora_dropout=trained_model.bidirectional_attention.lora_dropout if hasattr(trained_model.bidirectional_attention, 'lora_dropout') else 0.1,
                top_k=trained_model.top_k,
                pair_score_method=trained_model.pair_score_method,
                share_weights=trained_model.share_weights,
                use_refinement=trained_model.use_refinement,
                use_self_attention=trained_model.use_self_attention,
                attention_type=trained_model.bidirectional_attention.attention_type if hasattr(trained_model.bidirectional_attention, 'attention_type') else "standard",
                sparse_top_k=getattr(trained_model.bidirectional_attention, 'sparse_top_k', 5),
                window_size=getattr(trained_model.bidirectional_attention, 'window_size', 5),
                threshold_base=getattr(trained_model.bidirectional_attention, 'threshold_base', 0.1),
                use_header_conditioning=getattr(trained_model, 'use_header_conditioning', False),
                init_method=init_method,  # Use passed initialization method
                init_method_params=init_method_params,
                verbose=False  # Suppress init logs during visualization
            )
        else:
            sophisticated_untrained = TableTextEmbeddingModel(
                sentence_encoder=trained_model.sentence_encoder,
                embedding_dim=trained_model.embedding_dim,
                trainable_encoder=trained_model.trainable_encoder,
                use_cross_attention_lora=trained_model.cross_attention.use_lora,
                lora_rank=getattr(trained_model.cross_attention, 'lora_rank', 128),
                lora_alpha=getattr(trained_model.cross_attention, 'lora_alpha', 512),
                lora_dropout=getattr(trained_model.cross_attention, 'lora_dropout', 0.1),
                top_k=trained_model.top_k,
                attention_type=getattr(trained_model.cross_attention, 'attention_type', "standard"),
                sparse_top_k=getattr(trained_model.cross_attention, 'sparse_top_k', 5),
                window_size=getattr(trained_model.cross_attention, 'window_size', 5),
                threshold_base=getattr(trained_model.cross_attention, 'threshold_base', 0.1),
                init_method=init_method,  # Use passed initialization method
                init_method_params=init_method_params
            )
        
        sophisticated_untrained.to(device)
        sophisticated_untrained.eval()
        
        # Compute Stage 2 similarities
        print("  🚀 Stage 2: Computing sophisticated pre-training similarities...")
        if model_type == "bidirectional":
            # For bidirectional models, we want the pair_scores matrix, not the global similarity
            global_sim, stage2_similarities = sophisticated_untrained(rows_tensor, sentences_tensor, aggregation_method="top_k_pairs")
            stage2_similarities = safe_tensor_to_numpy(stage2_similarities[0])  # Remove batch dimension
        else:
            stage2_similarities = compute_attention_matrix(sophisticated_untrained, rows, sentences, return_type="similarities")
        
        similarities['Stage 2: Sophisticated (Pre)'] = stage2_similarities
        
        # Clean up sophisticated untrained model
        del sophisticated_untrained
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Stage 3: Post-training (the actual trained model)
        print("  🏆 Stage 3: Computing trained model similarities...")
        if model_type == "bidirectional":
            # For bidirectional models, we want the pair_scores matrix, not the global similarity
            global_sim, stage3_similarities = trained_model(rows_tensor, sentences_tensor, aggregation_method="top_k_pairs")
            stage3_similarities = safe_tensor_to_numpy(stage3_similarities[0])  # Remove batch dimension
        else:
            stage3_similarities = compute_attention_matrix(trained_model, rows, sentences, return_type="similarities")
        
        # Use dynamic label for stage 3 key
        similarities[stage_3_label] = stage3_similarities
    
    # Create 4-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    stage_names = list(similarities.keys())
    stage_emojis = ['🔥', '🎯', '🚀', '🏆']
    
    # Debug: Print shapes of all matrices
    print(f"\n🔍 DEBUG: Matrix shapes for visualization:")
    for stage_name, matrix in similarities.items():
        print(f"  {stage_name}: {matrix.shape}")
    
    for idx, (stage_name, matrix) in enumerate(similarities.items()):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # Validate matrix shape before visualization
        if matrix.ndim != 2:
            print(f"  ⚠️ Warning: {stage_name} matrix has invalid shape {matrix.shape}, expected 2D")
            if matrix.ndim == 1:
                # If 1D, try to reshape into a matrix
                expected_size = len(rows) * len(sentences)
                if len(matrix) == expected_size:
                    print(f"    Reshaping 1D array ({len(matrix)}) to 2D matrix ({len(rows)}, {len(sentences)})")
                    matrix = matrix.reshape(len(rows), len(sentences))
                else:
                    print(f"    Cannot reshape: array size {len(matrix)} != expected {expected_size}")
                    # Create a placeholder matrix
                    matrix = np.zeros((len(rows), len(sentences)))
            else:
                print(f"    Creating placeholder matrix with shape ({len(rows)}, {len(sentences)})")
                matrix = np.zeros((len(rows), len(sentences)))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
        
        # Set title with emoji
        ax.set_title(f"{stage_emojis[idx]} {stage_name}", fontsize=12, fontweight='bold', pad=10)
        
        # Set labels
        ax.set_xlabel('Sentences', fontsize=10)
        ax.set_ylabel('Rows', fontsize=10)
        
        # Set ticks
        ax.set_xticks(range(len(sentences)))
        ax.set_yticks(range(len(rows)))
        ax.set_xticklabels([f"S{i+1}" for i in range(len(sentences))], fontsize=8)
        ax.set_yticklabels([f"R{i+1}" for i in range(len(rows))], fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add mean similarity as text
        mean_sim = np.mean(matrix)
        ax.text(0.02, 0.98, f'Avg: {mean_sim:.3f}', transform=ax.transAxes, 
                fontsize=9, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
        print(f"  💾 4-stage comparison saved to: {output_file}")
        plt.close()  # Close the figure to save memory
    else:
        plt.close()  # Close the figure even if not saving
    
    # Print progression analysis
    print(f"\n📊 4-STAGE PROGRESSION ANALYSIS:")
    print(f"{'Stage':<35} {'Avg Similarity':<15} {'Improvement':<15}")
    print(f"-" * 65)
    
    prev_avg = 0
    for idx, (stage_name, matrix) in enumerate(similarities.items()):
        avg_sim = np.mean(matrix)
        improvement = avg_sim - prev_avg if idx > 0 else 0
        print(f"{stage_emojis[idx]} {stage_name:<30} {avg_sim:<15.4f} {improvement:+.4f}" if idx > 0 else f"{stage_emojis[idx]} {stage_name:<30} {avg_sim:<15.4f} {'baseline':<15}")
        prev_avg = avg_sim
    
    return similarities

def create_complete_four_stage_analysis(
    trained_model: Union[TableTextEmbeddingModel, BidirectionalTableTextModel],
    examples: List[Dict[str, Any]],
    output_dir: str,
    example_indices: str = "0,1,2",
    base_model_name: str = "all-roberta-large-v1",
    init_method: str = "xavier_uniform",
    init_method_params: dict = None,
    stage_3_label: str = "Stage 3: Trained"
) -> None:
    """
    Create complete 4-stage analysis for multiple examples.
    This is the main function to call from training/evaluation scripts.
    
    Args:
        trained_model: The trained model to analyze
        examples: List of examples to analyze  
        output_dir: Directory to save outputs
        example_indices: Comma-separated example indices (e.g., "0,1,2")
        base_model_name: Base model name for Stage 0 comparison
    """
    print(f"\n🎯 CREATING COMPLETE 4-STAGE ANALYSIS")
    print(f"="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse example indices
    try:
        if example_indices.lower() in ['all', '*']:
            indices = list(range(len(examples)))
        else:
            indices = [int(idx.strip()) for idx in example_indices.split(',')]
    except ValueError:
        print(f"Invalid example indices: {example_indices}. Using default: [0]")
        indices = [0]
    
    # Validate indices
    valid_indices = [idx for idx in indices if 0 <= idx < len(examples)]
    if not valid_indices:
        print(f"No valid example indices found. Available: 0-{len(examples)-1}")
        return
    
    print(f"Analyzing {len(valid_indices)} examples: {valid_indices}")
    
    # Process each example
    for idx in valid_indices:
        example = examples[idx]
        
        print(f"\n📊 Processing Example {idx}...")
        
        # Extract rows and sentences
        rows, sentences = extract_rows_and_sentences(example, idx)
        if rows is None or sentences is None:
            print(f"  ⚠️ Skipping example {idx} - could not extract data")
            continue
        
        print(f"  Found {len(rows)} rows and {len(sentences)} sentences")
        
        # Create example-specific output directory
        example_dir = output_path / f"example_{idx}"
        example_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 4-stage comparison visualization
        output_file = str(example_dir / f"four_stage_comparison_example_{idx}.png")
        
        try:
            similarities = visualize_four_stage_comparison(
                trained_model=trained_model,
                rows=rows,
                sentences=sentences,
                title=f"4-Stage Model Evolution - Example {idx}",
                output_file=output_file,
                base_model_name=base_model_name,
                init_method=init_method,
                init_method_params=init_method_params,
                stage_3_label=stage_3_label
            )
            
            # Save similarity matrices as numpy files for further analysis
            for stage_name, matrix in similarities.items():
                safe_name = stage_name.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
                np.save(str(example_dir / f"{safe_name}_similarities_example_{idx}.npy"), matrix)
            
            print(f"  ✅ Example {idx} analysis complete")
            
        except Exception as e:
            print(f"  ❌ Error processing example {idx}: {e}")
            continue
    
    print(f"\n🎯 4-STAGE ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {output_dir}")
    print(f"="*60)

if __name__ == "__main__":
    main() 