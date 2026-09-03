import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
import math

# Import the new initialization system
from initialization import initialize_attention_weights, get_available_methods

# Optional sparse attention activations
try:
    from entmax import entmax15, entmax_bisect
except Exception:
    entmax15 = None
    entmax_bisect = None

def apply_attention_activation(logits: torch.Tensor, dim: int, name: str = "softmax", alpha: float = 1.5) -> torch.Tensor:
    name = (name or "softmax").lower()
    if name == "entmax15" and entmax15 is not None:
        return entmax15(logits, dim=dim)
    if name in ("alpha_entmax", "entmax") and entmax_bisect is not None:
        return entmax_bisect(logits, alpha=alpha, dim=dim)
    return F.softmax(logits, dim=dim)

class AttentionOutputGate(nn.Module):
    """
    Query-dependent gating applied to the attention *output* vectors (post-SDPA),
    inspired by gated-attention variants (gate after SDPA output).
    """
    def __init__(
        self,
        embedding_dim: int,
        mode: str = "scalar",   # "scalar" (per-query) or "vector" (per-query, per-feature)
        hidden_dim: int = 0,    # 0 => single linear, >0 => 2-layer MLP
        dropout: float = 0.0,
        init_bias: float = 2.0,  # sigmoid(init_bias) ~ 0.88 -> near-pass-through init
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mode = (mode or "scalar").lower()
        self.hidden_dim = int(hidden_dim or 0)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        out_dim = 1 if self.mode == "scalar" else embedding_dim
        if self.hidden_dim > 0:
            self.net = nn.Sequential(
                create_norm(norm_type, embedding_dim),
                nn.Linear(embedding_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, out_dim),
            )
        else:
            self.net = nn.Sequential(
                create_norm(norm_type, embedding_dim),
                nn.Linear(embedding_dim, out_dim),
            )

        # Initialize last layer bias so gate starts near 1.0 (pass-through-ish)
        try:
            last = self.net[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                nn.init.constant_(last.bias, init_bias)
        except Exception:
            pass

    def forward(self, queries_emb: torch.Tensor, context_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries_emb: [B, N, D] query-side representations
            context_vectors: [B, N, D] attention output vectors to gate
        """
        gate_logits = self.net(queries_emb)
        gate = torch.sigmoid(gate_logits)
        gate = self.dropout(gate)
        return context_vectors * gate

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean centering).

    Matches common RMSNorm behavior used in recent LLMs. Scales inputs by the
    inverse RMS over the last dimension and applies a learned gain.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Root mean square over feature dimension
        rms = torch.sqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + self.eps)
        x_norm = (x / rms).to(x.dtype)
        return x_norm * self.weight

def create_norm(norm_type: str, dim: int) -> nn.Module:
    """Factory for normalization layers used across models.

    Args:
        norm_type: "layernorm" or "rmsnorm"
        dim: normalized dimension
    """
    norm_type = (norm_type or "layernorm").lower()
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    # default
    return nn.LayerNorm(dim)

class TableSchemaGate(nn.Module):
    """
    Lightweight table-schema conditioner for query/key routing.

    The module projects a pooled schema/sketch embedding into model space and uses
    it to multiplicatively gate table-side representations while preserving
    sequence length and downstream tensor contracts.
    """
    def __init__(self, embedding_dim: int, norm_type: str = "layernorm") -> None:
        super().__init__()
        self.schema_norm = create_norm(norm_type, embedding_dim)
        self.schema_proj = nn.Linear(embedding_dim, embedding_dim)

        # Zero-init keeps the conditioner as an exact identity at startup.
        nn.init.zeros_(self.schema_proj.weight)
        if self.schema_proj.bias is not None:
            nn.init.zeros_(self.schema_proj.bias)

    def forward(self, inputs: torch.Tensor, schema_embeddings: Optional[torch.Tensor]) -> torch.Tensor:
        if schema_embeddings is None:
            return inputs

        if schema_embeddings.dim() == 1:
            schema_embeddings = schema_embeddings.unsqueeze(0)
        if schema_embeddings.dim() == 2:
            schema_embeddings = schema_embeddings.unsqueeze(0)

        schema_embeddings = self.schema_norm(schema_embeddings)
        schema_delta = torch.tanh(self.schema_proj(schema_embeddings))

        valid_schema_mask = (schema_embeddings.abs().sum(dim=-1, keepdim=True) > 0).to(schema_delta.dtype)
        pooled_schema_delta = (schema_delta * valid_schema_mask).sum(dim=1, keepdim=True)
        pooled_schema_denominator = valid_schema_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        schema_delta = pooled_schema_delta / pooled_schema_denominator

        return inputs * (1.0 + schema_delta.to(dtype=inputs.dtype))

class FeedForward(nn.Module):
    """
    Qwen3-style gated feed-forward (SwiGLU) without dropout.

    Implements a projection to an expanded hidden dimension using two
    parallel linear layers: a gate projection and an up projection.
    The gated activation uses SiLU (SwiGLU): silu(gate(x)) * up(x),
    followed by a linear down projection back to the model dimension.

    Notes:
    - No dropout (as requested).
    - By default uses the common 2/3 adjustment for SwiGLU inner size:
      inner_dim = int(2/3 * (multiplier * d_model)).
    - Can optionally apply a pre-normalization inside the module for
      cases like refinement blocks that previously included an internal
      LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        multiplier: float = 4.0,
        use_swiglu_inner_2over3: bool = True,
        with_pre_norm: bool = False,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()

        raw_inner_dim = int(multiplier * d_model)
        if use_swiglu_inner_2over3:
            inner_dim = max(1, (2 * raw_inner_dim) // 3)
        else:
            inner_dim = raw_inner_dim

        self.with_pre_norm = with_pre_norm
        self.pre_norm = create_norm(norm_type, d_model) if with_pre_norm else nn.Identity()

        self.gate_proj = nn.Linear(d_model, inner_dim)
        self.up_proj = nn.Linear(d_model, inner_dim)
        self.down_proj = nn.Linear(inner_dim, d_model)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normed = self.pre_norm(x)
        gated = self.activation(self.gate_proj(x_normed))
        up = self.up_proj(x_normed)
        hidden = gated * up
        return self.down_proj(hidden)

class SelfAttentionBlock(nn.Module):
    """
    Self-attention block with residual connections following Transformer practices.
    
    This module applies self-attention within a sequence (rows or sentences)
    to create more contextual and distinctive representations before cross-attention.
    This helps prevent attention collapse by pre-conditioning the inputs.
    
    Architecture:
    1. LayerNorm  Multi-Head Self-Attention  Residual Connection
    2. LayerNorm  FFN  Residual Connection
    """
    def __init__(self, 
                 embedding_dim: int, 
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_lora: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.1,
                 init_method: str = "xavier_uniform",
                 init_method_params: dict = None,
                 norm_type: str = "layernorm"):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate for attention and FFN
            use_lora: Whether to use LoRA for attention projections
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            init_method: Initialization method for attention weights
            init_method_params: Parameters for initialization method
        """
        super(SelfAttentionBlock, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_lora = use_lora
        self.init_method = init_method
        self.init_method_params = init_method_params or {}
        
        assert embedding_dim % num_heads == 0, f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        
        # Self-attention projections
        if use_lora:
            print(f"Using LoRA for self-attention (rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})")
            self.q_proj = LoRALinear(embedding_dim, embedding_dim, 
                                   rank=lora_rank, lora_alpha=lora_alpha, 
                                   lora_dropout=lora_dropout, bias=True)
            self.k_proj = LoRALinear(embedding_dim, embedding_dim, 
                                   rank=lora_rank, lora_alpha=lora_alpha, 
                                   lora_dropout=lora_dropout, bias=True)
            self.v_proj = LoRALinear(embedding_dim, embedding_dim, 
                                   rank=lora_rank, lora_alpha=lora_alpha, 
                                   lora_dropout=lora_dropout, bias=True)
            self.out_proj = LoRALinear(embedding_dim, embedding_dim, 
                                     rank=lora_rank, lora_alpha=lora_alpha, 
                                     lora_dropout=lora_dropout, bias=True)
        else:
            print("Using standard linear layers for self-attention")
            self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        
        # Layer normalizations (pre-norm pattern)
        self.attention_norm = create_norm(norm_type, embedding_dim)
        self.ffn_norm = create_norm(norm_type, embedding_dim)
        
        # Feed-forward network (Qwen3-style SwiGLU FFN, no dropout)
        self.ffn = FeedForward(d_model=embedding_dim, multiplier=4.0, use_swiglu_inner_2over3=True, with_pre_norm=False, norm_type=norm_type)
        
        # Attention dropout
        self.attention_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        if not use_lora:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize self-attention weights using the global initialization method."""
        print(f" Initializing self-attention with method: {self.init_method}")

        # Use the centralized initialization system for Q, K, V to match args
        from initialization import initialize_attention_weights
        initialize_attention_weights(
            layers=[self.q_proj, self.k_proj, self.v_proj],
            attention_dim=self.embedding_dim,
            method=self.init_method,
            method_params=self.init_method_params,
        )

        # Initialize output projection using the same selected method
        if self.init_method == "orthogonal":
            nn.init.orthogonal_(self.out_proj.weight, gain=1.0)
        elif self.init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.out_proj.weight, mode='fan_in', nonlinearity='linear')
        elif self.init_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        else:
            # Default to orthogonal when a method does not define linear-out specifics
            nn.init.orthogonal_(self.out_proj.weight, gain=1.0)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        
        # Add identity matrix to Q and K weights to encourage self-attention
        with torch.no_grad():
            # For multi-head attention, we need to handle the identity matrix properly
            if self.embedding_dim % self.num_heads == 0:
                head_dim = self.embedding_dim // self.num_heads
                
                # Add identity matrix to each head's weights
                for head in range(self.num_heads):
                    start_idx = head * head_dim
                    end_idx = (head + 1) * head_dim
                    
                    # CRITICAL: Further increase identity strength for strong diagonal attention
                    identity_strength = 1.5  # Increased from 0.5 to 1.5 for stronger self-focus
                    
                    # For Q projection - add stronger identity
                    if start_idx < self.q_proj.weight.size(0) and end_idx <= self.q_proj.weight.size(1):
                        self.q_proj.weight[start_idx:end_idx, start_idx:end_idx] += torch.eye(head_dim) * identity_strength
                    
                    # For K projection - add stronger identity
                    if start_idx < self.k_proj.weight.size(0) and end_idx <= self.k_proj.weight.size(1):
                        self.k_proj.weight[start_idx:end_idx, start_idx:end_idx] += torch.eye(head_dim) * identity_strength
        
        print(f" Added identity bias (strength={1.5}) to encourage self-attention patterns")
        
        # Initialize FFN layers (unified FeedForward module) using selected method
        if isinstance(self.ffn, FeedForward):
            if self.init_method == "orthogonal":
                nn.init.orthogonal_(self.ffn.gate_proj.weight, gain=1.0)
                nn.init.orthogonal_(self.ffn.up_proj.weight, gain=1.0)
                nn.init.orthogonal_(self.ffn.down_proj.weight, gain=1.0)
            elif self.init_method == "kaiming_uniform":
                nn.init.kaiming_uniform_(self.ffn.gate_proj.weight, mode='fan_in', nonlinearity='linear')
                nn.init.kaiming_uniform_(self.ffn.up_proj.weight, mode='fan_in', nonlinearity='linear')
                nn.init.kaiming_uniform_(self.ffn.down_proj.weight, mode='fan_in', nonlinearity='linear')
            elif self.init_method == "xavier_uniform":
                nn.init.xavier_uniform_(self.ffn.gate_proj.weight, gain=1.0)
                nn.init.xavier_uniform_(self.ffn.up_proj.weight, gain=1.0)
                nn.init.xavier_uniform_(self.ffn.down_proj.weight, gain=1.0)
            else:
                # Default to orthogonal for unsupported methods
                nn.init.orthogonal_(self.ffn.gate_proj.weight, gain=1.0)
                nn.init.orthogonal_(self.ffn.up_proj.weight, gain=1.0)
                nn.init.orthogonal_(self.ffn.down_proj.weight, gain=1.0)

            if self.ffn.gate_proj.bias is not None:
                nn.init.zeros_(self.ffn.gate_proj.bias)
            if self.ffn.up_proj.bias is not None:
                nn.init.zeros_(self.ffn.up_proj.bias)
            if self.ffn.down_proj.bias is not None:
                nn.init.zeros_(self.ffn.down_proj.bias)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply self-attention with residual connections.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # =================== SELF-ATTENTION WITH RESIDUAL ===================
        # Pre-norm pattern: norm first, then attention, then residual
        normed_x = self.attention_norm(x)
        
        # Multi-head self-attention
        q = self.q_proj(normed_x)  # [batch, seq_len, embedding_dim]
        k = self.k_proj(normed_x)  # [batch, seq_len, embedding_dim]
        v = self.v_proj(normed_x)  # [batch, seq_len, embedding_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, heads, seq_len, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, heads, seq_len, seq_len]
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)  # [batch, heads, seq_len, head_dim]
        
        # Reshape back to original dimensions
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        
        # Output projection
        attention_output = self.out_proj(attention_output)
        
        # RESIDUAL CONNECTION: Add original input to attention output
        x = x + attention_output
        
        # =================== FFN WITH RESIDUAL ===================
        # Pre-norm pattern: norm first, then FFN, then residual
        normed_x = self.ffn_norm(x)
        ffn_output = self.ffn(normed_x)
        
        # RESIDUAL CONNECTION: Add input to FFN output
        x = x + ffn_output
        
        return x

class LatentBottleneck(nn.Module):
    """
    Perceiver-style latent bottleneck to regularize inputs before cross-attention.

    Steps:
    1) Latents attend to inputs (Q=latents, K/V=inputs)  updated latents
    2) Inputs attend to updated latents (Q=inputs, K/V=latents)  bottlenecked inputs
    """
    def __init__(
        self,
        embedding_dim: int,
        num_latents: int = 64,
        dropout: float = 0.0,
        init_method: str = "xavier_uniform",
        init_method_params: Optional[dict] = None,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_latents = num_latents
        self.dropout = nn.Dropout(dropout)
        self.init_method = init_method
        self.init_method_params = init_method_params or {}

        # Learnable latent array [1, L, D]
        self.latents = nn.Parameter(torch.randn(1, num_latents, embedding_dim) * 0.02)

        # Projections for Latents -> Inputs attention
        self.latent_q = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.input_k = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.input_v = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # Projections for Inputs -> Latents attention
        self.input_q = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.latent_k = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.latent_v = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # Normalization and FFN for stability
        self.latent_norm1 = create_norm(norm_type, embedding_dim)
        self.input_norm1 = create_norm(norm_type, embedding_dim)
        self.latent_norm2 = create_norm(norm_type, embedding_dim)
        self.input_norm2 = create_norm(norm_type, embedding_dim)

        self.latent_ffn = FeedForward(d_model=embedding_dim, multiplier=2.0, use_swiglu_inner_2over3=True, with_pre_norm=False, norm_type=norm_type)
        self.input_ffn = FeedForward(d_model=embedding_dim, multiplier=2.0, use_swiglu_inner_2over3=True, with_pre_norm=False, norm_type=norm_type)

        # Initialize weights for attention projections
        try:
            initialize_attention_weights(
                layers=[
                    self.latent_q, self.input_k, self.input_v,
                    self.input_q, self.latent_k, self.latent_v,
                ],
                attention_dim=self.embedding_dim,
                method=self.init_method,
                method_params=self.init_method_params,
            )
        except Exception:
            for layer in [
                self.latent_q, self.input_k, self.input_v,
                self.input_q, self.latent_k, self.latent_v,
            ]:
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _scaled_dot(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)

        # Expand latents across batch
        latents = self.latents.expand(batch_size, -1, -1)

        # Stage 1: Latents attend to inputs
        norm_latents = self.latent_norm1(latents)
        norm_inputs = self.input_norm1(inputs)

        q_lat = self.latent_q(norm_latents)           # [B, L, D]
        k_in = self.input_k(norm_inputs)              # [B, N, D]
        v_in = self.input_v(norm_inputs)              # [B, N, D]
        attn_scores_li = self._scaled_dot(q_lat, k_in)
        attn_weights_li = F.softmax(attn_scores_li, dim=-1)
        attn_weights_li = self.dropout(attn_weights_li)
        updated_latents = torch.matmul(attn_weights_li, v_in)    # [B, L, D]

        # Residual + FFN on latents
        latents = latents + updated_latents
        latents = latents + self.latent_ffn(self.latent_norm2(latents))

        # Stage 2: Inputs attend to updated latents
        norm_inputs2 = self.input_norm1(inputs)
        norm_latents2 = self.latent_norm1(latents)

        q_in = self.input_q(norm_inputs2)        # [B, N, D]
        k_lat = self.latent_k(norm_latents2)     # [B, L, D]
        v_lat = self.latent_v(norm_latents2)     # [B, L, D]
        attn_scores_il = self._scaled_dot(q_in, k_lat)
        attn_weights_il = F.softmax(attn_scores_il, dim=-1)
        attn_weights_il = self.dropout(attn_weights_il)
        bottlenecked_inputs = torch.matmul(attn_weights_il, v_lat)  # [B, N, D]

        # Residual + FFN on inputs
        outputs = inputs + bottlenecked_inputs
        outputs = outputs + self.input_ffn(self.input_norm2(outputs))

        return outputs

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning.
    
    This replaces a linear layer W with W + B*A where:
    - W is the frozen pre-trained weight matrix
    - B and A are low-rank matrices (rank << hidden_dim)
    - Only B and A are trained, keeping W frozen
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 rank: int = 16, 
                 lora_alpha: float = 32.0, 
                 lora_dropout: float = 0.1,
                 bias: bool = True):
        super(LoRALinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        # Frozen base linear layer (initialized normally)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout for LoRA
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else None
        
        # Initialize LoRA weights
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA parameters using Kaiming uniform for A and zeros for B."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base_output + LoRA_output."""
        # Base layer output (frozen)
        base_output = self.base_layer(x)
        
        # LoRA adaptation
        if self.dropout is not None:
            lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        else:
            lora_output = x @ self.lora_A @ self.lora_B * self.scaling
            
        return base_output + lora_output

class CrossAttentionModule(nn.Module):
    """
    Cross-attention module to compute attention between table rows and text sentences.
    Supports LoRA for efficient adaptation while keeping base parameters frozen.
    **NOTE**: This module only supports "standard" attention type. For advanced attention
    mechanisms (top_k_sparse, windowed, threshold), use the unidirectional model which
    will automatically select the appropriate attention module.
    """
    def __init__(self, 
                 embedding_dim: int, 
                 attention_dim: int = None,
                 use_lora: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.1,
                 attention_type: str = "standard",  # Added attention_type parameter for compatibility
                 init_method: str = "xavier_uniform",  # NEW: Added initialization method
                 init_method_params: dict = None,  # NEW: Added initialization parameters
                 norm_type: str = "layernorm",
                 use_qk_rmsnorm: bool = False,
                 # Gated attention overlay (post-SDPA gating)
                 use_gated_attention: bool = False,
                 gated_attention_mode: str = "scalar",
                 gated_attention_hidden_dim: int = 0,
                 gated_attention_dropout: float = 0.0,
                 gated_attention_init_bias: float = 2.0,
                 # Temperature scaling control
                 disable_temperature: bool = False):
        super(CrossAttentionModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_lora = use_lora
        self.attention_type = attention_type  # Store for compatibility
        self.init_method = init_method  # NEW: Store initialization method
        self.init_method_params = init_method_params or {}  # NEW: Store initialization parameters
        self.use_qk_rmsnorm = use_qk_rmsnorm
        self.disable_temperature = disable_temperature
        self.use_gated_attention = use_gated_attention
        self.gated_attention_mode = gated_attention_mode
        self.gated_attention_hidden_dim = gated_attention_hidden_dim
        self.gated_attention_dropout = gated_attention_dropout
        self.gated_attention_init_bias = gated_attention_init_bias
        self.norm_type = norm_type
        
        # Validate attention_type - this module only supports "standard"
        if attention_type != "standard":
            print(f"  Warning: CrossAttentionModule only supports 'standard' attention type. "
                  f"Received '{attention_type}'. Using 'standard' instead.")
            self.attention_type = "standard"
        
        # Use full attention dimension for more expressive attention
        self.attention_dim = attention_dim if attention_dim is not None else embedding_dim
        
        # Projection matrices for query, key, and value
        if use_lora:
            print(f"Using LoRA for cross-attention (rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})")
            self.W_Q = LoRALinear(embedding_dim, self.attention_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
            self.W_K = LoRALinear(embedding_dim, self.attention_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
            self.W_V = LoRALinear(embedding_dim, embedding_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
        else:
            print("Using standard linear layers for cross-attention")
            self.W_Q = nn.Linear(embedding_dim, self.attention_dim, bias=True)
            self.W_K = nn.Linear(embedding_dim, self.attention_dim, bias=True)
            self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=True)
        
        # Optional Q/K RMSNorm
        if self.use_qk_rmsnorm:
            self.q_norm = RMSNorm(self.attention_dim)
            self.k_norm = RMSNorm(self.attention_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        # Learnable temperature parameter - start lower to allow peaky attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

        # Optional post-SDPA output gate
        if self.use_gated_attention:
            self.attention_output_gate = AttentionOutputGate(
                embedding_dim=self.embedding_dim,
                mode=self.gated_attention_mode,
                hidden_dim=self.gated_attention_hidden_dim,
                dropout=self.gated_attention_dropout,
                init_bias=self.gated_attention_init_bias,
                norm_type=self.norm_type,
            )
        else:
            self.attention_output_gate = None
        
        # Initialize weights
        if not use_lora:
            self._init_weights()
    
    def _init_weights(self):
        """
        Initialize the weights using the specified initialization method.
        """
        # Create list of layers for initialization
        layers = [self.W_Q, self.W_K, self.W_V]
        
        # Apply the selected initialization method - NO FALLBACK!
        print(f" Initializing cross-attention with method: {self.init_method}")
        from initialization import initialize_attention_weights
        initialize_attention_weights(
            layers=layers,
            attention_dim=self.attention_dim,
            method=self.init_method,
            method_params=self.init_method_params
        )
        print(f" Successfully applied {self.init_method} initialization")
    
    def forward(self, 
                rows_embeddings: torch.Tensor, 
                sentences_embeddings: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cross-attention between table rows and text sentences.
        
        Args:
            rows_embeddings: Tensor of shape [batch_size, num_rows, embedding_dim]
            sentences_embeddings: Tensor of shape [batch_size, num_sentences, embedding_dim]
            
        Returns:
            Tuple containing:
            - attention_weights: Tensor of shape [batch_size, num_rows, num_sentences] (softmax normalized)
            - context_vectors: Tensor of shape [batch_size, num_rows, embedding_dim]
        """
        # Project embeddings
        Q = self.W_Q(rows_embeddings)  # [batch_size, num_rows, attention_dim]
        K = self.W_K(sentences_embeddings)  # [batch_size, num_sentences, attention_dim]
        V = self.W_V(sentences_embeddings)  # [batch_size, num_sentences, embedding_dim]
        
        # Apply optional Q/K RMSNorm
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_rows, num_sentences]

        # Score centering (per-row) and scaling by sqrt(d)
        attention_scores = attention_scores - attention_scores.mean(dim=-1, keepdim=True)
        attention_scores = attention_scores / (self.attention_dim ** 0.5)
        
        # Apply temperature scaling (clip temperature to avoid extreme values)
        if not self.disable_temperature:
            temperature = torch.clamp(self.temperature, min=0.2, max=2.0)
            attention_scores = attention_scores / temperature
        
        # Create padding mask from inputs if not provided (treat all-zero key vectors as padding)
        if key_padding_mask is None:
            key_padding_mask = (sentences_embeddings.abs().sum(dim=-1) > 0)  # [batch_size, num_sentences]
        
        # Mask out padded keys
        if key_padding_mask is not None:
            mask_3d = ~key_padding_mask.unsqueeze(1)  # [batch_size, 1, num_sentences]
            attention_scores = attention_scores.masked_fill(mask_3d, -1e9)

        # Apply selected attention activation (softmax/entmax)
        activation = getattr(self, 'attention_activation', 'softmax') if hasattr(self, 'attention_activation') else 'softmax'
        alpha = getattr(self, 'attention_alpha', 1.5) if hasattr(self, 'attention_alpha') else 1.5
        attention_weights = apply_attention_activation(attention_scores, dim=-1, name=activation, alpha=alpha)
        
        # Compute context vectors
        context_vectors = torch.bmm(attention_weights, V)  # [batch_size, num_rows, embedding_dim]

        # Optional post-SDPA gating on attention output (query-dependent)
        if self.attention_output_gate is not None:
            context_vectors = self.attention_output_gate(rows_embeddings, context_vectors)
        
        return attention_weights, context_vectors
    
    def attention_entropy_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization loss to encourage diverse attention patterns.
        Higher entropy means more diverse attention.
        
        Args:
            attention_weights: Tensor of shape [batch_size, num_rows, num_sentences]
            
        Returns:
            Entropy loss (negative entropy, so minimizing this maximizes entropy)
        """
        # Compute entropy for each row's attention distribution
        # entropy = -sum(p * log(p))
        log_weights = torch.log(attention_weights + 1e-10)  # Add small epsilon for numerical stability
        entropy = -torch.sum(attention_weights * log_weights, dim=-1)  # [batch_size, num_rows]
        
        # Return negative entropy (so minimizing this loss maximizes entropy/diversity)
        return -torch.mean(entropy)

class LatentCrossAttention(nn.Module):
    """
    Single-head latent cross-attention bridge.

    Two steps per call:
    1) Latents attend to keys (K/V = keys_emb)  latent_context [B, L, D]
    2) Queries attend to latent_context (K/V = latent_context)  outputs [B, N, D]

    Returns (context_vectors, effective_attention_weights) with weights in [B, N, M]
    computed as (rowslatents) @ (latentskeys).
    """
    def __init__(
        self,
        embedding_dim: int,
        num_latents: int = 64,
        dropout: float = 0.0,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        init_method: str = "xavier_uniform",
        init_method_params: Optional[dict] = None,
        norm_type: str = "layernorm",
        # Gated attention overlay (post-SDPA gating)
        use_gated_attention: bool = False,
        gated_attention_mode: str = "scalar",
        gated_attention_hidden_dim: int = 0,
        gated_attention_dropout: float = 0.0,
        gated_attention_init_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_latents = num_latents
        self.dropout = nn.Dropout(dropout)
        self.use_lora = use_lora
        self.init_method = init_method
        self.init_method_params = init_method_params or {}
        self.use_gated_attention = use_gated_attention
        self.gated_attention_mode = gated_attention_mode
        self.gated_attention_hidden_dim = gated_attention_hidden_dim
        self.gated_attention_dropout = gated_attention_dropout
        self.gated_attention_init_bias = gated_attention_init_bias
        self.norm_type = norm_type

        # Learnable latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, embedding_dim) * 0.02)

        if self.use_gated_attention:
            self.attention_output_gate = AttentionOutputGate(
                embedding_dim=self.embedding_dim,
                mode=self.gated_attention_mode,
                hidden_dim=self.gated_attention_hidden_dim,
                dropout=self.gated_attention_dropout,
                init_bias=self.gated_attention_init_bias,
                norm_type=self.norm_type,
            )
        else:
            self.attention_output_gate = None

        # Projections for step 1 (Latents -> Keys)
        if use_lora:
            self.latent_q = LoRALinear(embedding_dim, embedding_dim, rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=True)
            self.key_k = LoRALinear(embedding_dim, embedding_dim, rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=True)
            self.key_v = LoRALinear(embedding_dim, embedding_dim, rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=True)
        else:
            self.latent_q = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.key_k = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.key_v = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # Projections for step 2 (Queries -> Latent context)
        if use_lora:
            self.query_q = LoRALinear(embedding_dim, embedding_dim, rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=True)
            self.lat_k = LoRALinear(embedding_dim, embedding_dim, rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=True)
            self.lat_v = LoRALinear(embedding_dim, embedding_dim, rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=True)
        else:
            self.query_q = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.lat_k = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.lat_v = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # Norms
        self.latent_norm = create_norm(norm_type, embedding_dim)
        self.key_norm = create_norm(norm_type, embedding_dim)
        self.query_norm = create_norm(norm_type, embedding_dim)

        # Initialize projections
        try:
            initialize_attention_weights(
                layers=[self.latent_q, self.key_k, self.key_v, self.query_q, self.lat_k, self.lat_v],
                attention_dim=self.embedding_dim,
                method=self.init_method,
                method_params=self.init_method_params,
            )
        except Exception:
            for layer in [self.latent_q, self.key_k, self.key_v, self.query_q, self.lat_k, self.lat_v]:
                if hasattr(layer, 'weight'):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _scaled_dot(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

    def forward(self, queries_emb: torch.Tensor, keys_emb: torch.Tensor, values_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_queries, _ = queries_emb.shape
        batch_size, num_keys, _ = keys_emb.shape

        # Expand latents
        latents = self.latents.expand(batch_size, -1, -1)

        # Step 1: Latents attend to keys
        norm_latents = self.latent_norm(latents)
        norm_keys = self.key_norm(keys_emb)
        q_lat = self.latent_q(norm_latents)               # [B, L, D]
        k_key = self.key_k(norm_keys)                     # [B, M, D]
        v_key = self.key_v(norm_keys)                     # [B, M, D]
        scores_lk = self._scaled_dot(q_lat, k_key)        # [B, L, M]
        attn_lk = F.softmax(scores_lk, dim=-1)            # per-latent distribution over keys
        attn_lk = self.dropout(attn_lk)
        latent_context = torch.matmul(attn_lk, v_key)     # [B, L, D]

        # Step 2: Queries attend to latent_context
        norm_queries = self.query_norm(queries_emb)
        q_q = self.query_q(norm_queries)                  # [B, N, D]
        k_lat = self.lat_k(latent_context)                # [B, L, D]
        v_lat = self.lat_v(latent_context)                # [B, L, D]
        scores_ql = self._scaled_dot(q_q, k_lat)          # [B, N, L]
        attn_ql = F.softmax(scores_ql, dim=-1)            # per-query distribution over latents
        attn_ql = self.dropout(attn_ql)
        outputs = torch.matmul(attn_ql, v_lat)            # [B, N, D]

        # Compose effective attention over keys: [B, N, M]
        effective_attn = torch.matmul(attn_ql, attn_lk)   # [B, N, M]

        if self.attention_output_gate is not None:
            outputs = self.attention_output_gate(queries_emb, outputs)

        return outputs, effective_attn

class TableTextEmbeddingModel(nn.Module):
    """
    Model for table-text embedding using cross-attention.
    Supports LoRA for efficient cross-attention adaptation.
    **NEW**: Now supports advanced attention mechanisms (Top-K Sparse, Windowed, Threshold-Based)
    **NEW**: Supports attention direction reversal (row_to_sentence or sentence_to_row)
    """
    def __init__(self, 
                 sentence_encoder: SentenceTransformer, 
                 embedding_dim: int, 
                 native_embedding_dim: int = None,
                 trainable_encoder: bool = False,
                 use_cross_attention_lora: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.1,
                 top_k: int = 3,
                 # **NEW**: Advanced attention mechanism parameters
                 attention_type: str = "standard",  # "standard", "top_k_sparse", "windowed", "threshold"
                 sparse_top_k: int = 3,
                 window_size: int = 5,
                 threshold_base: float = 0.1,
                 init_method: str = "xavier_uniform",
                 init_method_params: dict = None,
                 norm_type: str = "layernorm",
                 # **NEW**: Attention direction control
                 attention_direction: str = "row_to_sentence",  # "row_to_sentence" or "sentence_to_row"
                 # **NEW**: Latent bottleneck parameters
                 use_latent_bottleneck: bool = False,
                 latent_num: int = 64,
                 latent_dropout: float = 0.0,
                 # Gated attention overlay (post-SDPA gating)
                 use_gated_attention: bool = False,
                 gated_attention_mode: str = "scalar",
                 gated_attention_hidden_dim: int = 0,
                 gated_attention_dropout: float = 0.0,
                 gated_attention_init_bias: float = 2.0,
                 # Temperature scaling control
                 disable_temperature: bool = False,
                 # Skip FFN option (use raw cross-attention output)
                 skip_ffn: bool = False,
                 # Verbosity control
                 verbose: bool = True):
        """
        Args:
            attention_type: Type of attention mechanism:
                - "standard": Regular scaled dot-product attention (original CrossAttentionModule)
                - "top_k_sparse": Top-K sparse attention (prevents attention collapse)
                - "windowed": Content-based windowed attention  
                - "threshold": Threshold-based semantic filtering
            sparse_top_k: Number of top connections to keep in sparse attention
            window_size: Window size for windowed attention
            threshold_base: Base threshold for threshold attention
            init_method: Initialization method for attention weights
            init_method_params: Parameters for initialization method
            verbose: Whether to print initialization messages
        """
        super(TableTextEmbeddingModel, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.embedding_dim = embedding_dim
        self.use_cross_attention_lora = use_cross_attention_lora
        # Expose normalization type for downstream tools/visualizations
        self.norm_type = norm_type
        self.verbose = verbose
        self.top_k = top_k  # Store configurable top_k parameter for aggregation
        self.native_embedding_dim = native_embedding_dim or embedding_dim
        self.use_latent_bottleneck = use_latent_bottleneck
        self.latent_num = latent_num
        self.latent_dropout = latent_dropout
        self.use_gated_attention = use_gated_attention
        self.gated_attention_mode = gated_attention_mode
        self.gated_attention_hidden_dim = gated_attention_hidden_dim
        self.gated_attention_dropout = gated_attention_dropout
        self.gated_attention_init_bias = gated_attention_init_bias
        self.disable_temperature = disable_temperature
        self.skip_ffn = skip_ffn
        
        # **NEW**: Store attention mechanism configuration
        self.attention_type = attention_type
        self.sparse_top_k = sparse_top_k
        self.window_size = window_size
        self.threshold_base = threshold_base
        self.init_method = init_method
        self.init_method_params = init_method_params or {}
        
        # **NEW**: Store attention direction (row_to_sentence or sentence_to_row)
        self.attention_direction = attention_direction
        if self.verbose:
            if attention_direction == "sentence_to_row":
                print(f" Attention direction: SENTENCE -> ROW (sentences query rows)")
            else:
                print(f" Attention direction: ROW -> SENTENCE (rows query sentences)")
        
        # =====================================================================
        # Handle encoder trainability with PEFT/QLoRA awareness
        # =====================================================================
        # Check for actual PEFT parameters, since Unsloth FastModel or huggingface 
        # might inject empty or dummy peft_config dicts that evaluate to True.
        encoder_has_peft = any('lora' in name.lower() for name, _ in self.sentence_encoder.named_parameters())
        
        if encoder_has_peft:
            # PEFT is managing the encoder - DON'T override its frozen/trainable state
            if self.verbose:
                print("   [INFO] PEFT/QLoRA detected on sentence encoder - preserving PEFT freeze state")
                enc_trainable = sum(p.numel() for p in self.sentence_encoder.parameters() if p.requires_grad)
                enc_total = sum(p.numel() for p in self.sentence_encoder.parameters())
                print(f"      Encoder params: {enc_trainable:,}/{enc_total:,} trainable ({enc_trainable/enc_total*100:.2f}%)")
        elif not trainable_encoder:
            for param in self.sentence_encoder.parameters():
                param.requires_grad = False
            if self.verbose:
                print("Sentence encoder frozen (not trainable)")
        else:
            skipped_non_float_params = 0
            for param in self.sentence_encoder.parameters():
                if torch.is_floating_point(param) or torch.is_complex(param):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    skipped_non_float_params += param.numel()
            if self.verbose:
                print("Sentence encoder parameters are trainable (full fine-tuning)")
                if skipped_non_float_params > 0:
                    print(f"   Skipped {skipped_non_float_params:,} non-floating encoder params during unfreeze")
        
        if self.verbose:
            print(f"Model initialized with top_k={self.top_k}")
            print(f" Using {attention_type} attention mechanism in unidirectional model")
            if self.use_latent_bottleneck:
                print(f" Using latent bottleneck (L={self.latent_num}) before cross-attention")
        
        # Detect sentence encoder dtype for compatibility
        try:
            encoder_dtype = next(self.sentence_encoder.parameters()).dtype
            if self.verbose:
                print(f"Sentence encoder dtype detected: {encoder_dtype}")
        except:
            encoder_dtype = torch.bfloat16
            if self.verbose:
                print("Could not detect sentence encoder dtype, defaulting to bfloat16")
        
        # **NEW**: Dynamic attention module selection based on attention_type
        if attention_type == "top_k_sparse":
            if self.verbose:
                print(f" Initializing Top-K Sparse Attention (k={sparse_top_k}) for unidirectional model")
            self.cross_attention = TopKSparseAttention(
                embedding_dim=embedding_dim,
                attention_dim=embedding_dim,  # Use full embedding dim for unidirectional
                top_k=sparse_top_k,
                use_lora=use_cross_attention_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                use_gated_attention=self.use_gated_attention,
                gated_attention_mode=self.gated_attention_mode,
                gated_attention_hidden_dim=self.gated_attention_hidden_dim,
                gated_attention_dropout=self.gated_attention_dropout,
                gated_attention_init_bias=self.gated_attention_init_bias,
                norm_type=norm_type,
                disable_temperature=self.disable_temperature,
            )
            
        elif attention_type == "windowed":
            if self.verbose:
                print(f" Initializing Windowed Attention (window_size={window_size}) for unidirectional model")
            self.cross_attention = WindowedCrossAttention(
                embedding_dim=embedding_dim,
                attention_dim=embedding_dim,  # Use full embedding dim for unidirectional
                window_size=window_size,
                use_lora=use_cross_attention_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                use_gated_attention=self.use_gated_attention,
                gated_attention_mode=self.gated_attention_mode,
                gated_attention_hidden_dim=self.gated_attention_hidden_dim,
                gated_attention_dropout=self.gated_attention_dropout,
                gated_attention_init_bias=self.gated_attention_init_bias,
                norm_type=norm_type,
                disable_temperature=self.disable_temperature,
            )
            
        elif attention_type == "threshold":
            if self.verbose:
                print(f" Initializing Threshold Attention (threshold={threshold_base}) for unidirectional model")
            self.cross_attention = ThresholdAttention(
                embedding_dim=embedding_dim,
                attention_dim=embedding_dim,  # Use full embedding dim for unidirectional
                base_threshold=threshold_base,
                use_lora=use_cross_attention_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                use_gated_attention=self.use_gated_attention,
                gated_attention_mode=self.gated_attention_mode,
                gated_attention_hidden_dim=self.gated_attention_hidden_dim,
                gated_attention_dropout=self.gated_attention_dropout,
                gated_attention_init_bias=self.gated_attention_init_bias,
                norm_type=norm_type,
                disable_temperature=self.disable_temperature,
            )
            
        elif attention_type == "latent_cross":
            if self.verbose:
                print(f" Initializing Latent Cross-Attention (L={latent_num}) for unidirectional model")
            self.cross_attention = LatentCrossAttention(
                embedding_dim=embedding_dim,
                num_latents=latent_num,
                dropout=self.latent_dropout if hasattr(self, 'latent_dropout') else 0.0,
                use_lora=use_cross_attention_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                norm_type=norm_type,
                use_gated_attention=self.use_gated_attention,
                gated_attention_mode=self.gated_attention_mode,
                gated_attention_hidden_dim=self.gated_attention_hidden_dim,
                gated_attention_dropout=self.gated_attention_dropout,
                gated_attention_init_bias=self.gated_attention_init_bias,
            )
        else:  # "standard" or any other value
            if self.verbose:
                print(" Using standard scaled dot-product attention (original CrossAttentionModule)")
            # Cross-attention module with optional LoRA (original implementation)
            self.cross_attention = CrossAttentionModule(
                embedding_dim=embedding_dim,
                use_lora=use_cross_attention_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                attention_type=attention_type,  # Pass attention_type for compatibility
                init_method=init_method,  # NEW: Pass initialization method
                init_method_params=init_method_params,  # NEW: Pass initialization parameters
                norm_type=norm_type,
                use_qk_rmsnorm=True,
                use_gated_attention=self.use_gated_attention,
                gated_attention_mode=self.gated_attention_mode,
                gated_attention_hidden_dim=self.gated_attention_hidden_dim,
                gated_attention_dropout=self.gated_attention_dropout,
                gated_attention_init_bias=self.gated_attention_init_bias,
                disable_temperature=self.disable_temperature,
            )
        
        # **NEW**: Optional Perceiver-style latent bottleneck before cross-attention
        if self.use_latent_bottleneck:
            self.row_latent_bottleneck = LatentBottleneck(
                embedding_dim=embedding_dim,
                num_latents=self.latent_num,
                dropout=self.latent_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                norm_type=norm_type,
            )
            self.sentence_latent_bottleneck = LatentBottleneck(
                embedding_dim=embedding_dim,
                num_latents=self.latent_num,
                dropout=self.latent_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                norm_type=norm_type,
            )

        # Layer normalization after attention (for standard transformer architecture)
        self.norm1 = create_norm(norm_type, embedding_dim)
        
        # Feed-forward network (Qwen3-style SwiGLU FFN, no dropout)
        self.feed_forward = FeedForward(d_model=embedding_dim, multiplier=4.0, use_swiglu_inner_2over3=True, with_pre_norm=False, norm_type=norm_type)
        
        # Layer normalization after feed-forward (for standard transformer architecture)
        self.norm2 = create_norm(norm_type, embedding_dim)
        
        # Final aggregation layer (linear, no sigmoid to avoid early saturation)
        self.final_aggregation = nn.Linear(embedding_dim, 1)
        
        # Convert all custom components to BFloat16 unconditionally
        print("Converting custom model components to BFloat16 (ensuring compatibility)...")
        self.cross_attention.to(dtype=torch.bfloat16)
        self.norm1.to(dtype=torch.bfloat16)
        self.feed_forward.to(dtype=torch.bfloat16)
        self.norm2.to(dtype=torch.bfloat16)
        self.final_aggregation.to(dtype=torch.bfloat16)
        if self.use_latent_bottleneck:
            self.row_latent_bottleneck.to(dtype=torch.bfloat16)
            self.sentence_latent_bottleneck.to(dtype=torch.bfloat16)

        # Optional dimension projection for embedding dim override (non-Matryoshka)
        if self.native_embedding_dim != self.embedding_dim:
            self.dim_projection = nn.Linear(self.native_embedding_dim, self.embedding_dim, bias=False)
            self.dim_projection.to(dtype=torch.bfloat16)
            if self.verbose:
                print(f"[INFO] Added dim projection: {self.native_embedding_dim} -> {self.embedding_dim}")
        else:
            self.dim_projection = None

        print(" Custom components converted to BFloat16")
    
    def encode_sentences(self, 
                        sentences: List[str], 
                        batch_size: int = 32, 
                        normalize: bool = True) -> torch.Tensor:
        """
        Encode a list of sentences using the sentence encoder.
        This method is suitable for both training and inference.
        Since the sentence encoder is frozen by default, embeddings won't have gradients.
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            normalize: Whether to normalize the embeddings
            
        Returns:
            Tensor of shape [len(sentences), embedding_dim] containing the embeddings
        """
        # Use gradient-preserving forward when encoder has trainable params
        encoder_has_grad = any(p.requires_grad for p in self.sentence_encoder.parameters())
        if encoder_has_grad:
            features = self.sentence_encoder.tokenize(sentences)
            encoder_param = next(self.sentence_encoder.parameters())
            device = encoder_param.device
            features = {k: v.to(device) for k, v in features.items()}
            use_bf16_autocast = device.type == "cuda" and encoder_param.dtype == torch.bfloat16
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16_autocast
                else nullcontext()
            )
            with autocast_context:
                outputs = self.sentence_encoder(features)
            if isinstance(outputs, dict) and 'sentence_embedding' in outputs:
                embeddings = outputs['sentence_embedding']
            elif hasattr(outputs, 'sentence_embedding'):
                embeddings = outputs.sentence_embedding
            elif isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict) and 'sentence_embedding' in outputs[0]:
                embeddings = outputs[0]['sentence_embedding']
            else:
                raise RuntimeError("Unexpected SentenceTransformer output; cannot extract sentence embeddings with grad")
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        else:
            embeddings = self.sentence_encoder.encode(
                sentences,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
        
        # Apply dimension projection if needed (non-Matryoshka override)
        if self.dim_projection is not None:
            embeddings = self.dim_projection(embeddings.to(dtype=self.dim_projection.weight.dtype))
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings
    
    def forward(self, 
               rows_embeddings: torch.Tensor, 
               sentences_embeddings: torch.Tensor,
               aggregation_method: str = "entropy_regularized") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass following standard transformer practices with proper
        layer normalization and residual connections.
        
        Standard transformer block structure:
        1. Attention
        2. Add & Norm (residual connection + layer norm)
        3. Feed-Forward
        4. Add & Norm (residual connection + layer norm)
        
        Args:
            rows_embeddings: Tensor of shape [batch_size, num_rows, embedding_dim]
            sentences_embeddings: Tensor of shape [batch_size, num_sentences, embedding_dim]
            aggregation_method: Method for aggregating row scores:
                - "mean": Original mean aggregation (problematic)
                - "top_k_sum": Sum the top-k scores (recommended)
                - "top_k_mean": Take mean of top-k scores
                - "weighted_top_k": Weighted combination of top-k scores
                - "max": Focus only on single best pair (most aggressive)
                - "attention_weighted": Use attention weights for aggregation
                - "sparse_top_k": Sparse top-k with zero masking
                - "entropy_regularized": Top-k with entropy bonus
            
        Returns:
            Tuple containing:
            - similarity_scores: Tensor of shape [batch_size] containing the similarity scores
            - attention_scores: Tensor of shape [batch_size, num_rows, num_sentences] containing the attention scores
        """
        # Pre-norm residual pattern (Qwen3-style):
        # 1) x_attn = x + Attn(LN(x))
        # 2) x_ffn  = x_attn + FFN(LN(x_attn))

        # Optional latent bottleneck before cross-attention
        if self.use_latent_bottleneck:
            rows_inputs = self.row_latent_bottleneck(rows_embeddings)
            sentences_inputs = self.sentence_latent_bottleneck(sentences_embeddings)
        else:
            rows_inputs = rows_embeddings
            sentences_inputs = sentences_embeddings

        # **NEW**: Support attention direction reversal
        # row_to_sentence: rows are queries, sentences are keys/values (original)
        # sentence_to_row: sentences are queries, rows are keys/values (reversed)
        if self.attention_direction == "sentence_to_row":
            # Reversed: sentences query rows
            query_inputs = sentences_inputs
            key_value_inputs = rows_inputs
            normed_queries = self.norm1(query_inputs)
        else:
            # Original: rows query sentences
            query_inputs = rows_inputs
            key_value_inputs = sentences_inputs
            normed_queries = self.norm1(query_inputs)

        # Handle different attention module interfaces
        if self.attention_type == "standard":
            # (queries, keys/values, mask) -> (attention_weights, context_vectors)
            if self.attention_direction == "sentence_to_row":
                attention_weights, context_vectors = self.cross_attention(
                    normed_queries, key_value_inputs,
                    key_padding_mask=(rows_embeddings.abs().sum(dim=-1) > 0)
                )
            else:
                attention_weights, context_vectors = self.cross_attention(
                    normed_queries, key_value_inputs,
                    key_padding_mask=(sentences_embeddings.abs().sum(dim=-1) > 0)
                )
        else:
            # (queries_emb, keys_emb, values_emb) -> (context_vectors, attention_weights)
            context_vectors, attention_weights = self.cross_attention(
                queries_emb=normed_queries,
                keys_emb=key_value_inputs,
                values_emb=key_value_inputs
            )

        # Residual after attention
        x_after_attn = query_inputs + context_vectors

        # Optionally skip FFN (use raw cross-attention output)
        if self.skip_ffn:
            final_context = x_after_attn
        else:
            # Pre-norm before FFN
            normed_after_attn = self.norm2(x_after_attn)
            ffn_output = self.feed_forward(normed_after_attn)
            # Residual after FFN
            final_context = x_after_attn + ffn_output
        
        # **NEW**: Handle different output dimensions based on attention direction
        # Check if we should use cosine similarity (like bidirectional model) for better zero-init compatibility
        use_cosine_scoring = getattr(self, 'use_cosine_scoring', False)
        
        if use_cosine_scoring:
            # Use cosine similarity between contextualized queries and original keys (like bidirectional model)
            # This provides more robust scoring when attention weights are zero-initialized
            if self.attention_direction == "sentence_to_row":
                # Contextualized sentences vs original rows
                final_context_norm = torch.nn.functional.normalize(final_context, p=2, dim=-1)
                rows_norm = torch.nn.functional.normalize(rows_embeddings, p=2, dim=-1)
                # [batch, num_sentences, dim] @ [batch, dim, num_rows] -> [batch, num_sentences, num_rows]
                pair_scores = torch.bmm(final_context_norm, rows_norm.transpose(1, 2))
            else:
                # Contextualized rows vs original sentences
                final_context_norm = torch.nn.functional.normalize(final_context, p=2, dim=-1)
                sentences_norm = torch.nn.functional.normalize(sentences_embeddings, p=2, dim=-1)
                # [batch, num_rows, dim] @ [batch, dim, num_sentences] -> [batch, num_rows, num_sentences]
                pair_scores = torch.bmm(final_context_norm, sentences_norm.transpose(1, 2))
            
            # Aggregate pair scores using the specified method (adapted from bidirectional model)
            similarity_scores = self._aggregate_pair_scores_cosine(pair_scores, attention_weights, aggregation_method)
        elif self.attention_direction == "sentence_to_row":
            # Contextualized sentences: [batch_size, num_sentences, embedding_dim]
            batch_size, num_sentences, embedding_dim = final_context.shape
            flat_context = final_context.reshape(-1, embedding_dim)
            flat_scores = self.final_aggregation(flat_context)
            sentence_scores = flat_scores.reshape(batch_size, num_sentences)
            # For aggregation, we aggregate over sentences
            similarity_scores = self._aggregate_scores(sentence_scores, attention_weights, aggregation_method)
        else:
            # Contextualized rows: [batch_size, num_rows, embedding_dim]
            batch_size, num_rows, embedding_dim = final_context.shape
            flat_context = final_context.reshape(-1, embedding_dim)
            flat_scores = self.final_aggregation(flat_context)
            row_scores = flat_scores.reshape(batch_size, num_rows)
            similarity_scores = self._aggregate_scores(row_scores, attention_weights, aggregation_method)
        
        return similarity_scores, attention_weights
    
    def get_contextualized_pair_scores(self, 
                                        rows_embeddings: torch.Tensor, 
                                        sentences_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get pair scores using CONTEXTUALIZED embeddings (after cross-attention + FFN).
        
        Supports both attention directions:
        - row_to_sentence: Contextualized rows vs original sentences
        - sentence_to_row: Original rows vs contextualized sentences
        
        Args:
            rows_embeddings: Tensor of shape [batch_size, num_rows, embedding_dim]
            sentences_embeddings: Tensor of shape [batch_size, num_sentences, embedding_dim]
            
        Returns:
            pair_scores: Tensor of shape [batch_size, num_rows, num_sentences] containing
                        cosine similarity between (contextualized, original) pairs
        """
        # Compute contextualized embeddings (same as forward, up to final_context)
        # CRITICAL: Ensure inputs match model component dtypes
        model_dtype = next(self.cross_attention.parameters()).dtype
        rows_embeddings = rows_embeddings.to(dtype=model_dtype)
        sentences_embeddings = sentences_embeddings.to(dtype=model_dtype)
        
        if self.use_latent_bottleneck:
            rows_inputs = self.row_latent_bottleneck(rows_embeddings)
            sentences_inputs = self.sentence_latent_bottleneck(sentences_embeddings)
        else:
            rows_inputs = rows_embeddings
            sentences_inputs = sentences_embeddings

        # **NEW**: Support attention direction reversal
        if self.attention_direction == "sentence_to_row":
            query_inputs = sentences_inputs
            key_value_inputs = rows_inputs
        else:
            query_inputs = rows_inputs
            key_value_inputs = sentences_inputs

        normed_queries = self.norm1(query_inputs)

        # Get context vectors from cross-attention
        if self.attention_type == "standard":
            if self.attention_direction == "sentence_to_row":
                attention_weights, context_vectors = self.cross_attention(
                    normed_queries, key_value_inputs,
                    key_padding_mask=(rows_embeddings.abs().sum(dim=-1) > 0)
                )
            else:
                attention_weights, context_vectors = self.cross_attention(
                    normed_queries, key_value_inputs,
                    key_padding_mask=(sentences_embeddings.abs().sum(dim=-1) > 0)
                )
        else:
            context_vectors, attention_weights = self.cross_attention(
                queries_emb=normed_queries,
                keys_emb=key_value_inputs,
                values_emb=key_value_inputs
            )

        # Residual after attention
        x_after_attn = query_inputs + context_vectors

        # Optionally skip FFN (use raw cross-attention output)
        if self.skip_ffn:
            contextualized = x_after_attn
        else:
            # Pre-norm before FFN
            normed_after_attn = self.norm2(x_after_attn)
            ffn_output = self.feed_forward(normed_after_attn)
            # Residual after FFN -> contextualized embeddings
            contextualized = x_after_attn + ffn_output
        
        # **NEW**: Compute pair scores based on attention direction
        if self.attention_direction == "sentence_to_row":
            # Contextualized sentences vs original rows
            contextualized_sentences_norm = torch.nn.functional.normalize(contextualized, p=2, dim=-1)
            rows_norm = torch.nn.functional.normalize(rows_embeddings, p=2, dim=-1)
            # (B, S, D) @ (B, D, R) -> (B, S, R), then transpose to (B, R, S)
            pair_scores = torch.bmm(rows_norm, contextualized_sentences_norm.transpose(1, 2))
        else:
            # Contextualized rows vs original sentences
            contextualized_rows_norm = torch.nn.functional.normalize(contextualized, p=2, dim=-1)
            sentences_norm = torch.nn.functional.normalize(sentences_embeddings, p=2, dim=-1)
            # (B, R, D) @ (B, D, S) -> (B, R, S)
            pair_scores = torch.bmm(contextualized_rows_norm, sentences_norm.transpose(1, 2))
        
        return pair_scores
    
    def _aggregate_scores(self, 
                         row_scores: torch.Tensor, 
                         attention_weights: torch.Tensor,
                         method: str = "top_k_sum") -> torch.Tensor:
        """
        Aggregate row scores using different strategies to encourage focused attention.
        
        Args:
            row_scores: Tensor of shape [batch_size, num_rows]
            attention_weights: Tensor of shape [batch_size, num_rows, num_sentences]
            method: Aggregation method to use
            
        Returns:
            Tensor of shape [batch_size] containing aggregated similarity scores
        """
        batch_size, num_rows = row_scores.shape
        
        if method == "mean":
            # Original problematic approach - takes mean of all rows
            return torch.mean(row_scores, dim=1)
        
        elif method == "top_k_sum":
            # Sum the top-k row scores
            # This encourages the model to focus on multiple relevant pairs
            k = min(self.top_k, num_rows)
            top_k_scores, _ = torch.topk(row_scores, k=k, dim=1)
            return torch.sum(top_k_scores, dim=1)
        
        elif method == "top_k_mean":
            # Take the mean of the top-k row scores
            # Less aggressive than sum but still focuses on best pairs
            k = min(self.top_k, num_rows)
            top_k_scores, _ = torch.topk(row_scores, k=k, dim=1)
            return torch.mean(top_k_scores, dim=1)
        
        elif method == "weighted_top_k":
            # Weighted combination where top scores get higher weights
            k = min(5, num_rows)
            top_k_scores, top_k_indices = torch.topk(row_scores, k=k, dim=1)
            
            # Create exponentially decaying weights for top-k scores
            weights = torch.exp(torch.arange(k, 0, -1, dtype=row_scores.dtype, device=row_scores.device))
            weights = weights / weights.sum()  # Normalize
            weights = weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, k]
            
            return torch.sum(top_k_scores * weights, dim=1)
        
        elif method == "max":
            # Focus only on the single best row-sentence pair
            # Most aggressive approach - takes only the maximum score
            return torch.max(row_scores, dim=1)[0]
        
        elif method == "attention_weighted":
            # Use the attention weights themselves to compute weighted scores
            # Sum attention weights across sentences for each row
            row_attention_scores = torch.sum(attention_weights, dim=2)  # [batch_size, num_rows]
            
            # Normalize attention scores to get weights
            attention_probs = torch.softmax(row_attention_scores, dim=1)
            
            # Weighted combination of row scores based on attention
            return torch.sum(row_scores * attention_probs, dim=1)
        
        elif method == "sparse_top_k":
            # Sparse top-k: zero out all but top-k scores, then take mean
            k = min(self.top_k, num_rows)
            top_k_scores, top_k_indices = torch.topk(row_scores, k=k, dim=1)
            
            # Create sparse scores tensor
            sparse_scores = torch.zeros_like(row_scores)
            sparse_scores.scatter_(1, top_k_indices, top_k_scores)
            
            # Take mean of non-zero (top-k) scores
            return torch.sum(sparse_scores, dim=1) / k
        
        elif method == "entropy_regularized":
            # Advanced: Use entropy to regularize attention distribution
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)  # [batch_size, num_rows]
            # Encourage balanced attention (higher entropy = better)
            entropy_weights = torch.sigmoid(torch.mean(entropy, dim=1))  # [batch_size] - average entropy per batch
            
            # Use top-k aggregation as base, then apply entropy regularization
            k = min(self.top_k, num_rows)
            top_k_scores, _ = torch.topk(row_scores, k=k, dim=1)
            base_scores = torch.mean(top_k_scores, dim=1)  # [batch_size]
            
            return base_scores * entropy_weights
        
        else:
            # Fallback to mean if method is not recognized
            print(f"Warning: Unknown aggregation method '{method}', falling back to 'mean'")
            return torch.mean(row_scores, dim=1)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count the number of trainable and total parameters in the model.
        Useful for verifying LoRA is working correctly.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count LoRA parameters specifically
        lora_params = 0
        if self.use_cross_attention_lora:
            for name, module in self.named_modules():
                if isinstance(module, LoRALinear):
                    lora_params += sum(p.numel() for p in [module.lora_A, module.lora_B] if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lora_parameters': lora_params,
            'frozen_parameters': total_params - trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        }

class BidirectionalCrossAttention(nn.Module):
    """
    Enhanced Bidirectional cross-attention module with multiple attention mechanisms.
    
    Key features:
    1. Multiple attention types: standard, top-k sparse, windowed, threshold-based
    2. Forward attention: rows attend to sentences  contextualized row vectors (CR)
    3. Reverse attention: sentences attend to rows  contextualized sentence vectors (CS) 
    4. Proper residual connections: input + attention(norm(input)) for stable gradient flow
    5. Pre-norm FFN pattern: input + ffn(norm(input)) for refinement layers
    6. Flexible pair scoring: cosine_sim(CR_i, CS_j), dot(CR_i, CS_j), or MLP([CR_i; CS_j; FA_ij; RA_ji])
    7. **NEW**: Attention collapse prevention through sparse attention mechanisms
    """
    def __init__(self, 
                 embedding_dim: int, 
                 attention_dim: int = None,
                 use_lora: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.1,
                 pair_score_method: str = "cosine",
                 share_weights: bool = False,
                 use_refinement: bool = True,
                 use_self_attention: bool = False,
                 self_attention_heads: int = 8,
                 self_attention_dropout: float = 0.1,
                 init_method: str = "xavier_uniform",
                 init_method_params: dict = None,
                 # **NEW PARAMETERS**
                 attention_type: str = "standard",  # "standard", "top_k_sparse", "windowed", "threshold"
                 sparse_top_k: int = 3,
                 window_size: int = 5,
                 threshold_base: float = 0.1,
                 norm_type: str = "layernorm",
                 use_qk_rmsnorm: bool = False,
                 # **NEW**: Latent bottleneck parameters
                 use_latent_bottleneck: bool = False,
                 latent_num: int = 64,
                 latent_dropout: float = 0.0,
                 # Gated attention overlay (post-SDPA gating)
                 use_gated_attention: bool = False,
                 gated_attention_mode: str = "scalar",
                 gated_attention_hidden_dim: int = 0,
                 gated_attention_dropout: float = 0.0,
                 gated_attention_init_bias: float = 2.0,
                 # Inner gate (inside TopKSparseAttention / WindowedCrossAttention / ThresholdAttention)
                 use_inner_gate: bool = False,
                 use_header_conditioning: bool = False,
                 use_cell_level_matching: bool = False,
                 cell_matching_weight: float = 0.35,
                 cell_matching_pooling: str = "max",
                 cell_row_fusion_weight: float = 0.15,
                 # Temperature scaling control (only relevant for standard attention path)
                 disable_temperature: bool = False,
                 # Verbosity control
                 verbose: bool = True):
        # Expose normalization type for external access
        self.norm_type = norm_type
        self.verbose = verbose
        """
        Args:
            attention_type: Type of attention mechanism:
                - "standard": Regular scaled dot-product attention
                - "top_k_sparse": Top-K sparse attention (recommended for attention collapse)
                - "windowed": Content-based windowed attention  
                - "threshold": Threshold-based semantic filtering
            sparse_top_k: Number of top connections to keep in sparse attention
            window_size: Window size for windowed attention
            threshold_base: Base threshold for threshold attention
        """
        super(BidirectionalCrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.pair_score_method = pair_score_method
        self.share_weights = share_weights
        self.use_refinement = use_refinement
        self.use_self_attention = use_self_attention
        self.self_attention_heads = self_attention_heads
        self.self_attention_dropout = self_attention_dropout
        self.init_method = init_method
        self.init_method_params = init_method_params or {}
        self.use_latent_bottleneck = use_latent_bottleneck
        self.latent_num = latent_num
        self.latent_dropout = latent_dropout
        self.use_gated_attention = use_gated_attention
        self.gated_attention_mode = gated_attention_mode
        self.gated_attention_hidden_dim = gated_attention_hidden_dim
        self.gated_attention_dropout = gated_attention_dropout
        self.gated_attention_init_bias = gated_attention_init_bias
        self.use_inner_gate = use_inner_gate
        self.use_header_conditioning = use_header_conditioning
        self.use_cell_level_matching = use_cell_level_matching
        self.cell_matching_weight = float(cell_matching_weight)
        self.cell_matching_pooling = (cell_matching_pooling or "max").lower()
        self.cell_row_fusion_weight = float(cell_row_fusion_weight)
        self.disable_temperature = disable_temperature
        
        # **NEW**: Attention mechanism configuration
        self.attention_type = attention_type
        self.sparse_top_k = sparse_top_k
        self.window_size = window_size
        self.threshold_base = threshold_base
        
        # Use full attention dimension for more expressive attention
        self.attention_dim = attention_dim if attention_dim is not None else embedding_dim
        
        if self.verbose:
            print(f" Using {attention_type} attention mechanism")
            if self.use_header_conditioning:
                print(" Using optional table schema conditioning for Q/K routing")
            if self.use_cell_level_matching:
                print(
                    f" Using cell-level support matching (pooling={self.cell_matching_pooling}, "
                    f"weight={self.cell_matching_weight:.2f}, row_fusion={self.cell_row_fusion_weight:.2f})"
                )

        if self.use_header_conditioning:
            self.table_query_schema_gate = TableSchemaGate(embedding_dim, norm_type=norm_type)
            self.table_key_schema_gate = TableSchemaGate(embedding_dim, norm_type=norm_type)
        else:
            self.table_query_schema_gate = None
            self.table_key_schema_gate = None

        if self.use_cell_level_matching:
            self.table_cell_fusion_norm = create_norm(norm_type, embedding_dim)
            self.table_cell_fusion_proj = nn.Linear(embedding_dim, embedding_dim)
            nn.init.zeros_(self.table_cell_fusion_proj.weight)
            if self.table_cell_fusion_proj.bias is not None:
                nn.init.zeros_(self.table_cell_fusion_proj.bias)
        else:
            self.table_cell_fusion_norm = None
            self.table_cell_fusion_proj = None
        
        # Self-attention blocks for pre-conditioning
        if use_self_attention:
            if self.verbose:
                print(f" Adding self-attention blocks (heads={self_attention_heads}, dropout={self_attention_dropout})")
            self.row_self_attention = SelfAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=self_attention_heads,
                dropout=self_attention_dropout,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params
            )
            self.sentence_self_attention = SelfAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=self_attention_heads,
                dropout=self_attention_dropout,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params
            )
        else:
            self.row_self_attention = None
            self.sentence_self_attention = None
        
        # **NEW**: Optional latent bottleneck blocks (Perceiver-style)
        if self.use_latent_bottleneck:
            if self.verbose:
                print(f" Adding latent bottleneck (L={self.latent_num}) before cross-attention")
            self.row_latent_bottleneck = LatentBottleneck(
                embedding_dim=embedding_dim,
                num_latents=self.latent_num,
                dropout=self.latent_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                norm_type=norm_type,
            )
            self.sentence_latent_bottleneck = LatentBottleneck(
                embedding_dim=embedding_dim,
                num_latents=self.latent_num,
                dropout=self.latent_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                norm_type=norm_type,
            )
        
        # **NEW**: Initialize attention mechanisms based on type
        if attention_type == "top_k_sparse":
            if self.verbose:
                print(f" Initializing Top-K Sparse Attention (k={sparse_top_k})")
            # Inner gate (use_inner_gate) is an independent gate applied inside the attention module
            # before the output is returned. The outer gate (use_gated_attention) is applied in
            # BidirectionalCrossAttention via forward_output_gate / reverse_output_gate.
            # Both can be enabled independently via --use_inner_gate and --use_gated_attention.
            self.forward_attention = TopKSparseAttention(
                embedding_dim=embedding_dim,
                attention_dim=self.attention_dim,
                top_k=sparse_top_k,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                use_gated_attention=self.use_inner_gate,
                gated_attention_mode=self.gated_attention_mode,
                gated_attention_hidden_dim=self.gated_attention_hidden_dim,
                gated_attention_dropout=self.gated_attention_dropout,
                gated_attention_init_bias=self.gated_attention_init_bias,
                norm_type=norm_type,
                disable_temperature=self.disable_temperature,
            )
            self.reverse_attention = TopKSparseAttention(
                embedding_dim=embedding_dim,
                attention_dim=self.attention_dim,
                top_k=sparse_top_k,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                use_gated_attention=self.use_inner_gate,
                gated_attention_mode=self.gated_attention_mode,
                gated_attention_hidden_dim=self.gated_attention_hidden_dim,
                gated_attention_dropout=self.gated_attention_dropout,
                gated_attention_init_bias=self.gated_attention_init_bias,
                norm_type=norm_type,
                disable_temperature=self.disable_temperature,
            )
            
        elif attention_type == "windowed":
            if self.verbose:
                print(f" Initializing Windowed Attention (window_size={window_size})")
            self.forward_attention = WindowedCrossAttention(
                embedding_dim=embedding_dim,
                attention_dim=self.attention_dim,
                window_size=window_size,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                use_gated_attention=self.use_inner_gate,
                norm_type=norm_type,
                disable_temperature=self.disable_temperature,
            )
            self.reverse_attention = WindowedCrossAttention(
                embedding_dim=embedding_dim,
                attention_dim=self.attention_dim,
                window_size=window_size,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                use_gated_attention=self.use_inner_gate,
                norm_type=norm_type,
                disable_temperature=self.disable_temperature,
            )
            
        elif attention_type == "threshold":
            if self.verbose:
                print(f" Initializing Threshold Attention (threshold={threshold_base})")
            self.forward_attention = ThresholdAttention(
                embedding_dim=embedding_dim,
                attention_dim=self.attention_dim,
                base_threshold=threshold_base,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                use_gated_attention=self.use_inner_gate,
                norm_type=norm_type,
                disable_temperature=self.disable_temperature,
            )
            self.reverse_attention = ThresholdAttention(
                embedding_dim=embedding_dim,
                attention_dim=self.attention_dim,
                base_threshold=threshold_base,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                use_gated_attention=self.use_inner_gate,
                norm_type=norm_type,
                disable_temperature=self.disable_temperature,
            )
            
        elif attention_type == "latent_cross":
            if self.verbose:
                print(f" Initializing Latent Cross-Attention (L={self.latent_num})")
            self.forward_attention = LatentCrossAttention(
                embedding_dim=embedding_dim,
                num_latents=self.latent_num,
                dropout=self.latent_dropout,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                norm_type=norm_type,
                use_gated_attention=self.use_gated_attention,
                gated_attention_mode=self.gated_attention_mode,
                gated_attention_hidden_dim=self.gated_attention_hidden_dim,
                gated_attention_dropout=self.gated_attention_dropout,
                gated_attention_init_bias=self.gated_attention_init_bias,
            )
            self.reverse_attention = LatentCrossAttention(
                embedding_dim=embedding_dim,
                num_latents=self.latent_num,
                dropout=self.latent_dropout,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
                init_method_params=init_method_params,
                norm_type=norm_type,
                use_gated_attention=self.use_gated_attention,
                gated_attention_mode=self.gated_attention_mode,
                gated_attention_hidden_dim=self.gated_attention_hidden_dim,
                gated_attention_dropout=self.gated_attention_dropout,
                gated_attention_init_bias=self.gated_attention_init_bias,
            )
        else:  # "standard"
            if self.verbose:
                print(" Using standard scaled dot-product attention")
            # Keep the original implementation for standard attention
            self._initialize_standard_attention()
        
        # Optional Q/K RMSNorm (Qwen-style)
        self.use_qk_rmsnorm = use_qk_rmsnorm
        if self.use_qk_rmsnorm:
            self.forward_q_norm = RMSNorm(self.attention_dim)
            self.forward_k_norm = RMSNorm(self.attention_dim)
            self.reverse_q_norm = RMSNorm(self.attention_dim)
            self.reverse_k_norm = RMSNorm(self.attention_dim)

        # Refinement layers using the unified FFN with internal pre-norm
        self.row_refinement = FeedForward(
            d_model=embedding_dim,
            multiplier=2.0,
            use_swiglu_inner_2over3=True,
            with_pre_norm=True,
            norm_type=norm_type,
        )

        self.sentence_refinement = FeedForward(
            d_model=embedding_dim,
            multiplier=2.0,
            use_swiglu_inner_2over3=True,
            with_pre_norm=True,
            norm_type=norm_type,
        )
        
        # Layer norms for post-attention residual connections (pre-norm pattern)
        self.row_attention_norm = create_norm(norm_type, embedding_dim)
        self.sentence_attention_norm = create_norm(norm_type, embedding_dim)
        
        # Learnable temperature parameters for both directions (only for standard attention)
        if attention_type == "standard":
            self.forward_temperature = nn.Parameter(torch.ones(1) * 0.7)
            self.reverse_temperature = nn.Parameter(torch.ones(1) * 0.7)

        # Optional post-SDPA output gates (query-dependent)
        if self.use_gated_attention:
            self.forward_output_gate = AttentionOutputGate(
                embedding_dim=self.embedding_dim,
                mode=self.gated_attention_mode,
                hidden_dim=self.gated_attention_hidden_dim,
                dropout=self.gated_attention_dropout,
                init_bias=self.gated_attention_init_bias,
                norm_type=norm_type,
            )
            self.reverse_output_gate = AttentionOutputGate(
                embedding_dim=self.embedding_dim,
                mode=self.gated_attention_mode,
                hidden_dim=self.gated_attention_hidden_dim,
                dropout=self.gated_attention_dropout,
                init_bias=self.gated_attention_init_bias,
                norm_type=norm_type,
            )
        else:
            self.forward_output_gate = None
            self.reverse_output_gate = None
        
        # Optional MLP for enhanced pair scoring
        if self.pair_score_method == "mlp":
            mlp_input_dim = 2 * embedding_dim + 2
            self.pair_score_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )
            if self.verbose:
                print(f"Initialized MLP for pair scoring with input dim {mlp_input_dim}")
        else:
            self.pair_score_mlp = None
            if self.verbose:
                print(f"Using {self.pair_score_method} pair scoring method")
    
    def _init_weights(self):
        """Initialize weights using the new initialization system."""
        # Create list of layers for initialization
        layers = [
            self.forward_W_Q, self.forward_W_K, self.forward_W_V,
            self.reverse_W_Q, self.reverse_W_K, self.reverse_W_V
        ]
        
        if self.verbose:
            print(f" Initializing bidirectional attention with method: {self.init_method}")
        
        # Apply the selected initialization method - NO FALLBACK!
        from initialization import initialize_attention_weights
        initialize_attention_weights(
            layers=layers,
            attention_dim=self.attention_dim,
            method=self.init_method,
            method_params=self.init_method_params
        )
        if self.verbose:
            print(f" Successfully applied {self.init_method} initialization")
    
    def _initialize_standard_attention(self):
        """Initialize standard attention layers for bidirectional cross-attention."""
        # Forward attention projections  
        if self.use_lora:
            self.forward_W_Q = LoRALinear(self.embedding_dim, self.attention_dim, 
                                        rank=self.lora_rank, lora_alpha=self.lora_alpha, 
                                        lora_dropout=self.lora_dropout, bias=True)
            self.forward_W_K = LoRALinear(self.embedding_dim, self.attention_dim, 
                                        rank=self.lora_rank, lora_alpha=self.lora_alpha, 
                                        lora_dropout=self.lora_dropout, bias=True)
            self.forward_W_V = LoRALinear(self.embedding_dim, self.attention_dim, 
                                        rank=self.lora_rank, lora_alpha=self.lora_alpha, 
                                        lora_dropout=self.lora_dropout, bias=True)
        else:
            self.forward_W_Q = nn.Linear(self.embedding_dim, self.attention_dim, bias=True)
            self.forward_W_K = nn.Linear(self.embedding_dim, self.attention_dim, bias=True)
            self.forward_W_V = nn.Linear(self.embedding_dim, self.attention_dim, bias=True)
        
        # Reverse attention projections (shared or separate)
        if self.share_weights:
            if self.verbose:
                print(" Sharing attention weights between forward and reverse directions")
            self.reverse_W_Q = self.forward_W_Q
            self.reverse_W_K = self.forward_W_K
            self.reverse_W_V = self.forward_W_V
        else:
            if self.verbose:
                print(" Using separate attention weights for forward and reverse directions")
            if self.use_lora:
                self.reverse_W_Q = LoRALinear(self.embedding_dim, self.attention_dim, 
                                            rank=self.lora_rank, lora_alpha=self.lora_alpha, 
                                            lora_dropout=self.lora_dropout, bias=True)
                self.reverse_W_K = LoRALinear(self.embedding_dim, self.attention_dim, 
                                            rank=self.lora_rank, lora_alpha=self.lora_alpha, 
                                            lora_dropout=self.lora_dropout, bias=True)
                self.reverse_W_V = LoRALinear(self.embedding_dim, self.attention_dim, 
                                            rank=self.lora_rank, lora_alpha=self.lora_alpha, 
                                            lora_dropout=self.lora_dropout, bias=True)
            else:
                self.reverse_W_Q = nn.Linear(self.embedding_dim, self.attention_dim, bias=True)
                self.reverse_W_K = nn.Linear(self.embedding_dim, self.attention_dim, bias=True)
                self.reverse_W_V = nn.Linear(self.embedding_dim, self.attention_dim, bias=True)
        
        # Initialize weights if not using LoRA
        if not self.use_lora:
            self._init_weights()
    
    def _apply_attention(self, queries, keys, values, temperature):
        """Apply scaled dot-product attention with temperature (for standard attention only)."""
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Scale by sqrt of dimension
        attention_scores = attention_scores / (self.attention_dim ** 0.5)
        
        # Apply temperature scaling with better stability (higher temps = softer attention)
        if not getattr(self, 'disable_temperature', False):
            temperature = torch.clamp(temperature, min=1.0, max=5.0)
            attention_scores = attention_scores / temperature
        
        # Add numerical stability to prevent overflow/underflow
        attention_scores = torch.clamp(attention_scores, min=-50.0, max=50.0)
        
        # Apply selected attention activation (softmax/entmax)
        activation = getattr(self, 'attention_activation', 'softmax')
        alpha = getattr(self, 'attention_alpha', 1.5)
        attention_weights = apply_attention_activation(attention_scores, dim=-1, name=activation, alpha=alpha)
        
        # Check for NaN in attention weights and replace with uniform if needed
        if torch.isnan(attention_weights).any():
            print("Warning: NaN detected in attention weights, using uniform distribution")
            seq_len = attention_weights.shape[-1]
            attention_weights = torch.ones_like(attention_weights) / seq_len
        
        # Apply attention to values
        context_vectors = torch.bmm(attention_weights, values)
        
        return context_vectors, attention_weights
    
    def compute_attention_entropy_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization loss to encourage diverse attention patterns.
        Higher entropy means more diverse attention distribution.
        
        Args:
            attention_weights: [batch_size, seq_len, seq_len] attention weights
            
        Returns:
            Entropy loss (negative entropy, minimizing this maximizes diversity)
        """
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        log_weights = torch.log(attention_weights + eps)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(attention_weights * log_weights, dim=-1)  # [batch_size, seq_len]
        
        # Return negative mean entropy (minimizing this maximizes attention diversity)
        return -torch.mean(entropy)

    def _prepare_cell_embeddings(self, cell_embeddings: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if cell_embeddings is None:
            return None
        if cell_embeddings.dim() == 3:
            cell_embeddings = cell_embeddings.unsqueeze(0)
        return cell_embeddings

    def _pool_cell_embeddings(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        valid_mask = (cell_embeddings.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=cell_embeddings.dtype)
        pooled = (cell_embeddings * valid_mask).sum(dim=2)
        denominator = valid_mask.sum(dim=2).clamp(min=1.0)
        return pooled / denominator

    def _fuse_rows_with_cells(self, row_embeddings: torch.Tensor, cell_embeddings: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_cell_level_matching or self.cell_row_fusion_weight <= 0.0:
            return row_embeddings

        cell_embeddings = self._prepare_cell_embeddings(cell_embeddings)
        if cell_embeddings is None:
            return row_embeddings

        pooled_cells = self._pool_cell_embeddings(cell_embeddings).to(dtype=row_embeddings.dtype)
        fused_delta = torch.tanh(self.table_cell_fusion_proj(self.table_cell_fusion_norm(pooled_cells)))
        return row_embeddings + self.cell_row_fusion_weight * fused_delta.to(dtype=row_embeddings.dtype)

    def _compute_cell_support(self, table_cell_embeddings: Optional[torch.Tensor], other_embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_cell_level_matching or self.cell_matching_weight <= 0.0:
            return None

        table_cell_embeddings = self._prepare_cell_embeddings(table_cell_embeddings)
        if table_cell_embeddings is None:
            return None

        valid_mask = table_cell_embeddings.abs().sum(dim=-1) > 0
        if not valid_mask.any():
            return None

        metric_name = self.pair_score_method if self.pair_score_method in {"cosine", "dot"} else "cosine"
        if metric_name == "dot":
            cell_scores = torch.einsum("bncd,bmd->bncm", table_cell_embeddings, other_embeddings)
        else:
            normalized_cells = F.normalize(table_cell_embeddings, p=2, dim=-1)
            normalized_other = F.normalize(other_embeddings, p=2, dim=-1)
            cell_scores = torch.einsum("bncd,bmd->bncm", normalized_cells, normalized_other)

        expanded_mask = valid_mask.unsqueeze(-1)
        if self.cell_matching_pooling == "mean":
            masked_scores = cell_scores * expanded_mask.to(dtype=cell_scores.dtype)
            denominator = expanded_mask.sum(dim=2).clamp(min=1)
            return masked_scores.sum(dim=2) / denominator.to(dtype=cell_scores.dtype)

        masked_scores = cell_scores.masked_fill(~expanded_mask, float("-inf"))
        pooled_scores = masked_scores.max(dim=2).values
        no_valid_cells = ~expanded_mask.any(dim=2)
        pooled_scores = torch.where(no_valid_cells, torch.zeros_like(pooled_scores), pooled_scores)
        return pooled_scores
    
    def forward(self, 
                rows_embeddings: torch.Tensor, 
                sentences_embeddings: torch.Tensor,
                rows_schema_embeddings: Optional[torch.Tensor] = None,
                sentences_schema_embeddings: Optional[torch.Tensor] = None,
                rows_cell_embeddings: Optional[torch.Tensor] = None,
                sentences_cell_embeddings: Optional[torch.Tensor] = None,
                diagnostics: bool = False) -> tuple:
        """
        Apply bidirectional cross-attention with proper residual connections following Transformer practices.
        
        Architecture follows pre-norm pattern:
        1. LayerNorm  Cross-Attention  Residual Connection
        2. LayerNorm  FFN  Residual Connection
        
        This ensures proper gradient flow and prevents attention collapse.
        
        Args:
            rows_embeddings: [batch_size, num_rows, embedding_dim]
            sentences_embeddings: [batch_size, num_sentences, embedding_dim]
            
        Returns:
            Tuple containing:
            - pair_scores: [batch_size, num_rows, num_sentences] - Direct pair-wise similarity scores
            - contextualized_rows: [batch_size, num_rows, embedding_dim] - Row contexts
            - contextualized_sentences: [batch_size, num_sentences, embedding_dim] - Sentence contexts  
            - forward_attention_weights: [batch_size, num_rows, num_sentences]
            - reverse_attention_weights: [batch_size, num_sentences, num_rows]
        """
        batch_size, num_rows, _ = rows_embeddings.shape
        batch_size, num_sentences, _ = sentences_embeddings.shape

        if self.use_cell_level_matching:
            rows_embeddings = self._fuse_rows_with_cells(rows_embeddings, rows_cell_embeddings)
            sentences_embeddings = self._fuse_rows_with_cells(sentences_embeddings, sentences_cell_embeddings)
        
        # =================== SELF-ATTENTION PRE-CONDITIONING (NEW!) ===================
        if self.use_self_attention:
            # Apply self-attention to create more contextual and distinctive representations
            # This helps prevent attention collapse by pre-conditioning the inputs
            print(" Applying self-attention pre-conditioning...") if diagnostics else None
            
            # Row self-attention: rows attend to other rows
            self_attended_rows = self.row_self_attention(rows_embeddings)
            
            # Sentence self-attention: sentences attend to other sentences
            self_attended_sentences = self.sentence_self_attention(sentences_embeddings)
            
            # Use self-attended representations for cross-attention
            input_rows = self_attended_rows
            input_sentences = self_attended_sentences
        else:
            # Use original embeddings directly
            input_rows = rows_embeddings
            input_sentences = sentences_embeddings
        
        # =================== LATENT BOTTLENECK (Perceiver-style) ===================
        if self.use_latent_bottleneck:
            input_rows = self.row_latent_bottleneck(input_rows)
            input_sentences = self.sentence_latent_bottleneck(input_sentences)

        # =================== OPTIONAL TABLE SCHEMA CONDITIONING ===================
        conditioned_rows_for_query = input_rows
        conditioned_rows_for_key = input_rows
        conditioned_sentences_for_query = input_sentences
        conditioned_sentences_for_key = input_sentences

        if self.use_header_conditioning:
            if rows_schema_embeddings is not None:
                conditioned_rows_for_query = self.table_query_schema_gate(input_rows, rows_schema_embeddings)
                conditioned_rows_for_key = self.table_key_schema_gate(input_rows, rows_schema_embeddings)
            if sentences_schema_embeddings is not None:
                conditioned_sentences_for_query = self.table_query_schema_gate(input_sentences, sentences_schema_embeddings)
                conditioned_sentences_for_key = self.table_key_schema_gate(input_sentences, sentences_schema_embeddings)
        
        # =================== FORWARD AND REVERSE ATTENTION WITH RESIDUALS ===================
        # **NEW**: Different attention mechanisms based on attention_type
        if self.attention_type == "standard":
            # Original standard attention implementation
            # Forward attention: rows attend to sentences  contextualized row vectors
            normed_rows = self.row_attention_norm(conditioned_rows_for_query)
            forward_Q = self.forward_W_Q(normed_rows)
            forward_K = self.forward_W_K(conditioned_sentences_for_key)
            if self.use_qk_rmsnorm:
                forward_Q = self.forward_q_norm(forward_Q)
                forward_K = self.forward_k_norm(forward_K)
            forward_V = self.forward_W_V(input_sentences)
            
            attention_output_rows, forward_attention_weights = self._apply_attention(
                forward_Q, forward_K, forward_V, self.forward_temperature
            )

            if self.forward_output_gate is not None:
                attention_output_rows = self.forward_output_gate(normed_rows, attention_output_rows)
            
            # Reverse attention: sentences attend to rows  contextualized sentence vectors
            normed_sentences = self.sentence_attention_norm(conditioned_sentences_for_query)
            reverse_Q = self.reverse_W_Q(normed_sentences)
            reverse_K = self.reverse_W_K(conditioned_rows_for_key)
            if self.use_qk_rmsnorm:
                reverse_Q = self.reverse_q_norm(reverse_Q)
                reverse_K = self.reverse_k_norm(reverse_K)
            reverse_V = self.reverse_W_V(input_rows)
            
            attention_output_sentences, reverse_attention_weights = self._apply_attention(
                reverse_Q, reverse_K, reverse_V, self.reverse_temperature  
            )

            if self.reverse_output_gate is not None:
                attention_output_sentences = self.reverse_output_gate(normed_sentences, attention_output_sentences)
            
        else:
            # **NEW**: Sparse attention mechanisms (top-k, windowed, threshold)
            print(f" Applying {self.attention_type} attention mechanism...") if diagnostics else None
            
            # Forward attention: rows attend to sentences
            normed_rows = self.row_attention_norm(conditioned_rows_for_query)
            attention_output_rows, forward_attention_weights = self.forward_attention(
                normed_rows, conditioned_sentences_for_key, input_sentences
            )

            if self.forward_output_gate is not None:
                attention_output_rows = self.forward_output_gate(normed_rows, attention_output_rows)
            
            # Reverse attention: sentences attend to rows  
            normed_sentences = self.sentence_attention_norm(conditioned_sentences_for_query)
            attention_output_sentences, reverse_attention_weights = self.reverse_attention(
                normed_sentences, conditioned_rows_for_key, input_rows
            )

            if self.reverse_output_gate is not None:
                attention_output_sentences = self.reverse_output_gate(normed_sentences, attention_output_sentences)
        
        # RESIDUAL CONNECTIONS: Add original inputs to attention outputs
        contextualized_rows = input_rows + attention_output_rows
        contextualized_sentences = input_sentences + attention_output_sentences
        
        # =================== REFINEMENT WITH RESIDUAL ===================
        if self.use_refinement:
            refined_rows = contextualized_rows + self.row_refinement(contextualized_rows)
            refined_sentences = contextualized_sentences + self.sentence_refinement(contextualized_sentences)
        else:
            refined_rows = contextualized_rows
            refined_sentences = contextualized_sentences
        
        # =================== PAIR-WISE SIMILARITY ===================
        # Compute similarities at both stages if diagnostics is enabled
        if diagnostics:
            # Cosine similarity before refinement
            pre_refine_pair_scores = torch.cosine_similarity(
                contextualized_rows.unsqueeze(2),
                contextualized_sentences.unsqueeze(1),
                dim=-1
            )
        else:
            pre_refine_pair_scores = None
        # Standard pair scoring (after refinement)
        if self.pair_score_method == "cosine":
            pair_scores = torch.cosine_similarity(
                refined_rows.unsqueeze(2),
                refined_sentences.unsqueeze(1),
                dim=-1
            )
        elif self.pair_score_method == "dot":
            pair_scores = torch.bmm(refined_rows, refined_sentences.transpose(-2, -1))
        elif self.pair_score_method == "mlp":
            batch_size, num_rows, _ = refined_rows.shape
            _, num_sentences, _ = refined_sentences.shape
            expanded_rows = refined_rows.unsqueeze(2).expand(-1, -1, num_sentences, -1)
            expanded_sentences = refined_sentences.unsqueeze(1).expand(-1, num_rows, -1, -1)
            expanded_forward_attn = forward_attention_weights.unsqueeze(-1)
            expanded_reverse_attn = reverse_attention_weights.transpose(-2, -1).unsqueeze(-1)
            mlp_input = torch.cat([
                expanded_rows,
                expanded_sentences,
                expanded_forward_attn,
                expanded_reverse_attn
            ], dim=-1)
            flat_input = mlp_input.view(-1, mlp_input.shape[-1])
            flat_scores = self.pair_score_mlp(flat_input)
            pair_scores = flat_scores.view(batch_size, num_rows, num_sentences)
            if torch.isnan(pair_scores).any():
                print("Warning: NaN detected in MLP pair scores, using fallback cosine similarity")
                pair_scores = torch.cosine_similarity(
                    refined_rows.unsqueeze(2),
                    refined_sentences.unsqueeze(1),
                    dim=-1
                )
        else:
            raise ValueError(f"Unknown pair_score_method: {self.pair_score_method}")

        if self.use_cell_level_matching and self.cell_matching_weight > 0.0:
            cell_support_terms = []
            row_side_cell_support = self._compute_cell_support(rows_cell_embeddings, refined_sentences)
            if row_side_cell_support is not None:
                cell_support_terms.append(row_side_cell_support.to(dtype=pair_scores.dtype))

            sentence_side_cell_support = self._compute_cell_support(sentences_cell_embeddings, refined_rows)
            if sentence_side_cell_support is not None:
                cell_support_terms.append(sentence_side_cell_support.transpose(-2, -1).to(dtype=pair_scores.dtype))

            if cell_support_terms:
                combined_cell_support = torch.stack(cell_support_terms, dim=0).mean(dim=0)
                pair_scores = (1.0 - self.cell_matching_weight) * pair_scores + self.cell_matching_weight * combined_cell_support

        if diagnostics:
            return (pair_scores, refined_rows, refined_sentences, forward_attention_weights, reverse_attention_weights, {
                'contextualized_rows': contextualized_rows.detach().cpu(),
                'contextualized_sentences': contextualized_sentences.detach().cpu(),
                'refined_rows': refined_rows.detach().cpu(),
                'refined_sentences': refined_sentences.detach().cpu(),
                'pre_refine_pair_scores': pre_refine_pair_scores.detach().cpu() if pre_refine_pair_scores is not None else None,
                'post_refine_pair_scores': pair_scores.detach().cpu(),
                'row_side_cell_support': row_side_cell_support.detach().cpu() if 'row_side_cell_support' in locals() and row_side_cell_support is not None else None,
                'sentence_side_cell_support': sentence_side_cell_support.detach().cpu() if 'sentence_side_cell_support' in locals() and sentence_side_cell_support is not None else None,
                'forward_attention_weights': forward_attention_weights.detach().cpu(),
                'reverse_attention_weights': reverse_attention_weights.detach().cpu()
            })
        else:
            return (pair_scores, refined_rows, refined_sentences, forward_attention_weights, reverse_attention_weights)

class BidirectionalTableTextModel(nn.Module):
    """
    Enhanced model using bidirectional cross-attention for table-text embedding.
    
    Key innovations:
    1. Bidirectional cross-attention produces contextualized vectors for both rows and sentences
    2. Direct cosine similarity between contextualized vectors (normalized to [-1, 1])
    3. NM pair score matrix enables direct join-path discovery
    4. Pair-level aggregation methods for global similarity
    5. Optional weight sharing between forward and reverse attention to prevent attention collapse
    
    This architecture addresses the limitations of unidirectional attention by:
    - Creating symmetric representations via bidirectional attention
    - Preserving fine-grained pair-wise information for join-path extraction
    - Enabling interpretable similarity scores via cosine similarity of context vectors
    """
    def __init__(self, 
                 sentence_encoder: SentenceTransformer, 
                 embedding_dim: int, 
                 native_embedding_dim: int = None,
                 trainable_encoder: bool = False,
                 use_cross_attention_lora: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.1,
                 top_k: int = 3,
                 pair_score_method: str = "cosine",
                 share_weights: bool = False,
                 use_refinement: bool = True,
                 use_self_attention: bool = False,
                 self_attention_heads: int = 8,
                 self_attention_dropout: float = 0.1,
                 init_method: str = "xavier_uniform",
                 init_method_params: dict = None,
                 # **NEW**: Attention mechanism parameters
                 attention_type: str = "standard",
                 sparse_top_k: int = 3,
                 window_size: int = 5,
                 threshold_base: float = 0.1,
                 norm_type: str = "layernorm",
                 use_qk_rmsnorm: bool = False,
                 # **NEW**: Latent bottleneck parameters
                 use_latent_bottleneck: bool = False,
                 latent_num: int = 64,
                 latent_dropout: float = 0.0,
                 # Gated attention overlay (post-SDPA gating)
                 use_gated_attention: bool = False,
                 gated_attention_mode: str = "scalar",
                 gated_attention_hidden_dim: int = 0,
                 gated_attention_dropout: float = 0.0,
                 gated_attention_init_bias: float = 2.0,
                 # Inner gate (inside TopKSparseAttention / WindowedCrossAttention / ThresholdAttention)
                 use_inner_gate: bool = False,
                 use_header_conditioning: bool = False,
                 use_cell_level_matching: bool = False,
                 cell_matching_weight: float = 0.35,
                 cell_matching_pooling: str = "max",
                 cell_row_fusion_weight: float = 0.15,
                 # Temperature scaling control
                 disable_temperature: bool = False,
                 # Verbosity control
                 verbose: bool = True):
        """
        Args:
            sentence_encoder: Pre-trained sentence encoder
            embedding_dim: Dimension of embeddings
            trainable_encoder: Whether to fine-tune the sentence encoder
            use_cross_attention_lora: Whether to use LoRA for cross-attention
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha parameter  
            lora_dropout: LoRA dropout rate
            top_k: Number of top pairs to consider for aggregation
            pair_score_method: Method for computing pair scores:
                - "cosine": Cosine similarity (default, normalized [-1,1])
                - "dot": Dot product (raw similarity scores)
                - "mlp": MLP with concatenated features and attention weights
            share_weights: Whether to share weights between forward and reverse attention
            use_refinement: Whether to apply refinement layers after attention
            use_self_attention: Whether to apply self-attention before cross-attention (NEW!)
            self_attention_heads: Number of heads for self-attention
            self_attention_dropout: Dropout rate for self-attention
            init_method: Initialization method for attention weights (see initialization.py)
            init_method_params: Optional parameters for the initialization method
            verbose: Whether to print initialization messages
        """
        super(BidirectionalTableTextModel, self).__init__()
        self.verbose = verbose
        
        self.sentence_encoder = sentence_encoder
        self.embedding_dim = embedding_dim
        self.trainable_encoder = trainable_encoder
        # Expose normalization type
        self.norm_type = norm_type
        self.top_k = top_k
        self.native_embedding_dim = native_embedding_dim or embedding_dim
        self.pair_score_method = pair_score_method
        self.share_weights = share_weights
        self.use_refinement = use_refinement
        self.use_self_attention = use_self_attention
        self.self_attention_heads = self_attention_heads
        self.self_attention_dropout = self_attention_dropout
        self.use_latent_bottleneck = use_latent_bottleneck
        self.latent_num = latent_num
        self.latent_dropout = latent_dropout
        self.use_gated_attention = use_gated_attention
        self.gated_attention_mode = gated_attention_mode
        self.gated_attention_hidden_dim = gated_attention_hidden_dim
        self.gated_attention_dropout = gated_attention_dropout
        self.gated_attention_init_bias = gated_attention_init_bias
        self.use_inner_gate = use_inner_gate
        self.use_header_conditioning = use_header_conditioning
        self.use_cell_level_matching = use_cell_level_matching
        self.cell_matching_weight = float(cell_matching_weight)
        self.cell_matching_pooling = (cell_matching_pooling or "max").lower()
        self.cell_row_fusion_weight = float(cell_row_fusion_weight)
        self.disable_temperature = disable_temperature
        
        # =====================================================================
        # Handle encoder trainability with PEFT/QLoRA awareness
        # =====================================================================
        # Check if encoder has PEFT adapters (LoRA) attached
        # Check for actual PEFT parameters, since Unsloth FastModel or huggingface 
        # might inject empty or dummy peft_config dicts that evaluate to True.
        encoder_has_peft = any('lora' in name.lower() for name, _ in self.sentence_encoder.named_parameters())
        
        if encoder_has_peft:
            # PEFT is managing the encoder - DON'T override its frozen/trainable state
            if self.verbose:
                print("   [INFO] PEFT/QLoRA detected on sentence encoder - preserving PEFT freeze state")
                enc_trainable = sum(p.numel() for p in self.sentence_encoder.parameters() if p.requires_grad)
                enc_total = sum(p.numel() for p in self.sentence_encoder.parameters())
                print(f"      Encoder params: {enc_trainable:,}/{enc_total:,} trainable ({enc_trainable/enc_total*100:.2f}%)")
        elif not trainable_encoder:
            # No PEFT, and user wants encoder frozen
            for param in self.sentence_encoder.parameters():
                param.requires_grad = False
            if self.verbose:
                print("Sentence encoder frozen (not trainable)")
        else:
            # No PEFT, and user wants encoder trainable - this is full fine-tuning
            skipped_non_float_params = 0
            for param in self.sentence_encoder.parameters():
                if torch.is_floating_point(param) or torch.is_complex(param):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    skipped_non_float_params += param.numel()
            if self.verbose:
                print("Sentence encoder is trainable (full fine-tuning, no PEFT)")
                if skipped_non_float_params > 0:
                    print(f"   Skipped {skipped_non_float_params:,} non-floating encoder params during unfreeze")
        
        # Bidirectional cross-attention module with initialization parameters
        self.bidirectional_attention = BidirectionalCrossAttention(
            embedding_dim=embedding_dim,
            use_lora=use_cross_attention_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            pair_score_method=pair_score_method,
            share_weights=share_weights,
            use_refinement=use_refinement,
            use_self_attention=use_self_attention,
            self_attention_heads=self_attention_heads,
            self_attention_dropout=self_attention_dropout,
            init_method=init_method,
            init_method_params=init_method_params,
            # **NEW**: Attention mechanism parameters
            attention_type=attention_type,
            sparse_top_k=sparse_top_k,
            window_size=window_size,
            threshold_base=threshold_base,
            norm_type=norm_type,
            use_qk_rmsnorm=use_qk_rmsnorm,
            # Latent bottleneck parameters
            use_latent_bottleneck=self.use_latent_bottleneck,
            latent_num=self.latent_num,
            latent_dropout=self.latent_dropout,
            use_gated_attention=self.use_gated_attention,
            gated_attention_mode=self.gated_attention_mode,
            gated_attention_hidden_dim=self.gated_attention_hidden_dim,
            gated_attention_dropout=self.gated_attention_dropout,
            gated_attention_init_bias=self.gated_attention_init_bias,
            use_inner_gate=self.use_inner_gate,
            use_header_conditioning=self.use_header_conditioning,
            use_cell_level_matching=self.use_cell_level_matching,
            cell_matching_weight=self.cell_matching_weight,
            cell_matching_pooling=self.cell_matching_pooling,
            cell_row_fusion_weight=self.cell_row_fusion_weight,
            disable_temperature=self.disable_temperature,
            verbose=self.verbose,
        )
        
        if self.verbose:
            print(f"Bidirectional model initialized with top_k={self.top_k}, pair_score_method={self.pair_score_method}, share_weights={self.share_weights}, use_refinement={self.use_refinement}")
            print(f" Using initialization method: {init_method}")
            print(f" Header conditioning: {self.use_header_conditioning}")
            print(f" Cell-level matching: {self.use_cell_level_matching}")
            if init_method_params:
                print(f" Initialization parameters: {init_method_params}")
            
            if use_cross_attention_lora:
                print(f"Cross-attention LoRA enabled: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
            # Note: encoder trainability status already printed above with PEFT awareness
        
        # Detect sentence encoder dtype for compatibility
        try:
            encoder_dtype = next(self.sentence_encoder.parameters()).dtype
            if self.verbose:
                print(f"Sentence encoder dtype detected: {encoder_dtype}")
        except:
            encoder_dtype = torch.bfloat16
            if self.verbose:
                print("Could not detect sentence encoder dtype, defaulting to bfloat16")
        
        # Optional projection layer for similarity scaling
        self.similarity_projection = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # Convert all custom components to BFloat16 unconditionally to match typical compute precision
        print("Converting bidirectional components to BFloat16 (ensuring compatibility)...")
        # Use PyTorch standard module conversion instead of `.data` modification
        self.bidirectional_attention.to(dtype=torch.bfloat16)
        self.similarity_projection.to(dtype=torch.bfloat16)

        # Optional dimension projection for embedding dim override (non-Matryoshka)
        if self.native_embedding_dim != self.embedding_dim:
            self.dim_projection = nn.Linear(self.native_embedding_dim, self.embedding_dim, bias=False)
            self.dim_projection.to(dtype=torch.bfloat16)
            if self.verbose:
                print(f"[INFO] Added dim projection: {self.native_embedding_dim} -> {self.embedding_dim}")
        else:
            self.dim_projection = None

        print(" Bidirectional components converted to BFloat16")
    
    def encode_sentences(self, 
                        sentences: List[str], 
                        batch_size: int = 32, 
                        normalize: bool = True) -> torch.Tensor:
        """
        Encode a list of sentences using the sentence encoder.
        """
        embeddings = self.sentence_encoder.encode(
            sentences,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        # Apply dimension projection if needed (non-Matryoshka override)
        if self.dim_projection is not None:
            embeddings = self.dim_projection(embeddings.to(dtype=self.dim_projection.weight.dtype))
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings
    
    def forward(self, 
               rows_embeddings: torch.Tensor, 
               sentences_embeddings: torch.Tensor,
               aggregation_method: str = "top_k_pairs",
               diagnostics: bool = False,
               return_attention_weights: bool = False,
               return_contextualized: bool = False,
               rows_schema_embeddings: Optional[torch.Tensor] = None,
               sentences_schema_embeddings: Optional[torch.Tensor] = None,
               rows_cell_embeddings: Optional[torch.Tensor] = None,
               sentences_cell_embeddings: Optional[torch.Tensor] = None) -> tuple:
        """
        Forward pass using bidirectional cross-attention.
        
        Args:
            rows_embeddings: [batch_size, num_rows, embedding_dim]
            sentences_embeddings: [batch_size, num_sentences, embedding_dim]
            aggregation_method: Method for aggregating pair scores:
                - "top_k_pairs": Sum top-k pair scores (default)
                - "max_pairs": Take maximum pair score
                - "mean_pairs": Mean of all pair scores
                - "weighted_pairs": Attention-weighted pair scores
                - "sparse_pairs": Top-k pairs with sparsity
                
        Returns:
            Tuple containing:
            - similarity_score: [batch_size] global similarity scores
            - pair_scores: [batch_size, num_rows, num_sentences] pair-wise scores for join-path discovery
            - optionally forward/reverse attention weights when return_attention_weights=True
            - optionally contextualized_rows, contextualized_sentences when return_contextualized=True
        """
        # CRITICAL: Ensure inputs match model component dtypes
        model_dtype = next(self.bidirectional_attention.parameters()).dtype
        rows_embeddings = rows_embeddings.to(dtype=model_dtype)
        sentences_embeddings = sentences_embeddings.to(dtype=model_dtype)
        if rows_schema_embeddings is not None:
            rows_schema_embeddings = rows_schema_embeddings.to(dtype=model_dtype)
        if sentences_schema_embeddings is not None:
            sentences_schema_embeddings = sentences_schema_embeddings.to(dtype=model_dtype)
        if rows_cell_embeddings is not None:
            rows_cell_embeddings = rows_cell_embeddings.to(dtype=model_dtype)
        if sentences_cell_embeddings is not None:
            sentences_cell_embeddings = sentences_cell_embeddings.to(dtype=model_dtype)
        
        # Apply bidirectional cross-attention
        if diagnostics:
            result = self.bidirectional_attention(
                rows_embeddings,
                sentences_embeddings,
                rows_schema_embeddings=rows_schema_embeddings,
                sentences_schema_embeddings=sentences_schema_embeddings,
                rows_cell_embeddings=rows_cell_embeddings,
                sentences_cell_embeddings=sentences_cell_embeddings,
                diagnostics=True,
            )
            pair_scores, contextualized_rows, contextualized_sentences, forward_attn, reverse_attn, diag = result
        else:
            pair_scores, contextualized_rows, contextualized_sentences, forward_attn, reverse_attn = self.bidirectional_attention(
                rows_embeddings,
                sentences_embeddings,
                rows_schema_embeddings=rows_schema_embeddings,
                sentences_schema_embeddings=sentences_schema_embeddings,
                rows_cell_embeddings=rows_cell_embeddings,
                sentences_cell_embeddings=sentences_cell_embeddings,
            )
            diag = None
        # Aggregate raw pair scores to get global similarity. Attention maps are
        # diagnostics/auxiliary training signals, not a destructive score gate.
        global_similarity = self._aggregate_pair_scores(
            pair_scores, forward_attn, reverse_attn, aggregation_method
        )
        if diagnostics:
            diag['global_similarity'] = global_similarity.detach().cpu()
            if return_attention_weights:
                return global_similarity, pair_scores, forward_attn, reverse_attn, diag
            return global_similarity, pair_scores, diag
        else:
            if return_attention_weights and return_contextualized:
                return global_similarity, pair_scores, forward_attn, reverse_attn, contextualized_rows, contextualized_sentences
            elif return_attention_weights:
                return global_similarity, pair_scores, forward_attn, reverse_attn
            elif return_contextualized:
                return global_similarity, pair_scores, contextualized_rows, contextualized_sentences
            return global_similarity, pair_scores
    
    def _aggregate_pair_scores(self,
                              pair_scores: torch.Tensor,
                              forward_attention: torch.Tensor, 
                              reverse_attention: torch.Tensor,
                              method: str = "top_k_pairs") -> torch.Tensor:
        """
        Aggregate the NM pair score matrix into a global similarity score.
        
        Args:
            pair_scores: [batch_size, num_rows, num_sentences] pair-wise similarity matrix
            forward_attention: [batch_size, num_rows, num_sentences] forward attention weights
            reverse_attention: [batch_size, num_sentences, num_rows] reverse attention weights  
            method: Aggregation strategy
            
        Returns:
            [batch_size] global similarity scores
        """
        batch_size, num_rows, num_sentences = pair_scores.shape
        
        if method == "top_k_pairs":
            # Sum the top-k pair scores across the entire NM matrix
            k = min(self.top_k, num_rows * num_sentences)
            flat_scores = pair_scores.view(batch_size, -1)  # [batch_size, N*M]
            top_k_scores, _ = torch.topk(flat_scores, k=k, dim=1)
            return torch.sum(top_k_scores, dim=1)
        
        elif method == "max_pairs":
            # Take the maximum pair score
            flat_scores = pair_scores.view(batch_size, -1)
            return torch.max(flat_scores, dim=1)[0]
        
        elif method == "mean_pairs":
            # Mean of all pair scores
            return torch.mean(pair_scores.view(batch_size, -1), dim=1)
        
        elif method == "weighted_pairs":
            # Use attention weights to create weighted combination
            # Combine forward and reverse attention for symmetric weighting
            combined_attention = (forward_attention + reverse_attention.transpose(-2, -1)) / 2
            weighted_scores = pair_scores * combined_attention
            return torch.sum(weighted_scores.view(batch_size, -1), dim=1)
        
        elif method == "sparse_pairs":
            # Top-k pairs with sparsity (zero out non-top-k)
            k = min(self.top_k, num_rows * num_sentences)
            flat_scores = pair_scores.view(batch_size, -1)
            top_k_scores, top_k_indices = torch.topk(flat_scores, k=k, dim=1)
            
            # Create sparse tensor
            sparse_scores = torch.zeros_like(flat_scores)
            sparse_scores.scatter_(1, top_k_indices, top_k_scores)
            return torch.mean(sparse_scores, dim=1)  # Mean of sparse scores
            
        elif method == "entropy_regularized":
            # Bidirectional entropy regularization using forward attention weights
            # Calculate entropy across sentence dimension for each row
            epsilon = 1e-8
            forward_entropy = -torch.sum(forward_attention * torch.log(forward_attention + epsilon), dim=-1)  # [batch_size, num_rows]
            
            # Encourage balanced attention (higher entropy = better)
            entropy_weights = torch.sigmoid(torch.mean(forward_entropy, dim=1))  # [batch_size] - average entropy per batch
            
            # Use top-k pairs as base, then apply entropy regularization
            k = min(self.top_k, num_rows * num_sentences)
            flat_scores = pair_scores.view(batch_size, -1)  # [batch_size, N*M]
            top_k_scores, _ = torch.topk(flat_scores, k=k, dim=1)
            base_scores = torch.mean(top_k_scores, dim=1)  # [batch_size]
            
            return base_scores * entropy_weights
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def extract_join_paths(self,
                          pair_scores: torch.Tensor,
                          rows: List[str],
                          sentences: List[str], 
                          threshold: float = 0.1,
                          top_k: int = None) -> List[Tuple[int, int, float]]:
        """
        Extract join paths from the pair score matrix.
        
        Args:
            pair_scores: [num_rows, num_sentences] pair score matrix (single example)
            rows: List of row texts
            sentences: List of sentence texts
            threshold: Minimum score threshold for considering a pair
            top_k: Maximum number of pairs to return (if None, use threshold)
            
        Returns:
            List of (row_idx, sentence_idx, score) tuples representing join paths
        """
        if pair_scores.dim() == 3:
            # Remove batch dimension if present
            pair_scores = pair_scores.squeeze(0)
        
        num_rows, num_sentences = pair_scores.shape
        
        # Get all pairs above threshold
        pairs = []
        for i in range(num_rows):
            for j in range(num_sentences):
                score = pair_scores[i, j].item()
                if score >= threshold:
                    pairs.append((i, j, score))
        
        # Sort by score (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Apply top_k limit if specified
        if top_k is not None:
            pairs = pairs[:top_k]
        
        return pairs
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        }

class TopKSparseAttention(nn.Module):
    """
    Top-K Sparse Cross-Attention that forces attention to focus only on 
    the most relevant K connections, naturally preventing uniform attention.
    
    This addresses attention collapse by:
    1. Computing full attention scores
    2. Keeping only top-K scores per row/query
    3. Setting others to zero (or very negative values)
    4. Applying softmax only over the selected top-K
    """
    def __init__(self, 
                 embedding_dim: int,
                 attention_dim: int = None,
                 top_k: int = 3,
                 temperature: float = 1.0,
                 use_lora: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.1,
                 init_method: str = "xavier_uniform",
                 init_method_params: dict = None,
                 # Gated attention overlay (post-SDPA gating)
                 use_gated_attention: bool = False,
                 gated_attention_mode: str = "scalar",
                 gated_attention_hidden_dim: int = 0,
                 gated_attention_dropout: float = 0.0,
                 gated_attention_init_bias: float = 2.0,
                 norm_type: str = "layernorm",
                 # Temperature scaling control
                 disable_temperature: bool = False):
        super(TopKSparseAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim if attention_dim is not None else embedding_dim
        self.top_k = top_k
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.init_method = init_method
        self.init_method_params = init_method_params or {}
        self.disable_temperature = disable_temperature
        self.use_gated_attention = use_gated_attention
        self.gated_attention_mode = gated_attention_mode
        self.gated_attention_hidden_dim = gated_attention_hidden_dim
        self.gated_attention_dropout = gated_attention_dropout
        self.gated_attention_init_bias = gated_attention_init_bias
        self.norm_type = norm_type
        
        # Projection layers
        if use_lora:
            self.W_Q = LoRALinear(embedding_dim, self.attention_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
            self.W_K = LoRALinear(embedding_dim, self.attention_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
            self.W_V = LoRALinear(embedding_dim, embedding_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
        else:
            self.W_Q = nn.Linear(embedding_dim, self.attention_dim, bias=True)
            self.W_K = nn.Linear(embedding_dim, self.attention_dim, bias=True)
            self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=True)

        if self.use_gated_attention:
            self.attention_output_gate = AttentionOutputGate(
                embedding_dim=self.embedding_dim,
                mode=self.gated_attention_mode,
                hidden_dim=self.gated_attention_hidden_dim,
                dropout=self.gated_attention_dropout,
                init_bias=self.gated_attention_init_bias,
                norm_type=self.norm_type,
            )
        else:
            self.attention_output_gate = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using the specified initialization method."""
        layers = [self.W_Q, self.W_K, self.W_V]
        
        print(f" Initializing top-k sparse attention with method: {self.init_method}")
        from initialization import initialize_attention_weights
        initialize_attention_weights(
            layers=layers,
            attention_dim=self.attention_dim,
            method=self.init_method,
            method_params=self.init_method_params
        )
        print(f" Successfully applied {self.init_method} initialization to top-k sparse attention")
    
    def forward(self, queries_emb: torch.Tensor, keys_emb: torch.Tensor, values_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply top-K sparse attention.
        
        Args:
            queries_emb: [batch_size, num_queries, embedding_dim]
            keys_emb: [batch_size, num_keys, embedding_dim] 
            values_emb: [batch_size, num_keys, embedding_dim]
            
        Returns:
            Tuple of (context_vectors, sparse_attention_weights)
        """
        batch_size, num_queries, _ = queries_emb.shape
        batch_size, num_keys, _ = keys_emb.shape
        
        # Project to attention space
        Q = self.W_Q(queries_emb)  # [batch_size, num_queries, attention_dim]
        K = self.W_K(keys_emb)     # [batch_size, num_keys, attention_dim]
        V = self.W_V(values_emb)   # [batch_size, num_keys, embedding_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_queries, num_keys]
        attention_scores = (attention_scores / (self.attention_dim ** 0.5)).to(attention_scores.dtype)
        
        # Apply temperature
        if not self.disable_temperature:
            temperature = torch.clamp(self.temperature, min=0.1, max=5.0).to(attention_scores.dtype)
            attention_scores = attention_scores / temperature
            
        # **KEY INNOVATION: Top-K Sparsity**
        # For each query, keep only top-K key similarities
        k = min(self.top_k, num_keys)
        
        # Get top-K indices and values for each query
        topk_values, topk_indices = torch.topk(attention_scores, k=k, dim=-1)  # [batch_size, num_queries, k]
        
        # Create sparse attention scores
        sparse_scores = torch.full_like(attention_scores, -1e9, dtype=attention_scores.dtype)  # Initialize with very negative values
        
        # Scatter top-K values back to their positions
        # Must cast topk_indices to int64 for scatter_
        sparse_scores.scatter_(-1, topk_indices.to(torch.int64), topk_values.to(attention_scores.dtype))
        
        # Apply softmax only over the top-K (others are effectively zero)
        attention_weights = F.softmax(sparse_scores, dim=-1)  # [batch_size, num_queries, num_keys]
        
        # Compute context vectors
        context_vectors = torch.bmm(attention_weights, V)  # [batch_size, num_queries, embedding_dim]

        if self.attention_output_gate is not None:
            context_vectors = self.attention_output_gate(queries_emb, context_vectors)
        
        return context_vectors, attention_weights

class WindowedCrossAttention(nn.Module):
    """
    Windowed Cross-Attention that creates attention windows based on content similarity.
    
    Instead of fixed position windows, this creates dynamic windows based on:
    1. Content similarity between queries and keys
    2. Learnable window size per query
    3. Adaptive window positioning
    
    This prevents attention collapse by forcing attention to be local and focused.
    """
    def __init__(self, 
                 embedding_dim: int,
                 attention_dim: int = None,
                 window_size: int = 5,
                 adaptive_window: bool = True,
                 temperature: float = 1.0,
                 use_lora: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.1,
                 init_method: str = "orthogonal",
                 init_method_params: dict = None,
                 # Gated attention overlay (post-SDPA gating)
                 use_gated_attention: bool = False,
                 gated_attention_mode: str = "scalar",
                 gated_attention_hidden_dim: int = 0,
                 gated_attention_dropout: float = 0.0,
                 gated_attention_init_bias: float = 2.0,
                 norm_type: str = "layernorm",
                 # Temperature scaling control
                 disable_temperature: bool = False):
        super(WindowedCrossAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim if attention_dim is not None else embedding_dim
        self.window_size = window_size
        self.adaptive_window = adaptive_window
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.init_method = init_method
        self.init_method_params = init_method_params or {}
        self.disable_temperature = disable_temperature
        self.use_gated_attention = use_gated_attention
        self.gated_attention_mode = gated_attention_mode
        self.gated_attention_hidden_dim = gated_attention_hidden_dim
        self.gated_attention_dropout = gated_attention_dropout
        self.gated_attention_init_bias = gated_attention_init_bias
        self.norm_type = norm_type
        
        # Projection layers
        if use_lora:
            self.W_Q = LoRALinear(embedding_dim, self.attention_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
            self.W_K = LoRALinear(embedding_dim, self.attention_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
            self.W_V = LoRALinear(embedding_dim, embedding_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
        else:
            self.W_Q = nn.Linear(embedding_dim, self.attention_dim, bias=True)
            self.W_K = nn.Linear(embedding_dim, self.attention_dim, bias=True)
            self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=True)

        if self.use_gated_attention:
            self.attention_output_gate = AttentionOutputGate(
                embedding_dim=self.embedding_dim,
                mode=self.gated_attention_mode,
                hidden_dim=self.gated_attention_hidden_dim,
                dropout=self.gated_attention_dropout,
                init_bias=self.gated_attention_init_bias,
                norm_type=self.norm_type,
            )
        else:
            self.attention_output_gate = None
        
        # Adaptive window size predictor (if enabled)
        if adaptive_window:
            self.window_predictor = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output in [0, 1], will be scaled to [1, max_window]
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Centralized initialization for Q/K/V."""
        from initialization import initialize_attention_weights
        initialize_attention_weights(
            layers=[self.W_Q, self.W_K, self.W_V],
            attention_dim=self.attention_dim,
            method=getattr(self, 'init_method', 'orthogonal'),
            method_params=getattr(self, 'init_method_params', None)
        )
    
    def _create_content_based_windows(self, Q: torch.Tensor, K: torch.Tensor, 
                                    queries_emb: torch.Tensor) -> torch.Tensor:
        """
        Create attention windows based on content similarity.
        
        Args:
            Q: Query projections [batch_size, num_queries, attention_dim]
            K: Key projections [batch_size, num_keys, attention_dim]
            queries_emb: Original query embeddings for window size prediction
            
        Returns:
            Window mask [batch_size, num_queries, num_keys]
        """
        batch_size, num_queries, _ = Q.shape
        batch_size, num_keys, _ = K.shape
        
        # Compute initial similarity scores for window positioning
        similarity_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_queries, num_keys]
        
        # Predict adaptive window sizes if enabled
        if self.adaptive_window:
            window_sizes = self.window_predictor(queries_emb)  # [batch_size, num_queries, 1]
            window_sizes = 1 + (self.window_size - 1) * window_sizes  # Scale to [1, window_size]
            window_sizes = window_sizes.squeeze(-1).round().long()  # [batch_size, num_queries]
        else:
            window_sizes = torch.full((batch_size, num_queries), self.window_size, 
                                    dtype=torch.long, device=Q.device)
        
        # Create window masks
        window_mask = torch.zeros_like(similarity_scores, dtype=torch.bool)  # [batch_size, num_queries, num_keys]
        
        for b in range(batch_size):
            for q in range(num_queries):
                # Find the center of attention window based on max similarity
                center_idx = torch.argmax(similarity_scores[b, q]).item()
                
                # Calculate window boundaries
                window_size = window_sizes[b, q].item()
                half_window = window_size // 2
                
                start_idx = max(0, center_idx - half_window)
                end_idx = min(num_keys, center_idx + half_window + 1)
                
                # Set window mask
                window_mask[b, q, start_idx:end_idx] = True
        
        return window_mask
    
    def forward(self, queries_emb: torch.Tensor, keys_emb: torch.Tensor, values_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply windowed cross-attention.
        
        Args:
            queries_emb: [batch_size, num_queries, embedding_dim]
            keys_emb: [batch_size, num_keys, embedding_dim]
            values_emb: [batch_size, num_keys, embedding_dim]
            
        Returns:
            Tuple of (context_vectors, windowed_attention_weights)
        """
        batch_size, num_queries, _ = queries_emb.shape
        
        # Project to attention space
        Q = self.W_Q(queries_emb)  # [batch_size, num_queries, attention_dim]
        K = self.W_K(keys_emb)     # [batch_size, num_keys, attention_dim]
        V = self.W_V(values_emb)   # [batch_size, num_keys, embedding_dim]
        
        # Create content-based attention windows
        window_mask = self._create_content_based_windows(Q, K, queries_emb)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_queries, num_keys]
        attention_scores = attention_scores / (self.attention_dim ** 0.5)
        
        # Apply temperature
        if not self.disable_temperature:
            temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
            attention_scores = attention_scores / temperature
        
        # **KEY INNOVATION: Window-based masking**
        # Mask out attention scores outside the windows
        masked_scores = attention_scores.masked_fill(~window_mask, -1e9)
        
        # Apply softmax only within windows
        attention_weights = F.softmax(masked_scores, dim=-1)  # [batch_size, num_queries, num_keys]
        
        # Compute context vectors
        context_vectors = torch.bmm(attention_weights, V)  # [batch_size, num_queries, embedding_dim]

        if self.attention_output_gate is not None:
            context_vectors = self.attention_output_gate(queries_emb, context_vectors)
        
        return context_vectors, attention_weights 

class ThresholdAttention(nn.Module):
    """
    Threshold-based Cross-Attention that only attends to semantically similar pairs.
    
    This prevents attention collapse by:
    1. Computing similarity scores between queries and keys
    2. Applying a learned threshold to filter weak connections
    3. Only computing attention over above-threshold connections
    4. Using content-based adaptive thresholds
    
    Perfect for table-text alignment where only meaningful connections matter.
    """
    def __init__(self, 
                 embedding_dim: int,
                 attention_dim: int = None,
                 base_threshold: float = 0.1,
                 adaptive_threshold: bool = True,
                 min_connections: int = 2,
                 temperature: float = 1.0,
                 use_lora: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.1,
                 init_method: str = "orthogonal",
                 init_method_params: dict = None,
                 # Gated attention overlay (post-SDPA gating)
                 use_gated_attention: bool = False,
                 gated_attention_mode: str = "scalar",
                 gated_attention_hidden_dim: int = 0,
                 gated_attention_dropout: float = 0.0,
                 gated_attention_init_bias: float = 2.0,
                 norm_type: str = "layernorm",
                 # Temperature scaling control
                 disable_temperature: bool = False):
        super(ThresholdAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim if attention_dim is not None else embedding_dim
        self.base_threshold = base_threshold
        self.adaptive_threshold = adaptive_threshold
        self.min_connections = min_connections
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.init_method = init_method
        self.init_method_params = init_method_params or {}
        self.disable_temperature = disable_temperature
        self.use_gated_attention = use_gated_attention
        self.gated_attention_mode = gated_attention_mode
        self.gated_attention_hidden_dim = gated_attention_hidden_dim
        self.gated_attention_dropout = gated_attention_dropout
        self.gated_attention_init_bias = gated_attention_init_bias
        self.norm_type = norm_type
        
        # Projection layers
        if use_lora:
            self.W_Q = LoRALinear(embedding_dim, self.attention_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
            self.W_K = LoRALinear(embedding_dim, self.attention_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
            self.W_V = LoRALinear(embedding_dim, embedding_dim, 
                                rank=lora_rank, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=True)
        else:
            self.W_Q = nn.Linear(embedding_dim, self.attention_dim, bias=True)
            self.W_K = nn.Linear(embedding_dim, self.attention_dim, bias=True)
            self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=True)

        if self.use_gated_attention:
            self.attention_output_gate = AttentionOutputGate(
                embedding_dim=self.embedding_dim,
                mode=self.gated_attention_mode,
                hidden_dim=self.gated_attention_hidden_dim,
                dropout=self.gated_attention_dropout,
                init_bias=self.gated_attention_init_bias,
                norm_type=self.norm_type,
            )
        else:
            self.attention_output_gate = None
        
        # Adaptive threshold predictor
        if adaptive_threshold:
            self.threshold_predictor = nn.Sequential(
                nn.Linear(embedding_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Centralized initialization for Q/K/V."""
        from initialization import initialize_attention_weights
        initialize_attention_weights(
            layers=[self.W_Q, self.W_K, self.W_V],
            attention_dim=self.attention_dim,
            method=getattr(self, 'init_method', 'orthogonal'),
            method_params=getattr(self, 'init_method_params', None)
        )
    
    def _compute_adaptive_thresholds(self, queries_emb: torch.Tensor, 
                                   similarity_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive thresholds based on query content and score statistics.
        
        Args:
            queries_emb: [batch_size, num_queries, embedding_dim]
            similarity_scores: [batch_size, num_queries, num_keys]
            
        Returns:
            Adaptive thresholds [batch_size, num_queries, 1]
        """
        if not self.adaptive_threshold:
            # Use fixed threshold
            batch_size, num_queries = similarity_scores.shape[:2]
            return torch.full((batch_size, num_queries, 1), self.base_threshold, 
                            device=queries_emb.device, dtype=queries_emb.dtype)
        
        # Predict base threshold from query content
        content_thresholds = self.threshold_predictor(queries_emb)  # [batch_size, num_queries, 1]
        
        # Adapt based on score statistics (percentile-based)
        score_percentiles = torch.quantile(similarity_scores, q=0.7, dim=-1, keepdim=True)  # [batch_size, num_queries, 1]
        
        # Combine content-based and statistics-based thresholds
        adaptive_thresholds = content_thresholds * self.base_threshold + (1 - content_thresholds) * score_percentiles
        
        return adaptive_thresholds
    
    def forward(self, queries_emb: torch.Tensor, keys_emb: torch.Tensor, values_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply threshold-based cross-attention.
        
        Args:
            queries_emb: [batch_size, num_queries, embedding_dim]
            keys_emb: [batch_size, num_keys, embedding_dim]
            values_emb: [batch_size, num_keys, embedding_dim]
            
        Returns:
            Tuple of (context_vectors, threshold_attention_weights)
        """
        batch_size, num_queries, _ = queries_emb.shape
        batch_size, num_keys, _ = keys_emb.shape
        
        # Project to attention space
        Q = self.W_Q(queries_emb)  # [batch_size, num_queries, attention_dim]
        K = self.W_K(keys_emb)     # [batch_size, num_keys, attention_dim]
        V = self.W_V(values_emb)   # [batch_size, num_keys, embedding_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_queries, num_keys]
        attention_scores = (attention_scores / (self.attention_dim ** 0.5)).to(attention_scores.dtype)
        
        # Apply temperature
        if not self.disable_temperature:
            temperature = torch.clamp(self.temperature, min=0.1, max=5.0).to(attention_scores.dtype)
            attention_scores = attention_scores / temperature
        
        # **KEY INNOVATION: Threshold-based filtering**
        # Compute adaptive thresholds
        thresholds = self._compute_adaptive_thresholds(queries_emb, attention_scores)  # [batch_size, num_queries, 1]
        
        # Create threshold mask
        threshold_mask = attention_scores >= thresholds  # [batch_size, num_queries, num_keys]
        
        # Ensure minimum connections per query (prevent complete disconnection)
        for b in range(batch_size):
            for q in range(num_queries):
                if threshold_mask[b, q].sum() < self.min_connections:
                    # Keep top min_connections if threshold is too strict
                    # But ensure we don't ask for more than available keys
                    actual_k = min(self.min_connections, num_keys)
                    _, top_indices = torch.topk(attention_scores[b, q], k=actual_k)
                    threshold_mask[b, q].fill_(False)
                    threshold_mask[b, q, top_indices] = True
        
        # Apply threshold mask
        masked_scores = attention_scores.masked_fill(~threshold_mask, -1e9)
        
        # Apply softmax only over above-threshold connections
        attention_weights = F.softmax(masked_scores, dim=-1)  # [batch_size, num_queries, num_keys]
        
        # Compute context vectors
        context_vectors = torch.bmm(attention_weights, V)  # [batch_size, num_queries, embedding_dim]

        if self.attention_output_gate is not None:
            context_vectors = self.attention_output_gate(queries_emb, context_vectors)
        
        return context_vectors, attention_weights



    def _init_weights(self):
        """Initialize weights using the new initialization system (for standard attention only)."""
        # Check if this is being called from a BidirectionalCrossAttention instance
        if hasattr(self, 'attention_type') and self.attention_type != "standard":
            return  # Other attention types handle their own initialization
            
        # Create list of layers for initialization (only for standard attention)
        if not hasattr(self, 'forward_W_Q'):
            # This means we're in a non-standard attention class, skip
            return
            
        layers = [
            self.forward_W_Q, self.forward_W_K, self.forward_W_V,
            self.reverse_W_Q, self.reverse_W_K, self.reverse_W_V
        ]
        
        if self.verbose:
            print(f" Initializing bidirectional attention with method: {self.init_method}")
        
        # Apply the selected initialization method - NO FALLBACK!
        from initialization import initialize_attention_weights
        initialize_attention_weights(
            layers=layers,
            attention_dim=self.attention_dim,
            method=self.init_method,
            method_params=self.init_method_params
        )
        if self.verbose:
            print(f" Successfully applied {self.init_method} initialization")

    def _apply_attention(self, queries, keys, values, temperature):
        """Apply scaled dot-product attention with temperature (for standard attention only)."""
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Scale by sqrt of dimension
        attention_scores = (attention_scores / (self.attention_dim ** 0.5)).to(attention_scores.dtype)
        
        # Apply temperature scaling with better stability (higher temps = softer attention)
        temperature = torch.clamp(temperature, min=1.0, max=5.0).to(attention_scores.dtype)
        attention_scores = attention_scores / temperature
        
        # Add numerical stability to prevent overflow/underflow
        attention_scores = torch.clamp(attention_scores, min=-50.0, max=50.0).to(attention_scores.dtype)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Check for NaN in attention weights and replace with uniform if needed
        if torch.isnan(attention_weights).any():
            print("Warning: NaN detected in attention weights, using uniform distribution")
            seq_len = attention_weights.shape[-1]
            attention_weights = torch.ones_like(attention_weights) / seq_len
        
        # Apply attention to values
        context_vectors = torch.bmm(attention_weights, values)
        
        return context_vectors, attention_weights