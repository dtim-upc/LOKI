import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, List, Dict, Any, Tuple
from sentence_transformers import util


def _normalize_schema_texts(schema_texts: Any) -> List[str]:
    if schema_texts is None:
        return []
    if isinstance(schema_texts, str):
        schema_text = schema_texts.strip()
        return [schema_text] if schema_text else []
    if isinstance(schema_texts, list):
        return [str(schema_text).strip() for schema_text in schema_texts if str(schema_text).strip()]
    return []


def _encode_schema_texts(model, schema_texts: Any, batch_size: int = 32) -> Optional[torch.Tensor]:
    normalized_texts = _normalize_schema_texts(schema_texts)
    if not normalized_texts:
        return None
    return model.encode_sentences(
        normalized_texts,
        batch_size=min(batch_size, len(normalized_texts)),
    )


def _encode_cell_text_rows(model, cell_text_rows: Any, batch_size: int = 32) -> Optional[torch.Tensor]:
    if not cell_text_rows:
        return None

    num_rows = len(cell_text_rows)
    max_cols = max((len(row) for row in cell_text_rows), default=0)
    if max_cols == 0:
        return None

    flat_texts: List[str] = []
    flat_positions: List[Tuple[int, int]] = []
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
    )
    cell_grid = torch.zeros(num_rows, max_cols, flat_embeddings.shape[-1], device=flat_embeddings.device, dtype=flat_embeddings.dtype)
    for embedding_index, (row_index, col_index) in enumerate(flat_positions):
        cell_grid[row_index, col_index] = flat_embeddings[embedding_index]
    return cell_grid


def _build_padded_schema_batch(
    ids: List[Any],
    schema_dict: Dict[Any, Optional[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    embedding_dim: int,
) -> Optional[torch.Tensor]:
    schema_tensors: List[Optional[torch.Tensor]] = []
    max_schema_len = 0

    for item_id in ids:
        schema_emb = schema_dict.get(item_id)
        if schema_emb is None:
            schema_tensors.append(None)
            continue
        if schema_emb.dim() > 2 and schema_emb.size(0) == 1:
            schema_emb = schema_emb.squeeze(0)
        if schema_emb.dim() == 1:
            schema_emb = schema_emb.unsqueeze(0)
        schema_emb = schema_emb.to(device=device, dtype=dtype)
        max_schema_len = max(max_schema_len, schema_emb.size(0))
        schema_tensors.append(schema_emb)

    if max_schema_len == 0:
        return None

    padded_schema_tensors = []
    for schema_emb in schema_tensors:
        if schema_emb is None:
            padded_schema_tensors.append(torch.zeros(max_schema_len, embedding_dim, device=device, dtype=dtype))
            continue
        if schema_emb.size(0) < max_schema_len:
            pad_rows = max_schema_len - schema_emb.size(0)
            padding = torch.zeros(pad_rows, embedding_dim, device=device, dtype=dtype)
            schema_emb = torch.cat([schema_emb, padding], dim=0)
        padded_schema_tensors.append(schema_emb)

    return torch.stack(padded_schema_tensors, dim=0)


def _build_padded_cell_batch(
    ids: List[Any],
    cell_dict: Dict[Any, Optional[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    embedding_dim: int,
    target_row_count: Optional[int] = None,
) -> Optional[torch.Tensor]:
    cell_tensors: List[Optional[torch.Tensor]] = []
    max_row_len = int(target_row_count or 0)
    max_col_len = 0

    for item_id in ids:
        cell_emb = cell_dict.get(item_id)
        if cell_emb is None:
            cell_tensors.append(None)
            continue
        if cell_emb.dim() > 3 and cell_emb.size(0) == 1:
            cell_emb = cell_emb.squeeze(0)
        if cell_emb.dim() == 2:
            cell_emb = cell_emb.unsqueeze(1)
        cell_emb = cell_emb.to(device=device, dtype=dtype)
        max_row_len = max(max_row_len, cell_emb.size(0))
        max_col_len = max(max_col_len, cell_emb.size(1))
        cell_tensors.append(cell_emb)

    if max_row_len == 0 or max_col_len == 0:
        return None

    padded_cell_tensors = []
    for cell_emb in cell_tensors:
        if cell_emb is None:
            padded_cell_tensors.append(torch.zeros(max_row_len, max_col_len, embedding_dim, device=device, dtype=dtype))
            continue

        if cell_emb.size(1) < max_col_len:
            pad_cols = max_col_len - cell_emb.size(1)
            col_padding = torch.zeros(cell_emb.size(0), pad_cols, embedding_dim, device=device, dtype=dtype)
            cell_emb = torch.cat([cell_emb, col_padding], dim=1)
        if cell_emb.size(0) < max_row_len:
            pad_rows = max_row_len - cell_emb.size(0)
            row_padding = torch.zeros(pad_rows, max_col_len, embedding_dim, device=device, dtype=dtype)
            cell_emb = torch.cat([cell_emb, row_padding], dim=0)

        padded_cell_tensors.append(cell_emb)

    return torch.stack(padded_cell_tensors, dim=0)

def sigreg_loss(
    embeddings: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    SIGReg-style regularization (Spectral Invariance + Covariance).

    Inspired by VICReg / SIGReg from I-JEPA:
      variance_loss:  each feature dim should have std >= target_std
      covariance_loss: off-diagonal of the covariance matrix should -> 0

    Operates on the *contextualized* row or sentence embeddings produced by
    cross-attention (NOT the raw encoder embeddings).  Applying it there
    prevents mode-collapse in the representation space that cross-attention
    produces.

    Args:
        embeddings: [B, N, D] contextualized vectors (batch, sequence, dim)
        target_std: desired minimum per-dimension standard deviation
        eps: small constant for numerical stability

    Returns:
        Scalar loss = variance_loss + covariance_loss
    """
    if embeddings.numel() == 0 or embeddings.shape[-1] == 0:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    # Flatten batch + sequence into one sample dimension: [B*N, D]
    flat = embeddings.reshape(-1, embeddings.shape[-1]).float()
    n_samples = flat.shape[0]
    if n_samples < 2:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    # ---- Variance term: each dimension should have std >= target_std ----
    std = flat.std(dim=0, unbiased=False)
    # hinge: penalize only if std < target_std
    variance_loss = F.relu(target_std - std).mean()

    # ---- Covariance term: off-diagonal of cov matrix should be zero ----
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(n_samples - 1, 1)
    D = cov.shape[0]
    # Zero out diagonal, keep off-diagonal L2
    diag_mask = torch.eye(D, device=cov.device, dtype=torch.bool)
    off_diag = cov.masked_fill(diag_mask, 0.0)
    covariance_loss = off_diag.pow(2).sum() / max(D, 1)

    loss = variance_loss + covariance_loss
    return torch.nan_to_num(loss.to(dtype=embeddings.dtype), nan=0.0, posinf=0.0, neginf=0.0)


class EppsPulleySIGReg(nn.Module):
    """
    True SIGReg: Sketch Isotropic Gaussian Regularizer using Epps-Pulley test.
    
    From LeWorldModel (Maes et al., 2026):
    Projects embeddings onto M random unit-norm directions and tests univariate
    normality via the Epps-Pulley characteristic function test.
    
    By the Cramér-Wold theorem, matching all 1D marginals = matching the full
    joint distribution -> enforces isotropic Gaussian without explicit covariance
    matrix computation (which is O(D^2) and memory-heavy).
    
    Replaces the old VICReg-style sigreg_loss() which only checked variance
    hinge + off-diagonal covariance.
    
    Args:
        knots: Number of quadrature knots for the Epps-Pulley integral (default: 17)
        num_proj: Number of random projection directions (default: 1024)
    """
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, N, D] contextualized vectors (batch, sequence, dim)
        Returns:
            Scalar loss
        """
        if embeddings.numel() == 0 or embeddings.shape[-1] == 0:
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        
        # Reshape to [N, B, D] (LeWM convention: first dim = time/sequence)
        if embeddings.dim() == 3:
            proj = embeddings.transpose(0, 1).float()  # [N, B, D]
        elif embeddings.dim() == 2:
            proj = embeddings.unsqueeze(0).float()  # [1, B, D]
        else:
            proj = embeddings.float()
        
        if proj.shape[-2] < 2:
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        
        # Sample random unit-norm projections
        D = proj.size(-1)
        A = torch.randn(D, self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        
        # Compute the Epps-Pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t.to(device=proj.device, dtype=proj.dtype)
        err = (x_t.cos().mean(-3) - self.phi.to(device=proj.device, dtype=proj.dtype)).square() \
              + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(device=proj.device, dtype=proj.dtype)) * proj.size(-2)
        
        result = statistic.mean()
        return torch.nan_to_num(
            result.to(dtype=embeddings.dtype), nan=0.0, posinf=0.0, neginf=0.0
        )


def sinkhorn_reg_loss(
    attention_weights: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    key_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sinkhorn-style marginal constraint on attention matrices.

    Prevents hub keys (sentences that absorb attention from *all* queries)
    by penalising deviation of column marginals from the uniform target.

    For a rectangular attention matrix [N_q, N_k] normalised row-wise
    (each row sums to 1), the doubly-stochastic target column sum is
    N_q / N_k.  We penalise  || col_sums - target ||^2.

    We also penalise excessive *variance* across column sums within each
    batch element, which directly measures how "hub-like" the distribution
    is (variance == 0 means perfectly balanced).

    Args:
        attention_weights: [B, N_q, N_k]  (assumed row-normalised)
        query_mask: [B, N_q]  bool – True for valid queries
        key_mask:   [B, N_k]  bool – True for valid keys

    Returns:
        Scalar loss
    """
    if attention_weights.numel() == 0:
        return torch.tensor(0.0, device=attention_weights.device, dtype=attention_weights.dtype)

    if attention_weights.dim() == 2:
        attention_weights = attention_weights.unsqueeze(0)
    if query_mask is not None and query_mask.dim() == 1:
        query_mask = query_mask.unsqueeze(0)
    if key_mask is not None and key_mask.dim() == 1:
        key_mask = key_mask.unsqueeze(0)

    losses = []
    eps = 1e-8
    for b in range(attention_weights.shape[0]):
        attn = attention_weights[b]  # [N_q, N_k]

        # Mask valid queries / keys
        if query_mask is not None:
            valid_q = query_mask[b].to(device=attn.device, dtype=torch.bool)
            attn = attn[valid_q]
        if key_mask is not None:
            valid_k = key_mask[b].to(device=attn.device, dtype=torch.bool)
            attn = attn[:, valid_k]

        N_q, N_k = attn.shape[-2], attn.shape[-1]
        if N_q < 1 or N_k < 1:
            continue

        # Ensure non-negative and row-normalised
        attn = torch.clamp(attn, min=0.0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(eps)

        # Column marginals
        col_sums = attn.sum(dim=-2)  # [N_k]
        target_col = float(N_q) / float(N_k)

        # L2 deviation from uniform target
        target = torch.full_like(col_sums, target_col)
        marginal_loss = F.mse_loss(col_sums, target)

        # Column-sum variance: additional hub penalty (perfectly balanced -> 0)
        col_var = col_sums.var(unbiased=False)

        loss = marginal_loss + col_var
        losses.append(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))

    if not losses:
        return torch.tensor(0.0, device=attention_weights.device, dtype=attention_weights.dtype)
    return torch.stack(losses).mean()


class IdBasedCachedTripletLoss(nn.Module):
    """
    Optimized triplet loss that uses the ID-based embedding cache for efficiency.
    
    This loss is designed to work with our new ID-based triplet format and caching system,
    leveraging pre-computed embeddings for tables and contexts to minimize redundant
    computation during training.
    
    OPTIMIZED: Now implements true batch processing by recognizing that all triplets
    in a batch share the same anchor and batching unique positive/negative contexts.
    """
    def __init__(self, model, cache=None, margin: float = 0.3, scale: float = 10.0,
                 use_hard_negative_mining: bool = False, hard_negative_topk: int = 0):
        """
        Initialize the cached triplet loss.
        
        Args:
            model: The TableTextEmbeddingModel
            cache: Optional IdBasedEmbeddingCache instance
            margin: Minimum margin between positive and negative scores
            scale: Scale factor for the similarity scores (amplified differences for better gradients)
        """
        super(IdBasedCachedTripletLoss, self).__init__()
        self.model = model
        self.cache = cache
        self.margin = margin
        self.scale = scale
        print(f"Initialized IdBasedCachedTripletLoss with margin={margin}, scale={scale}")
        print(f"Note: This loss function uses only triplet loss (no additional components)")
        self.use_hard_negative_mining = use_hard_negative_mining
        self.hard_negative_topk = hard_negative_topk
    
    def forward(self, triplet_batch):
        """
        Compute triplet loss using cached embeddings with optimized tensor operations.
        
        Key optimization: Full parallelization of all processing:
        1. Fetch anchor embeddings once
        2. Fetch unique positive/negative embeddings once
        3. Batch all operations with efficient tensor operations
        4. Single forward pass for all positives and all negatives
        
        Args:
            triplet_batch: A batch of triplets with anchor_id, positive_id, and negative_id
                          NOTE: Handles empty additional_positives correctly - will process 
                          only primary_positive if additional_positives is empty list
        
        Returns:
            The average triplet loss for the batch
        """
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # Cache is optional - if None, we'll compute embeddings on-the-fly
        
        # Handle empty triplet batch (can occur when additional_positives is empty and no valid triplets generated)
        if not triplet_batch:
            return torch.tensor(0.0, device=device, dtype=model_dtype, requires_grad=True)
        
        # OPTIMIZATION: Extract unique IDs once
        anchor_id = triplet_batch[0]['anchor_id']
        positive_ids = [triplet['positive_id'] for triplet in triplet_batch]
        negative_ids = [triplet['negative_id'] for triplet in triplet_batch]
        
        # OPTIMIZATION: Get anchor embeddings from cache or compute on-the-fly
        anchor_embeddings = None
        if self.cache is not None:
            anchor_embeddings = self.cache.get_table_embeddings(anchor_id)
        
        if anchor_embeddings is None:
            # Compute on-the-fly (either no cache or cache miss)
            anchor_rows = triplet_batch[0]['anchor_rows']
            row_embeddings = []
            for row in anchor_rows:
                emb = self.model.encode_sentences([row], batch_size=1)[0]
                row_embeddings.append(emb)
            anchor_embeddings = torch.stack(row_embeddings)
        
        # OPTIMIZATION: Extract unique IDs for cache lookup
        unique_positive_ids = list(set(positive_ids))
        unique_negative_ids = list(set(negative_ids))
        
        # OPTIMIZATION: Batch fetch all positive contexts at once or compute on-the-fly
        positive_embeddings_dict = {}
        for pos_id in unique_positive_ids:
            pos_emb = None
            if self.cache is not None:
                pos_emb = self.cache.get_context_embeddings(pos_id)
            
            if pos_emb is None:
                # Compute on-the-fly - find the context from any triplet with this positive_id
                for triplet in triplet_batch:
                    if triplet['positive_id'] == pos_id:
                        pos_context = triplet.get('positive_texts', triplet.get('positive_context', []))
                        pos_embeddings = []
                        for sent in pos_context:
                            emb = self.model.encode_sentences([sent], batch_size=1)[0]
                            pos_embeddings.append(emb)
                        pos_emb = torch.stack(pos_embeddings)
                        break
            if pos_emb is not None:
                positive_embeddings_dict[pos_id] = pos_emb
        
        # OPTIMIZATION: Batch fetch all negative contexts at once or compute on-the-fly
        negative_embeddings_dict = {}
        for neg_id in unique_negative_ids:
            neg_emb = None
            if self.cache is not None:
                neg_emb = self.cache.get_context_embeddings(neg_id)
            
            if neg_emb is None:
                # Compute on-the-fly - find the context from any triplet with this negative_id
                for triplet in triplet_batch:
                    if triplet['negative_id'] == neg_id:
                        neg_context = triplet.get('negative_texts', triplet.get('negative_context', []))
                        neg_embeddings = []
                        for sent in neg_context:
                            emb = self.model.encode_sentences([sent], batch_size=1)[0]
                            neg_embeddings.append(emb)
                        neg_emb = torch.stack(neg_embeddings)
                        break
            if neg_emb is not None:
                negative_embeddings_dict[neg_id] = neg_emb
        
        # Check if we have any valid embeddings
        if not positive_embeddings_dict or not negative_embeddings_dict:
            return torch.tensor(0.0, device=device, dtype=model_dtype, requires_grad=True)
        
        # OPTIMIZATION: Efficient padding with cached zero tensors
        def pad_embeddings_efficiently(embeddings_dict):
            if not embeddings_dict:
                return {}, 0, 0
            
            # Find max sequence length
            max_seq_len = max(emb.shape[0] for emb in embeddings_dict.values())
            embed_dim = list(embeddings_dict.values())[0].shape[1]
            
            # Pre-allocate padding tensors for different sizes
            padding_cache = {}
            padded_dict = {}
            
            for id_key, emb in embeddings_dict.items():
                seq_len = emb.shape[0]
                if seq_len < max_seq_len:
                    pad_size = max_seq_len - seq_len
                    # Use cached padding of this size if available
                    if pad_size not in padding_cache:
                        padding_cache[pad_size] = torch.zeros(pad_size, embed_dim, device=emb.device, dtype=emb.dtype)
                    padding = padding_cache[pad_size]
                    padded_emb = torch.cat([emb, padding], dim=0)
                else:
                    padded_emb = emb
                padded_dict[id_key] = padded_emb
            
            return padded_dict, max_seq_len, embed_dim
        
        # Apply efficient padding
        padded_positives, pos_max_len, embed_dim = pad_embeddings_efficiently(positive_embeddings_dict)
        padded_negatives, neg_max_len, _ = pad_embeddings_efficiently(negative_embeddings_dict)
        
        if not padded_positives or not padded_negatives:
            return torch.tensor(0.0, device=device, dtype=model_dtype, requires_grad=True)
        
        # OPTIMIZATION: Get filtered IDs (those that were successfully cached and padded)
        filtered_pos_ids = [pid for pid in unique_positive_ids if pid in padded_positives]
        filtered_neg_ids = [nid for nid in unique_negative_ids if nid in padded_negatives]
        
        if not filtered_pos_ids or not filtered_neg_ids:
            return torch.tensor(0.0, device=device, dtype=model_dtype, requires_grad=True)
        
        # OPTIMIZATION: Stack all embeddings once for maximum parallelism
        positive_embeddings_list = [padded_positives[pos_id] for pos_id in filtered_pos_ids]
        negative_embeddings_list = [padded_negatives[neg_id] for neg_id in filtered_neg_ids]
        
        # Stack with single operations
        batched_positives = torch.stack(positive_embeddings_list)  # [num_unique_pos, max_pos_len, embed_dim]
        batched_negatives = torch.stack(negative_embeddings_list)  # [num_unique_neg, max_neg_len, embed_dim]
        
        # Create efficient mapping from ID to tensor index
        pos_id_to_idx = {filtered_pos_ids[i]: i for i in range(len(filtered_pos_ids))}
        neg_id_to_idx = {filtered_neg_ids[i]: i for i in range(len(filtered_neg_ids))}
        
        # OPTIMIZATION: Expand anchor for batched computation in a single operation
        anchor_for_positives = anchor_embeddings.unsqueeze(0).expand(len(filtered_pos_ids), -1, -1)
        anchor_for_negatives = anchor_embeddings.unsqueeze(0).expand(len(filtered_neg_ids), -1, -1)
        
        # OPTIMIZATION: Perform batched forward passes for all contexts at once
        positive_scores, _ = self.model(anchor_for_positives, batched_positives)  # [num_unique_pos]
        negative_scores, _ = self.model(anchor_for_negatives, batched_negatives)  # [num_unique_neg]
        
        # OPTIMIZATION: Create lookup dictionaries for O(1) access
        positive_scores_dict = {filtered_pos_ids[i]: positive_scores[i] for i in range(len(filtered_pos_ids))}
        negative_scores_dict = {filtered_neg_ids[i]: negative_scores[i] for i in range(len(filtered_neg_ids))}
        
        # OPTIMIZATION: Pre-allocate list for collecting losses
        batch_size = len(triplet_batch)
        batch_losses = []
        batch_losses.reserve(batch_size) if hasattr(batch_losses, 'reserve') else None
        
        # Optional simple in-batch hard negative mining: replace each negative with the hardest available
        if self.use_hard_negative_mining and negative_scores is not None and positive_scores is not None:
            # For each positive, choose the top-k hardest negatives (highest score)
            neg_values, neg_idx = torch.topk(negative_scores, k=min(max(1, self.hard_negative_topk), negative_scores.shape[0]))
            # Map each triplet negative to the hardest global negative
            hardest_neg_value = neg_values[0] if neg_values.ndim == 1 else neg_values[:, 0]
            hardest_neg = hardest_neg_value
        else:
            hardest_neg = None

        # Process each triplet in the batch
        for triplet in triplet_batch:
            pos_id = triplet['positive_id']
            neg_id = triplet['negative_id']
            if pos_id not in positive_scores_dict:
                continue
            pos_score = positive_scores_dict[pos_id]
            if hardest_neg is not None:
                neg_score = hardest_neg
            else:
                if neg_id not in negative_scores_dict:
                    continue
                neg_score = negative_scores_dict[neg_id]
            diff = neg_score - pos_score + self.margin
            loss = F.softplus(diff * self.scale)
            batch_losses.append(loss)
        
        # Average the losses in a single operation
        if batch_losses:
            loss = torch.mean(torch.stack(batch_losses))
            # Add a small epsilon to ensure non-zero gradient
            loss = loss + 1e-6
            return loss
        else:
            # Return a small trainable value instead of zero
            return torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)

class EnhancedTripletLoss(nn.Module):
    """
    Enhanced triplet loss with configurable aggregation strategies and attention regularization.
    
    This advanced loss function supports multiple aggregation methods to prevent attention collapse:
    - Top-k methods: top_k_sum, top_k_mean, sparse_top_k, weighted_top_k
    - Single value methods: max, mean  
    - Attention-based: attention_weighted
    - Regularized methods: entropy_regularized
    
    Key innovations:
    1. Configurable aggregation prevents uniform attention patterns
    2. Attention regularization encourages focused, meaningful patterns
    3. Focused gradients create stronger learning signals
    4. OPTIMIZED: True batch processing for maximum GPU utilization
    
    This replaces the basic mean aggregation that often leads to attention collapse.
    """
    def __init__(self, 
                 model, 
                 cache=None, 
                 margin: float = 0.3, 
                 scale: float = 10.0,
                 aggregation_method: str = "entropy_regularized",
                 # FIXED: Normalized loss weights
                 triplet_weight: float = 0.8,      # Main triplet loss (80%)
                 attention_weight: float = 0.2,
                 use_hard_negative_mining: bool = False,
                 hard_negative_topk: int = 0):   # Attention regularization (20%)
        """
        Initialize the enhanced triplet loss with normalized weights.
        
        Args:
            model: The TableTextEmbeddingModel
            cache: Optional IdBasedEmbeddingCache instance
            margin: Minimum margin between positive and negative scores
            scale: Scale factor for the similarity scores (amplifies differences)
            aggregation_method: Method for aggregating row scores (see model for options)
            triplet_weight: Weight for main triplet loss (relative importance)
            attention_weight: Weight for attention regularization loss (relative importance)
        """
        super(EnhancedTripletLoss, self).__init__()
        self.model = model
        self.cache = cache
        self.margin = margin
        self.scale = scale
        self.aggregation_method = aggregation_method
        
        # FIXED: Normalize weights to sum to 1.0 for interpretable relative importance
        total_weight = triplet_weight + attention_weight
        if total_weight <= 0:
            raise ValueError("Sum of loss weights must be positive")
            
        self.triplet_weight = triplet_weight / total_weight
        self.attention_weight = attention_weight / total_weight
        
        print(f"Initialized EnhancedTripletLoss with margin={margin}, scale={scale}, aggregation={aggregation_method}")
        print(f"Normalized weights - Triplet: {self.triplet_weight:.3f}, Attention: {self.attention_weight:.3f}")
        print(f"Weight sum verification: {self.triplet_weight + self.attention_weight:.6f}")
        
        # Backward compatibility: store old parameter name for existing code
        self.attention_loss_weight = self.attention_weight
        self.use_hard_negative_mining = use_hard_negative_mining
        self.hard_negative_topk = hard_negative_topk
    
    def forward(self, triplet_batch):
        """
        Compute enhanced triplet loss with attention regularization and configurable aggregation.
        
        Args:
            triplet_batch: A batch of triplets with anchor_id, positive_id, and negative_id
                          NOTE: Handles empty additional_positives correctly - will process 
                          only primary_positive if additional_positives is empty list
        """
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        use_header_conditioning = bool(getattr(self.model, 'use_header_conditioning', False))
        use_cell_level_matching = bool(getattr(self.model, 'use_cell_level_matching', False))
        
        # Cache is optional - if None, we'll compute embeddings on-the-fly
        
        # Handle empty triplet batch (can occur when additional_positives is empty and no valid triplets generated)
        if not triplet_batch:
            return torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # Extract unique IDs (all triplets share same anchor in isolated batches)
        anchor_id = triplet_batch[0]['anchor_id']
        positive_ids = [triplet['positive_id'] for triplet in triplet_batch]
        negative_ids = [triplet['negative_id'] for triplet in triplet_batch]
        
        # OPTIMIZATION: Fetch anchor embeddings once from cache or compute on-the-fly
        anchor_embeddings = None
        if self.cache is not None:
            anchor_embeddings = self.cache.get_table_embeddings(anchor_id)
        
        if anchor_embeddings is None:
            # Compute on-the-fly (either no cache or cache miss)
            anchor_rows = triplet_batch[0].get('anchor_texts', triplet_batch[0].get('anchor_rows', []))
            row_embeddings = []
            for row in anchor_rows:
                emb = self.model.encode_sentences([row], batch_size=1)[0]
                row_embeddings.append(emb)
            anchor_embeddings = torch.stack(row_embeddings)
        
        # Extract unique positive and negative IDs for cache lookup
        unique_positive_ids = list(set(positive_ids))
        unique_negative_ids = list(set(negative_ids))
        
        # OPTIMIZATION: Batch fetch all positive contexts at once or compute on-the-fly
        positive_embeddings_dict = {}
        for pos_id in unique_positive_ids:
            pos_emb = None
            if self.cache is not None:
                pos_emb = self.cache.get_context_embeddings(pos_id)
            
            if pos_emb is None:
                # Compute on-the-fly - find the context from any triplet with this positive_id
                for triplet in triplet_batch:
                    if triplet['positive_id'] == pos_id:
                        pos_context = triplet.get('positive_texts', triplet.get('positive_context', []))
                        pos_embeddings = []
                        for sent in pos_context:
                            emb = self.model.encode_sentences([sent], batch_size=1)[0]
                            pos_embeddings.append(emb)
                        pos_emb = torch.stack(pos_embeddings)
                        break
            if pos_emb is not None:
                positive_embeddings_dict[pos_id] = pos_emb
        
        # OPTIMIZATION: Batch fetch all negative contexts at once or compute on-the-fly
        negative_embeddings_dict = {}
        for neg_id in unique_negative_ids:
            neg_emb = None
            if self.cache is not None:
                neg_emb = self.cache.get_context_embeddings(neg_id)
            
            if neg_emb is None:
                # Compute on-the-fly - find the context from any triplet with this negative_id
                for triplet in triplet_batch:
                    if triplet['negative_id'] == neg_id:
                        neg_context = triplet.get('negative_texts', triplet.get('negative_context', []))
                        neg_embeddings = []
                        for sent in neg_context:
                            emb = self.model.encode_sentences([sent], batch_size=1)[0]
                            neg_embeddings.append(emb)
                        neg_emb = torch.stack(neg_embeddings)
                        break
            if neg_emb is not None:
                negative_embeddings_dict[neg_id] = neg_emb
        
        if not positive_embeddings_dict or not negative_embeddings_dict:
            return torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # OPTIMIZATION: Efficient padding with cached zero tensors
        def pad_embeddings_efficiently(embeddings_dict):
            if not embeddings_dict:
                return {}, 0, 0
            
            # Find max sequence length for padding
            max_seq_len = max(emb.shape[0] for emb in embeddings_dict.values())
            embed_dim = list(embeddings_dict.values())[0].shape[1]
            
            # Pre-allocate padding tensor for efficiency
            padding_cache = {}
            padded_dict = {}
            
            for id_key, emb in embeddings_dict.items():
                seq_len = emb.shape[0]
                if seq_len < max_seq_len:
                    pad_size = max_seq_len - seq_len
                    # Use cached padding of this size if available
                    if pad_size not in padding_cache:
                        padding_cache[pad_size] = torch.zeros(pad_size, embed_dim, device=emb.device, dtype=emb.dtype)
                    padding = padding_cache[pad_size]
                    padded_emb = torch.cat([emb, padding], dim=0)
                else:
                    padded_emb = emb
                padded_dict[id_key] = padded_emb
            
            return padded_dict, max_seq_len, embed_dim
        
        # Apply efficient padding
        padded_positives, pos_max_len, embed_dim = pad_embeddings_efficiently(positive_embeddings_dict)
        padded_negatives, neg_max_len, _ = pad_embeddings_efficiently(negative_embeddings_dict)
        
        if not padded_positives or not padded_negatives:
            return torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # OPTIMIZATION: Stack embeddings once for efficient GPU processing
        # Get filtered IDs (those successfully retrieved from cache and padded)
        filtered_pos_ids = [pid for pid in unique_positive_ids if pid in padded_positives]
        filtered_neg_ids = [nid for nid in unique_negative_ids if nid in padded_negatives]
        
        if not filtered_pos_ids or not filtered_neg_ids:
            return torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # Stack embeddings with a single operation
        positive_embeddings_list = [padded_positives[pos_id] for pos_id in filtered_pos_ids]
        negative_embeddings_list = [padded_negatives[neg_id] for neg_id in filtered_neg_ids]
        
        # Single tensor operations for stacking contexts
        batched_positives = torch.stack(positive_embeddings_list)  # [num_unique_pos, max_pos_len, embed_dim]
        batched_negatives = torch.stack(negative_embeddings_list)  # [num_unique_neg, max_neg_len, embed_dim]
        
        # Create efficient mapping from ID to tensor index
        pos_id_to_idx = {filtered_pos_ids[i]: i for i in range(len(filtered_pos_ids))}
        neg_id_to_idx = {filtered_neg_ids[i]: i for i in range(len(filtered_neg_ids))}
        
        # OPTIMIZATION: Expand anchor in a single operation
        # This preserves relationship between anchor rows and context sentences
        anchor_for_positives = anchor_embeddings.unsqueeze(0).expand(len(filtered_pos_ids), -1, -1)
        anchor_for_negatives = anchor_embeddings.unsqueeze(0).expand(len(filtered_neg_ids), -1, -1)
        
        # OPTIMIZATION: Single forward pass for all anchors and contexts
        # This computes all row × sentence pairs in parallel for all contexts
        positive_scores, positive_attentions = self.model(
            anchor_for_positives, batched_positives, 
            aggregation_method=self.aggregation_method
        )  # [num_unique_pos], [num_unique_pos, num_rows, max_pos_len]
        
        negative_scores, negative_attentions = self.model(
            anchor_for_negatives, batched_negatives, 
            aggregation_method=self.aggregation_method
        )  # [num_unique_neg], [num_unique_neg, num_rows, max_neg_len]
        
        # OPTIMIZATION: Create dictionaries for O(1) ID lookup
        positive_scores_dict = {filtered_pos_ids[i]: positive_scores[i] for i in range(len(filtered_pos_ids))}
        negative_scores_dict = {filtered_neg_ids[i]: negative_scores[i] for i in range(len(filtered_neg_ids))}
        
        # OPTIMIZATION: Pre-allocate list for losses
        batch_size = len(triplet_batch)
        batch_losses = []
        batch_losses.reserve(batch_size) if hasattr(batch_losses, 'reserve') else None
        pair_losses = []
        pair_losses.reserve(batch_size) if hasattr(pair_losses, 'reserve') else None
        
        # Optional simple in-batch hard negative mining
        global_hardest_neg = None
        if self.use_hard_negative_mining and negative_scores is not None:
            k = min(max(1, self.hard_negative_topk), negative_scores.shape[0])
            vals, idx = torch.topk(negative_scores, k=k)
            # pick hardest (highest) negative score
            global_hardest_neg = vals[0] if vals.ndim == 1 else vals[:, 0]

        # Process all triplets in the batch
        for triplet in triplet_batch:
            pos_id = triplet['positive_id']
            neg_id = triplet['negative_id']
            
            # Skip triplets with missing embeddings
            if pos_id not in positive_scores_dict:
                continue
            
            pos_score = positive_scores_dict[pos_id]
            if global_hardest_neg is not None:
                neg_score = global_hardest_neg
            else:
                if neg_id not in negative_scores_dict:
                    continue
                neg_score = negative_scores_dict[neg_id]
            
            # Ranking loss switch: softplus (default) or InfoNCE
            if getattr(self, 'ranking_loss_type', 'softplus') == 'infonce':
                tau = getattr(self, 'infonce_tau', 0.7)
                # Build a tiny denominator with both positive and negative scores
                logits = torch.stack([pos_score, neg_score], dim=0) / max(tau, 1e-6)
                labels = torch.tensor(0, device=logits.device)
                triplet_loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
            else:
                diff = neg_score - pos_score + self.margin
                triplet_loss = F.softplus(diff * self.scale)
            batch_losses.append(triplet_loss)
        
        # FIXED: Properly combine losses with normalization and weighting
        if batch_losses:
            # Compute individual loss components
            triplet_loss = torch.mean(torch.stack(batch_losses))
            
            # Initialize total loss with triplet component
            total_loss = self.triplet_weight * triplet_loss
            
            # Add attention regularization component if enabled
            if self.attention_weight > 0:
                # Ensure attention weights have valid values with single batch operations
                pos_attention_valid = torch.nn.functional.softmax(positive_attentions + 1e-10, dim=-1)
                neg_attention_valid = torch.nn.functional.softmax(negative_attentions + 1e-10, dim=-1)
                
                # Compute attention loss using vectorized operations
                pos_attention_loss = self._compute_batch_attention_loss(pos_attention_valid)
                neg_attention_loss = self._compute_batch_attention_loss(neg_attention_valid)
                attention_loss = (pos_attention_loss + neg_attention_loss) / 2.0
                
                # Normalize attention loss to similar scale using sigmoid
                attention_loss_normalized = torch.sigmoid(attention_loss)  # Maps to [0, 1]
                total_loss = total_loss + self.attention_weight * attention_loss_normalized
        else:
            # Return small trainable value if no losses computed
            total_loss = torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # Add small epsilon to ensure non-zero gradient
        total_loss = total_loss + 1e-6
        
        return total_loss
    
    def _compute_batch_attention_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention regularization loss for batched attention weights.
        
        Args:
            attention_weights: Tensor of shape [batch_size, num_rows, num_sentences]
            
        Returns:
            Attention regularization loss
        """
        # Handle empty or invalid attention weights
        if attention_weights.numel() == 0:
            return torch.tensor(1e-6, device=attention_weights.device, dtype=attention_weights.dtype, requires_grad=True)
        
        # Compute entropy for each attention distribution (encourage diversity)
        # Use log_softmax for numerical stability
        log_attention = torch.log_softmax(attention_weights + 1e-10, dim=-1)
        entropy_per_row = -torch.sum(attention_weights * log_attention, dim=-1)  # [batch_size, num_rows]
        
        # Use negative entropy as loss (minimize this to maximize entropy/diversity)
        entropy_loss = -torch.mean(entropy_per_row)
        
        # Replace NaN values with zeros for stability
        entropy_loss = torch.where(torch.isnan(entropy_loss), torch.zeros_like(entropy_loss), entropy_loss)
        
        # Encourage balanced attention by penalizing too concentrated distributions
        target_entropy = torch.log(torch.tensor(attention_weights.shape[-1] / 2.0, device=attention_weights.device, dtype=attention_weights.dtype))
        entropy_penalty = F.mse_loss(torch.mean(entropy_per_row), target_entropy)
        
        return entropy_loss + entropy_penalty

class BidirectionalTripletLoss(nn.Module):
    """
    Bidirectional triplet loss designed for bidirectional cross-attention models.
    
    Key features:
    1. Optimized for pair-level aggregation methods
    2. Supports both attention and pair-wise contrastive losses
    3. Designed for join path discovery tasks
    4. Enhanced regularization for stable training
    5. FIXED: Proper loss weight normalization for stable training
    """
    def __init__(self, 
                 model, 
                 cache=None, 
                 margin: float = 0.3, 
                 scale: float = 10.0,
                 aggregation_method: str = "top_k_pairs",
                 triplet_weight: float = 0.7,      # Main triplet loss (70%)
                 attention_weight: float = 0.2,    # Attention regularization (20%)
                 diversity_weight: float = 0.1,    # Cross-row diversity regularization
                 pair_weight: float = 0.1,         # Pair contrastive loss (10%)
                 pair_margin: float = 0.1,
                 direct_attention_weight: float = 0.1,
                 direct_attention_diversity_weight: float = 1.0,
                 direct_attention_hub_weight: float = 0.5,
                 direct_attention_entropy_weight: float = 0.1,
                 direct_attention_entropy_floor_ratio: float = 0.5,
                 forward_attention_weight: float = 0.0,
                 pair_mil_weight: float = 0.0,
                 pair_mil_positive_margin: float = 0.2,
                 pair_mil_negative_margin: float = 0.05,
                 pair_mil_sparsity_weight: float = 0.0,
                 pair_mil_hub_weight: float = 0.0,
                 # NEW: Attention distillation parameters
                 use_attention_distillation: bool = False,
                 distillation_weight: float = 0.2,
                 teacher_temperature: float = 0.1,
                 student_temperature: float = 0.1,
                 distillation_loss_type: str = "kl_div",
                 teacher_hub_centering: bool = True,
                 # NEW: SIGReg and Sinkhorn regularization
                 sigreg_weight: float = 0.0,
                 sigreg_target_std: float = 1.0,
                 sigreg_num_proj: int = 1024,
                 sigreg_knots: int = 17,
                 sinkhorn_weight: float = 0.0):
        """
        Initialize the bidirectional triplet loss with normalized weights.
        
        Args:
            model: The BidirectionalTableTextModel
            cache: Optional IdBasedEmbeddingCache instance
            margin: Minimum margin between positive and negative scores
            scale: Scale factor for the similarity scores (amplifies differences)
            aggregation_method: Method for aggregating pair scores (top_k_pairs, max_pairs, etc.)
            triplet_weight: Weight for main triplet loss (relative importance)
            attention_weight: Weight for attention regularization loss (relative importance)
            diversity_weight: Weight for cross-row diversity loss (relative importance)
            pair_weight: Weight for pair-wise contrastive loss (relative importance)
            pair_margin: Margin for pair-wise contrastive loss
            direct_attention_weight: Weight for direct sparse-attention anti-collapse loss
            use_attention_distillation: Whether to use attention distillation from frozen encoder
            distillation_weight: Weight for distillation loss (will be added to normalization)
            teacher_temperature: Temperature for frozen encoder's pair similarities
            student_temperature: Temperature for LOKI's pair scores
            distillation_loss_type: Type of distillation loss ("kl_div", "mse", "cosine", "js_div")
            teacher_hub_centering: Whether to suppress universal hub sentences in teacher scores
        """
        super(BidirectionalTripletLoss, self).__init__()
        self.model = model
        self.cache = cache
        self.margin = margin
        self.scale = scale
        self.aggregation_method = aggregation_method
        self.pair_margin = pair_margin
        # Optional switches populated from model (set by training script via attributes)
        self.ranking_loss_type = getattr(model, 'ranking_loss_type', 'softplus')
        self.infonce_tau = getattr(model, 'infonce_tau', 0.7)
        self.pair_topk_mask = getattr(model, 'pair_topk_mask', False)
        self.pair_topk_k = getattr(model, 'pair_topk_k', 0)
        self.direct_attention_weight_raw = direct_attention_weight
        self.direct_attention_diversity_weight = direct_attention_diversity_weight
        self.direct_attention_hub_weight = direct_attention_hub_weight
        self.direct_attention_entropy_weight = direct_attention_entropy_weight
        self.direct_attention_entropy_floor_ratio = direct_attention_entropy_floor_ratio
        self.forward_attention_weight_raw = forward_attention_weight
        self.pair_mil_weight_raw = pair_mil_weight
        self.pair_mil_positive_margin = pair_mil_positive_margin
        self.pair_mil_negative_margin = pair_mil_negative_margin
        self.pair_mil_sparsity_weight = pair_mil_sparsity_weight
        self.pair_mil_hub_weight = pair_mil_hub_weight
        
        # NEW: Attention distillation settings
        self.use_attention_distillation = use_attention_distillation
        self.distillation_weight_raw = distillation_weight
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.distillation_loss_type = distillation_loss_type
        self.teacher_hub_centering = teacher_hub_centering
        
        # NEW: SIGReg and Sinkhorn settings
        self.sigreg_weight_raw = sigreg_weight
        self.sigreg_target_std = sigreg_target_std
        self.epps_pulley_sigreg = EppsPulleySIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)
        self.sinkhorn_weight_raw = sinkhorn_weight
        
        # Initialize distillation loss module if enabled
        if use_attention_distillation:
            self.distillation_loss = AttentionDistillationLoss(
                teacher_temperature=teacher_temperature,
                student_temperature=student_temperature,
                loss_type=distillation_loss_type,
                symmetric=False,
                teacher_hub_centering=teacher_hub_centering
            )
        else:
            self.distillation_loss = None
        
        # FIXED: Normalize weights to sum to 1.0 for interpretable relative importance
        # Include distillation_weight, sigreg_weight, sinkhorn_weight in normalization
        extra_weight = forward_attention_weight + pair_mil_weight + sigreg_weight + sinkhorn_weight
        if use_attention_distillation:
            total_weight = triplet_weight + attention_weight + diversity_weight + pair_weight + direct_attention_weight + distillation_weight + extra_weight
        else:
            total_weight = triplet_weight + attention_weight + diversity_weight + pair_weight + direct_attention_weight + extra_weight
            
        if total_weight <= 0:
            raise ValueError("Sum of loss weights must be positive")
            
        self.triplet_weight = triplet_weight / total_weight
        self.attention_weight = attention_weight / total_weight
        self.diversity_weight = diversity_weight / total_weight
        self.pair_weight = pair_weight / total_weight
        self.direct_attention_weight = direct_attention_weight / total_weight
        self.forward_attention_weight = forward_attention_weight / total_weight
        self.pair_mil_weight = pair_mil_weight / total_weight
        self.sigreg_weight = sigreg_weight / total_weight
        self.sinkhorn_weight = sinkhorn_weight / total_weight
        
        if use_attention_distillation:
            self.distillation_weight = distillation_weight / total_weight
        else:
            self.distillation_weight = 0.0
        
        print(f"Initialized BidirectionalTripletLoss with margin={margin}, scale={scale}")
        print(f"Aggregation: {aggregation_method}, pair_margin: {pair_margin}")
        weight_parts = [f"Triplet: {self.triplet_weight:.3f}", f"Attention: {self.attention_weight:.3f}",
                        f"Diversity: {self.diversity_weight:.3f}", f"DirectAttn: {self.direct_attention_weight:.3f}",
                        f"ForwardAttn: {self.forward_attention_weight:.3f}", f"PairMIL: {self.pair_mil_weight:.3f}",
                        f"Pair: {self.pair_weight:.3f}"]
        if use_attention_distillation:
            weight_parts.append(f"Distillation: {self.distillation_weight:.3f}")
        if sigreg_weight > 0:
            weight_parts.append(f"SIGReg: {self.sigreg_weight:.3f}")
        if sinkhorn_weight > 0:
            weight_parts.append(f"Sinkhorn: {self.sinkhorn_weight:.3f}")
        print(f"Normalized weights - {', '.join(weight_parts)}")
        if use_attention_distillation:
            print(f"Distillation config: teacher_temp={teacher_temperature}, student_temp={student_temperature}, loss_type={distillation_loss_type}, hub_centering={teacher_hub_centering}")
        if sigreg_weight > 0:
            print(f"SIGReg config: target_std={sigreg_target_std}")
        if sinkhorn_weight > 0:
            print(f"Sinkhorn marginal constraint enabled")
        all_weights_sum = (self.triplet_weight + self.attention_weight + self.diversity_weight +
                          self.direct_attention_weight + self.forward_attention_weight +
                          self.pair_mil_weight + self.pair_weight + self.distillation_weight +
                          self.sigreg_weight + self.sinkhorn_weight)
        print(f"Weight sum verification: {all_weights_sum:.6f}")
        
        # Backward compatibility: store old parameter names for existing code
        self.attention_loss_weight = self.attention_weight
        self.diversity_loss_weight = self.diversity_weight
        self.direct_attention_loss_weight = self.direct_attention_weight
        self.forward_attention_loss_weight = self.forward_attention_weight
        self.pair_mil_loss_weight = self.pair_mil_weight
        self.pair_loss_weight = self.pair_weight
    
    def forward(self, triplet_batch):
        """
        Compute bidirectional triplet loss with pair-level optimization.
        
        Args:
            triplet_batch: A batch of triplets with anchor_id, positive_id, and negative_id
                          NOTE: Handles empty additional_positives correctly - will process 
                          only primary_positive if additional_positives is empty list
        """
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        use_header_conditioning = bool(getattr(self.model, 'use_header_conditioning', False))
        use_cell_level_matching = bool(getattr(self.model, 'use_cell_level_matching', False))
        
        # Cache is optional - if None, we'll compute embeddings on-the-fly
        
        # Handle empty triplet batch (can occur when additional_positives is empty and no valid triplets generated)
        if not triplet_batch:
            return torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # Extract unique IDs (all triplets share same anchor in isolated batches)
        anchor_id = triplet_batch[0]['anchor_id']
        positive_ids = [triplet['positive_id'] for triplet in triplet_batch]
        negative_ids = [triplet['negative_id'] for triplet in triplet_batch]
        
        # OPTIMIZATION: Fetch anchor embeddings once from cache or compute on-the-fly
        anchor_embeddings = None
        anchor_schema_embedding = None
        anchor_cell_embeddings = None
        if self.cache is not None:
            anchor_embeddings = self.cache.get_table_embeddings(anchor_id)
            if anchor_embeddings is not None:
                if use_header_conditioning:
                    anchor_schema_embedding = self.cache.get_table_schema_embedding(anchor_id)
                if use_cell_level_matching:
                    anchor_cell_embeddings = self.cache.get_table_cell_embeddings(anchor_id)
            else:
                # Flipped schema: anchor is in context cache
                anchor_embeddings = self.cache.get_context_embeddings(anchor_id)
        
        if anchor_embeddings is None:
            # Compute on-the-fly (either no cache or cache miss)
            anchor_rows = triplet_batch[0].get('anchor_texts', triplet_batch[0].get('anchor_rows', []))
            anchor_embeddings = self.model.encode_sentences(anchor_rows, batch_size=32)
            if use_header_conditioning:
                anchor_schema_text = triplet_batch[0].get('anchor_schema_text')
                anchor_schema_embedding = _encode_schema_texts(self.model, anchor_schema_text, batch_size=32)
            if use_cell_level_matching:
                anchor_cell_texts = triplet_batch[0].get('anchor_cell_texts')
                anchor_cell_embeddings = _encode_cell_text_rows(self.model, anchor_cell_texts, batch_size=32)
        
        # Extract unique positive and negative IDs for cache lookup
        unique_positive_ids = list(set(positive_ids))
        unique_negative_ids = list(set(negative_ids))
        
        # OPTIMIZATION: Batch fetch all positive contexts at once or compute on-the-fly
        positive_embeddings_dict = {}
        positive_schema_embeddings_dict = {}
        positive_cell_embeddings_dict = {}
        for pos_id in unique_positive_ids:
            pos_emb = None
            pos_schema_emb = None
            pos_cell_emb = None
            if self.cache is not None:
                pos_emb = self.cache.get_context_embeddings(pos_id)
                if pos_emb is None:
                    # Flipped schema: positive target is in table cache
                    pos_emb = self.cache.get_table_embeddings(pos_id)
                    if use_header_conditioning and pos_emb is not None:
                        pos_schema_emb = self.cache.get_table_schema_embedding(pos_id)
                    if use_cell_level_matching and pos_emb is not None:
                        pos_cell_emb = self.cache.get_table_cell_embeddings(pos_id)
            
            if pos_emb is None:
                # Compute on-the-fly - find the context from any triplet with this positive_id
                for triplet in triplet_batch:
                    if triplet['positive_id'] == pos_id:
                        pos_context = triplet.get('positive_texts', triplet.get('positive_context', []))
                        pos_emb = self.model.encode_sentences(pos_context, batch_size=32)
                        if use_header_conditioning:
                            pos_schema_text = triplet.get('positive_schema_text')
                            pos_schema_emb = _encode_schema_texts(self.model, pos_schema_text, batch_size=32)
                        if use_cell_level_matching:
                            pos_cell_texts = triplet.get('positive_cell_texts')
                            pos_cell_emb = _encode_cell_text_rows(self.model, pos_cell_texts, batch_size=32)
                        break
            if pos_emb is not None:
                positive_embeddings_dict[pos_id] = pos_emb
                positive_schema_embeddings_dict[pos_id] = pos_schema_emb
                positive_cell_embeddings_dict[pos_id] = pos_cell_emb
        
        # OPTIMIZATION: Batch fetch all negative contexts at once or compute on-the-fly
        negative_embeddings_dict = {}
        negative_schema_embeddings_dict = {}
        negative_cell_embeddings_dict = {}
        for neg_id in unique_negative_ids:
            neg_emb = None
            neg_schema_emb = None
            neg_cell_emb = None
            if self.cache is not None:
                neg_emb = self.cache.get_context_embeddings(neg_id)
                if neg_emb is None:
                    # Flipped schema: negative target is in table cache
                    neg_emb = self.cache.get_table_embeddings(neg_id)
                    if use_header_conditioning and neg_emb is not None:
                        neg_schema_emb = self.cache.get_table_schema_embedding(neg_id)
                    if use_cell_level_matching and neg_emb is not None:
                        neg_cell_emb = self.cache.get_table_cell_embeddings(neg_id)
            
            if neg_emb is None:
                # Compute on-the-fly - find the context from any triplet with this negative_id
                for triplet in triplet_batch:
                    if triplet['negative_id'] == neg_id:
                        neg_context = triplet.get('negative_texts', triplet.get('negative_context', []))
                        neg_emb = self.model.encode_sentences(neg_context, batch_size=32)
                        if use_header_conditioning:
                            neg_schema_text = triplet.get('negative_schema_text')
                            neg_schema_emb = _encode_schema_texts(self.model, neg_schema_text, batch_size=32)
                        if use_cell_level_matching:
                            neg_cell_texts = triplet.get('negative_cell_texts')
                            neg_cell_emb = _encode_cell_text_rows(self.model, neg_cell_texts, batch_size=32)
                        break
            if neg_emb is not None:
                negative_embeddings_dict[neg_id] = neg_emb
                negative_schema_embeddings_dict[neg_id] = neg_schema_emb
                negative_cell_embeddings_dict[neg_id] = neg_cell_emb
        
        if not positive_embeddings_dict or not negative_embeddings_dict:
            return torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # OPTIMIZATION: Efficient padding with cached zero tensors
        def pad_embeddings_efficiently(embeddings_dict):
            if not embeddings_dict:
                return {}, 0, 0
            
            # Find max sequence length for padding
            max_seq_len = max(emb.shape[0] for emb in embeddings_dict.values())
            embed_dim = list(embeddings_dict.values())[0].shape[1]
            
            # Pre-allocate padding tensor for efficiency
            padding_cache = {}
            padded_dict = {}
            
            for id_key, emb in embeddings_dict.items():
                seq_len = emb.shape[0]
                if seq_len < max_seq_len:
                    pad_size = max_seq_len - seq_len
                    # Use cached padding of this size if available
                    if pad_size not in padding_cache:
                        padding_cache[pad_size] = torch.zeros(pad_size, embed_dim, device=emb.device, dtype=emb.dtype)
                    padding = padding_cache[pad_size]
                    padded_emb = torch.cat([emb, padding], dim=0)
                else:
                    padded_emb = emb
                padded_dict[id_key] = padded_emb
            
            return padded_dict, max_seq_len, embed_dim
        
        # Apply efficient padding
        padded_positives, pos_max_len, embed_dim = pad_embeddings_efficiently(positive_embeddings_dict)
        padded_negatives, neg_max_len, _ = pad_embeddings_efficiently(negative_embeddings_dict)
        
        if not padded_positives or not padded_negatives:
            return torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # OPTIMIZATION: Stack embeddings once for efficient GPU processing
        # Get filtered IDs (those successfully retrieved from cache and padded)
        filtered_pos_ids = [pid for pid in unique_positive_ids if pid in padded_positives]
        filtered_neg_ids = [nid for nid in unique_negative_ids if nid in padded_negatives]
        
        if not filtered_pos_ids or not filtered_neg_ids:
            return torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # Stack embeddings with a single operation
        positive_embeddings_list = [padded_positives[pos_id] for pos_id in filtered_pos_ids]
        negative_embeddings_list = [padded_negatives[neg_id] for neg_id in filtered_neg_ids]
        
        # Single tensor operations for stacking contexts
        batched_positives = torch.stack(positive_embeddings_list)  # [num_unique_pos, max_pos_len, embed_dim]
        batched_negatives = torch.stack(negative_embeddings_list)  # [num_unique_neg, max_neg_len, embed_dim]

        positive_schema_batch = _build_padded_schema_batch(
            filtered_pos_ids,
            positive_schema_embeddings_dict,
            device,
            model_dtype,
            batched_positives.shape[-1],
        ) if use_header_conditioning else None
        negative_schema_batch = _build_padded_schema_batch(
            filtered_neg_ids,
            negative_schema_embeddings_dict,
            device,
            model_dtype,
            batched_negatives.shape[-1],
        ) if use_header_conditioning else None
        positive_cell_batch = _build_padded_cell_batch(
            filtered_pos_ids,
            positive_cell_embeddings_dict,
            device,
            model_dtype,
            batched_positives.shape[-1],
            target_row_count=batched_positives.shape[1],
        ) if use_cell_level_matching else None
        negative_cell_batch = _build_padded_cell_batch(
            filtered_neg_ids,
            negative_cell_embeddings_dict,
            device,
            model_dtype,
            batched_negatives.shape[-1],
            target_row_count=batched_negatives.shape[1],
        ) if use_cell_level_matching else None
        
        # Create efficient mapping from ID to tensor index
        pos_id_to_idx = {filtered_pos_ids[i]: i for i in range(len(filtered_pos_ids))}
        neg_id_to_idx = {filtered_neg_ids[i]: i for i in range(len(filtered_neg_ids))}
        
        # OPTIMIZATION: Expand anchor in a single operation
        # This preserves relationship between anchor rows and context sentences
        anchor_for_positives = anchor_embeddings.unsqueeze(0).expand(len(filtered_pos_ids), -1, -1)
        anchor_for_negatives = anchor_embeddings.unsqueeze(0).expand(len(filtered_neg_ids), -1, -1)
        anchor_schema_for_positives = None
        anchor_schema_for_negatives = None
        anchor_cell_for_positives = None
        anchor_cell_for_negatives = None
        if use_header_conditioning and anchor_schema_embedding is not None:
            if anchor_schema_embedding.dim() > 2 and anchor_schema_embedding.size(0) == 1:
                anchor_schema_embedding = anchor_schema_embedding.squeeze(0)
            if anchor_schema_embedding.dim() == 1:
                anchor_schema_embedding = anchor_schema_embedding.unsqueeze(0)
            anchor_schema_embedding = anchor_schema_embedding.to(device=device, dtype=model_dtype)
            anchor_schema_for_positives = anchor_schema_embedding.unsqueeze(0).expand(len(filtered_pos_ids), -1, -1)
            anchor_schema_for_negatives = anchor_schema_embedding.unsqueeze(0).expand(len(filtered_neg_ids), -1, -1)
        if use_cell_level_matching and anchor_cell_embeddings is not None:
            if anchor_cell_embeddings.dim() > 3 and anchor_cell_embeddings.size(0) == 1:
                anchor_cell_embeddings = anchor_cell_embeddings.squeeze(0)
            if anchor_cell_embeddings.dim() == 2:
                anchor_cell_embeddings = anchor_cell_embeddings.unsqueeze(1)
            anchor_cell_embeddings = anchor_cell_embeddings.to(device=device, dtype=model_dtype)
            anchor_cell_for_positives = anchor_cell_embeddings.unsqueeze(0).expand(len(filtered_pos_ids), -1, -1, -1)
            anchor_cell_for_negatives = anchor_cell_embeddings.unsqueeze(0).expand(len(filtered_neg_ids), -1, -1, -1)
        
        return_attention_weights = (
            self.direct_attention_weight > 0
            or self.forward_attention_weight > 0
            or self.sinkhorn_weight > 0
        )
        return_contextualized = self.sigreg_weight > 0

        # OPTIMIZATION: Single forward pass for all anchors and contexts
        # Bidirectional model returns similarity scores and pair-level information
        positive_output = self.model(
            anchor_for_positives, batched_positives, 
            aggregation_method=self.aggregation_method,
            rows_schema_embeddings=anchor_schema_for_positives,
            sentences_schema_embeddings=positive_schema_batch,
            rows_cell_embeddings=anchor_cell_for_positives,
            sentences_cell_embeddings=positive_cell_batch,
            return_attention_weights=return_attention_weights,
            return_contextualized=return_contextualized,
        )  # [num_unique_pos], [num_unique_pos, num_rows, max_pos_len]
        if return_attention_weights and return_contextualized:
            positive_scores, positive_pair_scores, positive_forward_attn, positive_reverse_attn, positive_ctx_rows, positive_ctx_sentences = positive_output
        elif return_attention_weights:
            positive_scores, positive_pair_scores, positive_forward_attn, positive_reverse_attn = positive_output
            positive_ctx_rows = positive_ctx_sentences = None
        elif return_contextualized:
            positive_scores, positive_pair_scores, positive_ctx_rows, positive_ctx_sentences = positive_output
            positive_forward_attn = positive_reverse_attn = None
        else:
            positive_scores, positive_pair_scores = positive_output
            positive_forward_attn = positive_reverse_attn = None
            positive_ctx_rows = positive_ctx_sentences = None
        
        negative_output = self.model(
            anchor_for_negatives, batched_negatives, 
            aggregation_method=self.aggregation_method,
            rows_schema_embeddings=anchor_schema_for_negatives,
            sentences_schema_embeddings=negative_schema_batch,
            rows_cell_embeddings=anchor_cell_for_negatives,
            sentences_cell_embeddings=negative_cell_batch,
            return_attention_weights=return_attention_weights,
            return_contextualized=return_contextualized,
        )  # [num_unique_neg], [num_unique_neg, num_rows, max_neg_len]
        if return_attention_weights and return_contextualized:
            negative_scores, negative_pair_scores, negative_forward_attn, negative_reverse_attn, negative_ctx_rows, negative_ctx_sentences = negative_output
        elif return_attention_weights:
            negative_scores, negative_pair_scores, negative_forward_attn, negative_reverse_attn = negative_output
            negative_ctx_rows = negative_ctx_sentences = None
        elif return_contextualized:
            negative_scores, negative_pair_scores, negative_ctx_rows, negative_ctx_sentences = negative_output
            negative_forward_attn = negative_reverse_attn = None
        else:
            negative_scores, negative_pair_scores = negative_output
            negative_forward_attn = negative_reverse_attn = None
            negative_ctx_rows = negative_ctx_sentences = None
        positive_training_pairs = positive_pair_scores
        negative_training_pairs = negative_pair_scores
        
        # OPTIMIZATION: Create dictionaries for O(1) ID lookup
        positive_scores_dict = {filtered_pos_ids[i]: positive_scores[i] for i in range(len(filtered_pos_ids))}
        negative_scores_dict = {filtered_neg_ids[i]: negative_scores[i] for i in range(len(filtered_neg_ids))}
        positive_pairs_dict = {filtered_pos_ids[i]: positive_training_pairs[i] for i in range(len(filtered_pos_ids))}
        negative_pairs_dict = {filtered_neg_ids[i]: negative_training_pairs[i] for i in range(len(filtered_neg_ids))}
        
        # OPTIMIZATION: Pre-allocate list for losses
        batch_size = len(triplet_batch)
        batch_losses = []
        batch_losses.reserve(batch_size) if hasattr(batch_losses, 'reserve') else None
        pair_losses = []
        pair_losses.reserve(batch_size) if hasattr(pair_losses, 'reserve') else None
        
        # Process all triplets in the batch
        for triplet in triplet_batch:
            pos_id = triplet['positive_id']
            neg_id = triplet['negative_id']
            
            # Skip triplets with missing embeddings
            if pos_id not in positive_scores_dict or neg_id not in negative_scores_dict:
                continue
            
            pos_score = positive_scores_dict[pos_id]
            neg_score = negative_scores_dict[neg_id]
            
            # Ranking loss switch (softplus default)
            if self.ranking_loss_type == 'infonce':
                tau = max(self.infonce_tau, 1e-6)
                logits = torch.stack([pos_score, neg_score], dim=0) / tau
                labels = torch.tensor(0, device=logits.device)
                triplet_loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
            else:
                diff = neg_score - pos_score + self.margin
                triplet_loss = F.softplus(diff * self.scale)
            batch_losses.append(triplet_loss)
            
            # Add pair-wise contrastive loss if enabled
            if self.pair_weight > 0 and pos_id in positive_pairs_dict and neg_id in negative_pairs_dict:
                pos_pairs = positive_pairs_dict[pos_id]
                neg_pairs = negative_pairs_dict[neg_id]
                if self.pair_topk_mask:
                    # FIX: Create separate masks for pos_pairs and neg_pairs since they can have different shapes
                    # (positive and negative contexts may have different numbers of sentences)
                    
                    # Mask positive pairs
                    pos_flat = pos_pairs.reshape(-1)
                    pos_total = int(pos_flat.numel())
                    if pos_total > 0:
                        requested_k = int(self.pair_topk_k) if int(self.pair_topk_k) > 0 else int(getattr(self.model, 'top_k', 3))
                        pos_k = max(1, min(requested_k, pos_total))
                        _, pos_idx = torch.topk(pos_flat, k=pos_k)
                        pos_mask = torch.zeros_like(pos_flat, dtype=torch.bool)
                        pos_mask[pos_idx] = True
                        pos_mask = pos_mask.view_as(pos_pairs)
                        pos_pairs = torch.where(pos_mask, pos_pairs, torch.zeros_like(pos_pairs))
                    
                    # Mask negative pairs
                    neg_flat = neg_pairs.reshape(-1)
                    neg_total = int(neg_flat.numel())
                    if neg_total > 0:
                        requested_k = int(self.pair_topk_k) if int(self.pair_topk_k) > 0 else int(getattr(self.model, 'top_k', 3))
                        neg_k = max(1, min(requested_k, neg_total))
                        _, neg_idx = torch.topk(neg_flat, k=neg_k)
                        neg_mask = torch.zeros_like(neg_flat, dtype=torch.bool)
                        neg_mask[neg_idx] = True
                        neg_mask = neg_mask.view_as(neg_pairs)
                        neg_pairs = torch.where(neg_mask, neg_pairs, torch.zeros_like(neg_pairs))
                
                pair_loss = self._compute_pair_contrastive_loss(pos_pairs, neg_pairs, self.pair_margin)
                pair_losses.append(pair_loss)
        
        # FIXED: Properly combine losses with normalization and weighting
        if batch_losses:
            # Compute individual loss components
            triplet_loss = torch.mean(torch.stack(batch_losses))
            
            # Initialize total loss with triplet component
            total_loss = self.triplet_weight * triplet_loss
            
            # Add pair-wise contrastive loss component if enabled
            if pair_losses and self.pair_weight > 0:
                pair_loss = torch.mean(torch.stack(pair_losses))
                # Use linear scaling (no tanh squashing)
                total_loss = total_loss + self.pair_weight * pair_loss
            
            # Add attention regularization component if enabled
            if self.attention_weight > 0:
                # Get attention weights from model's bidirectional attention
                attention_loss = self._attention_entropy_loss(positive_training_pairs)
                # Linear weighting (no sigmoid squashing)
                total_loss = total_loss + self.attention_weight * attention_loss

            # Add cross-row diversity regularization if enabled
            if self.diversity_weight > 0:
                diversity_loss = self._cross_row_diversity_loss(positive_training_pairs)
                total_loss = total_loss + self.diversity_weight * diversity_loss

            # Direct anti-collapse regularization on actual sparse attention weights.
            if self.direct_attention_weight > 0 and positive_forward_attn is not None:
                anchor_positive_mask = self._embedding_valid_mask(anchor_for_positives)
                positive_sentence_mask = self._embedding_valid_mask(batched_positives)
                direct_attention_loss = self._bidirectional_attention_anti_collapse_loss(
                    positive_forward_attn,
                    positive_reverse_attn,
                    row_mask=anchor_positive_mask,
                    sentence_mask=positive_sentence_mask,
                )

                if negative_forward_attn is not None:
                    anchor_negative_mask = self._embedding_valid_mask(anchor_for_negatives)
                    negative_sentence_mask = self._embedding_valid_mask(batched_negatives)
                    negative_direct_loss = self._bidirectional_attention_anti_collapse_loss(
                        negative_forward_attn,
                        negative_reverse_attn,
                        row_mask=anchor_negative_mask,
                        sentence_mask=negative_sentence_mask,
                    )
                    direct_attention_loss = (direct_attention_loss + negative_direct_loss) / 2.0

                total_loss = total_loss + self.direct_attention_weight * direct_attention_loss

            if self.forward_attention_weight > 0 and positive_forward_attn is not None:
                anchor_positive_mask = self._embedding_valid_mask(anchor_for_positives)
                positive_sentence_mask = self._embedding_valid_mask(batched_positives)
                forward_attention_loss = self._direct_attention_anti_collapse_loss(
                    positive_forward_attn,
                    query_mask=anchor_positive_mask,
                    key_mask=positive_sentence_mask,
                )
                if negative_forward_attn is not None:
                    anchor_negative_mask = self._embedding_valid_mask(anchor_for_negatives)
                    negative_sentence_mask = self._embedding_valid_mask(batched_negatives)
                    negative_forward_loss = self._direct_attention_anti_collapse_loss(
                        negative_forward_attn,
                        query_mask=anchor_negative_mask,
                        key_mask=negative_sentence_mask,
                    )
                    forward_attention_loss = (forward_attention_loss + negative_forward_loss) / 2.0
                total_loss = total_loss + self.forward_attention_weight * forward_attention_loss

            if self.pair_mil_weight > 0:
                anchor_positive_mask = self._embedding_valid_mask(anchor_for_positives)
                positive_sentence_mask = self._embedding_valid_mask(batched_positives)
                anchor_negative_mask = self._embedding_valid_mask(anchor_for_negatives)
                negative_sentence_mask = self._embedding_valid_mask(batched_negatives)
                pair_mil_loss = self._pair_mil_loss(
                    positive_pair_scores,
                    negative_pair_scores,
                    positive_row_mask=anchor_positive_mask,
                    positive_sentence_mask=positive_sentence_mask,
                    negative_row_mask=anchor_negative_mask,
                    negative_sentence_mask=negative_sentence_mask,
                )
                total_loss = total_loss + self.pair_mil_weight * pair_mil_loss
            
            # NEW: Add attention distillation loss if enabled
            if self.use_attention_distillation and self.distillation_loss is not None and self.distillation_weight > 0:
                # Compute distillation loss for positive contexts
                # positive_pair_scores: [num_unique_pos, num_rows, max_pos_len]
                # anchor_for_positives: [num_unique_pos, num_rows, embed_dim] - raw anchor embeddings
                # batched_positives: [num_unique_pos, max_pos_len, embed_dim] - raw positive embeddings
                distill_loss = self.distillation_loss(
                    student_pair_scores=positive_pair_scores,
                    row_embeddings=anchor_for_positives,
                    sentence_embeddings=batched_positives
                )
                
                # Optionally also distill on negative contexts (helps prevent collapse)
                if negative_pair_scores is not None:
                    neg_distill_loss = self.distillation_loss(
                        student_pair_scores=negative_pair_scores,
                        row_embeddings=anchor_for_negatives,
                        sentence_embeddings=batched_negatives
                    )
                    distill_loss = (distill_loss + neg_distill_loss) / 2.0
                
                total_loss = total_loss + self.distillation_weight * distill_loss

            # NEW: SIGReg loss on contextualized embeddings
            if self.sigreg_weight > 0 and positive_ctx_rows is not None:
                # Apply SIGReg to positive contextualized rows AND sentences
                pos_row_sigreg = self.epps_pulley_sigreg(positive_ctx_rows)
                pos_sent_sigreg = self.epps_pulley_sigreg(positive_ctx_sentences)
                sigreg_total = (pos_row_sigreg + pos_sent_sigreg) / 2.0

                # Optionally also regularize negative context representations
                if negative_ctx_rows is not None:
                    neg_row_sigreg = self.epps_pulley_sigreg(negative_ctx_rows)
                    neg_sent_sigreg = self.epps_pulley_sigreg(negative_ctx_sentences)
                    neg_sigreg = (neg_row_sigreg + neg_sent_sigreg) / 2.0
                    sigreg_total = (sigreg_total + neg_sigreg) / 2.0

                total_loss = total_loss + self.sigreg_weight * sigreg_total

            # NEW: Sinkhorn marginal constraint on attention matrices
            if self.sinkhorn_weight > 0 and positive_forward_attn is not None:
                anchor_positive_mask = self._embedding_valid_mask(anchor_for_positives)
                positive_sentence_mask = self._embedding_valid_mask(batched_positives)

                # Forward attention: rows → sentences
                pos_fwd_sinkhorn = sinkhorn_reg_loss(
                    positive_forward_attn,
                    query_mask=anchor_positive_mask,
                    key_mask=positive_sentence_mask,
                )
                # Reverse attention: sentences → rows
                pos_rev_sinkhorn = sinkhorn_reg_loss(
                    positive_reverse_attn,
                    query_mask=positive_sentence_mask,
                    key_mask=anchor_positive_mask,
                )
                sinkhorn_total = (pos_fwd_sinkhorn + pos_rev_sinkhorn) / 2.0

                # Also constrain negative attention if available
                if negative_forward_attn is not None:
                    anchor_negative_mask = self._embedding_valid_mask(anchor_for_negatives)
                    negative_sentence_mask = self._embedding_valid_mask(batched_negatives)
                    neg_fwd_sinkhorn = sinkhorn_reg_loss(
                        negative_forward_attn,
                        query_mask=anchor_negative_mask,
                        key_mask=negative_sentence_mask,
                    )
                    neg_rev_sinkhorn = sinkhorn_reg_loss(
                        negative_reverse_attn,
                        query_mask=negative_sentence_mask,
                        key_mask=anchor_negative_mask,
                    )
                    neg_sinkhorn = (neg_fwd_sinkhorn + neg_rev_sinkhorn) / 2.0
                    sinkhorn_total = (sinkhorn_total + neg_sinkhorn) / 2.0

                total_loss = total_loss + self.sinkhorn_weight * sinkhorn_total
        else:
            # Return small trainable value if no losses computed
            total_loss = torch.tensor(1e-6, device=device, dtype=model_dtype, requires_grad=True)
        
        # Add small epsilon to ensure non-zero gradient
        total_loss = total_loss + 1e-6
        
        return total_loss

    def _embedding_valid_mask(self, embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Detect non-padding embeddings introduced by batch padding."""
        if embeddings.numel() == 0:
            return torch.zeros(embeddings.shape[:-1], device=embeddings.device, dtype=torch.bool)
        return embeddings.detach().norm(dim=-1) > eps

    def _bidirectional_attention_anti_collapse_loss(
        self,
        forward_attention: torch.Tensor,
        reverse_attention: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
        sentence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply anti-collapse regularization to both row->sentence and sentence->row attention."""
        forward_loss = self._direct_attention_anti_collapse_loss(
            forward_attention,
            query_mask=row_mask,
            key_mask=sentence_mask,
        )
        reverse_loss = self._direct_attention_anti_collapse_loss(
            reverse_attention,
            query_mask=sentence_mask,
            key_mask=row_mask,
        )
        return (forward_loss + reverse_loss) / 2.0

    def _direct_attention_anti_collapse_loss(
        self,
        attention_weights: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Penalize identical query attention patterns and universal hub keys.

        attention_weights: [batch_size, num_queries, num_keys]
        """
        if attention_weights.numel() == 0:
            return torch.tensor(0.0, device=attention_weights.device, dtype=attention_weights.dtype)

        if attention_weights.dim() == 2:
            attention_weights = attention_weights.unsqueeze(0)
        if query_mask is not None and query_mask.dim() == 1:
            query_mask = query_mask.unsqueeze(0)
        if key_mask is not None and key_mask.dim() == 1:
            key_mask = key_mask.unsqueeze(0)

        losses = []
        eps = 1e-8
        for batch_idx in range(attention_weights.shape[0]):
            attn = attention_weights[batch_idx]
            if query_mask is not None:
                valid_queries = query_mask[batch_idx].to(device=attn.device, dtype=torch.bool)
                attn = attn[valid_queries]
            if key_mask is not None:
                valid_keys = key_mask[batch_idx].to(device=attn.device, dtype=torch.bool)
                attn = attn[:, valid_keys]

            num_queries, num_keys = attn.shape[-2], attn.shape[-1]
            if num_queries <= 1 or num_keys <= 1:
                continue

            attn = torch.clamp(attn, min=0.0)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(eps)

            # Minimize off-diagonal cosine similarity between query distributions.
            normalized = F.normalize(attn, p=2, dim=-1, eps=1e-8)
            pairwise = torch.matmul(normalized, normalized.transpose(-2, -1))
            offdiag = pairwise[~torch.eye(num_queries, device=attn.device, dtype=torch.bool)]
            diversity_loss = offdiag.mean() if offdiag.numel() > 0 else torch.zeros((), device=attn.device, dtype=attn.dtype)

            # Penalize concentrated key mass: one sentence/row should not serve every query.
            key_mass = attn.mean(dim=-2)
            hub_baseline = torch.tensor(1.0 / num_keys, device=attn.device, dtype=attn.dtype)
            hub_loss = torch.clamp(key_mass.pow(2).sum() - hub_baseline, min=0.0)

            # Mild entropy floor for sparse attention; top-k should be selective, not one-hot everywhere.
            active_top_k = max(1, min(int(getattr(self.model, "top_k", 5)), num_keys))
            entropy_floor = torch.log(torch.tensor(float(active_top_k), device=attn.device, dtype=attn.dtype))
            entropy_floor = entropy_floor * float(self.direct_attention_entropy_floor_ratio)
            entropy = -(attn * torch.log(attn.clamp_min(1e-10))).sum(dim=-1)
            entropy_loss = F.relu(entropy_floor - entropy).mean()

            loss = (
                float(self.direct_attention_diversity_weight) * diversity_loss
                + float(self.direct_attention_hub_weight) * hub_loss
                + float(self.direct_attention_entropy_weight) * entropy_loss
            )
            losses.append(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))

        if not losses:
            return torch.tensor(0.0, device=attention_weights.device, dtype=attention_weights.dtype)
        return torch.stack(losses).mean()

    def _pair_mil_loss(
        self,
        positive_pair_scores: torch.Tensor,
        negative_pair_scores: torch.Tensor,
        positive_row_mask: Optional[torch.Tensor] = None,
        positive_sentence_mask: Optional[torch.Tensor] = None,
        negative_row_mask: Optional[torch.Tensor] = None,
        negative_sentence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weak MIL-style pressure on raw pair scores from global pairs."""
        losses = []

        def masked_values(
            pair_scores: torch.Tensor,
            row_mask: Optional[torch.Tensor],
            sentence_mask: Optional[torch.Tensor],
            batch_idx: int,
        ) -> torch.Tensor:
            matrix = pair_scores[batch_idx]
            if row_mask is not None:
                matrix = matrix[row_mask[batch_idx].to(device=matrix.device, dtype=torch.bool)]
            if sentence_mask is not None:
                matrix = matrix[:, sentence_mask[batch_idx].to(device=matrix.device, dtype=torch.bool)]
            return matrix

        batch_count = min(positive_pair_scores.shape[0], negative_pair_scores.shape[0])
        for batch_idx in range(batch_count):
            pos = masked_values(positive_pair_scores, positive_row_mask, positive_sentence_mask, batch_idx)
            neg = masked_values(negative_pair_scores, negative_row_mask, negative_sentence_mask, batch_idx)
            if pos.numel() == 0 or neg.numel() == 0:
                continue

            pos_flat = torch.nan_to_num(pos.reshape(-1).float(), nan=0.0, posinf=1.0, neginf=0.0)
            neg_flat = torch.nan_to_num(neg.reshape(-1).float(), nan=0.0, posinf=1.0, neginf=0.0)
            pos_k = max(1, min(int(getattr(self.model, "top_k", 5)), int(pos_flat.numel())))
            neg_k = max(1, min(int(getattr(self.model, "top_k", 5)), int(neg_flat.numel())))
            pos_top = torch.topk(pos_flat, k=pos_k).values.mean()
            neg_top = torch.topk(neg_flat, k=neg_k).values.mean()

            positive_bag_loss = F.relu(float(self.pair_mil_positive_margin) - pos_top)
            negative_bag_loss = F.relu(neg_top - float(self.pair_mil_negative_margin))
            sparsity_loss = 0.5 * (pos_flat.clamp_min(0.0).mean() + neg_flat.clamp_min(0.0).mean())

            if pos.dim() == 2 and pos.shape[0] > 1 and pos.shape[1] > 1:
                key_mass = pos.float().clamp_min(0.0).mean(dim=0)
                hub_baseline = torch.tensor(1.0 / pos.shape[1], device=pos.device, dtype=key_mass.dtype)
                hub_loss = torch.clamp(key_mass.pow(2).sum() - hub_baseline, min=0.0)
            else:
                hub_loss = torch.zeros((), device=pos.device, dtype=pos_flat.dtype)

            losses.append(
                positive_bag_loss
                + negative_bag_loss
                + float(self.pair_mil_sparsity_weight) * sparsity_loss
                + float(self.pair_mil_hub_weight) * hub_loss
            )

        if not losses:
            return torch.tensor(0.0, device=positive_pair_scores.device, dtype=positive_pair_scores.dtype)
        return torch.stack([loss.to(device=positive_pair_scores.device, dtype=positive_pair_scores.dtype) for loss in losses]).mean()

    def _cross_row_diversity_loss(self, pair_scores: torch.Tensor) -> torch.Tensor:
        """
        Encourage different rows to assign different sentence distributions.

        Args:
            pair_scores: Pair score matrix of shape [batch_size, num_rows, num_sentences]

        Returns:
            Negative cross-row variance. Minimizing this maximizes per-sentence
            variance across rows and penalizes identical row distributions.
        """
        if pair_scores.numel() == 0:
            return torch.tensor(0.0, device=pair_scores.device, dtype=pair_scores.dtype)

        if pair_scores.dim() == 2:
            pair_scores = pair_scores.unsqueeze(0)

        if pair_scores.shape[-2] <= 1:
            return torch.tensor(0.0, device=pair_scores.device, dtype=pair_scores.dtype)

        row_distributions = F.softmax(pair_scores, dim=-1)
        cross_row_variance = row_distributions.var(dim=-2, unbiased=False)
        diversity_loss = -cross_row_variance.mean()
        return torch.nan_to_num(diversity_loss, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _attention_entropy_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention regularization loss for bidirectional attention weights.
        
        Args:
            attention_weights: Pair score matrix of shape [batch_size, num_rows, num_sentences]
            
        Returns:
            Attention regularization loss
        """
        # Handle empty or invalid attention weights
        if attention_weights.numel() == 0:
            return torch.tensor(1e-6, device=attention_weights.device, dtype=attention_weights.dtype, requires_grad=True)
        
        # Convert pair scores to proper attention distributions
        row_attention = torch.softmax(attention_weights, dim=-1)
        col_attention = torch.softmax(attention_weights, dim=-2)
        
        # Compute Shannon entropy H(p) = -sum p * log(p)
        log_row_attention = torch.log(row_attention + 1e-10)
        log_col_attention = torch.log(col_attention + 1e-10)
        
        row_entropy = -torch.sum(row_attention * log_row_attention, dim=-1)  # [batch_size, num_rows]
        col_entropy = -torch.sum(col_attention * log_col_attention, dim=-2)  # [batch_size, num_sentences]
        
        # Use negative entropy as loss (minimize this to maximize entropy/diversity)
        entropy_loss = -(torch.mean(row_entropy) + torch.mean(col_entropy)) / 2.0
        
        # Replace NaN values with zeros for stability
        entropy_loss = torch.where(torch.isnan(entropy_loss), torch.zeros_like(entropy_loss), entropy_loss)
        
        # Encourage balanced attention by penalizing too concentrated distributions
        target_entropy = torch.log(torch.tensor(attention_weights.shape[-1] / 2.0, device=attention_weights.device, dtype=attention_weights.dtype))
        entropy_penalty = F.mse_loss(torch.mean(row_entropy), target_entropy)
        
        return entropy_loss + entropy_penalty
    
    def _compute_pair_contrastive_loss(self,
                                    positive_pairs: torch.Tensor,
                                    negative_pairs: torch.Tensor,
                                    pair_margin: float = 0.1) -> torch.Tensor:
        """
        Compute pair-wise contrastive loss to encourage better pair-level discriminability.
        
        Args:
            positive_pairs: Pair scores for positive context [num_rows, num_sentences]
            negative_pairs: Pair scores for negative context [num_rows, num_sentences]
            pair_margin: Margin for pair-wise contrastive loss
            
        Returns:
            Pair-wise contrastive loss
        """
        # Handle empty or mismatched tensor sizes
        if positive_pairs.numel() == 0 or negative_pairs.numel() == 0:
            return torch.tensor(0.0, device=positive_pairs.device, dtype=positive_pairs.dtype, requires_grad=True)
        
        # Ensure tensors have same shape for comparison
        min_rows = min(positive_pairs.shape[0], negative_pairs.shape[0])
        min_cols = min(positive_pairs.shape[1], negative_pairs.shape[1])
        
        pos_pairs_safe = torch.where(torch.isnan(positive_pairs), torch.zeros_like(positive_pairs), positive_pairs)
        neg_pairs_safe = torch.where(torch.isnan(negative_pairs), torch.zeros_like(negative_pairs), negative_pairs)
        
        # Truncate to same size
        pos_truncated = pos_pairs_safe[:min_rows, :min_cols]
        neg_truncated = neg_pairs_safe[:min_rows, :min_cols]
        
        # Compute contrastive loss: encourage positive pairs to be higher than negative pairs
        # Use margin-based contrastive loss with configurable margin
        pair_diff = pos_truncated - neg_truncated
        pair_loss = F.relu(pair_margin - pair_diff)
        
        # Average over all pairs
        if pair_loss.numel() > 0:
            return torch.mean(pair_loss)
        else:
            return torch.tensor(0.0, device=positive_pairs.device, dtype=positive_pairs.dtype, requires_grad=True) 


class EncoderOnlyTripletLoss(nn.Module):
    """
    Triplet loss that trains ONLY the sentence encoder (no cross-attention).
    Computes similarities by averaging row×sentence cosine similarities.
    Designed for Stage 0 fine-tuning comparison.
    
    OPTIMIZED FOR PEFT/LoRA:
    - Single batched forward pass for ALL texts (rows + positives + negatives)
    - Text deduplication to avoid redundant encoding
    - Full gradient flow preserved through LoRA adapters
    - 3-5x faster than separate encoding calls
    """
    def __init__(self,
                 sentence_encoder: nn.Module,
                 device: torch.device,
                 margin: float = 0.3,
                 scale: float = 10.0,
                 ranking_loss_type: str = 'softplus',
                 infonce_tau: float = 0.7,
                 use_hard_negative_mining: bool = False,
                 hard_negative_topk: int = 0,
                 max_batch_size: int = 64):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.device = device
        self.margin = margin
        self.scale = scale
        self.ranking_loss_type = ranking_loss_type
        self.infonce_tau = infonce_tau
        self.use_hard_negative_mining = use_hard_negative_mining
        self.hard_negative_topk = hard_negative_topk
        self.max_batch_size = max_batch_size  # Max texts per forward pass to avoid OOM
        
        print(f"Initialized EncoderOnlyTripletLoss with margin={margin}, scale={scale}, ranking={ranking_loss_type}")
        print(f"   ⚡ OPTIMIZED: Using batched encoding (max_batch_size={max_batch_size})")
        
        # Cache a tiny anchor param to cheaply reattach graph if needed
        try:
            self._anchor_param = next(p for p in self.sentence_encoder.parameters() if p.requires_grad)
        except StopIteration:
            self._anchor_param = None
        
        # Statistics tracking
        self._encoding_stats = {'total_texts': 0, 'unique_texts': 0, 'forward_passes': 0}

    def _encode_batch_with_grad(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts in a SINGLE forward pass with gradient tracking.
        
        This is the core optimization - instead of encoding each context separately,
        we batch all unique texts together for maximum GPU utilization.
        """
        if not texts:
            return torch.empty(0, device=self.device)
        
        self.sentence_encoder.train()
        with torch.set_grad_enabled(True):
            # Tokenize all texts at once
            features = self.sentence_encoder.tokenize(texts)
            features = {k: v.to(self.device) for k, v in features.items()}
            
            # Single forward pass through encoder (including LoRA layers)
            outputs = self.sentence_encoder(features)
        
        # Extract embeddings from output
        if isinstance(outputs, dict) and 'sentence_embedding' in outputs:
            embeddings = outputs['sentence_embedding']
        elif hasattr(outputs, 'sentence_embedding'):
            embeddings = outputs.sentence_embedding
        else:
            if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict) and 'sentence_embedding' in outputs[0]:
                embeddings = outputs[0]['sentence_embedding']
            else:
                raise RuntimeError("Unexpected output from SentenceTransformer forward; cannot extract sentence embeddings")
        
        # Ensure gradient flow (important for LoRA)
        if not embeddings.requires_grad and self._anchor_param is not None:
            embeddings = embeddings + 0.0 * self._anchor_param.view(-1)[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        self._encoding_stats['forward_passes'] += 1
        return embeddings

    def _chunked_encode_with_grad(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts with chunking to avoid OOM on very large batches.
        Still much more efficient than encoding one-by-one.
        """
        if len(texts) <= self.max_batch_size:
            return self._encode_batch_with_grad(texts)
        
        # Chunk large batches
        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            chunk = texts[i:i + self.max_batch_size]
            chunk_embeddings = self._encode_batch_with_grad(chunk)
            all_embeddings.append(chunk_embeddings)
        
        return torch.cat(all_embeddings, dim=0)

    def forward(self, triplet_batch: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute triplet loss with OPTIMIZED batched encoding.
        
        Key optimization: ALL texts are encoded in a SINGLE forward pass:
        1. Collect all unique texts (rows + positive sentences + negative sentences)
        2. Encode them all at once in one batched forward pass
        3. Use index mapping to retrieve embeddings for each context
        
        This reduces forward passes from O(num_contexts) to O(1), giving 3-5x speedup.
        """
        if not triplet_batch:
            return torch.tensor(1e-6, device=self.device, dtype=torch.bfloat16, requires_grad=True)

        # Assume isolated batches by default (same anchor). Use first anchor rows.
        anchor_rows = triplet_batch[0].get('anchor_texts', triplet_batch[0].get('anchor_rows', []))
        if not anchor_rows:
            return torch.tensor(1e-6, device=self.device, dtype=torch.bfloat16, requires_grad=True)

        # ============================================================
        # OPTIMIZATION: Collect ALL unique texts for batched encoding
        # ============================================================
        all_texts: List[str] = []
        text_to_idx: Dict[str, int] = {}
        
        # Track ranges for different text types
        row_indices: List[int] = []
        pos_context_indices: Dict[int, List[int]] = {}  # pid -> list of indices
        neg_context_indices: Dict[int, List[int]] = {}  # nid -> list of indices
        
        # Add anchor rows (deduplicated)
        for text in anchor_rows:
            if text not in text_to_idx:
                text_to_idx[text] = len(all_texts)
                all_texts.append(text)
            row_indices.append(text_to_idx[text])
        
        # Add positive and negative context sentences (deduplicated)
        unique_pos_contexts: Dict[int, List[str]] = {}
        unique_neg_contexts: Dict[int, List[str]] = {}
        
        for triplet in triplet_batch:
            pid = triplet['positive_id']
            nid = triplet['negative_id']
            
            if pid not in unique_pos_contexts:
                pos_texts = triplet.get('positive_texts', triplet.get('positive_context', []))
                unique_pos_contexts[pid] = pos_texts
                pos_context_indices[pid] = []
                for text in pos_texts:
                    if text not in text_to_idx:
                        text_to_idx[text] = len(all_texts)
                        all_texts.append(text)
                    pos_context_indices[pid].append(text_to_idx[text])
            
            if nid not in unique_neg_contexts:
                neg_texts = triplet.get('negative_texts', triplet.get('negative_context', []))
                unique_neg_contexts[nid] = neg_texts
                neg_context_indices[nid] = []
                for text in neg_texts:
                    if text not in text_to_idx:
                        text_to_idx[text] = len(all_texts)
                        all_texts.append(text)
                    neg_context_indices[nid].append(text_to_idx[text])
        
        # Update stats
        total_texts_before_dedup = len(anchor_rows) + sum(len(c) for c in unique_pos_contexts.values()) + sum(len(c) for c in unique_neg_contexts.values())
        self._encoding_stats['total_texts'] += total_texts_before_dedup
        self._encoding_stats['unique_texts'] += len(all_texts)
        
        # ============================================================
        # SINGLE BATCHED FORWARD PASS - The key optimization!
        # ============================================================
        all_embeddings = self._chunked_encode_with_grad(all_texts)
        
        # ============================================================
        # Extract embeddings using precomputed indices
        # ============================================================
        # Row embeddings
        row_embedding_indices = torch.tensor(row_indices, device=self.device, dtype=torch.long)
        row_embeddings = all_embeddings[row_embedding_indices]
        
        # Positive context embeddings (mean over sentences per context)
        positive_embeddings: Dict[int, torch.Tensor] = {}
        for pid, indices in pos_context_indices.items():
            idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
            pos_emb = all_embeddings[idx_tensor]
            positive_embeddings[pid] = pos_emb  # Keep all sentence embeddings for similarity computation
        
        # Negative context embeddings
        negative_embeddings: Dict[int, torch.Tensor] = {}
        for nid, indices in neg_context_indices.items():
            idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
            neg_emb = all_embeddings[idx_tensor]
            negative_embeddings[nid] = neg_emb
        
        # ============================================================
        # Compute losses (same logic as before, but using cached embeddings)
        # ============================================================
        pos_ids = list(positive_embeddings.keys())
        neg_ids = list(negative_embeddings.keys())
        
        # Precompute global hardest negative if enabled
        hardest_neg_id: Optional[int] = None
        if self.use_hard_negative_mining and len(neg_ids) > 0:
            scores = []
            with torch.no_grad():
                for nid in neg_ids:
                    neg_emb_ = negative_embeddings[nid]
                    score_ = util.cos_sim(row_embeddings, neg_emb_).mean()
                    scores.append(score_)
            if scores:
                stacked = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, device=self.device) for s in scores])
                _, idx = torch.topk(stacked, k=1)
                hardest_neg_id = neg_ids[int(idx[0].item())]

        batch_losses: List[torch.Tensor] = []
        
        # Compute loss per triplet
        for triplet in triplet_batch:
            pid = triplet['positive_id']
            nid = triplet['negative_id']
            if pid not in positive_embeddings or nid not in negative_embeddings:
                continue
            pos_emb = positive_embeddings[pid]
            neg_emb = negative_embeddings[nid]

            # Average row×sentence cosine similarities to scalar scores
            pos_sim = util.cos_sim(row_embeddings, pos_emb).mean()
            if not pos_sim.requires_grad and self._anchor_param is not None:
                pos_sim = pos_sim + 0.0 * self._anchor_param.view(-1)[0]
            
            if hardest_neg_id is not None:
                neg_emb_sel = negative_embeddings[hardest_neg_id]
                neg_sim = util.cos_sim(row_embeddings, neg_emb_sel).mean()
                if not neg_sim.requires_grad and self._anchor_param is not None:
                    neg_sim = neg_sim + 0.0 * self._anchor_param.view(-1)[0]
            else:
                neg_sim = util.cos_sim(row_embeddings, neg_emb).mean()
                if not neg_sim.requires_grad and self._anchor_param is not None:
                    neg_sim = neg_sim + 0.0 * self._anchor_param.view(-1)[0]

            if self.ranking_loss_type == 'infonce':
                tau = max(self.infonce_tau, 1e-6)
                logits = torch.stack([pos_sim, neg_sim], dim=0) / tau
                labels = torch.tensor(0, device=logits.device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
            else:
                diff = neg_sim - pos_sim + self.margin
                loss = F.softplus(diff * self.scale)

            batch_losses.append(loss)

        if batch_losses:
            total = torch.mean(torch.stack(batch_losses)) + 1e-6
            # Ensure the final loss retains a path to encoder parameters
            if not total.requires_grad:
                trainable_params = [p for p in self.sentence_encoder.parameters() if p.requires_grad]
                if trainable_params:
                    anchor = trainable_params[0].view(-1)[0]
                    total = total + 0.0 * anchor
            return total
        return torch.tensor(1e-6, device=self.device, dtype=torch.bfloat16, requires_grad=True)
    
    def get_encoding_stats(self) -> Dict[str, Any]:
        """Return encoding statistics for monitoring optimization effectiveness."""
        stats = self._encoding_stats.copy()
        if stats['total_texts'] > 0:
            stats['dedup_ratio'] = stats['unique_texts'] / stats['total_texts']
            stats['avg_texts_per_forward'] = stats['unique_texts'] / max(stats['forward_passes'], 1)
        return stats
    
    def reset_encoding_stats(self):
        """Reset encoding statistics (call at start of each epoch)."""
        self._encoding_stats = {'total_texts': 0, 'unique_texts': 0, 'forward_passes': 0}


def compute_teacher_pair_similarities(
    row_embeddings: torch.Tensor,
    sentence_embeddings: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Compute "teacher" pair similarities from raw embeddings (frozen encoder output).
    
    This creates the supervision signal for attention distillation: the direct
    cosine similarities between row and sentence embeddings serve as soft labels
    for what LOKI's pair_scores should approximate.
    
    Args:
        row_embeddings: [batch_size, num_rows, embedding_dim] or [num_rows, embedding_dim]
        sentence_embeddings: [batch_size, num_sentences, embedding_dim] or [num_sentences, embedding_dim]
        temperature: Temperature for softmax normalization (lower = sharper distribution)
        
    Returns:
        teacher_distribution: [batch_size, num_rows, num_sentences] soft probability distribution
    """
    # Handle both batched and unbatched inputs
    if row_embeddings.dim() == 2:
        row_embeddings = row_embeddings.unsqueeze(0)
    if sentence_embeddings.dim() == 2:
        sentence_embeddings = sentence_embeddings.unsqueeze(0)
    
    # Compute cosine similarity matrix: [batch_size, num_rows, num_sentences]
    # Using torch.cosine_similarity with unsqueeze for proper broadcasting
    teacher_scores = torch.cosine_similarity(
        row_embeddings.unsqueeze(2),      # [batch, num_rows, 1, embed_dim]
        sentence_embeddings.unsqueeze(1),  # [batch, 1, num_sents, embed_dim]
        dim=-1
    )  # [batch_size, num_rows, num_sentences]
    
    # Apply temperature scaling and softmax to create probability distribution
    # The teacher distribution represents "where attention should go" for each row
    teacher_distribution = F.softmax(teacher_scores / temperature, dim=-1)
    
    return teacher_distribution


class AttentionDistillationLoss(nn.Module):
    """
    Attention Distillation Loss for preserving zero-shot row-sentence alignment quality.
    
    Key Insight:
    - The frozen encoder already provides good row-sentence similarities zero-shot
    - LOKI's cross-attention transforms these into pair_scores
    - During training, the global loss may cause attention to collapse to salient pairs only
    - This distillation loss encourages LOKI's pair_scores to match the teacher distribution
    
    The distillation target is the row-sentence cosine similarity matrix from the frozen encoder,
    converted to a probability distribution. LOKI's pair_scores should approximate this.
    
    Loss Computation:
    L_distill = KL(teacher_distribution || student_distribution)
    
    where:
    - teacher_distribution = softmax(cosine_sim(row_emb, sent_emb) / tau_teacher)
    - student_distribution = softmax(pair_scores / tau_student)
    """
    
    def __init__(self,
                 teacher_temperature: float = 0.1,
                 student_temperature: float = 0.1,
                 loss_type: str = "kl_div",
                 symmetric: bool = False,
                 teacher_hub_centering: bool = True):
        """
        Initialize the attention distillation loss.
        
        Args:
            teacher_temperature: Temperature for teacher distribution (lower = sharper)
                Recommended: 0.05-0.2 for focused teacher signal
            student_temperature: Temperature for student distribution (matches teacher)
                Recommended: Same as teacher_temperature for consistent KL
            loss_type: Type of distillation loss:
                - "kl_div": KL divergence (default, asymmetric)
                - "mse": Mean squared error on distributions
                - "cosine": 1 - cosine similarity of flattened distributions
                - "js_div": Jensen-Shannon divergence (symmetric KL)
            symmetric: If True, compute bidirectional distillation:
                L = KL(teacher || student) + KL(student || teacher)
            teacher_hub_centering: Subtract per-sentence mean across rows before
                softmax so universal hub sentences do not dominate every row.
        """
        super().__init__()
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.loss_type = loss_type
        self.symmetric = symmetric
        self.teacher_hub_centering = teacher_hub_centering
        
        print(f"Initialized AttentionDistillationLoss:")
        print(f"  Teacher temperature: {teacher_temperature}")
        print(f"  Student temperature: {student_temperature}")
        print(f"  Loss type: {loss_type}")
        print(f"  Symmetric: {symmetric}")
        print(f"  Teacher hub centering: {teacher_hub_centering}")
    
    def forward(self,
                student_pair_scores: torch.Tensor,
                row_embeddings: torch.Tensor,
                sentence_embeddings: torch.Tensor,
                row_attention_weights: Optional[torch.Tensor] = None,
                col_attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the attention distillation loss.
        
        Args:
            student_pair_scores: LOKI's pair scores [batch, num_rows, num_sentences]
            row_embeddings: Raw row embeddings from encoder [batch, num_rows, embed_dim]
            sentence_embeddings: Raw sentence embeddings [batch, num_sentences, embed_dim]
            row_attention_weights: Optional row-wise attention for column distillation
            col_attention_weights: Optional col-wise attention for row distillation
            
        Returns:
            Distillation loss scalar
        """
        # Handle edge cases
        if student_pair_scores.numel() == 0:
            return torch.tensor(0.0, device=student_pair_scores.device, 
                              dtype=student_pair_scores.dtype, requires_grad=True)
        
        # Ensure consistent batch dimension
        if student_pair_scores.dim() == 2:
            student_pair_scores = student_pair_scores.unsqueeze(0)
        if row_embeddings.dim() == 2:
            row_embeddings = row_embeddings.unsqueeze(0)
        if sentence_embeddings.dim() == 2:
            sentence_embeddings = sentence_embeddings.unsqueeze(0)
        
        batch_size = student_pair_scores.shape[0]
        
        # Compute teacher distribution from raw embeddings
        with torch.no_grad():
            teacher_scores = torch.cosine_similarity(
                row_embeddings.unsqueeze(2),
                sentence_embeddings.unsqueeze(1),
                dim=-1
            )  # [batch, num_rows, num_sentences]

            if self.teacher_hub_centering and teacher_scores.shape[-2] > 1:
                teacher_scores = teacher_scores - teacher_scores.mean(dim=-2, keepdim=True)
            
            # Teacher row-wise distribution (for each row, where should it attend?)
            teacher_row_dist = F.softmax(
                teacher_scores / self.teacher_temperature, dim=-1
            )  # [batch, num_rows, num_sentences]
            
            # Teacher column-wise distribution (for each sentence, which rows match?)
            teacher_col_dist = F.softmax(
                teacher_scores / self.teacher_temperature, dim=-2
            )  # [batch, num_rows, num_sentences]
        
        # Compute student distribution from LOKI's pair_scores
        student_row_dist = F.softmax(
            student_pair_scores / self.student_temperature, dim=-1
        )
        student_col_dist = F.softmax(
            student_pair_scores / self.student_temperature, dim=-2
        )
        
        # Compute distillation loss based on loss_type
        if self.loss_type == "kl_div":
            # KL(teacher || student) - encourages student to cover teacher's mass
            # Using log_softmax for numerical stability
            student_log_row_dist = F.log_softmax(
                student_pair_scores / self.student_temperature, dim=-1
            )
            student_log_col_dist = F.log_softmax(
                student_pair_scores / self.student_temperature, dim=-2
            )
            
            # Row-wise KL: for each row, match attention over sentences
            row_kl = F.kl_div(
                student_log_row_dist, 
                teacher_row_dist, 
                reduction='batchmean',
                log_target=False
            )
            
            # Column-wise KL: for each sentence, match attention over rows
            col_kl = F.kl_div(
                student_log_col_dist,
                teacher_col_dist,
                reduction='batchmean',
                log_target=False
            )
            
            # Combine row and column distillation
            loss = (row_kl + col_kl) / 2.0
            
            if self.symmetric:
                # Add reverse KL: KL(student || teacher)
                teacher_log_row_dist = torch.log(teacher_row_dist + 1e-10)
                teacher_log_col_dist = torch.log(teacher_col_dist + 1e-10)
                
                reverse_row_kl = F.kl_div(
                    teacher_log_row_dist,
                    student_row_dist,
                    reduction='batchmean',
                    log_target=False
                )
                reverse_col_kl = F.kl_div(
                    teacher_log_col_dist,
                    student_col_dist,
                    reduction='batchmean',
                    log_target=False
                )
                
                loss = loss + (reverse_row_kl + reverse_col_kl) / 2.0
                
        elif self.loss_type == "mse":
            # MSE between distributions
            row_mse = F.mse_loss(student_row_dist, teacher_row_dist)
            col_mse = F.mse_loss(student_col_dist, teacher_col_dist)
            loss = (row_mse + col_mse) / 2.0
            
        elif self.loss_type == "cosine":
            # Cosine similarity loss on flattened distributions
            student_flat = student_row_dist.reshape(batch_size, -1)
            teacher_flat = teacher_row_dist.reshape(batch_size, -1)
            cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
            loss = 1.0 - cos_sim.mean()
            
        elif self.loss_type == "js_div":
            # Jensen-Shannon divergence (symmetric)
            m_row = 0.5 * (student_row_dist + teacher_row_dist)
            m_col = 0.5 * (student_col_dist + teacher_col_dist)
            
            student_log_row = torch.log(student_row_dist + 1e-10)
            teacher_log_row = torch.log(teacher_row_dist + 1e-10)
            m_log_row = torch.log(m_row + 1e-10)
            
            js_row = 0.5 * (
                F.kl_div(m_log_row, student_row_dist, reduction='batchmean', log_target=False) +
                F.kl_div(m_log_row, teacher_row_dist, reduction='batchmean', log_target=False)
            )
            
            student_log_col = torch.log(student_col_dist + 1e-10)
            teacher_log_col = torch.log(teacher_col_dist + 1e-10)
            m_log_col = torch.log(m_col + 1e-10)
            
            js_col = 0.5 * (
                F.kl_div(m_log_col, student_col_dist, reduction='batchmean', log_target=False) +
                F.kl_div(m_log_col, teacher_col_dist, reduction='batchmean', log_target=False)
            )
            
            loss = (js_row + js_col) / 2.0
            
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Handle NaN
        if torch.isnan(loss):
            return torch.tensor(0.0, device=student_pair_scores.device,
                              dtype=student_pair_scores.dtype, requires_grad=True)
        
        return loss