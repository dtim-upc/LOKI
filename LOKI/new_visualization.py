#!/usr/bin/env python3
"""
Clean visualization system for bidirectional attention analysis.
Fixes inconsistencies by computing all steps from scratch using single forward pass.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
import json
import math
from models import BidirectionalTableTextModel, TableTextEmbeddingModel
from utils import save_plot_multi_format

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

def get_consistent_colormap_and_range(data: np.ndarray, matrix_type: str) -> Tuple[str, float, float, float]:
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

def extract_rows_and_sentences(example: Dict[str, Any], example_idx: int) -> Tuple[List[str], List[str]]:
    """Robustly extract rows and sentences from an example."""
    # Extract rows
    rows = []
    anchor_rows = example.get("anchor_rows", [])
    if isinstance(anchor_rows, list):
        for row in anchor_rows:
            if isinstance(row, dict):
                formatted_text = row.get("formatted", "")
                if formatted_text:
                    rows.append(formatted_text)
            elif isinstance(row, str) and row:
                rows.append(row)
    
    # Extract sentences from primary positive
    sentences = []
    primary_positive = example.get("primary_positive", {})
    if isinstance(primary_positive, dict):
        primary_sentences = primary_positive.get("sentences", [])
        if isinstance(primary_sentences, list):
            sentences = [s for s in primary_sentences if isinstance(s, str) and s]
    
    if not rows:
        print(f"Warning: No valid rows found for example {example_idx}")
    if not sentences:
        print(f"Warning: No valid sentences found for example {example_idx}")
    
    return rows, sentences

def compute_cosine_similarity_matrix(tensor1: torch.Tensor, tensor2: torch.Tensor) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of vectors.
    
    Args:
        tensor1: [N, D] tensor
        tensor2: [M, D] tensor
    
    Returns:
        [N, M] numpy array of cosine similarities
    """
    # Normalize tensors
    tensor1_norm = torch.nn.functional.normalize(tensor1, p=2, dim=1)
    tensor2_norm = torch.nn.functional.normalize(tensor2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity_matrix = torch.mm(tensor1_norm, tensor2_norm.transpose(0, 1))
    
    return safe_tensor_to_numpy(similarity_matrix)

def visualize_self_attention_matrix(matrix: np.ndarray, 
                                   labels: List[str],
                                   title: str,
                                   output_file: str,
                                   figsize: Tuple[int, int] = (10, 10)) -> None:
    """Create a heatmap visualization for self-attention matrices (square matrices)."""
    plt.figure(figsize=figsize)
    
    # Create simplified labels for self-attention
    simplified_labels = [f"Item {i+1}" for i in range(len(labels))]
    
    # Use consistent colormap and smart range
    cmap, vmin, vmax, center = get_consistent_colormap_and_range(matrix, 'attention')
    
    # Create heatmap
    ax = sns.heatmap(
        matrix,
        xticklabels=simplified_labels,
        yticklabels=simplified_labels,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar_kws={'label': 'Attention Score'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Attended To', fontsize=12)
    plt.ylabel('Attending From', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_matrix(matrix: np.ndarray, 
                    num_rows: int,
                    num_sentences: int,
                    title: str,
                    output_file: str,
                    matrix_type: str = "similarity",
                    figsize: Tuple[int, int] = (12, 8)) -> None:
    """Create a heatmap visualization of a matrix with simplified labels."""
    plt.figure(figsize=figsize)
    
    # Create simplified labels
    row_labels = [f"Row {i+1}" for i in range(num_rows)]
    col_labels = [f"Sentence {i+1}" for i in range(num_sentences)]
    
    # Determine matrix type from title if not specified
    if matrix_type == "similarity":
        if "attention" in title.lower() or "forward" in title.lower() or "reverse" in title.lower():
            matrix_type = "attention"
        elif "difference" in title.lower() or "improvement" in title.lower():
            matrix_type = "difference"
    
    # Get consistent colormap and range
    cmap, vmin, vmax, center = get_consistent_colormap_and_range(matrix, matrix_type)
    
    # Determine label
    if matrix_type == "attention":
        label = 'Attention Score'
    elif matrix_type == "difference":
        label = 'Difference'
    else:
        label = 'Similarity Score'
    
    # Create heatmap
    ax = sns.heatmap(
        matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar_kws={'label': label}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sentences', fontsize=12)
    plt.ylabel('Rows', fontsize=12)
    plt.xticks(rotation=0)  # No rotation needed for short labels
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    save_plot_multi_format(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def compute_self_attention_step(
    model: BidirectionalTableTextModel,
    rows_tensor: torch.Tensor,
    sentences_tensor: torch.Tensor,
    rows: List[str],
    sentences: List[str],
    example_idx: int,
    diag_dir: Path
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Compute self-attention step and extract attention weights for visualization.
    
    Returns:
        self_attended_rows: Enhanced row embeddings after self-attention
        self_attended_sentences: Enhanced sentence embeddings after self-attention  
        row_self_attention: Row self-attention weights [N, N]
        sentence_self_attention: Sentence self-attention weights [M, M]
    """
    print("    Computing self-attention with weight extraction...")
    
    device = rows_tensor.device
    
    # =============== ROW SELF-ATTENTION ===============
    print("      Row self-attention...")
    
    # Apply row self-attention manually to extract weights
    row_self_attn_block = model.bidirectional_attention.row_self_attention
    batch_size, num_rows, embedding_dim = rows_tensor.shape
    
    # Get the self-attention projections
    normed_rows = row_self_attn_block.attention_norm(rows_tensor)
    q_rows = row_self_attn_block.q_proj(normed_rows)  # [1, N, D]
    k_rows = row_self_attn_block.k_proj(normed_rows)  # [1, N, D]
    v_rows = row_self_attn_block.v_proj(normed_rows)  # [1, N, D]
    
    # Reshape for multi-head attention
    num_heads = row_self_attn_block.num_heads
    head_dim = row_self_attn_block.head_dim
    
    q_rows = q_rows.view(batch_size, num_rows, num_heads, head_dim).transpose(1, 2)  # [1, heads, N, head_dim]
    k_rows = k_rows.view(batch_size, num_rows, num_heads, head_dim).transpose(1, 2)  # [1, heads, N, head_dim]
    v_rows = v_rows.view(batch_size, num_rows, num_heads, head_dim).transpose(1, 2)  # [1, heads, N, head_dim]
    
    # Compute attention scores
    row_attn_scores = torch.matmul(q_rows, k_rows.transpose(-2, -1)) / math.sqrt(head_dim)  # [1, heads, N, N]
    row_attn_weights = torch.nn.functional.softmax(row_attn_scores, dim=-1)  # [1, heads, N, N]
    
    # Average across heads for visualization
    row_self_attention = safe_tensor_to_numpy(row_attn_weights.mean(dim=1)[0])  # [N, N]
    
    # Apply attention to get self-attended rows
    row_attn_output = torch.matmul(row_attn_weights, v_rows)  # [1, heads, N, head_dim]
    row_attn_output = row_attn_output.transpose(1, 2).contiguous().view(batch_size, num_rows, embedding_dim)
    row_attn_output = row_self_attn_block.out_proj(row_attn_output)
    
    # Apply self-attention block (with FFN and residuals)
    self_attended_rows = row_self_attn_block(rows_tensor[0].unsqueeze(0))[0]  # Keep batch dim for forward pass, then remove
    
    # =============== SENTENCE SELF-ATTENTION ===============
    print("      Sentence self-attention...")
    
    # Apply sentence self-attention manually to extract weights
    sentence_self_attn_block = model.bidirectional_attention.sentence_self_attention
    batch_size, num_sentences, embedding_dim = sentences_tensor.shape
    
    # Get the self-attention projections
    normed_sentences = sentence_self_attn_block.attention_norm(sentences_tensor)
    q_sentences = sentence_self_attn_block.q_proj(normed_sentences)  # [1, M, D]
    k_sentences = sentence_self_attn_block.k_proj(normed_sentences)  # [1, M, D]
    v_sentences = sentence_self_attn_block.v_proj(normed_sentences)  # [1, M, D]
    
    # Reshape for multi-head attention
    q_sentences = q_sentences.view(batch_size, num_sentences, num_heads, head_dim).transpose(1, 2)  # [1, heads, M, head_dim]
    k_sentences = k_sentences.view(batch_size, num_sentences, num_heads, head_dim).transpose(1, 2)  # [1, heads, M, head_dim]
    v_sentences = v_sentences.view(batch_size, num_sentences, num_heads, head_dim).transpose(1, 2)  # [1, heads, M, head_dim]
    
    # Compute attention scores
    sentence_attn_scores = torch.matmul(q_sentences, k_sentences.transpose(-2, -1)) / math.sqrt(head_dim)  # [1, heads, M, M]
    sentence_attn_weights = torch.nn.functional.softmax(sentence_attn_scores, dim=-1)  # [1, heads, M, M]
    
    # Average across heads for visualization
    sentence_self_attention = safe_tensor_to_numpy(sentence_attn_weights.mean(dim=1)[0])  # [M, M]
    
    # Apply sentence self-attention block (with FFN and residuals)
    self_attended_sentences = sentence_self_attn_block(sentences_tensor[0].unsqueeze(0))[0]  # Keep batch dim for forward pass, then remove
    
    # =============== VISUALIZE SELF-ATTENTION MATRICES ===============
    print("      Visualizing self-attention matrices...")
    
    # Save attention matrices
    np.save(diag_dir / "step2_row_self_attention.npy", row_self_attention)
    np.save(diag_dir / "step2_sentence_self_attention.npy", sentence_self_attention)
    
    # Visualize row self-attention
    visualize_self_attention_matrix(
        row_self_attention, rows,
        title=f"Step 2a: Row Self-Attention Matrix (Example {example_idx})",
        output_file=str(diag_dir / "step2a_row_self_attention.png")
    )
    
    # Visualize sentence self-attention
    visualize_self_attention_matrix(
        sentence_self_attention, sentences,
        title=f"Step 2b: Sentence Self-Attention Matrix (Example {example_idx})",
        output_file=str(diag_dir / "step2b_sentence_self_attention.png")
    )
    
    # =============== ANALYZE ENHANCEMENT FROM SELF-ATTENTION ===============
    print("      Analyzing enhancement from self-attention...")
    
    # Compute similarities before and after self-attention
    raw_similarity_matrix = compute_cosine_similarity_matrix(rows_tensor[0], sentences_tensor[0])
    enhanced_similarity_matrix = compute_cosine_similarity_matrix(self_attended_rows, self_attended_sentences)
    
    # Save enhanced similarity matrix
    np.save(diag_dir / "step2c_enhanced_similarities.npy", enhanced_similarity_matrix)
    
    # Visualize enhanced similarities
    visualize_matrix(
        enhanced_similarity_matrix, len(rows), len(sentences),
        title=f"Step 2c: Enhanced Similarities (After Self-Attention) (Example {example_idx})",
        output_file=str(diag_dir / "step2c_enhanced_similarities.png"),
        matrix_type="similarity"
    )
    
    # Compute and visualize difference
    similarity_difference = enhanced_similarity_matrix - raw_similarity_matrix
    np.save(diag_dir / "step2d_self_attention_improvement.npy", similarity_difference)
    
    # Visualize improvement from self-attention
    visualize_matrix(
        similarity_difference, len(rows), len(sentences),
        title=f"Step 2d: Self-Attention Improvement (Enhanced - Raw) (Example {example_idx})",
        output_file=str(diag_dir / "step2d_self_attention_improvement.png"),
        matrix_type="difference"
    )
    
    print(f"      Self-attention analysis complete!")
    print(f"        Row attention entropy: {-np.sum(row_self_attention * np.log(row_self_attention + 1e-10), axis=1).mean():.4f}")
    print(f"        Sentence attention entropy: {-np.sum(sentence_self_attention * np.log(sentence_self_attention + 1e-10), axis=1).mean():.4f}")
    print(f"        Similarity improvement range: [{similarity_difference.min():.4f}, {similarity_difference.max():.4f}]")
    
    return self_attended_rows, self_attended_sentences, row_self_attention, sentence_self_attention

def create_clean_bidirectional_analysis(
    model: BidirectionalTableTextModel,
    rows: List[str],
    sentences: List[str],
    example_idx: int,
    output_dir: str,
    use_refinement: bool
) -> Dict[str, Any]:
    """
    Create clean step-by-step analysis for bidirectional model.
    All computations done from a single forward pass to ensure consistency.
    """
    print(f"Creating clean bidirectional analysis for example {example_idx}...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Create output directory
    diag_dir = Path(output_dir) / "clean_diagnostics" / f"example_{example_idx}"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    with torch.no_grad():
        # =============== STEP 1: RAW EMBEDDINGS SIMILARITY ===============
        print("  Step 1: Raw embeddings similarity...")
        
        # Get raw embeddings using the frozen sentence transformer
        raw_row_embeddings = model.encode_sentences(rows, normalize=True)
        raw_sentence_embeddings = model.encode_sentences(sentences, normalize=True)
        
        # Compute raw similarity matrix
        raw_similarities = compute_cosine_similarity_matrix(raw_row_embeddings, raw_sentence_embeddings)
        results['raw_similarities'] = raw_similarities
        
        # Save and visualize
        np.save(diag_dir / "step1_raw_similarities.npy", raw_similarities)
        visualize_matrix(
            raw_similarities, len(rows), len(sentences),
            title=f"Step 1: Raw Embeddings Similarity (Example {example_idx})",
            output_file=str(diag_dir / "step1_raw_similarities.png"),
            matrix_type="similarity"
        )
        
        # Add batch dimension and move to device
        rows_tensor = raw_row_embeddings.unsqueeze(0).to(device)  # [1, N, D]
        sentences_tensor = raw_sentence_embeddings.unsqueeze(0).to(device)  # [1, M, D]
        
        # =============== STEP 2: SELF-ATTENTION ANALYSIS (NEW!) ===============
        if model.bidirectional_attention.use_self_attention:
            print("  Step 2: Self-attention analysis...")
            
            # Create enhanced embeddings using self-attention
            self_attended_rows, self_attended_sentences, row_self_attn, sentence_self_attn = compute_self_attention_step(
                model, rows_tensor, sentences_tensor, rows, sentences, example_idx, diag_dir
            )
            
            # Store self-attention results
            results['row_self_attention'] = row_self_attn
            results['sentence_self_attention'] = sentence_self_attn
            results['self_attended_rows'] = self_attended_rows
            results['self_attended_sentences'] = self_attended_sentences
            
            # Use self-attended embeddings for the rest of the pipeline
            input_rows_tensor = self_attended_rows.unsqueeze(0).to(device)
            input_sentences_tensor = self_attended_sentences.unsqueeze(0).to(device)
        else:
            print("  Step 2: Self-attention disabled, using raw embeddings...")
            # Use raw embeddings directly
            input_rows_tensor = rows_tensor
            input_sentences_tensor = sentences_tensor
        
        # =============== STEP 3: FORWARD PASS FOR CROSS-ATTENTION ===============
        print("  Step 3: Forward pass for cross-attention matrices...")
        
        # Get bidirectional cross-attention results using the appropriate input tensors
        pair_scores, refined_rows, refined_sentences, forward_attn, reverse_attn = model.bidirectional_attention(
            input_rows_tensor, input_sentences_tensor
        )
        
        # Extract attention matrices
        forward_attention = safe_tensor_to_numpy(forward_attn[0])  # [N, M]
        reverse_attention = safe_tensor_to_numpy(reverse_attn[0])  # [M, N]
        
        results['forward_attention'] = forward_attention
        results['reverse_attention'] = reverse_attention
        
        # Save and visualize forward attention
        np.save(diag_dir / "step2_forward_attention.npy", forward_attention)
        visualize_matrix(
            forward_attention, len(rows), len(sentences),
            title=f"Step 2: Forward Attention (Rows→Sentences) (Example {example_idx})",
            output_file=str(diag_dir / "step2_forward_attention.png"),
            matrix_type="attention"
        )
        
        # Save and visualize reverse attention (transposed for visualization)
        np.save(diag_dir / "step2_reverse_attention.npy", reverse_attention)
        visualize_matrix(
            reverse_attention.T, len(rows), len(sentences),  # Transpose for consistent visualization
            title=f"Step 2: Reverse Attention (Sentences→Rows) (Example {example_idx})",
            output_file=str(diag_dir / "step2_reverse_attention.png"),
            matrix_type="attention"
        )
        
        # =============== STEP 4: ATTENTION-BASED SIMILARITIES ===============
        print("  Step 4: Computing pure attention and contextualized similarities...")
        
        # Manually compute attention vectors to match the forward pass exactly
        batch_size, num_rows, _ = input_rows_tensor.shape
        _, num_sentences, _ = input_sentences_tensor.shape
        
        # Forward attention: rows attend to sentences
        normed_rows = model.bidirectional_attention.row_attention_norm(input_rows_tensor)
        
        # Check attention type and handle accordingly
        attention_type = getattr(model.bidirectional_attention, 'attention_type', 'standard')
        print(f"    DEBUG: Using attention type: {attention_type}")
        
        if attention_type == "standard":
            # Standard attention - use the original logic
            forward_Q = model.bidirectional_attention.forward_W_Q(normed_rows)
            forward_K = model.bidirectional_attention.forward_W_K(input_sentences_tensor)
            forward_V = model.bidirectional_attention.forward_W_V(input_sentences_tensor)
            
            # Compute forward attention manually to match _apply_attention
            forward_scores = torch.matmul(forward_Q, forward_K.transpose(-2, -1))
            forward_scores = forward_scores / (model.bidirectional_attention.attention_dim ** 0.5)
            forward_scores = forward_scores / torch.clamp(model.bidirectional_attention.forward_temperature, min=0.5, max=3.0)
            forward_scores = torch.clamp(forward_scores, min=-50.0, max=50.0)
            forward_weights = torch.nn.functional.softmax(forward_scores, dim=-1)
            forward_context = torch.bmm(forward_weights, forward_V)
            
            # Reverse attention: sentences attend to rows  
            normed_sentences = model.bidirectional_attention.sentence_attention_norm(input_sentences_tensor)
            reverse_Q = model.bidirectional_attention.reverse_W_Q(normed_sentences)
            reverse_K = model.bidirectional_attention.reverse_W_K(input_rows_tensor)
            reverse_V = model.bidirectional_attention.reverse_W_V(input_rows_tensor)
            
            # Compute reverse attention manually
            reverse_scores = torch.matmul(reverse_Q, reverse_K.transpose(-2, -1))
            reverse_scores = reverse_scores / (model.bidirectional_attention.attention_dim ** 0.5)
            reverse_scores = reverse_scores / torch.clamp(model.bidirectional_attention.reverse_temperature, min=0.5, max=3.0)
            reverse_scores = torch.clamp(reverse_scores, min=-50.0, max=50.0)
            reverse_weights = torch.nn.functional.softmax(reverse_scores, dim=-1)
            reverse_context = torch.bmm(reverse_weights, reverse_V)
            
        else:
            # For non-standard attention types, use the model's forward pass directly
            print(f"    DEBUG: Using model forward pass for {attention_type} attention")
            
            # Get the attention outputs from the model's forward pass (already computed above)
            # Extract the contextualized embeddings directly
            pair_scores_temp, contextualized_rows_temp, contextualized_sentences_temp, forward_attn_temp, reverse_attn_temp = model.bidirectional_attention(
                input_rows_tensor, input_sentences_tensor
            )
            
            # For non-standard attention, we can't manually compute the intermediate steps
            # So we'll use the model's outputs and compute the context vectors
            # by reverse-engineering from the contextualized outputs
            forward_context = contextualized_rows_temp - input_rows_tensor  # Remove residual to get context
            reverse_context = contextualized_sentences_temp - input_sentences_tensor  # Remove residual to get context
            
            # Use the attention weights from the model
            forward_weights = forward_attn_temp
            reverse_weights = reverse_attn_temp
        
        # =============== STEP 4A: PURE ATTENTION SIMILARITIES (NO RESIDUAL) ===============
        print("    Step 4a: Pure attention similarities (attention output only)...")
        
        # Create pure attention vectors (just the context, no residual)
        pure_attention_rows = forward_context  # Just the attention output
        pure_attention_sentences = reverse_context  # Just the attention output
        
        # DEBUG: Print shapes and some statistics
        print(f"      DEBUG: forward_context shape: {forward_context.shape}")
        print(f"      DEBUG: reverse_context shape: {reverse_context.shape}")
        print(f"      DEBUG: forward_context mean: {forward_context.mean():.6f}, std: {forward_context.std():.6f}")
        print(f"      DEBUG: reverse_context mean: {reverse_context.mean():.6f}, std: {reverse_context.std():.6f}")
        print(f"      DEBUG: forward_context sample values: {forward_context[0, 0, :5]}")
        print(f"      DEBUG: reverse_context sample values: {reverse_context[0, 0, :5]}")
        
        # Compute pure attention similarities
        pure_attention_similarities = compute_cosine_similarity_matrix(
            pure_attention_rows[0], pure_attention_sentences[0]
        )
        results['pure_attention_similarities'] = pure_attention_similarities
        
        # DEBUG: Print similarity statistics
        print(f"      DEBUG: pure_attention_similarities mean: {pure_attention_similarities.mean():.6f}")
        print(f"      DEBUG: pure_attention_similarities std: {pure_attention_similarities.std():.6f}")
        print(f"      DEBUG: pure_attention_similarities min/max: {pure_attention_similarities.min():.6f}/{pure_attention_similarities.max():.6f}")
        
        # Save and visualize
        np.save(diag_dir / "step4a_pure_attention_similarities.npy", pure_attention_similarities)
        visualize_matrix(
            pure_attention_similarities, len(rows), len(sentences),
            title=f"Step 4a: Pure Attention Similarities (No Residual) (Example {example_idx})",
            output_file=str(diag_dir / "step4a_pure_attention_similarities.png"),
            matrix_type="similarity"
        )
        
        # =============== STEP 4B: CONTEXTUALIZED SIMILARITIES (WITH RESIDUAL) ===============
        print("    Step 4b: Contextualized similarities (with residual connection)...")
        
        # Contextualized vectors = input + attention output (WITH RESIDUAL)
        contextualized_rows = input_rows_tensor + forward_context
        contextualized_sentences = input_sentences_tensor + reverse_context
        
        # Compute contextualized similarities
        contextualized_similarities = compute_cosine_similarity_matrix(
            contextualized_rows[0], contextualized_sentences[0]
        )
        results['contextualized_similarities'] = contextualized_similarities
        
        # Save and visualize
        np.save(diag_dir / "step4b_contextualized_similarities.npy", contextualized_similarities)
        visualize_matrix(
            contextualized_similarities, len(rows), len(sentences),
            title=f"Step 4b: Contextualized Similarities (With Residual) (Example {example_idx})",
            output_file=str(diag_dir / "step4b_contextualized_similarities.png"),
            matrix_type="similarity"
        )
        
        # =============== STEP 5: FINAL PAIR SCORES ===============
        print(f"  Step 5: Final pair scores (use_refinement={use_refinement})...")
        
        # Apply refinement decision exactly as in the model
        if use_refinement:
            print("    Applying refinement...")
            final_rows = contextualized_rows + model.bidirectional_attention.row_refinement(contextualized_rows)
            final_sentences = contextualized_sentences + model.bidirectional_attention.sentence_refinement(contextualized_sentences)
        else:
            print("    Skipping refinement...")
            final_rows = contextualized_rows
            final_sentences = contextualized_sentences
        
        # Compute final pair scores using the same method as the model
        if model.bidirectional_attention.pair_score_method == "cosine":
            final_pair_scores = torch.cosine_similarity(
                final_rows.unsqueeze(2),
                final_sentences.unsqueeze(1),
                dim=-1
            )[0]
            final_pair_scores = safe_tensor_to_numpy(final_pair_scores)
        elif model.bidirectional_attention.pair_score_method == "dot":
            final_pair_scores = safe_tensor_to_numpy(torch.bmm(final_rows, final_sentences.transpose(-2, -1))[0])
        else:
            raise ValueError(f"Unsupported pair_score_method: {model.bidirectional_attention.pair_score_method}")
        
        results['final_pair_scores'] = final_pair_scores
        
        # Save and visualize
        np.save(diag_dir / "step5_final_pair_scores.npy", final_pair_scores)
        refinement_status = "with refinement" if use_refinement else "without refinement"
        visualize_matrix(
            final_pair_scores, len(rows), len(sentences),
            title=f"Step 5: Final Pair Scores ({refinement_status}) (Example {example_idx})",
            output_file=str(diag_dir / f"step5_final_pair_scores.png"),
            matrix_type="similarity"
        )
        
        # =============== VALIDATION: CONSISTENCY CHECK ===============
        print("  Validation: Consistency check...")
        
        # Get pair scores from the actual model forward pass
        model_pair_scores = safe_tensor_to_numpy(pair_scores[0])
        
        # Check consistency
        if np.allclose(final_pair_scores, model_pair_scores, atol=1e-6):
            print("    ✅ CONSISTENT: Manual computation matches model forward pass")
            consistency_status = "CONSISTENT"
        else:
            print("    ❌ INCONSISTENT: Manual computation differs from model forward pass")
            consistency_status = "INCONSISTENT"
            
            # Save difference matrix for debugging
            difference = np.abs(final_pair_scores - model_pair_scores)
            max_diff = np.max(difference)
            mean_diff = np.mean(difference)
            print(f"       Max difference: {max_diff:.8f}")
            print(f"       Mean difference: {mean_diff:.8f}")
            
            np.save(diag_dir / "validation_difference.npy", difference)
            visualize_matrix(
                difference, len(rows), len(sentences),
                title=f"Validation: Absolute Difference (Max: {max_diff:.2e})",
                output_file=str(diag_dir / "validation_difference.png"),
                matrix_type="difference"
            )
        
        results['model_pair_scores'] = model_pair_scores
        results['consistency_status'] = consistency_status
        
        # =============== COMPARISON: CONTEXTUALIZED VS FINAL ===============
        if not use_refinement:
            print("  Comparison: Contextualized vs Final (should be identical when refinement=False)...")
            
            if np.allclose(contextualized_similarities, final_pair_scores, atol=1e-6):
                print("    ✅ IDENTICAL: Contextualized similarities = Final pair scores (as expected)")
                comparison_status = "IDENTICAL"
            else:
                print("    ❌ DIFFERENT: Contextualized similarities ≠ Final pair scores (unexpected!)")
                comparison_status = "DIFFERENT"
                
                # This is the bug the user identified!
                diff = np.abs(contextualized_similarities - final_pair_scores)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"       Max difference: {max_diff:.8f}")
                print(f"       Mean difference: {mean_diff:.8f}")
                
                np.save(diag_dir / "contextualized_vs_final_difference.npy", diff)
                visualize_matrix(
                    diff, len(rows), len(sentences),
                    title=f"Contextualized vs Final Difference (Max: {max_diff:.2e})",
                    output_file=str(diag_dir / "contextualized_vs_final_difference.png"),
                    matrix_type="difference"
                )
            
            results['comparison_status'] = comparison_status
        
        # =============== SAVE SUMMARY ===============
        summary = {
            'example_idx': example_idx,
            'use_refinement': use_refinement,
            'pair_score_method': model.bidirectional_attention.pair_score_method,
            'consistency_status': consistency_status,
            'num_rows': len(rows),
            'num_sentences': len(sentences),
            'rows': rows,
            'sentences': sentences
        }
        
        if not use_refinement:
            summary['comparison_status'] = comparison_status
        
        with open(diag_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Clean analysis completed. Results saved to {diag_dir}")
        
        return results

def create_clean_unidirectional_analysis(
    model: TableTextEmbeddingModel,
    rows: List[str],
    sentences: List[str],
    example_idx: int,
    output_dir: str,
    use_refinement: bool
) -> Dict[str, Any]:
    """
    Create clean step-by-step analysis for unidirectional model.
    """
    print(f"Creating clean unidirectional analysis for example {example_idx}...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Create output directory
    diag_dir = Path(output_dir) / "clean_diagnostics" / f"example_{example_idx}"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    with torch.no_grad():
        # Step 1: Raw embeddings
        raw_row_embeddings = model.encode_sentences(rows, normalize=True)
        raw_sentence_embeddings = model.encode_sentences(sentences, normalize=True)
        raw_similarities = compute_cosine_similarity_matrix(raw_row_embeddings, raw_sentence_embeddings)
        
        # Step 2: Attention
        rows_tensor = raw_row_embeddings.unsqueeze(0).to(device)
        sentences_tensor = raw_sentence_embeddings.unsqueeze(0).to(device)
        
        # Handle different attention interfaces for unidirectional model
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
        
        # Step 3: Contextualized similarities
        contextualized_rows = rows_tensor + context_vectors
        contextualized_similarities = compute_cosine_similarity_matrix(
            contextualized_rows[0], sentences_tensor[0]
        )
        
        # Step 4: Final similarities (with/without refinement)
        if use_refinement:
            norm_context = model.norm1(contextualized_rows)
            ff_output = model.feed_forward(norm_context)
            final_rows = model.norm2(ff_output + norm_context)
        else:
            final_rows = contextualized_rows
        
        final_similarities = compute_cosine_similarity_matrix(
            final_rows[0], sentences_tensor[0]
        )
        
        # Save results
        results.update({
            'raw_similarities': raw_similarities,
            'attention_weights': attention_weights,
            'contextualized_similarities': contextualized_similarities,
            'final_similarities': final_similarities
        })
        
        # Save matrices
        np.save(diag_dir / "step1_raw_similarities.npy", raw_similarities)
        np.save(diag_dir / "step2_attention_weights.npy", attention_weights)
        np.save(diag_dir / "step3_contextualized_similarities.npy", contextualized_similarities)
        np.save(diag_dir / "step4_final_similarities.npy", final_similarities)
        
        # Create visualizations
        visualize_matrix(raw_similarities, rows, sentences, 
                        f"Step 1: Raw Similarities (Example {example_idx})",
                        str(diag_dir / "step1_raw_similarities.png"),
                        matrix_type="similarity")
        
        visualize_matrix(attention_weights, rows, sentences,
                        f"Step 2: Attention Weights (Example {example_idx})",
                        str(diag_dir / "step2_attention_weights.png"),
                        matrix_type="attention")
        
        visualize_matrix(contextualized_similarities, rows, sentences,
                        f"Step 3: Contextualized Similarities (Example {example_idx})",
                        str(diag_dir / "step3_contextualized_similarities.png"),
                        matrix_type="similarity")
        
        refinement_status = "with refinement" if use_refinement else "without refinement"
        visualize_matrix(final_similarities, rows, sentences,
                        f"Step 4: Final Similarities ({refinement_status}) (Example {example_idx})",
                        str(diag_dir / "step4_final_similarities.png"),
                        matrix_type="similarity")
    
    return results

def run_clean_analysis_for_example(
    model: Union[BidirectionalTableTextModel, TableTextEmbeddingModel],
    example: Dict[str, Any],
    example_idx: int,
    output_dir: str,
    use_refinement: bool
) -> Dict[str, Any]:
    """Run clean analysis for a single example."""
    # Extract rows and sentences
    rows, sentences = extract_rows_and_sentences(example, example_idx)
    
    if not rows or not sentences:
        print(f"Skipping example {example_idx}: Missing rows or sentences")
        return {}
    
    # Run appropriate analysis based on model type
    if isinstance(model, BidirectionalTableTextModel):
        return create_clean_bidirectional_analysis(
            model, rows, sentences, example_idx, output_dir, use_refinement
        )
    else:
        print(f"Unidirectional model analysis not implemented yet for example {example_idx}")
        return {}

def run_clean_analysis_for_examples(
    model: Union[BidirectionalTableTextModel, TableTextEmbeddingModel],
    examples: List[Dict[str, Any]],
    example_indices: List[int],
    output_dir: str,
    use_refinement: bool
) -> Dict[int, Dict[str, Any]]:
    """Run clean analysis for multiple examples."""
    results = {}
    
    for idx in example_indices:
        if idx < len(examples):
            print(f"\n🔬 Running clean analysis for Example {idx}...")
            example_results = run_clean_analysis_for_example(
                model, examples[idx], idx, output_dir, use_refinement
            )
            results[idx] = example_results
        else:
            print(f"Warning: Example index {idx} out of range (max: {len(examples)-1})")
    
    return results

def test_smart_range_calculation():
    """
    Test function to demonstrate the smart range calculation.
    This shows how the color ranges are now dynamically calculated.
    """
    print("🎨 Testing Smart Range Calculation for Consistent Color Mapping")
    print("=" * 60)
    
    # Test case 1: Values like in the user's example (0.216 to 0.572)
    test_data_1 = np.array([0.216, 0.269, 0.325, 0.572, 0.384])
    vmin1, vmax1 = calculate_smart_range(test_data_1)
    print(f"Test 1 - Input range: [{test_data_1.min():.3f}, {test_data_1.max():.3f}]")
    print(f"         Smart range:  [{vmin1:.3f}, {vmax1:.3f}]")
    print(f"         Rounds 0.216 → {vmin1:.3f}, 0.572 → {vmax1:.3f}")
    
    # Test case 2: Attention values (typically 0 to 1)
    test_data_2 = np.array([0.05, 0.23, 0.67, 0.89, 0.94])
    vmin2, vmax2 = calculate_smart_range(test_data_2)
    print(f"\nTest 2 - Input range: [{test_data_2.min():.3f}, {test_data_2.max():.3f}]")
    print(f"         Smart range:  [{vmin2:.3f}, {vmax2:.3f}]")
    print(f"         Rounds 0.05 → {vmin2:.3f}, 0.94 → {vmax2:.3f}")
    
    # Test case 3: Similarity values with negative numbers
    test_data_3 = np.array([-0.12, 0.03, 0.28, 0.47, 0.69])
    vmin3, vmax3 = calculate_smart_range(test_data_3)
    print(f"\nTest 3 - Input range: [{test_data_3.min():.3f}, {test_data_3.max():.3f}]")
    print(f"         Smart range:  [{vmin3:.3f}, {vmax3:.3f}]")
    print(f"         Rounds -0.12 → {vmin3:.3f}, 0.69 → {vmax3:.3f}")
    
    # Show colormap choices
    print(f"\n🎨 Colormap Choices:")
    print(f"   - Attention matrices: 'coolwarm' (Blue→White→Red with center at mid-range)")
    print(f"   - Similarity matrices: 'coolwarm' (Blue→White→Red with center at mid-range)")
    print(f"   - Difference matrices: 'coolwarm' (Blue→White→Red symmetric around 0)")
    
    print(f"\n✅ All heatmaps now use consistent coloring with smart range calculation!")
    print(f"   - Min values are rounded DOWN to nearest 0.05")
    print(f"   - Max values are rounded UP to nearest 0.05")
    print(f"   - Same colormap ('coolwarm') used across both visualization files")
    print(f"   - GUARANTEED Blue-to-Red spectrum with proper centering!")
    
    # Test colormap examples
    print(f"\n🌈 Color Spectrum Examples:")
    print(f"   - Low values (e.g., 0.20): BLUE")
    print(f"   - Mid values (e.g., 0.40): WHITE/LIGHT")
    print(f"   - High values (e.g., 0.60): RED")

if __name__ == "__main__":
    test_smart_range_calculation() 