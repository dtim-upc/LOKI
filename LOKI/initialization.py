"""
Advanced Initialization Techniques for Attention Mechanisms

This module provides various initialization strategies to prevent attention collapse
in transformer-based models, particularly for cross-attention mechanisms.

Available Methods:
1. xavier_uniform - Standard Xavier/Glorot uniform initialization (current baseline)
2. kaiming_uniform - Kaiming/He initialization with fan-out mode
3. orthogonal - Orthogonal initialization (best for attention collapse prevention)
4. attention_specific - Purpose-built for attention mechanisms
5. t5_style - T5/PaLM-style initialization used in modern LLMs
6. diverse_attention - Novel approach for diverse attention patterns
7. multiscale - Multi-scale initialization for robust attention
"""

import torch
import torch.nn as nn
import math
from typing import List, Union, Tuple


def _parse_layers(layers: List[nn.Module]) -> Tuple[List[nn.Module], bool]:
    """
    Parse layers to handle both unidirectional (3 layers) and bidirectional (6 layers) cases.
    
    Returns:
        Tuple of (all_layers, is_bidirectional)
        - For unidirectional: returns ([W_Q, W_K, W_V, W_Q, W_K, W_V], False)
        - For bidirectional: returns (layers, True)
    """
    if len(layers) == 3:
        # Unidirectional case: duplicate layers for compatibility
        W_Q, W_K, W_V = layers
        all_layers = [W_Q, W_K, W_V, W_Q, W_K, W_V]  # Duplicate for consistency
        return all_layers, False
    elif len(layers) == 6:
        # Bidirectional case: use as-is
        return layers, True
    else:
        raise ValueError(f"Expected 3 or 6 layers, got {len(layers)}")


class AttentionInitializer:
    """
    Centralized attention weight initialization with multiple strategies.
    """
    
    @staticmethod
    def xavier_uniform(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Standard Xavier/Glorot uniform initialization with configurable gains.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (q_gain, k_gain, v_gain, bias_range)
        """
        params = {
            'q_gain': 3.0,
            'k_gain': 3.0, 
            'v_gain': 1.0,
            'bias_range': 0.3
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        all_layers, is_bidirectional = _parse_layers(layers)
        forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V = all_layers
        
        # Apply initialization to unique layers to avoid double-initialization for unidirectional
        if is_bidirectional:
            # Bidirectional: initialize all 6 layers separately
            nn.init.xavier_uniform_(forward_W_Q.weight, gain=params['q_gain'])
            nn.init.xavier_uniform_(forward_W_K.weight, gain=params['k_gain'])
            nn.init.xavier_uniform_(forward_W_V.weight, gain=params['v_gain'])
            nn.init.xavier_uniform_(reverse_W_Q.weight, gain=params['q_gain'])
            nn.init.xavier_uniform_(reverse_W_K.weight, gain=params['k_gain'])
            nn.init.xavier_uniform_(reverse_W_V.weight, gain=params['v_gain'])
        else:
            # Unidirectional: only initialize the 3 unique layers
            nn.init.xavier_uniform_(forward_W_Q.weight, gain=params['q_gain'])  # Same as reverse_W_Q
            nn.init.xavier_uniform_(forward_W_K.weight, gain=params['k_gain'])  # Same as reverse_W_K
            nn.init.xavier_uniform_(forward_W_V.weight, gain=params['v_gain'])  # Same as reverse_W_V
        
        # Initialize biases with more variance to break symmetry
        for layer in layers:  # Use original layers to avoid duplicates
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -params['bias_range'], params['bias_range'])
        
        if params.get('_verbose', False):
            print(f"[OK] Applied Xavier Uniform initialization (Q/K gain: {params['q_gain']}, V gain: {params['v_gain']}, bias range: +/-{params['bias_range']})")

    @staticmethod
    def kaiming_uniform(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Kaiming/He initialization with fan-out mode for better diversity.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (qk_mode, v_mode, qk_nonlinearity, v_nonlinearity)
        """
        params = {
            'qk_mode': 'fan_out',
            'v_mode': 'fan_in',
            'qk_nonlinearity': 'relu',
            'v_nonlinearity': 'linear',
            'bias_std': 0.02
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        all_layers, is_bidirectional = _parse_layers(layers)
        forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V = all_layers
        
        # Apply initialization to unique layers to avoid double-initialization for unidirectional
        if is_bidirectional:
            # Bidirectional: initialize all 6 layers separately
            nn.init.kaiming_uniform_(forward_W_Q.weight, mode=params['qk_mode'], nonlinearity=params['qk_nonlinearity'])
            nn.init.kaiming_uniform_(forward_W_K.weight, mode=params['qk_mode'], nonlinearity=params['qk_nonlinearity'])
            nn.init.kaiming_uniform_(forward_W_V.weight, mode=params['v_mode'], nonlinearity=params['v_nonlinearity'])
            nn.init.kaiming_uniform_(reverse_W_Q.weight, mode=params['qk_mode'], nonlinearity=params['qk_nonlinearity'])
            nn.init.kaiming_uniform_(reverse_W_K.weight, mode=params['qk_mode'], nonlinearity=params['qk_nonlinearity'])
            nn.init.kaiming_uniform_(reverse_W_V.weight, mode=params['v_mode'], nonlinearity=params['v_nonlinearity'])
        else:
            # Unidirectional: only initialize the 3 unique layers
            nn.init.kaiming_uniform_(forward_W_Q.weight, mode=params['qk_mode'], nonlinearity=params['qk_nonlinearity'])
            nn.init.kaiming_uniform_(forward_W_K.weight, mode=params['qk_mode'], nonlinearity=params['qk_nonlinearity'])
            nn.init.kaiming_uniform_(forward_W_V.weight, mode=params['v_mode'], nonlinearity=params['v_nonlinearity'])
        
        # Small bias initialization
        for layer in layers:  # Use original layers to avoid duplicates
            if layer.bias is not None:
                nn.init.normal_(layer.bias, 0, params['bias_std'])
        
        if params.get('_verbose', False):
            print(f"[OK] Applied Kaiming Uniform initialization (Q/K: {params['qk_mode']}/{params['qk_nonlinearity']}, V: {params['v_mode']}/{params['v_nonlinearity']})")

    @staticmethod
    def orthogonal(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Orthogonal initialization for stable gradients and diverse attention patterns.
        Best choice for preventing attention collapse.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (q_gain, k_gain, v_gain, bias_std)
        """
        params = {
            'q_gain': 0.5,  # Further reduced from 0.8 to prevent attention collapse
            'k_gain': 0.5,  # Further reduced from 0.8 to prevent attention collapse
            'v_gain': 1.0,
            'bias_std': 0.02
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        all_layers, is_bidirectional = _parse_layers(layers)
        forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V = all_layers
        
        # Apply initialization to unique layers to avoid double-initialization for unidirectional
        if is_bidirectional:
            # Bidirectional: initialize all 6 layers separately
            nn.init.orthogonal_(forward_W_Q.weight, gain=params['q_gain'])
            nn.init.orthogonal_(forward_W_K.weight, gain=params['k_gain'])
            nn.init.orthogonal_(forward_W_V.weight, gain=params['v_gain'])
            nn.init.orthogonal_(reverse_W_Q.weight, gain=params['q_gain'])
            nn.init.orthogonal_(reverse_W_K.weight, gain=params['k_gain'])
            nn.init.orthogonal_(reverse_W_V.weight, gain=params['v_gain'])
        else:
            # Unidirectional: only initialize the 3 unique layers
            nn.init.orthogonal_(forward_W_Q.weight, gain=params['q_gain'])
            nn.init.orthogonal_(forward_W_K.weight, gain=params['k_gain'])
            nn.init.orthogonal_(forward_W_V.weight, gain=params['v_gain'])
        
        # Small bias initialization
        for layer in layers:  # Use original layers to avoid duplicates
            if layer.bias is not None:
                nn.init.normal_(layer.bias, 0, params['bias_std'])
        
        if params.get('_verbose', False):
            print(f"[OK] Applied Orthogonal initialization (Q/K gain: {params['q_gain']}, V gain: {params['v_gain']}, bias std: {params['bias_std']})")

    @staticmethod
    def attention_specific(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Attention-specific initialization based on Transformer best practices.
        Different variance for Q, K, V based on their roles.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (q_scale, k_scale, v_scale, zero_bias)
        """
        params = {
            'q_scale': 1.0,  # Reduced from 2.0 to prevent attention collapse
            'k_scale': 0.8,  # Reduced from 1.0 to prevent attention collapse
            'v_scale': 0.5,  # Smaller variance for stability
            'zero_bias': True
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        all_layers, is_bidirectional = _parse_layers(layers)
        forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V = all_layers
        
        d_model = attention_dim
        
        # Apply initialization to unique layers to avoid double-initialization for unidirectional
        if is_bidirectional:
            # Bidirectional: initialize all 6 layers separately
            nn.init.normal_(forward_W_Q.weight, 0, (params['q_scale'] / d_model) ** 0.5)
            nn.init.normal_(reverse_W_Q.weight, 0, (params['q_scale'] / d_model) ** 0.5)
            nn.init.normal_(forward_W_K.weight, 0, (params['k_scale'] / d_model) ** 0.5)
            nn.init.normal_(reverse_W_K.weight, 0, (params['k_scale'] / d_model) ** 0.5)
            nn.init.normal_(forward_W_V.weight, 0, (params['v_scale'] / d_model) ** 0.5)
            nn.init.normal_(reverse_W_V.weight, 0, (params['v_scale'] / d_model) ** 0.5)
        else:
            # Unidirectional: only initialize the 3 unique layers
            nn.init.normal_(forward_W_Q.weight, 0, (params['q_scale'] / d_model) ** 0.5)
            nn.init.normal_(forward_W_K.weight, 0, (params['k_scale'] / d_model) ** 0.5)
            nn.init.normal_(forward_W_V.weight, 0, (params['v_scale'] / d_model) ** 0.5)
        
        # Bias initialization
        for layer in layers:  # Use original layers to avoid duplicates
            if layer.bias is not None:
                if params['zero_bias']:
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.normal_(layer.bias, 0, 0.01)
        
        if params.get('_verbose', False):
            print(f"[OK] Applied Attention-Specific initialization (Q scale: {params['q_scale']}, K scale: {params['k_scale']}, V scale: {params['v_scale']})")

    @staticmethod
    def t5_style(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        T5/PaLM-style initialization used in modern large language models.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (factor, separate_v_scaling)
        """
        params = {
            'factor': 1.0,
            'separate_v_scaling': True
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        all_layers, is_bidirectional = _parse_layers(layers)
        forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V = all_layers
        
        # T5 uses a specific scaling factor
        factor = params['factor']
        d_model = attention_dim
        
        # Standard T5 scaling for Q and K
        qk_std = factor * (d_model ** -0.5)
        
        # Value scaling (can be same or different)
        if params['separate_v_scaling']:
            v_std = factor * (d_model ** -0.5) * 0.8  # Slightly smaller for V
        else:
            v_std = qk_std
        
        # Apply initialization to unique layers to avoid double-initialization for unidirectional
        if is_bidirectional:
            # Bidirectional: initialize all 6 layers separately
            for weight in [forward_W_Q.weight, forward_W_K.weight, 
                          reverse_W_Q.weight, reverse_W_K.weight]:
                nn.init.normal_(weight, 0, qk_std)
            for weight in [forward_W_V.weight, reverse_W_V.weight]:
                nn.init.normal_(weight, 0, v_std)
        else:
            # Unidirectional: only initialize the 3 unique layers
            for weight in [forward_W_Q.weight, forward_W_K.weight]:
                nn.init.normal_(weight, 0, qk_std)
            nn.init.normal_(forward_W_V.weight, 0, v_std)
        
        # Zero bias (T5 style)
        for layer in layers:
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        if params.get('_verbose', False):
            print(f"[OK] Applied T5-Style initialization (factor: {factor}, d_model: {d_model}, separate V scaling: {params['separate_v_scaling']})")

    @staticmethod
    def diverse_attention(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Novel approach to initialize weights for diverse attention patterns from the start.
        Designed to produce varied but controlled initial attention scores.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (target_range, v_gain, bias_diversity)
        """
        params = {
            'target_range': 1.0,  # Target initial attention scores in range [-target_range, target_range]
            'v_gain': 0.5,
            'bias_diversity': 0.1,
            'zero_v_bias': True
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        all_layers, is_bidirectional = _parse_layers(layers)
        forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V = all_layers
        
        d_model = attention_dim
        
        # Initialize Q and K to produce diverse but not extreme attention scores  
        # Target: initial attention scores in specified range after scaling
        q_std = (params['target_range'] / d_model) ** 0.5
        k_std = (params['target_range'] / d_model) ** 0.5
        
        # Apply initialization to unique layers to avoid double-initialization for unidirectional
        if is_bidirectional:
            # Bidirectional: initialize all 6 layers separately
            nn.init.normal_(forward_W_Q.weight, 0, q_std)
            nn.init.normal_(forward_W_K.weight, 0, k_std)
            nn.init.normal_(reverse_W_Q.weight, 0, q_std)
            nn.init.normal_(reverse_W_K.weight, 0, k_std)
            nn.init.xavier_uniform_(forward_W_V.weight, gain=params['v_gain'])
            nn.init.xavier_uniform_(reverse_W_V.weight, gain=params['v_gain'])
        else:
            # Unidirectional: only initialize the 3 unique layers
            nn.init.normal_(forward_W_Q.weight, 0, q_std)
            nn.init.normal_(forward_W_K.weight, 0, k_std)
            nn.init.xavier_uniform_(forward_W_V.weight, gain=params['v_gain'])
        
        # Bias initialization to encourage attention diversity
        for layer in [forward_W_Q, forward_W_K, reverse_W_Q, reverse_W_K]:
            if layer.bias is not None:
                # Random bias to break symmetry
                nn.init.uniform_(layer.bias, -params['bias_diversity'], params['bias_diversity'])
        
        for layer in [forward_W_V, reverse_W_V]:
            if layer.bias is not None:
                if params['zero_v_bias']:
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.normal_(layer.bias, 0, 0.01)
        
        if params.get('_verbose', False):
            print(f"[OK] Applied Diverse Attention initialization (target range: +/-{params['target_range']}, V gain: {params['v_gain']}, bias diversity: +/-{params['bias_diversity']})")

    @staticmethod
    def multiscale(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Multi-scale initialization with different scales for different components.
        Provides robust attention patterns across different scales.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (scales dict, adaptive_bias)
        """
        params = {
            'scales': {
                'query': 1.5,
                'key': 1.0,  
                'value': 0.8
            },
            'adaptive_bias': True,
            'base_gain': 1.0
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        all_layers, is_bidirectional = _parse_layers(layers)
        forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V = all_layers
        
        scales = params['scales']
        base_gain = params['base_gain']
        
        # Apply initialization to unique layers to avoid double-initialization for unidirectional
        if is_bidirectional:
            # Bidirectional: initialize all 6 layers separately
            for prefix, layers_group in [('forward', [forward_W_Q, forward_W_K, forward_W_V]),
                                       ('reverse', [reverse_W_Q, reverse_W_K, reverse_W_V])]:
                q_layer, k_layer, v_layer = layers_group
                nn.init.xavier_uniform_(q_layer.weight, gain=scales['query'] * base_gain)
                nn.init.xavier_uniform_(k_layer.weight, gain=scales['key'] * base_gain)
                nn.init.xavier_uniform_(v_layer.weight, gain=scales['value'] * base_gain)
        else:
            # Unidirectional: only initialize the 3 unique layers
            nn.init.xavier_uniform_(forward_W_Q.weight, gain=scales['query'] * base_gain)
            nn.init.xavier_uniform_(forward_W_K.weight, gain=scales['key'] * base_gain)
            nn.init.xavier_uniform_(forward_W_V.weight, gain=scales['value'] * base_gain)

            # Adaptive bias based on layer type
            if params['adaptive_bias']:
                if forward_W_Q.bias is not None:
                    nn.init.normal_(forward_W_Q.bias, 0, 0.05)
                if forward_W_K.bias is not None:
                    nn.init.normal_(forward_W_K.bias, 0, 0.02)
                if forward_W_V.bias is not None:
                    nn.init.zeros_(forward_W_V.bias)
            else:
                for layer in [forward_W_Q, forward_W_K, forward_W_V]:
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        if params.get('_verbose', False):
            print(f"[OK] Applied Multiscale initialization (Q: {scales['query']}, K: {scales['key']}, V: {scales['value']}, adaptive bias: {params['adaptive_bias']})")

    # =====================================================
    # DIAGNOSTIC INITIALIZATION METHODS FOR DEBUGGING
    # =====================================================
    
    @staticmethod
    def zeros(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Initialize all weights to zero.
        
        This creates a "blank slate" where attention is completely uniform initially.
        Useful for understanding if random structure is providing inductive bias.
        
        Args:
            layers: List of layers (3 for unidirectional, 6 for bidirectional)
            attention_dim: Attention dimension (unused here)
            method_params: Optional parameters (bias_value)
        """
        params = {
            'bias_value': 0.0
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        all_layers, is_bidirectional = _parse_layers(layers)
        
        # Apply zeros initialization to actual layers (avoid duplicates for unidirectional)
        unique_layers = layers  # Use original layers to avoid double-initialization
        roles = ['Q', 'K', 'V'] if len(layers) == 3 else ['fwd_Q', 'fwd_K', 'fwd_V', 'rev_Q', 'rev_K', 'rev_V']
        
        for layer, role in zip(unique_layers, roles):
            # Zero initialization for weights
            nn.init.zeros_(layer.weight)
            
            # Zero or small bias
            if layer.bias is not None:
                nn.init.constant_(layer.bias, params['bias_value'])
            
            if params.get('_verbose', False):
                print(f"    [OK] Applied zeros initialization to {role}")
    
    @staticmethod 
    def ones(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Initialize all weights to one (or a small constant).
        
        This creates strong uniform attention initially. Useful for testing
        if constant attention patterns are beneficial.
        
        Args:
            layers: List of layers (3 for unidirectional, 6 for bidirectional)
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (weight_value, bias_value)
        """
        params = {
            'weight_value': 0.01,  # Small constant instead of 1.0 to avoid exploding gradients
            'bias_value': 0.0
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        unique_layers = layers  # Use original layers to avoid double-initialization
        roles = ['Q', 'K', 'V'] if len(layers) == 3 else ['fwd_Q', 'fwd_K', 'fwd_V', 'rev_Q', 'rev_K', 'rev_V']
        
        for layer, role in zip(unique_layers, roles):
            # Constant initialization for weights
            nn.init.constant_(layer.weight, params['weight_value'])
            
            # Zero bias
            if layer.bias is not None:
                nn.init.constant_(layer.bias, params['bias_value'])
            
            if params.get('_verbose', False):
                print(f"    [OK] Applied ones initialization to {role} (value={params['weight_value']})")
    
    @staticmethod
    def diagonal(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Initialize weights to create identity-like mappings.
        
        Sets diagonal elements to a constant value and off-diagonal to zero/small values.
        This encourages direct row-to-sentence mappings initially.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (diagonal_value, off_diagonal_value, bias_value)
        """
        params = {
            'diagonal_value': 1.0,
            'off_diagonal_value': 0.0,
            'bias_value': 0.0
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        unique_layers = layers  # Use original layers to avoid double-initialization
        roles = ['Q', 'K', 'V'] if len(layers) == 3 else ['fwd_Q', 'fwd_K', 'fwd_V', 'rev_Q', 'rev_K', 'rev_V']
        
        for layer, role in zip(unique_layers, roles):
            weight = layer.weight
            out_features, in_features = weight.shape
            
            # Initialize to off-diagonal value
            nn.init.constant_(layer.weight, params['off_diagonal_value'])
            
            # Set diagonal elements
            min_dim = min(out_features, in_features)
            with torch.no_grad():
                for i in range(min_dim):
                    weight[i, i] = params['diagonal_value']
            
            # Zero bias
            if layer.bias is not None:
                nn.init.constant_(layer.bias, params['bias_value'])
            
            if params.get('_verbose', False):
                print(f"    [OK] Applied diagonal initialization to {role} (diag={params['diagonal_value']}, off-diag={params['off_diagonal_value']})")
    
    @staticmethod
    def identity_preserving(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Initialize to preserve input identity as much as possible.
        
        This makes the attention mechanism initially act like an identity function,
        useful for testing if the sophisticated architecture's benefit comes from
        the residual connections rather than the attention computation.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (identity_scale, noise_scale)
        """
        params = {
            'identity_scale': 1.0,
            'noise_scale': 0.01  # Small amount of noise to break symmetry
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        unique_layers = layers  # Use original layers to avoid double-initialization
        roles = ['Q', 'K', 'V'] if len(layers) == 3 else ['fwd_Q', 'fwd_K', 'fwd_V', 'rev_Q', 'rev_K', 'rev_V']
        
        for layer, role in zip(unique_layers, roles):
            weight = layer.weight
            out_features, in_features = weight.shape
            
            # Start with small random noise
            nn.init.normal_(layer.weight, 0, params['noise_scale'])
            
            # Add identity component where possible
            min_dim = min(out_features, in_features)
            with torch.no_grad():
                for i in range(min_dim):
                    weight[i, i] += params['identity_scale']
            
            # Zero bias
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            
            if params.get('_verbose', False):
                print(f"    [OK] Applied identity-preserving initialization to {role} (scale={params['identity_scale']}, noise={params['noise_scale']})")
    
    @staticmethod
    def sparse_random(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Initialize with sparse random connections.
        
        Most weights are zero, with only a fraction being non-zero random values.
        This tests if sparsity in initialization helps with the row-sentence task.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (sparsity_ratio, init_std)
        """
        params = {
            'sparsity_ratio': 0.1,  # 10% of weights are non-zero
            'init_std': 0.02
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        unique_layers = layers  # Use original layers to avoid double-initialization
        roles = ['Q', 'K', 'V'] if len(layers) == 3 else ['fwd_Q', 'fwd_K', 'fwd_V', 'rev_Q', 'rev_K', 'rev_V']
        
        for layer, role in zip(unique_layers, roles):
            weight = layer.weight
            
            # Start with zeros
            nn.init.zeros_(layer.weight)
            
            # Randomly select positions to be non-zero
            out_features, in_features = weight.shape
            total_elements = out_features * in_features
            num_nonzero = int(total_elements * params['sparsity_ratio'])
            
            # Random indices for non-zero elements
            flat_indices = torch.randperm(total_elements)[:num_nonzero]
            
            with torch.no_grad():
                flat_weight = weight.view(-1)
                flat_weight[flat_indices] = torch.randn(num_nonzero) * params['init_std']
            
            # Zero bias
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            
            if params.get('_verbose', False):
                print(f"    [OK] Applied sparse random initialization to {role} (sparsity={params['sparsity_ratio']:.1%}, std={params['init_std']})")
    
    @staticmethod
    def scaled_uniform(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Initialize with uniform distribution at different scales.
        
        Tests the effect of initialization magnitude on performance.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (scale_factor)
        """
        params = {
            'scale_factor': 0.1  # Scale factor for uniform distribution [-scale, scale]
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        unique_layers = layers  # Use original layers to avoid double-initialization
        roles = ['Q', 'K', 'V'] if len(layers) == 3 else ['fwd_Q', 'fwd_K', 'fwd_V', 'rev_Q', 'rev_K', 'rev_V']
        
        for layer, role in zip(unique_layers, roles):
            # Uniform initialization with specific scale
            scale = params['scale_factor']
            nn.init.uniform_(layer.weight, -scale, scale)
            
            # Small bias
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -scale/10, scale/10)
            
            if params.get('_verbose', False):
                print(f"    [OK] Applied scaled uniform initialization to {role} (scale=+/-{scale})")
    
    @staticmethod
    def asymmetric_forward_reverse(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Initialize forward and reverse attention differently.
        
        This tests if the bidirectional benefit comes from asymmetric attention patterns.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (forward_scale, reverse_scale, forward_method, reverse_method)
        """
        params = {
            'forward_scale': 1.0,
            'reverse_scale': 0.1,
            'forward_method': 'xavier',  # 'xavier', 'normal', 'uniform'
            'reverse_method': 'normal'
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        all_layers, is_bidirectional = _parse_layers(layers)
        forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V = all_layers
        
        def apply_method(layer, method, scale, role):
            if method == 'xavier':
                nn.init.xavier_uniform_(layer.weight, gain=scale)
            elif method == 'normal':
                nn.init.normal_(layer.weight, 0, scale * 0.02)
            elif method == 'uniform':
                nn.init.uniform_(layer.weight, -scale * 0.1, scale * 0.1)
            
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            
            if params.get('_verbose', False):
                print(f"    [OK] Applied {method} initialization to {role} (scale={scale})")
        
        # Apply initialization to unique layers to avoid double-initialization for unidirectional
        if is_bidirectional:
            # Bidirectional: initialize all 6 layers separately
            forward_layers = [forward_W_Q, forward_W_K, forward_W_V]
            reverse_layers = [reverse_W_Q, reverse_W_K, reverse_W_V]
            forward_roles = ['fwd_Q', 'fwd_K', 'fwd_V']
            reverse_roles = ['rev_Q', 'rev_K', 'rev_V']
            
            for layer, role in zip(forward_layers, forward_roles):
                apply_method(layer, params['forward_method'], params['forward_scale'], role)
            
            for layer, role in zip(reverse_layers, reverse_roles):
                apply_method(layer, params['reverse_method'], params['reverse_scale'], role)
        else:
            # Unidirectional: only initialize the 3 unique layers using forward method
            unidirectional_layers = [forward_W_Q, forward_W_K, forward_W_V]
            unidirectional_roles = ['Q', 'K', 'V']
            
            for layer, role in zip(unidirectional_layers, unidirectional_roles):
                apply_method(layer, params['forward_method'], params['forward_scale'], role)
    
    @staticmethod
    def tiny_random(layers: List[nn.Module], attention_dim: int, method_params: dict = None) -> None:
        """
        Initialize with very small random values.
        
        This tests if the current random initialization scale is too large.
        
        Args:
            layers: List of [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
            attention_dim: Attention dimension for scaling
            method_params: Optional parameters (scale)
        """
        params = {
            'scale': 1e-4  # Very small scale
        }
        if method_params:
            params.update(method_params)
        
        # Handle both unidirectional (3 layers) and bidirectional (6 layers) cases
        unique_layers = layers  # Use original layers to avoid double-initialization
        roles = ['Q', 'K', 'V'] if len(layers) == 3 else ['fwd_Q', 'fwd_K', 'fwd_V', 'rev_Q', 'rev_K', 'rev_V']
        
        for layer, role in zip(unique_layers, roles):
            # Very small random initialization
            nn.init.normal_(layer.weight, 0, params['scale'])
            
            # Zero bias
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            
            if params.get('_verbose', False):
                print(f"    [OK] Applied tiny random initialization to {role} (std={params['scale']})")


def initialize_attention_weights(layers: List[nn.Module], 
                               attention_dim: int,
                               method: str = "xavier_uniform",
                               method_params: dict = None) -> None:
    """
    Initialize attention weights using the specified method.
    
    Args:
        layers: List of 6 layers [forward_W_Q, forward_W_K, forward_W_V, reverse_W_Q, reverse_W_K, reverse_W_V]
        attention_dim: Attention dimension for scaling
        method: Initialization method name
        method_params: Optional parameters specific to the method
        
    Available methods:
        Standard methods:
        - xavier_uniform: Standard Xavier/Glorot uniform (current baseline)
        - kaiming_uniform: Kaiming/He initialization with fan-out mode
        - orthogonal: Orthogonal initialization (recommended for attention collapse)
        - attention_specific: Purpose-built for attention mechanisms  
        - t5_style: T5/PaLM-style initialization
        - diverse_attention: Novel approach for diverse attention patterns
        - multiscale: Multi-scale initialization for robust attention
        
        Diagnostic methods (for debugging initialization effects):
        - zeros: Initialize all weights to zero (tests uniform attention baseline)
        - ones: Initialize all weights to small constant (tests constant attention patterns)
        - diagonal: Initialize with identity-like diagonal structure (tests direct mappings)
        - identity_preserving: Initialize to preserve input identity (tests residual benefit)
        - sparse_random: Initialize with sparse random connections (tests sparsity effects)
        - scaled_uniform: Initialize with uniform distribution at specific scale (tests magnitude)
        - asymmetric_forward_reverse: Initialize forward/reverse differently (tests asymmetry)
        - tiny_random: Initialize with very small random values (tests scale effects)
    """
    
    # Validate layer count - support both unidirectional (3) and bidirectional (6) models
    if len(layers) not in [3, 6]:
        raise ValueError(f"Expected 3 layers (unidirectional) or 6 layers (bidirectional), got {len(layers)} layers")
    
    method_map = {
        "xavier_uniform": AttentionInitializer.xavier_uniform,
        "kaiming_uniform": AttentionInitializer.kaiming_uniform, 
        "orthogonal": AttentionInitializer.orthogonal,
        "attention_specific": AttentionInitializer.attention_specific,
        "t5_style": AttentionInitializer.t5_style,
        "diverse_attention": AttentionInitializer.diverse_attention,
        "multiscale": AttentionInitializer.multiscale,
        # Diagnostic initialization methods for debugging
        "zeros": AttentionInitializer.zeros,
        "ones": AttentionInitializer.ones,
        "diagonal": AttentionInitializer.diagonal,
        "identity_preserving": AttentionInitializer.identity_preserving,
        "sparse_random": AttentionInitializer.sparse_random,
        "scaled_uniform": AttentionInitializer.scaled_uniform,
        "asymmetric_forward_reverse": AttentionInitializer.asymmetric_forward_reverse,
        "tiny_random": AttentionInitializer.tiny_random
    }
    
    if method not in method_map:
        available_methods = list(method_map.keys())
        raise ValueError(f"Unknown initialization method '{method}'. Available methods: {available_methods}")
    
    # Only print if verbose (default off to reduce log spam)
    if method_params and method_params.get('_verbose', False):
        print(f"\n[INFO] Initializing attention weights using '{method}' method...")
    method_map[method](layers, attention_dim, method_params)
    if method_params and method_params.get('_verbose', False):
        print(f"[INFO] Attention dimension: {attention_dim}")


def get_available_methods() -> List[str]:
    """Return list of available initialization methods."""
    return [
        # Standard methods
        "xavier_uniform",
        "kaiming_uniform", 
        "orthogonal",
        "attention_specific",
        "t5_style",
        "diverse_attention",
        "multiscale",
        # Diagnostic methods for debugging
        "zeros",
        "ones", 
        "diagonal",
        "identity_preserving",
        "sparse_random",
        "scaled_uniform",
        "asymmetric_forward_reverse",
        "tiny_random"
    ]


def get_method_description(method: str) -> str:
    """Get description of an initialization method."""
    descriptions = {
        # Standard methods
        "xavier_uniform": "Standard Xavier/Glorot uniform initialization (current baseline)",
        "kaiming_uniform": "Kaiming/He initialization with fan-out mode for better diversity",
        "orthogonal": "Orthogonal initialization (best for preventing attention collapse)", 
        "attention_specific": "Purpose-built for attention mechanisms with role-based variance",
        "t5_style": "T5/PaLM-style initialization used in modern large language models",
        "diverse_attention": "Novel approach for diverse attention patterns from start",
        "multiscale": "Multi-scale initialization for robust attention across scales",
        
        # Diagnostic methods for debugging
        "zeros": "Initialize all weights to zero (tests uniform attention baseline)",
        "ones": "Initialize all weights to small constant (tests constant attention patterns)",
        "diagonal": "Initialize with identity-like diagonal structure (tests direct mappings)",
        "identity_preserving": "Initialize to preserve input identity (tests if benefit comes from residuals)",
        "sparse_random": "Initialize with sparse random connections (tests effect of sparsity)",
        "scaled_uniform": "Initialize with uniform distribution at specific scale (tests magnitude effects)",
        "asymmetric_forward_reverse": "Initialize forward/reverse attention differently (tests asymmetry)",
        "tiny_random": "Initialize with very small random values (tests if current scale is too large)"
    }
    return descriptions.get(method, "Unknown method")


def get_recommended_method_params(method: str) -> dict:
    """Get recommended parameters for each method."""
    recommendations = {
        # Standard methods
        "xavier_uniform": {
            "q_gain": 3.0,
            "k_gain": 3.0,
            "v_gain": 1.0,
            "bias_range": 0.3
        },
        "kaiming_uniform": {
            "qk_mode": "fan_out",
            "v_mode": "fan_in", 
            "qk_nonlinearity": "relu",
            "v_nonlinearity": "linear",
            "bias_std": 0.02
        },
        "orthogonal": {
            "q_gain": 0.5,
            "k_gain": 0.5,
            "v_gain": 1.0,
            "bias_std": 0.02
        },
        "attention_specific": {
            "q_scale": 1.0,
            "k_scale": 0.8,
            "v_scale": 0.5,
            "zero_bias": True
        },
        "t5_style": {
            "factor": 1.0,
            "separate_v_scaling": True
        },
        "diverse_attention": {
            "target_range": 1.0,
            "v_gain": 0.5,
            "bias_diversity": 0.1,
            "zero_v_bias": True
        },
        "multiscale": {
            "scales": {
                "query": 1.5,
                "key": 1.0,
                "value": 0.8
            },
            "adaptive_bias": True,
            "base_gain": 1.0
        },
        
        # Diagnostic methods
        "zeros": {
            "bias_value": 0.0
        },
        "ones": {
            "weight_value": 0.01,
            "bias_value": 0.0
        },
        "diagonal": {
            "diagonal_value": 1.0,
            "off_diagonal_value": 0.0,
            "bias_value": 0.0
        },
        "identity_preserving": {
            "identity_scale": 1.0,
            "noise_scale": 0.01
        },
        "sparse_random": {
            "sparsity_ratio": 0.1,
            "init_std": 0.02
        },
        "scaled_uniform": {
            "scale_factor": 0.1
        },
        "asymmetric_forward_reverse": {
            "forward_scale": 1.0,
            "reverse_scale": 0.1,
            "forward_method": "xavier",
            "reverse_method": "normal"
        },
        "tiny_random": {
            "scale": 1e-4
        }
    }
    return recommendations.get(method, {})


if __name__ == "__main__":
    # Example usage and testing
    print("Available initialization methods:")
    for method in get_available_methods():
        print(f"  - {method}: {get_method_description(method)}") 