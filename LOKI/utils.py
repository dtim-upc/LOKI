"""
Utils Module

Provides utility functions for GPU memory management and plot saving.
"""

import os
import gc
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

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


def save_plot_multi_format(output_file: str, dpi: int = 300, bbox_inches: str = 'tight', 
                           formats: tuple = ('png', 'pdf')) -> None:
    """
    Save the current matplotlib figure in multiple formats.
    
    Args:
        output_file: Path to save the plot (with or without extension)
        dpi: Resolution for raster formats (default: 300)
        bbox_inches: Bounding box setting (default: 'tight')
        formats: Tuple of formats to save (default: ('png', 'pdf'))
    
    Example:
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> save_plot_multi_format('my_plot.png')  # Saves both my_plot.png and my_plot.pdf
    """
    # Convert to Path object for easier manipulation
    output_path = Path(output_file)
    
    # Get the base path without extension
    # Note: We need to handle filenames with dots in them (e.g., "model-v0.1_plot.png")
    # Path.stem only removes the LAST extension, but with_suffix treats ".1" as a suffix
    # So we use the stem but construct the path via string concatenation
    base_name = output_path.stem  # e.g., "model-v0.1_plot" from "model-v0.1_plot.png"
    parent_dir = output_path.parent
    
    # Save in each requested format
    for fmt in formats:
        # Use string concatenation to avoid with_suffix treating ".1" as an extension
        save_path = parent_dir / f"{base_name}.{fmt}"
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, format=fmt)
        print(f"Saved plot to {save_path}")