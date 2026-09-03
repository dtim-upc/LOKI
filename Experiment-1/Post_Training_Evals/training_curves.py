"""
Training Curves Tracking and Visualization Module

This module provides comprehensive tracking and visualization of training metrics
including loss curves, accuracy curves, and other training statistics.

Features:
- Real-time tracking of training loss per epoch and batch
- Validation accuracy tracking per epoch
- Optional validation loss tracking
- Automatic plotting and saving of curves
- JSON serialization for persistence
- Statistical analysis of training progress
- Multi-model comparison plots (LOKI vs FT-Encoder vs Uni variants vs baseline)

=============================================================================
MULTI-MODEL COMPARISON - USAGE INSTRUCTIONS
=============================================================================

Command Reference:
------------------

# Initial run (auto-discover models, create combined JSON, generate plots)
python training_curves.py --compare

# With custom model-input directory
python training_curves.py --compare --base_dir "my_models"

# With custom baseline model
python training_curves.py --compare --baseline_model "ModelA"

# Regenerate plots from existing combined JSON
python training_curves.py --compare --combined_json "output_plots/combined_comparison_data.json"

Expected Model Input Structure:
-------------------------------
Input_Models/
├── LOKI/
│   └── training_data/
│       └── *_training_curves.json
├── Uni (R-S)/
│   └── <run_name>/
│       └── training_data/
│           └── *_training_curves.json
├── Uni (S-R)/
│   └── <run_name>/
│       └── training_data/
│           └── *_training_curves.json
└── Model-3/  (optional)
    └── training_data/
        └── *_training_curves.json

Generated Outputs:
------------------
- combined_comparison_data.json  : Combined data from all models
- model_comparison_test_metrics.png/.pdf : Test metrics comparison plot
- model_comparison_training_metrics.png/.pdf : Training metrics comparison plot

Notes:
------
- Model names are normalized into display labels (e.g., `Uni (R⟶S)` and `Uni (S⟶R)`)
- Baseline values (frozen encoder) are extracted from the --baseline_model (default: LOKI)
- Vertical arrows show relative improvement of LOKI over baseline
=============================================================================
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from datetime import datetime
from loki_path import ensure_loki_on_path
from model_download import download_input_models

ensure_loki_on_path()

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from utils import save_plot_multi_format

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


# =============================================================================
# Custom Scale for Squeezed Y-Axis (compresses middle region)
# =============================================================================
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator, MultipleLocator


class SqueezedTransform(Transform):
    """
    Custom transform that compresses a middle region of the axis.
    Similar to log scale compression but for a specific range.
    
    Maps: [0, low] -> [0, low_out]
          [low, high] -> [low_out, high_out] (compressed)
          [high, 1] -> [high_out, 1]
    """
    input_dims = output_dims = 1
    
    def __init__(self, low=0.2, high=0.6, compression=0.3):
        """
        Args:
            low: Start of compressed region (data coords)
            high: End of compressed region (data coords)
            compression: How much to squeeze (0.3 = 30% of original height)
        """
        super().__init__()
        self.low = low
        self.high = high
        self.compression = compression
        
        # Calculate output positions
        # Original: [0, 0.2] takes 20%, [0.2, 0.6] takes 40%, [0.6, 1.0] takes 40%
        # Squeezed: [0, 0.2] takes 20%, [0.2, 0.6] takes 12% (30% of 40%), [0.6, 1.0] takes 68%
        self.low_out = low  # Keep lower region same
        self.mid_span = (high - low) * compression  # Compressed middle span
        self.high_out = low + self.mid_span  # End of compressed region in output
        
    def transform_non_affine(self, values):
        values = np.asarray(values)
        result = np.zeros_like(values, dtype=float)
        
        # Below low: linear [0, low] -> [0, low]
        mask_low = values <= self.low
        result[mask_low] = values[mask_low]
        
        # In compressed region: [low, high] -> [low_out, high_out]
        mask_mid = (values > self.low) & (values <= self.high)
        # Linear interpolation within compressed region
        t = (values[mask_mid] - self.low) / (self.high - self.low)
        result[mask_mid] = self.low_out + t * self.mid_span
        
        # Above high: [high, 1] -> [high_out, 1]
        mask_high = values > self.high
        t = (values[mask_high] - self.high) / (1.0 - self.high)
        result[mask_high] = self.high_out + t * (1.0 - self.high_out)
        
        return result
    
    def inverted(self):
        return SqueezedInverseTransform(self.low, self.high, self.compression)


class SqueezedInverseTransform(Transform):
    """Inverse of SqueezedTransform."""
    input_dims = output_dims = 1
    
    def __init__(self, low=0.2, high=0.6, compression=0.3):
        super().__init__()
        self.low = low
        self.high = high
        self.compression = compression
        
        self.low_out = low
        self.mid_span = (high - low) * compression
        self.high_out = low + self.mid_span
        
    def transform_non_affine(self, values):
        values = np.asarray(values)
        result = np.zeros_like(values, dtype=float)
        
        # Below low_out: [0, low] -> [0, low]
        mask_low = values <= self.low_out
        result[mask_low] = values[mask_low]
        
        # In compressed region: [low_out, high_out] -> [low, high]
        mask_mid = (values > self.low_out) & (values <= self.high_out)
        t = (values[mask_mid] - self.low_out) / self.mid_span
        result[mask_mid] = self.low + t * (self.high - self.low)
        
        # Above high_out: [high_out, 1] -> [high, 1]
        mask_high = values > self.high_out
        t = (values[mask_high] - self.high_out) / (1.0 - self.high_out)
        result[mask_high] = self.high + t * (1.0 - self.high)
        
        return result
    
    def inverted(self):
        return SqueezedTransform(self.low, self.high, self.compression)


class SqueezedScale(ScaleBase):
    """
    Custom scale that compresses a middle region of the y-axis.
    Use: ax.set_yscale('squeezed', low=0.2, high=0.6, compression=0.3)
    """
    name = 'squeezed'
    
    def __init__(self, axis, low=0.2, high=0.6, compression=0.3):
        super().__init__(axis)
        self.low = low
        self.high = high
        self.compression = compression
        
    def get_transform(self):
        return SqueezedTransform(self.low, self.high, self.compression)
    
    def set_default_locators_and_formatters(self, axis):
        # Use all ticks from 0 to 1 in 0.1 increments
        axis.set_major_locator(FixedLocator([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        axis.set_major_formatter(FixedFormatter(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']))
    
    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, 0), min(vmax, 1)


# Register the custom scale
plt.matplotlib.scale.register_scale(SqueezedScale)


def round_half_up(value: float, decimals: int = 2) -> float:
    """
    Round a value to specified decimal places using standard rounding.
    
    Args:
        value: The float value to round
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Rounded float value
    """
    return round(value, decimals)


class TrainingCurves:
    """
    Comprehensive training curves tracker with real-time visualization capabilities.
    
    Tracks:
    - Training loss per epoch (mean, min, max, std)
    - Training loss per batch (optional, for detailed analysis)
    - Validation accuracy per epoch
    - Validation loss per epoch (optional)
    - Learning rate schedule
    - Training time per epoch
    """
    
    def __init__(self, 
                 output_dir: str,
                 run_name: str = "training_run",
                 track_batch_losses: bool = True,
                 track_val_loss: bool = False,
                 track_row_sent_metrics: bool = False,
                 auto_save: bool = True,
                 auto_plot: bool = True):
        """
        Initialize the training curves tracker.
        
        Args:
            output_dir: Directory to save curves data and plots
            run_name: Name of the training run for file naming
            track_batch_losses: Whether to track individual batch losses
            track_val_loss: Whether to compute and track validation loss
            track_row_sent_metrics: Whether to track row-sentence level metrics
            auto_save: Whether to automatically save data after each epoch
            auto_plot: Whether to automatically generate plots after each epoch
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.track_batch_losses = track_batch_losses
        self.track_val_loss = track_val_loss
        self.track_row_sent_metrics = track_row_sent_metrics
        self.auto_save = auto_save
        self.auto_plot = auto_plot
        
        # Initialize tracking data
        self.reset_tracking()
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "training_plots"
        self.data_dir = self.output_dir / "training_data"
        self.plots_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        print(f"🎯 Training curves tracker initialized:")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   📊 Tracking batch losses: {track_batch_losses}")
        print(f"   📈 Tracking validation loss: {track_val_loss}")
        print(f"   🔍 Tracking row-sentence metrics: {track_row_sent_metrics}")
        print(f"   💾 Auto-save: {auto_save}")
        print(f"   🎨 Auto-plot: {auto_plot}")
    
    def reset_tracking(self):
        """Reset all tracking data (useful for new training runs)."""
        self.epochs = []
        self.train_loss_mean = []
        self.train_loss_min = []
        self.train_loss_max = []
        self.train_loss_std = []
        self.val_accuracy = []
        self.val_loss = [] if self.track_val_loss else None
        self.learning_rates = []
        self.epoch_times = []
        self.batch_losses = [] if self.track_batch_losses else None
        # Row-sentence metrics
        self.row_sent_overall_accuracy = [] if self.track_row_sent_metrics else None
        self.row_sent_avg_precision = [] if self.track_row_sent_metrics else None
        # Initial stage metrics (Stage 0, 1, 2 before training starts)
        self.initial_stage_metrics = {}
        # Best metrics tracking
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_test_overall_accuracy = 0.0
        self.best_test_epoch = 0
        self.best_test_avg_precision = 0.0
        self.best_test_precision_epoch = 0
        self.start_time = datetime.now()
    
    def add_epoch_0_data(self,
                        val_accuracy: float,
                        row_sent_overall_accuracy: Optional[float] = None,
                        row_sent_avg_precision: Optional[float] = None):
        """
        Add Epoch 0 data (untrained model baseline).
        
        Args:
            val_accuracy: Validation accuracy of untrained model
            row_sent_overall_accuracy: Optional row-sentence overall accuracy of untrained model
            row_sent_avg_precision: Optional row-sentence average precision of untrained model
        """
        # Add epoch 0
        self.epochs.append(0)
        
        # No training losses for epoch 0
        self.train_loss_mean.append(0.0)
        self.train_loss_min.append(0.0) 
        self.train_loss_max.append(0.0)
        self.train_loss_std.append(0.0)
        
        # Store accuracy metrics
        self.val_accuracy.append(val_accuracy)
        
        # No learning rate or epoch time for untrained model
        self.learning_rates.append(0.0)
        self.epoch_times.append(0.0)
        
        # No batch losses for epoch 0
        if self.track_batch_losses:
            self.batch_losses.append([])
        
        # No validation loss for epoch 0
        if self.track_val_loss:
            self.val_loss.append(0.0)
        
        # Store row-sentence metrics if tracking is enabled
        if self.track_row_sent_metrics:
            self.row_sent_overall_accuracy.append(row_sent_overall_accuracy if row_sent_overall_accuracy is not None else 0.0)
            self.row_sent_avg_precision.append(row_sent_avg_precision if row_sent_avg_precision is not None else 0.0)
    
    def add_epoch_data(self,
                      epoch: int,
                      train_losses: List[float],
                      val_accuracy: float,
                      learning_rate: float,
                      epoch_time: float,
                      val_loss: Optional[float] = None,
                      row_sent_overall_accuracy: Optional[float] = None,
                      row_sent_avg_precision: Optional[float] = None):
        """
        Add data for a completed epoch.
        
        Args:
            epoch: Epoch number (1-indexed)
            train_losses: List of training losses for all batches in this epoch
            val_accuracy: Validation accuracy for this epoch
            learning_rate: Current learning rate
            epoch_time: Time taken for this epoch in seconds
            val_loss: Optional validation loss
            row_sent_overall_accuracy: Optional row-sentence overall accuracy
            row_sent_avg_precision: Optional row-sentence average precision
        """
        # Store epoch data
        self.epochs.append(epoch)
        
        # Compute training loss statistics
        train_losses_array = np.array(train_losses)
        self.train_loss_mean.append(np.mean(train_losses_array))
        self.train_loss_min.append(np.min(train_losses_array))
        self.train_loss_max.append(np.max(train_losses_array))
        self.train_loss_std.append(np.std(train_losses_array))
        
        # Store other metrics
        self.val_accuracy.append(val_accuracy)
        self.learning_rates.append(learning_rate)
        self.epoch_times.append(epoch_time)
        
        # Store batch losses if tracking
        if self.track_batch_losses:
            self.batch_losses.append(train_losses.copy())
        
        # Store validation loss if tracking is enabled
        if self.track_val_loss:
            # Always append something to maintain array length consistency
            self.val_loss.append(val_loss if val_loss is not None else 0.0)
        
        # Store row-sentence metrics if tracking is enabled
        if self.track_row_sent_metrics:
            # Always append something to maintain array length consistency
            self.row_sent_overall_accuracy.append(row_sent_overall_accuracy if row_sent_overall_accuracy is not None else 0.0)
            self.row_sent_avg_precision.append(row_sent_avg_precision if row_sent_avg_precision is not None else 0.0)
        
        # Update best metrics
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_epoch = epoch
        
        # Update best test metrics if tracking
        if self.track_row_sent_metrics:
            if row_sent_overall_accuracy is not None and row_sent_overall_accuracy > self.best_test_overall_accuracy:
                self.best_test_overall_accuracy = row_sent_overall_accuracy
                self.best_test_epoch = epoch
            if row_sent_avg_precision is not None and row_sent_avg_precision > self.best_test_avg_precision:
                self.best_test_avg_precision = row_sent_avg_precision
                self.best_test_precision_epoch = epoch
        
        # Auto-save and auto-plot if enabled
        if self.auto_save:
            self.save_data()
        
        if self.auto_plot:
            self.plot_curves()
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   🔥 Train Loss: {round_half_up(self.train_loss_mean[-1], 2):.2f} ± {round_half_up(self.train_loss_std[-1], 2):.2f}")
        print(f"   🎯 Val Accuracy: {round_half_up(val_accuracy, 2):.2f}")
        print(f"   🏆 Best Accuracy: {round_half_up(self.best_accuracy, 2):.2f} (Epoch {self.best_epoch})")
        print(f"   ⏱️  Epoch Time: {epoch_time:.1f}s")
        print(f"   📈 Learning Rate: {learning_rate:.2e}")
        
        if self.track_val_loss and val_loss is not None:
            print(f"   📉 Val Loss: {round_half_up(val_loss, 2):.2f}")
        
        if self.track_row_sent_metrics:
            if row_sent_overall_accuracy is not None:
                print(f"   🔍 Row-Sent F1: {round_half_up(row_sent_overall_accuracy, 2):.2f}")
                print(f"      🏆 Best Test F1: {round_half_up(self.best_test_overall_accuracy, 2):.2f} (Epoch {self.best_test_epoch})")
            if row_sent_avg_precision is not None:
                print(f"   🔍 Row-Sent Avg Precision: {round_half_up(row_sent_avg_precision, 2):.2f}")
                print(f"      🏆 Best Test Avg Precision: {round_half_up(self.best_test_avg_precision, 2):.2f} (Epoch {self.best_test_precision_epoch})")
    
    def add_initial_stage_metrics(self,
                                stage_0_accuracy: float = 0.0,
                                stage_1_accuracy: float = 0.0,
                                stage_2_accuracy: float = 0.0,
                                stage_0_row_sent_ap: float = 0.0,
                                stage_1_row_sent_ap: float = 0.0,
                                stage_2_row_sent_ap: float = 0.0,
                                stage_0_row_sent_acc: float = 0.0,
                                stage_1_row_sent_acc: float = 0.0,
                                stage_2_row_sent_acc: float = 0.0):
        """
        Add initial stage metrics (Stage 0, 1, 2 before training starts).
        
        Args:
            stage_0_accuracy: Stage 0 (frozen encoder) accuracy
            stage_1_accuracy: Stage 1 (sophisticated untrained) accuracy
            stage_2_accuracy: Stage 2 (trained model initial) accuracy
            stage_0_row_sent_ap: Stage 0 row-sentence average precision
            stage_1_row_sent_ap: Stage 1 row-sentence average precision
            stage_2_row_sent_ap: Stage 2 row-sentence average precision
            stage_0_row_sent_acc: Stage 0 row-sentence overall accuracy
            stage_1_row_sent_acc: Stage 1 row-sentence overall accuracy
            stage_2_row_sent_acc: Stage 2 row-sentence overall accuracy
        """
        self.initial_stage_metrics = {
            'stage_0_accuracy': stage_0_accuracy,
            'stage_1_accuracy': stage_1_accuracy,
            'stage_2_accuracy': stage_2_accuracy,
            'stage_0_row_sent_ap': stage_0_row_sent_ap,
            'stage_1_row_sent_ap': stage_1_row_sent_ap,
            'stage_2_row_sent_ap': stage_2_row_sent_ap,
            'stage_0_row_sent_f1': stage_0_row_sent_acc,
            'stage_1_row_sent_f1': stage_1_row_sent_acc,
            'stage_2_row_sent_f1': stage_2_row_sent_acc,
            'stage_0_row_sent_acc': stage_0_row_sent_acc,
            'stage_1_row_sent_acc': stage_1_row_sent_acc,
            'stage_2_row_sent_acc': stage_2_row_sent_acc
        }
        
        print(f"📊 Initial stage metrics recorded:")
        print(f"   🔥 Stage 0 (Frozen): {round_half_up(stage_0_accuracy, 2):.2f} acc, {round_half_up(stage_0_row_sent_ap, 2):.2f} row-sent AP, {round_half_up(stage_0_row_sent_acc, 2):.2f} row-sent F1")
        print(f"   🚀 Stage 1 (Sophisticated Untrained): {round_half_up(stage_1_accuracy, 2):.2f} acc, {round_half_up(stage_1_row_sent_ap, 2):.2f} row-sent AP, {round_half_up(stage_1_row_sent_acc, 2):.2f} row-sent F1")
        print(f"   🎯 Stage 2 (Initial): {round_half_up(stage_2_accuracy, 2):.2f} acc, {round_half_up(stage_2_row_sent_ap, 2):.2f} row-sent AP, {round_half_up(stage_2_row_sent_acc, 2):.2f} row-sent F1")
    
    def save_data(self, filename: Optional[str] = None):
        """
        Save all tracking data to JSON file.
        
        Args:
            filename: Optional custom filename. If None, uses run_name.
        """
        if filename is None:
            # Sanitize run_name to avoid directory separators in filename
            safe_run_name = self.run_name.replace('/', '_').replace('\\', '_')
            filename = f"{safe_run_name}_training_curves.json"
        
        filepath = self.data_dir / filename
        
        # Prepare data for JSON serialization
        data = {
            'metadata': {
                'run_name': self.run_name,
                'start_time': self.start_time.isoformat(),
                'total_epochs': len(self.epochs),
                'best_accuracy': self.best_accuracy,
                'best_epoch': self.best_epoch,
                'best_test_f1': self.best_test_overall_accuracy,
                'best_test_overall_accuracy': self.best_test_overall_accuracy,
                'best_test_epoch': self.best_test_epoch,
                'best_test_avg_precision': self.best_test_avg_precision,
                'best_test_precision_epoch': self.best_test_precision_epoch,
                'track_batch_losses': self.track_batch_losses,
                'track_val_loss': self.track_val_loss,
                'track_row_sent_metrics': self.track_row_sent_metrics,
                'initial_stage_metrics': self.initial_stage_metrics
            },
            'curves': {
                'epochs': self.epochs,
                'train_loss_mean': self.train_loss_mean,
                'train_loss_min': self.train_loss_min,
                'train_loss_max': self.train_loss_max,
                'train_loss_std': self.train_loss_std,
                'val_accuracy': self.val_accuracy,
                'learning_rates': self.learning_rates,
                'epoch_times': self.epoch_times
            }
        }
        
        # Add optional data
        if self.track_val_loss and self.val_loss:
            data['curves']['val_loss'] = self.val_loss
        
        if self.track_batch_losses and self.batch_losses:
            data['curves']['batch_losses'] = self.batch_losses
        
        if self.track_row_sent_metrics:
            if self.row_sent_overall_accuracy:
                data['curves']['row_sent_f1'] = self.row_sent_overall_accuracy
                data['curves']['row_sent_overall_accuracy'] = self.row_sent_overall_accuracy
            if self.row_sent_avg_precision:
                data['curves']['row_sent_avg_precision'] = self.row_sent_avg_precision
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_data(self, filename: Optional[str] = None):
        """
        Load tracking data from JSON file.
        
        Args:
            filename: Optional custom filename. If None, uses run_name.
        """
        if filename is None:
            # Sanitize run_name to avoid directory separators in filename
            safe_run_name = self.run_name.replace('/', '_').replace('\\', '_')
            filename = f"{safe_run_name}_training_curves.json"
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"⚠️ Training curves file not found: {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load metadata
        metadata = data['metadata']
        self.run_name = metadata['run_name']
        self.best_accuracy = metadata['best_accuracy']
        self.best_epoch = metadata['best_epoch']
        self.best_test_overall_accuracy = get_training_curves_best_row_sent_f1(metadata)
        self.best_test_epoch = metadata.get('best_test_epoch', 0)
        self.best_test_avg_precision = metadata.get('best_test_avg_precision', 0.0)
        self.best_test_precision_epoch = metadata.get('best_test_precision_epoch', 0)
        self.track_batch_losses = metadata['track_batch_losses']
        self.track_val_loss = metadata['track_val_loss']
        self.track_row_sent_metrics = metadata.get('track_row_sent_metrics', False)
        self.initial_stage_metrics = metadata.get('initial_stage_metrics', {})
        
        # Load curves data
        curves = data['curves']
        self.epochs = curves['epochs']
        self.train_loss_mean = curves['train_loss_mean']
        self.train_loss_min = curves['train_loss_min']
        self.train_loss_max = curves['train_loss_max']
        self.train_loss_std = curves['train_loss_std']
        self.val_accuracy = curves['val_accuracy']
        self.learning_rates = curves['learning_rates']
        self.epoch_times = curves['epoch_times']
        
        # Load optional data
        if 'val_loss' in curves:
            self.val_loss = curves['val_loss']
        
        if 'batch_losses' in curves:
            self.batch_losses = curves['batch_losses']
        
        row_sent_f1_curve = get_training_curves_row_sent_f1_curve(curves)
        if row_sent_f1_curve:
            self.row_sent_overall_accuracy = row_sent_f1_curve
        
        if 'row_sent_avg_precision' in curves:
            self.row_sent_avg_precision = curves['row_sent_avg_precision']
        
        print(f"📂 Training curves data loaded from {filepath}")
        return True
    
    def plot_curves(self, 
                   filename: Optional[str] = None,
                   figsize: Tuple[int, int] = (15, 10),
                   save_individual: bool = True):
        """
        Generate comprehensive training curves plots.
        
        Args:
            filename: Optional custom filename for the combined plot
            figsize: Figure size as (width, height)
            save_individual: Whether to save individual plots as well
        """
        if len(self.epochs) == 0:
            print("⚠️ No training data to plot")
            return
        
        # Create main figure with subplots (add extra row if tracking row-sentence metrics)
        if self.track_row_sent_metrics and self.row_sent_overall_accuracy and self.row_sent_avg_precision:
            fig, axes = plt.subplots(3, 3, figsize=(figsize[0], figsize[1] * 1.2))
        else:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Training Curves - {self.run_name}', fontsize=16, fontweight='bold')
        
        epochs_array = np.array(self.epochs)
        
        # Determine if Epoch 0 exists and create filtered arrays for training-related plots
        has_epoch_0 = len(epochs_array) > 0 and epochs_array[0] == 0
        
        if has_epoch_0:
            # Filter out Epoch 0 for training-related metrics
            training_epochs = epochs_array[1:]
            training_loss_mean = np.array(self.train_loss_mean[1:])
            training_loss_std = np.array(self.train_loss_std[1:])
            training_loss_min = np.array(self.train_loss_min[1:])
            training_loss_max = np.array(self.train_loss_max[1:])
            training_learning_rates = np.array(self.learning_rates[1:])
            training_epoch_times = np.array(self.epoch_times[1:])
        else:
            # No Epoch 0, use all data
            training_epochs = epochs_array
            training_loss_mean = np.array(self.train_loss_mean)
            training_loss_std = np.array(self.train_loss_std)
            training_loss_min = np.array(self.train_loss_min)
            training_loss_max = np.array(self.train_loss_max)
            training_learning_rates = np.array(self.learning_rates)
            training_epoch_times = np.array(self.epoch_times)
        
        # 1. Training Loss with Error Bars (exclude Epoch 0)
        ax1 = axes[0, 0]
        
        ax1.plot(training_epochs, training_loss_mean, 'b-', linewidth=2, label='Mean Loss')
        ax1.fill_between(training_epochs, 
                        training_loss_mean - training_loss_std,
                        training_loss_mean + training_loss_std,
                        alpha=0.3, color='blue', label='±1 Std')
        ax1.plot(training_epochs, training_loss_min, 'g--', alpha=0.7, label='Min Loss')
        ax1.plot(training_epochs, training_loss_max, 'r--', alpha=0.7, label='Max Loss')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Validation Accuracy (include Epoch 0)
        ax2 = axes[0, 1]
        ax2.plot(epochs_array, self.val_accuracy, 'g-o', linewidth=2, markersize=4)
        ax2.axhline(y=self.best_accuracy, color='r', linestyle='--', alpha=0.7, 
                   label=f'Best: {round_half_up(self.best_accuracy, 2):.2f} (Epoch {self.best_epoch})')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning Rate Schedule (exclude Epoch 0)
        ax3 = axes[0, 2]
        ax3.plot(training_epochs, training_learning_rates, 'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Training Time per Epoch (exclude Epoch 0)
        ax4 = axes[1, 0]
        ax4.bar(training_epochs, training_epoch_times, alpha=0.7, color='orange')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Training Time per Epoch')
        ax4.grid(True, alpha=0.3)
        
        # 5. Loss vs Accuracy Correlation (exclude Epoch 0)
        ax5 = axes[1, 1]
        # Use training arrays for this correlation plot
        correlation_val_accuracy = np.array(self.val_accuracy[1:]) if has_epoch_0 else np.array(self.val_accuracy)
        ax5.scatter(training_loss_mean, correlation_val_accuracy, c=training_epochs, 
                   cmap='viridis', s=50, alpha=0.7)
        ax5.set_xlabel('Training Loss')
        ax5.set_ylabel('Validation Accuracy')
        ax5.set_title('Loss vs Accuracy')
        ax5.grid(True, alpha=0.3)
        
        # Add colorbar for epoch progression
        cbar = plt.colorbar(ax5.collections[0], ax=ax5)
        cbar.set_label('Epoch')
        
        # 6. Training Progress Summary (text format - ORIGINAL BEHAVIOR)
        ax6 = axes[1, 2]
        # Split into two columns: Training Summary (left) and Test Summary (right)
        
        # === LEFT COLUMN: Training Summary ===
        ax6.text(0.05, 0.9, f'Training Summary', fontsize=12, fontweight='bold', 
                transform=ax6.transAxes)
        ax6.text(0.05, 0.8, f'Total Epochs: {len(self.epochs)}', transform=ax6.transAxes)
        ax6.text(0.05, 0.7, f'Best Accuracy: {round_half_up(self.best_accuracy, 2):.2f}', transform=ax6.transAxes)
        ax6.text(0.05, 0.6, f'Best Epoch: {self.best_epoch}', transform=ax6.transAxes)
        ax6.text(0.05, 0.5, f'Final Loss: {round_half_up(self.train_loss_mean[-1], 2):.2f}', transform=ax6.transAxes)
        ax6.text(0.05, 0.4, f'Final LR: {self.learning_rates[-1]:.2e}', transform=ax6.transAxes)
        
        # Add initial stage metrics if available
        if self.initial_stage_metrics:
            ax6.text(0.05, 0.25, f'Stage 0: {round_half_up(self.initial_stage_metrics.get("stage_0_accuracy", 0), 2):.2f}', 
                    transform=ax6.transAxes, fontsize=9)
            ax6.text(0.05, 0.20, f'Stage 1: {round_half_up(self.initial_stage_metrics.get("stage_1_accuracy", 0), 2):.2f}', 
                    transform=ax6.transAxes, fontsize=9)
            ax6.text(0.05, 0.15, f'Stage 2: {round_half_up(self.initial_stage_metrics.get("stage_2_accuracy", 0), 2):.2f}', 
                    transform=ax6.transAxes, fontsize=9)
            # Dynamic Final label if provided
            label_final = getattr(self, 'trained_stage_label', 'Final')
            ax6.text(0.05, 0.10, f'{label_final}: {round_half_up(self.best_accuracy, 2):.2f}', 
                    transform=ax6.transAxes, fontsize=9)
        
        # Calculate improvement metrics
        if len(self.val_accuracy) > 1:
            improvement = self.val_accuracy[-1] - self.val_accuracy[0]
            ax6.text(0.05, 0.3, f'Accuracy Δ: {round_half_up(improvement, 2):+.2f}', transform=ax6.transAxes)
            
            if len(self.train_loss_mean) > 1:
                loss_reduction = self.train_loss_mean[0] - self.train_loss_mean[-1]
                ax6.text(0.05, 0.05, f'Loss Δ: {round_half_up(loss_reduction, 2):.2f}', transform=ax6.transAxes, fontsize=9)
        
        # === RIGHT COLUMN: Test Summary ===
        if (self.track_row_sent_metrics and 
            self.row_sent_overall_accuracy and 
            self.row_sent_avg_precision and 
            len(self.row_sent_overall_accuracy) > 0):
            
            ax6.text(0.55, 0.9, f'Test Summary', fontsize=12, fontweight='bold', 
                    transform=ax6.transAxes)
            
            # Final test metrics
            final_overall_acc = self.row_sent_overall_accuracy[-1]
            ax6.text(0.55, 0.8, f'Final F1: {round_half_up(final_overall_acc, 2):.2f}', 
                    transform=ax6.transAxes)
                    
            final_avg_prec = self.row_sent_avg_precision[-1]
            ax6.text(0.55, 0.7, f'Final Avg Precision: {round_half_up(final_avg_prec, 2):.2f}', 
                    transform=ax6.transAxes)
            
            # Best test metrics
            ax6.text(0.55, 0.6, f'Best F1: {round_half_up(self.best_test_overall_accuracy, 2):.2f}', 
                    transform=ax6.transAxes, fontweight='bold')
            ax6.text(0.55, 0.55, f'  (Epoch {self.best_test_epoch})', 
                    transform=ax6.transAxes, fontsize=9)
            ax6.text(0.55, 0.5, f'Best Avg Precision: {round_half_up(self.best_test_avg_precision, 2):.2f}', 
                    transform=ax6.transAxes, fontweight='bold')
            ax6.text(0.55, 0.45, f'  (Epoch {self.best_test_precision_epoch})', 
                    transform=ax6.transAxes, fontsize=9)
            
            # Test improvement metrics
            if len(self.row_sent_overall_accuracy) > 1:
                test_acc_improvement = self.row_sent_overall_accuracy[-1] - self.row_sent_overall_accuracy[0]
                ax6.text(0.55, 0.35, f'F1 Δ: {round_half_up(test_acc_improvement, 2):+.2f}', 
                        transform=ax6.transAxes)
                
            if len(self.row_sent_avg_precision) > 1:
                test_prec_improvement = self.row_sent_avg_precision[-1] - self.row_sent_avg_precision[0]
                ax6.text(0.55, 0.3, f'Avg Precision Δ: {round_half_up(test_prec_improvement, 2):+.2f}', 
                        transform=ax6.transAxes)
            
            # Add initial stage row-sentence metrics if available
            if self.initial_stage_metrics:
                ax6.text(0.55, 0.2, f'Test Stage 0: {round_half_up(self.initial_stage_metrics.get("stage_0_row_sent_ap", 0), 2):.2f}', 
                        transform=ax6.transAxes, fontsize=9)
                ax6.text(0.55, 0.15, f'Test Stage 1: {round_half_up(self.initial_stage_metrics.get("stage_1_row_sent_ap", 0), 2):.2f}', 
                        transform=ax6.transAxes, fontsize=9)
                ax6.text(0.55, 0.1, f'Test Stage 2: {round_half_up(self.initial_stage_metrics.get("stage_2_row_sent_ap", 0), 2):.2f}', 
                        transform=ax6.transAxes, fontsize=9)
                label_stage_final = getattr(self, 'trained_stage_label', 'Final')
                ax6.text(0.55, 0.05, f'{label_stage_final}: {round_half_up(final_avg_prec, 2):.2f}', 
                        transform=ax6.transAxes, fontsize=9)
            
            # Add a vertical separator line between the two columns
            ax6.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        # 7. Row-Sentence Metrics (if tracked and available)
        if (self.track_row_sent_metrics and 
            self.row_sent_overall_accuracy and 
            self.row_sent_avg_precision and 
            len(self.row_sent_overall_accuracy) > 0):
            
            # Create filtered arrays for individual row-sentence plots (exclude Epoch 0)
            if has_epoch_0:
                row_sent_epochs = epochs_array[1:]
                row_sent_overall_acc = np.array(self.row_sent_overall_accuracy[1:])
                row_sent_avg_prec = np.array(self.row_sent_avg_precision[1:])
            else:
                row_sent_epochs = epochs_array
                row_sent_overall_acc = np.array(self.row_sent_overall_accuracy)
                row_sent_avg_prec = np.array(self.row_sent_avg_precision)
            
            # 7. Row-Sentence F1 (INCLUDE Epoch 0 for full picture)
            ax7 = axes[2, 0]
            # Use all epochs including epoch 0
            ax7.plot(epochs_array, self.row_sent_overall_accuracy, 'purple', 
                    linewidth=2, marker='o', markersize=4, label='F1')
            # Add frozen encoder baseline if available (Stage 0)
            if self.initial_stage_metrics and 'stage_0_row_sent_acc' in self.initial_stage_metrics:
                frozen_baseline = self.initial_stage_metrics['stage_0_row_sent_acc']
                if frozen_baseline > 0:
                    ax7.axhline(y=frozen_baseline, color='red', linestyle='--', alpha=0.7, 
                               label=f'Frozen Encoder: {round_half_up(frozen_baseline, 2):.2f}')
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('Row-Sentence F1')
            ax7.set_title('Row-Sentence F1')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
            
            # 8. Row-Sentence Average Precision (INCLUDE Epoch 0 for full picture)
            ax8 = axes[2, 1]
            # Use all epochs including epoch 0
            ax8.plot(epochs_array, self.row_sent_avg_precision, 'orange', 
                    linewidth=2, marker='s', markersize=4, label='Avg Precision')
            # Add frozen encoder baseline if available (Stage 0)
            if self.initial_stage_metrics and 'stage_0_row_sent_ap' in self.initial_stage_metrics:
                frozen_baseline_ap = self.initial_stage_metrics['stage_0_row_sent_ap']
                if frozen_baseline_ap > 0:
                    ax8.axhline(y=frozen_baseline_ap, color='red', linestyle='--', alpha=0.7, 
                               label=f'Frozen Encoder: {round_half_up(frozen_baseline_ap, 2):.2f}')
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Row-Sentence Avg Precision')
            ax8.set_title('Row-Sentence Average Precision')
            ax8.grid(True, alpha=0.3)
            ax8.legend()
            
            # 9. Row-Sentence Metrics Comparison (include Epoch 0)
            ax9 = axes[2, 2]
            ax9.plot(epochs_array, self.row_sent_overall_accuracy, 'purple', 
                    linewidth=2, marker='o', markersize=4, label='F1')
            ax9.plot(epochs_array, self.row_sent_avg_precision, 'orange', 
                    linewidth=2, marker='s', markersize=4, label='Avg Precision')
            # Add frozen encoder baselines if available (Stage 0)
            if self.initial_stage_metrics:
                if 'stage_0_row_sent_acc' in self.initial_stage_metrics:
                    frozen_acc = self.initial_stage_metrics['stage_0_row_sent_acc']
                    if frozen_acc > 0:
                        ax9.axhline(y=frozen_acc, color='purple', linestyle='--', alpha=0.5,
                                   label=f'Frozen Acc: {round_half_up(frozen_acc, 2):.2f}')
                if 'stage_0_row_sent_ap' in self.initial_stage_metrics:
                    frozen_ap = self.initial_stage_metrics['stage_0_row_sent_ap']
                    if frozen_ap > 0:
                        ax9.axhline(y=frozen_ap, color='orange', linestyle='--', alpha=0.5,
                                   label=f'Frozen AP: {round_half_up(frozen_ap, 2):.2f}')
            ax9.set_xlabel('Epoch')
            ax9.set_ylabel('Row-Sentence Metrics')
            ax9.set_title('Row-Sentence Metrics Comparison')
            ax9.grid(True, alpha=0.3)
            ax9.legend()
        
        plt.tight_layout()
        
        # Save combined plot
        if filename is None:
            # Sanitize run_name to avoid directory separators in filename
            safe_run_name = self.run_name.replace('/', '_').replace('\\', '_')
            filename = f"{safe_run_name}_training_curves.png"
        else:
            # Sanitize explicitly provided filename to avoid directory separators
            filename = filename.replace('/', '_').replace('\\', '_')
        
        combined_path = self.plots_dir / filename
        # Ensure the directory exists before saving
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        save_plot_multi_format(str(combined_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save individual plots if requested
        if save_individual:
            self._save_individual_plots()
    
    def _save_individual_plots(self):
        """Save individual plots for detailed analysis."""
        epochs_array = np.array(self.epochs)
        
        # 1. Training Loss Only
        plt.figure(figsize=(10, 6))
        train_loss_mean_array = np.array(self.train_loss_mean)
        train_loss_std_array = np.array(self.train_loss_std)
        
        plt.plot(epochs_array, train_loss_mean_array, 'b-', linewidth=2, label='Mean Loss')
        plt.fill_between(epochs_array, 
                        train_loss_mean_array - train_loss_std_array,
                        train_loss_mean_array + train_loss_std_array,
                        alpha=0.3, color='blue', label='±1 Std')
        plt.plot(epochs_array, self.train_loss_min, 'g--', alpha=0.7, label='Min Loss')
        plt.plot(epochs_array, self.train_loss_max, 'r--', alpha=0.7, label='Max Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f'Training Loss - {self.run_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sanitize run_name to avoid directory separators in filename
        safe_run_name = self.run_name.replace('/', '_').replace('\\', '_')
        loss_path = self.plots_dir / f"{safe_run_name}_training_loss.png"
        save_plot_multi_format(str(loss_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Validation Accuracy Only
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_array, self.val_accuracy, 'g-o', linewidth=2, markersize=6)
        plt.axhline(y=self.best_accuracy, color='r', linestyle='--', alpha=0.7, 
                   label=f'Best: {round_half_up(self.best_accuracy, 2):.2f} (Epoch {self.best_epoch})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title(f'Validation Accuracy - {self.run_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sanitize run_name to avoid directory separators in filename
        safe_run_name = self.run_name.replace('/', '_').replace('\\', '_')
        acc_path = self.plots_dir / f"{safe_run_name}_validation_accuracy.png"
        save_plot_multi_format(str(acc_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_batch_losses(self, epoch: Optional[int] = None):
        """
        Plot detailed batch-level losses for analysis.
        
        Args:
            epoch: Specific epoch to plot. If None, plots all epochs.
        """
        if not self.track_batch_losses or not self.batch_losses:
            print("⚠️ Batch losses not tracked. Enable track_batch_losses=True")
            return
        
        if epoch is not None:
            # Plot specific epoch
            if epoch < 1 or epoch > len(self.batch_losses):
                print(f"⚠️ Epoch {epoch} not found. Available: 1-{len(self.batch_losses)}")
                return
            
            plt.figure(figsize=(12, 6))
            batch_losses = self.batch_losses[epoch - 1]
            batch_indices = range(1, len(batch_losses) + 1)
            
            plt.plot(batch_indices, batch_losses, 'b-', linewidth=1, alpha=0.7)
            plt.scatter(batch_indices, batch_losses, c=batch_losses, cmap='viridis', s=20)
            
            # Add moving average
            if len(batch_losses) > 10:
                window = min(len(batch_losses) // 10, 20)
                moving_avg = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
                moving_indices = range(window, len(batch_losses) + 1)
                plt.plot(moving_indices, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window} batches)')
                plt.legend()
            
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title(f'Batch Losses - Epoch {epoch} - {self.run_name}')
            plt.grid(True, alpha=0.3)
            plt.colorbar(label='Loss Value')
            
            # Sanitize run_name to avoid directory separators in filename
            safe_run_name = self.run_name.replace('/', '_').replace('\\', '_')
            batch_path = self.plots_dir / f"{safe_run_name}_batch_losses_epoch_{epoch}.png"
            save_plot_multi_format(str(batch_path), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Batch losses plot for epoch {epoch} saved to {batch_path}")
        
        else:
            # Plot all epochs in a heatmap
            max_batches = max(len(losses) for losses in self.batch_losses)
            
            # Create matrix with NaN for missing values
            loss_matrix = np.full((len(self.batch_losses), max_batches), np.nan)
            
            for i, losses in enumerate(self.batch_losses):
                loss_matrix[i, :len(losses)] = losses
            
            plt.figure(figsize=(15, 8))
            
            # Create heatmap
            im = plt.imshow(loss_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
            plt.colorbar(im, label='Loss Value')
            
            plt.xlabel('Batch')
            plt.ylabel('Epoch')
            plt.title(f'Batch Losses Heatmap - {self.run_name}')
            
            # Set ticks
            plt.xticks(range(0, max_batches, max(1, max_batches // 10)))
            plt.yticks(range(len(self.batch_losses)))
            
            # Sanitize run_name to avoid directory separators in filename
            safe_run_name = self.run_name.replace('/', '_').replace('\\', '_')
            heatmap_path = self.plots_dir / f"{safe_run_name}_batch_losses_heatmap.png"
            save_plot_multi_format(str(heatmap_path), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🌡️ Batch losses heatmap saved to {heatmap_path}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics of the training run.
        
        Returns:
            Dictionary with summary statistics
        """
        if len(self.epochs) == 0:
            return {}
        
        # Check if epoch 0 exists (pre-trained baseline)
        has_epoch_0 = len(self.epochs) > 0 and self.epochs[0] == 0
        
        # Exclude epoch 0 from total training epochs count (it's just the pre-trained model)
        actual_training_epochs = len(self.epochs) - 1 if has_epoch_0 else len(self.epochs)
        
        # Initial loss should be from epoch 1 (first actual training), not epoch 0 (which is 0.0)
        if has_epoch_0 and len(self.train_loss_mean) > 1:
            initial_loss = self.train_loss_mean[1]  # Epoch 1's loss
        else:
            initial_loss = self.train_loss_mean[0]
        
        stats = {
            'total_epochs': actual_training_epochs,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'final_accuracy': self.val_accuracy[-1],
            'final_loss': self.train_loss_mean[-1],
            'initial_loss': initial_loss,
            'loss_reduction': initial_loss - self.train_loss_mean[-1],
            'accuracy_improvement': self.val_accuracy[-1] - self.val_accuracy[0] if len(self.val_accuracy) > 1 else 0.0,
            'average_epoch_time': np.mean(self.epoch_times[1:]) if has_epoch_0 else np.mean(self.epoch_times),  # Exclude epoch 0 time
            'total_training_time': sum(self.epoch_times[1:]) if has_epoch_0 else sum(self.epoch_times),  # Exclude epoch 0 time
            'loss_statistics': {
                'mean_loss_trend': np.mean(self.train_loss_mean[1:]) if has_epoch_0 else np.mean(self.train_loss_mean),
                'loss_variance': np.var(self.train_loss_mean[1:]) if has_epoch_0 else np.var(self.train_loss_mean),
                'min_loss_achieved': np.min(self.train_loss_mean[1:]) if has_epoch_0 else np.min(self.train_loss_mean),
                'max_loss_encountered': np.max(self.train_loss_mean[1:]) if has_epoch_0 else np.max(self.train_loss_mean)
            },
            'best_test_metrics': {
                'best_test_f1': self.best_test_overall_accuracy,
                'best_test_overall_accuracy': self.best_test_overall_accuracy,
                'best_test_epoch': self.best_test_epoch,
                'best_test_avg_precision': self.best_test_avg_precision,
                'best_test_precision_epoch': self.best_test_precision_epoch
            },
            'initial_stage_metrics': self.initial_stage_metrics
        }
        
        return stats
    
    def print_summary(self):
        """Print a comprehensive summary of the training run."""
        stats = self.get_summary_stats()
        
        if not stats:
            print("⚠️ No training data available for summary")
            return
        
        print("\n" + "="*70)
        print(f"🎯 TRAINING SUMMARY - {self.run_name}")
        print("="*70)
        print(f"📊 Total Epochs: {stats['total_epochs']}")
        print(f"🏆 Best Accuracy: {round_half_up(stats['best_accuracy'], 2):.2f} (Epoch {stats['best_epoch']})")
        print(f"📈 Final Accuracy: {round_half_up(stats['final_accuracy'], 2):.2f}")
        print(f"📉 Accuracy Improvement: {round_half_up(stats['accuracy_improvement'], 2):+.2f}")
        print(f"🔥 Initial Loss: {round_half_up(stats['initial_loss'], 2):.2f}")
        print(f"🎯 Final Loss: {round_half_up(stats['final_loss'], 2):.2f}")
        print(f"📉 Loss Reduction: {round_half_up(stats['loss_reduction'], 2):.2f}")
        print(f"⏱️  Average Epoch Time: {stats['average_epoch_time']:.1f}s")
        print(f"🕐 Total Training Time: {stats['total_training_time']:.1f}s ({stats['total_training_time']/60:.1f} min)")
        
        # Print best test metrics if available
        if stats['best_test_metrics']:
            print(f"🏆 Best Test Metrics:")
            test_metrics = stats['best_test_metrics']
            print(f"   🔍 Best F1: {round_half_up(test_metrics['best_test_overall_accuracy'], 2):.2f} (Epoch {test_metrics['best_test_epoch']})")
            print(f"   🔍 Best Average Precision: {round_half_up(test_metrics['best_test_avg_precision'], 2):.2f} (Epoch {test_metrics['best_test_precision_epoch']})")
        
        # Print initial stage metrics if available
        if stats['initial_stage_metrics']:
            print(f"📊 Initial Stage Metrics:")
            stage_metrics = stats['initial_stage_metrics']
            print(f"   🔥 Stage 0 (Frozen): {round_half_up(stage_metrics.get('stage_0_accuracy', 0), 2):.2f} acc, {round_half_up(stage_metrics.get('stage_0_row_sent_ap', 0), 2):.2f} row-sent AP, {round_half_up(get_training_curves_stage_row_sent_f1(stage_metrics, 'stage_0'), 2):.2f} row-sent F1")
            print(f"   🚀 Stage 1 (Sophisticated Untrained): {round_half_up(stage_metrics.get('stage_1_accuracy', 0), 2):.2f} acc, {round_half_up(stage_metrics.get('stage_1_row_sent_ap', 0), 2):.2f} row-sent AP, {round_half_up(get_training_curves_stage_row_sent_f1(stage_metrics, 'stage_1'), 2):.2f} row-sent F1")
            print(f"   🎯 Stage 2 (Initial): {round_half_up(stage_metrics.get('stage_2_accuracy', 0), 2):.2f} acc, {round_half_up(stage_metrics.get('stage_2_row_sent_ap', 0), 2):.2f} row-sent AP, {round_half_up(get_training_curves_stage_row_sent_f1(stage_metrics, 'stage_2'), 2):.2f} row-sent F1")
        
        print("="*70)


def create_training_curves_tracker(output_dir: str, 
                                 run_name: str,
                                 **kwargs) -> TrainingCurves:
    """
    Convenience function to create a TrainingCurves tracker.
    
    Args:
        output_dir: Directory to save curves data and plots
        run_name: Name of the training run
        **kwargs: Additional arguments for TrainingCurves
    
    Returns:
        Initialized TrainingCurves instance
    """
    return TrainingCurves(output_dir, run_name, **kwargs)


def plot_multiple_runs(curves_files: List[str], 
                      output_path: str,
                      run_names: Optional[List[str]] = None,
                      figsize: Tuple[int, int] = (15, 10)):
    """
    Plot multiple training runs for comparison.
    
    Args:
        curves_files: List of paths to training curves JSON files
        output_path: Path to save the comparison plot
        run_names: Optional list of run names for legend
        figsize: Figure size as (width, height)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training Runs Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves_files)))
    
    for i, curves_file in enumerate(curves_files):
        # Load data
        with open(curves_file, 'r') as f:
            data = json.load(f)
        
        curves = data['curves']
        run_name = run_names[i] if run_names else data['metadata']['run_name']
        color = colors[i]
        
        epochs = np.array(curves['epochs'])
        train_loss = np.array(curves['train_loss_mean'])
        val_accuracy = np.array(curves['val_accuracy'])
        learning_rates = np.array(curves['learning_rates'])
        
        # Plot training loss
        axes[0, 0].plot(epochs, train_loss, color=color, linewidth=2, label=run_name)
        
        # Plot validation accuracy
        axes[0, 1].plot(epochs, val_accuracy, color=color, linewidth=2, label=run_name)
        
        # Plot learning rate
        axes[1, 0].plot(epochs, learning_rates, color=color, linewidth=2, label=run_name)
        
        # Plot loss vs accuracy scatter
        axes[1, 1].scatter(train_loss, val_accuracy, color=color, alpha=0.6, label=run_name)
    
    # Customize plots
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Accuracy')
    axes[0, 1].set_title('Validation Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Comparison')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Training Loss')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Loss vs Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot_multi_format(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 Multiple runs comparison plot saved to {output_path}")


# =============================================================================
# Multi-Model Comparison Tools
# =============================================================================

UNI_R_TO_S = "Uni (R→S)"
UNI_S_TO_R = "Uni (S→R)"

MODEL_NAME_ALIASES = {
    UNI_R_TO_S: UNI_R_TO_S,
    "Uni (R-S)": UNI_R_TO_S,
    "Uni (R→S)": UNI_R_TO_S,
    "Uni-cross": UNI_R_TO_S,
    UNI_S_TO_R: UNI_S_TO_R,
    "Uni (S-R)": UNI_S_TO_R,
    "Uni (S→R)": UNI_S_TO_R,
}

MODEL_DIRECTORY_ALIASES = {
    "LOKI": ("LOKI",),
    "FT-Encoder": ("FT-Encoder",),
    UNI_R_TO_S: (UNI_R_TO_S, "Uni (R-S)", "Uni (R→S)", "Uni-cross"),
    UNI_S_TO_R: (UNI_S_TO_R, "Uni (S-R)", "Uni (S→R)"),
}

TRAINED_MODEL_ORDER = ["FT-Encoder", UNI_R_TO_S, UNI_S_TO_R, "LOKI"]
TRAINED_MODEL_DISPLAY_ORDER = ["LOKI", "FT-Encoder", UNI_R_TO_S, UNI_S_TO_R]
COMPARISON_MODEL_ORDER = ["Baseline", *TRAINED_MODEL_ORDER]

# Color scheme for model comparison plots
# Red is reserved for baseline; model colors match post-training comparison plots.
MODEL_COLORS = {
    "Baseline": "#D62728",
    "LOKI": "#1F77B4",
    "FT-Encoder": "#7B2D8E",
    UNI_R_TO_S: "#E67E22",
    UNI_S_TO_R: "#2CA02C",
    "Model-4": "#17BECF",  # Reserved for future extra model
}

MODEL_LINE_STYLES = {
    "LOKI": "-",
    "FT-Encoder": "--",
    UNI_R_TO_S: "-.",
    UNI_S_TO_R: ":",
}

MODEL_MARKERS = {
    "LOKI": "*",
    "FT-Encoder": "D",
    UNI_R_TO_S: "s",
    UNI_S_TO_R: "^",
}

MODEL_HATCHES = {
    "Baseline": "",
    "LOKI": "",
    "FT-Encoder": "//",
    UNI_R_TO_S: "\\\\",
    UNI_S_TO_R: "xx",
}

MODEL_SURFACE_COLORS = {
    "Baseline": {"surface": "#E74C3C", "edge": "#C0392B"},
    "LOKI": {"surface": "#2E86AB", "edge": "#1A5276"},
    "FT-Encoder": {"surface": "#9B59B6", "edge": "#7D3C98"},
    UNI_R_TO_S: {"surface": "#E67E22", "edge": "#D35400"},
    UNI_S_TO_R: {"surface": "#27AE60", "edge": "#1E8449"},
}

# Additional colors for models beyond the predefined ones
EXTRA_COLORS = ['brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']


def canonicalize_model_name(model_name: str) -> str:
    """Map legacy filesystem or display labels to canonical display labels."""
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def get_model_aliases(model_name: str) -> Tuple[str, ...]:
    """Return accepted directory/display aliases for a model."""
    canonical = canonicalize_model_name(model_name)
    return MODEL_DIRECTORY_ALIASES.get(canonical, (canonical,))


def get_present_models(models: Dict[str, Any], include_baseline: bool = False) -> List[str]:
    """Return models in a stable display order."""
    ordered = COMPARISON_MODEL_ORDER if include_baseline else TRAINED_MODEL_ORDER
    present = [name for name in ordered if (name == "Baseline" and include_baseline) or name in models]
    extras = [name for name in models.keys() if name not in ordered]
    return present + extras


def get_present_models_for_panels(models: Dict[str, Any]) -> List[str]:
    """Return models in the preferred display order for per-model panels."""
    ordered = [name for name in TRAINED_MODEL_DISPLAY_ORDER if name in models]
    extras = [name for name in models.keys() if name not in ordered]
    return ordered + extras


def normalize_model_mapping(models: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model dict keys and preserve a stable display order."""
    normalized: Dict[str, Any] = {}
    for name, payload in models.items():
        normalized[canonicalize_model_name(name)] = payload
    ordered_names = get_present_models(normalized)
    return {name: normalized[name] for name in ordered_names}


def get_training_curves_best_row_sent_f1(metadata: Dict[str, Any]) -> float:
    """Read row-sentence F1 from either new or legacy training-curves metadata."""
    return metadata.get("best_test_f1", metadata.get("best_test_overall_accuracy", 0.0))


def get_training_curves_row_sent_f1_curve(curves: Dict[str, Any]) -> List[Any]:
    """Read row-sentence F1 curve from either new or legacy training-curves payloads."""
    return curves.get("row_sent_f1", curves.get("row_sent_overall_accuracy", []))


def get_training_curves_stage_row_sent_f1(stage_metrics: Dict[str, Any], stage_prefix: str) -> float:
    """Read stage-level row-sentence F1 from either new or legacy key names."""
    return stage_metrics.get(f"{stage_prefix}_row_sent_f1", stage_metrics.get(f"{stage_prefix}_row_sent_acc", 0.0))


def normalize_combined_model_metrics(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize row-sentence F1 keys inside one combined model payload."""
    normalized = dict(model_data)
    curves = dict(normalized.get("curves", {}))
    best_test_f1 = normalized.get("best_test_f1", normalized.get("best_test_overall_accuracy", 0.0))
    normalized["best_test_f1"] = best_test_f1
    normalized["best_test_overall_accuracy"] = best_test_f1  # legacy alias
    row_sent_f1 = curves.get("row_sent_f1", curves.get("row_sent_overall_accuracy", []))
    curves["row_sent_f1"] = row_sent_f1
    curves["row_sent_overall_accuracy"] = row_sent_f1  # legacy alias
    normalized["curves"] = curves
    return normalized


def normalize_combined_baseline_metrics(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize row-sentence F1 baseline keys inside combined comparison data."""
    normalized = dict(baseline)
    row_sent_f1 = normalized.get("frozen_encoder_row_sent_f1", normalized.get("frozen_encoder_row_sent_acc", 0.0))
    normalized["frozen_encoder_row_sent_f1"] = row_sent_f1
    normalized["frozen_encoder_row_sent_acc"] = row_sent_f1  # legacy alias
    return normalized


def normalize_combined_data(combined_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model names inside combined comparison data."""
    normalized = dict(combined_data)
    metadata = dict(normalized.get("metadata", {}))
    metadata["baseline_model"] = canonicalize_model_name(metadata.get("baseline_model", "LOKI"))
    metadata["model_names"] = [
        canonicalize_model_name(name) for name in metadata.get("model_names", [])
    ]
    normalized["metadata"] = metadata
    normalized["baseline"] = normalize_combined_baseline_metrics(normalized.get("baseline", {}))
    normalized_models = normalize_model_mapping(normalized.get("models", {}))
    normalized["models"] = {
        name: normalize_combined_model_metrics(payload)
        for name, payload in normalized_models.items()
    }
    return normalized


def _alias_preference_index(model_name: str, alias_name: str) -> int:
    aliases = get_model_aliases(model_name)
    try:
        return aliases.index(alias_name)
    except ValueError:
        return len(aliases)


def find_training_curves_json(base_dir: str, model_name: str) -> Optional[Path]:
    """Resolve a training curves JSON for a model, allowing nested run folders."""
    base_path = Path(base_dir)
    for alias in get_model_aliases(model_name):
        alias_dir = base_path / alias
        if not alias_dir.exists():
            continue
        direct_matches = sorted(alias_dir.glob("training_data/*_training_curves.json"))
        if direct_matches:
            return direct_matches[0]
        nested_matches = sorted(alias_dir.glob("**/training_data/*_training_curves.json"))
        if nested_matches:
            return nested_matches[0]
    return None


def find_post_eval_results_json(model_name: str, results_dir: str = "Post_Training_Results") -> Optional[Path]:
    """Resolve a post-training evaluation JSON for a model across legacy/new folders."""
    results_base = Path(results_dir)
    for alias in get_model_aliases(model_name):
        candidate = results_base / alias / "results_post_training_eval.json"
        if candidate.exists():
            return candidate
    return None


def get_model_color(model_name: str, model_index: int = 0) -> str:
    """
    Get consistent color for a model.
    
    Args:
        model_name: Name of the model
        model_index: Index in the list of models (used for fallback colors)
    
    Returns:
        Color string for matplotlib
    """
    model_name = canonicalize_model_name(model_name)
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    # Use extra colors for additional models
    extra_idx = model_index % len(EXTRA_COLORS)
    return EXTRA_COLORS[extra_idx]


def discover_model_directories(base_dir: str = "Input_Models") -> List[Tuple[str, str]]:
    """
    Auto-discover model directories containing training curves JSON files.
    
    Args:
        base_dir: Base directory to search (default: Input_Models)
    
    Returns:
        List of tuples (model_name, json_file_path)
    """
    base_path = Path(base_dir)
    discovered: Dict[str, Tuple[Path, int]] = {}
    
    if not base_path.exists():
        print(f"⚠️ Base directory not found: {base_dir}")
        return discovered
    
    for model_dir in base_path.iterdir():
        if model_dir.is_dir() and model_dir.name not in ['__pycache__', '.git']:
            canonical_name = canonicalize_model_name(model_dir.name)
            direct_matches = sorted(model_dir.glob("training_data/*_training_curves.json"))
            nested_matches = sorted(model_dir.glob("**/training_data/*_training_curves.json"))
            json_files = direct_matches or nested_matches
            if not json_files:
                continue

            selected_json = json_files[0]
            current = discovered.get(canonical_name)
            current_rank = _alias_preference_index(canonical_name, model_dir.name)
            if current is None or current_rank < current[1]:
                discovered[canonical_name] = (selected_json, current_rank)
                print(f"   ✅ Found: {canonical_name} -> {selected_json}")

    return [(model_name, str(info[0])) for model_name, info in discovered.items()]


def load_model_training_data(json_path: str) -> Dict[str, Any]:
    """
    Load training curves data from a JSON file.
    
    Args:
        json_path: Path to the training_curves.json file
    
    Returns:
        Dictionary containing metadata and curves data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_combined_comparison_json(
    model_dirs: Optional[List[Tuple[str, str]]] = None,
    base_dir: str = "Input_Models",
    output_path: str = "output_plots/combined_comparison_data.json",
    baseline_model: str = "LOKI"
) -> Dict[str, Any]:
    """
    Create a combined JSON file containing data from multiple models for comparison.
    
    Args:
        model_dirs: List of (model_name, json_path) tuples. If None, auto-discover.
        base_dir: Base directory for auto-discovery
        output_path: Path to save the combined JSON
        baseline_model: Model to use for frozen encoder baseline values (default: LOKI)
    
    Returns:
        Combined data dictionary
    """
    baseline_model = canonicalize_model_name(baseline_model)

    print("\n" + "="*70)
    print("📊 Creating Combined Comparison Data")
    print("="*70)
    
    # Auto-discover if not provided
    if model_dirs is None:
        print(f"🔍 Auto-discovering models in: {base_dir}")
        model_dirs = discover_model_directories(base_dir)
    
    if not model_dirs:
        print("❌ No model directories found!")
        return {}
    
    model_dirs = [(canonicalize_model_name(name), path) for name, path in model_dirs]
    print(f"\n📁 Loading data from {len(model_dirs)} models...")
    
    combined_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "baseline_model": baseline_model,
            "model_count": len(model_dirs),
            "model_names": [m[0] for m in model_dirs],
        },
        "dataset_statistics": {},
        "baseline": {
            "frozen_encoder_accuracy": 0.0,
            "frozen_encoder_row_sent_acc": 0.0,
            "frozen_encoder_row_sent_ap": 0.0,
        },
        "models": {}
    }
    
    # Load each model's data
    for model_name, json_path in model_dirs:
        print(f"\n   📖 Loading {model_name}...")
        try:
            data = load_model_training_data(json_path)
            metadata = data['metadata']
            curves = data['curves']
            
            # Extract key metrics
            model_data = {
                "source_file": json_path,
                "run_name": metadata.get('run_name', model_name),
                "total_epochs": metadata.get('total_epochs', len(curves.get('epochs', []))),
                "best_accuracy": metadata.get('best_accuracy', 0.0),
                "best_epoch": metadata.get('best_epoch', 0),
                "best_test_f1": get_training_curves_best_row_sent_f1(metadata),
                "best_test_overall_accuracy": get_training_curves_best_row_sent_f1(metadata),
                "best_test_epoch": metadata.get('best_test_epoch', 0),
                "best_test_avg_precision": metadata.get('best_test_avg_precision', 0.0),
                "best_test_precision_epoch": metadata.get('best_test_precision_epoch', 0),
                "initial_stage_metrics": metadata.get('initial_stage_metrics', {}),
                "curves": {
                    "epochs": curves.get('epochs', []),
                    "train_loss_mean": curves.get('train_loss_mean', []),
                    "train_loss_std": curves.get('train_loss_std', []),
                    "train_loss_min": curves.get('train_loss_min', []),
                    "train_loss_max": curves.get('train_loss_max', []),
                    "val_accuracy": curves.get('val_accuracy', []),
                    "learning_rates": curves.get('learning_rates', []),
                    "epoch_times": curves.get('epoch_times', []),
                    "row_sent_f1": get_training_curves_row_sent_f1_curve(curves),
                    "row_sent_overall_accuracy": get_training_curves_row_sent_f1_curve(curves),
                    "row_sent_avg_precision": curves.get('row_sent_avg_precision', []),
                }
            }

            # Enrich with post-training evaluation raw counts when available.
            post_eval_path = find_post_eval_results_json(model_name)
            if post_eval_path and post_eval_path.exists():
                try:
                    with open(post_eval_path, "r", encoding="utf-8") as pf:
                        post_eval = json.load(pf)
                    evaluations = post_eval.get("evaluations", {})
                    stage3 = (
                        evaluations.get("stage_3_best_test_avg_precision")
                        or evaluations.get("stage_3_best_test_overall_acc")
                        or evaluations.get("stage_3_best")
                        or {}
                    )
                    model_data["post_training_eval"] = {
                        "source_file": str(post_eval_path),
                        "ranking_raw_counts": stage3.get("ranking_raw_counts", {}),
                        "prediction_breakdown": stage3.get("prediction_breakdown", {}),
                        "diagnosis_prediction_breakdown": stage3.get("diagnosis_prediction_breakdown", {}),
                        "medication_prediction_breakdown": stage3.get("medication_prediction_breakdown", {}),
                        "examples_evaluated": stage3.get("examples_evaluated", 0),
                    }

                    # Include split-level dataset statistics once (prefer baseline model, then first available).
                    ds_stats = post_eval.get("dataset_statistics", {})
                    if ds_stats and (
                        not combined_data["dataset_statistics"] or model_name == baseline_model
                    ):
                        combined_data["dataset_statistics"] = ds_stats
                except Exception as e:
                    print(f"      ⚠️ Could not enrich {model_name} with post-training raw counts: {e}")
            
            combined_data["models"][model_name] = model_data
            
            # Extract baseline from the specified baseline model (LOKI)
            if model_name == baseline_model:
                stage_metrics = metadata.get('initial_stage_metrics', {})
                combined_data["baseline"]["frozen_encoder_accuracy"] = stage_metrics.get('stage_0_accuracy', 0.0)
                combined_data["baseline"]["frozen_encoder_row_sent_f1"] = get_training_curves_stage_row_sent_f1(stage_metrics, 'stage_0')
                combined_data["baseline"]["frozen_encoder_row_sent_acc"] = get_training_curves_stage_row_sent_f1(stage_metrics, 'stage_0')
                combined_data["baseline"]["frozen_encoder_row_sent_ap"] = stage_metrics.get('stage_0_row_sent_ap', 0.0)
                print(f"      ✅ Using {model_name} for frozen encoder baseline values")
            
            print(f"      ✅ Loaded: {model_data['total_epochs']} epochs, "
                  f"best val acc: {round_half_up(model_data['best_accuracy'], 2):.2f}")
            
        except Exception as e:
            print(f"      ❌ Failed to load {model_name}: {e}")
            continue
    
    combined_data = normalize_combined_data(combined_data)
    combined_data["metadata"]["model_count"] = len(combined_data["models"])
    combined_data["metadata"]["model_names"] = list(combined_data["models"].keys())

    # Save combined JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\n✅ Combined comparison data saved to: {output_path}")
    print(f"   📊 Baseline (Frozen Encoder from {baseline_model}):")
    print(f"      - Validation Accuracy: {round_half_up(combined_data['baseline']['frozen_encoder_accuracy'], 2):.2f}")
    print(f"      - Row-Sent F1: {round_half_up(combined_data['baseline']['frozen_encoder_row_sent_acc'], 2):.2f}")
    print(f"      - Row-Sent Avg Precision: {round_half_up(combined_data['baseline']['frozen_encoder_row_sent_ap'], 2):.2f}")
    
    return combined_data


def plot_test_metrics_comparison(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[int, int] = (18, 5.5)
) -> None:
    """
    Plot test metrics comparison: Validation Accuracy, Row-Sentence F1, 
    and Row-Sentence Average Precision for multiple models.
    
    Conference-quality visualization (ICLR/NeurIPS/VLDB style):
    - Clean lines with distinct styles for B&W printing compatibility
    - Star markers for best points
    - Improvement annotations relative to baseline
    - No misleading area fills
    
    Args:
        combined_data: Combined comparison data dict (if already loaded)
        combined_json_path: Path to combined JSON file (loads if combined_data is None)
        output_dir: Directory to save the plots
        figsize: Figure size as (width, height)
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        print(f"📖 Loading combined data from: {combined_json_path}")
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    model_names = get_present_models(models)
    
    print(f"\n🎨 Generating Test Metrics Comparison Plot for {len(model_names)} models...")
    
    # Conference-style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    
    # Line styles for B&W compatibility (solid, dashed, dotted, dashdot)
    LINE_STYLES = ['-', '--', '-.', ':']
    # Markers for each model - LOKI always gets star (it's the star of the show!)
    # Other models get these markers in order
    DEFAULT_MARKERS = ['D', 's', '^', 'v', 'o', 'p', 'h']
    DEFAULT_MARKER_SIZES = [180, 180, 180, 180, 180, 180, 180]
    LOKI_MARKER = '*'
    LOKI_MARKER_SIZE = 550  # Star needs to be much larger to visually match solid shapes
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    for ax in axes:
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='major', labelsize=11, direction='in', length=4)
        ax.tick_params(axis='both', which='minor', direction='in', length=2)
    
    # Store data for improvement annotations
    improvements_acc = {}
    improvements_ap = {}
    loki_best_values = {}  # Store LOKI's best values for vertical arrows
    
    # Collect all best values for smart annotation positioning
    all_best_val_acc = {}
    for name in model_names:
        all_best_val_acc[name] = models[name]["best_accuracy"]
    
    # Plot each model
    non_loki_idx = 0  # Track index for non-LOKI models
    for idx, model_name in enumerate(model_names):
        model_data = models[model_name]
        curves = model_data["curves"]
        color = get_model_color(model_name, idx)
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        
        # LOKI always gets the star marker!
        if model_name == "LOKI":
            marker = LOKI_MARKER
            marker_size = LOKI_MARKER_SIZE
        else:
            marker = DEFAULT_MARKERS[non_loki_idx % len(DEFAULT_MARKERS)]
            marker_size = DEFAULT_MARKER_SIZES[non_loki_idx % len(DEFAULT_MARKER_SIZES)]
            non_loki_idx += 1
        
        epochs = np.array(curves["epochs"])
        val_accuracy = np.array(curves["val_accuracy"])
        row_sent_acc = np.array(curves["row_sent_overall_accuracy"]) if curves["row_sent_overall_accuracy"] else None
        row_sent_ap = np.array(curves["row_sent_avg_precision"]) if curves["row_sent_avg_precision"] else None
        
        # Best values
        best_val_acc = model_data["best_accuracy"]
        best_val_epoch = model_data["best_epoch"]
        best_row_acc = model_data["best_test_overall_accuracy"]
        best_row_epoch = model_data["best_test_epoch"]
        best_row_ap = model_data["best_test_avg_precision"]
        best_ap_epoch = model_data["best_test_precision_epoch"]
        
        # Store for improvement calculation
        improvements_acc[model_name] = best_row_acc
        improvements_ap[model_name] = best_row_ap
        
        # Store LOKI's best values for vertical arrow placement
        if model_name == "LOKI":
            loki_best_values = {
                'val_acc': best_val_acc,
                'row_acc': best_row_acc,
                'row_ap': best_row_ap
            }
        
        # 1. Table-Text Matching Accuracy
        axes[0].plot(epochs, val_accuracy, color=color, linewidth=2.5, linestyle=linestyle,
                    label=f'{model_name}', zorder=3)
        # Mark best point with marker
        axes[0].scatter([best_val_epoch], [best_val_acc], color=color, s=marker_size, 
                       marker=marker, edgecolors='white', linewidths=1.5, zorder=4)
        
        # Smart annotation positioning for Table-Text Matching to avoid overlap
        # Model-specific positioning to avoid baseline overlap
        if model_name == "LOKI":
            # LOKI (highest) goes above the marker
            x_offset = 5
            y_offset = 12
            va = 'bottom'
            ha = 'left'
        elif model_name == "FT-Encoder":
            # FT-Encoder: to the right and above its line (avoid baseline overlap)
            x_offset = 0
            y_offset = 12
            va = 'bottom'
            ha = 'left'
        elif model_name == UNI_R_TO_S:
            # Uni (R⟶S) goes below the marker
            x_offset = 5
            y_offset = -18
            va = 'top'
            ha = 'left'
        elif model_name == UNI_S_TO_R:
            # Uni (S⟶R) is placed slightly above-right to avoid overlap
            x_offset = 6
            y_offset = 12
            va = 'bottom'
            ha = 'left'
        else:
            # Default: to the right
            x_offset = 10
            y_offset = 5
            va = 'center'
            ha = 'left'
        
        axes[0].annotate(f'{round_half_up(best_val_acc, 2):.2f}', 
                        xy=(best_val_epoch, best_val_acc),
                        xytext=(x_offset, y_offset), textcoords='offset points',
                        fontsize=14, fontweight='bold', color=color, va=va, ha=ha)
        
        # 2. Row-Sentence F1
        if row_sent_acc is not None and len(row_sent_acc) > 0:
            axes[1].plot(epochs, row_sent_acc, color=color, linewidth=2.5, linestyle=linestyle,
                        label=f'{model_name}', zorder=3)
            if best_row_epoch < len(row_sent_acc):
                axes[1].scatter([best_row_epoch], [row_sent_acc[best_row_epoch]], color=color, 
                               s=marker_size, marker=marker, edgecolors='white', linewidths=1.5, zorder=4)
                axes[1].annotate(f'{round_half_up(best_row_acc, 2):.2f}', 
                                xy=(best_row_epoch, row_sent_acc[best_row_epoch]),
                                xytext=(5, 10), textcoords='offset points',
                                fontsize=14, fontweight='bold', color=color)
        
        # 3. Row-Sentence Average Precision
        if row_sent_ap is not None and len(row_sent_ap) > 0:
            axes[2].plot(epochs, row_sent_ap, color=color, linewidth=2.5, linestyle=linestyle,
                        label=f'{model_name}', zorder=3)
            if best_ap_epoch < len(row_sent_ap):
                axes[2].scatter([best_ap_epoch], [row_sent_ap[best_ap_epoch]], color=color,
                               s=marker_size, marker=marker, edgecolors='white', linewidths=1.5, zorder=4)
                axes[2].annotate(f'{round_half_up(best_row_ap, 2):.2f}', 
                                xy=(best_ap_epoch, row_sent_ap[best_ap_epoch]),
                                xytext=(5, 10), textcoords='offset points',
                                fontsize=14, fontweight='bold', color=color)
    
    # Add frozen encoder baselines (red dashed lines)
    frozen_val_acc = baseline.get("frozen_encoder_accuracy", 0.0)
    frozen_acc = baseline["frozen_encoder_row_sent_acc"]
    frozen_ap = baseline["frozen_encoder_row_sent_ap"]
    
    # Baseline for Table-Text Matching (subplot 1)
    if frozen_val_acc > 0:
        axes[0].axhline(y=frozen_val_acc, color='#E74C3C', linestyle='--', linewidth=2, 
                       label=f'Baseline', zorder=2)
        # Add baseline value annotation BELOW the line (to avoid overlap with FT-Encoder)
        axes[0].annotate(f'{round_half_up(frozen_val_acc, 2):.2f}', 
                        xy=(0.5, frozen_val_acc), xycoords=('axes fraction', 'data'),
                        xytext=(0, -8), textcoords='offset points',
                        fontsize=14, color='#E74C3C', ha='center', va='top', fontweight='bold')
    
    # Baseline for Row-Sentence F1 (subplot 2)
    if frozen_acc > 0:
        axes[1].axhline(y=frozen_acc, color='#E74C3C', linestyle='--', linewidth=2, 
                       label=f'Baseline', zorder=2)
        # Add baseline value annotation BELOW the line, no box (plain text)
        axes[1].annotate(f'{round_half_up(frozen_acc, 2):.2f}', 
                        xy=(0.5, frozen_acc), xycoords=('axes fraction', 'data'),
                        xytext=(0, -2), textcoords='offset points',
                        fontsize=14, color='#E74C3C', ha='center', va='top', fontweight='bold')
    
    # Baseline for Row-Sentence Average Precision (subplot 3)
    if frozen_ap > 0:
        axes[2].axhline(y=frozen_ap, color='#E74C3C', linestyle='--', linewidth=2, 
                       label=f'Baseline', zorder=2)
        # Add baseline value annotation BELOW the line, no box (plain text)
        axes[2].annotate(f'{round_half_up(frozen_ap, 2):.2f}', 
                        xy=(0.5, frozen_ap), xycoords=('axes fraction', 'data'),
                        xytext=(0, -8), textcoords='offset points',
                        fontsize=14, color='#E74C3C', ha='center', va='top', fontweight='bold')
    
    # Add vertical arrow with improvement annotation (LOKI only) - between baseline and LOKI line
    loki_color = get_model_color("LOKI", 0)
    
    # For Table-Text Matching Accuracy (subplot 1)
    if frozen_val_acc > 0 and "LOKI" in models:
        loki_val_acc = models["LOKI"]["best_accuracy"]
        if loki_val_acc > frozen_val_acc:
            rel_pct = ((loki_val_acc - frozen_val_acc) / frozen_val_acc) * 100
            mid_y = (frozen_val_acc + loki_val_acc) / 2
            # Draw vertical double-headed arrow at x=18 (right end of plot, between epoch 17-20)
            axes[0].annotate('', xy=(18, loki_val_acc - 0.01), xytext=(18, frozen_val_acc + 0.01),
                            arrowprops=dict(arrowstyle='<->', color=loki_color, lw=2.5,
                                           shrinkA=0, shrinkB=0))
            # Add percentage label in the middle of the arrow
            axes[0].annotate(f'+{round_half_up(rel_pct, 1):.1f}%', 
                            xy=(18, mid_y), xycoords='data',
                            fontsize=13, color=loki_color, ha='center', va='center',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                     edgecolor=loki_color, alpha=0.95))
    
    # For Row-Sentence F1 (subplot 2)
    if frozen_acc > 0 and "LOKI" in improvements_acc:
        loki_acc = improvements_acc["LOKI"]
        if loki_acc > frozen_acc:
            rel_pct = ((loki_acc - frozen_acc) / frozen_acc) * 100
            mid_y = (frozen_acc + loki_acc) / 2
            # Draw vertical double-headed arrow at x=10 (middle of plot)
            axes[1].annotate('', xy=(10, loki_acc - 0.01), xytext=(10, frozen_acc + 0.01),
                            arrowprops=dict(arrowstyle='<->', color=loki_color, lw=2.5,
                                           shrinkA=0, shrinkB=0))
            # Add percentage label in the middle of the arrow
            axes[1].annotate(f'+{round_half_up(rel_pct, 1):.1f}%', 
                            xy=(10, mid_y), xycoords='data',
                            fontsize=13, color=loki_color, ha='center', va='center',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                     edgecolor=loki_color, alpha=0.95))
    
    # For Row-Sentence Average Precision (subplot 3)
    if frozen_ap > 0 and "LOKI" in improvements_ap:
        loki_ap = improvements_ap["LOKI"]
        if loki_ap > frozen_ap:
            rel_pct = ((loki_ap - frozen_ap) / frozen_ap) * 100
            mid_y = (frozen_ap + loki_ap) / 2
            # Draw vertical double-headed arrow at x=10 (middle of plot)
            axes[2].annotate('', xy=(10, loki_ap - 0.01), xytext=(10, frozen_ap + 0.01),
                            arrowprops=dict(arrowstyle='<->', color=loki_color, lw=2.5,
                                           shrinkA=0, shrinkB=0))
            # Add percentage label in the middle of the arrow
            axes[2].annotate(f'+{round_half_up(rel_pct, 1):.1f}%', 
                            xy=(10, mid_y), xycoords='data',
                            fontsize=13, color=loki_color, ha='center', va='center',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                     edgecolor=loki_color, alpha=0.95))
    
    # Customize subplot 1: Table-Text Matching (with squeezed y-axis)
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='normal')
    axes[0].set_ylabel('Accuracy', fontsize=13, fontweight='normal')
    axes[0].set_title('Table-Text Matching: Accuracy', fontsize=14, fontweight='bold', pad=10)
    # Apply squeezed scale: compress region 0.2-0.6 to 30% of its original height
    axes[0].set_yscale('squeezed', low=0.2, high=0.6, compression=0.3)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[0].legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    # Customize subplot 2: Row-Sentence Alignment F1 - Legend in LOWER LEFT
    axes[1].set_xlabel('Epoch', fontsize=13, fontweight='normal')
    axes[1].set_ylabel('F1 Score', fontsize=13, fontweight='normal')
    axes[1].set_title('Row-Sentence Alignment: F1 Score', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_ylim(0.0, 0.5)
    axes[1].set_yticks(np.arange(0.0, 0.55, 0.1))
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[1].legend(loc='lower left', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    # Customize subplot 3: Row-Sentence Alignment Average Precision - Legend in LOWER LEFT
    axes[2].set_xlabel('Epoch', fontsize=13, fontweight='normal')
    axes[2].set_ylabel('Average Precision (AP)', fontsize=13, fontweight='normal')
    axes[2].set_title('Row-Sentence Alignment: Average Precision (AP)', fontsize=14, fontweight='bold', pad=10)
    axes[2].set_ylim(0.0, 0.6)
    axes[2].set_yticks(np.arange(0.0, 0.65, 0.1))
    axes[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[2].legend(loc='lower left', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "model_comparison_test_metrics"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Test metrics comparison plot saved to: {output_path}.png and {output_path}.pdf")


def plot_emergent_ability_showcase(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (10, 5)
) -> None:
    """
    THE KEY NOVELTY FIGURE - Crystal clear visualization of LOKI's emergent ability.
    
    Shows that LOKI learns local (row-sentence) alignment from only global 
    (table-text) supervision, while other methods completely fail at this.
    
    Design: Clean side-by-side comparison
    - Left panel: What models are trained on (Table-Text Matching) - ALL improve
    - Right panel: What emerges (Row-Sentence Alignment) - ONLY LOKI improves
    
    The contrast is immediately obvious to any reviewer.
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    
    print(f"\n🌟 Generating Emergent Ability Showcase (Key Novelty Figure)...")
    
    # Get baseline values
    baseline_acc = baseline.get("frozen_encoder_accuracy", 0.0)
    baseline_ap = baseline.get("frozen_encoder_row_sent_ap", 0.0)
    
    # Calculate improvements for each model
    model_data_list = []
    model_names = get_present_models_for_panels(models)
    for model_name in model_names:
        if model_name in models:
            m = models[model_name]
            # Get best values
            best_acc = m["best_accuracy"]
            best_ap = m["best_test_avg_precision"]
            
            # Calculate relative improvements
            acc_improvement = ((best_acc - baseline_acc) / baseline_acc) * 100
            ap_improvement = ((best_ap - baseline_ap) / baseline_ap) * 100
            
            model_data_list.append({
                'name': model_name,
                'acc_improvement': acc_improvement,
                'ap_improvement': ap_improvement,
                'best_acc': best_acc,
                'best_ap': best_ap,
            })
    
    # Conference-style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Colors
    colors = {name: get_model_color(name) for name in model_names}
    
    model_names = [d['name'] for d in model_data_list]
    x = np.arange(len(model_names))
    bar_width = 0.6
    
    # ===== LEFT PANEL: Training Task (Table-Text Matching) =====
    ax1 = axes[0]
    ax1.set_facecolor('white')
    
    acc_improvements = [d['acc_improvement'] for d in model_data_list]
    bars1 = ax1.bar(x, acc_improvements, bar_width, 
                    color=[colors[n] for n in model_names],
                    edgecolor='white', linewidth=2, alpha=0.9)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, acc_improvements)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'+{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=colors[model_names[i]])
    
    ax1.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Task\n(Table-Text Matching)', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11, fontweight='normal')
    ax1.set_ylim(0, max(acc_improvements) * 1.25)
    ax1.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # # Add "All models improve" annotation
    # ax1.text(0.5, 0.92, '✓ All models improve', transform=ax1.transAxes,
    #          ha='center', va='top', fontsize=11, color='#27AE60', fontweight='bold',
    #          bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5', edgecolor='#27AE60', alpha=0.9))
    
    # ===== RIGHT PANEL: Emergent Task (Row-Sentence Alignment) =====
    ax2 = axes[1]
    ax2.set_facecolor('white')
    
    ap_improvements = [d['ap_improvement'] for d in model_data_list]
    bars2 = ax2.bar(x, ap_improvements, bar_width,
                    color=[colors[n] for n in model_names],
                    edgecolor='white', linewidth=2, alpha=0.9)
    
    # Add value labels on bars (handle negative values)
    for i, (bar, val) in enumerate(zip(bars2, ap_improvements)):
        y_pos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2
        va = 'bottom' if val >= 0 else 'top'
        sign = '+' if val >= 0 else ''
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{sign}{val:.1f}%', ha='center', va=va, fontsize=12, fontweight='bold',
                color=colors[model_names[i]])
    
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Emergent Task\n(Row-Sentence Alignment)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=11, fontweight='normal')
    
    # Set symmetric y-limits for better comparison
    max_abs = max(abs(min(ap_improvements)), abs(max(ap_improvements)))
    ax2.set_ylim(min(-5, min(ap_improvements) * 1.3), max(ap_improvements) * 1.25)
    ax2.axhline(y=0, color='#E74C3C', linewidth=2, linestyle='-', label='Baseline')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # # Add "Only LOKI improves" annotation with emphasis
    # ax2.text(0.5, 0.92, '★ Only LOKI improves!', transform=ax2.transAxes,
    #          ha='center', va='top', fontsize=12, color='#2E86AB', fontweight='bold',
    #          bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#2E86AB', alpha=0.95, linewidth=2))
    
    # Highlight LOKI's bar with a star marker (using matplotlib's built-in '*')
    loki_idx = model_names.index('LOKI')
    ax2.scatter([loki_idx], [ap_improvements[loki_idx] + 3], marker='*', s=400, 
                color='#F1C40F', edgecolors='#2E86AB', linewidths=1.5, zorder=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "emergent_ability_showcase"
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Emergent ability showcase saved to: {output_path}.png and {output_path}.pdf")


def plot_emergence_cliff_3d(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (35, 8),
    emergent_metric: str = "ap",  # "ap" or "f1" ("accuracy" kept as alias)
    z_variation_scale: float = 50.0  # Scaling factor for Z-axis topographical variation
) -> None:
    """
    STUNNING 3D LAYERED SURFACE VISUALIZATION - Publication-quality.
    
    Creates horizontal planes/surfaces for each model where:
    - The AREA of each plane represents the model's improvement
    - X-dimension: Table-Text Accuracy improvement (Training Task)
    - Y-dimension: Row-Sentence improvement (Emergent Task) - AP or F1
    - Z-height: Model tier (stacked layers) + topographical variation
    
    The surface height variation within each plane shows the per-epoch deviation
    of the emergent metric from its mean value across all epochs.
    
    Args:
        emergent_metric: "ap" for Average Precision, "f1" for Row-Sentence F1
        z_variation_scale: Scaling factor for the Z-axis topographical variation.
                          Higher values = more pronounced peaks/valleys.
                          Recommended range: 20-100. Default: 50.0
    
    LOKI's plane is dramatically larger, showing superior emergence.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import interp1d
    
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    
    use_f1_metric = emergent_metric in ("f1", "accuracy")
    metric_label = "AP" if emergent_metric == "ap" else "F1"
    print(f"\n🏔️  Generating 3D Layered Surface Visualization (Emergent: {metric_label})...")
    print(f"   📊 Z-variation scale: {z_variation_scale}")

    # Font sizing for compact paper figures (increase legibility while shrinking figure)
    base_fs = 13
    small_fs = 11
    title_fs = base_fs + 2
    
    # Get baseline values
    baseline_acc = baseline.get("frozen_encoder_accuracy", 0.0)
    baseline_rs_acc = baseline.get("frozen_encoder_row_sent_acc", 0.215)
    baseline_ap = baseline.get("frozen_encoder_row_sent_ap", 0.409)
    
    # Choose which emergent metric to use
    if emergent_metric == "ap":
        baseline_emergent = baseline_ap
        emergent_key = "best_test_avg_precision"
        emergent_curve_key = "row_sent_avg_precision"
        y_label = "Δ Row-Sentence Alignment: AP (%)"
        output_suffix = ""  # Default filename
    elif use_f1_metric:
        baseline_emergent = baseline_rs_acc
        emergent_key = "best_test_overall_accuracy"
        emergent_curve_key = "row_sent_overall_accuracy"
        y_label = "Δ Row-Sentence Alignment: F1 (%)"
        output_suffix = "_f1"
    else:
        raise ValueError(f"Unsupported emergent_metric='{emergent_metric}'. Use 'ap' or 'f1'.")
    
    # Collect improvements and epoch deviations for each model
    model_improvements = {}
    model_epoch_deviations = {}  # Store per-epoch deviations from mean
    
    for model_name in get_present_models_for_panels(models):
        if model_name in models:
            m = models[model_name]
            best_acc = m["best_accuracy"]
            best_emergent = m[emergent_key]
            
            # Calculate % improvements
            training_imp = ((best_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
            emergent_imp = ((best_emergent - baseline_emergent) / baseline_emergent) * 100 if baseline_emergent > 0 else 0
            
            model_improvements[model_name] = {
                'training': max(training_imp, 0.5),  # Minimum visibility
                'emergent': max(emergent_imp, 0.5),
                'area': max(training_imp, 0.5) * max(emergent_imp, 0.5)
            }
            
            # Get per-epoch emergent metric values and calculate deviation from mean
            if "curves" in m and emergent_curve_key in m["curves"]:
                epoch_values = np.array(m["curves"][emergent_curve_key])
                mean_value = np.mean(epoch_values)
                deviations = epoch_values - mean_value  # Deviation from mean
                model_epoch_deviations[model_name] = deviations
                print(f"   📈 {model_name}: {len(epoch_values)} epochs, mean={mean_value:.4f}, dev range=[{deviations.min():.4f}, {deviations.max():.4f}]")
            else:
                model_epoch_deviations[model_name] = np.array([0.0])
    
    # Add Baseline (0 improvement = small reference plane, no variation)
    model_improvements["Baseline"] = {'training': 1, 'emergent': 1, 'area': 1}
    model_epoch_deviations["Baseline"] = np.array([0.0])
    
    # Create figure with CLEAN WHITE background for academic papers
    fig = plt.figure(figsize=figsize, facecolor='white')
    # Adjust subplot positioning to center the plot and prevent label cutoff
    # Increase left/bottom margins to ensure Z-axis label and ticks are visible
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.06, top=0.95)  # More space for labels
    
    # Professional academic color scheme
    colors = MODEL_SURFACE_COLORS
    
    def create_topographic_surface(x_size, y_size, z_height, epoch_deviations, scale):
        """
        Create a surface with topographical variation based on per-epoch deviations.
        
        The surface grid maps epochs across it, with Z-height varying based on
        how much each epoch's emergent metric deviated from the mean.
        
        Args:
            x_size: Width of the surface (training improvement %)
            y_size: Depth of the surface (emergent improvement %)
            z_height: Base tier height for the model
            epoch_deviations: Array of per-epoch deviations from mean (emergent metric)
            scale: Scaling factor for the Z variations
        
        Returns:
            X, Y, Z meshgrid arrays for the surface
        """
        resolution = 30
        x = np.linspace(0, x_size, resolution)
        y = np.linspace(0, y_size, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Shift X coordinates by -4.0 so surfaces that previously started at X=4 now start at X=0
        # This aligns the projected footprint with the (0,0) origin as requested.
        # X = X - 1.3
        # Y = Y - 0.3

        # Start with base height
        Z = np.ones_like(X) * z_height
        
        # Map epochs onto the grid
        n_epochs = len(epoch_deviations)
        if n_epochs > 1:
            # Create interpolator for smooth deviation mapping
            epoch_indices = np.linspace(0, 1, n_epochs)
            interp_func = interp1d(epoch_indices, epoch_deviations, kind='cubic',
                                   fill_value='extrapolate')
            
            # Map Y-axis (0 to y_size) to epoch progression (REVERSED: 1 to 0)
            # Y-axis = Emergent Task, so emergent metric deviation should vary along Y
            # Reversed so the curve/dip is on the FAR side, keeping near side clear for viewing other models
            y_normalized = 1.0 - (Y / y_size)  # 1 to 0 (reversed)
            
            # Map X-axis for secondary modulation (also reversed for consistency)
            x_normalized = 1.0 - (X / x_size)  # 1 to 0 (reversed)
            
            # Create 2D variation pattern:
            # - Along Y: epoch progression (MAIN variation) - this is the Emergent Task axis
            # - Along X: slight modulation for terrain effect
            z_variation_y = interp_func(y_normalized)
            
            # Add X-axis modulation (phase-shifted version for 2D terrain)
            x_phase_shift = 0.3  # How much X shifts the pattern
            x_modulated = np.clip(y_normalized + x_normalized * x_phase_shift, 0, 1)
            z_variation_x = interp_func(x_modulated) * 0.5  # Reduced amplitude
            
            # Combine Y and X variations (Y is primary = 70%, X is secondary = 30%)
            z_variation = z_variation_y * 0.7 + z_variation_x * 0.3
            
            # Apply scaling and add to base height
            Z = Z + z_variation * scale
            
            # Apply gentle Gaussian smoothing for natural terrain look
            Z = gaussian_filter(Z, sigma=1.0)
        
        return X, Y, Z
    
    # Model order from bottom to top (LOKI on top for emphasis)
    model_order = [name for name in COMPARISON_MODEL_ORDER if name == "Baseline" or name in model_improvements]
    z_levels = np.linspace(0.5, 3.8, len(model_order)).tolist()  # Condensed heights for each layer
    
    # NO SCALING - plane dimensions directly correspond to axis values!
    # X-axis = Training improvement (%)
    # Y-axis = Emergent improvement (%)
    # This makes the visualization self-explanatory
    
    # Track maximum dimensions for axis limits
    max_x, max_y = 0, 0
    max_z = 0  # Track max Z for proper axis limits
    min_z = float('inf')  # Track min Z
    # Track global X extents after applying the X offset so we can set correct x-limits
    min_x_all, max_x_all = float('inf'), -float('inf')
    
    # Track Z range and actual emergent scores for ALL models
    model_z_ranges = {}  # {model_name: (z_min, z_max, score_min, score_max)}
    
    # Draw each model's surface as a horizontally-oriented plane
    for idx, model_name in enumerate(model_order):
        if model_name not in model_improvements:
            continue
            
        imp = model_improvements[model_name]
        # Use actual percentage values directly (no arbitrary scaling)
        x_size = imp['training']   # Directly maps to X-axis (Training %)
        y_size = imp['emergent']   # Directly maps to Y-axis (Emergent %)
        z_height = z_levels[idx]
        
        max_x = max(max_x, x_size)
        max_y = max(max_y, y_size)
        
        col = colors[model_name]
        
        # Get epoch deviations for this model
        epoch_devs = model_epoch_deviations.get(model_name, np.array([0.0]))
        
        # Create the topographic surface with Z-variation
        X, Y, Z = create_topographic_surface(x_size, y_size, z_height, epoch_devs, z_variation_scale)
        
        # Track max/min Z for axis limits
        max_z = max(max_z, Z.max())
        min_z = min(min_z, Z.min())
        
        # Track Z range and actual emergent scores for this model
        if model_name != "Baseline" and model_name in models and "curves" in models[model_name] and emergent_curve_key in models[model_name]["curves"]:
            model_scores = np.array(models[model_name]["curves"][emergent_curve_key])
            model_z_ranges[model_name] = (Z.min(), Z.max(), model_scores.min(), model_scores.max())
        elif model_name == "Baseline":
            # Baseline uses the baseline value
            model_z_ranges[model_name] = (Z.min(), Z.max(), baseline_emergent, baseline_emergent)

        # Update global X extents to account for the -4.0 shift applied to this surface
        min_x_all = min(min_x_all, float(X.min()))
        max_x_all = max(max_x_all, float(X.max()))
        
        # Plot surface with professional styling
        if model_name == "LOKI":
            # LOKI gets special treatment - gradient colormap
            # Reversed colormap: darker blue at TOP (higher Z), lighter at bottom
            cmap = LinearSegmentedColormap.from_list('loki_academic', 
                ['#85C1E9', '#5DADE2', '#2E86AB', '#2874A6', '#1A5276'])
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9, 
                                   antialiased=True, shade=True,
                                   lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=45))
            # Add edge wireframe for definition
            ax.plot_wireframe(X, Y, Z, color='#1A5276', alpha=0.5, linewidth=0.8, 
                             rstride=5, cstride=5)
        else:
            # Other models get solid surfaces
            alpha_val = 0.85 if model_name != "Baseline" else 0.6
            surf = ax.plot_surface(X, Y, Z, color=col['surface'], alpha=alpha_val,
                                   antialiased=True, shade=True,
                                   lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45))
            ax.plot_wireframe(X, Y, Z, color=col['edge'], alpha=0.4, linewidth=0.5,
                             rstride=8, cstride=8)

        # Compute surface extents based on the (shifted) X/Y grid so labels are positioned correctly
        x_min, x_max = float(X.min()), float(X.max())
        y_min, y_max = float(Y.min()), float(Y.max())
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        
        # Add vertical pillars at corners — use the ACTUAL surface grid minima/maxima so
        # footprints and pillars align precisely with the plotted surface (fixes offset issue)
        # Note: use the X/Y arrays produced for this surface instead of static (0, x_size)
        x_min, x_max = float(X.min()), float(X.max())
        y_min, y_max = float(Y.min()), float(Y.max())
        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

        for cx, cy in corners:
            # Find nearest grid indices for the corner coordinates (meshgrid may be floating)
            try:
                ix = (np.abs(X[0, :] - cx)).argmin()
                iy = (np.abs(Y[:, 0] - cy)).argmin()
                z_top = float(Z[iy, ix])
            except Exception:
                # Fallback to tier base height if something goes wrong
                z_top = z_height

            # Draw the vertical line from the surface down to Z=0 so the projection matches
            ax.plot([cx, cx], [cy, cy], [0, z_top],
                    color=col['edge'], alpha=0.6 if model_name == "LOKI" else 0.35,
                    linewidth=2 if model_name == "LOKI" else 1, linestyle=':')

        # Add a subtle footprint rectangle on the XY plane to show the true projected area.
        # Use the surface's own min/max extents so the footprint shadows line up exactly.
        try:
            verts = [(x_min, y_min, 0), (x_max, y_min, 0), (x_max, y_max, 0), (x_min, y_max, 0)]
            poly = Poly3DCollection([verts], facecolors=[col['surface']], alpha=0.06, edgecolor=col['edge'])
            ax.add_collection3d(poly)

            # Additionally draw a shadow using the same X/Y grid to guarantee perfect alignment
            # (plotting a zero-height surface from the same X/Y ensures the projected shadow
            #  starts exactly where the surface does in data coordinates).
            try:
                Z_shadow = np.zeros_like(Z)
                # Use a very subtle shade for the shadow/footprint
                ax.plot_surface(X, Y, Z_shadow, color=col['edge'], alpha=0.06, shade=False)
            except Exception:
                pass
        except Exception:
            pass
        
        # Calculate area for display
        area = imp['training'] * imp['emergent']
        label_z = z_height  # Reduced offset for condensed layout
        
        if model_name == "LOKI":
            # LOKI gets prominent label centered on the actual surface footprint
            ax.text(x_center, y_center, label_z,  # Use computed center (accounts for X shift)
                    f'★ LOKI\nArea: {area:.0f}x\n+{imp["training"]:.0f}% Training\n+{imp["emergent"]:.0f}% Implicit',
                    fontsize=base_fs, fontweight='bold', color='#1A5276',
                    ha='center', va='bottom', zorder=100,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='#2E86AB', alpha=0.95, linewidth=2))
        else:
            # Position labels in the left empty grid space (X≈0), aligned next to each plane by Y and Z
            # Use the surface's x_min to calculate a stable left-side label position even after shifting X
            label_x = x_min - 3.5  # Place label left of the surface footprint
            label_y = y_center + 8 # Center on this plane's Y extent
            label_z = z_height + 1 # Just above this plane's tier
            
            # Create a 2D text object and convert it into 3D so it can be
            # oriented to lie in the Y-Z plane (i.e., parallel to the Y axis).
            from mpl_toolkits.mplot3d import art3d

            txt = ax.text(label_x, label_y, label_z,
                          f'{model_name}\n({imp["training"]:.0f}% × {imp["emergent"]:.0f}%)',
                          fontsize=base_fs, color=col['edge'], ha='left', va='center',
                          fontweight='bold')
            try:
                # zdir='x' rotates the 2D text into the Y-Z plane
                art3d.text_2d_to_3d(txt, z=label_z, zdir='y')
            except Exception:
                # Fallback: keep the original 3D text if conversion fails
                pass
    
    # Add "Zero plane" reference at the bottom (light gray)
    # ref_size = max_x * 1.1
    # X_ref, Y_ref = np.meshgrid(np.linspace(-2, ref_size + 2, 10), 
    #                            np.linspace(-2, max_y * 1.1, 10))
    # Z_ref = np.zeros_like(X_ref) - 0.5
    # ax.plot_surface(X_ref, Y_ref, Z_ref, color='#E8E8E8', alpha=0.3, shade=True)
    # ax.plot_wireframe(X_ref, Y_ref, Z_ref, color='#CCCCCC', alpha=0.3, linewidth=0.3)
    
    # Axis labels - professional black text
    metric_title = "AP" if emergent_metric == "ap" else "F1"
    
    ax.set_xlabel('\n\nΔ Table-Text Matching\n(Accuracy Improvement %)', 
                  fontsize=base_fs, fontweight='bold', color='#2C3E50', labelpad=6, rotation=22)
    # Nudge X-label to better align with axis line in 3D view (closer to axis)
    # NOTE: `set_label_coords` was removed because manual 2D coords can misplace labels
    #       when using a 3D projection and rotated view. Rely on `labelpad`/`rotation`
    #       and `ax.view_init` for correct placement instead.
    ax.set_ylabel('\n\n' + y_label, 
                  fontsize=base_fs, fontweight='bold', color='#2C3E50', labelpad=4)
    # Shift Y-label slightly left to avoid clipping and position it closer to axis
    # Removed manual 2D label coordinates for 3D axes. `labelpad` is used to control spacing.
    
    # Z-axis label - show Row-Sentence metric over epochs
    # z_axis_title = f'Row-Sentence {metric_title}\nover Epochs'
    # ax.set_zlabel('\n' + z_axis_title, 
    #               fontsize=11, fontweight='bold', color='#2C3E50', labelpad=35)
    
    # Set axis limits so X and Y start at 0 — Z-axis and ticks sit on the X=0 grid line (single 0-line, no redundant gap)
    # Set X-limits starting at 0 so the plane projection visually begins at (0,0).
    # Use the tracked global max X after the -4.0 shift so limits fit all surfaces.
    ax.set_xlim(0, max(0, max_x_all) + 5)
    ax.set_ylim(0, max_y+2)
    z_padding = 0.5 # Reduced padding for condensed vertical layout
    ax.set_zlim(min_z-z_padding, max_z)
    
    # X and Y grid lines at 0, 5, 10, 15, ...
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    
    # Custom Z-axis ticks showing PEAK (max) score for each model
    z_ticks = []
    z_labels = []
    
    # Add ticks for each model at their max-Z position (peak) with max score
    for model_name in model_order:
        if model_name in model_z_ranges:
            z_min, z_max, score_min, score_max = model_z_ranges[model_name]
            z_ticks.append(z_max)  # Use max Z position (peak)
            # Use only the numeric best score for the Z-axis tick label
            z_labels.append(f'{score_max:.2f}')
    
    if z_ticks:
        ax.set_zticks(z_ticks)
        ax.set_zticklabels(z_labels)
        # Make Z-axis tick labels bold to improve emphasis
        for tl in ax.get_zticklabels():
            tl.set_fontsize(base_fs)
            tl.set_fontweight('bold')

    # Add clear Z-axis label (title) describing the tick numbers
    z_label_text = f"\nBest {metric_title} Scores"
    # Move Z-axis label closer to the Z-axis using smaller labelpad. Avoid manual 2D
    # label coordinates for 3D axes because they are incompatible with rotated 3D views.
    ax.set_zlabel(z_label_text, fontsize=base_fs, fontweight='bold', color='#2C3E50', labelpad=6)
    
    # Customize ticks - bring Z ticks closer to axis so they align visually
    # Use `base_fs` so tick labels match other plot text sizes for paper
    ax.tick_params(axis='z', colors='#4A4A4A', labelsize=base_fs, pad=6)
    ax.tick_params(axis='x', colors='#4A4A4A', labelsize=base_fs)
    ax.tick_params(axis='y', colors='#4A4A4A', labelsize=base_fs, pad=6)
    
    # Set view angle - rotated to better show Y-axis (Emergent Task) surface variation
    ax.view_init(elev=28, azim=210)
    
    # Add perspective projection for depth effect
    # Smaller focal_length = more pronounced perspective (nearby bigger, far smaller)
    # This makes smaller surfaces (other models) more visible in foreground
    ax.set_proj_type('persp', focal_length=0.2)
    
    # Style the panes with light theme
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.9))
    ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.97, 0.9))
    ax.zaxis.pane.set_facecolor((0.93, 0.95, 0.98, 0.9))
    
    # Grid styling (subtle gray) - dashed and lower alpha for publication clarity
    # For 3D axes, configure grid via the axis _axinfo dict so linestyle/alpha apply
    for _axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            _axis._axinfo['grid']['linewidth'] = 0.6
            _axis._axinfo['grid']['linestyle'] = '--'
            _axis._axinfo['grid']['color'] = '#CCCCCC'
            _axis._axinfo['grid']['alpha'] = 0.15
        except Exception:
            # Fallback: apply generic 2D grid settings
            ax.grid(True, alpha=0.15, linestyle='--', color='#CCCCCC')
    
    
    # Title - professional styling, very small gap above plot
    fig.suptitle(f'Implicit Capabilities: Knowledge Transfer Comparison ({metric_title})',
                 fontsize=title_fs, fontweight='bold', color='#2C3E50', y=0.87)

    # Legend intentionally removed — annotations inside the plot are used instead.
    # Add legend with white background - bottom left where there is empty space, stacked vertically
    # legend_elements = [
    #     mpatches.Patch(facecolor=colors[m]['surface'], edgecolor=colors[m]['edge'], 
    #                   label=f'{m}', alpha=0.85)
    #     for m in model_order
    # ]
    # ax.legend(handles=legend_elements, loc='lower left', fontsize=base_fs, 
    #           facecolor='white', edgecolor='#CCCCCC', labelcolor='#2C3E50',
    #           framealpha=0.95, ncol=1, bbox_to_anchor=(-0.10, 0.02))  # Stacked one on top of another and further left into empty space
     
    # # Add annotation explaining the visualization
    # fig.text(0.02, 0.02,
    #          'Each surface represents a model\'s\nimprovement with topographical variation.\n',
    #          fontsize=11, color='#5D6D7E', style='italic',
    #          bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', 
    #                   edgecolor='#CCCCCC', alpha=0.95))
    
    # Manual subplot adjustment is used instead of tight_layout for better control
    
    # Save plot with white background
    output_path = Path(output_dir) / f"emergence_cliff_3d{output_suffix}"
    for fmt in ['png', 'pdf']:
        save_path = output_path.parent / f"{output_path.stem}.{fmt}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format=fmt)
        print(f"Saved plot to {save_path}")
    plt.close()
    
    print(f"✅ Emergence Area 3D saved to: {output_path}.png and {output_path}.pdf")


def plot_global_to_local_transfer(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (10, 7)
) -> None:
    """
    Visualize the "Global → Local" knowledge transfer that makes LOKI unique.
    
    Shows each model as a trajectory line from baseline (origin) to final state:
    - X-axis: Training Task Improvement (%) - Table-Text Matching Accuracy
    - Y-axis: Emergent Task Improvement (%) - Row-Sentence Average Precision (AP)
    
    Knowledge Transfer Zones:
    - "No Knowledge Transfer": Y-improvement ≤ 10% (model learns training task but not emergent task)
    - "Emergent Knowledge Transfer": Y-improvement > 10% (model transfers knowledge to emergent task)
    
    The threshold of 10% is based on:
    - Baseline row-sent AP ≈ 0.41, so 10% improvement = 0.451 AP
    - This is a meaningful improvement beyond noise/variance
    - Below 10%: improvements could be random or due to minor side effects
    - Above 10%: indicates genuine knowledge transfer from global to local alignment
    
    LOKI appears in upper-right (improves on both)
    Others appear along bottom or slightly above (improve mainly on training task)
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    
    print(f"\n🌟 Generating Global→Local Transfer Plot...")
    
    # Get baseline values
    baseline_acc = baseline.get("frozen_encoder_accuracy", 0.0)
    baseline_ap = baseline.get("frozen_encoder_row_sent_ap", 0.0)
    
    print(f"   📊 Baseline: Accuracy={baseline_acc:.4f}, Row-Sent AP={baseline_ap:.4f}")
    
    # Knowledge transfer threshold (percentage improvement in Y-axis)
    # Below this: "No Knowledge Transfer" - Above this: "Emergent Knowledge Transfer"
    TRANSFER_THRESHOLD = 10.0  # 10% improvement threshold
    
    # Conference-style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    
    # Colors matching other plots
    colors = {name: get_model_color(name) for name in get_present_models(models, include_baseline=True)}
    
    # Plot baseline at origin (small marker)
    ax.scatter([0], [0], s=80, color=colors['Baseline'], marker='x', 
               linewidths=2, zorder=5, label='Baseline')
    ax.annotate('Baseline', xy=(0, 0), xytext=(3, -7), fontsize=9, 
                color=colors['Baseline'], fontweight='bold')
    
    # Track min/max for axis limits
    x_vals = [0]
    y_vals = [0]
    
    # Plot each model as a line with arrow at the end
    for model_name in get_present_models_for_panels(models):
        if model_name not in models:
            print(f"   ⚠️ {model_name} not in models, skipping...")
            continue
            
        m = models[model_name]
        best_acc = m["best_accuracy"]
        best_ap = m["best_test_avg_precision"]
        
        # Calculate improvements
        x_val = ((best_acc - baseline_acc) / baseline_acc) * 100
        y_val = ((best_ap - baseline_ap) / baseline_ap) * 100
        
        x_vals.append(x_val)
        y_vals.append(y_val)
        
        # Classify transfer type
        transfer_type = "Emergent" if y_val > TRANSFER_THRESHOLD else "No Transfer"
        print(f"   📍 {model_name}: Δ Accuracy={x_val:.1f}%, Δ AP={y_val:.1f}% → {transfer_type}")
        
        # Draw arrow from origin to endpoint (this is the trajectory line with arrow)
        linewidth = 3.0 if model_name == "LOKI" else 2.5
        ax.annotate('', xy=(x_val, y_val), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=colors[model_name], 
                                   lw=linewidth, mutation_scale=15,
                                   connectionstyle='arc3,rad=0.08'),
                    zorder=6)
        
        # Add invisible line for legend
        ax.plot([], [], color=colors[model_name], linewidth=linewidth, label=model_name)
        
        # Add label near the arrow endpoint with proper positioning inside plot area
        if model_name == "LOKI":
            # LOKI label - position to left of arrow tip to stay inside plot
            ax.annotate(f'{model_name}\n(Δ Acc: +{x_val:.1f}%,\n Δ AP: +{y_val:.1f}%)', 
                        xy=(x_val, y_val), xytext=(-75, -10), textcoords='offset points',
                        fontsize=10, fontweight='bold', color=colors[model_name],
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                 edgecolor=colors[model_name], alpha=0.95, linewidth=2))
        elif model_name == "FT-Encoder":
            ax.annotate(f'{model_name}\n(Δ Acc: {x_val:+.1f}%, Δ AP: {y_val:+.1f}%)', 
                        xy=(x_val, y_val), xytext=(8, 10), textcoords='offset points',
                        fontsize=9, color=colors[model_name], fontweight='normal')
        elif model_name == UNI_R_TO_S:
            ax.annotate(f'{model_name}\n(Δ Acc: {x_val:+.1f}%, Δ AP: {y_val:+.1f}%)', 
                        xy=(x_val, y_val), xytext=(8, 12), textcoords='offset points',
                        fontsize=9, color=colors[model_name], fontweight='normal')
        else:
            ax.annotate(f'{model_name}\n(Δ Acc: {x_val:+.1f}%, Δ AP: {y_val:+.1f}%)', 
                        xy=(x_val, y_val), xytext=(8, -18), textcoords='offset points',
                        fontsize=9, color=colors[model_name], fontweight='normal')
    
    # Calculate axis limits based on data with padding
    x_min = min(x_vals) - 5
    x_max = max(x_vals) + 8
    y_min = min(y_vals) - 5
    y_max = max(y_vals) + 8
    
    # Add reference zones based on the transfer threshold
    # "No Transfer" zone (y ≤ threshold)
    ax.axhspan(y_min, TRANSFER_THRESHOLD, alpha=0.12, color='#FFCCCC', zorder=1)
    
    # "Emergent Transfer" zone (y > threshold)
    ax.axhspan(TRANSFER_THRESHOLD, y_max, alpha=0.12, color='#CCFFCC', zorder=1)
    
    # Add horizontal threshold line with label
    ax.axhline(y=TRANSFER_THRESHOLD, color='#888888', linewidth=1.5, linestyle='--', alpha=0.7, zorder=3)
    ax.text(x_max - 1, TRANSFER_THRESHOLD + 1.5, f'Transfer Threshold ({TRANSFER_THRESHOLD}%)', 
            fontsize=8, color='#555555', ha='right', va='bottom', style='italic')
    
    # Zone labels - positioned inside plot area
    ax.text(x_min + 2, (y_min + TRANSFER_THRESHOLD) / 2, 'No Knowledge\nTransfer Zone', 
            fontsize=10, color='#C0392B', alpha=0.8, style='italic', ha='left', va='center',
            fontweight='normal')
    ax.text(x_min + 2, (TRANSFER_THRESHOLD + y_max) / 2 + 3, 'Emergent Knowledge\nTransfer Zone', 
            fontsize=10, color='#27AE60', alpha=0.8, style='italic', ha='left', va='center',
            fontweight='bold')
    
    # Add diagonal "perfect transfer" line (where x improvement = y improvement)
    diag_min = min(x_min, y_min)
    diag_max = max(x_max, y_max)
    ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', alpha=0.25, linewidth=1.5, zorder=2)
    ax.text(min(x_max, y_max) - 5, min(x_max, y_max) - 3, 'Perfect\nTransfer', fontsize=8, 
            color='gray', alpha=0.6, ha='center', va='bottom', style='italic')
    
    # Axis labels and title - CLEARLY state what each axis measures
    ax.set_xlabel('Δ Table-Text Matching Accuracy (%)\n[Training Task - Global Alignment]', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ Row-Sentence Average Precision (%)\n[Emergent Task - Local Alignment]', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Global → Local Knowledge Transfer', fontsize=14, fontweight='bold', pad=15)
    
    # Set limits dynamically
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Grid and spines
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linewidth=1, linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=1, linestyle='-', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend positioned to avoid overlap with data
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "global_to_local_transfer"
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Global→Local transfer plot saved to: {output_path}.png and {output_path}.pdf")


def plot_global_to_local_transfer_3d(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (14, 10),
    elev: float = 28,
    azim: float = 225
) -> None:
    """
    Training-Emergence Relationship Surface: Shows how Training Task improvement 
    relates to Emergent Task improvement for each model.
    
    Like CalcPlot3D, this creates surfaces showing the RELATIONSHIP:
    - X-axis: Training Task Improvement (%) - what we optimize
    - Y-axis: Training Epochs (Time progression)
    - Z-axis (HEIGHT): Emergent Task Improvement (%) - what emerges
    
    Key visual message:
    - LOKI: Creates a rising SURFACE - more training → more emergence
    - FT-Encoder / Uni variants: FLAT surfaces - training doesn't yield emergence
    
    This shows the fundamental difference: LOKI has positive Training→Emergence coupling.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from scipy.interpolate import griddata
    
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    
    print(f"\n🌟 Generating Training-Emergence Relationship 3D Plot...")
    
    # Get baseline values
    baseline_acc = baseline.get("frozen_encoder_accuracy", 0.0)
    baseline_ap = baseline.get("frozen_encoder_row_sent_ap", 0.0)
    
    print(f"   📊 Baseline (Frozen Encoder): Accuracy={baseline_acc:.4f}, AP={baseline_ap:.4f}")
    
    # Conference-style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    
    # Colors with good contrast - now with surface colors
    colors = {name: get_model_color(name) for name in get_present_models(models, include_baseline=True)}
    
    # Model order (LOKI last so it's on top visually)
    model_order = [name for name in TRAINED_MODEL_ORDER if name in models]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Collect data
    trajectories = {}
    max_epoch = 0
    all_delta_acc = []
    all_delta_ap = []
    
    for model_name in model_order:
        if model_name not in models:
            continue
        m = models[model_name]
        curves = m.get("curves", {})
        epochs = curves.get("epochs", [])
        val_accuracies = curves.get("val_accuracy", [])
        row_sent_aps = curves.get("row_sent_avg_precision", [])
        
        if epochs and val_accuracies and row_sent_aps:
            # Calculate deltas from baseline (as percentages)
            delta_accs = [((acc - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0 
                         for acc in val_accuracies]
            delta_aps = [((ap - baseline_ap) / baseline_ap) * 100 if baseline_ap > 0 else 0 
                        for ap in row_sent_aps]
            
            all_delta_acc.extend(delta_accs)
            all_delta_ap.extend(delta_aps)
            max_epoch = max(max_epoch, max(epochs))
            
            trajectories[model_name] = {
                'epochs': np.array(epochs),
                'delta_acc': np.array(delta_accs),
                'delta_ap': np.array(delta_aps)
            }
    
    # Axis limits
    x_min, x_max = min(all_delta_acc) - 5, max(all_delta_acc) + 5
    z_min, z_max = min(all_delta_ap) - 5, max(all_delta_ap) + 10
    
    # =========================================================================
    # Draw the ZERO-EMERGENCE PLANE at Z=0
    # This is the "no emergence" reference - models that don't transfer stay here
    # =========================================================================
    # xx, yy = np.meshgrid(np.linspace(x_min - 3, x_max + 3, 15), np.linspace(0, max_epoch + 2, 15))
    # zz = np.zeros_like(xx)  # Z=0 plane (no emergent improvement)
    # ax.plot_surface(xx, yy, zz, alpha=0.15, color=colors['Baseline'], zorder=1,
    #                linewidth=0, antialiased=True)
    
    # # Add grid lines on the zero plane
    # for y_line in np.linspace(0, max_epoch, 5):
    #     ax.plot([x_min, x_max], [y_line, y_line], [0, 0], 
    #             color=colors['Baseline'], linewidth=0.5, alpha=0.4)
    # for x_line in np.linspace(x_min, x_max, 5):
    #     ax.plot([x_line, x_line], [0, max_epoch], [0, 0], 
    #             color=colors['Baseline'], linewidth=0.5, alpha=0.4)
    
    # # Label for zero plane
    # ax.text(0, max_epoch + 1, 0, 'Zero Emergence\n(No Transfer)', 
    #         fontsize=9, color=colors['Baseline'], ha='center', alpha=0.9, fontweight='bold')
    
    # =========================================================================
    # For each model: Create a "ribbon surface" showing how Training → Emergence
    # This creates a visual surface that shows the relationship
    # =========================================================================
    for model_name in model_order:
        if model_name not in trajectories:
            continue
        
        data = trajectories[model_name]
        epochs = data['epochs']
        delta_acc = data['delta_acc']  # X-axis: Training task Δ
        delta_ap = data['delta_ap']    # Z-axis (HEIGHT): Emergent task Δ
        color = colors[model_name]
        
        n_points = len(epochs)
        
        # =====================================================================
        # Create a ribbon/curtain surface that drops to Z=0
        # This shows the "height" of emergence at each point
        # =====================================================================
        # Create vertices for the ribbon: top edge (actual trajectory) and bottom edge (Z=0)
        verts = []
        face_colors = []
        
        for i in range(n_points - 1):
            # Four corners of each ribbon segment
            # Top-left, Top-right, Bottom-right, Bottom-left
            x1, y1, z1 = delta_acc[i], epochs[i], delta_ap[i]
            x2, y2, z2 = delta_acc[i+1], epochs[i+1], delta_ap[i+1]
            
            quad = [
                [x1, y1, z1],      # Top-left (actual trajectory)
                [x2, y2, z2],      # Top-right (actual trajectory)
                [x2, y2, 0],       # Bottom-right (Z=0, projection)
                [x1, y1, 0]        # Bottom-left (Z=0, projection)
            ]
            verts.append(quad)
            
            # Color intensity based on Z value (higher = more saturated)
            avg_z = (z1 + z2) / 2
            if model_name == 'LOKI':
                # LOKI: Blue gradient based on emergence height - HIGHER OPACITY for visibility
                intensity = min(1.0, max(0.5, avg_z / 35))
                face_colors.append((0.12, 0.53, 0.90, 0.65 * intensity + 0.2))
            else:
                rgba = plt.matplotlib.colors.to_rgba(colors[model_name], 0.4 if model_name == "FT-Encoder" else 0.45)
                face_colors.append(rgba)
        
        # Add the ribbon surface with enhanced visibility
        ribbon_alpha = 0.75 if model_name == 'LOKI' else 0.5
        ribbon = Poly3DCollection(verts, facecolors=face_colors, 
                                  edgecolors=color, linewidths=0.5 if model_name == 'LOKI' else 0.3, 
                                  alpha=ribbon_alpha)
        ax.add_collection3d(ribbon)
        
        # =====================================================================
        # Main 3D trajectory LINE on top of the ribbon
        # X=Training Δ, Y=Epoch, Z=Emergent Δ (HEIGHT!)
        # =====================================================================
        lw = 4.5 if model_name == 'LOKI' else 2.5
        ax.plot(delta_acc, epochs, delta_ap, 
                color=color, linewidth=lw, alpha=1.0, 
                zorder=10 if model_name == 'LOKI' else 5,
                label=model_name)
        
        # =====================================================================
        # Shadow projection on Z=0 plane (what training does without emergence)
        # =====================================================================
        ax.plot(delta_acc, epochs, np.zeros_like(epochs), 
                color=color, linewidth=1.5, alpha=0.35, linestyle=':', zorder=2)
        
        # Start and end markers
        ax.scatter([delta_acc[0]], [epochs[0]], [delta_ap[0]], 
                   s=100, color=color, marker='o', edgecolors='white', linewidths=1.5, 
                   zorder=11, depthshade=False)
        
        marker = '*' if model_name == 'LOKI' else 'o'
        size = 450 if model_name == 'LOKI' else 150
        ax.scatter([delta_acc[-1]], [epochs[-1]], [delta_ap[-1]], 
                   s=size, color=color, marker=marker, edgecolors='white', linewidths=2, 
                   zorder=12, depthshade=False)
        
        # =====================================================================
        # KEY: Vertical lines showing the Training→Emergence relationship
        # These connect the training-only projection to the actual trajectory
        # =====================================================================
        key_epochs_idx = [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]
        key_epochs_idx = list(set([min(i, n_points-1) for i in key_epochs_idx]))
        
        for i in key_epochs_idx:
            # Vertical line from Z=0 to actual Z
            ax.plot([delta_acc[i], delta_acc[i]], [epochs[i], epochs[i]], 
                    [0, delta_ap[i]], color=color, linewidth=2, alpha=0.6, linestyle='--', zorder=3)
        
        # =====================================================================
        # Annotations showing the Training→Emergence relationship
        # =====================================================================
        final_acc = delta_acc[-1]
        final_ap = delta_ap[-1]
        final_epoch = epochs[-1]
        
        # Calculate the "emergence ratio" - how much emergence per training improvement
        if abs(final_acc) > 1:
            emergence_ratio = final_ap / final_acc
        else:
            emergence_ratio = final_ap
        
        # Simplified annotations - only LOKI gets prominent annotation
        if model_name == "LOKI":
            ax.text(final_acc + 3, final_epoch - 2, final_ap + 5,
                    f'LOKI ★\n+{final_ap:.0f}% Emergence',
                    fontsize=12, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=color, 
                             edgecolor='white', alpha=0.95, linewidth=2),
                    ha='left', va='bottom', zorder=20)
        
        print(f"   📍 {model_name}: Training Δ={final_acc:+.1f}%, Emergent Δ={final_ap:+.1f}%, Ratio={emergence_ratio:.2f}×")
    
    # =========================================================================
    # Draw diagonal reference line showing "1:1 Training→Emergence"
    # This helps visualize that LOKI has HIGHER than 1:1 transfer
    # =========================================================================
    diag_range = np.linspace(0, min(x_max, z_max), 20)
    ax.plot(diag_range, np.ones_like(diag_range) * max_epoch/2, diag_range, 
            color='gray', linewidth=1.5, linestyle='--', alpha=0.5, zorder=1)
    ax.text(diag_range[-1]*0.7, max_epoch/2, diag_range[-1]*0.7 + 3, 
            '1:1 Transfer', fontsize=8, color='gray', alpha=0.7, ha='center')
    
    # =========================================================================
    # Axis configuration
    # =========================================================================
    ax.set_xlim(x_min - 2, x_max + 5)
    ax.set_ylim(0, max_epoch + 3)
    ax.set_zlim(z_min - 3, z_max + 8)
    
    # Axis labels - NOW showing the relationship
    ax.set_xlabel('\nΔ Training Task (%)\n(Table-Text Accuracy)', fontsize=11, fontweight='bold', labelpad=12)
    ax.set_ylabel('\nTraining Epochs', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('\nΔ Emergent Task (%)\n(Row-Sentence AP)', fontsize=11, fontweight='bold', labelpad=10)
    
    # View angle
    ax.view_init(elev=elev, azim=azim)
    
    # Style the panes
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.5))
    ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.5))
    ax.zaxis.pane.set_facecolor((0.95, 0.95, 1.0, 0.5))
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # =========================================================================
    # Legend
    # =========================================================================
    legend_elements = []
    for model_name in get_present_models_for_panels(models):
        if model_name == "LOKI":
            label = "LOKI (High Transfer)"
            linewidth = 4
        elif model_name == "FT-Encoder":
            label = "FT-Encoder (Low Transfer)"
            linewidth = 2.5
        else:
            label = f"{model_name} (Low Transfer)"
            linewidth = 2.5
        legend_elements.append(Line2D([0], [0], color=colors[model_name], linewidth=linewidth, label=label))
    legend_elements.append(
        Line2D([0], [0], color=colors['Baseline'], linewidth=0, marker='s', markersize=10,
               alpha=0.3, label='Zero Emergence Plane')
    )
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)
    
    # Title - cleaner, more impactful
    ax.set_title('Emergent Capability from Global Supervision\n' +
                 '★ Only LOKI transfers training gains to local alignment',
                 fontsize=15, fontweight='bold', pad=25, color='#1a1a1a')
    
    # Add explanatory text
    fig.text(0.02, 0.02, 
             'Surface height (Z) shows emergent ability gain at each training improvement level (X).\n'
             'LOKI: Rising surface = training gain translates to emergence. Others: Flat = no transfer.',
             fontsize=9, style='italic', color='#555555',
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DDD', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "global_to_local_transfer_3d"
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training-Emergence Relationship 3D plot saved to: {output_path}.png and {output_path}.pdf")


def plot_emergence_over_epochs(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (12, 5.5)  # Increased height to give bars more room
) -> None:
    """
    Show how LOKI's emergent ability develops over training epochs.
    
    Three synchronized line plots:
    - Left: Training signal (Table-Text Accuracy) - all models improve
    - Middle: Emergent ability (Row-Sentence F1) - only LOKI improves
    - Right: Emergent ability (Row-Sentence AP) - only LOKI improves
    
    Uses dual y-axes to show relative improvement from baseline.
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    
    print(f"\n🌟 Generating Emergence Over Epochs Plot (3-panel)...")
    
    baseline_acc = baseline.get("frozen_encoder_accuracy", 0.0)
    baseline_ap = baseline.get("frozen_encoder_row_sent_ap", 0.0)
    baseline_row_acc = baseline.get("frozen_encoder_row_sent_acc", 0.0)
    
    # Styling sizes (larger text for paper, compact figure)
    title_fs = 13
    label_fs = 13
    tick_fs = 12
    legend_fs = 12

    # Colors and styles - use shared color mapping for consistency across figures
    model_names = get_present_models_for_panels(models)
    colors = {name: get_model_color(name) for name in model_names}
    baseline_color = get_model_color('Baseline')
    linestyles = {name: MODEL_LINE_STYLES.get(name, '-') for name in model_names}
    linewidths = {name: (3 if name == 'LOKI' else 2) for name in model_names}
    
    # Apply font defaults for clearer paper figures (keeps look & feel from post_training)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = label_fs
    plt.rcParams['axes.linewidth'] = 1.2

    # Create 2x3 subplot grid: top row for line plots, bottom row for bar charts
    fig = plt.figure(figsize=figsize)
    # Increase vertical space for bottom row (bar charts) to allow thicker bars and bigger gaps
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.8], hspace=0.35, wspace=0.35)
    # Reserve modest space at the bottom so labels remain visible
    fig.subplots_adjust(bottom=0.14)
    
    # Top row: line plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Bottom row: bar charts (minimal)
    ax1_bar = fig.add_subplot(gs[1, 0])
    ax2_bar = fig.add_subplot(gs[1, 1])
    ax3_bar = fig.add_subplot(gs[1, 2])
    
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)  # Remove bottom spine for cleaner look
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        # Make grids clearly visible above background patches
        ax.set_axisbelow(False)
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.45, zorder=5)
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.25, zorder=4)
    
    # Minimal styling for bar charts
    for ax in [ax1_bar, ax2_bar, ax3_bar]:
        ax.set_facecolor('white')
        # Remove all spines for minimal look
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=tick_fs)
    
    # Store final improvements for bar charts (using best values, not last epoch)
    final_improvements = {
        'table_text': {},
        'row_sent_acc': {},
        'row_sent_ap': {}
    }
    # Use precomputed best values for each model (consistent with other plots)
    for model_name in model_names:
        if model_name not in models:
            continue
        model_data = models[model_name]
        best_val_acc = model_data.get("best_accuracy", 0.0)
        best_row_acc = model_data.get("best_test_overall_accuracy", 0.0)
        best_row_ap = model_data.get("best_test_avg_precision", 0.0)
        final_improvements['table_text'][model_name] = ((best_val_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc else 0.0
        final_improvements['row_sent_acc'][model_name] = ((best_row_acc - baseline_row_acc) / baseline_row_acc) * 100 if baseline_row_acc else 0.0
        final_improvements['row_sent_ap'][model_name] = ((best_row_ap - baseline_ap) / baseline_ap) * 100 if baseline_ap else 0.0
    
    # ===== LEFT: Training Task (Table-Text Accuracy) =====
    for model_name in model_names:
        if model_name not in models:
            continue
        curves = models[model_name]["curves"]
        epochs = np.array(curves["epochs"])
        val_acc = np.array(curves["val_accuracy"])
        # Convert to % improvement from baseline (for line plot)
        improvement = ((val_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc else np.zeros_like(val_acc)
        ax1.plot(epochs, improvement, color=colors[model_name], 
                linestyle=linestyles[model_name], linewidth=linewidths[model_name],
                label=model_name, marker='o' if model_name == 'LOKI' else None,
                markersize=4, markevery=3)
    
    ax1.axhline(y=0, color='#E74C3C', linewidth=2, linestyle='-', alpha=0.7, label='Baseline')
    ax1.set_xlabel('Epoch', fontsize=label_fs)
    ax1.set_title('Table-Text Matching', fontsize=title_fs, fontweight='bold', pad=8)
    ax1.set_ylabel('Acc. Improvement (%)', fontsize=label_fs, labelpad=12)
    # Compress negative range so the plot is focused and readable for paper
    ax1.set_ylim(-20, 20)
    ax1.set_yticks(np.arange(-20, 21, 10))
    
    # Add shading below baseline
    # Shade only the meaningful negative region and place label inside it
    ax1.axhspan(-20, 0, alpha=0.08, color='red', zorder=1)
    ax1.text(0.5, 0.35, 'Below Baseline', transform=ax1.transAxes, ha='center', va='bottom',
             fontsize=max(10, tick_fs-1), color='#E74C3C', fontweight='normal', style='italic', alpha=1)
    
    # ===== MIDDLE: Emergent Task (Row-Sentence F1) =====
    for model_name in model_names:
        if model_name not in models:
            continue
        curves = models[model_name]["curves"]
        epochs = np.array(curves["epochs"])
        row_acc = np.array(curves["row_sent_overall_accuracy"])
        # Convert to % improvement from baseline (for line plot)
        improvement = ((row_acc - baseline_row_acc) / baseline_row_acc) * 100 if baseline_row_acc else np.zeros_like(row_acc)
        ax2.plot(epochs, improvement, color=colors[model_name], 
                linestyle=linestyles[model_name], linewidth=linewidths[model_name],
                label=model_name, marker='o' if model_name == 'LOKI' else None,
                markersize=4, markevery=3)
    
    ax2.axhline(y=0, color='#E74C3C', linewidth=2, linestyle='-', alpha=0.7, label='Baseline')
    ax2.set_xlabel('Epoch', fontsize=label_fs)
    ax2.set_title('Row-Sentence Alignment', fontsize=title_fs, fontweight='bold', pad=12)
    ax2.set_ylabel('F1 Improvement (%)', fontsize=label_fs)
    ax2.set_ylim(-40, 100)  # Set consistent Y-axis scale for emergent tasks
    ax2.set_yticks(np.arange(-40, 101, 20))
    
    # Add shading below baseline
    ax2.axhspan(-40, 0, alpha=0.08, color='red', zorder=1)
    ax2.text(0.5, 0.1, 'Below Baseline', transform=ax2.transAxes, ha='center', va='bottom',
             fontsize=max(10, tick_fs-1), color='#E74C3C', fontweight='normal', style='italic', alpha=1)
    
    # ===== RIGHT: Emergent Task (Row-Sentence AP) =====
    for model_name in model_names:
        if model_name not in models:
            continue
        curves = models[model_name]["curves"]
        epochs = np.array(curves["epochs"])
        row_ap = np.array(curves["row_sent_avg_precision"])
        # Convert to % improvement from baseline (for line plot)
        improvement = ((row_ap - baseline_ap) / baseline_ap) * 100 if baseline_ap else np.zeros_like(row_ap)
        ax3.plot(epochs, improvement, color=colors[model_name], 
                linestyle=linestyles[model_name], linewidth=linewidths[model_name],
                label=model_name, marker='o' if model_name == 'LOKI' else None,
                markersize=4, markevery=3)
    
    ax3.axhline(y=0, color='#E74C3C', linewidth=2, linestyle='-', alpha=0.7, label='Baseline')
    ax3.set_xlabel('Epoch', fontsize=label_fs)
    ax3.set_title('Row-Sentence Alignment', fontsize=title_fs, fontweight='bold', pad=12)
    ax3.set_ylabel('AP Improvement (%)', fontsize=label_fs)
    ax3.set_ylim(-40, 40)  # Set consistent Y-axis scale for emergent tasks
    ax3.set_yticks(np.arange(-40, 41, 20))
    
    # Add shading below baseline
    ax3.axhspan(-40, 0, alpha=0.08, color='red', zorder=1)
    ax3.text(0.5, 0.2, 'Below Baseline', transform=ax3.transAxes, ha='center', va='bottom',
             fontsize=max(10, tick_fs-1), color='#E74C3C', fontweight='normal', style='italic', alpha=1)
    
    # ===== BAR CHARTS: Final Improvements at Epoch 20 (Horizontal, Minimal) =====
    model_names = get_present_models_for_panels(models)
    # Keep bars visually thick and add a healthy gap between them
    bar_height = 0.4
    bar_gap = 0.15
    y_pos = np.arange(len(model_names)) * (bar_height + bar_gap)
    
    # Bar chart 1: Table-Text Accuracy (Horizontal, Minimal) - Can go negative
    values1 = [final_improvements['table_text'].get(m, 0) for m in model_names]
    bars1 = ax1_bar.barh(y_pos, values1, bar_height, color=[colors[m] for m in model_names], alpha=0.85)
    ax1_bar.axvline(x=0, color='#E74C3C', linewidth=1.5, linestyle='-', alpha=0.7, zorder=0)  # Red baseline
    ax1_bar.set_yticks(y_pos)
    ax1_bar.set_yticklabels(model_names, fontsize=tick_fs, ha='left')
    ax1_bar.tick_params(axis='y', pad=55)  # Left-align text starting 55 points left of axis
    ax1_bar.tick_params(axis='both', length=0)  # No tick marks
    ax1_bar.set_xlabel('')  # No X-axis label
    ax1_bar.invert_yaxis()  # Top to bottom follows model_names order
    # Remove X-axis
    ax1_bar.spines['bottom'].set_visible(False)
    ax1_bar.set_xticks([])
    # Center horizontally with less padding (wider bars)
    min_val1, max_val1 = min(values1 + [0]), max(values1 + [0])
    range1 = max_val1 - min_val1
    padding1 = range1 * 0.1  # Reduced to 10% padding for wider bars
    ax1_bar.set_xlim(min_val1 - padding1, max_val1 + padding1)
    
    # Add value labels INSIDE bars with white color
    for i, (bar, val) in enumerate(zip(bars1, values1)):
        width = bar.get_width()
        # Place text inside the bar or to the right of baseline to prevent overlapping Y labels
        x_pos_text = width / 2 if abs(width) > 5 else (width + 2 if width > 0 else 2)
        text_color = 'white' if abs(width) > 5 else colors[model_names[i]]
        ha = 'center' if abs(width) > 5 else 'left'
        ax1_bar.text(x_pos_text, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%', ha=ha, va='center',
                fontsize=max(10, tick_fs-1), fontweight='bold', color=text_color)
    
    # (Label moved to the right side of the figure)
    
    # Bar chart 2: Row-Sentence F1 (Horizontal, Minimal) - Start at 0
    values2 = [final_improvements['row_sent_acc'].get(m, 0) for m in model_names]
    bars2 = ax2_bar.barh(y_pos, values2, bar_height, color=[colors[m] for m in model_names], alpha=0.85)
    ax2_bar.axvline(x=0, color='#E74C3C', linewidth=1.5, linestyle='-', alpha=0.7, zorder=0)  # Red baseline
    ax2_bar.set_yticks(y_pos)
    ax2_bar.set_yticklabels([])  # Remove redundant labels to save space
    ax2_bar.tick_params(axis='both', length=0)
    ax2_bar.set_xlabel('')
    ax2_bar.invert_yaxis()
    # Remove X-axis
    ax2_bar.spines['bottom'].set_visible(False)
    ax2_bar.set_xticks([])
    # Center horizontally with less padding (wider bars)
    max_val2 = max(values2)
    padding2 = max_val2 * 0.1  # Reduced to 10% padding for wider bars
    ax2_bar.set_xlim(-padding2, max_val2 + padding2)
    
    # Add value labels INSIDE bars with white color
    for i, (bar, val) in enumerate(zip(bars2, values2)):
        width = bar.get_width()
        x_pos_text = width / 2 if width > 5 else (width + 2 if width > 0 else 2)
        text_color = 'white' if width > 5 else colors[model_names[i]]
        ha = 'center' if width > 5 else 'left'
        ax2_bar.text(x_pos_text, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%', ha=ha, va='center',
                fontsize=max(10, tick_fs-1), fontweight='bold', color=text_color)
    
    # (Label moved to the right side of the figure)
    
    # Bar chart 3: Row-Sentence AP (Horizontal, Minimal) - Start at 0
    values3 = [final_improvements['row_sent_ap'].get(m, 0) for m in model_names]
    bars3 = ax3_bar.barh(y_pos, values3, bar_height, color=[colors[m] for m in model_names], alpha=0.85)
    ax3_bar.axvline(x=0, color='#E74C3C', linewidth=1.5, linestyle='-', alpha=0.7, zorder=0)  # Red baseline
    ax3_bar.set_yticks(y_pos)
    ax3_bar.set_yticklabels([])  # Remove redundant labels to save space
    ax3_bar.tick_params(axis='both', length=0)
    ax3_bar.set_xlabel('')
    ax3_bar.invert_yaxis()
    # Remove X-axis
    ax3_bar.spines['bottom'].set_visible(False)
    ax3_bar.set_xticks([])
    # Center horizontally with less padding (wider bars)
    max_val3 = max(values3)
    padding3 = max_val3 * 0.1  # Reduced to 10% padding for wider bars
    ax3_bar.set_xlim(-padding3, max_val3 * 1.35)  # Leave substantial room on the right for the vertical label
    
    # Add value labels INSIDE bars with white color
    for i, (bar, val) in enumerate(zip(bars3, values3)):
        width = bar.get_width()
        model = model_names[i]
        # Place FT-Encoder label outside the bar for readability (as requested)
        if model == 'FT-Encoder':
            x_pos_text = width + 2 if width >= 0 else 2
            text_color = colors[model]
            ha = 'left'
        else:
            x_pos_text = width / 2 if abs(width) > 5 else (width + 2 if width > 0 else 2)
            text_color = 'white' if abs(width) > 5 else colors[model]
            ha = 'center' if abs(width) > 5 else 'left'

        ax3_bar.text(x_pos_text, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%', ha=ha, va='center',
                fontsize=max(10, tick_fs-1), fontweight='bold', color=text_color)
    
    # Add a single unified vertical label on the far right, inside the right boundary
    ax3_bar.text(0.9, 1.0, 'Max/Min Δ (%)', transform=ax3_bar.transAxes,
                ha='center', va='top', rotation=-90, fontsize=label_fs, fontweight='normal')
    
    # Create unified legend at the bottom center
    handles1, labels1 = ax1.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles1, labels1):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    # Add unified legend below the plots (centered, closer to bar charts)
    # Legend is intentionally removed for this figure (colors are encoded in bars)
    # fig.legend(unique_handles, unique_labels, loc='upper center', 
    #           bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=legend_fs, 
    #           frameon=False)
    
    # Save plot
    output_path = Path(output_dir) / "emergence_over_epochs"
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Emergence over epochs (3-panel) plot saved to: {output_path}.png and {output_path}.pdf")


def plot_emergent_ability_delta(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (7, 6)
) -> None:
    """
    Create a visualization showing relative improvement (delta) from baseline
    on both Table-Text and Row-Sentence tasks.
    
    This normalizes all models to start from (0, 0) and shows their improvement
    trajectory, making LOKI's emergent ability even more stark.
    
    Args:
        combined_data: Combined comparison data dict
        combined_json_path: Path to combined JSON file
        output_dir: Directory to save the plots
        figsize: Figure size
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    model_names = get_present_models(models)
    
    print(f"\n🎨 Generating Emergent Ability Delta Plot for {len(model_names)} models...")
    
    # Get baseline values
    baseline_acc = baseline.get("frozen_encoder_accuracy", 0.0)
    baseline_ap = baseline.get("frozen_encoder_row_sent_ap", 0.0)
    
    # Conference-style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.linewidth'] = 1.5
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    
    # Markers and styles
    MARKERS = MODEL_MARKERS
    LINE_STYLES = MODEL_LINE_STYLES
    
    # Plot origin (baseline = 0, 0)
    ax.scatter([0], [0], color='#E74C3C', s=200, marker='X', 
               zorder=10, edgecolors='white', linewidths=2, label='Baseline')
    ax.annotate('Baseline\n(0%, 0%)', xy=(0, 0), xytext=(5, -12), 
                textcoords='offset points', fontsize=10, color='#E74C3C', fontweight='bold')
    
    # Track max values for axis limits
    max_delta_acc = 0
    max_delta_ap = 0
    min_delta_ap = 0
    
    for model_name in model_names:
        model_data = models[model_name]
        curves = model_data["curves"]
        color = get_model_color(model_name, 0)
        marker = MARKERS.get(model_name, 'o')
        linestyle = LINE_STYLES.get(model_name, '-')
        
        # Get trajectories and compute deltas (percentage improvement from baseline)
        val_acc = np.array(curves["val_accuracy"])
        row_sent_ap = np.array(curves["row_sent_avg_precision"])
        
        # Compute relative improvement (percentage)
        delta_acc = ((val_acc - baseline_acc) / baseline_acc) * 100
        delta_ap = ((row_sent_ap - baseline_ap) / baseline_ap) * 100
        
        # Update limits
        max_delta_acc = max(max_delta_acc, np.max(delta_acc))
        max_delta_ap = max(max_delta_ap, np.max(delta_ap))
        min_delta_ap = min(min_delta_ap, np.min(delta_ap))
        
        n_points = len(delta_acc)
        
        # Plot trajectory with gradient
        for i in range(n_points - 1):
            alpha = 0.3 + 0.7 * (i / max(n_points - 1, 1))
            lw = 2 + 2 * (i / max(n_points - 1, 1))
            ax.plot(delta_acc[i:i+2], delta_ap[i:i+2], 
                    color=color, linewidth=lw, alpha=alpha, 
                    linestyle=linestyle, zorder=3)
        
        # Start point
        ax.scatter([delta_acc[0]], [delta_ap[0]], color='white', s=120, 
                   marker='o', edgecolors=color, linewidths=2, zorder=5)
        
        # End point
        marker_size = 350 if model_name == "LOKI" else 180
        ax.scatter([delta_acc[-1]], [delta_ap[-1]], color=color, s=marker_size, 
                   marker=marker, edgecolors='white', linewidths=2, zorder=6,
                   label=f'{model_name}')
        
        # Arrow at 60%
        arrow_idx = max(1, int(n_points * 0.6))
        if arrow_idx < n_points - 1:
            dx = delta_acc[arrow_idx + 1] - delta_acc[arrow_idx]
            dy = delta_ap[arrow_idx + 1] - delta_ap[arrow_idx]
            ax.annotate('', 
                xy=(delta_acc[arrow_idx] + dx * 0.5, delta_ap[arrow_idx] + dy * 0.5),
                xytext=(delta_acc[arrow_idx], delta_ap[arrow_idx]),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2, 
                               mutation_scale=15), zorder=7)
        
        # Annotate final improvement
        if model_name == "LOKI":
            ax.annotate(f'{model_name}\n(+{round_half_up(delta_acc[-1], 1):.1f}%, +{round_half_up(delta_ap[-1], 1):.1f}%)',
                        xy=(delta_acc[-1], delta_ap[-1]), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=11, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=color, alpha=0.9))
        else:
            offset_y = -20 if model_name == "FT-Encoder" else 8
            ax.annotate(f'{model_name}\n(+{round_half_up(delta_acc[-1], 1):.1f}%, {round_half_up(delta_ap[-1], 1):+.1f}%)',
                        xy=(delta_acc[-1], delta_ap[-1]), 
                        xytext=(8, offset_y), textcoords='offset points',
                        fontsize=9, color=color)
    
    # Add reference lines
    ax.axhline(y=0, color='#E74C3C', linestyle=':', alpha=0.4, linewidth=1.5, zorder=1)
    ax.axvline(x=0, color='#E74C3C', linestyle=':', alpha=0.4, linewidth=1.5, zorder=1)
    
    # Add quadrant labels
    ax.text(max_delta_acc * 0.7, max_delta_ap * 0.85, 
            'EMERGENT\nLEARNING', fontsize=11, color='green', alpha=0.6,
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    ax.text(max_delta_acc * 0.7, min_delta_ap * 0.5, 
            'Training-only\nImprovement', fontsize=9, color='gray', alpha=0.5,
            ha='center', va='center', style='italic')
    
    # Customize axes
    ax.set_xlabel('Δ Table-Text Matching (%)\n(Improvement from Baseline)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ Row-Sentence AP (%)\n(Emergent Improvement)', fontsize=12, fontweight='bold')
    ax.set_title('Emergent Learning: Relative Improvement from Baseline', 
                 fontsize=13, fontweight='bold', pad=12)
    
    # Set symmetric-ish limits
    ax.set_xlim(-2, max_delta_acc * 1.15)
    ax.set_ylim(min_delta_ap * 1.3, max_delta_ap * 1.15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "emergent_ability_delta"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Emergent ability delta plot saved to: {output_path}.png and {output_path}.pdf")


def plot_emergent_gap_dual_line(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (12, 4)
) -> None:
    """
    DUAL-LINE EMERGENT GAP VISUALIZATION
    
    The clearest visualization of LOKI's emergent ability:
    - X-axis: Epochs
    - Y-axis: % improvement over baseline (normalized)
    - Two lines per model: Table-Text Matching (solid) + Row-Sentence AP (dashed)
    - SHADED AREA between lines shows the "emergent learning gap"
    
    Key visual insight:
    - LOKI: Both lines rise together → shaded area shows EMERGENT learning
    - Others: Only Table-Text rises, Row-Sentence stays flat → NO emergence
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    
    print(f"\n🌟 Generating Dual-Line Emergent Gap Visualization...")
    
    baseline_acc = baseline.get("frozen_encoder_accuracy", 0.0)
    baseline_ap = baseline.get("frozen_encoder_row_sent_ap", 0.0)
    
    # Model order and colors
    model_names = get_present_models_for_panels(models)
    colors = {name: get_model_color(name) for name in model_names}

    panel_width = max(figsize[0], 3.5 * max(len(model_names), 1))
    fig, axes = plt.subplots(1, len(model_names), figsize=(panel_width, figsize[1]), sharey=True)
    if len(model_names) == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if model_name not in models:
            continue
            
        curves = models[model_name]["curves"]
        epochs = np.array(curves["epochs"])
        val_acc = np.array(curves["val_accuracy"])
        row_ap = np.array(curves["row_sent_avg_precision"])
        
        # Convert to % improvement from baseline
        acc_improvement = ((val_acc - baseline_acc) / baseline_acc) * 100
        ap_improvement = ((row_ap - baseline_ap) / baseline_ap) * 100
        
        color = colors[model_name]
        
        # Plot Table-Text Matching (solid line, no markers)
        ax.plot(epochs, acc_improvement, color=color, linestyle='-', linewidth=2.5,
                label='Table-Text Match', alpha=0.9)
        
        # Plot Row-Sentence AP (dashed line, no markers)
        ax.plot(epochs, ap_improvement, color=color, linestyle='--', linewidth=2.5,
                label='Row-Sentence AP', alpha=0.9)
        
        # SHADE THE AREA BETWEEN THE TWO LINES - This is the "emergent gap"
        # For LOKI: positive area (both rise together)
        # For others: area shows divergence (only Table-Text rises)
        if model_name == "LOKI":
            # LOKI: Fill with green to show emergent learning
            ax.fill_between(epochs, acc_improvement, ap_improvement, 
                          alpha=0.3, color='#27AE60', 
                          label='Emergent Gap ★')
            # Add annotation
            ax.text(0.5, 0.92, '★ EMERGENT LEARNING', transform=ax.transAxes,
                   ha='center', va='top', fontsize=9, fontweight='bold', color='#27AE60',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5', 
                            edgecolor='#27AE60', alpha=0.95, linewidth=1.5))
        else:
            # Others: Fill with red to show NO emergent learning
            ax.fill_between(epochs, acc_improvement, ap_improvement,
                          alpha=0.15, color='#E74C3C',
                          label='No Emergence')
            # Add annotation
            ax.text(0.5, 0.92, '✗ No local learning', transform=ax.transAxes,
                   ha='center', va='top', fontsize=9, fontweight='bold', color='#C0392B',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', 
                            edgecolor='#C0392B', alpha=0.95, linewidth=1))
        
        # Baseline reference line
        ax.axhline(y=0, color='#7F8C8D', linewidth=1.5, linestyle=':', alpha=0.7)
        
        # Styling
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Improvement over Baseline (%)', fontsize=11, fontweight='bold')
        ax.set_title(model_name, fontsize=13, fontweight='bold', color=color, pad=10)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(0, max(epochs))
        
        # Legend
        ax.legend(loc='lower right', fontsize=8, framealpha=0.95, ncol=1)
    
    # Main title
    fig.suptitle('Emergent Local Alignment from Global Supervision', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "emergent_gap_dual_line"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Emergent gap dual-line plot saved to: {output_path}.png and {output_path}.pdf")


def plot_combined_metrics_comparison(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (7, 4.5)
) -> None:
    """
    Create a single compact plot combining all 3 test metrics for publication.
    
    Conference-quality visualization (ICML/NeurIPS/ICLR/VLDB style):
    - Grouped horizontal bar chart showing best values per model
    - Compact single-figure format ideal for paper space constraints
    - Clear baseline comparison with improvement percentages
    - Professional styling suitable for camera-ready submissions
    
    Args:
        combined_data: Combined comparison data dict (if already loaded)
        combined_json_path: Path to combined JSON file (loads if combined_data is None)
        output_dir: Directory to save the plots
        figsize: Figure size as (width, height) - default sized for single column
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    model_names = get_present_models(models)
    
    print(f"\n🎨 Generating Combined Metrics Plot (compact) for {len(model_names)} models...")
    
    # Conference-style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    
    # Define metrics to plot
    metrics = [
        ('Table-Text\nMatching', 'best_accuracy', 'frozen_encoder_accuracy'),
        ('Row-Sent\nF1', 'best_test_overall_accuracy', 'frozen_encoder_row_sent_acc'),
        ('Row-Sent\nAvg Precision', 'best_test_avg_precision', 'frozen_encoder_row_sent_ap'),
    ]
    
    # Colors for models (consistent with other plots)
    colors = {name: get_model_color(name, i) for i, name in enumerate(model_names)}
    colors['Baseline'] = '#E74C3C'  # Red for baseline
    
    # Patterns/hatches for B&W printing compatibility
    hatches = ['', '//', '\\\\', 'xx', '..', 'oo']
    
    # Bar positioning
    n_metrics = len(metrics)
    n_models = len(model_names) + 1  # +1 for baseline
    bar_height = 0.15
    group_spacing = 0.3
    
    # Y positions for each metric group
    y_positions = np.arange(n_metrics) * (n_models * bar_height + group_spacing)
    
    # Plot baseline bars first
    baseline_values = []
    for metric_label, model_key, baseline_key in metrics:
        baseline_values.append(baseline.get(baseline_key, 0.0))
    
    # Reverse order so Table-Text Matching is at top
    y_positions_rev = y_positions[::-1]
    baseline_values_rev = baseline_values[::-1]
    metrics_rev = metrics[::-1]
    
    # Plot baseline
    baseline_y = y_positions_rev + (n_models - 1) * bar_height / 2
    bars_baseline = ax.barh(baseline_y, baseline_values_rev, height=bar_height * 0.9,
                            color=colors['Baseline'], alpha=0.7, label='Baseline (Frozen)',
                            edgecolor='white', linewidth=1)
    
    # Add value labels for baseline
    for i, (y, val) in enumerate(zip(baseline_y, baseline_values_rev)):
        ax.text(val + 0.01, y, f'{round_half_up(val, 2):.2f}', va='center', ha='left',
                fontsize=9, color=colors['Baseline'], fontweight='normal')
    
    # Plot each model's bars
    for model_idx, model_name in enumerate(model_names):
        model_data = models[model_name]
        model_values = []
        for metric_label, model_key, baseline_key in metrics:
            model_values.append(model_data.get(model_key, 0.0))
        
        model_values_rev = model_values[::-1]
        
        # Y position for this model within each group
        model_y = y_positions_rev - (model_idx + 0.5) * bar_height + (n_models - 1) * bar_height / 2
        
        bars = ax.barh(model_y, model_values_rev, height=bar_height * 0.9,
                       color=colors[model_name], alpha=0.85, label=model_name,
                       edgecolor='white', linewidth=1,
                       hatch=hatches[model_idx % len(hatches)] if model_idx > 0 else '')
        
        # Add value labels and improvement percentages
        for i, (y, val, base_val) in enumerate(zip(model_y, model_values_rev, baseline_values_rev)):
            # Value label
            ax.text(val + 0.01, y, f'{round_half_up(val, 2):.2f}', va='center', ha='left',
                    fontsize=9, color=colors[model_name], fontweight='bold')
            
            # Improvement percentage for LOKI (best performing model)
            if model_name == "LOKI" and base_val > 0:
                improvement = ((val - base_val) / base_val) * 100
                if improvement > 0:
                    ax.text(val + 0.08, y, f'(+{round_half_up(improvement, 1):.1f}%)', 
                            va='center', ha='left', fontsize=8, color=colors[model_name],
                            fontweight='normal', style='italic')
    
    # Customize axes
    ax.set_yticks(y_positions_rev)
    ax.set_yticklabels([m[0] for m in metrics_rev], fontsize=11, fontweight='normal')
    ax.set_xlabel('Score', fontsize=12, fontweight='normal')
    ax.set_xlim(0, 1.05)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    
    # Add vertical gridlines
    ax.xaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend - compact, outside plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=n_models,
              fontsize=9, framealpha=0.95, edgecolor='gray', columnspacing=1)
    
    # Title
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold', pad=10)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "model_comparison_combined_metrics"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Combined metrics plot saved to: {output_path}.png and {output_path}.pdf")


def plot_combined_metrics_radar(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (5.2, 5.2)
) -> None:
    """
    Create a radar/spider chart combining all metrics for publication.
    
    Alternative visualization showing each model's profile across all metrics.
    Ideal for showing which model excels in which areas at a glance.
    
    Args:
        combined_data: Combined comparison data dict (if already loaded)
        combined_json_path: Path to combined JSON file (loads if combined_data is None)
        output_dir: Directory to save the plots
        figsize: Figure size as (width, height)
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    model_names = get_present_models(models)
    
    print(f"\n🎨 Generating Radar Chart for {len(model_names)} models...")
    
    # Define metrics (using shorter names for radar)
    metrics = [
        ('Table-Text\nMatching', 'best_accuracy', 'frozen_encoder_accuracy'),
        ('Row-Sent\nF1', 'best_test_overall_accuracy', 'frozen_encoder_row_sent_acc'),
        ('Row-Sent\nAvg Prec', 'best_test_avg_precision', 'frozen_encoder_row_sent_ap'),
    ]
    
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Conference-style settings and sizing to match other figures (compact for paper)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.linewidth'] = 1.2

    # Local sizing variables
    title_fs = 12
    label_fs = 15
    tick_fs = 13
    legend_fs = 13
    marker_loki = 10
    marker_other = 8
    model_linewidth = 3
    baseline_linewidth = 2
    fill_alpha = 0.12

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    
    # Colors
    colors = {name: get_model_color(name, i) for i, name in enumerate(model_names)}
    colors['Baseline'] = '#E74C3C'
    
    # Line styles for B&W
    linestyles = ['-', '--', '-.', ':']
    
    # Baseline raw values (for computing delta % improvement)
    baseline_raw = [baseline.get(m[2], 0.0) for m in metrics]
    baseline_raw = [max(v, 1e-9) for v in baseline_raw]  # Avoid division by zero
    
    # Plot baseline at 0% delta (reference = no improvement)
    baseline_delta = [0.0] * n_metrics
    baseline_delta_loop = baseline_delta + baseline_delta[:1]  # Complete the loop
    ax.plot(angles, baseline_delta_loop, color=colors['Baseline'], linewidth=baseline_linewidth,
            linestyle='--', label='Baseline', marker='o', markersize=marker_other)
    ax.fill(angles, baseline_delta_loop, color=colors['Baseline'], alpha=fill_alpha)
    
    # Plot each model as delta % improvement over baseline: (model - baseline) / baseline * 100
    all_deltas = [0.0]  # baseline
    for idx, model_name in enumerate(model_names):
        model_data = models[model_name]
        raw_values = [model_data.get(m[1], 0.0) for m in metrics]
        # Delta in %: e.g. +35 means 35% improvement over baseline for that metric
        deltas = [(raw_values[i] - baseline_raw[i]) / baseline_raw[i] * 100 for i in range(n_metrics)]
        all_deltas.extend(deltas)
        values_loop = deltas + deltas[:1]  # Complete the loop
        
        ax.plot(angles, values_loop, color=colors[model_name], linewidth=model_linewidth,
            linestyle=linestyles[idx % len(linestyles)], label=model_name,
            marker='*' if model_name == "LOKI" else 'o',
            markersize=marker_loki if model_name == "LOKI" else marker_other)
        ax.fill(angles, values_loop, color=colors[model_name], alpha=fill_alpha)
    
    # Radial limits first (needed for label placement)
    r_min = min(0, min(all_deltas)) - 2 if min(all_deltas) < 0 else 0
    r_max = max(max(all_deltas) * 1.05, 10)
    # Radial span used for padding calculations
    range_span = max(1e-6, r_max - r_min)
    ax.set_ylim(r_min, r_max)
    # Ticks in delta %: 0%, 10%, 20%, ...
    step = 10 if r_max <= 50 else (20 if r_max <= 100 else 25)
    y_ticks = list(range(0, int(r_max) + 1, step))
    if r_min < 0:
        y_ticks = [int(r_min)] + [t for t in y_ticks if t > 0]
    ax.set_yticks(y_ticks)
    # Hide default radial tick labels; we'll draw them manually offset inside the circles.
    ax.set_yticklabels([])
    try:
        default_tick_color = ax.yaxis.get_ticklabels()[0].get_color()
    except Exception:
        default_tick_color = '#555555'

    # Place radial tick labels (e.g., 10%,20%...) at a fixed angle offset
    # offset (degrees): negative moves clockwise; user requested -15° inside circles
    offset_deg = -30
    angle_pos = np.pi / 2 + np.deg2rad(offset_deg)
    # small inward padding so labels sit inside the gridline
    inward_pad = (r_max - r_min) * 1.05
    for t in y_ticks:
        if t == 0:
            continue
        # Place the tick label just outside the gridline and above it (higher zorder)
        outward_pad = max(range_span * 0.02, r_max * 0.02)
        r_label = t + outward_pad
        ax.text(angle_pos, r_label, f'{int(t)}%', fontsize=tick_fs, color=default_tick_color,
                ha='center', va='bottom', rotation=0, zorder=12)

    # Draw an explicit circular reference for 0% using the same style as
    # other radial gridlines, and place it behind other elements.
    try:
        # Try to copy the first radial gridline style so the 0% circle matches.
        gridlines = ax.yaxis.get_gridlines()
        if len(gridlines) > 0:
            tpl = gridlines[0]
            circle_color = tpl.get_color()
            circle_ls = tpl.get_linestyle()
            circle_lw = tpl.get_linewidth()
            circle_alpha = tpl.get_alpha() if tpl.get_alpha() is not None else 1.0
        else:
            circle_color = '#BFBFBF'
            circle_ls = '--'
            circle_lw = 0.8
            circle_alpha = 0.6

        theta_circle = np.linspace(0, 2 * np.pi, 360)
        r_zero = np.zeros_like(theta_circle)
        ax.plot(theta_circle, r_zero, color=circle_color, linewidth=circle_lw,
            linestyle=circle_ls, alpha=circle_alpha, zorder=0)
        # Add a 0% label on top of the circle that matches other radial tick labels.
        try:
            ticklabels = ax.yaxis.get_ticklabels()
            label_color = ticklabels[0].get_color() if len(ticklabels) > 0 else '#333333'
        except Exception:
            label_color = '#333333'
        # Place label at the top (90 degrees) slightly outside the 0 radius so it sits on the circle
        label_r = 0.0 + (r_max * 0.02)
        ax.text(np.pi / 2, label_r, '0%', color=label_color, fontsize=tick_fs,
                ha='center', va='bottom', zorder=5)
    except Exception:
        # Non-fatal: if polar plotting fails on a backend, skip gracefully
        pass
    
    # Hide default x tick labels (polar overrides rotation); draw metric labels manually with rotation
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    # Place labels just outside the outer grid: Table-Text 90°; Row-Sent Acc 30°; Row-Sent Prec -30°
    label_rotations_deg = [90, 30, -30]  # degrees from horizontal
    # Add padding so metric labels don't overlap the outer circle.
    # Increase multiplier slightly and add a small absolute padding based on range.
    padding_multiplier = 1.12
    extra_pad = range_span * 0.03
    label_radius = r_max * padding_multiplier + extra_pad
    for i, (angle_rad, (label_text, _, _)) in enumerate(zip(angles[:-1], metrics)):
        rot = label_rotations_deg[i] if i < len(label_rotations_deg) else 0
        # In polar, rotation is in degrees; use ha/va for alignment relative to (angle_rad, label_radius)
        ax.text(angle_rad, label_radius, label_text, fontsize=label_fs, fontweight='bold',
                rotation=rot, rotation_mode='anchor', ha='center', va='center')
    
    # Legend
    # ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.8), fontsize=legend_fs,
    #           framealpha=0.85, edgecolor='gray')
    
    # Title centered on the figure
    fig.suptitle('Multi-Metric Model Comparison (% Improvement over Baseline)', fontsize=title_fs, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "model_comparison_radar"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Radar chart saved to: {output_path}.png and {output_path}.pdf")


def plot_combined_metrics_table_style(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[float, float] = (8, 3.5)
) -> None:
    """
    Create a compact bar chart mimicking a visual table for publication.
    
    Conference-quality visualization inspired by VLDB/SIGMOD paper figures:
    - Vertical grouped bars for each metric
    - Very compact format for tight paper layouts
    - Clear value annotations
    - Best values highlighted with stars
    
    Args:
        combined_data: Combined comparison data dict (if already loaded)
        combined_json_path: Path to combined JSON file (loads if combined_data is None)
        output_dir: Directory to save the plots
        figsize: Figure size as (width, height)
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    baseline = combined_data["baseline"]
    model_names = get_present_models(models)
    
    print(f"\n🎨 Generating Table-Style Combined Plot for {len(model_names)} models...")
    
    # Conference-style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Define metrics
    metrics = [
        ('Table-Text\nMatching', 'best_accuracy', 'frozen_encoder_accuracy'),
        ('Row-Sent\nF1', 'best_test_overall_accuracy', 'frozen_encoder_row_sent_acc'),
        ('Row-Sent\nAvg Precision', 'best_test_avg_precision', 'frozen_encoder_row_sent_ap'),
    ]
    metric_labels = [m[0] for m in metrics]
    
    # Prepare data
    all_entities = ['Baseline'] + model_names
    n_entities = len(all_entities)
    n_metrics = len(metrics)
    
    # Colors
    colors = {name: get_model_color(name, i) for i, name in enumerate(model_names)}
    colors['Baseline'] = '#E74C3C'
    
    # Hatches for B&W compatibility
    hatches = MODEL_HATCHES
    
    # Bar positioning
    bar_width = 0.18
    x = np.arange(n_metrics)
    
    # Plot bars for each entity
    for i, entity in enumerate(all_entities):
        offset = (i - n_entities / 2 + 0.5) * bar_width
        
        if entity == 'Baseline':
            values = [baseline.get(m[2], 0.0) for m in metrics]
        else:
            values = [models[entity].get(m[1], 0.0) for m in metrics]
        
        bars = ax.bar(x + offset, values, bar_width * 0.9, 
                      color=colors[entity], alpha=0.85, label=entity,
                      edgecolor='white', linewidth=1,
                      hatch=hatches.get(entity, ''))
        
        # Add value labels on top of bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            # Check if this is the best value for this metric
            all_vals = [baseline.get(metrics[j][2], 0.0)] + [models[m].get(metrics[j][1], 0.0) for m in model_names]
            is_best = val == max(all_vals)
            
            y_pos = bar.get_height() + 0.02
            label_text = f'{round_half_up(val, 2):.2f}'
            
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, label_text,
                    ha='center', va='bottom', fontsize=8, fontweight='bold' if is_best else 'normal',
                    color=colors[entity], rotation=0)
            
            # Add star for best
            if is_best and entity == "LOKI":
                ax.text(bar.get_x() + bar.get_width()/2, y_pos + 0.06, '★',
                        ha='center', va='bottom', fontsize=10, color=colors[entity])
    
    # Customize axes
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight='normal')
    ax.set_ylabel('Score', fontsize=11, fontweight='normal')
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    
    # Grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend below plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_entities,
              fontsize=9, framealpha=0.95, edgecolor='gray')
    
    # Title
    ax.set_title('Model Performance Comparison (Best Values)', fontsize=12, fontweight='bold', pad=8)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "model_comparison_combined_bars"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Table-style bar chart saved to: {output_path}.png and {output_path}.pdf")


def plot_training_metrics_comparison(
    combined_data: Optional[Dict[str, Any]] = None,
    combined_json_path: Optional[str] = None,
    output_dir: str = "output_plots",
    figsize: Tuple[int, int] = (18, 5.5)
) -> None:
    """
    Plot training metrics comparison: Training Time (lollipop chart), Mean Loss with ±1 Std,
    and Loss vs Accuracy trajectory for multiple models.
    
    Conference-quality visualization (ICLR/NeurIPS/VLDB style):
    - Lollipop chart for training time comparison (model vs model)
    - Log scale for large time differences
    - Confidence bands for loss variance
    - Training trajectory shows optimization path
    
    Args:
        combined_data: Combined comparison data dict (if already loaded)
        combined_json_path: Path to combined JSON file (loads if combined_data is None)
        output_dir: Directory to save the plots
        figsize: Figure size as (width, height)
    """
    # Load data if not provided
    if combined_data is None:
        if combined_json_path is None:
            combined_json_path = f"{output_dir}/combined_comparison_data.json"
        print(f"📖 Loading combined data from: {combined_json_path}")
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
    
    combined_data = normalize_combined_data(combined_data)
    models = combined_data["models"]
    model_names = get_present_models(models)
    
    print(f"\n🎨 Generating Training Metrics Comparison Plot for {len(model_names)} models...")
    # Training-time specific font sizing (compact figure but larger text)
    tt_xtick_fs = 18
    tt_ytick_fs = 16
    tt_ylabel_fs = 20
    tt_title_fs = 20
    tt_total_annot_fs = 18
    tt_mean_annot_fs = 16
    
    # Conference-style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    
    # Line styles for B&W compatibility
    LINE_STYLES = ['-', '--', '-.', ':']
    MARKERS = ['*', 'D', 's', '^', 'v', 'o', 'p', 'h']
    MARKER_SIZES = [500, 160, 160, 160, 160, 160, 160, 160]  # Star needs to be much larger to visually match solid shapes
    
    # Create separate figure for training time
    fig_time, ax_time = plt.subplots(1, 1, figsize=(7, 6))
    fig_time.patch.set_facecolor('white')
    ax_time.set_facecolor('white')
    ax_time.tick_params(axis='both', which='major', labelsize=13, direction='in', length=4)
    ax_time.tick_params(axis='x', labelsize=tt_xtick_fs)
    ax_time.tick_params(axis='y', labelsize=tt_ytick_fs)

    # Create figure for loss and trajectory
    fig_metrics, axes_metrics = plt.subplots(1, 2, figsize=(14, 5.5))
    fig_metrics.patch.set_facecolor('white')
    for ax in axes_metrics:
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='major', labelsize=tt_ytick_fs, direction='in', length=4)
    
    # ==========================================================================
    # SUBPLOT 1: Training Time Comparison (Lollipop Chart - Model vs Model)
    # ==========================================================================
    # Collect training time statistics for each model
    time_stats = {}
    for model_name in model_names:
        curves = models[model_name]["curves"]
        epoch_times = np.array(curves["epoch_times"])
        # Skip epoch 0 if present
        epochs = np.array(curves["epochs"])
        if len(epochs) > 1 and epochs[0] == 0:
            epoch_times = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
        
        times_minutes = epoch_times / 60.0
        time_stats[model_name] = {
            'mean': np.mean(times_minutes),
            'std': np.std(times_minutes),
            'min': np.min(times_minutes),
            'max': np.max(times_minutes),
            'total': np.sum(times_minutes)
        }
    
    # Sort models by mean time for better visualization
    sorted_models = sorted(model_names, key=lambda x: time_stats[x]['mean'])
    
    # X positions for lollipop chart - compress horizontal spacing to reduce gaps
    x_spacing = 0.72
    x_positions = np.arange(len(sorted_models)) * x_spacing
    
    # Prepare lists to store annotations so we can place them after limits are set
    total_annots = []  # (x_pos, mean, total, color)
    mid_annots = []

    # Draw lollipop chart (vertical lines with markers and error bars)
    for i, model_name in enumerate(sorted_models):
        stats = time_stats[model_name]
        color = get_model_color(model_name, model_names.index(model_name))
        
        # LOKI always gets star, FT-Encoder gets diamond, others get from list
        if model_name == "LOKI":
            marker = '*'
            marker_size = 550  # Star needs larger size
        elif model_name == "FT-Encoder":
            marker = 'D'
            marker_size = 180
        else:
            marker_idx = model_names.index(model_name)
            marker = MARKERS[marker_idx % len(MARKERS)]
            marker_size = MARKER_SIZES[marker_idx % len(MARKER_SIZES)]
        
        # Draw vertical line from 0 to mean (thicker for compact figure)
        ax_time.plot([x_positions[i], x_positions[i]], [0.5, stats['mean']], color=color, linewidth=5,
                     solid_capstyle='round', zorder=3)

        # Draw marker at mean (make markers larger for visibility)
        ax_time.scatter([x_positions[i]], [stats['mean']], color=color, s=marker_size * 1.6,
                       marker=marker, edgecolors='white', linewidths=2.8, zorder=4)

        # Store TOTAL and mid annotations for placement after limits are set
        total_annots.append((x_positions[i], stats['mean'], stats['total'], color))
        # Store mid annotation (place now — will be safe as it's inside the bar)
        mid_y = np.sqrt(0.5 * stats['mean'])  # Geometric middle on log scale
        mid_annots.append((x_positions[i], mid_y, stats['mean'], color))
    
    # Customize subplot 1: Training Time Lollipop
    ax_time.set_xticks(x_positions)
    ax_time.set_xticklabels(sorted_models, fontsize=tt_xtick_fs, fontweight='bold', rotation=25, ha='right')
    ax_time.set_ylabel('Training Time (Min: Log Scale)', fontsize=tt_ylabel_fs, fontweight='normal')
    ax_time.set_title('Training Time Comparison', fontsize=tt_title_fs, fontweight='bold', pad=10)
    ax_time.set_yscale('log')
    # Set Y limits with extra headroom for annotations
    max_total = max(time_stats[m]['mean'] for m in sorted_models)
    ax_time.set_ylim(bottom=0.5, top=max_total * 3)
    ax_time.set_xlim(-0.5 * x_spacing, x_positions[-1] + 0.5 * x_spacing)

    # Now place the TOTAL annotations safely inside the axes so they don't overlap the bounding box
    y_top = ax_time.get_ylim()[1]
    for x_pos, mean_val, total_val, color in total_annots:
        # place near the top but inside (use 92% of y_top or 125% of mean, whichever is smaller)
        y_text = min(mean_val * 1.25, y_top * 0.92)
        ax_time.text(x_pos, y_text, f'{round_half_up(total_val, 0):.0f}m',
                     fontsize=tt_total_annot_fs, color=color, ha='center', va='bottom',
                     fontweight='bold', clip_on=False)

    # Place mid annotations (mean per epoch) at their mid positions
    for x_pos, mid_y, mean_val, color in mid_annots:
        ax_time.annotate(f'{round_half_up(mean_val, 1):.1f}m/ep',
                         xy=(x_pos, mid_y), xytext=(0, 0), textcoords='offset points',
                         fontsize=tt_mean_annot_fs, color=color, ha='center', va='center',
                         fontweight='normal',
                         bbox=dict(boxstyle='round,pad=0.32', facecolor='white',
                                   edgecolor=color, alpha=0.98, linewidth=1.4))
    ax_time.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y', which='both')
    # Remove x-axis grid
    ax_time.grid(False, axis='x')
    # Add subtle baseline at y=1
    ax_time.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # ==========================================================================
    # SUBPLOT 2 & 3: Loss and Trajectory (keep existing logic)
    # ==========================================================================
    # Plot each model for subplots 2 and 3
    for idx, model_name in enumerate(model_names):
        model_data = models[model_name]
        curves = model_data["curves"]
        color = get_model_color(model_name, idx)
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        
        # LOKI always gets star, FT-Encoder gets diamond, others get from list
        if model_name == "LOKI":
            marker = '*'
            marker_size = 550  # Star needs larger size
        elif model_name == "FT-Encoder":
            marker = 'D'
            marker_size = 180
        else:
            marker = MARKERS[idx % len(MARKERS)]
            marker_size = MARKER_SIZES[idx % len(MARKER_SIZES)]
        
        epochs = np.array(curves["epochs"])
        train_loss_mean = np.array(curves["train_loss_mean"])
        train_loss_std = np.array(curves["train_loss_std"])
        val_accuracy = np.array(curves["val_accuracy"])
        
        # Skip epoch 0 for training metrics
        if len(epochs) > 1 and epochs[0] == 0:
            plot_epochs = epochs[1:]
            plot_loss_mean = train_loss_mean[1:]
            plot_loss_std = train_loss_std[1:]
            plot_val_acc = val_accuracy[1:]
        else:
            plot_epochs = epochs
            plot_loss_mean = train_loss_mean
            plot_loss_std = train_loss_std
            plot_val_acc = val_accuracy
        
        # 2. Mean Loss with ±1 Std - Confidence band
        axes_metrics[0].plot(plot_epochs, plot_loss_mean, color=color, linewidth=4.0, 
                linestyle=linestyle, label=f'{model_name}', zorder=3)
        axes_metrics[0].fill_between(plot_epochs, 
                    plot_loss_mean - plot_loss_std, 
                    plot_loss_mean + plot_loss_std, 
                    color=color, alpha=0.28, zorder=2)
        # Mark final loss with a marker (bigger for small figure)
        axes_metrics[0].scatter([plot_epochs[-1]], [plot_loss_mean[-1]], color=color, s=marker_size * 1.4, 
                   marker=marker, edgecolors='white', linewidths=1.8, zorder=4)
        # Annotate final loss
        axes_metrics[0].annotate(f'{round_half_up(plot_loss_mean[-1], 2):.2f}', 
                xy=(plot_epochs[-1], plot_loss_mean[-1]),
                xytext=(-10, 10), textcoords='offset points',
                fontsize=18, color=color, va='bottom', ha='right', fontweight='bold')
        
        # 3. Training Trajectory: Loss → Accuracy
        n_points = len(plot_loss_mean)
        for i in range(n_points - 1):
            alpha_val = 0.35 + 0.65 * (i / n_points)
            lw = 2.8 + 2.2 * (i / n_points)
            axes_metrics[1].plot(plot_loss_mean[i:i+2], plot_val_acc[i:i+2], 
                        color=color, linewidth=lw, alpha=alpha_val, zorder=3)
        
        # Start point (hollow circle)
        axes_metrics[1].scatter([plot_loss_mean[0]], [plot_val_acc[0]], color='white', s=260, 
                   marker='o', edgecolors=color, linewidths=3.2, zorder=5)
        # End point (filled marker)
        axes_metrics[1].scatter([plot_loss_mean[-1]], [plot_val_acc[-1]], color=color, s=marker_size * 1.4, 
                   marker=marker, edgecolors='white', linewidths=1.8, zorder=5)
        
        # Add directional arrow at 60% of the path
        arrow_idx = int(n_points * 0.6)
        if arrow_idx > 0 and arrow_idx < n_points - 1:
            dx = plot_loss_mean[arrow_idx + 1] - plot_loss_mean[arrow_idx]
            dy = plot_val_acc[arrow_idx + 1] - plot_val_acc[arrow_idx]
            arrow_len = np.sqrt(dx**2 + dy**2)
            if arrow_len > 0:
                axes_metrics[1].annotate('', 
                    xy=(plot_loss_mean[arrow_idx] + dx*0.6, plot_val_acc[arrow_idx] + dy*0.6),
                    xytext=(plot_loss_mean[arrow_idx], plot_val_acc[arrow_idx]),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=3, 
                                   mutation_scale=20), zorder=6)
    
    # Customize subplot 2: Training Loss - Legend in UPPER RIGHT
    axes_metrics[0].set_xlabel('Epoch', fontsize=13, fontweight='normal')
    axes_metrics[0].set_ylabel('Training Loss', fontsize=13, fontweight='normal')
    axes_metrics[0].set_title('Training Loss (Mean ± Std)', fontsize=14, fontweight='bold', pad=10)
    axes_metrics[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes_metrics[0].legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    # Customize subplot 3: Training Trajectory - Legend in UPPER RIGHT
    axes_metrics[1].set_xlabel('Training Loss', fontsize=13, fontweight='normal')
    axes_metrics[1].set_ylabel('Validation Accuracy', fontsize=13, fontweight='normal')
    axes_metrics[1].set_title('Training Trajectory: Loss → Accuracy', fontsize=14, fontweight='bold', pad=10)
    axes_metrics[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Custom legend with model names - in UPPER RIGHT
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=get_model_color(name, i), lw=2.5,
                          linestyle=LINE_STYLES[i % len(LINE_STYLES)]) 
                   for i, name in enumerate(model_names)]
    axes_metrics[1].legend(custom_lines, model_names, loc='upper right', fontsize=11, 
                  framealpha=0.95, edgecolor='gray')
    
    # Add subtle annotation for trajectory direction
    axes_metrics[1].annotate('○ Start → End ★', xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=10, color='#666666', style='italic')
    
    # Layout and save time plot
    fig_time.tight_layout()
    plt.figure(fig_time.number)  # Set active figure for saving
    output_time_path = Path(output_dir) / "model_comparison_training_time"
    output_time_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_multi_format(str(output_time_path), dpi=300, bbox_inches='tight')
    plt.close(fig_time)
    
    # Layout and save metrics plot
    fig_metrics.tight_layout()
    plt.figure(fig_metrics.number)  # Set active figure for saving
    output_metrics_path = Path(output_dir) / "model_comparison_training_metrics"
    save_plot_multi_format(str(output_metrics_path), dpi=300, bbox_inches='tight')
    plt.close(fig_metrics)
    
    print(f"✅ Training time plot saved to: {output_time_path}.png and {output_time_path}.pdf")
    print(f"✅ Training metrics plot saved to: {output_metrics_path}.png and {output_metrics_path}.pdf")


def compare_models(
    model_dirs: Optional[List[Tuple[str, str]]] = None,
    base_dir: str = "Input_Models",
    output_dir: Optional[str] = "output_plots",
    baseline_model: str = "LOKI",
    combined_json_path: Optional[str] = None
) -> None:
    """
    Main function to compare multiple trained models.
    
    If combined_json_path is provided, loads existing combined data and regenerates plots.
    Otherwise, creates combined data from model directories and generates plots.
    
    Args:
        model_dirs: List of (model_name, json_path) tuples. If None, auto-discover.
        base_dir: Base directory for auto-discovery
        output_dir: Output directory for plots (default: output_plots)
        baseline_model: Model to use for frozen encoder baseline (default: LOKI)
        combined_json_path: Path to existing combined JSON (to regenerate plots only)
    """
    if output_dir is None:
        output_dir = base_dir
    
    print("\n" + "="*70)
    print("🔬 Multi-Model Comparison Tool")
    print("="*70)
    
    # Load or create combined data
    if combined_json_path and Path(combined_json_path).exists():
        print(f"📖 Loading existing combined data from: {combined_json_path}")
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
        combined_data = normalize_combined_data(combined_data)
        print(f"   ✅ Loaded data for {len(combined_data['models'])} models: {list(combined_data['models'].keys())}")
    else:
        # Create combined data
        combined_json_output = f"{output_dir}/combined_comparison_data.json"
        combined_data = create_combined_comparison_json(
            model_dirs=model_dirs,
            base_dir=base_dir,
            output_path=combined_json_output,
            baseline_model=baseline_model
        )
        combined_data = normalize_combined_data(combined_data)
    
    if not combined_data or not combined_data.get("models"):
        print("❌ No model data available for comparison!")
        return
    
    # Generate comparison plots
    print("\n" + "-"*70)
    print("📊 Generating Comparison Plots")
    print("-"*70)
    
    # Plot 1: Test Metrics Comparison (Val Acc, Row-Sent F1, Row-Sent AP)
    plot_test_metrics_comparison(combined_data=combined_data, output_dir=output_dir)
    
    # Plot 2: Training Metrics Comparison (Time, Loss±Std, Loss vs Acc)
    plot_training_metrics_comparison(combined_data=combined_data, output_dir=output_dir)
    
    # Plot 3: Combined Metrics (compact single-figure for papers)
    print("\n   📊 Generating compact combined plots for publication...")
    plot_combined_metrics_comparison(combined_data=combined_data, output_dir=output_dir)
    plot_combined_metrics_table_style(combined_data=combined_data, output_dir=output_dir)
    plot_combined_metrics_radar(combined_data=combined_data, output_dir=output_dir)
    
    # Plot 4: EMERGENT ABILITY - Key novelty figure showing LOKI's unique capability
    print("\n   🌟 Generating EMERGENT ABILITY plots (key novelty figure)...")
    # NEW intuitive visualizations (recommended for paper)
    plot_emergent_ability_showcase(combined_data=combined_data, output_dir=output_dir)
    plot_global_to_local_transfer(combined_data=combined_data, output_dir=output_dir)
    plot_global_to_local_transfer_3d(combined_data=combined_data, output_dir=output_dir)  # Improved: better camera angle & opacity
    # Topographic 3D plots with Z-variation scale (adjust scale for terrain prominence)
    plot_emergence_cliff_3d(combined_data=combined_data, output_dir=output_dir, emergent_metric="ap", z_variation_scale=50.0)  # Area plot with AP
    plot_emergence_cliff_3d(combined_data=combined_data, output_dir=output_dir, emergent_metric="f1", z_variation_scale=50.0)  # Area plot with F1
    plot_emergence_over_epochs(combined_data=combined_data, output_dir=output_dir)  # 3-panel: Table-Text Acc, Row-Sent F1, Row-Sent AP
    plot_emergent_gap_dual_line(combined_data=combined_data, output_dir=output_dir)
    # Legacy complex plots (optional, can be uncommented if needed)
    # plot_emergent_ability_trajectory(combined_data=combined_data, output_dir=output_dir)
    # plot_emergent_ability_delta(combined_data=combined_data, output_dir=output_dir)
    
    print("\n" + "="*70)
    print("✅ Multi-Model Comparison Complete!")
    print("="*70)
    print(f"📁 Output directory: {output_dir}")
    print(f"   📄 Combined JSON: combined_comparison_data.json")
    print(f"   📈 Test Metrics: model_comparison_test_metrics.png/.pdf")
    print(f"   📉 Training Metrics: model_comparison_training_metrics.png/.pdf")
    print(f"   📊 Combined (Horizontal): model_comparison_combined_metrics.png/.pdf")
    print(f"   📊 Combined (Vertical): model_comparison_combined_bars.png/.pdf")
    print(f"   📊 Combined (Radar): model_comparison_radar.png/.pdf")
    print(f"   🌟 Emergent Showcase: emergent_ability_showcase.png/.pdf")
    print(f"   🌟 Global→Local Transfer (2D): global_to_local_transfer.png/.pdf")
    print(f"   🌟 Global→Local Transfer (3D): global_to_local_transfer_3d.png/.pdf")
    print(f"   🏔️  Emergence Area (3D-AP): emergence_cliff_3d.png/.pdf")
    print(f"   🏔️  Emergence Area (3D-F1): emergence_cliff_3d_f1.png/.pdf")
    print(f"   🌟 Emergence Over Epochs (3-panel): emergence_over_epochs.png/.pdf")
    print(f"   🌟 Emergent Gap Dual-Line: emergent_gap_dual_line.png/.pdf")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Training Curves Utility - Load saved training data and regenerate plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate plots from default location
  python training_curves.py
  
  # Regenerate plots from a specific JSON file
  python training_curves.py --json_file "Results/training_plots/my_model_training_curves.json"
  
  # Regenerate plots without batch loss analysis
  python training_curves.py --skip_batch_losses
  
  # Print summary only (no plot regeneration)
  python training_curves.py --summary_only
  
    # Compare multiple models (auto-discover from Input_Models/)
  python training_curves.py --compare
  
  # Compare specific models
  python training_curves.py --compare --model_dirs "LOKI,Uni (R-S),Uni (S-R)"
  
  # Regenerate comparison plots from existing combined JSON
  python training_curves.py --compare --combined_json "output_plots/combined_comparison_data.json"
  
  # Compare with custom baseline model
  python training_curves.py --compare --baseline_model "LOKI"
        """
    )
    
    parser.add_argument(
        "--json_file", 
        type=str, 
        default="Input_Models/LOKI/training_data/abhinand_MedEmbed-large-v0.1_training_curves.json",
        help="Path to the training curves JSON file to load (default: Results/training_plots/abhinand_MedEmbed-large-v0.1_training_curves.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Post_Training_Results/training_plots",
        help="Output directory for plots. If not specified, uses the same directory as the JSON file."
    )
    parser.add_argument(
        "--skip_batch_losses",
        action="store_true",
        help="Skip batch-level loss analysis (faster)"
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print summary, don't regenerate plots"
    )
    
    # Multi-model comparison arguments
    parser.add_argument(
        "--compare",
        default=True,
        action="store_true",
        help="Enable multi-model comparison mode"
    )
    parser.add_argument(
        "--model_dirs",
        type=str,
        default=None,
        help="Comma-separated list of model directory names to compare (e.g., 'LOKI,Uni (R-S),Uni (S-R)'). If not specified, auto-discovers from base_dir."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="Input_Models",
        help="Base directory containing model input folders (default: Input_Models)"
    )
    parser.add_argument(
        "--combined_json",
        type=str,
        default=None,
        help="Path to existing combined comparison JSON file (to regenerate plots only)"
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="LOKI",
        help="Model to use for frozen encoder baseline values (default: LOKI)"
    )
    parser.add_argument(
        "--download_models",
        action="store_true",
        help="Download missing published model folders from Hugging Face before comparison",
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # Multi-Model Comparison Mode
    # =========================================================================
    if args.compare:
        print("\n" + "="*70)
        print("🔬 Multi-Model Comparison Mode")
        print("="*70)

        if args.download_models:
            download_input_models(destination=args.base_dir)
        
        # Parse model_dirs if provided
        model_dirs = None
        if args.model_dirs:
            model_names = [m.strip() for m in args.model_dirs.split(",")]
            print(f"📁 Specified models: {model_names}")
            model_dirs = []
            for model_name in model_names:
                json_file = find_training_curves_json(args.base_dir, model_name)
                if json_file:
                    model_dirs.append((canonicalize_model_name(model_name), str(json_file)))
                else:
                    print(f"   ⚠️ Warning: No training curves found for {model_name}")
        
        # Keep generated comparison artifacts out of the model-input directory.
        compare_output_dir = "output_plots"
        if args.output_dir and args.output_dir != "Post_Training_Results/training_plots":
            compare_output_dir = args.output_dir
        
        # Run comparison
        compare_models(
            model_dirs=model_dirs,
            base_dir=args.base_dir,
            output_dir=compare_output_dir,
            baseline_model=args.baseline_model,
            combined_json_path=args.combined_json
        )
        
        exit(0)
    
    args = parser.parse_args()
    
    # Resolve paths
    json_path = Path(args.json_file)
    
    if not json_path.exists():
        print(f"❌ Error: JSON file not found: {json_path}")
        print("   Please specify a valid path to your training_curves.json file")
        exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: use the parent of the JSON file's parent (go up from training_data/)
        output_dir = json_path.parent.parent if json_path.parent.name == "training_data" else json_path.parent
    
    print(f"📂 Loading training curves from: {json_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Extract run_name from the JSON filename
    run_name = json_path.stem.replace("_training_curves", "")
    
    # Create tracker with output directory for saving plots
    tracker = TrainingCurves(
        output_dir=str(output_dir),
        run_name=run_name,
        track_batch_losses=True,
        auto_save=False,  # Don't auto-save when regenerating
        auto_plot=False   # Don't auto-plot when loading
    )
    
    # Load the data directly from the source JSON file (not from output_dir)
    print(f"📖 Reading JSON data from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Load metadata
        metadata = data['metadata']
        tracker.run_name = metadata['run_name']
        tracker.best_accuracy = metadata['best_accuracy']
        tracker.best_epoch = metadata['best_epoch']
        tracker.best_test_overall_accuracy = get_training_curves_best_row_sent_f1(metadata)
        tracker.best_test_epoch = metadata.get('best_test_epoch', 0)
        tracker.best_test_avg_precision = metadata.get('best_test_avg_precision', 0.0)
        tracker.best_test_precision_epoch = metadata.get('best_test_precision_epoch', 0)
        tracker.track_batch_losses = metadata['track_batch_losses']
        tracker.track_val_loss = metadata['track_val_loss']
        tracker.track_row_sent_metrics = metadata.get('track_row_sent_metrics', False)
        tracker.initial_stage_metrics = metadata.get('initial_stage_metrics', {})
        
        # Load curves data
        curves = data['curves']
        tracker.epochs = curves['epochs']
        tracker.train_loss_mean = curves['train_loss_mean']
        tracker.train_loss_min = curves['train_loss_min']
        tracker.train_loss_max = curves['train_loss_max']
        tracker.train_loss_std = curves['train_loss_std']
        tracker.val_accuracy = curves['val_accuracy']
        tracker.learning_rates = curves['learning_rates']
        tracker.epoch_times = curves['epoch_times']
        
        # Load optional data
        if 'val_loss' in curves:
            tracker.val_loss = curves['val_loss']
        
        if 'batch_losses' in curves:
            tracker.batch_losses = curves['batch_losses']
        
        tracker.row_sent_overall_accuracy = get_training_curves_row_sent_f1_curve(curves)
        
        if 'row_sent_avg_precision' in curves:
            tracker.row_sent_avg_precision = curves['row_sent_avg_precision']
        
        print(f"✅ Loaded training data for run: {tracker.run_name}")
        print(f"   Total epochs: {len(tracker.epochs)}")
        print(f"   Best accuracy: {round_half_up(tracker.best_accuracy, 2):.2f} (Epoch {tracker.best_epoch})")
        
    except Exception as e:
        print(f"❌ Failed to load training curves data: {e}")
        exit(1)
    
    # Print summary
    tracker.print_summary()
    
    if not args.summary_only:
        print("\n🎨 Regenerating plots (with PNG + PDF)...")
        
        # Regenerate main training curves
        tracker.plot_curves()
        
        # Regenerate batch-level plots if enabled and available
        if not args.skip_batch_losses and tracker.batch_losses:
            print("📈 Generating batch-level analysis...")
            tracker.plot_batch_losses()  # All epochs heatmap
            
            # Plot first and last epochs
            if len(tracker.epochs) > 0:
                tracker.plot_batch_losses(epoch=1)
                if len(tracker.epochs) > 1:
                    tracker.plot_batch_losses(epoch=len(tracker.epochs))
        
        print(f"\n✅ Plots regenerated successfully!")
        print(f"📁 Plots saved to: {tracker.plots_dir}")
        print("   Both PNG and PDF formats generated for each plot.")
    else:
        print("\n📊 Summary printed (--summary_only mode, no plots regenerated)")
