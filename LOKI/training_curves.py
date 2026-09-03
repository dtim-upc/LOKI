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

Usage:
------
# Regenerate plots from default location
python training_curves.py

# Regenerate plots from a specific JSON file
python training_curves.py --json_file "Results/training_plots/my_model_training_curves.json"

# Regenerate plots without batch loss analysis
python training_curves.py --skip_batch_losses

# Print summary only (no plot regeneration)
python training_curves.py --summary_only
"""


import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from datetime import datetime
from utils import save_plot_multi_format

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


# =============================================================================
# Custom Scale for Squeezed Y-Axis (compresses middle region)
# =============================================================================
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from matplotlib.ticker import FixedLocator, FixedFormatter


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
        
        print(f"[INFO] Training curves tracker initialized:")
        print(f"   [INFO] Output directory: {self.output_dir}")
        print(f"   [INFO] Tracking batch losses: {track_batch_losses}")
        print(f"   [INFO] Tracking validation loss: {track_val_loss}")
        print(f"   [INFO] Tracking row-sentence metrics: {track_row_sent_metrics}")
        print(f"   [INFO] Auto-save: {auto_save}")
        print(f"   [INFO] Auto-plot: {auto_plot}")
    
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
        self.row_sent_f1 = [] if self.track_row_sent_metrics else None
        self.row_sent_avg_precision = [] if self.track_row_sent_metrics else None
        # Initial stage metrics (Stage 0, 1, 2 before training starts)
        self.initial_stage_metrics = {}
        # Best metrics tracking
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_test_f1 = 0.0
        self.best_test_epoch = 0
        self.best_test_avg_precision = 0.0
        self.best_test_precision_epoch = 0
        self.start_time = datetime.now()
    
    def add_epoch_0_data(self,
                        val_accuracy: float,
                        row_sent_f1: Optional[float] = None,
                        row_sent_avg_precision: Optional[float] = None):
        """
        Add Epoch 0 data (untrained model baseline).
        
        Args:
            val_accuracy: Validation accuracy of untrained model
            row_sent_f1: Optional row-sentence overall accuracy of untrained model
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
            self.row_sent_f1.append(row_sent_f1 if row_sent_f1 is not None else 0.0)
            self.row_sent_avg_precision.append(row_sent_avg_precision if row_sent_avg_precision is not None else 0.0)
    
    def add_epoch_data(self,
                      epoch: int,
                      train_losses: List[float],
                      val_accuracy: float,
                      learning_rate: float,
                      epoch_time: float,
                      val_loss: Optional[float] = None,
                      row_sent_f1: Optional[float] = None,
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
            row_sent_f1: Optional row-sentence overall accuracy
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
            self.row_sent_f1.append(row_sent_f1 if row_sent_f1 is not None else 0.0)
            self.row_sent_avg_precision.append(row_sent_avg_precision if row_sent_avg_precision is not None else 0.0)
        
        # Update best metrics
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_epoch = epoch
        
        # Update best test metrics if tracking
        if self.track_row_sent_metrics:
            if row_sent_f1 is not None and row_sent_f1 > self.best_test_f1:
                self.best_test_f1 = row_sent_f1
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
        print(f"\n[INFO] Epoch {epoch} Summary:")
        print(f"   [INFO] Train Loss: {round_half_up(self.train_loss_mean[-1], 2):.2f} +/- {round_half_up(self.train_loss_std[-1], 2):.2f}")
        print(f"   [INFO] Val Accuracy: {round_half_up(val_accuracy, 2):.2f}")
        print(f"   [BEST] Best Accuracy: {round_half_up(self.best_accuracy, 2):.2f} (Epoch {self.best_epoch})")
        print(f"   [TIME]  Epoch Time: {epoch_time:.1f}s")
        print(f"   [INFO] Learning Rate: {learning_rate:.2e}")
        
        if self.track_val_loss and val_loss is not None:
            print(f"   [INFO] Val Loss: {round_half_up(val_loss, 2):.2f}")
        
        if self.track_row_sent_metrics:
            if row_sent_f1 is not None:
                print(f"   [INFO] Row-Sent F1: {round_half_up(row_sent_f1, 2):.2f}")
                print(f"      [BEST] Best Test F1: {round_half_up(self.best_test_f1, 2):.2f} (Epoch {self.best_test_epoch})")
            if row_sent_avg_precision is not None:
                print(f"   [INFO] Row-Sent Avg Precision: {round_half_up(row_sent_avg_precision, 2):.2f}")
                print(f"      [BEST] Best Test Avg Precision: {round_half_up(self.best_test_avg_precision, 2):.2f} (Epoch {self.best_test_precision_epoch})")
    
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
            'stage_0_row_sent_acc': stage_0_row_sent_acc,
            'stage_1_row_sent_acc': stage_1_row_sent_acc,
            'stage_2_row_sent_acc': stage_2_row_sent_acc
        }
        
        print(f"[INFO] Initial stage metrics recorded:")
        print(f"   [INFO] Stage 0 (Frozen): {round_half_up(stage_0_accuracy, 2):.2f} acc, {round_half_up(stage_0_row_sent_ap, 2):.2f} row-sent AP, {round_half_up(stage_0_row_sent_acc, 2):.2f} row-sent F1")
        print(f"   [INFO] Stage 1 (Sophisticated Untrained): {round_half_up(stage_1_accuracy, 2):.2f} acc, {round_half_up(stage_1_row_sent_ap, 2):.2f} row-sent AP, {round_half_up(stage_1_row_sent_acc, 2):.2f} row-sent F1")
        print(f"   [INFO] Stage 2 (Initial): {round_half_up(stage_2_accuracy, 2):.2f} acc, {round_half_up(stage_2_row_sent_ap, 2):.2f} row-sent AP, {round_half_up(stage_2_row_sent_acc, 2):.2f} row-sent F1")
    
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
                'best_test_f1': self.best_test_f1,
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
            if self.row_sent_f1:
                data['curves']['row_sent_f1'] = self.row_sent_f1
            if self.row_sent_avg_precision:
                data['curves']['row_sent_avg_precision'] = self.row_sent_avg_precision
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
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
            print(f"[WARN] Training curves file not found: {filepath}")
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load metadata
        metadata = data['metadata']
        self.run_name = metadata['run_name']
        self.best_accuracy = metadata['best_accuracy']
        self.best_epoch = metadata['best_epoch']
        self.best_test_f1 = metadata.get('best_test_f1', 0.0)
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
        
        if 'row_sent_f1' in curves:
            self.row_sent_f1 = curves['row_sent_f1']
        
        if 'row_sent_avg_precision' in curves:
            self.row_sent_avg_precision = curves['row_sent_avg_precision']
        
        print(f"[INFO] Training curves data loaded from {filepath}")
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
            print("[WARN] No training data to plot")
            return
        
        # Create main figure with subplots (add extra row if tracking row-sentence metrics)
        if self.track_row_sent_metrics and self.row_sent_f1 and self.row_sent_avg_precision:
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
                        alpha=0.3, color='blue', label='+/-1 Std')
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
            self.row_sent_f1 and 
            self.row_sent_avg_precision and 
            len(self.row_sent_f1) > 0):
            
            ax6.text(0.55, 0.9, f'Test Summary', fontsize=12, fontweight='bold', 
                    transform=ax6.transAxes)
            
            # Final test metrics
            final_overall_acc = self.row_sent_f1[-1]
            ax6.text(0.55, 0.8, f'Final F1: {round_half_up(final_overall_acc, 2):.2f}', 
                    transform=ax6.transAxes)
                    
            final_avg_prec = self.row_sent_avg_precision[-1]
            ax6.text(0.55, 0.7, f'Final Avg Precision: {round_half_up(final_avg_prec, 2):.2f}', 
                    transform=ax6.transAxes)
            
            # Best test metrics
            ax6.text(0.55, 0.6, f'Best F1: {round_half_up(self.best_test_f1, 2):.2f}', 
                    transform=ax6.transAxes, fontweight='bold')
            ax6.text(0.55, 0.55, f'  (Epoch {self.best_test_epoch})', 
                    transform=ax6.transAxes, fontsize=9)
            ax6.text(0.55, 0.5, f'Best Avg Precision: {round_half_up(self.best_test_avg_precision, 2):.2f}', 
                    transform=ax6.transAxes, fontweight='bold')
            ax6.text(0.55, 0.45, f'  (Epoch {self.best_test_precision_epoch})', 
                    transform=ax6.transAxes, fontsize=9)
            
            # Test improvement metrics
            if len(self.row_sent_f1) > 1:
                test_f1_improvement = self.row_sent_f1[-1] - self.row_sent_f1[0]
                ax6.text(0.55, 0.35, f'F1 Δ: {round_half_up(test_f1_improvement, 2):+.2f}', 
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
            self.row_sent_f1 and 
            self.row_sent_avg_precision and 
            len(self.row_sent_f1) > 0):
            
            # Create filtered arrays for individual row-sentence plots (exclude Epoch 0)
            if has_epoch_0:
                row_sent_epochs = epochs_array[1:]
                row_sent_f1 = np.array(self.row_sent_f1[1:])
                row_sent_avg_prec = np.array(self.row_sent_avg_precision[1:])
            else:
                row_sent_epochs = epochs_array
                row_sent_f1 = np.array(self.row_sent_f1)
                row_sent_avg_prec = np.array(self.row_sent_avg_precision)
            
            # 7. Row-Sentence F1 (INCLUDE Epoch 0 for full picture)
            ax7 = axes[2, 0]
            # Use all epochs including epoch 0
            ax7.plot(epochs_array, self.row_sent_f1, 'purple', 
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
            ax9.plot(epochs_array, self.row_sent_f1, 'purple', 
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
                        alpha=0.3, color='blue', label='+/-1 Std')
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
            print("[WARN] Batch losses not tracked. Enable track_batch_losses=True")
            return
        
        if epoch is not None:
            # Plot specific epoch
            if epoch < 1 or epoch > len(self.batch_losses):
                print(f"[WARN] Epoch {epoch} not found. Available: 1-{len(self.batch_losses)}")
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
            
            print(f"[INFO] Batch losses plot for epoch {epoch} saved to {batch_path}")
        
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
            
            print(f"[INFO] Batch losses heatmap saved to {heatmap_path}")
    
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
                'best_test_f1': self.best_test_f1,
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
            print("[WARN] No training data available for summary")
            return
        
        print("\n" + "="*70)
        print(f"[INFO] TRAINING SUMMARY - {self.run_name}")
        print("="*70)
        print(f"[INFO] Total Epochs: {stats['total_epochs']}")
        print(f"[BEST] Best Accuracy: {round_half_up(stats['best_accuracy'], 2):.2f} (Epoch {stats['best_epoch']})")
        print(f"[INFO] Final Accuracy: {round_half_up(stats['final_accuracy'], 2):.2f}")
        print(f"[INFO] Accuracy Improvement: {round_half_up(stats['accuracy_improvement'], 2):+.2f}")
        print(f"[INFO] Initial Loss: {round_half_up(stats['initial_loss'], 2):.2f}")
        print(f"[INFO] Final Loss: {round_half_up(stats['final_loss'], 2):.2f}")
        print(f"[INFO] Loss Reduction: {round_half_up(stats['loss_reduction'], 2):.2f}")
        print(f"[TIME]  Average Epoch Time: {stats['average_epoch_time']:.1f}s")
        print(f"[TIME] Total Training Time: {stats['total_training_time']:.1f}s ({stats['total_training_time']/60:.1f} min)")
        
        # Print best test metrics if available
        if stats['best_test_metrics']:
            print(f"[BEST] Best Test Metrics:")
            test_metrics = stats['best_test_metrics']
            print(f"   [INFO] Best F1: {round_half_up(test_metrics['best_test_f1'], 2):.2f} (Epoch {test_metrics['best_test_epoch']})")
            print(f"   [INFO] Best Average Precision: {round_half_up(test_metrics['best_test_avg_precision'], 2):.2f} (Epoch {test_metrics['best_test_precision_epoch']})")
        
        # Print initial stage metrics if available
        if stats['initial_stage_metrics']:
            print(f"[INFO] Stage Metrics:")
            stage_metrics = stats['initial_stage_metrics']
            print(f"   [INFO] Stage 0 (Frozen): {round_half_up(stage_metrics.get('stage_0_accuracy', 0), 2):.2f} acc, {round_half_up(stage_metrics.get('stage_0_row_sent_ap', 0), 2):.2f} row-sent AP, {round_half_up(stage_metrics.get('stage_0_row_sent_acc', 0), 2):.2f} row-sent F1")
            print(f"   [INFO] Stage 1 (Sophisticated Untrained): {round_half_up(stage_metrics.get('stage_1_accuracy', 0), 2):.2f} acc, {round_half_up(stage_metrics.get('stage_1_row_sent_ap', 0), 2):.2f} row-sent AP, {round_half_up(stage_metrics.get('stage_1_row_sent_acc', 0), 2):.2f} row-sent F1")
            # Stage 2: Show final trained metrics instead of initial zeros
            test_metrics = stats.get('best_test_metrics', {})
            best_acc = stats.get('best_accuracy', 0)
            best_ap = test_metrics.get('best_test_avg_precision', 0)
            best_f1 = test_metrics.get('best_test_f1', 0)
            print(f"   [INFO] Stage 2 (Final Trained): {round_half_up(best_acc, 2):.2f} acc, {round_half_up(best_ap, 2):.2f} row-sent AP, {round_half_up(best_f1, 2):.2f} row-sent F1")
        
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
        with open(curves_file, 'r', encoding='utf-8') as f:
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
    
    print(f"[INFO] Multiple runs comparison plot saved to {output_path}")



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
        """
    )
    
    parser.add_argument(
        "--json_file", 
        type=str, 
        default="output_plots/LOKI/training_data/abhinand_MedEmbed-large-v0.1_training_curves.json",
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
    
    args = parser.parse_args()
    
    # Resolve paths
    json_path = Path(args.json_file)
    
    if not json_path.exists():
        print(f"[ERROR] Error: JSON file not found: {json_path}")
        print("   Please specify a valid path to your training_curves.json file")
        exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: use the parent of the JSON file's parent (go up from training_data/)
        output_dir = json_path.parent.parent if json_path.parent.name == "training_data" else json_path.parent
    
    print(f"[INFO] Loading training curves from: {json_path}")
    print(f"[INFO] Output directory: {output_dir}")
    
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
    print(f"[INFO] Reading JSON data from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load metadata
        metadata = data['metadata']
        tracker.run_name = metadata['run_name']
        tracker.best_accuracy = metadata['best_accuracy']
        tracker.best_epoch = metadata['best_epoch']
        tracker.best_test_f1 = metadata.get('best_test_f1', 0.0)
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
        
        if 'row_sent_f1' in curves:
            tracker.row_sent_f1 = curves['row_sent_f1']
        
        if 'row_sent_avg_precision' in curves:
            tracker.row_sent_avg_precision = curves['row_sent_avg_precision']
        
        print(f"[OK] Loaded training data for run: {tracker.run_name}")
        print(f"   Total epochs: {len(tracker.epochs)}")
        print(f"   Best accuracy: {round_half_up(tracker.best_accuracy, 2):.2f} (Epoch {tracker.best_epoch})")
        
    except Exception as e:
        print(f"[ERROR] Failed to load training curves data: {e}")
        exit(1)
    
    # Print summary
    tracker.print_summary()
    
    if not args.summary_only:
        print("\n[INFO] Regenerating plots (with PNG + PDF)...")
        
        # Regenerate main training curves
        tracker.plot_curves()
        
        # Regenerate batch-level plots if enabled and available
        if not args.skip_batch_losses and tracker.batch_losses:
            print("[INFO] Generating batch-level analysis...")
            tracker.plot_batch_losses()  # All epochs heatmap
            
            # Plot first and last epochs
            if len(tracker.epochs) > 0:
                tracker.plot_batch_losses(epoch=1)
                if len(tracker.epochs) > 1:
                    tracker.plot_batch_losses(epoch=len(tracker.epochs))
        
        print(f"\n[OK] Plots regenerated successfully!")
        print(f"[INFO] Plots saved to: {tracker.plots_dir}")
        print("   Both PNG and PDF formats generated for each plot.")
    else:
        print("\n[INFO] Summary printed (--summary_only mode, no plots regenerated)")