"""
Post-Training Multi-Model Comparison Visualization Script

This script generates comparative visualizations for:
- Baseline (Frozen Encoder / Stage 0)
- LOKI (Bidirectional Cross-Attention)
- FT-Encoder (Fine-tuned Encoder)
- Uni (R⟶S) (Unidirectional Cross-Attention, rows attend to sentences)
- Uni (S⟶R) (Unidirectional Cross-Attention, sentences attend to rows)

It loads post-training evaluation results and generates:
1. Combined bar charts (AP, F1, ROC-AUC)
2. Ranking metrics line plots (P@K, R@K, NDCG@K, MRR@K)
3. ROC/PR curve comparisons
4. Radar chart summary
5. Comprehensive dashboard

Usage:
    python post_training_comparison.py --loki_results Post_Training_Results/LOKI
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from loki_path import ensure_loki_on_path

ensure_loki_on_path()

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[WARNING] matplotlib/seaborn not available, visualizations disabled")

try:
    from utils import save_plot_multi_format  # pyright: ignore[reportMissingImports]
except ImportError:
    def save_plot_multi_format(path, dpi=150, bbox_inches='tight'):
        """Fallback save function for both PNG and PDF."""
        path_obj = Path(path)
        base = path_obj.with_suffix('')
        plt.savefig(str(base) + '.png', dpi=dpi, bbox_inches=bbox_inches)
        plt.savefig(str(base) + '.pdf', dpi=dpi, bbox_inches=bbox_inches)
        print(f"  Saved: {base}.png and {base}.pdf")


def round_half_up(value: float, decimals: int = 2) -> float:
    """Round a value using ROUND_HALF_UP."""
    d = Decimal(str(value))
    rounded = d.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP)
    return float(rounded)


UNI_R_TO_S = 'Uni (R⟶S)'
UNI_S_TO_R = 'Uni (S⟶R)'

MODEL_NAME_ALIASES = {
    'FT-Encoder': ('FT-Encoder',),
    'LOKI': ('LOKI',),
    UNI_R_TO_S: (UNI_R_TO_S, 'Uni (R-S)', 'Uni (R→S)', 'Uni-cross'),
    UNI_S_TO_R: (UNI_S_TO_R, 'Uni (S-R)', 'Uni (S→R)'),
}

DEFAULT_RESULTS_DIR_CANDIDATES = {
    UNI_R_TO_S: [
        "Post_Training_Results/Uni (R-S)",
        "Post_Training_Results/Uni (R→S)",
        "Post_Training_Results/Uni (R⟶S)",
        "Post_Training_Results/Uni-cross",
    ],
    UNI_S_TO_R: [
        "Post_Training_Results/Uni (S-R)",
        "Post_Training_Results/Uni (S→R)",
        "Post_Training_Results/Uni (S⟶R)",
    ],
}


# =============================================================================
# MODEL COLORS AND STYLING
# =============================================================================

MODEL_COLORS = {
    'Baseline': '#D62728',
    'LOKI': '#1F77B4',
    'FT-Encoder': '#7B2D8E',
    UNI_R_TO_S: '#E67E22',
    UNI_S_TO_R: '#2CA02C',
}

MODEL_MARKERS = {
    'Baseline': 'o',
    'LOKI': '*',        # Star for LOKI (user requested)
    'FT-Encoder': 'D',  # Diamond for FT-Encoder
    UNI_R_TO_S: 's',    # Square for row-to-sentence Uni model
    UNI_S_TO_R: '^',    # Triangle for sentence-to-row Uni model
}

# Per-model marker sizes (increase LOKI for visibility)
MODEL_MARKER_SIZES = {
    'LOKI': 12,
    'FT-Encoder': 5,
    UNI_R_TO_S: 5,
    UNI_S_TO_R: 6,
    'Baseline': 5,
}

# Legend-specific marker sizes (make LOKI star larger for visibility)
MODEL_LEGEND_MARKER_SIZES = {
    'LOKI': 20,
    'FT-Encoder': 10,
    UNI_R_TO_S: 10,
    UNI_S_TO_R: 11,
    'Baseline': 10,
}

MODEL_ORDER = ['Baseline', 'FT-Encoder', UNI_R_TO_S, UNI_S_TO_R, 'LOKI']  # Order for display


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_post_training_results(results_path: str) -> Optional[Dict[str, Any]]:
    """Load post-training evaluation results from JSON file."""
    json_path = Path(results_path)
    
    # Handle both directory and file paths
    if json_path.is_dir():
        json_path = json_path / "results_post_training_eval.json"
    
    if not json_path.exists():
        print(f"[WARNING] Results file not found: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_stage_metrics(results: Dict[str, Any], stage_key: str) -> Optional[Dict[str, Any]]:
    """Extract metrics for a specific stage from post-training results."""
    evaluations = results.get("evaluations", {})
    return evaluations.get(stage_key)


def find_trained_model_stage(results: Dict[str, Any], stage_keys: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Find the trained model stage (Stage 3) in post-training results.

    Args:
        results: The loaded results JSON containing an "evaluations" dict.
        stage_keys: Optional ordered list of stage keys to prefer. If not
            provided, defaults to the original priority order.
    """
    evaluations = results.get("evaluations", {})

    # Default priority order for trained model stages
    default_stage_keys = [
        "stage_3_best_test_avg_precision",
        "stage_3_best_test_overall_acc",
        "stage_3_best"
    ]

    keys_to_check = stage_keys if stage_keys is not None else default_stage_keys

    for key in keys_to_check:
        if key in evaluations:
            return evaluations[key]

    return None


def load_training_curves_data(curves_json_path: str) -> Optional[Dict[str, Any]]:
    """Load training curves JSON data as fallback for models without post-eval."""
    if not Path(curves_json_path).exists():
        return None
    
    with open(curves_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_combined_comparison_data(combined_json_path: str) -> Optional[Dict[str, Any]]:
    """Load combined comparison data from training_curves.py output."""
    if not Path(combined_json_path).exists():
        return None
    
    with open(combined_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_first_existing_path(candidate_paths: List[str]) -> Optional[str]:
    """Return the first existing path from a list of candidates."""
    for candidate in candidate_paths:
        if Path(candidate).exists():
            return candidate
    return None


def find_results_json_path(results_path: Optional[str]) -> Optional[Path]:
    """Find a post-training results JSON from a file path or directory."""
    if not results_path:
        return None

    path = Path(results_path)
    if path.is_file():
        return path

    direct_json = path / "results_post_training_eval.json"
    if direct_json.exists():
        return direct_json

    matches = sorted(path.glob("**/results_post_training_eval.json"))
    return matches[0] if matches else None


def get_model_aliases(display_name: str) -> Tuple[str, ...]:
    """Return accepted aliases for a model display name."""
    return MODEL_NAME_ALIASES.get(display_name, (display_name,))


def get_combined_model_entry(models_data: Dict[str, Any], display_name: str) -> Optional[Dict[str, Any]]:
    """Fetch model data from combined comparison JSON using display-name aliases."""
    for alias in get_model_aliases(display_name):
        if alias in models_data:
            return models_data[alias]
    return None


def get_combined_baseline_row_sent_f1(baseline_data: Dict[str, Any]) -> float:
    """Get the baseline row-sentence F1 from combined comparison data."""
    return baseline_data.get('frozen_encoder_row_sent_f1', baseline_data.get('frozen_encoder_row_sent_acc', 0))


def get_combined_best_row_sent_f1(model_data: Dict[str, Any]) -> float:
    """Get the best row-sentence F1 from combined comparison data."""
    return model_data.get('best_test_f1', model_data.get('best_test_overall_accuracy', 0))


def get_combined_row_sent_f1_curve(curves: Dict[str, Any]) -> List[Any]:
    """Get the row-sentence F1 curve from combined comparison data."""
    return curves.get('row_sent_f1', curves.get('row_sent_overall_accuracy', [0]))


def load_trained_metrics_from_results_dir(
    results_dir: Optional[str],
    display_name: str,
    stage_priority: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load trained metrics for a single model from a results directory or file."""
    results_json = find_results_json_path(results_dir)
    if not results_json:
        if results_dir:
            print(f"[WARNING] Results file not found for {display_name}: {results_dir}")
        return None

    model_results = load_post_training_results(str(results_json))
    if not model_results:
        return None

    trained_metrics = find_trained_model_stage(model_results, stage_keys=stage_priority)
    if trained_metrics:
        print(f"[INFO] Loaded {display_name} trained model metrics from post-eval")
    return trained_metrics


def load_all_model_metrics(
    loki_results_path: Optional[str] = None,
    ftencoder_dir: Optional[str] = None,
    uni_rs_dir: Optional[str] = None,
    uni_sr_dir: Optional[str] = None,
    unicross_dir: Optional[str] = None,
    combined_data_path: Optional[str] = None,
    stage_priority: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load metrics for all comparison models from various sources.
    
    Returns a dict keyed by display name, e.g. Baseline / FT-Encoder /
    Uni (R⟶S) / Uni (S⟶R) / LOKI.
    """
    all_metrics = {}
    baseline_metrics = None

    if not uni_rs_dir and unicross_dir:
        uni_rs_dir = unicross_dir
    
    # Try to load LOKI post-training results (primary source)
    if loki_results_path:
        loki_results_json = find_results_json_path(loki_results_path)
        loki_results = load_post_training_results(str(loki_results_json)) if loki_results_json else None
        if loki_results:
            # Extract baseline from Stage 0
            baseline_metrics = extract_stage_metrics(loki_results, "stage_0_frozen_encoder")
            if baseline_metrics:
                all_metrics['Baseline'] = baseline_metrics
                print(f"[INFO] Loaded Baseline metrics from LOKI Stage 0")
            
            # Extract LOKI trained model (Stage 3)
            loki_trained = find_trained_model_stage(loki_results, stage_keys=stage_priority)
            if loki_trained:
                all_metrics['LOKI'] = loki_trained
                print(f"[INFO] Loaded LOKI trained model metrics")

            # Additionally, always load the AP-optimal Stage-3 for PR curves (if present)
            loki_ap_stage = extract_stage_metrics(loki_results, "stage_3_best_test_avg_precision")
            if loki_ap_stage:
                all_metrics['LOKI_pr'] = loki_ap_stage
                print(f"[INFO] Loaded LOKI PR-stage (stage_3_best_test_avg_precision) for PR curves")
    
    # Try to load FT-Encoder results
    ft_trained = load_trained_metrics_from_results_dir(ftencoder_dir, 'FT-Encoder')
    if ft_trained:
        # Keep default behavior for FT-Encoder (do not override with LOKI-specific stage priority)
        all_metrics['FT-Encoder'] = ft_trained
    
    # Try to load both unidirectional variants
    uni_rs_trained = load_trained_metrics_from_results_dir(uni_rs_dir, UNI_R_TO_S)
    if uni_rs_trained:
        all_metrics[UNI_R_TO_S] = uni_rs_trained

    uni_sr_trained = load_trained_metrics_from_results_dir(uni_sr_dir, UNI_S_TO_R)
    if uni_sr_trained:
        all_metrics[UNI_S_TO_R] = uni_sr_trained
    
    # Fallback: load from combined_comparison_data.json
    if combined_data_path and Path(combined_data_path).exists():
        combined_data = load_combined_comparison_data(combined_data_path)
        if combined_data:
            baseline_data = combined_data.get("baseline", {})
            models_data = combined_data.get("models", {})

            # Always load Baseline from combined if not already loaded
            if 'Baseline' not in all_metrics and baseline_data:
                all_metrics['Baseline'] = {
                    'average_precision': baseline_data.get('frozen_encoder_row_sent_ap', 0),
                    'overall_accuracy': get_combined_baseline_row_sent_f1(baseline_data),
                    'dynamic_f1': get_combined_baseline_row_sent_f1(baseline_data),
                    'row_sent_f1': get_combined_baseline_row_sent_f1(baseline_data),
                    'roc_auc': 0.0,  # Not available from training curves
                    'from_training_curves': True
                }
                print(f"[INFO] Loaded Baseline metrics from combined_comparison_data.json")

            # Always load each model's metrics from combined data if not already loaded
            for display_name in ['FT-Encoder', 'LOKI', UNI_R_TO_S, UNI_S_TO_R]:
                model_data = get_combined_model_entry(models_data, display_name)
                if display_name not in all_metrics and model_data:
                    curves = model_data.get('curves', {})

                    # Get best epoch metrics
                    best_epoch = model_data.get('best_test_precision_epoch', 0)
                    row_sent_ap = curves.get('row_sent_avg_precision', [0])
                    row_sent_f1 = get_combined_row_sent_f1_curve(curves)

                    # Use best values or fallback to best_test_* if out of range
                    if best_epoch < len(row_sent_ap):
                        ap_val = row_sent_ap[best_epoch]
                        f1_val = row_sent_f1[best_epoch] if best_epoch < len(row_sent_f1) else 0
                    else:
                        ap_val = model_data.get('best_test_avg_precision', 0)
                        f1_val = get_combined_best_row_sent_f1(model_data)

                    all_metrics[display_name] = {
                        'average_precision': ap_val,
                        'overall_accuracy': f1_val,
                        'dynamic_f1': f1_val,
                        'row_sent_f1': f1_val,
                        'roc_auc': 0.0,  # Not available from training curves
                        'from_training_curves': True
                    }
                    print(f"[INFO] Loaded {display_name} metrics from combined_comparison_data.json")

    return all_metrics


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_four_model_bar_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """Create horizontal bar chart comparing all loaded models on key metrics."""
    if not PLOTTING_AVAILABLE:
        return
    
    print("\n[PLOT] Creating model bar comparison...")
    
    plt.style.use('default')
    has_dynamic_binary_accuracy = any(
        ('dynamic_binary_accuracy' in all_metrics[m]) for m in all_metrics
    )
    num_plots = 4 if has_dynamic_binary_accuracy else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(5.4 * num_plots, 6))
    fig.suptitle('Post-Training Model Comparison', fontsize=16, fontweight='bold')
    
    metric_configs = [
        ('average_precision', 'Average Precision', axes[0]),
        ('dynamic_f1', 'F1 Score', axes[1]),
    ]
    if has_dynamic_binary_accuracy:
        metric_configs.append(('dynamic_binary_accuracy', 'Dynamic Binary Accuracy', axes[2]))
        metric_configs.append(('roc_auc', 'ROC-AUC', axes[3]))
    else:
        metric_configs.append(('roc_auc', 'ROC-AUC', axes[2]))
    
    models_present = [m for m in MODEL_ORDER if m in all_metrics]
    
    for metric_key, metric_name, ax in metric_configs:
        values = []
        colors = []
        labels = []
        
        for model_name in models_present:
            metrics = all_metrics[model_name]
            val = metrics.get(metric_key, 0)
            values.append(val)
            colors.append(MODEL_COLORS.get(model_name, '#333333'))
            labels.append(model_name)
        
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_title(metric_name, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.2f}', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "post_training_4model_bars.png"
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_ranking_metrics_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """Create line plots comparing ranking metrics (P@K, R@K, NDCG@K, MRR@K) across K values."""
    if not PLOTTING_AVAILABLE:
        return
    
    print("\n[PLOT] Creating ranking metrics comparison...")
    plt.style.use('default')
    # Create a single-row layout with 6 compact subplots to fit paper columns
    # Use width ratios so the Precision-Recall subplot (index 4) is wider
    fig, axes = plt.subplots(
        1, 6, figsize=(24, 4), constrained_layout=False,
        gridspec_kw={
            # Make the Precision-Recall subplot wider and give a bit more
            # room to the Mean Rank subplot to avoid label overlap.
            'width_ratios': [0.9, 0.9, 0.9, 0.9, 1.0, 0.7]
        }
    )
    # Removed overall figure title - legend will occupy this space
    

    k_values = [1, 3, 5, 10]
    models_present = [m for m in MODEL_ORDER if m in all_metrics]

    # Map axes: 0=Precision@K,1=Recall@K,2=F1@K,3=NDCG@K,4=Precision-Recall,5=Mean Rank
    plot_axes = {
        'precision_at_k': axes[0],
        'recall_at_k': axes[1],
        'f1_at_k': axes[2],
        'ndcg_at_k': axes[3],
    }

    # Styling sizes for smaller plots but larger readable text
    title_fs = 17
    label_fs = 15
    tick_fs = 15
    # Increased legend font size for better readability in compact layout
    legend_fs = 20

    for metric_key, ax in plot_axes.items():
        metric_name = {
            'precision_at_k': 'Precision@K',
            'recall_at_k': 'Recall@K',
            'f1_at_k': 'F1@K',
            'ndcg_at_k': 'NDCG@K'
        }[metric_key]

        for model_name in models_present:
            metrics = all_metrics[model_name]
            ranking_dict = metrics.get(metric_key, {})
            if not ranking_dict:
                continue

            y_vals = []
            for k in k_values:
                val = ranking_dict.get(str(k), ranking_dict.get(k, 0))
                y_vals.append(val if val is not None else 0)

            ax.plot(k_values, y_vals,
                                     marker=MODEL_MARKERS.get(model_name, 'o'),
                                     color=MODEL_COLORS.get(model_name, '#333333'),
                                     label=model_name,
                                         linewidth=3,
                                     markersize=MODEL_MARKER_SIZES.get(model_name, 6))

        ax.set_xlabel('K', fontsize=label_fs)
        ax.set_ylabel(metric_name, fontsize=label_fs)
        ax.set_title(metric_name, fontweight='bold', fontsize=title_fs)
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        # Per-axis legends removed; a single horizontal legend will be placed
        # at the bottom of the full figure for clarity.

    # Mean Rank subplot (rightmost)
    ax_mr = axes[5]
    # Build a consistent bar layout in display order
    display_models = [m for m in MODEL_ORDER if m in all_metrics]

    mean_ranks = []
    labels = []
    colors = []
    missing_flags = []
    candidate_counts = []

    for model_name in display_models:
        metrics = all_metrics[model_name]
        mr = metrics.get('mean_rank', None)
        # Determine number of candidate pairs for this model (if available)
        num_cands = None
        if isinstance(metrics.get('num_pairs'), (int, float)):
            num_cands = int(metrics.get('num_pairs'))
        elif isinstance(metrics.get('num_candidates'), (int, float)):
            num_cands = int(metrics.get('num_candidates'))
        else:
            ps = metrics.get('pair_scores_data', [])
            if isinstance(ps, (list, tuple)):
                num_cands = len(ps)

        # Treat non-positive or missing as missing
        if mr is None or (isinstance(mr, (int, float)) and (mr <= 0 or mr == float('inf'))):
            mean_ranks.append(np.nan)
            missing_flags.append(True)
        else:
            mean_ranks.append(float(mr))
            missing_flags.append(False)

        candidate_counts.append(num_cands)

        labels.append(model_name)
        colors.append(MODEL_COLORS.get(model_name, '#333333'))

    mean_ranks_np = np.array(mean_ranks, dtype=float)
    candidate_counts_np = np.array([c if c is not None else np.nan for c in candidate_counts], dtype=float)

    if np.all(np.isnan(mean_ranks_np)):
        ax_mr.text(0.5, 0.5, 'No Mean Rank Data', ha='center', va='center', fontsize=12)
        ax_mr.set_title('Mean Rank', fontweight='bold', fontsize=title_fs)
    else:
        # Compute percentile of mean rank (0-100) when candidate count is available.
        # Compute a relative percentile across models so values map to 0-100
        # in an interpretable way: lower mean_rank (better) -> higher percent.
        # Simple percentile scaling: scale raw mean_rank to 0-100 by dividing
        # by the maximum mean_rank observed across models. This keeps the best
        # model (lowest raw mean_rank) as a lower percent number but keeps the
        # same numeric relation as the original raw scores. Example: raw 78 ->
        # 78 / 107.3 * 100 ~= 72.7
        mean_rank_pct = np.full_like(mean_ranks_np, np.nan)
        if np.any(~np.isnan(mean_ranks_np)):
            max_mr = np.nanmax(mean_ranks_np)
            if max_mr > 0:
                mean_rank_pct = (mean_ranks_np / max_mr) * 100.0
            else:
                mean_rank_pct = mean_ranks_np.copy()

        # Replace NaN with small placeholder for plotting (will annotate 'N/A')
        plot_vals = np.nan_to_num(mean_rank_pct, nan=0.0)

        # Compact vertical spacing and slightly smaller bar height so the
        # Mean Rank subplot height matches the other ranking plots.
        spacing = 0.75
        y_pos = np.arange(len(labels)) * spacing
        bar_height = 0.45

        # Offset bars to the right: draw them starting at `x_offset` instead of 0
        # Use a fixed percentile scale (0-100) so bars are visible and
        # comparable across models. Use a small left offset so bars don't
        # start at zero (improves label placement).
        x_max = 100.0
        x_offset = max(0.5, x_max * 0.02)

        bars = ax_mr.barh(y_pos, plot_vals, left=x_offset, color=colors, alpha=0.95, height=bar_height)
        ax_mr.set_yticks(y_pos)
        # Use a slightly smaller font for the Mean Rank y-labels to avoid
        # overlapping into the Precision-Recall subplot when space is tight.
        mean_rank_label_fs = max(10, label_fs - 3)
        ax_mr.set_yticklabels(labels, fontsize=mean_rank_label_fs)
        ax_mr.set_title('Mean Rank (%)', fontweight='bold', fontsize=title_fs)
        ax_mr.grid(True, alpha=0.25, axis='x')

        # Determine x-limits with padding; ensure non-zero range and leave space on left
        ax_mr.set_xlim(0, x_offset + x_max)

        # Show x-axis ticks (0, 50, 100) but hide the axis line (bottom spine)
        try:
            ax_mr.spines['bottom'].set_visible(False)
            ax_mr.spines['top'].set_visible(False)
            ax_mr.spines['right'].set_visible(False)
        except Exception:
            pass

        # Set explicit ticks and make sure they are visible (no axis line)
        ax_mr.set_xticks([0, 50, 100])
        ax_mr.tick_params(axis='x', which='major', labelsize=tick_fs, length=6, direction='out')
        # Add X-axis label indicating directionality (Lower is Better)
        try:
            # Use the same label padding as the first subplot so vertical
            # alignment of x-axis labels matches across plots.
            ref_labelpad = getattr(axes[0].xaxis, 'labelpad', None)
            if ref_labelpad is None:
                ax_mr.set_xlabel('(Lower is Better)', fontsize=label_fs)
            else:
                ax_mr.set_xlabel('(Lower is Better)', fontsize=label_fs, labelpad=ref_labelpad)
        except Exception:
            pass

        # Place numeric labels centered inside each bar (or show 'N/A')
        for idx, (bar, val, missing) in enumerate(zip(bars, plot_vals, missing_flags)):
            cy = bar.get_y() + bar.get_height() / 2
            if missing:
                # Show 'N/A' for missing values and draw a subtle hatch for visibility
                ax_mr.text(x_offset * 0.6, cy, 'N/A', va='center', ha='left', fontsize=label_fs, fontweight='bold', color='#222222')
                bar.set_alpha(0.25)
                bar.set_edgecolor('#777777')
                bar.set_hatch('//')
            else:
                txt = f'{val:.1f}%'
                # Compute textual x-position relative to the offset
                center_pos = x_offset + (val / 2.0)
                right_pos = x_offset + val + (0.02 * x_max)
                # If bar is wide enough, center the label inside the bar; otherwise place it just right
                if val >= 0.15 * x_max:
                    ax_mr.text(center_pos, cy, txt, va='center', ha='center', fontsize=label_fs, fontweight='bold', color='white')
                else:
                    ax_mr.text(right_pos, cy, txt, va='center', ha='left', fontsize=label_fs, fontweight='bold', color='#222222')

        # Tighten y-limits to fit the compact spacing
        ax_mr.set_ylim(-spacing * 0.4, y_pos[-1] + spacing * 0.4)

    # ------------------------------------------------------------------
    # Replace the removed MRR subplot with Precision-Recall Curves
    # ------------------------------------------------------------------
    try:
        from sklearn.metrics import precision_recall_curve
        ax_pr = axes[4]
        ax_pr.set_title('Precision-Recall Curves', fontweight='bold', fontsize=title_fs)
        ax_pr.set_xlabel('Recall', fontsize=label_fs)
        ax_pr.set_ylabel('Precision', fontsize=label_fs)

        pr_plotted = False
        for model_name in models_present:
            metrics_main = all_metrics[model_name]
            # For PR curves, prefer a dedicated AP-stage if available (LOKI_pr)
            metrics_pr = all_metrics.get(f"{model_name}_pr", metrics_main)
            pair_scores_data = metrics_pr.get('pair_scores_data', [])
            if not pair_scores_data:
                continue

            try:
                labels = [1 if item[3] else 0 for item in pair_scores_data]
                scores = [item[2] for item in pair_scores_data]
                if len(set(labels)) <= 1:
                    continue

                precision, recall, _ = precision_recall_curve(labels, scores)
                ap_score = metrics_pr.get('average_precision', metrics_main.get('average_precision', 0))
                ax_pr.plot(recall, precision,
                           color=MODEL_COLORS.get(model_name, '#333333'),
                           linewidth=3,
                           label=f"{model_name} (AP={ap_score:.2f})")
                pr_plotted = True
            except Exception:
                continue

        if not pr_plotted:
            ax_pr.text(0.5, 0.5, 'No pair_scores_data available\nfor PR curves',
                       ha='center', va='center', fontsize=12)

        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        ax_pr.grid(True, alpha=0.25)
        # Match tick label size to the other subplots for consistent appearance
        ax_pr.tick_params(axis='both', which='major', labelsize=tick_fs)
        # (Position adjustments moved below after layout to avoid being overridden)
    except Exception:
        # If sklearn not available or plotting fails, leave subplot blank with a message
        ax = axes[4]
        ax.text(0.5, 0.5, 'Precision-Recall plotting unavailable', ha='center', va='center')

    # Create a single horizontal legend at the top of the figure (in place of the removed title)
    try:
        # Remove any existing axis legends
        for ax in axes:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        # Build legend handles from model order
        legend_handles = []
        legend_labels = []
        for m in models_present:
            handle = Line2D([0], [0], color=MODEL_COLORS.get(m, '#333333'),
                            marker=MODEL_MARKERS.get(m, 'o'), linewidth=5,
                            markersize=MODEL_LEGEND_MARKER_SIZES.get(m, 10))
            legend_handles.append(handle)
            legend_labels.append(m)

        # Adjust top margin to fit the legend and place legend horizontally at top-center
        # Leave ~5 points of padding between legend and the plots by converting
        # points to a fraction of the figure height.
        try:
            fig_height_in = fig.get_size_inches()[1]
            pad_points = 10.0
            pad_frac = pad_points / (fig_height_in * 72.0)
            top_margin = 0.78 - pad_frac
        except Exception:
            top_margin = 0.80

        # Increase horizontal spacing and left margin slightly to prevent
        # Mean Rank y-labels from overlapping the Precision-Recall axis.
        # Reduce the overall gap slightly to tighten the spacing between
        # the first five subplots while keeping space before Mean Rank.
        plt.subplots_adjust(left=0.04, right=0.98, top=top_margin, bottom=0.18, wspace=0.32)

        # Give the last subplot (Mean Rank) extra horizontal gap by shifting
        # its axes slightly to the right. This increases the visual separation
        # between the Precision-Recall plot and the Mean Rank bar plot.
        try:
            # Desired small right-shift (in figure fraction coordinates).
            # We'll cap it to the maximum allowed so the axis sits at the
            # right margin without overflowing.
            desired_shift = 0.08
            ax_last = axes[5]
            pos = ax_last.get_position()

            # Compute maximum allowed x0 so the axis right edge <= 0.995
            max_allowed_x0 = 0.99 - pos.width

            # Compute candidate new x0 and clamp to allowable range
            candidate_x0 = pos.x0 + desired_shift
            new_x0 = min(candidate_x0, max_allowed_x0)

            # Make the last axis match the vertical position/height of the
            # first subplot so all plots align horizontally.
            try:
                first_pos = axes[0].get_position()
                target_y0 = first_pos.y0
                target_height = first_pos.height
            except Exception:
                target_y0 = pos.y0
                target_height = pos.height

            # Only set if there's a meaningful move
            if new_x0 > pos.x0 + 1e-6:
                ax_last.set_position([new_x0, target_y0, pos.width, target_height])
        except Exception:
            pass
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(legend_handles),
            fontsize=legend_fs,
            frameon=False,
            borderaxespad=0.5
        )
    except Exception:
        pass

    output_path = Path(output_dir) / "post_training_4model_ranking.png"
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_roc_pr_curves_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """Create ROC and PR curve comparison plots."""
    if not PLOTTING_AVAILABLE:
        return
    
    print("\n[PLOT] Creating ROC/PR curves comparison...")
    
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ROC and Precision-Recall Curves Comparison', fontsize=16, fontweight='bold')
    
    models_present = [m for m in MODEL_ORDER if m in all_metrics]
    
    ax_roc = axes[0]
    ax_pr = axes[1]
    
    curves_plotted = False
    
    for model_name in models_present:
        metrics_main = all_metrics[model_name]
        # For PR plotting, prefer AP-optimal stage if available (LOKI_pr)
        metrics_pr = all_metrics.get(f"{model_name}_pr", metrics_main)

        # ROC uses main metrics (selected stage), PR uses the AP-stage when present
        pair_scores_roc = metrics_main.get('pair_scores_data', [])
        pair_scores_pr = metrics_pr.get('pair_scores_data', [])

        if not pair_scores_roc and not pair_scores_pr:
            continue

        try:
            # ROC (use main stage)
            if pair_scores_roc:
                labels_roc = [1 if item[3] else 0 for item in pair_scores_roc]
                scores_roc = [item[2] for item in pair_scores_roc]
                if len(set(labels_roc)) > 1:
                    fpr, tpr, _ = roc_curve(labels_roc, scores_roc)
                    auc_score = metrics_main.get('roc_auc', 0)
                    ax_roc.plot(fpr, tpr,
                               color=MODEL_COLORS.get(model_name, '#333333'),
                               linewidth=2,
                               label=f"{model_name} (AUC={auc_score:.2f})")

            # PR (use AP-stage if available)
            if pair_scores_pr:
                labels_pr = [1 if item[3] else 0 for item in pair_scores_pr]
                scores_pr = [item[2] for item in pair_scores_pr]
                if len(set(labels_pr)) > 1:
                    precision, recall, _ = precision_recall_curve(labels_pr, scores_pr)
                    ap_score = metrics_pr.get('average_precision', metrics_main.get('average_precision', 0))
                    ax_pr.plot(recall, precision,
                              color=MODEL_COLORS.get(model_name, '#333333'),
                              linewidth=2,
                              label=f"{model_name} (AP={ap_score:.2f})")

            curves_plotted = True

        except Exception as e:
            print(f"  [WARNING] Could not plot curves for {model_name}: {e}")
            continue
    
    # ROC plot styling
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontsize=11)
    ax_roc.set_ylabel('True Positive Rate', fontsize=11)
    ax_roc.set_title('ROC Curves', fontweight='bold', fontsize=12)
    ax_roc.legend(fontsize=9)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    
    # PR plot styling
    ax_pr.set_xlabel('Recall', fontsize=11)
    ax_pr.set_ylabel('Precision', fontsize=11)
    ax_pr.set_title('Precision-Recall Curves', fontweight='bold', fontsize=12)
    ax_pr.legend(fontsize=9)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    
    if not curves_plotted:
        ax_roc.text(0.5, 0.5, 'No pair_scores_data available\nfor ROC curves', 
                   ha='center', va='center', fontsize=12)
        ax_pr.text(0.5, 0.5, 'No pair_scores_data available\nfor PR curves', 
                  ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "post_training_4model_roc_pr.png"
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_radar_chart_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """Create radar/spider chart comparing models across multiple metrics."""
    if not PLOTTING_AVAILABLE:
        return
    
    print("\n[PLOT] Creating radar chart comparison...")
    
    # Define metrics for radar chart
    radar_metrics = [
        ('average_precision', 'Avg Precision'),
        ('dynamic_f1', 'F1 Score'),
        ('roc_auc', 'ROC-AUC'),
    ]
    if any(('dynamic_binary_accuracy' in all_metrics[m]) for m in all_metrics):
        radar_metrics.append(('dynamic_binary_accuracy', 'Dyn Bin Acc'))
    
    # Add ranking metrics if available
    sample_model = next((m for m in all_metrics.values() if m.get('precision_at_k')), None)
    if sample_model:
        radar_metrics.extend([
            ('precision_at_k_5', 'P@5'),
            ('recall_at_k_5', 'R@5'),
            ('ndcg_at_k_5', 'NDCG@5'),
        ])
    
    labels = [m[1] for m in radar_metrics]
    num_vars = len(labels)
    
    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    models_present = [m for m in MODEL_ORDER if m in all_metrics]
    
    for model_name in models_present:
        metrics = all_metrics[model_name]
        values = []
        
        for metric_key, _ in radar_metrics:
            if metric_key.endswith('_5'):
                # Handle @K metrics
                base_key = metric_key[:-2]  # Remove _5
                val = metrics.get(base_key, {}).get('5', metrics.get(base_key, {}).get(5, 0))
            else:
                val = metrics.get(metric_key, 0)
            values.append(val if val is not None else 0)
        
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values,
             color=MODEL_COLORS.get(model_name, '#333333'),
             linewidth=2,
             label=model_name,
             marker=MODEL_MARKERS.get(model_name, 'o'),
             markersize=MODEL_MARKER_SIZES.get(model_name, 6))
        ax.fill(angles, values,
               color=MODEL_COLORS.get(model_name, '#333333'),
               alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison\n(Radar Chart)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "post_training_4model_radar.png"
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_comprehensive_dashboard(
    all_metrics: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """Create a comprehensive dashboard combining all key visualizations."""
    if not PLOTTING_AVAILABLE:
        return
    
    print("\n[PLOT] Creating comprehensive dashboard...")
    
    plt.style.use('default')
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('Post-Training Model Comparison Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid layout (reduced horizontal spacing for tighter layout)
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.15)
    
    models_present = [m for m in MODEL_ORDER if m in all_metrics]
    k_values = [1, 3, 5, 10]
    
    # -------------------------------------------------------------------------
    # Row 0: Key Metrics Bar Charts
    # -------------------------------------------------------------------------
    
    # 0,0-1: Average Precision
    ax_ap = fig.add_subplot(gs[0, 0:2])
    values = [all_metrics[m].get('average_precision', 0) for m in models_present]
    colors = [MODEL_COLORS.get(m, '#333333') for m in models_present]
    y_pos = np.arange(len(models_present))
    bars = ax_ap.barh(y_pos, values, color=colors, alpha=0.85, height=0.6)
    ax_ap.set_yticks(y_pos)
    ax_ap.set_yticklabels(models_present, fontsize=11)
    ax_ap.set_xlim(0, 1)
    ax_ap.set_title('Average Precision', fontweight='bold', fontsize=12)
    ax_ap.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, values):
        if val > 0:
            ax_ap.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.2f}', va='center', fontsize=12, fontweight='bold')
    
    # 0,2-3: F1 Score
    ax_f1 = fig.add_subplot(gs[0, 2:4])
    values = [all_metrics[m].get('dynamic_f1', all_metrics[m].get('overall_accuracy', 0)) for m in models_present]
    bars = ax_f1.barh(y_pos, values, color=colors, alpha=0.85, height=0.6)
    ax_f1.set_yticks(y_pos)
    ax_f1.set_yticklabels(models_present, fontsize=11)
    ax_f1.set_xlim(0, 1)
    ax_f1.set_title('F1 Score (Dynamic Threshold)', fontweight='bold', fontsize=12)
    ax_f1.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, values):
        if val > 0:
            ax_f1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.2f}', va='center', fontsize=12, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Row 1: Ranking Metrics
    # -------------------------------------------------------------------------
    
    ranking_configs = [
        ('precision_at_k', 'Precision@K', gs[1, 0]),
        ('recall_at_k', 'Recall@K', gs[1, 1]),
        ('ndcg_at_k', 'NDCG@K', gs[1, 2]),
        ('mrr_at_k', 'MRR@K', gs[1, 3]),
    ]
    
    for metric_key, metric_name, gs_pos in ranking_configs:
        ax = fig.add_subplot(gs_pos)
        
        for model_name in models_present:
            metrics = all_metrics[model_name]
            ranking_dict = metrics.get(metric_key, {})
            
            if not ranking_dict:
                continue
            
            y_vals = []
            for k in k_values:
                val = ranking_dict.get(str(k), ranking_dict.get(k, 0))
                y_vals.append(val if val is not None else 0)
            
            ax.plot(k_values, y_vals,
                                     marker=MODEL_MARKERS.get(model_name, 'o'),
                                     color=MODEL_COLORS.get(model_name, '#333333'),
                                     label=model_name,
                                         linewidth=3,
                                     markersize=MODEL_MARKER_SIZES.get(model_name, 6))
        
        ax.set_xlabel('K', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontweight='bold', fontsize=11)
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    
    # -------------------------------------------------------------------------
    # Row 2: Diagnosis/Medication + Improvement + Summary
    # -------------------------------------------------------------------------
    
    # 2,0-1: Diagnosis vs Medication F1
    ax_table = fig.add_subplot(gs[2, 0:2])
    
    diag_f1 = [all_metrics[m].get('diagnosis', {}).get('f1', 0) for m in models_present]
    med_f1 = [all_metrics[m].get('medication', {}).get('f1', 0) for m in models_present]
    
    x = np.arange(len(models_present))
    width = 0.35
    
    bars1 = ax_table.bar(x - width/2, diag_f1, width, label='Diagnosis', color='#1f77b4', alpha=0.8)
    bars2 = ax_table.bar(x + width/2, med_f1, width, label='Medication', color='#ff7f0e', alpha=0.8)
    
    ax_table.set_ylabel('F1 Score', fontsize=11)
    ax_table.set_title('F1 Score by Table Type', fontweight='bold', fontsize=12)
    ax_table.set_xticks(x)
    ax_table.set_xticklabels(models_present, fontsize=10)
    ax_table.legend(fontsize=9)
    ax_table.set_ylim(0, 1)
    ax_table.grid(True, alpha=0.3, axis='y')
    
    # 2,2: Improvement over Baseline
    ax_imp = fig.add_subplot(gs[2, 2])
    
    baseline_ap = all_metrics.get('Baseline', {}).get('average_precision', 0)
    improvements = []
    imp_colors = []
    imp_labels = []
    
    for model_name in models_present:
        if model_name == 'Baseline':
            continue
        ap = all_metrics[model_name].get('average_precision', 0)
        imp = ap - baseline_ap
        improvements.append(imp)
        imp_colors.append('green' if imp > 0 else 'red')
        imp_labels.append(model_name)
    
    if improvements:
        y_pos = np.arange(len(imp_labels))
        bars = ax_imp.barh(y_pos, improvements, color=imp_colors, alpha=0.7, height=0.5)
        ax_imp.set_yticks(y_pos)
        ax_imp.set_yticklabels(imp_labels, fontsize=10)
        ax_imp.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax_imp.set_xlabel('Δ Average Precision', fontsize=10)
        ax_imp.set_title('Improvement vs Baseline', fontweight='bold', fontsize=11)
        ax_imp.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, improvements):
            offset = 0.01 if val >= 0 else -0.01
            ha = 'left' if val >= 0 else 'right'
            ax_imp.text(val + offset, bar.get_y() + bar.get_height()/2,
                       f'{val:+.2f}', va='center', ha=ha, fontsize=11, fontweight='bold')
    
    # 2,3: Summary Stats Table
    ax_summary = fig.add_subplot(gs[2, 3])
    ax_summary.axis('off')
    
    # Create summary table data
    has_dynamic_binary_accuracy = any(
        ('dynamic_binary_accuracy' in all_metrics[m]) for m in models_present
    )
    summary_data = []
    for model_name in models_present:
        m = all_metrics[model_name]
        ap = m.get('average_precision', 0)
        f1 = m.get('dynamic_f1', m.get('overall_accuracy', 0))
        dyn_bin_acc = m.get('dynamic_binary_accuracy', 0)
        roc = m.get('roc_auc', 0)
        if has_dynamic_binary_accuracy:
            summary_data.append([model_name, f'{ap:.2f}', f'{f1:.2f}', f'{dyn_bin_acc:.2f}', f'{roc:.2f}'])
        else:
            summary_data.append([model_name, f'{ap:.2f}', f'{f1:.2f}', f'{roc:.2f}'])
    
    col_labels = ['Model', 'AP', 'F1', 'ROC-AUC']
    col_colors = ['#f0f0f0'] * 4
    if has_dynamic_binary_accuracy:
        col_labels = ['Model', 'AP', 'F1', 'DynBinAcc', 'ROC-AUC']
        col_colors = ['#f0f0f0'] * 5

    table = ax_summary.table(
        cellText=summary_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colColours=col_colors
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax_summary.set_title('Summary Statistics', fontweight='bold', fontsize=11, pad=10)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "post_training_4model_dashboard.png"
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def generate_all_comparison_plots(
    all_metrics: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """Generate all comparison plots."""
    if not PLOTTING_AVAILABLE:
        print("[ERROR] Plotting libraries not available. Cannot generate plots.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- OVERRIDE AP and Overall Accuracy from combined_comparison_data.json ---
    # Load combined data
    combined_json_path = 'output_plots/combined_comparison_data.json'
    if Path(combined_json_path).exists():
        with open(combined_json_path, 'r', encoding='utf-8') as f:
            combined_data = json.load(f)
        baseline_data = combined_data.get('baseline', {})
        models_data = combined_data.get('models', {})

        # Baseline
        if 'Baseline' in all_metrics and baseline_data:
            all_metrics['Baseline']['average_precision'] = baseline_data.get('frozen_encoder_row_sent_ap', 0)
            all_metrics['Baseline']['overall_accuracy'] = get_combined_baseline_row_sent_f1(baseline_data)
            all_metrics['Baseline']['row_sent_f1'] = get_combined_baseline_row_sent_f1(baseline_data)

        # All other models
        for model in ['FT-Encoder', 'LOKI', UNI_R_TO_S, UNI_S_TO_R]:
            model_data = get_combined_model_entry(models_data, model)
            if model in all_metrics and model_data:
                all_metrics[model]['average_precision'] = model_data.get('best_test_avg_precision', 0)
                all_metrics[model]['overall_accuracy'] = get_combined_best_row_sent_f1(model_data)
                all_metrics[model]['row_sent_f1'] = get_combined_best_row_sent_f1(model_data)

    print(f"\n{'='*60}")
    print("📊 GENERATING POST-TRAINING MODEL COMPARISON PLOTS")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    primary_model_names = [m for m in MODEL_ORDER if m in all_metrics]
    print(f"Models loaded: {primary_model_names}")

    # Generate each plot
    create_four_model_bar_comparison(all_metrics, output_dir)
    create_ranking_metrics_comparison(all_metrics, output_dir)
    create_roc_pr_curves_comparison(all_metrics, output_dir)
    create_radar_chart_comparison(all_metrics, output_dir)
    create_comprehensive_dashboard(all_metrics, output_dir)

    # Save compact raw-count summary for transparent ranking/F1 interpretation.
    raw_summary = {}
    for model_name in MODEL_ORDER:
        if model_name not in all_metrics:
            continue
        m = all_metrics[model_name]
        raw_summary[model_name] = {
            "ranking_raw_counts": m.get("ranking_raw_counts", {}),
            "prediction_breakdown": m.get("prediction_breakdown", {}),
            "diagnosis_prediction_breakdown": m.get("diagnosis_prediction_breakdown", {}),
            "medication_prediction_breakdown": m.get("medication_prediction_breakdown", {}),
            "examples_evaluated": m.get("examples_evaluated", 0),
            "diagnosis_examples": m.get("diagnosis_examples", 0),
            "medication_examples": m.get("medication_examples", 0),
        }
    with open(Path(output_dir) / "comparison_raw_counts.json", "w", encoding="utf-8") as f:
        json.dump(raw_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("✅ ALL COMPARISON PLOTS GENERATED")
    print(f"{'='*60}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for post-training comparison visualization."""
    parser = argparse.ArgumentParser(
        description="Generate comparative visualizations for Baseline, LOKI, FT-Encoder, Uni (R⟶S), and Uni (S⟶R)"
    )
    
    parser.add_argument(
        "--loki_results", type=str,
        default="Post_Training_Results/LOKI",
        help="Path to LOKI post-training evaluation results directory"
    )
    
    parser.add_argument(
        "--ftencoder_dir", type=str,
        default="Post_Training_Results/FT-Encoder",
        help="Path to FT-Encoder output directory"
    )
    
    parser.add_argument(
        "--uni_rs_dir", type=str,
        default=None,
        help="Path to Uni (R-S) output directory. If omitted, the script auto-resolves common folder names."
    )

    parser.add_argument(
        "--uni_sr_dir", type=str,
        default=None,
        help="Path to Uni (S-R) output directory. If omitted, the script auto-resolves common folder names."
    )

    parser.add_argument(
        "--unicross_dir", type=str,
        default=None,
        help="Deprecated alias for --uni_rs_dir."
    )
    
    parser.add_argument(
        "--combined_data", type=str,
        default="output_plots/combined_comparison_data.json",
        help="Path to combined comparison data JSON"
    )

    parser.add_argument(
        "--stage_priority", type=str,
        default=None,
        help="Comma-separated Stage-3 priority keys (e.g. 'stage_3_best_test_avg_precision,stage_3_best_test_overall_acc,stage_3_best')."
    )
    
    parser.add_argument(
        "--output_dir", type=str,
        default="Post_Training_Comparison_Plots",
        help="Output directory for comparison plots"
    )
    
    args = parser.parse_args()

    if args.unicross_dir and not args.uni_rs_dir:
        args.uni_rs_dir = args.unicross_dir

    if args.uni_rs_dir is None:
        args.uni_rs_dir = resolve_first_existing_path(DEFAULT_RESULTS_DIR_CANDIDATES[UNI_R_TO_S])

    if args.uni_sr_dir is None:
        args.uni_sr_dir = resolve_first_existing_path(DEFAULT_RESULTS_DIR_CANDIDATES[UNI_S_TO_R])
    
    print("\n" + "="*70)
    print("🎯 POST-TRAINING MODEL COMPARISON VISUALIZATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load all model metrics
    # Parse stage priority if provided; otherwise set default to prefer overall_acc
    # for the primary Stage-3 selection (applies when passed into loader for LOKI).
    if args.stage_priority:
        stage_priority_list = [k.strip() for k in args.stage_priority.split(',') if k.strip()]
    else:
        # Default override: prefer overall accuracy stage first for plotting/selection
        stage_priority_list = [
            'stage_3_best_test_overall_acc',
            'stage_3_best_test_avg_precision',
            'stage_3_best'
        ]

    all_metrics = load_all_model_metrics(
        loki_results_path=args.loki_results,
        ftencoder_dir=args.ftencoder_dir,
        uni_rs_dir=args.uni_rs_dir,
        uni_sr_dir=args.uni_sr_dir,
        unicross_dir=args.unicross_dir,
        combined_data_path=args.combined_data,
        stage_priority=stage_priority_list
    )
    
    if not all_metrics:
        print("[ERROR] No model metrics could be loaded. Exiting.")
        return 1
    
    primary_model_names = [m for m in MODEL_ORDER if m in all_metrics]
    print(f"\n[INFO] Successfully loaded {len(primary_model_names)} models: {primary_model_names}")
    
    # Generate all comparison plots
    generate_all_comparison_plots(all_metrics, args.output_dir)
    
    # Save combined metrics JSON for reference
    metrics_output_path = Path(args.output_dir) / "comparison_metrics.json"
    
    # Filter out pair_scores_data before saving (too large)
    filtered_metrics = {}
    for model_name, metrics in all_metrics.items():
        filtered_metrics[model_name] = {k: v for k, v in metrics.items() if k != 'pair_scores_data'}
    
    with open(metrics_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models': primary_model_names,
            'metrics': filtered_metrics,
            'raw_count_fields': {
                'ranking_raw_counts': "Per-model query/doc totals and top-k hit counts",
                'prediction_breakdown': "Global TP/FP/FN counts",
                'diagnosis_prediction_breakdown': "Diagnosis-only TP/FP/FN counts",
                'medication_prediction_breakdown': "Medication-only TP/FP/FN counts",
            }
        }, f, indent=2, default=str)
    
    print(f"\n[INFO] Saved comparison metrics to: {metrics_output_path}")

    # Save dataset statistics (train/val/test) if present in LOKI post-training results.
    try:
        loki_results = load_post_training_results(args.loki_results)
        if loki_results and loki_results.get("dataset_statistics"):
            dataset_stats_path = Path(args.output_dir) / "dataset_statistics.json"
            with open(dataset_stats_path, "w", encoding="utf-8") as f:
                json.dump(loki_results.get("dataset_statistics", {}), f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved dataset statistics to: {dataset_stats_path}")
    except Exception as e:
        print(f"[WARNING] Could not save dataset statistics: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
