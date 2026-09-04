"""
plot_scalability.py — Plot model performance across dataset sizes.

Generates publication-ready scalability figures showing how each model's
metrics change as the evaluation set grows from 50 to Full.

Usage:
  Called by run_scalability_pharma.py, or standalone:
    python plot_scalability.py --results_dir scalability_results
"""

import os
import json
import argparse
import re
from typing import Dict, List, Any, Optional

import numpy as np


# ===========================================================================
# Helpers
# ===========================================================================

def _get_k_metric(results: Dict, k: int, metric: str) -> float:
    """Safely extract a per-K metric from results dict (handles int/str keys)."""
    per_k = results.get("per_k", {})
    k_data = per_k.get(k, per_k.get(str(k), {}))
    return k_data.get(metric, 0.0)


def _size_label(size: int) -> str:
    """Convert a size int to a human-readable label."""
    return "Full" if size == 0 else str(size)


def _size_sort_key(size: int) -> int:
    """Sort key that puts 0 (Full) at the end."""
    return 999_999 if size == 0 else size


# ===========================================================================
# Main plotting functions
# ===========================================================================

PALETTE = ["#2ec4b6", "#e74c3c", "#ffb703", "#8e44ad",
           "#3498db", "#f39c12", "#1abc9c", "#d35400"]
MARKERS = ["o", "s", "D", "^", "v", "p", "*", "h"]


def plot_scalability_main(
    scalability_data: Dict[str, Dict[int, Dict]],
    sizes: List[int],
    focal_k: int = 8,
    output_dir: str = "scalability_results",
    suffix: str = "",
    metric_type: str = "macro",
):
    """
    Main scalability figure: 2×3 grid at a fixed focal K.

    Args:
        scalability_data: {model_name: {size: {macro: {...}, micro: {...}}}}
        sizes: sorted list of sizes evaluated
        focal_k: K value to use for P/R/F1/NDCG metrics
        output_dir: directory to save plots
        metric_type: "macro" or "micro"
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plots.")
        return

    models = list(scalability_data.keys())
    if not models:
        return

    sorted_sizes = sorted(sizes, key=_size_sort_key)
    x_labels = [_size_label(s) for s in sorted_sizes]
    x_pos = np.arange(len(sorted_sizes))

    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    # Define the 2x3 grid of metrics
    if metric_type == "macro":
        grid_metrics = [
            ("P@K", f"Precision@{focal_k}"),
            ("R@K", f"Recall@{focal_k}"),
            ("F1@K", f"F1@{focal_k}"),
            ("NDCG@K", f"NDCG@{focal_k}"),
            ("MAP", "MAP"),
            ("Mean_Rank", "Mean Rank (↓)"),
        ]
    else:
        grid_metrics = [
            ("P@K", f"Precision@{focal_k} (micro)"),
            ("R@K", f"Recall@{focal_k} (micro)"),
            ("F1@K", f"F1@{focal_k} (micro)"),
            ("MAP", "MAP (micro)"),
            ("Score_AP", "Score AP (micro)"),
            ("Mean_Rank", "Mean Rank (micro, ↓)"),
        ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Scalability Study — {metric_type.capitalize()}-Averaged Metrics at K={focal_k} (Pharma)",
        fontsize=16, fontweight="bold"
    )

    for idx, (metric_key, title) in enumerate(grid_metrics):
        ax = axes[idx // 3][idx % 3]

        for i, model in enumerate(models):
            values = []
            for size in sorted_sizes:
                res = scalability_data[model].get(size, {}).get(metric_type)
                if res is None:
                    values.append(np.nan)
                    continue

                if metric_key in ("MAP", "Score_AP", "Mean_Rank"):
                    val = res.get(metric_key, 0)
                    if val == float("inf"):
                        val = np.nan
                    values.append(val)
                else:
                    values.append(_get_k_metric(res, focal_k, metric_key))

            ax.plot(x_pos, values,
                    marker=MARKERS[i % len(MARKERS)],
                    color=colors[model], label=model,
                    linewidth=2, markersize=7)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Candidate Pool Size")
        ax.set_ylabel(metric_key.replace("_", " "))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Invert y-axis for Mean Rank (lower is better)
        if "Mean_Rank" in metric_key:
            ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fname = f"scalability_{metric_type}_K{focal_k}{suffix}.png"
    plot_path = os.path.join(output_dir, fname)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {metric_type} scalability plot to {plot_path}")


def plot_scalability_per_k(
    scalability_data: Dict[str, Dict[int, Dict]],
    sizes: List[int],
    k_values: List[int],
    output_dir: str = "scalability_results",
    suffix: str = "",
    metric_type: str = "macro",
):
    """
    Supplementary per-K detail figure: one row per metric, one column per K.

    Args:
        scalability_data: {model_name: {size: {macro: {...}, micro: {...}}}}
        sizes: sorted list of sizes evaluated
        k_values: all K values to plot
        output_dir: directory to save plots
        metric_type: "macro" or "micro"
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping per-K plots.")
        return

    models = list(scalability_data.keys())
    if not models:
        return

    sorted_sizes = sorted(sizes, key=_size_sort_key)
    x_labels = [_size_label(s) for s in sorted_sizes]
    x_pos = np.arange(len(sorted_sizes))

    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    if metric_type == "macro":
        per_k_metrics = ["P@K", "R@K", "F1@K", "NDCG@K", "MRR@K", "All@K"]
    else:
        per_k_metrics = ["P@K", "R@K", "F1@K"]

    n_rows = len(per_k_metrics)
    n_cols = len(k_values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(
        f"Scalability Study — {metric_type.capitalize()} Metrics per K (Pharma)",
        fontsize=16, fontweight="bold"
    )

    # Handle single-row or single-column edge cases
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, metric_key in enumerate(per_k_metrics):
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx][col_idx]

            for i, model in enumerate(models):
                values = []
                for size in sorted_sizes:
                    res = scalability_data[model].get(size, {}).get(metric_type)
                    if res is None:
                        values.append(np.nan)
                    else:
                        values.append(_get_k_metric(res, k, metric_key))

                ax.plot(x_pos, values,
                        marker=MARKERS[i % len(MARKERS)],
                        color=colors[model], label=model,
                        linewidth=1.5, markersize=5)

            ax.set_title(f"{metric_key} @ K={k}", fontsize=10, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

            # Only add legend to top-right subplot
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fname = f"scalability_{metric_type}_per_K{suffix}.png"
    plot_path = os.path.join(output_dir, fname)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {metric_type} per-K scalability plot to {plot_path}")


# ===========================================================================
# CLI (standalone usage)
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate scalability plots from pre-computed results."
    )
    parser.add_argument("--results_dir", type=str, default="scalability_results",
                        help="Directory containing scalability JSON files")
    parser.add_argument("--focal_k", type=int, default=8,
                        help="K value for the main 2×3 grid figure (default: 8)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Filename suffix for plots")
    args = parser.parse_args()

    from config import K_VALUES, SCALABILITY_SIZES

    # Load all scalability JSON files from the results directory
    all_data = {}
    for fname in os.listdir(args.results_dir):
        if not fname.endswith(".json"):
            continue
        if "scalability" not in fname or "full_scores" in fname:
            continue
        m = re.match(r"^(.+)_scalability\.json$", fname)
        if not m:
            continue
        # Model name is the last '_'-delimited token before _scalability.json
        prefix_clean = re.sub(r"_combined$", "", m.group(1))
        model_name = prefix_clean.rsplit("_", 1)[-1]
        if not model_name:
            continue
        fpath = os.path.join(args.results_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Data should be: {size_str: {macro: {...}, micro: {...}}}
        all_data[model_name] = {int(k): v for k, v in data.items()}

    if not all_data:
        print("[ERROR] No scalability JSON files found in %s" % args.results_dir)
        return

    sizes = SCALABILITY_SIZES
    os.makedirs(args.results_dir, exist_ok=True)

    plot_scalability_main(all_data, sizes, focal_k=args.focal_k,
                          output_dir=args.results_dir, suffix=args.suffix,
                          metric_type="macro")
    plot_scalability_main(all_data, sizes, focal_k=args.focal_k,
                          output_dir=args.results_dir, suffix=args.suffix,
                          metric_type="micro")
    plot_scalability_per_k(all_data, sizes, K_VALUES,
                            output_dir=args.results_dir, suffix=args.suffix,
                            metric_type="macro")

    print("\n[PLOT] All scalability plots generated in %s" % args.results_dir)


if __name__ == "__main__":
    main()
