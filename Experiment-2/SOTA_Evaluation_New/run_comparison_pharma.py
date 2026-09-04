"""
run_comparison_pharma.py — Unified SOTA comparison on Pharma Protocol (Flipped).

Evaluates up to 4 models (CMDL, LOKI, TabSTAR, TaBERT) on the SAME
subsampled test set.  Prints side-by-side Macro + Micro comparison
tables, generates plots, and exports results to Excel.

See HOW_TO_RUN.md for full setup and usage instructions.

Quick examples:
  python run_comparison_pharma.py --skip_cmdl                              # LOKI, TabSTAR, TaBERT
  python run_comparison_pharma.py --max_test_examples 50                   # quick test, 50 examples
  python run_comparison_pharma.py --skip_cmdl --skip_tabstar --skip_tabert # LOKI only
  python run_comparison_pharma.py --skip_loki --skip_tabstar --skip_tabert # CMDL only
  python run_comparison_pharma.py --skip_cmdl --skip_loki --skip_tabert    # TabSTAR only
  python run_comparison_pharma.py --skip_cmdl --skip_loki --skip_tabstar   # TaBERT only
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    K_VALUES, OUTPUT_DIR, MAX_TEST_EXAMPLES, SEED,
    LOKI_MODELS, LOKI_ACTIVE_MODEL, LOKI_USE_SCHEMA_AWARE_SCORER, LOKI_ARGS_PATH,
    LOKI_SCHEMA_AWARE_REPRESENTATION,
    LOKI_CELL_LEVEL_MATCHING_REPRESENTATION,
)
from metrics import print_results_table, print_results_table_micro

# Default test file for pharma flipped
DEFAULT_PHARMA_TEST_FILE = os.path.join(
    SCRIPT_DIR, "..", "Datasets", "pharma_flipped_structured", "test_row_level.json"
)

# ===========================================================================
# Result reuse helpers
# ===========================================================================
def load_results_json(path, label):
    """Load a JSON results file."""
    print("[RUN] Loading %s results from %s" % (label, path))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_loki_schema_mode(requested_use_schema_aware_loki) -> bool:
    if requested_use_schema_aware_loki is not None:
        return bool(requested_use_schema_aware_loki)

    try:
        with open(LOKI_ARGS_PATH, "r", encoding="utf-8") as f:
            loki_args = json.load(f)
        return bool(
            loki_args.get("use_header_conditioning", False)
            or loki_args.get("use_cell_level_matching", False)
        )
    except Exception:
        return False


def _resolve_loki_schema_representation(requested_use_schema_aware_loki) -> str:
    resolved_use_schema_aware_loki = _resolve_loki_schema_mode(requested_use_schema_aware_loki)
    try:
        with open(LOKI_ARGS_PATH, "r", encoding="utf-8") as f:
            loki_args = json.load(f)
        checkpoint_uses_header_conditioning = bool(loki_args.get("use_header_conditioning", False))
        checkpoint_uses_cell_level_matching = bool(loki_args.get("use_cell_level_matching", False))
    except Exception:
        checkpoint_uses_header_conditioning = False
        checkpoint_uses_cell_level_matching = False

    if resolved_use_schema_aware_loki:
        representation_parts = []
        if checkpoint_uses_header_conditioning:
            representation_parts.append(LOKI_SCHEMA_AWARE_REPRESENTATION)
        if checkpoint_uses_cell_level_matching:
            representation_parts.append(LOKI_CELL_LEVEL_MATCHING_REPRESENTATION)
        if representation_parts:
            return "+".join(representation_parts)
    return "legacy"


def _loki_results_compatible(results_obj, use_schema_aware_loki: bool, schema_representation: str) -> bool:
    if not isinstance(results_obj, dict):
        return False
    macro = results_obj.get("macro", results_obj)
    cached_flag = bool(macro.get("use_schema_aware_loki", False))
    cached_representation = macro.get("loki_schema_representation", "legacy")
    return cached_flag == bool(use_schema_aware_loki) and cached_representation == schema_representation


def ask_existing_results_choice(cmdl_path, loki_path, tabstar_path, tabert_path):
    """
    Ask user what to refresh when saved results already exist.

    Returns:
        "rerun_all" | "rerun_cmdl" | "rerun_loki" | "rerun_tabstar" | "rerun_tabert" | "skip"
    """
    print("\n" + "=" * 80)
    print("  Existing result files found:")
    if cmdl_path: print("    - CMDL: %s" % cmdl_path)
    if loki_path: print("    - LOKI: %s" % loki_path)
    if tabstar_path: print("    - TabSTAR: %s" % tabstar_path)
    if tabert_path: print("    - TaBERT: %s" % tabert_path)
    print("")
    print("  Choose what to do:")
    print("    1) Re-run ALL models (fresh evaluation)")
    print("    2) Re-run CMDL only")
    print("    3) Re-run LOKI only")
    print("    4) Re-run TabSTAR only")
    print("    5) Re-run TaBERT only")
    print("    6) Skip running, just generate comparison + plots")
    print("=" * 80)

    while True:
        choice = input("Enter choice [1/2/3/4/5/6]: ").strip()
        if choice == "1": return "rerun_all"
        if choice == "2": return "rerun_cmdl"
        if choice == "3": return "rerun_loki"
        if choice == "4": return "rerun_tabstar"
        if choice == "5": return "rerun_tabert"
        if choice == "6": return "skip"
        print("Invalid choice.")




# ===========================================================================
# Comparison tables
# ===========================================================================
def _get_k_metric(results, k, metric):
    """Helper to handle both int and str keys from JSON load."""
    per_k = results.get("per_k", {})
    k_data = per_k.get(k, per_k.get(str(k), {}))
    return k_data.get(metric, 0)


def print_comparison(models_results: dict, k_values, baseline_model_key="LOKI"):
    """Print a side-by-side comparison of MACRO metrics, adapting to N models."""
    active_models = [m for m, res in models_results.items() if res is not None]
    if not active_models: return

    baseline_match = [m for m in active_models if m.startswith(baseline_model_key)]
    baseline_key = baseline_match[0] if baseline_match else active_models[-1]
    other_models = [m for m in active_models if m != baseline_key]
    ordered_models = other_models + [baseline_key]

    print("\n" + "=" * 130)
    print("  Table-Text Discovery Comparison (MACRO)")
    print("=" * 130)

    header = f"\n  {'Metric':<12}  {'K':<5}"
    for m in ordered_models:
        header += f"  {m:<18}"
    
    delta_cols = [f"Diff(B-{m})" for m in other_models]
    if delta_cols:
        header += "  " + "  ".join(f"{d:<15}" for d in delta_cols)
    print(header)
    print(f"  {'-' * (len(header) - 3)}")

    for metric_name in ["P@K", "R@K", "F1@K", "NDCG@K", "MRR@K"]:
        for k in k_values:
            row_str = f"  {metric_name:<12}  {k:<5}"
            vals = {}
            for m in ordered_models:
                val = _get_k_metric(models_results[m], k, metric_name)
                vals[m] = val
                row_str += f"  {val:<18.4f}"
            
            baseline_val = vals[baseline_key]
            for m in other_models:
                delta = baseline_val - vals[m]
                marker = "^" if delta > 0 else "v" if delta < 0 else "="
                row_str += f"  {marker} {delta:+.4f}       "
            print(row_str)
        print()

    for metric_name in ["MAP", "Score_AP", "Mean_Rank"]:
        print_name = "Score AP" if metric_name == "Score_AP" else "Mean Rank" if metric_name == "Mean_Rank" else metric_name
        row_str = f"  {print_name:<12}  {'-':<5}"
        vals = {}
        for m in ordered_models:
            default_val = float("inf") if metric_name == "Mean_Rank" else 0
            val = models_results[m].get(metric_name)
            if val is None: val = default_val
            vals[m] = val
            row_str += f"  {val:<18.2f}" if metric_name == "Mean_Rank" else f"  {val:<18.4f}"
            
        baseline_val = vals[baseline_key]
        for m in other_models:
            delta = baseline_val - vals[m]
            if metric_name == "Mean_Rank":
                marker = "^" if delta < 0 else "v" if delta > 0 else "="
                row_str += f"  {marker} {delta:+.2f}         "
            else:
                marker = "^" if delta > 0 else "v" if delta < 0 else "="
                row_str += f"  {marker} {delta:+.4f}       "
        print(row_str)

    queries_str = ",  ".join(f"{m}={models_results[m].get('num_queries', '?')}" for m in ordered_models)
    print(f"\n  Queries:    {queries_str}")
    ex = models_results[baseline_key].get('num_examples', '?')
    max_ex = models_results[baseline_key].get('max_test_examples', '?')
    print(f"  Examples:   {ex} (max_test_examples={max_ex})")
    print("=" * 130 + "\n")


def print_comparison_micro(models_results: dict, k_values, baseline_model_key="LOKI"):
    """Print a side-by-side comparison of MICRO metrics, adapting to N models."""
    active_models = [m for m, res in models_results.items() if res is not None]
    if not active_models: return

    baseline_match = [m for m in active_models if m.startswith(baseline_model_key)]
    baseline_key = baseline_match[0] if baseline_match else active_models[-1]
    other_models = [m for m in active_models if m != baseline_key]
    ordered_models = other_models + [baseline_key]

    print("\n" + "=" * 130)
    print("  Table-Text Discovery Comparison (MICRO)")
    print("=" * 130)

    header = f"\n  {'Metric':<12}  {'K':<5}"
    for m in ordered_models:
        header += f"  {m:<18}"
    
    delta_cols = [f"Diff(B-{m})" for m in other_models]
    if delta_cols:
        header += "  " + "  ".join(f"{d:<15}" for d in delta_cols)
    print(header)
    print(f"  {'-' * (len(header) - 3)}")

    for metric_name in ["P@K", "R@K", "F1@K"]:
        for k in k_values:
            row_str = f"  {metric_name:<12}  {k:<5}"
            vals = {}
            for m in ordered_models:
                val = _get_k_metric(models_results[m], k, metric_name)
                vals[m] = val
                row_str += f"  {val:<18.4f}"
            
            baseline_val = vals[baseline_key]
            for m in other_models:
                delta = baseline_val - vals[m]
                marker = "^" if delta > 0 else "v" if delta < 0 else "="
                row_str += f"  {marker} {delta:+.4f}       "
            print(row_str)
        print()

    # Overall pool-based metrics
    for metric_name in ["MAP", "Score_AP", "Mean_Rank"]:
        print_name = "Score AP" if metric_name == "Score_AP" else "Mean Rank" if metric_name == "Mean_Rank" else metric_name
        row_str = f"  {print_name:<12}  {'-':<5}"
        vals = {}
        for m in ordered_models:
            default_val = float("inf") if metric_name == "Mean_Rank" else 0
            val = models_results[m].get(metric_name)
            if val is None: val = default_val
            vals[m] = val
            row_str += f"  {val:<18.2f}" if metric_name == "Mean_Rank" else f"  {val:<18.4f}"
            
        baseline_val = vals[baseline_key]
        for m in other_models:
            delta = baseline_val - vals[m]
            if metric_name == "Mean_Rank":
                marker = "^" if delta < 0 else "v" if delta > 0 else "="
                row_str += f"  {marker} {delta:+.2f}         "
            else:
                marker = "^" if delta > 0 else "v" if delta < 0 else "="
                row_str += f"  {marker} {delta:+.4f}       "
        print(row_str)

    queries_str = ",  ".join(f"{m}={models_results[m].get('num_queries', '?')}" for m in ordered_models)
    print(f"\n  Queries:    {queries_str}")
    # Micro dicts might not have max_test_examples, safely default it:
    ex = models_results[baseline_key].get('num_examples', '?')
    max_ex = models_results[baseline_key].get('max_test_examples', '?')
    print(f"  Examples:   {ex} (max_test_examples={max_ex})")
    print("=" * 130 + "\n")


# ===========================================================================
# Plotting
# ===========================================================================
def generate_plots(models_macro: dict, models_micro: dict, k_values, output_dir, baseline_model_key="LOKI", suffix=""):
    """Generate comparison plots for both macro and micro metrics adapting to N models."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plots.")
        return

    # Palette
    palette = ["#2ec4b6", "#e74c3c", "#ffb703", "#8e44ad", "#3498db", "#f39c12", "#1abc9c", "#d35400"]
    
    # Assign colors dynamically
    active_models_macro = [m for m, res in models_macro.items() if res is not None]
    if not active_models_macro: return
    
    baseline_match = [m for m in active_models_macro if m.startswith(baseline_model_key)]
    baseline_key = baseline_match[0] if baseline_match else active_models_macro[-1]
    other_models = [m for m in active_models_macro if m != baseline_key]
    ordered_models = other_models + [baseline_key] # put baseline last for bar charts

    colors = {m: palette[i % len(palette)] for i, m in enumerate(ordered_models)}

    # markers
    markers = ["o-", "s-", "d-", "^-", "v-", "p-", "*-", "h-"]

    # --- MACRO Plot ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("Table-Text Discovery — Macro-Averaged Metrics (Pharma)", fontsize=16, fontweight="bold")

    base_metrics = ["P@K", "R@K", "F1@K", "NDCG@K", "MRR@K"]
    for idx, metric in enumerate(base_metrics):
        ax = axes[idx // 4][idx % 4]
        for i, m in enumerate(ordered_models):
            vals = [_get_k_metric(models_macro[m], k, metric) for k in k_values]
            ax.plot(k_values, vals, markers[i % len(markers)], color=colors[m], label=m, linewidth=2, markersize=6)
        
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel("K")
        ax.set_ylabel(metric)
        ax.set_xticks(k_values)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # MAP + Score_AP bar chart
    ax = axes[1][1]
    x_labels = ["MAP", "Score AP"]
    x = np.arange(len(x_labels))
    n_models = len(ordered_models)
    
    total_group_width = 0.7
    bar_step = total_group_width / n_models
    w = bar_step * 0.85  # leaving a 15% gap between bars within a group
    start_offset = -total_group_width / 2 + bar_step / 2
    
    for i, m in enumerate(ordered_models):
        map_val = models_macro[m].get("MAP", 0)
        sap_val = models_macro[m].get("Score_AP") or 0
        pos = x + start_offset + i * bar_step
        bars = ax.bar(pos, [map_val, sap_val], w, color=colors[m], label=m, edgecolor="black", linewidth=0.5)
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("Average Precision", fontsize=13, fontweight="bold")
    ax.set_ylabel("AP")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Mean Rank bar chart
    ax = axes[1][2]
    x_labels_mr = ["Mean Rank"]
    x_mr = np.arange(len(x_labels_mr))
    
    for i, m in enumerate(ordered_models):
        mr_val = models_macro[m].get("Mean_Rank", 0)
        if mr_val == float("inf"): mr_val = 0
        pos = x_mr + start_offset + i * bar_step
        bars = ax.bar(pos, [mr_val], w, color=colors[m], label=m, edgecolor="black", linewidth=0.5)
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("Mean Rank (lower is better)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rank")
    ax.set_xticks(x_mr)
    ax.set_xticklabels(x_labels_mr)
    ax.set_xlim(-0.6, 0.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # PR Curve (@K) subplot
    ax = axes[1][3]
    for i, m in enumerate(ordered_models):
        precisions = [_get_k_metric(models_macro[m], k, "P@K") for k in k_values]
        recalls = [_get_k_metric(models_macro[m], k, "R@K") for k in k_values]
        # Adding points to line plot for PR curve
        ax.plot(recalls, precisions, markers[i % len(markers)], color=colors[m], label=m, linewidth=2, markersize=6)

    ax.set_title("Precision-Recall Curve (@K)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, "pharma%s_macro_plot.png" % suffix)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved macro comparison plot to %s" % plot_path)

    # --- MICRO Plot ---
    active_models_micro = [m for m, res in models_micro.items() if res is not None]
    if not active_models_micro: return

    # Recalculate ordered models for micro just in case
    baseline_match_m = [m for m in active_models_micro if m.startswith(baseline_model_key)]
    baseline_key_m = baseline_match_m[0] if baseline_match_m else active_models_micro[-1]
    ordered_models_m = [m for m in active_models_micro if m != baseline_key_m] + [baseline_key_m]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Table-Text Discovery — Micro-Averaged Metrics (Pharma)", fontsize=16, fontweight="bold")

    for idx, metric in enumerate(["P@K", "R@K", "F1@K"]):
        ax = axes[idx]
        for i, m in enumerate(ordered_models_m):
            vals = [_get_k_metric(models_micro[m], k, metric) for k in k_values]
            ax.plot(k_values, vals, markers[i % len(markers)], color=colors.get(m, palette[i % len(palette)]), label=m, linewidth=2, markersize=6)

        ax.set_title(metric + " (micro)", fontsize=13, fontweight="bold")
        ax.set_xlabel("K")
        ax.set_ylabel(metric)
        ax.set_xticks(k_values)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, "pharma%s_micro_plot.png" % suffix)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved micro comparison plot to %s" % plot_path)


# ===========================================================================
# Main
# ===========================================================================
def main():
    loki_model_names = ", ".join(LOKI_MODELS.keys())

    parser = argparse.ArgumentParser(description="CMDL vs LOKI Unified Pharma Comparison (Macro + Micro)")

    # Shared args
    parser.add_argument("--test_file", type=str, default=DEFAULT_PHARMA_TEST_FILE,
                        help="Path to test JSON (default: %s)" % DEFAULT_PHARMA_TEST_FILE)
    parser.add_argument("--max_test_examples", type=int, default=MAX_TEST_EXAMPLES,
                        help="Max test examples, 0=all (default: %d). "
                             "SHARED between CMDL and LOKI." % MAX_TEST_EXAMPLES)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for subsampling (default: %d)" % SEED)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory (default: %s)" % OUTPUT_DIR)
    parser.add_argument("--combined_tables", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Evaluate combined source tables (default: False, row-level fragments).")

    # Model selection
    parser.add_argument("--loki_model", type=str, default=LOKI_ACTIVE_MODEL,
                        choices=list(LOKI_MODELS.keys()),
                        help="Which LOKI checkpoint to use. Options: [%s]" % loki_model_names)
    parser.add_argument("--use_schema_aware_loki", action=argparse.BooleanOptionalAction,
                        default=LOKI_USE_SCHEMA_AWARE_SCORER,
                        help="Use the Rewind-compatible structured LOKI scorer path. Default auto-detects header conditioning and cell-level matching from the checkpoint args.json.")
    parser.add_argument("--skip_cmdl", action="store_true", help="Skip CMDL evaluation")
    parser.add_argument("--skip_loki", action="store_true", help="Skip LOKI evaluation")
    parser.add_argument("--skip_tabstar", action="store_true", help="Skip TabSTAR evaluation")
    parser.add_argument("--skip_tabert", action="store_true", help="Skip TaBERT evaluation")
    
    # Pre-computed results
    parser.add_argument("--cmdl_results", type=str, default=None,
                        help="Load pre-computed CMDL combined results JSON")
    parser.add_argument("--loki_results", type=str, default=None,
                        help="Load pre-computed LOKI combined results JSON")
    parser.add_argument("--tabstar_results", type=str, default=None,
                        help="Load pre-computed TabSTAR combined results JSON")
    parser.add_argument("--tabert_results", type=str, default=None,
                        help="Load pre-computed TaBERT combined results JSON")

    # Native LOKI params
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--eval_row_chunk_size", type=int, default=0,
                        help="default: 0 (no chunking: considers all rows at once). Set to >0 to chunk rows (useful for large datasets which might throw memory errors).")
    parser.add_argument("--task_direction", type=str, default="DOC_TO_TABLE",
                        choices=["DOC_TO_TABLE", "TABLE_TO_DOC"],
                        help="Task direction for evaluation.")
    parser.add_argument("--native_direction", type=str, default="DOC_TO_TABLE",
                        choices=["DOC_TO_TABLE", "TABLE_TO_DOC"],
                        help="Native direction of the dataset.")
    parser.add_argument("--dataset_format", type=str, default="other",
                        choices=["mimic", "other"],
                        help="Dataset format identifier.")
    
    # Embedding cache control
    parser.add_argument("--cache_table_embeddings", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Pre-encode and cache all table embeddings on GPU (default: True). "
                             "Use --no_cache_table_embeddings to encode on-the-fly.")
    parser.add_argument("--cache_doc_embeddings", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Pre-encode and cache all document embeddings on GPU (default: False). "
                             "Disabling saves GPU memory for large models.")

    # TaBERT-specific
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use BFloat16 inference for TaBERT on CUDA "
                             "(default: True). Matches LOKI's bfloat16 precision.")
    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Apply torch.compile() to TaBERT for optimized inference "
                             "(default: True). Requires PyTorch >= 2.0.")

    args = parser.parse_args()

    if args.combined_tables:
        parser.error(
            "--combined_tables is not supported by the unified evaluators. "
            "Use --no-combined_tables (the default) for row-level fragments."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    resolved_loki_schema_mode = _resolve_loki_schema_mode(args.use_schema_aware_loki)
    resolved_loki_schema_representation = _resolve_loki_schema_representation(args.use_schema_aware_loki)

    suffix = ""
    cmdl_default_path = os.path.join(args.output_dir, "CMDL_pharma%s_results.json" % suffix)
    loki_default_path = os.path.join(args.output_dir, "LOKI_pharma%s_results.json" % suffix)
    tabstar_default_path = os.path.join(args.output_dir, "TabSTAR_pharma%s_results.json" % suffix)
    tabert_default_path = os.path.join(args.output_dir, "TaBERT_pharma%s_results.json" % suffix)

    k_values = K_VALUES
    cmdl_results = None
    loki_results = None
    tabstar_results = None
    tabert_results = None
    
    run_cmdl = not args.skip_cmdl
    run_loki = not args.skip_loki
    run_tabstar = not args.skip_tabstar
    run_tabert = not args.skip_tabert

    example_label = "%d examples" % args.max_test_examples if args.max_test_examples > 0 else "full test set"

    # --- Interactive reuse prompt ---
    can_prompt_reuse = (
        run_cmdl or run_loki or run_tabstar or run_tabert
    ) and (
        os.path.exists(cmdl_default_path) or os.path.exists(loki_default_path) or os.path.exists(tabstar_default_path) or os.path.exists(tabert_default_path)
    ) and not (args.cmdl_results or args.loki_results or args.tabstar_results or args.tabert_results)
    
    if can_prompt_reuse:
        refresh_choice = ask_existing_results_choice(
            cmdl_default_path if os.path.exists(cmdl_default_path) else None,
            loki_default_path if os.path.exists(loki_default_path) else None,
            tabstar_default_path if os.path.exists(tabstar_default_path) else None,
            tabert_default_path if os.path.exists(tabert_default_path) else None
        )
        if refresh_choice == "rerun_all":
            pass # Keep defaults
        elif refresh_choice == "rerun_cmdl":
            run_loki = False
            run_tabstar = False
            run_tabert = False
        elif refresh_choice == "rerun_loki":
            run_cmdl = False
            run_tabstar = False
            run_tabert = False
        elif refresh_choice == "rerun_tabstar":
            run_cmdl = False
            run_loki = False
            run_tabert = False
        elif refresh_choice == "rerun_tabert":
            run_cmdl = False
            run_loki = False
            run_tabstar = False
        elif refresh_choice == "skip":
            run_cmdl = False
            run_loki = False
            run_tabstar = False
            run_tabert = False

    if not run_cmdl and os.path.exists(cmdl_default_path):
        cmdl_results = load_results_json(cmdl_default_path, "CMDL")
    if not run_loki and os.path.exists(loki_default_path):
        cached_loki_results = load_results_json(loki_default_path, "LOKI")
        if _loki_results_compatible(cached_loki_results, resolved_loki_schema_mode, resolved_loki_schema_representation):
            loki_results = cached_loki_results
        else:
            print("[RUN] Cached LOKI results ignored because the schema-aware scorer configuration differs.")
            if not args.skip_loki:
                run_loki = True
    if not run_tabstar and os.path.exists(tabstar_default_path):
        tabstar_results = load_results_json(tabstar_default_path, "TabSTAR")
    if not run_tabert and os.path.exists(tabert_default_path):
        tabert_results = load_results_json(tabert_default_path, "TaBERT")

    # --- CMDL ---
    if run_cmdl:
        if args.cmdl_results:
            cmdl_results = load_results_json(args.cmdl_results, "CMDL")
        else:
            from evaluate_cmdl import evaluate_cmdl
            print("\n" + "=" * 65)
            print("  Running CMDL Evaluation (%s)" % example_label)
            print("=" * 65)
            
            # Run CMDL evaluation
            macro, micro = evaluate_cmdl(
                test_file=args.test_file,
                max_test_examples=args.max_test_examples,
                seed=args.seed,
                task=args.task_direction,
                dataset_format=args.dataset_format,
                native_direction=args.native_direction,
                return_micro=True,
            )
            cmdl_results = {
                "macro": macro,
                "micro": micro
            }
            with open(cmdl_default_path, "w", encoding="utf-8") as f:
                json.dump(cmdl_results, f, indent=2)
                
        print_results_table(cmdl_results["macro"], "CMDL (Macro)")
        print_results_table_micro(cmdl_results["micro"], "CMDL (Micro)")
    elif cmdl_results is None and os.path.exists(cmdl_default_path):
        cmdl_results = load_results_json(cmdl_default_path, "CMDL")

    # --- LOKI ---
    if run_loki:
        if args.loki_results:
            loki_results = load_results_json(args.loki_results, "LOKI")
        else:
            from evaluate_loki import evaluate_loki
            
            print("\n" + "=" * 65)
            print("  Running LOKI Evaluation (%s)" % example_label)
            print("=" * 65)
            
            # Single pass: compute both MACRO and MICRO from one scoring run
            macro, micro = evaluate_loki(
                test_file=args.test_file,
                max_test_examples=args.max_test_examples,
                seed=args.seed,
                loki_model_key=args.loki_model,
                aggregate_to_global_tables=False,
                task=args.task_direction,
                dataset_format=args.dataset_format,
                native_direction=args.native_direction,
                encode_batch_size=args.encode_batch_size,
                return_micro=True,
                eval_row_chunk_size=args.eval_row_chunk_size,
                cache_table_embeddings=args.cache_table_embeddings,
                cache_doc_embeddings=args.cache_doc_embeddings,
                use_schema_aware_loki=args.use_schema_aware_loki,
            )
            
            loki_results = {
                "macro": macro,
                "micro": micro
            }
            with open(loki_default_path, "w", encoding="utf-8") as f:
                json.dump(loki_results, f, indent=2)

        print_results_table(loki_results["macro"], "LOKI (Macro)")
        print_results_table_micro(loki_results["micro"], "LOKI (Micro)")
    elif loki_results is None and os.path.exists(loki_default_path):
        cached_loki_results = load_results_json(loki_default_path, "LOKI")
        if _loki_results_compatible(cached_loki_results, resolved_loki_schema_mode, resolved_loki_schema_representation):
            loki_results = cached_loki_results

    # --- TabSTAR ---
    if run_tabstar:
        if args.tabstar_results:
            tabstar_results = load_results_json(args.tabstar_results, "TabSTAR")
        else:
            from evaluate_tabstar import evaluate_tabstar
            
            print("\n" + "=" * 65)
            print("  Running TabSTAR Evaluation (%s)" % example_label)
            print("=" * 65)
            
            macro, micro = evaluate_tabstar(
                test_file=args.test_file,
                max_test_examples=args.max_test_examples,
                seed=args.seed,
                task=args.task_direction,
                dataset_format=args.dataset_format,
                native_direction=args.native_direction,
                return_micro=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            tabstar_results = {
                "macro": macro,
                "micro": micro
            }
            with open(tabstar_default_path, "w", encoding="utf-8") as f:
                json.dump(tabstar_results, f, indent=2)

        print_results_table(tabstar_results["macro"], "TabSTAR (Macro)")
        print_results_table_micro(tabstar_results["micro"], "TabSTAR (Micro)")

    # --- TaBERT ---
    if run_tabert:
        if args.tabert_results:
            tabert_results = load_results_json(args.tabert_results, "TaBERT")
        else:
            from evaluate_tabert import evaluate_tabert
            
            print("\n" + "=" * 65)
            print("  Running TaBERT Evaluation (%s)" % example_label)
            print("=" * 65)
            
            macro, micro = evaluate_tabert(
                test_file=args.test_file,
                max_test_examples=args.max_test_examples,
                seed=args.seed,
                task=args.task_direction,
                dataset_format=args.dataset_format,
                native_direction=args.native_direction,
                return_micro=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
                bf16=args.bf16,
                torch_compile=args.torch_compile,
            )
            
            tabert_results = {
                "macro": macro,
                "micro": micro
            }
            with open(tabert_default_path, "w", encoding="utf-8") as f:
                json.dump(tabert_results, f, indent=2)

        print_results_table(tabert_results["macro"], "TaBERT (Macro)")
        print_results_table_micro(tabert_results["micro"], "TaBERT (Micro)")

    # --- Comparison ---
    if cmdl_results or loki_results or tabstar_results or tabert_results:
        models_macro = {}
        models_micro = {}
        
        if cmdl_results:
            models_macro["CMDL"] = cmdl_results.get("macro")
            models_micro["CMDL"] = cmdl_results.get("micro")
            
        if tabstar_results:
            models_macro["TabSTAR"] = tabstar_results.get("macro")
            models_micro["TabSTAR"] = tabstar_results.get("micro")
            
        if tabert_results:
            models_macro["TaBERT"] = tabert_results.get("macro")
            models_micro["TaBERT"] = tabert_results.get("micro")
            
        if loki_results:
            models_macro["LOKI"] = loki_results.get("macro")
            models_micro["LOKI"] = loki_results.get("micro")
            
        print_comparison(models_macro, k_values, baseline_model_key="LOKI")
        print_comparison_micro(models_micro, k_values, baseline_model_key="LOKI")

        # Generate comparison plots
        generate_plots(models_macro, models_micro, k_values, args.output_dir, baseline_model_key="LOKI", suffix=suffix)

        # Save combined results JSON
        combined = {"CMDL": cmdl_results, "LOKI": loki_results, "TabSTAR": tabstar_results, "TaBERT": tabert_results}
        combined_path = os.path.join(args.output_dir, "combined_pharma%s_results.json" % suffix)
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print("[RUN] Combined comparison results saved to %s" % combined_path)
    else:
        print("[WARN] Comparison skipped — no results found.")

    # --- Excel export ---
    all_results = {}
    if cmdl_results:
        all_results["CMDL (macro)"] = cmdl_results["macro"]
        all_results["CMDL (micro)"] = cmdl_results["micro"]
    if loki_results:
        all_results["LOKI (macro)"] = loki_results["macro"]
        all_results["LOKI (micro)"] = loki_results["micro"]
    if tabstar_results:
        all_results["TabSTAR (macro)"] = tabstar_results["macro"]
        all_results["TabSTAR (micro)"] = tabstar_results["micro"]
    if tabert_results:
        all_results["TaBERT (macro)"] = tabert_results["macro"]
        all_results["TaBERT (micro)"] = tabert_results["micro"]

    if all_results:
        try:
            from export_excel import export_all_excel
            export_all_excel(all_results, k_values, args.output_dir)
        except Exception as e:
            print("[WARN] Excel export failed: %s" % e)

    print("\nDone!")

if __name__ == "__main__":
    main()
