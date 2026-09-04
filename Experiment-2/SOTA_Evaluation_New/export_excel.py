"""
export_excel.py — Export SOTA Evaluation results to Excel workbooks.

Generates publication-ready Excel files similar to the post-training
comparison format:
  1. Result_Ranking.xlsx   — P@K, R@K, F1@K, NDCG@K, MRR@K, MAP, Score AP
  2. Result_Summary.xlsx   — Condensed paper-style table with all metrics
  3. Result_Details.xlsx    — Full comparison with metric notes

Usage:
  Called automatically by run_comparison.py, or standalone:
    python export_excel.py --results_dir results
"""

import os
import json
import argparse
from typing import Dict, List, Any

import pandas as pd


# ===========================================================================
# Helpers
# ===========================================================================

def _write_workbook(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    """Write multiple DataFrames to an Excel workbook with one sheet each."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_sheet = sheet_name[:31]  # Excel 31-char limit
            df.to_excel(writer, sheet_name=safe_sheet, index=False)


def _get_k_metric(results: Dict, k: int, metric: str) -> float:
    """Safely extract a per-K metric from results dict (handles int/str keys)."""
    per_k = results.get("per_k", {})
    k_data = per_k.get(k, per_k.get(str(k), {}))
    return k_data.get(metric, 0.0)


# ===========================================================================
# Table builders
# ===========================================================================

def _build_ranking_table(
    all_results: Dict[str, Dict[str, Any]],
    k_values: List[int],
) -> pd.DataFrame:
    """Build per-K ranking metrics table (one row per model × K)."""
    rows = []
    for model_name, results in all_results.items():
        for k in k_values:
            row = {
                "Model": model_name,
                "K": k,
                "P@K": _get_k_metric(results, k, "P@K"),
                "R@K": _get_k_metric(results, k, "R@K"),
                "F1@K": _get_k_metric(results, k, "F1@K"),
                "NDCG@K": _get_k_metric(results, k, "NDCG@K"),
                "MRR@K": _get_k_metric(results, k, "MRR@K"),
            }
            row["All@K"] = _get_k_metric(results, k, "All@K")
            rows.append(row)
    return pd.DataFrame(rows)


def _build_summary_table(
    all_results: Dict[str, Dict[str, Any]],
    k_values: List[int],
) -> pd.DataFrame:
    """Build condensed paper-style result table (one row per model)."""
    rows = []
    for model_name, results in all_results.items():
        row = {"Model": model_name}

        # Per-K metrics (flattened)
        for k in k_values:
            row[f"P@{k}"] = _get_k_metric(results, k, "P@K")
            row[f"R@{k}"] = _get_k_metric(results, k, "R@K")
            row[f"F1@{k}"] = _get_k_metric(results, k, "F1@K")
            row[f"NDCG@{k}"] = _get_k_metric(results, k, "NDCG@K")
            row[f"MRR@{k}"] = _get_k_metric(results, k, "MRR@K")
            row[f"All@{k}"] = _get_k_metric(results, k, "All@K")

        # Overall metrics
        row["MAP (rank-based)"] = results.get("MAP", 0.0)
        row["Score AP (PR-curve)"] = results.get("Score_AP") or 0.0
        row["Mean Rank"] = results.get("Mean_Rank", float("inf"))
        row["Num Queries"] = results.get("num_queries", 0)
        row["Num Examples"] = results.get("num_examples", 0)
        row["Max Test Examples"] = results.get("max_test_examples", 0)

        rows.append(row)
    return pd.DataFrame(rows)


def _build_comparison_delta_table(
    all_results: Dict[str, Dict[str, Any]],
    k_values: List[int],
) -> pd.DataFrame:
    """Build a table showing deltas between models (if exactly 2)."""
    model_names = list(all_results.keys())
    if len(model_names) != 2:
        return pd.DataFrame()

    name_a, name_b = model_names
    results_a, results_b = all_results[name_a], all_results[name_b]

    rows = []
    # Per-K
    for metric_name in ["P@K", "R@K", "F1@K", "NDCG@K", "MRR@K", "All@K"]:
        ks = k_values
        for k in ks:
            val_a = _get_k_metric(results_a, k, metric_name)
            val_b = _get_k_metric(results_b, k, metric_name)
            delta = val_b - val_a
            rows.append({
                "Metric": metric_name,
                "K": k,
                name_a: val_a,
                name_b: val_b,
                "Δ (%s - %s)" % (name_b, name_a): delta,
                "% Change": (delta / val_a * 100) if val_a != 0 else 0.0,
                "Winner": name_b if delta > 0 else name_a if delta < 0 else "Tie",
            })

    # Overall
    for metric, higher_is_better in [("MAP", True), ("Score_AP", True), ("Mean_Rank", False)]:
        val_a = results_a.get(metric) or 0
        val_b = results_b.get(metric) or 0
        delta = val_b - val_a
        if higher_is_better:
            winner = name_b if delta > 0 else name_a if delta < 0 else "Tie"
        else:
            winner = name_b if delta < 0 else name_a if delta > 0 else "Tie"
        rows.append({
            "Metric": metric,
            "K": "—",
            name_a: val_a,
            name_b: val_b,
            "Δ (%s - %s)" % (name_b, name_a): delta,
            "% Change": (delta / val_a * 100) if val_a != 0 else 0.0,
            "Winner": winner,
        })

    return pd.DataFrame(rows)


def _build_metric_notes_table() -> pd.DataFrame:
    """Build a reference table defining each metric."""
    notes = [
        {
            "Metric": "P@K (Precision@K)",
            "Definition": "Fraction of top-K retrieved tables that are relevant.",
            "Range": "[0, 1] — higher is better",
        },
        {
            "Metric": "R@K (Recall@K)",
            "Definition": "Fraction of relevant tables found in top-K.",
            "Range": "[0, 1] — higher is better",
        },
        {
            "Metric": "F1@K",
            "Definition": "Harmonic mean of P@K and R@K.",
            "Range": "[0, 1] — higher is better",
        },
        {
            "Metric": "NDCG@K",
            "Definition": "Normalized Discounted Cumulative Gain at K.",
            "Range": "[0, 1] — higher is better",
        },
        {
            "Metric": "MRR@K",
            "Definition": "Reciprocal rank of the first relevant table in top-K.",
            "Range": "[0, 1] — higher is better",
        },
        {
            "Metric": "All@K (Complete Retrieval Hit Rate)",
            "Definition": "Fraction of queries where ALL ground truth tables appear in top-K. "
                          "Measures complete retrieval success. Naturally 0 when K < |GT|.",
            "Range": "[0, 1] — higher is better",
        },
        {
            "Metric": "MAP (rank-based)",
            "Definition": "Mean Average Precision: mean of per-query AP computed from "
                          "ranked lists. AP = (1/|rel|) * Σ P(k) * rel(k).",
            "Range": "[0, 1] — higher is better",
        },
        {
            "Metric": "Score AP (PR-curve)",
            "Definition": "Average Precision from continuous similarity scores using "
                          "sklearn.metrics.average_precision_score (area under PR curve). "
                          "Matches the AP used in post-training evals.",
            "Range": "[0, 1] — higher is better",
        },
        {
            "Metric": "Mean Rank",
            "Definition": "Average rank position of relevant tables (1-indexed).",
            "Range": "[1, ∞) — lower is better",
        },
    ]
    return pd.DataFrame(notes)


# ===========================================================================
# Main export function
# ===========================================================================

def export_all_excel(
    all_results: Dict[str, Dict[str, Any]],
    k_values: List[int],
    output_dir: str,
) -> None:
    """
    Export all results to Excel workbooks.

    Args:
        all_results: {model_name: results_dict}
        k_values: list of K values used in evaluation
        output_dir: directory to save Excel files
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Result_Ranking.xlsx — per-K metrics in long format
    ranking_df = _build_ranking_table(all_results, k_values)
    ranking_path = os.path.join(output_dir, "Result_Ranking.xlsx")
    _write_workbook(ranking_path, {
        "ranking_metrics": ranking_df,
        "metric_notes": _build_metric_notes_table(),
    })
    print("[EXCEL] Saved: %s" % ranking_path)

    # 2. Result_Summary.xlsx — paper-condensed one-row-per-model
    summary_df = _build_summary_table(all_results, k_values)
    summary_path = os.path.join(output_dir, "Result_Summary.xlsx")
    _write_workbook(summary_path, {
        "result_summary": summary_df,
        "metric_notes": _build_metric_notes_table(),
    })
    print("[EXCEL] Saved: %s" % summary_path)

    # 3. Result_Comparison.xlsx — deltas between models (if 2 models)
    delta_df = _build_comparison_delta_table(all_results, k_values)
    if not delta_df.empty:
        comparison_path = os.path.join(output_dir, "Result_Comparison.xlsx")
        _write_workbook(comparison_path, {
            "comparison_deltas": delta_df,
            "result_summary": summary_df,
            "metric_notes": _build_metric_notes_table(),
        })
        print("[EXCEL] Saved: %s" % comparison_path)


# ===========================================================================
# CLI (standalone usage)
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export SOTA evaluation results to Excel workbooks."
    )
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing *_results.json files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for Excel files (default: same as results_dir)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_dir

    # Load all *_results.json files
    import glob
    json_files = glob.glob(os.path.join(args.results_dir, "*_results.json"))
    if not json_files:
        print("[ERROR] No *_results.json files found in %s" % args.results_dir)
        return

    all_results = {}
    k_values_from_files = None
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Derive model name from filename: CMDL_results.json -> CMDL
        basename = os.path.basename(jf)
        model_name = basename.replace("_results.json", "")
        all_results[model_name] = data

        # Extract K values from first file
        if k_values_from_files is None and "per_k" in data:
            k_values_from_files = sorted([int(k) for k in data["per_k"].keys()])

    if k_values_from_files is None:
        from config import K_VALUES
        k_values_from_files = K_VALUES

    export_all_excel(all_results, k_values_from_files, out_dir)
    print("\n[EXCEL] All Excel files exported to: %s" % out_dir)


if __name__ == "__main__":
    main()
